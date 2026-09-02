"""Turning a localization table into a super-resolved image or movie.

Pure numerics: no Qt, no napari, no pandas. The widget converts its dataframe
into plain arrays (positions in camera pixels, sigmas in camera pixels, optional
weights) and everything here works on those, so the renderer can be tested and
benchmarked without a GUI.

Three backends produce identical results and are chosen automatically:

  * CuPy, when a CUDA device is present. The whole render happens on the GPU;
    only the finished frame comes back.
  * numba, otherwise. The splat kernel is parallel over horizontal bands of the
    output image, which is what makes it safe to run multi-threaded: two threads
    never touch the same output row, so no atomics and no races.
  * plain numpy, if numba is unavailable. Correct but roughly an order of
    magnitude slower; numba is in install_requires, so this is a safety net.

Rendering is expressed as generators that yield progress in [0, 1] and return
the finished array. That is what lets the caller drive it from a background
thread, report a real percentage, and stop it part way (`gen.close()`) instead
of blocking the GUI for the minute a large reconstruction takes.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import numba

    if getattr(numba.config, "DISABLE_JIT", False):
        # The splat kernel is a per-pixel loop: interpreted, it is thousands of
        # times slower than the vectorized path, which would look like a hang
        # rather than the debugging aid NUMBA_DISABLE_JIT is meant to be.
        numba = None
except ImportError:  # pragma: no cover - exercised only without numba
    numba = None


# Render modes. The keys are what gets stored in metadata.json and restored from
# it, so they are stable identifiers - the UI shows the second element.
MODES = {
    "histogram": "Histogram (localization counts)",
    "scatter": "Scatter (one dot per localization)",
    "gaussian_global": "Gaussian, one width for all (user sigma)",
    "gaussian_local": "Gaussian, per-localization width (fitted precision)",
}

# How raw camera frames are grouped into one super-resolved movie frame.
GROUPINGS = {
    "blocks": "Independent blocks",
    "cumulative": "Cumulative build-up",
    "sliding": "Sliding window",
}

# Gaussians are cut off at this many sigma. Beyond 3 sigma a normalized Gaussian
# has < 0.3% of its mass left, which is far below the shot noise on any real
# localization count, and the cost of the splat grows with the square of it.
TRUNCATE = 3.0
# A Gaussian narrower than this (in super-resolved pixels) is indistinguishable
# from a single lit pixel but makes the render look like it has more precision
# than it does; below it the splat is clamped.
MIN_SIGMA_SR = 0.5
# Hard cap on the splat radius in super-resolved pixels. A single localization
# painting a 129x129 box already costs 16k writes; without a cap, one row with a
# nonsense uncertainty could stall the whole render.
MAX_RADIUS = 64
# Localizations per chunk on the vectorized (GPU / numpy) path, expressed as
# output elements so the temporary (n, K, K) array stays bounded whatever the
# kernel size.
CHUNK_ELEMENTS = 8_000_000
# Progress is reported at least this often while a single frame is rendered.
FRAME_PROGRESS_CHUNKS = 8

_GPU = None  # (cupy, cupyx.scipy.ndimage, reason) once probed


def _gpu():
    """Probe for a usable CuPy once and cache the result."""
    global _GPU
    if _GPU is None:
        try:
            import cupy
            from cupyx.scipy import ndimage as cupy_ndimage

            if cupy.cuda.runtime.getDeviceCount() < 1:
                _GPU = (None, None, "no CUDA device found")
            else:
                # Importing CuPy succeeds even when the driver is unusable; only
                # touching the device tells us whether it really works.
                cupy.zeros(1, dtype=cupy.float32) + 1
                _GPU = (cupy, cupy_ndimage, "")
        except Exception as exc:
            _GPU = (None, None, f"{type(exc).__name__}: {exc}")
    return _GPU


def is_render_gpu_available() -> bool:
    return _gpu()[0] is not None


def render_gpu_status() -> str:
    cupy, _ndimage, reason = _gpu()
    if cupy is not None:
        try:
            name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        except Exception:
            name = "CUDA device"
        return f"GPU rendering available ({name})"
    return f"GPU rendering unavailable ({reason}); using the CPU"


def is_numba_available() -> bool:
    return numba is not None


# A render is built on the device as one dense float32 array. Asking for more
# than this share of the *free* device memory is how a reconstruction turns into
# an out-of-memory error halfway through, so an oversized frame goes to the CPU
# instead - which is slower but has a 6 GB card's worth of headroom to spare.
GPU_FRAME_BUDGET = 0.4


def choose_backend(shape, oversampling, prefer_gpu=True):
    """Decide where a render of this size should run.

    Returns (use_gpu, explanation). The explanation is meant to be logged: a
    silent fallback to the CPU on a machine with a GPU looks like a bug.
    """
    rows, cols = output_shape(shape, oversampling)
    if not prefer_gpu:
        return False, "CPU rendering (GPU not requested)"
    cupy, _ndimage, reason = _gpu()
    if cupy is None:
        return False, f"CPU rendering ({reason})"
    needed = rows * cols * 4
    try:
        free, _total = cupy.cuda.runtime.memGetInfo()
    except Exception:
        return True, "GPU rendering"
    if needed > GPU_FRAME_BUDGET * free:
        return False, (
            f"CPU rendering: a {rows}x{cols} frame needs {needed / 1e9:.1f} GB, "
            f"too much of the {free / 1e9:.1f} GB free on the GPU"
        )
    return True, "GPU rendering"


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
def output_shape(shape, oversampling):
    """Super-resolved (rows, cols) for a camera field of view of `shape`."""
    height, width = int(shape[0]), int(shape[1])
    over = int(oversampling)
    return height * over, width * over


def estimate_bytes(shape, oversampling, n_frames=1):
    rows, cols = output_shape(shape, oversampling)
    return int(rows) * int(cols) * int(max(n_frames, 1)) * 4  # float32


def layer_transform(oversampling, origin=(-0.5, -0.5), source_scale=(1.0, 1.0),
                    source_translate=(0.0, 0.0)):
    """(scale, translate) placing a render on top of its source image in napari.

    A localization at camera-pixel coordinate x lands in super-resolved bin
    (x - x0) * oversampling, whose *centre* is therefore at camera coordinate
    x0 + (bin + 0.5) / oversampling. napari puts pixel `bin` of the render at
    translate + bin * scale, so the half-bin has to go into the translation -
    without it every render sits half a super-resolved pixel off its source,
    which is exactly the kind of shift that turns into a "drift" artefact.
    """
    over = float(oversampling)
    scale = tuple(float(s) / over for s in source_scale)
    translate = tuple(
        float(t) + float(s) * (float(o) + 0.5 / over)
        for t, s, o in zip(source_translate, source_scale, origin)
    )
    return scale, translate


def _group_layout(frame_min, frame_max, frames_per_group, grouping, step):
    if grouping not in GROUPINGS:
        raise ValueError(f"unknown grouping {grouping!r}")
    first, last = int(frame_min), int(frame_max)
    span = last - first + 1
    size = max(1, int(frames_per_group))
    stride = max(1, int(step or max(1, size // 2))) if grouping == "sliding" else size
    count = 0 if span <= 0 else int(math.ceil(span / stride))
    return first, last, size, stride, count


def group_count(frame_min, frame_max, frames_per_group, grouping="blocks", step=None):
    """How many super-resolved frames a movie would have.

    The panel updates this on every keystroke to show the size of what is about
    to be rendered, and a one-raw-frame-per-group movie of a long acquisition
    has tens of thousands of them - so the count is arithmetic, not the length
    of a list that gets built and thrown away.
    """
    return _group_layout(frame_min, frame_max, frames_per_group, grouping, step)[4]


def group_bounds(frame_min, frame_max, frames_per_group, grouping="blocks", step=None):
    """Half-open [lo, hi) raw-frame ranges, one per super-resolved movie frame."""
    first, last, size, stride, count = _group_layout(
        frame_min, frame_max, frames_per_group, grouping, step)
    if grouping == "cumulative":
        return [(first, min(first + (i + 1) * size, last + 1)) for i in range(count)]
    return [(first + i * stride, min(first + i * stride + size, last + 1))
            for i in range(count)]


# ----------------------------------------------------------------------
# Preparing localizations for a splat
# ----------------------------------------------------------------------
def _prepare(x_px, y_px, shape, oversampling, origin, sigma_px, weights, mode):
    """Bin coordinates, sub-pixel offsets, radii and amplitudes, in super-res px.

    Returns (iy, ix, fy, fx, radius, inv_two_sigma2, amp) with only the
    localizations that can contribute to the output kept.
    """
    over = float(oversampling)
    rows, cols = output_shape(shape, oversampling)

    sx = (np.asarray(x_px, dtype=np.float64) - float(origin[1])) * over
    sy = (np.asarray(y_px, dtype=np.float64) - float(origin[0])) * over

    if weights is None:
        amp = np.ones(sx.shape, dtype=np.float64)
    else:
        amp = np.asarray(weights, dtype=np.float64)
        if amp.shape != sx.shape:
            raise ValueError("weights must have one entry per localization")

    if mode == "gaussian_local":
        if sigma_px is None:
            raise ValueError("gaussian_local needs a per-localization sigma")
        sigma = np.asarray(sigma_px, dtype=np.float64) * over
        if sigma.shape != sx.shape:
            raise ValueError("sigma_px must have one entry per localization")
        # A localization with no recorded precision still exists; drawing it at
        # the smallest width keeps it in the picture instead of quietly
        # discarding it (np.clip would leave the NaN in place).
        sigma = np.where(np.isfinite(sigma), sigma, MIN_SIGMA_SR)
        sigma = np.clip(sigma, MIN_SIGMA_SR, MAX_RADIUS / TRUNCATE)
        radius = np.ceil(TRUNCATE * sigma).astype(np.int32)
        np.clip(radius, 1, MAX_RADIUS, out=radius)
        inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
    else:
        # Histogram, scatter and the global-sigma mode (blurred afterwards) all
        # deposit each localization into a single bin.
        radius = np.zeros(sx.shape, dtype=np.int32)
        inv_two_sigma2 = np.zeros(sx.shape, dtype=np.float64)

    finite = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(amp)
    if not finite.all():
        sx, sy, amp = sx[finite], sy[finite], amp[finite]
        radius, inv_two_sigma2 = radius[finite], inv_two_sigma2[finite]

    ix = np.floor(sx).astype(np.int64)
    iy = np.floor(sy).astype(np.int64)
    # Offset of the localization from the centre of the bin it fell in; this is
    # what keeps the Gaussian sub-pixel accurate instead of snapping to the grid.
    fx = sx - (ix + 0.5)
    fy = sy - (iy + 0.5)

    # Drop anything whose entire footprint is outside the field of view. Spots
    # only partly inside are kept: their visible part must still be drawn.
    inside = (
        (ix + radius >= 0) & (ix - radius < cols)
        & (iy + radius >= 0) & (iy - radius < rows)
    )
    if not inside.all():
        ix, iy, fx, fy = ix[inside], iy[inside], fx[inside], fy[inside]
        radius, inv_two_sigma2, amp = radius[inside], inv_two_sigma2[inside], amp[inside]

    return iy, ix, fy, fx, radius, inv_two_sigma2, amp


# ----------------------------------------------------------------------
# numba splat: parallel over bands of output rows
# ----------------------------------------------------------------------
if numba is not None:

    @numba.njit(cache=True, nogil=True, parallel=True)
    def _splat_nb(out, iy, ix, fy, fx, radius, inv_two_sigma2, amp,
                  band_row0, band_row1, band_lo, band_hi):
        rows, cols = out.shape
        n_bands = band_row0.shape[0]
        for b in numba.prange(n_bands):
            row0 = band_row0[b]
            row1 = band_row1[b]
            # Per-thread scratch for the separable kernel; MAX_RADIUS is a
            # compile-time constant so this is a fixed, small allocation.
            gx = np.empty(2 * MAX_RADIUS + 1, dtype=np.float64)
            gy = np.empty(2 * MAX_RADIUS + 1, dtype=np.float64)
            for k in range(band_lo[b], band_hi[b]):
                r = radius[k]
                inv = inv_two_sigma2[k]
                width = 2 * r + 1

                sum_x = 0.0
                sum_y = 0.0
                for d in range(width):
                    offset_x = (d - r) - fx[k]
                    offset_y = (d - r) - fy[k]
                    vx = math.exp(-offset_x * offset_x * inv)
                    vy = math.exp(-offset_y * offset_y * inv)
                    gx[d] = vx
                    gy[d] = vy
                    sum_x += vx
                    sum_y += vy
                if sum_x <= 0.0 or sum_y <= 0.0:
                    continue
                # Normalizing by the truncated sums makes the kernel integrate
                # to exactly amp[k], so a render's total is the localization
                # count (or photon sum) regardless of sigma or sub-pixel phase.
                scale = amp[k] / (sum_x * sum_y)

                for dy in range(width):
                    yy = iy[k] + dy - r
                    if yy < row0 or yy >= row1:
                        continue
                    row_value = gy[dy] * scale
                    for dx in range(width):
                        xx = ix[k] + dx - r
                        if xx < 0 or xx >= cols:
                            continue
                        out[yy, xx] += np.float32(row_value * gx[dx])


def _bands(rows, n_threads):
    """Split output rows into `n_threads` contiguous bands."""
    n_bands = max(1, min(int(n_threads), rows))
    edges = np.linspace(0, rows, n_bands + 1).astype(np.int64)
    return edges[:-1], edges[1:]


def _splat_numba(out, iy, ix, fy, fx, radius, inv_two_sigma2, amp):
    rows = out.shape[0]
    try:
        n_threads = numba.get_num_threads()
    except Exception:
        n_threads = 4
    band_row0, band_row1 = _bands(rows, n_threads)

    # A band only has to look at localizations that can reach into it. Sorting
    # by row once makes that a pair of searchsorted lookups per band instead of
    # a scan of every localization by every thread.
    order = np.argsort(iy, kind="stable")
    iy, ix = iy[order], ix[order]
    fy, fx = fy[order], fx[order]
    radius, inv_two_sigma2, amp = radius[order], inv_two_sigma2[order], amp[order]
    reach = int(radius.max()) if radius.size else 0
    band_lo = np.searchsorted(iy, band_row0 - reach, side="left")
    band_hi = np.searchsorted(iy, band_row1 + reach, side="right")

    _splat_nb(
        out, iy, ix, fy, fx, radius, inv_two_sigma2.astype(np.float64), amp,
        band_row0, band_row1, band_lo.astype(np.int64), band_hi.astype(np.int64),
    )


# ----------------------------------------------------------------------
# Vectorized splat, shared by the GPU (CuPy) and the no-numba fallback
# ----------------------------------------------------------------------
def _scatter_add(xp, flat_out, indices, values):
    # On the GPU this is an atomic add, which is why the vectorized splat can
    # let overlapping kernels land in the same pixel without a race.
    xp.add.at(flat_out, indices, values)


def _splat_vectorized(xp, out, iy, ix, fy, fx, radius, inv_two_sigma2, amp):
    """Splat by building the (n, K, K) kernel stack, one radius bucket at a time.

    Every localization in a bucket shares a kernel size, so the whole bucket is
    one broadcast plus one scatter-add - which is what a GPU wants. Buckets are
    chunked so the temporary never exceeds CHUNK_ELEMENTS.
    """
    rows, cols = out.shape
    flat = out.reshape(-1)
    radii = np.unique(np.asarray(radius))
    for r in radii:
        r = int(r)
        select = np.flatnonzero(radius == r)
        if select.size == 0:
            continue
        width = 2 * r + 1
        offsets = xp.arange(-r, r + 1, dtype=xp.float64)
        per_chunk = max(1, int(CHUNK_ELEMENTS // (width * width)))
        for start in range(0, select.size, per_chunk):
            part = select[start:start + per_chunk]
            cy = xp.asarray(iy[part])
            cx = xp.asarray(ix[part])
            gy = xp.exp(-((offsets[None, :] - xp.asarray(fy[part])[:, None]) ** 2)
                        * xp.asarray(inv_two_sigma2[part])[:, None])
            gx = xp.exp(-((offsets[None, :] - xp.asarray(fx[part])[:, None]) ** 2)
                        * xp.asarray(inv_two_sigma2[part])[:, None])
            norm = gy.sum(axis=1) * gx.sum(axis=1)
            scale = xp.asarray(amp[part]) / xp.where(norm > 0, norm, 1.0)
            values = (gy[:, :, None] * gx[:, None, :]) * scale[:, None, None]

            yy = cy[:, None] + xp.arange(-r, r + 1, dtype=cy.dtype)[None, :]
            xx = cx[:, None] + xp.arange(-r, r + 1, dtype=cx.dtype)[None, :]
            valid = ((yy >= 0) & (yy < rows))[:, :, None] & ((xx >= 0) & (xx < cols))[:, None, :]
            indices = (xp.clip(yy, 0, rows - 1)[:, :, None] * cols
                       + xp.clip(xx, 0, cols - 1)[:, None, :])
            _scatter_add(
                xp, flat,
                indices.reshape(-1),
                xp.where(valid, values, 0.0).astype(xp.float32).reshape(-1),
            )


# ----------------------------------------------------------------------
# One frame
# ----------------------------------------------------------------------
def _blur_sigma_sr(global_sigma_px, oversampling):
    sigma = float(global_sigma_px) * float(oversampling)
    return max(sigma, MIN_SIGMA_SR)


def render_frame_iter(x_px, y_px, *, shape, oversampling, mode="histogram",
                      origin=(-0.5, -0.5), sigma_px=None, global_sigma_px=None,
                      weights=None, gpu=None):
    """Generator: yields progress in [0, 1], returns the rendered float32 frame."""
    if mode not in MODES:
        raise ValueError(f"unknown render mode {mode!r}")
    rows, cols = output_shape(shape, oversampling)
    use_gpu = is_render_gpu_available() if gpu is None else bool(gpu)
    cupy, cupy_ndimage, _reason = _gpu()
    if use_gpu and cupy is None:
        use_gpu = False

    iy, ix, fy, fx, radius, inv_two_sigma2, amp = _prepare(
        x_px, y_px, shape, oversampling, origin, sigma_px, weights, mode
    )
    n = iy.size

    xp = cupy if use_gpu else np
    canvas = xp.zeros((rows, cols), dtype=xp.float32)

    if n:
        # Chunking exists purely so a long render reports progress and can be
        # interrupted; the splat is additive, so chunk boundaries are invisible
        # in the result.
        chunk = max(1, int(math.ceil(n / FRAME_PROGRESS_CHUNKS)))
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            piece = slice(start, stop)
            if use_gpu:
                _splat_vectorized(
                    cupy, canvas, iy[piece], ix[piece], fy[piece], fx[piece],
                    radius[piece], inv_two_sigma2[piece], amp[piece],
                )
            elif numba is not None:
                _splat_numba(
                    canvas, iy[piece], ix[piece], fy[piece], fx[piece],
                    radius[piece], inv_two_sigma2[piece], amp[piece],
                )
            else:  # pragma: no cover - numba is a hard dependency
                _splat_vectorized(
                    np, canvas, iy[piece], ix[piece], fy[piece], fx[piece],
                    radius[piece], inv_two_sigma2[piece], amp[piece],
                )
            yield (stop / n) * (0.9 if mode == "gaussian_global" else 1.0)

    if mode == "gaussian_global":
        sigma = _blur_sigma_sr(global_sigma_px or 0.0, oversampling)
        if use_gpu:
            canvas = cupy_ndimage.gaussian_filter(canvas, sigma=sigma, truncate=TRUNCATE)
        else:
            from scipy.ndimage import gaussian_filter

            canvas = gaussian_filter(canvas, sigma=sigma, truncate=TRUNCATE, output=np.float32)
        yield 1.0

    result = cupy.asnumpy(canvas) if use_gpu else canvas
    if mode == "scatter":
        result = (result > 0).astype(np.float32)
    return np.ascontiguousarray(result, dtype=np.float32)


def free_gpu_memory():
    """Hand the device memory back, once a whole render is done.

    Not per frame: CuPy's pool is what makes the next frame of a movie cheap to
    allocate, so releasing it between frames would trade a fast render for a
    slow one. The caller frees when the movie is finished.
    """
    cupy, _ndimage, _reason = _gpu()
    if cupy is not None:
        try:
            cupy.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def render_frame(*args, **kwargs):
    """Non-generator convenience wrapper around `render_frame_iter`."""
    gen = render_frame_iter(*args, **kwargs)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value


# ----------------------------------------------------------------------
# Movie
# ----------------------------------------------------------------------
def render_movie_iter(x_px, y_px, frames, *, shape, oversampling, frames_per_group,
                      grouping="blocks", step=None, mode="histogram",
                      origin=(-0.5, -0.5), sigma_px=None, global_sigma_px=None,
                      weights=None, gpu=None, frame_range=None):
    """Generator: yields progress in [0, 1], returns a (T, rows, cols) float32 stack.

    Localizations are sorted by frame once, so selecting a group is two
    searchsorted lookups rather than a pass over the table per movie frame.

    `frame_range` fixes the acquisition the movie covers, instead of taking it
    from these particular points. An overlay - trajectories, say - usually spans
    fewer frames than the reconstruction it is drawn on, and without this it
    would be cut into its own, shorter set of movie frames and stop lining up
    with the movie it is blended into.
    """
    frames = np.asarray(frames)
    rows, cols = output_shape(shape, oversampling)
    if frames.size == 0:
        if frame_range is None:
            return np.zeros((0, rows, cols), dtype=np.float32)
        n_empty = group_count(frame_range[0], frame_range[1], frames_per_group, grouping, step)
        return np.zeros((n_empty, rows, cols), dtype=np.float32)

    order = np.argsort(frames, kind="stable")
    frames_sorted = frames[order]
    x_sorted = np.asarray(x_px)[order]
    y_sorted = np.asarray(y_px)[order]
    sigma_sorted = None if sigma_px is None else np.asarray(sigma_px)[order]
    weights_sorted = None if weights is None else np.asarray(weights)[order]

    # The cumulative grouping is the same block render accumulated over time, and
    # every mode except "scatter" is linear in the localizations - so render the
    # disjoint blocks once and run a cumulative sum, instead of re-rendering the
    # whole history for every movie frame (which would make the last frame of a
    # 100-group movie cost 100 renders). "scatter" is recovered at the end, since
    # "was there ever a localization here" is "did the running count leave zero".
    cumulative = grouping == "cumulative"
    frame_mode = "histogram" if (cumulative and mode == "scatter") else mode
    first, last = (frames_sorted[0], frames_sorted[-1]) if frame_range is None else frame_range
    bounds = group_bounds(
        first, last, frames_per_group, "blocks" if cumulative else grouping, step,
    )
    movie = np.zeros((len(bounds), rows, cols), dtype=np.float32)

    total = max(len(bounds), 1)
    for index, (lo, hi) in enumerate(bounds):
        start = int(np.searchsorted(frames_sorted, lo, side="left"))
        stop = int(np.searchsorted(frames_sorted, hi, side="left"))
        piece = slice(start, stop)
        gen = render_frame_iter(
            x_sorted[piece], y_sorted[piece],
            shape=shape, oversampling=oversampling, mode=frame_mode, origin=origin,
            sigma_px=None if sigma_sorted is None else sigma_sorted[piece],
            global_sigma_px=global_sigma_px,
            weights=None if weights_sorted is None else weights_sorted[piece],
            gpu=gpu,
        )
        while True:
            try:
                inner = next(gen)
            except StopIteration as stop_signal:
                movie[index] = stop_signal.value
                break
            yield (index + inner) / total
        yield (index + 1) / total

    if cumulative:
        np.cumsum(movie, axis=0, out=movie)
        if mode == "scatter":
            movie = (movie > 0).astype(np.float32)
    return movie


def render_movie(*args, **kwargs):
    gen = render_movie_iter(*args, **kwargs)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value


# ----------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------
# ImageJ's TIFF flavour cannot address more than 4 GB; past that the file is
# written as a plain BigTIFF instead (and loses the ImageJ pixel-size tags,
# which is why the sidecar JSON is not optional).
_IMAGEJ_LIMIT_BYTES = 4 * 1024 ** 3 - 64 * 1024 ** 2


def save_render(path, image, metadata, *, super_pixel_size_nm, png=False,
                colormap="magma", frame_interval_s=None):
    """Write a render as TIFF, always with its metadata beside it.

    What arrives decides what is written. A float32 array is stored exactly as
    it is - the render's own values (counts, or photons when weighted), never
    rescaled, so two exports stay quantitatively comparable. A uint8 array is
    stored as the display version it already is, greyscale or RGB; that is a
    quarter of the size (a twelfth, against RGB) and is what a movie meant for
    looking at should be.

    The metadata goes in twice on purpose: embedded in the TIFF description,
    where it travels with the file, and as a sidecar JSON that stays readable
    without opening a multi-gigabyte image.

    Returns the list of paths written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tifffile

    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
    image = np.ascontiguousarray(image)

    is_rgb = image.ndim in (3, 4) and image.shape[-1] == 3 and image.dtype == np.uint8
    is_movie = image.ndim == (4 if is_rgb else 3)
    axes = ("T" if is_movie else "") + "YX" + ("S" if is_rgb else "")
    pixel_size_um = float(super_pixel_size_nm) / 1000.0

    record = dict(metadata or {})
    record.setdefault("written_at", datetime.now().isoformat(timespec="seconds"))
    record["super_resolved_pixel_size_nm"] = float(super_pixel_size_nm)
    record["image_shape"] = [int(v) for v in image.shape]
    record["dtype"] = str(image.dtype)
    record["axes"] = axes
    if image.dtype == np.uint8:
        # An 8-bit export is a picture, not a measurement; say so rather than
        # letting a reader assume the numbers still mean localizations.
        record["value_units"] = "display levels (0-255), contrast-stretched"
    else:
        record.setdefault("value_units", "localizations per pixel")
    as_json = json.dumps(record, indent=2, default=str)

    written = []
    photometric = "rgb" if is_rgb else "minisblack"
    if image.nbytes < _IMAGEJ_LIMIT_BYTES:
        ij_metadata = {"axes": axes, "unit": "um", "Info": as_json}
        if is_movie and frame_interval_s:
            ij_metadata["finterval"] = float(frame_interval_s)
            ij_metadata["fps"] = 1.0 / float(frame_interval_s)
        tifffile.imwrite(
            path, image, imagej=True, metadata=ij_metadata, photometric=photometric,
            resolution=(1.0 / pixel_size_um, 1.0 / pixel_size_um),
        )
    else:
        tifffile.imwrite(
            path, image, bigtiff=True, photometric=photometric,
            description=as_json,
            resolution=(1.0 / pixel_size_um, 1.0 / pixel_size_um),
        )
    written.append(path)

    sidecar = path.with_name(f"{path.stem}_metadata.json")
    sidecar.write_text(as_json, encoding="utf-8")
    written.append(sidecar)

    if png:
        written.append(save_png_snapshot(path.with_suffix(".png"), image, colormap=colormap))
    return written


def contrast_limits(image, upper_percentile=99.9, max_samples=4_000_000):
    """Display limits that survive the long tail of a super-resolved render.

    Most pixels of a reconstruction are empty and a handful hold hundreds of
    overlapping localizations, so a min/max stretch shows a black image. The
    limits come from the *non-empty* pixels only, and from a strided sample of
    them on a large movie - a percentile over ten gigapixels would cost more
    than the render did.
    """
    values = np.asarray(image).reshape(-1)
    if values.size > max_samples:
        values = values[:: int(math.ceil(values.size / max_samples))]
    filled = values[values > 0]
    if filled.size == 0:
        return 0.0, 1.0
    high = float(np.percentile(filled, upper_percentile))
    if not np.isfinite(high) or high <= 0:
        high = float(filled.max()) or 1.0
    return 0.0, high


# Colours offered for the overlays in a composite. Named rather than free-form
# so they survive a metadata round-trip, and chosen to stay distinguishable on
# top of a warm reconstruction colormap.
OVERLAY_COLORS = {
    "cyan": (0.0, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "magenta": (1.0, 0.0, 1.0),
    "white": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "blue": (0.3, 0.5, 1.0),
    "orange": (1.0, 0.55, 0.0),
}


def to_uint8(image, limits=None):
    """Contrast-stretch to 8 bits, with ONE range for the whole stack.

    Per-frame normalization is the tempting shortcut and it is wrong for a
    movie: as the localization density changes from frame to frame, every frame
    gets rescaled to its own maximum and the reconstruction appears to pulse.
    A single range across the stack makes a frame's brightness mean what it
    should - how much signal that frame actually holds.
    """
    values = np.asarray(image, dtype=np.float32)
    low, high = contrast_limits(values) if limits is None else limits
    span = float(high) - float(low)
    if span <= 0:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - float(low)) * (255.0 / span)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _lookup_table(color=None, colormap=None):
    """A (256, 3) uint8 ramp from black to `color`, or a matplotlib colormap."""
    if colormap:
        from matplotlib import colormaps

        table = colormaps[colormap](np.linspace(0.0, 1.0, 256))[:, :3]
    else:
        rgb = OVERLAY_COLORS.get(color, color if isinstance(color, tuple) else (1.0, 1.0, 1.0))
        table = np.linspace(0.0, 1.0, 256)[:, None] * np.asarray(rgb, dtype=np.float64)[None, :]
    return np.clip(table * 255.0, 0, 255).astype(np.uint8)


def colorize(image, *, color=None, colormap=None, limits=None):
    """Map a render to 8-bit RGB through a 256-entry lookup table.

    Going via the table rather than calling the colormap on the array keeps a
    movie's memory flat: matplotlib would expand every pixel to float64 RGBA -
    32 bytes each, so a 200-frame 2048x2048 movie would need 6.7 GB just to be
    coloured. Indexing a uint8 table produces the final bytes directly.
    """
    indexed = to_uint8(image, limits)
    return _lookup_table(color=color, colormap=colormap)[indexed]


def blend_additive(layers):
    """Additively blend 8-bit RGB layers, the way napari's additive mode does.

    Additive rather than alpha-over so that a trajectory crossing a bright part
    of the reconstruction stays visible instead of punching a hole in it.
    """
    layers = [np.asarray(layer) for layer in layers if layer is not None]
    if not layers:
        raise ValueError("nothing to blend")
    total = np.zeros(layers[0].shape, dtype=np.float32)
    for layer in layers:
        total += layer
    return np.clip(total, 0, 255).astype(np.uint8)


def trajectory_samples(x_px, y_px, frames, particles, spacing_px):
    """Points spaced along each trajectory, so tracks can be splatted as lines.

    Rendering a line by sampling it turns trajectory drawing into the same
    scatter the reconstruction already does - it inherits the sub-pixel
    accuracy, the GPU path and the frame grouping for free, instead of needing
    a separate rasterizer that would have to agree with all three.

    Each sample carries the frame of the segment it came from, so a movie shows
    each trajectory while it is actually being tracked.
    """
    x_px = np.asarray(x_px, dtype=np.float64)
    y_px = np.asarray(y_px, dtype=np.float64)
    frames = np.asarray(frames)
    particles = np.asarray(particles)
    empty = (np.empty(0), np.empty(0), np.empty(0, dtype=frames.dtype))
    if x_px.size < 2:
        return empty

    order = np.lexsort((frames, particles))
    x, y, frame, particle = x_px[order], y_px[order], frames[order], particles[order]

    # Consecutive rows belonging to the same trajectory are its segments.
    within = particle[1:] == particle[:-1]
    if not within.any():
        return empty
    x0, y0 = x[:-1][within], y[:-1][within]
    x1, y1 = x[1:][within], y[1:][within]
    start_frame = frame[:-1][within]

    spacing = max(float(spacing_px), 1e-6)
    counts = np.maximum(1, np.ceil(np.hypot(x1 - x0, y1 - y0) / spacing)).astype(np.int64)
    total = int(counts.sum())
    segment = np.repeat(np.arange(counts.size), counts)
    starts = np.zeros(counts.size, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    step = (np.arange(total) - starts[segment]) / counts[segment]

    return (
        x0[segment] + step * (x1[segment] - x0[segment]),
        y0[segment] + step * (y1[segment] - y0[segment]),
        start_frame[segment],
    )


# ----------------------------------------------------------------------
# Burning a time stamp in
# ----------------------------------------------------------------------
# Everything a formatted time can contain. The atlas is built once from these
# and strings are composed by blitting glyphs, so a thousand-frame movie costs
# one rasterization pass rather than a thousand - and the worker thread never
# has to touch matplotlib, which is busy drawing the GUI's own figures.
STAMP_CHARACTERS = "0123456789.:-sminh µu"


def glyph_atlas(height_px, characters=STAMP_CHARACTERS):
    """Rasterize each character once, scaled so digits are `height_px` tall."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    height_px = max(int(height_px), 4)
    dpi = 100
    # Rasterize large, then scale down: one size of type for every requested
    # height keeps the glyphs consistent and the hinting predictable.
    points = 60
    atlas = {}
    for character in characters:
        figure = Figure(figsize=(2.0, 2.0), dpi=dpi)
        canvas = FigureCanvasAgg(figure)
        figure.patch.set_alpha(0.0)
        figure.text(0.1, 0.3, character, fontsize=points, color="white")
        canvas.draw()
        alpha = np.asarray(canvas.buffer_rgba())[:, :, 3].astype(np.float32) / 255.0
        atlas[character] = _crop_to_ink(alpha, character)

    reference = atlas.get("0")
    scale = height_px / max(reference.shape[0], 1) if reference is not None else 1.0
    return {char: _scale_mask(mask, scale) for char, mask in atlas.items()}


def _crop_to_ink(mask, character):
    """Trim to the drawn pixels, keeping a space as a fixed-width gap."""
    rows = np.flatnonzero(mask.max(axis=1) > 0.01)
    cols = np.flatnonzero(mask.max(axis=0) > 0.01)
    if rows.size == 0 or cols.size == 0:  # a space has no ink
        return np.zeros((mask.shape[0] // 8, max(mask.shape[1] // 40, 1)), np.float32)
    return mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _scale_mask(mask, scale):
    if mask.size == 0 or abs(scale - 1.0) < 1e-3:
        return mask
    rows = max(1, int(round(mask.shape[0] * scale)))
    cols = max(1, int(round(mask.shape[1] * scale)))
    row_index = np.clip((np.arange(rows) / scale).astype(int), 0, mask.shape[0] - 1)
    col_index = np.clip((np.arange(cols) / scale).astype(int), 0, mask.shape[1] - 1)
    return mask[row_index][:, col_index]


def compose_text(atlas, text):
    """Lay glyphs out into one mask. Pure numpy - safe on a worker thread."""
    glyphs = [atlas[c] for c in str(text) if c in atlas]
    if not glyphs:
        return np.zeros((1, 1), np.float32)
    height = max(g.shape[0] for g in glyphs)
    spacing = max(1, int(round(height * 0.12)))
    width = sum(g.shape[1] for g in glyphs) + spacing * (len(glyphs) - 1)
    canvas = np.zeros((height, width), np.float32)
    at = 0
    for glyph in glyphs:
        # sit every glyph on a common baseline, so digits don't jitter
        top = height - glyph.shape[0]
        canvas[top:top + glyph.shape[0], at:at + glyph.shape[1]] = np.maximum(
            canvas[top:top + glyph.shape[0], at:at + glyph.shape[1]], glyph)
        at += glyph.shape[1] + spacing
    return canvas


def format_time(seconds, longest=None):
    """A time label: seconds while that stays short, mm:ss once it doesn't."""
    span = float(longest if longest is not None else seconds)
    seconds = float(seconds)
    if span < 60.0:
        return f"{seconds:.1f} s"
    if span < 3600.0:
        return f"{int(seconds // 60):02d}:{int(round(seconds % 60)):02d}"
    hours = int(seconds // 3600)
    return f"{hours:02d}:{int((seconds % 3600) // 60):02d}:{int(round(seconds % 60)):02d}"


def annotation_origin(mask_shape, image_shape, position, margin=None):
    """(top, left) of a mask blitted into a corner, or None if it will not fit.

    Split out of `burn_text` so that a preview drawn somewhere other than into
    the pixels - the layer the Render tab puts in the viewer - lands where the
    burned-in annotation will, by running the same arithmetic rather than an
    imitation of it that can drift out of step with this one.
    """
    rows, cols = image_shape[:2]
    text_rows, text_cols = mask_shape[:2]
    if text_rows >= rows or text_cols >= cols:
        return None  # asked for a label bigger than the picture

    gap = int(round(0.02 * min(rows, cols))) if margin is None else int(margin)
    top = gap if "top" in position else rows - text_rows - gap
    left = gap if "left" in position else cols - text_cols - gap
    return (int(np.clip(top, 0, rows - text_rows)),
            int(np.clip(left, 0, cols - text_cols)))


def burn_text(image, mask, *, color="white", position="top left", margin=None):
    """Blend a text mask into a frame, in place.

    Additive like everything else here, and clipped: a label over a bright part
    of the reconstruction stays readable instead of blanking what is underneath.
    """
    if mask.size <= 1:
        return image
    corner = annotation_origin(mask.shape, image.shape, position, margin)
    if corner is None:
        return image
    top, left = corner
    text_rows, text_cols = mask.shape

    window = image[top:top + text_rows, left:left + text_cols]
    rgb = OVERLAY_COLORS.get(color, (1.0, 1.0, 1.0))
    if image.ndim == 3 and image.shape[-1] == 3:
        ink = mask[:, :, None] * (np.asarray(rgb, dtype=np.float32)[None, None, :] * 255.0)
    else:
        ink = mask * 255.0
    window[...] = np.clip(window.astype(np.float32) + ink, 0, 255).astype(np.uint8)
    return image


# ----------------------------------------------------------------------
# Scale bar
# ----------------------------------------------------------------------
# A scale bar is read at a glance, so its length has to be a number nobody has
# to decode: 1, 2 or 5 times a power of ten.
SCALE_BAR_STEPS = (1.0, 2.0, 5.0)
# Fraction of the field of view the default bar spans. Long enough to measure
# against, short enough not to become part of the picture.
SCALE_BAR_FRACTION = 0.15


def nice_scale_length(span_nm, fraction=SCALE_BAR_FRACTION):
    """A round bar length covering about `fraction` of a `span_nm` wide view."""
    target = max(float(span_nm) * float(fraction), 1e-9)
    decade = math.floor(math.log10(target))
    candidates = [step * (10.0 ** power)
                  for power in (decade - 1, decade, decade + 1)
                  for step in SCALE_BAR_STEPS]
    return float(min(candidates, key=lambda value: abs(math.log10(value / target))))


def format_length(nanometres):
    """Label a bar in the unit a reader expects at that size."""
    nanometres = float(nanometres)
    if nanometres >= 1000.0:
        micrometres = nanometres / 1000.0
        return f"{micrometres:.10g} µm"
    return f"{nanometres:.10g} nm"


def scale_bar_mask(length_px, thickness_px, label_mask=None):
    """The bar, with its label centred above it, as one mask to blit.

    Building it as a single mask means the bar and its label are placed and
    clipped together - they can never drift apart or land half off the frame
    independently of each other.
    """
    length_px = max(int(length_px), 1)
    thickness_px = max(int(thickness_px), 1)
    bar = np.ones((thickness_px, length_px), np.float32)
    if label_mask is None or label_mask.size <= 1:
        return bar

    gap = max(2, thickness_px)
    width = max(length_px, label_mask.shape[1])
    canvas = np.zeros((label_mask.shape[0] + gap + thickness_px, width), np.float32)
    label_left = (width - label_mask.shape[1]) // 2
    canvas[:label_mask.shape[0], label_left:label_left + label_mask.shape[1]] = label_mask
    bar_left = (width - length_px) // 2
    canvas[-thickness_px:, bar_left:bar_left + length_px] = 1.0
    return canvas


# ----------------------------------------------------------------------
# Bringing other layers onto the render grid, and cropping
# ----------------------------------------------------------------------
def resample_to_grid(source, *, shape, origin, oversampling, source_scale=(1.0, 1.0),
                     source_translate=(0.0, 0.0)):
    """Nearest-neighbour sample a 2D array onto the super-resolved grid.

    Nearest neighbour on purpose: the point of putting the camera image under a
    reconstruction is to see where the raw signal was, and interpolation would
    invent detail the camera never resolved at exactly the scale the
    reconstruction is claiming to reveal.
    """
    source = np.asarray(source)
    rows, cols = output_shape(shape, oversampling)
    over = float(oversampling)

    # centre of super-resolved pixel i, in camera pixels, then into the source
    row_world = float(origin[0]) + (np.arange(rows) + 0.5) / over
    col_world = float(origin[1]) + (np.arange(cols) + 0.5) / over
    row_index = (row_world - float(source_translate[0])) / float(source_scale[0])
    col_index = (col_world - float(source_translate[1])) / float(source_scale[1])
    row_index = np.clip(np.round(row_index).astype(np.int64), 0, source.shape[0] - 1)
    col_index = np.clip(np.round(col_index).astype(np.int64), 0, source.shape[1] - 1)
    return source[row_index][:, col_index].astype(np.float32, copy=False)


def box_to_slices(box, *, shape, origin, oversampling):
    """Turn a (y0, x0, y1, x1) box in camera pixels into super-resolved slices."""
    rows, cols = output_shape(shape, oversampling)
    over = float(oversampling)
    y0, x0, y1, x1 = (float(v) for v in box)
    row_start = int(np.floor((min(y0, y1) - float(origin[0])) * over))
    row_stop = int(np.ceil((max(y0, y1) - float(origin[0])) * over))
    col_start = int(np.floor((min(x0, x1) - float(origin[1])) * over))
    col_stop = int(np.ceil((max(x0, x1) - float(origin[1])) * over))
    row_start = int(np.clip(row_start, 0, rows - 1))
    col_start = int(np.clip(col_start, 0, cols - 1))
    row_stop = int(np.clip(row_stop, row_start + 1, rows))
    col_stop = int(np.clip(col_stop, col_start + 1, cols))
    return slice(row_start, row_stop), slice(col_start, col_stop)


def crop(image, rows, cols, is_movie=False):
    """Crop the two spatial axes, leaving any time or colour axis alone.

    `is_movie` is told rather than guessed: a three-axis array is a movie when
    it is greyscale and a single RGB frame when it is not, and getting that
    backwards silently crops the wrong axis.
    """
    image = np.asarray(image)
    if is_movie:
        return image[:, rows, cols]
    return image[rows, cols]


def save_png_snapshot(path, image, colormap="magma"):
    """Contrast-stretched, colour-mapped PNG - a preview, not the data.

    A movie is projected over time first: the useful single picture of a movie
    is the reconstruction it builds up to. An RGB composite is already coloured,
    so it is written as it is rather than pushed through a colormap.
    """
    from matplotlib import image as mpl_image

    values = np.asarray(image)
    path = Path(path)
    if values.dtype == np.uint8 and values.shape[-1] == 3:
        if values.ndim == 4:  # a composite movie: brightest moment of each pixel
            values = values.max(axis=0)
        mpl_image.imsave(str(path), values, origin="upper")
        return path

    values = values.astype(np.float32, copy=False)
    if values.ndim == 3:
        values = values.sum(axis=0)
    low, high = contrast_limits(values)
    mpl_image.imsave(str(path), values, cmap=colormap, vmin=low, vmax=high, origin="upper")
    return path
