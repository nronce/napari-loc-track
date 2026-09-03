"""Opening image stacks without reading them.

A localization movie is routinely tens of GB. Nothing here decodes pixels up
front unless there is no alternative, so "Load data" returns as soon as the file
header has been parsed and frames arrive as they are actually looked at.
"""
from __future__ import annotations

import numpy as np

# Above this decoded size, a stack that cannot be memory-mapped is opened lazily
# rather than read into RAM. Decoding lazily costs more in total - every full
# pass (detect, then fit) decodes again, ~6x the cost of a single threaded read -
# so it is only worth it when the alternative is exhausting memory.
_EAGER_LIMIT_BYTES = 2 * 1024 ** 3
# Frames decoded per read when a stack has to be read into RAM. Bounds how long
# a cancel request waits, at the cost of a few extra reads.
_DECODE_CHUNK_FRAMES = 64


def _cancelled(cancel):
    return cancel is not None and cancel.is_set()


def _read_into_ram(image_path, cancel):
    """Decode a whole stack, in chunks so the read can be cancelled part way."""
    import tifffile

    with tifffile.TiffFile(image_path) as tif:
        series = tif.series[0]
        shape = series.shape
        n_frames = shape[0] if len(shape) > 2 else 0
        if n_frames == 0 or len(series.pages) != n_frames:
            # Not one page per frame; no safe way to slice it up.
            if _cancelled(cancel):
                return None
            return series.asarray()

        out = np.empty(shape, series.dtype)
        step = max(1, min(_DECODE_CHUNK_FRAMES, n_frames))
        for start in range(0, n_frames, step):
            if _cancelled(cancel):
                return None
            stop = min(start + step, n_frames)
            out[start:stop] = tif.asarray(key=slice(start, stop))
        return out


def open_image_stack(image_path, eager_limit_bytes=_EAGER_LIMIT_BYTES, cancel=None):
    """Open a TIFF stack with the cheapest strategy that works for that file.

    Returns (array, description). The array is always at least 2D and indexes
    frames on the first axis once the caller has added the leading axis for a
    single-page file. `cancel` is an event checked while decoding; if it is set,
    (None, "cancelled") is returned.
    """
    import tifffile

    mappable = False
    nbytes = 0
    try:
        with tifffile.TiffFile(image_path) as tif:
            series = tif.series[0]
            mappable = bool(
                series.dataoffset is not None and series.keyframe.is_memmappable
            )
            nbytes = int(series.size) * int(series.dtype.itemsize)
    except Exception:
        pass

    if _cancelled(cancel):
        return None, "cancelled"

    if mappable:
        # Uncompressed, native byte order, pages sequential in the file: the
        # array is just a window onto the file. Zero copy, nothing decoded, so
        # this is instant no matter how large the movie is.
        return tifffile.memmap(image_path, mode="r"), "memory-mapped"

    # Compressed, byte-swapped, or pages not sequential - OME and Micro-Manager
    # stacks are usually in this class, and decoding is unavoidable. What must be
    # avoided is imread(out="memmap"): despite the name it decodes the *entire*
    # movie into a full-size temporary file on disk, so a 20 GB stack is read,
    # decoded and written back to %TEMP% before the first frame is ever shown.
    if nbytes > eager_limit_bytes:
        try:
            import dask.array as da

            store = tifffile.imread(image_path, aszarr=True)
            return da.from_zarr(store), f"lazy, {nbytes / 1e9:.1f} GB (decoded per frame)"
        except Exception:
            pass

    stack = _read_into_ram(image_path, cancel)
    if stack is None:
        return None, "cancelled"
    return stack, "decoded into RAM"


# Bins summed per read when a stack is time-binned into RAM. Same reasoning as
# _DECODE_CHUNK_FRAMES: it bounds how long a cancel request waits.
_BIN_CHUNK_FRAMES = 256


def _is_dask(stack):
    """True for a dask array, without importing dask to find out.

    A dask array cannot exist unless dask is already imported, so the module
    name is as good as an isinstance check and costs nothing on the common path.
    """
    return type(stack).__module__.split(".")[0] == "dask"


def sum_dtype(dtype, factor):
    """A dtype wide enough to hold the sum of `factor` frames of `dtype`.

    Cameras produce uint16, and summing is the whole point of time binning: four
    frames of a bright pixel already overflow it. A wrapped sum is worse than an
    error because it looks like a legitimate dark pixel, so the frames are
    promoted rather than clipped.
    """
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return dtype
    if dtype.kind == "b":
        return np.dtype(np.uint32)
    if dtype.kind not in "ui":
        return np.dtype(np.float64)
    info = np.iinfo(dtype)
    peak, trough = int(info.max) * factor, int(info.min) * factor
    family = ((np.uint16, np.uint32, np.uint64) if dtype.kind == "u"
              else (np.int16, np.int32, np.int64))
    for candidate in family:
        limits = np.iinfo(candidate)
        if limits.min <= trough and peak <= limits.max:
            return np.dtype(candidate)
    return np.dtype(np.float64)


def _binned_lazily(stack, factor, n_bins, out_dtype):
    """Bin without materialising, for stacks too large to hold binned."""
    import dask.array as da

    frame_shape = tuple(int(n) for n in stack.shape[1:])
    if not _is_dask(stack):
        # One chunk per bin, so computing a bin never reads more than it needs.
        stack = da.from_array(stack, chunks=(factor,) + frame_shape)
    grouped = stack[: n_bins * factor].reshape((n_bins, factor) + frame_shape)
    return grouped.sum(axis=1, dtype=out_dtype)


def bin_frames(stack, factor, eager_limit_bytes=_EAGER_LIMIT_BYTES, cancel=None):
    """Sum every `factor` consecutive frames into one. Returns (array, description).

    Summed, not averaged. The fit is a maximum-likelihood estimate under photon
    statistics, and the sum of N Poisson frames is Poisson with N times the rate
    - so a summed bin is still something the fit can model, while an average is
    not: it would divide the photon count by N and leave the noise looking N
    times too large for the signal.

    Two consequences follow and are the caller's to apply: the binned frame
    carries N times the camera baseline, because each raw frame brought its own,
    and it spans N times the exposure, so the frame rate is N times lower.

    Frames left over at the end - fewer than `factor` of them - are dropped
    rather than forming a short bin, which would otherwise be the one frame in
    the movie with a different exposure and a different baseline.
    """
    factor = int(factor)
    if factor <= 1:
        return stack, "not time-binned"

    n_raw = int(stack.shape[0])
    n_bins = n_raw // factor
    if n_bins < 1:
        return stack, f"time binning by {factor} skipped: only {n_raw} frame(s)"

    frame_shape = tuple(int(n) for n in stack.shape[1:])
    out_dtype = sum_dtype(stack.dtype, factor)
    note = (f"time-binned {n_raw} frames into {n_bins} sums of {factor}, "
            f"{np.dtype(stack.dtype)} -> {out_dtype}")
    dropped = n_raw - n_bins * factor
    if dropped:
        note += f"; {dropped} trailing frame(s) dropped as an incomplete bin"

    nbytes = n_bins * int(np.prod(frame_shape)) * out_dtype.itemsize
    if _is_dask(stack) or nbytes > eager_limit_bytes:
        try:
            return _binned_lazily(stack, factor, n_bins, out_dtype), f"{note} (lazily)"
        except Exception:
            # No dask, or a stack it cannot wrap. Materialising is worse than
            # lazy here but better than refusing to bin at all.
            pass

    out = np.empty((n_bins,) + frame_shape, out_dtype)
    step = max(1, _BIN_CHUNK_FRAMES // factor)
    for start in range(0, n_bins, step):
        if _cancelled(cancel):
            return None, "cancelled"
        stop = min(start + step, n_bins)
        block = np.asarray(stack[start * factor:stop * factor])
        out[start:stop] = block.reshape(
            (stop - start, factor) + frame_shape).sum(axis=1, dtype=out_dtype)
    return out, note
