"""A movie of a growth deformation: how the map was found, and what it undoes.

Chapter 1, the measurement, on the first snapshot - the one furthest from the
last, with the most to correct. Each stage's map carries it onto the last
snapshot: as recorded, after the rough chaining of neighbouring snapshots, and
after each refinement pass. The first snapshot is magenta, the last green; walls
that agree are white. Yellow arrows: how far each patch was still off at that
stage, as the next pass measured it. Cyan arrows: what the model, fitted to all
pairs of snapshots at once, made of it.

Chapter 2, the time-lapse. Left: each snapshot as recorded, with arrows showing
how far each piece of it still has to go to reach the final geometry - the
growth, with the translation of the field centre taken out (it is printed).
Middle: moved by the drift record alone, as a rigid correction does; the tissue
still slides as it grows. Right: through the growth map; the growth is
cancelled, and the tissue stands still in the final geometry.

Plain numpy, scipy and matplotlib (drawn offscreen); written as a GIF, or as
an MP4 when imageio-ffmpeg is installed.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import _deform as deform

MOVIE_FILENAME = "deformation_movie"
NM_PER_PX = 81.28
FRAME_SECONDS = 0.25          # one snapshot of the time-lapse
STAGE_SECONDS = 1.6           # one stage of the measurement
FIGSIZE = (12.0, 6.72)
DPI = 100


def _stretch(img, lo, hi):
    return np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)


def _walls(img, lo, hi):
    """Walls bright and the cytoplasm between them dimmed, for overlays; black
    where the image did not look."""
    return np.where(np.isfinite(img), (1 - _stretch(img, lo, hi)) ** 2, 0.0)


def _sample(image, pts, shape):
    from scipy import ndimage

    out = ndimage.map_coordinates(np.asarray(image, np.float32), [pts[:, 1], pts[:, 0]], order=1,
                                  mode="constant", cval=np.nan)
    return out.reshape(shape)


def _round_gain(want):
    choices = (0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500)
    fitting = [g for g in choices if g <= want]
    return fitting[-1] if fitting else choices[0]


def _stage_label(i, n_passes):
    if i == 0:
        return "as recorded"
    if i == 1:
        return "rough maps: neighbours chained"
    return f"after refinement pass {i - 1} of {n_passes}"


class _Canvas:
    def __init__(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=FIGSIZE, dpi=DPI, facecolor="black")
        self.canvas = FigureCanvasAgg(self.figure)

    def frame(self):
        self.canvas.draw()
        rgba = np.asarray(self.canvas.buffer_rgba())
        return rgba[..., :3].copy()


def _style(ax, title):
    ax.set_title(title, color="w", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.3")


def movie_frames_iter(record, stack, translation=None, step=2, cancel=None, log=None):
    """Generator: yields progress 0..1, returns [(rgb frame, seconds)], or None
    if cancelled.

    record: a DeformationRecord (its `steps`, when it has them, make chapter 1).
    stack: the white-light snapshots it was measured on. translation: the drift
    record, smoothed - a function t -> (dx, dy) in white-light px - for the
    rigid panel; without it that panel shows the snapshots as recorded.
    step: the movie's pixel, in white-light pixels.
    """
    say = log or (lambda _m: None)
    n = len(record.t)
    K = n - 1
    if len(stack) != n:
        raise ValueError(f"the record has {n} snapshots and the stack {len(stack)}")
    x0, x1, y0, y1 = record.region
    H, W = record.frame_shape
    x0, x1, y0, y1 = max(0, x0), min(W, x1), max(0, y0), min(H, y1)
    yy, xx = np.mgrid[y0:y1:step, x0:x1:step]
    shape = yy.shape
    pts = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    extent = (x0, x1, y1, y0)
    raw = [np.asarray(stack[k], np.float32) for k in range(n)]
    last_box = raw[K][y0:y1, x0:x1]
    lo, hi = np.percentile(last_box, [1, 99.7])
    minutes = (record.t - record.t[-1]) / 60.0
    along, across = record.strain_along_axis()
    centre = np.array([[(x0 + x1) / 2.0, (y0 + y1) / 2.0]])
    frames = []
    canvas = _Canvas()
    fig = canvas.figure
    steps = record.steps or {}
    passes = record.stats.get("passes", [])
    work = n + (len(steps.get("forward", [])) if steps else 0)
    done = [0]

    def tick():
        done[0] += 1
        return min(done[0] / max(work, 1), 0.99)

    # --- chapter 1: the measurement, on the first snapshot -----------------------
    if steps and "forward" in steps and len(steps["forward"]) > 1:
        centres = np.asarray(steps["centres"])
        filtered_first = deform.wall_filter(raw[0])
        filtered_last = deform.wall_filter(raw[K])
        flo, fhi = np.percentile(filtered_last[y0:y1, x0:x1], [2, 99.5])
        stages = [np.eye(3)] + [None] * len(steps["forward"])
        n_passes = len(steps["measured_to_last"])
        rms = [p["rms_correction_px"] for p in passes][:n_passes]
        engine = deform.PatchRegistration(record.stats.get("patch_px", deform.PATCH_PX) // step)
        first_arrows = None
        for i in range(len(stages)):
            if cancel is not None and cancel.is_set():
                return None
            if i == 0:
                mapped = _sample(filtered_first, pts, shape)
                measured = fitted = None
            else:
                forward = deform.QuadMap(np.asarray(steps["forward"][i - 1][0]), record.forward[0].center,
                                         record.forward[0].scale)
                inverse = _invert(forward, record.region)
                mapped = _sample(filtered_first, inverse(pts), shape)
                if i - 1 < n_passes:
                    measured = np.asarray(steps["measured_to_last"][i - 1][0])
                    measured[np.asarray(steps["quality_to_last"][i - 1][0]) <= deform.MIN_QUALITY] = np.nan
                    fitted = -np.asarray(steps["correction"][i - 1][0])
                else:
                    # after the last pass nothing measured it: measure it now
                    measured = _measure_now(engine, mapped, _sample(filtered_last, pts, shape),
                                            centres, (x0, y0), step)
                    fitted = None
            if first_arrows is None and measured is not None:
                size = np.nanmax(np.hypot(*measured.T)) if np.isfinite(measured).any() else 1.0
                first_arrows = _round_gain(0.12 * (x1 - x0) / max(size, 1e-9))
            fig.clear()
            ax = fig.add_axes([0.02, 0.05, 0.58, 0.86])
            last = _walls(_sample(filtered_last, pts, shape), flo, fhi)
            first = _walls(mapped, flo, fhi)
            ax.imshow(np.dstack([first, last, first]), extent=extent, interpolation="bilinear")
            gain = first_arrows or 1.0
            if measured is not None:
                ax.quiver(centres[:, 0], centres[:, 1], measured[:, 0], measured[:, 1], color="yellow",
                          angles="xy", scale_units="xy", scale=1 / gain, width=0.004)
            if fitted is not None:
                ax.quiver(centres[:, 0], centres[:, 1], fitted[:, 0], fitted[:, 1], color="cyan",
                          angles="xy", scale_units="xy", scale=1 / gain, width=0.0025)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            _style(ax, f"The measurement - first snapshot ({-minutes[0]:.1f} min before the last) "
                       f"mapped onto the last:\n{_stage_label(i, n_passes)}")
            side = fig.add_axes([0.66, 0.52, 0.31, 0.36], facecolor="0.08")
            if rms:
                side.semilogy(np.arange(1, len(rms) + 1), np.asarray(rms) * NM_PER_PX, "o-", color="0.7")
                if 1 <= i <= len(rms):
                    # the pass that measured this stage, and corrected it
                    side.semilogy([i], [rms[i - 1] * NM_PER_PX], "o", color="yellow", ms=12)
            side.set_xlabel("refinement pass", color="w")
            side.set_ylabel("rms correction at the patches (nm)", color="w")
            side.tick_params(colors="w")
            side.set_title("each pass corrects what the one before left", color="w", fontsize=10)
            text = ["magenta: first snapshot, mapped", "green: last snapshot", "white: walls in register", ""]
            if measured is not None:
                off = np.nanmedian(np.hypot(*measured.T)) * NM_PER_PX
                text.append(f"yellow: still off at each patch (x{gain:g}),\n  median {off:.0f} nm")
            if fitted is not None:
                text.append(f"cyan: the model's fit to all pairs (x{gain:g})")
            if i == 0:
                disp = record.forward[0](centre)[0] - centre[0]
                text.append(f"to reach the last snapshot, the tissue here\nmoves {np.hypot(*disp) * NM_PER_PX / 1000:.1f} um "
                            f"and stretches {100 * along[0]:.1f} % along the root")
            fig.text(0.66, 0.44, "\n".join(text), color="w", fontsize=11, va="top", family="monospace")
            frames.append((canvas.frame(), STAGE_SECONDS * (1.5 if i in (0, len(stages) - 1) else 1)))
            yield tick()

    # --- chapter 2: the time-lapse ------------------------------------------------
    if translation is not None:
        tx, ty = translation(record.t)
        shifts = np.column_stack([tx[K] - np.asarray(tx), ty[K] - np.asarray(ty)])
    else:
        shifts = np.zeros((n, 2))
    grid_x, grid_y = np.meshgrid(np.linspace(x0 + 50, x1 - 50, 9), np.linspace(y0 + 50, y1 - 50, 11))
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    def growth_part(k, p):
        move = record.forward[k](p) - p
        return move - (record.forward[k](centre)[0] - centre[0])

    biggest = max(float(np.hypot(*growth_part(0, grid).T).max()), 1e-9)
    gain = _round_gain(0.18 * (x1 - x0) / biggest)
    for k in range(n):
        if cancel is not None and cancel.is_set():
            return None
        fig.clear()
        panels = [fig.add_axes([0.01 + 0.33 * c, 0.24, 0.31, 0.66]) for c in range(3)]
        recorded = _stretch(raw[k][y0:y1:step, x0:x1:step], lo, hi)
        panels[0].imshow(recorded, cmap="gray", extent=extent, vmin=0, vmax=1)
        g = growth_part(k, grid)
        panels[0].quiver(grid[:, 0], grid[:, 1], g[:, 0], g[:, 1], np.hypot(*g.T) * NM_PER_PX / 1000,
                         cmap="plasma", clim=(0, biggest * NM_PER_PX / 1000), angles="xy",
                         scale_units="xy", scale=1 / gain, width=0.005)
        move = (record.forward[k](centre)[0] - centre[0]) * NM_PER_PX / 1000
        _style(panels[0], f"as recorded, growth still to come (arrows x{gain:g})\n"
                          f"plus a move of ({move[0]:+.1f}, {move[1]:+.1f}) um")
        rigid = _sample(raw[k], pts - shifts[k], shape)
        panels[1].imshow(_stretch(rigid, lo, hi), cmap="gray", extent=extent, vmin=0, vmax=1)
        _style(panels[1], "drift record only (rigid)" if translation is not None
               else "as recorded (no drift record given)")
        cancelled_growth = _sample(raw[k], record.inverse[k](pts), shape)
        panels[2].imshow(_stretch(cancelled_growth, lo, hi), cmap="gray", extent=extent, vmin=0, vmax=1)
        _style(panels[2], "growth cancelled: in the final geometry")
        for ax in panels:
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
        strip = fig.add_axes([0.06, 0.06, 0.88, 0.12], facecolor="0.08")
        strip.plot(minutes, 100 * along, "-", color="0.7", label="along the root")
        strip.plot(minutes, 100 * across, "--", color="0.5", label="across")
        strip.plot([minutes[k]], [100 * along[k]], "o", color="yellow", ms=9)
        strip.set_xlabel("minutes before the last snapshot", color="w")
        strip.set_ylabel("stretch to go (%)", color="w")
        strip.tick_params(colors="w")
        strip.legend(fontsize=8, loc="upper right", facecolor="0.1", labelcolor="w")
        fig.text(0.5, 0.955, f"snapshot {k + 1} of {n}, {minutes[k]:+.1f} min", color="w",
                 fontsize=13, ha="center")
        frames.append((canvas.frame(), FRAME_SECONDS * (4 if k in (0, K) else 1)))
        yield tick()
    say(f"{len(frames)} movie frames drawn")
    return frames


def _invert(forward, region):
    """The inverse of a quadratic map over a region, by a fit on a grid."""
    x0, x1, y0, y1 = region
    gx, gy = np.meshgrid(np.linspace(x0, x1, 24), np.linspace(y0, y1, 24))
    src = np.column_stack([gx.ravel(), gy.ravel()])
    return deform.QuadMap.fit(forward(src), src, forward.center, forward.scale)


def _measure_now(engine, mapped, last, centres, origin, step):
    """How far each patch of the mapped image is off the last one, in px."""
    size = engine.n
    starts = np.column_stack([np.round((centres[:, 1] - origin[1]) / step) - size // 2,
                              np.round((centres[:, 0] - origin[0]) / step) - size // 2]).astype(int)
    d, q = engine.register(engine.bank(deform._cut(last, starts, size)),
                           engine.bank(deform._cut(mapped, starts, size)))
    d = d * step
    d[q <= deform.MIN_QUALITY] = np.nan
    return d


def write_movie(frames, path):
    """Write [(rgb, seconds)] as an MP4 when imageio-ffmpeg is there, else a GIF.
    Returns the path written."""
    path = Path(path)
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401

        fps = 20
        out = path.with_suffix(".mp4")
        with imageio.get_writer(out, fps=fps, codec="libx264", quality=8,
                                macro_block_size=8) as writer:
            for rgb, seconds in frames:
                for _ in range(max(1, int(round(seconds * fps)))):
                    writer.append_data(rgb)
        return out
    except ImportError:
        pass
    from PIL import Image

    out = path.with_suffix(".gif")
    # one palette for the whole movie, from a few frames of each chapter
    probe = [frames[i][0] for i in np.linspace(0, len(frames) - 1, 6).astype(int)]
    palette = Image.fromarray(np.vstack(probe)).quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    images = [Image.fromarray(rgb).quantize(palette=palette, dither=Image.Dither.NONE)
              for rgb, _s in frames]
    images[0].save(out, save_all=True, append_images=images[1:], loop=0,
                   duration=[int(1000 * s) for _rgb, s in frames], optimize=False)
    return out


def make_movie(record, stack, path, translation=None, step=2, log=None):
    frames = None
    gen = movie_frames_iter(record, stack, translation=translation, step=step, log=log)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            frames = stop.value
            break
    return write_movie(frames, path) if frames else None
