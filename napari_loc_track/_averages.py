"""Time-averaged images of a corrected movie: the diffraction-limited image, or the
white light, each in the geometry the correction puts it in.

A stack displayed drift-corrected (ShiftedStack) or in the final geometry of the
growth correction (WarpedStack) moves every frame its own way. Moving a
fluorescence movie of tens of thousands of frames one by one would take many
minutes, so the frames are summed as recorded in short blocks - within a second
or two the sample moves by far less than the diffraction limit - and each block
is moved once, by the correction of its middle frame. The white-light snapshots
are few and far apart: each is moved on its own (a block of one).

Pixels that no frame covered are NaN; each pixel is the mean over the frames
that covered it.
"""
from __future__ import annotations

import numpy as np

from ._drift import ShiftedStack, WarpedStack, unshifted

BLOCK_FRAMES = 60


def _move(displayed, index, plane, grid):
    """plane moved as `displayed` moves frame `index`: (image, covered) on its canvas."""
    from scipy import ndimage

    plane = np.asarray(plane, dtype=np.float32)
    if isinstance(displayed, WarpedStack):
        src = displayed._inverse(int(index), grid)
        shape = displayed._canvas
        out = ndimage.map_coordinates(plane, [src[:, 1], src[:, 0]], order=1, mode="constant",
                                      cval=np.nan, prefilter=False).reshape(shape)
        return out
    if isinstance(displayed, ShiftedStack):
        shifts = displayed.shifts_yx
        sy, sx = shifts[index] if index < len(shifts) else (0.0, 0.0)
        oy, ox = displayed.origin_yx
        return ndimage.affine_transform(plane, np.eye(2), offset=(-(sy - oy), -(sx - ox)),
                                        output_shape=displayed._canvas, order=1,
                                        mode="constant", cval=np.nan, prefilter=False)
    return plane


def _canvas_grid(displayed):
    if not isinstance(displayed, WarpedStack):
        return None
    rows, cols = np.mgrid[0:displayed._canvas[0], 0:displayed._canvas[1]]
    return np.column_stack([cols.ravel() + displayed._origin[1],
                            rows.ravel() + displayed._origin[0]]).astype(float)


def canvas_origin(displayed):
    """Where the averaged image's first pixel sits (row, column), in the stack's
    reference pixels."""
    return tuple(displayed.origin_yx) if isinstance(displayed, ShiftedStack) else (0, 0)


def average_iter(displayed, first=0, last=None, every=1, block=BLOCK_FRAMES, cancel=None):
    """Generator: yields progress 0..1, returns (mean image, frames used), or None if
    cancelled. `displayed`: the stack as drawn - ShiftedStack, WarpedStack, or a
    plain stack (then a plain mean)."""
    base = unshifted(displayed)
    n = int(base.shape[0])
    last = n - 1 if last is None else min(int(last), n - 1)
    first = max(0, int(first))
    chosen = np.arange(first, last + 1, max(1, int(every)))
    if not len(chosen):
        raise ValueError("no frames in that range")
    block = max(1, int(block))
    grid = _canvas_grid(displayed)
    total = None
    weight = None
    blocks = [chosen[i:i + block] for i in range(0, len(chosen), block)]
    for b, frames in enumerate(blocks):
        if cancel is not None and cancel.is_set():
            return None
        acc = np.zeros(base.shape[1:], np.float64)
        for i in frames:
            acc += np.asarray(base[int(i)], dtype=np.float64)
        mean = acc / len(frames)
        moved = _move(displayed, int(frames[len(frames) // 2]), mean, grid)
        covered = np.isfinite(moved)
        if total is None:
            total = np.zeros(moved.shape, np.float64)
            weight = np.zeros(moved.shape, np.float64)
        total[covered] += moved[covered] * len(frames)
        weight[covered] += len(frames)
        yield (b + 1) / len(blocks)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(weight > 0, total / np.maximum(weight, 1e-300), np.nan)
    return out.astype(np.float32), int(len(chosen))


def average(displayed, **kwargs):
    gen = average_iter(displayed, **kwargs)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value
