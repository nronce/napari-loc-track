"""The growth deformation drawn over the white-light snapshots, for a viewer.

Everything is placed in the geometry of the first snapshot - the frame the
drift-corrected snapshots are shown in, where snapshot k is moved by the drift
record since the first (`shifts[k]`, row and column; `shifts[0]` is zero). The
record's maps go from each snapshot to the last; a position g in the first
snapshot is where `forward[0]` takes it in the last, and `inverse[k]` of that is
the same piece of tissue in snapshot k.

- `StageStack`: (stage, snapshot, row, column), lazily. Snapshot k carried into
  the first snapshot's geometry through the maps of stage s: as recorded, after
  the rough chaining, after each refinement pass - the growth cancelled at the last.
- `growth_arrows`: per snapshot, from where pieces of tissue were in the first
  snapshot to where they are now - the growth so far, the field centre's move
  taken out. `growth_rate_arrows`: how fast they move now.
- `deformation_grid`: a grid on the tissue of the first snapshot, along and across
  the root axis, drawn where that tissue is in every snapshot - the model itself.
- `stretch_maps`: the local stretch along the root since the first snapshot.
- `patch_outlines`: the patches the measurement registered, and how much of their
  structure runs across the root (cross walls, which fix the shift along it).
- `stage_arrows`: per stage, what the pass measured was still off at each patch,
  or what the model made of it.
- `growth_metrics`: the numbers, for plots.

No Qt, no napari: arrays and lazy stacks a viewer can take.
"""
from __future__ import annotations

import math

import numpy as np

from . import _deform as deform


def _invert(forward, region):
    x0, x1, y0, y1 = region
    gx, gy = np.meshgrid(np.linspace(x0, x1, 24), np.linspace(y0, y1, 24))
    src = np.column_stack([gx.ravel(), gy.ravel()])
    return deform.QuadMap.fit(forward(src), src, forward.center, forward.scale)


def first_region(record):
    """(x0, x1, y0, y1): the measured region, in the first snapshot's pixels."""
    x0, x1, y0, y1 = record.region
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1],
                        [(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1], [x0, (y0 + y1) / 2],
                        [x1, (y0 + y1) / 2]], float)
    here = record.inverse[0](corners)
    return (int(np.floor(here[:, 0].min())), int(np.ceil(here[:, 0].max())),
            int(np.floor(here[:, 1].min())), int(np.ceil(here[:, 1].max())))


def _axis(record):
    a = np.array([math.cos(record.theta), math.sin(record.theta)])
    return a, np.array([-a[1], a[0]])


def _where(record, g, k):
    """The tissue at first-snapshot positions g (N, 2), in snapshot k's own pixels.
    The first snapshot is itself exactly - not through the last and back, which
    two separately fitted maps do only to a hundredth of a pixel."""
    g = np.atleast_2d(np.asarray(g, dtype=float))
    if int(k) == 0:
        return g.copy()
    return record.inverse[k](record.forward[0](g))


def _jacobian(record, g, k):
    """(N, 2, 2): how the first snapshot's tissue at g is stretched in snapshot k."""
    g = np.atleast_2d(np.asarray(g, dtype=float))
    if int(k) == 0:
        return np.broadcast_to(np.eye(2), (len(g), 2, 2)).copy()
    return np.einsum("nij,njk->nik", record.inverse[k].jacobian(record.forward[0](g)),
                     record.forward[0].jacobian(g))


def stage_maps(record):
    """[stage] -> (forward maps per snapshot, inverse maps per snapshot), and the
    stage names. Stage 0 is the snapshots as recorded (no map)."""
    first = record.forward[0]
    identity = deform.QuadMap.identity(first.center, first.scale)
    n = len(record.t)
    stages = [([identity] * n, [identity] * n)]
    names = ["as recorded"]
    steps = record.steps or {}
    forward = steps.get("forward")
    if forward is not None and len(forward) > 1:
        for s, coefs in enumerate(forward):
            fwd = [deform.QuadMap(np.asarray(c), first.center, first.scale) for c in coefs]
            inv = [_invert(m, record.region) for m in fwd[:-1]] + [identity]
            stages.append((fwd, inv))
            names.append("rough maps (neighbours chained)" if s == 0 else f"after pass {s}")
        stages[-1] = (list(record.forward), list(record.inverse))   # exactly the record's
    else:
        stages.append((list(record.forward), list(record.inverse)))
        names.append("growth map")
    return stages, names


class StageStack:
    """(stage, snapshot, row, column): each snapshot through each stage's maps, in
    the first snapshot's geometry, computed when shown."""

    def __init__(self, stack, record, cache=8):
        self.stack = stack
        self.record = record
        self.maps, self.names = stage_maps(record)
        self.box = first_region(record)
        x0, x1, y0, y1 = self.box
        yy, xx = np.mgrid[y0:y1, x0:x1]
        self._grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
        self._cache = {}
        self._cache_size = int(cache)
        self.dtype = np.dtype(np.float32)

    @property
    def shape(self):
        x0, x1, y0, y1 = self.box
        return (len(self.maps), len(self.record.t), y1 - y0, x1 - x0)

    @property
    def ndim(self):
        return 4

    @property
    def size(self):
        return int(np.prod(self.shape))

    def __len__(self):
        return self.shape[0]

    def plane(self, s, k):
        from scipy import ndimage

        key = (int(s), int(k))
        if key not in self._cache:
            fwd, inv = self.maps[int(s)]
            src = inv[int(k)](fwd[0](self._grid))
            image = np.asarray(self.stack[int(k)], dtype=np.float32)
            out = ndimage.map_coordinates(image, [src[:, 1], src[:, 0]], order=1,
                                          mode="constant", cval=0.0, prefilter=False)
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = out.reshape(self.shape[2:]).astype(np.float32)
        return self._cache[key]

    def __array__(self, dtype=None, copy=None):
        out = np.stack([np.stack([self.plane(s, k) for k in range(self.shape[1])])
                        for s in range(self.shape[0])])
        return out if dtype is None else out.astype(dtype)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        key = key + (slice(None),) * (4 - len(key))
        s_key, k_key, rows, cols = key[:4]
        s_list = np.atleast_1d(np.arange(self.shape[0])[s_key])
        k_list = np.atleast_1d(np.arange(self.shape[1])[k_key])
        planes = np.stack([np.stack([self.plane(s, k)[rows, cols] for k in k_list])
                           for s in s_list])
        if isinstance(s_key, (int, np.integer)):
            planes = planes[0]
            if isinstance(k_key, (int, np.integer)):
                planes = planes[0]
        elif isinstance(k_key, (int, np.integer)):
            planes = planes[:, 0]
        return planes


def _first_grid(record, spacing=None):
    x0, x1, y0, y1 = first_region(record)
    spacing = spacing or max(x1 - x0, y1 - y0) / 12.0
    gx, gy = np.meshgrid(np.arange(x0 + spacing / 2, x1, spacing),
                         np.arange(y0 + spacing / 2, y1, spacing))
    return np.column_stack([gx.ravel(), gy.ravel()])


def _displayed(record, g, k, shifts):
    """Where the tissue at first-snapshot positions g is drawn in snapshot k (x, y)."""
    return _where(record, g, k) + np.asarray(shifts, dtype=float)[k][::-1]


def growth_arrows(record, shifts, spacing=None):
    """(vectors (N, 2, 3), magnitude (N,) px): per snapshot k, arrows from where
    pieces of tissue were in the first snapshot to where they are drawn in
    snapshot k, less the field centre's move - the growth so far. Rows are
    [[k, row, col], [0, drow, dcol]] in the drift-corrected frame's pixels."""
    grid = _first_grid(record, spacing)
    x0, x1, y0, y1 = first_region(record)
    centre = np.array([[(x0 + x1) / 2.0, (y0 + y1) / 2.0]])
    rows, magnitude = [], []
    for k in range(len(record.t)):
        now = _displayed(record, grid, k, shifts)
        centre_move = _displayed(record, centre, k, shifts)[0] - centre[0]
        move = now - grid - centre_move
        base = grid + centre_move                 # the tissue's start, moved with the field
        for (bx, by), (dx, dy) in zip(base, move):
            rows.append([[k, by, bx], [0.0, dy, dx]])
            magnitude.append(math.hypot(dx, dy))
    return np.asarray(rows, dtype=float).reshape(-1, 2, 3), np.asarray(magnitude)


def growth_rate_arrows(record, shifts, spacing=None):
    """(vectors (N, 2, 3), speed (N,) px/min): per snapshot, how fast the tissue
    moves there now, the field centre's motion taken out; based on the tissue."""
    grid = _first_grid(record, spacing)
    x0, x1, y0, y1 = first_region(record)
    centre = np.array([[(x0 + x1) / 2.0, (y0 + y1) / 2.0]])
    minutes = np.asarray(record.t, dtype=float) / 60.0
    n = len(minutes)
    rows, speed = [], []
    for k in range(n):
        a, b = max(0, k - 1), min(n - 1, k + 1)
        if b == a:
            continue
        dt = minutes[b] - minutes[a]
        now = _displayed(record, grid, k, shifts)
        velocity = (_displayed(record, grid, b, shifts) - _displayed(record, grid, a, shifts)) / dt
        centre_v = (_displayed(record, centre, b, shifts) - _displayed(record, centre, a, shifts))[0] / dt
        velocity -= centre_v
        for (bx, by), (vx, vy) in zip(now, velocity):
            rows.append([[k, by, bx], [0.0, vy, vx]])
            speed.append(math.hypot(vx, vy))
    return np.asarray(rows, dtype=float).reshape(-1, 2, 3), np.asarray(speed)


def deformation_grid(record, shifts, n_lines=9, n_points=40):
    """The model, drawn: lines along (kind "along") and across ("across") the root
    axis through the first snapshot's region, carried with the tissue into every
    snapshot. Returns [(k, kind, (P, 2) row/col points)]."""
    x0, x1, y0, y1 = first_region(record)
    a, m = _axis(record)
    centre = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    half = 0.5 * math.hypot(x1 - x0, y1 - y0)
    offsets = np.linspace(-half, half, n_lines)
    t = np.linspace(-half, half, n_points)
    box = np.array([x0, y0]), np.array([x1, y1])
    lines = []
    for kind, along, across in (("along", a, m), ("across", m, a)):
        for off in offsets:
            points = centre + np.outer(t, along) + off * across
            inside = np.all((points >= box[0]) & (points <= box[1]), axis=1)
            if inside.sum() >= 2:
                lines.append((kind, points[inside]))
    out = []
    for k in range(len(record.t)):
        for kind, points in lines:
            xy = _displayed(record, points, k, shifts)
            out.append((k, kind, xy[:, ::-1]))
    return out


def stretch_maps(record, spacing=None):
    """((snapshots, rows, cols) stretch along the root since the first snapshot, %,
    (row, col) of the first sample, spacing) on a grid over the first region."""
    x0, x1, y0, y1 = first_region(record)
    spacing = spacing or max(x1 - x0, y1 - y0) / 48.0
    xs = np.arange(x0 + spacing / 2, x1, spacing)
    ys = np.arange(y0 + spacing / 2, y1, spacing)
    gx, gy = np.meshgrid(xs, ys)
    g = np.column_stack([gx.ravel(), gy.ravel()])
    a, _m = _axis(record)
    out = []
    for k in range(len(record.t)):
        J = _jacobian(record, g, k)
        out.append((np.einsum("i,nij,j->n", a, J, a) - 1.0).reshape(gx.shape) * 100.0)
    return np.asarray(out, dtype=np.float32), (float(ys[0]), float(xs[0])), float(spacing)


def patch_outlines(record, size_px=None):
    """([(4, 2) row/col corners] per patch, in the first snapshot's pixels, and the
    share of each patch's gradient that runs along the root axis - its cross
    walls, the only thing that fixes a shift along the root). size_px: the side
    of the squares drawn - the patch itself by default; the grid's step draws
    them as tiles that do not overlap, one per patch."""
    steps = record.steps or {}
    if "centres" not in steps:
        return [], np.zeros(0)
    size = float(size_px or record.stats.get("patch_px", deform.PATCH_PX))
    centres = np.asarray(steps["centres"], dtype=float)
    h = size / 2.0
    square = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    outlines = [record.inverse[0](c + square)[:, ::-1] for c in centres]
    tensor = steps.get("patch_tensor_last")
    along = np.full(len(centres), np.nan)
    if tensor is not None and len(tensor) == len(centres):
        a, _m = _axis(record)
        t = np.asarray(tensor, dtype=float)
        T = np.stack([np.stack([t[:, 0], t[:, 2]], -1), np.stack([t[:, 2], t[:, 1]], -1)], -2)
        trace = t[:, 0] + t[:, 1]
        along = np.einsum("i,nij,j->n", a, T, a) / np.where(trace > 0, trace, 1.0)
    return outlines, along


def stage_arrows(record, which="measured"):
    """(vectors (N, 2, 4), magnitude (N,) px) at the patches, in the first
    snapshot's pixels, per stage and snapshot: "measured" - how far the snapshot,
    mapped with that stage's maps, was still off; "model" - what the global fit
    made of all the pairs. Rows are [[stage, k, row, col], [0, 0, drow, dcol]];
    stages numbered as in `stage_maps` (stage 0, as recorded, has none)."""
    steps = record.steps or {}
    if "measured_to_last" not in steps:
        return np.zeros((0, 2, 4)), np.zeros(0)
    centres = np.asarray(steps["centres"], dtype=float)
    at = record.inverse[0](centres)
    Jinv = record.inverse[0].jacobian(centres)          # final -> first, locally
    rows, magnitude = [], []
    for p in range(len(steps["measured_to_last"])):
        for k in range(len(steps["measured_to_last"][p])):
            if which == "measured":
                d = np.asarray(steps["measured_to_last"][p][k], dtype=float)
                ok = ((np.asarray(steps["quality_to_last"][p][k]) > deform.MIN_QUALITY)
                      & np.isfinite(d[:, 0]))
            else:
                d = -np.asarray(steps["correction"][p][k], dtype=float)
                ok = np.isfinite(d[:, 0])
            d_first = np.einsum("nij,nj->ni", Jinv, np.nan_to_num(d))
            for (cx, cy), (dx, dy) in zip(at[ok], d_first[ok]):
                rows.append([[p + 1, k, cy, cx], [0.0, 0.0, dy, dx]])
                magnitude.append(math.hypot(dx, dy))
    return np.asarray(rows, dtype=float).reshape(-1, 2, 4), np.asarray(magnitude)


def _sliding_slope(x, y, window=5):
    """The slope of y against x, fitted over `window` neighbouring samples."""
    n = len(x)
    out = np.zeros(n)
    if n < 2:
        return out
    half = max(1, window // 2)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        xs, ys = x[lo:hi], y[lo:hi]
        if len(xs) >= 2 and np.ptp(xs) > 0:
            out[i] = np.polyfit(xs, ys, 1)[0]
    return out


def growth_metrics(record, px_nm, n_profile=41):
    """What the growth did, for plotting. Times in minutes from the first snapshot.

    stretch_along / stretch_across (%): since the first snapshot, at the region's
    centre. rate_along (%/min): how fast the tissue there elongates. centre_um:
    the move of the field centre since the first snapshot (x, y, um). profile: the
    stretch from the first snapshot to the last at points along the root axis
    through the centre - where along the root it grew. passes: rms correction per
    pass (nm); lags: median registration residual per pair distance (nm).
    """
    t = np.asarray(record.t, dtype=float)
    minutes = (t - t[0]) / 60.0
    x0, x1, y0, y1 = first_region(record)
    c0 = np.array([[(x0 + x1) / 2.0, (y0 + y1) / 2.0]])
    a, m = _axis(record)
    along, across, moves = [], [], []
    for k in range(len(t)):
        J = _jacobian(record, c0, k)[0]
        along.append(100.0 * (a @ J @ a - 1.0))
        across.append(100.0 * (m @ J @ m - 1.0))
        moves.append(_where(record, c0, k)[0] - c0[0])
    along, across, moves = np.array(along), np.array(across), np.array(moves)
    rate = _sliding_slope(minutes, along, window=5)
    reach = [0.5 * (x1 - x0) / abs(a[0]) if abs(a[0]) > 1e-9 else np.inf,
             0.5 * (y1 - y0) / abs(a[1]) if abs(a[1]) > 1e-9 else np.inf]
    half = max(0.95 * min(reach), 1.0)
    s = np.linspace(-half, half, n_profile)
    points = c0[0] + np.outer(s, a)                      # in the first snapshot
    Jend = _jacobian(record, points, len(t) - 1)
    profile = 100.0 * (np.einsum("i,nij,j->n", a, Jend, a) - 1.0)
    stats = record.stats or {}
    passes = [p.get("rms_correction_px", np.nan) * px_nm for p in stats.get("passes", [])]
    lags = (stats.get("passes") or [{}])[-1].get("median_abs_residual_px_by_lag") or {}
    lag_keys = sorted(lags, key=int)
    return {
        "minutes": minutes,
        "stretch_along": along, "stretch_across": across, "rate_along": rate,
        "centre_um": moves * px_nm / 1000.0,
        "profile_um": s * px_nm / 1000.0, "profile_stretch": profile,
        "passes_nm": np.asarray(passes, dtype=float),
        "lags": np.array([int(k) for k in lag_keys]),
        "lag_residual_nm": np.array([lags[k] * px_nm for k in lag_keys], dtype=float),
        "axis_deg": math.degrees(record.theta),
    }


def describe_measurement(record, px_nm):
    """How the patches and the pairs were chosen, in words, for the viewer."""
    stats = record.stats or {}
    patch = int(stats.get("patch_px", deform.PATCH_PX))
    step = int(stats.get("step_px", deform.STEP_PX))
    n = int(stats.get("n_patches", 0))
    lags = stats.get("lags", list(deform.LAGS))
    passes = len(stats.get("passes", []))
    axis = stats.get("axis") or {}
    return (
        f"Patches: squares of {patch} px ({patch * px_nm / 1000:.0f} µm) on a grid every "
        f"{step} px ({step * px_nm / 1000:.0f} µm) over the region measured (the fluorescence "
        f"field and a margin), laid in the last snapshot and kept where the tissue has "
        f"texture: {n} of them. Their outlines are drawn in the first snapshot, "
        f"coloured by the share of their structure that runs across the root - cross "
        f"walls, the only thing that fixes where the tissue is along it.\n"
        f"Pairs: every snapshot against those {', '.join(str(v) for v in lags)} snapshots "
        f"later, and against the last. Each patch of each pair is registered at wall "
        f"scale (band-passed, every {stats.get('sampling_px', 1)} px), and one weighted "
        f"fit of all of them gives every snapshot's map; {passes} passes.\n"
        f"Root axis: {math.degrees(record.theta):.0f}°, from the "
        f"{axis.get('axis_from', 'strain')}.")
