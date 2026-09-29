"""Lateral drift measured on the white-light camera, taken out of the data.

The acquisition software (recFL, xy_drift.py) registers every frame of the
focus-lock white-light camera against the first one and writes
<name>_xy_drift.csv: how far the sample has moved, in that camera's pixels,
against the time of each measurement. Beside it, <name>_frame_times.csv says
when every page of the fluorescence stack came off the camera. Between the two,
each frame of the movie can be given the drift of its own moment, and each
localization in it moved back by that much.

The white-light camera is a different camera, and not only in pixel size: its
frames are oriented on the acquisition side to match the fluorescence camera,
but it still sits turned 0.8 degrees from it. So its drift is carried into
fluorescence-camera pixels through a 2x2 matrix - the camera map - and from
there into nanometres with the fluorescence pixel the localizations were
computed with. A ratio of pixel sizes would leave the turn out, and put 1.4 %
of each axis's drift onto the other.

Conventions are the acquisition's, so a table corrected here agrees with one
corrected offline by recFL's scripts/xy_drift_correct.py: (x, y) is (column,
row); a drift (dx, dy) > 0 means the sample moved right/down since the
reference; corrected position = measured position - drift; and the drift is
zero at the first frame of the movie, so corrected coordinates are those of the
first frame.

Only numpy, pandas and scipy are used, so all of it can be tested without napari
or Qt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DRIFT_SUFFIX = "_xy_drift.csv"
FRAME_TIMES_SUFFIX = "_frame_times.csv"
# The white-light snapshots the acquisition keeps alongside the drift record:
# one every so many seconds, far fewer than the record's samples. They are
# oriented at the camera, exactly like the frames the drift was measured on.
WL_IMAGES_SUFFIX = "_WL_images.tif"
WL_TIMES_SUFFIX = "_WL_images_times.csv"

# The drift a corrected table carries, per localization: what was subtracted.
# Named as recFL's offline corrector names them, so a table corrected by either
# is recognised by the other, and a corrected table can always be put back.
DRIFT_X_COLUMN = "drift_x [nm]"
DRIFT_Y_COLUMN = "drift_y [nm]"
DRIFT_COLUMNS = (DRIFT_X_COLUMN, DRIFT_Y_COLUMN)

# A frame clock whose pages sit this close to a straight line (median absolute
# deviation) is regular, and the line is a better clock than the timestamps:
# each of those is when a polling loop happened to take the frame off the
# buffer, a few ms after it arrived. 3 ms is a tenth of a typical frame.
_REGULAR_CLOCK_MAD_S = 0.003
_MIN_CLOCK_FRAMES = 10

# Below this, a frame is not worth resampling for display: a hundredth of a
# pixel moves nothing anyone can see.
_NEGLIGIBLE_SHIFT_PX = 1e-2


def read_csv_with_meta(path):
    """(meta, table) of a CSV that may open with a '# {json}' line."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        position = handle.tell()
        first = handle.readline()
        meta = {}
        if first.startswith("#"):
            try:
                parsed = json.loads(first[1:].strip())
                meta = parsed if isinstance(parsed, dict) else {}
            except ValueError:
                meta = {}
        else:
            handle.seek(position)
        table = pd.read_csv(handle)
    return meta, table


@dataclass
class DriftRecord:
    """The white-light camera's drift trace, as recorded."""

    path: Path
    meta: dict
    t: np.ndarray    # epoch seconds, ascending
    dx: np.ndarray   # white-light camera px; NaN where the tracker lost the sample
    dy: np.ndarray
    quality: np.ndarray = None   # correlation peak against the reference, 1 = identical
    ref_id: np.ndarray = None    # how many times the tracker had changed reference

    @property
    def good(self):
        return np.isfinite(self.t) & np.isfinite(self.dx) & np.isfinite(self.dy)

    @property
    def span(self):
        """(first, last) time of a trusted sample."""
        t = self.t[self.good]
        return float(t[0]), float(t[-1])

    @property
    def sample_interval_s(self):
        t = self.t[self.good]
        return float(np.median(np.diff(t))) if len(t) > 1 else float("nan")

    def sample_noise_px(self):
        """Scatter of a single sample about the underlying drift, in px.

        From second differences, which cancel any drift that is smooth on the
        scale of three samples and leave 6 sigma^2 of white noise. Combined over
        x and y, as an rms per axis.
        """
        good = self.good
        variances = []
        for values in (self.dx[good], self.dy[good]):
            if len(values) < 3:
                return float("nan")
            second = values[2:] - 2.0 * values[1:-1] + values[:-2]
            variances.append(float(np.var(second)) / 6.0)
        return float(np.sqrt(np.mean(variances)))


def read_drift(path):
    """Read an <name>_xy_drift.csv. Samples the tracker did not trust are NaN."""
    path = Path(path)
    meta, table = read_csv_with_meta(path)
    kind = meta.get("kind")
    if kind not in (None, "xy_drift"):
        raise ValueError(f"{path.name} is a '{kind}' file, not an xy drift record")
    missing = {"t_epoch", "dx_px", "dy_px"} - set(table.columns)
    if missing:
        raise ValueError(f"{path.name} has no {', '.join(sorted(missing))} column")
    # Copies: under pandas' copy-on-write these would be read-only views.
    t = np.array(pd.to_numeric(table["t_epoch"], errors="coerce"), dtype=float)
    dx = np.array(pd.to_numeric(table["dx_px"], errors="coerce"), dtype=float)
    dy = np.array(pd.to_numeric(table["dy_px"], errors="coerce"), dtype=float)
    if "ok" in table.columns:
        untrusted = pd.to_numeric(table["ok"], errors="coerce").to_numpy(float) < 0.5
        dx[untrusted] = np.nan
        dy[untrusted] = np.nan
    order = np.argsort(t, kind="stable")
    extras = {}
    for column, name in (("quality", "quality"), ("ref_id", "ref_id")):
        if column in table.columns:
            extras[name] = np.array(pd.to_numeric(table[column], errors="coerce"),
                                    dtype=float)[order]
    record = DriftRecord(path, meta, t[order], dx[order], dy[order], **extras)
    if int(record.good.sum()) < 2:
        raise ValueError(f"{path.name} holds fewer than two usable drift samples")
    return record


@dataclass
class FrameClock:
    """When each page of the fluorescence stack was exposed.

    `t_mid` is the middle of each recorded page's exposure. Pages past the end
    of the record are timed by extending the fitted line: the log is flushed
    once a second, so a run that ended abruptly keeps every frame in the stack
    but loses up to a second of their timestamps.
    """

    path: Path
    meta: dict
    page: np.ndarray
    t_mid: np.ndarray
    slope: float        # s per page
    intercept: float    # t_mid of page 0 on the fitted line
    regular: bool       # the timestamps were replaced by the fitted line

    @property
    def n_recorded(self):
        return int(self.page.max()) + 1 if self.page.size else 0

    def times(self, n_pages):
        """(t_mid for pages 0..n_pages-1, how many of them were extrapolated)."""
        pages = np.arange(int(n_pages))
        t = self.intercept + self.slope * pages
        inside = (self.page >= 0) & (self.page < n_pages)
        t[self.page[inside]] = self.t_mid[inside]
        known = np.zeros(len(pages), dtype=bool)
        known[self.page[inside]] = True
        return t, int((~known).sum())


def read_frame_times(path):
    """Read an <name>_frame_times.csv into a frame clock.

    t_epoch there is when each page was taken off the camera buffer. When the
    pages came at a steady rate, the straight line through the frames that
    were not kept waiting (backlog 1) replaces the raw times, pinned to their
    least-delayed edge - the same regularisation recFL applies. The result is
    moved back to mid-exposure with the readout and exposure the file records.
    """
    path = Path(path)
    meta, table = read_csv_with_meta(path)
    kind = meta.get("kind")
    if kind not in (None, "frame_times"):
        raise ValueError(f"{path.name} is a '{kind}' file, not a frame-times record")
    missing = {"page", "t_epoch"} - set(table.columns)
    if missing:
        raise ValueError(f"{path.name} has no {', '.join(sorted(missing))} column")
    page = np.array(pd.to_numeric(table["page"], errors="coerce"), dtype=float)
    t = np.array(pd.to_numeric(table["t_epoch"], errors="coerce"), dtype=float)
    keep = np.isfinite(page) & np.isfinite(t)
    page, t = page[keep].astype(np.int64), t[keep]
    if len(page) < 2:
        raise ValueError(f"{path.name} times fewer than two frames")
    if "backlog" in table.columns:
        backlog = pd.to_numeric(table["backlog"], errors="coerce").to_numpy(float)[keep]
        fresh = backlog <= 1
    else:
        fresh = np.ones(len(t), dtype=bool)
    if fresh.sum() < 2:
        fresh = np.ones(len(t), dtype=bool)

    slope, intercept = np.polyfit(page[fresh], t[fresh], 1)
    residual = t[fresh] - (intercept + slope * page[fresh])
    regular = bool(
        fresh.sum() >= _MIN_CLOCK_FRAMES and slope > 0
        and np.median(np.abs(residual - np.median(residual))) < _REGULAR_CLOCK_MAD_S)
    # The lower envelope: the line through the frames taken off the buffer
    # soonest after they arrived, which is when they really arrived.
    intercept += float(np.percentile(residual, 5))
    if regular:
        t = intercept + slope * page

    to_mid = (float(meta.get("readout_s") or 0.0)
              + 0.5 * float(meta.get("exposure_span_s") or 0.0))
    return FrameClock(path, meta, page, t - to_mid, float(slope),
                      float(intercept) - to_mid, regular)


def smoother(t, dx, dy, sigma_s):
    """f(times) -> (dx, dy): the drift, smoothed over a Gaussian of sigma_s.

    A local-linear fit under a Gaussian window rather than a local average:
    an average of a steady ramp is unbiased in the middle but pulled towards
    the inside at both ends, which is exactly where the first and last frames
    of the movie sit. Averaging in fixed time bins is worse again - it
    staircases and lags a steady drift. NaN samples are ignored; outside the
    record the edge value is held. sigma_s <= 0 interpolates the raw samples.

    Ported from recFL's xy_drift.smooth_drift, so both ends of the pipeline
    smooth the same way.
    """
    from scipy import ndimage

    t = np.asarray(t, dtype=float)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    good = np.isfinite(t) & np.isfinite(dx) & np.isfinite(dy)
    tg, xg, yg = t[good], dx[good], dy[good]
    if len(tg) < 2:
        raise ValueError("fewer than two usable drift samples")
    order = np.argsort(tg, kind="stable")
    tg, xg, yg = tg[order], xg[order], yg[order]
    if sigma_s <= 0:
        return lambda q: (np.interp(q, tg, xg), np.interp(q, tg, yg))

    # Samples are binned onto a grid a third of sigma or finer, so the window
    # is a convolution instead of one weighted fit per output time.
    dt = min(float(np.median(np.diff(tg))) or sigma_s, sigma_s / 3.0)
    n = int((tg[-1] - tg[0]) / dt) + 2
    grid = tg[0] + dt * np.arange(n)
    tau = (tg - tg[0]) / dt
    k = np.clip(np.round(tau).astype(int), 0, n - 1)

    def window(weights):
        return ndimage.gaussian_filter1d(
            np.bincount(k, weights=weights, minlength=n), sigma_s / dt, mode="constant")

    s0, s1, s2 = window(np.ones_like(tau)), window(tau), window(tau * tau)
    det = s0 * s2 - s1 * s1
    index = np.arange(n, dtype=float)
    thin = (s0 < 1e-3) | (det <= 1e-9 * np.maximum(s0 * s2, 1e-30))
    smoothed = []
    for values in (xg, yg):
        sy, sty = window(values), window(tau * values)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (s2 * sy - s1 * sty) / det
            b = (s0 * sty - s1 * sy) / det
            curve = a + b * index
        curve[thin] = np.interp(grid[thin], tg, values)
        smoothed.append(curve)
    sx, sy_ = smoothed
    return lambda q: (np.interp(q, grid, sx), np.interp(q, grid, sy_))


@dataclass(frozen=True)
class CameraMap:
    """White-light camera px -> fluorescence camera px, for displacements.

    `matrix` is the local linear part of the map between the two cameras:
    ((a, b), (c, d)) fluorescence px per white-light px, so a drift (dx, dy) on
    the white-light camera is (a dx + b dy, c dx + d dy) on the fluorescence
    one. Only differences are ever converted, so where the two images sit
    relative to each other does not enter.
    """

    matrix: tuple
    source: str                   # where it came from, in words
    wl_orientation: str = None    # the white-light orientation it holds for
    wl_frame_shape: tuple = None  # (rows, columns) of the frames it holds for
    # Where white-light pixel (0, 0) lands, in fluorescence px of the whole
    # sensor: with the matrix, the map for positions rather than displacements -
    # what lays a white-light image over the fluorescence one. None if unknown.
    offset: tuple = None

    def to_fluorescence_px(self, dx, dy):
        (a, b), (c, d) = self.matrix
        dx = np.asarray(dx, dtype=float)
        dy = np.asarray(dy, dtype=float)
        return a * dx + b * dy, c * dx + d * dy

    def image_affine(self, pixel_nm, sensor_origin_xy=(0.0, 0.0)):
        """3x3 homogeneous (row, column) map from white-light px to the viewer's nm.

        For an image read out of a cropped sensor: `sensor_origin_xy` is where
        the fluorescence image's first pixel sits on the sensor, so a position
        on the sensor becomes one in that image. Pixel centres at integers on
        both sides, as the calibration and the localizations have them.
        """
        if self.offset is None:
            raise ValueError(f"{self.source} holds displacements only, not positions")
        (a, b), (c, d) = self.matrix
        tx = float(self.offset[0]) - float(sensor_origin_xy[0])
        ty = float(self.offset[1]) - float(sensor_origin_xy[1])
        p = float(pixel_nm)
        return np.array([[p * d, p * c, p * ty],
                         [p * b, p * a, p * tx],
                         [0.0, 0.0, 1.0]])

    @property
    def scale(self):
        """Fluorescence px per white-light px, the same in every direction."""
        (a, b), (c, d) = self.matrix
        return float(np.sqrt(abs(a * d - b * c)))

    @property
    def rotation_deg(self):
        """How far the white-light axes are turned, seen on the fluorescence image."""
        (a, b), (c, d) = self.matrix
        return float(np.degrees(np.arctan2(c - b, a + d)))

    def rounded(self):
        """The matrix as plain rounded values, for metadata and fingerprints."""
        return [[round(float(v), 8) for v in row] for row in self.matrix]


# Measured on 2026-09-25 on an Argolight Argo-SIM v2 slide (the 21 x 21 field of
# rings at a 5 um pitch, seen by both cameras at once; recFL keeps the full
# calibration in calibrations/2026-09-25_argosim_xy_calibration.json). It is used
# for records that do not carry a map of their own - everything recorded before
# the acquisition started writing one - and holds while neither camera has
# moved, for the orientation and frame it was measured on.
CALIBRATED_CAMERA_MAP = CameraMap(
    matrix=((0.50257570, 0.00717823), (-0.00700369, 0.50164060)),
    source="the Argo-SIM v2 calibration of 2026-09-25",
    wl_orientation="Rotate 180",
    wl_frame_shape=(2048, 2048),
    offset=(91.67071274, 81.81136736),
)


def camera_map(record, calibrated=CALIBRATED_CAMERA_MAP):
    """(map, None) for the record, or (None, why not) when no map applies.

    The record's own map comes first: the acquisition writes, beside each drift
    record, the matrix it held at the time (xy_calibration.basler_to_kuro_px).
    Without one, `calibrated` is used if the record was taken with the same
    white-light orientation and the same frame, or a frame binned from it by a
    whole factor - binning scales white-light pixels, nothing else. A record that
    says it was taken otherwise gets no map at all: a map for another
    orientation would move every localization the wrong way, confidently.
    """
    meta = record.meta or {}
    own = meta.get("xy_calibration")
    if isinstance(own, dict) and own.get("basler_to_kuro_px") is not None:
        try:
            matrix = np.asarray(own["basler_to_kuro_px"], dtype=float)
            affine = np.asarray(own.get("basler_to_kuro_affine") or [], dtype=float)
            if matrix.shape == (2, 2) and np.all(np.isfinite(matrix)):
                # A record from before the acquisition wrote the affine form
                # still has the calibrated position: the cameras have not moved.
                offset = calibrated.offset
                if affine.shape == (2, 3) and np.all(np.isfinite(affine)):
                    matrix, offset = affine[:, :2], (float(affine[0, 2]), float(affine[1, 2]))
                taken = own.get("taken")
                source = "the drift record" + (
                    f" (calibration of {str(taken)[:10]})" if taken else "")
                return CameraMap(tuple(map(tuple, matrix.tolist())), source,
                                 own.get("wl_orientation"),
                                 tuple(own["basler_frame_shape"])
                                 if own.get("basler_frame_shape") else None,
                                 offset), None
        except (TypeError, ValueError):
            pass

    orientation = meta.get("wl_orientation")
    if (calibrated.wl_orientation and orientation
            and orientation != calibrated.wl_orientation):
        return None, (f"the record was taken with the white-light frames oriented "
                      f"'{orientation}', and {calibrated.source} holds for "
                      f"'{calibrated.wl_orientation}'")
    shape = meta.get("frame_shape")
    factor = 1
    if calibrated.wl_frame_shape and shape and len(shape) == 2:
        rows, cols = (int(v) for v in shape)
        full_rows, full_cols = calibrated.wl_frame_shape
        exact = (rows > 0 and cols > 0 and full_rows % rows == 0
                 and full_cols % cols == 0 and full_rows // rows == full_cols // cols)
        if not exact:
            return None, (f"the record was measured on {rows} x {cols} white-light "
                          f"frames, and {calibrated.source} holds for "
                          f"{full_rows} x {full_cols}")
        factor = full_rows // rows
    if factor == 1:
        return calibrated, None
    matrix = tuple(tuple(factor * v for v in row) for row in calibrated.matrix)
    offset = None
    if calibrated.offset is not None:
        # binned pixel i is centred on raw pixel factor * i + (factor - 1) / 2
        (a, b), (c, d) = calibrated.matrix
        half = (factor - 1) / 2.0
        offset = (calibrated.offset[0] + (a + b) * half, calibrated.offset[1] + (c + d) * half)
    return CameraMap(matrix, f"{calibrated.source}, for {factor}x binned frames",
                     calibrated.wl_orientation, tuple(shape), offset), None


def drift_per_frame(record, clock, n_raw, bin_factor=1, sigma_s=1.0):
    """The drift of every frame the pipeline sees, in white-light camera px.

    n_raw raw pages are timed by the clock; with time binning, each binned
    frame gets the mean drift of the raw pages summed into it, since what was
    localized in it is their sum. Leftover pages that do not fill a bin are
    dropped, as `bin_frames` drops them. Zero at the first frame.

    Returns a dict: dx, dy per frame; t, the mean time of each frame in epoch
    seconds; outside, per frame, whether any of its pages falls outside the
    drift record (the edge value was held there); n_extrapolated, the raw
    pages timed by extending the frame clock past its record; origin, the
    recorded drift the first frame sat at, which was subtracted to zero it.
    """
    n_raw = int(n_raw)
    factor = max(1, int(bin_factor))
    if n_raw // factor < 1:
        factor = 1                      # bin_frames leaves such a stack unbinned
    if n_raw < 1:
        raise ValueError("no frames to time")
    t_raw, n_extrapolated = clock.times(n_raw)
    bx, by = smoother(record.t, record.dx, record.dy, sigma_s)(t_raw)
    first, last = record.span
    outside = (t_raw < first) | (t_raw > last)

    n_frames = n_raw // factor
    used = n_frames * factor

    def per_frame(values, reduce):
        return reduce(np.asarray(values[:used]).reshape(n_frames, factor), axis=1)

    dx = per_frame(bx, np.mean)
    dy = per_frame(by, np.mean)
    return {
        "dx": dx - dx[0],
        "dy": dy - dy[0],
        "t": per_frame(t_raw, np.mean),
        "outside": per_frame(outside, np.any),
        "n_extrapolated": int(n_extrapolated),
        "bin_factor": factor,
        "origin": (float(dx[0]), float(dy[0])),
    }


def lookup(frames, per_frame_dx, per_frame_dy):
    """Per-row drift for rows on the given frames, and how many were off the end.

    A row whose frame lies past either end of the movie - a frame shift that is
    not right yet, say - is given the drift of the nearest frame rather than
    none at all, and counted so the caller can say so.
    """
    frames = np.asarray(frames)
    n = len(per_frame_dx)
    index = np.clip(frames, 0, n - 1).astype(np.int64)
    clipped = int(np.count_nonzero((frames < 0) | (frames > n - 1)))
    return np.asarray(per_frame_dx)[index], np.asarray(per_frame_dy)[index], clipped


def _prefix_of(path, suffix):
    name = Path(path).name
    return name[: -len(suffix)] if name.endswith(suffix) else Path(path).stem


def _common_prefix_length(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _best_match(candidates, suffix, stem):
    """The candidate whose name shares the longest start with `stem`."""
    if not stem:
        return None
    scored = sorted(((_common_prefix_length(_prefix_of(c, suffix), stem), c)
                     for c in candidates), key=lambda sc: -sc[0])
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None                       # a tie is not a match
    return scored[0][1] if scored and scored[0][0] > 0 else None


def _glob(folder, suffix):
    try:
        return sorted(p for p in Path(folder).glob("*" + suffix) if p.is_file())
    except OSError:
        return []


def find_drift_file(folder, stem=None):
    """The <name>_xy_drift.csv of an acquisition folder, or None.

    One is the usual case. With several, the one whose name matches the
    stack's is taken, and without a clear winner none is - applying another
    acquisition's drift would be worse than applying none.
    """
    candidates = _glob(folder, DRIFT_SUFFIX)
    if len(candidates) == 1:
        return candidates[0]
    return _best_match(candidates, DRIFT_SUFFIX, stem)


def find_frame_times(folder, stem=None, drift_path=None):
    """The frame-times file for a stack, or None.

    The acquisition names it after the stack, so that is tried first. A
    recovered '<stack>.partial.tif' is the same acquisition, and a table
    loaded on its own has no stack name at all - both fall back to the frame
    times sharing the drift record's acquisition name.
    """
    folder = Path(folder)
    if stem:
        for name in (stem, stem[: -len(".partial")] if stem.endswith(".partial") else None):
            if not name:
                continue
            exact = folder / (name + FRAME_TIMES_SUFFIX)
            try:
                if exact.is_file():
                    return exact
            except OSError:
                pass
    candidates = _glob(folder, FRAME_TIMES_SUFFIX)
    if drift_path is not None:
        acquisition = _prefix_of(drift_path, DRIFT_SUFFIX)
        candidates = [c for c in candidates if c.name.startswith(acquisition)]
    if len(candidates) == 1:
        return candidates[0]
    return _best_match(candidates, FRAME_TIMES_SUFFIX, stem)


class ShiftedStack:
    """A stack whose frames are each moved by their own offset, as they are read.

    For display: the image layer shows the movie with the drift taken out,
    while detection and fitting unwrap it (`base`) and work on the frames as
    the camera recorded them - the localizations are corrected afterwards, and
    fitting interpolated pixels would both blur the spots and correct the
    drift twice. Nothing is computed up front; napari reads one frame at a
    time, and each is shifted as it is read.

    shifts_yx[i] is how far the content of frame i moves, in pixels (row,
    column). Bilinear, and zero where no frame looked.

    With `expand`, the frames are drawn on a canvas wide enough for all of
    them: a sample that drifts by a quarter of the field shows, in the
    coordinates of the first frame, a quarter of a field that the first frame
    never saw - cut to the first frame's size, the display would lose it,
    progressively, as the drift grows. `origin_yx` is where the canvas starts in
    those coordinates (zero or negative, whole pixels); a viewer places the
    layer there. Without `expand`, the canvas is the frame, and what moves off
    it is cut.
    """

    def __init__(self, base, shifts_yx, expand=True):
        self.base = base
        self.expand = bool(expand)
        self.shifts_yx = shifts_yx

    @property
    def shifts_yx(self):
        return self._shifts

    @shifts_yx.setter
    def shifts_yx(self, shifts):
        self._shifts = np.asarray(shifts, dtype=float).reshape(-1, 2)
        frame = tuple(int(n) for n in self.base.shape[1:3])
        if not self.expand or not len(self._shifts):
            self._origin, self._canvas = (0, 0), frame
            return
        low = np.floor(np.minimum(self._shifts.min(axis=0), 0.0)).astype(int)
        high = np.ceil(np.maximum(self._shifts.max(axis=0), 0.0)).astype(int)
        self._origin = (int(low[0]), int(low[1]))
        self._canvas = (frame[0] + int(high[0] - low[0]), frame[1] + int(high[1] - low[1]))

    @property
    def origin_yx(self):
        """Where the canvas's first pixel sits, in the first frame's pixels."""
        return self._origin

    @property
    def shape(self):
        return (int(self.base.shape[0]),) + tuple(self._canvas)

    @property
    def dtype(self):
        return np.dtype(self.base.dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        out = self[:]
        return out if dtype is None else out.astype(dtype)

    def frame(self, index):
        plane = np.asarray(self.base[index])
        sy, sx = self._shifts[index] if index < len(self._shifts) else (0.0, 0.0)
        # where the frame's content goes on the canvas
        dy, dx = sy - self._origin[0], sx - self._origin[1]
        canvas = self._canvas
        if (canvas == plane.shape and abs(dy) < _NEGLIGIBLE_SHIFT_PX
                and abs(dx) < _NEGLIGIBLE_SHIFT_PX):
            return plane
        whole_y, whole_x = int(round(dy)), int(round(dx))
        if (abs(dy - whole_y) < _NEGLIGIBLE_SHIFT_PX and abs(dx - whole_x) < _NEGLIGIBLE_SHIFT_PX
                and 0 <= whole_y and 0 <= whole_x
                and whole_y + plane.shape[0] <= canvas[0]
                and whole_x + plane.shape[1] <= canvas[1]):
            # a whole-pixel move onto a canvas that holds all of it: a paste
            out = np.zeros(canvas, dtype=plane.dtype)
            out[whole_y:whole_y + plane.shape[0], whole_x:whole_x + plane.shape[1]] = plane
            return out
        from scipy import ndimage

        # output[o] = input[o - (dy, dx)]: the content moves by (dy, dx).
        # "grid-constant" interpolates across the frame's edge too, so a pixel
        # half covered by the frame is half as bright rather than black.
        moved = ndimage.affine_transform(
            plane.astype(np.float32, copy=False), np.eye(2), offset=(-dy, -dx),
            output_shape=canvas, order=1, mode="grid-constant", cval=0.0,
            prefilter=False)
        if plane.dtype.kind in "ui":
            info = np.iinfo(plane.dtype)
            moved = np.clip(np.rint(moved), info.min, info.max)
        return moved.astype(plane.dtype, copy=False)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis for k in key):
            at = next(i for i, k in enumerate(key) if k is Ellipsis)
            fill = (slice(None),) * max(self.ndim - (len(key) - 1), 0)
            key = key[:at] + fill + key[at + 1:]
        if not key:
            key = (slice(None),)
        first, rest = key[0], key[1:]
        if isinstance(first, (int, np.integer)):
            index = int(first)
            if index < 0:
                index += self.shape[0]
            if not 0 <= index < self.shape[0]:
                raise IndexError(f"frame {first} out of range for {self.shape[0]} frames")
            plane = self.frame(index)
            return plane[rest] if rest else plane
        indices = np.atleast_1d(np.arange(self.shape[0])[first])
        if indices.size:
            frames = np.stack([self.frame(int(i)) for i in indices])
        else:
            frames = np.empty((0,) + self.shape[1:], self.dtype)
        return frames[(slice(None),) + rest] if rest else frames


def unshifted(stack):
    """The frames as recorded, whether or not they are displayed drift-corrected."""
    return stack.base if isinstance(stack, ShiftedStack) else stack


class WarpedStack(ShiftedStack):
    """A stack whose frames are each resampled through a map of their own, as read.

    For the growth correction, which is not a shift: every frame drawn in the
    final geometry. `inverse(i, xy)` takes (N, 2) canvas positions (x, y, in
    the final geometry's pixels) to where they are in frame i; `forward(i, xy)`
    the other way, for whatever is drawn over the frames. The canvas starts at
    `origin_yx` and is `canvas` (rows, columns) big. Like ShiftedStack - which
    this is, to everything that asks for the frames as recorded - nothing is
    computed until a frame is shown.
    """

    def __init__(self, base, inverse, forward, origin_yx, canvas, key=None):
        self.base = base
        self.expand = True
        self._inverse = inverse
        self._forward = forward
        self._origin = (int(origin_yx[0]), int(origin_yx[1]))
        self._canvas = (int(canvas[0]), int(canvas[1]))
        self._shifts = np.zeros((int(base.shape[0]), 2))
        self.key = key
        self._cache = {}

    @property
    def shifts_yx(self):
        return self._shifts            # warped, not shifted: nothing to report here

    @shifts_yx.setter
    def shifts_yx(self, shifts):
        pass

    def frame(self, index):
        if index in self._cache:
            return self._cache[index]
        from scipy import ndimage

        plane = np.asarray(self.base[index])
        rows, cols = np.mgrid[0:self._canvas[0], 0:self._canvas[1]]
        xy = np.column_stack([cols.ravel() + self._origin[1],
                              rows.ravel() + self._origin[0]]).astype(float)
        src = self._inverse(index, xy)
        out = ndimage.map_coordinates(plane.astype(np.float32, copy=False), [src[:, 1], src[:, 0]],
                                      order=1, mode="constant", cval=0.0, prefilter=False)
        out = out.reshape(self._canvas)
        if plane.dtype.kind in "ui":
            info = np.iinfo(plane.dtype)
            out = np.clip(np.rint(out), info.min, info.max)
        out = out.astype(plane.dtype, copy=False)
        if len(self._cache) > 6:
            self._cache.pop(next(iter(self._cache)))
        self._cache[index] = out
        return out

    def displace(self, index, xy):
        """Where positions (N, 2) of frame `index` are drawn on the canvas."""
        return self._forward(int(index), np.atleast_2d(np.asarray(xy, dtype=float)))


# ----------------------------------------------------------------------------------
#  Checking the record: its own diagnostics, and the white-light snapshots
# ----------------------------------------------------------------------------------

def _drain(steps):
    """Run a generator that yields progress to the end; return what it returns.

    The long computations here yield their progress so a background worker can
    show it and stop them between steps; called directly, they just run.
    """
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value

def abrupt_steps(record, threshold_px=0.5):
    """Sample-to-sample jumps larger than `threshold_px`, as (t, dx, dy) rows.

    A sample drifts smoothly; a jump of a pixel or more between two samples a
    tenth of a second apart is either the sample really lurching - the stage, the
    focus lock, the root - or the tracker latching onto something else. Either
    way it is worth knowing where they are, because smoothing rounds a jump off
    rather than removing it.
    """
    good = record.good
    t, dx, dy = record.t[good], record.dx[good], record.dy[good]
    if len(t) < 2:
        return np.zeros((0, 3))
    jx, jy = np.diff(dx), np.diff(dy)
    big = np.hypot(jx, jy) > float(threshold_px)
    return np.column_stack([t[1:][big], jx[big], jy[big]])


def find_wl_images(drift_path):
    """(<acq>_WL_images.tif, its _times.csv) beside a drift record, or (None, None)."""
    drift_path = Path(drift_path)
    prefix = _prefix_of(drift_path, DRIFT_SUFFIX)
    tif = drift_path.with_name(prefix + WL_IMAGES_SUFFIX)
    times = drift_path.with_name(prefix + WL_TIMES_SUFFIX)
    try:
        if tif.is_file() and times.is_file():
            return tif, times
    except OSError:
        pass
    return None, None


def find_wl_images_for_stack(folder, stem=None):
    """The snapshots of the acquisition a stack belongs to, found by name alone:
    for an acquisition that kept snapshots but no drift record."""
    candidates = []
    for tif in _glob(folder, WL_IMAGES_SUFFIX):
        times = tif.with_name(_prefix_of(tif, WL_IMAGES_SUFFIX) + WL_TIMES_SUFFIX)
        if times.is_file():
            candidates.append(tif)
    chosen = (candidates[0] if len(candidates) == 1
              else _best_match(candidates, WL_IMAGES_SUFFIX, stem))
    if chosen is None:
        return None, None
    return chosen, chosen.with_name(_prefix_of(chosen, WL_IMAGES_SUFFIX) + WL_TIMES_SUFFIX)


class WLImageStack:
    """The white-light snapshots, read a page at a time.

    A long acquisition keeps hundreds of 2048 x 2048 snapshots - gigabytes - and
    the classic TIFF they go into rolls over into <stem>_001.tif and on at 4 GB,
    which the times file records page by page. Nothing is read until a frame is
    asked for, and only that frame.
    """

    def __init__(self, entries, t_epoch):
        self.entries = list(entries)            # (path, page) per snapshot
        self.t_epoch = np.asarray(t_epoch, dtype=float)
        import tifffile

        first_path, first_page = self.entries[0]
        with tifffile.TiffFile(first_path) as tif:
            page = tif.pages[first_page]
            self._frame_shape = tuple(int(n) for n in page.shape[:2])
            self._dtype = np.dtype(page.dtype)
        self._cache = {}

    @property
    def shape(self):
        return (len(self.entries),) + self._frame_shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return 3

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self.entries)
        if index not in self._cache:
            import tifffile

            path, page = self.entries[index]
            if len(self._cache) > 8:            # a handful of 8 MB frames at most
                self._cache.pop(next(iter(self._cache)))
            self._cache[index] = tifffile.imread(path, key=page)
        return self._cache[index]


def read_wl_images(tif_path, times_path):
    """The snapshots of an acquisition, lazily, in the order they were taken."""
    tif_path, times_path = Path(tif_path), Path(times_path)
    table = pd.read_csv(times_path)
    missing = {"t_epoch"} - set(table.columns)
    if missing:
        raise ValueError(f"{times_path.name} has no t_epoch column")
    files = (table["file"].astype(str) if "file" in table.columns
             else pd.Series([tif_path.name] * len(table)))
    pages = (pd.to_numeric(table["page"], errors="coerce").fillna(0).astype(int)
             if "page" in table.columns else pd.Series(np.arange(len(table))))
    entries = [(tif_path.with_name(name), int(page)) for name, page in zip(files, pages)]
    present = [i for i, (path, _page) in enumerate(entries) if path.is_file()]
    if not present:
        raise ValueError(f"none of the files {times_path.name} lists are there")
    t = pd.to_numeric(table["t_epoch"], errors="coerce").to_numpy(float)
    return WLImageStack([entries[i] for i in present], t[present])


def nearest_snapshot(snapshot_t, frame_t):
    """For each frame time, the index of the snapshot taken closest to it."""
    snapshot_t = np.asarray(snapshot_t, dtype=float)
    frame_t = np.asarray(frame_t, dtype=float)
    order = np.argsort(snapshot_t, kind="stable")
    sorted_t = snapshot_t[order]
    right = np.clip(np.searchsorted(sorted_t, frame_t), 1, len(sorted_t) - 1)
    left = right - 1
    if len(sorted_t) == 1:
        return np.zeros(len(frame_t), dtype=np.int64)
    pick = np.where(np.abs(frame_t - sorted_t[left]) <= np.abs(sorted_t[right] - frame_t),
                    left, right)
    return order[pick].astype(np.int64)


class FrameIndexedStack:
    """A stack seen on another stack's frame axis: frame i shows page index[i].

    The white-light snapshots come one every so many seconds; laid over the
    fluorescence movie, each of its frames shows the snapshot taken nearest to
    it, so scrolling the movie scrolls the white light with it. Nothing is
    copied - a frame is read when it is shown.
    """

    def __init__(self, base, index):
        self.base = base
        self.index = np.asarray(index, dtype=np.int64)

    @property
    def shape(self):
        return (len(self.index),) + tuple(int(n) for n in self.base.shape[1:])

    @property
    def dtype(self):
        return np.dtype(self.base.dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        out = self[:]
        return out if dtype is None else out.astype(dtype)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis for k in key):
            at = next(i for i, k in enumerate(key) if k is Ellipsis)
            fill = (slice(None),) * max(self.ndim - (len(key) - 1), 0)
            key = key[:at] + fill + key[at + 1:]
        if not key:
            key = (slice(None),)
        first, rest = key[0], key[1:]
        if isinstance(first, (int, np.integer)):
            plane = np.asarray(self.base[int(self.index[int(first)])])
            return plane[rest] if rest else plane
        pages = np.atleast_1d(self.index[first])
        frames = (np.stack([np.asarray(self.base[int(p)]) for p in pages]) if pages.size
                  else np.empty((0,) + self.shape[1:], self.dtype))
        return frames[(slice(None),) + rest] if rest else frames


def _hann(shape):
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1]))


def _dft_patch(spec_w, shape, ys, xs):
    """Real inverse DFT of an rfft2 half-spectrum at fractional lags ys x xs."""
    H, W = shape
    ey = np.exp(2j * np.pi * np.outer(ys, np.fft.fftfreq(H)))
    ex = np.exp(2j * np.pi * np.outer(np.arange(spec_w.shape[1]) / W, xs))
    return np.real(ey @ (spec_w @ ex)) / (H * W)


def register(reference, image, upsample=100):
    """How far the content of `image` moved relative to `reference`: (dx, dy, quality).

    The registration recFL's tracker uses - a Hann-windowed FFT
    cross-correlation, its peak refined on an upsampled grid by a matrix-multiply
    DFT (Guizar-Sicairos et al., Opt. Lett. 33, 156 (2008)) - so a check against
    it measures the same thing, independently. Quality is the normalized
    correlation peak, 1 for identical content.

    Meant for shifts under a pixel or so: the window weights the overlapping
    part of the two images unevenly, and pulls a shift of several pixels
    about 1% towards zero. Callers move their crop by whole pixels first and
    measure only what is left, as the tracker does.
    """
    ref = np.asarray(reference, dtype=np.float64)
    img = np.asarray(image, dtype=np.float64)
    if ref.shape != img.shape or ref.ndim != 2:
        raise ValueError("register needs two 2D images of the same shape")
    window = _hann(ref.shape)
    a = (ref - ref.mean()) * window
    b = (img - img.mean()) * window
    spec = np.fft.rfft2(a) * np.conj(np.fft.rfft2(b))
    cc = np.fft.irfft2(spec, s=a.shape)
    iy, ix = np.unravel_index(int(np.argmax(cc)), cc.shape)
    H, W = a.shape
    y = float(iy if iy <= H // 2 else iy - H)
    x = float(ix if ix <= W // 2 else ix - W)
    peak = float(cc[iy, ix])
    # rfft2 keeps only x-frequencies >= 0; every column counts twice except DC
    # and, for an even width, Nyquist.
    weights = np.full(spec.shape[1], 2.0)
    weights[0] = 1.0
    if W % 2 == 0:
        weights[-1] = 1.0
    spec_w = spec * weights
    upsample = max(1, int(upsample))
    passes = [(1.0 / min(upsample, 10), 1.0)]
    if upsample > 10:
        passes.append((1.0 / upsample, 0.1))
    if upsample > 1:
        for step, half in passes:
            n = int(round(half / step))
            offsets = np.arange(-n, n + 1) * step
            patch = _dft_patch(spec_w, a.shape, y + offsets, x + offsets)
            j, i = np.unravel_index(int(np.argmax(patch)), patch.shape)
            y += float(offsets[j])
            x += float(offsets[i])
            peak = float(patch[j, i])
    norm = float(np.sqrt((a * a).sum() * (b * b).sum()))
    # b[m] ~ a[m + lag]: the content moved by -lag.
    return -x, -y, (peak / norm if norm > 0 else 0.0)


def check_wl_images(*args, **kwargs):
    """check_wl_images_iter, run to the end."""
    return _drain(check_wl_images_iter(*args, **kwargs))


def check_wl_images_iter(stack, record, sigma_s=0.0, roi=None, upsample=100, cancel=None):
    """Measure every snapshot's drift against the first, independently of the record.

    Each snapshot is cropped where the record says the sample went, in whole
    pixels, and the crop registered against the first snapshot's: what is left
    is the record's error, measured on images the record never saw in that
    pairing. A record that is right lands every snapshot on the first to within
    the registration's own precision; a jump the tracker invented shows as a
    step between two snapshots that the images do not share.

    roi: (top, left, height, width) in snapshot pixels; by default the one the
    tracker used, from the record's metadata. Returns a dict of arrays, all in
    white-light pixels and relative to the first snapshot: t, record_dx/dy,
    measured_dx/dy, quality - or None if cancelled.
    """
    n = len(stack)
    H, W = stack.shape[1:3]
    if roi is None:
        roi = record.meta.get("roi_top_left_h_w")
    if roi is None or len(roi) != 4:
        h, w = min(H, 1024), min(W, 1024)
        roi = ((H - h) // 2, (W - w) // 2, h, w)
    top, left, h, w = (int(v) for v in roi)
    h, w = min(max(8, h), H), min(max(8, w), W)
    top, left = min(max(0, top), H - h), min(max(0, left), W - w)

    f = smoother(record.t, record.dx, record.dy, sigma_s)
    rx, ry = f(stack.t_epoch)
    rx0, ry0 = rx[0], ry[0]
    rx, ry = rx - rx0, ry - ry0
    reference = np.asarray(stack[0], dtype=np.float64)[top:top + h, left:left + w]
    measured = np.full((n, 2), np.nan)
    quality = np.full(n, np.nan)
    for k in range(n):
        if cancel is not None and cancel.is_set():
            return None
        cx, cy = int(round(rx[k])), int(round(ry[k]))
        t0 = min(max(0, top + cy), H - h)
        l0 = min(max(0, left + cx), W - w)
        crop = np.asarray(stack[k], dtype=np.float64)[t0:t0 + h, l0:l0 + w]
        sx, sy, q = register(reference, crop, upsample)
        measured[k] = (l0 - left + sx, t0 - top + sy)
        quality[k] = q
        yield (k + 1) / n
    return {"t": stack.t_epoch.copy(), "record_dx": rx, "record_dy": ry,
            "measured_dx": measured[:, 0], "measured_dy": measured[:, 1],
            "quality": quality, "roi": (top, left, h, w),
            # where the record stood at the first snapshot, before zeroing
            "origin": (float(rx0), float(ry0))}


# ----------------------------------------------------------------------------------
#  RCC: the drift the localizations themselves still show
# ----------------------------------------------------------------------------------

# Past this many pixels a side, the segment images are rendered coarser: every
# segment's spectrum is kept for the pairwise correlations, and at 2048 px a
# side twenty of them are already ~350 MB.
RCC_MAX_IMAGE_PX = 2048


def _subpixel(values, index):
    """Peak position along one axis from three samples: Gaussian, else parabolic."""
    left, centre, right = values[index - 1], values[index], values[index + 1]
    if left > 0 and centre > 0 and right > 0:
        ll, lc, lr = np.log(left), np.log(centre), np.log(right)
        denominator = ll - 2.0 * lc + lr
        if denominator < 0:
            return index + 0.5 * (ll - lr) / denominator
    denominator = left - 2.0 * centre + right
    if denominator < 0:
        return index + 0.5 * (left - right) / denominator
    return float(index)


def _rcc_design(pairs, n_segments):
    """Rows of r_ij = d_j - d_i, with d_0 fixed at zero (so its column is dropped)."""
    A = np.zeros((len(pairs), n_segments))
    rows = np.arange(len(pairs))
    A[rows, pairs[:, 0]] = -1.0
    A[rows, pairs[:, 1]] = 1.0
    return A[:, 1:]


def _solve_rcc(A, shifts, keep):
    """Least-squares segment drifts from the kept pairs, and every pair's error."""
    solution = np.linalg.lstsq(A[keep], shifts[keep], rcond=None)[0]
    drift = np.vstack([np.zeros((1, 2)), solution])
    errors = np.hypot(*(A @ solution - shifts).T)
    return drift, errors


def rcc(*args, **kwargs):
    """rcc_iter, run to the end."""
    return _drain(rcc_iter(*args, **kwargs))


def rcc_iter(x_nm, y_nm, frames, n_frames, segment_frames=500, pixel_nm=40.0,
             blur_nm=40.0, max_shift_nm=500.0, rmax_nm=30.0, cancel=None):
    """Redundant cross-correlation drift estimate (Wang et al., Opt. Express 22,
    15982 (2014)).

    The movie is cut into segments of `segment_frames` frames; each is rendered
    into an image of `pixel_nm` pixels, blurred by `blur_nm`, and every pair of
    segments is cross-correlated - not only each against the first. That gives
    n(n-1)/2 measured shifts for n-1 unknown drifts: solved by least squares, the
    redundancy averages the noise down, and any pair whose shift disagrees with
    the solution by more than `rmax_nm` (a correlation that latched onto the
    wrong peak) is dropped and the rest solved again, one pair at a time, never
    dropping one that would leave a segment unconnected. Peaks are only searched
    for within `max_shift_nm` of zero.

    Returns a dict: segment centres (mean frame of their localizations), dx/dy
    per segment (nm, the drift of each segment relative to the first), the pairs
    with their shifts, errors and which were kept, the rms error of the kept
    pairs, and the pixel size actually used. None if cancelled.
    """
    from scipy import fft as sfft
    from scipy import ndimage

    x_nm = np.asarray(x_nm, dtype=float)
    y_nm = np.asarray(y_nm, dtype=float)
    frames = np.asarray(frames)
    ok = np.isfinite(x_nm) & np.isfinite(y_nm) & np.isfinite(frames)
    x_nm, y_nm, frames = x_nm[ok], y_nm[ok], frames[ok].astype(np.int64)
    n_frames = max(int(n_frames), int(frames.max()) + 1 if frames.size else 1)
    n_segments = max(2, int(round(n_frames / max(1, int(segment_frames)))))
    edges = np.linspace(0, n_frames, n_segments + 1)
    segment = np.clip(np.searchsorted(edges, frames, side="right") - 1, 0, n_segments - 1)
    counts = np.bincount(segment, minlength=n_segments)
    used = np.flatnonzero(counts > 0)
    if len(used) < 2:
        raise ValueError("RCC needs localizations in at least two segments")

    x0, y0 = float(x_nm.min()), float(y_nm.min())
    extent = max(float(x_nm.max()) - x0, float(y_nm.max()) - y0, 1.0)
    pixel = max(float(pixel_nm), extent / RCC_MAX_IMAGE_PX)
    shape = (int((float(y_nm.max()) - y0) / pixel) + 1, int((float(x_nm.max()) - x0) / pixel) + 1)
    reach = max(1, int(np.ceil(float(max_shift_nm) / pixel)))
    reach = min(reach, shape[0] // 2 - 2, shape[1] // 2 - 2)
    if reach < 1:
        raise ValueError("the localizations cover too small a field for RCC")
    blur = max(float(blur_nm) / pixel, 0.0)

    spectra = []
    total_steps = len(used) + len(used) * (len(used) - 1) // 2
    step = 0
    for s in used:
        if cancel is not None and cancel.is_set():
            return None
        mine = segment == s
        image, _, _ = np.histogram2d(
            y_nm[mine], x_nm[mine], bins=shape,
            range=((y0, y0 + shape[0] * pixel), (x0, x0 + shape[1] * pixel)))
        image = image.astype(np.float32)
        if blur > 0:
            image = ndimage.gaussian_filter(image, blur)
        image -= image.mean()
        spectra.append(sfft.rfft2(image, workers=-1))
        step += 1
        yield step / total_steps

    pairs, shifts = [], []
    cy, cx = shape[0] // 2, shape[1] // 2
    for a in range(len(used)):
        for b in range(a + 1, len(used)):
            if cancel is not None and cancel.is_set():
                return None
            cc = sfft.irfft2(spectra[a] * np.conj(spectra[b]), s=shape, workers=-1)
            cc = np.fft.fftshift(cc)
            window = cc[cy - reach - 1:cy + reach + 2, cx - reach - 1:cx + reach + 2]
            inner = window[1:-1, 1:-1]
            iy, ix = np.unravel_index(int(np.argmax(inner)), inner.shape)
            iy, ix = iy + 1, ix + 1
            # A peak on the edge of the search window is only the highest
            # point inside it, not a peak: the shift is out of reach.
            if 1 <= iy - 1 and iy + 1 < window.shape[0] - 1 and 1 <= ix - 1 and ix + 1 < window.shape[1] - 1:
                py = _subpixel(window[:, ix], iy) - (reach + 1)
                px = _subpixel(window[iy, :], ix) - (reach + 1)
                # the correlation peaks at the lag that undoes b's shift
                pairs.append((a, b))
                shifts.append((-px * pixel, -py * pixel))
            step += 1
            yield step / total_steps

    n_used = len(used)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    shifts = np.asarray(shifts, dtype=float).reshape(-1, 2)
    keep = np.ones(len(pairs), dtype=bool)
    A = _rcc_design(pairs, n_used)

    def connected(mask):
        # Every segment still tied to the first through some chain of pairs.
        return bool(mask.any()) and np.linalg.matrix_rank(A[mask]) == n_used - 1

    if not connected(keep):
        raise ValueError("too few segment pairs correlated within the search range - "
                         "raise the maximum shift, or use longer segments")
    drift, errors = _solve_rcc(A, shifts, keep)
    while True:
        worst = [i for i in np.argsort(-errors) if keep[i] and errors[i] > rmax_nm]
        dropped = False
        for i in worst:
            trial = keep.copy()
            trial[i] = False
            if connected(trial):
                keep = trial
                dropped = True
                break
        if not dropped:
            break
        drift, errors = _solve_rcc(A, shifts, keep)

    centres = np.array([frames[segment == s].mean() for s in used])
    kept_errors = errors[keep]
    return {
        "centres": centres, "dx": drift[:, 0], "dy": drift[:, 1],
        "n_localizations": counts[used], "pairs": pairs, "shifts": shifts,
        "errors": errors, "kept": keep,
        "rms_error_nm": float(np.sqrt(np.mean(kept_errors ** 2))) if kept_errors.size else 0.0,
        "pixel_nm": pixel, "n_frames": n_frames,
    }


def rcc_per_frame(result, n_frames):
    """The RCC drift of every frame, zero at the first.

    Straight lines between segment centres, and continued past the first and
    last centre along the end segments' slopes rather than held: a drift still
    going at the end of the movie keeps going, and holding it flat for half a
    segment at each end would leave exactly that much uncorrected.
    """
    centres = np.asarray(result["centres"], dtype=float)
    frames = np.arange(int(n_frames), dtype=float)
    out = []
    for values in (np.asarray(result["dx"], float), np.asarray(result["dy"], float)):
        if len(centres) == 1:
            curve = np.full(len(frames), values[0])
        else:
            curve = np.interp(frames, centres, values)
            first = (values[1] - values[0]) / max(centres[1] - centres[0], 1e-9)
            last = (values[-1] - values[-2]) / max(centres[-1] - centres[-2], 1e-9)
            before, after = frames < centres[0], frames > centres[-1]
            curve[before] = values[0] + first * (frames[before] - centres[0])
            curve[after] = values[-1] + last * (frames[after] - centres[-1])
        out.append(curve - curve[0])
    return out[0], out[1]
