"""Checking the drift record, refining it with RCC, and merging immobile molecules.

Three things built on the white-light drift correction:

* the record checked against the white-light snapshots saved with it - an
  independent registration, so a jump the tracker invented shows up as a step
  the images do not share;
* RCC, which measures from the localizations themselves the drift the
  white-light correction left, and adds it on top - but only for as long as the
  correction it refined is the one in force;
* merging each confidently immobile trajectory into one localization at its
  combined precision - where "confidently" means the static test could have
  seen motion down to a chosen D, which a two-point trajectory never can.
"""
import importlib
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

conftest = importlib.import_module("conftest")
drift = conftest.load_drift()

from test_drift import (  # noqa: E402
    N, PERIOD, PIXEL_NM, RAMP, T0, WL_NM, _acquisition, _write_drift,
)


def _texture(size=128, seed=0):
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.normal(0, 1, (size, size)), 2.0) * 1000 + 5000


def _moved(image, dx, dy):
    """The content of `image` moved by (dx, dy) px, exactly (Fourier shift)."""
    from scipy import ndimage

    spectrum = ndimage.fourier_shift(np.fft.fft2(image), (dy, dx))
    return np.real(np.fft.ifft2(spectrum))


# --- registration and the snapshots ------------------------------------------------


def test_registration_recovers_a_subpixel_shift():
    """Sub-pixel, because that is all it is ever asked for: the check moves its
    crop by the record's whole pixels first, as the tracker does, and the
    window would pull a shift of several pixels slightly towards zero.

    Cropped from a larger field, as a camera sees it: the Fourier shift wraps
    content round the edges, which a real frame never does.
    """
    base = _texture(size=192)
    inner = (slice(32, 160), slice(32, 160))
    dx, dy, quality = drift.register(base[inner], _moved(base, 0.37, -0.62)[inner])
    assert dx == pytest.approx(0.37, abs=0.01)
    assert dy == pytest.approx(-0.62, abs=0.01)
    assert quality > 0.8


def _snapshots(folder, times, true_drift, stem="acq_Normal"):
    """WL snapshots as the acquisition writes them: TIFF pages + times CSV."""
    import tifffile

    base = _texture(seed=3)
    with tifffile.TiffWriter(folder / f"{stem}_WL_images.tif") as tif:
        for t in times:
            dx, dy = true_drift(t)
            tif.write(np.clip(_moved(base, dx, dy), 0, 65535).astype(np.uint16))
    pd.DataFrame({"index": range(len(times)), "file": f"{stem}_WL_images.tif",
                  "page": range(len(times)), "t_epoch": times}).to_csv(
        folder / f"{stem}_WL_images_times.csv", index=False)
    return drift.find_wl_images(folder / f"{stem}_xy_drift.csv")


def _ramp(t):
    return RAMP[0] * (t - T0), RAMP[1] * (t - T0)


def test_a_right_record_lands_every_snapshot_on_the_first(tmp_path):
    _acquisition(tmp_path)
    times = T0 + np.array([-0.5, 0.3, 1.1, 1.9])
    tif, csv = _snapshots(tmp_path, times, _ramp)
    stack = drift.read_wl_images(tif, csv)
    record = drift.read_drift(tmp_path / "acq_Normal_xy_drift.csv")
    check = drift.check_wl_images(stack, record)
    np.testing.assert_allclose(check["measured_dx"], check["record_dx"], atol=0.05)
    np.testing.assert_allclose(check["measured_dy"], check["record_dy"], atol=0.05)


def test_a_jump_the_tracker_invented_shows_against_the_snapshots(tmp_path):
    """The record says the sample jumped two pixels; the images say it did not."""
    _acquisition(tmp_path)
    t = np.arange(T0 - 1.0, T0 + 3.0, 0.05)
    invented = np.where(t > T0 + 0.7, 2.0, 0.0)
    _write_drift(tmp_path / "acq_Normal_xy_drift.csv", t, RAMP[0] * (t - T0),
                 RAMP[1] * (t - T0) + invented)
    times = T0 + np.array([-0.5, 0.3, 1.1, 1.9])
    tif, csv = _snapshots(tmp_path, times, _ramp)
    record = drift.read_drift(tmp_path / "acq_Normal_xy_drift.csv")
    check = drift.check_wl_images(drift.read_wl_images(tif, csv), record)
    off = check["measured_dy"] - check["record_dy"]
    np.testing.assert_allclose(off[:2], 0.0, atol=0.05)
    np.testing.assert_allclose(off[2:], -2.0, atol=0.05)
    steps = drift.abrupt_steps(record, 0.5)
    # the invented two pixels, on top of one sample's worth of the ramp
    assert len(steps) == 1 and steps[0, 2] == pytest.approx(2.0, abs=0.1)


def test_snapshots_are_read_a_page_at_a_time_across_rollover_files(tmp_path):
    import tifffile

    frames = [np.full((8, 8), v, np.uint16) for v in (1, 2, 3)]
    tifffile.imwrite(tmp_path / "a_WL_images.tif", np.stack(frames[:2]))
    tifffile.imwrite(tmp_path / "a_WL_images_001.tif", frames[2])
    pd.DataFrame({"index": [0, 1, 2],
                  "file": ["a_WL_images.tif", "a_WL_images.tif", "a_WL_images_001.tif"],
                  "page": [0, 1, 0], "t_epoch": [1.0, 2.0, 3.0]}).to_csv(
        tmp_path / "a_WL_images_times.csv", index=False)
    stack = drift.read_wl_images(tmp_path / "a_WL_images.tif", tmp_path / "a_WL_images_times.csv")
    assert stack.shape == (3, 8, 8)
    assert [int(np.asarray(stack[k])[0, 0]) for k in range(3)] == [1, 2, 3]


# --- RCC -----------------------------------------------------------------------------


def _static_scene(n_emitters=400, n_frames=200, drift_per_frame=(0.5, -0.3),
                  noise_nm=10.0, detect=0.3, seed=0):
    """Emitters that never move, seen through a steadily drifting stage."""
    rng = np.random.default_rng(seed)
    where = rng.uniform(1000, 11000, (n_emitters, 2))
    rows = []
    for frame in range(n_frames):
        seen = rng.random(n_emitters) < detect
        xy = where[seen] + rng.normal(0, noise_nm, (seen.sum(), 2))
        xy += np.array(drift_per_frame) * frame
        rows.append(np.column_stack([np.full(seen.sum(), frame), xy]))
    table = np.vstack(rows)
    return table[:, 1], table[:, 2], table[:, 0].astype(int)


def test_rcc_recovers_a_known_drift():
    x, y, frames = _static_scene()
    result = drift.rcc(x, y, frames, 200, segment_frames=40, pixel_nm=10.0,
                       blur_nm=10.0, max_shift_nm=300.0, rmax_nm=20.0)
    assert len(result["centres"]) == 5
    assert result["kept"].all()
    px, py = drift.rcc_per_frame(result, 200)
    frames_checked = np.array([0, 50, 100, 150, 199])
    np.testing.assert_allclose(px[frames_checked], 0.5 * frames_checked, atol=3.0)
    np.testing.assert_allclose(py[frames_checked], -0.3 * frames_checked, atol=3.0)


def test_the_rcc_drift_continues_past_the_first_and_last_segment():
    """Held flat, the half segment at each end would keep its drift."""
    result = {"centres": np.array([10.0, 30.0]), "dx": np.array([0.0, 20.0]),
              "dy": np.array([0.0, 0.0])}
    px, _py = drift.rcc_per_frame(result, 41)
    assert px[0] == 0.0
    np.testing.assert_allclose(px[40] - px[0], 40.0)


def test_rcc_refuses_segments_it_cannot_tie_together():
    x, y, frames = _static_scene(n_frames=40)
    with pytest.raises(ValueError):
        # a search range too small for the drift between segments
        drift.rcc(x + 500.0 * (frames >= 20), y, frames, 40, segment_frames=20,
                  pixel_nm=10.0, blur_nm=10.0, max_shift_nm=50.0, rmax_nm=20.0)


# --- merging -------------------------------------------------------------------------

from napari_loc_track import _tracks as tracks_mod  # noqa: E402


def _trajectory_table(scatter_nm, sigma_nm=20.0, n=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "frame": np.arange(n), "x [nm]": 1000 + rng.normal(0, scatter_nm, n),
        "y [nm]": 2000 + rng.normal(0, scatter_nm, n),
        "uncertainty [nm]": np.full(n, sigma_nm), "intensity [photon]": np.full(n, 300.0),
    })


def _merge(table, ids=(0,)):
    return tracks_mod.merge_trajectories(
        table, np.zeros(len(table), int), set(ids), x_col="x [nm]", y_col="y [nm]",
        frame_col="frame", sigma_col="uncertainty [nm]",
        sum_columns=("intensity [photon]",))


def test_a_static_trajectory_becomes_one_localization_sqrt_n_more_precise():
    table = _trajectory_table(scatter_nm=20.0, sigma_nm=20.0, n=16)
    merged = _merge(table)
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["x [nm]"] == pytest.approx(table["x [nm]"].mean())
    assert row["uncertainty [nm]"] == pytest.approx(20.0 / 4.0, rel=0.35)
    assert row["uncertainty [nm]"] >= 20.0 / 4.0 - 1e-9     # never below the propagation
    assert row["intensity [photon]"] == pytest.approx(16 * 300.0)
    assert row["n_merged"] == 16
    assert (row["frame"], row["frame_last"]) == (0, 15)


def test_a_trajectory_that_scattered_more_than_its_precision_is_drawn_wider():
    """Motion too small to detect, or optimistic precisions, and the merged
    point says so rather than claiming the propagated precision."""
    tight = _merge(_trajectory_table(scatter_nm=20.0, n=16, seed=1)).iloc[0]
    loose = _merge(_trajectory_table(scatter_nm=80.0, n=16, seed=1)).iloc[0]
    assert loose["uncertainty [nm]"] > 3 * tight["uncertainty [nm]"]


def test_localizations_of_other_trajectories_are_left_as_they_were():
    table = pd.concat([_trajectory_table(20.0, n=5), _trajectory_table(20.0, n=5, seed=2)],
                      ignore_index=True)
    particles = np.r_[np.zeros(5, int), np.ones(5, int)]
    merged = tracks_mod.merge_trajectories(
        table, particles, {0}, x_col="x [nm]", y_col="y [nm]", frame_col="frame",
        sigma_col="uncertainty [nm]")
    assert len(merged) == 6
    assert (merged["n_merged"] == 1).sum() == 5
    np.testing.assert_allclose(np.sort(merged.loc[merged.particle == 1, "x [nm]"]),
                               np.sort(table["x [nm]"][5:]))


# --- through the widget ---------------------------------------------------------------

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_drift import _loaded  # noqa: E402
from test_render_widget import _pump_until  # noqa: E402


def test_the_frame_clock_sets_the_frame_rate_when_nothing_else_recorded_it(tmp_path):
    widget = _loaded(tmp_path)
    assert widget.fps_box.value() == pytest.approx(1.0 / PERIOD, rel=1e-4)
    assert "frame clock" in widget.log_box.toPlainText()


def test_a_recorded_frame_interval_beats_the_frame_clock(tmp_path):
    widget = _loaded(tmp_path)
    acquisition = {"values": {"frame_interval_ms": 31.34, "fps": 1000 / 31.34},
                   "sources": {"frame_interval_ms": "sidecar", "fps": "sidecar"}}
    assert widget._with_frame_clock(acquisition) is acquisition


def _drifting_static_table(wl_true_nm=WL_NM, n_emitters=300, seed=0):
    """Static emitters seen through the drift the test acquisition recorded."""
    rng = np.random.default_rng(seed)
    where = rng.uniform(1000, 9000, (n_emitters, 2))
    rows = []
    for frame in range(N):
        xy = where + rng.normal(0, 5.0, where.shape)
        xy += np.array(RAMP) * PERIOD * frame * wl_true_nm
        rows.append(pd.DataFrame({"frame": frame, "x [nm]": xy[:, 0], "y [nm]": xy[:, 1],
                                  "intensity [photon]": 500.0}))
    return pd.concat(rows, ignore_index=True), where


def _rcc_ready(widget):
    widget.rcc_segment_box.setValue(4)
    widget.rcc_pixel_box.setValue(10.0)
    widget.rcc_blur_box.setValue(10.0)
    widget.rcc_max_shift_box.setValue(200.0)
    widget.rcc_rmax_box.setValue(20.0)


def test_rcc_measures_what_a_wrong_camera_map_left_and_takes_it_out(tmp_path):
    """A map 20% short leaves a fifth of the drift in; RCC finds it."""
    table, _where = _drifting_static_table()
    widget = _loaded(tmp_path, table=table, wl_nm=0.8 * WL_NM)
    widget.apply_filters()
    _rcc_ready(widget)
    widget.estimate_rcc()
    assert _pump_until(lambda: widget._rcc_worker_ref is None), "RCC never finished"
    assert widget._applied_drift[2] == "record+rcc"

    left_x, left_y = widget._rcc_per_frame_nm()
    frames = np.arange(N)
    np.testing.assert_allclose(left_x, 0.2 * RAMP[0] * PERIOD * WL_NM * frames, atol=3.0)
    np.testing.assert_allclose(left_y, 0.2 * RAMP[1] * PERIOD * WL_NM * frames, atol=3.0)

    # every frame now lands on the first
    by_frame = widget.df.groupby("frame")[["x [nm]", "y [nm]"]].mean()
    np.testing.assert_allclose(by_frame["x [nm]"] - by_frame["x [nm]"].iloc[0], 0.0, atol=3.0)
    np.testing.assert_allclose(by_frame["y [nm]"] - by_frame["y [nm]"].iloc[0], 0.0, atol=3.0)

    # the export splits the correction into its two parts
    per_frame = dict(widget._export_tables())["drift_per_frame.csv"]
    np.testing.assert_allclose(
        per_frame["drift_x [nm]"],
        per_frame["white_light_drift_x [nm]"] + per_frame["rcc_drift_x [nm]"])


def test_an_rcc_estimate_stops_applying_when_the_correction_it_refined_changes(tmp_path):
    table, _where = _drifting_static_table()
    widget = _loaded(tmp_path, table=table, wl_nm=0.8 * WL_NM)
    widget.apply_filters()
    _rcc_ready(widget)
    widget.estimate_rcc()
    assert _pump_until(lambda: widget._rcc_worker_ref is None)
    # the pixel size carries the record into nm, so it is part of the correction
    widget.pixel_size_box.setValue(1.25 * PIXEL_NM)
    widget._refresh_drift_correction()
    assert widget._applied_drift[2] == "record"
    assert "Stale" in widget.rcc_status.text()
    widget.pixel_size_box.setValue(PIXEL_NM)
    widget._refresh_drift_correction()
    assert widget._applied_drift[2] == "record+rcc"


def test_estimating_again_does_not_estimate_on_top_of_itself(tmp_path):
    """Measured on positions it had already corrected, a second estimate would
    find nothing left and replace the first with zero."""
    table, _where = _drifting_static_table()
    widget = _loaded(tmp_path, table=table, wl_nm=0.8 * WL_NM)
    widget.apply_filters()
    _rcc_ready(widget)
    widget.estimate_rcc()
    assert _pump_until(lambda: widget._rcc_worker_ref is None)
    first = widget._rcc_per_frame_nm()[0].copy()
    widget.estimate_rcc()
    assert _pump_until(lambda: widget._rcc_worker_ref is None)
    np.testing.assert_allclose(widget._rcc_per_frame_nm()[0], first, atol=1.0)


def test_an_rcc_estimate_travels_in_the_metadata(tmp_path):
    table, _where = _drifting_static_table()
    widget = _loaded(tmp_path, table=table, wl_nm=0.8 * WL_NM)
    widget.apply_filters()
    _rcc_ready(widget)
    widget.estimate_rcc()
    assert _pump_until(lambda: widget._rcc_worker_ref is None)
    section = widget._collect_metadata(None)["drift_correction"]["rcc"]
    back = widget_mod.LocalizationTrackingWidget._rcc_from_metadata(section)
    np.testing.assert_allclose(back["result"]["dx"], widget._rcc["result"]["dx"])
    assert back["fingerprint"] == widget._rcc["fingerprint"]
    values, _notes = widget_mod.settings_from_metadata({"drift_correction": {"rcc": section}})
    assert values["rcc_segment_box"] == 4


def test_a_session_brings_its_rcc_estimate_back(tmp_path):
    """Re-running RCC on every restore would cost seconds and could land
    elsewhere; the estimate is a small table, and comes back as it was."""
    table, _where = _drifting_static_table()
    widget = _loaded(tmp_path, table=table, wl_nm=0.8 * WL_NM)
    widget.apply_filters()
    _rcc_ready(widget)
    widget.estimate_rcc()
    assert _pump_until(lambda: widget._rcc_worker_ref is None)
    manifest = {"settings": widget._collect_metadata(None), "sources": {}}
    expected = widget._rcc_per_frame_nm()[0].copy()

    other = _loaded(tmp_path, table=table, wl_nm=0.8 * WL_NM)
    other._session_restore = {"manifest": manifest, "session_dir": tmp_path}
    try:
        other._find_drift_record()
        other._ingest_localization_dataframe(table.copy(), "restored", frame_is_zero_indexed=True)
    finally:
        other._session_restore = None
    assert other._applied_drift[2] == "record+rcc"
    np.testing.assert_allclose(other._rcc_per_frame_nm()[0], expected)


def test_new_localizations_drop_the_last_rcc_estimate(tmp_path):
    table, _where = _drifting_static_table()
    widget = _loaded(tmp_path, table=table, wl_nm=0.8 * WL_NM)
    widget.apply_filters()
    _rcc_ready(widget)
    widget.estimate_rcc()
    assert _pump_until(lambda: widget._rcc_worker_ref is None)
    widget._ingest_localization_dataframe(table.copy(), "new", frame_is_zero_indexed=True)
    assert widget._rcc is None


def test_the_snapshots_are_checked_and_shown_drift_removed(tmp_path, monkeypatch):
    _acquisition(tmp_path)
    times = T0 + np.array([-0.5, 0.3, 1.1, 1.9])
    _snapshots(tmp_path, times, _ramp)
    widget = _loaded(tmp_path)
    assert widget._wl_stack is not None and len(widget._wl_stack) == 4
    widget.check_wl_snapshots()
    assert _pump_until(lambda: widget._wl_check_worker_ref is None)
    summary = widget._wl_check_summary()
    assert max(np.abs(summary["differences_y_px"])) < 0.05
    assert "registered independently" in widget.wl_check_status.text()

    shown = []

    class _Viewer:
        def __init__(self, **_kw):
            pass

        def add_image(self, data, name=None, **kw):
            shown.append((name, data))

        def add_shapes(self, *_a, **_kw):
            pass

    monkeypatch.setattr(widget_mod.napari, "Viewer", _Viewer)
    widget.show_wl_snapshots()
    fixed = dict(shown)["snapshots, drift removed"]
    first = np.asarray(fixed[0], float)
    last = np.asarray(fixed[len(times) - 1], float)
    inner = (slice(10, -10), slice(10, -10))
    # the drift out, the sample stands still
    assert np.corrcoef(first[inner].ravel(), last[inner].ravel())[0, 1] > 0.98
