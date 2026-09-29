"""Taking the sample's drift, measured on the white-light camera, out of the data.

The acquisition writes two records beside the stack: <name>_xy_drift.csv, the
sample's displacement in white-light pixels against time, and
<stack>_frame_times.csv, when each page of the stack came off the camera.
Between them every frame gets the drift of its own moment, and every
localization is moved back by the drift of its frame.

What has to hold: the drift is zero at the first frame and follows the record
exactly on a steady ramp, ends included; a binned frame gets the mean drift of
the raw frames summed into it; the correction can be taken off again and
never lands twice on the same table; and the image is only shifted for display
- detection and fitting still see the frames as recorded.
"""
import importlib
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

conftest = importlib.import_module("conftest")
drift = conftest.load_drift()

# A steady ramp, in white-light px per second: local-linear smoothing
# reproduces a straight line exactly, so every expected value below is exact.
RAMP = (2.0, -1.0)
T0 = 1000.0           # mid-exposure of page 0
PERIOD = 0.1          # s per page
READOUT = 0.01
EXPOSURE = 0.1


def _write_drift(path, t, dx, dy, ok=None, meta=None):
    meta = {"kind": "xy_drift", "version": 1, "wl_orientation": "Rotate 180",
            **(meta or {})}
    ok = np.ones(len(t), dtype=int) if ok is None else ok
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("# " + json.dumps(meta) + "\n")
        fh.write("t_epoch,dx_px,dy_px,quality,ok,ref_id\n")
        for row in zip(t, dx, dy, ok):
            fh.write(f"{row[0]:.4f},{row[1]:.4f},{row[2]:.4f},0.9,{int(row[3])},0\n")


def _write_frame_times(path, n_pages, jitter=None):
    """Pages popped `READOUT` + `EXPOSURE` after their mid-exposure time."""
    meta = {"kind": "frame_times", "mode": "normal", "exposure_span_s": EXPOSURE,
            "readout_s": READOUT}
    pages = np.arange(n_pages)
    t = T0 + PERIOD * pages + READOUT + EXPOSURE / 2
    if jitter is not None:
        t = t + jitter
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("# " + json.dumps(meta) + "\n")
        fh.write("page,t_epoch,backlog\n")
        for page, when in zip(pages, t):
            fh.write(f"{page},{when:.6f},1\n")


def _acquisition(folder, n_pages=20, stem="acq_Normal_normal", drift_stem="acq_Normal",
                 drift_meta=None):
    """A folder as the acquisition leaves it: drift record and frame times."""
    t = np.arange(T0 - 1.0, T0 + PERIOD * n_pages + 1.0, 0.05)
    _write_drift(folder / f"{drift_stem}_xy_drift.csv", t,
                 RAMP[0] * (t - T0), RAMP[1] * (t - T0), meta=drift_meta)
    _write_frame_times(folder / f"{stem}_frame_times.csv", n_pages)
    return folder / f"{stem}.tif"


# --- the records ---------------------------------------------------------------


def test_untrusted_drift_samples_are_dropped_not_believed(tmp_path):
    t = np.arange(10.0)
    ok = np.ones(10, dtype=int)
    ok[4] = 0
    _write_drift(tmp_path / "a_xy_drift.csv", t, t, -t, ok=ok)
    record = drift.read_drift(tmp_path / "a_xy_drift.csv")
    assert np.isnan(record.dx[4]) and np.isnan(record.dy[4])
    assert int(record.good.sum()) == 9
    assert record.meta["wl_orientation"] == "Rotate 180"


def test_another_kind_of_record_is_refused(tmp_path):
    path = tmp_path / "a_xy_drift.csv"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('# {"kind": "frame_times"}\npage,t_epoch,backlog\n0,1.0,1\n')
    with pytest.raises(ValueError):
        drift.read_drift(path)


def test_frame_times_are_moved_to_mid_exposure(tmp_path):
    """The file says when a page left the buffer - after its readout, which
    is after its exposure. The drift that moved the molecules is the drift
    in the middle of the exposure."""
    _write_frame_times(tmp_path / "s_frame_times.csv", 30)
    clock = drift.read_frame_times(tmp_path / "s_frame_times.csv")
    t, n_extra = clock.times(30)
    assert n_extra == 0
    np.testing.assert_allclose(t, T0 + PERIOD * np.arange(30), atol=1e-6)


def test_polling_jitter_is_replaced_by_the_frame_clock(tmp_path):
    """Each timestamp is when a polling loop got round to the frame; the
    frames themselves came at the camera's steady rate."""
    rng = np.random.default_rng(1)
    _write_frame_times(tmp_path / "s_frame_times.csv", 200,
                       jitter=rng.uniform(0.0, 0.002, 200))
    clock = drift.read_frame_times(tmp_path / "s_frame_times.csv")
    assert clock.regular
    t, _ = clock.times(200)
    np.testing.assert_allclose(np.diff(t), np.diff(t)[0], atol=1e-9)   # a clock
    assert np.diff(t)[0] == pytest.approx(PERIOD, abs=1e-5)
    # pinned to the least-delayed frames, not to the average delay
    assert abs(t[0] - T0) < 0.0005


def test_pages_past_the_end_of_the_frame_times_are_extrapolated(tmp_path):
    """The log is flushed once a second, so an acquisition that stopped
    abruptly keeps every frame in the stack but loses the last timestamps."""
    _write_frame_times(tmp_path / "s_frame_times.csv", 20)
    clock = drift.read_frame_times(tmp_path / "s_frame_times.csv")
    t, n_extra = clock.times(25)
    assert n_extra == 5
    np.testing.assert_allclose(t, T0 + PERIOD * np.arange(25), atol=1e-6)


# --- smoothing -----------------------------------------------------------------


def test_the_smoother_follows_a_steady_drift_right_to_both_ends():
    """A moving average pulls the ends of a ramp inwards - exactly where the
    first and last frames of the movie sit."""
    t = np.arange(0.0, 60.0, 0.1)
    f = drift.smoother(t, 3.0 * t, -2.0 * t, sigma_s=5.0)
    q = np.array([0.0, 0.05, 30.0, 59.9])
    dx, dy = f(q)
    np.testing.assert_allclose(dx, 3.0 * q, atol=1e-6)
    np.testing.assert_allclose(dy, -2.0 * q, atol=1e-6)


def test_the_smoother_takes_the_scatter_out():
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 60.0, 0.1)
    noisy = 0.5 * t + rng.normal(0.0, 0.1, len(t))
    dx, _ = drift.smoother(t, noisy, noisy, sigma_s=2.0)(t[50:-50])
    residual = dx - 0.5 * t[50:-50]
    assert np.std(residual) < 0.1 / 3


def test_no_smoothing_interpolates_the_raw_samples_and_skips_gaps():
    t = np.arange(10.0)
    dx = t.copy()
    dx[5] = np.nan
    fx, _ = drift.smoother(t, dx, t, sigma_s=0.0)(np.array([4.5, 5.0, 7.0]))
    np.testing.assert_allclose(fx, [4.5, 5.0, 7.0])


# --- per frame -----------------------------------------------------------------


def _records(tmp_path, n_pages=20):
    _acquisition(tmp_path, n_pages)
    return (drift.read_drift(tmp_path / "acq_Normal_xy_drift.csv"),
            drift.read_frame_times(tmp_path / "acq_Normal_normal_frame_times.csv"))


def test_every_frame_gets_the_drift_of_its_moment_zero_at_the_first(tmp_path):
    record, clock = _records(tmp_path)
    frames = drift.drift_per_frame(record, clock, 20, sigma_s=1.0)
    pages = np.arange(20)
    np.testing.assert_allclose(frames["dx"], RAMP[0] * PERIOD * pages, atol=1e-6)
    np.testing.assert_allclose(frames["dy"], RAMP[1] * PERIOD * pages, atol=1e-6)
    assert frames["dx"][0] == 0.0 and frames["dy"][0] == 0.0
    assert not frames["outside"].any()


def test_a_binned_frame_gets_the_mean_drift_of_the_frames_summed_into_it(tmp_path):
    """What was localized in a binned frame is the sum of its raw frames, so
    it sat, on average, at their average position."""
    record, clock = _records(tmp_path)
    frames = drift.drift_per_frame(record, clock, 20, bin_factor=4, sigma_s=1.0)
    assert len(frames["dx"]) == 5
    # bin b holds pages 4b..4b+3, centred on 4b + 1.5; zeroed at bin 0
    np.testing.assert_allclose(frames["dx"], RAMP[0] * PERIOD * 4 * np.arange(5), atol=1e-6)


def test_frames_the_drift_record_does_not_reach_are_flagged(tmp_path):
    record, clock = _records(tmp_path)
    t = record.t[record.t < T0 + 0.5]
    _write_drift(tmp_path / "short_xy_drift.csv", t, RAMP[0] * (t - T0), RAMP[1] * (t - T0))
    short = drift.read_drift(tmp_path / "short_xy_drift.csv")
    frames = drift.drift_per_frame(short, clock, 20, sigma_s=0.0)
    assert frames["outside"].sum() == 20 - 5
    # held at the last recorded value rather than extrapolated
    assert frames["dx"][-1] == pytest.approx(frames["dx"][-2])


def test_rows_on_frames_past_either_end_take_the_nearest_and_are_counted():
    dx, dy, off = drift.lookup(np.array([-1, 0, 2, 3, 9]), np.array([0.0, 1.0, 2.0]),
                               np.array([0.0, -1.0, -2.0]))
    np.testing.assert_array_equal(dx, [0.0, 0.0, 2.0, 2.0, 2.0])
    assert off == 3


# --- finding the records ---------------------------------------------------------


def test_the_records_are_found_beside_the_stack(tmp_path):
    _acquisition(tmp_path)
    stem = "acq_Normal_normal"
    assert drift.find_drift_file(tmp_path, stem).name == "acq_Normal_xy_drift.csv"
    assert (drift.find_frame_times(tmp_path, stem).name
            == "acq_Normal_normal_frame_times.csv")


def test_a_recovered_partial_stack_uses_the_frame_times_of_its_acquisition(tmp_path):
    _acquisition(tmp_path)
    found = drift.find_frame_times(tmp_path, "acq_Normal_normal.partial")
    assert found.name == "acq_Normal_normal_frame_times.csv"


def test_with_several_drift_records_the_stack_name_decides(tmp_path):
    for name in ("first_Normal", "second_Normal"):
        _write_drift(tmp_path / f"{name}_xy_drift.csv", np.arange(3.0), np.zeros(3), np.zeros(3))
    assert drift.find_drift_file(tmp_path, "second_Normal_normal").name == \
        "second_Normal_xy_drift.csv"
    # nothing to go on: better none than another acquisition's drift
    assert drift.find_drift_file(tmp_path, None) is None


# --- shifting the display ---------------------------------------------------------


def test_the_shifted_stack_moves_each_frame_by_its_own_offset():
    base = np.zeros((3, 16, 16), np.uint16)
    base[:, 8, 8] = 1000
    shifted = drift.ShiftedStack(base, [(0.0, 0.0), (2.0, -3.0), (0.0, 1.0)], expand=False)
    assert shifted.shape == base.shape and shifted.dtype == base.dtype
    np.testing.assert_array_equal(shifted[0], base[0])
    frame = shifted[1]
    assert frame.dtype == np.uint16
    assert frame[10, 5] == 1000 and frame[8, 8] == 0
    # every way napari may ask for it
    np.testing.assert_array_equal(shifted[1, 10], frame[10])
    np.testing.assert_array_equal(shifted[-2], frame)
    np.testing.assert_array_equal(shifted[1:2][0], frame)
    np.testing.assert_array_equal(shifted[..., 5][1], frame[:, 5])
    np.testing.assert_array_equal(np.asarray(shifted)[2, 8, 9], 1000)
    # and nothing written back to the recording
    assert base[1, 8, 8] == 1000 and base[1, 10, 5] == 0
    assert drift.unshifted(shifted) is base
    assert drift.unshifted(base) is base


def test_nothing_the_camera_saw_is_cut_as_the_sample_drifts():
    """In the first frame's coordinates, a drifting sample shows parts the
    first frame never saw. Cut to the first frame's size, the display lost
    them, more of them the further the drift went."""
    base = np.zeros((3, 16, 16), np.uint16)
    base[:, 8, 8] = 1000
    base[1, 15, 0] = 500                      # a corner the drift carries off-frame
    shifted = drift.ShiftedStack(base, [(0.0, 0.0), (2.0, -3.0), (0.0, 1.0)])
    assert shifted.origin_yx == (0, -3)
    assert shifted.shape == (3, 18, 20)       # every frame's field of view
    oy, ox = shifted.origin_yx
    # the spot stands still in the first frame's coordinates...
    for k in range(3):
        frame = shifted[k]
        y, x = np.unravel_index(np.argmax(frame), frame.shape)
        assert frame.max() == 1000
    assert shifted[1][8 + 2 - oy, 8 - 3 - ox] == 1000
    assert shifted[0][8 - oy, 8 - ox] == 1000
    # ...and the corner that left the first frame's field is still drawn
    assert shifted[1][15 + 2 - oy, 0 - 3 - ox] == 500


def test_a_subpixel_drift_lands_on_the_canvas_too():
    base = np.ones((2, 8, 8), np.float32)
    shifted = drift.ShiftedStack(base, [(0.0, 0.0), (0.5, -0.5)])
    assert shifted.origin_yx == (0, -1)
    frame = np.asarray(shifted[1])
    assert frame.shape == (9, 9)
    # the interior keeps its value; the half-covered edge is half as bright
    assert frame[4, 4] == pytest.approx(1.0)
    assert frame[0, 4] == pytest.approx(0.5)


# --- the map between the two cameras ------------------------------------------------


def _record(**meta):
    from types import SimpleNamespace
    return SimpleNamespace(meta=meta)


def test_the_calibrated_map_is_half_a_pixel_turned_by_eight_tenths_of_a_degree():
    """Argo-SIM v2, 2026-09-25: 81.28 nm white-light px at 161.87 nm."""
    calibrated = drift.CALIBRATED_CAMERA_MAP
    assert calibrated.scale * 161.87 == pytest.approx(81.28, abs=0.02)
    assert calibrated.rotation_deg == pytest.approx(-0.80, abs=0.01)
    # a white-light drift along x is mostly x here, and 1.4 % y
    fx, fy = calibrated.to_fluorescence_px(1.0, 0.0)
    assert fy / fx == pytest.approx(-0.0139, abs=0.0005)


def test_a_record_carrying_its_own_map_is_read_with_it():
    own = ((0.49, 0.01), (-0.01, 0.49))
    found, problem = drift.camera_map(_record(wl_orientation="None", **_own_map_meta(own)))
    assert problem is None
    assert found.matrix == own
    assert "drift record" in found.source


def _own_map_meta(matrix):
    return {"xy_calibration": {"basler_to_kuro_px": [list(r) for r in matrix],
                               "taken": "2026-10-01T10:00:00"}}


def test_the_calibration_stands_in_only_for_the_frame_it_was_measured_on():
    found, problem = drift.camera_map(_record(wl_orientation="Rotate 180",
                                              frame_shape=[2048, 2048]))
    assert found is drift.CALIBRATED_CAMERA_MAP and problem is None
    # binned frames: the same map, pixels twice as big
    found, _ = drift.camera_map(_record(wl_orientation="Rotate 180", frame_shape=[1024, 1024]))
    np.testing.assert_allclose(found.matrix,
                               2 * np.asarray(drift.CALIBRATED_CAMERA_MAP.matrix))
    # another orientation, or a cropped frame: no map rather than a wrong one
    for meta in ({"wl_orientation": "None"}, {"frame_shape": [2048, 2448]}):
        found, problem = drift.camera_map(_record(**meta))
        assert found is None and problem


# --- through the widget ---------------------------------------------------------

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_widget_interaction import make_widget  # noqa: E402

WL_NM = 50.0
PIXEL_NM = 100.0
N = 20


def _table(frames=None):
    frames = np.arange(N) if frames is None else np.asarray(frames)
    return pd.DataFrame({
        "frame": frames,
        "x [nm]": np.full(len(frames), 1000.0),
        "y [nm]": np.full(len(frames), 1000.0),
        "intensity [photon]": np.full(len(frames), 500.0),
    })


_NAPARI_WIDGETS = []


def _napari_widget():
    """A widget on a real (canvas-less) napari viewer, for what the stub cannot
    show: real Image layers, and how renders are placed on them."""
    from napari.components import ViewerModel
    from test_widget_interaction import ensure_qapp

    ensure_qapp()
    widget = widget_mod.LocalizationTrackingWidget(ViewerModel())
    _NAPARI_WIDGETS.append(widget)
    return widget


def _own_map(matrix):
    """Drift-record metadata carrying its own camera map, as recFL writes it."""
    return {"xy_calibration": {"basler_to_kuro_px": [list(row) for row in matrix],
                               "wl_orientation": "Rotate 180", "taken": "2026-09-25"}}


def _isotropic(wl_nm):
    """A camera map that is only a scale: one white-light px is wl_nm nm."""
    s = wl_nm / PIXEL_NM
    return _own_map(((s, 0.0), (0.0, s)))


def _loaded(tmp_path, table=None, wl_nm=WL_NM, raw_frames=N, image=None, bin_factor=1,
            napari_viewer=False, drift_meta=None):
    """Data loaded with its drift record. The record carries a map that makes one
    white-light px wl_nm nm, unless drift_meta says otherwise."""
    widget = _napari_widget() if napari_viewer else make_widget()
    widget.pixel_size_box.setValue(PIXEL_NM)
    widget._drift_timer.stop()
    if bin_factor > 1:
        widget.bin_factor_box.setValue(bin_factor)
        widget._time_bin_timer.stop()
    image_path = _acquisition(tmp_path, raw_frames,
                              drift_meta=_isotropic(wl_nm) if drift_meta is None else drift_meta)
    raw = np.zeros((raw_frames, 16, 16), np.uint16)
    image = raw if image is None else image
    widget.csv_edit.setText("")
    widget.image_edit.setText(str(image_path))
    widget._on_load_finished(
        (_table() if table is None else table, image, "decoded", None, raw),
        "", str(image_path))
    return widget


def _expected(frames, factor=1):
    """nm to subtract, per frame, on the ramp the acquisition was written with."""
    frames = np.asarray(frames, dtype=float)
    return (RAMP[0] * PERIOD * factor * frames * WL_NM,
            RAMP[1] * PERIOD * factor * frames * WL_NM)


def test_the_records_beside_the_stack_are_picked_up_on_loading(tmp_path):
    widget = _loaded(tmp_path)
    assert widget.drift_edit.text().endswith("acq_Normal_xy_drift.csv")
    assert widget._frame_clock.path.name == "acq_Normal_normal_frame_times.csv"
    assert "Frames timed" in widget.drift_status.text()
    assert "acq_Normal_normal_frame_times.csv" in widget.log_box.toPlainText()
    assert "drift-corrected" in widget.status_label.text()


def test_each_localization_is_moved_back_by_the_drift_of_its_frame(tmp_path):
    widget = _loaded(tmp_path)
    dx, dy = _expected(np.arange(N))
    np.testing.assert_allclose(widget.df["x [nm]"], 1000.0 - dx, atol=1e-6)
    np.testing.assert_allclose(widget.df["y [nm]"], 1000.0 - dy, atol=1e-6)
    # and says so, so an export can be read - and undone
    np.testing.assert_allclose(widget.df["drift_x [nm]"], dx, atol=1e-6)
    # the table as loaded is kept untouched
    np.testing.assert_allclose(widget._df_source["x [nm]"], 1000.0)


def test_a_record_without_a_map_of_its_own_is_read_with_the_calibration(tmp_path):
    """Everything recorded before the acquisition wrote its map: the calibrated
    one, turn included - x drift on the white-light camera is partly y here."""
    widget = _loaded(tmp_path, drift_meta={})
    (a, b), (c, d) = drift.CALIBRATED_CAMERA_MAP.matrix
    wx = RAMP[0] * PERIOD * np.arange(N)
    wy = RAMP[1] * PERIOD * np.arange(N)
    np.testing.assert_allclose(widget.df["drift_x [nm]"], PIXEL_NM * (a * wx + b * wy), atol=1e-6)
    np.testing.assert_allclose(widget.df["drift_y [nm]"], PIXEL_NM * (c * wx + d * wy), atol=1e-6)
    assert "Argo-SIM v2 calibration of 2026-09-25" in widget.drift_map_label.text()


def test_a_record_taken_with_another_orientation_is_not_corrected(tmp_path):
    """A map for another orientation would move every localization the wrong way."""
    widget = _loaded(tmp_path, drift_meta={"wl_orientation": "Flip vertical"})
    np.testing.assert_allclose(widget.df["x [nm]"], 1000.0)
    assert "drift_x [nm]" not in widget.df.columns
    assert "Flip vertical" in widget.drift_status.text()
    assert "None applies" in widget.drift_map_label.text()


def test_the_drift_is_in_the_tables_own_nanometres(tmp_path):
    """The record reaches nm through fluorescence pixels, so a table made at
    another pixel size gets its drift at that pixel size."""
    widget = _loaded(tmp_path)
    widget.pixel_size_box.setValue(2 * PIXEL_NM)
    widget._refresh_drift_correction()          # what the debounce timer runs
    np.testing.assert_allclose(widget.df["drift_x [nm]"], 2 * _expected(np.arange(N))[0],
                               atol=1e-6)


def test_the_correction_comes_off_again_exactly(tmp_path):
    widget = _loaded(tmp_path)
    widget.drift_enable_box.setChecked(False)
    assert widget.df is widget._df_source
    assert "drift-corrected" not in widget.status_label.text()
    widget.drift_enable_box.setChecked(True)
    np.testing.assert_allclose(widget.df["x [nm]"], 1000.0 - _expected(np.arange(N))[0],
                               atol=1e-6)


def test_a_table_that_arrives_corrected_is_not_corrected_twice(tmp_path):
    """An export carries the drift it had taken out; loading it again must
    land on the same coordinates, not move them a second time."""
    widget = _loaded(tmp_path)
    exported = widget.df.copy()
    widget._ingest_localization_dataframe(exported, "again", frame_is_zero_indexed=True)
    np.testing.assert_allclose(widget.df["x [nm]"], exported["x [nm]"], atol=1e-9)
    np.testing.assert_allclose(widget._df_source["x [nm]"], 1000.0, atol=1e-9)


def test_a_corrected_table_keeps_its_own_correction_without_a_record(tmp_path):
    widget = _loaded(tmp_path)
    exported = widget.df.copy()
    widget._read_drift_files(None)
    widget._ingest_localization_dataframe(exported, "elsewhere", frame_is_zero_indexed=True)
    np.testing.assert_allclose(widget.df["x [nm]"], exported["x [nm]"], atol=1e-9)
    widget.drift_enable_box.setChecked(False)
    np.testing.assert_allclose(widget.df["x [nm]"], 1000.0, atol=1e-9)


def test_shifting_the_frame_numbers_re_times_every_localization(tmp_path):
    widget = _loaded(tmp_path, table=_table(np.arange(1, N)))   # starts at 1: shifted -1
    assert widget._frame_offset() == -1
    np.testing.assert_allclose(widget.df["x [nm]"],
                               1000.0 - _expected(np.arange(0, N - 1))[0], atol=1e-6)
    widget.shift_frame_numbers(None)
    np.testing.assert_allclose(widget.df["x [nm]"],
                               1000.0 - _expected(np.clip(np.arange(1, N), 0, N - 1))[0],
                               atol=1e-6)


def test_localizations_past_the_end_of_the_stack_are_reported(tmp_path):
    widget = _loaded(tmp_path, table=_table(np.arange(N + 3)))
    assert widget._drift_rows_off_end == 3
    assert "outside the stack" in widget.log_box.toPlainText()


def test_binned_frames_are_corrected_by_the_mean_drift_of_their_raw_frames(tmp_path):
    factor = 4
    binned = np.zeros((N // factor, 16, 16), np.uint32)
    widget = _loaded(tmp_path, table=_table(np.arange(N // factor)),
                     image=binned, bin_factor=factor)
    np.testing.assert_allclose(widget.df["x [nm]"],
                               1000.0 - _expected(np.arange(N // factor), factor)[0],
                               atol=1e-6)


def test_wide_open_xy_bounds_follow_the_data_and_narrowed_ones_stay(tmp_path):
    widget = _loaded(tmp_path)
    x_lower, x_upper = widget.filter_controls["x [nm]"]
    y_lower, y_upper = widget.filter_controls["y [nm]"]
    y_upper.setValue(y_upper.value() - 1.0)       # someone narrowed y
    narrowed = y_upper.value()
    widget.drift_enable_box.setChecked(False)
    assert x_lower.value() == pytest.approx(1000.0)
    assert x_upper.value() == pytest.approx(1000.0)
    assert y_upper.value() == pytest.approx(narrowed)


def test_the_image_is_shown_shifted_and_fitted_as_recorded(tmp_path):
    widget = _loaded(tmp_path)
    layer = widget.viewer.layers[widget._image_layer_name]
    assert isinstance(layer.data, drift.ShiftedStack)
    dx, dy = _expected(np.arange(N))
    np.testing.assert_allclose(layer.data.shifts_yx[:, 0], -dy / PIXEL_NM, atol=1e-9)
    np.testing.assert_allclose(layer.data.shifts_yx[:, 1], -dx / PIXEL_NM, atol=1e-9)
    # detection and fitting read the frames as the camera recorded them
    assert widget._loc2d_stack(layer) is layer.data.base

    widget.drift_shift_image_box.setChecked(False)
    assert not isinstance(layer.data, drift.ShiftedStack)
    widget.drift_shift_image_box.setChecked(True)
    widget.drift_enable_box.setChecked(False)       # no image shift without correction
    assert not isinstance(layer.data, drift.ShiftedStack)


def test_the_shifted_image_is_placed_where_its_canvas_starts(tmp_path):
    widget = _loaded(tmp_path, napari_viewer=True)
    layer = widget.viewer.layers[widget._image_layer_name]
    oy, ox = layer.data.origin_yx
    assert (oy, ox) != (0, 0)                 # the test drift runs off one edge
    np.testing.assert_allclose(np.ravel(layer.translate)[-2:],
                               (oy * PIXEL_NM, ox * PIXEL_NM))
    widget.drift_enable_box.setChecked(False)
    np.testing.assert_allclose(np.ravel(layer.translate)[-2:], (0.0, 0.0))


def test_a_render_covers_every_frames_field_of_view(tmp_path):
    """Localizations from the part of the sample that drifted into view have
    to land on the render grid, not off it."""
    widget = _loaded(tmp_path, napari_viewer=True)
    layer = widget.viewer.layers[widget._image_layer_name]
    shape, origin, _source = widget._render_field_of_view()
    oy, ox = layer.data.origin_yx
    assert shape == layer.data.shape[-2:]
    assert origin == (oy - 0.5, ox - 0.5)
    corrected_x = widget.df["x [nm]"].to_numpy() / PIXEL_NM
    corrected_y = widget.df["y [nm]"].to_numpy() / PIXEL_NM
    assert corrected_x.min() >= origin[1] and corrected_x.max() <= origin[1] + shape[1]
    assert corrected_y.min() >= origin[0] and corrected_y.max() <= origin[0] + shape[0]


def test_a_render_on_a_drift_canvas_sits_where_its_localizations_are(tmp_path):
    widget = _loaded(tmp_path, napari_viewer=True)
    widget.render_png_box.setChecked(False)
    options, info, source = widget._render_inputs()
    widget._add_render_layer("image", np.zeros((4, 4), np.float32), info, source, options)
    render = [lay for lay in widget.viewer.layers if widget_mod.is_render_layer(lay)][0]
    over = options["oversampling"]
    # super-resolved pixel 0 is centred on localization pixel origin + 0.5/over
    expected = [PIXEL_NM * (o + 0.5 / over) for o in options["origin"]]
    np.testing.assert_allclose(np.ravel(render.translate)[-2:], expected)


def test_the_image_moves_by_the_same_camera_pixels_whatever_the_pixel_size(tmp_path):
    """The record reaches fluorescence pixels through the camera map before any
    pixel size enters: the sample moved that many camera pixels whatever a pixel
    is said to measure. Only the nm the localizations are moved by scale."""
    widget = _loaded(tmp_path)
    layer = widget.viewer.layers[widget._image_layer_name]
    before = layer.data.shifts_yx.copy()
    widget.pixel_size_box.setValue(2 * PIXEL_NM)
    np.testing.assert_allclose(layer.data.shifts_yx, before, atol=1e-9)


def test_detection_candidates_move_with_the_image_they_were_found_on(tmp_path):
    widget = _loaded(tmp_path)
    widget._loc2d_candidates = [None] * N
    widget._loc2d_candidates[10] = (np.array([5.0]), np.array([6.0]), np.array([1.0]))
    widget._update_loc2d_candidate_overlay()
    coords = np.asarray(widget.viewer.layers[widget_mod.LOC2D_CANDIDATES_LAYER_NAME].data)
    shift = widget.viewer.layers[widget._image_layer_name].data.shifts_yx[10]
    np.testing.assert_allclose(coords[0], [10, 5.0 + shift[0], 6.0 + shift[1]])


def test_drift_settings_round_trip_and_the_camera_map_is_recorded(tmp_path):
    widget = _loaded(tmp_path)
    widget.drift_smoothing_box.setValue(3.5)
    widget.drift_shift_image_box.setChecked(False)
    metadata = widget._collect_metadata(None)
    section = metadata["drift_correction"]
    assert section["applied_to_localizations"]
    assert section["drift_over_movie_nm"]["x"] == pytest.approx(_expected([N - 1])[0][0])
    assert section["camera_map"]["fluorescence_px_per_white_light_px"] == [[0.5, 0.0],
                                                                          [0.0, 0.5]]
    assert section["camera_map"]["white_light_nm_per_px"] == pytest.approx(WL_NM)
    values, _notes = widget_mod.settings_from_metadata(metadata)
    assert values["drift_smoothing_box"] == pytest.approx(3.5)
    assert values["drift_shift_image_box"] is False

    other = make_widget()
    other.apply_settings(metadata, include_instrument=False)
    assert other.drift_smoothing_box.value() == pytest.approx(3.5)


def test_a_run_made_with_a_typed_white_light_pixel_says_its_drift_differs():
    _values, notes = widget_mod.settings_from_metadata(
        {"drift_correction": {"wl_pixel_size_nm": 77.0}})
    assert any("77 nm" in note and "camera map" in note for note in notes)


def test_the_export_carries_the_drift_applied_to_each_frame(tmp_path):
    widget = _loaded(tmp_path)
    tables = dict(widget._export_tables())
    per_frame = tables["drift_per_frame.csv"]
    assert len(per_frame) == N
    np.testing.assert_allclose(per_frame["drift_x [nm]"], _expected(np.arange(N))[0],
                               atol=1e-6)
    assert "drift_x [nm]" in tables["localizations_filtered.csv"].columns


def test_a_session_keeps_the_table_as_fitted_and_the_record_it_used(tmp_path, monkeypatch):
    widget = _loaded(tmp_path)
    captured = {}

    class _Worker:
        def __init__(self, *args):
            captured["args"] = args
            signal = type("S", (), {"connect": lambda *_a, **_k: None})()
            self.returned = self.errored = self.finished = signal

        def start(self):
            pass

    monkeypatch.setattr(widget_mod, "_session_save_worker", _Worker)
    widget.save_session(tmp_path / "s.loctrack-session.json")
    _path, manifest, locs_frame, _locs_path = captured["args"]
    assert manifest["sources"]["drift"]["path"].endswith("acq_Normal_xy_drift.csv")
    np.testing.assert_allclose(locs_frame["x [nm]"], 1000.0)
    assert "drift_x [nm]" not in locs_frame.columns


def test_a_repeated_refresh_leaves_the_trajectories_alone(tmp_path):
    """A debounce firing after a session restored the same settings must not
    throw away the trajectories that restore has just re-linked."""
    widget = _loaded(tmp_path)
    widget.tracks = pd.DataFrame({"particle": [0, 0], "frame": [0, 1],
                                  "x": [1.0, 2.0], "y": [1.0, 2.0]})
    widget._refresh_drift_correction()
    assert widget.tracks is not None
