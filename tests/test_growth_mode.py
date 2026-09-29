"""The growth correction in the widget: every localization carried to the final geometry.

A synthetic acquisition: the drift record carries its own camera map, and a
deformation measured on its snapshots sits in analysis/ - a stretch along y
about the frame's centre, 6 % for the first snapshot, none for the last. What
has to hold: it is found on loading; each localization lands where the map puts
it; steps between neighbouring localizations are stretched in the final
geometry but not in the local coordinates the metrics use; and the image is
drawn warped.
"""
import importlib
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

conftest = importlib.import_module("conftest")
deform = conftest.load_deform()

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_drift import N, PIXEL_NM, T0, _loaded  # noqa: E402
from test_drift_checks import _ramp, _snapshots  # noqa: E402

OWN_MAP = {"xy_calibration": {"basler_to_kuro_px": [[0.5, 0.0], [0.0, 0.5]],
                              "basler_to_kuro_affine": [[0.5, 0.0, 10.0], [0.0, 0.5, 20.0]],
                              "wl_orientation": "Rotate 180"}}
SNAP_T = T0 + np.array([-1.0, 1.0, 3.0])
CENTRE = (1024.0, 1024.0)


def _stretch(k):
    return 1.0 + 0.06 * (len(SNAP_T) - 1 - k) / (len(SNAP_T) - 1)


def _record():
    fwd, inv = [], []
    for k in range(len(SNAP_T)):
        s = _stretch(k)
        M = np.array([[1.0, 0.0, 0.0], [0.0, s, CENTRE[1] * (1 - s)], [0.0, 0.0, 1.0]])
        fwd.append(deform.QuadMap.from_affine(M, CENTRE, 1024.0))
        inv.append(deform.QuadMap.from_affine(np.linalg.inv(M), CENTRE, 1024.0))
    return deform.DeformationRecord(SNAP_T, fwd, inv, "growth", np.pi / 2, (2048, 2048),
                                    (0, 2048, 0, 2048),
                                    source={"drift_record": "acq_Normal_xy_drift.csv"})


def _table():
    rng = np.random.default_rng(0)
    frames = np.repeat(np.arange(N), 3)
    return pd.DataFrame({"frame": frames,
                         "x [nm]": rng.uniform(200, 1400, len(frames)),
                         "y [nm]": rng.uniform(200, 1400, len(frames)),
                         "intensity [photon]": 500.0})


@pytest.fixture
def growing(tmp_path):
    _record().save(tmp_path / "analysis" / "2026-09-26_120000_deformation" / "deformation.json")
    _snapshots(tmp_path, SNAP_T, _ramp)
    widget = _loaded(tmp_path, table=_table(), napari_viewer=True, drift_meta=OWN_MAP)
    widget.wl_roi_x_box.setValue(3)
    widget.wl_roi_y_box.setValue(4)
    widget.drift_mode_box.setCurrentIndex(widget.drift_mode_box.findData("growth"))
    # these tests work in the last snapshot's geometry; the first frame's has its own below
    widget.growth_reference_box.setCurrentIndex(widget.growth_reference_box.findData("last"))
    widget._refresh_drift_correction()
    return widget


def test_the_measured_growth_is_found_on_loading(growing):
    assert growing._deformation is not None
    assert "+6.0 % along the root axis" in growing.growth_status.text()


def test_every_localization_lands_where_the_map_puts_it(growing):
    w = growing
    assert w._applied_drift[2] == "growth"
    fd = w._fluorescence_deformation()
    t = w._drift_per_frame()["t"]
    src = w._df_source
    q = src[["x [nm]", "y [nm]"]].to_numpy() / PIXEL_NM
    expected = fd.to_final(q, t[src["frame"].to_numpy()])
    np.testing.assert_allclose(w.df[["x [nm]", "y [nm]"]].to_numpy() / PIXEL_NM, expected, atol=1e-9)
    # by hand, for one of them: fluorescence px -> sensor -> white light -> stretched -> back
    x, y = q[0]
    b = np.array([(x + 3 - 10) / 0.5, (y + 4 - 20) / 0.5])
    tt = t[int(src["frame"].iloc[0])]
    fm = fd.fast_motion(np.array([tt]))[0]                  # the ramp is linear: ~0
    assert np.hypot(*fm) < 1e-6
    s = np.interp(tt, SNAP_T, [_stretch(k) for k in range(len(SNAP_T))])
    bf = np.array([b[0], CENTRE[1] + s * (b[1] - CENTRE[1])])
    qf = 0.5 * bf + np.array([10 - 3, 20 - 4])
    np.testing.assert_allclose(expected[0], qf, atol=1e-6)


def test_steps_are_measured_without_the_stretch(growing):
    """Two localizations 10 px apart along the axis, early in the movie: 10 px
    apart in local coordinates, ~10.6 px in the final geometry."""
    w = growing
    fd = w._fluorescence_deformation()
    t0 = w._drift_per_frame()["t"][0]
    q = np.array([[100.0, 100.0], [100.0, 110.0]])
    final, local = fd.map_rows(q, np.full(2, t0), (100.0, 100.0))
    stretch = np.interp(t0, SNAP_T, [_stretch(k) for k in range(len(SNAP_T))])
    assert final[1, 1] - final[0, 1] == pytest.approx(10 * stretch, rel=1e-6)
    assert local[1, 1] - local[0, 1] == pytest.approx(10.0, rel=1e-6)


def test_the_metrics_read_local_coordinates(growing):
    w = growing
    features = w._prepare_features()
    assert {"x_local", "y_local"} <= set(features.columns)
    w.tracks = features.assign(particle=0)
    metric_tracks = w._tracks_for_metrics()
    np.testing.assert_allclose(metric_tracks["y"], features["y_local"])
    assert not np.allclose(metric_tracks["y"], features["y"])


def test_the_image_and_the_white_light_are_drawn_in_the_final_geometry(growing):
    w = growing
    data = w.viewer.layers[w._image_layer_name].data
    assert isinstance(data, widget_mod.drift_io.WarpedStack)
    frame = np.asarray(data[0])
    assert frame.shape == data.shape[1:]
    w.show_wl_overlay()
    overlay = w._wl_overlay_layer()
    assert isinstance(overlay.data.base, widget_mod.drift_io.WarpedStack)


def test_the_export_and_metadata_say_what_was_done(growing):
    w = growing
    table = w._drift_table()
    assert "stretch_at_image_centre" in table.columns
    assert table["stretch_at_image_centre"].iloc[0] > table["stretch_at_image_centre"].iloc[-1]
    section = w._drift_metadata()
    assert section["mode"] == "growth"
    assert section["growth"]["applied"]
    assert section["growth"]["stretch_along_axis"] == pytest.approx(0.06, abs=1e-6)


def test_back_to_drift_mode_the_record_applies_as_before(growing):
    w = growing
    w.drift_mode_box.setCurrentIndex(w.drift_mode_box.findData("drift"))
    w._refresh_drift_correction()
    assert w._applied_drift[2] == "record"
    assert w._local_px is None


def _first_frame(widget):
    widget.growth_reference_box.setCurrentIndex(widget.growth_reference_box.findData("first"))
    widget._refresh_drift_correction()
    return widget


def test_the_first_frame_is_the_default_geometry(tmp_path):
    _record().save(tmp_path / "analysis" / "2026-09-26_120000_deformation" / "deformation.json")
    _snapshots(tmp_path, SNAP_T, _ramp)
    widget = _loaded(tmp_path, table=_table(), napari_viewer=True, drift_meta=OWN_MAP)
    assert widget.growth_reference_box.currentData() == "first"


def test_in_the_first_frames_geometry_the_first_frame_stays_put(growing):
    w = _first_frame(growing)
    fd = w._fluorescence_deformation()
    t = w._drift_per_frame()["t"]
    q = np.array([[100.0, 100.0], [150.0, 180.0]])
    np.testing.assert_allclose(fd.to_final(q, np.full(2, t[0])), q, atol=1e-6)
    # a later localization is carried back onto the first frame's tissue: the
    # stretch that happened since is taken out of it
    later = fd.to_final(q, np.full(2, t[-1]))
    s0 = np.interp(t[0], SNAP_T, [_stretch(k) for k in range(len(SNAP_T))])
    s1 = np.interp(t[-1], SNAP_T, [_stretch(k) for k in range(len(SNAP_T))])
    assert (later[1, 1] - later[0, 1]) == pytest.approx((q[1, 1] - q[0, 1]) * s1 / s0, rel=1e-6)
    # and back, at a snapshot (between snapshots the forward and inverse maps are
    # interpolated separately, and agree only to the square of their step)
    at_snapshot = np.full(2, SNAP_T[1])
    there = fd.to_final(q, at_snapshot)
    np.testing.assert_allclose(fd.from_final(there, at_snapshot), q, atol=1e-6)


def test_in_the_first_frames_geometry_steps_are_still_measured_without_the_stretch(growing):
    w = _first_frame(growing)
    fd = w._fluorescence_deformation()
    t_end = w._drift_per_frame()["t"][-1]
    q = np.array([[100.0, 100.0], [100.0, 110.0]])
    _final, local = fd.map_rows(q, np.full(2, t_end), (100.0, 100.0))
    assert local[1, 1] - local[0, 1] == pytest.approx(10.0, rel=1e-6)
    assert w._drift_metadata()["growth"]["geometry"] == "the first frame's"


def test_in_the_first_frames_geometry_the_white_light_follows(growing):
    w = _first_frame(growing)
    w.show_wl_overlay()
    overlay = w._wl_overlay_layer()
    stack = overlay.data.base
    assert isinstance(stack, widget_mod.drift_io.WarpedStack)
    assert stack.key[-1] == pytest.approx(w._growth_reference_time())
