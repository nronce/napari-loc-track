"""The white-light snapshots laid over the fluorescence, through the camera map.

What has to hold: a white-light pixel lands where the map puts it on the
fluorescence image - including an image read out of a cropped sensor, and
binned white-light frames; each frame of the movie shows the snapshot taken
nearest to it; the snapshots move back by the drift exactly when the
fluorescence image does; and the layer follows the pixel size like the rest.
"""
import importlib
import json
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

conftest = importlib.import_module("conftest")
drift = conftest.load_drift()
acqmeta = conftest.load_acqmeta()


def test_a_white_light_pixel_lands_where_the_map_puts_it():
    cmap = drift.CameraMap(((0.5, 0.01), (-0.01, 0.5)), "test", offset=(90.0, 80.0))
    affine = cmap.image_affine(160.0, sensor_origin_xy=(300, 450))
    bx, by = 1000.0, 700.0
    kx = 0.5 * bx + 0.01 * by + 90.0 - 300.0
    ky = -0.01 * bx + 0.5 * by + 80.0 - 450.0
    np.testing.assert_allclose(affine @ [by, bx, 1.0], [160.0 * ky, 160.0 * kx, 1.0])


def test_a_binned_white_light_pixel_lands_where_its_raw_pixels_do():
    binned, _ = drift.camera_map(SimpleNamespace(
        meta={"wl_orientation": "Rotate 180", "frame_shape": [1024, 1024]}))
    raw = drift.CALIBRATED_CAMERA_MAP
    # binned pixel (row 300, column 400) is centred on raw (600.5, 800.5)
    np.testing.assert_allclose(binned.image_affine(100.0) @ [300.0, 400.0, 1.0],
                               raw.image_affine(100.0) @ [600.5, 800.5, 1.0], atol=1e-9)


def test_a_record_with_its_own_affine_is_placed_by_it():
    own = {"basler_to_kuro_px": [[0.5, 0.0], [0.0, 0.5]],
           "basler_to_kuro_affine": [[0.5, 0.0, 12.0], [0.0, 0.5, 34.0]]}
    found, _ = drift.camera_map(SimpleNamespace(meta={"xy_calibration": own}))
    assert found.offset == (12.0, 34.0)
    # an older record, matrix only: the calibrated position stands in
    older, _ = drift.camera_map(SimpleNamespace(
        meta={"xy_calibration": {"basler_to_kuro_px": own["basler_to_kuro_px"]}}))
    assert older.offset == drift.CALIBRATED_CAMERA_MAP.offset


def test_each_frame_shows_the_snapshot_taken_nearest_to_it():
    index = drift.nearest_snapshot([10.0, 20.0, 30.0], [9.0, 14.9, 15.1, 26.0, 40.0])
    assert list(index) == [0, 0, 1, 2, 2]
    stack = np.arange(3)[:, None, None] * np.ones((3, 2, 2))
    framed = drift.FrameIndexedStack(stack, index)
    assert framed.shape == (5, 2, 2)
    assert framed[3][0, 0] == 2
    assert framed[1:3].shape == (2, 2, 2)


def test_the_sensor_roi_is_read_off_the_acquisition_a_zero_origin_too(tmp_path):
    meta = {"config": {"acquisition_options": {
        "camera_sensor_roi": {"x": 0, "y": 453, "width": 364, "height": 503}}}}
    (tmp_path / "s_003_Normal_root_metadata.json").write_text(json.dumps(meta))
    stack = tmp_path / "s_003_Normal_root_normal.tif"
    stack.write_bytes(b"")
    values = acqmeta.read_acquisition_metadata(stack)["values"]
    assert values["sensor_roi_x"] == 0.0
    assert values["sensor_roi_y"] == 453.0


# --- through the widget ------------------------------------------------------------

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_drift import N, PIXEL_NM, RAMP, T0, _loaded  # noqa: E402
from test_drift_checks import _ramp, _snapshots  # noqa: E402

OWN_MAP = {"xy_calibration": {"basler_to_kuro_px": [[0.5, 0.0], [0.0, 0.5]],
                              "basler_to_kuro_affine": [[0.5, 0.0, 10.0], [0.0, 0.5, 20.0]],
                              "wl_orientation": "Rotate 180"}}


def _overlaid(tmp_path, drift_meta=OWN_MAP):
    _snapshots(tmp_path, T0 + np.array([-0.5, 0.3, 1.1, 1.9]), _ramp)
    widget = _loaded(tmp_path, napari_viewer=True, drift_meta=drift_meta)
    widget.wl_roi_x_box.setValue(3)
    widget.wl_roi_y_box.setValue(4)
    widget.show_wl_overlay()
    return widget


def test_the_white_light_is_laid_over_the_fluorescence_frame_by_frame(tmp_path):
    widget = _overlaid(tmp_path)
    layer = widget._wl_overlay_layer()
    assert layer is not None
    assert layer.data.shape[0] == N                       # the movie's frame axis
    frames_t = widget._drift_per_frame()["t"]
    np.testing.assert_array_equal(
        layer.data.index, drift.nearest_snapshot(widget._wl_stack.t_epoch, frames_t))
    # data (frame, row i, column j) is white-light pixel (i + oy, j + ox), which
    # the map puts at fluorescence (0.5 (j + ox) + 10 - 3, 0.5 (i + oy) + 20 - 4)
    oy, ox = layer.data.base.origin_yx
    i, j = 5.0, 7.0
    world = np.asarray(layer.affine.affine_matrix) @ [0.0, i, j, 1.0]
    np.testing.assert_allclose(world[1:3], [PIXEL_NM * (0.5 * (i + oy) + 20.0 - 4.0),
                                            PIXEL_NM * (0.5 * (j + ox) + 10.0 - 3.0)])
    assert "nearest snapshot" in widget.wl_overlay_status.text()


def test_the_white_light_moves_back_by_the_drift_when_the_image_does(tmp_path):
    widget = _overlaid(tmp_path)
    shifted = widget._wl_overlay_layer().data.base
    t = widget._wl_stack.t_epoch
    t0 = widget._drift_per_frame()["t"][0]
    np.testing.assert_allclose(shifted.shifts_yx[:, 1], -RAMP[0] * (t - t0), atol=1e-3)
    np.testing.assert_allclose(shifted.shifts_yx[:, 0], -RAMP[1] * (t - t0), atol=1e-3)
    widget.drift_shift_image_box.setChecked(False)
    np.testing.assert_allclose(widget._wl_overlay_layer().data.base.shifts_yx, 0.0)
    assert "as recorded" in widget.wl_overlay_status.text()


def test_the_white_light_follows_the_pixel_size_and_the_roi(tmp_path):
    widget = _overlaid(tmp_path)
    layer = widget._wl_overlay_layer()
    before = np.asarray(layer.affine.affine_matrix).copy()
    widget.pixel_size_box.setValue(2 * PIXEL_NM)
    after = np.asarray(layer.affine.affine_matrix)
    np.testing.assert_allclose(after[1:3, 1:], 2 * before[1:3, 1:])
    widget.wl_roi_x_box.setValue(13)                      # 10 px further right on the sensor
    moved = np.asarray(layer.affine.affine_matrix)
    assert moved[2, 3] == pytest.approx(after[2, 3] - 10 * 2 * PIXEL_NM)


def test_snapshots_without_a_drift_record_are_laid_over_as_recorded(tmp_path):
    """An acquisition that saved snapshots but recorded no drift - a calibration
    slide, say - is still overlaid, with the calibrated map, on its frame clock."""
    from test_drift import _napari_widget, _table, _write_frame_times

    _write_frame_times(tmp_path / "acq_Normal_normal_frame_times.csv", N)
    _snapshots(tmp_path, T0 + np.array([-0.5, 0.3, 1.1, 1.9]), _ramp)
    widget = _napari_widget()
    widget.pixel_size_box.setValue(PIXEL_NM)
    image_path = tmp_path / "acq_Normal_normal.tif"
    raw = np.zeros((N, 16, 16), np.uint16)
    widget.image_edit.setText(str(image_path))
    widget._on_load_finished((_table(), raw, "decoded", None, raw), "", str(image_path))
    assert widget._drift_record is None and widget._wl_stack is not None
    widget.show_wl_overlay()
    layer = widget._wl_overlay_layer()
    assert layer is not None and layer.data.shape[0] == N
    np.testing.assert_allclose(layer.data.base.shifts_yx, 0.0)
    assert "as recorded" in widget.wl_overlay_status.text()


def test_the_roi_is_filled_in_from_the_acquisition_and_not_carried_over():
    from test_widget_interaction import make_widget

    widget = make_widget()
    widget._apply_acquisition_metadata(
        {"values": {"sensor_roi_x": 301.0, "sensor_roi_y": 453.0},
         "sources": {"sensor_roi_x": "m", "sensor_roi_y": "m"}})
    assert (widget.wl_roi_x_box.value(), widget.wl_roi_y_box.value()) == (301, 453)
    # the next acquisition did not record one: never the previous dataset's
    widget._apply_acquisition_metadata({"values": {}, "sources": {}})
    assert (widget.wl_roi_x_box.value(), widget.wl_roi_y_box.value()) == (0, 0)


def test_no_white_light_for_a_record_no_map_holds_for(tmp_path):
    widget = _overlaid(tmp_path, drift_meta={"wl_orientation": "Flip vertical"})
    assert widget._wl_overlay_layer() is None
    assert "Flip vertical" in widget.wl_overlay_status.text()
