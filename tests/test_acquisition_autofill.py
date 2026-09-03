"""Filling the acquisition parameters in from the file that recorded them.

These three boxes - pixel size, frame rate, camera offset - decide what every
physical result downstream *means*, so the rule is not "fill in as much as
possible". It is that a box only moves when the microscope actually recorded the
value, and that when one moves it says so, loudly enough that a value belonging
to a different microscope gets noticed rather than published.
"""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_widget_interaction import make_widget  # noqa: E402


def _acquisition(**values):
    """What the reader hands over: the values it found, and where each came from."""
    return {"values": values,
            "sources": {key: "test_metadata.txt: field" for key in values}}


def _log_of(widget):
    return widget.log_box.toPlainText()


def _applied(widget=None, **values):
    widget = widget or make_widget()
    widget.log_box.clear()
    widget._apply_acquisition_metadata(_acquisition(**values))
    return widget


# --- what gets filled in ------------------------------------------------------


def test_a_recorded_frame_rate_reaches_the_box():
    widget = _applied(fps=31.882, frame_interval_ms=31.3656)
    assert widget.fps_box.value() == pytest.approx(31.882, abs=1e-3)


def test_the_frame_interval_box_follows_the_frame_rate():
    """They are one setting shown two ways; filling one has to move the other."""
    widget = _applied(fps=31.882, frame_interval_ms=31.3656)
    assert widget.frame_interval_box.value() == pytest.approx(31.3656, abs=1e-2)


def test_a_recorded_camera_offset_reaches_the_fitter():
    widget = make_widget()
    widget.loc_offset_box.setValue(0.0)
    _applied(widget, camera_offset_adu=100.0)
    assert widget.loc_offset_box.value() == pytest.approx(100.0)


def test_a_calibrated_pixel_size_reaches_the_box():
    widget = _applied(pixel_size_nm=108.3)
    assert widget.pixel_size_box.value() == pytest.approx(108.3, abs=0.05)


# --- what deliberately does not ----------------------------------------------


def test_an_uncalibrated_pixel_size_leaves_the_box_exactly_as_it_was():
    """Micro-Manager records 0.0 for an objective nobody calibrated.

    The reader drops it rather than passing zero along, and nothing here may
    invent a replacement: a confidently wrong pixel size is worse than the
    default, because every distance and diffusion coefficient inherits it.
    """
    widget = make_widget()
    widget.pixel_size_box.setValue(161.0)
    _applied(widget, fps=31.882)
    assert widget.pixel_size_box.value() == pytest.approx(161.0)


def test_the_objective_is_named_when_the_pixel_size_is_missing():
    """The only clue left about whether 161 nm/px belongs to this acquisition."""
    widget = _applied(fps=31.882, objective="4-Plan Apo TIRF 60x NA 1.45 Oil")
    log = _log_of(widget)
    assert "not recorded" in log
    assert "60x NA 1.45 Oil" in log


def test_no_metadata_at_all_moves_nothing_and_says_so():
    widget = make_widget()
    before = (widget.pixel_size_box.value(), widget.fps_box.value(),
              widget.loc_offset_box.value())
    widget.log_box.clear()
    widget._apply_acquisition_metadata(None)
    assert (widget.pixel_size_box.value(), widget.fps_box.value(),
            widget.loc_offset_box.value()) == before
    assert "No acquisition metadata" in _log_of(widget)


# --- saying what happened -----------------------------------------------------


def test_a_change_is_logged_with_the_value_it_replaced_and_its_source():
    widget = make_widget()
    widget.fps_box.setValue(100.0)
    widget.log_box.clear()
    widget._apply_acquisition_metadata({
        "values": {"fps": 31.882},
        "sources": {"fps": "stab_metadata.txt: measured over 414 frames"},
    })
    log = _log_of(widget)
    assert "100.000 fps -> 31.882 fps" in log
    assert "measured over 414 frames" in log


def test_a_value_that_was_already_right_is_not_announced_as_a_change():
    widget = make_widget()
    widget.fps_box.setValue(31.882)
    _applied(widget, fps=31.882)
    assert "Frame rate" not in _log_of(widget)


def test_a_value_the_control_cannot_hold_is_logged_as_what_was_actually_set():
    """setValue clamps; the log has to report the box, not the wish."""
    widget = make_widget()
    widget.pixel_size_box.setValue(161.0)
    _applied(widget, pixel_size_nm=99999.0)
    assert widget.pixel_size_box.value() == widget.pixel_size_box.maximum()
    assert f"{widget.pixel_size_box.maximum():.1f} nm/px" in _log_of(widget)


def test_the_acquisition_context_is_reported_without_being_applied():
    widget = _applied(exposure_ms=20.0, n_frames=10000.0, camera_chip="GS144BSI",
                      objective="4-Plan Apo TIRF 60x NA 1.45 Oil")
    log = _log_of(widget)
    assert "20 ms" in log and "10000" in log and "GS144BSI" in log


# --- the whole way through, from a file on disk -------------------------------


def _micromanager_stack(tmp_path, interval_ms=31.365, n_frames=40, pixel_size_um=0.0):
    """A miniature of the real thing: an OME-TIFF and its Micro-Manager sidecar."""
    tifffile = pytest.importorskip("tifffile")
    stack = tmp_path / "stab_MMStack_Pos0.ome.tif"
    tifffile.imwrite(stack, np.zeros((n_frames, 8, 8), np.uint16),
                     ome=True, metadata={"axes": "TYX"})

    summary = {"Interval_ms": 0.0, "Frames": n_frames, "Width": 8, "Height": 8,
               "MdaSettings": json.dumps({"numFrames": n_frames, "comment": "{"})}
    blocks = ['"Summary": ' + json.dumps(summary, indent=2)]
    for index in range(n_frames):
        blocks.append(f'"FrameKey-{index}-0-0": ' + json.dumps({
            "Core-Camera": "Camera-1",
            "ElapsedTime-ms": round(index * interval_ms),
            "Exposure-ms": 20.0,
            "PixelSizeUm": pixel_size_um,
            "Camera-1-Offset": "250",
            "TINosePiece-Label": "4-Plan Apo TIRF 60x NA 1.45 Oil",
        }, indent=2))
    (tmp_path / "stab_MMStack_Pos0_metadata.txt").write_text(
        "{\n" + ",\n".join(blocks) + "\n}", encoding="utf-8")
    return stack


def test_loading_a_micromanager_stack_fills_the_boxes_in(tmp_path):
    """The end-to-end path: the worker's result applied by the load handler."""
    stack = _micromanager_stack(tmp_path)
    widget = make_widget()
    widget.pixel_size_box.setValue(161.0)
    widget.log_box.clear()

    acquisition = widget_mod.read_acquisition_metadata(stack)
    widget._apply_acquisition_metadata(acquisition)

    assert widget.fps_box.value() == pytest.approx(1000.0 / 31.365, abs=0.1)
    assert widget.loc_offset_box.value() == pytest.approx(250.0)
    # nothing calibrated the objective, so this one is left alone
    assert widget.pixel_size_box.value() == pytest.approx(161.0)
    assert "measured over" in _log_of(widget)


def test_a_calibrated_stack_fills_the_pixel_size_too(tmp_path):
    stack = _micromanager_stack(tmp_path, pixel_size_um=0.1083)
    widget = make_widget()
    widget.pixel_size_box.setValue(161.0)
    widget._apply_acquisition_metadata(widget_mod.read_acquisition_metadata(stack))
    assert widget.pixel_size_box.value() == pytest.approx(108.3, abs=0.05)


def test_the_load_worker_carries_the_acquisition_back_with_the_image(tmp_path):
    """The read is network I/O, so it belongs on the worker, not the GUI thread."""
    stack = _micromanager_stack(tmp_path)
    worker = widget_mod._load_worker("", str(stack))
    df, image, how, acquisition, _raw = worker.work()
    assert df is None and image is not None and how
    assert acquisition["values"]["fps"] == pytest.approx(1000.0 / 31.365, abs=0.1)
