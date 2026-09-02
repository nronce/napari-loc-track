"""The clock on the canvas, above the scale bar.

A screen-recorded movie carries no metadata, so whatever the viewer shows is all
a reader of that movie will ever have. The clock is therefore part of the
picture rather than part of the plugin's panel: it reads off the dims slider, so
it is right whether the frame moved by dragging or by the play button, and it
sits with the scale bar so the two travel together into any recording.

Run against a real `ViewerModel` - the overlay, its position and its stacking
order relative to the scale bar are exactly what is being checked.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_widget_interaction import ensure_qapp  # noqa: E402

_WIDGETS = []


def _widget(fps=32.0):
    from napari.components import ViewerModel

    ensure_qapp()
    widget = widget_mod.LocalizationTrackingWidget(ViewerModel())
    _WIDGETS.append(widget)
    widget.fps_box.setValue(fps)
    return widget


def _loaded(fps=32.0, frames=200):
    widget = _widget(fps)
    widget._on_load_finished(
        (None, np.zeros((frames, 32, 32), np.uint16), "decoded", None),
        "", "stack.tif")
    return widget


def test_the_clock_is_showing_once_an_image_loads():
    widget = _loaded()
    assert widget.viewer.text_overlay.visible
    assert widget.viewer.text_overlay.text


def test_it_sits_above_the_scale_bar_not_on_top_of_it():
    """napari tiles overlays sharing a corner, working outwards in `order`."""
    widget = _loaded()
    viewer = widget.viewer
    assert viewer.text_overlay.position == "bottom_right"
    assert viewer.scale_bar.position == "bottom_right"
    # the bar takes the canvas edge, the clock stacks directly above it
    assert viewer.scale_bar.order < viewer.text_overlay.order


def test_the_time_follows_the_frame_slider():
    """However the slider moved - dragged, or run by the play button."""
    widget = _loaded(fps=32.0)
    for frame, expected in ((0, "0.0 s"), (32, "1.0 s"), (160, "5.0 s")):
        widget.viewer.dims.set_current_step(0, frame)
        assert widget.viewer.text_overlay.text.startswith(expected)
        assert f"frame {frame}" in widget.viewer.text_overlay.text


def test_the_time_follows_the_acquisition_rate():
    """The same frame is a different moment at a different frame rate."""
    widget = _loaded(fps=32.0)
    widget.viewer.dims.set_current_step(0, 64)
    assert widget.viewer.text_overlay.text.startswith("2.0 s")
    widget.fps_box.setValue(16.0)
    assert widget.viewer.text_overlay.text.startswith("4.0 s")


def test_a_long_acquisition_is_clocked_in_minutes_throughout():
    """The format is chosen from the whole run, not from the current frame.

    A label that flips from "9.4 s" to "01:12" partway along the slider is
    unreadable, so the units are settled once from how long the run lasts.
    """
    widget = _loaded(fps=10.0, frames=3000)  # 300 s in total
    widget.viewer.dims.set_current_step(0, 50)  # 5 s in, early on
    assert ":" in widget.viewer.text_overlay.text
    widget.viewer.dims.set_current_step(0, 2000)
    assert ":" in widget.viewer.text_overlay.text


def test_the_clock_survives_having_no_stack_loaded():
    widget = _widget()
    widget._update_time_overlay()  # must not raise with an empty viewer
    assert isinstance(widget.viewer.text_overlay.text, str)
