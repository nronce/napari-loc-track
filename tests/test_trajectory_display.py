"""How trajectories are drawn: how long a trail they leave, and what colours it.

Two defaults are deliberate here. A trajectory keeps its whole trail unless a
length is asked for, and colours go on *identity* rather than on any measurement
- neighbouring tracks being tellable apart is what the display is for, and a
colour scale spent on time by default takes that away for something the frame
slider already shows. Time is offered as one metric among D, distance and
duration, for when it is the thing being looked at.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_widget_interaction import make_widget  # noqa: E402

TRACKS = widget_mod.TRACKS_LAYER_NAME


def _tracked(n_particles=6, n_points=20, fps=32.0):
    """A widget with trajectories already in it, without running the linker."""
    widget = make_widget()
    widget.fps_box.setValue(fps)
    rows = []
    for pid in range(n_particles):
        for frame in range(n_points):
            rows.append({"particle": pid, "frame": frame + pid,
                         "x": 10.0 + frame * 0.5, "y": 20.0 + pid})
    widget.tracks = pd.DataFrame(rows)
    widget.render_overlay()
    return widget


# --- the trail ----------------------------------------------------------------


def test_a_trajectory_keeps_its_whole_trail_by_default():
    widget = _tracked()
    assert widget.traj_fade_box.value() == 0
    layer = widget.viewer.layers[TRACKS]
    assert layer.tail_length == widget._tracks_full_span
    assert "whole length" in widget.traj_fade_status.text()


def test_a_trail_length_reaches_the_layer_without_a_rebuild():
    widget = _tracked()
    layer = widget.viewer.layers[TRACKS]
    before = widget.viewer.add_calls

    widget.traj_fade_box.setValue(12)
    assert layer.tail_length == 12
    assert widget.viewer.add_calls == before  # recoloured in place, not rebuilt


def test_zero_means_the_whole_trajectory_not_no_trail():
    """The number cannot say "no limit" on its own, so the box says it in words."""
    widget = _tracked()
    widget.traj_fade_box.setValue(25)
    assert widget.viewer.layers[TRACKS].tail_length == 25
    widget.traj_fade_box.setValue(0)
    assert widget.viewer.layers[TRACKS].tail_length == widget._tracks_full_span
    assert widget.traj_fade_box.specialValueText() == "the whole trajectory"


def test_the_trail_is_reported_in_seconds_as_well_as_frames():
    """It is chosen in seconds even though napari counts it in frames."""
    widget = _tracked(fps=32.0)
    widget.traj_fade_box.setValue(64)
    assert "2 s" in widget.traj_fade_status.text()
    widget.fps_box.setValue(16.0)
    assert "4 s" in widget.traj_fade_status.text()


def test_a_new_trajectory_layer_starts_with_the_chosen_trail():
    widget = _tracked()
    widget.traj_fade_box.setValue(8)
    widget.render_overlay()
    assert widget.viewer.layers[TRACKS].tail_length == 8


# --- the colours --------------------------------------------------------------


def test_trajectories_are_coloured_by_identity_by_default():
    """Many colours, unrelated to time - so two tracks crossing stay distinct."""
    widget = _tracked()
    assert not widget.color_trajectories_box.isChecked()
    layer = widget.viewer.layers[TRACKS]
    assert layer.color_by == "track_id"
    assert layer.colormap == "hsv"


def test_time_is_offered_alongside_the_measured_metrics():
    widget = make_widget()
    offered = [widget.color_metric_box.itemText(i)
               for i in range(widget.color_metric_box.count())]
    assert any(o.startswith("D") for o in offered)
    assert any(o.startswith("Distance") for o in offered)
    assert any(o.startswith("Track duration") for o in offered)
    assert any(o.startswith("Time") for o in offered)


def test_choosing_time_colours_by_when_each_trajectory_appeared():
    widget = _tracked()
    widget.color_trajectories_box.setChecked(True)
    widget.color_metric_box.setCurrentText("Time (frame first seen)")
    assert widget._current_metric_key() == "time"

    layer = widget.viewer.layers[TRACKS]
    assert layer.color_by == "metric_color"
    values = np.asarray(layer.properties["metric_color"], dtype=float)

    # Particle p first appears at frame p, and the scale is stretched over the
    # whole acquisition rather than over the first frames alone - so six tracks
    # that all start early share the early end of the colormap instead of being
    # spread across it, which is what "when did this happen" has to mean.
    last_frame = float(widget.tracks["frame"].max())
    assert values.min() == pytest.approx(0.0)
    assert values.max() == pytest.approx(5.0 / last_frame)
    # and later trajectories really do sit further along the scale
    by_particle = dict(zip(widget._tracks_layer_particles, values))
    assert by_particle[0] < by_particle[3] < by_particle[5]


def test_time_needs_no_computing_unlike_D():
    """It is in the table already, which is why it has no Compute button."""
    widget = _tracked()
    assert widget._track_diffusion_cache in (None, {})  # D was never computed
    cache = widget._metric_cache("time")
    assert cache == {pid: pid for pid in range(6)}  # each starts at its own frame


def test_time_spans_whatever_the_data_covers():
    """No bounds box to read, so the colormap is stretched over the acquisition."""
    widget = _tracked(n_particles=4, n_points=10)
    lo, hi, use_log = widget._metric_norm_range("time")
    assert (lo, hi) == (0.0, float(widget.tracks["frame"].max()))
    assert use_log is False


def test_switching_back_to_identity_colours_restores_them():
    widget = _tracked()
    widget.color_trajectories_box.setChecked(True)
    widget.color_metric_box.setCurrentText("Time (frame first seen)")
    assert widget.viewer.layers[TRACKS].color_by == "metric_color"
    widget.color_trajectories_box.setChecked(False)
    assert widget.viewer.layers[TRACKS].color_by == "track_id"
