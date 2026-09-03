"""Hovering a trajectory says what it is and what has been measured about it.

Trajectories are drawn as anonymous coloured lines. The colour carries one
metric and the legend carries its scale, but answering "which track is that, and
what is its D?" otherwise means exporting the metrics table and matching
coordinates by hand. The tooltip is the way back from a line on screen to the
trajectory it belongs to.

napari asks only the *active* layer for tooltip text, and only when tooltips are
switched on for the viewer at all - which is the part that used to be missing,
because they were switched on by the localizations layer and by nothing else.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_render_widget import _pump_until  # noqa: E402
from test_widget_interaction import ensure_qapp  # noqa: E402

PIXEL_SIZE_NM = 161.0
FPS = 31.882

# Each widget owns a dozen matplotlib canvases, and letting Qt collect them mid
# session crashes the interpreter, so they are held for the run.
_WIDGETS = []


def _widget():
    from napari.components import ViewerModel

    ensure_qapp()
    widget = widget_mod.LocalizationTrackingWidget(ViewerModel())
    _WIDGETS.append(widget)
    widget.pixel_size_box.setValue(PIXEL_SIZE_NM)
    widget.fps_box.setValue(FPS)
    return widget


def _straight_tracks(n_particles=3, n_points=10, first_frame=0, frame_step=1):
    """Well-separated horizontal lines, one per particle."""
    rows = []
    for pid in range(n_particles):
        for i in range(n_points):
            rows.append({
                "particle": pid,
                "frame": first_frame + pid * 20 + i * frame_step,
                "x": 10.0 * i,
                "y": 40.0 * pid,
            })
    return pd.DataFrame(rows)


def _showing(tracks, all_tracks=False):
    widget = _widget()
    widget.tracks = tracks
    widget.show_points_box.setChecked(False)
    widget.show_tracks_box.setChecked(True)
    widget.show_all_tracks_box.setChecked(all_tracks)
    widget.render_overlay()
    return widget


def _hover(widget, frame, y_px, x_px, layer_name=widget_mod.TRACKS_LAYER_NAME):
    """Ask a layer for its tooltip at a point, the way napari does.

    The position is in world coordinates - the layers are scaled to nanometres,
    so a pixel coordinate has to be multiplied through the layer scale exactly
    as the cursor position is.
    """
    layer = widget.viewer.layers[layer_name]
    if layer.ndim == 2:
        position = np.array([y_px, x_px]) * np.asarray(layer.scale[-2:])
        dims_displayed = [0, 1]
    else:
        position = np.array([frame, y_px, x_px]) * np.asarray(layer.scale[-3:])
        dims_displayed = [1, 2]
    return layer._get_tooltip_text(position, dims_displayed=dims_displayed, world=True)


# --- the part that was broken -------------------------------------------------


def test_tooltips_are_on_for_a_session_with_trajectories_and_no_localizations():
    """The case that never worked: trajectories loaded back from a previous run.

    The text was computed correctly all along and napari never asked for it,
    because `tooltip.visible` was set by the localizations layer alone.
    """
    widget = _showing(_straight_tracks())
    assert widget_mod.POINTS_LAYER_NAME not in widget.viewer.layers
    assert widget.viewer.tooltip.visible is True


def test_the_static_all_trajectories_layer_turns_them_on_too():
    widget = _widget()
    widget.tracks = _straight_tracks()
    widget.show_points_box.setChecked(False)
    widget.show_tracks_box.setChecked(False)
    widget.show_all_tracks_box.setChecked(True)
    widget.render_overlay()
    assert widget.viewer.tooltip.visible is True


# --- what it says -------------------------------------------------------------


def test_hovering_a_trajectory_names_it():
    widget = _showing(_straight_tracks())
    assert "track 0" in _hover(widget, 3, 0.0, 30.0)
    assert "track 1" in _hover(widget, 23, 40.0, 30.0)


def test_it_reports_the_frame_the_trajectory_starts_on():
    """Track 2 begins at frame 40 here, not at zero, and that is the number
    that lines the trajectory up with the slider and with the raw stack."""
    widget = _showing(_straight_tracks())
    text = _hover(widget, 43, 80.0, 30.0)
    assert "starts at frame 40" in text
    assert "ends at 49" in text


def test_the_frames_it_reports_are_the_ones_on_the_slider():
    """Whatever shift was applied to line the table up with the stack is
    already in `self.tracks`, so the tooltip and the slider cannot disagree."""
    widget = _showing(_straight_tracks(first_frame=5))
    layer = widget.viewer.layers[widget_mod.TRACKS_LAYER_NAME]
    assert int(layer.data[:, 1].min()) == 5
    assert "starts at frame 5" in _hover(widget, 8, 0.0, 30.0)


def test_a_gappy_trajectory_says_how_many_points_it_actually_has():
    """The linker bridges gaps up to the memory setting, so a trajectory can be
    absent from frames it spans - worth seeing before trusting its D."""
    widget = _showing(_straight_tracks(n_points=6, frame_step=3))
    text = _hover(widget, 6, 0.0, 20.0)
    assert "spans 16 frames, 6 points" in text


def test_hovering_empty_canvas_says_nothing():
    widget = _showing(_straight_tracks())
    assert _hover(widget, 3, 5000.0, 5000.0) == ""


# --- the measured properties --------------------------------------------------


def _with_metrics(widget):
    widget._start_fit_free_metrics_worker()
    assert _pump_until(lambda: widget._track_distance_cache is not None), \
        "the distance/duration computation never finished"
    return widget


def test_a_metric_that_has_not_been_computed_is_left_out_entirely():
    """An absent line means "not run". A zero would mean "measured, and zero"."""
    widget = _showing(_straight_tracks())
    text = _hover(widget, 3, 0.0, 30.0)
    for absent in ("duration", "D ", "distance travelled", "end-to-end", "straightness"):
        assert absent not in text


def test_the_fit_free_metrics_appear_once_they_are_computed():
    widget = _with_metrics(_showing(_straight_tracks()))
    text = _hover(widget, 3, 0.0, 30.0)
    assert "duration" in text
    assert "distance travelled" in text and "µm" in text
    assert "end-to-end" in text
    assert "straightness 1.00" in text          # these tracks are straight lines


def test_diffusion_appears_with_its_uncertainty():
    """D is a quarter of the fitted slope, so its error is a quarter of the
    slope error - the same arithmetic the MSD legend uses, so they agree."""
    widget = _with_metrics(_showing(_straight_tracks()))
    widget._track_diffusion_cache = {0: 0.2831}
    widget._track_msd_cache = {0: (None, None, 4 * 0.2831, 0.0, 4 * 0.0212)}
    text = _hover(widget, 3, 0.0, 30.0)
    assert "D 0.2831 ± 0.021 µm²/s" in text


def test_a_trajectory_too_short_to_pin_its_slope_down_shows_no_error():
    """Rather than a fabricated zero, which would read as a perfect measurement."""
    widget = _with_metrics(_showing(_straight_tracks()))
    widget._track_diffusion_cache = {0: 0.2831}
    widget._track_msd_cache = {0: (None, None, 4 * 0.2831, 0.0, float("nan"))}
    text = _hover(widget, 3, 0.0, 30.0)
    assert "D 0.2831 µm²/s" in text
    assert "±" not in text


def test_diffusion_shows_even_when_the_msd_fit_was_not_kept():
    widget = _with_metrics(_showing(_straight_tracks()))
    widget._track_diffusion_cache = {0: 0.5}
    widget._track_msd_cache = None
    assert "D 0.5 µm²/s" in _hover(widget, 3, 0.0, 30.0)


def test_a_trajectory_without_a_d_shows_the_rest_of_its_properties():
    """D is only computed for trajectories long enough for the MSD fit."""
    widget = _with_metrics(_showing(_straight_tracks()))
    widget._track_diffusion_cache = {0: 0.5}
    text = _hover(widget, 23, 40.0, 30.0)   # track 1, no D
    assert "track 1" in text
    assert "D " not in text
    assert "distance travelled" in text


# --- the static layer answers the same way ------------------------------------


def test_the_static_all_trajectories_layer_reports_the_same_properties():
    """It is 2D and always visible, so it is what the cursor usually meets."""
    widget = _with_metrics(_showing(_straight_tracks(), all_tracks=True))
    text = _hover(widget, 0, 40.0, 30.0,
                  layer_name=widget_mod.ALL_TRACKS_LAYER_NAME)
    assert "track 1" in text
    assert "starts at frame 20" in text
    assert "distance travelled" in text
