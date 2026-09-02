"""Building up from a chosen frame, for trajectories and for the movie.

Both answer the same question - "show me what has happened since here" - and
both default to the beginning, so leaving them alone changes nothing. The point
of being able to move the start is that the interesting part of an acquisition
rarely begins at frame 0: there is usually a stretch of bleaching or drift first
that a build-up from frame 0 spends its whole dynamic range on.

The two are deliberately separate controls, one per layer, because the frame
worth starting a reconstruction from is not usually the frame worth starting the
trajectories from.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)
render = widget_mod.smlm_render

from test_widget_interaction import make_widget  # noqa: E402
from test_render_widget import _loaded, _render_movie, FRAMES, PIXEL_SIZE_NM  # noqa: E402

TRACKS = widget_mod.TRACKS_LAYER_NAME


# --- trajectories -------------------------------------------------------------


def _tracked(n_particles=5, n_points=40):
    widget = make_widget()
    widget.tracks = pd.DataFrame([
        {"particle": pid, "frame": frame, "x": 10.0 + frame, "y": 20.0 + pid}
        for pid in range(n_particles) for frame in range(n_points)
    ])
    widget.render_overlay()
    return widget


def test_trajectories_do_not_accumulate_until_asked():
    widget = _tracked()
    assert not widget.traj_accumulate_box.isChecked()
    assert not widget.traj_start_frame_box.isEnabled()
    assert widget.viewer.layers[TRACKS].tail_length == widget._tracks_full_span


def test_the_trail_grows_to_reach_back_to_the_chosen_start():
    """That is what accumulating *is*: the trail is however far past the start."""
    widget = _tracked()
    widget.traj_accumulate_box.setChecked(True)
    widget.traj_start_frame_box.setValue(10)

    for frame, expected in ((10, 1), (20, 10), (35, 25)):
        widget.viewer.dims.current_step = (frame, 0, 0)
        widget._on_current_frame_changed()
        assert widget.viewer.layers[TRACKS].tail_length == expected


def test_the_trail_never_collapses_to_nothing_before_the_start():
    widget = _tracked()
    widget.traj_accumulate_box.setChecked(True)
    widget.traj_start_frame_box.setValue(30)
    widget.viewer.dims.current_step = (5, 0, 0)   # before the start frame
    widget._on_current_frame_changed()
    assert widget.viewer.layers[TRACKS].tail_length >= 1


def test_a_fixed_trail_and_an_accumulating_one_are_not_both_live():
    """Two answers to one question, so only one control is enabled at a time."""
    widget = _tracked()
    widget.traj_fade_box.setValue(15)
    assert widget.traj_fade_box.isEnabled()

    widget.traj_accumulate_box.setChecked(True)
    assert widget.traj_start_frame_box.isEnabled()
    assert not widget.traj_fade_box.isEnabled()

    widget.traj_accumulate_box.setChecked(False)
    assert widget.traj_fade_box.isEnabled()
    assert not widget.traj_start_frame_box.isEnabled()
    assert widget.viewer.layers[TRACKS].tail_length == 15  # the fixed trail returns


def test_the_status_line_says_which_mode_is_running():
    widget = _tracked()
    widget.fps_box.setValue(10.0)
    widget.traj_accumulate_box.setChecked(True)
    widget.traj_start_frame_box.setValue(50)
    text = widget.traj_fade_status.text()
    assert "build up from frame 50" in text
    assert "5 s" in text                      # 50 frames at 10 fps


def test_a_rebuilt_layer_keeps_accumulating():
    widget = _tracked()
    widget.traj_accumulate_box.setChecked(True)
    widget.traj_start_frame_box.setValue(10)
    widget.viewer.dims.current_step = (30, 0, 0)
    widget._on_current_frame_changed()
    widget.render_overlay()                   # a full rebuild, not a style change
    assert widget.viewer.layers[TRACKS].tail_length == 20


# --- the super-resolved movie -------------------------------------------------


def test_a_movie_starts_at_the_first_localization_by_default():
    widget = _loaded(3000, image_shape=(FRAMES, 48, 80))
    widget.render_frames_per_box.setValue(50)
    assert widget.render_start_frame_box.value() == 0
    _render_movie(widget)
    info = widget._render_movie_info
    assert info["first_raw_frame"] == int(widget.df_filtered["frame"].min())


def test_a_later_start_drops_what_came_before_it():
    """The localizations before the start fall outside every group.

    A cumulative movie can then build up from the moment something starts
    happening, rather than spending its dynamic range on the bleaching at the
    front of the acquisition.
    """
    widget = _loaded(3000, image_shape=(FRAMES, 48, 80))
    widget.render_frames_per_box.setValue(50)
    widget.render_grouping_box.setCurrentIndex(
        widget.render_grouping_box.findData("cumulative"))
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("histogram"))
    widget.render_scalebar_box.setChecked(False)
    widget.render_timestamp_box.setChecked(False)

    whole = _render_movie(widget)
    widget.render_start_frame_box.setValue(FRAMES // 2)
    later = _render_movie(widget)

    assert widget._render_movie_info["first_raw_frame"] == FRAMES // 2
    # half the acquisition, so about half the movie frames and far fewer counts
    assert len(later) < len(whole)
    assert float(later[-1].sum()) < float(whole[-1].sum())


def test_the_movie_lines_up_with_the_stack_from_its_new_start():
    widget = _loaded(3000, image_shape=(FRAMES, 48, 80))
    per_group = 25
    widget.render_frames_per_box.setValue(per_group)
    widget.render_start_frame_box.setValue(60)
    _render_movie(widget)

    layer = widget.viewer.layers[widget_mod.RENDER_MOVIE_LAYER_NAME]
    # still placed where each window closes, now counted from the chosen start
    assert layer.translate[0] == 60 + per_group - 1


def test_a_start_past_the_end_is_refused_rather_than_rendered_empty():
    widget = _loaded(500, image_shape=(FRAMES, 48, 80))
    widget.render_start_frame_box.setValue(FRAMES * 10)
    widget._render_movie = None
    widget.render_smlm_movie()
    assert widget._render_movie is None
    assert "after the last localization" in widget.log_box.toPlainText()


def test_the_two_starts_are_independent():
    """One is about the reconstruction, the other about the trajectories."""
    widget = _loaded(500, image_shape=(FRAMES, 48, 80))
    widget.render_start_frame_box.setValue(40)
    widget.traj_accumulate_box.setChecked(True)
    widget.traj_start_frame_box.setValue(10)
    assert widget.render_start_frame_box.value() == 40
    assert widget.traj_start_frame_box.value() == 10
