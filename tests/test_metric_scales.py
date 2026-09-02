"""What "distance travelled" measures, and the scale it is shown on.

Distance travelled is the *path length*: every step of the trajectory added up.
It is not the net start-to-end displacement, and it is not squared - which
matters because those three answers differ by orders of magnitude for the same
trajectory, and only the path length grows without bound as a molecule wanders.

That unbounded growth is why it is shown logarithmically by default. Across a
population these quantities span decades, and a linear axis puts nearly every
trajectory in the first bin while devoting the rest of the plot to the handful
of longest ones. The scale is a display choice and switchable, but it is not
only a display choice: the same setting spreads the colours on the trajectories.
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
from test_render_widget import _pump_until  # noqa: E402

PIXEL_SIZE_NM = 100.0


def _widget_with(tracks, fps=10.0):
    widget = make_widget()
    widget.pixel_size_box.setValue(PIXEL_SIZE_NM)
    widget.fps_box.setValue(fps)
    widget.tracks = pd.DataFrame(tracks)
    widget._start_fit_free_metrics_worker()
    assert _pump_until(lambda: widget._track_distance_cache is not None), \
        "the distance/duration computation never finished"
    return widget


# --- what the number means ----------------------------------------------------


def test_distance_travelled_is_the_path_length_not_the_net_displacement():
    """A molecule that goes out and comes back has travelled, but not moved.

    Ten steps of one camera pixel out and ten back is a path of 20 px - 2 µm at
    100 nm/px - and a net displacement of zero. Reporting the net displacement
    here would call that trajectory motionless.
    """
    steps = list(range(11)) + list(range(9, -1, -1))
    tracks = [{"particle": 0, "frame": f, "x": float(x), "y": 0.0}
              for f, x in enumerate(steps)]
    widget = _widget_with(tracks)

    distance = widget._track_distance_cache[0]
    assert distance == pytest.approx(20 * PIXEL_SIZE_NM / 1000.0)   # 2.0 µm
    net = abs(steps[-1] - steps[0]) * PIXEL_SIZE_NM / 1000.0
    assert net == pytest.approx(0.0)
    assert distance > net


def test_it_is_a_length_not_a_squared_length():
    """A 3-4-5 step is 5 px of path, not 25."""
    tracks = [{"particle": 0, "frame": 0, "x": 0.0, "y": 0.0},
              {"particle": 0, "frame": 1, "x": 3.0, "y": 4.0}]
    widget = _widget_with(tracks)
    assert widget._track_distance_cache[0] == pytest.approx(5 * PIXEL_SIZE_NM / 1000.0)


def test_it_adds_up_every_step_rather_than_the_straight_line():
    """An L-shaped path is longer than the diagonal that closes it."""
    tracks = [{"particle": 0, "frame": 0, "x": 0.0, "y": 0.0},
              {"particle": 0, "frame": 1, "x": 3.0, "y": 0.0},
              {"particle": 0, "frame": 2, "x": 3.0, "y": 4.0}]
    widget = _widget_with(tracks)
    assert widget._track_distance_cache[0] == pytest.approx(7 * PIXEL_SIZE_NM / 1000.0)


# --- the scale it is shown on -------------------------------------------------


def _spread_tracks(n=40):
    """Trajectories whose path lengths span three orders of magnitude."""
    rows = []
    for pid, step in enumerate(np.geomspace(0.05, 50.0, n)):
        for frame in range(10):
            rows.append({"particle": pid, "frame": frame,
                         "x": float(frame * step), "y": 0.0})
    return rows


def test_distance_is_shown_logarithmically_by_default():
    widget = make_widget()
    assert widget._metric_use_log["distance"] is True
    assert widget._metric_hist_widgets["distance"]["log_box"].isChecked()


def test_duration_stays_linear_because_it_is_bounded():
    """It cannot exceed the acquisition, so it has no decades to spread."""
    widget = make_widget()
    assert widget._metric_use_log["duration"] is False
    assert not widget._metric_hist_widgets["duration"]["log_box"].isChecked()


def test_the_tick_box_switches_it_back_to_linear():
    widget = _widget_with(_spread_tracks())
    axes = widget._metric_hist_widgets["distance"]["figure"].axes[0]
    assert axes.get_xscale() == "log"

    widget._metric_hist_widgets["distance"]["log_box"].setChecked(False)
    assert widget._metric_use_log["distance"] is False
    axes = widget._metric_hist_widgets["distance"]["figure"].axes[0]
    assert axes.get_xscale() == "linear"

    widget._metric_hist_widgets["distance"]["log_box"].setChecked(True)
    assert widget._metric_hist_widgets["distance"]["figure"].axes[0].get_xscale() == "log"


def test_the_scale_sets_the_colours_too_not_only_the_plot():
    """`_metric_norm_range` reads the same setting, so a colour keeps meaning
    the same thing in the viewer as on the histogram."""
    widget = _widget_with(_spread_tracks())
    lo, hi, use_log = widget._metric_norm_range("distance")
    assert use_log is True

    values = np.array([lo, np.sqrt(lo * hi), hi])
    log_norm = widget._normalize_metric("distance", values)
    widget._metric_hist_widgets["distance"]["log_box"].setChecked(False)
    linear_norm = widget._normalize_metric("distance", values)

    # the geometric midpoint sits mid-colormap on a log scale, and far down it
    # on a linear one - the whole reason for offering the choice
    assert log_norm[1] == pytest.approx(0.5, abs=0.05)
    assert linear_norm[1] < 0.2


def test_the_chosen_scale_is_recorded_with_the_run(tmp_path):
    widget = _widget_with(_spread_tracks())
    widget._metric_hist_widgets["distance"]["log_box"].setChecked(False)
    metadata = widget._collect_metadata(None)
    assert metadata["metric_histogram_display"]["distance"]["log_scale"] is False

    restored = make_widget()
    restored.apply_settings(metadata)
    assert restored._metric_use_log["distance"] is False
    assert not restored._metric_hist_widgets["distance"]["log_box"].isChecked()


# --- end-to-end displacement and straightness ---------------------------------


def _straight_line(n_steps=20):
    return [{"particle": 0, "frame": f, "x": float(f), "y": 0.0}
            for f in range(n_steps + 1)]


def _random_walk(n_steps=400, seed=0):
    rng = np.random.default_rng(seed)
    xy = np.cumsum(rng.normal(0, 1.0, size=(n_steps + 1, 2)), axis=0)
    return [{"particle": 0, "frame": f, "x": float(x), "y": float(y)}
            for f, (x, y) in enumerate(xy)]


def test_end_to_end_is_where_it_ended_not_how_far_it_went():
    """Out and back: a long path, and no displacement at all."""
    steps = list(range(11)) + list(range(9, -1, -1))
    widget = _widget_with([{"particle": 0, "frame": f, "x": float(x), "y": 0.0}
                           for f, x in enumerate(steps)])
    assert widget._track_distance_cache[0] == pytest.approx(20 * PIXEL_SIZE_NM / 1000.0)
    assert widget._track_net_cache[0] == pytest.approx(0.0)


def test_a_straight_trajectory_has_a_straightness_of_one():
    widget = _widget_with(_straight_line())
    assert widget._track_straightness_cache[0] == pytest.approx(1.0)
    assert widget._track_net_cache[0] == pytest.approx(
        widget._track_distance_cache[0])


def test_a_random_walk_is_nowhere_near_straight():
    """Roughly 1/sqrt(N), which is the whole basis for telling the two apart.

    A fast diffuser and a directed molecule can share an end-to-end
    displacement; they cannot share a straightness.
    """
    widget = _widget_with(_random_walk(n_steps=400))
    straightness = widget._track_straightness_cache[0]
    assert 0.0 < straightness < 0.3
    assert widget._track_net_cache[0] < widget._track_distance_cache[0]


def test_a_trajectory_that_never_moved_has_no_direction_to_be_straight_in():
    widget = _widget_with([{"particle": 0, "frame": f, "x": 5.0, "y": 5.0}
                           for f in range(6)])
    assert widget._track_distance_cache[0] == pytest.approx(0.0)
    assert np.isnan(widget._track_straightness_cache[0])


def test_both_can_be_coloured_by_and_histogrammed():
    widget = _widget_with(_straight_line())
    for key, choice in (("net", "End-to-end displacement"),
                        ("straightness", "Straightness (directed vs diffusive)")):
        assert key in widget._metric_hist_widgets
        assert key in widget._metric_bound_boxes
        widget.color_metric_box.setCurrentText(choice)
        assert widget._current_metric_key() == key


def test_straightness_is_linear_and_end_to_end_is_logarithmic():
    widget = make_widget()
    assert widget._metric_use_log["net"] is True          # spans decades
    assert widget._metric_use_log["straightness"] is False  # a ratio in [0, 1]


def test_both_reach_the_exported_table():
    widget = _widget_with(_straight_line())
    frame = widget._track_metrics_frame()
    assert "net_displacement_um" in frame.columns
    assert "straightness" in frame.columns
    assert frame["straightness"].iloc[0] == pytest.approx(1.0)


# --- the range controls themselves --------------------------------------------


def test_a_range_control_steps_by_a_fraction_of_its_own_value():
    """One notch of the wheel must move the digit being looked at.

    With a fixed step of 1.0, a diffusion bound of 4e-5 jumps to 1.00004 - the
    control moves by twenty thousand times its own value and the digit that
    matters cannot be reached at all.
    """
    widget = make_widget()
    box = widget.d_min_box
    for value, expected_step in ((4e-5, 1e-6), (0.0032, 1e-4), (3.26, 0.1)):
        box.setValue(value)
        box.stepBy(1)
        assert box.value() - value == pytest.approx(expected_step, rel=0.01)


@pytest.mark.parametrize("attr", [
    "d_min_box", "d_max_box", "dist_min_box", "dist_max_box",
    "net_min_box", "net_max_box", "dur_min_box", "dur_max_box",
])
def test_every_metric_bound_steps_adaptively(attr):
    widget = make_widget()
    box = getattr(widget, attr)
    box.setValue(0.5)
    box.stepBy(1)
    assert box.value() - 0.5 == pytest.approx(0.01, rel=0.01)


def test_the_view_boxes_can_hold_the_bounds_they_mirror():
    """"follow filter" copies bound to view; too few decimals rounds to zero.

    A D bound of 4e-5 arriving in a four-decimal box became a flat 0, and the
    log axis then started from nothing and spent eight empty decades getting
    back to the data.
    """
    widget = make_widget()
    state = widget._metric_hist_widgets["D"]
    assert state["view_min_box"].decimals() >= widget.d_min_box.decimals()

    state["follow_box"].setChecked(True)
    widget.d_min_box.setValue(4e-5)
    widget.d_max_box.setValue(3.26)
    assert state["view_min_box"].value() == pytest.approx(4e-5)


def test_a_zero_lower_bound_does_not_blow_the_log_axis_open():
    """Zero is the natural bound for a length and the one a log axis cannot show.

    Falling back to a fixed 1e-9 spent most of the axis on empty decades, which
    is what made every trajectory look jammed against the right-hand edge.
    """
    widget = _widget_with(_spread_tracks())
    widget.dist_min_box.setValue(0.0)
    lo, hi, use_log = widget._metric_norm_range("distance")

    smallest = min(v for v in widget._track_distance_cache.values() if v > 0)
    assert lo == pytest.approx(smallest)     # the data's own floor, not 1e-9
    assert lo > 1e-9

    # and an explicitly chosen positive bound is still obeyed
    widget.dist_min_box.setValue(smallest * 10)
    lo_explicit, _hi, _log = widget._metric_norm_range("distance")
    assert lo_explicit == pytest.approx(smallest * 10)
