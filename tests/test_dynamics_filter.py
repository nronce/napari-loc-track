"""Showing only the trajectories that behaved a certain way - and only their
localizations.

The reconstruction is the point. Colouring a reconstruction by diffusion
coefficient tells you the fast molecules are somewhere in the picture; building
it from the fast molecules alone tells you *where*. So the filter cannot stop at
the trajectory layers: it has to reach the localizations underneath them, and
through those the super-resolved render.

The controls are the min/max already beside each metric, with a tick box that
turns them from a colour scale into a selection. One set of numbers, drawn on
the histogram they came from.
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

PIXEL_SIZE_NM = 100.0
POINTS_PER_TRACK = 6
_WIDGETS = []


def _widget():
    from napari.components import ViewerModel

    ensure_qapp()
    widget = widget_mod.LocalizationTrackingWidget(ViewerModel())
    _WIDGETS.append(widget)
    widget.pixel_size_box.setValue(PIXEL_SIZE_NM)
    widget.fps_box.setValue(10.0)
    return widget


def _paired(n_particles=5, extra_unlinked=7):
    """Localizations and the trajectories over them, built together.

    Together on purpose: the filter has to get from a trajectory back to the
    localizations that made it, and it does that by matching frame and position.
    Building both from the same numbers is what a real link produces.

    `extra_unlinked` localizations belong to no trajectory - the linker discards
    plenty - and must survive or vanish with the filter just like the rest.
    """
    rows, track_rows = [], []
    for pid in range(n_particles):
        for i in range(POINTS_PER_TRACK):
            x_px, y_px = 10.0 * i + 100.0 * pid, 7.0 * pid
            rows.append({"frame": i, "x [nm]": x_px * PIXEL_SIZE_NM,
                         "y [nm]": y_px * PIXEL_SIZE_NM,
                         "sigma [nm]": 120.0, "intensity [photon]": 500.0})
            track_rows.append({"particle": pid, "frame": i, "x": x_px, "y": y_px})
    rng = np.random.default_rng(0)
    for _ in range(extra_unlinked):
        rows.append({"frame": int(rng.integers(0, POINTS_PER_TRACK)),
                     "x [nm]": float(rng.uniform(60000, 70000)),
                     "y [nm]": float(rng.uniform(60000, 70000)),
                     "sigma [nm]": 120.0, "intensity [photon]": 500.0})
    return pd.DataFrame(rows), pd.DataFrame(track_rows)


def _loaded(n_particles=5, **caches):
    widget = _widget()
    locs, tracks = _paired(n_particles)
    widget._ingest_localization_dataframe(locs, "loaded", True)
    widget.tracks = tracks
    widget._invalidate_track_filter()
    for key, values in caches.items():
        setattr(widget, widget_mod.METRIC_CACHE_ATTR[key], values)
    widget.render_overlay()
    return widget


def _linked_particles(widget):
    return sorted(set(widget._displayed_tracks()["particle"].tolist()))


# --- the mapping the whole thing rests on -------------------------------------


def test_every_localization_finds_the_trajectory_it_belongs_to():
    widget = _loaded()
    particles = widget._localization_particles()
    assert len(particles) == len(widget.df_filtered)

    linked = particles[particles >= 0]
    assert len(linked) == 5 * POINTS_PER_TRACK
    # and each one to the right trajectory, not merely to some trajectory
    for row, pid in zip(widget.df_filtered.itertuples(), particles):
        if pid < 0:
            continue
        expected = int(getattr(row, "_2") / PIXEL_SIZE_NM // 100)
        assert pid == expected


def test_localizations_that_belong_to_no_trajectory_are_marked_as_such():
    widget = _loaded()
    particles = widget._localization_particles()
    assert (particles < 0).sum() == 7


def test_the_mapping_survives_trajectories_being_read_back_from_a_csv(tmp_path):
    """Auto-loading a previous run is the common case, and it puts a float
    round-trip between the trajectories and the localizations they came from."""
    widget = _widget()
    locs, tracks = _paired()
    # A coordinate with a full mantissa, so a lossy round-trip would show up.
    tracks.loc[0, "x"] = 12.345678901234567
    locs.loc[0, "x [nm]"] = 12.345678901234567 * PIXEL_SIZE_NM
    path = tmp_path / "trajectories.csv"
    tracks.to_csv(path, index=False)

    widget._ingest_localization_dataframe(locs, "loaded", True)
    widget.tracks = pd.read_csv(path)
    widget._invalidate_track_filter()
    assert (widget._localization_particles() >= 0).sum() == 5 * POINTS_PER_TRACK


# --- selecting ----------------------------------------------------------------


def test_nothing_is_filtered_until_a_box_is_ticked():
    widget = _loaded(distance=dict.fromkeys(range(5), 1.0))
    assert widget._passing_particles() is None
    assert len(widget._displayed_tracks()) == len(widget.tracks)
    assert len(widget._displayed_localizations()) == len(widget.df_filtered)


def test_ticking_a_box_keeps_only_the_trajectories_inside_the_range():
    widget = _loaded(distance=dict(zip(range(5), [0.1, 0.5, 1.0, 5.0, 20.0])))
    widget.dist_min_box.setValue(0.4)
    widget.dist_max_box.setValue(6.0)
    widget.distance_filter_box.setChecked(True)
    assert _linked_particles(widget) == [1, 2, 3]


def test_the_localizations_of_the_discarded_trajectories_go_with_them():
    widget = _loaded(distance=dict(zip(range(5), [0.1, 0.5, 1.0, 5.0, 20.0])))
    widget.dist_min_box.setValue(0.4)
    widget.dist_max_box.setValue(6.0)
    widget.distance_filter_box.setChecked(True)

    shown = widget._displayed_localizations()
    assert len(shown) == 3 * POINTS_PER_TRACK
    # including the ones that belonged to no trajectory at all: they are not
    # trajectories inside the range, so they are not shown
    assert len(shown) < len(widget.df_filtered)


def test_the_reconstruction_is_built_from_those_localizations_alone():
    """The reason the feature exists: a render of the molecules that behaved
    this way, rather than a render of all of them with a colour hint."""
    widget = _loaded(distance=dict(zip(range(5), [0.1, 0.5, 1.0, 5.0, 20.0])))
    before, _y = widget._render_positions_px()
    widget.dist_min_box.setValue(0.4)
    widget.dist_max_box.setValue(6.0)
    widget.distance_filter_box.setChecked(True)

    after, _y = widget._render_positions_px()
    assert before.size == len(widget.df_filtered)
    assert after.size == 3 * POINTS_PER_TRACK
    assert widget._render_frames().size == after.size


def test_several_ranges_combine_as_and():
    widget = _loaded(
        distance=dict(zip(range(5), [0.1, 0.5, 1.0, 5.0, 20.0])),
        straightness=dict(zip(range(5), [0.9, 0.9, 0.1, 0.95, 0.1])),
    )
    widget.dist_min_box.setValue(0.4)
    widget.dist_max_box.setValue(6.0)
    widget.distance_filter_box.setChecked(True)
    assert _linked_particles(widget) == [1, 2, 3]

    widget.straight_min_box.setValue(0.5)
    widget.straight_max_box.setValue(1.0)
    widget.straightness_filter_box.setChecked(True)
    assert _linked_particles(widget) == [1, 3]      # in both ranges, not either


def test_a_trajectory_with_no_value_for_a_filtered_metric_is_excluded():
    """D is only fitted for trajectories long enough to support it, so
    filtering on D also drops the short ones - the count says so out loud."""
    widget = _loaded(D={0: 1.0, 1: 2.0})       # 2, 3, 4 were never fitted
    widget.d_min_box.setValue(0.0)
    widget.d_max_box.setValue(100.0)
    widget.d_filter_box.setChecked(True)

    assert _linked_particles(widget) == [0, 1]
    assert "3 have no value for a filtered metric" in widget.track_filter_label.text()


def test_a_range_nothing_satisfies_empties_the_view_rather_than_filling_it():
    """An empty selection and no selection at all are different things, and
    confusing them would show everything at the moment it should show nothing."""
    widget = _loaded(distance=dict.fromkeys(range(5), 1.0))
    widget.dist_min_box.setValue(50.0)
    widget.dist_max_box.setValue(60.0)
    widget.distance_filter_box.setChecked(True)

    assert widget._passing_particles() == set()
    assert widget._displayed_tracks().empty
    assert widget._displayed_localizations().empty
    assert widget._render_positions_px() == (None, None)


def test_an_empty_render_blames_the_filter_rather_than_the_data():
    """"Load some localizations" would send the user looking for data that is
    sitting right there behind a range they set."""
    widget = _loaded(distance=dict.fromkeys(range(5), 1.0))
    widget.dist_min_box.setValue(50.0)
    widget.dist_max_box.setValue(60.0)
    widget.distance_filter_box.setChecked(True)
    widget.log_box.clear()

    assert widget._render_inputs() is None
    assert "dynamics filter is keeping no localizations" in widget.log_box.toPlainText()


def test_moving_a_bound_moves_the_selection_with_it():
    widget = _loaded(distance=dict(zip(range(5), [0.1, 0.5, 1.0, 5.0, 20.0])))
    widget.dist_min_box.setValue(0.0)
    widget.dist_max_box.setValue(100.0)
    widget.distance_filter_box.setChecked(True)
    assert _linked_particles(widget) == [0, 1, 2, 3, 4]

    widget.dist_min_box.setValue(2.0)
    assert _linked_particles(widget) == [3, 4]


def test_clearing_brings_everything_back():
    widget = _loaded(distance=dict(zip(range(5), [0.1, 0.5, 1.0, 5.0, 20.0])))
    widget.dist_min_box.setValue(2.0)
    widget.distance_filter_box.setChecked(True)
    assert len(widget._displayed_tracks()) < len(widget.tracks)

    widget.clear_track_filters()
    assert widget._passing_particles() is None
    assert len(widget._displayed_tracks()) == len(widget.tracks)
    assert len(widget._displayed_localizations()) == len(widget.df_filtered)
    assert not widget.clear_track_filter_button.isEnabled()


# --- what it looks like -------------------------------------------------------


def test_the_layers_show_the_selection_and_nothing_else():
    widget = _loaded(distance=dict(zip(range(5), [0.1, 0.5, 1.0, 5.0, 20.0])))
    widget.show_points_box.setChecked(True)
    widget.show_tracks_box.setChecked(True)
    widget.dist_min_box.setValue(0.4)
    widget.dist_max_box.setValue(6.0)
    widget.distance_filter_box.setChecked(True)

    points = widget.viewer.layers[widget_mod.POINTS_LAYER_NAME]
    tracks = widget.viewer.layers[widget_mod.TRACKS_LAYER_NAME]
    assert len(points.data) == 3 * POINTS_PER_TRACK
    assert sorted(set(tracks.data[:, 0].astype(int))) == [1, 2, 3]


def test_the_header_counts_what_is_shown_against_what_exists():
    widget = _loaded(distance=dict(zip(range(5), [0.1, 0.5, 1.0, 5.0, 20.0])))
    widget.dist_min_box.setValue(0.4)
    widget.dist_max_box.setValue(6.0)
    widget.distance_filter_box.setChecked(True)
    assert "3 / 5 trajectories" in widget.status_label.text()


def test_the_summary_names_the_criteria_and_the_effect():
    widget = _loaded(distance=dict(zip(range(5), [0.1, 0.5, 1.0, 5.0, 20.0])))
    widget.dist_min_box.setValue(0.4)
    widget.dist_max_box.setValue(6.0)
    widget.distance_filter_box.setChecked(True)
    text = widget.track_filter_label.text()
    assert "Distance travelled 0.4-6" in text
    assert "3 of 5 trajectories" in text
    assert f"{3 * POINTS_PER_TRACK} of {len(widget.df_filtered)} localizations" in text


def test_the_bounds_are_the_same_ones_that_set_the_colour_scale():
    """One set of numbers, already drawn as lines on the histogram beside them -
    there is no second pair to keep in agreement with the first."""
    widget = _loaded(distance=dict.fromkeys(range(5), 1.0))
    assert widget._metric_bound_boxes["distance"] == (
        widget.dist_min_box, widget.dist_max_box)
    widget.dist_min_box.setValue(0.25)
    low, _high, _log = widget._metric_norm_range("distance")
    assert low == pytest.approx(0.25)


# --- carried with the run -----------------------------------------------------


def test_the_ticks_are_saved_and_restored_with_the_run():
    widget = _loaded(distance=dict.fromkeys(range(5), 1.0))
    widget.distance_filter_box.setChecked(True)
    metadata = widget._collect_metadata(None)
    assert metadata["dynamics_filter"]["distance"] is True
    assert metadata["dynamics_filter"]["D"] is False

    restored = _widget()
    restored.apply_settings(metadata)
    assert restored.distance_filter_box.isChecked()
    assert not restored.d_filter_box.isChecked()


def test_the_export_writes_what_is_on_screen():
    """An export holding more than the reconstruction beside it would be the
    more confusing of the two answers."""
    widget = _loaded(distance=dict(zip(range(5), [0.1, 0.5, 1.0, 5.0, 20.0])))
    widget.dist_min_box.setValue(0.4)
    widget.dist_max_box.setValue(6.0)
    widget.distance_filter_box.setChecked(True)

    tables = dict(widget._export_tables())
    assert len(tables["localizations_filtered.csv"]) == 3 * POINTS_PER_TRACK
    assert sorted(tables["trajectories.csv"]["particle"].unique()) == [1, 2, 3]
    assert sorted(tables["track_metrics.csv"]["particle"]) == [1, 2, 3]


def test_relinking_drops_a_selection_built_on_the_old_trajectories():
    widget = _loaded(distance=dict.fromkeys(range(5), 1.0))
    widget.dist_min_box.setValue(0.5)
    widget.dist_max_box.setValue(2.0)
    widget.distance_filter_box.setChecked(True)
    assert widget._passing_particles() == set(range(5))

    widget._invalidate_tracks(reason="test")
    assert widget._passing_particles() == set()      # no trajectories left to pass
    assert widget._displayed_localizations().empty


def test_a_filter_on_a_metric_computed_later_picks_it_up():
    """Ticking before the metric exists must not freeze an empty selection."""
    widget = _loaded()
    widget.dist_min_box.setValue(0.0)
    widget.dist_max_box.setValue(100.0)
    widget.distance_filter_box.setChecked(True)
    assert widget._passing_particles() == set()

    widget._on_fit_free_metrics_finished(
        {key: dict.fromkeys(range(5), 1.0)
         for key in ("distance", "net", "straightness", "duration",
                     "motion", "pstatic")})
    assert widget._passing_particles() == set(range(5))
