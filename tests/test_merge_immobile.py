"""Merging each confidently immobile trajectory into one localization, in a render.

A molecule that never moved, seen N times, is one localization about sqrt(N)
times more precise - and drawing it N times scatters it over its own error.
Where the line is drawn matters more than the merge: a trajectory passes the
static test just as easily by being too short or too dim to fail it, so only
those whose test could have detected motion down to a chosen D are merged.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_render_populations import N_POINTS, _analysed, _render  # noqa: E402


def _merging(widget, d_max=0.01, min_points=3):
    widget.immobile_dmax_box.setValue(d_max)
    widget.immobile_min_points_box.setValue(min_points)
    widget.merge_box.setChecked(True)
    return widget


def test_merging_is_off_by_default():
    widget = _analysed()
    assert not widget.merge_box.isChecked()
    assert len(widget._render_table()) == len(widget._displayed_localizations())


def test_each_static_trajectory_becomes_one_localization():
    widget = _merging(_analysed(n_static=30, n_mobile=30))
    ids = widget._merge_ids()
    # the static ones (particles 0-29), less the one in twenty a 5% test
    # calls mobile - and none of the mobile ones
    assert len(ids) >= 25 and ids <= set(range(30))
    table = widget._render_table()
    merged = table[table["n_merged"] > 1]
    assert len(merged) == len(ids)
    assert (merged["n_merged"] == N_POINTS).all()
    assert (table["n_merged"] == 1).sum() == len(widget.df_filtered) - N_POINTS * len(ids)


def test_a_merged_localization_is_drawn_at_its_combined_precision():
    widget = _merging(_analysed())
    table = widget._render_table()
    merged = table[table["n_merged"] > 1]["uncertainty [nm]"]
    single = widget.df_filtered["uncertainty [nm]"]
    # 15 points at 15-35 nm combine to well under 10 nm
    assert merged.median() < single.median() / 3


def test_a_count_render_keeps_the_signal_and_concentrates_it():
    """Merged points weigh as the localizations they replace: the image is as
    bright as before, and sharper."""
    widget = _analysed()
    widget._set_render_population("immobile")
    n_shown = len(widget._displayed_localizations())
    _merging(widget)
    options, info, _layer = widget._render_inputs()
    assert len(options["x_px"]) < n_shown
    assert options["weights"].sum() == pytest.approx(n_shown)
    assert info["merged_immobile"]["trajectories"] > 0
    _render(widget)
    # merged, it goes to a layer of its own, beside the unmerged render
    rendered = float(np.asarray(widget.viewer.layers["smlm_render_immobile_merged"].data).sum())
    assert rendered == pytest.approx(n_shown, rel=0.05)


def test_photon_weighting_sums_the_photons_of_a_merged_trajectory():
    widget = _merging(_analysed())
    widget.render_photons_box.setChecked(True)
    options, _info, _layer = widget._render_inputs()
    total = widget._displayed_localizations()["intensity [photon]"].sum()
    assert options["weights"].sum() == pytest.approx(total)


def test_a_strict_detection_floor_merges_nothing():
    """Asking the test to have seen motion slower than it ever could leaves
    every static trajectory as fitted, however static it looked."""
    widget = _merging(_analysed(), d_max=1e-6)
    assert widget._merge_ids() == set()
    assert widget.merge_label.text().startswith("0 immobile")


def test_a_two_point_trajectory_is_never_confident_enough():
    widget = _analysed(n_static=5, n_mobile=0)
    # cut every trajectory to two points: static, and certifying nothing
    tracks = widget.tracks[widget.tracks["frame"] < 2].reset_index(drop=True)
    widget.tracks = tracks
    widget._track_pstatic_cache = None
    widget._track_dmin_cache = None
    widget._invalidate_track_filter()
    widget._start_fit_free_metrics_worker()
    from test_render_widget import _pump_until
    assert _pump_until(lambda: widget._track_pstatic_cache is not None)
    assert widget._track_pstatic_cache            # the test did run, and passed them
    _merging(widget, d_max=100.0, min_points=2)
    assert widget._merge_ids() == set()


def test_the_minimum_length_is_a_floor_of_its_own():
    widget = _merging(_analysed(), min_points=N_POINTS + 1)
    assert widget._merge_ids() == set()


def test_the_panel_says_how_long_a_trajectory_must_be():
    widget = _merging(_analysed())
    widget._update_render_population_label()
    assert "points to count as immobile" in widget.population_counts_label.text()
    assert "immobile trajectories" in widget.merge_label.text()


def test_the_export_carries_the_merged_table_beside_the_fitted_one():
    widget = _merging(_analysed())
    tables = dict(widget._export_tables())
    assert "localizations_filtered.csv" in tables
    merged = tables["localizations_merged.csv"]
    assert {"n_merged", "particle", "frame_last"} <= set(merged.columns)


def test_merge_settings_round_trip():
    widget = _merging(_analysed(), d_max=0.004, min_points=7)
    values, _notes = widget_mod.settings_from_metadata(widget._collect_metadata(None))
    assert values["merge_box"] is True
    assert values["immobile_dmax_box"] == pytest.approx(0.004)
    assert values["immobile_min_points_box"] == 7


def test_the_merged_molecules_and_what_they_replaced_are_layers_to_compare():
    widget = _merging(_analysed())
    ids = widget._merge_ids()
    widget.show_merged_layers()
    after = widget.viewer.layers[widget_mod.MERGED_AFTER_LAYER_NAME]
    before = widget.viewer.layers[widget_mod.MERGED_BEFORE_LAYER_NAME]
    assert len(after.data) == len(ids)
    assert len(before.data) == N_POINTS * len(ids)
    assert (after.features["n_merged"] == N_POINTS).all()
    # each merged point sits among the localizations it replaced
    for pid in list(ids)[:5]:
        mine = before.data[before.features["particle"].to_numpy() == pid]
        here = after.data[after.features["particle"].to_numpy() == pid][0]
        assert np.linalg.norm(mine.mean(axis=0) - here) < 0.5


def test_the_merged_molecules_have_a_table_and_distributions_of_their_own():
    widget = _merging(_analysed())
    widget.show_merged_table()
    assert widget.merged_table_model.rowCount() == len(widget._merge_ids())
    assert "molecules" in widget.merged_table_label.text()
    widget.merged_table_dialog.hide()
    widget._draw_merge_distributions()
    assert len(widget.merge_figure.axes) == 4


def test_a_merged_render_is_named_apart_from_the_unmerged_one():
    widget = _analysed()
    plain = widget._render_layer_name("image")
    _merging(widget)
    assert widget._render_layer_name("image") == plain + "_merged"
    widget.merge_box.setChecked(False)
    assert widget._render_layer_name("image") == plain
