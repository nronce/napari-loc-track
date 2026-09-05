"""Default filter ranges and per-trajectory metric units.

These are the numbers a user sees before touching anything, so they are worth
pinning: a wrong sigma default squashes the useful part of the histogram, and a
wrong distance unit is silently off by 1000.
"""
import numpy as np
import pytest

widget = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)


@pytest.mark.parametrize(
    "column",
    ["sigma [nm]", "sigma_x [nm]", "sigma_y [nm]", "sigma", "Sigma1 [nm]", "sigma_nm"],
)
def test_sigma_columns_are_recognised(column):
    assert widget.is_sigma_column(column)


@pytest.mark.parametrize("column", ["x [nm]", "intensity [photon]", "frame", "uncertainty [nm]"])
def test_non_sigma_columns_are_not(column):
    assert not widget.is_sigma_column(column)


def test_sigma_default_range_is_0_to_500_nm():
    assert widget.SIGMA_DEFAULT_BOUNDS_NM == (0.0, 500.0)


def test_distance_is_reported_in_microns():
    """1000 nm of travel is 1 um, whatever the pixel size."""
    import pandas as pd

    pixel_size_nm = 100.0
    n_steps = 10
    tracks = pd.DataFrame(
        {
            "particle": [0] * (n_steps + 1),
            "frame": range(n_steps + 1),
            "x": np.arange(n_steps + 1, dtype=float),  # 1 px per frame
            "y": np.zeros(n_steps + 1),
        }
    )
    worker = widget._fit_free_metrics_worker(tracks, pixel_size_nm, 100.0)
    result = worker.work()
    distance_map, net_map, straightness_map = (
        result["distance"], result["net"], result["straightness"])

    expected_um = n_steps * pixel_size_nm / 1000.0  # 10 px * 100 nm = 1 um
    assert distance_map[0] == pytest.approx(expected_um)
    # a straight line, so the path and the end-to-end displacement agree
    assert net_map[0] == pytest.approx(expected_um)
    assert straightness_map[0] == pytest.approx(1.0)


def test_distance_label_says_microns():
    assert "µm" in widget.METRIC_LABELS["distance"]


# --- default bounds must include the data they were derived from --------------


@pytest.mark.parametrize("value,decimals,upward,expected", [
    (49995.8477774829, 6, True, 49995.847778),    # rounds down; must step up
    (49995.8477774829, 6, False, 49995.847777),   # rounds down; already outward
    (12.4234341111, 6, False, 12.423434),
    (0.5, 6, True, 0.5),                          # exact, left alone
    (500.0, 6, True, 500.0),
])
def test_a_bound_is_rounded_outwards_to_what_the_box_can_hold(
        value, decimals, upward, expected):
    assert widget.bound_to_box_precision(value, decimals, upward) == pytest.approx(
        expected, abs=1e-9)


def test_the_default_filter_keeps_every_localization_it_was_built_from():
    """A six-decimal box turns a maximum of 49995.8477774829 into 49995.847777,
    which is *below* it - so the filter built to keep everything dropped the
    single most extreme localization in every column at once, silently.
    """
    import os
    import sys
    from pathlib import Path

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("napari_loc_track.widget")
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).parent))
    from test_widget_interaction import make_widget

    rng = np.random.default_rng(0)
    n = 4000
    frame = pd.DataFrame({
        "frame": rng.integers(0, 50, n),
        "x [nm]": rng.uniform(0, 50000, n),
        "y [nm]": rng.uniform(0, 50000, n),
        "sigma [nm]": rng.uniform(50, 400, n),
        "intensity [photon]": rng.uniform(10, 9000, n),
    })
    w = make_widget()
    w._ingest_localization_dataframe(frame, "loaded", True)
    w.apply_filters()
    assert len(w.df_filtered) == n


# --- what the histograms show -------------------------------------------------
#
# Two distributions per column: everything loaded, pale, and what survives the
# filters, solid. Drawing only the first meant tightening a bound on sigma
# changed nothing visible in the intensity histogram beside it - and the
# coupling between columns is the reason to have these plots at all.


def _filtered_widget():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).parent))
    from test_widget_interaction import make_widget

    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({
        "frame": rng.integers(0, 10, n),
        "x [nm]": rng.uniform(0, 1e4, n),
        "y [nm]": rng.uniform(0, 1e4, n),
        "sigma [nm]": rng.uniform(80, 400, n),
        "intensity [photon]": rng.uniform(100, 3000, n),
    })
    w = make_widget()
    w._ingest_localization_dataframe(frame, "loaded", True)
    return w


def _bar_totals(widget, column):
    """(all localizations, surviving localizations) as drawn."""
    axes = widget._hist_widgets[column]["figure"].axes[0]
    return [sum(patch.get_height() for patch in group) for group in axes.containers]


def test_both_distributions_are_drawn():
    widget = _filtered_widget()
    totals = _bar_totals(widget, "intensity [photon]")
    assert len(totals) == 2
    assert totals[0] == totals[1] == 400      # nothing filtered yet


def test_filtering_one_column_changes_the_histogram_of_another():
    """The bug this fixes: cutting on sigma left every other plot untouched."""
    widget = _filtered_widget()
    _lower, upper = widget.filter_controls["sigma [nm]"]
    upper.setValue(200.0)
    widget.apply_filters()

    everything, kept = _bar_totals(widget, "intensity [photon]")
    assert everything == 400                  # the pale distribution stays put
    assert kept == len(widget.df_filtered)
    assert kept < everything


def test_the_title_says_how_many_survived():
    widget = _filtered_widget()
    assert widget._hist_widgets["intensity [photon]"]["figure"].axes[0].get_title() \
        == "intensity [photon]"

    _lower, upper = widget.filter_controls["sigma [nm]"]
    upper.setValue(200.0)
    widget.apply_filters()
    title = widget._hist_widgets["intensity [photon]"]["figure"].axes[0].get_title()
    assert title == f"intensity [photon]   {len(widget.df_filtered)} / 400"


def test_widening_the_filter_again_brings_the_bars_back():
    widget = _filtered_widget()
    _lower, upper = widget.filter_controls["sigma [nm]"]
    upper.setValue(200.0)
    widget.apply_filters()
    assert _bar_totals(widget, "intensity [photon]")[1] < 400

    widget.reset_filters()
    assert _bar_totals(widget, "intensity [photon]")[1] == 400
