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
    distance_map, net_map, straightness_map, _duration_map = worker.work()

    expected_um = n_steps * pixel_size_nm / 1000.0  # 10 px * 100 nm = 1 um
    assert distance_map[0] == pytest.approx(expected_um)
    # a straight line, so the path and the end-to-end displacement agree
    assert net_map[0] == pytest.approx(expected_um)
    assert straightness_map[0] == pytest.approx(1.0)


def test_distance_label_says_microns():
    assert "µm" in widget.METRIC_LABELS["distance"]
