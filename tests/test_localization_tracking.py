import numpy as np
import pandas as pd

from napari_loc_track._localize2d import (
    _background_std_from_fit,
    localize_frame,
)
from napari_loc_track.widget import apply_numeric_filters, infer_column_map


def test_infer_column_map_matches_expected_columns():
    df = pd.DataFrame(
        {
            "frame": [1, 2, 3],
            "x [nm]": [100.0, 200.0, 300.0],
            "y [nm]": [100.0, 200.0, 300.0],
            "sigma [nm]": [150.0, 160.0, 170.0],
            "intensity [photon]": [1000.0, 1200.0, 900.0],
            "bkgstd [photon]": [8.0, 9.0, 7.0],
        }
    )
    mapping = infer_column_map(df.columns)

    assert mapping["x"] == "x [nm]"
    assert mapping["y"] == "y [nm]"
    assert mapping["frame"] == "frame"
    assert mapping["sigma"] == "sigma [nm]"
    assert mapping["intensity"] == "intensity [photon]"
    assert mapping["bkgstd"] == "bkgstd [photon]"


def test_apply_numeric_filters_respects_bounds():
    df = pd.DataFrame(
        {
            "frame": [1, 2, 3, 4],
            "x [nm]": [100, 200, 300, 400],
            "y [nm]": [100, 200, 300, 400],
        }
    )

    filtered = apply_numeric_filters(
        df,
        {"frame": (1.5, 3.5), "x [nm]": (150.0, 350.0)},
    )

    assert len(filtered) == 2
    assert filtered["frame"].tolist() == [2, 3]


def test_background_std_matches_thunderstorm_residual_definition():
    yy, xx = np.indices((5, 5), dtype=np.float64)
    params = np.array([100.0, 20.0, 2.0, 2.0, 1.0, 1.0])
    amp, bg, x0, y0, sx, sy = params
    model = bg + amp * np.exp(
        -(
            ((xx - x0) ** 2) / (2.0 * sx**2)
            + ((yy - y0) ** 2) / (2.0 * sy**2)
        )
    )
    residual = np.arange(25, dtype=np.float64).reshape(5, 5) - 12.0

    measured = _background_std_from_fit(model + residual, xx, yy, params)

    assert np.isclose(measured, np.std(residual, ddof=0))


def test_localize_frame_returns_a_background_std_for_each_bead():
    rng = np.random.default_rng(4)
    yy, xx = np.indices((25, 45), dtype=np.float64)
    frame = np.full((25, 45), 50.0)
    frame[:, :22] += rng.normal(0.0, 1.0, size=(25, 22))
    frame[:, 22:] += rng.normal(0.0, 8.0, size=(25, 23))
    frame += 500.0 * np.exp(-((xx - 10.0) ** 2 + (yy - 12.0) ** 2) / 2.0)
    frame += 500.0 * np.exp(-((xx - 34.0) ** 2 + (yy - 12.0) ** 2) / 2.0)

    for backend in ("fast", "mle", "gpu"):
        locs = localize_frame(
            frame,
            np.array([12, 12]),
            np.array([10, 34]),
            9,
            frame_number=0,
            fit_backend=backend,
            camera_offset_adu=0.0,
            camera_gain_adu_per_photon=1.0,
        )

        assert locs["bkgstd"].shape == (2,)
        assert np.all(np.isfinite(locs["bkgstd"]))
        assert locs["bkgstd"][1] > 3.0 * locs["bkgstd"][0]
