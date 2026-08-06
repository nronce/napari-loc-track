import pandas as pd

from napari_loc_track.widget import apply_numeric_filters, infer_column_map


def test_infer_column_map_matches_expected_columns():
    df = pd.DataFrame(
        {
            "frame": [1, 2, 3],
            "x [nm]": [100.0, 200.0, 300.0],
            "y [nm]": [100.0, 200.0, 300.0],
            "sigma [nm]": [150.0, 160.0, 170.0],
            "intensity [photon]": [1000.0, 1200.0, 900.0],
        }
    )
    mapping = infer_column_map(df.columns)

    assert mapping["x"] == "x [nm]"
    assert mapping["y"] == "y [nm]"
    assert mapping["frame"] == "frame"
    assert mapping["sigma"] == "sigma [nm]"
    assert mapping["intensity"] == "intensity [photon]"


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
