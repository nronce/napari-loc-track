from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tifffile import imwrite

from napari_loc_track._localize2d import localize_frame
from napari_loc_track._sbr_analysis import (
    analyze_sbr_localizations,
    calculate_sbr,
    discover_tiff_files,
    load_projection_folder,
    match_reference_points,
    points_in_polygons,
    sum_project_array,
    sum_projection_camera_offset,
)


def _locs(x, y, amplitude, background=None):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    amplitude = np.asarray(amplitude, dtype=np.float32)
    if background is None:
        background = np.full(len(x), 10.0, dtype=np.float32)
    else:
        background = np.asarray(background, dtype=np.float32)
    ones = np.ones(len(x), dtype=np.float32)
    return {
        "frame": np.zeros(len(x), dtype=np.int32),
        "x": x,
        "y": y,
        "amp": amplitude,
        "photons": amplitude * 10.0,
        "sx": ones,
        "sy": ones,
        "bg": background,
        "bkgstd": ones * 2.0,
        "lpx": ones * 0.1,
        "lpy": ones * 0.1,
        "net_gradient": ones * 100.0,
    }


def test_sbr_formula_and_zero_background():
    assert calculate_sbr(20.0, 10.0) == 3.0
    measured = calculate_sbr(
        np.array([20.0, 5.0, np.nan]),
        np.array([10.0, 0.0, 2.0]),
    )
    assert measured[0] == 3.0
    assert np.isnan(measured[1])
    assert np.isnan(measured[2])


def test_sum_projection_avoids_uint16_overflow_and_accumulates_offset():
    stack = np.full((5, 3, 4), 60000, dtype=np.uint16)
    projection, layers = sum_project_array(stack, axes="ZYX")

    assert layers == 5
    assert projection.dtype == np.float32
    assert np.all(projection == 300000.0)
    assert sum_projection_camera_offset(100.0, layers) == 500.0

    image = np.arange(12, dtype=np.uint16).reshape(3, 4)
    unchanged, layers = sum_project_array(image, axes="YX")
    assert layers == 1
    assert np.array_equal(unchanged, image)

    try:
        sum_project_array(stack, axes="TYX")
    except ValueError as exc:
        assert "not Z/page data" in str(exc)
    else:
        raise AssertionError("TYX time series must not be treated as a Z-stack")


def test_sum_projection_fit_subtracts_one_camera_offset_per_layer():
    yy, xx = np.indices((25, 25), dtype=np.float64)
    gaussian = np.exp(-((xx - 12.0) ** 2 + (yy - 12.0) ** 2) / (2.0 * 1.2**2))
    # Each raw layer has camera offset 100, local background 10, and bead
    # amplitude 50. A three-layer sum should therefore fit b=30 and A=150.
    raw_stack = np.stack([110.0 + 50.0 * gaussian] * 3)
    projection, layer_count = sum_project_array(raw_stack, axes="ZYX")
    locs = localize_frame(
        projection,
        np.array([12]),
        np.array([12]),
        9,
        frame_number=0,
        fit_backend="fast",
        camera_offset_adu=sum_projection_camera_offset(100.0, layer_count),
        camera_gain_adu_per_photon=1.0,
    )

    assert np.isclose(locs["bg"][0], 30.0, atol=0.1)
    assert np.isclose(locs["amp"][0], 150.0, atol=0.2)
    assert np.isclose(calculate_sbr(locs["amp"][0], locs["bg"][0]), 6.0, atol=0.02)


def test_tiff_folder_uses_natural_order_and_exact_sum_projection():
    with TemporaryDirectory() as temporary_folder:
        folder = Path(temporary_folder)
        stack_2 = np.arange(5 * 4 * 6, dtype=np.uint16).reshape(5, 4, 6)
        stack_10 = stack_2 + 10
        imwrite(folder / "stack10.TIFF", stack_10, photometric="minisblack")
        imwrite(folder / "stack2.tif", stack_2, photometric="minisblack")
        (folder / "ignore.txt").write_text("not a TIFF", encoding="utf-8")

        discovered = discover_tiff_files(folder)
        projections, files, layer_counts = load_projection_folder(folder)

    assert [path.name for path in discovered] == ["stack2.tif", "stack10.TIFF"]
    assert [path.name for path in files] == ["stack2.tif", "stack10.TIFF"]
    assert layer_counts.tolist() == [5, 5]
    assert np.array_equal(projections[0], stack_2.sum(axis=0))
    assert np.array_equal(projections[1], stack_10.sum(axis=0))


def test_roi_union_includes_polygon_boundaries():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [5.0, 5.0],
            [9.0, 9.0],
        ]
    )
    first = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]])
    second = np.array([[4.0, 4.0], [4.0, 6.0], [6.0, 6.0], [6.0, 4.0]])

    mask = points_in_polygons(points, [first, second])

    assert mask.tolist() == [True, True, True, True, False]


def test_matching_is_one_to_one_and_honors_inclusive_tolerance():
    reference = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    target = np.array([[0.0, 0.4], [5.0, 7.0]])

    indices, distances = match_reference_points(reference, target, tolerance_px=2.0)

    assert np.count_nonzero(indices == 0) == 1
    assert indices[2] == 1
    assert distances[2] == 2.0
    assert np.count_nonzero(indices < 0) == 1


def test_matching_maximizes_cardinality_before_distance():
    # Both crossed matches (distance 0.9) must win over the tempting
    # zero-distance match plus an unmatched reference.
    reference = np.array([[0.0, 0.0], [0.0, 0.9]])
    target = np.array([[0.0, 0.0], [0.0, -0.9]])

    indices, distances = match_reference_points(reference, target, tolerance_px=1.0)

    assert indices.tolist() == [1, 0]
    assert np.allclose(distances, [0.9, 0.9])


def test_automatic_selection_prioritizes_brightness_during_a_collision():
    reference = _locs([0.0, 0.2], [0.0, 0.0], [100.0, 10.0])
    target = _locs([0.15], [0.0], [20.0])

    results, selected, _ = analyze_sbr_localizations(
        [reference, target],
        [Path("reference.tif"), Path("target.tif")],
        [1, 1],
        tolerance_px=0.2,
        default_bead_count=1,
    )

    assert selected.tolist() == [0]
    assert results["matched"].all()


def test_default_selection_uses_brightest_beads_matched_in_every_file():
    reference = _locs(
        x=np.arange(12, dtype=float),
        y=np.zeros(12),
        amplitude=np.arange(1, 13, dtype=float),
    )
    # The brightest reference bead (x=11) is missing, so it is ineligible.
    target = _locs(
        x=np.arange(11, dtype=float) + 0.1,
        y=np.zeros(11),
        amplitude=np.arange(1, 12, dtype=float),
    )

    results, selected, mode = analyze_sbr_localizations(
        [reference, target],
        [Path("image1.tif"), Path("image2.tif")],
        [3, 3],
        tolerance_px=0.25,
        default_bead_count=10,
    )

    assert mode == "auto_brightest"
    assert selected.tolist() == list(range(10, 0, -1))
    assert results["bead_id"].nunique() == 10
    assert results["matched"].all()
    assert np.allclose(results["sbr"], (results["amplitude [photon/px]"] + 10.0) / 10.0)


def test_roi_selection_keeps_an_unmatched_row():
    reference = _locs([1.0, 3.0], [1.0, 1.0], [20.0, 30.0])
    target = _locs([1.1], [1.0], [21.0])
    roi = np.array([[0.0, 0.0], [0.0, 4.0], [2.0, 4.0], [2.0, 0.0]])

    results, selected, mode = analyze_sbr_localizations(
        [reference, target],
        [Path("reference.tif"), Path("target.tif")],
        [2, 2],
        tolerance_px=0.25,
        roi_polygons=[roi],
    )

    assert mode == "roi"
    assert selected.tolist() == [0, 1]
    assert len(results) == 4
    target_rows = results[results["file_index"] == 1].sort_values("bead_id")
    assert target_rows["matched"].tolist() == [True, False]
    assert np.isnan(target_rows.iloc[1]["sbr"])
