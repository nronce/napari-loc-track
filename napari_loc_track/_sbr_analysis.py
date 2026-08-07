"""Batch TIFF sum-projection and per-bead signal/background analysis."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MatplotlibPath


TIFF_SUFFIXES = {".tif", ".tiff"}

SBR_RESULT_COLUMNS = [
    "file_index",
    "file_name",
    "file_path",
    "z_layers",
    "is_reference",
    "selection_mode",
    "bead_id",
    "reference_x_px",
    "reference_y_px",
    "x_px",
    "y_px",
    "match_distance_px",
    "matched",
    "amplitude [photon/px]",
    "offset [photon/px]",
    "bkgstd [photon]",
    "intensity [photon]",
    "sigma_x_px",
    "sigma_y_px",
    "sbr",
]


def natural_sort_key(value):
    """Case-insensitive filename key with numeric chunks sorted numerically."""
    name = Path(value).name.casefold()
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def discover_tiff_files(folder):
    """Return TIFF files immediately inside *folder* in deterministic order."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"TIFF folder does not exist: {folder}")
    files = sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.casefold() in TIFF_SUFFIXES),
        key=natural_sort_key,
    )
    if not files:
        raise ValueError(f"No .tif or .tiff files found in {folder}")
    return files


def sum_project_array(image, axes=None):
    """Sum a grayscale Z/page stack into YX without integer overflow.

    Two-dimensional inputs are returned as a one-layer projection. Singleton
    dimensions are removed first, which permits common ``1ZYX`` TIFFs. After
    squeezing, only YX and layer-YX arrays are accepted; RGB/channel layouts
    are rejected rather than silently summing the wrong dimension.
    """
    array = np.asarray(image)
    axes_text = str(axes) if axes is not None and len(str(axes)) == array.ndim else None

    if axes_text is not None:
        keep = [idx for idx, size in enumerate(array.shape) if size != 1]
        axes_text = "".join(axes_text[idx] for idx in keep)
    array = np.squeeze(array)

    if array.ndim not in (2, 3):
        raise ValueError(
            f"Expected a grayscale 2D image or 3D layer stack, got shape {array.shape}"
        )
    if axes_text is not None and not axes_text.endswith("YX"):
        raise ValueError(
            f"Expected TIFF axes ending in YX, got axes {axes_text!r} for shape {array.shape}"
        )
    if (
        array.ndim == 3
        and axes_text is not None
        and axes_text[0] not in {"Z", "I", "Q"}
    ):
        raise ValueError(
            f"TIFF layer axis {axes_text[0]!r} in axes {axes_text!r} is not Z/page data; "
            "provide grayscale ZYX stacks"
        )

    if array.ndim == 2:
        return array.astype(np.float32, copy=False), 1

    projection = np.sum(array, axis=0, dtype=np.float64)
    return projection.astype(np.float32), int(array.shape[0])


def load_tiff_projection(path):
    """Read one TIFF and return ``(sum_projection, layer_count)``."""
    from tifffile import TiffFile

    path = Path(path)
    with TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"TIFF contains no readable image series: {path}")
        series = tif.series[0]
        image = series.asarray()
        axes = getattr(series, "axes", None)
    return sum_project_array(image, axes=axes)


def load_projection_folder(folder):
    """Load and sum-project every TIFF in a folder."""
    files = discover_tiff_files(folder)
    projections = []
    layer_counts = []
    expected_shape = None
    for path in files:
        projection, layer_count = load_tiff_projection(path)
        if expected_shape is None:
            expected_shape = projection.shape
        elif projection.shape != expected_shape:
            raise ValueError(
                f"Projected TIFF shapes differ: expected {expected_shape}, "
                f"got {projection.shape} for {path.name}"
            )
        projections.append(projection)
        layer_counts.append(layer_count)
    return np.stack(projections), files, np.asarray(layer_counts, dtype=np.int32)


def calculate_sbr(amplitude, background):
    """Calculate dimensionless ``(amplitude + background) / background``."""
    amplitude_array, background_array = np.broadcast_arrays(
        np.asarray(amplitude, dtype=np.float64),
        np.asarray(background, dtype=np.float64),
    )
    result = np.full(amplitude_array.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(amplitude_array)
        & np.isfinite(background_array)
        & (background_array > 0.0)
    )
    result[valid] = (
        amplitude_array[valid] + background_array[valid]
    ) / background_array[valid]
    if result.ndim == 0:
        return float(result)
    return result


def sum_projection_camera_offset(camera_offset_adu, layer_count):
    """Return the accumulated camera baseline in a raw sum projection."""
    layer_count = int(layer_count)
    camera_offset_adu = float(camera_offset_adu)
    if layer_count < 1:
        raise ValueError("A projection must contain at least one layer")
    if not np.isfinite(camera_offset_adu) or camera_offset_adu < 0.0:
        raise ValueError("Camera offset must be a finite non-negative number")
    return camera_offset_adu * layer_count


def _as_yx_points(points):
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("Points must have shape (N, 2) in (y, x) order")
    return points[:, -2:]


def points_in_polygons(points_yx, polygons):
    """Return a union mask for points inside/on any 2D ROI polygon."""
    points = _as_yx_points(points_yx)
    inside = np.zeros(len(points), dtype=bool)
    for polygon in polygons or []:
        vertices = _as_yx_points(polygon)
        if len(vertices) < 3:
            continue
        if not np.allclose(vertices[0], vertices[-1]):
            vertices = np.vstack([vertices, vertices[0]])
        path = MatplotlibPath(vertices)
        # A positive radius expands one polygon winding direction and shrinks
        # the other. Union both signs so edge/corner inclusion is independent
        # of whether napari returned clockwise or counter-clockwise vertices.
        inside |= path.contains_points(points, radius=1e-9)
        inside |= path.contains_points(points, radius=-1e-9)
    return inside


def match_reference_points(reference_yx, target_yx, tolerance_px):
    """One-to-one minimum-distance matching with optional unmatched rows."""
    from scipy.optimize import linear_sum_assignment

    reference = _as_yx_points(reference_yx)
    target = _as_yx_points(target_yx)
    tolerance = float(tolerance_px)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Matching tolerance must be a finite non-negative number")

    matched_indices = np.full(len(reference), -1, dtype=np.int32)
    matched_distances = np.full(len(reference), np.nan, dtype=np.float64)
    if len(reference) == 0 or len(target) == 0:
        return matched_indices, matched_distances

    distances = np.linalg.norm(reference[:, None, :] - target[None, :, :], axis=2)
    # One dummy assignment must cost more than the largest possible sum of
    # all valid distances. This makes the assignment maximum-cardinality
    # first and minimum-distance second, instead of dropping valid-but-distant
    # pairs merely to reduce the total distance.
    distance_scale = max(1.0, tolerance)
    dummy_cost = (len(reference) + 1) * distance_scale
    invalid_cost = dummy_cost * (len(reference) + len(target) + 2)
    real_cost = np.where(distances <= tolerance, distances, invalid_cost)
    dummy_columns = np.full((len(reference), len(reference)), dummy_cost, dtype=np.float64)
    cost = np.concatenate([real_cost, dummy_columns], axis=1)

    row_indices, column_indices = linear_sum_assignment(cost)
    for row, column in zip(row_indices, column_indices):
        if column < len(target) and distances[row, column] <= tolerance:
            matched_indices[row] = int(column)
            matched_distances[row] = float(distances[row, column])
    return matched_indices, matched_distances


def _localization_points(localizations):
    return np.column_stack(
        [
            np.asarray(localizations["y"], dtype=np.float64),
            np.asarray(localizations["x"], dtype=np.float64),
        ]
    )


def _rank_by_amplitude(localizations, indices):
    indices = np.asarray(indices, dtype=np.int32)
    amplitudes = np.asarray(localizations["amp"], dtype=np.float64)[indices]
    safe_amplitudes = np.where(np.isfinite(amplitudes), amplitudes, -np.inf)
    order = np.lexsort((indices, -safe_amplitudes))
    return indices[order]


def analyze_sbr_localizations(
    localizations_by_file,
    file_paths,
    layer_counts,
    *,
    tolerance_px,
    roi_polygons=None,
    default_bead_count=10,
):
    """Select reference beads, match them, and build a long SBR table.

    ROI polygons select all enclosed localizations in the first TIFF. Without
    an ROI, the brightest reference beads that have a one-to-one match in
    every TIFF are selected, up to ``default_bead_count``.
    """
    if not localizations_by_file:
        raise ValueError("No localizations were provided")
    if len(file_paths) != len(localizations_by_file):
        raise ValueError("File/localization counts differ")
    if len(layer_counts) != len(localizations_by_file):
        raise ValueError("Layer/localization counts differ")

    reference = localizations_by_file[0]
    reference_points = _localization_points(reference)
    if len(reference_points) == 0:
        raise ValueError("The reference TIFF produced no localizations")

    has_roi = bool(roi_polygons)
    if has_roi:
        selected = np.flatnonzero(points_in_polygons(reference_points, roi_polygons)).astype(np.int32)
        if len(selected) == 0:
            raise ValueError("The SBR ROI contains no reference-image localizations")
        selection_mode = "roi"
        points_to_match = reference_points[selected]
        matches = [(selected.copy(), np.zeros(len(selected), dtype=np.float64))]
        for target in localizations_by_file[1:]:
            matches.append(
                match_reference_points(points_to_match, _localization_points(target), tolerance_px)
            )
    else:
        all_reference = np.arange(len(reference_points), dtype=np.int32)
        ranked = _rank_by_amplitude(reference, all_reference)
        count = max(1, int(default_bead_count))
        selected_list = []
        target_point_sets = [
            _localization_points(target) for target in localizations_by_file[1:]
        ]
        # Consider reference beads from brightest to dimmest. A bead is kept
        # only when the complete selected set still has a one-to-one match in
        # every target TIFF; this both honors brightness and prevents a dimmer
        # neighbor from stealing the only viable target of a brighter bead.
        for reference_index in ranked:
            trial = np.asarray(selected_list + [int(reference_index)], dtype=np.int32)
            trial_points = reference_points[trial]
            feasible = True
            for target_points in target_point_sets:
                target_indices, _ = match_reference_points(
                    trial_points, target_points, tolerance_px
                )
                if np.any(target_indices < 0):
                    feasible = False
                    break
            if feasible:
                selected_list.append(int(reference_index))
                if len(selected_list) >= count:
                    break
        if not selected_list:
            raise ValueError(
                "No reference beads could be matched across every TIFF within the tolerance"
            )
        selected = np.asarray(selected_list, dtype=np.int32)
        selection_mode = "auto_brightest"
        selected_points = reference_points[selected]
        matches = [(selected.copy(), np.zeros(len(selected), dtype=np.float64))]
        for target_points in target_point_sets:
            matches.append(
                match_reference_points(selected_points, target_points, tolerance_px)
            )

    rows = []
    for file_index, (localizations, path, layer_count, match) in enumerate(
        zip(localizations_by_file, file_paths, layer_counts, matches)
    ):
        target_indices, distances = match
        for selected_offset, reference_index in enumerate(selected):
            target_index = int(target_indices[selected_offset])
            matched = target_index >= 0
            reference_x = float(reference["x"][reference_index])
            reference_y = float(reference["y"][reference_index])

            if matched:
                amplitude = float(localizations["amp"][target_index])
                background = float(localizations["bg"][target_index])
                row = {
                    "x_px": float(localizations["x"][target_index]),
                    "y_px": float(localizations["y"][target_index]),
                    "match_distance_px": float(distances[selected_offset]),
                    "amplitude [photon/px]": amplitude,
                    "offset [photon/px]": background,
                    "bkgstd [photon]": float(localizations["bkgstd"][target_index]),
                    "intensity [photon]": float(localizations["photons"][target_index]),
                    "sigma_x_px": float(localizations["sx"][target_index]),
                    "sigma_y_px": float(localizations["sy"][target_index]),
                    "sbr": calculate_sbr(amplitude, background),
                }
            else:
                row = {
                    "x_px": np.nan,
                    "y_px": np.nan,
                    "match_distance_px": np.nan,
                    "amplitude [photon/px]": np.nan,
                    "offset [photon/px]": np.nan,
                    "bkgstd [photon]": np.nan,
                    "intensity [photon]": np.nan,
                    "sigma_x_px": np.nan,
                    "sigma_y_px": np.nan,
                    "sbr": np.nan,
                }

            row.update(
                {
                    "file_index": int(file_index),
                    "file_name": Path(path).name,
                    "file_path": str(path),
                    "z_layers": int(layer_count),
                    "is_reference": bool(file_index == 0),
                    "selection_mode": selection_mode,
                    "bead_id": int(selected_offset + 1),
                    "reference_x_px": reference_x,
                    "reference_y_px": reference_y,
                    "matched": bool(matched),
                }
            )
            rows.append(row)

    results = pd.DataFrame(rows, columns=SBR_RESULT_COLUMNS)
    return results, selected, selection_mode
