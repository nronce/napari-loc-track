"""Every run keeps its own folder, and nothing overwrites anything.

A fit is expensive and its result is easy to lose: the next one replaces it in
memory, and the settings that produced it live only in the controls until
something writes them down. So each fit writes itself out - localizations plus
the complete settings - into a folder stamped with when it happened, beside the
data it came from.

The property that matters is that re-fitting never costs you the previous
answer. Two fits at different thresholds are two results, and which came first
is part of what you need to know.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_render_widget import _pump_until  # noqa: E402
from test_widget_interaction import make_widget  # noqa: E402

_WIDGETS = []


def _locs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "frame": rng.integers(0, 20, n),
        "x [nm]": rng.uniform(0, 20000, n),
        "y [nm]": rng.uniform(0, 20000, n),
        "sigma [nm]": rng.uniform(100, 200, n),
        "intensity [photon]": rng.uniform(100, 2000, n),
        "uncertainty [nm]": rng.uniform(8, 40, n),
    })


def _widget_in(tmp_path, image_name="stack.tif"):
    widget = make_widget()
    _WIDGETS.append(widget)
    (tmp_path / image_name).write_bytes(b"not a real tiff")
    widget.image_edit.setText(str(tmp_path / image_name))
    widget.pixel_size_box.setValue(161.0)
    return widget


def _fit(widget, df=None, min_ng=None):
    """Drive the fit's completion handler, the way a finished worker would."""
    if min_ng is not None:
        widget.loc_min_ng_box.setValue(min_ng)
    df = _locs() if df is None else df
    widget._ingest_localization_dataframe(df, "fitted", True)
    widget._autosave_localization_run(df)
    assert _pump_until(lambda: widget._autosave_worker_ref is None), "the save never finished"


def _runs(tmp_path):
    root = tmp_path / widget_mod.ANALYSIS_ROOT
    return sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []


# --- one folder per run -------------------------------------------------------


def test_a_fit_writes_itself_out_beside_the_data(tmp_path):
    widget = _widget_in(tmp_path)
    _fit(widget)

    runs = _runs(tmp_path)
    assert len(runs) == 1
    assert (runs[0] / "data" / widget_mod.LOCS_RUN_FILENAME).is_file()
    assert (runs[0] / "metadata.json").is_file()


def test_the_folder_is_named_for_when_it_happened(tmp_path):
    widget = _widget_in(tmp_path)
    _fit(widget)
    name = _runs(tmp_path)[0].name
    assert name.endswith("_localization")
    stamp = name[: -len("_localization")]
    # Parses back as the date it claims to be, and is sortable as written.
    from datetime import datetime

    parsed = datetime.strptime(stamp, widget_mod.RUN_STAMP_FORMAT)
    assert abs((datetime.now() - parsed).total_seconds()) < 120


def test_a_second_fit_does_not_touch_the_first(tmp_path):
    """The whole point: re-fitting after changing a threshold costs nothing."""
    widget = _widget_in(tmp_path)
    _fit(widget, _locs(n=300, seed=1), min_ng=400.0)
    time.sleep(1.05)                       # a different second, as in real use
    _fit(widget, _locs(n=120, seed=2), min_ng=900.0)

    runs = _runs(tmp_path)
    assert len(runs) == 2
    rows = [len(pd.read_csv(r / "data" / widget_mod.LOCS_RUN_FILENAME)) for r in runs]
    assert sorted(rows) == [120, 300]


def test_two_runs_in_the_same_second_still_get_a_folder_each(tmp_path):
    """Takes deliberate effort to do, and is exactly when losing one is worst."""
    widget = _widget_in(tmp_path)
    _fit(widget, _locs(n=300, seed=1))
    _fit(widget, _locs(n=120, seed=2))
    assert len(_runs(tmp_path)) == 2


def test_each_folder_records_the_settings_that_produced_it(tmp_path):
    widget = _widget_in(tmp_path)
    _fit(widget, min_ng=400.0)
    time.sleep(1.05)
    _fit(widget, min_ng=900.0)

    thresholds = [
        json.loads((r / "metadata.json").read_text(encoding="utf-8"))
        ["localization_2d"]["min_net_gradient"]
        for r in _runs(tmp_path)
    ]
    assert thresholds == [400.0, 900.0]


def test_the_metadata_is_the_whole_of_it(tmp_path):
    """Not a summary - the same dict an export writes, so a run can be
    reproduced from its own folder."""
    widget = _widget_in(tmp_path)
    _fit(widget)
    metadata = json.loads((_runs(tmp_path)[0] / "metadata.json").read_text(encoding="utf-8"))
    for section in ("localization_2d", "smlm_rendering", "linking", "diffusion",
                    "filter_bounds", "immobility", "run"):
        assert section in metadata, section
    assert metadata["run"]["stamp"]
    assert metadata["pixel_size_nm_per_px"] == pytest.approx(161.0)


def test_what_is_saved_is_the_fit_not_the_filtered_view(tmp_path):
    """Filters are recorded and can be re-applied; a discarded localization
    cannot be recovered from a filtered table."""
    widget = _widget_in(tmp_path)
    df = _locs(n=300)
    widget._ingest_localization_dataframe(df, "fitted", True)
    lower, _upper = widget.filter_controls["intensity [photon]"]
    lower.setValue(1500.0)
    widget.apply_filters()
    assert len(widget.df_filtered) < len(df)

    widget._autosave_localization_run(df)
    assert _pump_until(lambda: widget._autosave_worker_ref is None)
    written = pd.read_csv(_runs(tmp_path)[0] / "data" / widget_mod.LOCS_RUN_FILENAME)
    assert len(written) == len(df)


def test_it_can_be_turned_off(tmp_path):
    widget = _widget_in(tmp_path)
    widget.loc_autosave_box.setChecked(False)
    _fit(widget)
    assert not (tmp_path / widget_mod.ANALYSIS_ROOT).exists()


def test_it_is_on_by_default():
    assert make_widget().loc_autosave_box.isChecked()


def test_results_land_beside_the_data_not_the_working_directory(tmp_path):
    widget = _widget_in(tmp_path)
    assert widget._analysis_base_dir() == tmp_path

    # A loaded CSV wins: it is the more specific statement of where the data is.
    other = tmp_path / "elsewhere"
    other.mkdir()
    widget.csv_edit.setText(str(other / "locs.csv"))
    assert widget._analysis_base_dir() == other


# --- exports use the same scheme ----------------------------------------------


def test_an_export_gets_a_dated_folder_too(tmp_path):
    widget = _widget_in(tmp_path)
    widget._ingest_localization_dataframe(_locs(), "loaded", True)
    widget.export_analysis()
    assert _pump_until(lambda: widget._export_worker_ref is None), "export never finished"

    runs = _runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].name.endswith("_export")
    assert (runs[0] / "data" / "localizations_filtered.csv").is_file()


def test_fits_and_exports_sit_side_by_side_and_are_told_apart(tmp_path):
    widget = _widget_in(tmp_path)
    _fit(widget)
    time.sleep(1.05)
    widget.export_analysis()
    assert _pump_until(lambda: widget._export_worker_ref is None)

    kinds = sorted(d.name.rsplit("_", 1)[-1] for d in _runs(tmp_path))
    assert kinds == ["export", "localization"]


# --- finding them again --------------------------------------------------------


def test_autoloading_picks_the_most_recent_run(tmp_path):
    widget = _widget_in(tmp_path)
    _fit(widget, _locs(n=300, seed=1))
    time.sleep(1.05)
    _fit(widget, _locs(n=120, seed=2))

    found = widget._find_companion_file(
        tmp_path / "stack.tif", widget_mod.LOCS_FILENAME_PATTERNS,
        widget_mod.LOCS_ANALYSIS_SUBPATH)
    assert found is not None
    assert found.parent.parent == _runs(tmp_path)[-1]
    assert len(pd.read_csv(found)) == 120


def test_analyses_made_before_runs_were_dated_are_still_found(tmp_path):
    """The old scheme was analysis, analysis_2, analysis_3 beside the data."""
    widget = _widget_in(tmp_path)
    legacy = tmp_path / "analysis_3" / "data"
    legacy.mkdir(parents=True)
    _locs(n=77).to_csv(legacy / "localizations_filtered.csv", index=False)

    found = widget._find_companion_file(
        tmp_path / "stack.tif", widget_mod.LOCS_FILENAME_PATTERNS,
        widget_mod.LOCS_ANALYSIS_SUBPATH)
    assert found is not None
    assert len(pd.read_csv(found)) == 77


def test_a_dated_run_is_preferred_over_an_older_numbered_one(tmp_path):
    widget = _widget_in(tmp_path)
    legacy = tmp_path / "analysis" / "data"
    legacy.mkdir(parents=True)
    _locs(n=77).to_csv(legacy / "localizations_filtered.csv", index=False)
    _fit(widget, _locs(n=300))

    found = widget._find_companion_file(
        tmp_path / "stack.tif", widget_mod.LOCS_FILENAME_PATTERNS,
        widget_mod.LOCS_ANALYSIS_SUBPATH)
    assert len(pd.read_csv(found)) == 300


def test_a_broken_analysis_folder_does_not_stop_the_search(tmp_path):
    widget = _widget_in(tmp_path)
    (tmp_path / widget_mod.ANALYSIS_ROOT).mkdir()
    (tmp_path / widget_mod.ANALYSIS_ROOT / "not-a-run").write_text("x", encoding="utf-8")
    assert widget._find_companion_file(
        tmp_path / "stack.tif", widget_mod.LOCS_FILENAME_PATTERNS,
        widget_mod.LOCS_ANALYSIS_SUBPATH) is None


# --- where results go, and what they are called --------------------------------
#
# Left to itself this produced analysis/data/analysis/<run>/data/localizations.csv
# and scattered gigabyte TIFFs called
# localizations_filtered_movie_gaussian_global_os10_data.tif across two folders.
# Both come from the same mistake: taking the folder of whatever file happened
# to be open as the place to put things.


@pytest.mark.parametrize("relative", [
    "stack.ome.tif",
    "analysis/data/localizations_filtered.csv",
    "analysis/2026-09-05_190813_localization/data/localizations.csv",
    "analysis/data/analysis/2026-09-05_222411_localization/data/localizations.csv",
])
def test_results_climb_back_out_to_the_dataset(tmp_path, relative):
    """Reloading an exported table used to nest one analysis/ inside the last."""
    widget = make_widget()
    assert widget._dataset_dir(tmp_path / relative) == tmp_path


def test_reloading_an_exported_table_does_not_nest(tmp_path):
    widget = _widget_in(tmp_path)
    widget.csv_edit.setText(str(
        tmp_path / "analysis" / "2026-09-05_190813_localization"
        / "data" / "localizations.csv"))
    assert widget._analysis_base_dir() == tmp_path

    folder = widget._make_analysis_folder(widget._analysis_base_dir(), "localization")
    assert folder.parent == tmp_path / widget_mod.ANALYSIS_ROOT
    assert "analysis" not in folder.relative_to(tmp_path).parts[1:]


def test_a_dataset_folder_that_merely_contains_the_word_is_left_alone(tmp_path):
    """Only a folder actually called analysis is climbed out of."""
    widget = make_widget()
    deep = tmp_path / "reanalysis_2026" / "stack.tif"
    assert widget._dataset_dir(deep) == tmp_path / "reanalysis_2026"


# --- renders --------------------------------------------------------------------


def _render_ready(tmp_path):
    widget = _widget_in(tmp_path)
    widget.csv_edit.setText("")
    return widget, {"mode": "gaussian_global", "oversampling": 10}


def test_renders_go_to_a_dated_run_folder(tmp_path):
    widget, info = _render_ready(tmp_path)
    path = Path(widget._default_render_path("image", info))
    assert path.parent.parent == tmp_path / widget_mod.ANALYSIS_ROOT
    assert path.parent.name.endswith("_render")


def test_a_render_is_named_for_the_layer_it_is(tmp_path):
    widget, info = _render_ready(tmp_path)
    assert Path(widget._default_render_path("image", info)).name == "smlm_render.tif"
    assert Path(widget._default_render_path("movie", info)).name == "smlm_render_movie.tif"

    widget.render_layer_name_edit.setText("smlm_render_immobile")
    assert Path(widget._default_render_path("image", info)).name \
        == "smlm_render_immobile.tif"


def test_a_composite_cannot_overwrite_the_render_it_came_from(tmp_path):
    widget, info = _render_ready(tmp_path)
    plain = Path(widget._default_render_path("image", info)).name
    composite = Path(widget._default_render_path("image", info, "composite")).name
    assert plain != composite
    assert composite == "smlm_render_composite.tif"


def test_an_image_and_a_movie_from_one_render_land_together(tmp_path):
    widget, info = _render_ready(tmp_path)
    image = Path(widget._default_render_path("image", info))
    movie = Path(widget._default_render_path("movie", info))
    assert image.parent == movie.parent


def test_a_new_render_starts_a_new_folder(tmp_path):
    widget, info = _render_ready(tmp_path)
    first = Path(widget._default_render_path("image", info)).parent
    first.mkdir(parents=True, exist_ok=True)      # as saving would

    widget._render_save_folder = None             # what finishing a render does
    second = Path(widget._default_render_path("image", info)).parent
    assert second != first


def test_the_folder_is_not_created_until_something_is_saved(tmp_path):
    widget, info = _render_ready(tmp_path)
    Path(widget._default_render_path("image", info))
    assert not (tmp_path / widget_mod.ANALYSIS_ROOT).exists()
