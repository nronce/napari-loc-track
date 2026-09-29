"""Filters kept as someone's defaults, and filters taken from another analysis.

"Set as default" keeps the filters as they are for every table loaded
afterwards and every later session; "Load filters from..." takes just the
filters of another run or session. Neither carries x, y or frame bounds, which
say where one dataset lies rather than what counts as a good localization, and
a saved bound only holds on the side that was moved - a side left at the data's
own range keeps following the data.
"""
import json

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_widget_interaction import make_widget  # noqa: E402


@pytest.fixture(autouse=True)
def config_dir(tmp_path, monkeypatch):
    folder = tmp_path / "config"
    monkeypatch.setenv("NAPARI_LOC_TRACK_CONFIG_DIR", str(folder))
    return folder


def _table(n=200, seed=0, x_span=10000.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "frame": rng.integers(0, 50, n),
        "x [nm]": rng.uniform(0, x_span, n),
        "y [nm]": rng.uniform(0, x_span, n),
        "sigma [nm]": rng.uniform(80, 400, n),
        "uncertainty [nm]": rng.uniform(5, 150, n),
        "net_gradient": rng.uniform(500, 5000, n),
    })


def _loaded(table=None):
    widget = make_widget()
    widget._ingest_localization_dataframe(_table() if table is None else table,
                                          "loaded", True)
    return widget


def _bounds(widget, column):
    lower, upper = widget.filter_controls[column]
    return lower.value(), upper.value()


def _narrow(widget):
    widget.filter_controls["sigma [nm]"][1].setValue(250.0)
    widget.filter_controls["net_gradient"][0].setValue(900.0)
    widget.filter_controls["x [nm]"][0].setValue(3000.0)
    widget.d_min_box.setValue(0.002)
    widget.distance_filter_box.setChecked(True)


def test_saved_filters_hold_for_the_next_table_and_the_next_session(config_dir):
    widget = _loaded()
    _narrow(widget)
    widget.save_filter_defaults()
    saved = json.loads((config_dir / widget_mod.FILTER_DEFAULTS_FILENAME).read_text())
    bounds = saved["settings"]["filter_bounds"]
    # only the sides that were moved, and never where the data lies
    assert bounds == {"sigma [nm]": {"max": 250.0}, "net_gradient": {"min": 900.0}}

    later = _loaded(_table(seed=1, x_span=30000.0))           # a new session
    assert _bounds(later, "sigma [nm]") == (0.0, 250.0)
    lower, upper = _bounds(later, "net_gradient")
    assert lower == pytest.approx(900.0)
    # the side nobody moved follows this dataset
    assert upper == pytest.approx(later.df["net_gradient"].max(), abs=1e-5)
    assert _bounds(later, "x [nm]")[0] == pytest.approx(later.df["x [nm]"].min(), abs=1e-5)
    assert later.d_min_box.value() == pytest.approx(0.002)
    assert later.distance_filter_box.isChecked()
    assert "yours" in later.filter_defaults_label.text()


def test_reset_goes_back_to_the_saved_defaults():
    widget = _loaded()
    _narrow(widget)
    widget.save_filter_defaults()
    widget.filter_controls["sigma [nm]"][1].setValue(120.0)
    widget.reset_filters()
    assert _bounds(widget, "sigma [nm]") == (0.0, 250.0)


def test_forgetting_the_defaults_brings_back_the_built_in_ones(config_dir):
    widget = _loaded()
    _narrow(widget)
    widget.save_filter_defaults()
    widget.forget_filter_defaults()
    assert not (config_dir / widget_mod.FILTER_DEFAULTS_FILENAME).exists()
    # what is on screen stays; what comes next does not
    assert _bounds(widget, "sigma [nm]") == (0.0, 250.0)
    later = _loaded()
    assert _bounds(later, "sigma [nm]") == widget_mod.SIGMA_DEFAULT_BOUNDS_NM
    assert "built-in" in later.filter_defaults_label.text()


def test_a_damaged_defaults_file_leaves_the_built_in_ones(config_dir):
    config_dir.mkdir(parents=True)
    (config_dir / widget_mod.FILTER_DEFAULTS_FILENAME).write_text("{not json")
    widget = _loaded()
    assert _bounds(widget, "sigma [nm]") == widget_mod.SIGMA_DEFAULT_BOUNDS_NM
    assert "built-in ones are used" in widget.log_box.toPlainText()


def _run_metadata():
    return {
        "pixel_size_nm_per_px": 108.0,
        "linking": {"search_range_nm": 777.0},
        "filter_bounds": {"sigma [nm]": {"min": 50.0, "max": 300.0},
                          "x [nm]": {"min": 4000.0, "max": 5000.0},
                          "frame": {"min": 10.0, "max": 20.0}},
        "diffusion": {"d_min": 0.003, "d_max": 2.0},
        "dynamics_filter": {"D": True},
    }


@pytest.mark.parametrize("as_session", [False, True])
def test_only_the_filters_are_taken_from_another_analysis(tmp_path, as_session):
    data = _run_metadata()
    if as_session:
        data = {"napari_loc_track_session": 1, "settings": data, "sources": {}}
    path = tmp_path / ("s.loctrack-session.json" if as_session else "metadata.json")
    path.write_text(json.dumps(data))
    widget = _loaded()
    search_before = widget.search_box.value()
    pixel_before = widget.pixel_size_box.value()
    x_before = _bounds(widget, "x [nm]")
    widget.load_filters_from(path)
    assert _bounds(widget, "sigma [nm]") == (50.0, 300.0)
    assert widget.d_min_box.value() == pytest.approx(0.003)
    assert widget.d_filter_box.isChecked()
    # not the filters: left alone
    assert _bounds(widget, "x [nm]") == x_before
    assert widget.search_box.value() == search_before
    assert widget.pixel_size_box.value() == pixel_before


def test_the_settings_loader_takes_a_session_file_too(tmp_path):
    path = tmp_path / "other.loctrack-session.json"
    path.write_text(json.dumps({"napari_loc_track_session": 1, "sources": {},
                                "settings": {"linking": {"search_range_nm": 777.0}}}))
    widget = make_widget()
    widget.load_settings_from_metadata(path)
    assert widget.search_box.value() == pytest.approx(777.0)


def test_bounds_restored_before_their_table_wait_for_it_and_name_every_column():
    widget = make_widget()
    bounds = {"sigma [nm]": {"min": 0.0, "max": 250.0},
              "net_gradient": {"min": 900.0, "max": 1e9},
              "photons": {"min": 100.0, "max": 1e9},
              "offset [photon]": {"min": 0.0, "max": 50.0}}
    _applied, _skipped, notes = widget.apply_settings({"filter_bounds": bounds},
                                                      include_instrument=False)
    note = next(n for n in notes if "filter bound" in n)
    assert "kept for when the localizations are loaded" in note
    assert all(name in note for name in bounds)          # every one, no "..."
    messages = []
    widget.log = messages.append
    widget._ingest_localization_dataframe(_table(), "loaded", True)
    assert _bounds(widget, "sigma [nm]")[1] == pytest.approx(250.0)
    unmatched = next(m for m in messages if "match no column" in m)
    assert "offset [photon]" in unmatched and "photons" in unmatched
    assert "net_gradient" not in unmatched.split("(it has")[0]


def test_a_run_records_every_column_of_its_table():
    widget = _loaded()
    metadata = widget._collect_metadata(None)
    assert metadata["localization_columns"] == list(_table().columns)
