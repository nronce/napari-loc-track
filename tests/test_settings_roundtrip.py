"""Restoring parameters from a previous run's metadata.json.

The contract: everything the exporter records as a *parameter* comes back,
nothing a run *produced* is ever applied, and a file from an older version
restores what it knows without failing on what it doesn't.
"""
import json
import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("qtpy", reason="needs Qt")
widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from qtpy.QtWidgets import QApplication

from test_widget_interaction import _StubViewer, make_widget  # same stub viewer and lifetime handling


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def widget(qapp):
    return make_widget()


def _localizations():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "frame": rng.integers(0, 50, n),
        "x [nm]": rng.uniform(0, 20000, n),
        "y [nm]": rng.uniform(0, 20000, n),
        "sigma [nm]": rng.uniform(80, 400, n),
        "intensity [photon]": rng.uniform(100, 5000, n),
    })


# --- pure extraction ----------------------------------------------------------


def test_only_parameters_are_extracted():
    """Counts, timestamps and paths describe a past run and must not be applied."""
    metadata = {
        "exported_at": "2026-01-01T00:00:00",
        "software": {"napari_version": "0.8.0"},
        "source_csv": "C:/somewhere/else.csv",
        "n_localizations_total": 999999,
        "pixel_size_nm_per_px": 108.0,
        "linking": {"search_range_nm": 400.0, "n_trajectories": 4321},
    }
    values, _notes = widget_mod.settings_from_metadata(metadata)
    assert values == {"pixel_size_box": 108.0, "search_box": 400.0}


def test_missing_sections_are_simply_absent():
    values, notes = widget_mod.settings_from_metadata({"pixel_size_nm_per_px": 100.0})
    assert values == {"pixel_size_box": 100.0}
    assert notes == []


@pytest.mark.parametrize("junk", [None, [], "text", 42])
def test_non_dict_input_is_rejected_cleanly(junk):
    values, notes = widget_mod.settings_from_metadata(junk)
    assert values == {}
    assert notes


def test_legacy_distance_in_nm_is_converted():
    values, notes = widget_mod.settings_from_metadata(
        {"distance_bounds_nm": {"min": 500.0, "max": 3000.0}}
    )
    assert values["dist_min_box"] == pytest.approx(0.5)
    assert values["dist_max_box"] == pytest.approx(3.0)
    assert any("nm to µm" in n for n in notes)


def test_current_distance_key_wins_over_legacy():
    values, _notes = widget_mod.settings_from_metadata({
        "distance_bounds_um": {"min": 1.0, "max": 2.0},
        "distance_bounds_nm": {"min": 500.0, "max": 3000.0},
    })
    assert values["dist_min_box"] == pytest.approx(1.0)
    assert values["dist_max_box"] == pytest.approx(2.0)


@pytest.mark.parametrize("section", ["diffusion", "linking"])
def test_frame_interval_is_used_when_fps_is_absent(section):
    values, notes = widget_mod.settings_from_metadata({section: {"frame_interval_ms": 25.0}})
    assert values["fps_box"] == pytest.approx(40.0)
    assert any("frame interval" in n for n in notes)


@pytest.mark.parametrize("section", ["diffusion", "linking"])
def test_fps_wins_when_both_are_present(section):
    values, _notes = widget_mod.settings_from_metadata(
        {section: {"fps": 100.0, "frame_interval_ms": 999.0}}
    )
    assert values["fps_box"] == pytest.approx(100.0)


def test_timing_read_from_either_section():
    """It moved from the diffusion section to linking; old files still work."""
    legacy = widget_mod.settings_from_metadata({"diffusion": {"fps": 40.0}})[0]
    current = widget_mod.settings_from_metadata({"linking": {"fps": 40.0}})[0]
    assert legacy["fps_box"] == pytest.approx(40.0)
    assert current["fps_box"] == pytest.approx(40.0)
    # If a file somehow has both, the current location wins.
    both = widget_mod.settings_from_metadata(
        {"diffusion": {"fps": 10.0}, "linking": {"fps": 40.0}}
    )[0]
    assert both["fps_box"] == pytest.approx(40.0)


# --- round trip through the widget -------------------------------------------


def test_full_round_trip(widget, tmp_path):
    """Change everything, export, reset, load it back, and compare."""
    widget.df = _localizations()
    widget.column_map = widget_mod.infer_column_map(widget.df.columns)
    widget._build_filter_tab_contents()

    changed = {
        "pixel_size_box": 108.0, "loc_gain_box": 2.5, "loc_offset_box": 250.0,
        "loc_min_ng_box": 1234.0, "search_box": 420.0, "memory_box": 3,
        "min_traj_box": 4, "max_lagtime_box": 9, "d_min_length_box": 11,
        "fps_box": 40.0, "d_min_box": 0.02, "d_max_box": 7.5,
        "msd_sample_box": 6, "dist_min_box": 0.25, "dist_max_box": 12.0,
        "dur_min_box": 0.1, "dur_max_box": 30.0, "marker_size_box": 9.0,
        "line_width_box": 3.0, "all_tracks_line_width_box": 1.7,
    }
    for attr, value in changed.items():
        getattr(widget, attr).setValue(value)
    widget.loc_backend_box.setCurrentText("mle")
    widget.d_colormap_box.setCurrentText("viridis")
    widget.marker_choice.setCurrentText("s")
    widget.color_trajectories_box.setChecked(True)
    widget.shift_frame_numbers(-1)
    widget.persist_tracks_box.setChecked(True)
    widget.filter_controls["sigma [nm]"][0].setValue(120.0)
    widget.filter_controls["sigma [nm]"][1].setValue(310.0)

    folder = tmp_path / "export"
    folder.mkdir()
    widget._write_metadata(folder, "source.csv")

    # Wipe the settings by rebuilding the widget, then restore them.
    fresh = make_widget()
    fresh.df = _localizations()
    fresh.column_map = widget_mod.infer_column_map(fresh.df.columns)
    fresh._build_filter_tab_contents()
    applied = fresh.load_settings_from_metadata(str(folder / "metadata.json"))
    assert applied

    for attr, value in changed.items():
        assert getattr(fresh, attr).value() == pytest.approx(value), attr
    assert fresh.loc_backend_box.currentText() == "mle"
    assert fresh.d_colormap_box.currentText() == "viridis"
    assert fresh.marker_choice.currentText() == "s"
    assert fresh.color_trajectories_box.isChecked()
    assert fresh._frame_shift == -1
    assert fresh.persist_tracks_box.isChecked()
    assert fresh.filter_controls["sigma [nm]"][0].value() == pytest.approx(120.0)
    assert fresh.filter_controls["sigma [nm]"][1].value() == pytest.approx(310.0)
    # fps and frame interval must stay coupled after a restore.
    assert fresh.frame_interval_box.value() == pytest.approx(25.0)


def test_filter_bounds_wait_for_their_data(widget, tmp_path):
    """Settings are usually loaded before the data; bounds must not be lost."""
    metadata = {"filter_bounds": {"sigma [nm]": {"min": 150.0, "max": 350.0}}}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    widget.load_settings_from_metadata(str(path))  # no data loaded yet
    assert widget._pending_filter_bounds

    widget.df = _localizations()
    widget.column_map = widget_mod.infer_column_map(widget.df.columns)
    widget._build_filter_tab_contents()

    lower, upper = widget.filter_controls["sigma [nm]"]
    assert lower.value() == pytest.approx(150.0)
    assert upper.value() == pytest.approx(350.0)
    assert not widget._pending_filter_bounds


@pytest.mark.parametrize("settings_first", [True, False])
def test_restored_bounds_actually_filter_the_data(widget, tmp_path, settings_first):
    """Whichever order data and settings arrive in, the bounds must take effect.

    Restoring the numbers into the spin boxes is not enough: if the data is not
    re-filtered, the full table stays on screen and everything downstream runs
    on unfiltered localizations.
    """
    metadata = {"filter_bounds": {"sigma [nm]": {"min": 150.0, "max": 250.0}}}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    data = _localizations()
    in_sigma = int(((data["sigma [nm]"] >= 150.0) & (data["sigma [nm]"] <= 250.0)).sum())
    assert 0 < in_sigma < len(data)  # the bounds must actually cut something

    if settings_first:
        widget.load_settings_from_metadata(str(path))
        widget._ingest_localization_dataframe(data, "loaded", True)
    else:
        widget._ingest_localization_dataframe(data, "loaded", True)
        widget.load_settings_from_metadata(str(path))

    assert len(widget.df_filtered) < len(data), (
        f"all {len(data)} rows kept; the restored bounds were not applied to the data"
    )
    assert widget.df_filtered["sigma [nm]"].between(150.0, 250.0).all()
    # The filtered table must be exactly what the current controls describe.
    bounds = {col: (lo.value(), hi.value()) for col, (lo, hi) in widget.filter_controls.items()}
    assert len(widget.df_filtered) == len(widget_mod.apply_numeric_filters(data, bounds))


def test_unknown_columns_and_values_do_not_raise(widget, tmp_path):
    metadata = {
        "pixel_size_nm_per_px": 120.0,
        "localization_2d": {"fit_backend": "a-backend-that-does-not-exist"},
        "coloring": {"colormap": "not-a-colormap"},
        "filter_bounds": {"no such column [nm]": {"min": 1.0, "max": 2.0}},
        "some_future_section": {"whatever": 1},
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    applied, skipped, _notes = widget.apply_settings(metadata)
    assert "pixel_size_box" in applied
    assert widget.pixel_size_box.value() == pytest.approx(120.0)
    # Values this build cannot represent are reported, not applied.
    assert "loc_backend_box" in skipped
    assert "d_colormap_box" in skipped
    assert widget.loc_backend_box.currentText() != "a-backend-that-does-not-exist"


def test_unreadable_file_is_reported_not_raised(widget, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert widget.load_settings_from_metadata(str(path)) is None
    assert widget.load_settings_from_metadata(str(tmp_path / "missing.json")) is None


def test_out_of_range_values_are_clamped_not_crashed(widget):
    widget.apply_settings({"linking": {"memory": 10 ** 9, "search_range_nm": -50.0}})
    assert widget.memory_box.value() == widget.memory_box.maximum()
    assert widget.search_box.value() == widget.search_box.minimum()


def test_histogram_view_is_restored_only_when_decoupled(widget):
    state = widget._metric_hist_widgets["D"]
    metadata = {"metric_histogram_display": {"D": {
        "bins": 44, "view_min": 0.3, "view_max": 4.0, "follow_filter": False,
    }}}
    widget.apply_settings(metadata)
    assert state["bins_box"].value() == 44
    assert not state["follow_box"].isChecked()
    assert state["view_max_box"].value() == pytest.approx(4.0)

    # With follow_filter on, the view is derived from the bounds instead.
    widget.apply_settings({
        "diffusion": {"d_min": 0.01, "d_max": 2.0},
        "metric_histogram_display": {"D": {
            "bins": 30, "view_min": 99.0, "view_max": 123.0, "follow_filter": True,
        }},
    })
    assert state["follow_box"].isChecked()
    assert state["view_max_box"].value() == pytest.approx(2.0)


# --- camera gain ---------------------------------------------------------------


def test_the_gain_defaults_to_the_camera_rather_than_to_one():
    """A gain of 1 says the camera is photon-counting, which almost none are,
    and every photon count and localization precision is scaled by whatever the
    real figure is."""
    widget = make_widget()
    assert widget.loc_gain_box.value() == pytest.approx(1.3)
    assert widget_mod.DEFAULT_GAIN_ADU_PER_ELECTRON == 1.3


def test_the_gain_is_recorded_in_electrons():
    widget = make_widget()
    widget.loc_gain_box.setValue(2.4)
    metadata = widget._collect_metadata(None)
    assert metadata["localization_2d"]["gain_adu_per_electron"] == pytest.approx(2.4)

    restored = make_widget()
    restored.apply_settings(metadata)
    assert restored.loc_gain_box.value() == pytest.approx(2.4)


def test_a_run_recorded_under_the_old_per_photon_key_still_restores():
    """Same number, differently named - the division always produced electrons."""
    widget = make_widget()
    applied, _skipped, notes = widget.apply_settings(
        {"localization_2d": {"gain_adu_per_photon": 1.85}})
    assert widget.loc_gain_box.value() == pytest.approx(1.85)
    assert "loc_gain_box" in applied
    assert any("per photon" in note for note in notes)


# --- restoring a run must not move the instrument silently ---------------------


def test_restoring_a_run_names_the_instrument_settings_it_changed():
    """Loading data next to a previous run restores that run's settings - which
    is right, since the localizations were computed with them - but a corrected
    calibration being reverted by opening a folder has to be visible."""
    widget = make_widget()
    widget.loc_gain_box.setValue(1.3)
    widget.pixel_size_box.setValue(161.0)

    _applied, _skipped, notes = widget.apply_settings({
        "pixel_size_nm_per_px": 108.0,
        "localization_2d": {"gain_adu_per_electron": 1.0},
    })
    joined = " | ".join(notes)
    assert "Camera gain 1.3" in joined and "1 ADU" in joined
    assert "Pixel size 161.0 nm/px -> 108.0 nm/px" in joined


def test_settings_that_did_not_move_are_not_announced():
    widget = make_widget()
    widget.loc_gain_box.setValue(1.3)
    _applied, _skipped, notes = widget.apply_settings(
        {"localization_2d": {"gain_adu_per_electron": 1.3}})
    assert not any("Camera gain" in note for note in notes)
