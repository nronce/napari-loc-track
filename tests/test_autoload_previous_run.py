"""Picking up where the last run left off, controls included.

When an image is loaded with no CSV, the plugin hunts for the localizations a
previous run exported beside it. That table is usually a *filtered* export, so
loading it without the parameters that produced it puts data on screen the
controls actively misdescribe - bounds that were applied showing as wide open, a
pixel size that is not the one those coordinates were computed with. Carrying on
where a run left off only means anything if the settings come too.
"""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_widget_interaction import make_widget  # noqa: E402


def _previous_run(tmp_path, **overrides):
    """An analysis folder as the exporter writes one: metadata.json + data/."""
    analysis = tmp_path / "analysis_2"
    (analysis / "data").mkdir(parents=True)

    table = pd.DataFrame({
        "frame": [1, 2, 3, 4],
        "x [nm]": [100.0, 200.0, 300.0, 400.0],
        "y [nm]": [100.0, 200.0, 300.0, 400.0],
        "intensity [photon]": [500.0, 900.0, 1500.0, 2500.0],
        "uncertainty [nm]": [10.0, 12.0, 14.0, 16.0],
    })
    table.to_csv(analysis / "data" / "localizations_filtered.csv", index=False)

    metadata = {
        "exported_at": "2026-08-03 14:36",
        "pixel_size_nm_per_px": 108.5,
        "localization_2d": {"box_size_px": 9, "min_net_gradient": 4321},
        "linking": {"search_range_nm": 777.0, "memory": 4},
        "smlm_rendering": {"oversampling": 7, "global_sigma_nm": 45.0},
        # the shape the exporter really writes, per column
        "filter_bounds": {"intensity [photon]": {"min": 800.0, "max": 2000.0}},
    }
    metadata.update(overrides)
    with open(analysis / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
    return analysis


def _load_stack_beside(widget, tmp_path, name="stack.tif"):
    """Run the load handler as a finished worker would, with no CSV chosen."""
    widget.csv_edit.setText("")
    widget.image_edit.setText(str(tmp_path / name))
    widget._on_load_finished(
        (None, np.zeros((6, 16, 16), np.uint16), "decoded", None),
        "", str(tmp_path / name))


def test_the_localizations_of_the_previous_run_are_found(tmp_path):
    _previous_run(tmp_path)
    widget = make_widget()
    _load_stack_beside(widget, tmp_path)
    assert widget.df is not None and len(widget.df) == 4
    assert "Auto-detected and loaded" in widget.log_box.toPlainText()


def test_the_analysis_parameters_of_that_run_come_with_it(tmp_path):
    _previous_run(tmp_path)
    widget = make_widget()
    _load_stack_beside(widget, tmp_path)

    assert widget.loc_box_size.value() == 9
    assert widget.loc_min_ng_box.value() == 4321
    assert widget.search_box.value() == pytest.approx(777.0)
    assert widget.memory_box.value() == 4
    assert widget.render_oversampling_box.value() == 7
    assert widget.render_sigma_box.value() == pytest.approx(45.0)


def test_the_filters_that_were_applied_are_shown_as_applied(tmp_path):
    """The exported table is already filtered; the controls have to say so."""
    _previous_run(tmp_path)
    widget = make_widget()
    _load_stack_beside(widget, tmp_path)

    controls = widget.filter_controls.get("intensity [photon]")
    assert controls is not None, "the filter row for that column was never built"
    assert controls[0].value() == pytest.approx(800.0)
    assert controls[1].value() == pytest.approx(2000.0)


def test_it_says_where_the_settings_came_from(tmp_path):
    _previous_run(tmp_path)
    widget = make_widget()
    _load_stack_beside(widget, tmp_path)
    log = widget.log_box.toPlainText()
    assert "Restored" in log and "metadata.json" in log
    assert "2026-08-03 14:36" in log


def test_a_run_with_no_metadata_is_loaded_but_flagged(tmp_path):
    """Better to say the controls are not describing the data than to imply they are."""
    analysis = _previous_run(tmp_path)
    (analysis / "metadata.json").unlink()

    widget = make_widget()
    before = widget.pixel_size_box.value()
    _load_stack_beside(widget, tmp_path)

    assert widget.df is not None and len(widget.df) == 4
    assert widget.pixel_size_box.value() == pytest.approx(before)
    assert "No metadata.json" in widget.log_box.toPlainText()


def test_metadata_beside_the_table_is_found_too(tmp_path):
    """Not every layout puts the tables in a data/ subfolder."""
    analysis = _previous_run(tmp_path)
    (analysis / "metadata.json").rename(analysis / "data" / "metadata.json")

    widget = make_widget()
    _load_stack_beside(widget, tmp_path)
    assert widget.loc_min_ng_box.value() == 4321


def test_the_microscope_is_not_restored_from_a_run_nobody_asked_for(tmp_path):
    """Opening data found this run beside it; that is not a request to change
    the pixel size, which on this setup is measured by hand and cannot be
    derived from the metadata at all. The disagreement is reported instead."""
    _previous_run(tmp_path)
    widget = make_widget()
    widget.pixel_size_box.setValue(161.0)
    _load_stack_beside(widget, tmp_path)

    assert widget.pixel_size_box.value() == pytest.approx(161.0)
    assert "108.5" in widget.log_box.toPlainText()
    assert "left alone" in widget.log_box.toPlainText()


def test_unreadable_metadata_does_not_stop_the_data_loading(tmp_path):
    analysis = _previous_run(tmp_path)
    (analysis / "metadata.json").write_text("{not json", encoding="utf-8")

    widget = make_widget()
    _load_stack_beside(widget, tmp_path)
    assert widget.df is not None and len(widget.df) == 4
    assert "could not read it" in widget.log_box.toPlainText()


def test_a_csv_chosen_by_hand_is_left_alone(tmp_path):
    """Only the *auto-detected* table implies a run whose settings should follow."""
    _previous_run(tmp_path)
    widget = make_widget()
    before = widget.pixel_size_box.value()

    table = pd.DataFrame({"frame": [1], "x [nm]": [10.0], "y [nm]": [10.0]})
    widget._on_load_finished(
        (table, np.zeros((6, 16, 16), np.uint16), "decoded", None),
        str(tmp_path / "mine.csv"), str(tmp_path / "stack.tif"))
    assert widget.pixel_size_box.value() == pytest.approx(before)
