"""Saving a whole session, and getting back to it.

A session is a manifest, not an archive: it records where the data is, every
parameter that was set, and what had been computed, and the pipeline is re-run
from that on load. So the test that matters is not "did the file get written"
but "does a fresh widget, handed nothing but this file, end up in the same
state" - same trajectories, same diffusion coefficients, same view.

The other half is what a manifest cannot do. It points at files, so it has to
survive the folder being moved and it has to say something useful when the
files have changed underneath it rather than quietly restoring a different run.
"""
import gzip
import json
import os
import shutil
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib.util

import numpy as np
import pandas as pd
import pytest

_PKG_DIR = Path(__file__).resolve().parents[1] / "napari_loc_track"
if "napari_loc_track" not in sys.modules:
    _pkg = types.ModuleType("napari_loc_track")
    _pkg.__path__ = [str(_PKG_DIR)]
    sys.modules["napari_loc_track"] = _pkg
_spec = importlib.util.spec_from_file_location(
    "napari_loc_track._session", _PKG_DIR / "_session.py"
)
session_io = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("napari_loc_track._session", session_io)
_spec.loader.exec_module(session_io)


# --- the file format, without any of the stack ---------------------------------


def test_the_suffix_is_added_once_however_the_name_arrives():
    for given in ("run1", "run1.loctrack-session.json", "run1.json"):
        path = session_io.session_path_for(Path("/data") / given)
        assert path.name.endswith(session_io.SESSION_SUFFIX)
        assert not path.name.endswith(
            session_io.SESSION_SUFFIX + session_io.SESSION_SUFFIX)


def test_the_localizations_file_is_named_after_the_session_not_after_its_suffix():
    path = session_io.session_path_for(Path("/data/run1"))
    assert session_io.locs_path_for(path).name == "run1_localizations.csv.gz"


def test_a_file_under_the_session_folder_is_recorded_both_ways(tmp_path):
    """Absolute survives the session file moving alone; relative survives the
    whole folder being moved, which is what actually happens to analysis
    folders on a shared drive."""
    data = tmp_path / "locs.csv"
    data.write_text("x\n1\n", encoding="utf-8")
    record = session_io.source_record(data, tmp_path)
    assert record["path"] == str(data)
    assert record["relative"] == "locs.csv"


def test_a_file_outside_the_session_folder_has_no_relative_path(tmp_path):
    record = session_io.source_record(Path("D:/elsewhere/stack.tif"), tmp_path)
    assert "relative" not in record


def test_resolving_prefers_the_neighbour_that_travelled_with_the_session(tmp_path):
    """After a copy, the absolute path still points at the original. The file
    beside the session is the one that belongs to this copy of it."""
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(); new.mkdir()
    (old / "locs.csv").write_text("original\n", encoding="utf-8")
    (new / "locs.csv").write_text("the copy\n", encoding="utf-8")
    record = session_io.source_record(old / "locs.csv", old)

    resolved = session_io.resolve_source(record, new)
    assert resolved.read_text(encoding="utf-8") == "the copy\n"


def test_resolving_falls_back_to_the_absolute_path(tmp_path):
    data = tmp_path / "stack.tif"
    data.write_text("x", encoding="utf-8")
    record = {"path": str(data), "relative": "not-here.tif"}
    assert session_io.resolve_source(record, tmp_path) == data


def test_a_source_that_is_gone_resolves_to_nothing(tmp_path):
    assert session_io.resolve_source({"path": str(tmp_path / "gone.tif")}, tmp_path) is None


def test_missing_sources_are_listed_by_name(tmp_path):
    manifest = {"sources": {"image": {"path": str(tmp_path / "gone.tif")},
                            "localizations": None}}
    assert session_io.missing_sources(manifest, tmp_path) == [
        ("image", str(tmp_path / "gone.tif"))]


def test_a_json_file_that_is_not_a_session_is_refused(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"pixel_size_nm_per_px": 161.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a napari-loc-track session"):
        session_io.read_session(path)


def test_a_session_from_a_newer_format_is_refused_rather_than_half_applied(tmp_path):
    """Applying the half it understood would restore the wrong state while
    looking like it had worked."""
    path = tmp_path / "future.loctrack-session.json"
    path.write_text(json.dumps({session_io.SESSION_KEY: session_io.SESSION_FORMAT + 1}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="session format"):
        session_io.read_session(path)


# --- through the widget -------------------------------------------------------

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_render_widget import _pump_until  # noqa: E402
from test_widget_interaction import ensure_qapp  # noqa: E402

PIXEL_SIZE_NM = 161.0
_WIDGETS = []


def _widget():
    from napari.components import ViewerModel

    ensure_qapp()
    widget = widget_mod.LocalizationTrackingWidget(ViewerModel())
    _WIDGETS.append(widget)
    return widget


def _locs(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "frame": rng.integers(0, 40, n),
        "x [nm]": rng.uniform(0, 50000, n),
        "y [nm]": rng.uniform(0, 50000, n),
        "sigma [nm]": rng.uniform(50, 400, n),
        "intensity [photon]": rng.uniform(10, 9000, n),
    })


def _analysed(tmp_path, with_csv=True, seed=0):
    """A widget carrying a real analysis: localizations, trajectories, D."""
    widget = _widget()
    widget.pixel_size_box.setValue(PIXEL_SIZE_NM)
    widget.fps_box.setValue(31.882)
    frame = _locs(seed=seed)
    if with_csv:
        csv = tmp_path / "locs.csv"
        frame.to_csv(csv, index=False)
        widget.csv_edit.setText(str(csv))
        widget.load_data()
        assert _pump_until(lambda: widget._load_worker_ref is None), "load never finished"
    else:
        # As if fitted here: in memory, and on disk nowhere.
        widget._ingest_localization_dataframe(frame, "fitted", True)
    widget.search_box.setValue(900.0)
    widget.min_traj_box.setValue(3)
    widget.link_tracks()
    assert _pump_until(lambda: widget._link_worker_ref is None), "linking never finished"
    widget.compute_d()
    assert _pump_until(lambda: widget._compute_d_worker_ref is None), "D never finished"
    return widget


def _saved(widget, path):
    session_path = widget.save_session(str(path))
    assert _pump_until(lambda: session_path.is_file()), "the session was never written"
    return session_path


def _restored(session_path):
    widget = _widget()
    widget.load_session(str(session_path))
    assert _pump_until(lambda: widget._session_restore is None, timeout_s=180), \
        "the restore never finished"
    return widget


def _state(widget):
    return {
        "pixel_size": widget.pixel_size_box.value(),
        "fps": round(widget.fps_box.value(), 6),
        "search": widget.search_box.value(),
        "localizations": len(widget.df),
        "filtered": len(widget.df_filtered),
        "trajectories": int(widget.tracks["particle"].nunique()),
        "n_with_D": len(widget._track_diffusion_cache or {}),
    }


# --- what a session costs -----------------------------------------------------


def test_a_session_over_data_already_on_disk_is_a_manifest_and_nothing_else(tmp_path):
    """The whole point of the design: the stack and the table stay where they
    are, so the session is kilobytes rather than gigabytes."""
    widget = _analysed(tmp_path)
    session_path = _saved(widget, tmp_path / "run1")

    assert session_path.stat().st_size < 100_000
    assert not session_io.locs_path_for(session_path).exists()
    manifest = session_io.read_session(session_path)
    assert manifest["localizations_saved_with_session"] is False


def test_localizations_that_exist_nowhere_else_travel_with_the_session(tmp_path):
    """Fitted here and never exported: re-fitting costs minutes, and the table
    gzips to a fraction of the stack it came from, so it is written out."""
    widget = _analysed(tmp_path, with_csv=False)
    session_path = _saved(widget, tmp_path / "fitted")

    locs_path = session_io.locs_path_for(session_path)
    assert _pump_until(lambda: locs_path.is_file())
    assert session_io.read_session(session_path)["localizations_saved_with_session"] is True
    with gzip.open(locs_path, "rt", encoding="utf-8") as handle:
        back = pd.read_csv(handle)
    assert len(back) == len(widget.df)


def test_the_raw_stack_is_never_copied(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    stack = tmp_path / "stack.tif"
    tifffile.imwrite(stack, np.zeros((8, 32, 32), np.uint16))
    widget = _analysed(tmp_path)
    widget.image_edit.setText(str(stack))
    session_path = _saved(widget, tmp_path / "run1")

    written = {p.name for p in tmp_path.iterdir()} - {"locs.csv", "stack.tif"}
    assert written == {session_path.name}


# --- getting back ------------------------------------------------------------


def test_a_fresh_widget_handed_only_the_session_arrives_in_the_same_state(tmp_path):
    widget = _analysed(tmp_path)
    before = _state(widget)
    session_path = _saved(widget, tmp_path / "run1")

    assert _state(_restored(session_path)) == before


def test_the_trajectories_are_rebuilt_rather_than_stored(tmp_path):
    """Nothing about them is in the file - they come back because the linking
    parameters and the localizations do."""
    widget = _analysed(tmp_path)
    session_path = _saved(widget, tmp_path / "run1")
    text = session_path.read_text(encoding="utf-8")
    assert "particle" not in text
    assert session_io.read_session(session_path)["rebuild"]["link"] is True

    restored = _restored(session_path)
    pd.testing.assert_frame_equal(
        restored.tracks.reset_index(drop=True), widget.tracks.reset_index(drop=True))


def test_the_diffusion_coefficients_come_back(tmp_path):
    widget = _analysed(tmp_path)
    session_path = _saved(widget, tmp_path / "run1")
    restored = _restored(session_path)

    assert restored._track_diffusion_cache.keys() == widget._track_diffusion_cache.keys()
    for pid, D in widget._track_diffusion_cache.items():
        assert restored._track_diffusion_cache[pid] == pytest.approx(D)


def test_the_viewer_is_looking_where_it_was_left(tmp_path):
    widget = _analysed(tmp_path)
    widget.viewer.dims.set_current_step(0, 11)
    widget.viewer.camera.zoom = 3.25
    session_path = _saved(widget, tmp_path / "run1")

    restored = _restored(session_path)
    assert restored.viewer.dims.current_step[0] == 11
    assert restored.viewer.camera.zoom == pytest.approx(3.25)


def test_the_time_binning_factor_comes_back(tmp_path):
    widget = _analysed(tmp_path)
    widget.bin_factor_box.setValue(4)
    widget._apply_time_binning()
    offset, fps = widget.loc_offset_box.value(), widget.fps_box.value()
    session_path = _saved(widget, tmp_path / "run1")

    restored = _restored(session_path)
    assert restored.bin_factor_box.value() == 4
    # Recorded as applied, so restoring must not scale them a second time.
    assert restored.loc_offset_box.value() == pytest.approx(offset)
    assert restored.fps_box.value() == pytest.approx(fps)


def test_a_filter_bound_that_was_narrowed_comes_back_narrowed(tmp_path):
    widget = _analysed(tmp_path)
    lower, upper = widget.filter_controls["sigma [nm]"]
    upper.setValue(200.0)
    widget.apply_filters()
    filtered = len(widget.df_filtered)
    assert filtered < len(widget.df), "the test's own filter did nothing"
    session_path = _saved(widget, tmp_path / "run1")

    restored = _restored(session_path)
    assert restored.filter_controls["sigma [nm]"][1].value() == pytest.approx(200.0)
    assert len(restored.df_filtered) == filtered


# --- when the world has moved -------------------------------------------------


def test_a_session_survives_its_whole_folder_being_moved(tmp_path):
    """The analysis folder gets copied to another drive; the session inside it
    has to find the data that came with it, not the original it was saved from."""
    original = tmp_path / "original"
    original.mkdir()
    widget = _analysed(original)
    before = _state(widget)
    session_path = _saved(widget, original / "run1")

    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    shutil.rmtree(original)          # the absolute paths now point at nothing

    restored = _restored(moved / session_path.name)
    assert _state(restored) == before


def test_a_source_that_has_gone_missing_is_reported(tmp_path):
    widget = _analysed(tmp_path)
    session_path = _saved(widget, tmp_path / "run1")
    (tmp_path / "locs.csv").unlink()

    restored = _widget()
    restored.load_session(str(session_path))
    assert _pump_until(lambda: restored._session_restore is None, timeout_s=60)
    assert "not where it was saved" in restored.log_box.toPlainText()


def test_a_source_that_has_changed_is_flagged_rather_than_restored_quietly(tmp_path):
    """Rebuilding from an edited file lands somewhere else entirely, and the
    counts are the cheapest way to notice it happened."""
    widget = _analysed(tmp_path)
    session_path = _saved(widget, tmp_path / "run1")
    _locs(n=1500, seed=7).to_csv(tmp_path / "locs.csv", index=False)

    restored = _restored(session_path)
    log = restored.log_box.toPlainText()
    assert "not to the same numbers" in log
    assert "localizations: 3000 then, 1500 now" in log


def test_a_clean_restore_says_so_without_qualification(tmp_path):
    widget = _analysed(tmp_path)
    session_path = _saved(widget, tmp_path / "run1")
    log = _restored(session_path).log_box.toPlainText()
    assert "restored." in log
    assert "not to the same numbers" not in log


def test_trajectories_read_from_a_file_are_not_re_linked(tmp_path):
    """They were not necessarily produced by these linking parameters, so
    re-linking would quietly replace them with different trajectories."""
    widget = _analysed(tmp_path)
    widget._tracks_source_path = tmp_path / "trajectories.csv"
    widget.tracks.to_csv(widget._tracks_source_path, index=False)
    session_path = _saved(widget, tmp_path / "run1")

    manifest = session_io.read_session(session_path)
    assert manifest["rebuild"]["link"] is False
    assert manifest["sources"]["trajectories"]["relative"] == "trajectories.csv"


def test_a_session_with_no_data_files_restores_the_settings_alone(tmp_path):
    widget = _widget()
    widget.pixel_size_box.setValue(108.3)
    widget.search_box.setValue(1234.0)
    widget.image_edit.setText("")
    # Nothing loaded at all, so there is nothing to point at.
    assert widget.save_session(str(tmp_path / "empty")) is None

    widget._ingest_localization_dataframe(_locs(n=50), "fitted", True)
    session_path = _saved(widget, tmp_path / "settings-only")
    restored = _restored(session_path)
    assert restored.pixel_size_box.value() == pytest.approx(108.3)
    assert restored.search_box.value() == pytest.approx(1234.0)
