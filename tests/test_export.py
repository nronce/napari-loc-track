"""Export writes the same files as before, off the GUI thread, and can be stopped.

The chunked writer exists only so a multi-second export can report progress and
be cancelled - the bytes it produces must be identical to a plain to_csv.
"""
import json
import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_widget_interaction import ensure_qapp, make_widget  # noqa: E402


def _locs(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "frame": rng.integers(0, 200, n),
        "x [nm]": rng.uniform(0, 50000, n),
        "y [nm]": rng.uniform(0, 50000, n),
        "sigma [nm]": rng.uniform(50, 500, n),
        "intensity [photon]": rng.uniform(10, 10000, n),
    })


def _tracks(n_tracks=40, n_points=10, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "particle": np.repeat(np.arange(n_tracks), n_points),
        "frame": np.tile(np.arange(n_points), n_tracks),
        "x": rng.uniform(0, 50000, n_tracks * n_points),
        "y": rng.uniform(0, 50000, n_tracks * n_points),
    })


def _run(worker):
    return worker.work()


@pytest.mark.parametrize("chunk", [100_000, 500, 1])
def test_chunked_write_is_byte_identical_to_to_csv(tmp_path, monkeypatch, chunk):
    """Whatever the chunk size, the file must match df.to_csv exactly."""
    monkeypatch.setattr(widget_mod, "EXPORT_CHUNK_ROWS", chunk)
    df = _locs(n=1200)
    folder = tmp_path / "analysis"
    folder.mkdir()

    _run(widget_mod._export_worker(folder, [("localizations_filtered.csv", df)], {}, None))

    reference = tmp_path / "reference.csv"
    df.to_csv(reference, index=False)
    written = (folder / "data" / "localizations_filtered.csv").read_bytes()
    assert written == reference.read_bytes()


def test_written_tables_round_trip(tmp_path):
    df = _locs()
    tracks = _tracks()
    folder = tmp_path / "analysis"
    folder.mkdir()
    tables = [("localizations_filtered.csv", df), ("trajectories.csv", tracks)]

    result = _run(widget_mod._export_worker(folder, tables, {"pixel_size_nm_per_px": 100.0}, None))
    assert result == folder

    back = pd.read_csv(folder / "data" / "localizations_filtered.csv")
    pd.testing.assert_frame_equal(back, df, check_dtype=False)
    back_tracks = pd.read_csv(folder / "data" / "trajectories.csv")
    pd.testing.assert_frame_equal(back_tracks, tracks, check_dtype=False)
    assert json.loads((folder / "metadata.json").read_text(encoding="utf-8"))["pixel_size_nm_per_px"] == 100.0


def test_empty_table_still_writes_a_header(tmp_path):
    folder = tmp_path / "analysis"
    folder.mkdir()
    empty = _locs(n=0)
    _run(widget_mod._export_worker(folder, [("localizations_filtered.csv", empty)], {}, None))
    text = (folder / "data" / "localizations_filtered.csv").read_text(encoding="utf-8")
    assert text.strip().split("\n")[0] == ",".join(empty.columns)


def test_export_can_be_cancelled_without_leaving_a_partial_table(tmp_path, monkeypatch):
    monkeypatch.setattr(widget_mod, "EXPORT_CHUNK_ROWS", 100)
    folder = tmp_path / "analysis"
    folder.mkdir()
    cancel = threading.Event()
    cancel.set()

    result = _run(widget_mod._export_worker(
        folder, [("localizations_filtered.csv", _locs())], {}, cancel))

    assert result is widget_mod.CANCELLED
    assert not (folder / "data" / "localizations_filtered.csv").exists()
    assert not (folder / "metadata.json").exists()


def test_progress_is_monotonic_and_reaches_100(tmp_path, monkeypatch):
    monkeypatch.setattr(widget_mod, "EXPORT_CHUNK_ROWS", 250)
    folder = tmp_path / "analysis"
    folder.mkdir()
    worker = widget_mod._export_worker(
        folder, [("localizations_filtered.csv", _locs()), ("trajectories.csv", _tracks())], {}, None)

    seen = []
    worker.yielded.connect(seen.append)
    worker.work()

    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)


# --- through the widget -------------------------------------------------------


def _pump_until(predicate, timeout_s=60.0):
    """Run the Qt event loop until `predicate` holds, so worker signals arrive."""
    import time

    app = ensure_qapp()
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_export_analysis_hands_off_to_a_worker(tmp_path):
    """The GUI call must return promptly, with the writing left to the worker."""
    widget = make_widget()
    widget._ingest_localization_dataframe(_locs(), "loaded", True)
    widget.tracks = _tracks()
    widget.csv_edit.setText(str(tmp_path / "source.csv"))

    widget.export_analysis()
    assert widget._export_worker_ref is not None, "export ran inline instead of in a worker"
    assert not widget.export_button.isEnabled(), "buttons stay disabled while it runs"

    assert _pump_until(lambda: widget._export_worker_ref is None), "export never finished"

    # One dated folder per run, so a second export never overwrites the first.
    runs = [d for d in (tmp_path / "analysis").iterdir() if d.is_dir()]
    assert len(runs) == 1 and runs[0].name.endswith("_export")
    folder = runs[0]
    assert (folder / "data" / "localizations_filtered.csv").exists()
    assert (folder / "data" / "trajectories.csv").exists()
    assert (folder / "data" / "track_metrics.csv").exists()
    assert (folder / "metadata.json").exists()
    assert (folder / "plots").is_dir()
    assert widget.export_button.isEnabled()
    assert not widget.export_progress.isVisible()


def test_track_metrics_frame_has_one_row_per_trajectory():
    widget = make_widget()
    widget.tracks = _tracks(n_tracks=7)
    widget._track_diffusion_cache = {0: 0.5, 3: 1.5}
    widget._track_distance_cache = {0: 2.0}
    widget._track_duration_cache = {}
    frame = widget._track_metrics_frame()
    assert len(frame) == 7
    assert list(frame.columns) == [
        "particle", "D_um2_per_s", "distance_um",
        "net_displacement_um", "straightness", "duration_s",
        "motion_ratio", "p_static", "d_detectable_um2_per_s", "sigma_from_msd_nm",
    ]
    assert frame.loc[frame["particle"] == 0, "D_um2_per_s"].iloc[0] == pytest.approx(0.5)
    assert pd.isna(frame.loc[frame["particle"] == 1, "D_um2_per_s"].iloc[0])
