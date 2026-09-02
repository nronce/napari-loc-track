"""Behaviour of the two paths that were rewritten for speed.

`apply_numeric_filters` builds one combined mask instead of re-copying the table
per bound, and `_detect_worker` runs frames on a thread pool. Both are supposed
to be *only* faster - identical results, same ordering, same cancellation.
"""
import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

apply_numeric_filters = widget_mod.apply_numeric_filters
reference = widget_mod._apply_numeric_filters_reference


def _table(n_rows=5000, seed=0, with_nans=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "frame": rng.integers(0, 500, n_rows),
        "x [nm]": rng.uniform(0, 50000, n_rows),
        "y [nm]": rng.uniform(0, 50000, n_rows),
        "sigma [nm]": rng.uniform(50, 500, n_rows),
        "intensity [photon]": rng.uniform(10, 10000, n_rows),
    })
    if with_nans:
        df.loc[rng.choice(n_rows, n_rows // 20, replace=False), "sigma [nm]"] = np.nan
    return df


# --- filtering ----------------------------------------------------------------


@pytest.mark.parametrize("with_nans", [False, True])
def test_matches_the_row_by_row_reference(with_nans):
    df = _table(with_nans=with_nans)
    bounds = {
        "frame": (10, 400),
        "sigma [nm]": (100.0, 400.0),
        "intensity [photon]": (500.0, 9000.0),
        "x [nm]": (1000.0, 49000.0),
    }
    fast = apply_numeric_filters(df, bounds)
    slow = reference(df, bounds)
    pd.testing.assert_frame_equal(fast, slow)


def test_nan_rows_are_dropped_like_before():
    df = _table(n_rows=100, with_nans=True)
    out = apply_numeric_filters(df, {"sigma [nm]": (0.0, 1e9)})
    assert out["sigma [nm]"].notna().all()
    assert len(out) < len(df)


@pytest.mark.parametrize("bounds", [
    {},                                          # nothing to do
    {"no such column": (0.0, 1.0)},              # unknown column ignored
    {"sigma [nm]": (None, 300.0)},               # one-sided
    {"sigma [nm]": (300.0, None)},
    {"sigma [nm]": (None, None)},
])
def test_edge_cases_match_the_reference(bounds):
    df = _table(n_rows=500)
    pd.testing.assert_frame_equal(apply_numeric_filters(df, bounds), reference(df, bounds))


def test_index_and_columns_are_preserved():
    df = _table(n_rows=200)
    df.index = np.arange(1000, 1200)
    out = apply_numeric_filters(df, {"frame": (100, 400)})
    assert list(out.columns) == list(df.columns)
    assert out.index.isin(df.index).all()
    assert (out["frame"] >= 100).all() and (out["frame"] <= 400).all()


def test_result_is_a_copy_not_a_view():
    """Callers mutate the filtered table; it must not write back into the source."""
    df = _table(n_rows=100)
    out = apply_numeric_filters(df, {})
    out.loc[out.index[0], "frame"] = -12345
    assert df.loc[df.index[0], "frame"] != -12345


def test_empty_input():
    empty = _table(n_rows=0)
    assert len(apply_numeric_filters(empty, {"frame": (0, 10)})) == 0
    assert apply_numeric_filters(None, {"frame": (0, 10)}) is None


# --- threaded detection -------------------------------------------------------


def _stack(n_frames=6, size=96, n_spots=8, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size), dtype=np.float32)
    frames = []
    for f in range(n_frames):
        frame = rng.poisson(100.0, size=(size, size)).astype(np.float32)
        # A different number of spots per frame, so a mix-up would show.
        for k in range(n_spots + f):
            cy, cx = rng.uniform(12, size - 12), rng.uniform(12, size - 12)
            frame += (900.0 * np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * 1.3 ** 2))).astype(np.float32)
        frames.append(frame)
    return np.stack(frames)


def _run(worker):
    """Drive a thread_worker generator to completion in this thread."""
    return worker.work()


def test_threaded_detection_matches_serial_frame_for_frame():
    stack = _stack()
    box, min_ng = 7, 500.0
    expected = [widget_mod.identify_in_frame(stack[i], min_ng, box) for i in range(stack.shape[0])]

    candidates, counts = _run(widget_mod._detect_worker(stack, box, min_ng, None))

    assert len(candidates) == stack.shape[0]
    for i, (want, got) in enumerate(zip(expected, candidates)):
        np.testing.assert_array_equal(got[0], want[0], err_msg=f"frame {i} y")
        np.testing.assert_array_equal(got[1], want[1], err_msg=f"frame {i} x")
        np.testing.assert_allclose(got[2], want[2], rtol=1e-6, err_msg=f"frame {i} ng")
        assert counts[i] == len(want[0])


def test_results_stay_matched_to_their_frame():
    """Frames finish out of order; each result must land on its own index."""
    stack = _stack(n_frames=8)
    candidates, counts = _run(widget_mod._detect_worker(stack, 7, 500.0, None))
    for i in range(stack.shape[0]):
        y, x, _ng = widget_mod.identify_in_frame(stack[i], 500.0, 7)
        assert counts[i] == len(y)
        np.testing.assert_array_equal(candidates[i][0], y)


def test_detection_can_still_be_cancelled():
    stack = _stack(n_frames=8)
    cancel = threading.Event()
    cancel.set()
    assert _run(widget_mod._detect_worker(stack, 7, 500.0, cancel)) is widget_mod.CANCELLED


def test_single_frame_stack():
    stack = _stack(n_frames=1)
    candidates, counts = _run(widget_mod._detect_worker(stack, 7, 500.0, None))
    assert len(candidates) == 1 and counts[0] == len(candidates[0][0])
