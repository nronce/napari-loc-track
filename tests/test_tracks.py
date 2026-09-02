"""Trajectory length filtering and batching.

The D computation runs tp.imsd batch by batch so it can be interrupted and can
report progress. That is only equivalent to one big call if every trajectory
lands in exactly one batch, whole - which is what these tests pin down.
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PKG_DIR = Path(__file__).resolve().parents[1] / "napari_loc_track"
if "napari_loc_track" not in sys.modules:
    _pkg = types.ModuleType("napari_loc_track")
    _pkg.__path__ = [str(_PKG_DIR)]
    sys.modules["napari_loc_track"] = _pkg
_SPEC = importlib.util.spec_from_file_location("napari_loc_track._tracks", _PKG_DIR / "_tracks.py")
tracks_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["napari_loc_track._tracks"] = tracks_mod
_SPEC.loader.exec_module(tracks_mod)

filter_tracks_by_length = tracks_mod.filter_tracks_by_length
iter_particle_batches = tracks_mod.iter_particle_batches
max_linkable_diffusion = tracks_mod.max_linkable_diffusion
rms_step = tracks_mod.rms_step


def _tracks(lengths):
    """One trajectory per entry of `lengths`, with that many points."""
    rows = []
    for pid, n in enumerate(lengths):
        for frame in range(n):
            rows.append({"particle": pid, "frame": frame, "x": float(frame), "y": float(pid)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "min_length, expected",
    [
        (1, [0, 1, 2, 3]),   # no-op
        (2, [0, 1, 2, 3]),
        (3, [1, 2, 3]),
        (5, [2, 3]),
        (10, [3]),
        (11, []),
    ],
)
def test_filter_keeps_trajectories_with_at_least_min_length(min_length, expected):
    """Threshold is inclusive, matching trackpy's filter_stubs."""
    df = _tracks([2, 3, 5, 10])
    kept = filter_tracks_by_length(df, min_length)
    assert sorted(kept["particle"].unique().tolist()) == expected


def test_filter_preserves_rows_of_kept_trajectories():
    df = _tracks([2, 7])
    kept = filter_tracks_by_length(df, 3)
    pd.testing.assert_frame_equal(kept, df[df["particle"] == 1])


def test_filter_handles_empty_and_none():
    assert filter_tracks_by_length(None, 5).empty
    assert filter_tracks_by_length(pd.DataFrame(columns=["particle"]), 5).empty


@pytest.mark.parametrize("batch_size", [1, 2, 3, 7, 50])
def test_batches_cover_every_trajectory_exactly_once(batch_size):
    df = _tracks([2, 3, 4, 5, 6, 7, 8])
    seen = []
    rows = 0
    for subset, done, total in iter_particle_batches(df, batch_size):
        pids = subset["particle"].unique().tolist()
        seen.extend(pids)
        rows += len(subset)
        assert total == 7
        assert 0 < done <= total
        # Trajectories must arrive whole, or per-trajectory MSD would be wrong.
        for pid in pids:
            assert len(subset[subset["particle"] == pid]) == len(df[df["particle"] == pid])
    assert sorted(seen) == list(range(7))
    assert len(seen) == len(set(seen))
    assert rows == len(df)


def test_batch_progress_is_monotonic_and_reaches_total():
    df = _tracks([3] * 10)
    progress = [done for _subset, done, _total in iter_particle_batches(df, 3)]
    assert progress == sorted(progress)
    assert progress[-1] == 10


def test_batching_empty_input_yields_nothing():
    assert list(iter_particle_batches(pd.DataFrame(columns=["particle"]), 5)) == []
    assert list(iter_particle_batches(None, 5)) == []


# --- linking cutoff: D_max = R^2 / (4 t ln(1/eps)) ----------------------------


def test_cutoff_matches_closed_form():
    import math

    search_nm, dt = 250.0, 0.01
    expected = (search_nm ** 2 * 1e-6) / (4 * dt * math.log(100))
    assert max_linkable_diffusion(search_nm, dt, error_rate=0.01) == pytest.approx(expected)


def test_cutoff_is_the_rate_a_simulation_sees():
    """Steps drawn at exactly D_max must exceed the search range ~1% of the time."""
    search_nm, dt, eps = 250.0, 0.01, 0.01
    d_max = max_linkable_diffusion(search_nm, dt, error_rate=eps)

    rng = np.random.default_rng(0)
    n = 400_000
    sigma_nm = np.sqrt(2 * d_max * dt * 1e6)  # per-axis sd, um^2 -> nm^2
    steps = np.hypot(rng.normal(0, sigma_nm, n), rng.normal(0, sigma_nm, n))
    observed = float((steps > search_nm).mean())
    assert observed == pytest.approx(eps, abs=0.002), f"{observed:.4f} vs {eps}"


def test_search_range_is_about_2_15_times_the_rms_step():
    """The 1% rule of thumb: R / sqrt(<r^2>) = sqrt(ln 100) = 2.146."""
    search_nm, dt = 300.0, 0.02
    d_max = max_linkable_diffusion(search_nm, dt, error_rate=0.01)
    assert search_nm / rms_step(d_max, dt) == pytest.approx(2.1460, rel=1e-3)


def test_cutoff_scales_as_expected():
    base = max_linkable_diffusion(250.0, 0.01)
    # Doubling the search range quadruples the linkable D.
    assert max_linkable_diffusion(500.0, 0.01) == pytest.approx(4 * base)
    # Doubling the frame interval halves it.
    assert max_linkable_diffusion(250.0, 0.02) == pytest.approx(base / 2)
    # Memory extends the lag to (memory + 1) frames.
    assert max_linkable_diffusion(250.0, 0.01, memory=1) == pytest.approx(base / 2)
    assert max_linkable_diffusion(250.0, 0.01, memory=3) == pytest.approx(base / 4)
    # A looser error budget allows a larger D.
    assert max_linkable_diffusion(250.0, 0.01, error_rate=0.05) > base


@pytest.mark.parametrize(
    "search_nm, dt, memory, error_rate",
    [(0.0, 0.01, 0, 0.01), (-5.0, 0.01, 0, 0.01), (250.0, 0.0, 0, 0.01),
     (250.0, 0.01, 0, 0.0), (250.0, 0.01, 0, 1.0)],
)
def test_cutoff_rejects_degenerate_input(search_nm, dt, memory, error_rate):
    assert np.isnan(max_linkable_diffusion(search_nm, dt, error_rate=error_rate, memory=memory))


def test_rms_step_round_trips():
    dt = 0.015
    d = 0.35
    assert rms_step(d, dt) == pytest.approx(np.sqrt(4 * d * dt * 1e6))
    assert rms_step(d, dt, memory=2) == pytest.approx(rms_step(d, 3 * dt))
