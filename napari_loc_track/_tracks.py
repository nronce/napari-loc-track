"""Trajectory helpers that are pure pandas.

Kept out of `widget.py` so they can be exercised without the napari/Qt stack -
these are the parts of the trajectory pipeline where an off-by-one silently
changes a result rather than raising.
"""
from __future__ import annotations

import math

import pandas as pd

# Fraction of true steps allowed to exceed the search range, i.e. the fraction of
# links the search range is expected to miss.
DEFAULT_LINKING_ERROR_RATE = 0.01


def max_linkable_diffusion(search_range_nm, frame_interval_s, error_rate=DEFAULT_LINKING_ERROR_RATE,
                           memory=0):
    """Largest D (µm²/s) a search range can follow, missing at most `error_rate` of steps.

    For 2D Brownian motion the per-axis displacement over a lag t is Gaussian
    with variance 2Dt, so the step length r = sqrt(dx² + dy²) is Rayleigh
    distributed with scale sigma = sqrt(2Dt) and mean square <r²> = 4Dt. The
    fraction of steps longer than a search range R is the Rayleigh survival
    function

        P(r > R) = exp(-R² / (2 sigma²)) = exp(-R² / (4Dt)).

    Setting that equal to the tolerated error rate eps and solving for D:

        D_max = R² / (4 t ln(1/eps)).

    At eps = 1% this is R² / (18.42 t), i.e. the search range has to be about
    2.15x the RMS step length. `memory` frames of gap-closing let a particle
    disappear and be picked up later, so the lag that must be covered is
    (memory + 1) * frame_interval and the cutoff drops proportionally.

    This is the single-particle criterion only. It says nothing about *wrong*
    links, which come from density: if another localization is within the search
    range, trackpy can still pick it. A high cutoff is necessary, not sufficient.
    """
    lag_s = float(frame_interval_s) * (int(memory) + 1)
    if search_range_nm <= 0 or lag_s <= 0 or not (0.0 < error_rate < 1.0):
        return float("nan")
    # nm² -> µm² is 1e-6.
    return (float(search_range_nm) ** 2 * 1e-6) / (4.0 * lag_s * math.log(1.0 / error_rate))


def rms_step(d_um2_s, frame_interval_s, memory=0):
    """RMS step length in nm for a given D (µm²/s): sqrt(<r²>) = sqrt(4 D t)."""
    lag_s = float(frame_interval_s) * (int(memory) + 1)
    if d_um2_s < 0 or lag_s <= 0:
        return float("nan")
    return math.sqrt(4.0 * float(d_um2_s) * lag_s * 1e6)  # µm² -> nm²


def filter_tracks_by_length(tracks, min_length):
    """Keep only trajectories with at least `min_length` localizations.

    Length is counted in points per trajectory - the same notion trackpy's
    filter_stubs uses for the linking filter - so the two thresholds are
    directly comparable and a value at or below the linking one is a no-op.
    """
    if tracks is None:
        return pd.DataFrame()
    if tracks.empty or min_length <= 1:
        return tracks
    counts = tracks["particle"].value_counts()
    return tracks[tracks["particle"].isin(counts.index[counts >= min_length])]


def iter_particle_batches(tracks, batch_size):
    """Split a trajectory table into batches of whole trajectories.

    Yields (subset, n_done, n_total) where n_done counts trajectories, not rows.
    Every trajectory appears in exactly one batch and arrives whole, which is
    what makes it valid to run a per-trajectory computation (MSD) batch by batch
    instead of in one uninterruptible call. Each subset carries a fresh index so
    it stands on its own, exactly like the table a single call would have seen.
    """
    if tracks is None or tracks.empty:
        return
    groups = dict(tuple(tracks.groupby("particle")))
    pids = list(groups)
    total = len(pids)
    step = max(1, int(batch_size))
    for start in range(0, total, step):
        batch = pids[start : start + step]
        subset = pd.concat([groups[pid] for pid in batch], ignore_index=True)
        yield subset, min(start + step, total), total
