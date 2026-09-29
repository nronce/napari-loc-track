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


# How many localizations a row of a merged table stands for: 1 for one left as
# it was, N for a trajectory's N localizations merged into one.
MERGED_COUNT_COLUMN = "n_merged"


def merge_trajectories(table, particles, merge_ids, *, x_col, y_col, frame_col,
                       sigma_col=None, sum_columns=(), fallback_sigma=None):
    """Replace the localizations of each trajectory in `merge_ids` by one.

    An emitter that never moved, seen N times, has been measured N times at the
    same place: the precision-weighted mean of its positions is a single
    localization N-fold better determined than any of them, and drawing it once
    at that precision is what a static structure actually looks like, where
    drawing all N scatters it over their error.

    The merged precision is sqrt(1 / sum(1/sigma_i^2)), inflated by the square
    root of the trajectory's reduced chi-square whenever that exceeds one (the
    Birge ratio): if the positions scatter more than their stated precisions
    allow - motion too small to have been detected, or precisions that are
    optimistic - the merged point is drawn as wide as the scatter says, never
    narrower than the data supports.

    `particles` gives each row's trajectory (-1 for none). Photon counts and any
    other `sum_columns` add up; the frame is the first the molecule was seen in,
    with the last in `frame_last`; every other numeric column is averaged.
    Returns a new table: the untouched rows first, then one row per merged
    trajectory, with MERGED_COUNT_COLUMN and `particle` columns added.
    """
    import numpy as np

    particles = np.asarray(particles)
    ids = np.fromiter((int(p) for p in merge_ids), np.int64) if merge_ids else np.zeros(0, np.int64)
    inside = np.isin(particles, ids) & (particles >= 0)

    rest = table[~inside].copy()
    rest[MERGED_COUNT_COLUMN] = 1
    rest["particle"] = particles[~inside]
    rest["frame_last"] = rest[frame_col]
    if not inside.any():
        return rest

    sub = table[inside]
    codes, uniques = pd.factorize(particles[inside])
    count = np.bincount(codes)
    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    if sigma_col and sigma_col in sub.columns:
        sigma = sub[sigma_col].to_numpy(dtype=float)
    else:
        sigma = np.full(len(sub), np.nan)
    valid = np.isfinite(sigma) & (sigma > 0)
    stand_in = (fallback_sigma if fallback_sigma and fallback_sigma > 0
                else (float(np.median(sigma[valid])) if valid.any() else 1.0))
    sigma = np.where(valid, sigma, stand_in)

    weight = 1.0 / sigma ** 2
    total = np.bincount(codes, weight)
    x_bar = np.bincount(codes, weight * x) / total
    y_bar = np.bincount(codes, weight * y) / total
    scatter = np.bincount(codes, weight * ((x - x_bar[codes]) ** 2 + (y - y_bar[codes]) ** 2))
    dof = 2 * (count - 1)
    ratio = np.divide(scatter, dof, out=np.ones_like(scatter), where=dof > 0)
    merged_sigma = np.sqrt(1.0 / total) * np.sqrt(np.maximum(ratio, 1.0))

    grouped = sub.groupby(codes, sort=True)
    merged = grouped.mean(numeric_only=True)
    merged[x_col] = x_bar
    merged[y_col] = y_bar
    merged[frame_col] = grouped[frame_col].min().to_numpy()
    merged["frame_last"] = grouped[frame_col].max().to_numpy()
    for column in sum_columns:
        if column and column in sub.columns:
            merged[column] = grouped[column].sum().to_numpy()
    if sigma_col and sigma_col in sub.columns:
        merged[sigma_col] = merged_sigma
    merged[MERGED_COUNT_COLUMN] = count
    merged["particle"] = uniques
    return pd.concat([rest, merged.reset_index(drop=True)], ignore_index=True)


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
