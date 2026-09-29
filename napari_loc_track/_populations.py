"""Populations of trajectories, fitted to all of them at once.

The static test asks one question of each trajectory - could it be standing
still? - and a short trajectory almost always could, so it cannot decide. A
population model asks the question the other way round as well. It fits the
whole dataset as a mixture: an immobile population, whose steps show nothing but
localization error, and one or two mobile ones, each with its own diffusion
coefficient, in fitted proportions. The long trajectories pin those down, and
then every trajectory - a two-point one too - gets a probability of belonging
to each: a small step is evidence for immobility once the model knows how large
a mobile molecule's steps usually are.

The model of one trajectory is Brownian motion seen through localization error
and motion blur (Berglund, Phys. Rev. E 82, 011917 (2010)). Per axis, the
increments between consecutive localizations j and j+1, dt frames apart, are
jointly Gaussian with

    var(d_j)          = 2D (dt*tau - 2R*tau) + sigma_j^2 + sigma_{j+1}^2
    cov(d_j, d_{j+1}) = 2D R tau - sigma_{j+1}^2

where tau is the frame interval, sigma each localization's own precision, and
R the motion-blur coefficient: 1/6 for an exposure that lasts the whole frame,
less for a shorter one. The covariance is tridiagonal, so the exact likelihood
of a whole trajectory costs one forward pass, done here for every trajectory
at once and for a grid of D values, which makes the mixture fit a matter of
array arithmetic.

Everything is in nanometres and seconds inside, µm²/s at the edges. Only numpy
and scipy, so it can be tested without napari or Qt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# The D grid the likelihoods are tabulated on, µm²/s: from well below anything
# a localization error of a few nanometres could resolve, to faster than a
# 30 ms frame can follow. 13 % steps, refined by interpolation.
GRID_MIN_UM2 = 1e-5
GRID_MAX_UM2 = 20.0
GRID_POINTS = 121
# D values evaluated together; bounds the memory of the pass to
# (trajectories x chunk) arrays.
_GRID_CHUNK = 24


@dataclass
class Trajectories:
    """Trajectories laid out for the likelihood pass.

    Sorted by number of increments, longest first, so that the trajectories
    still going at increment j are always the first counts[j]: each step of the
    pass is a slice, never a mask. Increments are stored flat, each trajectory's
    starting at offsets[i].
    """

    pid: np.ndarray          # particle id per trajectory
    n_points: np.ndarray     # localizations per trajectory
    offsets: np.ndarray      # flat index of each trajectory's first increment
    counts: np.ndarray       # trajectories with more than j increments, per j
    zx: np.ndarray           # increments, nm, flat
    zy: np.ndarray
    gap: np.ndarray          # frames spanned by each increment
    s_diag: np.ndarray       # sigma_j^2 + sigma_{j+1}^2, nm^2
    s_next: np.ndarray       # sigma_{j+1}^2: the point shared with the next increment

    @property
    def n_increments(self):
        return self.n_points - 1

    def __len__(self):
        return len(self.pid)


def build_trajectories(particle, frame, x_nm, y_nm, sigma_nm):
    """Lay trajectories out for the likelihood pass. One-point ones are dropped.

    A localization with no usable precision is given the median of the rest,
    rather than dropped - dropping it would join its neighbours into a step it
    never took.
    """
    particle = np.asarray(particle)
    frame = np.asarray(frame, dtype=np.int64)
    x = np.asarray(x_nm, dtype=float)
    y = np.asarray(y_nm, dtype=float)
    sigma = np.asarray(sigma_nm, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    particle, frame, x, y, sigma = particle[keep], frame[keep], x[keep], y[keep], sigma[keep]
    usable = np.isfinite(sigma) & (sigma > 0)
    sigma = np.where(usable, sigma, np.median(sigma[usable]) if usable.any() else 1.0)

    order = np.lexsort((frame, particle))
    particle, frame, x, y, sigma = (a[order] for a in (particle, frame, x, y, sigma))
    starts = np.flatnonzero(np.r_[True, particle[1:] != particle[:-1]])
    lengths = np.diff(np.r_[starts, len(particle)])
    multi = lengths >= 2
    starts, lengths = starts[multi], lengths[multi]
    rank = np.argsort(-lengths, kind="stable")
    starts, lengths = starts[rank], lengths[rank]

    n_incr = lengths - 1
    offsets = np.r_[0, np.cumsum(n_incr)[:-1]].astype(np.int64)
    # every increment's first point, in trajectory order
    first = np.repeat(starts, n_incr) + (np.arange(n_incr.sum()) - np.repeat(offsets, n_incr))
    second = first + 1
    sigma2 = sigma ** 2
    max_incr = int(n_incr.max()) if len(n_incr) else 0
    counts = np.array([int((n_incr > j).sum()) for j in range(max_incr)], dtype=np.int64)
    return Trajectories(
        pid=particle[starts], n_points=lengths.astype(np.int64), offsets=offsets,
        counts=counts, zx=x[second] - x[first], zy=y[second] - y[first],
        gap=(frame[second] - frame[first]).astype(float),
        s_diag=sigma2[first] + sigma2[second], s_next=sigma2[second])


def blur_coefficient(exposure_fraction):
    """R for an exposure lasting this fraction of the frame interval (Berglund 2010)."""
    return float(np.clip(exposure_fraction, 0.0, 1.0)) / 6.0


def log_likelihoods(traj, d_um2, frame_interval_s, blur=1.0 / 6.0):
    """(trajectories x D values) log-likelihood of each trajectory at each D.

    The exact Gaussian likelihood of its increments, x and y independent,
    through the tridiagonal LDL^T recursion: for each increment j,
        l_j = b_{j-1} / e_{j-1},  e_j = a_j - l_j b_{j-1},  u_j = z_j - l_j u_{j-1}
    and the log-likelihood adds -(u_x^2 + u_y^2) / (2 e_j) - log e_j - log 2 pi.
    """
    d_nm2 = np.atleast_1d(np.asarray(d_um2, dtype=float)) * 1e6
    tau = float(frame_interval_s)
    out = np.empty((len(traj), len(d_nm2)))
    for start in range(0, len(d_nm2), _GRID_CHUNK):
        d = d_nm2[start:start + _GRID_CHUNK][None, :]
        two_d = 2.0 * d
        couple = two_d * blur * tau
        n = len(traj)
        idx = traj.offsets
        a = two_d * (traj.gap[idx] * tau - 2.0 * blur * tau)[:, None] + traj.s_diag[idx][:, None]
        e = a
        ux = np.broadcast_to(traj.zx[idx][:, None], a.shape)
        uy = np.broadcast_to(traj.zy[idx][:, None], a.shape)
        total = -(ux * ux + uy * uy) / (2.0 * e) - np.log(e)
        for j in range(1, len(traj.counts)):
            c = int(traj.counts[j])
            previous = traj.offsets[:c] + (j - 1)
            current = previous + 1
            b = couple - traj.s_next[previous][:, None]
            e_prev = e[:c]
            lower = b / e_prev
            a = two_d * (traj.gap[current] * tau - 2.0 * blur * tau)[:, None] \
                + traj.s_diag[current][:, None]
            e = a - lower * b
            ux = traj.zx[current][:, None] - lower * ux[:c]
            uy = traj.zy[current][:, None] - lower * uy[:c]
            total[:c] += -(ux * ux + uy * uy) / (2.0 * e) - np.log(e)
        out[:, start:start + d.shape[1]] = total
    return out - (traj.n_increments * math.log(2.0 * math.pi))[:, None]


def _logsumexp(values, axis):
    peak = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(peak, axis=axis) + np.log(np.sum(np.exp(values - peak), axis=axis))


def _at(table, log_grid, log_d):
    """Each trajectory's log-likelihood at log_d, interpolated along the grid."""
    g = int(np.clip(np.searchsorted(log_grid, log_d) - 1, 0, len(log_grid) - 2))
    w = (log_d - log_grid[g]) / (log_grid[g + 1] - log_grid[g])
    return table[:, g] * (1.0 - w) + table[:, g + 1] * w


def _best_log_d(score, log_grid, allowed):
    """The log D that maximizes a score along the grid, within `allowed`."""
    candidates = np.flatnonzero(allowed)
    g = candidates[int(np.argmax(score[candidates]))]
    if g - 1 in candidates and g + 1 in candidates:
        left, centre, right = score[g - 1], score[g], score[g + 1]
        curvature = left - 2.0 * centre + right
        if curvature < 0:
            step = log_grid[g + 1] - log_grid[g]
            return float(log_grid[g] + 0.5 * step * (left - right) / curvature)
    return float(log_grid[g])


def fit_mixture(table, d_grid, n_mobile, d_immobile_max, max_iter=500, tol=1e-9):
    """EM on tabulated log-likelihoods: an immobile population with D at most
    `d_immobile_max`, and `n_mobile` populations above it, kept in order of D.

    Returns {"D": (K,), "fractions": (K,), "responsibilities": (n, K),
    "loglik": float, "n_iter": int}; D in the units of d_grid.
    """
    log_grid = np.log(np.asarray(d_grid, dtype=float))
    boundary = math.log(float(d_immobile_max))
    immobile_range = log_grid <= boundary
    mobile_range = log_grid >= boundary
    if not immobile_range.any() or not mobile_range.any():
        raise ValueError("the immobile limit lies outside the D grid")
    n, K = table.shape[0], 1 + int(n_mobile)

    # Start from each trajectory's own best D, split at the immobile limit.
    best = log_grid[np.argmax(table, axis=1)]
    slow = best[best <= boundary]
    fast = best[best > boundary]
    log_d = [float(np.median(slow)) if slow.size else boundary - 2.0]
    quantiles = np.linspace(0, 100, K + 1)[1:-1] if K > 2 else [50.0]
    for q in quantiles:
        log_d.append(float(np.percentile(fast, q)) if fast.size else boundary + 2.0)
    log_d = np.clip(np.array(log_d), log_grid[0], log_grid[-1])
    log_d[0] = min(log_d[0], boundary)
    log_d[1:] = np.maximum(log_d[1:], boundary)
    fractions = np.full(K, 1.0 / K)

    previous = -np.inf
    for iteration in range(1, max_iter + 1):
        at = np.column_stack([_at(table, log_grid, v) for v in log_d])
        joint = at + np.log(np.maximum(fractions, 1e-300))
        total = _logsumexp(joint, axis=1)
        loglik = float(total.sum())
        resp = np.exp(joint - total[:, None])
        fractions = resp.mean(axis=0)
        for k in range(K):
            score = resp[:, k] @ table
            log_d[k] = _best_log_d(score, log_grid, immobile_range if k == 0 else mobile_range)
        order = np.argsort(log_d[1:]) + 1
        log_d[1:], fractions[1:] = log_d[order], fractions[order]
        if abs(loglik - previous) <= tol * max(abs(loglik), 1.0):
            break
        previous = loglik
    return {"D": np.exp(log_d), "fractions": fractions, "responsibilities": resp,
            "loglik": loglik, "n_iter": iteration}


def fit_populations_iter(traj, frame_interval_s, blur=1.0 / 6.0, d_immobile_max=0.01,
                         n_mobile="auto", cancel=None):
    """Fit the populations; yields progress, returns the result (None if cancelled).

    n_mobile: 1, 2, or "auto" - both fitted, and the one with the lower BIC
    (Bayesian information criterion: -2 log L + parameters x log N) kept, so a
    second mobile population has to earn its place.

    The result: pid; posterior (trajectories x populations), each row summing to
    one; D (µm²/s) and fractions per population, fractions both of trajectories
    and of localizations; labels; bic per number of mobile populations; the
    log-likelihood; and the tabulated grid, for plotting.
    """
    if len(traj) == 0:
        raise ValueError("no trajectories of two or more points to fit")
    d_grid = np.geomspace(GRID_MIN_UM2, GRID_MAX_UM2, GRID_POINTS)
    columns = []
    for start in range(0, len(d_grid), _GRID_CHUNK):
        if cancel is not None and cancel.is_set():
            return None
        columns.append(log_likelihoods(traj, d_grid[start:start + _GRID_CHUNK],
                                       frame_interval_s, blur))
        yield min(0.8, 0.8 * (start + _GRID_CHUNK) / len(d_grid))
    table = np.hstack(columns)

    choices = (1, 2) if n_mobile == "auto" else (int(n_mobile),)
    fits = {}
    for m in choices:
        if cancel is not None and cancel.is_set():
            return None
        fit = fit_mixture(table, d_grid, m, d_immobile_max)
        # exact likelihoods at the fitted D, not interpolated ones
        exact = log_likelihoods(traj, fit["D"], frame_interval_s, blur)
        joint = exact + np.log(np.maximum(fit["fractions"], 1e-300))
        total = _logsumexp(joint, axis=1)
        fit["posterior"] = np.exp(joint - total[:, None])
        fit["loglik"] = float(total.sum())
        n_params = 2 * (1 + m) - 1
        fit["bic"] = -2.0 * fit["loglik"] + n_params * math.log(len(traj))
        fits[m] = fit
        yield 0.8 + 0.2 * len(fits) / len(choices)
    chosen = min(fits, key=lambda m: fits[m]["bic"])
    fit = fits[chosen]
    posterior = fit["posterior"]
    weights = traj.n_points[:, None] * posterior
    labels = ["immobile"] + (["mobile"] if chosen == 1 else ["slow", "fast"])
    return {
        "pid": traj.pid, "posterior": posterior, "D": fit["D"],
        "fractions": posterior.mean(axis=0),
        "localization_fractions": weights.sum(axis=0) / weights.sum(),
        "labels": labels, "n_mobile": chosen,
        "bic": {m: fits[m]["bic"] for m in fits}, "loglik": fit["loglik"],
        "n_trajectories": len(traj), "n_points": traj.n_points.copy(),
        "n_iter": fit["n_iter"],
        "frame_interval_s": float(frame_interval_s), "blur": float(blur),
        "d_immobile_max": float(d_immobile_max),
    }


def fit_populations(*args, **kwargs):
    """fit_populations_iter, run to the end."""
    steps = fit_populations_iter(*args, **kwargs)
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value


def step_length_densities(traj, result, bins):
    """Observed one-frame step lengths against what each population predicts.

    Returns (histogram density per bin, predicted density per population per
    bin). A population's prediction is exact under the model: every one-frame
    step, weighted by its trajectory's probability of belonging to that
    population, drawn from its own Rayleigh distribution - whose width is set
    by that population's D and the two localizations' own precisions.
    """
    single = traj.gap == 1
    lengths = np.hypot(traj.zx, traj.zy)[single]
    owner = np.repeat(np.arange(len(traj)), traj.n_increments)[single]
    observed, edges = np.histogram(lengths, bins=bins, density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    tau, blur = result["frame_interval_s"], result["blur"]
    predicted = []
    for k, d in enumerate(np.asarray(result["D"]) * 1e6):
        variance = 2.0 * d * (tau - 2.0 * blur * tau) + traj.s_diag[single]
        weight = result["posterior"][owner, k]
        density = np.zeros(len(centres))
        for chunk in range(0, len(lengths), 20000):
            v = variance[chunk:chunk + 20000][:, None]
            w = weight[chunk:chunk + 20000][:, None]
            r = centres[None, :]
            density += (w * r / v * np.exp(-r * r / (2.0 * v))).sum(axis=0)
        predicted.append(density / max(len(lengths), 1))
    return centres, observed, np.array(predicted)
