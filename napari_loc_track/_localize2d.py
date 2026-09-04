"""2D single-molecule localization: spot detection + sub-pixel Gaussian fitting.

Adapted from the legacy `forLocalization2D` prototype (itself inspired by the
Picasso package, Jungmann lab), trimmed to what the napari plugin's
"Localize (2D)" tab needs: candidate detection (local maxima + net gradient)
and per-candidate Gaussian fitting (least-squares, Poisson-MLE, or GPU via
Gpufit when available). Rendering is intentionally left out - the plugin
visualizes/filters localizations with its own napari layers once they're loaded
into a dataframe.

Fitting is batched per frame: `localize_frame` gathers every candidate window
into one (N, box, box) array and hands it to a jitted (and, above a threshold,
prange-parallel) Gauss-Newton kernel, or to a single Gpufit call. The pure-NumPy
fitters below stay as the reference implementation and the fallback for installs
without numba.
"""
from __future__ import annotations

import numpy as np

try:
    import numba
    _NUMBA_AVAILABLE = True
except Exception:
    numba = None
    _NUMBA_AVAILABLE = False

try:
    import pygpufit.gpufit as _GPUFIT                                                                   
    _GPUFIT_AVAILABLE = True
except Exception:
    _GPUFIT = None
    _GPUFIT_AVAILABLE = False

_EPS = 1e-9
_FIT_CACHE = {}

# Gpufit's per-fit state codes: 0 converged, 1 hit the iteration limit, 2 singular
# Hessian, 3 negative curvature (MLE), 4 GPU not ready. pygpufit exports no enum
# for these (its `Status` is the return code of the call itself, not the per-fit
# state), so the value is spelled out here rather than looked up on the module.
_GPUFIT_CONVERGED = 0


def is_gpufit_available() -> bool:
    return bool(_GPUFIT_AVAILABLE)


def is_numba_available() -> bool:
    return bool(_NUMBA_AVAILABLE)


# ----------------------------------------------------------------------
# Candidate detection: local maxima + net-gradient thresholding
# ----------------------------------------------------------------------
def _local_maxima_np(frame, box):
    from scipy.ndimage import maximum_filter

    box_half = box // 2
    h, w = frame.shape
    if h <= box or w <= box:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    local_max = maximum_filter(frame, size=box, mode="nearest")
    is_max = frame == local_max
    mask = np.zeros_like(is_max)
    mask[box_half : h - box_half - 1, box_half : w - box_half - 1] = True
    is_max &= mask
    y, x = np.where(is_max)
    return y.astype(np.int32), x.astype(np.int32)


def _net_gradient_np(frame, y, x, box):
    box_half = box // 2
    yy, xx = np.mgrid[-box_half : box_half + 1, -box_half : box_half + 1].astype(np.float64)
    unorm = np.sqrt(xx**2 + yy**2)
    unorm[unorm == 0] = 1.0
    # Unit vectors point *inwards*, towards the box centre, so that a bright spot
    # scores positive - same convention as _net_gradient_nb (and Picasso), which
    # builds them as (box_half - index). Without the sign the fallback path
    # returned the negated gradient and nothing passed a positive threshold.
    ux = -xx / unorm
    uy = -yy / unorm

    frame_f = frame.astype(np.float64)
    ng = np.zeros(len(x), dtype=np.float64)
    for idx in range(len(x)):
        yi, xi = int(y[idx]), int(x[idx])
        total = 0.0
        for ky in range(box):
            dy = ky - box_half
            yk = yi + dy
            for kx in range(box):
                dx = kx - box_half
                if dy == 0 and dx == 0:
                    continue
                xk = xi + dx
                gy = frame_f[yk + 1, xk] - frame_f[yk - 1, xk]
                gx = frame_f[yk, xk + 1] - frame_f[yk, xk - 1]
                total += gy * uy[ky, kx] + gx * ux[ky, kx]
        ng[idx] = total
    return ng


def _identify_in_frame_np(frame, minimum_ng, box):
    y, x = _local_maxima_np(frame, box)
    if len(x) == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )
    ng = _net_gradient_np(frame, y, x, box)
    keep = ng > minimum_ng
    return y[keep].astype(np.int32), x[keep].astype(np.int32), ng[keep].astype(np.float32)


if _NUMBA_AVAILABLE:

    @numba.njit(nogil=True, cache=True)
    def _local_maxima_nb(frame, box):
        """Centre-of-box local maxima, separable.

        The previous version took a (box, box) slice and np.argmax per pixel:
        O(H*W*box^2). This is a separable max filter - one horizontal pass, then
        one-sided vertical maxima per output row - which is O(H*W*box) with
        vectorisable inner loops and needs one extra frame-sized buffer.

        The tie-breaking of np.argmax is reproduced exactly: argmax returns the
        *first* maximum in row-major order, so the centre only counted as a
        maximum if it was strictly greater than everything scanned before it
        (rows above, then the pixels to its left) and >= everything after.
        """
        y_size, x_size = frame.shape
        maxima_map = np.zeros(frame.shape, np.uint8)
        box_half = int(box / 2)
        box_half_1 = box_half + 1
        if y_size <= box or x_size <= box:
            y, x = np.where(maxima_map)
            return y.astype(np.int32), x.astype(np.int32)

        x_lo = box_half
        x_hi = x_size - box_half  # rowmax valid on [x_lo, x_hi)

        # rowmax[y, x] = max(frame[y, x - box_half : x + box_half + 1])
        rowmax = np.zeros(frame.shape, dtype=frame.dtype)
        for y in range(y_size):
            for x in range(x_lo, x_hi):
                rowmax[y, x] = frame[y, x]
            for k in range(1, box_half + 1):
                for x in range(x_lo, x_hi):
                    rowmax[y, x] = max(rowmax[y, x], max(frame[y, x - k], frame[y, x + k]))

        left = np.zeros(x_size, dtype=frame.dtype)
        right = np.zeros(x_size, dtype=frame.dtype)
        up = np.zeros(x_size, dtype=frame.dtype)
        down = np.zeros(x_size, dtype=frame.dtype)
        for y in range(box_half, y_size - box_half_1):
            for x in range(x_lo, x_hi):
                left[x] = frame[y, x - 1]
                right[x] = frame[y, x + 1]
                up[x] = rowmax[y - 1, x]
                down[x] = rowmax[y + 1, x]
            for k in range(2, box_half + 1):
                for x in range(x_lo, x_hi):
                    left[x] = max(left[x], frame[y, x - k])
                    right[x] = max(right[x], frame[y, x + k])
                    up[x] = max(up[x], rowmax[y - k, x])
                    down[x] = max(down[x], rowmax[y + k, x])
            for x in range(box_half, x_size - box_half_1):
                centre = frame[y, x]
                if centre > up[x] and centre > left[x] and centre >= down[x] and centre >= right[x]:
                    maxima_map[y, x] = 1

        y, x = np.where(maxima_map)
        return y.astype(np.int32), x.astype(np.int32)

    @numba.njit(nogil=True, cache=True)
    def _net_gradient_nb(frame, y, x, box):
        box_half = int(box / 2)
        ux = np.zeros((box, box), dtype=np.float32)
        uy = np.zeros((box, box), dtype=np.float32)
        for i in range(box):
            val = box_half - i
            ux[:, i] = uy[i, :] = val
        unorm = np.sqrt(ux**2 + uy**2)
        for i in range(box):
            for j in range(box):
                if unorm[i, j] == 0:
                    unorm[i, j] = 1.0
        ux /= unorm
        uy /= unorm

        ng = np.zeros(len(x), dtype=np.float32)
        for i in range(len(x)):
            yi = y[i]
            xi = x[i]
            for k_index, k in enumerate(range(yi - box_half, yi + box_half + 1)):
                for l_index, m in enumerate(range(xi - box_half, xi + box_half + 1)):
                    if not (k == yi and m == xi):
                        gy = frame[k + 1, m] - frame[k - 1, m]
                        gx = frame[k, m + 1] - frame[k, m - 1]
                        ng[i] += gy * uy[k_index, l_index] + gx * ux[k_index, l_index]
        return ng

    @numba.njit(nogil=True, cache=True)
    def _identify_in_frame_nb(frame, minimum_ng, box):
        y, x = _local_maxima_nb(frame, box)
        if len(x) == 0:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )
        ng = _net_gradient_nb(frame, y, x, box)
        positives = ng > minimum_ng
        return y[positives].astype(np.int32), x[positives].astype(np.int32), ng[positives]


def identify_in_frame(frame, minimum_ng, box):
    frame = frame.astype(np.float32, copy=False)
    if _NUMBA_AVAILABLE:
        return _identify_in_frame_nb(frame, float(minimum_ng), int(box))
    return _identify_in_frame_np(frame, float(minimum_ng), int(box))


# ----------------------------------------------------------------------
# Sub-pixel Gaussian fitting
# ----------------------------------------------------------------------
# Below this many candidates in a frame the prange fan-out costs more than it
# saves, so the batch driver stays on the serial jitted kernel.
_FIT_PARALLEL_MIN = 64
# Spots per Gpufit call; bounds device memory on very large batches.
_GPU_CHUNK = 100_000
# Iterations used when a "gpu" request has to be served on the CPU.
_MLE_FALLBACK_ITER = 18

# LLVM fast-math flags *without* `nnan`/`ninf`: reassociation and FMA contraction
# are what we want in the pixel loop, but NaN/Inf must keep propagating so a
# diverged fit stays detectable by the isfinite() screen in the batch driver.
_FASTMATH = {"nsz", "arcp", "contract", "afn", "reassoc"}


def _get_fit_cache(size: int):
    cache_key = int(size)
    if cache_key not in _FIT_CACHE:
        yy, xx = np.indices((cache_key, cache_key), dtype=np.float64)
        _FIT_CACHE[cache_key] = {"yy": yy, "xx": xx, "eye6": np.eye(6, dtype=np.float64)}
    return _FIT_CACHE[cache_key]


def _initial_guess(patch, xx, yy):
    patch_f = patch.astype(np.float64, copy=False)
    bg = float(np.partition(patch_f.reshape(-1), max(0, patch_f.size // 10))[max(0, patch_f.size // 10)])
    signal = np.clip(patch_f - bg, 0.0, None)
    total = float(signal.sum())

    if total > _EPS:
        x0 = float((signal * xx).sum() / total)
        y0 = float((signal * yy).sum() / total)
        var_x = float((signal * (xx - x0) ** 2).sum() / total)
        var_y = float((signal * (yy - y0) ** 2).sum() / total)
    else:
        y0 = (patch_f.shape[0] - 1) / 2.0
        x0 = (patch_f.shape[1] - 1) / 2.0
        var_x = 1.0
        var_y = 1.0

    sx = float(np.sqrt(max(var_x, 0.5)))
    sy = float(np.sqrt(max(var_y, 0.5)))
    amp = float(max(patch_f.max() - bg, 1.0))
    return np.array([amp, bg, x0, y0, sx, sy], dtype=np.float64)


def gaussian_photons(amp, sx, sy):
    """Photons in the fitted spot: the analytic integral of its own Gaussian.

    Not the background-subtracted sum over the box, which is what this used to
    be and which is biased upward - badly. Summing needs the negative residuals
    to cancel the positive ones, and clipping at zero (necessary, since a
    negative photon count is not a thing) removes exactly half of that
    cancellation. Every background pixel in the box then donates about
    sigma_bg/sqrt(2*pi) phantom photons, and there are box*box of them.

    Measured against simulated spots of known brightness: the clipped sum
    over-reports by 7% at 300 photons on a 2/pixel background and by 59% at 150
    photons on 30/pixel, growing with the background and with the box. The
    integral is within 1% everywhere in that range, and its error does not
    depend on the background at all.
    """
    return 2.0 * np.pi * amp * sx * sy


def crlb_sigma(photons, bg_per_pixel, psf_sigma, mle):
    """Localization precision, per axis, in the units psf_sigma is given in.

    Mortensen et al. (2010). The previous estimate here was s/sqrt(N), which is
    the photon-limited term alone and therefore a floor rather than an estimate:
    it omits pixelation, and it omits the background term that dominates at low
    photon counts. Against simulated spots it under-reported the precision
    actually achieved by 40% at 300 photons and by a factor of 2.7 at 150
    photons on a bright background.

    Two forms, because the fit backends are genuinely different estimators: an
    unweighted least-squares fit pays a factor 16/9 in variance that a Poisson
    maximum-likelihood fit does not.

    `bg_per_pixel` is the background in photons; for Poisson counts that is also
    its variance, which is what the formula wants. Pixel size enters as the unit
    of `psf_sigma` - passing it in pixels makes the pixel one unit wide, which
    is what the a^2/12 pixelation term below assumes.
    """
    photons = np.maximum(photons, 1.0)
    bg = np.maximum(bg_per_pixel, 0.0)
    # The PSF as the pixel grid actually samples it: a pixel of width a adds
    # a^2/12, the variance of a uniform distribution across it.
    sa2 = psf_sigma ** 2 + 1.0 / 12.0
    if mle:
        tau = 2.0 * np.pi * sa2 * bg / photons
        factor = 1.0 + 4.0 * tau + np.sqrt(2.0 * tau / (1.0 + 4.0 * tau))
    else:
        factor = 16.0 / 9.0 + 8.0 * np.pi * sa2 * bg / photons
    return np.sqrt(sa2 / photons * factor)


def _fit_params_np(data, weighted, max_iter, tol, damping):
    """Gauss-Newton in pure NumPy. Returns (params(6,), ok).

    `weighted=True` selects the Poisson-MLE weighting. This is the reference
    implementation and the fallback when numba is unavailable; the jitted cores
    below reproduce it step for step.
    """
    size = int(data.shape[0])
    cache = _get_fit_cache(size)
    yy, xx, eye6 = cache["yy"], cache["xx"], cache["eye6"]
    params = _initial_guess(data, xx, yy)
    ok = True

    for _ in range(max_iter):
        amp, bg, x0, y0, sx, sy = params
        sx2 = max(sx * sx, 0.25)
        sy2 = max(sy * sy, 0.25)
        exp_term = np.exp(-(((xx - x0) ** 2) / (2.0 * sx2) + ((yy - y0) ** 2) / (2.0 * sy2)))
        model = bg + amp * exp_term
        residual = data - model

        jac = np.stack(
            [
                exp_term,
                np.ones_like(exp_term),
                amp * exp_term * ((xx - x0) / sx2),
                amp * exp_term * ((yy - y0) / sy2),
                amp * exp_term * (((xx - x0) ** 2) / max(sx**3, 0.125)),
                amp * exp_term * (((yy - y0) ** 2) / max(sy**3, 0.125)),
            ],
            axis=-1,
        ).reshape(-1, 6)
        r = residual.reshape(-1)

        if weighted:
            w = (1.0 / np.maximum(model, 1e-3)).reshape(-1)
            jtj = jac.T @ (jac * w[:, None]) + damping * eye6
            jtr = jac.T @ (w * r)
        else:
            jtj = jac.T @ jac + damping * eye6
            jtr = jac.T @ r

        try:
            delta = np.linalg.solve(jtj, jtr)
        except np.linalg.LinAlgError:
            ok = False
            break

        params += delta
        params[0] = max(params[0], 1e-3)
        params[1] = max(params[1], 0.0)
        params[2] = float(np.clip(params[2], 0.0, size - 1.0))
        params[3] = float(np.clip(params[3], 0.0, size - 1.0))
        params[4] = float(np.clip(params[4], 0.5, size))
        params[5] = float(np.clip(params[5], 0.5, size))

        if float(np.linalg.norm(delta)) < tol:
            break

    return params, ok


def _fit_dict(_data, params, lp_floor, mle=True):
    amp, bg, x0, y0, sx, sy = (float(v) for v in params)
    photons = float(gaussian_photons(amp, sx, sy))
    lpx = float(max(crlb_sigma(photons, bg, sx, mle), lp_floor))
    lpy = float(max(crlb_sigma(photons, bg, sy, mle), lp_floor))
    return {
        "x_patch": x0, "y_patch": y0, "photons": photons,
        "sx": sx, "sy": sy, "bg": bg,
        "lpx": lpx, "lpy": lpy,
    }


def fit_gaussian_2d(patch, *, max_iter=12, tol=1e-3, damping=1e-2):
    """Least-squares Gauss-Newton fit. Model: bg + amp * exp(-((x-x0)^2/2sx^2 + (y-y0)^2/2sy^2))."""
    if patch.ndim != 2 or patch.shape[0] != patch.shape[1]:
        return None
    if int(patch.shape[0]) < 3:
        return None
    data = patch.astype(np.float64, copy=False)
    params, _ok = _fit_params_np(data, False, max_iter, tol, damping)
    return _fit_dict(data, params, 0.05, mle=False)


def fit_gaussian_2d_mle(patch, *, max_iter=20, tol=5e-4, damping=5e-2):
    """Poisson-weighted Gauss-Newton fit (Picasso-style MLE approximation)."""
    if patch.ndim != 2 or patch.shape[0] != patch.shape[1]:
        return None
    if int(patch.shape[0]) < 3:
        return None
    data = patch.astype(np.float64, copy=False)
    params, _ok = _fit_params_np(data, True, max_iter, tol, damping)
    return _fit_dict(data, params, 0.02, mle=True)


if _NUMBA_AVAILABLE:

    @numba.njit(cache=True, nogil=True, fastmath=_FASTMATH)
    def _solve6_spd(a, b, x):
        """Cholesky solve of the 6x6 normal equations, in place on `a`.

        JtJ + damping*I is symmetric positive (semi-)definite by construction, so
        a Cholesky factorisation is both the cheapest solve and a natural
        singularity test: a non-positive pivot means the system is rank
        deficient. Returns False in that case instead of raising, since numba
        cannot catch LinAlgError out of np.linalg.solve.
        """
        for i in range(6):
            s = a[i, i]
            for k in range(i):
                s -= a[i, k] * a[i, k]
            if not (s > 1e-30):  # also catches NaN
                return False
            d = np.sqrt(s)
            a[i, i] = d
            for j in range(i + 1, 6):
                t = a[j, i]
                for k in range(i):
                    t -= a[j, k] * a[i, k]
                a[j, i] = t / d
        for i in range(6):  # forward substitution
            s = b[i]
            for k in range(i):
                s -= a[i, k] * x[k]
            x[i] = s / a[i, i]
        for i in range(5, -1, -1):  # back substitution
            s = x[i]
            for k in range(i + 1, 6):
                s -= a[k, i] * x[k]
            x[i] = s / a[i, i]
        return True

    @numba.njit(cache=True, nogil=True, fastmath=_FASTMATH)
    def _initial_guess_core(data):
        """Port of _initial_guess(): 10th-percentile-ish background + moments."""
        n = data.shape[0]
        npix = n * n
        flat = np.empty(npix, dtype=np.float64)
        idx = 0
        peak = data[0, 0]
        for i in range(n):
            for j in range(n):
                v = data[i, j]
                flat[idx] = v
                if v > peak:
                    peak = v
                idx += 1
        k = npix // 10
        bg = np.partition(flat, k)[k]

        total = 0.0
        sum_x = 0.0
        sum_y = 0.0
        for i in range(n):
            for j in range(n):
                s = data[i, j] - bg
                if s < 0.0:
                    s = 0.0
                total += s
                sum_x += s * j
                sum_y += s * i

        if total > _EPS:
            x0 = sum_x / total
            y0 = sum_y / total
            acc_x = 0.0
            acc_y = 0.0
            for i in range(n):
                dy = i - y0
                for j in range(n):
                    s = data[i, j] - bg
                    if s < 0.0:
                        s = 0.0
                    dx = j - x0
                    acc_x += s * dx * dx
                    acc_y += s * dy * dy
            var_x = acc_x / total
            var_y = acc_y / total
        else:
            x0 = (n - 1) / 2.0
            y0 = (n - 1) / 2.0
            var_x = 1.0
            var_y = 1.0

        sx = np.sqrt(var_x if var_x > 0.5 else 0.5)
        sy = np.sqrt(var_y if var_y > 0.5 else 0.5)
        amp = peak - bg
        if amp < 1.0:
            amp = 1.0
        return amp, bg, x0, y0, sx, sy

    @numba.njit(cache=True, nogil=True, fastmath=_FASTMATH)
    def _fit_lsq_core(data, max_iter, tol, damping):
        """Least-squares Gauss-Newton fit of one patch. data: float64 (n, n) C-contiguous.

        JtJ/Jtr are accumulated in scalar registers inside the pixel loop; the
        (n*n, 6) Jacobian is never materialised.
        """
        n = data.shape[0]
        amp, bg, x0, y0, sx, sy = _initial_guess_core(data)
        jtj = np.empty((6, 6), dtype=np.float64)
        jtr = np.empty(6, dtype=np.float64)
        delta = np.empty(6, dtype=np.float64)
        upper = n - 1.0
        ok = True

        for _ in range(max_iter):
            sx2 = sx * sx
            if sx2 < 0.25:
                sx2 = 0.25
            sy2 = sy * sy
            if sy2 < 0.25:
                sy2 = 0.25
            sx3 = sx * sx * sx
            if sx3 < 0.125:
                sx3 = 0.125
            sy3 = sy * sy * sy
            if sy3 < 0.125:
                sy3 = 0.125
            inv_sx2 = 1.0 / sx2
            inv_sy2 = 1.0 / sy2
            inv_sx3 = 1.0 / sx3
            inv_sy3 = 1.0 / sy3
            half_inv_sx2 = 0.5 * inv_sx2
            half_inv_sy2 = 0.5 * inv_sy2

            a00 = 0.0; a01 = 0.0; a02 = 0.0; a03 = 0.0; a04 = 0.0; a05 = 0.0
            a11 = 0.0; a12 = 0.0; a13 = 0.0; a14 = 0.0; a15 = 0.0
            a22 = 0.0; a23 = 0.0; a24 = 0.0; a25 = 0.0
            a33 = 0.0; a34 = 0.0; a35 = 0.0
            a44 = 0.0; a45 = 0.0
            a55 = 0.0
            b0 = 0.0; b1 = 0.0; b2 = 0.0; b3 = 0.0; b4 = 0.0; b5 = 0.0

            for iy in range(n):
                dy = iy - y0
                ey = dy * dy * half_inv_sy2
                for ix in range(n):
                    dx = ix - x0
                    e = np.exp(-(dx * dx * half_inv_sx2 + ey))
                    ae = amp * e
                    # Jacobian row: d(model)/d[amp, bg, x0, y0, sx, sy]; j1 == 1.
                    j0 = e
                    j2 = ae * dx * inv_sx2
                    j3 = ae * dy * inv_sy2
                    j4 = ae * dx * dx * inv_sx3
                    j5 = ae * dy * dy * inv_sy3
                    r = data[iy, ix] - bg - ae

                    a00 += j0 * j0
                    a01 += j0
                    a02 += j0 * j2
                    a03 += j0 * j3
                    a04 += j0 * j4
                    a05 += j0 * j5
                    a11 += 1.0
                    a12 += j2
                    a13 += j3
                    a14 += j4
                    a15 += j5
                    a22 += j2 * j2
                    a23 += j2 * j3
                    a24 += j2 * j4
                    a25 += j2 * j5
                    a33 += j3 * j3
                    a34 += j3 * j4
                    a35 += j3 * j5
                    a44 += j4 * j4
                    a45 += j4 * j5
                    a55 += j5 * j5
                    b0 += j0 * r
                    b1 += r
                    b2 += j2 * r
                    b3 += j3 * r
                    b4 += j4 * r
                    b5 += j5 * r

            jtj[0, 0] = a00 + damping; jtj[0, 1] = a01; jtj[0, 2] = a02
            jtj[0, 3] = a03; jtj[0, 4] = a04; jtj[0, 5] = a05
            jtj[1, 0] = a01; jtj[1, 1] = a11 + damping; jtj[1, 2] = a12
            jtj[1, 3] = a13; jtj[1, 4] = a14; jtj[1, 5] = a15
            jtj[2, 0] = a02; jtj[2, 1] = a12; jtj[2, 2] = a22 + damping
            jtj[2, 3] = a23; jtj[2, 4] = a24; jtj[2, 5] = a25
            jtj[3, 0] = a03; jtj[3, 1] = a13; jtj[3, 2] = a23
            jtj[3, 3] = a33 + damping; jtj[3, 4] = a34; jtj[3, 5] = a35
            jtj[4, 0] = a04; jtj[4, 1] = a14; jtj[4, 2] = a24
            jtj[4, 3] = a34; jtj[4, 4] = a44 + damping; jtj[4, 5] = a45
            jtj[5, 0] = a05; jtj[5, 1] = a15; jtj[5, 2] = a25
            jtj[5, 3] = a35; jtj[5, 4] = a45; jtj[5, 5] = a55 + damping
            jtr[0] = b0; jtr[1] = b1; jtr[2] = b2
            jtr[3] = b3; jtr[4] = b4; jtr[5] = b5

            if not _solve6_spd(jtj, jtr, delta):
                ok = False
                break

            amp += delta[0]
            if amp < 1e-3:
                amp = 1e-3
            bg += delta[1]
            if bg < 0.0:
                bg = 0.0
            x0 += delta[2]
            if x0 < 0.0:
                x0 = 0.0
            elif x0 > upper:
                x0 = upper
            y0 += delta[3]
            if y0 < 0.0:
                y0 = 0.0
            elif y0 > upper:
                y0 = upper
            sx += delta[4]
            if sx < 0.5:
                sx = 0.5
            elif sx > n:
                sx = n
            sy += delta[5]
            if sy < 0.5:
                sy = 0.5
            elif sy > n:
                sy = n

            nrm = 0.0
            for i in range(6):
                nrm += delta[i] * delta[i]
            if np.sqrt(nrm) < tol:
                break

        return amp, bg, x0, y0, sx, sy, ok

    @numba.njit(cache=True, nogil=True, fastmath=_FASTMATH)
    def _fit_mle_core(data, max_iter, tol, damping):
        """Poisson-weighted Gauss-Newton fit of one patch (w = 1/max(model, 1e-3))."""
        n = data.shape[0]
        amp, bg, x0, y0, sx, sy = _initial_guess_core(data)
        jtj = np.empty((6, 6), dtype=np.float64)
        jtr = np.empty(6, dtype=np.float64)
        delta = np.empty(6, dtype=np.float64)
        upper = n - 1.0
        ok = True

        for _ in range(max_iter):
            sx2 = sx * sx
            if sx2 < 0.25:
                sx2 = 0.25
            sy2 = sy * sy
            if sy2 < 0.25:
                sy2 = 0.25
            sx3 = sx * sx * sx
            if sx3 < 0.125:
                sx3 = 0.125
            sy3 = sy * sy * sy
            if sy3 < 0.125:
                sy3 = 0.125
            inv_sx2 = 1.0 / sx2
            inv_sy2 = 1.0 / sy2
            inv_sx3 = 1.0 / sx3
            inv_sy3 = 1.0 / sy3
            half_inv_sx2 = 0.5 * inv_sx2
            half_inv_sy2 = 0.5 * inv_sy2

            a00 = 0.0; a01 = 0.0; a02 = 0.0; a03 = 0.0; a04 = 0.0; a05 = 0.0
            a11 = 0.0; a12 = 0.0; a13 = 0.0; a14 = 0.0; a15 = 0.0
            a22 = 0.0; a23 = 0.0; a24 = 0.0; a25 = 0.0
            a33 = 0.0; a34 = 0.0; a35 = 0.0
            a44 = 0.0; a45 = 0.0
            a55 = 0.0
            b0 = 0.0; b1 = 0.0; b2 = 0.0; b3 = 0.0; b4 = 0.0; b5 = 0.0

            for iy in range(n):
                dy = iy - y0
                ey = dy * dy * half_inv_sy2
                for ix in range(n):
                    dx = ix - x0
                    e = np.exp(-(dx * dx * half_inv_sx2 + ey))
                    ae = amp * e
                    model = bg + ae
                    w = 1.0 / (model if model > 1e-3 else 1e-3)
                    j0 = e
                    j2 = ae * dx * inv_sx2
                    j3 = ae * dy * inv_sy2
                    j4 = ae * dx * dx * inv_sx3
                    j5 = ae * dy * dy * inv_sy3
                    wr = w * (data[iy, ix] - model)

                    wj0 = w * j0
                    wj2 = w * j2
                    wj3 = w * j3
                    wj4 = w * j4
                    wj5 = w * j5

                    a00 += wj0 * j0
                    a01 += wj0
                    a02 += wj0 * j2
                    a03 += wj0 * j3
                    a04 += wj0 * j4
                    a05 += wj0 * j5
                    a11 += w
                    a12 += wj2
                    a13 += wj3
                    a14 += wj4
                    a15 += wj5
                    a22 += wj2 * j2
                    a23 += wj2 * j3
                    a24 += wj2 * j4
                    a25 += wj2 * j5
                    a33 += wj3 * j3
                    a34 += wj3 * j4
                    a35 += wj3 * j5
                    a44 += wj4 * j4
                    a45 += wj4 * j5
                    a55 += wj5 * j5
                    b0 += j0 * wr
                    b1 += wr
                    b2 += j2 * wr
                    b3 += j3 * wr
                    b4 += j4 * wr
                    b5 += j5 * wr

            jtj[0, 0] = a00 + damping; jtj[0, 1] = a01; jtj[0, 2] = a02
            jtj[0, 3] = a03; jtj[0, 4] = a04; jtj[0, 5] = a05
            jtj[1, 0] = a01; jtj[1, 1] = a11 + damping; jtj[1, 2] = a12
            jtj[1, 3] = a13; jtj[1, 4] = a14; jtj[1, 5] = a15
            jtj[2, 0] = a02; jtj[2, 1] = a12; jtj[2, 2] = a22 + damping
            jtj[2, 3] = a23; jtj[2, 4] = a24; jtj[2, 5] = a25
            jtj[3, 0] = a03; jtj[3, 1] = a13; jtj[3, 2] = a23
            jtj[3, 3] = a33 + damping; jtj[3, 4] = a34; jtj[3, 5] = a35
            jtj[4, 0] = a04; jtj[4, 1] = a14; jtj[4, 2] = a24
            jtj[4, 3] = a34; jtj[4, 4] = a44 + damping; jtj[4, 5] = a45
            jtj[5, 0] = a05; jtj[5, 1] = a15; jtj[5, 2] = a25
            jtj[5, 3] = a35; jtj[5, 4] = a45; jtj[5, 5] = a55 + damping
            jtr[0] = b0; jtr[1] = b1; jtr[2] = b2
            jtr[3] = b3; jtr[4] = b4; jtr[5] = b5

            if not _solve6_spd(jtj, jtr, delta):
                ok = False
                break

            amp += delta[0]
            if amp < 1e-3:
                amp = 1e-3
            bg += delta[1]
            if bg < 0.0:
                bg = 0.0
            x0 += delta[2]
            if x0 < 0.0:
                x0 = 0.0
            elif x0 > upper:
                x0 = upper
            y0 += delta[3]
            if y0 < 0.0:
                y0 = 0.0
            elif y0 > upper:
                y0 = upper
            sx += delta[4]
            if sx < 0.5:
                sx = 0.5
            elif sx > n:
                sx = n
            sy += delta[5]
            if sy < 0.5:
                sy = 0.5
            elif sy > n:
                sy = n

            nrm = 0.0
            for i in range(6):
                nrm += delta[i] * delta[i]
            if np.sqrt(nrm) < tol:
                break

        return amp, bg, x0, y0, sx, sy, ok

    @numba.njit(cache=True, nogil=True, fastmath=_FASTMATH)
    def _fit_batch_lsq(patches, max_iter, tol, damping):
        n = patches.shape[0]
        params = np.empty((n, 6), dtype=np.float64)
        ok = np.empty(n, dtype=np.bool_)
        for k in range(n):
            amp, bg, x0, y0, sx, sy, good = _fit_lsq_core(patches[k], max_iter, tol, damping)
            params[k, 0] = amp; params[k, 1] = bg; params[k, 2] = x0
            params[k, 3] = y0; params[k, 4] = sx; params[k, 5] = sy
            ok[k] = good
        return params, ok

    @numba.njit(cache=True, nogil=True, fastmath=_FASTMATH)
    def _fit_batch_mle(patches, max_iter, tol, damping):
        n = patches.shape[0]
        params = np.empty((n, 6), dtype=np.float64)
        ok = np.empty(n, dtype=np.bool_)
        for k in range(n):
            amp, bg, x0, y0, sx, sy, good = _fit_mle_core(patches[k], max_iter, tol, damping)
            params[k, 0] = amp; params[k, 1] = bg; params[k, 2] = x0
            params[k, 3] = y0; params[k, 4] = sx; params[k, 5] = sy
            ok[k] = good
        return params, ok

    @numba.njit(cache=True, nogil=True, fastmath=_FASTMATH, parallel=True)
    def _fit_batch_lsq_par(patches, max_iter, tol, damping):
        n = patches.shape[0]
        params = np.empty((n, 6), dtype=np.float64)
        ok = np.empty(n, dtype=np.bool_)
        for k in numba.prange(n):
            amp, bg, x0, y0, sx, sy, good = _fit_lsq_core(patches[k], max_iter, tol, damping)
            params[k, 0] = amp; params[k, 1] = bg; params[k, 2] = x0
            params[k, 3] = y0; params[k, 4] = sx; params[k, 5] = sy
            ok[k] = good
        return params, ok

    @numba.njit(cache=True, nogil=True, fastmath=_FASTMATH, parallel=True)
    def _fit_batch_mle_par(patches, max_iter, tol, damping):
        n = patches.shape[0]
        params = np.empty((n, 6), dtype=np.float64)
        ok = np.empty(n, dtype=np.bool_)
        for k in numba.prange(n):
            amp, bg, x0, y0, sx, sy, good = _fit_mle_core(patches[k], max_iter, tol, damping)
            params[k, 0] = amp; params[k, 1] = bg; params[k, 2] = x0
            params[k, 3] = y0; params[k, 4] = sx; params[k, 5] = sy
            ok[k] = good
        return params, ok

    @numba.njit(cache=True, nogil=True, fastmath=_FASTMATH)
    def _refit_mle_rows(patches, rows, params, ok, max_iter, tol, damping):
        """In-place MLE re-fit of selected rows (used for GPU non-convergence)."""
        for t in range(rows.shape[0]):
            k = rows[t]
            amp, bg, x0, y0, sx, sy, good = _fit_mle_core(patches[k], max_iter, tol, damping)
            params[k, 0] = amp; params[k, 1] = bg; params[k, 2] = x0
            params[k, 3] = y0; params[k, 4] = sx; params[k, 5] = sy
            ok[k] = good

    @numba.njit(cache=True, nogil=True, fastmath=_FASTMATH, parallel=True)
    def _refit_mle_rows_par(patches, rows, params, ok, max_iter, tol, damping):
        """As _refit_mle_rows, in parallel. `rows` holds distinct indices, so the
        scattered writes never collide."""
        for t in numba.prange(rows.shape[0]):
            k = rows[t]
            amp, bg, x0, y0, sx, sy, good = _fit_mle_core(patches[k], max_iter, tol, damping)
            params[k, 0] = amp; params[k, 1] = bg; params[k, 2] = x0
            params[k, 3] = y0; params[k, 4] = sx; params[k, 5] = sy
            ok[k] = good


def warmup_fit_kernels():
    """Force JIT compilation of the fit kernels on a dummy patch.

    `cache=True` persists the machine code to __pycache__, but an editable
    install invalidates that on every pull, so without this the first real fit
    stalls for seconds and looks like a hang.
    """
    if not _NUMBA_AVAILABLE:
        return False
    dummy = np.zeros((2, 3, 3), dtype=np.float64)
    dummy[:, 1, 1] = 10.0
    _fit_batch_lsq(dummy, 2, 1e-3, 1e-2)
    _fit_batch_mle(dummy, 2, 5e-4, 5e-2)
    _fit_batch_lsq_par(dummy, 2, 1e-3, 1e-2)
    _fit_batch_mle_par(dummy, 2, 5e-4, 5e-2)
    _refit_mle_rows(dummy, np.zeros(1, dtype=np.int64), np.zeros((2, 6)),
                    np.zeros(2, dtype=np.bool_), 2, 5e-4, 5e-2)
    _refit_mle_rows_par(dummy, np.zeros(1, dtype=np.int64), np.zeros((2, 6)),
                        np.zeros(2, dtype=np.bool_), 2, 5e-4, 5e-2)
    return True


# ----------------------------------------------------------------------
# Batch drivers
# ----------------------------------------------------------------------
def _fit_defaults(mle):
    if mle:
        return 20, 5e-4, 5e-2
    return 12, 1e-3, 1e-2


def _fit_batch_cpu(patches, *, mle, max_iter=None, tol=None, damping=None):
    """Fit an (N, n, n) float64 C-contiguous batch. Returns (params (N, 6), ok (N,))."""
    d_iter, d_tol, d_damping = _fit_defaults(mle)
    max_iter = int(d_iter if max_iter is None else max_iter)
    tol = float(d_tol if tol is None else tol)
    damping = float(d_damping if damping is None else damping)
    n_spots = int(patches.shape[0])

    if _NUMBA_AVAILABLE:
        if n_spots >= _FIT_PARALLEL_MIN:
            fn = _fit_batch_mle_par if mle else _fit_batch_lsq_par
        else:
            fn = _fit_batch_mle if mle else _fit_batch_lsq
        return fn(patches, max_iter, tol, damping)

    params = np.empty((n_spots, 6), dtype=np.float64)
    ok = np.empty(n_spots, dtype=np.bool_)
    for k in range(n_spots):
        params[k], ok[k] = _fit_params_np(patches[k], mle, max_iter, tol, damping)
    return params, ok


def _initial_guess_batch(patches):
    """Vectorised _initial_guess over an (N, n, n) batch. Returns (N, 6)."""
    n_spots, n, _ = patches.shape
    flat = patches.reshape(n_spots, -1)
    k = max(0, (n * n) // 10)
    bg = np.partition(flat, k, axis=1)[:, k]
    signal = np.clip(flat - bg[:, None], 0.0, None).reshape(n_spots, n, n)
    total = signal.sum(axis=(1, 2))

    col_w = signal.sum(axis=1)  # weight per x column
    row_w = signal.sum(axis=2)  # weight per y row
    idx = np.arange(n, dtype=np.float64)
    safe = np.where(total > _EPS, total, 1.0)
    x0 = (col_w * idx).sum(axis=1) / safe
    y0 = (row_w * idx).sum(axis=1) / safe
    var_x = (col_w * (idx[None, :] - x0[:, None]) ** 2).sum(axis=1) / safe
    var_y = (row_w * (idx[None, :] - y0[:, None]) ** 2).sum(axis=1) / safe

    degenerate = total <= _EPS
    if degenerate.any():
        x0[degenerate] = (n - 1) / 2.0
        y0[degenerate] = (n - 1) / 2.0
        var_x[degenerate] = 1.0
        var_y[degenerate] = 1.0

    out = np.empty((n_spots, 6), dtype=np.float64)
    out[:, 0] = np.maximum(flat.max(axis=1) - bg, 1.0)
    out[:, 1] = bg
    out[:, 2] = x0
    out[:, 3] = y0
    out[:, 4] = np.sqrt(np.maximum(var_x, 0.5))
    out[:, 5] = np.sqrt(np.maximum(var_y, 0.5))
    return out


def fit_gaussian_2d_gpu_batch(patches, *, tolerance=1e-4, max_iterations=40, chunk=_GPU_CHUNK):
    """One Gpufit call per chunk for an (N, n, n) batch.

    Returns (params (N, 5) float32 [amp, x0, y0, sigma, bg], states (N,) int32),
    or None if Gpufit is unavailable or the call fails. The model is the
    isotropic GAUSS_2D, so there is a single sigma - see localize_frame().
    """
    if not _GPUFIT_AVAILABLE:
        return None
    if patches.ndim != 3 or patches.shape[1] != patches.shape[2] or patches.shape[1] < 3:
        return None

    n_spots, n, _ = patches.shape
    if n_spots == 0:
        return np.empty((0, 5), np.float32), np.empty((0,), np.int32)

    guess = _initial_guess_batch(patches)
    initial = np.empty((n_spots, 5), dtype=np.float32)
    initial[:, 0] = guess[:, 0]
    initial[:, 1] = guess[:, 2]
    initial[:, 2] = guess[:, 3]
    initial[:, 3] = np.maximum(0.5, 0.5 * (guess[:, 4] + guess[:, 5]))
    initial[:, 4] = guess[:, 1]
    data = np.ascontiguousarray(patches.reshape(n_spots, n * n), dtype=np.float32)

    out_params = np.empty((n_spots, 5), dtype=np.float32)
    out_states = np.empty(n_spots, dtype=np.int32)
    try:
        model_id = getattr(_GPUFIT.ModelID, "GAUSS_2D")
        estimator_id = getattr(_GPUFIT.EstimatorID, "MLE")
        step = max(1, int(chunk))
        for start in range(0, n_spots, step):
            stop = min(start + step, n_spots)
            result = _GPUFIT.fit(
                data[start:stop], None, model_id, initial[start:stop],
                tolerance=tolerance, max_number_iterations=max_iterations,
                estimator_id=estimator_id,
            )
            params, states = result[0], result[1]
            if params is None or len(params) != (stop - start):
                return None
            out_params[start:stop] = params
            if states is None:
                out_states[start:stop] = _GPUFIT_CONVERGED
            else:
                out_states[start:stop] = states
    except Exception:
        return None
    return out_params, out_states


def _fit_batch_gpu(patches):
    """GPU batch fit with a per-row CPU fallback for rows that did not converge."""
    result = fit_gaussian_2d_gpu_batch(patches)
    if result is None:
        return _fit_batch_cpu(patches, mle=True, max_iter=_MLE_FALLBACK_ITER)

    gpu_params, states = result
    n_spots = int(patches.shape[0])
    params = np.empty((n_spots, 6), dtype=np.float64)
    sigma = np.maximum(gpu_params[:, 3].astype(np.float64), 0.25)
    params[:, 0] = gpu_params[:, 0]
    params[:, 1] = np.maximum(gpu_params[:, 4].astype(np.float64), 0.0)
    params[:, 2] = gpu_params[:, 1]
    params[:, 3] = gpu_params[:, 2]
    params[:, 4] = sigma
    params[:, 5] = sigma
    ok = np.ones(n_spots, dtype=np.bool_)

    retry = np.flatnonzero((states != _GPUFIT_CONVERGED) | ~np.isfinite(params).all(axis=1))
    if retry.size:
        max_iter, tol, damping = _fit_defaults(True)
        max_iter = _MLE_FALLBACK_ITER
        if _NUMBA_AVAILABLE:
            refit = _refit_mle_rows_par if retry.size >= _FIT_PARALLEL_MIN else _refit_mle_rows
            refit(patches, retry.astype(np.int64), params, ok, max_iter, tol, damping)
        else:
            for k in retry:
                params[k], ok[k] = _fit_params_np(patches[k], True, max_iter, tol, damping)
    return params, ok


def _extract_patches(frame, yc, xc, box):
    """Gather an (N, box, box) float64 C-contiguous stack of windows."""
    r = box // 2
    offset = np.arange(-r, r + 1)
    rows = yc[:, None] + offset[None, :]
    cols = xc[:, None] + offset[None, :]
    patches = frame[rows[:, :, None], cols[:, None, :]]
    return np.ascontiguousarray(patches, dtype=np.float64)


def _apply_camera_calibration(patches, camera_offset_adu, camera_gain_adu_per_photon):
    """(patches - offset) / gain, clipped at 0, in place."""
    gain = float(camera_gain_adu_per_photon)
    if not np.isfinite(gain) or gain <= _EPS:
        gain = 1.0
    patches -= float(camera_offset_adu)
    patches /= gain
    np.clip(patches, 0.0, None, out=patches)
    return patches


def _empty_locs():
    return {
        key: np.empty((0,), dtype=dtype)
        for key, dtype in [
            ("frame", np.int32), ("x", np.float32), ("y", np.float32),
            ("photons", np.float32), ("sx", np.float32), ("sy", np.float32),
            ("bg", np.float32), ("lpx", np.float32), ("lpy", np.float32),
            ("net_gradient", np.float32),
        ]
    }


def localize_frame(
    frame,
    y,
    x,
    box,
    *,
    frame_number,
    net_gradient=None,
    fit_backend="fast",
    camera_offset_adu=100.0,
    camera_gain_adu_per_photon=1.0,
):
    """Sub-pixel Gaussian localization for candidate detections in one frame."""
    box = int(box)
    if frame.ndim != 2 or box < 3 or box % 2 == 0:
        return _empty_locs()

    r = box // 2
    height, width = frame.shape
    yc = np.asarray(y).astype(np.int64, copy=False).ravel()
    xc = np.asarray(x).astype(np.int64, copy=False).ravel()
    if yc.size == 0 or yc.size != xc.size:
        return _empty_locs()

    in_bounds = (yc >= r) & (xc >= r) & (yc + r + 1 <= height) & (xc + r + 1 <= width)
    kept = np.flatnonzero(in_bounds)
    if kept.size == 0:
        return _empty_locs()

    # One gather of the candidate windows, converted once - never the full frame.
    patches = _extract_patches(frame, yc[kept], xc[kept], box)
    _apply_camera_calibration(patches, camera_offset_adu, camera_gain_adu_per_photon)

    backend = str(fit_backend)
    if backend == "gpu" and _GPUFIT_AVAILABLE:
        params, ok = _fit_batch_gpu(patches)
        lp_floor = 0.02
    elif backend == "gpu":
        params, ok = _fit_batch_cpu(patches, mle=True, max_iter=_MLE_FALLBACK_ITER)
        lp_floor = 0.02
    elif backend == "mle":
        params, ok = _fit_batch_cpu(patches, mle=True)
        lp_floor = 0.02
    else:
        params, ok = _fit_batch_cpu(patches, mle=False)
        lp_floor = 0.05

    good = ok & np.isfinite(params).all(axis=1)
    if not good.all():
        kept = kept[good]
        patches = patches[good]
        params = params[good]
        if kept.size == 0:
            return _empty_locs()

    amp = params[:, 0]
    bg = params[:, 1]
    sx = params[:, 4]
    sy = params[:, 5]
    photons = gaussian_photons(amp, sx, sy)
    # Least squares pays a factor 16/9 in variance that the Poisson MLE does
    # not, so the precision reported has to know which fit produced it.
    is_mle = backend != "fast"
    lpx = np.maximum(crlb_sigma(photons, bg, sx, is_mle), lp_floor)
    lpy = np.maximum(crlb_sigma(photons, bg, sy, is_mle), lp_floor)

    ng_out = np.zeros(kept.size, dtype=np.float32)
    if net_gradient is not None:
        ng = np.asarray(net_gradient, dtype=np.float32).ravel()
        within = kept < ng.size
        ng_out[within] = ng[kept[within]]

    return {
        "frame": np.full(kept.size, int(frame_number), dtype=np.int32),
        "x": ((xc[kept] - r) + params[:, 2]).astype(np.float32),
        "y": ((yc[kept] - r) + params[:, 3]).astype(np.float32),
        "photons": photons.astype(np.float32),
        "sx": sx.astype(np.float32),
        "sy": sy.astype(np.float32),
        "bg": bg.astype(np.float32),
        "lpx": lpx.astype(np.float32),
        "lpy": lpy.astype(np.float32),
        "net_gradient": ng_out,
    }


def concatenate_localizations(localizations_by_frame):
    parts = [locs for locs in localizations_by_frame if locs is not None and locs["x"].size > 0]
    if not parts:
        return _empty_locs()

    keys = parts[0].keys()
    total_rows = int(sum(int(p["x"].size) for p in parts))
    out = {key: np.empty((total_rows,), dtype=parts[0][key].dtype) for key in keys}

    offset = 0
    for p in parts:
        n = int(p["x"].size)
        for key in keys:
            out[key][offset : offset + n] = p[key]
        offset += n

    order = np.argsort(out["frame"], kind="stable")
    for key in keys:
        out[key] = out[key][order]
    return out
