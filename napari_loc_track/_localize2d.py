"""2D single-molecule localization: spot detection + sub-pixel Gaussian fitting.

Adapted from the legacy `forLocalization2D` prototype (itself inspired by the
Picasso package, Jungmann lab), trimmed to what the napari plugin's
"Localize (2D)" tab needs: candidate detection (local maxima + net gradient)
and per-candidate Gaussian fitting (least-squares, Poisson-MLE, or GPU via
Gpufit when available). Rendering and GPU-batched multi-frame fitting are
intentionally left out - the plugin visualizes/filters localizations with its
own napari layers once they're loaded into a dataframe.
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
    ux = xx / unorm
    uy = yy / unorm

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
        y_size, x_size = frame.shape
        maxima_map = np.zeros(frame.shape, np.uint8)
        box_half = int(box / 2)
        box_half_1 = box_half + 1
        for y in range(box_half, y_size - box_half_1):
            for x in range(box_half, x_size - box_half_1):
                local = frame[
                    y - box_half : y + box_half + 1,
                    x - box_half : x + box_half + 1,
                ]
                flat_max = np.argmax(local)
                y_local = int(flat_max / box)
                x_local = int(flat_max % box)
                if (y_local == box_half) and (x_local == box_half):
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
def _convert_patch_to_photon_units(patch, *, camera_offset_adu, camera_gain_adu_per_photon):
    gain = float(camera_gain_adu_per_photon)
    if not np.isfinite(gain) or gain <= _EPS:
        gain = 1.0
    offset = float(camera_offset_adu)
    out = (patch.astype(np.float32, copy=False) - offset) / gain
    return np.clip(out, 0.0, None, out=out)


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


def _photons_bg_subtracted_from_patch(patch, bg):
    patch_f = patch.astype(np.float64, copy=False)
    signal = np.clip(patch_f - float(bg), 0.0, None)
    return float(signal.sum())


def _background_std_from_fit(patch, xx, yy, params):
    """Population standard deviation of the fitted-model residuals.

    This follows ThunderSTORM's ``bkgstd`` definition: the standard
    deviation of all pixels in the fitting ROI after subtracting the fitted
    Gaussian plus its constant offset.  ``patch`` and ``params`` are already
    in photon units here.
    """
    amp, bg, x0, y0, sx, sy = [float(value) for value in params]
    sx2 = max(sx * sx, 0.25)
    sy2 = max(sy * sy, 0.25)
    exp_term = np.exp(
        -(
            ((xx - x0) ** 2) / (2.0 * sx2)
            + ((yy - y0) ** 2) / (2.0 * sy2)
        )
    )
    model = bg + amp * exp_term
    return float(np.std(patch.astype(np.float64, copy=False) - model, ddof=0))


def fit_gaussian_2d(patch, *, max_iter=12, tol=1e-3, damping=1e-2):
    """Least-squares Gauss-Newton fit. Model: bg + amp * exp(-((x-x0)^2/2sx^2 + (y-y0)^2/2sy^2))."""
    if patch.ndim != 2 or patch.shape[0] != patch.shape[1]:
        return None
    size = int(patch.shape[0])
    if size < 3:
        return None

    data = patch.astype(np.float64, copy=False)
    cache = _get_fit_cache(size)
    yy, xx, eye6 = cache["yy"], cache["xx"], cache["eye6"]
    params = _initial_guess(data, xx, yy)

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

        jtj = jac.T @ jac + damping * eye6
        jtr = jac.T @ r
        try:
            delta = np.linalg.solve(jtj, jtr)
        except np.linalg.LinAlgError:
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

    amp, bg, x0, y0, sx, sy = params
    photons = _photons_bg_subtracted_from_patch(data, bg)
    bkgstd = _background_std_from_fit(data, xx, yy, params)
    lpx = float(max(sx / np.sqrt(max(photons, 1.0)), 0.05))
    lpy = float(max(sy / np.sqrt(max(photons, 1.0)), 0.05))
    return {
        "x_patch": float(x0), "y_patch": float(y0), "photons": photons,
        "sx": float(sx), "sy": float(sy), "bg": float(bg), "bkgstd": bkgstd,
        "lpx": lpx, "lpy": lpy,
    }


def fit_gaussian_2d_mle(patch, *, max_iter=20, tol=5e-4, damping=5e-2):
    """Poisson-weighted Gauss-Newton fit (Picasso-style MLE approximation)."""
    if patch.ndim != 2 or patch.shape[0] != patch.shape[1]:
        return None
    size = int(patch.shape[0])
    if size < 3:
        return None

    data = patch.astype(np.float64, copy=False)
    cache = _get_fit_cache(size)
    yy, xx, eye6 = cache["yy"], cache["xx"], cache["eye6"]
    params = _initial_guess(data, xx, yy)

    for _ in range(max_iter):
        amp, bg, x0, y0, sx, sy = params
        sx2 = max(sx * sx, 0.25)
        sy2 = max(sy * sy, 0.25)
        exp_term = np.exp(-(((xx - x0) ** 2) / (2.0 * sx2) + ((yy - y0) ** 2) / (2.0 * sy2)))
        model = bg + amp * exp_term
        model_safe = np.maximum(model, 1e-3)
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
        w = (1.0 / model_safe).reshape(-1)

        jtj = jac.T @ (jac * w[:, None]) + damping * eye6
        jtr = jac.T @ (w * r)
        try:
            delta = np.linalg.solve(jtj, jtr)
        except np.linalg.LinAlgError:
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

    amp, bg, x0, y0, sx, sy = params
    photons = _photons_bg_subtracted_from_patch(data, bg)
    bkgstd = _background_std_from_fit(data, xx, yy, params)
    lpx = float(max(sx / np.sqrt(max(photons, 1.0)), 0.02))
    lpy = float(max(sy / np.sqrt(max(photons, 1.0)), 0.02))
    return {
        "x_patch": float(x0), "y_patch": float(y0), "photons": photons,
        "sx": float(sx), "sy": float(sy), "bg": float(bg), "bkgstd": bkgstd,
        "lpx": lpx, "lpy": lpy,
    }


def fit_gaussian_2d_gpu(patch):
    if not _GPUFIT_AVAILABLE:
        return None
    if patch.ndim != 2 or patch.shape[0] != patch.shape[1]:
        return None
    size = int(patch.shape[0])
    if size < 3:
        return None

    cache = _get_fit_cache(size)
    yy, xx = cache["yy"], cache["xx"]
    p0 = _initial_guess(patch, xx, yy)
    amp0, bg0, x0, y0, sx0, sy0 = [float(v) for v in p0]
    sigma0 = max(0.5, 0.5 * (sx0 + sy0))

    data = patch.astype(np.float32, copy=False).reshape(1, -1)
    initial = np.asarray([[amp0, x0, y0, sigma0, bg0]], dtype=np.float32)

    try:
        model_id = getattr(_GPUFIT.ModelID, "GAUSS_2D")
        estimator_id = getattr(_GPUFIT.EstimatorID, "MLE")
        fit_result = _GPUFIT.fit(
            data, None, model_id, initial,
            tolerance=1e-4, max_number_iterations=40, estimator_id=estimator_id,
        )
        params, states = fit_result[0], fit_result[1]
        if params is None or len(params) == 0:
            return None
        if states is not None and len(states) > 0:
            converged_state = getattr(_GPUFIT.State, "CONVERGED", 0)
            if int(states[0]) != int(converged_state):
                return None
        amp = float(params[0, 0])
        x_patch = float(params[0, 1])
        y_patch = float(params[0, 2])
        sigma = max(float(params[0, 3]), 0.25)
        bg = max(float(params[0, 4]), 0.0)
    except Exception:
        return None

    photons = _photons_bg_subtracted_from_patch(patch, bg)
    bkgstd = _background_std_from_fit(
        patch, xx, yy, (amp, bg, x_patch, y_patch, sigma, sigma)
    )
    lpx = float(max(sigma / np.sqrt(max(photons, 1.0)), 0.02))
    return {
        "x_patch": x_patch, "y_patch": y_patch, "photons": photons,
        "sx": sigma, "sy": sigma, "bg": bg, "bkgstd": bkgstd,
        "lpx": lpx, "lpy": lpx,
    }


def _empty_locs():
    return {
        key: np.empty((0,), dtype=dtype)
        for key, dtype in [
            ("frame", np.int32), ("x", np.float32), ("y", np.float32),
            ("photons", np.float32), ("sx", np.float32), ("sy", np.float32),
            ("bg", np.float32), ("bkgstd", np.float32),
            ("lpx", np.float32), ("lpy", np.float32),
            ("net_gradient", np.float32),
        ]
    }


def _to_numpy_locs(locs):
    if not locs["x"]:
        return _empty_locs()
    return {
        "frame": np.asarray(locs["frame"], dtype=np.int32),
        "x": np.asarray(locs["x"], dtype=np.float32),
        "y": np.asarray(locs["y"], dtype=np.float32),
        "photons": np.asarray(locs["photons"], dtype=np.float32),
        "sx": np.asarray(locs["sx"], dtype=np.float32),
        "sy": np.asarray(locs["sy"], dtype=np.float32),
        "bg": np.asarray(locs["bg"], dtype=np.float32),
        "bkgstd": np.asarray(locs["bkgstd"], dtype=np.float32),
        "lpx": np.asarray(locs["lpx"], dtype=np.float32),
        "lpy": np.asarray(locs["lpy"], dtype=np.float32),
        "net_gradient": np.asarray(locs["net_gradient"], dtype=np.float32),
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
    if frame.ndim != 2 or box < 3 or box % 2 == 0:
        return _empty_locs()

    r = int(box // 2)
    out = {k: [] for k in _empty_locs()}
    frame_ph = _convert_patch_to_photon_units(
        frame, camera_offset_adu=camera_offset_adu, camera_gain_adu_per_photon=camera_gain_adu_per_photon
    )
    height, width = frame_ph.shape

    for i, (yi, xi) in enumerate(zip(y, x)):
        yi_i, xi_i = int(yi), int(xi)
        y0, y1 = yi_i - r, yi_i + r + 1
        x0, x1 = xi_i - r, xi_i + r + 1
        if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
            continue
        patch = frame_ph[y0:y1, x0:x1]

        fit = None
        if fit_backend == "gpu":
            fit = fit_gaussian_2d_gpu(patch)
            if fit is None:
                fit = fit_gaussian_2d_mle(patch, max_iter=18)
        elif fit_backend == "mle":
            fit = fit_gaussian_2d_mle(patch, max_iter=20)
        else:
            fit = fit_gaussian_2d(patch, max_iter=12)
        if fit is None:
            fit = fit_gaussian_2d(patch, max_iter=10)
        if fit is None:
            continue

        out["frame"].append(int(frame_number))
        out["x"].append(float(x0 + fit["x_patch"]))
        out["y"].append(float(y0 + fit["y_patch"]))
        out["photons"].append(float(fit["photons"]))
        out["sx"].append(float(fit["sx"]))
        out["sy"].append(float(fit["sy"]))
        out["bg"].append(float(fit["bg"]))
        out["bkgstd"].append(float(fit["bkgstd"]))
        out["lpx"].append(float(fit["lpx"]))
        out["lpy"].append(float(fit["lpy"]))
        out["net_gradient"].append(float(net_gradient[i]) if net_gradient is not None and i < len(net_gradient) else 0.0)

    return _to_numpy_locs(out)


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
