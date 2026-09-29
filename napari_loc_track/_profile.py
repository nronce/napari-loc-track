"""Intensity profiles along a line drawn over the image layers.

Each visible image layer is read at the time point on the slider, along the line,
at half its own pixel, and averaged across the line over a chosen width - the
values as stored, not as coloured. Optionally a Gaussian on a constant is fitted
to each profile, for a width (FWHM) and a centre.
"""
from __future__ import annotations

import math

import numpy as np

from ._view_export import _plane, _sample, layer_pixel_nm

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def sample_line(layer, world_point, start, end, width=0.0, step=None):
    """(distance along the line, mean value across it) for one layer.

    start, end: world (y, x). width: across the line, in world units (0: one
    sample). step: spacing along and across, default half the layer's pixel.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.hypot(*(end - start)))
    step = float(step) if step else layer_pixel_nm(layer) / 2.0
    n_along = max(2, int(math.ceil(length / step)) + 1)
    t = np.linspace(0.0, 1.0, n_along)
    along = start[None, :] + t[:, None] * (end - start)[None, :]
    direction = (end - start) / max(length, 1e-12)
    normal = np.array([-direction[1], direction[0]])
    n_across = max(1, int(round(float(width) / step)) + 1) if width > 0 else 1
    offsets = (np.linspace(-width / 2.0, width / 2.0, n_across) if n_across > 1
               else np.zeros(1))
    points = along[:, None, :] + offsets[None, :, None] * normal[None, None, :]
    plane = _plane(layer, world_point)
    if plane.ndim == 3:                    # RGB: its brightness
        plane = plane[..., :3].mean(axis=-1)
    values = _sample(plane, layer, points[..., 0], points[..., 1], order=1)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values, axis=1) if n_across > 1 else values[:, 0]
    return t * length, mean


def fit_gaussian(distance, values):
    """{centre, fwhm, amplitude, offset} of a Gaussian on a constant, or None."""
    from scipy.optimize import curve_fit

    ok = np.isfinite(values)
    if ok.sum() < 5:
        return None
    d, v = distance[ok], values[ok]
    offset = float(np.percentile(v, 10))
    peak = int(np.argmax(v))
    amplitude = float(v[peak] - offset)
    above = d[v > offset + amplitude / 2.0]
    sigma = max(float(above.max() - above.min()) / FWHM_PER_SIGMA if above.size > 1
                else (d[-1] - d[0]) / 20.0, 1e-9)

    def model(x, a, c, s, b):
        return a * np.exp(-0.5 * ((x - c) / s) ** 2) + b

    try:
        p, _cov = curve_fit(model, d, v, p0=(amplitude, d[peak], sigma, offset), maxfev=4000)
    except (RuntimeError, ValueError):
        return None
    a, c, s, b = p
    if not (np.isfinite(p).all() and d[0] <= c <= d[-1]):
        return None
    return {"centre": float(c), "fwhm": float(abs(s) * FWHM_PER_SIGMA), "amplitude": float(a),
            "offset": float(b), "curve": model(d, *p), "distance": d}
