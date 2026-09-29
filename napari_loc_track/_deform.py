"""Growth deformation of the tissue, measured on the white-light snapshots.

A growing root does not only drift: it stretches, several percent along its
axis over twenty minutes, faster in some places than others. A translation per
frame - the drift record - leaves micrometres of misplacement across a field of
view, so here the whole displacement field is measured, and every position is
carried to where that piece of tissue is in the *last* snapshot: the final
geometry.

The measurement, on the snapshots saved with the acquisition:

- The images are band-passed to cell-wall scale. Finer detail is streaming
  cytoplasm and dust on the optics - the first moves on its own, the second
  not at all - and either pulls a registration off the tissue.
- A rough map of each snapshot to the last is chained from consecutive pairs,
  only to get within a few pixels.
- Then, like RCC: every snapshot is warped into the final geometry with its
  current map; patches of many pairs of them - neighbours, and pairs further
  apart, and each against the last - are registered; each measured shift is the
  difference of two snapshots' remaining errors, and one least-squares solve
  over all pairs finds them all. Nothing accumulates along a chain, and every
  snapshot's map is tied to the last by many paths. Repeated until the
  corrections vanish.
- Each patch's shift weighs in each direction by how firmly its walls fix it
  there: in the elongation zone many patches hold only long walls parallel to
  the root, which say where the tissue is across the root and nothing about
  where it is along it.
- The maps are growth along the root axis (with a rate that may vary along it),
  growth across it, a rotation and a translation - seven numbers per snapshot,
  the axis taken from the direction of the cell walls (or, where the walls
  have none, from the measured strain) - or, for comparison, a general
  quadratic.

The registrations are batched - every snapshot's patch spectra computed once,
all the patches of a pair in one array operation, on every second pixel - and
run on the GPU through CuPy when there is one, else on the CPU (the warps
compiled with numba).

Positions are (x, y) = (column, row), pixel centres at integers, in pixels of
the oriented white-light frame.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

DEFORMATION_FILENAME = "deformation.json"
# The intermediate stages of a measurement, beside the record (see `steps`).
STEPS_FILENAME = "deformation_steps.npz"
DEFORMATION_KIND = "napari-loc-track growth deformation"

# Cell walls are a few pixels wide on the white-light camera (81 nm pixels):
# keep that scale, drop the cytoplasm below it and the illumination above it.
WALL_SIGMA_PX = 2.0
WALL_BACKGROUND_PX = 20.0
PATCH_PX = 384            # ~31 um: several cross walls in every patch
STEP_PX = 128
LAGS = (1, 2, 4, 8, 16)   # snapshot pairs registered, besides each against the last
MIN_QUALITY = 0.3
MODELS = ("growth", "quadratic", "affine")
# Stop refining once no map moves anywhere by more than this (px, ~4 nm).
CONVERGED_PX = 0.05


def wall_filter(image):
    """The image at cell-wall scale: smoothed, minus its own background.

    Both Gaussians in one Fourier transform, on the image mirrored at its edges
    as a direct filter would see it - the background's is 160 px wide, and
    convolving with it directly took most of a second per snapshot.
    """
    from scipy import fft

    f = np.asarray(image, dtype=np.float32)
    pad = int(math.ceil(4 * WALL_BACKGROUND_PX))
    g = np.pad(f, pad, mode="symmetric")
    fy = fft.fftfreq(g.shape[0]).astype(np.float32)[:, None]
    fx = fft.rfftfreq(g.shape[1]).astype(np.float32)[None, :]
    r2 = fy * fy + fx * fx
    two_pi2 = np.float32(2 * math.pi ** 2)
    transfer = (np.exp(-two_pi2 * np.float32(WALL_SIGMA_PX ** 2) * r2)
                * (1 - np.exp(-two_pi2 * np.float32(WALL_BACKGROUND_PX ** 2) * r2)))
    out = fft.irfft2(fft.rfft2(g, workers=-1) * transfer, s=g.shape, workers=-1)
    return np.ascontiguousarray(out[pad:pad + f.shape[0], pad:pad + f.shape[1]], dtype=np.float32)


# --- maps ---------------------------------------------------------------------------


def _quad_terms(pts, center, scale):
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    u = (pts[:, 0] - center[0]) / scale
    v = (pts[:, 1] - center[1]) / scale
    return u, v, np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])


@dataclass
class QuadMap:
    """A position map p -> B(p) @ coef, B the six quadratic terms of normalized p."""

    coef: np.ndarray          # (6, 2)
    center: tuple
    scale: float

    @classmethod
    def identity(cls, center, scale):
        coef = np.zeros((6, 2))
        coef[0] = center
        coef[1, 0] = coef[2, 1] = scale
        return cls(coef, tuple(float(c) for c in center), float(scale))

    @classmethod
    def fit(cls, src, dst, center, scale):
        _u, _v, B = _quad_terms(src, center, scale)
        coef = np.linalg.lstsq(B, np.asarray(dst, dtype=float), rcond=None)[0]
        return cls(coef, tuple(float(c) for c in center), float(scale))

    @classmethod
    def from_affine(cls, matrix, center, scale):
        """From a 3x3 (x, y, 1) affine, exactly."""
        m = np.asarray(matrix, dtype=float)
        grid = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2]], float) * scale + center
        return cls.fit(grid, (m[:2, :2] @ grid.T).T + m[:2, 2], center, scale)

    def __call__(self, pts):
        _u, _v, B = _quad_terms(pts, self.center, self.scale)
        return B @ self.coef

    def jacobian(self, pts):
        u, v, _B = _quad_terms(pts, self.center, self.scale)
        zero = np.zeros_like(u)
        du = np.column_stack([zero, np.ones_like(u), zero, 2 * u, v, zero])
        dv = np.column_stack([zero, zero, np.ones_like(u), zero, u, 2 * v])
        J = np.empty((len(u), 2, 2))
        J[:, :, 0] = du @ self.coef
        J[:, :, 1] = dv @ self.coef
        return J / self.scale

    def to_dict(self):
        return {"coef": np.round(self.coef, 8).tolist(), "center": list(self.center),
                "scale": self.scale}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["coef"], dtype=float), tuple(d["center"]), float(d["scale"]))


def displacement_basis(pts, model, theta, center, scale):
    """(N, 2, P): displacement = basis @ params, for the chosen model.

    growth: along the axis a = (cos t, sin t) a translation, a stretch, a stretch
    varying linearly along the axis (the s^2 term) and a shear; across it (m) a
    translation, a shear and a stretch - 7 numbers, affine plus the one quadratic
    term a growing root needs. quadratic: every quadratic term in x and y (12).
    affine: 6.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    x = (pts[:, 0] - center[0]) / scale
    y = (pts[:, 1] - center[1]) / scale
    one = np.ones_like(x)
    if model == "growth":
        a = np.array([math.cos(theta), math.sin(theta)])
        m = np.array([-a[1], a[0]])
        s = x * a[0] + y * a[1]
        n = x * m[0] + y * m[1]
        along = np.column_stack([one, s, s * s, n])          # times a
        across = np.column_stack([one, s, n])                 # times m
        out = np.zeros((len(x), 2, 7))
        out[:, :, :4] = along[:, None, :] * a[None, :, None]
        out[:, :, 4:] = across[:, None, :] * m[None, :, None]
        return out * scale
    terms = [one, x, y] if model == "affine" else [one, x, y, x * x, x * y, y * y]
    T = np.column_stack(terms)
    out = np.zeros((len(x), 2, 2 * T.shape[1]))
    out[:, 0, :T.shape[1]] = T
    out[:, 1, T.shape[1]:] = T
    return out * scale


def n_params(model):
    return {"growth": 7, "quadratic": 12, "affine": 6}[model]


# --- registration -------------------------------------------------------------------

# Registration runs on every SAMPLING-th pixel. The walls are smoothed over
# WALL_SIGMA_PX first, so there is nothing finer than two pixels left to lose -
# on real snapshots 2x2 binning left the registration residuals unchanged - and
# a quarter of the pixels is a quarter of the work.
SAMPLING = 2
# The correlation peak is refined to 1/UPSAMPLE of a sample: 1.6 nm here.
UPSAMPLE = 100

try:
    import numba

    if getattr(numba.config, "DISABLE_JIT", False):
        numba = None
except ImportError:  # pragma: no cover - numba is a dependency
    numba = None


def _register(a, b, upsample=50):
    from ._drift import register
    return register(a, b, upsample=upsample)


def _patch_shifts(ref, img, centres, patch, upsample=50):
    """Shift of img's content against ref's around each centre: (N, 2), quality (N,).
    One patch at a time, in double precision - for checks."""
    half = patch // 2
    out = np.full((len(centres), 2), np.nan)
    qual = np.zeros(len(centres))
    for i, (cx, cy) in enumerate(centres):
        y0, x0 = int(round(cy)) - half, int(round(cx)) - half
        if y0 < 0 or x0 < 0 or y0 + patch > ref.shape[0] or x0 + patch > ref.shape[1]:
            continue
        pa = ref[y0:y0 + patch, x0:x0 + patch]
        pb = img[y0:y0 + patch, x0:x0 + patch]
        if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
            continue
        dx, dy, q = _register(pa, pb, upsample)
        out[i] = (dx, dy)
        qual[i] = q
    return out, qual


def array_backend(backend="auto"):
    """(array module, "gpu" or "cpu"): CuPy when asked for or, with "auto", when a
    CUDA device is there and works; numpy otherwise."""
    if backend not in ("auto", "gpu", "cpu"):
        raise ValueError(f"unknown backend {backend!r}")
    if backend != "cpu":
        try:
            from ._render import _gpu
            cupy = _gpu()[0]
        except Exception:
            cupy = None
        if cupy is not None:
            return cupy, "gpu"
        if backend == "gpu":
            raise RuntimeError("the GPU was asked for, but no usable CUDA device was found")
    return np, "cpu"


def _to_host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _module_of(a):
    return np if isinstance(a, np.ndarray) else __import__(type(a).__module__.split(".")[0])


@dataclass
class PatchBank:
    """One image's patches, ready to register against another image's: windowed
    spectra, energies, which of them are whole, and each one's gradient
    structure (Gxx, Gyy, Gxy) - how firmly it pins a shift in each direction."""

    spectra: object
    energy: object
    valid: object
    tensor: object

    def take(self, index):
        return PatchBank(self.spectra[index], self.energy[index], self.valid[index],
                         self.tensor[index])


class PatchRegistration:
    """`_drift.register`, for many patches at once.

    The same recipe - Hann window, FFT cross-correlation, the peak refined by a
    matrix-multiply DFT on an upsampled grid (Guizar-Sicairos et al. 2008) -
    batched: each image's patch spectra are computed once and serve every pair
    the image is in, and all the patches of a pair are one array operation. In
    single precision, which is far below the registration's own noise; on the
    GPU when given CuPy as `xp`.
    """

    def __init__(self, size, xp=np, upsample=UPSAMPLE):
        self.xp = xp
        self.n = n = int(size)
        f32 = xp.float32
        self.window = xp.asarray(np.outer(np.hanning(n), np.hanning(n)), dtype=f32)
        # rfft2 keeps only x-frequencies >= 0: every column stands for two,
        # except DC and, for an even width, Nyquist.
        weights = np.full(n // 2 + 1, 2.0)
        weights[0] = 1.0
        if n % 2 == 0:
            weights[-1] = 1.0
        self.weights = xp.asarray(weights, dtype=f32)
        fy = np.fft.fftfreq(n)
        fx = np.arange(n // 2 + 1) / n
        self.fy = xp.asarray(fy, dtype=f32)
        self.fx = xp.asarray(fx, dtype=f32)
        # Parseval: the sum of a patch's squared gradient, from its spectrum
        FY, FX = np.meshgrid(fy, fx, indexing="ij")
        w = weights[None, :] * (2 * math.pi) ** 2 / (n * n)
        self.gradient_weights = xp.asarray(np.stack([FX * FX * w, FY * FY * w, FX * FY * w]),
                                           dtype=f32)
        up = max(1, int(upsample))
        self.passes = []
        if up > 1:
            self.passes.append((1.0 / min(up, 10), 1.0))
        if up > 10:
            self.passes.append((1.0 / up, 0.1))

    def bank(self, patches):
        """A PatchBank of (P, n, n) patches, NaN marking a patch that is not whole."""
        xp = self.xp
        p = xp.asarray(patches, dtype=xp.float32)
        valid = xp.all(xp.isfinite(p), axis=(1, 2))
        p = xp.where(valid[:, None, None], p, xp.float32(0))
        p = (p - p.mean(axis=(1, 2), keepdims=True)) * self.window
        spectra = xp.fft.rfft2(p).astype(xp.complex64, copy=False)
        power = spectra.real ** 2 + spectra.imag ** 2
        tensor = xp.einsum("pij,cij->pc", power, self.gradient_weights)
        energy = (p * p).sum(axis=(1, 2))
        return PatchBank(spectra, energy, valid & (energy > 0), tensor)

    def _cis(self, phase):
        xp = self.xp
        return (xp.cos(phase) + 1j * xp.sin(phase)).astype(xp.complex64, copy=False)

    def register(self, a, b):
        """How far b's content moved against a's, patch by patch: (P, 2) shifts
        (x, y) in samples, NaN where either patch was not whole, and the
        normalized correlation peak (P,) - 1 for identical content."""
        xp, n = self.xp, self.n
        spec = a.spectra * xp.conj(b.spectra)
        P = spec.shape[0]
        if P == 0:
            return np.empty((0, 2)), np.empty(0)
        cc = xp.fft.irfft2(spec, s=(n, n))
        flat = cc.reshape(P, -1)
        best = xp.argmax(flat, axis=1)
        rows = xp.arange(P)
        peak = flat[rows, best]
        iy, ix = best // n, best % n
        y = xp.where(iy <= n // 2, iy, iy - n).astype(xp.float32)
        x = xp.where(ix <= n // 2, ix, ix - n).astype(xp.float32)
        spec_w = spec * self.weights
        two_pi = xp.float32(2 * math.pi)
        for step, half in self.passes:
            m = int(round(half / step))
            offsets = xp.arange(-m, m + 1, dtype=xp.float32) * xp.float32(step)
            ey = self._cis(two_pi * (y[:, None, None] + offsets[None, :, None])
                           * self.fy[None, None, :])                          # (P, M, n)
            ex = self._cis(two_pi * self.fx[None, :, None]
                           * (x[:, None, None] + offsets[None, None, :]))     # (P, n/2+1, M)
            grid = xp.real(xp.matmul(xp.matmul(ey, spec_w), ex)).reshape(P, -1)
            k = xp.argmax(grid, axis=1)
            M = 2 * m + 1
            y = y + offsets[k // M]
            x = x + offsets[k % M]
            peak = grid[rows, k] / xp.float32(n * n)
        ok = a.valid & b.valid
        q = xp.where(ok, peak / xp.sqrt(xp.maximum(a.energy * b.energy, 1e-30)), 0)
        # b[m] ~ a[m + lag]: the content moved by -lag
        d = xp.where(ok[:, None], xp.stack([-x, -y], axis=1), xp.float32(np.nan))
        return _to_host(d).astype(float), _to_host(q).astype(float)


def _cut(image, starts, size, step=1):
    """(P, size/step, size/step) patches of image at (row, col) starts, taking
    every step-th pixel; NaN for one that falls off the image. On the image's
    own device."""
    xp = _module_of(image)
    n = size // step
    H, W = image.shape
    out = xp.full((len(starts), n, n), np.nan, dtype=xp.float32)
    for i, (r, c) in enumerate(starts):
        r, c = int(r), int(c)
        if r >= 0 and c >= 0 and r + size <= H and c + size <= W:
            out[i] = image[r:r + size:step, c:c + size:step]
    return out


if numba is not None:
    @numba.njit(parallel=True, cache=True)
    def _warp_kernel(img, coef, cx, cy, scale, x0, y0, step, out):  # pragma: no cover - jitted
        H, W = img.shape
        ny, nx = out.shape
        for i in numba.prange(ny):
            v = (y0 + i * step - cy) / scale
            for j in range(nx):
                u = (x0 + j * step - cx) / scale
                sx = (coef[0, 0] + coef[1, 0] * u + coef[2, 0] * v + coef[3, 0] * u * u
                      + coef[4, 0] * u * v + coef[5, 0] * v * v)
                sy = (coef[0, 1] + coef[1, 1] * u + coef[2, 1] * v + coef[3, 1] * u * u
                      + coef[4, 1] * u * v + coef[5, 1] * v * v)
                if not (sx >= 0.0 and sy >= 0.0 and sx <= W - 1 and sy <= H - 1):
                    out[i, j] = np.nan
                    continue
                jx = min(int(sx), W - 2)
                jy = min(int(sy), H - 2)
                fx = sx - jx
                fy = sy - jy
                out[i, j] = ((1.0 - fy) * ((1.0 - fx) * img[jy, jx] + fx * img[jy, jx + 1])
                             + fy * ((1.0 - fx) * img[jy + 1, jx] + fx * img[jy + 1, jx + 1]))


def _warp_into(image, inverse, box, order=1, step=1, xp=np):
    """image resampled onto every step-th pixel of box = (x0, x1, y0, y1) of
    reference coordinates, through inverse (reference -> this image); NaN where
    the image did not look. Bilinear; on the GPU when xp is CuPy."""
    x0, x1, y0, y1 = box
    ny, nx = len(range(y0, y1, step)), len(range(x0, x1, step))
    if xp is not np:
        from cupyx.scipy import ndimage as cundi

        yy, xx = xp.mgrid[y0:y1:step, x0:x1:step]
        u = (xx - inverse.center[0]) / inverse.scale
        v = (yy - inverse.center[1]) / inverse.scale
        c = inverse.coef
        src = [c[0, d] + c[1, d] * u + c[2, d] * v + c[3, d] * u * u + c[4, d] * u * v
               + c[5, d] * v * v for d in (1, 0)]
        return cundi.map_coordinates(xp.asarray(image, dtype=xp.float32), xp.stack(src),
                                     order=order, mode="constant", cval=np.nan)
    if numba is not None and order == 1:
        out = np.empty((ny, nx), np.float32)
        _warp_kernel(np.ascontiguousarray(image, dtype=np.float32), inverse.coef,
                     float(inverse.center[0]), float(inverse.center[1]), float(inverse.scale),
                     float(x0), float(y0), float(step), out)
        return out
    from scipy import ndimage

    yy, xx = np.mgrid[y0:y1:step, x0:x1:step]
    src = inverse(np.column_stack([xx.ravel(), yy.ravel()]).astype(float))
    out = ndimage.map_coordinates(image, [src[:, 1], src[:, 0]], order=order,
                                  mode="constant", cval=np.nan, prefilter=False)
    return out.reshape(yy.shape).astype(np.float32)


def _sqrt_weights(tensor):
    """(m, 2, 2) square roots of each measurement's directional weight.

    A patch holding only walls that run one way fixes a shift across them and
    not along them - the correlation peak is a ridge, and where along it the
    maximum falls is noise. Each patch keeps the same total weight as any
    other, shared between the directions in proportion to its gradient
    structure: a patch of long walls parallel to the root says where the
    tissue is across the root, and next to nothing about along it.
    """
    gxx, gyy, gxy = tensor[:, 0], tensor[:, 1], tensor[:, 2]
    trace = gxx + gyy
    T = np.empty((len(tensor), 2, 2))
    T[:, 0, 0], T[:, 1, 1], T[:, 0, 1], T[:, 1, 0] = gxx, gyy, gxy, gxy
    flat = trace <= 0
    T /= np.where(flat, 1.0, trace / 2.0)[:, None, None]
    T[flat] = np.eye(2)
    w, v = np.linalg.eigh(T)
    return np.einsum("mij,mj,mkj->mik", v, np.sqrt(np.clip(w, 0.0, None)), v)


def _robust_lstsq(A, b, n_iter=6, floor=0.05):
    """Least squares, dropping measurements beyond 4 robust sigma. A may be
    sparse: the normal equations are only parameters x parameters."""
    from scipy import sparse

    keep = np.ones(len(b), bool)
    for _ in range(n_iter):
        if sparse.issparse(A):
            Ak = A[keep]
            sol = np.linalg.lstsq((Ak.T @ Ak).toarray(), Ak.T @ b[keep], rcond=None)[0]
        else:
            sol = np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]
        r = np.abs(A @ sol - b)
        mad = 1.4826 * float(np.median(r[keep])) + 1e-9
        new = r < max(4.0 * mad, floor)
        if np.array_equal(new, keep):
            break
        keep = new
    return sol, keep, r


# --- the record ---------------------------------------------------------------------


def steps_path(path):
    """Where a record's intermediate stages are kept: beside it, named after it
    (deformation.json -> deformation_steps.npz)."""
    path = Path(path)
    return path.with_name(path.stem + "_steps.npz")



@dataclass
class DeformationRecord:
    """Where every piece of tissue of each snapshot is in the last one.

    forward[k] maps positions in snapshot k to the reference (the last
    snapshot); inverse[k] the other way. Between snapshots the map is
    interpolated in time, coefficient by coefficient; before the first and
    after the last it is held.
    """

    t: np.ndarray
    forward: list
    inverse: list
    model: str
    theta: float
    frame_shape: tuple
    region: tuple
    stats: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    # The measurement's intermediate stages, as arrays (see
    # `measure_deformation_iter`); saved beside the record, not in it.
    steps: dict = field(default_factory=dict, repr=False)

    @property
    def reference_time(self):
        return float(self.t[-1])

    def _coef_at(self, which, t):
        maps = self.forward if which == "forward" else self.inverse
        stack = np.stack([m.coef for m in maps])                   # (K+1, 6, 2)
        t = np.atleast_1d(np.asarray(t, dtype=float))
        flat = stack.reshape(len(stack), -1)
        out = np.column_stack([np.interp(t, self.t, flat[:, i]) for i in range(flat.shape[1])])
        return out.reshape(len(t), 6, 2)

    def map_at(self, t, which="forward"):
        m = self.forward[0]
        return QuadMap(self._coef_at(which, t)[0], m.center, m.scale)

    def apply(self, pts, t, which="forward"):
        """Each point (N, 2) at its own time t (N,) through the time-interpolated map."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        t = np.broadcast_to(np.asarray(t, dtype=float), (len(pts),))
        m = self.forward[0]
        _u, _v, B = _quad_terms(pts, m.center, m.scale)
        coef = self._coef_at(which, t)
        return np.einsum("nk,nkd->nd", B, coef)

    def jacobian(self, pts, t, which="forward"):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        t = np.broadcast_to(np.asarray(t, dtype=float), (len(pts),))
        m = self.forward[0]
        u, v, _B = _quad_terms(pts, m.center, m.scale)
        zero = np.zeros_like(u)
        du = np.column_stack([zero, np.ones_like(u), zero, 2 * u, v, zero])
        dv = np.column_stack([zero, zero, np.ones_like(u), zero, u, 2 * v])
        coef = self._coef_at(which, t)
        J = np.empty((len(u), 2, 2))
        J[:, :, 0] = np.einsum("nk,nkd->nd", du, coef)
        J[:, :, 1] = np.einsum("nk,nkd->nd", dv, coef)
        return J / m.scale

    def strain_along_axis(self, point=None):
        """Per snapshot: (along, across) stretch of that snapshot relative to the
        last, at `point` (default: the region's centre), from the forward map."""
        if point is None:
            x0, x1, y0, y1 = self.region
            point = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        a = np.array([math.cos(self.theta), math.sin(self.theta)])
        m = np.array([-a[1], a[0]])
        along, across = [], []
        for f in self.forward:
            J = f.jacobian(np.array([point]))[0]
            along.append(float(a @ J @ a) - 1.0)
            across.append(float(m @ J @ m) - 1.0)
        return np.array(along), np.array(across)

    def to_dict(self):
        return {"kind": DEFORMATION_KIND, "version": 1,
                "created": datetime.now().isoformat(timespec="seconds"),
                "units": "px of the oriented white-light frame; forward maps snapshot k "
                         "to the last snapshot",
                "t_epoch": [float(v) for v in self.t], "model": self.model,
                "theta_rad": float(self.theta), "frame_shape": list(self.frame_shape),
                "region_x0_x1_y0_y1": [int(v) for v in self.region],
                "forward": [m.to_dict() for m in self.forward],
                "inverse": [m.to_dict() for m in self.inverse],
                "stats": self.stats, "source": self.source}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, default=float), encoding="utf-8")
        if self.steps:
            np.savez_compressed(steps_path(path), **self.steps)
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("kind") != DEFORMATION_KIND:
            raise ValueError(f"{path.name} is not a growth deformation record")
        steps = {}
        sidecar = steps_path(path)
        if sidecar.is_file():
            try:
                with np.load(sidecar) as npz:
                    steps = {key: npz[key] for key in npz.files}
            except (OSError, ValueError):
                steps = {}
        return cls(np.asarray(d["t_epoch"], dtype=float),
                   [QuadMap.from_dict(m) for m in d["forward"]],
                   [QuadMap.from_dict(m) for m in d["inverse"]],
                   d.get("model", "growth"), float(d.get("theta_rad", 0.0)),
                   tuple(d.get("frame_shape", (0, 0))),
                   tuple(d.get("region_x0_x1_y0_y1", (0, 0, 0, 0))),
                   d.get("stats", {}), d.get("source", {}), steps)


# --- the measurement ---------------------------------------------------------------


def _grid(box, patch, step):
    x0, x1, y0, y1 = box
    half = patch // 2
    xs = np.arange(x0 + half, x1 - half + 1, step)
    ys = np.arange(y0 + half, y1 - half + 1, step)
    if not len(xs) or not len(ys):
        return np.empty((0, 2))
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()]).astype(float)


def _affine_from_shifts(centres, shifts, keep):
    """3x3 affine from centre -> centre + shift, robustly."""
    A = np.column_stack([centres[keep], np.ones(int(keep.sum()))])
    sol = np.linalg.lstsq(A, centres[keep] + shifts[keep], rcond=None)[0]
    M = np.eye(3)
    M[:2, :2] = sol[:2].T
    M[:2, 2] = sol[2]
    return M


def _principal_axis(M):
    """Direction of the largest stretch of an affine's linear part, in radians."""
    L = np.asarray(M)[:2, :2]
    S = 0.5 * (L + L.T) - np.eye(2)
    w, v = np.linalg.eigh(S)
    axis = v[:, int(np.argmax(np.abs(w)))]
    return float(math.atan2(axis[1], axis[0]) % math.pi)


def root_axis(filtered, box=None):
    """Which way the cell files run in a wall-filtered image, in radians in
    [0, pi), and how clearly (0: no preferred direction, 1: all walls one way).

    The walls along the root are long and continuous; the cross walls are
    short. So the image's gradients point mostly across the root, and the root
    runs along the direction they avoid - the structure tensor's minor axis.
    """
    x0, x1, y0, y1 = box if box is not None else (0, filtered.shape[1], 0, filtered.shape[0])
    img = np.asarray(filtered[y0:y1, x0:x1], dtype=float)
    gy, gx = np.gradient(img)
    jxx, jyy, jxy = float(np.mean(gx * gx)), float(np.mean(gy * gy)), float(np.mean(gx * gy))
    across = 0.5 * math.atan2(2 * jxy, jxx - jyy)
    coherence = math.hypot(jxx - jyy, 2 * jxy) / max(jxx + jyy, 1e-30)
    return float((across + math.pi / 2) % math.pi), float(coherence)


# Walls this clearly oriented (see `root_axis`) give the root's direction;
# below it - a tip, whose cells are as wide as long - the strain does.
AXIS_COHERENCE = 0.15


def _choose_axis(axis, filtered, region, rough):
    """The root axis for the growth model, and a note of how it was found."""
    theta_image, coherence = root_axis(filtered, region)
    theta_strain = _principal_axis(rough)
    info = {"theta_image_rad": theta_image, "wall_coherence": coherence,
            "theta_strain_rad": theta_strain}
    if axis == "image" or (axis == "auto" and coherence >= AXIS_COHERENCE):
        return theta_image, dict(info, axis_from="cell walls")
    if axis in ("auto", "strain"):
        return theta_strain, dict(info, axis_from="strain")
    return float(axis), dict(info, axis_from="given")


def _pairs(K, lags):
    pairs = set()
    for k in range(K):
        for lag in lags:
            if k + lag <= K:
                pairs.add((k, k + lag))
        pairs.add((k, K))
    return sorted(pairs)


def _translation(shift):
    M = np.eye(3)
    M[:2, 2] = shift
    return M


def _shift_patches(a, b, centres, patch, shift):
    """Patches of a at centres against b at centres + shift: total displacement.
    One patch at a time, in double precision - for checks."""
    half = patch // 2
    out = np.full((len(centres), 2), np.nan)
    qual = np.zeros(len(centres))
    for i, (cx, cy) in enumerate(centres):
        ya, xa = int(round(cy)) - half, int(round(cx)) - half
        yb, xb = ya + int(shift[1]), xa + int(shift[0])
        if min(ya, xa, yb, xb) < 0 or max(ya, yb) + patch > a.shape[0] or max(xa, xb) + patch > a.shape[1]:
            continue
        dx, dy, q = _register(a[ya:ya + patch, xa:xa + patch], b[yb:yb + patch, xb:xb + patch], 50)
        out[i] = (shift[0] + dx, shift[1] + dy)
        qual[i] = q
    return out, qual


def _register_pairs(engine, warped, starts, size, pairs, K, workers, device, cancelled, progress):
    """Every pair's patch shifts - of snapshot k's content against snapshot j's -
    and each snapshot's patch gradient structure. Each snapshot's patch spectra
    are computed once and dropped once no pair left needs them. A generator:
    yields progress, returns ({pair: (shifts, quality)}, {snapshot: tensor}),
    or None if cancelled."""
    last_use = {}
    for k, j in pairs:
        last_use[k] = max(last_use.get(k, -1), k)
        last_use[j] = max(last_use.get(j, -1), k)
    firsts = sorted({k for k, _j in pairs})
    banks, measured, tensors = {}, {}, {}
    # On the CPU the patches of several pairs are registered in parallel
    # threads. On the GPU everything stays in the calling thread: CuPy used
    # from two threads at once corrupted device memory (an illegal address,
    # which then poisons every later call in the process).
    pool = ThreadPoolExecutor(max_workers=workers) if device == "cpu" else None
    run = pool.map if pool is not None else map
    try:
        for b0 in range(0, len(firsts), workers):
            if cancelled():
                return None
            ks = set(firsts[b0:b0 + workers])
            todo = [pr for pr in pairs if pr[0] in ks]
            need = sorted({i for pr in todo for i in pr} - set(banks))
            for i, bank in zip(need, list(run(lambda i: engine.bank(_cut(warped[i], starts, size)),
                                              need))):
                banks[i] = bank
                tensors[i] = _to_host(bank.tensor).astype(float)
            for pr, res in zip(todo, list(run(lambda pr: engine.register(banks[pr[1]], banks[pr[0]]),
                                              todo))):
                measured[pr] = res
            for i in list(banks):
                if last_use.get(i, -1) <= max(ks) and i != K:
                    del banks[i]
            yield progress(len(todo))
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    return measured, tensors


def measure_deformation_iter(stack, region=None, model="growth", patch=PATCH_PX, step=STEP_PX,
                             lags=LAGS, iterations=10, threads=8, cancel=None, log=None,
                             sampling=SAMPLING, backend="auto", directional=True, axis="auto"):
    """Generator: yields progress 0..1, returns a DeformationRecord (None if cancelled).

    stack: the snapshots (indexable, .t_epoch, .shape) in the oriented white-light
    frame. region (x0, x1, y0, y1): where to measure, in the last snapshot's
    coordinates - the part of the frame that matters, e.g. the fluorescence
    camera's field plus a margin; default the whole frame less a border.
    sampling: register on every n-th pixel. backend: "auto" (the GPU when there
    is one), "gpu" or "cpu". directional: weigh each patch's shift by how
    firmly its walls fix it in each direction (see `_sqrt_weights`). axis: the
    root axis - "auto" (the cell walls when they are clearly oriented, else
    the strain), "image", "strain", or an angle in radians.

    The record's `steps` keep the intermediate stages - every snapshot's map
    after the rough chaining and after each pass, what each pass measured of
    each snapshot against the last and the correction it made - to show how
    the measurement got where it did.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    say = log or (lambda _m: None)
    n = len(stack)
    if n < 2:
        raise ValueError("need at least two snapshots")
    K = n - 1
    H, W = (int(v) for v in stack.shape[1:3])
    center, scale = (W / 2.0, H / 2.0), max(W, H) / 2.0
    s = max(1, int(sampling))
    if patch % s:
        raise ValueError(f"the patch ({patch} px) must be a multiple of the sampling ({s})")
    if region is None:
        region = (patch // 2, W - patch // 2, patch // 2, H - patch // 2)
    region = (max(0, int(region[0])), min(W, int(region[1])),
              max(0, int(region[2])), min(H, int(region[3])))
    xp, device = array_backend(backend)
    engine = PatchRegistration(patch // s, xp)
    workers = max(1, int(threads))

    def cancelled():
        return cancel is not None and cancel.is_set()

    pairs = _pairs(K, lags)
    total_work = n + K + 3 * (n + len(pairs))
    done = [0]

    def tick(amount=1):
        done[0] += amount
        return min(done[0] / total_work, 0.99)

    # Every snapshot filtered once, up front: everything after reads them many times.
    filtered = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k, image in enumerate(pool.map(lambda k: wall_filter(np.asarray(stack[k])), range(n))):
            filtered[k] = image
            if cancelled():
                return None
            yield tick()

    # Patches: on a grid over the region of the last snapshot, where there is tissue.
    ref = filtered[K]
    centres = _grid(region, patch, step)
    if not len(centres):
        raise ValueError("the region is smaller than one patch")
    half = patch // 2
    texture = np.array([float(np.std(ref[int(cy) - half:int(cy) + half,
                                         int(cx) - half:int(cx) + half])) for cx, cy in centres])
    textured = texture > 0.25 * float(np.median(texture[texture > 0])) if np.any(texture > 0) else texture > 0
    centres = centres[textured]
    if len(centres) < 6:
        raise ValueError(f"only {len(centres)} patches with tissue in the region")
    say(f"{len(centres)} patches of {patch} px over the tissue, sampled every {s} px, "
        f"registered on the {device.upper()}")

    # 1. rough maps, chained from consecutive snapshots (affine steps), followed
    # backwards from the last snapshot so the patches stay on the tissue
    pos = centres.copy()                       # the grid's positions in snapshot k + 1
    to_last = [np.eye(3) for _ in range(n)]
    for k in range(K - 1, -1, -1):
        if cancelled():
            return None
        a, b = filtered[k], filtered[k + 1]
        bx0 = int(max(0, pos[:, 0].min() - half))
        bx1 = int(min(W, pos[:, 0].max() + half))
        by0 = int(max(0, pos[:, 1].min() - half))
        by1 = int(min(H, pos[:, 1].max() + half))
        gx, gy, _q = _register(b[by0:by1:s, bx0:bx1:s], a[by0:by1:s, bx0:bx1:s], upsample=10)
        shift = np.array([round(gx * s), round(gy * s)], float)
        at = np.column_stack([np.round(pos[:, 1]) - half, np.round(pos[:, 0]) - half]).astype(int)
        moved = at + shift[::-1].astype(int)
        dd, q = engine.register(engine.bank(_cut(b, at, patch, s)),
                                engine.bank(_cut(a, moved, patch, s)))
        d = shift + dd * s                                     # where each patch was in k
        keep = (q > MIN_QUALITY) & np.isfinite(d[:, 0])
        back = (_affine_from_shifts(pos, d, keep) if keep.sum() >= 3
                else _translation(shift))                      # snapshot k+1 -> snapshot k
        to_last[k] = to_last[k + 1] @ np.linalg.inv(back)
        pos = (back[:2, :2] @ pos.T).T + back[:2, 2]
        yield tick()
    theta, axis_info = _choose_axis(axis, ref, region, to_last[0])
    forward = [QuadMap.from_affine(M, center, scale) for M in to_last]
    inverse = [QuadMap.from_affine(np.linalg.inv(M), center, scale) for M in to_last]
    say(f"rough maps chained; root axis at {math.degrees(theta):.1f} deg, from the "
        f"{axis_info['axis_from']} (walls {axis_info['wall_coherence']:.2f} oriented, along "
        f"{math.degrees(axis_info['theta_image_rad']):.0f} deg; largest strain along "
        f"{math.degrees(axis_info['theta_strain_rad']):.0f} deg)")

    # 2. refine all snapshots at once, from pairs registered in the final geometry
    from scipy import sparse

    P = n_params(model)
    history = []
    steps = {"forward": [np.stack([m.coef for m in forward])], "measured_to_last": [],
             "quality_to_last": [], "correction": []}
    G_centres = displacement_basis(centres, model, theta, center, scale)       # (Np, 2, P)
    x0, x1, y0, y1 = region
    ns = patch // s
    starts = np.column_stack([np.round((centres[:, 1] - y0) / s) - ns // 2,
                              np.round((centres[:, 0] - x0) / s) - ns // 2]).astype(int)
    sample = _grid(region, 64, 64)
    G_sample = displacement_basis(sample, model, theta, center, scale)
    # Neighbours alone would chain the errors along; the redundant set ties
    # every snapshot to the last through many paths, and converges.
    for it in range(iterations):
        warped = {}
        for k in range(n):
            if cancelled():
                return None
            # the last snapshot is the reference geometry: cropped, not warped
            warped[k] = (xp.asarray(np.ascontiguousarray(ref[y0:y1:s, x0:x1:s])) if k == K
                         else _warp_into(filtered[k], inverse[k], region, step=s, xp=xp))
            yield tick()
        result = yield from _register_pairs(engine, warped, starts, ns, pairs, K, workers,
                                            device, cancelled, tick)
        del warped
        if result is None:
            return None
        measured, tensors = result

        # r = e_j - e_k at each centre, e the error left in each map
        rows, cols, vals, rhs, lag_of_row, used = [], [], [], [], [], []
        r0 = 0
        for (k, j), (d, q) in measured.items():
            d = d * s
            ok = (q > MIN_QUALITY) & np.isfinite(d[:, 0])
            m = int(ok.sum())
            if not m:
                continue
            Wh = (_sqrt_weights(0.5 * (tensors[k] + tensors[j])[ok]) if directional
                  else np.broadcast_to(np.eye(2), (m, 2, 2)))
            Gw = np.einsum("mab,mbp->map", Wh, G_centres[ok]).reshape(2 * m, P)
            rr = np.repeat(r0 + np.arange(2 * m), P)
            for sign, blk in ((1.0, j), (-1.0, k)):
                if blk < K:
                    vals.append(sign * Gw.ravel())
                    rows.append(rr)
                    cols.append(np.tile(blk * P + np.arange(P), 2 * m))
            rhs.append(np.einsum("mab,mb->ma", Wh, d[ok]).ravel())
            lag_of_row.append(np.full(2 * m, j - k))
            used.append((k, j, ok, d))
            r0 += 2 * m
        A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(r0, K * P))
        b = np.concatenate(rhs)
        lag_of_row = np.concatenate(lag_of_row)
        sol, keep, _resid = _robust_lstsq(A, b)
        errors = np.vstack([sol.reshape(K, P), np.zeros((1, P))])
        # the residuals in pixels, unweighted, for the record
        lag_stats = {}
        at_patches = np.einsum("ncp,kp->knc", G_centres, errors)          # (n, Np, 2)
        kept_rows = keep.reshape(-1, 2).all(axis=1)
        start = 0
        by_lag = {}
        for k, j, ok, d in used:
            m = int(ok.sum())
            resid = np.abs(at_patches[j][ok] - at_patches[k][ok] - d[ok])
            by_lag.setdefault(j - k, []).append(resid[kept_rows[start:start + m]].ravel())
            start += m
        for lag in sorted(by_lag):
            values = np.concatenate(by_lag[lag])
            if values.size:
                lag_stats[str(lag)] = float(np.median(values))
        to_last_now = np.full((K, len(centres), 2), np.nan)
        quality_now = np.zeros((K, len(centres)))
        for k in range(K):
            d, q = measured.get((k, K), (None, None))
            if d is not None:
                to_last_now[k] = d * s
                quality_now[k] = q
        steps["measured_to_last"].append(to_last_now)
        steps["quality_to_last"].append(quality_now)
        steps["correction"].append(at_patches[:K])
        # e_k is a displacement in final coordinates: the map becomes p -> Phi(p) + e(Phi(p))
        squares = []
        for k in range(K):
            e = G_sample @ errors[k]
            squares.append(np.mean(np.sum(at_patches[k] ** 2, axis=1)))
            src = inverse[k](sample)                   # sample points in snapshot k
            forward[k] = QuadMap.fit(src, sample + e, center, scale)
            inverse[k] = QuadMap.fit(sample + e, src, center, scale)
        steps["forward"].append(np.stack([m.coef for m in forward]))
        # judged where there is data: at the patches, not at the region's edges
        # where the model extrapolates
        rms = float(np.sqrt(np.mean(squares)))
        history.append({"iteration": it + 1,
                        "rms_correction_px": rms,
                        "measurements": int(len(b)), "kept": int(keep.sum()),
                        "median_abs_residual_px_by_lag": lag_stats})
        say(f"pass {it + 1}: rms correction {rms:.3f} px at the patches, "
            f"{int(keep.sum())}/{len(b)} measurements kept")
        # Done when nothing moves - or when a pass no longer does better than
        # the last: the corrections are then re-measurement noise, not error.
        previous = history[-2]["rms_correction_px"] if len(history) > 1 else float("inf")
        if rms < CONVERGED_PX or rms > 0.7 * previous:
            break

    steps = {key: np.stack(value) for key, value in steps.items() if value}
    steps["centres"] = centres
    steps["patch_tensor_last"] = tensors[K]
    record = DeformationRecord(np.asarray(stack.t_epoch, dtype=float), forward, inverse, model,
                               theta, (H, W), region,
                               stats={"passes": history, "n_patches": int(len(centres)),
                                      "patch_px": int(patch), "step_px": int(step),
                                      "sampling_px": s, "device": device,
                                      "directional_weights": bool(directional),
                                      "lags": [int(v) for v in lags], "axis": axis_info},
                               steps=steps)
    return record


def measure_deformation(stack, **kwargs):
    gen = measure_deformation_iter(stack, **kwargs)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value


# --- on the fluorescence camera ---------------------------------------------------


class FluorescenceDeformation:
    """The growth map carried onto the fluorescence camera.

    A position in the fluorescence image at time t goes to the white-light
    camera through the camera map, to where that tissue is in the last snapshot
    through the growth map, and back: its position in the reference geometry, in
    the fluorescence image's pixels. `translation` is the drift record (smoothed,
    white-light px): the snapshots come every ~30 s, and whatever the tissue
    did faster than that - a jolt of the stage - is taken from the record, as
    its difference from the record interpolated between the snapshots.

    The reference geometry is the tissue as it was at `reference_time` - the
    first frame, say, where the drift correction puts everything too - or, with
    None, as it is in the last snapshot. (The methods keep the names
    to_final/from_final: "final" is whichever reference was chosen.)
    """

    def __init__(self, record, matrix, offset, sensor_origin=(0.0, 0.0), translation=None,
                 reference_time=None):
        self.record = record
        self.J = np.asarray(matrix, dtype=float)
        self.Jinv = np.linalg.inv(self.J)
        self.T = np.asarray(offset, dtype=float) - np.asarray(sensor_origin, dtype=float)
        self.translation = translation
        if translation is not None:
            sx, sy = translation(record.t)
            self._snap = (np.asarray(sx, float), np.asarray(sy, float))
        self.reference_time = None if reference_time is None else float(reference_time)
        if self.reference_time is not None:
            self._ref_fast = self.fast_motion(np.array([self.reference_time]))[0]
            # The map at the reference time, and its own inverse fitted to it: the
            # forward and inverse maps interpolated separately between snapshots
            # are not quite each other's inverse, and the reference moment must map
            # onto itself exactly.
            self._ref_forward = record.map_at(self.reference_time, "forward")
            x0, x1, y0, y1 = record.region
            margin = 0.25 * max(x1 - x0, y1 - y0)
            gx, gy = np.meshgrid(np.linspace(x0 - margin, x1 + margin, 25),
                                 np.linspace(y0 - margin, y1 + margin, 25))
            src = np.column_stack([gx.ravel(), gy.ravel()])
            self._ref_inverse = QuadMap.fit(self._ref_forward(src), src,
                                            self._ref_forward.center, self._ref_forward.scale)

    def _to_reference(self, f):
        """Last-snapshot geometry (white-light px) -> the reference geometry."""
        if self.reference_time is None:
            return f
        return self._ref_inverse(f) + self._ref_fast

    def _from_reference(self, r):
        if self.reference_time is None:
            return r
        return self._ref_forward(np.atleast_2d(r) - self._ref_fast)

    def fast_motion(self, t):
        """(N, 2) white-light px the record moved beyond the snapshots' interpolation."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self.translation is None:
            return np.zeros((len(t), 2))
        tx, ty = self.translation(t)
        return np.column_stack([np.asarray(tx) - np.interp(t, self.record.t, self._snap[0]),
                                np.asarray(ty) - np.interp(t, self.record.t, self._snap[1])])

    def to_white_light(self, q):
        return (np.atleast_2d(q) - self.T) @ self.Jinv.T

    def to_fluorescence(self, b):
        return np.atleast_2d(b) @ self.J.T + self.T

    def to_final(self, q, t):
        """Fluorescence px (N, 2) at times t -> fluorescence px in the reference geometry."""
        t = np.broadcast_to(np.asarray(t, dtype=float), (len(np.atleast_2d(q)),))
        b = self.to_white_light(q) - self.fast_motion(t)
        return self.to_fluorescence(self._to_reference(self.record.apply(b, t)))

    def from_final(self, qf, t):
        """The other way: where a reference-geometry position was at time t."""
        t = np.broadcast_to(np.asarray(t, dtype=float), (len(np.atleast_2d(qf)),))
        f = self._from_reference(self.to_white_light(qf))
        b = self.record.apply(f, t, which="inverse") + self.fast_motion(t)
        return self.to_fluorescence(b)

    def jacobian(self, q, t):
        """(N, 2, 2): how a small step at q, t is stretched on its way to the reference geometry."""
        t = np.broadcast_to(np.asarray(t, dtype=float), (len(np.atleast_2d(q)),))
        b = self.to_white_light(q) - self.fast_motion(t)
        D = self.record.jacobian(b, t)
        if self.reference_time is not None:
            R = self._ref_inverse.jacobian(self.record.apply(b, t))
            D = np.einsum("nij,njk->nik", R, D)
        return np.einsum("ij,njk,kl->nil", self.J, D, self.Jinv)

    def map_rows(self, q, t, center, chunk=200000):
        """(final, local) positions for many localizations, in chunks.

        final: where each one is in the final geometry. local: the same, with the
        local stretch taken back out around `center` - steps between neighbouring
        frames at their true length, which is what diffusion is measured on.
        """
        q = np.atleast_2d(np.asarray(q, dtype=float))
        t = np.broadcast_to(np.asarray(t, dtype=float), (len(q),))
        final = np.empty_like(q)
        local = np.empty_like(q)
        c = np.asarray(center, dtype=float)
        for start in range(0, len(q), chunk):
            sl = slice(start, start + chunk)
            f = self.to_final(q[sl], t[sl])
            J = self.jacobian(q[sl], t[sl])
            final[sl] = f
            local[sl] = np.einsum("nij,nj->ni", np.linalg.inv(J), f - c) + c
        return final, local
