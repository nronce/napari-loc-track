"""Microbenchmark for the 2D localization fit backends.

Reports microseconds per spot for every fitting path so regressions are visible.

    python benchmarks/bench_fit.py --spots 2000 --box 7

Reference numbers that motivated the jitted cores (7x7 box, per spot):
NumPy LSQ 735 us, NumPy Poisson-MLE 2762 us.
"""
import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parents[1] / "napari_loc_track"


def _load_localize2d():
    """Load _localize2d without importing the napari/Qt stack.

    Registered under its real dotted name because numba's on-disk cache pickles
    the defining module name - an alias would poison the plugin's own cache.
    """
    if "napari_loc_track" not in sys.modules:
        pkg = types.ModuleType("napari_loc_track")
        pkg.__path__ = [str(PKG_DIR)]
        sys.modules["napari_loc_track"] = pkg
    name = "napari_loc_track._localize2d"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PKG_DIR / "_localize2d.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


loc = _load_localize2d()


def make_patches(n_spots, box, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((box, box), dtype=np.float64)
    centre = (box - 1) / 2.0
    patches = np.empty((n_spots, box, box), dtype=np.float64)
    for k in range(n_spots):
        amp = rng.uniform(80.0, 900.0)
        bg = rng.uniform(2.0, 25.0)
        x0 = centre + rng.uniform(-1.0, 1.0)
        y0 = centre + rng.uniform(-1.0, 1.0)
        sx = rng.uniform(0.9, 1.7)
        sy = rng.uniform(0.9, 1.7)
        model = bg + amp * np.exp(
            -(((xx - x0) ** 2) / (2 * sx * sx) + ((yy - y0) ** 2) / (2 * sy * sy))
        )
        patches[k] = rng.poisson(model)
    return patches


def make_frame(shape, n_spots, box, seed=2):
    """A frame with `n_spots` well-separated Gaussians, plus their pixel coords."""
    rng = np.random.default_rng(seed)
    frame = rng.poisson(100.0, size=shape).astype(np.float32)
    yy, xx = np.indices(shape, dtype=np.float32)
    margin = box + 1
    per_side = int(np.ceil(np.sqrt(n_spots)))
    step_y = (shape[0] - 2 * margin) / max(per_side, 1)
    step_x = (shape[1] - 2 * margin) / max(per_side, 1)
    ys, xs = [], []
    for k in range(n_spots):
        cy = margin + step_y * (k // per_side) + rng.uniform(0, 0.9)
        cx = margin + step_x * (k % per_side) + rng.uniform(0, 0.9)
        frame += (600.0 * np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * 1.3 ** 2))).astype(np.float32)
        ys.append(int(round(cy)))
        xs.append(int(round(cx)))
    return frame, np.asarray(ys, np.int32), np.asarray(xs, np.int32)


def timeit(fn, n_spots, repeats):
    fn()  # warm caches / JIT
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e6 / n_spots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", type=int, default=2000)
    ap.add_argument("--box", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--numpy-spots", type=int, default=100,
                    help="the pure-NumPy paths are ~1000x slower; time fewer spots")
    ap.add_argument("--frame-shape", type=int, nargs=2, default=(512, 512))
    ap.add_argument("--detect", action="store_true", help="also time candidate detection")
    args = ap.parse_args()

    print(f"numba available: {loc.is_numba_available()}   gpufit available: {loc.is_gpufit_available()}")
    print(f"box={args.box}  spots={args.spots}  repeats={args.repeats}")
    if loc.is_numba_available():
        t0 = time.perf_counter()
        loc.warmup_fit_kernels()
        print(f"kernel warmup: {time.perf_counter() - t0:.2f} s")

    patches = make_patches(args.spots, args.box)
    small = patches[: args.numpy_spots]
    rows = []

    rows.append(("numpy   lsq (per spot)", timeit(
        lambda: [loc.fit_gaussian_2d(p) for p in small], small.shape[0], max(1, args.repeats // 2))))
    rows.append(("numpy   mle (per spot)", timeit(
        lambda: [loc.fit_gaussian_2d_mle(p) for p in small], small.shape[0], max(1, args.repeats // 2))))

    if loc.is_numba_available():
        rows.append(("jit     lsq serial", timeit(
            lambda: loc._fit_batch_lsq(patches, 12, 1e-3, 1e-2), args.spots, args.repeats)))
        rows.append(("jit     mle serial", timeit(
            lambda: loc._fit_batch_mle(patches, 20, 5e-4, 5e-2), args.spots, args.repeats)))
        rows.append(("jit     lsq parallel", timeit(
            lambda: loc._fit_batch_lsq_par(patches, 12, 1e-3, 1e-2), args.spots, args.repeats)))
        rows.append(("jit     mle parallel", timeit(
            lambda: loc._fit_batch_mle_par(patches, 20, 5e-4, 5e-2), args.spots, args.repeats)))

    if loc.is_gpufit_available():
        rows.append(("gpufit  batch", timeit(
            lambda: loc.fit_gaussian_2d_gpu_batch(patches), args.spots, args.repeats)))

    frame, ys, xs = make_frame(tuple(args.frame_shape), args.spots, args.box)
    ng = np.ones(ys.size, dtype=np.float32)
    for backend in ("fast", "mle", "gpu") if loc.is_gpufit_available() else ("fast", "mle"):
        rows.append((f"localize_frame({backend})", timeit(
            lambda b=backend: loc.localize_frame(
                frame, ys, xs, args.box, frame_number=0, net_gradient=ng, fit_backend=b),
            ys.size, args.repeats)))

    width = max(len(name) for name, _ in rows)
    print()
    print(f"{'backend'.ljust(width)}   us/spot")
    print("-" * (width + 12))
    for name, us in rows:
        print(f"{name.ljust(width)}   {us:9.2f}")

    if args.detect:
        print()
        big = np.random.default_rng(0).poisson(100.0, size=tuple(args.frame_shape)).astype(np.float32)
        t = timeit(lambda: loc.identify_in_frame(big, 500.0, args.box), 1, args.repeats)
        print(f"identify_in_frame on {big.shape[0]}x{big.shape[1]}: {t / 1000:.2f} ms/frame")


if __name__ == "__main__":
    main()
