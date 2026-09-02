"""Benchmark for SMLM rendering, so a regression in the splat is visible.

Reports wall time per render and per million localizations, for every mode and
whichever backends are installed.

    python benchmarks/bench_render.py --locs 2000000 --field 512 --oversampling 10

The shape of the numbers matters more than their absolute value: the histogram
is a lower bound (one write per localization), the per-localization Gaussian
should be within a small factor of it, and the global-sigma mode is dominated by
the separable blur over the whole output, so it grows with the *image* size
rather than the localization count.
"""
import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parents[1] / "napari_loc_track"


def _load_render():
    """Load _render without importing the napari/Qt stack.

    Registered under its real dotted name because numba's on-disk cache pickles
    the defining module name - an alias would poison the plugin's own cache.
    """
    if "napari_loc_track" not in sys.modules:
        pkg = types.ModuleType("napari_loc_track")
        pkg.__path__ = [str(PKG_DIR)]
        sys.modules["napari_loc_track"] = pkg
    name = "napari_loc_track._render"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PKG_DIR / "_render.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


render = _load_render()


def make_localizations(n, field, frames, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "x": rng.uniform(-0.5, field - 0.5, n),
        "y": rng.uniform(-0.5, field - 0.5, n),
        # 10-40 nm precision at a 100 nm pixel
        "sigma": rng.uniform(0.1, 0.4, n),
        "photons": rng.uniform(200, 5000, n),
        "frames": rng.integers(0, frames, n),
    }


def timeit(call, repeats):
    call()  # warm up: numba compiles, CuPy builds its kernels and its pool
    best = min(_time_once(call) for _ in range(repeats))
    return best


def _time_once(call):
    start = time.perf_counter()
    call()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--locs", type=int, default=1_000_000)
    parser.add_argument("--field", type=int, default=512, help="camera pixels per side")
    parser.add_argument("--oversampling", type=int, default=10)
    parser.add_argument("--frames", type=int, default=10_000)
    parser.add_argument("--movie-groups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    data = make_localizations(args.locs, args.field, args.frames)
    shape = (args.field, args.field)
    rows, cols = render.output_shape(shape, args.oversampling)

    print(f"{args.locs} localizations, {args.field}x{args.field} camera px, "
          f"{args.oversampling}x oversampling -> {cols}x{rows} px "
          f"({render.estimate_bytes(shape, args.oversampling) / 1e6:.0f} MB per frame)")
    print(render.render_gpu_status())
    print(f"numba: {'yes' if render.is_numba_available() else 'no'}")

    backends = [False, True] if render.is_render_gpu_available() else [False]
    results = []
    for mode in render.MODES:
        extra = {}
        if mode == "gaussian_local":
            extra["sigma_px"] = data["sigma"]
        elif mode == "gaussian_global":
            extra["global_sigma_px"] = 0.2
        for gpu in backends:
            elapsed = timeit(lambda m=mode, g=gpu, e=extra: render.render_frame(
                data["x"], data["y"], shape=shape, oversampling=args.oversampling,
                mode=m, gpu=g, **e), args.repeats)
            results.append((f"{mode} ({'gpu' if gpu else 'cpu'})", elapsed))

    width = max(len(name) for name, _ in results)
    print()
    print(f"{'mode'.ljust(width)}      s/render   s/million locs")
    print("-" * (width + 32))
    for name, elapsed in results:
        print(f"{name.ljust(width)}   {elapsed:9.3f}   {elapsed / (args.locs / 1e6):12.3f}")

    per_group = max(1, args.frames // args.movie_groups)
    print()
    for grouping in render.GROUPINGS:
        elapsed = timeit(lambda g=grouping: render.render_movie(
            data["x"], data["y"], data["frames"], shape=shape,
            oversampling=max(1, args.oversampling // 4), frames_per_group=per_group,
            grouping=g, step=per_group // 2, mode="gaussian_local",
            sigma_px=data["sigma"], gpu=render.is_render_gpu_available()), 1)
        print(f"movie, {grouping:11s} {per_group} raw frames per frame: {elapsed:6.2f} s")


if __name__ == "__main__":
    main()
