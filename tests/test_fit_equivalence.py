"""The jitted fit cores must reproduce the pure-NumPy reference fits.

The NumPy implementations in `_localize2d` stay the documented fallback for
installs without numba, so they are the reference here: any divergence beyond
floating-point noise means the ported initial guess or the parameter clamping
drifted.

See `conftest.load_localize2d` for why the module is loaded from its file.
"""
import numpy as np
import pytest

from conftest import load_localize2d

loc = load_localize2d()


BOX = 7
N_PATCHES = 500


def _synthetic_patches(n_patches=N_PATCHES, box=BOX, seed=12345):
    """Poisson realisations of 2D Gaussians, with the ground truth that made them."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((box, box), dtype=np.float64)
    centre = (box - 1) / 2.0

    truth = np.empty((n_patches, 6), dtype=np.float64)
    truth[:, 0] = rng.uniform(80.0, 900.0, n_patches)      # amp
    truth[:, 1] = rng.uniform(2.0, 25.0, n_patches)        # bg
    truth[:, 2] = centre + rng.uniform(-1.0, 1.0, n_patches)   # x0
    truth[:, 3] = centre + rng.uniform(-1.0, 1.0, n_patches)   # y0
    truth[:, 4] = rng.uniform(0.9, 1.7, n_patches)         # sx
    truth[:, 5] = rng.uniform(0.9, 1.7, n_patches)         # sy

    patches = np.empty((n_patches, box, box), dtype=np.float64)
    for k in range(n_patches):
        amp, bg, x0, y0, sx, sy = truth[k]
        model = bg + amp * np.exp(
            -(((xx - x0) ** 2) / (2.0 * sx * sx) + ((yy - y0) ** 2) / (2.0 * sy * sy))
        )
        patches[k] = rng.poisson(model)
    return patches, truth


def _reference_params(patches, mle):
    max_iter, tol, damping = loc._fit_defaults(mle)
    out = np.empty((patches.shape[0], 6), dtype=np.float64)
    for k in range(patches.shape[0]):
        out[k], _ok = loc._fit_params_np(patches[k], mle, max_iter, tol, damping)
    return out


def _assert_agrees(reference, jitted):
    # x0, y0 (columns 2, 3) to < 1e-3 px; sx, sy (columns 4, 5) to < 1e-3.
    for col, name in ((2, "x_patch"), (3, "y_patch"), (4, "sx"), (5, "sy")):
        delta = np.abs(reference[:, col] - jitted[:, col])
        worst = int(np.argmax(delta))
        assert delta.max() < 1e-3, (
            f"{name} differs by {delta.max():.3e} on patch {worst} "
            f"(numpy={reference[worst, col]:.6f}, jitted={jitted[worst, col]:.6f})"
        )


@pytest.mark.skipif(not loc.is_numba_available(), reason="numba not installed")
@pytest.mark.parametrize("mle", [False, True])
def test_jitted_core_matches_numpy(mle):
    patches, _truth = _synthetic_patches()
    reference = _reference_params(patches, mle)

    max_iter, tol, damping = loc._fit_defaults(mle)
    core = loc._fit_mle_core if mle else loc._fit_lsq_core
    jitted = np.empty_like(reference)
    for k in range(patches.shape[0]):
        amp, bg, x0, y0, sx, sy, ok = core(patches[k], max_iter, tol, damping)
        jitted[k] = (amp, bg, x0, y0, sx, sy)
        assert ok

    _assert_agrees(reference, jitted)


@pytest.mark.skipif(not loc.is_numba_available(), reason="numba not installed")
@pytest.mark.parametrize("mle", [False, True])
def test_serial_and_parallel_batches_agree(mle):
    """The prange driver must not reorder anything that changes a fit."""
    patches, _truth = _synthetic_patches()
    max_iter, tol, damping = loc._fit_defaults(mle)
    serial_fn = loc._fit_batch_mle if mle else loc._fit_batch_lsq
    par_fn = loc._fit_batch_mle_par if mle else loc._fit_batch_lsq_par

    serial, ok_serial = serial_fn(patches, max_iter, tol, damping)
    parallel, ok_par = par_fn(patches, max_iter, tol, damping)

    assert ok_serial.all() and ok_par.all()
    np.testing.assert_array_equal(serial, parallel)


@pytest.mark.skipif(not loc.is_numba_available(), reason="numba not installed")
def test_parallel_row_refit_matches_serial():
    """The GPU fallback re-fits scattered rows; prange must not change a value."""
    patches, _truth = _synthetic_patches(n_patches=300)
    rows = np.arange(1, 300, 3, dtype=np.int64)  # a non-contiguous subset
    max_iter, tol, damping = loc._fit_defaults(True)

    serial = np.zeros((300, 6))
    par = np.zeros((300, 6))
    ok_serial = np.zeros(300, dtype=np.bool_)
    ok_par = np.zeros(300, dtype=np.bool_)
    loc._refit_mle_rows(patches, rows, serial, ok_serial, max_iter, tol, damping)
    loc._refit_mle_rows_par(patches, rows, par, ok_par, max_iter, tol, damping)

    np.testing.assert_array_equal(serial, par)
    np.testing.assert_array_equal(ok_serial, ok_par)
    # Rows outside the selection must be untouched.
    untouched = np.setdiff1d(np.arange(300), rows)
    assert not par[untouched].any()


@pytest.mark.parametrize("mle", [False, True])
def test_batch_driver_recovers_ground_truth(mle):
    """Sanity floor: the fits must actually find the simulated spots."""
    patches, truth = _synthetic_patches(n_patches=200)
    params, ok = loc._fit_batch_cpu(patches, mle=mle)

    assert ok.all()
    for col in (2, 3):
        assert np.median(np.abs(params[:, col] - truth[:, col])) < 0.1
    for col in (4, 5):
        assert np.median(np.abs(params[:, col] - truth[:, col])) < 0.25


def test_localize_frame_matches_per_patch_fits():
    """Batched extraction + assembly must line up with a per-candidate fit."""
    rng = np.random.default_rng(7)
    frame = rng.uniform(90.0, 110.0, size=(64, 64))
    yy, xx = np.indices((64, 64), dtype=np.float64)
    ys = np.array([10, 20, 33, 47, 55], dtype=np.int32)
    xs = np.array([12, 41, 25, 30, 50], dtype=np.int32)
    for cy, cx in zip(ys, xs):
        frame += 400.0 * np.exp(-(((xx - cx - 0.3) ** 2) + ((yy - cy + 0.2) ** 2)) / (2 * 1.3 ** 2))

    # Out-of-bounds candidates must be dropped without shifting net_gradient.
    y = np.concatenate([ys, [1, 62]]).astype(np.int32)
    x = np.concatenate([xs, [62, 1]]).astype(np.int32)
    ng = np.arange(y.size, dtype=np.float32) * 10.0

    locs = loc.localize_frame(
        frame, y, x, BOX, frame_number=3, net_gradient=ng, fit_backend="fast",
        camera_offset_adu=100.0, camera_gain_adu_per_electron=2.0,
    )

    assert locs["x"].size == ys.size
    np.testing.assert_array_equal(locs["net_gradient"], ng[: ys.size])
    np.testing.assert_array_equal(locs["frame"], np.full(ys.size, 3, dtype=np.int32))
    for key, dtype in loc._empty_locs().items():
        assert locs[key].dtype == dtype.dtype

    r = BOX // 2
    for i, (cy, cx) in enumerate(zip(ys, xs)):
        patch = np.clip((frame[cy - r: cy + r + 1, cx - r: cx + r + 1] - 100.0) / 2.0, 0.0, None)
        fit = loc.fit_gaussian_2d(patch)
        assert abs((cx - r + fit["x_patch"]) - locs["x"][i]) < 1e-3
        assert abs((cy - r + fit["y_patch"]) - locs["y"][i]) < 1e-3
        assert abs(fit["photons"] - locs["photons"][i]) < 1e-2
        assert abs(fit["lpx"] - locs["lpx"][i]) < 1e-5


def test_localize_frame_backends_agree_on_shape():
    rng = np.random.default_rng(3)
    frame = rng.poisson(120.0, size=(48, 48)).astype(np.float32)
    y = np.array([10, 24, 38], dtype=np.int32)
    x = np.array([11, 25, 30], dtype=np.int32)
    for backend in ("fast", "mle", "gpu"):
        locs = loc.localize_frame(frame, y, x, BOX, frame_number=0, fit_backend=backend)
        assert locs["x"].size == 3
        assert np.isfinite(locs["x"]).all() and np.isfinite(locs["sx"]).all()


def test_empty_and_degenerate_inputs():
    frame = np.zeros((32, 32), dtype=np.float32)
    empty = np.empty((0,), dtype=np.int32)
    assert loc.localize_frame(frame, empty, empty, BOX, frame_number=0)["x"].size == 0
    # Even boxes and out-of-range candidates yield no rows rather than raising.
    assert loc.localize_frame(frame, np.array([16]), np.array([16]), 6, frame_number=0)["x"].size == 0
    assert loc.localize_frame(frame, np.array([0]), np.array([0]), BOX, frame_number=0)["x"].size == 0
    # A flat patch is degenerate but must still produce a finite row.
    locs = loc.localize_frame(frame, np.array([16]), np.array([16]), BOX, frame_number=0)
    assert locs["x"].size == 1 and np.isfinite(locs["x"][0])
