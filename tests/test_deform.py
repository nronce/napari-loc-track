"""Measuring a growing tissue's deformation from its white-light snapshots.

Snapshots are made here from one texture seen through known maps - a
translation, a small turn, and a stretch along the "root" that grows with time
and varies along the axis - and the measurement has to hand the maps back.
"""
import importlib
import math

import numpy as np
import pytest

conftest = importlib.import_module("conftest")
deform = conftest.load_deform()

SIZE = 768
CENTER = (SIZE / 2.0, SIZE / 2.0)
SCALE = SIZE / 2.0


def _texture(seed=0):
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    walls = ndimage.gaussian_filter(rng.normal(0, 1, (SIZE, SIZE)), 3.0)
    return (walls * 1000 + 5000).astype(np.float32)


def _true_forward(k, K):
    """Snapshot k -> last: the tissue grows along y (the axis) by up to 6 %,
    faster lower down, turns by up to 0.5 deg and moves 30 px along the axis."""
    frac = (K - k) / K                      # 1 for the first snapshot, 0 for the last
    strain = 0.06 * frac
    gradient = 0.02 * frac
    turn = math.radians(0.5) * frac

    def fwd(p):
        p = np.atleast_2d(p)
        x = (p[:, 0] - CENTER[0]) / SCALE
        y = (p[:, 1] - CENTER[1]) / SCALE
        # along y: stretch (1 + strain), plus a rate rising along y
        yy = y * (1 + strain) + gradient * y * y
        xx = x
        c, s = math.cos(turn), math.sin(turn)
        X = c * xx - s * yy
        Y = s * xx + c * yy
        return np.column_stack([X * SCALE + CENTER[0], Y * SCALE + CENTER[1] + 30 * frac])
    return fwd


class _Stack:
    def __init__(self, frames, t):
        self.frames = frames
        self.t_epoch = np.asarray(t, float)
        self.shape = (len(frames),) + frames[0].shape

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, k):
        return self.frames[k]


def _snapshots(K=6):
    from scipy import ndimage

    final = _texture()
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    pts = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    frames, maps = [], []
    for k in range(K + 1):
        f = _true_forward(k, K)
        q = f(pts)                            # where each pixel of snapshot k is in the last
        img = ndimage.map_coordinates(final, [q[:, 1], q[:, 0]], order=3, mode="reflect")
        frames.append(img.reshape(SIZE, SIZE).astype(np.float32))
        maps.append(f)
    return _Stack(frames, 1000.0 + 30.0 * np.arange(K + 1)), maps


@pytest.fixture(scope="module")
def measured():
    stack, maps = _snapshots()
    region = (160, SIZE - 160, 160, SIZE - 160)
    record = deform.measure_deformation(stack, region=region, model="growth", patch=128,
                                       step=64, lags=(1, 2, 4), threads=4)
    return record, maps, region


def test_every_snapshot_is_carried_onto_the_last(measured):
    record, maps, region = measured
    x0, x1, y0, y1 = region
    gx, gy = np.meshgrid(np.linspace(x0 + 40, x1 - 40, 9), np.linspace(y0 + 40, y1 - 40, 9))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    worst = 0.0
    for k, truth in enumerate(maps):
        # points of snapshot k that land in the region
        src = record.inverse[k](pts)
        err = np.hypot(*(record.forward[k](src) - truth(src)).T)
        worst = max(worst, float(err.max()))
    assert worst < 0.15                       # px: a few nm on the white-light camera


def test_the_axis_is_found_and_the_growth_read_off(measured):
    record, _maps, _region = measured
    assert abs(math.degrees(record.theta) - 90.0) < 3.0
    along, across = record.strain_along_axis(point=CENTER)
    # the first snapshot is 6 % shorter along the axis than the last, measured in
    # its own pixels - so the map stretches it by ~6 % at the centre
    assert along[0] == pytest.approx(0.06, abs=0.004)
    assert abs(across[0]) < 0.004
    assert along[-1] == pytest.approx(0.0, abs=1e-6)


def test_between_snapshots_the_map_is_interpolated_and_held_outside(measured):
    record, _maps, _region = measured
    p = np.array([[CENTER[0], CENTER[1]]])
    mid = record.apply(p, 1000.0 + 15.0)
    a, b = record.forward[0](p), record.forward[1](p)
    np.testing.assert_allclose(mid, (a + b) / 2, atol=1e-9)
    np.testing.assert_allclose(record.apply(p, 0.0), a, atol=1e-9)
    np.testing.assert_allclose(record.apply(p, 1e12), record.forward[-1](p), atol=1e-9)


def test_a_record_survives_a_file(tmp_path, measured):
    record, _maps, _region = measured
    back = deform.DeformationRecord.load(record.save(tmp_path / "deformation.json"))
    p = np.array([[300.0, 400.0], [500.0, 250.0]])
    np.testing.assert_allclose(back.apply(p, 1045.0), record.apply(p, 1045.0), atol=1e-6)
    np.testing.assert_allclose(back.jacobian(p, 1045.0), record.jacobian(p, 1045.0), atol=1e-6)


def test_a_quad_map_differentiates_itself():
    m = deform.QuadMap(np.random.default_rng(0).normal(0, 50, (6, 2)) + np.array(
        [[400, 300], [300, 0], [0, 300], [0, 0], [0, 0], [0, 0]]), (400.0, 300.0), 300.0)
    p = np.array([[350.0, 280.0]])
    h = 1e-4
    num = np.column_stack([(m(p + [h, 0]) - m(p - [h, 0]))[0] / (2 * h),
                           (m(p + [0, h]) - m(p - [0, h]))[0] / (2 * h)])
    np.testing.assert_allclose(m.jacobian(p)[0], num, rtol=1e-6, atol=1e-8)


def test_the_intermediate_steps_are_kept_beside_the_record(tmp_path, measured):
    record, _maps, _region = measured
    steps = record.steps
    n, passes = len(record.t), len(record.stats["passes"])
    assert steps["forward"].shape == (passes + 1, n, 6, 2)
    assert steps["measured_to_last"].shape == (passes, n - 1, len(steps["centres"]), 2)
    np.testing.assert_allclose(steps["forward"][-1], np.stack([m.coef for m in record.forward]))
    path = record.save(tmp_path / "deformation.json")
    assert deform.steps_path(path).is_file()
    back = deform.DeformationRecord.load(path)
    np.testing.assert_allclose(back.steps["correction"], steps["correction"])


def test_each_pass_leaves_less_to_correct(measured):
    record, _maps, _region = measured
    rms = [p["rms_correction_px"] for p in record.stats["passes"]]
    assert rms[-1] < 0.2 * rms[0]
    # what the first pass measured of the first snapshot is what it then corrected
    measured_first = record.steps["measured_to_last"][0][0]
    corrected_first = -record.steps["correction"][0][0]
    ok = np.isfinite(measured_first[:, 0])
    assert np.median(np.hypot(*(measured_first[ok] - corrected_first[ok]).T)) < 0.3


def test_the_cpu_and_the_gpu_measure_the_same():
    xp, device = deform.array_backend("auto")
    if device != "gpu":
        pytest.skip("no CUDA device")
    stack, _maps = _snapshots(K=3)
    region = (160, SIZE - 160, 160, SIZE - 160)
    kw = dict(region=region, patch=128, step=64, lags=(1, 2), threads=2)
    cpu = deform.measure_deformation(stack, backend="cpu", **kw)
    gpu = deform.measure_deformation(stack, backend="gpu", **kw)
    p = np.array([[300.0, 300.0], [450.0, 500.0]])
    for k in range(len(cpu.t)):
        np.testing.assert_allclose(gpu.forward[k](p), cpu.forward[k](p), atol=0.05)


def test_a_batched_registration_matches_the_single_one():
    rng = np.random.default_rng(3)
    from scipy import ndimage

    base = ndimage.gaussian_filter(rng.normal(0, 1, (300, 300)), 2.0).astype(np.float32)
    shifted = ndimage.shift(base, (0.37, -1.62), order=3, mode="reflect")
    starts = np.array([[60, 60], [60, 140], [140, 60], [140, 140]])
    engine = deform.PatchRegistration(96, np)
    d, q = engine.register(engine.bank(deform._cut(base, starts, 96)),
                           engine.bank(deform._cut(shifted, starts, 96)))
    for (r, c), (dx, dy) in zip(starts, d):
        ref = deform._register(base[r:r + 96, c:c + 96], shifted[r:r + 96, c:c + 96], 100)
        assert abs(dx - ref[0]) < 0.02 and abs(dy - ref[1]) < 0.02
    np.testing.assert_allclose(d.mean(axis=0), (-1.62, 0.37), atol=0.05)
    assert (q > 0.9).all()


def test_the_root_axis_is_read_off_the_walls():
    from scipy import ndimage

    yy, xx = np.mgrid[0:400, 0:400].astype(float)
    angle = math.radians(30.0)
    across = -xx * math.sin(angle) + yy * math.cos(angle)       # distance across the files
    along = xx * math.cos(angle) + yy * math.sin(angle)
    walls = (np.cos(2 * math.pi * across / 25.0) > 0.95) | (np.cos(2 * math.pi * along / 120.0) > 0.99)
    image = ndimage.gaussian_filter(walls.astype(float), 1.5)
    theta, coherence = deform.root_axis(deform.wall_filter(image))
    assert abs(math.degrees(theta) - 30.0) < 3.0
    assert coherence > deform.AXIS_COHERENCE


def test_a_patch_of_parallel_walls_fixes_only_the_shift_across_them():
    # walls along y: the gradient is along x, so the weight goes to x
    tensor = np.array([[10.0, 0.1, 0.0], [5.0, 5.0, 0.0]])
    W = deform._sqrt_weights(tensor)
    full = np.einsum("mij,mkj->mik", W, W)
    assert full[0, 0, 0] > 1.9 and full[0, 1, 1] < 0.05
    np.testing.assert_allclose(full[1], np.eye(2), atol=1e-9)
