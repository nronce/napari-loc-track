"""The separable local-maxima kernel must match the brute-force scan exactly.

`_local_maxima_nb` used to take a (box, box) slice per pixel and call np.argmax.
The replacement is separable, and the subtle part is tie-breaking: np.argmax
returns the *first* maximum in row-major order, so a plateau pixel only counted
as a maximum when nothing earlier in the scan equalled it. `_reference_maxima`
below is the old kernel, transcribed, and is the contract the new one must meet.
"""
import numpy as np
import pytest

from conftest import load_localize2d

loc = load_localize2d()

requires_numba = pytest.mark.skipif(not loc.is_numba_available(), reason="numba not installed")


def _reference_maxima(frame, box):
    """The original O(H*W*box^2) argmax scan."""
    y_size, x_size = frame.shape
    maxima_map = np.zeros(frame.shape, np.uint8)
    box_half = int(box / 2)
    box_half_1 = box_half + 1
    for y in range(box_half, y_size - box_half_1):
        for x in range(box_half, x_size - box_half_1):
            local = frame[y - box_half: y + box_half + 1, x - box_half: x + box_half + 1]
            flat_max = int(np.argmax(local))
            if (flat_max // box == box_half) and (flat_max % box == box_half):
                maxima_map[y, x] = 1
    y, x = np.where(maxima_map)
    return y.astype(np.int32), x.astype(np.int32)


@requires_numba
@pytest.mark.parametrize("box", [3, 5, 7, 9])
def test_matches_reference_on_continuous_frame(box):
    frame = np.random.default_rng(1).uniform(0.0, 1000.0, size=(96, 80)).astype(np.float32)
    ry, rx = _reference_maxima(frame, box)
    ny, nx = loc._local_maxima_nb(frame, box)
    np.testing.assert_array_equal(ry, ny)
    np.testing.assert_array_equal(rx, nx)


@requires_numba
@pytest.mark.parametrize("box", [3, 5, 7])
def test_matches_reference_with_plateaus(box):
    """Quantised camera data is full of ties; the tie-breaking must survive."""
    frame = np.random.default_rng(2).integers(0, 6, size=(80, 96)).astype(np.float32)
    ry, rx = _reference_maxima(frame, box)
    ny, nx = loc._local_maxima_nb(frame, box)
    np.testing.assert_array_equal(ry, ny)
    np.testing.assert_array_equal(rx, nx)


@requires_numba
def test_matches_reference_on_spot_image():
    rng = np.random.default_rng(3)
    frame = rng.poisson(100.0, size=(128, 128)).astype(np.float32)
    yy, xx = np.indices(frame.shape, dtype=np.float32)
    for cy, cx in [(20, 30), (64, 64), (100, 40), (33, 99)]:
        frame += (800.0 * np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * 1.4 ** 2))).astype(np.float32)
    for box in (5, 7, 11):
        ry, rx = _reference_maxima(frame, box)
        ny, nx = loc._local_maxima_nb(frame, box)
        np.testing.assert_array_equal(ry, ny)
        np.testing.assert_array_equal(rx, nx)


@requires_numba
def test_constant_frame_has_no_maxima():
    frame = np.full((40, 40), 7.0, dtype=np.float32)
    ny, nx = loc._local_maxima_nb(frame, 7)
    assert ny.size == 0 and nx.size == 0


@requires_numba
@pytest.mark.parametrize("shape", [(6, 40), (40, 6), (7, 7), (2, 2)])
def test_frames_smaller_than_box_are_empty(shape):
    frame = np.random.default_rng(4).uniform(size=shape).astype(np.float32)
    ny, nx = loc._local_maxima_nb(frame, 7)
    assert ny.size == 0 and nx.size == 0


@requires_numba
def test_identify_in_frame_end_to_end():
    """The full detection path still agrees with the brute-force candidates."""
    rng = np.random.default_rng(5)
    frame = rng.poisson(100.0, size=(96, 96)).astype(np.float32)
    yy, xx = np.indices(frame.shape, dtype=np.float32)
    for cy, cx in [(20, 30), (60, 70), (44, 12)]:
        frame += (900.0 * np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * 1.3 ** 2))).astype(np.float32)

    box = 7
    y, x, ng = loc.identify_in_frame(frame, 500.0, box)
    ry, rx = _reference_maxima(frame, box)
    rng_vals = loc._net_gradient_nb(frame, ry, rx, box)
    keep = rng_vals > 500.0
    np.testing.assert_array_equal(y, ry[keep])
    np.testing.assert_array_equal(x, rx[keep])
    np.testing.assert_allclose(ng, rng_vals[keep], rtol=1e-6)


def test_identify_in_frame_finds_planted_spots():
    """Backend-agnostic: also covers the scipy path when numba is absent."""
    rng = np.random.default_rng(8)
    frame = rng.poisson(100.0, size=(96, 96)).astype(np.float32)
    yy, xx = np.indices(frame.shape, dtype=np.float32)
    planted = [(20, 30), (60, 70), (44, 12)]
    for cy, cx in planted:
        frame += (900.0 * np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * 1.3 ** 2))).astype(np.float32)

    y, x, ng = loc.identify_in_frame(frame, 2000.0, 7)
    assert y.dtype == np.int32 and x.dtype == np.int32 and ng.dtype == np.float32
    found = set(zip(y.tolist(), x.tolist()))
    for spot in planted:
        assert spot in found
    assert (ng > 2000.0).all()


@requires_numba
def test_numba_and_numpy_net_gradients_agree():
    """The fallback net gradient must have the same sign convention as the kernel."""
    rng = np.random.default_rng(9)
    frame = rng.poisson(120.0, size=(64, 64)).astype(np.float32)
    yy, xx = np.indices(frame.shape, dtype=np.float32)
    frame += (700.0 * np.exp(-(((xx - 31) ** 2) + ((yy - 22) ** 2)) / (2 * 1.4 ** 2))).astype(np.float32)
    y, x = loc._local_maxima_nb(frame, 7)
    nb = loc._net_gradient_nb(frame, y, x, 7)
    np_ng = loc._net_gradient_np(frame, y, x, 7)
    np.testing.assert_allclose(np_ng, nb, rtol=1e-4, atol=1e-3)


@requires_numba
def test_numba_and_numpy_kernels_agree_without_ties():
    """Without plateaus the numba and scipy paths must select the same pixels.

    They differ by design on ties: maximum_filter + equality flags every pixel on
    a plateau, while the argmax scan flags none of them.
    """
    frame = np.random.default_rng(6).uniform(0.0, 1000.0, size=(70, 90)).astype(np.float32)
    for box in (5, 7):
        ny, nx = loc._local_maxima_nb(frame, box)
        py, px = loc._local_maxima_np(frame, box)
        np.testing.assert_array_equal(ny, py)
        np.testing.assert_array_equal(nx, px)
