"""How many photons, and how well was it localized.

These two numbers are load-bearing well beyond the localization table. The
photon count sets the intensity filter and every brightness-based selection;
the localization precision is what the immobility test compares motion against,
so an under-reported precision makes stationary molecules look mobile.

Both were wrong, in ways that reinforced each other. The photon count summed
background-subtracted pixels clipped at zero, which counts half of the
background noise as signal - and an inflated N then divides into s/sqrt(N) to
make the precision look better still.

Every number in these tests is measured against simulated spots of known
brightness, so "correct" means "recovers the value that was put in".
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_PKG_DIR = Path(__file__).resolve().parents[1] / "napari_loc_track"
if "napari_loc_track" not in sys.modules:
    _pkg = types.ModuleType("napari_loc_track")
    _pkg.__path__ = [str(_PKG_DIR)]
    sys.modules["napari_loc_track"] = _pkg

from conftest import load_localize2d  # noqa: E402

localize = load_localize2d()

PIXEL_NM, BOX, OFFSET, GAIN = 161.0, 7, 100.0, 1.0
PSF_PX = 0.6


def _spot(n_photons, bg, rng, sigma_px=PSF_PX, size=21):
    """One Poisson-noisy Gaussian spot of known total brightness, in ADU."""
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    centre = size / 2 - 0.5
    psf = np.exp(-((yy - centre) ** 2 + (xx - centre) ** 2) / (2 * sigma_px ** 2))
    psf *= n_photons / psf.sum()
    image = rng.poisson(psf + bg) * GAIN + OFFSET
    return image.astype(np.float32), int(round(centre))


def _localized(n_photons, bg, n_spots=600, seed=0, backend="mle"):
    rng = np.random.default_rng(seed)
    photons, precision, positions = [], [], []
    for _ in range(n_spots):
        image, c = _spot(n_photons, bg, rng)
        out = localize.localize_frame(
            image, np.array([c]), np.array([c]), BOX, frame_number=0,
            fit_backend=backend, camera_offset_adu=OFFSET,
            camera_gain_adu_per_electron=GAIN)
        if out["x"].size == 0:
            continue
        photons.append(float(out["photons"][0]))
        precision.append(float(out["lpx"][0]))
        positions.append(float(out["x"][0]))
    return (np.array(photons), np.array(precision) * PIXEL_NM,
            np.array(positions) * PIXEL_NM)


# --- photon count -------------------------------------------------------------


@pytest.mark.parametrize("n_true,bg", [(150, 2.0), (300, 10.0), (300, 30.0), (1000, 30.0)])
def test_the_photon_count_recovers_the_brightness_that_was_put_in(n_true, bg):
    photons, _p, _x = _localized(n_true, bg)
    assert np.median(photons) == pytest.approx(n_true, rel=0.05)


def test_the_photon_count_does_not_grow_with_the_background():
    """The failure that was there before: summing clipped residuals counts half
    the background noise as signal, so a brighter background looked brighter."""
    dim, _p, _x = _localized(300, 2.0)
    bright, _p, _x = _localized(300, 30.0)
    assert np.median(bright) / np.median(dim) == pytest.approx(1.0, abs=0.06)


def test_summing_the_box_would_have_over_reported():
    """Pinning the size of the bias this replaced, at the parameters it hurt
    most: a dim molecule on a bright background."""
    rng = np.random.default_rng(3)
    n_true, bg = 150, 30.0
    box_sums = []
    for _ in range(400):
        image, c = _spot(n_true, bg, rng)
        patches = localize._extract_patches(image, np.array([c]), np.array([c]), BOX)
        localize._apply_camera_calibration(patches, OFFSET, GAIN)
        params, ok = localize._fit_batch_cpu(patches, mle=True)
        if not ok[0]:
            continue
        box_sums.append(float(np.clip(patches[0] - params[0, 1], 0.0, None).sum()))
    assert np.median(box_sums) > 1.4 * n_true      # was ~+59%


def test_the_analytic_integral_is_the_gaussian_it_fitted():
    assert localize.gaussian_photons(2.0, 1.5, 2.5) == pytest.approx(
        2.0 * np.pi * 2.0 * 1.5 * 2.5)


# --- localization precision ---------------------------------------------------


@pytest.mark.parametrize("n_true,bg", [(150, 2.0), (150, 30.0), (300, 10.0), (1000, 30.0)])
def test_the_reported_precision_matches_the_scatter_actually_achieved(n_true, bg):
    """It is a bound, so it may sit a little above the measured scatter - but
    not below it, and not by a factor of two."""
    _photons, precision, positions = _localized(n_true, bg)
    achieved = positions.std()
    reported = np.median(precision)
    assert reported >= achieved * 0.9
    assert reported <= achieved * 1.4


def test_the_precision_degrades_with_background_as_it_must():
    """s/sqrt(N) - what this used to report - gets *better* with a brighter
    background, because the inflated photon count divides into it."""
    _p, dim, _x = _localized(300, 2.0)
    _p, bright, _x = _localized(300, 30.0)
    assert np.median(bright) > np.median(dim) * 1.2


def test_precision_improves_as_the_square_root_of_brightness():
    _p, dim, _x = _localized(250, 2.0)
    _p, bright, _x = _localized(1000, 2.0)
    assert np.median(dim) / np.median(bright) == pytest.approx(2.0, rel=0.25)


def test_least_squares_is_reported_as_less_precise_than_maximum_likelihood():
    """Unweighted least squares pays a factor 16/9 in variance, so 4/3 in the
    precision; a fit that reported the same either way would be flattering one
    of them."""
    mle = localize.crlb_sigma(300.0, 10.0, PSF_PX, mle=True)
    lsq = localize.crlb_sigma(300.0, 10.0, PSF_PX, mle=False)
    assert lsq > mle


def test_pixelation_is_included():
    """A pixel of width a adds a^2/12 to the effective PSF width; leaving it out
    flatters an undersampled setup, which at 161 nm pixels is this one."""
    with_pixel = localize.crlb_sigma(1e9, 0.0, 1.0, mle=True)
    assert with_pixel ** 2 * 1e9 == pytest.approx(1.0 + 1.0 / 12.0, rel=1e-6)


def test_a_background_free_bright_spot_approaches_the_photon_limit():
    sigma = localize.crlb_sigma(10000.0, 0.0, 1.0, mle=True)
    assert sigma == pytest.approx(np.sqrt((1.0 + 1.0 / 12.0) / 10000.0), rel=1e-6)


# --- the gain, and what it converts to ----------------------------------------
#
# (ADU - offset) / gain yields photoELECTRONS, not photons - they differ by the
# quantum efficiency, which is neither known here nor needed. Electrons are the
# right quantity anyway: the Poisson statistics the fit and the Cramer-Rao bound
# both assume hold for the charge collected, not for the photons that arrived
# and were mostly not detected.


def test_the_gain_scales_the_photon_count_it_divides_by():
    rng = np.random.default_rng(11)
    n_true, bg = 400, 5.0
    counted = {}
    for gain in (1.0, 1.3, 2.0):
        totals = []
        for _ in range(300):
            image, c = _spot(n_true, bg, rng)
            # re-express the same frame on a camera with this gain
            scaled = ((image - OFFSET) * gain + OFFSET).astype(np.float32)
            out = localize.localize_frame(
                scaled, np.array([c]), np.array([c]), BOX, frame_number=0,
                fit_backend="mle", camera_offset_adu=OFFSET,
                camera_gain_adu_per_electron=gain)
            if out["x"].size:
                totals.append(float(out["photons"][0]))
        counted[gain] = np.median(totals)
    # dividing by the gain the frame was scaled by recovers the same count
    for gain, value in counted.items():
        assert value == pytest.approx(n_true, rel=0.06), gain


def test_getting_the_gain_wrong_scales_the_count_by_exactly_that_factor():
    """Which is why the default matters: a gain of 1 on a 1.3 ADU/e- camera
    reports 1.3x the electrons that were actually collected."""
    rng = np.random.default_rng(12)
    image, c = _spot(400, 5.0, rng)
    right = localize.localize_frame(
        image, np.array([c]), np.array([c]), BOX, frame_number=0,
        fit_backend="mle", camera_offset_adu=OFFSET, camera_gain_adu_per_electron=1.3)
    wrong = localize.localize_frame(
        image, np.array([c]), np.array([c]), BOX, frame_number=0,
        fit_backend="mle", camera_offset_adu=OFFSET, camera_gain_adu_per_electron=1.0)
    assert float(wrong["photons"][0]) / float(right["photons"][0]) == pytest.approx(1.3, rel=0.02)
