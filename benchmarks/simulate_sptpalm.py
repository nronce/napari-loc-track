"""A synthetic sptPALM acquisition with known ground truth.

Half the molecules are immobile, half diffuse at a known D. Track lifetimes are
drawn to match the duration distribution of a real acquisition on this
microscope - a geometric on-time with a mode of about nine frames - because the
whole point is to test the analysis where it is actually used, on short tracks.

Everything the analysis is supposed to recover is written out beside the movie:
the true position of every emitter in every frame, its true photon count, the
localization precision the Cramer-Rao bound predicts for it, which population it
belongs to, and the true D. Nothing is hidden.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

OUT = Path("//10.118.210.182/nano-bio/nathan.ronceray/synthetic data")
STEM = "synthetic_sptPALM_D0p3_immobile_mix"

# --- acquisition ---------------------------------------------------------
SIZE = 256                 # pixels square
N_FRAMES = 2000
PIXEL_NM = 161.0           # measured, not derived from magnification
EXPOSURE_MS = 30.0
INTERVAL_MS = 31.3         # exposure + readout, as the real camera runs
DT = INTERVAL_MS / 1000.0

# --- camera --------------------------------------------------------------
OFFSET_ADU = 100.0
GAIN_ADU_PER_PHOTON = 1.0  # 1:1 so photon counts can be read straight off
READ_NOISE_E = 1.6         # sCMOS, per pixel per frame
BACKGROUND_PHOTONS = 20.0  # per pixel per frame, HiLo-like

# --- fluorophores --------------------------------------------------------
PSF_SIGMA_NM = 130.0       # what a 1.45 NA objective delivers in practice
PHOTON_MODE = 300.0        # per localization, per frame
PHOTON_SHAPE = 4.0         # gamma shape; mode = (shape-1)*scale
ACTIVE_PER_FRAME = 50      # sparse enough that linking is not the thing on trial

# --- populations ---------------------------------------------------------
D_MOBILE_UM2_S = 0.3
FRACTION_MOBILE = 0.5

# --- track lifetimes -----------------------------------------------------
# Geometric on-times: a constant per-frame bleaching probability, which is what
# produces the exponential tail seen in the real duration histogram.
MEAN_ON_FRAMES = 9.0
MIN_ON_FRAMES = 3
MAX_ON_FRAMES = 60


def crlb_nm(photons, bg_per_px, psf_nm, pixel_nm=PIXEL_NM):
    """Mortensen 2010, maximum-likelihood form. The precision to expect."""
    sa2 = psf_nm ** 2 + pixel_nm ** 2 / 12.0
    tau = 2 * np.pi * sa2 * bg_per_px / (photons * pixel_nm ** 2)
    return np.sqrt(sa2 / photons * (1 + 4 * tau + np.sqrt(2 * tau / (1 + 4 * tau))))


def build(seed=20260904):
    rng = np.random.default_rng(seed)
    psf_px = PSF_SIGMA_NM / PIXEL_NM
    step_px = np.sqrt(2 * D_MOBILE_UM2_S * 1e6 * DT) / PIXEL_NM   # per axis, per frame
    margin = 4

    # How many molecules to start so that ACTIVE_PER_FRAME are on at any moment.
    n_molecules = int(N_FRAMES * ACTIVE_PER_FRAME / MEAN_ON_FRAMES)
    print(f"{n_molecules} molecules, {step_px:.2f} px/frame RMS step for the mobile ones")

    stack = np.zeros((N_FRAMES, SIZE, SIZE), np.float32)
    rows = []
    truth = []
    p_bleach = 1.0 / MEAN_ON_FRAMES

    for pid in range(n_molecules):
        mobile = rng.random() < FRACTION_MOBILE
        on = int(np.clip(rng.geometric(p_bleach), MIN_ON_FRAMES, MAX_ON_FRAMES))
        start = int(rng.integers(0, N_FRAMES))
        on = min(on, N_FRAMES - start)
        if on < MIN_ON_FRAMES:
            continue

        x = rng.uniform(margin, SIZE - margin)
        y = rng.uniform(margin, SIZE - margin)
        # One brightness per molecule, then Poisson counting on top of it: the
        # spread this produces is what makes the per-spot precision vary, which
        # is the thing a single averaged sigma cannot represent.
        brightness = rng.gamma(PHOTON_SHAPE, PHOTON_MODE / (PHOTON_SHAPE - 1))

        for k in range(on):
            frame = start + k
            if mobile and k > 0:
                x += rng.normal(0, step_px)
                y += rng.normal(0, step_px)
            if not (margin <= x < SIZE - margin and margin <= y < SIZE - margin):
                break
            photons = float(rng.poisson(brightness))
            if photons <= 0:
                continue
            _draw(stack[frame], x, y, photons, psf_px)
            rows.append((frame, pid, x, y, photons))

        truth.append((pid, bool(mobile), D_MOBILE_UM2_S if mobile else 0.0, on, brightness))

    print(f"{len(rows)} localizations from {len(truth)} molecules")

    # Background, shot noise, read noise, offset - in that order, as a camera does.
    stack += BACKGROUND_PHOTONS
    stack = rng.poisson(stack).astype(np.float32)
    stack += rng.normal(0.0, READ_NOISE_E, stack.shape).astype(np.float32)
    stack = stack * GAIN_ADU_PER_PHOTON + OFFSET_ADU
    stack = np.clip(stack, 0, 65535).astype(np.uint16)

    locs = pd.DataFrame(rows, columns=["frame", "particle", "x_px", "y_px", "photons"])
    locs["x [nm]"] = locs["x_px"] * PIXEL_NM
    locs["y [nm]"] = locs["y_px"] * PIXEL_NM
    locs["expected_precision [nm]"] = crlb_nm(
        locs["photons"].to_numpy(), BACKGROUND_PHOTONS, PSF_SIGMA_NM)

    tracks = pd.DataFrame(truth, columns=[
        "particle", "is_mobile", "D_true_um2_per_s", "n_frames_on", "mean_photons"])
    tracks = tracks[tracks["particle"].isin(locs["particle"].unique())]
    counts = locs.groupby("particle").size().rename("n_localizations")
    tracks = tracks.merge(counts, on="particle")
    return stack, locs, tracks


def _draw(plane, x, y, photons, sigma_px, half=6):
    """Add one Gaussian spot, integrated over pixels rather than sampled at
    their centres - the difference matters at 0.8 px sigma.

    Pixel i covers [i - 0.5, i + 0.5), so its centre is at exactly i. That is
    the convention the fitter uses (its coordinate grid is np.mgrid over pixel
    indices), and integrating over [i, i + 1) instead - which is the natural
    thing to write - puts every molecule half a pixel from where the fit will
    report it. At 161 nm pixels that is an 80 nm systematic error, which is
    seven times the localization precision and would have made the ground truth
    useless while looking like a fault in the analysis.
    """
    from scipy.special import erf

    x0, y0 = int(np.round(x)), int(np.round(y))
    xs = np.arange(max(0, x0 - half), min(plane.shape[1], x0 + half + 1))
    ys = np.arange(max(0, y0 - half), min(plane.shape[0], y0 + half + 1))
    if xs.size == 0 or ys.size == 0:
        return
    root2s = np.sqrt(2.0) * sigma_px
    cx = 0.5 * (erf((xs + 0.5 - x) / root2s) - erf((xs - 0.5 - x) / root2s))
    cy = 0.5 * (erf((ys + 0.5 - y) / root2s) - erf((ys - 0.5 - y) / root2s))
    plane[np.ix_(ys, xs)] += photons * np.outer(cy, cx)


def write_micromanager_sidecar(path, n_frames):
    """A minimal Micro-Manager sidecar, so the plugin's autofill has something
    to read and that path gets exercised too."""
    start = datetime(2026, 9, 4, 10, 0, 0)
    doc = {
        "Summary": {
            "Frames": n_frames, "Width": SIZE, "Height": SIZE,
            "Interval_ms": 0.0,                       # free-running, as the real one is
            "PixelSize_um": 0.0,                      # uncalibrated, as the real one is
            "StartTime": start.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " +0200",
            "InitialScopeData": {"Core-Camera": "Camera-1"},
        },
    }
    for i in range(n_frames):
        doc[f"FrameKey-{i}-0-0"] = {
            "FrameIndex": i,
            "ElapsedTime-ms": round(i * INTERVAL_MS, 3),
            "Exposure-ms": EXPOSURE_MS,
            "Camera-1-Offset": str(int(OFFSET_ADU)),
            "Camera-1-ChipName": "SYNTHETIC",
            "Camera-1-ActualInterval-ms": round(INTERVAL_MS, 4),
            "BitDepth": 16,
            "ROI": f"0-0-{SIZE}-{SIZE}",
            "PixelSizeUm": 0.0,
        }
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stack, locs, tracks = build()

    tif = OUT / f"{STEM}_MMStack_Pos0.ome.tif"
    print(f"writing {tif.name} ({stack.nbytes / 1e6:.0f} MB)...")
    tifffile.imwrite(tif, stack, photometric="minisblack")
    write_micromanager_sidecar(OUT / f"{STEM}_MMStack_Pos0_metadata.txt", N_FRAMES)

    locs.to_csv(OUT / f"{STEM}_ground_truth_localizations.csv", index=False)
    tracks.to_csv(OUT / f"{STEM}_ground_truth_tracks.csv", index=False)

    mobile = tracks["is_mobile"]
    durations = tracks["n_frames_on"] * DT
    readme = f"""SYNTHETIC sptPALM DATASET - ground truth included
{'=' * 62}
Written {datetime.now().strftime('%Y-%m-%d %H:%M')} by napari-loc-track.

WHAT IT IS
  A live-cell-like acquisition with two populations mixed together, built to
  test whether the analysis can tell them apart on short trajectories.

    immobile   {(~mobile).sum():>6} molecules   D = 0
    mobile     {mobile.sum():>6} molecules   D = {D_MOBILE_UM2_S} um2/s

  Track lifetimes follow a constant per-frame bleaching probability, giving a
  mode near {MEAN_ON_FRAMES:.0f} frames and an exponential tail - the shape a real
  duration histogram has. Median {np.median(durations):.2f} s, 90th pct {np.percentile(durations, 90):.2f} s.

FILES
  {STEM}_MMStack_Pos0.ome.tif
      {N_FRAMES} frames, {SIZE}x{SIZE}, uint16. Load this.
  {STEM}_MMStack_Pos0_metadata.txt
      Micro-Manager sidecar. The plugin reads the frame interval and the
      camera offset from it on load; pixel size is deliberately absent
      (recorded as 0.0), exactly as on the real microscope.
  {STEM}_ground_truth_localizations.csv
      Every emitter in every frame: frame, particle, x_px, y_px, photons,
      x [nm], y [nm], expected_precision [nm].
  {STEM}_ground_truth_tracks.csv
      Per molecule: is_mobile, D_true_um2_per_s, n_frames_on,
      mean_photons, n_localizations.

ACQUISITION
  pixel size            {PIXEL_NM} nm      (NOT in the metadata - set it by hand)
  exposure              {EXPOSURE_MS} ms
  frame interval        {INTERVAL_MS} ms   ({1000 / INTERVAL_MS:.2f} fps)
  frames                {N_FRAMES}

CAMERA
  offset                {OFFSET_ADU:.0f} ADU     (in the metadata; should autofill)
  gain                  {GAIN_ADU_PER_PHOTON} ADU/photon   <- set this, the default is right
  read noise            {READ_NOISE_E} e- rms
  background            {BACKGROUND_PHOTONS:.0f} photons/pixel/frame

FLUOROPHORES
  PSF sigma             {PSF_SIGMA_NM} nm  ({PSF_SIGMA_NM / PIXEL_NM:.2f} px)
  photons/localization  gamma, mode {PHOTON_MODE:.0f}, median {np.median(locs['photons']):.0f},
                        10th-90th pct {np.percentile(locs['photons'], 10):.0f}-{np.percentile(locs['photons'], 90):.0f}
  active per frame      ~{ACTIVE_PER_FRAME}

WHAT TO CHECK
  1. Photon count. The median fitted photon count should land near
     {np.median(locs['photons']):.0f}. If it comes back high, the photometry is counting
     background as signal.

  2. Localization precision. The Cramer-Rao bound for these spots is a median
     of {np.median(locs['expected_precision [nm]']):.1f} nm (10th-90th pct
     {np.percentile(locs['expected_precision [nm]'], 10):.1f}-{np.percentile(locs['expected_precision [nm]'], 90):.1f} nm).
     A reported uncertainty well below that is an under-estimate, and it will
     make immobile molecules look mobile.

  3. The immobility test. Over the immobile half the motion ratio should read
     1.00. If it reads higher, the calibration factor is what corrects it -
     and the amount it needs is a direct measure of how far the reported
     precision is off.

  4. The D distribution. The mobile population sits at {D_MOBILE_UM2_S} um2/s. Anything
     the MSD fit reports below about 0.01 um2/s on these short tracks is the
     immobile half leaking through, not a slow diffusive state.

  5. Diffusion. RMS step for the mobile population is
     {np.sqrt(4 * D_MOBILE_UM2_S * 1e6 * DT):.0f} nm per frame; a search range of 400-600 nm links them.
"""
    (OUT / f"{STEM}_README.txt").write_text(readme, encoding="utf-8")
    print(readme)


if __name__ == "__main__":
    main()
