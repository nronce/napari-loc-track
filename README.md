# napari-loc-track

A napari plugin for 2D single-molecule localization microscopy: detect and
fit localizations directly from a raw image stack (or import localizations
from another program), filter them, link them into trajectories, and run
diffusion/distance/duration analysis - all inside napari, with no external
software required for the localization step.

## Features

- **Localize (2D)** - spot detection (local maxima + net gradient, numba-accelerated)
  and sub-pixel Gaussian fitting (least-squares, Poisson-MLE, or GPU via
  Gpufit if installed) directly on a loaded image stack. Live detection
  preview overlay, background-threaded detect/fit with progress bars.
- **Drift correction** - takes the sample drift recorded on the focus-lock
  white-light camera (`<name>_xy_drift.csv`, written by the acquisition) out of
  the localizations, frame by frame, using the stack's `_frame_times.csv` to
  place each frame in the record. Both files are picked up on loading. The
  drift is smoothed first (Gaussian-weighted local line fit, width adjustable)
  so the tracker's scatter is not added to every localization, a time-binned
  frame gets the mean drift of its raw frames, and the image can be shown
  drift-corrected too - for display only, fitting still reads the frames as
  recorded. The corrected display, and every render, span the field of view of
  all frames together, so the part of the sample that drifts into view is
  shown and rendered rather than cut at the first frame's edge. The record
  reaches the fluorescence camera through a 2x2 camera map, because the
  white-light camera is turned 0.8 degrees from the fluorescence camera. The
  map comes from the record itself, or, for records made before 2026-09-25,
  from that day's Argo-SIM v2 calibration: 81.28 nm white-light pixels at
  161.87 nm fluorescence pixels.
  The **Drift** tab also checks the record - correlation quality, jumps
  between samples, and an independent registration of the white-light
  snapshots saved with it, drawn over the record and viewable drift-removed -
  and can refine it with **RCC** (redundant cross-correlation of the
  localizations), which measures the drift the white-light correction left and
  adds it on top. When no Micro-Manager metadata records the frame interval,
  the frame clock sets the frame rate.
  **Growth correction.** A growing root stretches, several percent along its
  axis in twenty minutes, so a translation leaves micrometres of misplacement
  across the field. "Measure growth deformation" registers the white-light
  snapshots to the last one, patch by patch at cell-wall scale. It fits one map
  per snapshot: translation, a turn, stretch along and across the root axis,
  and a stretch rate that varies along the axis. The axis comes from the
  direction of the cell walls (from the strain where the walls have none). All
  snapshot pairs are solved together, so no error accumulates. Each patch's
  shift is weighted by direction: a patch holding only walls parallel to the
  root says where the tissue is across the root, not along it. It takes
  seconds on a CUDA GPU (CuPy) and about a minute on the CPU. The result is
  saved in `analysis/<stamp>_deformation/` with its intermediate stages, and
  found again on loading. "Show it..." (or "Show the snapshots...") opens the
  snapshots in a viewer of their own with the growth drawn on them, all in the
  first snapshot's geometry:
  - arrows of the growth since the first snapshot, ending on the tissue, and
    of its current rate;
  - the model itself, as a grid of lines along and across the root axis,
    carried with the tissue, and a map of the stretch so far;
  - every snapshot carried into the first snapshot's geometry at each stage
    of the measurement, on a "stage" slider, with the first snapshot to lay
    them on;
  - the measurement's patches and what each pass still found off at them;
  - a docked panel of the growth's numbers (stretch and elongation rate over
    time, the move of the field centre, the stretch along the root axis, the
    measurement's convergence) and of how it was measured.

  "Movie file..." writes a GIF. `docs/deformation_model.pdf` sketches the
  model and the correction. With "Correct for: Growth", every localization is
  carried to where its tissue was at the first frame (by default, the
  geometry the drift correction uses) or is in the last snapshot ("Geometry").
  The drift record adds the motion faster than the snapshots. Filtering,
  linking, rendering and the image display all work in that geometry. D,
  distances and the immobility test use local coordinates with the stretch
  taken back out, which keeps D from being scaled by the growth.
- **Sessions** - "Save session..." records every render in the viewer (images
  and movies, each with the settings and the selection it was made from),
  the time averages, the white-light overlay, the merged-molecule layers and
  the line profile, and how every layer looked. "Load session..." makes them
  all again.
  "Show the white light in the viewer" adds the white-light snapshots as a
  layer placed on the fluorescence image through the camera map, so the
  localizations and the reconstructions sit on them. Each frame of the movie
  shows its nearest snapshot, and the drift is taken out whenever the image is
  shown drift-corrected. A cropped acquisition is placed with its sensor ROI,
  which recFL records. Snapshots saved without a drift record (a calibration
  slide, say) are laid over too.
- **Filter localizations** - per-column histograms with draggable filter
  bounds, adjustable bin count and view range, plus a draggable box on the
  image itself for x/y filtering. "Set as default" keeps the filters, and
  the trajectory filters, for every table loaded afterwards, in this session
  and the next. Only the sides you moved are kept, and never x, y or frame.
  "Load filters from..." takes just the filters of another session or run.
- **Render (SMLM)** - super-resolved reconstruction from the localizations
  that currently pass the filters, in four modes: localization histogram,
  scatter (one dot per localization), Gaussian with a single user-set width,
  and Gaussian with each molecule drawn at its own fitted precision. Any of
  them can be weighted by photon count instead of counting each localization
  once. Renders a **movie** too, grouping a user-set number of raw camera
  frames into each super-resolved frame - as independent blocks, as a
  cumulative build-up, or as a sliding window. Saves as float32 data, as a
  light 8-bit display copy, or as an RGB **composite** blending the
  reconstruction with the localizations and the trajectories drawn over it.
  GPU-accelerated via CuPy when it is installed, numba-parallel otherwise;
  background-threaded, with progress and a working Cancel.
- **Save the view** (Save tab) - the visible image layers at the time point on
  the slider, at the finest resolution among them rather than the screen's.
  Each keeps its own contrast, gamma and colormap and is blended as napari
  blends it. A PNG or RGB TIFF gets the scale bar burned in. A channel TIFF
  keeps each layer's values with its LUT and display range, so ImageJ opens
  it as a composite in the same colours.
- **Images tab** - time-averaged images of the movie as displayed:
  drift-corrected, or in the final geometry of the growth correction. That is
  the diffraction-limited image of what stayed put. The white-light snapshots
  are averaged the same way and laid on the fluorescence. Also a **line
  profile**: every visible image layer's values along a drawn line, averaged
  over a width, with an optional Gaussian fit (FWHM), exportable as CSV.
- **Link** - trajectory linking via [trackpy](http://soft-matter.github.io/trackpy/),
  background-threaded with progress. Trajectories are kept in memory with the
  localizations they were linked from: going back to a filter setting that was
  linked before restores them, with their metrics and population fit, without
  linking again.
- **Merged immobile molecules** (Render tab) - besides rendering them, "Show as
  layers" puts the immobile localizations before merging and the molecules
  after as two point layers to compare. "Merged table..." lists the molecules,
  one row each. Plots show precision before and after, localizations per
  molecule, photons, and how long each was seen. A render made with merging on
  goes to its own layer (`..._merged`), beside the unmerged one.
- **Trajectory analysis** - diffusion coefficient (D) extraction from a
  linear MSD fit with an MSD-vs-lag validation plot, plus fit-free distance
  travelled and trajectory duration distributions. Trajectories can be
  colored by any of the three metrics (log-scale, several colormaps).
- **Export** - one click exports every plot, the filtered localizations,
  linked trajectories, per-track metrics, and a `metadata.json` describing
  every parameter used, into a timestamped `analysis/` folder next to your
  data. Works with or without trajectory linking.
- Auto-detects companion localization/trajectory CSVs sitting next to a
  loaded image or CSV (e.g. from a previous export).

## Requirements

- Python >= 3.9
- [napari](https://napari.org)
- numpy, pandas, matplotlib, scipy
- [trackpy](http://soft-matter.github.io/trackpy/)
- qtpy (with a Qt binding such as PyQt5/PySide2 - usually pulled in by napari)
- tifffile
- numba (strongly recommended - detection falls back to a much slower pure-Python
  path without it)

Optional, auto-detected if present:
- [Gpufit](https://github.com/gpufit/Gpufit) (`pygpufit`) for GPU-accelerated fitting
- [CuPy](https://cupy.dev) for GPU-accelerated rendering. Match the wheel to the
  CUDA version `nvidia-smi` reports, and install the toolkit headers with it -
  CuPy 13+ compiles its kernels at runtime and fails without them:

  ```bash
  pip install "cupy-cuda13x[ctk]"    # or cupy-cuda12x[ctk] for CUDA 12
  ```

  Rendering is quick without it - 5 million localizations onto a 16384x16384 px
  reconstruction takes about 1.7 s on the CPU - so this is a convenience, not a
  requirement. A frame too large for the free device memory, or a GPU that fails
  part way, falls back to the CPU on its own and says so in the log.

## Installation

1. Install [Git](https://git-scm.com/downloads) and either
   [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or another
   Python >= 3.9 environment manager, if you don't already have one.

2. Clone the repository:

   ```bash
   git clone https://github.com/nronce/napari-loc-track.git
   cd napari-loc-track
   ```

3. Create and activate an environment, then install the plugin (editable
   install, so you can pull updates without reinstalling):

   ```bash
   conda create -n napari-loc-track python=3.11 -y
   conda activate napari-loc-track
   pip install -e .
   ```

4. Launch napari and open the plugin from **Plugins -> Localization
   Tracking**:

   ```bash
   napari
   ```

### Troubleshooting: napari crashes / freezes as soon as you add any layer

On some machines (older CPUs without AVX-512, seen on an Intel Kaby Lake
system), a conda-forge NumPy build linked against a recent Intel MKL can
crash the whole process the moment any real linear algebra call happens
(including deep inside `napari`/`skimage` on layer creation) - it fails
silently up front and only crashes once you actually try to use napari.
If you hit this, switch that environment's BLAS backend to OpenBLAS:

```bash
conda install -n napari-loc-track -c conda-forge "blas=*=openblas" --force-reinstall
```

## Usage

1. **Load data**: browse to an image stack (and/or an existing localization
   CSV), set the pixel size, and click "Load data".
2. If you don't already have localizations, use **Localize (2D)** to detect
   and fit them directly from the loaded image.

   If the acquisition recorded the drift, the **Drift** tab reports what it
   found and plots it. The localizations (and, optionally, the image on screen)
   are corrected through the camera map, and the tab says which map was used.
   "Check against the snapshots" registers the saved white-light snapshots
   independently and draws them over the record; "Show the snapshots..." opens
   them drift-removed, where the sample should stand still. RCC then estimates
   whatever drift is left from the localizations themselves (segment length,
   render pixel, blur, search range and outlier threshold are adjustable, and
   it can use only the immobile population) and applies it on top, for as long
   as the white-light correction it refined is unchanged. Everything
   downstream uses the corrected positions, and exported tables carry
   `drift_x [nm]` / `drift_y [nm]` columns with what was subtracted, plus a
   `drift_per_frame.csv` splitting it into its white-light and RCC parts - a
   corrected table loaded again is recognised and never corrected twice.
3. **Filter localizations** to remove bad fits (sigma, intensity, uncertainty,
   etc., plus a draggable box on the image for spatial filtering).
4. **Render (SMLM)** a super-resolved image or movie from whatever passes the
   filters, and save it in one of three formats:

   - **Data** - float32 holding the render's own values (localization counts,
     or photons when weighted), never rescaled, so two renders stay
     quantitatively comparable. The default for a still image.
   - **Display** - an 8-bit contrast-stretched copy, a quarter of the size,
     stretched *once for the whole movie* so the brightness of a frame still
     means how much signal it holds instead of pulsing frame to frame. The
     default for a movie.
   - **Composite** - 8-bit RGB, blending the reconstruction (in its colormap)
     with the localizations and the trajectories drawn over it in colours you
     pick, and optionally every other visible layer (the raw stack included).
     In a movie each layer is grouped the same way as the reconstruction, so a
     trajectory shows up while it is actually being tracked, and the overlays
     share the reconstruction's grid exactly.

   A **scale bar** and a **time stamp** can be burned into the display and
   composite formats - never into the float32 data, which stays untouched. The
   bar defaults to a round 1/2/5 length covering about a seventh of the saved
   width and follows the field of view, the pixel size and the crop, or you can
   set it by hand. A resizable **crop box** limits the save to a region without
   re-rendering.

   Every format is written with the super-resolved pixel size in its ImageJ
   tags, a PNG preview, and a `<name>_metadata.json` recording every setting
   behind it, from the camera gain through the filter bounds to the render
   options. The same JSON is embedded in the TIFF itself, and can be loaded
   back with "Load settings from a previous analysis...".
5. Optionally **Link** trajectories and run **Trajectory analysis** (D,
   distance, duration).

   The Render tab sorts trajectories into three classes, each rendered into a
   layer of its own: **Immobile**, **Undetermined** and **Mobile**. By the
   static test, mobile means motion was detected, immobile means static *and*
   the test could have detected motion down to a chosen D (0.01 µm²/s by
   default) - and a trajectory static only because it was too short or too dim
   to show otherwise is undetermined, not folded into either. The panel says
   how many points that takes at the data's precision, and counts the classes.

   **Populations** (Track tab) fits an immobile population and one or two
   mobile ones to every trajectory at once - exact likelihoods with each
   localization's own precision, motion blur and frame gaps, the number of
   mobile populations chosen by BIC - and plots the fit against the observed
   step lengths. Every trajectory then has a probability of being immobile.
   Classified "by the population fit", the Render tab sorts at a chosen
   probability, or - soft sorting - weights every localization by its
   probability, so the immobile and mobile images add up to the whole and no
   trajectory is set aside. The probabilities are exported with the
   trajectory metrics.

   **Merge immobile trajectories** draws each immobile molecule certified by
   the static test as one localization at its combined precision, carrying
   the signal of all its localizations.
6. **Export** whenever you're ready - from either the Filter or Trajectory
   analysis tab, exports whatever you currently have.

## Development

Run the tests with:

```bash
pip install pytest
pytest tests/
```

The benchmarks report per-spot and per-render timings, so a performance
regression shows up as a number rather than as "it feels slow":

```bash
python benchmarks/bench_fit.py
python benchmarks/bench_render.py --locs 2000000 --field 512 --oversampling 10
```
