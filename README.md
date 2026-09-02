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
- **Filter localizations** - per-column histograms with draggable filter
  bounds, adjustable bin count and view range, plus a draggable box on the
  image itself for x/y filtering.
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
- **Link** - trajectory linking via [trackpy](http://soft-matter.github.io/trackpy/),
  background-threaded with progress.
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
