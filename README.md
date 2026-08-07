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
  preview overlay, per-localization background level and background-noise
  standard deviation (`offset [photon]` and `bkgstd [photon]`), and
  background-threaded detect/fit with progress bars.
- **Filter localizations** - per-column histograms with draggable filter
  bounds, adjustable bin count and view range, plus a draggable box on the
  image itself for x/y filtering.
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
4. Optionally **Link** trajectories and run **Trajectory analysis** (D,
   distance, duration).
5. **Export** whenever you're ready - from either the Filter or Trajectory
   analysis tab, exports whatever you currently have.

## Development

Run the tests with:

```bash
pip install pytest
pytest tests/
```
