import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import napari
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt, QAbstractTableModel, QModelIndex, QTimer
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QComboBox,
    QCheckBox,
    QPlainTextEdit,
    QScrollArea,
    QDoubleSpinBox,
    QAbstractSpinBox,
    QSpinBox,
    QTableView,
    QTabWidget,
    QToolButton,
    QProgressBar,
    QDialog,
    QMessageBox,
)

import matplotlib
matplotlib.use("qtagg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
from napari.utils.colormaps import Colormap as NapariColormap
import trackpy as tp

from ._acqmeta import read_acquisition_metadata
from ._imageio import bin_frames, open_image_stack
from . import _deform as deform_io
from . import _drift as drift_io
from . import _populations as populations_io
from . import _deform_movie as deform_movie
from . import _view_export as view_export
from . import _averages as averages_io
from . import _profile as profile_io
from . import _growth_view as growth_view
from . import _session as session_io
from ._tracks import (
    DEFAULT_LINKING_ERROR_RATE,
    MERGED_COUNT_COLUMN,
    filter_tracks_by_length,
    iter_particle_batches,
    max_linkable_diffusion,
    merge_trajectories,
    rms_step,
)
from ._localize2d import (
    identify_in_frame,
    localize_frame,
    concatenate_localizations,
    is_gpufit_available,
    is_numba_available,
    warmup_fit_kernels,
)
from . import _render as smlm_render

# --- palette ---------------------------------------------------------------
# One set of colours for the whole plugin: the Qt stylesheet, every matplotlib
# figure and the track overlays all read from here, so nothing can drift out of
# step the way the figures had (some dark-themed, some on matplotlib's white
# default, which showed as white boxes inside a dark napari).
ACCENT = "#20b2aa"          # lightseagreen - the primary action on each tab
ACCENT_HOVER = "#2ac9c0"
ACCENT_PRESSED = "#178f88"
LAVENDER = "#b7a9e3"        # the secondary accent: selections, ranges, links
LAVENDER_HOVER = "#c8bdea"
LAVENDER_PRESSED = "#9a88d6"
AMBER = "#e8a33d"
# The root axis in the growth drawings: along it red, across it blue, as in the
# model's sketch (docs/deformation_model.pdf).
AXIS_ALONG = "#e05a4f"
AXIS_ACROSS = "#4f9ee0"           # reserved for "this stops something" and warnings
PANEL_BG = "#20242b"        # matches napari's dark theme panels
# Plots are screenshotted straight into talks, where anything short of pure
# black shows up as a grey rectangle on a black slide. The panel around them
# keeps napari's own shade; only the figures go fully black.
PLOT_BG = "#000000"
PANEL_LINE = "#39414d"
INK = "#d7dbe0"             # body text on a dark panel
INK_DIM = "#8d97a5"
INK_ON_ACCENT = "#0e1116"   # dark text, for sitting on top of the accent

# Deliberately NOT from the palette above: these colour trajectories drawn on
# the image, where the job is telling neighbouring tracks apart, not matching
# the interface.
TRACK_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
DEFAULT_D_COLORMAP = "coolwarm"
D_COLORMAP_CHOICES = ["coolwarm", "cool", "spring", "autumn", "bwr", "viridis"]

# What the plots are on screen. This is a reading size for a side panel and
# nothing else: the size a figure is *saved* at is chosen in the export dialog,
# against a live preview, and never touches what is on screen.
DEFAULT_PLOT_HEIGHT = 260

# Every plot in the plugin answers to one size, because these end up in talks
# and a figure that is the right shape is most of what makes one look
# deliberate. Width 0 means "fill the panel", which is the behaviour these had
# before and stays the default; anything else pins it, so a screenshot has the
# aspect ratio you chose rather than the one the window happened to have.
PLOT_WIDTH_FILL = 0
# Twice the screen resolution: a histogram that reads fine in a side panel is a
# blurred rectangle once a projector has stretched it across a wall.
FIGURE_SAVE_DPI = 200
PLOT_SIZE_LIMITS = (240, 3000, 120, 1600)   # min/max width, min/max height
# Shapes offered, as width/height. Setting a ratio and a height beats setting
# two pixel counts: the shape is what makes a set of figures look like a set,
# and it is the thing you actually have an opinion about.
PLOT_ASPECTS = (
    ("Fill the panel", None),
    ("16:9", 16 / 9), ("3:2", 3 / 2), ("4:3", 4 / 3), ("1:1", 1.0),
    ("5:2 (wide)", 5 / 2), ("2:3 (tall)", 2 / 3),
)
# Point size of the body text in every plot. Ticks sit one point below it, axis
# titles one above, a legend two below - the relationships the plots were drawn
# with, kept so that changing one number rescales the lot without any of them
# colliding.
_PLOT_FONT_PT = 8.0


def set_plot_font_size(points):
    global _PLOT_FONT_PT
    _PLOT_FONT_PT = float(points)


def plot_font(delta=0.0):
    """A point size relative to the current plot font."""
    return max(_PLOT_FONT_PT + delta, 3.0)

FILTER_HIST_BG = PANEL_BG
FILTER_HIST_BAR = ACCENT
FILTER_HIST_FG = INK
FILTER_HIST_LINE = LAVENDER

# Columns matched to these column_map keys get shown first, in this order;
# everything else follows in its original column order.
FILTER_PRIORITY_KEYS = ["sigma", "intensity", "uncertainty", "offset"]
# PSF widths for a typical single-molecule image sit around 100-200 nm, and a fit
# that ran away can report the whole fitting box. Defaulting the sigma filters and
# their histogram axes to this range keeps the useful part of the distribution
# readable instead of squashing it against a long tail.
SIGMA_DEFAULT_BOUNDS_NM = (0.0, 500.0)

POINTS_LAYER_NAME = "localizations"
TRACKS_LAYER_NAME = "tracks"
ALL_TRACKS_LAYER_NAME = "tracks_all"
ROI_LAYER_NAME = "xy_filter_roi"
LOC2D_CANDIDATES_LAYER_NAME = "loc2d_candidates"
RENDER_LAYER_NAME = "smlm_render"
RENDER_MOVIE_LAYER_NAME = "smlm_render_movie"
RENDER_CROP_LAYER_NAME = "smlm_render_crop"

# Layers this plugin produces itself. They are Image layers like the raw stack,
# so without this list a render would be offered as the thing to localize in, or
# as the field of view for the next render.
DERIVED_IMAGE_LAYERS = (RENDER_LAYER_NAME, RENDER_MOVIE_LAYER_NAME)

# napari draws its own scale bar from world coordinates, so the world has to be a
# physical space rather than a grid of camera pixels: every layer carries the
# pixel size as its scale and nanometres as its unit. This is a *display*
# transform only - layer data stays in camera pixels, which is what detection,
# linking and the render grid all work in - so nothing in the analysis sees it.
VIEWER_SPATIAL_UNIT = "nm"
# The leading axis of a stack is a frame index, not a length, and saying so keeps
# napari from labelling the dims slider in nanometres.
VIEWER_FRAME_UNIT = "pixel"
# These place themselves relative to the layer beneath them, through
# `layer_transform`, so they are already in world units and must not be rescaled
# from scratch - only carried along when the world itself changes.
DERIVED_SCALE_LAYERS = (RENDER_LAYER_NAME, RENDER_MOVIE_LAYER_NAME)

# Renders used to be one layer with one fixed name, replaced on every run.
# Several can now coexist - one per dynamics selection, which is the point of
# rendering the mobile and immobile populations separately - so they are
# recognised by a mark left in the layer's own metadata rather than by name.
RENDER_LAYER_TAG = "napari_loc_track_render"
# The white-light snapshots laid over the fluorescence. Placed by an affine of
# their own (the camera map), not by the pixel scale every other layer carries.
WL_OVERLAY_TAG = "napari_loc_track_white_light"
WL_OVERLAY_LAYER_NAME = "white light (registered)"


def is_render_layer(layer):
    """A layer this plugin rendered, whatever it ended up being called."""
    metadata = getattr(layer, "metadata", None) or {}
    if metadata.get(RENDER_LAYER_TAG):
        return True
    # Layers from before the tag existed, and sessions restored from them.
    return getattr(layer, "name", "") in DERIVED_IMAGE_LAYERS

RENDER_COLORMAPS = ["magma", "inferno", "viridis", "hot", "gray", "twilight"]
# What a saved render holds. Keys are stable identifiers stored in metadata.
RENDER_SAVE_FORMATS = {
    "data": "Data - float32, the render's own values",
    "display": "Display - 8-bit, contrast-stretched",
    "composite": "Composite - 8-bit RGB, layers blended",
}
# Refuse a render bigger than this rather than letting numpy raise MemoryError
# with a mountain of Qt state half-updated behind it. 8 GB is roughly a
# 45000x45000 single frame, or a 200-frame movie of 3300x3300.
RENDER_MAX_BYTES = 8 * 1024 ** 3
# Sliding windows re-render the localizations they share with their neighbours;
# past this overlap factor the render is mostly repeated work and says so.
RENDER_OVERLAP_WARN = 8

# Filenames checked next to a loaded CSV/image when auto-detecting companion
# files - "{stem}" is substituted with the source file's stem.
LOCS_FILENAME_PATTERNS = ["locs.csv", "{stem}_locs.csv", "{stem}-locs.csv", "{stem}.csv"]
TRAJ_FILENAME_PATTERNS = [
    "trajectories.csv", "{stem}_trajectories.csv", "{stem}_tracks.csv", "{stem}-tracks.csv",
]
LOCS_ANALYSIS_SUBPATH = ("data/localizations_filtered.csv", "data/localizations.csv")
TRAJ_ANALYSIS_SUBPATH = ("data/trajectories.csv",)

# Every run - a fit, an export - lands in its own dated folder under this one,
# beside the data it came from. Runs are never merged and never overwritten:
# two fits of the same stack with different thresholds are two results, and
# which came first is part of the answer.
ANALYSIS_ROOT = "analysis"
# Sortable by construction, so "newest" is a string comparison and the folder
# listing is already in order. Seconds, because re-fitting after changing one
# threshold takes less than a minute.
RUN_STAMP_FORMAT = "%Y-%m-%d_%H%M%S"
LOCS_RUN_FILENAME = "localizations.csv"

METRIC_LABELS = {
    "D": "Diffusion coefficient D (µm²/s)",
    "distance": "Distance travelled (µm)",
    "net": "End-to-end displacement (µm)",
    "straightness": "Straightness (end-to-end / path)",
    "duration": "Trajectory duration (s)",
    "motion": "Motion ratio (spread / localization error)",
    "pstatic": "p_static",
    "dmin": "Smallest detectable D (µm²/s)",
    # Colouring only: time needs no computing and has no bounds to filter on, so
    # it is absent from METRIC_CACHE_ATTR and from the histogram/bounds machinery.
    "time": "Frame first seen",
}
# Time-averaged images the Images tab made: the metadata key says which kind,
# so they can be placed again when the pixel size changes.
AVERAGE_TAG = "napari_loc_track_average"
PROFILE_LAYER_NAME = "line profile"
MERGED_BEFORE_LAYER_NAME = "immobile, before merging"
MERGED_AFTER_LAYER_NAME = "immobile, merged"
# A render made with merging on goes to a layer of its own, beside the unmerged one.
MERGED_RENDER_SUFFIX = "_merged"
# Kept with every render layer: the settings it was made with, so a session can
# make it again.
RENDER_RECIPE_KEY = "napari_loc_track_render_recipe"
VIEW_SAVE_FILTERS = {
    "png": "PNG - RGB, as displayed, with the scale bar (*.png)",
    "rgb_tiff": "TIFF - RGB, as displayed, with the scale bar (*.tif)",
    "channels_tiff": "TIFF - one channel per layer, with its LUT and contrast, for ImageJ (*.tif)",
}

# Trajectory sets kept in memory for going back to (see `_remember_tracks`):
# each is about as large as the localizations it was linked from.
LINKED_MEMORY_SIZE = 3

METRIC_CACHE_ATTR = {
    "D": "_track_diffusion_cache",
    "distance": "_track_distance_cache",
    "net": "_track_net_cache",
    "straightness": "_track_straightness_cache",
    "duration": "_track_duration_cache",
    "motion": "_track_motion_cache",
    "pstatic": "_track_pstatic_cache",
    "dmin": "_track_dmin_cache",
}

# A p-value has no lower limit worth plotting: a trajectory that is obviously
# moving returns something like 1e-200, and a log axis running that far spends
# every decade but the last on nothing. Below this floor the answer is the same
# either way - it moved - so the value is clamped rather than plotted honestly.
P_STATIC_FLOOR = 1e-10

# Matplotlib renders a real subscript through mathtext, which Qt labels and
# combo entries cannot - so the axis gets p with a subscript and everything else
# gets the plain p_static that reads the same way in a plain-text control.
METRIC_AXIS_LABELS = {"pstatic": r"$p_\mathrm{static}$"}

# The closed form for the detection floor equates the *expected* statistic to
# the critical value, while "detected half the time" wants its median - and the
# statistic is right-skewed under motion, so the median sits below the mean and
# the closed form comes out optimistic. Measured against simulation it is low by
# a factor 0.815 +/- 0.05 across N = 3..100 and alpha = 0.05..0.01, so this
# constant corrects it. After correcting, the closed form lands within about 7%
# of the simulated floor for N >= 5, and 13% at N = 3.
D_FLOOR_MEDIAN_CORRECTION = 1.227
# Every metric that is computed per trajectory and can be histogrammed, coloured
# by and bounded. "time" is deliberately absent: it colours but has nothing to
# compute and nothing to filter on.
COMPUTED_METRICS = ("D", "distance", "net", "straightness", "duration",
                    "motion", "pstatic", "dmin")
# The metric view boxes mirror the bound boxes when "follow filter" is on, so
# they need at least the precision of the finest of those (D, at six) or a small
# bound is silently rounded to zero on the way across.
METRIC_VIEW_DECIMALS = 6
# Filter columns carry whatever units the data came in - photons, nanometres,
# micrometres - so the bounds need room for the small ones too.
FILTER_BOUND_DECIMALS = 6

# How closely a localization has to match a trajectory point, in camera pixels,
# to be recognised as the same one. The two are computed from the same numbers
# by the same division, so in a single session they agree exactly; this is a
# guard for trajectories read back from a CSV, where the only thing between them
# is a float round-trip. Four decimals of a pixel is well under a nanometre.
LOC_MATCH_DECIMALS = 4

# Counts the sensor reports per photoelectron. Not 1.0: a gain of 1 says the
# camera is photon-counting, which almost none are, and every photon count and
# every localization precision derived from one is scaled by whatever the real
# figure is. Read it off the camera's specification - this default is the sCMOS
# on the microscope this plugin was written for.
DEFAULT_GAIN_ADU_PER_ELECTRON = 1.3

# The fluorescence camera's pixel in the sample plane, measured on 2026-09-25 on
# an Argo-SIM v2 slide (its 5 um grid of rings; the slide's concentric circles
# agree to 0.1 %). It cannot be derived from the metadata on this microscope; an
# acquisition that records it (recFL, from 2026-09-25 on) fills it in on
# loading. Tables localized before then were made at 161.0 nm/px.
DEFAULT_PIXEL_SIZE_NM = 161.87
# Width of the Gaussian window the drift record is smoothed over. The record is
# sampled ten times a second; a second of smoothing averages ~25 samples, which
# takes the tracker's scatter well below any localization precision while
# following a sample that drifts over minutes exactly.
DEFAULT_DRIFT_SMOOTHING_S = 1.0
# Typing a pixel size or a smoothing width re-corrects every localization, so
# wait for the typing to stop, as the time-binning box does.
DRIFT_DEBOUNCE_MS = 400
# A jump between two drift samples a tenth of a second apart bigger than this,
# in white-light pixels, is reported: the sample does not drift that fast.
DEFAULT_DRIFT_STEP_PX = 0.5
# Where each part of the correction a table carries came from, in words.
DRIFT_ORIGIN_LABELS = {
    "record": "the white-light record",
    "growth": "the growth deformation (to the last snapshot's geometry)",
    "table": "the drift columns the table came with",
    "rcc": "the RCC refinement",
}

# RCC. Ten or so segments is the usual compromise: fewer and the drift is
# sampled too coarsely in time, more and each segment holds too few
# localizations to correlate. The render pixel and blur sit near the
# localization precision, which is the scale of structure the correlation can
# use. The search is kept small because RCC here refines a drift that the
# white-light record has already taken out; alone, it wants the whole drift.
RCC_SOURCES = {
    "filtered": "Every localization that passes the filters",
    "shown": "The localizations on screen (dynamics filter included)",
}
DEFAULT_RCC_SEGMENT_FRAMES = 500
DEFAULT_RCC_PIXEL_NM = 40.0
DEFAULT_RCC_BLUR_NM = 40.0
DEFAULT_RCC_MAX_SHIFT_NM = 500.0
DEFAULT_RCC_RMAX_NM = 30.0

# Merging immobile trajectories. A trajectory is merged only if the static test
# could have caught motion down to this D: 0.01 µm²/s moves ~20 nm per 10 ms and
# ~200 nm over a second, which is the scale a merged point at a few nanometres
# would otherwise misrepresent. At 30 nm precision and 30 ms frames it takes
# eight points to reach it.
DEFAULT_MERGE_MAX_DETECTABLE_D = 0.01
DEFAULT_MERGE_MIN_POINTS = 3

# How trajectories are sorted into immobile, undetermined and mobile. The static
# test decides on each trajectory alone, and a short one it can only leave
# undetermined; the population fit decides against the whole dataset, and gives
# every trajectory a probability instead.
CLASSIFY_METHODS = {
    "test": "The static test and its detection floor",
    "fit": "The population fit (probabilities)",
}
# A trajectory is put in a class by the population fit when it is at least this
# likely to belong there; below it on both sides it is undetermined.
DEFAULT_CLASS_PROBABILITY = 0.9
POPULATION_MOBILE_CHOICES = {
    "auto": "Let the fit decide (BIC)",
    "1": "One",
    "2": "Two (slow and fast)",
}
POPULATION_COLORS = (ACCENT, LAVENDER, AMBER)

# The settings that describe the instrument rather than a choice about the
# analysis. Restoring a previous run moves these along with everything else -
# correctly, since the loaded localizations were computed with them - but doing
# it silently means a corrected calibration can be reverted by opening a folder,
# and nothing on screen says so. Every one of these that a restore changes is
# named in the log.
INSTRUMENT_SETTINGS = (
    ("pixel_size_box", "Pixel size", "{:.2f} nm/px"),
    ("loc_gain_box", "Camera gain", "{:.3g} ADU/e⁻"),
    ("loc_offset_box", "Camera offset", "{:.0f} ADU"),
    ("fps_box", "Frame rate", "{:.3f} fps"),
    ("bin_factor_box", "Time binning", "{:.0f} raw frames"),
)


def bound_to_box_precision(value, decimals, upward):
    """A bound rounded *outwards* to what a spin box can hold.

    A default bound is derived from the data and has to include the data it
    came from. A six-decimal box turns a maximum of 49995.8477774829 into
    49995.847777, which is below the value it was computed from - so the filter
    built to keep everything drops the single most extreme localization in the
    column, and does it silently in every column at once.

    Rounding to the box's precision and then stepping one unit outwards if that
    went the wrong way is exact in both directions, where scaling by a power of
    ten and flooring is not.
    """
    rounded = round(float(value), decimals)
    step = 10.0 ** -decimals
    if upward and rounded < value:
        return rounded + step
    if not upward and rounded > value:
        return rounded - step
    return rounded

# Only the pieces that need to differ from napari's own theme: the plugin sits
# inside napari's dock, so inheriting its background and text keeps it looking
# native, and the accent is spent on the few things worth pointing at.
STYLESHEET = f"""
QGroupBox {{
    border: 1px solid {PANEL_LINE};
    border-radius: 6px;
    margin-top: 10px;
    padding: 10px 6px 6px 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {ACCENT};
    font-weight: 600;
}}
QPushButton[primary="true"] {{
    background-color: {ACCENT};
    color: {INK_ON_ACCENT};
    border: none;
    border-radius: 4px;
    padding: 5px 14px;
    font-weight: 600;
}}
QPushButton[primary="true"]:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton[primary="true"]:pressed {{ background-color: {ACCENT_PRESSED}; }}
QPushButton[secondary="true"] {{
    background-color: transparent;
    color: {LAVENDER};
    border: 1px solid {LAVENDER_PRESSED};
    border-radius: 4px;
    padding: 5px 12px;
}}
QPushButton[secondary="true"]:hover {{
    background-color: {LAVENDER_PRESSED};
    color: {INK_ON_ACCENT};
}}
QPushButton[stop="true"] {{
    background-color: transparent;
    color: {AMBER};
    border: 1px solid {AMBER};
    border-radius: 4px;
    padding: 4px 10px;
}}
QPushButton[stop="true"]:disabled {{ color: {INK_DIM}; border-color: {PANEL_LINE}; }}
QPushButton[stop="true"]:hover:enabled {{ background-color: {AMBER}; color: {INK_ON_ACCENT}; }}
QPushButton:disabled[primary="true"] {{ background-color: {PANEL_LINE}; color: {INK_DIM}; }}
QProgressBar {{
    border: 1px solid {PANEL_LINE};
    border-radius: 4px;
    text-align: center;
    height: 14px;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 3px; }}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QLabel[role="heading"] {{ color: {ACCENT}; font-weight: 600; }}
QLabel[role="note"] {{ color: {INK_DIM}; }}
QCheckBox::indicator:checked {{ background-color: {ACCENT}; border-radius: 3px; }}
"""


def adaptive_steps(*boxes):
    """Make spin boxes step by a sensible fraction of their own value.

    A range control holding six decimals with Qt's default step of 1.0 is
    unusable on a value like 4e-5: one notch of the wheel moves it twenty
    thousand times its own size, and the digit that actually matters is
    unreachable. Qt's adaptive step chooses a power of ten from the current
    value instead - 1e-6 near 4e-5, 0.1 near 3, 100 near 1500 - so the wheel
    always moves the digit being looked at, whatever the scale of the number.

    These are exactly the controls that span decades: diffusion coefficients,
    distances, and any filter bound over a column whose units nobody chose.
    """
    for box in boxes:
        try:
            box.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        except Exception:
            pass  # a Qt too old for adaptive steps keeps its fixed one
    return boxes[0] if len(boxes) == 1 else boxes


def style_axes(figure, axes, *, title=None):
    """Give every plot in the plugin the same dark, low-contrast look.

    Called from all four figure families; before this the filter histograms were
    themed and the detection-count and MSD plots were not, so half the plots
    showed as white rectangles inside a dark napari.
    """
    figure.patch.set_facecolor(PLOT_BG)
    # A colorbar brings an Axes of its own that the caller never sees, so it is
    # picked up from the figure rather than waited for: styling only what was
    # passed in leaves the colorbar's tick labels in matplotlib's near-black
    # default, invisible against a black background.
    passed = list(np.atleast_1d(axes).ravel())
    extra = [ax for ax in figure.axes if ax not in passed]
    for ax in extra:
        ax.tick_params(labelsize=plot_font(-1), colors=INK)
        for spine in ax.spines.values():
            spine.set_color(PANEL_LINE)
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_color(INK)
            label.set_fontsize(plot_font())

    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(PLOT_BG)
        # Light enough to read off a projected slide, where the dimmed grey that
        # suits a screen at arm's length disappears entirely.
        ax.tick_params(labelsize=plot_font(-1), colors=INK)
        ax.grid(color=PANEL_LINE, linestyle="-", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(PANEL_LINE)
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_color(INK)
            label.set_fontsize(plot_font())
        if title is not None:
            ax.set_title(title, fontsize=plot_font(1), color=INK)


_napari_colormap_cache = {}


def _get_napari_colormap(name):
    # napari's Tracks layer `colormap=` kwarg only accepts names from its own
    # registry (AVAILABLE_COLORMAPS), which doesn't include most matplotlib
    # diverging maps (coolwarm, bwr, ...). Build a napari Colormap from the
    # matplotlib one, once per name, and hand it to `colormaps_dict` instead,
    # which accepts an arbitrary Colormap object per property.
    if name not in _napari_colormap_cache:
        mpl_colors = matplotlib.colormaps[name](np.linspace(0, 1, 256))
        _napari_colormap_cache[name] = NapariColormap(mpl_colors, name=name)
    return _napari_colormap_cache[name]


def infer_column_map(columns):
    def pick(candidates):
        for candidate in candidates:
            if candidate in columns:
                return candidate
        return None

    return {
        "frame": pick(["frame", "Frame", "t", "T"]),
        "x": pick(["x [nm]", "x (nm)", "x_nm", "x", "X"]),
        "y": pick(["y [nm]", "y (nm)", "y_nm", "y", "Y"]),
        "sigma": pick(["sigma [nm]", "sigma", "sigma_nm"]),
        "intensity": pick(["intensity [photon]", "intensity", "intensity [counts]"]),
        "offset": pick(["offset [photon]", "offset"]),
        "bkgstd": pick(["bkgstd [photon]", "bkgstd"]),
        "chi2": pick(["chi2", "chi-square"]),
        "uncertainty": pick(["uncertainty [nm]", "uncertainty"]),
    }


# Which entries of a metadata.json are settings that can be restored, and which
# widget each one belongs to. Everything not listed here - counts, timestamps,
# software versions, source paths - describes what a past run *produced* and is
# deliberately never applied.
SETTINGS_SPEC = (
    (("pixel_size_nm_per_px",), "pixel_size_box"),
    (("preprocessing", "time_bin_frames"), "bin_factor_box"),
    (("localization_2d", "gain_adu_per_electron"), "loc_gain_box"),
    (("localization_2d", "offset_adu"), "loc_offset_box"),
    (("localization_2d", "box_size_px"), "loc_box_size"),
    (("localization_2d", "min_net_gradient"), "loc_min_ng_box"),
    (("localization_2d", "fit_backend"), "loc_backend_box"),
    (("smlm_rendering", "oversampling"), "render_oversampling_box"),
    (("smlm_rendering", "mode"), "render_mode_box"),
    (("smlm_rendering", "global_sigma_nm"), "render_sigma_box"),
    (("smlm_rendering", "sigma_column"), "render_sigma_column_box"),
    (("smlm_rendering", "sigma_clamp_min_nm"), "render_sigma_min_box"),
    (("smlm_rendering", "sigma_clamp_max_nm"), "render_sigma_max_box"),
    (("smlm_rendering", "weight_by_photons"), "render_photons_box"),
    (("smlm_rendering", "colormap"), "render_colormap_box"),
    (("smlm_rendering", "use_gpu"), "render_gpu_box"),
    (("smlm_rendering", "frames_per_group"), "render_frames_per_box"),
    (("smlm_rendering", "grouping"), "render_grouping_box"),
    (("smlm_rendering", "window_step_frames"), "render_step_box"),
    (("smlm_rendering", "start_frame"), "render_start_frame_box"),
    (("smlm_rendering", "add_layer_to_viewer"), "render_add_layer_box"),
    (("smlm_rendering", "layer_name"), "render_layer_name_edit"),
    (("smlm_rendering", "population_split_p"), "render_population_p_box"),
    (("smlm_rendering", "immobile_definition", "max_detectable_d_um2_s"), "immobile_dmax_box"),
    (("smlm_rendering", "immobile_definition", "min_points"), "immobile_min_points_box"),
    (("smlm_rendering", "merge_immobile", "enabled"), "merge_box"),
    (("smlm_rendering", "classify_by"), "classify_method_box"),
    (("smlm_rendering", "class_probability"), "class_probability_box"),
    (("smlm_rendering", "soft_by_probability"), "class_soft_box"),
    (("population_fit", "mobile_populations"), "population_mobile_box"),
    (("smlm_rendering", "write_png_snapshot"), "render_png_box"),
    (("smlm_rendering", "image_save_format"), "render_image_format_box"),
    (("smlm_rendering", "movie_save_format"), "render_movie_format_box"),
    (("smlm_rendering", "movie_save_stride"), "movie_stride_box"),
    (("smlm_rendering", "rotate_degrees"), "render_rotate_box"),
    (("smlm_rendering", "composite", "reconstruction"), "render_composite_base_box"),
    (("smlm_rendering", "composite", "localizations"), "render_composite_locs_box"),
    (("smlm_rendering", "composite", "localization_color"), "render_locs_color_box"),
    (("smlm_rendering", "composite", "localization_size_nm"), "render_locs_size_box"),
    (("smlm_rendering", "composite", "trajectories"), "render_composite_tracks_box"),
    (("smlm_rendering", "composite", "trajectory_color"), "render_tracks_color_box"),
    (("smlm_rendering", "composite", "trajectory_width_nm"), "render_tracks_width_box"),
    (("smlm_rendering", "composite", "every_visible_layer"), "render_composite_all_box"),
    (("smlm_rendering", "timestamp", "enabled"), "render_timestamp_box"),
    (("smlm_rendering", "timestamp", "height_px"), "render_timestamp_size_box"),
    (("smlm_rendering", "timestamp", "color"), "render_timestamp_color_box"),
    (("smlm_rendering", "timestamp", "position"), "render_timestamp_position_box"),
    (("smlm_rendering", "scale_bar", "enabled"), "render_scalebar_box"),
    (("smlm_rendering", "scale_bar", "automatic"), "render_scalebar_auto_box"),
    (("smlm_rendering", "scale_bar", "length_nm"), "render_scalebar_length_box"),
    (("smlm_rendering", "scale_bar", "color"), "render_scalebar_color_box"),
    (("smlm_rendering", "scale_bar", "position"), "render_scalebar_position_box"),
    (("drift_correction", "enabled"), "drift_enable_box"),
    (("drift_correction", "shift_image"), "drift_shift_image_box"),
    (("drift_correction", "smoothing_s"), "drift_smoothing_box"),
    (("drift_correction", "mode"), "drift_mode_box"),
    (("drift_correction", "growth_reference"), "growth_reference_box"),
    (("drift_correction", "step_threshold_px"), "drift_step_box"),
    (("drift_correction", "rcc", "apply"), "rcc_apply_box"),
    (("drift_correction", "rcc", "source"), "rcc_source_box"),
    (("drift_correction", "rcc", "segment_frames"), "rcc_segment_box"),
    (("drift_correction", "rcc", "pixel_nm"), "rcc_pixel_box"),
    (("drift_correction", "rcc", "blur_nm"), "rcc_blur_box"),
    (("drift_correction", "rcc", "max_shift_nm"), "rcc_max_shift_box"),
    (("drift_correction", "rcc", "rmax_nm"), "rcc_rmax_box"),
    (("linking", "search_range_nm"), "search_box"),
    (("linking", "memory"), "memory_box"),
    (("linking", "min_track_length"), "min_traj_box"),
    (("diffusion", "max_lagtime_frames"), "max_lagtime_box"),
    (("diffusion", "min_track_length_for_d"), "d_min_length_box"),
    (("diffusion", "d_min"), "d_min_box"),
    (("diffusion", "d_max"), "d_max_box"),
    (("diffusion", "msd_validation_sample_count"), "msd_sample_box"),
    (("distance_bounds_um", "min"), "dist_min_box"),
    (("distance_bounds_um", "max"), "dist_max_box"),
    (("net_displacement_bounds_um", "min"), "net_min_box"),
    (("net_displacement_bounds_um", "max"), "net_max_box"),
    (("straightness_bounds", "min"), "straight_min_box"),
    (("straightness_bounds", "max"), "straight_max_box"),
    (("duration_bounds_s", "min"), "dur_min_box"),
    (("duration_bounds_s", "max"), "dur_max_box"),
    (("immobility", "fallback_precision_nm"), "immobility_sigma_box"),
    (("immobility", "precision_calibration"), "immobility_calibration_box"),
    (("motion_ratio_bounds", "min"), "motion_min_box"),
    (("motion_ratio_bounds", "max"), "motion_max_box"),
    (("p_static_bounds", "min"), "pstatic_min_box"),
    (("p_static_bounds", "max"), "pstatic_max_box"),
    (("dynamics_filter", "motion"), "motion_filter_box"),
    (("dynamics_filter", "pstatic"), "pstatic_filter_box"),
    (("dynamics_filter", "dmin"), "dmin_filter_box"),
    (("immobility", "significance"), "immobility_alpha_box"),
    (("detectable_d_bounds", "min"), "dmin_min_box"),
    (("detectable_d_bounds", "max"), "dmin_max_box"),
    (("dynamics_filter", "D"), "d_filter_box"),
    (("dynamics_filter", "distance"), "distance_filter_box"),
    (("dynamics_filter", "net"), "net_filter_box"),
    (("dynamics_filter", "straightness"), "straightness_filter_box"),
    (("dynamics_filter", "duration"), "duration_filter_box"),
    (("coloring", "enabled"), "color_trajectories_box"),
    (("coloring", "metric"), "color_metric_box"),
    (("coloring", "colormap"), "d_colormap_box"),
    (("display_layers", "show_localizations"), "show_points_box"),
    (("display_layers", "show_active_growing_tracks"), "show_tracks_box"),
    (("display_layers", "show_static_all_tracks"), "show_all_tracks_box"),
    (("rendering", "marker_size"), "marker_size_box"),
    (("rendering", "marker_edge_width"), "marker_edge_width_box"),
    (("rendering", "marker_symbol"), "marker_choice"),
    (("rendering", "active_track_line_width"), "line_width_box"),
    (("rendering", "static_track_line_width"), "all_tracks_line_width_box"),
    (("rendering", "persist_completed_tracks"), "persist_tracks_box"),
    (("rendering", "plot_aspect"), "plot_aspect_box"),
    (("rendering", "plot_height_px"), "plot_height_box"),
    (("rendering", "plot_font_pt"), "plot_font_box"),
)


# Which acquisition-metadata field fills which control when a stack is loaded,
# and how the change is worded in the log. Only fields the microscope genuinely
# recorded reach this table - `read_acquisition_metadata` omits the rest - so a
# control whose value was never calibrated keeps whatever it had.
ACQUISITION_AUTOFILL = (
    ("pixel_size_nm", "pixel_size_box", "Pixel size", "{:.2f} nm/px"),
    ("fps", "fps_box", "Frame rate", "{:.3f} fps"),
    ("camera_offset_adu", "loc_offset_box", "Camera offset", "{:.0f} ADU"),
    ("sensor_roi_x", "wl_roi_x_box", "Fluorescence ROI x", "{:.0f} px"),
    ("sensor_roi_y", "wl_roi_y_box", "Fluorescence ROI y", "{:.0f} px"),
)

# How each autofilled value has to change when raw frames are summed in groups
# of N. The microscope recorded single raw frames; the pipeline sees their sums,
# and every quantity that is per-frame rather than per-pixel moves with N. The
# camera baseline scales up because each raw frame brought its own, and the
# frame rate scales down because a binned frame spans N exposures. A value not
# listed here - the pixel size - is unaffected by binning in time.
ACQUISITION_BIN_EXPONENT = {"camera_offset_adu": 1, "fps": -1}

# Largest group a raw stack can be binned into. Well past anything useful; it is
# here so a typo cannot ask for a bin longer than any real movie.
TIME_BIN_MAX = 1000

# How long the time-binning box waits after the last keystroke before re-binning
# the loaded stack. Long enough that scrolling from 1 to 8 bins once, not eight
# times.
TIME_BIN_DEBOUNCE_MS = 400

# Read off the acquisition but deliberately not applied to anything: they are
# context for judging whether the values above belong to this run. The objective
# is the important one - it is the only clue to the pixel size when nobody
# calibrated it, and it cannot become one without the sensor pitch, which is not
# recorded anywhere in the file.
ACQUISITION_CONTEXT = (
    ("objective", "objective", "{}"),
    ("camera_chip", "camera", "{}"),
    ("exposure_ms", "exposure", "{:g} ms"),
    ("n_frames", "frames", "{:.0f}"),
)


# What napari does when the play button reaches the last frame. Keys are its own
# LoopMode values; the UI shows the second element.
PLAYBACK_MODES = {
    "loop": "Start over",
    "once": "Stop",
    "back_and_forth": "Play backwards",
}


# The built-in theme whose canvas is already pure black.
BLACK_CANVAS_THEME = "dark"


def canvas_is_black(theme_id):
    """True if that theme paints the canvas pure black."""
    try:
        from napari.utils.theme import get_theme

        return tuple(get_theme(str(theme_id)).canvas.as_rgb_tuple()[:3]) == (0, 0, 0)
    except Exception:
        return False


def apply_black_canvas(viewer):
    """Put the viewer on a theme with a pure black canvas, for screenshots.

    napari's own "dark" theme already paints the canvas black, so this switches
    to it rather than registering a theme of its own. That distinction matters
    more than it looks: `viewer.theme` is persisted to napari's *global*
    settings, so a made-up theme id ends up in a config file that plain napari -
    launched without this plugin, which is the only thing that registers it -
    cannot resolve. It then reports a validation error and resets the field on
    every start. Only built-in theme names are safe to put there.

    A viewer already on a black-canvas theme is left alone.
    """
    try:
        if not canvas_is_black(viewer.theme):
            viewer.theme = BLACK_CANVAS_THEME
    except Exception:
        pass


def _napari_playback_settings():
    """napari's own playback settings, or None on a build that has none.

    The play button belongs to napari, not to this plugin, and it reads its
    speed from here - so driving these settings is what makes the button run at
    the requested rate, rather than reimplementing playback.
    """
    try:
        from napari.settings import get_settings

        return get_settings().application
    except Exception:
        return None


def _playback_fps(default=10):
    """napari's playback rate, as the whole number of frames per second it is."""
    settings = _napari_playback_settings()
    try:
        return max(1, int(round(float(getattr(settings, "playback_fps", None)))))
    except (TypeError, ValueError):
        return default


def _playback_mode(default="loop"):
    settings = _napari_playback_settings()
    mode = getattr(settings, "playback_mode", None)
    mode = str(getattr(mode, "value", mode))
    return mode if mode in PLAYBACK_MODES else default


def _dig(mapping, path):
    """Follow a key path into nested dicts. Returns (found, value)."""
    node = mapping
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False, None
        node = node[key]
    return True, node


# --- the filters, as a set that can be kept as defaults or taken from elsewhere ---

# The metadata sections that hold the trajectory filters - bounds and on/off -
# alongside the per-column localization bounds in "filter_bounds".
FILTER_SECTIONS = ("dynamics_filter", "distance_bounds_um", "net_displacement_bounds_um",
                   "straightness_bounds", "duration_bounds_s", "motion_ratio_bounds",
                   "p_static_bounds", "detectable_d_bounds")
FILTER_SETTING_PATHS = (("diffusion", "d_min"), ("diffusion", "d_max"))
# Where a dataset lies, not a judgement about quality: never carried from one
# dataset to another. Every name `resolve_columns` accepts for them.
GEOMETRY_COLUMNS = frozenset({"frame", "Frame", "t", "T", "x [nm]", "x (nm)", "x_nm", "x",
                              "X", "y [nm]", "y (nm)", "y_nm", "y", "Y"})
FILTER_DEFAULTS_FILENAME = "filter_defaults.json"
FILTER_DEFAULTS_KIND = "napari-loc-track filter defaults"
# Both kinds of file that carry the settings of an analysis.
SETTINGS_FILE_FILTER = ("Analysis settings (metadata.json *.loctrack-session.json);;"
                        "JSON files (*.json)")


def user_config_dir():
    """Where the plugin keeps what a person chose to keep between sessions.

    NAPARI_LOC_TRACK_CONFIG_DIR moves it - the tests point it at a scratch
    folder so they never read or overwrite anyone's real defaults.
    """
    override = os.environ.get("NAPARI_LOC_TRACK_CONFIG_DIR")
    if override:
        return Path(override)
    base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "napari-loc-track"


def is_filter_setting(path):
    path = tuple(path)
    return path[0] in FILTER_SECTIONS or path in FILTER_SETTING_PATHS


def _put(mapping, path, value):
    node = mapping
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def filter_settings_of(metadata):
    """Only the filters of a metadata dict - a run's metadata.json, a session's
    settings or a defaults file - in the same layout, so `apply_settings` takes
    it as it takes any other. x, y and frame bounds are left out."""
    out = {}
    if not isinstance(metadata, dict):
        return out
    for path, _attr in SETTINGS_SPEC:
        if is_filter_setting(path):
            found, value = _dig(metadata, path)
            if found and value is not None:
                _put(out, path, value)
    if "distance_bounds_um" not in out and isinstance(metadata.get("distance_bounds_nm"), dict):
        out["distance_bounds_nm"] = metadata["distance_bounds_nm"]
    bounds = metadata.get("filter_bounds")
    if isinstance(bounds, dict):
        kept = {column: dict(limits) for column, limits in bounds.items()
                if column not in GEOMETRY_COLUMNS and isinstance(limits, dict)}
        if kept:
            out["filter_bounds"] = kept
    return out


def read_filter_defaults(path):
    """(settings, saved_at) from a defaults file; (None, None) when there is none.

    Raises ValueError for a file that is there but is not one.
    """
    path = Path(path)
    if not path.is_file():
        return None, None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not be read: {exc}") from exc
    if not isinstance(data, dict) or data.get("kind") != FILTER_DEFAULTS_KIND:
        raise ValueError("is not a filter defaults file")
    return filter_settings_of(data.get("settings") or {}), data.get("saved_at")


def write_filter_defaults(path, settings):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_at = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps({"kind": FILTER_DEFAULTS_KIND, "version": 1,
                                "saved_at": saved_at, "settings": settings}, indent=2),
                    encoding="utf-8")
    return saved_at


def settings_of_file(data):
    """The settings in a metadata.json or a session file, whichever this is."""
    if isinstance(data, dict) and session_io.SESSION_KEY in data:
        return data.get("settings") or {}
    return data


def settings_from_metadata(metadata):
    """Read the restorable parameters out of a metadata.json dict.

    Returns (values, notes): `values` maps widget attribute names to the value to
    apply, `notes` collects human-readable remarks about anything converted or
    ignored. Missing entries are simply absent from `values`, so a metadata file
    from an older version restores what it knows and leaves the rest alone.
    """
    values = {}
    notes = []
    if not isinstance(metadata, dict):
        return values, ["not a settings file"]

    for path, attr in SETTINGS_SPEC:
        found, value = _dig(metadata, path)
        if found and value is not None:
            values[attr] = value

    # Not a widget either: the class the population buttons pointed at.
    found, population = _dig(metadata, ("smlm_rendering", "population"))
    if found and population in ("immobile", "undetermined", "mobile"):
        values["_population_class"] = population

    # Not a widget: the frame shift is plugin state, restored by hand below.
    found, shift = _dig(metadata, ("frame_number_shift",))
    if found and isinstance(shift, (int, float)):
        values["_frame_shift"] = int(shift)
    else:
        found, legacy = _dig(metadata, ("frame_one_indexed",))
        if found:
            values["_frame_shift"] = -1 if legacy else 0
            notes.append("frame indexing tick box converted to a frame shift")

    # Distance was reported in nm before it was changed to µm; convert rather
    # than silently applying a value 1000x too large.
    if "dist_min_box" not in values and "dist_max_box" not in values:
        found, legacy = _dig(metadata, ("distance_bounds_nm",))
        if found and isinstance(legacy, dict):
            for key, attr in (("min", "dist_min_box"), ("max", "dist_max_box")):
                if isinstance(legacy.get(key), (int, float)):
                    values[attr] = float(legacy[key]) / 1000.0
            notes.append("converted distance bounds from nm to µm")

    # The gain used to be recorded as ADU per photon, which is what it was
    # called rather than what it was: the division has always produced
    # photoelectrons. Same number, so it restores unchanged.
    if "loc_gain_box" not in values:
        found, legacy = _dig(metadata, ("localization_2d", "gain_adu_per_photon"))
        if found and isinstance(legacy, (int, float)):
            values["loc_gain_box"] = float(legacy)
            notes.append("camera gain read from the older 'per photon' key")

    # Acquisition timing used to live under "diffusion" and now lives under
    # "linking"; read either, preferring the current location. Frame rate and
    # frame interval are the same setting, so the interval is only consulted
    # when no frame rate was recorded.
    for section in ("diffusion", "linking"):
        found, fps = _dig(metadata, (section, "fps"))
        if found and isinstance(fps, (int, float)) and fps > 0:
            values["fps_box"] = float(fps)
    if "fps_box" not in values:
        for section in ("diffusion", "linking"):
            found, interval_ms = _dig(metadata, (section, "frame_interval_ms"))
            if found and isinstance(interval_ms, (int, float)) and interval_ms > 0:
                values["fps_box"] = 1000.0 / float(interval_ms)
                notes.append("frame rate taken from the frame interval")

    # The drift record used to be scaled by a white-light pixel size typed in by
    # hand, with no turn between the cameras. There is nothing to restore: the
    # camera map replaces it, and the drift now taken out differs from that run's.
    found, legacy = _dig(metadata, ("drift_correction", "wl_pixel_size_nm"))
    if found and isinstance(legacy, (int, float)) and legacy > 0:
        notes.append(f"that run scaled its drift record by a white-light pixel of "
                     f"{float(legacy):g} nm with no turn between the cameras; the "
                     "calibrated camera map is used instead, so its drift differs")

    return values, notes


def widget_value(widget):
    """A Qt input's value as the plain JSON value `set_widget_value` takes back."""
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        data = widget.currentData()
        return data if isinstance(data, (str, int, float)) else widget.currentText()
    if isinstance(widget, QSpinBox):
        return int(widget.value())
    if isinstance(widget, QDoubleSpinBox):
        return float(widget.value())
    if isinstance(widget, QLineEdit):
        return widget.text()
    return None


def set_widget_value(widget, value):
    """Set a Qt input from a plain JSON value. Returns True if it took."""
    if isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
        return True
    if isinstance(widget, QComboBox):
        text = str(value)
        index = widget.findText(text)
        if index < 0:
            # Some combos show a sentence but record a short stable key (render
            # modes, movie groupings), so the key is matched too - otherwise
            # rewording a label would silently stop restoring that setting.
            index = widget.findData(text)
        if index < 0:
            return False  # a backend/colormap/column this build does not offer
        widget.setCurrentIndex(index)
        return True
    if isinstance(widget, QSpinBox):
        widget.setValue(int(round(float(value))))  # setValue clamps to the range
        return True
    if isinstance(widget, QDoubleSpinBox):
        widget.setValue(float(value))
        return True
    if isinstance(widget, QLineEdit):
        # Only settings, never paths: the ones restored through here are names
        # the user chose (the render layer's), and a path from another machine
        # would point at nothing.
        widget.setText("" if value is None else str(value))
        return True
    return False


def is_sigma_column(column):
    """True for any PSF-width column: sigma, sigma_x/sigma_y, sigma1/sigma2 [nm]."""
    return str(column).strip().lower().startswith("sigma")


def apply_numeric_filters(df, bounds):
    """Keep the rows inside every bound.

    Combines the per-column tests into one boolean mask and indexes once. The
    obvious loop - re-filtering the frame per column - copies the whole table
    once per bound, which on a million localizations with eight filters costs
    ~420 ms against ~20 ms here, on every keystroke in the Filter tab.
    """
    if df is None:
        return df
    mask = None
    for column, (lower, upper) in bounds.items():
        if column not in df.columns:
            continue
        values = df[column].to_numpy()
        for limit, test in ((lower, np.greater_equal), (upper, np.less_equal)):
            if limit is None:
                continue
            column_mask = test(values, limit)
            mask = column_mask if mask is None else (mask & column_mask)
    if mask is None:
        return df.copy()
    return df[mask]


def _apply_numeric_filters_reference(df, bounds):
    """Row-by-row equivalent of apply_numeric_filters, kept as the test oracle."""
    filtered = df.copy()
    for column, (lower, upper) in bounds.items():
        if not column or column not in filtered.columns:
            continue
        if lower is not None:
            filtered = filtered[filtered[column] >= lower]
        if upper is not None:
            filtered = filtered[filtered[column] <= upper]
    return filtered


class _Cancelled:
    """Sentinel returned by a worker that stopped because the user asked it to.

    Cancellation is cooperative: the widget sets a `threading.Event`, the worker
    notices it at the next iteration boundary and returns this instead of a
    result. Every `returned` handler checks for it before touching state.
    """

    __slots__ = ()

    def __repr__(self):
        return "CANCELLED"


CANCELLED = _Cancelled()


def _is_cancelled(cancel):
    return cancel is not None and cancel.is_set()


@thread_worker
def _load_worker(csv_path, image_path, bin_factor=1, cancel=None):
    # A single pd.read_csv cannot be interrupted part way, so cancellation is
    # checked around it; the image decode is chunked and checks continuously.
    if _is_cancelled(cancel):
        return CANCELLED
    df = pd.read_csv(csv_path) if csv_path else None
    image = None
    raw_image = None
    how = ""
    acquisition = None
    if image_path:
        if _is_cancelled(cancel):
            return CANCELLED
        t0 = time.perf_counter()
        image, how = open_image_stack(image_path, cancel=cancel)
        if image is None:
            return CANCELLED
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        # The raw stack is kept so the binning factor can be changed later
        # without re-reading the file. It is normally a memory map or a lazy
        # handle, so holding on to it costs nothing.
        raw_image = image
        if bin_factor > 1:
            image, binned_how = bin_frames(image, bin_factor, cancel=cancel)
            if image is None:
                return CANCELLED
            how = f"{how}, {binned_how}"
        how = f"{how} in {time.perf_counter() - t0:.2f} s"
        # Reading the acquisition parameters means a second pass over a TIFF
        # header and, for Micro-Manager, a few MB off a sidecar that usually
        # lives on the same network share as the movie - a second or so that
        # belongs on this thread rather than in front of the GUI.
        if not _is_cancelled(cancel):
            acquisition = read_acquisition_metadata(image_path)
    return df, image, how, acquisition, raw_image


@thread_worker
def _deform_worker(stack, region, cancel=None):
    """Measure the growth deformation on the snapshots, off the GUI thread."""
    result = yield from deform_io.measure_deformation_iter(stack, region=region, cancel=cancel)
    return CANCELLED if result is None else result


@thread_worker
def _average_worker(displayed, kwargs, cancel=None):
    """A time-averaged image of a displayed stack, off the GUI thread."""
    result = yield from averages_io.average_iter(displayed, cancel=cancel, **kwargs)
    return CANCELLED if result is None else result


@thread_worker
def _deform_movie_worker(record, stack, translation, path, cancel=None):
    """Draw and write the deformation movie off the GUI thread."""
    frames = yield from deform_movie.movie_frames_iter(record, stack, translation=translation,
                                                       cancel=cancel)
    if frames is None:
        return CANCELLED
    return deform_movie.write_movie(frames, path)


@thread_worker
def _wl_check_worker(stack, record, sigma_s, cancel=None):
    """Register the white-light snapshots against the first, off the GUI thread."""
    result = yield from drift_io.check_wl_images_iter(stack, record, sigma_s=sigma_s,
                                                      cancel=cancel)
    return CANCELLED if result is None else result


@thread_worker
def _rcc_worker(x_nm, y_nm, frames, n_frames, params, cancel=None):
    """RCC off the GUI thread: n(n-1)/2 correlations of full-field images."""
    result = yield from drift_io.rcc_iter(x_nm, y_nm, frames, n_frames, cancel=cancel,
                                          **params)
    return CANCELLED if result is None else result


@thread_worker
def _population_worker(particle, frame, x_nm, y_nm, sigma_nm, frame_interval_s, blur,
                       d_immobile_max, n_mobile, cancel=None):
    """Fit the trajectory populations off the GUI thread, and the plot's curves."""
    traj = populations_io.build_trajectories(particle, frame, x_nm, y_nm, sigma_nm)
    result = yield from populations_io.fit_populations_iter(
        traj, frame_interval_s, blur=blur, d_immobile_max=d_immobile_max,
        n_mobile=n_mobile, cancel=cancel)
    if result is None:
        return CANCELLED
    steps = np.hypot(traj.zx, traj.zy)[traj.gap == 1]
    top = float(np.percentile(steps, 99.5)) if steps.size else 1.0
    curves = populations_io.step_length_densities(traj, result, np.linspace(0.0, top, 61))
    return result, curves


@thread_worker
def _session_save_worker(session_path, manifest, locs_frame, locs_path):
    """Write a session, and the localizations it cannot recover any other way.

    Off the GUI thread because of that second part: gzipping a table of a few
    million localizations takes tens of seconds, and the manifest itself is a
    few kilobytes written in no time at all.
    """
    written = 0
    if locs_frame is not None:
        locs_frame.to_csv(locs_path, index=False, compression="gzip")
        written += locs_path.stat().st_size
    written += session_io.write_session(session_path, manifest)
    return session_path, written


@thread_worker
def _bin_worker(raw_image, bin_factor, cancel=None):
    """Re-bin an already-open stack, for when the factor changes after loading."""
    t0 = time.perf_counter()
    image, how = bin_frames(raw_image, bin_factor, cancel=cancel)
    if image is None:
        return CANCELLED
    return image, f"{how} in {time.perf_counter() - t0:.2f} s"


@thread_worker
def _link_worker(features, search_range_px, memory, n_frames, cancel=None):
    results = []
    frame_iter = (group for _, group in features.groupby("frame"))
    linked_iter = tp.link_df_iter(
        frame_iter,
        search_range=search_range_px,
        memory=memory,
        pos_columns=["y", "x"],
        t_column="frame",
    )
    total = max(n_frames, 1)
    last_pct = -1
    for i, linked_frame in enumerate(linked_iter):
        if _is_cancelled(cancel):
            return CANCELLED
        results.append(linked_frame)
        pct = int(100 * (i + 1) / total)
        if pct != last_pct:
            last_pct = pct
            yield pct / 100.0
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


@thread_worker
def _warmup_worker():
    """Compile the jitted fit kernels off the GUI thread."""
    t0 = time.perf_counter()
    warmup_fit_kernels()
    return time.perf_counter() - t0


# Frames are detected in parallel. The detection kernels are nogil, so threads
# give real parallelism (~5x measured); capped because each worker holds a frame
# and a frame-sized buffer, which adds up on 2048x2048 stacks.
DETECT_MAX_WORKERS = 8


@thread_worker
def _detect_worker(stack, box, min_ng, cancel=None):
    n_frames = stack.shape[0]
    candidates = [None] * n_frames
    counts = np.zeros(n_frames, dtype=int)
    workers = max(1, min(DETECT_MAX_WORKERS, os.cpu_count() or 1, n_frames))

    def detect(index):
        # np.asarray so a lazily-decoded (dask) stack materialises one frame here.
        return index, identify_in_frame(np.asarray(stack[index]), min_ng, box)

    done = 0
    last_pct = -1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(detect, i) for i in range(n_frames)]
        try:
            for future in as_completed(futures):
                # Checked per frame, while progress is only emitted per percent:
                # cancel latency is one frame, not one percent of the whole run.
                if _is_cancelled(cancel):
                    for pending in futures:
                        pending.cancel()
                    return CANCELLED
                index, (y, x, ng) = future.result()
                candidates[index] = (y, x, ng)
                counts[index] = len(y)
                done += 1
                # One cross-thread signal + progress-bar repaint per percent.
                pct = int(100 * done / max(n_frames, 1))
                if pct != last_pct:
                    last_pct = pct
                    yield pct / 100.0
        except GeneratorExit:
            for pending in futures:
                pending.cancel()
            raise
    return candidates, counts


@thread_worker
def _fit_worker(stack, candidates, box, backend, offset, gain, cancel=None):
    n_with_candidates = sum(1 for c in candidates if c is not None and len(c[0]) > 0)
    results = [None] * len(candidates)
    done = 0
    last_pct = -1
    for i, cand in enumerate(candidates):
        if _is_cancelled(cancel):
            return CANCELLED
        if cand is None or len(cand[0]) == 0:
            continue
        y, x, ng = cand
        results[i] = localize_frame(
            np.asarray(stack[i], dtype=np.float32),
            y,
            x,
            box,
            frame_number=i,
            net_gradient=ng,
            fit_backend=backend,
            camera_offset_adu=offset,
            camera_gain_adu_per_electron=gain,
        )
        done += 1
        pct = int(100 * done / max(n_with_candidates, 1))
        if pct != last_pct:
            last_pct = pct
            yield pct / 100.0
    return concatenate_localizations(results)


D_BATCH_TRAJECTORIES = 500


# Rows per to_csv call. Only affects how often the export can notice a cancel
# request and move the progress bar; the file written is identical either way.
EXPORT_CHUNK_ROWS = 100_000


class _RenderFailure:
    """A render error carried back as a result instead of raised. See `_render_worker`."""

    __slots__ = ("error",)

    def __init__(self, error):
        self.error = error


@thread_worker
def _render_worker(kind, options, cancel=None):
    """Drive the renderer's generator, forwarding progress and honouring cancel.

    The engine (`_render`) yields a fraction and returns the finished array;
    this adds the bridge to napari's worker signals, plus two things the GUI
    cannot do for itself. Cancelling closes the generator, which drops the
    half-finished canvas on the spot instead of waiting for a multi-gigapixel
    reconstruction nobody is going to look at. And a GPU that fails part way -
    out of memory, a driver reset, another process taking the card - falls back
    to the CPU rather than losing the render: it is slower, not wrong.

    Returns (image, backend) so the caller can report and record which one ran.
    """
    attempts = [True, False] if options.get("gpu") else [False]
    for use_gpu in attempts:
        iterator = None
        try:
            iterator = (
                smlm_render.render_frame_iter(**{**options, "gpu": use_gpu})
                if kind == "image"
                else smlm_render.render_movie_iter(**{**options, "gpu": use_gpu})
            )
            while True:
                if _is_cancelled(cancel):
                    iterator.close()
                    return CANCELLED
                try:
                    fraction = next(iterator)
                except StopIteration as finished:
                    return finished.value, ("gpu" if use_gpu else "cpu")
                yield float(fraction)
        except Exception as error:
            if iterator is not None:
                iterator.close()
            if not use_gpu:
                # Deliberately returned, never raised. An exception escaping a
                # worker is not always delivered as `errored` - a RuntimeError
                # is swallowed as "the widget went away" - and then `finished`
                # never fires either, leaving the tab stuck with its buttons
                # disabled and its progress bar spinning. Returning the failure
                # keeps the normal completion path, which always tidies up.
                return _RenderFailure(error)
        finally:
            if use_gpu:
                smlm_render.free_gpu_memory()
        yield 0.0  # restarting on the CPU; the progress bar starts over


def rotate_save_array(image, degrees, is_movie):
    """Turn the image before it is written. Returns it unchanged at 0.

    Quarter turns go through np.rot90 and are exact: pixels are permuted, never
    resampled, so a float32 reconstruction still holds the localization counts
    it held before.

    Any other angle resamples, and resampling does not conserve the total -
    interpolation samples the rotated grid rather than redistributing what was
    there, so a sparse reconstruction can lose a large fraction of its counts
    (a one-pixel line loses about 40%). That is acceptable for a figure and
    wrong for anything measured off the file afterwards, so which of the two
    happened is written into the metadata beside the image.

    Positive is counter-clockwise as the image is displayed, matching napari's
    own `layer.rotate`.
    """
    degrees = float(degrees) % 360.0
    if degrees == 0.0:
        return image
    # A movie carries frames on the first axis, and a composite carries colour
    # on the last, so the image plane is the middle pair either way.
    axes = (1, 2) if is_movie else (0, 1)
    if degrees % 90.0 == 0.0:
        return np.ascontiguousarray(np.rot90(image, k=int(degrees // 90), axes=axes))
    from scipy.ndimage import rotate as _ndrotate

    return _ndrotate(image, degrees, axes=axes, reshape=True, order=1,
                     mode="constant", cval=0).astype(image.dtype, copy=False)


def build_save_array(image, spec):
    """Turn a finished render into the array that gets written.

    Pure numpy: `spec` is the plain description assembled by the widget (see
    `_save_spec`), holding no Qt objects, so this runs on the worker thread.

    A composite re-renders each overlay through the same reconstruction path as
    the base image, which is what guarantees they line up; that is real work,
    and the reason this is not done inline in the save handler.
    """
    save_format = spec.get("format", "data")
    if save_format == "data":
        result = image
    elif save_format == "display":
        result = smlm_render.to_uint8(image, smlm_render.contrast_limits(image))
    else:
        result = smlm_render.blend_additive(
            [_composite_layer(image, layer, spec) for layer in spec["layers"]])

    crop_box = spec.get("crop")
    if crop_box is not None:
        options = spec["render"]
        rows, cols = smlm_render.box_to_slices(
            crop_box, shape=options["shape"], origin=options["origin"],
            oversampling=options["oversampling"])
        result = smlm_render.crop(result, rows, cols, is_movie=spec["is_movie"])

    # Before the annotations and after the crop: a rotated scale bar or clock
    # would be unreadable, and rotating first would leave the crop box pointing
    # at the wrong part of the image.
    result = rotate_save_array(result, spec.get("rotate_degrees", 0), spec["is_movie"])

    # Annotations are burned into the pixels, so they go on after the crop -
    # otherwise cropping could cut one in half or throw it away entirely. They
    # are skipped for a float32 "data" save, where they would corrupt the
    # numbers the export exists to preserve.
    if result.dtype == np.uint8:
        stamp = spec.get("timestamp")
        if stamp is not None:
            _annotate(result, spec["is_movie"], stamp["color"], stamp["position"],
                      labels=stamp["labels"], atlas=stamp["atlas"])
        bar = spec.get("scalebar")
        if bar is not None:
            _annotate(result, spec["is_movie"], bar["color"], bar["position"],
                      mask=bar["mask"])
    return result


def _annotate(image, is_movie, color, position, *, mask=None, labels=None, atlas=None):
    """Draw one annotation into every frame, in place.

    `mask` is the same on each frame (a scale bar); `labels` change from frame
    to frame (the clock) and are assembled per frame from the glyph atlas.
    """
    frames = image if is_movie else [image]
    for index, frame in enumerate(frames):
        if labels is not None:
            text = labels[min(index, len(labels) - 1)] if labels else ""
            mask_for_frame = smlm_render.compose_text(atlas, text)
        else:
            mask_for_frame = mask
        smlm_render.burn_text(frame, mask_for_frame, color=color, position=position)
    return image


def _composite_layer(image, layer, spec):
    """One layer of a composite, rendered onto the grid and coloured."""
    options = spec["render"]
    if layer["source"] == "base":
        values = image
    elif layer["source"] == "image":
        values = _resample_layer(layer, spec)
    else:
        common = dict(
            x_px=layer["x_px"], y_px=layer["y_px"], shape=options["shape"],
            origin=options["origin"], oversampling=options["oversampling"],
            mode="gaussian_global", global_sigma_px=layer["global_sigma_px"],
            gpu=options["gpu"],
        )
        if spec["is_movie"] and layer.get("frames") is not None:
            values = smlm_render.render_movie(
                frames=layer["frames"],
                frames_per_group=options["frames_per_group"],
                grouping=options["grouping"], step=options["step"],
                frame_range=options["frame_range"], **common)
        else:
            values = smlm_render.render_frame(**common)
            if spec["is_movie"]:
                # a layer with no frame axis belongs on every movie frame
                values = np.broadcast_to(values, (_movie_length(spec),) + values.shape)
    limits = layer.get("limits") or smlm_render.contrast_limits(values)
    return smlm_render.colorize(
        values, color=layer.get("color"), colormap=layer.get("colormap"), limits=limits)


def _movie_length(spec):
    options = spec["render"]
    first, last = options["frame_range"]
    return smlm_render.group_count(
        first, last, options["frames_per_group"], options["grouping"], options["step"])


def _resample_layer(layer, spec):
    """Bring another Image layer onto the render grid, frame by frame."""
    options = spec["render"]
    stack = layer["data"]
    common = dict(
        shape=options["shape"], origin=options["origin"],
        oversampling=options["oversampling"],
        source_scale=layer["scale"], source_translate=layer["translate"],
    )
    if not spec["is_movie"]:
        plane = stack[stack.shape[0] // 2] if layer["has_frames"] else stack
        return smlm_render.resample_to_grid(plane, **common)

    first, last = options["frame_range"]
    bounds = smlm_render.group_bounds(
        first, last, options["frames_per_group"], options["grouping"], options["step"])
    frames = []
    for start, _stop in bounds:
        # the raw frame each group opens on: a single representative plane,
        # rather than a projection that would look nothing like the movie
        index = int(np.clip(start, 0, stack.shape[0] - 1)) if layer["has_frames"] else None
        plane = stack[index] if index is not None else stack
        frames.append(smlm_render.resample_to_grid(plane, **common))
    return np.stack(frames, axis=0)


@thread_worker
def _save_render_worker(path, image, spec, metadata, super_pixel_size_nm, png, colormap,
                        frame_interval_s=None):
    """Build and write a render off the GUI thread.

    Both halves belong here: a composite has to re-render its overlays, and a
    5 GB TIFF takes a while to write - doing either on the GUI thread would
    freeze the window for exactly as long as the render took.
    """
    return smlm_render.save_render(
        path, build_save_array(image, spec), metadata,
        super_pixel_size_nm=super_pixel_size_nm, png=png, colormap=colormap,
        frame_interval_s=frame_interval_s,
    )


@thread_worker
def _export_worker(folder, tables, metadata, cancel=None):
    """Write the exported tables and metadata off the GUI thread.

    Writing a few hundred thousand localizations to CSV takes seconds, and doing
    it inline froze the whole window. Everything Qt-owned - the figures, and the
    widget values behind `metadata` - is prepared by the caller; this only
    touches plain DataFrames and dicts.

    `tables` is a list of (filename, DataFrame).
    """
    data_dir = folder / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Progress is weighted by rows: the localization table usually dwarfs the rest.
    total_rows = sum(max(len(frame), 1) for _name, frame in tables) or 1
    written_rows = 0
    last_pct = -1

    for name, frame in tables:
        path = data_dir / name
        n_rows = len(frame)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            if n_rows == 0:
                frame.to_csv(handle, index=False)
            for start in range(0, n_rows, EXPORT_CHUNK_ROWS):
                if _is_cancelled(cancel):
                    handle.close()
                    path.unlink(missing_ok=True)  # no half-written table left behind
                    return CANCELLED
                stop = min(start + EXPORT_CHUNK_ROWS, n_rows)
                frame.iloc[start:stop].to_csv(handle, index=False, header=(start == 0))
                pct = int(100 * (written_rows + stop) / total_rows)
                if pct != last_pct:
                    last_pct = pct
                    yield pct / 100.0
        written_rows += max(n_rows, 1)

    if _is_cancelled(cancel):
        return CANCELLED
    with open(folder / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    return folder


def fit_msd_slope(tau, msd):
    """Least-squares MSD = 4D*tau + c. Returns (slope, intercept, slope_error).

    The fit is done on the raw values and stays that way however the validation
    plot chooses to *draw* them. Fitting in log space instead would minimise
    relative rather than absolute residuals, which hands the short lag times -
    the noisiest, and the ones most contaminated by localization error - far
    more weight than they have earned.

    The error is the standard error of the slope, and it is an underestimate of
    the real uncertainty on D: MSD points at different lag times come from
    overlapping displacements of the same trajectory, so they are strongly
    correlated, which is exactly what ordinary least squares assumes they are
    not. It is worth showing because it separates a trajectory long enough to
    pin its slope down from one that is not, but it is not a confidence
    interval to quote.
    """
    try:
        (slope, intercept), covariance = np.polyfit(tau, msd, 1, cov=True)
        error = float(np.sqrt(abs(covariance[0, 0])))
    except (ValueError, np.linalg.LinAlgError):
        # The covariance needs more points than parameters + 2. Below that the
        # slope is still the best line through them; its error is undefined.
        slope, intercept = np.polyfit(tau, msd, 1)
        error = float("nan")
    return float(slope), float(intercept), error


def msd_sigma_nm(intercept_um2):
    """The localization precision the MSD intercept implies, per axis, in nm.

    In two dimensions MSD(tau) = 4*D*tau + 4*sigma^2, so the intercept is four
    times the squared precision and sqrt(intercept)/2 recovers it. This is a
    second, completely independent estimate of the same quantity the spot fitter
    reports: one comes from the shape of a single spot, the other from how much
    a trajectory jitters. Where they disagree, something is wrong with one of
    them, and the ratio is the correction the immobility test needs.

    Two things bias it low and both matter. Motion blur subtracts 8*R*D*dt from
    the intercept (R = 1/6 for continuous illumination), so a fast molecule can
    even produce a negative one - which is why this is read off the slow end of
    the population. And the intercept is extrapolated from a handful of
    correlated MSD points, so it is noisy per trajectory and only worth
    believing in aggregate.
    """
    if not np.isfinite(intercept_um2) or intercept_um2 <= 0:
        return float("nan")
    return float(np.sqrt(intercept_um2) / 2.0 * 1000.0)


@thread_worker
def _compute_d_worker(tracks_df, max_lagtime, fps, mpp, cancel=None):
    # MSD is computed independently per trajectory, so running tp.imsd on
    # batches of whole trajectories is equivalent to one call over all of them -
    # but it gives the run somewhere to notice a cancel request and a real
    # percentage to report, instead of one opaque blocking call.
    d_map = {}
    msd_map = {}
    last_pct = -1

    for subset, done, total in iter_particle_batches(tracks_df, D_BATCH_TRAJECTORIES):
        if _is_cancelled(cancel):
            return CANCELLED
        im = tp.imsd(subset, mpp=mpp, fps=fps, max_lagtime=max_lagtime, pos_columns=["x", "y"])
        for pid in im.columns:
            msd_series = im[pid].dropna()
            if len(msd_series) < 3:
                continue
            tau = msd_series.index.to_numpy(float)
            msd_vals = msd_series.to_numpy(float)
            slope, intercept, slope_error = fit_msd_slope(tau, msd_vals)
            D = slope / 4.0
            if D > 0 and np.isfinite(D):
                d_map[pid] = D
                msd_map[pid] = (tau, msd_vals, slope, intercept, slope_error)
        pct = int(100 * done / max(total, 1))
        if pct != last_pct:
            last_pct = pct
            yield pct / 100.0

    return d_map, msd_map


def immobility_statistic(x, y, sigma):
    """Test a trajectory against the hypothesis that it never moved.

    A static emitter is a completely specified statistical object: every
    position it reports is its true position plus localization error, and that
    error is measured for each spot by the same fit that produced the position.
    So the scatter of a trajectory about its own centre, with each residual
    divided by its own uncertainty, is a weighted residual sum of squares of
    Gaussians about their fitted mean - which is chi-squared with 2(N-1) degrees
    of freedom, exactly, for every N. Two of the 2N coordinates are spent
    estimating the centre, one per spatial dimension; nothing else is estimated.

    Returns (T, dof). `x`, `y` and `sigma` must share a unit; the statistic is
    dimensionless, so pixels and nanometres both work as long as they agree.

    Unlike a diffusion coefficient this costs one pass and no fit, and unlike a
    diffusion coefficient it is well behaved on the trajectories that matter
    here - the short ones, where the MSD slope is at its least reliable.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sigma = np.asarray(sigma, float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    if good.sum() < 2:
        # One point cannot scatter, and a missing precision cannot be guessed.
        return float("nan"), 0
    x, y, sigma = x[good], y[good], sigma[good]

    # Precision weights: the maximum-likelihood centre lets a well-measured spot
    # pull harder than a dim one. Weighting matters more than it looks - photon
    # count varies several-fold between spots, and treating a 15 nm and a 45 nm
    # localization as equally informative both loses power and breaks the null.
    weight = 1.0 / sigma ** 2
    total = weight.sum()
    x_bar = float((x * weight).sum() / total)
    y_bar = float((y * weight).sum() / total)
    T = float((weight * ((x - x_bar) ** 2 + (y - y_bar) ** 2)).sum())
    return T, 2 * (len(x) - 1)


def detectable_diffusion(n_points, sigma, frame_interval_s, alpha):
    """The smallest D this trajectory could have told apart from standing still.

    "Not significantly moving" is not "static": a three-point trajectory of a
    dim molecule cannot detect anything slower than a fair fraction of a square
    micron per second, and reporting it as immobile without saying so hides the
    single largest bias in the whole classification.

    Under motion with per-axis step variance 2*D*dt, a random walk of N points
    has an expected sum of squared deviations from its own mean of
    2*D*dt*(N^2-1)/6, so

        E[T] = 2(N-1) + (2(N^2-1)/3) * D*dt / sigma^2

    and the D that pushes E[T] up to the critical value is the detection floor.
    `sigma` and the returned D share a length unit: nanometres in, nm^2/s out.

    Depends on nothing but the trajectory itself - its length and its own
    localization precision - the frame interval, and the significance the
    detection is claimed at. That last is the only free choice here.
    """
    n_points = int(n_points)
    if n_points < 3 or sigma <= 0 or frame_interval_s <= 0:
        # Two points have one degree of freedom between them after the centre is
        # estimated, and nothing to say about a rate.
        return float("inf")
    from scipy import stats

    dof = 2 * (n_points - 1)
    critical = float(stats.chi2.isf(alpha, dof))
    floor = ((critical - dof) * 3.0 * sigma ** 2
             / (2.0 * (n_points ** 2 - 1) * frame_interval_s))
    return float(floor * D_FLOOR_MEDIAN_CORRECTION)


def immobility_maps(tracks_df, sigma_column, pixel_size=1.0,
                    frame_interval_s=None, alpha=0.05):
    """Motion ratio and p(static) per trajectory, or ({}, {}) with no precision.

    The ratio is the effect size - the trajectory's spread as a multiple of the
    spread localization error alone would produce, so 1 means "moved exactly as
    much as a stationary molecule would appear to". The p-value is the
    significance, and the two answer different questions: the ratio does not
    depend on how long the trajectory was watched, while the p-value does, which
    is what lets a long trajectory certify a smaller motion than a short one.

    The third map is the detection floor: the smallest D this trajectory could
    have distinguished from standing still, given its own length and precision.
    Without it "not significantly moving" reads as "static", which for a short
    trajectory it very often is not.
    """
    if not sigma_column or sigma_column not in tracks_df.columns:
        return {}, {}, {}
    from scipy import stats

    motion_map, pstatic_map, floor_map = {}, {}, {}
    for pid, group in tracks_df.groupby("particle"):
        sigma = group[sigma_column].to_numpy(float)
        T, dof = immobility_statistic(
            group["x"].to_numpy(float), group["y"].to_numpy(float), sigma)
        if dof <= 0 or not np.isfinite(T):
            continue
        motion_map[pid] = T / dof
        pstatic_map[pid] = max(float(stats.chi2.sf(T, dof)), P_STATIC_FLOOR)

        if frame_interval_s:
            usable = sigma[np.isfinite(sigma) & (sigma > 0)]
            if usable.size >= 3:
                # The precision-weighted effective sigma, matching the weighting
                # the statistic itself uses, in nanometres.
                effective = np.sqrt(usable.size / (1.0 / usable ** 2).sum()) * pixel_size
                floor_map[pid] = detectable_diffusion(
                    usable.size, effective, frame_interval_s, alpha) / 1e6  # -> µm²/s
    return motion_map, pstatic_map, floor_map


SIGMA_COLUMN = "_sigma"


@thread_worker
def _fit_free_metrics_worker(tracks_df, pixel_size, fps, alpha=0.05):
    """Per-trajectory quantities that need no model fitted to them.

    Three ways of asking how far a molecule went, which answer differently and
    are only worth having together:

      * distance - the path length, every step added up. Grows without bound
        while a molecule wanders, and is inflated by localization noise: even a
        stationary spot accumulates roughly one precision per step.
      * net      - the end-to-end displacement, start to finish. Where it ended
        up, regardless of how it got there.
      * straightness - net / distance, between 0 and 1. This is the one that
        separates directed motion from diffusion: a molecule moving in a line
        approaches 1, while an N-step random walk sits near 1/sqrt(N) however
        fast it diffuses. Net displacement on its own cannot make that
        distinction, because a fast diffuser also ends up a long way away.
    """
    fps_safe = max(fps, 1e-9)
    distance_map = {}
    net_map = {}
    straightness_map = {}
    duration_map = {}
    for pid, group in tracks_df.groupby("particle"):
        group = group.sort_values("frame")
        x = group["x"].to_numpy(float)
        y = group["y"].to_numpy(float)
        # pixel_size is nm/px; every length here is reported in µm.
        to_um = pixel_size / 1000.0
        path = float(np.hypot(np.diff(x), np.diff(y)).sum() * to_um)
        net = float(np.hypot(x[-1] - x[0], y[-1] - y[0]) * to_um) if len(x) else 0.0
        distance_map[pid] = path
        net_map[pid] = net
        # A trajectory that never moved has no direction to be straight in.
        straightness_map[pid] = net / path if path > 0 else float("nan")
        span = int(group["frame"].max() - group["frame"].min()) + 1
        duration_map[pid] = span / fps_safe
    motion_map, pstatic_map, floor_map = immobility_maps(
        tracks_df, SIGMA_COLUMN, pixel_size=pixel_size,
        frame_interval_s=1.0 / fps_safe, alpha=alpha)
    # A dict rather than a tuple: there are six of these now, and a positional
    # unpack that has to be corrected everywhere each time one is added is a
    # standing invitation to swap two of them silently.
    return {"distance": distance_map, "net": net_map,
            "straightness": straightness_map, "duration": duration_map,
            "motion": motion_map, "pstatic": pstatic_map, "dmin": floor_map}


class PandasTableModel(QAbstractTableModel):
    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def set_dataframe(self, df):
        self.beginResetModel()
        self._df = df if df is not None else pd.DataFrame()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.index)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or not index.isValid():
            return None
        value = self._df.iat[index.row(), index.column()]
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(section)



# Formats offered when a graph is lifted out. The vector ones are first because
# a reconstruction histogram in a talk is usually projected: PDF and SVG stay
# sharp at any size, where a PNG is fixed at whatever resolution it was written.
FIGURE_FORMATS = (
    ("PDF (vector)", "pdf"),
    ("SVG (vector)", "svg"),
    ("PNG (raster)", "png"),
    ("TIFF (raster)", "tiff"),
)


class FigureExportDialog(QDialog):
    """Choose how one graph is written out, against a live preview.

    Separate from the panel's own size control on purpose. The plots on screen
    are sized for reading in a side dock; a figure for a slide or a paper wants
    a different shape, a larger font and often different furniture, and choosing
    those should not disturb the thing being read. Nothing here touches the
    panel: every preview is drawn into this dialog's own figure.
    """

    def __init__(self, parent, redraw, name, folder):
        super().__init__(parent)
        self.setWindowTitle(f"Save graph - {name}")
        self._redraw = redraw
        self._name = name
        self._folder = folder
        self._saved_path = None

        layout = QHBoxLayout(self)
        layout.setSpacing(12)

        # --- the preview ---------------------------------------------------
        preview_box = QVBoxLayout()
        self.preview_figure = Figure()
        self.preview_canvas = FigureCanvas(self.preview_figure)
        self.preview_canvas.setMinimumSize(560, 380)
        preview_box.addWidget(self.preview_canvas, 1)
        self.size_label = QLabel()
        self.size_label.setProperty("role", "note")
        preview_box.addWidget(self.size_label)
        layout.addLayout(preview_box, 1)

        # --- the controls ---------------------------------------------------
        controls = QVBoxLayout()
        controls.setSpacing(8)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.width_box = QSpinBox()
        self.width_box.setRange(200, 6000)
        self.width_box.setValue(1200)
        self.width_box.setSingleStep(50)
        self.width_box.setSuffix(" px")
        form.addRow("Width", self.width_box)

        self.height_box = QSpinBox()
        self.height_box.setRange(150, 6000)
        self.height_box.setValue(750)
        self.height_box.setSingleStep(50)
        self.height_box.setSuffix(" px")
        form.addRow("Height", self.height_box)

        self.shape_box = QComboBox()
        for label, ratio in PLOT_ASPECTS[1:]:
            self.shape_box.addItem(label, ratio)
        self.shape_box.addItem("free", None)
        self.shape_box.setCurrentIndex(self.shape_box.count() - 1)
        self.shape_box.setToolTip(
            "Lock the width to the height at a fixed ratio, or leave both free.")
        form.addRow("Shape", self.shape_box)

        self.font_box = QSpinBox()
        self.font_box.setRange(4, 48)
        self.font_box.setValue(14)
        self.font_box.setSuffix(" pt")
        self.font_box.setToolTip(
            "Body text. Ticks sit a point below, titles a point above. 14-18 "
            "reads from the back of a room; 8-10 suits a figure panel.")
        form.addRow("Font", self.font_box)

        self.dpi_box = QSpinBox()
        self.dpi_box.setRange(72, 1200)
        self.dpi_box.setValue(300)
        self.dpi_box.setSingleStep(50)
        self.dpi_box.setToolTip(
            "Raster resolution. Ignored by the vector formats, which have none.")
        form.addRow("Resolution", self.dpi_box)

        self.format_box = QComboBox()
        for label, ext in FIGURE_FORMATS:
            self.format_box.addItem(label, ext)
        form.addRow("Format", self.format_box)
        controls.addLayout(form)

        self.transparent_box = QCheckBox("Transparent background")
        self.transparent_box.setChecked(True)
        self.transparent_box.setToolTip(
            "Drop the background so the graph sits on whatever the slide "
            "provides. The axes and labels are light, so it reads on a dark "
            "background and not on a white one.")
        self.errorbars_box = QCheckBox("Counting error bars (√n)")
        self.errorbars_box.setToolTip(
            "The only error bar a histogram of independent samples honestly "
            "has: the Poisson counting error on each bin.")
        self.grid_box = QCheckBox("Grid")
        self.grid_box.setChecked(True)
        self.bounds_box = QCheckBox("Filter bounds")
        self.bounds_box.setChecked(True)
        self.bounds_box.setToolTip("The shaded band and lines showing the range in use.")
        self.title_box = QCheckBox("Title")
        self.title_box.setChecked(True)
        self.colorbar_box = QCheckBox("Colour bar")
        self.colorbar_box.setChecked(True)
        for box in (self.transparent_box, self.errorbars_box, self.grid_box,
                    self.bounds_box, self.title_box, self.colorbar_box):
            controls.addWidget(box)

        controls.addStretch(1)
        self.destination_label = QLabel()
        self.destination_label.setWordWrap(True)
        self.destination_label.setProperty("role", "note")
        controls.addWidget(self.destination_label)

        buttons = QHBoxLayout()
        save = QPushButton("Save")
        save.setProperty("primary", True)
        save.clicked.connect(self._save)
        cancel = QPushButton("Cancel")
        cancel.setProperty("secondary", True)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(save)
        buttons.addWidget(cancel)
        controls.addLayout(buttons)
        layout.addLayout(controls)

        for widget in (self.width_box, self.height_box, self.font_box):
            widget.valueChanged.connect(self._refresh)
        self.shape_box.currentIndexChanged.connect(self._on_shape_changed)
        self.format_box.currentIndexChanged.connect(self._refresh)
        for box in (self.transparent_box, self.errorbars_box, self.grid_box,
                    self.bounds_box, self.title_box, self.colorbar_box):
            box.stateChanged.connect(self._refresh)
        self._refresh()

    # -- options ---------------------------------------------------------
    def options(self):
        return {
            "errorbars": self.errorbars_box.isChecked(),
            "grid": self.grid_box.isChecked(),
            "bounds": self.bounds_box.isChecked(),
            "title": self.title_box.isChecked(),
            "colorbar": self.colorbar_box.isChecked(),
            "legend": True,
        }

    def _on_shape_changed(self, _index):
        ratio = self.shape_box.currentData()
        if ratio:
            self.width_box.blockSignals(True)
            self.width_box.setValue(int(round(self.height_box.value() * ratio)))
            self.width_box.blockSignals(False)
        self._refresh()

    def _draw_into(self, figure, width, height, dpi):
        """Render at a known pixel size, with the font this dialog asked for."""
        previous = plot_font()
        set_plot_font_size(self.font_box.value())
        try:
            figure.set_dpi(dpi)
            figure.set_size_inches(width / dpi, height / dpi, forward=False)
            self._redraw(figure, self.options())
        finally:
            set_plot_font_size(previous)

    def _refresh(self):
        ratio = self.shape_box.currentData()
        if ratio:
            self.width_box.blockSignals(True)
            self.width_box.setValue(int(round(self.height_box.value() * ratio)))
            self.width_box.blockSignals(False)
        width, height = self.width_box.value(), self.height_box.value()

        # The preview is the same figure at a smaller scale, so what is on
        # screen is what gets written - proportions, font size relative to the
        # frame, and everything the options changed.
        scale = min(self.preview_canvas.width() / max(width, 1),
                    self.preview_canvas.height() / max(height, 1), 1.0) or 1.0
        self._draw_into(self.preview_figure, max(width * scale, 120),
                        max(height * scale, 90), 100)
        self.preview_canvas.draw_idle()

        extension = self.format_box.currentData()
        vector = extension in ("pdf", "svg")
        self.dpi_box.setEnabled(not vector)
        self.size_label.setText(
            f"{width} × {height} px"
            + ("  ·  vector, resolution-independent" if vector
               else f"  ·  {self.dpi_box.value()} dpi → "
                    f"{int(width * self.dpi_box.value() / 100)} × "
                    f"{int(height * self.dpi_box.value() / 100)} px written"))
        self.destination_label.setText(f"Saves into {self._folder}")

    def _save(self):
        extension = self.format_box.currentData()
        stem = LocalizationTrackingWidget._safe_filename(self._name) or "figure"
        path = self._folder / f"{stem}.{extension}"
        i = 2
        while path.exists():
            path = self._folder / f"{stem}_{i}.{extension}"
            i += 1

        dpi = 100 if extension in ("pdf", "svg") else self.dpi_box.value()
        figure = Figure()
        try:
            self._folder.mkdir(parents=True, exist_ok=True)
            self._draw_into(figure, self.width_box.value(), self.height_box.value(), 100)
            figure.savefig(path, dpi=dpi, transparent=self.transparent_box.isChecked(),
                           bbox_inches="tight", pad_inches=0.05, format=extension)
        except Exception as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return
        self._saved_path = path
        self.accept()

    def saved_path(self):
        return self._saved_path


class LocalizationTrackingWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        # Any layer arriving in the viewer - added here, or dragged in by the
        # user - has to be put in the same physical world as the rest, or the
        # scale bar would be describing only some of what is on screen.
        try:
            viewer.layers.events.inserted.connect(
                lambda event=None: self._apply_viewer_scale())
        except Exception:
            pass
        # The canvas clock follows the slider, however the slider was moved -
        # dragged, or run by the play button.
        try:
            viewer.dims.events.current_step.connect(
                lambda event=None: self._on_current_frame_changed())
        except Exception:
            pass
        apply_black_canvas(viewer)
        self.df = None
        self.df_filtered = None
        self.column_map = {}
        self.tracks = None
        self._hist_widgets = {}
        self._metric_hist_widgets = {}
        self._metric_bound_boxes = {}
        self._metric_filter_boxes = {}
        # The dynamics selection, derived on demand from whichever ranges are
        # ticked and dropped whenever anything it is derived from moves.
        self._passing_particles_cache = None
        self._loc_particle_cache = None
        # Distance travelled, like D, is spread over orders of magnitude across
        # a population: on a linear axis nearly every trajectory lands in the
        # first bin and the plot ends up describing the handful of longest ones.
        # Duration is bounded by the acquisition and stays linear.
        # Distance and net displacement, like D, are spread over orders of
        # magnitude across a population; on a linear axis nearly every
        # trajectory lands in the first bin. Straightness is a ratio in [0, 1]
        # and duration is bounded by the acquisition, so both stay linear.
        # The motion ratio spans decades between a bound molecule and a fast one,
        # and a p-value spans every decade it has; straightness is a ratio in
        # [0,1] and duration is bounded by the acquisition, so both stay linear.
        self._metric_use_log = {"D": True, "distance": True, "net": True,
                                "straightness": False, "duration": False,
                                "motion": True, "pstatic": True, "dmin": True}
        self._default_bounds = {}
        self.filter_controls = {}
        # The filters someone saved as their defaults (the metadata layout of
        # `filter_settings_of`), read at start-up; None means the built-in ones.
        self._user_filter_defaults = None
        self._user_filter_defaults_saved_at = None
        self._roi_updating = False
        self._track_diffusion_cache = None
        self._track_msd_cache = None
        self._track_distance_cache = None
        self._track_net_cache = None
        self._track_straightness_cache = None
        self._track_duration_cache = None
        self._track_motion_cache = None
        self._track_pstatic_cache = None
        self._track_dmin_cache = None
        self._all_tracks_particle_ids = []
        self._load_worker_ref = None
        self._link_worker_ref = None
        self._loc2d_candidates = []
        self._loc2d_counts = np.zeros(0, dtype=int)
        self._loc2d_detect_worker_ref = None
        self._loc2d_fit_worker_ref = None
        self._loc2d_warmup_worker_ref = None
        self._loc2d_warmup_started = False
        # Cooperative cancel flags, one per long-running operation. Cleared when
        # the operation starts, set by its Cancel button.
        self._load_cancel = threading.Event()
        self._loc2d_detect_cancel = threading.Event()
        self._loc2d_fit_cancel = threading.Event()
        self._render_cancel = threading.Event()
        self._link_cancel = threading.Event()
        self._compute_d_cancel = threading.Event()
        self._export_cancel = threading.Event()
        self._export_worker_ref = None
        self._compute_d_worker_ref = None
        self._metrics_worker_ref = None
        self._d_input_track_count = None
        self._syncing_timing = False
        self._tracks_layer_particles = None
        # Last render kept in memory so it can be saved (and re-saved with
        # different options) without recomputing it.
        self._render_image = None
        self._render_movie = None
        self._render_extent_px = None
        self._render_frame_range = None
        self._render_image_info = None
        self._render_movie_info = None
        self._render_worker_ref = None
        self._render_save_worker_ref = None
        self._syncing_paths = False
        # How far the loaded frame numbers are shifted to line up with the image
        # stack; set from the buttons under the CSV field, or guessed on load.
        self._frame_shift = 0
        # Filter bounds from a settings file whose columns are not loaded yet.
        self._pending_filter_bounds = None
        # Time binning: the unbinned stack as it was opened, so the factor can
        # be changed without re-reading the file, and the factor the camera
        # baseline and frame rate in the boxes currently account for.
        self._raw_image = None
        self._image_layer_name = None
        self._time_bin_applied = 1
        self._bin_worker_ref = None
        self._bin_cancel = threading.Event()
        # Drift correction. The table is kept as it was loaded, so the
        # correction can be changed or taken off again; `self.df` is that table
        # with the drift of the moment taken out. A table that arrived already
        # corrected carries its own drift, kept here to fall back on.
        self._df_source = None
        self._embedded_drift = None
        self._drift_record = None
        self._frame_clock = None
        self._drift_frames = None
        self._drift_frames_key = None
        # How the record's white-light px become fluorescence px (a CameraMap),
        # or why none applies to it.
        self._camera_map = None
        self._camera_map_problem = None
        # Whether the acquisition said where its image sits on the sensor, and
        # everything else it said, as read off its files.
        self._sensor_roi_recorded = False
        self._acquisition_values = {}
        # The tissue's growth measured on the snapshots (a DeformationRecord) and
        # where it was read from; with it in force, the localizations' positions
        # with the local stretch taken out, for the metrics.
        self._deformation = None
        self._deformation_path = None
        self._local_px = None
        self._pending_local = None
        self._deform_worker_ref = None
        self._deform_cancel = threading.Event()
        self._movie_worker_ref = None
        self._movie_cancel = threading.Event()
        self._average_worker_ref = None
        self._average_cancel = threading.Event()
        self._profile_data = None
        # What the table on screen was last corrected by, so a refresh that
        # would land on the same numbers leaves everything built on it alone.
        self._applied_drift = None
        self._drift_rows_off_end = 0
        # The white-light snapshots beside the record, read lazily, and the
        # last check of the record against them.
        self._wl_stack = None
        self._wl_check = None
        self._wl_check_worker_ref = None
        self._wl_check_cancel = threading.Event()
        self._wl_viewer = None
        # The RCC estimate: what it found, and the white-light correction it
        # was estimated on top of - it only refines that one.
        self._rcc = None
        self._rcc_worker_ref = None
        self._rcc_cancel = threading.Event()
        # Which trajectories a render merges, and the merged table, kept until
        # anything they are built from changes.
        self._merge_cache = None
        # The immobility class the population buttons selected ("immobile",
        # "undetermined", "mobile" or None), and every trajectory's class.
        self._population_class = None
        self._class_cache = None
        # The population fit: its result, and what it was fitted on.
        self._population_fit = None
        self._population_worker_ref = None
        self._population_cancel = threading.Event()
        # Where the trajectories came from, when they were read rather than
        # linked: a session re-links its own trajectories but must not re-link
        # someone else's, which these parameters would not reproduce.
        self._tracks_source_path = None
        # Trajectories already linked, by the localizations they were linked
        # from (see `_localization_set_key`): going back to a filter setting,
        # a drift correction or a frame shift that was linked before brings its
        # trajectories back - with their metrics and population fit - instead
        # of linking them again.
        self._tracks_key = None
        self._linked_memory = OrderedDict()
        # The queue of restore steps while a session is being reloaded, and None
        # at every other moment - the completion hooks test it to tell an
        # ordinary load from one step of a restore.
        self._session_restore = None
        self._session_save_worker_ref = None
        self._autosave_worker_ref = None
        # Where this render session's files go; cleared by each new render.
        self._render_save_folder = None
        # Where graphs lifted out for a slide collect, one folder per session.
        self._figure_save_folder = None
        # Every matplotlib canvas, so one size control can reach all of them
        # without each having to be found by name.
        self._plot_canvases = []
        self.setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def setup_ui(self):
        self.setStyleSheet(STYLESHEET)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addWidget(self._build_status_header())

        self.tabs = QTabWidget(self)
        root.addWidget(self.tabs)

        self._build_load_tab()
        self._build_localize_tab()
        # Between localizing and filtering: every later step works on the
        # positions, so they are corrected before anything is decided on them.
        self._build_drift_tab()
        self._build_filter_tab()
        # Track before Render: a reconstruction is now as often built from a
        # dynamics selection as from every localization, and you cannot make
        # that selection until the trajectories and their metrics exist. The
        # tab order is the order the work is done in.
        self._build_track_tab()
        self._build_render_tab()
        self._build_images_tab()
        self._build_save_tab()
        # The data table is a view of the current data, not a step in the
        # pipeline, so it opens on demand instead of taking up a tab.
        self._build_data_table_dialog()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(110)
        self.log_box.setFont(QFont("Consolas", 8))
        self.log_box.setStyleSheet(
            f"QPlainTextEdit {{ background-color: {PANEL_BG}; color: {INK_DIM};"
            f" border: 1px solid {PANEL_LINE}; border-radius: 4px; }}"
        )
        root.addWidget(self.log_box)

        self._metric_render_timer = QTimer(self)
        self._metric_render_timer.setSingleShot(True)
        self._metric_render_timer.timeout.connect(self._refresh_metric_colors)
        self._track_filter_timer = QTimer(self)
        self._track_filter_timer.setSingleShot(True)
        self._track_filter_timer.timeout.connect(self._apply_track_filter)
        self._update_track_filter_label()
        self._update_link_cutoff_label()
        self._update_status_header()
        self._update_immobility_status()
        self._apply_plot_size()
        # Last: the defaults move controls that all have to exist, and say so
        # in the log.
        self._load_filter_defaults()

    def _build_status_header(self):
        """A one-line summary of where the data stands, visible from every tab.

        The counts used to be spread across the tab that produced them, so
        answering "how many localizations survived the filter" meant leaving
        whatever you were doing to go and look.
        """
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(2, 0, 2, 0)
        self.status_label = QLabel("No data loaded")
        self.status_label.setProperty("role", "heading")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label, 1)

        self.show_table_button = QPushButton("Data table")
        self.show_table_button.setProperty("secondary", True)
        self.show_table_button.setToolTip("Open the current localizations as a table")
        self.show_table_button.clicked.connect(self.show_data_table)
        layout.addWidget(self.show_table_button)

        self.export_button = QPushButton("Export...")
        self.export_button.setProperty("primary", True)
        self.export_button.setToolTip(
            "Export whatever is currently available: the filtered localizations "
            "always, plus trajectories and their metrics once they exist."
        )
        self.export_button.clicked.connect(self.export_analysis)
        layout.addWidget(self.export_button)

        self.export_cancel_button = self._cancel_button(self._export_cancel, "the export")
        layout.addWidget(self.export_cancel_button)
        self.export_progress = QProgressBar()
        self.export_progress.setRange(0, 100)
        self.export_progress.setVisible(False)
        self.export_progress.setMaximumWidth(120)
        layout.addWidget(self.export_progress)
        return header

    def _update_status_header(self):
        """Refresh the header after anything that changes the data."""
        if not hasattr(self, "status_label"):
            return
        parts = []
        layer = self._source_image_layer()
        if layer is not None:
            shape = getattr(drift_io.unshifted(getattr(layer, "data", None)), "shape", None)
            if shape is not None:
                parts.append(f"{layer.name}  {'x'.join(str(int(v)) for v in shape)}")
        if self.df is not None:
            shown = self._displayed_localizations()
            kept = len(shown) if shown is not None else len(self.df)
            parts.append(f"{kept} / {len(self.df)} localizations")
        if self.tracks is not None and not self.tracks.empty:
            n_all = int(self.tracks["particle"].nunique())
            shown_tracks = self._displayed_tracks()
            n_shown = int(shown_tracks["particle"].nunique()) if len(shown_tracks) else 0
            parts.append(f"{n_shown} trajectories" if n_shown == n_all
                         else f"{n_shown} / {n_all} trajectories")
        if self.df is not None:
            parts.append(f"{self.pixel_size_box.value():.0f} nm/px")
            if self._applied_drift is not None:
                parts.append("drift-corrected")
        self.status_label.setText("     ".join(parts) if parts else "No data loaded")
        has_data = self.df is not None
        self.show_table_button.setEnabled(has_data)
        self.export_button.setEnabled(has_data)

    def _get_current_frame(self):
        try:
            step = self.viewer.dims.current_step
            if isinstance(step, tuple) and len(step) > 0:
                return int(step[0])
        except Exception:
            pass
        return 0

    # ------------------------------------------------------------------
    # Restoring settings from a previous analysis
    # ------------------------------------------------------------------
    def load_settings_from_metadata(self, path=None):
        """Restore the parameters recorded in a previous run's metadata.json, or
        in a session file - the settings only, not its data."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load settings from a previous analysis or session",
                filter=SETTINGS_FILE_FILTER,
            )
        if not path:
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                metadata = settings_of_file(json.load(handle))
        except Exception as exc:
            self.log(f"Could not read settings from {Path(path).name}: {exc}")
            return None

        applied, skipped, notes = self.apply_settings(metadata)
        source = metadata.get("source_csv") or metadata.get("source_image")
        exported = metadata.get("exported_at")
        self.log(
            f"Restored {len(applied)} settings from {Path(path).name}"
            + (f", exported {exported}" if exported else "")
        )
        if source:
            self.log(f"Those settings came from an analysis of {source}")
        for note in notes:
            self.log(f"  note: {note}")
        if skipped:
            self.log(f"  {len(skipped)} setting(s) not applied: {', '.join(sorted(skipped)[:6])}")
        return applied

    # ------------------------------------------------------------------
    # Whole sessions: a manifest, and the pipeline re-run from it
    # ------------------------------------------------------------------
    def _capture_session_view(self):
        """Where the viewer is looking, as far as it will say.

        Guarded throughout: this has to work against whatever viewer the plugin
        was handed, and a view that cannot be read back is not a reason to
        refuse to save the session.
        """
        view = {}
        dims = getattr(self.viewer, "dims", None)
        if dims is not None:
            try:
                view["current_step"] = [int(v) for v in dims.current_step]
                view["ndisplay"] = int(dims.ndisplay)
            except (AttributeError, TypeError, ValueError):
                pass
        camera = getattr(self.viewer, "camera", None)
        if camera is not None:
            try:
                view["camera"] = {
                    "center": [float(v) for v in camera.center],
                    "zoom": float(camera.zoom),
                    "angles": [float(v) for v in camera.angles],
                }
            except (AttributeError, TypeError, ValueError):
                pass
        try:
            view["layer_visibility"] = {
                layer.name: bool(layer.visible) for layer in self.viewer.layers
            }
        except (AttributeError, TypeError):
            pass
        if hasattr(self, "tabs"):
            view["active_tab"] = int(self.tabs.currentIndex())
        view["layer_display"] = self._layer_display()
        return view

    def _session_extras(self):
        """The layers made from the data rather than loaded: made again on restore."""
        averages = []
        for layer in self.viewer.layers:
            meta = getattr(layer, "metadata", None) or {}
            if meta.get(AVERAGE_TAG):
                averages.append({"kind": meta[AVERAGE_TAG], "params": meta.get("params") or {}})
        profile = self._profile_line_world() if hasattr(self, "profile_figure") else None
        return {
            "wl_overlay": self._wl_overlay_layer() is not None,
            "averages": averages,
            "merged_layers": MERGED_AFTER_LAYER_NAME in self.viewer.layers,
            "profile_line": ([[float(v) for v in profile[0]], [float(v) for v in profile[1]]]
                             if profile is not None else None),
        }

    def _layer_display(self):
        """How each image layer looks: colours, contrast, blending, visibility."""
        out = {}
        for layer in self.viewer.layers:
            try:
                entry = {"visible": bool(layer.visible), "opacity": float(layer.opacity),
                         "blending": str(layer.blending)}
            except (AttributeError, TypeError, ValueError):
                continue          # not a layer that has a look to keep
            if isinstance(layer, napari.layers.Image):
                try:
                    entry["colormap"] = str(layer.colormap.name)
                    entry["contrast_limits"] = [float(v) for v in layer.contrast_limits]
                    entry["gamma"] = float(layer.gamma)
                except Exception:
                    pass
            out[layer.name] = entry
        return out

    def _restore_layer_display(self, display):
        for name, entry in (display or {}).items():
            if name not in self.viewer.layers:
                continue
            layer = self.viewer.layers[name]
            for attr in ("colormap", "contrast_limits", "gamma", "blending", "opacity", "visible"):
                if attr in entry:
                    try:
                        setattr(layer, attr, entry[attr])
                    except Exception:
                        pass

    def _restore_session_view(self, view):
        """Put the viewer back where it was, after the data is in place.

        Last of all, because loading resets the camera to the whole field and
        rebuilding the layers renumbers the slider - both of which would undo
        this if it ran any earlier.
        """
        if not view:
            return
        dims = getattr(self.viewer, "dims", None)
        if dims is not None:
            try:
                if "ndisplay" in view:
                    dims.ndisplay = int(view["ndisplay"])
                step = view.get("current_step")
                if step:
                    # The stack may be shorter than it was - a different binning
                    # factor, or a truncated file - so every axis is clamped to
                    # what exists now rather than trusted.
                    limits = getattr(dims, "nsteps", None)
                    for axis, value in enumerate(step):
                        if axis >= len(dims.current_step):
                            break
                        if limits is not None and axis < len(limits):
                            value = min(int(value), max(int(limits[axis]) - 1, 0))
                        dims.set_current_step(axis, int(value))
            except (AttributeError, TypeError, ValueError, IndexError):
                pass
        camera = getattr(self.viewer, "camera", None)
        stored = view.get("camera") or {}
        if camera is not None and stored:
            try:
                camera.center = tuple(stored["center"])
                camera.zoom = float(stored["zoom"])
                camera.angles = tuple(stored.get("angles", camera.angles))
            except (AttributeError, TypeError, ValueError, KeyError):
                pass
        for name, visible in (view.get("layer_visibility") or {}).items():
            try:
                if name in self.viewer.layers:
                    self.viewer.layers[name].visible = bool(visible)
            except (AttributeError, TypeError):
                pass
        self._restore_layer_display(view.get("layer_display"))
        tab = view.get("active_tab")
        if tab is not None and hasattr(self, "tabs"):
            try:
                self.tabs.setCurrentIndex(int(tab))
            except (TypeError, ValueError):
                pass

    def _session_manifest(self, session_path):
        """Everything needed to arrive back here, minus anything reproducible."""
        session_dir = Path(session_path).parent
        csv_path = self.csv_edit.text().strip()
        image_path = self.image_edit.text().strip()
        has_csv_on_disk = bool(csv_path) and Path(csv_path).is_file()
        # Localizations that exist only in memory came from fitting in this
        # session and were never written out. Re-fitting them costs minutes, so
        # they travel with the session; everything else is a pointer.
        writes_locs = self.df is not None and not has_csv_on_disk
        locs_record = (session_io.source_record(session_io.locs_path_for(session_path), session_dir)
                       if writes_locs
                       else session_io.source_record(csv_path, session_dir))

        tracks_path = getattr(self, "_tracks_source_path", None)
        has_tracks = self.tracks is not None and not self.tracks.empty
        return {
            session_io.SESSION_KEY: session_io.SESSION_FORMAT,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "sources": {
                "image": session_io.source_record(image_path, session_dir),
                "localizations": locs_record,
                "trajectories": (session_io.source_record(tracks_path, session_dir)
                                 if has_tracks and tracks_path else None),
                # Recorded even though it is found on its own: it may have been
                # chosen by hand, and a restore should not quietly pick another.
                "drift": session_io.source_record(
                    self._drift_record.path if self._drift_record is not None else None,
                    session_dir),
            },
            "localizations_saved_with_session": bool(writes_locs),
            # The same dict the exporter writes, so one restore path serves both
            # and a setting can never be recorded by one and forgotten by the other.
            "settings": self._collect_metadata(csv_path),
            "rebuild": {
                # Only if they were linked here: trajectories read from a file
                # were not necessarily produced by these linking parameters, and
                # re-linking would quietly replace them with different ones.
                "link": bool(has_tracks and not tracks_path),
                "diffusion": bool(self._track_diffusion_cache),
                "populations": self._population_fit is not None,
                "render_image": self._render_image is not None,
                # every render in the viewer, bottom to top, each with its recipe
                "renders": [dict(layer.metadata[RENDER_RECIPE_KEY])
                            for layer in self.viewer.layers
                            if RENDER_RECIPE_KEY in (getattr(layer, "metadata", None) or {})],
                "extras": self._session_extras(),
            },
            "view": self._capture_session_view(),
            # Checked after the restore. A source file edited since the session
            # was saved rebuilds into something else entirely, and the count is
            # the cheapest way to notice.
            "expected": {
                "localizations": int(len(self.df)) if self.df is not None else 0,
                "filtered": int(len(self.df_filtered)) if self.df_filtered is not None else 0,
                "trajectories": int(self.tracks["particle"].nunique()) if has_tracks else 0,
            },
        }

    def save_session(self, path=None):
        """Write the whole working state to a session file."""
        if self.df is None and not self.image_edit.text().strip():
            self.log("Nothing to save yet - load an image or some localizations first.")
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save session", "", filter=session_io.SESSION_FILTER)
            if not path:
                return None
        session_path = session_io.session_path_for(path)

        try:
            manifest = self._session_manifest(session_path)
        except Exception as exc:
            self.log(f"Could not build the session: {exc}")
            return None

        # As fitted, not as corrected: the restore applies the drift afresh from
        # the settings, and this is the table those settings apply to.
        table = self._df_source if self._df_source is not None else self.df
        locs_frame = table if manifest["localizations_saved_with_session"] else None
        locs_path = session_io.locs_path_for(session_path)
        if locs_frame is not None:
            self.log(f"These {len(locs_frame)} localizations are not on disk anywhere "
                     f"else, so they are being saved with the session...")

        self.save_session_button.setEnabled(False)
        worker = _session_save_worker(session_path, manifest, locs_frame, locs_path)
        worker.returned.connect(self._on_session_saved)
        worker.errored.connect(lambda exc: self.log(f"Saving the session failed: {exc}"))
        worker.finished.connect(
            lambda: self.save_session_button.setEnabled(True))
        self._session_save_worker_ref = worker
        worker.start()
        return session_path

    def _on_session_saved(self, result):
        path, written = result
        self.log(f"Session saved to {path.name} ({written / 1024:.0f} kB) - "
                 f"it points at the data rather than copying it, and rebuilds "
                 f"the analysis on load.")

    def load_session(self, path=None):
        """Restore a session: apply its settings, then re-run the pipeline."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load session", "", filter=session_io.SESSION_FILTER)
            if not path:
                return None
        path = Path(path)
        try:
            manifest = session_io.read_session(path)
        except ValueError as exc:
            self.log(f"{path.name} {exc}")
            return None

        session_dir = path.parent
        missing = session_io.missing_sources(manifest, session_dir)
        for name, where in missing:
            self.log(f"The {name} this session refers to is not where it was saved: {where}")

        self.log(f"Loading session {path.name} ({session_io.describe_session(manifest)})")

        # The stack is about to be re-opened from scratch, so the handle held
        # for re-binning is stale; clearing it stops the restored binning factor
        # from re-binning the *previous* session's stack on its way past.
        self._raw_image = None
        # Likewise the drift record: the restored drift settings would
        # otherwise be applied, on their way past, with the previous data's.
        self._read_drift_files(None)
        applied, skipped, notes = self.apply_settings(manifest.get("settings") or {})
        self.log(f"Restored {len(applied)} settings")
        for note in notes:
            self.log(f"  note: {note}")
        if skipped:
            self.log(f"  {len(skipped)} setting(s) not applied: {', '.join(sorted(skipped)[:6])}")

        sources = manifest.get("sources") or {}
        image = session_io.resolve_source(sources.get("image"), session_dir)
        locs = session_io.resolve_source(sources.get("localizations"), session_dir)
        self.image_edit.setText(str(image) if image else "")
        self.csv_edit.setText(str(locs) if locs else "")

        rebuild = manifest.get("rebuild") or {}
        recipes = [r for r in (rebuild.get("renders") or []) if isinstance(r, dict)]
        extras = rebuild.get("extras") or {}
        steps = ["link", "diffusion", "populations"]
        if recipes:
            # every render there was, in its order, then the selection as saved
            steps += [("render", recipe) for recipe in recipes] + ["selection"]
        else:
            steps.append("render")          # a session from before renders were recorded
        if extras.get("wl_overlay"):
            steps.append("wl_overlay")
        steps += [("average", entry) for entry in (extras.get("averages") or [])]
        if extras.get("merged_layers"):
            steps.append("merged_layers")
        if extras.get("profile_line"):
            steps.append(("profile", extras["profile_line"]))
        steps.append("view")
        self._session_restore = {
            "manifest": manifest,
            "session_dir": session_dir,
            "name": path.name,
            "steps": steps,
        }
        if image or locs:
            self.load_data()
            if self._load_worker_ref is not None:
                return manifest      # the chain resumes when the load finishes
        else:
            self.log("This session records no data files, so only its settings were restored.")
        self._session_advance()
        return manifest

    def _session_advance(self):
        """Run the next restore step, or finish. Called as each worker ends.

        Every step here is asynchronous, so the sequence cannot be a function:
        it is a queue that each completed worker pushes along. A step that turns
        out to have nothing to do falls through to the next one in the same
        pass rather than stalling the chain waiting for a worker that was never
        started.
        """
        plan = self._session_restore
        if plan is None:
            return
        while plan["steps"]:
            step = plan["steps"].pop(0)
            if step == "link" and self._session_relink_wanted(plan):
                self.link_tracks()
                if self._link_worker_ref is not None:
                    return
            elif step == "diffusion" and self._session_diffusion_wanted(plan):
                self.compute_d()
                if self._compute_d_worker_ref is not None:
                    return
            elif step == "populations" and self._session_populations_wanted(plan):
                self.fit_populations()
                if self._population_worker_ref is not None:
                    return
            elif step == "render" and self._session_render_wanted(plan):
                self.render_smlm_image()
                if self._render_worker_ref is not None:
                    return
            elif isinstance(step, tuple) and step[0] == "render":
                if self._replay_render(step[1]) and self._render_worker_ref is not None:
                    return
            elif step == "selection":
                self._restore_saved_selection(plan)
            elif step == "wl_overlay":
                self.show_wl_overlay()
            elif isinstance(step, tuple) and step[0] == "average":
                if self._replay_average(step[1]) and self._average_worker_ref is not None:
                    return
            elif step == "merged_layers":
                self.show_merged_layers()
            elif isinstance(step, tuple) and step[0] == "profile":
                self._replay_profile(step[1])
            elif step == "view":
                self._restore_session_view(plan["manifest"].get("view") or {})
        self._finish_session_restore()

    def _replay_render(self, recipe):
        """Make one recorded render again, into the layer it was in. False if
        there is nothing to render it from."""
        if self.df_filtered is None or self.df_filtered.empty:
            return False
        settings = {k: v for k, v in recipe.items() if k not in ("render_kind", "layer_name")}
        # the recipe's own selection, from scratch: no class, then its filters
        self._population_class = None
        self.apply_settings(settings, include_instrument=False)
        self.render_add_layer_box.setChecked(True)
        self._apply_track_filter()
        self.log(f"Session: rendering '{recipe.get('layer_name', '?')}' again...")
        if recipe.get("render_kind") == "movie":
            self.render_smlm_movie()
        else:
            self.render_smlm_image()
        return True

    def _restore_saved_selection(self, plan):
        """After the renders, the selection and render settings as they were saved."""
        saved = plan["manifest"].get("settings") or {}
        subset = {section: saved[section] for section in ("smlm_rendering",) + FILTER_SECTIONS
                  if section in saved}
        if "diffusion" in saved:
            subset["diffusion"] = saved["diffusion"]
        self._population_class = None
        self.apply_settings(subset, include_instrument=False)
        self._apply_track_filter()

    def _replay_average(self, entry):
        kind = entry.get("kind")
        params = entry.get("params") or {}
        if kind == "fluorescence":
            self.average_first_box.setValue(int(params.get("first", 0)))
            self.average_last_box.setValue(int(params.get("last", -1)))
            self.average_every_box.setValue(int(params.get("every", 1)))
            self.average_fluorescence()
        elif kind == "white light":
            self.average_white_light()
        else:
            return False
        return True

    def _replay_profile(self, line):
        try:
            self.start_profile_line()
            layer = self._profile_layer()
            layer.add_lines(np.array([layer.world_to_data(line[0]), layer.world_to_data(line[1])]))
            layer.mode = "pan_zoom"
            self.update_profile()
        except Exception as exc:
            self.log(f"Session: the line profile could not be drawn again: {exc}")

    def _session_relink_wanted(self, plan):
        if not (plan["manifest"].get("rebuild") or {}).get("link"):
            return False
        if self.tracks is not None and not self.tracks.empty:
            return False           # a recorded trajectories file was loaded instead
        return self.df_filtered is not None and not self.df_filtered.empty

    def _session_diffusion_wanted(self, plan):
        if not (plan["manifest"].get("rebuild") or {}).get("diffusion"):
            return False
        return self.tracks is not None and not self.tracks.empty

    def _session_populations_wanted(self, plan):
        if not (plan["manifest"].get("rebuild") or {}).get("populations"):
            return False
        return self.tracks is not None and not self.tracks.empty

    def _session_render_wanted(self, plan):
        if not (plan["manifest"].get("rebuild") or {}).get("render_image"):
            return False
        return self.df_filtered is not None and not self.df_filtered.empty

    def _finish_session_restore(self):
        plan, self._session_restore = self._session_restore, None
        if plan is None:
            return
        expected = (plan["manifest"].get("expected") or {})
        actual = {
            "localizations": int(len(self.df)) if self.df is not None else 0,
            "filtered": int(len(self.df_filtered)) if self.df_filtered is not None else 0,
            "trajectories": (int(self.tracks["particle"].nunique())
                             if self.tracks is not None and not self.tracks.empty else 0),
        }
        # A source edited since the session was saved rebuilds into something
        # else, and a session that says it restored a state it did not reach is
        # worse than one that admits it.
        differences = [f"{name}: {expected[name]} then, {actual[name]} now"
                       for name in expected
                       if int(expected.get(name, 0)) != actual.get(name, 0)]
        if differences:
            self.log("Session restored, but not to the same numbers - "
                     + "; ".join(differences))
        else:
            self.log(f"Session {plan['name']} restored.")
        self._update_status_header()

    def apply_settings(self, metadata, include_instrument=True):
        """Apply a metadata dict to the controls. Returns (applied, skipped, notes).

        `include_instrument` is False when the settings were not asked for -
        when opening data happens to find a previous run beside it. See
        `_restore_previous_run_settings` for why the microscope is left alone
        on that path.
        """
        values, notes = settings_from_metadata(metadata)
        applied, skipped = [], []
        # Read before anything moves, compared after, so the log can say which
        # instrument parameters this file changed and what they were before.
        before = {attr: getattr(self, attr).value()
                  for attr, _label, _fmt in INSTRUMENT_SETTINGS
                  if hasattr(self, attr)}
        if not include_instrument:
            for attr, label, template in INSTRUMENT_SETTINGS:
                if attr not in values or attr not in before:
                    continue
                recorded = values.pop(attr)
                try:
                    differs = abs(float(recorded) - before[attr]) > 1e-9
                except (TypeError, ValueError):
                    differs = False
                if differs:
                    notes.append(
                        f"that run used {label.lower()} {template.format(float(recorded))}; "
                        f"yours is {template.format(before[attr])} and was left alone")

        # The frame shift is plugin state rather than a control, so it is
        # applied directly instead of being pushed into a widget.
        if "_frame_shift" in values:
            self._frame_shift = int(values.pop("_frame_shift"))
            self._update_frame_shift_label()
            applied.append("_frame_shift")

        population = values.pop("_population_class", None)
        for attr, value in values.items():
            widget = getattr(self, attr, None)
            if widget is None:
                skipped.append(attr)
                continue
            try:
                if set_widget_value(widget, value):
                    applied.append(attr)
                else:
                    skipped.append(attr)
            except Exception:
                skipped.append(attr)

        # After the filter boxes, whose toggles clear any class a preset had
        # chosen: this one is part of the selection being restored.
        if population is not None:
            self._population_class = population
            self._invalidate_track_filter()
            applied.append("_population_class")

        bounds = metadata.get("filter_bounds") if isinstance(metadata, dict) else None
        if isinstance(bounds, dict):
            n_bounds, unmatched = self._apply_filter_bounds(bounds)
            if n_bounds:
                notes.append(f"{n_bounds} filter bound(s) applied")
            if unmatched:
                names = ", ".join(sorted(unmatched))
                if not self.filter_controls:
                    # Settings come before the table they belong to: nothing can
                    # match yet, and they are applied the moment it loads.
                    notes.append(f"{len(unmatched)} filter bound(s) kept for when the "
                                 f"localizations are loaded: {names}")
                else:
                    notes.append(f"{len(unmatched)} filter bound(s) match no column of this "
                                 f"table: {names} - kept in case matching data is loaded next")

        self._apply_histogram_display(metadata.get("metric_histogram_display"),
                                      self._metric_hist_widgets, notes)
        self._apply_histogram_display(metadata.get("filter_histogram_display"),
                                      self._hist_widgets, notes)

        # A restored binning factor arrives alongside a camera baseline and a
        # frame rate that already account for it, so the stack is re-binned to
        # match but those two are left exactly as the file set them - rescaling
        # them here would apply the factor a second time.
        if "bin_factor_box" in applied:
            self._time_bin_timer.stop()
            factor = int(self.bin_factor_box.value())
            if factor != self._time_bin_applied:
                self._time_bin_applied = factor
                self._update_time_bin_label()
                if self._raw_image is not None:
                    self._rebin_loaded_stack(factor)

        for attr, label, template in INSTRUMENT_SETTINGS:
            if attr not in before:
                continue
            after = getattr(self, attr).value()
            if abs(after - before[attr]) <= 1e-9:
                continue
            notes.append(f"{label} {template.format(before[attr])} -> "
                         f"{template.format(after)}, as that run recorded it")

        # One refresh at the end rather than one per control.
        self.apply_filters()
        self._update_link_cutoff_label()
        self.render_overlay()
        return applied, skipped, notes

    def _apply_filter_bounds(self, bounds):
        """Apply per-column filter bounds; stash the ones whose column isn't loaded.

        Settings are often loaded before the data they belong to, and the filter
        controls only exist once a table is in. Anything that cannot be applied
        now is kept and applied when a matching column shows up.
        """
        applied = 0
        pending = {}
        for column, limits in bounds.items():
            if not isinstance(limits, dict):
                continue
            controls = self.filter_controls.get(column)
            if controls is None:
                pending[column] = limits
                continue
            lower_box, upper_box = controls
            try:
                if isinstance(limits.get("min"), (int, float)):
                    lower_box.setValue(float(limits["min"]))
                if isinstance(limits.get("max"), (int, float)):
                    upper_box.setValue(float(limits["max"]))
                applied += 1
            except Exception:
                pending[column] = limits
        self._pending_filter_bounds = pending or None
        return applied, list(pending)

    def _apply_histogram_display(self, section, widgets, notes):
        if not isinstance(section, dict):
            return
        for key, entry in section.items():
            state = widgets.get(key)
            if state is None or not isinstance(entry, dict):
                continue
            try:
                if isinstance(entry.get("bins"), (int, float)):
                    state["bins_box"].setValue(int(entry["bins"]))
                follow = state.get("follow_box")
                if follow is not None and isinstance(entry.get("follow_filter"), bool):
                    follow.setChecked(entry["follow_filter"])
                log = state.get("log_box")
                if log is not None and isinstance(entry.get("log_scale"), bool):
                    log.setChecked(entry["log_scale"])
                # While the view follows the filter it is derived, not stored.
                if follow is None or not follow.isChecked():
                    for name, box in (("view_min", "view_min_box"), ("view_max", "view_max_box")):
                        if isinstance(entry.get(name), (int, float)):
                            state[box].setValue(float(entry[name]))
            except Exception:
                notes.append(f"could not restore the {key} histogram view")

    # ------------------------------------------------------------------
    # Acquisition timing: frame rate and frame interval are one setting
    # ------------------------------------------------------------------
    def _frame_interval_s(self):
        return 1.0 / max(self.fps_box.value(), 1e-9)

    def _on_fps_changed(self, fps):
        if self._syncing_timing:
            return
        self._syncing_timing = True
        try:
            self.frame_interval_box.setValue(1000.0 / max(float(fps), 1e-9))
        finally:
            self._syncing_timing = False
        self._update_link_cutoff_label()

    def _on_frame_interval_changed(self, interval_ms):
        if self._syncing_timing:
            return
        self._syncing_timing = True
        try:
            self.fps_box.setValue(1000.0 / max(float(interval_ms), 1e-9))
        finally:
            self._syncing_timing = False
        self._update_link_cutoff_label()

    def _update_link_cutoff_label(self, *_args):
        """Live readout of the largest D the current linking parameters can follow."""
        # The Link tab is built before the D tab, so the timing boxes may not exist yet.
        if not hasattr(self, "link_cutoff_label") or not hasattr(self, "fps_box"):
            return
        search_nm = self.search_box.value()
        interval_s = self._frame_interval_s()
        memory = self.memory_box.value()
        percent = DEFAULT_LINKING_ERROR_RATE * 100

        d_max = max_linkable_diffusion(search_nm, interval_s, memory=0)
        step = rms_step(d_max, interval_s)
        lines = [
            f"Links D up to <b>{d_max:.3g} µm²/s</b> "
            f"(RMS step {step:.0f} nm; {percent:g}% of steps exceed {search_nm:.0f} nm "
            f"at Δt = {interval_s * 1000:.3g} ms)"
        ]
        if memory > 0:
            d_gap = max_linkable_diffusion(search_nm, interval_s, memory=memory)
            lines.append(
                f"With memory {memory} a gap spans {(memory + 1) * interval_s * 1000:.3g} ms, "
                f"so closing those gaps needs D ≤ {d_gap:.3g} µm²/s"
            )

        measured = self._track_diffusion_cache or {}
        if measured:
            values = np.asarray(list(measured.values()), float)
            values = values[np.isfinite(values)]
            if values.size:
                over = float((values > d_max).mean() * 100)
                verdict = "search range looks adequate" if over <= percent else "consider a larger search range"
                lines.append(
                    f"{over:.1f}% of your {values.size} measured D values exceed it - {verdict}"
                )
        self.link_cutoff_label.setText("<br>".join(lines))

    # ------------------------------------------------------------------
    # Cancelling long-running operations
    # ------------------------------------------------------------------
    def _cancel_button(self, event, label):
        """A Cancel button wired to `event`, enabled only while work is running."""
        button = QPushButton("Cancel")
        button.setProperty("stop", True)
        button.setEnabled(False)
        button.setToolTip(f"Stop {label} at the next frame boundary")
        button.clicked.connect(lambda: self._request_cancel(event, button, label))
        return button

    def _request_cancel(self, event, button, label):
        event.set()
        button.setEnabled(False)
        self.log(f"Cancelling {label}...")

    def _arm_cancel(self, event, button):
        event.clear()
        button.setEnabled(True)

    def _build_load_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Load")
        # Scrolls, like Localize, so a short laptop screen still reaches the
        # playback settings at the bottom.
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(tab)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        data_group = QGroupBox("Data")
        data_layout = QFormLayout(data_group)
        self.csv_edit = QLineEdit()
        self.csv_button = QPushButton("Browse CSV")
        self.csv_button.setProperty("secondary", True)
        self.csv_button.clicked.connect(self.browse_csv)
        csv_row = QHBoxLayout()
        csv_row.addWidget(self.csv_edit)
        csv_row.addWidget(self.csv_button)
        data_layout.addRow("Localization CSV", csv_row)

        # Right under the CSV, because it is a property of that file: some
        # software numbers the first frame 0 and some numbers it 1, and a
        # table that disagrees with its image stack puts every localization on
        # the wrong frame. Shifting is explicit and reversible - press until
        # the localizations sit on their spots.
        shift_row = QHBoxLayout()
        self.frame_shift_down_button = QPushButton("-1")
        self.frame_shift_down_button.setProperty("secondary", True)
        self.frame_shift_down_button.setToolTip("Shift every frame number down by one")
        self.frame_shift_down_button.clicked.connect(lambda: self.shift_frame_numbers(-1))
        self.frame_shift_up_button = QPushButton("+1")
        self.frame_shift_up_button.setProperty("secondary", True)
        self.frame_shift_up_button.setToolTip("Shift every frame number up by one")
        self.frame_shift_up_button.clicked.connect(lambda: self.shift_frame_numbers(+1))
        self.frame_shift_reset_button = QPushButton("Reset")
        self.frame_shift_reset_button.setProperty("secondary", True)
        self.frame_shift_reset_button.clicked.connect(lambda: self.shift_frame_numbers(None))
        self.frame_shift_label = QLabel("no shift")
        shift_row.addWidget(self.frame_shift_down_button)
        shift_row.addWidget(self.frame_shift_up_button)
        shift_row.addWidget(self.frame_shift_reset_button)
        shift_row.addWidget(self.frame_shift_label, 1)
        data_layout.addRow("Shift frame numbers", shift_row)

        self.image_edit = QLineEdit()
        self.image_button = QPushButton("Browse image")
        self.image_button.setProperty("secondary", True)
        self.image_button.clicked.connect(self.browse_image)
        image_row = QHBoxLayout()
        image_row.addWidget(self.image_edit)
        image_row.addWidget(self.image_button)
        data_layout.addRow("Image", image_row)

        # Time binning. A dim emitter spread over several frames is often below
        # the detection threshold in every one of them and comfortably above it
        # in their sum, so this is the one preprocessing step worth having in
        # front of the fit. It costs time resolution, so it is off by default.
        bin_row = QHBoxLayout()
        self.bin_factor_box = QSpinBox()
        self.bin_factor_box.setRange(1, TIME_BIN_MAX)
        self.bin_factor_box.setValue(1)
        self.bin_factor_box.setSuffix(" raw frames")
        self.bin_factor_box.setToolTip(
            "Sum every N consecutive raw frames into one before detection and "
            "fitting.\n\n"
            "Summed, not averaged, so the result still obeys the photon "
            "statistics the fit assumes. The camera baseline and the frame rate "
            "are adjusted to match: a binned frame carries N baselines and lasts "
            "N exposures.\n\n"
            "Trades time resolution for signal - it will merge blinks and blur "
            "anything moving faster than N frames."
        )
        self.bin_label = QLabel("off")
        bin_row.addWidget(self.bin_factor_box)
        bin_row.addWidget(self.bin_label, 1)
        data_layout.addRow("Time binning", bin_row)
        self._time_bin_timer = QTimer(self)
        self._time_bin_timer.setSingleShot(True)
        self._time_bin_timer.setInterval(TIME_BIN_DEBOUNCE_MS)
        self._time_bin_timer.timeout.connect(self._apply_time_binning)
        self.bin_factor_box.valueChanged.connect(
            lambda _v: self._time_bin_timer.start())

        self.pixel_size_box = QDoubleSpinBox()
        self.pixel_size_box.setRange(1.0, 10000.0)
        self.pixel_size_box.setDecimals(2)
        self.pixel_size_box.setValue(DEFAULT_PIXEL_SIZE_NM)
        self.pixel_size_box.setToolTip(
            "Camera pixel size in the sample plane. Everything physical - the search range, D, the scale bar, the super-resolved pixel - is derived from it, so it is worth getting right."
        )
        self.pixel_size_box.setSuffix(" nm/px")
        data_layout.addRow("Pixel size", self.pixel_size_box)


        self.load_button = QPushButton("Load data")
        self.load_button.setProperty("primary", True)
        self.load_button.clicked.connect(self.load_data)
        self.load_cancel_button = self._cancel_button(self._load_cancel, "loading")
        load_row = QHBoxLayout()
        load_row.addWidget(self.load_button)
        load_row.addWidget(self.load_cancel_button)
        data_layout.addRow("", load_row)

        # A session is the whole working state, not just the numbers in the
        # boxes: which files, which parameters, what had been computed, and
        # where the viewer was looking. It stays small by pointing at the data
        # instead of copying it and by rebuilding the analysis on load.
        session_row = QHBoxLayout()
        self.save_session_button = QPushButton("Save session...")
        self.save_session_button.setProperty("secondary", True)
        self.save_session_button.setToolTip(
            "Write the whole working state to a small session file: the data "
            "paths, every parameter, and what had been computed.\n\n"
            "The raw stack is never copied. Trajectories, diffusion "
            "coefficients and reconstructions are rebuilt on load from the "
            "parameters that produced them, so a session is a few kilobytes.\n\n"
            "The exception is localizations fitted here and never saved "
            "anywhere - those are written beside the session, gzipped, because "
            "re-fitting them costs minutes."
        )
        self.save_session_button.clicked.connect(lambda: self.save_session())
        self.load_session_button = QPushButton("Load session...")
        self.load_session_button.setProperty("secondary", True)
        self.load_session_button.setToolTip(
            "Reopen a saved session: restore every parameter, reload the data, "
            "then re-run linking, diffusion and rendering to arrive back where "
            "it was left."
        )
        self.load_session_button.clicked.connect(lambda: self.load_session())
        session_row.addWidget(self.save_session_button)
        session_row.addWidget(self.load_session_button)
        data_layout.addRow("Session", session_row)

        self.load_settings_button = QPushButton("Load settings from a previous analysis or session...")
        self.load_settings_button.setProperty("secondary", True)
        self.load_settings_button.clicked.connect(lambda: self.load_settings_from_metadata())
        self.load_settings_button.setToolTip(
            "Read the metadata.json written by a previous export, or a session\n"
            "file, and restore the parameters it recorded: camera, detection,\n"
            "fitting, linking, diffusion, filter bounds and display settings.\n\n"
            "Only the filters? The Filter tab's 'Load filters from...' takes\n"
            "just those.\n\n"
            "Only settings are restored - never the data, the file paths, or the\n"
            "results of that run. Filter bounds for columns that are not loaded yet\n"
            "are kept and applied when matching data arrives, so settings can be\n"
            "loaded before the data."
        )
        data_layout.addRow("", self.load_settings_button)
        self.load_progress = QProgressBar()
        self.load_progress.setRange(0, 0)  # indeterminate
        self.load_progress.setVisible(False)
        data_layout.addRow("", self.load_progress)
        layout.addWidget(data_group)

        display_group = QGroupBox("Localization display")
        display_layout = QFormLayout(display_group)
        self.show_points_box = QCheckBox("Show localizations")
        self.show_points_box.setChecked(True)
        display_layout.addRow("", self.show_points_box)
        self.marker_size_box = QDoubleSpinBox()
        self.marker_size_box.setRange(1.0, 20.0)
        self.marker_size_box.setValue(6.0)
        display_layout.addRow("Marker size", self.marker_size_box)
        self.marker_edge_width_box = QDoubleSpinBox()
        self.marker_edge_width_box.setRange(0.01, 1.0)
        self.marker_edge_width_box.setSingleStep(0.05)
        self.marker_edge_width_box.setDecimals(2)
        self.marker_edge_width_box.setValue(0.1)
        display_layout.addRow("Marker edge width (relative)", self.marker_edge_width_box)
        self.marker_choice = QComboBox()
        self.marker_choice.addItems(["o", "s", "+", "x", "D"])
        self.marker_choice.setCurrentText("o")
        display_layout.addRow("Marker type", self.marker_choice)
        layout.addWidget(display_group)

        playback_group = QGroupBox("Playback")
        playback_layout = QFormLayout(playback_group)
        playback_note = QLabel(
            "How fast napari's play button (▶, next to the frame slider) "
            "runs the stack. Set it here, then screen-record the viewer to make "
            "a movie with every layer exactly as it looks."
        )
        playback_note.setWordWrap(True)
        playback_layout.addRow(playback_note)

        # Whole frames per second: napari's playback_fps is an int, and offering
        # a fractional one here would only produce a setting it refuses.
        self.playback_fps_box = QSpinBox()
        self.playback_fps_box.setRange(1, 1000)
        self.playback_fps_box.setValue(_playback_fps())
        self.playback_fps_box.setSuffix(" fps")
        self.playback_fps_box.setToolTip(
            "Frames of the stack shown per second of wall clock. napari plays "
            "at whole frames per second, so this is rounded."
        )
        self.playback_realtime_button = QPushButton("Real time")
        self.playback_realtime_button.setProperty("secondary", True)
        self.playback_realtime_button.setToolTip(
            "Play at the rate the camera acquired at, so a second on screen is "
            "a second at the microscope."
        )
        self.playback_realtime_button.clicked.connect(self._set_playback_to_real_time)
        fps_row = QHBoxLayout()
        fps_row.addWidget(self.playback_fps_box, 1)
        fps_row.addWidget(self.playback_realtime_button)
        playback_layout.addRow("Speed", fps_row)

        self.playback_mode_box = QComboBox()
        for key, label in PLAYBACK_MODES.items():
            self.playback_mode_box.addItem(label, key)
        self.playback_mode_box.setCurrentIndex(
            max(0, self.playback_mode_box.findData(_playback_mode())))
        playback_layout.addRow("At the end", self.playback_mode_box)

        self.playback_status = QLabel("-")
        self.playback_status.setWordWrap(True)
        playback_layout.addRow("", self.playback_status)
        layout.addWidget(playback_group)

        self.playback_fps_box.valueChanged.connect(self._on_playback_changed)
        self.playback_mode_box.currentIndexChanged.connect(
            lambda _i: self._on_playback_changed())
        self._on_playback_changed()

        layout.addStretch(1)

        # Marker style only concerns the points layer, which updates in place -
        # no reason to rebuild the trajectory layers as well.
        self.show_points_box.stateChanged.connect(lambda _checked: self._sync_points_layer())
        self.marker_size_box.valueChanged.connect(lambda _v: self._sync_points_layer())
        self.marker_edge_width_box.valueChanged.connect(lambda _v: self._sync_points_layer())
        self.marker_choice.currentTextChanged.connect(lambda _v: self._sync_points_layer())

    def _build_drift_group(self):
        """Drift correction from the white-light camera's record.

        The record and the frame clock both sit beside the stack and are picked
        up on loading, and so is the map between the two cameras - the record's
        own, or the calibrated one - so the group is mostly a report of what was
        found and what it did, plus the one choice left: how much to smooth.
        """
        group = QGroupBox("Drift correction (white-light camera)")
        form = QFormLayout(group)

        self.drift_edit = QLineEdit()
        self.drift_edit.setPlaceholderText("picked up from beside the stack on loading")
        self.drift_edit.setToolTip(
            "The <name>_xy_drift.csv the acquisition writes: the sample's drift, "
            "measured on the white-light camera throughout the run.\n\n"
            "Its frames are placed in it by <stack>_frame_times.csv, which the "
            "acquisition writes beside the stack. Both are found when the data "
            "is loaded; browse only to use a different record."
        )
        self.drift_edit.editingFinished.connect(self._on_drift_path_edited)
        self.drift_button = QPushButton("Browse")
        self.drift_button.setProperty("secondary", True)
        self.drift_button.clicked.connect(self.browse_drift)
        drift_row = QHBoxLayout()
        drift_row.addWidget(self.drift_edit)
        drift_row.addWidget(self.drift_button)
        form.addRow("Drift record", drift_row)

        self.drift_map_label = QLabel("No drift record loaded")
        self.drift_map_label.setWordWrap(True)
        self.drift_map_label.setToolTip(
            "How a drift measured on the white-light camera becomes a drift on "
            "the fluorescence camera. Its pixels are about half the size, and it "
            "sits turned 0.8 degrees from the fluorescence camera, so the drift "
            "goes through a 2x2 matrix rather than a ratio of pixel sizes - and "
            "then into nanometres with the pixel size above, the one the "
            "localizations were computed with.\n\n"
            "Recent records carry the matrix the acquisition held; older ones "
            "are read with the calibration of 2026-09-25 (Argo-SIM v2 slide), "
            "which holds for white-light frames oriented 'Rotate 180', 2048 x "
            "2048. A record taken otherwise is not corrected: a map for another "
            "orientation would move every localization the wrong way."
        )
        form.addRow("Camera map", self.drift_map_label)

        self.drift_mode_box = QComboBox()
        self.drift_mode_box.addItem("Drift: one translation per frame (first frame's coordinates)", "drift")
        self.drift_mode_box.addItem("Growth: every position carried to one geometry (below)", "growth")
        self.drift_mode_box.setToolTip(
            "Drift takes one translation per frame out of every localization - "
            "right for a sample that only moves.\n\n"
            "A growing root also stretches: several percent along its axis over "
            "twenty minutes, which a translation leaves as micrometres of "
            "misplacement across the field. Growth carries every localization to "
            "where its piece of tissue is in the last white-light snapshot, "
            "through the deformation measured on the snapshots (below), so "
            "everything is mapped onto the final, stretched geometry.\n\n"
            "Trajectories are linked in that geometry, but D, distances and the "
            "immobility test are measured with the local stretch taken back out: "
            "in the final geometry every step would be stretched by the growth "
            "still to come, and D inflated by its square.")
        self.drift_mode_box.currentIndexChanged.connect(lambda _i: self._refresh_drift_correction())
        form.addRow("Correct for", self.drift_mode_box)
        self.growth_reference_box = QComboBox()
        self.growth_reference_box.addItem("the first frame's (as the drift correction)", "first")
        self.growth_reference_box.addItem("the last snapshot's (stretched)", "last")
        self.growth_reference_box.setToolTip(
            "Where the growth correction carries everything: to where each piece of "
            "tissue was at the first frame - the geometry the drift correction and RCC "
            "use too, and the one the densest part of the movie, before the dyes "
            "bleach, is least moved into - or to where it is in the last snapshot, "
            "fully stretched.\n\n"
            "Either way D, distances and the immobility test are measured with the "
            "local stretch taken back out.")
        self.growth_reference_box.currentIndexChanged.connect(
            lambda _i: self._refresh_drift_correction())
        form.addRow("Geometry", self.growth_reference_box)

        growth_row = QHBoxLayout()
        self.growth_measure_button = QPushButton("Measure growth deformation")
        self.growth_measure_button.setProperty("secondary", True)
        self.growth_measure_button.setToolTip(
            "Register the white-light snapshots against the last one, patch by "
            "patch at cell-wall scale, over the fluorescence camera's field, and "
            "fit the growth: translation, turn, stretch along and across the root "
            "axis, and a stretch that varies along it. Seconds on a GPU, a minute "
            "or so on the CPU; the result is saved in the acquisition's analysis "
            "folder and found again on loading.")
        self.growth_measure_button.clicked.connect(lambda: self.measure_growth())
        self.growth_cancel_button = self._cancel_button(self._deform_cancel, "the measurement")
        self.growth_show_button = QPushButton("Show it...")
        self.growth_show_button.setProperty("secondary", True)
        self.growth_show_button.setToolTip(
            "Open the white-light snapshots in a viewer of their own, with the growth: "
            "arrows of the growth still to come on every snapshot, every snapshot "
            "carried into the final geometry at each stage of the measurement (a "
            "'stage' slider beside the snapshot one), the last snapshot to lay them "
            "on, what each pass still found off at each patch - and the growth's "
            "numbers as plots. The same viewer as 'Show the snapshots...'.")
        self.growth_show_button.clicked.connect(lambda: self.show_wl_snapshots())
        self.growth_show_button.setEnabled(False)
        self.growth_movie_button = QPushButton("Movie file...")
        self.growth_movie_button.setProperty("secondary", True)
        self.growth_movie_button.setToolTip(
            "The same as a movie file, saved beside the deformation.\n\n"
            "First how the map was found: the first snapshot laid on the last, as "
            "recorded, after the rough chaining of neighbouring snapshots and after "
            "each refinement pass, with what each pass still found off at every "
            "patch.\n\n"
            "Then the time-lapse: each snapshot as recorded with arrows for the "
            "growth still to come, moved by the drift record alone, and carried "
            "through the growth map - where the growth is cancelled and the tissue "
            "stands still.")
        self.growth_movie_button.clicked.connect(lambda: self.make_growth_movie())
        self.growth_movie_button.setEnabled(False)
        growth_row.addWidget(self.growth_measure_button)
        growth_row.addWidget(self.growth_cancel_button)
        growth_row.addWidget(self.growth_show_button)
        growth_row.addWidget(self.growth_movie_button)
        growth_row.addStretch(1)
        form.addRow("", growth_row)
        self.growth_progress = QProgressBar()
        self.growth_progress.setRange(0, 100)
        self.growth_progress.setVisible(False)
        form.addRow("", self.growth_progress)
        self.growth_status = QLabel("No growth deformation measured")
        self.growth_status.setWordWrap(True)
        form.addRow("", self.growth_status)

        self.drift_smoothing_box = QDoubleSpinBox()
        self.drift_smoothing_box.setRange(0.0, 120.0)
        self.drift_smoothing_box.setDecimals(1)
        self.drift_smoothing_box.setSingleStep(0.5)
        self.drift_smoothing_box.setSuffix(" s")
        self.drift_smoothing_box.setSpecialValueText("none")
        self.drift_smoothing_box.setValue(DEFAULT_DRIFT_SMOOTHING_S)
        self.drift_smoothing_box.setToolTip(
            "Width (sigma) of the Gaussian window the drift record is smoothed "
            "over before it is applied, so that the tracker's measurement "
            "scatter is not added to every localization.\n\n"
            "A local straight-line fit rather than an average, and rather than "
            "averaging in time bins: both of those lag a steady drift and bend "
            "it at the start and end of the movie, which a line fit does not. "
            "It will round off a sudden jump over about this long.\n\n"
            "'none' interpolates the raw samples."
        )
        form.addRow("Smoothing", self.drift_smoothing_box)

        self.drift_enable_box = QCheckBox("Correct the localizations for drift")
        self.drift_enable_box.setChecked(True)
        self.drift_enable_box.setToolTip(
            "Subtract, from every localization, the drift of the frame it was "
            "found in, so all of them are in the coordinates of the first "
            "frame. Everything downstream - filters, linking, rendering, export "
            "- then works on the corrected positions.\n\n"
            "The table as loaded is kept: untick to go back to it. Exported "
            "tables carry the drift that was subtracted (drift_x/drift_y "
            "columns), so a corrected table is recognised if it is loaded again "
            "and is never corrected twice."
        )
        form.addRow("", self.drift_enable_box)
        self.drift_shift_image_box = QCheckBox("Shift the image to match (display only)")
        self.drift_shift_image_box.setChecked(True)
        self.drift_shift_image_box.setToolTip(
            "Show every frame of the stack moved back by its drift, so the "
            "corrected localizations sit on their spots.\n\n"
            "Display only: frames are shifted as they are shown, and detection "
            "and fitting still read them as the camera recorded them - the "
            "localizations are corrected afterwards, which is both sharper and "
            "the only way not to correct them twice."
        )
        form.addRow("", self.drift_shift_image_box)

        self.drift_status = QLabel("No drift record loaded")
        self.drift_status.setWordWrap(True)
        form.addRow("", self.drift_status)

        self.drift_figure = Figure(figsize=(5, 2.2))
        self.drift_canvas = FigureCanvas(self.drift_figure)
        self._plot_canvases.append(self.drift_canvas)
        self.drift_canvas.setMinimumHeight(200)
        form.addRow("", self.drift_canvas)
        drift_tools = QHBoxLayout()
        drift_tools.addStretch(1)
        drift_tools.addWidget(self._png_button(
            lambda fig, opts: self._draw_drift_plot(fig, opts), lambda: "drift"))
        form.addRow("", drift_tools)

        self._drift_timer = QTimer(self)
        self._drift_timer.setSingleShot(True)
        self._drift_timer.setInterval(DRIFT_DEBOUNCE_MS)
        self._drift_timer.timeout.connect(lambda: self._refresh_drift_correction())
        self.drift_smoothing_box.valueChanged.connect(lambda _v: self._drift_timer.start())
        self.drift_enable_box.toggled.connect(lambda _c: self._refresh_drift_correction())
        self.drift_shift_image_box.toggled.connect(lambda _c: self._apply_drift_to_image())
        # The drift is in nanometres, the image in camera pixels: a new pixel
        # size is a new place for every frame on screen, and - since the record
        # reaches nanometres through fluorescence pixels - a new drift to take
        # out of the localizations.
        self.pixel_size_box.valueChanged.connect(lambda _v: self._apply_drift_to_image())
        self.pixel_size_box.valueChanged.connect(lambda _v: self._drift_timer.start())
        return group

    def _build_drift_tab(self):
        """Everything about drift in one place: apply it, check it, refine it."""
        tab = QWidget()
        self._drift_tab_index = self.tabs.addTab(tab, "Drift")
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(tab)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        note = QLabel(
            "The white-light camera measures the drift ten times a second, "
            "whatever the sample is doing; the localizations measure it "
            "directly, but only as finely as there is static structure to "
            "correlate. The first is applied, checked against the white-light "
            "snapshots, and can then be refined by the second (RCC)."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addWidget(self._build_drift_group())
        layout.addWidget(self._build_wl_check_group())
        layout.addWidget(self._build_wl_overlay_group())
        layout.addWidget(self._build_rcc_group())
        layout.addStretch(1)

    def _build_wl_check_group(self):
        """Is the record right? Its own diagnostics, and the snapshots."""
        group = QGroupBox("Check the record")
        form = QFormLayout(group)

        self.drift_step_box = QDoubleSpinBox()
        self.drift_step_box.setRange(0.01, 100.0)
        self.drift_step_box.setDecimals(2)
        self.drift_step_box.setSingleStep(0.1)
        self.drift_step_box.setValue(DEFAULT_DRIFT_STEP_PX)
        self.drift_step_box.setSuffix(" white-light px")
        self.drift_step_box.setToolTip(
            "Report every jump between two consecutive drift samples larger "
            "than this. A sample drifts smoothly; a pixel in a tenth of a second "
            "is the stage, the focus lock or the sample lurching - or the tracker "
            "latching onto something else, which the snapshots below can tell "
            "apart. Smoothing rounds a jump off; it does not remove it."
        )
        self.drift_step_box.valueChanged.connect(lambda _v: self._update_wl_check_status())
        form.addRow("Report jumps over", self.drift_step_box)

        self.wl_check_status = QLabel("No drift record loaded")
        self.wl_check_status.setWordWrap(True)
        form.addRow("", self.wl_check_status)

        buttons = QHBoxLayout()
        self.wl_check_button = QPushButton("Check against the snapshots")
        self.wl_check_button.setProperty("primary", True)
        self.wl_check_button.setToolTip(
            "Register every white-light snapshot saved with the acquisition "
            "against the first, independently of the record, and compare.\n\n"
            "The snapshots are a sparse subset - one every so many seconds - so "
            "they cannot see between themselves; what they do show is whether "
            "the record's drift at their moments is the drift the images show. "
            "A jump the tracker invented appears as a step between two snapshots "
            "that the images do not share. The results are drawn as markers on "
            "the drift plot."
        )
        self.wl_check_button.clicked.connect(self.check_wl_snapshots)
        self.wl_check_cancel_button = self._cancel_button(self._wl_check_cancel, "the check")
        self.wl_show_button = QPushButton("Show the snapshots...")
        self.wl_show_button.setProperty("secondary", True)
        self.wl_show_button.setToolTip(
            "Open the snapshots in a viewer of their own: as recorded, and with "
            "the record's drift taken out. Toggle between the two layers - with "
            "the drift out the sample should stand still, snapshot after "
            "snapshot. The rectangle is the region the tracker followed.\n\n"
            "With a growth deformation measured, the growth comes too: its arrows, "
            "the snapshots with the growth cancelled at each stage of the "
            "measurement, and its numbers as plots."
        )
        self.wl_show_button.clicked.connect(self.show_wl_snapshots)
        buttons.addWidget(self.wl_check_button)
        buttons.addWidget(self.wl_check_cancel_button)
        buttons.addWidget(self.wl_show_button)
        buttons.addStretch(1)
        form.addRow("", buttons)
        self.wl_check_progress = QProgressBar()
        self.wl_check_progress.setRange(0, 100)
        self.wl_check_progress.setVisible(False)
        form.addRow("", self.wl_check_progress)
        return group

    def _build_wl_overlay_group(self):
        """The white-light snapshots laid over the fluorescence, through the camera map."""
        group = QGroupBox("White light over the fluorescence")
        form = QFormLayout(group)
        intro = QLabel(
            "The white-light snapshots saved with the acquisition, carried onto "
            "the fluorescence camera by the camera map - turned, scaled and "
            "placed - as a layer of their own, so the localizations and the "
            "reconstructions sit on them. Each frame of the movie shows the "
            "snapshot taken nearest to it, drift-corrected whenever the "
            "fluorescence image is."
        )
        intro.setWordWrap(True)
        intro.setProperty("role", "note")
        form.addRow(intro)

        roi_row = QHBoxLayout()
        self.wl_roi_x_box = QSpinBox()
        self.wl_roi_y_box = QSpinBox()
        for axis, box in (("x", self.wl_roi_x_box), ("y", self.wl_roi_y_box)):
            box.setRange(0, 100000)
            box.setPrefix(f"{axis} ")
            box.setSuffix(" px")
            box.valueChanged.connect(lambda _v: self._update_wl_overlay())
            roi_row.addWidget(box)
        roi_row.addStretch(1)
        roi_tip = (
            "Where the fluorescence image's first pixel sits on the camera sensor "
            "(its Micro-Manager ROI). The camera map is for the whole sensor, so a "
            "cropped image is placed with this. Filled in from the acquisition's "
            "metadata when it recorded the ROI (recFL does); otherwise 0, 0 - set "
            "it by hand for a cropped acquisition that did not.")
        self.wl_roi_x_box.setToolTip(roi_tip)
        self.wl_roi_y_box.setToolTip(roi_tip)
        form.addRow("Fluorescence ROI at", roi_row)

        buttons = QHBoxLayout()
        self.wl_overlay_button = QPushButton("Show the white light in the viewer")
        self.wl_overlay_button.setProperty("secondary", True)
        self.wl_overlay_button.setToolTip(
            "Add the white-light snapshots as a layer registered to the "
            "fluorescence image. It follows the drift correction, the pixel size "
            "and the ROI above as they change; delete the layer to drop it.")
        self.wl_overlay_button.clicked.connect(lambda: self.show_wl_overlay())
        buttons.addWidget(self.wl_overlay_button)
        buttons.addStretch(1)
        form.addRow("", buttons)
        self.wl_overlay_status = QLabel("No white-light snapshots loaded")
        self.wl_overlay_status.setWordWrap(True)
        form.addRow("", self.wl_overlay_status)
        return group

    def _build_rcc_group(self):
        """Redundant cross-correlation, as a refinement of the white-light drift."""
        group = QGroupBox("Refine with the localizations (RCC)")
        form = QFormLayout(group)
        intro = QLabel(
            "Redundant cross-correlation (Wang et al., 2014): the movie is cut "
            "into segments, each rendered, every pair cross-correlated, and the "
            "drift of each segment solved from all the pairs at once. Run on "
            "top of the white-light correction, it measures what that left: a "
            "camera map slightly off, or a focal plane that moved "
            "differently from the one the white-light camera watches."
        )
        intro.setWordWrap(True)
        intro.setProperty("role", "note")
        form.addRow(intro)

        self.rcc_source_box = QComboBox()
        for key, label in RCC_SOURCES.items():
            self.rcc_source_box.addItem(label, key)
        self.rcc_source_box.setToolTip(
            "Moving molecules carry no fixed structure to correlate - in a "
            "live-cell movie they mostly dilute the peak. Selecting the immobile "
            "population on the Render tab and correlating only those is often "
            "sharper, at the price of fewer localizations per segment."
        )
        form.addRow("Correlate", self.rcc_source_box)

        self.rcc_segment_box = QSpinBox()
        self.rcc_segment_box.setRange(2, 1000000)
        self.rcc_segment_box.setValue(DEFAULT_RCC_SEGMENT_FRAMES)
        self.rcc_segment_box.setSuffix(" frames")
        self.rcc_segment_box.setToolTip(
            "Frames per segment - the time resolution of the estimate. Shorter "
            "follows faster drift but gives each segment fewer localizations, "
            "and a sparse segment correlates badly."
        )
        form.addRow("Segments of", self.rcc_segment_box)

        self.rcc_pixel_box = QDoubleSpinBox()
        self.rcc_pixel_box.setRange(1.0, 2000.0)
        self.rcc_pixel_box.setDecimals(1)
        self.rcc_pixel_box.setValue(DEFAULT_RCC_PIXEL_NM)
        self.rcc_pixel_box.setSuffix(" nm")
        self.rcc_pixel_box.setToolTip(
            "Pixel of the segment images. Near the localization precision; "
            "coarsened automatically if the field would exceed "
            f"{drift_io.RCC_MAX_IMAGE_PX} pixels a side."
        )
        form.addRow("Render pixel", self.rcc_pixel_box)

        self.rcc_blur_box = QDoubleSpinBox()
        self.rcc_blur_box.setRange(0.0, 2000.0)
        self.rcc_blur_box.setDecimals(1)
        self.rcc_blur_box.setValue(DEFAULT_RCC_BLUR_NM)
        self.rcc_blur_box.setSuffix(" nm")
        self.rcc_blur_box.setToolTip(
            "Gaussian blur of the segment images (sigma). Sparse segments need "
            "more of it for the correlation to have a peak to find; too much "
            "smears the structure it is measuring."
        )
        form.addRow("Blur", self.rcc_blur_box)

        self.rcc_max_shift_box = QDoubleSpinBox()
        self.rcc_max_shift_box.setRange(10.0, 100000.0)
        self.rcc_max_shift_box.setDecimals(0)
        self.rcc_max_shift_box.setValue(DEFAULT_RCC_MAX_SHIFT_NM)
        self.rcc_max_shift_box.setSuffix(" nm")
        self.rcc_max_shift_box.setToolTip(
            "Largest shift between two segments that is searched for. On top of "
            "the white-light correction only a small residual is expected; "
            "without it, this has to cover the whole drift."
        )
        form.addRow("Search up to", self.rcc_max_shift_box)

        self.rcc_rmax_box = QDoubleSpinBox()
        self.rcc_rmax_box.setRange(0.1, 10000.0)
        self.rcc_rmax_box.setDecimals(1)
        self.rcc_rmax_box.setValue(DEFAULT_RCC_RMAX_NM)
        self.rcc_rmax_box.setSuffix(" nm")
        self.rcc_rmax_box.setToolTip(
            "A pair of segments whose measured shift disagrees with the solution "
            "by more than this is taken to have correlated on the wrong peak, "
            "and dropped - the worst first, never one that would leave a segment "
            "with no pair tying it to the rest."
        )
        form.addRow("Drop pairs off by", self.rcc_rmax_box)

        self.rcc_segment_label = QLabel("-")
        self.rcc_segment_label.setWordWrap(True)
        self.rcc_segment_label.setProperty("role", "note")
        form.addRow("", self.rcc_segment_label)

        buttons = QHBoxLayout()
        self.rcc_button = QPushButton("Estimate the drift left (RCC)")
        self.rcc_button.setProperty("primary", True)
        self.rcc_button.clicked.connect(self.estimate_rcc)
        self.rcc_cancel_button = self._cancel_button(self._rcc_cancel, "RCC")
        buttons.addWidget(self.rcc_button)
        buttons.addWidget(self.rcc_cancel_button)
        buttons.addStretch(1)
        form.addRow("", buttons)
        self.rcc_progress = QProgressBar()
        self.rcc_progress.setRange(0, 100)
        self.rcc_progress.setVisible(False)
        form.addRow("", self.rcc_progress)

        self.rcc_apply_box = QCheckBox("Apply the RCC refinement")
        self.rcc_apply_box.setChecked(True)
        self.rcc_apply_box.setToolTip(
            "Add the RCC estimate to the white-light drift, for the "
            "localizations and the shifted image alike. It refines the "
            "white-light correction it was estimated on: change that - the "
            "pixel size, the smoothing, the record - and the estimate stops "
            "applying until it is run again."
        )
        self.rcc_apply_box.toggled.connect(lambda _c: self._refresh_drift_correction())
        form.addRow("", self.rcc_apply_box)

        self.rcc_status = QLabel("No estimate yet")
        self.rcc_status.setWordWrap(True)
        form.addRow("", self.rcc_status)

        self.rcc_figure = Figure(figsize=(5, 2.2))
        self.rcc_canvas = FigureCanvas(self.rcc_figure)
        self._plot_canvases.append(self.rcc_canvas)
        self.rcc_canvas.setMinimumHeight(200)
        form.addRow("", self.rcc_canvas)
        rcc_tools = QHBoxLayout()
        rcc_tools.addStretch(1)
        rcc_tools.addWidget(self._png_button(
            lambda fig, opts: self._draw_rcc_plot(fig, opts), lambda: "rcc_drift"))
        form.addRow("", rcc_tools)

        self.rcc_segment_box.valueChanged.connect(lambda _v: self._update_rcc_status())
        self.rcc_source_box.currentIndexChanged.connect(lambda _i: self._update_rcc_status())
        return group

    def _build_localize_tab(self):
        tab = QWidget()
        self._localize_tab_index = self.tabs.addTab(tab, "Localize")
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(tab)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        note = QLabel(
            "Detect and fit 2D single-molecule localizations directly from "
            "the image loaded (or the active image layer) - no external "
            "software required. Importing already-localized data is still "
            "available via the CSV field in Load data."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        cam_group = QGroupBox("Camera")
        cam_layout = QFormLayout(cam_group)
        self.loc_gain_box = QDoubleSpinBox()
        self.loc_gain_box.setRange(0.01, 1000.0)
        self.loc_gain_box.setDecimals(3)
        self.loc_gain_box.setValue(DEFAULT_GAIN_ADU_PER_ELECTRON)
        self.loc_gain_box.setToolTip(
            "Camera gain: how many counts the sensor reports per photoelectron. "
            "(ADU - offset) / gain is what gets fitted.\n\n"
            "Electrons, not photons - they differ by the quantum efficiency, "
            "which is not needed here. Electrons are the right quantity anyway: "
            "the Poisson statistics the fit and the reported precision both "
            "assume hold for the charge collected, not for the photons that "
            "arrived and were mostly not detected.\n\n"
            "A wrong gain scales every photon count and every localization "
            "precision derived from it, so it is worth reading off the camera's "
            "own specification rather than leaving at a default."
        )
        cam_layout.addRow("Gain (ADU/e⁻)", self.loc_gain_box)
        self.loc_offset_box = QDoubleSpinBox()
        # Wide enough to hold a time-binned baseline: N summed frames carry N
        # baselines, and the old ceiling of 20000 clipped that silently from
        # about N=200 on a typical sCMOS.
        self.loc_offset_box.setRange(0.0, 1e7)
        self.loc_offset_box.setValue(100.0)
        self.loc_offset_box.setSuffix(" ADU")
        self.loc_offset_box.setToolTip(
            "Camera baseline subtracted before fitting. Autofilled from the "
            "acquisition metadata when the camera recorded it.\n\n"
            "This is the baseline of the frames the fit actually sees, so with "
            "time binning it is N times the per-frame baseline and moves with "
            "the binning factor."
        )
        cam_layout.addRow("Offset", self.loc_offset_box)
        layout.addWidget(cam_group)

        det_group = QGroupBox("Detection (local maxima + net gradient)")
        det_layout = QFormLayout(det_group)
        self.loc_box_size = QSpinBox()
        self.loc_box_size.setRange(3, 51)
        self.loc_box_size.setSingleStep(2)
        self.loc_box_size.setValue(7)
        self.loc_box_size.setSuffix(" px")
        det_layout.addRow("Box size (odd)", self.loc_box_size)
        self.loc_min_ng_box = QDoubleSpinBox()
        self.loc_min_ng_box.setRange(0.0, 1e6)
        self.loc_min_ng_box.setDecimals(1)
        self.loc_min_ng_box.setValue(800.0)
        self.loc_min_ng_box.setToolTip(
            "Detection threshold: the summed inward intensity gradient around a candidate. Raise it to reject noise, lower it to catch dim spots - use Preview to see the effect before running every frame."
        )
        det_layout.addRow("Min net gradient", self.loc_min_ng_box)
        det_buttons = QHBoxLayout()
        self.loc_preview_button = QPushButton("Preview (current frame)")
        self.loc_preview_button.setProperty("secondary", True)
        self.loc_preview_button.clicked.connect(self.loc2d_preview)
        self.loc_detect_button = QPushButton("Detect all frames")
        self.loc_detect_button.setProperty("primary", True)
        self.loc_detect_button.clicked.connect(self.loc2d_detect_all)
        self.loc_detect_cancel_button = self._cancel_button(self._loc2d_detect_cancel, "detection")
        det_buttons.addWidget(self.loc_preview_button)
        det_buttons.addWidget(self.loc_detect_button)
        det_buttons.addWidget(self.loc_detect_cancel_button)
        det_layout.addRow("", det_buttons)
        self.loc_autosave_box = QCheckBox(
            "Save every fit to a dated folder beside the data")
        self.loc_autosave_box.setChecked(True)
        self.loc_autosave_box.setToolTip(
            "After each fit, write its localizations and the complete settings "
            "to analysis/<date>_localization/ next to the image.\n\n"
            "Runs are never merged or overwritten: re-fitting after changing a "
            "threshold leaves both results side by side, and the timestamps say "
            "which was which. The table written is the fit's own output, before "
            "filtering - filters are recorded in the metadata and can be "
            "re-applied, but a discarded localization cannot be recovered."
        )
        det_layout.addRow("", self.loc_autosave_box)
        self.loc_show_candidates_box = QCheckBox("Show detection candidates on the image")
        self.loc_show_candidates_box.setChecked(True)
        self.loc_show_candidates_box.stateChanged.connect(lambda _c: self._update_loc2d_candidate_overlay())
        det_layout.addRow("", self.loc_show_candidates_box)
        self.loc_detect_progress = QProgressBar()
        self.loc_detect_progress.setRange(0, 100)
        self.loc_detect_progress.setVisible(False)
        det_layout.addRow("", self.loc_detect_progress)
        self.loc_counts_figure = Figure(figsize=(5, 2.2))
        self.loc_counts_canvas = FigureCanvas(self.loc_counts_figure)
        self._plot_canvases.append(self.loc_counts_canvas)
        self.loc_counts_canvas.setMinimumHeight(200)
        det_layout.addRow("", self.loc_counts_canvas)
        counts_tools = QHBoxLayout()
        counts_tools.addStretch(1)
        counts_tools.addWidget(self._png_button(
            lambda fig, opts: self._draw_loc2d_counts(fig, opts),
            lambda: "detection_counts"))
        det_layout.addRow("", counts_tools)
        layout.addWidget(det_group)

        fit_group = QGroupBox("Sub-pixel Gaussian fitting")
        fit_layout = QFormLayout(fit_group)
        self.loc_backend_box = QComboBox()
        self.loc_backend_box.addItems(["auto", "mle", "fast", "gpu"])
        self.loc_backend_box.setToolTip(
            "fast: least-squares Gauss-Newton, elliptical (sx and sy fitted separately).\n"
            "mle:  Poisson-weighted Gauss-Newton, also elliptical - slower, better at low photon counts.\n"
            "gpu:  Gpufit GAUSS_2D, which fits a single isotropic sigma, so sx == sy\n"
            "      for every spot it converges on. Sigma-based filtering therefore\n"
            "      behaves differently than it does for fast/mle. Spots the GPU does\n"
            "      not converge on are re-fitted with the CPU MLE, and those few rows\n"
            "      are elliptical again (sx != sy)."
        )
        fit_layout.addRow("Backend", self.loc_backend_box)
        gpu_note = QLabel(
            "\"gpu\" needs Gpufit installed; falls back to CPU MLE automatically otherwise. "
            "It fits one isotropic sigma (sx == sy), except for spots it fails to "
            "converge on, which are re-fitted on the CPU."
        )
        gpu_note.setWordWrap(True)
        fit_layout.addRow("", gpu_note)
        self.loc_fit_button = QPushButton("Fit all detected frames")
        self.loc_fit_button.setProperty("primary", True)
        self.loc_fit_button.clicked.connect(self.loc2d_fit_all)
        self.loc_fit_button.setEnabled(False)
        self.loc_fit_cancel_button = self._cancel_button(self._loc2d_fit_cancel, "fitting")
        fit_buttons = QHBoxLayout()
        fit_buttons.addWidget(self.loc_fit_button)
        fit_buttons.addWidget(self.loc_fit_cancel_button)
        fit_layout.addRow("", fit_buttons)
        self.loc_fit_progress = QProgressBar()
        self.loc_fit_progress.setRange(0, 100)
        self.loc_fit_progress.setVisible(False)
        fit_layout.addRow("", self.loc_fit_progress)
        layout.addWidget(fit_group)
        layout.addStretch(1)

        self.loc_box_size.valueChanged.connect(self._on_loc2d_box_changed)

    def _build_filter_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Filter")
        root = QVBoxLayout(tab)
        root.setContentsMargins(0, 0, 0, 0)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        self.filter_status = QLabel("No data loaded")
        header_layout.addWidget(self.filter_status)
        buttons_row = QHBoxLayout()
        self.reset_filters_button = QPushButton("Reset filters")
        self.reset_filters_button.setProperty("secondary", True)
        self.reset_filters_button.clicked.connect(self.reset_filters)
        self.reset_filters_button.setEnabled(False)
        self.apply_filters_button = QPushButton("Apply filters")
        self.apply_filters_button.setProperty("primary", True)
        self.apply_filters_button.clicked.connect(self.apply_filters)
        self.apply_filters_button.setEnabled(False)
        buttons_row.addWidget(self.reset_filters_button)
        buttons_row.addWidget(self.apply_filters_button)
        buttons_row.addStretch(1)
        header_layout.addLayout(buttons_row)

        defaults_row = QHBoxLayout()
        self.save_filter_defaults_button = QPushButton("Set as default")
        self.save_filter_defaults_button.setProperty("secondary", True)
        self.save_filter_defaults_button.setToolTip(
            "Keep the filters as they are now as your defaults: every table "
            "loaded from now on - in this session and the next ones - starts "
            "with them, and 'Reset filters' goes back to them.\n\n"
            "Saved: the localization bounds here that differ from the built-in "
            "ones, side by side (a side left at the data's own range stays "
            "free to follow the next dataset), and the trajectory filters of "
            "the Track tab. Never x, y or frame: those say where one dataset "
            "lies, not what counts as a good localization.")
        self.save_filter_defaults_button.clicked.connect(lambda: self.save_filter_defaults())
        self.load_filters_button = QPushButton("Load filters from...")
        self.load_filters_button.setProperty("secondary", True)
        self.load_filters_button.setToolTip(
            "Take the filters of another analysis - a session file or a run's "
            "metadata.json - and apply them here: the localization bounds and "
            "the trajectory filters, nothing else (not x, y or frame, not the "
            "linking or the rendering).")
        self.load_filters_button.clicked.connect(lambda: self.load_filters_from())
        self.forget_filter_defaults_button = QPushButton("Forget my defaults")
        self.forget_filter_defaults_button.setProperty("secondary", True)
        self.forget_filter_defaults_button.setToolTip(
            "Go back to the built-in filter defaults, now and in later sessions.")
        self.forget_filter_defaults_button.clicked.connect(
            lambda: self.forget_filter_defaults())
        defaults_row.addWidget(self.save_filter_defaults_button)
        defaults_row.addWidget(self.load_filters_button)
        defaults_row.addWidget(self.forget_filter_defaults_button)
        defaults_row.addStretch(1)
        header_layout.addLayout(defaults_row)
        self.filter_defaults_label = QLabel("Defaults: built-in")
        self.filter_defaults_label.setProperty("role", "note")
        self.filter_defaults_label.setWordWrap(True)
        header_layout.addWidget(self.filter_defaults_label)
        note = QLabel(
            "x / y are filtered with the yellow box drawn on the image: drag "
            "the middle to move it, drag a corner/edge handle to resize it. "
            "Changing any filter clears trajectories - relink afterwards."
        )
        note.setWordWrap(True)
        header_layout.addWidget(note)

        root.addWidget(header)

        scroll = QScrollArea(tab)
        scroll.setWidgetResizable(True)
        self.filter_content = QWidget()
        self.filter_layout = QGridLayout(self.filter_content)
        scroll.setWidget(self.filter_content)
        root.addWidget(scroll)
        self.filter_layout.addWidget(QLabel("Load data to see filters"), 0, 0)

    def _build_render_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Render")
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(tab)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        note = QLabel(
            "Reconstruct a super-resolved image from the localizations that "
            "pass the current filters - so tightening a filter and rendering "
            "again shows exactly what that filter does to the reconstruction. "
            "Pixel values are localization counts (or photons), never rescaled, "
            "so two renders can be compared quantitatively."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        layout.addWidget(self._build_render_population_group())
        layout.addWidget(self._build_merge_group())

        # --- source ---------------------------------------------------------
        source_group = QGroupBox("Localizations to render")
        source_layout = QFormLayout(source_group)
        self.render_source_label = QLabel("No localizations loaded")
        source_layout.addRow("", self.render_source_label)

        # Same two paths as the Load data tab, kept in sync both ways, so a
        # session that only ever renders never has to leave this tab.
        self.render_csv_edit = QLineEdit()
        render_csv_button = QPushButton("Browse CSV")
        render_csv_button.clicked.connect(self.browse_csv)
        csv_row = QHBoxLayout()
        csv_row.addWidget(self.render_csv_edit)
        csv_row.addWidget(render_csv_button)
        source_layout.addRow("Localization CSV", csv_row)

        self.render_image_edit = QLineEdit()
        render_image_button = QPushButton("Browse image")
        render_image_button.clicked.connect(self.browse_image)
        image_row = QHBoxLayout()
        image_row.addWidget(self.render_image_edit)
        image_row.addWidget(render_image_button)
        source_layout.addRow("Image (field of view)", image_row)

        self.render_load_button = QPushButton("Load")
        self.render_load_button.setProperty("secondary", True)
        self.render_load_button.clicked.connect(self.load_data)
        self.render_load_button.setToolTip(
            "Loads through the same path as the Load data tab, so the "
            "localizations end up filterable and linkable as usual."
        )
        source_layout.addRow("", self.render_load_button)
        layout.addWidget(source_group)

        self._sync_line_edits(self.csv_edit, self.render_csv_edit)
        self._sync_line_edits(self.image_edit, self.render_image_edit)

        # --- sampling -------------------------------------------------------
        sampling_group = QGroupBox("Sampling")
        sampling_layout = QFormLayout(sampling_group)
        self.render_oversampling_box = QSpinBox()
        self.render_oversampling_box.setRange(1, 200)
        self.render_oversampling_box.setValue(10)
        self.render_oversampling_box.setToolTip(
            "Super-resolved pixels per camera pixel. The reconstruction should "
            "be sampled finer than the localization precision, but every "
            "doubling costs four times the memory."
        )
        self.render_oversampling_box.setSuffix(" x")
        sampling_layout.addRow("Oversampling", self.render_oversampling_box)
        self.render_size_label = QLabel("-")
        self.render_size_label.setWordWrap(True)
        sampling_layout.addRow("", self.render_size_label)
        self.render_backend_label = QLabel(smlm_render.render_gpu_status())
        self.render_backend_label.setWordWrap(True)
        sampling_layout.addRow("", self.render_backend_label)
        self.render_gpu_box = QCheckBox("Use the GPU when it is available and the frame fits")
        self.render_gpu_box.setChecked(True)
        sampling_layout.addRow("", self.render_gpu_box)
        layout.addWidget(sampling_group)

        # --- mode -----------------------------------------------------------
        mode_group = QGroupBox("Mode")
        mode_layout = QFormLayout(mode_group)
        self.render_mode_box = QComboBox()
        for key, label in smlm_render.MODES.items():
            self.render_mode_box.addItem(label, key)
        self.render_mode_box.setCurrentIndex(
            self.render_mode_box.findData("gaussian_global")
        )
        mode_layout.addRow("Render as", self.render_mode_box)

        self.render_sigma_box = QDoubleSpinBox()
        self.render_sigma_box.setRange(0.1, 5000.0)
        self.render_sigma_box.setDecimals(1)
        self.render_sigma_box.setValue(30.0)
        self.render_sigma_label = QLabel("Blur width sigma")
        self.render_sigma_box.setSuffix(" nm")
        mode_layout.addRow(self.render_sigma_label, self.render_sigma_box)

        self.render_sigma_column_box = QComboBox()
        self.render_sigma_column_label = QLabel("Width from column")
        self.render_sigma_column_box.setToolTip(
            "Usually the localization uncertainty: each molecule is drawn as "
            "wide as it was actually located, which is the honest picture. "
            "Choosing a PSF sigma column instead just redraws the diffraction "
            "limit and throws the super-resolution away."
        )
        mode_layout.addRow(self.render_sigma_column_label, self.render_sigma_column_box)

        clamp_row = QHBoxLayout()
        self.render_sigma_min_box = QDoubleSpinBox()
        self.render_sigma_min_box.setRange(0.1, 5000.0)
        self.render_sigma_min_box.setDecimals(1)
        self.render_sigma_min_box.setValue(5.0)
        self.render_sigma_max_box = QDoubleSpinBox()
        self.render_sigma_max_box.setRange(0.1, 5000.0)
        self.render_sigma_max_box.setDecimals(1)
        self.render_sigma_max_box.setValue(100.0)
        clamp_row.addWidget(self.render_sigma_min_box)
        clamp_row.addWidget(QLabel("to"))
        clamp_row.addWidget(self.render_sigma_max_box)
        self.render_clamp_label = QLabel("Clamp width to (nm)")
        self.render_clamp_label.setToolTip(
            "A row whose fit returned an absurd precision would otherwise paint "
            "a huge blob over the reconstruction; rows with no precision at all "
            "are drawn at the lower bound rather than dropped."
        )
        mode_layout.addRow(self.render_clamp_label, clamp_row)

        self.render_photons_box = QCheckBox(
            "Weight each localization by its photon count instead of counting it once"
        )
        mode_layout.addRow("", self.render_photons_box)

        self.render_colormap_box = QComboBox()
        self.render_colormap_box.addItems(RENDER_COLORMAPS)
        self.render_colormap_box.setToolTip(
            "Used for the render layer in the viewer, for the PNG preview, "
            "and for the reconstruction inside a composite."
        )
        mode_layout.addRow("Colormap (display and PNG)", self.render_colormap_box)
        layout.addWidget(mode_group)

        # --- image ----------------------------------------------------------
        image_group = QGroupBox("Image")
        image_layout = QFormLayout(image_group)
        image_buttons = QHBoxLayout()
        self.render_image_button = QPushButton("Render image")
        self.render_image_button.setProperty("primary", True)
        self.render_image_button.clicked.connect(self.render_smlm_image)
        self.render_image_button.setEnabled(False)
        image_buttons.addWidget(self.render_image_button)
        image_buttons.addStretch(1)
        image_layout.addRow("", image_buttons)
        layout.addWidget(image_group)

        # --- movie ----------------------------------------------------------
        movie_group = QGroupBox("Movie")
        movie_layout = QFormLayout(movie_group)
        self.render_frames_per_box = QSpinBox()
        self.render_frames_per_box.setRange(1, 1_000_000)
        self.render_frames_per_box.setValue(1000)
        self.render_frames_per_box.setSuffix(" frames")
        movie_layout.addRow("Raw frames per super-resolved frame", self.render_frames_per_box)

        self.render_grouping_box = QComboBox()
        for key, label in smlm_render.GROUPINGS.items():
            self.render_grouping_box.addItem(label, key)
        self.render_grouping_box.setToolTip(
            "Independent blocks: each movie frame holds only its own raw frames.\n"
            "Cumulative build-up: each frame adds to everything before it, so the\n"
            "  reconstruction fills in as the movie plays.\n"
            "Sliding window: a window of that many raw frames advanced by the step\n"
            "  below, which gives a smoother movie at the cost of re-rendering the\n"
            "  overlap."
        )
        movie_layout.addRow("Grouping", self.render_grouping_box)

        self.render_step_box = QSpinBox()
        self.render_step_box.setRange(1, 1_000_000)
        self.render_step_box.setValue(500)
        self.render_step_box.setSuffix(" frames")
        self.render_step_label = QLabel("Window step")
        movie_layout.addRow(self.render_step_label, self.render_step_box)

        self.render_start_frame_box = QSpinBox()
        self.render_start_frame_box.setRange(0, 100_000_000)
        self.render_start_frame_box.setValue(0)
        self.render_start_frame_box.setSpecialValueText("the first frame")
        self.render_start_frame_box.setToolTip(
            "Where the movie begins, and for a cumulative build-up where the "
            "accumulation starts from.\n\n"
            "Localizations before this frame are left out entirely, so a "
            "cumulative movie can be made to build up from the moment something "
            "starts happening rather than from a stretch of bleaching or drift "
            "at the beginning of the acquisition."
        )
        movie_layout.addRow("Start from", self.render_start_frame_box)
        self.render_start_frame_box.valueChanged.connect(
            lambda _v: self._update_render_info())

        self.render_movie_label = QLabel("-")
        self.render_movie_label.setWordWrap(True)
        movie_layout.addRow("", self.render_movie_label)

        movie_buttons = QHBoxLayout()
        self.render_movie_button = QPushButton("Render movie")
        self.render_movie_button.setProperty("primary", True)
        self.render_movie_button.clicked.connect(self.render_smlm_movie)
        self.render_movie_button.setEnabled(False)
        movie_buttons.addWidget(self.render_movie_button)
        movie_buttons.addStretch(1)
        movie_layout.addRow("", movie_buttons)
        layout.addWidget(movie_group)

        layout.addStretch(1)

        self.render_mode_box.currentIndexChanged.connect(self._on_render_mode_changed)
        self.render_grouping_box.currentIndexChanged.connect(self._on_render_grouping_changed)
        self.render_oversampling_box.valueChanged.connect(lambda _v: self._update_render_info())
        self.render_frames_per_box.valueChanged.connect(lambda _v: self._update_render_info())
        self.render_step_box.valueChanged.connect(lambda _v: self._update_render_info())
        self.pixel_size_box.valueChanged.connect(lambda _v: self._update_render_info())
        # The world is measured in nanometres, so it stretches when the pixel
        # size does - and napari's scale bar with it.
        self.pixel_size_box.valueChanged.connect(lambda _v: self._apply_viewer_scale())
        self._on_render_mode_changed()
        self._on_render_grouping_changed()

    def _build_view_save_group(self):
        """Save the image layers as they are displayed."""
        group = QGroupBox("The view, as displayed")
        form = QFormLayout(group)
        intro = QLabel(
            "Every visible image layer - reconstructions, the movie, the white light, "
            "averages - at the time point on the slider, each through its own "
            "contrast, gamma and colormap and blended as napari blends it, at the "
            "finest resolution among them rather than the screen's.")
        intro.setWordWrap(True)
        intro.setProperty("role", "note")
        form.addRow(intro)
        self.view_region_box = QComboBox()
        self.view_region_box.addItem("What the canvas shows (as zoomed)", "view")
        self.view_region_box.addItem("All of the visible layers", "all")
        form.addRow("Region", self.view_region_box)
        self.view_scalebar_box = QCheckBox("Burn in a scale bar (bottom right, white)")
        self.view_scalebar_box.setChecked(True)
        self.view_scalebar_box.setToolTip(
            "A round length, about a sixth of the picture's width, with its label. "
            "In the RGB files only: a channel TIFF keeps its values untouched and "
            "carries the pixel size instead, for ImageJ's own scale bar.")
        form.addRow("", self.view_scalebar_box)
        row = QHBoxLayout()
        self.view_save_button = QPushButton("Save the view...")
        self.view_save_button.setProperty("primary", True)
        self.view_save_button.setToolTip(
            "PNG: the blended colours, plain RGB.\n"
            "TIFF, RGB: the same, lossless, with the pixel size.\n"
            "TIFF, channels: each single-channel layer's values on the common grid, "
            "with its LUT and display range - ImageJ opens it as a composite in the "
            "same colours, every channel's contrast still adjustable.")
        self.view_save_button.clicked.connect(lambda: self.save_view())
        row.addWidget(self.view_save_button)
        row.addStretch(1)
        form.addRow("", row)
        self.view_save_status = QLabel("-")
        self.view_save_status.setWordWrap(True)
        self.view_save_status.setProperty("role", "note")
        form.addRow("", self.view_save_status)
        return group

    def _visible_image_layers(self):
        return [layer for layer in self.viewer.layers
                if isinstance(layer, napari.layers.Image) and layer.visible]

    def save_view(self, path=None, kind=None):
        """Write the visible image layers, composed as displayed. Returns the path."""
        layers = self._visible_image_layers()
        if not layers:
            self.log("No visible image layer to save")
            return None
        if path is None:
            start = str(self._figure_save_dir() / "view.png")
            path, chosen = QFileDialog.getSaveFileName(
                self, "Save the view", start, ";;".join(VIEW_SAVE_FILTERS.values()))
            if not path:
                return None
            kind = next((k for k, v in VIEW_SAVE_FILTERS.items() if v == chosen), None)
        path = Path(path)
        if kind is None:
            kind = "png" if path.suffix.lower() == ".png" else "rgb_tiff"
        suffix = ".png" if kind == "png" else ".tif"
        if path.suffix.lower() not in ((".png",) if kind == "png" else (".tif", ".tiff")):
            path = path.with_suffix(suffix)
        region = self.view_region_box.currentData()
        box = (view_export.view_box(self.viewer, layers) if region == "view"
               else view_export.layers_box(layers))
        try:
            composite = view_export.compose(layers, self.viewer.dims.point, box)
        except Exception as exc:
            self.log(f"Could not compose the view: {exc}")
            return None
        names = ", ".join(layer.name for layer in layers)
        description = (f"napari-loc-track view: {names}; {composite.pixel_nm:.2f} nm/px; "
                       f"time point {self._get_current_frame()}")
        bar = None
        try:
            if kind == "channels_tiff":
                view_export.save_channels_tiff(path, composite, description)
            else:
                rgb8 = composite.rgb8()
                if self.view_scalebar_box.isChecked():
                    bar = view_export.burn_scale_bar(rgb8, composite.pixel_nm)
                if kind == "png":
                    view_export.save_png(path, rgb8, composite.pixel_nm)
                else:
                    view_export.save_rgb_tiff(path, rgb8, composite.pixel_nm, description)
        except Exception as exc:
            self.log(f"Could not save the view: {exc}")
            return None
        h, w = composite.rgb.shape[:2]
        text = (f"Saved {path.name}: {w} x {h} px at {composite.pixel_nm:.1f} nm/px, "
                f"{len(layers)} layer(s)"
                + (f", scale bar {smlm_render.format_length(bar)}" if bar else "")
                + ("; " + "; ".join(composite.notes) if composite.notes else ""))
        self.view_save_status.setText(text)
        self.log(text + f" - {path.parent}")
        return path

    # --- the Images tab: time averages and line profiles -------------------------
    def _build_images_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Images")
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(tab)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)
        layout.addWidget(self._build_average_group())
        layout.addWidget(self._build_profile_group())
        layout.addStretch(1)

    def _build_average_group(self):
        group = QGroupBox("Time-averaged images")
        form = QFormLayout(group)
        intro = QLabel(
            "The movie averaged over time as it is displayed - drift-corrected, or "
            "with the growth correction in the final, stretched geometry - is the "
            "diffraction-limited image of everything that stayed put. The "
            "white-light snapshots averaged the same way sharpen the tissue: its "
            "walls stand still while the cytoplasm streaming through averages out. "
            "Each lands in the viewer as a layer, in place.")
        intro.setWordWrap(True)
        intro.setProperty("role", "note")
        form.addRow(intro)
        frames = QHBoxLayout()
        self.average_first_box = QSpinBox()
        self.average_first_box.setRange(0, 10_000_000)
        self.average_last_box = QSpinBox()
        self.average_last_box.setRange(-1, 10_000_000)
        self.average_last_box.setValue(-1)
        self.average_last_box.setSpecialValueText("the last")
        self.average_every_box = QSpinBox()
        self.average_every_box.setRange(1, 100_000)
        self.average_every_box.setToolTip(
            "Every Nth frame. The average of a long movie changes little past a "
            "few thousand frames; skipping makes it that much faster to read.")
        for widget in (QLabel("frames"), self.average_first_box, QLabel("to"),
                       self.average_last_box, QLabel("every"), self.average_every_box):
            frames.addWidget(widget)
        frames.addStretch(1)
        form.addRow("Fluorescence", frames)
        buttons = QHBoxLayout()
        self.average_fluorescence_button = QPushButton("Average the fluorescence")
        self.average_fluorescence_button.setProperty("secondary", True)
        self.average_fluorescence_button.setToolTip(
            "The mean of the movie's frames as displayed. Frames are summed as "
            "recorded in blocks of a couple of seconds, and each block is moved "
            "once, by its middle frame's correction - the sample moves far less "
            "than the diffraction limit in that time.")
        self.average_fluorescence_button.clicked.connect(lambda: self.average_fluorescence())
        self.average_wl_button = QPushButton("Average the white light")
        self.average_wl_button.setProperty("secondary", True)
        self.average_wl_button.setToolTip(
            "The mean of the white-light snapshots, each moved as the white-light "
            "overlay moves it - stabilized, and in the final geometry when the "
            "growth correction is on - and laid on the fluorescence through the "
            "camera map.")
        self.average_wl_button.clicked.connect(lambda: self.average_white_light())
        self.average_cancel_button = self._cancel_button(self._average_cancel, "the average")
        for widget in (self.average_fluorescence_button, self.average_wl_button,
                       self.average_cancel_button):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        form.addRow("", buttons)
        self.average_progress = QProgressBar()
        self.average_progress.setRange(0, 100)
        self.average_progress.setVisible(False)
        form.addRow("", self.average_progress)
        self.average_status = QLabel("-")
        self.average_status.setWordWrap(True)
        self.average_status.setProperty("role", "note")
        form.addRow("", self.average_status)
        return group

    def _correction_label(self, displayed):
        if isinstance(displayed, drift_io.WarpedStack):
            return "growth-corrected, final geometry"
        if isinstance(displayed, drift_io.ShiftedStack):
            return "drift-corrected"
        return "as recorded"

    def average_fluorescence(self):
        name = self._image_layer_name
        if not name or name not in self.viewer.layers:
            self.log("Load the movie first - its frames are what is averaged")
            return
        displayed = self.viewer.layers[name].data
        n = int(drift_io.unshifted(displayed).shape[0])
        last = self.average_last_box.value()
        last = n - 1 if last < 0 else min(last, n - 1)
        first = min(self.average_first_box.value(), last)
        every = self.average_every_box.value()
        label = self._correction_label(displayed)
        self.log(f"Averaging frames {first}-{last} every {every} ({label})...")
        self._start_average(displayed, dict(first=first, last=last, every=every),
                            kind="fluorescence",
                            name=f"average fluorescence ({label})",
                            origin=averages_io.canvas_origin(displayed),
                            note=f"frames {first}-{last}, every {every}; {label}",
                            params={"first": first, "last": last, "every": every})

    def average_white_light(self):
        result, problem = self._wl_overlay_data()
        if result is None:
            self.log(f"No white-light average: {problem}")
            return
        _cmap, data, origin_yx, _drift_out, _timed, _key = result
        stack = data.base if isinstance(data, drift_io.FrameIndexedStack) else data
        label = self._correction_label(stack)
        self.log(f"Averaging the {len(stack)} white-light snapshots ({label})...")
        self._start_average(stack, dict(block=1), kind="white light",
                            name=f"average white light ({label})", origin=tuple(origin_yx),
                            note=f"{len(stack)} snapshots; {label}", params={})

    def _start_average(self, displayed, kwargs, kind, name, origin, note, params=None):
        self.average_fluorescence_button.setEnabled(False)
        self.average_wl_button.setEnabled(False)
        self.average_progress.setVisible(True)
        self.average_progress.setValue(0)
        self._arm_cancel(self._average_cancel, self.average_cancel_button)
        worker = _average_worker(displayed, kwargs, self._average_cancel)
        worker.yielded.connect(lambda frac: self.average_progress.setValue(int(frac * 100)))
        worker.returned.connect(
            lambda result: self._on_average_done(result, kind, name, origin, note, params))
        worker.errored.connect(lambda exc: self.log(f"The average failed: {exc}"))
        worker.finished.connect(self._on_average_worker_finished)
        self._average_worker_ref = worker
        worker.start()

    def _on_average_worker_finished(self):
        self.average_fluorescence_button.setEnabled(True)
        self.average_wl_button.setEnabled(True)
        self.average_cancel_button.setEnabled(False)
        self.average_progress.setVisible(False)
        self._average_worker_ref = None
        self._session_advance()

    def _on_average_done(self, result, kind, name, origin, note, params=None):
        if result is CANCELLED or result is None:
            self.log("Average cancelled")
            return
        image, used = result
        finite = image[np.isfinite(image)]
        if not finite.size:
            self.log("The average is empty")
            return
        low, high = (float(v) for v in np.percentile(finite, (0.5, 99.8)))
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        metadata = {AVERAGE_TAG: kind, "origin_yx": [float(v) for v in origin], "note": note,
                    "params": dict(params or {})}
        colormap = "gray" if kind == "white light" else "inferno"
        layer = self.viewer.add_image(
            np.nan_to_num(image, nan=low), name=name, colormap=colormap,
            contrast_limits=(low, max(high, low + 1e-6)), blending="additive",
            metadata=metadata, units=self._viewer_units(2))
        self._place_average_layer(layer)
        self._apply_viewer_scale()
        text = f"{name}: {used} {'snapshots' if kind == 'white light' else 'frames'} ({note})"
        self.average_status.setText(text)
        self.log("Added " + text)

    def _place_average_layer(self, layer):
        """Put a time-averaged image where its stack is drawn, in the viewer's nm."""
        meta = getattr(layer, "metadata", None) or {}
        kind = meta.get(AVERAGE_TAG)
        origin = meta.get("origin_yx", (0, 0))
        pixel_nm = self.pixel_size_box.value()
        if kind == "white light":
            cmap, _problem = self._overlay_camera_map()
            if cmap is None or cmap.offset is None:
                return
            plane = self._wl_overlay_affine(cmap, origin)[1:, 1:]
            layer.scale = (1.0, 1.0)
            layer.translate = (0.0, 0.0)
            layer.affine = plane
        else:
            layer.scale = (pixel_nm, pixel_nm)
            layer.translate = (origin[0] * pixel_nm, origin[1] * pixel_nm)

    def _build_profile_group(self):
        group = QGroupBox("Line profile")
        form = QFormLayout(group)
        intro = QLabel(
            "Draw a line on the image; the values of every visible image layer "
            "along it are plotted, at the time point on the slider - the numbers "
            "the layers hold, not their colours. Moving or redrawing the line, or "
            "the slider, updates the plot.")
        intro.setWordWrap(True)
        intro.setProperty("role", "note")
        form.addRow(intro)
        row = QHBoxLayout()
        self.profile_draw_button = QPushButton("Draw a line")
        self.profile_draw_button.setProperty("secondary", True)
        self.profile_draw_button.clicked.connect(lambda: self.start_profile_line())
        row.addWidget(self.profile_draw_button)
        self.profile_width_box = QDoubleSpinBox()
        self.profile_width_box.setRange(0.0, 100000.0)
        self.profile_width_box.setDecimals(0)
        self.profile_width_box.setSuffix(" nm")
        self.profile_width_box.setSpecialValueText("a single line")
        self.profile_width_box.setToolTip("Averaged across the line over this width.")
        self.profile_width_box.valueChanged.connect(lambda _v: self.update_profile())
        row.addWidget(QLabel("width"))
        row.addWidget(self.profile_width_box)
        row.addStretch(1)
        form.addRow("", row)
        options = QHBoxLayout()
        self.profile_normalize_box = QCheckBox("Each layer 0 to 1")
        self.profile_normalize_box.setToolTip(
            "Scale each profile from its own minimum to its own maximum, to compare "
            "layers whose values are in different units.")
        self.profile_normalize_box.toggled.connect(lambda _c: self.update_profile())
        self.profile_fit_box = QCheckBox("Fit a Gaussian (FWHM)")
        self.profile_fit_box.setToolTip(
            "A Gaussian on a constant, fitted to each profile: its full width at half "
            "maximum and its centre, for a line drawn across one structure.")
        self.profile_fit_box.toggled.connect(lambda _c: self.update_profile())
        options.addWidget(self.profile_normalize_box)
        options.addWidget(self.profile_fit_box)
        options.addStretch(1)
        form.addRow("", options)
        self.profile_figure = Figure(figsize=(5, 2.4))
        self.profile_canvas = FigureCanvas(self.profile_figure)
        self._plot_canvases.append(self.profile_canvas)
        self.profile_canvas.setMinimumHeight(220)
        form.addRow("", self.profile_canvas)
        tools = QHBoxLayout()
        self.profile_csv_button = QPushButton("CSV...")
        self.profile_csv_button.setToolTip("The profiles as a table: distance and value, per layer.")
        self.profile_csv_button.setProperty("secondary", True)
        self.profile_csv_button.setMaximumWidth(100)
        self.profile_csv_button.clicked.connect(lambda: self.save_profile_csv())
        tools.addStretch(1)
        tools.addWidget(self.profile_csv_button)
        tools.addWidget(self._png_button(
            lambda fig, opts: self._draw_profile_plot(fig, opts), lambda: "line_profile"))
        form.addRow("", tools)
        self.profile_status = QLabel("No line drawn.")
        self.profile_status.setWordWrap(True)
        self.profile_status.setProperty("role", "note")
        form.addRow("", self.profile_status)
        self._profile_timer = QTimer(self)
        self._profile_timer.setSingleShot(True)
        self._profile_timer.timeout.connect(self.update_profile)
        return group

    def _profile_layer(self):
        layer = self.viewer.layers[PROFILE_LAYER_NAME] if PROFILE_LAYER_NAME in self.viewer.layers else None
        return layer if isinstance(layer, napari.layers.Shapes) else None

    def start_profile_line(self):
        """Give the user a line tool on a shapes layer of its own."""
        layer = self._profile_layer()
        if layer is None:
            # in camera pixels, like the image it is drawn on
            layer = self.viewer.add_shapes(
                name=PROFILE_LAYER_NAME, ndim=2, edge_color="yellow", face_color="transparent",
                edge_width=0.5, **self._placed({}, 2))
            layer.events.data.connect(lambda _e=None: self._profile_timer.start(150))
            self.viewer.dims.events.current_step.connect(
                lambda _e=None: self._profile_timer.start(150) if self._profile_layer() else None)
        self.viewer.layers.selection.active = layer
        layer.mode = "add_line"
        self.profile_status.setText("Click and drag on the image to draw the line.")

    def _profile_line_world(self):
        """(start, end) world (y, x) of the last line drawn, or None."""
        layer = self._profile_layer()
        if layer is None or not len(layer.data):
            return None
        for data, kind in zip(reversed(layer.data), reversed(layer.shape_type)):
            if kind in ("line", "path") and len(data) >= 2:
                vertices = np.asarray(data, dtype=float)[[0, -1]]
                world = np.array([layer.data_to_world(v) for v in vertices], dtype=float)
                return world[0][-2:], world[1][-2:]
        return None

    def update_profile(self):
        if not hasattr(self, "profile_figure"):
            return
        line = self._profile_line_world()
        if line is None:
            self._profile_data = None
            self._draw_profile_plot()
            return
        start, end = line
        profiles = []
        for layer in self._visible_image_layers():
            try:
                distance, values = profile_io.sample_line(
                    layer, self.viewer.dims.point, start, end, width=self.profile_width_box.value())
            except Exception as exc:
                self.log(f"No profile for {layer.name}: {exc}")
                continue
            color = np.asarray(layer.colormap.map(np.array([0.85]))[0][:3], dtype=float)
            if np.ptp(color) < 0.1:                  # a grey colormap: draw it in the ink
                color = None
            profiles.append({"name": layer.name, "distance_um": distance / 1000.0,
                             "values": values, "color": color})
        self._profile_data = profiles
        self._draw_profile_plot()
        length = float(np.hypot(*(np.asarray(end) - np.asarray(start)))) / 1000.0
        fits = [f"{p['name']}: FWHM {p['fit']['fwhm'] * 1000:.0f} nm"
                for p in profiles if p.get("fit")]
        self.profile_status.setText(
            f"{length:.2f} µm line, {len(profiles)} layer(s)"
            + (" - " + "; ".join(fits) if fits else ""))

    def _draw_profile_plot(self, figure=None, opts=None):
        export = figure is not None
        opts = opts or {}
        figure = figure if export else self.profile_figure
        figure.clear()
        profiles = self._profile_data or []
        ax = figure.add_subplot(111)
        for p in profiles:
            y = np.asarray(p["values"], dtype=float)
            if self.profile_normalize_box.isChecked() and np.isfinite(y).any():
                low, high = np.nanmin(y), np.nanmax(y)
                y = (y - low) / (high - low) if high > low else y * 0
            label = p["name"]
            p["fit"] = None
            if self.profile_fit_box.isChecked():
                fit = profile_io.fit_gaussian(np.asarray(p["distance_um"]), y)
                if fit is not None:
                    p["fit"] = dict(fit, fwhm=fit["fwhm"], centre=fit["centre"])
                    label += f" (FWHM {fit['fwhm'] * 1000:.0f} nm)"
            color = p["color"] if p["color"] is not None else INK
            ax.plot(p["distance_um"], y, color=color, linewidth=1.2, label=label)
            if p["fit"] is not None:
                ax.plot(p["fit"]["distance"], p["fit"]["curve"], color=color, linewidth=0.8,
                        linestyle="--")
        ax.set_xlabel("Distance along the line (µm)")
        ax.set_ylabel("Value (0-1)" if self.profile_normalize_box.isChecked() else "Value")
        if profiles:
            legend = ax.legend(fontsize=plot_font(-2), loc="upper right", facecolor=PLOT_BG,
                               edgecolor=PANEL_LINE, labelcolor=INK)
            legend.get_frame().set_alpha(0.85)
        style_axes(figure, ax, title="Line profile" if opts.get("title", True) else None)
        figure.tight_layout()
        if not export:
            self.profile_canvas.draw_idle()

    def save_profile_csv(self, path=None):
        profiles = self._profile_data or []
        if not profiles:
            self.log("Draw a line first")
            return None
        if path is None:
            start = str(self._figure_save_dir() / "line_profile.csv")
            path, _ = QFileDialog.getSaveFileName(self, "Save the line profile", start,
                                                  "CSV files (*.csv)")
            if not path:
                return None
        columns = {}
        for p in profiles:
            columns[f"{p['name']} distance [um]"] = pd.Series(p["distance_um"])
            columns[f"{p['name']} value"] = pd.Series(p["values"])
        pd.DataFrame(columns).to_csv(path, index=False)
        self.log(f"Line profile saved to {path}")
        return Path(path)

    def _build_save_tab(self):
        """Everything about turning a finished render into files on disk.

        Rendering and saving were one long tab; they are different jobs done at
        different moments - you render once and then try several ways of
        writing it out - so the options for each now sit where that job is.
        """
        tab = QWidget()
        self.tabs.addTab(tab, "Save")
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(tab)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        note = QLabel(
            "Render an image or a movie on the Render tab first; this tab "
            "writes whichever of them you have, as often as you like and in "
            "whichever format, without rendering again."
        )
        note.setWordWrap(True)
        note.setProperty("role", "note")
        layout.addWidget(note)
        layout.addWidget(self._build_view_save_group())

        # --- the save buttons ------------------------------------------------
        buttons_group = QGroupBox("Write to disk")
        buttons_layout = QVBoxLayout(buttons_group)
        image_row = QHBoxLayout()
        self.render_save_image_button = QPushButton("Save image...")
        self.render_save_image_button.setProperty("primary", True)
        self.render_save_image_button.clicked.connect(self.save_render_image)
        self.render_save_image_button.setEnabled(False)
        self.render_save_composite_image_button = QPushButton("Save composite image...")
        self.render_save_composite_image_button.setProperty("secondary", True)
        self.render_save_composite_image_button.setToolTip(
            "Save the blend of the layers ticked below, whatever the format "
            "box above is set to."
        )
        self.render_save_composite_image_button.clicked.connect(self.save_composite_image)
        self.render_save_composite_image_button.setEnabled(False)
        image_row.addWidget(self.render_save_image_button)
        image_row.addWidget(self.render_save_composite_image_button)
        buttons_layout.addLayout(image_row)

        movie_row = QHBoxLayout()
        self.render_save_movie_button = QPushButton("Save movie...")
        self.render_save_movie_button.setProperty("primary", True)
        self.render_save_movie_button.clicked.connect(self.save_render_movie)
        self.render_save_movie_button.setEnabled(False)
        # No composite movie button. A blended movie has to be reconstructed at
        # the render's own resolution and written frame by frame, which made it
        # both slow and lower resolution than the thing on screen. Screen-record
        # the viewer instead - set the speed under Playback on the Load tab -
        # and every layer appears exactly as it looks, at display resolution.
        movie_row.addWidget(self.render_save_movie_button)
        buttons_layout.addLayout(movie_row)

        # A reconstruction movie is usually far longer than anything you would
        # show, and with a sliding window most of its frames are the same
        # picture again. Both are chosen here rather than by re-rendering.
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Frames"))
        self.movie_first_box = QSpinBox()
        self.movie_first_box.setRange(0, 0)
        self.movie_first_box.setToolTip("First rendered frame to write.")
        range_row.addWidget(self.movie_first_box)
        range_row.addWidget(QLabel("to"))
        self.movie_last_box = QSpinBox()
        self.movie_last_box.setRange(0, 0)
        self.movie_last_box.setToolTip("Last rendered frame to write, inclusive.")
        range_row.addWidget(self.movie_last_box)
        range_row.addWidget(QLabel("every"))
        self.movie_stride_box = QSpinBox()
        self.movie_stride_box.setRange(1, 10000)
        self.movie_stride_box.setValue(1)
        self.movie_stride_box.setToolTip(
            "Keep every Nth rendered frame.\n\n"
            "With a sliding window, consecutive frames overlap by most of their "
            "raw frames and carry almost the same localizations - the line below "
            "says by how much, and which stride makes them independent. The "
            "frame interval written into the TIFF is multiplied by this, so the "
            "movie still plays at the right speed."
        )
        range_row.addWidget(self.movie_stride_box)
        range_row.addStretch(1)
        buttons_layout.addLayout(range_row)

        # Orienting a structure for a figure. napari's own layer.rotate turns a
        # layer in the canvas but not the pixels, so it cannot reach a saved
        # file; this does the turning on the array on its way out.
        rotate_row = QHBoxLayout()
        rotate_row.addWidget(QLabel("Rotate"))
        self.render_rotate_box = QDoubleSpinBox()
        self.render_rotate_box.setRange(0.0, 359.9)
        self.render_rotate_box.setDecimals(1)
        self.render_rotate_box.setValue(0.0)
        self.render_rotate_box.setSuffix("°")
        self.render_rotate_box.setSingleStep(90.0)
        self.render_rotate_box.setToolTip(
            "Turn the image counter-clockwise before writing it, for orienting "
            "a structure in a figure.\n\n"
            "Quarter turns are exact - the pixels are permuted, never resampled, "
            "so a float32 reconstruction still holds the localization counts it "
            "held before.\n\n"
            "Any other angle resamples the image, which does not conserve the "
            "total: a sparse reconstruction can lose a large fraction of its "
            "counts, because interpolation samples the rotated grid rather than "
            "redistributing what was there. Use it for a figure, never for "
            "anything measured off the file afterwards. Which of the two "
            "happened is recorded in the metadata beside the image.\n\n"
            "The scale bar and clock are drawn after the turn, so they stay "
            "upright and readable."
        )
        rotate_row.addWidget(self.render_rotate_box)
        for label, angle in (("0°", 0.0), ("90°", 90.0), ("180°", 180.0), ("270°", 270.0)):
            button = QPushButton(label)
            button.setProperty("secondary", True)
            button.setMaximumWidth(52)
            button.setToolTip("Exact - no interpolation.")
            button.clicked.connect(
                lambda _c, a=angle: self.render_rotate_box.setValue(a))
            rotate_row.addWidget(button)
        self.render_rotate_label = QLabel()
        self.render_rotate_label.setProperty("role", "note")
        rotate_row.addWidget(self.render_rotate_label, 1)
        buttons_layout.addLayout(rotate_row)
        self.render_rotate_box.valueChanged.connect(
            lambda _v: self._update_rotate_label())
        self._update_rotate_label()

        self.movie_save_label = QLabel("Render a movie first.")
        self.movie_save_label.setWordWrap(True)
        self.movie_save_label.setProperty("role", "note")
        buttons_layout.addWidget(self.movie_save_label)
        for box in (self.movie_first_box, self.movie_last_box, self.movie_stride_box):
            box.valueChanged.connect(lambda _v: self._update_movie_save_label())

        self.render_save_status = QLabel("Nothing rendered yet.")
        self.render_save_status.setWordWrap(True)
        self.render_save_status.setProperty("role", "note")
        buttons_layout.addWidget(self.render_save_status)
        layout.addWidget(buttons_group)

        # --- output ---------------------------------------------------------
        output_group = QGroupBox("Output")
        output_layout = QFormLayout(output_group)
        self.render_add_layer_box = QCheckBox("Add the render to the viewer, aligned with the raw stack")
        self.render_add_layer_box.setChecked(True)
        output_layout.addRow("", self.render_add_layer_box)
        self.render_png_box = QCheckBox("Also write a contrast-stretched PNG next to the TIFF")
        self.render_png_box.setChecked(True)
        output_layout.addRow("", self.render_png_box)

        # Images default to the quantitative format and movies to the light one:
        # a still is usually the figure or the thing you measure on, while a
        # movie is almost always something you watch, and float32 makes it four
        # times larger for a precision nobody plays back.
        self.render_image_format_box = QComboBox()
        for key, label in RENDER_SAVE_FORMATS.items():
            self.render_image_format_box.addItem(label, key)
        self.render_image_format_box.setCurrentIndex(self.render_image_format_box.findData("data"))
        output_layout.addRow("Save image as", self.render_image_format_box)

        self.render_movie_format_box = QComboBox()
        for key, label in RENDER_SAVE_FORMATS.items():
            if key == "composite":
                continue  # screen-recorded from the viewer now, not written here
            self.render_movie_format_box.addItem(label, key)
        self.render_movie_format_box.setCurrentIndex(self.render_movie_format_box.findData("display"))
        output_layout.addRow("Save movie as", self.render_movie_format_box)

        save_note = QLabel(
            "<b>Data</b> keeps the render's own values as float32 - counts, or "
            "photons when weighted - so two exports stay comparable. "
            "<b>Display</b> is a contrast-stretched 8-bit copy, a quarter of the "
            "size, stretched once for the whole movie so frames don't pulse. "
            "<b>Composite</b> blends the layers below into RGB, for stills only - "
            "for a blended movie, screen-record the viewer instead (Playback, on "
            "the Load tab), which keeps every layer as it looks on screen. "
            "Every one gets a &lt;name&gt;_metadata.json recording the settings "
            "behind it - the same snapshot the analysis export writes."
        )
        save_note.setWordWrap(True)
        output_layout.addRow("", save_note)

        progress_row = QHBoxLayout()
        self.render_progress = QProgressBar()
        self.render_progress.setRange(0, 100)
        self.render_progress.setVisible(False)
        self.render_cancel_button = self._cancel_button(self._render_cancel, "rendering")
        progress_row.addWidget(self.render_progress)
        progress_row.addWidget(self.render_cancel_button)
        output_layout.addRow("", progress_row)
        layout.addWidget(output_group)

        # --- composite ------------------------------------------------------
        self.render_composite_group = QGroupBox("Composite layers (for the Composite format)")
        composite_layout = QFormLayout(self.render_composite_group)
        composite_note = QLabel(
            "Blended additively, like napari's additive layers: the "
            "reconstruction in its colormap, with the localizations and the "
            "trajectories drawn over it. In a movie each layer is grouped the "
            "same way as the reconstruction, so a trajectory appears while it "
            "is actually being tracked."
        )
        composite_note.setWordWrap(True)
        composite_layout.addRow("", composite_note)

        self.render_composite_base_box = QCheckBox("Super-resolved reconstruction")
        self.render_composite_base_box.setChecked(True)
        composite_layout.addRow("", self.render_composite_base_box)

        locs_row = QHBoxLayout()
        self.render_composite_locs_box = QCheckBox("Localizations")
        self.render_composite_locs_box.setChecked(False)
        self.render_locs_color_box = QComboBox()
        self.render_locs_color_box.addItems(list(smlm_render.OVERLAY_COLORS))
        self.render_locs_color_box.setCurrentText("cyan")
        self.render_locs_size_box = QDoubleSpinBox()
        self.render_locs_size_box.setRange(1.0, 2000.0)
        self.render_locs_size_box.setDecimals(0)
        self.render_locs_size_box.setValue(30.0)
        locs_row.addWidget(self.render_composite_locs_box)
        locs_row.addWidget(self.render_locs_color_box)
        locs_row.addWidget(QLabel("size (nm)"))
        locs_row.addWidget(self.render_locs_size_box)
        composite_layout.addRow("", locs_row)

        tracks_row = QHBoxLayout()
        self.render_composite_tracks_box = QCheckBox("Trajectories")
        self.render_composite_tracks_box.setChecked(False)
        self.render_tracks_color_box = QComboBox()
        self.render_tracks_color_box.addItems(list(smlm_render.OVERLAY_COLORS))
        self.render_tracks_color_box.setCurrentText("yellow")
        self.render_tracks_width_box = QDoubleSpinBox()
        self.render_tracks_width_box.setRange(1.0, 2000.0)
        self.render_tracks_width_box.setDecimals(0)
        self.render_tracks_width_box.setValue(30.0)
        tracks_row.addWidget(self.render_composite_tracks_box)
        tracks_row.addWidget(self.render_tracks_color_box)
        tracks_row.addWidget(QLabel("width (nm)"))
        tracks_row.addWidget(self.render_tracks_width_box)
        composite_layout.addRow("", tracks_row)

        self.render_composite_all_box = QCheckBox(
            "Every other visible layer too (the raw stack, candidates, ...)")
        self.render_composite_all_box.setToolTip(
            "Adds each visible Image and Points layer in the viewer, drawn with "
            "its own colormap or colour and its own contrast, sampled onto the "
            "super-resolved grid. Shapes layers - the filter and crop boxes - "
            "are controls rather than data, so they are left out."
        )
        composite_layout.addRow("", self.render_composite_all_box)

        self.render_composite_status = QLabel("-")
        self.render_composite_status.setWordWrap(True)
        composite_layout.addRow("", self.render_composite_status)
        layout.addWidget(self.render_composite_group)

        # --- time stamp and crop --------------------------------------------
        stamp_group = QGroupBox("Time stamp and crop")
        stamp_layout = QFormLayout(stamp_group)

        time_row = QHBoxLayout()
        self.render_timestamp_box = QCheckBox("Burn in the time")
        self.render_timestamp_box.setToolTip(
            "Drawn into the saved pixels, using the frame rate from the Link "
            "tab. A movie frame is labelled with the time its group starts at; "
            "a still is labelled with the span it covers."
        )
        self.render_timestamp_size_box = QSpinBox()
        self.render_timestamp_size_box.setRange(4, 2000)
        self.render_timestamp_size_box.setValue(40)
        self.render_timestamp_size_box.setSuffix(" px")
        self.render_timestamp_color_box = QComboBox()
        self.render_timestamp_color_box.addItems(list(smlm_render.OVERLAY_COLORS))
        self.render_timestamp_color_box.setCurrentText("white")
        self.render_timestamp_position_box = QComboBox()
        self.render_timestamp_position_box.addItems(
            ["top left", "top right", "bottom left", "bottom right"])
        time_row.addWidget(self.render_timestamp_box)
        time_row.addWidget(QLabel("height (px)"))
        time_row.addWidget(self.render_timestamp_size_box)
        time_row.addWidget(self.render_timestamp_color_box)
        time_row.addWidget(self.render_timestamp_position_box)
        stamp_layout.addRow("", time_row)

        bar_row = QHBoxLayout()
        self.render_scalebar_box = QCheckBox("Burn in a scale bar")
        self.render_scalebar_box.setChecked(True)
        self.render_scalebar_box.setToolTip(
            "Burns the bar into the pixels of the saved image or movie.\n\n"
            "For reading sizes on screen, use napari's own scale bar instead - "
            "it is always on, sits in the corner of the view, and follows the "
            "zoom. This one only affects the file that gets written.\n\n"
            "Both are drawn from the pixel size in the Data tab, so a wrong "
            "pixel size gives a confidently wrong bar."
        )
        self.render_scalebar_auto_box = QCheckBox("auto")
        self.render_scalebar_auto_box.setChecked(True)
        self.render_scalebar_auto_box.setToolTip(
            "Pick a round length - 1, 2 or 5 times a power of ten - covering "
            "about a seventh of the saved width, and keep it up to date as the "
            "field of view, pixel size or crop changes."
        )
        self.render_scalebar_length_box = QDoubleSpinBox()
        self.render_scalebar_length_box.setRange(1.0, 1e7)
        self.render_scalebar_length_box.setDecimals(0)
        self.render_scalebar_length_box.setValue(1000.0)
        self.render_scalebar_length_box.setSuffix(" nm")
        self.render_scalebar_length_box.setEnabled(False)
        self.render_scalebar_color_box = QComboBox()
        self.render_scalebar_color_box.addItems(list(smlm_render.OVERLAY_COLORS))
        self.render_scalebar_color_box.setCurrentText("white")
        self.render_scalebar_position_box = QComboBox()
        self.render_scalebar_position_box.addItems(
            ["bottom right", "bottom left", "top right", "top left"])
        bar_row.addWidget(self.render_scalebar_box)
        bar_row.addWidget(self.render_scalebar_auto_box)
        bar_row.addWidget(QLabel("length (nm)"))
        bar_row.addWidget(self.render_scalebar_length_box)
        bar_row.addWidget(self.render_scalebar_color_box)
        bar_row.addWidget(self.render_scalebar_position_box)
        stamp_layout.addRow("", bar_row)
        self.render_scalebar_status = QLabel("-")
        self.render_scalebar_status.setWordWrap(True)
        stamp_layout.addRow("", self.render_scalebar_status)

        self.render_scalebar_auto_box.stateChanged.connect(self._on_scalebar_auto_changed)
        self.render_scalebar_length_box.valueChanged.connect(
            lambda _v: self._update_scalebar_status())
        self.render_crop_box = QCheckBox("Save only what is inside the crop box")
        self.render_crop_box.setToolTip(
            "Puts a resizable rectangle on the image: drag the middle to move "
            "it, a handle to resize. Only the region inside it is written, at "
            "full resolution - the render itself still covers the whole field."
        )
        self.render_crop_box.stateChanged.connect(lambda _c: self._sync_render_crop_layer())
        stamp_layout.addRow("", self.render_crop_box)
        self.render_crop_status = QLabel("-")
        self.render_crop_status.setWordWrap(True)
        stamp_layout.addRow("", self.render_crop_status)
        layout.addWidget(stamp_group)
        layout.addStretch(1)

    def _sync_line_edits(self, first, second):
        """Keep two line edits showing the same path without looping forever."""
        def copy(source, target):
            def handler(text):
                if self._syncing_paths:
                    return
                self._syncing_paths = True
                try:
                    target.setText(text)
                finally:
                    self._syncing_paths = False
            return handler

        first.textChanged.connect(copy(first, second))
        second.textChanged.connect(copy(second, first))
        second.setText(first.text())

    def _build_immobility_group(self):
        """Did this molecule move at all? Asked without fitting anything.

        A static emitter is a fully specified object: its reported positions are
        its true position plus localization error, and that error is measured
        per spot by the localization fit itself. So the scatter of a trajectory,
        in units of its own precision, is chi-squared with 2(N-1) degrees of
        freedom under "this never moved" - exactly, at every trajectory length.

        This is the same question a small D is usually asked to answer, but it
        is asked directly. It costs one pass instead of a per-trajectory
        regression, it is at its best on the short trajectories where the MSD
        slope is at its worst, and it returns a probability rather than a number
        that has to be thresholded by eye.
        """
        group = QGroupBox("Immobility test (spread against localization error)")
        group.setToolTip(
            "Tests each trajectory against the hypothesis that the molecule "
            "never moved and every displacement was localization error.\n\n"
            "Filtering to p > 0.05 leaves the molecules that are immobile within "
            "your precision - render those and you have a super-resolved image "
            "of the bound population. Filtering to p < 0.05 leaves the ones that "
            "genuinely moved."
        )
        layout = QVBoxLayout(group)

        precision_row = QHBoxLayout()
        precision_row.addWidget(QLabel("Fallback precision"))
        self.immobility_sigma_box = QDoubleSpinBox()
        self.immobility_sigma_box.setRange(0.1, 10000.0)
        self.immobility_sigma_box.setDecimals(1)
        self.immobility_sigma_box.setValue(25.0)
        self.immobility_sigma_box.setSuffix(" nm")
        self.immobility_sigma_box.setToolTip(
            "Used only when the localization table has no uncertainty column. "
            "A single precision for every spot is a worse assumption than it "
            "looks: photon count varies several-fold between molecules, and "
            "averaging over that inflates the false-positive rate."
        )
        precision_row.addWidget(self.immobility_sigma_box)
        precision_row.addWidget(QLabel("× calibration"))
        self.immobility_calibration_box = QDoubleSpinBox()
        self.immobility_calibration_box.setRange(0.05, 20.0)
        self.immobility_calibration_box.setDecimals(3)
        self.immobility_calibration_box.setValue(1.0)
        self.immobility_calibration_box.setToolTip(
            "Scales the reported precision before testing.\n\n"
            "The one assumption this test makes from outside itself is that the "
            "reported uncertainty is the true localization error - and most "
            "fitters report a Cramér-Rao bound, which is a lower bound. A 20% "
            "underestimate makes half of a genuinely immobile population look "
            "mobile.\n\n"
            "It is checkable: over molecules you believe are immobile the median "
            "motion ratio must be 1.00. If it reads 1.41, set this to 1.19."
        )
        adaptive_steps(self.immobility_calibration_box)
        precision_row.addWidget(self.immobility_calibration_box)
        precision_row.addWidget(QLabel("detected at p <"))
        self.immobility_alpha_box = QDoubleSpinBox()
        self.immobility_alpha_box.setRange(1e-6, 0.5)
        self.immobility_alpha_box.setDecimals(6)
        self.immobility_alpha_box.setValue(0.05)
        self.immobility_alpha_box.setToolTip(
            "The significance at which motion counts as detected.\n\n"
            "The only free choice in the detection floor below - everything "
            "else comes from the trajectory's own length and precision and the "
            "frame interval. Tightening it raises the floor for every "
            "trajectory by about the same factor; it does not make short ones "
            "behave like long ones."
        )
        adaptive_steps(self.immobility_alpha_box)
        precision_row.addWidget(self.immobility_alpha_box)
        precision_row.addStretch(1)
        layout.addLayout(precision_row)
        self.immobility_alpha_box.valueChanged.connect(
            self._on_immobility_settings_changed)

        self.immobility_status_label = QLabel()
        self.immobility_status_label.setWordWrap(True)
        self.immobility_status_label.setProperty("role", "note")
        layout.addWidget(self.immobility_status_label)

        for key, title, low, high, decimals in (
            ("motion", "Motion ratio — 1.0 is a molecule that did not move", 0.0, 1e6, 4),
            ("pstatic", "p_static — filter to p_static > 0.05 for the immobile "
                        "population", 0.0, 1.0, 6),
            ("dmin", "Smallest detectable D — what this trajectory could have "
                     "ruled out, µm²/s", 0.0, 1e6, 6),
        ):
            sub = QGroupBox(title)
            sub_layout = QVBoxLayout(sub)
            bounds_row = QHBoxLayout()
            bounds_row.addWidget(QLabel("Min"))
            min_box = QDoubleSpinBox()
            min_box.setRange(low, high)
            min_box.setDecimals(decimals)
            bounds_row.addWidget(min_box)
            bounds_row.addWidget(QLabel("Max"))
            max_box = QDoubleSpinBox()
            max_box.setRange(low, high)
            max_box.setDecimals(decimals)
            max_box.setValue(1.0 if key == "pstatic" else 1000.0)
            adaptive_steps(min_box, max_box)
            bounds_row.addWidget(max_box)
            bounds_row.addWidget(self._make_metric_filter_box(key))
            sub_layout.addLayout(bounds_row)
            self._metric_bound_boxes[key] = (min_box, max_box)
            setattr(self, f"{key}_min_box", min_box)
            setattr(self, f"{key}_max_box", max_box)
            sub_layout.addWidget(self._make_metric_histogram(key))
            min_box.valueChanged.connect(lambda _v, k=key: self._on_metric_bounds_changed(k))
            max_box.valueChanged.connect(lambda _v, k=key: self._on_metric_bounds_changed(k))
            layout.addWidget(sub)

        self.immobility_sigma_box.valueChanged.connect(self._on_immobility_settings_changed)
        self.immobility_calibration_box.valueChanged.connect(self._on_immobility_settings_changed)
        return group

    def _build_population_fit_group(self):
        """Populations fitted to every trajectory at once.

        The static test can only say of a short trajectory that it could be
        standing still. The fit adds what the test lacks - how large a mobile
        molecule's steps usually are - and so turns even a two-point trajectory
        into a probability of being immobile.
        """
        group = QGroupBox("Populations (fitted to every trajectory)")
        form = QFormLayout(group)
        intro = QLabel(
            "An immobile population and one or two mobile ones, each with its "
            "own D, fitted to all trajectories at once by maximum likelihood - "
            "every localization with its own precision, motion blur and frame "
            "gaps accounted for. Each trajectory then gets a probability of "
            "belonging to each population, which the Render tab can sort by. "
            "The immobile population's D is kept at or below the Immobile limit "
            "on the Render tab."
        )
        intro.setWordWrap(True)
        intro.setProperty("role", "note")
        form.addRow(intro)

        self.population_mobile_box = QComboBox()
        for key, label in POPULATION_MOBILE_CHOICES.items():
            self.population_mobile_box.addItem(label, key)
        self.population_mobile_box.setToolTip(
            "How many mobile populations to fit beside the immobile one. Left to "
            "the fit, both are tried and a second is kept only if it improves "
            "the likelihood by more than its extra parameters cost (BIC)."
        )
        form.addRow("Mobile populations", self.population_mobile_box)

        buttons = QHBoxLayout()
        self.population_fit_button = QPushButton("Fit populations")
        self.population_fit_button.setProperty("primary", True)
        self.population_fit_button.clicked.connect(self.fit_populations)
        self.population_cancel_button = self._cancel_button(
            self._population_cancel, "the population fit")
        buttons.addWidget(self.population_fit_button)
        buttons.addWidget(self.population_cancel_button)
        buttons.addStretch(1)
        form.addRow("", buttons)
        self.population_progress = QProgressBar()
        self.population_progress.setRange(0, 100)
        self.population_progress.setVisible(False)
        form.addRow("", self.population_progress)

        self.population_fit_status = QLabel("Link trajectories, then fit.")
        self.population_fit_status.setWordWrap(True)
        form.addRow("", self.population_fit_status)

        self.population_figure = Figure(figsize=(5, 2.2))
        self.population_canvas = FigureCanvas(self.population_figure)
        self._plot_canvases.append(self.population_canvas)
        self.population_canvas.setMinimumHeight(200)
        form.addRow("", self.population_canvas)
        tools = QHBoxLayout()
        tools.addStretch(1)
        tools.addWidget(self._png_button(
            lambda fig, opts: self._draw_population_plot(fig, opts), lambda: "populations"))
        form.addRow("", tools)
        return group

    def _make_metric_filter_box(self, key):
        """The tick box that turns a metric's range from a colour scale into a
        selection.

        Deliberately the *same* min/max boxes rather than a second pair. Those
        bounds are already on screen, already drawn as draggable lines on the
        histogram beside them, and already saved with the run - so the range you
        have just set by eye on the distribution is the range you filter on, and
        there is no second set of numbers to keep in agreement with the first.
        """
        box = QCheckBox("filter")
        box.setToolTip(
            f"Show only trajectories whose {METRIC_LABELS[key].lower()} falls "
            "between the two values on the left - and only the localizations "
            "belonging to them.\n\n"
            "This carries through to the super-resolved reconstruction, so a "
            "render becomes a picture of the molecules that behaved this way "
            "rather than of all of them.\n\n"
            "Trajectories with no value for this metric are excluded."
        )
        box.stateChanged.connect(lambda _s, k=key: self._on_metric_filter_toggled(k))
        self._metric_filter_boxes[key] = box
        # Also as a plain attribute, which is how the settings machinery finds
        # a control by name when restoring a run or a session.
        setattr(self, f"{key.lower()}_filter_box", box)
        return box

    def _build_track_filter_group(self):
        """One line saying what the dynamics filter is currently doing."""
        group = QGroupBox("Dynamics filter")
        group.setToolTip(
            "Tick 'filter' beside any of the ranges below to show only the "
            "trajectories inside it. Several can be combined - a trajectory has "
            "to satisfy all of them."
        )
        layout = QVBoxLayout(group)
        self.track_filter_label = QLabel()
        self.track_filter_label.setWordWrap(True)
        self.track_filter_label.setProperty("role", "note")
        layout.addWidget(self.track_filter_label)
        row = QHBoxLayout()
        self.clear_track_filter_button = QPushButton("Show all trajectories again")
        self.clear_track_filter_button.setProperty("secondary", True)
        self.clear_track_filter_button.clicked.connect(self.clear_track_filters)
        self.clear_track_filter_button.setEnabled(False)
        row.addWidget(self.clear_track_filter_button)
        row.addStretch(1)
        layout.addLayout(row)
        return group

    def _build_track_tab(self):
        """Linking and trajectory analysis, in the order they are used.

        They were two tabs, but nobody links without then analysing: splitting
        them only meant switching tabs mid-thought and losing sight of the
        parameters that produced the trajectories being analysed.
        """
        tab = QWidget()
        self.tabs.addTab(tab, "Track")
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(tab)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)
        self._build_link_section(layout)
        self._build_trajectory_section(layout)
        layout.addWidget(self._build_track_filter_group())

    def _build_link_section(self, layout):

        tracking_group = QGroupBox("Tracking")
        tracking_layout = QFormLayout(tracking_group)
        # Acquisition timing lives here rather than in the diffusion tab: the
        # frame interval is what turns a search range into a diffusion
        # coefficient, and it is the lag time D is fitted against. One setting,
        # two views of it - edit whichever your acquisition software reports.
        self.fps_box = QDoubleSpinBox()
        self.fps_box.setRange(0.001, 1e5)
        self.fps_box.setDecimals(3)
        self.fps_box.setValue(100.0)
        self.fps_box.setSuffix(" fps")
        tracking_layout.addRow("Acquisition frame rate", self.fps_box)
        self.frame_interval_box = QDoubleSpinBox()
        self.frame_interval_box.setRange(0.01, 1e6)
        self.frame_interval_box.setDecimals(3)
        self.frame_interval_box.setValue(1000.0 / self.fps_box.value())
        self.frame_interval_box.setToolTip(
            "Time between consecutive frames, kept in sync with the frame rate above.\n"
            "Used both for the linking cutoff below and for D in the trajectory tab."
        )
        self.frame_interval_box.setSuffix(" ms")
        tracking_layout.addRow("Frame interval", self.frame_interval_box)
        self.fps_box.valueChanged.connect(self._on_fps_changed)
        # Playback speed is quoted against the acquisition rate, so the Load tab
        # has to hear about this even though the box lives here.
        self.fps_box.valueChanged.connect(lambda _v: self._update_playback_status())
        # The canvas clock turns frames into seconds with it, too.
        self.fps_box.valueChanged.connect(lambda _v: self._update_time_overlay())
        self.frame_interval_box.valueChanged.connect(self._on_frame_interval_changed)

        self.search_box = QDoubleSpinBox()
        self.search_box.setRange(1.0, 10000.0)
        self.search_box.setValue(250.0)
        self.search_box.setDecimals(0)
        self.search_box.valueChanged.connect(self._update_link_cutoff_label)
        tracking_layout.addRow("Search range (nm)", self.search_box)
        self.memory_box = QSpinBox()
        self.memory_box.setRange(0, 20)
        self.memory_box.setValue(1)
        self.memory_box.valueChanged.connect(self._update_link_cutoff_label)
        tracking_layout.addRow("Memory", self.memory_box)
        self.min_traj_box = QSpinBox()
        self.min_traj_box.setRange(1, 1000)
        self.min_traj_box.setValue(2)
        tracking_layout.addRow("Min track length", self.min_traj_box)
        self.link_cutoff_label = QLabel()
        self.link_cutoff_label.setWordWrap(True)
        self.link_cutoff_label.setToolTip(
            "For 2D Brownian motion the step length over a lag t is Rayleigh distributed\n"
            "with mean square <r^2> = 4Dt, so the fraction of steps longer than the search\n"
            "range R is exp(-R^2 / 4Dt). Requiring that to stay under the error rate gives\n"
            "    D_max = R^2 / (4 t ln(1/error)),\n"
            "which at 1% means the search range must be ~2.15x the RMS step.\n\n"
            "Memory extends the lag that has to be covered to (memory + 1) * frame interval.\n"
            "This is a single-particle bound: it does not account for wrong links, which\n"
            "come from density rather than step length."
        )
        tracking_layout.addRow("", self.link_cutoff_label)
        self.link_button = QPushButton("Link trajectories")
        self.link_button.setProperty("primary", True)
        self.link_button.clicked.connect(self.link_tracks)
        self.link_button.setEnabled(False)
        self.link_cancel_button = self._cancel_button(self._link_cancel, "linking")
        link_buttons = QHBoxLayout()
        link_buttons.addWidget(self.link_button)
        link_buttons.addWidget(self.link_cancel_button)
        tracking_layout.addRow("", link_buttons)
        self.link_progress = QProgressBar()
        self.link_progress.setRange(0, 100)
        self.link_progress.setVisible(False)
        tracking_layout.addRow("", self.link_progress)
        layout.addWidget(tracking_group)

        render_group = QGroupBox("Trajectory display")
        render_layout = QFormLayout(render_group)
        self.line_width_box = QDoubleSpinBox()
        self.line_width_box.setRange(0.5, 10.0)
        self.line_width_box.setValue(1.5)
        render_layout.addRow("Active track line width", self.line_width_box)

        self.traj_fade_box = QSpinBox()
        self.traj_fade_box.setRange(0, 1000000)
        self.traj_fade_box.setValue(0)
        # 0 is not "no tail" but "no limit", which is what the plain number
        # cannot say - napari fades the tail out over this many frames.
        self.traj_fade_box.setSpecialValueText("the whole trajectory")
        self.traj_fade_box.setSuffix(" frames")
        self.traj_fade_box.setToolTip(
            "How far behind the current frame a trajectory stays visible before "
            "it fades out. Short values show where things are moving now; the "
            "whole trajectory shows where they have been."
        )
        render_layout.addRow("Trail length", self.traj_fade_box)

        accumulate_row = QHBoxLayout()
        self.traj_accumulate_box = QCheckBox("Accumulate from frame")
        self.traj_accumulate_box.setToolTip(
            "Instead of a trail of fixed length, keep everything drawn from a "
            "chosen frame onwards, so the trajectories build up as the movie "
            "plays.\n\n"
            "The trail is regrown as the slider moves, which is what makes it "
            "reach further back the further in you are."
        )
        self.traj_start_frame_box = QSpinBox()
        self.traj_start_frame_box.setRange(0, 100_000_000)
        self.traj_start_frame_box.setValue(0)
        self.traj_start_frame_box.setEnabled(False)
        accumulate_row.addWidget(self.traj_accumulate_box)
        accumulate_row.addWidget(self.traj_start_frame_box, 1)
        render_layout.addRow("", accumulate_row)

        self.traj_fade_status = QLabel("-")
        self.traj_fade_status.setWordWrap(True)
        render_layout.addRow("", self.traj_fade_status)
        self.traj_fade_box.valueChanged.connect(lambda _v: self._on_fade_changed())
        self.traj_accumulate_box.stateChanged.connect(
            lambda _c: self._on_accumulate_changed())
        self.traj_start_frame_box.valueChanged.connect(lambda _v: self._on_fade_changed())
        self.fps_box.valueChanged.connect(lambda _v: self._update_fade_status())
        self.show_tracks_box = QCheckBox("Show trajectories (active, growing)")
        self.show_tracks_box.setChecked(True)
        render_layout.addRow("", self.show_tracks_box)
        self.persist_tracks_box = QCheckBox("Persist completed trajectories")
        self.persist_tracks_box.setChecked(True)
        render_layout.addRow("", self.persist_tracks_box)
        self.show_all_tracks_box = QCheckBox("Show all trajectories (static layer)")
        self.show_all_tracks_box.setChecked(False)
        render_layout.addRow("", self.show_all_tracks_box)
        self.all_tracks_line_width_box = QDoubleSpinBox()
        self.all_tracks_line_width_box.setRange(0.1, 10.0)
        self.all_tracks_line_width_box.setSingleStep(0.1)
        self.all_tracks_line_width_box.setValue(0.3)
        render_layout.addRow("Static layer line width", self.all_tracks_line_width_box)
        self.render_button = QPushButton("Render overlay")
        self.render_button.clicked.connect(self.render_overlay)
        self.render_button.setEnabled(False)
        render_layout.addRow("", self.render_button)
        layout.addWidget(render_group)
        layout.addStretch(1)

        # Each control touches only the layer it belongs to. Rebuilding a Tracks
        # layer costs seconds once there are a few thousand trajectories, so
        # anything that is only a style change is applied to the live layer.
        self.show_tracks_box.stateChanged.connect(lambda _checked: self._sync_tracks_layer())
        self.show_all_tracks_box.stateChanged.connect(lambda _checked: self._sync_all_tracks_layer())
        self.persist_tracks_box.stateChanged.connect(lambda _checked: self._apply_track_style())
        self.line_width_box.valueChanged.connect(lambda _v: self._apply_track_style())
        self.all_tracks_line_width_box.valueChanged.connect(lambda _v: self._apply_track_style())

    def _build_trajectory_section(self, layout):

        # Above the plots it sizes rather than below them: it used to sit in the
        # colouring group at the bottom of this tab, under eight histograms, and
        # a control you have to scroll past everything it affects to reach is a
        # control nobody finds.
        size_group = QGroupBox("Plot size (every graph in the plugin)")
        size_group.setToolTip(
            "Sizes the histograms below, the filter histograms and the MSD "
            "validation plot together, so a set of figures for a talk can be "
            "made the right shape once rather than one at a time."
        )
        size_layout = QVBoxLayout(size_group)
        size_layout.addLayout(self._build_plot_size_row())
        layout.addWidget(size_group)

        # --- D (requires a linear MSD fit) ---
        d_group = QGroupBox("Diffusion coefficient D (needs a linear MSD fit)")
        d_layout = QVBoxLayout(d_group)
        params_row = QFormLayout()
        self.max_lagtime_box = QSpinBox()
        self.max_lagtime_box.setRange(2, 200)
        self.max_lagtime_box.setValue(5)
        params_row.addRow("Max lag time (frames)", self.max_lagtime_box)
        self.d_min_length_box = QSpinBox()
        self.d_min_length_box.setRange(1, 10000)
        self.d_min_length_box.setValue(2)
        self.d_min_length_box.setToolTip(
            "Second, independent length filter, applied on top of the one used when\n"
            "linking. Set it higher than the linking filter to fit D only on the\n"
            "longer trajectories: a linear MSD fit on very few points is noisy, but\n"
            "short tracks are still worth keeping for display and for the fit-free\n"
            "metrics. Values at or below the linking filter change nothing."
        )
        params_row.addRow("Min track length for D (points)", self.d_min_length_box)
        timing_note = QLabel(
            "Frame rate / interval is set in the Link tab: the same number sets the "
            "lag time behind D and the step length the search range has to cover."
        )
        timing_note.setWordWrap(True)
        params_row.addRow("", timing_note)
        self.msd_sample_box = QSpinBox()
        self.msd_sample_box.setRange(1, 50)
        self.msd_sample_box.setValue(10)
        params_row.addRow("Example trajectories to validate", self.msd_sample_box)
        d_layout.addLayout(params_row)
        self.compute_d_button = QPushButton("Compute D")
        self.compute_d_button.setProperty("primary", True)
        self.compute_d_button.clicked.connect(self.compute_d)
        self.compute_d_button.setEnabled(False)
        self.compute_d_cancel_button = self._cancel_button(self._compute_d_cancel, "the D computation")
        d_buttons = QHBoxLayout()
        d_buttons.addWidget(self.compute_d_button)
        d_buttons.addWidget(self.compute_d_cancel_button)
        d_layout.addLayout(d_buttons)
        self.compute_d_progress = QProgressBar()
        self.compute_d_progress.setRange(0, 100)  # batched over trajectories, so real progress
        self.compute_d_progress.setVisible(False)
        d_layout.addWidget(self.compute_d_progress)

        d_bounds_row = QHBoxLayout()
        d_bounds_row.addWidget(QLabel("Min D"))
        self.d_min_box = QDoubleSpinBox()
        self.d_min_box.setRange(1e-6, 1e6)
        self.d_min_box.setDecimals(6)
        self.d_min_box.setValue(1e-4)
        d_bounds_row.addWidget(self.d_min_box)
        d_bounds_row.addWidget(QLabel("Max D"))
        self.d_max_box = QDoubleSpinBox()
        self.d_max_box.setRange(1e-6, 1e6)
        self.d_max_box.setDecimals(6)
        self.d_max_box.setValue(1e2)
        adaptive_steps(self.d_min_box, self.d_max_box)
        d_bounds_row.addWidget(self.d_max_box)
        d_bounds_row.addWidget(self._make_metric_filter_box("D"))
        d_layout.addLayout(d_bounds_row)
        self._metric_bound_boxes["D"] = (self.d_min_box, self.d_max_box)
        d_layout.addWidget(self._make_metric_histogram("D"))
        self.d_min_box.valueChanged.connect(lambda _v: self._on_metric_bounds_changed("D"))
        self.d_max_box.valueChanged.connect(lambda _v: self._on_metric_bounds_changed("D"))

        msd_sub = QGroupBox("MSD fit validation (sample trajectories + their linear fit)")
        msd_sub_layout = QVBoxLayout(msd_sub)
        self.msd_figure = Figure(figsize=(5, 2.8))
        self.msd_canvas = FigureCanvas(self.msd_figure)
        self._plot_canvases.append(self.msd_canvas)
        self.msd_canvas.setMinimumHeight(240)
        msd_sub_layout.addWidget(self.msd_canvas)
        # The intercept is fitted anyway - MSD = 4*D*tau + 4*sigma^2 - so the
        # localization precision it implies is free, and it is an estimate of
        # the same quantity the spot fitter reports by an entirely different
        # route. Reporting only the slope threw half the fit away.
        msd_tools = QHBoxLayout()
        msd_tools.addStretch(1)
        msd_tools.addWidget(self._png_button(
            lambda fig, opts: self._draw_msd_validation(fig, opts),
            lambda: "msd_validation"))
        msd_sub_layout.addLayout(msd_tools)
        self.msd_sigma_label = QLabel()
        self.msd_sigma_label.setWordWrap(True)
        self.msd_sigma_label.setProperty("role", "note")
        msd_sub_layout.addWidget(self.msd_sigma_label)
        d_layout.addWidget(msd_sub)
        layout.addWidget(d_group)

        # --- Distance travelled (fit-free) ---
        dist_group = QGroupBox("Distance travelled (fit-free: total path length)")
        dist_layout = QVBoxLayout(dist_group)
        dist_bounds_row = QHBoxLayout()
        dist_bounds_row.addWidget(QLabel("Min (µm)"))
        self.dist_min_box = QDoubleSpinBox()
        self.dist_min_box.setRange(0, 1e6)
        self.dist_min_box.setDecimals(6)
        dist_bounds_row.addWidget(self.dist_min_box)
        dist_bounds_row.addWidget(QLabel("Max (µm)"))
        self.dist_max_box = QDoubleSpinBox()
        self.dist_max_box.setRange(0, 1e6)
        self.dist_max_box.setDecimals(6)
        self.dist_max_box.setValue(1.0)
        adaptive_steps(self.dist_min_box, self.dist_max_box)
        dist_bounds_row.addWidget(self.dist_max_box)
        dist_bounds_row.addWidget(self._make_metric_filter_box("distance"))
        dist_layout.addLayout(dist_bounds_row)
        self._metric_bound_boxes["distance"] = (self.dist_min_box, self.dist_max_box)
        dist_layout.addWidget(self._make_metric_histogram("distance"))
        self.dist_min_box.valueChanged.connect(lambda _v: self._on_metric_bounds_changed("distance"))
        self.dist_max_box.valueChanged.connect(lambda _v: self._on_metric_bounds_changed("distance"))
        layout.addWidget(dist_group)

        # --- End-to-end displacement (fit-free) ---
        net_group = QGroupBox("End-to-end displacement (fit-free: start to finish)")
        net_group.setToolTip(
            "How far the molecule ended up from where it started, ignoring the "
            "route. Compare it with the path length above: the two are similar "
            "for directed motion and very different for a molecule that wandered."
        )
        net_layout = QVBoxLayout(net_group)
        net_bounds_row = QHBoxLayout()
        net_bounds_row.addWidget(QLabel("Min (µm)"))
        self.net_min_box = QDoubleSpinBox()
        self.net_min_box.setRange(0, 1e6)
        self.net_min_box.setDecimals(6)
        net_bounds_row.addWidget(self.net_min_box)
        net_bounds_row.addWidget(QLabel("Max (µm)"))
        self.net_max_box = QDoubleSpinBox()
        self.net_max_box.setRange(0, 1e6)
        self.net_max_box.setDecimals(6)
        self.net_max_box.setValue(1.0)
        adaptive_steps(self.net_min_box, self.net_max_box)
        net_bounds_row.addWidget(self.net_max_box)
        net_bounds_row.addWidget(self._make_metric_filter_box("net"))
        net_layout.addLayout(net_bounds_row)
        self._metric_bound_boxes["net"] = (self.net_min_box, self.net_max_box)
        net_layout.addWidget(self._make_metric_histogram("net"))
        self.net_min_box.valueChanged.connect(lambda _v: self._on_metric_bounds_changed("net"))
        self.net_max_box.valueChanged.connect(lambda _v: self._on_metric_bounds_changed("net"))
        layout.addWidget(net_group)

        # --- Straightness (fit-free) ---
        straight_group = QGroupBox("Straightness (end-to-end / path length)")
        straight_group.setToolTip(
            "The measure that separates directed motion from diffusion.\n\n"
            "A molecule travelling in a line approaches 1. An N-step random "
            "walk sits near 1/sqrt(N) however fast it diffuses - so with 25 "
            "steps, plain diffusion clusters around 0.2 and anything much above "
            "that is going somewhere.\n\n"
            "End-to-end displacement on its own cannot make that distinction, "
            "because a fast diffuser also ends up a long way from the start; it "
            "is the comparison with the path length that does."
        )
        straight_layout = QVBoxLayout(straight_group)
        straight_bounds_row = QHBoxLayout()
        straight_bounds_row.addWidget(QLabel("Min"))
        self.straight_min_box = QDoubleSpinBox()
        self.straight_min_box.setRange(0.0, 1.0)
        self.straight_min_box.setDecimals(3)
        self.straight_min_box.setSingleStep(0.05)
        straight_bounds_row.addWidget(self.straight_min_box)
        straight_bounds_row.addWidget(QLabel("Max"))
        self.straight_max_box = QDoubleSpinBox()
        self.straight_max_box.setRange(0.0, 1.0)
        self.straight_max_box.setDecimals(3)
        self.straight_max_box.setSingleStep(0.05)
        self.straight_max_box.setValue(1.0)
        adaptive_steps(self.straight_min_box, self.straight_max_box)
        straight_bounds_row.addWidget(self.straight_max_box)
        straight_layout.addLayout(straight_bounds_row)
        straight_bounds_row.addWidget(self._make_metric_filter_box("straightness"))
        self._metric_bound_boxes["straightness"] = (
            self.straight_min_box, self.straight_max_box)
        straight_layout.addWidget(self._make_metric_histogram("straightness"))
        self.straight_min_box.valueChanged.connect(
            lambda _v: self._on_metric_bounds_changed("straightness"))
        self.straight_max_box.valueChanged.connect(
            lambda _v: self._on_metric_bounds_changed("straightness"))
        layout.addWidget(straight_group)

        # --- Trajectory duration (fit-free) ---
        dur_group = QGroupBox("Trajectory duration (fit-free)")
        dur_layout = QVBoxLayout(dur_group)
        dur_bounds_row = QHBoxLayout()
        dur_bounds_row.addWidget(QLabel("Min (s)"))
        self.dur_min_box = QDoubleSpinBox()
        self.dur_min_box.setRange(0, 1e6)
        self.dur_min_box.setDecimals(3)
        dur_bounds_row.addWidget(self.dur_min_box)
        dur_bounds_row.addWidget(QLabel("Max (s)"))
        self.dur_max_box = QDoubleSpinBox()
        self.dur_max_box.setRange(0, 1e6)
        self.dur_max_box.setDecimals(3)
        self.dur_max_box.setValue(10.0)
        adaptive_steps(self.dur_min_box, self.dur_max_box)
        dur_bounds_row.addWidget(self.dur_max_box)
        dur_bounds_row.addWidget(self._make_metric_filter_box("duration"))
        dur_layout.addLayout(dur_bounds_row)
        self._metric_bound_boxes["duration"] = (self.dur_min_box, self.dur_max_box)
        dur_layout.addWidget(self._make_metric_histogram("duration"))
        self.dur_min_box.valueChanged.connect(lambda _v: self._on_metric_bounds_changed("duration"))
        self.dur_max_box.valueChanged.connect(lambda _v: self._on_metric_bounds_changed("duration"))
        layout.addWidget(dur_group)

        layout.addWidget(self._build_immobility_group())
        layout.addWidget(self._build_population_fit_group())

        # --- Coloring ---
        color_group = QGroupBox("Trajectory coloring")
        color_layout = QFormLayout(color_group)
        self.color_trajectories_box = QCheckBox("Color trajectories by the metric below")
        self.color_trajectories_box.setToolTip(
            "Off by default: every trajectory gets its own colour, which is what "
            "makes neighbouring tracks tellable apart. Tick this to spend the "
            "colours on a measurement instead - including time, which is then "
            "one metric among the others rather than the default."
        )
        color_layout.addRow("", self.color_trajectories_box)
        self.color_metric_box = QComboBox()
        self.color_metric_box.addItems([
            "D (diffusion coefficient)", "Distance travelled",
            "End-to-end displacement", "Straightness (directed vs diffusive)",
            "Track duration",
            "Motion ratio (moved vs its own precision)",
            "p_static (consistent with a static emitter)",
            "Smallest detectable D",
            "Time (frame first seen)",
        ])
        color_layout.addRow("Metric", self.color_metric_box)
        self.d_colormap_box = QComboBox()
        self.d_colormap_box.addItems(D_COLORMAP_CHOICES)
        self.d_colormap_box.setCurrentText(DEFAULT_D_COLORMAP)
        color_layout.addRow("Colormap", self.d_colormap_box)
        apply_row = QHBoxLayout()
        self.live_display_box = QCheckBox("Update live")
        self.live_display_box.setChecked(True)
        self.live_display_box.setToolTip(
            "Recolour the trajectories as soon as a bound changes. Uncheck to set\n"
            "several values first and apply them in one go."
        )
        apply_row.addWidget(self.live_display_box)
        self.apply_display_button = QPushButton("Apply display settings")
        self.apply_display_button.clicked.connect(self.apply_display_settings)
        apply_row.addWidget(self.apply_display_button)
        apply_row.addStretch(1)
        color_layout.addRow("", apply_row)
        layout.addWidget(color_group)

        self.color_trajectories_box.stateChanged.connect(self._on_color_mode_changed)
        self.color_metric_box.currentTextChanged.connect(self._on_color_settings_changed)
        self.d_colormap_box.currentTextChanged.connect(self._on_color_settings_changed)

        # Export itself lives in the header, reachable from every tab - it was
        # previously offered from two different tabs, wired to the same handler.
        export_note = QLabel(
            "\"Export...\" in the header above saves every plot, the filtered "
            "localizations, linked trajectories, per-track metrics and a "
            "metadata.json of the parameters used, into a new \"analysis\" "
            "folder next to the source data."
        )
        export_note.setWordWrap(True)
        export_note.setProperty("role", "note")
        layout.addWidget(export_note)
        layout.addStretch(1)

    def _build_data_table_dialog(self):
        self.data_table_dialog = QDialog(self)
        self.data_table_dialog.setWindowTitle("Localizations")
        self.data_table_dialog.resize(900, 600)
        layout = QVBoxLayout(self.data_table_dialog)
        self.data_table_label = QLabel("No data loaded")
        layout.addWidget(self.data_table_label)
        self.data_table_model = PandasTableModel()
        self.data_table_view = QTableView()
        self.data_table_view.setModel(self.data_table_model)
        layout.addWidget(self.data_table_view)

    def log(self, message):
        self.log_box.appendPlainText(message)

    def browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select localization CSV", filter="CSV files (*.csv)")
        if path:
            self.csv_edit.setText(path)

    def browse_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", filter="Image files (*.tif *.tiff *.png *.jpg *.jpeg)")
        if path:
            self.image_edit.setText(path)

    def show_data_table(self):
        self.data_table_model.set_dataframe(self.df_filtered)
        if self.df_filtered is not None:
            self.data_table_label.setText(f"{len(self.df_filtered)} rows x {len(self.df_filtered.columns)} columns")
        self.data_table_dialog.show()
        self.data_table_dialog.raise_()

    def _frame_offset(self):
        return int(self._frame_shift)

    def shift_frame_numbers(self, step):
        """Move every localization's frame number, or put it back (step=None).

        Applied as an offset rather than by rewriting the frame column, so it
        stays reversible and the loaded table keeps the numbers the file
        actually contained; everything that consumes frames - the overlays,
        the movie grouping, the linking - reads it through `_frame_offset`.
        """
        self._frame_shift = 0 if step is None else self._frame_shift + int(step)
        self._update_frame_shift_label()
        if self.df is None:
            return
        self.log(
            f"Frame numbers shifted by {self._frame_shift:+d}" if self._frame_shift
            else "Frame numbers back to the values in the file"
        )
        self._invalidate_tracks(reason="frame numbers shifted")
        # Each localization now belongs to a different frame, and so to a
        # different moment of the drift.
        self._recorrect_table()
        self.apply_filters()
        self._show_drift_state()

    def _update_frame_shift_label(self):
        if not hasattr(self, "frame_shift_label"):
            return
        if not self._frame_shift:
            self.frame_shift_label.setText("no shift")
        else:
            first = ""
            frames = self._render_frames()
            if frames is not None and frames.size:
                first = f", first frame now {int(frames.min())}"
            self.frame_shift_label.setText(f"{self._frame_shift:+d}{first}")
        self.frame_shift_reset_button.setEnabled(bool(self._frame_shift))

    # ------------------------------------------------------------------
    # Data loading (background)
    # ------------------------------------------------------------------
    def load_data(self):
        csv_path = self.csv_edit.text().strip()
        image_path = self.image_edit.text().strip()
        if not csv_path and not image_path:
            self.log("Choose a localization CSV, an image stack, or both.")
            return
        if csv_path and not os.path.exists(csv_path):
            self.log("Please choose a valid CSV file.")
            return
        if image_path and not os.path.exists(image_path):
            self.log("Please choose a valid image file.")
            return

        self.log("Loading data in the background...")
        self.load_button.setEnabled(False)
        self.load_progress.setVisible(True)
        self._arm_cancel(self._load_cancel, self.load_cancel_button)

        worker = _load_worker(csv_path, image_path,
                              int(self.bin_factor_box.value()), self._load_cancel)
        worker.returned.connect(lambda result: self._on_load_finished(result, csv_path, image_path))
        worker.errored.connect(self._on_load_errored)
        worker.finished.connect(self._on_load_worker_finished)
        self._load_worker_ref = worker
        worker.start()

    def _on_load_worker_finished(self):
        self.load_button.setEnabled(True)
        self.load_cancel_button.setEnabled(False)
        self.load_progress.setVisible(False)
        self._load_worker_ref = None
        self._session_advance()

    def _on_load_errored(self, exc):
        self.log(f"Failed to load data: {exc}")

    def _on_load_finished(self, result, csv_path, image_path):
        if result is CANCELLED:
            self.log("Loading cancelled")
            return
        # The raw stack is a later addition to the result; older callers (and
        # the tests that stand in for a worker) still hand over four values.
        df, image, how, acquisition = result[:4]
        self._raw_image = result[4] if len(result) > 4 else image
        # A different stack is a different analysis, even before any
        # localizations arrive from it.
        self._start_new_output_folders()
        # Loading can outrun the debounce on the binning box: the stack has just
        # been opened at the new factor, so the baseline and frame rate move with
        # it here rather than waiting for a timer that would then find the factor
        # already applied and do nothing. Before the autofill, which reads it.
        self._time_bin_timer.stop()
        factor = int(self.bin_factor_box.value())
        if factor != self._time_bin_applied:
            self._rescale_for_time_binning(max(1, int(self._time_bin_applied)), factor)
        self._time_bin_applied = factor
        # Before anything else moves: the pixel-size autofill below redraws the
        # shifted image, which must not use the previous dataset's record. And
        # before any table is ingested, so it arrives already corrected rather
        # than on screen uncorrected for a moment and then moved.
        self._find_drift_record()
        self.viewer.layers.clear()
        if image is not None:
            self._image_layer_name = Path(image_path).name
            self.viewer.add_image(
                image, name=self._image_layer_name, colormap="gray",
                **self._placed({}, image.ndim))
            self.log(f"Image {tuple(image.shape)} {image.dtype}: {how}")
            # The pixel size may move here, so the units follow rather than lead.
            self._apply_acquisition_metadata(self._with_frame_clock(acquisition))
            # After the autofill, so the binned exposure it reports is the one
            # the metadata just set rather than the one it replaced.
            self._update_time_bin_label()
            self._apply_viewer_scale()
            # A stack measured in nanometres is hundreds of times "bigger" than
            # one measured in pixels, and a camera left where the previous world
            # put it opens somewhere deep inside the first field of view.
            self._reset_view()

        auto_loaded = False
        if df is None and not csv_path and image_path:
            found = self._find_companion_file(Path(image_path), LOCS_FILENAME_PATTERNS, LOCS_ANALYSIS_SUBPATH)
            if found is not None:
                try:
                    df = pd.read_csv(found)
                    csv_path = str(found)
                    self.csv_edit.setText(csv_path)
                    auto_loaded = True
                    # Before ingesting: filter bounds restored now are applied to
                    # the table as it arrives, rather than leaving a filtered
                    # export on screen under controls that say otherwise.
                    self._restore_previous_run_settings(found)
                except Exception as exc:
                    self.log(f"Found candidate localizations file {found.name} but could not read it: {exc}")

        if df is not None:
            prefix = "Auto-detected and loaded" if auto_loaded else "Loaded"
            self._ingest_localization_dataframe(
                df,
                f"{prefix} {len(df)} localizations from {Path(csv_path).name}",
                frame_is_zero_indexed=False,
            )
            base = Path(csv_path) if csv_path else Path(image_path)
            self._try_autoload_trajectories(base)
        elif image is not None:
            self.log(
                f"Loaded image stack from {Path(image_path).name}. "
                "Use the Localize (2D) tab to detect and fit localizations, "
                "or load a CSV above to import localizations from elsewhere."
            )
            self._show_drift_state()

    def _with_frame_clock(self, acquisition):
        """The acquisition metadata, with the frame clock's interval if it has none.

        The frame-times file is the camera's own timing, measured page by page
        as the movie was taken, so it is a measurement and not a guess - but
        only a fallback: an interval the image metadata recorded wins. Without
        it, an acquisition that writes no Micro-Manager metadata leaves the
        frame rate at whatever the box last held, and every D inherits that.
        """
        clock = self._frame_clock
        values = dict((acquisition or {}).get("values") or {})
        sources = dict((acquisition or {}).get("sources") or {})
        if clock is None or not clock.slope > 0 or "frame_interval_ms" in values:
            return acquisition
        values["frame_interval_ms"] = clock.slope * 1000.0
        values["fps"] = 1.0 / clock.slope
        where = (f"{clock.path.name}: the frame clock fitted over "
                 f"{clock.n_recorded} pages")
        sources["frame_interval_ms"] = sources["fps"] = where
        return {"values": values, "sources": sources}

    def _apply_acquisition_metadata(self, acquisition):
        """Fill the acquisition parameters in from what the microscope recorded.

        Every change is logged with the value it replaced and the field it came
        from. That is the whole safety net here: these boxes decide what the
        physical results mean, so silently moving one would be worse than not
        moving it at all, and the log line is what lets the user notice and undo
        a value that belongs to a different microscope than they thought.
        """
        values = (acquisition or {}).get("values") or {}
        sources = (acquisition or {}).get("sources") or {}
        self._acquisition_values = dict(values)
        # Where the image sits on the sensor belongs to this acquisition alone:
        # never carried over from the previous dataset, whatever else is.
        self._sensor_roi_recorded = "sensor_roi_x" in values and "sensor_roi_y" in values
        if hasattr(self, "wl_roi_x_box") and not self._sensor_roi_recorded:
            for box in (self.wl_roi_x_box, self.wl_roi_y_box):
                box.blockSignals(True)
                box.setValue(0)
                box.blockSignals(False)
        if not values:
            self.log("No acquisition metadata found alongside the image - "
                     "the parameters in the Data tab are unchanged.")
            return

        factor = max(1, int(self._time_bin_applied))
        for key, attr, label, template in ACQUISITION_AUTOFILL:
            if key not in values:
                continue
            box = getattr(self, attr, None)
            if box is None:
                continue
            # The microscope wrote down what a single raw frame did; the
            # pipeline sees sums of `factor` of them.
            recorded = float(values[key]) * factor ** ACQUISITION_BIN_EXPONENT.get(key, 0)
            before = box.value()
            box.setValue(int(round(recorded)) if isinstance(box, QSpinBox) else recorded)
            after = box.value()  # setValue clamps to the control's range
            if abs(after - before) <= 1e-9:
                continue
            binned = ("" if factor == 1 or key not in ACQUISITION_BIN_EXPONENT
                      else f", for {factor}-frame bins")
            self.log(f"{label}: {template.format(before)} -> "
                     f"{template.format(after)}, from "
                     f"{sources.get(key, 'the metadata')}{binned}")

        context = [f"{label} {template.format(values[key])}"
                   for key, label, template in ACQUISITION_CONTEXT if key in values]
        if context:
            self.log("Acquisition: " + ", ".join(context))

        if "pixel_size_nm" not in values:
            # Micro-Manager records 0.0 for an objective nobody calibrated, and
            # the magnification alone cannot recover it without the sensor pitch.
            note = ("Pixel size is not recorded in this acquisition, so "
                    f"{self.pixel_size_box.value():.2f} nm/px was left as it was")
            if "objective" in values:
                note += f" - check it against the {values['objective']} objective"
            self.log(note + ".")
        self._update_scalebar_status()

    # ------------------------------------------------------------------
    # Time binning: summing raw frames in groups before anything else
    # ------------------------------------------------------------------
    def _update_time_bin_label(self):
        factor = int(self.bin_factor_box.value())
        if factor <= 1:
            self.bin_label.setText("off")
            return
        parts = [f"summed in {factor}s"]
        if self._raw_image is not None:
            n_raw = int(self._raw_image.shape[0])
            parts.append(f"{n_raw} -> {n_raw // factor} frames")
        parts.append(f"{self._frame_interval_s() * 1000:.1f} ms each")
        self.bin_label.setText(", ".join(parts))

    def _rescale_for_time_binning(self, previous, factor):
        """Move the per-frame quantities from one binning factor to another.

        The camera baseline and the frame rate describe the frames the pipeline
        is handed, not the frames the camera wrote. Summing N of them multiplies
        the baseline by N and divides the frame rate by N, and neither is
        recoverable afterwards: a baseline left at its raw value would be
        under-subtracted N-fold, and a frame rate left too fast would scale
        every diffusion coefficient by exactly the same factor - a wrong answer
        that looks entirely reasonable.
        """
        ratio = float(factor) / float(previous)
        offset_before = self.loc_offset_box.value()
        self.loc_offset_box.setValue(offset_before * ratio)
        fps_before = self.fps_box.value()
        self.fps_box.setValue(fps_before / ratio)  # the frame interval follows
        self.log(
            f"Time binning {previous} -> {factor}: camera offset "
            f"{offset_before:.0f} -> {self.loc_offset_box.value():.0f} ADU, "
            f"frame rate {fps_before:.3f} -> {self.fps_box.value():.3f} fps"
        )

    def _apply_time_binning(self):
        """Act on a change to the binning factor: rescale, then re-bin the stack."""
        factor = int(self.bin_factor_box.value())
        previous = max(1, int(self._time_bin_applied))
        if factor == previous:
            return
        self._time_bin_applied = factor
        self._rescale_for_time_binning(previous, factor)
        self._update_time_bin_label()
        if self._raw_image is None:
            # Nothing open yet; the factor is picked up by the next load. A
            # table loaded on its own still has its frames re-timed.
            self._refresh_drift_correction()
            return
        self._rebin_loaded_stack(factor)

    def _image_layer(self):
        """The layer holding the loaded stack, by name and then by kind."""
        if self._image_layer_name and self._image_layer_name in self.viewer.layers:
            return self.viewer.layers[self._image_layer_name]
        return self._get_localize_image_layer()

    def _rebin_loaded_stack(self, factor):
        # A superseded run is told to stop through the flag it was started with,
        # and this one gets a fresh flag - clearing the shared one instead would
        # un-cancel the run that has not noticed yet.
        self._bin_cancel.set()
        self._bin_cancel = threading.Event()
        self.bin_factor_box.setEnabled(False)
        self.load_progress.setVisible(True)
        worker = _bin_worker(self._raw_image, factor, self._bin_cancel)
        worker.returned.connect(self._on_rebin_finished)
        worker.errored.connect(lambda exc: self.log(f"Time binning failed: {exc}"))
        worker.finished.connect(self._on_rebin_worker_finished)
        self._bin_worker_ref = worker
        worker.start()

    def _on_rebin_worker_finished(self):
        self.bin_factor_box.setEnabled(True)
        self.load_progress.setVisible(False)
        self._bin_worker_ref = None

    def _on_rebin_finished(self, result):
        if result is CANCELLED:
            self.log("Time binning cancelled")
            return
        image, how = result
        layer = self._image_layer()
        if layer is None:
            self.log("Time binning: the image layer is gone; reload to apply it.")
            return
        layer.data = image
        self.log(f"Image {tuple(image.shape)} {image.dtype}: {how}")
        # Candidates are indexed by frame, and the frames have just been
        # renumbered. Anything detected against the old binning is meaningless
        # now, so it goes rather than being silently misapplied.
        self._loc2d_candidates = [None] * int(image.shape[0])
        self._loc2d_counts = np.zeros(int(image.shape[0]), dtype=int)
        self._update_loc2d_candidate_overlay()
        self._update_time_bin_label()
        if self.df is not None and len(self.df):
            self.log("The loaded localizations were produced at a different "
                     "binning - re-run detection and fitting before trusting them.")
        # A binned frame spans different moments than the frame it replaced,
        # and the new layer data is unshifted.
        self._refresh_drift_correction()

    # ------------------------------------------------------------------
    # Drift correction: the white-light camera's record, applied
    # ------------------------------------------------------------------
    def browse_drift(self):
        folder = self._drift_search_folder()
        path, _ = QFileDialog.getOpenFileName(
            self, "Select the xy drift record", str(folder or ""),
            filter="xy drift records (*_xy_drift.csv);;CSV files (*.csv)")
        if path:
            self._read_drift_files(path)
            self._refresh_drift_correction()

    def _on_drift_path_edited(self):
        text = self.drift_edit.text().strip()
        current = str(self._drift_record.path) if self._drift_record is not None else ""
        if text == current:
            return
        self._read_drift_files(text or None)
        self._refresh_drift_correction()

    def _drift_search_folder(self):
        """Where the acquisition's files are: beside the stack, else the table's dataset."""
        image_path = self.image_edit.text().strip()
        if image_path:
            return Path(image_path).parent
        csv_path = self.csv_edit.text().strip()
        if csv_path:
            return self._dataset_dir(csv_path)
        return None

    def _stack_stem(self):
        image_path = self.image_edit.text().strip()
        return Path(image_path).stem if image_path else None

    def _find_drift_record(self):
        """Pick up the drift record of the data just loaded, if it has one.

        A session names the record it used, and that wins - it may have been
        chosen by hand. Otherwise it is looked for beside the stack, where the
        acquisition writes it. Looked for afresh on every load: a record left
        over from the previous dataset would move this one by another sample's
        drift.
        """
        chosen = None
        plan = self._session_restore
        if plan is not None:
            recorded = (plan["manifest"].get("sources") or {}).get("drift")
            chosen = session_io.resolve_source(recorded, plan["session_dir"])
        folder = self._drift_search_folder()
        if chosen is None:
            chosen = drift_io.find_drift_file(folder, self._stack_stem()) if folder else None
        self._read_drift_files(chosen)
        if self._drift_record is None and folder is not None:
            self._read_snapshots_without_record(folder)
        if plan is not None:
            # An RCC estimate is a result, not a setting, but re-running it on
            # every restore would cost seconds and could land elsewhere; the
            # segment table it found is small, and travels with the session.
            section = ((plan["manifest"].get("settings") or {}).get("drift_correction")
                       or {}).get("rcc") or {}
            self._rcc = self._rcc_from_metadata(section)

    def _read_snapshots_without_record(self, folder):
        """The white-light snapshots and frame clock of an acquisition that kept
        no drift record: enough to lay the white light over the fluorescence,
        nothing to correct with."""
        tif, times = drift_io.find_wl_images_for_stack(folder, self._stack_stem())
        if tif is None:
            return
        try:
            self._wl_stack = drift_io.read_wl_images(tif, times)
        except Exception as exc:
            self.log(f"Found {tif.name} but could not read it: {exc}")
            return
        times_path = drift_io.find_frame_times(folder, self._stack_stem())
        if times_path is not None:
            try:
                self._frame_clock = drift_io.read_frame_times(times_path)
            except Exception as exc:
                self.log(f"Could not read the frame times {times_path.name}: {exc}")
        self.log(f"{len(self._wl_stack)} white-light snapshots beside the stack, and no "
                 "drift record - the Drift tab can still lay them over the fluorescence")

    def _find_deformation(self, record_path):
        """The newest growth deformation measured on this record's snapshots, if any."""
        record_path = Path(record_path)
        folders = []
        for folder in (self._drift_search_folder(), record_path.parent):
            if folder is not None and folder not in folders:
                folders.append(folder)
        for folder in folders:
            for run_dir in self._analysis_run_dirs(folder):
                candidate = run_dir / deform_io.DEFORMATION_FILENAME
                try:
                    if not candidate.is_file():
                        continue
                    found = deform_io.DeformationRecord.load(candidate)
                except Exception as exc:
                    self.log(f"Could not read {candidate}: {exc}")
                    continue
                source = Path(str(found.source.get("drift_record", ""))).name
                if source and source != record_path.name:
                    continue
                self._deformation, self._deformation_path = found, candidate
                along, _across = found.strain_along_axis()
                self.log(f"Growth deformation from {candidate.parent.name}: "
                         f"{100 * along[0]:+.1f} % along the root axis over "
                         f"{(found.t[-1] - found.t[0]) / 60:.0f} min")
                self._update_growth_status()
                return found
        self._update_growth_status()
        return None

    def _update_growth_status(self):
        if not hasattr(self, "growth_status"):
            return
        busy = self._deform_worker_ref is not None
        self.growth_measure_button.setEnabled(
            not busy and self._wl_stack is not None and self._drift_record is not None)
        matching = (self._deformation is not None and self._wl_stack is not None
                    and len(self._wl_stack) == len(self._deformation.t))
        self.growth_movie_button.setEnabled(
            not busy and self._movie_worker_ref is None and matching)
        self.growth_show_button.setEnabled(not busy and matching
                                           and self._drift_record is not None)
        if self._deformation is None:
            text = ("No growth deformation measured for this record."
                    if self._drift_record is not None else "No drift record loaded.")
            if self.drift_mode_box.currentData() == "growth":
                # Kept as chosen - a session restores it before its data - but said.
                text += (" Growth is chosen, so until one is measured the drift record "
                         "is applied as a translation.")
            self.growth_status.setText(text)
            return
        d = self._deformation
        along, across = d.strain_along_axis()
        passes = d.stats.get("passes") or [{}]
        lags = passes[-1].get("median_abs_residual_px_by_lag") or {}
        wl_nm = self._camera_map.scale * self.pixel_size_box.value() if self._camera_map else 0.0
        first_lag = min(lags, key=int) if lags else None
        last_lag = max(lags, key=int) if lags else None
        precision = (f"; patches agree to {lags[first_lag] * wl_nm:.0f} nm 30 s apart, "
                     f"{lags[last_lag] * wl_nm:.0f} nm across the recording"
                     if lags and wl_nm else "")
        self.growth_status.setText(
            f"Measured ({self._deformation_path.parent.name}): the tissue stretched "
            f"{100 * along[0]:+.1f} % along the root axis ({np.degrees(d.theta):.0f} deg) and "
            f"{100 * across[0]:+.1f} % across it over {(d.t[-1] - d.t[0]) / 60:.0f} min"
            + precision + ".")

    def measure_growth(self):
        """Measure the growth deformation on the snapshots, in the background."""
        stack, cmap = self._wl_stack, self._camera_map
        if stack is None or self._drift_record is None:
            self.log("The growth is measured on the white-light snapshots saved with a drift record")
            return
        region = None
        if cmap is not None and cmap.offset is not None:
            name = self._image_layer_name
            shape = (getattr(drift_io.unshifted(self.viewer.layers[name].data), "shape", None)
                     if name and name in self.viewer.layers else None)
            if shape is not None:
                h, w = shape[-2:]
                ox, oy = self.wl_roi_x_box.value(), self.wl_roi_y_box.value()
                corners = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5],
                                    [-0.5, h - 0.5]]) + [ox, oy]
                J = np.asarray(cmap.matrix)
                b = np.linalg.solve(J, (corners - np.asarray(cmap.offset)).T).T
                margin = 150
                region = (b[:, 0].min() - margin, b[:, 0].max() + margin,
                          b[:, 1].min() - margin, b[:, 1].max() + margin)
        if region is None:
            top, left, h, w = self._drift_record.meta.get("roi_top_left_h_w", (0, 0, 0, 0))
            region = (left, left + w, top, top + h) if w and h else None
        folder = self._drift_search_folder() or self._drift_record.path.parent
        self.log(f"Measuring the growth on {len(stack)} snapshots"
                 + (f" over x {region[0]:.0f}-{region[1]:.0f}, y {region[2]:.0f}-{region[3]:.0f} "
                    "white-light px" if region else "")
                 + (" on the GPU..." if deform_io.array_backend()[1] == "gpu"
                    else " on the CPU - a minute or so..."))
        self.growth_measure_button.setEnabled(False)
        self.growth_progress.setVisible(True)
        self.growth_progress.setValue(0)
        self._arm_cancel(self._deform_cancel, self.growth_cancel_button)
        record = self._drift_record
        worker = _deform_worker(stack, region, self._deform_cancel)
        worker.yielded.connect(lambda frac: self.growth_progress.setValue(int(frac * 100)))
        worker.returned.connect(lambda result: self._on_growth_measured(result, folder, record))
        worker.errored.connect(lambda exc: self.log(f"Measuring the growth failed: {exc}"))
        worker.finished.connect(self._on_deform_worker_finished)
        self._deform_worker_ref = worker
        worker.start()

    def make_growth_movie(self):
        """A movie of the growth deformation - how it was found, what it undoes."""
        record, stack = self._deformation, self._wl_stack
        if record is None or stack is None:
            self.log("Measure the growth deformation first - the movie shows it")
            return
        if len(stack) != len(record.t):
            self.log(f"The deformation has {len(record.t)} snapshots and the loaded stack "
                     f"{len(stack)} - measure again on these snapshots first")
            return
        translation = None
        if self._drift_record is not None:
            r = self._drift_record
            translation = drift_io.smoother(r.t, r.dx, r.dy, self.drift_smoothing_box.value())
        folder = (Path(self._deformation_path).parent if self._deformation_path is not None
                  else Path.cwd())
        if not record.steps:
            self.log("This deformation was measured before its intermediate steps were kept: "
                     "the movie shows the time-lapse only. Measure again for the steps too.")
        self.log("Drawing the deformation movie...")
        self.growth_movie_button.setEnabled(False)
        self.growth_progress.setVisible(True)
        self.growth_progress.setValue(0)
        self._arm_cancel(self._movie_cancel, self.growth_cancel_button)
        worker = _deform_movie_worker(record, stack, translation,
                                      folder / deform_movie.MOVIE_FILENAME, self._movie_cancel)
        worker.yielded.connect(lambda frac: self.growth_progress.setValue(int(frac * 100)))
        worker.returned.connect(self._on_growth_movie_made)
        worker.errored.connect(lambda exc: self.log(f"The deformation movie failed: {exc}"))
        worker.finished.connect(self._on_movie_worker_finished)
        self._movie_worker_ref = worker
        worker.start()

    def _on_growth_movie_made(self, result):
        if result is CANCELLED or result is None:
            self.log("Deformation movie cancelled")
            return
        self.log(f"Deformation movie saved: {result}")

    def _on_movie_worker_finished(self):
        self.growth_cancel_button.setEnabled(False)
        self.growth_progress.setVisible(False)
        self._movie_worker_ref = None
        self._update_growth_status()

    def _on_deform_worker_finished(self):
        self.growth_cancel_button.setEnabled(False)
        self.growth_progress.setVisible(False)
        self._deform_worker_ref = None
        self._update_growth_status()

    def _on_growth_measured(self, result, folder, record):
        if result is CANCELLED or result is None:
            self.log("Growth measurement cancelled")
            return
        if record is not self._drift_record:
            return                         # another dataset was loaded meanwhile
        result.source = {"drift_record": str(record.path),
                         "wl_images": str(self._wl_stack.entries[0][0])}
        run_dir = Path(folder) / ANALYSIS_ROOT / (
            datetime.now().strftime(RUN_STAMP_FORMAT) + "_deformation")
        try:
            path = result.save(run_dir / deform_io.DEFORMATION_FILENAME)
        except Exception as exc:
            self.log(f"Could not save the growth deformation: {exc}")
            path = None
        self._deformation, self._deformation_path = result, path or run_dir
        self.log(f"Growth deformation measured and saved in {run_dir}")
        self._update_growth_status()
        self.drift_mode_box.setCurrentIndex(self.drift_mode_box.findData("growth"))
        self._refresh_drift_correction()

    def _read_drift_files(self, path):
        """Read a drift record, and the frame clock that places the movie in it."""
        self._drift_record = None
        self._frame_clock = None
        self._drift_frames = None
        self._drift_frames_key = None
        self._camera_map = None
        self._camera_map_problem = None
        self._wl_stack = None
        self._wl_check = None
        self._deformation = None
        self._deformation_path = None
        self.drift_edit.setText(str(path) if path else "")
        if not path:
            self._update_growth_status()
            return
        path = Path(path)
        try:
            record = drift_io.read_drift(path)
        except Exception as exc:
            self.log(f"Could not read the drift record {path.name}: {exc}")
            return
        self._drift_record = record
        first, last = record.span
        note = (f"Drift record {path.name}: {int(record.good.sum())} samples "
                f"over {last - first:.0f} s")
        orientation = record.meta.get("wl_orientation")
        if orientation:
            note += f", white-light frames oriented '{orientation}'"
        self.log(note)
        self._camera_map, self._camera_map_problem = drift_io.camera_map(record)
        if self._camera_map is not None:
            self.log(f"  read through the camera map from {self._camera_map.source}: "
                     f"1 white-light px = {self._camera_map.scale:.4f} fluorescence px, "
                     f"turned {self._camera_map.rotation_deg:+.2f} deg")
        else:
            self.log(f"  not applied: {self._camera_map_problem}")

        tif, times = drift_io.find_wl_images(path)
        if tif is not None:
            try:
                self._wl_stack = drift_io.read_wl_images(tif, times)
                n = len(self._wl_stack)
                spacing = (float(np.median(np.diff(self._wl_stack.t_epoch)))
                           if n > 1 else 0.0)
                self.log(f"{n} white-light snapshots beside it"
                         + (f", one every ~{spacing:.0f} s" if n > 1 else "")
                         + " - the Drift tab can check the record against them")
            except Exception as exc:
                self.log(f"Found {tif.name} but could not read it: {exc}")
        self._find_deformation(path)

        # The frame times are named after the stack and sit beside it; a record
        # browsed to from elsewhere may have its own beside it instead.
        folders = []
        for folder in (self._drift_search_folder(), path.parent):
            if folder is not None and folder not in folders:
                folders.append(folder)
        times_path = None
        for folder in folders:
            times_path = drift_io.find_frame_times(folder, self._stack_stem(), path)
            if times_path is not None:
                break
        if times_path is None:
            self.log("No <stack>_frame_times.csv beside the stack, so its frames "
                     "cannot be placed in the drift record - drift not corrected.")
            return
        try:
            clock = drift_io.read_frame_times(times_path)
        except Exception as exc:
            self.log(f"Could not read the frame times {times_path.name}: {exc}")
            return
        self._frame_clock = clock
        self.log(f"Frames timed by {times_path.name}: {clock.n_recorded} pages, "
                 f"one every {clock.slope * 1000:.2f} ms"
                 + ("" if clock.regular else " (irregular, so the raw timestamps are used)"))

    def _drift_raw_frame_count(self):
        """How many raw pages to time: the stack's, or as many as the table needs."""
        n = (int(self._raw_image.shape[0]) if self._raw_image is not None
             else self._frame_clock.n_recorded)
        frame_col = self._resolve_column("frame")
        source = self._df_source
        if (self._raw_image is None and source is not None and len(source)
                and frame_col in source.columns):
            last = int(np.nanmax(source[frame_col].to_numpy(dtype=float))) + self._frame_offset()
            n = max(n, (last + 1) * max(1, int(self._time_bin_applied)))
        return max(n, 1)

    def _drift_per_frame(self):
        """The drift of every frame of the stack, in white-light px, or None."""
        record, clock = self._drift_record, self._frame_clock
        if record is None or clock is None:
            return None
        key = (id(record), id(clock), self._drift_raw_frame_count(),
               max(1, int(self._time_bin_applied)),
               round(float(self.drift_smoothing_box.value()), 6))
        if key != self._drift_frames_key:
            self._drift_frames_key = key
            try:
                self._drift_frames = drift_io.drift_per_frame(
                    record, clock, key[2], bin_factor=key[3], sigma_s=key[4])
            except Exception as exc:
                self.log(f"Could not work out the drift of each frame: {exc}")
                self._drift_frames = None
        return self._drift_frames

    def _wl_to_nm(self, dx, dy):
        """A white-light displacement in the localizations' nm, or None.

        Through the camera map into fluorescence pixels, then the pixel size the
        localizations were computed with - so the drift is in the table's own
        nanometres, whichever pixel size that is.
        """
        if self._camera_map is None:
            return None
        pixel_nm = float(self.pixel_size_box.value())
        fx, fy = self._camera_map.to_fluorescence_px(dx, dy)
        return fx * pixel_nm, fy * pixel_nm

    def _wl_nm_per_px(self):
        """One white-light px in nm, for lengths (scatter, rms); 0 without a map."""
        if self._camera_map is None:
            return 0.0
        return self._camera_map.scale * float(self.pixel_size_box.value())

    def _drift_nm_per_frame(self):
        """(dx, dy) in nm for every frame, or None while it cannot be applied."""
        frames = self._drift_per_frame()
        if frames is None:
            return None
        return self._wl_to_nm(frames["dx"], frames["dy"])

    def _set_source_table(self, df):
        """Keep the table as recorded, taking out a correction it already carries.

        A table exported corrected - or corrected offline by recFL - says so in
        its drift columns. Correcting it again would move every localization
        twice as far, so the correction is put back first and the table is
        then treated like any other; the columns are kept to fall back on when
        no drift record is loaded.
        """
        self._embedded_drift = None
        x_col, y_col = self._resolve_column("x"), self._resolve_column("y")
        carried = all(c in df.columns for c in drift_io.DRIFT_COLUMNS)
        if carried and x_col in df.columns and y_col in df.columns:
            dx = np.nan_to_num(df[drift_io.DRIFT_X_COLUMN].to_numpy(dtype=float))
            dy = np.nan_to_num(df[drift_io.DRIFT_Y_COLUMN].to_numpy(dtype=float))
            raw = df.drop(columns=list(drift_io.DRIFT_COLUMNS))
            raw[x_col] = raw[x_col].to_numpy(dtype=float) + dx
            raw[y_col] = raw[y_col].to_numpy(dtype=float) + dy
            self._embedded_drift = (dx, dy)
            self.log("This table was already drift-corrected (it carries "
                     "drift_x/drift_y columns): that correction is taken back out "
                     "and the drift applied afresh with the settings here.")
            df = raw
        self._df_source = df

    def _localization_drift(self, table):
        """(dx, dy, rows off the end, origin) to subtract from `table`, or None.

        The white-light record when one is loaded and usable, else the drift
        the table itself came with - plus the RCC refinement on top, when one
        was estimated on that same correction. `origin` names the parts, joined
        by '+': "record", "table", "rcc".
        """
        if table is None:
            return None
        if not all(self._resolve_column(key) in table.columns for key in ("x", "y")):
            return None
        frame_col = self._resolve_column("frame")
        frames = None
        if frame_col in table.columns:
            frames = (np.nan_to_num(table[frame_col].to_numpy(dtype=float))
                      + self._frame_offset())
        dx = dy = None
        off_end = 0
        parts = []
        self._pending_local = None
        if self.drift_enable_box.isChecked():
            per_frame = self._drift_nm_per_frame()
            growth = self._growth_row_correction(table, frames)
            if growth is not None:
                dx, dy, off_end, self._pending_local = growth
                parts.append("growth")
            elif per_frame is not None and frames is not None:
                dx, dy, off_end = drift_io.lookup(frames, *per_frame)
                parts.append("record")
            elif self._embedded_drift is not None:
                dx, dy = self._embedded_drift
                parts.append("table")
        refinement = self._rcc_per_frame_nm()
        if refinement is not None and frames is not None:
            rdx, rdy, rcc_off = drift_io.lookup(frames, *refinement)
            dx = rdx if dx is None else dx + rdx
            dy = rdy if dy is None else dy + rdy
            off_end = max(off_end, rcc_off)
            parts.append("rcc")
        if dx is None:
            return None
        return dx, dy, off_end, "+".join(parts)

    # --- the growth deformation ---------------------------------------------
    def _growth_active(self):
        return (hasattr(self, "drift_mode_box") and self.drift_mode_box.currentData() == "growth"
                and self.drift_enable_box.isChecked())

    def _fluorescence_deformation(self):
        """The growth map on this camera, or None while it cannot be applied."""
        if not self._growth_active() or self._deformation is None or self._drift_record is None:
            return None
        cmap = self._camera_map
        if cmap is None or cmap.offset is None:
            return None
        record = self._drift_record
        sigma = float(self.drift_smoothing_box.value())
        reference = self._growth_reference_time()
        key = (id(self._deformation), id(record), cmap.rounded(), tuple(cmap.offset),
               self.wl_roi_x_box.value(), self.wl_roi_y_box.value(), round(sigma, 6),
               reference)
        if key != getattr(self, "_fluorescence_deformation_key", None):
            smooth = drift_io.smoother(record.t, record.dx, record.dy, sigma)
            self._fluorescence_deformation_cache = deform_io.FluorescenceDeformation(
                self._deformation, cmap.matrix, cmap.offset,
                sensor_origin=(self.wl_roi_x_box.value(), self.wl_roi_y_box.value()),
                translation=smooth, reference_time=reference)
            self._fluorescence_deformation_key = key
        return self._fluorescence_deformation_cache

    def _growth_reference_time(self):
        """The moment whose geometry everything is carried to: the first frame's,
        or None for the last snapshot's."""
        box = getattr(self, "growth_reference_box", None)
        if box is None or box.currentData() != "first":
            return None
        frames = self._drift_per_frame()
        if frames is None or not len(frames["t"]):
            return None
        return float(frames["t"][0])

    def _image_centre_px(self):
        name = self._image_layer_name
        if name and name in self.viewer.layers:
            shape = getattr(drift_io.unshifted(self.viewer.layers[name].data), "shape", None)
            if shape is not None and len(shape) >= 2:
                return ((shape[-1] - 1) / 2.0, (shape[-2] - 1) / 2.0)
        return (0.0, 0.0)

    def _growth_row_correction(self, table, frames):
        """(dx, dy, rows off the end, local px) carrying every localization to the
        final geometry, or None. dx, dy in nm are what is subtracted, as for the
        drift: each localization's own, since the correction depends on where it
        is. local: its position with the local stretch taken back out, which is
        what steps - and so diffusion - are measured on."""
        fd = self._fluorescence_deformation()
        per_frame = self._drift_per_frame()
        if fd is None or per_frame is None or frames is None:
            return None
        pixel = max(float(self.pixel_size_box.value()), 1e-9)
        x_col, y_col = self._resolve_column("x"), self._resolve_column("y")
        q = np.column_stack([table[x_col].to_numpy(dtype=float) / pixel,
                             table[y_col].to_numpy(dtype=float) / pixel])
        t_frame = np.asarray(per_frame["t"], dtype=float)
        index = np.clip(frames, 0, len(t_frame) - 1).astype(np.int64)
        off_end = int(np.count_nonzero((frames < 0) | (frames > len(t_frame) - 1)))
        final, local = fd.map_rows(q, t_frame[index], self._image_centre_px())
        dx = (q[:, 0] - final[:, 0]) * pixel
        dy = (q[:, 1] - final[:, 1]) * pixel
        local = pd.DataFrame({"x_local": local[:, 0], "y_local": local[:, 1]}, index=table.index)
        return dx, dy, off_end, local

    def _tracks_for_metrics(self):
        """The trajectories as diffusion, distances and the immobility test read
        them: in local coordinates when the growth correction is in force - in
        the final geometry every step would be stretched by the growth still to
        come, and D inflated by its square."""
        tracks = self.tracks
        if (tracks is not None and "x_local" in tracks.columns
                and "y_local" in tracks.columns):
            tracks = tracks.assign(x=tracks["x_local"], y=tracks["y_local"])
        return tracks

    def _stack_frame_count(self):
        """Frames in the stack the localizations index, however that is known."""
        per_frame = self._drift_nm_per_frame()
        if per_frame is not None:
            return len(per_frame[0])
        name = self._image_layer_name
        if name and name in self.viewer.layers:
            data = drift_io.unshifted(getattr(self.viewer.layers[name], "data", None))
            if getattr(data, "ndim", 0) >= 3:
                return int(data.shape[0])
        frame_col = self._resolve_column("frame")
        source = self._df_source
        if source is not None and len(source) and frame_col in source.columns:
            return int(np.nanmax(source[frame_col].to_numpy(dtype=float))) + self._frame_offset() + 1
        return 1

    def _wl_fingerprint(self):
        """The correction an RCC estimate refines, as plain values.

        RCC measures what the white-light correction left, so an estimate only
        holds while that correction is the same one: the record, its smoothing,
        the camera map and pixel size that bring it to nm, and which frame each
        localization is taken to be in. Plain values rather than objects, so it
        survives a session file.
        """
        wl = self.drift_enable_box.isChecked() and self._drift_nm_per_frame() is not None
        record = self._drift_record
        return [
            record.path.name if wl else None,
            round(float(self.drift_smoothing_box.value()), 6) if wl else None,
            ([self._camera_map.rounded(), round(float(self.pixel_size_box.value()), 6)]
             if wl else None),
            bool(self.drift_enable_box.isChecked() and not wl
                 and self._embedded_drift is not None),
            int(self._frame_offset()),
            int(max(1, int(self._time_bin_applied))),
            (["growth", str(self._deformation_path), round(self.wl_roi_x_box.value()),
              round(self.wl_roi_y_box.value())]
             if self._fluorescence_deformation() is not None else "drift"),
        ]

    def _rcc_is_current(self):
        return self._rcc is not None and list(self._rcc["fingerprint"]) == self._wl_fingerprint()

    def _rcc_per_frame_nm(self):
        """The RCC refinement of every frame in nm, or None when it does not apply."""
        if (self._rcc is None or not hasattr(self, "rcc_apply_box")
                or not self.rcc_apply_box.isChecked() or not self._rcc_is_current()):
            return None
        n = self._stack_frame_count()
        cached = self._rcc.get("per_frame")
        if cached is None or len(cached[0]) != n:
            cached = drift_io.rcc_per_frame(self._rcc["result"], n)
            self._rcc["per_frame"] = cached
        return cached

    def _total_drift_nm_per_frame(self):
        """White-light drift plus RCC refinement per frame, as applied; or None."""
        parts = []
        if self.drift_enable_box.isChecked():
            per_frame = self._drift_nm_per_frame()
            if per_frame is not None:
                parts.append(per_frame)
        refinement = self._rcc_per_frame_nm()
        if refinement is not None:
            parts.append(refinement)
        if not parts:
            return None
        n = min(len(part[0]) for part in parts)
        return (sum(part[0][:n] for part in parts), sum(part[1][:n] for part in parts))

    def _corrected_table(self, drift):
        """The source table with `drift` (from _localization_drift) taken out."""
        source = self._df_source
        # Local positions come with a growth correction, and go with any other.
        self._local_px = (getattr(self, "_pending_local", None)
                          if drift is not None and "growth" in drift[3].split("+") else None)
        if source is None or drift is None:
            self._applied_drift = None
            self._drift_rows_off_end = 0
            return source
        dx, dy, off_end, origin = drift
        x_col, y_col = self._resolve_column("x"), self._resolve_column("y")
        table = source.copy()
        table[x_col] = source[x_col].to_numpy(dtype=float) - dx
        table[y_col] = source[y_col].to_numpy(dtype=float) - dy
        # What was subtracted travels with the table, so an export says what
        # was done to it and can always be put back.
        table[drift_io.DRIFT_X_COLUMN] = dx
        table[drift_io.DRIFT_Y_COLUMN] = dy
        self._applied_drift = (dx, dy, origin)
        self._drift_rows_off_end = int(off_end)
        return table

    def _recorrect_table(self):
        """Bring the table in line with the drift settings. True if it moved.

        Idempotent: settings that land on the same drift - a smoothing box
        scrolled away and back, a debounce firing after a session restored the
        same values - leave the table, and the trajectories built on it, alone.
        """
        if self._df_source is None:
            return False
        drift = self._localization_drift(self._df_source)
        applied = self._applied_drift
        if drift is None and applied is None:
            return False
        if (drift is not None and applied is not None
                and np.array_equal(drift[0], applied[0])
                and np.array_equal(drift[1], applied[1])):
            return False
        self.df = self._corrected_table(drift)
        self._follow_xy_defaults()
        return True

    def _follow_xy_defaults(self):
        """Keep x/y filter bounds that were left wide open, wide open.

        Their defaults are the extent of the data, and correcting the drift
        moves the data: bounds left at the old extent would quietly drop the
        localizations that moved past it. Bounds someone narrowed are theirs,
        and stay where they were put.
        """
        for key in ("x", "y"):
            column = self._resolve_column(key)
            controls = self.filter_controls.get(column)
            if controls is None or self.df is None or column not in self.df.columns:
                continue
            lower_box, upper_box = controls
            old_lower, old_upper = self._default_bounds.get(column, (None, None))
            new_lower, new_upper = self._default_bounds_for(column)
            new_lower = bound_to_box_precision(new_lower, FILTER_BOUND_DECIMALS, False)
            new_upper = bound_to_box_precision(new_upper, FILTER_BOUND_DECIMALS, True)
            if old_lower is not None and abs(lower_box.value() - old_lower) <= 1e-6:
                lower_box.setValue(new_lower)
            if old_upper is not None and abs(upper_box.value() - old_upper) <= 1e-6:
                upper_box.setValue(new_upper)
            self._default_bounds[column] = (new_lower, new_upper)

    def _log_drift_applied(self):
        applied = self._applied_drift
        if applied is None:
            if self._df_source is not None and self._drift_nm_per_frame() is not None:
                self.log("Localizations shown as recorded, without drift correction")
            return
        dx, dy, origin = applied
        if not len(dx):
            return
        where = " plus ".join(DRIFT_ORIGIN_LABELS[part] for part in origin.split("+"))
        self.log(f"Drift taken out of {len(dx)} localizations, by up to "
                 f"{float(np.hypot(dx, dy).max()):.0f} nm, from {where}")
        if self._drift_rows_off_end:
            self.log(f"  {self._drift_rows_off_end} of them are on frames outside "
                     "the stack and took the drift of its nearest end - check the "
                     "frame shift.")

    def _refresh_drift_correction(self):
        """Re-apply the drift after anything it depends on has changed."""
        if hasattr(self, "_drift_timer"):
            self._drift_timer.stop()
        if self._recorrect_table():
            self._invalidate_tracks(reason="drift correction changed")
            self.apply_filters()
            self._log_drift_applied()
        self._show_drift_state()

    def _show_drift_state(self):
        """Bring the image, the reports and the plots in line with the drift."""
        if not hasattr(self, "rcc_status"):
            return
        self._apply_drift_to_image()
        self._update_drift_status()
        self._update_wl_check_status()
        self._update_rcc_status()
        self._draw_drift_plot()
        self._draw_rcc_plot()
        self._update_status_header()

    def _image_drift_shifts(self, n_frames):
        """(row, column) px each displayed frame moves by, or None to show it as recorded.

        The whole correction the localizations get - white-light and RCC -
        so that they sit on their spots whichever parts are in force.
        """
        if not self.drift_shift_image_box.isChecked():
            return None
        per_frame = self._total_drift_nm_per_frame()
        if per_frame is None:
            return None
        pixel_nm = max(self.pixel_size_box.value(), 1e-9)
        dx, dy = per_frame
        index = np.clip(np.arange(int(n_frames)), 0, len(dx) - 1)
        return np.column_stack([-dy[index] / pixel_nm, -dx[index] / pixel_nm])

    def _growth_warped_stack(self, base):
        """The stack drawn in the final geometry, frame by frame, or None."""
        fd = self._fluorescence_deformation()
        frames = self._drift_per_frame()
        if fd is None or frames is None:
            return None
        t = np.asarray(frames["t"], dtype=float)
        n = int(base.shape[0])
        times = t[np.clip(np.arange(n), 0, len(t) - 1)]
        h, w = (int(v) for v in base.shape[-2:])
        corners = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])
        sample = times[np.unique(np.linspace(0, n - 1, min(n, 40)).astype(int))]
        placed = np.vstack([fd.to_final(corners, np.full(4, tt)) for tt in sample])
        x0, y0 = np.floor(placed.min(axis=0) + 0.5).astype(int)
        x1, y1 = np.ceil(placed.max(axis=0) + 0.5).astype(int)
        key = (id(fd), n, int(x0), int(y0), int(x1), int(y1))
        return drift_io.WarpedStack(
            base,
            inverse=lambda i, xy: fd.from_final(xy, np.full(len(xy), times[i])),
            forward=lambda i, xy: fd.to_final(xy, np.full(len(xy), times[i])),
            origin_yx=(int(y0), int(x0)), canvas=(int(y1 - y0), int(x1 - x0)), key=key)

    def _displayed_warp(self):
        """The warped stack on screen, when the growth correction draws it."""
        name = self._image_layer_name
        if not name or name not in self.viewer.layers:
            return None
        data = getattr(self.viewer.layers[name], "data", None)
        return data if isinstance(data, drift_io.WarpedStack) else None

    def _displayed_image_shifts(self):
        """The shifts the stack is on screen with right now, or None."""
        name = self._image_layer_name
        if not name or name not in self.viewer.layers:
            return None
        data = getattr(self.viewer.layers[name], "data", None)
        return data.shifts_yx if isinstance(data, drift_io.ShiftedStack) else None

    def _apply_drift_to_image(self):
        """Show the loaded stack with the drift taken out, or as recorded.

        Only the stack this plugin loaded: another image dragged into the
        viewer has no frame clock, and nothing says its frames are these.
        """
        name = self._image_layer_name
        if name and name in self.viewer.layers:
            layer = self.viewer.layers[name]
            data = getattr(layer, "data", None)
            base = drift_io.unshifted(data)
            warped = (self._growth_warped_stack(base)
                      if getattr(base, "ndim", 0) >= 3 and self.drift_shift_image_box.isChecked()
                      else None)
            shifts = (self._image_drift_shifts(base.shape[0])
                      if getattr(base, "ndim", 0) >= 3 and warped is None else None)
            if warped is not None:
                if not (isinstance(data, drift_io.WarpedStack) and data.key == warped.key):
                    layer.data = warped
                self._place_image_canvas(layer, warped.origin_yx)
            elif shifts is None:
                if data is not base:
                    layer.data = base
                self._place_image_canvas(layer, (0, 0))
            else:
                shifted = drift_io.ShiftedStack(base, shifts)
                if (isinstance(data, drift_io.ShiftedStack) and data.shape == shifted.shape
                        and data.origin_yx == shifted.origin_yx):
                    if not np.array_equal(data.shifts_yx, shifts):
                        data.shifts_yx = shifts
                        refresh = getattr(layer, "refresh", None)
                        if callable(refresh):
                            refresh()
                else:
                    # A different canvas is a different shape, which napari
                    # only learns from new data.
                    layer.data = shifted
                self._place_image_canvas(layer, shifted.origin_yx)
        self._update_loc2d_candidate_overlay()
        # The white light moves with the fluorescence image, whichever way.
        self._update_wl_overlay()

    def _place_image_canvas(self, layer, origin_yx):
        """Put the stack's first pixel where its canvas starts, in the viewer's nm."""
        pixel_nm = self.pixel_size_box.value()
        current = np.array(np.ravel(getattr(layer, "translate", ())), dtype=float)
        ndim = max(len(current), len(getattr(layer.data, "shape", ())), 2)
        translate = np.zeros(ndim) if len(current) != ndim else current.copy()
        translate[-2:] = (origin_yx[0] * pixel_nm, origin_yx[1] * pixel_nm)
        if len(current) != ndim or not np.allclose(current, translate):
            layer.translate = tuple(translate)

    @staticmethod
    def _canvas_origin(layer):
        """Where a layer's data grid starts, in the first frame's camera pixels."""
        data = getattr(layer, "data", None)
        return data.origin_yx if isinstance(data, drift_io.ShiftedStack) else (0, 0)

    def _update_drift_status(self):
        record, clock = self._drift_record, self._frame_clock
        lines = []
        if record is None:
            if self._embedded_drift is not None:
                lines.append("The table came drift-corrected (drift_x/drift_y "
                             "columns), and no drift record is loaded to redo it.")
            else:
                lines.append("No drift record loaded. The acquisition writes "
                             "<name>_xy_drift.csv beside the stack, and it is picked "
                             "up on loading; or browse to one.")
        else:
            # The file names are in the field above and in the log; repeating
            # two sixty-character names here only buries what was found.
            lines.append(f"{int(record.good.sum())} drift samples, one every "
                         f"{record.sample_interval_s * 1000:.0f} ms.")
            frames = self._drift_per_frame()
            if clock is None:
                lines.append("No frame times, so the frames cannot be placed in it.")
            else:
                timed = f"Frames timed by their _frame_times.csv ({clock.n_recorded} pages)"
                if frames is not None and frames["n_extrapolated"]:
                    timed += (f"; the last {frames['n_extrapolated']} frames are past "
                              "its end and were timed by extending its clock")
                lines.append(timed + ".")
                if frames is not None and frames["outside"].any():
                    lines.append(f"{int(frames['outside'].sum())} frames fall outside the "
                                 "drift record and were given the drift at its nearest end.")
            if self._camera_map is None:
                lines.append(f"Not applied: {self._camera_map_problem}.")
            elif frames is not None:
                dx, dy = self._wl_to_nm(frames["dx"], frames["dy"])
                lines.append(
                    f"Over the movie: {float(np.hypot(dx[-1], dy[-1])):.0f} nm "
                    f"(x {dx[-1]:+.0f}, y {dy[-1]:+.0f} nm). A single sample "
                    f"scatters by ~{record.sample_noise_px() * self._wl_nm_per_px():.1f} "
                    "nm before smoothing.")
        self._update_camera_map_label()
        if self._drift_rows_off_end:
            lines.append(f"{self._drift_rows_off_end} localizations are on frames outside "
                         "the stack - check the frame shift.")
        if record is not None and not self.drift_enable_box.isChecked():
            lines.append("Not applied: the data is shown as recorded.")
        self.drift_status.setText("\n".join(lines))

    def _update_camera_map_label(self):
        """Say which camera map the record is read through, and what it amounts to."""
        if self._drift_record is None:
            self.drift_map_label.setText("No drift record loaded")
        elif self._camera_map is None:
            self.drift_map_label.setText(f"None applies: {self._camera_map_problem}.")
        else:
            m = self._camera_map
            self.drift_map_label.setText(
                f"1 white-light px = {m.scale:.4f} fluorescence px, turned "
                f"{m.rotation_deg:+.2f}° - {self._wl_nm_per_px():.2f} nm at "
                f"{self.pixel_size_box.value():.2f} nm/px. From {m.source}.")

    def _draw_drift_plot(self, figure=None, opts=None):
        """The drift against time: the raw samples, and what is applied."""
        export = figure is not None
        opts = opts or {}
        figure = figure if export else self.drift_figure
        figure.clear()
        record = self._drift_record
        if record is None:
            figure.patch.set_facecolor(PANEL_BG)
            if not export:
                self.drift_canvas.draw_idle()
            return
        # On the fluorescence camera's axes, in nm, whenever a camera map says how;
        # the map mixes the two white-light axes, so x and y convert together.
        if self._camera_map is not None:
            convert, unit = self._wl_to_nm, "nm"
        else:
            def convert(x, y):
                return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            unit = "white-light px"
        good = record.good
        t, dx, dy = record.t[good], record.dx[good], record.dy[good]
        frames = self._drift_per_frame()
        ax = figure.add_subplot(111)
        if frames is not None:
            t0, t1 = float(frames["t"][0]), float(frames["t"][-1])
            margin = 0.05 * max(t1 - t0, 1.0)
            span = (t >= t0 - margin) & (t <= t1 + margin)
            ox, oy = frames["origin"]
            # Dashed: what is actually applied, once RCC adds to the record.
            refinement = (self._rcc_per_frame_nm()
                          if self._camera_map is not None and self.drift_enable_box.isChecked()
                          else None)
            samples = convert(dx[span] - ox, dy[span] - oy)
            applied = convert(frames["dx"], frames["dy"])
            for axis, color, label in ((0, ACCENT, "x"), (1, LAVENDER, "y")):
                per_frame = applied[axis]
                ax.plot(t[span] - t0, samples[axis], ".",
                        color=color, alpha=0.35, markersize=2)
                ax.plot(frames["t"] - t0, per_frame, color=color,
                        linewidth=1.3, label=label)
                if refinement is not None:
                    n = min(len(per_frame), len(refinement[axis]))
                    ax.plot(frames["t"][:n] - t0, per_frame[:n] + refinement[axis][:n],
                            color=color, linewidth=1.0, linestyle="--",
                            label=f"{label} + RCC")
            origin = (ox, oy)
            ax.set_xlabel("Time since the first frame (s)")
        else:
            t0 = float(t[0])
            px, py = convert(dx, dy)
            ax.plot(t - t0, px, color=ACCENT, linewidth=1.0, label="x")
            ax.plot(t - t0, py, color=LAVENDER, linewidth=1.0, label="y")
            origin = (0.0, 0.0)
            ax.set_xlabel("Time since the record started (s)")
        check = self._wl_check
        if check is not None:
            # The snapshots, registered independently, on the record's own
            # scale: measured relative to the first snapshot, which the record
            # puts wherever it stood then.
            f = drift_io.smoother(record.t, record.dx, record.dy,
                                  float(self.drift_smoothing_box.value()))
            sx, sy = f(np.array([check["t"][0]]))
            marks = convert(check["measured_dx"] + sx[0] - origin[0],
                            check["measured_dy"] + sy[0] - origin[1])
            for axis, color, label in ((0, ACCENT, "x snapshots"), (1, LAVENDER, "y snapshots")):
                ax.plot(check["t"] - t0, marks[axis], "o",
                        markerfacecolor="none", markeredgecolor=color,
                        markeredgewidth=1.2, markersize=6, label=label)
        ax.axhline(0.0, color=INK_DIM, linewidth=0.6)
        ax.set_ylabel(f"Drift ({unit})")
        legend = ax.legend(fontsize=plot_font(-2), loc="upper left", ncol=2,
                           facecolor=PLOT_BG, edgecolor=PANEL_LINE, labelcolor=INK)
        legend.get_frame().set_alpha(0.85)
        style_axes(figure, ax, title="Sample drift" if opts.get("title", True) else None)
        if not opts.get("grid", True):
            ax.grid(False)
        figure.tight_layout()
        if not export:
            self.drift_canvas.draw_idle()

    def _drift_metadata(self):
        """What was done about drift, for metadata.json and sessions."""
        record, clock = self._drift_record, self._frame_clock
        frames = self._drift_per_frame()
        applied = self._applied_drift
        growth = None
        if self._deformation is not None:
            along, across = self._deformation.strain_along_axis()
            growth = {"deformation_record": str(self._deformation_path),
                      "applied": self._fluorescence_deformation() is not None,
                      "model": self._deformation.model,
                      "root_axis_deg": float(np.degrees(self._deformation.theta)),
                      "stretch_along_axis": float(along[0]),
                      "stretch_across_axis": float(across[0]),
                      "geometry": ("the first frame's" if self._growth_reference_time() is not None
                                   else "the last white-light snapshot's"),
                      "diffusion_measured_in": "local coordinates (the stretch taken back out)"}
        section = {
            "enabled": self.drift_enable_box.isChecked(),
            "mode": self.drift_mode_box.currentData(),
            "growth": growth,
            "shift_image": self.drift_shift_image_box.isChecked(),
            "smoothing_s": self.drift_smoothing_box.value(),
            # the record reached nm through this matrix and the pixel size
            "camera_map": ({"fluorescence_px_per_white_light_px": self._camera_map.rounded(),
                            "source": self._camera_map.source,
                            "white_light_nm_per_px": round(self._wl_nm_per_px(), 4)}
                           if self._camera_map is not None else None),
            "camera_map_problem": self._camera_map_problem,
            "drift_record": str(record.path) if record is not None else None,
            "frame_times": str(clock.path) if clock is not None else None,
            "applied_to_localizations": applied is not None,
            "drift_source": (" plus ".join(DRIFT_ORIGIN_LABELS[part]
                                           for part in applied[2].split("+"))
                             if applied is not None else None),
            "wl_orientation": record.meta.get("wl_orientation") if record is not None else None,
            "step_threshold_px": self.drift_step_box.value(),
            "rcc": self._rcc_metadata(),
            "snapshot_check": self._wl_check_summary(),
        }
        if frames is not None:
            section["frames_timed_by_extrapolation"] = int(frames["n_extrapolated"])
            section["frames_outside_drift_record"] = int(frames["outside"].sum())
            if self._camera_map is not None:
                dx, dy = self._wl_to_nm(frames["dx"][-1], frames["dy"][-1])
                section["drift_over_movie_nm"] = {"x": float(dx), "y": float(dy)}
        if applied is not None:
            section["localizations_on_frames_outside_stack"] = int(self._drift_rows_off_end)
        return section

    def _drift_table(self):
        """The drift applied to each frame, and its two parts, for the export."""
        applied = self._applied_drift
        if applied is None or not ({"record", "rcc", "growth"} & set(applied[2].split("+"))):
            return None
        if "growth" in applied[2].split("+"):
            return self._growth_drift_table()
        total = self._total_drift_nm_per_frame()
        if total is None:
            return None
        n = len(total[0])
        table = {"frame": np.arange(n)}
        frames = self._drift_per_frame()
        if frames is not None and len(frames["t"]) >= n:
            table["t_s"] = frames["t"][:n] - frames["t"][0]
        table[drift_io.DRIFT_X_COLUMN] = total[0]
        table[drift_io.DRIFT_Y_COLUMN] = total[1]
        wl = self._drift_nm_per_frame() if self.drift_enable_box.isChecked() else None
        refinement = self._rcc_per_frame_nm()
        for name, part in (("white_light", wl), ("rcc", refinement)):
            if part is not None:
                table[f"{name}_drift_x [nm]"] = part[0][:n]
                table[f"{name}_drift_y [nm]"] = part[1][:n]
        if frames is not None and len(frames["outside"]) >= n:
            table["outside_drift_record"] = frames["outside"][:n].astype(int)
        return pd.DataFrame(table)

    def _growth_drift_table(self):
        """Under the growth correction each localization has a correction of its own;
        per frame, what it is at the image centre - the table says so."""
        fd, frames = self._fluorescence_deformation(), self._drift_per_frame()
        if fd is None or frames is None:
            return None
        pixel = float(self.pixel_size_box.value())
        t = np.asarray(frames["t"], dtype=float)
        centre = np.tile(self._image_centre_px(), (len(t), 1))
        final = fd.to_final(centre, t)
        J = fd.jacobian(centre, t)
        table = {"frame": np.arange(len(t)), "t_s": t - t[0],
                 "correction_at_image_centre_x [nm]": (centre[:, 0] - final[:, 0]) * pixel,
                 "correction_at_image_centre_y [nm]": (centre[:, 1] - final[:, 1]) * pixel,
                 # how much a length at the centre is stretched on its way to the final geometry
                 "stretch_at_image_centre": np.sqrt(np.abs(np.linalg.det(J)))}
        refinement = self._rcc_per_frame_nm()
        if refinement is not None:
            n = min(len(t), len(refinement[0]))
            table["rcc_drift_x [nm]"] = np.pad(refinement[0][:n], (0, len(t) - n))
            table["rcc_drift_y [nm]"] = np.pad(refinement[1][:n], (0, len(t) - n))
        return pd.DataFrame(table)

    # --- checking the record ------------------------------------------------
    def _update_wl_check_status(self):
        """What the record says about itself, and what the snapshots say about it."""
        if not hasattr(self, "wl_check_status"):
            return
        record = self._drift_record
        has_snapshots = self._wl_stack is not None
        self.wl_check_button.setEnabled(record is not None and has_snapshots
                                        and self._wl_check_worker_ref is None)
        self.wl_show_button.setEnabled(record is not None and has_snapshots)
        if record is None:
            self.wl_check_status.setText("No drift record loaded")
            return
        wl_nm = self._wl_nm_per_px()

        def length(px):
            return f"{px * wl_nm:.0f} nm" if wl_nm > 0 else f"{px:.2f} px"

        lines = []
        quality = record.quality[record.good] if record.quality is not None else None
        if quality is not None and quality.size:
            lines.append(f"Correlation with the reference: {np.nanmin(quality):.2f} to "
                         f"{np.nanmax(quality):.2f} (1 = identical).")
        if record.ref_id is not None and record.ref_id.size:
            links = int(np.nanmax(record.ref_id))
            lines.append("Never re-referenced." if links == 0 else
                         f"Re-referenced {links} time(s) as the sample changed - each "
                         "link adds about one measurement's error.")
        lines.append(f"A single sample scatters by ~{length(record.sample_noise_px())}.")

        steps = drift_io.abrupt_steps(record, self.drift_step_box.value())
        frames = self._drift_per_frame()
        zero = float(frames["t"][0]) if frames is not None else float(record.t[record.good][0])
        if len(steps):
            shown = ", ".join(f"{t - zero:.1f} s ({jx:+.2f}, {jy:+.2f})"
                              for t, jx, jy in steps[:6])
            more = f" and {len(steps) - 6} more" if len(steps) > 6 else ""
            lines.append(f"{len(steps)} jump(s) between consecutive samples, at time "
                         f"(dx, dy) in white-light px: {shown}{more}.")
        else:
            lines.append("No jumps between consecutive samples.")

        if not has_snapshots:
            lines.append("No white-light snapshots beside the record to check it against.")
        else:
            n = len(self._wl_stack)
            lines.append(f"{n} white-light snapshots to check it against.")
            summary = self._wl_check_summary()
            if summary is not None:
                off = np.hypot(summary["differences_x_px"], summary["differences_y_px"])
                worst = int(np.argmax(off))
                lines.append(
                    f"The snapshots, registered independently, sit within "
                    f"{length(float(np.sqrt(np.mean(off ** 2))))} rms of the record "
                    f"(worst {length(float(off[worst]))}, snapshot {worst} at "
                    f"{summary['t_s'][worst]:.0f} s).")
        self.wl_check_status.setText("\n".join(lines))

    def _wl_check_summary(self):
        """The snapshot check against the record as currently smoothed, or None."""
        check, record = self._wl_check, self._drift_record
        if check is None or record is None:
            return None
        f = drift_io.smoother(record.t, record.dx, record.dy,
                              float(self.drift_smoothing_box.value()))
        rx, ry = f(check["t"])
        rx, ry = rx - rx[0], ry - ry[0]
        frames = self._drift_per_frame()
        zero = float(frames["t"][0]) if frames is not None else float(check["t"][0])
        return {
            "t_s": [float(v) for v in check["t"] - zero],
            "measured_x_px": [float(v) for v in check["measured_dx"]],
            "measured_y_px": [float(v) for v in check["measured_dy"]],
            "differences_x_px": [float(v) for v in check["measured_dx"] - rx],
            "differences_y_px": [float(v) for v in check["measured_dy"] - ry],
            "quality": [float(v) for v in check["quality"]],
            "region_top_left_h_w": [int(v) for v in check["roi"]],
        }

    def check_wl_snapshots(self):
        record, stack = self._drift_record, self._wl_stack
        if record is None or stack is None:
            self.log("Load data with a drift record and white-light snapshots first")
            return
        self.log(f"Registering {len(stack)} white-light snapshots against the first...")
        self.wl_check_button.setEnabled(False)
        self.wl_check_progress.setVisible(True)
        self.wl_check_progress.setValue(0)
        self._arm_cancel(self._wl_check_cancel, self.wl_check_cancel_button)
        worker = _wl_check_worker(stack, record, float(self.drift_smoothing_box.value()),
                                  self._wl_check_cancel)
        worker.yielded.connect(lambda frac: self.wl_check_progress.setValue(int(frac * 100)))
        worker.returned.connect(lambda result, r=record: self._on_wl_check_finished(result, r))
        worker.errored.connect(lambda exc: self.log(f"The snapshot check failed: {exc}"))
        worker.finished.connect(self._on_wl_check_worker_finished)
        self._wl_check_worker_ref = worker
        worker.start()

    def _on_wl_check_worker_finished(self):
        self.wl_check_cancel_button.setEnabled(False)
        self.wl_check_progress.setVisible(False)
        self._wl_check_worker_ref = None
        self._update_wl_check_status()

    def _on_wl_check_finished(self, result, record):
        if result is CANCELLED:
            self.log("Snapshot check cancelled")
            return
        if record is not self._drift_record:
            return          # another record was loaded while it ran
        self._wl_check = result
        summary = self._wl_check_summary()
        off = np.hypot(summary["differences_x_px"], summary["differences_y_px"])
        wl_nm = self._wl_nm_per_px()
        unit = (f"{float(off.max()) * wl_nm:.1f} nm" if wl_nm > 0
                else f"{float(off.max()):.2f} white-light px")
        self.log(f"Snapshot check: the record agrees with every snapshot to within {unit} "
                 "- the snapshots are drawn as circles on the drift plot")
        self._update_wl_check_status()
        self._draw_drift_plot()

    def show_wl_snapshots(self):
        """The snapshots in a viewer of their own: as recorded, and drift removed."""
        record, stack = self._drift_record, self._wl_stack
        if record is None or stack is None:
            self.log("Load data with a drift record and white-light snapshots first")
            return
        f = drift_io.smoother(record.t, record.dx, record.dy,
                              float(self.drift_smoothing_box.value()))
        rx, ry = f(stack.t_epoch)
        shifts = np.column_stack([-(ry - ry[0]), -(rx - rx[0])])
        # The white-light camera's own frame, so only its pixel size is shown.
        wl_nm = self._wl_nm_per_px()
        scale = (1.0, wl_nm, wl_nm) if wl_nm > 0 else (1.0, 1.0, 1.0)
        first = np.asarray(stack[0])
        limits = tuple(float(v) for v in np.percentile(first, (0.5, 99.8)))
        if limits[1] <= limits[0]:
            limits = (limits[0], limits[0] + 1.0)

        viewer = self._new_wl_viewer()
        viewer.add_image(drift_io.ShiftedStack(stack, np.zeros_like(shifts)),
                         name="snapshots as recorded", colormap="gray",
                         contrast_limits=limits, scale=scale, visible=False)
        fixed = drift_io.ShiftedStack(stack, shifts)
        viewer.add_image(fixed, name="snapshots, drift removed", colormap="gray",
                         contrast_limits=limits, scale=scale,
                         translate=(0.0, fixed.origin_yx[0] * scale[1],
                                    fixed.origin_yx[1] * scale[2]))
        roi = record.meta.get("roi_top_left_h_w")
        if roi is not None and len(roi) == 4:
            top, left, h, w = (float(v) for v in roi)
            rect = np.array([[top, left], [top, left + w], [top + h, left + w], [top + h, left]])
            viewer.add_shapes([rect], shape_type="rectangle", name="region the tracker followed",
                              edge_color="yellow", face_color="transparent", edge_width=4,
                              scale=scale[1:])
        rec = self._deformation
        if rec is not None and len(rec.t) == len(stack):
            try:
                self._add_growth_layers(viewer, rec, stack, shifts, scale[1], limits)
            except Exception as exc:
                self.log(f"The growth could not be drawn over the snapshots: {exc}")
                rec = None
        self.log(f"Opened the {len(stack)} snapshots in a viewer of their own: step "
                 "through them and toggle the two layers - with the drift removed "
                 "the sample should stand still."
                 + (" The growth is there too, in the first snapshot's geometry: the "
                    "arrows of the growth since then, the model as a grid on the tissue, "
                    "'growth cancelled, by measurement stage' (magenta) to lay on 'first "
                    "snapshot' (green) - white where they agree; the 'stage' slider steps "
                    "through the measurement, and the patches it used are a layer."
                    if rec is not None else ""))

    def _add_growth_layers(self, viewer, rec, stack, shifts, px_nm, limits):
        """The growth over the snapshots, all in the first snapshot's geometry - the
        frame the drift-removed snapshots are drawn in."""
        x0, x1, y0, y1 = growth_view.first_region(rec)
        start = (y0 * px_nm, x0 * px_nm)
        scale2 = (px_nm, px_nm)
        viewer.add_image(np.asarray(stack[0], dtype=np.float32)[max(y0, 0):y1, max(x0, 0):x1],
                         name="first snapshot", colormap="green", blending="additive",
                         contrast_limits=limits, scale=scale2,
                         translate=(max(y0, 0) * px_nm, max(x0, 0) * px_nm), visible=False)
        stages = growth_view.StageStack(stack, rec)
        n_stages = stages.shape[0]
        viewer.add_image(stages, name="growth cancelled, by measurement stage",
                         colormap="magenta", blending="additive", contrast_limits=limits,
                         scale=(1.0, 1.0) + scale2, translate=(0.0, 0.0) + start, visible=False)
        maps, origin, spacing = growth_view.stretch_maps(rec)
        top = float(np.nanmax(np.abs(maps))) or 1.0
        viewer.add_image(maps, name="stretch along the root since the first snapshot (%)",
                         colormap="plasma", blending="translucent", opacity=0.5,
                         contrast_limits=(0.0, max(top, 1e-6)),
                         scale=(1.0, spacing * px_nm, spacing * px_nm),
                         translate=(0.0, (origin[0] - spacing / 2) * px_nm,
                                    (origin[1] - spacing / 2) * px_nm),
                         visible=False)
        lines = growth_view.deformation_grid(rec, shifts)
        if lines:
            paths = [np.column_stack([np.full(len(pts), k), pts]) for k, _kind, pts in lines]
            colors = [AXIS_ALONG if kind == "along" else AXIS_ACROSS for _k, kind, _p in lines]
            viewer.add_shapes(paths, shape_type="path", name="deformation model (grid on the tissue)",
                              edge_color=colors, edge_width=2.5, face_color="transparent",
                              scale=(1.0,) + scale2)
        size = max(x1 - x0, y1 - y0)
        vectors, magnitude = growth_view.growth_arrows(rec, shifts)
        if len(vectors):
            gain = max(1.0, round(0.1 * size / max(float(magnitude.max()), 1e-9)))
            viewer.add_vectors(
                vectors, name=f"growth since the first snapshot (arrows x{gain:g})", length=gain,
                edge_width=4, vector_style="arrow", scale=(1.0,) + scale2,
                features={"growth_um": magnitude * px_nm / 1000.0},
                edge_color="growth_um", edge_colormap="plasma")
        rate, speed = growth_view.growth_rate_arrows(rec, shifts)
        if len(rate):
            gain = max(1.0, round(0.06 * size / max(float(np.percentile(speed, 95)), 1e-9)))
            viewer.add_vectors(
                rate, name=f"growth rate, per minute (arrows x{gain:g})", length=gain,
                edge_width=3, vector_style="arrow", scale=(1.0,) + scale2, visible=False,
                features={"um_per_min": speed * px_nm / 1000.0},
                edge_color="um_per_min", edge_colormap="viridis")
        step_px = int(rec.stats.get("step_px", 128))
        tiles, along = growth_view.patch_outlines(rec, size_px=step_px)
        if tiles:
            # One tile a grid step wide per patch: the patches themselves overlap
            # three times over and drawn whole are a mesh nobody can read.
            viewer.add_shapes(
                tiles, shape_type="polygon", name="measurement patches",
                edge_width=1, edge_color="white", opacity=0.6, scale=scale2, visible=False,
                features={"cross_wall_share": np.nan_to_num(along)},
                face_color="cross_wall_share", face_colormap="viridis")
            whole, _share = growth_view.patch_outlines(rec)
            centres = np.array([t.mean(axis=0) for t in tiles])
            middle = int(np.argmin(np.hypot(*(centres - centres.mean(axis=0)).T)))
            viewer.add_shapes(
                [whole[middle]], shape_type="polygon",
                name=f"one patch ({int(rec.stats.get('patch_px', 384))} px)", edge_width=4,
                edge_color="yellow", face_color="transparent", scale=scale2, visible=False)
        # one gain for both, set by the first pass - what the rough maps left - so
        # the arrows visibly shrink stage after stage as the measurement converges
        measured, m_size = growth_view.stage_arrows(rec, "measured")
        first = m_size[measured[:, 0, 0] == 1] if len(measured) else m_size
        gain = max(1.0, round(0.06 * size / max(float(np.median(first)) if first.size else 1.0, 1e-9)))
        for which, name, color in (("measured", "still off at each patch, by stage", "yellow"),
                                   ("model", "the model's correction, by stage", "cyan")):
            v = measured if which == "measured" else growth_view.stage_arrows(rec, which)[0]
            if len(v):
                viewer.add_vectors(v, name=f"{name} (x{gain:g})", length=gain, edge_width=3,
                                   vector_style="arrow", edge_color=color, visible=False,
                                   scale=(1.0, 1.0) + scale2)
        try:
            viewer.dims.axis_labels = ("stage", "snapshot", "y", "x")
            viewer.dims.set_current_step(0, n_stages - 1)
        except Exception:
            pass
        self._growth_metrics_dock(viewer, rec, px_nm)

    def _growth_metrics_dock(self, viewer, rec, px_nm):
        """The growth's numbers as plots, and how it was measured, beside the snapshots."""
        window = getattr(viewer, "window", None)
        if window is None or not hasattr(window, "add_dock_widget"):
            return None
        metrics = growth_view.growth_metrics(rec, px_nm)
        figure = Figure(figsize=(5.2, 7.6))
        canvas = FigureCanvas(figure)
        markers = self._draw_growth_metrics(figure, metrics)
        panel = QWidget()
        box = QVBoxLayout(panel)
        box.setContentsMargins(4, 4, 4, 4)
        box.addWidget(canvas)
        tools = QHBoxLayout()
        tools.addStretch(1)
        tools.addWidget(self._png_button(
            lambda fig, opts: self._draw_growth_metrics(fig, metrics, opts), lambda: "growth_metrics"))
        box.addLayout(tools)
        how = QLabel(growth_view.describe_measurement(rec, px_nm))
        how.setWordWrap(True)
        how.setProperty("role", "note")
        box.addWidget(how)

        def follow(_event=None):
            step = viewer.dims.current_step
            k = int(step[-3]) if len(step) >= 3 else 0
            k = min(max(k, 0), len(metrics["minutes"]) - 1)
            for line in markers:
                line.set_xdata([metrics["minutes"][k]] * 2)
            canvas.draw_idle()

        viewer.dims.events.current_step.connect(follow)
        follow()
        window.add_dock_widget(panel, name="growth", area="right")
        return panel

    def _draw_growth_metrics(self, figure, m, opts=None):
        """Six small plots of the growth; returns the time markers to move."""
        opts = opts or {}
        figure.clear()
        axes = figure.subplots(3, 2)
        minutes = m["minutes"]
        markers = []
        ax = axes[0, 0]
        ax.plot(minutes, m["stretch_along"], "o-", ms=2.5, color=AXIS_ALONG, label="along the root")
        ax.plot(minutes, m["stretch_across"], "s--", ms=2.5, color=AXIS_ACROSS, label="across")
        ax.set_ylabel("stretch since the start (%)")
        ax.legend(fontsize=plot_font(-3), facecolor=PLOT_BG, edgecolor=PANEL_LINE, labelcolor=INK)
        ax = axes[0, 1]
        ax.plot(minutes, m["rate_along"], "-", color=AXIS_ALONG)
        ax.set_ylabel("elongation rate (%/min)")
        ax = axes[1, 0]
        ax.plot(minutes, m["centre_um"][:, 0], "-", color=ACCENT, label="x")
        ax.plot(minutes, m["centre_um"][:, 1], "-", color=AMBER, label="y")
        ax.set_ylabel("field centre moved (µm)")
        ax.legend(fontsize=plot_font(-3), facecolor=PLOT_BG, edgecolor=PANEL_LINE, labelcolor=INK)
        for a in (axes[0, 0], axes[0, 1], axes[1, 0]):
            a.set_xlabel("minutes from the first snapshot")
            markers.append(a.axvline(minutes[0], color=INK_DIM, linewidth=0.8))
        ax = axes[1, 1]
        ax.plot(m["profile_um"], m["profile_stretch"], "-", color=AXIS_ALONG)
        ax.set_xlabel(f"along the root axis, µm (+ toward {m['axis_deg']:.0f}°)")
        ax.set_ylabel("stretch, first to last (%)")
        ax = axes[2, 0]
        if len(m["passes_nm"]):
            ax.semilogy(np.arange(1, len(m["passes_nm"]) + 1), m["passes_nm"], "o-", color=ACCENT)
        ax.set_xlabel("refinement pass")
        ax.set_ylabel("rms correction (nm)")
        ax = axes[2, 1]
        if len(m["lags"]):
            ax.plot(m["lags"], m["lag_residual_nm"], "o", ms=3, color=ACCENT)
        ax.set_xlabel("snapshots apart")
        ax.set_ylabel("residual per patch (nm)")
        style_axes(figure, axes, title=None)
        if opts.get("title", True):
            figure.suptitle("Growth", color=INK, fontsize=plot_font(1))
        figure.tight_layout()
        return markers

    def _new_wl_viewer(self):
        """A fresh viewer for the snapshots, replacing the last one."""
        previous, self._wl_viewer = self._wl_viewer, None
        if previous is not None:
            try:
                previous.close()
            except Exception:
                pass
        self._wl_viewer = napari.Viewer(title="White-light snapshots - drift check")
        return self._wl_viewer

    # --- the white light over the fluorescence ------------------------------
    def _wl_overlay_layer(self):
        for layer in list(self.viewer.layers):
            if (getattr(layer, "metadata", None) or {}).get(WL_OVERLAY_TAG):
                return layer
        return None

    def _wl_overlay_data(self):
        """(data, canvas origin, drift taken out?, timed?) - or (None, why not).

        The snapshots move back by the drift since the movie's first frame
        whenever the fluorescence image does, so both stand in the first
        frame's coordinates; and they take the movie's frame axis when the
        frame clock can place them on it.
        """
        stack, record = self._wl_stack, self._drift_record
        if stack is None:
            return None, "no white-light snapshots beside this acquisition"
        cmap, problem = self._overlay_camera_map()
        if cmap is None:
            return None, problem or "no camera map applies"
        if cmap.offset is None:
            return None, f"{cmap.source} holds displacements only"
        frames = self._drift_per_frame()
        shifts = np.zeros((len(stack), 2))
        drift_out = bool(record is not None and frames is not None
                         and self.drift_enable_box.isChecked()
                         and self.drift_shift_image_box.isChecked())
        if drift_out:
            f = drift_io.smoother(record.t, record.dx, record.dy,
                                  float(self.drift_smoothing_box.value()))
            rx, ry = f(stack.t_epoch)
            ox, oy = frames["origin"]
            shifts = np.column_stack([-(ry - oy), -(rx - ox)])
        shifted = drift_io.ShiftedStack(stack, shifts)
        # Rebuilding is cheap; handing napari new data is not - it re-reads
        # and re-shifts the frame on screen - so the layer keeps its data
        # unless this key says the frames themselves changed.
        key = (id(stack), drift_out, np.round(shifts, 4).tobytes())
        rec = self._deformation
        fd = self._fluorescence_deformation()
        if (drift_out and self._growth_active() and rec is not None and fd is not None
                and len(rec.t) == len(stack)):
            # the growth correction: every snapshot drawn in the geometry the
            # fluorescence is carried to - the first frame's or the last snapshot's
            x0, x1, y0, y1 = (float(v) for v in rec.region)
            corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            placed = fd._to_reference(corners)
            cx0, cy0 = np.floor(placed.min(axis=0)).astype(int)
            cx1, cy1 = np.ceil(placed.max(axis=0)).astype(int)
            shifted = drift_io.WarpedStack(
                stack, inverse=lambda k, xy: rec.inverse[k](fd._from_reference(xy)),
                forward=lambda k, xy: fd._to_reference(rec.forward[k](xy)),
                origin_yx=(int(cy0), int(cx0)), canvas=(int(cy1 - cy0), int(cx1 - cx0)),
                key=("growth", id(rec), fd.reference_time))
            key = (id(stack), "growth", id(rec), fd.reference_time)
        frame_t = frames["t"] if frames is not None else self._clock_frame_times()
        if frame_t is None:
            return (cmap, shifted, shifted.origin_yx, drift_out, False, key), None
        index = drift_io.nearest_snapshot(stack.t_epoch, frame_t)
        return (cmap, drift_io.FrameIndexedStack(shifted, index), shifted.origin_yx,
                drift_out, True, key + (index.tobytes(),)), None

    def _overlay_camera_map(self):
        """(map, why not) for laying the snapshots over the image: the drift
        record's, or - without a record - the calibrated one, if the
        acquisition's white-light orientation and frame say it holds."""
        if self._drift_record is not None:
            return self._camera_map, self._camera_map_problem
        meta = {"frame_shape": [int(n) for n in self._wl_stack.shape[1:]]}
        orientation = self._acquisition_values.get("wl_orientation")
        if orientation:
            meta["wl_orientation"] = orientation
        return drift_io.camera_map(SimpleNamespace(meta=meta))

    def _clock_frame_times(self):
        """Mean time of each frame the pipeline sees, from the frame clock alone."""
        clock = self._frame_clock
        if clock is None:
            return None
        n_raw = self._drift_raw_frame_count()
        factor = max(1, int(self._time_bin_applied))
        if n_raw // factor < 1:
            factor = 1
        t, _ = clock.times(n_raw)
        n = n_raw // factor
        return t[: n * factor].reshape(n, factor).mean(axis=1)

    def _wl_overlay_affine(self, cmap, origin_yx):
        """4x4 (frame, row, column): white-light canvas px to the viewer's nm."""
        roi = (self.wl_roi_x_box.value(), self.wl_roi_y_box.value())
        plane = cmap.image_affine(self.pixel_size_box.value(), roi)
        # canvas pixel (i, j) is white-light pixel (i + origin_row, j + origin_col)
        plane = plane @ np.array([[1.0, 0.0, origin_yx[0]],
                                  [0.0, 1.0, origin_yx[1]],
                                  [0.0, 0.0, 1.0]])
        affine = np.eye(4)
        affine[1:3, 1:3] = plane[:2, :2]
        affine[1:3, 3] = plane[:2, 2]
        return affine

    def show_wl_overlay(self):
        """Add the white light, registered to the fluorescence, as a layer."""
        self._update_wl_overlay(create=True)

    def _update_wl_overlay(self, create=False):
        """Bring the white-light layer in line with the drift, pixel size and ROI."""
        if not hasattr(self, "wl_overlay_status") or getattr(self, "_wl_overlay_busy", False):
            return
        layer = self._wl_overlay_layer()
        if layer is None and not create:
            self._update_wl_overlay_status(None)
            return
        result, problem = self._wl_overlay_data()
        if result is None:
            if create:
                self.log(f"White light not shown: {problem}.")
            self._update_wl_overlay_status(problem)
            return
        # Adding a layer fires the viewer's own update, which comes back here.
        self._wl_overlay_busy = True
        try:
            self._place_wl_overlay(layer, *result)
        finally:
            self._wl_overlay_busy = False

    def _place_wl_overlay(self, layer, cmap, data, origin_yx, drift_out, timed, key):
        affine = self._wl_overlay_affine(cmap, origin_yx)
        if layer is None:
            first = np.asarray(self._wl_stack[0], dtype=float)
            low, high = (float(v) for v in np.percentile(first, (0.5, 99.8)))
            layer = self.viewer.add_image(
                data, name=WL_OVERLAY_LAYER_NAME, colormap="cyan", blending="additive",
                opacity=0.7, contrast_limits=(low, max(high, low + 1.0)), affine=affine,
                units=self._viewer_units(3), metadata={WL_OVERLAY_TAG: True})
            # Just above the fluorescence image, under everything drawn on it.
            try:
                name = self._image_layer_name
                if name and name in self.viewer.layers:
                    self.viewer.layers.move(self.viewer.layers.index(layer),
                                            self.viewer.layers.index(name) + 1)
            except Exception:
                pass
            self.log("White light laid over the fluorescence through the camera map from "
                     f"{cmap.source}"
                     + ("" if self._sensor_roi_recorded else
                        " - the fluorescence ROI was not recorded, so it is placed as "
                        "if the image started at the sensor's corner"))
        else:
            if key != getattr(self, "_wl_overlay_key", None):
                layer.data = data
            current = np.asarray(getattr(layer.affine, "affine_matrix", layer.affine))
            if current.shape != affine.shape or not np.allclose(current, affine):
                layer.affine = affine
        self._wl_overlay_key = key
        self._update_wl_overlay_status(None, drift_out, timed)

    def _update_wl_overlay_status(self, problem, drift_out=False, timed=False):
        shown = self._wl_overlay_layer() is not None
        self.wl_overlay_button.setEnabled(self._wl_stack is not None)
        if problem:
            self.wl_overlay_status.setText(f"Not shown: {problem}.")
        elif not shown:
            self.wl_overlay_status.setText(
                "No white-light snapshots loaded" if self._wl_stack is None else
                f"{len(self._wl_stack)} snapshots ready to lay over the fluorescence.")
        else:
            parts = ["Shown" + (", drift-corrected" if drift_out else ", as recorded")]
            parts.append("each frame with its nearest snapshot" if timed else
                         "on the snapshots' own frame axis (no frame clock)")
            if not self._sensor_roi_recorded:
                parts.append("fluorescence ROI not recorded - check the x, y above")
            self.wl_overlay_status.setText("; ".join(parts) + ".")

    # --- RCC ----------------------------------------------------------------
    def _rcc_input_table(self):
        source = self.rcc_source_box.currentData()
        if source == "shown":
            return self._displayed_localizations()
        return self.df_filtered

    def estimate_rcc(self):
        table = self._rcc_input_table()
        x_col, y_col = self._resolve_column("x"), self._resolve_column("y")
        frame_col = self._resolve_column("frame")
        if table is None or table.empty or not all(
                c in table.columns for c in (x_col, y_col, frame_col)):
            self.log("RCC needs localizations - load or fit some first")
            return
        frames = table[frame_col].to_numpy(dtype=float) + self._frame_offset()
        x = table[x_col].to_numpy(dtype=float)
        y = table[y_col].to_numpy(dtype=float)
        # On the positions without the RCC now in force: estimated on top of
        # itself, it would measure what it had already removed - nothing - and
        # replace itself with that.
        applied = self._applied_drift
        if applied is not None and "rcc" in applied[2].split("+"):
            refinement = self._rcc_per_frame_nm()
            if refinement is not None:
                rdx, rdy, _off = drift_io.lookup(np.nan_to_num(frames), *refinement)
                x, y = x + rdx, y + rdy
        params = {
            "segment_frames": int(self.rcc_segment_box.value()),
            "pixel_nm": float(self.rcc_pixel_box.value()),
            "blur_nm": float(self.rcc_blur_box.value()),
            "max_shift_nm": float(self.rcc_max_shift_box.value()),
            "rmax_nm": float(self.rcc_rmax_box.value()),
        }
        n_frames = self._stack_frame_count()
        on_top = ("the white-light correction" if self._wl_fingerprint()[0]
                  else "the table's own correction" if self._wl_fingerprint()[3]
                  else "no other correction")
        self.log(f"RCC on {len(x)} localizations, on top of {on_top}: "
                 f"segments of {params['segment_frames']} frames...")
        self.rcc_button.setEnabled(False)
        self.rcc_progress.setVisible(True)
        self.rcc_progress.setValue(0)
        self._arm_cancel(self._rcc_cancel, self.rcc_cancel_button)
        run = {"fingerprint": self._wl_fingerprint(), "params": params,
               "source": self.rcc_source_box.currentData(), "n_input": int(len(x))}
        worker = _rcc_worker(x, y, frames, n_frames, params, self._rcc_cancel)
        worker.yielded.connect(lambda frac: self.rcc_progress.setValue(int(frac * 100)))
        worker.returned.connect(lambda result: self._on_rcc_finished(result, run))
        worker.errored.connect(lambda exc: self.log(f"RCC failed: {exc}"))
        worker.finished.connect(self._on_rcc_worker_finished)
        self._rcc_worker_ref = worker
        worker.start()

    def _on_rcc_worker_finished(self):
        self.rcc_button.setEnabled(True)
        self.rcc_cancel_button.setEnabled(False)
        self.rcc_progress.setVisible(False)
        self._rcc_worker_ref = None

    def _on_rcc_finished(self, result, run):
        if result is CANCELLED:
            self.log("RCC cancelled - the previous estimate, if any, is unchanged")
            return
        self._rcc = dict(run, result=result,
                         estimated_at=datetime.now().isoformat(timespec="seconds"))
        n_pairs = len(result["pairs"])
        dropped = int(n_pairs - int(np.sum(result["kept"])))
        px, py = drift_io.rcc_per_frame(result, self._stack_frame_count())
        self.log(f"RCC: {len(result['centres'])} segments, {n_pairs} pairs"
                 + (f" ({dropped} dropped as off by more than {run['params']['rmax_nm']:g} nm)"
                    if dropped else "")
                 + f", agreeing to {result['rms_error_nm']:.1f} nm rms. Drift left over the "
                 f"movie: x {px[-1]:+.0f}, y {py[-1]:+.0f} nm"
                 + ("" if result["pixel_nm"] <= run["params"]["pixel_nm"] + 1e-9 else
                    f" (rendered at {result['pixel_nm']:.0f} nm to fit the field)"))
        self._refresh_drift_correction()

    def _update_rcc_status(self):
        if not hasattr(self, "rcc_status"):
            return
        table = self._rcc_input_table() if self.df_filtered is not None else None
        if table is None or table.empty:
            self.rcc_segment_label.setText("Load or fit localizations first.")
            self.rcc_button.setEnabled(False)
        else:
            n_frames = self._stack_frame_count()
            n_segments = max(2, int(round(n_frames / max(1, self.rcc_segment_box.value()))))
            self.rcc_segment_label.setText(
                f"{n_segments} segments of about {len(table) // n_segments} localizations "
                f"({n_segments * (n_segments - 1) // 2} pairs to correlate).")
            self.rcc_button.setEnabled(self._rcc_worker_ref is None)

        rcc = self._rcc
        if rcc is None:
            self.rcc_status.setText("No estimate yet.")
            return
        result = rcc["result"]
        lines = []
        pairs = result.get("pairs")
        kept = result.get("kept")
        if pairs is not None and kept is not None:
            lines.append(f"{len(result['centres'])} segments from {rcc.get('n_input', 0)} "
                         f"localizations; {int(np.sum(kept))} of {len(pairs)} pairs kept, "
                         f"agreeing to {result.get('rms_error_nm', float('nan')):.1f} nm rms.")
        else:
            lines.append(f"{len(result['centres'])} segments, restored from the session.")
        px, py = drift_io.rcc_per_frame(result, self._stack_frame_count())
        lines.append(f"Drift left over the movie: x {px[-1]:+.0f}, y {py[-1]:+.0f} nm "
                     f"(largest {float(np.hypot(px, py).max()):.0f} nm).")
        if not self._rcc_is_current():
            lines.append("Stale: the correction it refined has changed since (record, "
                         "smoothing, pixel size or frame numbering) - estimate it again. "
                         "Not applied.")
        elif not self.rcc_apply_box.isChecked():
            lines.append("Not applied.")
        else:
            lines.append("Applied, on top of the white-light drift." if rcc["fingerprint"][0]
                         else "Applied.")
        self.rcc_status.setText("\n".join(lines))

    def _draw_rcc_plot(self, figure=None, opts=None):
        """The RCC estimate: each segment, and the line through them."""
        export = figure is not None
        opts = opts or {}
        figure = figure if export else self.rcc_figure
        figure.clear()
        rcc = self._rcc
        if rcc is None:
            figure.patch.set_facecolor(PANEL_BG)
            if not export:
                self.rcc_canvas.draw_idle()
            return
        result = rcc["result"]
        n_frames = self._stack_frame_count()
        px, py = drift_io.rcc_per_frame(result, n_frames)
        # The frame clock's own times when there is one - the same axis as the
        # white-light plot above - else the frame rate in the Track tab.
        timed = self._drift_per_frame()
        if timed is not None and len(timed["t"]) >= n_frames:
            t = timed["t"][:n_frames] - timed["t"][0]
        else:
            t = np.arange(n_frames) * self._frame_interval_s()
        centres = np.asarray(result["centres"], dtype=float)
        ax = figure.add_subplot(111)
        for curve, color, label in ((px, ACCENT, "x"), (py, LAVENDER, "y")):
            ax.plot(t, curve, color=color, linewidth=1.2, label=label)
            ax.plot(np.interp(centres, np.arange(n_frames), t),
                    np.interp(centres, np.arange(n_frames), curve),
                    "o", color=color, markersize=4)
        ax.axhline(0.0, color=INK_DIM, linewidth=0.6)
        ax.set_xlabel("Time since the first frame (s)")
        ax.set_ylabel("Drift left (nm)" if rcc["fingerprint"][0] else "Drift (nm)")
        legend = ax.legend(fontsize=plot_font(-2), loc="upper left", ncol=2,
                           facecolor=PLOT_BG, edgecolor=PANEL_LINE, labelcolor=INK)
        legend.get_frame().set_alpha(0.85)
        title = "RCC estimate" + ("" if self._rcc_is_current() else " (stale)")
        style_axes(figure, ax, title=title if opts.get("title", True) else None)
        if not opts.get("grid", True):
            ax.grid(False)
        figure.tight_layout()
        if not export:
            self.rcc_canvas.draw_idle()

    def _rcc_metadata(self):
        """The RCC settings, and the estimate itself when there is one."""
        section = {
            "apply": self.rcc_apply_box.isChecked(),
            "source": self.rcc_source_box.currentData(),
            "segment_frames": self.rcc_segment_box.value(),
            "pixel_nm": self.rcc_pixel_box.value(),
            "blur_nm": self.rcc_blur_box.value(),
            "max_shift_nm": self.rcc_max_shift_box.value(),
            "rmax_nm": self.rcc_rmax_box.value(),
            "estimate": None,
        }
        rcc = self._rcc
        if rcc is not None:
            result = rcc["result"]
            kept = result.get("kept")
            section["estimate"] = {
                "centres_frame": [float(v) for v in result["centres"]],
                "dx_nm": [float(v) for v in result["dx"]],
                "dy_nm": [float(v) for v in result["dy"]],
                "rms_error_nm": float(result.get("rms_error_nm", float("nan"))),
                "n_pairs": int(len(result["pairs"])) if result.get("pairs") is not None else None,
                "n_pairs_dropped": (int(len(kept) - int(np.sum(kept)))
                                    if kept is not None else None),
                "pixel_nm_used": float(result.get("pixel_nm", float("nan"))),
                "n_localizations": int(rcc.get("n_input", 0)),
                "estimated_with": rcc.get("params"),
                "estimated_on": rcc.get("source"),
                "estimated_at": rcc.get("estimated_at"),
                "refines": list(rcc["fingerprint"]),
                "applied": bool(self._rcc_per_frame_nm() is not None),
            }
        return section

    @staticmethod
    def _rcc_from_metadata(section):
        """An RCC estimate back from _rcc_metadata's record of it, or None."""
        estimate = section.get("estimate") if isinstance(section, dict) else None
        if not isinstance(estimate, dict):
            return None
        try:
            result = {
                "centres": np.asarray(estimate["centres_frame"], dtype=float),
                "dx": np.asarray(estimate["dx_nm"], dtype=float),
                "dy": np.asarray(estimate["dy_nm"], dtype=float),
                "rms_error_nm": float(estimate.get("rms_error_nm") or float("nan")),
                "pixel_nm": float(estimate.get("pixel_nm_used") or float("nan")),
            }
            return {"result": result, "fingerprint": list(estimate["refines"]),
                    "params": estimate.get("estimated_with"),
                    "source": estimate.get("estimated_on"),
                    "n_input": int(estimate.get("n_localizations") or 0),
                    "estimated_at": estimate.get("estimated_at")}
        except (KeyError, TypeError, ValueError):
            return None

    def _restore_previous_run_settings(self, locs_path):
        """Restore the analysis settings of the run that produced a table found
        beside the data - but not the microscope.

        An auto-loaded table is usually a *filtered* export, so loading it
        without its parameters puts data on screen that the controls actively
        misdescribe: bounds that were applied showing as wide open, a colour
        scale that belongs to a different metric. Picking the table up
        automatically is meant to carry on where that run left off, and that
        only holds if the controls come with it.

        The instrument is the exception, and stays where the user put it. Pixel
        size, gain, offset and frame rate describe the microscope, not a choice
        about the analysis: on this setup the pixel size cannot be derived from
        the metadata at all and is measured by hand, so an older run is as
        likely to hold a stale value as a correct one. Nobody asked for these
        settings - opening data merely found them - and silently undoing a
        calibration is a worse failure than leaving a control the user set. A
        disagreement is reported instead, which is the part actually worth
        knowing.
        """
        locs_path = Path(locs_path)
        # The exporter writes metadata.json at the root of the analysis folder
        # and the tables into data/ beneath it, so look beside and one level up.
        for folder in (locs_path.parent, locs_path.parent.parent):
            candidate = folder / "metadata.json"
            try:
                if not candidate.is_file():
                    continue
            except OSError:
                continue
            try:
                with open(candidate, encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except Exception as exc:
                self.log(f"Found {candidate.name} from that run but could not read it: {exc}")
                return
            applied, _skipped, notes = self.apply_settings(
                metadata, include_instrument=False)
            exported = metadata.get("exported_at")
            self.log(
                f"Restored {len(applied)} analysis settings from that run's "
                f"{candidate.name}"
                + (f", exported {exported}" if exported else "")
                + " - the pixel size, gain, offset and frame rate are yours and "
                "were left as they are."
            )
            for note in notes:
                self.log(f"  note: {note}")
            return
        self.log("No metadata.json beside those localizations, so the parameters "
                 "on screen are not necessarily the ones that produced them.")

    def _find_companion_file(self, base_path, filename_patterns, analysis_relative_path=None):
        base_path = Path(base_path)
        folder = base_path.parent
        stem = base_path.stem
        for pattern in filename_patterns:
            candidate = folder / pattern.format(stem=stem)
            if candidate.is_file():
                return candidate

        if analysis_relative_path:
            relatives = ((analysis_relative_path,)
                         if isinstance(analysis_relative_path, str)
                         else tuple(analysis_relative_path))
            for run_dir in self._analysis_run_dirs(folder):
                for relative in relatives:
                    candidate = run_dir / relative
                    if candidate.is_file():
                        return candidate
        return None

    @staticmethod
    def _analysis_run_dirs(folder):
        """Every run folder beside `folder`, most recent first.

        Two layouts, because analyses made before runs were dated should still
        be found: the dated ones under analysis/, and the older numbered
        analysis, analysis_2, analysis_3 siblings. Dated runs come first - if
        both exist, the dated ones are the newer scheme and so the newer work.
        """
        runs = []
        root = Path(folder) / ANALYSIS_ROOT
        try:
            if root.is_dir():
                # The stamp format sorts lexicographically, so this is by date.
                runs.extend(sorted((d for d in root.iterdir() if d.is_dir()),
                                   key=lambda d: d.name, reverse=True))
        except OSError:
            pass

        numbered = []
        try:
            for d in Path(folder).glob("analysis*"):
                if not d.is_dir():
                    continue
                suffix = d.name[len(ANALYSIS_ROOT):]
                if suffix == "":
                    numbered.append((1, d))
                elif suffix.startswith("_") and suffix[1:].isdigit():
                    numbered.append((int(suffix[1:]), d))
        except OSError:
            pass
        runs.extend(d for _n, d in sorted(numbered, key=lambda t: -t[0]))
        return runs

    def _try_autoload_trajectories(self, base_path):
        if self._session_restore is not None:
            # A session says for itself where its trajectories came from, and
            # rebuilds them the way it recorded. Picking up whatever file
            # happens to sit next to the data would restore a different run.
            return
        found = self._find_companion_file(base_path, TRAJ_FILENAME_PATTERNS, TRAJ_ANALYSIS_SUBPATH)
        if found is None:
            return
        try:
            traj = pd.read_csv(found)
        except Exception as exc:
            self.log(f"Found candidate trajectories file {found.name} but could not read it: {exc}")
            return
        if not {"particle", "frame", "x", "y"}.issubset(traj.columns):
            self.log(f"Found {found.name} but it doesn't look like a trajectories file - skipped")
            return
        self.tracks = traj.reset_index(drop=True)
        self._tracks_source_path = found     # read, not linked: never re-linked
        self._tracks_key = self._localization_set_key()
        self._invalidate_track_filter()
        self._track_diffusion_cache = None
        self._track_msd_cache = None
        self.compute_d_button.setEnabled(True)
        self._start_fit_free_metrics_worker()
        self._update_status_header()
        self.log(
            f"Auto-detected and loaded {self.tracks['particle'].nunique()} pre-linked "
            f"trajectories from {found.name}"
        )
        self.render_overlay()

    def _ingest_localization_dataframe(self, df, log_message, frame_is_zero_indexed):
        # Shared by CSV import (Load data tab) and in-app 2D localization
        # (Localize tab): whichever produced the dataframe, wire it into the
        # same filter/link/analysis pipeline.
        self.df = df
        self.df_filtered = self.df.copy()
        self.column_map = infer_column_map(self.df.columns)
        # New localizations are a new analysis, so its outputs start a folder of
        # their own. Without this the figures folder was chosen once per session
        # and every later run wrote into the first one's - so a second fit's
        # graphs landed among the first's, describing different data under
        # neighbouring filenames.
        self._start_new_output_folders()
        self.tracks = None
        self._tracks_key = None
        self._linked_memory.clear()
        self._invalidate_track_filter()
        self._track_diffusion_cache = None
        self._track_msd_cache = None
        self._track_distance_cache = None
        self._track_net_cache = None
        self._track_straightness_cache = None
        self._track_duration_cache = None
        self.log(log_message)

        if frame_is_zero_indexed:
            self._frame_shift = 0
        else:
            # A table whose first frame is 1 is almost certainly 1-indexed, so
            # start it shifted; the buttons under the CSV field undo or extend
            # that if the guess is wrong.
            frame_col = self._resolve_column("frame")
            self._frame_shift = 0
            if frame_col and frame_col in self.df.columns and not self.df[frame_col].empty:
                if int(self.df[frame_col].min()) == 1:
                    self._frame_shift = -1
                    self.log("Frame numbers start at 1: shifted by -1 to match the image stack")

        # An RCC estimate was measured on the localizations it came with; new
        # ones get their own. A session restoring its estimate is the exception.
        if self._rcc is not None and self._session_restore is None:
            self._rcc = None
            self.log("The RCC estimate belonged to the previous localizations and "
                     "was dropped - estimate it again on the Drift tab.")
        # After the frame shift, which decides which frame - and so which
        # moment of the drift record - each localization belongs to. Before the
        # filter tab, whose default bounds are the extent of what it is given.
        self._set_source_table(self.df)
        self.df = self._corrected_table(self._localization_drift(self._df_source))
        self.df_filtered = self.df.copy()
        self._log_drift_applied()

        self._build_filter_tab_contents()
        self.apply_filters_button.setEnabled(True)
        self.reset_filters_button.setEnabled(True)
        self.link_button.setEnabled(True)
        self.render_button.setEnabled(True)
        self.compute_d_button.setEnabled(False)

        self.render_overlay()
        self._sync_xy_roi_layer()
        self.viewer.tooltip.visible = True
        self._refresh_render_tab()
        self._update_frame_shift_label()
        self._update_status_header()
        self.data_table_model.set_dataframe(self.df_filtered)
        self.data_table_label.setText(f"{len(self.df_filtered)} rows x {len(self.df_filtered.columns)} columns")
        self._show_drift_state()

    # ------------------------------------------------------------------
    # Localize (2D): detection + sub-pixel Gaussian fitting
    # ------------------------------------------------------------------
    def _get_localize_image_layer(self):
        return self._source_image_layer()

    def _source_image_layer(self):
        """The raw stack: the first Image layer this plugin did not produce.

        Renders are Image layers too, so without the exclusion the Localize tab
        would happily start detecting spots inside a reconstruction, and the
        next render would take its field of view from the previous one.
        """
        for layer in list(self.viewer.layers.selection) + list(self.viewer.layers):
            if isinstance(layer, napari.layers.Image) and not is_render_layer(layer):
                return layer
        return None

    def _loc2d_box_size(self):
        box = self.loc_box_size.value()
        return box if box % 2 == 1 else box + 1

    def _on_loc2d_box_changed(self, value):
        if value % 2 == 0:
            self.loc_box_size.blockSignals(True)
            self.loc_box_size.setValue(value + 1)
            self.loc_box_size.blockSignals(False)

    def _loc2d_stack(self, layer):
        # The frames as the camera recorded them, even while they are shown
        # drift-corrected: the localizations are corrected after the fit.
        stack = drift_io.unshifted(layer.data)
        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]
        return stack

    def loc2d_preview(self):
        layer = self._get_localize_image_layer()
        if layer is None:
            self.log("Load or select an image stack first")
            return
        stack = self._loc2d_stack(layer)
        frame_idx = int(np.clip(self._get_current_frame(), 0, stack.shape[0] - 1))
        box = self._loc2d_box_size()

        if len(self._loc2d_candidates) != stack.shape[0]:
            self._loc2d_candidates = [None] * stack.shape[0]
            self._loc2d_counts = np.zeros(stack.shape[0], dtype=int)

        y, x, ng = identify_in_frame(
            np.asarray(stack[frame_idx], dtype=np.float32), self.loc_min_ng_box.value(), box
        )
        self._loc2d_candidates[frame_idx] = (y, x, ng)
        self._loc2d_counts[frame_idx] = len(y)
        self.loc_fit_button.setEnabled(bool(len(y)))
        self._update_loc2d_candidate_overlay()
        self.log(f"Preview: {len(y)} candidates on frame {frame_idx}")

    def loc2d_detect_all(self):
        layer = self._get_localize_image_layer()
        if layer is None:
            self.log("Load or select an image stack first")
            return
        stack = self._loc2d_stack(layer)
        box = self._loc2d_box_size()
        min_ng = self.loc_min_ng_box.value()
        self.log(f"Detecting candidates on {stack.shape[0]} frames (box={box}, min NG={min_ng:.1f})...")

        self.loc_detect_button.setEnabled(False)
        self.loc_detect_progress.setVisible(True)
        self.loc_detect_progress.setValue(0)
        self._arm_cancel(self._loc2d_detect_cancel, self.loc_detect_cancel_button)

        worker = _detect_worker(stack, box, min_ng, self._loc2d_detect_cancel)
        worker.yielded.connect(lambda frac: self.loc_detect_progress.setValue(int(frac * 100)))
        worker.returned.connect(self._on_loc2d_detect_finished)
        worker.errored.connect(lambda exc: self.log(f"Detection failed: {exc}"))
        worker.finished.connect(self._on_loc2d_detect_worker_finished)
        self._loc2d_detect_worker_ref = worker
        worker.start()

    def _on_loc2d_detect_worker_finished(self):
        self.loc_detect_button.setEnabled(True)
        self.loc_detect_cancel_button.setEnabled(False)
        self.loc_detect_progress.setVisible(False)
        self._loc2d_detect_worker_ref = None

    def _on_loc2d_detect_finished(self, result):
        if result is CANCELLED:
            self.log("Detection cancelled - no candidates were kept")
            return
        candidates, counts = result
        self._loc2d_candidates = candidates
        self._loc2d_counts = counts
        total = int(counts.sum())
        self.log(f"Detected {total} candidates across {len(candidates)} frames")
        self.loc_fit_button.setEnabled(total > 0)
        self._update_loc2d_candidate_overlay()
        self._draw_loc2d_counts()

    def _draw_loc2d_counts(self, figure=None, opts=None):
        export = figure is not None
        opts = opts or {}
        figure = figure if export else self.loc_counts_figure
        figure.clear()
        if self._loc2d_counts is None or len(self._loc2d_counts) == 0:
            figure.patch.set_facecolor(PANEL_BG)
            if not export:
                self.loc_counts_canvas.draw_idle()
            return
        ax = figure.add_subplot(111)
        ax.plot(np.arange(len(self._loc2d_counts)), self._loc2d_counts,
                color=ACCENT, linewidth=1.2)
        ax.fill_between(np.arange(len(self._loc2d_counts)), self._loc2d_counts,
                        color=ACCENT, alpha=0.18)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Detections")
        style_axes(figure, ax,
                   title="Detections vs frame" if opts.get("title", True) else None)
        if not opts.get("grid", True):
            ax.grid(False)
        figure.tight_layout()
        if not export:
            self.loc_counts_canvas.draw_idle()

    def _update_loc2d_candidate_overlay(self):
        """Show the detection candidates as squares, on every frame at once.

        The candidates carry their frame index as their first coordinate, so
        napari slices them itself as the dims slider moves - exactly like the
        localizations layer. The earlier version rebuilt a per-frame Shapes
        layer from a timer hooked to `dims.events.current_step`, which meant
        the boxes only appeared if that callback fired, only while the Localize
        tab happened to be showing, and cost a layer rebuild per frame while
        scrubbing. A Points layer with square symbols draws the same thing with
        none of that: no callback to miss, no tab gate, and no per-frame work
        at all (napari handles hundreds of thousands of points comfortably,
        where a Shapes layer of the same size is unusable - which is why the
        per-frame rebuild existed in the first place).
        """
        frames, centres = self._loc2d_candidate_points()
        if centres is None or not self.loc_show_candidates_box.isChecked():
            self._remove_layer(LOC2D_CANDIDATES_LAYER_NAME)
            return

        coords = np.column_stack([frames, centres[:, 0], centres[:, 1]])
        # Found on the frames as recorded; on a stack shown drift-corrected
        # they move with the spots they were found on.
        shifts = self._displayed_image_shifts()
        warped = self._displayed_warp()
        if warped is not None:
            for f in np.unique(frames):
                sel = frames == f
                xy = warped.displace(int(f), centres[sel][:, ::-1])
                coords[sel, 1], coords[sel, 2] = xy[:, 1], xy[:, 0]
        elif shifts is not None and len(shifts):
            index = np.clip(frames.astype(np.int64), 0, len(shifts) - 1)
            coords[:, 1] += shifts[index, 0]
            coords[:, 2] += shifts[index, 1]
        size = float(self._loc2d_box_size())
        if LOC2D_CANDIDATES_LAYER_NAME in self.viewer.layers:
            layer = self.viewer.layers[LOC2D_CANDIDATES_LAYER_NAME]
            layer.data = coords
            layer.size = size
            layer.visible = True
            return
        kwargs = dict(
            name=LOC2D_CANDIDATES_LAYER_NAME,
            symbol="square",
            size=size,
            face_color="transparent",
        )
        # napari renamed the Points outline from edge_* to border_* in 0.5.
        if napari.__version__.startswith("0.4"):
            kwargs.update(edge_color="yellow", edge_width=0.08, edge_width_is_relative=True)
        else:
            kwargs.update(border_color="yellow", border_width=0.08, border_width_is_relative=True)
        try:
            self.viewer.add_points(
                coords, **self._placed(kwargs, np.asarray(coords).shape[-1]))
            self._apply_viewer_scale()
        except Exception as exc:
            self.log(f"Could not draw the detection candidates: {exc}")

    def _loc2d_candidate_points(self):
        """(frame index, (y, x)) for every detected candidate, or (None, None)."""
        per_frame = [
            (index, cand) for index, cand in enumerate(self._loc2d_candidates or [])
            if cand is not None and len(cand[0]) > 0
        ]
        if not per_frame:
            return None, None
        frames = np.concatenate([
            np.full(len(cand[0]), index, dtype=np.float64) for index, cand in per_frame
        ])
        centres = np.concatenate([
            np.column_stack([np.asarray(cand[0], dtype=np.float64),
                             np.asarray(cand[1], dtype=np.float64)])
            for _index, cand in per_frame
        ])
        return frames, centres

    def _on_tab_changed(self, index=None):
        if index is not None and index == getattr(self, "_localize_tab_index", None):
            self._start_fit_kernel_warmup()
        # The x/y filter box appears with the Filter tab and goes away again
        # unless it is actually cropping something.
        if self.df is not None:
            self._sync_xy_roi_layer()

    def _start_fit_kernel_warmup(self):
        """Compile the numba fit kernels in the background on first tab open.

        cache=True persists them to __pycache__, but an editable install
        invalidates that cache on every pull, and paying the compile inside the
        first fit looks like a hang.
        """
        if self._loc2d_warmup_started or not is_numba_available():
            return
        self._loc2d_warmup_started = True
        worker = _warmup_worker()
        worker.returned.connect(
            lambda elapsed: self.log(f"Fit kernels compiled in {elapsed:.1f} s")
            if elapsed > 0.5 else None
        )
        worker.errored.connect(lambda exc: self.log(f"Fit kernel warmup failed: {exc}"))
        worker.finished.connect(self._on_warmup_worker_finished)
        self._loc2d_warmup_worker_ref = worker
        worker.start()

    def _on_warmup_worker_finished(self):
        self._loc2d_warmup_worker_ref = None

    def loc2d_fit_all(self):
        layer = self._get_localize_image_layer()
        if layer is None or not self._loc2d_candidates:
            self.log("Run detection first")
            return
        stack = self._loc2d_stack(layer)

        backend = self.loc_backend_box.currentText()
        if backend == "auto":
            backend = "gpu" if is_gpufit_available() else "fast"
            self.log(f"Auto backend selected: {backend}")

        box = self._loc2d_box_size()
        gain = self.loc_gain_box.value()
        offset = self.loc_offset_box.value()

        self.loc_fit_button.setEnabled(False)
        self.loc_fit_progress.setVisible(True)
        self.loc_fit_progress.setValue(0)
        self._arm_cancel(self._loc2d_fit_cancel, self.loc_fit_cancel_button)

        worker = _fit_worker(
            stack, self._loc2d_candidates, box, backend, offset, gain, self._loc2d_fit_cancel
        )
        worker.yielded.connect(lambda frac: self.loc_fit_progress.setValue(int(frac * 100)))
        worker.returned.connect(self._on_loc2d_fit_finished)
        worker.errored.connect(lambda exc: self.log(f"Fitting failed: {exc}"))
        worker.finished.connect(self._on_loc2d_fit_worker_finished)
        self._loc2d_fit_worker_ref = worker
        worker.start()

    def _on_loc2d_fit_worker_finished(self):
        self.loc_fit_button.setEnabled(True)
        self.loc_fit_cancel_button.setEnabled(False)
        self.loc_fit_progress.setVisible(False)
        self._loc2d_fit_worker_ref = None

    def _on_loc2d_fit_finished(self, locs):
        if locs is CANCELLED:
            self.log("Fitting cancelled - localizations from finished frames were discarded")
            return
        n = len(locs["x"])
        if n == 0:
            self.log("Fitting produced no localizations")
            return
        pixel_size = self.pixel_size_box.value()
        df = pd.DataFrame(
            {
                "frame": locs["frame"].astype(int),
                "x [nm]": locs["x"].astype(float) * pixel_size,
                "y [nm]": locs["y"].astype(float) * pixel_size,
                "sigma [nm]": 0.5 * (locs["sx"].astype(float) + locs["sy"].astype(float)) * pixel_size,
                "sigma_x [nm]": locs["sx"].astype(float) * pixel_size,
                "sigma_y [nm]": locs["sy"].astype(float) * pixel_size,
                "intensity [photon]": locs["photons"].astype(float),
                "offset [photon]": locs["bg"].astype(float),
                "uncertainty [nm]": 0.5 * (locs["lpx"].astype(float) + locs["lpy"].astype(float)) * pixel_size,
                "net_gradient": locs["net_gradient"].astype(float),
            }
        )
        # The candidate squares stay: seeing which detections the fit kept, and
        # which it threw away, is the point of having both overlays. Untick
        # "Show detection candidates" on the Localize tab to hide them.
        self._ingest_localization_dataframe(
            df,
            f"Fitted {n} localizations from the loaded image stack (in-app 2D localization)",
            frame_is_zero_indexed=True,
        )
        # After the ingest, so the metadata written alongside describes the data
        # that is actually loaded rather than the state just before it arrived.
        self._autosave_localization_run(df)

    def _autosave_localization_run(self, df):
        """Write this fit to a dated folder of its own, beside the data.

        A fit is expensive and its result is easy to lose: the next one replaces
        it in memory, and the settings that produced it live only in the
        controls until something writes them down. Every run therefore gets its
        own folder with the localizations and the complete settings, and no run
        is ever overwritten by a later one - re-fitting after changing a single
        threshold leaves both results side by side, with the timestamps saying
        which was which.

        The table saved is the fit's own output, before any filtering: filters
        are recorded in the metadata and can be re-applied, but a discarded
        localization cannot be recovered from a filtered table.
        """
        if not self.loc_autosave_box.isChecked():
            return
        try:
            folder = self._make_analysis_folder(self._analysis_base_dir(), "localization")
            folder.mkdir(parents=True, exist_ok=True)
            metadata = self._collect_metadata(self.csv_edit.text().strip() or None)
        except Exception as exc:
            self.log(f"Could not start the automatic save of this fit: {exc}")
            return

        self.log(f"Saving this fit to {folder}...")
        worker = _export_worker(folder, [(LOCS_RUN_FILENAME, df)], metadata, None)
        worker.returned.connect(
            lambda result: self.log(f"Fit saved to {result}")
            if result is not CANCELLED else None)
        worker.errored.connect(lambda exc: self.log(f"Could not save this fit: {exc}"))
        worker.finished.connect(self._on_autosave_worker_finished)
        self._autosave_worker_ref = worker
        worker.start()

    def _on_autosave_worker_finished(self):
        self._autosave_worker_ref = None

    # ------------------------------------------------------------------
    # Render (SMLM): reconstructing an image from the localizations
    # ------------------------------------------------------------------
    def _on_render_mode_changed(self, *_args):
        mode = self.render_mode_box.currentData()
        for widget in (self.render_sigma_label, self.render_sigma_box):
            widget.setVisible(mode == "gaussian_global")
        for widget in (self.render_sigma_column_label, self.render_sigma_column_box,
                       self.render_clamp_label, self.render_sigma_min_box,
                       self.render_sigma_max_box):
            widget.setVisible(mode == "gaussian_local")
        # Counting a localization once is the point of a scatter render; a
        # photon weight would only turn it back into a brightness map.
        self.render_photons_box.setEnabled(mode != "scatter")
        self._update_render_info()

    def _on_render_grouping_changed(self, *_args):
        sliding = self.render_grouping_box.currentData() == "sliding"
        self.render_step_label.setVisible(sliding)
        self.render_step_box.setVisible(sliding)
        self._update_render_info()

    def _populate_render_sigma_columns(self):
        """Offer the columns that could describe how wide to draw a molecule."""
        columns = []
        if self.df is not None:
            for column in self.df.columns:
                name = str(column).lower()
                if ("uncert" in name or "precision" in name or name.startswith("lp")
                        or is_sigma_column(column)):
                    columns.append(str(column))
        previous = self.render_sigma_column_box.currentText()
        self.render_sigma_column_box.blockSignals(True)
        self.render_sigma_column_box.clear()
        self.render_sigma_column_box.addItems(columns)
        preferred = self.column_map.get("uncertainty")
        if previous in columns:
            self.render_sigma_column_box.setCurrentText(previous)
        elif preferred and str(preferred) in columns:
            self.render_sigma_column_box.setCurrentText(str(preferred))
        self.render_sigma_column_box.blockSignals(False)

    def _render_positions_px(self):
        """(x, y) in camera pixels for the localizations that pass the filters.

        The dynamics filter is one of those filters, which is what makes a
        reconstruction of only the fast - or only the directed - molecules a
        matter of ticking a box rather than of exporting and re-importing. With
        merging on, each confidently immobile trajectory is one of them.
        """
        df = self._render_table()
        if df is None or df.empty:
            return None, None
        x_col = self._resolve_column("x")
        y_col = self._resolve_column("y")
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            return None, None
        pixel_size = max(self.pixel_size_box.value(), 1e-9)
        return (df[x_col].to_numpy(dtype=float) / pixel_size,
                df[y_col].to_numpy(dtype=float) / pixel_size)

    def _render_frames(self):
        df = self._render_table()
        if df is None or df.empty:
            return None
        frame_col = self._resolve_column("frame")
        if not frame_col or frame_col not in df.columns:
            return None
        return df[frame_col].to_numpy(dtype=np.int64) + self._frame_offset()

    def _refresh_render_tab(self):
        """Re-read what the Render tab summarises, after any change to the data.

        The extent and frame range are cached here rather than recomputed in
        `_update_render_info`, which runs on every spin-box keystroke - a full
        pass over a million localizations per keystroke is exactly the kind of
        thing that makes a panel feel stuck.
        """
        if not hasattr(self, "render_source_label"):
            return
        self._render_frame_range = None
        self._render_extent_px = None
        x, y = self._render_positions_px()
        if x is not None and x.size:
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.any():
                self._render_extent_px = (
                    float(y[finite].min()), float(x[finite].min()),
                    float(y[finite].max()), float(x[finite].max()),
                )
        frames = self._render_frames()
        if frames is not None and frames.size:
            self._render_frame_range = (int(frames.min()), int(frames.max()))

        count = 0 if self.df_filtered is None else len(self.df_filtered)
        self.render_source_label.setText(
            f"{count} localizations pass the current filters"
            if self._render_extent_px else "No localizations loaded"
        )
        self.render_image_button.setEnabled(self._render_extent_px is not None)
        self.render_movie_button.setEnabled(
            self._render_extent_px is not None and self._render_frame_range is not None
        )
        self._populate_render_sigma_columns()
        self._update_render_info()

    def _render_field_of_view(self):
        """(shape, origin, source_layer) of the render, in camera pixels.

        With a raw stack loaded the render covers exactly it, so the two overlay
        pixel for pixel. Without one, the render covers the whole camera pixels
        the localizations fall in - the same grid the image would have imposed,
        so a table renders identically whether or not its image is open.
        """
        layer = self._source_image_layer()
        shape = getattr(getattr(layer, "data", None), "shape", None)
        if shape is not None and len(shape) >= 2:
            # A stack shown drift-corrected spans every frame's field of view,
            # and so does the render: localizations from the part of the sample
            # that drifted into view land on the grid, not off it.
            oy, ox = self._canvas_origin(layer)
            return (int(shape[-2]), int(shape[-1])), (oy - 0.5, ox - 0.5), layer

        extent = getattr(self, "_render_extent_px", None)
        if not extent:
            return None, None, None
        min_y, min_x, max_y, max_x = extent
        first_row, first_col = float(np.floor(min_y)), float(np.floor(min_x))
        rows = int(np.floor(max_y) - first_row) + 1
        cols = int(np.floor(max_x) - first_col) + 1
        return (rows, cols), (first_row - 0.5, first_col - 0.5), None

    def _render_movie_frame_count(self):
        if not getattr(self, "_render_frame_range", None):
            return None
        first, last = self._render_frame_range
        return smlm_render.group_count(
            first, last, self.render_frames_per_box.value(),
            self.render_grouping_box.currentData(), self.render_step_box.value(),
        )

    def _update_render_info(self):
        if not hasattr(self, "render_size_label"):
            return
        oversampling = self.render_oversampling_box.value()
        super_pixel_nm = self.pixel_size_box.value() / oversampling
        shape, _origin, _layer = self._render_field_of_view()
        if shape is None:
            self.render_size_label.setText(
                f"Super-resolved pixel {super_pixel_nm:.1f} nm. Load localizations "
                "to see the output size."
            )
            self.render_movie_label.setText("-")
            return

        rows, cols = smlm_render.output_shape(shape, oversampling)
        frame_bytes = smlm_render.estimate_bytes(shape, oversampling)
        self.render_size_label.setText(
            f"{shape[1]} x {shape[0]} camera px -> {cols} x {rows} super-resolved px "
            f"at {super_pixel_nm:.1f} nm/px, {frame_bytes / 1e9:.2f} GB per frame"
            + ("" if frame_bytes <= RENDER_MAX_BYTES else "  - too large, reduce the oversampling")
        )

        n_movie_frames = self._render_movie_frame_count()
        if not n_movie_frames:
            self.render_movie_label.setText("Load localizations with a frame column to render a movie.")
            return
        total = frame_bytes * n_movie_frames
        message = f"{n_movie_frames} super-resolved frames, {total / 1e9:.2f} GB in memory"
        if total > RENDER_MAX_BYTES:
            message += "  - too large, reduce the oversampling or use more raw frames per frame"
        if self.render_grouping_box.currentData() == "sliding":
            overlap = self.render_frames_per_box.value() / max(self.render_step_box.value(), 1)
            if overlap > RENDER_OVERLAP_WARN:
                message += (
                    f"  - each localization is redrawn ~{overlap:.0f}x because the "
                    "windows overlap that much; a larger step renders far faster"
                )
        self.render_movie_label.setText(message)
        self._update_scalebar_status()

    def _render_inputs(self):
        """Everything the engine needs, or None once the reason has been logged."""
        x, y = self._render_positions_px()
        if x is None or x.size == 0:
            # Two very different reasons to have nothing to draw, and saying the
            # wrong one sends the user to look for missing data that is there.
            if (self._active_metric_filters()
                    and self.df_filtered is not None and not self.df_filtered.empty):
                self.log("The dynamics filter is keeping no localizations, so "
                         "there is nothing to render - widen a range or untick it.")
            else:
                self.log("Load or fit localizations before rendering")
            return None
        shape, origin, source_layer = self._render_field_of_view()
        if shape is None:
            self.log("Could not work out a field of view to render into")
            return None

        pixel_size = max(self.pixel_size_box.value(), 1e-9)
        mode = self.render_mode_box.currentData()
        options = {
            "x_px": x, "y_px": y, "shape": shape, "origin": origin,
            "oversampling": self.render_oversampling_box.value(), "mode": mode,
        }
        info = {
            "mode": mode,
            "mode_label": smlm_render.MODES[mode],
            "oversampling": options["oversampling"],
            "pixel_size_nm_per_px": self.pixel_size_box.value(),
            "super_resolved_pixel_size_nm": pixel_size / options["oversampling"],
            "field_of_view_camera_px": [int(shape[0]), int(shape[1])],
            "origin_camera_px": [float(origin[0]), float(origin[1])],
            "field_of_view_from": "image layer" if source_layer is not None else "localization extent",
            "n_localizations": int(x.size),
            "value_units": "localizations per pixel",
        }

        # Widths and weights come from the same rows as the positions: the
        # localizations on screen, dynamics filter included. Read from every
        # filtered localization instead, a mobile or immobile render had one
        # width per localization of the whole table - refused outright for a
        # still image, and silently mismatched for a movie, whose frame sort
        # picked widths belonging to other molecules.
        shown = self._render_table()
        n_merged, n_merged_locs = 0, 0
        if MERGED_COUNT_COLUMN in shown.columns:
            counts = shown[MERGED_COUNT_COLUMN].to_numpy()
            n_merged = int((counts > 1).sum())
            n_merged_locs = int(counts[counts > 1].sum())
        if n_merged:
            info["merged_immobile"] = {
                "trajectories": n_merged, "localizations_merged": n_merged_locs,
                "static_at_p": self.render_population_p_box.value(),
                "max_detectable_d_um2_s": self.immobile_dmax_box.value(),
                "min_points": self.immobile_min_points_box.value(),
            }
            self.log(f"Merging {n_merged} immobile trajectories: their "
                     f"{n_merged_locs} localizations are drawn as {n_merged}")
        if mode == "gaussian_global":
            options["global_sigma_px"] = self.render_sigma_box.value() / pixel_size
            info["global_sigma_nm"] = self.render_sigma_box.value()
        elif mode == "gaussian_local":
            column = self.render_sigma_column_box.currentText()
            if not column or column not in shown.columns:
                self.log(
                    "No localization-precision column to take the width from - "
                    "pick another render mode, or one of the columns in the list."
                )
                return None
            low = self.render_sigma_min_box.value()
            high = max(self.render_sigma_max_box.value(), low)
            widths = np.clip(shown[column].to_numpy(dtype=float), low, high)
            options["sigma_px"] = widths / pixel_size
            info.update({"sigma_column": column, "sigma_clamp_nm": [low, high]})

        if mode != "scatter" and self.render_photons_box.isChecked():
            column = self._resolve_column("intensity")
            if column and column in shown.columns:
                options["weights"] = shown[column].to_numpy(dtype=float)
                info["weighted_by"] = column
                info["value_units"] = "photons per pixel"
            else:
                self.log("No photon-count column found - rendering unweighted counts")
        elif mode != "scatter" and n_merged:
            # A merged point stands for the localizations it replaced, so a
            # count render keeps the structure's brightness and gains only its
            # sharpness. A scatter render draws each molecule once, by design.
            options["weights"] = shown[MERGED_COUNT_COLUMN].to_numpy(dtype=float)
            info["weighted_by"] = "localizations merged into each point"

        # Soft sorting: every fitted trajectory, weighted by the probability
        # that it belongs to the population being rendered.
        which = self._population_class
        if which in ("immobile", "mobile") and self._soft_classes():
            if mode == "scatter":
                self.log("A scatter render draws each molecule once and cannot "
                         "weight by probability - pick a density render to sort softly")
            else:
                probability = self._row_probability(shown, which)
                if probability is not None:
                    base = options.get("weights")
                    options["weights"] = probability if base is None else base * probability
                    info["weighted_by"] = ((info["weighted_by"] + " x " if base is not None
                                            else "") + f"probability of being {which}")
                    info["soft_population"] = {
                        "population": which,
                        "expected_localizations": float(probability.sum()),
                    }

        return options, info, source_layer

    def _render_size_is_sane(self, shape, oversampling, n_frames):
        needed = smlm_render.estimate_bytes(shape, oversampling, n_frames)
        if needed <= RENDER_MAX_BYTES:
            return True
        rows, cols = smlm_render.output_shape(shape, oversampling)
        self.log(
            f"Refusing to render {n_frames} x {cols}x{rows} px = {needed / 1e9:.1f} GB "
            f"(the limit is {RENDER_MAX_BYTES / 1e9:.0f} GB). Reduce the oversampling"
            + (", or group more raw frames per super-resolved frame." if n_frames > 1 else ".")
        )
        return False

    def render_smlm_image(self):
        prepared = self._render_inputs()
        if prepared is None:
            return
        options, info, source_layer = prepared
        if not self._render_size_is_sane(options["shape"], options["oversampling"], 1):
            return
        options["gpu"], why = smlm_render.choose_backend(
            options["shape"], options["oversampling"], self.render_gpu_box.isChecked()
        )
        info["backend"] = "gpu" if options["gpu"] else "cpu"
        info["kind"] = "image"
        self.render_backend_label.setText(why)
        self.log(f"Rendering {info['n_localizations']} localizations ({info['mode_label']}) - {why}...")
        self._start_render("image", options, info, source_layer)

    def render_smlm_movie(self):
        prepared = self._render_inputs()
        if prepared is None:
            return
        options, info, source_layer = prepared
        frames = self._render_frames()
        if frames is None or frames.size == 0:
            self.log("The localizations have no frame column, so there is nothing to make a movie over")
            return

        grouping = self.render_grouping_box.currentData()
        per_group = self.render_frames_per_box.value()
        step = self.render_step_box.value()
        # Where the movie begins. Localizations before it fall outside every
        # group and are dropped by the engine's own frame lookup, so a
        # cumulative movie accumulates from here rather than from whatever the
        # earliest surviving localization happens to be.
        last_frame = int(frames.max())
        start_frame = max(int(frames.min()), self.render_start_frame_box.value())
        if start_frame > last_frame:
            self.log(
                f"Nothing to render: the movie is set to start at frame "
                f"{start_frame}, after the last localization at {last_frame}."
            )
            return
        options["frame_range"] = (start_frame, last_frame)
        n_movie_frames = smlm_render.group_count(
            start_frame, last_frame, per_group, grouping, step)
        if not self._render_size_is_sane(
                options["shape"], options["oversampling"], n_movie_frames):
            return
        options["gpu"], why = smlm_render.choose_backend(
            options["shape"], options["oversampling"], self.render_gpu_box.isChecked()
        )
        options.update({
            "frames": frames, "frames_per_group": per_group,
            "grouping": grouping, "step": step,
        })
        stride = step if grouping == "sliding" else per_group
        info.update({
            "kind": "movie",
            "backend": "gpu" if options["gpu"] else "cpu",
            "frames_per_group": per_group,
            "grouping": grouping,
            "grouping_label": smlm_render.GROUPINGS[grouping],
            "window_step_frames": step if grouping == "sliding" else None,
            "n_movie_frames": n_movie_frames,
            "first_raw_frame": start_frame,
            "last_raw_frame": last_frame,
            "raw_frames_per_movie_frame": stride,
            "frame_interval_s": self._frame_interval_s() * stride,
        })
        self.render_backend_label.setText(why)
        self.log(
            f"Rendering a {n_movie_frames}-frame movie ({info['grouping_label']}, "
            f"{per_group} raw frames per frame) - {why}..."
        )
        self._start_render("movie", options, info, source_layer)

    def _start_render(self, kind, options, info, source_layer):
        self._set_render_busy(True)
        self.render_progress.setVisible(True)
        self.render_progress.setValue(0)
        self._arm_cancel(self._render_cancel, self.render_cancel_button)
        started = time.perf_counter()

        worker = _render_worker(kind, options, self._render_cancel)
        worker.yielded.connect(lambda fraction: self.render_progress.setValue(int(fraction * 100)))
        worker.returned.connect(
            lambda result: self._on_render_finished(result, kind, info, source_layer, options, started)
        )
        worker.errored.connect(lambda exc: self.log(f"Rendering failed: {exc}"))
        worker.finished.connect(self._on_render_worker_finished)
        self._render_worker_ref = worker
        worker.start()

    def _set_render_busy(self, busy):
        self.render_image_button.setEnabled(not busy and self._render_extent_px is not None)
        self.render_movie_button.setEnabled(
            not busy and self._render_extent_px is not None
            and getattr(self, "_render_frame_range", None) is not None
        )

    def _on_render_worker_finished(self):
        self._set_render_busy(False)
        self.render_cancel_button.setEnabled(False)
        self.render_progress.setVisible(False)
        self._render_worker_ref = None
        self._session_advance()

    def _on_render_finished(self, result, kind, info, source_layer, options, started):
        if result is CANCELLED:
            self.log("Rendering cancelled")
            return
        if isinstance(result, _RenderFailure):
            self.log(f"Rendering failed: {result.error}")
            return
        result, backend = result
        info = dict(info)
        if backend != info["backend"]:
            self.log("The GPU render failed part way - it was finished on the CPU instead")
            info["backend"] = backend
        info["render_seconds"] = round(time.perf_counter() - started, 3)
        info["output_shape"] = [int(v) for v in result.shape]
        info["total_signal"] = float(result.sum())

        # A new render is a new run, so its outputs go somewhere new rather than
        # joining the previous one's folder.
        self._render_save_folder = None
        if kind == "image":
            self._render_image, self._render_image_info = result, info
            self.render_save_image_button.setEnabled(True)
        else:
            self._render_movie, self._render_movie_info = result, info
            self.render_save_movie_button.setEnabled(True)
            # A new movie has a new length, so the save range follows it rather
            # than keeping bounds that belonged to the previous one.
            self._sync_movie_save_range()
        self._update_save_tab()

        if self.render_add_layer_box.isChecked():
            self._add_render_layer(kind, result, info, source_layer, options)
        self.log(
            f"Rendered {'x'.join(str(v) for v in result.shape)} in "
            f"{info['render_seconds']:.2f} s on the {info['backend'].upper()}"
        )

    # ------------------------------------------------------------------
    # Rendering one population at a time, into a layer of its own
    # ------------------------------------------------------------------
    def _render_layer_name(self, kind):
        """Where this render lands. A name per selection, so they accumulate - and
        one more with the immobile trajectories merged, beside the unmerged one."""
        base = self.render_layer_name_edit.text().strip() or RENDER_LAYER_NAME
        if (getattr(self, "merge_box", None) is not None and self.merge_box.isChecked()
                and not base.endswith(MERGED_RENDER_SUFFIX)):
            base += MERGED_RENDER_SUFFIX
        return base if kind == "image" else f"{base}_movie"

    def _render_recipe(self, kind, name):
        """How to make this render again: its render settings, the selection it
        was built from (dynamics filters, population, merging), and what it is."""
        out = {}
        for path, attr in SETTINGS_SPEC:
            if path[0] != "smlm_rendering" and not is_filter_setting(path):
                continue
            widget = getattr(self, attr, None)
            if widget is None:
                continue
            value = widget_value(widget)
            if value is not None:
                _put(out, path, value)
        _put(out, ("smlm_rendering", "population"), self._population_class)
        out["render_kind"] = kind
        out["layer_name"] = name
        return json.loads(json.dumps(out, default=float))

    def _render_population_label(self):
        """What this render was built from, recorded with the layer."""
        active = self._active_metric_filters()
        if not active:
            return "all localizations"
        return "; ".join(f"{METRIC_LABELS[key].split(' (')[0]} {low:g}-{high:g}"
                         for key, low, high in active)

    def _set_render_population(self, which):
        """Point the dynamics filter at a named population, and name the layer.

        The presets are the common case and the reason the feature exists: one
        reconstruction of the molecules that stayed put, one of those that
        moved, and one of those the test could not decide about, from a single
        acquisition, in separate layers that blend additively so they can be
        read together or alone. The three classes split every trajectory
        exactly once - see `_trajectory_classes`.
        """
        threshold = self.render_population_p_box.value()
        self._population_class = None if which == "all" else which
        boxes = self._metric_filter_boxes
        if which != "all" and self._classify_method() == "fit":
            # The probabilities decide on their own; a p_static range left on
            # would quietly drop every trajectory the test calls the other way.
            for box in boxes.values():
                box.blockSignals(True)
                box.setChecked(False)
                box.blockSignals(False)
            self.render_layer_name_edit.setText(f"{RENDER_LAYER_NAME}_{which}")
            self._apply_track_filter()
            self._update_render_population_label()
            return
        for key, box in boxes.items():
            # "dmin" survives: it is not a competing statement about how the
            # molecule behaved but about what its trajectory was capable of
            # measuring, and it is the natural companion to the immobile
            # preset - p > alpha alone puts every trajectory too short to
            # detect anything into the immobile pile.
            if key not in ("pstatic", "dmin"):
                box.blockSignals(True)
                box.setChecked(False)
                box.blockSignals(False)

        pstatic = boxes.get("pstatic")
        if which == "all":
            # "All" means all: the detection-floor qualifier goes too, or the
            # button would not do what it says.
            for key in ("pstatic", "dmin"):
                box = boxes.get(key)
                if box is not None:
                    box.blockSignals(True)
                    box.setChecked(False)
                    box.blockSignals(False)
            self.render_layer_name_edit.setText(RENDER_LAYER_NAME)
        else:
            # Immobile and undetermined are both static by the test; which of
            # the two is decided by the class, on top of this range.
            if which in ("immobile", "undetermined"):
                low, high = threshold, 1.0
            else:
                low, high = 0.0, threshold
            self.pstatic_min_box.setValue(low)
            self.pstatic_max_box.setValue(high)
            pstatic.blockSignals(True)
            pstatic.setChecked(True)
            pstatic.blockSignals(False)
            self.render_layer_name_edit.setText(f"{RENDER_LAYER_NAME}_{which}")

        self._apply_track_filter()
        self._update_render_population_label()
        if which != "all" and not (self._track_pstatic_cache or {}):
            self.log("No immobility test results yet - link trajectories, then "
                     "the test runs with the rest of the fit-free metrics.")

    def _update_render_population_label(self):
        """Say what the next render will be built from, without leaving the tab."""
        if not hasattr(self, "render_population_label"):
            return
        summary = self._track_filter_summary()
        name = self._render_layer_name("image")
        self.render_population_label.setText(
            f"{summary}  →  layer '{name}'")
        self._update_population_counts()
        self._update_merge_label()

    def _build_merge_group(self):
        """One localization per molecule that demonstrably stood still."""
        group = QGroupBox("Merge immobile trajectories")
        form = QFormLayout(group)
        intro = QLabel(
            "A molecule that never moved, seen N times, was measured N times at "
            "one place: merged, it is one localization about sqrt(N) times more "
            "precise, carrying all N localizations' signal. Only the Immobile "
            "class above is merged - trajectories that could have shown motion "
            "and did not."
        )
        intro.setWordWrap(True)
        intro.setProperty("role", "note")
        form.addRow(intro)

        self.merge_box = QCheckBox("Merge each immobile trajectory into one localization")
        self.merge_box.setToolTip(
            "Renders, and the merged table in an export, then hold one "
            "localization per immobile trajectory: at the precision-weighted "
            "mean position, drawn at the combined precision sqrt(1/sum 1/sigma²) "
            "- widened by the square root of the trajectory's reduced chi-square "
            "whenever it scattered more than its precisions allow - with the "
            "photons summed and the frame it first appeared in.\n\n"
            "In a count render it weighs as the N localizations it replaces, so "
            "the structure keeps its brightness and gains its sharpness. The "
            "points on the image are left as they were fitted."
        )
        form.addRow("", self.merge_box)

        self.merge_label = QLabel("-")
        self.merge_label.setWordWrap(True)
        self.merge_label.setProperty("role", "note")
        form.addRow("", self.merge_label)

        merge_row = QHBoxLayout()
        self.merge_layers_button = QPushButton("Show as layers")
        self.merge_layers_button.setProperty("secondary", True)
        self.merge_layers_button.setToolTip(
            "Two point layers, to compare before and after: every localization of "
            "the trajectories being merged, and the one localization each becomes - "
            "sized by its merged precision, coloured by how many it merged. A "
            "render made while merging is on also goes to a layer of its own "
            f"(name + '{MERGED_RENDER_SUFFIX}'), beside the unmerged one.")
        self.merge_layers_button.clicked.connect(lambda: self.show_merged_layers())
        self.merge_table_button = QPushButton("Merged table...")
        self.merge_table_button.setProperty("secondary", True)
        self.merge_table_button.setToolTip(
            "The merged molecules, one row each, in a table of their own: position, "
            "merged precision, photons, how many localizations, first and last frame.")
        self.merge_table_button.clicked.connect(lambda: self.show_merged_table())
        merge_row.addWidget(self.merge_layers_button)
        merge_row.addWidget(self.merge_table_button)
        merge_row.addStretch(1)
        form.addRow("", merge_row)
        self.merge_figure = Figure(figsize=(5, 3.4))
        self.merge_canvas = FigureCanvas(self.merge_figure)
        self._plot_canvases.append(self.merge_canvas)
        self.merge_canvas.setMinimumHeight(260)
        form.addRow("", self.merge_canvas)
        merge_tools = QHBoxLayout()
        merge_tools.addStretch(1)
        merge_tools.addWidget(self._png_button(
            lambda fig, opts: self._draw_merge_distributions(fig, opts), lambda: "merged_molecules"))
        form.addRow("", merge_tools)
        self._merge_plot_key = None

        self.merge_box.toggled.connect(lambda _c: self._on_merge_toggled())
        for box in (self.immobile_dmax_box, self.immobile_min_points_box,
                    self.render_population_p_box, self.class_probability_box):
            box.valueChanged.connect(lambda _v: self._on_immobile_definition_changed())
        self.classify_method_box.currentIndexChanged.connect(
            lambda _i: self._on_immobile_definition_changed())
        self.class_soft_box.toggled.connect(lambda _c: self._on_immobile_definition_changed())
        self._sync_classify_controls()
        return group

    def _on_merge_toggled(self):
        self._merge_cache = None
        self._update_merge_label()
        self._update_render_info()
        self._update_render_population_label()

    # --- the merged molecules, to look at -------------------------------------
    def _merged_molecules(self):
        """(one row per merged molecule, the localizations they replaced), or
        (None, None) when nothing is merged."""
        table = self._render_table()
        if table is None or MERGED_COUNT_COLUMN not in table.columns:
            return None, None
        merged = table[table[MERGED_COUNT_COLUMN].to_numpy() > 1].reset_index(drop=True)
        if merged.empty:
            return None, None
        shown, particles = self._displayed_with_particles()
        if shown is None or particles is None:
            return merged, None
        inside = np.isin(particles, merged["particle"].to_numpy())
        before = shown[inside].copy()
        before["particle"] = particles[inside]
        return merged, before.reset_index(drop=True)

    def show_merged_layers(self):
        """The immobile localizations before merging, and the molecules after, as layers."""
        merged, before = self._merged_molecules()
        if merged is None:
            self.log("Nothing is merged - tick 'Merge each immobile trajectory', with "
                     "trajectories linked and tested")
            return
        pixel = max(self.pixel_size_box.value(), 1e-9)
        x_col, y_col = self._resolve_column("x"), self._resolve_column("y")
        sigma_col = self.column_map.get("uncertainty")
        photon_col = self._resolve_column("intensity")
        for name in (MERGED_BEFORE_LAYER_NAME, MERGED_AFTER_LAYER_NAME):
            self._remove_layer(name)
        if before is not None and not before.empty:
            size = (2.0 * before[sigma_col].to_numpy(float) / pixel
                    if sigma_col in before.columns else np.full(len(before), 0.3))
            self.viewer.add_points(
                np.column_stack([before[y_col].to_numpy(float) / pixel,
                                 before[x_col].to_numpy(float) / pixel]),
                name=MERGED_BEFORE_LAYER_NAME, size=np.clip(size, 0.05, 5.0),
                face_color="#ff8c00", border_width=0, opacity=0.5,
                features={"particle": before["particle"].to_numpy()},
                **self._placed({}, 2))
        features = {"n_merged": merged[MERGED_COUNT_COLUMN].to_numpy(dtype=float),
                    "particle": merged["particle"].to_numpy()}
        if sigma_col in merged.columns:
            features["precision_nm"] = merged[sigma_col].to_numpy(float)
        if photon_col and photon_col in merged.columns:
            features["photons"] = merged[photon_col].to_numpy(float)
        size = (2.0 * features["precision_nm"] / pixel if "precision_nm" in features
                else np.full(len(merged), 0.2))
        self.viewer.add_points(
            np.column_stack([merged[y_col].to_numpy(float) / pixel,
                             merged[x_col].to_numpy(float) / pixel]),
            name=MERGED_AFTER_LAYER_NAME, size=np.clip(size, 0.03, 5.0),
            features=features, face_color="n_merged", face_colormap="plasma",
            border_color="white", border_width=0.1, **self._placed({}, 2))
        self._apply_viewer_scale()
        self.log(f"Layers '{MERGED_BEFORE_LAYER_NAME}' ({0 if before is None else len(before)} "
                 f"localizations) and '{MERGED_AFTER_LAYER_NAME}' ({len(merged)} molecules): "
                 "toggle them to compare; a render made now goes to "
                 f"'{self._render_layer_name('image')}'.")

    def show_merged_table(self):
        merged, _before = self._merged_molecules()
        if merged is None:
            self.log("Nothing is merged - tick 'Merge each immobile trajectory', with "
                     "trajectories linked and tested")
            return
        if getattr(self, "merged_table_dialog", None) is None:
            self.merged_table_dialog = QDialog(self)
            self.merged_table_dialog.setWindowTitle("Merged immobile molecules")
            self.merged_table_dialog.resize(900, 500)
            layout = QVBoxLayout(self.merged_table_dialog)
            self.merged_table_label = QLabel("")
            layout.addWidget(self.merged_table_label)
            self.merged_table_model = PandasTableModel()
            view = QTableView()
            view.setModel(self.merged_table_model)
            layout.addWidget(view)
        first = [c for c in ("particle", MERGED_COUNT_COLUMN, self._resolve_column("x"),
                             self._resolve_column("y"), self.column_map.get("uncertainty"),
                             self._resolve_column("intensity"), self._resolve_column("frame"),
                             "frame_last") if c and c in merged.columns]
        ordered = merged[first + [c for c in merged.columns if c not in first]]
        self.merged_table_model.set_dataframe(ordered)
        self.merged_table_label.setText(
            f"{len(ordered)} molecules, merged from {int(ordered[MERGED_COUNT_COLUMN].sum())} "
            "localizations - one row each, at the precision-weighted mean position")
        self.merged_table_dialog.show()
        self.merged_table_dialog.raise_()

    def _update_merge_plot(self):
        if not hasattr(self, "merge_figure"):
            return
        key = (self.merge_box.isChecked(), id(self._merge_cache[1]) if self._merge_cache else None)
        if key == self._merge_plot_key:
            return
        self._merge_plot_key = key
        self._draw_merge_distributions()

    def _draw_merge_distributions(self, figure=None, opts=None):
        """What the merge did: precision, localizations per molecule, photons, duration."""
        export = figure is not None
        opts = opts or {}
        figure = figure if export else self.merge_figure
        figure.clear()
        merged, before = (self._merged_molecules() if self.merge_box.isChecked()
                          else (None, None))
        if merged is None:
            figure.patch.set_facecolor(PANEL_BG)
            if not export:
                self.merge_canvas.draw_idle()
            return
        axes = figure.subplots(2, 2)
        sigma_col = self.column_map.get("uncertainty")
        photon_col = self._resolve_column("intensity")
        frame_col = self._resolve_column("frame")
        legend_kw = dict(fontsize=plot_font(-3), facecolor=PLOT_BG, edgecolor=PANEL_LINE,
                         labelcolor=INK)
        ax = axes[0, 0]
        if sigma_col and sigma_col in merged.columns and before is not None:
            values = np.concatenate([before[sigma_col].to_numpy(float), merged[sigma_col].to_numpy(float)])
            values = values[np.isfinite(values) & (values > 0)]
            bins = np.linspace(0, float(np.percentile(values, 99.5)) if values.size else 1, 40)
            # as densities: the merged are a tenth as many, and it is the shapes that compare
            ax.hist(before[sigma_col], bins=bins, color=AMBER, alpha=0.6, density=True,
                    label=f"each localization ({len(before)})")
            ax.hist(merged[sigma_col], bins=bins, color=ACCENT, alpha=0.8, density=True,
                    label=f"merged ({len(merged)})")
            ax.legend(**legend_kw)
        ax.set_xlabel("precision (nm)")
        ax.set_ylabel("density")
        ax = axes[0, 1]
        counts = merged[MERGED_COUNT_COLUMN].to_numpy()
        ax.hist(counts, bins=np.arange(1.5, counts.max() + 1.5) if counts.max() < 60
                else 40, color=ACCENT)
        ax.set_xlabel("localizations per molecule")
        ax.set_ylabel("molecules")
        ax = axes[1, 0]
        if photon_col and photon_col in merged.columns and before is not None:
            values = np.concatenate([before[photon_col].to_numpy(float), merged[photon_col].to_numpy(float)])
            values = values[np.isfinite(values) & (values > 0)]
            if values.size:
                bins = np.geomspace(values.min(), values.max(), 40)
                ax.hist(before[photon_col], bins=bins, color=AMBER, alpha=0.6, density=True,
                        label="each localization")
                ax.hist(merged[photon_col], bins=bins, color=ACCENT, alpha=0.8, density=True,
                        label="merged (sum)")
                ax.set_xscale("log")
                ax.legend(**legend_kw)
        ax.set_xlabel("photons")
        ax.set_ylabel("density")
        ax = axes[1, 1]
        if frame_col in merged.columns and "frame_last" in merged.columns:
            span = (merged["frame_last"].to_numpy(float) - merged[frame_col].to_numpy(float) + 1)
            ax.hist(span * self._frame_interval_s(), bins=40, color=ACCENT)
        ax.set_xlabel("seen for (s)")
        ax.set_ylabel("molecules")
        style_axes(figure, axes, title=None)
        if opts.get("title", True):
            figure.suptitle(f"{len(merged)} merged molecules", color=INK, fontsize=plot_font(1))
        figure.tight_layout()
        if not export:
            self.merge_canvas.draw_idle()

    def _sync_classify_controls(self):
        """Only offer what the chosen way of classifying can use."""
        by_fit = self.classify_method_box.currentData() == "fit"
        soft = by_fit and self.class_soft_box.isChecked()
        self.class_probability_box.setEnabled(by_fit and not soft)
        self.class_soft_box.setEnabled(by_fit)
        button = self.population_buttons.get("undetermined")
        if button is not None:
            button.setEnabled(not soft)

    def _on_immobile_definition_changed(self):
        """The line between immobile and undetermined moved."""
        self._sync_classify_controls()
        if self._soft_classes() and self._population_class == "undetermined":
            # soft sorting has no undetermined class to show
            self._population_class = None
        self._merge_cache = None
        if self._population_class is not None:
            # The selection itself changes, which means rebuilding the
            # layers - coalesced, as for any other bound being typed.
            self._invalidate_track_filter()
            self._track_filter_timer.start(250)
        self._update_render_population_label()
        self._update_render_info()

    def _classify_method(self):
        """"fit" when the population fit is chosen and applies, else "test"."""
        if (getattr(self, "classify_method_box", None) is not None
                and self.classify_method_box.currentData() == "fit"
                and self._population_fit_current()):
            return "fit"
        return "test"

    def _soft_classes(self):
        """True when a population render weights by probability instead of sorting."""
        return (self._classify_method() == "fit"
                and getattr(self, "class_soft_box", None) is not None
                and self.class_soft_box.isChecked())

    def _probability_immobile(self):
        """Each fitted trajectory's probability of being immobile, as a dict."""
        fit = self._population_fit
        if fit is None:
            return {}
        cached = fit.get("p_immobile")
        if cached is None:
            cached = dict(zip(fit["result"]["pid"].tolist(),
                              fit["result"]["posterior"][:, 0].tolist()))
            fit["p_immobile"] = cached
        return cached

    def _trajectory_classes(self, method=None):
        """Every trajectory's class: "immobile", "undetermined" or "mobile".

        One split, three answers, each trajectory in exactly one. By the static
        test (`method` "test"):

          mobile       - the static test detected motion (p_static below the
                         population p): positive evidence, wrong one time in
                         1/p.
          immobile     - static by the test, AND the test could have detected
                         motion down to the chosen D (detection floor at or
                         below it), with at least the minimum points.
          undetermined - static by the test only because it could not have
                         failed it: too short or too dim. Also any trajectory
                         the test could not be run on.

        By the population fit ("fit"): immobile or mobile when the trajectory
        is at least the chosen probability likely to be, undetermined between,
        and undetermined too for any trajectory the fit did not see.

        Undetermined is not a polite word for either of the others. Motion is
        easy to prove and immobility hard - a fast molecule shows itself within
        a few points, a static one needs a dozen to be certified - so this pool
        fills with bound molecules seen briefly and slow movers, and folding it
        into mobile would inflate the mobile fraction and drag its D down.
        """
        if self.tracks is None or self.tracks.empty:
            return {}
        method = method or self._classify_method()
        pstatic = self._track_pstatic_cache or {}
        floors = self._track_dmin_cache or {}
        p_split = self.render_population_p_box.value()
        d_max = self.immobile_dmax_box.value()
        n_min = self.immobile_min_points_box.value()
        threshold = self.class_probability_box.value()
        key = (method, id(self.tracks), id(pstatic), id(floors), id(self._population_fit),
               p_split, d_max, n_min, threshold)
        cache = self._class_cache or {}
        if cache.get(method, (None,))[0] == key:
            return cache[method][1]
        lengths = self.tracks.groupby("particle").size()
        classes = {}
        if method == "fit":
            p_immobile = self._probability_immobile()
            for pid in lengths.index:
                p = p_immobile.get(pid)
                if p is not None and p >= threshold:
                    classes[pid] = "immobile"
                elif p is not None and 1.0 - p >= threshold:
                    classes[pid] = "mobile"
                else:
                    classes[pid] = "undetermined"
        else:
            for pid, n_points in lengths.items():
                p = pstatic.get(pid)
                if p is not None and p < p_split:
                    classes[pid] = "mobile"
                elif (p is not None and floors.get(pid, float("inf")) <= d_max
                      and n_points >= n_min):
                    classes[pid] = "immobile"
                else:
                    classes[pid] = "undetermined"
        cache = dict(cache)
        cache[method] = (key, classes)
        self._class_cache = cache
        return classes

    def _class_members(self, which, method=None):
        """Trajectories a class selects. Sorted softly, every fitted trajectory
        belongs to both immobile and mobile, in proportion - its weight says how
        much."""
        if method is None and which in ("immobile", "mobile") and self._soft_classes():
            return set(self._probability_immobile())
        return {pid for pid, cls in self._trajectory_classes(method).items() if cls == which}

    def _merge_ids(self):
        """Trajectories a render merges - certified immobile by the static test -
        or None when off. Never by probability: one position is the strongest
        claim a render makes."""
        if not getattr(self, "merge_box", None) or not self.merge_box.isChecked():
            return None
        if not (self._track_pstatic_cache or {}):
            return set()
        return self._class_members("immobile", method="test")

    def _class_counts(self):
        """Trajectories per class, or None before the immobility test has run."""
        if not (self._track_pstatic_cache or {}):
            return None
        counts = {"immobile": 0, "undetermined": 0, "mobile": 0}
        for cls in self._trajectory_classes().values():
            counts[cls] += 1
        return counts

    def _displayed_with_particles(self):
        """(localizations on screen, the trajectory of each or None)."""
        df = self.df_filtered
        if df is None or df.empty:
            return df, None
        particles = self._localization_particles()
        if particles.size != len(df):
            return self._displayed_localizations(), None
        passing = self._passing_particles()
        if passing is None:
            return df, particles
        if not passing:
            return df.iloc[:0], particles[:0]
        keep = np.isin(particles, np.fromiter(passing, np.int64, len(passing)))
        return df[keep], particles[keep]

    def _render_table(self):
        """The localizations a render is built from.

        Those on screen, dynamics filter included - with every immobile
        trajectory merged into one row when merging is on.
        """
        ids = self._merge_ids()
        if not ids:
            return self._displayed_localizations()
        key = (id(self.df_filtered), id(self.tracks), id(self._track_pstatic_cache),
               id(self._track_dmin_cache), id(self._passing_particles()),
               id(self._loc_particle_cache), self.render_population_p_box.value(),
               self.immobile_dmax_box.value(), self.immobile_min_points_box.value())
        if self._merge_cache is not None and self._merge_cache[0] == key:
            return self._merge_cache[1]
        shown, particles = self._displayed_with_particles()
        if shown is None or shown.empty or particles is None:
            return shown
        merged = merge_trajectories(
            shown, particles, ids,
            x_col=self._resolve_column("x"), y_col=self._resolve_column("y"),
            frame_col=self._resolve_column("frame"),
            sigma_col=self.column_map.get("uncertainty"),
            sum_columns=(self._resolve_column("intensity"),),
            fallback_sigma=self.immobility_sigma_box.value())
        self._merge_cache = (key, merged)
        return merged

    def _merge_summary(self):
        """(trajectories merged, localizations they held) in the next render."""
        table = self._render_table()
        if table is None or MERGED_COUNT_COLUMN not in table.columns:
            return 0, 0
        merged = table[MERGED_COUNT_COLUMN].to_numpy() > 1
        return int(merged.sum()), int(table[MERGED_COUNT_COLUMN].to_numpy()[merged].sum())

    def _points_needed_for_floor(self, d_max_um2):
        """(trajectory length at which the median precision reaches the floor, that
        precision in nm) - the first None if no length does.

        The median of the uncertainty column rather than of the trajectory
        points: the same number to within a few percent, and a column median
        costs nothing where the join onto every trajectory point does not.
        """
        calibration = max(self.immobility_calibration_box.value(), 1e-6)
        column = self.column_map.get("uncertainty")
        df = self.df_filtered
        if column and df is not None and column in df.columns and len(df):
            sigma_nm = float(np.nanmedian(df[column].to_numpy(dtype=float))) * calibration
        else:
            sigma_nm = self.immobility_sigma_box.value() * calibration
        interval = self._frame_interval_s()
        alpha = self.immobility_alpha_box.value()
        if not np.isfinite(sigma_nm) or sigma_nm <= 0 or interval <= 0:
            return None, None
        for n in range(3, 5001):
            if detectable_diffusion(n, sigma_nm, interval, alpha) / 1e6 <= d_max_um2:
                return n, sigma_nm
        return None, sigma_nm

    def _update_population_counts(self):
        """How the trajectories split, and what the line costs in points."""
        if not hasattr(self, "population_counts_label"):
            return
        lines = []
        has_tracks = self.tracks is not None and not self.tracks.empty
        if has_tracks:
            needed, sigma_nm = self._points_needed_for_floor(self.immobile_dmax_box.value())
            if sigma_nm:
                lines.append(
                    f"At the median precision ({sigma_nm:.0f} nm) and this frame rate, "
                    + (f"a static trajectory needs {needed} points to count as immobile."
                       if needed else "no trajectory length reaches that floor."))
        if (has_tracks and self.classify_method_box.currentData() == "fit"
                and not self._population_fit_current()):
            lines.append("The population fit is " + ("stale" if self._population_fit
                         else "not run yet") + " (Track tab) - classifying by the "
                         "static test meanwhile.")
        if not has_tracks:
            lines.append("Link trajectories to split them.")
        elif not (self._track_pstatic_cache or {}):
            lines.append("No immobility test results yet - they come with the "
                         "trajectory metrics, and need a localization precision.")
        elif self._soft_classes():
            p = np.array(list(self._probability_immobile().values()))
            lines.append(f"Weighted by probability: {p.sum():.0f} immobile and "
                         f"{(1 - p).sum():.0f} mobile trajectories expected, of {len(p)}.")
        else:
            counts = self._class_counts()
            total = max(sum(counts.values()), 1)
            if self._classify_method() == "fit":
                lines.append(f"By the population fit, at P >= "
                             f"{self.class_probability_box.value():g}:")
            lines.append("  ·  ".join(
                f"{name.capitalize()} {counts[name]} ({100.0 * counts[name] / total:.0f}%)"
                for name in ("immobile", "undetermined", "mobile")) + " trajectories.")
        self.population_counts_label.setText("\n".join(lines))

    def _update_merge_label(self):
        if not hasattr(self, "merge_label"):
            return
        if not self.merge_box.isChecked():
            self.merge_label.setText("Off: every localization is rendered as fitted.")
        elif self.tracks is None or self.tracks.empty:
            self.merge_label.setText("Link trajectories first - merging works on them.")
        elif not (self._track_pstatic_cache or {}):
            self.merge_label.setText("No immobility test results yet.")
        else:
            n_traj, n_locs = self._merge_summary()
            self.merge_label.setText(
                f"{len(self._merge_ids())} immobile trajectories; the next render "
                f"merges the {n_traj} on screen ({n_locs} localizations into "
                f"{n_traj}). Undetermined and mobile ones stay as fitted.")
        self._update_merge_plot()

    # --- the population fit -------------------------------------------------
    def _population_exposure_fraction(self):
        """How much of each frame interval the camera was exposing, for motion blur.

        From the frame clock's record of the exposure when there is one; a
        camera streaming frames back to back exposes nearly all of it, which is
        the assumption otherwise. A time-binned frame is N exposures with N-1
        readouts between them.
        """
        clock = self._frame_clock
        fraction = 1.0
        if clock is not None and clock.slope > 0:
            exposure = clock.meta.get("exposure_span_s")
            if exposure:
                fraction = min(float(exposure) / clock.slope, 1.0)
        factor = max(1, int(self._time_bin_applied))
        return 1.0 - (1.0 - fraction) / factor

    def _population_fit_fingerprint(self):
        """What a fit depends on. The trajectories' identity stands for the
        localizations, filters and drift correction behind them: a change to
        any of those relinks, and replaces the trajectory table."""
        return (id(self.tracks), round(self._frame_interval_s(), 9),
                round(self.pixel_size_box.value(), 6),
                round(self.immobility_calibration_box.value(), 6),
                round(self.immobility_sigma_box.value(), 6),
                round(self.immobile_dmax_box.value(), 9),
                round(self._population_exposure_fraction(), 6))

    def _population_fit_current(self):
        fit = self._population_fit
        return fit is not None and fit["fingerprint"] == self._population_fit_fingerprint()

    def fit_populations(self):
        if self.tracks is None or self.tracks.empty:
            self.log("Link trajectories first - the populations are fitted to them")
            return
        label, sigma_px, _measured = self._sigma_source()
        if sigma_px is None:
            self.log("No localization precision for the trajectories - cannot fit")
            return
        pixel = max(self.pixel_size_box.value(), 1e-9)
        tracks = self._tracks_for_metrics()
        interval = self._frame_interval_s()
        fraction = self._population_exposure_fraction()
        choice = self.population_mobile_box.currentData()
        run = {"fingerprint": self._population_fit_fingerprint(),
               "precision": label, "exposure_fraction": fraction,
               "mobile_choice": choice}
        self.log(f"Fitting populations to {tracks['particle'].nunique()} trajectories "
                 f"(precision {label}, frames {interval * 1000:.2f} ms apart)...")
        self.population_fit_button.setEnabled(False)
        self.population_progress.setVisible(True)
        self.population_progress.setValue(0)
        self._arm_cancel(self._population_cancel, self.population_cancel_button)
        worker = _population_worker(
            tracks["particle"].to_numpy(), tracks["frame"].to_numpy(),
            tracks["x"].to_numpy(dtype=float) * pixel, tracks["y"].to_numpy(dtype=float) * pixel,
            np.asarray(sigma_px, dtype=float) * pixel, interval,
            populations_io.blur_coefficient(fraction), self.immobile_dmax_box.value(),
            "auto" if choice == "auto" else int(choice), self._population_cancel)
        worker.yielded.connect(lambda frac: self.population_progress.setValue(int(frac * 100)))
        worker.returned.connect(lambda result: self._on_population_fit_finished(result, run))
        worker.errored.connect(lambda exc: self.log(f"The population fit failed: {exc}"))
        worker.finished.connect(self._on_population_worker_finished)
        self._population_worker_ref = worker
        worker.start()

    def _on_population_worker_finished(self):
        self.population_fit_button.setEnabled(True)
        self.population_cancel_button.setEnabled(False)
        self.population_progress.setVisible(False)
        self._population_worker_ref = None
        self._session_advance()

    def _on_population_fit_finished(self, result, run):
        if result is CANCELLED:
            self.log("Population fit cancelled - the previous fit, if any, is unchanged")
            return
        fitted, curves = result
        self._population_fit = dict(run, result=fitted, curves=curves,
                                    fitted_at=datetime.now().isoformat(timespec="seconds"))
        parts = [f"{label} D = {d:.3g} µm²/s ({100 * f:.0f}%)"
                 for label, d, f in zip(fitted["labels"], fitted["D"], fitted["fractions"])]
        self.log("Populations: " + ", ".join(parts)
                 + (f"; BIC kept {fitted['n_mobile']} mobile population(s)"
                    if len(fitted["bic"]) > 1 else ""))
        self._class_cache = None
        self._merge_cache = None
        self._invalidate_track_filter()
        if self._population_class is not None:
            self._apply_track_filter()
        self._update_population_fit_status()
        self._draw_population_plot()
        self._update_render_population_label()
        self._update_render_info()

    def _update_population_fit_status(self):
        if not hasattr(self, "population_fit_status"):
            return
        fit = self._population_fit
        if fit is None:
            self.population_fit_status.setText(
                "Link trajectories, then fit." if self.tracks is None or self.tracks.empty
                else "Not fitted yet.")
            return
        result = fit["result"]
        lines = [f"{label.capitalize()}: D = {d:.3g} µm²/s - {100 * f:.0f}% of "
                 f"trajectories, {100 * lf:.0f}% of localizations."
                 for label, d, f, lf in zip(result["labels"], result["D"],
                                            result["fractions"],
                                            result["localization_fractions"])]
        bic = result["bic"]
        if len(bic) > 1:
            lines.append(f"BIC, one mobile population {bic[1]:.0f}, two {bic[2]:.0f}: "
                         f"kept {result['n_mobile']}.")
        # Immobile against mobile, whatever the mobile ones' split into slow and
        # fast: that is the sorting the renders do.
        p_immobile = result["posterior"][:, 0]
        decided = np.maximum(p_immobile, 1.0 - p_immobile) >= self.class_probability_box.value()
        short = result.get("n_points")
        text = (f"{100 * decided.mean():.0f}% of the {result['n_trajectories']} "
                f"trajectories are at least {self.class_probability_box.value():g} "
                "likely to be immobile, or to be mobile")
        if short is not None and (short == 2).any():
            text += f" ({100 * decided[short == 2].mean():.0f}% of the two-point ones)"
        lines.append(text + ".")
        lines.append(f"Precision {fit['precision']}; motion blur for an exposure of "
                     f"{100 * fit['exposure_fraction']:.0f}% of each frame.")
        if not self._population_fit_current():
            lines.append("Stale: the trajectories, their timing or precision, or the "
                         "Immobile limit changed since - fit again. Not used.")
        self.population_fit_status.setText("\n".join(lines))

    def _draw_population_plot(self, figure=None, opts=None):
        """One-frame step lengths, against what each fitted population predicts."""
        export = figure is not None
        opts = opts or {}
        figure = figure if export else self.population_figure
        figure.clear()
        fit = self._population_fit
        if fit is None:
            figure.patch.set_facecolor(PANEL_BG)
            if not export:
                self.population_canvas.draw_idle()
            return
        centres, observed, predicted = fit["curves"]
        result = fit["result"]
        ax = figure.add_subplot(111)
        width = float(centres[1] - centres[0]) if len(centres) > 1 else 1.0
        ax.bar(centres, observed, width=width, color=INK_DIM, alpha=0.35,
               label="observed")
        for k, (label, d) in enumerate(zip(result["labels"], result["D"])):
            ax.plot(centres, predicted[k], color=POPULATION_COLORS[k % len(POPULATION_COLORS)],
                    linewidth=1.4, label=f"{label}, D = {d:.2g}")
        ax.plot(centres, predicted.sum(axis=0), color=INK, linewidth=1.0,
                linestyle="--", label="all")
        ax.set_xlabel("One-frame step length (nm)")
        ax.set_ylabel("Density")
        legend = ax.legend(fontsize=plot_font(-2), loc="upper right",
                           facecolor=PLOT_BG, edgecolor=PANEL_LINE, labelcolor=INK)
        legend.get_frame().set_alpha(0.85)
        title = "Populations" + ("" if self._population_fit_current() else " (stale)")
        style_axes(figure, ax, title=title if opts.get("title", True) else None)
        if not opts.get("grid", True):
            ax.grid(False)
        figure.tight_layout()
        if not export:
            self.population_canvas.draw_idle()

    def _population_fit_metadata(self):
        fit = self._population_fit
        section = {"mobile_populations": self.population_mobile_box.currentData(),
                   "result": None}
        if fit is not None:
            result = fit["result"]
            section["result"] = {
                "labels": list(result["labels"]),
                "D_um2_per_s": [float(v) for v in result["D"]],
                "trajectory_fractions": [float(v) for v in result["fractions"]],
                "localization_fractions": [float(v) for v in result["localization_fractions"]],
                "bic_by_mobile_populations": {str(k): float(v) for k, v in result["bic"].items()},
                "n_trajectories": int(result["n_trajectories"]),
                "precision": fit["precision"],
                "exposure_fraction": float(fit["exposure_fraction"]),
                "immobile_d_limit_um2_per_s": float(result["d_immobile_max"]),
                "fitted_at": fit.get("fitted_at"),
                "current": bool(self._population_fit_current()),
            }
        return section

    def _row_probability(self, table, which):
        """Per row of a render table, the probability its trajectory is `which`."""
        if "particle" in table.columns:
            particles = table["particle"].to_numpy()
        else:
            shown, particles = self._displayed_with_particles()
            if particles is None or len(particles) != len(table):
                return None
        p_immobile = pd.Series(particles).map(self._probability_immobile()).to_numpy(dtype=float)
        p_immobile = np.nan_to_num(p_immobile, nan=0.0 if which == "immobile" else 1.0)
        return p_immobile if which == "immobile" else 1.0 - p_immobile

    def _build_render_population_group(self):
        self.population_buttons = {}
        group = QGroupBox("Which molecules to render")
        group.setToolTip(
            "A reconstruction of a chosen population rather than of everything.\n\n"
            "Each render goes into the layer named below, so rendering the "
            "immobile molecules and then the mobile ones leaves two layers that "
            "blend additively - the structural half and the dynamic half of the "
            "same acquisition, side by side or on top of each other."
        )
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        for label, which, tip in (
            ("Immobile", "immobile",
             "Trajectories that passed the static test AND could have shown "
             "motion down to the D below - immobile with confidence. These are "
             "the ones merging merges."),
            ("Undetermined", "undetermined",
             "Trajectories that passed the static test only because they could "
             "not have failed it: too short or too dim to have shown motion "
             "slower than the D below. Neither immobile nor mobile can claim "
             "them - they are mostly bound molecules seen briefly, and slow "
             "movers."),
            ("Mobile", "mobile",
             "Trajectories whose motion was detected: they moved further than "
             "their own localization error, at the p below."),
            ("All", "all", "Clear the dynamics filter and render everything."),
        ):
            button = QPushButton(label)
            button.setProperty("secondary", True)
            button.setToolTip(tip)
            button.clicked.connect(lambda _c, w=which: self._set_render_population(w))
            row.addWidget(button)
            self.population_buttons[which] = button
        row.addWidget(QLabel("at p ="))
        self.render_population_p_box = QDoubleSpinBox()
        self.render_population_p_box.setRange(0.0, 1.0)
        self.render_population_p_box.setDecimals(4)
        self.render_population_p_box.setValue(0.05)
        self.render_population_p_box.setToolTip(
            "The significance the split is made at. At 0.05, one immobile "
            "molecule in twenty is misfiled as mobile - the price of a test "
            "with a calibrated false-positive rate."
        )
        adaptive_steps(self.render_population_p_box)
        row.addWidget(self.render_population_p_box)
        row.addStretch(1)
        layout.addLayout(row)

        # What "immobile" has to mean before a trajectory is trusted with it:
        # passing a test is not the same as having been able to fail it.
        certainty = QFormLayout()
        self.immobile_dmax_box = QDoubleSpinBox()
        self.immobile_dmax_box.setRange(1e-6, 100.0)
        self.immobile_dmax_box.setDecimals(6)
        self.immobile_dmax_box.setValue(DEFAULT_MERGE_MAX_DETECTABLE_D)
        self.immobile_dmax_box.setSuffix(" µm²/s")
        adaptive_steps(self.immobile_dmax_box)
        self.immobile_dmax_box.setToolTip(
            "A static trajectory counts as immobile only if its test could have "
            "detected motion this slow - its detection floor is at or below it. "
            "The rest of the static ones are undetermined.\n\n"
            "This is where the line is drawn, and it is drawn per trajectory: "
            "the floor falls with the number of points and rises with the "
            "localization error, so a long bright trajectory certifies itself "
            "quickly and a two-point one never does. Any motion an immobile "
            "trajectory could still be hiding is slower than this."
        )
        certainty.addRow("Immobile if it could have seen D down to", self.immobile_dmax_box)
        self.immobile_min_points_box = QSpinBox()
        self.immobile_min_points_box.setRange(2, 100000)
        self.immobile_min_points_box.setValue(DEFAULT_MERGE_MIN_POINTS)
        self.immobile_min_points_box.setSuffix(" points")
        self.immobile_min_points_box.setToolTip(
            "A floor on trajectory length on top of the detection floor. Three "
            "is the fewest the static test has any power at; raise it to be "
            "stricter regardless of precision."
        )
        certainty.addRow("and has at least", self.immobile_min_points_box)

        self.classify_method_box = QComboBox()
        for key, label in CLASSIFY_METHODS.items():
            self.classify_method_box.addItem(label, key)
        self.classify_method_box.setToolTip(
            "The static test decides each trajectory on its own: motion "
            "detected, or immobility certified - and a short trajectory is "
            "neither, so it is undetermined.\n\n"
            "The population fit (Track tab) decides each trajectory against the "
            "whole dataset, and gives it a probability of being immobile. A "
            "small step is evidence for immobility once the fit knows how large "
            "a mobile molecule's steps are, so far fewer trajectories are left "
            "undetermined.\n\n"
            "Merging always uses the static test: one position is the strongest "
            "claim a render makes, and it takes a certified trajectory."
        )
        certainty.addRow("Classify by", self.classify_method_box)
        self.class_probability_box = QDoubleSpinBox()
        self.class_probability_box.setRange(0.5, 0.9999)
        self.class_probability_box.setDecimals(4)
        self.class_probability_box.setSingleStep(0.05)
        self.class_probability_box.setValue(DEFAULT_CLASS_PROBABILITY)
        self.class_probability_box.setToolTip(
            "With the population fit, a trajectory is immobile when it is at "
            "least this likely to be, mobile when it is at least this likely to "
            "be mobile, and undetermined in between. At 0.9, about one in ten "
            "of the trajectories sorted at the threshold is in the wrong image - "
            "fewer above it."
        )
        certainty.addRow("with probability at least", self.class_probability_box)
        self.class_soft_box = QCheckBox("Weight every localization by its probability instead")
        self.class_soft_box.setToolTip(
            "Soft sorting: an Immobile render then holds every fitted "
            "trajectory, each localization weighted by the probability that its "
            "trajectory is immobile - and a Mobile render the complement. "
            "Nothing is thrown away and nothing is counted twice: the two "
            "images add up to the whole, and a trajectory the fit cannot decide "
            "about lands partly in each, in proportion. There is no undetermined "
            "class. Needs a density render (not scatter)."
        )
        certainty.addRow("", self.class_soft_box)
        layout.addLayout(certainty)

        self.population_counts_label = QLabel()
        self.population_counts_label.setWordWrap(True)
        self.population_counts_label.setProperty("role", "note")
        layout.addWidget(self.population_counts_label)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Layer name"))
        self.render_layer_name_edit = QLineEdit(RENDER_LAYER_NAME)
        self.render_layer_name_edit.setToolTip(
            "Renders replace the layer of this name and leave every other one "
            "alone, so changing it before each render is what builds a set."
        )
        self.render_layer_name_edit.textChanged.connect(
            lambda _t: self._update_render_population_label())
        name_row.addWidget(self.render_layer_name_edit, 1)
        layout.addLayout(name_row)

        self.render_population_label = QLabel()
        self.render_population_label.setWordWrap(True)
        self.render_population_label.setProperty("role", "note")
        layout.addWidget(self.render_population_label)

        note = QLabel(
            "Any of the ranges on the Track tab select here too - these three "
            "are shortcuts for the common split. Ticking 'filter' beside "
            "diffusion, path length or straightness works the same way."
        )
        note.setWordWrap(True)
        note.setProperty("role", "note")
        layout.addWidget(note)
        return group

    def _add_render_layer(self, kind, image, info, source_layer, options):
        source_scale, source_translate = (1.0, 1.0), (0.0, 0.0)
        if source_layer is not None:
            scale = tuple(float(v) for v in np.ravel(getattr(source_layer, "scale", ())))
            translate = tuple(float(v) for v in np.ravel(getattr(source_layer, "translate", ())))
            if len(scale) >= 2:
                source_scale = scale[-2:]
            if len(translate) >= 2:
                source_translate = translate[-2:]
            # The origin is already in the localizations' pixels, which is
            # where a drift-corrected canvas's translate puts its first pixel;
            # counting that translate again would place the render twice over.
            oy, ox = self._canvas_origin(source_layer)
            source_translate = (source_translate[0] - oy * source_scale[0],
                                source_translate[1] - ox * source_scale[1])
        scale, translate = smlm_render.layer_transform(
            options["oversampling"], options["origin"], source_scale, source_translate
        )
        name = self._render_layer_name(kind)
        if kind == "movie":
            # One movie frame spans `raw_frames_per_movie_frame` raw frames, so
            # scaling the time axis by it keeps the dims slider meaning the same
            # thing for the render as for the raw stack underneath.
            scale = (float(info["raw_frames_per_movie_frame"]),) + tuple(scale)
            # A group is placed at the raw frame where it *finishes*, not where
            # it starts. Placed at its first frame - which is what a bare
            # `first_raw_frame` does - the reconstruction of frames 0..N-1 is
            # already on screen at frame 0, so the movie shows molecules before
            # the stack underneath has seen them and runs a whole window ahead
            # of the trajectories built from the same data. The offset is the
            # same for every group in all three groupings, so it stays a
            # translation rather than needing a per-frame mapping.
            lag = max(1, int(info.get("frames_per_group", 1) or 1)) - 1
            translate = (float(info["first_raw_frame"] + lag),) + tuple(translate)

        self._remove_layer(name)
        try:
            self.viewer.add_image(
                image, name=name, colormap=self.render_colormap_box.currentText(),
                blending="additive", scale=scale, translate=translate,
                units=self._viewer_units(len(scale)),
                contrast_limits=smlm_render.contrast_limits(image),
                # Marked rather than named, so a render keeps being recognised
                # as one when it is called "immobile" instead of "smlm_render".
                # "additive" blending above is what lets two populations
                # rendered separately be read as one picture.
                metadata={RENDER_LAYER_TAG: True,
                          "dynamics_selection": self._render_population_label(),
                          RENDER_RECIPE_KEY: self._render_recipe(kind, name)},
            )
        except Exception as exc:
            self.log(f"Could not add the render to the viewer: {exc}")
            return

    # --- saving a render ------------------------------------------------
    def _render_save_dir(self):
        """A dated folder for this render session's outputs.

        Renders used to default to whatever folder the loaded table sat in,
        which scattered gigabyte TIFFs across analysis/ and analysis/data/ among
        the tables and plots. They belong in a run folder like everything else.

        The folder is remembered for the session so an image and a movie saved
        from the same render land together, and a fresh render starts a new one.
        """
        folder = getattr(self, "_render_save_folder", None)
        if folder is None or not folder.exists():
            folder = self._make_analysis_folder(self._analysis_base_dir(), "render")
            self._render_save_folder = folder
        return folder

    def _default_render_path(self, kind, info, force_format=None):
        # Named for the layer it is, not for the table it came from: the mode,
        # the oversampling and every other setting are in the metadata written
        # beside it, and putting them in the filename produced things like
        # localizations_filtered_movie_gaussian_global_os10_data.tif.
        base = self._render_layer_name("image").strip() or RENDER_LAYER_NAME
        parts = [base]
        if kind == "movie":
            parts.append("movie")
        box = self.render_movie_format_box if kind == "movie" else self.render_image_format_box
        save_format = force_format or box.currentData()
        # Named only when it is not the usual one for that kind - a still is
        # normally the quantitative render and a movie the light one - so a
        # composite or a data movie cannot overwrite what it came from, without
        # every ordinary file carrying a redundant word.
        if save_format and save_format != ("display" if kind == "movie" else "data"):
            parts.append(save_format)
        return str(self._render_save_dir() / ("_".join(parts) + ".tif"))

    def save_render_image(self):
        self._save_render("image", self._render_image, self._render_image_info)

    def save_render_movie(self):
        self._save_render("movie", self._render_movie, self._render_movie_info)

    def save_composite_image(self):
        """Save the blend directly, without going via the format box."""
        self._save_render("image", self._render_image, self._render_image_info,
                          force_format="composite")

    def _update_save_tab(self):
        """Say what there is to save, and only offer what actually exists."""
        if not hasattr(self, "render_save_status"):
            return
        ready = []
        for label, image, info in (
            ("image", self._render_image, self._render_image_info),
            ("movie", self._render_movie, self._render_movie_info),
        ):
            if image is None:
                continue
            size = " x ".join(str(int(v)) for v in image.shape)
            ready.append(f"{label} {size} ({info['mode_label'].split(' (')[0]})")
        for button in (self.render_save_image_button, self.render_save_composite_image_button):
            button.setEnabled(self._render_image is not None)
        for button in (self.render_save_movie_button,):
            button.setEnabled(self._render_movie is not None)
        for box in (self.movie_first_box, self.movie_last_box, self.movie_stride_box):
            box.setEnabled(self._render_movie is not None)
        self._update_movie_save_label()
        self.render_save_status.setText(
            "Ready to save: " + ", ".join(ready) if ready else "Nothing rendered yet.")

    # --- the crop box ----------------------------------------------------
    def _sync_render_crop_layer(self):
        """Put a resizable rectangle on the image, or take it away again."""
        if not self.render_crop_box.isChecked():
            self._remove_layer(RENDER_CROP_LAYER_NAME)
            self.render_crop_status.setText("-")
            self._update_render_crop_status()
            return
        if RENDER_CROP_LAYER_NAME in self.viewer.layers:
            self._update_render_crop_status()
            return

        shape, origin, _layer = self._render_field_of_view()
        if shape is None:
            self.log("Load localizations before setting a crop box")
            self.render_crop_box.setChecked(False)
            return
        # Start at the middle half of the field, so the box is obviously a crop
        # and both handles are on screen.
        y0 = origin[0] + shape[0] * 0.25
        y1 = origin[0] + shape[0] * 0.75
        x0 = origin[1] + shape[1] * 0.25
        x1 = origin[1] + shape[1] * 0.75
        rect = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])
        try:
            layer = self.viewer.add_shapes(
                [rect], shape_type="rectangle", name=RENDER_CROP_LAYER_NAME,
                edge_color="lime", face_color="transparent", edge_width=2,
                **self._placed({}, 2),
            )
            layer.mode = "select"
            layer.selected_data = {0}
            layer.events.data.connect(lambda event=None: self._update_render_crop_status())
            self._apply_viewer_scale()
        except Exception as exc:
            self.log(f"Could not add the crop box: {exc}")
            self.render_crop_box.setChecked(False)
            return
        self._update_render_crop_status()

    def _render_crop_bounds(self):
        """(y0, x0, y1, x1) of the crop box in camera pixels, or None."""
        if not self.render_crop_box.isChecked():
            return None
        if RENDER_CROP_LAYER_NAME not in self.viewer.layers:
            return None
        data = self.viewer.layers[RENDER_CROP_LAYER_NAME].data
        if len(data) == 0:
            return None
        rect = np.asarray(data[0])[:, -2:]
        return (float(rect[:, 0].min()), float(rect[:, 1].min()),
                float(rect[:, 0].max()), float(rect[:, 1].max()))

    def _update_render_crop_status(self):
        box = self._render_crop_bounds()
        if box is None:
            self.render_crop_status.setText(
                "Whole field of view is saved." if not self.render_crop_box.isChecked()
                else "Drag the green box on the image to choose the region.")
            return
        oversampling = self.render_oversampling_box.value()
        pixel_nm = self.pixel_size_box.value()
        height_px = int(round((box[2] - box[0]) * oversampling))
        width_px = int(round((box[3] - box[1]) * oversampling))
        self.render_crop_status.setText(
            f"Saving {width_px} x {height_px} super-resolved px "
            f"({(box[3] - box[1]) * pixel_nm / 1000:.1f} x "
            f"{(box[2] - box[0]) * pixel_nm / 1000:.1f} um)"
        )
        self._update_scalebar_status()

    # --- what a save actually writes ------------------------------------
    def _overlay_spec(self, x_px, y_px, frames, sigma_nm, color, info):
        """One overlay, described as plain data for the worker to render.

        The overlay goes through the same reconstruction path as the image it
        sits on - same origin, oversampling, frame grouping and backend - so it
        cannot drift relative to it. Sigma is half the requested width, so the
        drawn feature is about as wide as the number in the box says.
        """
        pixel_size = max(self.pixel_size_box.value(), 1e-9)
        return {
            "source": "overlay", "color": color,
            "x_px": x_px, "y_px": y_px, "frames": frames,
            "global_sigma_px": max(float(sigma_nm), 1.0) / 2.0 / pixel_size,
        }

    # --- scale bar --------------------------------------------------------
    def _saved_width_nm(self):
        """How wide the saved picture is, in nanometres - the crop if there is one."""
        shape, _origin, _layer = self._render_field_of_view()
        if shape is None:
            return None
        pixel_size = self.pixel_size_box.value()
        box = self._render_crop_bounds()
        width_px = (box[3] - box[1]) if box is not None else shape[1]
        return float(width_px) * pixel_size

    def _on_scalebar_auto_changed(self, *_args):
        self.render_scalebar_length_box.setEnabled(
            not self.render_scalebar_auto_box.isChecked())
        self._update_scalebar_status()

    def _update_scalebar_status(self):
        """Keep the automatic length in step with the view it has to suit."""
        if not hasattr(self, "render_scalebar_status"):
            return
        width_nm = self._saved_width_nm()
        if width_nm is None:
            self.render_scalebar_status.setText("Load localizations to size the scale bar.")
            return
        if self.render_scalebar_auto_box.isChecked():
            length = smlm_render.nice_scale_length(width_nm)
            self.render_scalebar_length_box.blockSignals(True)
            self.render_scalebar_length_box.setValue(length)
            self.render_scalebar_length_box.blockSignals(False)
        else:
            length = self.render_scalebar_length_box.value()

        share = 100.0 * length / width_nm
        note = (f"{smlm_render.format_length(length)} bar across a "
                f"{smlm_render.format_length(width_nm)} view ({share:.0f}% of it)")
        if share > 90:
            note += " - too long to fit, it will be skipped"
        self.render_scalebar_status.setText(note)

    def _scalebar_spec(self):
        """The bar and its label, rasterized here for the same reason the clock is."""
        width_nm = self._saved_width_nm()
        if width_nm is None:
            return None
        self._update_scalebar_status()
        length_nm = self.render_scalebar_length_box.value()
        super_pixel_nm = self.pixel_size_box.value() / self.render_oversampling_box.value()
        length_px = int(round(length_nm / max(super_pixel_nm, 1e-9)))
        if length_px < 2:
            self.log("The scale bar is shorter than a pixel - not drawn")
            return None

        # The bar is sized against the picture, not against a fixed number of
        # pixels, so it looks the same at any oversampling.
        rows = int(round(width_nm / max(super_pixel_nm, 1e-9)))
        thickness = max(2, int(round(rows * 0.006)))
        label_height = max(8, int(round(rows * 0.03)))
        atlas = smlm_render.glyph_atlas(label_height)
        label = smlm_render.compose_text(atlas, smlm_render.format_length(length_nm))
        return {
            "mask": smlm_render.scale_bar_mask(length_px, thickness, label),
            "color": self.render_scalebar_color_box.currentText(),
            "position": self.render_scalebar_position_box.currentText(),
            "length_nm": length_nm,
            "length_px": length_px,
        }

    def _timestamp_spec(self, info, is_movie):
        """The labels to burn in, already rasterized.

        The glyphs are drawn here, on the GUI thread, and the worker only
        assembles and blits them: matplotlib is busy drawing this window's own
        figures, and rasterizing text from two threads at once is asking for
        trouble. One pass over the alphabet covers a movie of any length.
        """
        interval = self._frame_interval_s()
        stride = float(info.get("raw_frames_per_movie_frame", 1) or 1)
        if is_movie:
            n_frames = int(info.get("n_movie_frames", 1) or 1)
            # The time a group's window *closes*, matching where the viewer puts
            # it on the slider. Labelling it with the moment the window opened
            # would date every frame a whole window earlier than the data in it.
            lag = max(1, int(info.get("frames_per_group", 1) or 1)) - 1
            times = [(lag + index * stride) * interval for index in range(n_frames)]
        else:
            first = float(info.get("first_raw_frame", 0) or 0)
            last = float(info.get("last_raw_frame", first) or first)
            times = [(last - first + 1) * interval]
        longest = max(times) if times else 0.0
        return {
            "atlas": smlm_render.glyph_atlas(self.render_timestamp_size_box.value()),
            "labels": [smlm_render.format_time(t, longest) for t in times],
            "color": self.render_timestamp_color_box.currentText(),
            "position": self.render_timestamp_position_box.currentText(),
        }

    def _save_spec(self, kind, info, force_format=None):
        """Everything the worker needs to build the saved array, as plain data.

        Assembled here because it reads Qt widgets and the dataframes, which
        only the GUI thread may touch; the rendering and blending it describes
        can then all happen off it. Returns (spec, extra metadata), or
        (None, None) once the reason has been logged.
        """
        box = self.render_movie_format_box if kind == "movie" else self.render_image_format_box
        save_format = force_format or box.currentData()
        label = RENDER_SAVE_FORMATS[save_format] if force_format else box.currentText()
        extra = {"save_format": save_format, "save_format_label": label}
        rotation = float(self.render_rotate_box.value()) % 360.0
        if rotation:
            extra["rotated_degrees"] = rotation
            extra["rotation_is_exact"] = rotation % 90.0 == 0.0
        is_movie = kind == "movie"
        spec = {
            "format": save_format,
            "is_movie": is_movie,
            "render": {
                "shape": tuple(info["field_of_view_camera_px"]),
                "origin": tuple(info["origin_camera_px"]),
                "oversampling": info["oversampling"],
                "gpu": info.get("backend") == "gpu",
                "frames_per_group": info.get("frames_per_group"),
                "grouping": info.get("grouping"),
                "step": info.get("window_step_frames"),
                # the acquisition the base movie covers, so an overlay that
                # spans fewer frames still gets the same movie frames
                "frame_range": (info.get("first_raw_frame"), info.get("last_raw_frame")),
            },
            "layers": [],
            "crop": None,
            "timestamp": None,
            "scalebar": None,
            "rotate_degrees": float(self.render_rotate_box.value()),
        }

        crop_box = self._render_crop_bounds()
        if crop_box is not None:
            spec["crop"] = crop_box
            rows, cols = smlm_render.box_to_slices(
                crop_box, shape=spec["render"]["shape"], origin=spec["render"]["origin"],
                oversampling=spec["render"]["oversampling"])
            extra["crop_camera_px"] = [round(v, 3) for v in crop_box]
            extra["crop_output_px"] = [rows.start, cols.start, rows.stop, cols.stop]

        if self.render_timestamp_box.isChecked():
            spec["timestamp"] = self._timestamp_spec(info, is_movie)
            extra["timestamp"] = {
                "height_px": self.render_timestamp_size_box.value(),
                "color": self.render_timestamp_color_box.currentText(),
                "position": self.render_timestamp_position_box.currentText(),
                "frame_interval_s": self._frame_interval_s(),
            }

        if self.render_scalebar_box.isChecked():
            scalebar = self._scalebar_spec()
            spec["scalebar"] = scalebar
            if scalebar is not None:
                extra["scale_bar"] = {
                    "length_nm": scalebar["length_nm"],
                    "length_super_resolved_px": scalebar["length_px"],
                    "automatic": self.render_scalebar_auto_box.isChecked(),
                    "color": scalebar["color"],
                    "position": scalebar["position"],
                }

        if save_format != "composite":
            return spec, extra

        included = []
        if self.render_composite_base_box.isChecked():
            colormap = self.render_colormap_box.currentText()
            spec["layers"].append({"source": "base", "colormap": colormap})
            included.append(f"reconstruction ({colormap})")

        if self.render_composite_locs_box.isChecked():
            x_px, y_px = self._render_positions_px()
            if x_px is None:
                self.log("No localizations to draw into the composite")
            else:
                color = self.render_locs_color_box.currentText()
                spec["layers"].append(self._overlay_spec(
                    x_px, y_px, self._render_frames(),
                    self.render_locs_size_box.value(), color, info))
                included.append(f"{x_px.size} localizations ({color})")

        if self.render_composite_tracks_box.isChecked():
            if self.tracks is None or self.tracks.empty:
                self.log("No trajectories to draw into the composite - link them first")
            else:
                # trackpy works in camera pixels, the units the render grid uses
                x_s, y_s, frames_s = smlm_render.trajectory_samples(
                    self.tracks["x"].to_numpy(dtype=float),
                    self.tracks["y"].to_numpy(dtype=float),
                    self.tracks["frame"].to_numpy(),
                    self.tracks["particle"].to_numpy(),
                    spacing_px=0.5 / info["oversampling"],
                )
                if x_s.size == 0:
                    self.log("The trajectories have no linked segments to draw")
                else:
                    color = self.render_tracks_color_box.currentText()
                    spec["layers"].append(self._overlay_spec(
                        x_s, y_s, frames_s,
                        self.render_tracks_width_box.value(), color, info))
                    included.append(
                        f"{self.tracks['particle'].nunique()} trajectories ({color})")

        if self.render_composite_all_box.isChecked():
            included.extend(self._viewer_layer_specs(spec, info))

        if not spec["layers"]:
            self.log("Nothing to composite - tick at least one layer that has data")
            return None, None
        extra["composite_layers"] = included
        self.render_composite_status.setText("Last composite: " + ", ".join(included))
        return spec, extra

    def _viewer_layer_specs(self, spec, info):
        """Add every other visible layer to the composite, as plain arrays.

        Image layers are sampled onto the render grid with their own colormap
        and contrast; Points layers are splatted in their own colour. Shapes
        layers (the filter box, the crop box) are controls, not data, and are
        left out. The layer data is read here because only the GUI thread may
        touch the viewer - what reaches the worker is numpy.
        """
        # Every layer about to be read is placed in world units below, so make
        # sure they are all in the same world first - one that arrived without
        # going past the inserted event would otherwise be resampled as though
        # its camera pixels were nanometres.
        self._apply_viewer_scale()

        included = []
        already = {RENDER_CROP_LAYER_NAME, ROI_LAYER_NAME}
        already.update(layer.name for layer in self.viewer.layers
                       if is_render_layer(layer))
        if any(layer["source"] == "overlay" for layer in spec["layers"]):
            already.add(POINTS_LAYER_NAME)  # already added as "Localizations"

        for layer in list(self.viewer.layers):
            name = getattr(layer, "name", "")
            if name in already or not getattr(layer, "visible", True):
                continue
            # Back out of world units: the render grid is described in camera
            # pixels, so a layer's placement has to be expressed in those before
            # it can be resampled onto it.
            world_to_px = 1.0 / max(self.pixel_size_box.value(), 1e-9)
            scale = tuple(float(v) * world_to_px
                          for v in np.ravel(getattr(layer, "scale", (1.0, 1.0))))
            translate = tuple(float(v) * world_to_px
                              for v in np.ravel(getattr(layer, "translate", (0.0, 0.0))))
            data = getattr(layer, "data", None)
            if data is None:
                continue

            if isinstance(layer, napari.layers.Image):
                # Frame by frame, as the composite reads it: np.asarray would
                # decode a lazy stack, or shift every frame of a drift-corrected
                # one, before a single plane was needed.
                stack = data if hasattr(data, "ndim") and hasattr(data, "shape") else np.asarray(data)
                limits = tuple(getattr(layer, "contrast_limits", (None, None)) or (None, None))
                spec["layers"].append({
                    "source": "image", "data": stack,
                    "colormap": self._layer_colormap_name(layer),
                    "limits": limits if all(v is not None for v in limits) else None,
                    "scale": scale[-2:] if len(scale) >= 2 else (1.0, 1.0),
                    "translate": translate[-2:] if len(translate) >= 2 else (0.0, 0.0),
                    "has_frames": stack.ndim >= 3,
                })
                included.append(f"{name} (image)")
            elif isinstance(layer, napari.layers.Points):
                points = np.asarray(data, dtype=float)
                if points.size == 0:
                    continue
                frames = points[:, 0] if points.shape[1] >= 3 else None
                spec["layers"].append(self._overlay_spec(
                    points[:, -1], points[:, -2], frames,
                    self.render_locs_size_box.value(),
                    self._layer_color_name(layer), info))
                included.append(f"{name} (points)")
        return included

    @staticmethod
    def _layer_colormap_name(layer):
        colormap = getattr(layer, "colormap", None)
        name = getattr(colormap, "name", colormap)
        return name if isinstance(name, str) and name in matplotlib.colormaps else "gray"

    @staticmethod
    def _layer_color_name(layer):
        """Match a Points layer's outline to one of the overlay colours."""
        for attribute in ("border_color", "edge_color", "face_color"):
            value = np.ravel(np.asarray(getattr(layer, attribute, []), dtype=float))
            if value.size >= 3 and value[:3].max() > 0:
                target = value[:3]
                return min(
                    smlm_render.OVERLAY_COLORS,
                    key=lambda name: float(np.sum(
                        (np.asarray(smlm_render.OVERLAY_COLORS[name]) - target) ** 2)),
                )
        return "cyan"

    def _update_rotate_label(self):
        """Say whether this angle preserves the numbers or only the picture."""
        if not hasattr(self, "render_rotate_label"):
            return
        degrees = float(self.render_rotate_box.value()) % 360.0
        if degrees == 0.0:
            self.render_rotate_label.setText("not rotated")
        elif degrees % 90.0 == 0.0:
            self.render_rotate_label.setText("exact — pixels permuted, counts preserved")
        else:
            self.render_rotate_label.setText(
                "resampled — counts are not conserved, figures only")

    def _movie_save_slice(self, n_frames):
        """(first, last, stride) for saving, clamped to what was rendered."""
        first = int(np.clip(self.movie_first_box.value(), 0, max(n_frames - 1, 0)))
        last = int(np.clip(self.movie_last_box.value(), first, max(n_frames - 1, 0)))
        return first, last, max(1, int(self.movie_stride_box.value()))

    def _sync_movie_save_range(self):
        """Follow the render: a new movie resets the range to all of it."""
        if not hasattr(self, "movie_first_box"):
            return
        n_frames = 0 if self._render_movie is None else int(self._render_movie.shape[0])
        for box in (self.movie_first_box, self.movie_last_box):
            box.blockSignals(True)
            box.setRange(0, max(n_frames - 1, 0))
            box.blockSignals(False)
        self.movie_last_box.blockSignals(True)
        self.movie_last_box.setValue(max(n_frames - 1, 0))
        self.movie_first_box.setValue(0)
        self.movie_last_box.blockSignals(False)
        self._update_movie_save_label()

    def _update_movie_save_label(self):
        """Say how many frames will be written, and how redundant they are.

        A sliding-window reconstruction advances by `window_step_frames` raw
        frames while each output frame covers `frames_per_group` of them, so
        consecutive frames share most of their localizations. Saving every one
        of them writes a great deal of the same picture over and over; the
        overlap is spelled out here because it is what tells you the stride to
        use.
        """
        if not hasattr(self, "movie_save_label"):
            return
        if self._render_movie is None:
            self.movie_save_label.setText("Render a movie first.")
            return
        n_frames = int(self._render_movie.shape[0])
        first, last, stride = self._movie_save_slice(n_frames)
        kept = len(range(first, last + 1, stride))
        per_frame = self._render_movie[0].nbytes if n_frames else 0
        parts = [f"Saving {kept} of {n_frames} frames "
                 f"(~{kept * per_frame / 1e6:.0f} MB of {n_frames * per_frame / 1e6:.0f})."]

        info = self._render_movie_info or {}
        window = int(info.get("frames_per_group") or 0)
        step = int(info.get("window_step_frames") or 0)
        if window and step and step < window:
            overlap = 100.0 * (window - step) / window
            independent = int(np.ceil(window / step))
            parts.append(
                f"Consecutive frames share {overlap:.0f}% of their raw frames "
                f"({window}-frame window advancing {step}); every {independent}"
                f"{'st' if independent == 1 else 'th'} frame is independent.")
        self.movie_save_label.setText(" ".join(parts))

    def _apply_movie_save_range(self, image, info):
        """Cut the rendered movie down to what was asked for, and say so."""
        n_frames = int(image.shape[0])
        first, last, stride = self._movie_save_slice(n_frames)
        if (first, last, stride) == (0, n_frames - 1, 1):
            return image, info
        image = image[first:last + 1:stride]
        info = dict(info)
        info["saved_frame_range"] = [first, last, stride]
        info["n_frames_saved"] = int(image.shape[0])
        # The frames written are `stride` apart, so each one now spans that much
        # more time. A viewer told otherwise plays the clip at the wrong speed.
        if info.get("frame_interval_s"):
            info["frame_interval_s"] = info["frame_interval_s"] * stride
        self.log(f"Saving frames {first}-{last} every {stride}: "
                 f"{image.shape[0]} of {n_frames}")
        return image, info

    def _save_render(self, kind, image, info, force_format=None):
        if image is None or info is None:
            self.log(f"Render the {kind} first")
            return
        if kind == "movie":
            image, info = self._apply_movie_save_range(image, info)
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save rendered {kind}", self._default_render_path(kind, info, force_format),
            "TIFF files (*.tif *.tiff)",
        )
        if not path:
            return
        try:
            # Created here rather than when the dialog opened, so cancelling out
            # of it does not leave an empty run folder behind.
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.log(f"Could not create {Path(path).parent}: {exc}")
            return
        try:
            spec, extra = self._save_spec(kind, info, force_format)
            if spec is None:
                return
            metadata = self._render_metadata({**info, **extra})
        except Exception as exc:
            self.log(f"Could not prepare the {kind} to save: {exc}")
            return

        for button in (self.render_save_image_button, self.render_save_movie_button,
                       self.render_save_composite_image_button):
            button.setEnabled(False)
        self.log(f"Writing the {kind} ({extra['save_format']}) to {Path(path).name}...")
        worker = _save_render_worker(
            Path(path), image, spec, metadata,
            info["super_resolved_pixel_size_nm"], self.render_png_box.isChecked(),
            self.render_colormap_box.currentText(), info.get("frame_interval_s"),
        )
        worker.returned.connect(self._on_render_saved)
        worker.errored.connect(lambda exc: self.log(f"Saving the render failed: {exc}"))
        worker.finished.connect(self._on_render_save_worker_finished)
        self._render_save_worker_ref = worker
        worker.start()

    def _on_render_save_worker_finished(self):
        self._update_save_tab()
        self._render_save_worker_ref = None

    def _on_render_saved(self, written):
        names = ", ".join(Path(p).name for p in written)
        self.log(f"Saved {names} in {Path(written[0]).parent}")

    def _render_metadata(self, info):
        """The full analysis snapshot, with what this particular render did.

        Deliberately the same dict the analysis export writes: a saved render
        then records the camera, detection, fitting and filter settings that
        produced the localizations behind it, not just the render options - so
        the picture can be traced back to the data without hunting for the
        export folder it came from.
        """
        metadata = self._collect_metadata(self.csv_edit.text().strip())
        metadata.setdefault("smlm_rendering", {}).update(info)
        return metadata

    # ------------------------------------------------------------------
    # Filtering (+ per-column histograms)
    # ------------------------------------------------------------------
    def _default_bounds_for(self, column):
        """Where a column's filter starts: yours, for each side you saved a
        default for, and the built-in bound otherwise."""
        lower, upper = self._builtin_bounds_for(column)
        saved = (((self._user_filter_defaults or {}).get("filter_bounds") or {})
                 .get(column))
        if isinstance(saved, dict) and column not in GEOMETRY_COLUMNS:
            if isinstance(saved.get("min"), (int, float)):
                lower = float(saved["min"])
            if isinstance(saved.get("max"), (int, float)):
                upper = float(saved["max"])
        return lower, upper

    def _builtin_bounds_for(self, column):
        col_key = next((k for k, v in self.column_map.items() if v == column), None)
        # Covers sigma_x/sigma_y too, which are not in the column map but need
        # the same scale as sigma itself.
        if col_key == "sigma" or is_sigma_column(column):
            return SIGMA_DEFAULT_BOUNDS_NM
        if col_key == "uncertainty":
            return 0.0, 200.0
        if col_key == "intensity":
            col_values = self.df[column].dropna()
            positive = col_values[col_values > 0]
            obs_max = float(positive.max()) if not positive.empty else 1e5
            return 10.0, min(1e5, max(obs_max, 10.0 * 1.0001))
        col_values = self.df[column].dropna()
        if not col_values.empty:
            return float(col_values.min()), float(col_values.max())
        return 0.0, 1.0

    def _column_priority(self, column):
        col_key = next((k for k, v in self.column_map.items() if v == column), None)
        if col_key in FILTER_PRIORITY_KEYS:
            return FILTER_PRIORITY_KEYS.index(col_key)
        return len(FILTER_PRIORITY_KEYS)

    def _build_filter_tab_contents(self):
        while self.filter_layout.count():
            item = self.filter_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.filter_controls = {}
        self._default_bounds = {}
        self._hist_widgets = {}
        # The canvases these held are being destroyed; drop them before the new
        # ones are appended, or the list grows a dead entry per load.
        self._plot_canvases = [c for c in self._plot_canvases
                               if c is getattr(self, "msd_canvas", None)
                               or c is getattr(self, "loc_counts_canvas", None)
                               or c is getattr(self, "drift_canvas", None)
                               or c is getattr(self, "rcc_canvas", None)
                               or c is getattr(self, "population_canvas", None)
                               or c is getattr(self, "profile_canvas", None)
                               or c is getattr(self, "merge_canvas", None)
                               or c in {s["canvas"] for s in self._metric_hist_widgets.values()}]

        if self.df is None:
            self.filter_layout.addWidget(QLabel("Load data to see filters"), 0, 0)
            return

        x_col = self._resolve_column("x")
        y_col = self._resolve_column("y")
        # The drift columns record what was done to x and y; there is nothing
        # to select by in them, and they come and go with the correction.
        numeric_columns = [c for c in self.df.columns
                           if pd.api.types.is_numeric_dtype(self.df[c])
                           and c not in drift_io.DRIFT_COLUMNS]
        ordered_columns = sorted(numeric_columns, key=self._column_priority)

        grid_row = grid_col = 0
        for column in ordered_columns:
            lower_box = QDoubleSpinBox()
            lower_box.setRange(-1e9, 1e9)
            # Six, not three: a filter column can be anything from a photon
            # count to an uncertainty in micrometres, and three decimals silently
            # rounds the small ones to zero.
            lower_box.setDecimals(FILTER_BOUND_DECIMALS)
            upper_box = QDoubleSpinBox()
            upper_box.setRange(-1e9, 1e9)
            upper_box.setDecimals(FILTER_BOUND_DECIMALS)
            adaptive_steps(lower_box, upper_box)
            default_lower, default_upper = self._default_bounds_for(column)
            # Outwards, so a default derived from the data cannot exclude the
            # extreme it was derived from once the box has rounded it.
            default_lower = bound_to_box_precision(default_lower, FILTER_BOUND_DECIMALS, False)
            default_upper = bound_to_box_precision(default_upper, FILTER_BOUND_DECIMALS, True)
            lower_box.setValue(default_lower)
            upper_box.setValue(default_upper)
            self.filter_controls[column] = (lower_box, upper_box)
            self._default_bounds[column] = (default_lower, default_upper)

            if column in (x_col, y_col):
                # x/y aren't shown here at all - they're controlled entirely
                # via the draggable ROI box on the image. lower_box/upper_box
                # still exist (referenced in filter_controls) and stay in
                # sync with the ROI box, just never added to a visible layout.
                continue

            group = QGroupBox(column)
            vlayout = QVBoxLayout(group)
            row = QHBoxLayout()
            row.addWidget(QLabel("min"))
            row.addWidget(lower_box)
            row.addWidget(QLabel("max"))
            row.addWidget(upper_box)
            vlayout.addLayout(row)
            vlayout.addWidget(self._make_histogram_widget(column))
            # Typing a bound (not just dragging on the plot) should move
            # the shaded bar/lines immediately too.
            lower_box.valueChanged.connect(lambda _v, c=column: self._sync_histogram_lines(c))
            upper_box.valueChanged.connect(lambda _v, c=column: self._sync_histogram_lines(c))

            self.filter_layout.addWidget(group, grid_row, grid_col)
            grid_col += 1
            if grid_col >= 2:
                grid_col = 0
                grid_row += 1

        # These histograms are rebuilt for every new table, so the chosen size
        # has to be re-applied or it silently reverts on the next load.
        self._apply_plot_size()

        # Settings loaded before the data they belong to: now that the controls
        # exist, apply whatever those bounds match.
        if self._pending_filter_bounds:
            pending = self._pending_filter_bounds
            self._pending_filter_bounds = None
            applied, unmatched = self._apply_filter_bounds(pending)
            if applied:
                self.log(f"Applied {applied} saved filter bound(s) to the new data")
                # Restoring the bounds is not enough - the data has to actually
                # be filtered by them, or the values sit in the boxes doing
                # nothing while the full table stays on screen.
                self.apply_filters()
            if unmatched:
                self.log(
                    f"{len(unmatched)} saved filter bound(s) match no column of this table: "
                    + ", ".join(sorted(unmatched))
                    + f" (it has: {', '.join(str(c) for c in self.df.columns)})"
                )

    def apply_filters(self):
        if self.df is None:
            return
        bounds = {}
        for column, (lower_box, upper_box) in self.filter_controls.items():
            lower = lower_box.value()
            upper = upper_box.value()
            bounds[column] = (lower, upper)
        self.df_filtered = apply_numeric_filters(self.df, bounds)
        self._invalidate_track_filter()
        self._update_status_header()
        self.filter_status.setText(f"Showing {len(self.df_filtered)} localizations")
        self.log(f"Filtered to {len(self.df_filtered)} localizations")
        self._invalidate_tracks(reason="filters changed", recall=True)
        self.render_overlay()
        self._refresh_render_tab()
        self._refresh_histogram_bounds()
        self._sync_xy_roi_layer()
        self.data_table_model.set_dataframe(self.df_filtered)
        self.data_table_label.setText(f"{len(self.df_filtered)} rows x {len(self.df_filtered.columns)} columns")

    def reset_filters(self):
        if self.df is None:
            return
        for column, (lower_box, upper_box) in self.filter_controls.items():
            default_lower, default_upper = self._default_bounds.get(
                column, (lower_box.value(), upper_box.value())
            )
            lower_box.setValue(default_lower)
            upper_box.setValue(default_upper)
        self.log("Filters reset to defaults")
        self.apply_filters()

    # --- the filters kept as defaults, or taken from another analysis -------
    def _filter_defaults_path(self):
        return user_config_dir() / FILTER_DEFAULTS_FILENAME

    def _trajectory_filter_values(self):
        """The trajectory filters as they are set, in the metadata layout."""
        out = {}
        for path, attr in SETTINGS_SPEC:
            widget = getattr(self, attr, None)
            if widget is not None and is_filter_setting(path):
                value = widget_value(widget)
                if value is not None:
                    _put(out, path, value)
        return out

    def _current_filter_settings(self):
        """The filters as they stand, as they would be kept as defaults.

        A localization bound is kept only on the sides that differ from the
        built-in bound: one left at the data's own range stays free to follow
        the next dataset's. Columns this table does not have keep whatever
        default they already had.
        """
        out = self._trajectory_filter_values()
        kept = dict((self._user_filter_defaults or {}).get("filter_bounds") or {})
        if self.df is not None and self.filter_controls:
            geometry = set(GEOMETRY_COLUMNS) | {
                self._resolve_column(key) for key in ("x", "y", "frame")}
            for column, (lower_box, upper_box) in self.filter_controls.items():
                kept.pop(column, None)
                if column in geometry:
                    continue
                built_lower, built_upper = self._builtin_bounds_for(column)
                built_lower = bound_to_box_precision(built_lower, FILTER_BOUND_DECIMALS, False)
                built_upper = bound_to_box_precision(built_upper, FILTER_BOUND_DECIMALS, True)
                entry = {}
                if abs(lower_box.value() - built_lower) > 1e-6:
                    entry["min"] = float(lower_box.value())
                if abs(upper_box.value() - built_upper) > 1e-6:
                    entry["max"] = float(upper_box.value())
                if entry:
                    kept[column] = entry
        if kept:
            out["filter_bounds"] = kept
        return out

    def _refresh_filter_default_bounds(self):
        """What 'Reset filters' goes back to, after the defaults changed."""
        for column in self.filter_controls:
            lower, upper = self._default_bounds_for(column)
            self._default_bounds[column] = (
                bound_to_box_precision(lower, FILTER_BOUND_DECIMALS, False),
                bound_to_box_precision(upper, FILTER_BOUND_DECIMALS, True))

    def _update_filter_defaults_label(self):
        if not hasattr(self, "filter_defaults_label"):
            return
        settings = self._user_filter_defaults
        if not settings:
            self.filter_defaults_label.setText(
                "Defaults: built-in. 'Set as default' keeps the filters as they "
                "are now for every table you load, in this session and the next.")
            return
        columns = sorted((settings.get("filter_bounds") or {}))
        shown = ", ".join(columns)
        self.filter_defaults_label.setText(
            f"Defaults: yours, saved {self._user_filter_defaults_saved_at or '?'} - "
            + (f"bounds on {shown}" if columns else "no localization bounds")
            + ", and the trajectory filters.")

    def _load_filter_defaults(self):
        """At start-up: the filters someone saved as theirs, if they did."""
        path = self._filter_defaults_path()
        try:
            settings, saved_at = read_filter_defaults(path)
        except ValueError as exc:
            self.log(f"Your filter defaults ({path}) {exc} - the built-in ones are used.")
            settings, saved_at = None, None
        self._user_filter_defaults = settings
        self._user_filter_defaults_saved_at = saved_at
        if settings:
            for path_, attr in SETTINGS_SPEC:
                widget = getattr(self, attr, None)
                found, value = _dig(settings, path_)
                if widget is None or not found or value is None or not is_filter_setting(path_):
                    continue
                try:
                    set_widget_value(widget, value)
                except Exception:
                    pass
            self.log(f"Filters start from your defaults, saved {saved_at} ({path.name}).")
        self._update_filter_defaults_label()

    def save_filter_defaults(self):
        """Keep the filters as they are now for every later table and session."""
        settings = self._current_filter_settings()
        path = self._filter_defaults_path()
        try:
            saved_at = write_filter_defaults(path, settings)
        except OSError as exc:
            self.log(f"Could not save the filter defaults to {path}: {exc}")
            return None
        self._user_filter_defaults = settings
        self._user_filter_defaults_saved_at = saved_at
        self._refresh_filter_default_bounds()
        self._update_filter_defaults_label()
        columns = sorted(settings.get("filter_bounds") or {})
        self.log("Filters saved as your defaults - "
                 + (f"bounds on {', '.join(columns)}" if columns
                    else "no localization bound differs from the built-in ones")
                 + f", and the trajectory filters ({path})")
        return path

    def forget_filter_defaults(self):
        """Back to the built-in defaults, from the next table and session on."""
        path = self._filter_defaults_path()
        try:
            if path.is_file():
                path.unlink()
        except OSError as exc:
            self.log(f"Could not remove {path}: {exc}")
            return
        self._user_filter_defaults = None
        self._user_filter_defaults_saved_at = None
        self._refresh_filter_default_bounds()
        self._update_filter_defaults_label()
        self.log("Filter defaults are the built-in ones again; the filters on "
                 "screen are left as they are.")

    def load_filters_from(self, path=None):
        """Apply only the filters of another analysis: a session or a metadata.json."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load filters from another analysis", "",
                filter=SETTINGS_FILE_FILTER)
            if not path:
                return None
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            self.log(f"Could not read {Path(path).name}: {exc}")
            return None
        settings = filter_settings_of(settings_of_file(data))
        if not settings:
            self.log(f"{Path(path).name} holds no filters.")
            return None
        applied, _skipped, notes = self.apply_settings(settings, include_instrument=False)
        columns = len(settings.get("filter_bounds") or {})
        self.log(f"Filters taken from {Path(path).name}: {len(applied)} trajectory "
                 f"filter setting(s) and bounds on {columns} localization column(s); "
                 "x, y and frame left as they are.")
        for note in notes:
            self.log(f"  note: {note}")
        return applied

    def _invalidate_tracks(self, reason="", recall=False):
        # Localizations that fall outside the current filters/ROI must not
        # keep showing up as part of "already linked" trajectories: those
        # trajectories were computed from a different (usually larger) set
        # of localizations, so they'd extend past the new bounds. Rather
        # than silently showing stale/inconsistent tracks, clear them and
        # require an explicit re-link - unless these very localizations were
        # linked before (recall=True, once the new set is in place), in which
        # case their trajectories come back from memory.
        if self.tracks is None:
            if recall:
                self._recall_tracks()
            return
        self._remember_tracks()
        self.tracks = None
        self._tracks_key = None
        self._invalidate_track_filter()
        self._track_diffusion_cache = None
        self._track_msd_cache = None
        self._track_distance_cache = None
        self._track_net_cache = None
        self._track_straightness_cache = None
        self._track_duration_cache = None
        self._track_motion_cache = None
        self._track_pstatic_cache = None
        self._track_dmin_cache = None
        self.compute_d_button.setEnabled(False)
        self._remove_layer(TRACKS_LAYER_NAME)
        self._remove_layer(ALL_TRACKS_LAYER_NAME)
        self._clear_metric_histograms()
        self._update_status_header()
        if recall and self._recall_tracks():
            return
        if reason:
            self.log(f'Trajectories cleared ({reason}) - click "Link trajectories" again'
                     " (or go back to settings already linked: they come back from memory).")

    # --- trajectories kept in memory ---------------------------------------
    def _localization_set_key(self):
        """What a link would be given now, or None: the filtered localizations'
        frames and positions - so after the filters, the drift correction and the
        frame shift - and the linking settings. Equal keys link to the same
        trajectories. A sum of row hashes: the order of the rows does not
        matter to the linker, and a million rows hash in a fraction of a second.
        """
        df = self.df_filtered
        cols = [self._resolve_column(k) for k in ("frame", "x", "y")]
        if df is None or df.empty or not all(cols) or not set(cols) <= set(df.columns):
            return None
        rows = pd.util.hash_pandas_object(df[cols], index=False).to_numpy()
        return (len(df), int(rows.sum(dtype=np.uint64)), self._frame_offset(),
                round(self.pixel_size_box.value(), 6), round(self.search_box.value(), 6),
                int(self.memory_box.value()), int(self.min_traj_box.value()))

    def _metric_settings(self):
        """What the fit-free metrics were computed with."""
        return (round(self.pixel_size_box.value(), 6), round(self.fps_box.value(), 6),
                round(self.immobility_alpha_box.value(), 9))

    def _remember_tracks(self):
        """Keep the current trajectories, and all that was computed on them."""
        key = self._tracks_key
        if key is None or self.tracks is None or self.tracks.empty:
            return
        caches = {attr: getattr(self, attr, None) for attr in METRIC_CACHE_ATTR.values()}
        caches["_track_msd_cache"] = self._track_msd_cache
        self._linked_memory[key] = {
            "tracks": self.tracks, "caches": caches, "metric_settings": self._metric_settings(),
            "population_fit": self._population_fit, "source_path": self._tracks_source_path}
        self._linked_memory.move_to_end(key)
        while len(self._linked_memory) > LINKED_MEMORY_SIZE:
            self._linked_memory.popitem(last=False)

    def _recall_tracks(self):
        """Bring back the trajectories linked before from these localizations.
        True if there were any."""
        key = self._localization_set_key()
        kept = self._linked_memory.get(key) if key is not None else None
        if kept is None:
            return False
        self._linked_memory.move_to_end(key)
        self.tracks = kept["tracks"]
        self._tracks_key = key
        self._tracks_source_path = kept["source_path"]
        same_settings = kept["metric_settings"] == self._metric_settings()
        for attr, value in kept["caches"].items():
            setattr(self, attr, value if same_settings else None)
        self._invalidate_track_filter()
        self._class_cache = None
        fit = kept["population_fit"]
        if fit is not None and fit.get("fingerprint") == self._population_fit_fingerprint():
            # the same trajectories, the same settings: that fit is this one's
            self._population_fit = fit
        self.compute_d_button.setEnabled(True)
        n = int(self.tracks["particle"].nunique())
        self.log(f"Trajectories restored from memory: {n} linked earlier from exactly "
                 "these localizations - no relinking needed")
        if not (self._track_pstatic_cache or self._track_distance_cache):
            self._start_fit_free_metrics_worker()
        else:
            for metric in METRIC_CACHE_ATTR:
                self._draw_metric_histogram(metric)
            self._update_immobility_status()
            self._update_population_counts()
            self._update_merge_label()
        self._update_population_fit_status()
        self._draw_population_plot()
        self._update_status_header()
        return True

    def _clear_metric_histograms(self):
        for state in self._metric_hist_widgets.values():
            state["figure"].clear()
            state["lower_line"] = None
            state["upper_line"] = None
            state["span"] = None
            state["canvas"].draw_idle()
        if hasattr(self, "msd_figure"):
            self.msd_figure.clear()
            self.msd_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Linking (background, with progress)
    # ------------------------------------------------------------------
    def link_tracks(self):
        if self.df_filtered is None or self.df_filtered.empty:
            return
        features = self._prepare_features()
        if features.empty:
            return
        tp.quiet()
        search_range_nm = self.search_box.value()
        search_range_px = max(1.0, search_range_nm / max(self.pixel_size_box.value(), 1.0))
        n_frames = int(features["frame"].nunique())
        self.log(f"Linking with search range {search_range_nm:.1f} nm ({search_range_px:.2f} px)")

        self.link_button.setEnabled(False)
        self.link_progress.setVisible(True)
        self.link_progress.setValue(0)
        self._arm_cancel(self._link_cancel, self.link_cancel_button)

        worker = _link_worker(
            features, search_range_px, self.memory_box.value(), n_frames, self._link_cancel
        )
        worker.yielded.connect(lambda frac: self.link_progress.setValue(int(frac * 100)))
        key = self._localization_set_key()
        worker.returned.connect(lambda linked, key=key: self._on_link_finished(linked, key))
        worker.errored.connect(self._on_link_errored)
        worker.finished.connect(self._on_link_worker_finished)
        self._link_worker_ref = worker
        worker.start()

    def _on_link_worker_finished(self):
        self.link_button.setEnabled(True)
        self.link_cancel_button.setEnabled(False)
        self.link_progress.setVisible(False)
        self._link_worker_ref = None
        self._session_advance()

    def _on_link_errored(self, exc):
        self.log(f"Linking failed: {exc}")

    def _on_link_finished(self, linked, key=None):
        if linked is CANCELLED:
            self.log("Linking cancelled - trajectories are unchanged")
            return
        if linked is None or linked.empty:
            self.log("No trajectories linked")
            self.tracks = None
            self.compute_d_button.setEnabled(False)
            return
        traj = tp.filter_stubs(linked, self.min_traj_box.value())
        if traj.empty:
            self.log("All trajectories were shorter than the minimum track length")
            self.tracks = None
            self.compute_d_button.setEnabled(False)
            return
        self.tracks = traj.reset_index(drop=True)
        self._tracks_source_path = None      # linked here, so a session may re-link
        self._tracks_key = key if key is not None else self._localization_set_key()
        self._invalidate_track_filter()
        self._track_diffusion_cache = None
        self._track_msd_cache = None
        self.compute_d_button.setEnabled(True)
        self.log(f"Linked {traj['particle'].nunique()} trajectories")
        self._start_fit_free_metrics_worker()
        self.render_overlay()
        self._update_status_header()

    def _resolve_column(self, key):
        if self.column_map.get(key):
            return self.column_map[key]
        source = self.df_filtered if self.df_filtered is not None else self.df
        if source is None:
            return None
        for candidate in ["x [nm]", "y [nm]", "frame"]:
            if candidate in source.columns:
                return candidate
        return None

    def _prepare_features(self):
        df = self.df_filtered.copy()
        x_col = self._resolve_column("x")
        y_col = self._resolve_column("y")
        frame_col = self._resolve_column("frame")
        if frame_col is None or x_col is None or y_col is None:
            self.log("Missing required columns for tracking")
            return pd.DataFrame()
        pixel_size = self.pixel_size_box.value()
        features = pd.DataFrame(
            {
                "x": df[x_col].astype(float).to_numpy() / pixel_size,
                "y": df[y_col].astype(float).to_numpy() / pixel_size,
                "frame": df[frame_col].astype(int).to_numpy() + self._frame_offset(),
            }
        )
        # Positions with the growth's local stretch taken out ride along, for the
        # metrics: linking itself works in the final geometry like everything else.
        local = getattr(self, "_local_px", None)
        if local is not None:
            aligned = local.reindex(df.index)
            if aligned.notna().all().all():
                features["x_local"] = aligned["x_local"].to_numpy()
                features["y_local"] = aligned["y_local"].to_numpy()
        features = features.sort_values(["frame", "x", "y"]).reset_index(drop=True)
        return features

    def _remove_layer(self, name):
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------
    def _on_playback_changed(self):
        """Hand the requested speed to napari, which owns the play button."""
        settings = _napari_playback_settings()
        if settings is not None:
            try:
                settings.playback_fps = int(self.playback_fps_box.value())
                settings.playback_mode = self.playback_mode_box.currentData()
            except Exception as exc:
                self.log(f"Could not set the playback speed: {exc}")
        self._update_playback_status()

    def _set_playback_to_real_time(self):
        """The nearest whole frame rate to the one the camera acquired at.

        The rounding is why the status line below quotes the pace rather than
        claiming real time: 31.9 fps acquired can only be played at 32.
        """
        self.playback_fps_box.setValue(
            max(1, int(round(float(self.fps_box.value())))))

    def _update_playback_status(self):
        """Say what the chosen speed means against the rate it was acquired at.

        A frame rate on its own says nothing about whether what you are watching
        is sped up or slowed down, which is the only thing a reader of the
        finished movie will want to know.
        """
        if not hasattr(self, "playback_status"):
            return
        playback = int(self.playback_fps_box.value())
        acquired = getattr(self, "fps_box", None)
        acquired = float(acquired.value()) if acquired is not None else 0.0
        if acquired <= 0:
            self.playback_status.setText(f"{playback:d} frames per second")
            return
        ratio = playback / acquired
        if abs(ratio - 1.0) < 0.005:
            pace = "real time"
        elif ratio > 1.0:
            pace = f"{ratio:.3g}x faster than real time"
        else:
            pace = f"{1.0 / ratio:.3g}x slower than real time"
        self.playback_status.setText(
            f"{playback:d} fps shown against {acquired:.4g} fps acquired - {pace}"
        )

    # ------------------------------------------------------------------
    # Physical units in the viewer
    # ------------------------------------------------------------------
    def _reset_view(self):
        """Frame everything that is loaded, the way opening a file should."""
        try:
            self.viewer.reset_view()
        except Exception:
            pass  # a viewer without a camera to reset

    def _viewer_scale(self, ndim):
        """Display scale for a layer whose data is indexed in camera pixels."""
        scale = [1.0] * max(int(ndim), 2)
        scale[-2:] = [self.pixel_size_box.value()] * 2
        return tuple(scale)

    def _viewer_units(self, ndim):
        units = [VIEWER_FRAME_UNIT] * max(int(ndim), 2)
        units[-2:] = [VIEWER_SPATIAL_UNIT] * 2
        return tuple(units)

    def _placed(self, kwargs, ndim):
        """Add the display transform to the kwargs of a layer measured in pixels."""
        kwargs = dict(kwargs)
        kwargs.setdefault("scale", self._viewer_scale(ndim))
        kwargs.setdefault("units", self._viewer_units(ndim))
        return kwargs

    @staticmethod
    def _stretch_layer(layer, factor):
        """Move a layer with the world when the pixel size changes under it."""
        scale = np.array(np.ravel(layer.scale), dtype=float)
        translate = np.array(np.ravel(layer.translate), dtype=float)
        scale[-2:] *= factor
        translate[-2:] *= factor
        layer.scale = scale
        layer.translate = translate

    def _apply_viewer_scale(self):
        """Put the viewer in nanometres and show napari's scale bar.

        The bar is napari's own canvas overlay rather than anything drawn here:
        it sits in the corner of the *viewport*, above every layer, follows pan
        and zoom, and picks its own round length as the zoom changes. All it
        needs is for the world to be measured in something physical, which is
        what this establishes.

        Called whenever a layer is added or the pixel size changes, so the two
        can never disagree - a scale bar sized by a stale pixel size is worse
        than none, because it looks authoritative.
        """
        if not hasattr(self, "pixel_size_box"):
            return  # a layer arrived before the Data tab was built
        pixel_nm = self.pixel_size_box.value()
        previous = getattr(self, "_viewer_pixel_size_nm", None)
        for layer in list(self.viewer.layers):
            meta = getattr(layer, "metadata", None) or {}
            if meta.get(WL_OVERLAY_TAG):
                continue  # placed by its own affine, redone below
            if meta.get(AVERAGE_TAG):
                self._place_average_layer(layer)
                try:
                    layer.units = self._viewer_units(2)
                except Exception:
                    pass
                continue
            try:
                ndim = len(np.ravel(layer.scale))
                if is_render_layer(layer):
                    # Its scale is derived from the layer beneath it, so it is
                    # only carried along - but it still has to be labelled. A
                    # single layer left in pixels makes the units inconsistent
                    # across the viewer, and napari then discards all of them
                    # and the scale bar silently falls back to counting pixels.
                    if previous:
                        self._stretch_layer(layer, pixel_nm / previous)
                else:
                    layer.scale = self._viewer_scale(ndim)
                layer.units = self._viewer_units(ndim)
            except Exception:
                continue  # a layer type this napari will not let us annotate
        self._viewer_pixel_size_nm = pixel_nm
        self._update_wl_overlay()

        bar = getattr(self.viewer, "scale_bar", None)
        # An empty canvas has nothing to measure: typing a pixel size before
        # loading must not put a bar on it. It comes with the first layer.
        if bar is not None and len(self.viewer.layers):
            try:
                # Only what the bar needs to exist and sit out of the way; colour,
                # box and ticks are left to the user's napari preferences.
                bar.visible = True
                bar.position = "bottom_right"
                # napari tiles overlays that share a corner, working outwards from
                # the edge in `order`. The bar takes the edge and the clock stacks
                # directly above it.
                bar.order = 0
            except Exception:
                pass
        self._update_time_overlay()

    def _on_current_frame_changed(self):
        """Everything that has to follow the frame slider, in one place."""
        self._update_time_overlay()
        self._sync_accumulating_tracks()

    def _update_time_overlay(self):
        """Keep a clock on the canvas, just above the scale bar.

        Read off the dims slider rather than tracked here, so it is right
        whether the frame changed by dragging, by the play button, or from
        anything else that moves the slider.
        """
        overlay = getattr(self.viewer, "text_overlay", None)
        if overlay is None or not hasattr(self, "fps_box"):
            return
        frame = self._get_current_frame()
        interval = self._frame_interval_s()
        last = self._last_loaded_frame()
        try:
            overlay.text = "{} (frame {})".format(
                smlm_render.format_time(frame * interval,
                                        (last if last else frame) * interval),
                frame,
            )
            overlay.position = "bottom_right"
            overlay.order = 1  # above the scale bar, which took order 0
            overlay.visible = True
        except Exception:
            pass

    def _last_loaded_frame(self):
        """The final frame index in view, so the clock picks one time format.

        A label that switches between "9.4 s" and "01:12" as the slider moves
        is unreadable; the format is chosen once, from how long the whole
        acquisition runs.
        """
        best = 0
        for layer in list(self.viewer.layers):
            try:
                extent = layer.extent.data
                if extent is not None and len(extent[1]) >= 3:
                    best = max(best, int(extent[1][0]))
            except Exception:
                continue
        if not best and self.df is not None and "frame" in self.df:
            best = int(self.df["frame"].max())
        return best

    # ------------------------------------------------------------------
    # napari layer synchronization
    # ------------------------------------------------------------------
    def render_overlay(self):
        # Rebuilds the points/tracks layers from the current data/style. This
        # is only called on load/filter/link/style-change actions, never on
        # a dims-slider move: Points/Tracks layers carry the full multi-frame
        # data and let napari slice them natively, so moving the slider does
        # no Python-side work and stays smooth even with many localizations
        # or trajectories.
        #
        # No global emptiness check: each layer decides for itself, so that
        # trajectory settings still apply when there are trajectories but no
        # localizations to draw (and vice versa).
        self._sync_points_layer()
        self._sync_tracks_layer()
        self._sync_all_tracks_layer()
        # Whatever brought us here - a link, a new metric, a moved bound - the
        # one line describing the dynamics filter is now out of date.
        self._update_track_filter_label()

    def _sync_points_layer(self):
        x_col = self._resolve_column("x")
        y_col = self._resolve_column("y")
        frame_col = self._resolve_column("frame")

        shown = self._displayed_localizations()
        has_rows = shown is not None and not shown.empty
        if not (self.show_points_box.isChecked() and has_rows and x_col and y_col and frame_col):
            self._remove_layer(POINTS_LAYER_NAME)
            return

        geom_cols = [frame_col, y_col, x_col]
        valid = shown.dropna(subset=geom_cols)
        if valid.empty:
            self._remove_layer(POINTS_LAYER_NAME)
            return

        pixel_size = self.pixel_size_box.value()
        frame_idx = valid[frame_col].astype(int).to_numpy() + self._frame_offset()
        y_px = valid[y_col].astype(float).to_numpy() / pixel_size
        x_px = valid[x_col].astype(float).to_numpy() / pixel_size
        coords = np.column_stack([frame_idx, y_px, x_px])

        prop_cols = [c for c in valid.columns if c not in geom_cols]
        features = valid[prop_cols].reset_index(drop=True) if prop_cols else None

        border_width = self.marker_edge_width_box.value()

        if POINTS_LAYER_NAME in self.viewer.layers:
            layer = self.viewer.layers[POINTS_LAYER_NAME]
            layer.data = coords
            if features is not None:
                layer.features = features
            layer.size = self.marker_size_box.value()
            layer.symbol = self.marker_choice.currentText()
            layer.face_color = "transparent"
            layer.border_color = "cyan"
            layer.border_width = border_width
            layer.border_width_is_relative = True
            layer.visible = True
        else:
            kwargs = dict(
                name=POINTS_LAYER_NAME,
                face_color="transparent",
                border_color="cyan",
                border_width=border_width,
                border_width_is_relative=True,
                size=self.marker_size_box.value(),
                symbol=self.marker_choice.currentText(),
                visible=True,
            )
            if features is not None:
                kwargs["features"] = features
            self.viewer.add_points(
                coords, **self._placed(kwargs, np.asarray(coords).shape[-1]))
        self.viewer.tooltip.visible = True
        self._apply_viewer_scale()

    # ------------------------------------------------------------------
    # Per-trajectory metrics: D (fit-based), distance & duration (fit-free)
    # ------------------------------------------------------------------
    def compute_d(self):
        if self.tracks is None or self.tracks.empty:
            self.log("Link trajectories first")
            return
        max_lagtime = self.max_lagtime_box.value()
        fps = max(self.fps_box.value(), 1e-6)
        mpp = max(self.pixel_size_box.value(), 1e-6) / 1000.0  # nm/px -> um/px

        # Second, independent length filter: a linear MSD fit wants more points
        # than trajectory display or the fit-free metrics do, so D can be
        # restricted to the longer tracks without discarding the short ones.
        min_length = int(self.d_min_length_box.value())
        tracks = filter_tracks_by_length(self._tracks_for_metrics(), min_length)
        n_total = int(self.tracks["particle"].nunique())
        if tracks.empty:
            self.log(
                f"No trajectory has {min_length} or more points "
                f"(longest of {n_total} is shorter) - lower 'Min track length for D'"
            )
            return
        n_kept = int(tracks["particle"].nunique())
        self._d_input_track_count = n_kept
        if n_kept < n_total:
            self.log(f"D: using {n_kept} of {n_total} trajectories with >= {min_length} points")

        self.log("Computing D in the background...")
        self.compute_d_button.setEnabled(False)
        self.compute_d_progress.setVisible(True)
        self.compute_d_progress.setValue(0)
        self._arm_cancel(self._compute_d_cancel, self.compute_d_cancel_button)

        worker = _compute_d_worker(tracks, max_lagtime, fps, mpp, self._compute_d_cancel)
        worker.yielded.connect(lambda frac: self.compute_d_progress.setValue(int(frac * 100)))
        worker.returned.connect(self._on_compute_d_finished)
        worker.errored.connect(lambda exc: self.log(f"D computation failed: {exc}"))
        worker.finished.connect(self._on_compute_d_worker_finished)
        self._compute_d_worker_ref = worker
        worker.start()

    def _on_compute_d_worker_finished(self):
        self.compute_d_button.setEnabled(True)
        self.compute_d_cancel_button.setEnabled(False)
        self.compute_d_progress.setVisible(False)
        self._compute_d_worker_ref = None
        self._session_advance()

    def _on_compute_d_finished(self, result):
        if result is CANCELLED:
            self.log("D computation cancelled - previous results are unchanged")
            return
        d_map, msd_map = result
        self._track_diffusion_cache = d_map
        self._track_msd_cache = msd_map
        self._invalidate_track_filter()
        n_input = getattr(self, "_d_input_track_count", None) or int(self.tracks["particle"].nunique())
        self.log(f"Computed D for {len(d_map)} of {n_input} trajectories")
        # The linking readout can now say how many measured D exceed the cutoff.
        self._update_link_cutoff_label()
        self._set_metric_default_bounds("D", d_map)
        self._set_metric_view_default("D", d_map)
        self._draw_metric_histogram("D")
        self._draw_msd_validation()
        self._update_msd_sigma_label()
        sigmas = [s for s in self._msd_sigma_map().values() if np.isfinite(s)]
        if sigmas:
            self.log(f"MSD intercept implies a localization precision of "
                     f"{np.median(sigmas):.1f} nm (median of {len(sigmas)} trajectories)")
        if self.color_trajectories_box.isChecked():
            self.render_overlay()

    def _sigma_source(self):
        """Where the localization precision for the immobility test comes from.

        Returns (label, per-row array in camera pixels, is_measured). Measured
        per-spot precision is strongly preferred: it is what makes the null
        exact, and a single average σ over a table whose precision varies
        three-fold inflates the false-positive rate from 5% to about 15%.
        """
        calibration = max(self.immobility_calibration_box.value(), 1e-6)
        pixel_size = max(self.pixel_size_box.value(), 1e-9)
        if self.tracks is None or self.tracks.empty:
            return "no trajectories", None, False

        measured = self._track_sigma_nm()
        if measured is not None:
            column = self.column_map.get("uncertainty")
            return (f"per localization, from '{column}'",
                    measured * calibration / pixel_size, True)
        fixed = self.immobility_sigma_box.value()
        return (f"fixed {fixed:.1f} nm (no uncertainty column in this table)",
                np.full(len(self.tracks), fixed * calibration / pixel_size), False)

    def _track_sigma_nm(self):
        """The reported uncertainty of each trajectory point, in nm, or None.

        Matched back from the localization table rather than carried through the
        linker, for the same reason `_localization_particles` is: trajectories
        are as often read back from a previous run's CSV as linked here, and the
        join works for both.
        """
        column = self.column_map.get("uncertainty")
        df = self.df_filtered
        if not column or df is None or df.empty or column not in df.columns:
            return None
        x_col = self._resolve_column("x")
        y_col = self._resolve_column("y")
        frame_col = self._resolve_column("frame")
        if not (x_col and y_col and frame_col):
            return None

        pixel_size = max(self.pixel_size_box.value(), 1e-9)
        known = pd.DataFrame({
            "frame": df[frame_col].to_numpy(np.int64) + self._frame_offset(),
            "x": np.round(df[x_col].to_numpy(float) / pixel_size, LOC_MATCH_DECIMALS),
            "y": np.round(df[y_col].to_numpy(float) / pixel_size, LOC_MATCH_DECIMALS),
            SIGMA_COLUMN: df[column].to_numpy(float),
        }).drop_duplicates(subset=["frame", "x", "y"])
        wanted = pd.DataFrame({
            "frame": self.tracks["frame"].to_numpy(np.int64),
            "x": np.round(self.tracks["x"].to_numpy(float), LOC_MATCH_DECIMALS),
            "y": np.round(self.tracks["y"].to_numpy(float), LOC_MATCH_DECIMALS),
        })
        merged = wanted.merge(known, on=["frame", "x", "y"], how="left", sort=False)
        sigma = merged[SIGMA_COLUMN].to_numpy(float)
        # A join that matched almost nothing means these trajectories do not
        # belong to this table; a fixed precision is the honest fallback.
        return sigma if np.isfinite(sigma).mean() > 0.5 else None

    def _update_immobility_status(self):
        """Say which precision is in use, and whether it looks calibrated."""
        if not hasattr(self, "immobility_status_label"):
            return
        label, _sigma, measured = self._sigma_source()
        lines = [f"Localization precision: {label}."]
        if not measured and self.tracks is not None and not self.tracks.empty:
            lines.append("Add an uncertainty column to the localizations for a "
                         "per-spot precision - it is what makes the test exact.")
        ratios = np.array(list((self._track_motion_cache or {}).values()), float)
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size:
            # The calibration check reads off the *immobile* population, so the
            # useful statistic is the low end rather than the median: on a mixed
            # sample the median is pulled up by molecules that really did move,
            # and reporting that as a precision error would be wrong.
            median = float(np.median(ratios))
            floor = float(np.percentile(ratios, 10))
            lines.append(
                f"Motion ratio over {ratios.size} trajectories: median {median:.2f}, "
                f"10th percentile {floor:.2f}.")
            lines.append(
                "The calibration check is on the immobile end: whichever of these "
                "corresponds to molecules you believe are stationary should read "
                f"1.00. At {floor:.2f} the reported precision would be low by "
                f"{100 * (np.sqrt(max(floor, 1e-9)) - 1):.0f}%, correctable by "
                f"setting the calibration to {np.sqrt(max(floor, 1e-9)):.2f}.")
        floors = np.array([f for f in (self._track_dmin_cache or {}).values()
                           if np.isfinite(f)])
        if floors.size:
            lines.append(
                f"Detection floor: the median trajectory could only have ruled "
                f"out D above {np.median(floors):.4g} µm²/s "
                f"(10th-90th pct {np.percentile(floors, 10):.3g}-"
                f"{np.percentile(floors, 90):.3g}). Below that, 'not "
                f"significantly moving' means the trajectory was too short or "
                f"too imprecise to tell, not that the molecule was still.")
        self.immobility_status_label.setText(" ".join(lines))

    def _on_immobility_settings_changed(self, *_args):
        """Precision or calibration moved, so the test has to be run again."""
        if self.tracks is None or self.tracks.empty:
            self._update_immobility_status()
            return
        self._start_fit_free_metrics_worker()

    def _start_fit_free_metrics_worker(self):
        # Fit-free (distance travelled, duration) but still a full pass over
        # every trajectory - background it too so linking/auto-loading a
        # large trajectories file doesn't freeze the UI while it runs.
        if self.tracks is None or self.tracks.empty:
            return
        # The immobility test needs a precision per trajectory point. Attaching
        # it as a column keeps the worker's input self-contained.
        tracks = self._tracks_for_metrics()
        _label, sigma_px, _measured = self._sigma_source()
        if sigma_px is not None and len(sigma_px) == len(tracks):
            tracks = tracks.assign(**{SIGMA_COLUMN: sigma_px})
        worker = _fit_free_metrics_worker(
            tracks, self.pixel_size_box.value(), self.fps_box.value(),
            alpha=self.immobility_alpha_box.value())
        worker.returned.connect(
            lambda result, ref=self.tracks: self._on_fit_free_metrics_finished(result, ref))
        worker.errored.connect(lambda exc: self.log(f"Distance/duration computation failed: {exc}"))
        self._metrics_worker_ref = worker
        worker.start()

    def _on_fit_free_metrics_finished(self, result, tracks=None):
        if tracks is not None and tracks is not self.tracks:
            # Computed for trajectories that have since gone to memory: they
            # belong with those, not with whatever is linked now.
            for kept in self._linked_memory.values():
                if kept["tracks"] is tracks:
                    for key, values in result.items():
                        kept["caches"][METRIC_CACHE_ATTR[key]] = values
            return
        for key, values in result.items():
            setattr(self, METRIC_CACHE_ATTR[key], values)
        self._invalidate_track_filter()
        for key, values in result.items():
            self._set_metric_default_bounds(key, values)
            self._set_metric_view_default(key, values)
            self._draw_metric_histogram(key)
        self.log(
            f"Computed distance, end-to-end displacement, straightness and "
            f"duration for {len(result['distance'])} trajectories"
        )
        if result.get("motion"):
            moving = sum(1 for p in result["pstatic"].values() if p < 0.05)
            self.log(f"Immobility test: {len(result['motion']) - moving} of "
                     f"{len(result['motion'])} trajectories are consistent with "
                     f"a static emitter (p > 0.05)")
        elif self.tracks is not None and not self.tracks.empty:
            self.log("Immobility test skipped: no localization precision available "
                     "for these trajectories.")
        self._update_immobility_status()
        self._update_population_counts()
        self._update_merge_label()

    def _set_metric_default_bounds(self, key, cache):
        boxes = self._metric_bound_boxes.get(key)
        if not boxes or not cache:
            return
        min_box, max_box = boxes
        values = np.asarray(list(cache.values()), float)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        min_box.blockSignals(True)
        max_box.blockSignals(True)
        min_box.setValue(float(values.min()))
        max_box.setValue(float(values.max()))
        min_box.blockSignals(False)
        max_box.blockSignals(False)

    def _metric_cache(self, key):
        if key == "time":
            return self._time_metric_cache()
        return getattr(self, METRIC_CACHE_ATTR[key]) or {}

    def _time_metric_cache(self):
        """The frame each trajectory first appears in.

        Time is the one colouring that needs no computing - it is in the table
        already - which is why it is built on demand here instead of being
        cached like D, distance and duration.
        """
        if self.tracks is None or self.tracks.empty:
            return {}
        return self.tracks.groupby("particle")["frame"].min().to_dict()

    def _current_metric_key(self):
        choice = self.color_metric_box.currentText()
        if choice.startswith("D ("):
            return "D"
        if choice.startswith("Distance"):
            return "distance"
        if choice.startswith("End-to-end"):
            return "net"
        if choice.startswith("Straightness"):
            return "straightness"
        if choice.startswith("Motion ratio"):
            return "motion"
        if choice.startswith("p_static"):
            return "pstatic"
        if choice.startswith("Smallest detectable"):
            return "dmin"
        if choice.startswith("Time"):
            return "time"
        return "duration"

    def _log_floor(self, key, requested):
        """A positive lower end for a log scale, from the data when need be.

        Zero is both the natural lower bound for a length or a rate and the one
        value a log axis cannot place. Substituting a fixed tiny constant - which
        this used to do - spends most of the axis, and most of the colormap, on
        decades that hold nothing: the histogram then looks like every
        trajectory is jammed against the right-hand edge with eight empty
        decades to its left. The smallest value actually present is the honest
        floor, and an explicit positive bound is always respected.
        """
        if requested > 0:
            return float(requested)
        values = np.asarray(list(self._metric_cache(key).values()), float)
        values = values[np.isfinite(values) & (values > 0)]
        return float(values.min()) if values.size else 1e-9

    # ------------------------------------------------------------------
    # Selecting trajectories by what was measured about them
    # ------------------------------------------------------------------
    def _active_metric_filters(self):
        """The (metric, low, high) ranges currently selecting, in tick order."""
        active = []
        for key in COMPUTED_METRICS:
            box = self._metric_filter_boxes.get(key)
            if box is None or not box.isChecked():
                continue
            low_box, high_box = self._metric_bound_boxes[key]
            active.append((key, low_box.value(), high_box.value()))
        return active

    def _invalidate_track_filter(self):
        """Drop the derived selection. Cheap; it is rebuilt on the next read."""
        self._passing_particles_cache = None
        self._loc_particle_cache = None
        self._merge_cache = None

    def _localization_particles(self):
        """Which trajectory each filtered localization belongs to, or -1.

        Matched on frame and position rather than carried through the linker as
        an extra column, because trajectories are as often read back from a
        previous run's CSV as linked in this session, and a join works the same
        for both. Within a session the coordinates on the two sides are the same
        floats - one is computed from the other - so the match is exact.
        """
        if self._loc_particle_cache is not None:
            return self._loc_particle_cache

        df = self.df_filtered
        n_rows = 0 if df is None else len(df)
        particles = np.full(n_rows, -1, dtype=np.int64)
        x_col = self._resolve_column("x")
        y_col = self._resolve_column("y")
        frame_col = self._resolve_column("frame")
        have_tracks = self.tracks is not None and not self.tracks.empty
        if n_rows and have_tracks and x_col and y_col and frame_col:
            pixel_size = max(self.pixel_size_box.value(), 1e-9)
            wanted = pd.DataFrame({
                "frame": df[frame_col].to_numpy(np.int64) + self._frame_offset(),
                "x": np.round(df[x_col].to_numpy(float) / pixel_size, LOC_MATCH_DECIMALS),
                "y": np.round(df[y_col].to_numpy(float) / pixel_size, LOC_MATCH_DECIMALS),
            })
            known = pd.DataFrame({
                "frame": self.tracks["frame"].to_numpy(np.int64),
                "x": np.round(self.tracks["x"].to_numpy(float), LOC_MATCH_DECIMALS),
                "y": np.round(self.tracks["y"].to_numpy(float), LOC_MATCH_DECIMALS),
                "particle": self.tracks["particle"].to_numpy(np.int64),
            }).drop_duplicates(subset=["frame", "x", "y"])
            merged = wanted.merge(known, on=["frame", "x", "y"], how="left", sort=False)
            particles = merged["particle"].fillna(-1).to_numpy(np.int64)

        self._loc_particle_cache = particles
        return particles

    def _passing_particles(self):
        """Trajectories inside every active range, or None when none is active.

        None and the empty set mean different things and both happen: None is
        "no dynamics filter, show everything", the empty set is "a filter that
        nothing satisfies", which has to leave the canvas empty rather than
        quietly showing all of it.

        A trajectory with no value for a metric being filtered on is excluded.
        D in particular is only fitted for trajectories long enough to support
        it, so filtering on D also drops the short ones - which is why the
        summary line counts them out loud.
        """
        active = self._active_metric_filters()
        if not active and self._population_class is None:
            return None
        if self._passing_particles_cache is not None:
            return self._passing_particles_cache
        if self.tracks is None or self.tracks.empty:
            return set()

        passing = set(self.tracks["particle"].to_numpy().tolist())
        for key, low, high in active:
            cache = self._metric_cache(key) or {}
            passing = {pid for pid in passing
                       if pid in cache
                       and np.isfinite(cache[pid])
                       and low <= cache[pid] <= high}
        if self._population_class is not None:
            passing &= self._class_members(self._population_class)
        self._passing_particles_cache = passing
        return passing

    def _displayed_tracks(self):
        """The trajectories to show: all of them, or those the filter kept."""
        passing = self._passing_particles()
        if passing is None or self.tracks is None or self.tracks.empty:
            return self.tracks
        return self.tracks[self.tracks["particle"].isin(passing)]

    def _displayed_localizations(self):
        """The localizations to show and to render.

        This is the point of the whole feature: a reconstruction built from
        these is a reconstruction of the molecules that behaved a certain way,
        so "where do the fast ones go?" becomes a picture rather than a table.
        """
        passing = self._passing_particles()
        df = self.df_filtered
        if passing is None or df is None or df.empty:
            return df
        particles = self._localization_particles()
        if particles.size != len(df):        # caches out of step; show everything
            return df
        if not passing:
            return df.iloc[:0]
        keep = np.isin(particles, np.fromiter(passing, np.int64, len(passing)))
        return df[keep]

    def _track_filter_summary(self):
        """What the dynamics filter is doing, in one line."""
        active = self._active_metric_filters()
        if not active and self._population_class is None:
            return "No dynamics filter - every trajectory is shown."
        if self.tracks is None or self.tracks.empty:
            return "No trajectories to filter yet - link some first."
        passing = self._passing_particles()
        n_total = int(self.tracks["particle"].nunique())
        n_kept = len(passing)
        criteria = ", ".join(
            f"{METRIC_LABELS[key].split(' (')[0]} {low:g}-{high:g}"
            for key, low, high in active)
        if self._population_class is not None:
            if self._soft_classes():
                how = "every fitted trajectory, weighted by its probability"
            elif self._classify_method() == "fit":
                how = f"population fit, P >= {self.class_probability_box.value():g}"
            else:
                how = criteria
            criteria = f"{self._population_class.capitalize()} ({how})"
        line = f"{criteria}: {n_kept} of {n_total} trajectories"
        df = self.df_filtered
        if df is not None and not df.empty:
            line += f", {len(self._displayed_localizations())} of {len(df)} localizations"
        # Missing values are the surprise worth naming: filtering on D drops
        # every trajectory too short for the fit, and nothing else says so.
        unmeasured = 0
        for key, _low, _high in active:
            cache = self._metric_cache(key) or {}
            unmeasured = max(unmeasured, sum(
                1 for pid in self.tracks["particle"].unique()
                if pid not in cache or not np.isfinite(cache[pid])))
        if unmeasured:
            line += f" ({unmeasured} have no value for a filtered metric and are excluded)"
        return line

    def _update_track_filter_label(self):
        if hasattr(self, "track_filter_label"):
            self.track_filter_label.setText(self._track_filter_summary())
        if hasattr(self, "clear_track_filter_button"):
            self.clear_track_filter_button.setEnabled(bool(self._active_metric_filters()))

    def _apply_track_filter(self):
        """Rebuild everything the selection feeds: layers, render, counts."""
        self._invalidate_track_filter()
        self._update_track_filter_label()
        self.render_overlay()
        self._refresh_render_tab()
        self._update_status_header()
        self._update_render_population_label()

    def _on_metric_filter_toggled(self, key):
        # Ticking a range by hand is a selection of its own; a class chosen by a
        # preset button would otherwise go on narrowing it out of sight.
        self._population_class = None
        box = self._metric_filter_boxes.get(key)
        if box is not None:
            low_box, high_box = self._metric_bound_boxes[key]
            state = "on" if box.isChecked() else "off"
            self.log(f"Dynamics filter on {METRIC_LABELS[key]} {state}"
                     + (f" ({low_box.value():g} to {high_box.value():g})"
                        if box.isChecked() else ""))
        self._apply_track_filter()

    def clear_track_filters(self):
        self._population_class = None
        for box in self._metric_filter_boxes.values():
            box.blockSignals(True)
            box.setChecked(False)
            box.blockSignals(False)
        self.log("Dynamics filters cleared - every trajectory is shown again")
        self._apply_track_filter()

    def _metric_norm_range(self, key):
        if key == "time":
            # No bounds box to read: time is spread over whatever the data
            # covers, so the first trajectory is at one end of the colormap and
            # the last at the other however long the acquisition ran.
            frames = (self.tracks["frame"] if self.tracks is not None
                      and not self.tracks.empty else None)
            lo = float(frames.min()) if frames is not None else 0.0
            hi = float(frames.max()) if frames is not None else 1.0
            return lo, max(hi, lo + 1e-9), False
        min_box, max_box = self._metric_bound_boxes[key]
        use_log = self._metric_use_log[key]
        lo = min_box.value()
        hi = max_box.value()
        if use_log:
            lo = self._log_floor(key, lo)
            hi = max(hi, lo * 1.0001)
        else:
            hi = max(hi, lo + 1e-9)
        return lo, hi, use_log

    def _normalize_metric(self, key, values):
        lo, hi, use_log = self._metric_norm_range(key)
        values = np.clip(values, lo, hi)
        if use_log:
            return np.clip((np.log10(values) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)), 0.0, 1.0)
        return np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    def _on_color_mode_changed(self, *_args):
        # Switching between metric colouring and per-track colours changes what
        # the layers are coloured *by*, not just the values, so this one does
        # need a rebuild. It is a single checkbox click, not a dragged value.
        for key in COMPUTED_METRICS:
            self._draw_metric_histogram(key)
        self.render_overlay()

    def _on_color_settings_changed(self, *_args):
        for key in COMPUTED_METRICS:
            self._draw_metric_histogram(key)
        # Metric choice and colormap are display-only: recolour, do not rebuild.
        self._refresh_metric_colors()

    def _refresh_metric_colors(self):
        """Recolour the trajectory layers in place, without rebuilding geometry.

        Metric bounds, the chosen metric and the colormap only affect colour, so
        touching layer.properties / layer.edge_color is enough. Rebuilding the
        Tracks and Shapes layers costs seconds once there are a few thousand
        trajectories; this costs tens of milliseconds.
        """
        if self.tracks is None or self.tracks.empty:
            return
        if not self.color_trajectories_box.isChecked():
            # Colours come from track identity, not from a metric: nothing to update.
            return

        key = self._current_metric_key()
        cache = self._metric_cache(key)

        if TRACKS_LAYER_NAME in self.viewer.layers and self._tracks_layer_particles is not None:
            layer = self.viewer.layers[TRACKS_LAYER_NAME]
            raw = np.array([cache.get(pid, np.nan) for pid in self._tracks_layer_particles], float)
            norm = np.zeros_like(raw)
            valid = np.isfinite(raw)
            if valid.any():
                norm[valid] = self._normalize_metric(key, raw[valid])
            try:
                layer.properties = {"metric_color": norm}
                layer.colormaps_dict = {
                    "metric_color": _get_napari_colormap(self.d_colormap_box.currentText())
                }
                layer.color_by = "metric_color"
            except Exception:
                # Any napari-side refusal falls back to the full rebuild.
                self.render_overlay()
                return

        if ALL_TRACKS_LAYER_NAME in self.viewer.layers and self._all_tracks_particle_ids:
            layer = self.viewer.layers[ALL_TRACKS_LAYER_NAME]
            cmap = matplotlib.colormaps[self.d_colormap_box.currentText()]
            colors = np.empty((len(self._all_tracks_particle_ids), 4), float)
            for i, pid in enumerate(self._all_tracks_particle_ids):
                val = cache.get(pid)
                if val is None or not np.isfinite(val):
                    colors[i] = (0.53, 0.53, 0.53, 1.0)
                else:
                    colors[i] = cmap(float(self._normalize_metric(key, np.array([val]))[0]))
            layer.edge_color = colors

    def apply_display_settings(self):
        """Explicit refresh, for when live updating is switched off."""
        self._metric_render_timer.stop()
        self._apply_track_style()
        self._refresh_metric_colors()

    def _accumulating_tracks(self):
        box = getattr(self, "traj_accumulate_box", None)
        return box is not None and box.isChecked()

    def _tail_length(self):
        """Frames of trail behind the current one; 0 in the box means all of it.

        Accumulating is the same setting made to grow: the trail is however far
        the current frame is past the chosen start, so it always reaches back to
        exactly that frame and no further.
        """
        if self._accumulating_tracks():
            return max(1, self._get_current_frame() - self.traj_start_frame_box.value())
        return self.traj_fade_box.value() or getattr(self, "_tracks_full_span", 1)

    def _on_accumulate_changed(self):
        accumulating = self._accumulating_tracks()
        self.traj_start_frame_box.setEnabled(accumulating)
        # A fixed trail and an accumulating one are two answers to the same
        # question, so only one of them is live at a time.
        self.traj_fade_box.setEnabled(not accumulating)
        self._on_fade_changed()

    def _on_fade_changed(self):
        self._apply_track_style()
        self._update_fade_status()

    def _sync_accumulating_tracks(self):
        """Regrow the trail as the slider moves, so it still reaches the start."""
        if not self._accumulating_tracks():
            return
        if TRACKS_LAYER_NAME not in self.viewer.layers:
            return
        try:
            self.viewer.layers[TRACKS_LAYER_NAME].tail_length = self._tail_length()
        except Exception:
            pass

    def _update_fade_status(self):
        """Say the trail length in seconds, which is what it is really chosen in."""
        if not hasattr(self, "traj_fade_status"):
            return
        interval = 1.0 / max(float(self.fps_box.value()), 1e-9)
        if self._accumulating_tracks():
            start = self.traj_start_frame_box.value()
            self.traj_fade_status.setText(
                f"Trajectories build up from frame {start} "
                f"({start * interval:.3g} s) onwards.")
            return
        frames = self.traj_fade_box.value()
        if frames <= 0:
            self.traj_fade_status.setText(
                "Trajectories stay drawn for their whole length.")
            return
        self.traj_fade_status.setText(
            f"{frames} frames of trail = {frames * interval:.3g} s of acquisition.")

    def _apply_track_style(self):
        """Widths and tail behaviour are properties of the live layers, not a rebuild."""
        if TRACKS_LAYER_NAME in self.viewer.layers:
            layer = self.viewer.layers[TRACKS_LAYER_NAME]
            layer.tail_width = self.line_width_box.value()
            layer.tail_length = self._tail_length()
            layer.hide_completed_tracks = not self.persist_tracks_box.isChecked()
        if ALL_TRACKS_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[ALL_TRACKS_LAYER_NAME].edge_width = (
                self.all_tracks_line_width_box.value()
            )

    # --- generic metric histogram (used for D, distance, duration) ---
    def _on_metric_log_toggled(self, key, use_log):
        """Switch a metric between a logarithmic and a linear scale.

        The setting is not the histogram's alone: `_metric_norm_range` reads it
        too, so it decides how the same numbers are spread across the colormap
        on the trajectories themselves. Changing one without the other would
        leave the plot and the viewer disagreeing about what a colour means.
        """
        self._metric_use_log[key] = bool(use_log)
        self._draw_metric_histogram(key)
        self._refresh_metric_colors()

    def _make_metric_histogram(self, key):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Bins:"))
        bins_box = QSpinBox()
        bins_box.setRange(5, 500)
        bins_box.setValue(30)
        bins_box.setMaximumWidth(55)
        toolbar.addWidget(bins_box)
        toolbar.addWidget(QLabel("View:"))
        # Enough decimals to hold the bounds these mirror when "follow filter"
        # is on. At four, a D bound of 4e-5 was rounded to a flat zero on the
        # way in, and the log axis then had to start from nothing and spent
        # eight decades getting back to the data.
        view_min_box = QDoubleSpinBox()
        view_min_box.setRange(-1e9, 1e9)
        view_min_box.setDecimals(METRIC_VIEW_DECIMALS)
        view_min_box.setMaximumWidth(90)
        toolbar.addWidget(view_min_box)
        toolbar.addWidget(QLabel(u"–"))
        view_max_box = QDoubleSpinBox()
        view_max_box.setRange(-1e9, 1e9)
        view_max_box.setDecimals(METRIC_VIEW_DECIMALS)
        view_max_box.setMaximumWidth(90)
        adaptive_steps(view_min_box, view_max_box)
        toolbar.addWidget(view_max_box)
        follow_box = QCheckBox("follow filter")
        follow_box.setChecked(True)
        view_min_box.setEnabled(False)  # follow_box starts checked
        view_max_box.setEnabled(False)
        follow_box.setToolTip(
            "Keep the plotted range equal to the min/max bounds above.\n"
            "Uncheck to set the view range independently of the filter."
        )
        toolbar.addWidget(follow_box)
        log_box = QCheckBox("log")
        log_box.setChecked(self._metric_use_log.get(key, False))
        log_box.setToolTip(
            "Spread the axis logarithmically. These quantities run over orders "
            "of magnitude across a population - a linear axis piles most "
            "trajectories into the first bin and shows the spread of the few "
            "fastest instead.\n\n"
            "It sets the colour scale as well as the histogram, so the "
            "trajectories in the viewer are shaded on the same scale."
        )
        toolbar.addWidget(log_box)
        toolbar.addWidget(self._png_button(
            lambda fig, opts, k=key: self._draw_metric_histogram(k, fig, opts),
            lambda k=key: f"{k}_histogram"))
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        figure = Figure(figsize=(5, 2.4))
        canvas = FigureCanvas(figure)
        self._plot_canvases.append(canvas)
        canvas.setMinimumHeight(220)
        layout.addWidget(canvas)

        state = {
            "figure": figure,
            "canvas": canvas,
            "bins_box": bins_box,
            "view_min_box": view_min_box,
            "view_max_box": view_max_box,
            "follow_box": follow_box,
            "log_box": log_box,
            "drag": None,
            "lower_line": None,
            "upper_line": None,
            "span": None,
        }
        self._metric_hist_widgets[key] = state
        bins_box.valueChanged.connect(lambda _v, k=key: self._draw_metric_histogram(k))
        view_min_box.valueChanged.connect(lambda _v, k=key: self._draw_metric_histogram(k))
        view_max_box.valueChanged.connect(lambda _v, k=key: self._draw_metric_histogram(k))
        follow_box.toggled.connect(lambda _on, k=key: self._on_metric_follow_toggled(k))
        log_box.toggled.connect(lambda on, k=key: self._on_metric_log_toggled(k, on))

        canvas.mpl_connect("button_press_event", lambda evt, k=key: self._on_metric_hist_press(k, evt))
        canvas.mpl_connect("motion_notify_event", lambda evt, k=key: self._on_metric_hist_motion(k, evt))
        canvas.mpl_connect("button_release_event", lambda evt, k=key: self._on_metric_hist_release(k, evt))
        return container

    def _on_metric_follow_toggled(self, key):
        state = self._metric_hist_widgets.get(key)
        if not state:
            return
        following = state["follow_box"].isChecked()
        # While following, the view boxes mirror the bounds and are not editable.
        state["view_min_box"].setEnabled(not following)
        state["view_max_box"].setEnabled(not following)
        if following:
            self._sync_metric_view_to_bounds(key)
        else:
            self._draw_metric_histogram(key)

    def _sync_metric_view_to_bounds(self, key):
        """Mirror the filter bounds into the plotted range, unless decoupled."""
        state = self._metric_hist_widgets.get(key)
        if not state or not state["follow_box"].isChecked():
            return
        min_box, max_box = self._metric_bound_boxes[key]
        view_min_box, view_max_box = state["view_min_box"], state["view_max_box"]
        lower, upper = min_box.value(), max_box.value()
        if upper <= lower:
            upper = lower + 1e-9
        view_min_box.blockSignals(True)
        view_max_box.blockSignals(True)
        view_min_box.setValue(lower)
        view_max_box.setValue(upper)
        view_min_box.blockSignals(False)
        view_max_box.blockSignals(False)
        self._draw_metric_histogram(key)

    def _set_metric_view_default(self, key, cache):
        state = self._metric_hist_widgets.get(key)
        if not state or not cache:
            return
        if state["follow_box"].isChecked():
            # The plotted range is the filter range; don't override it with the
            # data range.
            self._sync_metric_view_to_bounds(key)
            return
        values = np.asarray(list(cache.values()), float)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        view_min_box, view_max_box = state["view_min_box"], state["view_max_box"]
        view_min_box.blockSignals(True)
        view_max_box.blockSignals(True)
        view_min_box.setValue(float(values.min()))
        view_max_box.setValue(float(max(values.max(), values.min() + 1e-9)))
        view_min_box.blockSignals(False)
        view_max_box.blockSignals(False)

    def _draw_metric_histogram(self, key, figure=None, opts=None):
        """Draw one dynamics histogram. See `_draw_histogram` for `figure`."""
        state = self._metric_hist_widgets.get(key)
        if not state:
            return
        export = figure is not None
        opts = opts or {}
        figure = figure if export else state["figure"]
        figure.clear()
        if not export:
            state["lower_line"] = None
            state["upper_line"] = None
            state["span"] = None
        cache = self._metric_cache(key)
        if not cache:
            if not export:
                state["canvas"].draw_idle()
            return
        values = np.asarray(list(cache.values()), float)
        values = values[np.isfinite(values)]
        if not len(values):
            if not export:
                state["canvas"].draw_idle()
            return

        ax = figure.add_subplot(111)
        use_log = self._metric_use_log[key]
        n_bins = state["bins_box"].value()
        view_lo = state["view_min_box"].value()
        view_hi = state["view_max_box"].value()
        if view_hi <= view_lo:
            view_hi = view_lo + 1e-9

        centers = np.array([])
        if use_log:
            view_lo = self._log_floor(key, view_lo)
            shown = values[(values >= view_lo) & (values <= view_hi)]
            if len(shown):
                bins = np.logspace(np.log10(view_lo), np.log10(view_hi), n_bins + 1)
                counts, edges = np.histogram(shown, bins=bins)
                centers = np.sqrt(edges[:-1] * edges[1:])
        else:
            shown = values[(values >= view_lo) & (values <= view_hi)]
            if len(shown):
                bins = np.linspace(view_lo, view_hi, n_bins + 1)
                counts, edges = np.histogram(shown, bins=bins)
                centers = 0.5 * (edges[:-1] + edges[1:])

        lo, hi, _ = self._metric_norm_range(key)
        cmap_name = self.d_colormap_box.currentText()
        if len(centers):
            norm = LogNorm(vmin=lo, vmax=hi) if use_log else Normalize(vmin=lo, vmax=hi)
            colors = matplotlib.colormaps[cmap_name](norm(np.clip(centers, lo, hi)))
            ax.bar(edges[:-1], counts, width=np.diff(edges), color=colors, align="edge", edgecolor="none")
            if opts.get("errorbars"):
                ax.errorbar(centers, counts, yerr=np.sqrt(np.maximum(counts, 0)),
                            fmt="none", ecolor=INK, elinewidth=0.9, capsize=2, alpha=0.7)
            if opts.get("colorbar", True):
                sm = cm.ScalarMappable(norm=norm, cmap=cmap_name)
                sm.set_array([])
                figure.colorbar(sm, ax=ax)
        if use_log:
            ax.set_xscale("log")
        if view_hi <= view_lo:
            # Degenerate range (both bounds still at the same value): give the
            # axis a nominal span rather than letting matplotlib warn about it.
            span = abs(view_lo) * 0.5 or 1.0
            view_lo, view_hi = view_lo - span, view_lo + span
        ax.set_xlim(view_lo, view_hi)

        min_box, max_box = self._metric_bound_boxes[key]
        lower, upper = min_box.value(), max_box.value()
        if opts.get("bounds", True):
            span = ax.axvspan(lower, upper, color=LAVENDER, alpha=0.15, zorder=0)
            low_line = ax.axvline(lower, color=LAVENDER, linewidth=1.5)
            high_line = ax.axvline(upper, color=LAVENDER, linewidth=1.5)
        else:
            span = low_line = high_line = None

        ax.set_xlabel(METRIC_AXIS_LABELS.get(key, METRIC_LABELS[key]))
        ax.set_ylabel("Count")
        style_axes(figure, ax,
                   title=f"{len(values)} trajectories" if opts.get("title", True) else None)
        if not opts.get("grid", True):
            ax.grid(False)
        figure.tight_layout()
        if export:
            return
        state["span"], state["lower_line"], state["upper_line"] = span, low_line, high_line
        state["canvas"].draw_idle()

    def _sync_metric_hist_lines(self, key):
        state = self._metric_hist_widgets.get(key)
        if not state or not state["figure"].axes:
            return
        ax = state["figure"].axes[0]
        min_box, max_box = self._metric_bound_boxes[key]
        lower, upper = min_box.value(), max_box.value()
        if state.get("span") is not None:
            state["span"].remove()
        state["span"] = ax.axvspan(lower, upper, color=LAVENDER, alpha=0.15, zorder=0)
        if state.get("lower_line") is not None:
            state["lower_line"].set_xdata([lower, lower])
        if state.get("upper_line") is not None:
            state["upper_line"].set_xdata([upper, upper])
        state["canvas"].draw_idle()

    def _on_metric_bounds_changed(self, key):
        self._sync_metric_hist_lines(key)
        self._sync_metric_view_to_bounds(key)
        self._invalidate_track_filter()
        if not self.live_display_box.isChecked():
            return
        if self._active_metric_filters():
            # Now the bound decides *which* trajectories exist, not just what
            # colour they are, so the layers have to be rebuilt. Coalesced
            # harder than a recolour because it costs a great deal more.
            self._track_filter_timer.start(250)
            return
        # A metric bound otherwise only changes the colour scale - no geometry
        # moves - so this recolours the existing layers instead of rebuilding
        # them. The rebuild it used to do costs ~2.8 s for 1500 trajectories
        # against ~70 ms for a recolour, which is what made every keystroke
        # freeze the UI. Still coalesced: typing fires it per intermediate value.
        self._metric_render_timer.start(120)

    def _on_metric_hist_press(self, key, event):
        state = self._metric_hist_widgets.get(key)
        if not state or event.xdata is None or not state["figure"].axes:
            return
        lower_line, upper_line = state.get("lower_line"), state.get("upper_line")
        if lower_line is None or upper_line is None:
            return
        lower_x = lower_line.get_xdata()[0]
        upper_x = upper_line.get_xdata()[0]
        xlim = state["figure"].axes[0].get_xlim()
        tol = 0.03 * (xlim[1] - xlim[0])
        dist_lower = abs(event.xdata - lower_x)
        dist_upper = abs(event.xdata - upper_x)
        if dist_lower <= tol and dist_lower <= dist_upper:
            state["drag"] = "lower"
        elif dist_upper <= tol:
            state["drag"] = "upper"
        else:
            state["drag"] = None

    def _on_metric_hist_motion(self, key, event):
        state = self._metric_hist_widgets.get(key)
        if not state or state.get("drag") is None or event.xdata is None:
            return
        min_box, max_box = self._metric_bound_boxes[key]
        if state["drag"] == "lower":
            value = min(event.xdata, max_box.value())
            min_box.setValue(value)
        else:
            value = max(event.xdata, min_box.value())
            max_box.setValue(value)
        # min_box/max_box.valueChanged -> _on_metric_bounds_changed already
        # redraws the lines, so nothing else to do here.

    def _on_metric_hist_release(self, key, event):
        state = self._metric_hist_widgets.get(key)
        if not state or state.get("drag") is None:
            return
        state["drag"] = None
        self._metric_render_timer.stop()
        if self.live_display_box.isChecked():
            self._refresh_metric_colors()

    @staticmethod
    def _msd_label(pid, D, slope_error):
        """One legend entry: the trajectory, its D, and how well D is pinned down.

        D is a quarter of the fitted slope, so the error on it is a quarter of
        the error on the slope. A trajectory too short for the covariance to be
        defined simply shows no error rather than a fabricated zero.
        """
        if not np.isfinite(slope_error):
            return f"#{pid} D={D:.3g} µm²/s"
        return f"#{pid} D={D:.3g}±{slope_error / 4.0:.2g} µm²/s"

    def _msd_sigma_map(self):
        """Precision from the MSD intercept, per trajectory, in nm."""
        return {pid: msd_sigma_nm(fit[3])
                for pid, fit in (self._track_msd_cache or {}).items()
                if len(fit) > 3}

    def _update_msd_sigma_label(self):
        """Cross-check the two precisions against each other.

        The spot fit and the MSD intercept measure the same thing by completely
        different routes, so their ratio is a calibration with no free
        parameters - and it is exactly the factor the immobility test needs when
        the reported uncertainty is a Cramer-Rao bound rather than the error
        actually achieved.
        """
        if not hasattr(self, "msd_sigma_label"):
            return
        sigmas = np.array([s for s in self._msd_sigma_map().values() if np.isfinite(s)])
        if not sigmas.size:
            self.msd_sigma_label.setText(
                "Compute D to read the localization precision off the MSD intercept.")
            return
        from_msd = float(np.median(sigmas))
        total = len(self._track_msd_cache or {})
        text = [f"MSD intercept implies σ = {from_msd:.1f} nm "
                f"(median of {sigmas.size} of {total} trajectories; the rest have a "
                f"negative intercept, which motion blur alone can produce)."]

        _label, sigma_px, measured = self._sigma_source()
        if sigma_px is not None and len(sigma_px):
            reported = float(np.median(sigma_px)) * self.pixel_size_box.value()
            ratio = from_msd / max(reported, 1e-9)
            text.append(f"The localization fit reports {reported:.1f} nm"
                        + ("" if measured else " (fallback value)") + ".")
            text.append(
                f"Ratio {ratio:.2f}. These measure the same quantity by different "
                "routes, so on a population dominated by slow molecules this is "
                "the calibration factor for the immobility test - motion blur "
                "biases the intercept low, so read it off the slow end.")
        self.msd_sigma_label.setText(" ".join(text))

    def _draw_msd_validation(self, figure=None, opts=None):
        export = figure is not None
        opts = opts or {}
        figure = figure if export else self.msd_figure
        figure.clear()
        ax = figure.add_subplot(111)
        if not self._track_msd_cache:
            style_axes(figure, ax)
            if not export:
                self.msd_canvas.draw_idle()
            return

        items = sorted(
            self._track_msd_cache.items(),
            key=lambda kv: self._track_diffusion_cache.get(kv[0], 0.0),
        )
        n_sample = min(self.msd_sample_box.value(), len(items))
        idxs = np.linspace(0, len(items) - 1, n_sample).astype(int)
        colors = matplotlib.colormaps[self.d_colormap_box.currentText()](
            np.linspace(0.05, 0.95, max(n_sample, 1))
        )

        for i, idx in enumerate(idxs):
            pid, (tau, msd_vals, slope, intercept, slope_error) = items[idx]
            D = self._track_diffusion_cache.get(pid, float("nan"))
            # Log axes cannot show a non-positive value, and an MSD of exactly
            # zero at a lag nothing moved over is perfectly possible, so the
            # points are masked rather than left for matplotlib to drop silently.
            drawable = np.isfinite(msd_vals) & (msd_vals > 0) & (tau > 0)
            ax.plot(tau[drawable], msd_vals[drawable], "o-", color=colors[i],
                    alpha=0.85, markersize=3, linewidth=1)

            # The fit is a straight line in MSD, which on log axes is a curve,
            # so it needs sampling rather than its two end points. It starts at
            # the first lag time rather than at zero: tau=0 cannot be drawn, and
            # a fit with a negative intercept has no positive MSD there anyway.
            positive_tau = tau[tau > 0]
            if positive_tau.size:
                fit_tau = np.geomspace(positive_tau.min(), positive_tau.max(), 100)
                fit_msd = slope * fit_tau + intercept
                visible = fit_msd > 0
                ax.plot(fit_tau[visible], fit_msd[visible], "--", color=colors[i],
                        alpha=0.6, linewidth=1, label=self._msd_label(pid, D, slope_error))

        # Display only: the fit above was made on the raw values. A log-log MSD
        # is read for its *slope* - 1 for free diffusion, flatter for confined,
        # steeper for directed - which a linear plot buries at the short lags
        # where the difference actually shows.
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Lag time (s)")
        ax.set_ylabel("MSD (µm²)")
        style_axes(figure, ax,
                   title=(f"MSD fit validation ({n_sample} example trajectories)"
                          if opts.get("title", True) else None))
        if opts.get("legend", True):
            legend = ax.legend(fontsize=plot_font(-2), loc="upper left", ncol=2,
                               facecolor=PLOT_BG, edgecolor=PANEL_LINE, labelcolor=INK)
            legend.get_frame().set_alpha(0.85)
        if not opts.get("grid", True):
            ax.grid(False)
        figure.tight_layout()
        if not export:
            self.msd_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Export: plots + data + metadata
    # ------------------------------------------------------------------
    def export_analysis(self):
        if self.df is None:
            self.log("Load data first")
            return
        self.export_button.setEnabled(False)
        try:
            csv_path = self.csv_edit.text().strip()
            folder = self._make_analysis_folder(self._analysis_base_dir(), "export")
            folder.mkdir(parents=True, exist_ok=True)

            # The figures belong to Qt canvases, so they have to be rendered
            # here; the tables are just data and go to a worker.
            n_plots = self._export_plots(folder)
            tables = self._export_tables()
            metadata = self._collect_metadata(csv_path)
        except Exception as exc:
            self.log(f"Export failed: {exc}")
            self.export_button.setEnabled(True)
            return

        rows = sum(len(frame) for _name, frame in tables)
        self.log(f"Exported {n_plots} plots; writing {rows} rows to {folder}...")
        self.export_progress.setVisible(True)
        self.export_progress.setValue(0)
        self._arm_cancel(self._export_cancel, self.export_cancel_button)

        worker = _export_worker(folder, tables, metadata, self._export_cancel)
        worker.yielded.connect(lambda frac: self.export_progress.setValue(int(frac * 100)))
        worker.returned.connect(self._on_export_finished)
        worker.errored.connect(lambda exc: self.log(f"Export failed: {exc}"))
        worker.finished.connect(self._on_export_worker_finished)
        self._export_worker_ref = worker
        worker.start()

    def _export_tables(self):
        """The tables to write, prepared on the GUI thread.

        What is exported is what is on screen, dynamics filter included: an
        export that quietly held more than the reconstruction beside it would
        be the more confusing of the two answers.
        """
        tables = []
        shown_locs = self._displayed_localizations()
        if shown_locs is not None:
            tables.append(("localizations_filtered.csv", shown_locs))
        shown_tracks = self._displayed_tracks()
        if shown_tracks is not None and not shown_tracks.empty:
            tables.append(("trajectories.csv", shown_tracks))
            tables.append(("track_metrics.csv", self._track_metrics_frame()))
        drift = self._drift_table()
        if drift is not None:
            tables.append(("drift_per_frame.csv", drift))
        # Beside the fitted table, not instead of it: a merged table has lost
        # the per-frame positions, and it is only right for as long as the
        # immobility test and its thresholds are.
        if self._merge_ids():
            merged = self._render_table()
            if merged is not None and MERGED_COUNT_COLUMN in merged.columns:
                tables.append(("localizations_merged.csv", merged))
        return tables

    def _on_export_worker_finished(self):
        self.export_button.setEnabled(True)
        self.export_cancel_button.setEnabled(False)
        self.export_progress.setVisible(False)
        self._export_worker_ref = None

    def _on_export_finished(self, result):
        if result is CANCELLED:
            self.log("Export cancelled - the folder may hold a partial export")
            return
        self.log(f"Export complete: {result}")

    @staticmethod
    def _dataset_dir(path):
        """The dataset folder a file belongs to, climbing out of any analysis tree.

        A previous run's table lives at <dataset>/analysis/<run>/data/locs.csv,
        and taking its parent as the place to put results is how
        analysis/data/analysis/<run>/data/ came about: every reload of an
        exported table nested one level deeper. Results belong beside the raw
        data, wherever within the tree the file that was opened happens to sit.
        """
        folder = Path(path).parent
        parts = folder.parts
        lowered = [part.lower() for part in parts]
        if ANALYSIS_ROOT in lowered:
            # The outermost analysis/ - a nested one is itself the symptom.
            cut = lowered.index(ANALYSIS_ROOT)
            if cut > 0:
                return Path(*parts[:cut])
        return folder

    def _analysis_base_dir(self):
        """The folder results go beside: the dataset's, not the plugin's.

        Prefer the localization CSV, then the image - an in-app fit has no CSV -
        and only fall back to the working directory if neither is known, so
        results never end up outside the dataset they describe.
        """
        csv_path = self.csv_edit.text().strip()
        image_path = self.image_edit.text().strip()
        if csv_path:
            return self._dataset_dir(csv_path)
        if image_path:
            return self._dataset_dir(image_path)
        return Path.cwd()

    def _make_analysis_folder(self, base_dir, kind="export"):
        """A folder of its own for this run, stamped with when it happened.

        Runs used to be numbered - analysis, analysis_2, analysis_3 - which kept
        them from overwriting each other but said nothing about when any of them
        happened or which came from which settings. A timestamp answers both,
        and sorts.

        The suffix only appears if two runs land in the same second, which takes
        deliberate effort but is exactly when losing one would be worst.
        """
        root = Path(base_dir) / ANALYSIS_ROOT
        stamp = datetime.now().strftime(RUN_STAMP_FORMAT)
        candidate = root / f"{stamp}_{kind}"
        i = 2
        while candidate.exists():
            candidate = root / f"{stamp}_{kind}_{i}"
            i += 1
        return candidate

    @staticmethod
    def _safe_filename(name):
        keep = "".join(c if c.isalnum() else "_" for c in name)
        while "__" in keep:
            keep = keep.replace("__", "_")
        return keep.strip("_") or "column"

    def _export_plots(self, folder):
        plots_dir = folder / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for column, state in self._hist_widgets.items():
            if state["figure"].axes:
                state["figure"].savefig(
                    plots_dir / f"filter_{self._safe_filename(column)}.png",
                    dpi=200,
                    facecolor=state["figure"].get_facecolor(),
                )
                count += 1
        for key, state in self._metric_hist_widgets.items():
            if state["figure"].axes:
                state["figure"].savefig(plots_dir / f"{key}_distribution.png", dpi=200)
                count += 1
        if self.msd_figure.axes:
            self.msd_figure.savefig(plots_dir / "msd_validation.png", dpi=200)
            count += 1
        if self.drift_figure.axes:
            self.drift_figure.savefig(plots_dir / "drift.png", dpi=200)
            count += 1
        if self.rcc_figure.axes:
            self.rcc_figure.savefig(plots_dir / "rcc_drift.png", dpi=200)
            count += 1
        if self.population_figure.axes:
            self.population_figure.savefig(plots_dir / "populations.png", dpi=200)
            count += 1
        return count

    def _track_metrics_frame(self):
        shown = self._displayed_tracks()
        particle_ids = sorted(shown["particle"].unique())
        d_map = self._track_diffusion_cache or {}
        distance_map = self._track_distance_cache or {}
        net_map = self._track_net_cache or {}
        straightness_map = self._track_straightness_cache or {}
        duration_map = self._track_duration_cache or {}
        motion_map = self._track_motion_cache or {}
        pstatic_map = self._track_pstatic_cache or {}
        dmin_map = self._track_dmin_cache or {}
        sigma_msd_map = self._msd_sigma_map()
        posteriors = {}
        if self._population_fit is not None and self._population_fit_current():
            result = self._population_fit["result"]
            row = {pid: i for i, pid in enumerate(result["pid"].tolist())}
            for k, label in enumerate(result["labels"]):
                column = result["posterior"][:, k]
                posteriors[f"P_{label}"] = {pid: float(column[i]) for pid, i in row.items()}
        table = pd.DataFrame([
            {
                "particle": pid,
                "D_um2_per_s": d_map.get(pid),
                "distance_um": distance_map.get(pid),
                "net_displacement_um": net_map.get(pid),
                "straightness": straightness_map.get(pid),
                "duration_s": duration_map.get(pid),
                "motion_ratio": motion_map.get(pid),
                "p_static": pstatic_map.get(pid),
                "d_detectable_um2_per_s": dmin_map.get(pid),
                "sigma_from_msd_nm": sigma_msd_map.get(pid),
            }
            for pid in particle_ids
        ])
        for name, values in posteriors.items():
            table[name] = [values.get(pid) for pid in particle_ids]
        return table

    def _write_metadata(self, folder, csv_path):
        with open(folder / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._collect_metadata(csv_path), f, indent=2, default=str)

    def _collect_metadata(self, csv_path):
        """Snapshot every setting as a plain dict, on the GUI thread."""
        n_tracks = int(self.tracks["particle"].nunique()) if self.tracks is not None and not self.tracks.empty else 0
        n_candidate_frames = sum(1 for c in self._loc2d_candidates if c is not None and len(c[0]) > 0)

        try:
            trackpy_version = tp.__version__
        except Exception:
            trackpy_version = None
        try:
            napari_version = napari.__version__
        except Exception:
            napari_version = None

        metadata = {
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "software": {"napari_version": napari_version, "trackpy_version": trackpy_version},
            "source_csv": csv_path or None,
            # Which run this is and what it was for, so a folder full of them
            # can be read without opening every file to tell them apart.
            "run": {
                "stamp": datetime.now().strftime(RUN_STAMP_FORMAT),
                "analysis_root": str(self._analysis_base_dir() / ANALYSIS_ROOT),
            },
            "source_image": self.image_edit.text().strip() or None,
            "pixel_size_nm_per_px": self.pixel_size_box.value(),
            "frame_number_shift": int(self._frame_shift),
            # The camera offset and frame rate below are recorded as applied,
            # so they already account for this factor; restoring both together
            # reproduces the run without applying the binning twice.
            "preprocessing": {"time_bin_frames": int(self.bin_factor_box.value())},
            "drift_correction": self._drift_metadata(),
            "population_fit": self._population_fit_metadata(),
            "n_localizations_total": len(self.df) if self.df is not None else 0,
            "n_localizations_filtered": len(self.df_filtered) if self.df_filtered is not None else 0,
            "filter_bounds": {
                col: {"min": lo.value(), "max": hi.value()} for col, (lo, hi) in self.filter_controls.items()
            },
            # Every column the table had, filtered on or not: a later load can
            # then say exactly what the new table lacks.
            "localization_columns": ([str(c) for c in self.df.columns]
                                     if self.df is not None else []),
            "localization_2d": {
                "gain_adu_per_electron": self.loc_gain_box.value(),
                "offset_adu": self.loc_offset_box.value(),
                "box_size_px": self._loc2d_box_size(),
                "min_net_gradient": self.loc_min_ng_box.value(),
                "fit_backend": self.loc_backend_box.currentText(),
                "n_frames_with_candidates": int(n_candidate_frames),
                "n_candidates_total": int(self._loc2d_counts.sum()) if len(self._loc2d_counts) else 0,
            },
            "smlm_rendering": {
                "oversampling": self.render_oversampling_box.value(),
                "mode": self.render_mode_box.currentData(),
                "mode_label": self.render_mode_box.currentText(),
                "global_sigma_nm": self.render_sigma_box.value(),
                "sigma_column": self.render_sigma_column_box.currentText() or None,
                "sigma_clamp_min_nm": self.render_sigma_min_box.value(),
                "sigma_clamp_max_nm": self.render_sigma_max_box.value(),
                "weight_by_photons": self.render_photons_box.isChecked(),
                "colormap": self.render_colormap_box.currentText(),
                "use_gpu": self.render_gpu_box.isChecked(),
                "frames_per_group": self.render_frames_per_box.value(),
                "grouping": self.render_grouping_box.currentData(),
                "grouping_label": self.render_grouping_box.currentText(),
                "window_step_frames": self.render_step_box.value(),
                "add_layer_to_viewer": self.render_add_layer_box.isChecked(),
                "layer_name": self.render_layer_name_edit.text(),
                "population_split_p": self.render_population_p_box.value(),
                # Which class the dynamics filter was pointed at by the
                # population buttons, and where immobile ends.
                "population": self._population_class,
                "classify_by": self.classify_method_box.currentData(),
                "class_probability": self.class_probability_box.value(),
                "soft_by_probability": self.class_soft_box.isChecked(),
                "immobile_definition": {
                    "max_detectable_d_um2_s": self.immobile_dmax_box.value(),
                    "min_points": self.immobile_min_points_box.value(),
                    "n_trajectories_by_class": self._class_counts(),
                },
                "merge_immobile": {
                    "enabled": self.merge_box.isChecked(),
                },
                "dynamics_selection": self._render_population_label(),
                "write_png_snapshot": self.render_png_box.isChecked(),
                "image_save_format": self.render_image_format_box.currentData(),
                "movie_save_format": self.render_movie_format_box.currentData(),
                "movie_save_stride": self.movie_stride_box.value(),
                "rotate_degrees": self.render_rotate_box.value(),
                "composite": {
                    "reconstruction": self.render_composite_base_box.isChecked(),
                    "localizations": self.render_composite_locs_box.isChecked(),
                    "localization_color": self.render_locs_color_box.currentText(),
                    "localization_size_nm": self.render_locs_size_box.value(),
                    "trajectories": self.render_composite_tracks_box.isChecked(),
                    "trajectory_color": self.render_tracks_color_box.currentText(),
                    "trajectory_width_nm": self.render_tracks_width_box.value(),
                    "every_visible_layer": self.render_composite_all_box.isChecked(),
                },
                "timestamp": {
                    "enabled": self.render_timestamp_box.isChecked(),
                    "height_px": self.render_timestamp_size_box.value(),
                    "color": self.render_timestamp_color_box.currentText(),
                    "position": self.render_timestamp_position_box.currentText(),
                },
                "scale_bar": {
                    "enabled": self.render_scalebar_box.isChecked(),
                    "automatic": self.render_scalebar_auto_box.isChecked(),
                    "length_nm": self.render_scalebar_length_box.value(),
                    "color": self.render_scalebar_color_box.currentText(),
                    "position": self.render_scalebar_position_box.currentText(),
                },
                "crop_to_box": self.render_crop_box.isChecked(),
                "gpu_status": smlm_render.render_gpu_status(),
            },
            "linking": {
                "search_range_nm": self.search_box.value(),
                "memory": self.memory_box.value(),
                "min_track_length": self.min_traj_box.value(),
                # Acquisition timing belongs to linking now; older files carry it
                # under "diffusion" and are still read from there.
                "fps": self.fps_box.value(),
                "frame_interval_ms": self.frame_interval_box.value(),
                "n_trajectories": n_tracks,
            },
            "diffusion": {
                "max_lagtime_frames": self.max_lagtime_box.value(),
                "min_track_length_for_d": self.d_min_length_box.value(),
                "d_min": self.d_min_box.value(),
                "d_max": self.d_max_box.value(),
                "n_tracks_with_D": len(self._track_diffusion_cache or {}),
                "msd_validation_sample_count": self.msd_sample_box.value(),
                # The other half of the same fit: MSD = 4*D*tau + 4*sigma^2.
                "localization_precision_from_msd_nm": (
                    float(np.median([s for s in self._msd_sigma_map().values()
                                     if np.isfinite(s)]))
                    if any(np.isfinite(s) for s in self._msd_sigma_map().values())
                    else None),
            },
            "distance_bounds_um": {"min": self.dist_min_box.value(), "max": self.dist_max_box.value()},
            "net_displacement_bounds_um": {
                "min": self.net_min_box.value(), "max": self.net_max_box.value()},
            "straightness_bounds": {
                "min": self.straight_min_box.value(), "max": self.straight_max_box.value()},
            "duration_bounds_s": {"min": self.dur_min_box.value(), "max": self.dur_max_box.value()},
            "immobility": {
                "fallback_precision_nm": self.immobility_sigma_box.value(),
                "precision_calibration": self.immobility_calibration_box.value(),
                "significance": self.immobility_alpha_box.value(),
                "precision_source": self._sigma_source()[0],
                "n_consistent_with_static": sum(
                    1 for p in (self._track_pstatic_cache or {}).values() if p >= 0.05),
            },
            "motion_ratio_bounds": {"min": self.motion_min_box.value(),
                                    "max": self.motion_max_box.value()},
            "p_static_bounds": {"min": self.pstatic_min_box.value(),
                                "max": self.pstatic_max_box.value()},
            "detectable_d_bounds": {"min": self.dmin_min_box.value(),
                                    "max": self.dmin_max_box.value()},
            # Which ranges were selecting rather than only colouring. Recorded
            # beside the bounds themselves, which are already here under
            # "diffusion", "distance_bounds_um" and the rest.
            "dynamics_filter": {
                key: box.isChecked() for key, box in self._metric_filter_boxes.items()
            },
            "n_trajectories_after_dynamics_filter": (
                int(self._displayed_tracks()["particle"].nunique()) if n_tracks else 0),
            "coloring": {
                "enabled": self.color_trajectories_box.isChecked(),
                "metric": self.color_metric_box.currentText(),
                "colormap": self.d_colormap_box.currentText(),
            },
            "display_layers": {
                "show_localizations": self.show_points_box.isChecked(),
                "show_active_growing_tracks": self.show_tracks_box.isChecked(),
                "show_static_all_tracks": self.show_all_tracks_box.isChecked(),
            },
            "rendering": {
                "marker_size": self.marker_size_box.value(),
                "marker_edge_width": self.marker_edge_width_box.value(),
                "marker_symbol": self.marker_choice.currentText(),
                "active_track_line_width": self.line_width_box.value(),
                "static_track_line_width": self.all_tracks_line_width_box.value(),
                "persist_completed_tracks": self.persist_tracks_box.isChecked(),
                "plot_aspect": self.plot_aspect_box.currentText(),
                "plot_height_px": self.plot_height_box.value(),
                "plot_font_pt": self.plot_font_box.value(),
            },
            "filter_histogram_display": {
                col: {
                    "bins": state["bins_box"].value(),
                    "view_min": state["view_min_box"].value(),
                    "view_max": state["view_max_box"].value(),
                }
                for col, state in self._hist_widgets.items()
            },
            "metric_histogram_display": {
                key: {
                    "bins": state["bins_box"].value(),
                    "view_min": state["view_min_box"].value(),
                    "view_max": state["view_max_box"].value(),
                    "follow_filter": state["follow_box"].isChecked(),
                    # Recorded because it decides how the colours were spread,
                    # not just how the histogram looked.
                    "log_scale": state["log_box"].isChecked(),
                }
                for key, state in self._metric_hist_widgets.items()
            },
        }
        return metadata

    # ------------------------------------------------------------------
    # Tracks / all-tracks / ROI layers
    # ------------------------------------------------------------------
    def _sync_tracks_layer(self):
        self._remove_layer(TRACKS_LAYER_NAME)
        shown = self._displayed_tracks()
        has_tracks = shown is not None and not shown.empty
        if not (self.show_tracks_box.isChecked() and has_tracks):
            return

        traj = shown.sort_values(["particle", "frame"])
        track_id = traj["particle"].to_numpy(int)
        # Remembered so colours can later be recomputed in this exact row order
        # without rebuilding the layer.
        self._tracks_layer_particles = track_id
        t = traj["frame"].to_numpy(int)
        y = traj["y"].to_numpy(float)
        x = traj["x"].to_numpy(float)
        data = np.column_stack([track_id, t, y, x])
        # Remembered so the trail can be set back to "the whole trajectory"
        # without rebuilding the layer to find out how long that is.
        self._tracks_full_span = int(t.max() - t.min()) + 1 if len(t) else 1
        tail_length = self._tail_length()
        hide_completed = not self.persist_tracks_box.isChecked()

        kwargs = dict(
            name=TRACKS_LAYER_NAME,
            tail_width=self.line_width_box.value(),
            tail_length=tail_length,
            hide_completed_tracks=hide_completed,
            visible=True,
        )

        if self.color_trajectories_box.isChecked():
            key = self._current_metric_key()
            cache = self._metric_cache(key)
            raw = np.array([cache.get(pid, np.nan) for pid in track_id], dtype=float)
            valid = np.isfinite(raw)
            norm = np.zeros_like(raw)
            if valid.any():
                norm[valid] = self._normalize_metric(key, raw[valid])
            colormap_name = self.d_colormap_box.currentText()
            kwargs["properties"] = {"metric_color": norm}
            kwargs["color_by"] = "metric_color"
            kwargs["colormap"] = "viridis"  # valid registry fallback, unused: colormaps_dict wins below
            kwargs["colormaps_dict"] = {"metric_color": _get_napari_colormap(colormap_name)}
        else:
            kwargs["color_by"] = "track_id"
            kwargs["colormap"] = "hsv"

        tracks_layer = self.viewer.add_tracks(data, **self._placed(kwargs, 3))
        tracks_layer._get_tooltip_text = self._tracks_layer_tooltip
        # napari only asks the *active* layer for tooltip text, and only when
        # tooltips are on at all. They used to be switched on by the points
        # layer alone, so a session showing trajectories without localizations -
        # every session that loads trajectories back from a previous run - had a
        # tooltip that was computed correctly and never displayed.
        self.viewer.tooltip.visible = True
        self._apply_viewer_scale()

    def _tooltip_diffusion(self, pid):
        """The D line for one trajectory, with its uncertainty when there is one.

        D is a quarter of the fitted MSD slope, so the error on it is a quarter
        of the error on the slope. A trajectory too short for the covariance to
        be defined shows no error rather than a fabricated zero - the same rule
        the MSD validation legend follows, so the two agree on screen.
        """
        D = (self._track_diffusion_cache or {}).get(pid)
        if D is None:
            return None
        fit = (self._track_msd_cache or {}).get(pid)
        slope_error = fit[4] if fit is not None and len(fit) > 4 else float("nan")
        if not np.isfinite(slope_error):
            return f"D {D:.4g} µm²/s"
        return f"D {D:.4g} ± {slope_error / 4.0:.2g} µm²/s"

    def _track_tooltip_lines(self, pid):
        """What to say about the trajectory under the cursor.

        What identifies it first, then what has been measured about it. A metric
        that has not been computed yet is left out rather than shown as zero or
        as a dash: an absent line means "not run", which is a different thing
        from a trajectory whose straightness really is zero.
        """
        if self.tracks is None:
            return []
        track_rows = self.tracks[self.tracks["particle"] == pid]
        if track_rows.empty:
            return []
        frames = track_rows["frame"].to_numpy(int)
        first, last = int(frames.min()), int(frames.max())
        span = last - first + 1
        # Points can be fewer than the span: the linker bridges gaps up to the
        # memory setting, so a trajectory may be absent from frames it spans.
        lines = [
            f"track {int(pid)}",
            f"starts at frame {first}, ends at {last}",
            f"spans {span} frames, {len(track_rows)} points",
        ]
        duration_map = self._track_duration_cache or {}
        if pid in duration_map:
            lines.append(f"duration {duration_map[pid]:.3g} s")
        diffusion = self._tooltip_diffusion(pid)
        if diffusion is not None:
            lines.append(diffusion)
        distance_map = self._track_distance_cache or {}
        if pid in distance_map:
            lines.append(f"distance travelled {distance_map[pid]:.3g} µm")
        net_map = self._track_net_cache or {}
        if pid in net_map:
            lines.append(f"end-to-end {net_map[pid]:.3g} µm")
        straightness_map = self._track_straightness_cache or {}
        if pid in straightness_map and np.isfinite(straightness_map[pid]):
            lines.append(f"straightness {straightness_map[pid]:.2f}")
        return lines

    def _tracks_layer_tooltip(self, position, *, view_direction=None, dims_displayed=None, world=False):
        if TRACKS_LAYER_NAME not in self.viewer.layers or self.tracks is None:
            return ""
        layer = self.viewer.layers[TRACKS_LAYER_NAME]
        pid = layer.get_value(
            position, view_direction=view_direction, dims_displayed=dims_displayed, world=world
        )
        if pid is None:
            return ""
        return "\n".join(self._track_tooltip_lines(pid))

    def _all_tracks_layer_tooltip(self, position, *, view_direction=None, dims_displayed=None, world=False):
        if ALL_TRACKS_LAYER_NAME not in self.viewer.layers or self.tracks is None:
            return ""
        layer = self.viewer.layers[ALL_TRACKS_LAYER_NAME]
        result = layer.get_value(
            position, view_direction=view_direction, dims_displayed=dims_displayed, world=world
        )
        shape_idx = result[0] if isinstance(result, tuple) else result
        particle_ids = getattr(self, "_all_tracks_particle_ids", [])
        if shape_idx is None or shape_idx >= len(particle_ids):
            return ""
        return "\n".join(self._track_tooltip_lines(particle_ids[shape_idx]))

    def _sync_all_tracks_layer(self):
        self._remove_layer(ALL_TRACKS_LAYER_NAME)
        shown = self._displayed_tracks()
        has_tracks = shown is not None and not shown.empty
        if not (self.show_all_tracks_box.isChecked() and has_tracks):
            return

        color_by_metric = self.color_trajectories_box.isChecked()
        key = self._current_metric_key() if color_by_metric else None
        cache = self._metric_cache(key) if color_by_metric else {}
        cmap = matplotlib.colormaps[self.d_colormap_box.currentText()]

        paths = []
        edge_colors = []
        particle_ids = []
        for i, (pid, group) in enumerate(shown.groupby("particle")):
            y = group["y"].to_numpy(float)
            x = group["x"].to_numpy(float)
            if len(x) < 2:
                continue
            paths.append(np.column_stack([y, x]))
            particle_ids.append(pid)
            if color_by_metric:
                val = cache.get(pid)
                if val is None or not np.isfinite(val):
                    edge_colors.append("#888888")
                else:
                    norm = float(self._normalize_metric(key, np.array([val]))[0])
                    edge_colors.append(cmap(norm))
            else:
                edge_colors.append(TRACK_PALETTE[i % len(TRACK_PALETTE)])

        if not paths:
            return

        self._all_tracks_particle_ids = particle_ids
        # Deliberately 2D (y, x) only, with no frame axis: napari shows
        # layers with fewer dims than the viewer on every slice, so this is
        # a static, always-visible reference of every trajectory,
        # independent of the growing/active tracks layer above.
        all_tracks_layer = self.viewer.add_shapes(
            paths,
            shape_type="path",
            edge_color=edge_colors,
            face_color="transparent",
            edge_width=self.all_tracks_line_width_box.value(),
            name=ALL_TRACKS_LAYER_NAME,
            **self._placed({}, 2),
        )
        all_tracks_layer._get_tooltip_text = self._all_tracks_layer_tooltip
        self.viewer.tooltip.visible = True
        self._apply_viewer_scale()

    def _xy_filter_is_in_use(self, x_col, y_col):
        """True when the x/y box is actually cropping something.

        Compared against the bounds the filters were built with: if neither has
        been moved, the box is selecting the whole field and is only clutter on
        top of a reconstruction.
        """
        for column in (x_col, y_col):
            lower_box, upper_box = self.filter_controls[column]
            default = self._default_bounds.get(column)
            if default is None:
                continue
            span = abs(default[1] - default[0]) or 1.0
            if (abs(lower_box.value() - default[0]) > 1e-6 * span
                    or abs(upper_box.value() - default[1]) > 1e-6 * span):
                return True
        return False

    def _sync_xy_roi_layer(self):
        x_col = self._resolve_column("x")
        y_col = self._resolve_column("y")
        if (
            not x_col
            or not y_col
            or x_col not in self.filter_controls
            or y_col not in self.filter_controls
        ):
            self._remove_layer(ROI_LAYER_NAME)
            return

        # The box is the x/y filter's only control, so it has to be there while
        # the Filter tab is open. Everywhere else it is just a yellow rectangle
        # sitting on top of the picture, so it is only kept when it is actually
        # excluding localizations.
        on_filter_tab = self.tabs.tabText(self.tabs.currentIndex()) == "Filter"
        if not on_filter_tab and not self._xy_filter_is_in_use(x_col, y_col):
            self._remove_layer(ROI_LAYER_NAME)
            return

        pixel_size = self.pixel_size_box.value()
        x_lo = self.filter_controls[x_col][0].value() / pixel_size
        x_hi = self.filter_controls[x_col][1].value() / pixel_size
        y_lo = self.filter_controls[y_col][0].value() / pixel_size
        y_hi = self.filter_controls[y_col][1].value() / pixel_size
        rect = np.array(
            [
                [y_lo, x_lo],
                [y_lo, x_hi],
                [y_hi, x_hi],
                [y_hi, x_lo],
            ]
        )

        self._roi_updating = True
        try:
            if ROI_LAYER_NAME in self.viewer.layers:
                layer = self.viewer.layers[ROI_LAYER_NAME]
                layer.data = [rect]
                # Reassigning .data invalidates the cached selection/resize
                # box, so the drag handles silently stop working (and can
                # crash on the next drag) unless we reselect right after.
                layer.selected_data = {0}
            else:
                # Also deliberately 2D-only, so the ROI box stays visible and
                # editable on every frame regardless of the dims slider.
                layer = self.viewer.add_shapes(
                    [rect],
                    shape_type="rectangle",
                    name=ROI_LAYER_NAME,
                    edge_color="yellow",
                    face_color="transparent",
                    edge_width=2,
                    **self._placed({}, 2),
                )
                layer.mode = "select"
                layer.selected_data = {0}
                layer.events.data.connect(self._on_roi_changed)
                self._apply_viewer_scale()
        finally:
            self._roi_updating = False

    def _on_roi_changed(self, event=None):
        if self._roi_updating:
            return
        if ROI_LAYER_NAME not in self.viewer.layers:
            return
        layer = self.viewer.layers[ROI_LAYER_NAME]
        if len(layer.data) == 0:
            return
        rect = layer.data[0]
        y_vals = rect[:, 0]
        x_vals = rect[:, 1]
        pixel_size = self.pixel_size_box.value()
        x_col = self._resolve_column("x")
        y_col = self._resolve_column("y")
        if x_col in self.filter_controls:
            self.filter_controls[x_col][0].setValue(float(x_vals.min()) * pixel_size)
            self.filter_controls[x_col][1].setValue(float(x_vals.max()) * pixel_size)
        if y_col in self.filter_controls:
            self.filter_controls[y_col][0].setValue(float(y_vals.min()) * pixel_size)
            self.filter_controls[y_col][1].setValue(float(y_vals.max()) * pixel_size)
        self.apply_filters()

    # ------------------------------------------------------------------
    # Per-column histograms (Filter localizations tab)
    # ------------------------------------------------------------------
    def _make_histogram_widget(self, column):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Size:"))
        toolbar.addWidget(self._png_button(
            lambda fig, opts, c=column: self._draw_histogram(c, fig, opts),
            lambda c=column: f"{c}_histogram"))
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        bins_row = QHBoxLayout()
        bins_row.addWidget(QLabel("Bins:"))
        bins_box = QSpinBox()
        bins_box.setRange(5, 500)
        bins_box.setValue(50)
        bins_box.setMaximumWidth(55)
        bins_row.addWidget(bins_box)
        bins_row.addWidget(QLabel("View:"))
        view_min_box = QDoubleSpinBox()
        view_min_box.setRange(-1e9, 1e9)
        view_min_box.setDecimals(FILTER_BOUND_DECIMALS)
        view_min_box.setMaximumWidth(90)
        bins_row.addWidget(view_min_box)
        bins_row.addWidget(QLabel(u"–"))
        view_max_box = QDoubleSpinBox()
        view_max_box.setRange(-1e9, 1e9)
        view_max_box.setDecimals(FILTER_BOUND_DECIMALS)
        view_max_box.setMaximumWidth(90)
        adaptive_steps(view_min_box, view_max_box)
        bins_row.addWidget(view_max_box)
        bins_row.addStretch(1)
        layout.addLayout(bins_row)

        default_lo, default_hi = self._default_bounds_for(column)
        view_min_box.setValue(default_lo)
        view_max_box.setValue(default_hi)

        figure = Figure(figsize=(4.2, DEFAULT_PLOT_HEIGHT / 100))
        canvas = FigureCanvas(figure)
        self._plot_canvases.append(canvas)
        canvas.setMinimumHeight(DEFAULT_PLOT_HEIGHT)
        canvas.setMaximumHeight(DEFAULT_PLOT_HEIGHT)
        canvas.setMinimumWidth(260)
        layout.addWidget(canvas)

        state = {
            "figure": figure,
            "canvas": canvas,
            "height": DEFAULT_PLOT_HEIGHT,
            "bins_box": bins_box,
            "view_min_box": view_min_box,
            "view_max_box": view_max_box,
            "drag": None,
            "lower_line": None,
            "upper_line": None,
            "span": None,
        }
        self._hist_widgets[column] = state

        bins_box.valueChanged.connect(lambda _v, c=column: self._draw_histogram(c))
        view_min_box.valueChanged.connect(lambda _v, c=column: self._draw_histogram(c))
        view_max_box.valueChanged.connect(lambda _v, c=column: self._draw_histogram(c))

        canvas.mpl_connect("button_press_event", lambda evt, c=column: self._on_hist_press(c, evt))
        canvas.mpl_connect("motion_notify_event", lambda evt, c=column: self._on_hist_motion(c, evt))
        canvas.mpl_connect("button_release_event", lambda evt, c=column: self._on_hist_release(c, evt))

        self._draw_histogram(column)
        return container

    # ------------------------------------------------------------------
    # Lifting a single graph out for a slide
    # ------------------------------------------------------------------
    def _start_new_output_folders(self):
        """Forget the folders this session was collecting outputs into.

        Both are remembered so that several things saved from one analysis land
        together - an image and a movie from the same render, a set of graphs for
        one figure. That is only right while the analysis is the same one; the
        moment the data changes they have to be given up, or the next run's
        outputs are filed under the previous run's timestamp.
        """
        self._render_save_folder = None
        self._figure_save_folder = None

    def _figure_save_dir(self):
        """One dated folder per session for the graphs lifted out of the panel.

        A dialog per graph is the thing that stops anyone building a figure set:
        eight plots is eight trips through a file browser. One click writes it,
        and they accumulate somewhere findable together.
        """
        folder = getattr(self, "_figure_save_folder", None)
        if folder is None or not folder.exists():
            folder = self._make_analysis_folder(self._analysis_base_dir(), "figures")
            folder.mkdir(parents=True, exist_ok=True)
            self._figure_save_folder = folder
        return folder

    def _save_figure_png(self, figure, name):
        """Write one figure as a transparent PNG, ready to drop on a slide.

        Transparent rather than black: the plots are drawn light-on-dark for
        napari, so with the background dropped the axes, labels and data keep
        their light colours and sit on whatever the slide provides. Saved at
        twice the screen resolution, because a histogram that looks fine in a
        panel is a blurred rectangle on a projector.
        """
        try:
            folder = self._figure_save_dir()
            stem = self._safe_filename(name) or "figure"
            path = folder / f"{stem}.png"
            i = 2
            while path.exists():
                path = folder / f"{stem}_{i}.png"
                i += 1
            figure.savefig(path, dpi=FIGURE_SAVE_DPI, transparent=True,
                           bbox_inches="tight", pad_inches=0.05)
        except Exception as exc:
            self.log(f"Could not save that graph: {exc}")
            return None
        self.log(f"Saved {path.name} to {path.parent}")
        return path

    def _png_button(self, redraw, name_getter):
        """The small button that opens the export dialog, for one graph."""
        button = QPushButton("Save…")
        button.setProperty("secondary", True)
        button.setMaximumWidth(60)
        button.setToolTip(
            "Open this graph in the export window: choose its size, shape, font "
            "and format against a live preview, add counting error bars, and "
            "drop the background for a slide.\n\n"
            "Nothing chosen there changes the plot on screen - the panel is "
            "sized for reading, a figure is sized for wherever it is going."
        )
        button.clicked.connect(
            lambda _c=False: self._open_figure_export(redraw, name_getter()))
        return button

    def _open_figure_export(self, redraw, name):
        dialog = FigureExportDialog(self, redraw, name, self._figure_save_dir())
        if dialog.exec() and dialog.saved_path() is not None:
            path = dialog.saved_path()
            self.log(f"Saved {path.name} to {path.parent}")

    def _build_plot_size_row(self):
        """One size for every plot in the plugin.

        These end up in talks, and a figure that is the right shape is most of
        what makes one look deliberate - but resizing a dozen of them by hand is
        exactly the effort nobody spends. So it is one control, applied live, and
        saved with the run.
        """
        row = QHBoxLayout()
        _min_w, _max_w, min_h, max_h = PLOT_SIZE_LIMITS
        row.addWidget(QLabel("Shape"))
        self.plot_aspect_box = QComboBox()
        for label, ratio in PLOT_ASPECTS:
            self.plot_aspect_box.addItem(label, ratio)
        self.plot_aspect_box.setToolTip(
            "The shape of every plot. 'Fill the panel' is the default and lets "
            "them stretch as the window does; any ratio pins the width to the "
            "height, so a graph saved for a slide is the shape you chose rather "
            "than the one the window happened to have."
        )
        row.addWidget(self.plot_aspect_box)
        row.addWidget(QLabel("Height"))
        self.plot_height_box = QSpinBox()
        self.plot_height_box.setRange(min_h, max_h)
        self.plot_height_box.setValue(DEFAULT_PLOT_HEIGHT)
        self.plot_height_box.setSingleStep(20)
        self.plot_height_box.setSuffix(" px")
        row.addWidget(self.plot_height_box)
        row.addWidget(QLabel("Font"))
        self.plot_font_box = QSpinBox()
        self.plot_font_box.setRange(4, 40)
        self.plot_font_box.setValue(int(plot_font()))
        self.plot_font_box.setSuffix(" pt")
        self.plot_font_box.setToolTip(
            "Body text in every plot. Ticks sit a point below it and titles a "
            "point above, so one number rescales the lot.\n\n"
            "The default suits a side panel; a graph going on a slide usually "
            "wants 12-16."
        )
        row.addWidget(self.plot_font_box)
        row.addStretch(1)
        self.plot_aspect_box.currentIndexChanged.connect(lambda _i: self._apply_plot_size())
        self.plot_height_box.valueChanged.connect(lambda _v: self._apply_plot_size())
        self.plot_font_box.valueChanged.connect(lambda _v: self._apply_plot_size())
        return row

    def _plot_pixel_size(self):
        """(width, height) in pixels; width 0 when the panel decides it."""
        height = int(self.plot_height_box.value())
        ratio = self.plot_aspect_box.currentData()
        width = int(round(height * float(ratio))) if ratio else PLOT_WIDTH_FILL
        return width, height

    def _set_plot_size(self, width, height):
        """Set the height, and the nearest offered shape to width/height."""
        self.plot_height_box.setValue(int(height))
        if not width:
            self.plot_aspect_box.setCurrentIndex(0)
            return
        wanted = float(width) / max(int(height), 1)
        best = min(range(1, self.plot_aspect_box.count()),
                   key=lambda i: abs(self.plot_aspect_box.itemData(i) - wanted))
        self.plot_aspect_box.setCurrentIndex(best)

    def _apply_plot_size(self):
        """Push the chosen shape and font onto every canvas, and redraw."""
        if not hasattr(self, "plot_aspect_box"):
            return  # a canvas built before the control that sizes them
        set_plot_font_size(self.plot_font_box.value())
        width, height = self._plot_pixel_size()
        for canvas in list(self._plot_canvases):
            try:
                if width > PLOT_WIDTH_FILL:
                    canvas.setFixedWidth(width)
                else:
                    # Back to filling the panel: undo the pin in both directions,
                    # or the canvas keeps whatever width it was last given.
                    canvas.setMinimumWidth(0)
                    canvas.setMaximumWidth(16777215)
                canvas.setFixedHeight(height)
                canvas.updateGeometry()
                # The figure's own size is what savefig uses, and it otherwise
                # only follows the widget on a Qt layout pass - so a graph saved
                # for a slide would come out the shape it was before the size
                # was chosen. Set both and they cannot disagree.
                figure = canvas.figure
                dpi = figure.get_dpi() or 100.0
                pixels = width if width > PLOT_WIDTH_FILL else max(canvas.width(), 1)
                figure.set_size_inches(pixels / dpi, height / dpi, forward=False)
                canvas.draw_idle()
            except RuntimeError:
                # Its Qt object is gone - a filter panel rebuilt for new data.
                self._plot_canvases.remove(canvas)
        # The font sizes are applied while the contents are drawn, so the
        # artists already on a canvas keep the old ones until it is redrawn.
        self._redraw_all_plots()

    def _redraw_all_plots(self):
        for column in list(self._hist_widgets):
            self._draw_histogram(column)
        for key in list(self._metric_hist_widgets):
            self._draw_metric_histogram(key)
        if hasattr(self, "msd_figure"):
            self._draw_msd_validation()
        if hasattr(self, "loc_counts_figure"):
            self._draw_loc2d_counts()
        if hasattr(self, "drift_figure"):
            self._draw_drift_plot()
        if hasattr(self, "rcc_figure"):
            self._draw_rcc_plot()
        if hasattr(self, "population_figure"):
            self._draw_population_plot()

    def _draw_histogram(self, column, figure=None, opts=None):
        """Draw one filter histogram.

        With `figure` given it draws into that instead of the panel's canvas and
        leaves the interactive state alone - which is what the export preview
        uses, so that choosing a size for a slide cannot disturb the plot being
        read on screen.
        """
        state = self._hist_widgets.get(column)
        if not state or self.df is None:
            return
        export = figure is not None
        opts = opts or {}
        values = self.df[column].dropna().to_numpy(float)
        # Two distributions, not one: everything loaded, and what survives the
        # filters. Drawing only the first meant tightening a bound on sigma
        # changed nothing visible in the intensity histogram beside it - and the
        # coupling between columns is exactly what these plots are for.
        if (self.df_filtered is not None and not self.df_filtered.empty
                and column in self.df_filtered.columns):
            kept = self.df_filtered[column].dropna().to_numpy(float)
        else:
            kept = values if self.df_filtered is None else values[:0]
        figure = figure if export else state["figure"]
        figure.clear()
        figure.patch.set_facecolor(FILTER_HIST_BG)
        ax = figure.add_subplot(111)
        ax.set_facecolor(FILTER_HIST_BG)

        intensity_col = self._resolve_column("intensity")
        n_bins = state["bins_box"].value()
        view_lo = state["view_min_box"].value()
        view_hi = state["view_max_box"].value()
        if view_hi <= view_lo:
            view_hi = view_lo + 1e-9

        if column == intensity_col:
            lo = max(view_lo, 1e-9)
            bins = np.logspace(np.log10(lo), np.log10(view_hi), n_bins + 1)
            in_view = lambda a: a[(a > 0) & (a >= lo) & (a <= view_hi)]  # noqa: E731
            ax.set_xscale("log")
            ax.set_xlim(lo, view_hi)
        else:
            bins = np.linspace(view_lo, view_hi, n_bins + 1)
            in_view = lambda a: a[(a >= view_lo) & (a <= view_hi)]  # noqa: E731
            ax.set_xlim(view_lo, view_hi)

        # Same bin edges for both, so the pale bars read as "what the filters
        # removed" rather than as a second, differently-binned distribution.
        all_shown, kept_shown = in_view(values), in_view(kept)
        if len(all_shown):
            ax.hist(all_shown, bins=bins, color=FILTER_HIST_BAR, alpha=0.28)
        if len(kept_shown):
            counts, _edges, _patches = ax.hist(
                kept_shown, bins=bins, color=FILTER_HIST_BAR, alpha=0.95)
            if opts.get("errorbars"):
                # Counting error on a histogram bin is sqrt(n) - the only error
                # bar a histogram of independent samples honestly has.
                centres = 0.5 * (bins[:-1] + bins[1:])
                ax.errorbar(centres, counts, yerr=np.sqrt(np.maximum(counts, 0)),
                            fmt="none", ecolor=FILTER_HIST_FG, elinewidth=0.9,
                            capsize=2, alpha=0.7)
        if opts.get("grid"):
            ax.grid(color=FILTER_HIST_FG, alpha=0.18, linewidth=0.5)
            ax.set_axisbelow(True)
        if opts.get("ylabel", export):
            ax.set_ylabel("Count", fontsize=plot_font(), color=FILTER_HIST_FG)

        title = column
        if len(kept) != len(values):
            title = f"{column}   {len(kept)} / {len(values)}"
        if opts.get("title", True):
            ax.set_title(title, fontsize=plot_font(1), color=FILTER_HIST_FG)
        ax.tick_params(labelsize=plot_font(-1), colors=FILTER_HIST_FG)
        for spine in ax.spines.values():
            spine.set_color(FILTER_HIST_FG)
            spine.set_alpha(0.4)
        figure.tight_layout()

        lower_box, upper_box = self.filter_controls.get(column, (None, None))
        show_bounds = opts.get("bounds", True)
        if lower_box is not None and upper_box is not None and show_bounds:
            lower, upper = lower_box.value(), upper_box.value()
            span = ax.axvspan(lower, upper, color=FILTER_HIST_LINE, alpha=0.18, zorder=0)
            low_line = ax.axvline(lower, color=FILTER_HIST_LINE, linewidth=1.5)
            high_line = ax.axvline(upper, color=FILTER_HIST_LINE, linewidth=1.5)
        else:
            span = low_line = high_line = None
        if export:
            return                      # the panel's drag handles stay as they were
        state["span"], state["lower_line"], state["upper_line"] = span, low_line, high_line
        state["canvas"].draw_idle()

    def _sync_histogram_lines(self, column):
        state = self._hist_widgets.get(column)
        if not state or not state["figure"].axes:
            return
        lower_box, upper_box = self.filter_controls.get(column, (None, None))
        if lower_box is None or upper_box is None:
            return
        ax = state["figure"].axes[0]
        lower, upper = lower_box.value(), upper_box.value()
        if state.get("span") is not None:
            state["span"].remove()
        state["span"] = ax.axvspan(lower, upper, color=FILTER_HIST_LINE, alpha=0.18, zorder=0)
        if state.get("lower_line") is not None:
            state["lower_line"].set_xdata([lower, lower])
        if state.get("upper_line") is not None:
            state["upper_line"].set_xdata([upper, upper])
        state["canvas"].draw_idle()

    def _refresh_histogram_bounds(self):
        """Redraw every filter histogram against the surviving localizations.

        A full redraw rather than only moving the bound lines: the bars have to
        move too, or filtering one column leaves every other distribution
        looking untouched. Called from `apply_filters`, which runs on a button
        press or the release of a dragged bound - never per keystroke - so the
        cost of re-binning each column is paid once per deliberate action.
        """
        for column in self._hist_widgets:
            self._draw_histogram(column)

    def _on_hist_press(self, column, event):
        state = self._hist_widgets.get(column)
        if not state or event.xdata is None or not state["figure"].axes:
            return
        lower_line, upper_line = state.get("lower_line"), state.get("upper_line")
        if lower_line is None or upper_line is None:
            return
        lower_x = lower_line.get_xdata()[0]
        upper_x = upper_line.get_xdata()[0]
        xlim = state["figure"].axes[0].get_xlim()
        tol = 0.03 * (xlim[1] - xlim[0])
        dist_lower = abs(event.xdata - lower_x)
        dist_upper = abs(event.xdata - upper_x)
        if dist_lower <= tol and dist_lower <= dist_upper:
            state["drag"] = "lower"
        elif dist_upper <= tol:
            state["drag"] = "upper"
        else:
            state["drag"] = None

    def _on_hist_motion(self, column, event):
        state = self._hist_widgets.get(column)
        if not state or state.get("drag") is None or event.xdata is None:
            return
        lower_box, upper_box = self.filter_controls.get(column, (None, None))
        if lower_box is None or upper_box is None:
            return
        if state["drag"] == "lower":
            value = min(event.xdata, upper_box.value())
            lower_box.setValue(value)
        else:
            value = max(event.xdata, lower_box.value())
            upper_box.setValue(value)
        # lower_box/upper_box.valueChanged -> _sync_histogram_lines already
        # redraws the bar/lines, so nothing else to do here.

    def _on_hist_release(self, column, event):
        state = self._hist_widgets.get(column)
        if not state or state.get("drag") is None:
            return
        state["drag"] = None
        self.apply_filters()
