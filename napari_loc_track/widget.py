import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

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
from . import _session as session_io
from ._tracks import (
    DEFAULT_LINKING_ERROR_RATE,
    filter_tracks_by_length,
    iter_particle_batches,
    max_linkable_diffusion,
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
AMBER = "#e8a33d"           # reserved for "this stops something" and warnings
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

DEFAULT_HIST_HEIGHT = 190

# Every plot in the plugin answers to one size, because these end up in talks
# and a figure that is the right shape is most of what makes one look
# deliberate. Width 0 means "fill the panel", which is the behaviour these had
# before and stays the default; anything else pins it, so a screenshot has the
# aspect ratio you chose rather than the one the window happened to have.
PLOT_WIDTH_FILL = 0
PLOT_SIZE_LIMITS = (240, 3000, 120, 1600)   # min/max width, min/max height
HIST_HEIGHT_STEP = 50
MIN_HIST_HEIGHT = 110
MAX_HIST_HEIGHT = 600

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
    "pstatic": "p (consistent with static)",
    "dmin": "Smallest detectable D (µm²/s)",
    # Colouring only: time needs no computing and has no bounds to filter on, so
    # it is absent from METRIC_CACHE_ATTR and from the histogram/bounds machinery.
    "time": "Frame first seen",
}
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

# The settings that describe the instrument rather than a choice about the
# analysis. Restoring a previous run moves these along with everything else -
# correctly, since the loaded localizations were computed with them - but doing
# it silently means a corrected calibration can be reverted by opening a folder,
# and nothing on screen says so. Every one of these that a restore changes is
# named in the log.
INSTRUMENT_SETTINGS = (
    ("pixel_size_box", "Pixel size", "{:.1f} nm/px"),
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
        ax.tick_params(labelsize=7, colors=INK)
        for spine in ax.spines.values():
            spine.set_color(PANEL_LINE)
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_color(INK)
            label.set_fontsize(8)

    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(PLOT_BG)
        # Light enough to read off a projected slide, where the dimmed grey that
        # suits a screen at arm's length disappears entirely.
        ax.tick_params(labelsize=7, colors=INK)
        ax.grid(color=PANEL_LINE, linestyle="-", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(PANEL_LINE)
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_color(INK)
            label.set_fontsize(8)
        if title is not None:
            ax.set_title(title, fontsize=9, color=INK)


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
    (("smlm_rendering", "add_layer_to_viewer"), "render_add_layer_box"),
    (("smlm_rendering", "layer_name"), "render_layer_name_edit"),
    (("smlm_rendering", "population_split_p"), "render_population_p_box"),
    (("smlm_rendering", "write_png_snapshot"), "render_png_box"),
    (("smlm_rendering", "image_save_format"), "render_image_format_box"),
    (("smlm_rendering", "movie_save_format"), "render_movie_format_box"),
    (("smlm_rendering", "movie_save_stride"), "movie_stride_box"),
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
    (("rendering", "plot_width_px"), "plot_width_box"),
    (("rendering", "plot_height_px"), "plot_height_box"),
)


# Which acquisition-metadata field fills which control when a stack is loaded,
# and how the change is worded in the log. Only fields the microscope genuinely
# recorded reach this table - `read_acquisition_metadata` omits the rest - so a
# control whose value was never calibrated keeps whatever it had.
ACQUISITION_AUTOFILL = (
    ("pixel_size_nm", "pixel_size_box", "Pixel size", "{:.1f} nm/px"),
    ("fps", "fps_box", "Frame rate", "{:.3f} fps"),
    ("camera_offset_adu", "loc_offset_box", "Camera offset", "{:.0f} ADU"),
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

    return values, notes


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
        # Where the trajectories came from, when they were read rather than
        # linked: a session re-links its own trajectories but must not re-link
        # someone else's, which these parameters would not reproduce.
        self._tracks_source_path = None
        # The queue of restore steps while a session is being reloaded, and None
        # at every other moment - the completion hooks test it to tell an
        # ordinary load from one step of a restore.
        self._session_restore = None
        self._session_save_worker_ref = None
        self._autosave_worker_ref = None
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
        self._build_filter_tab()
        # Track before Render: a reconstruction is now as often built from a
        # dynamics selection as from every localization, and you cannot make
        # that selection until the trajectories and their metrics exist. The
        # tab order is the order the work is done in.
        self._build_track_tab()
        self._build_render_tab()
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
            shape = getattr(getattr(layer, "data", None), "shape", None)
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
        """Restore the parameters recorded in a previous run's metadata.json."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load settings from a previous analysis",
                filter="Analysis metadata (metadata.json);;JSON files (*.json)",
            )
        if not path:
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                metadata = json.load(handle)
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
        return view

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
                "render_image": self._render_image is not None,
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

        locs_frame = self.df if manifest["localizations_saved_with_session"] else None
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

        self._session_restore = {
            "manifest": manifest,
            "session_dir": session_dir,
            "name": path.name,
            "steps": ["link", "diffusion", "render", "view"],
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
            elif step == "render" and self._session_render_wanted(plan):
                self.render_smlm_image()
                if self._render_worker_ref is not None:
                    return
            elif step == "view":
                self._restore_session_view(plan["manifest"].get("view") or {})
        self._finish_session_restore()

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

        bounds = metadata.get("filter_bounds") if isinstance(metadata, dict) else None
        if isinstance(bounds, dict):
            n_bounds, unmatched = self._apply_filter_bounds(bounds)
            if n_bounds:
                notes.append(f"{n_bounds} filter bound(s) applied")
            if unmatched:
                shown = ", ".join(sorted(unmatched)[:3])
                notes.append(
                    f"{len(unmatched)} filter bound(s) match no loaded column ({shown}"
                    + (", ...)" if len(unmatched) > 3 else ")")
                    + " - kept in case matching data is loaded next"
                )

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
        layout = QVBoxLayout(tab)

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
        self.pixel_size_box.setValue(161.0)
        self.pixel_size_box.setDecimals(1)
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

        self.load_settings_button = QPushButton("Load settings from a previous analysis...")
        self.load_settings_button.setProperty("secondary", True)
        self.load_settings_button.clicked.connect(lambda: self.load_settings_from_metadata())
        self.load_settings_button.setToolTip(
            "Read the metadata.json written by a previous export and restore the\n"
            "parameters it recorded: camera, detection, fitting, linking, diffusion,\n"
            "filter bounds and display settings.\n\n"
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
            ("pstatic", "p (consistent with static) — filter to p > 0.05 for the "
                        "immobile population", 0.0, 1.0, 6),
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
            "p (consistent with static)",
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
        # Applies to every plot in the plugin, not only the ones on this tab -
        # the filter histograms and the MSD validation follow it too.
        color_layout.addRow("Plot size", self._build_plot_size_row())
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
        self.apply_filters()

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
        # Loading can outrun the debounce on the binning box: the stack has just
        # been opened at the new factor, so the baseline and frame rate move with
        # it here rather than waiting for a timer that would then find the factor
        # already applied and do nothing. Before the autofill, which reads it.
        self._time_bin_timer.stop()
        factor = int(self.bin_factor_box.value())
        if factor != self._time_bin_applied:
            self._rescale_for_time_binning(max(1, int(self._time_bin_applied)), factor)
        self._time_bin_applied = factor
        self.viewer.layers.clear()
        if image is not None:
            self._image_layer_name = Path(image_path).name
            self.viewer.add_image(
                image, name=self._image_layer_name, colormap="gray",
                **self._placed({}, image.ndim))
            self.log(f"Image {tuple(image.shape)} {image.dtype}: {how}")
            # The pixel size may move here, so the units follow rather than lead.
            self._apply_acquisition_metadata(acquisition)
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
            box.setValue(recorded)
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
                    f"{self.pixel_size_box.value():.1f} nm/px was left as it was")
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
            # Nothing open yet; the factor is picked up by the next load.
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
        self.tracks = None
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
        stack = layer.data
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

    def _draw_loc2d_counts(self):
        figure = self.loc_counts_figure
        figure.clear()
        if self._loc2d_counts is None or len(self._loc2d_counts) == 0:
            figure.patch.set_facecolor(PANEL_BG)
            self.loc_counts_canvas.draw_idle()
            return
        ax = figure.add_subplot(111)
        ax.plot(np.arange(len(self._loc2d_counts)), self._loc2d_counts,
                color=ACCENT, linewidth=1.2)
        ax.fill_between(np.arange(len(self._loc2d_counts)), self._loc2d_counts,
                        color=ACCENT, alpha=0.18)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Detections")
        style_axes(figure, ax, title="Detections vs frame")
        figure.tight_layout()
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
        matter of ticking a box rather than of exporting and re-importing.
        """
        df = self._displayed_localizations()
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
        df = self._displayed_localizations()
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
            return (int(shape[-2]), int(shape[-1])), (-0.5, -0.5), layer

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

        if mode == "gaussian_global":
            options["global_sigma_px"] = self.render_sigma_box.value() / pixel_size
            info["global_sigma_nm"] = self.render_sigma_box.value()
        elif mode == "gaussian_local":
            column = self.render_sigma_column_box.currentText()
            if not column or column not in self.df_filtered.columns:
                self.log(
                    "No localization-precision column to take the width from - "
                    "pick another render mode, or one of the columns in the list."
                )
                return None
            low = self.render_sigma_min_box.value()
            high = max(self.render_sigma_max_box.value(), low)
            widths = np.clip(self.df_filtered[column].to_numpy(dtype=float), low, high)
            options["sigma_px"] = widths / pixel_size
            info.update({"sigma_column": column, "sigma_clamp_nm": [low, high]})

        if mode != "scatter" and self.render_photons_box.isChecked():
            column = self._resolve_column("intensity")
            if column and column in self.df_filtered.columns:
                options["weights"] = self.df_filtered[column].to_numpy(dtype=float)
                info["weighted_by"] = column
                info["value_units"] = "photons per pixel"
            else:
                self.log("No photon-count column found - rendering unweighted counts")

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
        """Where this render lands. A name per selection, so they accumulate."""
        base = self.render_layer_name_edit.text().strip() or RENDER_LAYER_NAME
        return base if kind == "image" else f"{base}_movie"

    def _render_population_label(self):
        """What this render was built from, recorded with the layer."""
        active = self._active_metric_filters()
        if not active:
            return "all localizations"
        return "; ".join(f"{METRIC_LABELS[key].split(' (')[0]} {low:g}-{high:g}"
                         for key, low, high in active)

    def _set_render_population(self, which):
        """Point the dynamics filter at a named population, and name the layer.

        The two presets are the common case and the reason the feature exists:
        one reconstruction of the molecules that stayed put and one of the
        molecules that moved, from a single acquisition, in separate layers that
        blend additively so they can be read together or alone.
        """
        threshold = self.render_population_p_box.value()
        boxes = self._metric_filter_boxes
        for key, box in boxes.items():
            if key != "pstatic":
                box.blockSignals(True)
                box.setChecked(False)
                box.blockSignals(False)

        pstatic = boxes.get("pstatic")
        if which == "all":
            pstatic.blockSignals(True)
            pstatic.setChecked(False)
            pstatic.blockSignals(False)
            self.render_layer_name_edit.setText(RENDER_LAYER_NAME)
        else:
            if which == "immobile":
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

    def _build_render_population_group(self):
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
             "Trajectories consistent with a molecule that never moved."),
            ("Mobile", "mobile",
             "Trajectories that moved further than their own localization error."),
            ("All", "all", "Clear the dynamics filter and render everything."),
        ):
            button = QPushButton(label)
            button.setProperty("secondary", True)
            button.setToolTip(tip)
            button.clicked.connect(lambda _c, w=which: self._set_render_population(w))
            row.addWidget(button)
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
                          "dynamics_selection": self._render_population_label()},
            )
        except Exception as exc:
            self.log(f"Could not add the render to the viewer: {exc}")
            return

    # --- saving a render ------------------------------------------------
    def _default_render_path(self, kind, info, force_format=None):
        csv_path = self.csv_edit.text().strip()
        image_path = self.image_edit.text().strip()
        if csv_path:
            base = Path(csv_path)
        elif image_path:
            base = Path(image_path)
        else:
            base = Path.cwd() / "localizations"
        suffix = "movie" if kind == "movie" else "render"
        box = self.render_movie_format_box if kind == "movie" else self.render_image_format_box
        # The format is in the name so a composite and the data it came from
        # never overwrite each other.
        save_format = force_format or box.currentData()
        name = f"{base.stem}_{suffix}_{info['mode']}_os{info['oversampling']}_{save_format}.tif"
        return str(base.parent / name)

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
                stack = np.asarray(data)
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
                               or c in {s["canvas"] for s in self._metric_hist_widgets.values()}]

        if self.df is None:
            self.filter_layout.addWidget(QLabel("Load data to see filters"), 0, 0)
            return

        x_col = self._resolve_column("x")
        y_col = self._resolve_column("y")
        numeric_columns = [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]
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
                    f"{len(unmatched)} saved filter bound(s) match no column here: "
                    + ", ".join(sorted(unmatched)[:3])
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
        self._invalidate_tracks(reason="filters changed")
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

    def _invalidate_tracks(self, reason=""):
        # Localizations that fall outside the current filters/ROI must not
        # keep showing up as part of "already linked" trajectories: those
        # trajectories were computed from a different (usually larger) set
        # of localizations, so they'd extend past the new bounds. Rather
        # than silently showing stale/inconsistent tracks, clear them and
        # require an explicit re-link.
        if self.tracks is None:
            return
        self.tracks = None
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
        if reason:
            self.log(f'Trajectories cleared ({reason}) - click "Link trajectories" again.')

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
        worker.returned.connect(self._on_link_finished)
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

    def _on_link_finished(self, linked):
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

        bar = getattr(self.viewer, "scale_bar", None)
        if bar is None:
            return
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
        tracks = filter_tracks_by_length(self.tracks, min_length)
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
        tracks = self.tracks
        _label, sigma_px, _measured = self._sigma_source()
        if sigma_px is not None and len(sigma_px) == len(tracks):
            tracks = tracks.assign(**{SIGMA_COLUMN: sigma_px})
        worker = _fit_free_metrics_worker(
            tracks, self.pixel_size_box.value(), self.fps_box.value(),
            alpha=self.immobility_alpha_box.value())
        worker.returned.connect(self._on_fit_free_metrics_finished)
        worker.errored.connect(lambda exc: self.log(f"Distance/duration computation failed: {exc}"))
        self._metrics_worker_ref = worker
        worker.start()

    def _on_fit_free_metrics_finished(self, result):
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
        if choice.startswith("p ("):
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
        if not active:
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
        if not active:
            return "No dynamics filter - every trajectory is shown."
        if self.tracks is None or self.tracks.empty:
            return "No trajectories to filter yet - link some first."
        passing = self._passing_particles()
        n_total = int(self.tracks["particle"].nunique())
        n_kept = len(passing)
        criteria = ", ".join(
            f"{METRIC_LABELS[key].split(' (')[0]} {low:g}-{high:g}"
            for key, low, high in active)
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
        box = self._metric_filter_boxes.get(key)
        if box is not None:
            low_box, high_box = self._metric_bound_boxes[key]
            state = "on" if box.isChecked() else "off"
            self.log(f"Dynamics filter on {METRIC_LABELS[key]} {state}"
                     + (f" ({low_box.value():g} to {high_box.value():g})"
                        if box.isChecked() else ""))
        self._apply_track_filter()

    def clear_track_filters(self):
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

    def _draw_metric_histogram(self, key):
        state = self._metric_hist_widgets.get(key)
        if not state:
            return
        figure = state["figure"]
        figure.clear()
        state["lower_line"] = None
        state["upper_line"] = None
        state["span"] = None
        cache = self._metric_cache(key)
        if not cache:
            state["canvas"].draw_idle()
            return
        values = np.asarray(list(cache.values()), float)
        values = values[np.isfinite(values)]
        if not len(values):
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
        state["span"] = ax.axvspan(lower, upper, color=LAVENDER, alpha=0.15, zorder=0)
        state["lower_line"] = ax.axvline(lower, color=LAVENDER, linewidth=1.5)
        state["upper_line"] = ax.axvline(upper, color=LAVENDER, linewidth=1.5)

        ax.set_xlabel(METRIC_LABELS[key])
        ax.set_ylabel("Count")
        style_axes(figure, ax, title=f"{len(values)} trajectories")
        figure.tight_layout()
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

    def _draw_msd_validation(self):
        figure = self.msd_figure
        figure.clear()
        ax = figure.add_subplot(111)
        if not self._track_msd_cache:
            style_axes(figure, ax)
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
                   title=f"MSD fit validation ({n_sample} example trajectories)")
        legend = ax.legend(fontsize=6, loc="upper left", ncol=2,
                           facecolor=PLOT_BG, edgecolor=PANEL_LINE, labelcolor=INK)
        legend.get_frame().set_alpha(0.85)
        figure.tight_layout()
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

    def _analysis_base_dir(self):
        """The folder results go beside: the data's, not the plugin's.

        Prefer the localization CSV's folder, then the image's - an in-app fit
        has no CSV - and only fall back to the working directory if neither is
        known, so results never end up outside the dataset they describe.
        """
        csv_path = self.csv_edit.text().strip()
        image_path = self.image_edit.text().strip()
        if csv_path:
            return Path(csv_path).parent
        if image_path:
            return Path(image_path).parent
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
        return pd.DataFrame([
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
            "n_localizations_total": len(self.df) if self.df is not None else 0,
            "n_localizations_filtered": len(self.df_filtered) if self.df_filtered is not None else 0,
            "filter_bounds": {
                col: {"min": lo.value(), "max": hi.value()} for col, (lo, hi) in self.filter_controls.items()
            },
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
                "dynamics_selection": self._render_population_label(),
                "write_png_snapshot": self.render_png_box.isChecked(),
                "image_save_format": self.render_image_format_box.currentData(),
                "movie_save_format": self.render_movie_format_box.currentData(),
                "movie_save_stride": self.movie_stride_box.value(),
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
                "plot_width_px": self.plot_width_box.value(),
                "plot_height_px": self.plot_height_box.value(),
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
        shrink_btn = QToolButton()
        shrink_btn.setText("-")
        grow_btn = QToolButton()
        grow_btn.setText("+")
        reset_btn = QToolButton()
        reset_btn.setText("Reset")
        toolbar.addWidget(shrink_btn)
        toolbar.addWidget(grow_btn)
        toolbar.addWidget(reset_btn)
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

        figure = Figure(figsize=(4.2, DEFAULT_HIST_HEIGHT / 100))
        canvas = FigureCanvas(figure)
        self._plot_canvases.append(canvas)
        canvas.setMinimumHeight(DEFAULT_HIST_HEIGHT)
        canvas.setMaximumHeight(DEFAULT_HIST_HEIGHT)
        canvas.setMinimumWidth(260)
        layout.addWidget(canvas)

        state = {
            "figure": figure,
            "canvas": canvas,
            "height": DEFAULT_HIST_HEIGHT,
            "bins_box": bins_box,
            "view_min_box": view_min_box,
            "view_max_box": view_max_box,
            "drag": None,
            "lower_line": None,
            "upper_line": None,
            "span": None,
        }
        self._hist_widgets[column] = state

        shrink_btn.clicked.connect(lambda checked=False, c=column: self._resize_histogram(c, -HIST_HEIGHT_STEP))
        grow_btn.clicked.connect(lambda checked=False, c=column: self._resize_histogram(c, HIST_HEIGHT_STEP))
        reset_btn.clicked.connect(lambda checked=False, c=column: self._resize_histogram(c, None))
        bins_box.valueChanged.connect(lambda _v, c=column: self._draw_histogram(c))
        view_min_box.valueChanged.connect(lambda _v, c=column: self._draw_histogram(c))
        view_max_box.valueChanged.connect(lambda _v, c=column: self._draw_histogram(c))

        canvas.mpl_connect("button_press_event", lambda evt, c=column: self._on_hist_press(c, evt))
        canvas.mpl_connect("motion_notify_event", lambda evt, c=column: self._on_hist_motion(c, evt))
        canvas.mpl_connect("button_release_event", lambda evt, c=column: self._on_hist_release(c, evt))

        self._draw_histogram(column)
        return container

    def _build_plot_size_row(self):
        """One size for every plot in the plugin.

        These end up in talks, and a figure that is the right shape is most of
        what makes one look deliberate - but resizing a dozen of them by hand is
        exactly the effort nobody spends. So it is one control, applied live, and
        saved with the run.
        """
        row = QHBoxLayout()
        min_w, max_w, min_h, max_h = PLOT_SIZE_LIMITS
        row.addWidget(QLabel("Width"))
        self.plot_width_box = QSpinBox()
        self.plot_width_box.setRange(0, max_w)
        self.plot_width_box.setValue(PLOT_WIDTH_FILL)
        self.plot_width_box.setSpecialValueText("fill")   # 0 reads as "fill"
        self.plot_width_box.setSingleStep(20)
        self.plot_width_box.setSuffix(" px")
        self.plot_width_box.setToolTip(
            f"Width of every plot. Leave at 'fill' and they stretch to the panel "
            f"as they always have; set a width ({min_w}-{max_w} px) and they hold "
            "it, so a screenshot has the aspect ratio you chose rather than the "
            "one the window happened to have."
        )
        row.addWidget(self.plot_width_box)
        row.addWidget(QLabel("Height"))
        self.plot_height_box = QSpinBox()
        self.plot_height_box.setRange(min_h, max_h)
        self.plot_height_box.setValue(DEFAULT_HIST_HEIGHT)
        self.plot_height_box.setSingleStep(20)
        self.plot_height_box.setSuffix(" px")
        row.addWidget(self.plot_height_box)
        for preset, width, height in (("16:9", 960, 540), ("4:3", 800, 600),
                                      ("Wide", 1200, 400)):
            button = QPushButton(preset)
            button.setProperty("secondary", True)
            button.setToolTip(f"{width} x {height} px")
            button.clicked.connect(
                lambda _c, w=width, h=height: self._set_plot_size(w, h))
            row.addWidget(button)
        row.addStretch(1)
        self.plot_width_box.valueChanged.connect(lambda _v: self._apply_plot_size())
        self.plot_height_box.valueChanged.connect(lambda _v: self._apply_plot_size())
        return row

    def _set_plot_size(self, width, height):
        self.plot_width_box.setValue(int(width))
        self.plot_height_box.setValue(int(height))

    def _apply_plot_size(self):
        """Push the chosen size onto every canvas, and redraw so labels re-fit."""
        if not hasattr(self, "plot_width_box"):
            return  # a canvas built before the control that sizes them
        width = int(self.plot_width_box.value())
        height = int(self.plot_height_box.value())
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
                canvas.draw_idle()
            except RuntimeError:
                # Its Qt object is gone - a filter panel rebuilt for new data.
                self._plot_canvases.remove(canvas)
        # The per-column height buttons work from this as their baseline, so a
        # plot nudged by hand starts from the size everything else is now.
        for state in self._hist_widgets.values():
            state["height"] = height

    def _resize_histogram(self, column, delta):
        state = self._hist_widgets.get(column)
        if not state:
            return
        if delta is None:
            state["height"] = DEFAULT_HIST_HEIGHT
        else:
            state["height"] = int(np.clip(state["height"] + delta, MIN_HIST_HEIGHT, MAX_HIST_HEIGHT))
        state["canvas"].setMinimumHeight(state["height"])
        state["canvas"].setMaximumHeight(state["height"])
        state["canvas"].updateGeometry()

    def _draw_histogram(self, column):
        state = self._hist_widgets.get(column)
        if not state or self.df is None:
            return
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
        figure = state["figure"]
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
            ax.hist(kept_shown, bins=bins, color=FILTER_HIST_BAR, alpha=0.95)

        title = column
        if len(kept) != len(values):
            title = f"{column}   {len(kept)} / {len(values)}"
        ax.set_title(title, fontsize=9, color=FILTER_HIST_FG)
        ax.tick_params(labelsize=7, colors=FILTER_HIST_FG)
        for spine in ax.spines.values():
            spine.set_color(FILTER_HIST_FG)
            spine.set_alpha(0.4)
        figure.tight_layout()

        lower_box, upper_box = self.filter_controls.get(column, (None, None))
        state["lower_line"] = None
        state["upper_line"] = None
        state["span"] = None
        if lower_box is not None and upper_box is not None:
            lower, upper = lower_box.value(), upper_box.value()
            state["span"] = ax.axvspan(lower, upper, color=FILTER_HIST_LINE, alpha=0.18, zorder=0)
            state["lower_line"] = ax.axvline(lower, color=FILTER_HIST_LINE, linewidth=1.5)
            state["upper_line"] = ax.axvline(upper, color=FILTER_HIST_LINE, linewidth=1.5)
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
