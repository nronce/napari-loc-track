import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import napari
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt, QAbstractTableModel, QModelIndex, QTimer
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
    QSpinBox,
    QTableView,
    QTabWidget,
    QToolButton,
    QProgressBar,
)

import matplotlib
matplotlib.use("qtagg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
from napari.utils.colormaps import Colormap as NapariColormap
import trackpy as tp

from ._localize2d import (
    identify_in_frame,
    localize_frame,
    concatenate_localizations,
    is_gpufit_available,
)

TRACK_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
DEFAULT_D_COLORMAP = "coolwarm"
D_COLORMAP_CHOICES = ["coolwarm", "cool", "spring", "autumn", "bwr", "viridis"]

DEFAULT_HIST_HEIGHT = 190
HIST_HEIGHT_STEP = 50
MIN_HIST_HEIGHT = 110
MAX_HIST_HEIGHT = 600

FILTER_HIST_BG = "#20242b"
FILTER_HIST_BAR = "#2fbfae"
FILTER_HIST_FG = "#d7dbe0"
FILTER_HIST_LINE = "#ffb454"

# Columns matched to these column_map keys get shown first, in this order;
# everything else follows in its original column order.
FILTER_PRIORITY_KEYS = ["sigma", "intensity", "offset", "bkgstd", "uncertainty"]

POINTS_LAYER_NAME = "localizations"
TRACKS_LAYER_NAME = "tracks"
ALL_TRACKS_LAYER_NAME = "tracks_all"
ROI_LAYER_NAME = "xy_filter_roi"
LOC2D_CANDIDATES_LAYER_NAME = "loc2d_candidates"

# Filenames checked next to a loaded CSV/image when auto-detecting companion
# files - "{stem}" is substituted with the source file's stem.
LOCS_FILENAME_PATTERNS = ["locs.csv", "{stem}_locs.csv", "{stem}-locs.csv", "{stem}.csv"]
TRAJ_FILENAME_PATTERNS = [
    "trajectories.csv", "{stem}_trajectories.csv", "{stem}_tracks.csv", "{stem}-tracks.csv",
]
LOCS_ANALYSIS_SUBPATH = "data/localizations_filtered.csv"
TRAJ_ANALYSIS_SUBPATH = "data/trajectories.csv"

METRIC_LABELS = {
    "D": "Diffusion coefficient D (µm²/s)",
    "distance": "Distance travelled (nm)",
    "duration": "Trajectory duration (s)",
}
METRIC_CACHE_ATTR = {
    "D": "_track_diffusion_cache",
    "distance": "_track_distance_cache",
    "duration": "_track_duration_cache",
}

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


def apply_numeric_filters(df, bounds):
    filtered = df.copy()
    for column, (lower, upper) in bounds.items():
        if not column or column not in filtered.columns:
            continue
        if lower is not None:
            filtered = filtered[filtered[column] >= lower]
        if upper is not None:
            filtered = filtered[filtered[column] <= upper]
    return filtered


@thread_worker
def _load_worker(csv_path, image_path):
    df = pd.read_csv(csv_path) if csv_path else None
    image = None
    if image_path:
        from tifffile import imread
        try:
            # Memory-map instead of eagerly reading the whole stack into RAM:
            # for a large, uncompressed movie (the common case) this is a
            # near-instant zero-copy open instead of a multi-second (or much
            # longer) read, with frames paged in lazily as they're accessed.
            image = imread(image_path, out="memmap")
        except Exception:
            image = imread(image_path)
        if image.ndim == 2:
            image = image[np.newaxis, ...]
    return df, image


@thread_worker
def _link_worker(features, search_range_px, memory, n_frames):
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
    for i, linked_frame in enumerate(linked_iter):
        results.append(linked_frame)
        yield (i + 1) / total
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


@thread_worker
def _detect_worker(stack, box, min_ng):
    n_frames = stack.shape[0]
    candidates = [None] * n_frames
    counts = np.zeros(n_frames, dtype=int)
    for i in range(n_frames):
        y, x, ng = identify_in_frame(stack[i], min_ng, box)
        candidates[i] = (y, x, ng)
        counts[i] = len(y)
        yield (i + 1) / max(n_frames, 1)
    return candidates, counts


@thread_worker
def _fit_worker(stack, candidates, box, backend, offset, gain):
    n_with_candidates = sum(1 for c in candidates if c is not None and len(c[0]) > 0)
    results = [None] * len(candidates)
    done = 0
    for i, cand in enumerate(candidates):
        if cand is None or len(cand[0]) == 0:
            continue
        y, x, ng = cand
        results[i] = localize_frame(
            stack[i].astype(np.float32, copy=False),
            y,
            x,
            box,
            frame_number=i,
            net_gradient=ng,
            fit_backend=backend,
            camera_offset_adu=offset,
            camera_gain_adu_per_photon=gain,
        )
        done += 1
        yield done / max(n_with_candidates, 1)
    return concatenate_localizations(results)


@thread_worker
def _compute_d_worker(tracks_df, max_lagtime, fps, mpp):
    im = tp.imsd(tracks_df, mpp=mpp, fps=fps, max_lagtime=max_lagtime, pos_columns=["x", "y"])
    d_map = {}
    msd_map = {}
    for pid in im.columns:
        msd_series = im[pid].dropna()
        if len(msd_series) < 3:
            continue
        tau = msd_series.index.to_numpy(float)
        msd_vals = msd_series.to_numpy(float)
        slope, intercept = np.polyfit(tau, msd_vals, 1)
        D = slope / 4.0
        if D > 0 and np.isfinite(D):
            d_map[pid] = D
            msd_map[pid] = (tau, msd_vals, slope, intercept)
    return d_map, msd_map


@thread_worker
def _fit_free_metrics_worker(tracks_df, pixel_size, fps):
    fps_safe = max(fps, 1e-9)
    distance_map = {}
    duration_map = {}
    for pid, group in tracks_df.groupby("particle"):
        group = group.sort_values("frame")
        dx = np.diff(group["x"].to_numpy(float))
        dy = np.diff(group["y"].to_numpy(float))
        distance_map[pid] = float(np.hypot(dx, dy).sum() * pixel_size)
        span = int(group["frame"].max() - group["frame"].min()) + 1
        duration_map[pid] = span / fps_safe
    return distance_map, duration_map


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
        self.df = None
        self.df_filtered = None
        self.column_map = {}
        self.tracks = None
        self._hist_widgets = {}
        self._metric_hist_widgets = {}
        self._metric_bound_boxes = {}
        self._metric_use_log = {"D": True, "distance": False, "duration": False}
        self._default_bounds = {}
        self.filter_controls = {}
        self._roi_updating = False
        self._track_diffusion_cache = None
        self._track_msd_cache = None
        self._track_distance_cache = None
        self._track_duration_cache = None
        self._all_tracks_particle_ids = []
        self._load_worker_ref = None
        self._link_worker_ref = None
        self._loc2d_candidates = []
        self._loc2d_counts = np.zeros(0, dtype=int)
        self._loc2d_detect_worker_ref = None
        self._loc2d_fit_worker_ref = None
        self._compute_d_worker_ref = None
        self._metrics_worker_ref = None
        self.setup_ui()
        self._connect_viewer_events()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        self.tabs = QTabWidget(self)
        root.addWidget(self.tabs)

        self._build_load_tab()
        self._build_localize_tab()
        self._build_filter_tab()
        self._build_link_tab()
        self._build_trajectory_analysis_tab()
        self._build_data_table_tab()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(110)
        root.addWidget(self.log_box)

        self._loc2d_preview_timer = QTimer(self)
        self._loc2d_preview_timer.setSingleShot(True)
        self._loc2d_preview_timer.timeout.connect(self._update_loc2d_candidate_overlay)

        self._metric_render_timer = QTimer(self)
        self._metric_render_timer.setSingleShot(True)
        self._metric_render_timer.timeout.connect(self.render_overlay)

    def _connect_viewer_events(self):
        # Only needed to keep the detection-candidate preview boxes (Localize
        # tab) in sync with the current frame; Points/Tracks layers elsewhere
        # slice themselves natively and need no callback.
        try:
            self.viewer.dims.events.current_step.connect(self._on_current_step_changed)
        except Exception:
            pass

    def _on_current_step_changed(self, event=None):
        self._loc2d_preview_timer.start(60)

    def _get_current_frame(self):
        try:
            step = self.viewer.dims.current_step
            if isinstance(step, tuple) and len(step) > 0:
                return int(step[0])
        except Exception:
            pass
        return 0

    def _build_load_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Load data")
        layout = QVBoxLayout(tab)

        data_group = QGroupBox("Data")
        data_layout = QFormLayout(data_group)
        self.csv_edit = QLineEdit()
        self.csv_button = QPushButton("Browse CSV")
        self.csv_button.clicked.connect(self.browse_csv)
        csv_row = QHBoxLayout()
        csv_row.addWidget(self.csv_edit)
        csv_row.addWidget(self.csv_button)
        data_layout.addRow("Localization CSV", csv_row)

        self.image_edit = QLineEdit()
        self.image_button = QPushButton("Browse image")
        self.image_button.clicked.connect(self.browse_image)
        image_row = QHBoxLayout()
        image_row.addWidget(self.image_edit)
        image_row.addWidget(self.image_button)
        data_layout.addRow("Image", image_row)

        self.pixel_size_box = QDoubleSpinBox()
        self.pixel_size_box.setRange(1.0, 10000.0)
        self.pixel_size_box.setValue(161.0)
        self.pixel_size_box.setDecimals(1)
        data_layout.addRow("Pixel size (nm/px)", self.pixel_size_box)

        self.frame_one_indexed_box = QCheckBox(
            "Localization frame numbers start at 1 (offset -1 to align with image stack)"
        )
        self.frame_one_indexed_box.setChecked(False)
        data_layout.addRow("", self.frame_one_indexed_box)

        self.load_button = QPushButton("Load data")
        self.load_button.clicked.connect(self.load_data)
        data_layout.addRow("", self.load_button)
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
        layout.addStretch(1)

        self.show_points_box.stateChanged.connect(lambda _checked: self.render_overlay())
        self.marker_size_box.valueChanged.connect(lambda _v: self.render_overlay())
        self.marker_edge_width_box.valueChanged.connect(lambda _v: self.render_overlay())
        self.marker_choice.currentTextChanged.connect(lambda _v: self.render_overlay())
        self.frame_one_indexed_box.stateChanged.connect(self._on_frame_offset_changed)

    def _build_localize_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Localize (2D)")
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
        self.loc_gain_box.setValue(1.0)
        cam_layout.addRow("Gain (ADU/photon)", self.loc_gain_box)
        self.loc_offset_box = QDoubleSpinBox()
        self.loc_offset_box.setRange(0.0, 20000.0)
        self.loc_offset_box.setValue(100.0)
        cam_layout.addRow("Offset (ADU)", self.loc_offset_box)
        layout.addWidget(cam_group)

        det_group = QGroupBox("Detection (local maxima + net gradient)")
        det_layout = QFormLayout(det_group)
        self.loc_box_size = QSpinBox()
        self.loc_box_size.setRange(3, 51)
        self.loc_box_size.setSingleStep(2)
        self.loc_box_size.setValue(7)
        det_layout.addRow("Box size (px, odd)", self.loc_box_size)
        self.loc_min_ng_box = QDoubleSpinBox()
        self.loc_min_ng_box.setRange(0.0, 1e6)
        self.loc_min_ng_box.setDecimals(1)
        self.loc_min_ng_box.setValue(800.0)
        det_layout.addRow("Min net gradient", self.loc_min_ng_box)
        det_buttons = QHBoxLayout()
        self.loc_preview_button = QPushButton("Preview (current frame)")
        self.loc_preview_button.clicked.connect(self.loc2d_preview)
        self.loc_detect_button = QPushButton("Detect all frames")
        self.loc_detect_button.clicked.connect(self.loc2d_detect_all)
        det_buttons.addWidget(self.loc_preview_button)
        det_buttons.addWidget(self.loc_detect_button)
        det_layout.addRow("", det_buttons)
        self.loc_show_candidates_box = QCheckBox("Show detection candidates on current frame")
        self.loc_show_candidates_box.setChecked(True)
        self.loc_show_candidates_box.stateChanged.connect(lambda _c: self._update_loc2d_candidate_overlay())
        det_layout.addRow("", self.loc_show_candidates_box)
        self.loc_detect_progress = QProgressBar()
        self.loc_detect_progress.setRange(0, 100)
        self.loc_detect_progress.setVisible(False)
        det_layout.addRow("", self.loc_detect_progress)
        self.loc_counts_figure = Figure(figsize=(5, 2.2))
        self.loc_counts_canvas = FigureCanvas(self.loc_counts_figure)
        self.loc_counts_canvas.setMinimumHeight(200)
        det_layout.addRow("", self.loc_counts_canvas)
        layout.addWidget(det_group)

        fit_group = QGroupBox("Sub-pixel Gaussian fitting")
        fit_layout = QFormLayout(fit_group)
        self.loc_backend_box = QComboBox()
        self.loc_backend_box.addItems(["auto", "mle", "fast", "gpu"])
        fit_layout.addRow("Backend", self.loc_backend_box)
        gpu_note = QLabel("\"gpu\" needs Gpufit installed; falls back to CPU MLE automatically otherwise.")
        gpu_note.setWordWrap(True)
        fit_layout.addRow("", gpu_note)
        background_note = QLabel(
            "Each fitted localization reports offset [photon] (mean background) "
            "and bkgstd [photon] (background-noise standard deviation) in the "
            "Filter tab, full table, and exported CSV."
        )
        background_note.setWordWrap(True)
        fit_layout.addRow("", background_note)
        self.loc_fit_button = QPushButton("Fit all detected frames")
        self.loc_fit_button.clicked.connect(self.loc2d_fit_all)
        self.loc_fit_button.setEnabled(False)
        fit_layout.addRow("", self.loc_fit_button)
        self.loc_fit_progress = QProgressBar()
        self.loc_fit_progress.setRange(0, 100)
        self.loc_fit_progress.setVisible(False)
        fit_layout.addRow("", self.loc_fit_progress)
        layout.addWidget(fit_group)
        layout.addStretch(1)

        self.loc_box_size.valueChanged.connect(self._on_loc2d_box_changed)

    def _build_filter_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Filter localizations")
        root = QVBoxLayout(tab)
        root.setContentsMargins(0, 0, 0, 0)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        self.filter_status = QLabel("No data loaded")
        header_layout.addWidget(self.filter_status)
        buttons_row = QHBoxLayout()
        self.reset_filters_button = QPushButton("Reset filters")
        self.reset_filters_button.clicked.connect(self.reset_filters)
        self.reset_filters_button.setEnabled(False)
        self.apply_filters_button = QPushButton("Apply filters")
        self.apply_filters_button.clicked.connect(self.apply_filters)
        self.apply_filters_button.setEnabled(False)
        self.show_table_button = QPushButton("Show full data table")
        self.show_table_button.clicked.connect(self.show_data_table)
        buttons_row.addWidget(self.reset_filters_button)
        buttons_row.addWidget(self.apply_filters_button)
        buttons_row.addWidget(self.show_table_button)
        header_layout.addLayout(buttons_row)
        note = QLabel(
            "x / y are filtered with the yellow box drawn on the image: drag "
            "the middle to move it, drag a corner/edge handle to resize it. "
            "Changing any filter clears trajectories - relink afterwards."
        )
        note.setWordWrap(True)
        header_layout.addWidget(note)

        # Localizations are usable end-to-end without ever touching the
        # Link/Trajectory analysis tabs, so exporting lives here too, not
        # only alongside the tracking-specific controls.
        export_row = QHBoxLayout()
        self.export_button_filter = QPushButton("Export localizations / analysis")
        self.export_button_filter.clicked.connect(self.export_analysis)
        export_row.addWidget(self.export_button_filter)
        export_row.addStretch(1)
        header_layout.addLayout(export_row)
        export_note = QLabel(
            "Exports whatever is currently available: filtered localizations "
            "always, plus trajectories/D/distance/duration if you've linked "
            "and analyzed them - tracking is entirely optional."
        )
        export_note.setWordWrap(True)
        header_layout.addWidget(export_note)
        root.addWidget(header)

        scroll = QScrollArea(tab)
        scroll.setWidgetResizable(True)
        self.filter_content = QWidget()
        self.filter_layout = QGridLayout(self.filter_content)
        scroll.setWidget(self.filter_content)
        root.addWidget(scroll)
        self.filter_layout.addWidget(QLabel("Load data to see filters"), 0, 0)

    def _build_link_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Link")
        layout = QVBoxLayout(tab)

        tracking_group = QGroupBox("Tracking")
        tracking_layout = QFormLayout(tracking_group)
        self.search_box = QDoubleSpinBox()
        self.search_box.setRange(1.0, 10000.0)
        self.search_box.setValue(250.0)
        self.search_box.setDecimals(0)
        tracking_layout.addRow("Search range (nm)", self.search_box)
        self.memory_box = QSpinBox()
        self.memory_box.setRange(0, 20)
        self.memory_box.setValue(1)
        tracking_layout.addRow("Memory", self.memory_box)
        self.min_traj_box = QSpinBox()
        self.min_traj_box.setRange(1, 1000)
        self.min_traj_box.setValue(2)
        tracking_layout.addRow("Min track length", self.min_traj_box)
        self.link_button = QPushButton("Link trajectories")
        self.link_button.clicked.connect(self.link_tracks)
        self.link_button.setEnabled(False)
        tracking_layout.addRow("", self.link_button)
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

        for box in (self.show_tracks_box, self.persist_tracks_box, self.show_all_tracks_box):
            box.stateChanged.connect(lambda _checked: self.render_overlay())
        self.line_width_box.valueChanged.connect(lambda _v: self.render_overlay())
        self.all_tracks_line_width_box.valueChanged.connect(lambda _v: self.render_overlay())

    def _build_trajectory_analysis_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Trajectory analysis")
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(tab)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        # --- D (requires a linear MSD fit) ---
        d_group = QGroupBox("Diffusion coefficient D (needs a linear MSD fit)")
        d_layout = QVBoxLayout(d_group)
        params_row = QFormLayout()
        self.max_lagtime_box = QSpinBox()
        self.max_lagtime_box.setRange(2, 200)
        self.max_lagtime_box.setValue(5)
        params_row.addRow("Max lag time (frames)", self.max_lagtime_box)
        self.fps_box = QDoubleSpinBox()
        self.fps_box.setRange(0.001, 1e5)
        self.fps_box.setValue(100.0)
        params_row.addRow("Acquisition frame rate (fps)", self.fps_box)
        self.msd_sample_box = QSpinBox()
        self.msd_sample_box.setRange(1, 50)
        self.msd_sample_box.setValue(10)
        params_row.addRow("Example trajectories to validate", self.msd_sample_box)
        d_layout.addLayout(params_row)
        self.compute_d_button = QPushButton("Compute D")
        self.compute_d_button.clicked.connect(self.compute_d)
        self.compute_d_button.setEnabled(False)
        d_layout.addWidget(self.compute_d_button)
        self.compute_d_progress = QProgressBar()
        self.compute_d_progress.setRange(0, 0)  # indeterminate: tp.imsd has no natural progress granularity
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
        d_bounds_row.addWidget(self.d_max_box)
        d_layout.addLayout(d_bounds_row)
        self._metric_bound_boxes["D"] = (self.d_min_box, self.d_max_box)
        d_layout.addWidget(self._make_metric_histogram("D"))
        self.d_min_box.valueChanged.connect(lambda _v: self._on_metric_bounds_changed("D"))
        self.d_max_box.valueChanged.connect(lambda _v: self._on_metric_bounds_changed("D"))

        msd_sub = QGroupBox("MSD fit validation (sample trajectories + their linear fit)")
        msd_sub_layout = QVBoxLayout(msd_sub)
        self.msd_figure = Figure(figsize=(5, 2.8))
        self.msd_canvas = FigureCanvas(self.msd_figure)
        self.msd_canvas.setMinimumHeight(240)
        msd_sub_layout.addWidget(self.msd_canvas)
        d_layout.addWidget(msd_sub)
        layout.addWidget(d_group)

        # --- Distance travelled (fit-free) ---
        dist_group = QGroupBox("Distance travelled (fit-free: total path length)")
        dist_layout = QVBoxLayout(dist_group)
        dist_bounds_row = QHBoxLayout()
        dist_bounds_row.addWidget(QLabel("Min (nm)"))
        self.dist_min_box = QDoubleSpinBox()
        self.dist_min_box.setRange(0, 1e9)
        self.dist_min_box.setDecimals(1)
        dist_bounds_row.addWidget(self.dist_min_box)
        dist_bounds_row.addWidget(QLabel("Max (nm)"))
        self.dist_max_box = QDoubleSpinBox()
        self.dist_max_box.setRange(0, 1e9)
        self.dist_max_box.setDecimals(1)
        self.dist_max_box.setValue(1000.0)
        dist_bounds_row.addWidget(self.dist_max_box)
        dist_layout.addLayout(dist_bounds_row)
        self._metric_bound_boxes["distance"] = (self.dist_min_box, self.dist_max_box)
        dist_layout.addWidget(self._make_metric_histogram("distance"))
        self.dist_min_box.valueChanged.connect(lambda _v: self._on_metric_bounds_changed("distance"))
        self.dist_max_box.valueChanged.connect(lambda _v: self._on_metric_bounds_changed("distance"))
        layout.addWidget(dist_group)

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
        dur_bounds_row.addWidget(self.dur_max_box)
        dur_layout.addLayout(dur_bounds_row)
        self._metric_bound_boxes["duration"] = (self.dur_min_box, self.dur_max_box)
        dur_layout.addWidget(self._make_metric_histogram("duration"))
        self.dur_min_box.valueChanged.connect(lambda _v: self._on_metric_bounds_changed("duration"))
        self.dur_max_box.valueChanged.connect(lambda _v: self._on_metric_bounds_changed("duration"))
        layout.addWidget(dur_group)

        # --- Coloring ---
        color_group = QGroupBox("Trajectory coloring")
        color_layout = QFormLayout(color_group)
        self.color_trajectories_box = QCheckBox("Color trajectories by the metric below")
        color_layout.addRow("", self.color_trajectories_box)
        self.color_metric_box = QComboBox()
        self.color_metric_box.addItems(["D (diffusion coefficient)", "Distance travelled", "Track duration"])
        color_layout.addRow("Metric", self.color_metric_box)
        self.d_colormap_box = QComboBox()
        self.d_colormap_box.addItems(D_COLORMAP_CHOICES)
        self.d_colormap_box.setCurrentText(DEFAULT_D_COLORMAP)
        color_layout.addRow("Colormap", self.d_colormap_box)
        layout.addWidget(color_group)

        self.color_trajectories_box.stateChanged.connect(self._on_color_settings_changed)
        self.color_metric_box.currentTextChanged.connect(self._on_color_settings_changed)
        self.d_colormap_box.currentTextChanged.connect(self._on_color_settings_changed)

        # --- Export ---
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        export_note = QLabel(
            "Saves every plot shown above, the filtered localizations, linked "
            "trajectories, per-track metrics, and a metadata.json describing "
            "the parameters used, into a new \"analysis\" subfolder next to "
            "the source CSV (analysis_2, analysis_3, ... if already present)."
        )
        export_note.setWordWrap(True)
        export_layout.addWidget(export_note)
        self.export_button = QPushButton("Export analysis")
        self.export_button.clicked.connect(self.export_analysis)
        export_layout.addWidget(self.export_button)
        layout.addWidget(export_group)
        layout.addStretch(1)

    def _build_data_table_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Data table")
        layout = QVBoxLayout(tab)
        top_row = QHBoxLayout()
        self.data_table_label = QLabel("No data loaded")
        top_row.addWidget(self.data_table_label)
        top_row.addStretch(1)
        layout.addLayout(top_row)
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
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Data table":
                self.tabs.setCurrentIndex(i)
                break

    def _frame_offset(self):
        return -1 if self.frame_one_indexed_box.isChecked() else 0

    def _on_frame_offset_changed(self, _checked=None):
        if self.df is None:
            return
        self.log("Frame indexing changed - re-link trajectories to use the new offset.")
        self.apply_filters()

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

        worker = _load_worker(csv_path, image_path)
        worker.returned.connect(lambda result: self._on_load_finished(result, csv_path, image_path))
        worker.errored.connect(self._on_load_errored)
        worker.finished.connect(self._on_load_worker_finished)
        self._load_worker_ref = worker
        worker.start()

    def _on_load_worker_finished(self):
        self.load_button.setEnabled(True)
        self.load_progress.setVisible(False)
        self._load_worker_ref = None

    def _on_load_errored(self, exc):
        self.log(f"Failed to load data: {exc}")

    def _on_load_finished(self, result, csv_path, image_path):
        df, image = result
        self.viewer.layers.clear()
        if image is not None:
            self.viewer.add_image(image, name=Path(image_path).name, colormap="gray")

        auto_loaded = False
        if df is None and not csv_path and image_path:
            found = self._find_companion_file(Path(image_path), LOCS_FILENAME_PATTERNS, LOCS_ANALYSIS_SUBPATH)
            if found is not None:
                try:
                    df = pd.read_csv(found)
                    csv_path = str(found)
                    self.csv_edit.setText(csv_path)
                    auto_loaded = True
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

    def _find_companion_file(self, base_path, filename_patterns, analysis_relative_path=None):
        base_path = Path(base_path)
        folder = base_path.parent
        stem = base_path.stem
        for pattern in filename_patterns:
            candidate = folder / pattern.format(stem=stem)
            if candidate.is_file():
                return candidate

        if analysis_relative_path:
            numbered_dirs = []
            for d in folder.glob("analysis*"):
                if not d.is_dir():
                    continue
                suffix = d.name[len("analysis"):]
                if suffix == "":
                    num = 1
                elif suffix.startswith("_") and suffix[1:].isdigit():
                    num = int(suffix[1:])
                else:
                    continue
                numbered_dirs.append((num, d))
            for _, d in sorted(numbered_dirs, key=lambda t: -t[0]):
                candidate = d / analysis_relative_path
                if candidate.is_file():
                    return candidate
        return None

    def _try_autoload_trajectories(self, base_path):
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
        self._track_diffusion_cache = None
        self._track_msd_cache = None
        self.compute_d_button.setEnabled(True)
        self._start_fit_free_metrics_worker()
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
        self._track_diffusion_cache = None
        self._track_msd_cache = None
        self._track_distance_cache = None
        self._track_duration_cache = None
        self.log(log_message)

        if frame_is_zero_indexed:
            self.frame_one_indexed_box.blockSignals(True)
            self.frame_one_indexed_box.setChecked(False)
            self.frame_one_indexed_box.blockSignals(False)
        else:
            frame_col = self._resolve_column("frame")
            if frame_col and frame_col in self.df.columns and not self.df[frame_col].empty:
                min_frame = self.df[frame_col].min()
                self.frame_one_indexed_box.blockSignals(True)
                self.frame_one_indexed_box.setChecked(bool(min_frame == 1))
                self.frame_one_indexed_box.blockSignals(False)

        self._build_filter_tab_contents()
        self.apply_filters_button.setEnabled(True)
        self.reset_filters_button.setEnabled(True)
        self.link_button.setEnabled(True)
        self.render_button.setEnabled(True)
        self.compute_d_button.setEnabled(False)

        self.render_overlay()
        self._sync_xy_roi_layer()
        self.viewer.tooltip.visible = True
        self.data_table_model.set_dataframe(self.df_filtered)
        self.data_table_label.setText(f"{len(self.df_filtered)} rows x {len(self.df_filtered.columns)} columns")

    # ------------------------------------------------------------------
    # Localize (2D): detection + sub-pixel Gaussian fitting
    # ------------------------------------------------------------------
    def _get_localize_image_layer(self):
        for layer in list(self.viewer.layers.selection) + list(self.viewer.layers):
            if isinstance(layer, napari.layers.Image):
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
            stack[frame_idx].astype(np.float32, copy=False), self.loc_min_ng_box.value(), box
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

        worker = _detect_worker(stack, box, min_ng)
        worker.yielded.connect(lambda frac: self.loc_detect_progress.setValue(int(frac * 100)))
        worker.returned.connect(self._on_loc2d_detect_finished)
        worker.errored.connect(lambda exc: self.log(f"Detection failed: {exc}"))
        worker.finished.connect(self._on_loc2d_detect_worker_finished)
        self._loc2d_detect_worker_ref = worker
        worker.start()

    def _on_loc2d_detect_worker_finished(self):
        self.loc_detect_button.setEnabled(True)
        self.loc_detect_progress.setVisible(False)
        self._loc2d_detect_worker_ref = None

    def _on_loc2d_detect_finished(self, result):
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
            self.loc_counts_canvas.draw_idle()
            return
        ax = figure.add_subplot(111)
        ax.plot(np.arange(len(self._loc2d_counts)), self._loc2d_counts, color="#ff9800", linewidth=1.2)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Detections")
        ax.set_title("Detections vs frame")
        ax.grid(alpha=0.3)
        figure.tight_layout()
        self.loc_counts_canvas.draw_idle()

    def _update_loc2d_candidate_overlay(self):
        # This is the one place in the plugin that still does per-frame
        # Python work (rebuilding a Shapes layer of candidate boxes), since
        # it has to track the current frame. It's only useful as a detection
        # preview aid, so skip it entirely - not just cheaply, but with zero
        # work - whenever the Localize tab isn't the one showing, or the user
        # has turned it off, so scrubbing through the movie on other tabs
        # (with locs/tracks overlaid) never triggers it.
        show = (
            bool(self._loc2d_candidates)
            and self.tabs.tabText(self.tabs.currentIndex()) == "Localize (2D)"
            and self.loc_show_candidates_box.isChecked()
        )
        if not show:
            self._remove_layer(LOC2D_CANDIDATES_LAYER_NAME)
            return
        frame_idx = self._get_current_frame()
        if frame_idx < 0 or frame_idx >= len(self._loc2d_candidates):
            self._remove_layer(LOC2D_CANDIDATES_LAYER_NAME)
            return
        cand = self._loc2d_candidates[frame_idx]
        if cand is None or len(cand[0]) == 0:
            self._remove_layer(LOC2D_CANDIDATES_LAYER_NAME)
            return
        y, x, _ = cand
        half = self._loc2d_box_size() / 2.0
        centers = np.column_stack([np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)])
        offsets = np.array([[-half, -half], [-half, half], [half, half], [half, -half]])
        rects = centers[:, None, :] + offsets[None, :, :]  # (N, 4, 2), vectorized instead of a per-box Python loop
        if LOC2D_CANDIDATES_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[LOC2D_CANDIDATES_LAYER_NAME].data = rects
        else:
            self.viewer.add_shapes(
                rects,
                name=LOC2D_CANDIDATES_LAYER_NAME,
                shape_type="rectangle",
                edge_color="yellow",
                face_color="transparent",
                edge_width=1,
            )

    def _on_tab_changed(self, _index=None):
        self._update_loc2d_candidate_overlay()

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

        worker = _fit_worker(stack, self._loc2d_candidates, box, backend, offset, gain)
        worker.yielded.connect(lambda frac: self.loc_fit_progress.setValue(int(frac * 100)))
        worker.returned.connect(self._on_loc2d_fit_finished)
        worker.errored.connect(lambda exc: self.log(f"Fitting failed: {exc}"))
        worker.finished.connect(self._on_loc2d_fit_worker_finished)
        self._loc2d_fit_worker_ref = worker
        worker.start()

    def _on_loc2d_fit_worker_finished(self):
        self.loc_fit_button.setEnabled(True)
        self.loc_fit_progress.setVisible(False)
        self._loc2d_fit_worker_ref = None

    def _on_loc2d_fit_finished(self, locs):
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
                "bkgstd [photon]": locs["bkgstd"].astype(float),
                "uncertainty [nm]": 0.5 * (locs["lpx"].astype(float) + locs["lpy"].astype(float)) * pixel_size,
                "net_gradient": locs["net_gradient"].astype(float),
            }
        )
        self._remove_layer(LOC2D_CANDIDATES_LAYER_NAME)
        self._ingest_localization_dataframe(
            df,
            f"Fitted {n} localizations from the loaded image stack (in-app 2D localization)",
            frame_is_zero_indexed=True,
        )

    # ------------------------------------------------------------------
    # Filtering (+ per-column histograms)
    # ------------------------------------------------------------------
    def _default_bounds_for(self, column):
        col_key = next((k for k, v in self.column_map.items() if v == column), None)
        if col_key == "sigma":
            return 0.0, 1000.0
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
            lower_box.setDecimals(3)
            upper_box = QDoubleSpinBox()
            upper_box.setRange(-1e9, 1e9)
            upper_box.setDecimals(3)
            default_lower, default_upper = self._default_bounds_for(column)
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

    def apply_filters(self):
        if self.df is None:
            return
        bounds = {}
        for column, (lower_box, upper_box) in self.filter_controls.items():
            lower = lower_box.value()
            upper = upper_box.value()
            bounds[column] = (lower, upper)
        self.df_filtered = apply_numeric_filters(self.df, bounds)
        self.filter_status.setText(f"Showing {len(self.df_filtered)} localizations")
        self.log(f"Filtered to {len(self.df_filtered)} localizations")
        self._invalidate_tracks(reason="filters changed")
        self.render_overlay()
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
        self._track_diffusion_cache = None
        self._track_msd_cache = None
        self._track_distance_cache = None
        self._track_duration_cache = None
        self.compute_d_button.setEnabled(False)
        self._remove_layer(TRACKS_LAYER_NAME)
        self._remove_layer(ALL_TRACKS_LAYER_NAME)
        self._clear_metric_histograms()
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

        worker = _link_worker(features, search_range_px, self.memory_box.value(), n_frames)
        worker.yielded.connect(lambda frac: self.link_progress.setValue(int(frac * 100)))
        worker.returned.connect(self._on_link_finished)
        worker.errored.connect(self._on_link_errored)
        worker.finished.connect(self._on_link_worker_finished)
        self._link_worker_ref = worker
        worker.start()

    def _on_link_worker_finished(self):
        self.link_button.setEnabled(True)
        self.link_progress.setVisible(False)
        self._link_worker_ref = None

    def _on_link_errored(self, exc):
        self.log(f"Linking failed: {exc}")

    def _on_link_finished(self, linked):
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
        self._track_diffusion_cache = None
        self._track_msd_cache = None
        self.compute_d_button.setEnabled(True)
        self.log(f"Linked {traj['particle'].nunique()} trajectories")
        self._start_fit_free_metrics_worker()
        self.render_overlay()

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
    # napari layer synchronization
    # ------------------------------------------------------------------
    def render_overlay(self):
        # Rebuilds the points/tracks layers from the current data/style. This
        # is only called on load/filter/link/style-change actions, never on
        # a dims-slider move: Points/Tracks layers carry the full multi-frame
        # data and let napari slice them natively, so moving the slider does
        # no Python-side work and stays smooth even with many localizations
        # or trajectories.
        if self.df_filtered is None or self.df_filtered.empty:
            return
        self._sync_points_layer()
        self._sync_tracks_layer()
        self._sync_all_tracks_layer()

    def _sync_points_layer(self):
        x_col = self._resolve_column("x")
        y_col = self._resolve_column("y")
        frame_col = self._resolve_column("frame")

        if not (self.show_points_box.isChecked() and x_col and y_col and frame_col):
            self._remove_layer(POINTS_LAYER_NAME)
            return

        geom_cols = [frame_col, y_col, x_col]
        valid = self.df_filtered.dropna(subset=geom_cols)
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
            self.viewer.add_points(coords, **kwargs)
        self.viewer.tooltip.visible = True

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

        self.log("Computing D in the background...")
        self.compute_d_button.setEnabled(False)
        self.compute_d_progress.setVisible(True)

        worker = _compute_d_worker(self.tracks, max_lagtime, fps, mpp)
        worker.returned.connect(self._on_compute_d_finished)
        worker.errored.connect(lambda exc: self.log(f"D computation failed: {exc}"))
        worker.finished.connect(self._on_compute_d_worker_finished)
        self._compute_d_worker_ref = worker
        worker.start()

    def _on_compute_d_worker_finished(self):
        self.compute_d_button.setEnabled(True)
        self.compute_d_progress.setVisible(False)
        self._compute_d_worker_ref = None

    def _on_compute_d_finished(self, result):
        d_map, msd_map = result
        self._track_diffusion_cache = d_map
        self._track_msd_cache = msd_map
        self.log(f"Computed D for {len(d_map)} of {self.tracks['particle'].nunique()} trajectories")
        self._set_metric_default_bounds("D", d_map)
        self._set_metric_view_default("D", d_map)
        self._draw_metric_histogram("D")
        self._draw_msd_validation()
        if self.color_trajectories_box.isChecked():
            self.render_overlay()

    def _start_fit_free_metrics_worker(self):
        # Fit-free (distance travelled, duration) but still a full pass over
        # every trajectory - background it too so linking/auto-loading a
        # large trajectories file doesn't freeze the UI while it runs.
        if self.tracks is None or self.tracks.empty:
            return
        worker = _fit_free_metrics_worker(self.tracks, self.pixel_size_box.value(), self.fps_box.value())
        worker.returned.connect(self._on_fit_free_metrics_finished)
        worker.errored.connect(lambda exc: self.log(f"Distance/duration computation failed: {exc}"))
        self._metrics_worker_ref = worker
        worker.start()

    def _on_fit_free_metrics_finished(self, result):
        distance_map, duration_map = result
        self._track_distance_cache = distance_map
        self._track_duration_cache = duration_map
        self._set_metric_default_bounds("distance", distance_map)
        self._set_metric_default_bounds("duration", duration_map)
        self._set_metric_view_default("distance", distance_map)
        self._set_metric_view_default("duration", duration_map)
        self._draw_metric_histogram("distance")
        self._draw_metric_histogram("duration")
        self.log(f"Computed distance/duration for {len(distance_map)} trajectories")

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
        return getattr(self, METRIC_CACHE_ATTR[key]) or {}

    def _current_metric_key(self):
        choice = self.color_metric_box.currentText()
        if choice.startswith("D"):
            return "D"
        if choice.startswith("Distance"):
            return "distance"
        return "duration"

    def _metric_norm_range(self, key):
        min_box, max_box = self._metric_bound_boxes[key]
        use_log = self._metric_use_log[key]
        lo = min_box.value()
        hi = max_box.value()
        if use_log:
            lo = max(lo, 1e-12)
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

    def _on_color_settings_changed(self, *_args):
        for key in ("D", "distance", "duration"):
            self._draw_metric_histogram(key)
        self.render_overlay()

    # --- generic metric histogram (used for D, distance, duration) ---
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
        view_min_box = QDoubleSpinBox()
        view_min_box.setRange(-1e9, 1e9)
        view_min_box.setDecimals(4)
        view_min_box.setMaximumWidth(75)
        toolbar.addWidget(view_min_box)
        toolbar.addWidget(QLabel(u"–"))
        view_max_box = QDoubleSpinBox()
        view_max_box.setRange(-1e9, 1e9)
        view_max_box.setDecimals(4)
        view_max_box.setMaximumWidth(75)
        toolbar.addWidget(view_max_box)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        figure = Figure(figsize=(5, 2.4))
        canvas = FigureCanvas(figure)
        canvas.setMinimumHeight(220)
        layout.addWidget(canvas)

        state = {
            "figure": figure,
            "canvas": canvas,
            "bins_box": bins_box,
            "view_min_box": view_min_box,
            "view_max_box": view_max_box,
            "drag": None,
            "lower_line": None,
            "upper_line": None,
            "span": None,
        }
        self._metric_hist_widgets[key] = state
        bins_box.valueChanged.connect(lambda _v, k=key: self._draw_metric_histogram(k))
        view_min_box.valueChanged.connect(lambda _v, k=key: self._draw_metric_histogram(k))
        view_max_box.valueChanged.connect(lambda _v, k=key: self._draw_metric_histogram(k))

        canvas.mpl_connect("button_press_event", lambda evt, k=key: self._on_metric_hist_press(k, evt))
        canvas.mpl_connect("motion_notify_event", lambda evt, k=key: self._on_metric_hist_motion(k, evt))
        canvas.mpl_connect("button_release_event", lambda evt, k=key: self._on_metric_hist_release(k, evt))
        return container

    def _set_metric_view_default(self, key, cache):
        state = self._metric_hist_widgets.get(key)
        if not state or not cache:
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
            view_lo = max(view_lo, 1e-9)
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
        ax.set_xlim(view_lo, view_hi)

        min_box, max_box = self._metric_bound_boxes[key]
        lower, upper = min_box.value(), max_box.value()
        state["span"] = ax.axvspan(lower, upper, color="black", alpha=0.08, zorder=0)
        state["lower_line"] = ax.axvline(lower, color="black", linewidth=1.5)
        state["upper_line"] = ax.axvline(upper, color="black", linewidth=1.5)

        ax.set_xlabel(METRIC_LABELS[key])
        ax.set_ylabel("Count")
        ax.set_title(f"{len(values)} trajectories")
        ax.grid(True, which="both", axis="both", alpha=0.3)
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
        state["span"] = ax.axvspan(lower, upper, color="black", alpha=0.08, zorder=0)
        if state.get("lower_line") is not None:
            state["lower_line"].set_xdata([lower, lower])
        if state.get("upper_line") is not None:
            state["upper_line"].set_xdata([upper, upper])
        state["canvas"].draw_idle()

    def _on_metric_bounds_changed(self, key):
        self._sync_metric_hist_lines(key)
        # Dragging a bound (or typing into the spinbox) fires this on every
        # intermediate value. Rebuilding the trajectory layers on every one
        # of those - especially with many trajectories - is what made
        # "rescaling colors" freeze the UI; coalesce into a single rebuild
        # a short delay after the last change instead.
        self._metric_render_timer.start(150)

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
        self.render_overlay()

    def _draw_msd_validation(self):
        figure = self.msd_figure
        figure.clear()
        ax = figure.add_subplot(111)
        if not self._track_msd_cache:
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
            pid, (tau, msd_vals, slope, intercept) = items[idx]
            D = self._track_diffusion_cache.get(pid, float("nan"))
            ax.plot(tau, msd_vals, "o-", color=colors[i], alpha=0.85, markersize=3, linewidth=1)
            fit_tau = np.array([0.0, tau.max()])
            ax.plot(
                fit_tau, slope * fit_tau + intercept, "--", color=colors[i], alpha=0.6, linewidth=1,
                label=f"#{pid} D={D:.3g} µm²/s",
            )
        ax.set_xlabel("Lag time (s)")
        ax.set_ylabel("MSD (µm²)")
        ax.set_title(f"MSD fit validation ({n_sample} example trajectories)")
        ax.legend(fontsize=6, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.3)
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
        self.export_button_filter.setEnabled(False)
        try:
            csv_path = self.csv_edit.text().strip()
            image_path = self.image_edit.text().strip()
            # Prefer the locs CSV's folder, then the image's (e.g. after an
            # in-app 2D localization run where no CSV was ever loaded) - only
            # fall back to the current working directory if neither is known,
            # so exports don't end up outside the data folder.
            if csv_path:
                base_dir = Path(csv_path).parent
            elif image_path:
                base_dir = Path(image_path).parent
            else:
                base_dir = Path.cwd()
            folder = self._make_analysis_folder(base_dir)
            folder.mkdir(parents=True, exist_ok=True)
            n_plots = self._export_plots(folder)
            self._export_data(folder)
            self._write_metadata(folder, csv_path)
            self.log(f"Exported {n_plots} plots + data + metadata to {folder}")
        except Exception as exc:
            self.log(f"Export failed: {exc}")
        finally:
            self.export_button.setEnabled(True)
            self.export_button_filter.setEnabled(True)

    def _make_analysis_folder(self, base_dir):
        candidate = base_dir / "analysis"
        if not candidate.exists():
            return candidate
        i = 2
        while (base_dir / f"analysis_{i}").exists():
            i += 1
        return base_dir / f"analysis_{i}"

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

    def _export_data(self, folder):
        data_dir = folder / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        if self.df_filtered is not None:
            self.df_filtered.to_csv(data_dir / "localizations_filtered.csv", index=False)
        if self.tracks is not None and not self.tracks.empty:
            self.tracks.to_csv(data_dir / "trajectories.csv", index=False)
            self._export_track_metrics(data_dir)

    def _export_track_metrics(self, data_dir):
        particle_ids = sorted(self.tracks["particle"].unique())
        d_map = self._track_diffusion_cache or {}
        distance_map = self._track_distance_cache or {}
        duration_map = self._track_duration_cache or {}
        rows = [
            {
                "particle": pid,
                "D_um2_per_s": d_map.get(pid),
                "distance_nm": distance_map.get(pid),
                "duration_s": duration_map.get(pid),
            }
            for pid in particle_ids
        ]
        pd.DataFrame(rows).to_csv(data_dir / "track_metrics.csv", index=False)

    def _write_metadata(self, folder, csv_path):
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
            "source_image": self.image_edit.text().strip() or None,
            "pixel_size_nm_per_px": self.pixel_size_box.value(),
            "frame_one_indexed": self.frame_one_indexed_box.isChecked(),
            "n_localizations_total": len(self.df) if self.df is not None else 0,
            "n_localizations_filtered": len(self.df_filtered) if self.df_filtered is not None else 0,
            "filter_bounds": {
                col: {"min": lo.value(), "max": hi.value()} for col, (lo, hi) in self.filter_controls.items()
            },
            "localization_2d": {
                "gain_adu_per_photon": self.loc_gain_box.value(),
                "offset_adu": self.loc_offset_box.value(),
                "box_size_px": self._loc2d_box_size(),
                "min_net_gradient": self.loc_min_ng_box.value(),
                "fit_backend": self.loc_backend_box.currentText(),
                "n_frames_with_candidates": int(n_candidate_frames),
                "n_candidates_total": int(self._loc2d_counts.sum()) if len(self._loc2d_counts) else 0,
            },
            "linking": {
                "search_range_nm": self.search_box.value(),
                "memory": self.memory_box.value(),
                "min_track_length": self.min_traj_box.value(),
                "n_trajectories": n_tracks,
            },
            "diffusion": {
                "max_lagtime_frames": self.max_lagtime_box.value(),
                "fps": self.fps_box.value(),
                "d_min": self.d_min_box.value(),
                "d_max": self.d_max_box.value(),
                "n_tracks_with_D": len(self._track_diffusion_cache or {}),
                "msd_validation_sample_count": self.msd_sample_box.value(),
            },
            "distance_bounds_nm": {"min": self.dist_min_box.value(), "max": self.dist_max_box.value()},
            "duration_bounds_s": {"min": self.dur_min_box.value(), "max": self.dur_max_box.value()},
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
                }
                for key, state in self._metric_hist_widgets.items()
            },
        }
        with open(folder / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Tracks / all-tracks / ROI layers
    # ------------------------------------------------------------------
    def _sync_tracks_layer(self):
        self._remove_layer(TRACKS_LAYER_NAME)
        has_tracks = self.tracks is not None and not self.tracks.empty
        if not (self.show_tracks_box.isChecked() and has_tracks):
            return

        traj = self.tracks.sort_values(["particle", "frame"])
        track_id = traj["particle"].to_numpy(int)
        t = traj["frame"].to_numpy(int)
        y = traj["y"].to_numpy(float)
        x = traj["x"].to_numpy(float)
        data = np.column_stack([track_id, t, y, x])
        tail_length = int(t.max() - t.min()) + 1 if len(t) else 1
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

        tracks_layer = self.viewer.add_tracks(data, **kwargs)
        tracks_layer._get_tooltip_text = self._tracks_layer_tooltip

    def _track_tooltip_lines(self, pid):
        if self.tracks is None:
            return []
        track_rows = self.tracks[self.tracks["particle"] == pid]
        if track_rows.empty:
            return []
        frame_span = int(track_rows["frame"].max() - track_rows["frame"].min() + 1)
        lines = [
            f"track: {int(pid)}",
            f"length: {len(track_rows)} points",
            f"span: {frame_span} frames",
        ]
        duration_map = self._track_duration_cache or {}
        if pid in duration_map:
            lines.append(f"duration: {duration_map[pid]:.3g} s")
        distance_map = self._track_distance_cache or {}
        if pid in distance_map:
            lines.append(f"distance: {distance_map[pid]:.3g} nm")
        d_map = self._track_diffusion_cache or {}
        if pid in d_map:
            lines.append(f"D: {d_map[pid]:.4g} µm²/s")
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
        has_tracks = self.tracks is not None and not self.tracks.empty
        if not (self.show_all_tracks_box.isChecked() and has_tracks):
            return

        color_by_metric = self.color_trajectories_box.isChecked()
        key = self._current_metric_key() if color_by_metric else None
        cache = self._metric_cache(key) if color_by_metric else {}
        cmap = matplotlib.colormaps[self.d_colormap_box.currentText()]

        paths = []
        edge_colors = []
        particle_ids = []
        for i, (pid, group) in enumerate(self.tracks.groupby("particle")):
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
        )
        all_tracks_layer._get_tooltip_text = self._all_tracks_layer_tooltip

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
                )
                layer.mode = "select"
                layer.selected_data = {0}
                layer.events.data.connect(self._on_roi_changed)
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
        view_min_box.setDecimals(3)
        view_min_box.setMaximumWidth(75)
        bins_row.addWidget(view_min_box)
        bins_row.addWidget(QLabel(u"–"))
        view_max_box = QDoubleSpinBox()
        view_max_box.setRange(-1e9, 1e9)
        view_max_box.setDecimals(3)
        view_max_box.setMaximumWidth(75)
        bins_row.addWidget(view_max_box)
        bins_row.addStretch(1)
        layout.addLayout(bins_row)

        default_lo, default_hi = self._default_bounds_for(column)
        view_min_box.setValue(default_lo)
        view_max_box.setValue(default_hi)

        figure = Figure(figsize=(4.2, DEFAULT_HIST_HEIGHT / 100))
        canvas = FigureCanvas(figure)
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
            positive = values[values > 0]
            shown = positive[(positive >= lo) & (positive <= view_hi)]
            if len(shown):
                bins = np.logspace(np.log10(lo), np.log10(view_hi), n_bins + 1)
                ax.hist(shown, bins=bins, color=FILTER_HIST_BAR, alpha=0.9)
            ax.set_xscale("log")
            ax.set_xlim(lo, view_hi)
        else:
            shown = values[(values >= view_lo) & (values <= view_hi)]
            if len(shown):
                ax.hist(shown, bins=n_bins, range=(view_lo, view_hi), color=FILTER_HIST_BAR, alpha=0.9)
            ax.set_xlim(view_lo, view_hi)

        ax.set_title(column, fontsize=9, color=FILTER_HIST_FG)
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
        for column in self._hist_widgets:
            self._sync_histogram_lines(column)

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
