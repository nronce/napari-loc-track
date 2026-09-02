"""Interactive behaviour of the trajectory tabs, driven through the real widget.

napari's canvas needs OpenGL, which a headless run does not have, so the widget
is built against a stub viewer that records layer calls. That is enough to pin
the things a user actually feels: the linking readout, the frame-rate/interval
coupling, and - the important one - that changing a colour bound recolours the
existing layers instead of rebuilding them.
"""
import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy", reason="needs Qt")
widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return ensure_qapp()


class _Layers(dict):
    # napari's LayerList always carries a selection, and the widget consults it
    # to find the layer the user means (the raw stack to localize in, the field
    # of view to render into), so the stub carries one too.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection = set()

    def __iter__(self):
        return iter(list(self.values()))

    def remove(self, layer):
        self.selection.discard(layer)
        self.pop(getattr(layer, "name", layer), None)

    def clear(self):
        self.selection.clear()
        dict.clear(self)


class _StubViewer:
    """Records add_* calls so a test can tell a rebuild from an in-place update."""

    def __init__(self):
        self.layers = _Layers()
        self.add_calls = 0
        connect = types.SimpleNamespace(connect=lambda *_a, **_k: None)
        self.dims = types.SimpleNamespace(
            events=types.SimpleNamespace(current_step=connect), current_step=(0,)
        )
        self.tooltip = types.SimpleNamespace(visible=False)

    def _add(self, name, **kwargs):
        self.add_calls += 1
        signal = types.SimpleNamespace(connect=lambda *_a, **_k: None)
        layer = types.SimpleNamespace(
            name=name, events=types.SimpleNamespace(data=signal), **kwargs
        )
        self.layers[name] = layer
        return layer

    def add_image(self, data, name="image", **kw):
        # kwargs kept like the other layer types: a render's scale/translate is
        # what places it on top of the raw stack, so tests have to see them.
        return self._add(name, data=data, **kw)

    def add_points(self, data, name="points", **kw):
        return self._add(name, data=data, **kw)

    def add_shapes(self, data, name="shapes", **kw):
        return self._add(name, data=data, **kw)

    def add_tracks(self, data, name="tracks", **kw):
        return self._add(name, data=data, **kw)


# Widgets are kept alive for the whole session on purpose: each one owns a dozen
# matplotlib canvases, and letting Qt garbage-collect them between tests crashes
# the interpreter rather than failing a test. The QApplication needs the same
# treatment - dropping the last reference destroys it, and the next QWidget then
# aborts the process.
_WIDGETS = []
_QAPP = None


def ensure_qapp():
    global _QAPP
    if _QAPP is None:
        _QAPP = QApplication.instance() or QApplication([])
    return _QAPP


def make_widget():
    # Test modules run in any order, so the factory creates the application
    # itself rather than relying on a fixture in whichever module ran first.
    ensure_qapp()
    widget = widget_mod.LocalizationTrackingWidget(_StubViewer())
    _WIDGETS.append(widget)
    return widget


@pytest.fixture
def widget(qapp):
    return make_widget()


def _tracks_frame(n_particles=6, n_points=8):
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(n_particles):
        for frame in range(n_points):
            rows.append({
                "particle": pid, "frame": frame,
                "x": float(rng.uniform(0, 5000)), "y": float(rng.uniform(0, 5000)),
            })
    import pandas as pd

    return pd.DataFrame(rows)


# --- frame rate and frame interval are one setting ---------------------------


def test_fps_and_interval_stay_in_sync(widget):
    widget.fps_box.setValue(50.0)
    assert widget.frame_interval_box.value() == pytest.approx(20.0)
    widget.frame_interval_box.setValue(5.0)
    assert widget.fps_box.value() == pytest.approx(200.0)
    widget.fps_box.setValue(100.0)
    assert widget.frame_interval_box.value() == pytest.approx(10.0)


# --- linking cutoff readout ---------------------------------------------------


def test_cutoff_readout_tracks_the_parameters(widget):
    widget.fps_box.setValue(100.0)      # 10 ms
    widget.search_box.setValue(250.0)
    widget.memory_box.setValue(0)
    text = widget.link_cutoff_label.text()
    assert "0.339" in text, text     # R^2 / (4 t ln 100) with R=250 nm, t=10 ms
    assert "116 nm" in text          # RMS step at that D

    widget.search_box.setValue(500.0)  # 4x the D for 2x the range
    assert "1.36" in widget.link_cutoff_label.text()

    widget.frame_interval_box.setValue(5.0)  # half the lag, double the D
    assert "2.71" in widget.link_cutoff_label.text()


def test_cutoff_readout_mentions_memory(widget):
    widget.memory_box.setValue(0)
    assert "memory" not in widget.link_cutoff_label.text()
    widget.memory_box.setValue(2)
    text = widget.link_cutoff_label.text()
    assert "memory 2" in text
    assert "gap" in text


def test_cutoff_readout_compares_against_measured_d(widget):
    widget.fps_box.setValue(100.0)
    widget.search_box.setValue(250.0)
    widget.memory_box.setValue(0)
    # Cutoff is 0.339; two of four trajectories are above it.
    widget._track_diffusion_cache = {0: 0.01, 1: 0.05, 2: 1.0, 3: 2.0}
    widget._update_link_cutoff_label()
    text = widget.link_cutoff_label.text()
    assert "50.0% of your 4 measured D values exceed it" in text
    assert "consider a larger search range" in text


# --- the display must not rebuild trajectory layers ---------------------------


def test_metric_bound_change_recolours_without_rebuilding(widget):
    widget.tracks = _tracks_frame()
    widget._track_diffusion_cache = {pid: 0.1 * (pid + 1) for pid in range(6)}
    widget.color_trajectories_box.setChecked(True)
    widget.show_tracks_box.setChecked(True)
    widget.show_all_tracks_box.setChecked(True)
    widget._sync_tracks_layer()
    widget._sync_all_tracks_layer()

    tracks_layer = widget.viewer.layers[widget_mod.TRACKS_LAYER_NAME]
    shapes_layer = widget.viewer.layers[widget_mod.ALL_TRACKS_LAYER_NAME]
    before_colors = np.array(shapes_layer.edge_color, dtype=object)
    widget.viewer.add_calls = 0

    widget.d_min_box.setValue(0.2)
    widget.d_max_box.setValue(0.4)
    widget._refresh_metric_colors()

    assert widget.viewer.add_calls == 0, "changing a colour bound rebuilt a layer"
    # Same layer objects, new colours.
    assert widget.viewer.layers[widget_mod.TRACKS_LAYER_NAME] is tracks_layer
    assert widget.viewer.layers[widget_mod.ALL_TRACKS_LAYER_NAME] is shapes_layer
    assert not np.array_equal(np.array(shapes_layer.edge_color, dtype=object), before_colors)


def test_colormap_change_recolours_but_metric_toggle_rebuilds(widget):
    """Changing what the colours mean needs a rebuild; changing the palette does not."""
    widget.tracks = _tracks_frame()
    widget.df_filtered = None
    widget._track_diffusion_cache = {pid: 0.1 * (pid + 1) for pid in range(6)}
    widget.color_trajectories_box.setChecked(True)
    widget.show_all_tracks_box.setChecked(True)
    widget._sync_all_tracks_layer()

    widget.viewer.add_calls = 0
    widget.d_colormap_box.setCurrentText("viridis")
    assert widget.viewer.add_calls == 0, "a colormap change rebuilt a layer"

    # Turning metric colouring off changes color_by, which does need a rebuild.
    widget.viewer.add_calls = 0
    widget.color_trajectories_box.setChecked(False)
    assert widget.viewer.add_calls > 0


def test_line_width_change_does_not_rebuild(widget):
    widget.tracks = _tracks_frame()
    widget.show_tracks_box.setChecked(True)
    widget.show_all_tracks_box.setChecked(True)
    widget._sync_tracks_layer()
    widget._sync_all_tracks_layer()
    widget.viewer.add_calls = 0

    widget.line_width_box.setValue(4.0)
    widget.all_tracks_line_width_box.setValue(2.5)

    assert widget.viewer.add_calls == 0
    assert widget.viewer.layers[widget_mod.TRACKS_LAYER_NAME].tail_width == 4.0
    assert widget.viewer.layers[widget_mod.ALL_TRACKS_LAYER_NAME].edge_width == 2.5


def test_apply_button_works_when_live_update_is_off(widget):
    widget.tracks = _tracks_frame()
    widget._track_diffusion_cache = {pid: 0.1 * (pid + 1) for pid in range(6)}
    widget.color_trajectories_box.setChecked(True)
    widget.show_all_tracks_box.setChecked(True)
    widget._sync_all_tracks_layer()
    shapes_layer = widget.viewer.layers[widget_mod.ALL_TRACKS_LAYER_NAME]

    widget.live_display_box.setChecked(False)
    before = np.array(shapes_layer.edge_color, dtype=object)
    widget.d_max_box.setValue(0.15)
    assert not widget._metric_render_timer.isActive(), "live update was off but a refresh was queued"

    widget.apply_display_settings()
    assert not np.array_equal(np.array(shapes_layer.edge_color, dtype=object), before)


# --- histogram view range follows the filter by default -----------------------


@pytest.mark.parametrize("key", ["D", "distance", "duration"])
def test_view_range_follows_filter_by_default(widget, key):
    state = widget._metric_hist_widgets[key]
    assert state["follow_box"].isChecked()
    assert not state["view_min_box"].isEnabled()

    min_box, max_box = widget._metric_bound_boxes[key]
    min_box.setValue(0.25)
    max_box.setValue(3.5)
    assert state["view_min_box"].value() == pytest.approx(0.25)
    assert state["view_max_box"].value() == pytest.approx(3.5)


@pytest.mark.parametrize("key", ["D", "distance", "duration"])
def test_view_range_can_be_decoupled(widget, key):
    state = widget._metric_hist_widgets[key]
    min_box, max_box = widget._metric_bound_boxes[key]
    min_box.setValue(0.1)
    max_box.setValue(1.0)

    state["follow_box"].setChecked(False)
    assert state["view_min_box"].isEnabled()
    state["view_max_box"].setValue(42.0)

    max_box.setValue(7.0)  # filter moves, view must not
    assert state["view_max_box"].value() == pytest.approx(42.0)

    state["follow_box"].setChecked(True)  # re-coupling snaps back to the filter
    assert state["view_max_box"].value() == pytest.approx(7.0)
