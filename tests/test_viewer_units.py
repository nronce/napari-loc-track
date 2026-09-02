"""The viewer works in nanometres, so napari's own scale bar can size itself.

napari draws its scale bar from world coordinates, in the corner of the viewport
and above every layer, following pan and zoom and choosing its own round length.
All of that is free - but only if the world is a physical space rather than a
grid of camera pixels, which is what these tests pin down.

The invariant that matters most is the one in the middle: *scale is a display
transform only*. Detection, linking and the render grid all work in camera
pixels, and every layer's `data` must stay in them. If that ever stops being
true, the analysis silently changes meaning while still looking right.

These run against a real `ViewerModel` rather than the stub viewer the other
widget tests use: layer scale, per-axis units and their propagation to
`dims.units` are exactly what is under test, so a stand-in would prove nothing.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_widget_interaction import ensure_qapp  # noqa: E402

PIXEL_SIZE_NM = 161.0
FRAMES = 20

# Same reasoning as the other widget tests: each widget owns a dozen matplotlib
# canvases, and letting Qt collect them mid-session crashes the interpreter.
_WIDGETS = []


def _widget(pixel_size_nm=PIXEL_SIZE_NM):
    from napari.components import ViewerModel

    ensure_qapp()
    widget = widget_mod.LocalizationTrackingWidget(ViewerModel())
    _WIDGETS.append(widget)
    widget.pixel_size_box.setValue(pixel_size_nm)
    return widget


def _load_image(widget, shape=(6, 32, 32), name="stack.tif"):
    """Drive the real load handler, the way a finished load worker would."""
    image = np.zeros(shape, np.uint16)
    widget._on_load_finished((None, image, "decoded", None), "", name)
    return widget.viewer.layers[name]


def _locs(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "frame": rng.integers(0, FRAMES, n),
        "x [nm]": rng.uniform(0, 32 * PIXEL_SIZE_NM, n),
        "y [nm]": rng.uniform(0, 32 * PIXEL_SIZE_NM, n),
        "sigma [nm]": rng.uniform(100, 200, n),
        "intensity [photon]": rng.uniform(200, 5000, n),
        "uncertainty [nm]": rng.uniform(8, 40, n),
    })


def _with_locs(n=4000):
    widget = _widget()
    _load_image(widget, (FRAMES, 32, 32))
    widget._ingest_localization_dataframe(_locs(n), "loaded", True)
    return widget


def _unit_names(layer):
    return tuple(str(u) for u in layer.units)


# --- the world the scale bar reads -------------------------------------------


def test_an_image_is_placed_in_nanometres_as_soon_as_it_loads():
    widget = _widget()
    layer = _load_image(widget)
    assert tuple(layer.scale) == pytest.approx((1.0, PIXEL_SIZE_NM, PIXEL_SIZE_NM))
    assert _unit_names(layer)[-2:] == ("nanometer", "nanometer")


def test_the_frame_axis_is_not_a_length():
    """Or napari would label the dims slider in nanometres."""
    widget = _widget()
    layer = _load_image(widget)
    assert layer.scale[0] == 1.0
    assert _unit_names(layer)[0] == "pixel"


def test_the_scale_bar_is_on_and_out_of_the_way_once_an_image_loads():
    widget = _widget()
    assert not widget.viewer.scale_bar.visible
    _load_image(widget)
    assert widget.viewer.scale_bar.visible
    assert widget.viewer.scale_bar.position == "bottom_right"


def test_loading_frames_the_whole_field_rather_than_a_corner_of_it():
    """A world in nanometres is ~160x "bigger" than the same one in pixels.

    A camera left where the previous world put it then opens deep inside the
    first field of view, which looks like the plugin has zoomed in on nothing.
    """
    widget = _widget()
    _load_image(widget, (6, 64, 64))
    camera = widget.viewer.camera
    extent = widget.viewer.layers.extent.world[:, -2:]
    span = float(np.max(extent[1] - extent[0]))
    # the view covers the stack rather than a fraction of one corner of it
    assert camera.zoom > 0
    assert span / camera.zoom > 0  # a real, finite field
    centre = np.asarray(camera.center)[-2:]
    middle = (extent[0] + extent[1]) / 2.0
    assert np.allclose(centre, middle, atol=span * 0.05)


def test_the_units_reach_the_viewer_not_just_the_layer():
    """`dims.units` is what napari actually sizes the bar from."""
    widget = _widget()
    _load_image(widget)
    assert tuple(str(u) for u in widget.viewer.dims.units)[-2:] \
        == ("nanometer", "nanometer")


def test_a_render_is_labelled_as_well_as_placed():
    """One layer left in pixels costs the scale bar its units entirely.

    napari refuses to mix units: a single layer still saying "pixel" makes it
    warn "inconsistent units across layers" and drop them all, and the bar
    silently goes back to counting pixels while everything still looks right.
    """
    from test_render_widget import _pump_until

    widget = _with_locs()
    widget.render_smlm_image()
    assert _pump_until(lambda: widget._render_worker_ref is None), "render never finished"

    render_layer = widget.viewer.layers[widget_mod.RENDER_LAYER_NAME]
    assert _unit_names(render_layer) == ("nanometer", "nanometer")
    assert widget.viewer.layers.extent.units is not None


def test_every_layer_the_plugin_makes_keeps_the_units_consistent():
    """The exact check napari runs before it will use units at all.

    `_update_world_units` reads `viewer.layers.extent.units`, and warns and
    falls back to pixels the moment that is None - so this walks the whole
    pipeline and asserts it after each layer appears, rather than testing the
    layers one at a time and missing the combination that breaks it.

    It is also why units are passed to `add_*` instead of being patched on
    afterwards: napari defers this check to the next draw, so a layer inserted
    in pixels and corrected a moment later can still be caught in between.
    """
    from test_render_widget import _pump_until

    widget = _with_locs()
    steps = []

    def still_consistent(label):
        steps.append((label, widget.viewer.layers.extent.units))

    still_consistent("localizations")
    widget.render_smlm_image()
    assert _pump_until(lambda: widget._render_worker_ref is None)
    still_consistent("render image")
    widget.render_smlm_movie()
    assert _pump_until(lambda: widget._render_worker_ref is None)
    still_consistent("render movie")
    widget.render_crop_box.setChecked(True)
    still_consistent("crop box")
    # Trajectories put in directly rather than linked: the linker's own worker is
    # tested elsewhere, and what matters here is the layers it ends up producing.
    widget.tracks = pd.DataFrame([
        {"particle": pid, "frame": frame, "x": 4.0 + frame, "y": 4.0 + pid}
        for pid in range(4) for frame in range(8)
    ])
    widget.render_overlay()
    still_consistent("tracks")
    widget.show_all_tracks_box.setChecked(True)
    still_consistent("all trajectories")

    broken = [label for label, units in steps if units is None]
    assert not broken, f"units went inconsistent after: {broken}"
    # and the layers really are a mix of 2D and 3D, which is the awkward part
    assert {widget.viewer.layers[n].ndim for n in
            (widget_mod.RENDER_LAYER_NAME, widget_mod.TRACKS_LAYER_NAME)} == {2, 3}


# --- what must not change -----------------------------------------------------


def test_layer_data_stays_in_camera_pixels():
    """The invariant the whole analysis rests on: scale is display only.

    Detection, linking and the render grid all index camera pixels. If putting
    the viewer in nanometres ever reached the arrays, every one of them would
    quietly start meaning something else.
    """
    widget = _with_locs()
    image = widget.viewer.layers["stack.tif"]
    assert image.data.shape == (FRAMES, 32, 32)
    assert image.data.max() == 0

    points = widget.viewer.layers[widget_mod.POINTS_LAYER_NAME]
    coords = np.asarray(points.data)
    # localizations span 32 camera pixels, not 32 * 161 nanometres
    assert coords[:, -2:].max() < 32.0


def test_the_crop_box_is_still_read_in_camera_pixels():
    widget = _with_locs()
    widget.render_crop_box.setChecked(True)
    box = widget._render_crop_bounds()
    assert box is not None
    # the middle half of a 32 px field, in camera pixels
    assert box == pytest.approx((7.5, 7.5, 23.5, 23.5))


def test_the_roi_filter_is_still_read_in_camera_pixels():
    """`_on_roi_changed` multiplies the box by the pixel size to get nanometres.

    If the shapes layer's data had become nanometres, that multiplication would
    apply the pixel size twice and the filter would exclude everything.
    """
    widget = _with_locs()
    layer = widget.viewer.add_shapes(
        [np.array([[4.0, 4.0], [4.0, 20.0], [20.0, 20.0], [20.0, 4.0]])],
        shape_type="rectangle", name=widget_mod.ROI_LAYER_NAME,
    )
    assert np.asarray(layer.data[0]).max() == pytest.approx(20.0)
    widget._on_roi_changed()
    x_col = widget._resolve_column("x")
    if x_col in widget.filter_controls:
        assert widget.filter_controls[x_col][1].value() == pytest.approx(
            20.0 * PIXEL_SIZE_NM, rel=1e-3)


# --- keeping in step ----------------------------------------------------------


def test_changing_the_pixel_size_stretches_the_world_with_it():
    """A bar sized by a stale pixel size is worse than no bar - it looks certain."""
    widget = _widget()
    layer = _load_image(widget)
    widget.pixel_size_box.setValue(80.5)
    assert tuple(layer.scale) == pytest.approx((1.0, 80.5, 80.5))
    assert layer.data.shape == (6, 32, 32)  # and the data did not move


def test_every_layer_shares_one_world():
    """Localizations, tracks and the stack must agree, or the bar describes one."""
    widget = _with_locs()
    widget._ingest_localization_dataframe(_locs(), "loaded", True)
    spatial = {
        name: tuple(np.ravel(widget.viewer.layers[name].scale))[-2:]
        for name in (n for n in ("stack.tif", widget_mod.POINTS_LAYER_NAME)
                     if n in widget.viewer.layers)
    }
    assert len(spatial) >= 2
    for scale in spatial.values():
        assert scale == pytest.approx((PIXEL_SIZE_NM, PIXEL_SIZE_NM))


def test_a_layer_the_user_drags_in_is_put_in_the_same_world():
    """Otherwise the scale bar would be describing only some of what is shown."""
    widget = _widget()
    _load_image(widget)
    foreign = widget.viewer.add_image(np.zeros((8, 8), np.uint16), name="theirs.tif")
    assert tuple(foreign.scale) == pytest.approx((PIXEL_SIZE_NM, PIXEL_SIZE_NM))
    assert _unit_names(foreign) == ("nanometer", "nanometer")


def test_a_render_is_placed_in_the_same_world_as_the_stack():
    from test_render_widget import _pump_until

    widget = _with_locs()
    widget.render_oversampling_box.setValue(4)
    widget.render_smlm_image()
    assert _pump_until(lambda: widget._render_worker_ref is None), "render never finished"

    render_layer = widget.viewer.layers[widget_mod.RENDER_LAYER_NAME]
    raw = widget.viewer.layers["stack.tif"]
    # a super-resolved pixel is a quarter of a camera pixel, in nanometres
    assert render_layer.scale[-1] == pytest.approx(raw.scale[-1] / 4)


def test_a_render_moves_with_the_world_when_the_pixel_size_changes():
    """Its transform is derived from the layer beneath, so it is carried along."""
    from test_render_widget import _pump_until

    widget = _with_locs()
    widget.render_oversampling_box.setValue(4)
    widget.render_smlm_image()
    assert _pump_until(lambda: widget._render_worker_ref is None), "render never finished"

    render_layer = widget.viewer.layers[widget_mod.RENDER_LAYER_NAME]
    before_scale = np.array(render_layer.scale, dtype=float)
    before_translate = np.array(render_layer.translate, dtype=float)

    widget.pixel_size_box.setValue(PIXEL_SIZE_NM / 2.0)
    assert np.allclose(render_layer.scale, before_scale / 2.0)
    assert np.allclose(render_layer.translate, before_translate / 2.0)
    # still a quarter of a camera pixel, which is what must never drift
    raw = widget.viewer.layers["stack.tif"]
    assert render_layer.scale[-1] == pytest.approx(raw.scale[-1] / 4)


def test_the_canvas_is_pure_black_for_screenshots():
    """A screenshot goes onto a slide, where near-black is a visible grey box."""
    widget = _widget()
    assert widget_mod.canvas_is_black(widget.viewer.theme)


def test_only_a_theme_napari_itself_knows_is_ever_persisted():
    """`viewer.theme` is written to napari's *global* settings.

    A theme id invented by this plugin therefore lands in a config file that
    plain napari - launched without the plugin that registers it - cannot
    resolve, and it reports a validation error and resets the field on every
    start. Only built-in names may go there.
    """
    from napari.utils.theme import available_themes

    widget = _widget()
    assert str(widget.viewer.theme) in available_themes()
    assert widget_mod.BLACK_CANVAS_THEME in ("dark", "light", "system")


def test_a_viewer_already_on_a_black_canvas_is_left_alone():
    from napari.components import ViewerModel

    viewer = ViewerModel()
    viewer.theme = "dark"
    widget_mod.apply_black_canvas(viewer)
    assert viewer.theme == "dark"


def test_a_light_viewer_is_switched_to_one_that_is_black():
    from napari.components import ViewerModel

    viewer = ViewerModel()
    viewer.theme = "light"
    assert not widget_mod.canvas_is_black(viewer.theme)
    widget_mod.apply_black_canvas(viewer)
    assert widget_mod.canvas_is_black(viewer.theme)
