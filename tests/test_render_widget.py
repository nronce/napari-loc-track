"""The Render tab, driven through the real widget.

Two things matter here beyond "it produces an array". First, a reconstruction of
a long acquisition takes tens of seconds, so it has to run on a worker and be
interruptible - the tests check that the click returns before the render does,
that the result arrives afterwards, and that cancelling leaves nothing behind.
Second, a saved render is a result someone will put in a paper, so it must carry
the settings that produced it and land on the same grid as the raw stack.
"""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)
render = widget_mod.smlm_render

from test_widget_interaction import ensure_qapp, make_widget  # noqa: E402

PIXEL_SIZE_NM = 100.0
FRAMES = 200


def _locs(n=20000, seed=0, columns=None):
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({
        "frame": rng.integers(0, FRAMES, n),
        # 64 camera pixels across, at 100 nm/px
        "x [nm]": rng.uniform(0, 64 * PIXEL_SIZE_NM, n),
        "y [nm]": rng.uniform(0, 64 * PIXEL_SIZE_NM, n),
        "sigma [nm]": rng.uniform(100, 200, n),
        "intensity [photon]": rng.uniform(200, 5000, n),
        "uncertainty [nm]": rng.uniform(8, 40, n),
    })
    return table[columns] if columns else table


def _pump_until(predicate, timeout_s=120.0):
    """Run the Qt event loop until `predicate` holds, so worker signals arrive."""
    import time

    app = ensure_qapp()
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _loaded(n=20000, columns=None, image_shape=None):
    widget = make_widget()
    widget.pixel_size_box.setValue(PIXEL_SIZE_NM)
    widget._ingest_localization_dataframe(_locs(n, columns=columns), "loaded", True)
    if image_shape is not None:
        import napari

        layer = napari.layers.Image(np.zeros(image_shape, np.uint16), name="raw.tif")
        widget.viewer.layers["raw.tif"] = layer
        # The stub viewer has no inserted event, so the physical units the real
        # load path applies have to be applied by hand here.
        widget._apply_viewer_scale()
        widget._update_render_info()
    return widget


def _render_image(widget):
    widget.render_smlm_image()
    assert _pump_until(lambda: widget._render_worker_ref is None), "render never finished"
    return widget._render_image


def _render_movie(widget):
    widget.render_smlm_movie()
    assert _pump_until(lambda: widget._render_worker_ref is None), "render never finished"
    return widget._render_movie


def _in_field(table, shape):
    """How many localizations fall inside a `shape` camera-pixel field of view."""
    x = table["x [nm]"] / PIXEL_SIZE_NM
    y = table["y [nm]"] / PIXEL_SIZE_NM
    return int((x.between(-0.5, shape[1] - 0.5, inclusive="left")
                & y.between(-0.5, shape[0] - 0.5, inclusive="left")).sum())


# --- the tab and its state ---------------------------------------------------


def test_the_tabs_follow_the_pipeline():
    """Load, then detect, then filter, then the two things you can do with it."""
    widget = make_widget()
    titles = [widget.tabs.tabText(i) for i in range(widget.tabs.count())]
    assert titles == ["Load", "Localize", "Filter", "Render", "Track", "Save"]
    # the Localize tab is remembered by index, which a reshuffle must not break
    assert widget.tabs.tabText(widget._localize_tab_index) == "Localize"
    # the data table is a view, not a step: it opens on demand
    assert widget.data_table_dialog is not None
    assert not widget.data_table_dialog.isVisible()


def test_rendering_is_only_offered_once_there_is_something_to_render():
    widget = make_widget()
    assert not widget.render_image_button.isEnabled()
    assert not widget.render_movie_button.isEnabled()
    widget.pixel_size_box.setValue(PIXEL_SIZE_NM)
    widget._ingest_localization_dataframe(_locs(100), "loaded", True)
    assert widget.render_image_button.isEnabled()
    assert widget.render_movie_button.isEnabled()


def test_the_two_path_fields_stay_in_step():
    widget = make_widget()
    widget.csv_edit.setText("C:/data/locs.csv")
    assert widget.render_csv_edit.text() == "C:/data/locs.csv"
    widget.render_image_edit.setText("C:/data/stack.tif")
    assert widget.image_edit.text() == "C:/data/stack.tif"


def test_the_width_column_defaults_to_the_localization_precision():
    widget = _loaded(100)
    offered = [widget.render_sigma_column_box.itemText(i)
               for i in range(widget.render_sigma_column_box.count())]
    assert widget.render_sigma_column_box.currentText() == "uncertainty [nm]"
    assert "sigma [nm]" in offered  # the PSF width is offered as an alternative


@pytest.mark.parametrize("mode,shown,hidden", [
    ("gaussian_global", "render_sigma_box", "render_sigma_column_box"),
    ("gaussian_local", "render_sigma_column_box", "render_sigma_box"),
])
def test_only_the_controls_that_apply_are_shown(mode, shown, hidden):
    widget = _loaded(100)
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData(mode))
    assert not getattr(widget, shown).isHidden()
    assert getattr(widget, hidden).isHidden()


def test_photon_weighting_is_meaningless_for_a_scatter_render():
    widget = _loaded(100)
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("scatter"))
    assert not widget.render_photons_box.isEnabled()
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("histogram"))
    assert widget.render_photons_box.isEnabled()


# --- field of view -----------------------------------------------------------


def test_the_render_covers_the_image_when_there_is_one():
    widget = _loaded(1000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(4)
    shape, origin, layer = widget._render_field_of_view()
    assert shape == (48, 80)
    assert origin == (-0.5, -0.5)  # exactly the image extent
    assert layer is widget.viewer.layers["raw.tif"]
    assert "80 x 48 camera px -> 320 x 192" in widget.render_size_label.text()


def test_the_render_falls_back_to_the_localizations_extent():
    widget = _loaded(20000)
    shape, origin, layer = widget._render_field_of_view()
    assert shape == (64, 64)  # the whole camera pixels the data occupies
    assert origin == (-0.5, -0.5)
    assert layer is None


def test_a_render_layer_is_never_mistaken_for_the_raw_stack():
    """Renders are Image layers too - the plugin must not localize inside one."""
    import napari

    widget = _loaded(100, image_shape=(FRAMES, 48, 80))
    raw = widget.viewer.layers["raw.tif"]
    widget.viewer.layers[widget_mod.RENDER_LAYER_NAME] = napari.layers.Image(
        np.zeros((192, 320), np.float32), name=widget_mod.RENDER_LAYER_NAME)
    assert widget._source_image_layer() is raw
    assert widget._get_localize_image_layer() is raw


# --- what gets rendered ------------------------------------------------------


def test_the_click_returns_before_the_render_does():
    widget = _loaded(20000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(4)
    widget.render_smlm_image()
    # the work is on a thread: nothing is finished yet, and the user can stop it
    assert widget._render_image is None
    assert not widget.render_progress.isHidden()
    assert widget.render_cancel_button.isEnabled()

    assert _pump_until(lambda: widget._render_worker_ref is None)
    assert widget._render_image is not None
    assert widget.render_progress.isHidden()
    assert widget.render_image_button.isEnabled()
    assert widget.render_save_image_button.isEnabled()


def test_a_render_holds_exactly_the_localizations_it_was_given():
    widget = _loaded(20000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(4)
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("histogram"))
    image = _render_image(widget)
    assert image.shape == (48 * 4, 80 * 4)
    assert image.dtype == np.float32
    assert image.sum() == _in_field(widget.df_filtered, (48, 80))


def test_filtering_changes_what_is_reconstructed():
    widget = _loaded(20000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(4)
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("histogram"))
    lower, _upper = widget.filter_controls["intensity [photon]"]
    lower.setValue(3000.0)
    widget.apply_filters()
    kept = len(widget.df_filtered)
    assert 0 < kept < 20000
    assert f"{kept} localizations" in widget.render_source_label.text()

    image = _render_image(widget)
    assert image.sum() == _in_field(widget.df_filtered, (48, 80))
    assert widget._render_image_info["n_localizations"] == kept


def test_the_render_is_placed_on_top_of_the_raw_stack():
    """The viewer works in nanometres, so the render is placed in them too.

    A super-resolved pixel is a quarter of a camera pixel at 4x oversampling
    whatever the world is measured in; expressing that in nanometres is what
    lets napari's own scale bar describe the render and the raw stack at once.
    """
    widget = _loaded(1000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(4)
    _render_image(widget)
    layer = widget.viewer.layers[widget_mod.RENDER_LAYER_NAME]
    raw = widget.viewer.layers["raw.tif"]

    assert tuple(layer.scale) == pytest.approx((PIXEL_SIZE_NM / 4,) * 2)
    assert tuple(layer.translate) == pytest.approx((-0.375 * PIXEL_SIZE_NM,) * 2)
    # a quarter of a raw camera pixel, which is the part that must not drift
    assert layer.scale[-1] == pytest.approx(raw.scale[-1] / 4)
    assert layer.contrast_limits[1] > layer.contrast_limits[0]


def test_the_viewer_is_left_alone_when_the_box_is_unticked():
    widget = _loaded(1000, image_shape=(FRAMES, 48, 80))
    widget.render_add_layer_box.setChecked(False)
    _render_image(widget)
    assert widget_mod.RENDER_LAYER_NAME not in widget.viewer.layers


def test_photon_weighting_reaches_the_engine():
    widget = _loaded(5000, image_shape=(FRAMES, 48, 80))
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("histogram"))
    widget.render_photons_box.setChecked(True)
    options, info, _layer = widget._render_inputs()
    assert np.array_equal(options["weights"],
                          widget.df_filtered["intensity [photon]"].to_numpy())
    assert info["value_units"] == "photons per pixel"


def test_widths_are_converted_to_pixels_and_clamped():
    widget = _loaded(5000, image_shape=(FRAMES, 48, 80))
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("gaussian_local"))
    widget.render_sigma_min_box.setValue(10.0)
    widget.render_sigma_max_box.setValue(30.0)
    options, info, _layer = widget._render_inputs()
    assert options["sigma_px"].min() >= 10.0 / PIXEL_SIZE_NM - 1e-12
    assert options["sigma_px"].max() <= 30.0 / PIXEL_SIZE_NM + 1e-12
    assert info["sigma_clamp_nm"] == [10.0, 30.0]


# --- movies ------------------------------------------------------------------


def test_a_movie_groups_the_requested_number_of_raw_frames():
    widget = _loaded(20000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(4)
    widget.render_frames_per_box.setValue(50)
    # counts, so the total is exactly the number of localizations
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("histogram"))
    assert "4 super-resolved frames" in widget.render_movie_label.text()

    movie = _render_movie(widget)
    assert movie.shape == (4, 192, 320)
    assert movie.sum() == _in_field(widget.df_filtered, (48, 80))
    assert widget._render_movie_info["raw_frames_per_movie_frame"] == 50


def test_the_movie_time_axis_matches_the_raw_stack():
    widget = _loaded(5000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(2)
    widget.render_frames_per_box.setValue(50)
    _render_movie(widget)
    layer = widget.viewer.layers[widget_mod.RENDER_MOVIE_LAYER_NAME]
    # one movie frame spans 50 raw frames, so the dims slider still reads in
    # raw frames for the render and the stack alike
    assert layer.scale[0] == 50.0
    # and a group appears where its window closes, not where it opens: at raw
    # frame 0 the reconstruction of frames 0-49 must not already be on screen
    assert layer.translate[0] == 49.0


def test_a_movie_frame_never_shows_localizations_from_the_future():
    """The reconstruction must not run ahead of the stack that produced it.

    Placed at the first frame of its window, the render of frames 0..N-1 is
    already on screen at frame 0 - so the movie shows molecules before the
    images underneath have seen them, and a whole window ahead of the
    trajectories built from the same localizations.
    """
    widget = _loaded(5000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(2)
    per_group = 25
    widget.render_frames_per_box.setValue(per_group)
    _render_movie(widget)

    info = widget._render_movie_info
    layer = widget.viewer.layers[widget_mod.RENDER_MOVIE_LAYER_NAME]
    bounds = render.group_bounds(
        info["first_raw_frame"], info["last_raw_frame"], per_group,
        info["grouping"], info["window_step_frames"])

    for index, (_lo, hi) in enumerate(bounds):
        shown_at = layer.translate[0] + index * layer.scale[0]
        last_frame_in_group = hi - 1
        # never before the last frame that fed it
        assert shown_at >= last_frame_in_group, (
            f"movie frame {index} is shown at raw frame {shown_at}, "
            f"before frame {last_frame_in_group} which it was built from")


def test_a_cumulative_movie_fills_the_reconstruction_in():
    widget = _loaded(20000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(2)
    widget.render_frames_per_box.setValue(50)
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("histogram"))
    widget.render_grouping_box.setCurrentIndex(
        widget.render_grouping_box.findData("cumulative"))
    widget.render_scalebar_box.setChecked(False)  # measuring brightness, not annotations
    movie = _render_movie(widget)
    totals = [float(frame.sum()) for frame in movie]
    assert totals == sorted(totals) and totals[0] < totals[-1]
    assert totals[-1] == _in_field(widget.df_filtered, (48, 80))


def test_a_sliding_window_advances_by_its_step():
    widget = _loaded(20000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(2)
    widget.render_frames_per_box.setValue(50)
    widget.render_grouping_box.setCurrentIndex(
        widget.render_grouping_box.findData("sliding"))
    widget.render_step_box.setValue(25)
    assert not widget.render_step_box.isHidden()

    movie = _render_movie(widget)
    assert movie.shape[0] == len(render.group_bounds(0, FRAMES - 1, 50, "sliding", 25))
    assert widget.viewer.layers[widget_mod.RENDER_MOVIE_LAYER_NAME].scale[0] == 25.0


def test_heavy_window_overlap_is_pointed_out():
    widget = _loaded(1000)
    widget.render_grouping_box.setCurrentIndex(
        widget.render_grouping_box.findData("sliding"))
    widget.render_frames_per_box.setValue(1000)
    widget.render_step_box.setValue(1)
    assert "redrawn" in widget.render_movie_label.text()


# --- refusing to do something silly ------------------------------------------


def test_an_oversized_render_is_refused_rather_than_attempted(monkeypatch):
    widget = _loaded(1000, image_shape=(FRAMES, 48, 80))
    monkeypatch.setattr(widget_mod, "RENDER_MAX_BYTES", 100_000)
    widget.render_oversampling_box.setValue(8)

    before = len(widget.log_box.toPlainText())
    widget.render_smlm_image()
    message = widget.log_box.toPlainText()[before:]
    assert "Refusing to render" in message
    assert widget._render_worker_ref is None  # nothing was started
    assert "too large" in widget.render_size_label.text()


def test_an_oversized_movie_suggests_grouping_more_frames(monkeypatch):
    widget = _loaded(1000, image_shape=(FRAMES, 48, 80))
    monkeypatch.setattr(widget_mod, "RENDER_MAX_BYTES", 100_000)
    widget.render_frames_per_box.setValue(50)
    before = len(widget.log_box.toPlainText())
    widget.render_smlm_movie()
    assert "raw frames per super-resolved frame" in widget.log_box.toPlainText()[before:]


def test_rendering_without_data_says_so_instead_of_raising():
    widget = make_widget()
    widget.render_smlm_image()
    assert "Load or fit localizations" in widget.log_box.toPlainText()
    assert widget._render_image is None


def test_a_table_with_no_precision_column_is_reported_not_crashed():
    columns = ["frame", "x [nm]", "y [nm]", "intensity [photon]"]
    widget = _loaded(1000, columns=columns)
    widget.render_mode_box.setCurrentIndex(
        widget.render_mode_box.findData("gaussian_local"))
    assert widget._render_inputs() is None
    assert "precision column" in widget.log_box.toPlainText()

    # and the modes that do not need one still work
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("histogram"))
    assert _render_image(widget) is not None


def test_localizations_with_no_precision_still_reach_the_picture():
    widget = make_widget()
    widget.pixel_size_box.setValue(PIXEL_SIZE_NM)
    table = _locs(500, seed=3)
    table.loc[:99, "uncertainty [nm]"] = np.nan
    widget._ingest_localization_dataframe(table, "loaded", True)
    widget.render_mode_box.setCurrentIndex(
        widget.render_mode_box.findData("gaussian_local"))
    widget.render_oversampling_box.setValue(4)
    image = _render_image(widget)
    assert image.sum() == pytest.approx(500, rel=0.05)


def test_a_gpu_that_fails_part_way_finishes_on_the_cpu(monkeypatch):
    """Losing the GPU mid-render must cost speed, not the reconstruction."""
    widget = _loaded(2000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(2)
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("histogram"))

    real_iter = render.render_frame_iter
    calls = []

    def flaky(*args, **kwargs):
        calls.append(kwargs.get("gpu"))
        if kwargs.get("gpu"):
            raise RuntimeError("out of memory on device 0")
        return real_iter(*args, **kwargs)

    monkeypatch.setattr(render, "render_frame_iter", flaky)
    monkeypatch.setattr(render, "choose_backend", lambda *a, **k: (True, "GPU rendering"))

    image = _render_image(widget)
    assert calls == [True, False]
    assert image is not None and image.sum() == _in_field(widget.df_filtered, (48, 80))
    assert widget._render_image_info["backend"] == "cpu"
    assert "finished on the CPU" in widget.log_box.toPlainText()


def test_a_render_that_really_fails_is_reported(monkeypatch):
    widget = _loaded(1000, image_shape=(FRAMES, 48, 80))

    def broken(*_args, **_kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(render, "render_frame_iter", broken)
    widget.render_smlm_image()
    assert _pump_until(lambda: widget._render_worker_ref is None)
    assert "Rendering failed" in widget.log_box.toPlainText()
    assert widget._render_image is None
    assert widget.render_image_button.isEnabled()  # the tab is still usable


# --- cancelling --------------------------------------------------------------


def test_a_long_render_can_be_stopped_and_leaves_nothing_behind():
    widget = make_widget()
    widget.pixel_size_box.setValue(PIXEL_SIZE_NM)
    widget._ingest_localization_dataframe(_locs(200000, seed=7), "loaded", True)
    widget.render_oversampling_box.setValue(16)
    widget.render_frames_per_box.setValue(1)  # 200 frames: long enough to interrupt

    widget.render_smlm_movie()
    ensure_qapp().processEvents()
    widget._request_cancel(widget._render_cancel, widget.render_cancel_button, "rendering")

    assert _pump_until(lambda: widget._render_worker_ref is None, timeout_s=60)
    assert widget._render_movie is None  # no half-finished reconstruction kept
    assert "Rendering cancelled" in widget.log_box.toPlainText()
    assert widget.render_image_button.isEnabled()  # and the tab is usable again


# --- saving ------------------------------------------------------------------


def test_a_saved_render_carries_the_settings_that_produced_it(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    widget = _loaded(5000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(4)
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("histogram"))
    image = _render_image(widget)

    spec, extra = widget._save_spec("image", widget._render_image_info)
    metadata = widget._render_metadata({**widget._render_image_info, **extra})
    paths = widget_mod._save_render_worker(
        tmp_path / "recon.tif", image, spec, metadata,
        widget._render_image_info["super_resolved_pixel_size_nm"],
        True, "magma", None,
    ).work()

    assert sorted(path.name for path in paths) == [
        "recon.png", "recon.tif", "recon_metadata.json"]
    assert np.array_equal(tifffile.imread(tmp_path / "recon.tif"), image)

    sidecar = json.loads((tmp_path / "recon_metadata.json").read_text(encoding="utf-8"))
    assert sidecar["smlm_rendering"]["mode"] == "histogram"
    assert sidecar["smlm_rendering"]["oversampling"] == 4
    assert sidecar["smlm_rendering"]["n_localizations"] == 5000
    assert sidecar["super_resolved_pixel_size_nm"] == 25.0
    # the whole analysis snapshot, not just the render options: the picture can
    # be traced back to the localizations and the filters behind it
    assert sidecar["pixel_size_nm_per_px"] == PIXEL_SIZE_NM
    assert "intensity [photon]" in sidecar["filter_bounds"]
    assert "localization_2d" in sidecar
    with tifffile.TiffFile(tmp_path / "recon.tif") as handle:
        embedded = json.loads(handle.imagej_metadata["Info"])
        assert embedded["smlm_rendering"]["mode"] == "histogram"


def test_the_suggested_file_name_describes_the_render(tmp_path):
    widget = _loaded(1000, image_shape=(FRAMES, 48, 80))
    widget.csv_edit.setText(str(tmp_path / "locs.csv"))
    widget.render_oversampling_box.setValue(4)
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("histogram"))
    _render_image(widget)
    suggested = widget._default_render_path("image", widget._render_image_info)
    assert os.path.basename(suggested) == "locs_render_histogram_os4_data.tif"


def _saved(widget, kind, image, tmp_path, name="out.tif"):
    """Run a save the way the widget does, synchronously."""
    spec, extra = widget._save_spec(kind, widget._render_movie_info
                                    if kind == "movie" else widget._render_image_info)
    assert spec is not None
    info = widget._render_movie_info if kind == "movie" else widget._render_image_info
    metadata = widget._render_metadata({**info, **extra})
    paths = widget_mod._save_render_worker(
        tmp_path / name, image, spec, metadata,
        info["super_resolved_pixel_size_nm"], False, "magma", None,
    ).work()
    return paths, metadata


def test_a_movie_is_saved_light_by_default(tmp_path):
    """A movie is something you watch: 8-bit, a quarter of the float32 size."""
    tifffile = pytest.importorskip("tifffile")
    widget = _loaded(5000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(2)
    widget.render_frames_per_box.setValue(50)
    movie = _render_movie(widget)
    assert widget.render_movie_format_box.currentData() == "display"
    assert widget.render_image_format_box.currentData() == "data"

    _saved(widget, "movie", movie, tmp_path, "movie.tif")
    written = tifffile.imread(tmp_path / "movie.tif")
    assert written.dtype == np.uint8
    assert written.shape == movie.shape
    assert written.nbytes * 4 == movie.nbytes
    sidecar = json.loads((tmp_path / "movie_metadata.json").read_text(encoding="utf-8"))
    assert sidecar["dtype"] == "uint8"
    assert "display levels" in sidecar["value_units"]
    assert sidecar["smlm_rendering"]["save_format"] == "display"


def test_the_display_stretch_is_shared_by_every_frame(tmp_path):
    """Per-frame normalization would make the movie pulse; it must not."""
    tifffile = pytest.importorskip("tifffile")
    widget = _loaded(20000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(2)
    widget.render_frames_per_box.setValue(50)
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("histogram"))
    widget.render_grouping_box.setCurrentIndex(
        widget.render_grouping_box.findData("cumulative"))
    # annotations are burned in at full brightness on every frame, which is
    # exactly what this test must not measure
    widget.render_scalebar_box.setChecked(False)
    widget.render_timestamp_box.setChecked(False)
    movie = _render_movie(widget)

    _saved(widget, "movie", movie, tmp_path, "cumulative.tif")
    written = tifffile.imread(tmp_path / "cumulative.tif")
    # a cumulative movie gets brighter; with one shared range that must survive
    brightness = [float(frame.mean()) for frame in written]
    assert brightness == sorted(brightness)
    assert brightness[0] < brightness[-1]
    # every frame normalized on its own would put 255 in all of them
    assert written[0].max() < written[-1].max()


def test_a_data_save_still_holds_the_untouched_values(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    widget = _loaded(5000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(2)
    widget.render_frames_per_box.setValue(50)
    widget.render_movie_format_box.setCurrentIndex(
        widget.render_movie_format_box.findData("data"))
    movie = _render_movie(widget)
    assert movie.shape[0] > 1

    _saved(widget, "movie", movie, tmp_path, "raw.tif")
    written = tifffile.imread(tmp_path / "raw.tif")
    assert written.dtype == np.float32
    assert np.array_equal(written, movie)


def _composite_widget(n=5000):
    widget = _loaded(n, image_shape=(FRAMES, 48, 80))
    # off by default here: these tests count lit pixels, and an annotation
    # burned into every frame would be counted as signal
    widget.render_scalebar_box.setChecked(False)
    widget.render_oversampling_box.setValue(2)
    widget.render_frames_per_box.setValue(50)
    # No composite movie format: a blended movie is screen-recorded from the
    # viewer now rather than written frame by frame at the render's resolution.
    widget.render_image_format_box.setCurrentIndex(
        widget.render_image_format_box.findData("composite"))
    return widget


def _fake_tracks(widget, n_tracks=6, n_points=12, seed=2):
    """Trajectories in camera pixels, the way trackpy leaves them."""
    rng = np.random.default_rng(seed)
    rows = []
    for particle in range(n_tracks):
        x, y = rng.uniform(10, 70), rng.uniform(10, 38)
        for frame in range(n_points):
            x += rng.normal(0, 0.5)
            y += rng.normal(0, 0.5)
            rows.append({"particle": particle, "frame": frame * 5, "x": x, "y": y})
    widget.tracks = pd.DataFrame(rows)
    return widget.tracks


def test_a_composite_blends_the_reconstruction_and_the_overlays(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    widget = _composite_widget()
    _fake_tracks(widget)
    widget.render_composite_locs_box.setChecked(True)
    widget.render_composite_tracks_box.setChecked(True)
    image = _render_image(widget)

    paths, metadata = _saved(widget, "image", image, tmp_path, "composite.tif")
    written = tifffile.imread(tmp_path / "composite.tif")
    assert written.dtype == np.uint8
    assert written.shape == image.shape + (3,)          # RGB
    assert written.max() > 0
    included = metadata["smlm_rendering"]["composite_layers"]
    assert len(included) == 3
    assert any("reconstruction" in layer for layer in included)
    assert any("localizations" in layer for layer in included)
    assert any("trajectories" in layer for layer in included)


def test_composite_colours_land_in_their_own_channels(tmp_path):
    """A yellow overlay must not turn up in the blue channel."""
    widget = _composite_widget()
    _fake_tracks(widget)
    widget.render_composite_base_box.setChecked(False)
    widget.render_composite_locs_box.setChecked(False)
    widget.render_composite_tracks_box.setChecked(True)
    widget.render_tracks_color_box.setCurrentText("yellow")
    image = _render_image(widget)

    spec, _extra = widget._save_spec("image", widget._render_image_info)
    composite = widget_mod.build_save_array(image, spec)
    assert composite.shape == image.shape + (3,)
    assert composite[..., 0].max() > 0 and composite[..., 1].max() > 0  # red + green
    assert composite[..., 2].max() == 0                                # no blue
    # and the trajectories really are drawn, not an empty canvas
    assert (composite[..., 0] > 0).sum() > 100


def test_a_composite_with_nothing_ticked_is_refused():
    widget = _composite_widget()
    widget.render_composite_base_box.setChecked(False)
    widget.render_composite_locs_box.setChecked(False)
    widget.render_composite_tracks_box.setChecked(False)
    _render_image(widget)
    spec, extra = widget._save_spec("image", widget._render_image_info)
    assert spec is None and extra is None
    assert "tick at least one layer" in widget.log_box.toPlainText()


def test_asking_for_trajectories_without_linking_says_so():
    widget = _composite_widget()
    widget.tracks = None
    widget.render_composite_tracks_box.setChecked(True)
    _render_image(widget)
    spec, _extra = widget._save_spec("image", widget._render_image_info)
    assert "link them first" in widget.log_box.toPlainText()
    # the reconstruction is still composited, so the save is not lost
    assert [layer["source"] for layer in spec["layers"]] == ["base"]


def test_the_save_format_is_in_the_file_name():
    widget = _composite_widget()
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("histogram"))
    _render_image(widget)
    name = os.path.basename(widget._default_render_path("image", widget._render_image_info))
    assert name.endswith("_composite.tif")


# --- time stamp --------------------------------------------------------------


def test_a_movie_can_be_stamped_with_the_time(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    widget = _loaded(5000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(4)
    widget.render_frames_per_box.setValue(50)
    widget.fps_box.setValue(10.0)                 # 0.1 s per raw frame
    widget.render_timestamp_box.setChecked(True)
    widget.render_timestamp_size_box.setValue(20)
    movie = _render_movie(widget)

    _saved(widget, "movie", movie, tmp_path, "stamped.tif")
    written = tifffile.imread(tmp_path / "stamped.tif")
    assert written.shape == movie.shape

    # 50 raw frames per movie frame at 0.1 s each = 5 s apart, and each frame is
    # dated when its window closes (raw frame 49 = 4.9 s) rather than when it
    # opened - the same instant the viewer puts that frame at on the slider
    spec, extra = widget._save_spec("movie", widget._render_movie_info)
    assert spec["timestamp"]["labels"][:3] == ["4.9 s", "9.9 s", "14.9 s"]
    assert extra["timestamp"]["height_px"] == 20

    # the label is in the pixels, in the corner, and differs between frames
    corner = written[:, :30, :120]
    assert corner.max() > 200
    assert not np.array_equal(corner[0], corner[1])


def test_the_stamp_size_is_the_users_to_choose(tmp_path):
    widget = _loaded(2000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(8)
    widget.render_timestamp_box.setChecked(True)
    _render_image(widget)

    heights = {}
    for size in (12, 48):
        widget.render_timestamp_size_box.setValue(size)
        spec, _extra = widget._save_spec("image", widget._render_image_info)
        mask = render.compose_text(spec["timestamp"]["atlas"], "1 s")
        heights[size] = mask.shape[0]
    assert heights[48] > 2.5 * heights[12]


def test_a_data_save_is_never_written_over(tmp_path):
    """A float32 export is a measurement; a burned-in label would corrupt it."""
    tifffile = pytest.importorskip("tifffile")
    widget = _loaded(2000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(4)
    widget.render_timestamp_box.setChecked(True)
    image = _render_image(widget)               # image format defaults to data

    _saved(widget, "image", image, tmp_path, "untouched.tif")
    assert np.array_equal(tifffile.imread(tmp_path / "untouched.tif"), image)


# --- scale bar ---------------------------------------------------------------


def test_the_scale_bar_sizes_itself_to_the_field_of_view():
    widget = _loaded(2000, image_shape=(FRAMES, 48, 80))
    # 80 camera px at 100 nm = 8 um across
    assert widget._saved_width_nm() == pytest.approx(8000.0)
    assert widget.render_scalebar_auto_box.isChecked()
    assert widget.render_scalebar_length_box.value() == pytest.approx(1000.0)
    assert "1 µm" in widget.render_scalebar_status.text()

    # a finer camera pixel means a smaller view, so a shorter bar
    widget.pixel_size_box.setValue(20.0)
    widget._update_render_info()
    assert widget.render_scalebar_length_box.value() < 1000.0


def test_the_bar_follows_the_crop():
    widget = _loaded(2000, image_shape=(FRAMES, 48, 80))
    full = widget.render_scalebar_length_box.value()
    widget.render_crop_box.setChecked(True)
    widget.viewer.layers[widget_mod.RENDER_CROP_LAYER_NAME].data = [
        np.array([[20.0, 20.0], [20.0, 28.0], [28.0, 28.0], [28.0, 20.0]])]
    widget._update_render_crop_status()
    # zoomed in to 8 camera px = 800 nm, so the bar has to shrink
    assert widget._saved_width_nm() == pytest.approx(800.0)
    assert widget.render_scalebar_length_box.value() < full


def test_the_length_can_be_set_by_hand():
    widget = _loaded(2000, image_shape=(FRAMES, 48, 80))
    widget.render_scalebar_auto_box.setChecked(False)
    assert widget.render_scalebar_length_box.isEnabled()
    widget.render_scalebar_length_box.setValue(2500.0)
    widget.pixel_size_box.setValue(50.0)
    widget._update_render_info()
    assert widget.render_scalebar_length_box.value() == 2500.0  # not overwritten
    assert "2.5 µm" in widget.render_scalebar_status.text()


def test_the_bar_is_burned_into_a_saved_display_movie(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    widget = _loaded(5000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(4)
    widget.render_frames_per_box.setValue(50)
    widget.render_scalebar_position_box.setCurrentText("bottom right")
    movie = _render_movie(widget)

    _saved(widget, "movie", movie, tmp_path, "bar.tif")
    written = tifffile.imread(tmp_path / "bar.tif")
    # the bar is at full brightness in the bottom-right corner of every frame,
    # and in the same place each time (the reconstruction around it varies)
    corner = written[:, -60:, -180:]
    assert all(frame.max() == 255 for frame in corner)
    saturated = [set(zip(*np.nonzero(frame == 255))) for frame in corner]
    assert set.intersection(*saturated)

    sidecar = json.loads((tmp_path / "bar_metadata.json").read_text(encoding="utf-8"))
    bar = sidecar["smlm_rendering"]["scale_bar"]
    assert bar["length_nm"] == 1000.0
    assert bar["automatic"] is True
    # 1000 nm at a 25 nm super-resolved pixel
    assert bar["length_super_resolved_px"] == 40


def test_the_bar_is_the_length_it_claims(tmp_path):
    """The whole point of a scale bar: it has to measure what it says."""
    widget = _loaded(2000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(4)
    widget.render_scalebar_auto_box.setChecked(False)
    widget.render_scalebar_length_box.setValue(2000.0)   # 2 um
    widget.render_image_format_box.setCurrentIndex(
        widget.render_image_format_box.findData("display"))
    widget.render_timestamp_box.setChecked(False)
    image = _render_image(widget)

    spec, _extra = widget._save_spec("image", widget._render_image_info)
    blank = np.zeros_like(image)
    drawn = widget_mod.build_save_array(blank, spec)
    rows, cols = np.nonzero(drawn)
    # 2000 nm / 25 nm per super-resolved px = 80 px, and the solid bar is the
    # widest run in the annotation
    bottom = drawn[rows.max()]
    assert int((bottom > 0).sum()) == 80


def test_no_bar_when_it_is_turned_off():
    widget = _loaded(2000, image_shape=(FRAMES, 48, 80))
    widget.render_scalebar_box.setChecked(False)
    widget.render_timestamp_box.setChecked(False)
    widget.render_image_format_box.setCurrentIndex(
        widget.render_image_format_box.findData("display"))
    image = _render_image(widget)
    spec, extra = widget._save_spec("image", widget._render_image_info)
    assert spec["scalebar"] is None
    assert "scale_bar" not in extra
    assert widget_mod.build_save_array(np.zeros_like(image), spec).max() == 0


def test_a_data_save_is_never_annotated(tmp_path):
    """Neither the clock nor the bar may touch a quantitative export."""
    tifffile = pytest.importorskip("tifffile")
    widget = _loaded(2000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(4)
    widget.render_scalebar_box.setChecked(True)
    widget.render_timestamp_box.setChecked(True)
    image = _render_image(widget)                 # image format defaults to data

    _saved(widget, "image", image, tmp_path, "clean.tif")
    assert np.array_equal(tifffile.imread(tmp_path / "clean.tif"), image)


# --- crop box ----------------------------------------------------------------


def test_the_crop_box_appears_and_disappears_with_its_checkbox():
    widget = _loaded(2000, image_shape=(FRAMES, 48, 80))
    assert widget_mod.RENDER_CROP_LAYER_NAME not in widget.viewer.layers
    widget.render_crop_box.setChecked(True)
    assert widget_mod.RENDER_CROP_LAYER_NAME in widget.viewer.layers
    assert widget._render_crop_bounds() is not None
    widget.render_crop_box.setChecked(False)
    assert widget_mod.RENDER_CROP_LAYER_NAME not in widget.viewer.layers
    assert widget._render_crop_bounds() is None


def test_the_crop_box_can_be_resized_and_the_save_follows(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    widget = _loaded(20000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(4)
    widget.render_mode_box.setCurrentIndex(widget.render_mode_box.findData("histogram"))
    image = _render_image(widget)
    assert image.shape == (192, 320)

    widget.render_crop_box.setChecked(True)
    # drag it to a known rectangle, in camera pixels
    layer = widget.viewer.layers[widget_mod.RENDER_CROP_LAYER_NAME]
    layer.data = [np.array([[10.0, 20.0], [10.0, 40.0], [30.0, 40.0], [30.0, 20.0]])]
    widget._update_render_crop_status()
    assert widget._render_crop_bounds() == (10.0, 20.0, 30.0, 40.0)
    assert "80 x 80 super-resolved px" in widget.render_crop_status.text()

    _saved(widget, "image", image, tmp_path, "cropped.tif")
    written = tifffile.imread(tmp_path / "cropped.tif")
    assert written.shape == (80, 80)
    assert np.array_equal(written, image[42:122, 82:162])


def test_a_cropped_movie_keeps_every_frame(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    widget = _loaded(20000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(4)
    widget.render_frames_per_box.setValue(50)
    movie = _render_movie(widget)
    widget.render_crop_box.setChecked(True)
    widget.viewer.layers[widget_mod.RENDER_CROP_LAYER_NAME].data = [
        np.array([[10.0, 20.0], [10.0, 40.0], [30.0, 40.0], [30.0, 20.0]])]

    _saved(widget, "movie", movie, tmp_path, "cropped_movie.tif")
    written = tifffile.imread(tmp_path / "cropped_movie.tif")
    assert written.shape == (movie.shape[0], 80, 80)
    sidecar = json.loads(
        (tmp_path / "cropped_movie_metadata.json").read_text(encoding="utf-8"))
    assert sidecar["smlm_rendering"]["crop_camera_px"] == [10.0, 20.0, 30.0, 40.0]


def test_the_stamp_survives_the_crop(tmp_path):
    """Burning in before cropping would cut the label in half."""
    tifffile = pytest.importorskip("tifffile")
    widget = _loaded(20000, image_shape=(FRAMES, 48, 80))
    widget.render_oversampling_box.setValue(4)
    widget.render_frames_per_box.setValue(50)
    widget.render_timestamp_box.setChecked(True)
    widget.render_timestamp_size_box.setValue(16)
    movie = _render_movie(widget)
    widget.render_crop_box.setChecked(True)
    widget.viewer.layers[widget_mod.RENDER_CROP_LAYER_NAME].data = [
        np.array([[20.0, 20.0], [20.0, 44.0], [40.0, 44.0], [40.0, 20.0]])]

    _saved(widget, "movie", movie, tmp_path, "stamped_crop.tif")
    written = tifffile.imread(tmp_path / "stamped_crop.tif")
    assert written.shape[1:] == (80, 96)
    assert written[0, :25, :80].max() > 200     # the label is inside the crop


# --- every visible layer -----------------------------------------------------


def test_the_composite_can_take_in_the_rest_of_the_viewer():
    widget = _composite_widget()
    widget.render_composite_all_box.setChecked(True)
    _render_image(widget)
    spec, extra = widget._save_spec("image", widget._render_image_info)

    sources = [layer["source"] for layer in spec["layers"]]
    assert "base" in sources
    assert "image" in sources          # the raw stack came along
    assert any("raw.tif" in name for name in extra["composite_layers"])
    # the crop and filter boxes are controls, not data
    assert not any("roi" in name or "crop" in name
                   for name in extra["composite_layers"])


def test_the_raw_stack_shows_through_a_composite(tmp_path):
    """A blend of everything visible has to actually contain the camera image."""
    import napari

    widget = _composite_widget()
    # a raw stack with a bright stripe, so it is unmistakable in the output
    stack = np.zeros((FRAMES, 48, 80), np.uint16)
    stack[:, 10:14, :] = 4000
    widget.viewer.layers["raw.tif"] = napari.layers.Image(stack, name="raw.tif")
    widget.render_composite_base_box.setChecked(False)   # only the raw stack
    widget.render_composite_all_box.setChecked(True)
    widget.render_oversampling_box.setValue(4)
    image = _render_image(widget)

    spec, _extra = widget._save_spec("image", widget._render_image_info)
    composite = widget_mod.build_save_array(image, spec)
    assert composite.shape == image.shape + (3,)
    brightness = composite.max(axis=-1)
    stripe = brightness[40:56].mean()          # camera rows 10-14 at 4x
    elsewhere = brightness[100:150].mean()
    assert stripe > 10 * max(elsewhere, 1)


def test_a_hidden_layer_stays_out_of_the_composite():
    widget = _composite_widget()
    widget.render_composite_all_box.setChecked(True)
    widget.viewer.layers["raw.tif"].visible = False
    _render_image(widget)
    _spec, extra = widget._save_spec("image", widget._render_image_info)
    assert not any("raw.tif" in name for name in extra["composite_layers"])


# --- the x/y filter box ------------------------------------------------------


def test_the_filter_box_is_only_on_the_image_while_it_is_used():
    widget = _loaded(5000, image_shape=(FRAMES, 48, 80))
    filter_tab = [i for i in range(widget.tabs.count())
                  if widget.tabs.tabText(i) == "Filter"][0]
    render_tab = [i for i in range(widget.tabs.count())
                  if widget.tabs.tabText(i) == "Render"][0]

    widget.tabs.setCurrentIndex(filter_tab)
    assert widget_mod.ROI_LAYER_NAME in widget.viewer.layers  # it is the control

    widget.tabs.setCurrentIndex(render_tab)
    assert widget_mod.ROI_LAYER_NAME not in widget.viewer.layers  # not clutter

    # once it actually crops, it stays wherever you are - it is excluding data
    x_col = widget._resolve_column("x")
    lower, _upper = widget.filter_controls[x_col]
    lower.setValue(lower.value() + 1000.0)
    widget.apply_filters()
    assert widget_mod.ROI_LAYER_NAME in widget.viewer.layers

    lower.setValue(widget._default_bounds[x_col][0])
    widget.apply_filters()
    assert widget_mod.ROI_LAYER_NAME not in widget.viewer.layers


def test_saving_before_rendering_says_so():
    widget = _loaded(100)
    widget.save_render_image()
    assert "Render the image first" in widget.log_box.toPlainText()


# --- settings round-trip -----------------------------------------------------


def test_render_settings_survive_a_metadata_round_trip():
    widget = _loaded(1000)
    widget.render_oversampling_box.setValue(7)
    widget.render_mode_box.setCurrentIndex(
        widget.render_mode_box.findData("gaussian_local"))
    widget.render_sigma_box.setValue(33.0)
    widget.render_photons_box.setChecked(True)
    widget.render_grouping_box.setCurrentIndex(
        widget.render_grouping_box.findData("cumulative"))
    widget.render_step_box.setValue(17)
    widget.render_colormap_box.setCurrentText("viridis")
    widget.render_png_box.setChecked(False)
    snapshot = widget._collect_metadata("")

    restored = _loaded(1000)
    restored.apply_settings(snapshot)
    assert restored.render_oversampling_box.value() == 7
    assert restored.render_mode_box.currentData() == "gaussian_local"
    assert restored.render_sigma_box.value() == 33.0
    assert restored.render_photons_box.isChecked()
    assert restored.render_grouping_box.currentData() == "cumulative"
    assert restored.render_step_box.value() == 17
    assert restored.render_colormap_box.currentText() == "viridis"
    assert not restored.render_png_box.isChecked()
    assert restored.render_sigma_column_box.currentText() == "uncertainty [nm]"


def test_metadata_from_before_the_render_tab_existed_still_loads():
    widget = _loaded(1000)
    before = widget.render_mode_box.currentData()
    widget.apply_settings({"pixel_size_nm_per_px": 120.0})
    assert widget.pixel_size_box.value() == 120.0
    assert widget.render_mode_box.currentData() == before
