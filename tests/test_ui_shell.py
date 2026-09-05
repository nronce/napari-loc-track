"""The shell around the tabs: the status header, and one consistent look.

These are cheap tests for things that are easy to break silently and annoying
to notice - a count that stops updating, an action that goes missing when a tab
is reorganized, or a plot that reverts to matplotlib's white default and shows
as a bright rectangle inside a dark napari.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_widget_interaction import make_widget  # noqa: E402

PIXEL_SIZE_NM = 100.0


def _locs(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "frame": rng.integers(0, 50, n),
        "x [nm]": rng.uniform(0, 6400, n),
        "y [nm]": rng.uniform(0, 6400, n),
        "sigma [nm]": rng.uniform(100, 200, n),
        "intensity [photon]": rng.uniform(200, 5000, n),
        "uncertainty [nm]": rng.uniform(8, 40, n),
    })


def _loaded(n=500):
    widget = make_widget()
    widget.pixel_size_box.setValue(PIXEL_SIZE_NM)
    widget._ingest_localization_dataframe(_locs(n), "loaded", True)
    return widget


# --- status header -----------------------------------------------------------


def test_the_header_starts_empty_and_offers_nothing_to_act_on():
    widget = make_widget()
    assert widget.status_label.text() == "No data loaded"
    assert not widget.export_button.isEnabled()
    assert not widget.show_table_button.isEnabled()


def test_the_header_counts_the_localizations():
    widget = _loaded(500)
    assert "500 / 500 localizations" in widget.status_label.text()
    assert "100 nm/px" in widget.status_label.text()
    assert widget.export_button.isEnabled()
    assert widget.show_table_button.isEnabled()


def test_the_header_follows_the_filter():
    widget = _loaded(500)
    lower, _upper = widget.filter_controls["intensity [photon]"]
    lower.setValue(3000.0)
    widget.apply_filters()
    kept = len(widget.df_filtered)
    assert 0 < kept < 500
    assert f"{kept} / 500 localizations" in widget.status_label.text()


def test_the_header_reports_trajectories_and_forgets_them_again():
    widget = _loaded(200)
    widget.tracks = pd.DataFrame({
        "particle": np.repeat(np.arange(4), 5),
        "frame": np.tile(np.arange(5), 4),
        "x": np.zeros(20), "y": np.zeros(20),
    })
    widget._update_status_header()
    assert "4 trajectories" in widget.status_label.text()

    widget._invalidate_tracks(reason="filters changed")
    assert "trajectories" not in widget.status_label.text()


def test_the_header_names_the_loaded_image():
    import napari

    widget = _loaded(200)
    widget.viewer.layers["raw.tif"] = napari.layers.Image(
        np.zeros((30, 48, 80), np.uint16), name="raw.tif")
    widget._update_status_header()
    assert "raw.tif" in widget.status_label.text()
    assert "30x48x80" in widget.status_label.text()


# --- one place per action ----------------------------------------------------


def test_the_tabs_are_the_pipeline_and_nothing_else():
    widget = make_widget()
    assert [widget.tabs.tabText(i) for i in range(widget.tabs.count())] == [
        "Load", "Localize", "Filter", "Track", "Render", "Save"]


def test_export_is_offered_once():
    """It used to be wired from two buttons in two different tabs."""
    widget = _loaded(200)
    buttons = [child for child in widget.findChildren(widget_mod.QPushButton)
               if "export" in child.text().lower()]
    assert len(buttons) == 1
    assert buttons[0] is widget.export_button


def test_the_data_table_opens_as_a_dialog():
    widget = _loaded(200)
    assert not widget.data_table_dialog.isVisible()
    widget.show_data_table()
    assert widget.data_table_dialog.isVisible()
    assert "200 rows" in widget.data_table_label.text()
    widget.data_table_dialog.hide()


def test_linking_and_analysing_share_one_tab():
    widget = make_widget()
    track_tab = [i for i in range(widget.tabs.count())
                 if widget.tabs.tabText(i) == "Track"][0]
    page = widget.tabs.widget(track_tab)
    titles = {box.title() for box in page.findChildren(widget_mod.QGroupBox)}
    assert any("Tracking" in title for title in titles)          # from Link
    assert any("Diffusion" in title for title in titles)         # from analysis


# --- shifting frame numbers --------------------------------------------------


def test_a_one_indexed_table_is_shifted_on_load():
    """The commonest cause of localizations landing on the wrong frame."""
    widget = make_widget()
    widget.pixel_size_box.setValue(PIXEL_SIZE_NM)
    table = _locs(200)
    table["frame"] = table["frame"] + 1          # first frame numbered 1
    widget._ingest_localization_dataframe(table, "loaded", frame_is_zero_indexed=False)
    assert widget._frame_shift == -1
    assert int(widget._render_frames().min()) == 0
    assert "-1" in widget.frame_shift_label.text()


def test_the_buttons_move_every_frame_number():
    widget = _loaded(200)
    before = widget._render_frames().min()
    widget.shift_frame_numbers(+1)
    assert widget._render_frames().min() == before + 1
    widget.shift_frame_numbers(+1)
    assert widget._frame_shift == 2
    widget.shift_frame_numbers(-1)
    assert widget._frame_shift == 1
    assert "+1" in widget.frame_shift_label.text()


def test_the_shift_can_be_put_back():
    widget = _loaded(200)
    widget.apply_filters()   # settle on the filtered set before comparing
    original = widget._render_frames().copy()
    widget.shift_frame_numbers(-3)
    assert not np.array_equal(widget._render_frames(), original)
    widget.shift_frame_numbers(None)
    assert widget._frame_shift == 0
    assert np.array_equal(widget._render_frames(), original)
    assert widget.frame_shift_label.text() == "no shift"
    assert not widget.frame_shift_reset_button.isEnabled()


def test_shifting_reaches_the_layers_and_the_tracking():
    widget = _loaded(200)
    widget.shift_frame_numbers(+5)
    # the points layer carries the shifted frame on its first axis
    coords = np.asarray(widget.viewer.layers[widget_mod.POINTS_LAYER_NAME].data)
    assert coords[:, 0].min() == widget.df["frame"].min() + 5
    # and so does what gets handed to trackpy
    assert widget._prepare_features()["frame"].min() == widget.df["frame"].min() + 5


def test_the_old_tick_box_is_gone_but_its_setting_still_loads():
    widget = _loaded(200)
    assert not hasattr(widget, "frame_one_indexed_box")
    # a metadata.json from before the buttons existed
    widget.apply_settings({"frame_one_indexed": True})
    assert widget._frame_shift == -1
    widget.apply_settings({"frame_one_indexed": False})
    assert widget._frame_shift == 0


# --- the Save tab ------------------------------------------------------------


def test_rendering_comes_after_tracking():
    """A reconstruction is as often built from a dynamics selection as from
    every localization, and that selection needs the trajectories first."""
    widget = make_widget()
    titles = [widget.tabs.tabText(i) for i in range(widget.tabs.count())]
    assert titles.index("Render") == titles.index("Track") + 1


def test_saving_comes_last():
    widget = make_widget()
    titles = [widget.tabs.tabText(i) for i in range(widget.tabs.count())]
    assert titles.index("Save") == len(titles) - 1


def test_the_render_save_buttons_live_on_the_save_tab_only():
    widget = make_widget()
    save_page = widget.tabs.widget(
        [i for i in range(widget.tabs.count())
         if widget.tabs.tabText(i) == "Save"][0])
    # "Save session..." is deliberately not one of these: a session is state to
    # be reopened rather than a render written out, so it sits beside Load data.
    saves = [b for b in widget.findChildren(widget_mod.QPushButton)
             if b.text().startswith("Save ") and "session" not in b.text()]
    # image, movie, and a composite still - a composite *movie* is screen-
    # recorded from the viewer now, not written frame by frame from here
    assert len(saves) == 3
    for button in saves:
        assert save_page.isAncestorOf(button)


def test_the_session_buttons_sit_with_the_data_loading_controls():
    widget = make_widget()
    load_page = widget.tabs.widget(
        [i for i in range(widget.tabs.count())
         if widget.tabs.tabText(i) == "Load"][0])
    for button in (widget.save_session_button, widget.load_session_button):
        assert load_page.isAncestorOf(button)


def test_nothing_is_offered_until_something_is_rendered():
    widget = _loaded(200)
    assert "Nothing rendered yet" in widget.render_save_status.text()
    for button in (widget.render_save_image_button,
                   widget.render_save_composite_image_button,
                   widget.render_save_movie_button):
        assert not button.isEnabled()


def test_a_composite_button_composites_whatever_the_format_box_says():
    widget = _loaded(200)
    widget.render_image_format_box.setCurrentIndex(
        widget.render_image_format_box.findData("data"))
    widget.render_smlm_image()
    assert _pump(lambda: widget._render_worker_ref is None)

    assert widget.render_save_image_button.isEnabled()
    assert "Ready to save: image" in widget.render_save_status.text()

    plain, _extra = widget._save_spec("image", widget._render_image_info)
    assert plain["format"] == "data"
    forced, extra = widget._save_spec("image", widget._render_image_info,
                                      force_format="composite")
    assert forced["format"] == "composite"
    assert extra["save_format"] == "composite"
    assert forced["layers"]                      # something to blend
    # and the suggested file name follows the format actually written
    assert widget._default_render_path(
        "image", widget._render_image_info, "composite").endswith("_composite.tif")


def _pump(predicate, timeout_s=120.0):
    import time

    from test_widget_interaction import ensure_qapp

    app = ensure_qapp()
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


# --- consistent look ---------------------------------------------------------


def test_the_primary_action_of_each_tab_is_marked_as_such():
    widget = _loaded(200)
    for button in (widget.load_button, widget.loc_detect_button, widget.loc_fit_button,
                   widget.apply_filters_button, widget.render_image_button,
                   widget.link_button, widget.export_button,
                   widget.render_save_image_button, widget.render_save_movie_button):
        assert button.property("primary") is True, button.text()
    for button in (widget.reset_filters_button, widget.show_table_button,
                   widget.render_save_composite_image_button):
        assert button.property("secondary") is True, button.text()
    # cancels are marked apart from both, so stopping never looks like doing
    assert widget.load_cancel_button.property("stop") is True
    assert widget.render_cancel_button.property("stop") is True


def test_the_accent_is_lightseagreen():
    assert widget_mod.ACCENT.lower() == "#20b2aa"
    assert widget_mod.ACCENT in widget_mod.STYLESHEET
    assert widget_mod.LAVENDER in widget_mod.STYLESHEET


@pytest.mark.parametrize("figure_attr", [
    "loc_counts_figure",   # Localize tab
    "msd_figure",          # Track tab
])
def test_every_figure_is_drawn_on_pure_black(figure_attr):
    """Plots get screenshotted onto slides, where near-black is a grey box.

    Black rather than the panel's own shade: the panel sits inside napari and
    matches it, but a figure lifted out of the panel and dropped on a slide has
    to disappear into the slide instead.
    """
    widget = _loaded(200)
    figure = getattr(widget, figure_attr)
    axes = figure.add_subplot(111)
    widget_mod.style_axes(figure, axes)
    assert figure.patch.get_facecolor()[:3] == pytest.approx((0.0, 0.0, 0.0), abs=0.01)
    assert axes.get_facecolor()[:3] == pytest.approx((0.0, 0.0, 0.0), abs=0.01)
    # and the writing on it is light enough to read off a projector
    assert axes.xaxis.label.get_color() == widget_mod.INK


def _rgb(hex_colour):
    hex_colour = hex_colour.lstrip("#")
    return tuple(int(hex_colour[i:i + 2], 16) / 255 for i in (0, 2, 4))


def test_the_detection_plot_themes_itself_when_drawn():
    widget = _loaded(200)
    widget._loc2d_counts = np.arange(20)
    widget._draw_loc2d_counts()
    assert widget.loc_counts_figure.patch.get_facecolor()[:3] == pytest.approx(
        _rgb(widget_mod.PLOT_BG), abs=0.01)
    # and it is drawn in the accent, not matplotlib's default blue
    line = widget.loc_counts_figure.axes[0].lines[0]
    assert line.get_color().lower() == widget_mod.ACCENT


# --- plot size ----------------------------------------------------------------
#
# These plots end up in talks, and a figure that is the right shape is most of
# what makes one look deliberate - but resizing a dozen by hand is exactly the
# effort nobody spends. One control, applied live, saved with the run.


def _canvas_sizes(widget):
    return sorted({(c.width() if c.maximumWidth() < 16777215 else 0, c.height())
                   for c in widget._plot_canvases})


def test_one_control_sizes_every_plot():
    widget = _loaded(400)
    assert len(widget._plot_canvases) > 5      # filters, metrics, MSD, counts
    widget._set_plot_size(960, 540)
    assert _canvas_sizes(widget) == [(960, 540)]


def test_the_default_leaves_them_filling_the_panel():
    """Width 0 reads as 'fill' and is what these did before there was a control."""
    widget = _loaded(400)
    assert widget.plot_width_box.value() == widget_mod.PLOT_WIDTH_FILL
    assert all(width == 0 for width, _height in _canvas_sizes(widget))


def test_going_back_to_fill_releases_the_pinned_width():
    widget = _loaded(400)
    widget._set_plot_size(1200, 400)
    assert _canvas_sizes(widget) == [(1200, 400)]

    widget.plot_width_box.setValue(widget_mod.PLOT_WIDTH_FILL)
    assert _canvas_sizes(widget) == [(0, 400)]


def test_the_size_survives_loading_new_data():
    """The filter histograms are rebuilt per table, so the size has to be
    re-applied or it silently reverts on the next load."""
    widget = _loaded(400)
    widget._set_plot_size(1200, 400)
    before = len(widget._plot_canvases)

    widget._ingest_localization_dataframe(_locs(300), "reloaded", True)
    assert _canvas_sizes(widget) == [(1200, 400)]
    # and the dead canvases went with the panel that held them
    assert len(widget._plot_canvases) == before


def test_the_size_is_saved_with_the_run():
    widget = _loaded(200)
    widget._set_plot_size(1200, 400)
    metadata = widget._collect_metadata(None)
    assert metadata["rendering"]["plot_width_px"] == 1200
    assert metadata["rendering"]["plot_height_px"] == 400

    restored = make_widget()
    restored.apply_settings(metadata)
    assert restored.plot_width_box.value() == 1200
    assert restored.plot_height_box.value() == 400
