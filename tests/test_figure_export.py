"""Lifting one graph out of the panel and onto a slide.

Transparent rather than black: the plots are drawn light-on-dark for napari, so
with the background dropped the axes, labels and data keep their light colours
and sit on whatever the slide provides. One click, no dialog - eight plots is
eight trips through a file browser otherwise, which is what stops anyone
building a figure set at all.
"""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)
Image = pytest.importorskip("PIL.Image", reason="needs pillow to inspect the PNG")

from test_widget_interaction import make_widget  # noqa: E402

_WIDGETS = []


def _loaded(tmp_path, n=600):
    widget = make_widget()
    _WIDGETS.append(widget)
    (tmp_path / "stack.tif").write_bytes(b"not a real tiff")
    widget.image_edit.setText(str(tmp_path / "stack.tif"))
    rng = np.random.default_rng(0)
    widget._ingest_localization_dataframe(pd.DataFrame({
        "frame": rng.integers(0, 20, n),
        "x [nm]": rng.uniform(0, 1e4, n),
        "y [nm]": rng.uniform(0, 1e4, n),
        "sigma [nm]": rng.uniform(80, 400, n),
        "intensity [photon]": rng.uniform(100, 3000, n),
        "uncertainty [nm]": rng.uniform(8, 40, n),
    }), "loaded", True)
    return widget


def _png_buttons(widget):
    return [b for b in widget.findChildren(widget_mod.QPushButton)
            if b.text() == "Save…"]


def _saved(widget, column="sigma [nm]", name="check"):
    return widget._save_figure_png(widget._hist_widgets[column]["figure"], name)


# --- every graph offers it -----------------------------------------------------


def test_every_graph_has_a_button(tmp_path):
    widget = _loaded(tmp_path)
    # one per filter histogram, one per metric histogram, MSD, detection counts
    assert len(_png_buttons(widget)) == len(widget._plot_canvases)


def test_every_graph_can_be_drawn_into_a_figure_of_its_own(tmp_path):
    """What the export dialog needs from each plot family: draw yourself into
    this figure, with these options, without touching the panel."""
    from matplotlib.figure import Figure

    widget = _loaded(tmp_path)
    widget.tracks = pd.DataFrame({"particle": [0] * 4, "frame": [0, 1, 2, 3],
                                  "x": [0.0, 1, 2, 3], "y": [0.0, 0, 0, 0]})
    widget._track_distance_cache = {0: 1.0}
    widget._loc2d_counts = np.array([3, 5, 4, 6, 2])
    opts = {"errorbars": True, "grid": True, "bounds": True, "title": True,
            "colorbar": True, "legend": True}
    for redraw in (
        lambda f, o: widget._draw_histogram("sigma [nm]", f, o),
        lambda f, o: widget._draw_metric_histogram("distance", f, o),
        lambda f, o: widget._draw_msd_validation(f, o),
        lambda f, o: widget._draw_loc2d_counts(f, o),
    ):
        figure = Figure()
        redraw(figure, opts)
        assert figure.axes, "nothing was drawn"


def test_they_collect_in_a_dated_folder_beside_the_data(tmp_path):
    widget = _loaded(tmp_path)
    path = _saved(widget)
    assert path.parent.parent == tmp_path / widget_mod.ANALYSIS_ROOT
    assert path.parent.name.endswith("_figures")


def test_the_name_says_which_graph_it_is(tmp_path):
    widget = _loaded(tmp_path)
    widget._metric_hist_widgets["pstatic"]  # exists
    assert _saved(widget, name="sigma [nm]_histogram").name == "sigma_nm_histogram.png"


def test_saving_the_same_graph_twice_keeps_both(tmp_path):
    """Re-saving after adjusting a bound should not silently replace the one
    already dropped into a talk."""
    widget = _loaded(tmp_path)
    first = _saved(widget, name="hist")
    second = _saved(widget, name="hist")
    assert first != second
    assert first.exists() and second.exists()


# --- what the file is ----------------------------------------------------------


def test_the_background_is_transparent(tmp_path):
    widget = _loaded(tmp_path)
    pixels = np.array(Image.open(_saved(widget)))
    assert pixels.shape[2] == 4, "no alpha channel"
    assert pixels[0, 0, 3] == 0, "the corner is not transparent"
    # A margin's worth, whatever the plot's proportions - the point is that the
    # ground shows through, not that any particular fraction of it does.
    assert (pixels[..., 3] == 0).mean() > 0.05, "hardly any of it is transparent"


def test_the_graph_itself_is_actually_drawn(tmp_path):
    """A fully transparent rectangle would also pass the test above."""
    widget = _loaded(tmp_path)
    pixels = np.array(Image.open(_saved(widget)))
    assert (pixels[..., 3] > 200).mean() > 0.05


def test_what_is_drawn_reads_against_a_dark_slide(tmp_path):
    """The plots are light-on-dark, which is the whole reason to drop the
    background rather than keep the black one."""
    widget = _loaded(tmp_path)
    pixels = np.array(Image.open(_saved(widget)))
    drawn = pixels[pixels[..., 3] > 200][:, :3]
    assert drawn.mean() > 90          # light overall
    assert drawn.max() > 200          # and genuinely bright somewhere


def test_it_is_written_above_screen_resolution(tmp_path):
    """A histogram that reads in a side panel is a blurred rectangle once a
    projector has stretched it across a wall."""
    widget = _loaded(tmp_path)
    canvas = widget._hist_widgets["sigma [nm]"]["canvas"]
    pixels = np.array(Image.open(_saved(widget)))
    assert pixels.shape[1] > canvas.width()
    assert widget_mod.FIGURE_SAVE_DPI >= 200


def test_the_plot_size_control_carries_through_to_the_file(tmp_path):
    widget = _loaded(tmp_path)
    narrow = np.array(Image.open(_saved(widget, name="narrow"))).shape[1]
    widget._set_plot_size(1200, 400)
    wide = np.array(Image.open(_saved(widget, name="wide"))).shape[1]
    assert wide > narrow


# --- failure ------------------------------------------------------------------


def test_an_unwritable_destination_is_reported_not_raised(tmp_path, monkeypatch):
    widget = _loaded(tmp_path)
    monkeypatch.setattr(widget, "_figure_save_dir",
                        lambda: Path(tmp_path / "nope" / "\0bad"))
    widget.log_box.clear()
    assert _saved(widget) is None
    assert "Could not save that graph" in widget.log_box.toPlainText()


# --- one folder per analysis, not one per session -----------------------------


def test_a_second_analysis_does_not_write_into_the_first_ones_folder(tmp_path):
    """The folder is remembered so that a set of graphs from one analysis lands
    together. That is only right while the analysis is the same one - it used to
    be chosen once per session, so every later run filed its graphs under the
    first run's timestamp, describing different data under neighbouring names."""
    widget = _loaded(tmp_path)
    first = _saved(widget, name="sigma").parent

    # a second fit, or a second table loaded - either is a new analysis
    widget._ingest_localization_dataframe(
        pd.DataFrame({
            "frame": [0, 1, 2, 3], "x [nm]": [0.0, 1, 2, 3], "y [nm]": [0.0, 1, 2, 3],
            "sigma [nm]": [120.0] * 4, "intensity [photon]": [900.0] * 4,
        }), "refitted", True)
    second = _saved(widget, name="sigma").parent

    assert second != first
    assert sorted(p.name for p in first.iterdir()) == ["sigma.png"]
    assert sorted(p.name for p in second.iterdir()) == ["sigma.png"]


def test_graphs_from_one_analysis_still_land_together(tmp_path):
    widget = _loaded(tmp_path)
    a = _saved(widget, name="one")
    b = _saved(widget, "intensity [photon]", name="two")
    assert a.parent == b.parent


def test_loading_a_different_stack_also_starts_a_new_folder(tmp_path):
    widget = _loaded(tmp_path)
    first = _saved(widget, name="sigma").parent

    widget._on_load_finished((None, None, "decoded", None, None), "", "")
    widget._ingest_localization_dataframe(widget.df, "reloaded", True)
    assert _saved(widget, name="sigma").parent != first


# --- the export dialog ---------------------------------------------------------
#
# The plots on screen are sized for reading in a side dock; a figure for a slide
# wants a different shape, a larger font and often different furniture. Choosing
# those must not disturb the thing being read, so the dialog draws into its own
# figure and never into the panel's.


def _dialog(widget, column="sigma [nm]", name="hist"):
    return widget_mod.FigureExportDialog(
        widget, lambda fig, opts: widget._draw_histogram(column, fig, opts),
        name, widget._figure_save_dir())


def test_the_panel_keeps_its_reading_size():
    widget = make_widget()
    assert widget.plot_height_box.value() == widget_mod.DEFAULT_PLOT_HEIGHT
    assert widget_mod.DEFAULT_PLOT_HEIGHT == 260


def test_the_dialog_opens_on_a_drawn_preview(tmp_path):
    """Not an empty frame waiting for a control to be touched."""
    dialog = _dialog(_loaded(tmp_path))
    assert dialog.preview_figure.axes


def test_choosing_an_export_size_does_not_touch_the_plot_on_screen(tmp_path):
    widget = _loaded(tmp_path)
    canvas = widget._hist_widgets["sigma [nm]"]["canvas"]
    axes = widget._hist_widgets["sigma [nm]"]["figure"].axes[0]
    before = (canvas.width(), canvas.height(), axes.title.get_fontsize())

    dialog = _dialog(widget)
    dialog.font_box.setValue(24)
    dialog.height_box.setValue(900)
    dialog.errorbars_box.setChecked(True)

    axes = widget._hist_widgets["sigma [nm]"]["figure"].axes[0]
    assert (canvas.width(), canvas.height(), axes.title.get_fontsize()) == before


def test_the_preview_follows_the_font(tmp_path):
    dialog = _dialog(_loaded(tmp_path))
    dialog.font_box.setValue(20)
    assert dialog.preview_figure.axes[0].title.get_fontsize() == 21   # title is +1


def test_a_locked_shape_drives_the_width(tmp_path):
    dialog = _dialog(_loaded(tmp_path))
    dialog.height_box.setValue(540)
    dialog.shape_box.setCurrentText("16:9")
    assert dialog.width_box.value() == round(540 * 16 / 9)


@pytest.mark.parametrize("fmt", ["pdf", "svg", "png", "tiff"])
def test_it_writes_every_format(tmp_path, fmt):
    widget = _loaded(tmp_path)
    dialog = _dialog(widget, name=f"as_{fmt}")
    index = [dialog.format_box.itemData(i) for i in range(dialog.format_box.count())].index(fmt)
    dialog.format_box.setCurrentIndex(index)
    dialog._save()

    path = dialog.saved_path()
    assert path is not None and path.suffix == f".{fmt}"
    assert path.stat().st_size > 0


def test_the_vector_formats_ignore_the_resolution(tmp_path):
    dialog = _dialog(_loaded(tmp_path))
    dialog.format_box.setCurrentIndex(0)          # PDF
    assert not dialog.dpi_box.isEnabled()
    assert "vector" in dialog.size_label.text()

    png = [dialog.format_box.itemData(i) for i in range(dialog.format_box.count())].index("png")
    dialog.format_box.setCurrentIndex(png)
    assert dialog.dpi_box.isEnabled()
    assert "dpi" in dialog.size_label.text()


def test_error_bars_are_optional_and_actually_appear(tmp_path):
    from matplotlib.figure import Figure

    widget = _loaded(tmp_path)
    counts = {}
    for wanted in (False, True):
        figure = Figure()
        widget._draw_histogram("sigma [nm]", figure, {"errorbars": wanted})
        counts[wanted] = len(figure.axes[0].containers)
    assert counts[True] > counts[False]


def test_the_furniture_can_be_turned_off(tmp_path):
    from matplotlib.figure import Figure

    widget = _loaded(tmp_path)
    figure = Figure()
    widget._draw_histogram("sigma [nm]", figure, {"title": False, "bounds": False})
    axes = figure.axes[0]
    assert axes.get_title() == ""
    assert not axes.lines           # no bound lines drawn


def test_saving_twice_from_the_dialog_keeps_both(tmp_path):
    widget = _loaded(tmp_path)
    first = _dialog(widget, name="twice")
    first._save()
    second = _dialog(widget, name="twice")
    second._save()
    assert first.saved_path() != second.saved_path()
    assert first.saved_path().exists() and second.saved_path().exists()
