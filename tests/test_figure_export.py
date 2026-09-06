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
    return [b for b in widget.findChildren(widget_mod.QPushButton) if b.text() == "PNG"]


def _saved(widget, column="sigma [nm]", name="check"):
    return widget._save_figure_png(widget._hist_widgets[column]["figure"], name)


# --- every graph offers it -----------------------------------------------------


def test_every_graph_has_a_button(tmp_path):
    widget = _loaded(tmp_path)
    # one per filter histogram, one per metric histogram, MSD, detection counts
    assert len(_png_buttons(widget)) == len(widget._plot_canvases)


def test_clicking_them_all_writes_them_all(tmp_path):
    widget = _loaded(tmp_path)
    buttons = _png_buttons(widget)
    for button in buttons:
        button.click()

    written = sorted(p.name for p in widget._figure_save_folder.iterdir())
    assert len(written) == len(buttons)
    for expected in ("msd_validation.png", "detection_counts.png",
                     "D_histogram.png", "dmin_histogram.png",
                     "sigma_nm_histogram.png"):
        assert expected in written


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
    assert (pixels[..., 3] == 0).mean() > 0.2, "hardly any of it is transparent"


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
