"""The view saved as displayed, time averages of a corrected movie, and line profiles."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import math

import numpy as np
import pytest

napari = pytest.importorskip("napari", reason="needs napari")
from napari.components import ViewerModel  # noqa: E402

from napari_loc_track import _averages as averages_io  # noqa: E402
from napari_loc_track import _drift as drift_io  # noqa: E402
from napari_loc_track import _profile as profile_io  # noqa: E402
from napari_loc_track import _view_export as view_export  # noqa: E402


def _two_layers():
    viewer = ViewerModel()
    coarse = np.zeros((20, 20), np.float32)
    coarse[5, 5] = 100.0                         # one bright pixel, 100 nm px
    fine = np.zeros((40, 40), np.float32)
    fine[30, 30] = 10.0                          # 50 nm px
    a = viewer.add_image(coarse, scale=(100, 100), colormap="green", blending="additive",
                         contrast_limits=(0, 100), name="coarse")
    b = viewer.add_image(fine, scale=(50, 50), translate=(-25, -25), colormap="magenta",
                         blending="additive",
                         contrast_limits=(0, 10), name="fine")
    return viewer, a, b


def test_the_layers_are_blended_at_the_finest_pixel_with_their_own_contrast_and_colours():
    viewer, a, b = _two_layers()
    box = view_export.layers_box([a, b])
    comp = view_export.compose([a, b], viewer.dims.point, box)
    assert comp.pixel_nm == pytest.approx(50.0)
    rgb = comp.rgb
    assert rgb.shape[:2] == (40, 40)                   # 2 um at 50 nm
    # the coarse bright pixel covers 2 x 2 output pixels, pure green at full contrast
    green = np.all(np.abs(rgb - [0, 1, 0]) < 0.02, axis=-1)
    assert green.sum() == 4 and np.argwhere(green).mean(axis=0) == pytest.approx((10.5, 10.5))
    # the fine one is one output pixel, magenta
    magenta = np.all(np.abs(rgb - [1, 0, 1]) < 0.02, axis=-1)
    assert np.argwhere(magenta).tolist() == [[30, 30]]
    assert len(comp.channels) == 2
    assert comp.channels[1].display_range == (0.0, 10.0)


def test_a_hidden_layer_is_left_out():
    viewer, a, b = _two_layers()
    b.visible = False
    comp = view_export.compose([a, b], viewer.dims.point, view_export.layers_box([a]))
    assert [c.name for c in comp.channels] == ["coarse"]


def test_the_files_carry_the_scale_bar_and_the_channels(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    from PIL import Image

    viewer, a, b = _two_layers()
    comp = view_export.compose([a, b], viewer.dims.point, view_export.layers_box([a, b]))
    rgb8 = comp.rgb8()
    length = view_export.burn_scale_bar(rgb8, comp.pixel_nm)
    assert length is not None
    corner = rgb8[-rgb8.shape[0] // 4:, -rgb8.shape[1] // 2:]
    assert (corner.min(axis=-1) > 200).sum() >= int(length / comp.pixel_nm)   # a white bar
    png = view_export.save_png(tmp_path / "view.png", rgb8, comp.pixel_nm)
    assert np.asarray(Image.open(png)).shape == rgb8.shape
    path = view_export.save_channels_tiff(tmp_path / "view.tif", comp, "test")
    with tifffile.TiffFile(path) as tif:
        data = tif.asarray()
        meta = tif.imagej_metadata
    assert data.shape == (2,) + comp.rgb.shape[:2]
    assert meta["mode"] == "composite" and len(meta["LUTs"]) == 2
    assert list(meta["Ranges"]) == [0.0, 100.0, 0.0, 10.0]


def test_the_time_point_on_the_slider_is_the_one_saved():
    viewer = ViewerModel()
    movie = np.zeros((3, 10, 10), np.float32)
    for t in range(3):
        movie[t, t, t] = 1.0
    layer = viewer.add_image(movie, contrast_limits=(0, 1), colormap="gray")
    viewer.dims.set_current_step(0, 2)
    comp = view_export.compose([layer], viewer.dims.point, view_export.layers_box([layer]))
    assert comp.rgb[2, 2].min() > 0.9 and comp.rgb[0, 0].max() < 0.1


def _drifting_spot(n=40, drift_px=6.0):
    yy, xx = np.mgrid[0:32, 0:32]
    frames, shifts = [], []
    for i in range(n):
        dx = drift_px * i / (n - 1)
        frames.append(np.exp(-((xx - 12 - dx) ** 2 + (yy - 16) ** 2) / (2 * 1.2 ** 2)).astype(np.float32))
        shifts.append((0.0, -dx))                                   # moves it back
    return np.stack(frames), np.array(shifts)


def test_the_average_of_a_corrected_movie_is_sharp():
    frames, shifts = _drifting_spot()
    corrected = drift_io.ShiftedStack(frames, shifts)
    image, used = averages_io.average(corrected, block=4)
    assert used == len(frames)
    oy, ox = averages_io.canvas_origin(corrected)
    peak = np.unravel_index(np.nanargmax(image), image.shape)
    assert (peak[0] + oy, peak[1] + ox) == (16, 12)
    raw, _used = averages_io.average(frames)
    assert np.nanmax(image) > 1.6 * np.nanmax(raw)          # the raw average is smeared


def test_a_profile_across_a_gaussian_line_reads_its_width():
    viewer = ViewerModel()
    xx = np.arange(200)[None, :] * np.ones((50, 1))
    sigma_px = 3.0
    image = (np.exp(-((xx - 100) ** 2) / (2 * sigma_px ** 2)) * 50 + 5).astype(np.float32)
    layer = viewer.add_image(image, scale=(20, 20))           # 20 nm px
    distance, values = profile_io.sample_line(layer, viewer.dims.point, (500, 1000), (500, 3000),
                                              width=200)
    fit = profile_io.fit_gaussian(distance, values)
    assert fit["fwhm"] == pytest.approx(sigma_px * 20 * profile_io.FWHM_PER_SIGMA, rel=0.05)
    assert fit["centre"] + 1000 == pytest.approx(2000, abs=10)
    assert math.isclose(distance[-1], 2000.0, rel_tol=1e-9)
