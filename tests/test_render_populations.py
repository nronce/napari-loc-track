"""Rendering one population at a time, into a layer of its own.

The immobile molecules of a live-cell acquisition are its structure and the
mobile ones are its dynamics. Reconstructing each separately, from one dataset,
into two layers that blend additively is the whole point of having a calibrated
immobility test - and it only works if a render stops replacing the previous
one the moment its selection changes.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_render_widget import _pump_until  # noqa: E402
from test_widget_interaction import ensure_qapp  # noqa: E402

PIXEL_NM, DT, N_POINTS = 100.0, 0.0313, 15
_WIDGETS = []


def _widget():
    from napari.components import ViewerModel

    ensure_qapp()
    widget = widget_mod.LocalizationTrackingWidget(ViewerModel())
    _WIDGETS.append(widget)
    widget.pixel_size_box.setValue(PIXEL_NM)
    widget.fps_box.setValue(1.0 / DT)
    widget.render_png_box.setChecked(False)
    return widget


def _analysed(n_static=30, n_mobile=30, seed=0):
    """Two populations, well separated, with a real uncertainty column."""
    rng = np.random.default_rng(seed)
    rows, tracks = [], []
    for pid in range(n_static + n_mobile):
        step = 0.0 if pid < n_static else np.sqrt(2 * 0.02e6 * DT)
        base = np.array([500.0 + 400.0 * (pid % 10), 500.0 + 400.0 * (pid // 10)])
        true = np.cumsum(rng.normal(0, step, (N_POINTS, 2)), axis=0) + base
        sigma = rng.uniform(15.0, 35.0, N_POINTS)
        seen = true + rng.normal(0, 1, (N_POINTS, 2)) * sigma[:, None]
        for frame, (point, s) in enumerate(zip(seen, sigma)):
            rows.append({"frame": frame, "x [nm]": point[0], "y [nm]": point[1],
                         "uncertainty [nm]": s, "sigma [nm]": 150.0,
                         "intensity [photon]": 900.0})
            tracks.append({"particle": pid, "frame": frame,
                           "x": point[0] / PIXEL_NM, "y": point[1] / PIXEL_NM})

    widget = _widget()
    widget._ingest_localization_dataframe(pd.DataFrame(rows), "loaded", True)
    widget.tracks = pd.DataFrame(tracks)
    widget._invalidate_track_filter()
    widget._start_fit_free_metrics_worker()
    assert _pump_until(lambda: widget._track_motion_cache), "the metrics never finished"
    return widget


def _render(widget):
    widget.render_smlm_image()
    assert _pump_until(lambda: widget._render_worker_ref is None), "render never finished"


def _render_layers(widget):
    return [layer for layer in widget.viewer.layers if widget_mod.is_render_layer(layer)]


# --- the tab order ------------------------------------------------------------


def test_render_sits_after_track():
    """The selection a render is built from needs the trajectories first."""
    widget = _widget()
    titles = [widget.tabs.tabText(i) for i in range(widget.tabs.count())]
    assert titles.index("Render") == titles.index("Track") + 1


# --- one layer per population -------------------------------------------------


def test_the_two_populations_render_into_two_layers():
    widget = _analysed()
    widget._set_render_population("immobile")
    _render(widget)
    widget._set_render_population("mobile")
    _render(widget)

    names = sorted(layer.name for layer in _render_layers(widget))
    assert names == ["smlm_render_immobile", "smlm_render_mobile"]


def test_each_layer_holds_only_its_own_population():
    widget = _analysed()
    widget._set_render_population("immobile")
    immobile_locs = len(widget._displayed_localizations())
    _render(widget)
    widget._set_render_population("mobile")
    mobile_locs = len(widget._displayed_localizations())
    _render(widget)

    total = len(widget.df_filtered)
    assert immobile_locs + mobile_locs == pytest.approx(total, rel=0.05)
    for name, expected in (("smlm_render_immobile", immobile_locs),
                           ("smlm_render_mobile", mobile_locs)):
        rendered = float(np.asarray(widget.viewer.layers[name].data).sum())
        assert rendered == pytest.approx(expected, rel=0.05)


def test_the_layers_blend_so_they_can_be_read_together():
    widget = _analysed()
    widget._set_render_population("immobile")
    _render(widget)
    assert widget.viewer.layers["smlm_render_immobile"].blending == "additive"


def test_each_layer_records_what_it_was_built_from():
    widget = _analysed()
    widget._set_render_population("mobile")
    _render(widget)
    metadata = widget.viewer.layers["smlm_render_mobile"].metadata
    assert metadata[widget_mod.RENDER_LAYER_TAG] is True
    assert "p" in metadata["dynamics_selection"]


def test_rendering_the_same_name_twice_replaces_rather_than_accumulates():
    widget = _analysed()
    widget._set_render_population("immobile")
    _render(widget)
    _render(widget)
    assert len(_render_layers(widget)) == 1


# --- the presets --------------------------------------------------------------


def test_the_presets_split_the_population_at_the_chosen_significance():
    widget = _analysed()
    widget._set_render_population("immobile")
    immobile = set(widget._displayed_tracks()["particle"])
    widget._set_render_population("mobile")
    mobile = set(widget._displayed_tracks()["particle"])

    assert not (immobile & mobile)                  # a molecule is in one or the other
    assert len(immobile & set(range(30))) >= 27     # nearly all the static ones
    assert len(mobile & set(range(30, 60))) >= 27


def test_the_split_threshold_is_the_p_value_box():
    widget = _analysed()
    widget.render_population_p_box.setValue(0.5)
    widget._set_render_population("immobile")
    assert widget.pstatic_min_box.value() == pytest.approx(0.5)
    assert widget.pstatic_max_box.value() == pytest.approx(1.0)


def test_all_clears_the_selection_and_the_layer_name_with_it():
    widget = _analysed()
    widget._set_render_population("mobile")
    assert widget._passing_particles() is not None

    widget._set_render_population("all")
    assert widget._passing_particles() is None
    assert widget.render_layer_name_edit.text() == widget_mod.RENDER_LAYER_NAME
    assert len(widget._displayed_localizations()) == len(widget.df_filtered)


def test_a_preset_replaces_any_other_behavioural_filter():
    """Two selections at once would silently intersect, and the layer name
    would then describe only half of what produced it."""
    widget = _analysed()
    widget.dist_min_box.setValue(0.0)
    widget.dist_max_box.setValue(1e6)
    widget.distance_filter_box.setChecked(True)

    widget._set_render_population("immobile")
    active = [key for key, _low, _high in widget._active_metric_filters()]
    assert active == ["pstatic"]


def test_a_preset_keeps_a_detection_floor_qualifier():
    """dmin is not a competing claim about how the molecule behaved - it says
    what the trajectory was capable of measuring - and it is the companion the
    immobile preset needs: p > alpha alone sweeps every trajectory too short to
    detect anything into the immobile pile."""
    widget = _analysed()
    widget.dmin_min_box.setValue(0.0)
    widget.dmin_max_box.setValue(0.02)
    widget.dmin_filter_box.setChecked(True)

    widget._set_render_population("immobile")
    active = sorted(key for key, _low, _high in widget._active_metric_filters())
    assert active == ["dmin", "pstatic"]

    # "All" means all, though - including dropping the qualifier, or the button
    # would not do what its label says.
    widget._set_render_population("all")
    assert not widget.dmin_filter_box.isChecked()
    assert widget._passing_particles() is None


def test_the_panel_says_what_the_next_render_will_contain():
    widget = _analysed()
    widget._set_render_population("immobile")
    text = widget.render_population_label.text()
    assert "trajectories" in text
    assert "smlm_render_immobile" in text


# --- a named render is still a render -----------------------------------------


def test_a_renamed_render_is_not_mistaken_for_the_raw_stack():
    """The Localize tab picks the image layer to detect in, and it has to skip
    reconstructions - which it used to do by name."""
    import napari

    widget = _analysed()
    widget.viewer.add_image(np.zeros((5, 32, 32), np.uint16), name="raw.tif")
    widget._set_render_population("immobile")
    _render(widget)

    chosen = widget._get_localize_image_layer()
    assert chosen is not None
    assert chosen.name == "raw.tif"


def test_a_renamed_render_keeps_its_derived_scale():
    """Its scale comes from the layer beneath it, so it must not be rescaled
    like a raw stack - a single layer left in pixels makes napari discard the
    units across the whole viewer."""
    widget = _analysed()
    widget._set_render_population("immobile")
    _render(widget)
    layer = widget.viewer.layers["smlm_render_immobile"]
    before = tuple(np.ravel(layer.scale))

    widget._apply_viewer_scale()
    assert tuple(np.ravel(layer.scale)) == pytest.approx(before)
    assert tuple(str(u) for u in layer.units) == ("nanometer", "nanometer")


def test_the_layer_name_and_threshold_survive_a_round_trip():
    widget = _analysed()
    widget._set_render_population("mobile")
    widget.render_population_p_box.setValue(0.01)
    metadata = widget._collect_metadata(None)
    assert metadata["smlm_rendering"]["layer_name"] == "smlm_render_mobile"
    assert metadata["smlm_rendering"]["population_split_p"] == pytest.approx(0.01)

    restored = _widget()
    restored.apply_settings(metadata)
    assert restored.render_layer_name_edit.text() == "smlm_render_mobile"
    assert restored.render_population_p_box.value() == pytest.approx(0.01)


# --- saving part of a movie ----------------------------------------------------
#
# A reconstruction movie is usually far longer than anything you would show, and
# with a sliding window most of its frames are the same picture again: a
# 100-frame window advancing 10 leaves consecutive frames sharing 90% of their
# raw frames. Both the clip and the stride are chosen at save time rather than
# by re-rendering.


def _with_movie(n_frames=40, window=100, step=10, interval=0.313):
    widget = _widget()
    widget._render_movie = np.zeros((n_frames, 64, 64), np.float32)
    widget._render_movie_info = {
        "frames_per_group": window, "window_step_frames": step,
        "mode_label": "Gaussian", "frame_interval_s": interval,
        "super_resolved_pixel_size_nm": 20.0,
    }
    widget._sync_movie_save_range()
    return widget


def test_the_range_starts_as_the_whole_movie():
    widget = _with_movie(40)
    assert (widget.movie_first_box.value(), widget.movie_last_box.value()) == (0, 39)
    assert widget.movie_last_box.maximum() == 39
    assert "Saving 40 of 40 frames" in widget.movie_save_label.text()


def test_a_new_render_resets_the_range_rather_than_keeping_stale_bounds():
    widget = _with_movie(40)
    widget.movie_first_box.setValue(10)
    widget.movie_last_box.setValue(30)

    widget._render_movie = np.zeros((7, 64, 64), np.float32)
    widget._sync_movie_save_range()
    assert (widget.movie_first_box.value(), widget.movie_last_box.value()) == (0, 6)


def test_only_the_chosen_frames_are_written():
    widget = _with_movie(40)
    widget.movie_first_box.setValue(5)
    widget.movie_last_box.setValue(24)
    widget.movie_stride_box.setValue(10)

    image, info = widget._apply_movie_save_range(
        widget._render_movie, widget._render_movie_info)
    assert image.shape[0] == 2                       # frames 5 and 15
    assert info["saved_frame_range"] == [5, 24, 10]
    assert info["n_frames_saved"] == 2


def test_the_stride_stretches_the_frame_interval_written_to_the_file():
    """Every tenth frame still spans ten frames of time - a viewer told
    otherwise plays the clip ten times too fast."""
    widget = _with_movie(40, interval=0.313)
    widget.movie_stride_box.setValue(10)
    _image, info = widget._apply_movie_save_range(
        widget._render_movie, widget._render_movie_info)
    assert info["frame_interval_s"] == pytest.approx(3.13)


def test_saving_all_of_it_changes_nothing():
    widget = _with_movie(40)
    image, info = widget._apply_movie_save_range(
        widget._render_movie, widget._render_movie_info)
    assert image is widget._render_movie
    assert "saved_frame_range" not in info


def test_the_panel_says_how_redundant_the_frames_are():
    """The number that tells you which stride to use."""
    widget = _with_movie(40, window=100, step=10)
    text = widget.movie_save_label.text()
    assert "share 90% of their raw frames" in text
    assert "every 10th frame is independent" in text


def test_independent_blocks_get_no_redundancy_warning():
    widget = _with_movie(40, window=50, step=50)
    assert "share" not in widget.movie_save_label.text()


def test_the_label_follows_the_boxes():
    widget = _with_movie(40)
    widget.movie_stride_box.setValue(4)
    assert "Saving 10 of 40 frames" in widget.movie_save_label.text()


def test_the_stride_survives_a_round_trip():
    widget = _with_movie(40)
    widget.movie_stride_box.setValue(7)
    metadata = widget._collect_metadata(None)
    assert metadata["smlm_rendering"]["movie_save_stride"] == 7

    restored = _widget()
    restored.apply_settings(metadata)
    assert restored.movie_stride_box.value() == 7


# --- turning the image on its way out ------------------------------------------
#
# napari's own layer.rotate turns a layer in the canvas but never the pixels, so
# it cannot reach a saved file. This does the turning on the array itself, after
# the crop and before the annotations - a rotated scale bar would be unreadable,
# and rotating first would leave the crop box pointing at the wrong region.


def _marked(shape=(6, 8)):
    """An L with a distinct corner mark, so orientation is unambiguous."""
    img = np.zeros(shape, np.float32)
    img[1, 1:5] = 1.0
    img[1:4, 1] = 1.0
    img[0, shape[1] - 1] = 9.0          # top-right
    return img


def test_no_rotation_hands_back_the_same_array():
    img = _marked()
    assert widget_mod.rotate_save_array(img, 0, False) is img
    assert widget_mod.rotate_save_array(img, 360, False) is img


@pytest.mark.parametrize("degrees", [90, 180, 270])
def test_a_quarter_turn_is_exact(degrees):
    """Pixels permuted, never resampled - a float32 reconstruction still holds
    the localization counts it held before."""
    img = _marked()
    turned = widget_mod.rotate_save_array(img, degrees, False)
    assert turned.sum() == img.sum()
    assert sorted(turned.ravel().tolist()) == sorted(img.ravel().tolist())


def test_a_quarter_turn_swaps_the_axes():
    img = _marked((6, 8))
    assert widget_mod.rotate_save_array(img, 90, False).shape == (8, 6)
    assert widget_mod.rotate_save_array(img, 180, False).shape == (6, 8)


def test_turning_twice_by_ninety_is_turning_once_by_one_eighty():
    img = _marked()
    once = widget_mod.rotate_save_array(img, 180, False)
    twice = widget_mod.rotate_save_array(
        widget_mod.rotate_save_array(img, 90, False), 90, False)
    assert np.array_equal(once, twice)


def test_the_direction_is_counter_clockwise():
    """Matching napari's own layer.rotate, so the two agree."""
    img = _marked((6, 8))
    turned = widget_mod.rotate_save_array(img, 90, False)
    assert tuple(np.argwhere(turned == 9.0)[0]) == (0, 0)   # top-right -> top-left


def test_a_movie_turns_every_frame_and_keeps_its_time_axis():
    movie = np.stack([_marked()] * 3)
    turned = widget_mod.rotate_save_array(movie, 90, True)
    assert turned.shape == (3, 8, 6)


def test_a_composite_keeps_its_colour_axis():
    rgb = np.zeros((6, 8, 3), np.uint8)
    rgb[1, 1:5] = 255
    assert widget_mod.rotate_save_array(rgb, 90, False).shape == (8, 6, 3)


def test_an_arbitrary_angle_resamples_and_says_so():
    """Interpolation samples the rotated grid rather than redistributing what
    was there, so the total is not conserved. The label has to admit it."""
    img = _marked()
    turned = widget_mod.rotate_save_array(img, 37.0, False)
    assert turned.shape != img.shape          # reshaped to fit
    assert turned.sum() != img.sum()          # and not count-preserving

    widget = _widget()
    widget.render_rotate_box.setValue(37.0)
    assert "not conserved" in widget.render_rotate_label.text()
    widget.render_rotate_box.setValue(90.0)
    assert "exact" in widget.render_rotate_label.text()
    widget.render_rotate_box.setValue(0.0)
    assert "not rotated" in widget.render_rotate_label.text()


def test_the_angle_is_recorded_with_the_saved_image():
    widget = _widget()
    widget._render_image = np.zeros((8, 8), np.float32)
    widget._render_image_info = {"field_of_view_camera_px": [8, 8],
                                 "origin_camera_px": [0.0, 0.0], "oversampling": 1,
                                 "mode_label": "Gaussian"}
    widget.render_rotate_box.setValue(90.0)
    _spec, extra = widget._save_spec("image", widget._render_image_info)
    assert extra["rotated_degrees"] == pytest.approx(90.0)
    assert extra["rotation_is_exact"] is True

    widget.render_rotate_box.setValue(37.0)
    _spec, extra = widget._save_spec("image", widget._render_image_info)
    assert extra["rotation_is_exact"] is False


def test_the_angle_survives_a_round_trip():
    widget = _widget()
    widget.render_rotate_box.setValue(270.0)
    restored = _widget()
    restored.apply_settings(widget._collect_metadata(None))
    assert restored.render_rotate_box.value() == pytest.approx(270.0)
