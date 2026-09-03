"""Summing raw frames in groups of N before anything else looks at them.

Time binning trades time resolution for signal: a dim emitter that sits under
the detection threshold in every single frame can be comfortably over it in the
sum of four. That only works if the sum is a sum - the fit is a maximum
likelihood estimate under photon statistics, and the sum of N Poisson frames is
still Poisson, while their average is not.

Two numbers have to follow the factor or every physical result downstream is
wrong by exactly N: the camera baseline, because each raw frame brought its own
and they add, and the frame rate, because a binned frame spans N exposures. The
second is the dangerous one - a diffusion coefficient computed at N times the
real frame rate is off by a factor of N and looks entirely plausible.
"""
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib.util

import numpy as np
import pytest

_PKG_DIR = Path(__file__).resolve().parents[1] / "napari_loc_track"
if "napari_loc_track" not in sys.modules:
    _pkg = types.ModuleType("napari_loc_track")
    _pkg.__path__ = [str(_PKG_DIR)]
    sys.modules["napari_loc_track"] = _pkg
_spec = importlib.util.spec_from_file_location(
    "napari_loc_track._imageio", _PKG_DIR / "_imageio.py"
)
imageio = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("napari_loc_track._imageio", imageio)
_spec.loader.exec_module(imageio)


# --- the arithmetic -----------------------------------------------------------


def _raw(n=12, size=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4000, size=(n, size, size), dtype=np.uint16)


def test_frames_are_summed_not_averaged():
    """The fit assumes photon statistics; only the sum still obeys them."""
    raw = _raw(n=8)
    binned, _how = imageio.bin_frames(raw, 4)
    assert binned.shape == (2, 8, 8)
    np.testing.assert_array_equal(binned[0], raw[:4].sum(axis=0, dtype=np.uint32))
    np.testing.assert_array_equal(binned[1], raw[4:].sum(axis=0, dtype=np.uint32))


def test_the_sum_is_promoted_so_it_cannot_wrap():
    """Four saturated uint16 frames overflow uint16, and a wrapped pixel reads
    as a legitimately dark one rather than as an error."""
    raw = np.full((4, 2, 2), np.iinfo(np.uint16).max, np.uint16)
    binned, _how = imageio.bin_frames(raw, 4)
    assert binned.dtype == np.uint32
    assert binned[0, 0, 0] == 4 * np.iinfo(np.uint16).max


@pytest.mark.parametrize("dtype,factor,expected", [
    (np.uint8, 3, np.uint16),
    (np.uint16, 4, np.uint32),
    (np.uint16, 100_000, np.uint64),
    (np.int16, 4, np.int32),
    (np.float32, 8, np.float32),   # floats already have the range
])
def test_the_promoted_dtype_is_the_smallest_that_fits(dtype, factor, expected):
    assert imageio.sum_dtype(dtype, factor) == np.dtype(expected)


def test_a_partial_bin_at_the_end_is_dropped_rather_than_kept_short():
    """A short final bin would be the one frame with a different exposure and a
    different baseline - worse than losing three frames of a ten thousand."""
    raw = _raw(n=10)
    binned, how = imageio.bin_frames(raw, 4)
    assert binned.shape[0] == 2                 # not 3
    assert "2 trailing frame(s) dropped" in how


def test_a_factor_of_one_hands_back_the_very_same_stack():
    raw = _raw()
    binned, how = imageio.bin_frames(raw, 1)
    assert binned is raw
    assert how == "not time-binned"


def test_a_factor_larger_than_the_movie_leaves_it_alone():
    raw = _raw(n=10)
    binned, how = imageio.bin_frames(raw, 20)
    assert binned is raw
    assert "skipped" in how


def test_a_stack_too_large_to_hold_binned_is_binned_lazily():
    """The binned form of a 20 GB movie does not fit in RAM either. It has to
    stay a handle, and it has to produce the same numbers when asked."""
    raw = _raw(n=12)
    lazy, how = imageio.bin_frames(raw, 3, eager_limit_bytes=0)
    eager, _how = imageio.bin_frames(raw, 3)
    assert "lazily" in how
    assert type(lazy).__module__.startswith("dask")
    assert lazy.shape == eager.shape
    np.testing.assert_array_equal(np.asarray(lazy), eager)


def test_binning_can_be_cancelled_part_way():
    import threading

    cancel = threading.Event()
    cancel.set()
    binned, how = imageio.bin_frames(_raw(n=64), 2, cancel=cancel)
    assert binned is None
    assert how == "cancelled"


def test_a_spot_under_threshold_in_every_frame_is_over_it_in_the_sum():
    """The reason to bin at all, end to end through the real detector.

    A dim emitter contributes the same photons every frame; noise does not add
    coherently, so N frames of signal beat sqrt(N) of noise.
    """
    conftest = importlib.import_module("conftest")
    localize = conftest.load_localize2d()

    rng = np.random.default_rng(3)
    n, size, box = 8, 32, 7
    yy, xx = np.mgrid[0:size, 0:size]
    spot = 26.0 * np.exp(-((yy - 16.0) ** 2 + (xx - 16.0) ** 2) / (2 * 1.3 ** 2))
    raw = np.clip(rng.normal(100.0, 6.0, (n, size, size)) + spot, 0, None)
    raw = raw.astype(np.uint16)

    threshold = 900.0
    per_frame = [len(localize.identify_in_frame(
        np.asarray(f, np.float32), threshold, box)[0]) for f in raw]
    binned, _how = imageio.bin_frames(raw, n)
    in_sum = len(localize.identify_in_frame(
        np.asarray(binned[0], np.float32), threshold, box)[0])

    assert max(per_frame) == 0, "the spot was already detectable unbinned"
    assert in_sum == 1


# --- through the widget -------------------------------------------------------

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_render_widget import _pump_until  # noqa: E402
from test_widget_interaction import make_widget  # noqa: E402


def _binned_widget(factor, offset=100.0, fps=100.0):
    widget = make_widget()
    widget.loc_offset_box.setValue(offset)
    widget.fps_box.setValue(fps)
    widget.bin_factor_box.setValue(factor)
    widget._apply_time_binning()
    return widget


def test_binning_is_off_by_default():
    """It costs time resolution and merges blinks, so it is never silent."""
    widget = make_widget()
    assert widget.bin_factor_box.value() == 1
    assert widget.bin_label.text() == "off"


def test_the_camera_baseline_scales_up_with_the_factor():
    """Each of the N summed frames brought its own baseline, and they add.

    Left at the per-frame value, the baseline is under-subtracted N-fold and
    every fitted photon count inherits the difference.
    """
    widget = _binned_widget(4, offset=100.0)
    assert widget.loc_offset_box.value() == pytest.approx(400.0)


def test_the_frame_rate_scales_down_with_the_factor():
    """A binned frame spans N exposures. Left too fast, every D is N times
    too large and nothing about the number looks wrong."""
    widget = _binned_widget(4, fps=31.882)
    assert widget.fps_box.value() == pytest.approx(31.882 / 4, abs=1e-3)
    assert widget.frame_interval_box.value() == pytest.approx(
        4 * 1000.0 / 31.882, abs=1e-2)


def test_the_pixel_size_is_untouched_by_binning_in_time():
    widget = make_widget()
    widget.pixel_size_box.setValue(161.0)
    widget.bin_factor_box.setValue(8)
    widget._apply_time_binning()
    assert widget.pixel_size_box.value() == pytest.approx(161.0)


def test_changing_the_factor_rescales_from_the_old_one_not_from_scratch():
    """1 -> 4 -> 2 -> 1 has to land exactly back on the per-frame values."""
    widget = _binned_widget(4, offset=100.0, fps=100.0)
    for factor, offset, fps in ((2, 200.0, 50.0), (8, 800.0, 12.5), (1, 100.0, 100.0)):
        widget.bin_factor_box.setValue(factor)
        widget._apply_time_binning()
        assert widget.loc_offset_box.value() == pytest.approx(offset)
        assert widget.fps_box.value() == pytest.approx(fps)


def _acquisition(**values):
    return {"values": values,
            "sources": {key: "metadata.txt: field" for key in values}}


def test_a_recorded_baseline_arrives_scaled_for_the_binning_in_force():
    """The camera wrote down what one raw frame does; the fit sees sums of N."""
    widget = _binned_widget(4, offset=0.0)
    widget._apply_acquisition_metadata(_acquisition(camera_offset_adu=100.0))
    assert widget.loc_offset_box.value() == pytest.approx(400.0)


def test_a_recorded_frame_rate_arrives_divided_by_the_binning_in_force():
    widget = _binned_widget(4, fps=1.0)
    widget._apply_acquisition_metadata(_acquisition(fps=31.882))
    assert widget.fps_box.value() == pytest.approx(31.882 / 4, abs=1e-3)


def test_a_recorded_pixel_size_arrives_unscaled():
    widget = _binned_widget(4)
    widget._apply_acquisition_metadata(_acquisition(pixel_size_nm=108.3))
    assert widget.pixel_size_box.value() == pytest.approx(108.3, abs=0.05)


def test_unbinned_autofill_is_exactly_what_the_metadata_said():
    widget = make_widget()
    widget.loc_offset_box.setValue(0.0)
    widget._apply_acquisition_metadata(
        _acquisition(camera_offset_adu=100.0, fps=31.882))
    assert widget.loc_offset_box.value() == pytest.approx(100.0)
    assert widget.fps_box.value() == pytest.approx(31.882, abs=1e-3)


def test_the_load_worker_bins_the_stack_it_opens(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    raw = _raw(n=12, size=16)
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, raw)

    result = widget_mod._load_worker("", str(path), 4).work()
    _df, image, how, _acq, raw_back = result
    assert image.shape == (3, 16, 16)
    np.testing.assert_array_equal(
        np.asarray(image[0]), raw[:4].sum(axis=0, dtype=np.uint32))
    assert "time-binned" in how
    # The unbinned stack comes back too, so the factor can be changed later
    # without re-reading a file that may be tens of GB on a network share.
    assert raw_back.shape == raw.shape


def _loaded(widget, raw, name="stack.tif"):
    """Drive the real load handler the way a finished worker would."""
    factor = int(widget.bin_factor_box.value())
    image, _how = imageio.bin_frames(raw, factor)
    widget._on_load_finished((None, image, "decoded", None, raw), "", name)
    return widget.viewer.layers[name]


def test_changing_the_factor_rebins_the_loaded_stack_without_reloading():
    widget = make_widget()
    raw = _raw(n=12, size=8)
    layer = _loaded(widget, raw)
    assert layer.data.shape[0] == 12

    widget.bin_factor_box.setValue(3)
    widget._apply_time_binning()
    assert _pump_until(lambda: widget._bin_worker_ref is None), "re-binning never finished"

    assert widget.viewer.layers["stack.tif"].data.shape[0] == 4
    np.testing.assert_array_equal(
        np.asarray(widget.viewer.layers["stack.tif"].data[0]),
        raw[:3].sum(axis=0, dtype=np.uint32))


def test_rebinning_throws_away_candidates_found_against_the_old_frames():
    """Candidates are indexed by frame and the frames have just been renumbered."""
    widget = make_widget()
    raw = _raw(n=12, size=8)
    _loaded(widget, raw)
    widget._loc2d_candidates = [("y", "x", "ng")] * 12
    widget._loc2d_counts = np.ones(12, dtype=int)

    widget.bin_factor_box.setValue(4)
    widget._apply_time_binning()
    assert _pump_until(lambda: widget._bin_worker_ref is None)

    assert widget._loc2d_candidates == [None] * 3
    assert widget._loc2d_counts.tolist() == [0, 0, 0]


def test_the_factor_survives_a_settings_round_trip():
    widget = _binned_widget(4, offset=100.0, fps=100.0)
    metadata = widget._collect_metadata(None)
    assert metadata["preprocessing"]["time_bin_frames"] == 4
    # Recorded as applied: the baseline in the file is the binned one.
    assert metadata["localization_2d"]["offset_adu"] == pytest.approx(400.0)

    restored = make_widget()
    restored.apply_settings(metadata)
    assert restored.bin_factor_box.value() == 4
    assert restored._time_bin_applied == 4


def test_restoring_a_run_does_not_apply_its_binning_a_second_time():
    """The file carries a baseline that already accounts for N. Rescaling it on
    the way back in would multiply by N again and quadruple a 4-frame bin."""
    widget = _binned_widget(4, offset=100.0, fps=100.0)
    metadata = widget._collect_metadata(None)

    restored = make_widget()
    restored.apply_settings(metadata)
    assert restored.loc_offset_box.value() == pytest.approx(400.0)
    assert restored.fps_box.value() == pytest.approx(25.0)


def test_the_baseline_box_reaches_a_binned_value():
    """The old ceiling of 20000 ADU clipped a 100 ADU baseline from N=200 up,
    and a clipped baseline is under-subtracted rather than reported."""
    widget = _binned_widget(500, offset=100.0)
    assert widget.loc_offset_box.value() == pytest.approx(50000.0)


# --- against a real napari viewer ---------------------------------------------
#
# The widget tests above run on a stub viewer, which accepts a new `data` without
# having any opinion about it. Swapping the frame count on a live Image layer is
# exactly the operation napari might object to, so it is worth doing for real.


def test_a_live_napari_layer_follows_the_new_frame_count():
    from napari.components import ViewerModel

    from test_widget_interaction import ensure_qapp

    ensure_qapp()
    widget = widget_mod.LocalizationTrackingWidget(ViewerModel())
    _LIVE_WIDGETS.append(widget)
    raw = _raw(n=12, size=16)
    widget._on_load_finished((None, raw, "decoded", None, raw), "", "stack.tif")
    assert widget.viewer.dims.nsteps[0] == 12

    widget.bin_factor_box.setValue(4)
    widget._apply_time_binning()
    assert _pump_until(lambda: widget._bin_worker_ref is None)

    layer = widget.viewer.layers["stack.tif"]
    assert layer.data.shape == (3, 16, 16)
    assert widget.viewer.dims.nsteps[0] == 3
    np.testing.assert_array_equal(
        np.asarray(layer.data[0]), raw[:4].sum(axis=0, dtype=np.uint32))


# Each widget owns a dozen matplotlib canvases; letting Qt collect them mid
# session crashes the interpreter, so they are held for the run.
_LIVE_WIDGETS = []


def test_loading_straight_after_changing_the_factor_still_rescales():
    """The box is debounced; pressing Load can beat the timer to it.

    The stack is opened at the new factor either way, so the baseline and the
    frame rate have to be there when the fit runs - not one timer tick later,
    by which point the factor already looks applied and nothing moves.
    """
    widget = make_widget()
    widget.loc_offset_box.setValue(100.0)
    widget.fps_box.setValue(100.0)
    raw = _raw(n=12, size=8)

    widget.bin_factor_box.setValue(4)          # debounce timer armed, not fired
    _loaded(widget, raw)

    assert widget.loc_offset_box.value() == pytest.approx(400.0)
    assert widget.fps_box.value() == pytest.approx(25.0)
    # and the pending timer must not then apply it a second time
    widget._apply_time_binning()
    assert widget.loc_offset_box.value() == pytest.approx(400.0)
