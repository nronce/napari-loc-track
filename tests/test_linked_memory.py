"""Trajectories kept for the localizations they were linked from.

Changing a localization filter clears the trajectories - they were linked from
other localizations. Going back to a setting that was linked before must bring
them back as they were, metrics and all, instead of linking again: on a full
acquisition a link takes minutes.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_dynamics_filter import _loaded  # noqa: E402


def _linked(widget):
    """Mark the trajectories on the widget as linked from what is filtered now."""
    widget._tracks_key = widget._localization_set_key()
    return widget.tracks


def _sigma_bounds(widget):
    return widget.filter_controls["sigma [nm]"]


def test_going_back_to_a_filter_setting_brings_its_trajectories_back():
    widget = _loaded(distance=dict.fromkeys(range(5), 1.0))
    tracks = _linked(widget)
    low, high = _sigma_bounds(widget)
    original = low.value()

    low.setValue(130.0)                       # every localization filtered out
    widget.apply_filters()
    assert widget.tracks is None

    low.setValue(original)
    widget.apply_filters()
    assert widget.tracks is tracks
    assert widget._track_distance_cache == dict.fromkeys(range(5), 1.0)


def test_other_localizations_are_not_given_those_trajectories():
    widget = _loaded()
    _linked(widget)
    low, _high = _sigma_bounds(widget)
    low.setValue(130.0)
    widget.apply_filters()
    assert widget.tracks is None
    assert widget._localization_set_key() not in widget._linked_memory


def test_changing_a_linking_setting_is_a_different_link():
    widget = _loaded()
    _linked(widget)
    low, _high = _sigma_bounds(widget)
    original = low.value()
    low.setValue(130.0)
    widget.apply_filters()
    widget.search_box.setValue(widget.search_box.value() * 2)
    low.setValue(original)
    widget.apply_filters()
    assert widget.tracks is None


def test_new_data_forgets_what_was_linked():
    widget = _loaded()
    _linked(widget)
    widget._invalidate_tracks(reason="test")
    assert widget._linked_memory
    locs = widget.df.copy()
    widget._ingest_localization_dataframe(locs, "loaded", True)
    assert not widget._linked_memory


def test_metrics_finishing_late_go_to_the_trajectories_they_were_computed_for():
    widget = _loaded()
    tracks = _linked(widget)
    widget._invalidate_tracks(reason="test")          # to memory, before the metrics arrive
    widget._on_fit_free_metrics_finished({"distance": dict.fromkeys(range(5), 2.0)}, tracks)
    assert widget._track_distance_cache is None
    kept = next(iter(widget._linked_memory.values()))
    assert kept["caches"]["_track_distance_cache"] == dict.fromkeys(range(5), 2.0)
