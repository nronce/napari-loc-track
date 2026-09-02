"""Setting the speed of napari's play button, for screen-recorded movies.

Composite movies are no longer written frame by frame at the render's own
resolution - they are screen-recorded from the viewer, where every layer already
looks exactly as it should. What that needs from the plugin is control over how
fast the play button runs, and an honest statement of what the chosen speed
means against the rate the camera actually acquired at.

The playback settings belong to napari, not to this plugin: the play button
reads them, so these tests check the values land there rather than in some
private copy that nothing would consult.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_widget_interaction import make_widget  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_napari_settings():
    """These are the user's real napari settings, so put them back."""
    application = widget_mod._napari_playback_settings()
    if application is None:
        yield
        return
    before = (application.playback_fps, application.playback_mode)
    yield
    application.playback_fps, application.playback_mode = before


def _settings():
    application = widget_mod._napari_playback_settings()
    if application is None:
        pytest.skip("this napari has no playback settings")
    return application


def test_the_speed_reaches_napari_which_owns_the_play_button():
    widget = make_widget()
    widget.playback_fps_box.setValue(24)
    assert _settings().playback_fps == 24


def test_what_happens_at_the_end_reaches_napari_too():
    widget = make_widget()
    index = widget.playback_mode_box.findData("once")
    widget.playback_mode_box.setCurrentIndex(index)
    assert str(getattr(_settings().playback_mode, "value",
                       _settings().playback_mode)) == "once"


def test_the_box_opens_on_whatever_napari_is_already_set_to():
    """napari persists these between sessions, so the plugin must not fight it."""
    _settings().playback_fps = 17.0
    widget = make_widget()
    assert widget.playback_fps_box.value() == 17


def test_real_time_means_the_nearest_whole_rate_it_was_acquired_at():
    """napari plays at whole frames per second, so 31.9 fps can only be 32."""
    widget = make_widget()
    widget.fps_box.setValue(31.882)
    widget.playback_realtime_button.click()
    assert widget.playback_fps_box.value() == 32
    assert "real time" in widget.playback_status.text()


def test_the_speed_napari_is_given_is_always_a_whole_number():
    """`playback_fps` is an int field: a fractional one is refused outright.

    The control used to be a decimal box whose own minimum, 0.1, was a value
    napari would not accept - so simply opening the tab could log a validation
    error before anything had been asked for.
    """
    widget = make_widget()
    assert widget.playback_fps_box.minimum() >= 1
    widget.playback_fps_box.setValue(widget.playback_fps_box.minimum())
    assert isinstance(_settings().playback_fps, int)
    assert "Could not set the playback speed" not in widget.log_box.toPlainText()

    # and the whole offered range is acceptable to napari
    for value in (widget.playback_fps_box.minimum(), 30, widget.playback_fps_box.maximum()):
        widget.playback_fps_box.setValue(value)
        assert _settings().playback_fps == value
    assert "Could not set the playback speed" not in widget.log_box.toPlainText()


@pytest.mark.parametrize("acquired,playback,expected", [
    (32.0, 32, "real time"),
    (32.0, 64, "2x faster than real time"),
    (32.0, 16, "2x slower than real time"),
])
def test_the_speed_is_quoted_against_the_acquisition_rate(acquired, playback, expected):
    """A frame rate alone does not say whether what you are watching is sped up."""
    widget = make_widget()
    widget.fps_box.setValue(acquired)
    widget.playback_fps_box.setValue(int(playback))
    assert expected in widget.playback_status.text()


def test_changing_the_acquisition_rate_restates_the_pace():
    """The frame rate box is on another tab, and is autofilled from metadata."""
    widget = make_widget()
    widget.playback_fps_box.setValue(10)
    widget.fps_box.setValue(10.0)
    assert "real time" in widget.playback_status.text()
    widget.fps_box.setValue(100.0)
    assert "10x slower than real time" in widget.playback_status.text()


def test_a_composite_movie_can_no_longer_be_written_frame_by_frame():
    """It is screen-recorded from the viewer instead - the reason speed is here."""
    widget = make_widget()
    formats = [widget.render_movie_format_box.itemData(i)
               for i in range(widget.render_movie_format_box.count())]
    assert "composite" not in formats
    assert "data" in formats and "display" in formats
    assert not hasattr(widget, "render_save_composite_movie_button")
    # a composite still is unaffected
    image_formats = [widget.render_image_format_box.itemData(i)
                     for i in range(widget.render_image_format_box.count())]
    assert "composite" in image_formats
