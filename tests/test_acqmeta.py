"""Reading the acquisition parameters back off the file that recorded them.

The point of this reader is to stop people typing the frame rate in by hand, so
what it must never do is invent one. Most of what follows is about the two ways
that could happen: reporting a calibration the microscope never made (Micro-
Manager writes 0.0, not nothing, for an uncalibrated pixel size), and trusting
the interval the software was *asked* for over the one the camera achieved.

The other half is about size. A sidecar for a long movie is ~100 MB, so it is
read from the front and always stops mid-record; the parser has to lose that
half-record rather than let it shift every timestamp against its frame number.
"""
import json

import pytest

from conftest import load_acqmeta

acqmeta = load_acqmeta()


# The MDA settings block is embedded in the summary as an escaped JSON *string*.
# Its braces are text, and a brace counter that does not know that stops the
# summary halfway through - which is why it is in every fixture here.
_MDA_SETTINGS = json.dumps(
    {"numFrames": 10000, "root": "D:\\Nathan\\2026", "comment": "{unclosed",
     "channels": [], "save": True}
)


def _summary(**overrides):
    summary = {
        "Prefix": "100mW-HiLo-stab_3",
        "MicroManagerVersion": "2.0.3 20260407",
        "MdaSettings": _MDA_SETTINGS,
        "Interval_ms": 0.0,          # free-running: nothing was requested
        "Frames": 10000,
        "Width": 478,
        "Height": 503,
        "StartTime": "2026-07-24 19:22:44.511 +0200",
        "InitialScopeData": {
            "Camera-1-ActualInterval-ms": {"type": "STRING", "scalar": "35.4100"},
            "Core-Camera": {"type": "STRING", "scalar": "Camera-1"},
        },
    }
    summary.update(overrides)
    return summary


def _record(elapsed, **overrides):
    record = {
        "Camera": "Camera-1",
        "Core-Camera": "Camera-1",
        "Binning": 1,
        "ROI": "321-270-478-503",
        "BitDepth": 16,
        "Exposure-ms": 20.0,
        "ElapsedTime-ms": elapsed,
        "PixelSizeUm": 0.0,          # nobody calibrated the objective
        "Camera-1-Offset": "100",
        "Camera-1-Exposure": "20.00",
        "Camera-1-ChipName": "GS144BSI",
        "Camera-1-ActualInterval-ms": "35.4100",
        "TINosePiece-Label": "4-Plan Apo TIRF 60x NA 1.45 Oil",
    }
    record.update(overrides)
    return record


def _sidecar_text(summary=None, times=(), channels=1, record_overrides=None):
    """A Micro-Manager metadata.txt: the summary, then one record per frame."""
    blocks = ['"Summary": ' + json.dumps(summary or _summary(), indent=2)]
    for index, elapsed in enumerate(times):
        for channel in range(channels):
            record = _record(elapsed, **(record_overrides or {}))
            blocks.append(f'"FrameKey-{index}-{channel}-0": '
                          + json.dumps(record, indent=2))
    return "{\n" + ",\n".join(blocks) + "\n}"


def _write(tmp_path, text, name="stab_MMStack_Pos0_metadata.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _read(tmp_path, text, stack="stab_MMStack_Pos0.ome.tif", **kwargs):
    """Run the reader against a sidecar, with no TIFF beside it."""
    _write(tmp_path, text)
    return acqmeta.read_acquisition_metadata(tmp_path / stack, **kwargs)


# ----------------------------------------------------------------------
# Parsing a file that is too big to parse
# ----------------------------------------------------------------------
def test_summary_survives_the_json_string_embedded_in_it():
    text = _sidecar_text(times=[0.0])
    summary = acqmeta.summary_of(text)
    assert summary["Frames"] == 10000
    assert summary["Width"] == 478
    # The braces inside MdaSettings are characters in a string, not structure.
    assert json.loads(summary["MdaSettings"])["numFrames"] == 10000


def test_object_running_past_the_end_of_the_read_is_refused():
    text = _sidecar_text(times=[0.0])
    assert acqmeta.json_object_at(text[:200], text.index("{")) is None


def test_frame_times_pair_each_key_with_its_own_timestamp():
    text = _sidecar_text(times=[10.0, 42.0, 73.0])
    assert acqmeta.frame_times(text) == [(0, 10.0), (1, 42.0), (2, 73.0)]


def test_a_record_cut_in_half_is_dropped_not_misaligned():
    """The head read always stops mid-record; that must cost one frame, not all.

    Collecting keys and timestamps separately and zipping them would pair frame
    0 with frame 0's time only until the first truncated record, after which
    every pair would be off by one - a plausible, silently wrong frame rate.
    """
    text = _sidecar_text(times=[10.0, 42.0, 73.0])
    truncated = text[:text.index('"FrameKey-2-0-0"') + 60]
    assert acqmeta.frame_times(truncated) == [(0, 10.0), (1, 42.0)]


def test_extra_channels_and_slices_do_not_time_the_run_twice():
    text = _sidecar_text(times=[10.0, 42.0, 73.0], channels=3)
    assert acqmeta.frame_times(text) == [(0, 10.0), (1, 42.0), (2, 73.0)]


# ----------------------------------------------------------------------
# Measuring the frame rate
# ----------------------------------------------------------------------
def test_interval_is_measured_across_the_span_not_between_neighbours():
    """Micro-Manager rounds each timestamp to the millisecond.

    On a 31.36 ms frame that is a 3% error on any single difference - every
    neighbouring pair reads 31 or 32 ms and nothing reads the truth. Measured
    over the span, the same rounding is spread across hundreds of frames.
    """
    true_interval = 31.365
    times = [(index, round(index * true_interval)) for index in range(400)]
    measured = acqmeta.frame_interval_ms(times)
    assert measured == pytest.approx(true_interval, abs=0.01)

    neighbours = {times[i + 1][1] - times[i][1] for i in range(len(times) - 1)}
    assert neighbours == {31, 32}  # no single difference could have given it


def test_interval_is_per_frame_of_the_movie_not_per_sampled_record():
    """Frame indices, not record count: the sampled records need not be dense."""
    sparse = [(0, 0.0), (10, 500.0), (20, 1000.0), (30, 1500.0),
              (40, 2000.0), (50, 2500.0), (60, 3000.0), (70, 3500.0)]
    assert acqmeta.frame_interval_ms(sparse) == pytest.approx(50.0)


def test_too_few_frames_is_not_a_measurement():
    assert acqmeta.frame_interval_ms([(0, 0.0), (1, 30.0)]) is None


def test_frames_that_share_a_timestamp_give_no_measurement():
    assert acqmeta.frame_interval_ms([(index, 5.0) for index in range(20)]) is None


# ----------------------------------------------------------------------
# Which source wins
# ----------------------------------------------------------------------
def test_measured_timing_beats_what_the_acquisition_asked_for(tmp_path):
    """The real acquisition: free-running, and the camera's own figure is stale.

    Interval_ms is 0.0 because nothing was requested, and the camera reports the
    35.41 ms it was last configured for while actually delivering ~31.4 ms.
    """
    times = [round(index * 31.365) for index in range(400)]
    found = _read(tmp_path, _sidecar_text(times=times))["values"]
    assert found["frame_interval_ms"] == pytest.approx(31.365, abs=0.01)
    assert found["fps"] == pytest.approx(1000.0 / 31.365, abs=0.02)


def test_requested_interval_is_used_when_nothing_was_timed(tmp_path):
    summary = _summary(Interval_ms=50.0)
    found = _read(tmp_path, _sidecar_text(summary=summary, times=[0.0]))["values"]
    assert found["frame_interval_ms"] == pytest.approx(50.0)
    assert found["fps"] == pytest.approx(20.0)


def test_camera_reported_interval_is_the_last_resort(tmp_path):
    """Nothing timed and nothing requested: fall back to the camera's figure."""
    result = _read(tmp_path, _sidecar_text(times=[0.0]))
    assert result["values"]["frame_interval_ms"] == pytest.approx(35.41)
    assert "ActualInterval-ms" in result["sources"]["frame_interval_ms"]


def test_a_free_running_acquisition_is_not_read_as_zero_interval(tmp_path):
    """Interval_ms 0.0 means "not requested" and must not become 0 ms, or inf fps."""
    text = _sidecar_text(times=[0.0], record_overrides={"Camera-1-ActualInterval-ms": "0"})
    found = _read(tmp_path, text)["values"]
    assert "frame_interval_ms" not in found
    assert "fps" not in found


# ----------------------------------------------------------------------
# Never inventing a calibration
# ----------------------------------------------------------------------
def test_uncalibrated_pixel_size_is_reported_as_absent(tmp_path):
    """PixelSizeUm 0.0 is Micro-Manager for "nobody set this"."""
    found = _read(tmp_path, _sidecar_text(times=[0.0]))["values"]
    assert "pixel_size_nm" not in found


def test_a_real_pixel_size_is_converted_to_nanometres(tmp_path):
    text = _sidecar_text(times=[0.0], record_overrides={"PixelSizeUm": 0.1083})
    found = _read(tmp_path, text)["values"]
    assert found["pixel_size_nm"] == pytest.approx(108.3)


def test_the_objective_is_reported_but_never_turned_into_a_pixel_size(tmp_path):
    """Magnification alone cannot give a pixel size - the sensor pitch is not recorded."""
    found = _read(tmp_path, _sidecar_text(times=[0.0]))["values"]
    assert found["objective"] == "4-Plan Apo TIRF 60x NA 1.45 Oil"
    assert "pixel_size_nm" not in found


def test_camera_offset_and_exposure_are_read(tmp_path):
    found = _read(tmp_path, _sidecar_text(times=[0.0]))["values"]
    assert found["camera_offset_adu"] == pytest.approx(100.0)
    assert found["exposure_ms"] == pytest.approx(20.0)
    assert found["camera_chip"] == "GS144BSI"


def test_a_camera_offset_of_zero_is_a_real_value(tmp_path):
    """Unlike a pixel size, zero offset is a legitimate camera setting."""
    text = _sidecar_text(times=[0.0], record_overrides={"Camera-1-Offset": "0"})
    assert _read(tmp_path, text)["values"]["camera_offset_adu"] == 0.0


def test_the_camera_need_not_be_called_camera_1(tmp_path):
    """The device name is part of every key, so one hard-coded prefix is not enough."""
    text = _sidecar_text(times=[0.0], record_overrides={
        "Core-Camera": "Andor-Zyla",
        "Andor-Zyla-Offset": "250",
        "Andor-Zyla-ActualInterval-ms": "9.5",
    })
    found = _read(tmp_path, text)["values"]
    assert found["camera_offset_adu"] == pytest.approx(250.0)
    assert found["frame_interval_ms"] == pytest.approx(9.5)


def test_every_value_says_where_it_came_from(tmp_path):
    result = _read(tmp_path, _sidecar_text(times=[round(i * 31.0) for i in range(50)]))
    assert set(result["sources"]) == set(result["values"])
    assert "ElapsedTime-ms" in result["sources"]["frame_interval_ms"]
    assert "stab_MMStack_Pos0_metadata.txt" in result["sources"]["exposure_ms"]


# ----------------------------------------------------------------------
# Finding the sidecar, and surviving what is in it
# ----------------------------------------------------------------------
def test_sidecar_is_found_beside_an_ome_tif(tmp_path):
    path = _write(tmp_path, _sidecar_text(times=[0.0]))
    assert acqmeta.sidecar_path(tmp_path / "stab_MMStack_Pos0.ome.tif") == path


def test_a_split_acquisition_shares_the_first_parts_sidecar(tmp_path):
    """MM writes ..._Pos0.ome.tif, ..._Pos0_1.ome.tif, one metadata.txt for all."""
    path = _write(tmp_path, _sidecar_text(times=[0.0]))
    assert acqmeta.sidecar_path(tmp_path / "stab_MMStack_Pos0_1.ome.tif") == path
    assert acqmeta.sidecar_path(tmp_path / "stab_MMStack_Pos0_7.ome.tif") == path


def test_no_sidecar_and_no_readable_tiff_is_empty_not_an_error(tmp_path):
    result = acqmeta.read_acquisition_metadata(tmp_path / "nothing.tif")
    assert result == {"values": {}, "sources": {}}


def test_a_truncated_head_read_still_yields_a_frame_rate(tmp_path):
    """What the real file forces: 100 MB of records, a few hundred of them read."""
    text = _sidecar_text(times=[round(index * 31.365) for index in range(500)])
    result = _read(tmp_path, text, head_bytes=60_000)
    assert result["values"]["frame_interval_ms"] == pytest.approx(31.365, abs=0.05)
    # Only part of the file was consulted, and it says so.
    sampled = int(result["sources"]["frame_interval_ms"].split("over ")[1].split()[0])
    assert 8 <= sampled < 500


def test_damaged_json_does_not_stop_the_rest_being_read(tmp_path):
    text = _sidecar_text(times=[round(i * 31.0) for i in range(50)])
    text = text.replace('"Summary": {', '"Summary": {,,,', 1)
    found = _read(tmp_path, text)["values"]
    assert found["frame_interval_ms"] == pytest.approx(31.0, abs=0.1)
    assert "n_frames" not in found  # the summary itself is gone, and stays gone


# ----------------------------------------------------------------------
# OME-TIFF, for stacks that carry their calibration in the header
# ----------------------------------------------------------------------
def test_ome_calibration_is_read_from_the_stack(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    import numpy as np

    path = tmp_path / "plain.ome.tif"
    tifffile.imwrite(
        path, np.zeros((6, 8, 8), np.uint16), ome=True,
        metadata={"axes": "TYX", "PhysicalSizeX": 0.1083,
                  "PhysicalSizeXUnit": "\u00b5m", "TimeIncrement": 0.0314,
                  "TimeIncrementUnit": "s"},
    )
    found = acqmeta.read_acquisition_metadata(path)["values"]
    assert found["pixel_size_nm"] == pytest.approx(108.3)
    assert found["frame_interval_ms"] == pytest.approx(31.4)
    assert found["fps"] == pytest.approx(1000.0 / 31.4)
    assert found["n_frames"] == 6


def test_a_sidecar_measurement_outranks_the_ome_header(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    import numpy as np

    path = tmp_path / "stab_MMStack_Pos0.ome.tif"
    tifffile.imwrite(
        path, np.zeros((6, 8, 8), np.uint16), ome=True,
        metadata={"axes": "TYX", "TimeIncrement": 1.0, "TimeIncrementUnit": "s"},
    )
    _write(tmp_path, _sidecar_text(times=[round(i * 31.365) for i in range(400)]))
    found = acqmeta.read_acquisition_metadata(path)["values"]
    assert found["frame_interval_ms"] == pytest.approx(31.365, abs=0.01)


def test_pixel_size_units_other_than_micrometres(tmp_path):
    for unit, value, expected in (("nm", 108.3, 108.3), ("um", 0.1083, 108.3),
                                  ("mm", 0.0001083, 108.3)):
        attributes = acqmeta._pixels_attributes(
            f'<OME><Image><Pixels SizeX="8" PhysicalSizeX="{value}" '
            f'PhysicalSizeXUnit="{unit}"></Pixels></Image></OME>'
        )
        collector = acqmeta._Collector()
        acqmeta._read_ome(collector, f'<Pixels PhysicalSizeX="{value}" '
                                     f'PhysicalSizeXUnit="{unit}">', "x")
        assert attributes["PhysicalSizeXUnit"] == unit
        assert collector.values["pixel_size_nm"] == pytest.approx(expected)


def test_an_unrecognised_unit_is_refused_rather_than_assumed(tmp_path):
    collector = acqmeta._Collector()
    acqmeta._read_ome(collector, '<Pixels PhysicalSizeX="3" PhysicalSizeXUnit="furlong">', "x")
    assert "pixel_size_nm" not in collector.values
