"""What the microscope wrote down about an acquisition.

The numbers a fit needs - how fast the camera ran, how big a pixel is on the
sample, where the camera's zero sits - are all recorded at acquisition time and
then, traditionally, typed back into the plugin by hand. Everything here exists
to read them off the file instead, so that loading a stack fills the boxes in.

Two rules shape the whole module. Nothing is guessed: a value the microscope
never recorded is left absent rather than approximated, because a plausible
wrong pixel size is far more damaging than an empty box the user notices. And
nothing large is read: a Micro-Manager sidecar for a long movie is bigger than
some of the movies, so it is sampled from the front, never parsed whole.

Only the TIFF reader is imported lazily inside the function that needs it, so
this module can be exercised without napari, Qt or tifffile present.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# How much of a Micro-Manager sidecar to read. It holds one ~8 KB record per
# frame - a 10000 frame movie runs to ~100 MB - so json.load on the whole file
# would cost more than opening the movie it describes. The summary sits at the
# front and a few hundred frame records time the acquisition far better than it
# needs, so a bounded head read answers every question asked here.
_SIDECAR_HEAD_BYTES = 4 * 1024 ** 2

# Below this many timed frames, a measured interval is noise; whatever the
# acquisition software said it was aiming for is the better answer.
_MIN_TIMED_FRAMES = 8

# Only the opening <Pixels> tag of an OME header carries the calibration. The
# TiffData block after it is one entry per plane and runs to megabytes on a long
# movie, so the search is bounded to the front of the document.
_OME_HEAD_CHARS = 65536

_LENGTH_TO_NM = {"nm": 1.0, "um": 1000.0, "µm": 1000.0, "μm": 1000.0,
                 "mm": 1e6, "cm": 1e7, "m": 1e9}
_TIME_TO_MS = {"ms": 1.0, "s": 1000.0, "us": 1e-3, "µs": 1e-3,
               "μs": 1e-3, "ns": 1e-6, "min": 60000.0}

_TIFF_SUFFIXES = (".tif", ".tiff")


def _number(value):
    """A finite float, or None for anything that cannot stand in for one."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _positive(value):
    """A strictly positive float, or None.

    Micro-Manager writes 0.0 for a calibration nobody set - PixelSizeUm on an
    uncalibrated objective, Interval_ms on a free-running acquisition - so zero
    here means "not recorded", not "zero", and must never reach a control.
    """
    number = _number(value)
    return number if number is not None and number > 0.0 else None


def json_object_at(text, brace_index):
    """The JSON object starting at `brace_index`, as a string, or None.

    Brace counting has to understand string literals: a Micro-Manager summary
    embeds the entire MDA settings block as an escaped JSON *string*, so the
    braces inside it are text and closing on one would truncate the object
    halfway through.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(brace_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_index:index + 1]
    return None  # the object runs past the end of what was read


def _object_after(text, pattern):
    """Parse the JSON object introduced by the first match of `pattern`."""
    match = re.search(pattern, text)
    if match is None:
        return {}
    blob = json_object_at(text, match.end() - 1)
    if blob is None:
        return {}
    try:
        parsed = json.loads(blob)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def summary_of(text):
    """The "Summary" block of a Micro-Manager sidecar."""
    return _object_after(text, r'"Summary"\s*:\s*\{')


def first_frame_record(text):
    """The first per-frame record, which carries the live device settings.

    The summary describes what was *requested*; the frame record describes what
    the hardware was actually set to when the shutter opened, which is what the
    fit has to be told about.
    """
    return _object_after(text, r'"FrameKey-\d+-\d+-\d+"\s*:\s*\{')


_FRAME_FIELD = re.compile(
    r'"FrameKey-(\d+)-(\d+)-(\d+)"|"ElapsedTime-ms"\s*:\s*(-?[\d.]+(?:[eE][-+]?\d+)?)'
)


def frame_times(text):
    """[(frame index, elapsed ms)] for the first channel and slice.

    The key and the timestamp are matched in a single pass rather than collected
    by two separate searches and zipped. A head read almost always stops in the
    middle of a record, and pairing each timestamp with the key that introduced
    it means a truncated tail is dropped, instead of silently shifting the
    timestamps against the frame numbers.
    """
    times = []
    pending = None
    for match in _FRAME_FIELD.finditer(text):
        if match.group(1) is not None:
            pending = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            continue
        if pending is None:
            continue
        frame, channel, z_slice = pending
        pending = None
        if channel or z_slice:
            continue  # one channel and one slice is enough to time the run
        elapsed = _number(match.group(4))
        if elapsed is not None:
            times.append((frame, elapsed))
    return times


def frame_interval_ms(times):
    """Milliseconds per frame, measured across the sampled records.

    Taken from the span between the first and last sampled frame rather than the
    mean or median of consecutive differences. Micro-Manager rounds every
    timestamp to the millisecond, which on a ~30 ms frame is a 3% quantisation on
    any single difference but a rounding error once spread over hundreds of
    frames. Dividing by the difference of the frame *indices*, not by the number
    of records, also keeps the answer per-frame-of-the-movie even when the
    sampled records are not consecutive.
    """
    if len(times) < _MIN_TIMED_FRAMES:
        return None
    ordered = sorted(times)
    span_frames = ordered[-1][0] - ordered[0][0]
    span_ms = ordered[-1][1] - ordered[0][1]
    if span_frames <= 0 or span_ms <= 0:
        return None
    return span_ms / span_frames


def sidecar_path(image_path):
    """The Micro-Manager *_metadata.txt beside a stack, if there is one."""
    path = Path(image_path)
    base = path.name
    lowered = base.lower()
    for suffix in _TIFF_SUFFIXES:
        if lowered.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.lower().endswith(".ome"):
        base = base[:-4]

    names = [f"{base}_metadata.txt"]
    # A long acquisition is split into ..._Pos0.ome.tif, ..._Pos0_1.ome.tif and
    # so on, and every part is described by the first part's sidecar.
    trimmed = re.sub(r"_\d+$", "", base)
    if trimmed != base:
        names.append(f"{trimmed}_metadata.txt")

    for name in names:
        candidate = path.parent / name
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _scalar(record, key):
    """Read a device property, which is a bare value or a {"scalar": ...} pair."""
    value = record.get(key)
    if isinstance(value, dict):
        value = value.get("scalar")
    return value


def _camera_property(record, scope_data, name, camera):
    """A camera property, looked up under whatever the camera is actually called.

    The device name is part of every key ("Camera-1-Exposure"), so a reader that
    hard-codes one prefix works on exactly one microscope.
    """
    keys = [f"{camera}-{name}"] if camera else []
    keys.append(f"Camera-{name}")
    for source in (record, scope_data):
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = _scalar(source, key)
            if value not in (None, ""):
                return value
        # Fall back to any device that names itself a camera and has the field.
        for key in source:
            if key.endswith(f"-{name}") and key.lower().startswith("camera"):
                value = _scalar(source, key)
                if value not in (None, ""):
                    return value
    return None


class _Collector:
    """Accumulates the first value found for each field, and where it came from.

    First writer wins, so callers apply their sources in order of trust and never
    have to check what an earlier one already established.
    """

    def __init__(self):
        self.values = {}
        self.sources = {}

    def add(self, key, value, source):
        if value is None or key in self.values:
            return
        self.values[key] = value
        self.sources[key] = source

    def add_number(self, key, value, source):
        self.add(key, _positive(value), source)

    def add_text(self, key, value, source):
        if value is None:
            return
        text = str(value).strip()
        if text:
            self.add(key, text, source)


def _read_sidecar(collector, path, head_bytes):
    """Fill from a Micro-Manager sidecar: measured timing, then device state."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read(head_bytes)
    except OSError:
        return
    text = raw.decode("utf-8-sig", "replace")
    where = path.name

    times = frame_times(text)
    measured = frame_interval_ms(times)
    if measured is not None:
        collector.add(
            "frame_interval_ms", measured,
            f"{where}: measured over {len(times)} frames of ElapsedTime-ms",
        )

    record = first_frame_record(text)
    summary = summary_of(text)
    scope_data = summary.get("InitialScopeData") if isinstance(summary, dict) else {}
    camera = _scalar(record, "Core-Camera") or _scalar(scope_data, "Core-Camera")

    # The requested interval, for a timed acquisition; 0.0 when free-running.
    collector.add_number("frame_interval_ms", summary.get("Interval_ms"),
                         f"{where}: Summary.Interval_ms")
    actual = _camera_property(record, scope_data, "ActualInterval-ms", camera)
    collector.add_number("frame_interval_ms", actual,
                         f"{where}: {camera or 'Camera'}-ActualInterval-ms")

    for source_name, holder, key in (
        (f"{where}: FrameKey.PixelSizeUm", record, "PixelSizeUm"),
        (f"{where}: Summary.PixelSize_um", summary, "PixelSize_um"),
    ):
        micrometres = _positive(holder.get(key) if isinstance(holder, dict) else None)
        if micrometres is not None:
            collector.add("pixel_size_nm", micrometres * 1000.0, source_name)

    exposure = record.get("Exposure-ms")
    if _positive(exposure) is None:
        exposure = _camera_property(record, scope_data, "Exposure", camera)
    collector.add_number("exposure_ms", exposure, f"{where}: Exposure-ms")

    offset = _camera_property(record, scope_data, "Offset", camera)
    offset_value = _number(offset)
    if offset_value is not None and offset_value >= 0.0:
        collector.add("camera_offset_adu", offset_value,
                      f"{where}: {camera or 'Camera'}-Offset")

    collector.add_number("n_frames", summary.get("Frames"), f"{where}: Summary.Frames")
    collector.add_number("width", summary.get("Width"), f"{where}: Summary.Width")
    collector.add_number("height", summary.get("Height"), f"{where}: Summary.Height")
    collector.add_number("bit_depth", record.get("BitDepth"), f"{where}: BitDepth")
    collector.add_text("start_time", summary.get("StartTime"), f"{where}: Summary.StartTime")
    collector.add_text("roi", record.get("ROI"), f"{where}: ROI")
    collector.add_text("camera_chip",
                       _camera_property(record, scope_data, "ChipName", camera),
                       f"{where}: ChipName")

    # The objective is not convertible into a pixel size on its own - that needs
    # the sensor pitch, which is not recorded - but naming it lets the user
    # confirm at a glance whether the pixel size in the box belongs to this run.
    for key in record:
        if key.endswith("NosePiece-Label") or key.endswith("Objective-Label"):
            collector.add_text("objective", _scalar(record, key), f"{where}: {key}")
            break


def _pixels_attributes(ome_xml):
    """The attributes of the OME <Pixels> tag, as a dict."""
    head = ome_xml[:_OME_HEAD_CHARS]
    match = re.search(r"<(?:\w+:)?Pixels\b([^>]*)>", head)
    if match is None:
        return {}
    return dict(re.findall(r'([\w:]+)\s*=\s*"([^"]*)"', match.group(1)))


def _read_ome(collector, ome_xml, where):
    attributes = _pixels_attributes(ome_xml)
    if not attributes:
        return

    physical = _positive(attributes.get("PhysicalSizeX"))
    if physical is not None:
        unit = attributes.get("PhysicalSizeXUnit", "µm")
        factor = _LENGTH_TO_NM.get(unit)
        if factor is not None:
            collector.add("pixel_size_nm", physical * factor,
                          f"{where}: OME PhysicalSizeX")

    increment = _positive(attributes.get("TimeIncrement"))
    if increment is not None:
        unit = attributes.get("TimeIncrementUnit", "s")
        factor = _TIME_TO_MS.get(unit)
        if factor is not None:
            collector.add("frame_interval_ms", increment * factor,
                          f"{where}: OME TimeIncrement")

    collector.add_number("n_frames", attributes.get("SizeT"), f"{where}: OME SizeT")
    collector.add_number("width", attributes.get("SizeX"), f"{where}: OME SizeX")
    collector.add_number("height", attributes.get("SizeY"), f"{where}: OME SizeY")


def _read_tiff(collector, image_path):
    """Fill from the TIFF header: the embedded MM summary, then the OME block."""
    try:
        import tifffile
    except ImportError:
        return

    where = Path(image_path).name
    try:
        with tifffile.TiffFile(image_path) as tif:
            embedded = getattr(tif, "micromanager_metadata", None) or {}
            summary = embedded.get("Summary") if isinstance(embedded, dict) else None
            if isinstance(summary, dict):
                collector.add_number("frame_interval_ms", summary.get("Interval_ms"),
                                     f"{where}: MM Summary.Interval_ms")
                micrometres = _positive(summary.get("PixelSize_um"))
                if micrometres is not None:
                    collector.add("pixel_size_nm", micrometres * 1000.0,
                                  f"{where}: MM Summary.PixelSize_um")
                collector.add_number("n_frames", summary.get("Frames"),
                                     f"{where}: MM Summary.Frames")

            ome_xml = tif.ome_metadata if tif.is_ome else None
            if ome_xml:
                _read_ome(collector, ome_xml, where)
    except Exception:
        # A header this reader cannot make sense of must never stop the stack
        # from loading; the boxes simply keep the values they had.
        return


def read_acquisition_metadata(image_path, head_bytes=_SIDECAR_HEAD_BYTES):
    """Everything worth autofilling, read from a stack and its sidecar.

    Returns {"values", "sources"}: the parameters that were genuinely recorded,
    and a human-readable note of which file and field each one came from. Fields
    the acquisition never recorded are absent, so a caller can apply every value
    it finds without having to decide which ones are real.

    Sources are consulted in order of trust. Timing measured from the frame
    timestamps beats the interval the acquisition *requested*, which in turn
    beats what the camera reports its interval to be - a free-running
    acquisition records a requested interval of zero, and a camera's reported
    interval is the one it was last configured for rather than the one it
    achieved.
    """
    collector = _Collector()

    sidecar = sidecar_path(image_path)
    if sidecar is not None:
        _read_sidecar(collector, sidecar, head_bytes)
    _read_tiff(collector, image_path)

    interval = collector.values.get("frame_interval_ms")
    if interval:
        collector.values["fps"] = 1000.0 / interval
        collector.sources["fps"] = collector.sources["frame_interval_ms"]

    return {"values": collector.values, "sources": collector.sources}
