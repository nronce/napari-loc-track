"""Opening image stacks without reading them.

A localization movie is routinely tens of GB. Nothing here decodes pixels up
front unless there is no alternative, so "Load data" returns as soon as the file
header has been parsed and frames arrive as they are actually looked at.
"""
from __future__ import annotations

import numpy as np

# Above this decoded size, a stack that cannot be memory-mapped is opened lazily
# rather than read into RAM. Decoding lazily costs more in total - every full
# pass (detect, then fit) decodes again, ~6x the cost of a single threaded read -
# so it is only worth it when the alternative is exhausting memory.
_EAGER_LIMIT_BYTES = 2 * 1024 ** 3
# Frames decoded per read when a stack has to be read into RAM. Bounds how long
# a cancel request waits, at the cost of a few extra reads.
_DECODE_CHUNK_FRAMES = 64


def _cancelled(cancel):
    return cancel is not None and cancel.is_set()


def _read_into_ram(image_path, cancel):
    """Decode a whole stack, in chunks so the read can be cancelled part way."""
    import tifffile

    with tifffile.TiffFile(image_path) as tif:
        series = tif.series[0]
        shape = series.shape
        n_frames = shape[0] if len(shape) > 2 else 0
        if n_frames == 0 or len(series.pages) != n_frames:
            # Not one page per frame; no safe way to slice it up.
            if _cancelled(cancel):
                return None
            return series.asarray()

        out = np.empty(shape, series.dtype)
        step = max(1, min(_DECODE_CHUNK_FRAMES, n_frames))
        for start in range(0, n_frames, step):
            if _cancelled(cancel):
                return None
            stop = min(start + step, n_frames)
            out[start:stop] = tif.asarray(key=slice(start, stop))
        return out


def open_image_stack(image_path, eager_limit_bytes=_EAGER_LIMIT_BYTES, cancel=None):
    """Open a TIFF stack with the cheapest strategy that works for that file.

    Returns (array, description). The array is always at least 2D and indexes
    frames on the first axis once the caller has added the leading axis for a
    single-page file. `cancel` is an event checked while decoding; if it is set,
    (None, "cancelled") is returned.
    """
    import tifffile

    mappable = False
    nbytes = 0
    try:
        with tifffile.TiffFile(image_path) as tif:
            series = tif.series[0]
            mappable = bool(
                series.dataoffset is not None and series.keyframe.is_memmappable
            )
            nbytes = int(series.size) * int(series.dtype.itemsize)
    except Exception:
        pass

    if _cancelled(cancel):
        return None, "cancelled"

    if mappable:
        # Uncompressed, native byte order, pages sequential in the file: the
        # array is just a window onto the file. Zero copy, nothing decoded, so
        # this is instant no matter how large the movie is.
        return tifffile.memmap(image_path, mode="r"), "memory-mapped"

    # Compressed, byte-swapped, or pages not sequential - OME and Micro-Manager
    # stacks are usually in this class, and decoding is unavoidable. What must be
    # avoided is imread(out="memmap"): despite the name it decodes the *entire*
    # movie into a full-size temporary file on disk, so a 20 GB stack is read,
    # decoded and written back to %TEMP% before the first frame is ever shown.
    if nbytes > eager_limit_bytes:
        try:
            import dask.array as da

            store = tifffile.imread(image_path, aszarr=True)
            return da.from_zarr(store), f"lazy, {nbytes / 1e9:.1f} GB (decoded per frame)"
        except Exception:
            pass

    stack = _read_into_ram(image_path, cancel)
    if stack is None:
        return None, "cancelled"
    return stack, "decoded into RAM"
