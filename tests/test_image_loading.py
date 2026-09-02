"""Opening a stack must be cheap, correct, and must never write a temp copy.

The failure mode this guards against: `tifffile.imread(path, out="memmap")`
looks like a memory map, but for any file that is not directly mappable it
decodes the whole movie into a full-size temporary file on disk first.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

_PKG_DIR = Path(__file__).resolve().parents[1] / "napari_loc_track"
if "napari_loc_track" not in sys.modules:
    _pkg = types.ModuleType("napari_loc_track")
    _pkg.__path__ = [str(_PKG_DIR)]
    sys.modules["napari_loc_track"] = _pkg
import importlib

_spec = importlib.util.spec_from_file_location(
    "napari_loc_track._imageio", _PKG_DIR / "_imageio.py"
)
imageio = importlib.util.module_from_spec(_spec)
sys.modules["napari_loc_track._imageio"] = imageio
_spec.loader.exec_module(imageio)


N_FRAMES, SIZE = 12, 32


@pytest.fixture(scope="module")
def stack_data():
    rng = np.random.default_rng(0)
    return rng.integers(80, 4000, size=(N_FRAMES, SIZE, SIZE), dtype=np.uint16)


def _written(tmp_path_factory, name, data, **kwargs):
    path = tmp_path_factory.mktemp("tif") / name
    tifffile.imwrite(path, data, **kwargs)
    return path


def test_contiguous_stack_is_memory_mapped(tmp_path_factory, stack_data):
    path = _written(tmp_path_factory, "contiguous.tif", stack_data)
    array, how = imageio.open_image_stack(path)
    assert how == "memory-mapped"
    assert isinstance(array, np.memmap)  # a window onto the file, not a copy
    assert array.shape == stack_data.shape
    np.testing.assert_array_equal(np.asarray(array), stack_data)


def test_compressed_stack_reads_correctly(tmp_path_factory, stack_data):
    path = _written(tmp_path_factory, "compressed.tif", stack_data, compression="zlib")
    array, how = imageio.open_image_stack(path)
    assert how != "memory-mapped"  # cannot be mapped; must not pretend otherwise
    assert array.shape == stack_data.shape
    np.testing.assert_array_equal(np.asarray(array), stack_data)


@pytest.mark.parametrize("eager_limit", [2 * 1024 ** 3, 0])
def test_compressed_stack_supports_per_frame_access(tmp_path_factory, stack_data, eager_limit):
    """Detection and fitting index frame by frame; every strategy must allow it.

    eager_limit=0 forces the lazy path that a stack too large for RAM would take.
    """
    path = _written(tmp_path_factory, "compressed2.tif", stack_data, compression="zlib")
    array, _how = imageio.open_image_stack(path, eager_limit_bytes=eager_limit)
    for i in (0, N_FRAMES // 2, N_FRAMES - 1):
        frame = np.asarray(array[i], dtype=np.float32)
        assert frame.shape == (SIZE, SIZE)
        np.testing.assert_array_equal(frame, stack_data[i].astype(np.float32))


def test_oversized_stack_goes_lazy_when_dask_is_available(tmp_path_factory, stack_data):
    pytest.importorskip("dask.array")
    pytest.importorskip("zarr")
    path = _written(tmp_path_factory, "big.tif", stack_data, compression="zlib")
    array, how = imageio.open_image_stack(path, eager_limit_bytes=0)
    assert "lazy" in how
    assert not isinstance(array, np.ndarray)  # nothing decoded yet
    np.testing.assert_array_equal(np.asarray(array), stack_data)


def test_single_page_file(tmp_path_factory, stack_data):
    path = _written(tmp_path_factory, "single.tif", stack_data[0])
    array, _how = imageio.open_image_stack(path)
    assert array.shape == (SIZE, SIZE)
    np.testing.assert_array_equal(np.asarray(array), stack_data[0])


def test_no_temporary_copy_is_written(tmp_path_factory, stack_data, monkeypatch):
    """imread(out='memmap') would create a NamedTemporaryFile; nothing may."""
    import tempfile

    created = []
    real = tempfile.NamedTemporaryFile

    def spy(*args, **kwargs):
        created.append(kwargs.get("suffix"))
        return real(*args, **kwargs)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", spy)
    for name, kwargs in (("plain.tif", {}), ("zlib.tif", {"compression": "zlib"})):
        path = _written(tmp_path_factory, name, stack_data, **kwargs)
        array, _how = imageio.open_image_stack(path)
        np.testing.assert_array_equal(np.asarray(array), stack_data)
    assert created == [], f"a temporary file was created: {created}"


def test_decode_can_be_cancelled(tmp_path_factory, stack_data):
    """A cancel set before the read starts must stop it, not return a partial stack."""
    import threading

    path = _written(tmp_path_factory, "cancel.tif", stack_data, compression="zlib")
    cancel = threading.Event()
    cancel.set()
    array, how = imageio.open_image_stack(path, eager_limit_bytes=1 << 40, cancel=cancel)
    assert array is None
    assert how == "cancelled"


def test_decode_completes_when_not_cancelled(tmp_path_factory, stack_data):
    """The chunked read must reassemble the stack exactly."""
    import threading

    path = _written(tmp_path_factory, "chunked.tif", stack_data, compression="zlib")
    array, how = imageio.open_image_stack(
        path, eager_limit_bytes=1 << 40, cancel=threading.Event()
    )
    assert how == "decoded into RAM"
    np.testing.assert_array_equal(array, stack_data)


def test_decode_chunk_boundaries(tmp_path_factory, stack_data, monkeypatch):
    """Chunking must not drop or duplicate frames when it does not divide evenly."""
    monkeypatch.setattr(imageio, "_DECODE_CHUNK_FRAMES", 5)  # 12 frames / 5 -> 5, 5, 2
    path = _written(tmp_path_factory, "boundaries.tif", stack_data, compression="zlib")
    array, _how = imageio.open_image_stack(path, eager_limit_bytes=1 << 40)
    np.testing.assert_array_equal(array, stack_data)


def test_unreadable_file_raises(tmp_path_factory):
    path = tmp_path_factory.mktemp("tif") / "broken.tif"
    path.write_bytes(b"not a tiff at all")
    with pytest.raises(Exception):
        imageio.open_image_stack(path)
