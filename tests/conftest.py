"""Shared helpers for the localization tests."""
import importlib.util
import sys
import types
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1] / "napari_loc_track"
_MODULE_NAME = "napari_loc_track._localize2d"
_RENDER_MODULE_NAME = "napari_loc_track._render"
_ACQMETA_MODULE_NAME = "napari_loc_track._acqmeta"


def _load_standalone(dotted_name, filename):
    """Import one module of the package without running its __init__."""
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    if "napari_loc_track" not in sys.modules:
        pkg = types.ModuleType("napari_loc_track")
        pkg.__path__ = [str(PKG_DIR)]
        sys.modules["napari_loc_track"] = pkg
    spec = importlib.util.spec_from_file_location(dotted_name, PKG_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = module
    spec.loader.exec_module(module)
    return module


def load_render():
    """Load `_render` without importing the napari/Qt/trackpy stack.

    Same reasoning as `load_localize2d`: the renderer is plain numpy/numba and
    has no intra-package imports, so the arithmetic can be tested on a machine
    that has no working napari at all.
    """
    return _load_standalone(_RENDER_MODULE_NAME, "_render.py")


def load_localize2d():
    """Load `_localize2d` without importing the napari/Qt/trackpy stack.

    `napari_loc_track/__init__` pulls in napari and trackpy, which are slow and
    irrelevant to the arithmetic under test; `_localize2d` has no intra-package
    imports, so loading it from its file is equivalent. It is still registered
    under its real dotted name because numba's on-disk cache pickles the
    defining module name - an alias would poison the cache the plugin uses.
    """
    return _load_standalone(_MODULE_NAME, "_localize2d.py")


def load_acqmeta():
    """Load `_acqmeta` without importing the napari/Qt/trackpy stack.

    The reader is plain text and JSON handling and imports tifffile only inside
    the one function that needs it, so the parsing can be tested anywhere.
    """
    return _load_standalone(_ACQMETA_MODULE_NAME, "_acqmeta.py")
