"""The growth drawn over the snapshots, in the snapshot viewer - in the first
snapshot's geometry."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib

import numpy as np
import pytest

conftest = importlib.import_module("conftest")
deform = conftest.load_deform()
growth_view = conftest._load_standalone("napari_loc_track._growth_view", "_growth_view.py")
test_deform = importlib.import_module("test_deform")


@pytest.fixture(scope="module")
def measured():
    stack, maps = test_deform._snapshots(K=4)
    region = (160, test_deform.SIZE - 160, 160, test_deform.SIZE - 160)
    record = deform.measure_deformation(stack, region=region, patch=128, step=64, lags=(1, 2),
                                        threads=2, backend="cpu")
    return record, stack


def _first(stack, record):
    x0, x1, y0, y1 = growth_view.first_region(record)
    return np.asarray(stack[0], float)[max(y0, 0):y1, max(x0, 0):x1]


def test_the_last_stage_carries_every_snapshot_onto_the_first(measured):
    record, stack = measured
    stages = growth_view.StageStack(stack, record)
    n_stages, n, h, w = stages.shape
    assert n == len(record.t) and n_stages == len(record.steps["forward"]) + 1
    x0, x1, y0, y1 = growth_view.first_region(record)
    assert min(x0, y0) >= 0                              # the synthetic region fits the frame
    first = _first(stack, record)
    inner = (slice(40, -40), slice(40, -40))

    def agreement(s, k):
        return np.corrcoef(stages[s, k][inner].ravel(), first[inner].ravel())[0, 1]

    assert agreement(n_stages - 1, n - 1) > 0.98            # the growth cancelled
    assert agreement(0, n - 1) < agreement(n_stages - 1, n - 1) - 0.1   # as recorded, not
    assert stages[n_stages - 1].shape == (n, h, w)       # napari's slicing: a stage at a time


def test_the_arrows_show_the_growth_so_far_ending_on_the_tissue(measured):
    record, _stack = measured
    n = len(record.t)
    shifts = np.zeros((n, 2))
    vectors, magnitude = growth_view.growth_arrows(record, shifts)
    first = vectors[vectors[:, 0, 0] == 0]
    last = vectors[vectors[:, 0, 0] == n - 1]
    assert np.abs(first[:, 1, 1:]).max() < 1e-6            # nothing has grown yet
    # the synthetic tissue grows along y: by the end, away from the centre along y
    x0, x1, y0, y1 = growth_view.first_region(record)
    rows = last[:, 0, 1] - (y0 + y1) / 2
    assert np.corrcoef(rows, last[:, 1, 1])[0, 1] > 0.9
    assert magnitude.max() > 5
    rate, speed = growth_view.growth_rate_arrows(record, shifts)
    assert len(rate) and (speed >= 0).all()


def test_the_model_is_a_grid_on_the_tissue_that_stretches(measured):
    record, _stack = measured
    n = len(record.t)
    lines = growth_view.deformation_grid(record, np.zeros((n, 2)))
    kinds = {kind for _k, kind, _p in lines}
    assert kinds == {"along", "across"}

    def extent(k):
        pts = np.vstack([p for kk, kind, p in lines if kk == k and kind == "along"])
        return np.ptp(pts[:, 0])                              # rows: the axis is along y

    assert extent(n - 1) > 1.03 * extent(0)
    maps, _origin, _spacing = growth_view.stretch_maps(record)
    assert np.nanmedian(maps[0]) == pytest.approx(0.0, abs=1e-6)
    assert np.nanmedian(maps[-1]) > 4.0


def test_the_patches_are_drawn_where_they_were_measured(measured):
    record, _stack = measured
    outlines, along = growth_view.patch_outlines(record)
    assert len(outlines) == len(record.steps["centres"])
    assert all(o.shape == (4, 2) for o in outlines)
    assert np.all((along >= 0) & (along <= 1))
    text = growth_view.describe_measurement(record, 80.0)
    assert "Patches: squares of 128 px" in text and "Pairs:" in text


def test_the_growth_numbers(measured):
    record, _stack = measured
    m = growth_view.growth_metrics(record, px_nm=80.0)
    assert m["stretch_along"][0] == pytest.approx(0.0, abs=1e-6)
    assert m["stretch_along"][-1] > 4.0
    assert (m["rate_along"] > 0).all()
    assert len(m["passes_nm"]) == len(record.stats["passes"])
    # the synthetic rate rises along y: more stretch further along
    assert m["profile_stretch"][-1] > m["profile_stretch"][0]


def test_the_snapshot_viewer_gets_the_growth(measured):
    widget_mod = pytest.importorskip("napari_loc_track.widget")
    from napari.components import ViewerModel
    from test_widget_interaction import ensure_qapp

    record, stack = measured
    docks = []

    class _Window:
        def add_dock_widget(self, widget, name=None, area=None):
            docks.append(name)

    viewer = ViewerModel()
    object.__setattr__(viewer, "window", _Window())
    ensure_qapp()
    widget = widget_mod.LocalizationTrackingWidget(ViewerModel())
    widget._add_growth_layers(viewer, record, stack, np.zeros((len(record.t), 2)), 80.0,
                              (0.0, 10000.0))
    names = [layer.name for layer in viewer.layers]
    for wanted in ("first snapshot", "growth cancelled, by measurement stage",
                   "deformation model (grid on the tissue)", "measurement patches"):
        assert wanted in names
    assert any(name.startswith("growth since the first snapshot") for name in names)
    assert viewer.dims.axis_labels == ("stage", "snapshot", "y", "x")
    assert docks == ["growth"]
