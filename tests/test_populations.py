"""Populations of trajectories, fitted to all of them at once.

The static test can only say of a short trajectory that it could be standing
still. The population fit adds what the test lacks - how large the mobile
molecules' steps are - and turns every trajectory, a two-point one too, into a
probability of being immobile. What has to hold: the likelihood is exact; the
fit finds the populations a simulation was built from; its probabilities mean
what they say (of the trajectories given 0.7, about 70% are immobile); a second
mobile population is only kept when there is one; and softly sorted renders add
up to the whole.
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

_PKG_DIR = Path(__file__).resolve().parents[1] / "napari_loc_track"
if "napari_loc_track" not in sys.modules:
    _pkg = types.ModuleType("napari_loc_track")
    _pkg.__path__ = [str(_PKG_DIR)]
    sys.modules["napari_loc_track"] = _pkg
_spec = importlib.util.spec_from_file_location(
    "napari_loc_track._populations", _PKG_DIR / "_populations.py")
populations = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("napari_loc_track._populations", populations)
_spec.loader.exec_module(populations)

TAU = 0.03


def _simulate(n_traj=6000, mixture=((0.4, 0.0), (0.6, 0.1)), mean_extra_points=2.3,
              sub=30, seed=0):
    """Trajectories from known populations: (fraction, D in µm²/s) each.

    Positions are averaged over a whole-frame exposure (motion blur), then
    given each localization's own error, 15-40 nm.
    """
    rng = np.random.default_rng(seed)
    fractions = np.array([f for f, _d in mixture])
    rows, truth = [], []
    for pid in range(n_traj):
        k = int(rng.choice(len(mixture), p=fractions / fractions.sum()))
        d_nm2 = mixture[k][1] * 1e6
        n = 1 + rng.geometric(1.0 / mean_extra_points)
        steps = rng.normal(0.0, np.sqrt(2 * d_nm2 * TAU / sub), (n * sub, 2))
        true = np.cumsum(steps, axis=0).reshape(n, sub, 2).mean(axis=1)
        sigma = rng.uniform(15.0, 40.0, n)
        seen = true + rng.normal(0.0, 1.0, (n, 2)) * sigma[:, None]
        for f in range(n):
            rows.append((pid, f, seen[f, 0], seen[f, 1], sigma[f]))
        truth.append(k)
    table = np.array(rows)
    traj = populations.build_trajectories(
        table[:, 0].astype(int), table[:, 1].astype(int), table[:, 2], table[:, 3], table[:, 4])
    return traj, np.array(truth)[traj.pid.astype(int)]


# --- the likelihood -------------------------------------------------------------------


def test_the_likelihood_is_the_exact_gaussian_of_the_increments():
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(1)
    frames = np.array([0, 1, 2, 4, 5, 6])        # a gap of one frame
    sigma = rng.uniform(15, 40, 6)
    x, y = rng.normal(0, 50, 6), rng.normal(0, 50, 6)
    traj = populations.build_trajectories(np.zeros(6, int), frames, x, y, sigma)
    d, blur = 0.05, 1.0 / 6.0
    fast = populations.log_likelihoods(traj, [d], TAU, blur)[0, 0]

    gaps = np.diff(frames)
    cov = np.zeros((5, 5))
    for j in range(5):
        cov[j, j] = 2 * d * 1e6 * (gaps[j] * TAU - 2 * blur * TAU) + sigma[j] ** 2 + sigma[j + 1] ** 2
        if j < 4:
            cov[j, j + 1] = cov[j + 1, j] = 2 * d * 1e6 * blur * TAU - sigma[j + 1] ** 2
    dense = (multivariate_normal(np.zeros(5), cov).logpdf(np.diff(x))
             + multivariate_normal(np.zeros(5), cov).logpdf(np.diff(y)))
    assert fast == pytest.approx(dense, abs=1e-9)


def test_trajectories_of_every_length_are_computed_together():
    """Sorted longest first, the pass works on slices; each trajectory's value
    must be the one it would get alone."""
    traj, _truth = _simulate(n_traj=50, seed=3)
    together = populations.log_likelihoods(traj, [0.0, 0.1], TAU)
    for i in (0, len(traj) // 2, len(traj) - 1):
        start, n = traj.offsets[i], traj.n_points[i] - 1
        alone = populations.Trajectories(
            pid=traj.pid[i:i + 1], n_points=traj.n_points[i:i + 1], offsets=np.array([0]),
            counts=np.ones(n, dtype=np.int64),
            zx=traj.zx[start:start + n], zy=traj.zy[start:start + n],
            gap=traj.gap[start:start + n], s_diag=traj.s_diag[start:start + n],
            s_next=traj.s_next[start:start + n])
        np.testing.assert_allclose(populations.log_likelihoods(alone, [0.0, 0.1], TAU)[0],
                                   together[i])


def test_one_point_trajectories_are_left_out():
    traj = populations.build_trajectories(
        np.array([0, 1, 1]), np.array([0, 0, 1]), np.zeros(3), np.zeros(3), np.full(3, 20.0))
    assert list(traj.pid) == [1]


# --- the fit ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_populations():
    traj, truth = _simulate()
    return traj, truth, populations.fit_populations(traj, TAU, d_immobile_max=0.01)


def test_the_fit_finds_the_populations_it_was_built_from(two_populations):
    _traj, _truth, result = two_populations
    assert result["n_mobile"] == 1                 # BIC does not invent a second
    assert result["D"][0] <= 0.01
    assert result["D"][1] == pytest.approx(0.1, rel=0.1)
    assert result["fractions"][0] == pytest.approx(0.4, abs=0.03)


def test_the_probabilities_mean_what_they_say(two_populations):
    """Of the trajectories given P(immobile) between 0.5 and 0.9, the share
    that really is immobile is their mean probability."""
    _traj, truth, result = two_populations
    p = result["posterior"][:, 0]
    immobile = truth == 0
    for low, high in ((0.0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.01)):
        band = (p >= low) & (p < high)
        assert band.sum() > 100
        assert immobile[band].mean() == pytest.approx(p[band].mean(), abs=0.06)


def test_a_small_step_speaks_for_immobility_and_a_large_one_against(two_populations):
    _traj, _truth, result = two_populations
    fitted = {"D": result["D"], "fractions": result["fractions"]}
    probabilities = []
    for step in (10.0, 60.0, 250.0):
        one = populations.build_trajectories(np.zeros(2, int), np.array([0, 1]),
                                             np.array([0.0, step]), np.zeros(2),
                                             np.full(2, 25.0))
        ll = populations.log_likelihoods(one, fitted["D"], TAU)[0]
        joint = ll + np.log(fitted["fractions"])
        probabilities.append(np.exp(joint[0] - np.logaddexp.reduce(joint)))
    assert probabilities[0] > result["fractions"][0] > probabilities[2]
    assert probabilities[0] > probabilities[1] > probabilities[2]
    assert probabilities[2] < 0.01


def test_a_second_mobile_population_is_kept_when_there_is_one():
    traj, _truth = _simulate(mixture=((0.3, 0.0), (0.35, 0.03), (0.35, 1.0)),
                             mean_extra_points=6.0, n_traj=4000, seed=5)
    result = populations.fit_populations(traj, TAU, d_immobile_max=0.01)
    assert result["n_mobile"] == 2
    assert result["labels"] == ["immobile", "slow", "fast"]
    assert result["D"][1] == pytest.approx(0.03, rel=0.3)
    assert result["D"][2] == pytest.approx(1.0, rel=0.2)


def test_the_predicted_step_lengths_match_the_observed(two_populations):
    traj, _truth, result = two_populations
    centres, observed, predicted = populations.step_length_densities(
        traj, result, np.linspace(0, 400, 41))
    total = predicted.sum(axis=0)
    width = centres[1] - centres[0]
    assert (total * width).sum() == pytest.approx((observed * width).sum(), rel=0.05)
    assert np.abs(total - observed).max() < 0.15 * observed.max()


# --- through the widget ------------------------------------------------------------

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)

from test_render_populations import _analysed, _render  # noqa: E402
from test_render_widget import _pump_until  # noqa: E402


def _fitted(**kwargs):
    widget = _analysed(**kwargs)
    widget.fit_populations()
    assert _pump_until(lambda: widget._population_worker_ref is None), "fit never finished"
    assert widget._population_fit is not None
    return widget


def _by_fit(widget, soft=False, probability=0.9):
    widget.classify_method_box.setCurrentIndex(widget.classify_method_box.findData("fit"))
    widget.class_probability_box.setValue(probability)
    widget.class_soft_box.setChecked(soft)
    return widget


def test_the_fit_reports_its_populations():
    widget = _fitted()
    text = widget.population_fit_status.text()
    assert "Immobile: D =" in text and "Mobile: D =" in text
    result = widget._population_fit["result"]
    # the test data: half static, half at 0.02 µm²/s
    assert result["fractions"][0] == pytest.approx(0.5, abs=0.1)
    assert result["D"][-1] == pytest.approx(0.02, rel=0.4)


def test_classifying_by_the_fit_sorts_by_probability():
    widget = _by_fit(_fitted())
    classes = widget._trajectory_classes()
    p = widget._probability_immobile()
    assert all(p[pid] >= 0.9 for pid, cls in classes.items() if cls == "immobile")
    assert all(1 - p[pid] >= 0.9 for pid, cls in classes.items() if cls == "mobile")
    widget._set_render_population("immobile")
    # the probabilities decide alone: no p_static range left on
    assert widget._active_metric_filters() == []
    assert widget._passing_particles() == widget._class_members("immobile")


def test_soft_renders_add_up_to_the_whole():
    """Every localization lands in the immobile and the mobile image in
    proportion to its probability - nothing lost, nothing counted twice."""
    widget = _by_fit(_fitted(), soft=True)
    assert not widget.population_buttons["undetermined"].isEnabled()
    totals = {}
    for which in ("immobile", "mobile"):
        widget._set_render_population(which)
        options, info, _layer = widget._render_inputs()
        assert info["soft_population"]["population"] == which
        totals[which] = options["weights"].sum()
        _render(widget)
    n_fitted = len(widget._displayed_localizations())
    assert totals["immobile"] + totals["mobile"] == pytest.approx(n_fitted)
    rendered = sum(float(np.asarray(widget.viewer.layers[f"smlm_render_{w}"].data).sum())
                   for w in ("immobile", "mobile"))
    assert rendered == pytest.approx(n_fitted, rel=0.05)


def test_a_fit_on_other_terms_goes_stale_and_is_not_used():
    widget = _by_fit(_fitted())
    assert widget._classify_method() == "fit"
    widget.immobile_dmax_box.setValue(0.05)
    assert not widget._population_fit_current()
    assert widget._classify_method() == "test"
    widget._update_population_fit_status()
    assert "Stale" in widget.population_fit_status.text()


def test_merging_is_certified_by_the_static_test_whatever_the_sorting():
    widget = _by_fit(_fitted(), probability=0.5)
    widget.merge_box.setChecked(True)
    assert widget._merge_ids() == widget._class_members("immobile", method="test")


def test_the_trajectory_table_carries_each_probability():
    widget = _fitted()
    table = widget._track_metrics_frame()
    assert {"P_immobile", "P_mobile"} <= set(table.columns)
    np.testing.assert_allclose(table["P_immobile"] + table["P_mobile"], 1.0)


def test_how_to_classify_round_trips():
    widget = _by_fit(_fitted(), soft=True, probability=0.8)
    metadata = widget._collect_metadata(None)
    assert metadata["population_fit"]["result"]["labels"] == ["immobile", "mobile"]
    values, _notes = widget_mod.settings_from_metadata(metadata)
    assert values["classify_method_box"] == "fit"
    assert values["class_probability_box"] == pytest.approx(0.8)
    assert values["class_soft_box"] is True
