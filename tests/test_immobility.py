"""Did this molecule move at all? Asked without fitting anything.

A static emitter is a completely specified statistical object: every position
it reports is its true position plus localization error, and that error is
measured per spot by the same fit that produced the position. So the scatter of
a trajectory in units of its own precision is chi-squared with 2(N-1) degrees of
freedom under "this never moved" - exactly, at every trajectory length.

That is the same question a small diffusion coefficient is usually asked to
answer, but asked directly: one pass instead of a per-trajectory regression, at
its best on exactly the short trajectories where the MSD slope is at its worst,
and returning a probability instead of a number to be thresholded by eye.

The tests that matter here are the calibration ones. A test whose stated null
distribution is not its actual null distribution is worse than no test, because
its p-values look meaningful.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)
stats = pytest.importorskip("scipy.stats")

from test_render_widget import _pump_until  # noqa: E402
from test_widget_interaction import ensure_qapp  # noqa: E402

PIXEL_SIZE_NM = 100.0
FPS = 32.0
_WIDGETS = []


# --- the statistic itself -----------------------------------------------------


def test_a_trajectory_that_never_moved_scores_its_degrees_of_freedom():
    """T is a sum of 2(N-1) squared standard normals, so it averages 2(N-1)."""
    rng = np.random.default_rng(0)
    n, sigma = 12, 25.0
    totals = []
    for _ in range(4000):
        xy = rng.normal(0, sigma, (n, 2))
        T, dof = widget_mod.immobility_statistic(xy[:, 0], xy[:, 1],
                                                 np.full(n, sigma))
        assert dof == 2 * (n - 1)
        totals.append(T)
    assert np.mean(totals) == pytest.approx(2 * (n - 1), rel=0.03)


@pytest.mark.parametrize("n", [3, 5, 20, 60])
def test_the_null_holds_at_every_trajectory_length(n):
    """Exact, not asymptotic: the false-positive rate must be 5% at every N."""
    rng = np.random.default_rng(n)
    sigma = 25.0
    p = []
    for _ in range(4000):
        xy = rng.normal(0, sigma, (n, 2))
        T, dof = widget_mod.immobility_statistic(xy[:, 0], xy[:, 1], np.full(n, sigma))
        p.append(stats.chi2.sf(T, dof))
    assert (np.array(p) < 0.05).mean() == pytest.approx(0.05, abs=0.015)


def test_heterogeneous_precision_must_be_weighted_not_averaged():
    """Photon count varies several-fold between spots, so precision does too.

    Substituting one average sigma breaks the null - the residuals stop being
    identically distributed - and the test starts calling static molecules
    mobile at three times the rate it claims.
    """
    rng = np.random.default_rng(3)
    n = 30
    weighted, averaged = [], []
    for _ in range(4000):
        sigma = rng.uniform(15, 45, n)
        xy = rng.normal(0, 1, (n, 2)) * sigma[:, None]
        T, dof = widget_mod.immobility_statistic(xy[:, 0], xy[:, 1], sigma)
        weighted.append(stats.chi2.sf(T, dof))
        T2, dof2 = widget_mod.immobility_statistic(
            xy[:, 0], xy[:, 1], np.full(n, sigma.mean()))
        averaged.append(stats.chi2.sf(T2, dof2))
    assert (np.array(weighted) < 0.05).mean() == pytest.approx(0.05, abs=0.015)
    assert (np.array(averaged) < 0.05).mean() > 0.10


def test_a_moving_molecule_is_detected():
    rng = np.random.default_rng(5)
    n, sigma, step = 20, 25.0, 80.0
    caught = 0
    for _ in range(2000):
        true = np.cumsum(rng.normal(0, step, (n, 2)), axis=0)
        xy = true + rng.normal(0, sigma, (n, 2))
        T, dof = widget_mod.immobility_statistic(xy[:, 0], xy[:, 1], np.full(n, sigma))
        caught += stats.chi2.sf(T, dof) < 0.05
    assert caught / 2000 > 0.95


def test_the_statistic_does_not_care_what_unit_it_is_given():
    """Dimensionless by construction - pixels and nanometres must agree."""
    rng = np.random.default_rng(7)
    xy = rng.normal(0, 25.0, (15, 2))
    sigma = np.full(15, 25.0)
    T_nm, dof_nm = widget_mod.immobility_statistic(xy[:, 0], xy[:, 1], sigma)
    T_px, dof_px = widget_mod.immobility_statistic(
        xy[:, 0] / 100.0, xy[:, 1] / 100.0, sigma / 100.0)
    assert T_px == pytest.approx(T_nm)
    assert dof_px == dof_nm


def test_a_single_point_cannot_scatter():
    T, dof = widget_mod.immobility_statistic([1.0], [2.0], [25.0])
    assert dof == 0 and np.isnan(T)


def test_localizations_without_a_precision_are_dropped_not_guessed():
    """A missing uncertainty is not a licence to invent one."""
    x = [0.0, 1.0, 2.0, 3.0]
    y = [0.0, 0.0, 0.0, 0.0]
    sigma = [25.0, np.nan, 25.0, 0.0]        # one missing, one non-positive
    T, dof = widget_mod.immobility_statistic(x, y, sigma)
    assert dof == 2 * (2 - 1)                # two usable points remain


# --- through the widget -------------------------------------------------------


def _widget():
    from napari.components import ViewerModel

    ensure_qapp()
    widget = widget_mod.LocalizationTrackingWidget(ViewerModel())
    _WIDGETS.append(widget)
    widget.pixel_size_box.setValue(PIXEL_SIZE_NM)
    widget.fps_box.setValue(FPS)
    return widget


def _population(n_static=20, n_moving=20, n_points=15, step_nm=120.0,
                sigma_range=(15.0, 45.0), with_uncertainty=True, seed=0):
    """Localizations and their trajectories, half immobile and half diffusing."""
    rng = np.random.default_rng(seed)
    rows, tracks = [], []
    for pid in range(n_static + n_moving):
        step = 0.0 if pid < n_static else step_nm
        true = np.cumsum(rng.normal(0, step, (n_points, 2)), axis=0)
        true += [4000.0 * (pid % 8), 4000.0 * (pid // 8)]
        sigma = rng.uniform(*sigma_range, n_points)
        seen = true + rng.normal(0, 1, (n_points, 2)) * sigma[:, None]
        for frame, (point, s) in enumerate(zip(seen, sigma)):
            row = {"frame": frame, "x [nm]": point[0], "y [nm]": point[1],
                   "sigma [nm]": 150.0, "intensity [photon]": 900.0}
            if with_uncertainty:
                row["uncertainty [nm]"] = s
            rows.append(row)
            tracks.append({"particle": pid, "frame": frame,
                           "x": point[0] / PIXEL_SIZE_NM, "y": point[1] / PIXEL_SIZE_NM})
    return pd.DataFrame(rows), pd.DataFrame(tracks)


def _analysed(**kwargs):
    widget = _widget()
    locs, tracks = _population(**kwargs)
    widget._ingest_localization_dataframe(locs, "loaded", True)
    widget.tracks = tracks
    widget._invalidate_track_filter()
    widget._start_fit_free_metrics_worker()
    assert _pump_until(lambda: widget._track_distance_cache is not None), \
        "the metrics never finished"
    return widget


def test_the_precision_is_taken_per_spot_from_the_localization_table():
    widget = _analysed()
    label, sigma_px, measured = widget._sigma_source()
    assert measured is True
    assert "uncertainty [nm]" in label
    assert len(sigma_px) == len(widget.tracks)
    # nm converted to camera pixels, and genuinely varying spot to spot
    assert sigma_px.min() == pytest.approx(15.0 / PIXEL_SIZE_NM, abs=0.02)
    assert sigma_px.std() > 0


def test_a_table_without_uncertainties_falls_back_and_says_so():
    widget = _analysed(with_uncertainty=False)
    label, sigma_px, measured = widget._sigma_source()
    assert measured is False
    assert "no uncertainty column" in label
    assert np.allclose(sigma_px, widget.immobility_sigma_box.value() / PIXEL_SIZE_NM)
    assert "Add an uncertainty column" in widget.immobility_status_label.text()


def test_immobile_molecules_land_on_a_motion_ratio_of_one():
    """The calibration check: this is what says the precision is trustworthy."""
    widget = _analysed(n_static=200, n_moving=0)
    ratios = np.array([widget._track_motion_cache[pid] for pid in range(200)])
    assert np.median(ratios) == pytest.approx(1.0, abs=0.12)


def test_the_two_populations_separate():
    widget = _analysed(n_static=60, n_moving=60)
    p = widget._track_pstatic_cache
    static_called = sum(1 for pid in range(60) if p[pid] > 0.05)
    moving_called = sum(1 for pid in range(60, 120) if p[pid] < 0.05)
    assert static_called >= 54          # ~5% false positives expected
    assert moving_called >= 54


def test_the_p_value_is_floored_so_a_log_axis_stays_readable():
    """An obviously mobile molecule returns something like 1e-200, and a log
    axis running that far spends every decade but the last on nothing."""
    widget = _analysed(n_static=0, n_moving=20, step_nm=2000.0)
    assert min(widget._track_pstatic_cache.values()) >= widget_mod.P_STATIC_FLOOR


def test_the_calibration_factor_scales_the_reported_precision():
    """The one assumption from outside the test, and the knob that fixes it."""
    widget = _analysed(n_static=100, n_moving=0)
    before = np.median(list(widget._track_motion_cache.values()))

    widget.immobility_calibration_box.setValue(2.0)
    assert _pump_until(
        lambda: np.median(list(widget._track_motion_cache.values())) < before * 0.5)
    after = np.median(list(widget._track_motion_cache.values()))
    # sigma doubled, and the statistic divides by sigma squared
    assert after == pytest.approx(before / 4.0, rel=0.05)


def test_both_metrics_reach_the_histograms_bounds_and_colouring():
    widget = _analysed()
    for key, choice in (("motion", "Motion ratio (moved vs its own precision)"),
                        ("pstatic", "p (consistent with static)")):
        assert key in widget._metric_hist_widgets
        assert key in widget._metric_bound_boxes
        assert key in widget._metric_filter_boxes
        widget.color_metric_box.setCurrentText(choice)
        assert widget._current_metric_key() == key


def test_both_reach_the_exported_metrics_table():
    widget = _analysed()
    frame = widget._track_metrics_frame()
    assert "motion_ratio" in frame.columns
    assert "p_static" in frame.columns
    assert frame["motion_ratio"].notna().all()


# --- the point of it: the structural half of the acquisition -------------------


def test_filtering_to_p_above_five_percent_renders_the_immobile_population():
    """Structural PALM is the p > 0.05 subset of a live-cell acquisition, and
    sptPALM is the rest. This is that cut, made once, on one dataset."""
    widget = _analysed(n_static=40, n_moving=40)
    everything = widget._render_positions_px()[0].size

    widget.pstatic_min_box.setValue(0.05)
    widget.pstatic_max_box.setValue(1.0)
    widget.pstatic_filter_box.setChecked(True)

    immobile = widget._render_positions_px()[0].size
    assert immobile == pytest.approx(everything / 2, rel=0.2)
    kept = set(widget._displayed_tracks()["particle"])
    assert len(kept & set(range(40))) >= 36        # nearly all the static ones
    assert len(kept & set(range(40, 80))) <= 4     # and almost no moving ones


def test_the_complementary_cut_renders_the_mobile_population():
    widget = _analysed(n_static=40, n_moving=40)
    widget.pstatic_min_box.setValue(0.0)
    widget.pstatic_max_box.setValue(0.05)
    widget.pstatic_filter_box.setChecked(True)

    kept = set(widget._displayed_tracks()["particle"])
    assert len(kept & set(range(40, 80))) >= 36
    assert len(kept & set(range(40))) <= 4


def test_filtering_on_the_motion_ratio_works_the_same_way():
    widget = _analysed(n_static=40, n_moving=40)
    widget.motion_min_box.setValue(0.0)
    widget.motion_max_box.setValue(1.5)
    widget.motion_filter_box.setChecked(True)
    kept = set(widget._displayed_tracks()["particle"])
    assert len(kept & set(range(40))) > len(kept & set(range(40, 80)))


def test_the_settings_survive_a_round_trip():
    widget = _analysed()
    widget.immobility_calibration_box.setValue(1.19)
    widget.immobility_sigma_box.setValue(31.0)
    widget.pstatic_min_box.setValue(0.05)
    widget.pstatic_filter_box.setChecked(True)
    metadata = widget._collect_metadata(None)

    assert metadata["immobility"]["precision_calibration"] == pytest.approx(1.19)
    assert "uncertainty" in metadata["immobility"]["precision_source"]
    assert metadata["dynamics_filter"]["pstatic"] is True

    restored = _widget()
    restored.apply_settings(metadata)
    assert restored.immobility_calibration_box.value() == pytest.approx(1.19)
    assert restored.immobility_sigma_box.value() == pytest.approx(31.0)
    assert restored.pstatic_min_box.value() == pytest.approx(0.05)
    assert restored.pstatic_filter_box.isChecked()


# --- what the trajectory could have seen ---------------------------------------
#
# "Not significantly moving" is not "static". The false-positive rate of the
# test has no length bias at all - 5% at N = 3 and at N = 100 - but its *power*
# is dominated by length: at D = 0.005 um2/s a genuinely moving molecule is
# missed 87% of the time at N = 3 and 1% of the time at N = 30. Tightening the
# threshold does not help; it raises the floor for every length by about the
# same factor. The only honest fix is to report the floor.


@pytest.mark.parametrize("n,expected", [
    (3, 0.0288), (5, 0.0120), (8, 0.0056), (15, 0.0021), (30, 0.0007), (100, 0.00012),
])
def test_the_floor_matches_the_power_actually_achieved(n, expected):
    """Checked against a bisection on simulated trajectories: the D detected in
    half of them, at sigma = 25 nm and a 31.3 ms frame interval."""
    floor = widget_mod.detectable_diffusion(n, 25.0, 0.0313, 0.05) / 1e6
    assert floor == pytest.approx(expected, rel=0.15)


def test_the_floor_falls_steeply_with_trajectory_length():
    """Roughly forty-fold from three points to thirty - which is the whole
    reason a bare 'immobile' verdict is misleading on short tracks."""
    short = widget_mod.detectable_diffusion(3, 25.0, 0.0313, 0.05)
    long = widget_mod.detectable_diffusion(30, 25.0, 0.0313, 0.05)
    assert short / long == pytest.approx(33.0, rel=0.25)


def test_a_tighter_threshold_raises_the_floor_but_does_not_flatten_it():
    """The two effects are nearly orthogonal, which is why the threshold cannot
    be used to fix the length bias: tightening it raises every length together.

    Measured as the two changes side by side - how far the floor itself moves
    against how far the *ratio* between lengths moves.
    """
    floors = {}
    for alpha in (0.05, 0.001):
        floors[alpha] = {n: widget_mod.detectable_diffusion(n, 25.0, 0.0313, alpha)
                         for n in (3, 30)}
    floor_change = floors[0.001][30] / floors[0.05][30]
    ratio_change = ((floors[0.001][3] / floors[0.001][30])
                    / (floors[0.05][3] / floors[0.05][30]))
    assert floor_change > 2.0          # the floor really does move
    assert ratio_change < floor_change / 1.5   # the length dependence barely does


def test_a_worse_precision_raises_the_floor_as_the_square():
    a = widget_mod.detectable_diffusion(15, 20.0, 0.0313, 0.05)
    b = widget_mod.detectable_diffusion(15, 40.0, 0.0313, 0.05)
    assert b / a == pytest.approx(4.0, rel=0.01)


def test_two_points_have_no_rate_to_report():
    assert widget_mod.detectable_diffusion(2, 25.0, 0.0313, 0.05) == float("inf")
    assert widget_mod.detectable_diffusion(10, 0.0, 0.0313, 0.05) == float("inf")


def test_the_floor_is_reported_per_trajectory():
    widget = _analysed(n_static=20, n_moving=0)
    floors = widget._track_dmin_cache
    assert len(floors) == 20
    assert all(np.isfinite(v) and v > 0 for v in floors.values())
    # 15 points at 15-45 nm precision and ~32 fps lands in this range
    assert 1e-4 < np.median(list(floors.values())) < 1e-1


def test_a_shorter_trajectory_reports_a_higher_floor():
    """The comparison that makes the length bias legible in the data itself."""
    import pandas as pd

    widget = _widget()
    rng = np.random.default_rng(4)
    rows, tracks = [], []
    for pid, n in ((0, 5), (1, 40)):
        xy = rng.normal(0, 25.0, (n, 2)) + [2000.0 * pid, 2000.0]
        for frame, point in enumerate(xy):
            rows.append({"frame": frame, "x [nm]": point[0], "y [nm]": point[1],
                         "uncertainty [nm]": 25.0, "sigma [nm]": 150.0,
                         "intensity [photon]": 900.0})
            tracks.append({"particle": pid, "frame": frame,
                           "x": point[0] / PIXEL_SIZE_NM, "y": point[1] / PIXEL_SIZE_NM})
    widget._ingest_localization_dataframe(pd.DataFrame(rows), "loaded", True)
    widget.tracks = pd.DataFrame(tracks)
    widget._invalidate_track_filter()
    widget._start_fit_free_metrics_worker()
    assert _pump_until(lambda: widget._track_dmin_cache)

    assert widget._track_dmin_cache[0] > 10 * widget._track_dmin_cache[1]


def test_the_significance_box_moves_the_floor():
    widget = _analysed(n_static=20, n_moving=0)
    loose = np.median(list(widget._track_dmin_cache.values()))
    widget.immobility_alpha_box.setValue(0.001)
    assert _pump_until(
        lambda: np.median(list(widget._track_dmin_cache.values())) > loose * 1.2)
    assert np.median(list(widget._track_dmin_cache.values())) > loose


def test_it_is_a_metric_like_the_others():
    widget = _analysed()
    assert "dmin" in widget._metric_hist_widgets
    assert "dmin" in widget._metric_bound_boxes
    assert "dmin" in widget._metric_filter_boxes
    widget.color_metric_box.setCurrentText("Smallest detectable D")
    assert widget._current_metric_key() == "dmin"


def test_it_can_select_the_trajectories_with_enough_power():
    """The point of having it as a filter: keep only trajectories that could
    have ruled out the D you care about."""
    widget = _analysed(n_static=30, n_moving=30)
    widget.dmin_min_box.setValue(0.0)
    widget.dmin_max_box.setValue(np.median(list(widget._track_dmin_cache.values())))
    widget.dmin_filter_box.setChecked(True)
    kept = widget._displayed_tracks()["particle"].nunique()
    assert 0 < kept < 60


def test_it_reaches_the_metrics_table():
    widget = _analysed()
    frame = widget._track_metrics_frame()
    assert "d_detectable_um2_per_s" in frame.columns
    assert frame["d_detectable_um2_per_s"].notna().all()


def test_the_panel_says_what_immobile_is_worth():
    widget = _analysed(n_static=20, n_moving=0)
    text = widget.immobility_status_label.text()
    assert "Detection floor" in text
    assert "too short or too imprecise" in text
