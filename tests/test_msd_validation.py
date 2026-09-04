"""The MSD validation plot: what it is drawn on, and what it admits to.

The plot is read for the *slope* of MSD against lag time - 1 for free
diffusion, flatter for confined, steeper for directed - which is a straight line
only on log-log axes. The fit behind it stays on the raw values, because
minimising relative residuals instead would hand the short lags, the noisiest
and the ones most polluted by localization error, far more weight than they have
earned. So the scale is a display choice and nothing but.

The error beside each D says how well that one trajectory pins its slope down.
It is deliberately the plain least-squares standard error, and it understates
the truth - MSD points at different lags share displacements and are strongly
correlated - so what these tests hold it to is being *comparable between
trajectories*, not being a confidence interval.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

widget_mod = pytest.importorskip(
    "napari_loc_track.widget", reason="needs the napari/Qt/trackpy stack"
)
fit_msd_slope = widget_mod.fit_msd_slope

from test_widget_interaction import make_widget  # noqa: E402
from test_render_widget import _pump_until  # noqa: E402

PIXEL_SIZE_NM = 100.0
FPS = 20.0


# --- the fit ------------------------------------------------------------------


def test_a_clean_line_is_recovered_exactly():
    tau = np.arange(1, 11) * 0.05
    slope, intercept, error = fit_msd_slope(tau, 2.0 * tau + 0.1)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(0.1)
    assert error == pytest.approx(0.0, abs=1e-9)


def test_the_fit_is_made_on_the_raw_values_not_on_their_logarithms():
    """The log-log axes are a display choice; fitting in log space is not.

    A fit that minimised log residuals would be pulled towards the short lags,
    which is where localization error dominates - so the recovered slope would
    no longer be 4D.
    """
    tau = np.arange(1, 21) * 0.05
    truth = 4.0 * 0.3 * tau + 0.02          # D = 0.3 µm²/s, positive offset
    slope, intercept, _error = fit_msd_slope(tau, truth)
    assert slope / 4.0 == pytest.approx(0.3)
    assert intercept == pytest.approx(0.02)

    log_slope = np.polyfit(np.log(tau), np.log(truth), 1)[0]
    assert log_slope != pytest.approx(slope)  # the two really do differ


def test_a_noisier_trajectory_admits_a_larger_error():
    rng = np.random.default_rng(0)
    tau = np.arange(1, 21) * 0.05
    base = 2.0 * tau + 0.05
    _s, _i, tight = fit_msd_slope(tau, base + rng.normal(0, 0.002, tau.size))
    _s, _i, loose = fit_msd_slope(tau, base + rng.normal(0, 0.05, tau.size))
    assert tight < loose


@pytest.mark.filterwarnings("ignore:Polyfit may be poorly conditioned")
@pytest.mark.parametrize("tau,msd", [
    # two points: exactly determined, so there is nothing left to estimate from
    (np.array([0.05, 0.10]), np.array([0.2, 0.3])),
    # every lag the same: the fit is singular, there is no slope to put an error on
    (np.array([0.05, 0.05, 0.05, 0.05]), np.array([0.1, 0.2, 0.3, 0.4])),
])
def test_an_error_that_cannot_be_computed_is_not_invented(tau, msd):
    """Whether numpy refuses depends on its version, so the contract is what
    this holds: no error is ever fabricated where none can be had."""
    slope, intercept, error = fit_msd_slope(tau, msd)
    assert np.isfinite(slope) or np.isnan(slope)  # it must not raise
    assert np.isnan(error)


# --- the label ----------------------------------------------------------------


def test_the_error_on_D_is_a_quarter_of_the_error_on_the_slope():
    """MSD = 4D*tau, so D and its error are both a quarter of the slope's."""
    label = widget_mod.LocalizationTrackingWidget._msd_label(7, 0.5, 0.08)
    assert "#7" in label and "0.5" in label
    assert "0.02" in label                  # 0.08 / 4


def test_an_undefined_error_is_left_out_rather_than_shown_as_zero():
    label = widget_mod.LocalizationTrackingWidget._msd_label(3, 0.5, float("nan"))
    assert "#3" in label and "0.5" in label
    assert "±" not in label and "nan" not in label


# --- the plot -----------------------------------------------------------------


def _with_diffusion(n_particles=4, n_points=100, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for pid, d_true in enumerate(np.geomspace(0.02, 1.0, n_particles)):
        step_um = np.sqrt(2.0 * d_true / FPS)
        xy = np.cumsum(rng.normal(0, step_um, size=(n_points, 2)), axis=0) + 5.0
        for frame, (x, y) in enumerate(xy):
            rows.append({"particle": pid, "frame": frame,
                         "x": x * 1000.0 / PIXEL_SIZE_NM,
                         "y": y * 1000.0 / PIXEL_SIZE_NM})
    widget = make_widget()
    widget.pixel_size_box.setValue(PIXEL_SIZE_NM)
    widget.fps_box.setValue(FPS)
    widget.tracks = pd.DataFrame(rows)
    widget.compute_d()
    assert _pump_until(lambda: getattr(widget, "_compute_d_worker_ref", None) is None), \
        "the D computation never finished"
    return widget


def test_the_validation_plot_is_drawn_on_log_log_axes():
    widget = _with_diffusion()
    axes = widget.msd_figure.axes[0]
    assert axes.get_xscale() == "log"
    assert axes.get_yscale() == "log"


def test_every_legend_entry_carries_its_own_error():
    widget = _with_diffusion()
    entries = [t.get_text() for t in widget.msd_figure.axes[0].get_legend().get_texts()]
    assert entries, "nothing was plotted"
    for entry in entries:
        assert "D=" in entry and "µm²/s" in entry
        assert "±" in entry, entry


def test_the_cache_carries_the_error_beside_the_fit():
    widget = _with_diffusion()
    assert widget._track_msd_cache
    for tau, msd, slope, intercept, error in widget._track_msd_cache.values():
        assert len(tau) == len(msd)
        assert np.isfinite(slope) and np.isfinite(error)
        assert error >= 0.0


def test_a_zero_msd_point_does_not_break_the_log_plot():
    """A lag nothing moved over gives an MSD of exactly 0, which log cannot show.

    Left to matplotlib these vanish silently; masking them keeps the drawn
    points and the plotted fit describing the same set of data.
    """
    widget = _with_diffusion(n_particles=2)
    pid = next(iter(widget._track_msd_cache))
    tau, msd, slope, intercept, error = widget._track_msd_cache[pid]
    spoiled = msd.copy()
    spoiled[0] = 0.0
    spoiled[1] = -1e-9
    widget._track_msd_cache[pid] = (tau, spoiled, slope, intercept, error)

    widget._draw_msd_validation()           # must not raise
    axes = widget.msd_figure.axes[0]
    assert axes.get_yscale() == "log"
    drawn = np.concatenate([line.get_ydata() for line in axes.lines])
    assert (drawn > 0).all(), "a non-positive value reached a log axis"


# --- the other half of the fit -------------------------------------------------
#
# MSD = 4*D*tau + 4*sigma^2. The slope was being reported and the intercept
# thrown away, even though it is a second estimate of the localization precision
# by a completely different route from the spot fit - which makes the two
# together a calibration with no free parameters.


def test_the_intercept_recovers_the_localization_precision():
    """4*sigma^2 is the intercept, so sqrt(intercept)/2 is sigma."""
    sigma_um = 0.025                      # 25 nm
    assert widget_mod.msd_sigma_nm(4 * sigma_um ** 2) == pytest.approx(25.0)


def test_a_negative_intercept_reports_nothing_rather_than_a_number():
    """Motion blur subtracts from the intercept and can take it below zero.
    A fast molecule has no precision to report here, and inventing one - by
    taking the root of a negative, or clamping to zero - would be worse."""
    assert np.isnan(widget_mod.msd_sigma_nm(-0.001))
    assert np.isnan(widget_mod.msd_sigma_nm(0.0))
    assert np.isnan(widget_mod.msd_sigma_nm(float("nan")))


def test_the_precision_from_the_intercept_is_reported_per_trajectory():
    import pandas as pd

    widget = make_widget()
    widget.pixel_size_box.setValue(100.0)
    widget.tracks = pd.DataFrame({
        "particle": [0] * 6 + [1] * 6,
        "frame": list(range(6)) * 2,
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] * 2,
        "y": [0.0] * 12,
    })
    widget._track_diffusion_cache = {0: 0.5, 1: 0.5}
    widget._track_msd_cache = {
        0: (None, None, 2.0, 4 * 0.025 ** 2, 0.1),      # sigma = 25 nm
        1: (None, None, 2.0, -0.004, 0.1),              # blurred past zero
    }
    sigmas = widget._msd_sigma_map()
    assert sigmas[0] == pytest.approx(25.0)
    assert np.isnan(sigmas[1])

    frame = widget._track_metrics_frame()
    assert "sigma_from_msd_nm" in frame.columns
    assert frame.loc[frame["particle"] == 0, "sigma_from_msd_nm"].iloc[0] == \
        pytest.approx(25.0)


def test_the_two_precisions_are_reported_against_each_other():
    """The point of surfacing it: the spot fit and the MSD intercept measure the
    same thing, so their ratio is the calibration the immobility test needs."""
    import pandas as pd

    widget = make_widget()
    widget.pixel_size_box.setValue(100.0)
    widget.tracks = pd.DataFrame({
        "particle": [0] * 6, "frame": list(range(6)),
        "x": [0.0] * 6, "y": [0.0] * 6,
    })
    widget._track_msd_cache = {0: (None, None, 2.0, 4 * 0.030 ** 2, 0.1)}
    widget._update_msd_sigma_label()
    text = widget.msd_sigma_label.text()
    assert "30.0 nm" in text
    assert "Ratio" in text


def test_it_says_so_before_d_has_been_computed():
    widget = make_widget()
    widget._update_msd_sigma_label()
    assert "Compute D" in widget.msd_sigma_label.text()
