import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.measures.calibration import bucket_calibration
from src.models.stage4_persistence import (
    acf_pacf_table,
    adf_test,
    ar_structural_break_test,
    difference_series,
    fit_ar_persistence,
    period_bias_series,
    residual_diagnostics,
    run_stage4_persistence,
    select_ar_order,
    sup_wald,
    supwald_null,
)


def _ar1_monthly_series(n_months: int = 96, phi: float = 0.6, seed: int = 0) -> pd.Series:
    """Pure synthetic AR(1): x_t = phi * x_{t-1} + eps_t, eps ~ N(0, 1)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n_months)
    x = np.zeros(n_months)
    for t in range(1, n_months):
        x[t] = phi * x[t - 1] + eps[t]
    idx = pd.date_range("2016-01-01", periods=n_months, freq="MS")
    return pd.Series(x, index=idx)


def _trend_monthly_series(n_months: int = 48, slope: float = 0.15, noise_sd: float = 0.5,
                          seed: int = 7) -> pd.Series:
    """Deterministic linear trend + i.i.d. noise -- a genuinely non-stationary series."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    x = slope * t + rng.normal(0, noise_sd, n_months)
    idx = pd.date_range("2016-01-01", periods=n_months, freq="MS")
    return pd.Series(x, index=idx)


def _ar1_daily_series(n_days: int = 60, phi: float = 0.6, seed: int = 0) -> pd.Series:
    """Same AR(1) construction as `_ar1_monthly_series`, but one point per calendar day --
    for exercising `freq="D"` aggregation against a recurring daily-cadence series
    like Kalshi's KXTRUMPAPPROVE."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n_days)
    x = np.zeros(n_days)
    for t in range(1, n_days):
        x[t] = phi * x[t - 1] + eps[t]
    idx = pd.date_range("2026-01-01", periods=n_days, freq="D")
    return pd.Series(x, index=idx)


def _panel_from_bias_series(bias: pd.Series, n_markets: int = 50, price: float = 0.5,
                            offset: pd.Timedelta = pd.Timedelta(days=15)) -> pd.DataFrame:
    """Build a per-market panel whose aggregation (at the matching freq) reproduces `bias`
    exactly: fix implied_prob at `price` for every market, and choose how many resolve
    YES in each period so that mean_outcome = price - bias[period] precisely. `offset` is
    added to each period's start so close_time lands inside that period's bucket --
    pass a sub-day offset (e.g. `pd.Timedelta(hours=12)`) when `bias`'s index is daily."""
    rows = []
    for period_start, b in bias.items():
        mean_outcome = np.clip(price - b, 0.0, 1.0)
        n_yes = int(round(mean_outcome * n_markets))
        for i in range(n_markets):
            rows.append(dict(
                platform="kalshi", market_id=f"{period_start.strftime('%Y%m%d')}-{i}", title="t",
                category="c", close_time=period_start + offset,
                outcome=1.0 if i < n_yes else 0.0, implied_prob=price, volume=100,
            ))
    return pd.DataFrame(rows)


def test_monthly_bias_series_aggregates_correctly():
    # scaled down: bias = price - mean_outcome is bounded to [-price, 1-price] (+-0.5 here),
    # so an unscaled AR(1) (std ~1.15 at phi=0.5) would clip and silently corrupt this test.
    bias = _ar1_monthly_series(n_months=12, phi=0.5, seed=1) * 0.05
    panel = _panel_from_bias_series(bias, n_markets=50, price=0.5)
    out = period_bias_series(panel, freq="M", min_markets_per_period=10)

    assert len(out) == 12
    assert (out["n_markets"] == 50).all()
    assert not out["flag_small_n"].any()
    # reconstructed bias should be close to the planted value (rounding to whole markets)
    assert np.allclose(out["bias"].to_numpy(), bias.to_numpy(), atol=1.0 / 50)


def test_monthly_bias_series_flags_small_months():
    bias = _ar1_monthly_series(n_months=6, phi=0.5, seed=2)
    panel = _panel_from_bias_series(bias, n_markets=50)
    # shrink one month to below the threshold
    first_month = panel["close_time"].min().to_period("M").to_timestamp()
    mask = panel["close_time"].dt.to_period("M").dt.to_timestamp() == first_month
    panel = pd.concat([panel[mask].iloc[:2], panel[~mask]], ignore_index=True)

    out = period_bias_series(panel, freq="M", min_markets_per_period=10)
    assert out.loc[out["period"] == first_month, "flag_small_n"].iloc[0]
    assert out["flag_small_n"].sum() == 1


def test_monthly_bias_series_sign_matches_calibration_convention():
    # same panel read two ways: bucket_calibration's price-axis bias and
    # period_bias_series's time-axis bias should agree on sign/definition
    # (price - outcome) when computed over the whole sample.
    bias = _ar1_monthly_series(n_months=12, phi=0.3, seed=3)
    panel = _panel_from_bias_series(bias, n_markets=50, price=0.7)
    panel_renamed = panel.rename(columns={"implied_prob": "implied_prob"})

    bucket = bucket_calibration(panel_renamed, n_buckets=2)
    monthly = period_bias_series(panel, freq="M", min_markets_per_period=10)

    overall_bucket_bias = (bucket["mean_price"] - bucket["mean_outcome"]).mean()
    overall_monthly_bias = monthly["bias"].mean()
    assert np.sign(overall_bucket_bias) == np.sign(overall_monthly_bias) or abs(overall_monthly_bias) < 1e-9


def test_select_ar_order_recovers_ar1():
    x = _ar1_monthly_series(n_months=96, phi=0.6, seed=0)
    assert select_ar_order(x, max_p=3) == 1


def test_fit_ar_persistence_recovers_known_phi():
    x = _ar1_monthly_series(n_months=96, phi=0.6, seed=0)
    fit = fit_ar_persistence(x, p=1)
    assert abs(fit["coefs"]["lag1"] - 0.6) < 0.2


def test_adf_flags_stationary_ar1_series():
    x = _ar1_monthly_series(n_months=96, phi=0.6, seed=0)
    assert adf_test(x)["stationary"] is True


def test_adf_flags_nonstationary_trend_series():
    x = _trend_monthly_series(n_months=48, slope=0.15, noise_sd=0.5, seed=7)
    assert adf_test(x)["stationary"] is False


def test_differencing_resolves_trend_stationarity():
    x = _trend_monthly_series(n_months=48, slope=0.15, noise_sd=0.5, seed=7)
    diffed = difference_series(x)
    assert len(diffed) == len(x) - 1
    assert adf_test(diffed)["stationary"] is True


def test_differenced_residuals_are_white():
    x = _trend_monthly_series(n_months=48, slope=0.15, noise_sd=0.5, seed=7)
    diffed = difference_series(x)
    p = select_ar_order(diffed, max_p=3)
    fit = fit_ar_persistence(diffed, p)
    diag = residual_diagnostics(fit["resid"])
    assert diag["residual_white_noise"] is True


def test_orchestrator_triggers_and_resolves_differencing_on_trend_panel():
    # bias is bounded to [-0.5, 0.5] at price=0.5 -- the original slope=0.15/noise_sd=0.5
    # fixture ranges up to ~7.3 and would clip flat, which (confirmed empirically) corrupts
    # the ADF read. Same slope:noise ratio (0.3), scaled down to fit the representable range.
    bias = _trend_monthly_series(n_months=48, slope=0.009, noise_sd=0.03, seed=7)
    panel = _panel_from_bias_series(bias, n_markets=200, price=0.5)
    result = run_stage4_persistence(panel, freq="M", min_markets_per_period=10, max_p=3, min_periods=24)

    assert result.level["adf"]["stationary"] is False
    assert result.differencing_applied is True
    assert result.differenced is not None
    assert result.differenced["adf"]["stationary"] is True
    assert result.differenced["diagnostics"]["residual_white_noise"] is True


def test_orchestrator_does_not_difference_stable_ar1_panel():
    # unscaled phi=0.6 ranges to +-3 and would clip flat against bias's +-0.5 bound, which
    # (confirmed empirically) made this test pass for the wrong reason -- an artificially
    # saturated series, not a faithfully reconstructed stable AR(1). Scaled down to fit.
    # seed=0 was tried first and is a genuine ADF false negative, not a bug: at n=48,
    # phi=0.6 is only moderately far from a unit root, and ADF fails to reject (p=0.88)
    # on ~26% of seeds checked (13/50) -- a real, worth-knowing illustration of ADF's
    # known low power at moderate sample sizes, not something to paper over. seed=3 is a
    # robust instance (p=0.004), not a cherry-picked lucky one -- confirmed against the
    # same 50-seed scan used to rule out seed=0.
    bias = _ar1_monthly_series(n_months=48, phi=0.6, seed=3) * 0.1
    panel = _panel_from_bias_series(bias, n_markets=200, price=0.5)
    result = run_stage4_persistence(panel, freq="M", min_markets_per_period=10, max_p=3, min_periods=24)

    assert result.level["adf"]["stationary"] is True
    assert result.differencing_applied is False
    assert result.differenced is None


def test_raises_on_too_few_months():
    bias = _ar1_monthly_series(n_months=10, phi=0.5, seed=0)
    panel = _panel_from_bias_series(bias, n_markets=50)
    try:
        run_stage4_persistence(panel, min_periods=24)
        assert False, "expected ValueError on too few months"
    except ValueError:
        pass


def test_small_monthly_counts_are_flagged_in_warnings():
    bias = _ar1_monthly_series(n_months=30, phi=0.5, seed=4)
    panel = _panel_from_bias_series(bias, n_markets=50)
    first_month = panel["close_time"].min().to_period("M").to_timestamp()
    mask = panel["close_time"].dt.to_period("M").dt.to_timestamp() == first_month
    panel = pd.concat([panel[mask].iloc[:2], panel[~mask]], ignore_index=True)

    result = run_stage4_persistence(panel, min_markets_per_period=10, min_periods=24)
    assert result.n_periods_flagged_small >= 1
    assert any("markets" in w for w in result.warnings)


def test_acf_pacf_table_caps_nlags_for_short_series():
    x = _ar1_monthly_series(n_months=20, phi=0.5, seed=5)
    table = acf_pacf_table(x)
    assert table["lag"].max() < len(x) // 2


def test_period_bias_series_aggregates_daily():
    # same clipping concern as the monthly version: scale down to stay inside [-0.5, 0.5].
    bias = _ar1_daily_series(n_days=20, phi=0.5, seed=1) * 0.05
    panel = _panel_from_bias_series(bias, n_markets=50, price=0.5, offset=pd.Timedelta(hours=12))
    out = period_bias_series(panel, freq="D", min_markets_per_period=10)

    assert len(out) == 20
    assert (out["n_markets"] == 50).all()
    assert np.allclose(out["bias"].to_numpy(), bias.to_numpy(), atol=1.0 / 50)


def test_orchestrator_runs_on_daily_freq_recurring_series():
    # mirrors the stable-AR1 orchestrator test above, at freq="D" instead of "M" -- this is
    # the shape of a real recurring daily market (e.g. KXTRUMPAPPROVE), where forcing a
    # calendar-month bucket would throw away almost all of the temporal spread.
    bias = _ar1_daily_series(n_days=60, phi=0.6, seed=3) * 0.1
    panel = _panel_from_bias_series(bias, n_markets=200, price=0.5, offset=pd.Timedelta(hours=12))
    result = run_stage4_persistence(panel, freq="D", min_markets_per_period=10, max_p=3, min_periods=30)

    assert result.freq == "D"
    assert result.n_periods == 60
    assert result.level["adf"]["stationary"] is True
    assert result.differencing_applied is False


def test_supwald_null_matches_andrews_published_value():
    # Andrews (1993) Table I: p=1, trim=0.15 -> 5% critical value ~= 8.85. Reduced sim size
    # (4000 reps / 400 grid, vs. the 20000/1000 default) for test speed -- introduces Monte
    # Carlo noise, hence the loose absolute tolerance rather than an exact match.
    null = supwald_null(1, 0.15, 4000, 400, seed=603)
    q95 = np.quantile(null, 0.95)
    assert abs(q95 - 8.85) < 1.5


def _ar1_series_with_break(n: int = 80, break_t: int = 40, phi_before: float = 0.2,
                           phi_after: float = 0.85, const_before: float = 0.0,
                           const_after: float = 3.0, noise_sd: float = 0.3, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    phi = np.where(np.arange(n) < break_t, phi_before, phi_after)
    const = np.where(np.arange(n) < break_t, const_before, const_after)
    for t in range(1, n):
        x[t] = const[t] + phi[t] * x[t - 1] + rng.standard_normal() * noise_sd
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(x, index=idx)


def test_supwald_detects_break_not_stability():
    broken = _ar1_series_with_break(seed=11)
    y_b, X_b = broken.iloc[1:], pd.DataFrame({"lag1": broken.shift(1).dropna()})
    r_broken = sup_wald(y_b, X_b, trim=0.15, boot_reps=499, sim_reps=4000, sim_grid=400, seed=1)
    assert r_broken["p_value"] < 0.01
    # recovered break date should land within a few days of the true one (t=40 -> 2020-02-09)
    assert abs((r_broken["break_date"] - pd.Timestamp("2020-02-09")).days) < 10

    rng = np.random.default_rng(11)
    stable = np.zeros(80)
    for t in range(1, 80):
        stable[t] = 0.5 * stable[t - 1] + rng.standard_normal() * 0.3
    stable = pd.Series(stable, index=pd.date_range("2020-01-01", periods=80, freq="D"))
    y_s, X_s = stable.iloc[1:], pd.DataFrame({"lag1": stable.shift(1).dropna()})
    r_stable = sup_wald(y_s, X_s, trim=0.15, boot_reps=499, sim_reps=4000, sim_grid=400, seed=1)
    assert r_stable["p_value"] > 0.05


def test_ar_structural_break_test_wrapper_runs_on_ar1_fixture():
    x = _ar1_monthly_series(n_months=80, phi=0.6, seed=0)
    result = ar_structural_break_test(x, p=1, trim=0.15, boot_reps=199, sim_reps=2000, sim_grid=300, seed=0)
    assert 0.0 <= result["p_value"] <= 1.0
    assert x.index.min() <= result["break_date"] <= x.index.max()
    assert len(result["profile_wald"]) == len(result["profile_dates"])
    assert np.isfinite(result["profile_wald"]).all()
