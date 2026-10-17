import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.measures.calibration import bucket_calibration
from src.models.stage4_persistence import (
    acf_pacf_table,
    adf_test,
    difference_series,
    fit_ar_persistence,
    monthly_bias_series,
    residual_diagnostics,
    run_stage4_persistence,
    select_ar_order,
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


def _panel_from_bias_series(bias: pd.Series, n_markets: int = 50, price: float = 0.5) -> pd.DataFrame:
    """Build a per-market panel whose monthly aggregation reproduces `bias` exactly:
    fix implied_prob at `price` for every market, and choose how many resolve
    YES each month so that mean_outcome = price - bias[m] precisely."""
    rows = []
    for month_start, b in bias.items():
        mean_outcome = np.clip(price - b, 0.0, 1.0)
        n_yes = int(round(mean_outcome * n_markets))
        for i in range(n_markets):
            rows.append(dict(
                platform="kalshi", market_id=f"{month_start.strftime('%Y%m')}-{i}", title="t",
                category="c", close_time=month_start + pd.Timedelta(days=15),
                outcome=1.0 if i < n_yes else 0.0, implied_prob=price, volume=100,
            ))
    return pd.DataFrame(rows)


def test_monthly_bias_series_aggregates_correctly():
    # scaled down: bias = price - mean_outcome is bounded to [-price, 1-price] (+-0.5 here),
    # so an unscaled AR(1) (std ~1.15 at phi=0.5) would clip and silently corrupt this test.
    bias = _ar1_monthly_series(n_months=12, phi=0.5, seed=1) * 0.05
    panel = _panel_from_bias_series(bias, n_markets=50, price=0.5)
    out = monthly_bias_series(panel, min_markets_per_month=10)

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

    out = monthly_bias_series(panel, min_markets_per_month=10)
    assert out.loc[out["month"] == first_month, "flag_small_n"].iloc[0]
    assert out["flag_small_n"].sum() == 1


def test_monthly_bias_series_sign_matches_calibration_convention():
    # same panel read two ways: bucket_calibration's price-axis bias and
    # monthly_bias_series's time-axis bias should agree on sign/definition
    # (price - outcome) when computed over the whole sample.
    bias = _ar1_monthly_series(n_months=12, phi=0.3, seed=3)
    panel = _panel_from_bias_series(bias, n_markets=50, price=0.7)
    panel_renamed = panel.rename(columns={"implied_prob": "implied_prob"})

    bucket = bucket_calibration(panel_renamed, n_buckets=2)
    monthly = monthly_bias_series(panel, min_markets_per_month=10)

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
    result = run_stage4_persistence(panel, min_markets_per_month=10, max_p=3, min_months=24)

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
    result = run_stage4_persistence(panel, min_markets_per_month=10, max_p=3, min_months=24)

    assert result.level["adf"]["stationary"] is True
    assert result.differencing_applied is False
    assert result.differenced is None


def test_raises_on_too_few_months():
    bias = _ar1_monthly_series(n_months=10, phi=0.5, seed=0)
    panel = _panel_from_bias_series(bias, n_markets=50)
    try:
        run_stage4_persistence(panel, min_months=24)
        assert False, "expected ValueError on too few months"
    except ValueError:
        pass


def test_small_monthly_counts_are_flagged_in_warnings():
    bias = _ar1_monthly_series(n_months=30, phi=0.5, seed=4)
    panel = _panel_from_bias_series(bias, n_markets=50)
    first_month = panel["close_time"].min().to_period("M").to_timestamp()
    mask = panel["close_time"].dt.to_period("M").dt.to_timestamp() == first_month
    panel = pd.concat([panel[mask].iloc[:2], panel[~mask]], ignore_index=True)

    result = run_stage4_persistence(panel, min_markets_per_month=10, min_months=24)
    assert result.n_months_flagged_small >= 1
    assert any("markets" in w for w in result.warnings)


def test_acf_pacf_table_caps_nlags_for_short_series():
    x = _ar1_monthly_series(n_months=20, phi=0.5, seed=5)
    table = acf_pacf_table(x)
    assert table["lag"].max() < len(x) // 2
