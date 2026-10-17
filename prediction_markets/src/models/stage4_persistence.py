"""
Stage 4: persistence -- is this month's average favorite-longshot bias
predictive of next month's?

Mirrors the VRP project's own Stage 1 (a direct, full-sample AR(1)/AR(2)
persistence regression with an ADF stationarity check -- not a walk-forward
forecast; Stage 5 in this project already owns the out-of-sample question).
Goes beyond the roadmap's original one-line idea ("OLS of bias_t on
bias_{t-1}") and does the full Box-Jenkins cycle instead of assuming AR(1):

    aggregate per-market bias into a monthly series
        -> identify (ACF/PACF)
        -> check stationarity (ADF)
        -> select order and fit (AIC-selected AR(p))
        -> diagnose residuals (Ljung-Box, ARCH-LM)
        -> if the level series isn't stationary, difference once and
           repeat identify/fit/diagnose on the differenced series

That last step is the part worth being explicit about: a model that looks
fine on a non-stationary series can be spurious, and the only way to know
whether differencing actually fixed it is to re-run the SAME diagnostics
on the differenced series, not just assume one difference is always enough.

Reference pattern: this mirrors `exploration/driver_surprises.py` on the
VRP project's `main` branch (AIC order selection via
`statsmodels.tsa.ar_model.ar_select_order`, fixed once rather than
re-tuned), but WITHOUT that module's walk-forward `first_origin` --
Stage 4 here is a single full-sample identification/fit cycle, exactly
like VRP's own Stage 1. The residual-diagnostic functions are a
self-contained reimplementation of VRP's `src/diagnostics/
foundational_checks.py` (that module doesn't exist on this older branch
snapshot), not an import.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.stattools import acf, adfuller, pacf


# --------------------------------------------------------------------------- 1. aggregation
def monthly_bias_series(df: pd.DataFrame, min_markets_per_month: int = 10) -> pd.DataFrame:
    """Aggregate the per-market panel (the `combine()`-schema: platform, market_id,
    title, category, close_time, outcome, implied_prob, volume) into one row per
    calendar month of `close_time`.

    Same sign convention as `calibration.bucket_calibration`: bias = mean_price -
    mean_outcome (positive = overpriced, i.e. favorites underpriced / longshots
    overpriced on average that month).

    Months with fewer than `min_markets_per_month` markets are FLAGGED, not
    dropped -- dropping a month would put a silent gap in what's meant to be an
    evenly-spaced monthly series, which would corrupt the AR lag structure
    downstream (bias_{t-1} would silently skip a month). Treat flagged months'
    contribution to the AR fit with real caution; they are not excluded by
    default (see `run_stage4_persistence`'s `drop_flagged` parameter).
    """
    work = df.dropna(subset=["implied_prob", "outcome", "close_time"]).copy()
    close_time = work["close_time"]
    if close_time.dt.tz is not None:
        close_time = close_time.dt.tz_convert(None)
    work["month"] = close_time.dt.to_period("M")

    agg = work.groupby("month", observed=True).agg(
        mean_price=("implied_prob", "mean"),
        mean_outcome=("outcome", "mean"),
        n_markets=("outcome", "size"),
    )
    agg["bias"] = agg["mean_price"] - agg["mean_outcome"]
    agg["se"] = np.sqrt(agg["mean_outcome"] * (1 - agg["mean_outcome"]) / agg["n_markets"])
    agg["flag_small_n"] = agg["n_markets"] < min_markets_per_month

    out = agg.reset_index()
    out["month"] = out["month"].dt.to_timestamp()
    return out.sort_values("month").reset_index(drop=True)


# --------------------------------------------------------------------------- 2. identify
def acf_pacf_table(x: pd.Series, nlags: int | None = None) -> pd.DataFrame:
    """ACF and PACF of `x`, for order identification before fitting anything.

    `nlags` is auto-capped at `min(12, len(x) // 2 - 1)` -- statsmodels refuses
    PACF beyond ~50% of the sample size, and a short monthly series (a couple
    of years of data) hits that ceiling fast.
    """
    x = x.dropna()
    if nlags is None:
        nlags = max(1, min(12, len(x) // 2 - 1))
    a = acf(x, nlags=nlags, fft=False)
    p = pacf(x, nlags=nlags, method="ywm")
    return pd.DataFrame({"lag": range(nlags + 1), "acf": a, "pacf": p})


# --------------------------------------------------------------------------- 3. stationarity
def adf_test(x: pd.Series, alpha: float = 0.05) -> dict:
    """Augmented Dickey-Fuller test. `stationary = p_value < alpha`.

    Borderline p-values (alpha, 2*alpha) are not auto-escalated into triggering
    differencing by the orchestrator below -- one clean, predictable threshold
    for the automated cycle; a borderline case shows up as a warning string
    instead, for a human to judge (see `run_stage4_persistence`).
    """
    x = x.dropna()
    stat, p, used_lag, nobs, crit, _ = adfuller(x, autolag="AIC")
    return dict(stat=float(stat), p_value=float(p), used_lag=int(used_lag), nobs=int(nobs),
               critical_values={k: float(v) for k, v in crit.items()}, stationary=bool(p < alpha))


# --------------------------------------------------------------------------- 4. order selection + fit
def select_ar_order(x: pd.Series, max_p: int = 3) -> int:
    """AIC-selected AR order over the FULL sample.

    Same `ar_select_order(..., ic="aic", old_names=False)` call as VRP's
    `exploration/driver_surprises.py`, but with no walk-forward `first_origin`:
    Stage 4 is a single full-sample identification/fit cycle (matching VRP's
    own Stage 1 persistence test), not a walk-forward forecast -- that's
    Stage 5's job in this project, for the bias-corrected probability forecast.
    """
    x = x.dropna()
    sel = ar_select_order(x.to_numpy(), maxlag=max_p, ic="aic", old_names=False)
    lags = sel.ar_lags or [1]
    return max(lags)


def fit_ar_persistence(x: pd.Series, p: int) -> dict:
    """Full-sample OLS of x_t on its own first p lags (plus a constant).

    Plain OLS standard errors this round, not HAC/Newey-West: unlike VRP's
    VRP_{t+k} construction, this monthly bias series has no overlapping-window
    structure forcing HAC, so plain OLS is defensible for now. Flagged in the
    README roadmap as a future refinement alongside Stage 3's clustering
    question, not built this round.
    """
    x = x.dropna()
    cols = {f"lag{k}": x.shift(k) for k in range(1, p + 1)}
    design = pd.DataFrame(cols, index=x.index)
    design["y"] = x
    design = design.dropna()

    X = sm.add_constant(design.drop(columns="y"))
    result = sm.OLS(design["y"], X).fit()
    return dict(order=p, result=result, coefs=result.params.to_dict(), nobs=int(result.nobs),
               resid=result.resid)


# --------------------------------------------------------------------------- 5. residual diagnostics
def residual_diagnostics(resid: pd.Series, lb_lags: tuple[int, ...] | None = None,
                         arch_nlags: int | None = None) -> dict:
    """Ljung-Box (whiteness) and Engle's ARCH-LM (volatility clustering) on a
    fitted model's residuals -- self-contained here since VRP's
    `src/diagnostics/foundational_checks.py` doesn't exist on this branch's
    older `src/` tree. Field names (`white_noise`, `arch_effects`) deliberately
    echo that module's tables for anyone cross-referencing the two projects.

    Lags are adaptively capped to the residual series' length: a fixed
    (1, 3, 6, 12) would error or mislead on a short monthly series.
    """
    resid = resid.dropna()
    n = len(resid)
    if lb_lags is None:
        lb_lags = tuple(l for l in (1, 3, 6, 12) if l < n // 2) or (1,)
    if arch_nlags is None:
        arch_nlags = max(1, min(6, n // 4))

    lb = acorr_ljungbox(resid, lags=list(lb_lags), return_df=True)
    lb = lb.rename(columns={"lb_stat": "lb_stat", "lb_pvalue": "lb_pvalue"})
    lb["white_noise"] = lb["lb_pvalue"] > 0.05

    lm_stat, lm_p, f_stat, f_p = het_arch(resid, nlags=arch_nlags)
    arch = dict(lm_stat=float(lm_stat), lm_p=float(lm_p), f_stat=float(f_stat), f_p=float(f_p),
               arch_effects=bool(lm_p < 0.05))

    return dict(ljung_box=lb, arch_lm=arch, residual_white_noise=bool(lb["white_noise"].all()),
               arch_effects_present=arch["arch_effects"])


# --------------------------------------------------------------------------- 6. differencing
def difference_series(x: pd.Series) -> pd.Series:
    """First difference, dropping the resulting leading NaN."""
    return x.diff().dropna()


# --------------------------------------------------------------------------- 7. orchestrator
@dataclass
class Stage4Result:
    monthly: pd.DataFrame
    n_months: int
    n_months_flagged_small: int
    level: dict
    differencing_applied: bool
    differenced: dict | None
    warnings: list[str]


def _run_cycle(x: pd.Series, max_p: int, adf_alpha: float) -> dict:
    """One identify/fit/diagnose pass: ACF/PACF, ADF, AIC-selected AR(p) fit,
    Ljung-Box/ARCH-LM on its residuals. Used for both the level series and,
    if triggered, the differenced series."""
    ar_order = select_ar_order(x, max_p=max_p)
    fit = fit_ar_persistence(x, ar_order)
    return dict(
        acf_pacf=acf_pacf_table(x),
        adf=adf_test(x, alpha=adf_alpha),
        ar_order=ar_order,
        ar_fit=fit,
        diagnostics=residual_diagnostics(fit["resid"]),
    )


def run_stage4_persistence(panel: pd.DataFrame, min_markets_per_month: int = 10, max_p: int = 3,
                           adf_alpha: float = 0.05, min_months: int = 24,
                           force_difference: bool = False, drop_flagged: bool = False) -> Stage4Result:
    """Run the full Box-Jenkins cycle on the monthly bias series built from `panel`.

    Raises ValueError if there aren't enough months to fit anything meaningful
    -- same "raise, don't silently proceed" discipline as
    `stage5_oos.run_oos_test`'s too-small-split guard. This is a *feature*, not
    an edge case to work around: a prediction market pull that only spans a
    few days (a known issue with Kalshi's settled-markets feed, see
    `stage5_oos.py`'s module docstring) should refuse to produce a persistence
    number at all, rather than silently reporting one computed on 2 months.
    """
    monthly = monthly_bias_series(panel, min_markets_per_month=min_markets_per_month)
    n_flagged = int(monthly["flag_small_n"].sum())

    usable = monthly[~monthly["flag_small_n"]] if drop_flagged else monthly
    if len(usable) < min_months:
        raise ValueError(
            f"only {len(usable)} usable month(s) of bias data (min_months={min_months}); "
            "pull more data, widen the date range, or lower min_months/min_markets_per_month"
        )

    x = usable.set_index("month")["bias"]
    level = _run_cycle(x, max_p=max_p, adf_alpha=adf_alpha)

    differencing_applied = bool(force_difference or not level["adf"]["stationary"])
    differenced = None
    if differencing_applied:
        diffed = difference_series(x)
        if len(diffed) >= 2 * max_p + 2:
            differenced = _run_cycle(diffed, max_p=max_p, adf_alpha=adf_alpha)
        else:
            differencing_applied = False  # not enough data left after differencing to fit anything

    warnings: list[str] = []
    if n_flagged:
        warnings.append(f"{n_flagged}/{len(monthly)} months have <{min_markets_per_month} markets "
                        "-- treat their contribution to the AR fit cautiously")
    if 0.05 <= level["adf"]["p_value"] < 0.10:
        warnings.append(f"level ADF p={level['adf']['p_value']:.3f} is borderline; "
                        "differencing was not auto-triggered at alpha=0.05")
    if level["diagnostics"]["arch_effects_present"]:
        warnings.append("ARCH effects present in the level fit's residuals -- the bias series' own "
                        "volatility may cluster over time (cf. VRP's calm/stress regime interaction)")
    final = differenced if differencing_applied and differenced is not None else level
    if not final["diagnostics"]["residual_white_noise"]:
        warnings.append("final model's residuals do not pass Ljung-Box at all tested lags -- "
                        "there may be more structure (a higher AR order, or a regime effect) left over")

    return Stage4Result(monthly=monthly, n_months=len(monthly), n_months_flagged_small=n_flagged,
                        level=level, differencing_applied=differencing_applied,
                        differenced=differenced, warnings=warnings)
