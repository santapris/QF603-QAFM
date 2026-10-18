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
from functools import lru_cache

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.stattools import acf, adfuller, pacf


# --------------------------------------------------------------------------- 1. aggregation
def period_bias_series(df: pd.DataFrame, freq: str = "M", min_markets_per_period: int = 10) -> pd.DataFrame:
    """Aggregate the per-market panel (the `combine()`-schema: platform, market_id,
    title, category, close_time, outcome, implied_prob, volume) into one row per
    calendar period of `close_time` -- `freq` is any pandas period alias
    ("M" monthly, "W" weekly, "D" daily). Monthly was the original design, but a
    series that genuinely settles daily (e.g. a daily approval-rating market)
    should be aggregated daily -- forcing it into monthly buckets would throw
    away real, usable temporal spread rather than respect how that series
    actually resolves.

    Same sign convention as `calibration.bucket_calibration`: bias = mean_price -
    mean_outcome (positive = overpriced, i.e. favorites underpriced / longshots
    overpriced on average that period).

    Periods with fewer than `min_markets_per_period` markets are FLAGGED, not
    dropped -- dropping one would put a silent gap in what's meant to be an
    evenly-spaced series, which would corrupt the AR lag structure downstream
    (period_{t-1} would silently skip a period). Treat flagged periods'
    contribution to the AR fit with real caution; they are not excluded by
    default (see `run_stage4_persistence`'s `drop_flagged` parameter).
    """
    work = df.dropna(subset=["implied_prob", "outcome", "close_time"]).copy()
    close_time = work["close_time"]
    if close_time.dt.tz is not None:
        close_time = close_time.dt.tz_convert(None)
    work["period"] = close_time.dt.to_period(freq)

    agg = work.groupby("period", observed=True).agg(
        mean_price=("implied_prob", "mean"),
        mean_outcome=("outcome", "mean"),
        n_markets=("outcome", "size"),
    )
    agg["bias"] = agg["mean_price"] - agg["mean_outcome"]
    agg["se"] = np.sqrt(agg["mean_outcome"] * (1 - agg["mean_outcome"]) / agg["n_markets"])
    agg["flag_small_n"] = agg["n_markets"] < min_markets_per_period

    out = agg.reset_index()
    out["period"] = out["period"].dt.to_timestamp()
    return out.sort_values("period").reset_index(drop=True)


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
    periods: pd.DataFrame
    freq: str
    n_periods: int
    n_periods_flagged_small: int
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


def run_stage4_persistence(panel: pd.DataFrame, freq: str = "M", min_markets_per_period: int = 10,
                           max_p: int = 3, adf_alpha: float = 0.05, min_periods: int = 24,
                           force_difference: bool = False, drop_flagged: bool = False) -> Stage4Result:
    """Run the full Box-Jenkins cycle on the bias series built from `panel`, aggregated at `freq`.

    `freq` must match how the underlying series actually settles -- forcing a
    daily-cadence series (e.g. a daily approval-rating market) into monthly
    buckets would throw away real temporal spread; forcing a sparse series
    into daily buckets would just produce mostly-flagged, near-empty periods.

    Raises ValueError if there aren't enough periods to fit anything meaningful
    -- same "raise, don't silently proceed" discipline as
    `stage5_oos.run_oos_test`'s too-small-split guard. This is a *feature*, not
    an edge case to work around: a prediction market pull that only spans a
    few days (a known issue with Kalshi's settled-markets feed, see
    `stage5_oos.py`'s module docstring) should refuse to produce a persistence
    number at all, rather than silently reporting one computed on 2 periods.
    """
    periods = period_bias_series(panel, freq=freq, min_markets_per_period=min_markets_per_period)
    n_flagged = int(periods["flag_small_n"].sum())

    usable = periods[~periods["flag_small_n"]] if drop_flagged else periods
    if len(usable) < min_periods:
        raise ValueError(
            f"only {len(usable)} usable period(s) of bias data at freq={freq!r} (min_periods={min_periods}); "
            "pull more data, widen the date range, choose a coarser/finer freq, or lower "
            "min_periods/min_markets_per_period"
        )

    x = usable.set_index("period")["bias"]
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
        warnings.append(f"{n_flagged}/{len(periods)} periods have <{min_markets_per_period} markets "
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

    return Stage4Result(periods=periods, freq=freq, n_periods=len(periods),
                        n_periods_flagged_small=n_flagged, level=level,
                        differencing_applied=differencing_applied,
                        differenced=differenced, warnings=warnings)


# --------------------------------------------------------------------------- 8. structural break
# (Andrews 1993 sup-Wald + Hansen 2000 wild bootstrap -- same test, same reasoning, as the VRP
# project's own `src/models/stage3.py::sup_wald`/`supwald_null`/`_wald_stats` on `main`. Ported
# here to answer a concrete question raised by Stage 4's own diagnostics: the KXTRUMPAPPROVE AR(1)
# fit's residuals failed Ljung-Box/ARCH-LM, and an eyeballed "exclude the first week" cut resolved
# both -- this formalizes that as a break test with an unknown break date, instead of a hand-picked
# cutoff. See README.md's "Stage 4" section for the real-data result and why a bootstrap p-value is
# used as the headline rather than the asymptotic one.)
@lru_cache(maxsize=None)
def supwald_null(p: int, trim: float, reps: int, grid: int, seed: int) -> np.ndarray:
    """Simulated draws of sup_tau ||B_p(tau) - tau*B_p(1)||^2 / (tau*(1-tau)) over
    tau in [trim, 1-trim] -- Andrews' (1993) asymptotic limiting distribution for a
    p-parameter sup-Wald break test, via Monte Carlo simulation of the underlying
    p-dimensional Brownian bridge functional (reproduces Andrews' published Table I
    critical values, e.g. ~8.85 at p=1, trim=0.15, 95th percentile)."""
    rng = np.random.default_rng(seed)
    tau = np.arange(1, grid + 1) / grid
    keep = (tau >= trim) & (tau <= 1 - trim)
    out = np.empty(reps)
    chunk = 500
    for i in range(0, reps, chunk):
        m = min(chunk, reps - i)
        W = np.cumsum(rng.standard_normal((m, grid, p)) / np.sqrt(grid), axis=1)
        bridge = W - tau[None, :, None] * W[:, -1:, :]
        stat = (bridge ** 2).sum(axis=2)[:, keep] / (tau[keep] * (1 - tau[keep]))[None, :]
        out[i:i + m] = stat.max(axis=1)
    return np.sort(out)


def _wald_stats(Z: np.ndarray, Y: np.ndarray, p: int) -> np.ndarray:
    """White-robust Wald statistics for the post-break block of Z (last p columns),
    one per column of Y -- vectorized so the real data and every bootstrap draw are
    tested in a single pass.

    `np.errstate` suppresses spurious divide/overflow/invalid RuntimeWarnings that
    Apple's Accelerate BLAS backend raises on these matmuls even when every
    intermediate value is well-conditioned and finite (confirmed by hand on this
    project's real data: A/B/U/meat/V all O(1) or smaller, Vd's condition number in
    the low thousands) -- a known cosmetic quirk of that backend, not a real
    numerical issue. The isnan check below is the actual safety net: a genuine
    singular/ill-conditioned break would show up there, not as a silent warning.
    """
    with np.errstate(all="ignore"):
        A = np.linalg.inv(Z.T @ Z)
        B = A @ Z.T @ Y                                         # (2p, R)
        U = Y - Z @ B
        meat = np.einsum("ti,tj,tr->rij", Z, Z, U ** 2)          # (R, 2p, 2p)
        V = A[None] @ meat @ A[None]
        d = B[p:].T                                             # (R, p)
        Vd = V[:, p:, p:]
        out = np.einsum("ri,ri->r", d, np.linalg.solve(Vd, d[..., None])[..., 0])
    if np.isnan(out).any() or np.isinf(out).any():
        raise FloatingPointError(
            "non-finite Wald statistic -- likely a genuinely rank-deficient design at this "
            "break point (e.g. too few post-break observations, or a post-break regressor "
            "with ~zero variance); lower `trim` cautiously or check the input series"
        )
    return out


def sup_wald(y: pd.Series, X: pd.DataFrame, trim: float = 0.15, boot_reps: int = 499,
            sim_reps: int = 20000, sim_grid: int = 1000, seed: int = 0) -> dict:
    """Andrews (1993) sup-Wald test for a break in all regression coefficients (incl.
    intercept) of `y ~ X` at an unknown date.

    White-robust Wald at each candidate date in [trim, 1-trim] (no overlapping-window
    issue here, unlike VRP's own VRP_{t+k} construction). ``p_value`` is the headline,
    from Hansen's (2000) fixed-regressor wild bootstrap (Rademacher signs, regressors
    held fixed, residuals from the no-break fit) -- the asymptotic test is known to
    over-reject at the sample sizes this project works with (same finding as VRP's own
    D035). ``p_asymptotic`` (from the simulated limiting distribution) is reported
    alongside for reference, not as the decision rule.
    """
    Xc = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    yv = y.to_numpy(float)
    T, p = Xc.shape
    breaks = list(range(int(np.floor(trim * T)), int(np.ceil((1 - trim) * T)) + 1))
    beta0, _, _, _ = np.linalg.lstsq(Xc, yv, rcond=None)
    u0 = yv - Xc @ beta0
    rng = np.random.default_rng(seed)
    eta = rng.choice([-1.0, 1.0], size=(T, boot_reps))
    Ystar = (Xc @ beta0)[:, None] + u0[:, None] * eta
    Y = np.column_stack([yv, Ystar])                       # column 0 = real data
    sup = np.full(boot_reps + 1, -np.inf)
    arg = breaks[0]
    profile = np.empty(len(breaks))
    for i, b in enumerate(breaks):
        post = (np.arange(T) >= b).astype(float)[:, None]
        w = _wald_stats(np.column_stack([Xc, Xc * post]), Y, p)
        profile[i] = w[0]
        if w[0] > sup[0]:
            arg = b
        sup = np.maximum(sup, w)
    null = supwald_null(p, trim, sim_reps, sim_grid, seed)
    return dict(sup_wald=float(sup[0]), break_date=y.index[arg], p_value=float((sup[1:] >= sup[0]).mean()),
               p_asymptotic=float((null >= sup[0]).mean()), crit_5pct_asymptotic=float(np.quantile(null, 0.95)),
               crit_5pct_bootstrap=float(np.quantile(sup[1:], 0.95)), n_params=p, nobs=T, boot_reps=boot_reps,
               profile_dates=[y.index[b] for b in breaks], profile_wald=profile)


def ar_structural_break_test(x: pd.Series, p: int = 1, trim: float = 0.15, boot_reps: int = 499,
                             sim_reps: int = 20000, sim_grid: int = 1000, seed: int = 0) -> dict:
    """Andrews sup-Wald test for a break in the AR(p) persistence fit (const + all p lag
    coefficients) at an unknown date, instead of assuming one. Built specifically to
    replace an ad hoc "exclude the first N periods" cut with a principled test that lets
    the data find the break -- see the module docstring's "structural break" section.

    Needs real statistical power to mean anything: with trim=0.15 this requires roughly
    `p / (1 - 2*trim)` usable observations at minimum to have any candidate break dates at
    all, and considerably more than that to resolve a break near either edge of the sample.
    """
    x = x.dropna()
    cols = {f"lag{k}": x.shift(k) for k in range(1, p + 1)}
    design = pd.DataFrame(cols, index=x.index)
    design["y"] = x
    design = design.dropna()
    return sup_wald(design["y"], design.drop(columns="y"), trim=trim, boot_reps=boot_reps,
                    sim_reps=sim_reps, sim_grid=sim_grid, seed=seed)


# --------------------------------------------------------------------------- 9. plots (visual
# counterparts of the numeric tables above -- same data, nothing new computed here. Mirrors the
# style of VRP's `src/diagnostics/foundational_checks.py` plot_rv_acf_pacf / plot_har_residual_
# diagnostics on the main branch, so the two projects' figures read the same way side by side.)
def plot_bias_series(periods: pd.DataFrame, path, exclude_before=None, title: str | None = None) -> None:
    """Bias-by-period time series. `exclude_before` (a timestamp) shades everything strictly
    before it -- e.g. a new series' volatile launch window -- so a regime transient is visible
    at a glance rather than only in a before/after diagnostics table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(periods["period"], periods["bias"], marker="o", ms=3, lw=1, color="C0")
    ax.axhline(0, color="k", lw=0.5)
    if exclude_before is not None:
        ax.axvspan(periods["period"].min(), exclude_before, color="C3", alpha=0.12,
                  label="excluded window (regime check)")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title or "Favorite-longshot bias by period")
    ax.set_ylabel("bias = mean(price) - mean(outcome)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_level_acf_pacf(x: pd.Series, path, nlags: int | None = None, title: str | None = None) -> None:
    """ACF/PACF of the level bias series -- the identify step, visualized."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    x = x.dropna()
    if nlags is None:
        nlags = max(1, min(12, len(x) // 2 - 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_acf(x, lags=nlags, ax=axes[0])
    plot_pacf(x, lags=nlags, ax=axes[1], method="ywm")
    for ax, name in zip(axes, ["ACF", "PACF"]):
        ax.set_title(f"{title or 'Bias series'} -- {name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_residual_diagnostics(resid: pd.Series, path, lags: int | None = None,
                              title: str | None = None) -> None:
    """Residual series, ACF(residual) (what Ljung-Box tests), ACF(residual^2) (what
    ARCH-LM tests) -- the visual counterpart of residual_diagnostics()."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf

    resid = resid.dropna()
    n = len(resid)
    if lags is None:
        lags = max(1, min(12, n // 2 - 1))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes[0].plot(resid.index, resid.to_numpy(), lw=1, color="C0")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title(f"{title or 'AR fit'}: residuals")
    axes[0].tick_params(axis="x", labelrotation=30)
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(6))
    plot_acf(resid, lags=lags, ax=axes[1])
    axes[1].set_title("ACF(residual) -- Ljung-Box")
    plot_acf(resid ** 2, lags=lags, ax=axes[2])
    axes[2].set_title("ACF(residual²) -- ARCH-LM")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_break_test_profile(result: dict, path, title: str | None = None) -> None:
    """Wald statistic at every candidate break date (`sup_wald`'s `profile_dates`/
    `profile_wald`), with the bootstrap and asymptotic 5% critical values as reference
    lines -- shows at a glance whether one date dominates (a real break candidate) or
    the profile is flat/noisy throughout (no real evidence of a single sharp break,
    even if some individual date's Wald stat looks large in isolation)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = result["profile_dates"]
    wald = result["profile_wald"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, wald, marker="o", ms=3, lw=1, color="C0")
    ax.axhline(result["crit_5pct_bootstrap"], color="C3", ls="--", lw=1,
              label=f"bootstrap 5% crit ({result['crit_5pct_bootstrap']:.1f})")
    ax.axhline(result["crit_5pct_asymptotic"], color="C1", ls=":", lw=1,
              label=f"asymptotic 5% crit ({result['crit_5pct_asymptotic']:.1f})")
    ax.axvline(result["break_date"], color="k", lw=0.7, alpha=0.5)
    ax.set_title(title or f"Andrews sup-Wald profile (bootstrap p={result['p_value']:.2f})")
    ax.set_ylabel("Wald statistic")
    ax.legend(loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
