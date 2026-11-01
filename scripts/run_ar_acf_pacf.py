"""DESCRIPTIVE: order of integration, AR order, ACF and PACF of the 7 traded series -> output/exploratory/ar_acf_pacf/.

Usage:  .venv/bin/python scripts/run_ar_acf_pacf.py
Requires build_data.py (futures). Series: the back-adjusted outrights and spreads of `trend_carry.build_instruments`,
plus their market levels as a check. `ar_on_changes.csv` also reports a `full_ex_roll_days` sample.

Added 2026-11-01, after every registered and exploratory result was seen. Not a gate or a trading rule, so it changes
no verdict. The tables hold statistics only, no prices.

d: the smallest difference at which ADF and Phillips-Perron reject a unit root and KPSS does not reject stationarity,
all at 5% (`stationary_all` in unit_roots.csv). AR order: AIC and BIC over AR(0..20) on one common sample. ACF and PACF
bands use standard errors robust to volatility clustering; significance across lags 1-20 is also corrected by
Benjamini-Hochberg over all series.
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.unitroot import PhillipsPerron
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from oilrv import diagnostics as dg
from oilrv import futures
from oilrv import trend_carry as TC
from oilrv.config import ROOT, load_config, set_source
from oilrv.plotting import GRID, REF_LINE, SERIES, TEXT_2, use_style
from oilrv.spreads import to_bbl

OUT = ROOT / "output" / "exploratory" / "ar_acf_pacf"
LABELS = {"cl": "WTI (CL)", "brent": "Brent (CO)", "rb": "RBOB (XB)", "ho": "ULSD (HO)",
          "crack321": "3-2-1 crack", "brent_wti": "Brent − WTI", "ho_rb": "HO − RB"}
SEASONAL = ["rb", "ho", "crack321", "ho_rb"]
PERIODS = {"2007-13": ("2007-01-01", "2013-12-31"), "2014-19": ("2014-01-01", "2019-12-31"), "2020-26": ("2020-01-01", None)}
MAXLAG = 20
ALPHA = 0.05


def acf_robust(x, nlags: int) -> pd.DataFrame:
    """ACF by lag with the white-noise se and sqrt(tau_k / n), tau_k as in `diagnostics.ljung_box`."""
    e = np.asarray(x, dtype=float)
    e = e - e.mean()
    n, g0 = len(e), np.mean(e ** 2)
    k = np.arange(1, nlags + 1)
    rho = np.array([np.sum(e[j:] * e[:-j]) / (n * g0) for j in k])
    tau = np.array([np.sum(e[j:] ** 2 * e[:-j] ** 2) / (n * g0 ** 2) for j in k])
    return pd.DataFrame({"acf": rho, "se_bartlett": 1 / np.sqrt(n), "se_robust": np.sqrt(tau / n)}, index=pd.Index(k, name="lag"))


def lag_matrix(v: np.ndarray, p: int, start: int) -> np.ndarray:
    """Lags 1..p of v for rows start..end, so models with different p can share one sample."""
    return np.column_stack([v[start - j:len(v) - j] for j in range(1, p + 1)]) if p else np.empty((len(v) - start, 0))


def pacf_robust(x, nlags: int) -> pd.DataFrame:
    """PACF by lag: the last coefficient of an OLS AR(k) with a constant, and its White (HC1) se."""
    v = np.asarray(x, dtype=float)
    rows = {}
    for k in range(1, nlags + 1):
        r = sm.OLS(v[k:], sm.add_constant(lag_matrix(v, k, k))).fit(cov_type="HC1")
        rows[k] = {"pacf": r.params[-1], "se_robust": r.bse[-1]}
    return pd.DataFrame(rows).T.rename_axis("lag")


def unit_root(s: pd.Series, reg: str) -> dict:
    v = s.to_numpy(dtype=float)
    a = adfuller(v, regression=reg, autolag="AIC", result_object=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)   # p clipped to [0.01, 0.1]
        k = kpss(v, regression=reg, nlags="auto", result_object=False)
    return {"adf_stat": a[0], "adf_p": a[1], "adf_lags": a[2], "kpss_stat": k[0], "kpss_p": k[1],
            "pp_p": PhillipsPerron(v, trend=reg).pvalue}


def order_d(x: pd.Series, reg: str, dmax: int = 2) -> pd.DataFrame:
    """The three tests at each d = 0..dmax, with the over-differencing checks (sd, lag-1 ACF)."""
    rows = []
    s = x.dropna()
    for d in range(dmax + 1):
        rows.append({"d": d, "regression": reg, **unit_root(s, reg), "sd": s.std(), "acf1": acf(s, nlags=1, fft=True)[1]})
        s = s.diff().dropna()
    t = pd.DataFrame(rows)
    t["stationary_adf"] = t.adf_p < ALPHA
    t["stationary_pp"] = t.pp_p < ALPHA
    t["stationary_kpss"] = t.kpss_p >= ALPHA
    t["stationary_all"] = t.stationary_adf & t.stationary_pp & t.stationary_kpss
    return t


def select_ar(y, maxlag: int = MAXLAG) -> pd.DataFrame:
    """AIC and BIC for AR(0..maxlag) with a constant on one common sample."""
    v = np.asarray(y, dtype=float)
    n, out = len(v) - maxlag, {}
    for p in range(maxlag + 1):
        r = sm.OLS(v[maxlag:], sm.add_constant(lag_matrix(v, p, maxlag), has_constant="add")).fit()
        ll = n * np.log(r.ssr / n)
        out[p] = {"aic": ll + 2 * (p + 1), "bic": ll + np.log(n) * (p + 1), "r2": r.rsquared}
    return pd.DataFrame(out).T.rename_axis("p")


def fit_ar(y, p: int) -> tuple:
    v = np.asarray(y, dtype=float)
    r = sm.OLS(v[p:], sm.add_constant(lag_matrix(v, p, p), has_constant="add")).fit(cov_type="HC1")
    m = max(20, p + 5)
    lb = dg.ljung_box(r.resid, lags=(m,), model_df=p)
    return r, {"lb_lag": m, "lb_p": lb[f"lb_p_{m}"], "robust_p": lb[f"robust_p_{m}"]}


def deseason(s: pd.Series) -> pd.Series:
    return s - s.groupby(s.index.month).transform("mean") + s.mean()


def ar_row(y: pd.Series, label: str) -> dict:
    ic = select_ar(y)
    pa, pb = int(ic.aic.idxmin()), int(ic.bic.idxmin())
    r1, _ = fit_ar(y, 1)
    rb, lb = fit_ar(y, pb)
    ra, la = fit_ar(y, pa)
    sig = [j for j in range(1, pa + 1) if ra.pvalues[j] < ALPHA]
    return {"sample": label, "n": len(y), "p_bic": pb, "p_aic": pa, "ar1_phi": r1.params[1], "ar1_se": r1.bse[1], "ar1_p": r1.pvalues[1],
            "ar1_r2": r1.rsquared, "r2_bic": rb.rsquared, "r2_aic": ra.rsquared, "aic_sig_lags": " ".join(map(str, sig)),
            "bic_lb_lag": lb["lb_lag"], "bic_lb_plain_p": lb["lb_p"], "bic_lb_robust_p": lb["robust_p"],
            "aic_lb_robust_p": la["robust_p"], "aic_coefs": " ".join(f"{c:+.3f}" for c in ra.params[1:])}


def build_series(cfg: dict) -> tuple[dict, dict]:
    """({series: back-adjusted level}, {series: market level}), both in $/bbl on the common futures dates."""
    inst = TC.build_instruments(cfg=cfg)
    fut = futures.load_futures()
    panels, roots = futures.load_panels(cfg), futures.leg_roots(cfg)
    rb, ov = cfg["futures"]["roll_bdays_before_expiry"], futures.load_expiry_overrides()
    x = {n: f["x"].astype(float) for n, f in inst.items()}
    mkt = {}
    for n in LABELS:
        if n in roots:
            c = futures.continuous({n: panels[n]}, {n: roots[n]}, 1, rb, ov)
            mkt[n] = to_bbl(c[[n]], cfg)[n].reindex(fut.index).astype(float)
        else:
            mkt[n] = fut[f"{n}_unadj"].astype(float)
    return x, mkt


def unit_root_tables(x: dict, mkt: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ur = []
    for n in LABELS:
        forms = {"back_adjusted": x[n], "market": mkt[n]}
        if n in SEASONAL:
            forms["market_deseasonalised"] = deseason(mkt[n].dropna())
        for form, s in forms.items():
            for reg in ("c", "ct"):
                ur.append(order_d(s, reg).assign(series=n, form=form, period="full"))
        for lab, (a, b) in PERIODS.items():
            for form, s in {"back_adjusted": x[n], "market": mkt[n]}.items():
                ur.append(order_d(s.loc[a:b], "c", dmax=1).assign(series=n, form=form, period=lab))
    ur = pd.concat(ur, ignore_index=True)
    full = ur[ur.period == "full"]
    smallest = full.groupby(["series", "form", "regression"], sort=False).apply(
        lambda g: g.loc[g.stationary_all, "d"].min(), include_groups=False).unstack()
    return ur, smallest


def level_table(x: dict, mkt: dict) -> pd.DataFrame:
    rows = []
    for n in LABELS:
        for form, s in {"back_adjusted": x[n], "market": mkt[n]}.items():
            v = s.dropna().to_numpy()
            b, hl = dg.ar1(v)
            se = sm.OLS(v[1:], sm.add_constant(v[:-1])).fit(cov_type="HC1").bse[1]
            ic = select_ar(v)
            lb = dg.ljung_box(dg.ar1_residuals(v), (20,), 1)
            rows.append({"series": n, "form": form, "b": b, "se": se, "half_life": hl, "p_bic_level": int(ic.bic.idxmin()),
                         "p_aic_level": int(ic.aic.idxmin()), "innov_lb_p20": lb["lb_p_20"], "innov_robust_p20": lb["robust_p_20"]})
    return pd.DataFrame(rows)


def change_table(x: dict, rolled: dict) -> pd.DataFrame:
    rows = []
    for n in LABELS:
        dx = x[n].diff().dropna()
        lo, hi = dx.quantile([0.005, 0.995])
        rows.append({"series": n, **ar_row(dx, "full")})
        rows.append({"series": n, **ar_row(dx.clip(lo, hi), "full_winsorised")})
        rows.append({"series": n, **ar_row(dx[~rolled[n].reindex(dx.index)], "full_ex_roll_days")})
        for lab, (a, b) in PERIODS.items():
            rows.append({"series": n, **ar_row(dx.loc[a:b], lab)})
    return pd.DataFrame(rows)


def acf_table(x: dict) -> pd.DataFrame:
    rows = []
    for n in LABELS:
        for form, s, K, Kp in [("level", x[n], 252, 20), ("change", x[n].diff().dropna(), 40, 40), ("change2", x[n].diff().diff().dropna(), 40, 40)]:
            a, p = acf_robust(s, K), pacf_robust(s, Kp)
            if not np.allclose(pacf(s.to_numpy(), nlags=Kp, method="ols")[1:], p.pacf.to_numpy(), atol=1e-8):
                raise ValueError(f"{n} {form}: PACF does not match statsmodels")
            if not np.allclose(acf(s.to_numpy(), nlags=K, fft=True)[1:], a.acf.to_numpy(), atol=1e-8):
                raise ValueError(f"{n} {form}: ACF does not match statsmodels")
            rows.append(a.join(p, rsuffix="_pacf").reset_index().assign(series=n, form=form))
    ac = pd.concat(rows, ignore_index=True)
    ac["acf_p_robust"] = 2 * norm.sf(np.abs(ac.acf / ac.se_robust))
    ac["pacf_p_robust"] = 2 * norm.sf(np.abs(ac.pacf / ac.se_robust_pacf))
    return ac


def lag_tests(ac: pd.DataFrame) -> pd.DataFrame:
    ch = ac[(ac.form == "change") & (ac.lag <= MAXLAG)]
    t = pd.concat([ch.assign(kind="acf", p=ch.acf_p_robust, val=ch.acf), ch.assign(kind="pacf", p=ch.pacf_p_robust, val=ch.pacf)])
    t["bh_reject"] = multipletests(t.p, alpha=ALPHA, method="fdr_bh")[0]
    return t[["series", "kind", "lag", "val", "p", "bh_reject"]]


def plot(ac: pd.DataFrame) -> None:
    for form, name, Ka, Kp, ylim, title in [
        ("level", "levels", 252, 20, (-0.25, 1.05), "Back-adjusted level: ACF decays slowly and PACF cuts off after lag 1"),
        ("change", "changes", 40, 40, (-0.17, 0.17), "Daily change (d = 1): ACF and PACF. Shaded band: 95% robust; dashed: 95% white noise")]:
        fig, axes = plt.subplots(7, 2, figsize=(10.5, 12), sharex="col", sharey=True)
        for i, n in enumerate(LABELS):
            d = ac[(ac.series == n) & (ac.form == form)]
            for j, (col, se, K) in enumerate([("acf", "se_robust", Ka), ("pacf", "se_robust_pacf", Kp)]):
                t, a = d[d.lag <= K], axes[i, j]
                if form == "change":
                    a.fill_between(t.lag, -1.96 * t[se], 1.96 * t[se], color=GRID, lw=0)
                    for k in (-1.96, 1.96):
                        a.plot(t.lag, k * t.se_bartlett, color=TEXT_2, lw=0.7, ls="--")
                a.axhline(0, color=REF_LINE, lw=0.7)
                a.vlines(t.lag, 0, t[col], color=SERIES[0], lw=1.4 if K <= 40 else 0.7)
                a.set_ylim(*ylim)
                a.set_title(f"{LABELS[n]}: {col.upper()}", fontsize=9)
                a.grid(axis="x", visible=False)
                a.xaxis.get_major_locator().set_params(integer=True)
            axes[i, 0].set_ylabel("Correlation")
        axes[-1, 0].set_xlabel("Lag (trading days)")
        axes[-1, 1].set_xlabel("Lag (trading days)")
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(OUT / "figures" / f"acf_pacf_{name}.png")
        plt.close(fig)


def main() -> None:
    use_style()
    cfg = load_config()
    set_source("futures")
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    x, mkt = build_series(cfg)
    ur, smallest = unit_root_tables(x, mkt)
    ac = acf_table(x)
    tests = lag_tests(ac)
    ur.to_csv(OUT / "tables" / "unit_roots.csv", index=False)
    smallest.to_csv(OUT / "tables" / "smallest_d.csv")
    level_table(x, mkt).to_csv(OUT / "tables" / "ar1_levels.csv", index=False)
    rolled = {n: f["rolled"].astype(bool) for n, f in TC.build_instruments(cfg=cfg).items()}
    change_table(x, rolled).to_csv(OUT / "tables" / "ar_on_changes.csv", index=False)
    ac.to_csv(OUT / "tables" / "acf_pacf.csv", index=False)
    tests.to_csv(OUT / "tables" / "change_lag_tests.csv", index=False)
    plot(ac)
    print(smallest.to_string())
    print(f"lags 1-{MAXLAG} on daily changes: {len(tests)} tests, {int((tests.p < ALPHA).sum())} with raw p < {ALPHA}, "
          f"{int(tests.bh_reject.sum())} after Benjamini-Hochberg")


if __name__ == "__main__":
    main()
