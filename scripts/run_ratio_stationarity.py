"""DESCRIPTIVE: do ratios and log-ratios of the three spreads have a home? -> output/exploratory/ratio_stationarity/.

Usage:  .venv/Scripts/python.exe scripts/run_ratio_stationarity.py
Requires build_data.py (futures). Reads data/clean/futures_spreads.csv.

Added 2026-11-04 after every registered and exploratory result was seen, in response to the professor's point
that a dollar gap inherits the trend in prices (P_A = k P_B  =>  P_A - P_B = (k-1) P_B), so the RATIO, or
log P_A - log P_B, is the object that should be stationary. Not a gate or a trading rule: it changes no verdict.

Which price, for which job.
  - Stationarity tests and z-scores use the MARKET price of the held contracts (`*_unadj`). That is what refiners
    and shippers face. The back-adjusted level is market + the sum of all past roll gaps, and it goes negative, so
    it is neither testable in logs nor economically meaningful as a level.
  - P&L always uses back-adjusted daily changes. The held-contract return of a leg is
    (back-adjusted change) / (market price the day before), and a spread's held log-ratio return is
    ln(1 + rA) - ln(1 + rB), which equals the market log-ratio change on every non-roll day.
  - Market ratios jump at rolls. Step 6 asks whether the reversion is earned by someone holding the contracts.

Series (all $/bbl, same delivery month, from the `*_unadj` legs):
  crack     log((2 RB + HO) / (3 CL))
  brent_wti log(Brent / CL)
  ho_rb     log(HO / RB)         (crack and ho_rb are also shown deseasonalised by calendar month, full sample)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.unitroot import ZivotAndrews
from scipy.signal import lfilter
from scipy.stats import norm
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import VECM

from oilrv import diagnostics as dg
from oilrv import horizon as H
from oilrv.config import ROOT
from oilrv.plotting import REF_LINE, SERIES, TEXT_2, use_style

sys.path.insert(0, str(ROOT / "scripts"))
import run_ar_acf_pacf as AR  # noqa: E402  (unit_root, order_d, deseason, acf_robust, select_ar, PERIODS)

warnings.filterwarnings("ignore", category=InterpolationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
OUT = ROOT / "output" / "exploratory" / "ratio_stationarity"
LAB = {"crack321": "3-2-1 crack", "brent_wti": "Brent / WTI", "ho_rb": "HO / RB"}
ERAS = {"2007-10": ("2007-01-01", "2010-12-31"), "2011-13": ("2011-01-01", "2013-12-31"),
        "2014-19": ("2014-01-01", "2019-12-31"), "2020-26": ("2020-01-01", None)}
ALPHA, WINDOW, HL_BAND = 0.05, 756, (5, 120)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- data
def load() -> dict[str, dict]:
    """Per spread: market legs A (long side) and B (short side), the dollar difference, the held-contract returns."""
    f = pd.read_csv(ROOT / "data" / "clean" / "futures_spreads.csv", index_col=0, parse_dates=True)
    g = lambda c: f[c].astype(float)
    out = {}
    cl, rb, ho = g("crack321__cl_bbl_unadj"), g("crack321__rb_bbl_unadj"), g("crack321__ho_bbl_unadj")
    out["crack321"] = dict(A=2 * rb + ho, B=3 * cl, diff=g("crack321_unadj"), rolled=f["rolled_crack321"].astype(bool),
                           A_adj=2 * g("crack321__rb_bbl") + g("crack321__ho_bbl"), B_adj=3 * g("crack321__cl_bbl"),
                           level=cl, level_name="WTI", seasonal=True, adj_diff=g("crack321"),
                           legs_log={"cl": cl, "rb": rb, "ho": ho})
    out["brent_wti"] = dict(A=g("brent_wti__brent_bbl_unadj"), B=g("brent_wti__cl_bbl_unadj"), diff=g("brent_wti_unadj"),
                            rolled=f["rolled_brent_wti"].astype(bool), A_adj=g("brent_wti__brent_bbl"),
                            B_adj=g("brent_wti__cl_bbl"), level=g("brent_wti__cl_bbl_unadj"), level_name="WTI",
                            seasonal=False, adj_diff=g("brent_wti"))
    out["ho_rb"] = dict(A=g("ho_rb__ho_bbl_unadj"), B=g("ho_rb__rb_bbl_unadj"), diff=g("ho_rb_unadj"),
                        rolled=f["rolled_ho_rb"].astype(bool), A_adj=g("ho_rb__ho_bbl"), B_adj=g("ho_rb__rb_bbl"),
                        level=g("ho_rb__rb_bbl_unadj"), level_name="RBOB", seasonal=True, adj_diff=g("ho_rb"))
    for n, d in out.items():
        d["ratio"] = d["A"] / d["B"]
        d["lr"] = np.log(d["ratio"])
        d["rA"] = d["A_adj"].diff() / d["A"].shift(1)         # held-contract return of each side
        d["rB"] = d["B_adj"].diff() / d["B"].shift(1)
        d["held"] = np.log1p(d["rA"]) - np.log1p(d["rB"])     # held-contract log-ratio return (exact in logs)
        d["dlr"] = d["lr"].diff()                             # market log-ratio change, includes roll jumps
        d["roll_jump"] = d["dlr"] - d["held"]
        d["lr_ds"] = AR.deseason(d["lr"]) if d["seasonal"] else d["lr"]
        d["diff_ds"] = AR.deseason(d["diff"]) if d["seasonal"] else d["diff"]
    return out


def forms(d: dict) -> dict[str, pd.Series]:
    s = {"difference": d["diff"], "ratio": d["ratio"], "log_ratio": d["lr"]}
    if d["seasonal"]:
        s["log_ratio_deseasonalised"] = d["lr_ds"]
    return s


# --------------------------------------------------------------------------- step 1
def step1(D: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for n, d in D.items():
        for form, s in forms(d).items():
            s = s.dropna()
            for freq, sf in (("daily", s), ("month_end", s.resample("ME").last())):
                for reg in ("c", "ct"):
                    t = AR.order_d(sf, reg, dmax=1)
                    rows.append(t.assign(series=n, form=form, freq=freq, period="full"))
            for lab, (a, b) in AR.PERIODS.items():
                rows.append(AR.order_d(s.loc[a:b], "c", dmax=0).assign(series=n, form=form, freq="daily", period=lab))
    ur = pd.concat(rows, ignore_index=True)
    # regression check: market differences must reproduce the committed unit_roots.csv
    ref = pd.read_csv(ROOT / "output/exploratory/ar_acf_pacf/tables/unit_roots.csv")
    ref = ref[(ref.period == "full") & (ref.regression == "c") & (ref.form == "market") & (ref.d == 0)].set_index("series")
    chk = []
    for n in D:
        mine = ur[(ur.series == n) & (ur.form == "difference") & (ur.freq == "daily") & (ur.period == "full")
                  & (ur.regression == "c") & (ur.d == 0)].iloc[0]
        chk.append({"series": n, "adf_p_here": mine.adf_p, "adf_p_committed": ref.loc[n, "adf_p"],
                    "pp_p_here": mine.pp_p, "pp_p_committed": ref.loc[n, "pp_p"],
                    "match": bool(abs(mine.adf_p - ref.loc[n, "adf_p"]) < 1e-6 and abs(mine.pp_p - ref.loc[n, "pp_p"]) < 1e-6)})
    return ur, pd.DataFrame(chk)


# --------------------------------------------------------------------------- step 2
def kpss_size(phi: float, n: int, reps: int = 200, seed: int = 0) -> tuple[float, float]:
    """Share of simulated stationary AR(1) series that KPSS and that ADF reject at 5% (KPSS should reject 5%)."""
    rng = np.random.default_rng(seed)
    k = a = 0
    for _ in range(reps):
        x = lfilter([1.0], [1.0, -phi], rng.standard_normal(n + 300))[300:]
        k += kpss(x, regression="c", nlags="auto")[1] < ALPHA
        a += adfuller(x, regression="c", autolag="AIC")[1] >= ALPHA          # ADF fails to see stationarity
    return k / reps, a / reps


def eras(D: dict) -> pd.DataFrame:
    """Era means of the market log-ratio, with the implied k = exp(mean) (the constant in P_A = k P_B)."""
    rows = []
    for n, d in D.items():
        for lab, (a, b) in ERAS.items():
            x = d["lr"].loc[a:b]
            rows.append({"series": n, "era": lab, "mean_log_ratio": x.mean(), "implied_k": float(np.exp(x.mean())),
                         "premium_pct": float(100 * (np.exp(x.mean()) - 1)), "sd": x.std(), "n": len(x),
                         "adf_p": adfuller(x, regression="c", autolag="AIC")[1],
                         "kpss_p": kpss(x, regression="c", nlags="auto")[1]})
    return pd.DataFrame(rows)


def step2(D: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    size, za = [], []
    for n, d in D.items():
        for form, s in {"difference": d["diff"], "log_ratio": d["lr"]}.items():
            s = (d["diff_ds"] if form == "difference" else d["lr_ds"]).dropna()
            phi, hl = dg.ar1(s.to_numpy())
            kp, af = kpss_size(phi, len(s))
            size.append({"series": n, "form": form + ("_deseasonalised" if d["seasonal"] else ""), "phi": phi,
                         "half_life_days": hl, "n": len(s), "kpss_rejects_if_truly_stationary": kp,
                         "adf_misses_if_truly_stationary": af})
        for form, s in {"difference": d["diff"], "log_ratio": d["lr"]}.items():
            s = s.dropna()
            z = ZivotAndrews(s.to_numpy(), lags=12, trend="c", trim=0.15)
            t_min, brk = za_break(s.to_numpy(), 12)
            za.append({"series": n, "form": form, "za_stat": z.stat, "za_p": z.pvalue, "lags": z.lags,
                       "own_min_t": t_min, "break_date": s.index[brk].date().isoformat()})
    return pd.DataFrame(size), pd.DataFrame(za)


def za_break(y: np.ndarray, lags: int, trim: float = 0.15) -> tuple[float, int]:
    """Zivot-Andrews with a level break: the break position whose ADF t-statistic on y_{t-1} is smallest.
    arch reports the statistic and p-value but not the break, so the regression is repeated here
    (own_min_t is checked against arch's statistic in the output)."""
    dy = np.diff(y)
    n = len(dy) - lags
    Y = dy[lags:]
    base = np.column_stack([np.ones(n), y[lags:-1]])
    L = np.column_stack([dy[lags - j:len(dy) - j] for j in range(1, lags + 1)])
    best = (np.inf, 0)
    for tb in range(int(trim * n), int((1 - trim) * n)):
        du = (np.arange(n) >= tb).astype(float)
        X = np.column_stack([base, du, L])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        s2 = res @ res / (n - X.shape[1])
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
        t = beta[1] / se
        if t < best[0]:
            best = (float(t), tb + lags + 1)
    return best


# --------------------------------------------------------------------------- step 3
def step3(D: dict) -> pd.DataFrame:
    rows = []
    for n, d in D.items():
        for form, s in {"difference": d["diff_ds"], "log_ratio": d["lr_ds"]}.items():
            s = s.dropna()
            phi, hl = dg.ar1(s.to_numpy())
            r = dg.rolling_stationarity(s, WINDOW)
            ok = (r.half_life >= HL_BAND[0]) & (r.half_life <= HL_BAND[1])
            rows.append({"series": n, "form": form, "half_life_full": hl, "windows": len(r),
                         "share_adf_p_lt_05": float((r.adf_p < ALPHA).mean()),
                         "share_halflife_in_band": float(ok.mean()),
                         "share_both": float(((r.adf_p < ALPHA) & ok).mean()),
                         "median_halflife": float(r.half_life.replace(np.inf, np.nan).median()),
                         "share_no_reversion": float(np.isinf(r.half_life).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- step 4
def pair_legs(D: dict, which: str) -> tuple[pd.Series, pd.Series]:
    d = D[which]
    return d["A"], d["B"]


def step4(D: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, roll = [], []
    for n in ("brent_wti", "ho_rb"):
        y, x = pair_legs(D, n)
        for form, (yy, xx) in {"levels": (y, x), "logs": (np.log(y), np.log(x))}.items():
            full = dg.eg_pair(yy, xx)
            r = dg.rolling_eg(yy, xx, WINDOW)
            roll.append(r.assign(pair=n, form=form).reset_index())
            rows.append({"pair": n, "form": form, "beta_full": full["beta"], "beta_se_hac": full["beta_se_hac"],
                         "t_beta_eq_1_indicative": full["t_beta_eq_1"], "eg_p_full": full["eg_p"],
                         "half_life_resid_full": full["half_life_resid"], "windows": len(r),
                         "share_eg_p_lt_05": float((r.eg_p < ALPHA).mean()), "share_eg_p_lt_10": float((r.eg_p < 0.10).mean()),
                         "share_fixed_adf_lt_05": float((r.adf_p_fixed_1to1 < ALPHA).mean()),
                         "median_beta": float(r.beta.median()), "beta_min": float(r.beta.min()), "beta_max": float(r.beta.max())})
        log(f"  step 4 EG {n}")
    cr = D["crack321"]["legs_log"]
    jrows = []
    for form, lv in {"levels": pd.DataFrame(cr), "logs": np.log(pd.DataFrame(cr))}.items():
        j = dg.rolling_johansen(lv[["cl", "rb", "ho"]], WINDOW, k_ar_diff=1)
        jrows.append({"form": form, "windows": len(j), "share_rank0": float((j["rank"] == 0).mean()),
                      "share_rank1_plus": float((j["rank"] >= 1).mean()),
                      "median_v1_rb": float(j.v1_rb.median()), "median_v1_ho": float(j.v1_ho.median())})
        log(f"  step 4 Johansen {form}")
    return pd.DataFrame(rows), pd.concat(roll, ignore_index=True), pd.DataFrame(jrows)


# --------------------------------------------------------------------------- step 5
def step5(D: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, reg = [], []
    for n, d in D.items():
        P = d["level"]
        sd_d = d["adj_diff"].diff().rolling(60).std()          # held change in dollars (exact: back-adjusted change)
        sd_r = d["held"].rolling(60).std()                     # held change in percent
        rows.append({"series": n, "level_variable": d["level_name"],
                     "corr_rolling_sd_dollar_change_vs_price": float(sd_d.corr(P)),
                     "corr_rolling_sd_log_ratio_return_vs_price": float(sd_r.corr(P)),
                     "corr_difference_level_vs_price": float(d["diff"].corr(P)),
                     "corr_log_ratio_level_vs_price": float(d["lr"].corr(P)),
                     # the back-adjusted level minus the market level is the sum of all past roll gaps
                     "adf_p_accumulated_roll_gap": dg.adf_p((d["adj_diff"] - d["diff"]).to_numpy())})
        df = pd.concat([d["diff"], P], axis=1, keys=["diff", "P"]).dropna()
        r = sm.OLS(df["diff"], sm.add_constant(df["P"])).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
        u = df["diff"] - r.params["const"] - r.params["P"] * df["P"]
        reg.append({"series": n, "a_in_diff_eq_a_plus_cP": r.params["const"], "t_a_hac": r.tvalues["const"],
                    "c_in_diff_eq_a_plus_cP": r.params["P"], "t_c_hac": r.tvalues["P"], "r2": r.rsquared,
                    "adf_p_diff_level": dg.adf_p(df["diff"].to_numpy()), "adf_p_residual_u": dg.adf_p(u.to_numpy()),
                    "half_life_diff": dg.ar1(df["diff"].to_numpy())[1], "half_life_u": dg.ar1(u.to_numpy())[1]})
    return pd.DataFrame(rows), pd.DataFrame(reg)


# --------------------------------------------------------------------------- step 6
def rolling_z(lr: pd.Series, seasonal: bool, window: int = WINDOW) -> pd.Series:
    """z of today's market log-ratio against the `window` days strictly before, month-of-year effect removed
    (estimated inside the same window) for the seasonal spreads. Uses information up to the close of day t only."""
    v, mo = lr.to_numpy(), lr.index.month.to_numpy()
    z = np.full(len(v), np.nan)
    for t in range(window, len(v)):
        w, wm = v[t - window:t], mo[t - window:t]
        if seasonal:
            eff = np.array([w[wm == m].mean() - w.mean() if (wm == m).any() else 0.0 for m in range(1, 13)])
            w, cur = w - eff[wm - 1], v[t] - eff[mo[t] - 1]
        else:
            cur = v[t]
        z[t] = (cur - w.mean()) / w.std(ddof=1)
    return pd.Series(z, index=lr.index)


def hac_slope(y: pd.Series, x: pd.Series, h: int) -> dict:
    df = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    if len(df) < 100:
        return {"slope": np.nan, "t": np.nan, "n": len(df)}
    r = sm.OLS(df["y"], sm.add_constant(df["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": h})
    return {"slope": float(r.params["x"]), "t": float(r.tvalues["x"]), "n": len(df)}


def step6(D: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rt, ex, path = [], [], []
    for n, d in D.items():
        z = rolling_z(d["lr"], d["seasonal"])
        log(f"  step 6 z-score {n}")
        r = d["rolled"]
        ex.append({"series": n, "n_rolls": int(r.sum()),
                   "median_abs_move_roll_day": float(d["dlr"][r].abs().median()),
                   "median_abs_move_other_day": float(d["dlr"][~r].abs().median()),
                   "sum_roll_jumps": float(d["roll_jump"][r].sum()), "sum_held_returns": float(d["held"].sum()),
                   "range_of_market_log_ratio": float(d["lr"].max() - d["lr"].min()),
                   "sum_market_change_equals_roll_plus_held": float(d["dlr"].sum())})
        for h in (21, 63):
            fh = d["held"].rolling(h).sum().shift(-h)          # forward held-contract log-ratio return, days t+1..t+h
            fm = d["dlr"].rolling(h).sum().shift(-h)           # forward market log-ratio change (includes roll jumps)
            for lab, (a, b) in {"full": (None, None), "2010-26": ("2010-01-01", None), **{k: v for k, v in AR.PERIODS.items()}}.items():
                sl = slice(a, b)
                held, mkt = hac_slope(fh.loc[sl], z.loc[sl], h), hac_slope(fm.loc[sl], z.loc[sl], h)
                rt.append({"series": n, "horizon": h, "period": lab, "held_slope": held["slope"], "held_t": held["t"],
                           "market_slope": mkt["slope"], "market_t": mkt["t"], "n": held["n"],
                           "roll_share_of_market_reversion": (1 - held["slope"] / mkt["slope"]) if mkt["slope"] else np.nan})
        # simple trading-style read: after |z| >= 2, what does a holder of the fade earn over the next 21 days?
        fh = d["held"].rolling(21).sum().shift(-21)
        sig = -np.sign(z).where(z.abs() >= 2)
        pnl = (sig * fh).dropna()
        path.append({"series": n, "days_with_abs_z_ge_2": int(len(pnl)), "share_of_days": float(len(pnl) / z.notna().sum()),
                     "mean_fwd21_held_return_of_fade": float(pnl.mean()),
                     # overlapping 21-day windows: divide n by 21 for a rough effective sample size
                     "t_approx": float(pnl.mean() / (pnl.std() / np.sqrt(len(pnl) / 21))) if len(pnl) > 21 else np.nan})
        # step 6c: the held-contract path (cumulated held returns) is what a holder owns, so test it directly
        h_path = d["held"].fillna(0).cumsum()
        t = AR.order_d(h_path, "c", dmax=0).iloc[0]
        ex[-1].update({"held_path_adf_p": float(t.adf_p), "held_path_pp_p": float(t.pp_p), "held_path_kpss_p": float(t.kpss_p),
                       "held_path_half_life": float(dg.ar1(h_path.to_numpy())[1])})
    return pd.DataFrame(rt), pd.DataFrame(ex), pd.DataFrame(path)


# --------------------------------------------------------------------------- step 7
def step7(D: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ac, rows = [], []
    for n, d in D.items():
        for form, s in {"held_log_ratio_return": d["held"], "market_log_ratio_change_ex_roll": d["dlr"].where(~d["rolled"])}.items():
            y = s.dropna()
            a = AR.acf_robust(y.to_numpy(), 20)
            a["p"] = 2 * norm.sf(np.abs(a["acf"]) / a["se_robust"])
            ac.append(a.assign(series=n, form=form).reset_index())
            ic = AR.select_ar(y.to_numpy())
            pa, pb = int(ic.aic.idxmin()), int(ic.bic.idxmin())
            lb = dg.ljung_box(y.to_numpy(), lags=(10, 20))
            rows.append({"series": n, "form": form, "n": len(y), "bic_order": pb, "aic_order": pa,
                         "r2_aic": float(ic.loc[pa, "r2"]), "lb_plain_p_20": lb["lb_p_20"], "lb_robust_p_20": lb["robust_p_20"],
                         "arch_lm_p_10": float(het_arch(y.to_numpy() - y.mean(), nlags=10)[1])})
    ac = pd.concat(ac, ignore_index=True)
    for form, g in ac.groupby("form"):
        rej = multipletests(g["p"].to_numpy(), alpha=ALPHA, method="fdr_bh")[0]
        ac.loc[g.index, "bh_reject"] = rej
    return ac, pd.DataFrame(rows)


# --------------------------------------------------------------------------- step 8
VR_DAILY, VR_MONTHLY = (5, 21, 63, 126, 252), (3, 6, 12)
OUTRIGHTS = {"cl": ("crack321__cl_bbl", "WTI"), "brent": ("brent_wti__brent_bbl", "Brent"),
             "rb": ("crack321__rb_bbl", "RBOB"), "ho": ("crack321__ho_bbl", "ULSD")}


def outright_held() -> dict[str, pd.DataFrame]:
    """Per outright: held-contract log return and back-adjusted dollar change (both exact on every day)."""
    f = pd.read_csv(ROOT / "data" / "clean" / "futures_spreads.csv", index_col=0, parse_dates=True)
    out = {}
    for k, (col, _) in OUTRIGHTS.items():
        adj, mkt = f[col].astype(float), f[f"{col}_unadj"].astype(float)
        out[k] = pd.DataFrame({"log_held": np.log1p(adj.diff() / mkt.shift(1)), "dollar": adj.diff()})
    return out


def vr_rows(label: str, basis: str, dx: pd.Series) -> list[dict]:
    """Variance ratios of the increments `dx` at daily horizons (robust z) and, on month sums, at 3/6/12 months."""
    rows = []
    dx = dx.dropna()
    for freq, x, qs in (("daily", dx, VR_DAILY), ("monthly", dx.resample("ME").sum(), VR_MONTHLY)):
        for q in qs:
            v = H.variance_ratio(x.to_numpy(), q)
            rows.append({"series": label, "basis": basis, "freq": freq, "q": q, "n": v["n"], "vr": v["vr"],
                         "z_robust": v["z2"], "p_robust": v["p2"]})
    return rows


def step8(D: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for n, d in D.items():
        rows += vr_rows(n, "held_log_ratio", d["held"])
        rows += vr_rows(n, "dollar", d["adj_diff"].diff())
    for k, o in outright_held().items():
        rows += vr_rows(k, "held_log", o["log_held"])
        rows += vr_rows(k, "dollar", o["dollar"])
    vr = pd.DataFrame(rows)
    # regression check: the dollar rows must reproduce the committed horizon-test table at q = 21..252
    ref = pd.read_csv(ROOT / "output/exploratory/horizon_tests/tables/variance_ratios.csv", index_col=0)
    chk = []
    for s in ref.index:
        for q in (21, 63, 126, 252):
            mine = vr[(vr.series == s) & (vr.basis == "dollar") & (vr.freq == "daily") & (vr.q == q)]
            if len(mine):
                chk.append({"series": s, "q": q, "vr_here": float(mine.vr.iloc[0]), "vr_committed": float(ref.loc[s, f"vr{q}"])})
    chk = pd.DataFrame(chk)
    chk["abs_diff"] = (chk.vr_here - chk.vr_committed).abs()
    return vr, chk


# --------------------------------------------------------------------------- step 9
def step9(D: dict) -> pd.DataFrame:
    """Trend rule is the sign of the 252-day back-adjusted dollar change. How often does it disagree with the sign of
    the 252-day held log-ratio return, and what does the dollar-spread trend earn on days of agreement and disagreement?"""
    rows = []
    for n, d in D.items():
        dollar = np.sign(d["adj_diff"] - d["adj_diff"].shift(252))
        logr = np.sign(d["held"].rolling(252).sum())
        nxt = d["adj_diff"].diff().shift(-1)                   # next day's change in the dollar spread ($/bbl per unit)
        ok = dollar.notna() & logr.notna() & (dollar != 0) & (logr != 0) & nxt.notna()
        dis = ok & (dollar != logr)
        agr = ok & (dollar == logr)
        pnl = dollar * nxt
        rows.append({"series": n, "days": int(ok.sum()), "share_disagree": float(dis.sum() / ok.sum()),
                     "mean_daily_dollar_trend_pnl_agree": float(pnl[agr].mean()),
                     "mean_daily_dollar_trend_pnl_disagree": float(pnl[dis].mean()),
                     "mean_daily_logratio_trend_pnl_all": float((logr * d["held"].shift(-1))[ok].mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- step 10: the moving anchor
def uc_fit(y: np.ndarray, level: str):
    mod = UnobservedComponents(y, level=level, autoregressive=1)
    return mod, mod.fit(disp=False, maxiter=300)


def half_life(phi: float) -> float:
    return float(np.log(2) / -np.log(phi)) if 0 < phi < 1 else np.inf


def step10(D: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Local level + AR(1):  l_t = log k_t + u_t,  log k_t = log k_{t-1} + eta_t,  u_t = phi u_{t-1} + eps_t.

    Fitted on the market log-ratio (calendar-month effects removed for the crack and HO/RB). Compared with a fixed mean
    (sigma_eta = 0). The anchor the strategy uses is the trailing 756-day mean; how far is it from the filtered log k_t?"""
    rows, series, err = [], [], []
    for n, d in D.items():
        y = d["lr_ds"].dropna()
        mod, res = uc_fit(y.to_numpy(), "llevel")
        mod0, res0 = uc_fit(y.to_numpy(), "fixed intercept")
        p, p0 = dict(zip(mod.param_names, res.params)), dict(zip(mod0.param_names, res0.params))
        phi, phi0 = p["ar.L1"], p0["ar.L1"]
        s_eta, s_ar = np.sqrt(p["sigma2.level"]), np.sqrt(p["sigma2.ar"])
        sd_u = s_ar / np.sqrt(1 - phi ** 2)
        lev_s = pd.Series(res.smoothed_state[0], index=y.index)
        lev_f = pd.Series(res.filtered_state[0], index=y.index)
        u_f = y - lev_f
        rows.append({"series": n, "phi_moving_anchor": phi, "half_life_moving_anchor_days": half_life(phi),
                     "phi_fixed_mean": phi0, "half_life_fixed_mean_days": half_life(phi0),
                     "sigma_level_daily": s_eta, "sigma_k_annual": s_eta * np.sqrt(252),
                     "sd_deviation_u": sd_u, "sd_total": float(y.std()),
                     "share_of_variance_in_anchor": float(1 - sd_u ** 2 / y.var()),
                     "llf_moving": float(res.llf), "llf_fixed": float(res0.llf), "lr_stat": float(2 * (res.llf - res0.llf)),
                     "converged": bool(res.mle_retvals.get("converged", True))})
        series.append(pd.DataFrame({"series": n, "k_smoothed": np.exp(lev_s), "k_filtered": np.exp(lev_f),
                                    "u_filtered_over_sd": u_f / sd_u}).rename_axis("date").reset_index())
        # the strategy's anchor against the filtered one
        m756 = y.shift(1).rolling(WINDOW).mean()
        gap = (m756 - lev_f.shift(1)).dropna()
        z = rolling_z(d["lr"], d["seasonal"]).reindex(y.index)
        big = z.abs() >= 2
        uf = (u_f / sd_u).abs()
        err.append({"series": n, "rmse_anchor_error_over_sd_u": float(np.sqrt((gap ** 2).mean()) / sd_u),
                    "days_abs_z756_ge_2": int(big.sum()),
                    "share_of_those_where_filtered_deviation_lt_1sd": float((uf[big] < 1).mean()),
                    "share_where_filtered_deviation_lt_2sd": float((uf[big] < 2).mean()),
                    "share_where_filtered_deviation_ge_2sd": float((uf[big] >= 2).mean()),
                    "days_filtered_dev_ge_2sd": int((uf >= 2).sum()),
                    "share_of_those_missed_by_z756": float((z[uf >= 2].abs() < 2).mean()),
                    "corr_z756_vs_filtered_dev": float(z.corr(u_f / sd_u))})
    return pd.DataFrame(rows), pd.concat(series, ignore_index=True), pd.DataFrame(err)


# --------------------------------------------------------------------------- step 11: is k the same number everywhere?
def step11(D: dict) -> pd.DataFrame:
    """The dollar regression S = a + c P_B gives a premium of c + a E[1/P_B] over B; the log-ratio gives exp(mean l) - 1.
    For the crack S is per barrel of crude and P_B is WTI, so S / P_B = A/B - 1 exactly."""
    rows = []
    for n, d in D.items():
        df = pd.concat([d["diff"], d["level"]], axis=1, keys=["S", "P"]).dropna()
        r = sm.OLS(df["S"], sm.add_constant(df["P"])).fit()
        a, c = float(r.params["const"]), float(r.params["P"])
        rows.append({"series": n, "a_fixed_dollar": a, "c_proportional": c, "mean_price_P_B": float(df["P"].mean()),
                     "premium_from_dollar_regression": c + a * float((1 / df["P"]).mean()),
                     "premium_from_mean_log_ratio": float(np.exp(d["lr"].mean()) - 1),
                     "premium_mean_of_ratio_minus_1": float((d["A"] / d["B"] - 1).mean()),
                     "premium_proportional_part_only": c})
    t = pd.DataFrame(rows)
    t["gap_dollar_vs_log"] = t.premium_from_dollar_regression - t.premium_from_mean_log_ratio
    return t


# --------------------------------------------------------------------------- step 12: the 2013 heating oil -> ULSD change
HO_SWITCH = pd.Timestamp("2013-05-01")        # NYMEX HO became ULSD in 2013 (oil-rv-plan.md); the exact month is an assumption


def window_jump(y: pd.Series, at: pd.Timestamp, w: int = 252) -> tuple[float, float]:
    """Mean of y over the w days from `at` minus the mean over the w days before it, with a Newey-West t."""
    pos = y.index.searchsorted(at)
    if pos < w or pos + w > len(y):
        return np.nan, np.nan
    seg = y.iloc[pos - w:pos + w]
    dummy = (np.arange(2 * w) >= w).astype(float)
    r = sm.OLS(seg.to_numpy(), sm.add_constant(dummy)).fit(cov_type="HAC", cov_kwds={"maxlags": 60})
    return float(r.params[1]), float(r.tvalues[1])


def step12(D: dict) -> pd.DataFrame:
    rows = []
    starts = pd.date_range("2008-01-01", "2025-06-01", freq="MS")
    for n, d in D.items():
        y = d["lr_ds"].dropna()
        pl = pd.DataFrame([(s, *window_jump(y, s)) for s in starts], columns=["date", "b", "t"]).dropna()
        for lab, dt in [("2013-03", "2013-03-01"), ("2013-05 (assumed switch)", "2013-05-01"), ("2013-07", "2013-07-01")]:
            b, t = window_jump(y, pd.Timestamp(dt))
            rows.append({"series": n, "break_date": lab, "jump_log_ratio": b, "t_hac": t,
                         "percentile_of_abs_jump_among_all_month_starts": float((pl.b.abs() < abs(b)).mean()),
                         "placebo_sd_of_jump": float(pl.b.std()), "n_placebo": len(pl)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- step 13: does k follow Cushing stocks?
def step13(D: dict, uc_series: pd.DataFrame) -> pd.DataFrame:
    f = pd.read_csv(ROOT / "data" / "clean" / "features_futures.csv", index_col=0, parse_dates=True)
    cu = f["cushing_dev_5y"]
    k = uc_series[uc_series.series == "brent_wti"].set_index("date")["k_smoothed"]
    wk = pd.DataFrame({"lr": D["brent_wti"]["lr"], "cushing_dev": cu, "log_k": np.log(k)}).dropna(subset=["cushing_dev"]) \
        .resample("W-FRI").last().dropna()
    rows = []
    for lab, y, x, lags in [("log-ratio on Cushing deviation (levels, weekly)", wk["lr"], wk["cushing_dev"], 26),
                            ("smoothed log k on Cushing deviation (levels, weekly)", wk["log_k"], wk["cushing_dev"], 26),
                            ("weekly change in log-ratio on weekly change in Cushing deviation", wk["lr"].diff(), wk["cushing_dev"].diff(), 4)]:
        df = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
        r = sm.OLS(df["y"], sm.add_constant(df["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        rows.append({"regression": lab, "slope": float(r.params["x"]), "t_hac": float(r.tvalues["x"]), "r2": float(r.rsquared),
                     "n_weeks": len(df), "start": str(df.index[0].date())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- step 14: who moves, in logs
def step14(D: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(a) free VECM(1), rank 2, on the log market prices of CL, RB, HO (as Test 7 did in dollars).
    (b) the known log-ratio as the error-correction term, one equation per leg on the leg's held-contract log return."""
    f = pd.read_csv(ROOT / "data" / "clean" / "futures_spreads.csv", index_col=0, parse_dates=True)
    lv = pd.DataFrame({"cl": np.log(f["crack321__cl_bbl_unadj"]), "rb": np.log(f["crack321__rb_bbl_unadj"]),
                       "ho": np.log(f["crack321__ho_bbl_unadj"])})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VECM(lv.to_numpy(), k_ar_diff=1, coint_rank=2, deterministic="ci").fit()
    cols = ["ect1", "ect2"]
    free = pd.concat({"alpha": pd.DataFrame(res.alpha, index=lv.columns, columns=cols),
                      "alpha_t": pd.DataFrame(res.alpha / res.stderr_alpha, index=lv.columns, columns=cols),
                      "beta": pd.DataFrame(res.beta, index=lv.columns, columns=cols)})
    legs = {"crack321": [("rb", "crack321__rb_bbl"), ("ho", "crack321__ho_bbl"), ("cl", "crack321__cl_bbl")],
            "brent_wti": [("brent", "brent_wti__brent_bbl"), ("cl", "brent_wti__cl_bbl")],
            "ho_rb": [("ho", "ho_rb__ho_bbl"), ("rb", "ho_rb__rb_bbl")]}
    rows = []
    for n, d in D.items():
        ect = d["lr_ds"].shift(1).rename("ect")
        held = {}
        for k, col in legs[n]:
            adj, mkt = f[col].astype(float), f[f"{col}_unadj"].astype(float)
            held[k] = np.log1p(adj.diff() / mkt.shift(1))
        H = pd.DataFrame(held)
        for k in H:
            X = pd.concat([ect, H.shift(1).add_suffix("_lag")], axis=1)
            df = pd.concat([H[k].rename("y"), X], axis=1).dropna()
            r = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit(cov_type="HC1")
            rows.append({"spread": n, "equation": f"held log return of {k.upper()}", "alpha": float(r.params["ect"]),
                         "t": float(r.tvalues["ect"])})
        X = pd.concat([ect, d["held"].shift(1).rename("held_lag")], axis=1)
        df = pd.concat([d["held"].rename("y"), X], axis=1).dropna()
        r = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit(cov_type="HC1")
        sp = float(r.params["ect"])
        rows.append({"spread": n, "equation": "held log-ratio return (speed of the ratio on itself)", "alpha": sp,
                     "t": float(r.tvalues["ect"]), "implied_half_life_days": half_life(1 + sp)})
    return free, pd.DataFrame(rows)


# --------------------------------------------------------------------------- step 15: ARIMA identification
ARIMA_LAB = {"crack321": "3-2-1 crack", "brent_wti": "Brent / WTI", "ho_rb": "HO / RB", "cl": "WTI", "brent": "Brent", "rb": "RBOB",
             "ho": "ULSD"}
FORM_PRODUCT = "log price, d = 1 (ARMA on held log returns)"
FORM_DOLLAR = "dollar gap, d = 1 (ARMA on back-adjusted daily change)"
FORM_LOG0 = "log-ratio, d = 0 (market)"
FORM_LOG1 = "log-ratio, d = 1 (market, roll jumps included)"
FORM_HELD = "held log-ratio index, d = 1 (ARMA on held return)"


def arima_grid(y: np.ndarray, d: int, pmax: int = 3, qmax: int = 3) -> pd.DataFrame:
    rows = []
    for p in range(pmax + 1):
        for q in range(qmax + 1):
            try:
                r = ARIMA(y, order=(p, d, q), trend="c" if d == 0 else "t" if d == 1 else "n").fit()
                rows.append({"p": p, "q": q, "aic": float(r.aic), "bic": float(r.bic), "res": r})
            except Exception:
                rows.append({"p": p, "q": q, "aic": np.inf, "bic": np.inf, "res": None})
    return pd.DataFrame(rows)


def arima_row(series: str, form: str, y: np.ndarray, d: int, d_label: int | None = None) -> tuple[list, list]:
    """Best AIC and BIC models for `y` fitted as ARIMA(p, d, q); `d_label` is the order to print when `y` is already differenced."""
    g = arima_grid(y, d)
    rows, factors = [], []
    for crit in ("bic", "aic"):
        b = g.loc[g[crit].idxmin()]
        r = b["res"]
        p, q = int(b.p), int(b.q)
        resid = np.asarray(r.resid)[max(p, q, 1) + 5:]
        lb = dg.ljung_box(resid, lags=(20,), model_df=p + q)
        ar_r, ma_r = (np.asarray(r.arroots) if p else np.array([])), (np.asarray(r.maroots) if q else np.array([]))
        slow = float(np.log(2) / np.log(np.abs(ar_r).min())) if p and np.abs(ar_r).min() > 1 else np.nan
        dd = d if d_label is None else d_label
        rows.append({"series": series, "form": form, "criterion": crit.upper(), "order": f"({p},{dd},{q})",
                     "aic": float(b.aic), "bic": float(b.bic), "slowest_ar_root_half_life_days": slow,
                     "lb_plain_p20": lb["lb_p_20"], "lb_robust_p20": lb["robust_p_20"],
                     "arch_lm_p10": float(het_arch(resid - resid.mean(), nlags=10)[1]), "n": len(y),
                     "params": json.dumps({k: round(float(v), 4) for k, v in zip(r.param_names, r.params) if not k.startswith("sigma")})})
        if crit == "bic":
            for kind, roots in (("AR", ar_r), ("MA", ma_r)):
                for z in roots:
                    factors.append({"series": series, "form": form, "polynomial": kind, "root_in_L": complex(z).__repr__(),
                                    "modulus": float(abs(z)), "real": float(np.real(z)), "imag": float(np.imag(z)),
                                    "half_life_days": float(np.log(2) / np.log(abs(z))) if kind == "AR" and abs(z) > 1 else np.nan})
            if p and q:                                                        # AR and MA roots that sit on top of each other cancel
                dist = np.abs(ar_r[:, None] - ma_r[None, :])
                for f in factors[-(p + q):]:
                    f["has_cancelling_partner_within_0.15"] = bool((dist < 0.15).any())
    return rows, factors


def ar_forecast_h(y: np.ndarray, beta: np.ndarray, p: int, h: int) -> float:
    hist = list(y[-p:][::-1])
    for _ in range(h):
        nxt = beta[0] + sum(beta[1 + j] * hist[j] for j in range(p))
        hist = ([nxt] + hist[:p - 1]) if p > 1 else [nxt]
    return hist[0]


def ar_ols(w: np.ndarray, p: int) -> np.ndarray:
    X = np.column_stack([np.ones(len(w) - p)] + [w[p - j:len(w) - j] for j in range(1, p + 1)])
    return np.linalg.lstsq(X, w[p:], rcond=None)[0]


def step15(D: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, factors, integ = [], [], []
    held = outright_held()
    for k, o in held.items():
        r_ = o["log_held"].dropna()
        idx = r_.cumsum()
        integ.append({"series": ARIMA_LAB[k], "adf_p_log_index": adfuller(idx, autolag="AIC")[1], "kpss_p_log_index": AR.unit_root(idx, "c")["kpss_p"],
                      "adf_p_returns": adfuller(r_, autolag="AIC")[1], "kpss_p_returns": AR.unit_root(r_, "c")["kpss_p"]})
        a, b = arima_row(ARIMA_LAB[k], FORM_PRODUCT, r_.to_numpy(), 0, d_label=1)
        rows, factors = rows + a, factors + b
    # WTI: the cancelling ARMA(2,2) comes from the March to May 2020 episode
    w = held["cl"]["log_held"].dropna()
    wc = []
    for lab, x in [("full sample", w), ("excluding 2020-03-09 to 2020-05-29", w[(w.index < "2020-03-09") | (w.index > "2020-05-29")])]:
        g = arima_grid(x.to_numpy(), 0)
        bb = g.loc[g.bic.idxmin()]
        wc.append({"sample": lab, "n": len(x), "bic_order": f"({int(bb.p)},1,{int(bb.q)})", "bic_best": float(bb.bic),
                   "bic_random_walk": float(g[(g.p == 0) & (g.q == 0)].bic.iloc[0])})
    for n, d in D.items():
        a, b = arima_row(ARIMA_LAB[n], FORM_DOLLAR, d["adj_diff"].diff().dropna().to_numpy(), 0, d_label=1)
        rows, factors = rows + a, factors + b
        y = d["lr_ds"].dropna().to_numpy()
        a, b = arima_row(ARIMA_LAB[n], FORM_LOG0, y, 0)
        rows, factors = rows + a, factors + b
        a, b = arima_row(ARIMA_LAB[n], FORM_LOG1, y, 1)
        rows, factors = rows + a, factors + b
        a, b = arima_row(ARIMA_LAB[n], FORM_HELD, d["held"].dropna().to_numpy(), 0, d_label=1)
        rows, factors = rows + a, factors + b
        log(f"  step 15 {n} fitted")
    # out of sample: does an AR model of the log-ratio beat a random walk? Parameters refit monthly on the trailing 756 days.
    hs, oos = (1, 5, 21, 63), []
    for n, d in D.items():
        y = d["lr_ds"].dropna().to_numpy()
        errs = {(m, h): [] for m in ("mean", "ar1", "ar3", "rw") for h in hs}
        for m0 in range(WINDOW, len(y) - max(hs), 21):
            w_ = y[m0 - WINDOW:m0]
            b1, b3, mu = ar_ols(w_, 1), ar_ols(w_, 3), w_.mean()
            for t in range(m0, min(m0 + 21, len(y) - max(hs))):
                for h in hs:
                    tgt = y[t + h]
                    errs[("rw", h)].append(tgt - y[t])
                    errs[("mean", h)].append(tgt - mu)
                    errs[("ar1", h)].append(tgt - ar_forecast_h(y[:t + 1], b1, 1, h))
                    errs[("ar3", h)].append(tgt - ar_forecast_h(y[:t + 1], b3, 3, h))
        for h in hs:
            rw = np.array(errs[("rw", h)])
            for m in ("mean", "ar1", "ar3"):
                e = np.array(errs[(m, h)])
                dl = rw ** 2 - e ** 2                                          # positive: the model beats the random walk
                rr = sm.OLS(dl, np.ones(len(dl))).fit(cov_type="HAC", cov_kwds={"maxlags": h + 5})
                oos.append({"series": ARIMA_LAB[n], "horizon_days": h, "model": m,
                            "rmse_ratio_vs_random_walk": float(np.sqrt((e ** 2).mean() / (rw ** 2).mean())), "dm_t": float(rr.tvalues[0]), "n": len(dl)})
    return pd.DataFrame(rows), pd.DataFrame(factors), pd.DataFrame(oos), pd.DataFrame(wc), pd.DataFrame(integ)


# --------------------------------------------------------------------------- figures
def figures(D: dict, era: pd.DataFrame, size: pd.DataFrame) -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(13, 5.6))
    for j, (n, d) in enumerate(D.items()):
        a = ax[0, j]
        a.plot(d["diff"].index, d["diff"], color=SERIES[1], lw=0.8)
        a.axhline(d["diff"].mean(), color=REF_LINE, lw=0.8, ls="--")
        a.set_title(f"{LAB[n]}: difference ($/bbl)")
        b = ax[1, j]
        b.plot(d["lr"].index, d["lr"], color=SERIES[0], lw=0.8)
        for lab, (s0, s1) in ERAS.items():
            x = d["lr"].loc[s0:s1]
            b.hlines(x.mean(), x.index[0], x.index[-1], color="#000", lw=1.6)
        b.set_title(f"{LAB[n]}: log-ratio, with era means")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "01_difference_vs_log_ratio.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    s = size.copy()
    s["lab"] = s.series.map(LAB) + "\n" + s.form.str.replace("_deseasonalised", "").str.replace("_", " ")
    ax.bar(range(len(s)), s.kpss_rejects_if_truly_stationary * 100, color=[SERIES[0] if "log" in f else SERIES[1] for f in s.form])
    ax.axhline(5, color=REF_LINE, ls="--", lw=0.8)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s.lab, fontsize=7)
    ax.set_ylabel("% of simulated stationary series\nthat KPSS rejects")
    ax.set_title("KPSS false alarms at each series' own persistence (should be 5%)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "02_kpss_false_alarms.png", dpi=140)
    plt.close(fig)


def fig_vr(vr: pd.DataFrame) -> None:
    """Variance ratio against horizon: held log returns against back-adjusted dollar changes."""
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    names = [("crack321", "3-2-1 crack"), ("brent_wti", "Brent / WTI"), ("ho_rb", "HO / RB"), ("cl", "WTI"), ("brent", "Brent"),
             ("rb", "RBOB"), ("ho", "ULSD")]
    fig, ax = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    for a, (k, lab) in zip(ax.flat, names):
        for basis, col, ls in (("dollar", SERIES[1], "--"), ("held_log_ratio", SERIES[0], "-"), ("held_log", SERIES[0], "-")):
            s = vr[(vr.series == k) & (vr.basis == basis) & (vr.freq == "daily")].sort_values("q")
            if len(s):
                a.plot(s.q, s.vr, color=col, ls=ls, marker="o", ms=3,
                       label="Back-adjusted dollar changes" if basis == "dollar" else "Held log returns")
        a.axhline(1, color=REF_LINE, lw=0.8)
        a.set_xscale("log")
        a.set_xticks([5, 21, 63, 252], ["5", "21", "63", "252"])
        a.set_title(lab)
    ax[1, 3].axis("off")
    ax[0, 0].legend(loc="lower left", fontsize=7)
    for a in ax[1, :3]:
        a.set_xlabel("Horizon q (trading days)")
    ax[0, 0].set_ylabel("Variance ratio")
    ax[1, 0].set_ylabel("Variance ratio")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "03_variance_ratio_profile.png", dpi=140)
    plt.close(fig)


ALL_STEPS = ["1", "2", "2e", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(ALL_STEPS),
                    help="comma list from " + ",".join(ALL_STEPS) + " (2 = KPSS simulation and Zivot-Andrews, ~5 min)")
    steps = set(ap.parse_args().steps.split(","))
    use_style()
    T = OUT / "tables"
    T.mkdir(parents=True, exist_ok=True)
    D = load()
    log("loaded")
    if "1" in steps:
        ur, chk = step1(D)
        ur.to_csv(T / "unit_roots.csv", index=False)
        chk.to_csv(T / "regression_check.csv", index=False)
        log(f"step 1 done; regression check match = {bool(chk.match.all())}")
    if "2" in steps:
        size, za = step2(D)
        size.to_csv(T / "kpss_size.csv", index=False)
        za.to_csv(T / "zivot_andrews.csv", index=False)
        log("step 2 done")
    if "2e" in steps:
        eras(D).to_csv(T / "eras.csv", index=False)
        log("eras done")
    if "3" in steps:
        step3(D).to_csv(T / "rolling_stationarity.csv", index=False)
        log("step 3 done")
    if "4" in steps:
        eg, eg_roll, joh = step4(D)
        eg.to_csv(T / "cointegration_summary.csv", index=False)
        eg_roll.to_csv(T / "rolling_eg.csv", index=False)
        joh.to_csv(T / "johansen_summary.csv", index=False)
        log("step 4 done")
    if "5" in steps:
        vol, reg = step5(D)
        vol.to_csv(T / "scale_dependence.csv", index=False)
        reg.to_csv(T / "diff_on_price.csv", index=False)
        log("step 5 done")
    if "6" in steps:
        rt, ex, fade = step6(D)
        rt.to_csv(T / "tradeable_reversion.csv", index=False)
        ex.to_csv(T / "roll_jumps.csv", index=False)
        fade.to_csv(T / "fade_after_z2.csv", index=False)
        log("step 6 done")
    if "7" in steps:
        ac, ch = step7(D)
        ac.to_csv(T / "acf_log_ratio_changes.csv", index=False)
        ch.to_csv(T / "changes_summary.csv", index=False)
        log("step 7 done")
    if "8" in steps:
        vr, vchk = step8(D)
        vr.to_csv(T / "variance_ratios.csv", index=False)
        vchk.to_csv(T / "variance_ratio_check.csv", index=False)
        log(f"step 8 done; max |VR here - committed| = {vchk.abs_diff.max():.4f}")
    if "9" in steps:
        step9(D).to_csv(T / "trend_sign_agreement.csv", index=False)
        log("step 9 done")
    if "10" in steps or "13" in steps:
        uc_t, uc_s, uc_e = step10(D)
        if "10" in steps:
            uc_t.to_csv(T / "moving_anchor_model.csv", index=False)
            uc_s.to_csv(T / "moving_anchor_series.csv", index=False)
            uc_e.to_csv(T / "anchor_error.csv", index=False)
            log("step 10 done")
    if "11" in steps:
        step11(D).to_csv(T / "k_consistency.csv", index=False)
        log("step 11 done")
    if "12" in steps:
        step12(D).to_csv(T / "heating_oil_2013.csv", index=False)
        log("step 12 done")
    if "13" in steps:
        step13(D, uc_s).to_csv(T / "cushing_k.csv", index=False)
        log("step 13 done")
    if "15" in steps:
        a_o, a_f, a_oos, a_wti, a_int = step15(D)
        a_o.to_csv(T / "arima_orders.csv", index=False)
        a_f.to_csv(T / "arima_factors.csv", index=False)
        a_oos.to_csv(T / "arima_oos.csv", index=False)
        a_wti.to_csv(T / "arima_wti_check.csv", index=False)
        a_int.to_csv(T / "arima_integration.csv", index=False)
        log("step 15 done")
    if "14" in steps:
        free14, known14 = step14(D)
        free14.to_csv(T / "vecm_logs_free.csv")
        known14.to_csv(T / "vecm_logs_known.csv", index=False)
        log("step 14 done")

    rd = lambda n: pd.read_csv(T / n) if (T / n).exists() else None
    if rd("eras.csv") is not None and rd("kpss_size.csv") is not None:
        figures(D, rd("eras.csv"), rd("kpss_size.csv"))
    if rd("variance_ratios.csv") is not None:
        fig_vr(rd("variance_ratios.csv"))
    ur, ac = rd("unit_roots.csv"), rd("acf_log_ratio_changes.csv")
    summary = {"registered_on": "2026-11-04", "n_days": int(len(D["crack321"]["lr"]))}
    if ur is not None:
        summary["unit_roots_daily_full_c_d0"] = ur[(ur.freq == "daily") & (ur.period == "full") & (ur.regression == "c") & (ur.d == 0)][["series", "form", "adf_p", "pp_p", "kpss_p"]].round(4).to_dict("records")
    if rd("regression_check.csv") is not None:
        summary["regression_check_match"] = bool(rd("regression_check.csv").match.all())
    for key, name in {"kpss_size": "kpss_size.csv", "eras": "eras.csv", "zivot_andrews": "zivot_andrews.csv",
                      "rolling_stationarity": "rolling_stationarity.csv", "cointegration": "cointegration_summary.csv",
                      "johansen": "johansen_summary.csv", "scale_dependence": "scale_dependence.csv",
                      "diff_on_price": "diff_on_price.csv", "roll_jumps": "roll_jumps.csv",
                      "tradeable_reversion": "tradeable_reversion.csv", "fade_after_z2": "fade_after_z2.csv",
                      "changes": "changes_summary.csv", "variance_ratios": "variance_ratios.csv",
                      "variance_ratio_check": "variance_ratio_check.csv", "trend_sign_agreement": "trend_sign_agreement.csv",
                      "moving_anchor_model": "moving_anchor_model.csv", "anchor_error": "anchor_error.csv", "k_consistency": "k_consistency.csv",
                      "heating_oil_2013": "heating_oil_2013.csv", "cushing_k": "cushing_k.csv", "vecm_logs_known": "vecm_logs_known.csv",
                      "arima_orders": "arima_orders.csv", "arima_oos": "arima_oos.csv", "arima_wti_check": "arima_wti_check.csv"}.items():
        t = rd(name)
        if t is not None:
            summary[key] = t.round(4).to_dict("records")
    if ac is not None:
        summary["acf_bh_rejections"] = ac.groupby("form").bh_reject.sum().astype(int).to_dict()
    (T / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    log("wrote summary.json")


if __name__ == "__main__":
    main()
