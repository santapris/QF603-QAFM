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
from statsmodels.tsa.stattools import adfuller, kpss

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
                     "corr_log_ratio_level_vs_price": float(d["lr"].corr(P))})
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


ALL_STEPS = ["1", "2", "2e", "3", "4", "5", "6", "7", "8", "9"]


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
                      "variance_ratio_check": "variance_ratio_check.csv", "trend_sign_agreement": "trend_sign_agreement.csv"}.items():
        t = rd(name)
        if t is not None:
            summary[key] = t.round(4).to_dict("records")
    if ac is not None:
        summary["acf_bh_rejections"] = ac.groupby("form").bh_reject.sum().astype(int).to_dict()
    (T / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    log("wrote summary.json")


if __name__ == "__main__":
    main()
