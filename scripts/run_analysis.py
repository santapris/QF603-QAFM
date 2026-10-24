"""Parts 2-3: summary moments, diagnostics and seasonality -> output/tables, output/figures, data/derived.

Usage:  .venv/bin/python scripts/run_analysis.py [--skip-slow]
Requires scripts/build_data.py to have run. --skip-slow skips the rolling SARIMA forecasts and the
rolling Johansen tests.

All results are on spot data (a spot proxy; see PREREGISTRATION.md section 5).
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.vector_ar.vecm import select_order

from oilrv import diagnostics as dg
from oilrv import explore as ex
from oilrv.config import load_config, path
from oilrv.data import load_clean
from oilrv.plotting import CRITICAL, REF_LINE, SERIES, TAB_DIR, TEXT_2, save, save_table, use_style

LABELS = {"crack321": "3-2-1 crack", "brent_wti": "Brent − WTI", "ho_rb": "HO − RB"}
SEASONAL = ["crack321", "ho_rb"]
LEGS = ["cl_bbl", "rb_bbl", "ho_bbl"]
TARGETS = {"crack321": [-3, 2, 1], "ho_rb": [0, -1, 1], "gasoline_crack": [-1, 1, 0], "diesel_crack": [-1, 0, 1]}
EVENTS = [("2008-09-15", "Lehman"), ("2011-02-15", "Libya / Cushing glut"), ("2014-11-27", "OPEC price war"),
          ("2020-04-20", "WTI < 0"), ("2022-02-24", "Ukraine"), ("2026-02-28", "Hormuz")]


def _shade_stress(ax, cfg, last):
    for name, (a, b) in cfg["evaluation"]["stress_windows"].items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b) if b else last, color=REF_LINE, alpha=0.10, lw=0)


# --------------------------------------------------------------------------- Part 2
def part2(sp: pd.DataFrame, cfg: dict) -> dict:
    d = ex.daily_changes(sp)
    mom = ex.moments_table(d)
    save_table(mom, "02_moments.csv")
    keep = ~sp["flag_apr2020"].reindex(d.index).astype(bool)
    mom_x = ex.moments_table(d[keep])
    save_table(mom_x, "02_moments_excl_apr2020.csv")
    byp = ex.moments_by_period(d, cfg["evaluation"]["moments_periods"])
    save_table(byp, "02_moments_by_period.csv")

    # Fat tails: log-density histograms against fitted normal and Student-t
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for a, c in zip(ax, d):
        x = d.loc[keep, c]
        lim = np.quantile(x.abs(), 0.999) * 1.2
        bins = np.linspace(-lim, lim, 81)
        a.hist(x, bins=bins, density=True, color=SERIES[0], alpha=0.85, label="Observed")
        grid = np.linspace(-lim, lim, 400)
        a.plot(grid, stats.norm.pdf(grid, x.mean(), x.std()), color=SERIES[1], lw=1.6, label="Normal")
        df_, loc, scale = stats.t.fit(x.to_numpy())
        a.plot(grid, stats.t.pdf(grid, df_, loc, scale), color=SERIES[2], lw=1.6, label=f"Student-t (ν = {df_:.1f})")
        a.set_yscale("log")
        a.set_ylim(1e-4, None)
        a.set_xlabel(f"Daily change in {LABELS[c]} ($/bbl)")
        a.set_title(f"{LABELS[c]}: excess kurtosis {x.kurt():.1f}")
        a.legend(loc="upper left", fontsize=7)
    ax[0].set_ylabel("Density (log scale)")
    fig.suptitle("Daily spread changes have fat tails (April 2020 week excluded from the plot)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "02_spread_change_distribution.png")

    # Volatility clustering
    rv = pd.DataFrame({c: ex.rolling_vol(d[c]) for c in d})
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for i, c in enumerate(rv):
        ax.plot(rv.index, rv[c], color=SERIES[i], lw=0.9, label=LABELS[c])
    ax.set_yscale("log")
    ax.set_ylabel("60-day sd of daily change ($/bbl, log)")
    ax.set_title("Volatility clusters: 2022 and 2026 sit far above the calm years")
    ax.legend(loc="upper left", ncols=3)
    for date, lab in EVENTS:
        ax.axvline(pd.Timestamp(date), color=REF_LINE, lw=0.6, ls=":")
        ax.text(pd.Timestamp(date), 0.02, f" {lab}", transform=ax.get_xaxis_transform(), fontsize=7,
                color=TEXT_2, va="bottom", ha="left", rotation=90)
    save(fig, "02_rolling_volatility.png")

    # Seasonality on monthly averages: STL strength and the detrended month-of-year effect
    strength, months = {}, {}
    stl = {}
    for c in d:
        m = ex.monthly_mean(sp[c])
        stl[c] = ex.stl_decompose(m, period=12)
        strength[c] = ex.stl_strength(stl[c])
        mt = ex.detrended_by_month(m)
        strength[c]["kruskal_p_month"] = mt.attrs["kruskal_p"]
        months[c] = mt["mean"]
    st = pd.DataFrame(strength).T
    save_table(st, "02_stl_strength.csv")
    mon = pd.DataFrame(months)
    save_table(mon, "02_month_effects.csv")

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    for i, c in enumerate(d):
        idx = ex.monthly_mean(sp[c]).index
        ax[0].plot(idx, np.asarray(stl[c].seasonal), color=SERIES[i], lw=0.9, label=LABELS[c])
    ax[0].axhline(0, color=REF_LINE, lw=0.7)
    ax[0].set_ylabel("$/bbl")
    ax[0].set_title("STL seasonal component of monthly averages")
    ax[0].legend(loc="upper left", ncols=3)
    w = 0.27
    for i, c in enumerate(d):
        ax[1].bar(mon.index + (i - 1) * w, mon[c], width=w, color=SERIES[i],
                  label=f"{LABELS[c]} (Kruskal p = {strength[c]['kruskal_p_month']:.3f})")
    ax[1].axhline(0, color=REF_LINE, lw=0.7)
    ax[1].set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax[1].set_ylabel("$/bbl vs 13-month centred average")
    ax[1].set_title("Month-of-year effect (detrended)")
    ax[1].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, "02_seasonality.png")

    return {
        "moments": mom[["std", "skew", "excess_kurt", "t_df", "p_gt_4sd_obs", "arch_lm_p"]].astype(float).round(4).to_dict("index"),
        "excess_kurt_excl_apr2020": mom_x["excess_kurt"].astype(float).round(2).to_dict(),
        "std_by_period": byp["std"].unstack().round(3).to_dict("index"),
        "stl_and_month_effect": st.astype(float).round(4).to_dict("index"),
    }


# --------------------------------------------------------------------------- Part 3
def part3(sp: pd.DataFrame, cfg: dict, skip_slow: bool) -> dict:
    w = cfg["model"]["window"]
    band = (cfg["model"]["half_life_min"], cfg["model"]["half_life_max"])
    periods = cfg["evaluation"]["sub_periods"]
    out = {}

    # 4a. Unit roots: legs and spreads
    ur = dg.unit_root_table({**{c.replace("_bbl", ""): sp[c] for c in ["cl_bbl", "brent_bbl", "rb_bbl", "ho_bbl"]},
                             **{c: sp[c] for c in LABELS}})
    save_table(ur, "03_unit_roots.csv")
    out["unit_roots"] = ur[["adf_p", "kpss_p", "adf_says", "kpss_says"]].reset_index().to_dict("records")

    # 4a. Rolling stationarity of each spread (feeds the regime filter's rolling_adf_p)
    rs = {c: dg.rolling_stationarity(sp[c], window=w) for c in LABELS}
    long = pd.concat({c: r for c, r in rs.items()}, names=["spread", "date"])
    long.to_csv(path("derived_dir") / "rolling_stationarity.csv")
    summ = pd.concat({c: dg.stationarity_summary(r, periods, band) for c, r in rs.items()}, names=["spread", "period"])
    save_table(summ, "03_stationarity_summary.csv")
    out["stationarity_summary"] = summ.astype(float).round(3).reset_index().to_dict("records")

    win_rows = {}
    for alt in cfg["robustness"]["window"]:
        for c in LABELS:
            r = rs[c] if alt == w else dg.rolling_stationarity(sp[c], window=alt)
            win_rows[(c, alt)] = {"adf_pass_rate": (r["adf_p"] < 0.05).mean(), "median_adf_p": r["adf_p"].median(),
                                  "hl_in_band_rate": r["half_life"].between(*band).mean(),
                                  "median_half_life": r.loc[np.isfinite(r["half_life"]), "half_life"].median()}
    win_tab = pd.DataFrame(win_rows).T.rename_axis(["spread", "window"])
    save_table(win_tab, "03_stationarity_by_window.csv")
    out["stationarity_by_window"] = win_tab.astype(float).round(3).reset_index().to_dict("records")

    last = sp.index[-1]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for i, c in enumerate(LABELS):
        ax[0].plot(rs[c].index, rs[c]["adf_p"], color=SERIES[i], lw=1, label=LABELS[c])
        hl = rs[c]["half_life"].replace(np.inf, np.nan)
        ax[1].plot(hl.index, hl, color=SERIES[i], lw=1, label=LABELS[c])
        ax[2].plot(rs[c].index, rs[c]["z_last"], color=SERIES[i], lw=1, label=LABELS[c])
    ax[0].axhline(0.05, color=CRITICAL, lw=1, ls="--", label="5%")
    ax[0].set_ylabel("ADF p-value")
    ax[0].set_title(f"Rolling {w}-day ADF p-value of each spread (shaded: stress windows)")
    ax[0].legend(loc="upper left", ncols=4)
    ax[1].axhspan(*band, color=SERIES[2], alpha=0.08)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Half-life (days)")
    ax[1].set_title("AR(1) half-life of the raw spread (green band: 5–120 days, the entry gate)")
    ax[2].axhline(0, color=REF_LINE, lw=0.7)
    for k in (-2, 2):
        ax[2].axhline(k, color=REF_LINE, lw=0.7, ls="--")
    ax[2].set_ylabel("z at month start")
    ax[2].set_title("Spread z-score against the trailing window (raw, not deseasonalised)")
    for a in ax:
        _shade_stress(a, cfg, last)
    fig.tight_layout()
    save(fig, "03_rolling_stationarity.png")

    # 5a. Engle-Granger: Brent on WTI, HO on RB ($/bbl)
    keep = ~sp["flag_apr2020"].astype(bool)
    eg = {"brent_on_wti": dg.eg_pair(sp["brent_bbl"], sp["cl_bbl"]),
          "brent_on_wti_excl_apr2020": dg.eg_pair(sp.loc[keep, "brent_bbl"], sp.loc[keep, "cl_bbl"]),
          "ho_on_rb": dg.eg_pair(sp["ho_bbl"], sp["rb_bbl"])}
    eg_tab = pd.DataFrame(eg).T
    save_table(eg_tab, "03_engle_granger.csv")
    out["engle_granger"] = eg_tab.astype(float).round(4).to_dict("index")
    reg = {"brent_wti": dg.rolling_eg(sp["brent_bbl"], sp["cl_bbl"], w), "ho_rb": dg.rolling_eg(sp["ho_bbl"], sp["rb_bbl"], w)}
    pd.concat(reg, names=["pair", "date"]).to_csv(path("derived_dir") / "rolling_eg.csv")
    out["rolling_eg"] = {k: {"eg_pass_rate": float((r["eg_p"] < 0.05).mean()),
                             "median_beta": float(r["beta"].median()),
                             "beta_range": [float(r["beta"].min()), float(r["beta"].max())]} for k, r in reg.items()}

    # 5b. Johansen on (CL, RB, HO) in $/bbl
    lv = sp[LEGS].rename(columns=lambda s: s.replace("_bbl", ""))
    lags = select_order(lv.to_numpy(), maxlags=10, deterministic="ci")
    jo = {"k_ar_diff=1": dg.johansen(lv, 1, targets=TARGETS),
          f"k_ar_diff={lags.aic} (AIC)": dg.johansen(lv, int(lags.aic), targets=TARGETS),
          "k_ar_diff=1, excl Apr 2020": dg.johansen(lv[keep], 1, targets=TARGETS)}
    jo_tab = pd.DataFrame(jo).T
    save_table(jo_tab, "03_johansen_full.csv")
    out["johansen_full"] = jo_tab.astype(float).round(4).to_dict("index")
    out["fixed_crack_adf_p"] = dg.adf_p(sp["crack321"])

    if not skip_slow:
        rj = dg.rolling_johansen(lv, window=w, k_ar_diff=1, targets=TARGETS)
        rj.to_csv(path("derived_dir") / "rolling_johansen.csv")
        out["rolling_johansen"] = {
            "rank_share": rj["rank"].value_counts(normalize=True).sort_index().round(3).to_dict(),
            "median_dist_crack321": float(rj["dist_crack321"].median()),
            "median_v1": {k: float(rj[f"v1_{k}"].median()) for k in ["cl", "rb", "ho"]},
        }
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax[0].step(rj.index, rj["rank"], where="post", color=SERIES[0], lw=1.2)
        ax[0].set_yticks([0, 1, 2, 3])
        ax[0].set_ylabel("Johansen rank (95%)")
        ax[0].set_title(f"Rolling {w}-day Johansen trace test on (CL, RB, HO): expected rank 2, one vector per product crack")
        for k, lab, col in [("dist_crack321", "3:2:1 crack", SERIES[0]), ("dist_ho_rb", "HO − RB", SERIES[2])]:
            ax[1].plot(rj.index, rj[k], color=col, lw=1, label=lab)
        ax[1].set_ylim(-0.02, 1.02)
        ax[1].set_ylabel("Distance (0 = inside)")
        ax[1].set_title("How far each fixed-weight spread lies from the estimated cointegration space")
        ax[1].legend(loc="upper left")
        for a in ax:
            _shade_stress(a, cfg, last)
        fig.tight_layout()
        save(fig, "03_rolling_johansen.png")

    # 4b. Seasonality and SARIMA on the monthly crack and HO-RB
    sar = {}
    for c in SEASONAL:
        m = ex.monthly_mean(sp[c])
        sar[c] = {"residuals": dg.sarima_residual_check(m)}
        if not skip_slow:
            fc = dg.sarima_vs_naive(m, train=60)
            fc.to_csv(path("derived_dir") / f"sarima_forecasts_{c}.csv")
            sar[c]["forecast"] = dg.forecast_summary(fc)
    out["sarima"] = sar
    if not skip_slow:
        fs = pd.DataFrame({c: sar[c]["forecast"] for c in SEASONAL}).T
        save_table(fs, "03_sarima_vs_naive.csv")
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
        for a, c in zip(ax, SEASONAL):
            fc = pd.read_csv(path("derived_dir") / f"sarima_forecasts_{c}.csv", index_col=0, parse_dates=True)
            for i, mname in enumerate(["rw", "snaive"]):
                adv = (fc[f"e_{mname}"] ** 2 - fc["e_sarima"] ** 2).cumsum()
                a.plot(adv.index, adv, color=SERIES[i + 1], lw=1, label=f"vs {'random walk' if mname == 'rw' else 'seasonal naive'}")
            a.axhline(0, color=REF_LINE, lw=0.8)
            a.set_title(f"{LABELS[c]}: cumulative squared-error advantage of SARIMA")
            a.set_ylabel("($/bbl)², rising = SARIMA better")
            a.legend(loc="upper left", fontsize=7.5)
        fig.tight_layout()
        save(fig, "03_sarima_vs_naive.png")
    return out


def main(skip_slow: bool = False) -> dict:
    use_style()
    cfg = load_config()
    _, sp = load_clean()
    res = {"source": "spot", "part2": part2(sp, cfg), "part3": part3(sp, cfg, skip_slow)}
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    (TAB_DIR / "analysis_summary.json").write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-slow", action="store_true")
    print(json.dumps(main(ap.parse_args().skip_slow), indent=2, default=float))
