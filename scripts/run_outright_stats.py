"""DESCRIPTIVE: the statistical tests on the four outrights in logs, next to the three spreads
-> output/exploratory/outright_stats/.

Usage:  .venv/bin/python scripts/run_outright_stats.py
Requires build_data.py (futures). Levels are the log MARKET price of the held contract (2007-26 daily, 2010-26 month-end,
as the log-ratios are). Changes are the held log return, which excludes roll gaps; the monthly change is its month sum.

Added 2026-11-05, after every registered and exploratory result was seen. Not a gate or a trading rule; it changes no
verdict. Cointegration and VECM are spread-only, so they are not here.
"""
from __future__ import annotations

import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from arch.unitroot import ZivotAndrews
from statsmodels.stats.diagnostic import het_arch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import run_ratio_monthly as RM  # noqa: E402
import run_ratio_stationarity as RS  # noqa: E402
from oilrv import diagnostics as dg  # noqa: E402
from oilrv import model  # noqa: E402
from oilrv.config import ROOT, load_config, set_source  # noqa: E402
from oilrv.readout import garch_path  # noqa: E402

warnings.filterwarnings("ignore")
R = RS.AR
OUT = ROOT / "output" / "exploratory" / "outright_stats"
OUTRIGHTS = {"cl": "WTI", "brent": "Brent", "rb": "RBOB", "ho": "ULSD"}
SPREADS = {"crack321": "3-2-1 crack", "brent_wti": "Brent / WTI", "ho_rb": "HO / RB"}
SEASONAL = ["RBOB", "ULSD", "3-2-1 crack", "HO / RB"]
INK, MUTED, BAND, OUTRIGHT_COL, SPREAD_COL = "#1f2933", "#6b7280", "#dfe3e8", "#6b7280", "#0f6fc6"


def outrights() -> dict[str, dict[str, pd.Series]]:
    """Each outright on its own front month, as the trend sleeve holds it (`run_ar_acf_pacf.build_series`)."""
    adj, mkt = R.build_series(load_config())
    out = {}
    for k, label in OUTRIGHTS.items():
        a, m = adj[k].astype(float).dropna(), mkt[k].astype(float).dropna()
        ret = np.log1p(a.diff() / m.shift(1)).dropna()
        out[label] = {"level": np.log(m), "ret": ret, "dollar_market": m, "dollar_adj": a, "held_index": ret.cumsum()}
    return out


def residual_lb(y: pd.Series, lags: tuple) -> dict:
    """Ljung-Box, plain and robust to volatility clustering, on the AR(1) residuals of a level."""
    r, _ = R.fit_ar(y.to_numpy(), 1)
    return dg.ljung_box(r.resid, lags=lags, model_df=1)


def main() -> None:
    set_source("futures")
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    O, S = outrights(), RS.load()
    D = {**{k: {"level": v["level"], "ret": v["ret"]} for k, v in O.items()},
         **{SPREADS[k]: {"level": S[k]["lr"].dropna(), "ret": S[k]["held"].dropna()} for k in SPREADS}}

    # 1. unit roots: daily 2007-26, month-end 2010-26
    rows = []
    for name, d in D.items():
        for freq, lv, ch in (("daily", d["level"], d["ret"]), ("monthly", RM.month_end(d["level"]), RM.month_sum(d["ret"]))):
            for form, y, regs in (("log level", lv, ("c", "ct")), ("log return", ch, ("c",))):
                for reg in regs:
                    u = R.unit_root(y, reg)
                    rows.append({"series": name, "freq": freq, "form": form, "regression": reg, "n": len(y),
                                 "adf_p": u["adf_p"], "pp_p": u["pp_p"], "kpss_p": u["kpss_p"]})
    ur = pd.DataFrame(rows)
    ur["stationary_all"] = (ur.adf_p < .05) & (ur.pp_p < .05) & (ur.kpss_p >= .05)
    ur.to_csv(OUT / "tables" / "unit_roots.csv", index=False)

    # 2. slide-10 comparison for the outrights: dollar and log, half-life, Zivot-Andrews
    rows = []
    for name, v in O.items():
        row = {"series": name, "adf_adjusted_dollar": R.unit_root(v["dollar_adj"], "c")["adf_p"]}
        for tag, s in (("dollar", v["dollar_market"]), ("log", v["level"])):
            u = R.unit_root(s, "c")
            z = ZivotAndrews(s.to_numpy(), lags=12, trend="c", trim=0.15)
            row.update({f"adf_{tag}": u["adf_p"], f"pp_{tag}": u["pp_p"], f"kpss_{tag}": u["kpss_p"],
                        f"half_life_{tag}": dg.ar1(s.to_numpy())[1], f"za_{tag}": z.pvalue})
        row["adf_log_held_index"] = R.unit_root(v["held_index"], "c")["adf_p"]
        if name in SEASONAL:
            row.update(half_life_dollar_ds=dg.ar1(R.deseason(v["dollar_market"]).to_numpy())[1],
                       half_life_log_ds=dg.ar1(R.deseason(v["level"]).to_numpy())[1])
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "tables" / "outright_levels.csv", index=False)

    # ARIMA by AIC and BIC on the held log returns of each outright's own front month
    arima = []
    for name, v in O.items():
        arima += RS.arima_row(name, "log price, d = 1 (ARMA on held log returns)", v["ret"].to_numpy(), 0, d_label=1)[0]
    pd.DataFrame(arima).to_csv(OUT / "tables" / "arima.csv", index=False)

    # 3. ACF, PACF and AR order on the changes; AR(1) half-life on the levels
    ac, rows = [], []
    for name, d in D.items():
        for freq, lv, ch, k, maxlag in (("daily", d["level"], d["ret"], RM.K_DAILY, 20),
                                        ("monthly", RM.month_end(d["level"]), RM.month_sum(d["ret"]), RM.K_MONTHLY, 12)):
            t = RM.acf_pacf(ch, k).assign(series=name, freq=freq, form="log return")
            ac.append(t)
            ac.append(RM.acf_pacf(lv, k).assign(series=name, freq=freq, form="log level"))
            ic = R.select_ar(ch.to_numpy(), maxlag)
            r1, _ = R.fit_ar(ch.to_numpy(), 1)
            b, hl = dg.ar1(lv.to_numpy())
            rows.append({"series": name, "freq": freq, "n_ret": len(ch), "acf1_ret": t.acf.iloc[0], "pacf1_ret": t.pacf.iloc[0],
                         "ar1_phi_ret": r1.params[1], "ar1_p_ret": r1.pvalues[1], "p_bic_ret": int(ic.bic.idxmin()),
                         "p_aic_ret": int(ic.aic.idxmin()), "bh_rejections_ret": RM.bh(t), "bh_tests_ret": 2 * len(t),
                         "ar1_b_level": b, "half_life_level": hl})
    pd.concat(ac, ignore_index=True).to_csv(OUT / "tables" / "acf_pacf.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "tables" / "ar_summary.csv", index=False)

    # 4. Ljung-Box: changes as they are, and the AR(1) residuals of the levels
    rows = []
    for name, d in D.items():
        for freq, lv, ch, lags in (("daily", d["level"], d["ret"], (20,)), ("monthly", RM.month_end(d["level"]), RM.month_sum(d["ret"]), (12, 24))):
            a, b = dg.ljung_box(ch.to_numpy(), lags=lags), residual_lb(lv, lags)
            row = {"series": name, "freq": freq}
            for lag in lags:
                row.update({f"ret_plain_p_{lag}": a[f"lb_p_{lag}"], f"ret_robust_p_{lag}": a[f"robust_p_{lag}"],
                            f"level_ar1_plain_p_{lag}": b[f"lb_p_{lag}"], f"level_ar1_robust_p_{lag}": b[f"robust_p_{lag}"]})
            if name in SEASONAL and freq == "monthly":
                c = residual_lb(RM.month_end(R.deseason(d["level"])), lags)
                row.update({f"level_ar1_ds_robust_p_{lag}": c[f"robust_p_{lag}"] for lag in lags})
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "tables" / "ljung_box.csv", index=False)

    # 5. GARCH(1,1)-t on daily log returns in percent: median rolling fit in 2020-26, as the spread figures, and ARCH-LM on the full sample
    rows = []
    for name, d in D.items():
        pct = 100 * d["ret"]
        fits = []
        month = pct.index.to_period("M")
        for i in np.where(np.r_[True, month[1:] != month[:-1]])[0]:          # rolling 756 days, refit monthly, as in the deck
            if i >= 756 and pct.index[i].year >= 2020:
                f = model.fit_garch(pct.iloc[i - 756:i].to_numpy())
                if f["ok"]:
                    fits.append({"a": f["a"], "b": f["b"], "nu": f["nu"], "hl": np.log(.5) / np.log(f["a"] + f["b"])})
        g = pd.DataFrame(fits).median()
        sig = garch_path(pct.cumsum().rename(name), 756).shift(1)
        real = pct.reindex(sig.index)
        win = real.rolling(756).std().shift(1)
        q = pd.DataFrame({"g": np.log(sig ** 2) + real ** 2 / sig ** 2, "w": np.log(win ** 2) + real ** 2 / win ** 2}).dropna()
        dq = (q.g - q.w).to_numpy()
        t = sm.OLS(dq, np.ones(len(dq))).fit(cov_type="HAC", cov_kwds={"maxlags": 10}).tvalues[0]
        rows.append({"series": name, "a": g["a"], "b": g["b"], "nu": g["nu"], "fits_ok": len(fits),
                     "shock_half_life_days": g["hl"], "arch_lm_p10": het_arch(pct.to_numpy() - pct.mean(), nlags=10)[1],
                     "qlike_gain_vs_window_sd": -dq.mean(), "qlike_t": t, "n_oos": len(dq)})
    pd.DataFrame(rows).to_csv(OUT / "tables" / "garch.csv", index=False)

    fig_acf(D)
    fig_variance_ratios()


def acf_panels(ax, row: int, D: dict, get, k: int, ylabel: str, xlabel: str | None = None) -> None:
    """One row of ACF or PACF bars, a panel per series, spreads darker than outrights."""
    for j, (name, d) in enumerate(D.items()):
        a = ax[row, j]
        t = get(d)
        a.fill_between(t["lag"], -1.96 * t["se"], 1.96 * t["se"], color=BAND, lw=0, zorder=0, step="mid")
        a.bar(t["lag"], t["v"], width=.7, color=SPREAD_COL if name in SPREADS.values() else OUTRIGHT_COL, zorder=2)
        a.axhline(0, color=INK, lw=.6, zorder=3)
        a.set_xlim(0, k + 1)
        a.tick_params(labelsize=8, length=2, colors=MUTED)
        a.spines[["top", "right"]].set_visible(False)
        a.spines[["left", "bottom"]].set_color(MUTED)
        if row == 0:
            a.set_title(name, fontsize=10, color=INK, fontweight="bold", loc="left")
        if j == 0:
            a.set_ylabel(ylabel, fontsize=9, color=INK)
        if xlabel and j == 3:
            a.set_xlabel(xlabel, fontsize=9, color=INK)


def fig_acf(D: dict) -> None:
    """Daily log returns, then month-end log levels: ACF on top, PACF below."""
    for tag, get_y, k, ylim, xl in (("daily_returns", lambda d: d["ret"], RM.K_DAILY, (-.25, .25), "lag (trading days); grey = 95% band, robust to volatility clustering"),
                                    ("monthly_levels", lambda d: RM.month_end(d["level"]), RM.K_MONTHLY, (-.45, 1.05), "lag (months); grey = 95% band, robust to volatility clustering")):
        fig, ax = plt.subplots(2, 7, figsize=(12.2, 2.9), sharey=True, sharex=True)
        for row, col, se, lab, xlab in ((0, "acf", "se_robust", "ACF", None), (1, "pacf", "se_robust_pacf", "PACF", xl)):
            acf_panels(ax, row, D, lambda d: RM.acf_pacf(get_y(d), k).rename(columns={col: "v", se: "se"}), k, lab, xlab)
        ax[0, 0].set_ylim(*ylim)
        fig.tight_layout(pad=.4, w_pad=.5, h_pad=.3)
        fig.savefig(OUT / "figures" / f"acf_pacf_{tag}.png", dpi=220)
        plt.close(fig)


def fig_variance_ratios() -> None:
    """Variance ratio by horizon with a 95% band from the robust z, a panel per series: the band covers 1 at long horizons."""
    v = pd.read_csv(ROOT / "output" / "exploratory" / "ratio_stationarity" / "tables" / "variance_ratios.csv")
    v = v[(v.freq == "daily") & v.basis.isin(["held_log", "held_log_ratio"])]
    keys = {"WTI": "cl", "Brent": "brent", "RBOB": "rb", "ULSD": "ho", "3-2-1 crack": "crack321", "Brent / WTI": "brent_wti", "HO / RB": "ho_rb"}
    fig, ax = plt.subplots(1, 7, figsize=(12.2, 2.5), sharey=True)
    for a, (name, key) in zip(ax, keys.items()):
        t = v[v.series == key].sort_values("q")
        se = ((t.vr - 1) / t.z_robust).abs()
        col = SPREAD_COL if name in SPREADS.values() else OUTRIGHT_COL
        a.fill_between(t.q, t.vr - 1.96 * se, t.vr + 1.96 * se, color=col, alpha=.16, lw=0)
        a.plot(t.q, t.vr, color=col, lw=2, marker="o", ms=4)
        a.axhline(1, color=INK, lw=.7, ls="--")
        a.set_xscale("log")
        a.set_xticks([5, 21, 63, 252], ["5", "21", "63", "252"])
        a.minorticks_off()
        a.tick_params(labelsize=8, length=2, colors=MUTED)
        a.spines[["top", "right"]].set_visible(False)
        a.spines[["left", "bottom"]].set_color(MUTED)
        a.set_title(name, fontsize=10, color=INK, fontweight="bold", loc="left")
    ax[0].set_ylim(0, 2.2)
    ax[0].set_ylabel("variance ratio", fontsize=9, color=INK)
    ax[3].set_xlabel("horizon (trading days); 1 = random walk, shaded = 95% band", fontsize=9, color=INK)
    fig.tight_layout(pad=.4, w_pad=.5)
    fig.savefig(OUT / "figures" / "variance_ratio_bands.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
