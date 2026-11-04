"""DESCRIPTIVE: ACF/PACF and unit roots of the log-ratio, daily and MONTHLY, against the dollar spread
-> output/exploratory/ratio_monthly/.

Usage:  .venv/bin/python scripts/run_ratio_monthly.py
Requires build_data.py (futures). Series come from run_ratio_stationarity.load() (market log-ratio of the held
contracts, held-contract log-ratio return) and trend_carry.build_instruments (back-adjusted dollar spread, the
deck's basis).

Added 2026-11-04, after every registered and exploratory result was seen. Not a gate or a trading rule; it
changes no verdict.

Constructions:
- Monthly level: month-end value (last close of the calendar month). Not the monthly mean: averaging a random
  walk gives a lag-1 ACF of changes of about +0.25 (Working 1960), and on the log-ratio it flips the sign of the
  lag-1 ACF from negative to positive (`month_end_vs_mean.csv`).
- Monthly change: the sum of daily held log-ratio returns in the month (roll-clean first difference), or the
  month-end to month-end change of the market log-ratio (includes roll jumps).
- Main window 2010-01 to 2026-09 (201 month-ends, band 1.96/sqrt(n) = 0.138), as in run_monthly_acf.py;
  2007-26 is reported as a check for the unit roots.
- ACF/PACF standard errors are robust to volatility clustering (helpers from run_ar_acf_pacf.py); the figures
  draw the plain white-noise band for monthly data, as the deck's A1 figure does.
"""
from __future__ import annotations

import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import run_ratio_stationarity as RS  # noqa: E402
from oilrv import diagnostics as dg  # noqa: E402
from oilrv import trend_carry as TC  # noqa: E402
from oilrv.config import ROOT, load_config, set_source  # noqa: E402

warnings.filterwarnings("ignore")
R = RS.AR
OUT = ROOT / "output" / "exploratory" / "ratio_monthly"
LO, HI = "2010-01", "2026-09"
WINDOWS = {"2010-26": (LO, HI), "2007-26": (None, None)}
K_DAILY, K_MONTHLY, ALPHA = 20, 24, 0.05
LAB = {"crack321": "3-2-1 crack", "brent_wti": "Brent / WTI", "ho_rb": "HO / RB"}
COL = {"crack321": "#1f5fa8", "brent_wti": "#c4531a", "ho_rb": "#2a8a6e"}
OLD, OLD_EDGE = "#9a9a9a", "#555"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "axes.axisbelow": True, "grid.color": "#e6e6e6", "grid.linewidth": .6, "axes.titlesize": 10.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "savefig.dpi": 140,
                     "legend.frameon": False})


def month_end(s: pd.Series, lo=LO, hi=HI) -> pd.Series:
    return s.astype(float).resample("ME").last().loc[lo:hi].dropna()


def month_sum(s: pd.Series, lo=LO, hi=HI) -> pd.Series:
    return s.astype(float).resample("ME").sum(min_count=1).loc[lo:hi].dropna()


def acf_pacf(y: pd.Series, k: int) -> pd.DataFrame:
    v = y.dropna().to_numpy()
    t = R.acf_robust(v, k).join(R.pacf_robust(v, k), rsuffix="_pacf")
    t["acf_p"] = 2 * norm.sf(np.abs(t.acf / t.se_robust))
    t["pacf_p"] = 2 * norm.sf(np.abs(t.pacf / t.se_robust_pacf))
    return t.assign(n=len(v)).reset_index()


def bh(t: pd.DataFrame) -> int:
    return int(multipletests(np.r_[t.acf_p, t.pacf_p], alpha=ALPHA, method="fdr_bh")[0].sum())


# --------------------------------------------------------------------------- tables
def daily_table(D: dict) -> pd.DataFrame:
    rows = []
    for n, d in D.items():
        for form, s in {"log_ratio_level": d["lr"], "held_log_ratio_return": d["held"],
                        "market_log_ratio_change_ex_roll": d["dlr"].where(~d["rolled"])}.items():
            rows.append(acf_pacf(s, K_DAILY).assign(series=n, form=form))
    return pd.concat(rows, ignore_index=True)


def monthly_forms(d: dict) -> dict[str, pd.Series]:
    f = {"log_ratio_level": month_end(d["lr"]), "held_return": month_sum(d["held"]),
         "market_log_ratio_change": month_end(d["lr"]).diff()}
    if d["seasonal"]:
        f["log_ratio_level_ds"] = month_end(d["lr_ds"])
        h = f["held_return"]
        f["held_return_ds"] = h - h.groupby(h.index.month).transform("mean")
    return f


def monthly_tables(D: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ac, rows = [], []
    for n, d in D.items():
        for form, s in monthly_forms(d).items():
            y = s.dropna()
            t = acf_pacf(y, K_MONTHLY).assign(series=n, form=form)
            ac.append(t)
            row = {"series": n, "form": form, "n": len(y), "band": 1.96 / np.sqrt(len(y)),
                   "acf1": t.acf.iloc[0], "pacf1": t.pacf.iloc[0]}
            if "level" in form:
                b, hl = dg.ar1(y.to_numpy())
                u = R.unit_root(y, "c")
                row.update(ar1_b=b, half_life_months=hl, adf_p=u["adf_p"], pp_p=u["pp_p"], kpss_p=u["kpss_p"])
            else:
                ic = R.select_ar(y.to_numpy(), 12)
                r1, _ = R.fit_ar(y.to_numpy(), 1)
                lb = dg.ljung_box(y.to_numpy(), lags=(12, 24))
                row.update(p_bic=int(ic.bic.idxmin()), p_aic=int(ic.aic.idxmin()), ar1_phi=r1.params[1],
                           ar1_p=r1.pvalues[1], lb_robust_p_12=lb["robust_p_12"], lb_robust_p_24=lb["robust_p_24"],
                           bh_rejections=bh(t), bh_tests=2 * len(t))
            rows.append(row)
    return pd.concat(ac, ignore_index=True), pd.DataFrame(rows)


def unit_root_table(D: dict, inst: dict) -> pd.DataFrame:
    rows = []
    for win, (lo, hi) in WINDOWS.items():
        for n, d in D.items():
            ser = {"dollar_backadjusted": inst[n]["x"], "dollar_market": d["diff"], "log_ratio": d["lr"]}
            if d["seasonal"]:
                ser["log_ratio_ds"] = d["lr_ds"]
            cases = [(f, w, y) for f, s in ser.items()
                     for w, y in (("level", month_end(s, lo, hi)), ("change", month_end(s, lo, hi).diff().dropna()))]
            cases.append(("held_log_ratio_return", "change", month_sum(d["held"], lo, hi)))
            for form, what, y in cases:
                for reg in ("c", "ct"):
                    if what == "change" and reg == "ct":
                        continue
                    u = R.unit_root(y, reg)
                    rows.append({"window": win, "series": n, "form": form, "what": what, "regression": reg,
                                 "n": len(y), "adf_p": u["adf_p"], "pp_p": u["pp_p"], "kpss_p": u["kpss_p"]})
    t = pd.DataFrame(rows)
    t["stationary_all"] = (t.adf_p < ALPHA) & (t.pp_p < ALPHA) & (t.kpss_p >= ALPHA)
    return t


def end_vs_mean(D: dict) -> pd.DataFrame:
    rows = []
    for n, d in D.items():
        for how in ("last", "mean"):
            m = getattr(d["lr"].resample("ME"), how)().loc[LO:HI].dropna()
            c = m.diff().dropna()
            r1, _ = R.fit_ar(c.to_numpy(), 1)
            u = R.unit_root(m, "c")
            rows.append({"series": n, "how": "month_end" if how == "last" else "month_mean",
                         "change_acf1": R.acf_robust(c.to_numpy(), 1).acf[1], "change_ar1_p": r1.pvalues[1],
                         "level_acf1": R.acf_robust(m.to_numpy(), 1).acf[1], "level_adf_p": u["adf_p"],
                         "level_kpss_p": u["kpss_p"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures
def bars(ax, new: pd.Series, old: pd.Series, band, color, ylim, title, xlabel=None):
    if np.ndim(band):
        ax.fill_between(new.index, -band, band, color="#e3e3e3", lw=0, step="mid")
    else:
        ax.fill_between(np.arange(.5, len(new) + 1), -band, band, color="#e3e3e3", lw=0)
    ax.bar(new.index, new.values, color=color, width=.7, label="log-ratio")
    ax.plot(old.index, old.values, "o", ms=4.5, mfc="white", mec=OLD_EDGE, mew=1.1, label="dollar spread")
    ax.axhline(0, color="#444", lw=.6)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)


def fig_levels(D: dict, inst: dict) -> None:
    fig, axs = plt.subplots(3, 3, figsize=(12, 8.2))
    for j, n in enumerate(LAB):
        d, dol = D[n], inst[n]["x"].astype(float).dropna()
        a_new = R.acf_robust(d["lr"].dropna().to_numpy(), 252).acf
        a_old = R.acf_robust(dol.to_numpy(), 252).acf
        ax = axs[0, j]
        ax.plot(a_old.index, a_old.values, color=OLD, lw=1.8, label="dollar spread (back-adjusted)")
        ax.plot(a_new.index, a_new.values, color=COL[n], lw=1.8, label="log-ratio (market)")
        ax.axhline(0, color="#444", lw=.6)
        ax.set_ylim(-.2, 1.02)
        ax.set_title(f"{LAB[n]}: daily level ACF")
        ax.set_xlabel("lag (trading days)")
        yl, yo = month_end(d["lr"]), month_end(dol)
        band = 1.96 / np.sqrt(len(yl))
        bars(axs[1, j], R.acf_robust(yl.to_numpy(), K_MONTHLY).acf, R.acf_robust(yo.to_numpy(), K_MONTHLY).acf,
             band, COL[n], (-.35, 1.05), f"{LAB[n]}: monthly level ACF")
        bars(axs[2, j], R.pacf_robust(yl.to_numpy(), K_MONTHLY).pacf, R.pacf_robust(yo.to_numpy(), K_MONTHLY).pacf,
             band, COL[n], (-.35, 1.05), f"{LAB[n]}: monthly level PACF", "lag (months)")
    for i in range(3):
        axs[i, 0].set_ylabel("correlation")
    axs[0, 0].legend(loc="upper right", fontsize=8.5)
    axs[1, 0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Levels: the dollar spread barely decays; the log-ratio decays like AR(1)   "
                 f"(monthly: month-end {LO} to {HI}, shaded ±1.96/√n)", x=.01, ha="left", fontsize=10.5,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "acf_levels.png")
    plt.close(fig)


def fig_changes(D: dict, inst: dict) -> None:
    fig, axs = plt.subplots(3, 3, figsize=(12, 8.2))
    for j, n in enumerate(LAB):
        d, dol = D[n], inst[n]["x"].astype(float).dropna()
        a = R.acf_robust(d["held"].dropna().to_numpy(), K_DAILY)
        ao = R.acf_robust(dol.diff().dropna().to_numpy(), K_DAILY).acf
        bars(axs[0, j], a.acf, ao, 1.96 * a.se_robust, COL[n], (-.3, .3), f"{LAB[n]}: daily change ACF",
             "lag (trading days)")
        yl, yo = month_sum(d["held"]), month_end(dol).diff().dropna()
        band = 1.96 / np.sqrt(len(yl))
        bars(axs[1, j], R.acf_robust(yl.to_numpy(), K_MONTHLY).acf, R.acf_robust(yo.to_numpy(), K_MONTHLY).acf,
             band, COL[n], (-.4, .4), f"{LAB[n]}: monthly change ACF")
        bars(axs[2, j], R.pacf_robust(yl.to_numpy(), K_MONTHLY).pacf, R.pacf_robust(yo.to_numpy(), K_MONTHLY).pacf,
             band, COL[n], (-.4, .4), f"{LAB[n]}: monthly change PACF", "lag (months)")
    for i in range(3):
        axs[i, 0].set_ylabel("correlation")
    axs[0, 0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Changes: white noise daily; monthly lag 1 turns negative for the log-ratio   "
                 "(bars: held log-ratio return; daily band robust to clustering, monthly ±1.96/√n)",
                 x=.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "acf_changes.png")
    plt.close(fig)


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    D = RS.load()
    set_source("futures")
    inst = TC.build_instruments(cfg=load_config())

    daily = daily_table(D)
    daily.to_csv(OUT / "tables" / "daily_acf_pacf.csv", index=False)
    mac, msum = monthly_tables(D)
    mac.to_csv(OUT / "tables" / "monthly_acf_pacf.csv", index=False)
    msum.to_csv(OUT / "tables" / "monthly_summary.csv", index=False)
    ur = unit_root_table(D, inst)
    ur.to_csv(OUT / "tables" / "monthly_unit_roots.csv", index=False)
    evm = end_vs_mean(D)
    evm.to_csv(OUT / "tables" / "month_end_vs_mean.csv", index=False)
    fig_levels(D, inst)
    fig_changes(D, inst)

    pd.set_option("display.width", 200)
    for form in ("held_log_ratio_return", "market_log_ratio_change_ex_roll"):
        g = daily[daily.form == form]
        print(f"daily {form}: BH rejections {bh(g)} of {2 * len(g)}")
    print(msum.round(3).to_string(index=False))
    print(ur[(ur.window == "2010-26") & (ur.what == "level")].round(3).to_string(index=False))
    print(evm.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
