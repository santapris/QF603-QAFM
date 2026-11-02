"""FIGURES for the GARCH slide: does the volatility forecast work, and does sizing on it help?

Usage:  PYTHONPATH=src python scripts/make_garch_figures.py
Reads the repo's own saved daily backtest output (data/derived/backtest_daily_fixed_*.csv,
which carries the `arch` GARCH forecast `sigma_next` and the positions) and the back-adjusted
spread series. Nothing is refitted.

Figures (output/exploratory/horizon_tests/figures/):
  garch_forecast_vs_realised.png  forecast, frozen-window-sd benchmark and realised vol
  garch_qlike_cumulative.png      cumulative QLIKE gain over time: broad or crisis-only?
  garch_units.png                 position size the rule implies, with the latest date marked
  garch_sizing_swap.png           same trades, two sizing rules: realised risk and drawdown

The sizing-swap figure keeps the gated strategy's entries and exits and changes only the unit
count (GARCH vs frozen window sd), gross of costs. It is NOT the ladder comparison in
06_ladder_summary.csv (that one also removes the blackout and logit filters), which needs the
full backtest rerun with statsmodels and arch.

Added 2026-11-02, after every registered and exploratory result was seen. Descriptive only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from oilrv import futures
from oilrv.config import ROOT

DER, OUT = ROOT / "data/derived", ROOT / "output/exploratory/horizon_tests"
TAB, FIG = OUT / "tables", OUT / "figures"
WINDOW, R = 756, 6000.0
SPREADS = {"crack321": ("3-2-1 crack", 3000.0), "brent_wti": ("Brent − WTI", 1000.0),
           "ho_rb": ("HO − RB", 1000.0)}
def fmt_dates(a, every=1):
    import matplotlib.dates as md
    a.xaxis.set_major_locator(md.YearLocator(every))
    a.xaxis.set_major_formatter(md.DateFormatter("%Y"))


C = {"garch": "#047857", "win": "#dc2626", "real": "#9ca3af"}


def load():
    fut = futures.build_futures()
    out = {}
    for k in SPREADS:
        d = pd.read_csv(DER / f"backtest_daily_fixed_{k}.csv", index_col=0, parse_dates=True)
        dx = fut[k].dropna().diff().dropna()
        idx = dx.index
        month = pd.Series(idx.to_period("M"), index=idx)
        starts = np.flatnonzero((month != month.shift()).to_numpy())
        sd = pd.Series(np.nan, index=idx)
        v = dx.to_numpy()
        for j, i0 in enumerate(starts):
            if i0 < WINDOW:
                continue
            i1 = starts[j + 1] if j + 1 < len(starts) else len(v)
            sd.iloc[i0:i1] = v[i0 - WINDOW:i0].std(ddof=1)
        # sigma_next on date t is the forecast for day t+1; align to the day it forecasts
        g = d["sigma_next"].shift(1)
        df = pd.DataFrame({"dx": dx, "g": g.reindex(idx), "w": sd}).dropna()
        out[k] = (df, d)
    return out


def qlike(x, s):
    s2 = s ** 2
    return np.log(s2) + x ** 2 / s2


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    data = load()
    print("QLIKE check against output/tables/04_vol_forecast_test.csv (garch / const):")
    for k, (df, _) in data.items():
        print(f"  {k}: n={len(df)}  garch {qlike(df.dx, df.g).mean():.3f}  "
              f"window sd {qlike(df.dx, df.w).mean():.3f}")

    # ---- 1. forecast vs realised ------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), sharey=False)
    for a, (k, (df, _)) in zip(ax, data.items()):
        w = df.loc["2019-10-01":"2022-12-31"]
        realised = df.dx.abs().rolling(21).mean().mul(np.sqrt(np.pi / 2)).loc[w.index]
        a.plot(w.index, realised, color=C["real"], lw=1.4, label="realised (21-day)")
        a.plot(w.index, w.w, color=C["win"], lw=1.6, drawstyle="steps-post",
               label="frozen window sd")
        a.plot(w.index, w.g, color=C["garch"], lw=1.3, label="GARCH forecast")
        a.set_title(SPREADS[k][0], fontweight="bold", fontsize=11.5)
        fmt_dates(a)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=.25, lw=.7)
    ax[0].set_ylabel("daily volatility of the spread (\\$/bbl)")
    ax[0].legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.suptitle("GARCH rises into the shock; the frozen window sd waits for the next refit",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(FIG / "garch_forecast_vs_realised.png", dpi=165, facecolor="white")

    # ---- 2. cumulative QLIKE gain ----------------------------------------------
    fig, a = plt.subplots(figsize=(10.5, 4.6))
    for (k, (df, _)), c in zip(data.items(), ["#2563eb", "#dc2626", "#059669"]):
        gain = (qlike(df.dx, df.w) - qlike(df.dx, df.g)).cumsum()
        a.plot(gain.index, gain, color=c, lw=1.8, label=SPREADS[k][0])
    a.axhline(0, color="#111", lw=.8)
    fmt_dates(a, 2)
    a.set_ylabel("cumulative QLIKE gain over the frozen sd\n(up = GARCH better)")
    a.legend(frameon=False, fontsize=10, loc="upper left")
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", alpha=.25, lw=.7)
    fig.suptitle("Is GARCH's forecasting win steady, or earned in a few crises?",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(FIG / "garch_qlike_cumulative.png", dpi=165, facecolor="white")

    # ---- 3. units implied by the rule -------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    rows = []
    for a, (k, (_, d)) in zip(ax, data.items()):
        name, M = SPREADS[k]
        s = d["sigma_next"].dropna()
        units = R / (s * M)
        pct = s.rank(pct=True)
        a.plot(units.index, units, color=C["garch"], lw=1.0)
        a.scatter([units.index[-1]], [units.iloc[-1]], color="#111", zorder=3, s=26)
        a.annotate(f"{units.index[-1]:%d %b}: {units.iloc[-1]:.1f} units\n"
                   f"vol at {pct.iloc[-1] * 100:.0f}th pct", (units.index[-1], units.iloc[-1]),
                   xytext=(-150, -52), textcoords="offset points", fontsize=9.5, fontweight="bold",
                   arrowprops=dict(arrowstyle="-", color="#555"))
        a.set_yscale("log")
        import matplotlib.ticker as mt
        a.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
        a.yaxis.set_minor_formatter(mt.NullFormatter())
        fmt_dates(a, 4)
        a.set_title(name, fontweight="bold", fontsize=11.5)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=.25, lw=.7)
        rows.append({"spread": name, "last_date": units.index[-1].date(),
                     "units": units.iloc[-1], "vol_pct": pct.iloc[-1]})
    ax[0].set_ylabel("units = \\$6,000 / (forecast vol × \\$ per unit move)", fontsize=9.5)
    fig.suptitle("The sizing rule in action: calm markets get more units, wild ones get fewer",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(FIG / "garch_units.png", dpi=165, facecolor="white")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # ---- 4. sizing swap on identical trades --------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.2), gridspec_kw={"height_ratios": [1, 1]})
    tab = []
    for j, (k, (df, d)) in enumerate(data.items()):
        name, M = SPREADS[k]
        side = d["side"]
        trade = (side != 0) & (side != side.shift())            # first day of each position
        tid = trade.cumsum().where(side != 0)
        # sigma used at entry: decision made the day before the fill
        sg = d["sigma_next"].shift(1)
        sw = df["w"].reindex(d.index).ffill()
        g_units = d["units"]
        w_units = pd.Series(np.nan, index=d.index)
        for t, grp in d[side != 0].groupby(tid[side != 0]):
            e = grp.index[0]
            w_units.loc[grp.index] = R / (sw.shift(1).loc[e] * M)
        pnl_g = (g_units * M * d["pnl_bbl"]).fillna(0.0)
        pnl_w = (w_units.fillna(0.0) * M * d["pnl_bbl"]).fillna(0.0)
        for lab, pl, c in (("frozen window sd", pnl_w, C["win"]), ("GARCH", pnl_g, C["garch"])):
            rv = pl.rolling(63).std() * np.sqrt(252)
            ax[0, j].plot(rv.index, rv / 1e3, color=c, lw=1.2, label=lab)
            eq = pl.cumsum()
            dd = eq - eq.cummax()
            ax[1, j].plot(dd.index, dd / 1e3, color=c, lw=1.6, label=lab)
            tab.append({"spread": name, "sizing": lab, "max_dd_usd": dd.min(),
                        "ann_vol_usd": pl.std() * np.sqrt(252), "total_usd": pl.sum()})
        ax[0, j].set_title(name, fontweight="bold", fontsize=11.5)
        for r in (0, 1):
            fmt_dates(ax[r, j], 4)
            ax[r, j].spines[["top", "right"]].set_visible(False)
            ax[r, j].grid(axis="y", alpha=.25, lw=.7)
    ax[0, 0].set_ylabel("rolling 63-day P&L vol, \\$k annualised")
    ax[1, 0].set_ylabel("drawdown from peak, \\$k")
    ax[0, 0].legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.suptitle("Same trades, two sizing rules (gross of costs)", fontsize=13.5,
                 fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(FIG / "garch_sizing_swap.png", dpi=165, facecolor="white")
    t = pd.DataFrame(tab)
    TAB.mkdir(parents=True, exist_ok=True)
    t.to_csv(TAB / "garch_sizing_swap.csv", index=False)
    print(t.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
