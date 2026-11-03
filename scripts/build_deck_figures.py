"""Figures for report/oil_deck.md -> report/deck_figures/.

Usage:  .venv/bin/python scripts/build_deck_figures.py
Requires build_data.py, run_backtest.py (futures), run_trend_carry.py, run_ar_acf_pacf.py and
run_monthly_acf.py outputs. Every figure is drawn from data/ or output/ tables, with plain titles and no
commentary inside the plot. Shaded bands are the stress windows in config.yaml.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oilrv.config import ROOT, load_config  # noqa: E402

OUT = ROOT / "report" / "deck_figures"
D, T = ROOT / "data", ROOT / "output"
LAB = {"crack321": "3-2-1 crack", "brent_wti": "Brent − WTI", "ho_rb": "HO − RB"}
COL = {"crack321": "#1f5fa8", "brent_wti": "#c4531a", "ho_rb": "#2a8a6e",
       "trend": "#4c78a8", "carry": "#e38b2c", "combined": "#222222"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": .6, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "figure.dpi": 100, "savefig.dpi": 140, "legend.frameon": False})
LAST = pd.Timestamp("2026-10-26")
WIN = {k: tuple(pd.Timestamp(x) if x else LAST for x in v)
       for k, v in load_config()["evaluation"]["stress_windows"].items()}
WIN_LAB = {"brent_wti_break": "2011–13", "covid": "2020", "ukraine": "2022", "hormuz": "2026"}


def shade(ax, legend=False):
    for k, (a, b) in WIN.items():
        ax.axvspan(a, b, color="#000", alpha=.06, lw=0, label=WIN_LAB[k] if legend else None)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


def fut():
    return pd.read_csv(D / "clean" / "futures_spreads.csv", index_col=0, parse_dates=True)


def fig_adjustment():
    f = fut()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(f.index, f["crack321_unadj"], color=COL["crack321"], lw=1, label="Market level of the held contracts")
    ax.plot(f.index, f["crack321"], color="#888", lw=1, label="Back-adjusted level")
    ax.axhline(0, color="#999", lw=.6)
    ax.set_ylabel("$/bbl")
    ax.set_title("3-2-1 crack, futures")
    ax.legend(loc="lower right")
    save(fig, "01_backadjusted_vs_market.png")


def fig_spreads():
    f = fut()
    fig, axs = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)
    for ax, n in zip(axs, LAB):
        ax.plot(f.index, f[f"{n}_unadj"], color=COL[n], lw=.9)
        shade(ax, legend=n == "crack321")
        ax.set_title(LAB[n])
        ax.set_ylabel("$/bbl")
    axs[0].legend(ncol=4, loc="upper left", fontsize=8)
    save(fig, "02_spreads_market_level.png")


def fig_acf(kind="daily"):
    """Daily: bands are 95% and robust to volatility clustering (se_robust, per lag).
    Monthly: the plain white-noise band 1.96/sqrt(n)."""
    if kind == "daily":
        a = pd.read_csv(T / "exploratory" / "ar_acf_pacf" / "tables" / "acf_pacf.csv")
        a = a[(a.form == "change") & (a.lag <= 20)]
        k, ttl, xl, name, ylim = 20, "daily changes", "lag (trading days)", "03_acf_pacf_daily.png", .15
        note = "shaded: 95% band robust to volatility clustering"

        def get(s, c):
            d = a[a.series == s].set_index("lag")
            return d[c], 1.96 * d["se_robust" if c == "acf" else "se_robust_pacf"]
    else:
        a = pd.read_csv(T / "exploratory" / "horizon_support" / "tables" / "monthly_acf_pacf.csv")
        a = a[(a.window == "2010-01 to 2026-09") & (a.how == "month-end")]
        k, ttl, xl, name, ylim = 24, "monthly changes, month-end", "lag (months)", "A1_acf_pacf_monthly.png", .4
        note = f"shaded: 95% band, ±{float(a.band.iloc[0]):.3f}"

        def get(s, c):
            d = a[a.series == s].set_index("lag")
            return d[c], d["band"]
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.2), sharey=True)
    for j, n in enumerate(LAB):
        for i, c in enumerate(["acf", "pacf"]):
            (v, band), ax = get(n, c), axs[i, j]
            ax.fill_between(v.index, -band, band, color="#d9d9d9", lw=0, step="mid")
            ax.bar(v.index, v.values, color=np.where(np.abs(v.values) > band.values, COL[n], "#a9b9d0"), width=.7)
            ax.axhline(0, color="#444", lw=.6)
            ax.set_title(f"{LAB[n]}: {c.upper()}")
            ax.set_ylim(-ylim, ylim)
            if i:
                ax.set_xlabel(xl)
    axs[0, 0].set_ylabel("correlation")
    axs[1, 0].set_ylabel("correlation")
    fig.suptitle(f"Autocorrelation of {ttl} ({note})", x=.01, ha="left", fontsize=10.5, fontweight="bold")
    save(fig, name)


def fig_rolling_eg():
    r = pd.read_csv(D / "derived" / "rolling_eg.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for pair, lab, c in [("brent_wti", "Brent vs WTI", COL["brent_wti"]), ("ho_rb", "HO vs RB", COL["ho_rb"])]:
        s = r[r.pair == pair]
        ax.plot(s.date, s.eg_p, color=c, lw=1.2, label=lab)
    ax.axhline(.05, color="#444", ls="--", lw=.8)
    ax.text(r.date.min(), .07, "5% level", fontsize=8)
    ax.set_ylabel("Engle–Granger p-value")
    ax.set_ylim(0, 1)
    ax.set_title("Rolling 756-day cointegration test, futures legs")
    ax.legend(loc="upper right")
    save(fig, "05_rolling_cointegration.png")


def fig_vol():
    f = fut()
    fig, axs = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for ax, n in zip(axs, LAB):
        v = f[n].diff().rolling(63).std()
        ax.plot(v.index, v, color=COL[n], lw=.9)
        shade(ax)
        ax.set_title(f"{LAB[n]}: standard deviation of daily changes, trailing 63 days")
        ax.set_ylabel("$/bbl")
    save(fig, "06_rolling_volatility.png")


def fig_mr_pnl():
    ladder = pd.read_csv(T / "tables" / "06_ladder_summary.csv")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for n in LAB:
        d = pd.read_csv(D / "derived" / f"backtest_daily_fixed_{n}.csv", index_col=0, parse_dates=True)
        net = d["pnl_usd"].cumsum() / 1e3   # pnl_usd is already net of costs
        ref = ladder[(ladder.key == f"fixed|{n}") & (ladder.step == "logit_filter")]["total"].iloc[0] / 1e3
        assert abs(net.iloc[-1] - ref) < 1, (n, net.iloc[-1], ref)
        ax.plot(net.index, net, color=COL[n], lw=1.1, label=LAB[n])
    ax.axhline(0, color="#999", lw=.6)
    shade(ax, legend=True)
    ax.set_ylabel("cumulative net P&L, $k")
    ax.set_title("Mean reversion on the three spreads, futures")
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(.5, -.1), fontsize=8)
    save(fig, "07_mean_reversion_pnl.png")


def fig_books():
    b = pd.read_csv(T / "exploratory" / "trend_carry" / "tables" / "daily_books.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for c in ["trend", "carry", "combined"]:
        ax.plot(b.index, b[c].cumsum() / 1e3, color=COL[c], lw=1.6 if c == "combined" else 1.1, label=c.capitalize())
    ax.axhline(0, color="#999", lw=.6)
    shade(ax, legend=True)
    ax.set_ylabel("cumulative net P&L, $k")
    ax.set_title("Trend, carry and the 50/50 book, futures")
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(.5, -.1), fontsize=8)
    save(fig, "08_trend_carry_pnl.png")


def fig_annual():
    a = pd.read_csv(T / "exploratory" / "trend_carry" / "tables" / "annual.csv", header=[0, 1], index_col=0)
    a = a.drop(index="year", errors="ignore")
    yr = a.index.astype(int)
    v = a[("net", "combined")].astype(float).to_numpy() / 1e3
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(yr, v, color=np.where(v >= 0, COL["combined"], "#b5b5b5"), width=.7)
    ax.axhline(0, color="#444", lw=.6)
    ax.set_ylabel("net P&L, $k")
    ax.set_title("50/50 book, net P&L by calendar year (2026 to 26 Oct)")
    ax.set_xticks(yr)
    ax.set_xticklabels([str(y)[2:] for y in yr])
    ax.set_xlabel("year (20xx)")
    save(fig, "09_annual_pnl.png")


if __name__ == "__main__":
    fig_adjustment()
    fig_spreads()
    fig_acf("daily")
    fig_rolling_eg()
    fig_vol()
    fig_mr_pnl()
    fig_books()
    fig_annual()
    fig_acf("monthly")
