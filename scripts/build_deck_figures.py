"""Figures for report/oil_deck.md -> report/deck_figures/.

Usage:  .venv/bin/python scripts/build_deck_figures.py
Requires build_data.py, run_backtest.py (futures), run_trend_carry.py, run_ar_acf_pacf.py and
run_monthly_acf.py and run_ratio_monthly.py outputs. Every figure is drawn from data/ or output/ tables, with plain titles and no
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


RS = T / "exploratory" / "ratio_stationarity" / "tables"
ERA_BOUNDS = [("2007-01-01", "2010-12-31"), ("2011-01-01", "2013-12-31"), ("2014-01-01", "2019-12-31"), ("2020-01-01", None)]


def fig_diff_vs_logratio():
    """Market dollar gap against market log-ratio, with era means of the log-ratio (k moves between eras)."""
    f = fut()
    cl, rb, ho = f["crack321__cl_bbl_unadj"], f["crack321__rb_bbl_unadj"], f["crack321__ho_bbl_unadj"]
    s = {"crack321": (f["crack321_unadj"], np.log((2 * rb + ho) / (3 * cl))),
         "brent_wti": (f["brent_wti_unadj"], np.log(f["brent_wti__brent_bbl_unadj"] / f["brent_wti__cl_bbl_unadj"])),
         "ho_rb": (f["ho_rb_unadj"], np.log(f["ho_rb__ho_bbl_unadj"] / f["ho_rb__rb_bbl_unadj"]))}
    fig, ax = plt.subplots(2, 3, figsize=(11, 5), sharex=True)
    for j, n in enumerate(LAB):
        diff, lr = s[n]
        ax[0, j].plot(diff.index, diff, color="#999", lw=.8)
        ax[0, j].set_title(f"{LAB[n]}: dollar gap")
        ax[1, j].plot(lr.index, lr, color=COL[n], lw=.8)
        for a, b in ERA_BOUNDS:
            x = lr.loc[a:b]
            ax[1, j].hlines(x.mean(), x.index[0], x.index[-1], color="#000", lw=1.5)
        ax[1, j].set_title(f"{LAB[n]}: log-ratio, era means")
    ax[0, 0].set_ylabel("$/bbl")
    ax[1, 0].set_ylabel("log(A / B)")
    save(fig, "10_difference_vs_log_ratio.png")


def fig_kpss_false_alarms():
    k = pd.read_csv(RS / "kpss_size.csv")
    k["lab"] = k.series.map(lambda x: {"crack321": "Crack", "brent_wti": "Brent/WTI", "ho_rb": "HO/RB"}[x]) + chr(10) + np.where(
        k.form.str.startswith("log"), "log-ratio", "dollar gap")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(range(len(k)), k.kpss_rejects_if_truly_stationary * 100,
           color=np.where(k.form.str.startswith("log"), "#2a78d6", "#b5b5b5"), width=.7)
    ax.axhline(5, color="#444", ls="--", lw=.8)
    ax.text(len(k) - .5, 7, "5% (what a good test gives)", fontsize=8, ha="right")
    ax.set_xticks(range(len(k)))
    ax.set_xticklabels(k.lab, fontsize=8)
    ax.set_ylabel("% of stationary series rejected")
    ax.set_title("KPSS on series we know are stationary, at each spread's own persistence")
    save(fig, "11_kpss_false_alarms.png")


def fig_variance_ratio():
    v = pd.read_csv(RS / "variance_ratios.csv")
    v = v[v.freq == "daily"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for a, group, title, col in [(ax[0], {"crack321": "3-2-1 crack", "brent_wti": "Brent / WTI", "ho_rb": "HO / RB"},
                                    "Spreads: held log-ratio return", None),
                                 (ax[1], {"cl": "WTI", "brent": "Brent", "rb": "RBOB", "ho": "ULSD"},
                                  "Outrights: held log return", None)]:
        for i, (k, lab) in enumerate(group.items()):
            s = v[(v.series == k) & v.basis.isin(["held_log_ratio", "held_log"])].sort_values("q")
            a.plot(s.q, s.vr, marker="o", ms=3, lw=1.3, color=["#1f5fa8", "#c4531a", "#2a8a6e", "#e38b2c"][i], label=lab)
        a.axhline(1, color="#444", lw=.8)
        a.set_xscale("log")
        a.set_xticks([5, 21, 63, 126, 252])
        a.set_xticklabels(["5", "21", "63", "126", "252"])
        a.set_xlabel("horizon (trading days)")
        a.set_title(title)
        a.legend(fontsize=8)
    ax[0].set_ylabel("variance ratio (1 = random walk)")
    save(fig, "12_variance_ratio_profile.png")


def fig_ratio_mr_pnl():
    """Cumulative net P&L of the pre-registered log-ratio mean reversion: the three spreads and the book."""
    b = pd.read_csv(T / "exploratory" / "ratio_mr" / "tables" / "daily_books.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for n in LAB:
        ax.plot(b.index, b[n].cumsum() / 1e3, color=COL[n], lw=1.0, label=LAB[n] + " (full risk)")
    ax.plot(b.index, b["book"].cumsum() / 1e3, color=COL["combined"], lw=1.8, label="Three-spread book")
    ax.axhline(0, color="#999", lw=.6)
    shade(ax, legend=True)
    ax.set_ylabel("cumulative net P&L, $k")
    ax.set_title("Log-ratio mean reversion, futures (pre-registered 4 Nov 2026)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.1), fontsize=8)
    save(fig, "13_ratio_mean_reversion_pnl.png")


def fig_horizon_vs_hold():
    """Where the log-ratio reverts (variance ratio by horizon) against how long the trades were held, and the book Sharpe by
    decision interval."""
    v = pd.read_csv(RS / "variance_ratios.csv")
    v = v[(v.basis == "held_log_ratio") & (v.freq == "daily")]
    sl = T / "exploratory" / "ratio_slow" / "tables"
    prof = pd.read_csv(sl / "decision_interval_profile.csv", index_col=0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    for n in LAB:
        s = v[v.series == n].sort_values("q")
        ax[0].plot(s.q, s.vr, marker="o", ms=3, lw=1.3, color=COL[n], label=LAB[n])
    ax[0].axhline(1, color="#444", lw=.8)
    for col, lab, c in [("daily (reference)", "daily decisions", "#1f5fa8"), ("monthly (base)", "monthly decisions", "#c4531a")]:
        h = [prof.loc[col, f"avg_hold|{n}"] for n in LAB]
        ax[0].axvspan(min(h), max(h), color=c, alpha=.15, lw=0, label=f"average hold, {lab}")
    ax[0].set_xscale("log")
    ax[0].set_xticks([5, 21, 63, 126, 252])
    ax[0].set_xticklabels(["5", "21", "63", "126", "252"])
    ax[0].set_xlabel("horizon (trading days)")
    ax[0].set_ylabel("variance ratio of log-ratio return")
    ax[0].set_title("Where the ratio reverts, and how long we held")
    ax[0].legend(fontsize=7, loc="lower left")
    names = ["daily (reference)", "monthly (base)", "quarterly"]
    sharpe_ = [prof.loc[k, "sharpe"] for k in names]
    ax[1].bar(range(3), sharpe_, color=["#1f5fa8", "#c4531a", "#b5b5b5"], width=.6)
    for i, k in enumerate(names):
        ax[1].text(i, max(sharpe_[i], 0) + .008, f"{int(prof.loc[k, 'round_trips'])} trades", ha="center", fontsize=8)
    ax[1].set_ylim(-.22, .10)
    ax[1].axhline(0, color="#444", lw=.8)
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels(["daily", "monthly", "quarterly"])
    ax[1].set_ylabel("book net Sharpe")
    ax[1].set_title("Slowing the decisions")
    save(fig, "14_horizon_vs_hold.png")


def fig_moving_anchor():
    """The market log-ratio, the filtered anchor (local level + AR(1)) and the trailing 756-day mean the registered z uses."""
    s = pd.read_csv(RS / "moving_anchor_series.csv", parse_dates=["date"])
    f = fut()
    raw = {"brent_wti": np.log(f["brent_wti__brent_bbl_unadj"] / f["brent_wti__cl_bbl_unadj"]),
           "ho_rb": np.log(f["ho_rb__ho_bbl_unadj"] / f["ho_rb__rb_bbl_unadj"])}
    raw["ho_rb"] = raw["ho_rb"] - raw["ho_rb"].groupby(raw["ho_rb"].index.month).transform("mean") + raw["ho_rb"].mean()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, n in zip(ax, ["brent_wti", "ho_rb"]):
        y = raw[n]
        k = s[s.series == n].set_index("date")["k_filtered"]
        a.plot(y.index, y, color="#b5b5b5", lw=.7, label="log-ratio")
        a.plot(k.index, np.log(k), color=COL[n], lw=1.6, label="filtered anchor, log k")
        a.plot(y.index, y.shift(1).rolling(756).mean(), color="#000", lw=1.2, ls="--", label="756-day mean (registered signal)")
        a.set_title(LAB[n].replace(" − ", " / "))
        a.set_ylim(-.12, .45 if n == "brent_wti" else .6)
    ax[0].set_ylabel("log(A / B)")
    ax[0].legend(fontsize=7, loc="upper right")
    save(fig, "15_moving_anchor.png")


def fig_model_worlds():
    """The registered rule simulated on four model worlds against what it earned, and G1 power."""
    w = pd.read_csv(T / "exploratory" / "gate_power" / "model_worlds.csv")
    pw = pd.read_csv(T / "exploratory" / "gate_power" / "g1_power.csv")
    names = list(LAB)
    cols = {"A": "#9bb9e0", "B": "#2a78d6", "C": "#eb6834", "D": "#1baf7a"}
    labs = {"A": "A  fixed k, known", "B": "B  fixed k, estimated", "C": "C  moving k, estimated", "D": "D  moving k, anchor known"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    for j, wd in enumerate("ABCD"):
        v = w[w.world == wd].set_index("spread").reindex(names)
        ax[0].bar(np.arange(3) + (j - 1.5) * .2, v.mean_sharpe, .18, color=cols[wd], label=labs[wd])
    real = w[w.world == "C"].set_index("spread").reindex(names).realised_daily_decisions
    ax[0].scatter(np.arange(3), real, marker="D", color="#000", zorder=5, label="realised (daily decisions)")
    ax[0].axhline(.5, color="#444", ls="--", lw=.8)
    ax[0].axhline(0, color="#999", lw=.6)
    ax[0].set_xticks(np.arange(3))
    ax[0].set_xticklabels([LAB[n].replace(" − ", " / ") for n in names])
    ax[0].set_ylabel("net Sharpe of the registered rule")
    ax[0].set_title("The rule on four model worlds")
    ax[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.1), ncol=3)
    ax[1].plot(pw.true_sharpe, pw.p_pass_G1, marker="o", ms=4, color="#2a78d6")
    ax[1].axhline(.5, color="#999", lw=.6)
    ax[1].set_xlabel("true Sharpe")
    ax[1].set_ylabel("probability of passing G1 (> 0.5)")
    ax[1].set_title("G1 over 16.7 years")
    save(fig, "16_model_worlds.png")


def fig_ratio_monthly():
    """Monthly ACF/PACF of the log-ratio against the dollar spread, drawn by run_ratio_monthly.py (they need the
    daily series, not a table), copied here under the deck's numbering."""
    import shutil
    src = T / "exploratory" / "ratio_monthly" / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    for a, b in [("acf_levels.png", "17_monthly_acf_levels.png"), ("acf_changes.png", "18_monthly_acf_changes.png")]:
        shutil.copyfile(src / a, OUT / b)
        print("wrote", b)


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
    fig_diff_vs_logratio()
    fig_kpss_false_alarms()
    fig_variance_ratio()
    fig_ratio_mr_pnl()
    fig_horizon_vs_hold()
    fig_moving_anchor()
    fig_model_worlds()
    fig_ratio_monthly()
