"""EXPLORATORY: trend and carry on the Bloomberg futures -> output/exploratory/trend_carry/.

Usage:  .venv/bin/python scripts/run_trend_carry.py
Requires build_data.py (futures) and, for the descriptive correlation, run_backtest.py --source futures.
Rules: PREREGISTRATION_TREND_CARRY.md (registered 2026-10-28, after the binding mean-reversion verdict,
before any trend or carry result). Nothing here changes that verdict.
"""
from __future__ import annotations

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oilrv import trend_carry as TC
from oilrv.config import ROOT, load_config, path, set_source
from oilrv.metrics import subperiods, window_pnl
from oilrv.plotting import REF_LINE, SERIES, use_style

OUT = ROOT / "output" / "exploratory" / "trend_carry"
LABELS = {"cl": "WTI (CL)", "brent": "Brent (CO)", "rb": "RBOB (XB)", "ho": "ULSD (HO)",
          "crack321": "3-2-1 crack", "brent_wti": "Brent − WTI", "ho_rb": "HO − RB"}


def main() -> dict:
    use_style()
    cfg = load_config()
    set_source("futures")
    tc = cfg["exploratory_trend_carry"]
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    inst = TC.build_instruments(cfg=cfg)
    sig = TC.sigmas(inst, cfg)
    base = TC.run_sleeves(inst, cfg, sig=sig)
    L0, S0 = tc["trend"]["lookback_days"], tc["carry"]["slope"]

    # grid: lookbacks (carry at base) and slopes (trend at base)
    grid, grid_books = {}, {}
    for L in tc["trend"]["lookback_grid"]:
        r = base if L == L0 else TC.run_sleeves(inst, cfg, lookback=L, sig=sig)
        grid[("trend lookback", L)] = {"trend": TC.sharpe(r["trend"]), "combined": TC.sharpe(r["combined"])}
        grid_books.update({f"lookback={L}|trend": r["trend"], f"lookback={L}|combined": r["combined"]})
    for s in tc["carry"]["slope_grid"]:
        r = base if s == S0 else TC.run_sleeves(inst, cfg, slope=s, sig=sig)
        grid[("carry slope", s)] = {"carry": TC.sharpe(r["carry"]), "combined": TC.sharpe(r["combined"])}
        grid_books.update({f"slope={s}|carry": r["carry"], f"slope={s}|combined": r["combined"]})
    grid_t = pd.DataFrame(grid).T.rename_axis(["grid", "value"])
    grid_t.to_csv(OUT / "tables" / "grid.csv")
    pd.DataFrame(grid_books).to_csv(OUT / "tables" / "grid_daily_books.csv")

    g_comb = {f"{a}={b}": v for (a, b), v in grid_t["combined"].items()}
    g_trend = {f"lookback={b}": v for (a, b), v in grid_t["trend"].dropna().items()}
    g_carry = {f"slope={b}": v for (a, b), v in grid_t["carry"].dropna().items()}
    verdicts = {"combined": TC.verdict(base["combined"], base["changes"]["combined"], g_comb, cfg),
                "trend": TC.verdict(base["trend"], base["changes"]["trend"], g_trend, cfg),
                "carry": TC.verdict(base["carry"], base["changes"]["carry"], g_carry, cfg)}
    pd.DataFrame({k: {kk: (json.dumps(v, default=float) if isinstance(v, dict) else v) for kk, v in d.items()}
                  for k, d in verdicts.items()}).T.to_csv(OUT / "tables" / "gate_verdicts.csv")

    books = pd.DataFrame({k: base[k] for k in ["trend", "carry", "combined"]})
    books.to_csv(OUT / "tables" / "daily_books.csv")
    summ = pd.DataFrame({k: TC.summary(books[k]) for k in books}).T
    inst_rows = {}
    for key, d in base["instruments"].items():
        inst_rows[key] = {**TC.summary(d["pnl"]), "gross_total": float(d["gross"].sum()),
                          "cost_total": float(d["cost"].sum()), "roll_total": float(d["roll"].sum()),
                          "position_changes": TC.position_changes(d["target"]),
                          "share_days_long": float((d["pos"] > 0).mean())}
    inst_t = pd.DataFrame(inst_rows).T
    summ.to_csv(OUT / "tables" / "summary_books.csv")
    inst_t.to_csv(OUT / "tables" / "summary_instruments.csv")
    sub = pd.concat({k: subperiods(books[[k]], k, cfg["evaluation"]["sub_periods"]) for k in books},
                    names=["book", "period"])
    sub.to_csv(OUT / "tables" / "subperiods.csv")
    stress = pd.DataFrame({w: {k: window_pnl(books, k, a, b) for k in books}
                           for w, (a, b) in cfg["evaluation"]["stress_windows"].items()}).T
    stress.to_csv(OUT / "tables" / "stress_windows.csv")

    # descriptive: correlation with the registered mean-reversion research stream (futures, gated)
    mr = sum(pd.read_csv(path("derived_dir") / f"backtest_daily_fixed_{s}.csv", index_col=0, parse_dates=True)["pnl_usd"]
             for s in ["crack321", "brent_wti", "ho_rb"])
    both = books.join(mr.rename("mean_reversion"), how="inner")
    corr = both.corr().round(3)
    corr.to_csv(OUT / "tables" / "correlation_with_mean_reversion.csv")

    out = {"status": "EXPLORATORY (PREREGISTRATION_TREND_CARRY.md); does not change the binding verdict",
           "source": "futures", "start": str(base["start"].date()), "end": str(books.index[-1].date()),
           "verdicts": verdicts, "books": summ.astype(float).round(3).to_dict("index"),
           "instruments": inst_t[["sharpe", "total", "max_dd", "position_changes"]].astype(float).round(3).to_dict("index"),
           "subperiods": {f"{a}|{b}": v for (a, b), v in sub.astype(float).round(1).to_dict("index").items()},
           "stress_windows": stress.round(1).to_dict("index"),
           "corr_with_mean_reversion": corr["mean_reversion"].drop("mean_reversion").to_dict()}
    (OUT / "tables" / "summary.json").write_text(json.dumps(out, indent=2, default=float))

    # descriptive, not gated: every instrument's daily path, calendar years, drawdowns, today's book
    paths = TC.daily_paths(base)
    (path("derived_dir") / "exploratory").mkdir(exist_ok=True)
    paths.to_csv(path("derived_dir") / "exploratory" / "trend_carry_daily.csv", index=False)
    ann = TC.annual(books)
    ann.to_csv(OUT / "tables" / "annual.csv")
    dd = pd.concat({k: TC.drawdowns(books[k]) for k in books}, names=["book", "rank"])
    dd.to_csv(OUT / "tables" / "drawdowns.csv")
    now = TC.current_book(base, cfg)
    now.to_csv(OUT / "tables" / "current_book.csv")
    by_sleeve = paths.groupby("sleeve")[["gross", "cost", "roll", "pnl"]].sum() / paths.groupby("sleeve")[
        "instrument"].nunique().to_numpy()[:, None]      # a sleeve is the average of its instruments
    by_sleeve["cost_share_of_gross"] = (by_sleeve["cost"] + by_sleeve["roll"]) / by_sleeve["gross"]
    by_sleeve.to_csv(OUT / "tables" / "cost_breakdown.csv")

    plot_books(books, base, verdicts, cfg)
    plot_diagnostics(books, ann, grid_books, cfg)
    plot_signals(inst, base, cfg)
    return out


def shade_stress(a, cfg, last):
    for a0, b0 in cfg["evaluation"]["stress_windows"].values():
        a.axvspan(pd.Timestamp(a0), pd.Timestamp(b0) if b0 else last, color=REF_LINE, alpha=0.10, lw=0)


def plot_books(books, base, verdicts, cfg):
    tc = cfg["exploratory_trend_carry"]
    last = books.index[-1]
    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for i, k in enumerate(["trend", "carry", "combined"]):
        tag = "PASS" if verdicts[k]["PASS"] else "FAIL"
        ax[0].plot(books.index, books[k].cumsum() / 1e3, color=SERIES[i], lw=1.1 if k == "combined" else 0.9,
                   label=f"{k} (Sharpe {TC.sharpe(books[k]):.2f}, {tag})")
    for sleeve, a in [("trend", ax[1]), ("carry", ax[2])]:
        for i, n in enumerate(tc[sleeve]["instruments"]):
            d = base["instruments"][f"{sleeve}|{n}"]
            a.plot(d.index, d["pnl"].cumsum() / 1e3, color=SERIES[i % len(SERIES)], lw=0.8,
                   label=f"{LABELS[n]} ({TC.sharpe(d['pnl']):.2f})")
        a.set_title(f"{sleeve.capitalize()} sleeve by instrument: cumulative net P&L (\\$k, R = \\$6k/day each)")
    ax[0].set_title("EXPLORATORY trend and carry on futures: cumulative net P&L of the books (\\$k)")
    for a in ax:
        a.axhline(0, color=REF_LINE, lw=0.7)
        shade_stress(a, cfg, last)
        a.set_ylabel("\\$k")
        a.legend(loc="upper left", ncols=4, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "cumulative_pnl.png")
    plt.close(fig)


def plot_diagnostics(books, ann, grid_books, cfg):
    """Drawdown and rolling 1-year Sharpe of the books; calendar-year P&L; the grid's cumulative P&L."""
    last = books.index[-1]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for i, k in enumerate(books):
        cum = books[k].cumsum()
        ax[0, 0].plot(books.index, (cum - cum.cummax()) / 1e3, color=SERIES[i], lw=0.9, label=k)
        ax[0, 1].plot(books.index, TC.rolling_sharpe(books[k]), color=SERIES[i], lw=0.9, label=k)
    ax[0, 0].set_title("Drawdown from the running peak (\\$k)")
    ax[0, 1].set_title("Rolling 252-day net Sharpe")
    ax[0, 1].axhline(cfg["exploratory_trend_carry"]["gates"]["min_net_sharpe"], color=REF_LINE, lw=0.7, ls="--")
    years = ann.index.to_numpy()
    w = 0.27
    for i, k in enumerate(books):
        ax[1, 0].bar(years + (i - 1) * w, ann[("net", k)] / 1e3, width=w, color=SERIES[i], label=k)
    ax[1, 0].set_title(f"Net P&L by calendar year (\\$k; {years[-1]} to date)")
    ax[1, 0].set_xticks(years, [str(y)[2:] for y in years])
    for i, (k, s) in enumerate(grid_books.items()):        # one colour per setting; sleeve dashed, book solid
        book = k.endswith("|combined")
        ax[1, 1].plot(s.index, s.cumsum() / 1e3, lw=1.0 if book else 0.7, ls="-" if book else "--",
                      color=SERIES[(i // 2) % len(SERIES)], label=f"{k} ({TC.sharpe(s):.2f})")
    ax[1, 1].set_title("Grid (G4): cumulative net P&L (\\$k; dashed = sleeve, solid = combined)")
    for a in ax.flat:
        a.axhline(0, color=REF_LINE, lw=0.7)
        a.legend(loc="upper left", fontsize=7, ncols=2)
    for a in [ax[0, 0], ax[0, 1], ax[1, 1]]:
        shade_stress(a, cfg, last)
    fig.suptitle("EXPLORATORY trend and carry on futures: diagnostics (descriptive, not gated)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "diagnostics.png")
    plt.close(fig)


def plot_signals(inst, base, cfg):
    """Carry: the M1 - M2 slope of WTI and Brent; trend: each instrument's signal through time."""
    tc = cfg["exploratory_trend_carry"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [1, 1.2]})
    start = base["start"]
    for i, n in enumerate(tc["carry"]["instruments"]):
        f = inst[n].loc[start:]
        ax[0].plot(f.index, f["m1"] - f["m2"], color=SERIES[i], lw=0.8, label=f"{LABELS[n]} M1 − M2")
    ax[0].axhline(0, color=REF_LINE, lw=0.7)
    ax[0].set_ylim(-3, 3)
    ax[0].set_ylabel("\\$/bbl")
    ax[0].set_title("Carry signal: front-curve slope (> 0 backwardation → long; clipped to ±\\$3)")
    ax[0].legend(loc="upper left", fontsize=8)
    names = tc["trend"]["instruments"]
    grid = pd.DataFrame({n: base["instruments"][f"trend|{n}"]["signal"] for n in names}).T
    ax[1].imshow(grid.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="RdBu", vmin=-1.6, vmax=1.6,
                 extent=[mdates.date2num(grid.columns[0]), mdates.date2num(grid.columns[-1]), len(names) - 0.5, -0.5])
    ax[1].xaxis_date()
    ax[1].set_yticks(range(len(names)), [LABELS[n] for n in names])
    ax[1].grid(False)
    ax[1].set_title(f"Trend signal by instrument (blue = long, red = short; {tc['trend']['lookback_days']}-day lookback)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "signals.png")
    plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=float))
