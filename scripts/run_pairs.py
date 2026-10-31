"""EXPLORATORY: cointegration pairs on the Bloomberg futures -> output/exploratory/pairs/.

Usage:  .venv/bin/python scripts/run_pairs.py
Requires build_data.py (futures) and, for the descriptive correlations, run_backtest.py --source futures
and run_trend_carry.py. Rules: PREREGISTRATION_PAIRS.md (registered 2026-10-31, after the binding verdict and
the trend and carry result, before any signal or P&L of this strategy). Nothing here changes that verdict.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oilrv import metrics as MT
from oilrv import pairs as P
from oilrv import trend_carry as TC
from oilrv.config import ROOT, load_config, path, set_source
from oilrv.metrics import sharpe, subperiods, window_pnl
from oilrv.plotting import REF_LINE, SERIES, use_style

OUT = ROOT / "output" / "exploratory" / "pairs"
LABELS = {"base": "Base: adjusted with trend (gated)", "no_gate": "No Engle–Granger gate",
          "adjusted": "Adjusted, no trend", "market": "Market-level signal"}


def main() -> dict:
    use_style()
    cfg = load_config()
    set_source("futures")
    px = cfg["exploratory_pairs"]
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    data = P.build_data(cfg)
    names = list(data["specs"])
    base = P.run_book(data, cfg)
    Z0, E0 = cfg["strategy"]["entry_z"], px["eg_p_max"]

    # grid (G4): entry |z| with the EG threshold at base, and the EG threshold with entry at base
    runs = {f"entry_z={z}": base if z == Z0 else P.run_book(data, cfg, entry_z=z) for z in px["grid"]["entry_z"]}
    runs.update({f"eg_p_max={e}": base if e == E0 else P.run_book(data, cfg, eg_p_max=e) for e in px["grid"]["eg_p_max"]})
    grid_book = {k: sharpe(r["book"]) for k, r in runs.items()}
    grid_pair = {n: {k: sharpe(r["pairs"][n]["daily"]["pnl_usd"]) for k, r in runs.items()} for n in names}
    grid_t = pd.DataFrame({"book": grid_book, **grid_pair}).rename_axis("setting")
    grid_t["round_trips"] = pd.Series({k: r["round_trips"] for k, r in runs.items()})
    grid_t.to_csv(OUT / "tables" / "grid.csv")

    # verdicts: the book is primary; each pair at full R, with the single-pair evidence rule
    verdicts = {"book": P.verdict(base["book"], base["round_trips"], grid_book, cfg)}
    pair_rows = {}
    for n in names:
        r = base["pairs"][n]
        verdicts[n] = P.verdict(r["daily"]["pnl_usd"], P.round_trips(r["trades"]), grid_pair[n], cfg)
        pair_rows[n] = MT.summarize(r["daily"], r["trades"])
        pair_rows[n]["gate_open_share"] = float(r["daily"]["gate_open"].mean())
        pair_rows[n]["counts_as_evidence"] = P.single_pair_evidence(n, verdicts[n], pair_rows[n]["sharpe_t"], cfg)
    pd.DataFrame({k: {kk: (json.dumps(v, default=float) if isinstance(v, dict) else v) for kk, v in d.items()}
                  for k, d in verdicts.items()}).T.to_csv(OUT / "tables" / "gate_verdicts.csv")
    pair_t = pd.DataFrame(pair_rows).T
    pair_t.to_csv(OUT / "tables" / "summary_pairs.csv")

    # comparisons and robustness: reported, not gated
    comps = {"base": base, **{m: P.run_book(data, cfg, mode=m) for m in px["comparisons"]}}
    comp_t = pd.DataFrame({m: {**TC.summary(r["book"]), "round_trips": r["round_trips"],
                               "y2022": window_pnl(r["book"].to_frame("p"), "p", *cfg["evaluation"]["stress_windows"]["ukraine"]),
                               "y2026": window_pnl(r["book"].to_frame("p"), "p", *cfg["evaluation"]["stress_windows"]["hormuz"]),
                               **{f"sharpe|{n}": sharpe(r["pairs"][n]["daily"]["pnl_usd"]) for n in names},
                               **{f"trips|{n}": P.round_trips(r["pairs"][n]["trades"]) for n in names}}
                           for m, r in comps.items()}).T
    comp_t.to_csv(OUT / "tables" / "comparisons.csv")
    rob = {("window", cfg["model"]["window"]): base, ("cost_usd_per_bbl", cfg["strategy"]["cost_usd_per_bbl"]): base}
    rob.update({("window", w): P.run_book(data, cfg, window=w) for w in px["robustness"]["window"]})
    rob.update({("cost_usd_per_bbl", c): P.run_book(data, cfg, cost_bbl=c) for c in px["robustness"]["cost_usd_per_bbl"]})
    rob_t = pd.DataFrame({k: {"sharpe": sharpe(r["book"]), "total": float(r["book"].sum()), "round_trips": r["round_trips"],
                              "start": str(r["book"].index[0].date())} for k, r in rob.items()}).T.sort_index()
    rob_t.index.names = ["setting", "value"]
    rob_t.to_csv(OUT / "tables" / "robustness.csv")

    # base run: book, sub-periods, stress, calendar years, drawdowns
    pnl = pd.DataFrame({n: base["pairs"][n]["daily"]["pnl_usd"] for n in names})
    books = pd.concat([base["book"].rename("book"), pnl], axis=1)
    books.to_csv(OUT / "tables" / "daily_books.csv")
    TC.annual(books).to_csv(OUT / "tables" / "annual.csv")
    sub = pd.concat({k: subperiods(books[[k]], k, cfg["evaluation"]["sub_periods"]) for k in books},
                    names=["book", "period"])
    sub.to_csv(OUT / "tables" / "subperiods.csv")
    stress = pd.DataFrame({w: {k: window_pnl(books, k, a, b) for k in books}
                           for w, (a, b) in cfg["evaluation"]["stress_windows"].items()}).T
    stress.to_csv(OUT / "tables" / "stress_windows.csv")
    pd.concat({k: TC.drawdowns(books[k]) for k in books}, names=["book", "rank"]).to_csv(OUT / "tables" / "drawdowns.csv")

    # base run: trades, direction, exits, carry
    trades = pd.concat([P.carry_split(base["pairs"][n]["trades"], cfg).assign(pair=n) for n in names], ignore_index=True)
    trades.to_csv(OUT / "tables" / "trades.csv", index=False)
    side = pd.concat({n: P.by_side(base["pairs"][n]["trades"]) for n in names}, names=["pair", "side"])
    side.to_csv(OUT / "tables" / "by_side.csv")
    pd.DataFrame({n: MT.exit_reasons(base["pairs"][n]["trades"]) for n in names}).fillna(0).astype(int).to_csv(
        OUT / "tables" / "exit_reasons.csv")
    carry = trades.groupby("pair")[["gross_usd", "carry_usd", "convergence_usd", "net_usd"]].sum().reindex(names)
    carry.loc["all"] = carry.sum()
    carry.to_csv(OUT / "tables" / "carry_split.csv")
    now = P.latest_state(base["pairs"], data, cfg)
    now.to_csv(OUT / "tables" / "latest_state.csv")

    # descriptive: gate-open share by calendar year; correlations with the other two strategies
    gate = pd.DataFrame({n: base["pairs"][n]["daily"]["gate_open"].astype(float) for n in names})
    gate.groupby(gate.index.year).mean().rename_axis("year").to_csv(OUT / "tables" / "gate_open_by_year.csv")
    mr = sum(pd.read_csv(path("derived_dir") / f"backtest_daily_fixed_{s}.csv", index_col=0, parse_dates=True)["pnl_usd"]
             for s in ["crack321", "brent_wti", "ho_rb"])
    tc = pd.read_csv(ROOT / "output" / "exploratory" / "trend_carry" / "tables" / "daily_books.csv", index_col=0,
                     parse_dates=True)["combined"]
    both = pd.concat([base["book"].rename("pairs"), mr.rename("mean_reversion"), tc.rename("trend_carry")], axis=1, join="inner")
    corr = both.corr().round(3)
    corr.to_csv(OUT / "tables" / "correlation.csv")

    (path("derived_dir") / "exploratory").mkdir(exist_ok=True)
    P.daily_paths(base["pairs"]).to_csv(path("derived_dir") / "exploratory" / "pairs_daily.csv", index=False)

    out = {"status": "EXPLORATORY (PREREGISTRATION_PAIRS.md); does not change the binding verdict",
           "source": "futures", "start": str(base["book"].index[0].date()), "end": str(base["book"].index[-1].date()),
           "verdicts": verdicts, "book": TC.summary(base["book"]), "round_trips": base["round_trips"],
           "pairs": pair_t.drop(columns=["top5_share_of_pnl"]).to_dict("index"),
           "comparisons": comp_t[["sharpe", "sharpe_t", "total", "max_dd", "round_trips", "y2022", "y2026"]].astype(float).round(3).to_dict("index"),
           "grid": grid_book, "carry_split": carry.round(0).to_dict("index"),
           "corr": corr["pairs"].drop("pairs").to_dict()}
    (OUT / "tables" / "summary.json").write_text(json.dumps(out, indent=2, default=float))

    plot_book(comps, books, verdicts, cfg)
    plot_pairs(base, names, verdicts)
    plot_attribution(carry.drop("all"), side, names)
    return out


def shade_stress(a, cfg, last):
    for a0, b0 in cfg["evaluation"]["stress_windows"].values():
        a.axvspan(pd.Timestamp(a0), pd.Timestamp(b0) if b0 else last, color=REF_LINE, alpha=0.10, lw=0)


def plot_book(comps, books, verdicts, cfg):
    """The book against its comparisons, then the base book by calendar year."""
    last = books.index[-1]
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})
    tag = "PASS" if verdicts["book"]["PASS"] else "FAIL"
    for i, (m, r) in enumerate(comps.items()):
        s = r["book"].cumsum() / 1e3
        ax[0].plot(s.index, s, color=SERIES[i], lw=2.0 if m == "base" else 1.0,
                   label=f"{LABELS[m]}: Sharpe {sharpe(r['book']):.2f}, {r['round_trips']} trades"
                         + (f" ({tag})" if m == "base" else ", not gated"))
    ax[0].axhline(0, color=REF_LINE, lw=0.7)
    shade_stress(ax[0], cfg, last)
    ax[0].set_ylabel("\\$k")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title("EXPLORATORY pairs on futures: cumulative net P&L of the six-pair book (\\$k; shaded = stress windows)")
    ann = books["book"].groupby(books.index.year).sum() / 1e3
    ax[1].set_axisbelow(True)
    ax[1].bar(ann.index, ann, color=SERIES[0], width=0.8)
    ax[1].axhline(0, color=REF_LINE, lw=0.7)
    ax[1].set_xticks(ann.index, [str(y)[2:] for y in ann.index])
    ax[1].set_ylabel("\\$k")
    ax[1].set_title(f"Base book: net P&L by calendar year (\\$k; {ann.index[-1]} to date)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "cumulative_pnl.png")
    plt.close(fig)


def plot_pairs(base, names, verdicts):
    """One panel per pair at full R, on one scale; grey marks days the gate was open."""
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for a, n in zip(ax.flat, names):
        r = base["pairs"][n]
        d = r["daily"]
        a.fill_between(d.index, 0, 1, where=d["gate_open"].to_numpy(dtype=bool), color=REF_LINE, alpha=0.18, lw=0,
                       transform=a.get_xaxis_transform())
        a.plot(d.index, d["pnl_usd"].cumsum() / 1e3, color=SERIES[0], lw=1.2)
        a.axhline(0, color=REF_LINE, lw=0.7)
        tag = "PASS" if verdicts[n]["PASS"] else "FAIL"
        a.set_title(f"{n.upper()}: Sharpe {sharpe(d['pnl_usd']):.2f}, {P.round_trips(r['trades'])} trades, {tag}", fontsize=10)
    for a in ax[:, 0]:
        a.set_ylabel("\\$k")
    fig.suptitle("Cumulative net P&L by pair at R = \\$6k/day (\\$k; grey = gate open)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "pairs.png")
    plt.close(fig)


def plot_attribution(carry, side, names):
    """Fitted carry against convergence, and long against short, by pair."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    y = np.arange(len(names))
    h = 0.38
    ax[0].barh(y - h / 2, carry["carry_usd"] / 1e3, h * 0.92, color=SERIES[0], label="Fitted carry")
    ax[0].barh(y + h / 2, carry["convergence_usd"] / 1e3, h * 0.92, color=SERIES[1], label="Convergence (gross − carry)")
    ax[0].set_title("Gross P&L split by pair (\\$k)")
    net = side["net"].unstack("side").reindex(index=names, columns=[1, -1]).fillna(0.0) / 1e3
    ax[1].barh(y - h / 2, net[1], h * 0.92, color=SERIES[0], label="Long the spread")
    ax[1].barh(y + h / 2, net[-1], h * 0.92, color=SERIES[1], label="Short the spread")
    ax[1].set_title("Net P&L by direction (\\$k)")
    ax[0].set_yticks(y, [n.upper() for n in names])
    ax[0].invert_yaxis()
    for a in ax:
        a.set_axisbelow(True)
        a.axvline(0, color=REF_LINE, lw=0.7)
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "attribution.png")
    plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=float))
