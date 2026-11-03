"""EXPLORATORY: log-ratio mean reversion on the Bloomberg futures -> output/exploratory/ratio_mr/.

Usage:  .venv/Scripts/python.exe scripts/run_ratio_mr.py
Requires build_data.py (futures), run_backtest.py --source futures, run_trend_carry.py and run_pairs.py for the descriptive
correlations. Rules: PREREGISTRATION_RATIO.md (registered 2026-11-04, after the binding verdict, the trend and carry and pairs
results and the ratio stationarity study, before any signal or P&L of this strategy). Nothing here changes the binding verdict.
Run once; every spread, grid point and comparison is reported.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oilrv import metrics as MT
from oilrv import pairs as P
from oilrv import ratio as R
from oilrv import trend_carry as TC
from oilrv.config import ROOT, load_config, path, set_source
from oilrv.metrics import sharpe, subperiods, window_pnl
from oilrv.plotting import REF_LINE, SERIES, use_style

OUT = ROOT / "output" / "exploratory" / "ratio_mr"
LAB = {"crack321": "3-2-1 crack", "brent_wti": "Brent / WTI", "ho_rb": "HO / RB"}
MODE_LAB = {"base": "Base: log-ratio signal (gated)", "dollar_market": "Dollar gap on market prices",
            "no_gate": "No half-life gate"}


def main() -> dict:
    use_style()
    cfg = load_config()
    set_source("futures")
    rx = cfg["exploratory_ratio"]
    T = OUT / "tables"
    T.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    data = R.build_data(cfg)
    names = list(data["specs"])
    base = R.run_book(data, cfg)
    Z0 = cfg["strategy"]["entry_z"]

    # G4 grid: entry |z|
    runs = {f"entry_z={z}": base if z == Z0 else R.run_book(data, cfg, entry_z=z) for z in rx["grid"]["entry_z"]}
    grid_book = {k: sharpe(r["book"]) for k, r in runs.items()}
    grid_sp = {n: {k: sharpe(r["spreads"][n]["daily"]["pnl_usd"]) for k, r in runs.items()} for n in names}
    grid_t = pd.DataFrame({"book": grid_book, **grid_sp}).rename_axis("setting")
    grid_t["round_trips"] = pd.Series({k: r["round_trips"] for k, r in runs.items()})
    grid_t.to_csv(T / "grid.csv")

    # verdicts: the book is primary; each spread at full R, with the single-spread evidence rule
    verdicts = {"book": R.verdict(base["book"], base["round_trips"], grid_book, cfg)}
    rows = {}
    for n in names:
        r = base["spreads"][n]
        verdicts[n] = R.verdict(r["daily"]["pnl_usd"], R.round_trips(r["trades"]), grid_sp[n], cfg)
        rows[n] = MT.summarize(r["daily"], r["trades"])
        rows[n]["gate_open_share"] = float(r["daily"]["gate_open"].mean())
        rows[n]["counts_as_evidence"] = R.single_spread_evidence(verdicts[n], rows[n]["sharpe_t"], cfg)
    pd.DataFrame({k: {kk: (json.dumps(v, default=float) if isinstance(v, dict) else v) for kk, v in d.items()}
                  for k, d in verdicts.items()}).T.to_csv(T / "gate_verdicts.csv")
    sp_t = pd.DataFrame(rows).T
    sp_t.to_csv(T / "summary_spreads.csv")

    # comparisons and robustness: reported, not gated
    comps = {"base": base, **{m: R.run_book(data, cfg, mode=m) for m in rx["comparisons"]}}
    stress = cfg["evaluation"]["stress_windows"]
    comp_t = pd.DataFrame({m: {**TC.summary(r["book"]), "round_trips": r["round_trips"],
                               "y2022": window_pnl(r["book"].to_frame("p"), "p", *stress["ukraine"]),
                               "y2026": window_pnl(r["book"].to_frame("p"), "p", *stress["hormuz"]),
                               **{f"sharpe|{n}": sharpe(r["spreads"][n]["daily"]["pnl_usd"]) for n in names},
                               **{f"trips|{n}": R.round_trips(r["spreads"][n]["trades"]) for n in names}}
                           for m, r in comps.items()}).T
    comp_t.to_csv(T / "comparisons.csv")
    rob = {("window", cfg["model"]["window"]): base, ("exit_z", cfg["strategy"]["exit_z"]): base,
           ("cost_usd_per_bbl", cfg["strategy"]["cost_usd_per_bbl"]): base}
    rob.update({("window", w): R.run_book(data, cfg, window=w) for w in rx["robustness"]["window"]})
    rob.update({("exit_z", e): R.run_book(data, cfg, exit_z=e) for e in rx["robustness"]["exit_z"]})
    rob.update({("cost_usd_per_bbl", c): R.run_book(data, cfg, cost_bbl=c) for c in rx["robustness"]["cost_usd_per_bbl"]})
    rob_t = pd.DataFrame({k: {"sharpe": sharpe(r["book"]), "total": float(r["book"].sum()), "round_trips": r["round_trips"],
                              "start": str(r["book"].index[0].date())} for k, r in rob.items()}).T.sort_index()
    rob_t.index.names = ["setting", "value"]
    rob_t.to_csv(T / "robustness.csv")

    # base run: book, sub-periods, stress, calendar years, drawdowns
    pnl = pd.DataFrame({n: base["spreads"][n]["daily"]["pnl_usd"] for n in names})
    books = pd.concat([base["book"].rename("book"), pnl], axis=1)
    books.to_csv(T / "daily_books.csv")
    TC.annual(books).to_csv(T / "annual.csv")
    sub = pd.concat({k: subperiods(books[[k]], k, cfg["evaluation"]["sub_periods"]) for k in books}, names=["book", "period"])
    sub.to_csv(T / "subperiods.csv")
    pd.DataFrame({w: {k: window_pnl(books, k, a, b) for k in books} for w, (a, b) in stress.items()}).T.to_csv(
        T / "stress_windows.csv")
    pd.concat({k: TC.drawdowns(books[k]) for k in books}, names=["book", "rank"]).to_csv(T / "drawdowns.csv")

    # base run: trades, direction, exits, roll attribution
    trades = pd.concat([R.roll_attribution(base["spreads"][n], data["rolled"][n]).assign(spread=n) for n in names],
                       ignore_index=True)
    trades.to_csv(T / "trades.csv", index=False)
    pd.concat({n: P.by_side(base["spreads"][n]["trades"]) for n in names}, names=["spread", "side"]).to_csv(T / "by_side.csv")
    pd.DataFrame({n: MT.exit_reasons(base["spreads"][n]["trades"]) for n in names}).fillna(0).astype(int).to_csv(
        T / "exit_reasons.csv")
    roll = trades.groupby("spread")[["gross_usd", "roll_day_gross_usd", "net_usd"]].sum().reindex(names)
    roll["roll_day_share_of_gross"] = roll["roll_day_gross_usd"] / roll["gross_usd"]
    roll.to_csv(T / "roll_attribution.csv")

    # descriptive: gate-open share by year; latest state; correlations
    gate = pd.DataFrame({n: base["spreads"][n]["daily"]["gate_open"].astype(float) for n in names})
    gate.groupby(gate.index.year).mean().rename_axis("year").to_csv(T / "gate_open_by_year.csv")
    now = {}
    for n in names:
        sim, last = base["spreads"][n]["sim"], base["spreads"][n]["daily"].iloc[-1]
        p = sim.frozen if sim.side != 0 else sim.fit
        units = float(sim.cfg["sizing"]["risk_per_day_usd"] / (last["sigma_next"] * sim.M))
        legs = [k for k in sim.leg if f"w_{k}" in p and np.isfinite(p[f"w_{k}"])]
        now[n] = {"date": base["spreads"][n]["daily"].index[-1].date(), "z": last["z"], "gate_open": bool(last["gate_open"]),
                  "side": int(last["side"]), "held_units": last["units"], "implied_k": p["k_hat"],
                  "half_life": p["half_life_raw"], "sigma_usd_bbl": last["sigma_next"],
                  "implied_contracts": ", ".join(f"{k.upper()} {p[f'w_{k}'] * units:+.2f}" for k in legs)}
    pd.DataFrame(now).T.to_csv(T / "latest_state.csv")
    mr = sum(pd.read_csv(path("derived_dir") / f"backtest_daily_fixed_{s}.csv", index_col=0, parse_dates=True)["pnl_usd"]
             for s in ["crack321", "brent_wti", "ho_rb"])
    tc = pd.read_csv(ROOT / "output/exploratory/trend_carry/tables/daily_books.csv", index_col=0, parse_dates=True)["combined"]
    pr = pd.read_csv(ROOT / "output/exploratory/pairs/tables/daily_books.csv", index_col=0, parse_dates=True)["book"]
    both = pd.concat([base["book"].rename("ratio_mr"), mr.rename("dollar_mean_reversion"), pr.rename("pairs"),
                      tc.rename("trend_carry")], axis=1, join="inner")
    corr = both.corr().round(3)
    corr.to_csv(T / "correlation.csv")
    (path("derived_dir") / "exploratory").mkdir(exist_ok=True)
    R.daily_paths(base["spreads"]).to_csv(path("derived_dir") / "exploratory" / "ratio_mr_daily.csv", index=False)

    out = {"status": "EXPLORATORY (PREREGISTRATION_RATIO.md); does not change the binding verdict", "source": "futures",
           "start": str(base["book"].index[0].date()), "end": str(base["book"].index[-1].date()),
           "verdicts": verdicts, "book": TC.summary(base["book"]), "round_trips": base["round_trips"],
           "spreads": sp_t.drop(columns=["top5_share_of_pnl"]).to_dict("index"),
           "comparisons": comp_t[["sharpe", "sharpe_t", "total", "max_dd", "round_trips", "y2022", "y2026"]].astype(float).round(3).to_dict("index"),
           "grid": grid_book, "roll_attribution": roll.round(3).to_dict("index"), "corr": corr["ratio_mr"].drop("ratio_mr").to_dict()}
    (T / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    plot_book(comps, books, verdicts, cfg)
    plot_spreads(base, names, verdicts)
    return out


def plot_book(comps, books, verdicts, cfg):
    last = books.index[-1]
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})
    tag = "PASS" if verdicts["book"]["PASS"] else "FAIL"
    for i, (m, r) in enumerate(comps.items()):
        s = r["book"].cumsum() / 1e3
        ax[0].plot(s.index, s, color=SERIES[i], lw=2.0 if m == "base" else 1.0,
                   label=f"{MODE_LAB[m]}: Sharpe {sharpe(r['book']):.2f}, {r['round_trips']} trades"
                         + (f" ({tag})" if m == "base" else ", not gated"))
    ax[0].axhline(0, color=REF_LINE, lw=0.7)
    for a0, b0 in cfg["evaluation"]["stress_windows"].values():
        ax[0].axvspan(pd.Timestamp(a0), pd.Timestamp(b0) if b0 else last, color=REF_LINE, alpha=0.10, lw=0)
    ax[0].set_ylabel("\\$k")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title("EXPLORATORY log-ratio mean reversion on futures: cumulative net P&L of the three-spread book (\\$k)")
    ann = books["book"].groupby(books.index.year).sum() / 1e3
    ax[1].bar(ann.index, ann, color=SERIES[0], width=0.8)
    ax[1].axhline(0, color=REF_LINE, lw=0.7)
    ax[1].set_xticks(ann.index, [str(y)[2:] for y in ann.index])
    ax[1].set_ylabel("\\$k")
    ax[1].set_title(f"Base book: net P&L by calendar year (\\$k; {ann.index[-1]} to date)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "cumulative_pnl.png")
    plt.close(fig)


def plot_spreads(base, names, verdicts):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for a, n in zip(ax.flat, names):
        d = base["spreads"][n]["daily"]
        a.fill_between(d.index, 0, 1, where=d["gate_open"].to_numpy(dtype=bool), color=REF_LINE, alpha=0.18, lw=0,
                       transform=a.get_xaxis_transform())
        a.plot(d.index, d["pnl_usd"].cumsum() / 1e3, color=SERIES[0], lw=1.2)
        a.axhline(0, color=REF_LINE, lw=0.7)
        tag = "PASS" if verdicts[n]["PASS"] else "FAIL"
        a.set_title(f"{LAB[n]}: Sharpe {sharpe(d['pnl_usd']):.2f}, {R.round_trips(base['spreads'][n]['trades'])} trades, {tag}", fontsize=10)
    ax[0].set_ylabel("\\$k")
    fig.suptitle("Cumulative net P&L by spread at R = \\$6k/day (\\$k; grey = gate open)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "spreads.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
