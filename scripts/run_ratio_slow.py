"""EXPLORATORY: slow (monthly-decision) log-ratio mean reversion on the futures -> output/exploratory/ratio_slow/.

Usage:  .venv/Scripts/python.exe scripts/run_ratio_slow.py
Requires build_data.py (futures), run_backtest.py --source futures, run_trend_carry.py, run_pairs.py and run_ratio_mr.py for
the descriptive correlations and the reference check. Rules: PREREGISTRATION_RATIO_SLOW.md (registered 2026-11-04, after the
daily ratio strategy failed and before any monthly-decision signal or P&L). It changes no verdict. Run once; every spread,
grid point and comparison is reported.
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

OUT = ROOT / "output" / "exploratory" / "ratio_slow"
LAB = {"crack321": "3-2-1 crack", "brent_wti": "Brent / WTI", "ho_rb": "HO / RB"}


def years_needed(sr: float, t: float) -> float:
    """Years of data at annualised Sharpe `sr` for the t-statistic to reach `t` (t = SR * sqrt(years))."""
    return float((t / sr) ** 2) if np.isfinite(sr) and sr > 0 else np.nan


def main() -> dict:
    use_style()
    cfg = load_config()
    set_source("futures")
    rs, rx = cfg["exploratory_ratio_slow"], cfg["exploratory_ratio"]
    T = OUT / "tables"
    T.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    data = R.build_data(cfg)
    names = list(data["specs"])
    K = rs["decision_months"]

    # reference: decisions every day must reproduce the registered ratio result
    daily = R.run_book(data, cfg)
    committed = json.loads((ROOT / "output/exploratory/ratio_mr/tables/summary.json").read_text())
    chk = pd.DataFrame([{"book_sharpe_here": sharpe(daily["book"]), "book_sharpe_committed": committed["book"]["sharpe"],
                         "round_trips_here": daily["round_trips"], "round_trips_committed": committed["round_trips"]}])
    chk["match"] = bool(abs(chk.book_sharpe_here[0] - chk.book_sharpe_committed[0]) < 1e-6
                        and chk.round_trips_here[0] == chk.round_trips_committed[0])
    chk.to_csv(T / "reference_check.csv", index=False)

    base = R.run_book(data, cfg, decision_months=K)
    quarter = R.run_book(data, cfg, decision_months=rs["robustness"]["decision_months"][0])
    Z0 = cfg["strategy"]["entry_z"]

    # G4 grid: entry |z| at monthly decisions
    runs = {f"entry_z={z}": base if z == Z0 else R.run_book(data, cfg, entry_z=z, decision_months=K) for z in rs["grid"]["entry_z"]}
    grid_book = {k: sharpe(r["book"]) for k, r in runs.items()}
    grid_sp = {n: {k: sharpe(r["spreads"][n]["daily"]["pnl_usd"]) for k, r in runs.items()} for n in names}
    grid_t = pd.DataFrame({"book": grid_book, **grid_sp}).rename_axis("setting")
    grid_t["round_trips"] = pd.Series({k: r["round_trips"] for k, r in runs.items()})
    grid_t.to_csv(T / "grid.csv")

    # verdicts: the book is primary; each spread at full R
    verdicts = {"book": R.verdict(base["book"], base["round_trips"], grid_book, cfg)}
    rows = {}
    for n in names:
        r = base["spreads"][n]
        verdicts[n] = R.verdict(r["daily"]["pnl_usd"], R.round_trips(r["trades"]), grid_sp[n], cfg)
        rows[n] = MT.summarize(r["daily"], r["trades"])
        rows[n]["gate_open_share"] = float(r["daily"]["gate_open"].mean())
        rows[n]["counts_as_evidence"] = R.single_spread_evidence(verdicts[n], rows[n]["sharpe_t"], cfg)
        rows[n]["years_for_t2"] = years_needed(rows[n]["sharpe"], 2.0)
        rows[n]["years_for_t3"] = years_needed(rows[n]["sharpe"], 3.0)
    pd.DataFrame({k: {kk: (json.dumps(v, default=float) if isinstance(v, dict) else v) for kk, v in d.items()}
                  for k, d in verdicts.items()}).T.to_csv(T / "gate_verdicts.csv")
    sp_t = pd.DataFrame(rows).T
    sp_t.to_csv(T / "summary_spreads.csv")
    book_sum = TC.summary(base["book"])
    book_sum["years_for_t2"] = years_needed(book_sum["sharpe"], 2.0)
    book_sum["years_for_t3"] = years_needed(book_sum["sharpe"], 3.0)
    book_pass = bool(verdicts["book"]["PASS"])
    book_evidence = bool(book_pass and book_sum["sharpe_t"] > rs["book_min_t_to_call_evidence"])

    # the decision-interval profile (reported): daily reference, monthly base, quarterly
    prof = {"daily (reference)": daily, "monthly (base)": base, "quarterly": quarter}
    prof_t = pd.DataFrame({k: {**TC.summary(r["book"]), "round_trips": r["round_trips"],
                               **{f"sharpe|{n}": sharpe(r["spreads"][n]["daily"]["pnl_usd"]) for n in names},
                               **{f"trips|{n}": R.round_trips(r["spreads"][n]["trades"]) for n in names},
                               **{f"avg_hold|{n}": float(r["spreads"][n]["trades"].query("exit_reason != 'open_at_end'")["hold_days"].mean())
                                  for n in names}} for k, r in prof.items()}).T
    prof_t.to_csv(T / "decision_interval_profile.csv")

    # comparisons and robustness: reported, not gated
    comps = {"base": base, **{m: R.run_book(data, cfg, mode=m, decision_months=K) for m in rs["comparisons"]}}
    stress = cfg["evaluation"]["stress_windows"]
    comp_t = pd.DataFrame({m: {**TC.summary(r["book"]), "round_trips": r["round_trips"],
                               "y2022": window_pnl(r["book"].to_frame("p"), "p", *stress["ukraine"]),
                               "y2026": window_pnl(r["book"].to_frame("p"), "p", *stress["hormuz"]),
                               **{f"sharpe|{n}": sharpe(r["spreads"][n]["daily"]["pnl_usd"]) for n in names}}
                           for m, r in comps.items()}).T
    comp_t.to_csv(T / "comparisons.csv")
    rob = {("window", cfg["model"]["window"]): base, ("exit_z", cfg["strategy"]["exit_z"]): base,
           ("cost_usd_per_bbl", cfg["strategy"]["cost_usd_per_bbl"]): base}
    rob.update({("window", w): R.run_book(data, cfg, window=w, decision_months=K) for w in rs["robustness"]["window"]})
    rob.update({("exit_z", e): R.run_book(data, cfg, exit_z=e, decision_months=K) for e in rs["robustness"]["exit_z"]})
    rob.update({("cost_usd_per_bbl", c): R.run_book(data, cfg, cost_bbl=c, decision_months=K)
                for c in rs["robustness"]["cost_usd_per_bbl"]})
    rob_t = pd.DataFrame({k: {"sharpe": sharpe(r["book"]), "total": float(r["book"].sum()), "round_trips": r["round_trips"],
                              "start": str(r["book"].index[0].date())} for k, r in rob.items()}).T.sort_index()
    rob_t.index.names = ["setting", "value"]
    rob_t.to_csv(T / "robustness.csv")

    # base run: book, sub-periods, stress, calendar years, drawdowns, trades
    pnl = pd.DataFrame({n: base["spreads"][n]["daily"]["pnl_usd"] for n in names})
    books = pd.concat([base["book"].rename("book"), pnl], axis=1)
    books.to_csv(T / "daily_books.csv")
    TC.annual(books).to_csv(T / "annual.csv")
    pd.concat({k: subperiods(books[[k]], k, cfg["evaluation"]["sub_periods"]) for k in books},
              names=["book", "period"]).to_csv(T / "subperiods.csv")
    pd.DataFrame({w: {k: window_pnl(books, k, a, b) for k in books} for w, (a, b) in stress.items()}).T.to_csv(T / "stress_windows.csv")
    pd.concat({k: TC.drawdowns(books[k]) for k in books}, names=["book", "rank"]).to_csv(T / "drawdowns.csv")
    trades = pd.concat([R.roll_attribution(base["spreads"][n], data["rolled"][n]).assign(spread=n) for n in names], ignore_index=True)
    trades.to_csv(T / "trades.csv", index=False)
    pd.concat({n: P.by_side(base["spreads"][n]["trades"]) for n in names}, names=["spread", "side"]).to_csv(T / "by_side.csv")
    pd.DataFrame({n: MT.exit_reasons(base["spreads"][n]["trades"]) for n in names}).fillna(0).astype(int).to_csv(T / "exit_reasons.csv")

    # correlations
    mr = sum(pd.read_csv(path("derived_dir") / f"backtest_daily_fixed_{s}.csv", index_col=0, parse_dates=True)["pnl_usd"]
             for s in ["crack321", "brent_wti", "ho_rb"])
    tc = pd.read_csv(ROOT / "output/exploratory/trend_carry/tables/daily_books.csv", index_col=0, parse_dates=True)["combined"]
    pr = pd.read_csv(ROOT / "output/exploratory/pairs/tables/daily_books.csv", index_col=0, parse_dates=True)["book"]
    both = pd.concat([base["book"].rename("ratio_slow"), daily["book"].rename("ratio_daily"), mr.rename("dollar_mean_reversion"),
                      pr.rename("pairs"), tc.rename("trend_carry")], axis=1, join="inner")
    corr = both.corr().round(3)
    corr.to_csv(T / "correlation.csv")

    out = {"status": "EXPLORATORY (PREREGISTRATION_RATIO_SLOW.md); does not change the binding verdict", "source": "futures",
           "start": str(base["book"].index[0].date()), "end": str(base["book"].index[-1].date()),
           "reference_check_match": bool(chk.match[0]), "verdicts": verdicts, "book": book_sum, "round_trips": base["round_trips"],
           "book_passes_gates": book_pass, "book_counts_as_evidence": book_evidence,
           "spreads": sp_t.drop(columns=["top5_share_of_pnl"]).to_dict("index"),
           "profile": prof_t[["sharpe", "sharpe_t", "total", "round_trips"]].astype(float).round(3).to_dict("index"),
           "comparisons": comp_t[["sharpe", "sharpe_t", "total", "round_trips", "y2022", "y2026"]].astype(float).round(3).to_dict("index"),
           "grid": grid_book, "corr": corr["ratio_slow"].drop("ratio_slow").to_dict()}
    (T / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    plot_profile(prof, books, verdicts, cfg)
    return out


def plot_profile(prof, books, verdicts, cfg):
    last = books.index[-1]
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})
    for i, (k, r) in enumerate(prof.items()):
        s = r["book"].cumsum() / 1e3
        ax[0].plot(s.index, s, color=SERIES[i], lw=2.0 if "base" in k else 1.0,
                   label=f"Decisions {k}: Sharpe {sharpe(r['book']):.2f}, {r['round_trips']} trades")
    ax[0].axhline(0, color=REF_LINE, lw=0.7)
    for a0, b0 in cfg["evaluation"]["stress_windows"].values():
        ax[0].axvspan(pd.Timestamp(a0), pd.Timestamp(b0) if b0 else last, color=REF_LINE, alpha=0.10, lw=0)
    ax[0].set_ylabel("\\$k")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title("EXPLORATORY slow ratio mean reversion: cumulative net P&L of the three-spread book (\\$k)")
    for i, n in enumerate(LAB):
        s = books[n].cumsum() / 1e3
        ax[1].plot(s.index, s, color=SERIES[i], lw=1.1, label=f"{LAB[n]}: Sharpe {sharpe(books[n]):.2f}")
    ax[1].axhline(0, color=REF_LINE, lw=0.7)
    ax[1].legend(loc="upper left", fontsize=8)
    ax[1].set_ylabel("\\$k")
    ax[1].set_title("Base (monthly decisions), by spread at full risk (\\$k)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "cumulative_pnl.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
