"""DESCRIPTIVE: the ratio strategies on the registered integer book -> output/exploratory/ratio_integer/.

Usage:  .venv/Scripts/python.exe scripts/run_ratio_integer.py
Requires build_data.py (futures) and run_backtest.py (for the registered dollar integer book used as a reference).

Added 2026-11-04 after every strategy result was seen. The gates for the ratio strategies were judged on the research stream, whose
units are R / (sigma M) with no rounding or cap. The registered sizing also defines an integer book: $2m, units = min(vol limit,
$40k stop limit) floored and capped at 3, halved after a 10% drawdown, stopped after 20% (PREREGISTRATION.md). This script runs the
same ratio rules through it. It is reported next to the research stream and judges nothing.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oilrv import metrics as MT
from oilrv import ratio as R
from oilrv.config import ROOT, load_config, set_source
from oilrv.plotting import REF_LINE, SERIES, use_style

OUT = ROOT / "output" / "exploratory" / "ratio_integer"
LAB = {"crack321": "3-2-1 crack", "brent_wti": "Brent / WTI", "ho_rb": "HO / RB"}


def book_row(res: dict, cfg: dict) -> dict:
    b = res["book"]
    tr = pd.concat([r["trades"].assign(spread=n) for n, r in res["spreads"].items() if len(r["trades"])], ignore_index=True)
    closed = tr[tr["exit_reason"] != "open_at_end"]
    st = cfg["evaluation"]["stress_windows"]
    return {"final_equity": float(b["equity"].iloc[-1]), "total": float(b["pnl_usd"].sum()), "sharpe": MT.sharpe(b["pnl_usd"]),
            "max_dd_usd": MT.max_drawdown(b["pnl_usd"]), "max_drawdown_pct": float(b["drawdown"].max()), "halted": bool(b["halted"].any()),
            "n_round_trips": int(len(closed)), "blocked_by_size": int(sum(r["daily"].attrs["blocked_by_size"] for r in res["spreads"].values())),
            "days_half_size": int((b["size_mult"] == 0.5).sum()), "max_units_used": float(closed["units"].max()) if len(closed) else 0.0,
            "mean_units": float(closed["units"].mean()) if len(closed) else 0.0, "worst_trade_usd": float(closed["net_usd"].min()),
            "y2022": MT.window_pnl(b, "pnl_usd", *st["ukraine"]), "y2026": MT.window_pnl(b, "pnl_usd", *st["hormuz"])}


def main() -> dict:
    use_style()
    cfg = load_config()
    set_source("futures")
    T = OUT / "tables"
    T.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    data = R.build_data(cfg)
    runs = {"daily decisions (registered ratio rules)": R.run_integer_book(data, cfg),
            "monthly decisions (slow version)": R.run_integer_book(data, cfg, decision_months=1)}
    rows = {k: book_row(r, cfg) for k, r in runs.items()}
    reg = pd.read_csv(ROOT / "output" / "tables" / "06_integer_book.csv", index_col=0).loc["gated"]
    reference = {"final_equity": float(reg.final_equity), "total": float(reg.total), "sharpe": float(reg.sharpe),
                 "max_dd_usd": float(reg.max_dd_usd), "max_drawdown_pct": float(reg.max_drawdown_pct), "halted": bool(reg.halted),
                 "n_round_trips": int(reg.n_trades), "blocked_by_size": int(reg.blocked_by_size)}
    tab = pd.DataFrame({**rows, "registered dollar-gap strategy (reference)": reference}).T
    tab.to_csv(T / "integer_book.csv")

    # per spread, same units as the book
    sp = []
    for k, r in runs.items():
        for n, s in r["spreads"].items():
            tr = s["trades"][s["trades"]["exit_reason"] != "open_at_end"]
            sp.append({"run": k, "spread": n, "sharpe": MT.sharpe(s["daily"]["pnl_usd"]), "total": float(s["daily"]["pnl_usd"].sum()),
                       "round_trips": int(len(tr)), "mean_units": float(tr["units"].mean()) if len(tr) else np.nan,
                       "worst_trade_usd": float(tr["net_usd"].min()) if len(tr) else np.nan,
                       "y2026": MT.window_pnl(s["daily"], "pnl_usd", *cfg["evaluation"]["stress_windows"]["hormuz"])})
    pd.DataFrame(sp).to_csv(T / "by_spread.csv", index=False)

    # the research-stream numbers, for the side-by-side
    research = json.loads((ROOT / "output/exploratory/ratio_mr/tables/summary.json").read_text())
    slow = json.loads((ROOT / "output/exploratory/ratio_slow/tables/summary.json").read_text())
    rs = {"daily decisions (registered ratio rules)": research, "monthly decisions (slow version)": slow}
    cmp_ = []
    for k in runs:
        b = rs[k]
        cmp_.append({"run": k, "research_sharpe": b["book"]["sharpe"], "research_total": b["book"]["total"], "research_round_trips": b["round_trips"],
                     "research_worst_trade": float(pd.read_csv(ROOT / "output/exploratory" / ("ratio_mr" if "daily" in k else "ratio_slow") / "tables" / "trades.csv").net_usd.min()),
                     "integer_sharpe": rows[k]["sharpe"], "integer_total": rows[k]["total"], "integer_round_trips": rows[k]["n_round_trips"],
                     "integer_worst_trade": rows[k]["worst_trade_usd"], "integer_2026": rows[k]["y2026"]})
    pd.DataFrame(cmp_).to_csv(T / "research_vs_integer.csv", index=False)

    for k, r in runs.items():
        r["book"].to_csv(T / f"book_{'daily' if 'daily' in k else 'monthly'}.csv")
    out = {"status": "DESCRIPTIVE: the ratio strategies on the registered integer book; judges nothing",
           "integer_book": tab.round(4).to_dict("index"), "research_vs_integer": pd.DataFrame(cmp_).round(4).to_dict("records"),
           "by_spread": pd.DataFrame(sp).round(4).to_dict("records")}
    (T / "summary.json").write_text(json.dumps(out, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (k, r) in enumerate(runs.items()):
        ax.plot(r["book"].index, r["book"]["equity"] / 1e6, color=SERIES[i], lw=1.3, label=f"{k}: Sharpe {rows[k]['sharpe']:.2f}")
    refb = pd.read_csv(ROOT / "data/derived/integer_book_gated.csv", index_col=0, parse_dates=True)
    ax.plot(refb.index, refb["equity"] / 1e6, color=REF_LINE, lw=1.1, label=f"registered dollar-gap strategy: Sharpe {reference['sharpe']:.2f}")
    ax.set_ylabel("book equity, $m")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Integer book ($2m, units capped at 3, drawdown rules)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "integer_book_equity.png", dpi=140)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
