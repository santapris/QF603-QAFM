"""DESCRIPTIVE: does curve carry pay because inventories are tight?

Usage:  .venv/bin/python scripts/run_inventory_carry.py
Requires the Bloomberg export. Writes output/exploratory/horizon_tests/tables/
inventory_*.csv and figures/inventory_carry.png.

This is prediction 3 of the restated thesis in report/strategy_thesis.md. The
theory of storage (Kaldor 1939, Working 1949) says the forward curve slopes
down from the front when stocks are scarce: holders of the physical earn a
convenience yield, so the front trades above the deferred. If carry's return
is compensation for bearing that scarcity, two things must hold.

  A. The slope must track inventories. Regress M1 - M2 on the point-in-time
     5-year inventory deviation. The theory predicts a NEGATIVE coefficient:
     low stocks -> backwardation -> positive M1 - M2.
  B. The RETURN must concentrate when stocks are tight. Split carry P&L by
     inventory tercile, and regress forward carry P&L on the deviation.

B is the one that discriminates. A can hold mechanically while the strategy
still earns nothing conditional on it; if carry pays the same whatever the
inventory state, the premium has no inventory mechanism and the thesis is
wrong, whatever the backtest says.

Inventory features are stamped on the day they became known (weekly EIA, known
from the Thursday after the week ends), so there is no look-ahead.

Added 2026-11-02, after every registered and exploratory result was seen. Not a
gate and not a trading rule, so it changes no verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from oilrv import futures
from oilrv import horizon as H
from oilrv.config import ROOT

OUT = ROOT / "output/exploratory/horizon_tests"
TAB, FIG = OUT / "tables", OUT / "figures"
START = "2010-02-01"

# each root paired with the stock series its own curve should respond to
STOCK = {"cl": "crude_ex_spr_dev_5y", "brent": "crude_ex_spr_dev_5y",
         "ho": "distillate_dev_5y", "rb": "gasoline_dev_5y"}
LABEL = {"ho": "ULSD", "brent": "Brent", "cl": "WTI", "rb": "RBOB"}
ORDER = ["ho", "brent", "cl", "rb"]


def save(df, name):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / name)
    print(f"  wrote {name}")


def build():
    fut = futures.build_futures()
    panels = futures.load_panels()
    roots = futures.leg_roots()
    feats = pd.read_csv(ROOT / "data/clean/features.csv", index_col=0,
                        parse_dates=True)
    badj = {"cl": fut["crack321__cl_bbl"], "brent": fut["brent_wti__brent_bbl"],
            "rb": fut["crack321__rb_bbl"], "ho": fut["crack321__ho_bbl"]}
    slope = {}
    for leg, panel in panels.items():
        one = {leg: panel}
        h1 = futures.held_months(one, roots, k=1, roll_bdays=5).dropna()
        h2 = futures.held_months(one, roots, k=2, roll_bdays=5).dropna()
        p1 = pd.Series([panel.at[t, m] for t, m in zip(h1.index, h1)], index=h1.index)
        p2 = pd.Series([panel.at[t, m] for t, m in zip(h2.index, h2)], index=h2.index)
        sc = 42.0 / 100.0 if roots[leg] in ("XB", "HO") else 1.0
        slope[leg] = ((p1 - p2) * sc).dropna()
    return badj, slope, feats


# ------------------------------------------------------------------- A
def part_a(slope, feats):
    print("\n[A] Does the slope track inventories? (theory of storage)")
    print("    M1 - M2  =  a + b * (5-year stock deviation, %).  b < 0 expected.\n")
    rows = []
    for leg in ORDER:
        col = STOCK[leg]
        d = pd.concat([slope[leg].rename("sl"), feats[col].rename("inv")],
                      axis=1).dropna()
        d = d[d.index >= pd.Timestamp(START)]
        X = np.column_stack([d["inv"].to_numpy(), np.ones(len(d))])
        r = H.ols(d["sl"].to_numpy(), X, hac_lags=60)
        rows.append({"instr": LABEL[leg], "stock_series": col,
                     "beta": r["b"][0], "t_nw": r["t"][0], "r2": r["r2"],
                     "n": len(d),
                     "mean_slope_low_stocks": float(d[d["inv"] <= d["inv"].quantile(1/3)]["sl"].mean()),
                     "mean_slope_high_stocks": float(d[d["inv"] >= d["inv"].quantile(2/3)]["sl"].mean())})
    df = pd.DataFrame(rows).set_index("instr")
    print(df.round(4).to_string())
    save(df, "inventory_slope_regression.csv")
    return df


# ------------------------------------------------------------------- B
def part_b(badj, slope, feats):
    print("\n[B] Does the carry RETURN concentrate when stocks are tight?")
    print("    Daily carry P&L (long if M1>M2), split by inventory tercile.\n")
    rows, pnls = [], {}
    for leg in ORDER:
        b = badj[leg].dropna()
        sig = np.sign(slope[leg]).reindex(b.index).ffill().shift(1)
        pnl = (sig * b.diff()).dropna()
        pnl = pnl[pnl.index >= pd.Timestamp(START)]
        inv = feats[STOCK[leg]].reindex(pnl.index).ffill()
        ok = inv.notna()
        pnl, inv = pnl[ok], inv[ok]
        q1, q2 = inv.quantile(1 / 3), inv.quantile(2 / 3)
        buckets = {"low stocks": inv <= q1,
                   "mid": (inv > q1) & (inv < q2),
                   "high stocks": inv >= q2}
        row = {"instr": LABEL[leg]}
        for name, m in buckets.items():
            x = pnl[m]
            row[f"{name} Sharpe"] = x.mean() / x.std() * np.sqrt(252)
            row[f"{name} $/day"] = x.mean()
        row["low-high gap"] = row["low stocks Sharpe"] - row["high stocks Sharpe"]
        rows.append(row)
        pnls[leg] = (pnl, inv)
    df = pd.DataFrame(rows).set_index("instr")
    print(df.round(3).to_string())
    save(df, "inventory_carry_terciles.csv")
    return df, pnls


# ------------------------------------------------------------------- C
def part_c(pnls):
    print("\n[C] Predictive regression: next-21-day carry P&L on today's stock")
    print("    deviation. Newey-West, 26 lags. b < 0 = pays more when tight.\n")
    rows = []
    for leg in ORDER:
        pnl, inv = pnls[leg]
        fwd = pnl[::-1].rolling(21).sum()[::-1].shift(-1)
        d = pd.concat([inv.rename("inv"), fwd.rename("f")], axis=1).dropna()
        X = np.column_stack([d["inv"].to_numpy(), np.ones(len(d))])
        r = H.ols(d["f"].to_numpy(), X, hac_lags=26)
        rows.append({"instr": LABEL[leg], "beta": r["b"][0], "t_nw": r["t"][0],
                     "r2": r["r2"], "n": len(d)})
    df = pd.DataFrame(rows).set_index("instr")
    print(df.round(4).to_string())
    save(df, "inventory_carry_predictive.csv")
    return df


def figure(terciles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    cats = ["low stocks", "mid", "high stocks"]
    cols = ["#047857", "#9ca3af", "#dc2626"]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    w = 0.26
    for j, c in enumerate(cats):
        ax.bar([i + (j - 1) * w for i in range(4)],
               [terciles.loc[LABEL[k], f"{c} Sharpe"] for k in ORDER],
               width=w, color=cols[j], label=c)
    ax.axhline(0, color="#111", lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels([LABEL[k] for k in ORDER], fontsize=11)
    ax.set_ylabel("carry Sharpe within the tercile", fontsize=11)
    ax.legend(frameon=False, fontsize=10, ncol=1, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.25, lw=.7)
    ax.set_axisbelow(True)
    fig.suptitle("Does carry pay because inventories are tight?",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.text(.5, .885, "Daily carry P&L split by the point-in-time 5-year deviation "
             "of the matching stock series.\nThe thesis predicts green above red.",
             ha="center", fontsize=10, color="#444")
    fig.tight_layout(rect=[0, 0, 1, .85])
    fig.savefig(FIG / "inventory_carry.png", dpi=170, facecolor="white")
    print("  wrote figures/inventory_carry.png")


def main():
    if not futures.available():
        raise SystemExit("Bloomberg export not found")
    badj, slope, feats = build()
    part_a(slope, feats)
    terc, pnls = part_b(badj, slope, feats)
    part_c(pnls)
    figure(terc)
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
