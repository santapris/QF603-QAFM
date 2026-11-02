"""Figure for report/strategy_thesis.md: carry by instrument and sub-period.

The broadened thesis predicts carry is largest where storage is tightest
(distillate), smallest or negative where it is most abundant and seasonal
(gasoline). This plots the pre-test result against that prediction.

Reads output/exploratory/horizon_tests/tables/carry_pretest.csv.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from oilrv.config import ROOT

TAB = ROOT / "output/exploratory/horizon_tests/tables"
FIG = ROOT / "output/exploratory/horizon_tests/figures"
LABEL = {"ho": "ULSD\n(distillate)", "brent": "Brent\n(crude)",
         "cl": "WTI\n(crude)", "rb": "RBOB\n(gasoline)"}
ORDER = ["ho", "brent", "cl", "rb"]
PERIODS = ["2007-2013", "2014-2019", "2020-2026"]


def main() -> None:
    d = pd.read_csv(TAB / "carry_pretest.csv")
    m1 = d[d["slope"] == "m1_m2"].set_index("instr")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # ---- left: full-sample Sharpe, coloured by significance ----------
    vals = [m1.loc[k, "sharpe"] for k in ORDER]
    ts = [m1.loc[k, "t"] for k in ORDER]
    cols = ["#047857" if t > 2 else "#34d399" if t > 1.3 else
            "#d1d5db" if t > -1 else "#dc2626" for t in ts]
    ax1.bar(range(4), vals, color=cols, width=.64)
    ax1.axhline(0, color="#111", lw=1)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([LABEL[k] for k in ORDER], fontsize=10)
    ax1.set_ylabel("Sharpe, M1 − M2 carry", fontsize=11)
    ax1.set_title("Carry is largest where storage is tightest",
                  fontweight="bold", fontsize=12, pad=8)
    for i, (v, t) in enumerate(zip(vals, ts)):
        off = .035 if v >= 0 else -.035
        ax1.text(i, v + off, f"{v:+.2f}\nt={t:.2f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=9.5,
                 fontweight="bold" if abs(t) > 2 else "normal")
    ax1.set_ylim(-0.46, 0.72)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", alpha=.25, lw=.7)
    ax1.set_axisbelow(True)

    # ---- right: by sub-period ---------------------------------------
    w = 0.26
    pcol = ["#cbd5e1", "#64748b", "#1e3a5f"]
    for j, p in enumerate(PERIODS):
        ax2.bar([i + (j - 1) * w for i in range(4)],
                [m1.loc[k, p] for k in ORDER], width=w,
                color=pcol[j], label=p)
    ax2.axhline(0, color="#111", lw=1)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([LABEL[k] for k in ORDER], fontsize=10)
    ax2.set_title("but each instrument pays in a different decade",
                  fontweight="bold", fontsize=12, pad=8)
    ax2.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper center")
    ax2.set_ylim(-0.75, 1.32)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=.25, lw=.7)
    ax2.set_axisbelow(True)

    fig.suptitle("Curve carry across the complex — one unit, gross of costs, "
                 "from February 2010", fontsize=13.5, fontweight="bold", y=1.0)
    fig.text(.5, .885, "The broadened thesis predicts compensation accrues to bearing "
             "inventory risk, so carry should rank with storage tightness: distillate "
             "above crude, gasoline last.\nIt does. But no instrument pays in every "
             "sub-period, so the ranking is firmer than the timing.",
             ha="center", fontsize=10, color="#444")
    fig.tight_layout(rect=[0, 0, 1, .855])
    out = FIG / "carry_by_instrument.png"
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
