"""DESCRIPTIVE: horizon-matched reversion, variance ratios, trend and carry pre-tests.

Usage:  .venv/bin/python scripts/run_horizon_tests.py [--reps 600] [--quick]
Requires the Bloomberg export (futures.available()). Writes
output/exploratory/horizon_tests/{tables,figures}/.

Why this exists. Steps 4-6 of the test plan select an AR order on daily
differences. That answers "does yesterday's change predict today's", which is
not the question either traded strategy asks: mean reversion bets on today's
DEVIATION over the next few weeks, and trend bets on a 252-day move persisting.
`horizon.ar_power_sim` shows the daily test cannot see a 30-150 day half-life
at all, so its AR(0) result carries no information about mean reversion either
way. This script runs the horizon-matched versions, each against a simulated
null rather than an asymptotic one.

Added 2026-11-02, after every registered and exploratory result was seen. Not a
gate and not a trading rule, so it changes no verdict. It reads the futures
panel and writes only the files above.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from oilrv import futures
from oilrv import horizon as H
from oilrv.config import ROOT

OUT = ROOT / "output/exploratory/horizon_tests"
TAB, FIG = OUT / "tables", OUT / "figures"
HORIZONS = (5, 10, 21, 63, 126, 189, 252, 378, 504)
NULL_HORIZONS = (21, 63, 126, 189, 252, 378, 504)
# second trailing window: is the result an artefact of the 252-day anchor?
ANCHOR_WINDOWS = (252, 504)
VR_Q = (21, 63, 126, 252)
TRADE_START = "2010-02-01"
SUBPERIODS = [("2007-2013", "2007-01-01", "2013-12-31"),
              ("2014-2019", "2014-01-01", "2019-12-31"),
              ("2020-2026", "2020-01-01", "2026-12-31")]


def save(df: pd.DataFrame, name: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / name)
    print(f"  wrote {name}  ({len(df)} rows)")


# --------------------------------------------------------------- panel
def build_series() -> tuple[dict, dict]:
    """market level and back-adjusted series per instrument, plus curve slopes."""
    fut = futures.build_futures()
    sf = futures.spread_frames(fut)
    S: dict[str, pd.DataFrame] = {}
    for k, v in sf.items():
        S[k] = pd.DataFrame({"market": v[f"{k}_unadj"], "badj": v[k]}).dropna()
    for leg, col in [("cl", "crack321__cl_bbl"), ("brent", "brent_wti__brent_bbl"),
                     ("rb", "crack321__rb_bbl"), ("ho", "crack321__ho_bbl")]:
        S[leg] = pd.DataFrame({"market": fut[f"{col}_unadj"],
                               "badj": fut[col]}).dropna()

    # curve slopes M1 - Mk on the repo's own held-month convention
    panels = futures.load_panels()
    roots = futures.leg_roots()
    rb_days = 5
    slopes: dict[str, pd.DataFrame] = {}
    for leg, panel in panels.items():
        one = {leg: panel}
        held = {k: futures.held_months(one, roots, k=k, roll_bdays=rb_days)
                for k in (1, 2, 3, 6)}
        px = {}
        for k, h in held.items():
            h = h.dropna()
            px[k] = pd.Series([panel.at[t, m] for t, m in zip(h.index, h)],
                              index=h.index)
        div = {"XB": 100.0, "HO": 100.0}.get(roots[leg], 1.0)
        scale = 42.0 / div if roots[leg] in ("XB", "HO") else 1.0
        sl = pd.DataFrame({f"m1_m{k}": (px[1] - px[k]) * scale
                           for k in (2, 3, 6)}).dropna(how="all")
        slopes[leg] = sl
    return S, slopes


# ------------------------------------------------------------ sections
def part_power(reps_ar: int) -> None:
    print("\n[1] Power audits: what the existing tests could have detected")
    ar = H.ar_power_sim(reps=reps_ar)
    print(ar.to_string(index=False))
    save(ar.set_index("half_life"), "ar_order_power.csv")

    vr = H.vr_power_curve()
    print(vr.to_string(index=False))
    save(vr.set_index("sd_eta"), "vr252_power.csv")


def part_reversion(S: dict, reps: int) -> pd.DataFrame:
    print("\n[2] Horizon-matched mean reversion, with a random-walk null")
    print("    Horizons to 504 days and a 504-day anchor added 2026-11-02:")
    print("    does reversion appear if the horizon is longer than 126 days?")
    rows = []
    for nm in ("crack321", "brent_wti", "ho_rb"):
        lv = S[nm]["market"]
        sd = S[nm]["badj"].diff().std()
        for win in ANCHOR_WINDOWS:
            for k in HORIZONS:
                r = H.reversion_regression(lv, k, window=win)
                if k in NULL_HORIZONS:
                    null = H.reversion_rw_null(len(lv), sd, k, window=win, reps=reps)
                    r["crit5_sim"] = float(np.percentile(null, 5))
                    r["p_sim"] = float((null <= r["t"]).mean())
                rows.append({"spread": nm, "anchor_window": win, **r})
    df = pd.DataFrame(rows).set_index(["spread", "anchor_window", "k"])
    print(df.round(4).to_string())
    save(df, "reversion_by_horizon.csv")
    return df


def part_vr(S: dict) -> pd.DataFrame:
    print("\n[3] Variance ratios on back-adjusted increments (Lo-MacKinlay)")
    rows = []
    for nm, df in S.items():
        dx = df["badj"].diff().dropna().to_numpy()
        r = {"series": nm}
        for q in VR_Q:
            v = H.variance_ratio(dx, q)
            r[f"vr{q}"], r[f"z{q}"], r[f"p{q}"] = v["vr"], v["z2"], v["p2"]
        rows.append(r)
    df = pd.DataFrame(rows).set_index("series")
    print(df.round(3).to_string())
    save(df, "variance_ratios.csv")
    return df


def _subperiods(badj: pd.Series, sig: pd.Series) -> dict:
    out = {}
    for lab, a, z in SUBPERIODS:
        pnl = (sig.reindex(badj.index).ffill().shift(1) * badj.diff()).dropna()
        pnl = pnl[(pnl.index >= pd.Timestamp(a)) & (pnl.index <= pd.Timestamp(z))]
        out[lab] = (float(pnl.mean() / pnl.std() * np.sqrt(252))
                    if len(pnl) > 60 else np.nan)
    return out


def part_trend(S: dict) -> pd.DataFrame:
    print("\n[4] Trend pre-test: TSMOM regression and 252-day signal Sharpe")
    rows = []
    for nm, df in S.items():
        b = df["badj"]
        reg = H.tsmom_regression(b)
        sig = np.sign(b - b.shift(252))
        sh = H.signal_sharpe(b, sig, start=TRADE_START)
        rows.append({"series": nm, "tsmom_beta": reg["beta"], "tsmom_t": reg["t"],
                     "sharpe": sh["sharpe"], "t": sh["t"],
                     **_subperiods(b, sig)})
    df = pd.DataFrame(rows).set_index("series")
    print(df.round(3).to_string())
    save(df, "trend_pretest.csv")
    return df


def part_carry(S: dict, slopes: dict) -> pd.DataFrame:
    print("\n[5] Carry pre-test, ALL FOUR roots (the deck tests only CL and CO)")
    rows = []
    for leg in ("cl", "brent", "rb", "ho"):
        b = S[leg]["badj"]
        for col in slopes[leg].columns:
            sig = np.sign(slopes[leg][col])
            sh = H.signal_sharpe(b, sig, start=TRADE_START)
            row = {"instr": leg, "slope": col, "sharpe": sh["sharpe"], "t": sh["t"],
                   "total": sh["total"]}
            if col == "m1_m2":
                row.update(_subperiods(b, sig))
            rows.append(row)
    df = pd.DataFrame(rows).set_index(["instr", "slope"])
    print(df.round(3).to_string())
    save(df, "carry_pretest.csv")
    return df


def part_roll(S: dict) -> pd.DataFrame:
    print("\n[6] Roll hygiene: how much of a back-adjusted LEVEL is roll yield")
    rows = []
    for nm, df in S.items():
        adj = float(df["badj"].iloc[0] - df["market"].iloc[0])
        rng = float(df["market"].max() - df["market"].min())
        rows.append({"series": nm, "cumulative_roll_adj": adj, "level_range": rng,
                     "ratio": abs(adj) / rng if rng else np.nan})
    df = pd.DataFrame(rows).set_index("series").sort_values("ratio", ascending=False)
    print(df.round(2).to_string())
    save(df, "roll_adjustment.csv")
    return df


# -------------------------------------------------------------- figure
def figure(rev: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    names = {"crack321": "3-2-1 crack", "brent_wti": "Brent \u2212 WTI", "ho_rb": "HO \u2212 RB"}
    col = {"crack321": "#2563eb", "brent_wti": "#dc2626", "ho_rb": "#059669"}
    ks = list(NULL_HORIZONS)
    r = rev.xs(252, level="anchor_window")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.1), sharey=True)
    for ax, nm in zip(axes, names):
        vals = [r.loc[(nm, k), "share_of_1sd"] for k in ks]
        ps = [r.loc[(nm, k), "p_sim"] for k in ks]
        ax.bar(range(len(ks)), vals,
               color=[col[nm] if p < .05 else "#d1d5db" for p in ps], width=.66)
        ax.axhline(0, color="#111", lw=1)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"{k}d" for k in ks], fontsize=10)
        ax.set_title(names[nm], fontweight="bold", fontsize=13, pad=9)
        ax.set_xlabel("holding horizon", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.25, lw=.7)
        ax.set_axisbelow(True)
        for i, (v, p) in enumerate(zip(vals, ps)):
            lab = "p<0.01" if p < .01 else f"p={p:.2f}"
            strong = p < .05
            ax.text(i, v - .03, lab, ha="center", va="top", fontsize=9,
                    fontweight="bold" if strong else "normal",
                    color="#111" if strong else "#888")
    axes[0].set_ylabel("share of a 1\u03c3 deviation given back", fontsize=11)
    lo = min(r.loc[(n, k), "share_of_1sd"] for n in names for k in ks)
    axes[0].set_ylim(min(-0.2, lo * 1.22), 0.12)
    fig.suptitle("Mean reversion appears only at multi-month horizons, and only in HO \u2212 RB",
                 fontsize=14.5, fontweight="bold", y=1.0)
    fig.text(.5, .875, "Forward change regressed on today's deviation from a trailing 252-day mean "
             "of the MARKET level. More negative = stronger reversion; coloured = p < 0.05.\n"
             "p-values from a simulated random-walk null, not the normal \u2014 this regression is "
             "biased negative in finite samples. Longer horizons have fewer independent windows.",
             ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=[0, 0, 1, .81])
    fig.savefig(FIG / "horizon_reversion.png", dpi=170, facecolor="white")
    print("  wrote figures/horizon_reversion.png")


def main(reps: int = 600, quick: bool = False) -> dict:
    if not futures.available():
        raise SystemExit("Bloomberg export not found; see data/raw/bloomberg/README.md")
    print("Building the futures panel (repo roll convention: same delivery month, "
          "5 business days before the earliest leg expiry)")
    S, slopes = build_series()
    print(f"  {len(S)} series, {len(next(iter(S.values())))} days")

    part_power(reps_ar=10 if quick else 40)
    rev = part_reversion(S, reps=100 if quick else reps)
    part_vr(S)
    part_trend(S)
    part_carry(S, slopes)
    part_roll(S)
    figure(rev)

    summary = {"status": "DESCRIPTIVE; changes no registered verdict",
               "source": "futures", "reps": reps,
               "series": sorted(S), "horizons": list(HORIZONS)}
    (TAB / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Outputs in {OUT}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=600,
                    help="random-walk null replications per spread and horizon")
    ap.add_argument("--quick", action="store_true", help="fewer reps, for a smoke run")
    a = ap.parse_args()
    main(reps=a.reps, quick=a.quick)
