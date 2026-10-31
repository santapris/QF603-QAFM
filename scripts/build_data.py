"""Part 1 pipeline: download -> clean spot spreads -> features -> calendar -> futures (if exported) -> figures.

Usage:  .venv/bin/python scripts/build_data.py [--refresh]
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt

from oilrv import futures
from oilrv.calendars import build_events, event_features
from oilrv.config import ROOT, path
from oilrv.data import build_clean, build_features, data_report
from oilrv.plotting import REF_LINE, SERIES, save, use_style

SPREAD_LABELS = {"crack321": "3-2-1 crack", "brent_wti": "Brent − WTI", "ho_rb": "Diesel − gasoline (HO − RB)"}


def plot_prices_and_spreads(px, sp, ev) -> None:
    use_style()
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for i, (col, lab) in enumerate([("cl_bbl", "WTI"), ("brent_bbl", "Brent"), ("rb_bbl", "NYH gasoline"),
                                    ("ho_bbl", "NYH ULSD")]):
        ax[0].plot(sp.index, sp[col], color=SERIES[i], lw=0.7, label=lab)
    ax[0].axhline(0, color=REF_LINE, lw=0.6)
    ax[0].set_ylabel("$/bbl")
    ax[0].set_title("Spot legs in $/bbl (FRED, EIA)")
    ax[0].legend(loc="upper left", ncols=4)
    hormuz = ev.loc[(ev["type"] == "HORMUZ") & ev["name"].str.startswith("closure"), "date"].min()
    for a, (name, lab) in zip(ax[1:], SPREAD_LABELS.items()):
        a.plot(sp.index, sp[name], color=SERIES[0], lw=0.7)
        a.axhline(sp[name].median(), color=REF_LINE, lw=0.7, ls="--")
        a.axvline(hormuz, color=REF_LINE, lw=0.7, ls=":")
        a.set_ylabel("$/bbl")
        a.set_title(f"{lab} (dashed: full-sample median; dotted: Hormuz closure, 28 Feb 2026)")
    fig.tight_layout()
    save(fig, "01_prices_and_spreads.png")


def plot_futures_vs_spot(fut, sp) -> None:
    """The futures spreads at market level (unadjusted, held month) and back-adjusted, against spot."""
    use_style()
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for a, (name, lab) in zip(ax, SPREAD_LABELS.items()):
        a.plot(sp.index, sp[name], color=SERIES[1], lw=0.6, alpha=0.8, label="Spot (FRED/EIA)")
        a.plot(fut.index, fut[f"{name}_unadj"], color=SERIES[0], lw=0.7, label="Futures, held month (market level)")
        a.plot(fut.index, fut[name], color=SERIES[2], lw=0.7, label="Futures, additive back-adjusted (signals, P&L)")
        a.axhline(0, color=REF_LINE, lw=0.6)
        a.set_ylabel("$/bbl")
        a.set_title(f"{lab}: futures (Bloomberg) and spot")
        a.legend(loc="lower left", ncols=3, fontsize=7.5)
    fig.tight_layout()
    save(fig, "01_futures_vs_spot.png")


def main(refresh: bool = False) -> dict:
    px, sp = build_clean(refresh=refresh)
    feats = build_features(sp.index, refresh=refresh)
    ev = build_events(refresh=refresh)
    cal = event_features(sp.index, ev)
    cal.to_csv(path("clean_dir") / "event_features.csv")

    rep = data_report(px, sp, feats)
    rep["events"] = ev["type"].value_counts().to_dict()
    rep["opec_events_verified"] = ev.loc[ev["type"] == "OPEC", "verified"].value_counts().to_dict()
    rep["pct_days_in_event_blackout"] = round(float(100 * cal["event_blackout"].mean()), 1)
    rep["futures_available"] = futures.available()
    if rep["futures_available"]:
        fut = futures.build_futures()
        ff = build_features(fut.index, refresh=False, name="features_futures.csv")
        ff[["brent_m1m6", "brent_m1"]] = fut[["brent_m1m6", "brent_m1"]]   # settles: known at the close
        ff.to_csv(path("clean_dir") / "features_futures.csv")
        event_features(fut.index, ev).to_csv(path("clean_dir") / "event_features_futures.csv")
        names = list(SPREAD_LABELS)
        rep["futures_export"] = futures.export_file().name
        rep["futures_first_date"] = fut.index[0].date().isoformat()
        rep["futures_last_date"] = fut.index[-1].date().isoformat()
        rep["futures_n_days"] = len(fut)
        rep["futures_rolls"] = {c: int(fut[c].sum()) for c in fut.columns if c.startswith("rolled_")}
        rep["futures_stale_leg_days"] = {c: int(fut[c].sum()) for c in fut.columns if c.startswith("stale_")}
        rep["futures_latest_unadj"] = {n: round(float(fut[f"{n}_unadj"].iloc[-1]), 2) for n in names}
        rep["futures_median_unadj"] = {n: round(float(fut[f"{n}_unadj"].median()), 2) for n in names}
        rep["futures_range_unadj"] = {n: [round(float(fut[f"{n}_unadj"].min()), 2),
                                          round(float(fut[f"{n}_unadj"].max()), 2)] for n in names}
        rep["futures_range_adjusted"] = {n: [round(float(fut[n].min()), 2), round(float(fut[n].max()), 2)]
                                         for n in names}
        both = fut.index.intersection(sp.index)
        rep["futures_vs_spot_daily_change_corr"] = {
            n: round(float(fut.loc[both, n].diff().corr(sp.loc[both, n].diff())), 3) for n in names}
        plot_futures_vs_spot(fut, sp)

    tables = ROOT / "output" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "data_summary.json").write_text(json.dumps(rep, indent=2))
    plot_prices_and_spreads(px, sp, ev)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download everything")
    print(json.dumps(main(ap.parse_args().refresh), indent=2))
