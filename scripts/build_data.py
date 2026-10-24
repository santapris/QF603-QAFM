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
        rep["futures_first_date"] = fut.index[0].date().isoformat()
        rep["futures_last_date"] = fut.index[-1].date().isoformat()
        rep["futures_rolls"] = {c: int(fut[c].sum()) for c in fut.columns if c.startswith("rolled_")}

    tables = ROOT / "output" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "data_summary.json").write_text(json.dumps(rep, indent=2))
    plot_prices_and_spreads(px, sp, ev)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download everything")
    print(json.dumps(main(ap.parse_args().refresh), indent=2))
