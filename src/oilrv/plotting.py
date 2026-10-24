"""Shared static-figure style for the report (light mode, print).

Colours are slots from the validated reference palette in fixed order: series 1 is always
blue, series 2 always orange, and so on. Grid and axes stay recessive, and text never takes a series colour.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ROOT

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
REF_LINE = "#8a8984"                 # thresholds / zero lines: neutral, never a series hue
CRITICAL = "#e34948"                 # reserved: stop / failure markers only

FIG_DIR = ROOT / "output" / "figures"
TAB_DIR = ROOT / "output" / "tables"


def use_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.0,
        "font.size": 9, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p)
    plt.close(fig)
    return p


def save_table(df, name: str) -> Path:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    p = TAB_DIR / name
    df.to_csv(p)
    return p
