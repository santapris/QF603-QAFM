"""Phase 7 🚦 reproducibility check: rebuild from raw data and compare with the current outputs.

1. Snapshot every ``data/processed/*.parquet`` and ``outputs/tables/*.csv`` (holdout folder and the network-based
   coverage audit excluded).
2. ``python run_all.py --rebuild [--only …]`` — raw downloads are skipped, everything else is rebuilt.
3. Compare each file with its snapshot (numeric tolerance 1e-9 relative); report identical / changed / missing.

Run:  python -m src.final.repro [--only rv mfiv har …]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from src.utils.io import ROOT, project_path

EXCLUDE = {"data_coverage.csv", "data_coverage_gaps.csv"}


def tracked() -> list:
    files = sorted(project_path("data", "processed").glob("*.parquet")) + \
        sorted(p for p in project_path("outputs", "tables").glob("*.csv") if p.name not in EXCLUDE)
    return files


def read(path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def same(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False, f"shape/columns {a.shape} vs {b.shape}"
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            if not np.allclose(xv, yv, rtol=1e-9, atol=1e-12, equal_nan=True):
                return False, f"column {c}: max |diff| {np.nanmax(np.abs(xv - yv)):.3g}"
        elif not x.astype(str).equals(y.astype(str)):
            return False, f"column {c} differs"
    return True, ""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args(argv)
    snap = project_path("outputs", ".repro_snapshot")
    shutil.rmtree(snap, ignore_errors=True)
    snap.mkdir(parents=True)
    before = {p: snap / p.relative_to(ROOT).as_posix().replace("/", "__") for p in tracked()}
    for src, dst in before.items():
        shutil.copy2(src, dst)
    cmd = [sys.executable, "run_all.py", "--rebuild"] + (["--only", *args.only] if args.only else [])
    subprocess.run(cmd, cwd=ROOT, check=True)
    rows = []
    for src, dst in before.items():
        if not src.exists():
            rows.append((src.relative_to(ROOT), "MISSING after rebuild", ""))
            continue
        ok, why = same(read(dst), read(src))
        rows.append((src.relative_to(ROOT), "identical" if ok else "CHANGED", why))
    shutil.rmtree(snap, ignore_errors=True)
    bad = [r for r in rows if r[1] != "identical"]
    for r in rows:
        print(f"  [{'ok' if r[1] == 'identical' else '!!'}] {r[0]} {r[1]} {r[2]}")
    print(f"🚦 reproducibility: {len(rows) - len(bad)}/{len(rows)} files identical" + ("" if not bad else " — CHECK"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
