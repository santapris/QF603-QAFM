"""Single pipeline entrypoint: runs every phase in order.

Each step is ``module:function`` under ``src/``; the function is called as ``fn(cfg=cfg, force=force)``.
A step is skipped when all its outputs already exist (unless ``--force``). Steps whose module has not
been written yet are reported as "not implemented" and skipped.

The Phase 7 final holdout run is deliberately NOT part of this pipeline (hard rule 2).

Usage:
    python run_all.py --dry-run            # list steps and their status
    python run_all.py                      # run everything that is implemented and not yet done
    python run_all.py --phase 2 --force    # re-run one phase
    python run_all.py --only har vrp       # run named steps
"""

from __future__ import annotations

import argparse
import glob
import importlib
import importlib.util
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from src.utils.io import load_config, project_path


@dataclass
class Step:
    name: str
    phase: str
    target: str                       # "src.package.module:function"
    outputs: list[str] = field(default_factory=list)   # glob patterns relative to project root
    kind: str = "build"               # raw: downloads only; pullbuild: reuses raw files, rebuilds processed; build

    def module_name(self) -> str:
        return self.target.split(":")[0]

    def implemented(self) -> bool:
        if importlib.util.find_spec(self.module_name()) is None:
            return False
        return hasattr(importlib.import_module(self.module_name()), self.target.split(":")[1])

    def done(self) -> bool:
        return bool(self.outputs) and all(glob.glob(str(project_path(p))) for p in self.outputs)


STEPS: list[Step] = [
    # Phase 1 — data (kind "raw" = downloads only; "pullbuild" = reuses raw files and rebuilds processed ones)
    Step("audit", "1", "src.data.audit_coverage:main", ["outputs/tables/data_coverage.csv"], kind="raw"),
    Step("optionmetrics", "1", "src.data.pull_optionmetrics:main", ["data/raw/optionmetrics/*.parquet"], kind="raw"),
    Step("optionmetrics_check", "1", "src.data.check_optionmetrics:main", ["outputs/tables/optionmetrics_pull_summary.csv"]),
    Step("taq", "1", "src.data.pull_taq:main", ["data/raw/taq/spy_5min_*.parquet"], kind="raw"),
    Step("taq_check", "1", "src.data.check_taq:main", ["outputs/tables/taq_pull_summary.csv"]),
    Step("bloomberg", "1", "src.data.load_bloomberg:main", ["data/processed/bloomberg_daily.parquet"]),
    Step("fred", "1", "src.data.pull_fred:main", ["data/raw/fred/*.csv", "data/processed/fred_monthly.parquet"],
         kind="pullbuild"),
    Step("cftc", "1", "src.data.pull_cftc:main", ["data/raw/cftc/*.csv", "data/processed/cftc_monthly.parquet"],
         kind="pullbuild"),
    Step("hkm", "1", "src.data.load_hkm:main", ["data/processed/hkm_monthly.parquet"]),
    # Phase 1.5 — throwaway prototype
    Step("prototype", "1.5", "src.models.prototype:main", []),
    # Phase 2 — measures
    Step("rv", "2", "src.measures.rv:main", ["data/processed/rv_daily.parquet"]),
    Step("mfiv", "2", "src.measures.mfiv:main", ["data/processed/iv_monthly.parquet"]),
    Step("har", "2", "src.measures.har:main", ["data/processed/har_forecasts.parquet"]),
    Step("vrp", "2", "src.measures.vrp:main", ["data/processed/panel_monthly.parquet"]),
    # Phase 3 — Q1–Q3 (pre-holdout)
    Step("stage1", "3", "src.models.stage1:main", ["outputs/tables/stage1_*.csv"]),
    Step("stage2", "3", "src.models.stage2:main", ["outputs/tables/stage2_*.csv"]),
    Step("stage3", "3", "src.models.stage3:main", ["outputs/tables/stage3_*.csv"]),
    # Phase 4 — Q4–Q5 (validation only)
    Step("stage4", "4", "src.models.stage4:main", ["outputs/tables/stage4_*.csv"]),
    Step("stage5", "4", "src.models.stage5:main", ["outputs/tables/stage5_*.csv"]),
    # Phase 6 — strategy (validation only)
    Step("strategy_data", "6", "src.data.pull_optionmetrics_daily:main",
         ["data/processed/strategy_options_daily.parquet"], kind="raw"),
    Step("strategy", "6", "src.strategy.backtest:main", ["outputs/tables/strategy_validation.csv"]),
    # Phase 7 — report (the freeze and the single holdout run are separate, manual commands)
    Step("report", "7", "src.report.build:main", ["report/report.md"]),
]


def status(step: Step) -> str:
    if not step.implemented():
        return "not implemented"
    return "done" if step.done() else "pending"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list steps and status, run nothing")
    ap.add_argument("--force", action="store_true", help="re-run steps even if outputs exist")
    ap.add_argument("--phase", nargs="*", help="only run steps in these phases")
    ap.add_argument("--only", nargs="*", help="only run these step names")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild everything from raw files: skip raw downloads, re-run all other steps")
    args = ap.parse_args(argv)

    cfg = load_config()
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    known = {s.name for s in STEPS}
    unknown = set(args.only or []) - known
    if unknown:
        ap.error(f"unknown step(s): {sorted(unknown)}; known: {sorted(known)}")

    selected = [s for s in STEPS
                if (not args.phase or s.phase in args.phase) and (not args.only or s.name in args.only)]

    if args.dry_run:
        print(f"{'phase':<6}{'step':<22}{'status':<18}target")
        for s in selected:
            print(f"{s.phase:<6}{s.name:<22}{status(s):<18}{s.target}")
        return 0

    for s in selected:
        st = status(s)
        if args.rebuild:
            if s.kind == "raw" or st == "not implemented":
                print(f"[skip] {s.name}: {'raw download' if s.kind == 'raw' else st}")
                continue
            module, fn = s.target.split(":")
            print(f"[run ] {s.name} (rebuild)", flush=True)
            getattr(importlib.import_module(module), fn)(cfg=cfg, force=s.kind != "pullbuild")
            continue
        if st == "not implemented" or (st == "done" and not args.force):
            print(f"[skip] {s.name}: {st}")
            continue
        module, fn = s.target.split(":")
        print(f"[run ] {s.name} ({s.target})", flush=True)
        t0 = time.time()
        getattr(importlib.import_module(module), fn)(cfg=cfg, force=args.force)
        print(f"[ok  ] {s.name} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
