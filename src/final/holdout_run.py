"""Phase 7 step 2 — THE single holdout run (PLAN hard rule 2, [F14]). Runs at most once.

Preconditions (all checked, otherwise it refuses):
* ``outputs/freeze.json`` exists and ``approved_by_user`` is true (recorded after the user approved in chat);
* config.yaml and every ``src/**/*.py`` hash equal the frozen hashes (no re-tuning after the freeze);
* no ``outputs/HOLDOUT_DONE.json`` lock (a completed run can never be repeated);
* ``--confirm HOLDOUT`` on the command line.

It then, with ``final=True``: evaluates the frozen Stage 4 models (+ benchmarks) on holdout origins and saves the
holdout payoff forecasts; runs Stage 5 OOS on the holdout; runs the frozen strategy variants on the holdout
window; and re-runs the Q1–Q3 headline tables on the full sample as a confirmatory "extended sample" [F1].
Results are written to a temporary folder and moved to ``outputs/tables/holdout/`` only if everything succeeds;
the lock is written last. If it fails part-way nothing is kept, so no holdout result has been seen.

Run:  python -m src.final.holdout_run --confirm HOLDOUT
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone

import pandas as pd

from src.final.freeze import FREEZE, code_hashes, sha256
from src.utils.io import ROOT, load_config, project_path, write_parquet

LOCK = project_path("outputs", "HOLDOUT_DONE.json")
OUT = project_path("outputs", "tables", "holdout")


def preconditions() -> dict:
    if LOCK.exists():
        raise SystemExit(f"holdout already run ({json.loads(LOCK.read_text())['completed_utc']}) — it cannot be repeated")
    if not FREEZE.exists():
        raise SystemExit("no outputs/freeze.json — run python -m src.final.freeze first")
    rec = json.loads(FREEZE.read_text())
    if not rec.get("approved_by_user"):
        raise SystemExit("freeze not approved by the user — record approval only after the user says so in chat")
    if rec["config_sha256"] != sha256(ROOT / "config.yaml"):
        raise SystemExit("config.yaml changed since the freeze")
    changed = [f for f, h in code_hashes().items() if rec["code_sha256"].get(f) != h]
    changed += [f for f in rec["code_sha256"] if f not in code_hashes()]
    if changed:
        raise SystemExit(f"code changed since the freeze: {changed}")
    return rec


def run(cfg: dict, tmp) -> dict:
    from src.models import stage1, stage2, stage3, stage4, stage5
    from src.strategy.run import run_strategy

    panel = pd.read_parquet(project_path("data", "processed", "panel_monthly.parquet"))
    fz, preds, horizons = cfg["frozen"], cfg["predictors"], cfg["horizons"]
    models = {int(k): v for k, v in fz["stage4_models"].items()}

    fcs, metrics, _ = stage4.run_oos(panel, preds, cfg, final=True, models=models)
    metrics.to_csv(tmp / "stage4_oos_metrics_holdout.csv", index=False)
    for label, fc in fcs.items():
        fc.to_csv(tmp / f"stage4_oos_forecasts_holdout_{label.replace(' ', '_').replace('=', '')}.csv")
    write_parquet(fcs["payoff k=0"], "data/processed/payoff_forecasts_holdout.parquet")

    met5, fc5 = stage5.oos(panel, cfg["stage5"]["predictors"], horizons, cfg, final=True)
    met5[met5["variant"].isin(fz["stage5_variants"])].to_csv(tmp / "stage5_oos_holdout.csv", index=False)
    fc5.to_csv(tmp / "stage5_oos_forecasts_holdout.csv", index=False)

    table, gates = run_strategy(cfg, final=True, out_dir=tmp)
    table = table.loc[[v for v in fz["strategy_variants"] if v in table.index]]
    table.to_csv(tmp / "strategy_holdout_frozen_variants.csv")

    # extended-sample (full 2008 → sample.end) confirmatory Q1–Q3 headline tables, no re-tuning [F1]
    stage1.persistence(panel, "vrp_exante", horizons, cfg, final=True).to_csv(tmp / "extended_stage1_persistence.csv")
    stage1.single_predictors(panel, "vrp_exante", preds, horizons, cfg, final=True).to_csv(
        tmp / "extended_stage1_single.csv", index=False)
    stage2.full_ols(panel, "vrp_exante", preds, horizons, cfg, final=True, bootstrap=False).to_csv(
        tmp / "extended_stage2_ols.csv", index=False)
    stage3.break_tests(panel, "vrp_exante", preds, cfg, final=True).to_csv(tmp / "extended_stage3_break_test.csv", index=False)
    return dict(strategy_gates=[(n, bool(ok), d) for n, ok, d in gates])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", required=True, help='must be exactly "HOLDOUT"')
    args = ap.parse_args(argv)
    if args.confirm != "HOLDOUT":
        raise SystemExit('refusing: pass --confirm HOLDOUT')
    rec = preconditions()
    cfg = load_config()
    tmp = project_path("outputs", "tables", ".holdout_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        info = run(cfg, tmp)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        project_path("data", "processed", "payoff_forecasts_holdout.parquet").unlink(missing_ok=True)
        raise
    shutil.rmtree(OUT, ignore_errors=True)
    tmp.rename(OUT)
    LOCK.write_text(json.dumps(dict(completed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    freeze_config_sha256=rec["config_sha256"], **info), indent=2))
    print(f"[ok  ] holdout run complete → {OUT.relative_to(ROOT)}; lock written. Do not re-tune (PLAN rule 9).")


if __name__ == "__main__":
    main()
