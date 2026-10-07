"""Phase 7 step 1 — freeze (pre-register) every specification before the single holdout run.

``python -m src.final.freeze``            → lists open decisions; if none, writes ``outputs/freeze.json`` with
                                          the SHA-256 of config.yaml and of every ``src/**/*.py`` file, the
                                          frozen choices, and ``approved_by_user: false``; appends a D-entry.
``python -m src.final.freeze --record-approval "<who/when>"``
                                        → only after the user approves the freeze in chat: marks it approved.
Any later change to config.yaml or code invalidates the freeze (the holdout runner re-checks the hashes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

from src.utils.io import ROOT, load_config, project_path

FREEZE = project_path("outputs", "freeze.json")


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes() -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src").rglob("*.py"))}


def open_decisions(cfg: dict) -> list[str]:
    """Everything that must be decided (🧑) and present before the holdout may run."""
    todo = []
    if cfg["har"]["headline"] is None:
        todo.append("har.headline (Phase 2c)")
    if cfg["strategy"]["option_fee_per_contract"] is None:
        todo.append("strategy.option_fee_per_contract (Phase 6)")
    for key in ["stage4_models", "stage5_variants", "strategy_variants", "n_trials"]:
        if cfg["frozen"][key] in (None, [], {}):
            todo.append(f"frozen.{key}")
    if cfg["frozen"]["strategy_variants"] and "S2" in cfg["frozen"]["strategy_variants"] and not cfg["strategy"]["s2_model"]:
        todo.append("strategy.s2_model (S2 is in the frozen variants)")
    required = ["data/processed/panel_monthly.parquet", "outputs/tables/stage4_selection.csv",
                "outputs/tables/stage3_regime_note.csv", "outputs/tables/strategy_validation.csv"]
    todo += [f"missing output {r}" for r in required if not project_path(r).exists()]
    return todo


def write_freeze(cfg: dict) -> dict:
    rec = dict(frozen_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               config_sha256=sha256(ROOT / "config.yaml"), code_sha256=code_hashes(),
               frozen=cfg["frozen"], har_headline=cfg["har"]["headline"], strategy=cfg["strategy"],
               approved_by_user=False, approval_note=None)
    FREEZE.write_text(json.dumps(rec, indent=2, default=str))
    return rec


def record_approval(note: str) -> dict:
    rec = json.loads(FREEZE.read_text())
    if rec["config_sha256"] != sha256(ROOT / "config.yaml") or rec["code_sha256"] != code_hashes():
        raise SystemExit("config or code changed since the freeze — re-freeze before recording approval")
    rec.update(approved_by_user=True, approval_note=note,
               approved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    FREEZE.write_text(json.dumps(rec, indent=2, default=str))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-approval", metavar="NOTE", help="only after the user approved the freeze in chat")
    args = ap.parse_args(argv)
    if args.record_approval:
        record_approval(args.record_approval)
        print(f"[ok  ] freeze approved: {args.record_approval}")
        return 0
    cfg = load_config()
    todo = open_decisions(cfg)
    if todo:
        print("Cannot freeze yet — open items:\n  - " + "\n  - ".join(todo))
        return 1
    rec = write_freeze(cfg)
    print(f"[ok  ] outputs/freeze.json written (config {rec['config_sha256'][:12]}…, {len(rec['code_sha256'])} code files).")
    print("🧑 Ask the user to approve the freeze, then: python -m src.final.freeze --record-approval \"<note>\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
