import json

import pytest

from src.final import freeze, holdout_run
from src.utils.io import load_config


def test_open_decisions_block_freeze_now():
    todo = freeze.open_decisions(load_config())
    assert any(t.startswith("frozen.stage4_models") for t in todo)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(holdout_run, "FREEZE", tmp_path / "freeze.json")
    monkeypatch.setattr(holdout_run, "LOCK", tmp_path / "HOLDOUT_DONE.json")
    return tmp_path


def good_record():
    from src.utils.io import ROOT
    return dict(config_sha256=freeze.sha256(ROOT / "config.yaml"), code_sha256=freeze.code_hashes(),
                approved_by_user=True)


def test_holdout_refuses_without_freeze(sandbox):
    with pytest.raises(SystemExit, match="no outputs/freeze.json"):
        holdout_run.preconditions()


def test_holdout_refuses_unapproved_or_changed(sandbox):
    rec = good_record() | {"approved_by_user": False}
    (sandbox / "freeze.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="not approved"):
        holdout_run.preconditions()
    rec = good_record() | {"config_sha256": "0" * 64}
    (sandbox / "freeze.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="config.yaml changed"):
        holdout_run.preconditions()
    rec = good_record()
    rec["code_sha256"] = dict(rec["code_sha256"]) | {"src/models/stage4.py": "0" * 64}
    (sandbox / "freeze.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="code changed"):
        holdout_run.preconditions()


def test_holdout_passes_when_frozen_and_refuses_twice(sandbox):
    (sandbox / "freeze.json").write_text(json.dumps(good_record()))
    assert holdout_run.preconditions()["approved_by_user"]
    (sandbox / "HOLDOUT_DONE.json").write_text(json.dumps({"completed_utc": "2026-01-01T00:00:00"}))
    with pytest.raises(SystemExit, match="cannot be repeated"):
        holdout_run.preconditions()


def test_holdout_requires_confirm_word():
    with pytest.raises(SystemExit):
        holdout_run.main(["--confirm", "yes"])
