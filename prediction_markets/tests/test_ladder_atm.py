import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ladder_atm import build_atm_panel, build_full_ladder_panel, select_boundary_strikes


def _ladder_day(event_ticker: str, strikes: list[float], results: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "event_ticker": event_ticker,
        "series_ticker": "KXTEST",
        "ticker": [f"{event_ticker}-{s}" for s in strikes],
        "title": "t",
        "floor_strike": strikes,
        "result": results,
        "open_time": pd.Timestamp("2026-01-01T12:00:00Z"),
        "close_time": pd.Timestamp("2026-01-02T04:00:00Z"),
    })


def test_select_boundary_strikes_finds_clean_transition():
    day = _ladder_day("D1", [1.0, 2.0, 3.0, 4.0, 5.0], ["yes", "yes", "no", "no", "no"])
    out = select_boundary_strikes(day, max_strikes=2)
    assert list(out["floor_strike"]) == [2.0, 3.0]


def test_select_boundary_strikes_respects_max_strikes():
    day = _ladder_day("D1", [1.0, 2.0, 3.0, 4.0], ["yes", "yes", "no", "no"])
    out = select_boundary_strikes(day, max_strikes=1)
    # with max_strikes=1, the single strike returned is the first one on the new-result
    # side of the transition (floor_strike=3.0, where "no" first appears)
    assert len(out) == 1
    assert out["floor_strike"].iloc[0] == 3.0


def test_select_boundary_strikes_empty_on_no_transition():
    day = _ladder_day("D1", [1.0, 2.0, 3.0], ["no", "no", "no"])
    out = select_boundary_strikes(day)
    assert out.empty


def test_select_boundary_strikes_empty_on_multiple_transitions():
    # not a well-formed single-boundary ladder -- e.g. a two-sided "range" market
    day = _ladder_day("D1", [1.0, 2.0, 3.0, 4.0], ["no", "yes", "no", "yes"])
    out = select_boundary_strikes(day)
    assert out.empty


def test_select_boundary_strikes_empty_on_too_few_strikes():
    day = _ladder_day("D1", [1.0], ["yes"])
    out = select_boundary_strikes(day)
    assert out.empty


def test_build_atm_panel_assembles_rows_and_skips_missing_snapshots():
    markets = pd.concat([
        _ladder_day("D1", [1.0, 2.0, 3.0], ["yes", "yes", "no"]),
        _ladder_day("D2", [1.0, 2.0, 3.0], ["no", "no", "no"]),  # no transition -> skipped entirely
    ], ignore_index=True)

    def fake_snapshot(series_ticker, ticker, open_time, close_time, frac):
        if ticker == "D1-2.0":
            return None  # simulate "never traded before the target time"
        return dict(mid=0.42, volume_fp=3.0)

    panel = build_atm_panel(markets, fake_snapshot, frac=0.4, max_strikes_per_day=2)
    assert len(panel) == 1  # D1-2.0 skipped (None snapshot), D1-3.0 kept, D2 had no transition
    row = panel.iloc[0]
    assert row["market_id"] == "D1-3.0"
    assert row["outcome"] == 0.0
    assert row["implied_prob"] == 0.42
    assert row["category"] == "KXTEST"


def test_build_full_ladder_panel_keeps_every_strike_no_outcome_selection():
    # unlike build_atm_panel, every row is a candidate regardless of whether that day's
    # ladder has a clean transition -- D2 (all "no") should still contribute all 3 rows.
    markets = pd.concat([
        _ladder_day("D1", [1.0, 2.0, 3.0], ["yes", "yes", "no"]),
        _ladder_day("D2", [1.0, 2.0, 3.0], ["no", "no", "no"]),
    ], ignore_index=True)

    def fake_snapshot(series_ticker, ticker, open_time, close_time, frac):
        if ticker == "D1-2.0":
            return None
        return dict(mid=0.3, volume_fp=1.0)

    panel = build_full_ladder_panel(markets, fake_snapshot, frac=0.4)
    assert len(panel) == 5  # 6 total rows minus the 1 skipped (None snapshot)
    assert set(panel["market_id"]) == {"D1-1.0", "D1-3.0", "D2-1.0", "D2-2.0", "D2-3.0"}
