import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.align import combine, harmonize_kalshi, harmonize_polymarket


def test_harmonize_kalshi_basic():
    raw = pd.DataFrame(
        [
            {
                "ticker": "FED-24DEC-T4.50",
                "title": "Will the Fed cut to 4.50% by Dec 2024?",
                "category": "Economics",
                "close_time": "2024-12-18T19:00:00Z",
                "result": "yes",
                "last_price_dollars": "0.82",
                "yes_bid_dollars": "0.80",
                "yes_ask_dollars": "0.83",
                "volume_fp": "15234",
            }
        ]
    )
    out = harmonize_kalshi(raw)
    assert out.loc[0, "platform"] == "kalshi"
    assert out.loc[0, "outcome"] == 1.0
    assert out.loc[0, "implied_prob"] == 0.82


def test_harmonize_polymarket_basic():
    raw = pd.DataFrame(
        [
            {
                "id": "123",
                "question": "Will BTC close above 100k on Dec 31?",
                "category": "Crypto",
                "closed_time": "2024-12-31T23:59:00Z",
                "outcomes": '["Yes","No"]',
                "outcome_prices": '["0","1"]',
                "clob_token_ids": '["tok_yes","tok_no"]',
                "volume": "500000",
            }
        ]
    )
    out = harmonize_polymarket(raw)
    assert out.loc[0, "platform"] == "polymarket"
    assert out.loc[0, "outcome"] == 0.0


def test_combine_drops_unresolved_rows():
    k = pd.DataFrame(
        {
            "platform": ["kalshi"],
            "market_id": ["A"],
            "title": ["t"],
            "category": ["c"],
            "close_time": [pd.Timestamp.now('UTC')],
            "outcome": [float("nan")],  # should be dropped
            "implied_prob": [0.5],
            "volume": [10],
        }
    )
    p = pd.DataFrame(
        {
            "platform": ["polymarket"],
            "market_id": ["B"],
            "title": ["t2"],
            "category": ["c2"],
            "close_time": [pd.Timestamp.now('UTC')],
            "outcome": [1.0],
            "implied_prob": [0.9],
            "volume": [20],
        }
    )
    out = combine(k, p)
    assert len(out) == 1
    assert out.loc[0, "market_id"] == "B"
