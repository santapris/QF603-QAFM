import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.measures.calibration import brier_score, brier_score_baseline, bucket_calibration


def _toy_panel():
    # Deliberately biased: every market priced at 0.1 ("longshot") actually
    # resolves YES only 2% of the time (overpriced favorite-longshot style
    # bias); every market priced at 0.9 resolves YES 98% of the time.
    rows = []
    for _ in range(100):
        rows.append({"implied_prob": 0.1, "outcome": 1.0 if _ < 2 else 0.0})
    for _ in range(100):
        rows.append({"implied_prob": 0.9, "outcome": 1.0 if _ < 98 else 0.0})
    return pd.DataFrame(rows)


def test_bucket_calibration_shows_bias_direction():
    panel = _toy_panel()
    curve = bucket_calibration(panel, n_buckets=10)
    low = curve[curve["mean_price"] < 0.5].iloc[0]
    high = curve[curve["mean_price"] > 0.5].iloc[0]
    # longshot bucket: price > realized frequency (overpriced)
    assert low["bias"] > 0
    # favorite bucket: price < realized frequency (underpriced)
    assert high["bias"] < 0


def test_brier_score_perfect_forecast_is_zero():
    perfect = pd.DataFrame({"implied_prob": [1.0, 0.0, 1.0], "outcome": [1.0, 0.0, 1.0]})
    assert brier_score(perfect) == 0.0


def test_brier_score_baseline_matches_variance_style_formula():
    panel = pd.DataFrame({"outcome": [1.0, 0.0, 1.0, 0.0]})
    # base rate = 0.5, so baseline Brier = mean((0.5 - outcome)^2) = 0.25
    assert abs(brier_score_baseline(panel) - 0.25) < 1e-9
