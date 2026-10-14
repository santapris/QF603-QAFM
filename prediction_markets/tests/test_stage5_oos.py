import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.stage5_oos import run_oos_test, split_by_fraction, split_by_time


def _biased_panel(n=2000, seed=0):
    """Synthetic markets with a real, stable favorite-longshot bias: the
    true resolution probability is a known nonlinear squash of the quoted
    price, so a logit fit on train SHOULD generalize to test."""
    rng = np.random.default_rng(seed)
    price = rng.uniform(0.01, 0.99, n)
    true_prob = price ** 1.8  # longshots overpriced, favorites underpriced
    outcome = (rng.uniform(0, 1, n) < true_prob).astype(float)
    close_time = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"implied_prob": price, "outcome": outcome, "close_time": close_time})


def test_split_by_time():
    df = _biased_panel(100)
    cutoff = df["close_time"].iloc[50]
    train, test = split_by_time(df, cutoff)
    assert (train["close_time"] < cutoff).all()
    assert (test["close_time"] >= cutoff).all()
    assert len(train) + len(test) == len(df)


def test_split_by_fraction():
    df = _biased_panel(100)
    train, test = split_by_fraction(df, train_frac=0.7)
    assert len(train) == 70
    assert len(test) == 30


def test_oos_correction_beats_naive_on_genuine_bias():
    df = _biased_panel(3000, seed=1)
    train, test = split_by_fraction(df, train_frac=0.7)
    result = run_oos_test(train, test)

    # the whole point of stage 5: on data with a real, stable bias, the
    # bias-corrected forecast should out-of-sample beat just trusting price
    assert result["improvement_vs_naive"] > 0
    assert result["brier_corrected"] < result["brier_naive_price"]


def test_oos_raises_on_too_little_data():
    df = _biased_panel(20)
    train, test = split_by_fraction(df, train_frac=0.5)
    try:
        run_oos_test(train, test)
        assert False, "expected ValueError on too-small split"
    except ValueError:
        pass
