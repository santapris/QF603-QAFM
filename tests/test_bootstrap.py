import numpy as np
import pandas as pd

from src.utils.bootstrap import block_indices, mbb_pvalues


def test_block_indices_shape_and_contiguity():
    rng = np.random.default_rng(0)
    idx = block_indices(100, 6, rng)
    assert len(idx) == 100 and idx.min() >= 0 and idx.max() < 100
    assert all((idx[i + 1] - idx[i]) % 100 == 1 for i in range(5))   # first block is contiguous


def test_pvalues_detect_signal():
    rng = np.random.default_rng(1)
    T = 150
    x = pd.Series(rng.standard_normal(T), name="x")
    strong = mbb_pvalues(0.8 * x + pd.Series(rng.standard_normal(T)), x, lags=3, reps=500, seed=1)
    assert strong["x"] < 0.01
    assert ((strong >= 0) & (strong <= 1)).all()


def test_pvalues_under_null_are_not_small():
    """Across null samples, p-values are spread out and rarely below 5%."""
    pvals = []
    for i in range(40):
        rng = np.random.default_rng(100 + i)
        x = pd.Series(rng.standard_normal(150), name="x")
        y = pd.Series(rng.standard_normal(150))
        pvals.append(mbb_pvalues(y, x, lags=3, reps=200, seed=i)["x"])
    pvals = np.array(pvals)
    assert pvals.mean() > 0.3
    assert (pvals < 0.05).mean() <= 0.2


def test_reproducible_with_seed():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.standard_normal(80), name="x")
    y = pd.Series(rng.standard_normal(80))
    a = mbb_pvalues(y, x, lags=2, reps=200, seed=7)
    b = mbb_pvalues(y, x, lags=2, reps=200, seed=7)
    pd.testing.assert_series_equal(a, b)
