import numpy as np
import pandas as pd
import pytest

from src.models.stage2 import collinearity, expanding_folds, full_ols, pca, ridge_cv, ridge_path
from src.utils.dates import month_end_trading_days
from src.utils.io import load_config


@pytest.fixture
def cfg():
    c = load_config()
    c["stage2"]["ridge_log10_alpha"] = [-2, 3, 6]          # small grid keeps tests fast
    c["bootstrap"]["reps"] = 200
    return c


def factor_panel(seed=0):
    """Two latent factors drive six predictors; VRP loads on factor 1."""
    rng = np.random.default_rng(seed)
    idx = month_end_trading_days("2008-01-01", "2025-08-31")
    n = len(idx)
    f = rng.standard_normal((n, 2))
    X = np.column_stack([f[:, 0] + 0.2 * rng.standard_normal(n) for _ in range(3)] +
                        [f[:, 1] + 0.2 * rng.standard_normal(n) for _ in range(3)])
    v = np.zeros(n)
    for t in range(1, n):
        v[t] = 0.4 * v[t - 1] + 0.5 * f[t - 1, 0] + rng.standard_normal()
    cols = ["a1", "a2", "a3", "b1", "b2", "b3"]
    return pd.DataFrame(np.column_stack([v, X]), index=idx, columns=["vrp_exante"] + cols), cols


def test_collinearity_flags_correlated_block(cfg):
    panel, cols = factor_panel()
    c = collinearity(panel, "vrp_exante", cols, cfg)
    assert (c.loc[cols, "vif"] > 5).all()
    assert c.loc["a1", "a2"] > 0.8 and abs(c.loc["a1", "b1"]) < 0.3


def test_pca_recovers_two_factors(cfg):
    panel, cols = factor_panel()
    out = pca(panel, "vrp_exante", cols, [1], cfg)
    assert out["variance"].loc["PC2", "cumulative"] > 0.9
    top1 = set(out["loadings"]["PC1"].abs().nlargest(3).index)
    assert top1 in ({"a1", "a2", "a3"}, {"b1", "b2", "b3"})
    reg = out["regressions"]
    pc_a = "PC1" if top1 == {"a1", "a2", "a3"} else "PC2"
    t = reg[(reg["k"] == 1) & (reg["m"] == 2) & (reg["term"] == pc_a)]["t_nw"].iloc[0]
    assert abs(t) > 3


def test_full_ols_shapes(cfg):
    panel, cols = factor_panel()
    t = full_ols(panel, "vrp_exante", cols, [1, 3], cfg)
    assert set(t["term"]) == {"const", "lag", *cols} and set(t["k"]) == {1, 3}
    assert t["p_mbb"].between(0, 1).all()


def test_expanding_folds_never_train_on_unobserved_targets(cfg):
    panel, _ = factor_panel()
    k = 3
    for origin, tr in expanding_folds(panel.loc[:"2020-12-31"], "vrp_exante", k, 60, cfg):
        last = tr.max()
        assert (last.to_period("M") + k) <= origin.to_period("M")      # target of every train row known by origin
        assert len(tr) >= 60


def test_ridge_cv_and_path(cfg):
    panel, cols = factor_panel()
    cv, best = ridge_cv(panel, "vrp_exante", cols, [1], cfg)
    assert best.loc[1, "n_folds"] > 50 and best.loc[1, "alpha"] in cv["alpha"].to_numpy()
    path = ridge_path(panel, "vrp_exante", cols, 1, cfg)
    norms = np.sqrt((path[cols] ** 2).sum(axis=1))
    assert norms.is_monotonic_decreasing                           # more shrinkage → smaller coefficients
