"""
Stage 5: out-of-sample test -- the one that actually answers "is this
predictable, or does the bias only look real in-sample."

Mirrors the VRP project's out-of-sample R^2 against an expanding-mean
benchmark: fit the Stage 2 calibration logit on a TRAIN slice only, then
score it against the raw market price on a held-out TEST slice, using
Brier score (mean squared error of a probability forecast) as the metric.

KNOWN DATA CAVEAT (found while building this): an unfiltered pull of
Kalshi's settled-markets feed is dominated by very high-frequency,
short-duration crypto and sports micro-markets -- even a 10k-market pull
spans only a few days of close_time, regardless of how far back you ask
via min/max_settled_ts (see kalshi_client.py docstring). A calendar-date
split on a pull like that is really closer to a random split than a
genuine "does this generalize across market regimes/time" test. Before
trusting this stage's result, pull a larger, category-diverse,
longer-horizon sample (e.g. filter to specific non-crypto, non-sports
series, or pull enough pages to get real month-to-month spread) --
split_by_time() and split_by_fraction() both work unchanged once you do.
"""
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from src.measures.calibration import brier_score, brier_score_baseline
from src.models.stage2_logit import fit_calibration_logit


def split_by_time(df: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = markets that closed before cutoff, test = on/after cutoff."""
    train = df[df["close_time"] < cutoff].copy()
    test = df[df["close_time"] >= cutoff].copy()
    return train, test


def split_by_fraction(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by close_time and take the first train_frac as train, rest as test.

    Use this when the pulled sample doesn't span enough calendar time for
    split_by_time() to be meaningful (see module docstring) -- it still
    tests generalization to unseen markets, just not across a long horizon.
    """
    ordered = df.sort_values("close_time").reset_index(drop=True)
    cut = int(len(ordered) * train_frac)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def run_oos_test(train: pd.DataFrame, test: pd.DataFrame, include_price_sq: bool = True) -> dict:
    """Fit the calibration logit on train, score it (and the naive raw-price
    forecast) on test. Returns a dict with everything needed to answer
    'did the correction actually help, out of sample.'
    """
    train = train.dropna(subset=["implied_prob", "outcome"])
    test = test.dropna(subset=["implied_prob", "outcome"])

    if len(train) < 30 or len(test) < 30:
        raise ValueError(
            f"train={len(train)}, test={len(test)} -- too small for a meaningful "
            "logit fit / OOS comparison, pull more data or widen the split"
        )

    result = fit_calibration_logit(train, include_price_sq=include_price_sq)

    X_test = pd.DataFrame({"price": test["implied_prob"]})
    if include_price_sq:
        X_test["price_sq"] = test["implied_prob"] ** 2
    X_test = sm.add_constant(X_test, has_constant="add")

    test = test.copy()
    test["corrected_prob"] = result.predict(X_test)

    brier_naive = brier_score(test.rename(columns={"implied_prob": "implied_prob"}))
    brier_corrected = float(((test["corrected_prob"] - test["outcome"]) ** 2).mean())
    brier_base = brier_score_baseline(test)

    return {
        "n_train": len(train),
        "n_test": len(test),
        "brier_naive_price": brier_naive,
        "brier_corrected": brier_corrected,
        "brier_base_rate_benchmark": brier_base,
        # positive => correction beats trusting the raw market price OOS
        "improvement_vs_naive": brier_naive - brier_corrected,
        "logit_result": result,
    }
