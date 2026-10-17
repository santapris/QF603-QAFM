"""
Stage 1 measurement: is the market's implied probability well-calibrated
against what actually happened?

This is the prediction-market analogue of VRP = implied variance - realized
variance. Here: bias = implied probability - realized outcome, averaged
within price buckets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bucket_calibration(df: pd.DataFrame, n_buckets: int = 10) -> pd.DataFrame:
    """Group markets into price buckets and compare mean price to mean outcome.

    A perfectly calibrated market has mean_outcome == mean_price in every
    bucket (points sit on the 45-degree line). The favorite-longshot bias
    shows up as mean_outcome < mean_price in the low-price buckets (longshots
    overpriced) and mean_outcome > mean_price in the high-price buckets
    (favorites underpriced).
    """
    work = df.dropna(subset=["implied_prob", "outcome"]).copy()
    work["bucket"] = pd.cut(work["implied_prob"], bins=np.linspace(0, 1, n_buckets + 1), include_lowest=True)

    agg = work.groupby("bucket", observed=True).agg(
        mean_price=("implied_prob", "mean"),
        mean_outcome=("outcome", "mean"),
        n=("outcome", "size"),
    )
    agg["bias"] = agg["mean_price"] - agg["mean_outcome"]
    agg["se"] = np.sqrt(agg["mean_outcome"] * (1 - agg["mean_outcome"]) / agg["n"])
    return agg.reset_index()


def brier_score(df: pd.DataFrame) -> float:
    """Mean squared error of price vs. outcome -- standard forecast-accuracy
    score for probabilistic binary predictions. Lower is better; a
    reasonable reference point is the Brier score of always predicting the
    unconditional base rate."""
    work = df.dropna(subset=["implied_prob", "outcome"])
    return float(np.mean((work["implied_prob"] - work["outcome"]) ** 2))


def brier_score_baseline(df: pd.DataFrame) -> float:
    """Brier score of the naive 'predict the sample base rate every time'
    forecast -- the direct analogue of the VRP project's expanding-mean
    benchmark. Any model/bucket-correction should beat this out of sample,
    not just beat zero."""
    work = df.dropna(subset=["outcome"])
    base_rate = work["outcome"].mean()
    return float(np.mean((base_rate - work["outcome"]) ** 2))
