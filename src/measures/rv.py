"""Phase 2a: daily realised variance of SPY from 5-minute TAQ bars (Hansen & Lunde 2005 style).

Bad prints: a bucket's last trade that deviates > 1% from the bucket VWAP and is reversed by the next
trade is replaced by the VWAP (D036).

Grid (per day, from the NYSE schedule): K = 78 five-minute buckets on full days, 42 on 13:00 early closes
(buckets after the scheduled close are dropped — they hold thin post-close trades, D031).
* P_open = first qualifying trade at/after 09:30 (first trade of the first non-empty bucket).
* P_k    = last trade in bucket k (previous-tick: carried forward through an empty bucket), k = 0..K−1.
* intraday returns: r_0 = log P_0 − log P_open, r_k = log P_k − log P_{k−1};  close = P_{K−1}.
* overnight return = log P_open(t) − log close(t−1)  (previous trading day in the data).
* RV_t = overnight² + Σ_k r_k²   (daily variance, not annualised; early-close days are not padded).

Outputs: ``data/processed/rv_daily.parquet`` and the 🚦 figure ``outputs/figures/rv_check.png``
(21-day RV vs 21-day sum of squared SPX close-to-close returns: high correlation, spikes in Oct 2008 and
Mar 2020).

Run:  python -m src.measures.rv
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.check_taq import expected_buckets  # noqa: E402
from src.utils.io import load_config, project_path, write_parquet  # noqa: E402

TAQ_DIR = ("data", "raw", "taq")


BAD_PRINT_DEV = 0.01
BAD_PRINT_REVERT = 0.7


def clean_bad_prints(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace a bucket's last price by its VWAP when it is a reverting bad print (D036): the last trade is more
    than BAD_PRINT_DEV from the bucket VWAP and the next bucket's first trade reverses more than
    BAD_PRINT_REVERT of that gap. Returns the cleaned bars and the list of replaced buckets."""
    b = bars.sort_values(["date", "b"]).reset_index(drop=True)
    last, vwap = b["last_price"].astype(float), b["vwap"].astype(float)
    nxt = b.groupby("date")["first_price"].shift(-1).astype(float)
    dev = np.log(last / vwap)
    rev = np.log(nxt / last)
    bad = (dev.abs() > BAD_PRINT_DEV) & (np.sign(rev) == -np.sign(dev)) & (rev.abs() > BAD_PRINT_REVERT * dev.abs())
    fixed = b.loc[bad, ["date", "b", "last_price", "vwap"]].copy()
    b.loc[bad, "last_price"] = vwap[bad]
    return b, fixed


def daily_rv(bars: pd.DataFrame, expected: pd.Series) -> pd.DataFrame:
    """Daily RV from bucket bars (columns date, b, first_price, last_price) and buckets per day."""
    bars = bars[["date", "b", "first_price", "last_price"]].copy()
    bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
    exp = expected.reindex(pd.DatetimeIndex(bars["date"].unique())).astype(int)
    bars = bars[bars["b"].to_numpy() < exp.reindex(bars["date"]).to_numpy()]

    days = exp.index.sort_values()
    K = exp.loc[days].to_numpy()
    grid = pd.MultiIndex.from_arrays([np.repeat(days, K), np.concatenate([np.arange(k) for k in K])],
                                     names=["date", "b"])
    last = (bars.set_index(["date", "b"])["last_price"].astype(float).reindex(grid))
    n_empty = last.isna().groupby(level="date").sum()
    lp = np.log(last).groupby(level="date").ffill()

    first_bucket = bars.sort_values(["date", "b"]).groupby("date").first()
    log_open = np.log(first_bucket["first_price"].astype(float)).reindex(days)

    r = lp.groupby(level="date").diff()
    b0 = lp.groupby(level="date").transform("first")            # first non-missing grid price
    first_ret = (b0.groupby(level="date").first() - log_open)
    rv_intraday = (r ** 2).groupby(level="date").sum(min_count=1).reindex(days).fillna(0) + first_ret ** 2
    log_close = lp.groupby(level="date").last().reindex(days)

    overnight = log_open - log_close.shift(1)
    out = pd.DataFrame({
        "rv": overnight ** 2 + rv_intraday,
        "rv_intraday": rv_intraday,
        "overnight_ret": overnight,
        "open": np.exp(log_open),
        "close": np.exp(log_close),
        "n_buckets": K,
        "n_empty_buckets": n_empty.reindex(days).to_numpy(),
        "early_close": K < 78,
    }, index=pd.DatetimeIndex(days, name="date"))
    out.loc[out.index[0], "rv"] = np.nan                         # no previous close for the first day
    return out


def load_bars(cfg) -> pd.DataFrame:
    files = sorted(project_path(*TAQ_DIR).glob("spy_5min_[0-9][0-9][0-9][0-9].parquet"))
    if not files:
        raise FileNotFoundError("no TAQ bars found — run Phase 1c (python -m src.data.pull_taq)")
    return pd.concat([pd.read_parquet(f, columns=["date", "b", "first_price", "last_price", "vwap"]) for f in files],
                     ignore_index=True)


def gate_figure(rv: pd.DataFrame, cfg, path) -> float:
    h = cfg["vrp_tenor"]["rv_trading_days"]
    spx = pd.read_parquet(project_path("data", "raw", "optionmetrics", "spx_index_daily.parquet")).set_index("date")
    r2 = np.log(spx["close"].astype(float)).diff() ** 2
    both = pd.concat({"rv": rv["rv"].rolling(h).sum(), "sq": r2.rolling(h).sum()}, axis=1).dropna()
    corr = both.corr().iloc[0, 1]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(both.index, 100 * np.sqrt(both["rv"] * 252 / h), label="5-min RV + overnight (SPY, TAQ)", lw=0.8)
    ax.plot(both.index, 100 * np.sqrt(both["sq"] * 252 / h), label="squared daily SPX returns", lw=0.8, alpha=0.7)
    ax.set_yscale("log"); ax.set_ylabel("21-day realised vol, annualised % (log scale)")
    ax.set_title(f"21-day realised variance: TAQ RV vs daily squared returns — corr {corr:.3f}")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return corr


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    bars = load_bars(cfg)
    start = pd.Timestamp(cfg["sample"]["har_burnin_start"])
    end = pd.Timestamp(cfg["sample"]["end"])
    expected = expected_buckets(start, end)
    bars, fixed = clean_bad_prints(bars)
    fixed.to_csv(project_path("outputs", "tables", "rv_bad_prints_replaced.csv"), index=False)
    rv = daily_rv(bars, expected)

    missing = expected.index.difference(rv.index)
    write_parquet(rv, "data/processed/rv_daily.parquet")
    corr = gate_figure(rv, cfg, project_path("outputs", "figures", "rv_check.png"))
    ann = 100 * np.sqrt(252 * rv["rv"].rolling(cfg["vrp_tenor"]["rv_trading_days"]).mean())
    print(f"[ok  ] rv_daily.parquet: {rv['rv'].notna().sum()} days {rv.index.min().date()} → {rv.index.max().date()}; "
          f"missing trading days: {len(missing)}; early-close days: {int(rv['early_close'].sum())}; "
          f"days with empty buckets: {int((rv['n_empty_buckets'] > 0).sum())}")
    print(f"       bad prints replaced by bucket VWAP: {len(fixed)} ({[str(d.date()) for d in fixed['date']]})")
    print(f"       overnight share of total RV: {(rv['overnight_ret'] ** 2).sum() / rv['rv'].sum():.1%}")
    for label, lo, hi in [("Oct 2008", "2008-10-01", "2008-11-30"), ("Mar 2020", "2020-03-01", "2020-04-15")]:
        pk = ann.loc[lo:hi]
        print(f"       peak 21-day RV vol {label}: {pk.max():.0f}% on {pk.idxmax().date()} "
              f"(full-sample rank {int((ann > pk.max()).sum()) + 1})")
    print(f"🚦 corr(21-day TAQ RV, 21-day squared SPX returns) = {corr:.3f} → {'PASS' if corr > 0.9 else 'CHECK'}")
    return rv, corr


if __name__ == "__main__":
    main()
