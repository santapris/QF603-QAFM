"""Phase 2b: 30-day model-free implied variance (MFIV) from SPX options, Cboe VIX methodology.

For each month-end t (valuation time 16:00 ET):
1. Expiries are (exdate, am_settlement) pairs. Time to expiry in minutes: AM-settled → 09:30 ET on the
   settlement date, PM-settled → 16:00 ET. OptionMetrics dates pre-2015 standard monthlies to the Saturday
   after the third Friday; those settle Friday morning, so a Saturday exdate is moved to Friday (D028).
2. Candidates need ≥ ``MIN_DAYS`` calendar days and a valid term variance. Near term = longest expiry
   with T ≤ 30 days, next term = shortest with T > 30 days; if no near term exists, the two shortest
   terms above 30 days are used (extrapolation, flagged).
3. Per term: midquotes (crossed quotes and zero offers dropped, D020); forward F = K* + e^{RT}(C − P) at
   the strike minimising |C − P|; K0 = largest strike ≤ F with both quotes; OTM puts below K0 and OTM
   calls above K0, skipping zero-bid strikes and stopping after two consecutive zero bids; average of
   put and call at K0;  σ² = (2/T)·Σ ΔK/K²·e^{RT}·Q(K) − (1/T)·(F/K0 − 1)².
4. σ²_30 = [T1σ1²(N2 − N30)/(N2 − N1) + T2σ2²(N30 − N1)/(N2 − N1)] · N365/N30 (minutes N).
Risk-free rate: OptionMetrics zero curve on t, linear in days, percent → decimal (D020).
If a month-end fails, the nearest prior buffer day is used and logged.

Outputs: ``data/processed/iv_monthly.parquet`` (``iv`` = annualised 30-day variance, decimal) and the
gate figure ``outputs/figures/mfiv_vs_vix2.png`` (🚦 corr with VIX²/10000 must exceed 0.99).

Run:  python -m src.measures.mfiv
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.dates import month_end_trading_days, to_month_end  # noqa: E402
from src.utils.io import load_config, project_path, write_parquet  # noqa: E402

MIN_PER_YEAR = 525_600
MIN_DAYS = 7
MIN_STRIKES = 5
OPEN_MIN, CLOSE_MIN = 9 * 60 + 30, 16 * 60
RAW = ("data", "raw", "optionmetrics")
COLS = ["date", "exdate", "am_settlement", "cp_flag", "strike_price", "best_bid", "best_offer"]


# ------------------------------------------------------------------ building blocks
def minutes_to_expiry(date: pd.Timestamp, exdate: pd.Series, am: pd.Series) -> pd.Series:
    """Minutes from 16:00 ET on ``date`` to settlement (09:30 ET for AM, 16:00 ET for PM)."""
    settle = exdate - pd.to_timedelta((exdate.dt.weekday == 5).astype(int), unit="D")   # Saturday → Friday
    days = (settle - date).dt.days
    return days * 1440 + np.where(am == 1, OPEN_MIN, CLOSE_MIN) - CLOSE_MIN


def zero_rate(zero_day: pd.DataFrame, days: float) -> float:
    """Continuously compounded rate (decimal) for ``days``, linear interpolation, flat beyond the ends."""
    z = zero_day.sort_values("days")
    return float(np.interp(days, z["days"], z["rate"])) / 100


def clean_quotes(q: pd.DataFrame) -> pd.DataFrame:
    q = q[(q["best_offer"] > 0) & (q["best_bid"] >= 0) & (q["best_bid"] <= q["best_offer"])].copy()
    q["strike"] = q["strike_price"] / 1000
    q["mid"] = (q["best_bid"] + q["best_offer"]) / 2
    return q


def otm_tail(side: pd.DataFrame, strikes) -> list[tuple[float, float]]:
    """Walk strikes away from K0: skip zero bids, stop after two consecutive zero bids."""
    out, zeros = [], 0
    for k in strikes:
        if side.loc[k, "best_bid"] <= 0:
            zeros += 1
            if zeros == 2:
                break
            continue
        zeros = 0
        out.append((k, side.loc[k, "mid"]))
    return out


def term_variance(q: pd.DataFrame, T: float, R: float) -> dict | None:
    """Cboe single-term variance from one expiry's quotes (columns cp_flag, strike, best_bid, mid)."""
    calls = q[q["cp_flag"] == "C"].set_index("strike").sort_index()
    puts = q[q["cp_flag"] == "P"].set_index("strike").sort_index()
    both = calls.index.intersection(puts.index)
    both = both[(calls.loc[both, "best_bid"] > 0).to_numpy() & (puts.loc[both, "best_bid"] > 0).to_numpy()]
    if len(both) < 2:
        return None
    diff = (calls.loc[both, "mid"] - puts.loc[both, "mid"])
    k_star = diff.abs().idxmin()
    growth = np.exp(R * T)
    F = k_star + growth * diff.loc[k_star]
    below = both[both <= F]
    if len(below) == 0:
        return None
    K0 = below.max()

    put_side = otm_tail(puts, puts.index[puts.index < K0][::-1])
    call_side = otm_tail(calls, calls.index[calls.index > K0])
    if len(put_side) < MIN_STRIKES or len(call_side) < MIN_STRIKES:
        return None
    k0_q = (calls.loc[K0, "mid"] + puts.loc[K0, "mid"]) / 2
    pts = sorted(put_side + [(K0, k0_q)] + call_side)
    K = np.array([p[0] for p in pts])
    Q = np.array([p[1] for p in pts])
    dK = np.empty_like(K)
    dK[1:-1] = (K[2:] - K[:-2]) / 2
    dK[0], dK[-1] = K[1] - K[0], K[-1] - K[-2]
    sigma2 = (2 / T) * np.sum(dK / K ** 2 * growth * Q) - (1 / T) * (F / K0 - 1) ** 2
    return dict(sigma2=sigma2, F=F, K0=K0, n_strikes=len(K), k_min=K[0], k_max=K[-1])


def interpolate_30d(T1, s1, T2, s2, target_days: int) -> float:
    """Cboe interpolation of total variance to ``target_days``, annualised (T in years)."""
    N1, N2, N30, N365 = T1 * MIN_PER_YEAR, T2 * MIN_PER_YEAR, target_days * 1440, 365 * 1440
    return (T1 * s1 * (N2 - N30) / (N2 - N1) + T2 * s2 * (N30 - N1) / (N2 - N1)) * N365 / N30


def mfiv_on_date(day: pd.DataFrame, zero_day: pd.DataFrame, target_days: int = 30) -> dict | None:
    date = day["date"].iloc[0]
    q = clean_quotes(day)
    q["minutes"] = minutes_to_expiry(date, q["exdate"], q["am_settlement"])
    q = q[q["minutes"] >= MIN_DAYS * 1440]
    terms = q.groupby(["minutes", "exdate", "am_settlement"])
    keys = sorted(terms.groups)                          # ascending time to expiry

    def evaluate(key):
        g = terms.get_group(key)
        T = key[0] / MIN_PER_YEAR
        R = zero_rate(zero_day, key[0] / 1440)
        res = term_variance(g, T, R)
        return None if res is None else dict(res, T=T, R=R, exdate=key[1], am=int(key[2]))

    cut = target_days * 1440
    near = next((r for k in [k for k in keys if k[0] <= cut][::-1] if (r := evaluate(k))), None)
    nxt = [r for k in [k for k in keys if k[0] > cut] if (r := evaluate(k))][:2]
    if near is not None and nxt:
        t1, t2, method = near, nxt[0], "interpolate"
    elif near is None and len(nxt) == 2:
        t1, t2, method = nxt[0], nxt[1], "extrapolate"
    else:
        return None
    iv = interpolate_30d(t1["T"], t1["sigma2"], t2["T"], t2["sigma2"], target_days)
    if not np.isfinite(iv) or iv <= 0:
        return None
    return dict(iv=iv, method=method, T1_days=t1["T"] * 365, T2_days=t2["T"] * 365,
                sigma2_1=t1["sigma2"], sigma2_2=t2["sigma2"], F1=t1["F"], F2=t2["F"], K0_1=t1["K0"],
                K0_2=t2["K0"], n_k1=t1["n_strikes"], n_k2=t2["n_strikes"], r1=t1["R"], r2=t2["R"],
                exdate1=t1["exdate"], exdate2=t2["exdate"], am1=t1["am"], am2=t2["am"])


# ------------------------------------------------------------------ pipeline
def build(cfg) -> pd.DataFrame:
    start, end = pd.Timestamp(cfg["sample"]["start"]), pd.Timestamp(cfg["sample"]["end"])
    target_days = cfg["vrp_tenor"]["iv_calendar_days"]
    month_ends = month_end_trading_days(start - pd.DateOffset(months=1), end)
    zero = pd.read_parquet(project_path(*RAW, "zerocd_request_dates.parquet"))
    zero_by_date = dict(tuple(zero.groupby("date")))

    rows = []
    for path in sorted(project_path(*RAW).glob("spx_options_*.parquet")):
        df = pd.read_parquet(path, columns=COLS)
        by_date = dict(tuple(df.groupby("date")))
        for me in month_ends[month_ends.year == int(path.stem[-4:])]:
            prior = sorted(d for d in by_date if d <= me)[::-1][:4]          # month-end, then buffer days
            for used in prior:
                res = mfiv_on_date(by_date[used], zero_by_date[used], target_days)
                if res is not None:
                    rows.append(dict(date=me, date_used=used, fallback=used != me, **res))
                    break
            else:
                rows.append(dict(date=me, iv=np.nan, method="failed"))
        print(f"  {path.stem}: done", flush=True)
    return pd.DataFrame(rows).set_index("date").sort_index()


def gate(iv: pd.DataFrame, cfg, path) -> tuple[float, pd.Series]:
    from src.data.pull_fred import load_raw
    vix = to_month_end(load_raw("VIXCLS"), iv.index, max_stale_days=5)
    vix2 = vix ** 2 / 10000
    both = pd.concat({"mfiv": iv["iv"], "vix2": vix2}, axis=1).dropna()
    corr = both.corr().iloc[0, 1]
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(both.index, both["mfiv"], label="MFIV (OptionMetrics SPX, 30d)")
    ax[0].plot(both.index, both["vix2"], label="VIX²/10000", alpha=0.7)
    ax[0].set_ylabel("annualised variance"); ax[0].legend()
    ax[0].set_title(f"MFIV vs VIX² at month-ends — corr {corr:.4f}")
    ax[1].plot(both.index, both["mfiv"] - both["vix2"], color="C3")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylabel("MFIV − VIX²/10000")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return corr, both["mfiv"] - both["vix2"]


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    iv = build(cfg)
    write_parquet(iv, "data/processed/iv_monthly.parquet")
    fig_path = project_path("outputs", "figures", "mfiv_vs_vix2.png")
    corr, diff = gate(iv, cfg, fig_path)
    failed = iv.index[iv["iv"].isna()]
    print(f"[ok  ] iv_monthly.parquet: {iv['iv'].notna().sum()}/{len(iv)} month-ends; "
          f"fallback days used: {int(iv['fallback'].fillna(False).sum())}; failed: {[str(d.date()) for d in failed]}")
    print(f"       methods: {iv['method'].value_counts().to_dict()}; "
          f"T1 {iv['T1_days'].min():.1f}–{iv['T1_days'].max():.1f}d, T2 {iv['T2_days'].min():.1f}–{iv['T2_days'].max():.1f}d")
    print(f"       MFIV − VIX²: mean {diff.mean():.5f}, sd {diff.std():.5f}, max |.| {diff.abs().max():.5f} "
          f"at {diff.abs().idxmax().date()}")
    print(f"🚦 MFIV vs VIX²/10000 correlation = {corr:.4f} → {'PASS' if corr > 0.99 else 'FAIL — stop and debug'}")
    return iv, corr


if __name__ == "__main__":
    main()
