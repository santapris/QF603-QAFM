"""VRP measures at month-ends (PLAN 2d). All variances annualised, in decimals.

* ``vrp_exante``  = IV_t − E_t[RV_{t→t+h}]                           (headline; known at t)
* ``vrp_expost``  = IV_t − RV_{t→t+h}                                 (robustness target / strategy payoff;
                    known only at ``obs_date_vrp_expost`` = trading day t+h, passed to the holdout guard, D006)
* ``vrp_trail``   = IV_t − RV_{t−h+1→t}                               (Q5 only, BTZ convention)
* ``rv_surprise`` = RV over (t−1, t] annualised with its own day count − E_{t−1}[RV]   (D026)

``iv`` must be implied variance at month-end t; ``har_fcst`` the HAR forecast made at t; ``rv`` daily
realised variance (not annualised) on the trading-day index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.measures.har import ANNUAL


def vrp_measures(iv: pd.Series, har_fcst: pd.Series, rv: pd.Series, horizon: int = 21) -> pd.DataFrame:
    rv = rv.sort_index().astype(float)
    idx = pd.DatetimeIndex(iv.index, name="date")
    pos = rv.index.get_indexer(idx)
    if (pos < 0).any():
        raise ValueError(f"month-ends missing from the RV index: {list(idx[pos < 0].date)}")
    csum = np.concatenate([[0.0], rv.cumsum().to_numpy()])      # csum[i] = sum of rv[:i]
    n = len(rv)

    fwd_end = pos + horizon
    ok_fwd = fwd_end < n
    rv_fwd = np.where(ok_fwd, (csum[np.minimum(fwd_end, n - 1) + 1] - csum[pos + 1]) * ANNUAL / horizon, np.nan)
    obs_date = pd.DatetimeIndex(np.where(ok_fwd, rv.index.values[np.minimum(fwd_end, n - 1)],
                                         np.datetime64("NaT", "ns")))

    ok_past = pos - horizon + 1 >= 0
    rv_past = np.where(ok_past, (csum[pos + 1] - csum[np.maximum(pos - horizon + 1, 0)]) * ANNUAL / horizon, np.nan)

    prev = np.concatenate([[-1], pos[:-1]])
    ok_m = prev >= 0
    days_m = pos - prev
    rv_month = np.where(ok_m, (csum[pos + 1] - csum[np.maximum(prev, 0) + 1]) * ANNUAL / np.maximum(days_m, 1), np.nan)

    out = pd.DataFrame({
        "iv": iv.to_numpy(),
        "har_fcst": har_fcst.reindex(idx).to_numpy(),
        "rv_fwd": rv_fwd,
        "obs_date_vrp_expost": obs_date,
        "rv_past": rv_past,
        "rv_month": rv_month,
    }, index=idx)
    out["vrp_exante"] = out["iv"] - out["har_fcst"]
    out["vrp_expost"] = out["iv"] - out["rv_fwd"]
    out["vrp_trail"] = out["iv"] - out["rv_past"]
    out["rv_surprise"] = out["rv_month"] - out["har_fcst"].shift(1)
    return out


# ====================================================================== monthly panel (PLAN 2d)
# Columns whose value depends on data after t (targets / payoffs); each has a known observation date.
FORWARD_COLUMNS = ["rv_fwd", "vrp_expost", "obs_date_vrp_expost", "exret_1", "exret_3", "exret_6",
                   "put_ret_fwd1", "bxm_ret_fwd1"]

DICTIONARY = {
    "iv": ("30-day model-free implied variance (MFIV, OptionMetrics SPX), annualised", "variance", "t", "2b"),
    "har_fcst": ("Headline HAR forecast of RV over the next 21 trading days", "variance", "t", "2c"),
    "vrp_exante": ("Headline ex-ante VRP = iv − har_fcst", "variance", "t", "2d"),
    "vrp_exante_level": ("Ex-ante VRP with the level HAR", "variance", "t", "2d"),
    "vrp_exante_log": ("Ex-ante VRP with the log HAR", "variance", "t", "2d"),
    "vrp_exante_iv": ("Ex-ante VRP with the IV-augmented HAR", "variance", "t", "2d"),
    "rv_fwd": ("Realised variance over trading days t+1..t+21 (TAQ 5-min + overnight), annualised", "variance",
               "obs_date_vrp_expost", "2a"),
    "vrp_expost": ("Ex-post VRP = iv − rv_fwd (short variance-swap payoff); target only, never a predictor",
                   "variance", "obs_date_vrp_expost", "2d"),
    "obs_date_vrp_expost": ("Trading day t+21, when rv_fwd / vrp_expost become known (D006)", "date", "—", "2d"),
    "rv_past": ("Realised variance over trading days t−20..t, annualised", "variance", "t", "2a"),
    "vrp_trail": ("Trailing VRP = VIX²/10000 − rv_past (BTZ convention; Q5 only)", "variance", "t", "2d"),
    "rv_month": ("Realised variance over the trading days in (t−1, t], annualised", "variance", "t", "2d"),
    "rv_surprise": ("rv_month − HAR forecast made at t−1 (D026)", "variance", "t", "2d"),
    "vix": ("VIX close", "index points", "t", "1d"),
    "vix_ts": ("VIX3M / VIX (term structure; > 1 = contango)", "ratio", "t", "1d"),
    "vvix": ("VVIX close", "index points", "t", "1d"),
    "skew": ("Cboe SKEW close", "index points", "t", "1d"),
    "fund_spread": ("3M AA financial CP − 3M T-bill (1-day publication lag; NaN if > 14 days stale)", "decimal", "t", "1e"),
    "fund_spread_nonfin": ("3M AA non-financial CP − 3M T-bill (robustness)", "decimal", "t", "1e"),
    "credit_spread": ("Moody's BAA − AAA (1-day publication lag)", "decimal", "t", "1e"),
    "hkm": ("HKM intermediary capital ratio, 3-month publication lag", "ratio", "t", "1g"),
    "hkm_lag1": ("HKM capital ratio, 1-month lag (robustness)", "ratio", "t", "1g"),
    "cftc_pos": ("Dealer net VIX-futures position / open interest, last report released ≤ t (D024, D025)", "ratio", "t", "1f"),
    "cftc_pos_lev": ("Leveraged-money net VIX-futures position / open interest (alternative)", "ratio", "t", "1f"),
    "rf": ("T-bill holding-period return t → t+1, set at t (D023)", "decimal", "t", "1e"),
    "exret_1": ("Log SPX total-return excess return t → t+1", "log return", "t+1 month-end", "2d"),
    "exret_3": ("Cumulative log excess return t → t+3", "log return", "t+3 month-end", "2d"),
    "exret_6": ("Cumulative log excess return t → t+6", "log return", "t+6 month-end", "2d"),
    "put_ret_fwd1": ("Log return of Cboe PUT index t → t+1 (strategy benchmark)", "log return", "t+1 month-end", "2d"),
    "bxm_ret_fwd1": ("Log return of Cboe BXM index t → t+1 (strategy benchmark)", "log return", "t+1 month-end", "2d"),
}


def assemble(cfg, iv: pd.Series, har: pd.DataFrame, rv: pd.Series, vix_daily: pd.Series,
             bbg: pd.DataFrame, fred: pd.DataFrame, cftc: pd.DataFrame, hkm: pd.DataFrame | None) -> pd.DataFrame:
    """Pure panel assembly from already-loaded inputs (every input dated by when it is known)."""
    from src.utils.dates import to_month_end

    h = cfg["vrp_tenor"]["rv_trading_days"]
    start, end = pd.Timestamp(cfg["sample"]["start"]), pd.Timestamp(cfg["sample"]["end"])
    stale = cfg["max_stale_days"].get("bloomberg", 5)
    idx = iv.loc[:end].index                                     # month-ends incl. one pre-sample month
    headline = cfg["har"]["headline"]

    base_fcst = har[f"har_{headline}"] if headline else har["har_level"]
    p = vrp_measures(iv.loc[idx], base_fcst.reindex(idx), rv, horizon=h)
    if not headline:
        p[["har_fcst", "vrp_exante", "rv_surprise"]] = np.nan
    for v in cfg["har"]["variants"]:
        p[f"vrp_exante_{v}"] = p["iv"] - har[f"har_{v}"].reindex(idx)

    vix = to_month_end(vix_daily, idx, max_stale_days=stale)
    p["vix"] = vix
    p["vrp_trail"] = vix ** 2 / 10000 - p["rv_past"]
    me = {c: to_month_end(bbg[c], idx, max_stale_days=stale) if c in bbg else pd.Series(np.nan, index=idx)
          for c in ["vix3m", "vvix", "skew", "spxt", "put", "bxm"]}
    p["vix_ts"] = me["vix3m"] / vix
    p["vvix"], p["skew"] = me["vvix"], me["skew"]
    p = p.join(fred[["fund_spread", "fund_spread_nonfin", "credit_spread", "rf"]])
    p = p.join(cftc[["cftc_pos", "cftc_pos_lev"]])
    if hkm is not None:
        p = p.join(hkm[["hkm", "hkm_lag1"]])
    else:
        p[["hkm", "hkm_lag1"]] = np.nan

    log_rf = np.log1p(p["rf"])
    log_px = np.log(me["spxt"])
    for k in cfg["horizons"]:
        p[f"exret_{k}"] = (log_px.shift(-k) - log_px) - log_rf[::-1].rolling(k).sum()[::-1]
    p["put_ret_fwd1"] = np.log(me["put"]).shift(-1) - np.log(me["put"])
    p["bxm_ret_fwd1"] = np.log(me["bxm"]).shift(-1) - np.log(me["bxm"])
    cols = [c for c in DICTIONARY if c in p.columns]
    return p.loc[start:end, cols]


def dictionary_markdown(panel: pd.DataFrame) -> str:
    lines = ["# panel_monthly — data dictionary", "",
             "Index: `date` = last NYSE trading day of each month. Variances annualised, decimals (0.04 = 20% vol).",
             "\"Known at\" = when the value is observable; columns known after t are targets only.", "",
             "| Column | Description | Units | Known at | Built in | Non-missing |", "| --- | --- | --- | --- | --- | --- |"]
    for c in panel.columns:
        d, u, k, ph = DICTIONARY[c]
        lines.append(f"| `{c}` | {d} | {u} | {k} | Phase {ph} | {int(panel[c].notna().sum())}/{len(panel)} |")
    return "\n".join(lines) + "\n"


def load_inputs(cfg, draft: bool) -> tuple[dict, list[str]]:
    """Load every processed input; in draft mode missing manual sources become empty (and are listed)."""
    from src.data import load_bloomberg, load_hkm
    from src.measures.har import load_iv_daily
    from src.utils.io import project_path

    proc = lambda name: pd.read_parquet(project_path("data", "processed", name))  # noqa: E731
    missing = []
    bbg, miss_bbg = load_bloomberg.load(cfg, require_all=not draft)
    missing += miss_bbg
    try:
        hkm = load_hkm.build(cfg)
    except FileNotFoundError as e:
        if not draft:
            raise
        hkm, missing = None, missing + [f"HKM ({e})"]
    if cfg["har"]["headline"] is None:
        if not draft:
            raise ValueError("config har.headline is not set — 🧑 choose the headline HAR variant (PLAN 2c)")
        missing.append("har.headline not chosen → vrp_exante / rv_surprise left empty")
    vix_daily = bbg["vix"] if "vix" in bbg else load_iv_daily(cfg).pow(0.5).mul(100)
    if "vix" not in bbg:
        missing.append("VIX from FRED VIXCLS (Bloomberg VIX not exported yet)")
    return dict(iv=proc("iv_monthly.parquet")["iv"], har=proc("har_forecasts.parquet"),
                rv=proc("rv_daily.parquet")["rv"], vix_daily=vix_daily, bbg=bbg,
                fred=proc("fred_monthly.parquet"), cftc=proc("cftc_monthly.parquet"), hkm=hkm), missing


def sanity(panel: pd.DataFrame, cfg) -> list[tuple[str, bool, str]]:
    """🚦 PLAN 2d checks, on data known by oos.validate_end only."""
    from src.utils.holdout import restrict
    pre = panel.loc[:cfg["oos"]["validate_end"]]
    post = restrict(panel.dropna(subset=["obs_date_vrp_expost"]), "vrp_expost", 0, cfg=cfg,
                    obs_dates=panel["obs_date_vrp_expost"].dropna())
    out = []
    for col in [c for c in ["vrp_exante", "vrp_exante_level", "vrp_exante_log", "vrp_exante_iv"] if pre[c].notna().any()]:
        share = (pre[col] > 0).mean()
        top = pre[col].nlargest(3).index
        out.append((f"{col} mostly positive", share > 0.6, f"{share:.0%} > 0; largest in {[d.strftime('%Y-%m') for d in top]}"))
    for origin, label in [("2008-09-30", "Oct 2008"), ("2020-02-28", "Mar 2020")]:
        v = post["vrp_expost"].get(pd.Timestamp(origin), np.nan)
        rank = post["vrp_expost"].rank().get(pd.Timestamp(origin), np.nan)
        out.append((f"ex-post payoff sharply negative ({label})", bool(v < 0 and rank <= 5),
                    f"{v:.3f}, rank {rank:.0f} from worst of {post['vrp_expost'].notna().sum()}"))
    return out


def summary_stats(panel: pd.DataFrame, cfg) -> pd.DataFrame:
    pre = panel.loc[:cfg["oos"]["validate_end"]]
    num = pre.select_dtypes("number")
    s = num.describe(percentiles=[0.05, 0.5, 0.95]).T
    s["ar1"] = [num[c].autocorr(1) for c in num.columns]
    s["skew"], s["kurt"] = num.skew(), num.kurt()
    return s


def figures(panel: pd.DataFrame, har: pd.DataFrame, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.io import project_path
    pre = panel.loc[:cfg["oos"]["validate_end"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    ex = "vrp_exante" if pre["vrp_exante"].notna().any() else "vrp_exante_log"
    ax.plot(pre.index, pre[ex], label=f"ex-ante ({ex})")
    ok = pre["obs_date_vrp_expost"] <= pd.Timestamp(cfg["oos"]["validate_end"])
    ax.plot(pre.index[ok], pre.loc[ok, "vrp_expost"], label="ex-post (payoff)", alpha=0.6)
    ax.plot(pre.index, pre["vrp_trail"], label="trailing (VIX² − past RV)", alpha=0.6)
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("annualised variance"); ax.legend()
    ax.set_title("VRP definitions (pre-holdout)")
    fig.tight_layout(); fig.savefig(project_path("outputs", "figures", "vrp_definitions.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    h = har.loc[:cfg["oos"]["validate_end"]]
    ok = h["obs_date_rv_fwd"] <= pd.Timestamp(cfg["oos"]["validate_end"])
    ax.plot(h.index[ok], 100 * np.sqrt(h.loc[ok, "rv_fwd"]), label="realised next 21 days", color="k", lw=1)
    for v in cfg["har"]["variants"]:
        ax.plot(h.index, 100 * np.sqrt(h[f"har_{v}"].clip(lower=0)), label=f"HAR {v}", alpha=0.8, lw=0.9)
    ax.set_yscale("log"); ax.set_ylabel("annualised vol % (log)"); ax.legend()
    ax.set_title("HAR forecasts vs realised (pre-holdout)")
    fig.tight_layout(); fig.savefig(project_path("outputs", "figures", "har_vs_realised.png"), dpi=120); plt.close(fig)


def main(cfg=None, force=False, draft: bool = False):
    from src.utils.io import load_config, project_path, write_parquet
    cfg = cfg or load_config()
    inputs, missing = load_inputs(cfg, draft)
    panel = assemble(cfg, **inputs)
    name = "panel_monthly_draft" if draft else "panel_monthly"
    write_parquet(panel, f"data/processed/{name}.parquet")
    (project_path("data", "processed", f"{name.replace('panel_monthly', 'panel_dictionary')}.md")
     .write_text(dictionary_markdown(panel)))
    summary_stats(panel, cfg).to_csv(project_path("outputs", "tables", f"summary_stats{'_draft' if draft else ''}.csv"))
    figures(panel, inputs["har"], cfg)
    checks = sanity(panel, cfg)
    print(f"[ok  ] {name}.parquet: {len(panel)} month-ends × {panel.shape[1]} columns")
    if missing:
        print("⚠️  DRAFT — missing inputs:\n   - " + "\n   - ".join(missing))
    print("🚦 Panel sanity (pre-holdout):")
    for n, ok, d in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}")
    return panel, checks


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true", help="build with missing manual inputs as NaN (not for modelling)")
    main(draft=ap.parse_args().draft)
