"""Phase 1.5 — thin end-to-end prototype with quick stand-ins (results are THROWAWAY).

Stand-ins (logged, D026):
* IV  = FRED VIXCLS² / 10000 at month-end            → swap-in point: Phase 2b MFIV (``iv_monthly.parquet``)
* RV  = squared daily SPX close-to-close log returns  → swap-in point: Phase 2a TAQ RV (``rv_daily.parquet``)

Everything downstream is the real code: recursive HAR (``src.measures.har``), VRP measures
(``src.measures.vrp``), Stage 1 regressions (``src.models.stage1``), holdout guard. Predictors available
now: VIX level (VIXCLS stand-in), funding and credit spreads (FRED), CFTC positioning.

Nothing here looks past ``oos.validate_end`` (targets included): the toy strategy and all tables stop at
data known by then.

Outputs go to ``outputs/prototype/`` only (delete once Phase 2 is complete; keep this code).

Run:  python -m src.models.prototype
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.pull_fred import load_raw  # noqa: E402
from src.measures.har import recursive_har  # noqa: E402
from src.measures.vrp import vrp_measures  # noqa: E402
from src.models.stage1 import persistence, single_predictors  # noqa: E402
from src.utils.dates import month_end_trading_days, to_month_end  # noqa: E402
from src.utils.holdout import restrict  # noqa: E402
from src.utils.io import load_config, project_path  # noqa: E402

OUT = ("outputs", "prototype")


def rv_standin(cfg) -> pd.Series:
    """Daily 'realised variance' = squared close-to-close log return of the SPX index."""
    spx = pd.read_parquet(project_path("data", "raw", "optionmetrics", "spx_index_daily.parquet")).set_index("date")
    r = np.log(spx["close"].astype(float)).diff()
    return (r ** 2).dropna().rename("rv")


def build_panel(cfg) -> pd.DataFrame:
    h = cfg["vrp_tenor"]["rv_trading_days"]
    start, end = pd.Timestamp(cfg["sample"]["start"]), pd.Timestamp(cfg["sample"]["end"])
    rv = rv_standin(cfg)
    me = month_end_trading_days(start - pd.DateOffset(months=1), end)     # one extra month for rv_surprise
    vix = to_month_end(load_raw("VIXCLS"), me, max_stale_days=5)
    har = recursive_har(rv, me, horizon=h)
    panel = vrp_measures(vix ** 2 / 10000, har["har_fcst"], rv, horizon=h)
    panel["vix"] = vix
    panel["har_n_train"] = har["n_train"]

    fred = pd.read_parquet(project_path("data", "processed", "fred_monthly.parquet"))
    cftc = pd.read_parquet(project_path("data", "processed", "cftc_monthly.parquet"))
    panel = panel.join(fred[["fund_spread", "credit_spread", "rf"]]).join(cftc[["cftc_pos"]])
    return panel.loc[start:end]


def toy_strategy(panel: pd.DataFrame, cfg) -> pd.DataFrame:
    """Paper short one-month variance swap, 1 unit of variance notional: P&L_t = vrp_expost_t (known at t+21d)."""
    df = panel[["vrp_expost", "obs_date_vrp_expost"]].dropna()
    df = restrict(df, "vrp_expost", 0, cfg=cfg, obs_dates=df["obs_date_vrp_expost"])
    df = df.assign(pnl=df["vrp_expost"], pnl_volpts=100 * (np.sqrt(panel.loc[df.index, "iv"])
                                                           - np.sqrt(panel.loc[df.index, "rv_fwd"])))
    return df


def gate(panel: pd.DataFrame, toy: pd.DataFrame, cfg) -> list[tuple[str, bool, str]]:
    pre = panel.loc[:cfg["oos"]["validate_end"]]
    checks = []
    share_pos = (pre["vrp_exante"] > 0).mean()
    checks.append(("ex-ante VRP mostly positive", share_pos > 0.6, f"{share_pos:.0%} of months > 0"))
    for origin, label in [("2008-09-30", "Oct 2008"), ("2020-02-28", "Mar 2020")]:
        v = toy.loc[origin, "pnl"] if pd.Timestamp(origin) in toy.index else np.nan
        worst = toy["pnl"].rank().loc[origin] if pd.Timestamp(origin) in toy.index else np.nan
        checks.append((f"short-variance payoff sharply negative in {label}", bool(v < 0 and worst <= 5),
                       f"payoff {v:.3f} (rank {worst:.0f} of {len(toy)} from worst)"))
    spike = pre["vrp_exante"].abs().idxmax()
    checks.append(("largest |ex-ante VRP| in a stress period", spike.year in (2008, 2009, 2020, 2011),
                   f"largest at {spike.date()} ({pre['vrp_exante'].loc[spike]:.3f})"))
    checks.append(("toy short-variance P&L positive on average", toy["pnl"].mean() > 0,
                   f"mean {toy['pnl'].mean():.4f} var units, {toy['pnl_volpts'].mean():.2f} vol pts; "
                   f"hit rate {(toy['pnl'] > 0).mean():.0%}"))
    neg = int((panel["har_fcst"] <= 0).sum())
    checks.append(("HAR forecasts positive", neg == 0, f"{neg} non-positive forecasts"))
    return checks


def plot(panel: pd.DataFrame, toy: pd.DataFrame, cfg, path):
    pre = panel.loc[:cfg["oos"]["validate_end"]]
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(pre.index, np.sqrt(pre["iv"]), label="√IV (VIX stand-in)")
    ax[0].plot(pre.index, np.sqrt(pre["har_fcst"].clip(lower=0)), label="√E_t[RV] (HAR, stand-in RV)")
    ax[0].plot(pre.index, np.sqrt(pre["rv_fwd"]), label="√RV realised next 21d", alpha=0.6)
    ax[0].set_ylabel("annualised vol"); ax[0].legend(fontsize=8)
    ax[1].plot(pre.index, pre["vrp_exante"], label="ex-ante VRP")
    ax[1].plot(toy.index, toy["vrp_expost"], label="ex-post VRP (payoff)", alpha=0.6)
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylabel("variance"); ax[1].legend(fontsize=8)
    ax[2].plot(toy.index, toy["pnl"].cumsum(), label="cumulative short-variance payoff (1 var notional)")
    ax[2].set_ylabel("cum. variance units"); ax[2].legend(fontsize=8)
    fig.suptitle("PROTOTYPE (stand-in data, throwaway) — pre-holdout only")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    out = project_path(*OUT)
    out.mkdir(parents=True, exist_ok=True)

    panel = build_panel(cfg)
    toy = toy_strategy(panel, cfg)
    horizons = cfg["horizons"]
    pers = persistence(panel, "vrp_exante", horizons, cfg=cfg)
    single = single_predictors(panel, "vrp_exante", ["vix", "fund_spread", "credit_spread", "cftc_pos"],
                               horizons, cfg=cfg)

    restrict(panel, cfg=cfg).to_parquet(out / "panel_prototype.parquet")
    pers.to_csv(out / "proto_stage1_persistence.csv")
    single.to_csv(out / "proto_stage1_single.csv", index=False)
    plot(panel, toy, cfg, out / "proto_vrp.png")

    checks = gate(panel, toy, cfg)
    pd.DataFrame(checks, columns=["check", "pass", "detail"]).to_csv(out / "proto_gate.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print("Persistence (ex-ante VRP):\n", pers.to_string(), "\n")
        print("Single predictors:\n", single[["predictor", "k", "beta", "t_nw", "p_holm", "incr_r2", "nobs"]].to_string(index=False), "\n")
    print("🚦 Prototype gate:")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return panel, pers, single, checks


if __name__ == "__main__":
    main()
