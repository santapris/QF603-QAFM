"""Phase 7 report builder: ``report/report.md`` regenerated from code outputs only (PLAN: every figure and table in
the report comes from ``outputs/`` built by the pipeline). Sections whose inputs do not exist yet are marked
_pending_ with the missing file named, so the draft is always honest about what is and isn't done.

Run:  python -m src.report.build
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.utils.io import load_config, project_path

TAB = project_path("outputs", "tables")
FIG = "../outputs/figures"


def table(name: str, cols=None, index: bool = False, floatfmt: str = ".4f", rows=None) -> str:
    path = TAB / name
    if not path.exists():
        return f"_Pending: `outputs/tables/{name}` not built yet._\n"
    df = pd.read_csv(path)
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    if rows is not None:
        df = rows(df)
    return df.to_markdown(index=index, floatfmt=floatfmt) + "\n"


def figure(name: str, caption: str) -> str:
    if not project_path("outputs", "figures", name).exists():
        return f"_Pending figure: `outputs/figures/{name}`._\n"
    return f"![{caption}]({FIG}/{name})\n\n*{caption}*\n"


def gate_numbers(cfg) -> dict:
    """Headline measurement-validation numbers recomputed from the processed data."""
    out = {}
    p = lambda *a: project_path(*a)  # noqa: E731
    if p("data", "processed", "iv_monthly.parquet").exists():
        from src.data.pull_fred import load_raw
        from src.utils.dates import to_month_end
        iv = pd.read_parquet(p("data", "processed", "iv_monthly.parquet"))
        vix = to_month_end(load_raw("VIXCLS"), iv.index, max_stale_days=5)
        both = pd.concat({"m": iv["iv"], "v": vix ** 2 / 1e4}, axis=1).dropna()
        gap = 100 * (np.sqrt(both["m"]) - np.sqrt(both["v"]))
        out.update(mfiv_n=len(iv), mfiv_corr=both.corr().iloc[0, 1], mfiv_gap_mean=gap.mean(),
                   mfiv_within1=(gap.abs() < 1).mean(), mfiv_fallback=int(iv["fallback"].sum()))
    if p("data", "processed", "rv_daily.parquet").exists():
        rv = pd.read_parquet(p("data", "processed", "rv_daily.parquet"))
        spx = pd.read_parquet(p("data", "raw", "optionmetrics", "spx_index_daily.parquet")).set_index("date")
        r2 = np.log(spx["close"].astype(float)).diff() ** 2
        both = pd.concat({"rv": rv["rv"].rolling(21).sum(), "sq": r2.rolling(21).sum()}, axis=1).dropna()
        out.update(rv_days=int(rv["rv"].notna().sum()), rv_corr=both.corr().iloc[0, 1],
                   rv_overnight=(rv["overnight_ret"] ** 2).sum() / rv["rv"].sum(),
                   rv_first=rv.index.min().date(), rv_last=rv.index.max().date())
    return out


def build(cfg) -> str:
    g = gate_numbers(cfg)
    s, o = cfg["sample"], cfg["oos"]
    n = lambda k, fmt: format(g[k], fmt) if k in g else "_pending_"  # noqa: E731
    parts = [f"""# Equity Variance Risk Premium: Predictability, Stability and an Economical Trading Strategy

QF603 Quantitative Analysis of Financial Markets — draft generated {date.today()} by `python -m src.report.build`.
Every table and figure below is produced by the project pipeline; sections still waiting on inputs say so.

## 1. Introduction

Option prices embed a premium for bearing variance risk. We measure the S&P 500 one-month variance risk
premium (VRP) as model-free implied variance minus a real-time forecast of realised variance, and ask
(Q1) how persistent it is, (Q2) which volatility-market, funding, credit, dealer-capital and positioning
variables forecast it 1, 3 and 6 months ahead, (Q3) whether those relationships are stable across calm and
stressed markets, (Q4) whether they survive the ex-post payoff definition and genuine out-of-sample tests,
(Q5) whether VRP forecasts equity excess returns, and (Q6) whether the findings support an economical,
cost-aware short-variance strategy.

Sample: {s['start']} to {s['end']} (month-ends); train to {o['train_end']}, validate to {o['validate_end']},
holdout to {o['holdout_end']} (used once, after every specification was frozen).

## 2. Data and measurement

### 2.1 Sources

| Series | Source | Coverage used |
| --- | --- | --- |
| SPX options (month-end ± 3 trading days), zero curve, SPX index | WRDS OptionMetrics | 2007-12 → 2025-08 |
| SPY trades (5-minute bars, regular hours) | WRDS daily TAQ | 2003-09-10 → 2025-08 (HAR burn-in from 2003) |
| CP and T-bill rates, Moody's BAA/AAA | FRED (1-business-day publication lag) | 2008 → 2025-08 |
| VIX-futures positioning (TFF, dealers / leveraged funds) | CFTC Public Reporting API | release-dated; Jan–May 2009 and Jan–Feb 2019 missing |
| VIX, VIX3M, VVIX, SKEW, SPXT, PUT, BXM | Bloomberg | {'available' if project_path('data', 'processed', 'bloomberg_daily.parquet').exists() else '_pending user export_'} |
| Intermediary capital ratio | He–Kelly–Manela (3-month publication lag) | {'available' if project_path('data', 'processed', 'hkm_monthly.parquet').exists() else '_pending user download_'} |

OptionMetrics on WRDS ends on 2025-08-29, so the sample was shortened from December to August 2025 (D017).
All conventions, with reasons and alternatives, are logged in `DECISIONS.md`; coverage details in `DATA_INVENTORY.md`.

### 2.2 Implied variance

Thirty-day model-free implied variance follows the Cboe VIX methodology on OptionMetrics closing quotes
(two expiries bracketing 30 days, forward from put–call parity, OTM strip with the two-zero-bid rule, total
variance interpolated to 30 days; AM-settled series expire at 09:30 on the settlement Friday, D028).
All {n('mfiv_n', 'd')} month-ends were computed on the month-end itself ({n('mfiv_fallback', 'd')} fallbacks). Correlation
with VIX²/10000 is **{n('mfiv_corr', '.4f')}** (gate > 0.99); the average gap is {n('mfiv_gap_mean', '.2f')} vol points and
{n('mfiv_within1', '.0%')} of months are within one vol point.

{figure('mfiv_vs_vix2.png', 'MFIV vs VIX² at month-ends')}
### 2.3 Realised variance

Daily realised variance of SPY is the sum of squared 5-minute returns on the NYSE regular-hours grid (78
buckets; 42 on early closes) plus the squared overnight return (Hansen & Lunde 2005), from {g.get('rv_first', '_pending_')} to
{g.get('rv_last', '_pending_')} ({n('rv_days', ',d')} days). Two reverting bad prints were replaced by the bucket VWAP (D036). The
overnight return contributes {n('rv_overnight', '.1%')} of total variance. The 21-day RV correlates **{n('rv_corr', '.3f')}**
with 21-day sums of squared SPX returns and peaks in March 2020 and October–November 2008.

{figure('rv_check.png', '21-day realised volatility: TAQ RV vs squared daily SPX returns')}
### 2.4 Expected realised variance (HAR) and the VRP

Expected one-month RV is a recursive HAR forecast re-estimated at each month-end on targets already observed.
Three variants were evaluated on pre-holdout data only; the **{cfg['har']['headline']} HAR** was chosen as the headline
before any predictability regression was run (D037–D038):

{table('har_evaluation.csv', ['period', 'variant', 'nobs', 'mz_b', 'mz_wald_p', 'mz_r2', 'mse', 'qlike', 'dm_t_qlike_vs_level', 'share_vrp_exante_pos', 'min_vrp_exante'], floatfmt='.3f')}
Ex-ante VRP = MFIV − E_t[RV] (known at t). The ex-post VRP (MFIV − realised RV over the next 21 trading days)
is the short variance-swap payoff; it is a target only, dated by when it becomes known. The trailing VRP
(VIX² − past RV) follows Bollerslev, Tauchen & Zhou (2009) and is used for Q5 only.

{figure('vrp_definitions.png', 'VRP definitions, pre-holdout')}
{figure('har_vs_realised.png', 'HAR forecasts vs realised variance, pre-holdout')}
""",
    "## 3. Results (in-sample 2008–2020)\n",
    "### Q1 — Persistence\n", table("stage1_persistence.csv", index=False), table("stage1_ar2.csv"),
    figure("lp_rv_surprise.png", "Local projections: response of VRP to a realised-variance surprise"),
    "### Q2 — Drivers\n",
    table("stage1_single.csv", ["predictor", "k", "beta", "t_nw", "p_mbb", "p_holm", "incr_r2", "nobs"], floatfmt=".3f"),
    table("stage2_collinearity.csv", floatfmt=".2f"), table("stage2_ols.csv", floatfmt=".3f"),
    table("stage2_pca_variance.csv", floatfmt=".3f"), figure("stage2_scree.png", "PCA scree plot"),
    figure("stage2_ridge_path.png", "Ridge coefficient paths (dashed: CV-chosen λ)"),
    "### Q3 — Stability\n", table("stage3_interactions.csv", floatfmt=".3f"), table("stage3_break_test.csv", floatfmt=".3f"),
    figure("rolling_betas_all.png", "Rolling 60-month betas"), table("stage3_ex2008_09_single.csv", floatfmt=".3f"),
    "## 4. Robustness and out-of-sample (Q4)\n",
    table("stage4_bridge_insample.csv", floatfmt=".3f"), table("stage4_expost_single.csv", floatfmt=".3f"),
    table("stage4_selection.csv", floatfmt=".3f"), figure("oos_cum_sse.png", "Goyal–Welch cumulative SSE differences, validation"),
    "_Validation metrics for PCA / ridge are tuned on the same window; only the holdout (§7) is a clean test._\n",
    "## 5. Returns (Q5)\n", table("stage5_insample.csv", floatfmt=".3f"), table("stage5_oos_validation.csv", floatfmt=".3f"),
    "## 6. Economic significance: trading the VRP (Q6)\n",
    """The traded unit is a short one-month at-the-money SPX straddle with a long ~10-delta put wing, delta-hedged
daily and held to cash settlement; each position is sized so that an SPX −15% / IV × 2 shock costs 20% of
capital, and a month is traded only if its expected P&L exceeds 1.5 × expected round-trip costs (D041–D043).
S0 is always short; S1 is timed by ex-ante VRP; S2 by the Stage 4 payoff forecast.
""",
    table("strategy_validation.csv", floatfmt=".3f"), table("strategy_costs.csv", floatfmt=".3f"),
    figure("strategy_cum_pnl.png", "Cumulative value, validation"), figure("strategy_drawdown.png", "Drawdowns, validation"),
    figure("strategy_cost_breakeven.png", "Break-even cost multiples"), table("strategy_stress.csv", floatfmt=".3f"),
    "## 7. Holdout (2021–2025, run once)\n",
    table("holdout/stage4_oos_metrics_holdout.csv", floatfmt=".3f"), table("holdout/stage5_oos_holdout.csv", floatfmt=".3f"),
    table("holdout/strategy_holdout_frozen_variants.csv", floatfmt=".3f"),
    "### Extended sample (confirmatory)\n", table("holdout/extended_stage1_persistence.csv", floatfmt=".3f"),
    "## 8. Case studies\n_Pending: Bloomberg event-window exports (PLAN Phase 5)._\n",
    """## 9. Limitations

* Sample ends 2025-08 (OptionMetrics coverage), shortening the holdout to 56 months.
* ~155 pre-holdout months with overlapping targets: inference relies on HAC and block-bootstrap corrections.
* SPY's quarterly ex-dividend drops enter overnight returns (not adjusted, D031).
* AM settlement is proxied by the SPX open; the hedge uses the SPX index as a futures proxy (no carry).
* CFTC release dates are inferred; shutdown backlogs are dated conservatively (D025).
* Before 2013 the traded expiry is usually the third-Friday monthly, so positions cover ~2/3 of each month.

## 10. Conclusion
_Written after the holdout run._

## References
Bailey & López de Prado (2014); Bekaert & Hoerova (2014); Bollerslev, Tauchen & Zhou (2009); Bollerslev,
Todorov & Xu (2015); Britten-Jones & Neuberger (2000); Campbell & Thompson (2008); Carr & Wu (2009); Clark & West
(2007); Corsi (2009); Diebold & Mariano (1995); Fleming, Kirby & Ostdiek (2001); Fournier, Jacobs & Orłowski (2024);
Goetzmann, Ingersoll, Spiegel & Welch (2007); Goyal, Welch & Zafirov (2024); Hansen (2000); Hansen & Lunde (2005);
He, Kelly & Manela (2017); Hodrick (1992); Ledoit & Wolf (2008); Patton (2011); Stambaugh (1999).
"""]
    return "\n".join(parts)


def main(cfg=None, force=False):
    cfg = cfg or load_config()
    md = build(cfg)
    out = project_path("report", "report.md")
    out.write_text(md)
    pending = md.count("_Pending")
    print(f"[ok  ] report/report.md written ({len(md.split()):,} words; {pending} pending items)")
    return md


if __name__ == "__main__":
    main()
