# PLAN.md — Equity Variance Risk Premium (QF603)

This file is the working plan for the project. Work through it phase by phase, in order.
Tick checkboxes (`- [x]`) as tasks are completed and keep this file up to date.
Items tagged `[F#]` implement a fix from the logic review (§4); items tagged **NEW** were added in the
holistic revision.

---

## 1. Project summary

Study the S&P 500 one-month variance risk premium (VRP): implied variance minus expected
realised variance. Test whether VRP can be predicted 1, 3 and 6 months ahead by its own lag
and by volatility-market, funding, credit, dealer-capital and positioning variables, whether
those relationships are stable across calm and stressed markets, whether they survive
out-of-sample testing, (secondary) whether trailing VRP predicts S&P 500 excess returns, and
finally whether the findings translate into an **economical, cost-aware VRP trading strategy**.

| Item | Specification |
| --- | --- |
| Target | Ex-ante one-month VRP_t = IV_t − E_t[RV_{t→t+21}] (recursive HAR-RV forecast) |
| Forecast horizons | k = 1, 3, 6 months; target is VRP_{t+k}, predicted with info at t. k = 0 payoff bridge for trading (Stage 4) |
| Predictors | Lagged VRP (control), VIX, VIX3M/VIX, VVIX, SKEW, CP − T-bill, BAA − AAA, HKM capital ratio, CFTC VIX-futures positioning |
| Sample | Jan 2008 – **Aug 2025**, monthly (last trading day of month). Shortened from Dec 2025: OptionMetrics on WRDS ends 2025-08-29 (D017) |
| Split | Train 2008–2015 · Validate 2016–2020 · Holdout Jan 2021 – Aug 2025, 56 months (touched once, Phase 7) |
| Research questions | Q1 persistence · Q2 drivers · Q3 stability · Q4 robustness/OOS · Q5 returns · Q6 economic significance (trading strategy) |

### Phase map and timeline (12 weeks)

| Phase | Weeks | Output | Gate to exit |
| --- | --- | --- | --- |
| 0 Setup | 1 | Repo, config, utils, tests, WRDS access | Tests pass, WRDS connects |
| 1 Data audit & acquisition | 1–3 | Raw data + `DATA_INVENTORY.md` | Coverage gate 🚦 |
| 1.5 Prototype | 2 | Throwaway end-to-end run | Runs start to finish 🚦 |
| 2 Measures | 2–5 | RV, MFIV, HAR, VRP panel | RV / MFIV / VRP sanity gates 🚦 |
| 3 Stages 1–3 (Q1–Q3) | 5–8 | In-sample results 2008–2020 | Tables/figures complete |
| 4 Stages 4–5 (Q4–Q5) | 8–10 | Ex-post, OOS validation, k = 0 bridge, returns | Specs chosen 🧑 |
| 5 Case studies | 9–10 | Event-window analysis | — |
| 6 Trading strategy (Q6) | 9–11 | Validated, cost-aware strategy | Strategy gates 🚦 |
| 7 Freeze, holdout, write-up | 11–12 | Single holdout run, report, slides | Reproducibility check 🚦 |

---

## 2. Hard rules (always follow)

1. **No look-ahead.** Every quantity dated t may use only information **available** at the close of t.
   This applies to the HAR fit, PCA/ridge fits, standardisation, regime thresholds, CFTC report timing,
   **predictor publication lags** `[F4]` and **target observation dates** `[F2][F3]`.
2. **Holdout guard.** No modelling code may use observations whose **target is observed** after
   `oos.validate_end` unless called with `final=True` `[F2]`. Implement this in `src/utils/holdout.py`
   (Phase 0) and use it everywhere. `final=True` is run **exactly once**, in Phase 7, covering Stage 4,
   Stage 5 and Phase 6 together, and only when the user says so `[F14]`.
3. **Never fabricate or silently substitute data.** If a source is missing, stop and tell the user.
   Synthetic data is allowed only in unit tests; prototype stand-ins (Phase 1.5) are logged and deleted.
4. **Log every convention** in `DECISIONS.md` (date, decision, reason, alternatives considered).
   Never change a logged convention without adding a new entry. **Every model/strategy variant tried is
   logged** (needed for the deflated Sharpe ratio in Phase 6).
5. **Single source of truth.** All dates, horizons, split points, lag rules and strategy parameters come
   from `config.yaml`. No hard-coded dates or parameters in scripts.
6. **Units.** All variances are annualised: implied variance over 30 calendar days × 365/30;
   realised variance over 21 trading days × 252/21. Store variances in decimal units (e.g. 0.04 = 20% vol).
   Strategy P&L uses **actual** month-end-to-month-end dates (19–23 trading days), not a fixed 21.
7. **Validation gates** (marked 🚦) must pass before moving to the next step. If one fails, stop and report.
8. **Manual steps** (marked 🧑) need the user (credentials, Bloomberg terminal, approvals). Stop and ask; do not guess.
9. **NEW — Null results are valid.** A failure of OOS predictability or of the strategy to beat its
   benchmark is reported honestly; never re-tune after seeing holdout results.

---

## 3. Environment and repo layout

Python 3.11+. Core packages: `pandas numpy pyarrow statsmodels scikit-learn scipy matplotlib pyyaml wrds pytest`.
Pin exact versions in `requirements.txt`. **NEW**

```
.
├── PLAN.md
├── DECISIONS.md
├── DATA_INVENTORY.md
├── config.yaml
├── requirements.txt
├── run_all.py              # single pipeline entrypoint (or Makefile)
├── .gitignore              # data/, outputs/ excluded — WRDS/Bloomberg data are licensed, not redistributed
├── data/
│   ├── raw/            # untouched downloads, one subfolder per source
│   │   ├── optionmetrics/  taq/  bloomberg/  fred/  cftc/  hkm/
│   └── processed/      # rv_daily, iv_monthly, har_forecasts, panel_monthly, strategy_options_daily (.parquet)
├── src/
│   ├── data/           # pull_optionmetrics.py, pull_optionmetrics_daily.py, pull_taq.py, load_bloomberg.py,
│   │                   # pull_fred.py, pull_cftc.py, load_hkm.py
│   ├── measures/       # rv.py, mfiv.py, har.py, vrp.py
│   ├── models/         # stage1.py … stage5.py
│   ├── strategy/       # signals.py, instruments.py, costs.py, backtest.py, metrics.py   (NEW, Phase 6)
│   └── utils/          # holdout.py, newey_west.py, bootstrap.py, io.py, dates.py
├── tests/
├── outputs/
│   ├── figures/
│   └── tables/
└── report/
```

### `config.yaml` (initial contents)

```yaml
sample:
  start: 2008-01-31
  end: 2025-08-29                # D017
  har_burnin_start: null        # set in Phase 1 to first available daily TAQ date (expected ~2003)
horizons: [1, 3, 6]
vrp_tenor:
  iv_calendar_days: 30
  rv_trading_days: 21
oos:
  train_end: 2015-12-31
  validate_end: 2020-12-31
  holdout_end: 2025-08-29        # D017
target_observation_lag_months:  # [F2][F3] months after t+k at which the target is known
  vrp_exante: 0
  vrp_expost: 2                 # month-granular fallback; exact obs_date overrides (DECISIONS D006)
  excess_return: 0
publication_lag:                # [F4]
  fred_daily_bdays: 1           # H.15 CP / T-bill and Moody's yields posted next business day
  hkm_months: 3                 # quarterly book debt released with a lag; robustness: 1
  cftc: release_date            # Tuesday as-of, Friday release
  bloomberg_bdays: 0
rf:
  series: DTB3                  # [F13] converted to a monthly holding-period return
newey_west:
  rule: "max(k, floor(4*(T/100)**(2/9)))"
  expost_rule: "max(k+1, floor(4*(T/100)**(2/9)))"   # [F6]
bootstrap:
  method: moving_block          # [F11]
  block_length: 6
  reps: 5000
regimes:
  high_vol_vix_percentile: 80
rolling_window_months: 60
seed: 603
strategy:                       # NEW — Phase 6
  execution_lag_days: 1
  risk_budget_stress_loss: 0.20
  stress_scenario: {spx_move: -0.15, iv_mult: 2.0}
  cost_hurdle: 1.5
  multiplier_tiers: [0, 0.5, 1.0, 1.5]
  wing_delta: 0.10
  option_spread_theta: {base: 0.5, stress: 1.0}
  option_fee_per_contract: null # set and log in Phase 6 — never assumed silently
  hedge_cost_bp: {base: 1, stress: 2}
  hedge_band_delta: null        # chosen on validation only
  max_oi_share: 0.05
  crra_gammas: [3, 5]
```

---

## 4. Logic review log (applied throughout this plan)

| ID | Issue in the previous plan | Fix |
| --- | --- | --- |
| F1 | Proposal said Stages 1–3 use the full sample; the plan restricted them to ≤2020. | Primary Q1–Q3 on 2008–2020. After the single holdout run, re-run Q1–Q3 once on the full sample as a confirmatory "extended sample" table, no re-tuning (Phase 7). |
| F2 | Holdout guard checked row dates, not target dates (origin 2020-12, k = 6 → target in 2021). | `guard()` checks the target observation date: t+k (ex-ante VRP, returns); ex-post VRP is known ~t+k+1 but its 21-day window can end inside t+k+2, so the month-granular fallback uses t+k+2 and exact per-row `obs_date` overrides it (D006). |
| F3 | Expanding-window training rows not restricted to already-observed targets. | At origin t use only rows s with target observation date ≤ t. Panel-level look-ahead perturbation test. |
| F4 | Predictor publication lags ignored (FRED next-day; HKM quarterly book debt). | `publication_lag` block in config applied when building the panel. |
| F5 | Holm family counted 9 × 3, but lagged VRP is the control. | Family = 8 predictors × 3 horizons = 24 tests. |
| F6 | Ex-post target spans one extra month; NW lags too short. | NW lags ≥ k+1 for ex-post targets; same HAC lags in CW/DM statistics. |
| F7 | No bridge from forecasting VRP_{t+k} to a trade: the expected payoff of selling variance at t is VRP_t itself. | Stage 4 k = 0 payoff regression; it becomes the strategy signal. |
| F8 | OptionMetrics pull lacked `optionid`, Greeks; strategy needs daily marks of held contracts. | Add fields now; Phase 6 pulls daily history only for selected optionids. |
| F9 | IV-augmented HAR makes VRP partly mechanical in IV. | Flag in the headline-variant decision; report VIFs under each variant. |
| F10 | Validation period is used to tune λ / PCs / specs. | Validation metrics labelled "tuned"; only the holdout is clean. |
| F11 | Small sample (156 months pre-holdout; ~26 non-overlapping at k = 6); persistent predictors. | Moving-block bootstrap p-values for headline tests; Stambaugh-bias flag on persistent predictors. |
| F12 | Prototype blocked on the 🧑 Bloomberg export. | Prototype uses FRED `VIXCLS` + OptionMetrics `secprd` stand-ins. |
| F13 | Risk-free rate undefined. | DTB3 → monthly holding-period return; logged. |
| F14 | Two separate holdout unlocks possible (Stage 4, then strategy). | One combined final holdout run in Phase 7. |

Conventions to log explicitly (not errors): 30-calendar-day IV vs 21-trading-day RV; HAR monthly
regressor over 22 days vs 21-day target; month-end spans of 19–23 trading days.

---

## Phase 0 — Setup (week 1)

- [x] Create the repo layout above, `config.yaml`, empty `DECISIONS.md` and `DATA_INVENTORY.md`.
- [x] **NEW** `git init`, `.gitignore` (data/, outputs/, credentials), `requirements.txt` with pinned versions.
- [x] **NEW** `run_all.py` (or `Makefile`) that runs every phase in order; each step skips if its output exists
      unless `--force`. Set the global seed from `config.seed`.
- [x] Implement `src/utils/holdout.py`: `guard(df, target, k, final=False)` that raises if any row's **target
      observation date** (t + k + `target_observation_lag_months[target]`) is > `oos.validate_end` and
      `final` is False `[F2]`. Pytests: (a) plain row after the cut-off raises; (b) origin 2020-12, k = 6
      raises; (c) ex-post target at origin 2020-11, k = 1 raises; (d) `final=True` passes.
- [x] **NEW** Helper `training_rows(panel, origin, target, k)` returning only rows whose target is observed
      by `origin` `[F3]`. Used by every expanding-window loop.
- [x] Implement `src/utils/dates.py`: month-end trading-day calendar (last NYSE trading day of each month),
      a helper to map any daily series to month-end values (last observation on or before month-end), and a
      helper applying `publication_lag` `[F4]`.
- [x] Implement `src/utils/newey_west.py`: OLS with Newey-West SEs using the lag rules in config (`rule`,
      `expost_rule` `[F6]`); also Hodrick (1992) 1B SEs for overlapping return regressions.
- [x] **NEW** Implement `src/utils/bootstrap.py`: moving-block bootstrap p-values for regression slopes `[F11]`.
- [x] 🧑 Confirm WRDS access works (`wrds.Connection()`; user enters credentials / sets up `.pgpass`).

**Done when:** tests pass, WRDS connection succeeds, `run_all.py --dry-run` lists all steps.
✅ Completed 2026-09-23: 28 tests pass; WRDS connects via `.pgpass` (OptionMetrics visible; TAQ in per-year
`taqm_YYYY` libraries from 2003); decisions D001–D012 logged.

---

## Phase 1 — Data audit and acquisition (weeks 1–3)

### 1a. Coverage audit (do first, before bulk downloads)
- [x] OptionMetrics: find last available date for SPX (secid 108105) in `optionm.opprcdYYYY`.
- [x] TAQ: discover the daily TAQ library/table naming with `db.list_libraries()` / `db.list_tables()`
      (do not hard-code schema names). Find the first available daily-TAQ date for SPY → set
      `sample.har_burnin_start`.
- [ ] HKM: 🧑 user downloads the monthly intermediary capital ratio from the authors' website into
      `data/raw/hkm/`; check last available month.
- [x] CFTC: confirm Traders in Financial Futures (TFF) history covers Cboe VIX futures from 2008.
- [x] Write results to `DATA_INVENTORY.md`: series, source, frequency, first date, last date, gaps,
      month-end sampling rule, **publication lag** `[F4]`, and **which phase uses it**.
- [x] 🚦 If any series ends before 2025-12, **stop and report to the user** with options
      (shorten sample / VIX² fallback for missing IV months / drop predictor). Log the decision.
      ✅ 2026-09-23: OptionMetrics ends 2025-08-29 → user chose to end the sample at Aug 2025 (D017).
      CFTC has no reports 2008-12-16 → 2009-06-02 → Jan–May 2009 left missing (D018).

### 1b. OptionMetrics (`src/data/pull_optionmetrics.py`)
- [x] Pull SPX options **only for month-end trading days plus ±3 trading days buffer**
      (not all days). Fields: `date, exdate, optionid, cp_flag, strike_price, best_bid, best_offer,
      volume, open_interest, am_settlement, impl_volatility, delta, gamma, vega` `[F8]`.
      Note `strike_price` is ×1000.
- [x] Pull the zero curve (`optionm.zerocd`) for the same dates, and SPX index daily `close` and `open`
      (`optionm.secprd`; `open` is the AM-settlement proxy — confirm the field exists, log if not).
- [x] Save per-year parquet files in `data/raw/optionmetrics/`.
- [x] **NEW** Check the raw pull (`python -m src.data.check_optionmetrics` → `outputs/tables/optionmetrics_pull_summary.csv`).
      ✅ 2026-09-23: 13.6M rows, 1,488/1,488 request dates, all 212 sample month-ends present with
      expiries bracketing 30 days; no duplicates; 23 crossed quotes (2013: 16, 2025: 7) to be dropped in 2b (D020).

### 1c. TAQ SPY (`src/data/pull_taq.py`)
- [ ] Server-side extraction, one year per job: SPY trades 09:30–16:00, valid/uncorrected trades only.
      Reduce on the server to 5-minute last-trade prices plus first trade after 09:30 and last before 16:00.
- [ ] Save `data/raw/taq/spy_5min_YYYY.parquet`. Make the script resumable (skip years already saved).
- [ ] Log trade-condition filters in `DECISIONS.md`.

### 1d. Bloomberg (`src/data/load_bloomberg.py`)
- [ ] 🧑 User exports daily PX_LAST from the terminal for `VIX Index`, `VIX3M Index`, `VVIX Index`,
      `SKEW Index`, `SPXT Index`, 2000-01-01 to 2025-12-31, as CSV into `data/raw/bloomberg/`
      (one file per ticker, columns `date,px_last`). Write the loader to validate that format.
- [ ] **NEW** 🧑 Same format, for the strategy benchmarks: `PUT Index`, `BXM Index` (Cboe PutWrite and
      BuyWrite). Optional: `UX1 Index`, `UX2 Index` (front/second VIX futures, cheap alternative instrument).

### 1e. FRED (`src/data/pull_fred.py`)
- [x] Download daily `DCPF3M`, `DCPN3M`, `DTB3`, `DBAA`, `DAAA` via the public FRED CSV endpoint.
- [x] Log which CP series is used for the funding spread (default: `DCPF3M`).
- [x] **NEW** Log the risk-free convention: `DTB3` (discount basis) → monthly holding-period return `[F13]`.
- [x] Apply the 1-business-day publication lag when mapping to month-end `[F4]`.
      ✅ 2026-09-23: raw CSVs + `data/processed/fred_monthly.parquet` (212 month-ends; fund/credit spreads, rf).
      Staleness cap 14 days blanks `fund_spread` at 2020-04 and 2024-02 (D022); rf convention D023.

### 1f. CFTC (`src/data/pull_cftc.py`)
- [x] Download historical TFF files from the CFTC website; extract Cboe VIX futures rows.
- [x] Positioning measure (log in DECISIONS.md before looking at results):
      default = dealer net position / total open interest; alternative = leveraged-fund net / OI.
- [x] Timing: report is as-of Tuesday, released Friday. Assign each report to the month by
      **release date**; month-end value = last report released on or before the month-end date.
      ✅ 2026-09-23: 1,017 weekly reports via the CFTC API (D015) → `cftc_weekly.parquet`, `cftc_monthly.parquet`.
      Measures fixed before computing (D024); release dates inferred, shutdown backlogs at upper-bound dates (D025);
      NaN month-ends: 2009-01 … 2009-05 (no reports, D018) and 2019-01, 2019-02 (shutdown).

### 1g. HKM (`src/data/load_hkm.py`)
- [ ] Load the capital ratio; apply `publication_lag.hkm_months` (default 3; robustness 1) `[F4]`.

**Done when:** every raw series is saved and `DATA_INVENTORY.md` is complete (including lags).

---

## Phase 1.5 — Thin end-to-end prototype (week 2, in parallel with slow pulls)

Purpose: fix timing, unit and P&L-sign conventions early using quick stand-ins.

- [x] IV stand-in: FRED `VIXCLS`²/10000 at month-end (avoids waiting on the 🧑 Bloomberg export) `[F12]`.
- [x] RV stand-in: squared daily SPX close-to-close log returns (OptionMetrics `secprd`).
- [x] Basic recursive HAR (see Phase 2c) on the stand-in RV.
- [x] Build a prototype monthly panel and run Stage 1 single-predictor regressions (Phase 3).
- [x] **NEW** Toy strategy P&L: paper short-variance-swap P&L = notional × `vrp_expost` — confirms sign
      (positive on average, large losses in Oct 2008 / Mar 2020) and units.
- [x] Log in DECISIONS.md that stand-ins are prototype-only.
- [x] 🚦 Check the prototype runs start to finish; results are throwaway. Delete prototype outputs
      once Phase 2 is complete, but keep the code paths (they become the swap-in points).

---
      ✅ 2026-09-23: `python -m src.models.prototype` runs end to end; all 6 sanity checks pass
      (`outputs/prototype/proto_gate.csv`). Outputs still to delete after Phase 2 (kept until then for comparison).
      Level HAR overshoots after spikes with daily-return RV (D027). rv_surprise redefined (D026).

## Phase 2 — Building the measures (weeks 2–5)

### 2a. Daily realised variance (`src/measures/rv.py`)
- [x] 5-minute log returns on the regular-hours grid (78 per full day).
- [x] Early-close days (13:00 close): use the shortened grid; do not pad with zero returns. Flag them.
- [x] Overnight return = log(first price after 09:30) − log(previous day's last price before 16:00).
- [x] `RV_d = overnight² + Σ intraday²`. Save `data/processed/rv_daily.parquet`.
- [x] Unit test: synthetic GBM price path with known σ → RV ≈ σ²·Δt within tolerance.
- [x] 🚦 Plot 21-day summed RV vs 21-day sum of squared SPX daily returns; correlation should be high
      and both should spike in Oct 2008 and Mar 2020. Save to `outputs/figures/rv_check.png`.
      ✅ 2026-09-23: 5,528 days; corr 0.982 with squared SPX returns; peaks Mar 2020 and Oct–Nov 2008; 2 bad prints
      replaced (D036). TAQ checks: all days present, SPY–SPX daily return corr 0.9968.

### 2b. Model-free implied variance (`src/measures/mfiv.py`)
Follow the Cboe VIX methodology:
- [x] At each month-end, select the two expiries bracketing 30 calendar days. Use `am_settlement`
      to set expiry time (AM-settled → market open on expiry date; PM-settled → close). Time in minutes.
- [x] Midquotes; forward F from put-call parity at the strike minimising |C − P|; K₀ = first strike ≤ F.
- [x] OTM puts below K₀, OTM calls above K₀, average of put and call at K₀.
      Stop extending each tail after two consecutive zero-bid strikes.
- [x] Per expiry: σ² = (2/T)·Σ(ΔK/K²)·e^{RT}·Q(K) − (1/T)·(F/K₀ − 1)².
- [x] Interpolate total variance to 30 days; annualise × 365/30.
- [x] If a month-end date has bad/missing data, use the nearest prior buffer date and log it.
- [x] Unit test: synthetic Black-Scholes prices at flat vol σ on a dense strike grid → MFIV ≈ σ².
- [x] Save `data/processed/iv_monthly.parquet`.
- [x] 🚦 Plot MFIV vs VIX²/10000 and their difference over time. Correlation must be > 0.99.
      If not, stop and debug before continuing. Save `outputs/figures/mfiv_vs_vix2.png`.
      ✅ 2026-09-23: 213/213 month-ends, all interpolated, no fallback; corr 0.9986 with VIX²/10000 (FRED VIXCLS
      until the Bloomberg export arrives); 97% within 1 vol pt (D028, D029). Figure `outputs/figures/mfiv_vs_vix2.png`.

### 2c. Recursive HAR forecast (`src/measures/har.py`)
- [x] Target: y_s = (252/21)·Σ_{j=1..21} RV_{s+j}.
- [x] Regressors at s: RV_d = RV_s; RV_w = mean(RV_{s−4..s}); RV_m = mean(RV_{s−21..s}), each annualised
      consistently. (22-day monthly regressor vs 21-day target — log the convention.)
- [x] At each month-end t: estimate OLS on rows s with **s + 21 ≤ t** (targets fully observed by t),
      then forecast E_t[RV] from regressors at t.
- [x] Unit test (look-ahead): perturb all RV after date t and confirm E_t[RV] is unchanged. (`tests/test_har.py`, level variant)
- [x] Implement three variants, selectable by argument (D030):
      (i) level HAR; (ii) log HAR with back-transform exp(ŷ + σ̂²/2);
      (iii) HAR augmented with IV_t (Bekaert & Hoerova style).
- [x] Evaluate each: Mincer-Zarnowitz regression of realised on forecast, QLIKE loss, MSE. (code + tests done; run on TAQ RV after 2a)
      Save table `outputs/tables/har_evaluation.csv`.
- [x] 🧑 Present the evaluation to the user and ask which variant is the headline. → **log HAR** (D037, D038). **Note for the decision:
      variant (iii) makes VRP partly mechanical in IV, so VRP and the VIX predictor become collinear by
      construction — report VIFs under each variant** `[F9]`. Log in DECISIONS.md.
- [x] Also compute RV surprise: realised variance over the trading days in (t−1, t], annualised, minus
      E_{t−1}[RV] (D026 — the literal 21-day window can end after t). Implemented in `vrp_measures`.
- [x] Save `data/processed/har_forecasts.parquet`.

### 2d. VRP measures and monthly panel (`src/measures/vrp.py`)
- [x] **Code ready** (`src/measures/vrp.py` assemble/main, `src/data/load_bloomberg.py`, `src/data/load_hkm.py`;
      panel look-ahead test passes). Final build waits for 🧑 Bloomberg exports, 🧑 HKM file and 🧑 headline HAR
      choice; `python -m src.measures.vrp --draft` builds a NaN-filled draft meanwhile (D032).
- [ ] Ex-ante (headline): `vrp_exante = IV_t − E_t[RV]`.
- [ ] Ex-post (robustness target and strategy payoff, never a predictor): `vrp_expost = IV_t − RV_{t→t+21}`.
      Record its exact observation date `obs_date_vrp_expost` (last day of the 21-trading-day window) in the panel and pass it as `obs_dates` to `guard`/`training_rows` `[F2]` (D006).
- [ ] Trailing (Q5 only): `vrp_trail = VIX²_t/10000 − RV_{t−21→t}`.
- [ ] Merge with month-end predictors **after publication lags** `[F4]`: `vix, vix_ts = VIX3M/VIX, vvix, skew,
      fund_spread, credit_spread, hkm, cftc_pos`, plus `rv_surprise`, SPX excess returns for 1/3/6 months,
      `rf` `[F13]`, and (NEW) `put_ret`, `bxm_ret` benchmark returns.
- [x] **NEW** Panel-level look-ahead test: perturb all raw inputs after date t, rebuild the panel, and confirm
      every column dated ≤ t is unchanged `[F3]`.
- [ ] Save `data/processed/panel_monthly.parquet` and a data dictionary `data/processed/panel_dictionary.md`.
- [ ] Figures: three VRP definitions over time; HAR forecast vs realised.
- [ ] 🚦 Sanity: ex-ante VRP mostly positive; spikes in late 2008 and Mar 2020; ex-post payoff sharply
      negative in those months. Report summary stats table `outputs/tables/summary_stats.csv`.
- [ ] Draft the report's Data & Measurement section now (from DECISIONS.md and DATA_INVENTORY.md).

**Done when:** all four 🚦 gates in Phase 2 pass and the panel + dictionary are saved.

---

## Phase 3 — Stages 1–3: Q1–Q3 (weeks 5–8)

All Phase 3 code must pass data through `holdout.guard()`. The primary sample is Jan 2008 – Dec 2020
(by target observation date). The full-sample version is produced once, after the holdout, in Phase 7 `[F1]`.

### Stage 1 — Baseline (`src/models/stage1.py`)
- [x] **Code + synthetic tests ready** (D033); runs on real data after the final panel and the 🧑 headline HAR choice.
- [ ] Persistence: VRP_{t+k} = a + ρ_k·VRP_t for k = 1, 3, 6. Compare ρ₃ vs ρ₁³, ρ₆ vs ρ₁⁶; if they
      differ materially, test AR(2).
- [ ] Slow updating: VRP_{t+k} on rv_surprise_t for k = 1..6 (local projections). Plot coefficients with
      95% bands → `outputs/figures/lp_rv_surprise.png`.
- [ ] Single predictors: VRP_{t+k} = a + ρ·VRP_t + β·z(X_t) + e, z = standardised using training-window
      mean/sd only. Report β, NW t-stat, **block-bootstrap p-value** `[F11]`, incremental R² over AR(1).
- [ ] Multiple testing: Holm-adjusted p-values across **8 predictors × 3 horizons = 24 tests** `[F5]`.
- [ ] ADF tests on all predictors; flag highly persistent ones (Stambaugh-bias risk) in the table notes `[F11]`.
- [ ] Output: `outputs/tables/stage1_*.csv`.

### Stage 2 — Multivariate (`src/models/stage2.py`)
- [x] **Code + synthetic tests ready** (D034); runs on real data after the final panel and the 🧑 headline HAR choice.
- [ ] Correlation matrix + VIFs → `outputs/tables/stage2_collinearity.csv`.
- [ ] Full OLS with all predictors (reference), with bootstrap p-values.
- [ ] PCA on standardised predictors: loadings table, scree plot, economic labels for leading PCs
      (propose labels; 🧑 user confirms).
- [ ] Ridge with λ chosen by expanding-window time-series CV (never random k-fold), using `training_rows()`
      `[F3]`. Coefficient-path plot. (In-sample/descriptive; the final λ is chosen in Stage 4.)
- [ ] Output tables and figures under `stage2_*`.

### Stage 3 — State dependence (`src/models/stage3.py`)
- [x] **Code + synthetic tests ready** (D035; sup-Wald uses a wild bootstrap after the asymptotic test
      over-rejected 15–36% in simulation); runs on real data after the final panel and the 🧑 headline HAR choice.
- [ ] Interactions: X_t × 1{VIX_t > p80 of VIX}, threshold computed from training data only; also X_t × VIX_t.
- [ ] Rolling 60-month betas for VRP lag and key predictors with bands → `outputs/figures/rolling_betas_*.png`.
- [ ] Andrews sup-Wald break test (15% trimming) on the main specification.
- [ ] Re-run main Stage 1–2 results excluding 2008–2009.
- [ ] Optional (only if time permits): 2-state Markov-switching AR on VRP.
- [ ] **NEW** Note for Phase 6: record whether predictor betas are unstable in the high-VIX regime (this sets
      the strategy's multiplier cap in stress).

**Done when:** all Stage 1–3 tables/figures exist and the key findings are summarised in 5–10 bullet points
at the top of `outputs/tables/README.md`.

---

## Phase 4 — Stages 4–5: Q4–Q5 (weeks 8–10)

### Stage 4 — Robustness and out of sample (`src/models/stage4.py`)
- [x] **Code + synthetic tests ready** (D039): ex-post re-runs (control = ex-ante VRP), k = 0 bridge, OOS engine,
      CW/DM, GW plot, selection table. Runs once the final panel exists (🧑 Bloomberg + HKM).
- [ ] Re-run Stage 1–3 specifications with `vrp_expost` as dependent variable, NW lags ≥ k+1 `[F6]`.
- [ ] **NEW — k = 0 payoff bridge** `[F7]`: `vrp_expost_t = a + b·vrp_exante_t + c'X_t + e`.
      Test b = 1 (is ex-ante VRP an unbiased forecast of the short-variance payoff?) and c = 0 (do predictors
      forecast the payoff and crash losses beyond ex-ante VRP?). NW lags ≥ 1.
- [ ] OOS design: train 2008-01 – 2015-12; validate 2016-01 – 2020-12; holdout 2021-01 – 2025-08 (D017).
- [ ] Expanding-window monthly re-estimation during validation, using only rows whose target is observed
      by the origin (`training_rows()`) `[F3]`. Inside each training window, re-fit the HAR, standardisation,
      PCA and ridge — nothing fitted on future data.
- [ ] Run the k = 0 bridge through the same loop → OOS `E_t[payoff]` series for 2016–2020 (Phase 6 signal).
- [ ] Benchmarks: historical mean, AR(1). Metrics: OOS R² vs AR(1); Clark-West (nested);
      Diebold-Mariano (non-nested), with HAC lags matching the forecast overlap `[F6]`.
- [ ] Goyal-Welch cumulative SSE-difference plot → `outputs/figures/oos_cum_sse.png`.
- [ ] Use validation results only to choose λ, number of PCs and the final specification set. Label all
      validation metrics as **"tuned"** in tables; only the holdout is a clean test `[F10]`.
      🧑 Present the chosen specifications to the user and log them in DECISIONS.md.
- [ ] The final holdout run is **not** done here — it moves to Phase 7 `[F14]`.

### Stage 5 — Returns check (`src/models/stage5.py`)
- [x] **Code + synthetic tests ready** (D040); runs once the final panel exists (🧑 SPXT and VIX from Bloomberg).
- [ ] Cumulative log excess returns (over `rf` `[F13]`) over k = 1, 3, 6 months on `vrp_trail`.
- [ ] Newey-West and Hodrick (1992) 1B SEs side by side.
- [ ] OOS (validation period): Campbell-Thompson R² vs historical mean, with and without sign restrictions
      (non-negative slope, non-negative equity premium forecast).
- [ ] Extension: same regressions with `vrp_exante`.
- [ ] Output `outputs/tables/stage5_*.csv`.

**Done when:** specifications are chosen and logged 🧑, and the validation OOS tables exist.

---

## Phase 5 — Case studies (weeks 9–10)

- [ ] Identify the 3–4 largest ex-ante VRP shocks (largest monthly changes). Likely candidates:
      Oct 2008, Aug 2011, Feb 2018, Mar 2020, Apr 2025. Confirm with the data.
- [ ] 🧑 For each, user pulls from Bloomberg (±2 weeks): option surface / skew, VIX term structure,
      VVIX, positioning. Build event-window plots from those exports.
- [ ] **NEW** Overlay the Phase 6 strategy P&L (S0 and S1, with and without the wing) on each event window:
      what did the short-variance book lose, and did the timing signal reduce exposure beforehand?

---

## Phase 6 — Economical VRP trading strategy (Q6, weeks 9–11) — NEW

**Status 2026-09-23:** contracts + daily data pulled (6.3 ✅, D041); instruments, costs, backtest engine,
signals, metrics and the Phase 6 driver (`src/strategy/run.py`) written and unit-tested (D042). S0 engineering check on
the training window only. Validation runs wait for: 🧑 option fee, final panel (Bloomberg/HKM), Stage 4 selection (S2),
Stage 3 high-VIX finding (cap).

### 6.1 Economic rationale
Option sellers earn the VRP as payment for bearing variance and crash risk (Carr & Wu 2009; Bollerslev,
Todorov & Xu 2015). The expected payoff of selling one-month variance at t is E_t[IV_t − RV_{t→t+1}] = the
ex-ante VRP_t `[F7]`. The strategy sells one-month S&P 500 variance **only when the expected premium, after
costs, is large relative to its tail risk**, sizes each position from a risk budget, and caps crash losses
with a long put wing.

"Economical" here means: (1) it trades only when the expected edge exceeds expected costs; (2) turnover is
low (monthly, with a no-trade band); (3) it avoids paying spreads it does not have to (hold to settlement);
(4) capital use is efficient (defined-risk structure, idle capital earns rf); (5) it has a clear economic
reason to work, taken from Q1–Q5.

| Research result | Strategy use |
| --- | --- |
| Q1 persistence | Signals move slowly → monthly rebalancing with a no-trade band (low turnover) |
| Q2 drivers + Stage 4 k = 0 bridge | Expected payoff `μ̂_t = E_t[vrp_expost_t]` is the timing signal |
| Q3 regimes / breaks | Cap the multiplier in the high-VIX regime if betas are unstable there |
| Q4 OOS gate | The model-timed variant (S2) is kept only if validation shows OOS value net of costs |
| Q5 returns | Optional equity-timing overlay, only if Q5 OOS R² > 0 |

### 6.2 Instruments (fixed across variants, so only the timing differs)
- **Primary (implementable):** short one-month SPX ATM straddle, delta-hedged with ES futures, plus a **long
  put at `strategy.wing_delta` (~10Δ)** for defined crash risk and lower margin. Expiry: the listed SPX/SPXW
  expiry nearest to the next month-end rebalance date (on or after it where available; log the rule). Hold
  to cash settlement (AM-settled → SPX open proxy), so no exit spread is paid.
- **Paper benchmark (frictionless):** variance-swap payoff = notional × `vrp_expost_t` (already in the panel).
- **Optional cheap alternative:** short front-month VIX future (UX1) on the same signal — shows the
  cost-vs-fidelity trade-off.
- **External check:** Cboe PUT and BXM index returns over the same dates.

### 6.3 Data (`src/data/pull_optionmetrics_daily.py`)
- [x] For each rebalance date, choose the contracts (straddle strikes, wing strike, expiry) from the 1b
      month-end data, then pull **daily** `best_bid, best_offer, impl_volatility, delta, gamma, vega,
      open_interest` for **only those optionids** through expiry `[F8]`. Save
      `data/processed/strategy_options_daily.parquet`.
- [x] Daily SPX close/open from `secprd` for delta hedging and settlement.

### 6.4 Signal, sizing and trade rules (`src/strategy/signals.py`, `backtest.py`)
- [ ] **Timing:** signal computed at month-end close t; trade executed at the **t+1 close with t+1 quotes**
      (`execution_lag_days`), so no quote is shared with the MFIV used in the signal.
- [ ] **Base size from risk budget:** notional such that the stressed loss (`stress_scenario`: SPX −15%,
      IV × 2, full repricing including the wing) ≤ `risk_budget_stress_loss` × capital. Unused capital earns rf.
- [ ] **Cost hurdle:** trade only if `μ̂_t > cost_hurdle × expected round-trip cost` (from 6.5); otherwise
      hold T-bills.
- [ ] **Multiplier:** m_t ∈ `multiplier_tiers` from terciles of `μ̂_t / σ̂_t` over the expanding training
      window. σ̂_t from an expanding-window regression of squared payoff residuals on VIX² and VVIX. The size
      changes only when the tier changes (no-trade band). Cap m_t at 1.0 in the high-VIX regime if Stage 3
      found unstable betas there.
- [ ] **Delta hedging:** daily ES hedge with a band (`hedge_band_delta`); choose band vs strict daily on
      validation only.
- [ ] **Pre-registered variants (kept small to limit data mining):**
      - **S0 Always-short:** unconditional, constant risk budget (benchmark).
      - **S1 VRP-timed:** `μ̂_t = vrp_exante_t` — no fitted predictors, the most robust variant.
      - **S2 Model-timed:** `μ̂_t` = the Stage 4 k = 0 OOS forecast with the selected predictors.
      - **B (optional) Equity overlay:** ES exposure scaled by `vrp_trail`; only if Q5 OOS R² > 0.
- [ ] Log every variant and parameter value tried in DECISIONS.md (counts feed the deflated Sharpe ratio).

### 6.5 Costs and capacity (`src/strategy/costs.py`)
- [ ] Options: fill at mid ± θ × half-spread from OptionMetrics bid/ask (θ base 0.5, stress 1.0), plus
      `option_fee_per_contract` (value set and logged, never assumed silently).
- [ ] Hedge: ES at `hedge_cost_bp` of traded notional per side (base 1 bp, stress 2 bp).
- [ ] Capacity: position ≤ `max_oi_share` of open interest at each strike; report the AUM at which it binds.
- [ ] Break-even analysis: the cost multiple at which S1/S2 lose their Sharpe and CE advantage over S0 →
      `outputs/figures/strategy_cost_breakeven.png`.

### 6.6 Evaluation (`src/strategy/metrics.py`) — validation 2016–2020 only until Phase 7
- [ ] Returns on capital: mean excess return, vol, Sharpe, Sortino, skew, kurtosis, max drawdown, worst month,
      CVaR 95/99, turnover, cost drag.
- [ ] Short-vol-aware: Goetzmann et al. (2007) MPPM; certainty-equivalent gain vs S0 at CRRA γ ∈ `crra_gammas`
      (Fleming, Kirby & Ostdiek 2001 performance-fee style).
- [ ] Inference: Ledoit–Wolf (2008) HAC Sharpe-difference test vs S0; Bailey & López de Prado (2014) deflated
      Sharpe ratio using the logged number of variants.
- [ ] Attribution: regress strategy excess returns on SPX excess return and on PUT index returns (separates
      timing alpha from passive short-vol beta).
- [ ] Stress table: P&L in Oct 2008, Aug 2011, Aug 2015, Feb 2018, Mar 2020, 2022, Apr 2025 — with and without
      the wing. (Pre-2016 episodes come from the expanding-window backfill used only for description.)
- [ ] Outputs: `outputs/tables/strategy_{validation,costs,stress,attribution}.csv`,
      `outputs/figures/strategy_cum_pnl.png`, `strategy_drawdown.png`, `strategy_cost_breakeven.png`.
- [ ] 🚦 Gates:
      - Unit test: paper variance-swap P&L reconciles to `vrp_expost × notional`; zero costs ⇒ net = gross.
      - S0 monthly returns correlate positively with the PUT index.
      - If S2 does not beat S1 net of costs on validation, drop S2 before the holdout and report it.
- [ ] 🧑 Present the validated strategy set and parameters to the user; log them in DECISIONS.md.

**Done when:** gates pass and the strategy spec is ready to be frozen.

---

## Phase 7 — Freeze, single holdout run, write-up (weeks 11–12) — NEW

**Status 2026-09-24:** tooling built and tested (D044): `python -m src.final.freeze` (refuses while decisions are
open), `python -m src.final.holdout_run --confirm HOLDOUT` (single-use, hash-checked, locked), `python -m src.report.build`
(report draft with Data & Measurement written from real outputs), `python -m src.final.repro` (15/15 files identical so far).
Freeze → approval → holdout wait for Stages 3–6 on the final panel. Slides after the holdout results.

- [ ] **Freeze:** add a pre-registration entry to DECISIONS.md: final forecasting specs (HAR variant,
      predictors, λ, number of PCs), Stage 5 specs, strategy variants and all parameters, plus a hash of
      `config.yaml`.
- [ ] **Final holdout run:** 🧑 only after the user explicitly approves. Run **once** with `final=True` for
      Stage 4, Stage 5 and Phase 6 together `[F14]`. Save results to `outputs/tables/*_holdout.csv`.
      Do not re-tune afterwards.
- [ ] **Extended sample:** re-run the Q1–Q3 headline tables on the full 2008-01 – 2025-08 sample, no re-tuning,
      labelled "confirmatory" `[F1]`.
- [ ] **Report** in `report/`: Introduction; Data & measurement (incl. MFIV and HAR validation);
      Results Q1–Q5; Robustness & OOS; **Economic significance: trading the VRP (Q6)**; Case studies;
      Limitations; Conclusion; References.
- [ ] **Slides** for the presentation (same figures).
- [ ] 🚦 **Reproducibility check** (tool ready; phases 1–2 already pass 15/15): on a clean checkout with raw data in place, `run_all.py` regenerates every
      figure and table in the report from `panel_monthly.parquet` and the strategy data.

---

## Testing summary

| Test | Where |
| --- | --- |
| Holdout guard incl. target-date boundary cases `[F2]` | `tests/test_holdout.py` |
| `training_rows()` excludes unobserved targets `[F3]` | `tests/test_holdout.py` |
| Publication-lag mapping `[F4]` | `tests/test_dates.py` |
| RV on synthetic GBM | `tests/test_rv.py` |
| MFIV on synthetic Black-Scholes | `tests/test_mfiv.py` |
| HAR look-ahead perturbation | `tests/test_har.py` |
| Panel-level look-ahead perturbation `[F3]` | `tests/test_panel.py` |
| Strategy P&L reconciliation; zero-cost ⇒ gross | `tests/test_strategy.py` |

---

## Deliverables checklist

**Figures:** rv_check · mfiv_vs_vix2 · vrp_definitions · har_vs_realised · lp_rv_surprise ·
stage2 scree/coef paths · rolling_betas · oos_cum_sse · case-study event plots (with strategy P&L overlay) ·
strategy_cum_pnl · strategy_drawdown · strategy_cost_breakeven

**Tables:** summary_stats · har_evaluation · stage1 · stage2_collinearity · stage2 (OLS/PCA/ridge) ·
stage3 (interactions, break test, ex-2008–09) · stage4 (ex-post, k = 0 bridge, OOS metrics) · stage5 ·
strategy (validation, costs, stress, attribution) · holdout results (Stage 4, 5, strategy) ·
extended-sample Q1–Q3

**Documents:** DECISIONS.md (incl. pre-registration) · DATA_INVENTORY.md · panel_dictionary.md · report · slides

---

## Risks and fallbacks

| Risk | Fallback |
| --- | --- |
| OptionMetrics ends before Dec 2025 | **Occurred** (ends 2025-08-29). User chose to shorten the sample to Aug 2025 (D017) |
| TAQ pull too slow | Burn-in from first daily-TAQ date; resumable yearly jobs; prototype stand-ins meanwhile |
| MFIV fails the VIX² gate | Stop; debug expiry timing, forward, strike filters before anything else |
| Poor HAR forecast | Compare level/log/IV-augmented variants; report evaluation openly |
| Overfitting (~216 obs; 156 pre-holdout) | Standardisation, PCA/ridge, holdout guard, Holm adjustment, block bootstrap |
| HKM or CFTC gaps | Report subsample results; do not interpolate without logging. CFTC: Jan–May 2009 left missing (D018) |
| HKM real-time timing uncertain | Default 3-month lag; robustness with 1-month lag |
| Strategy daily option data too costly to pull | Paper variance-swap P&L + VIX-futures variant; report the limitation |
| SPXW expiries not aligned to month-end before ~2016 | Nearest listed expiry on/after rebalance date; log the rule |
| Short-vol blow-up dominates results | Wing is mandatory; report stress table, MPPM, CVaR alongside Sharpe |
| Strategy overfitting / data mining | ≤ 3 variants (+1 optional overlay), all logged; deflated Sharpe; single holdout run |

---

## References

Britten-Jones & Neuberger (2000); Carr & Wu (2009); Corsi (2009); Hansen & Lunde (2005);
Bekaert & Hoerova (2014); Bollerslev, Tauchen & Zhou (2009); Bollerslev, Todorov & Xu (2015);
He, Kelly & Manela (2017); Fournier, Jacobs & Orłowski (2024); Goyal, Welch & Zafirov (2024);
Stambaugh (1999); Fleming, Kirby & Ostdiek (2001); Goetzmann, Ingersoll, Spiegel & Welch (2007);
Ledoit & Wolf (2008); Bailey & López de Prado (2014).
