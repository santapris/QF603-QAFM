> **Status 2026-10-24:** Parts 0–7 and 10 done on spot data. A code review fixed three timing errors (blackout, OVX, COT); all spreads still fail G1–G5 (not binding). Pending: wiring futures into the model and backtest, the Bloomberg export and binding rerun, the check of 75 OPEC dates, and the report's author line. Optional: Part 8 (live paper routine), Part 9 (hazard model).

# Plan: Oil complex relative value (QF603 Project 3)

## Context
`oil-rv-plan.md` (24 Oct 2026) sets out a 10-step research plan that takes the Project 2 2s10s cointegration trade and applies it to oil. It trades three gaps inside the complex, not the price of oil:
- the 3-2-1 crack, the flagship spread
- Brent–WTI
- diesel vs gasoline (HO−RB)

Brent M1−M6 is used only as a regime input. The folder holds only that markdown, so everything is built from scratch. The structure and much of the code are adapted from `../QAFM Project 2` (package `curve2s10s`).

**Decisions (24 Oct):**
- Mirror Project 2's layout. It becomes a new package, `src/oilrv/`, with `config.yaml`, `PREREGISTRATION.md`, scripts, notebooks, tests and a report.
- **The core trades three spreads:** the crack, Brent–WTI and HO−RB.
- **Data:** build on free FRED/EIA spot data now, with a Bloomberg futures loader specified up front and swapped in once the CSVs are exported from SMU.
- **Extensions:** no preference was given. I've included the Step 10 live paper routine and the hazard model as optional parts. The refiner-equity sleeve and the Asia version are out of scope unless asked for.

**Carried over from Project 2:**
- Pre-register the gates before any analysis.
- No look-ahead: everything at day t uses data up to the close of t, the decision is made at close t and the fill is at close t+1.
- Trade counts and sub-periods are reported next to every metric.
- Both filter-on and filter-off are always shown.
- **Lesson from Project 2:** a formal cointegration test used as the *entry gate* (EG p < 0.05) left almost no tradeable months. Here, cointegration and ADF are reported and used as logit features. They are not hard gates.

## Project layout
```
QAFM Project 3/
  oil-rv-plan.md            (existing, untouched)
  PROJECT_PLAN.md           copy of this plan + status line
  PREREGISTRATION.md        gates, variants, what was known when registered
  config.yaml               single source of truth for every parameter
  requirements.txt  pyproject.toml  README.md  .gitignore
  src/oilrv/
    config.py      load_config(), path()                      ← copy of curve2s10s/config.py
    data.py        FRED cache, EIA spot xls, OVX, GPR, CFTC, EIA weekly stocks
    futures.py     Bloomberg loader, roll schedule, same-delivery-month alignment, back-adjust
    spreads.py     SPREADS spec (legs, weights, $/bbl multiplier), build_spreads()
    calendars.py   events file (OPEC, STEO, IEA, SPR, HURRICANE, HORMUZ), blackout, days-to-event
    explore.py     moments, rolling vol, STL, month-of-year table   ← adapted from curve2s10s
    diagnostics.py ADF/KPSS, rolling ADF, Engle–Granger, Johansen, SARIMA vs seasonal naive
    model.py       fit_window (in-window seasonal + AR(1) + mean/sd), VECM, GARCH-t + frozen filter, model_path
    regime.py      episodes, labels, pooled point-in-time logit/probit, AUC/Brier/calibration
    backtest.py    walk-forward engine (generalised from curve2s10s/backtest.py)
    sizing.py      vol limit, stop limit, cap, book drawdown rules
    metrics.py     Sharpe, max DD, subperiods, exit reasons, gate verdicts  ← copy, extended for G5
    plotting.py    style, save, save_table                           ← copy
    live.py        (optional) daily signal snapshot for Step 10
  scripts/  build_data.py  run_analysis.py  run_model.py  run_backtest.py  daily_signal.py  build_report.sh  check_report.py
  notebooks/ 01_data 02_explore 03_diagnostics 04_model 05_regime 06_backtest 07_current_regime
  data/raw/{fred,eia,gpr,cftc,bloomberg}/  data/clean/  data/derived/  data/calendars/
  output/figures/  output/tables/
  tests/  report/
```

**Code reused from Project 2** (copy into `oilrv` and adapt; nothing is imported across projects):
- `config.py`: `load_config`, `path`
- `data.py`: `_parse_fred_csv`, `fred` (with its on-disk cache), `flag_spikes`
- `calendars.py`: `_get_cached`, `trading_days_to_next`, `extended_trading_days`
- `explore.py`: `daily_changes`, `moments_table`, `moments_by_period`, `rolling_vol`, `stl_decompose`, `stl_strength`, `seasonal_by_month`
- `diagnostics.py`: `month_starts`, `window_before`, `unit_root_table`, `diebold_mariano`
- `model.py`: `fit_garch`, `garch_filter`, `fit_vecm`, `model_path` pattern, `qlike`, `vol_forecast_test`
- `regime.py`: `label_episode`, `build_episodes`, `fit_logit`, `predict`, `regime_probability`, `auc`, `evaluate_filter`
- `backtest.py`: the `Rules` / `FitCache` / `run_backtest` loop (pending order, fill at t+1, re-arm, re-estimate only when flat, exit reasons, mark-to-market at the end)
- `metrics.py`: everything, including `gate_verdict`
- `plotting.py`: everything
- `scripts/build_report.sh` and `check_report.py` (pandoc + Chrome)

---

## Part 0 · Setup and pre-registration (before touching data)
- venv, `requirements.txt` with Project 2's pins plus `xlrd`/`openpyxl` (EIA and GPR xls files), then `pip install -e .`, and an optional `git init`.
- `config.yaml`, with `registered_on: 2026-10-24`:
  - **Sample:** 2007-01-02 to latest; 756-day window, refit monthly (and when flat after an exit).
  - **Rules:**
    - entry 2 ≤ |z| < 4
    - take profit |z| ≤ 0.5
    - stop |z| ≥ 4
    - time stop 3 × half-life
    - re-arm when |z| < 2
    - half-life band [5, 120]
    - logit: enter P ≥ 0.6, exit P < 0.4, min 20 resolved episodes
    - blackout: 1 trading day before OPEC+ meetings and Gulf hurricane landfalls
    - fill lag 1 day
  - **Costs:** $0.05/bbl per unit on every change in position, plus a roll cost of 1 tick per leg rolled (futures only).
  - **Spreads:** see the table below.
  - **Sub-periods:** 2007–13, 2014–19, 2020–26. Stress windows (descriptive): 2011–13, 2020, 2022, 2026.
  - **Robustness grid:** entry {1.5, 2, 2.5}, exit {0.25, 0.5, 0.75}, window {504, 756, 1008}, cost {0, 0.025, 0.05, 0.10, 0.20} $/bbl, label horizon {10, 20, 40}.
  - **Sizing:** $2m book, R = $6,000 a day, L = $40,000 a trade, cap 3 units, halve sizes after a 10% drawdown, stop after 20%.
- **Gates.** A spread passes only if all five hold, on its base parameters, filter on:
  - **G1:** net Sharpe > 0.5.
  - **G2:** at least 20 round trips. The sample is shorter than Project 2's, so the bar is lower than its 30.
  - **G3:** net P&L positive in at least 2 of the 3 sub-periods.
  - **G4:** net Sharpe > 0 at every entry |z| ∈ {1.5, 2, 2.5}.
  - **G5:** in both 2022 and 2026, filtered net P&L ≥ unfiltered net P&L and > −L. This turns "did the filter keep us out" into a gate.
- **Binding data rule.** The verdict is binding on **futures** (Bloomberg). Any backtest run on spot data before then is a *proxy*, labelled as one, and no rule may be changed after seeing it. If Bloomberg never arrives, the spot verdict stands, labelled "spot proxy".
- **Variants.** All are pre-registered and reported side by side:
  - `fixed`: the traded weights. This is the base and carries the gates.
  - `johansen`: monthly Johansen weights, crack only. Robustness, not gated.
  - Filter ladder: core → + GARCH sizing → + blackout → + logit filter.
- `PREREGISTRATION.md` written in Project 2's format, plus `tests/test_preregistration.py` to freeze the gates and the base rules.

| Spread | Definition ($/bbl) | Legs per unit | M ($ per $1/bbl per unit) | Seasonal adj. |
|---|---|---|---|---|
| `crack321` | (2·42·RB + 42·HO − 3·CL)/3 | 3 CL, 2 RB, 1 HO | 3,000 | yes |
| `brent_wti` | Brent − WTI, same delivery month | 1 CO, 1 CL | 1,000 | no (tested in Part 3) |
| `ho_rb` | 42·(HO − RB) | 1 HO, 1 RB | 1,000 | yes |
| `brent_m1m6` | Brent M1 − M6 (futures only) | n/a | n/a | filter input only |

## Part 1 · Data (Steps 01–02) — Group A
- **Spot (now):** FRED DCOILWTICO, DCOILBRENTEU, DGASNYH and OVXCLS. NYH ULSD comes from EIA's daily spot xls (`PET_PRI_SPT_S1_D`), which needs no API key. ULSD history starts in mid-2006, so the sample starts in 2007.
- **Weekly stocks:** EIA WPSR history xls files: crude ex-SPR, Cushing, gasoline, distillate, refinery utilisation. **Stamp each week as available from week-end + 6 calendar days (Thursday).** This is a conservative point-in-time rule that covers holiday-delayed releases.
- **Other features:**
  - Caldara–Iacoviello daily GPR, lagged 1 day.
  - CFTC disaggregated COT, managed-money net for CL, RB and HO. Each report is as of Tuesday and released Friday, so stamp it on the Friday release.
- **Bloomberg (later), `futures.py`:**
  - The export format is fixed now so the SMU pull is mechanical. Run BDH on CL1–12, CO1–12, XB1–6 and HO1–6 for PX_SETTLE and FUT_CUR_GEN_TICKER from 2007. Then run BDP for LAST_TRADEABLE_DT on every unique contract seen. `data/raw/bloomberg/README.md` documents this.
  - Build the roll schedule yourself: hold the nearest contract until 5 business days before its last trade date.
  - Map ICE Brent to the **same delivery month** as WTI.
  - Keep unadjusted prices for P&L checks. Build **additive** back-adjusted series for signals, since additive adjustment survives negative prices.
- **Cleaning:**
  - RB and HO × 42.
  - Work in levels, never logs. Flag 20 Apr 2020 and its week, and run a sensitivity check that drops it.
  - Inner-join dates and flag spikes with `flag_spikes`.
- **Events file** (`data/calendars/events.csv`: date, type, source), built by hand:
  - OPEC/OPEC+ meetings since 2007
  - STEO and IEA OMR release dates
  - SPR and IEA releases
  - Gulf Coast hurricane landfalls (NHC HURDAT2)
  - Hormuz closure and reopening, 2026
- **Outputs:**
  - `data/clean/spot_prices.csv`, `futures_prices.csv` (once available), `spreads.csv`, `features.csv`
  - `output/tables/data_summary.json`
  - a figure of the prices and the three spreads

## Part 2 · Summary moments (Step 03) — Group A
- Mean, sd, skew, excess kurtosis and ARCH-LM of daily spread changes, overall and by 2007–13, 2014–19, 2020, 2022 and 2026.
- 60-day rolling vol.
- Spread-change histograms against normal and Student-t fits.
- **Takeaways:** fat tails point to the t-GARCH and the hard stop; the table shows how far 2026 sits outside history.

## Part 3 · Diagnostics and seasonality (Steps 04–05) — Groups A/B
- **Stationarity:**
  - ADF and KPSS on price levels (expect I(1)) and on each spread (expect stationary, with breaks).
  - Rolling 756-day ADF p-value on each spread. This is a key figure and becomes a logit feature.
- **Seasonality:**
  - STL of monthly `crack321` and `ho_rb`, with a month-of-year table.
  - SARIMA on the monthly crack against a seasonal-naive forecast (rolling OOS RMSE, Diebold–Mariano), with Ljung–Box on the residuals.
  - Test whether `brent_wti` needs seasonal adjustment. Report this, but don't change the pre-registered setting.
- **Long-run link:**
  - **Brent–WTI:** Engle–Granger, full sample and rolling, with a test that the slope = 1.
  - **Crack:** Johansen on (CL, RB, HO) levels. Report the trace rank (expect 2) and compare the first vector with 3:2:1. Check that the fixed-weight spread is itself stationary.
  - **HO−RB:** Engle–Granger on HO against RB.

## Part 4 · Model formation (Step 06) — Group B
- `model.fit_window(win, spread)` returns:
  - in-window month-of-year effects (seasonal spreads only)
  - the deseasonalised mean and sd
  - AR(1) b and the half-life
  - the ADF p-value
  - GARCH(1,1)-t parameters on ΔS
- **Seasonal effects are estimated only inside the window.** A unit test proves that changing future data leaves the fit unchanged.
- **Monthly point-in-time path:** z_t = (S_t − seas_m(t) − μ)/σ. The GARCH σ̂ is filtered forward daily using the frozen parameters, as in Project 2's `garch_filter`, so there is no daily refit.
- **VECM:** VECM on (CL, RB, HO) with the Johansen rank. Rolling α loadings show which leg corrects; the plan expects the products to.
- **Volatility forecast test:** GARCH σ̂ against rolling sd (QLIKE, `vol_forecast_test`).

## Part 5 · Regime filter (Step 07) — Group C
- **Episodes:** one row per first crossing into 2 ≤ |z| < 4, using the parameters frozen on the signal day. Episodes never overlap.
- **Label:** 1 if |z| ≤ 1 within 20 trading days before touching 4. The label counts as resolved on the hit day or at t + 20.
- **Pooled logit:** one model across the three spreads, with spread dummies, because there are too few episodes per spread. It is refit at each month start on episodes resolved *before* that date. With fewer than 20 episodes there is no filter.
- **Core features** (kept small to limit overfitting):
  - OVX percentile
  - GARCH σ̂ percentile
  - rolling ADF p-value
  - GPR
  - inventories against their 5-year seasonal average
- **Robustness features** (pre-registered now, not core):
  - CFTC managed-money z-score
  - hurricane-season dummy
  - event-within-3-days dummy
  - Brent M1−M6 slope / price (futures only)
- **Evaluation:** out-of-sample AUC and Brier score, a calibration plot, and success rates with and without the filter. Probit as a robustness check.
- The OOS diagnostics are reported. **No feature may be added after seeing the 2022/2026 results.**

## Part 6 · Rules, sizing and backtest (Steps 08–09) — Groups B/C
- **Generalise `run_backtest`:**
  - **P&L:** units × M × Δ(back-adjusted spread), with roll costs charged on roll days. On spot data the Δ is just the spot spread change.
  - **Exits:** stop, take profit, time stop, regime break (P < 0.4) and half-life out of band at the monthly re-test.
- **Two P&L streams, as in Project 2:**
  1. **Research stream:** unrounded vol-target units (R/(σ̂·M)), used for the gates. It is scale-free and comparable across spreads.
  2. **Integer book:** the plan's N = min(⌊R/(σ̂·M)⌋, ⌊L/((4 − |z|)·σ_w·M)⌋), capped at 3, with book drawdown rules. This shows the "size alone keeps you flat in 2026" effect from worked example B.
- **Outputs, each with and without the filter:**
  - Sharpe, max drawdown, hit rate, average hold against half-life, worst trade, share of P&L from the top 5 trades
  - trade log with exit reasons
  - sub-periods and stress windows
  - entry × exit grid, cost sweep, window robustness, `fixed` vs `johansen` weights
  - Sharpe t-stats with a multiple-testing note (3 spreads × variants)
- `metrics.gate_verdict` is extended for G5, and a G1–G5 verdict is reported per spread.

## Part 7 · Current regime and strategy-now readout — all groups
- `notebooks/07_current_regime`: for each spread, show today's z, σ̂ and its percentile, P(revert), and the integer size. Also show the status of each switch-on trigger from the plan:
  - P ≥ 0.6
  - HO-crack GARCH vol below its 80th percentile
  - no Hormuz re-closure for 4 weeks
- This is descriptive only, and it frames the HO−RB "diesel first" candidate trade as a test, not a call.
- Dates-to-watch table generated from `events.csv`: 1 Nov OPEC+, Wednesday WPSRs, 10 Nov STEO, 30 Nov end of hurricane season, the G7 release through Feb 2027.

## Part 8 (optional) · Live paper routine (Step 10)
- `scripts/daily_signal.py`:
  - pull the latest settles (Bloomberg CSV drop or FRED/EIA)
  - update σ̂, z and P using the frozen month-start fits
  - print entries, exits and sizes, and blackouts for the next 3 days
  - append to `data/derived/paper_log.csv`
- The monthly refit reuses `run_model.py`.
- **Paper only.** The README notes that the employer's personal-dealing policy needs checking before any real futures trading.

## Part 9 (optional) · Hazard extension
- A discrete-time hazard logit on trade-days (days since entry, |z|, σ̂ percentile) gives a data-driven time stop.
- It is compared with 3 × half-life on the same trades and labelled **exploratory**, since it was designed after the base rules.

## Part 10 · Report
- `report/report.md` → PDF/DOCX through the adapted `build_report.sh`.
- Sections follow the 10 steps.
- Headline: the per-spread G1–G5 verdicts, side by side, with the filter ladder and the 2022/2026 stress result up front.

## Group split and order
Following the plan:
- Group A owns Parts 1–3 (data, moments, diagnostics).
- Group B owns Parts 4 and 6 (model, sizing, engine).
- Group C owns Parts 5 and 6 (filter, backtest evaluation).

**Order:** Part 0 → 1 → 2/3 → 4 → 5 → 6 (spot proxy) → swap in Bloomberg and re-run 1–6 (binding) → 7 → 10, with 8 and 9 if time allows. Roughly 8 weeks, as in the plan, with the week-7 go/no-go replaced by the G1–G5 verdicts.

## Verification
- `pytest -q` covers:
  - **Units:** a known RB/HO/CL input gives the expected crack. On 20 Oct 2026 spot (2.897, 5.265, 107.89) the crack should be about $47/bbl.
  - **Roll schedule:** 5 business days before LAST_TRADEABLE_DT, same delivery month for Brent/WTI, and back-adjusted Δ equals the held contract's Δ.
  - **Point-in-time stamps:** WPSR on Thursday, COT on Friday.
  - **Leakage:** perturbing data after t leaves `fit_window`, the z-path, the logit P(t) and the backtest decisions up to t unchanged.
  - **Engine:** fill lag, cost on every position change, roll cost.
  - **Synthetic data:** a planted AR(1) spread with a 20-day half-life earns a positive Sharpe; a random walk earns about zero.
  - **Pre-registration:** the gates are frozen.
- Run `scripts/build_data.py → run_analysis.py → run_model.py → run_backtest.py` end to end on spot data. Check that `backtest_summary.json` has verdicts for all three spreads and that figures and tables land in `output/`.
- **Sanity checks against the plan's facts:**
  - the rolling ADF dips around 2011–13 (Brent–WTI), 2022 and 2026
  - the crack z is extreme through 2026
  - worked example B (σ̂ = $2.50 gives 0 units) is reproduced by `sizing.py`
- `check_report.py` on the built report.
