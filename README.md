# QF603 Project 3: oil complex relative value

Implements `oil-rv-plan.md`. See `PROJECT_PLAN.md` for the plan, split into Parts 0–10.
The strategy trades three spreads inside the oil complex: the 3-2-1 crack, Brent–WTI and diesel minus gasoline (HO−RB).

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
```
Register the notebook kernel:
```bash
.venv/bin/python -m ipykernel install --user --name oilrv --display-name "Python (oilrv)"
```

## Run
| Part | Command | Output |
|---|---|---|
| 1 · Data | `.venv/bin/python scripts/build_data.py [--refresh]` (~1 min first run, CFTC history) | `data/clean/` (`spot_prices`, `spreads`, `features`, `weekly_stocks`, `event_features`; `futures_spreads` once Bloomberg files exist), `data/calendars/events.csv`, `output/tables/data_summary.json`, `output/figures/01_*.png` |
| 2–3 · Analysis | `.venv/bin/python scripts/run_analysis.py [--skip-slow]` (~40 s) | `output/tables/02_*`, `03_*`, `analysis_summary.json`; `output/figures/02_*`, `03_*`; `data/derived/` (`rolling_stationarity`, `rolling_eg`, `rolling_johansen`, `sarima_forecasts_*`) |
| 4 · Model | `.venv/bin/python scripts/run_model.py` (~20 s) | `data/derived/rolling_fits.csv`, `model_path.csv` (keyed `variant\|spread`), `output/tables/04_*`, `model_summary.json`, `output/figures/04_*` |
| 5 · Regime filter | `.venv/bin/python scripts/run_regime.py` (~10 s; needs `run_model.py` first) | `data/derived/regime_episodes.csv`, `regime_probability.csv`, `regime_coefficients.csv`, `output/tables/05_*`, `regime_summary.json`, `output/figures/05_*` |
| 6 · Backtest | `.venv/bin/python scripts/run_backtest.py` (~70 s; needs `run_model.py` and `run_regime.py`) | `output/tables/06_*` (ladder, gate verdicts, sub-periods, stress windows, grid, cost sweep, robustness, integer book, trades), `backtest_summary.json`, `output/figures/06_*`, `data/derived/backtest_daily_*`, `integer_book_*` |
| 7 · Current regime | `.venv/bin/python scripts/run_readout.py [--today YYYY-MM-DD]` (~6 s; needs Parts 4–6) | `output/tables/07_state.csv`, `07_triggers.csv`, `07_watch_list.csv`, `readout_summary.json`, `output/figures/07_current_regime.png` |
| 10 · Report | `scripts/build_report.sh` then `.venv/bin/python scripts/check_report.py` | `report/report.pdf`, `report/report.docx` (source `report/report.md`; needs Homebrew `pandoc` and Google Chrome). The check confirms that every quoted number matches `output/` |
| Tests | `.venv/bin/python -m pytest -q` | |
| Notebooks | `notebooks/01_data` … `07_current_regime` (kernel "Python (oilrv)") | |

## Pre-registration
- `PREREGISTRATION.md` and `config.yaml` fix the following, all registered on 2026-10-24 before any data was downloaded:
  - the five success gates (G1–G5)
  - the base rules
  - the spread definitions
  - the regime-filter features
  - the binding-data rule
  - two variants: `fixed`, which is gated, and `johansen` (crack only), which is robustness
- `tests/test_preregistration.py` fails if any of them change.
- Robustness settings go under `robustness:`.
- **The verdict is binding on Bloomberg futures.** Any backtest run on FRED/EIA spot data is labelled "spot proxy".

## Layout
- `src/oilrv/`: the package. So far:
  - `config.py`
  - `data.py`: FRED, EIA, GPR, CFTC, point-in-time stamping, cleaning
  - `spreads.py`: spread definitions and units
  - `futures.py`: Bloomberg loader, expiry rules, same-delivery-month rolls, back-adjustment
  - `calendars.py`: events file, HURDAT2 landfalls, blackout and calendar features
  - `explore.py`: moments, rolling vol, STL, month-of-year effects
  - `diagnostics.py`: ADF/KPSS, rolling stationarity and half-life, SARIMA against naive forecasts, Engle–Granger, Johansen
  - `backtest.py`: walk-forward engine (`SpreadSim` state machine, `run_backtest`, `run_book` for the integer book)
  - `sizing.py`: vol and stop limits, integer units, drawdown multiplier
  - `metrics.py`: Sharpe, drawdown, trade statistics, G1–G5 verdicts
  - `readout.py`: point-in-time GARCH path and expanding percentile, the plan's switch-on triggers, the watch list
  - `regime.py`: filter features, episodes and frozen-parameter labels, the pooled point-in-time logit/probit, AUC/Brier/calibration
  - `model.py`: `fit_window` (in-window seasonal fair value, AR(1) half-life, ADF, GARCH-t, VECM with the traded vector), Johansen weights, rolling fits, the point-in-time daily path, the volatility-forecast test
  - `plotting.py`
  
  The other modules are added part by part, following `PROJECT_PLAN.md`.
- `data/raw/`: downloads exactly as received, never edited. Subfolders: `fred`, `eia`, `gpr`, `cftc`, `bloomberg`. See `data/raw/bloomberg/README.md` for the export format.
- `data/clean/`, `data/derived/`: cleaned inputs and model outputs.
- `data/calendars/`:
  - `events_manual.csv` is built by hand, with a `verified` column and a source for every row.
  - `events.csv` is generated from it, with HURDAT2 landfalls added.
- `output/figures/`, `output/tables/`, `notebooks/`, `report/`.

## Implementation choices not fixed by the pre-registration
These were decided in Part 1, before any model or backtest was run.
- **Sample alignment.** Spot data keeps only days with all four legs (WTI, Brent, NYH gasoline, NYH ULSD), from 2007-01-03: 4,848 days to 2026-10-20. UK-only holidays (no Brent print) drop out for every spread.
- **Point-in-time stamps.**
  - Weekly EIA stocks are usable from week-end + 6 days (Thursday).
  - COT positions from as-of + 6 days (the Monday after the Friday 3:30 pm ET release, which comes after the NYMEX settle).
  - GPR is lagged 1 day.
  - OVX is lagged one day (it closes at 4:15 pm ET, after the 2:30 pm settle) and carried forward on days it is missing.
- **`inventory_dev_5y`.** Total crude ex-SPR + gasoline + distillate, in % against the same week (52·k weeks earlier) in the previous five years. Only past weeks are used.
- **Hurricanes.**
  - Hurricane-strength landfalls (HURDAT2 record `L`, status `HU`) within 25–31.5°N, 98–85°W: Texas to the Florida Panhandle.
  - Tropical storms and Florida west-coast landfalls are excluded.
  - Seasons not yet in HURDAT2 are added by hand (2026: Isaias).
- **Events.**
  - Only scheduled OPEC meetings drive the blackout. Surprise announcements are kept but flagged `scheduled=False`.
  - The blackout applies to the fill: a decision at t is blocked when the event is within `event_blackout_days + fill_lag_days` (= 2) trading days, so no entry is filled on the day before an event.
  - STEO and IEA report dates are left out, because their archives refuse automated access.
- **Model (Part 4).**
  - Seasonal effects are in-window month means of (S − window mean); a month missing from the window gets 0.
  - AR(1) (with constant), the half-life and the ADF p-value are computed on the deseasonalised spread. For seasonal spreads, the regime filter's `rolling_adf_p` will be this deseasonalised ADF p from each monthly fit. Part 3's raw-spread ADF is the descriptive version.
  - The z-score on day t uses the month effect of t's calendar month.
  - GARCH(1,1)-t is fitted on raw daily spread changes (the P&L series), with `rescale=False`. It falls back to the window sd if the fit fails, does not converge, or has α + β ≥ 1.
  - The VECM fixes the cointegrating vector at the traded weights (rank 1, one lag) and estimates it by OLS equation by equation.
  - `johansen` variant: the first vector on (CL, RB, HO) from the same window, scaled to −3 CL. A month has no usable vector, and the gate is closed, if the trace rank is 0, the vector is not finite, or any |weight| > 10. Otherwise it is fitted exactly like `fixed`, seasonal adjustment included.
- **Regime filter (Part 5).**
  - **Episodes:** every crossing into 2 ≤ |z| < 4 after |z| has been below 2 counts, whether or not the half-life gate is open. The gate is recorded, and a gate-open-only evaluation is reported. An episode not yet resolved at the end of the data is dropped.
  - **Features:**
    - `ovx_pct` and `garch_vol_pct` are percentiles of today's value within the trailing 756 days (at least 126).
    - `gpr` is log(1 + GPR), because one day (2025-02-10) has GPR = 0.
    - `rolling_adf_p` is the deseasonalised ADF p from the current monthly fit.
  - **Robustness features:**
    - `cftc_mm_z` is the 756-day z-score of managed-money net in the spread's direction: RB + HO for the crack, −CL for Brent–WTI, HO − RB for HO−RB.
    - `brent_m1m6_slope_pct` exists only with futures.
  - **Training:** only `fixed` episodes train the pooled logit. The `johansen` crack gets its P from that model, with the crack dummy and its own features.
  - **Failed fits:** a fit that fails or does not converge leaves P missing for that month, so no filter applies. The entry rule is P ≥ 0.6, inclusive.
- **Backtest (Part 6).**
  - **Start:** the first month start with a full window (2010-02-01 for 756 days). Daily Sharpe counts flat days as zero.
  - **Fills and costs:** decide at close t, fill at close t+1. The cost is \$0.05/bbl × units × M per side. Research units are fixed at entry.
  - **Refits:** at each month start while flat, and right after an exit, on rows strictly before the fill day. While in a trade, the month-start refit is only used for the model-break test (half-life out of band, or no usable Johansen vector).
  - **G4 grid:** the re-arm level equals the grid's entry |z|, and the exit stays at 0.5. G5's "unfiltered" run is ladder step `blackout` (every rule except the logit).
  - **Integer book:**
    - σ_w in the stop limit is the flat fit's window sd of the deseasonalised spread.
    - The drawdown multiplier (1 / 0.5 / 0) applies to new entries only.
    - Reaching the 20% stop closes every position and ends trading for good ("review").
- **Readout (Part 7).**
  - "Volatility under its 80th percentile" uses the full point-in-time history since 2010 (an expanding percentile), not the trailing window.
  - The relevant volatility is the diesel crack (HO − CL) for HO−RB and the 3-2-1 itself for the crack.
  - The Hormuz condition counts calendar days from the last logged closure to today.
  - The integer size is shown at max(|z|, 2), and as 0 at or past the stop.
- **Futures.**
  - Expiries come from the exchange rules, and contract years are resolved from the observation date.
  - All legs of a spread hold the same delivery month and roll 5 business days before the earliest leg expiry.
  - Back-adjustment is additive.

## Data caveats
- **The futures run is not wired in yet.** `futures.build_futures()` builds the per-spread series, but `model`, `regime` and `backtest` still take one shared set of spot legs. Futures need per-spread legs (each spread holds its own delivery month) and the `rolled` flags (for roll costs) passed through. This has to be done before the binding run.
- **GPR timing.** The pre-registered spec lags daily GPR by one day, but the published file is updated only periodically, so in real time yesterday's value is not always available.
- **OPEC dates need checking.** 75 of the 98 OPEC events were compiled from memory (`verified=no`), because opec.org blocks automated access. Check them against the opec.org press-release archive. They only move the 1-day entry blackout.
- **Gasoline grade switches.** NYH spot gasoline changes grade (RVP) around 1 Feb and 1 Mar, which shows up as reversing spikes in the spot crack and HO−RB. They are flagged, not dropped. Futures avoid this.
- **April 2020 distorts the rolling half-life.** The single 20 Apr 2020 print pulls the raw-spread AR(1) half-life of Brent–WTI to about 1.4 days, and the crack's to about 4, in every 756-day window that contains it (June 2020 to April 2023). That closes the 5–120-day gate there. Without the week, the gate is open in 90% of months for Brent–WTI, against 71% with it. The core run keeps the week, as pre-registered, and the pre-registered `drop_flagged_dates` robustness run shows the effect. Futures avoid it, because the roll leaves the June contract held on that day.
- **EIA spot timing.** EIA posts spot prices weekly. A live trader would see the same prices daily from a price reporting agency, so the backtest treats them as known at that day's close.

## Code review (2026-10-24)
A line-by-line review with independent re-computations found and fixed three timing errors. Each fix makes the code match a registered rule or principle; no registered value changed.
1. **Blackout one day early.** It was applied to the decision day, which blocked fills after the event and allowed fills the day before it. It now blocks the fill.
2. **OVX** used the same day's close, which comes after the NYMEX settle. Now lagged one day.
3. **COT** was stamped on Friday, but the release comes after the settle. Now stamped on Monday.

**Effect:** gated HO−RB goes from Sharpe 0.28 on 20 trades to 0.25 on 19, and now fails G2 and G3 as well. The crack and Brent–WTI are unchanged. Pooled AUC goes from 0.50 to 0.51. Every verdict is still FAIL.

**Verified unchanged:**
- **Fits and filter:** the fit statistics and GARCH forecasts (against `arch`), the episode labels and the logit training sets all match independent recomputations.
- **Trades:** every entry and exit condition holds, and an independent replay of the entry logic matches the engine exactly.
- **P&L** reconciles.

## Status
- **2026-10-24:** Part 0 is done (pre-registration, committed and tagged `prereg-2026-10-24`).
- **2026-10-24:** Part 1 is done on spot data. The Bloomberg loader is built and tested on synthetic curves; the export is still pending.
- **2026-10-24:** Parts 2–3 are done on spot data. Main findings:
  - very fat tails (ν ≈ 2–2.5 even without April 2020)
  - significant seasonality in the crack and HO−RB but not in Brent–WTI, matching the pre-registered settings
  - rolling ADF pass rates of 35–46%, with the plan's 2011–13, 2022 and 2026 breaks visible
  - SARIMA does not beat a random walk
  - an unstable rolling Johansen rank (most often 1, not the expected 2)
- **2026-10-24:** Part 4 is done on spot data.
  - **Entry gate:** open in 85% / 71% / 100% of months (crack / Brent–WTI / HO−RB).
  - **Volatility forecast:** GARCH-t beats the constant window sd out of sample for every spread, significantly for HO−RB.
  - **VECM:** the product legs do most of the correcting of the crack, and Brent does most of the correcting of Brent–WTI.
  - **Johansen variant:** usable in only 58% of months.
  - **Today:** the crack and HO−RB GARCH σ̂ are in the 99th percentile of their history, and HO−RB z ≈ 6.2, past the stop.
- **2026-10-24:** Part 5 is done on spot data. **The pre-registered filter has no out-of-sample skill.**
  - **Episodes:** 115 for the fixed spreads, with low base rates (24% crack, 47% Brent–WTI and HO−RB). The 20-day label is short next to half-lives of 25–35 days.
  - **Skill:** pooled AUC is 0.51, and the Brier score is worse than the base rate. Calibration shows no relationship.
  - **What passes:** P ≥ 0.6 on only 4% of days for the crack, 3% for Brent–WTI and 16% for HO−RB. Only 9 of the 92 episodes with a P pass.
  - **Stress years:** the filter blocks every crack and Brent–WTI episode in 2020, 2022 and 2026, but it is close to always off.
  - **Robustness:** probit, the robustness features and the 10/40-day labels give the same verdict.
- **2026-10-24:** Part 6 is done. **Spot-proxy verdict: all three spreads FAIL G1–G5.** This is not binding; the binding run needs futures.
  - **Crack:** Sharpe −0.02 on 6 trades; fails G1–G4.
  - **Brent–WTI:** Sharpe 0.13 on 2 trades; fails G1, G2 and G4.
  - **HO−RB:** Sharpe 0.25 on 19 trades; passes only G4, and fails G1, G2, G3 and G5 (the filter kept it out of a profitable 2026).
  - **What the filter does:** it mostly removes trades. It kept the crack and Brent–WTI flat through 2022 and 2026, avoiding −\$86k to −\$50k losses.
  - **Best signal:** HO−RB without the filter (Sharpe 0.27, t = 1.1), but that is exploratory.
  - **Integer book:** \$2m → \$2.04m (gated) or \$2.13m (no logit, half size since April 2022).
- **2026-10-24:** Part 7 is done. **No spread meets its switch-on triggers.** On spot data to 20 Oct:
  - **Crack:** z 1.8, P 0.14, σ̂ in the 99th percentile; it would size to 0 contracts.
  - **HO−RB:** z 6.2, past the stop; the diesel crack is \$113 with σ̂ in the 99th percentile.
  - **Brent–WTI:** z 0.6, and its trigger needs futures.
  - **Hormuz:** the last logged closure was 27 Sep, so the 4-week condition clears on 25 Oct.
- **2026-10-24:** Part 10 is done: the written report (`report/report.pdf`, `.docx`), labelled throughout as a spot proxy. `check_report.py` confirms all 92 quoted numbers match the outputs.
  - Open: the author line is still a placeholder.
  - Open: the binding futures rerun, once the Bloomberg export arrives.

## Note
This is an educational research project, not investment advice. Paper trading only. Check your employer's personal-dealing policy before trading futures in a personal account.
