# Equity Variance Risk Premium (QF603 project)

A research project for SMU QF603 (Quantitative Analysis of Financial Markets). It studies the
**variance risk premium (VRP)** of the S&P 500: the gap between the variance that option prices imply
and the variance investors should expect to be realised. It then asks whether that gap can be traded
economically.

## The idea in one paragraph

Option prices contain more than a volatility forecast. They also include a premium paid to whoever
bears volatility risk. We measure that premium each month-end as

```
VRP_t = implied variance (next 30 days, from SPX options) − expected realised variance (next 21 trading days)
```

and ask what moves it, whether those relationships are stable and survive out-of-sample tests, and
whether selling variance only when the premium is worth its risk beats selling it all the time.

## Research questions

| | Question |
|---|---|
| Q1 | **Persistence:** does VRP predict its own value 1, 3 and 6 months ahead? |
| Q2 | **Drivers:** do VIX, the VIX term structure, VVIX, SKEW, funding and credit spreads, dealer capital (HKM) and VIX-futures positioning (CFTC) add information beyond lagged VRP? |
| Q3 | **Stability:** do those relationships change in high-volatility periods or over time? |
| Q4 | **Robustness:** do results hold for the ex-post payoff definition and in genuine out-of-sample tests? |
| Q5 | **Returns:** does VRP predict S&P 500 excess returns (Bollerslev, Tauchen & Zhou 2009)? |
| Q6 | **Economic significance:** does a cost-aware short-variance strategy timed by VRP beat an always-short benchmark? |

## How VRP is measured

| Piece | Method | Data |
|---|---|---|
| Implied variance | 30-day model-free implied variance, Cboe VIX methodology | SPX options, WRDS OptionMetrics |
| Realised variance | Sum of squared 5-minute SPY returns plus the squared overnight return | WRDS TAQ |
| Expected realised variance | Recursive HAR forecast, re-fitted every month-end on past data only (headline: log HAR) | from realised variance |
| Ex-ante VRP (headline) | implied − expected realised | known at month-end t |
| Ex-post VRP | implied − actually realised (the payoff of selling a variance swap) | a target only, never a predictor |
| Trailing VRP | VIX² − last month's realised variance | Q5 only |

Sample: January 2008 – August 2025, monthly. It is split into train (2008–2015), validation
(2016–2020) and a holdout (2021 – Aug 2025) that is used **once**, after every choice is frozen.

## The trading strategy (Q6)

Each month: short a one-month at-the-money SPX straddle, delta-hedged daily, with a long ~10-delta put
to cap crash losses, held to cash settlement. Three versions share everything except timing:

- **S0** always short (benchmark)
- **S1** sized by today's ex-ante VRP
- **S2** sized by the model's payoff forecast (kept only if it beats S1 after costs on validation)

"Economical" means: trade only when the expected payoff exceeds 1.5× expected costs; size each position
so a −15% SPX / doubled-volatility shock costs 20% of capital; pay no exit spread; and cost everything
with real bid/ask spreads plus fees.

## Rules the project follows

1. **No look-ahead.** Every number dated t uses only what was public at t (publication lags included).
2. **One holdout run.** Nothing after 2020 is used for modelling until the single, approved final run.
3. **No fabricated or silently substituted data.** Missing inputs stop the pipeline.
4. **Every convention is logged** in `DECISIONS.md` with its reason and the alternatives considered.
5. **One source of truth:** all dates and parameters live in `config.yaml`.

## Where things are

| Path | What it is |
|---|---|
| `PLAN.md` | The working plan, phase by phase, with checkboxes and gates |
| `DECISIONS.md` | Log of every convention and choice (D001 onward) |
| `DATA_INVENTORY.md` | Each data series: source, coverage, gaps, lags |
| `config.yaml` | All dates, horizons, lags and strategy parameters |
| `run_all.py` | Single entry point for the pipeline (`--dry-run` shows status) |
| `src/data/` | Data pulls and loaders (OptionMetrics, TAQ, FRED, CFTC, Bloomberg, HKM) |
| `src/measures/` | Realised variance, implied variance, HAR forecasts, the monthly panel |
| `src/models/` | Stages 1–5: persistence, drivers, stability, out-of-sample, returns |
| `src/strategy/` | Contracts, costs, backtest, signals, performance metrics |
| `src/final/` | Freeze, the single holdout run, reproducibility check |
| `src/report/` | Builds `report/report.md` from the outputs |
| `tests/` | Unit tests, including look-ahead and holdout-guard tests |
| `data/`, `outputs/` | Data and results (not in git: WRDS and Bloomberg data are licensed) |

## Status (6 October 2026)

**Done**
- Data pulled and checked: SPX options, SPY 5-minute bars, FRED rates, CFTC positioning.
- Implied variance built: correlation with VIX² is 0.9986.
- Realised variance built: correlation with squared daily SPX returns is 0.982.
- HAR variants evaluated; the log HAR was chosen as the headline.
- Strategy contracts selected (210 monthly cycles) and their daily prices pulled.
- Code written and tested for Stages 1–5, the strategy, the freeze/holdout tools and the report builder.

**Waiting on manual inputs**
- Bloomberg exports into `data/raw/bloomberg/`, one CSV per ticker with columns `date,px_last`:
  `VIX_Index.csv`, `VIX3M_Index.csv`, `VVIX_Index.csv`, `SKEW_Index.csv`, `SPXT_Index.csv`,
  `PUT_Index.csv`, `BXM_Index.csv`.
- The He–Kelly–Manela monthly capital-ratio CSV into `data/raw/hkm/`.

**Then**
1. Build the final panel and run Stages 1–5 on 2008–2020.
2. Run the strategy on the validation window; choose what to freeze.
3. Freeze, approve, run the holdout once.
4. Finish the report and slides; run the full reproducibility check.

## Running it

```bash
python run_all.py --dry-run          # what is done, pending or not yet possible
python run_all.py --phase 1 2 3 4    # build the panel and run Stages 1–5 (once the inputs above exist)
python -m pytest -q                  # run the tests
python -m src.report.build           # rebuild the report draft
```

Requires Python 3.11+ and the packages pinned in `requirements.txt`, plus WRDS access for the data pulls.

## Key references

Bollerslev, Tauchen & Zhou (2009); Bekaert & Hoerova (2014); Carr & Wu (2009); Corsi (2009);
Britten-Jones & Neuberger (2000); Hansen & Lunde (2005); He, Kelly & Manela (2017);
Goyal, Welch & Zafirov (2024). The full list is in `PLAN.md`.
