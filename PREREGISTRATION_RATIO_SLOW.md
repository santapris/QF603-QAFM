# Pre-registration addendum: slow ratio mean reversion (exploratory)

**Registered 2026-11-04**, after the registered ratio mean reversion failed its gates, and **before any monthly-decision signal or P&L was computed**. The machine-readable values are in `config.yaml` → `exploratory_ratio_slow`. `tests/test_preregistration.py` fails if they change.

**Status: exploratory, and a second look.** This strategy was chosen after seeing the first one fail. It does not change the pre-registered verdict, and it is reported next to it, never in place of it.

## 1. Why this is being tried, and what was known
- **The motivation.** The variance ratio of the held log-ratio return falls below 1 only at long horizons:

  | Horizon | Crack | Brent / WTI | HO / RB |
  |---|---|---|---|
  | 21 days | 0.84 | 0.73 | 1.11 |
  | 63 days | 0.66 | 0.63 | 0.77 |
  | 252 days | 0.50 | 0.41 | 0.48 |

  The registered ratio strategy held positions for 23–31 days on average, where the ratios are close to random walks. Reversion shows up at 3–12 months.
- **The first strategy's result** (`report/exploratory_ratio_mr.md`): book net Sharpe 0.055 with 71 round trips. Of those, 32 reached take-profit, 19 were stopped out and 19 hit the time stop. Average holds were 31, 23 and 26 days.
- **The forward-return slopes** are larger at 63 days than at 21 days for every spread (HO/RB −0.030 against −0.016, crack −0.011 against −0.008, Brent/WTI −0.005 against −0.002), which points the same way.
- **Why it was not tried first.** The variance ratio on back-adjusted dollar changes was about 1 at every horizon for the crack and Brent/WTI, so there was no evidence of a slow reversion to match. Only the log-ratio variance ratios, computed after the professor's feedback on 2026-11-04, show one.

**Prior: failure is the likely outcome.**
- No log-ratio variance ratio is significant (robust p from 0.15 to 0.98), so the horizon itself is a point estimate.
- The anchor k moves between eras, so a longer hold meets more regime breaks, and Brent/WTI broke in 2026.
- Monthly decisions give fewer trades, so G2 may fail and t-statistics will be wide.
- **Forking paths.** This is the sixth strategy variant tried on one 16.7-year sample (mean reversion, pairs, trend, carry, ratio mean reversion, and this one). Each look raises the chance that one passes by luck.

## 2. Strategy
The registered ratio strategy (`PREREGISTRATION_RATIO.md`) with **one change**: when decisions are made.

- **Spreads, signal, fit, weights, gate, sizing, costs, rolls, blackout:** exactly as in `PREREGISTRATION_RATIO.md`.
- **Decision days.** Entries, take-profits and stop-losses are decided on the **first trading day of every month** (`decision_months: 1`). On every other day z is not read, so nothing is entered, exited or stopped. A decision is filled one close later, as registered.
- **Unchanged and still daily:** the time stop (3 half-lives), and the monthly re-test of the half-life gate (`model_break`), which already runs at month starts. Weights and the fair value are frozen while in a trade.
- **Re-arm:** |z| must be seen inside 2.0 on a decision day after an exit before the next entry.
- **Reference.** The same code with decisions on every day (`reference_decision: daily`) is re-run as a regression check. It must reproduce the registered ratio result (book Sharpe 0.055, 71 round trips).

## 3. Gates (on the book)
The same five gates as the registered ratio strategy, with the same thresholds (G1 Sharpe > 0.5, G2 round trips ≥ 20 summed over spreads, G3 at least 2 of 3 positive sub-periods, G4 Sharpe > 0 at entry |z| ∈ {1.5, 2.0, 2.5}, G5 net P&L above −$40,000 in 2022 and in 2026). Each spread is also reported at full R = $6,000, and counts as evidence only if it passes every gate and clears t > 2.39 (three tests).

**Calling a pass.** Because this is the sixth variant tried on the sample, passing the gates is **not described as evidence** unless the book's t-statistic is above 3.0 (the convention for new strategies after many looks; Harvey, Liu and Zhu, 2016). Otherwise it is described as "passes the gates after multiple looks, needs forward tracking".

## 4. Reported, not gated
- **Decision interval:** every day (the reference), every month (base), every three months (January, April, July, October).
- **Comparison:** `dollar_market`, with z on the market dollar spread, same monthly decisions.
- **Exit z** 0.25 and 0.75, **estimation window** 504 and 1,008 days, costs of $0, $0.025, $0.10 and $0.20/bbl.
- **Hold days, exit reasons, long and short P&L**, per spread.
- **Daily-P&L correlation** with the registered mean reversion, the daily ratio strategy, pairs, and the trend and carry book.
- **The sample needed for significance:** the number of years at the observed Sharpe for t = 2 and t = 3.

## 5. Reporting rules
- Run once, on the futures data, with these values. Every spread, grid point and comparison is reported.
- No parameter may change after a result is seen. Any later change is a new exploratory run with its own date.
- Fixing a coding error that makes the code match these rules is allowed, and must be disclosed in the findings with its effect on the results.
