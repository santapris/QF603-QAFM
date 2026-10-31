# Pre-registration addendum: cointegration pairs (exploratory)

**Registered 2026-10-31**, after the binding futures verdict on the mean-reversion strategy (all three spreads FAIL G1–G5) and the exploratory trend and carry result, and **before any signal or P&L of this strategy was computed**. The machine-readable values are in `config.yaml` → `exploratory_pairs`. `tests/test_preregistration.py` fails if they change.

**Status: exploratory.** This strategy was chosen after those results were seen. It does not change the pre-registered verdict, and it is reported next to it, never in place of it.

## 1. What was known at registration
- **The binding mean-reversion results and the trend and carry results**, including their sub-periods and stress years. The registered signal's back-adjustment drift left the crack and Brent–WTI books almost entirely short.
- **A rough check on the fixed-weight spreads (2026-10-31, not in the repo).** Simplified mean-reversion rules gave these Sharpes:

  | Signal level | Crack | Brent–WTI | HO−RB |
  |---|---|---|---|
  | Back-adjusted | 0.09 | −0.68 | −0.03 |
  | Market | 0.06 | −0.42 | 0.08 |

- **Variance ratios of tradable daily changes**, near 1 for the crack and 0.86–0.87 for Brent–WTI and HO−RB at 250 days.
- **Formation statistics for the six pairs below (2026-10-31).** No z-score, entry, exit or P&L was computed. At each month start, on the prior 756 days:

  | Pair (y on x) | Median β | Gate open (EG p < 0.10, with trend) | Median half-life, days |
  |---|---|---|---|
  | Brent / CL | 1.02 | 31% | 37 |
  | HO / RB | 0.95 | 17% | 24 |
  | RB / CL | 0.95 | 27% | 23 |
  | HO / CL | 1.06 | 10% | 22 |
  | RB / Brent | 0.98 | 26% | 23 |
  | HO / Brent | 1.07 | 21% | 19 |

  - On roll days, market-level jumps in the 1:1 product pairs have a standard deviation of \$1.6–3.8/bbl, against \$0.8–1.6 for a normal daily change.
  - The half-life gate is in band in almost every month, so the Engle–Granger test is the effective gate.
  - Gate rates are not far above the 10% expected by chance. All six are near zero in 2024.
  - RB/CL and RB/Brent were open in 100% of 2021–22 months, so the gate is not expected to keep the book out of 2022.
  - Fitted carry reaches \$18–33/bbl a year at the 95th percentile on Brent/CL, RB/CL and HO/CL, so extrapolating the trend can misplace fair value.

**Not independent of those results.** Brent/CL and HO/RB use the same legs as two registered spreads, so they are not a fresh test. Only RB/CL, HO/CL, RB/Brent and HO/Brent are new.

**Prior: failure is the likely outcome.**

The choices below come from the cointegration approach to pairs trading (Engle & Granger 1987; Vidyamurthy 2004) and the registered rules, not from a search over this data.

## 2. Strategy
**Universe.** Six pairs, fixed in advance, in y-on-x order: `brent/cl`, `ho/rb`, `rb/cl`, `ho/cl`, `rb/brent`, `ho/brent`. Both legs of a pair hold the same delivery month and roll 5 business days before the earlier expiry (`futures.continuous`). All series sit on the common futures dates.

**Monthly fit**, on the 756 days strictly before the first trading day of the month, and whenever flat after an exit, on back-adjusted \$/bbl levels:
1. OLS of y on a constant, x and a linear trend. β is the slope on x and c is the trend per trading day. The trend absorbs average roll carry, and back-adjusted changes are what a position earns, so all modelled reversion comes from tradable price changes.
2. Weights are scaled to gross 2 contracts per unit: w_y = 2 / (1 + |β|), w_x = −2β / (1 + |β|). At M = \$1,000 per \$1/bbl per unit, the \$0.05/bbl cost is \$25 per contract, the same as the registered spreads.
3. S = w_y·y + w_x·x. The residual R = S − c_S·t − month effect, where c_S is the trend of S on t in the window (equal to w_y·c) and t counts trading days from the window start. Month-of-year effects are on for the five product pairs and off for `brent/cl`, as in the registered rules.
4. z_t = (R_t − mean) / sd, with mean and sd from the window. The trend keeps extrapolating until the next refit.

**Gate.** Engle–Granger `coint(y, x, trend="ct", autolag="aic")` p < 0.10, and the AR(1) half-life of R in [5, 120] days.

**Trading.** The registered rules, unchanged:
- Enter at 2.0 ≤ |z| < 4.0 when re-armed and the gate is open. Sell the spread when z > 0, buy it when z < 0.
- Exit at |z| ≤ 0.5. Stop at |z| ≥ 4.0. Time stop at 3 half-lives. Exit (`model_break`) if the gate is closed at a monthly re-test.
- Weights and fair value are frozen while in a trade.
- Decide at close t, fill at close t+1. \$0.05/bbl per unit of position change, plus 1 tick per contract leg per roll.
- Units = R / (σ̂ × M), unrounded, with R = \$6,000 a day and σ̂ the GARCH(1,1)-t forecast on the fit's frozen-weight daily changes.
- No entry within the registered OPEC and hurricane blackout. No logit filter, because it never converges on futures.

**Book.** Equal-risk average of the six pairs' daily research P&L, as with the trend and carry sleeves. Research stream only: fractional hedge ratios cannot be held in 1–3 lots, so no integer book is run.

**Sample.** Trading starts at the first month start with a full 756-day window, as in the registered backtest.

## 3. Gates (on the book)
| # | Gate | Threshold |
|---|---|---|
| G1 | Net annualised Sharpe | > 0.5 |
| G2 | Round trips, summed over pairs | ≥ 20 |
| G3 | Sub-periods with positive net P&L (2007–13, 2014–19, 2020–present) | ≥ 2 of 3 |
| G4 | Net Sharpe > 0 at every entry \|z\| ∈ {1.5, 2.0, 2.5} (EG p at base) and every EG p threshold ∈ {0.05, 0.10} (entry at base) | all |
| G5 | Net P&L in 2022 and in 2026 to date | > −\$40,000 in each |

Averaging six pairs makes G5 lenient, so each pair is also reported against G1–G5 at its full R = \$6,000.

**One pair passing is not a result on its own.** With six pairs, one is counted as evidence only if it passes G1–G5, clears the Bonferroni bar of t > 2.64 (two-sided 5% over six tests; Sharpe ≈ 0.65 over 16.7 years), and is one of the four new pairs.

## 4. Reported, not gated
- **Signal comparisons**, same rules otherwise:
  - `adjusted`: no trend term, EG `trend="c"`
  - `market`: fit and z on unadjusted levels, P&L on adjusted changes, EG `trend="c"`
  - `no_gate`: base signal with no Engle–Granger gate
- **Estimation window** 504 and 1,008 days, and costs of \$0, \$0.025, \$0.10 and \$0.20/bbl.
- **Long and short P&L, and exit reasons**, per pair.
- **Carry split per trade:** carry = side × units × M × c_S × days held, and convergence = gross P&L − carry.
- **Daily-P&L correlation** with the registered mean-reversion stream and the trend and carry book.
- **Latest state per pair:** z, gate and the contracts per leg implied by a target position.

## 5. Reporting rules
- Run once, on the futures data, with these values. Every pair, grid point and comparison is reported.
- No parameter may change after a result is seen. Any later change is a new exploratory run with its own date.
- Fixing a coding error that makes the code match these rules is allowed, and must be disclosed in the findings with its effect on the results.
