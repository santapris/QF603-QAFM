# Exploratory: monthly trend and carry

**Status: exploratory, a seventh look.** Rules: `PREREGISTRATION_TREND_CARRY_MONTHLY.md`, registered 2026-11-05 after the daily result was seen. Nothing here changes the registered verdict. Tables: `output/exploratory/trend_carry_monthly/tables/`.

**Result: it fails G1, like the daily book, with a lower Sharpe.** The monthly book has a net Sharpe of 0.37 (t = 1.52) and passes G2 to G5. It is not called evidence of anything.

| 50/50 book, 2010–26 | Registered daily | Daily, re-run here | Monthly (base) | Monthly signal, daily size |
|---|---|---|---|---|
| Net Sharpe | 0.45 | 0.44 | **0.37** | 0.36 |
| t-statistic | 1.83 | 1.80 | **1.52** | 1.49 |
| Trend / carry Sharpe | 0.29 / 0.44 | 0.27 / 0.45 | **0.25 / 0.37** | 0.17 / 0.41 |
| Net P&L | $460k | $452k | **$408k** | $366k |
| Max drawdown | −$209k | −$215k | **−$167k** | −$161k |
| Position changes | 1,150 | 1,150 | **268** | 268 |
| Costs, % of gross (trend / carry) | 39% / 10% | 40% / 10% | **18% / 4%** | 39% / 7% |
| Share of profit from 2026 | 48% | 49% | **57%** | 55% |

## Reading
- **Costs fell, but gross fell too.** Trend's net P&L is about the same, because the lower costs offset the lower gross. Carry's net P&L falls.
- **The gap is within noise.** The monthly book is 0.07 below the daily re-run. With a daily-P&L correlation of 0.905, that is about 0.6 standard errors of the difference over 16.7 years (a rough estimate).
- **No setting reaches the bar.** The best is 0.44, at a 504-day lookback; the daily book reached 0.52 there.
- **Daily sizing does not help.** Monthly signal with daily sizing gives 0.36.
- **Oil exposure.** Trend's alpha t-statistic is 0.08 and the book's is 0.86 (`oil_beta.csv`).
- **All nine positions are long at the 26 October close** (`current_book.csv`).

## Disclosures
- **The reference check was amended before any monthly result was seen** (section 6 of the pre-registration). Daily gives 0.4396 on this machine against the committed 0.4471, with identical signals. The likely cause is GARCH fits differing between machines.
- **A second run added descriptive tables only** (`oil_beta.csv`, `current_book.csv`). `comparison.csv` was byte-identical afterwards.
- **Another monthly run exists.** A group member reported "still 0.4 something" from a script not seen here. Neither variant above matches it, so its definition of monthly must differ.
