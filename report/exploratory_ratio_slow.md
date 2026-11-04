# Slow ratio mean reversion: matching the horizon where the reversion shows up (exploratory)

Rules: `PREREGISTRATION_RATIO_SLOW.md`, registered 2026-11-04 (commit `f49eeb4`) before any monthly-decision signal or P&L. Code: `src/oilrv/ratio.py` (`decision_months`), `scripts/run_ratio_slow.py`. Tables and figures: `output/exploratory/ratio_slow/`. Run once, every spread, grid point and comparison reported. It changes no binding verdict.

**The change from the daily ratio strategy is one thing:** entries, take-profits and stops are decided on the first trading day of a month (instead of every day). Everything else, including the 3-half-life time stop, is as registered.

**Regression check.** The same code with decisions every day reproduces the registered ratio result exactly: book Sharpe 0.055 and 71 round trips.

## Result: slowing down made it worse

2010-02-01 to 2026-10-26, futures, net of costs, R = $6,000 a day per spread, equal-risk average of three spreads.

| Decisions | Book net Sharpe | Net P&L | Round trips | Crack | Brent/WTI | HO/RB |
|---|---|---|---|---|---|---|
| Every day (reference) | 0.055 | +$21k | 71 | 0.22 | −0.49 | 0.38 |
| **Every month (base)** | **−0.131** (t = −0.54) | −$40k | 22 | 0.19 | −0.12 | −0.27 |
| Every three months | −0.192 | −$39k | 8 | −0.36 | −0.26 | 0.30 |

| Gate | Book (monthly) | Result |
|---|---|---|
| G1 net Sharpe > 0.5 | −0.131 | ✗ |
| G2 round trips ≥ 20 | 22 | ✓ |
| G3 positive sub-periods ≥ 2 of 3 | 2007–13 −$24k, 2014–19 +$14k, 2020–26 −$31k | ✗ |
| G4 Sharpe > 0 at entry \|z\| 1.5 / 2.0 / 2.5 | +0.26 / −0.13 / +0.05 | ✗ |
| G5 net P&L > −$40k in 2022 and 2026 | 2022 $0, **2026 −$61k** | ✗ |

By spread (each at full R):

| Spread | Sharpe (t) | Round trips | Average hold, days (daily → monthly) | Verdict |
|---|---|---|---|---|
| 3-2-1 crack | 0.19 (0.79) | 11 | 31 → 52 | fails G1 and G2 |
| Brent / WTI | −0.12 (−0.50) | 2 | 23 → 106 | fails G1 to G4 (2 trades, nothing to judge) |
| HO / RB | −0.27 (−1.09) | 9 | 26 → 36 | fails all five gates; 2026 −$171k |

- HO/RB, the best spread in the daily version (0.38), is the worst here. One trade lost $147k: a short entered on 3 February 2026 and closed by the time stop on 26 March.
- **No spread passes, and none counts as evidence.** The book does not pass the gates, so the stricter bar of t > 3.0 is not reached.

## Why it did not help

- **Exits: 14 of 22 trades ended on the time stop, 4 on stop-loss, 3 on take-profit and 1 on a model break.**
  - Time stops lost $125k in total, stop-losses lost $154k, and the three take-profits made $132k.
  - In the daily version 32 of 71 trades reached take-profit. Here 3 of 22 did.
- **The time stop caps the hold at about 1 to 2 months, not 6 to 12.** The registered time stop is 3 × the half-life fitted at entry, and that fitted half-life is only 12.7 days for HO/RB, 21.6 for the crack and 48.7 for Brent/WTI (the full-sample half-lives are 35 to 50 days). So average holds were 36, 52 and 106 days. **This version matches the horizon only in part.** It is a test of "decide monthly, hold up to a few months", not of "hold for a year".
- **Fewer trades, more variance per trade.** 22 round trips (11, 2 and 9 per spread) cannot separate an edge from luck. Quarterly decisions give 8.

## Comparisons and robustness (not gated)

| | Book Sharpe | Round trips |
|---|---|---|
| Log-ratio signal, monthly (base) | −0.131 | 22 |
| Dollar gap on market prices, monthly | −0.054 | 29 |
| Entry \|z\| 1.5 / 2.0 / 2.5 | 0.26 / −0.13 / 0.05 | 42 / 22 / 16 |
| Exit \|z\| 0.25 / 0.5 / 0.75 | −0.13 / −0.13 / −0.05 | 22 / 22 / 23 |
| Window 504 / 756 / 1,008 days | −0.21 / −0.13 / +0.07 | 40 / 22 / 20 |
| Cost $0 / $0.05 / $0.20 per bbl | −0.12 / −0.13 / −0.18 | 22 |

- The sign flips with the entry threshold and the window, which is what noise looks like. The positive cells (entry 1.5, window 1,008) are reported and **not** chosen.
- Costs are not the cause (−0.12 at zero cost).
- The log-ratio is no better than the dollar gap here either.
- Daily P&L correlation: +0.52 with the daily ratio strategy, +0.21 with the registered mean reversion, +0.04 with pairs, **−0.20** with the trend and carry book.

## Should we take this step further? Why not

The question is whether to go on from here (for example to register a version that holds up to a year). My assessment is no, for these reasons.

1. **The evidence for slow reversion is a point estimate.** No log-ratio variance ratio is significant (robust p from 0.15 to 0.98), and the forward-return slopes were seen before the rules were written. A test built to chase that horizon starts from an unconfirmed premise.
2. **The sample cannot settle a 6–12 month effect.** In 16.7 years there are about 17 non-overlapping one-year windows per spread, and the three spreads are related. At the best Sharpe seen anywhere (HO/RB daily, 0.38) it takes about 28 years to reach t = 2 and 62 years to reach t = 3. For the slow crack (0.19) it is 106 and 239 years.
3. **Forking paths.** This is the sixth strategy variant tried on one sample (mean reversion, pairs, trend, carry, ratio mean reversion, and this one). A seventh version tuned to the horizon would be fitted to what we already saw. It could not be told apart from luck, and the bar would have to be higher, not lower.
4. **Regime breaks grow with the holding period.** The anchor k moves between eras. The longer the hold, the more often a trade straddles a break, and the two biggest losses here (HO/RB in 2026, Brent/WTI in 2011) both came from that.
5. **Why it was not tried first.** On back-adjusted dollar changes the variance ratio was about 1 at every horizon for the crack and Brent/WTI, so there was no slow reversion to match. The log-ratio profile only appeared after the professor's feedback on 2026-11-04.
6. **What would change the answer** is more independent bets, not more tuning: more spreads or markets (to raise power), or forward tracking of the fixed rules on new data. Neither can be done on this sample.

## Reading

1. **Matching the horizon did not rescue mean reversion on this sample.** Decisions every month turn +0.06 into −0.13, with a book that no longer passes G3 either.
2. **Part of the horizon argument is untested.** The time stop kept holds short, so a full 6–12 month hold remains unexamined. Examining it would be a seventh look at the same data, for the reasons above.
3. **The binding conclusion does not change.** Mean reversion, in dollars or ratios, daily or monthly, does not pass. Trend and carry remains the only strategy with four of five gates passed.

## Limits

- Monthly decisions give 22 trades, so t-statistics are wide.
- The time stop (3 half-lives) was kept as registered and caps holds, as described above.
- The GARCH fit printed convergence warnings on some windows. The engine falls back to the window standard deviation when a fit is not usable.
- Equal-risk averaging of three spreads makes G5 lenient, so each spread is also shown at full R.
