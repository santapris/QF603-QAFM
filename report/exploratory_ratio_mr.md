# Ratio mean reversion on the futures (exploratory)

Rules: `PREREGISTRATION_RATIO.md`, registered 2026-11-04 (commit `bd725b7`) before any signal or P&L of this strategy. Code: `src/oilrv/ratio.py`, `scripts/run_ratio_mr.py`. Tables and figures: `output/exploratory/ratio_mr/`. The run was made once, and every spread, grid point and comparison is reported. It changes no binding verdict.

**Status of the evidence.** The forward-return regressions that motivated the rules had been seen first (HO/RB t = −2.8 and −3.5), so a pass would have meant "passes the gates, not independently confirmed". It did not pass.

## Result: the book fails

2010-02-01 to 2026-10-26, futures, net of costs, research stream at R = $6,000 a day per spread, equal-risk average of the three spreads.

| Gate | Book | Result |
|---|---|---|
| G1 net Sharpe > 0.5 | **0.055** (t = 0.23), total +$21k, max drawdown −$140k | ✗ |
| G2 round trips ≥ 20 | 71 | ✓ |
| G3 positive sub-periods ≥ 2 of 3 | 2007–13 +$17k, 2014–19 +$45k, 2020–26 −$40k | ✓ |
| G4 Sharpe > 0 at entry \|z\| 1.5, 2.0, 2.5 | 0.13, 0.055, 0.22 | ✓ |
| G5 net P&L > −$40k in 2022 and 2026 | 2022 −$4k, **2026 −$133k** | ✗ |

### By spread, each at full R

| Spread | G1 Sharpe (t) | Round trips | G3 eras | G4 | G5: 2022 / 2026 | Net P&L | Counts as evidence? |
|---|---|---|---|---|---|---|---|
| 3-2-1 crack | 0.22 (0.89) ✗ | 29 ✓ | 3 ✓ | ✓ | −$13k / −$34k ✓ | +$148k | No |
| Brent / WTI | −0.49 (−1.99) ✗ | 15 ✗ | 1 ✗ | ✗ | $0 / **−$348k** ✗ | −$334k | No |
| HO / RB | 0.38 (1.55) ✗ | 27 ✓ | 3 ✓ | ✓ | +$2k / −$17k ✓ | +$250k | No (t < 2.39) |

- **HO/RB and the crack fail on G1 alone.** HO/RB is the best of the three and positive in all three eras (Sharpe 1.13, 0.21, 0.41), but its t-statistic of 1.55 is well short of the Bonferroni bar of 2.39.
- **Brent/WTI fails every gate.** Its 2026 loss is two stop-outs. The first was a short entered 16 January at z = 2.1 with 65 units (the GARCH forecast was very small before the shock, so the risk-parity size was large), stopped on 3 March for −$190k. The second was entered on 11 September with 19 units and stopped on 2 October for −$129k. This is the same weak spot as the registered dollar strategy, whose largest loss was also in Brent − WTI (−$151k in 2022).
- **The gate does not select.** It was open on 99%, 90% and 100% of days. As the pre-registration notes for pairs, a half-life band does not separate a reverting series from a random walk.

## Does the object matter? Comparisons (not gated)

| | Book Sharpe | Net P&L | Round trips | Crack | Brent/WTI | HO/RB |
|---|---|---|---|---|---|---|
| **Log-ratio signal (base)** | 0.055 | +$21k | 71 | 0.22 | −0.49 | 0.38 |
| Dollar gap on market prices | −0.053 | −$24k | 92 | 0.25 | −0.53 | 0.20 |
| No half-life gate | 0.087 | +$36k | 74 | 0.25 | −0.40 | 0.38 |
| Registered dollar strategy (back-adjusted, from the deck) | n/a | n/a | n/a | 0.32 | −0.42 | 0.06 |

- **Moving from the dollar gap to the log-ratio did not rescue mean reversion.** On market prices the book goes from −0.05 to +0.06. HO/RB improves (0.20 → 0.38, and 0.06 under the registered signal), the crack and Brent/WTI do not.
- So the professor's point is right about *stationarity*, and it explains why the registered back-adjusted dollar signal behaves oddly, but it does not make the strategy pass.
- `no_gate` is implemented by clipping the half-life into the registered band so the gate never closes (the time stop then uses the clipped value).

## Robustness (not gated)

| Setting | Book Sharpe | Round trips |
|---|---|---|
| Cost $0 / $0.025 / **$0.05** / $0.10 / $0.20 per bbl | 0.10 / 0.08 / **0.055** / 0.01 / −0.09 | 71 |
| Exit \|z\| 0.25 / **0.5** / 0.75 | −0.02 / **0.055** / 0.05 | 66 / 71 / 75 |
| Window 504 / **756** / 1,008 days | −0.06 / **0.055** / 0.24 | 112 / 71 / 61 |

- **Costs are not the cause.** At zero cost the Sharpe is only 0.10.
- The sign flips with the estimation window, so the base Sharpe is not stable. The 1,008-day window starts a year later (February 2011), and it is reported, not chosen.

## What drove it

- **Roll jumps in the signal are not the explanation.** Gross P&L earned on roll days is $16k of $168k (crack), +$25k of −$302k (Brent/WTI) and −$8k of $270k (HO/RB): −8% to +10% of gross.
- **Direction.** Brent/WTI shorts lost $392k over 14 trades (one long trade made $58k). HO/RB shorts made $186k over 19 trades and longs $65k over 9. The crack's longs made $126k over 7 trades at an 86% win rate, its shorts $22k over 22.
- **Exits.** Stop-losses: 4 of 29 (crack), 7 of 15 (Brent/WTI), 8 of 27 (HO/RB). Take-profits: 14, 5 and 13. Time stops: 11, 2 and 6.
- **Bad years.** 2015 (−$78k: crack −$143k, HO/RB −$112k), 2011 (−$34k) and 2026 (−$133k).
- **Correlation of daily P&L:** with the registered dollar mean reversion +0.21, with pairs +0.08, and with the trend and carry book **−0.18**.

## Reading

1. **Stationary is not the same as profitable.** The log-ratio has a home (half-lives of 35–50 days, ADF and PP reject), but a trade held for about 25–31 days against a moving anchor earns small amounts per trade and gives it back on a few stop-outs. The book's Sharpe of 0.055 cannot be told from zero.
2. **HO/RB is the one spread worth keeping an eye on** (3 of 3 eras positive, 27 trades, Sharpe 0.38, but t = 1.55). The step 6 regression and this backtest agree on its direction, and neither is strong enough to commit capital.
3. **Brent/WTI is where it breaks.** Both stop-outs came in the 2026 supply shock. A link that works in normal times and breaks in crises is what the gates were written to catch.
4. **The binding recommendation does not change.** Trend and carry remains the only strategy with four of five gates passed.

## Limits

- Three spreads, one run, a sample of about 16.7 years: t-statistics of 1–2 cannot separate a small edge from none.
- The signal reads the market ratio, so a roll can trigger an entry. The roll-day share of gross P&L is small, so it is not material here.
- The GARCH fit printed convergence warnings on some windows. The engine falls back to the window standard deviation when a fit is not usable, as registered.
- Hedge ratios are fixed dollar-neutral at each fit and do not drift while a trade is open.
- Equal-risk averaging of three spreads makes G5 lenient, so each spread is also shown at full R.

**Follow-up.** The slower version (monthly decisions) is in `report/exploratory_ratio_slow.md`: it did worse (book Sharpe −0.13), and HO/RB fell to −0.27.

## Next experiments (each to be registered before it is run)

1. Whether a small HO/RB ratio sleeve adds to the trend and carry book, given the −0.18 correlation. This is an overlay question, and the stand-alone sleeve does not pass.
2. Tracking the fixed rules forward on new data, which is the only test free of hindsight.
