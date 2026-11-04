# Could the statistics have predicted the gate outcomes? (descriptive)

Added 2026-11-04, after every strategy result was seen. It is a model-based bridge from the stationarity results to the gates, not a backtest, and it changes no verdict. Code: `scripts/run_gate_power.py`. Tables and figure: `output/exploratory/gate_power/`. Models: `scripts/run_ratio_stationarity.py` steps 10 to 14.

## 1. The moving-anchor model

ℓ<sub>t</sub> = ln k<sub>t</sub> + u<sub>t</sub>, with ln k<sub>t</sub> = ln k<sub>t−1</sub> + η<sub>t</sub> and u<sub>t</sub> = φ u<sub>t−1</sub> + ε<sub>t</sub>. A fixed mean is the case η = 0. Fitted by maximum likelihood (a local level plus an AR(1)) on the market log-ratio, with calendar-month effects removed for the crack and HO/RB. All three fits converge.

| Spread | Half-life if k is fixed | Half-life of u | Anchor drift per year | Standard deviation of u | Share of the variance that is anchor | LR statistic against a fixed mean |
|---|---|---|---|---|---|---|
| 3-2-1 crack | 51 days | 7.2 days | 10.4% | 3.4% | 87% | 53.8 |
| Brent / WTI | 51 days | 2.2 days | 9.0% | 1.0% | 97% | 84.3 |
| HO / RB | 36 days | 18.4 days | 8.4% | 6.9% | 50% | 7.1 |

- **The slow half-life is mostly the anchor drifting.** Around a moving anchor the reversion is fast (2 to 18 days) and small (1 to 7%).
- **The registered signal measures z against the wrong anchor.** The trailing 756-day mean differs from the filtered anchor by 5.5 (Brent/WTI), 2.0 (crack) and 0.8 (HO/RB) standard deviations of u. When the registered |z| ≥ 2, the filtered deviation is under 1 standard deviation on 86%, 65% and 36% of days. The order matches the order of the realised Sharpes (−0.49, 0.22, 0.38). That is three points, so it is suggestive.
- **Limits.** Parameters use the full sample. The fitted Brent/WTI anchor follows the data almost point by point, and the model cannot tell a drifting anchor from a slowly reverting series. The fast half-lives are statements about the model. The fast component may also include roll and settlement noise.

## 2. Does k tie out? Three descriptive checks (`ratio_stationarity` steps 11 to 13)

- **Same number from two routes.** The dollar regression S = a + c · P<sub>B</sub> gives a premium c + a · E[1/P<sub>B</sub>] of 29.6%, 7.0% and 7.4%. The mean log-ratio gives 28.7%, 6.8% and 6.9%. The gaps are at most 0.8 percentage points.
- **The 2013 heating oil to ULSD switch** (the month is an assumption from `oil-rv-plan.md`). HO/RB moves by +2.1% in May 2013 (t = 2.0, 29th percentile of all month-start jumps over 2008 to 2025): not unusual. The 2013 falls in the crack (−10.0%, 89th percentile) and Brent/WTI (−9.6%, 94th percentile) share the WTI leg and match the end of the Cushing glut. There is no HO-specific break.
- **Cushing stocks.** The deviation of Cushing stocks from their five-year average does not explain k: the log-ratio on it has t = 0.3 and R² 0.3% in weekly levels, and weekly changes give t = 2.4 and R² 0.9%. The glut story is consistent with the timing and not supported by this proxy.

## 3. Who moves, in logs (`ratio_stationarity` step 14)

With the log-ratio as the error-correction term and each leg's held-contract log return as the dependent variable (roll-clean):

| Spread | Leg | α per day | t |
|---|---|---|---|
| Crack | RB / HO / CL | +0.007 / +0.007 / **+0.017** | 1.6 / 1.8 / **2.6** |
| Brent / WTI | Brent / CL | +0.003 / +0.012 | 0.4 / 1.3 |
| HO / RB | HO / RB | −0.007 / −0.000 | −1.5 / −0.0 |

A positive α on the denominator leg (WTI) is the correcting direction. So **crude adjusts** in the crack and Brent/WTI, the product legs have the wrong sign, and HO has the right sign in HO/RB. In dollars (Test 7) the only significant adjuster was RBOB. The answer depends on the specification, so no strategy is built on it. The ratio's own pull is −0.0098, −0.0089 and −0.0072 per day (t = −2.3, −1.8, −1.2), half-lives 70, 77 and 96 days. The free log VECM (rank 2) loads on WTI (t = −5.0 on the first relation), but its normalisation makes the signs hard to read, so it is saved and not used.

The ARCH test rejects on daily log-ratio returns for all three spreads (p < 0.001), so GARCH sizing is unchanged.

## 4. The registered rule on four model worlds

The rule is simulated with constant-size positions (the simulated volatility is constant), P&L in log-ratio units and a cost of 0.05 / P<sub>B</sub> per change (P<sub>B</sub> the mean price, so $0.05 per barrel). 200 paths of 4,987 days per world and spread. No rolls or gaps. Rule: enter at |z| ≥ 2 when re-armed, exit at |z| ≤ 0.5, stop at 4, time stop at 3 half-lives, fill one close later. The estimating worlds refresh the 756-day mean, standard deviation and AR(1) half-life every 21 days and apply the registered 5 to 120 day half-life gate.

| Net Sharpe, mean of paths | A: k fixed, known | B: k fixed, estimated | C: k moves, estimated | D: k moves, anchor known | Realised, daily decisions |
|---|---|---|---|---|---|
| 3-2-1 crack | 0.83 | 0.80 | 0.47 | 1.64 | 0.22 |
| Brent / WTI | 0.80 | 0.79 | 0.07 | 1.59 | −0.49 |
| HO / RB | 0.99 | 0.96 | 0.83 | 1.25 | 0.38 |

- If k were fixed (A, B) the registered rule would clear the G1 bar on 93% to 100% of simulated paths (world B, Brent/WTI, is the 93%). With a moving k it has to estimate (C) the expected Sharpe is 0.47, 0.07 and 0.83.
- **World C reproduces the trade counts** (26, 16 and 26 round trips simulated, 29, 15 and 27 realised) **and the order of the spreads**. It does not reproduce the level: realised Brent/WTI and HO/RB are below the 5th percentile of world C, and the crack is at the 11th. Real data have rolls, fat tails, volatility clustering and crises. The moving anchor explains the ranking and part of the gap, not all of it.
- **World D** earns 1.25 to 1.64 in a model with none of the real-world frictions (the modelled cost is included). It is an upper bound on what better anchor tracking could be worth, and it assumes the anchor is known. We do not test a rule that tracks it on this sample (it would be the seventh look). If it is ever tested, the 1990 to 2006 data in the planned Bloomberg pull are the out-of-sample years for it.
- The conclusions hold at price levels of $40 and $120 (world C Sharpe within 0.05 of the base case).

## 5. How noisy is the gate?

The standard error of a measured annualised Sharpe over 16.7 years is about √((1 + SR² / 2) / 16.7) ≈ 0.25.

| True Sharpe | Probability that the measured Sharpe passes G1 (> 0.5) |
|---|---|
| 0 | 2% |
| 0.3 | 21% |
| 0.45 (our trend and carry estimate) | 42% |
| 0.5 | 50% |
| 0.75 | 82% |

- A strategy with no skill passes G1 about 2% of the time. If six skill-less variants were independent, at least one would pass 12% of the time (an upper bound, because the variants are related).
- Trend and carry at 0.45 is as likely a 0.5 strategy that was unlucky as a 0.4 strategy that was lucky.
- **Deflated Sharpe ratio** (Bailey and López de Prado, 2014) of the trend and carry book, whose measured Sharpe is 0.447, on its 4,213 daily returns (skew 0.17, kurtosis 10.7): **0.70** with six trials (the six variants: Sharpes 0.06, −0.19, 0.29, 0.45, 0.06, −0.13), 0.57 with twelve and 0.45 with twenty-four. It is the probability that the true Sharpe exceeds the best that skill-less trials would show. The trial variance comes from six numbers, so treat it as approximate.

## Limits

- The models are fitted on the full sample, so their parameters look ahead.
- The simulation has constant volatility and normal errors, no rolls, no regime breaks beyond the random-walk anchor, and a known half-life for the time stop in worlds A and D.
- The G1 standard error assumes independent returns. Positive autocorrelation in P&L would widen it.
