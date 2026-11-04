# Ratios and log-ratios: do the three spreads have a home? (exploratory)

Added 2026-11-04, after every registered and exploratory result was seen. It is descriptive, not a gate or a trading rule, and it changes no verdict. Script: `scripts/run_ratio_stationarity.py`. Tables and figures: `output/exploratory/ratio_stationarity/`.

## Why

The professor's argument: if the true link is P_A = k·P_B, then P_A − P_B = (k − 1)·P_B inherits the trend in prices, so a dollar gap can look non-stationary even though the economic link holds. The ratio, or log P_A − log P_B, should be the stationary object.

## The economic intuition: a constant k, not a constant gap

**Algebra.** If one product is worth a fixed multiple of another, P_A = k·P_B, then

- log P_A − log P_B = log k, a constant, and
- P_A − P_B = (k − 1)·P_B, which rises and falls with P_B.

A 6% premium is $1.20 a barrel when the benchmark is $20 and $8.40 when it is $140. A dollar-gap model has to treat those as two different regimes. The log-ratio sees the same 6% both times, so it is the right scale on which to ask "is this gap unusually wide?".

**Fixed versus proportional frictions.** What holds the two prices together decides which form is stationary:

- *Fixed dollar costs* (freight, pipeline tariffs, refinery operating cost per barrel) pull towards a stable dollar gap.
- *Proportional costs* (quality differences priced as a percentage, the cost of financing and storing the barrel, product values that move with crude) pull towards a stable ratio.

The regression difference = a + c·P (Step 5) separates them. The intercept a is the fixed part and c·P is the proportional part:

| Spread | a ($/bbl) | t of a | c | t of c | Reading |
|---|---|---|---|---|---|
| Brent − WTI | −0.71 | −0.7 | 0.080 | 4.7 | Purely proportional: the gap is about 8% of WTI, with no fixed dollar part |
| HO − RB | −2.35 | −0.9 | 0.102 | 3.3 | Purely proportional |
| 3-2-1 crack | **+7.90** | 4.0 | 0.176 | 5.6 | Both: about $8 a barrel fixed (refining cost) plus about 18% of the crude price |

This is why the professor's mechanism explains Brent/WTI and HO/RB fully and the crack only in part, and why the crack's log-ratio is still drifting across eras (below).

**Why each k should exist.**

- *Brent/WTI*: two light, sweet crudes that differ by location and a small quality gap. If Brent trades far above WTI, US exporters ship to Europe until the premium falls to the shipping cost, which is a small and roughly proportional slice of the price.
- *3-2-1 crack*: product value is built from the crude cost plus a refining margin. If the margin is large, refiners run harder and buy more crude, which pushes the ratio back down.
- *HO/RB*: both come from the same barrel, and refiners can shift yield between diesel and gasoline.

**Why k moves.** The implied k = exp(era mean of the log-ratio) steps between eras (premium of A over B):

| Era | Crack | Brent / WTI | HO / RB |
|---|---|---|---|
| 2007–10 | +14% | +0.1% | +4.0% |
| 2011–13 | +27% | +15.0% | +5.0% |
| 2014–19 | +32% | +8.0% | +5.0% |
| 2020–26 | +36% | +6.3% | +11.1% |

These shifts are consistent with known regime changes, but I have not tested the causes: the Cushing glut and pipeline bottleneck (Brent/WTI in 2011–13), the end of the US crude export ban in December 2015 (the premium narrows afterwards), refinery closures and the diesel shortage after 2020 and in 2022 (HO/RB). Zivot–Andrews dates the log-ratio breaks at July 2010 (Brent/WTI), December 2010 (crack) and May 2022 (HO/RB). So **k is a slowly moving anchor**: it supports trading reversion over weeks to months against a rolling average, and it does not support a fixed 20-year mean.

**Where trend fits.** Over one to three years the outright prices trend and k itself steps, so the two-horizon reading is: reversion in the ratio over weeks to months, persistence in outright prices (and in k) over a year or more. The variance-ratio profile (Step 8) shows the same split.

**A note on the word "ratio".** The *variance ratio* used for trend is unrelated to the price ratio. It is Var(q-day change) / (q × Var(1-day change)): above 1 means moves persist, below 1 means they reverse. The question the professor's point raises is which *changes* go in. The earlier test used back-adjusted dollar changes, which scale with the price level. Step 8 repeats it on held-contract log returns (for outrights) and held log-ratio returns (for spreads).

## Which price, for which job

- **Stationarity tests and z-scores use the market price of the held contracts** (`*_unadj`, $/bbl, same delivery month). That is what refiners and shippers face.
- **P&L uses back-adjusted daily changes.** The back-adjusted level is market price plus every past roll gap, which is non-stationary by construction (ADF on the accumulated gap: p = 0.97 for the crack). It also goes negative, so logs are impossible on it.
- The held-contract return of a leg is (back-adjusted change) / (market price the day before). A spread's held log-ratio return is ln(1 + rA) − ln(1 + rB). On every non-roll day this equals the change in the market log-ratio exactly (gap about 1e-15), so the roll-day remainder is the roll jump, and held return + roll jump = market change closes for all three spreads. (An earlier version used rA − rB and did not close, because simple and log returns diverge on the 2020 crash days.)
- The deck's Test 1 table (crack ADF 0.981) is on back-adjusted levels, so it partly measures roll gaps. At market prices the dollar differences were already borderline.

Series: crack = log((2·RB + HO) / (3·CL)), Brent/WTI = log(Brent / CL), HO/RB = log(HO / RB). The crack and HO/RB are also shown with calendar-month means removed (full sample).

## Results

### 1. Unit roots (daily, 2007–26, constant)

| Series | Form | ADF p | PP p | KPSS p |
|---|---|---|---|---|
| 3-2-1 crack | difference ($/bbl) | 0.059 | 0.055 | ≤ 0.01 |
| | log-ratio | 0.014 | 0.000 | ≤ 0.01 |
| | log-ratio, deseasonalised | 0.044 | 0.000 | ≤ 0.01 |
| Brent / WTI | difference | 0.027 | 0.018 | 0.015 |
| | log-ratio | 0.002 | 0.000 | ≤ 0.01 |
| HO / RB | difference | 0.017 | 0.006 | ≤ 0.01 |
| | log-ratio | 0.000 | 0.000 | ≤ 0.01 |

- ADF and PP reject a unit root in every ratio and log-ratio, with p falling by an order of magnitude or more against the difference (the exception is the crack's ADF, 0.059 → 0.014). The plain ratio gives the same picture as the log-ratio.
- With a trend term, Brent − WTI is not rejected (ADF 0.107, PP 0.078), and its log-ratio is (0.013, 0.000).
- **Under the deck's "all three tests agree" rule, no daily series is stationary, because KPSS rejects everywhere.** Month-end data: only Brent/WTI (ratio and log-ratio) passes all three. Crack and HO/RB fail on KPSS, and the HO/RB log-ratio also fails ADF at month-end (0.25) unless deseasonalised (0.000).
- Dropping 13 April to 8 May 2020 changes nothing material (log-ratio ADF 0.038, 0.002, 0.000; PP ≤ 0.001).

### 2. The KPSS disagreement

- **False alarms.** I simulated truly stationary AR(1) series at each series' fitted persistence and sample size (4,987 days, 200 runs each). KPSS at 5% rejects:

| | Crack | Brent | HO / RB |
|---|---|---|---|
| Difference (half-life 134, 143, 55 days) | 85% | 87% | 56% |
| Log-ratio (half-life 50, 49, 35 days) | 52% | 52% | 43% |

  So a KPSS rejection is close to uninformative here, and for the dollar differences it would almost always reject even if they were stationary. Caveat: level shifts push the fitted φ up, which overstates these rates somewhat. ADF failed to detect stationarity in 0–6% of the same simulations.
- **Level shifts are visible.** Era means of the log-ratio (2007–10, 2011–13, 2014–19, 2020–26): Brent/WTI 0.001, 0.140, 0.077, 0.061; crack 0.131, 0.239, 0.274, 0.311; HO/RB 0.039, 0.049, 0.049, 0.106. Inside eras, ADF fails in the crack in 2007–10 and 2020–26, and in Brent/WTI in 2011–13. KPSS passes inside some eras (crack 2014–19, Brent/WTI 2007–10 and 2020–26, HO/RB 2007–10 and 2011–13) and not others.
- **One structural break (Zivot–Andrews, 12 lags, intercept break):**

| | Difference p | Log-ratio p | Log-ratio break (approx.) |
|---|---|---|---|
| 3-2-1 crack | 0.24 | 0.001 | Dec 2010 |
| Brent / WTI | 0.22 | 0.001 | Jul 2010 |
| HO / RB | 0.07 | 0.0004 | May 2022 |

  With one level break allowed, the dollar differences do not reject a unit root and the log-ratios do. Break dates come from my own regression and are approximate. `arch` reports the statistic and p-value but not the date, and my t-statistic matches arch's for HO/RB and is 0.2–0.7 less negative for the others.

### 3. Persistence

Rolling 756-day windows, series deseasonalised for crack and HO/RB:

| | Windows with ADF p < 5%: difference → log-ratio | Median half-life (days) |
|---|---|---|
| Crack | 32% → 45% | 36 → 26 |
| Brent / WTI | 12% → 40% | 51 → 33 |
| HO / RB | 33% → 46% | 24 → 23 |

Full-sample half-life: 134 → 50, 143 → 49, 55 → 35 days. A three-year window still rejects only 40–46% of the time. The anchor is real but the evidence is thin inside any one window.

### 4. Cointegration in logs

| Pair | Form | Slope β | β = 1? (t, indicative) | Full-sample EG p | Rolling windows with EG p < 5% | Rolling windows where 1:1 residual passes ADF |
|---|---|---|---|---|---|---|
| Brent on WTI | levels | 1.080 | 6.7 | 0.048 | 18% | 12% |
| | logs | **1.009** | 0.9 | 0.010 | 24% | 40% |
| HO on RB | levels | 1.102 | 4.5 | 0.003 | 5% | 12% |
| | logs | **0.991** | −0.4 | 0.000 | 14% | 25% |

- In logs the slope is 1 and cannot be distinguished from 1. In levels the 1:1 restriction is rejected. That is the professor's constant-k model, and the log form is the correct specification.
- The rolling pass rates here are on market legs, so they are not comparable with the deck's 11% and 8% (the deck's own legs). The like-for-like comparison is levels against logs on the same legs: it improves, but a rope exists in a minority of three-year windows.
- Johansen on (CL, RB, HO): no cointegrating vector in 72% of windows in levels and 59% in logs (deck: 65%).

### 5. The professor's mechanism

- The rolling 60-day sd of the held dollar change correlates with the price level at +0.35 (crack, vs WTI), +0.42 (Brent − WTI, vs WTI) and +0.46 (HO − RB, vs RBOB). For the held log-ratio return the correlations are −0.31, −0.20 and −0.19, so scaling by price removes the dependence (slightly over-corrects).
- Regressing difference = a + c·P: c = 0.176 (t = 5.6), 0.080 (4.7) and 0.102 (3.3), with R² of 12%, 10% and 5%. The residual is more stationary than the difference (ADF p 0.059 → 0.020, 0.027 → 0.013, 0.017 → 0.001; half-life 157 → 122, 143 → 113, 91 → 76 days). **The price level is part of the story, but explains only 5–12% of the variance of the dollar gap.**

### 6. Can a holder earn the reversion?

- **Roll jumps are large.** The median roll-day move in the market log-ratio is 0.011 against 0.006 on other days (crack), 0.005 against 0.003 (Brent/WTI) and 0.016 against 0.007 (HO/RB). The roll jumps add up to −2.08, −1.21 and +0.44, against ranges of 0.85, 0.58 and 1.15. They are offset by held-contract returns of +2.38, +1.33 and −0.05, leaving net market changes of +0.30, +0.11 and +0.39.
- **The cumulated held return is itself non-stationary** (ADF p 0.67, 0.51, 0.13; the only rejection is HO/RB on PP, 0.048), as expected for a P&L path. The professor's anchor is in the market ratio, not in the P&L path.
- **Predictive regression** of the forward held-contract log-ratio return on today's market z-score (756-day window, month effect removed in-window for crack and HO/RB, Newey–West with lags = horizon). Warm-up means the sample starts in late 2009 (n ≈ 4,200):

| Series | 21-day held slope (t) | 63-day held slope (t) | Share of market reversion that is roll, 21 / 63 days |
|---|---|---|---|
| 3-2-1 crack | −0.008 (−1.72) | −0.011 (−1.52) | 42% / 54% |
| Brent / WTI | −0.002 (−1.20) | −0.005 (−0.96) | 43% / 41% |
| HO / RB | **−0.016 (−2.78)** | **−0.030 (−3.51)** | 28% / 22% |

  Slopes are per 1 z, in log-ratio units. HO/RB is negative in all three eras at 21 days (t = −3.6, −1.5, −2.5). Brent/WTI is significant in one era at each horizon only (2014–19 at 21 days, t = −2.1; 2020–26 at 63 days, t = −2.3).
- **Fade after |z| ≥ 2** (about 9–10% of days): average forward 21-day held log-ratio return of fading the signal is +1.9% (crack, approx. t 0.7), +0.1% (Brent/WTI, 0.1) and +6.2% (HO/RB, 1.8). t-statistics divide n by 21 and are rough.

### 7. Daily changes of the ratio

- Robust ACF, lags 1–20, Benjamini–Hochberg across series: 0 rejections on held log-ratio returns and 0 on market changes excluding roll days.
- Robust Ljung–Box p at lag 20: 0.36, 0.44 and 0.80 (held returns). The plain test gives 0.000 and the ARCH test gives 0.000, the same volatility-clustering pattern as before.
- **Not identical to the dollar changes.** BIC picks AR(6), AR(3) and AR(0) for the held log-ratio returns (the deck's dollar changes: AR(0), AR(2), AR(0)), and AIC-model R² is 6.8%, 4.1% and 1.8%. The crack lag-1 ACF is +0.09, and one lag is nominally significant before the Benjamini–Hochberg correction. None is significant on the robust ACF, so I read it as weak, not as an edge, but it is more than for the dollar changes and should be re-checked before the deck says "unchanged".

### 8. Variance-ratio profile: two horizons

Variance ratio by horizon in trading days, on held log returns (outrights) and held log-ratio returns (spreads), against the back-adjusted dollar changes of the earlier horizon test (which this script reproduces exactly: max difference 0.0000 at q = 21 to 252).

| Series | Basis | q = 5 | 21 | 63 | 126 | 252 | 12 months (monthly) |
|---|---|---|---|---|---|---|---|
| 3-2-1 crack | dollar | 0.97 | 1.00 | 1.04 | 1.00 | 0.98 | 1.17 |
| | log-ratio | 0.92 | 0.84 | 0.66 | 0.55 | **0.50** | 0.64 |
| Brent / WTI | dollar | 1.02 | 1.06 | 1.15 | 1.00 | 1.03 | 0.99 |
| | log-ratio | 0.84 | 0.73 | 0.63 | 0.49 | **0.41** | 0.54 |
| HO / RB | dollar | 0.97 | 0.87 | 0.75 | 0.68 | 0.66 | 0.75 |
| | log-ratio | 1.00 | 1.11 | 0.77 | 0.61 | **0.48** | 0.41 |
| Brent | log return | 0.97 | 1.05 | 1.38 | 1.40 | 1.34 | 1.33 |
| ULSD (HO) | log return | 1.00 | 1.09 | 1.38 | 1.47 | 1.46 | 1.51 |
| WTI | log return | 0.94 | 0.99 | 1.20 | 1.17 | 1.11 | 1.17 |
| RBOB | log return | 0.97 | 1.15 | 1.31 | 1.20 | 1.05 | 0.97 |

- On dollar changes the crack and Brent/WTI look like random walks (VR about 1). On log-ratio returns the variance at one year is only 41–50% of the random-walk level for all three spreads. **The dollar-based test hid the reversion that the professor's argument predicts.**
- Outrights lean the other way (VR 1.1–1.5 at 63 to 252 days), which is the trend horizon.
- **None of the log-ratio variance ratios is significant** (robust p from 0.15 to 0.98; Brent/WTI monthly 6 and 12 months p = 0.15) and the outrights are mostly not either (only ULSD at 63 days, p = 0.04). The test has low power at these horizons, so this is supportive, not confirming.

### 9. Does trend change if it trades the log-ratio?

The registered trend rule takes the sign of the 252-day back-adjusted dollar change. The sign of the 252-day held log-ratio return **differs on only 6–12% of days** (crack 11.8%, Brent/WTI 11.3%, HO/RB 6.4%), so trading the sign of the log-ratio would be almost the same rule. Trend is left as registered.

### 15. ARIMA identification (step 15)

Orders chosen by BIC over p, q ≤ 3 (AIC is reported in `arima_orders.csv`). A series modelled as ARIMA(p,1,q) is fitted as ARMA(p,q) on its first difference. Roots are of the AR and MA polynomials in the lag operator L (a root of modulus above 1 is stable).

| Series | Form | BIC order | Slowest AR root, half-life |
|---|---|---|---|
| Brent, RBOB, ULSD | log price, d = 1 (held-contract log return) | ARIMA(0,1,0) | none |
| WTI | log price, d = 1 | ARIMA(2,1,2), but (0,1,0) once 9 March to 29 May 2020 is excluded | none |
| Crack, HO/RB | dollar gap, d = 1 (back-adjusted daily change) | ARIMA(0,1,0) | none |
| Brent − WTI | dollar gap, d = 1 | ARIMA(2,1,0), coefficients 0.06 and −0.04 | none |
| Crack | log-ratio, d = 0 (market) | ARIMA(1,0,3) | 81 days |
| Brent / WTI | log-ratio, d = 0 | ARIMA(3,0,2) | 98 days |
| HO / RB | log-ratio, d = 0 | ARIMA(1,0,0) | 35 days |
| Crack, Brent / WTI, HO / RB | roll-clean held index, d = 1 | (2,1,3), (3,1,0), (0,1,0) | none |

- **Integration order.** All four log prices fail ADF (p 0.25 to 0.93) and KPSS rejects, while the returns pass both: d = 1.
- **WTI's ARMA(2,2) is an artefact.** Its AR roots (0.24 ± 1.16i) and MA roots (0.28 ± 1.20i) cancel, and it beats the random walk by only 13 BIC points. Excluding 9 March to 29 May 2020, BIC picks the random walk.
- **Residuals.** The robust Ljung–Box test finds nothing left in any fit except HO/RB's market log-ratio (p = 0.001 for d = 0, 0.000 for d = 1). The roll-clean index passes (p = 0.80): the leftover structure is the roll jumps. ARCH is present in every series (p < 0.001).
- **Dollar gap against log-ratio.** The dollar gap has d = 1 with almost no short-run structure. The log-ratio is d = 0 with one root of modulus 1.007 to 1.02, which closes 0.7% to 2% of the gap a day. Next to the daily noise that is invisible, so the ARIMA on the dollar change sees a random walk. It is one process seen through two lenses. HO/RB's d = 1 fit has θ = −0.99, the signature of over-differencing, which supports d = 0.
- **Brent/WTI ARMA(3,0,2), factored.** AR = (1 − 0.993 L)(1 − 0.447 L + 0.275 L²): a slow root (half-life 98 days) and a pair of complex roots (0.814 ± 1.726i, modulus 1.909) that decay by 0.52 a day with a period of about 5.6 days. The MA roots (2.79 and 16.5) are far from the complex pair, so nothing cancels. Reading: months for cargo, pipeline and export arbitrage to move the ratio, and a short-run oscillation as one benchmark reacts first. The order counts polynomial terms. It is not five separate economic facts. (An earlier version of this note said the lag-1 terms nearly cancel. That was a coincidence of coefficient sizes, 0.447 against 0.419, not a root cancellation.)
- **Out of sample** (756-day window, refit monthly, 1, 5, 21 and 63 days ahead). No AR(1) or AR(3) forecast of the log-ratio beats a random walk: RMSE ratios 0.94 to 1.14, Diebold–Mariano |t| of at most 1.8. The best is HO/RB at 21 and 63 days (0.94 to 0.96, t up to 0.8). The trailing 756-day mean, the anchor of the registered z, is worse than no-change at every horizon except HO/RB at 63 days (0.986).
- **Roll-clean index.** It is a random walk with drift (partly cancelling terms for the crack, coefficients under 0.14 for Brent/WTI). The market ratio is stationary because the roll jumps offset that drift.

## Reading

1. **The professor's claim is supported.** ADF, PP, Zivot–Andrews and log cointegration all point the same way, and the log slope of 1.009 (Brent/WTI) and 0.991 (HO/RB) is his constant-k model. Half-lives fall from 134–143 to about 50 days for crack and Brent/WTI.
2. **The deck's "no anchor" headline was overstated.** It rested on back-adjusted levels, which include 20 years of accumulated roll gaps.
3. **"Stationary around one fixed mean for 20 years" is not supported.** KPSS rejects everywhere, but at these half-lives it would reject about half the time on truly stationary data. The era means show the equilibrium moves with regimes (Cushing glut, post-2022 diesel). The anchor holds, and k shifts.
4. **Tradeability differs by spread.** HO/RB: a holder earns the reversion (t = −2.6 and −3.2, all three eras negative). Crack and Brent/WTI: the sign is right, but significance is weak (|t| 1.0–1.7) and roll jumps account for 41–54% of the market reversion, so they are the hardest to earn.

## Limits

- Three spreads, two horizons and several eras: the t-statistics above are not corrected for multiple testing. HO/RB at both horizons clears the six-test Bonferroni threshold of |t| = 2.64 (the one used for pairs), and the crack and Brent/WTI do not.
- The month effects for crack and HO/RB are removed over the full sample in the stationarity tests (look-ahead), and in-window in the predictive regression (no look-ahead).
- The held return is computed on the contract held over each day, so it excludes roll jumps by construction. It is not a full P&L of a dollar-neutral position: no costs, no hedge-ratio drift, no sizing.
- The simulation assumes AR(1) with normal errors. Real series have fat tails and clustering.
- Month-end tests have about 237 points and low power.
- Zivot–Andrews allows one break only. Brent/WTI has at least two visible shifts.

## What followed

- Ratio mean reversion was pre-registered for all three spreads on 2026-11-04 (`PREREGISTRATION_RATIO.md`, commit `bd725b7`) and run once: the book fails (net Sharpe 0.055, G1 and G5 fail). HO/RB, the best of the three, has Sharpe 0.38 (t = 1.55). See `report/exploratory_ratio_mr.md`.
- A slower version (decisions monthly, to match the horizon where the variance ratios fall below 1) was pre-registered next (`PREREGISTRATION_RATIO_SLOW.md`, commit `f49eeb4`) and made things worse: book Sharpe −0.13 on 22 trades. See `report/exploratory_ratio_slow.md`.
- Steps 10 to 14 of `scripts/run_ratio_stationarity.py` (a moving-anchor model, a tie-out of k, the 2013 heating oil switch, Cushing stocks, and the VECM in logs) and `scripts/run_gate_power.py` (the registered rule on model worlds, the noise in G1, a deflated Sharpe ratio) are written up in `report/exploratory_gate_power.md`.
- The deck's Test 1 now has the back-adjusted result as the "P&L basis" row, then the market-price dollar gap and the log-ratio, with new slides on the professor's argument, KPSS, the roll, and the two horizons.
