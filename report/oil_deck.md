<!-- One section per slide, separated by "---". Renders with Marp or pandoc -t revealjs.
     Figures: report/deck_figures/ (scripts/build_deck_figures.py). Numbers: scripts/check_deck.py. -->

# Can the oil complex be traded?

Crude, gasoline and diesel prices are tied together by refining. We tested whether the gaps between them can be traded.

- 3-2-1 crack, Brent − WTI, diesel − gasoline (HO − RB)
- Bloomberg futures, January 2007 to October 2026
- QF603 project, 3 November 2026

<!-- Speaker: the result is a near miss, not a winner. The value is in what the tests rule out and why. -->

---

## One barrel of crude becomes several products

- A refinery splits crude into gasoline, diesel and other fuels. A rule of thumb is **3 barrels of crude give 2 of gasoline and 1 of diesel**.
- The **3-2-1 crack** is what the refiner earns on that split: 2 gasoline + 1 diesel − 3 crude, per barrel of crude.
- Product prices and crude prices must stay linked. If gasoline sells for far more than the crude that makes it, refiners buy more crude and make more gasoline, until the gap narrows.

**Why this matters for trading:** a link that pulls prices back together is the whole reason to expect a spread to snap back.

---

## Three spreads, three economic links

| Spread | What ties it together | What can break the tie |
|---|---|---|
| **3-2-1 crack** (products − crude) | Refiners switch to the more profitable product and add or cut crude runs | Refinery outages, hurricanes, product shortages, a Hormuz closure |
| **Brent − WTI** (two crudes) | Cargoes move between the Atlantic and the US Gulf, so the gap should not exceed shipping cost and quality difference | Pipeline and export limits (Cushing glut 2011–13), export rules, regional supply shocks |
| **HO − RB** (diesel − gasoline) | Refineries can shift yield between diesel and gasoline | Season (summer driving, winter heating), grade changes, diesel shortages as in 2022 |

**For a trade:** each link is a reason the gap should revert. Each break is a reason it might not.

---

## The hypothesis, and what "worth trading" means

**Hypothesis.** When a spread is unusually wide, it will narrow. Sell it wide, buy it narrow, collect the reversion.

**Worth trading** means passing five questions, **written down before we saw any data** (24 October 2026):

| Gate | The question in plain English |
|---|---|
| G1 | Is the reward big enough? Net Sharpe above 0.5 |
| G2 | Is it enough trades to mean anything? At least 20 |
| G3 | Does it work in every era? Positive in at least 2 of 3 periods (2007–13, 2014–19, 2020–26) |
| G4 | Does it survive changing the settings? Positive at every setting tried |
| G5 | Does it survive a crisis? No loss worse than $40k in 2022 or 2026 |

**Sharpe ratio, plainly:** average profit divided by how bumpy the profit is. Above 0.5 after costs is a modest bar.

<!-- Speaker: trend/carry (28 Oct) and pairs (31 Oct) rules were written after the mean-reversion verdict, so they are labelled exploratory. They are judged on the same five gates. -->

---

## How we got there: tests first, then strategies

1. **Tests** tell us what the prices do: do they wander, do they come back, can yesterday predict today, does risk cluster.
2. Each test ends with **"For a trade"**: what it allows or rules out.
3. A **bridge slide** turns the findings into a shortlist of strategies.
4. Each strategy is judged on the **same five gates**.

**Why this order:** testing first stops us from backtesting ideas the data already contradicts.

---

## The data: futures expire, so the series are stitched

- A futures contract ends. To follow "the price of oil" for 20 years we hold one contract, then **roll** into the next about 5 business days before expiry.
- **Back-adjusting** shifts old prices so that daily changes are exact. The cost is that the **level** drifts: every roll gap is added to it.
- The 3-2-1 crack on the back-adjusted basis runs from −$163 to +$65 a barrel. At market prices it ranges from $2 to $70.

![3-2-1 crack, back-adjusted against market level](deck_figures/01_backadjusted_vs_market.png)

**For a trade:** the registered signal used the adjusted level, so a "wide" spread can partly be accumulated roll gaps, not an economic extreme. This matters in the results. Anything that needs a real price (a ratio, a log, a stationarity test) uses the market level, and P&L always uses the adjusted changes.

---

## What the three spreads look like

![Three spreads at market level, with stress windows shaded](deck_figures/02_spreads_market_level.png)

- Levels sit in ranges for years, then move to a new range: Brent − WTI in 2011–13, the crack and HO − RB in 2022 and 2026.
- The shaded windows are the periods gates G3 and G5 look at.

---

## A professor's question: dollar gaps, or ratios?

**The argument.** If one product is worth a fixed multiple of the other, P<sub>A</sub> = k · P<sub>B</sub>, then

- the **ratio**, log P<sub>A</sub> − log P<sub>B</sub> = log k, is a constant, but
- the **dollar gap**, P<sub>A</sub> − P<sub>B</sub> = (k − 1) · P<sub>B</sub>, rises and falls with the price level.

A 6% premium is $1.20 a barrel when the benchmark is $20 and $8.40 when it is $140. A dollar-gap test can say "no anchor" even though the economic link holds.

![Dollar gap against log-ratio, with era means](deck_figures/10_difference_vs_log_ratio.png)

- The rolling volatility of the dollar gap moves with the price level (correlation +0.35 to +0.46). For the log-ratio return it does not (−0.19 to −0.31).
- Brent − WTI and HO − RB have **no fixed dollar part** (intercepts −$0.71 and −$2.35, not different from zero): the gap is proportional to price. The crack has about **$8 fixed** (refining cost) plus about 18% of the crude price.

**What changes.** Stationarity tests and signals use the **market price of the held contracts**. **P&L still uses back-adjusted daily changes**, which are exact. A back-adjusted level carries 20 years of roll gaps, so it is not a price.

---

## Test 1: does the series have a home? (ADF, PP, KPSS)

**In plain English.** A rubber band always pulls back to its anchor. A drunk walk has no anchor, and the further he is from the start, the more likely he is to keep going. These tests ask which one a price looks like.

- **ADF and PP** start by assuming there is no home (a unit root). A small p-value means "there is a home".
- **KPSS** starts by assuming there is a home. A small p-value means "there is not".
- We call a series stationary only if all three agree.

| Back-adjusted, 2007–26 | Level: ADF p | PP p | KPSS p | Daily change: ADF p | KPSS p |
|---|---|---|---|---|---|
| 3-2-1 crack | 0.981 | 0.981 | ≤ 0.01 | < 0.001 | ≥ 0.10 |
| Brent − WTI | 0.652 | 0.692 | ≤ 0.01 | < 0.001 | ≥ 0.10 |
| HO − RB | 0.774 | 0.775 | ≤ 0.01 | < 0.001 | 0.078 |

**For a trade:** back-adjusted dollar levels do not return to a fixed home. But these levels contain 20 years of accumulated roll gaps (the gap alone fails ADF at p = 0.97 for the crack), so the next slide asks the question on market prices.

---

## Test 1 again, on market prices: the ratio has a home

| Daily, 2007–26, futures | Back-adjusted dollar gap, ADF p | Market dollar gap, ADF / PP p | **Log-ratio, ADF / PP p** | Half-life, dollar → log (days) | Zivot–Andrews p, dollar → log |
|---|---|---|---|---|---|
| 3-2-1 crack | 0.981 | 0.059 / 0.055 | **0.014 / 0.000** | 134 → 50 | 0.24 → 0.0009 |
| Brent / WTI | 0.652 | 0.027 / 0.018 | **0.002 / 0.000** | 143 → 49 | 0.22 → 0.001 |
| HO / RB | 0.774 | 0.017 / 0.006 | **0.000 / 0.000** | 55 → 35 | 0.07 → 0.0004 |

- ADF and PP reject a unit root in every log-ratio. Allowing one structural break (Zivot–Andrews), the **dollar gaps still do not reject and the log-ratios do**.
- **The log slope is 1.009 for Brent on WTI and 0.991 for HO on RB**, indistinguishable from 1. In dollars the 1:1 link is rejected (slopes 1.08 and 1.10). That is the constant-k model.
- Half-lives are for the crack and HO/RB with calendar-month effects removed.

**For a trade:** the dollar gap wanders, the ratio does not. The strict "all three tests agree" rule still fails, because KPSS rejects everywhere (next slide).

---

## Why KPSS still disagrees, and what the anchor really is

- KPSS assumes the series is stationary and rejects if not, so a rejection is real evidence against **one fixed mean for 20 years**.
- But it also **cries wolf on slow series**. On simulated series that are truly stationary at each spread's own persistence, KPSS rejects 52%, 52% and 43% of the log-ratios, and 85%, 87% and 56% of the dollar gaps (should be 5%).

![KPSS false alarms](deck_figures/11_kpss_false_alarms.png)

- The mean does move. Implied k = exp(average log-ratio in the era):

| Era | Crack | Brent / WTI | HO / RB |
|---|---|---|---|
| 2007–10 | 1.14 | 1.00 | 1.04 |
| 2011–13 | 1.27 | 1.15 | 1.05 |
| 2014–19 | 1.32 | 1.08 | 1.05 |
| 2020–26 | 1.36 | 1.06 | 1.11 |

- These steps fit known regime changes (Brent/WTI in the 2011–13 Cushing glut, narrowing after the US export ban ended in December 2015; HO/RB after the 2020 and 2022 diesel shortages), but the causes are not tested here.

**For a trade:** the anchor is real and it moves. Compare against a rolling average, never a fixed 20-year mean.

---

## One difference is enough, two is too many

- Taking daily changes (d = 1) makes every series pass all three tests.
- Taking changes of changes (d = 2) **adds noise**: the first autocorrelation falls to about −0.5, and the spread of the data rises by about 40%.
- At market prices Brent − WTI and HO − RB reject a unit root on ADF and PP (p = 0.027 and 0.017), but KPSS also rejects stationarity. So the tests disagree, and the honest reading is "very persistent, possibly slow to return".

**For a trade:** treat prices as random walks over days to months, and judge any reversion claim by trading it, not by one p-value.

---

## Test 2: are two prices on a leash? (cointegration)

**In plain English.** Two drunks who each wander randomly, but are tied together by a rope. Neither has a home, yet the **distance between them** does. Engle–Granger and Johansen test for the rope.

![Rolling Engle–Granger p-value](deck_figures/05_rolling_cointegration.png)

- A rope would show as p below 5% (the dashed line). Brent vs WTI passes in 11% of rolling 3-year windows, HO vs RB in 8%.
- Johansen on the crack legs finds no rope in 65% of windows.
- **In logs, on market prices of the held contracts:** a rope is found in 24% of Brent/WTI windows (18% in dollars) and 14% of HO/RB windows (5% in dollars). Better, but still a minority of three-year windows.

**For a trade:** a link exists in some periods, not others. Entry needs a stability check, which is what the regime filter was for.

---

## Test 3: does yesterday's move echo today's? (ACF and PACF)

**In plain English.** Shout in a valley and listen for the echo. The ACF measures how much of today's change is repeated at each lag. The PACF measures the same after removing echoes already explained by shorter lags.

![ACF and PACF of daily changes](deck_figures/03_acf_pacf_daily.png)

- Bars outside the grey band are "significant". By chance about 1 in 20 will be, and these tests look at 280 lags.
- After correcting for testing many lags, **0 of 280** survive.
- The same holds for daily changes of the log-ratio: no lag survives the correction.

**For a trade:** no daily forecasting edge. Yesterday does not tell us today.

---

## Test 4: how many past days should a model use? (AIC and BIC)

**In plain English.** Adding past days to a model always fits the past a bit better, like adding more squiggles to a curve. AIC and BIC charge a fee per extra day. **BIC charges more**, so it only keeps terms that clearly earn their place.

| Daily changes, 2007–26 | BIC picks | AIC picks | R² of the AIC model |
|---|---|---|---|
| 3-2-1 crack | AR(0) | AR(20) | 0.9% |
| Brent − WTI | AR(2) | AR(19) | 1.4% |
| HO − RB | AR(0) | AR(16) | 1.3% |

- AR(0) means "the best forecast of tomorrow's change is zero".
- AIC's long orders explain about 1% of the variance and vary by period. That is over-fitting, not memory.
- On log-ratio changes BIC picks AR(6), AR(3) and AR(0) (R² of the AIC model 6.8%, 4.1%, 1.8%): a little more memory than in dollars, but nothing significant on the robust ACF.

**For a trade:** an AR(4) or AR(20) forecaster would be fitting noise, and has no economic story. No daily AR strategy.

---

## Test 5: is anything left in the leftovers? (Ljung–Box)

**In plain English.** After the model is fitted, the leftover errors should look like pure noise. Ljung–Box tests whether any pattern remains.

- Oil returns have bursts of volatility, which makes the plain test **cry wolf**: on simulated data with a true AR(1), it falsely rejects 32–50% of the time. The robust version rejects 4–7%.
- On the stored fits, the robust test rejects in 0–12% of windows, in line with chance.
- The test is weak against small misses: it catches a real −0.10 effect only 15–30% of the time.

**For a trade:** the simple short-lag model is adequate, but absence of evidence is not proof of no pattern.

---

## Test 6: can't predict direction, can predict size (ARCH and GARCH)

**In plain English.** We cannot say whether tomorrow's move is up or down, but we can say whether it is likely to be big. Calm days follow calm days, and storms follow storms.

![Rolling volatility of daily changes](deck_figures/06_rolling_volatility.png)

- The ARCH test rejects "no volatility clustering" for all three spreads (p < 0.001).
- In oil this is natural: news, inventory data and supply shocks arrive in clusters.
- GARCH forecasts beat a constant standard deviation out of sample on all three spreads.

**For a trade:** size each position by forecast volatility, so a calm day and a stormy day risk about the same dollars.

---

## Test 7: when the gap is off, who moves? (VECM)

**In plain English.** If two legs are tied by a rope, one of them must give. A VECM estimates how fast each leg pulls back.

- On futures, only **RBOB** shows a clearly significant adjustment speed (t = −2.9). Crude and diesel adjust slowly or not at all.
- Plausibly because gasoline is the most seasonal leg, so its price has to do the adjusting.

**For a trade:** any reversion is slow (median half-lives of 92 to 125 days in the rolling fits), so trades are held for weeks and costs and risk limits matter.

---

## Can a holder earn the reversion? (the roll)

A market ratio **jumps when we roll** into the next contract month, and a holder does not earn that jump. So we split every change in the market log-ratio into the **held-contract return** (what a position earns) and the **roll jump**.

- The median move on a roll day is about twice a normal day (crack 0.011 against 0.006, Brent/WTI 0.005 against 0.003, HO/RB 0.016 against 0.007).
- Over 20 years roll jumps add up to −2.08, −1.21 and +0.44, against ranges of 0.85, 0.58 and 1.15. Held-contract returns offset them (+2.38, +1.33, −0.05).

Forward held-contract return regressed on today's z-score (756-day window), slope per 1 z:

| Spread | 21 days (t) | 63 days (t) | Share of the market reversion that is roll jump |
|---|---|---|---|
| HO / RB | **−0.016 (−2.78)** | **−0.030 (−3.51)** | 28% / 22% |
| 3-2-1 crack | −0.008 (−1.72) | −0.011 (−1.52) | 42% / 54% |
| Brent / WTI | −0.002 (−1.20) | −0.005 (−0.96) | 43% / 41% |

**For a trade:** only HO/RB shows reversion a holder clearly earns. The crack and Brent/WTI have the right sign but are not significant, and about half of their apparent reversion is the roll.

---

## Two horizons: ratios revert, outright prices persist

![Variance ratio by horizon](deck_figures/12_variance_ratio_profile.png)

- **Variance ratio** = variance of the q-day change / (q × variance of the 1-day change). Above 1, moves persist (trend). Below 1, they partly undo themselves.
- On **dollar changes** the crack and Brent/WTI looked like random walks (about 1.0 at one year). On **log-ratio returns** the variance at one year is only **50%, 41% and 48%** of the random-walk level.
- Outright prices lean the other way: Brent 1.34 and ULSD 1.46 at one year.
- **None of the log-ratio variance ratios is statistically significant** (robust p from 0.15 to 0.98), so this supports the picture but cannot confirm it.

**For a trade:** reversion in the **ratio over months** (mean reversion, tested on the next slides: it does not pay), persistence in **outright prices over a year** (trend). The trend rule is unchanged: the sign of the 252-day change in the log-ratio differs from the sign of the dollar change on only 6–12% of days.

---

## What the tests allow

| Finding | What it means | Strategy it motivates | Judged on |
|---|---|---|---|
| Changes are close to white noise (BIC: AR(0) for 6 of 7) | No daily forecasting edge | Drop daily AR and SARIMA trades | not carried forward |
| Dollar levels wander, link unstable (rope in 8–11% of windows) | Reversion is possible but only sometimes | Mean reversion and cointegration pairs, with a stability check | G3, G4, G5 |
| **Ratios are stationary** (ADF, PP, Zivot–Andrews), half-life 35–50 days, but k moves between eras | The anchor is proportional, and it drifts | **Mean reversion on the log-ratio**, against a rolling mean | G1 to G5 |
| Variance ratios: log-ratios below 1, outright prices above 1 | Two horizons: reversion in weeks to months, persistence over a year | Mean reversion (short horizon) and trend (long horizon) | G1 to G5 |
| Volatility clusters (ARCH p < 0.001) | Risk is forecastable | Volatility-based sizing in every strategy | G5 |
| Daily tests cannot see 12-month moves or the futures curve | A gap in the evidence | Trend and carry, tested directly (appendix) | G1 to G5 |

**The tests rule things out. They cannot pick a winner, so the shortlist goes to the backtest.**

---

## Five candidates, each with an economic reason

| Strategy | Idea | Economic reason | Motivated by |
|---|---|---|---|
| **Mean reversion** (registered, dollar gap) | Fade spreads that are 2 standard deviations from the 3-year average | Arbitrage and substitution pull gaps back | Persistent but possibly anchored levels |
| **Ratio mean reversion** (registered 4 Nov) | The same rules on the log-ratio of market prices | Substitution works in percent, so k is the anchor | Ratios are stationary, half-life 35–50 days |
| **Pairs** | Trade legs tied by a rope (cointegration) | A long-run link between two prices | The rope test, when it passes |
| **Trend** | Follow the sign of the past 12 months | Supply and demand adjust slowly, so big shocks keep going | Levels that wander in one direction |
| **Carry** | Long if the curve is backwardated, short if in contango | Paid for insuring hedgers; scarce prompt barrels sell at a premium | The curve, which no daily test looks at |

---

## Mean reversion: all three spreads fail

| Futures, 2010–26, net of costs | G1 Sharpe | G2 trades | G3 eras | G4 settings | G5 crises |
|---|---|---|---|---|---|
| 3-2-1 crack | 0.32 ✗ | 18 ✗ | 2 of 3 ✓ | ✓ | ✗ (−$73k in 2026) |
| Brent − WTI | −0.42 ✗ | 9 ✗ | 0 of 3 ✗ | ✗ | ✗ (−$151k in 2022) |
| HO − RB | 0.06 ✗ | 18 ✗ | 2 of 3 ✓ | ✗ | ✗ (−$84k in 2026) |

![Cumulative net P&L of mean reversion](deck_figures/07_mean_reversion_pnl.png)

**Why, in terms of the tests.** The levels drift (slide on back-adjusting), so |z| ≥ 2 holds on 40%, 21% and 33% of days, roughly twice as often as on spot prices, which is too often for a true extreme. The ties break in exactly the stress years the gates check. The registered signal also traded the **dollar gap**, which grows with the price level, so it is the wrong object if the anchor is a ratio. The ratio version is tested after the pairs.

---

## Pairs: convergence lost money

- Six cointegrated pairs: net Sharpe **−0.19** on 80 round trips. Only G2 passes.
- Costs are not the cause. At zero cost the Sharpe is −0.14.
- Split the P&L: **convergence lost $294k**, while the pairs' own carry made $129k.
- Without the cointegration gate, the Sharpe is 0.17 (t = 0.70), which still fails G1 and G5.

**What it says economically:** the leash does not hold on a trading horizon. What did pay in the pairs book was the carry between the legs, a hint for the next strategy.

---

## Ratio mean reversion: stationary, but still not tradeable

Pre-registered on 4 November, before any signal or P&L. Same rules and gates as the registered strategy, with one change: z is read from the **log-ratio of market prices** and the legs are dollar-neutral.

| Futures, 2010–26, net of costs | G1 Sharpe | G2 trades | G3 eras | G4 settings | G5 crises |
|---|---|---|---|---|---|
| 3-2-1 crack | 0.22 ✗ | 29 ✓ | 3 ✓ | ✓ | ✓ |
| Brent / WTI | −0.49 ✗ | 15 ✗ | 1 ✗ | ✗ | ✗ (−$348k in 2026) |
| HO / RB | 0.38 ✗ (t = 1.55) | 27 ✓ | 3 ✓ | ✓ | ✓ |
| **Three-spread book** | **0.055 ✗** | 71 ✓ | 2 ✓ | ✓ | ✗ (−$133k in 2026) |

![Cumulative net P&L of ratio mean reversion](deck_figures/13_ratio_mean_reversion_pnl.png)

- **Changing the object did not rescue mean reversion.** The same weights with z on the market *dollar* gap give a book Sharpe of −0.05, against +0.06 for the log-ratio. HO/RB is the one spread that improves (0.20 → 0.38).
- Costs are not the cause: at zero cost the book Sharpe is 0.10. It flips sign with the estimation window (−0.06 at 504 days, 0.24 at 1,008).
- Brent/WTI loses in the 2026 supply shock: a short entered in January with 65 units was stopped for −$190k. The half-life gate is open on 90–100% of days, so it does not screen anything out.
- Daily P&L is **−0.18 correlated with the trend and carry book**, and +0.21 with the registered dollar strategy.

**For a trade:** a series can be stationary and still not pay. A 35–50 day half-life, a moving anchor and a few crisis stop-outs leave a Sharpe indistinguishable from zero. HO/RB is the one to watch.

---

## Does the horizon rescue it? A slower version, registered first

![Variance ratio against holding period, and Sharpe by decision interval](deck_figures/14_horizon_vs_hold.png)

- The ratio reverts at **3 to 12 months** (variance ratio 0.66, 0.63 and 0.77 at 63 days, 0.50, 0.41 and 0.48 at one year). The daily strategy held for only **31, 23 and 26 days** on average, where the ratios are close to random walks (0.84, 0.73 and 1.11 at 21 days).
- So we registered one change on 4 November, before any result: decide **monthly** instead of daily. Everything else is as registered, and the daily version re-run in the same code reproduces the earlier result.
- **It made things worse.** Book Sharpe −0.13 against +0.06 daily, on 22 trades instead of 71. It fails G1, G3, G4 and G5. HO/RB, the best spread in the daily version, falls from 0.38 to −0.27.
- **14 of 22 trades ended on the time stop**, and only 3 reached take-profit. The time stops lost $125k in total.
- **A limit:** the time stop is 3 times the half-life fitted at entry (13 to 49 days), so average holds were 52, 106 and 36 days. A full 6 to 12 month hold is still untested.

**For a trade:** slowing the decisions does not help here. Waiting for the gap to close mostly ends in a time stop, not a profit.

---

## Should we test a 12-month hold? Why we stop here

**The case for.** The variance ratios fall with the horizon, the 63-day forward slopes are larger than the 21-day ones, and a slower strategy is cheap to build.

**The case against, which we find stronger:**

- **The evidence is a point estimate.** No log-ratio variance ratio is significant, and we saw it before writing the rules.
- **The sample cannot decide.** There are about 17 non-overlapping years. At the best Sharpe seen anywhere (HO/RB daily, 0.38) it takes about **28 years for t = 2 and 62 years for t = 3**.
- **Forking paths.** This is the **sixth** strategy variant tried on one sample. A seventh, tuned to a horizon we already saw, could not be told apart from luck, and the bar would have to rise (we set t above 3.0 before a pass is called evidence).
- **Longer holds meet more regime breaks.** The two largest losses here are HO/RB in 2026 (−$147k on one trade) and Brent/WTI in 2011 (−$74k).
- **Why we did not do it first.** On back-adjusted dollar changes the variance ratio was about 1 at every horizon, so there was no slow reversion to match. It appeared only in the log-ratio, after the professor's feedback.

**What would change the answer:** more independent bets (more spreads or markets) to raise the power, or tracking the fixed rules forward on new data. More tuning on these three spreads cannot.

---

## Trend and carry: a near miss

| Net of costs, 2010–26 | G1 Sharpe | G2 changes | G3 eras | G4 settings | G5 crises |
|---|---|---|---|---|---|
| Trend | 0.29 ✗ | 971 ✓ | 3 ✓ | ✓ | ✓ |
| Carry | 0.44 ✗ | 179 ✓ | 2 ✓ | ✓ | ✓ |
| **50/50 book** | **0.45 ✗** | 1,150 ✓ | 3 ✓ | ✓ | ✓ |

![Cumulative net P&L of trend, carry and the 50/50 book](deck_figures/08_trend_carry_pnl.png)

- Passes four of five gates, and made money in 2022 and 2026, the years mean reversion lost most (Brent − WTI −$151k in 2022; all three spreads lost in 2026).
- Daily P&L is slightly negatively correlated with mean reversion (−0.10), so it is a natural partner.
- t-statistic 1.83 over 16.7 years: short of the 2 normally wanted.

---

## Where the money came from

![Net P&L of the 50/50 book by calendar year](deck_figures/09_annual_pnl.png)

- Gains arrive in bursts: the 2014–15 price collapse, 2021–22 and 2026.
- 2023, 2024 and 2025 all lost money (−$145k in total), and the book spent 1,089 trading days below its June 2022 peak.
- **Costs** take 39% of trend's gross profit and 10% of carry's. Carry is cheaper to hold, because it changes sign rarely.

**For a trade:** this is a strategy that needs patience through multi-year dry spells, and the evidence for it is a few large episodes.

---

## Why the work is worth having, even without a passing Sharpe

- **It stopped us trading a failed idea.** Mean reversion on Brent − WTI would have lost $224k, mostly in 2022 and 2026.
- **It explains the failure.** Levels drift, links break in crises, and any reversion is slower than the holding period.
- **It tested the obvious rescue.** Matching the strategy to the horizon where the ratio reverts did not help, and the reasons for not going further are on the record.
- **It found what survives.** Carry earns most of the Sharpe at a tenth of the cost drag of trend.
- **It kept us honest.** Trend's 504-day lookback scores 0.52, above the bar. Picking it after seeing the table would be overfitting, so it is not counted.
- **It is reusable.** The data pipeline, pre-registered gates, seven tests and the reporting checks run on any new data.

---

## Recommendation

**Paper trade the 50/50 trend and carry book on fixed rules. Do not commit real capital yet.**

- Net Sharpe 0.45 misses the 0.5 bar, t-statistic 1.83, and 48% of the profit came in 2026.
- All nine positions are long at the 26 October close, so today the book is effectively one long-oil bet.
- Next experiments, each registered before it is run:
  1. Trade less (weekly, or only when the signal flips), to cut trend's 39% cost drag.
  2. Overlay a small HO/RB ratio sleeve (stand-alone Sharpe 0.38, t = 1.55, daily P&L correlation −0.18 with the book), as a registered test of whether it adds to trend and carry. It is fragile: the same spread falls to −0.27 when decisions go monthly.
  3. Track the fixed rules forward on new data, the only test free of hindsight.

---

## Caveats

- **Chosen after seeing results.** Trend and carry came from the literature, but were picked after mean reversion failed, so they are labelled exploratory. The ratio strategy followed the professor's feedback, and its rules were registered after we had seen the forward-return regressions on HO/RB, so even a pass would not be independent.
- **Six looks at one sample.** Mean reversion, pairs, trend, carry, ratio mean reversion and its slower version were all tried on the same 16.7 years, the last three after seeing earlier results. Any single pass would have to be discounted for that.
- **KPSS and a moving anchor.** KPSS rejects for every spread, and the era means of the log-ratio shift. We read this as "anchored around a moving k", and the test cannot prove it.
- **Few, related instruments.** All nine instruments are oil products and move together. Published trend and carry results draw on many unrelated markets.
- **Simplifications.** Flat $0.05/bbl costs, no market impact, carry measured at the front of the curve only, unrounded contract sizes.
- **Test power.** About 200 monthly points cannot reliably separate a slow reversion from none.

---

## Summary: test, finding, economics, consequence

| Test | Finding | Economic reading | Consequence |
|---|---|---|---|
| ADF, PP, KPSS, Zivot–Andrews | Dollar gaps need one difference, log-ratios are stationary around a moving mean | The anchor is a ratio k, and k shifts between eras | Trade the ratio against a rolling mean, never a fixed one |
| Variance ratio | Log-ratios 0.4–0.5 at one year, outrights 1.1–1.5 | Two horizons | Mean reversion months, trend a year; but a monthly-decision version fails (−0.13) |
| Cointegration | Link in 8–11% of windows | Arbitrage works, but regimes shift | Reversion needs a stability check |
| ACF, PACF | 0 of 280 lags survive | Markets absorb yesterday's news | No daily forecasting |
| AIC, BIC, Ljung–Box | AR(0) for 6 of 7 | No short-term memory to exploit | Drop AR and SARIMA |
| ARCH, GARCH | Volatility clusters | Shocks arrive in bursts | Size by forecast volatility |
| Gates on strategies | Mean reversion fails (dollar gap and ratio), trend and carry near-miss | Reversion exists but is too slow and breaks in crises; curve and shock persistence pay | Paper trade trend and carry |

---

# Appendix

Two follow-up checks: whether the conclusions change at a monthly horizon, and direct tests of the trend and carry signals.

---

## A1. Horizon check: do monthly changes tell a different story?

**Why.** The daily tests look at lags of up to 40 days. A 12-month trend signal acts at a longer horizon, so we repeat the ACF and PACF on monthly changes.

![ACF and PACF of monthly changes](deck_figures/A1_acf_pacf_monthly.png)

- Month-end back-adjusted values, January 2010 to September 2026 (200 changes, band ±0.139).
- Lag 1: crack +0.15, Brent − WTI +0.13, HO − RB −0.20. Others scatter across lags 5 to 24.
- About 5% of 144 bars (about 7) would fall outside the band by chance.

---

## A2. Is it AR(1), AR(4), or neither?

| Monthly changes, month-end | Lag-1 ACF | AR(1) φ p-value | BIC picks | AIC picks |
|---|---|---|---|---|
| 3-2-1 crack | +0.15 | 0.087 | AR(0) | AR(5) |
| Brent − WTI | +0.13 | 0.196 | AR(0) | AR(6) |
| HO − RB | −0.20 | 0.103 | AR(1) | AR(4) |

- **AR(4)** comes from AIC and only lag 1 is ever significant in those fits. It has no economic story: why would a spread depend on the move four months ago?
- **AR(1)** is not significant for any spread at 5%.
- **Averaging creates echoes.** If you average daily prices into a monthly mean, a pure random walk gets a lag-1 autocorrelation of about +0.25 (simulated: 0.24). Monthly means here give +0.20, +0.31 and +0.23. Month-end sampling avoids this.
- HO − RB's negative lag 1 is short-term reversal of changes, not a stable mean for the level.

---

## A3. Are the three spreads random walks?

| Month-end levels, 2010–26 | ADF p | KPSS p | Half-life (months) |
|---|---|---|---|
| 3-2-1 crack | 0.989 | 0.010 | none |
| Brent − WTI | 0.029 | 0.010 | 26 |
| HO − RB | 0.844 | 0.065 | 25 |

- **Economics says anchored in the long run**: refiners, shippers and storage hold gaps in range.
- **The data say the pull is too slow to trade**: half-lives of 25 months or more, and none for the crack. Over 2007–26 the half-lives are 97 months for Brent − WTI and 34 for HO − RB.
- Brent − WTI rejects a unit root on ADF but KPSS also rejects stationarity, so the tests disagree.
- **Reading:** close to a random walk at trading horizons. This supports the failed mean-reversion gates, but does not prove them: about 200 monthly points is low power, and the backtest uses daily data.
- **Month-end log-ratios of market prices (2007–26):** ADF p is 0.000 for the crack, 0.001 for Brent/WTI and 0.251 for HO/RB (0.000 once calendar-month effects are removed). Brent/WTI passes all three tests (KPSS p ≥ 0.10), the other two fail KPSS.

---

## A4. Direct test of trend: do moves persist?

**In plain English.** The variance ratio asks whether a 12-month move is bigger than 12 times a 1-month move. Above 1 the moves feed on themselves (trend). Below 1 they partly undo themselves.

| Month-end, 2007–26 | Variance ratio (12 m) | p-value | Trend Sharpe, before costs |
|---|---|---|---|
| WTI | 1.17 | 0.56 | 0.03 |
| Brent | 1.32 | 0.32 | 0.27 |
| RBOB | 0.99 | 0.96 | 0.04 |
| ULSD | 1.37 | 0.27 | 0.37 |
| 3-2-1 crack | 1.11 | 0.69 | 0.42 |
| Brent − WTI | 0.94 | 0.85 | 0.37 |
| HO − RB | 0.71 | 0.47 | −0.08 |

- Brent and ULSD lean toward persistence, HO − RB leans toward reversal. **None is significant.**
- These are on back-adjusted dollar changes. On held log returns the outrights at one year are Brent 1.34, ULSD 1.46, WTI 1.11 and RBOB 1.05, also not significant.
- That matches the strategy result: trend is positive but statistically thin.

---

## A5. Direct test of carry: does the curve slope predict returns?

Change in the held contract over the next 21 trading days, regressed on the curve slope (M1 − M2).

| Instrument | Sign of slope: coefficient ($/bbl) | t | Size of slope: coefficient (per 1%) | t |
|---|---|---|---|---|
| WTI | +0.46 | 1.14 | −0.22 | −1.27 |
| Brent | +0.62 | 1.53 | +0.19 | 0.61 |

- **Sign** has the right direction on both (backwardation, then higher returns), but neither clears t = 2.
- **Size** adds nothing, and for WTI it points the wrong way. Only the sign of the slope carries information.
- The strategy uses the sign only, so this is consistent with the rules.
- RBOB and ULSD are outside the registered carry sleeve (seasonal curves). Their results are mixed (RBOB t = −2.7, wrong sign; ULSD t = 2.1).

---

## A6. Is carry timing, or a disguised long?

**Idea.** Split each position into its average (a constant long or short, "passive") and its deviation from that average ("dynamic", the timing). If the dynamic part earns the P&L, the signal is timing.

| Carry sleeve, 2010–26 | Average position (units) | Dynamic share of gross P&L | t of dynamic part |
|---|---|---|---|
| WTI | −0.52 | 101% | 1.7 |
| Brent | +1.35 | 77% | 1.4 |

- WTI's average position is slightly short, yet it earned money, so the P&L is timing and not a long bias.
- Brent has a long bias, but timing still explains about three-quarters of its P&L.
- In an unregistered extension, ULSD carry's dynamic share is 125% (t = 1.95).

<!-- Speaker: an independent run reported 105% / 101% / 76% for ULSD / WTI / Brent with ULSD t = 2.20. WTI and Brent reproduce; ULSD does not exactly. -->

---

## A7. Is it alpha, or oil beta?

Daily sleeve P&L regressed on daily WTI and Brent price changes. The intercept is alpha, what remains after the exposure to oil.

| Book | Raw Sharpe | Alpha t-statistic | Sharpe from alpha |
|---|---|---|---|
| Trend | 0.27 | 0.26 | 0.06 |
| Carry | 0.45 | 1.53 | 0.38 |
| 50/50 book | 0.44 | 1.17 | 0.29 |

- Trend's return is **mostly oil exposure**: its alpha is indistinguishable from zero.
- Carry keeps most of its return after removing oil exposure, though its t-statistic falls from 1.8 to 1.5.
- This weakens the case for trend and strengthens carry's.

<!-- Speaker: rebuilt here, the GARCH refit differs slightly from the committed run, so trend Sharpe is 0.27 against 0.29 committed. The independent run reported 0.29 / 0.33 / 0.09 for trend. Carry matches. -->

---

## A8. What these checks do not show

- **Low power.** Roughly 200 monthly points: a slow reversion or a small trend would not be detected.
- **Different horizons.** Tests on monthly changes and a backtest on daily data answer related, not identical, questions.
- **Several tests, few corrections.** Daily lags are corrected for multiple testing. The monthly lags are not.
- **Unregistered extras.** RBOB and ULSD carry and the beta regressions were run after seeing the results, so they describe, and do not change any verdict.
- **Ratio studies.** The variance ratios, the roll decomposition and the forward-return regressions on log-ratios are descriptive and were run after the other results. No variance ratio on a log-ratio is significant (robust p from 0.15 to 0.98).
