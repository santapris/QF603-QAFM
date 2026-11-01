---
title: "Exploratory: cointegration pairs on oil futures"
subtitle: "Findings for six estimated-hedge pairs, run 31 October 2026"
---

# Summary

**Status: exploratory.** This strategy was chosen *after* the binding mean-reversion verdict (all three spreads FAIL G1–G5 on futures) and the trend and carry result. It does not change that verdict. Its rules were written down in `PREREGISTRATION_PAIRS.md` and `config.yaml` → `exploratory_pairs` before any signal or P&L of this strategy was computed.

**Verdict: FAIL.** The six-pair book earned a net Sharpe of **−0.19** (t = −0.78) over 16.7 years and lost \$39k. It passed only the trade count (G2). No single pair passed, and none counts as evidence.

**What it means.** Estimating the hedge ratio, requiring cointegration, adding four new pairs and removing the back-adjustment bias from the signal did not produce an edge. Costs are not the reason: at zero cost the Sharpe is still −0.14. Half the trades win, but the losers are larger, and stops outnumber take-profits. In this sample, a measured long-run link between two legs did not translate into profitable reversion.

# 1. What was tested

The full rules are in `PREREGISTRATION_PAIRS.md`. In brief:

- **Pairs (y on x):** Brent/CL, HO/RB, RB/CL, HO/CL, RB/Brent, HO/Brent, each on one delivery month. Only the last four are new; the first two use the legs of registered spreads.
- **Fit:** every month, on the prior 756 days of back-adjusted levels, y on a constant, x and a linear trend. The trend is meant to absorb average roll carry.
- **Gate:** Engle–Granger p < 0.10 (with trend) and a half-life of 5–120 days.
- **Trading:** the registered rules, unchanged: enter at 2 ≤ |z| < 4, exit at |z| ≤ 0.5, stop at |z| ≥ 4, time stop at 3 half-lives, next-close fills, \$25 per contract plus roll ticks, GARCH sizing at R = \$6,000 a day per pair, and the event blackout.
- **Book:** the equal-risk average of the six pairs.
- **Data:** Bloomberg futures. Trading runs from 1 Feb 2010 to 26 Oct 2026.

# 2. Results

## 2.1 Gate verdicts

| Stream | G1 Sharpe > 0.5 | G2 ≥ 20 trades | G3 ≥ 2 of 3 periods | G4 grid all > 0 | G5 2022 and 2026 > −\$40k | Verdict |
|---|---|---|---|---|---|---|
| **Book** | −0.19 ✗ | 80 ✓ | 1 ✗ | ✗ | −\$40.3k, −\$12.0k ✗ | **FAIL** |
| Brent/CL | −0.14 ✗ | 11 ✗ | 1 ✗ | ✗ | −\$31.5k, −\$41.9k ✗ | FAIL |
| HO/RB | −0.13 ✗ | 16 ✗ | 1 ✗ | ✗ | −\$71.4k, \$0 ✗ | FAIL |
| RB/CL | −0.22 ✗ | 14 ✗ | 0 ✗ | ✗ | −\$10.3k, −\$22.4k ✓ | FAIL |
| HO/CL | −0.37 ✗ | 7 ✗ | 1 ✗ | ✗ | −\$94.8k, \$0 ✗ | FAIL |
| RB/Brent | 0.09 ✗ | 19 ✗ | 2 ✓ | ✗ | −\$45.8k, −\$7.4k ✗ | FAIL |
| HO/Brent | 0.08 ✗ | 13 ✗ | 2 ✓ | ✗ | +\$12.0k, \$0 ✓ | FAIL |

The book's G5 miss in 2022 is narrow (−\$40.3k against −\$40k), but G1, G3 and G4 fail by clear margins, so it does not affect the verdict. The largest pair t-statistic is 0.35, far below the 2.64 Bonferroni bar.

**G4, book Sharpe at each setting:**

| Entry \|z\| | 1.5 | 2.0 | 2.5 | | EG p | 0.05 | 0.10 |
|---|---|---|---|---|---|---|---|
| Sharpe | −0.24 | −0.19 | −0.38 | | | −0.18 | −0.19 |
| Trades | 97 | 80 | 60 | | | 61 | 80 |

## 2.2 Over time

| Period | Book net P&L |
|---|---|
| 2007–13 (trading from 2010) | −\$2.2k |
| 2014–19 | +\$2.0k |
| 2020–26 | −\$39.1k |

- **The book peaked on 23 Feb 2022 and has not recovered.** It was \$71.9k below that peak on 22 Oct 2026. It lost \$40.3k in 2022, its worst year, and five of the six pairs lost that year.
- **The gate did not screen out 2022,** as expected at registration. It was open 100% of 2021 and 2022 for RB/CL, and 100% and 98% for RB/Brent.
- **The gate was shut for the whole of 2024,** and the book made no trades in 2024 or 2025.

## 2.3 Comparisons (reported, not gated)

| Signal | Book Sharpe | t | Trades | Net | 2022 | 2026 |
|---|---|---|---|---|---|---|
| **Base: adjusted with trend, gated** | −0.19 | −0.78 | 80 | −\$39k | −\$40k | −\$12k |
| Adjusted, no trend | −0.01 | −0.06 | 60 | −\$3k | −\$2k | −\$11k |
| Market-level signal | −0.58 | −2.37 | 55 | −\$89k | −\$29k | −\$14k |
| No Engle–Granger gate | 0.17 | 0.70 | 233 | +\$79k | −\$57k | −\$27k |

- **The trend term did not help.** It did nothing for returns compared with no trend at all (−0.19 against −0.01), so the signal fix I expected to matter was not the missing ingredient.
- **Market-level signals were clearly the worst,** consistent with the pre-run concern that they read roll jumps as reversion.
- **The cointegration gate did not add value.** Removing it took the Sharpe from −0.19 to 0.17. That is not a discovery: t = 0.70, it would still fail G1 and G5, and it was found by looking. It does not justify dropping the gate and trying again.

**Robustness:**

| Setting | Value | Book Sharpe | Trades |
|---|---|---|---|
| Window (days) | 504 | 0.04 | 99 |
| | 756 (base) | −0.19 | 80 |
| | 1,008 | −0.41 | 60 |
| Cost (\$/bbl) | 0.00 | −0.14 | 80 |
| | 0.025 | −0.16 | 80 |
| | 0.05 (base) | −0.19 | 80 |
| | 0.10 | −0.24 | 80 |
| | 0.20 | −0.34 | 80 |

# 3. Why it lost

Dollar figures in this section are at each pair's full R = \$6,000 and are summed across pairs, so they are six times the book's scale.

- **Losers outweigh winners.** 40 of 80 trades made money, but the average trade lost about \$2,950. Exits: 29 stop-losses, 23 take-profits, 14 model breaks, 14 time stops.
- **Shorts lost most of the money.**

  | Direction | Trades | Net | Win rate |
  |---|---|---|---|
  | Short the spread | 47 | −\$290k | 49% |
  | Long the spread | 33 | +\$54k | 52% |

  The win rates are similar, so the gap comes from the size of the losses. The worst trade was a −\$94.8k HO/CL short in 2022. The result is not uniform: HO/Brent shorts made \$168k while its longs lost \$128k. The short share fell to 59%, from 71% in the registered run, but on a different set of spreads.
- **Convergence itself lost money.**

  | Pair | Gross | Fitted carry | Convergence |
  |---|---|---|---|
  | Brent/CL | −\$22k | +\$9k | −\$31k |
  | HO/RB | −\$52k | +\$100k | −\$152k |
  | RB/CL | −\$67k | −\$10k | −\$56k |
  | HO/CL | −\$141k | +\$8k | −\$150k |
  | RB/Brent | +\$61k | +\$23k | +\$38k |
  | HO/Brent | +\$55k | −\$1k | +\$56k |
  | **All** | **−\$165k** | **+\$129k** | **−\$294k** |

  Costs take the total from −\$165k gross to −\$236k net. Only the two Brent-product pairs have positive convergence, but with t-statistics near 0.3 that is not evidence. Treat this split as indicative: "carry" here is the fitted trend extrapolated forward, not measured roll yield.
- **The gate has limited power.** On simulated pairs with 756-day windows (run after registration and before the backtest; 300 draws each), the test opens 100% of the time for a true 10-day half-life, 66% at 25 days, 27% at 50 days and 14% at 100 days, against 10% for no relationship. So a closed gate often does not mean there is no relationship, and an open one is weak confirmation when half-lives are long.
- **Hedge ratios move.** At the latest fit, HO/CL has β = 2.50 and HO/Brent β = 2.19, against full-history medians of 1.06 and 1.07.

# 4. Other descriptive results

- **Correlation.** Daily book P&L correlates 0.02 with the registered mean-reversion stream and −0.13 with the trend and carry book. Low correlation does not help a book with a negative return.
- **Current state, 26 Oct 2026.** All six pairs are flat. Brent/CL has z = 3.31 with its gate open, but it was stopped out on 23 Oct, and z has not come back inside 2, so it is not re-armed. The other five gates are shut.

## 4.1 Residual checks (added after the results; not registered, not a gate)

`scripts/run_residual_checks.py` runs Ljung–Box at lags 10 and 20 on every monthly fit. Each residual is rebuilt from a stored fit and must reproduce that fit's AR(1). Plain Ljung–Box assumes constant variance, which these series do not have, so a version robust to volatility clustering is shown beside it. The table gives the share of windows rejecting at 5%, at lag 10, with lag 20 in brackets.

| AR(1) innovations, futures | Plain | Robust |
|---|---|---|
| 3-2-1 crack | 20% (15%) | 2% (0%) |
| Brent−WTI | 64% (69%) | 6% (6%) |
| HO−RB | 33% (29%) | 3% (9%) |
| Brent/CL | 69% (75%) | 7% (7%) |
| HO/RB | 35% (35%) | 3% (12%) |
| RB/CL | 37% (41%) | 1% (0%) |
| HO/CL | 27% (32%) | 1% (2%) |
| RB/Brent | 23% (21%) | 0% (0%) |
| HO/Brent | 25% (38%) | 4% (2%) |

**Benchmark** (`output/tables/04_ljung_box_size_power.csv`), simulated with volatility clustering like these fits: a true AR(1) is falsely rejected 32–50% of the time by the plain test and 4–7% by the robust one. For missed dynamics, the robust test catches φ₂ = −0.20 in 54–89% of samples, φ₂ = −0.10 in 15–30% and φ₂ = −0.05 in about 10%.

- **The plain rejections are mostly volatility clustering.** Most are within or below the plain test's false-positive range. Only Brent−WTI and Brent/CL sit above it.
- **AR(1) looks adequate at short lags.** Robust rates are close to what a correct AR(1) gives, apart from HO/RB at lag 20 (12%). Brent−WTI and Brent/CL stand out in 2007–13 (26–28%), the period of the Cushing break. Windows overlap, so these shares are not independent.
- **This does not validate mean reversion.** A random walk also has white-noise AR(1) innovations, with b near 1. The check says the AR(1) form is not obviously wrong, not that the series reverts. It leaves the conclusion above unchanged.
- **The test is weak against small misses,** so modest extra dynamics cannot be ruled out.
- **The pair residual itself rejects in every window, on both tests.** That is what any persistent series does, whether it reverts or not, so it carries no information about reversion.
- **Spot differs.** On the spot proxy, Brent−WTI rejects in 91–92% of windows even on the robust test, and the crack in 12–14%. This was not investigated further; spot is not the binding data.

# 5. Limits of this test

- **Not fully independent.** The registered results and a rough check on 1:1 Brent–WTI and HO−RB were known before registration. Brent/CL and HO/RB are not fresh tests.
- **Not out of sample in the strict sense.** Every date had already been seen in other studies of this data, and the regime breaks were known.
- **Few trades per pair.** 7–19 per pair, so per-pair estimates are noisy. The book's 80 trades are overlapping bets on one oil complex, not 80 independent ones.
- **Unrounded hedges.** Fractional hedge ratios cannot be held in 1–3 lots, so there is no integer book, and a real small account would carry extra hedge error.
- **Scale.** The book averages six pairs, so much of the time it runs well below \$6,000 a day of risk. This makes G5 lenient and does not change Sharpe's sign.

# 6. Verification

- **Independent replay.** A separate numpy recalculation of all 80 trades matches β, entry z, gross P&L and costs to rounding error, with no wrong-side, closed-gate, out-of-range or blackout entries.
- **Engine parity.** `PairSim` reproduces `SpreadSim` exactly on the real Brent–WTI data, and matches the registered daily P&L file within \$2e-10 (9 trades, −\$224,379).
- **Data.** `pair_legs` reproduces the registered Brent–WTI and HO−RB legs and roll days to rounding error.
- **Reconciliation.** The book's −\$39,331 is the pairs' −\$235,987 divided by six, and trade totals tie to daily totals for each pair.
- **Reproducibility.** A second run gave byte-identical tables. A third, after a chart-layout fix, also matched. No parameter changed between runs.
- **Registered results.** Checksums of the engine files, the registered outputs and the trend and carry outputs were unchanged by this work. `tests/test_pairs.py` covers the strategy on synthetic data, and the full test suite passes.

**Files:** tables and figures in `output/exploratory/pairs/`, daily paths in `data/derived/exploratory/pairs_daily.csv`.
