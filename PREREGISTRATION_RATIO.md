# Pre-registration addendum: ratio mean reversion (exploratory)

**Registered 2026-11-04**, after the binding futures verdict on the mean-reversion strategy (all three dollar spreads FAIL G1–G5), the exploratory trend and carry result, the exploratory pairs result and the descriptive ratio stationarity study, and **before any signal or P&L of this strategy was computed**. The machine-readable values are in `config.yaml` → `exploratory_ratio`. `tests/test_preregistration.py` fails if they change.

**Status: exploratory.** This strategy was chosen after those results were seen. It does not change the pre-registered verdict, and it is reported next to it, never in place of it.

## 1. What was known at registration
- **The binding mean-reversion results.** The registered strategy traded the back-adjusted *dollar* gap of the three spreads and failed (net Sharpe 0.32, −0.42, 0.06 on 2010–26 futures).
- **The professor's feedback (2026-11-04).** If P_A = k·P_B then P_A − P_B = (k − 1)·P_B inherits the trend in prices, so the ratio, or log P_A − log P_B, should be the stationary object.
- **The descriptive ratio stationarity study** (`report/exploratory_ratio_stationarity.md`, `scripts/run_ratio_stationarity.py`), which used the same 2007–26 sample the strategy will trade:
  - ADF, PP and Zivot–Andrews reject a unit root in the market log-ratios. The dollar differences do not reject with a break.
  - The log slope is 1.009 (Brent on WTI) and 0.991 (HO on RB).
  - Half-lives are about 35–50 days. KPSS rejects everywhere, which at these half-lives it does about half the time on truly stationary data.
  - The era means of the log-ratio step between regimes, so the anchor moves.
- **Forward-return regressions (Step 6 of that study)**, on the same window and z-score this strategy uses: the forward held-contract log-ratio return regressed on today's z, 756-day window, 21 and 63 days.

  | Spread | Slope, 21 / 63 days (t) | Fade after \|z\| ≥ 2, 21 days (approx. t) |
  |---|---|---|
  | HO / RB | −0.016 (−2.78) / −0.030 (−3.51) | +6.2% (1.8) |
  | 3-2-1 crack | −0.008 (−1.72) / −0.011 (−1.53) | +1.9% (0.7) |
  | Brent / WTI | −0.002 (−1.20) / −0.005 (−0.96) | +0.1% (0.1) |

  These are before costs, sizing, stops and gates. **So this is not a clean out-of-sample test.** The signal was tuned by nobody, but the HO/RB result in particular was seen before the rules below were written.
- **Roll jumps.** Market log-ratios jump at rolls (median roll-day move about twice a normal day). Between 22% and 54% of the market reversion in the study is roll jump that a holder does not earn. This strategy reads the market ratio, including roll jumps, so a roll can trigger an entry. That is a known cost of the signal and is not corrected.
- **The pairs result**, which is the closest earlier test: net Sharpe −0.19 on 80 round trips. Its `market` comparison (a market-level dollar signal with fitted hedge ratios) scored Sharpe −0.58.

**Prior.** Failure is the likely outcome for the book. HO/RB is the favourite to pass individually, the crack and Brent/WTI are likely to fail G1. All three are registered with equal standing, and no spread is dropped after the fact.

## 2. Strategy
**Universe.** Three spreads, fixed in advance (the same three as the registered strategy):
- `crack321`: log((2·RB + HO) / (3·CL)), month-of-year effects on.
- `brent_wti`: log(Brent / CL), month-of-year effects off.
- `ho_rb`: log(HO / RB), month-of-year effects on.

All legs are market prices of the held contracts in $/bbl (`*_bbl_unadj` in `data/clean/futures_spreads.csv`), one delivery month per spread, rolled 5 business days before the earlier expiry, on the common futures dates.

**Monthly fit**, on the 756 days strictly before the first trading day of the month, and whenever flat after an exit:
1. Log-ratio L = log(A / B) of the window's market prices, month effects removed inside the window where on, giving R. Mean and sd of R from the window. Half-life = AR(1) half-life of R.
2. **Weights are dollar-neutral at the fit.** With P_A and P_B the market values of the two sides at the last close of the window, the contracts per unit are w_k = +s_A·n_k on side A and w_k = −s_B·n_k on side B, with s_A·P_A = s_B·P_B and gross contracts Σ|w_k| = 2. At M = $1,000 per $1/bbl per unit, the $0.05/bbl cost is $25 per contract, as in the registered spreads.
3. The traded spread S = Σ w_k·(back-adjusted leg k), in $/bbl, so a unit earns approximately the change in the log-ratio times the side notional.
4. **z_t = (R_t − mean) / sd**, where R_t is the day-t market log-ratio net of its month effect. z is read from market prices. **P&L, GARCH state and costs use back-adjusted legs.**

**Gate.** The AR(1) half-life of R in [5, 120] days. There is no Engle–Granger gate, because β = 1 is imposed. The gate is the registered one.

**Trading.** The registered rules, unchanged:
- Enter at 2.0 ≤ |z| < 4.0 when re-armed and the gate is open. Sell the spread when z > 0, buy it when z < 0.
- Exit at |z| ≤ 0.5. Stop at |z| ≥ 4.0. Time stop at 3 half-lives. Exit (`model_break`) if the gate is closed at a monthly re-test.
- Weights and the fair value are frozen while in a trade.
- Decide at close t, fill at close t+1. $0.05/bbl per unit of position change, plus 1 tick per contract leg per roll.
- Units = R / (σ̂ × M), unrounded, with R = $6,000 a day and σ̂ the GARCH(1,1)-t forecast on the fit's frozen-weight daily changes of S.
- No entry within the registered OPEC and hurricane blackout. No logit filter.

**Book.** Equal-risk average of the three spreads' daily research P&L. Research stream only (no integer book).

**Sample.** Trading starts at the first month start with a full 756-day window, as in the registered backtest.

## 3. Gates (on the book)
| # | Gate | Threshold |
|---|---|---|
| G1 | Net annualised Sharpe | > 0.5 |
| G2 | Round trips, summed over spreads | ≥ 20 |
| G3 | Sub-periods with positive net P&L (2007–13, 2014–19, 2020–present) | ≥ 2 of 3 |
| G4 | Net Sharpe > 0 at every entry \|z\| ∈ {1.5, 2.0, 2.5} | all |
| G5 | Net P&L in 2022 and in 2026 to date | > −$40,000 in each |

Each spread is also reported against G1–G5 at its full R = $6,000.

**One spread passing is not a result on its own.** With three spreads, one is counted as evidence only if it passes G1–G5 and clears the Bonferroni bar of t > 2.39 (two-sided 5% over three tests).

## 4. Reported, not gated
- **Comparisons**, same rules otherwise:
  - `dollar_market`: the same dollar-neutral weights and the same window, but z from the market *dollar* spread S_unadj = Σ w_k·(market leg k), with the half-life of that series. This isolates whether the *object* (log-ratio against dollar gap) matters when both are measured on market prices.
  - `no_gate`: no half-life gate.
- **Exit z** 0.25 and 0.75, **estimation window** 504 and 1,008 days, costs of $0, $0.025, $0.10 and $0.20/bbl.
- **Long and short P&L, and exit reasons**, per spread.
- **Roll attribution:** the share of each trade's gross P&L earned on roll days.
- **Daily-P&L correlation** with the registered mean-reversion stream, the pairs book and the trend and carry book.

## 5. Reporting rules
- Run once, on the futures data, with these values. Every spread, grid point and comparison is reported.
- No parameter may change after a result is seen. Any later change is a new exploratory run with its own date.
- Fixing a coding error that makes the code match these rules is allowed, and must be disclosed in the findings with its effect on the results.
- Because the Step 6 regressions were seen first, a pass is described as "passes the gates, not independently confirmed". Only forward tracking on new data is free of hindsight.
