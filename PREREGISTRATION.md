# Pre-registration: oil complex relative value

The machine-readable values are in `config.yaml`, in the `gates`, `strategy`, `model`, `spreads`, `regime`, `sizing`, `binding_data` and `variants` blocks.
`tests/test_preregistration.py` fails if any registered value changes.

Everything below was registered on **2026-10-24, in Part 0**. At that point no data had been downloaded, no diagnostic had been computed and no backtest had been run.

## 1. Success gates

A traded spread **passes** only if **all five** gates hold. Gates are judged:
- on the `fixed` variant
- at base parameters
- with the logit filter on (the `logit_filter` ladder step)
- on the research P&L stream
- at $0.05/bbl cost
- on the binding data source (§5)

| # | Gate | Threshold |
|---|---|---|
| G1 | Net annualised Sharpe | > 0.5 |
| G2 | Round trips | ≥ 20 |
| G3 | Sub-periods with positive net P&L (2007–13, 2014–19, 2020–present) | ≥ 2 of 3 |
| G4 | Entry-threshold stability: net Sharpe at every entry \|z\| ∈ {1.5, 2.0, 2.5} | > 0 |
| G5 | Stress filter: in 2022 **and** in 2026 to date, filtered net P&L ≥ unfiltered net P&L **and** > −$40,000 | both years |

G1–G4 match Project 2, except G2. The oil sample is shorter, so G2 is lowered from 30 to 20 round trips. G5 is new. The plan says the test that matters most is whether the filter kept the strategy out of 2022 and 2026, and G5 makes that a pass/fail rule rather than a talking point.

The verdict is given **per spread**: `crack321`, `brent_wti`, `ho_rb`. A spread that fails is reported as failing; it is not dropped from the report.

## 2. Base rules (all spreads)

| Rule | Setting |
|---|---|
| Estimation | 756-trading-day window. Refit on the first trading day of each month, and whenever flat after an exit. |
| Fair value | Fixed-weight spread minus in-window month-of-year means (seasonal spreads only). Then the window mean and sd give z. |
| Entry gate | AR(1) half-life of the deseasonalised spread in [5, 120] days at the latest refit. ADF and cointegration p-values are logged but **not required**. |
| Entry | 2.0 ≤ \|z\| < 4.0, re-armed (\|z\| has been back inside 2.0 since the last exit), no blackout, and P(revert) ≥ 0.6 when the filter is active. Sell the spread when z > 0, buy it when z < 0. |
| Take profit | \|z\| ≤ 0.5 |
| Stops | \|z\| ≥ 4.0; held longer than 3 × half-life; P(revert) < 0.4; half-life out of band at a scheduled monthly re-test |
| Blackout | No new entry within 1 trading day before an OPEC/OPEC+ meeting or a Gulf Coast hurricane landfall |
| Timing | Decide at close t, fill at close t+1. Every feature must be known at close t. |
| Costs | $0.05/bbl of spread per unit of position change. On futures, 1 tick per leg rolled is added. |
| Sizing (research stream, gated) | Units = R / (σ̂ × M), unrounded, with R = $6,000 a day. Size is fixed at entry. |
| Sizing (integer book, reported) | N = min(⌊R/(σ̂·M)⌋, ⌊L/((4 − \|z\|)·σ_w·M)⌋), capped at 3, with L = $40,000. Sizes are halved after a 10% book drawdown, and trading stops after 20%. |

## 3. Spreads

| Spread | Definition ($/bbl) | Legs per unit | M ($ per $1/bbl per unit) | Seasonal adj. |
|---|---|---|---|---|
| `crack321` | (2·42·RB + 42·HO − 3·CL)/3 | short 3 CL, long 2 RB, long 1 HO | 3,000 | yes |
| `brent_wti` | Brent − WTI, same delivery month | long 1 Brent, short 1 CL | 1,000 | no |
| `ho_rb` | 42·(HO − RB) | long 1 HO, short 1 RB | 1,000 | yes |
| `brent_m1m6` | Brent M1 − M6 | not traded | n/a | filter input (futures only) |

Prices are always in levels, never logs, because WTI settled negative on 20 Apr 2020. That week is flagged and kept in the core run. A robustness run drops it.

## 4. Variants and ladder

### `fixed` (gated)
- Covers all three traded spreads, at the fixed weights in §3.
- This is the base, and it is the only variant judged against G1–G5.

### `johansen` (robustness, not gated)
- `crack321` only. The weights are the first Johansen cointegrating vector on (CL, RB, HO) levels, refit monthly in the same window and scaled to 3 CL.
- Everything else is identical to `fixed`. It answers whether estimated refinery yields beat the 3:2:1 convention.
- Both the signal and the research-stream P&L use the Johansen-weighted spread, with legs held in unrounded contract amounts. The two variants are reported side by side.

### Ladder (reported for every spread and variant)

| Step | Sizing | Blackout | Logit filter |
|---|---|---|---|
| core | window sd of ΔS | no | no |
| + GARCH sizing | GARCH(1,1)-t σ̂ | no | no |
| + blackout | GARCH | yes | no |
| + logit filter (**gated**) | GARCH | yes | yes |

## 5. Binding data rule

- The verdict is **binding on futures**: Bloomberg CL, CO, XB and HO curves, rolled 5 business days before expiry, with Brent matched to WTI by delivery month and additive back-adjustment.
- Backtests run on FRED/EIA spot data before the Bloomberg export arrives are a **spot proxy**:
  - They are run with the identical frozen config.
  - Every table carries the label.
  - **No registered value may change after a spot result has been seen.** Any change is exploratory (§7).
- If futures data never becomes available, the spot-proxy verdict stands. It is reported as "spot proxy", never as the binding result.

## 6. Regime filter

- **Episode:** a first crossing into 2 ≤ |z| < 4, with the parameters frozen on the signal day. Episodes never overlap.
- **Label:** 1 if |z| ≤ 1 within 20 trading days before first touching 4, else 0. A label counts as resolved on the hit day or at t + 20.
- **Model:** a pooled logit across the three traded spreads, with spread dummies.
  - It is refit at each month start on episodes resolved **strictly before** that date.
  - With fewer than 20 resolved episodes, no filter applies.
- **Core features** (fixed now):
  - OVX percentile
  - GARCH σ̂ percentile
  - rolling 756-day ADF p-value of the spread
  - daily GPR (lagged 1 day)
  - US inventories against their 5-year seasonal average (stamped on the Thursday after the week-end)
- **Robustness features** (fixed now, reported separately, never gated):
  - CFTC managed-money net z-score
  - hurricane-season dummy
  - event within 3 trading days
  - Brent M1−M6 slope as a share of price (futures only)
- **Robustness model:** probit.
- **No feature may be added or removed** after any 2022 or 2026 result has been seen.

## 7. Reporting rules

These are binding for every spread and variant:
1. Every spread and every variant is reported side by side in each table and figure:
   - metrics (Sharpe, net P&L, max drawdown, hit rate, average hold against half-life, worst trade, share of P&L from the five best trades, trade count)
   - sub-periods and stress windows
   - the trade log with exit reasons
   - the ladder
   - the entry × exit grid, cost sweep, window and label-horizon robustness
2. Each spread gets its own G1–G5 verdict. A failing spread is never dropped, and no spread is presented as "the strategy" without the others.
3. Multiple testing: three spreads and two variants were tested. The report says so and gives each Sharpe's t-stat (Sharpe × √years).
4. Any rule change made **after** a backtest has been seen is labelled **exploratory**. This covers a new gate, threshold, window, feature, fair-value model or spread. Exploratory results are reported separately and cannot be used to claim a gate pass. The hazard-model time stop (Part 9) is exploratory by construction.
5. Results based on spot data always carry the "spot proxy" label (§5).

## 8. What was known at registration

Being transparent about what was seen before these values were fixed:

- **From `oil-rv-plan.md`:** the 2026 regime is extreme.
  - The 3-2-1 crack averaged about $65/bbl in early October 2026, against a long-run median near $19.
  - The diesel crack was about $113/bbl on 20 Oct 2026 spot.
  - Brent–WTI spot swung from $35 to $7 in October 2026.
  - CME and The Trading Tools figures show that fading the crack in June 2026 would have lost about $26/bbl by mid-August.
- That knowledge of 2026 outcomes shaped **G5**. It is a deliberately harder bar, not an easier one: a filter that ignored 2026 would fail it.
- **From Project 2:** a formal Engle–Granger gate left almost no tradeable months (14 of 405), and EG has low power at typical half-lives. So cointegration is not an entry gate here, and the half-life band is the only one. The half-life gate is known to be permissive, which leaves most of the protection to the stops and the filter.
- **Not seen:** no oil price, spread, diagnostic, z-path, episode label or P&L had been computed or downloaded for any parameter set. `data/` and `output/` were empty.

## 9. Known weaknesses, stated in advance

- **Few stress episodes.** 2011–13, 2020, 2022 and 2026 provide few filter labels, so the logit is easy to overfit even with five features. That is why the filter is pooled across spreads and features are frozen now.
- **The first sub-period is short.** With a 756-day window from 2007, backtests start around early 2010, so 2007–13 effectively means 2010–13.
- **Spot vs futures.** Spot and futures cracks differ, most of all during physical dislocations like 2026. The spot proxy may disagree with the binding futures run.
- **3:2:1 is a convention.** Real refinery yields differ and drift; the `johansen` variant measures this but is not gated.
- **Historical blackouts.** Hurricane blackouts use actual landfall dates, because historical landfall *forecasts* are not available point-in-time. This is slightly optimistic.
