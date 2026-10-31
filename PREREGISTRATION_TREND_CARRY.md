# Pre-registration addendum: trend and carry (exploratory)

**Registered 2026-10-28**, after the binding futures verdict on the mean-reversion strategy (all three spreads FAIL G1–G5) and **before any trend or carry result was computed**. The machine-readable values are in `config.yaml` → `exploratory_trend_carry`. `tests/test_preregistration.py` fails if they change.

**Status: exploratory.** These strategies were chosen after the mean-reversion results were seen. They do not change the pre-registered verdict, and they are reported next to it, never in place of it.

## 1. What was known at registration
- The Bloomberg futures curves (CL, CO to 12 months; XB, HO to 6) and every spot and filter series had been loaded and plotted. Spread levels, regime breaks (2011–13, 2020, 2022, 2026) and the failure of mean reversion were known.
- **Not computed:** any trend signal, carry signal, or their P&L, on any data.
- The choices below come from the literature, not from this data:
  - the 12-month lookback (Moskowitz, Ooi & Pedersen 2012)
  - carry as the front-curve slope (Koijen, Moskowitz, Pedersen & Vrugt 2018)
  - sign-only signals with volatility scaling

## 2. Strategies
| | Strategy 1: trend | Strategy 2: carry |
|---|---|---|
| Instruments | Outrights CL, CO, XB, HO; spreads crack321, brent_wti, ho_rb | Outrights CL, CO |
| Signal at close t | Sign of the back-adjusted change over the past 252 trading days | Sign of M1 − M2 (held month minus next month, unadjusted): + backwardation → long, − contango → short |
| Why these instruments | All the data supports | RBOB and ULSD curves carry a seasonal shape, so their slope is not a clean carry measure |

- **Contracts:** each outright holds its own front month, rolled 5 business days before expiry (as in `futures.py`). The spreads are the registered same-delivery-month spreads from the binding run.
- **Sizing:** units = sign × R / (σ̂ × M), unrounded, recomputed every close.
  - R = \$6,000 a day per instrument.
  - σ̂ is the point-in-time GARCH(1,1)-t forecast of the next daily change, refit monthly on the 756 days before (`readout.garch_path`).
  - M = \$1,000 per \$1/bbl for an outright; for a spread, its registered `usd_per_1bbl_per_unit`.
- **Timing:** decide at close t, fill at close t+1 (as in the registered strategy).
- **Costs:**
  - \$0.05/bbl × M on every unit of position change.
  - 1 tick per contract leg held through a roll (CL/CO \$10, RBOB/ULSD \$4.20; a spread pays |weight| ticks per leg).
- **Books:**
  - Each sleeve's daily P&L is the equal-risk average of its instruments.
  - The combined book is 0.5 × trend + 0.5 × carry.
- **Sample:** the common futures dates. Trading starts at the first month start with a full 756-day volatility window and the lookback available. That is the same start rule as the registered backtest.

## 3. Gates (on the combined book; reported per sleeve too)
| # | Gate | Threshold |
|---|---|---|
| G1 | Net annualised Sharpe | > 0.5 |
| G2 | Position changes (entries and sign flips), summed over instruments | ≥ 20 |
| G3 | Sub-periods with positive net P&L (2007–13, 2014–19, 2020–present) | ≥ 2 of 3 |
| G4 | Net Sharpe > 0 at every trend lookback {126, 252, 504} (carry at base) and every carry slope {M1−M2, M1−M3, M1−M6} (trend at base) | all |
| G5 | Net P&L in 2022 and in 2026 to date | > −\$40,000 in each |

## 4. Reporting rules
- Run once, on the futures data, with these values. Every instrument, sleeve and grid point is reported.
- Descriptive only: the daily-P&L correlation with the registered mean-reversion research stream, and per-instrument Sharpe.
- No parameter may change after a result is seen. Any later change is a new exploratory run with its own date.
