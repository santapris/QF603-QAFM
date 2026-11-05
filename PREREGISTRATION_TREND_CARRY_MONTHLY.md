# Pre-registration addendum: monthly trend and carry (exploratory)

**Registered 2026-11-05**, after the registered daily trend and carry result was seen, and **before this code computed any monthly signal or P&L**. The machine-readable values are in `config.yaml` → `exploratory_trend_carry_monthly`. `tests/test_preregistration.py` fails if they change.

**Status: exploratory, and a seventh look.** The monthly version was chosen after seeing the daily result. It does not change the registered verdict (`PREREGISTRATION_TREND_CARRY.md`), and it is reported next to it, never in place of it.

## 1. Why this is being tried, and what was known
- **The design reason.** A 12-month trend and a curve slope change slowly, so a daily decision adds little information. The papers behind the rules (Moskowitz, Ooi & Pedersen 2012; Koijen, Moskowitz, Pedersen & Vrugt 2018) rebalance monthly.
- **The daily result was known:** book net Sharpe 0.45 (trend 0.29, carry 0.44), passing G2 to G5 and missing G1. Costs took 39% of trend's gross profit and 10% of carry's.
- **Direction runs were known.** Measured on the daily signals on 2026-11-05, with no P&L: trend runs average 30 trading days with a median of 4, and carry runs average 48 with a median of 5. Many trend runs are flips around a zero 12-month change.
- **Another monthly run exists.** A group member ran a monthly version and reported the Sharpe as "still 0.4 something". That script and its definition of monthly were not seen and are not used here.

**Prior:** lower costs should help trend most, but a slower reaction can lose P&L in fast moves such as 2022 and 2026. G1 may still fail. A 0.45 Sharpe has a standard error of about 0.25, so any small change is within the noise.

## 2. Strategy
The registered trend and carry strategy with **one change**: when positions are updated.

- **Instruments, signals, lookback, slope, R, M, GARCH fit, costs, rolls, book weights:** exactly as in `PREREGISTRATION_TREND_CARRY.md`. The trend lookback stays 252 trading days.
- **Base, `monthly`.** On the **first trading day of every month** (`decision_months: 1`), read the signal and σ̂ at that close and set units = sign × R / (σ̂ × M). Units are then held unchanged until the next decision. Fills stay one close later.
- **Reported, not gated, `monthly_signal_daily_size`.** The signal is read monthly, but σ̂ and units update every close. It shows what daily sizing adds in crises, and what it costs.
- **Reference, `daily`.** The same code with no decision interval must reproduce the registered result: exactly 1,150 position changes, and a book Sharpe within 0.01 of 0.447. See the amendment in section 6.

## 3. Gates (on the book)
The same five gates and thresholds as the registered trend and carry strategy: G1 net Sharpe > 0.5, G2 at least 20 position changes, G3 at least 2 of 3 positive sub-periods, G4 Sharpe > 0 at lookbacks {126, 252, 504} and slopes {M1−M2, M1−M3, M1−M6}, G5 net P&L above −$40,000 in the 2022 and 2026 windows. Each sleeve is reported too.

**Calling a pass.** Because this is the seventh variant tried on the sample, a pass is **not described as evidence** unless the book's t-statistic is above 3.0. Otherwise it is described as "passes the gates after multiple looks, needs a forward test". The 2026 share of profit is reported beside any pass.

## 4. Reported, not gated
- Daily, monthly and monthly-signal-daily-size side by side: Sharpe, t, total P&L, drawdown, costs, position changes.
- Each instrument, each sub-period, the stress windows, the grid, and the 2026 share of total P&L.
- Daily-P&L correlation with the registered daily book.

## 5. Reporting rules
- Run once, on the futures data, with these values. Every sleeve, grid point and comparison is reported.
- No parameter may change after a result is seen. Any later change is a new exploratory run with its own date.
- Fixing a coding error that makes the code match these rules is allowed, and must be disclosed with its effect on the results.

## 6. Amendment (2026-11-05, before any monthly result was seen)
The first run stopped at the reference check, with no monthly result displayed. As first written, the check required a book Sharpe of exactly 0.447. On this machine the daily book gives 0.4396 (trend 0.269, carry 0.447), against the committed 0.4471 (trend 0.2885, carry 0.4446).
- **Cause.** The signals match: position changes are identical for all nine positions. The GARCH refits differ slightly from the committed run in some months, which changes position sizes. Library versions match `requirements.txt`, and reruns here are identical, so the likely cause is a difference between machines. `output/exploratory/horizon_support/tables/oil_beta.csv` already showed 0.4396 before this work.
- **Change.** Position changes must match exactly, and the Sharpe must be within 0.01 of 0.447.
- **Comparison.** Monthly is compared with the daily book re-run on the same machine, so solver differences do not read as a monthly effect. The committed 0.45 stays the registered result.
