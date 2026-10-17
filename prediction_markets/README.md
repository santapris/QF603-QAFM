# Prediction Market Calibration — Draft Code

Companion code for `Project/Prediction_Markets_Proposal.docx`. Implements
Stage 1 (measurement: calibration curve), Stage 2 (significance: logit
fit), Stage 4 (persistence: does this month's bias predict next month's?),
and Stage 5 (out-of-sample test) end to end. Stage 5 was pulled forward
out of order because it's the one that actually answers whether this
direction is worth pursuing further — see "Stage 5 result so far" below.
Stages 3 and 6 are scoped but not written yet — see
`src/models/roadmap_stages_3to6.py`.

## Current status (read this first)

Stage 1, 2, 4, and 5 all run end to end — but **Stage 4 and the latest
Stage 5 run are both wiring smoke tests, not results**. Every live pull so
far has spanned under a day of `close_time`, which isn't enough distinct
months (Stage 4) or enough time-diversity (Stage 5's `--split time`) for
either to mean what it's supposed to mean. Concretely: Stage 4 correctly
refuses to run at all on this data (`ValueError`, not enough months), and
re-running Stage 5 on a second pull flipped its sign from the
originally-documented result — both are exactly what you'd expect from a
same-day-clustered sample, not a contradiction or a bug.

**Next step, before trusting either stage's output**: pull a
category-filtered or longer-horizon sample (e.g. `extra_params =
{"series_ticker": "..."}` for a slower-resolving category like
Fed-decision/economics markets, in `kalshi_client.py`) so the sample spans
enough genuinely distinct months/weeks. See "Stage 4 result so far" and
"Stage 5 result so far" below for the exact numbers and reasoning.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

```bash
# smoke-test pulls (a couple hundred markets each, fast)
python scripts/pull_kalshi.py --max-pages 2
python scripts/pull_polymarket.py --max-pages 2

# calibration curve + logit fit on whatever's been pulled
python scripts/run_calibration.py --platform both

# Stage 5: does the correction beat the raw price out of sample?
python scripts/run_oos.py --platform kalshi --split fraction --train-frac 0.7

# Stage 4: is this month's average bias predictive of next month's?
python scripts/run_stage4.py --platform kalshi

# unit tests (no network needed -- pure logic on toy data)
pytest tests/
```

## Stage 5 result so far

On a live pull of ~10k recent settled Kalshi markets (70/30 train/test
split): the logit-corrected forecast beat the raw market price
out-of-sample (Brier 0.0695 vs. 0.0776), and both decisively beat the
base-rate benchmark (0.2444) — so the favorite-longshot correction is
*not* purely an in-sample artifact, at least on this sample.

**Caveat worth fixing before leaning on this further**: that pull spanned
under a day of `close_time` — Kalshi's settled-markets feed is dominated
by very high-frequency crypto and sports micro-markets regardless of how
far back you ask (see `src/data/kalshi_client.py` and
`src/models/stage5_oos.py` docstrings for what we found digging into
this). A 70/30 split on that sample is closer to a random split than a
real test of generalizing across market regimes or time. Next step:
pull a category-filtered, longer-horizon sample (e.g. exclude crypto/
sports series, or paginate much further back) and rerun `run_oos.py`
with `--split time` before treating this result as more than "promising
enough to keep building."

**Addendum, not a correction — a second data point that argues for the
same caveat rather than against it**: while building Stage 4 (below), this
exact test was rerun fresh on a new, different ~5k-market live pull and
came out with the *opposite* sign (correction worse than raw price
out-of-sample, not better). Both pulls are short, same-day samples, so
neither should be read as "the" answer — see the Stage 4 section for the
numbers and why this is expected, not alarming, given the caveat already
above.

Drop `--max-pages` once you're ready to pull full history; a full Kalshi
settled-markets pull is tens of thousands of markets, so expect it to take
a while and consider narrowing with `extra_params` in `kalshi_client.py`
(e.g. a `series_ticker` for one category, or `min_settled_ts`/`max_settled_ts`
for a date range) rather than pulling everything on every run.

## Stage 4 — persistence (and the statistical foundations roadmap)

### Method: the full Box-Jenkins cycle, not just "assume AR(1)"

Stage 4 asks whether this month's average favorite-longshot bias predicts
next month's — the same question VRP's own Stage 1 asks of the variance
risk premium. The original one-line idea for this stage was "monthly bias
series -> OLS of bias_t on bias_{t-1}," i.e. assume AR(1) and fit it. What's
actually built in `src/models/stage4_persistence.py` goes further, because
assuming the order is exactly the kind of shortcut that caused a real,
documented problem on the VRP side of this repo (a frozen AR(1) control
that turned out to be missing a second lag, found only months later by a
residual diagnostic nobody had run yet). The pipeline, in order:

1. **Aggregate.** Group every resolved market by the calendar month it
   closed in; `bias_t = mean(implied_prob) - mean(outcome)` that month
   (same sign convention as Stage 1's `bucket_calibration`: positive means
   markets were overpriced that month). Months with too few markets are
   *flagged*, not silently dropped — dropping one would leave a gap in
   what's supposed to be an evenly-spaced monthly series, which would
   corrupt every lag computed after it.
2. **Identify (ACF/PACF).** Look at the series' own autocorrelation and
   partial autocorrelation before deciding how many lags of memory it
   actually has, instead of assuming AR(1) by default.
3. **Check stationarity (ADF).** A series that's trending or wandering
   (no fixed mean to revert to) makes any AR coefficient fit on it
   potentially spurious — nonsense that looks like a real relationship
   purely because both sides of the regression are drifting together.
4. **Fit (AIC-selected AR(p)).** Pick the order via
   `statsmodels.tsa.ar_model.ar_select_order` (the same tool and
   discipline as VRP's own `ar_baseline.py` / `driver_surprises.py`),
   then fit one full-sample OLS regression — not a walk-forward forecast.
   This stage is descriptive (does memory exist at all?), the same way
   VRP's Stage 1 is; Stage 5 above already owns the separate
   out-of-sample question, for the bias-corrected probability forecast.
5. **Diagnose (Ljung-Box, ARCH-LM).** Check the fitted model's residuals
   for leftover structure (Ljung-Box) and for volatility clustering
   (ARCH-LM — does how *noisy* the bias is change over time, e.g. does
   miscalibration get wilder during high-uncertainty periods, the
   discrete-market analogue of VRP's calm/stress regime interaction?).
6. **Difference and re-test, if needed.** If step 3 found the level
   series non-stationary, difference it once (`x_t - x_{t-1}`) and repeat
   steps 2–5 *on the differenced series*. This is the step that's easy to
   skip and shouldn't be — a model fit on a non-stationary series can
   look perfectly reasonable and still be spurious, and the only way to
   know differencing actually fixed it is to re-run the same diagnostics
   again, not just assume one difference is always enough.

Run it: `python scripts/run_stage4.py --platform kalshi` (flags:
`--min-markets-per-month`, `--max-p`, `--min-months`, `--force-difference`,
`--drop-flagged` — see the script's docstring).

### Result so far

Ran the full chain on a fresh live pull (`pull_kalshi.py --max-pages 5`, 5,000 settled
markets, 4,932 usable after dropping unresolved/non-binary rows) to check that
everything wires together end to end. The honest outcome: **this was a wiring
smoke test, not a trustworthy persistence number** — the pull spanned only
23 hours and 23 minutes of `close_time`, which is exactly **1 distinct
calendar month**. `run_stage4_persistence` correctly raised `ValueError`
("only 1 usable month(s) of bias data (min_months=24)") rather than report an
AR coefficient computed on a single data point. **That refusal is the
successful part of this test** — the guard exists specifically to prevent
exactly this failure mode, and it worked.

This also surfaced something worth knowing about the Stage 5 result quoted
above: running Stage 5 again, fresh, on this new 5,000-market pull (same
70/30 fraction split) gave **the opposite sign** from the original ~10k-market
pull — Brier 0.0430 (corrected) vs. 0.0391 (raw price), i.e. the correction
did *worse* out-of-sample this time (`improvement_vs_naive = -0.0039`),
not better. Both pulls span under a day of `close_time`, so both are
closer to a random split of same-day markets than a genuine test of
generalizing across time — and the fact that the sign of "does the
correction help" flips between two quick live pulls is itself evidence for
exactly that caveat, not a contradiction of it. This is the same shape of
lesson as a much-discussed moment on the VRP side of this repo: a number
computed on a short, non-representative window can look however you like
depending on which short window you happen to draw, which is precisely why
Stage 4's `min_months` guard refuses to run at all rather than report
something equally fragile.

**Next step, to get a real answer (not a repeat of this)**: pull a
category-filtered or much longer-horizon sample — e.g. a `series_ticker`
for a slower-moving category (economics/Fed decisions tend to resolve over
weeks to months, unlike crypto/sports micro-markets) via `extra_params` in
`kalshi_client.py` — specifically so the pull spans enough genuinely
distinct months for both Stage 4's persistence test and Stage 5's
`--split time` to mean what they're supposed to mean.

### Statistical foundations roadmap (not implemented this round — a concrete checklist for next time)

This project is also a deliberate place to practice statistical foundations
more broadly; the following are scoped out of *this* round on purpose, not
overlooked, so the next pass has a precise list rather than a vague sense
of "more could be done":

- **Stage 1 (measurement).** A Brier score decomposition (Murphy, 1973:
  reliability + resolution − uncertainty) would separate "how well
  calibrated" from "how discriminating" — right now one Brier number
  conflates both. Isotonic regression is worth trying alongside the Stage 2
  logit as a non-parametric alternative calibration correction, since it
  makes no assumption about the correction curve's shape.
- **Stage 2 (significance).** The bucket chart in Stage 1 shows the
  calibration curve bending away from the 45° line, but nothing yet tests
  *as a whole* whether that bend is statistically significant beyond the
  logit's own coefficients — a Hosmer-Lemeshow goodness-of-fit test would.
  Deviance/Pearson residuals and a link-function test (the logit analogue
  of Ramsey's RESET test) are the other standard GLM diagnostics missing.
- **Stage 3 (drivers, not yet built).** Its own roadmap above already
  flags the open question correctly: markets on the same event are not
  independent draws, and may need cluster-robust standard errors (the
  cross-sectional sibling of VRP's HAC correction — same sandwich-estimator
  idea, different correlation structure). Once several covariates are
  tested at once, a Holm/Bonferroni correction across them becomes relevant
  too, for the same reason VRP's Q2 needed one across 8 predictors × 3
  horizons.
- **Stage 4 (this stage).** Plain OLS standard errors were used for the
  AR(p) fit, not HAC/Newey-West — defensible since this monthly series
  doesn't have VRP's overlapping-window structure, but worth revisiting
  alongside the Stage 3 clustering question above.
- **Cross-cutting.** A hierarchical/partial-pooling model for
  platform × category effects (letting each group's calibration curve be
  shrunk toward a shared one, rather than either pooling everything or
  splitting into fully separate fits) would be a more principled version
  of the "pool vs. split" choice described below. `time_to_resolution_days`
  (already planned as a Stage 3 covariate) could be modeled properly as a
  survival/duration process instead of a single linear control. Stage 6's
  Bayesian updating is the one piece of this whole project with a genuinely
  different inferential philosophy from VRP's entirely frequentist toolkit.

## Kalshi vs. Polymarket — which one, or both?

**Start with one platform to get the pipeline validated end to end, then
add the second as a robustness check — don't pool blindly from day one.**
Recommendation: start with **Kalshi**. It's CFTC-regulated with
unambiguous settlement rules, strong coverage of economic/Fed-decision
markets (continuity with the original VRP project's macro flavor), and the
cleanest public API (documented fixed-point dollar fields, a real
candlestick/history endpoint). Add Polymarket once Stage 1/2 work on
Kalshi alone, for a larger sample and broader category coverage (politics,
crypto, sports).

**Can they be combined?** Yes, but not naively — there are real
differences to account for, not just a schema mismatch:

| | Kalshi | Polymarket |
|---|---|---|
| Settlement | Kalshi itself, by stated contract rules | UMA oracle + dispute window |
| Regulation | CFTC-regulated exchange | Offshore, unregulated |
| Price units | Dollars, $0–$1 (already a probability) | Decimal outcome price, 0–1 |
| Typical fees | Small, price-dependent trading fee | ~0 trading fee, but gas + slippage |
| Category strength | US economics, Fed, politics | Crypto, global politics, pop culture, sports |

`src/data/align.py` harmonizes both into one schema and keeps a `platform`
column specifically so every downstream regression (Stage 3 onward) can
control for platform (or be run per-platform and compared) rather than
assuming Kalshi and Polymarket rows are interchangeable draws from the same
data-generating process. The one thing *not* yet reconciled: `volume` is
on different scales across platforms (contract counts vs. notional-ish
dollar volume) — treat it as within-platform-only until that's fixed, e.g.
by converting to a percentile rank within each platform before using it as
a liquidity control in Stage 3.

No data discrepancy issue is a blocker — it just means "platform" needs to
stay a first-class variable in the analysis, not get pooled away.

## API access notes

Both platforms' **market data** endpoints used here are public — no account,
API key, or auth header required. (An API key would only be needed if you
later want to place real orders, which this project doesn't do.)

- **Kalshi**: `GET https://api.elections.kalshi.com/trade-api/v2/markets`
  — try it directly in a browser or `curl`, no setup needed. Docs:
  https://docs.kalshi.com/api-reference/market/get-markets
- **Polymarket**: `GET https://gamma-api.polymarket.com/markets?closed=true`
  for market metadata; `GET https://clob.polymarket.com/prices-history?market=<token_id>`
  for a price series. Docs: https://docs.polymarket.com/api-spec/gamma-openapi.yaml
  and .../clob-openapi.yaml

APIs do change — if a pull script errors on an unexpected field or 404,
check the live docs link in the relevant client file before assuming the
code is wrong.

## Layout

```
src/data/        kalshi_client.py, polymarket_client.py, align.py
src/measures/    calibration.py       (Stage 1)
src/models/      stage2_logit.py      (Stage 2)
                 stage4_persistence.py  (Stage 4)
                 stage5_oos.py        (Stage 5)
                 roadmap_stages_3to6.py  (Stages 3, 6 -- TODO; Stage 4 implemented)
scripts/         pull_kalshi.py, pull_polymarket.py, run_calibration.py, run_oos.py, run_stage4.py
tests/           unit tests, no network required
data/            gitignored raw pulls land here
```
