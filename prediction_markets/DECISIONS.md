# Decisions log — prediction_markets

Append-only record of why the data source and methodology changed shape
over this project, kept separate from README.md so the README can stay
focused on current best understanding while this file preserves the
reasoning trail (same convention as the VRP project's own `DECISIONS.md`
on `main`). Numbered sequentially; never renumber or delete an entry, only
add corrections that reference the one they correct.

## D001 — General Kalshi pulls can't test persistence: same-day clustering

Two independent general pulls (`pull_kalshi.py`, unfiltered except
`mve_filter=exclude`) — ~10k markets and, later, ~5k markets — each spanned
under a day of `close_time`. Kalshi's settled-markets feed is dominated by
high-frequency sports/crypto micro-markets in every window tried (confirmed
via `min_settled_ts`/`max_settled_ts` probing: settled history is empty
beyond ~2 months back, and even a 20k-market pull with obvious hourly
commodity series excluded still only spanned ~2 days).

**Why this mattered beyond Stage 4**: the two pulls' Stage 5 OOS results
disagreed in *sign* (Brier 0.0695 vs 0.0776 favoring the correction on the
first pull; 0.0430 vs 0.0391 favoring raw price on the second) — a same-day
sample is close to a random split of simultaneous markets, not a test of
generalizing across time, so this instability was expected, not a paradox.

## D002 — KXTRUMPAPPROVE adopted for Stage 4's calendar-spread problem

Switched strategy from "find a slower-settling series" (D001's instinct)
to "find a series with a *predictable short* cadence and aggregate at that
cadence" — `KXTRUMPAPPROVE` (daily Trump approval-rating ladder) resolves
once a day, every day, so it was never going to produce distinct *months*
but trivially produces distinct *days*. Added `freq` as a parameter to
`stage4_persistence.py`'s aggregation step and orchestrator (previously
hardcoded to monthly) specifically to use this.

**Result**: a real Stage 4 finding for the first time — 45 days, AR(1)
stationary, φ≈0.41, though Ljung-Box/ARCH flagged leftover structure. An
eyeballed "exclude the volatile launch week" cut resolved both diagnostics
(φ→0.13), but a formal Andrews (1993) sup-Wald test (Hansen (2000) wild
bootstrap, ported from VRP's `main:src/models/stage3.py`) found **no
significant break anywhere** in the sample (bootstrap p=0.78; best
candidate break date was Sept 30, not the launch week) — the ad hoc cut
did not survive a principled test, a live instance of the "garden of
forking paths" risk already documented on the VRP side.

## D003 — KXTRUMPAPPROVE is the wrong population for Stage 1/2/5 (superseded as the project's primary dataset)

Running Stage 1/2/5 on the same pull (while validating Stage 4) found a
structural problem independent of D002's finding: `KXTRUMPAPPROVE`
resolves as a daily *ladder* of ~9 strike markets, and the `implied_prob`
used (`last_price_dollars`, i.e. the price right before settlement) is
already near-certain for almost every strike by the time it's read —
355/405 markets priced <0.2 and resolved NO essentially every time. This
isn't evidence of good calibration, it's an artifact of reading the price
too late. Concretely: Stage 2's logit hit quasi-separation (statsmodels:
"a fraction 0.88 of observations can be perfectly predicted"), and Stage
5's Brier scores were artificially tiny (0.00019 vs 0.0000316) because
most markets were nearly certain either way.

**Decision**: `KXTRUMPAPPROVE` remains valid evidence that Stage 4's
*code* works (Tier 1 synthetic tests + this real run), but is retired as
the project's primary data source — a structurally different population
is needed for Stage 1/2/5 to mean anything. See D004 onward.

## D004 — Polymarket ruled out: network-blocked in this environment

Attempted to pivot to Polymarket (genuinely single-outcome markets, no
ladder structure, `implied_prob` would need wiring up via
`get_price_history()`). Both the Gamma API and the CLOB price-history API
returned `403 Forbidden` from this environment via direct `requests`, raw
`curl`, and `WebFetch` — a Zscaler-style network security gateway blocking
the domain, while Kalshi's API works fine from the same environment. This
is an environment-level restriction, not a data or code problem. Decision
(explicit, after presenting the alternatives): drop Polymarket, look for
another Kalshi angle instead.

## D005 — Diagnosed the real root cause: settlement-time price, not ladder structure per se

Scanned ~173 Kalshi series marked `frequency: "daily"` across Financials,
Crypto, Economics, World, Politics, Transportation, and Commodities.
Almost every series with real volume (gas prices across ~50 states,
oil/gold/silver/copper/natural gas, FX pairs, Treasury yields) turned out
to be built the same way as `KXTRUMPAPPROVE` -- a ladder of threshold-price
markets settling on the same day/event (confirmed by inspecting raw
tickers, e.g. `KXAAAGASDIL-26OCT17-4.6850` et al., all sharing one
`close_time`). Only two series found were genuine single binary
markets/day with no ladder (`KXINXDUD`, `KXNASDAQDUD` — S&P/Nasdaq daily
up-or-down), both shallow (n=48, same depth as KXTRUMPAPPROVE).

**The actual fix, found by checking real candlestick data**: pulled hourly
candlesticks for one near-the-money gas-price strike
(`KXAAAGASDIL-26OCT17-4.6850`) and found its bid price moved genuinely
through the day (0.27 -> 0.50 -> 0.36 before settling near 0.98 once the
day's real AAA gas-price reading came in). `last_price_dollars` (used as
`implied_prob` in every pull so far, including D001-D003) is captured
right at settlement, when uncertainty has already resolved — that is why
ladder series looked near-deterministic, not because the underlying
markets are uninformative. An **early** candlestick snapshot (a fraction
of the way between `open_time` and `close_time`) shows real, informative
uncertainty instead. This reopens the entire ladder-series universe
(D003's "wrong population" finding) as usable, high-volume data, provided
the price is read at the right time.

## D006 — First attempt (boundary-strike / "ATM" selection) has a selection-bias flaw

Built `src/data/ladder_atm.py`'s `select_boundary_strikes` +
`build_atm_panel`: for each day's ladder, select the 1-2 strikes bracketing
the *realized* value, then pull an early snapshot for just those. Pulled
14 series (gas x5 states, commodities x3, FX x3, Treasuries x2, diesel),
1039 observations, genuinely spread `implied_prob` (mean 0.505, std 0.24).

**Found, before trusting it further**: this selection rule conditions the
sample on the outcome — by construction, exactly one bracketing strike
resolves YES and one resolves NO every single day, regardless of what the
early price said. This is the econometric analogue of selecting on the
dependent variable, not a timing artifact; re-running at a later snapshot
fraction (0.75 instead of 0.4) shifted the calibration curve's exact shape
but not the core problem. **Decision: `build_atm_panel`/this pull is kept
in the codebase as a documented cautionary example (and as the cheap,
`--mode atm` option in `pull_kalshi_ladder_atm.py`), not used for the
project's findings.**

## D007 — Adopted: full-ladder pull (every strike, no outcome-based selection)

Built `build_full_ladder_panel`: snapshot **every** settled strike's early
price, no selection based on the realized outcome at all — the
methodologically clean choice, at ~10x the API calls of D006's approach
(cost accepted: the API is free and public). Validated on 2 series first
(`KXAAAGASDIL`, `KXAAAGASDTX` — Illinois and Texas gas prices), 2046
observations, 55 distinct days, near-100% snapshot retention.

**Result — the first economically coherent, statistically clean finding
in this project**:
- **Stage 1 (calibration)**: a sharp, classic favorite-longshot bias.
  Low/mid-priced strikes are *overpriced* relative to how often they
  resolve YES (priced 25%, resolves 6.5% of the time — bias +0.18, se=0.02,
  ~8 standard errors from zero; priced 46%, resolves 22% — bias +0.25).
  High-priced strikes are *underpriced* (priced 65%, resolves 84% — bias
  -0.19). Only the top decile (90-100% priced) is near-perfectly
  calibrated. This is the textbook direction (longshots overpriced,
  favorites underpriced), not the reversed pattern D006's biased sample
  showed.
- **Stage 2 (logit)**: converges cleanly, no quasi-separation. Pseudo
  R²=0.67; const, price, and price² all significant at p<0.01.
- **Stage 5 (OOS)**: Brier 0.0665 (corrected) vs. 0.0903 (raw), fraction
  split (+0.0238 improvement); Brier 0.0604 vs. 0.0819, genuine time split
  at a 2026-09-25 cutoff (+0.0216 improvement). Both splits agree in sign
  and rough magnitude — a real, credible out-of-sample result, unlike
  D001's sign-flipping same-day splits.
- **Stage 4**: runs end-to-end (55 daily periods) but is not yet a fully
  clean diagnostic story — level ADF is borderline non-stationary
  (p=0.325; possibly a power issue given the series visually oscillates
  without an obvious trend, not confirmed either way), differencing
  resolves stationarity (p≈0.0000) but the differenced AR(3)'s residuals
  still fail Ljung-Box at some lags, and the Andrews sup-Wald break test
  finds no significant break (bootstrap p=0.332, consistent with D002's
  KXTRUMPAPPROVE finding that this project's break tests so far don't
  support a single sharp regime change).

**Caveat carried forward, not hidden**: only 2 series so far (both US gas
prices, plausibly correlated with each other via shared regional/national
fuel-price drivers, not fully independent evidence). The natural next
step is scaling the same `--mode full` pull to the other asset classes
already identified in D006's scan (commodities, FX, Treasuries) to check
whether the favorite-longshot pattern found here generalizes, or is
specific to gas-price markets' particular settlement mechanics.

## D008 — Scaled `--mode full` to 11 series across 4 asset classes: bias and Stage 5 result both generalize; Stage 4 still not fully clean

Ran `pull_kalshi_ladder_atm.py --mode full` on 5 gas-price series
(`KXAAAGASDIL`, `KXAAAGASDTX`, plus `KXAAAGASDCA`, `KXAAAGASDNY`,
`KXAAAGASDFL`), 3 commodities (`KXGOLDD`, `KXBRENTD`, `KXSILVERD`), 2 FX
pairs (`KXGBPUSDAD`, `KXUSDJPYAD`), and 2 Treasury-yield series
(`KXUST5A`, `KXUST30A`) — chosen from D005/D006's series scan specifically
to cover every asset class not yet tried, not just more gas states.
10,229 observations, 66 days of calendar spread (vs. D007's 55 days on 2
series alone — Treasury series only have ~9 days of settled history so
far, gas/commodities/FX series go back 28-57 days).

**Result — the bias and Stage 5 pattern hold up across asset classes, not
just gas prices**:
- **Stage 1 (calibration)**: same textbook favorite-longshot shape as
  D007's gas-only finding — low-priced buckets overpriced (e.g. priced
  35.7%, resolves 24.5% of the time — bias +0.11, se=0.02), high-priced
  buckets underpriced (priced 54.4%, resolves 70.1% — bias -0.157,
  se=0.019; priced 65.2%, resolves 76.6% — bias -0.114, se=0.02), both
  many standard errors from zero.
- **Stage 2 (logit)**: converges cleanly (no quasi-separation), pseudo
  R²=0.54. `price` itself is highly significant (coef 7.76, p<0.001);
  unlike D007's 2-series fit, `price²` is *not* significant here (p=0.36)
  — plausibly because pooling 4 asset classes with somewhat different
  calibration-curve shapes partially washes out the quadratic term that
  was cleaner on gas alone. Worth revisiting with a `category`/`platform`-
  style control once Stage 3 (drivers of the bias) is built, rather than
  assuming the quadratic term just isn't real.
- **Stage 5 (OOS)**: Brier 0.0782 (corrected) vs. 0.0825 (raw), fraction
  split (+0.0043); Brier 0.0789 vs. 0.0832, genuine time split at a
  2026-09-30 cutoff (+0.0043) — same sign, almost identical magnitude on
  both splits, same qualitative pattern as D007 (smaller improvement in
  absolute Brier terms than D007's 2-series gas-only result, consistent
  with a real but more modest effect once diluted across assets with
  somewhat different bias dynamics, not a contradiction).
- **Stage 4 (persistence)**: runs end-to-end on 64 daily periods. Level
  series is non-stationary (ADF p=0.288), consistent with D007; after
  differencing it's stationary (p≈0.0000) and AR(1)-selected with a
  substantial *negative* coefficient (φ≈-0.57 — day-to-day bias swings
  tend to partially reverse, not persist), but Ljung-Box still rejects
  white-noise residuals at some lags, same unresolved diagnostic gap as
  D007. Andrews sup-Wald break test: no significant break (bootstrap
  p=0.194, best candidate 2026-10-10) — a third dataset now agreeing with
  D002 and D007 that this project's break tests don't support a single
  sharp regime change anywhere tried so far.

**Environment note, not a methodology issue**: `select_ar_order`'s
`old_names=False` kwarg (deprecated-but-functional on statsmodels 0.14)
was removed outright in statsmodels 0.15 and raised `TypeError` on a clean
install from `requirements.txt` (which only pins `>=0.14`) — fixed by
dropping the kwarg (already a no-op). Separately, `select_boundary_strikes`
in `ladder_atm.py` raised `ValueError: assignment destination is read-only`
under pandas 3.0 + numpy 2.5 (newer copy-on-write semantics can hand back a
read-only array from a boolean-comparison `.to_numpy()`) — fixed with an
explicit `.copy()`. Neither bug affects any number reported above (both
are on `--mode atm`/order-selection code paths exercised by the test suite,
not by this run's `--mode full` pull), but both would have silently broken
CI/a teammate's clean install without a pinned-down statsmodels/numpy/
pandas version, so worth tightening `requirements.txt` before the next
environment reset.

**Caveat carried forward**: Treasury series have much shorter settled
history (~9 days) than the rest, so they contribute far fewer independent
days to Stage 4's persistence estimate than gas/commodities/FX do — the
64-period series is not evenly weighted across asset classes. Scaling
further (more Treasury history as it accumulates, or more FX pairs/
commodities) is the natural next step, same spirit as D007's own
caveat, just one level up.
