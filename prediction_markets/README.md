# Prediction Market Calibration — Draft Code

Companion code for `Project/Prediction_Markets_Proposal.docx`. Implements
Stage 1 (measurement: calibration curve) and Stage 2 (significance: logit
fit) end to end. Stages 3-6 are scoped but not written yet — see
`src/models/roadmap_stages_3to6.py`.

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

# unit tests (no network needed -- pure logic on toy data)
pytest tests/
```

Drop `--max-pages` once you're ready to pull full history; a full Kalshi
settled-markets pull is tens of thousands of markets, so expect it to take
a while and consider narrowing with `extra_params` in `kalshi_client.py`
(e.g. a `series_ticker` for one category, or `min_settled_ts`/`max_settled_ts`
for a date range) rather than pulling everything on every run.

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
                 roadmap_stages_3to6.py  (TODO, not implemented)
scripts/         pull_kalshi.py, pull_polymarket.py, run_calibration.py
tests/           unit tests, no network required
data/            gitignored raw pulls land here
```
