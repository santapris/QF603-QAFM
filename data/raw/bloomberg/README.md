# Bloomberg export (SMU terminal)

`oil_futures_bdh_2007-01-01_2026-10-27.xlsx` is the export as received, renamed only (it arrived as
`LATEST COMMOD DATA EXTRACT BLOOMBERGGGG.xlsx`). `config.yaml` → `futures.export_file` points to it. A
later pull goes in as a new file with its own date range, and `export_file` is updated to match.
`src/oilrv/futures.py` reads it and builds the continuous series itself, so Bloomberg's generic roll is
never used for prices.

## What is in it: daily history (BDH), 2007-01-01 to 2026-10-27
One sheet per root (plus a `README` sheet with the pull instructions), each with `PX_SETTLE` and
`FUT_CUR_GEN_TICKER` for every generic:

| Sheet | Tickers | Units as exported |
|---|---|---|
| `CL` | `CL1 Comdty` … `CL12 Comdty` | $/bbl |
| `CO` | `CO1 Comdty` … `CO12 Comdty` | $/bbl |
| `XB` | `XB1 Comdty` … `XB6 Comdty` | **cents/gal** (divided by 100 via `futures.price_divisor`) |
| `HO` | `HO1 Comdty` … `HO6 Comdty` | **cents/gal** (divided by 100 via `futures.price_divisor`) |

Each sheet has two header rows (tickers, then fields) and the dates in the first column:
```
,CL1 Comdty,CL1 Comdty,CL2 Comdty,CL2 Comdty,...
Dates,PX_SETTLE,FUT_CUR_GEN_TICKER,PX_SETTLE,FUT_CUR_GEN_TICKER,...
2026-10-26,101.76,CLZ6,99.49,CLF7,...
```
- **Missing values.** Weekdays without a settle (exchange holidays, and the last row, 2026-10-27, which had not settled) have blank prices. A ` .NA.` ticker counts as missing.
- **Tickers.** Expired contracts carry two-digit years (`CLG07`), recent ones one digit (`CLZ6`). `parse_contract` resolves both.
- **Checks.** No contract is quoted at two different prices on the same day. The loader stops if XB or HO look like cents/gal after scaling.
- **CSV fallback.** Per-root CSVs (`CL.csv`, `CO.csv`, `XB.csv`, `HO.csv`, same layout) still work if `export_file` is unset.

## No expiry pull needed
- **Expiries** are computed from the exchange rules in `futures.expiry()` (CL, RB/HO, and ICE Brent before and after its 2016 rule change).
- **Contract years** are resolved from the observation date. `CLZ6` seen in 2016 is Dec 2016; seen in 2026 it is Dec 2026.
- **Optional override:** an `expiries.csv` with columns `contract, last_tradeable_dt` (e.g. `CLZ2016, 2016-11-21`) overrides the rule for any contract it lists.

## How the data is used
- All legs of a spread hold the **same delivery month**. Brent and WTI are matched by delivery month, never front against front.
- The position rolls 5 business days before the earliest leg expiry. For Brent–WTI that is Brent's expiry.
- **Common dates.** Only days on which every leg of every spread settles are kept (4,987 days, 2007-01-03 to 2026-10-26). A roll on a dropped day is charged on the next kept day.
- In `data/clean/futures_spreads.csv`:
  - `<spread>` is built from additive back-adjusted legs, so its daily change equals the held contracts' change. It is used for signals and P&L.
  - `<spread>_unadj` is the market level of the held contracts.
  - `rolled_<spread>` marks the closes where a roll cost is charged.

Then run `scripts/build_data.py`. It detects the workbook and writes `data/clean/futures_spreads.csv`, `features_futures.csv` and `event_features_futures.csv`.
