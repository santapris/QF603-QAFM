# Bloomberg export (SMU terminal)

Put the exported files in this folder exactly as received. `src/oilrv/futures.py` reads them and builds the continuous series itself, so Bloomberg's generic roll is never used for prices.

## What to export: daily history (BDH), 2007-01-01 to latest
Export one CSV per root, with these fields for every generic:
- `PX_SETTLE`: the settlement price
- `FUT_CUR_GEN_TICKER`: the actual contract behind the generic on that day, e.g. `CLZ6`

| File | Tickers | Units |
|---|---|---|
| `CL.csv` | `CL1 Comdty` … `CL12 Comdty` | $/bbl |
| `CO.csv` | `CO1 Comdty` … `CO12 Comdty` | $/bbl |
| `XB.csv` | `XB1 Comdty` … `XB6 Comdty` | $/gal |
| `HO.csv` | `HO1 Comdty` … `HO6 Comdty` | $/gal |

Settings: `Non-trading days = Exclude`, `Fill = Blank` (no carry-forward), and dates shown once in the first column.

Each file needs **two header rows**: tickers on the first, fields on the second.
```
,CL1 Comdty,CL1 Comdty,CL2 Comdty,CL2 Comdty,...
Dates,PX_SETTLE,FUT_CUR_GEN_TICKER,PX_SETTLE,FUT_CUR_GEN_TICKER,...
2026-10-20,107.89,CLX6,106.70,CLZ6,...
```
In Excel, lay out the BDH output like this, then *Save As* CSV.

## No expiry pull needed
- **Expiries** are computed from the exchange rules in `futures.expiry()` (CL, RB/HO, and ICE Brent before and after its 2016 rule change).
- **Contract years** are resolved from the observation date. `CLZ6` seen in 2016 is Dec 2016; seen in 2026 it is Dec 2026.
- **Optional override:** an `expiries.csv` with columns `contract, last_tradeable_dt` (e.g. `CLZ2016, 2016-11-21`) overrides the rule for any contract it lists.
- Don't use BDP on `CLZ6 Comdty` for old contracts: today that ticker means Dec 2026.

## How the files are used
- All legs of a spread hold the **same delivery month**. Brent and WTI are matched by delivery month, never front against front.
- The position rolls 5 business days before the earliest leg expiry. For Brent–WTI that is Brent's expiry.
- `<leg>` is the held contract's unadjusted price. `<leg>_adj` is additive back-adjusted, so its daily change equals the held contract's change; it is used for signals and P&L.
- `rolled_<spread>` marks the closes where a roll cost is charged.

Then run `scripts/build_data.py`. It detects the files and writes `data/clean/futures_spreads.csv`.
