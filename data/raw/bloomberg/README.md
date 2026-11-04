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

---

# Next pull (planned): more years and more independent spreads

**Why.** The findings say the binding constraint is statistical power, not method. Over 16.7 years and three related spreads, even the best Sharpe seen (0.38) needs about 28 years to reach t = 2. More *independent* bets and more years are the only fixes (`report/exploratory_ratio_slow.md`, `output/exploratory/gate_power/`). Keep the same file layout (two header rows, `PX_SETTLE` and `FUT_CUR_GEN_TICKER` per generic, one sheet per root), save each pull as a **new file with its own date range**, and leave the existing export untouched. `futures.py` reads one workbook today, so using a second one needs a small loader change (merge by root and date), done after the pull.

**Tickers marked (verify) are from memory. Confirm each one with `CTM <GO>` (contract table) or `SECF <GO>` before pulling, and write the confirmed ticker down here.**

## P1. More years for the three existing spreads (do this first)
| Sheet | Generics | Dates | Why |
|---|---|---|---|
| `CL` | `CL1`…`CL12 Comdty` | 1990-01-02 to 2006-12-29 | WTI history before the current sample |
| `CO` | `CO1`…`CO12 Comdty` | 1990-01-02 to 2006-12-29 | ICE Brent history (check the first available date) |
| `HO` | `HO1`…`HO6 Comdty` | 1990-01-02 to 2006-12-29 | Heating oil before ULSD (the contract became ULSD in 2013, which is inside the current sample) |
| `HU` | `HU1`…`HU6 Comdty` (verify) | 1990-01-02 to 2006-12-29 | NYMEX unleaded gasoline, which RBOB (`XB`, from late 2005) replaced. Pull the overlap with `XB` (2005-10 to 2006-12) to measure the break in k at the swap |
| `XB` | `XB1`…`XB6 Comdty` | 2005-10-01 to 2006-12-29 | Overlap with `HU` only |

- Same fields: `PX_SETTLE` and `FUT_CUR_GEN_TICKER`. `HU` and `XB` come in cents/gal like `XB` today.
- Roughly doubles the years for Brent/WTI (to about 35). Grade changes (`HU` → `XB`, heating oil → ULSD in 2013) are breaks in k, which the log-ratio framework treats as a moving anchor.
- **Check older expiry rules.** `futures.expiry()` encodes the current CL, XB/HO and the two ICE Brent rules. Pulling `LAST_TRADEABLE_DT` (via BDP, one value per contract ticker) removes the need to hand-code the 1990s rules: put it in `expiries.csv` (`contract,last_tradeable_dt`).

## P2. Independent spreads with the same constant-k economics (the real fix for power)
Each of these is a new, mostly unrelated bet. Pull the generic curve as above, 1990 or the first available date to now, with `PX_SETTLE` and `FUT_CUR_GEN_TICKER`.

| Spread | Legs (verify tickers) | Economics | Notes |
|---|---|---|---|
| Transatlantic diesel arbitrage | ICE Low Sulphur Gasoil (`QS` family, verify) against `HO` | Cargoes move between Europe and New York Harbor, so the log-ratio should hold near freight plus a quality difference | Gasoil is quoted in **$/tonne**. Convert to $/bbl with the exchange's tonne-to-barrel factor (about 7.45; confirm) |
| European diesel crack | Gasoil against `CO` | Refiners switch between crude and product runs | Same units note |
| East–west crude arbitrage | Dubai or Oman crude (DME Oman or Platts Dubai swaps; verify) against `CO` | Cargoes move between Asia and the Atlantic basin | Check the series is a **futures or swap** with contract months, not a spot assessment |
| European gasoline | Eurobob gasoline (ICE; verify) against `XB` | Gasoline moves across the Atlantic | $/tonne, convert to $/bbl with the exchange factor |

- **Prefer fewer, cleaner series to many noisy ones.** Two of these with 20+ years would already take the bets from three related spreads to five.
- For each new root note the first date, the roll convention, and whether the front contract is liquid. If the front is thin, pull generics 2 to 4 as well.

## P3. Freight and flows (drivers of k, small pulls)
| Series | Source | Why |
|---|---|---|
| Baltic Dirty Tanker Index (`BDTI Index`, verify) and Baltic Clean Tanker Index (`BCTI Index`, verify), `PX_LAST`, daily, 2007 to now | Bloomberg | The arbitrage band is freight: tests whether k tracks it |
| US weekly crude oil exports (EIA series WCREXUS2, verify the ID) | EIA, **free, no terminal needed** | Tests the 2015 end of the export ban against the fall in the Brent/WTI premium |

## P4. Measured trading costs (small pull)
`PX_BID`, `PX_ASK`, `PX_VOLUME` (or `VOLUME`) and `OPEN_INT` for `CL1`…`CL3`, `CO1`…`CO3`, `XB1`…`XB3`, `HO1`…`HO3`, 2007 to now. Replaces the flat $0.05/bbl cost assumption with the actual bid-ask by year and contract position, and shows whether liquidity falls at the roll.

## Not needed
- Intraday data: the tests find no daily edge to refine.
- More curve months (12 for CL and CO, 6 for XB and HO is enough).
- More back-history for trend and carry beyond what P1 gives: the existing CFTC and EIA files already cover the positioning and inventory features.

## After the pull
1. Save under `data/raw/bloomberg/` with the date range in the name, and add a row to the table at the top of this file.
2. Run `scripts/build_data.py` once the loader merges the files, then rerun `scripts/run_ratio_stationarity.py` and the gate-power script on the longer sample. The pre-registered rules apply unchanged.
3. Treat the pre-2007 years as an **out-of-sample test** of what 2007 to 2026 found: the rules must not be re-tuned on them first.
