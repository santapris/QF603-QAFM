# DATA_INVENTORY.md

Filled in during Phase 1a (coverage audit, run 2026-09-23 with `python -m src.data.audit_coverage`).
Machine-readable results: `outputs/tables/data_coverage.csv` and `outputs/tables/data_coverage_gaps.csv`.
"Lag" is the publication lag applied before month-end sampling [F4]; "Used in" lists consuming phases.

## Coverage

**Decisions from the audit:** sample now ends **2025-08-29** (D017, OptionMetrics coverage); CFTC positioning left missing for Jan–May 2009 (D018).

| Series | Source | Library / ticker / file | Freq. | First date | Last date | Gaps (audited against 2008-01 → 2025-12) | Month-end sampling rule | Lag | Used in |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPX options | WRDS OptionMetrics | `optionm.opprcdYYYY` (1996–2025), secid 108105 | daily | 2008-01-02 (in sample) | **2025-08-29** | None through 2025-08-29. **Sep–Dec 2025 missing (84 trading days, 4 month-ends)** — 🚦 gate | month-end trading day; nearest prior buffer day if bad (logged) | 0 | 2b, 6 |
| Zero curve | WRDS OptionMetrics | `optionm.zerocd` | daily | 1996-01-02 | **2025-08-29** | Same as SPX options: Sep–Dec 2025 missing | same dates as options | 0 | 2b |
| SPX index close/open | WRDS OptionMetrics | `optionm.secprd`, secid 108105 | daily | 2008-01-02 (in sample) | **2025-08-29** | None through 2025-08-29; `open` present on every row | last obs ≤ month-end | 0 | 1.5, 2, 6 |
| SPY trades | WRDS TAQ (daily, millisecond) | `taqm_YYYY.ctm_YYYYMMDD` (2003–2025) | intraday | **2003-09-10** | 2025-12-31 | None — every NYSE trading day has a `ctm` table; SPY present in all yearly first/last-day spot checks | — (daily RV) | 0 | 2a |
| VIX | Bloomberg | `VIX Index` → `VIX_Index.csv` | daily | 🧑 pending export | | | last obs ≤ month-end | 0 | 2, 3 |
| VIX3M | Bloomberg | `VIX3M Index` → `VIX3M_Index.csv` | daily | 🧑 pending | | | last obs ≤ month-end | 0 | 3 |
| VVIX | Bloomberg | `VVIX Index` → `VVIX_Index.csv` | daily | 🧑 pending | | | last obs ≤ month-end | 0 | 3, 6 |
| SKEW | Bloomberg | `SKEW Index` → `SKEW_Index.csv` | daily | 🧑 pending | | | last obs ≤ month-end | 0 | 3 |
| SPX total return | Bloomberg | `SPXT Index` → `SPXT_Index.csv` | daily | 🧑 pending | | | last obs ≤ month-end | 0 | 4 (Q5) |
| Cboe PutWrite / BuyWrite | Bloomberg | `PUT Index`, `BXM Index` → `PUT_Index.csv`, `BXM_Index.csv` | daily | 🧑 pending | | | last obs ≤ month-end | 0 | 6 |
| VIX futures (optional) | Bloomberg | `UX1 Index`, `UX2 Index` → `UX1_Index.csv`, `UX2_Index.csv` | daily | 🧑 pending (optional) | | | last obs ≤ month-end | 0 | 6 |
| 3M AA financial CP | FRED | `DCPF3M` | daily | 1997-01-02 | 2026-09-21 | Month-end value >7 days stale at 8 month-ends: 2008-12 (8d), **2020-04 (35d)**, 2023-05 (12d), 2023-12 (14d), 2024-09 (10d), 2025-08 (10d), 2025-09 (8d), 2025-10 (9d) | last obs ≤ month-end after lag; staleness cap set in 1e | 1 bday | 3 |
| 3M AA non-financial CP | FRED | `DCPN3M` | daily | 1997-01-02 | 2026-09-21 | **38 stale month-ends (up to 55 days)** — unsuitable as the headline series; robustness only | as above | 1 bday | 3 (robustness) |
| 3M T-bill | FRED | `DTB3` | daily | 1954-01-04 | 2026-09-21 | None beyond holidays | last obs ≤ month-end after lag | 1 bday | 3, rf |
| Moody's BAA / AAA | FRED | `DBAA`, `DAAA` | daily | 1986 / 1983 | 2026-09-21 | None beyond holidays | last obs ≤ month-end after lag | 1 bday | 3 |
| VIX futures positioning | CFTC TFF (futures only) | Public Reporting API dataset `gpe5-46if`, contract code `1170E1` | weekly (Tue as-of) | 2006-08-29 | 2026-09-15 | **No reports between 2008-12-16 and 2009-06-02** (legacy COT has the same hole) → month-ends **Jan–May 2009** lack a report released that month. Other gaps are holiday shifts (8–28 days). Position fields never null. | last report **released** ≤ month-end; release date inferred from as-of date (Phase 1f) | release date | 3 |
| HKM capital ratio | He–Kelly–Manela website | `data/raw/hkm/` | monthly | 🧑 pending download | | | value for month m used at month-end m+3 | 3 months | 3 |
| VIX (prototype stand-in only) | FRED | `VIXCLS` | daily | 1990-01-02 | 2026-09-22 | None beyond holidays | last obs ≤ month-end | 0 | 1.5 only |

## Pulled so far

| Dataset | File(s) | Rows | Dates | Pulled | Check |
| --- | --- | --- | --- | --- | --- |
| SPX options, month-end ±3 trading days, all expiries | `data/raw/optionmetrics/spx_options_YYYY.parquet` (2007–2025) | 13,550,172 | 1,488 request dates (2007-12-26 → 2025-08-29), all with data | 2026-09-23 | `optionmetrics_pull_summary.csv`: all 212 month-ends present, 30-day bracket available at each, no duplicates, 23 crossed quotes |
| Zero curve (same dates) | `zerocd_request_dates.parquet` | 56,776 | 1,488 | 2026-09-23 | rates 0.06–5.79 (percent, continuously compounded) |
| SPX index daily | `spx_index_daily.parquet` | 5,529 | 2003-09-10 → 2025-08-29 | 2026-09-23 | every NYSE trading day present; `open` > 0 on all days |
| FRED rates (DCPF3M, DCPN3M, DTB3, DBAA, DAAA; VIXCLS prototype only) | `data/raw/fred/{SERIES}.csv` → `data/processed/fred_monthly.parquet` | 212 month-ends | 2008-01-31 → 2025-08-29 | 2026-09-23 | fund_spread NaN 2020-04, 2024-02 (D022); fund_spread_nonfin NaN at 15 month-ends; rf NaN at the last month (holding period ends after sample) |
| CFTC TFF VIX futures (code 1170E1) | `data/raw/cftc/tff_futures_only_1170E1.csv` → `data/processed/cftc_weekly.parquet`, `cftc_monthly.parquet` | 1,017 reports; 212 month-ends | as-of 2006-08-29 → 2026-09-15 | 2026-09-23 | release dates inferred (D025); `cftc_pos` NaN 2009-01…05 and 2019-01, 2019-02; median release age at month-end 3 days, max 34 |

## Notes from the audit

* OptionMetrics `opprcd` also has `forward_price`, `expiry_indicator`, `root` (SPX vs SPXW) and `theta`,
  which are useful for MFIV checks and for choosing strategy expiries (Phase 6).
* TAQ is organised as one library per year (`taqm_2003` … `taqm_2025`, plus `taqmsec` spanning all
  years); tables are `ctm_YYYYMMDD` (trades), `complete_nbbo_YYYYMMDD`, `wct_YYYYMMDD`. SPY filter:
  `sym_root = 'SPY' and coalesce(sym_suffix, '') = ''`. No pre-2003 daily TAQ is available, so the HAR
  burn-in starts 2003-09-10 (≈ 4.3 years before the first forecast origin).
* The CFTC API gives the Tuesday as-of date only. The Friday release date must be inferred; weeks
  affected by holidays and the 2013 / 2018–19 government shutdowns were released late (Phase 1f decision).
