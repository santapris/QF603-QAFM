# panel_monthly — data dictionary

Index: `date` = last NYSE trading day of each month. Variances annualised, decimals (0.04 = 20% vol).
"Known at" = when the value is observable; columns known after t are targets only.

| Column | Description | Units | Known at | Built in | Non-missing |
| --- | --- | --- | --- | --- | --- |
| `iv` | 30-day model-free implied variance (MFIV, OptionMetrics SPX), annualised | variance | t | Phase 2b | 212/212 |
| `har_fcst` | Headline HAR forecast of RV over the next 21 trading days | variance | t | Phase 2c | 212/212 |
| `vrp_exante` | Headline ex-ante VRP = iv − har_fcst | variance | t | Phase 2d | 212/212 |
| `vrp_exante_level` | Ex-ante VRP with the level HAR | variance | t | Phase 2d | 212/212 |
| `vrp_exante_log` | Ex-ante VRP with the log HAR | variance | t | Phase 2d | 212/212 |
| `vrp_exante_iv` | Ex-ante VRP with the IV-augmented HAR | variance | t | Phase 2d | 212/212 |
| `rv_fwd` | Realised variance over trading days t+1..t+21 (TAQ 5-min + overnight), annualised | variance | obs_date_vrp_expost | Phase 2a | 211/212 |
| `vrp_expost` | Ex-post VRP = iv − rv_fwd (short variance-swap payoff); target only, never a predictor | variance | obs_date_vrp_expost | Phase 2d | 211/212 |
| `obs_date_vrp_expost` | Trading day t+21, when rv_fwd / vrp_expost become known (D006) | date | — | Phase 2d | 211/212 |
| `rv_past` | Realised variance over trading days t−20..t, annualised | variance | t | Phase 2a | 212/212 |
| `vrp_trail` | Trailing VRP = VIX²/10000 − rv_past (BTZ convention; Q5 only) | variance | t | Phase 2d | 212/212 |
| `rv_month` | Realised variance over the trading days in (t−1, t], annualised | variance | t | Phase 2d | 212/212 |
| `rv_surprise` | rv_month − HAR forecast made at t−1 (D026) | variance | t | Phase 2d | 212/212 |
| `vix` | VIX close | index points | t | Phase 1d | 212/212 |
| `vix_ts` | VIX3M / VIX (term structure; > 1 = contango) | ratio | t | Phase 1d | 0/212 |
| `vvix` | VVIX close | index points | t | Phase 1d | 0/212 |
| `skew` | Cboe SKEW close | index points | t | Phase 1d | 0/212 |
| `fund_spread` | 3M AA financial CP − 3M T-bill (1-day publication lag; NaN if > 14 days stale) | decimal | t | Phase 1e | 210/212 |
| `fund_spread_nonfin` | 3M AA non-financial CP − 3M T-bill (robustness) | decimal | t | Phase 1e | 197/212 |
| `credit_spread` | Moody's BAA − AAA (1-day publication lag) | decimal | t | Phase 1e | 212/212 |
| `hkm` | HKM intermediary capital ratio, 3-month publication lag | ratio | t | Phase 1g | 0/212 |
| `hkm_lag1` | HKM capital ratio, 1-month lag (robustness) | ratio | t | Phase 1g | 0/212 |
| `cftc_pos` | Dealer net VIX-futures position / open interest, last report released ≤ t (D024, D025) | ratio | t | Phase 1f | 205/212 |
| `cftc_pos_lev` | Leveraged-money net VIX-futures position / open interest (alternative) | ratio | t | Phase 1f | 205/212 |
| `rf` | T-bill holding-period return t → t+1, set at t (D023) | decimal | t | Phase 1e | 211/212 |
| `exret_1` | Log SPX total-return excess return t → t+1 | log return | t+1 month-end | Phase 2d | 0/212 |
| `exret_3` | Cumulative log excess return t → t+3 | log return | t+3 month-end | Phase 2d | 0/212 |
| `exret_6` | Cumulative log excess return t → t+6 | log return | t+6 month-end | Phase 2d | 0/212 |
| `put_ret_fwd1` | Log return of Cboe PUT index t → t+1 (strategy benchmark) | log return | t+1 month-end | Phase 2d | 0/212 |
| `bxm_ret_fwd1` | Log return of Cboe BXM index t → t+1 (strategy benchmark) | log return | t+1 month-end | Phase 2d | 0/212 |
