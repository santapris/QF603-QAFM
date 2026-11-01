All seven series need one difference (d = 1) to become stationary. After that, daily changes are close to white noise, so an AR model adds very little: BIC picks AR(0) for six series and AR(2) for Brent−WTI. That
  points to ARIMA(0,1,0) for six series and ARIMA(2,1,0) for Brent−WTI.

  I used the back-adjusted futures the strategies trade, in $/bbl, over 4,987 days from Jan 2007 to Oct 2026. No project files changed; everything is in the scratchpad.

  Number of differences

  A series counts as stationary when ADF and Phillips–Perron (PP) reject a unit root and KPSS doesn't reject stationarity, all at 5%. KPSS p-values are capped between 0.01 and 0.10.

  ┌─────────────┬──────────────┬─────────────┬───────────────┬───────────────────┬─────────────┬────────────────┬─────┐
  │   Series    │ Level: ADF p │ Level: PP p │ Level: KPSS p │ d=1: ADF and PP p │ d=1: KPSS p │ d=2: lag-1 ACF │  d  │
  ├─────────────┼──────────────┼─────────────┼───────────────┼───────────────────┼─────────────┼────────────────┼─────┤
  │ WTI         │ 0.394        │ 0.467       │ ≤0.01         │ <0.001            │ ≥0.10       │ −0.499         │ 1   │
  ├─────────────┼──────────────┼─────────────┼───────────────┼───────────────────┼─────────────┼────────────────┼─────┤
  │ Brent       │ 0.618        │ 0.813       │ ≤0.01         │ <0.001            │ ≥0.10       │ −0.495         │ 1   │
  ├─────────────┼──────────────┼─────────────┼───────────────┼───────────────────┼─────────────┼────────────────┼─────┤
  │ RBOB        │ 0.954        │ 0.937       │ ≤0.01         │ <0.001            │ ≥0.10       │ −0.497         │ 1   │
  ├─────────────┼──────────────┼─────────────┼───────────────┼───────────────────┼─────────────┼────────────────┼─────┤
  │ ULSD        │ 0.994        │ 0.996       │ ≤0.01         │ <0.001            │ 0.028       │ −0.503         │ 1*  │
  ├─────────────┼──────────────┼─────────────┼───────────────┼───────────────────┼─────────────┼────────────────┼─────┤
  │ 3-2-1 crack │ 0.981        │ 0.981       │ ≤0.01         │ <0.001            │ ≥0.10       │ −0.473         │ 1   │
  ├─────────────┼──────────────┼─────────────┼───────────────┼───────────────────┼─────────────┼────────────────┼─────┤
  │ Brent−WTI   │ 0.652        │ 0.692       │ ≤0.01         │ <0.001            │ ≥0.10       │ −0.448         │ 1   │
  ├─────────────┼──────────────┼─────────────┼───────────────┼───────────────────┼─────────────┼────────────────┼─────┤
  │ HO−RB       │ 0.774        │ 0.775       │ ≤0.01         │ <0.001            │ 0.078       │ −0.480         │ 1   │
  └─────────────┴──────────────┴─────────────┴───────────────┴───────────────────┴─────────────┴────────────────┴─────┘

  - *ULSD is my judgement call: the strict rule gives d = 2 because KPSS rejects at d = 1. I kept d = 1 because it passes with a trend term and on the market-price series.
  - d = 2 is over-differencing: lag-1 ACF falls to about −0.5 and the standard deviation rises about 40% for every series.
  - Sub-periods agree: ADF and KPSS support d = 1 in 41 of 42 checks across 2007–13, 2014–19 and 2020–26. The exception is back-adjusted Brent in 2014–19 (KPSS p = 0.024).

  Market-price spreads are less clear-cut. Brent−WTI and HO−RB reject a unit root on ADF and PP (p = 0.027/0.018 and 0.017/0.006), but KPSS also rejects stationarity. So d = 0 can't be ruled out for those two on market
  prices, though they're highly persistent at best.

  AR results

  ┌─────────────┬──────────────────────────────────────────┬───────────────────────────────────┬─────────────────────────┬───────────┬───────────┬──────────────┐
  │   Series    │ Level b (half-life, days): back-adjusted │ Level b (half-life, days): market │ AR(1) on changes: φ (p) │ BIC order │ AIC order │ AIC model R² │
  ├─────────────┼──────────────────────────────────────────┼───────────────────────────────────┼─────────────────────────┼───────────┼───────────┼──────────────┤
  │ WTI         │ 0.9989 (659)                             │ 0.9968 (213)                      │ −0.022 (0.31)           │ 0         │ 20        │ 1.6%         │                 
  ├─────────────┼──────────────────────────────────────────┼───────────────────────────────────┼─────────────────────────┼───────────┼───────────┼──────────────┤
  │ Brent       │ 0.9995 (1,315)                           │ 0.9974 (264)                      │ −0.021 (0.34)           │ 0         │ 20        │ 1.4%         │
  ├─────────────┼──────────────────────────────────────────┼───────────────────────────────────┼─────────────────────────┼───────────┼───────────┼──────────────┤
  │ RBOB        │ 0.9999 (11,029)                          │ 0.9967 (210)                      │ −0.023 (0.28)           │ 0         │ 4         │ 0.3%         │
  ├─────────────┼──────────────────────────────────────────┼───────────────────────────────────┼─────────────────────────┼───────────┼───────────┼──────────────┤
  │ ULSD        │ 1.0008 (none)                            │ 0.9978 (320)                      │ −0.005 (0.91)           │ 0         │ 16        │ 1.5%         │
  ├─────────────┼──────────────────────────────────────────┼───────────────────────────────────┼─────────────────────────┼───────────┼───────────┼──────────────┤
  │ 3-2-1 crack │ 1.0001 (none)                            │ 0.9956 (157)                      │ +0.019 (0.50)           │ 0         │ 20        │ 0.9%         │
  ├─────────────┼──────────────────────────────────────────┼───────────────────────────────────┼─────────────────────────┼───────────┼───────────┼──────────────┤
  │ Brent−WTI   │ 0.9997 (2,158)                           │ 0.9952 (143)                      │ +0.056 (0.034)          │ 2         │ 19        │ 1.4%         │
  ├─────────────┼──────────────────────────────────────────┼───────────────────────────────────┼─────────────────────────┼───────────┼───────────┼──────────────┤
  │ HO−RB       │ 0.9989 (643)                             │ 0.9924 (91)                       │ +0.019 (0.68)           │ 0         │ 16        │ 1.3%         │
  └─────────────┴──────────────────────────────────────────┴───────────────────────────────────┴─────────────────────────┴───────────┴───────────┴──────────────┘

  - Why I use BIC: AIC's long models explain at most 1.6% of the variance, and most of their lags aren't individually significant.
  - Brent−WTI AR(2): Δyₜ = 0.018 + 0.059Δyₜ₋₁ − 0.045Δyₜ₋₂, with R² of 0.51%. The second lag isn't significant on robust errors (p = 0.081).
  - Residual check: the BIC models pass a robust Ljung–Box test at lag 20, except WTI (p = 0.025). WTI passes once extreme days are capped at the 0.5th and 99.5th percentiles (p = 0.166), so outliers drive that
    failure.
  - Half-lives: on market prices, only HO−RB's 91 days falls inside the strategy's 5–120 day gate over the full sample.

  ACF and PACF

  Levels show the classic unit-root pattern:
  - ACF starts at 0.997–0.999 at lag 1 and is still 0.45–0.81 at lag 252.
  - PACF is about 1.0 at lag 1, then close to zero.

  Daily changes show almost nothing reliable:
  - Multiple testing: across lags 1–20, 25 of 280 tests are nominally significant, against about 14 expected by chance. None survive a false discovery rate correction (Benjamini–Hochberg).
  - Lag 4 in outrights: this is the most consistent pattern, with ACF of +0.042 to +0.068 across all four. It explains only about 0.2–0.5% of variance and doesn't survive the correction. The four series also move
    together, so this isn't four independent results.
  - Brent−WTI lag 1: +0.056 (p = 0.035), a small short-term continuation effect.
  - Standard bands mislead here: volatility clustering makes the correct lag-1 standard errors 1.5–3.2 times the usual white-noise ones, so default ACF plots would flag too many lags.

  The small effects aren't stable over time:
  - Outrights: BIC picks AR(1) in 2014–19, with φ of −0.08 to −0.11 and R² of 0.6–1.2%. It's back to AR(0) in both 2007–13 and 2020–26.
  - Brent−WTI: BIC order goes 2, 0, 3 across the three periods.
  - HO−RB: BIC order goes 1, 0, 0, and it changes from 0 to 3 when extreme days are capped.

  What this means

  Differencing is the right transformation for modelling the traded series, but it removes the level that any reversion would act on. Daily changes being near white noise also doesn't test the 252-day trend signal,
  - Outrights: BIC picks AR(1) in 2014–19, with φ of −0.08 to −0.11 and R² of 0.6–1.2%. It's back to AR(0) in both 2007–13 and 2020–26.
  - Brent−WTI: BIC order goes 2, 0, 3 across the three periods.
  - HO−RB: BIC order goes 1, 0, 0, and it changes from 0 to 3 when extreme days are capped.

  What this means

  Differencing is the right transformation for modelling the traded series, but it removes the level that any reversion would act on. Daily changes being near white noise also doesn't test the 252-day trend signal,
  because I only checked changes out to lag 40.

  Limitations

  - Near-unit-root estimates: b is biased downward and its standard error is unreliable, so treat half-lives as rough. The bias is larger in short samples, so half-lives from the strategy's 756-day windows can look
    shorter than these.
  - Market series: they include price jumps at contract rolls, which can make them look more mean-reverting than they are.
  - Structural breaks: I didn't test for them, and breaks such as Brent−WTI in 2011–13 can make a stationary series look like it has a unit root.
  - Seasonality: I only removed full-sample month means from the market series, and didn't test seasonal differencing.

  Checks

  ACF and PACF match statsmodels within 1e-8. ADF results match output/tables/03_unit_roots.csv for most series, with small differences for RBOB (0.954 vs 0.940) and ULSD's d = 1 KPSS (0.028 vs 0.023). I didn't trace
  the cause of those two differences.

  Files

  The scratchpad is temporary, so copy anything you want to keep into the project. Everything is in
  /private/tmp/claude-501/-Users-nglei-Desktop-Academics-SMU-Modules-QF603-Quant-Analysis-of-Financial-Markets-Oil-Strategy/3d0e6f6e-1ece-4a07-a888-5d7a69566803/scratchpad/:
  - ar_acf_pacf.py
  - ar_acf_pacf_results/acf_pacf_levels.png
  - ar_acf_pacf_results/acf_pacf_changes.png
  - ar_acf_pacf_results/unit_roots.csv
  - ar_acf_pacf_results/ar1_levels.csv
  - ar_acf_pacf_results/ar_on_changes.csv
  - ar_acf_pacf_results/acf_pacf.csv

  ***How is the differencing determined?***
  ⏺ I tested each series at d = 0, 1 and 2, then took the smallest d where three stationarity tests agreed. The rule is in order_d in ar_acf_pacf.py.

  The steps

  1. Test the level (d = 0) with all three tests.
  2. Difference once and retest (d = 1): Δyₜ = yₜ − yₜ₋₁.
  3. Difference again and retest (d = 2): Δ²yₜ = Δyₜ − Δyₜ₋₁.
  4. Take the first d that passes all three tests at 5%.

  ┌──────────────────────┬─────────────────┬─────────────┬──────────────────────────┐
  │         Test         │ Null hypothesis │ Passes when │         Settings         │
  ├──────────────────────┼─────────────────┼─────────────┼──────────────────────────┤
  │ ADF                  │ Unit root       │ p < 0.05    │ Lag length chosen by AIC │                                                                                             
  ├──────────────────────┼─────────────────┼─────────────┼──────────────────────────┤
  │ Phillips–Perron (PP) │ Unit root       │ p < 0.05    │ arch package defaults    │
  ├──────────────────────┼─────────────────┼─────────────┼──────────────────────────┤
  │ KPSS                 │ Stationary      │ p ≥ 0.05    │ Automatic lag selection  │
  └──────────────────────┴─────────────────┴─────────────┴──────────────────────────┘

  Using tests with opposite null hypotheses matters. Failing to reject a unit root on ADF alone could just mean the test has low power, so KPSS provides a check from the other direction.

  I ran the whole procedure twice: with a constant only, and with a constant and linear trend. The tables in my last answer show the constant-only results.

  Worked examples

  WTI gets d = 1:

  ┌─────┬────────┬────────┬────────┬──────────────────┐
  │  d  │ ADF p  │  PP p  │ KPSS p │      Result      │
  ├─────┼────────┼────────┼────────┼──────────────────┤
  │ 0   │ 0.394  │ 0.467  │ ≤0.01  │ Fails all three  │
  ├─────┼────────┼────────┼────────┼──────────────────┤
  │ 1   │ <0.001 │ <0.001 │ ≥0.10  │ Passes all three │
  └─────┴────────┴────────┴────────┴──────────────────┘

  ULSD needed a judgement call:

  ┌─────┬────────┬────────┬────────┬──────────────────┐
  │  d  │ ADF p  │  PP p  │ KPSS p │      Result      │
  ├─────┼────────┼────────┼────────┼──────────────────┤
  │ 0   │ 0.994  │ 0.996  │ ≤0.01  │ Fails all three  │
  ├─────┼────────┼────────┼────────┼──────────────────┤
  │ 1   │ <0.001 │ <0.001 │ 0.028  │ Fails KPSS       │
  ├─────┼────────┼────────┼────────┼──────────────────┤
  │ 2   │ <0.001 │ <0.001 │ ≥0.10  │ Passes all three │
  └─────┴────────┴────────┴────────┴──────────────────┘

  The rule gives d = 2 here, but I overrode it and reported d = 1, for three reasons:
  - ULSD passes at d = 1 once a trend term is included.
  - Its market-price series passes at d = 1, even with a constant only.
  - At d = 2 it looks over-differenced: lag-1 ACF is −0.503 and the standard deviation rises from 2.40 to 3.40.

  Supporting checks

  These didn't feed into the rule, but they back up the result:
  - ACF of the level: it decays very slowly, which points to differencing being needed.
  - ACF after one difference: the correlations nearly vanish, so one difference is enough.
  - Lag-1 ACF after two differences: it's near −0.5, which is what differencing white noise a second time produces.
  - Standard deviation: it rises about 40% from d = 1 to d = 2, close to the √2 expected from over-differencing. Differencing further adds noise without removing any persistence.

  Limits of this approach

  - The rule is my choice: requiring all three tests at 5% is a common conservative convention, not a formal standard. With ADF alone, every series would get d = 1, and on market prices some would get d = 0.
  - Very large sample: with about 5,000 observations, KPSS can reject on small departures from stationarity, such as a shift in the mean or volatility. That may explain ULSD's result, but I didn't test it.
  - No multiple-testing adjustment: I ran many tests across series, forms and periods, so a few failures at 5% would be expected by chance.
  - Not tested beyond d = 2: this didn't limit the results, because every series passes by then.


  Other Stuff
  ❯ so for the trend carry strategy, do they trade on individual instruments only, or do they trade on the spread as well?                                                                                                  

⏺ Both, but only the trend sleeve trades spreads. Carry trades outrights only.

  ┌────────┬────────────────────────────────────────────┬───────────────────────────────┬────────────────────────────────────────┐
  │ Sleeve │                 Outrights                  │            Spreads            │     Share of combined book's risk      │
  ├────────┼────────────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────┤
  │ Trend  │ WTI (CL), Brent (CO), RBOB (XB), ULSD (HO) │ 3-2-1 crack, Brent−WTI, HO−RB │ 50% (about 29% outrights, 21% spreads) │                                                
  ├────────┼────────────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────┤
  │ Carry  │ WTI (CL), Brent (CO)                       │ None                          │ 50%                                    │
  └────────┴────────────────────────────────────────────┴───────────────────────────────┴────────────────────────────────────────┘

  RBOB and ULSD were left out of carry because their curves have a seasonal shape, so their slope measures the season more than carry. The spreads aren't in carry at all.
  ├────────┼────────────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────┤
  │ Carry  │ WTI (CL), Brent (CO)                       │ None                          │ 50%                                    │
  └────────┴────────────────────────────────────────────┴───────────────────────────────┴────────────────────────────────────────┘

  RBOB and ULSD were left out of carry because their curves have a seasonal shape, so their slope measures the season more than carry. The spreads aren't in carry at all.

  How the spreads trade in the trend sleeve:
  - Signal: each spread is treated as its own instrument, with a signal from the sign of the spread's own back-adjusted 252-day change, not its legs' signals.
  - Execution: all legs are held in the same delivery month and sized together. It targets the same $6,000 a day of risk as each outright.

  Positions at the 26 Oct 2026 close, before sleeve weighting:

  ┌─────────────┬───────────┬──────────────────────────────┐
  │   Spread    │ Direction │       Legs (contracts)       │
  ├─────────────┼───────────┼──────────────────────────────┤
  │ 3-2-1 crack │ Long      │ CL −2.96, RB +1.97, HO +0.99 │
  ├─────────────┼───────────┼──────────────────────────────┤
  │ Brent−WTI   │ Long      │ Brent +16.16, CL −16.16      │
  ├─────────────┼───────────┼──────────────────────────────┤
  │ HO−RB       │ Long      │ HO +1.38, RB −1.38           │
  └─────────────┴───────────┴──────────────────────────────┘

  So a "long" spread still has short legs, which is why the book isn't simply long oil.

  Spreads and outrights overlap: the same contract can appear in several instruments. On that date, WTI was long in the trend and carry outrights but short through the crack and Brent−WTI. Each instrument is sized and
  costed separately, so these offsetting legs aren't netted when costs are calculated. A real trader would net them, so the reported costs may be somewhat too high.

  Spread results in the trend sleeve: Sharpe was 0.39 for the crack, 0.28 for Brent−WTI and −0.17 for HO−RB.