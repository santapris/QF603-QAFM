"""
Stages 3-6 are intentionally NOT implemented yet -- this file is a scoped
TODO list, not working code, so the team can divide it up.

Stage 3: Drivers of the bias (OLS)
-----------------------------------
Regress `bias` (or the logit residual from stage2) on:
  - time_to_resolution_days  (close_time - as_of_time)
  - log(volume)
  - category dummies
  - platform dummy
Question: does the bias widen for thin/illiquid markets, or for markets far
from resolution, the way the VRP project checked for regime/vol dependence?
Use statsmodels OLS with HC1/HC3 robust standard errors (we don't have
VRP's overlapping-window autocorrelation problem here since each market is
closer to an independent draw, but confirm this -- markets on the same
event, e.g. multiple strikes of "will CPI print above X%", are NOT
independent of each other and may need clustering by event).

Stage 4: Persistence
---------------------
Is this month's average bias (in some bucket/category) predictive of next
month's? Mirrors the VRP project's Stage 1 (AR(1) persistence test).
Simple to start: monthly `bias` time series -> OLS of bias_t on bias_{t-1}.

Stage 5: Out-of-sample test (the one that matters most)
---------------------------------------------------------
Split resolved markets by close_time into train/test (e.g. train on
everything before a cutoff date, test after). Fit stage2's logit on train
only. On test:
  - naive forecast = raw market price
  - corrected forecast = logit fitted probability (refit on train, applied
    out-of-sample to test prices)
Compare Brier score (see measures/calibration.py) of corrected vs naive on
the test set ONLY. This is the direct analogue of the VRP project's
out-of-sample R^2 against the expanding-mean benchmark, and it is the
number that answers "can you actually predict this, or does the bias just
look real in-sample."

Stage 6 (optional): Bayesian updating
---------------------------------------
Pick a handful of markets with a clear sequence of public news events
before resolution (e.g. a Fed-decision market with CPI prints in between).
Use Bayes' theorem (as in Week 1 of QF603) to compute what the "rational"
updated probability should be after each news event, and compare it to how
much the market price actually moved. Large, systematic gaps are another
angle on the same mispricing question, framed as a forecasting-update
story rather than a static calibration story.
"""
