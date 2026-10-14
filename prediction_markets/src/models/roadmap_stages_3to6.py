"""
Stages 3, 4, and 6 are intentionally NOT implemented yet -- this file is a
scoped TODO list, not working code, so the team can divide it up. Stage 5
IS implemented, in src/models/stage5_oos.py (see scripts/run_oos.py for a
runnable example) -- it was pulled forward because it's the stage that
actually answers whether this whole direction is worth pursuing further.
Real result so far on a quick live pull: the logit correction beats the
raw market price out of sample (Brier 0.0695 vs 0.0776), though the pull
only spans about a day of close_time (see stage5_oos.py's docstring) so
treat this as "promising enough to keep building," not a final answer.

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

Stage 5: DONE -- see src/models/stage5_oos.py

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
