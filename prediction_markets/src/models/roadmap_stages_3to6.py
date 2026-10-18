"""
Stages 3 and 6 are intentionally NOT implemented yet -- this file is a
scoped TODO list, not working code, so the team can divide it up. Stage 5
IS implemented, in src/models/stage5_oos.py (see scripts/run_oos.py for a
runnable example) -- it was pulled forward because it's the stage that
actually answers whether this whole direction is worth pursuing further.
Stage 4 is ALSO implemented now -- see below.

Current primary data source and findings: see README.md (current state)
and DECISIONS.md (the full reasoning trail for how the data source got
here -- several earlier attempts, including this module's own earlier
"KXTRUMPAPPROVE" and general-pull results, were tried and retired; see
DECISIONS.md D001-D007 before reusing any specific number from an older
version of this docstring).

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

Stage 4: DONE -- see src/models/stage4_persistence.py (and
scripts/run_stage4.py for a runnable example)
---------------------------------------------------------
Goes beyond the original one-line idea above it used to say ("OLS of
bias_t on bias_{t-1}") -- instead of assuming AR(1) is the right model,
it runs the full identify/fit/diagnose cycle and only commits to a model
once the diagnostics actually support it:

    1. aggregate per-market bias into one period's series at a chosen
       `freq` (bias_t = mean(implied_prob) - mean(outcome) that period --
       `freq` matters: see DECISIONS.md D002 for why monthly never had a
       chance against Kalshi's general feed, and daily was the fix)
    2. IDENTIFY  -- ACF/PACF, to see what order of memory the series
       actually has, rather than assuming AR(1)
    3. CHECK STATIONARITY -- ADF test on the level series. A trending or
       wandering series would make any AR coefficient on it spurious.
    4. FIT -- AIC-selected AR(p) over the full sample (not walk-forward;
       this mirrors the VRP project's OWN Stage 1 persistence test, which
       is also a direct full-sample AR(1)/AR(2) regression, not a
       walk-forward forecast -- Stage 5 in THIS project already owns the
       out-of-sample question, for the bias-corrected probability itself)
    5. DIAGNOSE -- Ljung-Box (is anything left over in the residuals?)
       and ARCH-LM (does the residual variance cluster over time, the
       way VRP's own calm/stress regime does?)
    6. IF STEP 3 SAID NON-STATIONARY: difference once, then repeat
       steps 2-5 on the DIFFERENCED series. This is the step that's easy
       to skip and shouldn't be: a model fit on a non-stationary series
       can look fine and still be spurious, and the only way to know
       differencing actually fixed it is to re-run the same diagnostics
       on the differenced series, not just assume one difference is
       always enough.
    7. STRUCTURAL-BREAK CHECK -- Andrews (1993) sup-Wald test for a break
       in the AR(p) coefficients at an unknown date (Hansen (2000) wild
       bootstrap for the headline p-value, ported from the VRP side's
       src/models/stage3.py) -- exists specifically so an eyeballed
       "exclude this noisy window" cut can be checked against a
       principled test rather than trusted on sight (see DECISIONS.md D002
       for the episode that motivated building this: an ad hoc cut that
       looked like it found a real regime change did NOT survive this
       test).

Result so far: see README.md's "Stage 4" section for the current primary
result (a 2-series gas-price panel: runs end to end, but residual
diagnostics on the differenced AR(3) still show leftover structure, and
the break test finds no significant break) and DECISIONS.md D002-D003 for
the earlier KXTRUMPAPPROVE result (real persistence finding for Stage 4,
but discovered to be the wrong population for Stages 1/2/5).

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

