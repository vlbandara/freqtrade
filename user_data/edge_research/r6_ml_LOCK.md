# Round-6 ML pre-registration (committed before the 2024-25 holdout was opened)

Locked on the 2022-01..2023-12 walk-forward predictions only (r6_ml_design.log):

* Model: LightGBM, parameters as in r6_ml_alpha.PARAMS, 3 seeds, monthly purged walk-forward.
* Book: `ls_bn_q0.2`: long top 20% / short bottom 20% of predictions, equal weight, 0.5/0.5 gross,
  BTC hedge cancelling the book's 30-day net beta, average of the last 3 daily books.
* Design result: Sharpe 1.45, CAGR 27.9%, max DD -13.8%, 2022 +32.0%, 2023 +23.9%, corr BTC 0.11.
* Pass bar for the holdout (2024-01..2025-05): Sharpe >= 1.0, positive in 2024 and 2025 YTD,
  max DD <= 25%, positive at 2x costs. No change to model, features or book after opening.
* Ridge on the same features is reported as a baseline only (design Sharpe ~0).

# Carry pre-registration (committed before the V1-V3 OOS was opened)

Design 2020-2023 (r6_carry_design.log): V1 (interval-normalised signal) and V2 (forecast, same
thresholds) are rejected: both lower Sharpe than V0 in IS and VAL and negative in VAL at 2x costs.
V3 (forecast + 1.5x round-trip-cost hurdle) is carried to OOS as a *cost-robust alternative*:
lower IS CAGR (12.7% vs 21.3%) but VAL Sharpe 6.2 vs 4.8 and, at 2x costs, VAL Sharpe 2.5 vs 0.2.
V3 is adopted only if its OOS Sharpe at 2x costs beats V0's and its OOS return at 1x is >= 70%
of V0's.
