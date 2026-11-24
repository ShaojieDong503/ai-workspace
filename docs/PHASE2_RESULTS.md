# Ontario PM2.5 model — Phase 2 results

Running report, updated as decisions are made. Last update: 2026-11-23.
Source: `data/interim/model/results_dev.md` and `params_*.json` from the Trillium run of 2026-11-22.

## Summary

LightGBM on dataset A (R1) is the best Phase 2 model: leave-one-region-out RMSE 3.09 µg/m³
(R² 0.63) and leave-one-year-out RMSE 3.86 µg/m³ (R² 0.43). XGBoost (R3) is tied with it,
random forest (R4) is behind. Removing the MERRA-2 aerosol columns (R2, dataset C) raises the
error in every comparison, most on rows without a satellite AOD retrieval.

The main weakness is extreme smoke. Above 50 µg/m³ the models underpredict by about 28 µg/m³
across regions, and holding out 2023 gives RMSE 9.5 against 2.5–3.6 for every other year.
Without 2023, leave-one-year-out RMSE falls to 3.06, level with the spatial estimate.

All 20 jobs completed. The 2024 temporal and spatial test sets have not been opened.

## Design

Four runs compare learners and predictor sets on daily station PM2.5 in Ontario, 2010–2023
(145,315 scored rows per scheme).

| Run | Learner | Predictors | Tuning |
| --- | --- | --- | --- |
| R1 | LightGBM | Dataset A | 100 Optuna trials |
| R2 | LightGBM | Dataset C: A without the 15 MERRA-2 aerosol columns | none: reuses R1's settings |
| R3 | XGBoost | Dataset A | 50 Optuna trials |
| R4 | Random forest | Dataset A | 12-point grid, 300 trees |

Splits are cell-level. The spatial test set holds one cell per region, the temporal test set
is 2024, and a calibration set holds one more cell per region. All three stay closed in Phase 2.

Each run is scored under three cross-validation schemes:

- **Grouped 10-fold** (group10): the folds the settings were tuned on, so its numbers are optimistic.
- **Leave-one-region-out** (LORO, 6 folds): the spatial transfer estimate.
- **Leave-one-year-out** (LOYO, 14 folds): the temporal transfer estimate.

## Tuning

The selected settings are small, strongly regularised trees at moderate learning rates, chosen
by the one-standard-error rule rather than by the lowest error.

Tuning minimised mean RMSE over the grouped 10-fold folds. Gradient boosting used early stopping
(patience 100 rounds, cap 5,000 rounds). The one-standard-error rule then picked the simplest
trial within one standard error of the best.

| Setting | LightGBM (R1, R2) | XGBoost (R3) |
| --- | --- | --- |
| Chosen trial | 58 of 100 | 6 of 50 |
| CV RMSE of chosen trial (µg/m³) | 2.673 | 2.676 |
| Learning rate | 0.104 | 0.178 |
| Tree size | 13 leaves, max depth 15 | max depth 4 |
| Row / column sampling | bagging 0.75, features 0.60 | subsample 0.96, columns 0.42 |
| L1 / L2 penalty | 8.63 / 5.66 | 7.62 / 4.27 |
| Other | min data in leaf 82, min gain 0.67 | min child weight 79.6, gamma 0.16 |

The lowest-error LightGBM trials sat near the bottom of the learning-rate range (0.06; one
scored 2.582). At 0.06 with default settings, 8 of 10 LightGBM folds and 5 of 10 XGBoost folds
reached the 5,000-round cap before early stopping.

Still to read from `params_lgbm.json` / `params_xgb.json`: the best trial's final RMSE
(`best`), the chosen trials' tree counts (`n_trees_final`), and whether any fold of a chosen
trial hit the cap (`not_stopped_chosen`).

## Results

R1 and R3 differ by at most 0.06 µg/m³ under any scheme, well inside the fold-to-fold spread
(LORO standard deviation about 0.8 µg/m³).

### Overall, by scheme

| Run | LORO RMSE | LORO R² | LOYO RMSE | LOYO R² | LOYO RMSE without 2023 | group10 RMSE | group10 R² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 LightGBM, A | **3.09** | **0.63** | **3.86** | **0.43** | **3.06** | 2.76 | 0.71 |
| R3 XGBoost, A | 3.10 | 0.63 | 3.92 | 0.41 | 3.12 | 2.77 | 0.71 |
| R4 Random forest, A | 3.22 | 0.60 | 4.13 | 0.35 | 3.32 | 2.96 | 0.66 |
| R2 LightGBM, C | 3.56 | 0.51 | 4.46 | 0.23 | 3.69 | 3.35 | 0.57 |

RMSE in µg/m³, pooled over all scored rows. group10 is optimistic because it is the tuning scheme.

### High concentrations and the north, R1

| Subgroup | Rows | LORO RMSE | LORO bias | LOYO RMSE | LOYO bias |
| --- | --- | --- | --- | --- | --- |
| ≤ 25 µg/m³ | 144,370 | 2.62 | +0.14 | 2.93 | +0.06 |
| > 25 µg/m³ | 945 | 20.6 | −8.9 | 31.4 | −20.5 |
| > 50 µg/m³ | 101 | 55.2 | −28.3 | 81.9 | −67.2 |
| Heavy smoke | 1,479 | 14.4 | −0.3 | 21.1 | −3.0 |
| North | 18,883 | 4.00 | −0.91 | 4.75 | −0.40 |

Bias is predicted minus observed in µg/m³, so negative means underprediction. Heavy-smoke R² is
0.54 under LORO but 0.01 under LOYO; north R² is 0.47 under LORO. The other runs show the same
pattern.

### Without MERRA-2 aerosol (C) against with (A), by AOD availability (LORO)

| Run | All rows | With AOD (43,492) | Without AOD (101,823) |
| --- | --- | --- | --- |
| R1, dataset A | 3.09 | 3.03 | 3.12 |
| R2, dataset C | 3.56 | 3.45 | 3.61 |

RMSE in µg/m³. Dropping MERRA-2 aerosol costs 0.42 µg/m³ where AOD was retrieved and
0.49 µg/m³ where it was not, so the reanalysis aerosol matters most when the satellite has
no retrieval. LOYO shows the same pattern (+0.56 and +0.62).

### Leave-one-year-out by held-out year

| Run | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 3.23 | 3.02 | 2.89 | 3.26 | 3.61 | 3.14 | 2.48 | 2.65 | 3.30 | 3.26 | 2.65 | 3.31 | 2.74 | **9.48** |
| R2 | 3.64 | 3.34 | 3.19 | 3.73 | 4.42 | 3.84 | 3.02 | 3.08 | 4.60 | 3.62 | 3.19 | 4.57 | 3.18 | **10.22** |
| R3 | 3.21 | 3.11 | 2.94 | 3.31 | 3.65 | 3.15 | 2.50 | 2.72 | 3.49 | 3.32 | 2.72 | 3.38 | 2.81 | **9.53** |
| R4 | 3.47 | 3.28 | 2.97 | 3.54 | 4.02 | 3.58 | 2.73 | 2.82 | 3.78 | 3.31 | 3.08 | 3.39 | 2.97 | **9.89** |

RMSE in µg/m³. Every learner fails the same way on 2023, so the gap is in the data, not the
model choice: no other year in training resembles its smoke.

## Compute

The whole pipeline used about 1 hour 40 minutes of node time in total; the wait for nodes, not
the models, set the turnaround.

Jobs ran on SciNet Trillium (whole 192-core nodes, account def-kuan). Each tuning trial fits
10 folds at once with 19 threads each and takes 9–90 seconds.

| Step | Jobs | Elapsed per job |
| --- | --- | --- |
| LightGBM tuning, 100 trials | 4 chunks of 25 | 11–16 min |
| XGBoost tuning, 50 trials | 2 chunks of 25 | 15–16 min |
| Random forest grid, 12 points | 1 | 5 min |
| Cross-validation runs, R1–R4 × 3 schemes | 12 | 25–46 s |
| Summary | 1 | 12 s |

The first submission asked for 6–8 hour wall times and was estimated to start the next day.
Cutting the limits to 1–2 hours let the jobs start through backfill within the hour. Thirteen
LightGBM trials ran in an interactive debug session before the batch job took over; they count
toward the 100.

Tuning resumes across jobs from one Optuna database per model, so a timed-out chunk loses at
most the trial in progress. Future runs should request 1-hour limits from the start
(`T_TUNE_LGBM=01:00:00 T_TUNE_XGB=01:00:00 T_TUNE_RF=01:00:00 T_RUN=00:30:00`).

## Open questions for the next round

The biggest open question is the smoke underprediction; the tuning questions below are cheap
to test because a full retune takes about an hour.

- [ ] **Round cap.** Raise the 5,000-round cap (or the learning-rate floor of 0.06) so
  low-learning-rate trials are judged at convergence, then retune. Depends on whether the
  chosen trials hit the cap.
- [ ] **Selection rule.** Keep the one-standard-error rule, or select the lowest-error trial.
  Report how much RMSE the rule gives up.
- [ ] **R2 tuning.** R2 reuses R1's settings. Tune it on dataset C once to check how much of
  its 0.47 µg/m³ gap (LORO) comes from the borrowed settings rather than the missing MERRA-2
  aerosol.
- [ ] **Smoke and 2023.** Decide how to target the high tail: features, sample weighting, or a
  separate treatment for smoke days. LOYO without 2023 shows the rest of the series transfers well.
- [ ] **The north.** Spatial R² is 0.47 there against 0.63 overall.
- [ ] **Model choice.** R1 and R3 are tied; pick one for Phase 3 on simplicity or keep both.
- [ ] **Phase 3 gaps** from the run plan: per-region interval coverage, the guard against
  re-running the final calibration, and the 2024 smoke-day groups.

The test sets stay closed until Phase 2 settings are final.
