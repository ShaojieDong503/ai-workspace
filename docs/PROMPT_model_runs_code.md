# Task: write the modelling code for the run plan, in `code/model`

You're writing the code that carries out `docs/MODEL_RUN_PLAN.md`: frozen splits, folds, feature reduction, tuning, the R1–R12 cross-validation runs, and the code paths for the final retrain and test scoring. Read the plan first, then `data/processed/README.md`, which defines every column used below. The plan is the authority. Where this prompt and the plan disagree, follow the plan and tell me.

## Rules

- Python is `C:\Python314\python.exe`; run from `D:\ai-workspace\project` with `$env:PYTHONIOENCODING='utf-8'`.
- New code goes in `code/model/`. Outputs go in `data/interim/model/`. Never write to `data/raw/` or `data/processed/`.
- Add `optuna` and `xgboost` to a new `code/model/requirements.txt` (copy the pins from `code/panel/requirements.txt` for the rest) and install them.
- Match the style of `code/panel/` (argparse CLIs, logging, small modules). Reuse `code/panel/utils.py` for content hashing.
- **Don't open the test sets.** No code you run may score, print or summarise rows from the spatial-test cells, the calibration cells or 2024, except to count them when you build the splits. Steps 6–7 get written and guarded, but not run.
- **Don't start the full runs** (100-trial Optuna, full R2–R12). Prove everything on the quick mode described at the end, then stop and report.
- Don't decide the open questions in plan §9. Make each one a config value with the default given below, and list it in your report.
- Report only what you ran and saw.

## 0. Config and data loading

- **One config file**, `code/model/config.json`, holds every seed, threshold, grid and default named below. Each run records a hash of the config it used.
- **One loader** (`data.py`) reads `data/processed/target/target_table.parquet` and adds the derived columns:
  - `aod_missing = (aod055_npix == 0)`. Assert that it equals `aod055.isna()` on every row.
  - Ontario rows are `on_region > 0`. Never use `province_state` for this (README §5).
- **The training-row filter lives in one function.** It applies the split masks, and every model fit in every step gets its rows from that function. Add an assertion inside it: no spatial-test cell, no calibration cell and no 2024 row ever appears in a training set or an early-stopping set.
- **Sanity checks against the plan §8**, for Ontario 2010–2023 before any cells are removed: 197,612 rows, 1,318 above 25 µg/m³, 134 above 50, and 25,005 in North (`on_region == 6`). Fail if any count differs.

## 1. Splits and folds (plan step 1)

Write `splits.py`. Its outputs are `splits.parquet` (`cell_id → role`), `folds_B1.parquet` and `folds_B2.parquet` (`cell_id → fold`), and `splits_summary.md`, which gives cells, rows, rows above 25 and rows above 50 per role and region. Counts only, no metrics.

- **Spatial test cells:** whole Ontario cells, at least one from each of the 6 regions. Default `n_test_cells = 10`, seeded.
- **Calibration cells:** whole Ontario cells, separate from the test cells. Default `n_calib_cells = 6`.
- **Cell selection:** both sets come only from cells that have development rows (2010–2023). Stratify by region and report how many rows each picked cell has.
- **Temporal test:** all 2024 rows, every cell, in both branches.
- **Training domains:**
  - B2 = Ontario development cells.
  - B1 = B2 plus non-Ontario cells with `dist_ontario_km <= b1_buffer_km`. Default 200. The citation for this distance is my job.
- **Grouped 10-fold:**
  - Each fold holds whole cells, and every fold gets a share of every region.
  - Ontario cells get **the same fold in B1 and B2**, so the branches compare fold by fold.
  - B1's extra cells are spread across the 10 folds as well, balanced by distance band and row count.
- **Freezing:** refuse to overwrite existing split or fold files unless `--force` is passed.

## 2. CV engine (`cv.py`)

Write one engine for every run. It takes a branch, a scheme, a dataset (A–D), a model spec and a params source.

- **Schemes:**
  - `group10`: the folds above.
  - `loro`: 6 folds, one per Ontario region. In B1 the non-Ontario cells always stay in training.
  - `loyo`: 14 folds, 2010–2023. Each fold drops that year's rows from every cell.
- **Held-out rows:** in each fold, drop all of the held-out cells and years from training, including B1's cross-border cells, but **score Ontario rows only**.
- **Early stopping:**
  - Hold out 15% of the fold's training cells as an inner validation set (grouped by cell, stratified by region, seeded).
  - Monitor RMSE on the **Ontario rows** of that inner set.
  - Settings: patience 100, maximum 5,000 rounds.
  - Predict with the early-stopped model; there is no refit (config `refit_after_es: false`).
  - Early stopping always uses unweighted L2 RMSE, including in R2 and R3.
- **Fold-level fitting:** anything the run needs that is learned from data, such as R7's gap-filler or a feature-reduction step, is fitted inside the fold on that fold's training rows.
- **Outputs per run × branch × scheme:**
  - Out-of-fold predictions as `oof.parquet`, with columns `cell_id, date, fold, y, yhat`.
  - Per-fold best iteration and per-fold metrics.
  - `run_meta.json` with the config hash, the feature-list hash, a fingerprint of the target table, package versions and wall time.
- **Resume:** folds that are already finished are skipped when a run restarts.
- **Metrics (plan §8):** RMSE, MAE, bias and R² for each of these groups:
  - all Ontario rows
  - `pm25 <= 25`
  - `> 25`
  - `> 50`
  - North (`on_region == 6`)
  - heavy smoke (`hms_density_max == 27`)

  Leave R² out for the `> 25` and `> 50` groups.
- **Where metrics are computed:** compute them once, both pooled over the OOF predictions and per fold, in a single `metrics.py`.

## 3. Feature sets (plan §4, §7)

Define the four datasets in code, not by hand-typed lists scattered through the scripts.

| Group | Columns |
|---|---|
| Satellite AOD (16) | `aod055`, `aod047`, `aod_missing`; the 10 `aod_nbr_*` / `aod_last_obs*` columns; `injection_height`, `smoke_aod055`, `smoke_mask_frac` |
| MERRA-2 aerosol (15) | the 15 mass and extinction columns, README §4.2 |
| Base (31) | MERRA-2 meteorology without `merra_u2m` and `merra_v2m` (17); the 11 fire columns; the 3 HMS columns |

- **Composition:** A = all three groups, B = MERRA-2 aerosol + Base, C = Satellite AOD + Base, D = Base only.
- **`maiac_covered`:** add it in B1, for every dataset that has AOD.
- **No other columns:** no calendar columns, no coordinates. R8–R9 will add those later.
- **Extension point:** provide a registry where an extra feature set can be added for R8–R9. Leave it empty.
- **Feature list:** the frozen list from step 4 is applied on top of these groups.

**Random Forest missing-value rules (`rf_fill`, plan §7):**

- AOD-related NaN → −1. That covers `aod055`, `aod047`, the context means, `aod_last_obs`, `aod_last_obs_age_d`, `injection_height`, `smoke_aod055` and `smoke_mask_frac`.
- `fire_dist_nearest_km` NaN → 5000.
- `hms_density_max` NaN where `hms_any == 0` → 0.
- The 5 missing HMS days (`hms_any` NaN) → −1 in all three HMS columns.
- Two cases the plan doesn't settle. Fill them with −1, but count them and list them in your report:
  - `hms_density_max` is NaN where `hms_any == 1`: no density recorded, February–May 2010.
  - `injection_height` has real negative values, down to −647, which collide with the −1 sentinel.
- After filling, assert that no NaN is left.

## 4. Feature reduction (plan step 2)

Write `feature_reduction.py`. It writes a report and a **draft** list. It never freezes the list itself.

- **Constant columns:** drop columns that are constant on the development rows.
- **Correlation groups:**
  - Inside each group10 training fold, group predictors with |Spearman| ≥ `corr_threshold` (default 0.95).
  - Report each group and how often it appears across the 10 folds.
- **Per-column report:**
  - completeness: share of rows that aren't NaN, overall, for Ontario and by month
  - LightGBM gain importance, per fold, from a default-parameter fit
- **Output:** `feature_reduction_report.md` and `features_draft.json`, holding one suggested representative per correlation group.
- **Freezing:** I edit the draft and freeze it as `features_frozen.json`. Later steps refuse to run without that file, except in quick mode (end of this prompt).

## 5. Tuning (plan step 3)

Write `tune.py`, with Optuna on group10.

- **Objective:** mean Ontario OOF RMSE across folds. Store the per-fold RMSE and the best iteration as trial attributes.
- **Sampler and storage:** TPE sampler, seeded. Store the study in SQLite under `data/interim/model/optuna/` so it can resume.
- **LightGBM (R1, 100 trials):**
  - Learning rate fixed at 0.05 (config), so "number of trees" means the same thing in every run.
  - Search over `num_leaves`, `max_depth`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction` (with `bagging_freq=1`), `lambda_l1`, `lambda_l2` and `min_gain_to_split`.
  - Put the ranges in the config.
- **XGBoost (R10, 50 trials):**
  - Uses the same folds, objective and selection rule.
  - Use `tree_method="hist"`, keep NaN, and search the matching parameters.
- **Selection rule:**
  - The 1-SE band covers trials with mean RMSE ≤ best mean + SE, where SE = standard deviation of the best trial's per-fold RMSE / √10.
  - Within the band, pick the simplest trial: fewest `num_leaves` (for XGBoost, smallest `max_depth`), then the largest `min_data_in_leaf` / `min_child_weight`.
  - Write `params_B1.json` and `params_B2.json` (and the XGBoost equivalents), recording the chosen trial, the best trial, the SE and `n_trees_final` = median of the per-fold best iterations.
- **Random Forest (R11):**
  - Use sklearn `RandomForestRegressor` with `criterion="squared_error"`, `n_estimators` fixed at 300 and `n_jobs=-1`.
  - Run a small grid over `max_features` and `min_samples_leaf`, at most 12 points, on group10 with the same 1-SE rule.

## 6. Runs (plan step 4)

Write `runs.py`, with one entry per run ID, so that `python code/model/runs.py R4 --branch B1 --scheme all` works.

| Run | What to implement |
|---|---|
| R1 | Dataset A, L2, R1 params |
| R2 | Huber (`objective="huber"`, `alpha=δ`), with δ from config (default {2, 5, 10, 20} µg/m³) |
| R3 | L2 with sample weights `w25` for `pm25 > 25` and `w50` for `pm25 > 50`, ~10 combinations from config (default: `w25` in {2, 3, 5}, `w50/w25` in {1, 2, 3}, plus (1, 1)) |
| R4 / R5 / R6 | Datasets B / C / D, L2, R1 params |
| R7 | Dataset A with `aod055` and `aod047` gap-filled inside each fold. The filler is a LightGBM with default parameters, fitted on that fold's training rows that have observed AOD. Its inputs are every non-AOD input plus the AOD context columns; it never sees `pm25`. It fills only the NaN values, and `aod_missing` keeps the original flag. |
| R8–R9 | Stub that raises "feature sets not defined yet" |
| R10 | XGBoost, dataset A, R10 params |
| R11 | Random Forest, dataset A with `rf_fill`, R11 params |
| R12 | XGBoost, dataset B, R10 params; takes `--branch`, since the main branch isn't chosen yet |

**Variant selection for R2 and R3:**

- Run every variant on group10 only.
- Select one variant with the config rule. Default: the lowest RMSE above 25 among variants whose overall Ontario RMSE is no more than 2% worse than R1's.
- Run LORO and LOYO for that selected variant only.
- Keep the results of every variant.

Frozen runs re-pick only the number of trees per fold, by early stopping, as the plan says.

## 7. Summary and decision helper (plan step 5)

- **`summarise.py`:** builds the comparison tables from the stored metrics: runs × groups, per scheme and branch.
- **`decide.py`:**
  - Applies the plan's step-5 rule: R3 replaces R1 only if it does better above 25 µg/m³ with overall RMSE at most 2% worse.
  - Shows the branch comparison.
  - Writes `DECISION_draft.md`. I make the decision; the script doesn't.

## 8. Final retrain and test scoring (plan steps 6–7): write, don't run

- **`final.py`:**
  - Retrains a chosen run on all development rows with `n_trees_final`.
  - Computes 90% intervals by split conformal: the ⌈(n+1)·0.9⌉-th smallest absolute residual on the calibration cells. Also report the same width per region.
- **`score_test.py`:**
  - Scores the spatial-test cells and 2024 once.
  - Computes cell-block bootstrap confidence intervals (resample cells, 2,000 draws, seeded) for every metric and group.
  - Reports interval coverage.
- **Guard on both scripts:** they refuse to run unless `data/interim/model/DECISION.md` exists (I write it) **and** `--open-test` is passed.
- **Test:** check the guard works by running both scripts without those two conditions and showing the refusal.
- **Out of scope:** steps 8 (CanOSSEM comparison) and 9 (grid prediction) aren't part of this task.

## Quick mode and required evidence

Every script takes `--quick`, which:

- keeps the real splits and folds;
- subsamples training rows to 10% per fold (seeded);
- caps boosting at 300 rounds;
- runs 3 Optuna trials;
- writes everything under `data/interim/model/quick/`.

Run and show me:

1. `splits.py` (real, not quick) and its `splits_summary.md`, plus the §0 sanity counts passing.
2. A leakage test: for B1 and B2 under every scheme, assert that no held-out cell, test cell, calibration cell or 2024 row is in any fold's training or early-stopping rows. Also show that the assertion fires on a deliberately broken split.
3. `feature_reduction.py --quick` and the head of its report.
4. In quick mode, both branches: R1 tuning, then R1–R7 and R10–R12 on group10, plus R1 on LORO and LOYO. Show the summary table and the wall time of every step.
5. An estimate of the full-mode wall time per run, from the quick timings, and the number of CPU cores used.
6. The `rf_fill` counts from §3, and a list of every config default that stands in for an open decision in plan §9.

Then stop.
