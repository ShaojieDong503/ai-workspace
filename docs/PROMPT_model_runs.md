# Task: build and run the PM₂.₅ models in `docs/MODEL_RUN_PLAN.md`

You're writing the modelling code and running it on the finished training table. Read `docs/MODEL_RUN_PLAN.md` first, then `data/processed/README.md`, which defines every column used below. The plan is the authority. Where this prompt and the plan disagree, follow the plan and tell me.

The work has three phases, and **each one ends with a stop**. Don't start the next phase until I say so.

## Existing code

`code/model/` already holds code written from an earlier version of this plan. It has never been run. Rewrite it to this prompt; don't extend it.

- **Remove:** branch B1, datasets B and D, the Random Forest fill rules (`rf_fill`), the Huber and weighted runs, and `decide.py`.
- **Problems found in it**, which the rewrite must not repeat:
  - `data.scoring_rows` filters only by Ontario plus the held-out cells or year. So grouped 10-fold and leave-one-region-out score the held-out cells' 2024 rows. Leave-one-year-out scores every Ontario cell in that year, test and calibration cells included.
  - `data.training_rows` calls `domain()` without its `cfg` argument, which raises a TypeError.
  - The `aod_missing` check in `data.load()` calls `int()` on a whole Series, which also raises a TypeError.
  - `splits.py` picks 6 test cells instead of the configured number.

## Rules

- **All runs happen on SciNet** (Linux, Slurm), not on the local Windows machine. Locally, only edit code; don't run it. The code must therefore be portable:
  - **Paths:** no hard-coded Windows paths. The project root is `$PM25_ROOT` when set, otherwise the folder two levels above `code/model/data.py`.
  - **Files to copy:** `code/model/` and `code/panel/` (`data.py` imports `fingerprint` from `code/panel/utils.py`), `data/processed/target/target_table.parquet`, and the frozen `data/interim/model/splits.parquet` and `folds.parquet`. Don't re-run `splits.py` there.
  - **Environment:** a virtual environment built on a login node (compute nodes have no internet), pinned to `code/model/requirements.txt` and `versions.txt`.
  - **Threads:** `n_jobs = -1` uses the cores visible to the job. Set `OMP_NUM_THREADS` to `$SLURM_CPUS_PER_TASK` in each job script, so LightGBM and XGBoost don't oversubscribe the node.
  - **Wall time:** jobs stop at the cluster's time limit (24 h), so everything must resume when resubmitted. Tuning counts trials towards the total (it runs only the missing ones), the CV runs skip finished folds, and the Random Forest grid caches each point.
  - **Optuna storage:** the SQLite file must be used by one process at a time. Don't run parallel tuning workers against the same `studies.db` on the shared filesystem.
  - **Outputs:** write to `$SCRATCH`, and copy results to `$PROJECT` or home before scratch purging can remove them.
- Locally, Python is `C:\Python314\python.exe`; run from `D:\ai-workspace\project` with `$env:PYTHONIOENCODING='utf-8'` (only if the author asks for a local run).
- Code goes in `code/model/`. Outputs go in `data/interim/model/`. Never write to `data/raw/` or `data/processed/`.
- Create `code/model/requirements.txt` from the pins in `code/panel/requirements.txt`, add `optuna` and `xgboost`, install them, and record the installed versions.
- Match the style of `code/panel/` (argparse CLIs, logging, small modules). Reuse `code/panel/utils.py` for content hashing.
- **Don't open the test sets** before Phase 3. Until then, no code may score, print or summarise the spatial-test cells, the calibration cells or 2024, except to count rows when the splits are built.
- Long jobs run in the background, log to a file, and can resume after an interruption.
- Don't decide the plan's open questions (§8). Each one is a config value; list them in your reports.
- Report only what you ran and saw. If a step fails or a check doesn't pass, stop and tell me; don't work around it.

## Scope, from the plan

- **Rows:** Ontario only, `on_region > 0`. Never use `province_state` for this (README §5). Monitors outside Ontario are not used at all.
- **Loss:** L2 only.
- **Runs:**

  | Run | Model | Data |
  |---|---|---|
  | R1 | LightGBM, Optuna-tuned | A |
  | R2 | LightGBM, R1's settings | C |
  | R3 | XGBoost, Optuna-tuned | A |
  | R4 | scikit-learn Random Forest, small grid | A |

- **Contingency runs** X1–X3 (gap-filled AOD, extra feature sets): **don't build or run them.** I decide on them after Phase 2.

## Phase 1: data, splits, feature reduction

### 1.1 Config and loader

- **Config:** one file, `code/model/config.json`, holds every seed, threshold, search range and default named in this prompt. Each run records the hash of the config it used.
- **Loader (`data.py`):**
  - Reads `data/processed/target/target_table.parquet` and keeps Ontario rows.
  - Adds `aod_missing = (aod055_npix == 0)`, and asserts that it equals `aod055.isna()` on every row.
- **Sanity check:** for Ontario 2010–2023, before any cells are removed, the counts must match the plan §7: 197,612 rows, 1,318 above 25 µg/m³, 134 above 50, and 25,005 in North (`on_region == 6`). Fail if any differs.
- **Training-row filter:** a single function returns training rows. Every fit, in every phase, gets its rows from it. Inside it, assert that no spatial-test cell, no calibration cell and no 2024 row is present, in either the training set or the early-stopping set.
- **Scoring-row filter:** a single function returns the rows a fold is scored on, and every prediction that feeds a metric goes through it.
  - In Phases 1–2 it returns only development cells (role `development` in `splits.parquet`) in 2010–2023, intersected with the fold's held-out cells or year.
  - It asserts the same thing as the training filter: no spatial-test cell, no calibration cell, no 2024 row.
  - For leave-one-year-out, that means the held-out year's rows **from development cells only**, never every Ontario cell in that year.
  - The test and calibration rows come only from a separate function used in Phase 3. That function carries the same guard as the Phase 3 scripts.
- **No other paths:** no code selects rows from the target table except through these filters. The early-stopping set is cut from the training rows; the row-count and summary code may count rows but never compute metrics on them.

### 1.2 Splits and folds (plan step 1)

Write `splits.py`. Outputs:

- `splits.parquet`: `cell_id → role`.
- `folds.parquet`: `cell_id → fold`.
- `splits_summary.md`: cells, rows, rows above 25 and rows above 50, per role and region. Counts only, no metrics.

How to build them:

- **Spatial test cells:** whole Ontario cells, at least one from each of the 6 regions. Default `n_test_cells = 10`, seeded.
- **Calibration cells:** whole Ontario cells, separate from the test cells. Default `n_calib_cells = 6`.
- **Picking cells:** only from cells that have 2010–2023 rows, stratified by region. Report the row count of each picked cell.
- **Temporal test:** all 2024 rows.
- **Grouped 10-fold:** the remaining development cells, whole cells per fold. Spread the regions across folds and balance the row counts.
- **Freezing:** refuse to overwrite existing split or fold files unless `--force` is passed.

### 1.3 Feature sets (plan §3, §6)

Define the groups in one place in code:

| Group | Columns |
|---|---|
| Satellite AOD (16) | `aod055`, `aod047`, `aod_missing`; the 10 `aod_nbr_*` / `aod_last_obs*` columns; `injection_height`, `smoke_aod055`, `smoke_mask_frac` |
| MERRA-2 aerosol (15) | the 15 mass and extinction columns, README §4.2 |
| Base (31) | MERRA-2 meteorology without `merra_u2m` and `merra_v2m` (17); the 11 fire columns; the 3 HMS columns |

- **Datasets:** A = all three groups. C = Satellite AOD + Base.
- **No other columns:** no calendar fields, no coordinates, no `maiac_covered`. That last one is constant over Ontario; check this and say so.
- **Missing values:** no filling anywhere. NaN goes straight into LightGBM, XGBoost and `RandomForestRegressor`, which have handled NaN natively since scikit-learn 1.4. Check the installed version.
- **No pruning:** models use every column of their dataset, 62 for A and 47 for C. There is no frozen feature list.

### 1.4 Feature reduction: dropped (plan step 2)

No pruning, by author decision. `feature_reduction.py` may stay as a descriptive report (completeness, correlations, gain), but nothing in Phase 2 reads its output.

### 1.5 Leakage test

For every scheme and every fold, assert:

- **Training and early-stopping rows:** no held-out cell or year, no test cell, no calibration cell, no 2024 row.
- **Scored rows:** only development cells in 2010–2023, and exactly the fold's held-out cells or year.
- **Coverage:** across each scheme's folds, every development row is scored exactly once.

Then break things on purpose and show that each assertion fires:

- put a test cell into one fold's training set
- add a 2024 row to one fold's scored rows
- add a calibration cell to one leave-one-year-out fold's scored rows

### Stop 1. Report:

- the sanity counts
- `splits_summary.md`
- the leakage test output

Then wait for my approval of Phase 1.

> **Phase 1 approved, 19 Nov 2026.** The approved state:
> - **Splits:** `splits.parquet` and `folds.parquet` from the 19 Nov run with `calib_cells_per_region = 1`. Test cells 856931, 895350, 910214, 918734, 924048, 958101; calibration cells 863269, 864472, 918737, 928317, 930445, 933610; 36 development cells, 145,315 rows.
> - **Leakage test:** PASS.
> - **Features:** datasets A and C in full, no pruning.
> - **Contingency (X1–X3):** judged by hand after R1.
>
> Don't re-run `splits.py`. Phase 2 may start.

## Phase 2: tuning and cross-validation runs

### 2.1 CV engine (`cv.py`)

One engine serves every run. It takes a scheme, a dataset, a model spec and a source of parameters.

- **Schemes (plan §2):**
  - `group10`: the folds from 1.2.
  - `loro`: 6 folds, one per region.
  - `loyo`: 14 folds, one per year 2010–2023. Each fold removes that year's rows from every cell.
- **Early stopping (LightGBM, XGBoost):**
  - Hold out 15% of the fold's training cells as an inner validation set (grouped by cell, stratified by region, seeded).
  - Monitor RMSE on that set; patience 100, maximum 5,000 rounds.
  - Predict with the early-stopped model; no refit.
- **Outputs per run × scheme:**
  - `oof.parquet` with columns `cell_id, date, fold, y, yhat`
  - per-fold metrics and best iteration
  - `run_meta.json` with the config hash, the feature-list hash, the target-table fingerprint, package versions and wall time
- **Resume:** folds already finished are skipped when a run restarts.
- **Check before writing metrics:** before writing a run's metrics, re-check that its OOF file holds only development cells in 2010–2023, and that each development row appears exactly once.

### 2.2 Metrics (`metrics.py`, plan §7)

- **Metrics:** RMSE, MAE, bias and R².
- **Groups:**
  - all rows
  - `pm25 <= 25`
  - `> 25`
  - `> 50`
  - North (`on_region == 6`)
  - heavy smoke (`hms_density_max == 27`)
- **Exception:** leave R² out for `> 25` and `> 50`.
- **Computed both ways:** pooled over the OOF predictions, and per fold.

### 2.3 Tuning (`tune.py`, plan step 3)

All tuning runs on `group10`.

- **Optuna setup:**
  - Objective: mean OOF RMSE across folds, with per-fold RMSE and best iteration stored on each trial.
  - TPE sampler, seeded.
  - Studies stored in SQLite under `data/interim/model/optuna/`, so they resume.
- **R1, LightGBM, 100 trials:**
  - Learning rate searched, log-uniform over `learning_rate_range` (0.03–0.2); early stopping picks the trees for each trial.
  - Search over `num_leaves`, `max_depth`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction` (with `bagging_freq=1`), `lambda_l1`, `lambda_l2` and `min_gain_to_split`.
  - Ranges go in the config.
- **R3, XGBoost, 50 trials:**
  - Use `tree_method="hist"`; learning rate searched over the same range.
  - Search the matching parameters.
- **R4, Random Forest, grid:**
  - `RandomForestRegressor` with `criterion="squared_error"`, `n_estimators=300` and `n_jobs=-1`.
  - The grid covers `max_features` × `min_samples_leaf`, at most 12 points.
- **Selection rule, for all three:**
  - Take the trials whose mean RMSE is no more than best + SE, where SE = standard deviation of the best trial's per-fold RMSE / √10.
  - From those, pick the simplest:
    - LightGBM: fewest `num_leaves`, then largest `min_data_in_leaf`
    - XGBoost: smallest `max_depth`, then largest `min_child_weight`
    - Random Forest: largest `min_samples_leaf`, then smallest `max_features`
- **Output:** `params_R1.json`, `params_R3.json` and `params_R4.json`. Each records the chosen trial, the best trial, the SE and, for the boosted models, `n_trees_final` = median of the per-fold best iterations.

### 2.4 Runs (`runs.py`, plan step 4)

- **Interface:** `python code/model/runs.py R1 --scheme all`.
- **R1–R4:** each runs on `group10`, `loro` and `loyo`. R2 uses R1's parameters and re-picks only the number of trees.
- **`summarise.py`:** writes `results_dev.md`, with runs × groups for each scheme and the per-fold spread.

### 2.5 Order

1. Run the whole of Phase 2 with `--quick` first:
   - same splits and folds
   - 10% of training rows per fold
   - at most 300 rounds
   - 3 Optuna trials
   - output under `data/interim/model/quick/`

   From the quick timings, estimate the full wall time of each step and report it with the number of CPU cores.
2. Then start the full Phase 2 in the background, in this order:
   1. R1 tuning
   2. R1 runs
   3. R2
   4. R3 tuning and runs
   5. R4 tuning and runs

### Stop 2. Report:

- `results_dev.md`
- the chosen parameters and their 1-SE bands
- wall times

Then wait. I decide whether X1–X3 are needed and when the test sets are opened.

## Phase 3: final models and test scoring (only when I say so)

Write both scripts in Phase 1 or 2, but run them only in Phase 3.

- **Guard:** both scripts refuse to run unless `data/interim/model/OPEN_TEST` exists (I create it) and `--open-test` is passed. Show the refusal during Phase 2.
- **`final.py` (plan step 5):**
  - Retrains each run on all development rows. The boosted models use `n_trees_final`.
  - Computes 90% intervals by split conformal: the ⌈(n+1)·0.9⌉-th smallest absolute residual on the calibration cells. Also report the width per region.
- **`score_test.py` (plan step 6):**
  - Scores the spatial-test cells and 2024, once.
  - Computes cell-block bootstrap confidence intervals (resample cells, 2,000 draws, seeded) for every metric and group.
  - Reports interval coverage.
  - Remember that 2024 has too few smoke days for the smoke-day groups (plan §1). Report them, but flag them.
- **Out of scope:** steps 7 (CanOSSEM comparison) and 8 (grid prediction) aren't part of this task.
