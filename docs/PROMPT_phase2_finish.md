# Task: finish Phase 2 so it can run on SciNet Trillium

You're making the last Phase 2 changes in `code/model/`. Background: `docs/MODEL_RUN_PLAN.md`, `docs/PROMPT_model_runs.md`, and the open items in `docs/REVIEW_MODEL_PHASE2_ROUND4_and_S7.md` (Parts 1–3).

## Rules

- **Code changes only. Don't run anything locally:** not the pipeline, not tuning, not the leakage test, not timing fits. Everything runs on **SciNet Trillium** later. The review is a code audit.
- Don't touch `splits.parquet`, `folds.parquet`, `splits.py` or the Phase 1 modules, except where an item below says so.
- Keep the existing guarantees intact:
  - all rows come from `data.fold_rows`
  - `run_key` / code-hash resume
  - `.tmp` → rename writes
  - `leak_check` before metrics
  - the `OPEN_TEST` guard
  - crc32 fold seeds
  - quick and full kept apart
- Match the existing style: module docstrings with a "Rule" section, small functions.

## Trillium facts to design for

- **Confirmed:** CPU nodes have **192 cores and 768 GB RAM**, with the Slurm scheduler.
- **Assumed, to be confirmed by the author:**
  - whole-node CPU jobs
  - a 24 h wall-time limit
  - compute nodes can write only to `$SCRATCH` (not `$HOME` / `$PROJECT`), and scratch is purged
  - no internet on compute nodes
  - Python through the Alliance module stack and a venv

  Where the code or the docs depend on an assumption, mark it `# confirm on Trillium` / "(confirm)" rather than stating it as fact.

## 1. Fit a trial's folds in parallel (`tune.py`, `cv.py`, `models.py`)

**Problem:** `tune.py:175` calls `study.optimize(objective, n_trials=remaining)`. `--workers` only divides each fit's threads, so trials still run one at a time and most of the node sits idle. `cv.run_cv` also fits folds one after another.

**Change:** make the **folds** the parallel unit, both in tuning and in the runs.
- **Concurrency:** add one helper, for example `cv.fit_folds(specs, fn, fold_workers)`, that fits folds concurrently with `concurrent.futures.ThreadPoolExecutor`. LightGBM, XGBoost and scikit-learn's forest release the GIL, so threads run in parallel and share the loaded frame without copying.
- **Order:** collect results **by fold index, never by completion order**. Per-fold RMSE lists, tree counts and OOF files must be identical however the threads finish.
- **Threads per fit:** `models.resolve_threads` divides the CPUs by `trial_workers × fold_workers`. Use `detect_cpus()` as now.
- **Defaults:** `--fold-workers` defaults to the number of folds in the scheme (10, 6 or 14), capped by the CPUs. `--workers` (parallel trials) defaults to 1.
- **If `--workers > 1`:** call `study.optimize(..., n_jobs=workers)` and build the sampler as `TPESampler(seed=..., constant_liar=True)`. Document that a value above 8 degrades the search.
- **Pruning:**
  - With parallel folds, the per-fold `trial.report` / `should_prune` loop no longer makes sense.
  - Remove the `MedianPruner` when `fold_workers > 1`. Keep it only for `fold_workers == 1`.
  - Record in `params_*.json` which one was used.
- **`cv.run_cv`** uses the same helper. Each fold still writes its own `fold_<label>.parquet` through `.tmp` → rename, and the "skip finished folds" check still runs before submitting a fold.
- **Random Forest:** the `rf` grid uses the same fold-parallel helper. Points still run serially and stay cached.
- **Record** `fold_workers`, `trial_workers` and the threads per fit in `params_*.json` and `run_meta.json`.

## 2. Thread caps (`config.json`)

**Problem:** lines 86, 123 and 139 set `"n_jobs": 8` for LightGBM, XGBoost and Random Forest. On a 192-core node that's about 4%.

**Change:** set all three to `-1`, meaning all CPUs of the allocation, shared out by item 1. Changing the config changes the config hash, so studies and run keys start fresh. That's expected.

## 3. `SCINET.md` for Trillium

Rewrite it for Trillium:
- **Setup on a login node:**
  - copy `code/model/`, `code/panel/` (`data.py` imports `utils.fingerprint`), `data/processed/target/target_table.parquet`, `data/processed/grid/cells.parquet` (used by `score_test.py`), and the frozen `data/interim/model/splits.parquet` and `folds.parquet` to `$SCRATCH/pm25/` (confirm)
  - `module load` steps (confirm)
  - `python -m venv`, then `pip install` pinned to `code/model/requirements.txt` and `data/interim/model/versions.txt` (pandas 3.0.3, numpy 2.4.6, pyarrow 24.0.0, scipy 1.17.1, scikit-learn 1.9.0, lightgbm 4.6.0, xgboost 3.2.0, optuna 4.9.0), with `--no-index` where Alliance wheels exist (confirm)
- **One full-node batch script template** (`--nodes=1`, all 192 cores, `--time=24:00:00`; confirm the exact options) that:
  - exports `PM25_ROOT=$SCRATCH/pm25`
  - exports `OMP_NUM_THREADS` equal to the threads per fit from item 1
  - writes logs under `$SCRATCH/pm25/logs/`
  - runs one command
- **Job sequence, one job each:**
  1. `leakage_test.py`
  2. the quick pass (`tune.py --quick` for lgbm and xgb, `runs.py R1–R3 --quick`, `summarise.py --quick`)
  3. timing
  4. `tune.py --model lgbm`
  5. `tune.py --model xgb`
  6. `tune.py --model rf`
  7. `runs.py R1`, `R2`, `R3`, `R4`
  8. `summarise.py`

  Say that a job hitting the wall time is resubmitted unchanged, and why that's safe: trials count towards the total, finished folds are skipped, grid points are cached.
- **Timing snippet:** time one full-size LightGBM fit at `learning_rate` **0.03 and 0.2**, not the default. Explain that the slow end sets the tuning budget, and whether the 5,000-round cap is reached.
- **One SQLite file, one process:** `studies.db` is used by one process at a time. Don't submit two tuning jobs for the same model at once.
- **After the runs:** copy `data/interim/model/` back from `$SCRATCH` before purging.

## 4. R1 vs R2 on the untuned schemes (`summarise.py`)

**Problem:** line 118 calls `aod_split(root, d)`, which is grouped 10-fold only. R1 was tuned on those folds and R2 reuses its parameters, so that comparison favours R1.

**Change:**
- Report R1 vs R2, overall and by `aod_missing`, for **`loro` and `loyo`**.
- Keep grouped 10-fold as a third block, labelled "tuning folds, for reference only".

## 5. Pin the 10 km subset (`score_test.py`)

**Problem:** the subset is derived at run time from `FAR_KM` and the grid, and nothing checks it.

**Change:** add a constant `FAR_CELLS = {910214, 895350, 958101, 924048}` (pre-registered in plan §7) and assert that the computed set equals it. Refuse with a clear message if not.

## 6. Never overwrite test results (`score_test.py`)

**Problem:** re-running overwrites `test_results_{run}.csv` and `test_per_cell_{run}.csv`.

**Change:**
- If either file exists, refuse, unless `--supersede` is passed.
- With `--supersede`, rename the existing files to `*_superseded_<UTC timestamp>.csv` first, and print the reason it requires (`--reason "..."`, mandatory with `--supersede`).
- Append each scoring (time, run, reason, model file) to `data/interim/model/test_log.jsonl`.

## 7. Merge §7 into the plan (`docs/MODEL_RUN_PLAN.md`)

Merge `docs/PLAN_S7_REPORTING_DRAFT.md` into the plan:
- **Structure:** don't replace the current §7. Make it "## 7. Reporting". Keep the current groups, metrics and development counts as **7.0**, and add the draft's 7.1–7.8 after them.
- **7.3:** use "29 cells" (README §3), not "about 29".
- **7.7 (R1 vs R2 by AOD):** reported on leave-one-region-out and leave-one-year-out; grouped 10-fold for reference only.
- **Add 7.9:** "All four runs (R1–R4) are finalised and scored on the test sets, once each. No run is dropped after its test score is seen."
- **Add 7.10:** "The contingency decision on X1–X3, and the decision note giving its reasons, is recorded before `OPEN_TEST` is created."
- **Add 7.11:** "The 90% interval is calibrated once, on the calibration cells (2010–2023), and isn't recalibrated after the test sets are seen. Coverage is reported per test set, per group and per region. For 2024, coverage also tests whether the interval holds over time, since it was calibrated on earlier years."
- **Plan header:** add a line that §7 was fixed on <date>, before any test row was scored.

## What to send back

Send no run output. For each item 1–7:
- the files and line ranges changed
- a two-line description

Also list every place marked "(confirm)" or `# confirm on Trillium`, so the author can check them in one pass. The next review is a code audit of exactly those changes.
