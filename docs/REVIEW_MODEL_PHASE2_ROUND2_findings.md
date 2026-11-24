# Phase 2 code review (round 2): findings

**Method:** a code audit. I read the logic of the changed modules (`models.py`, `cv.py`, `tune.py`, `runs.py`, `final.py`, `score_test.py`, `leakage_test.py`, `summarise.py`, all dated 19 Nov, 13:47–13:48) against `docs/REVIEW_MODEL_PHASE2_findings.md`. I ran nothing.

> **For the coding agent:** as before, make the fixes as code changes only, with no local runs. For each finding, point to the lines that fix it.

**Verdict:** every round-1 blocker is fixed. I found four new issues to fix before the full runs (A1–A4), two before Phase 3 (A5–A6), and a few minor ones.

## Round-1 findings: status

| Finding | Status | Where |
|---|---|---|
| B1, tuning ignored the configured settings | **Fixed.** Tuning, the runs and the final retrain all fit through `models.fit_predict`, which merges in `lgbm_params` / `xgb_params` / `rf_params`. | `models.fit_predict`, `tune.evaluate` |
| B2, shared parameter file | **Fixed.** Quick parameters go to `quick/params_*.json`, and `runs.load_params` refuses a mode mismatch. | `tune.main`, `runs.load_params` |
| B3, resume reused old folds | **Fixed** (but see A3): `run_key.json`, refusal unless `--fresh`, and `.tmp` files renamed into place. | `cv._check_key`, `cv.run_cv` |
| B4, `hash()` seeds | **Fixed.** `zlib.crc32` | `cv.fold_seed` |
| B5, tree count off by one | **Fixed.** Counted per library, with safe fallbacks. | `models.best_iter_lgbm`, `best_iter_xgb` |
| B6, Random Forest search | **Fixed.** A full 12-point grid, with `max_features` compared as a fraction (`sqrt` → 1/√p). | `tune.tune_grid`, `feature_fraction` |
| P3-1, Phase 3 LightGBM-only | **Fixed** (but see A5, A6): `RUNS` dispatch in both scripts. | `final.py`, `score_test.py` |
| P3-2, bootstrap RMSE only | **Fixed.** All four metrics, coverage per group, and the cell count per set. | `score_test.bootstrap`, `coverage` |
| P2-2, leakage test not using the engine | **Fixed.** The test consumes `cv.dry_run` and checks train, inner and score are pairwise disjoint, with exact coverage. | `leakage_test.py` |
| Minor: completeness check, dead code, explicit seeds, `n`/MAE/bias in the summary, group10 caveat | **Fixed.** | `cv.leak_check`, `models.*_params`, `summarise.py` |

## Fix before the full runs

### A1. "Hit the cap" misses runs where early stopping never triggered (`models.fit_predict`)

`hit = n >= cap`, where `n` is the best iteration. When early stopping never fires, LightGBM still reports the best iteration it saw, which is usually just below the cap. The brief's own example was 4,956 out of 5,000. XGBoost does the same. So `hit` is False in exactly the case the flag exists to catch, and `folds_hit_cap` / `cap_hits_chosen` will read 0.

**Fix:**
- Treat a fold as "not stopped" when `n > cap - patience`: stopping can only have fired if the best iteration was more than `patience` rounds before the cap.
- Rename the field `folds_not_stopped`.

### A2. Random Forest trains without the inner early-stopping cells

`cv.fold_rows` always cuts the inner set, and `fit_predict("rf", ...)` fits on `train` (the rest) only. Random Forest has no early stopping, so about 15% of each fold's training cells are discarded for nothing, in tuning (R4 grid) and in the runs. That makes R4 look worse than it is, and its CV fits use less data than its final retrain.

**Fix:** for `rf`, fit on `pd.concat([train, inner])`. That's a single change inside `fit_predict`.

### A3. Fold files from before the run-key mechanism are adopted silently (`cv._check_key`)

If a run directory has `fold_*.parquet` files but no `run_key.json`, which is the case for every folder in today's `quick/`, `_check_key` writes the new key and `run_cv` resumes from those files. The old files are exactly the stale ones B3 was about.

**Fix:** if fold files exist and there is no `run_key.json`, refuse unless `--fresh`, and with `--fresh`, delete them.

### A4. Optuna studies resume across changes in config or code (`tune.tune_optuna`)

The study name is `{model}_{dataset}_{mode}`, with `load_if_exists=True`. If the config changes (learning rate, search ranges, cap) or the fit code changes, a re-run adds new trials to the old ones. The 1-SE band and the "best" trial are then picked from a mix of settings. It's the B3 problem for tuning.

**Fix:**
- Put the config hash in the study name (`lgbm_A_full_<hash>`), or store it as a study attribute and refuse a mismatch.
- Record the study name in `params_*.json`.

## Fix before Phase 3

### A5. R1 and R2 overwrite each other's final model (`final.fit_final`, `score_test.load_model`)

- **The overwrite:** the model file is named per **model**, not per run (`final_lgbm.txt`). Finalising R2 after R1 replaces R1's booster.
- **The mismatch:** `final_R1.json` would still describe R1, while `score_test --run R1` loads R2's 47-feature booster and feeds it 62 columns.
- **Fix:**
  - Name the files per run (`final_R1.txt`, `final_R3.json`, `final_R4.joblib`) and record the file name in `final_{run}.json`.
  - In `score_test`, load the file named there, and assert the model's feature count equals `len(cols)`.

### A6. R2's final model uses R1's tree count (`final.main`)

`n_trees = par["n_trees_final"]` comes from `params_lgbm.json`, which is R1's tuning on dataset A. But the plan says R2 re-picks its trees, and in cross-validation it does: early stopping per fold, on dataset C.

**Fix:** for R2, take the median of `n_trees` from `full/R2/group10/oof.parquet` (each fold stores it). Record which source was used. The same rule works for R1, as a cross-check against `n_trees_final`.

## Minor

- **The cap count is lost on resume.** `run_cv` counts `capped` only for folds fitted in the current call. Compute it from the stored `n_trees` in the OOF (`> cap - patience`, see A1), so a resumed run reports every fold.
- **The Random Forest grid can't resume.** All 120 fits restart after a crash. Cache each grid point's result (a small JSON per point) and skip points already done.
- **`summarise.py` table call.** `p.reindex(columns=[...], level=0)` is applied to a flat column index; `level=` is meant for a MultiIndex. Plain column selection, `p[["rmse", "mae", "bias", "n"]]`, does the intended thing without depending on how pandas treats `level` on a flat index.
- **Docstring.** The `leakage_test.py` docstring says "the scheduling/early-stopping set never touches a held-out or test cell". The code checks pairwise disjointness. Inner is a subset of training, which `data.fold_rows` already keeps clean, so the claim holds, but say that's why.

## Still the author's decision

- **Learning rate.** `config.json` still has 0.05. If the full runs show most folds not stopping (A1 makes that visible), 0.1 is the remedy suggested in round 1. Deciding before the full tuning saves a re-run.
- **Where Phase 2 runs.** This affects Random Forest, `n_jobs = -1`, which doesn't work in the Windows sandbox.

## What to send back (code only, no runs)

Send the changed files, with a note per finding (A1–A6, minor) and the lines that implement each fix. The next review is a code audit.
