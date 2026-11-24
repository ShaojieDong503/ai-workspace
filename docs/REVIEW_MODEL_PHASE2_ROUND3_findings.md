# Phase 2 code review (round 3): findings

**Method:** a code audit of `models.py`, `cv.py`, `tune.py`, `final.py`, `score_test.py`, `summarise.py` and `leakage_test.py` (all dated 19 Nov, 14:15) against `docs/REVIEW_MODEL_PHASE2_ROUND2_findings.md`. I ran nothing.

> **For the coding agent:** code changes only, with no local runs. Point to the lines for each fix.

**Verdict:** A1–A6 and the minor items are fixed. Two new defects come from the same mistake: **counting per row where the code means per fold** (C1, C2). Both are one-line fixes. After them, Phase 2 is ready to run.

## Round-2 findings: status

| Finding | Status |
|---|---|
| A1, "never stopped" flag | **Fixed.** It is now `n > cap - patience` in `fit_predict`, renamed `not_stopped`. The cv-side count has a new defect (C1). |
| A2, Random Forest discarded the inner cells | **Fixed.** `fit_predict("rf")` fits on `train + inner`. |
| A3, fold files without a run key | **Fixed.** `_check_key` refuses unless `--fresh`, and `--fresh` deletes them. |
| A4, Optuna studies spanning config changes | **Fixed.** The config hash is in the study name, which is recorded in `params_*.json`. The Random Forest grid cache is keyed the same way. |
| A5, R1 and R2 overwrote each other's model | **Fixed.** Model files are named per run and recorded in `final_{run}.json`. `score_test` loads that file and checks the feature count against the dataset for all three libraries. |
| A6, R2 borrowed R1's tree count | **Fixed in intent.** `choose_trees` reads the run's own group10 OOF and refuses R2 without it. The median it takes has a new defect (C2). |
| Minor: cap count on resume, resumable RF grid, the `summarise` table call, leakage docstring | **Fixed.** |

## Fix before the full runs

### C1. `folds_not_stopped` counts rows, not folds (`cv.run_cv`, line 175)

```python
not_stopped = int((oof["n_trees"] > cap - patience).sum())
```

`oof` has one row per scored cell-day, and every row repeats its fold's `n_trees`. So this counts **rows**: if every fold ran to the cap, it would report ~145,315, not 10, 6 or 14.

**Fix:**

```python
per_fold = oof.groupby("fold")["n_trees"].first()
not_stopped = int((per_fold > cap - patience).sum())
```

Also skip the count for `model == "rf"`, which doesn't stop early. In quick mode (cap 300, patience 100), Random Forest's fixed 300 trees would otherwise count as "not stopped". Write `null` for Random Forest.

## Fix before Phase 3

### C2. The final tree count is a row-weighted median (`final.choose_trees`)

```python
med = int(pd.read_parquet(oof, columns=["n_trees"])["n_trees"].median())
```

**Problem:** this is the median over **rows**. Folds with more scored rows get more weight, and fold sizes range from 3 to 4 cells and roughly 12,000 to 17,000 rows. The plan's rule is the median of the **per-fold** best iterations, which is what `tune.py` computes for `n_trees_final`. The two numbers can therefore disagree for R1.

**Fix:** read `fold` and `n_trees`, then take `.groupby("fold")["n_trees"].first().median()`.

**Also:** record the per-fold values in `final_{run}.json` next to the median. For R1, they should match `params_lgbm.json` `best_iters`, because the frozen run uses the same parameters, folds and seeds as the chosen trial. A mismatch would point to non-determinism worth knowing about.

## Minor

- **`final.py` doesn't check the parameter file's mode.** `runs.load_params` refuses a quick parameter file in a full run, but `final.py` doesn't. Quick files now live under `quick/`, so it can't happen by path, but one `assert not par["quick"]` keeps the two scripts consistent.

## Still the author's decision

- **Learning rate:** `config.json` still has 0.05. With C1 fixed, `folds_not_stopped` in `run_meta.json` and `not_stopped_chosen` in `params_*.json` will show whether early stopping fires. The study name now includes the config hash, so changing the learning rate later starts a clean study rather than mixing trials. The cost of changing it after the full tuning is a second tuning run.
- **Where Phase 2 runs:** Random Forest with `n_jobs = -1`.

## What to send back (code only, no runs)

The changed lines for C1, C2 and the minor item. After that, Phase 2 is ready to run, starting with the quick pass.
