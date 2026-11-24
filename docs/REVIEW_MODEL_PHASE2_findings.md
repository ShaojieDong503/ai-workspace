# Phase 2 code review: findings

Review of `code/model` Phase 2 against `docs/MODEL_RUN_PLAN.md`, `docs/PROMPT_model_runs.md` and the brief `docs/REVIEW_MODEL_PHASE2.md`.

**What I read:** `cv.py`, `metrics.py`, `models.py`, `tune.py`, `runs.py`, `summarise.py`, `final.py` and `score_test.py`, plus the quick outputs: `params_lgbm.json`, `results_dev_quick.md` and the `quick/` folders.

**Method:** a code audit. I examined the logic of each module and traced how parameters, seeds and rows flow between them. Findings rest on the code and the outputs already on disk, not on test runs.

> **For the coding agent: no local runs this round.** Phase 2 won't be run on this machine for now. Make the fixes below as code changes only. Don't run the quick or full pipeline, the tuning, or the leakage test to verify them. The next review checks the fixes by reading the code.

**Verdict: don't start the full runs yet.** Six defects would make the full results wrong or untraceable (B1–B6). B1 is the most serious: tuning searches with different settings from the ones the runs use. Two defects in the Phase 3 scripts must be fixed before Phase 3 (P3-1, P3-2). The engine, row selection, metrics and the 1-SE band are sound.

## Checked and fine

- **`cv.run_cv`:**
  - It takes rows only from `data.fold_rows`, then subsamples (quick mode only), then cuts the inner split.
  - `inner_split` picks whole training cells per region, and `assert_clean` checks the inner set.
  - `leak_check` runs before any metric is written.
- **`metrics.py`:**
  - Six groups, with RMSE, MAE, bias and R², pooled and per fold.
  - R² is dropped for `gt25` and `gt50` only (brief item 6), and `n` is stored.
- **`tune.py`, 1-SE rule (brief item 3a and 3c):**
  - The band is `value <= best + SE`, with SE = standard deviation (ddof 1) of the best trial's per-fold RMSE / √10.
  - The simplest-trial rule is applied to the band, not to all trials.
- **R2 (brief item 4):** `runs.py` uses R1's chosen LightGBM parameters on dataset C, and the number of trees is re-picked per fold by early stopping. This matches the plan.
- **XGBoost prediction (brief item 7):** `best_iteration` counts from 0, so `iteration_range=(0, best + 1)` is right. But see B5 for its fallback.
- **Phase 3 guard (brief item 8):** both scripts refuse unless `OPEN_TEST` exists and `--open-test` is passed. That matches the prompt. No separate decision file is required: creating `OPEN_TEST` is the decision.

## Blockers before the full runs

### B1. Tuning ignores the configured model settings (`tune.py`)

`objective()` passes the raw sampled dict straight to `evaluate()`, and `evaluate()` passes it to `fit_lgbm` / `fit_xgb`. It never goes through `models.lgbm_params` or `models.xgb_params`, which `runs.py` does use. So during tuning:

| Setting | Tuning uses | Runs use |
|---|---|---|
| LightGBM learning rate | **0.1** (library default) | 0.05 |
| LightGBM `bagging_freq` | **0**, so `bagging_fraction` is searched but has no effect | 1 |
| LightGBM `verbose`, `num_threads` | library defaults | config values |
| XGBoost learning rate | **0.3** (library default) | 0.05 |
| XGBoost `nthread` | library default | config value |

**Why it matters:**
- The 1-SE choice and `n_trees_final` are picked at one learning rate and used at another.
- R1's search over `bagging_fraction` is wasted.
- It may also explain the brief's quick results, where R3 (XGBoost) beats R1.

**Fix:**
- In `evaluate`, build `params = models.lgbm_params(cfg, p)` or `models.xgb_params(cfg, p)` before fitting.
- Make tuning and the runs call one shared `fit_predict(model, params, train, inner, score)`, so this can't diverge again.

### B2. Quick and full runs write the same parameter file

`tune.py` always writes `params_{model}.json`. The current `params_lgbm.json` has `"quick": true` and `n_trees_final` 301. A full `runs.py R1` started before full tuning finishes would silently load the quick parameters.

**Fix:**
- Write `quick/params_{model}.json` in quick mode.
- In `runs.py`, assert that the parameter file's `quick` flag matches `--quick`.

### B3. Resume reuses fold files built with different settings (`cv.run_cv`)

`run_cv` skips any fold whose `fold_<label>.parquet` exists. It never checks that the file was made with the current parameters, config or features. So after re-tuning, or after fixing B1, B4 or B5, a re-run mixes old and new folds in one OOF file without any warning.

The brief's partial-write concern (item 5) is the same problem.

**Fix:**
- Write a `run_key.json` in each run directory: the config hash, a hash of the parameter set, the feature hash, and the quick flag.
- On resume, refuse if the key differs, unless `--fresh` is given, which deletes the old fold files.
- Write each fold to `*.tmp`, then rename.
- The existing `data/interim/model/quick/` folder is stale and shouldn't be trusted or resumed from. Delete it when Phase 2 is next run. It holds `fold0.parquet`…`fold9.parquet` from the old naming next to `fold_fold0.parquet`… The two sets already differ in size (fold0 250,116 vs 249,908 bytes), because of B4.

### B4. Seeds come from `hash()` (P2-1, confirmed)

`cv.fold_rows` seeds the subsample and the inner split with `hash(spec.label) % 1000`. Python salts string hashes per process, so:
- the full runs' **inner early-stopping cells** change on every invocation, not just the quick subsample
- R1 and R2 get different inner sets
- a resumed run differs from an uninterrupted one

**Fix:** use `zlib.crc32(spec.label.encode()) % 1000`, or the fold's index in `FLD.plan`. Both are stable across processes.

### B5. LightGBM tree count is one too high (`models.best_iter`)

`best_iter` returns `best_iteration + 1` for both libraries. LightGBM's `best_iteration` already counts from 1: it is the number of iterations of the best model. XGBoost's counts from 0. So every LightGBM fold records one tree too many. The quick `n_trees_final` of **301 against a cap of 300** shows it.

Also, `predict_xgb` falls back to `best_iteration = 0`, which means **one tree**, if the attribute is missing.

**Fix:**
- Add +1 only for XGBoost.
- Make the XGBoost fallback `booster.num_boosted_rounds()`.

### B6. Random Forest tuning is not a grid, and "simplest" sorts `max_features` as text

- **Not a grid.** R4 uses TPE over two categorical parameters for 12 trials. TPE can repeat points and skip others, so it isn't the plan's grid. Use `optuna.samplers.GridSampler` over the full `max_features` × `min_samples_leaf` grid (12 points), or a plain loop.
- **Wrong sort (brief item 3b).** `simplest` sorts `max_features` as a string: `"0.3" < "0.5" < "sqrt"`. But `sqrt` is the smallest of the three (√62 ≈ 8 features, about 0.13). Map each value to its feature fraction before sorting.
- **The sandbox issue** is an environment limit. Run R4 from your own terminal (`n_jobs = -1`) rather than serially. Serially it is roughly 150 fits of several minutes each.

## Fix before Phase 3

- **P3-1. `final.py` and `score_test.py` are LightGBM-only.**
  - `final.py --run R3` or `R4` would train a LightGBM with `params_lgbm.json` and label it R3 or R4.
  - `score_test.py` always uses dataset A's columns, which is wrong for R2.
  - Fix: dispatch on the run (model type, parameter file, dataset) through the same shared fit function as B1.
- **P3-2. The bootstrap covers RMSE only.** The prompt asks for confidence intervals for every metric and group.
  - Add MAE, bias and R².
  - Report coverage per group as well as overall.
  - Report the number of cells behind each interval: the spatial test has 6, so a cell bootstrap over 6 cells is very coarse. Say so in the results.

## Design questions from the brief

1. **Grouped 10-fold is optimistic for tuned runs.** Yes, mildly: R1 and R3 are chosen on those folds. Mark the grouped 10-fold block in `results_dev.md` as "tuning folds (selection-biased)". Present leave-one-region-out and leave-one-year-out as the honest development estimates. The final word stays with the test sets.
2. **Early stopping never triggers** (best iteration 4,956 out of 5,000).
   - **Check first:** after B1, the runs train at 0.05 while that measurement may come from a different setting. Re-measure when Phase 2 is run.
   - **If it still runs to the cap:** raise the learning rate to 0.1 for every model, which roughly halves the trees and the time, and keep the cap at 5,000.
   - **Flag it:** record in `run_meta.json` how many folds hit the cap, so it's visible.
   - **On the median:** "median of per-fold best iterations" is only meaningful when folds actually stop early. If most hit the cap, it just means "the cap".
3. **The inner early-stopping set is small.** One cell per region, about 6 of the ~32 training cells. That's acceptable, but early stopping on 6 cells is noisy, which is another reason to prefer a higher learning rate with a real stopping point.
4. **Metric groups (brief item 6).** `summarise.py` shows RMSE only, and no `n`. Add `n` per group, plus MAE and bias. North and heavy smoke are small in some leave-one-region-out and leave-one-year-out folds.

## Minor

- **P2-2 (confirmed):** `leakage_test.py` doesn't use `cv.dry_run`, although the `cv.py` docstring says it does. Wire it in: for every scheme, run `cv.dry_run(..., quick=False)` and assert:
  - train, inner and score are pairwise disjoint
  - the score keys are exactly the development rows, each once

  That also covers the inner set.
- **`leak_check` doesn't check completeness.** It checks for duplicates and for rows outside development, but not that every development row is scored. Add `set(oof keys) == development keys` for every scheme.
- **Dead code:** `runs.make_lgbm` has `cols = None`. Remove it.
- **Explicit seeds:** pass `seed` (LightGBM) and `seed` (XGBoost) from `seeds.tune`, so fits don't rely on library defaults.

## Tuning budget (author question)

At ~77 s per fit, 100 trials × 10 folds is about 21 hours for R1. Three options, in order of preference:
1. **A learning rate of 0.1** (see design question 2).
2. **Optuna pruning:** report the running mean after each fold and use `MedianPruner(n_warmup_steps=3)`. Bad trials stop after 3–4 folds.
3. **Fewer trials.** 50 is usually enough for 8 LightGBM parameters with TPE.

Options 1 and 2 together should bring R1 under about 6 hours.

## What to send back (code only, no runs)

Don't run anything locally. Send the changed files, with a short note per finding:
- **B1–B6:** what changed, and where.
- **P3-1 and P3-2:** the same.
- **P2-2 and the minor items:** done, or why not.

For each fix, point to the lines that implement it. The next review is a code audit of those changes. The runs (quick mode first, then full) happen later, where Phase 2 is executed. At that point these checks apply:
- the quick `params_*.json` shows the configured learning rate, and `n_trees_final` ≤ the cap
- a changed `run_key` is refused
- `leakage_test.py`, now through `cv.dry_run`, passes
