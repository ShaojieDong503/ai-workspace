# Phase 2 code review brief: CV engine, metrics, tuning, runs

For a reviewer of `code/model` Phase 2. Phase 1 (splits, features) is approved and
frozen; do not re-review it. Authority: `docs/MODEL_RUN_PLAN.md`.

The approved split: 6 test cells, 6 calibration cells, 36 development cells,
145,315 development rows. Config hash is recorded in every `run_meta.json`.

## What exists and what was run

| module | what it does | run? |
|---|---|---|
| `cv.py` | the engine: folds, inner split, resume, OOF, metrics, `dry_run` | yes, quick |
| `metrics.py` | 6 groups x 4 metrics, pooled and per fold | yes, via cv |
| `models.py` | LightGBM / XGBoost / Random Forest fitters | yes, except RF |
| `tune.py` | Optuna (LightGBM, XGBoost) and a grid (RF) | LightGBM and XGBoost only |
| `runs.py` | R1-R4 over group10 / loro / loyo | R1, R2, R3 |
| `summarise.py` | `results_dev_quick.md` | yes |
| `final.py`, `score_test.py` | retrain, conformal intervals, test scoring | **written, guarded, never run** |

Quick mode = same splits, 10% of training rows per fold, 300 rounds, 3 trials,
output under `data/interim/model/quick/`. Full mode has not been started.

```
python code/model/tune.py --model lgbm --quick     # 25 s
python code/model/runs.py R1 --quick               # 29 s, all three schemes
python code/model/summarise.py --quick
```

## Two defects I found and have not yet fixed — check these first

**P2-1. The quick-mode seeds are not reproducible.** `cv.fold_rows` derives the
subsample and inner-split seeds from `hash(spec.label)`. Python salts string
hashing per process, so the value differs between runs:

```
python -c "print(hash('fold0'))"   -> -1817795847438914752
python -c "print(hash('fold0'))"   ->  579689743987623095
```

Consequences: two quick runs sample different rows, so quick numbers are not
comparable between runs; and within one tuning study, trials do not share the
same 10% sample. The fix is a stable per-fold integer (a lookup from the label,
or the fold's position in the plan), not `hash()`.

**P2-2. `cv.dry_run` is not consumed by `leakage_test.py`.** Round-1 F3 asked
for the engine's dry run to be the leakage test's evidence, so the test would
constrain the engine rather than a copy of its logic. `cv.dry_run` exists and I
verified it by hand across all three schemes, but `leakage_test.py` still builds
its own fold list from `folds.plan` and calls `data.fold_rows` directly. The
shared fold plan binds the two, the dry run does not.

## Ranked things to examine

**1. Tuning and reporting share the same folds.** `tune.evaluate` scores every
trial on group10, and the chosen parameters are then reported on group10 as well.
So R1's and R3's group10 numbers are optimistically biased by selection; the
honest comparison is LORO and LOYO. The plan says group10 is for tuning only,
which is why LORO/LOYO exist, but the summary presents all three side by side.
Is that presentation misleading, and should group10 carry a warning?

**2. Early stopping never triggers.** One full-size fit: 77 s, `best_iter` 4,956
against a cap of 5,000. Patience 100 is therefore inert and `n_trees_final` is
effectively the cap. Two questions: is 5,000 the right cap, and is "median of
per-fold best iterations" meaningful when no fold stops early?

**3. The 1-SE rule.** `tune.simplest` ranks LightGBM by fewest leaves then
largest `min_data_in_leaf`; XGBoost by smallest depth then largest
`min_child_weight`; RF by largest `min_samples_leaf` then `max_features` as a
string. Check that (a) the rule is applied to the band and not to all trials,
(b) the RF comparison of `max_features` by string is sound, (c) the SE uses the
best trial's per-fold RMSE over the number of folds.

**4. R2 re-picks only the trees.** It shares R1's parameters and differs only in
dataset C, with the number of trees chosen by early stopping — verify that is
what `runs.py` does, rather than loading `params_lgbm` wholesale for both.

**5. Resume and partial writes.** Each fold writes `fold_<label>.parquet` and
`run_cv` skips it if present. A crash mid-write leaves a truncated parquet that
the next run would accept. Phase 1 used temp-then-rename; Phase 2 does not.

**6. Metric groups.** `north` and `heavy_smoke` can be small; check the `n` column
before trusting those numbers, and that R² is absent for `gt25`/`gt50` only.

**7. XGBoost prediction.** `predict_xgb` uses `iteration_range=(0, best+1)`;
confirm that matches `best_iteration` semantics and that it is not double-counting.

**8. Guards.** `final.py` and `score_test.py` refuse without `OPEN_TEST` and
`--open-test`; I showed both refusals. Confirm the guard also covers the case
where `OPEN_TEST` exists but the decision file does not.

## What I verified by running

- `cv.dry_run`, all three schemes: no overlap between the inner set and the
  scoring set, no overlap between the inner set and the remaining training rows,
  and each scheme scores 145,315 distinct rows exactly once.
- Quick runs complete for R1, R2, R3 on group10, LORO and LOYO; `run_meta.json`
  records config hash, feature hash, target fingerprint, versions, wall time.
- `summarise.py` reads those metrics and writes `results_dev_quick.md`.

## Defects already found and fixed during the run

- `metrics.by_fold` cast the fold label with `int("fold0")` — now keeps the label.
- Per-fold files were globbed as `fold*.parquet`, so LORO (`region1.parquet`) and
  LOYO (`year2010.parquet`) produced an empty frame — now `fold_<label>.parquet`.

## Do not re-report

- Random Forest cannot run under the Windows sandbox (`n_jobs=-1` needs joblib
  process pipes; escalation declined). That is an environment limit awaiting a
  decision, not a code defect.
- Quick-mode numbers are noisy by construction (10% of rows). They are there to
  time the pipeline, not to be interpreted.
- Early stopping being inert is listed as a design question above, not as a bug.

## Open questions for the author, not the reviewer

- Tuning budget: 100 trials at ~77 s per fit is ~21 h for R1 alone.
- RF: serial `n_jobs = 1`, or run outside the sandbox.
