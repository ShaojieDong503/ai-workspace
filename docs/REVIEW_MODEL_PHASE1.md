# Phase 1 review brief: data, splits, feature reduction

For an independent reviewer of `code/model`. Only Phase 1 exists. Phases 2–3
(CV engine, tuning, runs, final models, test scoring) are **not written**, by
instruction — the author runs each phase only after the previous one is approved.

The authority is `docs/MODEL_RUN_PLAN.md` (revised 18 Nov). Where this code and the
plan disagree, the plan wins.

## What was run, and what it printed

| command | result |
|---|---|
| `python code/model/data.py` | Ontario rows 208,093, 48 cells; **sanity `{rows: 197612, gt25: 1318, gt50: 134, north: 25005}`** — matches plan §7 exactly |
| `python code/model/splits.py` | test 10, calibration 6, development 32; writes `splits.parquet`, `folds.parquet`, `splits_summary.md` |
| `python code/model/leakage_test.py` | all three schemes OK, 136,757 dev rows each scored exactly once, 0 duplicates; 3 deliberate breakages all raised |
| `python code/model/feature_reduction.py` | 62 columns, 0 constant, 15 correlation groups, all present in 10/10 folds |

Nothing else was run. No metric has been computed on any row. `phase3_test_rows()`
is defined in `data.py` and **called by nobody** — grep confirms it.

## What is deliberately absent

- Phase 2: `cv.py`, `metrics.py`, `models.py`, `tune.py`, `runs.py`, `summarise.py`.
- Phase 3: `final.py`, `score_test.py`.
- `features_frozen.json` — the author freezes it.
- `contingency.criterion` is `null` in `config.json` — deliberately.

## Where I am least confident (please look here first)

1. **The leakage test does not yet constrain the engine, because the engine does
   not exist.** `leakage_test.py` builds folds itself from `folds.parquet` and
   calls `data.fold_rows`. When `cv.py` is written it must use the same function;
   nothing enforces that today. If the reviewer can suggest a way to make the
   test bind the engine (e.g. the engine exposing its fold plan for inspection),
   that would close a real hole.

2. **The early-stopping set has no assertion yet.** The prompt requires "no test
   cell, no calibration cell, no 2024 row, in either the training set or the
   early-stopping set". The inner 15% split is cut inside the fold in Phase 2, so
   its assertion is not implemented. The plan is to call `data.assert_clean` on
   the inner frame as well — but that is a promise, not code.

3. **Fold balance is greedy and not optimal.** Folds run 10,027–14,959 rows
   (1.49×) and hold 2–4 cells; **fold 8 has only 2 cells**. A reviewer may think
   a 2-cell fold is a bug; it is the greedy allocator balancing on rows, not cell
   counts. My view: acceptable for grouped CV, but it will widen fold variance.

4. **The region-6 test cell is thin**: 1,095 rows, 1 day above 25 µg/m³. North is
   one of the six reported groups, so its test estimate will be noisy. This is a
   consequence of 48 cells, not of the code.

5. **`smoke_aod055` is 1.1% complete and sits inside the `aod055` correlation
   group.** The draft suggests `aod055` as the representative, so it would drop
   out — but if the author keeps it, the model gets a column that is almost always
   NaN. Worth an explicit recommendation.

## Things that look like bugs but are not

- **One spatial-test cell is labelled `province_state = "Michigan"`** (cell
  944250, region 4, 2,165 rows). It holds an Ontario monitor, and the region
  follows the monitor while the province follows the polygon
  (`data/processed/README.md` §5). Excluding it would drop a real Ontario site.
- **Dataset A is 62 columns, not 63.** `maiac_covered` is excluded because it is
  constant over Ontario (verified: one unique value, 1). The 63rd column in the
  training table is the target.
- **`data.load()` keeps 208,093 rows, not 197,612.** The extra 10,481 are 2024;
  the §7 check filters to 2010–2023 itself.

## Environment traps a reviewer will hit

- **Do not set `PYTHONPATH=D:\ai-workspace\project\.pylibs`.** It breaks
  `import pyarrow`, which breaks every parquet read. The model code imports
  everything from the user site-packages; `.pylibs` was only for the earlier
  panel work.
- pandas is 3.0.3: `astype("datetime64[D]")` raises, and merging
  `datetime64[ms]` with `[ns]` needs care. Date keys in the leakage test are
  strings for this reason.
- `to_markdown()` needs `tabulate`, which is not installed; the summary files use
  fenced `to_string()`.

## What to run to reproduce

```
python code/model/data.py
python code/model/splits.py --force
python code/model/leakage_test.py
python code/model/feature_reduction.py
```

## What not to run

Anything that does not exist yet, and anything that touches
`data/interim/model/splits.parquet` without `--force` — the splits are frozen and
every later number depends on them.

## Decisions pending from the author

1. Freeze the feature list as `features_frozen.json`.
2. Set `contingency.criterion` in `config.json`.
3. Keep or drop `smoke_aod055`.
