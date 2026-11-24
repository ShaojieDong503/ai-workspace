# Phase 1 review (round 3): cell-level splits

Review of `code/model` after the author's decision (18 Nov) to drop sites and split at cell level. The decision is recorded in `docs/MODEL_RUN_PLAN.md` §1–2.

**What I did:**
- Read `config.json`, `folds.py`, `splits.py`, `leakage_test.py` and `feature_reduction.py`. `data.py` and `features.py` are unchanged since round 2.
- Read `splits_summary.md` and `feature_reduction_report.md`.
- Ran `leakage_test.py`, which only reads data. It printed **PASS**.

**Verdict:** the splits follow the author's rules, the seeds are used, and nothing leaks. **One bug must be fixed before approval** (R3-1): it's in the summary file only, and the splits themselves are fine. One regression in the feature ranking needs fixing before the list is frozen (R3-2).

## Checked and fine

- **Spatial test:** one cell drawn per region with `rng.choice` from the sorted cell list, seeded by `seeds.split`; 6 cells.
- **Calibration:** 6 cells drawn from the rest with the same generator.
- **Folds:** the development cells shuffled with `seeds.fold` and dealt round-robin into 10 folds of 3–4 cells (12,001–16,834 rows).
- **Site code:** all of it has been removed.
- **Row totals:** development 149,846 + calibration 21,491 + test 26,275 = 197,612, the plan's §7 count.
- **Leakage test, all three schemes:** 149,846 rows scored exactly once, 0 shared cells or years, 0 outside development. All three breakages raise through the real paths.
- **Feature reduction:**
  - NaN correlations no longer pass (`not (rho >= thr)`).
  - The rejected-pair table is de-duplicated (3 genuine rows).
  - `aod_nbr_25km_d0` / `aod_nbr_50km_d0` is now 10/10, via the declared nested family.

## Must fix

### R3-1. `splits_summary.md` writes one line per data row (8.7 MB, 197,659 lines)

`splits.py`: the "Picked cells" loop iterates over `r = splits.merge(dev, ...)`, which has one row per **cell-day**, not per cell. So every calibration, development and test cell is printed once for each day it has data.

The development cells also show "nearest dev cell (km) = 0.0", because each one is its own nearest development cell. That's meaningless.

**Fix:**
- Build the table from `splits.merge(near, on="cell_id")`, one row per cell, and list only the `spatial_test` and `calibration` cells.
- In the role-by-region table, leave the distance column blank for development.
- Re-run `splits.py --force`. The draw is seeded, so the splits come out identical; only the summary changes. Check that `splits.parquet` and `folds.parquet` are byte-identical to the current files, or at least equal as tables.

## Fix before freezing the feature list

### R3-2. The within-group ranking now ignores gain for complete columns

`feature_reduction.dominant_column` puts the member with the widest coverage first, and breaks ties by position in the group. In every group where both columns are complete, which is most of them, the "ranking" is now just list order. Round 2 ranked by gain. Examples from the current report:
- `aod_nbr_25km_d0_n` is ranked above `aod_nbr_50km_d0_n`. That's the opposite of the nesting the config declares.
- `hms_any` is ranked above `hms_n_polygons`, and `merra_speed` above `merra_speedmax`.

**Fix:**
- For a **declared** nested family, put the wider member first (50 km before 25 km).
- For every other group, rank by gain, then completeness, as in round 2.

**Smaller points:**
- **Check the declared nesting.** For each family in `nested_families`, assert that the narrower column is never present where the wider one is missing (A-only = 0 on the development rows). Today the nesting is declared but never checked.
- **Report wording.** It says "Nested pairs (one column present wherever the other is) are merged whatever their overlap". Change it to "Pairs declared in `feature_reduction.nested_families` are merged…": the code merges only declared pairs, on purpose, so that `aod055` / `aod_last_obs` is not merged.

## Notes for the author (follow your rules; not bugs)

- **The North test cell is thin.** The random draw picked 856931: 1,095 rows, 2010–2012 only, 1 day above 25 µg/m³. It sits 5 km from development cell 856930, the same monitor after it moved in 2013. North test results will rest on three years and one smoke day. This follows from "random, no other constraint".
- **No North calibration cell.** Calibration was drawn at random across Ontario, so it has 2 East cells and none in North. A per-region interval width can't be reported for North, and the 90% interval there is calibrated on southern cells only. If that matters, draw calibration one per region, like the test set. Decide before re-running R3-1; the re-run must not change the seed.
- **Test and calibration cells next to development cells:**

  | Cell | Role | Next to |
  |---|---|---|
  | 918734 | test, region 1 | a development cell 5 km away |
  | 856931 | test, North | 856930, development (relocation) |
  | 916606 | calibration | 916605, development (relocation) |
  | four other calibration cells | calibration | a development cell 5 km away |

  These are accepted under the cell-level decision. The per-cell distances (R3-1) make them visible in the paper.

## Minor

- **"Scoring the test year" passes for the wrong reason.** The breakage raises with "empty selection", because `scoring_rows` keeps 2010–2023 before applying the year filter. The 2024 rows can't be reached, so the guard holds, but the message doesn't show it. Optionally, make `scoring_rows` reject `years` outside 2010–2023 with an explicit error, as `fold_rows` already does.
- **No log is saved.** `leakage_test.py` prints to the console only. Save its output to `data/interim/model/leakage_test.log`, so the PASS is on record with the splits.

## Re-run expected

```
python code/model/splits.py --force       # R3-1; same seeds, splits must be unchanged
python code/model/leakage_test.py         # save the log
python code/model/feature_reduction.py    # R3-2
```

After that, Phase 1 is ready to approve. Then freeze `features_frozen.json` and set `contingency.criterion`.
