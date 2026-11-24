# Phase 1 review: findings

Review of `code/model` Phase 1 against `docs/MODEL_RUN_PLAN.md`, `docs/PROMPT_model_runs.md` and the brief `docs/REVIEW_MODEL_PHASE1.md`. I read all six files and the outputs in `data/interim/model/`, and ran one distance check on the frozen splits. I did not re-run any script.

**Verdict: not approved yet.**
- **Blocks Phase 2:** F1. The splits and folds have to be rebuilt.
- **Must be fixed before the feature list is frozen:** F2.
- **Must be fixed with the Phase 2 code:** F3.
- **Minor:** F4–F6.

**What is correct:**
- the row filters (`training_rows`, `scoring_rows`, `fold_rows`)
- the sanity counts
- the scoring-row exactness checks
- the coverage check (every development row scored once)
- the Phase 3 isolation of `phase3_test_rows`

## F1. Neighbouring and relocated monitors sit on both sides of every split (blocker)

Cells are 5 km apart. Many Ontario monitors either moved one cell over during 2010–2024, or have a second monitor in the next cell. The splits and folds treat these as independent locations.

Pairs within 10 km that ended up in different roles or folds (distances from `grid/cells.parquet`):

| Cell A | Role A | Cell B | Role B | km | Overlap |
|---|---|---|---|---:|---|
| 916605 | calibration | 916606 | dev, fold 3 | 5 | **relocation**: 2017–24 vs 2010–16 |
| 856930 | calibration | 856931 | **spatial test** (North) | 5 | **relocation**: 2013–24 vs 2010–12 |
| 944250 | **spatial test** | 944251 | dev, fold 4 | 5 | **relocation**: 2010–15 vs 2016–24 |
| 944250 | **spatial test** | 946380 | calibration | 10 | **relocation**: 2010–15 vs 2017–24 |
| 922993 | **spatial test** | 924058 | dev, fold 6 | 5 | simultaneous, 5,303 shared days |
| 918734 | **spatial test** | 919799 | dev, fold 5 | 5 | simultaneous, 257 shared days |
| 930446 | calibration | 930445 | dev, fold 0 | 5 | simultaneous, 5,026 shared days |
| 918737 | dev, fold 2 | 919802 | dev, fold 4 | 5 | **relocation**: 2010–19 vs 2019–24 |
| 940010 | dev, fold 3 | 941075 | dev, fold 0 | 5 | **relocation**: 2010–12 vs 2013–24 |
| 863407 | dev, fold 3 | 864472 | dev, fold 9 | 5 | simultaneous, 2,812 shared days |
| 919799 | dev, fold 5 | 920865 | dev, fold 1 | 7 | simultaneous, 258 shared days |

Within 25 km there are 49 such pairs, and within 50 km there are 110.

**Why it matters:**
- **Shared predictors.** Every MERRA-2 input is shared at ~50 km, and fire, HMS and the AOD context columns are nearly identical 5 km apart. So two neighbouring cells give near-duplicate predictor rows with near-equal PM₂.₅.
- **Relocations.** A monitor moved one cell over is the same site in a different period.
- **The test set isn't unseen.** 4 of the 10 spatial-test cells have a development or calibration cell 5 km away. That includes the only North test cell, which is the earlier period of a calibration site. The spatial-test estimate will be optimistic, and so will the grouped 10-fold CV used for tuning.
- **The 1-SE rule is affected too.** Its fold SE comes from those same folds.

**Fix:**
1. **Build sites:** group cells by single-linkage clustering at a distance threshold. I suggest 10 km, which catches every relocation and co-located pair above. Put the threshold in the config.
2. **Split by site:** assign test, calibration, development and the 10 folds by site, not by cell.
3. **Check region:** no site may straddle a region. That holds for every pair above; assert it.
4. **Inner early-stopping split:** cut it by site as well.
5. **Leakage test:** add "no site appears in two roles or two folds", plus a nearest-cross-role distance table in `splits_summary.md`.
6. **Optional sensitivity:** city-level grouping (25 km). It would merge most of Toronto, so it's too coarse for the main split.

Rebuilding the splits now is legitimate: nothing has been fitted or scored on them. The re-split must not use any target-derived quantity beyond row counts.

## F2. The draft feature list would drop the most useful AOD columns (fix before freezing)

**Problem 1: correlation computed on pairwise-complete rows.** `feature_reduction.correlation_groups` uses `DataFrame.corr`, which computes each pair only on the rows where both columns are present. For NaN-heavy columns this measures the wrong thing:

- **`aod_last_obs` vs `aod055`.** By definition, `aod_last_obs` equals `aod055` on every day AOD is observed (age 0). So on their co-observed rows the correlation is ~1. But `aod_last_obs` exists to fill the ~70% of rows where `aod055` is missing (completeness 0.731 vs 0.297). It's the AOD column with the highest gain (198k vs 10k for `aod055`).
- **`smoke_aod055` vs `aod055`.** Same problem: only 1.1% of rows overlap.
- **The draft picks `grp[0]` as each group's representative.** That keeps `aod055` and drops `aod_last_obs` and `smoke_aod055`, which is the opposite of what the data supports.

**Problem 2: the representative is just the first column in list order.** It isn't chosen by meaning, completeness or importance, which is what plan step 2 asks for.

**Fix:**
- **Report co-observation:** for each correlated pair, report the share of rows where both are present, and the share where one fills the other's gaps.
- **Don't merge on partial overlap:** don't group two columns when either one is present on rows where the other is NaN in more than, say, 10% of rows.
- **Representatives:** leave them blank in the draft, or rank candidates by gain and completeness, and mark them as suggestions.

**Recommendations for the frozen list:**

| Column(s) | Recommendation | Reason |
|---|---|---|
| `aod_last_obs` | keep | Top AOD column by gain; fills AOD gaps |
| `aod055` | keep | |
| `aod047` | drop | 0.297 completeness, same as `aod055`, and genuinely redundant with it |
| `smoke_aod055` | keep | 1.1% complete, but those are smoke days, the extremes the paper reports. Trees ignore it elsewhere. Its high gain spread (sd 37k on a mean of 48k) shows it matters in folds with smoke. |
| `smoke_mask_frac` | keep | |
| `injection_height` | drop | 0.1% complete, low gain, and has negative values |
| `aod_missing` | keep | Zero gain is expected while `aod055` NaN carries the same split. Keep it for interpretation, as decided earlier. |
| `fire_count_25km`, `fire_count_50km`, `fire_frp_sum_25km` | drop | Zero gain in all folds; fire within 50 km is rare at Ontario monitors |
| Other `count` / `frp_sum` pairs at the same radius | keep one of each pair | |
| `hms_any` / `hms_n_polygons` | keep `hms_n_polygons` | It carries `hms_any` |
| MERRA-2 wind triplets (lowest level / 10 m / 50 m) | keep one per component | |
| `merra_speed` / `merra_speedmax` | keep one | |
| `merra_qlml` / `merra_tlml` | keep both | Physically distinct; the correlation is seasonal |

## F3. The leakage test doesn't bind the Phase 2 engine, and misses one check (fix with Phase 2)

**Gaps in `leakage_test.py`:**
1. **No disjointness check.** It builds `train, score` per fold but never asserts that they are disjoint (no held-out cell in `train` for group10 and LORO, no held-out year for LOYO). `training_rows` does the dropping, but nothing checks the result. Add `assert train ∩ score = ∅` on cell or year, and on site once F1 is in.
2. **The breakages only test `assert_clean`.** All three call it on a hand-built frame, so they prove the guard works but not that `scoring_rows` or `fold_rows` would reach it. Add breakages that go through the real path:
   - a held-out cell list containing a test cell → `fold_rows` must raise
   - `scoring_rows(years=[2024])` → must raise
3. **The test checks its own copy of the folds.** `fold_list` is the test's copy of the fold logic.

**Binding the engine (answers brief item 1):**
- Move the fold specs into one module (for example `folds.py: plan(scheme) -> list[FoldSpec]`). Both `cv.py` and `leakage_test.py` must import it, and neither may build folds itself.
- Give the engine a `--dry-run` that runs everything except the fit. That covers fold specs, `fold_rows`, the inner split and the quick-mode subsample. In dry-run mode it returns the (cell, date) keys of train, inner and score for each fold.
- `leakage_test.py` then tests the engine's dry-run output instead of its own reconstruction. That also covers the early-stopping set (brief item 2), which must be asserted to be a subset of train's sites and disjoint from score.

## F4. North spatial-test cell (minor; resolved by F1)

The only North test cell (856931) has 1,095 rows, one day above 25, and is the earlier period of calibration site 856930.
- **Site-level re-split:** after F1, the North test site must be a whole site.
- **Minimum size:** consider a rule that each test or calibration site has at least `min_site_rows` (for example 2,000) development-period rows.
- **Fallback:** if North can't provide a test site and a calibration site that both meet the rule, report it and ask the author. Don't loosen the rule silently.

## F5. Fold assignment details (minor)

- **Redundant sort:** `assign_folds` sorts by `n_rows`, shuffles, then sorts again, so the shuffle only breaks ties. Either drop it or make that intent explicit.
- **Fold balance:** the 1.49× row range and the 2-cell fold 8 are acceptable for grouped CV, as the brief says. Re-check them after F1, since sites are coarser units.
- **Summary formatting:** `splits_summary.md` prints regions as `np.int8(1)`; cast them to `int`.

## F6. Report wording (minor)

- **The brief's "15 correlation groups, all present in 10/10 folds"** reads as robust redundancy. Given F2, three of those groups are artefacts of pairwise-complete correlation. Correct the report once F2 is fixed.
- **Gain importance is split among correlated columns.** Say this in the report, so a low-gain member of a correlated pair isn't read as useless.

## Answers to the brief's open items

| Brief item | Answer |
|---|---|
| 1. Engine binding | F3: one shared fold-plan module plus an engine dry-run that the leakage test consumes |
| 2. Early-stopping assertion | Same mechanism. The inner set must be asserted clean, a subset of train's sites, and disjoint from score. |
| 3. Greedy fold balance | Acceptable; re-check after F1 |
| 4. Thin North test cell | F4; resolved by F1 plus a minimum site size |
| 5. `smoke_aod055` | Keep it (F2) |
| "Michigan" cell 944250 | Agree it's correct as an Ontario site, but it's part of a relocation chain with 944251 and 946380 (F1) |
| 62 columns, 208,093 rows | Agree, no issue |

## Re-run expected after the fixes

```
python code/model/splits.py --force      # site-based, F1
python code/model/leakage_test.py        # with the F3 additions
python code/model/feature_reduction.py   # with the F2 changes
```

Send `splits_summary.md`, including the nearest cross-role distance per test and calibration site, back for review before the feature list is frozen.
