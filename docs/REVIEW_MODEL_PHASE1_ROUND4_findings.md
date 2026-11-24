# Phase 1 review (round 4)

Review of `code/model` after the round-3 fixes (files dated 19 Nov, 11:35–11:40).

**What I read:**
- `splits.py`, `data.py`, `leakage_test.py` and `feature_reduction.py`
- `splits_summary.md`, `leakage_test.log` and `feature_reduction_report.md`

`config.json`, `folds.py` and `features.py` are unchanged.

**Verdict: Phase 1 is approved.** Nothing blocks, and the two notes below are optional. Next come your three decisions, at the end.

## Round-3 items: all fixed

| Item | Status |
|---|---|
| R3-1, summary written per row | Fixed. `splits_summary.md` is 2 KB, with one line per picked cell, and the distance column is blank for development. The picked cells, role counts and fold sizes are the same as in round 3, so the splits didn't change. |
| R3-2, ranking ignores gain | Fixed. `rank_group` puts the wider member of a declared family first (50 km before 25 km, for both the means and the counts). Every other group is ranked by gain, then completeness. |
| Declared nesting not checked | Fixed. `check_nesting` runs on the development rows before any grouping. |
| Report wording | Fixed. The report now says declared pairs, and says the declarations are checked. |
| 2024 breakage passed for the wrong reason | Fixed. `scoring_rows` now rejects years outside 2010–2023 explicitly, and the log shows `scoring years outside 2010-2023: [2024]`. |
| Leakage test not logged | Fixed. `leakage_test.log` shows **PASS**: 149,846 rows scored once in each scheme, 0 shared cells or years, and all 3 breakages caught. |

## Optional

- **Weak check on the count columns.** `check_nesting` tests only "narrow present ⇒ wide present". For the count family (`aod_nbr_25km_d0_n` / `aod_nbr_50km_d0_n`), both columns are always present, so the check passes trivially. The real check is `n_25km <= n_50km` on every row. Add it for families whose columns never have missing values.
- **Hidden ordering convention.** `rank_group` and `check_nesting` assume each family is listed narrow → wide (`family[0]` narrow, `family[-1]` wide). Say so in `config.json` next to `nested_families`, so a future entry in the wrong order doesn't silently invert the check.

## Your decisions before Phase 2

1. **Calibration in North.** The random draw has no North calibration cell and two East cells. So the 90% intervals won't be checked for North. Keep it, or switch calibration to one random cell per region? If you switch, re-run `splits.py --force` with the same seeds. The test cells won't change, because they are drawn first.
2. **Freeze the feature list.** The report's suggested order is now usable. My recommendations from round 1 still apply:

   | Columns | Recommendation |
   |---|---|
   | `aod_last_obs`, `smoke_aod055`, `aod_missing` | Keep |
   | `aod047` | Drop |
   | `injection_height` | Drop |
   | `aod_nbr_25km_d0`, `aod_nbr_25km_d0_n` | Drop |
   | Fire `count` / `frp_sum` at the same radius | Keep one of each pair |
   | `fire_count_25km`, `fire_count_50km` | Drop (zero gain) |
   | `hms_any` / `hms_n_polygons` | Keep `hms_n_polygons` |
   | Wind triplets | Keep one column per component |
   | `merra_qlml` and `merra_tlml` | Keep both |

3. **Set `contingency.criterion`** in `config.json` before R1 runs.
