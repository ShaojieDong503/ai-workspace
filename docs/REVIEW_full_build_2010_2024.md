# Full build 2010–2024: run and QA report

2026-11-06 · build of all 15 years with QA after each year

**Verdict:** the full panel is built and passes QA.
- Every build step exited 0. Every year's own audit found no defect.
- The cross-year verifiers pass on all 180 months.
- The training table matches both the station data and the prediction table exactly.
- Four verifier steps returned FAIL. All four are false alarms from two weaknesses in `verify_maiac.py` (§4); the data behind them is correct.

## What exists now

| Output | Path | Files | Size |
|---|---|---|---|
| Base panel | `data/interim/base/all/base_all_YYYY_MM.parquet` | 180 | 18 GB |
| Context features | `data/interim/features/all/features_YYYY_MM.parquet` | 180 | 13 GB |
| Prediction table | `data/processed/prediction/YYYY/prediction_YYYY_MM.parquet` | 180 | 28 GB |
| Training table | `data/processed/target/target_table.parquet` | 1 | 157 MB |

D: has 125 GB free afterwards.

Run logs: `data/interim/scratch/full_run/<year>/` has one log per step; `progress.txt` has every step's exit code and duration; cross-year steps are under `final/`. My audit scripts are in `data/interim/scratch/audit_review/`.

## How it ran

The years ran in order 2010 → 2024, so every January was built after its December existed (`run_all.sh`). For each year:
1. **Build:** `build_base --years Y Y`, then a features pass and a prediction pass over all months. The latter two only rebuild months whose inputs changed.
2. **QA:**
   - `verify_merra`, `verify_maiac`, `verify_hms` and `verify_fire` on January and July;
   - `verify_features` on January (warm-up across the year boundary) and July;
   - my 12-month structural audit (`audit.py <year>`), reduced to invariant breaks by `flags.py`.

Then the cross-year steps (`run_final.sh`):
- a last features pass and a last prediction pass (both rebuilt nothing);
- `build_target_table`;
- `verify_base` and `verify_prediction` over all 180 months;
- an audit of the full training table (`audit_train_all.py`).

2015 already existed. Its only change was the planned one: once December 2014 existed, `features_2015_01` was rebuilt with warm-up (`warm_start=true`), and prediction 2015-01 followed automatically (logged "features changed").

Run time was 18.5 hours in all. 2010, 2011 and 2014 took about 30 minutes each; the other years took 75–100 minutes because the machine was slower for long stretches. Every step was slower then, not just the heavy-fire years.

## 1. Per-year results

"Fire rounding" means a cell-day where the nearest-fire distance is stored as exactly 25, 100, 250 or 500 km while the count at that radius is 0 (§3).

| Year | Build | Verifiers | Audit flags | January warm-up |
|---|---|---|---|---|
| 2010 | ✓ | ✓ | HMS density unknown Feb–May (raw "NA"); 1 fire rounding (Oct) | n/a, the period start |
| 2011 | ✓ | ✓ | 1 fire rounding | ✓ equals Dec+Jan continuous |
| 2012 | ✓ | ✓ | HMS missing 2012-10-07 | ✓ |
| 2013 | ✓ | MAIAC Jul FAIL (§4a) | 2 fire rounding | ✓ |
| 2014 | ✓ | ✓ | 1 fire rounding | ✓ |
| 2015 | ✓ (Jan rebuilt warm) | ✓ | HMS missing 2015-06-02, 2015-08-20 | ✓ now warm (was cold) |
| 2016 | ✓ | ✓ | HMS missing 2016-03-06, 2016-11-12 | ✓ |
| 2017 | ✓ | ✓ | none | ✓ |
| 2018 | ✓ | ✓ | 2 fire rounding | ✓ |
| 2019 | ✓ | MAIAC Jul FAIL (§4a) | 3 fire rounding | ✓ |
| 2020 | ✓ | ✓ | 3 fire rounding | ✓ |
| 2021 | ✓ | MAIAC Jul FAIL (§4a) | 1 fire rounding | ✓ |
| 2022 | ✓ | ✓ | 2 fire rounding | ✓ |
| 2023 | ✓ | ✓ | 1 fire rounding | ✓ |
| 2024 | ✓ | MAIAC Jan FAIL (§4b) | 1 fire rounding | ✓ |

**Clean in every year:**
- **Structure:** exact column sets, every dtype to spec, keys unique and sorted, every day holds all 163,787 spine cells (leap years included), ZSTD throughout.
- **Prediction against its sources:** equals base and features exactly, in every column of every month.
- **The no-NaN contract:** none of the 44 required columns has a NaN in any month.
- **The QA-column rules:**
  - `aod055_npix == 0` exactly where `aod055` is NaN;
  - `maiac_covered == 1` exactly where `maiac_npix_total > 0`, constant per cell;
  - `aod055_npix ≤ maiac_npix_total`.
- **HMS:** `hms_any` and `hms_n_polygons` agree. `hms_density_max` is finite only where there's smoke.
- **Fire:** counts and FRP sums are never NaN or negative, and both grow with the radius.
- **Context features:** each mean is NaN exactly where its count is 0; age is an integer from 0 to 7; age 0 exactly where `aod055` is observed.
- **Warm-up:** every month after 2010-01 has it.

## 2. Cross-year results

| Check | Result |
|---|---|
| `verify_base`, 180 months | 0 failures. It flags 3 months where the share of missing `hms_density_max` jumps more than 3× (Jul 2020, Jul 2022, Aug 2022). See below. |
| `verify_prediction`, 180 months | 0 failures. Every month's column set, row count, keys and contract hold. Every training row has a prediction row (0 unmatched in all 180 months), and all 63 inputs are identical between the two tables. |
| Training table, structure | 1,053,437 rows, 351 cells, 2010-01-01 to 2024-12-31; 79 columns; `date` at nanosecond precision; unique and sorted |
| Training table vs `station_day` | Every one of the 1,053,437 station cell-days is present, with no extras. `pm25` equals the station mean exactly, and `n_stations` matches. |
| Training table vs prediction | All 1,053,437 rows × 69 columns identical, NaN pattern included |
| Training table, values | `pm25` from −8.67 to 581.3 µg/m³, with 2,602 negatives (0.25%), no −999 and no NaN. `n_stations` is 1–4 and `n_hours_min` is 18–24. Calendar columns agree with `date`. Province matches `province.cell_to_code` for every row, and no row is "(water)". |

Rows per year run from 40,306 (2010) to 101,446 (2024), growing as networks add stations: 2010 40,306 · 2011 47,405 · 2012 47,551 · 2013 50,636 · 2014 53,923 · 2015 57,441 · 2016 63,616 · 2017 71,928 · 2018 75,138 · 2019 82,569 · 2020 85,165 · 2021 89,780 · 2022 91,968 · 2023 94,565 · 2024 101,446.

**The `verify_base` flags are seasonal smoke, not defects.** `hms_density_max` is NaN wherever there's no smoke, so its missing share tracks how much smoke there was. July 2020 fell to a 2.7% smoke share, against 11.8% in June. I recounted July 2020 from the raw HMS polygons, testing each cell's centre directly: 135,292 smoke cell-days, against 135,219 in the table. That's a 0.05% gap, all on polygon edges.

## 3. Known data properties (not defects)

- **Missing HMS days:** 2012-10-07, 2015-06-02, 2015-08-20, 2016-03-06, 2016-11-12. Each zip is absent from `data/HMS_SMOKE`, and `build_base` logs it. The whole day is NaN, as designed.
- **HMS density before June 2010:** the raw files record density as "NA" for every polygon until mid-2010. So `hms_density_max` is NaN on smoke cell-days from February to May 2010 (allowed as "unknown density").
- **31 December 2024:** `modis_2025_*.csv` doesn't exist, so the fire detections from 00:00–06:00 UTC on 1 January 2025 are missing. `build_base` logs the warning. That's one day.
- **2010-01:** no warm-up, because it's the first month. Flagged `warm_start=false`, never rebuilt.
- **Fire rounding:** 1–3 cell-days in some months (§1). The true distance is just over the radius, so the count at that radius is correctly 0, but storing the distance at single precision rounds it to exactly the radius. Harmless.
- **The MAIAC all-zero empty-day rule never fired** in any of the 180 months: no `build_base` log has an "empty tile-days" line. The all-NaN days that `verify_maiac` lists (e.g. 2013: 11) are a different, normal case: whole days with no valid MAIAC at all.

## 4. Verifier fixes needed (the data is correct)

**a. `injection_height` tolerance, failing in 2013, 2019 and 2021, each in July.**
- The FAILs were differences of 1.8e-05, 2.5e-05 and 7.7e-05.
- `injection_height` is in metres (median about 200, maximum about 6,550) and stored at single precision. At several hundred metres or more, single precision can't resolve differences that small, so these are rounding.
- `verify_maiac.py:165` uses `tol=1e-5` with no relative term. Re-run with `rtol=1e-6`, all three pass with 0 failures.
- **Fix:** pass `rtol=1e-6` for the float columns.

**b. "Median distinct `aod055` per cell > 1", failing in January 2024.**
- January 2024 is 98.6% NaN. Of the 36,380 observed cells, 18,700 were observed on only one day, so the median lands on 1.
- Of the 17,680 cells observed on 2 or more days, only 7 repeat a single value, and their median is 2 distinct values. No value is stuck.
- **Fix:** take the median over cells with at least 2 observed days.

## 5. Open items

- Apply the two verifier fixes in §4, and re-run `verify_maiac` for those four months to show they pass.
- Optional: get January 2025 MODIS fire data to complete 31 December 2024.
- Optional: the training-only `station_province` column from round 3, if anything is reported per province by the station's jurisdiction.
