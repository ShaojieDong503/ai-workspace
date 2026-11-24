# Data structure audit: the 2015 build

2026-11-05 · audit of the files on disk, not the code

**Verdict:** the layout is right. Across 36 monthly files and the training table, every column set, dtype, key, sort order and row count matches the design, and every value that appears in two tables agrees exactly.

The one real defect is in the values:
- **June 2015 context features were built without their warm-up.**
- The error carries into `prediction_2015_06` and the training table.
- It came from build order, so the full run can hit it again (V1, V2).

The scripts are in `data/interim/scratch/audit_review/`: `audit.py`, `audit_train.py`, `warmup.py`, with output in `audit.out`. Nothing else was written.

## What was checked

| Area | How |
|---|---|
| Column sets, order, dtypes | `pyarrow` schema of all 36 files, every column, against `model_inputs.py` and `build_prediction.ALLOWED_NAN` |
| Keys | Every row: strictly increasing `(date, cell_id)`, each day has exactly the 163,787 spine ids in spine order, ids within 573,402–1,017,901, keys identical across base, features and prediction |
| Base/features → prediction | All 68 data columns, every row of every month, exact equality including the NaN pattern |
| Contract | NaN count of every column of every month, against the 44 required / 19 allowed split recomputed from the code |
| NaN placement and QA rules | Every cell-day of every month (rules listed below) |
| Training table | All 57,441 rows against prediction (63 inputs + 5 QA + `province_code`), against `station_day.parquet`, and against the spine's province |
| Warm-up | June, July and August context recomputed in memory with the previous month's warm-up, then diffed against the files |

## Structural

**None in base, features or prediction.** In all 12 months:
- **Columns:** exactly 60 / 12 / 71, with no extras. One column order per stage across all months.
- **dtypes:** all to spec. `cell_id` int32, `date` timestamp[ns], floats float32, pixel and fire counts int16, `maiac_covered` and `province_code` int8. Nothing upcast to float64.
- **Rows and keys:** rows = days × 163,787, unique, strictly sorted. No pair repeats across months: every date lies inside its month and each month is complete. Every `cell_id` is in the spine.
- **Compression:** ZSTD throughout.

**Training table:**

| # | Column | Observed | Expected | Impact |
|---|---|---|---|---|
| S1 | `date` | `timestamp[ms]` | `datetime64[ns]`, like the other tables | Inherited from `station_day.parquet`, which stores ms. The values are right. A consumer that joins it to prediction without `pd.to_datetime` may get a dtype mismatch. Cast in `build_target_table`. |

Otherwise correct:
- 79 columns: prediction's 71 plus the 8 listed.
- 57,441 rows, 171 cells, 2015-01-01 to 2015-12-31.
- Keys unique and strictly sorted; one row group, ZSTD.

## Value placement

### V1. June 2015 context features have no warm-up (fix before use)

The files themselves show the cold start. On 1 June, `aod_last_obs_age_d` has a maximum of 0 and `aod_nbr_50km_d1_3` is 100% NaN; on 1 July they are 7 and 8.6%.

Recomputing June with May's last 7 days as warm-up, then diffing against `features_2015_06.parquet`:

| Column | Cell-days that differ | Days |
|---|---|---|
| `aod_nbr_50km_d1_3`, `_n` | 371,393 | 1–3 (137,960 / 127,091 / 106,342) |
| `aod_last_obs`, `_age_d` | 111,634 | 1–7 (48,202 → 360) |

The same recompute for July (with June) and August (with July) gives 0 differences, so the method is sound and only June is affected.

**Cause.** Features for June and July were built at 11:29–11:31. May's base was written at 12:23. When the previous month's base is missing, `cmd_context` (`build_features.py:165-171`) quietly runs without warm-up. The later all-months run then skipped June because the file existed.

**Reach.**
- `prediction_2015_06.parquet` carries the same values. `verify_prediction` can't see this, because it only compares the two tables to each other.
- The training table has 1,084 rows on 1–7 June, and part of them carry the wrong values.

**Fix now.**
- Delete `features/all/features_2015_06.parquet` and `prediction/2015/prediction_2015_06.parquet`.
- Rebuild both, then rebuild the training table.

### V2. The same trap in the full run

`features_2015_01` is also cold (1 Jan: max age 0, `d1_3` 100% NaN), because December 2014 doesn't exist yet. That is expected today. But when 2014 is built, resume will skip January 2015, so it stays cold for good. The same happens at every year boundary if features run before the previous year's base exists.

**Fix in code:**
- When the previous month's base is missing and the month isn't 2010-01, log a warning and mark the file, e.g. a `warm_start=false` entry in the parquet metadata.
- Have resume rebuild marked files once the previous month exists.

**Fix in procedure:** run `build_base` for all years before `build_features`, then delete and rebuild `features_2015_01`.

### V3. Six training cells have province "(water)"

2,037 training rows (3.5%) carry `province_state = "(water)"`. They match the spine, so it isn't a join error. The cell centroid falls on a river:

| cell_id | lat, lon | Place | Rows |
|---|---|---|---|
| 848529, 850659, 851721, 853851 | 45.46–45.63, −73.49 to −73.76 | Montreal (St. Lawrence, Rivière des Prairies) | 360 / 342 / 341 / 338 |
| 989118, 994439 | 39.97, −75.11 / 39.81, −75.40 | Philadelphia area (Delaware River) | 343 / 313 |

This matters for anything stratified or grouped by province, such as cross-validation folds or per-province scores. Montreal is the second-largest station group in Québec.

Decision needed: give water cells their nearest land province in the spine, or take the province from the station for training rows.

### Checked and correct

| Rule | Result, all 12 months unless stated |
|---|---|
| Required columns (44) NaN-free | 0 NaN in every month. Outside the contract, only `aod055_centroid` and `aod047_centroid` have NaN, which is allowed. |
| `aod055_npix == 0` ⇔ `aod055` NaN | 0 violations |
| `maiac_covered == 1` ⇔ `maiac_npix_total > 0` | 0 violations; values only 0/1; constant per cell through every month |
| `aod055_npix ≤ maiac_npix_total` | 0 violations |
| HMS NaN | Whole-day NaN on 2015-06-02 **and 2015-08-20** (163,787 each), nowhere else. Both zips are absent from `data/HMS_SMOKE`, and `build.log` logs both. The brief only lists 06-02. |
| HMS consistency | `hms_any` is 0/1 and equals `hms_n_polygons > 0` exactly. `hms_density_max` is finite exactly where `hms_any == 1`, with values {5, 16, 27}. |
| HMS against raw | Jan/Feb have no smoke. In January, 2 polygons touch the square but neither contains a cell centre. In December, polygons cover 4/18/11/10 cell centres on the 3rd/6th/7th/8th, and the table has exactly 4/18/11/10 smoke cell-days on those days. |
| Fire | `fire_dist_nearest_km` never NaN in 2015: every day has a detection within 500 km of the square. Counts and FRP never NaN or negative, and monotone in radius. `count_r > 0` ⇔ `dist ≤ r` exactly for every radius. 28 (Jan) and 16 (Dec) cell-days have a count but zero FRP, from detections with `frp = 0`. |
| Context | For 25/50/100 km and `d1_3`: value NaN ⇔ `_n == 0`, and `_n` is never NaN. `aod055` finite ⇒ `aod_nbr_25km_d0` finite. `aod_last_obs` NaN ⇔ age NaN. Age is an integer in 0–7. Age 0 ⇔ `aod055` finite, and then `aod_last_obs == aod055` exactly. |
| Training vs prediction | All 57,441 rows × 69 columns identical, NaN pattern included |
| Training vs stations | Same 57,441 cell-days as `station_day` 2015 (outer join, no orphans). `pm25` equals the station mean exactly and `n_stations` matches. |
| `pm25` | Range −6.82 to 80.0, 174 negatives, no −999, no NaN; `n_stations` 1 or 2; `n_hours_min` 18–24 |
| Calendar and province | `year`/`month`/`doy`/`dow` match `date`; `province_state` and `province_code` match the spine for every row |

### Brief statements the data don't support (the data are right; update the brief)

- **"aod055 NaN share 0.60–0.75."** That holds May–October (0.61–0.73). November–April run 0.81–0.99 (February 0.994), from snow and winter cloud.
- **"100% NaN south of 41°N."** 184 cells with centroids at 40.978–41.000°N are covered: they straddle the tile edge and hold the 62–2,526 non-NaN values a month. Separately, 132 cells north of 41° are uncovered, all at the north edge (59.12–59.19°N, −97.1 to −89.7°). Overall, 154,227 of 163,787 cells are covered.
- **"`aod055_centroid` NaN implies `aod055` NaN."** That direction is wrong: a cloudy centre pixel in a partly clear cell gives centroid NaN with `aod055` finite, 16k–504k times a month. The intended rule holds: centroid finite ⇒ `aod055` finite, 0 violations (checked January and July).

## Cosmetic

- **Sizes** track AOD coverage, not errors. Features run 19 MB (Feb, 99.4% NaN) to 105 MB (Jul). Prediction runs 89–207 MB, base 82–113 MB. Nothing is out of line once coverage is accounted for.
- **Row groups.** Base and features use pyarrow's default of 1,048,576 rows. Prediction uses 1,000,000, so 31-day months end with a 77,397-row group. Harmless.
- **Column order** differs from the brief's listing: `hms_n_polygons` comes before `hms_any`, and context lists the means before their `_n`. It is identical across every month and stage.
- **Context counts** (`*_n`) are float32. That's within spec, but int16 would be exact and smaller.
- **`province.cell_to_code()` rewrites `province_lookup.csv` on every call**, through `build_lookup`. It's deterministic today, but a read path shouldn't write.

## Fix order

- [ ] V1: delete and rebuild `features_2015_06` and `prediction_2015_06`, then rebuild the training table
- [ ] V2: add the warm-start guard in `cmd_context`. For the full run, build all base before any features, then rebuild `features_2015_01`.
- [ ] V3: decide how water cells get a province
- [ ] S1: cast the training `date` to ns
- [ ] Update the brief's expectations (AOD NaN share by season, 41°N edge cells, centroid rule direction, HMS 2015-08-20)
