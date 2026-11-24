# Data quality and raw-to-processed reconciliation

Audit date 2026-11-08. Evidence (logs, CSVs) is in `audit/evidence/`; the code that produced it is in `audit/scripts/` (`aNN_*.py`, named in each section). Findings are referenced as F-NN and detailed in `04_findings.md`.

## Summary

- **Structure is sound.** All 897,388,973 prediction rows have:
  - exact (day × 163,787) blocks;
  - unique, sorted keys;
  - no NaN in any of the 44 required inputs.

  The training table has 1,050,847 unique cell-days, and every one comes from a station-day.
- **Targets reconcile with raw data.**
  - NAPS: all 411,346 station-days match a fresh build from the raw CSVs. The Ontario values differ only by 2-decimal rounding.
  - AQS: all 667,759 station-days match a fresh build from the raw zips, except 1,053 on 31 December (F-01).
- **Features reconcile with raw data.** Independent recomputation matched every sampled value:
  - MERRA-2, fire, HMS: 200 cell-days in 5 months the existing verifiers skip;
  - MAIAC: 75 cell-days;
  - context features: 960 cell-days across month boundaries.

  MERRA-2 is UTC-6 in every year.
- **Defects and risks found.**
  - One confirmed target defect (F-01).
  - Two deviations from the supervisor's hourly-validity rule (F-02, F-03).
  - Two features whose zeros don't mean what a model will assume (F-04, F-05).
  - One provenance gap (F-06).
  - Several Low documentation issues.

No Critical or High finding was confirmed. Passing these tests doesn't prove the data is error-free: §5 lists what was not covered.

## 1. Profiling the prediction table, every row (`a01`, `a11`, `a16`)

Every row was scanned with DuckDB, with no sampling: NaN, zero, negative, min and max per column and month.

| Check | Result |
|---|---|
| Months / rows | 180 / 897,388,973; each month exactly days × 163,787 |
| Keys | Unique; sorted by (`date`, `cell_id`); every day holds exactly the spine's cell set in spine order; no missing days (`a16`, Python, all 180 months) |
| Required inputs (44) | 0 NaN in every month |
| MERRA-2 (34) | 0 NaN. Ranges are physical: TLML 227–308 K, PBLH 50–3,501 m, TOTEXTTAU 0.015–8.4. **But `merra_u2m` / `merra_v2m` are exactly 0 in 3.26 M cell-days (F-04)** |
| Fire (11) | No NaN, no negatives. Max `fire_count_500km` is 4,841 (June 2023), far below the int16 limit. `fire_dist_nearest_km` is never NaN, because some detection exists somewhere on every day |
| HMS | NaN only in the 5 documented missing-day months (0.09% of rows). Density is 5/16/27 only |
| MAIAC AOD | `aod055` NaN in 79.7% of rows (monthly 48–99%, worst in February). Range 0–5.88. No negatives. `aod055_npix == 0` exactly where `aod055` is NaN (715,526,161 both) |
| `injection_height` | 99.8% NaN; **1,924 negative values, minimum −647 m (F-07)** |
| `smoke_mask_frac` | NaN in 52,419,664 rows: 52,379,240 are south of 41° N, plus **40,424 in covered cells on 2011-06-10, 2011-06-20 and 2012-02-01 (F-08)** |
| Context features | Each mean is NaN exactly where its count is 0; ages 0–7 |

Seasonal missingness is as expected: winter AOD is mostly missing (snow and cloud), and `aod_last_obs` is missing up to 97% of the time in February. Missingness isn't flagged as an error where it follows the documented physical cause.

## 2. Profiling the training table (`a04`)

| Check | Result |
|---|---|
| Rows / cells / keys | 1,050,847 / 351 / 0 duplicates, sorted |
| `pm25` | 0–581.3 µg/m³, median 6.5, 99.9th percentile 51.9; 248 exact zeros; no negatives |
| Reconciliation | Cell-days formed from `station_day` with `pm25 ≥ 0` = 1,050,847, an exact match; 2,668 negative station-days removed (AQS 2,632, NAPS 36) |
| `n_stations` | 1: 1,029,830 · 2: 18,293 · 3: 875 · 4: 1,849 |
| Calendar | 0 mismatches with `date` |
| Required columns | No NaN in targets, keys, province, region or distance |
| AOD available | 17–27% of training rows per year; 282,530 rows south of 41° N never have MAIAC |
| Composition over time | AQS monitors with data grow from 54 (2010) to 239 (2024), while NAPS stays at 70–82. AQS daily means average higher (F-12) |

**Extreme values.** These are kept, and they are not errors in themselves:
- the top station-days are the June 2023 Quebec/Ontario fires (Senneterre 581, North Bay 405) and the 7 June 2023 smoke in Pennsylvania;
- one outlier lacks an obvious smoke cause: **Thompson, MB, 2017-03-08, 295 µg/m³** (F-11).

**Multi-monitor cells.** 21,017 cell-days average two or more monitors; 38 have a spread between monitors above 20 µg/m³. The largest are cell 962346 (Windsor plus four Detroit monitors), with spreads up to 46 µg/m³.

## 3. Raw-to-processed reconciliation

### 3.1 NAPS targets (`a03`, `a05`): 411,346 station-days, all compared

The reference rebuilt everything directly from the 15 raw CSVs:
- the header is found by its content;
- −999 is treated as missing;
- duplicate method rows are resolved with the stated rule;
- each hour is converted to UTC-6 by its start time;
- hours are pooled across years.

The station universe is every NAPS station whose coordinates fall in the spine, whatever the province: 111 (ON 59, QC 49, MB 3). No NAPS station outside ON/QC/MB is in the domain.

| Result | |
|---|---|
| Rows | 411,346 on both sides; 0 on one side only |
| Values | QC/MB: max difference 2e-5 (float32). ON: max 0.005, from the 2-decimal rounding of the reused Ontario product (F-10) |
| `n_valid_hours` | 0 mismatches |
| Station → cell | 111/111 identical. No station changed cell across its coordinate history (37 stations have more than one coordinate pair) |

The six Quebec City-area stations on the domain list (Québec ×5, Lévis) are correctly outside the domain. The domain is a Lambert rectangle whose east edge is at 71.33° W at that latitude (`a20`, F-09).

### 3.2 AQS targets (`a02`, `a09`, `a22`): 667,759 station-days, all compared

The reference streamed the 15 raw zips and processed them as follows:
- the cell is computed from each row's own coordinates;
- the UTC-6 day comes from hour-start GMT;
- **hours are pooled across annual files**;
- valid hours are counted as distinct hours with a value;
- days need 18 or more valid hours per POC, and the site value is the mean over passing POCs.

| Result | |
|---|---|
| Sites | 277 on both sides; identical cells; no site spans two cells |
| UTC offsets | Every site has one fixed local–GMT offset (177 at −5, 100 at −6), so there's no daylight-saving contamination |
| Duplicate hourly rows | 0 |
| Rows | 667,759 on both sides; 0 on one side only |
| Values | **1,053 mismatches, all on 31 December, all at UTC−5 sites (F-01)**. Every other row matches to 1e-4 |
| `n_poc` | 0 mismatches |
| Multi-POC agreement | 183,903 site-days with 2+ POCs; 141 differ by more than 10 µg/m³, 1 by more than 25 |

### 3.3 Hourly validity (`a05`, `a09`, `a10`)

The pipeline keeps every non-missing hourly value, including negatives and qualifier-flagged hours. It drops only station-days whose **daily mean** is negative.

| | NAPS | AQS |
|---|---:|---:|
| Station-days containing at least 1 negative hour | 941 | 90,176 (13.5%) |
| … of which the daily mean is still ≥ 0 (kept) | 905 | 87,544 |
| Days lost if negative hours were missing (fewer than 18 left) | 107 | 11,130 |
| Days whose mean rises by more than 0.5 µg/m³ under that rule | 296 | 34,462 |
| Mean shift under that rule | +0.001 | +0.071 µg/m³ |
| Station-days containing qualifier-flagged hours | n/a | 36,466 |

The AQS qualifiers on kept hours include:
- fire and exceptional-event codes (IF, IT, RF, RT): real smoke values that must be kept;
- QC-concern codes, among them QX (does not meet QC criteria; 14,165 hours in 2023 alone), SX (siting), 5 (outlier) and NS (nearby source).

See F-02 and F-03.

### 3.4 Base features against the raw sources (`a08`, `a18`, `a21`)

| Source | Method (different from the extractor) | Sample | Result |
|---|---|---|---|
| MERRA-2 (8 variables, 3 collections) | rasterio `index(lon, lat)` | 200 cell-days in Apr 2012, Nov 2016, Sep 2018, Jun 2023, Dec 2024 | 1,600/1,600 match |
| Fire (all 11 columns) | Brute-force haversine over that UTC-6 day's detections | Same 200 cell-days | 2,200/2,200 match |
| HMS count and density | shapely point-in-polygon on the lon/lat centre, timestamps by `strptime` | Same 200 (2 on a missing day: NaN on both sides) | 398/398 match |
| MAIAC (6 columns) | Pixel centres tested against the cell's EPSG:3347 box | 75 cell-days in Apr 2014, Sep 2018, Jun 2023 | 450/450 match |
| MERRA-2 day window | Fit of UTC-6(D) on UTC(D−1, D, D+1) from the older UTC export | 15 years × Apr/Oct × U10M, V10M, TLML | D+1 weight 0.21–0.25 and D−1 near 0 in every case: UTC-6 confirmed |

**MAIAC raw encoding** (`a07`, `a14`, `a15`):
- AOD missing is NaN, not 0. Exact-zero AOD pixels are scattered and rare (≤0.05%), with no all-zero blocks, so fill zeros don't leak into the means.
- `SMOKE_MASK` is 0 wherever there's no smoke flag, **including under cloud**, and NaN only in orbit gaps (F-05, F-08).

### 3.5 Derived features and joins (`a17`, `a18`, `a06`)

- **Context features.** All 10 columns were recomputed by brute force with lattice-disk loops on days 1, 2, 3 and 15 of three months (Jan 2021, Mar 2013, Sep 2018), so the history crosses the month boundary. 9,600/9,600 values match. Only day D and earlier are used, so no future information enters a feature.
- **Training ↔ prediction.** In three months, every training row joins, and all 69 shared columns are identical, NaN pattern included.
- **Target leakage.** No feature is computed from PM2.5. `n_stations`, `n_hours_min`, `dist_ontario_km` and `on_region` are carried in the table but excluded from `MODEL_INPUTS`.
- **Regions.** Recomputed from the thesis file for all 59 ON monitors:
  - all agree except Windsor 060211, which is deliberately excluded because its cell holds four Detroit monitors;
  - no non-Ontario monitor is in an Ontario region;
  - `dist_ontario_km` matches brute force except in the three documented border cells, where `province_state` and `on_region` disagree by design (F-13).

## 4. Temporal and spatial coverage

- **Days.** 5,479 days, none missing, in both tables. The training table has 2,871–8,741 rows per month, growing with network expansion.
- **Space.** 351 training cells:
  - 48 in Ontario regions, 303 outside;
  - 282,530 training rows are south of 41° N, where MAIAC is absent.

  The North region has 7 monitors in 6 cells.
- **Known source gaps** (F-14): 5 HMS days missing; HMS density not recorded February–May 2010; fire on 2024-12-31 lacks the 1 January 2025 file. The 2024-12-31 AQS values at UTC−5 sites also have no 2025 hour.

## 5. What these tests do not establish

- Only samples were recomputed for the features (about 5,000 values across 8 months). Rare defects confined to other months or cells would not be seen.
- The MAIAC values were checked against the exported stack, not against MCD19A2 itself (F-06).
- The scientific suitability of keeping negative and flagged hours is a decision for the supervisor, not a data error.
- The README's statement that the training table matches the prediction table "on all rows" comes from the pipeline's own verifier. I re-checked three months.
