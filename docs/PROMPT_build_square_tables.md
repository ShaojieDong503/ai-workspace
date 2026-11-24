# Task: build the training table and the gap-free prediction table for the square domain, 2010-2024

You are a coding agent on a local Windows 11 machine. Build the data for a LightGBM daily PM2.5
model (a thesis project, so provenance and correctness matter as much as speed). The model will
later produce a **gap-free daily PM2.5 product**: a prediction for every cell on every day.

1. **Training (target) table**: one row per *monitoring station per day* that has a valid daily
   PM2.5, with the features of the 5 km cell containing the station. About 1 M rows.
2. **Prediction table** (no target): one row per *5 km cell per day* for every cell in the domain,
   with the same features. About 897 M rows. Every **model-input** column must be gap-free, except
   the ones section 7 explicitly allows to be NaN.

Both tables come from **one feature code path** in two layers:

```
 base panel (cell x day, raw values, no filling)     MAIAC, MERRA-2 (UTC-6), HMS, fire     [sections 3-5]
        |
 feature stage (gap-filled AOD + context)           built from the base panel only        [section 6]
        |
        +-> TRAINING table   = station-days joined to (base + features) at the station's cell
        +-> PREDICTION table = (base + features) for every cell x day
```

The training table is the station-days **joined** to the feature rows of their cells. A station's
features are therefore byte-identical to its cell's row in the prediction table. Never compute
training features a second way.

**There are no static features.** No land cover, roads, population, elevation or CanOSSEM covariates,
and the CanOSSEM crosswalk is not needed. Features are only the daily sources below and what the
feature stage derives from them.

Project root: `D:\ai-workspace\project`. Work in stages. Stop where section 10 says to stop.

---

## 1. Machine and environment (measured; re-measure before heavy stages)

- 12 logical cores, **16 GB RAM**, `D:` has about **219 GB free**.
- `C:\Python314\python.exe` already has rasterio 1.5.0, pandas 3.0.3, pyarrow 24.0.0, numpy 2.4.6,
  scipy 1.17.1, shapely 2.1.2, pyproj 3.7.2, scikit-learn 1.9.0. Missing: geopandas, duckdb, lightgbm,
  and any shapefile reader.
  Create `D:\ai-workspace\project\.venv` with `--system-site-packages`, install only what's missing
  (e.g. `pyogrio` or `pyshp` for HMS, `lightgbm` for section 6, `psutil`), and pin versions in
  `code/panel/requirements.txt`.
- The shell is PowerShell 5.1. Set `$env:PYTHONIOENCODING='utf-8'` before running Python that prints
  non-ASCII.
- No data downloads. Everything is local.

## 2. Rules

1. Raw inputs are **read-only**: `data/raw/`, `data/Target_Data_2010_2024/`, `data/HMS_SMOKE/`,
   `data/MODIS_fire/`, and every `data/*.zip`. Read zips in place via GDAL `/vsizip/` or `zipfile`.
   Never extract them to disk. Write only to `data/interim/`, `data/processed/`, `code/panel/`, `docs/`.
2. **One day definition: the UTC-6 day** (fixed CST, no DST). Day D = [D 06:00 UTC, D+1 06:00 UTC).
3. **Base panel:** no imputation, no zero-filling, no lags, no spatial aggregates. Missing stays NaN.
   Fire radius features (4.4) are the exception, because they are the base definition of fire. All
   filling and context happens in the feature stage (section 6), which keeps the raw columns beside
   the filled ones.
4. **Look up raster bands by description (`YYYYMMDD_<VAR>`), never by position.** Index pixels by
   geotransform, never by assuming a raster shape.
5. **Keys:** `cell_id` int32 (= `row*1065 + col`, the lattice id), `date` date32. Note: older prompts
   used string ids. Use int32 here; at 900 M rows it matters.
6. Values are float32. Counts and flags are int8 or int16. Parquet with zstd, sorted by `(date, cell_id)`.
7. Never hold more than about one month of the prediction table in memory.
8. Style: match `code/data_preparation/*.py`. That means a module docstring stating the rule and why,
   `argparse`, and a separate verify script. Put a `README.md` and `qa_report.md` beside each output folder.
9. Never claim a check passed unless you ran it and saw the output.

## 3. Spine

- Cells: `data/interim/grid/cell_table_square.parquet`. Read its `README.md` first. Keep
  **`in_square == True`**: 163,787 cells, including 16,282 water cells (`country` null). Keep water
  cells; filtering is cheap later.
  The lattice is EPSG:3347 with cell centres at `x = 3688478.20 + 4999.827*col`, `y = 5239381.11 - 4998.864*row`.
  Each cell's footprint is its centre ± half a pixel. It's exact; no polygons are needed.
- Write `data/interim/spine/cells.parquet`: `cell_idx` (0..n-1 in cell_id order), `cell_id`, `row`, `col`,
  `x_3347`, `y_3347`, `lat`, `lon`, `country`, `province_state`. Coordinates live **only here**, not in the
  big tables.
- Days: 2010-01-01 .. 2024-12-31, 5,479 days.
- Inside a month, every source returns an array of shape `(n_days_in_month, n_cells)` in `cell_idx`
  order. Rows are then `date`-major, which gives the `(date, cell_id)` sort for free. Joining sources is
  positional concatenation. Assert that the key arrays match before writing.

## 4. Base-panel sources (each one a module with `extract(year, month, cell_idx) -> {column: array}`)

Each extractor takes a **subset of cells** (`cell_idx`), so the same code serves the station-cell run and
the full run. Precompute static index maps once into `data/interim/index/` (`.npy`).

### 4.1 MAIAC AOD: `data/raw/maiac_daily_stack/` (about 190 GB, the bottleneck)
- Files: `source_region_maiac_daily_stack_YYYY_MM-RRRRRRRRRR-CCCCCCCCCC.tif`, 8 tiles per month,
  EPSG:4326, 0.008983152841195215°. Bands per day: `YYYYMMDD_AOD055, _AOD047, _INJECTION_HEIGHT,
  _SMOKE_AOD055, _SMOKE_MASK`. Product: MCD19A2.061, QA bits 8-11 in {0, 1, 11}, daily mean over the
  UTC day. Terra/Aqua overpasses fall inside the same UTC-6 day, so UTC day = UTC-6 day. Say this in the README.
- **Coverage is 41.0°N - 69.5°N only.** 9,612 square cells lie south of 41°N (PA, OH, NJ, IN, IL, MD ...).
  They get NaN plus `maiac_covered = 0`. Discover tiles from file bounds, never from a hardcoded
  list, so a later re-export of the 37.5-41°N strip drops in without code changes.
- **Footprint mean, computed with an index map:** take every MAIAC pixel centre in the square's window,
  project it to EPSG:3347, and assign it to the lattice cell `(round((5239381.11-y)/4998.864),
  round((x-3688478.20)/4999.827))`. That gives an int32 pixel→`cell_idx` map (-1 = outside the spine).
  Per day and band: `np.bincount(idx[valid], v[valid])` / `np.bincount(idx[valid])`. No per-cell loop
  and no polygon tests.
- Columns: `aod055`, `aod047`, `injection_height`, `smoke_aod055` (footprint means), `smoke_mask_frac`
  (mean of the mask over valid pixels), `aod055_npix` (int16, valid pixels), `maiac_npix_total` (int16),
  `aod055_centroid` (pixel containing the cell centre; kept for comparison), `maiac_covered` (int8).
- **Empty-day trap:** the export wrote days with no granule as constant 0 in all five bands. If all five
  bands of a tile-day are exactly 0, set that tile-day to NaN. Count these per year.
- **I/O first:** check `block_shapes`, interleave and compression of one file. Benchmark reading one
  tile-month in a few patterns (full window by row chunks, or per-day band groups), then pick the fastest
  pattern that fits in about 3 GB per worker. Report MB/s and projected total hours before the full run.

### 4.2 MERRA-2, UTC-6 daily means: `data/raw/merra2_utc6_ee/`
- 540 monthly GeoTIFFs: `source_region_merra_{aer,flx,slv}_daily_stack_YYYY_MM.tif`, 2010-01 .. 2024-12.
  Earth Engine exports of the **mean of the hourly MERRA-2 images over the UTC-6 day**.
- Already measured:
  - every month has every day, and there are no NaN or -999 values inside the study square;
  - grid: EPSG:4326, 0.4940734062657368°, **69 x 44 px**, upper-left (-99.3087546594131, 59.78288215815415),
    covering 99.31°W-65.22°W, 38.04°N-59.78°N;
  - the only NaNs are in the last column (about 65.5°W, outside the square);
  - the files carry no metadata tags.
- **Confirm the day window before relying on it.** The files themselves don't record the window. Look
  for the export notebook (`code/colab/`, or ask the user). Also, as evidence, compare against the
  UTC-day files in `data/SourceRegion_MERRA_2010_2024-20261103T023716Z-1-001.zip` (same lattice,
  offset by exactly 2 columns and 2 rows). For a UTC-6 window, a day should sit closer to
  0.75·UTC(D) + 0.25·UTC(D+1) than to UTC(D). A spot check on 2015-07-15 showed this for PBLH, TLML
  and ULML. Report what you find. **Stop and ask** if it contradicts UTC-6.
- Variables (34): `BCCMASS OCCMASS SO4CMASS SSCMASS25 DUCMASS25`, `BCSMASS OCSMASS SO4SMASS SSSMASS25
  DUSMASS25`, `BCEXTTAU OCEXTTAU SUEXTTAU DUEXTTAU TOTEXTTAU`, `PBLH QLML TLML ULML VLML SPEED SPEEDMAX`,
  `U2M V2M U10M V10M U50M V50M U250 V250 U500 V500 U850 V850`. Columns are `merra_<var>` in lower case.
  Daily means only; there are no max/min statistics.
- Spatial rule: the pixel containing the cell centre. Precompute the index once.
- Units are not in the files. Take them from the MERRA-2 file specification (GMAO, Bosilovich et al.)
  and cite it in the data dictionary.
- The other MERRA files are superseded for this build. Don't use them:
  `SourceRegion_MERRA_*` zips (UTC day), `Canada_Source_*`, `US_Source_*`, `*_Ontario`.

### 4.3 HMS smoke: `data/HMS_SMOKE/hms_smokeYYYYMMDD.zip` (5,474 files; 5 days missing)
- Fields `Satellite, Start, End, Density`. `Start`/`End` are UTC, `YYYYDDD HHMM`. Assign each polygon to
  every UTC-6 day its [Start, End] interval overlaps, **using the timestamps, not the file name**.
  Process the files in date order and carry polygons over between days.
- Rasterize onto the lattice in EPSG:3347 (`rasterio.features.rasterize`, `all_touched=False`; that is
  exactly the centre-in-polygon rule):
  - `hms_n_polygons` (int8, rasterize with merge ADD)
  - `hms_any` (int8)
  - `hms_density_max` (Light 5, Medium 16, Heavy 27 µg/m³; NaN if smoke is present but density is unknown)
- A missing file means a missing day: NaN, **not 0**. Note that a missing file only removes the polygons
  it would have held. Document how you handle a UTC-6 day whose polygons come from two files when one of
  them is missing. List the missing dates in the QA report.

### 4.4 MODIS active fire: `data/MODIS_fire/modis_YYYY_{Canada,United_States}.csv` (30 files)
- `acq_date` + `acq_time` (HHMM) are UTC; convert to the UTC-6 day. Keep `type == 0` and report the
  counts dropped per type. Drop exact duplicates across the two country files.
- Columns: `fire_dist_nearest_km` (great-circle; NaN when there are no detections that day),
  `fire_count_{25,50,100,250,500}km` (int16; 0 when there are none), `fire_frp_sum_{...}km`.
- Fires up to 500 km **outside** the square count too.
- Efficient approach to evaluate:
  - nearest distance: `sklearn.neighbors.BallTree(haversine)` built on that day's fires, queried for all
    cells (exact, fast);
  - radius counts/sums: snap fires onto a lattice padded by 500 km and FFT-convolve with disk kernels
    (`scipy.signal.fftconvolve`).
  The definition is **great-circle**. Use the FFT path only if a verification on at least 30 high-fire
  days shows at least 99% of cell-days with identical counts and FRP within 1%. Otherwise use the exact
  path (BallTree on cells, accumulate per fire with `np.add.at`). Report the comparison either way.

### 4.5 Calendar
- Don't store calendar fields in the prediction table. Write `code/panel/calendar.py:add_calendar(df)`
  (`year, month, doy, dow`). The training table may include them by calling that same function.

## 5. Targets (station-day, UTC-6 day, at least 18 valid hours, same rule as `daily_average.py`)

- **NAPS** (`data/Target_Data_2010_2024/PM25_YYYY.csv`, hour-ending local standard time, -999 = missing):
  stations in `data/interim/domain/stations_inside_square.csv` (ON 59, QC 55, MB 3).
  - Ontario is already built and verified in `data/processed/on_utc6_daily/`. Reuse it.
  - For QC and MB, generalize `code/data_preparation/build_on_utc6.py` and `daily_average.py` (don't
    copy-paste them). Resolve the standard offset **per station**; MB is CST, so no shift. Verify with
    `verify_roundtrip.py` / `verify_daily.py`-style checks.
- **US AQS** (`data/raw/AQS/hourly_88101_YYYY.zip`, about 0.85 GB CSV each): stream with `usecols` and
  chunks. Keep sites whose location falls in an `in_square` cell. Use `Date GMT` + `Time GMT` (hour
  start) minus 6 h to get the UTC-6 day.
  - Apply the 18-hour rule per (site, POC, day). Site-day value = mean over the POCs that pass, plus `n_poc`.
  - Report sites with more than one POC and any disagreement between them.
  - Keep negative values as observed.
- Station → cell: invert the lattice formula (section 4.1) on the station's EPSG:3347 coordinates.
  Flag stations on water cells or outside the spine.
- Outputs in `data/interim/stations/`:
  - `stations.parquet`: `station_uid` (`NAPS:060104`, `AQS:36-061-0128`), `network`, `name`, `lat`,
    `lon`, `cell_id`, `cell_idx`, `province_state`, `dist_to_cell_centre_km`, `maiac_covered`
  - `station_day.parquet`: `station_uid, date, cell_id, pm25, n_valid_hours, n_poc`.

  Keep **one row per station** even when stations share a cell. Report which cells hold more than one
  station, and how many stations and station-days fall south of 41°N (no MAIAC).

## 6. Feature stage: gap-filled AOD and context (built only from the base panel)

The goal: every model input of the prediction table is gap-free. In the base panel only MAIAC has large
gaps: clouds and snow (often more than 50% of cell-days), plus the whole strip south of 41°N. HMS has
5 missing days. MERRA and fire have none. Raw columns stay; the feature stage **adds** columns.

**Time direction: backward-only.** The default is that every window uses only day D and earlier days,
so the same code can produce a daily product going forward. Make the direction a single config switch
(`--window backward|centred`). Record it in every output's README and manifest.

### 6.1 AOD context features (deterministic, no model)
Computed per day on the 2-D lattice (`scipy.ndimage`, or sums and counts with box/disk kernels). Only
**observed** `aod055` values are used, and each feature has an obs-count companion:
- `aod_nbr_{25,50,100}km_d0`: mean of observed `aod055` within the radius on day D, plus `_n` (cells used).
- `aod_nbr_50km_d1_3`: the same over days D-1 .. D-3.
- `aod_last_obs`, `aod_last_obs_age_d`: the cell's most recent observed `aod055` within the past 7 days,
  and its age in days (NaN / 0 when there is none).

### 6.2 AOD gap-fill model
- **Target:** `aod055` where observed.
- **Inputs, all gap-free:**
  - MERRA (`merra_totexttau`, the species EXTTAU bands, meteorology);
  - `hms_*`, `fire_*`, calendar;
  - the section 6.1 context features. These may be NaN; LightGBM handles that.
  - **No PM2.5**, so the gap-fill can't leak the target.
  - **No coordinates**, consistent with "no static features"; say so in the README.
- **Training sample:** stratified by month and by region (e.g. 2° blocks), about 10-30 M observed
  cell-days, sized to fit 16 GB RAM. Record the seed.
- **Validation:** "mask transplant". Take an observed day, impose the real missing-data mask of another,
  randomly chosen day of the same month, and score on the hidden cells. Hold out whole years or year
  blocks (e.g. every 5th year) so days from the same synoptic event never sit in both train and test.
  Report RMSE, R² and bias by month, by `aod_last_obs_age_d`, and for cells north vs south of 41°N.
- **Output columns:**
  - `aod055_filled` = observed `aod055` where present, the model prediction elsewhere;
  - `aod_is_filled` (int8: 0 observed, 1 model);
  - `aod_fill_model` (int8 version id).
  The model file, its config and its scores go to `data/interim/features/aod_fill/`.
- **South of 41°N:** with no MAIAC there, every value is filled from MERRA/met/smoke/fire only. Report
  validation separately for cells far from any observed MAIAC (e.g. more than 100 km from 41°N, as the
  closest available proxy), and say plainly in the README that this strip's AOD is MERRA-driven.

### 6.3 Outputs
- `data/interim/features/YYYY/features_YYYY_MM.parquet`: `cell_id`, `date`, and only the new
  columns, aligned positionally with the base panel month. Don't duplicate the base columns.
- The station-cell run and the full run use the same code and the same fitted model.
- Train the gap-fill model on the full base panel (sampled), not on station cells alone. Station cells
  are a biased sample of the domain.

## 7. Driver, tables, and the gap-free contract

- `code/panel/build_base.py --cells {station,all} --years ... --workers N`: loops over months, calls
  every extractor, concatenates positionally, writes Parquet.
  - **Resumable:** skip months whose output exists and passes a quick row-count and schema check.
  - **Logging:** log to a file and run long jobs in the background.
  - **Workers:** keep peak RAM under about 12 GB. Measure one month first.
- `code/panel/build_features.py --cells {station,all}` writes section 6's outputs.
- **Training table:** `code/panel/build_target_table.py` joins `station_day` onto base + features for
  station cells, writing `data/processed/target/target_table.parquet`. Columns: station fields + `date` +
  `pm25`, `n_valid_hours`, `n_poc` + every base and feature column + calendar.
  **Don't drop station-days whose raw `aod055` is NaN.** The model must see the same observed/filled
  mix it will meet when predicting.
- **Prediction table:** `data/processed/prediction/YYYY/prediction_YYYY_MM.parquet`, with `cell_id`,
  `date` + every base and feature column. Row groups of about 1-2 M rows.
  **Estimate disk from one written month before the full run.** Stop if the projection exceeds 120 GB.
- **Gap-free contract:** `docs/MODEL_INPUTS.md` lists the columns intended as model inputs. Each is
  marked gap-free, or "NaN allowed" with the reason. Allowed NaN:
  - raw `aod055`/`aod047`/`injection_height`/`smoke_*`, which are kept for reference; `aod055_filled`
    replaces them as the input;
  - the 6.1 context features;
  - `hms_*` on the 5 missing days;
  - `fire_dist_nearest_km` on days with no fire.

  Everything else must have 0 NaN in every prediction-table month. `verify_prediction.py` asserts this.
- `docs/DATA_DICTIONARY.md`: for each column, the source and version, units, day convention, spatial
  rule, missing-value meaning, gap-free status, and the producing module.
- `docs/BUILD_SQUARE.md`: the exact commands in order.
- `manifest.json` in each output folder: inputs (path, size, mtime; SHA-256 except MAIAC, where you
  hash a 1% sample), the script, package versions, timestamp, and row counts.

## 8. Verification (one `verify_*.py` per extractor and per stage)

- **Row counts.** Every month: rows = n_days x n_cells. `(cell_id, date)` unique. No dates outside
  2010-2024. Training table: `(station_uid, date)` unique, row count = `station_day` rows.
- **Independent spot checks.** About 500 random (cell, date) pairs per extractor, recomputed from the raw
  files by a different code path, with 0 mismatches at 1e-6:
  - MAIAC: open the tile, select pixels whose centres fall in the cell's EPSG:3347 box with shapely,
    and take the mean.
  - MERRA: `src.index(lon, lat)` on the `merra2_utc6_ee` file.
  - HMS: shapely `contains` on the centre, using that day's timestamped polygons.
  - Fire: brute-force haversine.
  - 6.1 context features: brute-force recomputation for a sample of cells.
- **Plausibility.**
  - Value ranges, e.g. AOD roughly -0.1 to 5. Report anything outside.
  - Missing share per variable per month in `qa_report.md`. Flag a month that jumps more than 3x versus
    the same month in adjacent years.
  - MAIAC: median distinct values per cell per month must be greater than 1, and report empty days per year.
- **Gap fill.**
  - `aod055_filled == aod055` wherever `aod_is_filled == 0`;
  - no NaN in `aod055_filled`;
  - filled values within the observed range;
  - the mask-transplant scores from 6.2;
  - a map figure for a sample cloudy day: raw, filled, flag.
- **Backward-only check.** Recompute a sample of features after truncating the input at day D. They must
  be identical, proving that no window looks at D+1 or later.
- **Consistency between the two tables.** Sample 1,000 training rows. Their features must equal the
  prediction-table row for `(cell_id, date)` exactly.
- **Gap-free contract** (section 7), on every prediction month.

## 9. Order

1. Spine, index maps, station mapping, target station-days (NAPS QC/MB, AQS).
2. Each base extractor, run on one month (2015-07) for all cells, plus its verify script. Include the
   MERRA day-window evidence (4.2).
3. Base panel: `--cells station` for all years, then `--cells all`.
4. Feature stage: context features, the gap-fill model with its validation, then the filled columns
   for station cells and for all cells.
5. Training table, prediction table, all verifications, `MODEL_INPUTS.md`, the data dictionary.

## 10. Where to stop and report

- **After step 1:** station counts by network and province, station-days per year, stations south of
  41°N, multi-station cells, flagged stations.
- **After step 2:** timings, MB/s, peak RAM, projected full-run hours and disk, the fire-method
  comparison, MAIAC empty-day counts, and the MERRA day-window evidence.
- **After the gap-fill validation in step 4,** before filling all cells: the scores by month, gap age and
  region. The user decides whether the gap fill is good enough.
- **After step 5.**

Also stop and ask if:
- any source has a whole month missing, or a band name or grid other than described;
- the MERRA files contradict UTC-6;
- a verify check fails twice after real debugging;
- free disk would drop below 40 GB;
- a stage projects more than about 12 hours.

## 11. Defaults already chosen (state them in the README; the user may change them later)

| question | default |
|---|---|
| MERRA-2 source | `data/raw/merra2_utc6_ee` (UTC-6 daily means, 69 x 44 px) |
| MAIAC gap south of 41°N | no MAIAC; AOD there is model-filled from MERRA/met/smoke/fire and flagged `maiac_covered = 0`, `aod_is_filled = 1`. Tiles are discovered from bounds, so a strip re-export can be added later. |
| window direction | backward-only (day D and earlier); switchable to centred |
| HMS missing days (5) | NaN, listed in `MODEL_INPUTS.md` as allowed |
| water cells | kept in the prediction table; `country` is in `cells.parquet` |
| cell coordinates as features | not stored in either table and not used by the gap fill; joinable from `cells.parquet` |
| US stations | included, with a `network` column |
| edge cells (centre outside the square) | excluded |

## 12. How to report each stage

What was built (paths, rows, columns), the QA numbers, every deviation from this document and why, and
open questions. Keep it short.
