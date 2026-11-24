# Task: build the Ontario cell-day data table, 2010-2024

You are working in `/workspace` (the host folder `D:\ai-workspace\project`, mounted into a
Docker container). Build a clean, verified, documented table with **one row per Ontario
CanOSSEM grid cell per day, 2010-01-01 to 2024-12-31**, joining the PM2.5 target and
every covariate source in `data/`. The table feeds a LightGBM PM2.5 model for a thesis,
so provenance and correctness matter more than speed.

Work in stages. **Finish, verify and document each stage before starting the next.**
After stage 1, and again whenever a "stop and ask" condition below is hit, stop and
report back instead of guessing.

---

## 1. What is in /workspace

```
CanOSSEM_V4/
  RASTER_CanOSSEM_CROSSWALK.RData            cell id <-> raster/lat/lon/province crosswalk
  CanOSSEM_estimates_covariates_WFS_YYYY.RData   benchmark estimates; years 2010-2018, 2022, 2024 only
code/data_preparation/
  build_on_utc6.py, verify_roundtrip.py      hourly PM2.5 -> UTC-6 clock      (done, verified)
  daily_average.py, verify_daily.py          UTC-6 daily means, >=18 h rule  (done, verified)
  inspect_raw.py
data/
  Target_Data_2010_2024/PM25_YYYY.csv        raw NAPS hourly, hour-ending local standard time
  processed/on_utc6/                         hourly, UTC-6           (README.md, qa_report.md)
  processed/on_utc6_daily/                   daily target, UTC-6     (README.md, qa_report.md)
      ON_PM25_UTC6_daily_2010_2024.csv       222,369 station-days, 59 stations; 6-line
                                             bilingual preamble, then header + rows:
                                             Pollutant, NAPS ID, City, Province/Territory,
                                             Latitude, Longitude, Date, PM25_daily, n_valid_hours
  raw/maiac_daily_stack/                     MAIAC AOD, 1,440 GeoTIFFs, 190 GB (see 3.2)
  raw/rclone_maiac*.log                      download logs
  MERRA2_FLX_Ontario_2010_2024/MERRA2_FLX_ON_YYYY.tif
  MERRA2_SLV_Winds_Ontario_2010_2024_Final/MERRA2_SLV_WINDS_ON_YYYY.tif
  MERRA_AER_Ontario/MERRA_AER_PANEL_YYYY.tif
  HMS_SMOKE/hms_smokeYYYYMMDD.zip            NOAA HMS smoke polygons, one zipped shapefile per day
  MODIS_fire/modis_YYYY_Canada.csv           MODIS active fire points, Canada only
```

The container has `python3` and `pip` but no geospatial libraries. Create a venv
(`python3 -m venv /workspace/.venv`) and install what you need: `numpy pandas pyarrow
rasterio geopandas shapely pyproj pyreadr`. Record exact versions in
`code/data_preparation/requirements.txt`.

---

## 2. Rules that apply to every stage

1. **`data/raw/` and every raw source folder are read-only.** Never modify, move or
   delete them. Write only into `data/interim/`, `data/processed/`, `code/` and `docs/`.
2. **One day definition: the UTC-6 day** (fixed CST, no DST), because the target is on
   it. Day D = [D 06:00 UTC, D+1 06:00 UTC). See section 3 for how each source maps onto it.
3. **One spine.** Every source is joined onto the same (cell, date) spine. Row count
   of every interim table = n_cells x n_days, exactly. `(cell_id, date)` is unique.
4. **No gap filling, no imputation, no zero-filling.** Missing stays NaN, with an
   observation flag or count column where relevant. Filling happens later, inside
   model folds, and is out of scope.
5. **No lags, rolling windows or neighbourhood aggregates of gridded variables** in this
   task. Those belong to a later feature stage built on top of this table. (Fire and
   smoke distance/radius features in 3.5 and 3.6 are the exception; they are the base
   definition of those variables.)
6. **Look up raster bands by their description (name), never by position.**
7. **Cell id dtype:** pick one (string of the integer, e.g. `"632191"`) and use it
   everywhere. The crosswalk may store ids as float; convert with
   `int()` -> `str()`, never `str(float)`.
8. **Process month by month** (or year by year for small sources). The full panel is
   roughly 40k cells x 5,479 days = 230M rows; never hold it in memory at once.
9. **Parquet**, one file per year per stage, explicit dtypes (`float32` values,
   `int8`/`int16` flags and counts, `datetime64` date), sorted by `(date, cell_id)`.
10. Follow the style of the existing scripts in `code/data_preparation/`: module
    docstring explaining the rule and why, `argparse`, a separate `verify_*.py` per
    build script, and a `README.md` + `qa_report.md` beside every output folder.

---

## 3. Stages

### 3.1 Spine: cells and days

- Read `CanOSSEM_V4/RASTER_CanOSSEM_CROSSWALK.RData` with `pyreadr`. Report every object
  and column it contains before deciding anything.
- Keep Ontario cells (province column; inspect its values first, do not assume "ON").
- Output `data/interim/spine/cells.parquet`: `cell_id`, centroid `lat`, `lon`, and, if
  the crosswalk allows it, the cell footprint (raster row/col and the raster's
  resolution and origin, or polygon bounds). Remove duplicate cell ids and report how
  many there were.
- The calendar is 2010-01-01 .. 2024-12-31 (5,479 days).
- For reference: the previous pipeline used 42,690 Ontario cells. Report your count
  and explain any difference.

**Stop and report after this stage**: cell count, id dtype, coordinate ranges,
whether a footprint is available, and anything surprising in the crosswalk.

### 3.2 MAIAC AOD (`data/raw/maiac_daily_stack/`)

Source: MCD19A2.061 (Terra + Aqua MAIAC Land AOD, 1 km), exported from Google Earth
Engine collection `MODIS/061/MCD19A2_GRANULES` with this recipe:

- `AOD_QA` bits 8-11 kept if in {0 best, 1 water sediments, 11 research quality}
- `Optical_Depth_055` / `_047` x 0.001
- daily value = mean of all Terra + Aqua granules on the **UTC** day

Files: `source_region_maiac_daily_stack_YYYY_MM-RRRRRRRRRR-CCCCCCCCCC.tif`, 180 months
x 8 tiles. Each tile has 5 bands per day, named
`YYYYMMDD_AOD055, YYYYMMDD_AOD047, YYYYMMDD_INJECTION_HEIGHT, YYYYMMDD_SMOKE_AOD055,
YYYYMMDD_SMOKE_MASK`. Grid: EPSG:4326, 0.008983152841195215 deg, full export origin
(-136.5080, 69.5027), 8,851 x 3,173 px, split into 2,816-px tiles. The suffix is the
tile's (row, col) pixel offset.

- **Only four tiles intersect Ontario**: row-col suffixes `0000000000-0000002816`,
  `0000000000-0000005632`, `0000002816-0000002816`, `0000002816-0000005632`. Verify this
  from the file bounds rather than trusting it, and read only the windows that cover
  Ontario cells.
- **Empty-day trap**: the export notebook wrote days with no granule at all as a
  constant 0 in all five bands (not NoData). A day whose five bands are all exactly 0
  over an entire tile is a no-data day -> NaN. Count and report these days per year.
  QA-masked pixels are already NaN.
- **Day convention**: the Terra and Aqua overpasses (~15:30-19:30 UTC over Ontario) fall
  inside the same UTC-6 day as their UTC day, so the UTC day is used as the UTC-6 day.
  Write this in the README.
- **Spatial rule, compute both**:
  - `aod055_mean`, `aod047_mean`: mean of all valid 1 km pixels whose centre falls in the
    cell footprint, plus `aod055_npix` (valid pixel count) and the total pixel count;
  - `aod055_centroid`, `aod047_centroid`: the pixel containing the cell centroid (this
    is what the previous pipeline did, so keep it for comparison).
  If no footprint is available from 3.1, compute only the centroid columns and stop
  and ask.
- Also extract `injection_height`, `smoke_aod055` and `smoke_mask` using the same
  rules (mean + centroid; for `smoke_mask` use the mean, i.e. the fraction of valid
  pixels flagged smoke).
- `aod_obs_flag` = 1 if `aod055_mean` is finite.

**Context: why this source is being rebuilt.** In the previous pipeline, Ontario AOD
for 2012-2017 was a *monthly* composite repeated on every day of the month (one value
per cell per month, about 9% missing in July), while 2018-2023 was daily (about 60%
missing in July), and AOD_047 was empty before 2018. This export replaces both with one
consistent daily series. Your QA must make that kind of defect impossible to miss: per
month, report the missing share and the **median number of distinct values per cell**
(it must be > 1 for a daily product).

### 3.3 MERRA-2 meteorology and aerosol

Three GeoTIFF families, one file per year, already **daily** (band names
`YYYYMMDD_<VAR>`), 43 x 32 px at ~0.494 deg, EPSG:4326, bounds about
(-95.36, 41.50, -74.11, 57.31):

| family | variables |
|---|---|
| FLX | PBLH, QLML, TLML, ULML, VLML, SPEED, SPEEDMAX |
| SLV winds | U2M, V2M, U10M, V10M, U50M, V50M, U250, V250, U500, V500, U850, V850 |
| AER | BCSMASS, OCSMASS, SO4SMASS, SSSMASS25, DUSMASS25, TOTEXTTAU, BCEXTTAU, DUEXTTAU, OCEXTTAU, SUEXTTAU |

- The daily aggregation was done upstream, almost certainly over the **UTC** day, and it
  cannot be re-centred onto UTC-6 from daily composites. Do not try. Record it in the
  README as a known 6-hour offset, and say whether you found any evidence about the
  aggregation window (band metadata, tags).
- Spatial rule: the MERRA-2 pixel is far larger than a cell, so take the value of the
  pixel containing the cell centroid. Name columns `merra_<var>` in lower case.
- Report units if the files carry them; otherwise list them as unknown in the README.
- These rasters cover Ontario only, so they cannot support the 250-1000 km neighbourhood
  aggregates the previous pipeline used. Note this as a limitation; do not attempt it.

### 3.4 Calendar and static

Add `year`, `month`, `doy`, `dow`. Add any static per-cell covariates the crosswalk
carries (for example land cover or road fields, if present); list what you found.

### 3.5 HMS smoke (`data/HMS_SMOKE/`)

One zipped shapefile per day, EPSG:4326, fields `Satellite, Start, End, Density`.
`Start` / `End` are **UTC** in the form `YYYYDDD HHMM` (year, day of year, time).
Some days are missing (2015 has 363 files, 2016 has 364); some old files may lack
`Density` or contain blank values.

- Assign each polygon to the UTC-6 day(s) that its [Start, End] interval overlaps,
  using the timestamps, not the file name. A polygon read from the file for D may
  belong partly to the UTC-6 day D-1.
- Per cell-day, using the cell centroid: `hms_any` (0/1), `hms_density_max`
  (Light=5, Medium=16, Heavy=27 ug/m3, NaN if smoke is present but density is unknown),
  `hms_n_polygons`.
- A missing file is a missing day (NaN), not "no smoke" (0). List missing dates in the
  QA report.

### 3.6 MODIS active fire (`data/MODIS_fire/`)

CSV columns: `latitude, longitude, brightness, scan, track, acq_date, acq_time,
satellite, instrument, confidence, version, bright_t31, frp, daynight, type`.

- `acq_date` + `acq_time` (HHMM) are **UTC**; convert to the UTC-6 day.
- Keep `type == 0` (presumed vegetation fire). Report counts dropped for each other type.
- Per cell-day: `fire_dist_nearest_km` (great-circle distance to the nearest detection
  that day), and `fire_count_<r>km`, `fire_frp_sum_<r>km` for r in {25, 50, 100, 250, 500}.
  Use a BallTree / KD-tree on radians, not a cell-by-fire loop.
- When there are no detections that day: count and FRP = 0, distance = NaN, plus a
  documented convention (do not invent a cap value).
- **Limitation to record**: the file is Canada-only. US fires, which matter for
  southern Ontario, are absent, so near-border radius features are biased low.

### 3.7 Target

- Map each of the 59 NAPS stations (lat/lon from the daily file) to the cell whose
  footprint contains it (or the nearest centroid if no footprint is available; report
  the distance). Output `data/interim/target/stations.parquet` with station -> cell and
  distance, and flag any station that falls outside the Ontario cells.
- Keep the station-day target as its own table (`naps_id, date, cell_id, pm25,
  n_valid_hours`).
- Cell-day target: if more than one station maps to a cell, `pm25` = mean over stations
  that day, with `n_stations`. Report which cells have more than one station.
- Do **not** drop cell-days without a target from the panel; the panel covers all cells,
  and the target is NaN where unmonitored.

### 3.8 Join and final QA

- `data/processed/panel/ontario_panel_YYYY.parquet`: spine + calendar + static + all
  interim tables, joined on `(cell_id, date)`, sorted by `(date, cell_id)`.
- A data dictionary `data/processed/panel/DATA_DICTIONARY.md`: for every column, the
  source, product and version, units, day convention, spatial rule, missing value
  meaning, and the script that made it.

---

## 4. Verification (every stage)

Each `verify_<stage>.py` must check at least:

- row count = n_cells x n_days for the period; `(cell_id, date)` unique; no dates outside
  2010-2024;
- value ranges are physically plausible (AOD roughly -0.1 to 5; report anything outside);
- missing share per month (table in `qa_report.md`); flag any month whose missing share
  jumps by more than 3x vs the same month in adjacent years;
- **spot check against raw**: about 500 random (cell, date) pairs recomputed directly from
  the raw files by an independent code path, with 0 mismatches (tolerance 1e-6);
- for MAIAC also: median distinct values per cell per month > 1, and the number of
  empty (all-zero) days per year.

Every output folder also gets a `manifest.json`: input files with size and SHA-256 (for
the 190 GB MAIAC folder, size + mtime is enough, plus SHA-256 of a 1% sample), the
script path, package versions, run timestamp, and row counts.

---

## 5. Stop and ask

Stop and report back, instead of choosing yourself, if:

- the crosswalk has no usable cell footprint (affects the MAIAC spatial rule);
- the Ontario cell count differs from 42,690 by more than 1%;
- any source has a whole month or year missing, or a band naming scheme other than the
  one described;
- a raster's grid differs from what is described above;
- disk use would exceed 100 GB of new output, or a stage would take longer than about
  6 hours.

---

## 6. Deliverables

```
code/data_preparation/build_spine.py        + verify_spine.py
code/data_preparation/build_maiac.py        + verify_maiac.py
code/data_preparation/build_merra.py        + verify_merra.py
code/data_preparation/build_hms.py          + verify_hms.py
code/data_preparation/build_fire.py         + verify_fire.py
code/data_preparation/build_target_cells.py + verify_target_cells.py
code/data_preparation/build_panel.py        + verify_panel.py
code/data_preparation/requirements.txt
data/interim/<stage>/...                    README.md, qa_report.md, manifest.json each
data/processed/panel/ontario_panel_YYYY.parquet, DATA_DICTIONARY.md, qa_report.md
docs/BUILD.md                               the commands to rebuild everything, in order
```

When you finish each stage, reply with: what was built, row and column counts, the QA
summary, every deviation from this spec and why, and open questions.
