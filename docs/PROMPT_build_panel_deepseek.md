# Build the Ontario cell-day panel, 2010-2024 — task prompt for DeepSeek

You are a coding agent running on Windows 11 in the DSH harness. Everything you need is
on disk; there is no Docker, no container mount and no `/workspace`. Work in stages, stop
where this document says to stop, and report before continuing.

Project root: `D:\ai-workspace\project`

## 0. Why this exists

The table you are building feeds a LightGBM PM2.5 model for a thesis. One row per Ontario
CanOSSEM grid cell per day, 2010-01-01 .. 2024-12-31, joining the PM2.5 target and every
covariate source under `data\`. Wrong provenance is worse than slow code: when in doubt,
measure the raw file and report the measurement rather than trusting this document.

## 1. Environment

- **OS / shell.** Windows 11. Shell calls go through PowerShell 5.1 semantics: `Set-Content -Encoding UTF8`
  writes a BOM and `-Encoding utf8BOM` does not exist; `$env:PYTHONIOENCODING='utf-8'` must be set before
  running Python that prints non-ASCII, otherwise stdout is cp1252 and you get mojibake or UnicodeDecodeError.
- **Python.** No geospatial libraries are installed. A bundled interpreter exists at
  `C:\Users\ab997\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe`
  (numpy 2.3.5, pandas 3.0.1, openpyxl, Pillow, lxml — **no pyarrow, no rasterio, no geopandas, no pyreadr**).
  Create a venv from it and install what you need:

  ```
  & "<that python.exe>" -m venv D:\ai-workspace\project\.venv
  D:\ai-workspace\project\.venv\Scripts\python.exe -m pip install --upgrade pip
  D:\ai-workspace\project\.venv\Scripts\python.exe -m pip install numpy pandas pyarrow rasterio geopandas shapely pyproj pyreadr scipy scikit-learn tqdm
  ```

  Then pin the resolved versions into `code\data_preparation\requirements.txt` (`pip freeze`).
  If PyPI is unreachable, **stop and report** — do not fall back to CSV or hand-rolled GeoTIFF parsing;
  the panel is far too large for CSV and hand parsing would be unverifiable.
- **Disk.** `D:` is 931.5 GB with **233.9 GB free** (measured 2026-10-31). Check `fsutil volume diskfree D:`
  before each stage. The finished panel alone is likely 60-80 GB (15.2M rows/year x ~80 float32 columns).
  If a stage projects more than 100 GB of new output, or free space would fall below 50 GB, stop and report.
- **No network for data.** Everything is local. Do not use rclone, do not download. Network is allowed only
  for pip.
- **WMI is blocked.** `Get-Volume`, `Get-CimInstance Win32_LogicalDisk` fail with Access denied under the
  sandbox. Use `fsutil volume diskfree D:` or `[System.IO.DriveInfo]` instead.
- **File tooling.** The editor's write operation has been observed to fail with `EISDIR` on paths under
  `project\`; if that happens, fall back to `Set-Content` from PowerShell and re-read the file to confirm.
- **Long work.** Run multi-hour steps as background jobs writing to a log file, then poll the log tail.
  Never hold the full panel in memory: stream month by month, write one Parquet per year.

## 2. Rules for every stage

1. `data\raw\` and every raw source folder are **read-only**. Never modify, move, rename or delete anything
   under `data\Target_Data_2010_2024\`, `data\raw\`, `CanOSSEM_V4\`, `data\MERRA*`, `data\HMS_SMOKE\`,
   `data\MODIS_fire\`. Write only into `data\interim\`, `data\processed\`, `code\` and `docs\`.
2. **One day definition:** the UTC-6 day, fixed Central Standard Time, no DST, because the target is on it.
   Day D = [D 06:00 UTC, D+1 06:00 UTC). How each source maps onto it is given per stage.
3. **One spine.** Every source joins onto the same (cell, date) spine. Every interim table has exactly
   `n_cells x n_days` rows and `(cell_id, date)` is unique. No stage may drop or add rows to fit a source.
4. **No gap filling, no imputation, no zero-filling.** Missing stays NaN, with an observation flag or count
   column where relevant. Filling happens later, inside model folds — out of scope here.
5. **No lags, rolling windows or neighbourhood aggregates** of gridded variables. Those belong to a later
   feature stage. The smoke and fire distance/radius features in stages 3.5/3.6 are the exception: they are
   the base definition of those variables.
6. **Look up raster bands by their description string, never by band index.** Band order varies between files.
7. **Cell id dtype:** string of the integer, e.g. `"632191"`. The crosswalk may store ids as float — convert
   with `int()` then `str()`, never `str(float)`.
8. **Parquet, one file per year per stage**, explicit dtypes: `float32` for values, `int8`/`int16` for flags
   and counts, `datetime64[ns]` for date, `string` for cell_id. Sorted by `(date, cell_id)`.
9. **Style.** Match the existing scripts in `code\data_preparation\` (`build_on_utc6.py`, `daily_average.py`,
   `verify_roundtrip.py`, `verify_daily.py`): module docstring that states the rule and the reason, `argparse`,
   a `build_*.py` plus a separate `verify_*.py`, and a `README.md` + `qa_report.md` next to every output folder.
10. **Honesty over completeness.** If a check fails, report it. Never fabricate a number, never describe a file
    you did not read, never assume a column exists.

## 3. What is on disk — verify, do not trust

```
CanOSSEM_V4\RASTER_CanOSSEM_CROSSWALK.RData                 cell id <-> raster/lat/lon/province
CanOSSEM_V4\CanOSSEM_estimates_covariates_WFS_YYYY.RData    2010-2018, 2022, 2024 (11 files)
data\Target_Data_2010_2024\PM25_YYYY.csv                    raw NAPS hourly, hour-ending local standard time
data\processed\on_utc6\                                     hourly UTC-6 (done)
data\processed\on_utc6_daily\ON_PM25_UTC6_daily_2010_2024.csv   the target: 222,369 station-days, 59 stations
data\raw\maiac_daily_stack\                                 1,440 GeoTIFFs, 189.33 GiB, 2010-2024
data\MERRA2_FLX_Ontario_2010_2024\          MERRA2_FLX_ON_YYYY.tif          15 files
data\MERRA2_SLV_Winds_Ontario_2010_2024_Final\ MERRA2_SLV_WINDS_ON_YYYY.tif 15 files
data\MERRA_AER_Ontario\                     MERRA_AER_PANEL_YYYY.tif        15 files
data\HMS_SMOKE\hms_smokeYYYYMMDD.zip        5,474 zips (5,479-day calendar => 5 missing days)
data\MODIS_fire\modis_YYYY_Canada.csv       15 files
```

The target table is already built and verified — read it, do not rebuild it. Its file has a 6-line bilingual
preamble before the header row, so it needs `skiprows=7`.

## 4. Stages

Stage numbers below follow the original specification (3.1-3.8).

### Stage 3.1 - Spine: cells and days
Read `RASTER_CanOSSEM_CROSSWALK.RData` with `pyreadr`. **Report every object and every column it contains
before deciding anything.** Do not assume a column name, a province code, or a coordinate convention.

Keep Ontario cells (inspect the province column's actual values first — do not assume `"ON"`).

Output `data\interim\spine\cells.parquet` with: `cell_id`, centroid latitude, centroid longitude, and, if the
crosswalk allows it, the cell footprint (raster row/col plus the raster's resolution and origin, or polygon
bounds). Remove duplicate cell ids and report how many there were.

Calendar: 2010-01-01 .. 2024-12-31, which is 5,479 days.

For reference, the previous pipeline used 42,690 Ontario cells. Report your count and explain any difference.

**Stop and report after this stage**: cell count, id dtype, coordinate ranges, whether a footprint is
available and in what form, and anything surprising in the crosswalk.

### Stage 3.2 - MAIAC AOD (`data\raw\maiac_daily_stack\`)
Source: MCD19A2.061 (Terra + Aqua MAIAC Land AOD, 1 km), exported from Google Earth Engine collection
`MODIS/061/MCD19A2_GRANULES` with this recipe:

- `AOD_QA` bits 8-11 kept if in {0 best, 1 water sediments, 11 research quality}
- `Optical_Depth_055` / `_047` scaled by 0.001
- daily value = mean of all Terra + Aqua granules on the UTC day

Files: `source_region_maiac_daily_stack_YYYY_MM-RRRRRRRRRR-CCCCCCCCCC.tif`, 180 months x 8 tiles.
Each tile has **5 bands per day**, named `YYYYMMDD_AOD055`, `YYYYMMDD_AOD047`,
`YYYYMMDD_INJECTION_HEIGHT`, `YYYYMMDD_SMOKE_AOD055`, `YYYYMMDD_SMOKE_MASK`.
Grid: EPSG:4326, 0.008983152841195215 deg, full export origin (-136.5080, 69.5027), 8,851 x 3,173 px,
split into 2,816-px tiles. The suffix is the tile's (row, col) pixel offset.

Only four tiles intersect Ontario: `0000000000-0000002816`, `0000000000-0000005632`,
`0000002816-0000002816`, `0000002816-0000005632`. **Verify this from each file's bounds rather than
trusting it**, and read only the windows that cover Ontario cells — never load a full tile.

**Empty-day trap.** The export notebook wrote days with no granule at all as a constant 0 in all five bands
(not NoData). A day whose five bands are all exactly 0 over an entire tile is a no-data day -> NaN. Count
and report these days per year.

**Day convention.** Terra and Aqua overpasses (~15:30-19:30 UTC over Ontario) fall inside the same UTC-6 day
as their UTC day, so the UTC day is used as the UTC-6 day. Write this in the README.

**Spatial rule — compute both:**
- `aod055_mean`, `aod047_mean`: mean of all valid 1 km pixels whose centre falls inside the cell footprint,
  plus `aod055_npix` (valid pixel count) and the total pixel count.
- `aod055_centroid`, `aod047_centroid`: the value of the pixel containing the cell centroid (this is what the
  previous pipeline did; keep it for comparison).
- If stage 3.1 produced no footprint, compute only the centroid columns and **stop and ask**.

Extract `injection_height`, `smoke_aod055` and `smoke_mask` by the same rules (mean + centroid; for
`smoke_mask` use the mean, i.e. the fraction of valid pixels flagged smoke). `aod_obs_flag = 1` if
`aod055_mean` is finite.

**Context — why this is being rebuilt.** In the previous pipeline, Ontario AOD for 2012-2017 was a monthly
composite repeated on every day of the month (one value per cell per month, ~9% missing in July), while
2018-2023 was daily (~60% missing in July), and AOD_047 was empty before 2018. This export replaces both with
one consistent daily series. Your QA must make that defect class impossible to miss: per month, report the
missing share **and** the median number of distinct values per cell, which must be > 1 for a daily product.

### Stage 3.3 - MERRA-2 meteorology and aerosol
Three GeoTIFF families, one file per year, already daily (band names `YYYYMMDD_<VAR>`), 43 x 32 px at
~0.494 deg, EPSG:4326, bounds about (-95.36, 41.50, -74.11, 57.31):

| family | variables |
|---|---|
| FLX | PBLH, QLML, TLML, ULML, VLML, SPEED, SPEEDMAX |
| SLV winds | U2M, V2M, U10M, V10M, U50M, V50M, U250, V250, U500, V500, U850, V850 |
| AER | BCSMASS, OCSMASS, SO4SMASS, SSSMASS25, DUSMASS25, TOTEXTTAU, BCEXTTAU, DUEXTTAU, OCEXTTAU, SUEXTTAU |

The daily aggregation was done upstream, almost certainly over the UTC day, and cannot be re-centred onto
UTC-6 from daily composites. **Do not try.** Record it in the README as a known 6-hour offset, and state
whether you found any evidence about the aggregation window (band metadata, tags).

Spatial rule: the MERRA-2 pixel is far larger than a cell, so take the value of the pixel containing the cell
centroid. Name columns `merra_<var>` in lower case.

Report units if the files carry them; otherwise list them as unknown in the README.

These rasters cover Ontario only, so they cannot support the 250-1000 km neighbourhood aggregates the previous
pipeline used. Note this as a limitation in the README; do not attempt it.

### Stage 3.4 - Calendar and static
Add `year`, `month`, `doy`, `dow`. Add any static per-cell covariates the crosswalk carries (land cover, road
fields, elevation, population, whatever is there); list what you found and what you added.

### Stage 3.5 - HMS smoke (`data\HMS_SMOKE\`)
One zipped shapefile per day, EPSG:4326, fields `Satellite`, `Start`, `End`, `Density`. `Start`/`End` are UTC
in the form `YYYYDDD HHMM` (year, day of year, time). Some days are missing; some old files may lack
`Density` or contain blank values.

Assign each polygon to the UTC-6 day(s) its `[Start, End]` interval overlaps, **using the timestamps, not the
file name**. A polygon read from the file for D may belong partly to the UTC-6 day D-1.

Per cell-day, using the cell centroid: `hms_any` (0/1), `hms_density_max` (Light=5, Medium=16, Heavy=27 ug/m3,
NaN if smoke is present but density is unknown), `hms_n_polygons`.

A missing file is a missing day (NaN), **not** "no smoke" (0). List the missing dates in the QA report.

### Stage 3.6 - MODIS active fire (`data\MODIS_fire\`)
CSV columns: `latitude, longitude, brightness, scan, track, acq_date, acq_time, satellite, instrument,
confidence, version, bright_t31, frp, daynight, type`.

`acq_date` + `acq_time` (HHMM) are UTC; convert to the UTC-6 day.

Keep `type == 0` (presumed vegetation fire). Report counts dropped for each other type.

Per cell-day: `fire_dist_nearest_km` (great-circle distance to the nearest detection that day), and
`fire_count_<r>km`, `fire_frp_sum_<r>km` for r in {25, 50, 100, 250, 500}. Use a BallTree / KD-tree on
radians — never a cell-by-fire loop.

When there are no detections that day: count and FRP = 0, distance = NaN, and document the convention.
Do not invent a cap value.

Limitation to record: the file is Canada-only. US fires, which matter for southern Ontario, are absent, so
near-border radius features are biased low.

### Stage 3.7 - Target
Map each of the 59 NAPS stations (lat/lon from the daily file) to the cell whose footprint contains it (or the
nearest centroid if no footprint is available; report the distance). Output
`data\interim\target\stations.parquet` with station -> cell and distance, and flag any station that falls
outside the Ontario cells.

Keep the station-day target as its own table (`naps_id, date, cell_id, pm25, n_valid_hours`).

Cell-day target: if more than one station maps to a cell, `pm25` = mean over stations that day, with
`n_stations`. Report which cells have more than one station.

Do not drop cell-days without a target from the panel; the panel covers all cells, and the target is NaN
where unmonitored.

### Stage 3.8 - Join and final QA
`data\processed\panel\ontario_panel_YYYY.parquet`: spine + calendar + static + all interim tables, joined on
`(cell_id, date)`, sorted by `(date, cell_id)`.

`data\processed\panel\DATA_DICTIONARY.md`: for every column — source, product and version, units, day
convention, spatial rule, missing-value meaning, and the script that made it.

## 5. Verification, every stage

Each `verify_<stage>.py` checks at least:

- row count = `n_cells x n_days` for the period; `(cell_id, date)` unique; no dates outside 2010-2024
- value ranges physically plausible (AOD roughly -0.1 to 5; report anything outside)
- missing share per month as a table in `qa_report.md`; flag any month whose missing share jumps by more
  than 3x versus the same month in adjacent years
- ~500 random (cell, date) pairs recomputed **directly from the raw files by an independent code path**,
  with 0 mismatches at tolerance 1e-6
- for MAIAC additionally: median distinct values per cell per month > 1, and empty (all-zero) days per year

Every output folder also gets a `manifest.json`: input files with size and SHA-256 (for the 190 GB MAIAC
folder, size + mtime is enough plus SHA-256 of a 1% sample), the script path, package versions, run
timestamp, and row counts.

## 6. Stop and report — do not guess

Stop and report back instead of choosing yourself if:

- the crosswalk has no usable cell footprint (affects the MAIAC spatial rule)
- the Ontario cell count differs from 42,690 by more than 1%
- any source has a whole month or year missing, or a band naming scheme other than the one described
- a raster's grid differs from what is described above
- disk use would exceed 100 GB of new output, or a stage would take longer than about 6 hours
- pip cannot install the geospatial stack
- a verification check fails twice in a row after genuine debugging

## 7. Deliverables

```
code\data_preparation\build_spine.py        + verify_spine.py
code\data_preparation\build_maiac.py        + verify_maiac.py
code\data_preparation\build_merra.py        + verify_merra.py
code\data_preparation\build_hms.py          + verify_hms.py
code\data_preparation\build_fire.py         + verify_fire.py
code\data_preparation\build_target_cells.py + verify_target_cells.py
code\data_preparation\build_panel.py        + verify_panel.py
code\data_preparation\requirements.txt
data\interim\<stage>\...        README.md, qa_report.md, manifest.json each
data\processed\panel\ontario_panel_YYYY.parquet, DATA_DICTIONARY.md, qa_report.md
docs\BUILD.md                   the commands to rebuild everything, in order
```

## 8. How to report each stage

When a stage is finished (or you hit a stop condition), reply with:

1. **What was built** — files, paths, row and column counts.
2. **QA summary** — the checks that ran, the numbers, and anything that failed.
3. **Deviations** — every place you departed from this document, and why.
4. **Open questions** — anything you had to assume, or that the user must decide.

Keep it short enough to read. Never claim a check passed unless you ran it in this session and saw the output.

## 9. Operating notes for this harness

- Work in stages: do not start stage N+1 until stage N's verify script passes and its README, qa_report and
  manifest exist.
- Use a task list to track the stages; keep it current.
- Long steps go to background jobs with a log file; poll the log tail rather than blocking.
- The harness truncates long command output. Redirect big listings to a file and read the tail.
- Absolute Windows paths everywhere. There is no `/workspace`, no `bash`, no `sudo`, no `apt`.
- Never write outside `D:\ai-workspace\project`. The sandbox will refuse, and that refusal is expected.


