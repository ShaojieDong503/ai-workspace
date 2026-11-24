# Task: Colab notebook to download hourly MERRA-2 for the study square and build UTC-6 daily stacks

You are a coding agent. Write a **Google Colab** notebook,
`D:\ai-workspace\project\code\colab\MERRA2_Hourly_UTC6_Colab.ipynb`, that the user will run in Colab.
It must:

1. download **hourly** MERRA-2 for the study region and every variable in section 3, 2010-2024, from
   NASA GES DISC with server-side subsetting, saving to **Google Drive**;
2. build **UTC-6 daily** stacks from those hourly files, in the band-naming layout the project already
   uses, so they can replace the current UTC-day MERRA files.

You **cannot run Colab or log in to Earthdata yourself.** Write the notebook so its first cells are the
tests the user runs, and test the pure logic locally where you can (section 8).

## 1. Why

The PM2.5 target is a daily mean over the **UTC-6 day** (fixed CST, no DST): day D = [D 06:00 UTC,
D+1 06:00 UTC). The MERRA-2 files we have now are Earth Engine exports of 24-hour means over the
**UTC** day, 6 hours off the target. Daily means can't be re-centred, so we go back to the hourly data.
Having hourly data also allows other daily statistics (e.g. PBLH max/min) without another download.

## 2. Colab environment

- **Install:** first cell is `!pip install -q earthaccess xarray netCDF4 rasterio tqdm`.
  Print the installed versions.
- **Drive:** mount it with `drive.mount('/content/drive')`. Every output goes under a single configurable
  root, `DRIVE_ROOT = '/content/drive/MyDrive/MERRA2_hourly_square'`. The Colab disk (`/content`) is
  scratch only: it's lost when the session ends.
- **Drive space:** print free Drive quota before downloading (e.g. via `shutil.disk_usage` on the mount).
  Make a one-month test, and from it project the total size. **Stop with a clear message if the
  projection exceeds the free space.** Free accounts have 15 GB, and the hourly set is likely 15-25 GB
  compressed. Offer `KEEP_HOURLY = True/False`. If it's False, a month's hourly files are kept on
  `/content` only until that month's UTC-6 stacks are written to Drive, then deleted.
- **Login:** use `earthaccess.login()`. Prefer Colab Secrets (`google.colab.userdata.get(
  'EARTHDATA_USERNAME')` / `'EARTHDATA_PASSWORD'`, exported as env vars for `strategy='environment'`),
  falling back to `strategy='interactive'`. **Never print credentials, and never write them into the
  notebook, a `.netrc` on Drive, or any output.** A markdown cell explains:
  - creating an Earthdata account;
  - approving the "NASA GESDISC DATA ARCHIVE" application;
  - adding the two Colab secrets.
- **Sessions drop** (idle timeouts, 12 h limit). Every long loop must resume from Drive state: skip
  units whose output exists and validates. Re-running the cell continues where it stopped. Flush the log
  and status table to Drive regularly (at least every 50 files).

## 3. What to download

| collection (ESDT, version 5.12.4) | variables |
|---|---|
| `M2T1NXAER` (`tavg1_2d_aer_Nx`) | `BCCMASS OCCMASS SO4CMASS SSCMASS25 DUCMASS25`, `BCSMASS OCSMASS SO4SMASS SSSMASS25 DUSMASS25`, `BCEXTTAU OCEXTTAU SUEXTTAU DUEXTTAU TOTEXTTAU` |
| `M2T1NXFLX` (`tavg1_2d_flx_Nx`) | `PBLH QLML TLML ULML VLML SPEED SPEEDMAX` |
| `M2T1NXSLV` (`tavg1_2d_slv_Nx`) | `U2M V2M U10M V10M U50M V50M U250 V250 U500 V500 U850 V850` |

- Keep the variable lists in one config cell, plus an `EXTRA_VARIABLES` dict that is **empty by
  default**. Add a comment suggesting candidates, e.g. SLV `T2M QV2M PS SLP TQV` and FLX `PRECTOT`.
  Before anything downloads, check every name against the dataset's actual variable list (fetch the
  OPeNDAP DDS/DMR of one granule).
- **Region:** lon -100.3 .. -65.2, lat 37.0 .. 60.8. This covers the study square
  (98.2°W-66.9°W, 38.5°N-59.1°N) with at least one native pixel of margin. The native grid is
  lon = -180 + 0.625 i, lat = -90 + 0.5 j. Compute the index ranges from the file's own `lon`/`lat`
  arrays.
- **Time:** UTC days 2010-01-01 .. **2025-01-01**. The last UTC-6 day, 2024-12-31, needs the hours up
  to 2025-01-01 05:30 UTC. That's 5,480 days x 3 collections, about 16,440 requests.
- **Granule names vary by stream:** `MERRA2_300` (2010), `MERRA2_400` (2011 onward), and `MERRA2_401`
  for reprocessed months (e.g. 2020-09, some 2021 months). **Never build file names by hand.**
  Discover granules through CMR (`earthaccess.search_data(short_name=..., version='5.12.4',
  temporal=...)`), **one query per collection per month** (not per day), and record the granule name used.
- **Subsetting:** use GES DISC OPeNDAP with a constraint expression (variables + `time`, `lat`, `lon`
  index ranges), returning NetCDF4. Use an authenticated `requests` session from earthaccess. Make the
  first test cell fetch one granule and print the exact URL. Full global files (several hundred MB per
  day for AER) must never be downloaded. If OPeNDAP proves unreliable, the fallback is GES DISC's
  subsetting service (Harmony); document why if you switch.

## 4. Download rules

- **Unit of work:** one collection-month. Download its daily subsets in parallel (`ThreadPoolExecutor`,
  default 6 workers, configurable) to `/content/scratch/`. Validate them. Merge the month into **one
  NetCDF4 file** (zlib level 4, float32), write it to Drive under a temporary name, then rename.
  That's 540 files on Drive instead of 16,440 small ones; Drive handles that far better.
  Path: `{DRIVE_ROOT}/hourly/{AER|FLX|SLV}/MERRA2_{AER|FLX|SLV}_hourly_YYYY_MM.nc4`. January 2025
  needs only its first day: name it `..._2025_01_01.nc4`.
- **Validate every daily subset before merging:**
  - exactly 24 time steps at HH:30 UTC (the tavg1 convention: each value is the mean over [HH:00, HH+1:00));
  - all requested variables present with the expected lat/lon shape;
  - the fill value (`_FillValue`, about 1e15) converted to NaN, with the NaN count recorded.
  The merged month must have 24 x n_days time steps in order with no duplicates. Keep units and
  long_name. Add global attributes: source granules, URL pattern, download time.
- **Errors:** retry with exponential backoff on 429/5xx and timeouts. After the maximum retries, record
  the failure and move on. A final cell re-tries only the failures.
- **State on Drive:**
  - `{DRIVE_ROOT}/download.log`;
  - `{DRIVE_ROOT}/manifest.csv` with columns collection, date, granule, url, bytes, n_time, status
    (append and dedupe; never rewrite past rows inconsistently).
- **One-month timed test** (2015-07, all three collections) before the full loop. Print MB per day,
  seconds per day, projected hours and GB. The full loop runs only when the user sets `RUN_FULL = True`.

## 5. UTC-6 daily stacks (a separate section; runs on validated monthly files)

- UTC-6 day D = the 24 hourly steps stamped D 06:30 UTC .. D+1 05:30 UTC. Day D therefore needs the last
  6 hours from the next month's file at month ends: load month M plus month M+1's first day. If any of
  the 24 is missing, that day-variable is NaN. Never average fewer hours.
- **Statistics:** `mean` for every variable. A config dict `EXTRA_STATS`, defaulting to
  `{'PBLH': ['max', 'min'], 'TLML': ['max', 'min']}`. Band names `YYYYMMDD_<VAR>` for the mean and
  `YYYYMMDD_<VAR>_<STAT>` for the extras, e.g. `20150701_PBLH_MAX`.
- **Output:** one GeoTIFF per collection per month,
  `{DRIVE_ROOT}/utc6_daily/merra2_{aer|flx|slv}_utc6_daily_YYYY_MM.tif`. float32, NaN as nodata,
  DEFLATE compression, EPSG:4326, on the **native** 0.625° x 0.5° grid. The geotransform comes from the
  lon/lat pixel centres ± half a pixel; check the lat order and flip so north is up.
  Set each band's description, and write units to band tags. Write to `/content` first, then copy to Drive.
- Resumable in the same way: skip months whose output exists and has the expected band count.
- Also build a **UTC-day mean for 2015 only**, the same way, to
  `{DRIVE_ROOT}/check_utc_2015/`. It's used only for check 2.

## 6. Checks (cells at the end; print the results and also write them to `{DRIVE_ROOT}/qa_report.md`)

1. **Completeness:** for every collection, hourly months 2010-01 .. 2024-12 plus 2025-01-01 present;
   every UTC-6 day 2010-01-01 .. 2024-12-31 has 24 hours for every variable; NaN counts inside the
   square (expect 0).
2. **Against the existing Earth Engine exports (UTC-day means):**
   - Configurable Drive paths to the unzipped export folders, default
     `/content/drive/MyDrive/SourceRegion_MERRA_2010_2024` (`source_region_merra_{aer,flx,slv}_daily_stack_YYYY_MM.tif`)
     and `/content/drive/MyDrive/SourceRegion_MERRA_AER_SFC_2010_2024` (`source_region_merra_aersfc_daily_stack_YYYY_MM.tif`).
   - Those files are on a resampled grid: 0.4940734062657368°, upper-left corner
     (-100.29690147194458, 60.77102897068563). Band names `YYYYMMDD_<VAR>`; look them up by
     description. Compare 2015 UTC-day means at their pixel centres against the native pixel containing
     each centre.
   - Expect a relative difference < 1e-5 for at least 99% of pixel-days. Print the distribution per
     variable. If the folders aren't found, skip the check with a clear message telling the user what
     to upload.
3. **UTC-6 vs UTC:** for a few sample days, recompute UTC-6 day D from the hourly files independently
   and check it equals the stack. Also show that it differs from the UTC-day mean. Include one month-end
   day, which uses the next month's file.
4. **Plausibility:** ranges per variable (PBLH > 0, TLML around 220-320 K, masses ≥ 0), and units as
   reported by the source.

## 7. Notebook layout

Cells in order:
1. A markdown overview: what the notebook does, how long it takes, how to resume after a disconnect.
2. Install.
3. Mount Drive.
4. Login.
5. Config.
6. Variable-name check.
7. One-granule subset test.
8. One-month timed test, with the Drive-space projection.
9. Full download (guarded by `RUN_FULL`).
10. Retry failures.
11. UTC-6 build.
12. Checks.
13. Write `{DRIVE_ROOT}/README.md`: collections, versions, region, index ranges, time-stamp
    convention, streams seen, fill-value handling, UTC-6 rule, statistics, grid, how to re-run.

Keep the logic in plain functions defined in one or two cells, and make the run cells short calls.
Match the style of `code/colab/MERRA2_AER_Surface_Gapfill_Colab.ipynb`: a markdown header with tables,
a config block of constants, short commented code.

## 8. Testing before handing over

- Unit-test the pure functions locally (Python on this machine, synthetic xarray data):
  - UTC-6 window selection, including month ends and 2024-12-31;
  - the 24-hour completeness rule;
  - the extra statistics;
  - the geotransform and north-up flip;
  - the band naming.
  Show the test code and its output in your reply; don't add it to the notebook.
- Check that the notebook is valid JSON (nbformat 4) and that every code cell at least compiles
  (`compile()` each cell, skipping `!` lines).

## 9. Report back with

- The notebook path, and the cells the user must edit (paths, secrets, `RUN_FULL`).
- Your local test results.
- Anything you couldn't verify without Colab/Earthdata access, so the user knows what the first test
  cells must confirm.
