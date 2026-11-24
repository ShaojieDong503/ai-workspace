# PM₂.₅ modelling data, 2010–2024: what is in this folder

Daily PM₂.₅ predictors on a 5 km grid over Ontario and its surroundings, plus the
monitor observations used as the training target. Prediction table built 2026-11-06;
training targets rebuilt 2026-11-08 (negative hourly values treated as missing, §3).
How this differs from the data used in the thesis: §9.

| Folder | What | Files | Size |
|---|---|---|---|
| `prediction/YYYY/prediction_YYYY_MM.parquet` | Every grid cell × every day: the 63 model inputs | 180 (one per month) | 28 GB |
| `target/target_table.parquet` | Cell-days that have a monitor: the 63 inputs + observed PM₂.₅ | 1 | 150 MB |
| `grid/` | The grid itself: cell coordinates, province, Ontario region, distance to Ontario | 3 | 5 MB |
| `on_utc6/`, `on_utc6_daily/` | Ontario NAPS monitors, hourly and daily, on the same UTC-6 day. The Ontario targets are computed from the hourly files; the daily files keep negative hours and round to 2 decimals, so they are not used for training | | 58 MB |

All tables are Parquet (ZSTD), readable with pandas, pyarrow, R `arrow`, DuckDB or Polars.

## 1. The grid and the day

- **Cell.** A 5 km × 5 km cell on the CanOSSEM Lambert grid (Statistics Canada Lambert, EPSG:3347). `cell_id = row × 1065 + col`. Cell centre coordinates are in `grid/cells.parquet`; the big tables carry only `cell_id`.
- **Domain.** 163,787 cells, latitude 38.5–59.2° N, longitude 98.2–67.0° W: Ontario and parts of Manitoba, Quebec, Nunavut and 21 US states. 16,269 cells are open water (Great Lakes, Hudson Bay).
- **Day.** Every daily value uses the same day, **UTC-6**: day D runs from D 06:00 UTC to D+1 06:00 UTC (local standard time in Manitoba, one hour behind Ontario standard time; no daylight saving). This is the CanOSSEM day.
- **Period.** 2010-01-01 to 2024-12-31, 5,479 days.

## 2. Prediction table (`prediction/`)

One file per month. Each file has **days × 163,787 rows**, one per cell per day, sorted by `date` then `cell_id`. A year is about 60 M rows; the whole table is 897 M rows.

Columns: `cell_id`, `date`, `province_code`, then the 63 model inputs (§4), then 5 MAIAC quality columns (§4.1). Calendar fields and coordinates are not stored; derive them from `date` and join `grid/cells.parquet`.

```python
import pandas as pd
p = pd.read_parquet("prediction/2023/prediction_2023_06.parquet")
cells = pd.read_parquet("grid/cells.parquet")          # lat, lon, x_3347, y_3347
p = p.merge(cells[["cell_id", "lat", "lon"]], on="cell_id")
```

A single month is 5 M rows × 71 columns, about 1.3 GB in memory. To read a subset, pass `columns=[...]` or `filters=[("cell_id", "in", ids)]`.

## 3. Training table (`target/target_table.parquet`)

**1,042,511 rows**, one per cell-day where at least one monitor has a valid daily mean. 351 cells, 2010-01-01 to 2024-12-31. Every predictor value is identical to the prediction table's row for the same cell and day (checked on all rows).

Columns: everything in the prediction table, plus:

| Column | Type | Meaning |
|---|---|---|
| `pm25` | float32 | Observed daily PM₂.₅, µg/m³. Mean over the UTC-6 day of the valid hours, at least 18 of 24. Where several monitors share a cell, the mean of their daily means. Range 0–581. |
| `n_stations` | int8 | Number of monitors averaged (1–4) |
| `n_hours_min` | int8 | Fewest valid hours among those monitors (18–24) |
| `province_state` | string | Province or state name of the cell |
| `year`, `month`, `doy`, `dow` | int | Calendar of `date`; `dow` is 1 = Monday … 7 = Sunday |
| `dist_ontario_km` | float32 | Distance from the cell to Ontario, km (0 = Ontario cell). Not a model input; use it to select cross-border monitors |
| `on_region` | int8 | Ontario evaluation region, as in the thesis (§5); 0 = outside Ontario. Not a model input |

**Monitors.** 388 monitors from two networks: NAPS (Canada; Ontario 59, Quebec 49, Manitoba 3) and US EPA AQS (277 monitors in 19 states). A monitor belongs to the cell its location falls in.

**Cells with observations.** 351 of the 163,787 cells (0.21%) have a monitor; 48 of them are among the 42,972 Ontario cells. Monitors open and close, so fewer are active in any one year: 130 cells in 2010 rising to 295 in 2024 overall, and 38–41 Ontario cells per year (29 in 2024). Where several monitors share a cell, `n_stations` counts them: 1 in 1,021,631 rows, 2 in 18,177, 3 in 929, 4 in 1,774.

Use `on_region > 0` (not `province_state`) to select Ontario rows: it follows the monitors' jurisdiction in three border cells (§5).

**Target rules.**
- Hourly values flagged missing (−999) are excluded; a daily mean needs at least 18 valid hours.
- **Negative hourly values are treated as missing**, like −999: they are not averaged and do not count toward the 18 hours. So no daily mean is negative. Compared with keeping negative hours, 11,237 station-days fall below 18 hours and are lost (AQS 11,130, NAPS 107); the rest rise by 0.044 µg/m³ on average.
- AQS hours are pooled across the annual files, so 31 December includes its hours from the next year's file.

Training rows per year: 2010 40,287 · 2011 47,298 · 2012 47,403 · 2013 50,596 · 2014 53,525 · 2015 56,709 · 2016 62,128 · 2017 70,538 · 2018 73,997 · 2019 81,553 · 2020 84,126 · 2021 88,863 · 2022 90,914 · 2023 93,924 · 2024 100,650.

Training rows by Ontario region, and cross-border rows by distance:

| `on_region` | Monitors | Cells | Rows |
|---|---:|---:|---:|
| 1 Toronto | 11 | 8 | 27,143 |
| 2 Central | 11 | 9 | 43,090 |
| 3 East | 9 | 8 | 35,513 |
| 4 West_SW | 10 | 10 | 38,906 |
| 5 West_NH | 10 | 7 | 36,995 |
| 6 North | 7 | 6 | 26,446 |
| **Ontario total** | **58** | **48** | **208,093** |
| 0 outside Ontario | | 303 | 834,418 |

One Ontario monitor (Windsor, NAPS 060211) is not counted: it shares a cell with four Detroit monitors (§5).

| Cells within … of Ontario (`dist_ontario_km`, Ontario included) | Rows | Cells |
|---|---:|---:|
| 100 km | 374,642 | 105 |
| 200 km | 500,126 | 151 |
| 300 km | 639,909 | 200 |

## 4. The 63 model inputs

"NaN allowed" means the value can be missing for the reason given; LightGBM handles NaN directly. Every other input has **no missing values** anywhere in the table. No gap filling is applied.

### 4.1 Satellite AOD: MODIS MAIAC (MCD19A2 C6.1), 5 inputs

Exported from Google Earth Engine (`MODIS/061/MCD19A2_GRANULES`): AOD scaled by 0.001, kept where `AOD_QA` bits 8–11 are 0 (best), 1 (water sediments) or 11 (research quality), and averaged over all Terra and Aqua granules of the day. Both overpasses fall in daytime, so the UTC day and the UTC-6 day hold the same granules. The 1 km pixels whose centre falls inside the cell are then averaged. Export coverage stops at 41° N; cells south of it are always NaN (`maiac_covered = 0`).

| Column | Unit | Meaning | NaN allowed |
|---|---|---|---|
| `aod055` | – | Aerosol optical depth at 550 nm, cell mean | yes: cloud, snow, no overpass, or south of 41° N |
| `aod047` | – | AOD at 470 nm, cell mean | as above |
| `injection_height` | m | Smoke plume injection height | as above; present only over detected smoke |
| `smoke_aod055` | – | MAIAC `SMOKE_AOD055` band: AOD at 550 nm over smoke-flagged pixels | as above; present only over detected smoke |
| `smoke_mask_frac` | 0–1 | Share of the cell's pixels flagged as smoke | south of 41° N only |

Quality columns (not model inputs):

| Column | Meaning |
|---|---|
| `aod055_npix` | Number of valid AOD pixels averaged; 0 exactly where `aod055` is NaN, so it doubles as the AOD-missing indicator |
| `maiac_npix_total` | Number of MAIAC pixels in the cell (valid or not) |
| `maiac_covered` | 1 if MAIAC covers the cell at all, 0 south of 41° N; constant per cell |
| `aod055_centroid`, `aod047_centroid` | AOD of the single pixel at the cell centre, for comparison with the cell mean; NaN when that pixel is missing |

### 4.2 Reanalysis: NASA MERRA-2, 34 inputs

Daily mean of the hourly fields over the UTC-6 day, exported from Google Earth Engine on a 0.494° grid (about 50 km). A cell takes the value of the pixel containing its centre, so neighbouring cells share a value. Never missing.

| Columns | Unit | Meaning |
|---|---|---|
| `merra_bccmass`, `merra_occmass`, `merra_so4cmass`, `merra_sscmass25`, `merra_ducmass25` | kg/m² | Column mass of black carbon, organic carbon, sulfate, sea salt PM₂.₅, dust PM₂.₅ |
| `merra_bcsmass`, `merra_ocsmass`, `merra_so4smass`, `merra_sssmass25`, `merra_dusmass25` | kg/m³ | Surface concentration of the same five species |
| `merra_bcexttau`, `merra_ocexttau`, `merra_suexttau`, `merra_duexttau`, `merra_totexttau` | – | Extinction AOD at 550 nm: black carbon, organic carbon, sulfate, dust, total |
| `merra_pblh` | m | Planetary boundary layer height |
| `merra_qlml` | kg/kg | Specific humidity, lowest model level |
| `merra_tlml` | K | Air temperature, lowest model level |
| `merra_ulml`, `merra_vlml` | m/s | Eastward / northward wind, lowest model level |
| `merra_speed`, `merra_speedmax` | m/s | Surface wind speed, mean and maximum |
| `merra_u2m`, `merra_v2m`, `merra_u10m`, `merra_v10m`, `merra_u50m`, `merra_v50m` | m/s | Eastward / northward wind at 2, 10 and 50 m |
| `merra_u850`, `merra_v850`, `merra_u500`, `merra_v500`, `merra_u250`, `merra_v250` | m/s | Eastward / northward wind at 850, 500 and 250 hPa |

For the aerosol-vs-meteorology comparisons: the 15 mass and extinction columns are the **MERRA-2 aerosol** group; the other 19 are **meteorology**.

### 4.3 Smoke plumes: NOAA HMS, 3 inputs

Analyst-drawn smoke polygons. A cell has smoke on a day when its centre is inside a polygon valid at any time in that UTC-6 day.

| Column | Unit | Meaning | NaN allowed |
|---|---|---|---|
| `hms_any` | 0/1 | Any smoke polygon over the cell | yes: the 5 days with no HMS file (2012-10-07, 2015-06-02, 2015-08-20, 2016-03-06, 2016-11-12) |
| `hms_n_polygons` | count | Number of overlapping polygons | as above |
| `hms_density_max` | 5 / 16 / 27 | Densest polygon: light / medium / heavy (approximate µg/m³ class values) | yes: no smoke over the cell, a missing day, or density not recorded (all polygons Feb–May 2010) |

### 4.4 Active fire: NASA FIRMS MODIS, 11 inputs

MODIS fire detections (vegetation fires only, `type = 0`), from Canada and US files, assigned to the UTC-6 day of their overpass time. Distances are great-circle; detections up to 500 km outside the domain count.

| Column | Unit | Meaning | NaN allowed |
|---|---|---|---|
| `fire_count_25km` … `fire_count_500km` | count | Detections within 25, 50, 100, 250, 500 km of the cell centre | no (0 = no fire) |
| `fire_frp_sum_25km` … `fire_frp_sum_500km` | MW | Summed fire radiative power within the same radii | no (0 = no fire) |
| `fire_dist_nearest_km` | km | Distance to the nearest detection that day | yes: no detection anywhere that day |

### 4.5 AOD context, 10 inputs

Built from the observed `aod055` only (no filling), looking backward in time only. They help where the cell itself is cloud-covered. Neighbourhoods are disks on the grid that include the cell itself.

| Column | Meaning | NaN allowed |
|---|---|---|
| `aod_nbr_25km_d0`, `aod_nbr_50km_d0`, `aod_nbr_100km_d0` | Mean observed AOD of cells within 25 / 50 / 100 km, same day | yes: no observed cell in the disk |
| `aod_nbr_25km_d0_n`, `aod_nbr_50km_d0_n`, `aod_nbr_100km_d0_n` | Number of observed cells behind each mean | no (0 = none) |
| `aod_nbr_50km_d1_3` | Mean observed AOD within 50 km over the previous 3 days (D−1 to D−3) | yes: none observed |
| `aod_nbr_50km_d1_3_n` | Number of observed cell-days behind it | no |
| `aod_last_obs` | The cell's most recent observed AOD, today or up to 7 days back | yes: none in the last 7 days |
| `aod_last_obs_age_d` | Its age in days (0 = observed today, up to 7) | as above |

These carry history across month boundaries; only January 2010 (the first month) starts without history.

## 5. Grid files (`grid/`)

| File | Rows | Columns |
|---|---|---|
| `cells.parquet` | 163,787 | `cell_idx` (0-based position within each day), `cell_id`, `row`, `col`, `x_3347`, `y_3347` (cell centre, m), `lat`, `lon`, `country`, `province_state` (null for water) |
| `province_lookup.csv` | 26 | `province_code` → `province_state`. Water cells take the province of the nearest land cell, and a water cell holding a monitor takes the monitor's |
| `cell_ontario.parquet` | 163,787 | `cell_id`, `dist_ontario_km`, `on_region` |

**Ontario regions (`on_region`).** The six evaluation regions of the thesis: 1 Toronto, 2 Central, 3 East, 4 West_SW (southwest), 5 West_NH (Hamilton–Niagara–Waterloo), 6 North; 0 = outside Ontario. The thesis assigned each monitor to a region by city (`naps_id_to_region.csv`, 46 monitors).
- The 13 monitors added since then are in the same cities and take that city's region; Morrisburg, the only new city, is East.
- A monitored cell takes its monitors' region.
- An unmonitored Ontario cell takes the region of the nearest Ontario monitor. This is for mapping only: the regions are defined by monitors, not by boundaries, so "North" covers everything north of the southern regions.

**Distance to Ontario (`dist_ontario_km`).** Distance from the cell centre to the nearest Ontario cell, so it follows the political border through the Great Lakes and is in 5 km steps.

**Border cells with monitors.** A cell holding monitors takes the monitors' jurisdiction when they all agree, since the target there is their mean:
- Sarnia (ON) sits in a cell whose centre is in Michigan; that cell counts as Ontario.
- The Port Huron (MI) and Gatineau (QC) monitors sit in cells whose centre is in Ontario; those cells count as outside Ontario.
- Windsor (ON) shares a cell with four Detroit monitors; that cell stays Michigan, so the Windsor monitor is not in the Ontario selection.

## 6. Quality checks

All 180 months passed:
- **Structure.** Every day holds all 163,787 cells; keys unique and sorted; types as specified; no missing value outside the "NaN allowed" columns above.
- **Against the raw sources.** MERRA-2, MAIAC, HMS and fire values recomputed independently from the raw files for January and July of every year; context features recomputed, including across the year boundary.
- **Internal consistency.** For example, `aod055_npix = 0` exactly where `aod055` is missing; fire counts never decrease with radius; smoke density present only where there is smoke.
- **Training table.** Matches the monitor daily means exactly and matches the prediction table on every row and column.
- **Independent audit (2026-11-08).** Every station-day and training row was recomputed from the raw NAPS and AQS files by separate code and matches (largest difference 0.00002 µg/m³). MERRA-2, fire, HMS and MAIAC values were spot-checked in months outside January and July, and the MERRA-2 UTC-6 window was confirmed for every year.

Details: `docs/REVIEW_full_build_2010_2024.md` and the `audit/` folder in the project repository (`audit/04_findings.md` lists every finding).

## 7. Known gaps and caveats

- **HMS:** 5 missing days (§4.3); no density recorded February–May 2010.
- **Fire, 31 December 2024:** detections between 00:00 and 06:00 UTC on 1 January 2025 are not included (the 2025 file is not yet added), so that day may undercount.
- **MAIAC:** no AOD south of 41° N. Winter AOD is sparse; in January most cells are missing on most days.
- **Fire distance rounding:** in a handful of cell-days `fire_dist_nearest_km` is stored as exactly 25/100/250/500 while the count at that radius is 0; the true distance is just over the radius (single-precision storage).
- **Resolution:** values are reported on a 5 km grid. MERRA-2 inputs are much coarser (~50 km), so this is a 5 km prediction grid, not independently validated 5 km resolution.
- **Monitors are sparse in Northern Ontario:** the North region has 7 monitors in 6 cells (Thunder Bay, Sault Ste. Marie, Sudbury, North Bay, Parry Sound), 26,446 training rows in all.
- **AQS quality flags:** US hours that EPA flags with a quality-concern code (for example QX "does not meet QC criteria", 5 "outlier") are still used as valid; only negative and missing hours are excluded. Wildfire and exceptional-event flags are real smoke and are kept on purpose.
- **AQS reads higher than NAPS:** US monitors average 0.6–5.4 µg/m³ above the Canadian ones in the same year, partly geography and partly instruments. The share of US rows grows over time.
- **MERRA-2 2 m wind:** `merra_u2m` and `merra_v2m` are exactly 0 in about 3.3 M cell-days over forest, April to November. The zeros are in the MERRA-2 export itself and most likely mean "not defined" (2 m is below the canopy), not calm air. The 10 m and 50 m winds are unaffected.
- **MAIAC smoke columns:** `smoke_mask_frac` counts cloud-covered pixels as "no smoke", so it understates smoke on cloudy days; it is also NaN in 40,424 covered cell-days on 2011-06-10, 2011-06-20 and 2012-02-01 (satellite orbit gaps). `injection_height` is negative in 1,924 cell-days (minimum −647 m), as in the MAIAC product.

## 8. Sources

| Source | Product | Used for |
|---|---|---|
| NAPS (Environment and Climate Change Canada) | Hourly PM₂.₅ | Canadian targets |
| US EPA AQS | Hourly PM₂.₅ (parameter 88101) | US targets |
| NASA MODIS MAIAC | MCD19A2 C6.1, 1 km AOD, via Google Earth Engine | §4.1 |
| NASA GMAO MERRA-2 | Hourly aerosol (AER), surface flux (FLX) and single-level (SLV) collections, via Google Earth Engine | §4.2 |
| NOAA Hazard Mapping System | Daily smoke polygons | §4.3 |
| NASA FIRMS | MODIS active fire archive, country files for Canada and the US | §4.4 |
| CanOSSEM | 5 km grid definition | §1 |
| Thesis (2026) | Monitor-to-region lookup | §5 |

Code: `code/panel/` in the project repository; `RUN.md` there lists the build order.

## 9. Comparison with the thesis data

The thesis results used the data in `D:\lambda` (thesis draft §2 and Appendix A, post-defence data in `Post_Defense_Code\Data`). Comment IDs (MN-xx) refer to the committee comments.

| | Thesis (lambda) | Now |
|---|---|---|
| **Period** | 2012–2023 (12 years) | 2010–2024 (15 years) |
| **Grid** | CanOSSEM raster in latitude/longitude, 10,648 Ontario cells | CanOSSEM 5 km Lambert grid (EPSG:3347), 163,787 cells covering Ontario (42,972 cells) plus neighbouring provinces and states |
| **Day** | Target on Eastern Standard Time; MERRA-2 on UTC (a known mismatch, MN-14) | Everything on the UTC-6 day, the same as CanOSSEM |
| **Monitors** | 46 Ontario NAPS | 388: NAPS in Ontario (59), Quebec (49) and Manitoba (3), plus 277 US AQS |
| **Cells with observations** | 41 of 10,648 Ontario cells (0.39%) | 351 of 163,787 cells (0.21%).<br>Ontario: 48 of 42,972 cells (0.11%).<br>2012–2023 only: 331 cells in all, 47 in Ontario.<br>In any one year, 38–41 Ontario cells (29 in 2024). |
| **Training rows** | 169,882 cell-days | 1,042,511 cell-days.<br>Ontario: 208,093, or 168,937 for 2012–2023. |
| **Daily completeness** | No minimum number of hours | At least 18 valid hours |
| **Negative values** | Kept (22 negative cell-days) | Negative hours treated as missing before the daily mean, so no target is negative |
| **AOD** | Gap-filled by a LightGBM model fitted on all data, which risks leakage (MN-17) | Observed AOD only, with missing values left as missing, plus a pixel count. Context features look backward only: neighbourhood means and the last observation within 7 days |
| **Fire** | VIIRS 375 m; counts, FRP and distance within 100–1000 km | MODIS (FIRMS); counts, FRP and distance within 25–500 km |
| **Smoke (HMS)** | Neighbourhood means within 100–1000 km | Values at the cell itself: any smoke, number of polygons, maximum density. The 5 missing days are NaN |
| **Burned area (MCD64A1)** | Used, with look-ahead within the month (MN-15) | Removed |
| **MERRA-2** | Neighbourhood means within 100–1000 km | The cell's own value, with more variables: species mass and extinction, winds at 2, 10 and 50 m, maximum wind speed |
| **NARR weather** | 14 variables, including precipitation, radiation, cloud cover and visibility | Removed |
| **Static features** | Land cover, roads, latitude/longitude, cell area, day of year | Removed |
| **Lag and rolling features** | Lag-1 and 3- and 7-day rolling statistics on 73 variables (511 features) | AOD only |
| **Predictors** | 651 | 63 |
| **Missing values** | Filled with 0 (MN-18) | Left as NaN. A 0 only means "no event", such as no fire or no smoke |
