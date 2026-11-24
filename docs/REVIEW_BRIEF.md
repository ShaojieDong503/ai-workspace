# Review brief — cell-day data build and QA

For an independent reviewer. Everything below describes code that exists in
`D:\ai-workspace\project\code\panel`. The build has **not** been executed past
stage 1; that is deliberate and is stated where it matters.

## 1. What the build produces

Two Parquet tables, both keyed on `(cell_id, date)`:

| table | unit | rows | columns |
|---|---|---|---|
| `data/processed/target/target_table.parquet` | cell-day **with at least one station** | 1,053,437 | features + target + province + calendar |
| `data/processed/prediction/YYYY/prediction_YYYY_MM.parquet` | every cell-day | 897,388,973 total | features + province only |

- Domain: CanOSSEM 5 km lattice, EPSG:3347, **163,787** in-square cells
  (16,269 of them water, kept), **5,479** days (2010-01-01 … 2024-12-31).
- `cell_id = row*1065 + col`; cell centre is
  `x = 3688478.20 + 4999.827*col`, `y = 5239381.11 - 4998.864*row`.
- Clock: **UTC−6 day**, fixed CST, no DST: [D 06:00 UTC, D+1 06:00 UTC).

## 2. Directory map and entry points

```
code/panel/
  spine.py              cells.parquet            (cell_idx 0..n-1, cell_id order)
  index_maps.py         merra_pix.npy, maiac_*.npy
  stations.py           stations.parquet         (388 stations -> cells)
  targets.py            station_day.parquet      (1,079,105 station-days)
  province.py           province_lookup.csv      (int8 code, 0=Ontario .. 25=water)
  build_base.py         base_{cells}_YYYY_MM.parquet   <- extractor driver
  build_features.py     features_YYYY_MM.parquet       <- context only
  build_target_table.py target_table.parquet
  build_prediction.py   prediction_YYYY_MM.parquet
  verify_*.py           QA (8 scripts)
  utils.py              lattice constants, month iteration, Failures.guard
```

`targets.py` imports `build_on_utc6` from `code/data_preparation` (the verified
UTC−6 target builder). That folder must stay in place.

## 3. Stages and invariants

| stage | input | output | invariant asserted |
|---|---|---|---|
| spine | grid cell table | 163,787 cells | `cell_id == row*1065+col`; `cell_idx` contiguous |
| index | cell centres + raster geotransforms | pixel↔cell maps | every cell maps into MERRA; 4 MAIAC tiles discovered from bounds |
| stations | NAPS extract + AQS zips | 388 stations | max station→centre distance 3.29 km (< half-cell diagonal 3.54) |
| targets | NAPS + AQS hourly | 1,079,105 station-days | ≥18 valid hours; `(station_uid,date)` unique |
| base | 4 extractors | 58 columns/month | every column shape `(n_days, n_cells)`; required schema present |
| features | base panel | 10 context columns | backward-only |
| tables | base + features | target / prediction | uniqueness; contract; sort `(date, cell_id)` |

## 4. Decisions a reviewer should challenge

1. **Cell-day, not station-day.** Stations are mapped to the cell containing
   them; multiple stations in a cell are averaged, with `n_stations` recorded.
   21,094 cell-days average 2–4 stations.
2. **UTC−6 window.** MERRA-2 in Earth Engine is *hourly*; the daily value is
   `filterDate(D 06:00, D+1 06:00).mean()` over 24 hourly images. Evidence that
   the exported archive really is UTC−6 is indirect: against the older UTC-day
   export the same cells differ with median ratio 0.973 and correlation 0.951.
   **Not proven numerically.**
3. **No AOD gap fill.** Raw `aod055`/`aod047` keep their missingness; the model
   must handle NaN. Consequence: 26.9% of training cell-days (282,990) are south
   of 41°N where MAIAC has no coverage at all.
4. **No static features.** No land cover, roads, population, elevation.
5. **Spatial rules.** Fine sources aggregated over the cell footprint (MAIAC 1 km,
   HMS polygons, fire detections); coarse sources sampled at the pixel containing
   the cell centre (MERRA 0.5°).
6. **Edge rule.** A cell is in the domain if its *centre* is inside the square.
   This excludes 6 Québec stations whose cells fall outside, and admits 15 AQS
   sites just outside whose cells fall inside.
7. **Water cells kept**, `country`/`province` null → `province_code = 25`.
8. **HMS missing file = NaN day**, not "no smoke". Five such days.
9. **Fire radii to 500 km** deliberately reach far outside the domain; the US and
   Canada CSVs are both used.

## 5. Gap-free contract (`build_prediction.REQUIRED_COMPLETE`)

Must be complete: `province_code`, fire counts and FRP sums, all `merra_*`.

Explicitly allowed NaN: raw AOD fields (`aod055`, `aod047`, `injection_height`,
`smoke_aod055`, `smoke_mask_frac`), `hms_*` on the five missing days,
`fire_dist_nearest_km` when no detection, and the context columns when no
observation is nearby.

`build_prediction.py` **refuses to write a month** that violates this and names
the offending columns.

## 6. QA scripts and what each proves

| script | method (independent of the builder) | proves |
|---|---|---|
| `verify_stage1.py` | rebuild lattice from row/col; re-derive station cells from coordinates | spine/index/stations/targets correct — **ran, 0 failures** |
| `verify_merra.py` | read one pixel via `src.index` from the geotransform | cell→pixel map and band lookup |
| `verify_maiac.py` | build the cell box in 3347, project pixel centres, average | bincount reduction == footprint mean; median distinct values per cell-month > 1; all-NaN days per year |
| `verify_hms.py` | `shapely.covers` on the cell point | rasterisation == centre-in-polygon; density max |
| `verify_fire.py` | brute-force haversine over the day's detections | BallTree radii, nearest distance, FRP sums |
| `verify_base.py` | full month scan | rows = days×cells, keys unique, dates in month, missing-share table, >3× jump flags |
| `verify_features.py` | brute-force neighbourhood loop; truncated-input recompute | context maths; **no forward-looking leakage** |
| `verify_prediction.py` | contract + train/predict join | gap-free contract; training features identical to prediction features |

All exit non-zero on failure.

## 7. Current state

| stage | code | executed |
|---|---|---|
| spine, index, stations, targets, province | yes | **yes, verified** |
| extractors + base driver | yes | **no** |
| features (context) | yes | **no** |
| target table, prediction table | yes | **no** |
| 8 QA scripts | yes | only `verify_stage1.py` |

## 8. Unverified assumptions (the reviewer should look here first)

1. `extract_hms.py` — `pyogrio.read_dataframe("/vsizip/…")`; whether `Start`/`End`
   parse after pandas type coercion; the `Density` vocabulary.
2. `extract_hms.py` — arena: `rasterio.features.rasterize(all_touched=False)` is
   centre-in-polygon; is that exactly what `shapely.covers` gives in `verify_hms.py`?
3. `extract_maiac.py` — tile windows, band order (day-major × 5 variables), the
   all-zero empty-day rule, `maiac_npix_total` accumulation across tiles.
4. `extract_fire.py` — CSV column names and dtypes (`type`, `frp`, `acq_time`);
   `BallTree(metric="haversine")` expects radians.
5. `extract_merra.py` — 34 variables present in three files; band description
   format `YYYYMMDD_VAR`.
6. `build_features.py` — the disk kernel radius in cells, and the last-observation
   age walk.
7. The UTC−6 evidence (§4.2) — an assumption, not a measurement.

## 9. What a reviewer should do

Cheapest meaningful review:

1. Read `extract_*.py` against the raw files — the four modules that have never
   run are where the errors will be.
2. Check that each `verify_*.py` genuinely uses a different code path from its
   builder (that is the whole value of the QA).
3. Run `build_base.py --cells station --years 2015 2015`, then the four extractor
   verifiers on 2015-07.
4. Check the contract in §5 against the model's actual tolerance for NaN.

## 10. Commands

```
python code/panel/spine.py
python code/panel/index_maps.py
python code/panel/stations.py
python code/panel/targets.py
python code/panel/province.py
python code/panel/verify_stage1.py

python code/panel/build_base.py --cells station
python code/panel/verify_merra.py --cells station --year 2015 --month 7
python code/panel/verify_maiac.py --cells station --year 2015 --month 7
python code/panel/verify_hms.py   --cells station --year 2015 --month 7
python code/panel/verify_fire.py  --cells station --year 2015 --month 7

python code/panel/build_base.py --cells all
python code/panel/verify_base.py --cells all --report data/interim/base/qa_report.md
python code/panel/build_features.py context --cells all
python code/panel/verify_features.py --cells all --year 2015 --month 7

python code/panel/build_target_table.py
python code/panel/build_prediction.py
python code/panel/verify_prediction.py
```
