# Audit coverage: datasets, lineage, transformations and QA

Audit date 2026-11-08. Project `D:\ai-workspace\project`. Scope: the 2010–2024 PM2.5 modelling data in `data/processed/` and the intermediate files that produce it.

**Independence caveat.** The pipeline was built by the same AI assistant that ran this audit, in earlier sessions. To limit that:
- every expected value in this audit was recomputed from the raw files by new code in `audit/scripts/`;
- the pipeline's functions, verifiers and earlier review documents were never used to compute expected values;
- earlier review conclusions (`docs/REVIEW_*.md`) are treated as claims to test, not as evidence.

A reviewer who did not build the pipeline would still add value.

## 1. Inventory

### 1.1 Raw inputs (read-only)

| Dataset | Location | Format / size | Used for |
|---|---|---|---|
| NAPS hourly PM2.5 | `data/Target_Data_2010_2024/PM25_YYYY.csv` | 15 CSV, 10–15 MB each. Hour-ending local standard time; −999 = missing | Canadian targets |
| AQS hourly PM2.5 (88101) | `data/raw/AQS/hourly_88101_YYYY.zip` | 15 zip, about 0.85 GB CSV each. Hour-start GMT and local | US targets |
| MAIAC AOD stack | `data/raw/maiac_daily_stack/*.tif` | 8 tiles per month, 5 bands per day, float32, no nodata tag, about 190 GB | AOD inputs |
| MERRA-2 UTC-6 daily | `data/raw/merra2_utc6_ee/*.tif` | 540 GeoTIFF (aer/flx/slv × 180 months), 69 × 44 px at 0.494° | Reanalysis inputs |
| MERRA-2 UTC daily (superseded) | `data/SourceRegion_MERRA_2010_2024-*.zip` | 540 GeoTIFF | Used here only as evidence for the day window |
| HMS smoke | `data/HMS_SMOKE/hms_smokeYYYYMMDD.zip` | 5,474 shapefile zips (5 days missing) | Smoke inputs |
| FIRMS MODIS fire | `data/MODIS_fire/modis_YYYY_{Canada,United_States}.csv` | 30 CSV (no 2025 file) | Fire inputs |
| CanOSSEM lattice | `data/interim/grid/cell_table_square.parquet` | 164,632 cells | Grid definition |
| Thesis region lookup | `data/raw/thesis_regions/naps_id_to_region.csv` | 46 monitors | `on_region` |

### 1.2 Intermediate datasets

| Dataset | Rows | Produced by |
|---|---|---|
| `data/interim/spine/cells.parquet` | 163,787 cells | `spine.py` |
| `data/interim/index/*.npy` | pixel ↔ cell maps | `index_maps.py` |
| `data/interim/stations/stations.parquet` | 388 monitors | `stations.py` |
| `data/interim/stations/station_day.parquet` | 1,079,105 station-days | `targets.py` (+ `code/data_preparation/build_on_utc6.py`) |
| `data/processed/on_utc6*/` | Ontario NAPS on UTC-6 | `build_on_utc6.py`, `daily_average.py` |
| `data/interim/base/all/base_all_YYYY_MM.parquet` | 180 months | `build_base.py` + `extract_{maiac,merra,hms,fire}.py` |
| `data/interim/features/all/features_YYYY_MM.parquet` | 180 months | `build_features.py` |
| `data/interim/spine/cell_ontario.parquet` | 163,787 | `ontario_geo.py` |

### 1.3 Processed datasets (audited)

| Dataset | Rows × columns | Key |
|---|---|---|
| `data/processed/prediction/YYYY/prediction_YYYY_MM.parquet` | 897,388,973 × 71 (180 files) | (`cell_id`, `date`) |
| `data/processed/target/target_table.parquet` | 1,050,847 × 81 | (`cell_id`, `date`) |
| `data/processed/grid/` | 163,787 cells; lookup of 26 provinces | `cell_id` |

## 2. Lineage: raw → processed

```
NAPS CSV ──(ON: build_on_utc6 → on_utc6_daily)──┐
NAPS CSV ──(QC, MB: targets.naps_qc_mb)─────────┼─► station_day ─► drop pm25<0 ─► mean per (cell, day) ─┐
AQS zips ──(targets.aqs, per year file)─────────┘                                                     │
                                                                                                      ▼
MAIAC tif ─(pixel→cell bincount)──┐                                                          target_table
MERRA tif ─(cell centre pixel)────┼─► base (cell × day) ─► features (AOD context) ─► prediction ──┘ (inner join on cell, day)
HMS zips ─(rasterise centre)──────┤                                                     ▲
FIRMS csv ─(haversine BallTree)───┘                      spine / province / cell_ontario ┘
```

## 3. Important transformations and joins

| # | Step | Rule as implemented | Risk if wrong |
|---|---|---|---|
| T1 | Station → cell | Round the EPSG:3347 coordinates onto the lattice | Wrong cell's features |
| T2 | NAPS local standard time → UTC-6 day | Shift by −(offset + 6) h; ON/QC are EST (shift −1 h), MB is CST (shift 0) | Day misalignment |
| T3 | AQS GMT → UTC-6 day | Hour-start GMT − 6 h, floored to the date. **Grouped per annual file** | Day-boundary errors |
| T4 | Daily completeness | At least 18 valid hours; −999 / null = missing | Biased daily means |
| T5 | NAPS duplicate days | Keep the most valid hours; ties go to method 184 | Wrong instrument |
| T6 | AQS multi-POC | Mean over the POCs that pass the 18-hour rule | — |
| T7 | Negative targets | Drop station-days with a negative **daily mean**; negative hours stay inside means | Spec conformance |
| T8 | Cell-day target | Mean of the station means in a cell | Mixing monitors |
| T9 | MAIAC | Mean of the 1 km pixels whose centre lies in the cell; tile-chunks where all 5 bands are 0 are skipped | AOD bias |
| T10 | MERRA-2 | Pixel containing the cell centre; window D 06:00 to D+1 06:00 UTC | Day / space shift |
| T11 | HMS | Polygon valid in [D 06:00, D+1 06:00) UTC; cell centre inside polygon; count, any, max density | Smoke misattribution |
| T12 | Fire | `type = 0`; UTC-6 day of the overpass; great-circle counts, FRP and nearest distance; December reads next year's file | Fire misattribution |
| T13 | AOD context | Lattice disks of 25/50/100 km on day D; D−1..D−3; last observation within 7 days; backward only; 7-day warm-up | Leakage, discontinuity |
| T14 | Training join | Cell-day targets inner-joined to the all-cell base and features | Feature mismatch |
| T15 | Province / region | Water cells take the nearest land province; `on_region` from the thesis lookup with the border-cell rule | Wrong evaluation subset |

## 4. Specification sources and gaps

Specification used: `docs/PROMPT_build_square_tables.md` (the original build specification), the supervisor's direction as relayed by the user, the user's decisions in the session, and `data/processed/README.md`.

These specifications are missing or unclear. I did not invent expected behaviour for them:
- **Hourly validity.** The supervisor asks for "invalid or negative hourly values as missing". The user decided to drop negative *daily* rows only. Neither defines "invalid" for AQS qualifier codes.
- **Training unit.** The specification asks for station-day rows. The build uses cell-day rows, which is documented as a design decision in the README and in `build_target_table.py`.
- **MAIAC export.** The Earth Engine script that produced the MAIAC stack is not in the project. Its QA-bit filter and daily averaging are known only from documentation.
- **AQS qualifiers.** The specification is silent on them.
- **MERRA-2 2 m winds.** The specification is silent on zero values.

## 5. Existing QA checks and what they cover

| Existing check | What it verifies | Sampling | Independent of builder? |
|---|---|---|---|
| `verify_stage1.py` | Spine lattice, station → cell, `station_day` uniqueness / 18 h / date range; 30 ON daily means | 30 ON station-days, recomputed from the **processed** hourly ON file, not raw | Partly |
| `verify_merra.py`, `verify_maiac.py`, `verify_hms.py`, `verify_fire.py` | Base values against the raw sources | About 120 cell-days, **January and July only** | Yes (different code path) |
| `verify_features.py` | Context brute force (1 day), backward-only truncation, warm-up | January and July | Partly (imports `build_features`) |
| `verify_base.py` | Rows, keys, missing share, 3× jumps | All 180 months | — |
| `verify_prediction.py` | Columns, rows, keys, NaN contract, training = prediction | All months, all training rows | — |
| `audit_train_all.py` (scratch) | Training vs `station_day` and vs prediction | All rows | Compares against `station_day`, not raw |
| `build_on_utc6.py` QA | Value multiset conserved through the UTC-6 shift | All ON values | Self-check |
| `docs/MERRA_WINDOW_EVIDENCE.md` | UTC-6 window | July 2015 only | Yes |

### 5.1 Coverage against risks

| Risk | Already tested? | This audit |
|---|---|---|
| Station-day targets for **QC/MB and AQS** match raw | **No.** Nothing recomputes them from raw | Full recomputation (§3 of report 03) |
| ON targets match raw NAPS | Indirectly (from the processed ON product) | Full recomputation from raw |
| Day boundary at year ends (31 Dec / 1 Jan) | **No** | Tested; found F-01 |
| Negative / flagged hourly values | **No** | Quantified; F-02, F-03 |
| Semantics of zeros (wind, smoke mask) | **No** | Found F-04, F-05 |
| Base values outside January / July | **No** | Spot-checked 5 other months (MERRA, fire, HMS) and 3 (MAIAC) |
| MERRA window in years other than 2015 | **No** | Tested all 15 years |
| Keys, rows, NaN contract on all months | Yes | Re-checked independently on all 180 months |
| Training = prediction features | Yes (all rows) | Re-checked on 3 months, all 69 shared columns |
| Context features backward-only | Yes | Brute force including month boundaries |
| Region / border assignment | Partly (in the builder) | Independent recomputation |
| MAIAC empty-day rule fired? | Counter only looks at the first 256-row chunk of each tile, so "never fired" is not proven | No all-zero blocks in 6 sampled tile-days (not a full proof) |
| Value plausibility (all columns) | Ranges on samples only | Full-table min/max/zero/negative/NaN for all 69 columns |

### 5.2 Not verified by this audit

- AOD values against the original MCD19A2 granules. Without the export script, only the exported stack can be checked.
- The MERRA-2 **AER** collection's day window. Only SLV and FLX were fitted; the export notebook uses the same window code for all three.
- A complete HMS recount for all days (200 sampled cell-days only).
- Base ↔ prediction equality, which the pipeline's own audit tested on all months; I re-tested only through the training join.
- Whether the CanOSSEM lattice parameters match CanOSSEM (taken from `data/interim/grid/README.md`).
