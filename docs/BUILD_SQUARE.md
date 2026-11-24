# Build the square-domain panel and tables

Commands from `D:\ai-workspace\project`, with `$env:PYTHONIOENCODING='utf-8'`.
Python: `C:\Python314\python.exe` (no venv needed; see
`code/panel/requirements.txt`).

## 1. Spine, index maps, stations, targets

```
python code/panel/spine.py
python code/panel/index_maps.py
python code/panel/stations.py
python code/panel/targets.py
python code/panel/province.py
python code/panel/verify_stage1.py
```

Built: 163,787 cells, 388 stations, 1,079,105 station-days, 0 failures.
`province.py` writes `data/interim/spine/province_lookup.csv`
(0 Ontario … 25 water).

## 2. Base panel

```
python code/panel/build_base.py --cells station
python code/panel/build_base.py --cells all --years 2010 2010     # timing run
python code/panel/build_base.py --cells all
```

Resumable per month, logs to `data/interim/base/<cells>/build.log`, failures with
tracebacks to `failures.json`.

## 3. Features

There is **no AOD gap-fill** in this build: the raw `aod055` / `aod047` keep
their native missingness and the model meets NaN directly. The only feature
stage is deterministic context around the observations.

```
python code/panel/build_features.py context --cells station
python code/panel/build_features.py context --cells all
```

## 4. Tables

```
python code/panel/build_target_table.py
python code/panel/build_prediction.py
```

- `target_table.parquet` — **one row per cell-day with at least one station**,
  target = mean of the stations' UTC-6 daily means, with `n_stations`.
  1,053,437 rows from 351 cells.
- `prediction/YYYY/prediction_YYYY_MM.parquet` — every cell-day,
  897,388,973 rows over the full period.

Both carry `province_code` (see the lookup) immediately after `date`, and the
prediction table refuses to write a month that breaks the gap-free contract.

## Alignment rules

| source | native | rule |
|---|---|---|
| MAIAC AOD | 1 km | aggregate over the cell footprint (mean, pixel count) |
| HMS smoke | ~1 km polygons | aggregate (presence, count, max density) |
| MODIS fire | 1 km detections | aggregate (count, summed FRP); nearest distance is a point statistic |
| MERRA-2 | 0.5° x 0.625° | sample the pixel containing the cell centre |

No static features: no land cover, roads, population or elevation.

## Notes

- MAIAC is the only heavy source: one month is 4 tiles x 155 bands of 2,816^2,
  roughly 25 GB of decompression. Measure timing and peak RAM on one month.
- One written prediction month is enough to project disk before the full run.

## 5. QA

Run each against the month it is meant to check. Every script recomputes the
sampled values from the raw source by a different code path than the builder, so
a bug that is plausible in the extractor cannot also be present in the check.

```
# base panel, one month first
python code/panel/verify_merra.py   --cells all --year 2015 --month 7
python code/panel/verify_maiac.py   --cells all --year 2015 --month 7
python code/panel/verify_hms.py     --cells all --year 2015 --month 7
python code/panel/verify_fire.py    --cells all --year 2015 --month 7
python code/panel/verify_base.py    --cells all --report data/interim/base/qa_report.md

# feature stage
python code/panel/verify_features.py --cells all --year 2015 --month 7

# tables
python code/panel/verify_prediction.py
```

Each exits non-zero on any failure, so they can gate a remote run:

```
python code/panel/verify_merra.py ... || exit 1
```

`verify_base.py` is the only one that must see every month: it checks row counts,
key uniqueness and the missing-share table, and flags any month whose missing
share is more than 3x the same month in the adjacent years.

`verify_maiac.py` also reports the two diagnostics that would expose a monthly
composite rather than a daily product: all-NaN days per year, and the median
number of distinct AOD values per cell per month (must be > 1).

`verify_features.py` ends with the backward-only proof: it recomputes a month
from a frame truncated at day D and asserts the feature row for D is unchanged.
