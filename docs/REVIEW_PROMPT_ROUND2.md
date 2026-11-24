# Review request — cell-day panel build, round 2

You are reviewing a data pipeline, not writing features for it. Your job is to
find bugs that would produce a *plausible but wrong* table, and to say which
findings block the full run.

Project root: `D:\ai-workspace\project`. Python: `C:\Python314\python.exe`.
Everything is local; there are no downloads to do.

## What the build produces

Two Parquet tables, keyed on `(cell_id, date)`:

| table | unit | rows |
|---|---|---|
| `data/processed/target/target_table.parquet` | cell-day with at least one station | 1,053,437 |
| `data/processed/prediction/YYYY/prediction_YYYY_MM.parquet` | every cell-day | 897,388,973 |

- 163,787 cells on the CanOSSEM 5 km lattice (EPSG:3347), 5,479 days
  (2010-01-01 … 2024-12-31).
- `cell_id = row*1065 + col`; centre `x = 3688478.20 + 4999.827*col`,
  `y = 5239381.11 - 4998.864*row`.
- Clock: UTC-6 day, [D 06:00, D+1 06:00).
- 63 model inputs from MAIAC (1 km, aggregated), MERRA-2 (0.5 deg, sampled),
  HMS smoke, MODIS fire, and 10 AOD-context columns. No static covariates.
- **No AOD gap fill**: raw `aod055`/`aod047` keep their missingness.

Code lives in `code/panel/` (24 modules). `docs/REVIEW_BRIEF.md` has the
round-1 context; `docs/BUILD_SQUARE.md` has the command order;
`docs/MERRA_WINDOW_EVIDENCE.md` has the day-window evidence.

## Round 1 findings — all claimed fixed

Sixteen findings were raised and reported fixed: MERRA band matching, MAIAC
windowed reads, fire per-cell counts, HMS unclosed rings, `cell_id` holding
`cell_idx`, HMS timestamp parsing, polygon counts, HMS NaN semantics, training
table sourcing, month-boundary warm-up, 31 December fire, prediction-verifier
NaN handling, contract coverage, feature-check geometry, MAIAC tile summing, and
the fire boundary case.

Four more were found afterwards:

1. **HMS polygons were tested half a cell (2.5 km) from the cell centre** — the
   raster origin was not offset by half a pixel. Fixed in `extract_hms._window`.
2. `build_features.py` had never actually been updated (a lost edit), so the dead
   `fit`/`apply` were still present. Rewritten: context only.
3. `build_base` crashed on static per-cell columns (`maiac_npix_total`). Now
   broadcast.
4. MAIAC was 37x slower than necessary. Rewritten around chunked all-band reads
   (1218 s -> 33 s per month), with `aod055`/`aod047` bit-identical to the slow
   version.

## What has actually been executed

Only a canary month: **2015-07, station cells (351 cells)**.

```
build_base   1 month, 42 s, 10,881 rows = 31 x 351, 0 failures
verify_merra 0 failures   200 checks, max|diff| 0.00e+00
verify_maiac 0 failures    50 checks, max|diff| 7.1e-08
verify_hms   0 failures   150 checks exact
verify_fire  0 failures   150 checks, boundary case pass
verify_base  0 failures   rows, keys, dates, spine membership
```

Nothing has run for all 163,787 cells, and the feature stage and both tables have
never executed. Treat every unrun path as unproven.

## Where I would look first

These are the changes with the least runtime evidence, in the order I trust them
least.

### `extract_maiac.py` — rewritten for speed, so re-verify correctness
- The empty-day rule is now evaluated **per 256-row chunk**, not per tile-day.
  Ask: can a real day contain an all-zero chunk that would be wrongly skipped,
  and is the `_empty_tile_days` counter meaningful when only `r0 == 0` counts?
- `npix_tot` accumulates across tiles via `bincount(target[target >= 0])`. Check
  that a cell straddling a tile seam does not have its pixel count reset.
- The centroid assignment uses `c_row`/`c_col` against the chunk range. Check the
  boundary row (`c_row == r0 + rows`) and that `c_loc` indexes the right cells.
- Confirm the chunk loop covers `h` exactly when `h % CHUNK_ROWS != 0`.

### `build_features.py` — never executed
- `context_month` re-expands arrays to the caller's row order via
  `v[day_ix, row_of]`. Check this against a shuffled `df`.
- The month warm-up concatenates the previous month's last 7 days, then drops
  rows before `first`. Check that the drop uses the same predicate as the keep,
  and that a January starts from the previous December.
- `grid_layout` is built from the cells *present in df*, so a station-only run
  uses a coarser lattice than the all-cell run. Confirm the tables never mix the
  two (the training table is built from `features/all`).

### `build_target_table.py` — now reads all-cell months
- It inner-joins targets onto `base/all` + `features/all` per month. A month
  missing either file is skipped and only counted. Check that a partially built
  all-cell run cannot silently produce a training table that drops station-days.

### `verify_prediction.py` and the contract
- `classify()` splits columns by prefix. Check that a renamed column cannot fall
  into neither list, and that `maiac_covered`/`maiac_npix_total` are correctly
  in the complete set.
- The identity check merges training rows into the prediction month and compares
  every shared column. Check the NaN pattern comparison actually fails on a
  dropped column, not just on a value change.

### `extract_hms.py`
- The half-pixel origin fix: confirm it is right in both axes given `DY` is
  negative, and that it agrees with `shapely.covers` on 150 samples (it did on
  the canary, but only for July).
- `_read_polygons` transforms every geometry to 3347 inside a loop. Check the
  cost for a heavy smoke day and whether the raw-WKB path handles multipolygons
  and holes correctly.

### `extract_fire.py`
- December loads `year + 1`. Check January: does a 1 January 00:00-06:00 UTC
  detection land in the previous year's December file, and is that file loaded
  when building December of year-1?
- The radius search is per fire per radius (5 searches x up to ~13,000 fires a
  day). Fine at 351 cells; measure at 163,787 and say whether it must change.

## Open decisions (do not fix silently — flag which block)

1. **`maiac_covered` is a static latitude flag** (1 north of 41 N, 0 south),
   which contradicts the "no static features" rule. Should it be a model input,
   a QA column, or dropped?
2. **26.9% of training cell-days are south of 41 N with no MAIAC AOD.** With no
   gap fill, the model sees NaN there. Is that acceptable, or does that strip
   need its own treatment?
3. **Fire speed**: is the current per-fire search acceptable for the all-cell
   run, or must it be rewritten before the full build?

## What to run

```
python code/panel/build_base.py --cells station --years 2015 2015 --month 7
python code/panel/verify_merra.py --cells station --year 2015 --month 7
python code/panel/verify_maiac.py --cells station --year 2015 --month 7
python code/panel/verify_hms.py   --cells station --year 2015 --month 7
python code/panel/verify_fire.py  --cells station --year 2015 --month 7
python code/panel/verify_base.py  --cells station --report data/interim/base/station/qa_report.md
```

A second canary worth running: `--years 2015 2015 --month 6 7`, then
`build_features.py context --cells station`, to exercise the month warm-up, which
a single month cannot test.

There is a built month on disk already:
`data/interim/base/station/base_station_2015_07.parquet`. Delete it to rebuild.

## How to report

Same shape as round 1, which was useful:

1. **Crashes** — location, observed error, fix.
2. **Wrong values that pass the shape checks** — these matter more; say how you
   detected each one.
3. **Gaps in the QA** — where a verifier is not independent, or would pass a bug.
4. **Checked and correct** — so I stop worrying about it.
5. **Fix order**, and which findings block the full run.

Do not write to `data/raw/` or `data/processed/`. `data/interim/` is fair game.
Run anything you need; report only what you actually saw.
