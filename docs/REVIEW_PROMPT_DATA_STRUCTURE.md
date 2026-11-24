# Review request — is the data structured as expected?

You are auditing **built data**, not code. The question is whether the files on
disk have the schema, keys, ordering, types and completeness the design says
they must. Do not rerun the pipeline; read what is there and check it.

Project root `D:\ai-workspace\project`. Python `C:\Python314\python.exe`
(pandas, pyarrow, pyarrow.parquet available). Everything is local.

## What exists

Only **2015** has been built. Twelve months, three stages, plus a training table
in a scratch folder. 2010–2014 and 2016–2024 do not exist yet; do not treat their
absence as a finding.

```
data/interim/base/all/        base_all_2015_01.parquet .. _12.parquet    1.13 GB
data/interim/features/all/    features_2015_01.parquet .. _12.parquet    0.80 GB
data/processed/prediction/2015/  prediction_2015_01.parquet .. _12.parquet  1.80 GB
data/interim/scratch/target2015/target_table.parquet                     57,441 rows
```

Each stage folder also holds `build.log`, `failures.json` (all empty lists) and,
for base, `qa_report.md`.

## The schema each file must have

Every file is one month, rows = days_in_month x 163,787 cells (all-cell stages).

| table | columns | composition |
|---|---:|---|
| base | **60** | `cell_id, date` + 58 source columns |
| features | **12** | `cell_id, date` + 10 context columns **only** |
| prediction | **71** | `cell_id, date, province_code` + 58 source + 10 context |
| training | **79** | prediction's 71, plus `province_state, pm25, n_stations, n_hours_min, year, month, doy, dow` |

The 58 source columns, by family:

- MAIAC (10): `aod055, aod047, injection_height, smoke_aod055, smoke_mask_frac,
  aod055_npix, maiac_npix_total, aod055_centroid, aod047_centroid, maiac_covered`
- MERRA (34): `merra_bccmass … merra_v850`
- HMS (3): `hms_any, hms_n_polygons, hms_density_max`
- Fire (11): `fire_dist_nearest_km`, `fire_count_{25,50,100,250,500}km`,
  `fire_frp_sum_{25,50,100,250,500}km`

The 10 context columns (features stage): `aod_nbr_{25,50,100}km_d0`, their `_n`
companions, `aod_nbr_50km_d1_3` + `_n`, `aod_last_obs`, `aod_last_obs_age_d`.

`code/panel/model_inputs.py` is the single source of truth for which of these are
model inputs (**63**) and which are QA-only (5: `aod055_npix`,
`maiac_npix_total`, `aod055_centroid`, `aod047_centroid`, `maiac_covered`).
Read that file rather than trusting this table.

## Invariants to check

**Keys and layout**
- `(cell_id, date)` unique within every file, and no pair repeated across months.
- Rows sorted by `(date, cell_id)`.
- `cell_id` is a lattice id in **573,402 … 1,017,901** (`row*1065 + col`), never a
  0-based position, and every id is in `data/interim/spine/cells.parquet`.
- Dates inside the month, all `days_in_month` present.
- dtypes: `cell_id` int32, `date` datetime64[ns], floats float32, `province_code`
  int8, pixel counts int16. **A silent upcast to float64 doubles the file and can
  hide integer overflow** — worth checking each column, not just a few.

**Completeness contract**
- 44 columns must have **zero** NaN in every month; 19 are allowed NaN by design.
- Recompute that split from `model_inputs.py` plus `build_prediction.ALLOWED_NAN`
  and confirm each file matches. A column that moved between the two lists is a
  finding even if it looks plausible.

**Allowed-NaN sanity** (the NaNs must be in the right *places*, not just allowed)
- `aod055` NaN share ~0.60–0.75 in 2015, and **100%** for cells south of 41°N.
- `hms_any`, `hms_n_polygons`: NaN on exactly the missing HMS day, 2015-06-02, and
  nowhere else.
- `fire_dist_nearest_km`: NaN only on days with no detection anywhere within
  500 km; counts and FRP sums must be 0 there, never NaN.
- Context columns: NaN only where no observation is in range.

**Internal consistency of the 5 QA columns**
- `aod055_npix == 0` exactly where `aod055` is NaN.
- `maiac_covered == 1` exactly where `maiac_npix_total > 0`.
- `aod055_centroid` finite implies `aod055` finite. The reverse does not hold:
  a cloudy centre pixel inside a partly clear cell gives centroid NaN with aod
  finite, 16k-504k times a month. That is correct, not a defect.

**Cross-table**
- Featuries files must not repeat any base column.
- Prediction must contain no `pm25`, no `lat`/`lon`/`x_3347`/`y_3347`, and no
  `year`/`month`/`doy`/`dow` — those are joined or recomputed, by design.
- Sample training rows and confirm all 63 model inputs equal the prediction row
  for the same `(cell_id, date)` exactly, NaN pattern included.

**Physical**
- ZSTD compression, row groups of roughly 1 M rows.
- Per-month sizes near 97 MB (base), 69 MB (features), 154 MB (prediction). A
  month 3x out of line is a finding.

## Already checked — don't redo, but spot-check if cheap

`verify_base` (12 months, 0 flags), `verify_prediction` (12 months, 44 contract
columns complete, 63 inputs identical, all training rows matched), `failures.json`
empty in all three stages, and the extractor verifiers on 2015-07.

What is **not** checked, and where you should spend your effort:

1. Whether the column *sets* are exactly right — the verifier checks that inputs
   are present, not that nothing extra slipped in.
2. dtypes column by column.
3. The placement of NaNs in the allowed-NaN columns (the HMS missing day aside).
4. The QA-column consistency rules above.
5. Sorting, cross-month duplication, and physical layout.
6. The training table's own shape: 57,441 rows from **171** cells in 2015 (not
   351 — only stations reporting that year). `pm25` should contain a few small
   negatives and no `-999`.
7. Whether the training table's `province_state`/`province_code` agree with the
   cell's province in the spine.

## How to report

Group by severity, and give the evidence inline:

1. **Structural** — wrong column set, wrong dtype, wrong row count, unsorted,
   duplicate keys, wrong path or naming. These invalidate a rerun.
2. **Value placement** — a NaN where it must not be, or a value where it must be
   NaN; QA columns disagreeing with their parents.
3. **Cosmetic** — sizes, row-group counts, log format.

For each: file, column, what you observed (a number, a count, a sample row), and
why you expected otherwise. If something is fine, say so in one line so the
reader knows it was looked at.

Do not modify any file. Do not write outside `data/interim/scratch/`.
