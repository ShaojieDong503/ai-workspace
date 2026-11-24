# Panel build review

2026-11-04 · reviewer findings for the author of `code/panel`

The build in `code/panel` can't run yet. All four extractors fail on 2015-07, and seven more bugs would write wrong values that pass the shape checks. Each finding gives the location, what happens and the fix.

The crashes were reproduced by calling each `extract()` directly on the 351 station cells for 2015-07. Nothing under `data/` was written.

## 1. Crashes

Every extractor raises on 2015-07, so `build_base.py` writes no month at all. The error messages below are the ones observed.

| # | Location | What happens | Fix |
|---|---|---|---|
| 1 | `extract_merra.py:41` | `_band_map` keeps only names exactly 12 characters long. Real names are 13–17 (`20150701_PBLH`), so 0 of 217 bands match: `RuntimeError: MERRA variables not found`. | Match `re.fullmatch(r"\d{8}_[A-Z0-9]+", d)` and compare `d.split("_", 1)[1] == var` exactly, not with `endswith`. |
| 2 | `extract_maiac.py:106` | `src.read(band)` reads the full tile (2816×2816), but the index map covers only the window (e.g. 1658×1368): `ValueError: operands could not be broadcast (2268144,) (7929856,)`. | Read with `window=Window(c_off, r_off, w, h)` from `t["window"]`. Make the same change for the centroid values at line 125; otherwise they'd be read at the wrong offset once the crash is fixed. |
| 3 | `extract_fire.py:91-92` | `counts` has one entry per fire (2,606 that day), but it's written into a row with one entry per cell (351): `ValueError: could not broadcast`. | `out[f"fire_count_{r}km"][i] = np.bincount(flat, minlength=m)`. |
| 4 | `extract_hms.py:66` | 14 of 5,474 HMS zips have unclosed rings (the first are 20100319, 20100320, 20100422, 20150120 and 20150318). `pyogrio.read_dataframe` raises `GEOSException`. `verify_hms.py` uses the same read, so it crashes too. | `meta, _, wkb, fields = pyogrio.raw.read(...)`, then `shapely.from_wkb(wkb, on_invalid="fix")`. Tested on 20150318: 11 of 11 polygons recovered. |

## 2. Wrong values that pass the shape checks

These are worse than the crashes. Once the crashes are fixed, each of these writes a full-size table that looks plausible but is wrong.

| # | Location | What happens | Fix |
|---|---|---|---|
| 5 | `build_base.py:75` | The column named `cell_id` is filled with `cell_idx`. Real ids run 573,402–1,017,901 and never overlap 0–163,786. As a result the target join returns 0 rows, `add_province` maps to NaN and fails on `astype("int8")`, and every verifier raises `KeyError` at `cs.loc[r.cell_id]`. | Write `cells["cell_id"].to_numpy()[cell_idx]`, with `cells` ordered by `cell_idx`. Add an assertion to `verify_base.py` that every `cell_id` is in the spine. |
| 6 | `extract_hms.py:44-45` | `ddd` is a string, so `ddd - 1` raises `TypeError` (reproduced). `except Exception: continue` swallows it for every polygon, so `hms_any` would be 0 on every cell-day. | Use `int(rest.split(" ")[0][:3])`. Count skipped rows, and fail the month if more than 1% are skipped instead of ignoring them. |
| 7 | `extract_hms.py:138` | `rasterize` defaults to replace mode, so `hms_n_polygons` is only ever 0 or 1. | Pass `merge_alg=rasterio.enums.MergeAlg.add`. |
| 8 | `extract_hms.py:100, 156-157` | A missing HMS file leaves the day at 0, and the int16/int8 casts can't hold NaN. That contradicts brief §4 ("missing file = NaN") and `ALLOWED_NAN` in `build_prediction.py`. | Start `poly_cnt` as NaN in float32 and keep `hms_any` as float32. Alternatively, keep the integer columns and add an `hms_file_missing` int8 flag. |
| 9 | `build_features.py:80`, `build_target_table.py:63` | `context --cells station` crashes on a shape mismatch, because `rr`/`cc` are built from all cells. Even if it ran, neighbourhoods over station cells alone aren't the feature the prediction table gets. The §10 commands only build `features/all`, so `if ff.exists()` would silently drop every context column from training. | Build the training table from `base/all` and `features/all`, or straight from the prediction files, and subset to station cell-days. The two tables then match by construction. Keep `--cells station` for base-panel QA only. |
| 10 | `build_features.py:92-93, 105` | `last_val`/`last_age` and the D−1..D−3 window start empty each month, so days 1–3 differ from a continuous run. The truncation test cuts mid-month (day 11), so it can't catch this. | Prepend the previous month's last 7 days (`AGES`) before the loop, then drop them from the output. Extend the truncation test to cover day 1 of a month. |
| 11 | `extract_fire.py:38-48` | For 31 December, the detections from 1 January 00:00–06:00 UTC are in next year's CSV, which is never loaded. That window includes the Terra night overpass. `verify_fire.py` copies the same loader, so it agrees with the bug. | For December, also load `year + 1` and keep rows whose UTC-6 day falls in the month. |

## 3. Gaps in the QA

The verifiers mostly use their own code paths, as intended. Two holes would let a real bug through, two checks would fail on correct code, and one (#16) isn't fully independent.

| # | Location | Problem | Fix |
|---|---|---|---|
| 12 | `verify_prediction.py:72` | `diff[np.isfinite(diff)]` discards NaN-versus-value mismatches, so a NaN in training against a value in prediction passes. That's exactly how bug #9 would show up. | Count a NaN on one side against a value on the other as a mismatch (`np.isnan(a) != np.isnan(b)`). Compare all training rows of the month, not `head(1000)`. |
| 13 | `build_prediction.py:32-36` | `REQUIRED_COMPLETE` lists 7 columns, but brief §5 promises all `merra_*` (34) and all fire counts and FRP sums (10). Columns in neither list are never checked: `aod_nbr_*_d0`, `*_n`, `aod055_centroid`, `aod047_centroid`, `aod055_npix`, `maiac_npix_total` and `maiac_covered`. | Build the list from prefixes (`merra_`, `fire_count_`, `fire_frp_sum_`, `province_code`). Fail on any column that's in neither `REQUIRED_COMPLETE` nor `ALLOWED_NAN`. Check every month, not just `files[:3]`. |
| 14 | `verify_features.py:57-61` | The brute-force check uses great-circle distance ≤ 50 km, while the kernel uses 10 lattice cells. EPSG:3347's scale varies by a few percent between 38° and 59°N, so cells near the edge of the disk will differ. That's a false failure at 1e-6. | Define the radius in lattice units and verify it with Euclidean distance on `x_3347`/`y_3347` against 10 × 5 km, or rename the features to say "10 cells". |
| 15 | `verify_maiac.py:39-62` | `recompute` returns at the first tile that overlaps the cell box, so cells on a tile seam disagree. It also doesn't apply the all-zero empty-day rule, so empty days disagree too. | Sum values and counts across all tiles, then divide. Apply the same five-bands-all-zero test. |
| 16 | `verify_fire.py:23-37` | The loader copies the builder's type filter, dedupe and UTC-6 shift, so the check isn't independent for how detections are assigned to days. It only tests the geometry. | Acceptable, but add one hand-checked detection near a day boundary (e.g. 05:59 and 06:01 UTC) as a fixed test case. |

## 4. Checked and correct

These brief §8 assumptions hold against the raw files:

- **MERRA lookup:** `merra_pix.npy` uses only columns 0–65 and rows 0–43, so it never touches the all-NaN column 68.
- **MAIAC bands:** descriptions are `YYYYMMDD_{AOD055, AOD047, INJECTION_HEIGHT, SMOKE_AOD055, SMOKE_MASK}`, day-major, in float32. As `maiac_tiles.json` says, 4 tiles reach the square.
- **MAIAC empty-day rule is safe:** a real-day tile (2015-07-15) is 69.0% NaN, 30.9% above 0 and 0.000% exactly 0, so an all-zero tile can only be an empty day.
- **HMS fields:** `Start` and `End` are strings like `2015182 1100` (UTC), and `Density` is Light/Medium/Heavy in both 2010 and 2015. The CRS is EPSG:4326.
- **Fire CSV:** `acq_date` is a string, `acq_time` is int64 (HHMM), `type` is int64 and `frp` is float64. `BallTree(metric="haversine")` is given radians, as it requires.
- **int16 fire counts can't overflow:** the 2023 peak was 13,319 type-0 detections in one day across Canada and the US.

## 5. UTC-6: measured, not assumed

The `merra2_utc6_ee` archive behaves like a [D 06:00, D+1 06:00) UTC window. Replace the brief's §4.2 evidence (ratio 0.973, correlation 0.951) with the fit below. Those two numbers can't tell a shifted day from an unshifted one.

**Method:** a least-squares fit of each UTC-6 daily value as a weighted mix of the UTC-day means for D−1, D and D+1.
- The UTC-day means come from the older export (`SourceRegion_MERRA_2010_2024-20261103T023716Z-1-001.zip`).
- The fit uses all pixels from 2015-07-02 to 2015-07-29.
- The two grids are offset by 2 columns and 2 rows.
- A true UTC-6 window predicts weights of about 0, 0.75 and 0.25.

| Variable | D−1 | D | D+1 | Residual / SD |
|---|---|---|---|---|
| PBLH | −0.06 | 0.72 | 0.33 | 0.38 |
| TLML | −0.11 | 0.88 | 0.24 | 0.09 |
| ULML | −0.11 | 0.88 | 0.25 | 0.19 |
| U10M | −0.11 | 0.90 | 0.21 | 0.18 |
| V2M | −0.11 | 0.91 | 0.21 | 0.22 |

- The D+1 weight is positive for every variable, and D−1 is about zero.
- A UTC-day window would put almost no weight on D+1, and a shift in the wrong direction would weight D−1 instead.
- The weights aren't exactly 0.75 and 0.25 because consecutive days are correlated.
- Absolute proof would need one hourly granule; this is enough to proceed. Put this fit and table in the build's QA report.

## 6. Decisions and fix order

Ask the user to settle these before building past the base panel:

- **AOD gap fill.** Brief §4.3 says there is none, but `build_features.py` has `fit` and `apply` steps whose output nothing reads. Either remove them, or have `build_prediction.py` read `filled_*` and add `aod055_filled` to the contract.
- **South of 41°N.** Without a fill, 26.9% of training cell-days have no AOD at all. The model can only tell those apart from cloud gaps through `maiac_covered`, which is effectively a static latitude flag and at odds with "no static features".
- **Fire speed (no decision needed, but worth doing).** The code runs a separate radius search for every fire and every radius: 5 searches × up to ~13,000 fires a day. Instead, query once at 500 km with distances and bin the results, and first drop fires more than 500 km from the square.

Fix and re-run in this order:

- [ ] #5: the `cell_id` key in `build_base.py`
- [ ] #1–#4: the four crashes
- [ ] #6–#8: HMS values
- [ ] #11: fire on 31 December
- [ ] #12 and #15: verifier holes, fixed before trusting any PASS
- [ ] Run `build_base.py --cells station --years 2015 2015`, then the four extractor verifiers on 2015-07. All must exit 0.
- [ ] #9, #10, #13 and #14: features, table and contract; then run `--cells all`

Report back with each verifier's output and each run's `failures.json`.
