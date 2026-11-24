# Panel build review — round 2

2026-11-04 · reviewer findings for the author of `code/panel`

**Verdict: two findings block the full run.**
- `build_prediction.py` can't write any month (crash C1).
- `build_target_table.py` writes a partial training table and exits 0 (W1).

Fix those plus the four QA gaps Q1–Q4, then start the full run. Every extractor verifier passes on the station canary and on all 163,787 cells for 2015-07. The tables match each other on real data.

## What I ran

| Run | Result |
|---|---|
| `build_base --cells station --years 2015 2015 --month 6 7` (July rebuilt) | 2 months written, 0 failures; HMS missing day 2015-06-02 logged |
| `verify_merra/maiac/hms/fire --cells station` 2015-07, `verify_hms` 2015-06 (`--n 400`), `verify_base` | All 0 failures, except `verify_features` (Q1) |
| `build_base --cells all --years 2015 2015 --month 6 7` | 2 months, 0 failures, 2 m 57 s total |
| `build_features context --cells station` and `--cells all`, June + July | 0 failures; all-cell 1 m 05 s for two months |
| `verify_merra/maiac/hms/fire --cells all` 2015-07 | All 0 failures |
| `build_target_table`, `build_prediction`, `verify_prediction`, with outputs redirected to my scratch folder, not `data/processed` | C1 and W1 found. After a manual `mkdir`: contract and identity pass for both months |
| Mutation tests on `verify_prediction` | Q3 and Q4 found |
| Independent probes: MAIAC zero scan (3 months), feature warm-up and row order, fire timing at all cells, HMS repaired geometry types | See the sections below |

Files left in `data/interim/`:
- `base/station/` for 2015-06 and 2015-07;
- `base/all/` for 2015-06 and 2015-07;
- `features/station/` and `features/all/` for both months.

They were built with the current code, so they're valid. Delete them after any fix that changes their values, because the resume logic would otherwise skip them.

## 1. Crashes

| # | Location | Observed | Fix | Blocks |
|---|---|---|---|---|
| C1 | `build_prediction.py:89, 110` | Every month fails with `OSError: Cannot save file into a non-existent directory: '...\prediction\2015'`. Only `out` is created, never `out/YYYY`. | `dst.parent.mkdir(parents=True, exist_ok=True)` before the write. | Yes |

## 2. Wrong values that pass the shape checks

| # | Location | What happens and how I detected it | Fix | Blocks |
|---|---|---|---|---|
| W1 | `build_target_table.py:63-72` | A month missing base or features is printed and skipped, and the script exits 0. **Observed:** with 2 of 180 months built, it reported "months without base or features: 178", wrote a 9,568-row table and returned 0. A half-finished all-cell run would produce a training table that silently lacks those station-days. | Exit non-zero if any month that has targets lacks base or features, unless `--allow-partial` is given, and print the number of lost station-days. | Yes |
| W2 | `build_features.py:76-79` | The grid is filled positionally (`v[rr*w+cc] = sub["aod055"]`), assuming each day's rows follow `present` order. The re-expansion at lines 128-132 maps by `cell_id`, so the two halves disagree when rows are out of order. **Observed:** shuffling July gave 5,213 NaN-pattern mismatches and a max difference of 1.92 in `aod_nbr_25km_d0`. Harmless today, because base files are written sorted (checked: `(date, cell_id)` sorted, day order equals `present` order). A day with a missing row crashes loudly (`ValueError: shape (350,) vs (351,)`). | Fill through the same `cell_id → position` map used for re-expansion. | No, but cheap |
| W3 | `build_prediction.py:96` | Features are left-joined. If a features file is short, the missing rows get NaN only in context columns, all of which are in `ALLOWED_NAN`, so the contract passes. Not observed; found by reading the code. | Assert that the base and features keys are identical, with equal length, before merging. | No, but cheap |
| W4 | `extract_fire.py:66` (data) | 31 December 2024 needs 1 January 2025 00:00–06:00 UTC detections, but `modis_2025_*.csv` doesn't exist. `_load_year` skips missing files silently. Found by listing `data/MODIS_fire`. | Get January 2025 fire data, or flag 2024-12-31. Either way, warn when the `year + 1` file is missing instead of skipping it. | No (one day) |

## 3. Gaps in the QA

| # | Location | Problem and evidence | Fix |
|---|---|---|---|
| Q1 | `verify_features.py:59-61` | **False failure.** `cells.parquet` stores `row`/`col` as int16, so `dr * dr` overflows once an offset exceeds 181 cells. **Observed:** FAIL with 24 matched, 36 NaN mismatches and max diff 0.33. With `astype("int64")` the same brute-force check gives 0 mismatches in 351 cells. The builder is correct. | Cast to int64 before squaring. |
| Q2 | `verify_features.py:65-76` | **The backward-only test is vacuous.** `context_month` now returns arrays per row, not per day, so `v[idx]` compares one scalar: the row at index 10, which is day 1. It also never crosses a month boundary. | Compare all rows where `date == day`. Add the warm-up test I ran: features from the warm-up path must equal a continuous run over months M−1 and M. |
| Q3 | `verify_prediction.py:77` | **A dropped column passes.** `shared` is the intersection of the two tables' columns, so a column missing from either side is simply not compared. **Observed:** a prediction month without `merra_pblh` gave 0 failures ("46 contract columns complete"), and so did a training table without `aod_nbr_50km_d0`. | Declare the 63 model inputs once, and assert that both tables carry exactly that set, every month. |
| Q4 | `verify_prediction.py:79-82` | **The row-existence check can't fail.** The merge is a left join, so `len(mrg) == len(tm)` always. **Observed:** after removing 50 station cells from a prediction month it still printed "4905/4905". The rows were caught only incidentally, through NaN mismatches. | Merge with `how="inner"`, or use `indicator=True`, and count the training rows with no prediction match. |
| Q5 | `verify_maiac.py` | Checks only `aod055`/`aod047`, and only 50 non-NaN pairs per run. Unverified: `injection_height`, `smoke_aod055`, `smoke_mask_frac`, `aod055_npix`, `maiac_npix_total`, both centroid columns and `maiac_covered`. The empty-day rule has never run: 0 empty tile-days in 2010-01, 2015-07 and 2023-06. | Extend the check to the other columns, and run it on a month that actually has empty days. |
| Q6 | `verify_fire.py` boundary case | Tests only 06:04 UTC. It doesn't test 05:59, which must go to the previous day, or the 1 January 00:00–06:00 → 31 December case across files. | Add both as fixed cases. |

## 4. Checked and correct

| Concern from the request | Evidence |
|---|---|
| MAIAC per-chunk empty rule | My own scan of 2010-01, 2015-07 and 2023-06 found 0 fully empty and 0 partly empty tile-days. A real chunk can't be all zero: NaN never equals 0, and exact-zero AOD055 is only 0.01–0.07% of finite pixels. `_empty_tile_days` counts tile-days through chunk 0, which is meaningful as long as empty days are whole-tile (Q5 asks to confirm that on a month with empty days). |
| `npix_tot` across tile seams | It is added per tile with `+=`, never reset. `maiac_covered` and `maiac_npix_total` are in the complete set and have 0 NaN in both months. |
| Centroid boundary and chunk coverage | `c_row < r1` is the right half-open test. `range(0, h, 256)` with `min(r0 + 256, h)` covers the last partial chunk. The `aod055` difference against the independent recompute is at most 1.7e-07 for all cells. |
| Month warm-up | July features equal a continuous June+July run in all 10 context columns, 10,881 of 10,881 rows. Without the warm-up, 290 cells differ in `aod_last_obs` and 343 in `aod_nbr_50km_d1_3`. Keep and drop use the same predicate (`date >= first`), and January reads December of `year − 1`. |
| Station versus all-cell lattice | The training table reads only `features/all`. 69 shared columns are identical between the tables in both months, and target rows (9,568) equal the station cell-days in `station_day.parquet`. |
| HMS half-pixel origin | x: `X0 + c0·DX − DX/2` is the left edge. y: `Y0 + r0·DY − DY/2` is the top edge, because −DY/2 > 0 when DY < 0. `verify_hms` matches exactly on 150 July and 400 June samples, including 12 on missing day 2015-06-02 (NaN on both sides). |
| HMS raw-WKB repair | The 14 repaired files give 964 Polygons and 5 empty geometries, which are skipped and counted (0.5%). No multipolygons, collections or holes. |
| Fire January | A 1 January 00:00–06:00 UTC detection is in year Y's file and lands on 31 December of year Y−1, which December's `year + 1` load picks up. Only 2024 lacks the next file (W4). |
| Fire speed at 163,787 cells | Heaviest near-square day, 2023-06-22 (5,426 detections): nearest distance 1.4 s plus radius sums 4.9 s, with a 2.4 GB peak. The single-query alternative is slower (15.7 s). **No rewrite needed**, but watch memory if months run in parallel. |
| Context cost | 0.5 s per day on the 418×427 lattice. |
| Full-run budget | About 85 s per month for base and 30 s for features, so roughly 4.5 h plus 1.5 h. Base is about 110 MB a month (~20 GB) and features about 100 MB (~18 GB); prediction will be similar. D: has 216 GB free. |

## 5. Open decisions

None of these blocks the build: the table can carry the columns, and the model step decides.

- **`maiac_covered`.** Recommendation: keep it in the table as a QA column and leave it out of the model inputs. Without it, LightGBM can't tell structural missingness (south of 41°N) from cloud gaps. With it, the model gets one static latitude flag. That trade-off belongs to the user, not the build.
- **South of 41°N (26.9% of training).** Acceptable for the build. In cross-validation, report scores separately for `maiac_covered = 0` and `= 1`, so the cost of having no AOD there is visible before deciding whether that strip needs its own treatment.
- **Fire speed.** Acceptable as is (see §4).

## 6. Fix order

- [ ] C1: create `out/YYYY` in `build_prediction.py` (**blocks**)
- [ ] W1: make `build_target_table.py` fail on missing months (**blocks**)
- [ ] Q1, Q2: fix `verify_features.py`, and add the warm-up continuity test
- [ ] Q3, Q4: fixed column set and inner-join row check in `verify_prediction.py`
- [ ] W2, W3: map-based grid fill; key equality before the features merge
- [ ] Delete the 2015-06/07 interim outputs touched by W2 (`features/station`, `features/all`), then start `build_base --cells all`
- [ ] During the run: Q5 on a month with empty days, Q6, and W4 (January 2025 fire data or a flag)

Report back with each verifier's output, `failures.json` from every stage, and the mutation tests for Q3/Q4 showing a FAIL.
