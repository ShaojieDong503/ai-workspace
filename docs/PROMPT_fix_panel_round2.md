# Task: fix the round-2 review findings in `code/panel`

You are fixing the cell-day panel build in `D:\ai-workspace\project\code\panel`. A reviewer found two bugs that block the full run, four QA checks that can pass on broken output, and some smaller issues. The full review, with evidence, is in `docs/REVIEW_panel_build_round2.md`. Read it first. This prompt lists what to change and how each fix will be judged.

## Rules

- Python is `C:\Python314\python.exe`.
- **Do not write to `data/raw/` or `data/processed/`.** You may write to `data/interim/`. Send table outputs (`build_target_table`, `build_prediction`) to a scratch folder, not `data/processed`.
- Never modify, move or delete raw data.
- Don't start the full all-cell run. Stop after the checks below and report.
- Match the existing code style: small functions, few comments, the same CLI patterns.
- Report only what you actually ran and saw. If a check fails, paste the output; don't paper over it.

## 1. Blockers (fix first)

**C1. `build_prediction.py`: every month crashes.**
Only `out` is created (line 80), never `out/YYYY`, so each write at lines 89/110 raises `OSError: Cannot save file into a non-existent directory`.
- Fix: call `dst.parent.mkdir(parents=True, exist_ok=True)` before writing each month.
- Done when: `build_prediction` writes 2015-06 and 2015-07 into an empty output folder with no manual `mkdir`.

**W1. `build_target_table.py:63-72`: writes a partial table and exits 0.**
A month that has targets but no base or features file is printed and skipped. With 178 of 180 months missing, the script wrote 9,568 rows and returned 0.
- Fix: if any month that has targets lacks base or features, print the months and the number of station-days lost, then exit non-zero. Add an `--allow-partial` flag that permits the partial write (still printing the loss).
- Done when: running it now, with only 2015-06/07 built, exits non-zero without `--allow-partial` and writes the 2-month table with it.

## 2. QA fixes (before anyone trusts a PASS)

**Q1. `verify_features.py:59-61`: false FAIL from int16 overflow.**
`cells.parquet` stores `row`/`col` as int16, so `dr*dr` overflows once an offset passes 181.
- Fix: cast to int64 before subtracting and squaring.
- Done when: it passes on 2015-07 station cells. The reviewer saw 0 of 351 mismatches after the cast.

**Q2. `verify_features.py:65-76`: the backward-only test checks almost nothing.**
`v[idx]` compares one row, not a day, and the test never crosses a month boundary.
- Fix: compare every row where `date == day`.
- Add a warm-up continuity test: context features for month M built through the normal warm-up path must equal those from one continuous run over M−1 and M, for all context columns. The reviewer's version gave 10,881/10,881 equal rows for July 2015.

**Q3. `verify_prediction.py:77`: a dropped column passes.**
`shared` is the intersection of the two tables' columns, so a column missing on either side is never compared.
- Fix: declare the model-input column list once (the 63 inputs), in a place both builders and the verifier import. Assert that every prediction month and the training table carry exactly that set; extra non-input columns such as the target or QA flags may be allowed explicitly.

**Q4. `verify_prediction.py:79-82`: the row-existence check can't fail.**
The merge is a left join, so `len(mrg) == len(tm)` is always true.
- Fix: use `how="inner"` or `indicator=True`, and fail on any training row with no prediction match.

**Mutation tests (required evidence for Q3/Q4).** On copies in scratch, never on real outputs:
1. Drop `merra_pblh` from a prediction month → `verify_prediction` must FAIL.
2. Drop `aod_nbr_50km_d0` from the training table → must FAIL.
3. Remove 50 station cells' rows from a prediction month → must FAIL on the row check itself, not only on the NaN comparison.

## 3. Cheap correctness fixes

**W2. `build_features.py:76-79`: positional fill vs. id-based read-back.**
The grid is filled with `v[rr*w+cc] = sub["aod055"]`, which assumes each day's rows are in `present` order. The re-expansion (lines 128-132) maps by `cell_id`. Shuffled input gave 5,213 mismatches. Harmless today only because base files are sorted.
- Fix: fill through the same `cell_id → (row, col)` map used for the re-expansion.
- Done when: features from a row-shuffled copy of the July base equal the unshuffled ones exactly.

**W3. `build_prediction.py:96`: a short features file passes.**
Features are left-joined, and all context columns are in `ALLOWED_NAN`, so missing feature rows look like normal gaps.
- Fix: before merging, assert that base and features have the same length and identical `(cell_id, date)` keys.

**After W2:** delete the 2015-06 and 2015-07 files in `data/interim/features/station/` and `data/interim/features/all/`, then rebuild them. The resume logic skips existing files, so stale ones would survive otherwise. The base files under `data/interim/base/` stay valid.

## 4. Lower priority (do if time allows, otherwise list as open)

- **Q5, `verify_maiac.py`:** it checks only `aod055`/`aod047` on 50 pairs. Extend it to `injection_height`, `smoke_aod055`, `smoke_mask_frac`, `aod055_npix`, `maiac_npix_total`, both centroid columns and `maiac_covered`. Find a month that actually has empty (all-zero) tile-days and run it there. 2010-01, 2015-07 and 2023-06 have none.
- **Q6, `verify_fire.py`:** add fixed cases for a 05:59 UTC detection (previous UTC-6 day) and a 1 Jan 00:00–06:00 UTC detection landing on 31 Dec of the year before.
- **W4, `extract_fire.py`:** there is no `modis_2025_*.csv`, so 31 Dec 2024 misses 1 Jan 2025 00–06 UTC detections, and `_load_year` skips the missing file silently. Make it warn when the `year + 1` file is missing. Don't fetch data; flag it for the user.

## 5. Final check and report

Run, on 2015-06/07:
1. `build_features context` for `--cells station` and `--cells all` (after deleting the old feature files).
2. All verifiers: `verify_merra`, `verify_maiac`, `verify_hms`, `verify_fire`, `verify_base`, `verify_features`, then `build_target_table` and `build_prediction` (into scratch), then `verify_prediction`.
3. The three mutation tests.

Report back with:
- a list of files changed and a one-line description of each change;
- each verifier's output and its exit code;
- `failures.json` from every stage;
- the mutation test results, each showing FAIL;
- the W1 run with and without `--allow-partial`, with exit codes;
- anything you couldn't fix or decided differently, and why.
