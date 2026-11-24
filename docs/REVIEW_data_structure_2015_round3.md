# Data structure audit, round 3: after the fixes

2026-11-05 · re-audit of the 2015 build

**Verdict:** the fixes worked. The warm-start rule behaves correctly in the dummy-December test and can't loop. June now matches a warm-up recompute exactly. The 12 rebuilt prediction months changed only in `province_code`.

Two things remain before the 180-month run:
- **R1:** a features rebuild never reaches the prediction table. This blocks the run.
- **R2:** the Philadelphia cell resolves to the wrong state.

Scripts are in `data/interim/scratch/audit_review/`: `sim_warm.py`, `province_check.py`, `jan_cold.py`, plus the earlier ones. The sandbox from the dummy test was deleted afterwards. Nothing was written outside `scratch/`.

## 1. Warm-start flag

**The metadata, read from the files:**
- `features_2015_01` is `false`, and `features_2015_06` is `true`.
- The other ten months have **no `warm_start` key**. They were written by the old code. The brief said "June onward `b'true'`"; that isn't what is on disk. It's harmless: `_needs_rebuild` treats a missing key as "don't rebuild", and those ten months were verified as correctly warm in round 2.

**January today** equals an in-memory cold recompute exactly (0 differing cell-days). The rebuild changed only the metadata.

**Dummy-predecessor test.** I ran the real `cmd_context` with `BASE`/`OUT` pointed at a sandbox under scratch. The sandbox held copies of base 2015-01, -02 and -04 (no -03), and features 01 (`false`) and 02 (no key).

| Pass | Setup | What happened |
|---|---|---|
| A | no Dec 2014 | Jan and Feb skipped; Apr built cold with a warning, flag `false` |
| B | unchanged | Nothing rebuilt; Apr stays `false` with the same mtime. **No loop.** |
| C | dummy `base_all_2014_12` added (Dec 2015 AOD shifted back a year) | "2015-01 was built without warm-up; rebuilding". Jan becomes `true`. Dec 2014 itself is built cold (`false`), and Apr is untouched. |
| D | unchanged | Nothing rebuilt |

After pass C, January had changed in 39,491 cell-days of `aod_last_obs` (days 1–7) and 73,666 of `aod_nbr_50km_d1_3` (days 1–3). On 1 January the maximum age is now 7. The rule works end to end.

### What the flag does not cover

**R1. A features rebuild doesn't reach prediction. This blocks the full run.**
- `build_prediction.py:125` skips any month whose file exists.
- When January's features flip to `true`, `prediction_YYYY_01` keeps the cold values. The training table is rebuilt from `features/all` on every run, so it gets the warm values.
- The two tables then disagree. `verify_prediction` would catch it, but only after the fact.
- In the full run this hits every month whose features were rebuilt, including 2015-01 once 2014 exists.
- **Fix:** copy `warm_start` (and the R3 fingerprint) from the features file into the prediction file's metadata. Rebuild prediction when it differs from the current features file. A simpler alternative: rebuild when an input's mtime is newer than the output.

**R3. No base fingerprint.** If a base month is rebuilt, for example after an extractor fix, its features and prediction go stale silently. That includes the next month's features, which read it for warm-up.
- Does it matter for the full run? Only if any base month is rebuilt after its features exist. That's likely, because extractor fixes during a 6-hour run are how earlier rounds went.
- **Fix, applied to features and prediction:** record each input's identity, its size plus mtime or a short hash of `aod055`. Rebuild when an input changes. For features, the inputs are base month M and base month M−1.

**R4. Run order still matters for the rebuild step.** The rule only acts when `context` runs again after the predecessor month's base exists. So the full run needs a final `context --cells all` pass after all base is built, followed by prediction once R1 is fixed. With R1 and R3 fixed, that final pass is enough.

Minor: `2010-01` is correctly `false` forever and never rebuilt, and the code suppresses the warning for it. The ten unflagged 2015 months aren't a risk, because they're correct.

## 2. Province rule

**Coverage:**
- `cell_to_code` returns a land code for all 163,787 cells. The "(water)" code is used by no cell, and no table row carries "(water)".
- The 16,269 spine water cells resolve to: Nunavut 6,035, Ontario 5,751, Manitoba 2,391, Québec 1,289, New Jersey 586, New York 89, Delaware 59, Maryland 54.
- All 12 prediction months and all 57,441 training rows equal `cell_to_code`. `province_state` matches the lookup for every training row.
- The audit's 2,037 "!= spine" rows are exactly the six water cells, as expected.

**The six cells, against `stations.parquet`:**

| cell | Resolves to | Station | Station's own province |
|---|---|---|---|
| 848529, 850659, 853851 | Québec | NAPS 050103, 050131, 050128 (Montréal) | QC ✓ |
| 851721 | Québec | NAPS 050113 (Laval) | QC ✓ |
| 989118 | Pennsylvania | AQS 42-101-0057 (Philadelphia) | Pennsylvania ✓ |
| 994439 | **Delaware** | AQS **42-045**-0109 | **Pennsylvania** ✗ |

**R2. The Philadelphia cell is wrong.** In AQS 42-045, 42 is Pennsylvania and 045 is Delaware County, PA (Chester), so the station is in Pennsylvania. Delaware is simply the nearest land on the lattice from that river cell.

**Is the rule defensible?** For unmonitored water cells, yes: nearest land is a sensible geographic label and it's the same in both tables. For monitored cells, `stations.parquet` shows the cell-centroid province differs from the station's own province more often than the six water cells suggest:
- 14 land cells disagree. These are border rivers: Detroit/Windsor, St. Croix (Minnesota vs Wisconsin), Delaware River (PA vs NJ), Ohio River (Ohio vs West Virginia), Ottawa River.
- 2 cells mix provinces: 962346 has Windsor ON and three Detroit MI stations; 1001801 has an Ohio and a West Virginia station.

So "the station's own province" isn't a cell property in general, and it can't be one rule for both tables.

**Recommendation:**
- Keep `province_code` as the cell's geographic province, one rule for both tables, as now.
- With one change: a water cell that contains a station takes the station's province, and nearest land applies to the rest. That moves only 994439, to Pennsylvania.
- If province is used for anything jurisdictional, such as per-province scores or network comparisons, add a training-only `station_province` column from `stations.parquet`, with "mixed" for the two mixed cells. Keep it out of the model inputs.

Note: `stations.parquet` stores Canadian provinces as abbreviations (`QC`, `ON`) and US states as full names. Map them before any comparison.

## 3. The 12 rebuilt prediction months

Only `province_code` changed. The evidence:
- **Inputs:** base was not rebuilt; mtimes are from 11-04 18:58 to 11-05 12:33, before this round. Features changed only in 01, which equals the cold recompute (so its values are unchanged), and 06, the intended fix.
- **Your `audit_after.out` against my round-2 `audit.out`:**
  - No new structural lines: same columns, order, dtypes, keys, sort and row counts.
  - No "pred != base/features" in any month, so every data column equals base and features exactly.
  - Every NaN-placement and QA count is unchanged.
  - The new lines are "province_code != spine", at 504,339 / 455,532 / 488,070 cell-days. That's 31 / 28 / 30 days × 16,269, exactly the water cells and nothing else.
  - June day-1 statistics are now warm (max age 7, `d1_3` 15.8% NaN). Sizes moved by under 0.1 MB, except June features (+1.2 MB, from the real value change).
- **`province_check.py`:** `province_code` equals `cell_to_code` in every cell-day of all 12 months.

## 4. June

- `warmup.py` reports **0 differing cell-days** for June, July and August.
- `prediction_2015_06` equals `features_2015_06` and `base_all_2015_06` in every column of every row (`audit_after.out`).
- All 57,441 training rows equal prediction in 69 columns, June included, with 0 mismatches.
- The training table has 1,084 June 1–7 rows; they now carry warm values.

## 5. Training table

`date` is now `datetime64[ns]`. Otherwise it is unchanged from round 2:
- 79 columns, 57,441 rows, 171 cells, sorted and unique;
- `pm25` from −6.82 to 80.0, with 174 negatives and no −999;
- exact match to `station_day`;
- no "(water)" rows.

## Fix order

- [ ] R1: make prediction rebuild when its features change (blocks the full run)
- [ ] R2: give water cells that contain a station the station's province (994439 → Pennsylvania). Optionally add a training-only `station_province`.
- [ ] R3: record base-file identity in features and prediction metadata, and rebuild on change
- [ ] R4: in the full-run procedure, run all base, then a final `context --cells all` pass, then prediction
