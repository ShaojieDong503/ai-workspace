# Task: input tracking for rebuilds, and the Philadelphia province, in `code/panel`

You're fixing the last items from the round-3 data audit before the full 2010–2024 run. Read `docs/REVIEW_data_structure_2015_round3.md` first; it has the evidence for each item below.

## Rules

- Python is `C:\Python314\python.exe`.
- Never write to `data/raw/`. You may rebuild the **2015** outputs in their usual places: `data/interim/features/all`, `data/processed/prediction/2015`, and the training table in `data/interim/scratch/target2015`.
- Run every simulation in a sandbox under `data/interim/scratch/`, never in the real `base/` or `features/` folders. `data/interim/scratch/audit_review/sim_warm.py` shows how: it points `build_features.BASE`/`OUT` at a sandbox and calls `cmd_context` directly. Delete the sandbox when you're done.
- Don't start the full run.
- Match the existing code style. Report only what you ran and saw.

## 1. Record each file's inputs and rebuild when they change (R1 + R3, blocks the full run)

**Problem:**
- `build_prediction.py:125` skips any month whose file exists. So when a month's features are rebuilt (for example January turning warm once the previous December exists), its prediction keeps the old values. The training table is rebuilt every run, so the two tables end up disagreeing.
- Nothing records which base file a features file was built from. If a base month is rebuilt mid-run, its features and prediction stay stale without any warning. So do the next month's features, which use it for warm-up.

**Change:** make every output record the identity of its inputs, and rebuild it whenever any of those inputs differs or is missing.

- **Fingerprint:** a content hash of the input file, such as SHA-1 or BLAKE2 of the bytes. Not the modification time, which changes on copy and can be the same for different contents. Put the hashing in `utils.py`.
- **Features file metadata:** the fingerprint of base M, and of base M−1 or `null` when it's missing. Keep `warm_start` as a readable flag, equal to "M−1's fingerprint is not null".
- **Prediction file metadata:** the fingerprints of base M and features M.
- **Rebuild rule, for both stages:** rebuild when
  - the output doesn't exist; or
  - it has no recorded inputs (a legacy file); or
  - any recorded fingerprint differs from the file currently on disk; or
  - a recorded `null` predecessor now exists.

  This replaces `_needs_rebuild`. A month whose predecessor is still missing must not be rebuilt on every pass, which is what the old rule got right.
- **Logging:** log why each rebuild happened ("features 2015-01: base 2014-12 now exists", "prediction 2015-03: features changed").
- **`verify_prediction.py`:** for every month, fail if the prediction file's recorded features fingerprint differs from the current features file's fingerprint.

Because legacy files carry no recorded inputs, the next run rebuilds all 12 months of 2015 features and prediction. That's intended.

### Sandbox test (required evidence)

Copy base 2015-01, -02, -04 and -05 into a sandbox, leaving out -03. Point both `build_features` and `build_prediction` at it. Run `context --cells all` and then prediction after each step:

| Step | Setup | Expected |
|---|---|---|
| A | first build | all 4 months built with recorded inputs; 01 and 04 have `warm_start=false` |
| B | rerun, nothing changed | **nothing rebuilt**, by mtime and by log |
| C | add a dummy `base_all_2014_12` (Dec 2015 shifted back a year; `sim_warm.py` has the code) | features 01 rebuilt with `warm_start=true` **and** prediction 01 rebuilt; 02, 04 and 05 untouched |
| D | rerun | nothing rebuilt |
| E | rewrite base 04 with one `aod055` value changed | features 04 **and 05** rebuilt (05 uses 04 for warm-up), prediction 04 and 05 rebuilt; 01 and 02 untouched |
| F | rerun | nothing rebuilt; 04 still `warm_start=false` because 03 is still missing, with no loop |

Then copy a prediction file, edit its recorded features fingerprint, and show that `verify_prediction` FAILs on that month.

## 2. Province for water cells that contain a station (R2)

**Problem:** cell 994439 resolves to Delaware by nearest land, but its station, AQS 42-045-0109, is in Delaware County, **Pennsylvania**.

**Change, in `province.py`:**
- A water cell that contains a station from `data/interim/stations/stations.parquet` takes that station's province.
- Every other water cell keeps the nearest-land rule.
- `stations.parquet` writes Canadian provinces as abbreviations (`QC`, `ON`, `MB`, …) and US states in full. Map them to the lookup's names (`Québec`, `Ontario`, …) and fail on any name that isn't in the lookup.
- If a water cell held stations from two provinces, fail loudly. None exist today.
- Land cells stay as they are, even where the station's province differs (14 border-river cases). That's the cell's geographic province, and it is the intended rule.

**Expected result, relative to the current files:**
- Only cell 994439 changes, from Delaware to Pennsylvania.
- In prediction that's one cell per day in every month. In the training table it's 313 rows.
- The other five station water cells stay as they are (four Québec, 989118 Pennsylvania).

## 3. Full-run order (R4)

Write down the run order where the run commands are documented. If no run document exists, add `code/panel/RUN.md`:

1. `build_base --cells all`, all years.
2. `build_features context --cells all`. After base is complete, run it **once more**, so months that were built cold get rebuilt.
3. `build_prediction`. Thanks to item 1, it rebuilds whatever changed.
4. `build_target_table`, then every verifier.

## 4. Rebuild 2015 and check it

1. Rebuild all 12 months of 2015 features, then prediction, then the training table.
2. Check that features values are **unchanged** from the current files: 0 differing cell-days in every month. January stays cold (`warm_start=false`, because December 2014 doesn't exist), and the other 11 months are `true`. Keep copies of the current files in scratch first, so you have something to compare against.
3. Check that prediction is unchanged apart from `province_code` on cell 994439.
4. Run these scripts in `data/interim/scratch/audit_review/` and report their output:
   - `audit.py`: the province line will differ from the spine, as expected;
   - `audit_train.py`;
   - `province_check.py`: expect 994439 → Pennsylvania;
   - `warmup.py`: expect 0 differences for June, July and August.

   Also run `verify_prediction`.

## Report back

- Files changed, with one line each.
- The sandbox table: steps A–F with what was rebuilt at each step (log lines and mtimes), and the `verify_prediction` FAIL on the edited fingerprint.
- `warm_start` and the recorded inputs for all 12 months of 2015 features and prediction, read from the metadata.
- Features differences against the pre-rebuild copies (expect 0), and the prediction difference (only 994439's `province_code`).
- The output of the four audit scripts and `verify_prediction`, with exit codes.
- Anything you couldn't do or did differently, and why.
