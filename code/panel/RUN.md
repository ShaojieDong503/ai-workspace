# Run order for the square-domain build

Run from `D:\ai-workspace\project`, `$env:PYTHONIOENCODING='utf-8'`,
Python `C:\Python314\python.exe`. Every stage is per month and resumable;
every output records the fingerprints of its inputs and rebuilds itself when
those change, so the order below matters less than it used to - but a cold
month still has to be revisited after its predecessor exists.

## 1. Base panel, all months

```
python code/panel/build_base.py --cells all
```

Nothing downstream can start until this is complete. A month that fails is
logged with a traceback and skipped; re-run the same command to retry it.

## 2. Features, twice

```
python code/panel/build_features.py context --cells all
python code/panel/build_features.py context --cells all
```

The **second pass is required**. A month whose predecessor's base does not exist
yet is built cold and flagged `warm_start=false`. Once base is complete the
second pass rebuilds exactly those months - typically one January per year, and
2010-01 never. The first pass over a complete base is enough only if base was
finished before features started.

## 3. Prediction

```
python code/panel/build_prediction.py
```

Rebuilds any month whose base or features fingerprint changed, so months made
warm in step 2 are picked up automatically.

## 4. Tables and verification

```
python code/panel/build_target_table.py
python code/panel/verify_base.py    --cells all --report data/interim/base/qa_report.md
python code/panel/verify_features.py --cells all --year 2015 --month 7
python code/panel/verify_prediction.py
```

`build_target_table` refuses to write a partial table unless `--allow-partial`
is given, and reports how many station-days a partial build would lose.

## What the fingerprints cover

| output | recorded inputs |
|---|---|
| `features_YYYY_MM` | `base_YYYY_MM`, `base_(YYYY-1)_MM` (its warm-up) |
| `prediction_YYYY_MM` | `base_YYYY_MM`, `features_YYYY_MM` |

A month is rebuilt when its output is missing, when it has no recorded inputs
(files written before this rule existed), when any recorded fingerprint differs
from the file on disk, or when a recorded null predecessor now exists. A month
whose predecessor is still missing is left alone, so nothing loops.

The one thing this does not catch: a base month rebuilt with *identical bytes*
keeps its fingerprint, which is correct - nothing downstream needs to change.

## Order that wastes the least time

Build base for **all years** first, then features once. The two-pass form in
step 2 costs a second full pass; the single-pass form costs rebuilding one cold
month per year. Both converge to the same files.