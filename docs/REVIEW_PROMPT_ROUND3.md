# Round 3 — re-audit after the fixes

For the reviewer who produced `audit.py`, `audit_train.py`, `warmup.py`.
Your tools are unchanged and still at `data/interim/scratch/audit_review/`.
My run of them is in `audit_after.out` in the same folder.

## What changed

| finding | fix |
|---|---|
| V1 June built without warm-up | `features_2015_06` deleted and rebuilt with May; `prediction_2015_06` and the training table rebuilt |
| V2 the same trap in the full run | `cmd_context` now warns on a cold start, writes `warm_start=true|false` into the parquet metadata, and rebuilds a `false` month on resume once its predecessor exists |
| V3 six cells with province `(water)` | `province.py` resolves water cells to the province of the nearest land cell, one rule for both tables |
| S1 training `date` was `timestamp[ms]` | cast to `datetime64[ns]` |

Because the province definition changed, **all 12 prediction months were rebuilt**, not just June. Base and features were rebuilt only for June (warm-up) and January (to carry the new flag).

## What I would check first

**The warm-start flag and its rebuild rule.** This is new logic and the only place I could have introduced a loop or a silent skip.

- `features_2015_01` should carry `warm_start=b'false'` and June onward `b'true'`. Read the metadata, don't infer it.
- January is cold *on purpose* — December 2014 does not exist yet. It must rebuild automatically once that month appears. Simulate it: touch a dummy `base_all_2014_12.parquet`, run `context --cells all`, confirm January rebuilds and then carries `true`; delete the dummy afterwards. **I have not run that end-to-end** — I only confirmed the predicate in isolation.
- Confirm the rule cannot loop: a month with no predecessor must stay `false` and not be rebuilt every pass.
- Note what the flag does **not** cover: if a base month is ever rebuilt with different values, its features are stale and nothing detects it. There is no base checksum in the feature file. Say whether that matters for the full run.

**The province change.**
- `province.cell_to_code` should now return a land code for all 163,787 cells; `(water)` should appear nowhere.
- The audit's `province_state != spine: 2,037` line in `audit_train.py` is now **expected**: the spine keeps the physical null, the tables carry the nearest land province. Compare against `province.cell_to_code`, not the spine.
- Spot-check the six cells by hand: four Montreal cells → Québec, 989118 → Pennsylvania, 994439 → Delaware. I am least confident about 994439 (a Philadelphia station resolving to Delaware); tell me if you think the station's own province is the better rule for training rows.
- Verify that the 12 rebuilt prediction months changed *only* in `province_code`: row counts, sort order, dtypes and every other column should be identical to what you audited.

**The June rebuild.**
- `warmup.py` should now print no differing cell-days for June, July or August. If June still differs, the rebuild did not take.
- Confirm `prediction_2015_06` agrees with `features_2015_06` and with the training table for June — the cross-table check covers it, but June specifically is worth a direct look.

## What not to re-report

These are correct and now documented in the brief:

- AOD NaN share is seasonal (0.61–0.73 May–Oct, 0.81–0.99 Nov–Apr), not 0.60–0.75.
- 154,227 of 163,787 cells are covered; 184 cells just south of 41°N are covered via the tile edge, 132 north-edge cells are not.
- `centroid` finite ⇒ `aod` finite, not the reverse.
- `count>0 & frp==0` is real (detections with `frp = 0`).
- Water cells resolving to a land province is the intended fix, so a spine mismatch is expected.
- Column *order* differs from the original brief's listing but is identical across every month and stage.

## What has not been touched

Base for any month except none — base was not rebuilt at all in this round; only features June/January and the prediction months were. If you find a base defect it is the same one you already audited.

## How I would like the answer

Same shape as last time, and specifically:

1. Did the warm-start rebuild behave under the dummy-predecessor test?
2. Is the province rule defensible, and what should the Philadelphia cell be?
3. Did any of the 12 rebuilt prediction months change in anything other than `province_code`?
4. Anything the flag does not cover that could bite during the 180-month run.

Run only against `data/interim/` and `data/processed/`; write nothing outside `data/interim/scratch/`.
