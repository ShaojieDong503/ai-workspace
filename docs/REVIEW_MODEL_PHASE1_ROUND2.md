# Phase 1 re-review: sites, overlap-aware correlation, fold plan

Second review of `code/model` Phase 1, after the fixes for F1–F6 in
`docs/REVIEW_MODEL_PHASE1_findings.md`. Nothing has been fitted or scored. The
splits were rebuilt from row counts and cell coordinates only — no target-derived
quantity entered the split decisions.

Authority: `docs/MODEL_RUN_PLAN.md` (18 Nov). Code: `code/model/`. Outputs:
`data/interim/model/`.

## What changed since the first review

| finding | change |
|---|---|
| **F1 (blocker)** | Splits and folds now operate on **sites**: single-linkage clustering of cell centres at 10 km, `folds.build_sites`. Roles and folds are assigned per site; a site may hold only one role and one fold. No site straddles a region (asserted). |
| **F2** | Correlation is no longer trusted on pairwise-complete rows. A pair is merged only when neither column is present where the other is missing on more than `max_partial_overlap` (0.10) of rows. Representatives are left blank in the draft, and candidates are ranked by gain then completeness as a suggestion. |
| **F3** | `folds.plan(scheme)` is the single source of fold membership; `leakage_test.py` imports it. Added per-fold disjointness on cells, years and sites; breakages now go through `fold_rows` / `scoring_rows`. `data.fold_rows` rejects a held-out list containing non-development cells or years outside 2010–2023. |
| **F4** | `min_site_rows = 2000` for any site chosen as test or calibration. The North test site went from 1,095 to 4,942 rows because its neighbour is now in the same site. |
| **F5** | Redundant sort removed; regions cast to `int`; fold balance re-checked on sites. |
| **F6** | Report wording corrected: no "all 10/10" claim, and an explicit note that gain is split among correlated columns. |

## Current numbers

```
48 cells -> 37 sites (largest 3 cells), no site straddles a region
test 8 sites / 12 cells   calibration 6 sites / 8 cells   development 23 sites / 28 cells
sanity: rows 197612, gt25 1318, gt50 134, north 25005
development rows 121,949
nearest cross-role cell distance: 11.2 km
folds: 10,043 - 15,067 rows (1.50x), 2-4 cells, 1-3 regions
feature reduction: 62 columns, 0 constant, 15 groups, 19 pairs rejected for overlap
config hash f9f18c88327486bd
```

## What I would check first

**1. The site rule itself.** `build_sites` clusters only the cells being studied —
clustering the full 163,787-cell lattice merges the province, because 5 km
neighbours chain. Confirm that choice is right and that the 10 km threshold is
defensible. At 15 km the picture changes sharply: 30 sites, largest 12 cells, 2
sites straddling regions. Is 10 km the right side of that cliff, or should the
threshold be justified another way?

**2. Cell targets versus site targets.** `n_test_cells = 10` now yields 12 cells
from 8 sites; `n_calib_cells = 6` yields 8 cells from 6 sites; development falls
to 28 cells against the plan's ~32. A site can carry 2–3 cells, so the cell count
cannot be hit exactly. Options: accept the overshoot, count sites in the config
instead, or allow partially taking a site (which would defeat the purpose). This
needs a decision.

**3. The overlap guard.** `max_partial_overlap = 0.10` is my number, not the
plan's. It removes `aod055`/`aod_last_obs` and `aod055`/`smoke_aod055` from the
groups, which is what the first review asked for, but the threshold itself is a
judgement. Check the rejected-pair table in
`feature_reduction_report.md` for pairs that should still have been merged, and
for pairs that survived that should not have.

**4. Whether the remaining grouping is meaningful.** `aod_nbr_25km_d0` and
`aod_nbr_50km_d0` now appear as a group in only 5/10 folds. Is a group that is
present in half the folds a real redundancy or a boundary case of the threshold?

**5. The engine is still only bound by sharing a module.** `cv.py` does not exist
yet, so the fold plan is common code but the *engine* has not been shown to use
it. The `--dry-run` proposed in F3 is Phase 2 work. Until then the leakage test
constrains `folds.plan` and `data.fold_rows`, not the engine.

**6. `data.fold_rows` now raises on a held-out list containing non-development
cells.** That is stricter than the plan states. It catches caller errors, but it
also means a future caller cannot pass a superset of the fold's cells. Check
whether that is the behaviour you want.

## What not to re-report

- The sanity counts, the row filters and the coverage check were accepted in
  round 1 and are unchanged in behaviour.
- `maiac_covered` being excluded (constant over Ontario) and dataset A holding 62
  rather than 63 columns.
- The "Michigan"-labelled test cell 944250 — it is an Ontario monitor, and it is
  now grouped with 944251 and 946380 in one site, which is the point of F1.
- The absence of Phases 2 and 3.

## What to run

```
python code/model/data.py
python code/model/splits.py --force
python code/model/leakage_test.py
python code/model/feature_reduction.py
```

`--force` is required because the splits exist and are frozen; rebuilding them is
the whole point of this round and is safe while nothing has been fitted.

## What not to do

Do not freeze `features_frozen.json` and do not start Phase 2. Both wait on this
review.

## Report

Same shape as round 1: blockers, must-fix-before-freezing, minor; each with the
file, the observation and the fix. Where something is fine, one line saying it was
checked. Two open questions I would like answered explicitly:

1. Is the site threshold right at 10 km, and is clustering only study cells the
   correct reading?
2. Should the config count cells or sites, given the 12/8 overshoot?
