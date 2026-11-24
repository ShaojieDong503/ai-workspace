# Phase 1 re-review (round 2): findings

> **Author decision, 18 Nov: no sites; simple cell-level splits.** See `docs/MODEL_RUN_PLAN.md` §1–2. This replaces round-1 F1 and F4, and R2-1, R2-4, R2-5, R2-6, Q1 and Q2 below. Ignore every mention of sites in this file.
>
> **For the code agent, the split rules are now:**
> - **Spatial test:** one cell chosen at random from each of the 6 regions, 6 cells in all, from a seeded `rng`. No minimum row count and no other constraint.
> - **Temporal test:** all of 2024, every cell.
> - **Calibration:** whole cells, `n_calib_cells` in the config (default 6), chosen at random (seeded) from the cells left over. The exact count is still the author's open decision.
> - **10-fold CV:** development cells shuffled with a seeded `rng` and dealt into 10 folds, whole cells per fold. No region stratification, no row balancing.
> - **Unchanged:** leave-one-region-out and leave-one-year-out.
>
> **Code changes:**
> - **Remove:** `folds.build_sites`, `site_of`, every `site_id` column, the site checks in `leakage_test.py`, `sites.*` and `min_site_rows` from `config.json`, and the size-ordered `pick_sites` and greedy `assign_folds`.
> - **Seeds:** make sure the seeds are actually used. `seeds.split` drives the test and calibration picks; `seeds.fold` drives the fold shuffle. Run once with the chosen seeds; don't re-draw.
> - **Still applies:** R2-2 (NaN correlations), R2-3 (dominance rule), and R2-7 (plan wording, now done).
> - **Keep:** the per-fold disjointness checks on cells and years, and the breakages through `fold_rows` and `scoring_rows`.
> - **Report addition:** in `splits_summary.md`, list each test and calibration cell's row count and its distance to the nearest development cell.

Review of `code/model` after the round-1 fixes, against `docs/MODEL_RUN_PLAN.md` and the brief `docs/REVIEW_MODEL_PHASE1_ROUND2.md`.

**What I did:**
- Read `config.json`, `data.py`, `folds.py`, `splits.py`, `leakage_test.py` and `feature_reduction.py`.
- Read `splits_summary.md` and `feature_reduction_report.md`.
- Ran one read-only check: the distances and active years of every pair of monitored cells 10–25 km apart. It used dates only, no PM₂.₅ values; an earlier version that correlated PM₂.₅ across test cells was stopped, rightly.
- I did not re-run the scripts.

**Verdict:** the F1 fix is right, and no row leaks. One rule must change before the splits are frozen (R2-1), and the feature-reduction code has one bug to fix before the list is frozen (R2-2). The rest is minor.

**Checked and fine:**
- the row filters, now with the stray-list checks
- the per-fold disjointness on cells, years and sites
- the breakages, which now go through `fold_rows` and `scoring_rows`
- the one-role and one-fold per site checks
- the sanity counts: development 121,949 + calibration 29,930 + test 45,733 = 197,612
- the overlap guard on `aod055` / `aod_last_obs` and `aod055` / `smoke_aod055`
- the corrected report wording

## Before freezing the splits

### R2-1. Site selection is deterministic and size-driven; the seed does nothing

> **Superseded by the author decision at the top:** a random cell per region, with no sites and no minimum row count. Only the point that the seed must actually be used still applies.

`splits.py`:
- **`pick_sites` never uses randomness.** Its first pick per region is the **smallest** eligible site (`sort_values(["n_rows", ...], ascending=True).iloc[0]`), and every extra pick is the **largest** remaining site.
- **The seed has no effect.** `rng` is created in `main` and passed only to `assign_folds`, which ignores it. So `seeds.split` and `seeds.fold` change nothing, although the docstring and the summary header say the split is seeded.

This isn't leakage, since the rule never sees the target. But it means the test set is "the smallest qualifying site in each region, plus the biggest leftovers". That's an arbitrary, systematic choice that's hard to defend in a paper. Smallest-eligible sites tend to be short-record or relocated monitors. The East test site that resulted (856987) has 4,294 rows and **1 day above 25 µg/m³**, against 134 in East development.

**Fix:**
- Pick sites by **seeded random choice** among eligible sites (n_rows ≥ `min_site_rows`): one per region first, then the extras spread across regions as now.
- Assert that every region keeps at least `min_dev_sites_per_region` development sites (default 2). Region 1 is at exactly 2 today.
- Pass `rng` into `pick_sites`, and drop the unused `rng` argument from `assign_folds`, or use it to break ties.

**Run it once.** `splits_summary.md` already shows how many days above 25 and 50 each site has. Fix the rule, pick the seed **before** looking, and accept what comes out. Re-drawing until the test set "looks balanced" would be selection on the target.

## Before freezing the feature list

### R2-2. NaN correlations pass the threshold test

`feature_reduction.correlation_groups`: `if corr.loc[c, d] < thr: continue` lets every **NaN** correlation through, because `NaN < thr` is False. A pair has NaN correlation when one column is constant on their co-observed rows, for example `aod055` / `aod_missing` or `hms_any` / `hms_density_max`.
- **Today:** these pairs reach the overlap test and get rejected there. That's why 11 of the 19 rows in the rejected table have `corr = nan`.
- **Latent risk:** a NaN pair with good co-observation would be **merged**.

**Fix:**
- Use `if not (corr.loc[c, d] >= thr): continue`.
- Drop the NaN rows from the rejected table.
- De-duplicate the table by unordered pair. Today `aod_nbr_25km_d0 / aod_nbr_50km_d0` and its reverse both appear, and so do `smoke_aod055 / aod_last_obs` and its reverse.

### R2-3. `aod_nbr_25km_d0` / `aod_nbr_50km_d0` sits exactly on the threshold (brief item 4)

- **Why it flips:** the penalty is 0.100 against `max_partial_overlap = 0.10`, so the pair merges in 5 of 10 folds. It's a threshold artefact, not evidence either way.
- **The real structure is nesting.** The 25 km disk sits inside the 50 km disk, so wherever the 25 km mean exists, the 50 km mean exists (A-only = 0.000). The two correlate at 0.97 where both exist, and the 50 km column adds coverage on 10% of rows.

**Fix:** replace the symmetric penalty with a dominance rule. If one column's presence is a superset of the other's (the other's A-only share is ~0) and |ρ| ≥ threshold, merge into the superset column. Otherwise, apply the penalty as now. This removes the flip.

**For the frozen list:** keep the 50 km and 100 km means; drop `aod_nbr_25km_d0` and `aod_nbr_25km_d0_n`.

### Overlap threshold (brief item 3)

- **The rejections are right.** Every genuine rejection (the non-NaN rows) is correct: each involves a column that fills another's gaps.
- **No wrong merges:** none of the surviving groups mixes columns with different coverage.
- **The 0.10 value** only matters for the nested pair in R2-3. With the dominance rule it stops mattering.

## Minor

- **R2-4. Two independent sources for sites.**
  - `leakage_test.py` takes role and fold membership from the `site_id` stored in `splits.parquet`, but per-fold site disjointness from a fresh `build_sites` call.
  - If `sites.max_distance_km` changes after the splits are frozen, the two disagree silently.
  - Fix: assert that the recomputed partition equals the stored one, comparing cell groupings since labels may differ.
- **R2-5. Grid file.**
  - `folds.CELLS` reads `data/interim/spine/cells.parquet`, while the documented grid is `data/processed/grid/cells.parquet`.
  - Point it at the processed file, or assert that both have identical coordinates for the 48 cells. Either way, the published split should depend on the published grid.
- **R2-6. Fold 0 is a single site from a single region** (region 1, 3 cells). That can't be avoided with 2 Toronto development sites, but say so in `splits_summary.md`. Plan §2 says "regions spread across folds".
- **R2-7. The plan's wording is now out of date.** Plan §1 says "whole Ontario monitor cells", and §8 lists the cell counts as an open decision. Both should say sites (see Q2).

## Answers to the two questions

### Q1. Is 10 km right, and is clustering only the study cells correct?

**Yes, to both.**
- **Study cells only.** A site means "the same monitoring location". Unmonitored cells carry no observation, so they have no part in deciding which monitors are the same place. On the full lattice, single linkage just measures that the grid is contiguous.
- **10 km fits the evidence.** In the round-1 data, every clean relocation is a one-cell (5 km) move. These are the pairs where one monitor's record ends as the next one starts, with no shared days: 916605/916606, 856930/856931, 944250/944251, 918737/919802 and 940010/941075. 10 km captures all of them with one cell of margin.
- **Nothing between 10 and 25 km is a relocation.** I listed all 40 pairs in that range. Every pair is either:
  - two monitors running at the same time (hundreds to 5,344 shared days), or
  - two records separated by a gap of years, such as 2010 vs 2017–24, or 2010–16 vs 2019–24.

  These are separate monitors in a dense urban network (the Toronto and GTA area), not one site.
- **15 km chains across regions:** 928317 (region 2) to 930446 (region 5) is 11.2 km, which is why 15 km gives straddling sites and breaks leave-one-region-out.

**Suggested wording for the paper:** "Monitors within 10 km were grouped into one site, because on the 5 km grid a relocated monitor appears as a neighbouring cell. All relocations in the record are one-cell moves."

**What remains, and is fine:** in Toronto and region 2, test and calibration sites are 11–25 km from development sites, which is ordinary network density. Describe the spatial test as "a new site within the existing monitoring network". Leave-one-region-out is the scheme that measures extrapolation to a new area.

### Q2. Should the config count cells or sites?

**Sites.**
- **Config:** replace `n_test_cells` / `n_calib_cells` with `n_test_sites` (8) and `n_calib_sites` (6, one per region), plus `min_site_rows` and `min_dev_sites_per_region`.
- **Report:** give the resulting cells and rows.
- **Partial sites:** taking part of a site would undo F1. Don't allow it.
- **Current shares:** 23% of 2010–2023 rows go to spatial test and 15% to calibration, leaving 62% (23 sites) for development.

I'd keep 8 test sites: it's still enough for one per region plus two extras, and leaves 10 folds of 2–3 sites. If you'd rather give development more sites, 6 test sites (one per region) is the alternative. That trades away a noisier per-region test estimate. Your call, but decide it before R2-1 is re-run.

## Remaining items from the brief

| Item | Answer |
|---|---|
| 5. Engine bound only by a shared module | Agreed. Make the `cv.py --dry-run` → `leakage_test.py` binding (round 1, F3) a **Phase 2 acceptance criterion**, including the inner early-stopping set cut by site. |
| 6. `fold_rows` rejects non-development cells | Keep it strict. A superset of the fold's cells is always a caller error; `folds.plan` passes exact lists. |

## Re-run expected

```
python code/model/splits.py --force       # R2-1 (seeded site choice), R2-5, R2-7 config change
python code/model/leakage_test.py         # with R2-4
python code/model/feature_reduction.py    # R2-2, R2-3
```

After that, Phase 1 can be approved without another full round: send `splits_summary.md` and the rejected-pair table.
