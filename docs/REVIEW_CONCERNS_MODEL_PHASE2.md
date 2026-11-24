# Review of `CONCERNS_MODEL_PHASE2.md`

**Method:** I read the concerns against the current split files, `data/processed/README.md`, the plan and the Phase 2 code. I ran nothing.

**Overall:** the concerns are useful, and most are valid. But three rest on out-of-date facts (A-2, A-3, B-1), and E-2 has already been done. Below, each concern gets a verdict and what to do. Where something should change in the reporting, I've written it as a rule to **fix now, before the test sets are opened**, so it can't be read as a choice made after seeing results.

## Corrections first: the document describes an older split

Calibration was redrawn one cell per region on 19 Nov (12:41), after the author's decision, and that is the approved Phase 1 state. The current cells:

| Role | Cells (region) | Nearest development cell |
|---|---|---|
| Spatial test | 918734 (1), 910214 (2), 895350 (3), 958101 (4), 924048 (5), 856931 (6) | 5.0, 30.4, 68.0, 62.6, 21.2, 5.0 km |
| Calibration | 918737 (1), 928317 (2), 864472 (3), 933610 (4), 930445 (5), **863269 (6)** | 5.0, 11.2, 5.0, 58.3, 5.0, 256.7 km |

So:
- **A-2, second half:** North does have a calibration cell (863269, 5,027 rows), and a North interval can be reported.
- **E-2:** already done.
- **A-3's table:** 916606 is now a development cell. The calibration cells within 5 km of a development cell are 918737, 864472 and 930445, not "916606 + 4 others".

Update the document from `splits_summary.md`.

## A. Evaluation design

**A-1. Six spatial-test cells: valid.** A block bootstrap over 6 cells gives intervals that are wide and unstable. **Rule:**
- Report the spatial test **per cell** (RMSE, bias, n, distance to the nearest development cell), plus the pooled metric.
- Show the bootstrap interval, but describe it as indicative.
- The temporal test (2024, about 29 cells) is the better-powered of the two test sets. Present it that way.

**A-2. Thin North test cell: valid** (the calibration half is out of date). North in the spatial test is one cell, 2010–2012, one smoke day. **Rule:**
- Report North's spatial-test number with n and the "1 day > 25" caveat.
- The North evidence that carries weight is leave-one-region-out (region 6 held out, 4 development cells) and the North group within the 2024 test.

**A-3. Test cells next to development cells: valid.** This is the known cost of the cell-level decision. Two of the six test cells (918734, 856931) are 5 km from development, and 856931 is the same monitor before it moved. **Rule:**
- Label the spatial test **"held-out monitor cells within the existing network"**, not "unseen locations".
- Report it twice:
  - all 6 cells
  - the 4 cells more than 10 km from any development cell (910214, 895350, 958101, 924048)

  The subset is defined now, from distances alone, so it isn't a post-hoc choice.

**A-4. Grouped 10-fold is selection-biased, and 2023 dominates leave-one-year-out: valid.** The summary already labels grouped 10-fold. **Rule for leave-one-year-out:** report a per-year table, and the pooled metric both with and without 2023. Name 2023 as the extreme smoke year. This answers E-4: yes.

## B. Target and features

**B-1. Negatives treated as missing: mostly no longer applies.** The 11,237 dropped station-days are **11,130 AQS and 107 NAPS** (README §3). The models now use Ontario rows only, which are NAPS. So in this dataset the rule removed about 107 station-days, about 0.05% of Ontario rows, and the upward-bias argument hardly applies. Keep the one-line caveat in the methods (negative hours were treated as missing), and drop the AQS-based numbers from this concern.

**B-2. AOD is ~73% missing, so A vs C isn't a clean AOD-vs-reanalysis comparison: valid, and the plan already frames it correctly.** R2's question is "what MERRA-2 aerosol adds beyond AOD", not AOD vs MERRA. A cheap addition with no new run: from the existing OOF files, report R1 vs R2 separately for **days with and without an AOD retrieval** (`aod_missing`). That shows directly whether MERRA-2 aerosol matters mainly on the 73% of days with no satellite retrieval. Add `aod_missing` as an optional split in `summarise.py`.

## C. Compute and environment

**C-1 and C-2. Early stopping doesn't fire at 0.05 / 5,000: valid, and decisive for E-3.** The 77 s fit was made through `lgbm_params`, so at 0.05, and its best iteration was 4,956 of 5,000. **Recommendation: set the learning rate to 0.1 for LightGBM and XGBoost before the full tuning, and keep the cap at 5,000.**
- **Fire:** that roughly halves the trees needed, so early stopping can actually fire.
- **Time:** fits get about twice as fast, so R1 tuning drops from about 21 h towards 8–10 h, before pruning.
- **Cost of waiting:** the study name now includes the config hash, so changing it later forces a complete second tuning run. Changing it now costs nothing.
- **If it still doesn't fire:** if `folds_not_stopped` is still high after the quick pass at 0.1, describe the tree count as a fixed round budget (C-2) rather than an early-stopped one.

This is the author's call. Make it before the full runs.

**C-3. Random Forest can't run in the sandbox: valid.** Phase 2 is going to run elsewhere anyway. Keep `n_jobs = -1` and run R4 where joblib works. Don't switch to `n_jobs = 1` just for this machine.

## D. Residual code risks

- **The early-stopping set is small: valid.** One correction: it isn't "the same six across folds". The inner seed is derived per fold (`fold_seed(..., "es_inner", label)`), so each fold draws its own cells. Six cells is still noisy. Accept it, and note it in the methods.
- **Quick mode:** agreed. Its numbers are for timing only.
- **`run_key` doesn't cover the code: valid, and easy to close.** Add a hash of the source of `models.py`, `cv.py`, `data.py`, `folds.py` and `features.py` to `run_key` and to the Optuna study name. Then editing the fit code forces `--fresh` instead of silently resuming. It's a few lines, and it's worth doing before the full runs.
- **Random Forest grid cost:** agreed, and it's cached. Nothing to change.

## E. Answers

| Question | Answer |
|---|---|
| 1. Report the spatial test as "unseen locations"? | No. Call it "held-out monitor cells within the existing network". Report it per cell, and both for all 6 cells and for the 4 cells more than 10 km from development (A-3). |
| 2. Redraw calibration one per region? | Already done (19 Nov, 12:41). North calibration is 863269. Update the document. |
| 3. Learning rate | Change to 0.1 **before** the full tuning (C-1). It's the author's decision. |
| 4. Leave-one-year-out with 2023 separate | Yes: a per-year table, and pooled with and without 2023 (A-4). |

## Actions

| Who | Action |
|---|---|
| **Coding agent, code only, no runs** | Refresh the concerns document from the current splits. Add the `run_key` / study-name code hash. Add the `aod_missing` split and the leave-one-year-out with/without-2023 table to `summarise.py`. Add per-cell output and the more-than-10-km subset to `score_test.py`. |
| **Author** | Learning rate (0.05 → 0.1?). Where to run Phase 2, including Random Forest. |
| **Plan** | Record the reporting rules from A-1 to A-4 in `MODEL_RUN_PLAN.md` §7 before the test sets are opened. |
