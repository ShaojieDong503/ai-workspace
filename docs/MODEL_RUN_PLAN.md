# Model run plan: single-stage LightGBM for daily PM₂.₅ in Ontario

Data: `data/processed/target/target_table.parquet`, 2010–2024, UTC-6 day, at least 18 valid hours, negative hours treated as missing (see `data/processed/README.md`).

**Where it runs:** all modelling code runs on **SciNet Trillium** (Linux, Slurm; see `code/model/SCINET.md`). The local Windows machine is for editing and code review only.

**Aligned with the code** in `code/model/` on 20 Nov 2026 (§1, §4, §5). Three Phase 3 gaps remain open in the code, listed in §8.

Training and scoring use **Ontario rows only** (`on_region > 0`, NAPS monitors). Monitors outside Ontario are not used. All models use the L2 (squared-error) loss.

## 1. Data splits (frozen before any training)

| Split | Content | Used for |
|---|---|---|
| **Spatial test** | 6 Ontario cells: one cell chosen at random from each region (seeded). Scored on their 2010–2023 rows | Final performance at unseen Ontario locations, once |
| **Temporal test** | All Ontario rows in 2024, every cell, including the 2024 rows of the test and calibration cells | Final performance in an unseen year, once (not for smoke-day groups: only 8 Ontario days above 25 µg/m³) |
| **Calibration** | 6 Ontario cells: one cell chosen at random from each region, from the cells left after the test draw (seeded) | 90% prediction intervals only |
| **Development** | Everything else, 2010–2023 | Tuning, all experiments |

The spatial test and calibration cells are excluded from every training set, in all years.

**The split unit is the cell** (decision 18 Nov), for the test, calibration and development roles and for the 10 folds. Neighbouring cells are not grouped, including monitors that moved one cell over, so a test or held-out cell can have a training cell 5 km away. The spatial test therefore measures prediction at a new monitor location within the existing network. The results report each test cell's distance to the nearest training cell.

## 2. Validation schemes (inside development)

| Scheme | Folds | Role |
|---|---|---|
| Grouped 10-fold: development cells assigned to folds at random (seeded), whole cells per fold | 10 | **The only scheme used for tuning** |
| Leave-one-region-out | 6 | Reporting only, with frozen settings |
| Leave-one-year-out | 14 (2010–2023) | Reporting only, with frozen settings. The 2023 fold is the extreme smoke year |

Fitted inside each fold, on that fold's training rows only: the AOD gap-filler (contingency run X1 only); early stopping (on an inner split).

## 3. Feature groups

| Group | Columns |
|---|---|
| Satellite AOD | `aod055`, `aod047`, `aod_missing`, 10 AOD context columns, 3 MAIAC smoke columns |
| MERRA-2 aerosol | 15 mass and extinction columns (including `merra_totexttau`) |
| Base | MERRA-2 meteorology (17; 2 m winds dropped), fire (11), HMS (3) |

| Dataset | Satellite AOD | MERRA-2 aerosol | Base |
|---|:---:|:---:|:---:|
| **A** Full | ✔ | ✔ | ✔ |
| **C** AOD, no MERRA-2 aerosol | ✔ | | ✔ |

No feature pruning: every model uses all columns of its dataset, 62 for A and 47 for C.

## 4. Runs

"Frozen" means the hyperparameters from R1; only the number of trees is re-picked by early stopping. Only R2 is frozen. R1, R3 and R4 are each tuned on grouped 10-fold with the same 1-SE rule.

| Run | Algorithm | Data | Hyperparameters | Question | Paper |
|---|---|---|---|---|---|
| **R1** | LightGBM | A, observed AOD + indicator | **Optuna, 100 trials, grouped 10-fold** | Main model | Main |
| R2 | LightGBM | C | Frozen (R1's) | What MERRA-2 aerosol adds beyond AOD | Main |
| R3 | XGBoost | A | Optuna, 50 trials, same folds and rule | Algorithm comparison | Main (one table) |
| R4 | Random Forest (scikit-learn, `criterion="squared_error"`, 300 trees) | A, NaN kept (§6) | Grid, 12 points, same folds and rule | Benchmark | Main (one table) |
| — | Two-stage corrector | — | — | Negative result from the thesis | Supplement, reported from thesis runs |

Totals: two LightGBM versions, one XGBoost, one Random Forest. Four models.

**Contingency runs**, only if the author judges R1's development performance to be bad, after reviewing the R1 results. There is no automated criterion; the judgement and its reasons are recorded in the decision note.

| Run | Algorithm | Data | Hyperparameters | Question |
|---|---|---|---|---|
| X1 | LightGBM | A, AOD gap-filled within each fold + indicator | Frozen | Does gap filling help over observed AOD? |
| X2–X3 | LightGBM | A + temporal / spatial feature sets (2–3 sets, to be defined) | Frozen | Do temporal and spatial features help? |

### Tuning and fitting details

All values are from `code/model/config.json`.

**Search spaces.**

| LightGBM (R1) | Range | XGBoost (R3) | Range |
|---|---|---|---|
| `learning_rate` | 0.06–0.3, log scale | `learning_rate` | 0.06–0.3, log scale |
| `num_leaves` | 7–255 | `max_depth` | 3–12 |
| `max_depth` | 3–16 | `min_child_weight` | 1–100 |
| `min_data_in_leaf` | 10–500 | `subsample` | 0.5–1.0 |
| `feature_fraction` | 0.4–1.0 | `colsample_bytree` | 0.4–1.0 |
| `bagging_fraction` (`bagging_freq` 1) | 0.5–1.0 | `reg_lambda`, `reg_alpha` | 0–10 each |
| `lambda_l1`, `lambda_l2` | 0–10 each | `gamma` | 0–1 |
| `min_gain_to_split` | 0–1 | `tree_method` | `hist` (fixed) |

- **Random Forest (R4):** a grid of `max_features` {sqrt, 0.3, 0.5} × `min_samples_leaf` {1, 2, 5, 10}, 12 points, with 300 trees.
- **Learning-rate range:** it starts at 0.06 because a slower rate needs more than the 5,000-round cap to converge, which would truncate the trial.

**Choosing the trial (the 1-SE rule).**
- **Standard error:** SE = standard deviation of the best trial's 10 per-fold RMSEs (ddof 1) / √10.
- **The candidates:** the trials whose mean RMSE is within one SE of the best.
- **The simplest candidate is chosen:**
  - LightGBM: the fewest `num_leaves`, then the largest `min_data_in_leaf`.
  - XGBoost: the smallest `max_depth`, then the largest `min_child_weight`.
  - Random Forest: the largest `min_samples_leaf`, then the smallest `max_features` as a fraction of the columns (sqrt = 1/√62).

**Early stopping.**
- **Inner split:** inside every fold, 15% of the training cells per region (at least one) are held out.
- **Stopping rule:** patience 100 rounds, with a cap of 5,000.
- **Random Forest:** it has no early stopping, so it fits on the training cells and the inner cells together.

**Number of trees for the final model.** The median of the per-fold best iterations from the run's own grouped 10-fold run. R2 uses its own count, not R1's.

**Pruning and parallelism.**
- **Folds run in parallel within a trial;** trials run one after another.
- **No pruning when folds run in parallel.** A median pruner is used only when folds run one at a time.

**Seeds.** Fixed in `config.json`: split, fold, early-stopping inner split, tuning and bootstrap.

**Quick pass.** A pipeline check before the full runs, written to a separate directory and never reported:
- 10% of training rows
- 300 rounds
- 3 trials

## 5. Steps in order

| Step | What | Output |
|---|---|---|
| 1 | Freeze the spatial test, calibration and 2024 splits, and the fold assignment | `splits.parquet`, `folds.parquet` |
| 2 | Feature sets fixed as datasets A (62 columns) and C (47 columns) in §3; no pruning | — |
| 3 | Tuning on grouped 10-fold: R1 (Optuna, 100 trials), R3 (Optuna, 50 trials), R4 (12-point grid). Choose the simplest trial within 1 SE of the best (§4, tuning details); number of trees = median of per-fold best | `params_lgbm.json`, `params_xgb.json`, `params_rf.json` |
| 4 | R1–R4 on grouped 10-fold, leave-one-region-out and leave-one-year-out, each with its tuned settings (R2 with R1's). The author reviews R1 and decides whether to run X1–X3 | Out-of-fold predictions per run and scheme |
| 5 | Retrain R1–R4 once each on all development data; compute 90% intervals from calibration-cell errors | Final models |
| 6 | Score each of R1–R4 once on the spatial test cells (2010–2023 rows) and on 2024; block-bootstrap confidence intervals (resample by cell); check interval coverage | Final results tables |
| 7 | Compare with CanOSSEM on the same test observations | Benchmark table |
| 8 | Predict the Ontario 5 km grid with the frozen main model | Final product |

## 6. Missing values

No filling: every model keeps NaN and handles it natively (LightGBM, XGBoost, and scikit-learn's `RandomForestRegressor`, which supports NaN since version 1.4).

| Model | AOD | Other NaN columns |
|---|---|---|
| LightGBM, XGBoost, Random Forest | NaN + `aod_missing` | Stay NaN |
| Gap-filled version (X1, contingency) | Filled value + `aod_missing` | Stay NaN |

## 7. Reporting

**Fixed on 19 Nov 2026, before any test row was scored.** The test sets are still
sealed (`OPEN_TEST` does not exist). Any deviation is a finding to be reported, not
a choice to be made quietly.

### 7.0 Groups and metrics (every run)

All Ontario rows · ordinary days (≤ 25 µg/m³) · above 25 · above 50 · North region · heavy HMS smoke days. Metrics: RMSE, MAE, bias, R² (R² not for the above-25 and above-50 groups).

Development counts (Ontario, 2010–2023): 197,612 rows; 1,318 above 25; 134 above 50; North 25,005.

### 7.1 The spatial test is "held-out monitor cells within the existing network"

Not "unseen locations". Two of the six cells (918734, 856931) are 5 km from a
development cell, and 856931 is the same monitor as 856930 before it moved in 2013.

### 7.2 The spatial test is reported three ways, always together

| presentation | cells | purpose |
|---|---|---|
| pooled | 6 | headline number |
| per cell | 6 | RMSE, bias, n, mean PM2.5, days above 25, distance to the nearest development cell |
| more than 10 km from development | 4 (910214, 895350, 958101, 924048) | closest available proxy for an unseen location |

The subset is fixed by distance alone, before scoring, so it cannot be a post-hoc
selection. The per-cell table is not optional: with six cells the pooled number
hides the spread.

### 7.3 Bootstrap intervals on the spatial test are indicative, not confirmatory

A cell-block bootstrap over six cells resamples six units. The 2024 temporal test
(29 cells) is the better-powered of the two sets and is presented as such.

### 7.4 North

The North spatial-test number is reported with its `n` and the caveat that the cell
covers 2010–2012 and one day above 25 µg/m³. The North evidence that carries weight
is leave-one-region-out with region 6 held out, and the North group inside the 2024
test.

### 7.5 Selection bias is labelled

Grouped 10-fold is the scheme the parameters were chosen on. Its block in the
results is titled "tuning folds (selection-biased)". Leave-one-region-out and
leave-one-year-out are the development estimates; the test sets have the final word.

### 7.6 Leave-one-year-out is reported per year, and pooled twice

With and without 2023, with 2023 named as the extreme smoke year that dominates that
scheme's variance.

### 7.7 R1 versus R2 is reported by AOD availability

Overall, and split by `aod_missing` (0/1), on **leave-one-region-out and
leave-one-year-out**. Grouped 10-fold is shown for reference only: R1 was tuned on
those folds and R2 reuses its parameters, so that comparison flatters R1.

### 7.8 The test sets are scored once

`score_test.py` runs once per chosen model after `OPEN_TEST` is created, and refuses
to overwrite an earlier result. Any change to the model, the features or the split
afterwards is recorded in the decision note with its reason, and the test results
are reported as superseded rather than overwritten.

### 7.9 All four runs are finalised and scored

R1–R4 are finalised and scored on the test sets, once each. No run is dropped after
its test score is seen.

### 7.10 The contingency decision comes first

The decision on X1–X3, and the decision note giving its reasons, is recorded before
`OPEN_TEST` is created.

### 7.11 The interval is calibrated once

The 90% interval is calibrated once, on the calibration cells (2010–2023), and is not
recalibrated after the test sets are seen. Coverage is reported per test set, per
group and per region. For 2024, coverage also tests whether the interval holds over
time, since it was calibrated on earlier years.

## 8. Open decisions

- The temporal and spatial feature sets for X2–X3, if they are needed.
- Whether compatible out-of-fold CanOSSEM predictions are available.
- Not yet written: X1–X3 (contingency), step 7 (CanOSSEM comparison), step 8 (grid prediction).
- Phase 3 code still short of §7, to fix before `OPEN_TEST`:
  - `score_test.py` reports coverage per group but not **per region** (7.11).
  - `final.py` can be re-run and silently re-calibrate the interval. It needs the same refusal and `--supersede` as `score_test.py` (7.11).
  - `score_test.py` reports the >25, >50 and heavy-smoke groups for 2024, but §1 says 2024 is not used for smoke-day groups. Either drop those rows or label all three as not interpretable.
