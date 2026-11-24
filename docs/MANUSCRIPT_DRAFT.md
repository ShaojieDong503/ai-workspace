# Manuscript draft: daily PM2.5 for Ontario from satellite AOD, reanalysis and fire data with gradient-boosted trees

Working draft, kept in step with the runs. Last update: 2026-11-23.

**Status.**
- Methods are complete for everything run so far.
- Results cover development cross-validation (Phase 2) only.
- The spatial and 2024 test sets are sealed; every test number below is a placeholder.
- `[TODO]` marks text that needs references or later results.
- Numbers come from `data/interim/model/results_dev.md` (Trillium run of 2026-11-22). The run log is [PHASE2_RESULTS.md](PHASE2_RESULTS.md); the analysis plan is [MODEL_RUN_PLAN.md](MODEL_RUN_PLAN.md).

---

## Title (working)

Daily 5 km PM2.5 for Ontario, 2010–2024, from MAIAC AOD, MERRA-2 and fire observations: a gradient-boosting model evaluated across regions and years

## Abstract (draft; test-set sentences pending)

We estimate daily mean PM2.5 on a 5 km grid over Ontario, Canada, for 2010–2024. The model combines MODIS MAIAC aerosol optical depth (AOD), NASA MERRA-2 aerosol and meteorology, NOAA Hazard Mapping System smoke plumes and FIRMS active-fire detections. It is trained on 36 Ontario monitor cells.

We compared LightGBM, XGBoost and random forest, and a LightGBM variant without the MERRA-2 aerosol fields, under three cross-validation schemes: grouped 10-fold, leave-one-region-out and leave-one-year-out.

LightGBM and XGBoost performed equally well. Under leave-one-region-out, LightGBM reached RMSE 3.09 µg/m³ (R² 0.63). Under leave-one-year-out it reached RMSE 3.86 µg/m³ (R² 0.43), or 3.06 µg/m³ with the 2023 wildfire-smoke year excluded. Removing MERRA-2 aerosol raised the spatial RMSE by 15%, most on days without an AOD retrieval.

All models underpredicted extreme smoke: by 28 µg/m³ on days above 50 µg/m³ across regions, and far more when 2023 was held out entirely. `[TODO: spatial-test and 2024-test results, interval coverage, CanOSSEM comparison.]`

---

## 1. Introduction (outline)

1. Daily PM2.5 exposure in Ontario matters for health studies, but monitors are sparse: 48 Ontario cells with a monitor, about 40 active in any year. `[TODO: health-burden and exposure-assessment references]`
2. Satellite AOD with machine learning is the standard way to fill the gaps. `[TODO: review references; North American and Canadian AOD–PM2.5 models; CanOSSEM]`
3. Two problems are specific to Ontario:
   - **AOD is missing on about 70% of monitor-days** (cloud, snow): 101,823 of 145,315 development rows. Reanalysis aerosol may fill that role.
   - **Wildfire smoke episodes, especially 2023,** produce concentrations far outside the training distribution.
4. Most published evaluations use random k-fold cross-validation, which overstates skill at new locations and in new years. `[TODO: references on spatial and temporal CV]`
5. **Contributions:**
   - (a) a 2010–2024 daily 5 km product for Ontario
   - (b) evaluation by held-out region and held-out year, alongside grouped k-fold
   - (c) a direct measure of what MERRA-2 aerosol adds beyond satellite AOD
   - (d) an explicit account of extreme-smoke performance

## 2. Data

### 2.1 Grid and day

- **Grid:** a 5 km × 5 km grid on the CanOSSEM Lambert projection (EPSG:3347), 163,787 cells, covering Ontario and its surroundings.
- **Day:** all daily values use the UTC−6 day (06:00 UTC to 06:00 UTC the next day), the CanOSSEM day.
- **Period:** 2010-01-01 to 2024-12-31.

### 2.2 PM2.5 observations

- **Source:** hourly PM2.5 from National Air Pollution Surveillance (NAPS) monitors in Ontario.
- **Daily mean:** the mean over the UTC−6 day, requiring at least 18 valid hours. Negative hourly values are treated as missing.
- **Shared cells:** where several monitors share a cell, their daily means are averaged.
- **Ontario rows:** Ontario monitors occupy 48 grid cells (208,093 cell-days, 2010–2024), grouped into six evaluation regions: Toronto, Central, East, West–Southwest, West–Hamilton/Niagara, North.
- **Out-of-province monitors:** the same table holds US EPA AQS and other Canadian monitors, but this study trains and evaluates on Ontario rows only.

### 2.3 Predictors

The predictors, 62 in total, fall into three groups (Table 1). Missing values are not filled; all three learners handle missing values natively.

**Table 1. Predictor groups.**

| Group | Columns | Source and construction |
| --- | --- | --- |
| Satellite AOD | 16 | MODIS MAIAC MCD19A2 C6.1: AOD at 550 and 470 nm, a missing-AOD indicator, 3 smoke columns (injection height, smoke AOD, smoke-mask fraction), and 10 backward-looking context columns (neighbourhood means within 25/50/100 km, the previous 3 days within 50 km, the last observed value up to 7 days back, with counts and age) |
| MERRA-2 aerosol | 15 | Column mass, surface mass and extinction AOD of black carbon, organic carbon, sulfate, dust and sea salt, plus total extinction AOD; daily means on the 0.5° grid |
| Base | 31 | MERRA-2 meteorology (17: boundary-layer height, near-surface humidity, temperature and winds, upper-air winds; 2 m winds dropped), FIRMS MODIS active fire (11: counts and fire radiative power within 25–500 km, distance to the nearest detection), NOAA HMS smoke (3: presence, polygon count, maximum density) |

Two predictor sets are compared:
- **Dataset A:** all 62 columns.
- **Dataset C:** A without the MERRA-2 aerosol group, 47 columns.

## 3. Methods

### 3.1 Data splits

Splits are made at the level of the grid cell and frozen before any model was trained (Table 2).
- **Spatial test set:** one cell drawn at random from each of the six regions (seeded).
- **Calibration set:** a second cell per region, drawn the same way from the cells left after the test draw.
- **Temporal test set:** all Ontario rows in 2024.
- **Development set:** the remaining 36 cells, 2010–2023, 145,315 cell-days.

Test and calibration cells are excluded from training in every year.

**Table 2. Data splits.**

| Split | Cells | Rows | Use |
| --- | --- | --- | --- |
| Development | 36 | 145,315 (2010–2023) | Tuning and cross-validation |
| Spatial test | 6 | 26,275 (2010–2023) | Final evaluation at held-out cells |
| Calibration | 6 | 26,022 (2010–2023) | 90% prediction intervals only |
| Temporal test | all Ontario cells active in 2024 (29) | `[TODO: n]` | Final evaluation in an unseen year |

Neighbouring cells are not grouped. Two of the six spatial-test cells lie 5 km from a development cell, and one of them is a monitor that moved one cell over in 2013. The spatial test therefore measures prediction at held-out monitor cells within the existing network rather than at unmonitored locations. It is reported pooled, per cell, and for the four cells more than 10 km from any development cell.

### 3.2 Models

We fitted four models (Table 3), all with squared-error loss.

**Table 3. Model runs.**

| Run | Learner | Predictors | Hyperparameters | Question |
| --- | --- | --- | --- | --- |
| R1 | LightGBM 4.5 | A | Tuned: Optuna TPE, 100 trials | Main model |
| R2 | LightGBM 4.5 | C | R1's settings; number of trees re-chosen | What MERRA-2 aerosol adds beyond AOD |
| R3 | XGBoost 3.2 (`hist`) | A | Tuned: Optuna TPE, 50 trials | Learner comparison |
| R4 | Random forest (scikit-learn 1.8), 300 trees | A | Tuned: 12-point grid | Benchmark |

### 3.3 Tuning

**Folds.** Hyperparameters were tuned on grouped 10-fold cross-validation over the 36 development cells: each cell is assigned to one fold at random (seeded), and the objective is mean RMSE across the 10 folds.

**Search spaces.**
- **LightGBM:** learning rate 0.06–0.3 (log scale), 7–255 leaves, depth 3–16, 10–500 samples per leaf, feature fraction 0.4–1.0, bagging fraction 0.5–1.0, L1 and L2 penalties 0–10, minimum split gain 0–1.
- **XGBoost:** analogous ranges (depth 3–12, minimum child weight 1–100, subsampling 0.5–1.0, column sampling 0.4–1.0, L1 and L2 penalties 0–10, gamma 0–1).
- **Random forest:** `max_features` ∈ {√p, 0.3, 0.5} × `min_samples_leaf` ∈ {1, 2, 5, 10}.

**Early stopping.** Gradient-boosting fits used early stopping on an inner validation split: 15% of each fold's training cells per region, at least one. The rule was patience 100 rounds, with a cap of 5,000 rounds.

**Selection rule.** Instead of the lowest-error trial, we took the simplest trial within one standard error of the best. The standard error is the standard deviation of the best trial's 10 fold RMSEs divided by √10. "Simplest" means:
- **LightGBM:** fewest leaves, then the most samples per leaf.
- **XGBoost:** shallowest depth, then the largest minimum child weight.
- **Random forest:** the largest leaf size, then the fewest features.

**Final tree count.** The number of trees for the final model is the median of the per-fold early-stopping iterations.

**Chosen settings.**
- **LightGBM:** learning rate 0.104, 13 leaves, depth 15, 82 samples per leaf, feature fraction 0.60, bagging fraction 0.75, L1 8.63, L2 5.66, minimum gain 0.67.
- **XGBoost:** learning rate 0.178, depth 4, minimum child weight 79.6, subsample 0.96, column sample 0.42, L1 7.62, L2 4.27, gamma 0.16.
- `[TODO: chosen random-forest point; final tree counts; best-trial RMSE versus chosen-trial RMSE.]`

### 3.4 Cross-validation schemes

Each run, with its frozen settings, was evaluated under three schemes on the development set:
- **Grouped 10-fold:** the tuning folds. Its estimates are selection-biased and are reported for reference only.
- **Leave-one-region-out (LORO, 6 folds):** spatial transfer to an unseen region.
- **Leave-one-year-out (LOYO, 14 folds, 2010–2023):** temporal transfer to an unseen year. It is reported per year and pooled, with and without 2023, the extreme wildfire-smoke year.

In each scheme every development row is predicted exactly once, out of fold.

### 3.5 Metrics and groups

**Metrics:** RMSE, MAE, bias (predicted minus observed) and R².

**Groups:**
- all rows
- days ≤ 25 µg/m³
- days > 25 µg/m³ and > 50 µg/m³ (R² not reported for these, given their truncated range)
- the North region
- days under heavy HMS smoke

R1 and R2 are also compared separately on rows with and without an AOD retrieval.

### 3.6 Prediction intervals and test evaluation (planned)

**Intervals.** 90% prediction intervals are calibrated once, from the errors on the calibration cells, and are not recalibrated after the test sets are seen.

**Test evaluation.** Each of R1–R4 is retrained on all development data and scored once on the spatial test cells and on 2024. Confidence intervals come from a cell-block bootstrap.

**Pre-specified reporting.** These rules were fixed on 2026-11-19, before any test row was scored:
- the spatial test is reported three ways (pooled, per cell, and cells more than 10 km from development)
- all four runs are scored
- coverage is reported per test set, group and region

### 3.7 Computation

All models were run on the SciNet Trillium cluster (192-core nodes). Each tuning trial fitted its 10 folds in parallel, 19 threads per fold, in 9–90 s. Tuning and cross-validation of all four runs took about 1 h 40 min of node time. Seeds for splits, folds, inner splits, tuning and bootstrap are fixed in the configuration file.

## 4. Results (development cross-validation)

### 4.1 Learner comparison

LightGBM (R1) and XGBoost (R3) performed indistinguishably, and random forest (R4) was consistently worse (Table 4).
- **R1 vs R3:** the difference was at most 0.06 µg/m³ RMSE in any scheme, well inside the fold-to-fold spread (LORO standard deviation about 0.8 µg/m³).
- **R4 vs R1:** RMSE 0.13 µg/m³ higher under LORO and 0.27 µg/m³ higher under LOYO.

**Table 4. Cross-validated performance on the development set (145,315 cell-days). RMSE in µg/m³.**

| Run | LORO RMSE | LORO R² | LOYO RMSE | LOYO R² | LOYO RMSE, 2023 excluded | Grouped 10-fold RMSE (biased) | Grouped 10-fold R² (biased) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 LightGBM, A | 3.09 | 0.63 | 3.86 | 0.43 | 3.06 | 2.76 | 0.71 |
| R3 XGBoost, A | 3.10 | 0.63 | 3.92 | 0.41 | 3.12 | 2.77 | 0.71 |
| R4 Random forest, A | 3.22 | 0.60 | 4.13 | 0.35 | 3.32 | 2.96 | 0.66 |
| R2 LightGBM, C | 3.56 | 0.51 | 4.46 | 0.23 | 3.69 | 3.35 | 0.57 |

The models transfer across regions better than across years. Grouped 10-fold, the scheme used for tuning, gives the most optimistic numbers.

### 4.2 Contribution of MERRA-2 aerosol

Removing the 15 MERRA-2 aerosol columns (R2) raised the LORO RMSE from 3.09 to 3.56 µg/m³ (+15%) and lowered R² from 0.63 to 0.51 (Table 5). The loss was larger on rows without an AOD retrieval, which make up 70% of the development rows. On those rows, MERRA-2 aerosol carries information the satellite cannot supply that day. Because R2 reuses R1's hyperparameters, part of the gap may come from the borrowed settings rather than the missing predictors. `[TODO: confirm with a tuned R2, if run.]`

**Table 5. R1 and R2 by AOD availability. RMSE in µg/m³.**

| Scheme | Run | All rows | AOD retrieved (43,492) | No AOD (101,823) |
| --- | --- | --- | --- | --- |
| LORO | R1, with MERRA-2 aerosol | 3.09 | 3.03 | 3.12 |
| LORO | R2, without | 3.56 | 3.45 | 3.61 |
| LOYO | R1, with MERRA-2 aerosol | 3.86 | 4.09 | 3.76 |
| LOYO | R2, without | 4.46 | 4.65 | 4.38 |

### 4.3 High concentrations and wildfire smoke

On ordinary days (≤ 25 µg/m³, 99.3% of rows) R1 was nearly unbiased: bias +0.14 µg/m³ under LORO, LORO RMSE 2.62 µg/m³. Error and bias grew sharply with concentration (Table 6).

**Table 6. R1 by group. RMSE and bias in µg/m³.**

| Group | Rows | LORO RMSE | LORO bias | LOYO RMSE | LOYO bias |
| --- | --- | --- | --- | --- | --- |
| ≤ 25 µg/m³ | 144,370 | 2.62 | +0.14 | 2.93 | +0.06 |
| > 25 µg/m³ | 945 | 20.6 | −8.9 | 31.4 | −20.5 |
| > 50 µg/m³ | 101 | 55.2 | −28.3 | 81.9 | −67.2 |
| Heavy HMS smoke | 1,479 | 14.4 | −0.3 | 21.1 | −3.0 |
| North region | 18,883 | 4.00 | −0.91 | 4.75 | −0.40 |

The underprediction is much worse when a whole year is held out:
- **2023 held out:** the LOYO RMSE was 9.5 µg/m³, against 2.5–3.6 µg/m³ for every other held-out year (Table 7).
- **Heavy-smoke days:** R² fell from 0.54 (LORO) to 0.01 (LOYO).
- **Without 2023:** the pooled LOYO RMSE (3.06 µg/m³) matches LORO.

So the temporal-transfer error is concentrated in one year whose smoke has no analogue in the other 13.

**Table 7. LOYO RMSE by held-out year, µg/m³.**

| Run | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 3.23 | 3.02 | 2.89 | 3.26 | 3.61 | 3.14 | 2.48 | 2.65 | 3.30 | 3.26 | 2.65 | 3.31 | 2.74 | 9.48 |
| R2 | 3.64 | 3.34 | 3.19 | 3.73 | 4.42 | 3.84 | 3.02 | 3.08 | 4.60 | 3.62 | 3.19 | 4.57 | 3.18 | 10.22 |
| R3 | 3.21 | 3.11 | 2.94 | 3.31 | 3.65 | 3.15 | 2.50 | 2.72 | 3.49 | 3.32 | 2.72 | 3.38 | 2.81 | 9.53 |
| R4 | 3.47 | 3.28 | 2.97 | 3.54 | 4.02 | 3.58 | 2.73 | 2.82 | 3.78 | 3.31 | 3.08 | 3.39 | 2.97 | 9.89 |

Every learner fails on 2023 by a similar margin (9.5–10.2 µg/m³). The limit lies in the training data, not in the choice of learner.

### 4.4 Spatial transfer by region

Spatial transfer differed widely between regions (Table 8).
- **Toronto and Central:** these transferred best. R1 RMSE was 2.08 and 2.06 µg/m³, R² 0.84 and 0.82.
- **East, West SW and West NH:** R1 RMSE was 3.10–3.67 µg/m³, R² 0.50–0.58.
- **North:** holding it out gave the highest RMSE under every run, 4.00 µg/m³ for R1 with R² 0.47. Its development data are four cells with 74 days above 25 µg/m³.

Across the six held-out regions, R1's RMSE ranged from 2.06 to 4.00 µg/m³ (mean 3.03, SD 0.81).

R1 and R3 were within 0.03 µg/m³ of each other in every region. Random forest (R4) beat both only when West NH was held out (3.07 against 3.26 µg/m³). Removing MERRA-2 aerosol (R2) raised RMSE in every region. The rise was largest for Toronto (+0.99 µg/m³) and smallest for West NH (+0.06 µg/m³).

**Table 8. LORO RMSE (R²) by held-out region. RMSE in µg/m³; development rows per region in brackets.**

| Run | Toronto (16,850) | Central (31,526) | East (26,174) | West SW (27,085) | West NH (24,797) | North (18,883) |
| --- | --- | --- | --- | --- | --- | --- |
| R1 LightGBM, A | 2.08 (0.84) | 2.06 (0.82) | 3.10 (0.58) | 3.67 (0.50) | 3.26 (0.56) | 4.00 (0.47) |
| R3 XGBoost, A | 2.08 (0.84) | 2.08 (0.82) | 3.08 (0.59) | 3.70 (0.49) | 3.27 (0.55) | 4.01 (0.47) |
| R4 Random forest, A | 2.12 (0.83) | 2.21 (0.79) | 3.28 (0.53) | 3.86 (0.44) | 3.07 (0.61) | 4.35 (0.38) |
| R2 LightGBM, C | 3.07 (0.64) | 2.50 (0.73) | 3.74 (0.39) | 4.06 (0.38) | 3.32 (0.54) | 4.60 (0.30) |

R² is computed within each held-out region, so it reflects how well the model tracks day-to-day variation there.

### 4.5 Test-set performance

`[TODO after OPEN_TEST: spatial test pooled / per cell / > 10 km subset; 2024 test; bootstrap CIs; interval coverage per set, group and region.]`

### 4.6 Comparison with CanOSSEM

`[TODO: if compatible predictions are available.]`

## 5. Conclusions

`[TODO after test results.]`

---

## Open items before submission

- [ ] Fill the tuning details (§3.3): chosen random-forest point, final tree counts, best vs chosen RMSE, and whether any chosen fold hit the round cap.
- [ ] Decide on the contingency runs X1–X3 (gap-filled AOD; temporal and spatial features) and record the decision note before opening the test sets.
- [ ] Close the three Phase 3 code gaps listed in the run plan (§8).
- [ ] Open the test sets once; fill §4.5, the abstract and the conclusions.
- [ ] CanOSSEM comparison (§4.6) and the 2010–2024 grid product.
- [ ] Introduction references.
