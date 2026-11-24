# Audit findings

Audit date 2026-11-08. Evidence is in `audit/evidence/`, and the scripts that produced it are in `audit/scripts/`.

**Severity:**
- **Critical:** invalidates results.
- **High:** materially biases training or evaluation.
- **Medium:** a real defect or deviation with limited or unquantified effect.
- **Low:** minor or documentation-only.

**Status:**
- **Confirmed:** reproduced with data.
- **Suspected:** evidence points to a problem, but its meaning is not settled.
- **Not Reproduced:** looked for and not found.
- **Expected Anomaly:** unusual, but explained by the source or design.

No Critical or High finding was confirmed.

## Index

| ID | Title | Severity | Status | Affected |
|---|---|---|---|---|
| F-01 | AQS 31 December loses its last hour at UTC−5 sites | Low | Confirmed | `targets.aqs` → `station_day` → training |
| F-02 | Negative hourly values stay inside daily means | Medium | Confirmed (deviation) | `targets.py`, `build_on_utc6.py`, training |
| F-03 | AQS QC-qualifier hours are used as valid | Medium | Suspected | `targets.aqs`, training |
| F-04 | MERRA-2 2 m winds are exactly 0 over forest in leaf-on months | Medium | Confirmed | `merra_u2m`, `merra_v2m`, both tables |
| F-05 | `smoke_mask_frac` counts cloud-covered pixels as "no smoke" | Low | Confirmed | `extract_maiac`, both tables |
| F-06 | MAIAC export script is absent from the project | Medium | Confirmed | MAIAC provenance |
| F-07 | Negative MAIAC injection heights passed through | Low | Suspected | `injection_height` |
| F-08 | README misstates when `smoke_mask_frac` is NaN | Low | Confirmed | `data/processed/README.md` |
| F-09 | README describes the domain as a lon/lat box; Quebec City exclusion not mentioned | Low | Confirmed | `data/processed/README.md` |
| F-10 | Ontario targets rounded to 0.01 µg/m³, others not | Low | Expected Anomaly | `on_utc6_daily` → `station_day` |
| F-11 | Thompson, MB, 2017-03-08: 295 µg/m³ with no smoke context | Low | Suspected | `station_day`, training |
| F-12 | Monitor composition shifts strongly over time | Low | Expected Anomaly | Training table |
| F-13 | `province_state` and `on_region` disagree in three border cells | Low | Expected Anomaly | Training table, grid |
| F-14 | Known source gaps (HMS days, HMS density 2010, fire 2024-12-31) | Low | Expected Anomaly | Base, both tables |
| F-15 | Training unit is cell-day, not station-day as specified | Low | Confirmed (documented deviation) | `build_target_table.py` |
| F-16 | Specified documentation artefacts missing | Low | Confirmed | `docs/`, output folders |
| F-17 | Existing QA never reconciles QC/MB/AQS targets with raw and samples features only in Jan/Jul | Medium | Confirmed | QA process |

Searched for and **not reproduced:**
- duplicate keys, missing days or cells, or NaN in required inputs (all 180 months);
- fill zeros leaking into AOD means (`a14`);
- int16 overflow in fire counts;
- daylight-saving contamination in AQS;
- duplicate AQS hourly rows;
- station → cell errors;
- the MERRA-2 window off UTC-6 in any year;
- future information in context features;
- training ↔ prediction feature mismatch.

---

### F-01 · AQS 31 December loses its last hour at UTC−5 sites
- **Severity:** Low · **Status:** Confirmed · **Affected:** `code/panel/targets.py` `aqs()`, then `station_day.parquet` and `target_table.parquet`.
- **What:** `aqs()` groups hours by (site, POC, UTC-6 day) separately within each annual file, then concatenates the files.
  - The UTC-6 day 31 December ends at 06:00 GMT on 1 January.
  - For Eastern-time sites, that last hour (local 00:00 on 1 January) is in the next year's file.
  - It forms a 1-hour partial group, which the 18-hour rule then discards.
  - The 31 December value is therefore the mean of at most 23 hours.
- **Evidence:** `a09_compare_aqs.log`, `a22_dec31_bias.log`. A reference that pools hours across files differs on exactly 1,053 station-days, all on 31 December 2010–2023, all at UTC−5 sites. Every other AQS row matches.
  - Reference minus pipeline: mean +0.044, range −1.02 to +2.35 µg/m³.
  - The missing hour is New Year's midnight.
- **Impact:** about 0.1% of training rows, with a small mean effect. No station-day was lost or gained.
- **Recommendation:** pool hourly rows across files before grouping (or read file Y+1's first 6 GMT hours), then rebuild `station_day` and the training table. 2024-12-31 needs the 2025 AQS file.

### F-02 · Negative hourly values stay inside daily means
- **Severity:** Medium · **Status:** Confirmed (deviation from the supervisor's direction) · **Affected:** NAPS (`build_on_utc6.clean`, `targets.from_wide`), AQS (`targets.aqs`), training targets.
- **What:** the supervisor asked that invalid or negative **hourly** values be treated as missing. The build follows the user's later decision instead: it keeps negative hours and drops only station-days with a negative **daily** mean (2,668).
- **Evidence:** `a05`, `a09`. Station-days containing negative hours: AQS 90,176 (13.5%); NAPS 941. Applying the hourly rule would:
  - remove 11,130 AQS and 107 NAPS station-days (fewer than 18 valid hours left);
  - raise 34,462 AQS and 296 NAPS daily means by more than 0.5 µg/m³;
  - shift the mean by +0.07 µg/m³ (AQS).
- **Impact:** a systematic small low bias at clean sites and days, concentrated in AQS. It mainly matters for low-concentration performance metrics.
- **Recommendation:** confirm with the supervisor which rule applies. If it's the hourly rule, set negative hours to missing before the 18-hour count and rebuild. The reference outputs in `audit/evidence/*_ref_station_day.parquet` already carry this variant (`pm25_negmissing`).

### F-03 · AQS QC-qualifier hours are used as valid
- **Severity:** Medium · **Status:** Suspected · **Affected:** `targets.aqs`, training targets.
- **What:** the `Qualifier` column is not read. Kept hours include QC-concern codes, alongside exceptional-event codes that must be kept. For spine sites in 2023, for example:

  | Code | Hours | Meaning |
  |---|---:|---|
  | QX | 14,165 | does not meet QC criteria |
  | SX | 8,242 | does not meet siting criteria |
  | 5 | 205 | outlier |
  | NS | 36 | influenced by nearby source |
  | 1, 2, 3, 6 | | deviation / QA codes |

  Fire and exceptional-event codes (IF, IT, RF, RT) mark real smoke values and must be kept.
- **Evidence:** `a10_aqs_qualifiers.log`. 36,466 kept AQS station-days contain at least one qualified hour.
- **Why only suspected:** the code meanings above come from the AQS qualifier code table and should be confirmed against EPA's current list. The effect on daily means hasn't been quantified.
- **Recommendation:** decide, with the supervisor, which codes count as "invalid" (QX, 5 and NS are likely candidates). Keep event flags, quantify, and rebuild if the rule changes.

### F-04 · MERRA-2 2 m winds are exactly 0 over forest in leaf-on months
- **Severity:** Medium · **Status:** Confirmed (values); interpretation Suspected · **Affected:** `merra_u2m`, `merra_v2m` in the prediction and training tables.
- **What:** in 3,262,029 (`u2m`) and 3,262,412 (`v2m`) cell-days, both are exactly 0.0, while `u10m` is not and wind speed is at least 2.6 m/s.
  - It happens only in April–November, every year.
  - It covers about 56 MERRA pixels (July 2015) over forested PA, NY, VT, Ontario and Québec.
  - The zeros are already in the raw Earth Engine export (27 zero pixels in the raw band on 2015-07-01), so extraction is faithful.
- **Likely cause:** MERRA-2 reports 2 m wind as 0 when 2 m lies below the canopy displacement height. A 0 therefore means "undefined", not "calm".
- **Evidence:** `a11_profile_report.log`, `a12_u2m_zero.log`.
- **Impact:** about 1% of rows in summer months. A tree model will read these values as calm air at forested cells.
- **Recommendation:** set exact-zero `u2m`/`v2m` to NaN (with an indicator), or drop the 2 m winds and rely on `u10m`/`v10m`. Document the choice.

### F-05 · `smoke_mask_frac` counts cloud-covered pixels as "no smoke"
- **Severity:** Low · **Status:** Confirmed · **Affected:** `extract_maiac`; `smoke_mask_frac` in both tables.
- **What:** the raw `SMOKE_MASK` band is 0 wherever there's no smoke flag, including pixels with no AOD retrieval (cloud, snow). It's NaN only in orbit gaps.
  - The cell mean is therefore taken over all pixels, not over valid pixels as the specification says (§4.1, "mean of the mask over valid pixels").
  - A cell fully under cloud gets 0.
- **Evidence:** `a07_maiac_raw.log` (SMOKE_MASK finite share 1.000 where AOD finite share is 0.26), `a15_smokemask_nan.log`.
- **Recommendation:** either recompute it over pixels with finite AOD, or document it as "share of all pixels" and rely on `aod055_npix` to separate the cases.

### F-06 · MAIAC export script is absent from the project
- **Severity:** Medium · **Status:** Confirmed · **Affected:** provenance of the 5 MAIAC inputs.
- **What:** the README states that the stack uses MCD19A2 C6.1, the QA bits, the 0.001 scale and a mean over all granules. No script in `code/` (or `.tmp/`) produces the stack, so none of this can be verified.
- **Recommendation:** add the Earth Engine export script (and its run date and parameters) to `code/colab/`. A thesis examiner will ask for it.

### F-07 · Negative MAIAC injection heights passed through
- **Severity:** Low · **Status:** Suspected · **Affected:** `injection_height`.
- **What:** 1,924 cell-days have values below 0, down to −647 m, mostly in heavy-fire summers (2017, 2021, 2023, 2024). They are already in the raw band (`a07`: −316 m on 2023-06-11).
- **Recommendation:** check the MCD19A2 documentation for the band's reference level. If negative values are invalid, set them to NaN.

### F-08 · README misstates when `smoke_mask_frac` is NaN
- **Severity:** Low · **Status:** Confirmed · **Affected:** `data/processed/README.md` §4.1.
- **What:** the README says it's NaN "south of 41° N only". There are 40,424 further NaN rows in covered cells (2011-06-10, 2011-06-20, 2012-02-01) where no granule passed.
- **Recommendation:** amend the README.

### F-09 · README describes the domain as a lon/lat box; Quebec City exclusion not mentioned
- **Severity:** Low · **Status:** Confirmed · **Affected:** `data/processed/README.md` §1, §3.
- **What:** "longitude 98.2–67.0° W" suggests a lon/lat box. The domain is in fact a rectangle in EPSG:3347, whose east edge is at 71.33° W at 46.8° N.
  - Six NAPS stations on the domain list (Québec 050308/050310/050311/050313/050314, Lévis 055702) are therefore correctly outside it.
  - The README doesn't say so, though it's why Quebec has 49 monitors rather than 55.
- **Evidence:** `a19_misc.log`, `a20_quebec_city.log`.
- **Recommendation:** describe the domain as a projected rectangle and list the excluded stations.

### F-10 · Ontario targets rounded to 0.01 µg/m³, others not
- **Severity:** Low · **Status:** Expected Anomaly · **Affected:** Ontario rows of `station_day`.
- **What:** Ontario daily means come from `on_utc6_daily`, which stores 2 decimals; QC/MB/AQS are full float32. 73 Ontario values differ from a raw rebuild by at most 0.005.
- **Recommendation:** none needed for modelling. Mention it in the data description.

### F-11 · Thompson, MB, 2017-03-08: 295 µg/m³ with no smoke context
- **Severity:** Low · **Status:** Suspected · **Affected:** one station-day (NAPS 070501).
- **What:** the raw hourly series:
  - jumps from 8 to 376 at H19 on 7 March, peaking at 893;
  - stays at 374–826 until about H10 on the 8th, then drops below 30 by H13;
  - reads a nearly constant 2–3 µg/m³ for the following 30 hours.

  That is plausible for a local fire, and also for an instrument fault.
- **Recommendation:** check with ECCC or local records. Keep the value unless a cause is found, and consider a robust loss (already in the supervisor's plan).

### F-12 · Monitor composition shifts strongly over time
- **Severity:** Low · **Status:** Expected Anomaly · **Affected:** training table.
- **What:** AQS monitors with data grow from 54 (2010) to 239 (2024), while NAPS goes from 78 to 70. AQS annual means are 0.6–5.4 µg/m³ higher than NAPS (`a04_train_profile.log`). Training rows per year therefore shift toward the US and south of 41° N, where there's no MAIAC.
- **Recommendation:** with 2024 as the test year, report Ontario-only metrics, and consider weighting or year-stratified checks.

### F-13 · `province_state` and `on_region` disagree in three border cells
- **Severity:** Low · **Status:** Expected Anomaly (documented) · **Affected:** training table, `grid/cell_ontario.parquet`.
- **What:**
  - Sarnia cell 944250 is labelled Michigan but `on_region` 4.
  - Port Huron 945315 and Gatineau 863406 are labelled Ontario but `on_region` 0.
  - Windsor 060211 is excluded from Ontario because its cell's target is mostly Detroit air.
- **Recommendation:** select Ontario with `on_region > 0`, as the README says. Confirm the Windsor decision with the supervisor.

### F-14 · Known source gaps
- **Severity:** Low · **Status:** Expected Anomaly (documented) · **Affected:** base and both tables.
- **What:**
  - HMS files are missing for 2012-10-07, 2015-06-02, 2015-08-20, 2016-03-06 and 2016-11-12 (NaN rows).
  - HMS density is "NA" February–May 2010.
  - Fire and AQS data for 2024-12-31 lack the hours from the 2025 files.
- **Recommendation:** add the 2025 MODIS and AQS files when available.

### F-15 · Training unit is cell-day, not station-day as specified
- **Severity:** Low · **Status:** Confirmed (documented deviation) · **Affected:** `build_target_table.py`, training table.
- **What:** the build specification (§1, §7, §8) asks for station-day rows with station fields, `n_valid_hours`, `n_poc`, and "row count = station_day rows". The table instead averages monitors per cell: 21,017 multi-monitor cell-days, with spreads up to 46 µg/m³ in the Windsor/Detroit cell.
  - This is documented in the README and in the code docstring.
  - As a consequence, station-level cross-validation or calibration needs `station_day.parquet`.
- **Recommendation:** keep it, but state it as a deliberate deviation in the thesis methods.

### F-16 · Specified documentation artefacts missing
- **Severity:** Low · **Status:** Confirmed · **Affected:** `docs/`, output folders.
- **What:** the specification asks for the following, which don't exist:
  - `docs/MODEL_INPUTS.md` and `docs/DATA_DICTIONARY.md`;
  - `manifest.json` (input hashes, package versions) beside each output (only the spine has one);
  - `README.md` and `qa_report.md` beside `data/processed/prediction/` and `target/`.

  `data/processed/README.md` covers most of the dictionary content.
- **Recommendation:** add manifests with input hashes before archiving the thesis data.

### F-17 · Existing QA never reconciles QC/MB/AQS targets with raw data and samples features only in January/July
- **Severity:** Medium · **Status:** Confirmed · **Affected:** QA process (`verify_*.py`).
- **What:**
  - `verify_stage1.py` recomputes 30 Ontario means from the *processed* hourly file, and no check touches raw AQS or raw QC/MB NAPS. F-01 went unnoticed because of this.
  - Feature verifiers sample January and July only, and the MERRA window evidence covers July 2015 only.
  - The MAIAC empty-day counter only counts the first 256-row chunk of a tile, so the earlier claim that the rule "never fired" is unproven.
  - `verify_maiac.py` has the two known false-fail issues (absolute tolerance on `injection_height`; distinct-value median in sparse months).
- **Recommendation:** keep `a02`/`a03`/`a05`/`a09`/`a21` as standing checks, add a year-end boundary test, and sample all months in the feature verifiers.

## Resolution log

**2026-11-08: F-01 and F-02 fixed** (user decision: negative hourly value = missing).
- `code/panel/targets.py`:
  - a negative hour is treated as missing for all networks (Ontario now reads the hourly `on_utc6` product, which also removes the 2-decimal rounding of F-10);
  - AQS hours are pooled across annual files as sums and counts.
- Rebuilt `station_day.parquet` (1,067,868 rows: AQS 656,629, NAPS 411,239) and `target_table.parquet` (1,042,511 rows, 351 cells). The previous files are in `_before_negmissing_20261108/`.
- Checked against the audit's own raw recomputation (`a23`, `a24`):
  - every station-day and cell-day matches (max |diff| 2e-5), as do `n_valid_hours`, `n_poc`, `n_stations` and `n_hours_min`;
  - 31 December matches;
  - feature columns are identical to the previous table on common rows, and the 16 new rows match the prediction table;
  - `verify_stage1.py` passes (34/34).
- F-03 (qualifier codes) remains open.
