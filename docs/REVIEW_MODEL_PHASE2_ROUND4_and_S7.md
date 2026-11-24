# Review: code (round 4) and the §7 reporting draft

**Method:** a code audit of the files changed since round 3 (`config.json`, `models.py`, `tune.py`, `cv.py`, `score_test.py`, `summarise.py` and `SCINET.md`, dated 19 Nov, 14:58–15:25), plus a read of `docs/PLAN_S7_REPORTING_DRAFT.md` against the plan and the code. I ran nothing.

> **For the coding agent:** code changes only, with no local runs. Everything runs on SciNet.

**Verdict:**
- **Code:** C1, C2 and the earlier items are fixed. There's one new bug that wastes most of the cores on SciNet (S1), and three SciNet set-up problems (S2–S4).
- **§7 draft:** sound, and it matches the code. It needs three additions before it's frozen, and one fix to how it slots into the plan (R1–R5 below).

---

## Part 1: code

### Checked and fine

| Item | Status |
|---|---|
| C1, `folds_not_stopped` counted rows | Fixed. Counted per fold, and skipped for Random Forest. |
| C2, final tree count weighted by rows | Fixed. Median of per-fold values, which are recorded. |
| Code hash in `run_key` and the study name | Done. `cv.code_hash()` covers `models.py`, `cv.py`, `data.py`, `folds.py` and `features.py`. |
| Tuning resume counts trials towards the total | In place. |
| Learning rate searched on a log scale, 0.03–0.2, and the tuned value used downstream | In place. `lgbm_params` / `xgb_params` merge tuned values over the config defaults. |
| `score_test.py`: per-cell table, the more-than-10-km subset, coverage per group | Implemented, and matches draft rules 7.2 and 7.3. |
| `summarise.py`: grouped 10-fold labelled, leave-one-year-out per year and with/without 2023, R1 vs R2 by `aod_missing` | Implemented (but see R4 for the scheme it uses). |

### S1. `--workers` splits the threads but still runs trials one at a time (`tune.tune_optuna`)

`models.set_workers(args.workers)` divides each fit's threads by `workers`, but `study.optimize(objective, n_trials=remaining)` has no `n_jobs`. So `--workers 4` on 8 CPUs runs **one** trial at a time with 2 threads, and 6 of the 8 cores sit idle. That's the batch script's exact recipe in `SCINET.md`.

**Fix:**
- Use `study.optimize(objective, n_trials=remaining, n_jobs=workers)`.
- Optuna then runs trials in threads within one process, which keeps to the one-process-per-SQLite rule. LightGBM and XGBoost release the GIL, so the threads do run in parallel.
- Note in the report that parallel TPE isn't bit-reproducible: trial order depends on timing.

**Same issue for Random Forest:** `tune.py --model rf --workers N` divides Random Forest's `n_jobs` by N, but the grid runs serially. For `rf`, ignore `--workers` (call `set_workers(1)`), or run grid points in parallel.

### S2. `n_jobs = 8` caps every fit at 8 threads (`config.json`)

`resolve_threads` takes `min(n_jobs, cpus)`. If the SciNet cluster allocates whole nodes, every fit still uses at most 8 threads, and with `--workers` even fewer. Niagara has 40 cores per node and Trillium 192. Check which cluster this runs on.

**Fix:** set `n_jobs = -1` (all CPUs the allocation gives), and let `--cpus-per-task` and `--workers` control parallelism.

### S3. `SCINET.md` is missing the set-up the code depends on

- **`PM25_ROOT`:** not mentioned. The script relies on `cd "$SLURM_SUBMIT_DIR"`, which only works if the job is submitted from the project root. Export `PM25_ROOT` explicitly.
- **Where files live:** on SciNet's large clusters, compute nodes can't write to `$HOME` or `$PROJECT`. Check this for the cluster in use. If it applies, the project copy, `data/interim/model/` (splits, `studies.db`, OOF files) and `logs/` must live on `$SCRATCH`, and results are copied back afterwards. Today the script writes into the submit directory.
- **`OMP_NUM_THREADS`:** not set. Export it as `$SLURM_CPUS_PER_TASK / workers`, so OpenMP inside LightGBM and XGBoost matches `resolve_threads`.
- **Python version:** `module load python/3.11` is fine, since pandas 3 and numpy 2.4 need 3.11 or later. But the environment must be pinned to `versions.txt` (pandas 3.0.3, numpy 2.4.6, lightgbm 4.6.0, xgboost 3.2.0, scikit-learn 1.9.0, optuna 4.9.0). Add the `python -m venv` and `pip install` steps, run on a login node.
- **Whole-node clusters:** on clusters that allocate whole nodes, `--cpus-per-task=8` and `--mem=32G` may be ignored or rejected. Give the script for the actual cluster.

### S4. The timing snippet in `SCINET.md` measures the wrong thing

It times one fit with `{}`, which means the default learning rate of 0.05. Now that the learning rate is searched over 0.03–0.2, time two fits, at 0.03 and at 0.2. The slow end decides the tuning budget and whether the 5,000 cap is reached.

### Minor

- **The 10 km subset is recomputed at scoring time.** `score_test` derives the subset from `FAR_KM` and the grid, which is deterministic. Also assert that it equals the pre-registered list `{910214, 895350, 958101, 924048}`, so a change to the grid or the splits can't silently change the subset.
- **Re-running `score_test.py` overwrites earlier results.** It writes `test_results_{run}.csv` and `test_per_cell_{run}.csv` unconditionally. Draft rule 7.8 says later results are "superseded, not overwritten". Refuse if the files exist, unless `--supersede`, which keeps the old files with a timestamp.

---

## Part 2: `PLAN_S7_REPORTING_DRAFT.md`

Rules 7.1–7.8 are right, and the code implements them (7.2 per cell and the 10 km subset, 7.5 the label, 7.6 the per-year table and with/without 2023, 7.7 the AOD split). Before freezing:

### R1. Don't replace the current §7; extend it

The plan's §7 is "Reported groups (every run)": the six groups, the metrics and the development counts. The draft is titled "## 7. Reporting rules" and would overwrite it if pasted as-is. Either:
- keep the groups as **7.0**, and add the rules as 7.1–7.8 under a combined heading "7. Reporting", or
- make the rules **§8**, and renumber "Open decisions" to §9.

### R2. Say which models are scored on the test sets

Rule 7.8 says "once per chosen model", but nothing says which models are chosen. If it's decided after the development results, the test set ends up deciding which models appear. **Add:** *"All four runs (R1–R4) are finalised and scored, once each. No run is dropped after its test score is seen."*

### R3. The contingency decision comes before the test sets open

X1–X3 are judged by hand from R1's development results. That decision, and the note giving its reasons, must be **recorded before `OPEN_TEST` is created**. Otherwise a disappointing test score could trigger extra runs. **Add as 7.9.**

### R4. Compare R1 and R2 on schemes where neither was tuned (7.7)

`summarise.aod_split` defaults to grouped 10-fold. R1's parameters were tuned on those folds for dataset A, and R2 reuses them on dataset C. So grouped 10-fold favours R1, and the A-vs-C gap there is inflated.

**Rule:** report R1 vs R2, overall and by `aod_missing`, on **leave-one-region-out and leave-one-year-out**, with grouped 10-fold for reference only. In the code: call `aod_split` for `loro` and `loyo` too.

### R5. Pre-register the interval rules

- **Rule:** the 90% interval is calibrated once, on the calibration cells (2010–2023), and isn't recalibrated after the test sets are seen.
- **Report:** coverage for each test set, per group and per region.
- **Caveat:** in the 2024 test, the interval is applied to a later year than it was calibrated on, so coverage there also tests whether the interval holds over time. Say so.

### Wording

7.3's "about 29 cells" for 2024: README §3 gives 29 Ontario cells active in 2024. Use the exact number.

---

## Part 3: Trillium specifics (author confirmed the cluster, 19 Nov)

**Confirmed** from the SciNet Trillium page: CPU nodes have **192 cores and 768 GB RAM**, run Rocky Linux 9, and use Slurm.

**To check in the Trillium quickstart** (the page blocks automated reading):
- whole-node scheduling for CPU jobs
- a 24 h wall-time limit
- `$HOME` / `$PROJECT` read-only from compute nodes, with `$SCRATCH` writable and subject to purging
- no internet on compute nodes
- Python through the Alliance software stack (`module load StdEnv python`, then a venv, and `pip install --no-index` where wheels exist)

The points below assume these hold.

**This settles S2 and S3:**
- **S2:** set `n_jobs = -1`. A whole node is allocated anyway, and a cap of 8 would use 4% of it.
- **S3:**
  - Request a full node (`--nodes=1`, all 192 cores), not `--cpus-per-task=8 --mem=32G`.
  - Keep `PM25_ROOT`, `data/interim/model/` and `logs/` on `$SCRATCH`, and copy results back afterwards.
  - Build the venv on a login node.

**How to use 192 cores without spoiling the search:**
- **The data is small.** About 145k rows and 62 columns. One LightGBM or XGBoost fit gains little beyond about 8–16 threads, so the node has to run several fits at once.
- **Don't run dozens of trials in parallel.** TPE learns from finished trials. With 48 trials running at once and 100 in total, most trials are proposed before any results are back, which makes it close to random search.
- **Parallelise the folds instead, recommended:**
  - Fit the 10 folds of a trial concurrently, with about 19 threads each (192 / 10), and run trials one after another, or two at a time.
  - Each trial then takes roughly one fit's time, and TPE stays sequential.
  - Trade-off: the median pruner currently decides after each fold, in order. With parallel folds it can only prune after all 10. Either accept that and drop pruning, or report folds as they finish. The fold-parallel speed-up is far larger than what pruning saves.
- **If trials stay the parallel unit (S1's fix):**
  - Keep `--workers` modest (4–8).
  - Set `TPESampler(constant_liar=True)`, so running trials are accounted for when new ones are proposed.
  - Record `workers` in `params_*.json` (already done).
- **Random Forest:** a single fit can use all 192 cores (`n_jobs = -1`). The grid can stay serial.
- **Memory** isn't a constraint at 768 GB, even with 10 copies of a fold's training frame.

**Wall time:** with the folds in parallel, R1's 100 trials should fit in one 24 h job, depending on the slow end of the learning-rate range (S4). Resume already handles a time-out.

## Actions

| Who | Action |
|---|---|
| Coding agent (code only) | S1 (`n_jobs=workers`; Random Forest ignores `--workers`), S2 (`n_jobs = -1`), S3 and S4 (`SCINET.md`), the minor items (pinned 10 km subset; `--supersede`), R4 (`aod_split` on `loro` and `loyo`) |
| Author | Cluster confirmed: **Trillium**. Check the Part 3 assumptions in the Trillium quickstart |
| Coding agent | Parallelise the folds within a trial (Part 3), or S1 with `constant_liar=True` and 4–8 workers. Write a full-node Trillium batch script in `SCINET.md` |
| Plan | Paste the §7 draft with R1–R5 applied, before `OPEN_TEST` exists |
