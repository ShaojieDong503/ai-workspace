# Phase 2 review (round 5): launch readiness on Trillium

**Method:** a code audit of `config.json`, `models.py`, `cv.py`, `tune.py`, `runs.py`, `score_test.py`, `summarise.py` and `SCINET.md` (19 Nov, 15:59–16:00), and the merged §7 of `MODEL_RUN_PLAN.md`, against `docs/PROMPT_phase2_finish.md`. I ran nothing.

> **For the coding agent:** code and doc changes only, no local runs.

**Verdict: close.** All seven items from the finish prompt are in. Before the first `sbatch`: one decision for the author (L1), two small fixes (L2, L3), and the Trillium confirmations. None needs another design round.

## The finish prompt: status

| Item | Status | Where |
|---|---|---|
| 1. Parallel folds | **Done.** `cv.fit_folds` uses threads and returns results in fold order. Used by `tune.evaluate`, `cv.run_cv` and the Random Forest grid. Pruning is used only when folds run serially. `constant_liar` is set when trials run in parallel. Parallelism and threads per fit are recorded. | `cv.py` `fit_folds`, `run_cv`; `tune.py` `evaluate`, `tune_optuna` |
| 2. `n_jobs = -1` | **Done** (all three models). Threads per fit = CPUs / (trials × folds): 19 for grouped 10-fold, 32 for leave-one-region-out, 13 for leave-one-year-out. | `config.json`, `models.resolve_threads` |
| 3. `SCINET.md` | **Done.** Login setup, full-node template, job sequence, timing at 0.03 and 0.2, resubmission, one SQLite per process, copy-back. Assumptions are marked "confirm". | `SCINET.md` |
| 4. R1 vs R2 on the untuned schemes | **Done.** Leave-one-region-out and leave-one-year-out first, grouped 10-fold "for reference only". | `summarise.py` |
| 5. Pinned 10 km subset | **Done.** `FAR_CELLS` asserted. | `score_test.py` |
| 6. Never overwrite test results | **Done.** Refuses unless `--supersede --reason`, keeps old files with a timestamp, writes `test_log.jsonl`. | `score_test.py` |
| 7. §7 merged | **Done.** 7.0 keeps the groups and metrics, 7.1–7.11 are added, and the section is dated. | `MODEL_RUN_PLAN.md` §7 |

Fold-order collection, atomic fold writes, the run key with code hash, `leak_check` before metrics, and the `OPEN_TEST` guard all survive the parallel rewrite.

## Before launch

### L1. The low end of the learning-rate range can't finish within the cap (author decision)

`SCINET.md` §4 says it outright: at `lr = 0.03`, LightGBM "needs roughly 8,000 trees, so it will" hit the 5,000 cap. The one measured fit at 0.05 already reached 4,956 of 5,000.
- **Effect on the search:** any trial with a learning rate below about 0.06 is a **truncated** model. TPE will mark that region as worse because of the cap, not because of the learning rate.
- **Effect on the result:** if the chosen trial lands there, `n_trees_final` is just the cap.

**Choose one before job 4:**
- **(a) Narrow the range to 0.06–0.3.** Recommended. The fits stay fast and the low end can actually stop. Nothing is lost: boosting accuracy is flat across that range once trees are early-stopped.
- **(b) Raise the cap to 10,000.** This keeps 0.03, but doubles the worst-case fit time, and the slow end then dominates the 24 h budget.

Whichever you choose, the config hash changes, so the studies start fresh. That's fine, because nothing has run yet.

### L2. The environment isn't actually pinned (`requirements.txt`)

`SCINET.md` installs from `code/model/requirements.txt`, but that file has `lightgbm>=4.3`, `xgboost>=2.0` and `optuna>=4.0`. The exact versions the code was written against are only in `versions.txt`: lightgbm 4.6.0, xgboost 3.2.0, optuna 4.9.0.

**Fix:**
- Pin these three in `requirements.txt` to `versions.txt`.
- Add a fallback line in `SCINET.md`: if `pip install --no-index` can't find a pinned wheel in the Alliance wheelhouse, install that package from PyPI on the login node, which has internet.
- Record the installed versions on Trillium (`pip freeze > $PM25_ROOT/logs/pip_freeze.txt`). Every `run_meta.json` already stores the library versions.

### L3. The quick pass skips Random Forest, which has never run anywhere (`SCINET.md` §3, job 2)

Job 2 covers lgbm, xgb and R1–R3, but not `tune.py --model rf --quick` or `runs.py R4 --quick`. The Random Forest path couldn't run in the Windows sandbox, so it has never executed: the fit with NaN inputs, `n_jobs` inside threads, the grid cache, and `choose_trees` for R4. The first time it ran would be the full 120-fit grid.

**Fix:** add both commands to job 2.

## Minor

- **`OMP_NUM_THREADS=19` is hard-coded in the template.** That's right for grouped 10-fold, but leave-one-region-out uses 32 threads per fit and leave-one-year-out 13. Each fit sets `num_threads` / `nthread` / `n_jobs` explicitly, and those override the environment, so the effect is small. Either compute it in the script (`$((SLURM_CPUS_PER_TASK / FOLD_WORKERS))`) or say the variable is only a default.
- **Pruned trials no longer count towards the total.** `remaining` counts `t.value is not None`. Pruned trials have no value, so with serial folds and pruning on, a resubmitted job runs more trials than requested. It doesn't matter for the default fold-parallel mode, where pruning is off. Count `COMPLETE` and `PRUNED` states, as the earlier version did.

## How far from launch

| Step | Who | Size |
|---|---|---|
| L1 learning-rate range or cap | Author decides, agent edits one config line | minutes |
| L2 pin versions, pip fallback | Agent | minutes |
| L3 Random Forest in the quick pass | Agent, edits `SCINET.md` | minutes |
| Trillium confirmations (whole node, 24 h, writable filesystem, modules, `--no-index`) | Author, from the quickstart | one read |
| Copy files and build the venv | Author on Trillium | under an hour |
| Jobs 1–3: leakage test, quick pass, timing | Trillium | short |

After the quick pass and the timing come back clean, jobs 4–8 (full tuning and runs) can be submitted. The timing result decides whether R1's 100 trials fit in one 24 h job or need a resubmit, which the code already handles.

---

## Follow-up audit (19 Nov, after the 16:08–16:10 changes)

| Item | Status |
|---|---|
| L1, learning-rate range | **Fixed, option (a).** `learning_rate_range` is 0.06–0.3 for LightGBM and XGBoost. The cap stays at 5,000. |
| L2, pinned environment | **Fixed.** `requirements.txt` pins all eight packages to `versions.txt`. `SCINET.md` falls back to PyPI on the login node if `--no-index` fails, and records `pip freeze`. |
| L3, Random Forest in the quick pass | **Fixed.** Job 2 now includes `tune.py --model rf --quick` and `runs.py R4 --quick`. |
| `OMP_NUM_THREADS` | **Fixed.** Derived as `SLURM_CPUS_PER_TASK / FOLD_WORKERS`, and documented as a default that each fit overrides. |
| Pruned trials and the total | **Fixed.** `remaining` counts `COMPLETE` and `PRUNED`. |

Two doc details, neither blocking:
- **Stale timing text.** `SCINET.md` §4, lines 150–151, still says "whether `lr=0.03` reaches the 5,000-round cap (it needs roughly 8,000 trees, so it will)". The snippet now times 0.06 and 0.3. Reword it: "whether `lr=0.06` stops before the cap".
- **`--fold-workers` in the runs.** The template passes `--fold-workers "$FOLD_WORKERS"` (10). For `runs.py`, leave the flag out: the default picks one worker per fold per scheme (6 for leave-one-region-out, 14 for leave-one-year-out). Passing 10 would run leave-one-year-out's 14 folds as 10 then 4.

**Verdict: the code is ready to launch.** What remains is on the author's side: the Trillium confirmations, copying the files, building the venv, then jobs 1–3 (leakage test, quick pass, timing).
