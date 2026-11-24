# Running Phase 2 on SciNet Trillium

Trillium CPU nodes: **192 cores, 755 GiB usable RAM**, Rocky Linux 9.6, Slurm.

Checked against the Trillium quickstart (19 Nov 2026):

| fact | status |
|---|---|
| CPU jobs get whole nodes; cores can't be requested | confirmed |
| `--mem` is ignored; every job gets the node's memory | confirmed |
| `compute` partition: min wall time 15 min, max 24 h | confirmed |
| `$HOME` and `$PROJECT` are read-only on compute nodes; job output must go to `$SCRATCH` | confirmed |
| Submit from `$SCRATCH`, since Slurm writes `%x-%j.out` to the submit directory | confirmed |
| No internet on compute nodes | confirmed (stated for debug jobs, assumed for batch jobs) |
| Quotas: `$HOME` 100 GB, `$SCRATCH` 25 TB (purge policy TBD), `$PROJECT` 1 TB default | confirmed; check yours with `diskusage_report` |
| `module load StdEnv/2023 python/3.11.5 arrow/23.0.1` | confirmed (quickstart examples) |
| `pip install --no-index` from the Alliance wheelhouse | **still to confirm**, with a PyPI fallback below |
| Slurm account | set `--account=def-<PI>` if you have more than one allocation |

The remaining `# confirm` marks below cover only the wheelhouse.

## What the code reads from the environment

| variable | used by | behaviour |
|---|---|---|
| `PM25_ROOT` | `data.ROOT` | project root; every path is derived from it |
| `SLURM_CPUS_PER_TASK` | `models.detect_cpus()` | thread shares; falls back to `os.cpu_count()` |
| `OMP_NUM_THREADS` | LightGBM / XGBoost internals | must match `threads per fit`, or the two settings fight |

`config.json` sets `n_jobs = -1` for all three models, meaning "all CPUs of the
allocation". The code divides that by `trial_workers * fold_workers`, so a
full-node job with 10 parallel folds gives each fit 19 threads.

## 1. Login-node setup

Getting the files onto Trillium: Globus (or scp) the project to the cluster. A full
copy of the workspace (~300 GB) belongs in `$PROJECT` as an archive, not on scratch.
Leave out `.venv`, `.wheels` and `.pylibs`, which hold Windows binaries. Then run
the commands below from inside that copy of `project/` to build the small working
copy on scratch. Or send only `code/` and the four files below straight to
`$SCRATCH/pm25`.

```bash
module load StdEnv/2023 python/3.11.5 arrow/23.0.1

export PM25_ROOT="$SCRATCH/pm25"          # compute nodes can write only to $SCRATCH
mkdir -p "$PM25_ROOT/logs"

# code (data.py imports utils.fingerprint from code/panel)
rsync -av code/ "$PM25_ROOT/code/"

# the four data files Phase 2 needs
mkdir -p "$PM25_ROOT/data/processed/target" "$PM25_ROOT/data/processed/grid" "$PM25_ROOT/data/interim/model"
rsync -av data/processed/target/target_table.parquet "$PM25_ROOT/data/processed/target/"
rsync -av data/processed/grid/cells.parquet         "$PM25_ROOT/data/processed/grid/"
rsync -av data/interim/model/splits.parquet         "$PM25_ROOT/data/interim/model/"
rsync -av data/interim/model/folds.parquet          "$PM25_ROOT/data/interim/model/"
```

Build the venv once, on a login node, pinned in `code/model/requirements.txt`:
numpy 2.4.2, pandas 3.0.5, pyarrow 23.0.1, scipy 1.17.1, scikit-learn 1.8.0,
lightgbm 4.6.0, xgboost 3.2.0, optuna 4.9.0:

```bash
python -m venv "$HOME/venvs/pm25"
source "$HOME/venvs/pm25/bin/activate"
pip install --upgrade pip
# confirm: --no-index works with the Alliance wheelhouse
pip install --no-index -r "$PM25_ROOT/code/model/requirements.txt" \
  || pip install -r "$PM25_ROOT/code/model/requirements.txt"
```

`requirements.txt` is pinned to what the Alliance wheelhouse provides (checked
20 Nov 2026). It differs from the local pins in `versions.txt` at patch or minor level
only: numpy 2.4.2, pandas 3.0.5, pyarrow 23.0.1, scikit-learn 1.8.0. The PyPI fallback
doesn't work as written, because Alliance pip searches only the wheelhouse, but the
login node can reach PyPI (`--index-url https://pypi.org/simple`) if ever needed.
Record what was actually built:

```bash
pip freeze > "$PM25_ROOT/logs/pip_freeze.txt"
```

Put the venv activation in every batch script; compute nodes start from a clean
environment. The venv lives in `$HOME`, which compute nodes can read, so that's fine.

Write job scripts on Trillium, or run `dos2unix job*.sh` on scripts made on Windows.
A script with Windows line endings fails with `\r: command not found`.

## 2. Batch script template (one full node, one command)

```bash
#!/bin/bash
#SBATCH --job-name=pm25-tune-lgbm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192        # whole node; Trillium allocates nothing smaller
#SBATCH --time=24:00:00            # compute-partition maximum (minimum is 15 min)
#SBATCH --output=%x-%j.out         # lands in the submit directory: submit from $SCRATCH/pm25
##SBATCH --account=def-<PI>        # uncomment if you have more than one allocation

set -euo pipefail

module load StdEnv/2023 python/3.11.5 arrow/23.0.1
source "$HOME/venvs/pm25/bin/activate"

export PM25_ROOT="$SCRATCH/pm25"
# flush every print at once; otherwise output piped to tee sits in a buffer and is
# lost if the job is killed at the wall time
export PYTHONUNBUFFERED=1

# threads per fit = cpus / (trial_workers * fold_workers); keep FOLD_WORKERS in step
# with --fold-workers below. Each fit also sets its own thread count, which overrides
# this, so the export is a default rather than the source of truth.
FOLD_WORKERS=10
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / FOLD_WORKERS))
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

mkdir -p "$PM25_ROOT/logs"
cd "$PM25_ROOT"

python code/model/tune.py --model lgbm --fold-workers "$FOLD_WORKERS" --workers 1 \
    2>&1 | tee "$PM25_ROOT/logs/tune_lgbm_${SLURM_JOB_ID}.log"
```

Thread shares for the schemes in use:

| scheme | folds | `--fold-workers` | threads per fit (192 cores) |
|---|---:|---:|---:|
| grouped 10-fold | 10 | 10 | 19 |
| leave-one-region-out | 6 | 6 | 32 |
| leave-one-year-out | 14 | 14 | 13 |

`--fold-workers` defaults to one per fold, capped by the CPUs, so the flag is only
needed to leave headroom. `--workers` above 1 runs trials concurrently; keep it at
1 for tuning, where sequential TPE is the point. For `runs.py`, drop
`--fold-workers` entirely: the default gives each scheme one worker per fold, whereas
10 would run leave-one-year-out's 14 folds as 10 and then 4.

Submit with `cd $SCRATCH/pm25 && sbatch job.sh`.

## 2b. The submit script (what is actually used)

`code/model/slurm/submit_all.sh` submits the full Phase 2 as separate jobs through
the generic `slurm/pm25_job.sh`:
- **Tuning chunks:** LightGBM in 4 chunks of 25 trials, XGBoost in 2 chunks of 25. Each
  model's chunks are chained, never run at once. The forest grid is one job.
- **Runs:** 12 jobs, one per run × CV scheme (R1–R4 × group10/loro/loyo), each
  starting when its model's tuning finishes.
- **Summary:** one job after all 12.

The three models' chains run side by side. Each tuning chunk restarts the TPE sampler
with its seed offset by the trials already finished, so a chunk or a resubmit never
re-proposes earlier trials. Wall times and chunk sizes are environment variables at
the top of the script.

## 3. Job sequence (one job each)

| # | command | why separate |
|---:|---|---|
| 1 | `python code/model/leakage_test.py` | cheap proof the folds are clean; writes `leakage_test.log` |
| 2 | quick pass: `tune.py --model lgbm --quick`, `--model xgb --quick`, **`--model rf --quick`**, `runs.py R1 --quick`, `R2 --quick`, `R3 --quick`, **`R4 --quick`**, `summarise.py --quick` | proves the pipeline end to end in minutes, including the Random Forest path that has never executed anywhere |
| 3 | timing: two full-size LightGBM fits (see §4) | sets the real budget for jobs 4–6 |
| 4 | `python code/model/tune.py --model lgbm` | 100 trials |
| 5 | `python code/model/tune.py --model xgb` | 50 trials |
| 6 | `python code/model/tune.py --model rf` | 12-point grid, cached; a single fit uses every thread, so its folds are the parallel unit like everywhere else |
| 7 | `runs.py R1`, `R2`, `R3`, `R4` | ~120 fits total |
| 8 | `summarise.py` | writes `results_dev.md` |

Jobs 1–3 are short. They can run as batch jobs with `--time=01:00:00` (the minimum is
15 min), or interactively in a `debugjob`: a full node for 60 minutes from a login
node, which usually starts right away. In a debug job, `cd $SCRATCH/pm25`, load the
modules, activate the venv, export `PM25_ROOT`, and run the commands directly. Don't
run them on the login node itself; it's limited to a few minutes and 1–4 cores.

## 4. Timing before committing to the full tuning

The learning rate is searched over **0.06–0.3** on a log scale. The range starts at
0.06 because anything below that needs more than the 5,000-round cap to converge, so
those trials would be truncated rather than judged. Time one whole trial at each end,
exactly as tuning runs it: 10 folds at once, 19 threads each. A single fit on all 192
threads would understate the time per trial.

```bash
python - <<'PY'
import sys, time, pandas as pd
sys.path.insert(0, "code/model")
import data as D, models, tune
cfg = D.load_config(); d = D.load()
splits = pd.read_parquet(D.OUT / "splits.parquet")
models.set_parallel(1, 10)
for lr in (0.06, 0.3):
    t0 = time.time()
    mean, per_fold, iters, not_stopped = tune.evaluate(
        "lgbm", {"learning_rate": lr}, "A", cfg, d, splits, False, trial=None, fold_workers=10)
    print(f"lr={lr}: trial {time.time()-t0:.0f}s, rmse {mean:.3f}, trees {sorted(iters)}, "
          f"folds at cap {not_stopped}, cpus {models.detect_cpus()}, "
          f"threads/fit {models.resolve_threads(cfg, 'lightgbm')}", flush=True)
PY
```

Two things to read from it:
- **Does the slow end converge?** "folds at cap" should be 0 at `lr=0.06`.
- **The trial time at `lr=0.06`.** That time × 100 trials is the worst-case budget for job 4. If it exceeds about 20 h, job 4 needs a resubmit, which is safe (§5). Other parameters (e.g. `num_leaves`) change the time too, so treat this as an estimate.

## 5. Resubmitting after the wall time

A job that hits 24 h is resubmitted **unchanged**, which is safe because every stage
resumes:

- Optuna counts finished trials towards the total, and the study lives in
  `data/interim/model/optuna/studies_<model>.db` (one file per model), keyed by model, dataset, mode, config hash
  and code hash;
- `cv.run_cv` skips folds whose `fold_<label>.parquet` already exists;
- Random Forest grid points are cached one JSON file each.

**One SQLite file per model.** LightGBM and XGBoost each have their own
`studies_<model>.db`, and the Random Forest grid uses JSON files, so jobs 4, 5 and 6
can run **at the same time**. What must not happen is two jobs tuning the **same**
model at once.

## 6. After the runs

Copy back before scratch is purged (Trillium's purge policy is still "TBD", so do it
promptly), with Globus or rsync:

```bash
rsync -av "$SCRATCH/pm25/data/interim/model/" data/interim/model/
```

That directory holds the parameter files, OOF files, metrics, `results_dev.md`, the
Optuna database and the run keys. The splits and folds are unchanged; do not copy
them back over the frozen ones.

## Not to do

- Don't run two tuning jobs for the same model at once.
- Don't edit `config.json` between jobs 4 and 7: the config hash is in the study
  name and in every run key, so a changed config starts fresh studies and refuses
  the old runs.
- Don't copy `data/processed/` back from scratch; the master copy is authoritative.