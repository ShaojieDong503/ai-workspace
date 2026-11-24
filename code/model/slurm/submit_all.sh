#!/bin/bash
# Submit the whole Phase 2 pipeline as separate jobs, chained by dependencies.
#
#   tune lgbm --> R1 x {group10, loro, loyo}, R2 x {group10, loro, loyo} --+
#   tune xgb  --> R3 x {group10, loro, loyo}                               +--> summarise
#   tune rf   --> R4 x {group10, loro, loyo}                               --+
#
# The three tuning jobs start as soon as the scheduler allows, side by side: each
# model has its own Optuna database (studies_<model>.db) and the forest grid uses
# JSON files. Each run x scheme job writes only its own folder, full/<run>/<scheme>/.
#
# Usage, from anywhere on a login node:
#   bash $SCRATCH/pm25/code/model/slurm/submit_all.sh
# Wall times can be overridden without editing, e.g.
#   T_TUNE_LGBM=03:00:00 T_RUN=01:00:00 bash .../submit_all.sh
# Trial totals and chunk sizes too: N_LGBM=100 CHUNK_LGBM=25 N_XGB=50 CHUNK_XGB=25
#
# If a tuning job hits its wall time, its dependents never start (afterok). Cancel
# them, resubmit the tuning job unchanged (it resumes), then resubmit the run jobs
# with the one-line commands this script prints at the end.

set -euo pipefail

export PM25_ROOT="${PM25_ROOT:-$SCRATCH/pm25}"
cd "$PM25_ROOT"
mkdir -p logs
JOB=code/model/slurm/pm25_job.sh

# wall times: placeholders until the timing job has run; see SCINET.md section 4
T_TUNE_LGBM="${T_TUNE_LGBM:-06:00:00}"   # per chunk of 25 trials
T_TUNE_XGB="${T_TUNE_XGB:-06:00:00}"     # per chunk of 25 trials
T_TUNE_RF="${T_TUNE_RF:-08:00:00}"
T_RUN="${T_RUN:-03:00:00}"
T_SUM="${T_SUM:-00:30:00}"
ACCOUNT_OPT=()
[ -n "${ACCOUNT:-}" ] && ACCOUNT_OPT=(--account="$ACCOUNT")
# MAIL=you@example.org: an email when any job fails, and when summarise ends
MAIL_OPT=()
[ -n "${MAIL:-}" ] && MAIL_OPT=(--mail-user="$MAIL" --mail-type=FAIL)

# threads per fit = 192 / folds running at once; each fit also sets this itself
omp_for() { case "$1" in group10) echo 19 ;; loro) echo 32 ;; loyo) echo 13 ;; esac; }

sub() { sbatch --parsable "${ACCOUNT_OPT[@]}" "${MAIL_OPT[@]}" "$@" | cut -d';' -f1; }

# Tuning in chunks: `--trials N` brings the study up to N finished trials, so chunk k
# asks for k*CHUNK. Chunks of one model are chained (afterok), never concurrent: they
# share that model's SQLite file, and TPE must see each chunk's results. Short jobs
# start sooner (backfill). A failed chunk stops the rest of its chain.
tune_chain() {   # model total chunk walltime -> prints the last chunk's job id
  local model=$1 total=$2 chunk=$3 t=$4 prev="" id n
  for ((n = chunk; n < total + chunk; n += chunk)); do
    (( n > total )) && n=$total
    local dep=(); [ -n "$prev" ] && dep=(--dependency=afterok:"$prev")
    id=$(sub --job-name="tune-$model-$n" --time="$t" "${dep[@]}" \
         --export=ALL,OMP_NUM_THREADS=19 \
         "$JOB" python code/model/tune.py --model "$model" --trials "$n")
    echo "  tune $model to $n trials: $id${prev:+ (after $prev)}" >&2
    prev=$id
  done
  echo "$prev"
}

N_LGBM="${N_LGBM:-100}"; CHUNK_LGBM="${CHUNK_LGBM:-25}"
N_XGB="${N_XGB:-50}";    CHUNK_XGB="${CHUNK_XGB:-25}"
LGBM=$(tune_chain lgbm "$N_LGBM" "$CHUNK_LGBM" "$T_TUNE_LGBM")
XGB=$(tune_chain xgb "$N_XGB" "$CHUNK_XGB" "$T_TUNE_XGB")
# the forest grid is 12 cached points; one job, resubmitted unchanged if it times out
RF=$(sub --job-name=tune-rf --time="$T_TUNE_RF" --export=ALL,OMP_NUM_THREADS=19 \
     "$JOB" python code/model/tune.py --model rf)
echo "tuning done when: lgbm $LGBM  xgb $XGB  rf $RF"

declare -A AFTER=([R1]=$LGBM [R2]=$LGBM [R3]=$XGB [R4]=$RF)
RUN_IDS=()
for run in R1 R2 R3 R4; do
  for scheme in group10 loro loyo; do
    id=$(sub --job-name="$run-$scheme" --time="$T_RUN" \
         --dependency=afterok:"${AFTER[$run]}" \
         --export=ALL,OMP_NUM_THREADS="$(omp_for "$scheme")" \
         "$JOB" python code/model/runs.py "$run" --scheme "$scheme")
    RUN_IDS+=("$id")
    echo "  $run $scheme: $id (after ${AFTER[$run]})"
  done
done

DEP=$(IFS=:; echo "${RUN_IDS[*]}")
SUM_MAIL=(); [ -n "${MAIL:-}" ] && SUM_MAIL=(--mail-type=END,FAIL)
SUM=$(sub --job-name=summarise --time="$T_SUM" --dependency=afterok:"$DEP" "${SUM_MAIL[@]}" \
      "$JOB" python code/model/summarise.py)
echo "summarise: $SUM"

cat <<EOF

Submitted $((N_LGBM / CHUNK_LGBM + N_XGB / CHUNK_XGB + 14)) jobs. Watch them with:  squeue -u \$USER
Logs: $PM25_ROOT/logs/<job-name>-<jobid>.out

Resubmitting one job by hand (from $PM25_ROOT):
  sbatch --job-name=tune-lgbm-50 --time=$T_TUNE_LGBM $JOB python code/model/tune.py --model lgbm --trials 50
  sbatch --job-name=R1-loro   --time=$T_RUN $JOB python code/model/runs.py R1 --scheme loro
  sbatch --job-name=summarise --time=$T_SUM $JOB python code/model/summarise.py
EOF
