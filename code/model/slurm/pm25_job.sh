#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --output=logs/%x-%j.out
# One full Trillium node, one command. The command is passed after the script name:
#   sbatch --job-name=R1-loro --time=02:00:00 code/model/slurm/pm25_job.sh \
#          python code/model/runs.py R1 --scheme loro
# Submit from $PM25_ROOT, so logs/ resolves there. Job name, wall time, dependency
# and OMP_NUM_THREADS come from submit_all.sh.

set -euo pipefail

module load StdEnv/2023 python/3.11.5 arrow/23.0.1
source "$HOME/venvs/pm25/bin/activate"

export PM25_ROOT="${PM25_ROOT:-$SCRATCH/pm25}"
export PYTHONUNBUFFERED=1
cd "$PM25_ROOT"

echo "start $(date -Is)  host $(hostname)  job ${SLURM_JOB_ID}  name ${SLURM_JOB_NAME}"
echo "cmd: $*"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}  cpus=${SLURM_CPUS_PER_TASK:-?}"
"$@"
echo "end $(date -Is)"
