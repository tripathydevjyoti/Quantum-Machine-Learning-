#!/bin/bash
#SBATCH --job-name=geom_fid5
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=2018
#SBATCH --qos=normal
#SBATCH --array=120-149%30
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --output=data_reupload/slurm/logs/geom_fid5_%A_%a.out
#SBATCH --error=data_reupload/slurm/logs/geom_fid5_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${QML_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "${QML_VENV:-${PROJECT_ROOT}/venv_req}/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -u \
  data_reupload/analysis/four_encoding_master/scripts/compute_exact_temporal_fidelity.py \
  --manifest_index "${SLURM_ARRAY_TASK_ID}" \
  --max_tau 40
