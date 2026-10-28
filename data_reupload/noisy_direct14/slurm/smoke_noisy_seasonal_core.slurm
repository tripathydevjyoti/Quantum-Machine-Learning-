#!/bin/bash
#SBATCH --job-name=seasonal_core
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=data_reupload/noisy_direct14/logs/seasonal_core_%j.out
#SBATCH --error=data_reupload/noisy_direct14/logs/seasonal_core_%j.err

set -euo pipefail

PROJECT_ROOT="${QML_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

cd "$PROJECT_ROOT"

module purge
module load Python/3.11.5-GCCcore-13.2.0

source "${QML_VENV:-${PROJECT_ROOT}/venv_noisy_qml}/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python -u \
  data_reupload/noisy_direct14/scripts/smoke_test_noisy_seasonal_core.py
