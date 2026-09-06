#!/bin/bash
#SBATCH --job-name=report_assets
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=general
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=data_reupload/slurm/logs/report_assets_%j.out
#SBATCH --error=data_reupload/slurm/logs/report_assets_%j.err

set -euo pipefail

PROJECT_ROOT="${QML_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"

module purge
module load Python/3.11.5-GCCcore-13.2.0

source "${QML_VENV:-${PROJECT_ROOT}/venv_req}/bin/activate"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

python -u \
  data_reupload/report/final_ideal_noisy/scripts/build_ideal_noisy_report_assets.py
