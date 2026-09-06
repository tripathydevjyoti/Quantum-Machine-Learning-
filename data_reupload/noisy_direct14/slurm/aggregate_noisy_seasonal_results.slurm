#!/bin/bash
#SBATCH --job-name=aggregate_noisy
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=data_reupload/noisy_direct14/logs/aggregate_noisy_%j.out
#SBATCH --error=data_reupload/noisy_direct14/logs/aggregate_noisy_%j.err

set -euo pipefail

QML_REPO_ROOT="${QML_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

cd "$QML_REPO_ROOT"

module purge
module load Python/3.11.5-GCCcore-13.2.0

source "${QML_VENV:-${QML_REPO_ROOT}/venv_req}/bin/activate"

python -u \
  data_reupload/noisy_direct14/scripts/aggregate_noisy_seasonal_results.py
