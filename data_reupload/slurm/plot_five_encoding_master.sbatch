#!/bin/bash
#SBATCH --job-name=enc5_plots
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=2018
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=data_reupload/slurm/logs/enc5_plots_%j.out
#SBATCH --error=data_reupload/slurm/logs/enc5_plots_%j.err

set -euo pipefail

PROJECT_ROOT="${QML_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "${QML_VENV:-${PROJECT_ROOT}/venv_req}/bin/activate"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

echo "======================================================================"
echo "FIVE-ENCODING MASTER FIGURE GENERATION"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "======================================================================"

python -u \
  data_reupload/analysis/four_encoding_master/scripts/plot_master_empirical_comparison.py

echo
echo "======================================================================"
echo "FIVE-ENCODING FIGURE GENERATION COMPLETE"
echo "Finished: $(date)"
echo "======================================================================"
