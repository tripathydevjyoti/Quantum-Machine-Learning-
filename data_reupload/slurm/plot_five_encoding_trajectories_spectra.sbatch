#!/bin/bash
#SBATCH --job-name=enc5_traj
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=2018
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=data_reupload/slurm/logs/enc5_traj_%j.out
#SBATCH --error=data_reupload/slurm/logs/enc5_traj_%j.err

set -euo pipefail

PROJECT_ROOT="/umbc/rs/pi_deffner/users/devjyot1/projects/Quantum-Machine-Learning-"
cd "${PROJECT_ROOT}"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source venv_req/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

echo "======================================================================"
echo "FIVE-ENCODING TRAJECTORY / SPECTRAL FIGURE GENERATION"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "======================================================================"

python -u \
  data_reupload/analysis/four_encoding_master/scripts/plot_master_dataset_trajectories_spectra.py

echo
echo "======================================================================"
echo "FIVE-ENCODING TRAJECTORY / SPECTRAL FIGURES COMPLETE"
echo "Finished: $(date)"
echo "======================================================================"
