#!/bin/bash
#SBATCH --job-name=geom_fid_agg
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=2018
#SBATCH --qos=normal
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=data_reupload/slurm/logs/geom_fid_agg_%j.out
#SBATCH --error=data_reupload/slurm/logs/geom_fid_agg_%j.err

set -euo pipefail

cd /umbc/rs/pi_deffner/users/devjyot1/projects/Quantum-Machine-Learning-

module purge
module load Python/3.11.5-GCCcore-13.2.0
source venv_req/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -u \
  data_reupload/analysis/four_encoding_master/scripts/aggregate_exact_temporal_fidelity.py
