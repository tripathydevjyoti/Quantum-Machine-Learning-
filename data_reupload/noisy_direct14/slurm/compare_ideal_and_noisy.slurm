#!/bin/bash
#SBATCH --job-name=ideal_vs_noisy
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/umbc/rs/pi_deffner/users/devjyot1/projects/Quantum-Machine-Learning-/data_reupload/noisy_direct14/logs/ideal_vs_noisy_%j.out
#SBATCH --error=/umbc/rs/pi_deffner/users/devjyot1/projects/Quantum-Machine-Learning-/data_reupload/noisy_direct14/logs/ideal_vs_noisy_%j.err

set -euo pipefail

QML_REPO_ROOT="/umbc/rs/pi_deffner/users/devjyot1/projects/Quantum-Machine-Learning-"

cd "$QML_REPO_ROOT"

module purge
module load Python/3.11.5-GCCcore-13.2.0

source "$QML_REPO_ROOT/venv_req/bin/activate"

python -u \
  data_reupload/noisy_direct14/scripts/compare_ideal_and_noisy.py
