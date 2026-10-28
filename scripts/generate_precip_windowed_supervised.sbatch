#!/usr/bin/env bash

#SBATCH --job-name=make_precip_splits
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/make_precip_splits_%j.out
#SBATCH --error=logs/make_precip_splits_%j.err

set -euo pipefail

PROJECT_ROOT="${QML_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

cd "$PROJECT_ROOT"

mkdir -p logs
mkdir -p data/processed

module purge
module load Python/3.11.5-GCCcore-13.2.0

source "${QML_VENV:-${PROJECT_ROOT}/venv_req}/bin/activate"

echo "===== JOB INFO ====="
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "DATE=$(date)"
echo

echo "===== PYTHON ====="
which python
python --version
echo

echo "===== GENERATING DATASET ====="

python scripts/generate_precip_windowed_supervised.py

echo
echo "===== OUTPUT FILES ====="
ls -lh data/processed/precip_mm_windowed_supervised/

echo
echo "===== DONE ====="
date
