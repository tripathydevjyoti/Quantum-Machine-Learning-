#!/bin/bash
#SBATCH --job-name=enc5_gate
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=2018
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=data_reupload/slurm/logs/enc5_gate_%j.out
#SBATCH --error=data_reupload/slurm/logs/enc5_gate_%j.err

set -euo pipefail

PROJECT_ROOT="${QML_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

cd "${PROJECT_ROOT}"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "${QML_VENV:-${PROJECT_ROOT}/venv_req}/bin/activate"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

VARIANT="smoke_learnable_seasonal_cdf_rz_tanh_a1"
OUTPUT_ROOT="data_reupload/results/encoding_sweep_gate"
RUN_DIR="${OUTPUT_ROOT}/${VARIANT}/depth_2/seed_42"

echo "========================================================================================================================"
echo "MODEL 5 FINAL PRODUCTION GATE"
echo "========================================================================================================================"
echo "Started: $(date)"
echo "Host:    $(hostname)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Python:  $(which python)"
echo "========================================================================================================================"

# A gate rerun must never inherit an older 3-epoch smoke result.
rm -rf "${RUN_DIR}"

echo
echo "STEP 1/2: real three-epoch end-to-end training"
echo

python -u \
  data_reupload/scripts/final_sweep/train_final_single_qnn_learnable_seasonal_cdf_rz.py \
  --csv_path "${QML_DATA_PATH:-data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv}" \
  --value_col 4 \
  --train_end_idx 434 \
  --window_size 14 \
  --train_window_end 350 \
  --scaler_data_min 0.0 \
  --scaler_data_max 350.0 \
  --scaler_feature_min -1.0 \
  --scaler_feature_max 1.0 \
  --depth 2 \
  --n_ansatz_layers 1 \
  --readout all \
  --seed 42 \
  --qnn_epochs 3 \
  --qnn_learning_rate 0.01 \
  --qnn_batch_size 16 \
  --angle_factor 3.141592653589793 \
  --qml_device default.qubit \
  --diff_method backprop \
  --device cpu \
  --use_output_tanh \
  --print_every 1 \
  --variant_id "${VARIANT}" \
  --output_dir "${OUTPUT_ROOT}"

echo
echo "STEP 2/2: strict production-gate validation"
echo

python -u \
  data_reupload/analysis/four_encoding_master/scripts/validate_model5_final_gate.py

echo
echo "========================================================================================================================"
echo "FINAL GATE PASSED"
echo "Finished: $(date)"
echo "========================================================================================================================"
