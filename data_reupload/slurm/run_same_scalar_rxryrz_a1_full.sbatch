#!/bin/bash
#SBATCH --job-name=rxryrz_a1
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-29
#SBATCH --output=data_reupload/slurm/logs/rxryrz_a1_%A_%a.out
#SBATCH --error=data_reupload/slurm/logs/rxryrz_a1_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${QML_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

cd "${PROJECT_ROOT}"

module purge
module load Python/3.11.5-GCCcore-13.2.0

source "${QML_VENV:-${PROJECT_ROOT}/venv_req}/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

TASK="${SLURM_ARRAY_TASK_ID}"

DEPTH=$(( TASK / 5 + 1 ))
SEED=$(( TASK % 5 + 42 ))

SCRIPT="data_reupload/scripts/final_sweep/train_final_single_qnn_same_scalar_rxryrz.py"

OUT_ROOT="data_reupload/results/encoding_sweep"

VARIANT="same_scalar_rxryrz_tanh_a1"

echo "============================================================"
echo "SAME-SCALAR RXRYRZ A=1 PRODUCTION SWEEP"
echo "============================================================"
echo "Array job:      ${SLURM_ARRAY_JOB_ID}"
echo "Array task:     ${SLURM_ARRAY_TASK_ID}"
echo "Depth:          ${DEPTH}"
echo "Seed:           ${SEED}"
echo "Host:           $(hostname)"
echo "Started:        $(date)"
echo "============================================================"

python -u "${SCRIPT}" \
  --csv_path "${QML_DATA_PATH:-data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv}" \
  --value_col 4 \
  --train_end_idx 434 \
  --window_size 14 \
  --train_window_end 350 \
  --scaler_data_min 0 \
  --scaler_data_max 350 \
  --scaler_feature_min -1 \
  --scaler_feature_max 1 \
  --depth "${DEPTH}" \
  --n_ansatz_layers 1 \
  --readout all \
  --seed "${SEED}" \
  --qnn_epochs 100 \
  --qnn_learning_rate 0.01 \
  --qnn_batch_size 16 \
  --angle_factor 3.141592653589793 \
  --qml_device default.qubit \
  --diff_method backprop \
  --device cpu \
  --use_output_tanh \
  --print_every 10 \
  --variant_id "${VARIANT}" \
  --output_dir "${OUT_ROOT}"

RUN_DIR="${OUT_ROOT}/${VARIANT}/depth_${DEPTH}/seed_${SEED}"

echo
echo "Checking required artifacts..."

for F in \
  result_summary.csv \
  best_model.pt \
  predictions.npz \
  history.csv \
  metrics_by_split.csv \
  fft_summary_by_split.csv \
  encoding_metadata.json
do

    if [[ ! -s "${RUN_DIR}/${F}" ]]; then
        echo "ERROR: missing ${RUN_DIR}/${F}"
        exit 1
    fi

    echo "OK: ${F}"

done

if ! grep -q '"status": "PASS"' \
  "${RUN_DIR}/encoding_metadata.json"
then
    echo "ERROR: encoding provenance did not PASS"
    exit 1
fi

echo
echo "============================================================"
echo "RUN STATUS: PASS"
echo "Depth=${DEPTH} Seed=${SEED}"
echo "Finished: $(date)"
echo "============================================================"
