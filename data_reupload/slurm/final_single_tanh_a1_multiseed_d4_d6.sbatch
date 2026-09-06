#!/bin/bash
#SBATCH --job-name=ideal_ms_d456
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-11
#SBATCH --output=data_reupload/slurm/logs/ideal_multiseed_d456_%A_%a.out
#SBATCH --error=data_reupload/slurm/logs/ideal_multiseed_d456_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${QML_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "${QML_VENV:-${PROJECT_ROOT}/venv_req}/bin/activate"

mkdir -p data_reupload/slurm/logs

SEEDS=(43 44 45 46)
DEPTHS=(4 5 6)

SEED_INDEX=$(( SLURM_ARRAY_TASK_ID / 3 ))
DEPTH_INDEX=$(( SLURM_ARRAY_TASK_ID % 3 ))

SEED=${SEEDS[$SEED_INDEX]}
DEPTH=${DEPTHS[$DEPTH_INDEX]}

OUT_DIR="data_reupload/results/final_model_sweep/single_tanh_a1/depth_${DEPTH}/seed_${SEED}"

echo "============================================================"
echo "Ideal single-QNN multiseed run"
echo "Job:   ${SLURM_JOB_ID}"
echo "Task:  ${SLURM_ARRAY_TASK_ID}"
echo "Seed:  ${SEED}"
echo "Depth: ${DEPTH}"
echo "Start: $(date)"
echo "============================================================"

if [[ -f "${OUT_DIR}/result_summary.csv" ]]; then
    echo "ERROR: result already exists:"
    echo "${OUT_DIR}/result_summary.csv"
    exit 2
fi

/usr/bin/time -v python -u \
  data_reupload/scripts/final_sweep/train_final_single_qnn.py \
  --csv_path "${QML_DATA_PATH:-data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv}" \
  --value_col 4 \
  --train_end_idx 434 \
  --window_size 14 \
  --train_window_end 350 \
  --scaler_data_min 0.0 \
  --scaler_data_max 350.0 \
  --scaler_feature_min -1.0 \
  --scaler_feature_max 1.0 \
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
  --variant_id single_tanh_a1 \
  --output_dir data_reupload/results/final_model_sweep

echo "Finished: $(date)"
echo "Seed=${SEED} Depth=${DEPTH}"
