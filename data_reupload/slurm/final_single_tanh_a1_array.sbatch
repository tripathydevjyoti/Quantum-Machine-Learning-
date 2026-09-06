#!/bin/bash
#SBATCH --job-name=final_s_a1
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-6
#SBATCH --output=data_reupload/slurm/logs/final_single_tanh_a1_%A_%a.out
#SBATCH --error=data_reupload/slurm/logs/final_single_tanh_a1_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${QML_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "${QML_VENV:-${PROJECT_ROOT}/venv_req}/bin/activate"

DEPTH=${SLURM_ARRAY_TASK_ID}

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Depth: ${DEPTH}"

mkdir -p data_reupload/slurm/logs

/usr/bin/time -v python -u data_reupload/scripts/final_sweep/train_final_single_qnn.py \
  --csv_path "${QML_DATA_PATH:-data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv}" \
  --value_col 4 \
  --train_end_idx 434 \
  --window_size 14 \
  --train_window_end 350 \
  --scaler_data_min 0.0 \
  --scaler_data_max 350.0 \
  --scaler_feature_min -1.0 \
  --scaler_feature_max 1.0 \
  --depth ${DEPTH} \
  --n_ansatz_layers 1 \
  --readout all \
  --seed 42 \
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
