#!/bin/bash
#SBATCH --job-name=ideal_ms_a1
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-11
#SBATCH --output=data_reupload/slurm/logs/ideal_multiseed_a1_%A_%a.out
#SBATCH --error=data_reupload/slurm/logs/ideal_multiseed_a1_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/umbc/rs/pi_deffner/users/devjyot1/projects/Quantum-Machine-Learning-
cd "$PROJECT_ROOT"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source venv_req/bin/activate

# 12 tasks:
# task 0-2   -> seed 43, depths 1-3
# task 3-5   -> seed 44, depths 1-3
# task 6-8   -> seed 45, depths 1-3
# task 9-11  -> seed 46, depths 1-3

SEEDS=(43 44 45 46)
DEPTHS=(1 2 3)

SEED_INDEX=$(( SLURM_ARRAY_TASK_ID / 3 ))
DEPTH_INDEX=$(( SLURM_ARRAY_TASK_ID % 3 ))

SEED=${SEEDS[$SEED_INDEX]}
DEPTH=${DEPTHS[$DEPTH_INDEX]}

OUT_DIR="data_reupload/results/final_model_sweep/single_tanh_a1/depth_${DEPTH}/seed_${SEED}"

mkdir -p data_reupload/slurm/logs

echo "============================================================"
echo "Ideal multi-seed QNN run"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Seed: ${SEED}"
echo "Depth: ${DEPTH}"
echo "Output: ${OUT_DIR}"
echo "============================================================"

# Safety check: do not silently overwrite an already-completed run.
if [[ -f "${OUT_DIR}/result_summary.csv" ]]; then
    echo "ERROR: Completed result already exists:"
    echo "${OUT_DIR}/result_summary.csv"
    echo "Refusing to overwrite."
    exit 2
fi

/usr/bin/time -v python -u \
  data_reupload/scripts/final_sweep/train_final_single_qnn.py \
  --csv_path "data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv" \
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

echo "============================================================"
echo "Finished: $(date)"
echo "Seed: ${SEED}"
echo "Depth: ${DEPTH}"
echo "Result directory: ${OUT_DIR}"
echo "============================================================"
