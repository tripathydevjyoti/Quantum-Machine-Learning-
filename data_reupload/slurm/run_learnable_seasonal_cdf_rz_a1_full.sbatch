#!/bin/bash
#SBATCH --job-name=enc5_full
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --array=0-29
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=data_reupload/slurm/logs/enc5_full_%A_%a.out
#SBATCH --error=data_reupload/slurm/logs/enc5_full_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/umbc/rs/pi_deffner/users/devjyot1/projects/Quantum-Machine-Learning-"

cd "${PROJECT_ROOT}"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source venv_req/bin/activate

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4


# ============================================================
# Map 30 tasks onto
#
# depth = 1,...,6
# seed  = 42,...,46
# ============================================================

TASK="${SLURM_ARRAY_TASK_ID}"

DEPTH=$(( TASK / 5 + 1 ))
SEED=$(( 42 + TASK % 5 ))

VARIANT="learnable_seasonal_cdf_rz_tanh_a1"

OUTPUT_ROOT="data_reupload/results/encoding_sweep"

RUN_DIR="${OUTPUT_ROOT}/${VARIANT}/depth_${DEPTH}/seed_${SEED}"


echo "========================================================================================================================"
echo "MODEL 5 FULL PRODUCTION RUN"
echo "========================================================================================================================"
echo "Started:       $(date)"
echo "Host:          $(hostname)"
echo "Array job:     ${SLURM_ARRAY_JOB_ID}"
echo "Array task:    ${SLURM_ARRAY_TASK_ID}"
echo "Depth:         ${DEPTH}"
echo "Seed:          ${SEED}"
echo "Variant:       ${VARIANT}"
echo "Run directory: ${RUN_DIR}"
echo "Python:        $(which python)"
echo "========================================================================================================================"


# ============================================================
# Safety against accidental contamination / overwrite
# ============================================================

if [[ -d "${RUN_DIR}" ]]; then

    COMPLETE=1

    for f in \
        best_model.pt \
        predictions.npz \
        result_summary.csv \
        learned_encoding_parameters.npz \
        encoding_metadata.json
    do
        if [[ ! -f "${RUN_DIR}/${f}" ]]; then
            COMPLETE=0
        fi
    done

    if [[ "${COMPLETE}" -eq 1 ]]; then
        echo
        echo "Run already appears complete."
        echo "Leaving existing production result untouched."
        echo "STATUS: ALREADY COMPLETE"
        exit 0
    fi

    if find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
        echo
        echo "ERROR: partial/nonempty run directory already exists:"
        echo "${RUN_DIR}"
        echo
        echo "Refusing to overwrite a partial production run."
        exit 20
    fi
fi


# ============================================================
# Production training
# ============================================================

python -u \
  data_reupload/scripts/final_sweep/train_final_single_qnn_learnable_seasonal_cdf_rz.py \
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
  --variant_id "${VARIANT}" \
  --output_dir "${OUTPUT_ROOT}"


# ============================================================
# Minimal task-local artifact check
# ============================================================

for f in \
    best_model.pt \
    predictions.npz \
    result_summary.csv \
    learned_encoding_parameters.npz \
    encoding_metadata.json
do

    if [[ ! -s "${RUN_DIR}/${f}" ]]; then
        echo "ERROR: production artifact missing or empty: ${RUN_DIR}/${f}"
        exit 21
    fi

done


echo
echo "========================================================================================================================"
echo "MODEL 5 PRODUCTION TASK PASS"
echo "Depth=${DEPTH} Seed=${SEED}"
echo "Finished: $(date)"
echo "========================================================================================================================"
