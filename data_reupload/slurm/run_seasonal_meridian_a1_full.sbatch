#!/bin/bash
#SBATCH --job-name=seas_a1
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-29
#SBATCH --output=data_reupload/slurm/logs/seasonal_meridian_a1_%A_%a.out
#SBATCH --error=data_reupload/slurm/logs/seasonal_meridian_a1_%A_%a.err

set -euo pipefail

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT="${QML_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CACHE_ROOT="${QML_CACHE_ROOT:-${PROJECT_ROOT}/.cache}"
TMP_ROOT="${QML_TMP_ROOT:-${TMPDIR:-${PROJECT_ROOT}/.tmp}}"

SCRIPT="${PROJECT_ROOT}/data_reupload/scripts/final_sweep/train_final_single_qnn_seasonal_meridian.py"

OUTPUT_ROOT="${PROJECT_ROOT}/data_reupload/results/encoding_sweep"

cd "${PROJECT_ROOT}"

# ============================================================
# PYTHON ENVIRONMENT
# ============================================================

module purge
module load Python/3.11.5-GCCcore-13.2.0

source "${QML_VENV:-${PROJECT_ROOT}/venv_req}/bin/activate"

# ============================================================
# KEEP CACHE/TEMP DATA OUT OF THE HOME FILESYSTEM
# ============================================================

export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export PYTHONPYCACHEPREFIX="${CACHE_ROOT}/python"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TMPDIR="${TMP_ROOT}"

mkdir -p \
    "${XDG_CACHE_HOME}" \
    "${MPLCONFIGDIR}" \
    "${PYTHONPYCACHEPREFIX}" \
    "${PIP_CACHE_DIR}" \
    "${TMPDIR}" \
    "${OUTPUT_ROOT}" \
    data_reupload/slurm/logs

# ============================================================
# THREAD CONTROL
# ============================================================

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ============================================================
# ARRAY MAPPING
#
# 30 jobs:
#
# task  0-4  -> L=1, seeds 42-46
# task  5-9  -> L=2, seeds 42-46
# task 10-14 -> L=3, seeds 42-46
# task 15-19 -> L=4, seeds 42-46
# task 20-24 -> L=5, seeds 42-46
# task 25-29 -> L=6, seeds 42-46
# ============================================================

TASK_ID="${SLURM_ARRAY_TASK_ID}"

DEPTH=$(( TASK_ID / 5 + 1 ))
SEED=$(( TASK_ID % 5 + 42 ))

A=1
VARIANT_ID="seasonal_meridian_tanh_a1"

RUN_DIR="${OUTPUT_ROOT}/${VARIANT_ID}/depth_${DEPTH}/seed_${SEED}"

# ============================================================
# PREFLIGHT CHECKS
# ============================================================

if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: training script not found:"
    echo "${SCRIPT}"
    exit 10
fi

if [[ ! -f "${QML_DATA_PATH:-data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv}" ]]; then
    echo "ERROR: input CSV not found."
    exit 11
fi

# Prevent accidental overwrite of a completed production run.
if [[ -f "${RUN_DIR}/result_summary.csv" ]]; then
    echo "ERROR: production result already exists:"
    echo "${RUN_DIR}/result_summary.csv"
    echo "Refusing to overwrite."
    exit 12
fi

# ============================================================
# JOB INFORMATION
# ============================================================

echo "===================================================================================================="
echo "SEASONAL-MERIDIAN A=1 PRODUCTION SWEEP"
echo "===================================================================================================="
echo "Started:              $(date)"
echo "Host:                 $(hostname)"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Array job ID:         ${SLURM_ARRAY_JOB_ID:-NA}"
echo "Array task ID:        ${SLURM_ARRAY_TASK_ID}"
echo
echo "Encoding:             seasonal_meridian"
echo "A:                    ${A}"
echo "Depth L:              ${DEPTH}"
echo "Seed:                 ${SEED}"
echo "Epochs:               100"
echo "Batch size:           16"
echo "Learning rate:        0.01"
echo "Qubits:               14"
echo "Season period:        12"
echo
echo "Variant:              ${VARIANT_ID}"
echo "Output:               ${RUN_DIR}"
echo
echo "CPUs:                 ${SLURM_CPUS_PER_TASK}"
echo "Memory request:       8G"
echo "Python:               $(which python)"
echo "Python version:       $(python --version)"
echo "===================================================================================================="

# ============================================================
# PRODUCTION TRAINING
#
# Controlled intervention:
#
# CHANGED:
#   data encoding only
#
# UNCHANGED:
#   data split
#   target scaling
#   ansatz
#   CNOT topology
#   Z readout
#   linear head + tanh
#   optimizer
#   LR
#   batch size
#   epochs
#   checkpoint selection
#   metrics
#
# Seasonal encoding:
#
#   z_s     = 2 F_train(x_s) - 1
#   theta_s = arccos(z_s)
#   phi_s   = 2*pi*(s mod 12)/12
#
#   |psi_s> = RZ(phi_s) RY(theta_s) |0>
#
# where s is the ABSOLUTE source observation index.
# ============================================================

/usr/bin/time -v python -u "${SCRIPT}" \
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
    --variant_id "${VARIANT_ID}" \
    --output_dir "${OUTPUT_ROOT}"

# ============================================================
# POST-RUN VALIDATION
# ============================================================

echo
echo "===================================================================================================="
echo "POST-RUN VALIDATION"
echo "===================================================================================================="

REQUIRED_FILES=(
    config.json
    history.csv
    best_model.pt
    final_model.pt
    learned_parameters.npz
    predictions.npz
    predictions_long.csv
    predictions_wide.csv
    metrics_by_split.csv
    fft_summary_by_split.csv
    fft_power_by_period.csv
    peak_low_error_summary.csv
    top_errors.csv
    result_summary.csv
    encoding_metadata.json
    seasonal_encoding_reference.npz
)

for F in "${REQUIRED_FILES[@]}"
do
    if [[ ! -s "${RUN_DIR}/${F}" ]]; then
        echo "ERROR: missing or empty artifact:"
        echo "${RUN_DIR}/${F}"
        exit 20
    fi

    echo "OK: ${F}"
done

# Verify audit status without running Python on the login node;
# this executes here inside the compute job.

python - "${RUN_DIR}/encoding_metadata.json" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path) as f:
    metadata = json.load(f)

assert metadata["status"] == "PASS"
assert metadata["encoding"] == "seasonal_meridian"
assert metadata["season_period"] == 12
assert metadata["cdf_reference_n"] == 364

assert metadata["train_target_start"] == 14
assert metadata["train_target_end"] == 363

assert metadata["val_target_start"] == 364
assert metadata["val_target_end"] == 433

assert metadata["test_target_start"] == 434
assert metadata["test_target_end"] == 538

assert metadata["first_train_source_mod12"] == \
       [0,1,2,3,4,5,6,7,8,9,10,11,0,1]

assert metadata["first_test_source_mod12"] == \
       [0,1,2,3,4,5,6,7,8,9,10,11,0,1]

print("Encoding provenance validation: PASS")
PY

echo
echo "===================================================================================================="
echo "PRODUCTION RUN COMPLETED SUCCESSFULLY"
echo "===================================================================================================="
echo "A=${A}"
echo "L=${DEPTH}"
echo "seed=${SEED}"
echo "Finished: $(date)"
echo "===================================================================================================="
