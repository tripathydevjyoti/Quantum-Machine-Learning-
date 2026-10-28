#!/usr/bin/env bash
#SBATCH --job-name=seasonal_noisy_prod
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --array=0-44
#SBATCH --output=data_reupload/noisy_direct14/logs/seasonal_noisy_prod_%A_%a.out
#SBATCH --error=data_reupload/noisy_direct14/logs/seasonal_noisy_prod_%A_%a.err

set -euo pipefail
umask 027

PROJECT_ROOT="${QML_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
NOISY_ROOT="$PROJECT_ROOT/data_reupload/noisy_direct14"

SCRIPT="$NOISY_ROOT/scripts/direct14_noisy_seasonal_two_pass_worker_pool_hpc.py"
CSV_PATH="${QML_DATA_PATH:-$PROJECT_ROOT/data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv}"
OUTPUT_ROOT="$NOISY_ROOT/results"
LOG_ROOT="$NOISY_ROOT/logs"
METADATA_ROOT="$LOG_ROOT/production_metadata"

ENCODINGS=(
    seasonal_meridian
    learnable_seasonal_cdf
    learnable_seasonal_cdf_rz
)

# Tasks 0-14: seasonal_meridian
# Tasks 15-29: learnable_seasonal_cdf
# Tasks 30-44: learnable_seasonal_cdf_rz
# Within each block: seeds 42-46 for depths 1, 2, and 3.

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}

if (( TASK_ID < 0 || TASK_ID > 44 )); then
    echo "ERROR: array task ID must be in 0..44; received ${TASK_ID}."
    exit 2
fi

ENCODING_INDEX=$(( TASK_ID / 15 ))
WITHIN_ENCODING=$(( TASK_ID % 15 ))
DEPTH=$(( WITHIN_ENCODING / 5 + 1 ))
SEED=$(( 42 + WITHIN_ENCODING % 5 ))
ENCODING=${ENCODINGS[$ENCODING_INDEX]}

SHOTS=512
EPOCHS=100
BATCH_SIZE=16
NUM_WORKERS=32
AER_THREADS_PER_WORKER=1

VARIANT_ID="${ENCODING}_w32_prod_d${DEPTH}_s${SHOTS}_e${EPOCHS}_seed${SEED}"
RUN_OUTPUT_DIR="$OUTPUT_ROOT/$VARIANT_ID/depth_${DEPTH}/seed_${SEED}"
TASK_METADATA_DIR="$METADATA_ROOT/$VARIANT_ID"

RESULT_SUMMARY="$RUN_OUTPUT_DIR/result_summary.csv"
RESUME_CHECKPOINT="$RUN_OUTPUT_DIR/last_training_checkpoint.pt"

cd "$PROJECT_ROOT"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source "${QML_VENV:-${PROJECT_ROOT}/venv_noisy_qml}/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT" "$METADATA_ROOT"

test -f "$SCRIPT"
test -f "$CSV_PATH"
test -x /usr/bin/time

echo "===================================================================================================="
echo "NOISY SEASONAL PRODUCTION ARRAY"
echo "===================================================================================================="
echo "Date:                     $(date --iso-8601=seconds)"
echo "Host:                     $(hostname)"
echo "Slurm job:                ${SLURM_JOB_ID}"
echo "Array task:               ${TASK_ID}"
echo "Encoding index:           ${ENCODING_INDEX}"
echo "Within-encoding index:    ${WITHIN_ENCODING}"
echo "Encoding:                 ${ENCODING}"
echo "Depth:                    ${DEPTH}"
echo "Seed:                     ${SEED}"
echo "Shots:                    ${SHOTS}"
echo "Epochs:                   ${EPOCHS}"
echo "Batch size:               ${BATCH_SIZE}"
echo "Validation interval:      5 epochs"
echo "Checkpoint interval:      1 epoch"
echo "Quantum workers:          ${NUM_WORKERS}"
echo "Aer threads per worker:   ${AER_THREADS_PER_WORKER}"
echo "Trainer:                  ${SCRIPT}"
echo "CSV:                      ${CSV_PATH}"
echo "Variant ID:               ${VARIANT_ID}"
echo "Output:                   ${RUN_OUTPUT_DIR}"
echo "Task metadata:            ${TASK_METADATA_DIR}"
echo "===================================================================================================="

if [[ -f "$RESULT_SUMMARY" ]]; then
    echo "ERROR: completed result_summary.csv already exists; refusing to overwrite:"
    echo "$RESULT_SUMMARY"
    exit 20
fi

if [[ -f "$RESUME_CHECKPOINT" ]]; then
    echo "ERROR: last_training_checkpoint.pt already exists; refusing to restart or resume silently:"
    echo "$RESUME_CHECKPOINT"
    exit 21
fi

if [[ -e "$RUN_OUTPUT_DIR" ]]; then
    echo "ERROR: production run directory already exists:"
    echo "$RUN_OUTPUT_DIR"
    echo "Inspect it explicitly before deciding whether it is safe to remove or rerun."
    exit 22
fi

if [[ -e "$TASK_METADATA_DIR" ]]; then
    echo "ERROR: task metadata directory already exists, indicating an earlier launch attempt:"
    echo "$TASK_METADATA_DIR"
    echo "Inspect it explicitly before deciding whether it is safe to remove or rerun."
    exit 23
fi

mkdir -p "$TASK_METADATA_DIR"

ENVIRONMENT_FILE="$TASK_METADATA_DIR/environment_versions.txt"
TIME_FILE="$TASK_METADATA_DIR/time_verbose.txt"
EXIT_FILE="$TASK_METADATA_DIR/trainer_exit_code.txt"

{
    echo "recorded_at=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID}"
    echo "slurm_array_task_id=${TASK_ID}"
    echo "variant_id=${VARIANT_ID}"
    echo "encoding=${ENCODING}"
    echo "depth=${DEPTH}"
    echo "seed=${SEED}"
    echo "python_executable=$(command -v python)"
    python --version

    echo
    echo "===== MODULES ====="
    module list 2>&1

    echo
    echo "===== PACKAGE VERSIONS ====="

    python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

packages = (
    "numpy",
    "pandas",
    "torch",
    "PennyLane",
    "PennyLane-qiskit",
    "qiskit",
    "qiskit-aer",
    "qiskit-ibm-runtime",
)

for package in packages:
    try:
        print(f"{package}=={version(package)}")
    except PackageNotFoundError:
        print(f"{package}=NOT_INSTALLED")
PY

    echo
    echo "===== SOURCE STATE ====="
    sha256sum "$SCRIPT"
    git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "git_commit=UNAVAILABLE"
} | tee "$ENVIRONMENT_FILE"

echo
echo "===== STARTING TRAINER ====="

set +e

/usr/bin/time -v -o "$TIME_FILE" \
python -u "$SCRIPT" \
    --csv_path "$CSV_PATH" \
    --value_col 4 \
    --train_end_idx 434 \
    --window_size 14 \
    --train_window_end 350 \
    --scaler_data_min 0.0 \
    --scaler_data_max 350.0 \
    --scaler_feature_min -1.0 \
    --scaler_feature_max 1.0 \
    --encoding "$ENCODING" \
    --depth "$DEPTH" \
    --n_ansatz_layers 1 \
    --readout all \
    --angle_factor 3.141592653589793 \
    --use_output_tanh \
    --q_init_scale 0.01 \
    --backend_mode qiskit_aer_noisy \
    --fake_backend_class FakeMelbourneV2 \
    --shots "$SHOTS" \
    --seed_simulator "$SEED" \
    --aer_method automatic \
    --aer_threads "$AER_THREADS_PER_WORKER" \
    --optimization_level 1 \
    --num_quantum_workers "$NUM_WORKERS" \
    --qnn_epochs "$EPOCHS" \
    --qnn_learning_rate 0.005 \
    --qnn_batch_size "$BATCH_SIZE" \
    --spsa_a 0.02 \
    --spsa_c 0.05 \
    --spsa_stability_constant 50 \
    --spsa_alpha 0.602 \
    --spsa_gamma 0.101 \
    --spsa_seed "$SEED" \
    --head_grad_clip 1.0 \
    --seed "$SEED" \
    --device cpu \
    --torch_threads 1 \
    --validate_every 5 \
    --eval_repeats 1 \
    --eval_batch_size 32 \
    --print_every 1 \
    --checkpoint_every 1 \
    --variant_id "$VARIANT_ID" \
    --output_dir "$OUTPUT_ROOT"

TRAINER_EXIT_CODE=$?
set -e

printf '%s\n' "$TRAINER_EXIT_CODE" > "$EXIT_FILE"

echo
echo "Trainer exit code: ${TRAINER_EXIT_CODE}"

if (( TRAINER_EXIT_CODE != 0 )); then
    echo "ERROR: trainer failed. No success status will be emitted."
    echo "Inspect the Slurm stdout/stderr and:"
    echo "  $TIME_FILE"
    echo "  $RUN_OUTPUT_DIR"
    exit "$TRAINER_EXIT_CODE"
fi

echo
echo "===== VERIFYING PRODUCTION ARTIFACTS ====="

EXPECTED_FILES=(
    best_model.pt
    config.json
    fft_power_by_period.csv
    fft_summary_by_split.csv
    final_model.pt
    history.csv
    last_training_checkpoint.pt
    learned_parameters.npz
    metrics_by_split.csv
    peak_low_error_summary.csv
    predictions_long.csv
    predictions.npz
    predictions_wide.csv
    result_summary.csv
    top_errors.csv
)

for EXPECTED_FILE in "${EXPECTED_FILES[@]}"; do
    if [[ ! -f "$RUN_OUTPUT_DIR/$EXPECTED_FILE" ]]; then
        echo "ERROR: expected production artifact is missing:"
        echo "$RUN_OUTPUT_DIR/$EXPECTED_FILE"
        exit 30
    fi
done

if [[ ! -s "$RESULT_SUMMARY" ]]; then
    echo "ERROR: result_summary.csv exists but is empty:"
    echo "$RESULT_SUMMARY"
    exit 31
fi

cp "$ENVIRONMENT_FILE" "$RUN_OUTPUT_DIR/environment_versions.txt"
cp "$TIME_FILE" "$RUN_OUTPUT_DIR/time_verbose.txt"
cp "$EXIT_FILE" "$RUN_OUTPUT_DIR/trainer_exit_code.txt"

echo
echo "===== GENERATED FILES ====="

find "$RUN_OUTPUT_DIR" \
    -maxdepth 1 \
    -type f \
    -printf "%f\t%s bytes\n" \
    | sort

echo
echo "NOISY SEASONAL PRODUCTION RUN: PASS"
echo "Array task: $TASK_ID"
echo "Encoding: $ENCODING"
echo "Depth: $DEPTH"
echo "Seed: $SEED"
echo "Output: $RUN_OUTPUT_DIR"
