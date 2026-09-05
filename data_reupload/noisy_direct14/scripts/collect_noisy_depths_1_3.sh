#!/usr/bin/env bash

set -euo pipefail
umask 027

PROJECT_ROOT="${QML_PROJECT_ROOT:-/umbc/rs/pi_deffner/users/devjyot1/projects/Quantum-Machine-Learning-}"
NOISY_ROOT="$PROJECT_ROOT/data_reupload/noisy_direct14"
SOURCE_RESULTS="$NOISY_ROOT/results"
PAPER_RESULTS_ROOT="$PROJECT_ROOT/data_reupload/paper_results"
PACKAGE_NAME="noisy_depths_1_3"
PACKAGE_ROOT="$PAPER_RESULTS_ROOT/$PACKAGE_NAME"
RESULTS_ROOT="$PACKAGE_ROOT/results"
ARCHIVE="$PAPER_RESULTS_ROOT/${PACKAGE_NAME}.tar.gz"
CHECKSUM="$ARCHIVE.sha256"
DATASET="$PROJECT_ROOT/data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv"

ENCODINGS=(
    seasonal_meridian
    learnable_seasonal_cdf
    learnable_seasonal_cdf_rz
)

DEPTHS=(1 2 3)
SEEDS=(42 43 44 45 46)

REQUIRED_RUN_FILES=(
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
    environment_versions.txt
    time_verbose.txt
    trainer_exit_code.txt
)

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

command -v rsync >/dev/null 2>&1 || fail "rsync is not available."
command -v tar >/dev/null 2>&1 || fail "tar is not available."
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is not available."

[[ -d "$PROJECT_ROOT" ]] || fail "Project root is missing: $PROJECT_ROOT"
[[ -d "$SOURCE_RESULTS" ]] || fail "Noisy result root is missing: $SOURCE_RESULTS"
[[ -d "$NOISY_ROOT/analysis" ]] || fail "Noisy analysis directory is missing: $NOISY_ROOT/analysis"
[[ -d "$NOISY_ROOT/scripts" ]] || fail "Noisy scripts directory is missing: $NOISY_ROOT/scripts"
[[ -d "$NOISY_ROOT/slurm" ]] || fail "Noisy Slurm directory is missing: $NOISY_ROOT/slurm"
[[ -f "$DATASET" ]] || fail "Dataset is missing: $DATASET"

[[ ! -e "$PACKAGE_ROOT" ]] || fail "Package destination already exists: $PACKAGE_ROOT"
[[ ! -e "$ARCHIVE" ]] || fail "Archive already exists: $ARCHIVE"
[[ ! -e "$CHECKSUM" ]] || fail "Checksum already exists: $CHECKSUM"

echo "Validating 45 noisy runs before copying..."

VALIDATED_RUNS=0

for ENCODING in "${ENCODINGS[@]}"; do
    for DEPTH in "${DEPTHS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            VARIANT_ID="${ENCODING}_w32_prod_d${DEPTH}_s512_e100_seed${SEED}"
            RUN_DIR="$SOURCE_RESULTS/$VARIANT_ID/depth_${DEPTH}/seed_${SEED}"

            [[ -d "$RUN_DIR" ]] || fail "Missing source run directory: $RUN_DIR"

            for REQUIRED_FILE in "${REQUIRED_RUN_FILES[@]}"; do
                [[ -s "$RUN_DIR/$REQUIRED_FILE" ]] || \
                    fail "Missing or empty required artifact: $RUN_DIR/$REQUIRED_FILE"
            done

            EXIT_CODE=$(tr -d '[:space:]' < "$RUN_DIR/trainer_exit_code.txt")
            [[ "$EXIT_CODE" == "0" ]] || fail "Nonzero trainer exit code in $RUN_DIR: $EXIT_CODE"

            VALIDATED_RUNS=$((VALIDATED_RUNS + 1))
        done
    done
done

[[ "$VALIDATED_RUNS" -eq 45 ]] || fail "Validated $VALIDATED_RUNS runs; expected 45."

echo "Run validation: PASS ($VALIDATED_RUNS/45)"

mkdir -p \
    "$RESULTS_ROOT" \
    "$PACKAGE_ROOT/analysis_output" \
    "$PACKAGE_ROOT/scripts_snapshot" \
    "$PACKAGE_ROOT/slurm_snapshot" \
    "$PACKAGE_ROOT/data" \
    "$PACKAGE_ROOT/manifests"

TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

RUN_MANIFEST="$PACKAGE_ROOT/manifests/noisy_run_manifest.tsv"
printf "encoding\tdepth\tseed\tvariant_id\tsource_directory\tpackaged_directory\n" > "$RUN_MANIFEST"

echo
echo "Copying and verifying noisy run artifacts..."

COPIED_RUNS=0

for ENCODING in "${ENCODINGS[@]}"; do
    for DEPTH in "${DEPTHS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            VARIANT_ID="${ENCODING}_w32_prod_d${DEPTH}_s512_e100_seed${SEED}"
            SOURCE_RUN="$SOURCE_RESULTS/$VARIANT_ID/depth_${DEPTH}/seed_${SEED}"
            DEST_RUN="$RESULTS_ROOT/$ENCODING/depth_${DEPTH}/seed_${SEED}"

            mkdir -p "$DEST_RUN"
            rsync -a "$SOURCE_RUN/" "$DEST_RUN/"

            SOURCE_INDEX="$TMP_ROOT/source_${ENCODING}_d${DEPTH}_s${SEED}.tsv"
            DEST_INDEX="$TMP_ROOT/destination_${ENCODING}_d${DEPTH}_s${SEED}.tsv"

            (
                cd "$SOURCE_RUN"
                find . -type f -printf '%P\t%s\n' | sort
            ) > "$SOURCE_INDEX"

            (
                cd "$DEST_RUN"
                find . -type f -printf '%P\t%s\n' | sort
            ) > "$DEST_INDEX"

            cmp -s "$SOURCE_INDEX" "$DEST_INDEX" || \
                fail "Path/size verification failed for $VARIANT_ID"

            printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$ENCODING" \
                "$DEPTH" \
                "$SEED" \
                "$VARIANT_ID" \
                "$SOURCE_RUN" \
                "$DEST_RUN" \
                >> "$RUN_MANIFEST"

            COPIED_RUNS=$((COPIED_RUNS + 1))
        done

        printf "  %-30s depth %s: PASS (5 seeds)\n" "$ENCODING" "$DEPTH"
    done
done

[[ "$COPIED_RUNS" -eq 45 ]] || fail "Copied $COPIED_RUNS runs; expected 45."

echo
echo "Copying aggregate analyses, provenance, and dataset..."

rsync -a "$NOISY_ROOT/analysis/" "$PACKAGE_ROOT/analysis_output/"
rsync -a "$NOISY_ROOT/scripts/" "$PACKAGE_ROOT/scripts_snapshot/"
rsync -a "$NOISY_ROOT/slurm/" "$PACKAGE_ROOT/slurm_snapshot/"
cp -p "$DATASET" "$PACKAGE_ROOT/data/"

TOTAL_RUNS=$(find "$RESULTS_ROOT" -type f -name result_summary.csv | wc -l)
[[ "$TOTAL_RUNS" -eq 45 ]] || fail "Packaged run count is $TOTAL_RUNS; expected 45."

FILE_INVENTORY="$PACKAGE_ROOT/manifests/file_inventory.tsv"
printf "relative_path\tsize_bytes\n" > "$FILE_INVENTORY"
(
    cd "$PACKAGE_ROOT"
    find \
        results \
        analysis_output \
        scripts_snapshot \
        slurm_snapshot \
        data \
        -type f \
        -printf '%p\t%s\n' \
        | sort
) >> "$FILE_INVENTORY"

FILE_CHECKSUMS="$PACKAGE_ROOT/manifests/file_sha256sum.txt"
(
    cd "$PACKAGE_ROOT"
    find \
        results \
        analysis_output \
        scripts_snapshot \
        slurm_snapshot \
        data \
        -type f \
        -print0 \
        | sort -z \
        | xargs -0 sha256sum
) > "$FILE_CHECKSUMS"

TOTAL_FILES=$(find "$PACKAGE_ROOT" -type f | wc -l)
TOTAL_BYTES=$(du -sb "$PACKAGE_ROOT" | awk '{print $1}')

SUMMARY_FILE="$PACKAGE_ROOT/COLLECTION_SUMMARY.txt"
{
    echo "Noisy QML result collection: depths 1-3"
    echo "created_at=$(date --iso-8601=seconds)"
    echo "project_root=$PROJECT_ROOT"
    echo "source_results=$SOURCE_RESULTS"
    echo "package_root=$PACKAGE_ROOT"
    echo "encodings=3"
    echo "depths=1,2,3"
    echo "seeds=42,43,44,45,46"
    echo "shots=512"
    echo "epochs=100"
    echo "completed_runs=$TOTAL_RUNS"
    echo "total_files=$TOTAL_FILES"
    echo "total_bytes=$TOTAL_BYTES"
    echo "status=PASS"
} > "$SUMMARY_FILE"

echo
echo "Creating downloadable archive..."

mkdir -p "$PAPER_RESULTS_ROOT"

tar -czf "$ARCHIVE" \
    -C "$PAPER_RESULTS_ROOT" \
    "$PACKAGE_NAME"

(
    cd "$PAPER_RESULTS_ROOT"
    sha256sum "$(basename "$ARCHIVE")" > "$(basename "$CHECKSUM")"
)

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE")

echo
echo "NOISY DEPTH-1/2/3 RESULT COLLECTION: PASS"
echo "Completed runs: $TOTAL_RUNS"
echo "Total files:    $TOTAL_FILES"
echo "Package bytes:  $TOTAL_BYTES"
echo "Archive bytes:  $ARCHIVE_BYTES"
echo "Package:        $PACKAGE_ROOT"
echo "Archive:        $ARCHIVE"
echo "Checksum:       $CHECKSUM"
echo "Run manifest:   $RUN_MANIFEST"
echo "File inventory: $FILE_INVENTORY"
