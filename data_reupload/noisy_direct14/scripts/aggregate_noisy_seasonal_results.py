from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

REPOSITORY_ROOT = Path(
    "/umbc/rs/pi_deffner/users/devjyot1/projects/"
    "Quantum-Machine-Learning-"
)

RESULTS_ROOT = (
    REPOSITORY_ROOT
    / "data_reupload"
    / "noisy_direct14"
    / "results"
)

OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "data_reupload"
    / "noisy_direct14"
    / "analysis"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

ENCODINGS = [
    "seasonal_meridian",
    "learnable_seasonal_cdf",
    "learnable_seasonal_cdf_rz",
]

DEPTHS = [1, 2, 3]
SEEDS = [42, 43, 44, 45, 46]

METRIC_COLUMNS = [
    "test_rmse_original",
    "test_fft_spectral_cosine_similarity",
    "test_pearson_corr",
    "test_amplitude_variance_ratio",
]

EXPECTED_RUNS = (
    len(ENCODINGS)
    * len(DEPTHS)
    * len(SEEDS)
)


# =============================================================================
# Load and validate every production run
# =============================================================================

metric_records = []
raw_summary_frames = []
prediction_frames = []

for encoding in ENCODINGS:
    for depth in DEPTHS:
        for seed in SEEDS:
            variant_id = (
                f"{encoding}_w32_prod_d{depth}"
                f"_s512_e100_seed{seed}"
            )

            run_directory = (
                RESULTS_ROOT
                / variant_id
                / f"depth_{depth}"
                / f"seed_{seed}"
            )

            summary_path = (
                run_directory
                / "result_summary.csv"
            )

            prediction_path = (
                run_directory
                / "predictions_long.csv"
            )

            if not summary_path.is_file():
                raise FileNotFoundError(summary_path)

            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)

            summary = pd.read_csv(summary_path)

            if len(summary) != 1:
                raise RuntimeError(
                    f"Expected one summary row in {summary_path}; "
                    f"found {len(summary)}."
                )

            row = summary.iloc[0]

            required_summary_columns = [
                "variant_id",
                "encoding",
                "depth",
                "seed",
                "test_rmse_original",
                "test_pearson_corr",
                "test_pred_std_over_true_std",
                "test_fft_spectral_cosine_similarity",
            ]

            missing_columns = [
                column
                for column in required_summary_columns
                if column not in summary.columns
            ]

            if missing_columns:
                raise RuntimeError(
                    f"Missing columns in {summary_path}: "
                    f"{missing_columns}"
                )

            if str(row["encoding"]) != encoding:
                raise RuntimeError(
                    f"Encoding mismatch in {summary_path}: "
                    f"{row['encoding']!r}"
                )

            if int(row["depth"]) != depth:
                raise RuntimeError(
                    f"Depth mismatch in {summary_path}."
                )

            if int(row["seed"]) != seed:
                raise RuntimeError(
                    f"Seed mismatch in {summary_path}."
                )

            test_rmse = float(
                row["test_rmse_original"]
            )

            test_pearson = float(
                row["test_pearson_corr"]
            )

            test_std_ratio = float(
                row["test_pred_std_over_true_std"]
            )

            test_fft_cosine = float(
                row[
                    "test_fft_spectral_cosine_similarity"
                ]
            )

            # Variance ratio = squared standard-deviation ratio.
            test_variance_ratio = test_std_ratio**2

            metric_records.append(
                {
                    "variant_id": variant_id,
                    "encoding": encoding,
                    "depth": depth,
                    "seed": seed,
                    "test_rmse_original": test_rmse,
                    "test_fft_spectral_cosine_similarity":
                        test_fft_cosine,
                    "test_pearson_corr": test_pearson,
                    "test_pred_std_over_true_std":
                        test_std_ratio,
                    "test_amplitude_variance_ratio":
                        test_variance_ratio,
                    "run_directory": str(run_directory),
                }
            )

            raw_summary = summary.copy()
            raw_summary.insert(
                0,
                "source_result_summary",
                str(summary_path),
            )

            raw_summary_frames.append(raw_summary)

            # -----------------------------------------------------------------
            # Load test predictions for later mean ± SD prediction plots
            # -----------------------------------------------------------------

            predictions = pd.read_csv(prediction_path)

            required_prediction_columns = [
                "split",
                "split_pos",
                "target_index",
                "true_original",
                "pred_original",
            ]

            missing_prediction_columns = [
                column
                for column in required_prediction_columns
                if column not in predictions.columns
            ]

            if missing_prediction_columns:
                raise RuntimeError(
                    f"Missing columns in {prediction_path}: "
                    f"{missing_prediction_columns}"
                )

            test_predictions = predictions[
                predictions["split"]
                .astype(str)
                .str.lower()
                .eq("test")
            ].copy()

            if test_predictions.empty:
                raise RuntimeError(
                    f"No test predictions in {prediction_path}"
                )

            if test_predictions["target_index"].duplicated().any():
                raise RuntimeError(
                    "Duplicate test target indices in "
                    f"{prediction_path}"
                )

            test_predictions = test_predictions[
                [
                    "split_pos",
                    "target_index",
                    "true_original",
                    "pred_original",
                ]
            ].copy()

            for column in [
                "split_pos",
                "target_index",
                "true_original",
                "pred_original",
            ]:
                test_predictions[column] = pd.to_numeric(
                    test_predictions[column],
                    errors="raise",
                )

            test_predictions.insert(
                0,
                "seed",
                seed,
            )

            test_predictions.insert(
                0,
                "depth",
                depth,
            )

            test_predictions.insert(
                0,
                "encoding",
                encoding,
            )

            prediction_frames.append(test_predictions)

            # -----------------------------------------------------------------
            # Verify saved RMSE, Pearson correlation, and amplitude ratio
            # -----------------------------------------------------------------

            observed = test_predictions[
                "true_original"
            ].to_numpy(dtype=float)

            predicted = test_predictions[
                "pred_original"
            ].to_numpy(dtype=float)

            calculated_rmse = float(
                np.sqrt(
                    np.mean(
                        (predicted - observed) ** 2
                    )
                )
            )

            calculated_pearson = float(
                np.corrcoef(
                    observed,
                    predicted,
                )[0, 1]
            )

            calculated_std_ratio = float(
                np.std(predicted, ddof=0)
                / np.std(observed, ddof=0)
            )

            checks = [
                (
                    "RMSE",
                    calculated_rmse,
                    test_rmse,
                ),
                (
                    "Pearson correlation",
                    calculated_pearson,
                    test_pearson,
                ),
                (
                    "standard-deviation ratio",
                    calculated_std_ratio,
                    test_std_ratio,
                ),
            ]

            for name, calculated, saved in checks:
                if not np.isclose(
                    calculated,
                    saved,
                    rtol=1.0e-7,
                    atol=1.0e-8,
                ):
                    raise RuntimeError(
                        f"{name} mismatch for {variant_id}: "
                        f"calculated={calculated}, saved={saved}"
                    )


# =============================================================================
# Run-level metrics
# =============================================================================

run_metrics = (
    pd.DataFrame(metric_records)
    .sort_values(
        ["encoding", "depth", "seed"]
    )
    .reset_index(drop=True)
)

if len(run_metrics) != EXPECTED_RUNS:
    raise RuntimeError(
        f"Expected {EXPECTED_RUNS} runs; "
        f"found {len(run_metrics)}."
    )

combination_counts = (
    run_metrics
    .groupby(["encoding", "depth"])
    ["seed"]
    .nunique()
)

if not combination_counts.eq(len(SEEDS)).all():
    raise RuntimeError(
        "At least one encoding-depth combination "
        "does not contain all five seeds."
    )

run_metrics.to_csv(
    OUTPUT_ROOT
    / "noisy_paper_metrics_by_run.csv",
    index=False,
)


# =============================================================================
# Mean and sample SD across five seeds
# =============================================================================

summary_wide = (
    run_metrics
    .groupby(
        ["encoding", "depth"],
        as_index=False,
    )
    .agg(
        n_seeds=("seed", "nunique"),
        test_rmse_mean=(
            "test_rmse_original",
            "mean",
        ),
        test_rmse_sd=(
            "test_rmse_original",
            "std",
        ),
        test_fft_cosine_mean=(
            "test_fft_spectral_cosine_similarity",
            "mean",
        ),
        test_fft_cosine_sd=(
            "test_fft_spectral_cosine_similarity",
            "std",
        ),
        test_pearson_mean=(
            "test_pearson_corr",
            "mean",
        ),
        test_pearson_sd=(
            "test_pearson_corr",
            "std",
        ),
        test_amplitude_variance_ratio_mean=(
            "test_amplitude_variance_ratio",
            "mean",
        ),
        test_amplitude_variance_ratio_sd=(
            "test_amplitude_variance_ratio",
            "std",
        ),
    )
    .sort_values(["encoding", "depth"])
    .reset_index(drop=True)
)

summary_wide.to_csv(
    OUTPUT_ROOT
    / "noisy_paper_metrics_seed_summary.csv",
    index=False,
)


# =============================================================================
# Long-form metric summary
# =============================================================================

metrics_long = run_metrics.melt(
    id_vars=[
        "variant_id",
        "encoding",
        "depth",
        "seed",
    ],
    value_vars=METRIC_COLUMNS,
    var_name="metric",
    value_name="value",
)

summary_long = (
    metrics_long
    .groupby(
        ["encoding", "depth", "metric"],
        as_index=False,
    )
    .agg(
        n_seeds=("seed", "nunique"),
        mean=("value", "mean"),
        sd=("value", "std"),
        sem=("value", "sem"),
        minimum=("value", "min"),
        maximum=("value", "max"),
    )
    .sort_values(
        ["metric", "encoding", "depth"]
    )
    .reset_index(drop=True)
)

summary_long.to_csv(
    OUTPUT_ROOT
    / "noisy_paper_metrics_seed_summary_long.csv",
    index=False,
)


# =============================================================================
# Pointwise test-prediction mean and SD across seeds
# =============================================================================

all_predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

prediction_summary = (
    all_predictions
    .groupby(
        [
            "encoding",
            "depth",
            "split_pos",
            "target_index",
        ],
        as_index=False,
    )
    .agg(
        n_seeds=("seed", "nunique"),
        true_original=("true_original", "first"),
        true_min=("true_original", "min"),
        true_max=("true_original", "max"),
        pred_mean=("pred_original", "mean"),
        pred_sd=("pred_original", "std"),
    )
    .sort_values(
        [
            "encoding",
            "depth",
            "split_pos",
        ]
    )
    .reset_index(drop=True)
)

if not prediction_summary["n_seeds"].eq(
    len(SEEDS)
).all():
    raise RuntimeError(
        "At least one prediction point does not "
        "contain all five seeds."
    )

true_range = (
    prediction_summary["true_max"]
    - prediction_summary["true_min"]
)

if not np.allclose(
    true_range,
    0.0,
    rtol=0.0,
    atol=1.0e-10,
):
    raise RuntimeError(
        "Observed test values differ across seeds."
    )

prediction_summary = prediction_summary.drop(
    columns=["true_min", "true_max"]
)

prediction_summary.to_csv(
    OUTPUT_ROOT
    / "noisy_test_predictions_seed_summary.csv",
    index=False,
)


# =============================================================================
# Preserve the complete result-summary tables
# =============================================================================

all_raw_summaries = pd.concat(
    raw_summary_frames,
    ignore_index=True,
    sort=False,
)

all_raw_summaries.to_csv(
    OUTPUT_ROOT
    / "noisy_result_summaries_all_runs.csv",
    index=False,
)


# =============================================================================
# Console summary
# =============================================================================

print()
print("Noisy seasonal aggregation completed successfully.")
print(f"Validated runs: {len(run_metrics)}")
print(f"Expected runs:  {EXPECTED_RUNS}")
print()
print(summary_wide.to_string(index=False))
print()
print(f"Outputs saved under: {OUTPUT_ROOT}")

for output_path in sorted(
    OUTPUT_ROOT.glob("noisy_*.csv")
):
    print(f"  {output_path.name}")
