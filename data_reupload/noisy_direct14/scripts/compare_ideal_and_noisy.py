from pathlib import Path
import os

import numpy as np
import pandas as pd


def _resolve_project_root() -> Path:
    configured = os.environ.get("QML_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    for candidate in Path(__file__).resolve().parents:
        if (candidate / "data_reupload").is_dir():
            return candidate

    raise RuntimeError(
        "Could not locate the repository root; set QML_PROJECT_ROOT."
    )


PROJECT_ROOT = _resolve_project_root()

IDEAL_RESULTS_ROOT = (
    PROJECT_ROOT
    / "data_reupload"
    / "paper_results"
    / "ideal"
    / "results"
)

NOISY_ANALYSIS_ROOT = (
    PROJECT_ROOT
    / "data_reupload"
    / "noisy_direct14"
    / "analysis"
)

NOISY_RUN_METRICS_PATH = (
    NOISY_ANALYSIS_ROOT
    / "noisy_paper_metrics_by_run.csv"
)

ENCODINGS = [
    "seasonal_meridian",
    "learnable_seasonal_cdf",
    "learnable_seasonal_cdf_rz",
]

DEPTHS = [1, 2, 3]
SEEDS = [42, 43, 44, 45, 46]

METRICS = {
    "test_rmse_original": {
        "label": "Test RMSE",
        "direction": "lower",
    },
    "test_fft_spectral_cosine_similarity": {
        "label": "FFT cosine similarity",
        "direction": "higher",
    },
    "test_pearson_corr": {
        "label": "Pearson correlation",
        "direction": "higher",
    },
    "test_amplitude_variance_ratio": {
        "label": "Amplitude variance ratio",
        "direction": "target_one",
    },
}

T_CRITICAL_DF4_95 = 2.7764451051977987


# =============================================================================
# Load ideal counterparts
# =============================================================================

ideal_records = []

for encoding in ENCODINGS:
    for depth in DEPTHS:
        for seed in SEEDS:
            result_path = (
                IDEAL_RESULTS_ROOT
                / encoding
                / f"depth_{depth}"
                / f"seed_{seed}"
                / "result_summary.csv"
            )

            if not result_path.is_file():
                raise FileNotFoundError(result_path)

            frame = pd.read_csv(result_path)

            if len(frame) != 1:
                raise RuntimeError(
                    f"Expected one row in {result_path}; "
                    f"found {len(frame)}."
                )

            row = frame.iloc[0]

            required_columns = [
                "test_rmse_original",
                "test_fft_spectral_cosine_similarity",
                "test_pearson_corr",
                "test_pred_std_over_true_std",
            ]

            missing = [
                column
                for column in required_columns
                if column not in frame.columns
            ]

            if missing:
                raise RuntimeError(
                    f"Missing ideal columns in {result_path}: "
                    f"{missing}"
                )

            std_ratio = float(
                row["test_pred_std_over_true_std"]
            )

            ideal_records.append(
                {
                    "encoding": encoding,
                    "depth": depth,
                    "seed": seed,
                    "test_rmse_original":
                        float(row["test_rmse_original"]),
                    "test_fft_spectral_cosine_similarity":
                        float(
                            row[
                                "test_fft_spectral_cosine_similarity"
                            ]
                        ),
                    "test_pearson_corr":
                        float(row["test_pearson_corr"]),
                    "test_amplitude_variance_ratio":
                        std_ratio**2,
                    "ideal_result_summary":
                        str(result_path),
                }
            )

ideal = pd.DataFrame(ideal_records)

if len(ideal) != 45:
    raise RuntimeError(
        f"Expected 45 ideal runs; found {len(ideal)}."
    )


# =============================================================================
# Load noisy results and pair by encoding, depth, and seed
# =============================================================================

if not NOISY_RUN_METRICS_PATH.is_file():
    raise FileNotFoundError(NOISY_RUN_METRICS_PATH)

noisy = pd.read_csv(NOISY_RUN_METRICS_PATH)

required_noisy_columns = [
    "encoding",
    "depth",
    "seed",
    *METRICS.keys(),
]

missing_noisy = [
    column
    for column in required_noisy_columns
    if column not in noisy.columns
]

if missing_noisy:
    raise RuntimeError(
        f"Missing noisy columns: {missing_noisy}"
    )

noisy = noisy[required_noisy_columns].copy()

paired = ideal.merge(
    noisy,
    on=["encoding", "depth", "seed"],
    how="inner",
    validate="one_to_one",
    suffixes=("_ideal", "_noisy"),
)

if len(paired) != 45:
    raise RuntimeError(
        f"Expected 45 paired runs; found {len(paired)}."
    )


# =============================================================================
# Calculate paired changes
# =============================================================================

for metric, specification in METRICS.items():
    ideal_column = f"{metric}_ideal"
    noisy_column = f"{metric}_noisy"
    delta_column = f"{metric}_delta_noisy_minus_ideal"

    paired[delta_column] = (
        paired[noisy_column]
        - paired[ideal_column]
    )

    paired[
        f"{metric}_relative_change_percent"
    ] = (
        100.0
        * paired[delta_column]
        / paired[ideal_column]
    )

    if specification["direction"] == "target_one":
        paired[
            f"{metric}_ideal_target_error"
        ] = np.abs(
            paired[ideal_column] - 1.0
        )

        paired[
            f"{metric}_noisy_target_error"
        ] = np.abs(
            paired[noisy_column] - 1.0
        )

        paired[
            f"{metric}_target_error_change"
        ] = (
            paired[
                f"{metric}_noisy_target_error"
            ]
            - paired[
                f"{metric}_ideal_target_error"
            ]
        )


# =============================================================================
# Aggregate paired changes across the five matched seeds
# =============================================================================

summary_records = []

for (encoding, depth), group in paired.groupby(
    ["encoding", "depth"],
    sort=True,
):
    if len(group) != 5:
        raise RuntimeError(
            f"{encoding}, depth {depth} has "
            f"{len(group)} paired seeds."
        )

    for metric, specification in METRICS.items():
        ideal_values = group[
            f"{metric}_ideal"
        ].to_numpy(dtype=float)

        noisy_values = group[
            f"{metric}_noisy"
        ].to_numpy(dtype=float)

        differences = noisy_values - ideal_values

        difference_mean = float(
            np.mean(differences)
        )

        difference_sd = float(
            np.std(differences, ddof=1)
        )

        confidence_half_width = (
            T_CRITICAL_DF4_95
            * difference_sd
            / np.sqrt(len(differences))
        )

        record = {
            "encoding": encoding,
            "depth": depth,
            "metric": metric,
            "metric_label": specification["label"],
            "preferred_direction":
                specification["direction"],
            "n_paired_seeds": len(differences),
            "ideal_mean":
                float(np.mean(ideal_values)),
            "ideal_sd":
                float(np.std(ideal_values, ddof=1)),
            "noisy_mean":
                float(np.mean(noisy_values)),
            "noisy_sd":
                float(np.std(noisy_values, ddof=1)),
            "paired_delta_mean":
                difference_mean,
            "paired_delta_sd":
                difference_sd,
            "paired_delta_ci95_low":
                difference_mean
                - confidence_half_width,
            "paired_delta_ci95_high":
                difference_mean
                + confidence_half_width,
            "mean_relative_change_percent":
                float(
                    np.mean(
                        100.0
                        * differences
                        / ideal_values
                    )
                ),
        }

        if specification["direction"] == "target_one":
            ideal_target_error = np.abs(
                ideal_values - 1.0
            )

            noisy_target_error = np.abs(
                noisy_values - 1.0
            )

            record.update(
                {
                    "ideal_mean_distance_from_one":
                        float(
                            np.mean(
                                ideal_target_error
                            )
                        ),
                    "noisy_mean_distance_from_one":
                        float(
                            np.mean(
                                noisy_target_error
                            )
                        ),
                    "target_error_change":
                        float(
                            np.mean(
                                noisy_target_error
                                - ideal_target_error
                            )
                        ),
                }
            )

        summary_records.append(record)

summary = (
    pd.DataFrame(summary_records)
    .sort_values(
        ["metric", "encoding", "depth"]
    )
    .reset_index(drop=True)
)


# =============================================================================
# Save outputs
# =============================================================================

ideal.to_csv(
    NOISY_ANALYSIS_ROOT
    / "ideal_counterpart_metrics_by_run.csv",
    index=False,
)

paired.to_csv(
    NOISY_ANALYSIS_ROOT
    / "ideal_vs_noisy_paired_by_run.csv",
    index=False,
)

summary.to_csv(
    NOISY_ANALYSIS_ROOT
    / "ideal_vs_noisy_paired_summary.csv",
    index=False,
)


# =============================================================================
# Print concise comparison tables
# =============================================================================

print()
print("IDEAL-VERSUS-NOISY PAIRED COMPARISON")
print("Noisy change is calculated as: noisy minus ideal.")
print()

for metric, specification in METRICS.items():
    metric_table = summary[
        summary["metric"] == metric
    ][
        [
            "encoding",
            "depth",
            "ideal_mean",
            "ideal_sd",
            "noisy_mean",
            "noisy_sd",
            "paired_delta_mean",
            "paired_delta_ci95_low",
            "paired_delta_ci95_high",
        ]
    ]

    print("=" * 100)
    print(specification["label"])
    print("=" * 100)
    print(
        metric_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()

print("Saved:")
print(
    NOISY_ANALYSIS_ROOT
    / "ideal_counterpart_metrics_by_run.csv"
)
print(
    NOISY_ANALYSIS_ROOT
    / "ideal_vs_noisy_paired_by_run.csv"
)
print(
    NOISY_ANALYSIS_ROOT
    / "ideal_vs_noisy_paired_summary.csv"
)
