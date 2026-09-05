#!/usr/bin/env python

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(
    "/umbc/rs/pi_deffner/users/devjyot1/projects/"
    "Quantum-Machine-Learning-"
)

OUT_ROOT = (
    PROJECT_ROOT
    / "data_reupload"
    / "analysis"
    / "four_encoding_master"
    / "output"
)

TABLE_DIR = OUT_ROOT / "tables"
VALIDATION_DIR = OUT_ROOT / "validation"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Canonical encoding definitions
# ============================================================

FAMILIES = [
    {
        "encoding": "ry",
        "encoding_label": "Standard Ry",
        "order": 1,
        "root": (
            PROJECT_ROOT
            / "data_reupload/results/final_model_sweep/"
            "single_tanh_a1"
        ),
        "is_seasonal": False,
        "is_learnable": False,
        "encoding_parameter_formula": "0",
        "encoding_parameters_per_layer": 0,
        "total_parameter_formula": "42*L + 15",
    },
    {
        "encoding": "same_scalar_rxryrz",
        "encoding_label": "Same-scalar RxRyRz",
        "order": 2,
        "root": (
            PROJECT_ROOT
            / "data_reupload/results/encoding_sweep/"
            "same_scalar_rxryrz_tanh_a1"
        ),
        "is_seasonal": False,
        "is_learnable": False,
        "encoding_parameter_formula": "0",
        "encoding_parameters_per_layer": 0,
        "total_parameter_formula": "42*L + 15",
    },
    {
        "encoding": "seasonal_meridian",
        "encoding_label": "Fixed seasonal CDF",
        "order": 3,
        "root": (
            PROJECT_ROOT
            / "data_reupload/results/encoding_sweep/"
            "seasonal_meridian_tanh_a1"
        ),
        "is_seasonal": True,
        "is_learnable": False,
        "encoding_parameter_formula": "0",
        "encoding_parameters_per_layer": 0,
        "total_parameter_formula": "42*L + 15",
    },
    {
        "encoding": "learnable_seasonal_cdf",
        "encoding_label": "Learnable seasonal CDF",
        "order": 4,
        "root": (
            PROJECT_ROOT
            / "data_reupload/results/encoding_sweep/"
            "learnable_seasonal_cdf_tanh_a1"
        ),
        "is_seasonal": True,
        "is_learnable": True,
        "encoding_parameter_formula": "24*L",
        "encoding_parameters_per_layer": 24,
        "total_parameter_formula": "66*L + 15",
    },
    {
        "encoding": "learnable_seasonal_cdf_rz",
        "encoding_label": "Learnable polar-azimuth seasonal CDF",
        "order": 5,
        "root": (
            PROJECT_ROOT
            / "data_reupload/results/encoding_sweep/"
            "learnable_seasonal_cdf_rz_tanh_a1"
        ),
        "is_seasonal": True,
        "is_learnable": True,
        "encoding_parameter_formula": "36*L",
        "encoding_parameters_per_layer": 36,
        "total_parameter_formula": "78*L + 15",
    },
]


EXPECTED_DEPTHS = list(range(1, 7))
EXPECTED_SEEDS = list(range(42, 47))
EXPECTED_RUNS = (
    len(FAMILIES)
    * len(EXPECTED_DEPTHS)
    * len(EXPECTED_SEEDS)
)

EXPECTED_COUNTS = {
    "train": 350,
    "val": 70,
    "test": 105,
}

EXPECTED_TARGET_INDICES = {
    "train": np.arange(14, 364, dtype=np.int64),
    "val": np.arange(364, 434, dtype=np.int64),
    "test": np.arange(434, 539, dtype=np.int64),
}


# ============================================================
# Helpers
# ============================================================

def scalar_or_nan(row, key):
    if key not in row.index:
        return np.nan

    value = row[key]

    if pd.isna(value):
        return np.nan

    return value


def load_json_if_exists(path):
    if not path.exists():
        return {}

    with open(path) as f:
        return json.load(f)


def require_file(path):
    if not path.exists():
        raise RuntimeError(f"Missing required file: {path}")

    if path.stat().st_size == 0:
        raise RuntimeError(f"Empty required file: {path}")


def array_max_abs_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        return np.inf

    if a.size == 0:
        return 0.0

    return float(
        np.max(
            np.abs(
                a.astype(np.float64)
                -
                b.astype(np.float64)
            )
        )
    )


# ============================================================
# Canonical reference arrays
# ============================================================

reference = None

manifest_rows = []
inventory_rows = []
truth_audit_rows = []
metric_audit_rows = []
spectral_audit_rows = []


# ============================================================
# Load all 120 runs
# ============================================================

for fam in FAMILIES:

    encoding = fam["encoding"]
    label = fam["encoding_label"]
    root = fam["root"]

    family_run_count = 0

    for L in EXPECTED_DEPTHS:

        for seed in EXPECTED_SEEDS:

            run_dir = (
                root
                / f"depth_{L}"
                / f"seed_{seed}"
            )

            result_path = run_dir / "result_summary.csv"
            pred_path = run_dir / "predictions.npz"
            history_path = run_dir / "history.csv"
            metrics_path = run_dir / "metrics_by_split.csv"
            fft_summary_path = run_dir / "fft_summary_by_split.csv"
            fft_power_path = run_dir / "fft_power_by_period.csv"
            checkpoint_path = run_dir / "best_model.pt"
            config_path = run_dir / "config.json"
            metadata_path = run_dir / "encoding_metadata.json"
            learned_path = run_dir / "learned_encoding_parameters.npz"

            required = [
                result_path,
                pred_path,
                history_path,
                metrics_path,
                fft_summary_path,
                fft_power_path,
                checkpoint_path,
                config_path,
            ]

            for p in required:
                require_file(p)

            if fam["is_learnable"]:
                require_file(metadata_path)
                require_file(learned_path)

            result = pd.read_csv(result_path)

            if len(result) != 1:
                raise RuntimeError(
                    f"Expected exactly one result row: {result_path}"
                )

            r = result.iloc[0]

            metrics = pd.read_csv(metrics_path)
            fft_summary = pd.read_csv(fft_summary_path)

            if set(metrics["split"]) != {"train", "val", "test"}:
                raise RuntimeError(
                    f"Unexpected metrics splits in {metrics_path}"
                )

            if set(fft_summary["split"]) != {"train", "val", "test"}:
                raise RuntimeError(
                    f"Unexpected FFT splits in {fft_summary_path}"
                )

            pred = np.load(
                pred_path,
                allow_pickle=True,
            )

            cfg = load_json_if_exists(config_path)
            metadata = load_json_if_exists(metadata_path)

            # ------------------------------------------------
            # Establish global truth/index reference
            # ------------------------------------------------

            current = {}

            for split in ["train", "val", "test"]:

                current[f"y_{split}_true_scaled"] = np.asarray(
                    pred[f"y_{split}_true_scaled"]
                )

                current[f"y_{split}_true_original"] = np.asarray(
                    pred[f"y_{split}_true_original"]
                )

                current[f"{split}_target_indices"] = np.asarray(
                    pred[f"{split}_target_indices"]
                )

            if reference is None:
                reference = {
                    key: value.copy()
                    for key, value
                    in current.items()
                }

            # ------------------------------------------------
            # Cross-run truth/index validation
            # ------------------------------------------------

            truth_pass = True

            truth_record = {
                "encoding": encoding,
                "encoding_label": label,
                "depth": L,
                "seed": seed,
            }

            for split in ["train", "val", "test"]:

                idx = current[f"{split}_target_indices"]
                y_scaled = current[f"y_{split}_true_scaled"]
                y_original = current[f"y_{split}_true_original"]

                if len(idx) != EXPECTED_COUNTS[split]:
                    truth_pass = False

                if len(y_scaled) != EXPECTED_COUNTS[split]:
                    truth_pass = False

                if len(y_original) != EXPECTED_COUNTS[split]:
                    truth_pass = False

                expected_idx = EXPECTED_TARGET_INDICES[split]

                idx_diff_expected = (
                    0.0
                    if np.array_equal(idx, expected_idx)
                    else np.inf
                )

                idx_diff_reference = (
                    0.0
                    if np.array_equal(
                        idx,
                        reference[f"{split}_target_indices"],
                    )
                    else np.inf
                )

                scaled_diff = array_max_abs_diff(
                    y_scaled,
                    reference[f"y_{split}_true_scaled"],
                )

                original_diff = array_max_abs_diff(
                    y_original,
                    reference[f"y_{split}_true_original"],
                )

                truth_record[
                    f"{split}_index_matches_expected"
                ] = bool(
                    idx_diff_expected == 0.0
                )

                truth_record[
                    f"{split}_index_matches_reference"
                ] = bool(
                    idx_diff_reference == 0.0
                )

                truth_record[
                    f"{split}_scaled_truth_max_abs_diff"
                ] = scaled_diff

                truth_record[
                    f"{split}_original_truth_max_abs_diff"
                ] = original_diff

                if (
                    idx_diff_expected != 0.0
                    or
                    idx_diff_reference != 0.0
                    or
                    scaled_diff > 1e-12
                    or
                    original_diff > 1e-12
                ):
                    truth_pass = False

            truth_record["status"] = (
                "PASS"
                if truth_pass
                else "FAIL"
            )

            truth_audit_rows.append(
                truth_record
            )

            if not truth_pass:
                raise RuntimeError(
                    f"Truth/index audit failed for "
                    f"{encoding}, L={L}, seed={seed}"
                )

            # ------------------------------------------------
            # Validate metrics_by_split against result_summary
            # ------------------------------------------------

            metric_pass = True
            metric_max_diff = 0.0

            result_metric_map = {
                "rmse": "rmse_original",
                "mae": "mae_original",
                "pearson_corr": "pearson_corr",
                "spearman_corr": "spearman_corr",
                "kge": "kge",
                "nse": "nse",
                "bias_mean_pred_minus_true": "bias",
                "pred_std_over_true_std": "pred_std_over_true_std",
            }

            for split in ["train", "val", "test"]:

                mrow = (
                    metrics[
                        metrics["split"] == split
                    ]
                    .iloc[0]
                )

                if int(mrow["n"]) != EXPECTED_COUNTS[split]:
                    metric_pass = False

                for metrics_col, result_suffix in result_metric_map.items():

                    result_col = (
                        f"{split}_{result_suffix}"
                    )

                    if result_col not in r.index:
                        continue

                    diff = abs(
                        float(mrow[metrics_col])
                        -
                        float(r[result_col])
                    )

                    metric_max_diff = max(
                        metric_max_diff,
                        diff,
                    )

                    if diff > 1e-10:
                        metric_pass = False

            metric_audit_rows.append({
                "encoding": encoding,
                "depth": L,
                "seed": seed,
                "max_abs_metric_difference":
                    metric_max_diff,
                "status":
                    "PASS" if metric_pass else "FAIL",
            })

            if not metric_pass:
                raise RuntimeError(
                    f"Metric audit failed for "
                    f"{encoding}, L={L}, seed={seed}; "
                    f"max diff={metric_max_diff}"
                )

            # ------------------------------------------------
            # Validate spectral summary
            # ------------------------------------------------

            spectral_pass = True
            spectral_max_reconstruction_error = 0.0
            spectral_max_result_difference = 0.0

            for split in ["train", "val", "test"]:

                frow = (
                    fft_summary[
                        fft_summary["split"] == split
                    ]
                    .iloc[0]
                )

                reconstructed_e12 = abs(
                    float(
                        frow[
                            "pred_12_step_band_power_share"
                        ]
                    )
                    -
                    float(
                        frow[
                            "true_12_step_band_power_share"
                        ]
                    )
                )

                reconstructed_eshort = abs(
                    float(
                        frow[
                            "pred_short_period_2_to_6_power_share"
                        ]
                    )
                    -
                    float(
                        frow[
                            "true_short_period_2_to_6_power_share"
                        ]
                    )
                )

                e12_error = abs(
                    reconstructed_e12
                    -
                    float(
                        frow[
                            "abs_12_step_band_error"
                        ]
                    )
                )

                eshort_error = abs(
                    reconstructed_eshort
                    -
                    float(
                        frow[
                            "abs_short_period_2_to_6_error"
                        ]
                    )
                )

                spectral_max_reconstruction_error = max(
                    spectral_max_reconstruction_error,
                    e12_error,
                    eshort_error,
                )

                if (
                    e12_error > 1e-12
                    or
                    eshort_error > 1e-12
                ):
                    spectral_pass = False

            # Compare test FFT metrics to result_summary.
            test_fft = (
                fft_summary[
                    fft_summary["split"] == "test"
                ]
                .iloc[0]
            )

            comparisons = {
                "test_fft_spectral_cosine_similarity":
                    "fft_spectral_cosine_similarity",

                "test_abs_12_step_band_error":
                    "abs_12_step_band_error",

                "test_abs_short_period_2_to_6_error":
                    "abs_short_period_2_to_6_error",
            }

            for result_col, fft_col in comparisons.items():

                if result_col not in r.index:
                    continue

                diff = abs(
                    float(r[result_col])
                    -
                    float(test_fft[fft_col])
                )

                spectral_max_result_difference = max(
                    spectral_max_result_difference,
                    diff,
                )

                if diff > 1e-10:
                    spectral_pass = False

            spectral_audit_rows.append({
                "encoding": encoding,
                "depth": L,
                "seed": seed,
                "max_band_reconstruction_error":
                    spectral_max_reconstruction_error,
                "max_result_summary_difference":
                    spectral_max_result_difference,
                "status":
                    "PASS" if spectral_pass else "FAIL",
            })

            if not spectral_pass:
                raise RuntimeError(
                    f"Spectral audit failed for "
                    f"{encoding}, L={L}, seed={seed}"
                )

            # ------------------------------------------------
            # Parameter counts
            # ------------------------------------------------

            variational_parameters = 42 * L
            head_parameters = 15

            encoding_parameters = (
                int(fam["encoding_parameters_per_layer"])
                * L
            )

            expected_total_parameters = (
                variational_parameters
                +
                encoding_parameters
                +
                head_parameters
            )

            if fam["is_learnable"]:

                stored_encoding_parameters = int(
                    scalar_or_nan(
                        r,
                        "encoding_parameter_count",
                    )
                )

                stored_total_parameters = int(
                    scalar_or_nan(
                        r,
                        "total_parameter_count",
                    )
                )

                if (
                    stored_encoding_parameters
                    != encoding_parameters
                ):
                    raise RuntimeError(
                        "Learnable encoding parameter "
                        f"count mismatch L={L}, seed={seed}"
                    )

                if (
                    stored_total_parameters
                    != expected_total_parameters
                ):
                    raise RuntimeError(
                        "Learnable total parameter "
                        f"count mismatch L={L}, seed={seed}"
                    )

            # ------------------------------------------------
            # Canonical test FFT row
            # ------------------------------------------------

            test_fft = (
                fft_summary[
                    fft_summary["split"] == "test"
                ]
                .iloc[0]
            )

            # Signed differences.
            signed_12 = (
                float(
                    test_fft[
                        "pred_12_step_band_power_share"
                    ]
                )
                -
                float(
                    test_fft[
                        "true_12_step_band_power_share"
                    ]
                )
            )

            signed_short = (
                float(
                    test_fft[
                        "pred_short_period_2_to_6_power_share"
                    ]
                )
                -
                float(
                    test_fft[
                        "true_short_period_2_to_6_power_share"
                    ]
                )
            )

            true_12 = float(
                test_fft[
                    "true_12_step_band_power_share"
                ]
            )

            pred_12 = float(
                test_fft[
                    "pred_12_step_band_power_share"
                ]
            )

            true_short = float(
                test_fft[
                    "true_short_period_2_to_6_power_share"
                ]
            )

            pred_short = float(
                test_fft[
                    "pred_short_period_2_to_6_power_share"
                ]
            )

            ratio_12 = (
                pred_12 / true_12
                if true_12 != 0
                else np.nan
            )

            ratio_short = (
                pred_short / true_short
                if true_short != 0
                else np.nan
            )

            # ------------------------------------------------
            # Main manifest row
            # ------------------------------------------------

            manifest_rows.append({

                # Identification
                "encoding_order": fam["order"],
                "encoding": encoding,
                "encoding_label": label,
                "depth": L,
                "seed": seed,
                "ansatz_layers": 1,
                "n_qubits": 14,

                # Encoding properties
                "is_seasonal": fam["is_seasonal"],
                "is_learnable_encoding":
                    fam["is_learnable"],

                # Counts
                "variational_parameter_count":
                    variational_parameters,
                "encoding_parameter_count":
                    encoding_parameters,
                "classical_head_parameter_count":
                    head_parameters,
                "total_parameter_count":
                    expected_total_parameters,

                # Checkpoint
                "best_epoch":
                    int(r["best_epoch"]),
                "best_val_loss_scaled":
                    float(r["best_val_loss_scaled"]),

                # Train
                "train_rmse_original":
                    float(r["train_rmse_original"]),
                "train_mae_original":
                    float(r["train_mae_original"]),
                "train_pearson_corr":
                    float(r["train_pearson_corr"]),
                "train_spearman_corr":
                    float(r["train_spearman_corr"]),
                "train_kge":
                    float(r["train_kge"]),
                "train_nse":
                    float(r["train_nse"]),
                "train_bias":
                    float(r["train_bias"]),
                "train_pred_std_over_true_std":
                    float(
                        r[
                            "train_pred_std_over_true_std"
                        ]
                    ),

                # Validation
                "val_rmse_original":
                    float(r["val_rmse_original"]),
                "val_mae_original":
                    float(r["val_mae_original"]),
                "val_pearson_corr":
                    float(r["val_pearson_corr"]),
                "val_spearman_corr":
                    float(r["val_spearman_corr"]),
                "val_kge":
                    float(r["val_kge"]),
                "val_nse":
                    float(r["val_nse"]),
                "val_bias":
                    float(r["val_bias"]),
                "val_pred_std_over_true_std":
                    float(
                        r[
                            "val_pred_std_over_true_std"
                        ]
                    ),

                # Test
                "test_rmse_original":
                    float(r["test_rmse_original"]),
                "test_mae_original":
                    float(r["test_mae_original"]),
                "test_pearson_corr":
                    float(r["test_pearson_corr"]),
                "test_spearman_corr":
                    float(r["test_spearman_corr"]),
                "test_kge":
                    float(r["test_kge"]),
                "test_nse":
                    float(r["test_nse"]),
                "test_bias":
                    float(r["test_bias"]),
                "test_pred_std_over_true_std":
                    float(
                        r[
                            "test_pred_std_over_true_std"
                        ]
                    ),

                # Test spectral
                "test_fft_spectral_cosine_similarity":
                    float(
                        test_fft[
                            "fft_spectral_cosine_similarity"
                        ]
                    ),

                "test_true_12_step_band_power_share":
                    true_12,

                "test_pred_12_step_band_power_share":
                    pred_12,

                "test_abs_12_step_band_error":
                    float(
                        test_fft[
                            "abs_12_step_band_error"
                        ]
                    ),

                "test_signed_12_step_band_error":
                    signed_12,

                "test_12_step_power_ratio":
                    ratio_12,

                "test_true_short_period_2_to_6_power_share":
                    true_short,

                "test_pred_short_period_2_to_6_power_share":
                    pred_short,

                "test_abs_short_period_2_to_6_error":
                    float(
                        test_fft[
                            "abs_short_period_2_to_6_error"
                        ]
                    ),

                "test_signed_short_period_2_to_6_error":
                    signed_short,

                "test_short_period_2_to_6_power_ratio":
                    ratio_short,

                "test_true_dominant_period":
                    float(
                        test_fft[
                            "true_dominant_period"
                        ]
                    ),

                "test_pred_dominant_period":
                    float(
                        test_fft[
                            "pred_dominant_period"
                        ]
                    ),

                # Runtime/resource
                "elapsed_time_sec":
                    float(r["elapsed_time_sec"]),

                "elapsed_time_min":
                    float(r["elapsed_time_sec"]) / 60.0,

                "max_rss_gb_python":
                    float(
                        scalar_or_nan(
                            r,
                            "max_rss_gb_python",
                        )
                    ),

                # Learnable-specific run summaries.
                "mean_final_reupload_strength":
                    float(
                        scalar_or_nan(
                            r,
                            "mean_final_reupload_strength",
                        )
                    ),

                "mean_abs_strength_deviation":
                    float(
                        scalar_or_nan(
                            r,
                            "mean_abs_strength_deviation",
                        )
                    ),

                "mean_abs_baseline_deviation":
                    float(
                        scalar_or_nan(
                            r,
                            "mean_abs_baseline_deviation",
                        )
                    ),

                # Exact paths used later by geometry.
                "run_dir":
                    str(run_dir),

                "checkpoint_path":
                    str(checkpoint_path),

                "predictions_path":
                    str(pred_path),

                "history_path":
                    str(history_path),

                "metrics_by_split_path":
                    str(metrics_path),

                "fft_summary_path":
                    str(fft_summary_path),

                "fft_power_path":
                    str(fft_power_path),

                "config_path":
                    str(config_path),

                "encoding_metadata_path":
                    (
                        str(metadata_path)
                        if metadata_path.exists()
                        else ""
                    ),

                "learned_encoding_parameters_path":
                    (
                        str(learned_path)
                        if learned_path.exists()
                        else ""
                    ),
            })

            inventory_rows.append({
                "encoding": encoding,
                "encoding_label": label,
                "depth": L,
                "seed": seed,
                "run_dir": str(run_dir),
                "result_summary_exists":
                    result_path.exists(),
                "checkpoint_exists":
                    checkpoint_path.exists(),
                "predictions_exists":
                    pred_path.exists(),
                "metrics_exists":
                    metrics_path.exists(),
                "fft_summary_exists":
                    fft_summary_path.exists(),
                "fft_power_exists":
                    fft_power_path.exists(),
                "config_exists":
                    config_path.exists(),
                "encoding_metadata_exists":
                    metadata_path.exists(),
                "learned_encoding_exists":
                    learned_path.exists(),
            })

            family_run_count += 1

    if family_run_count != 30:
        raise RuntimeError(
            f"{encoding}: expected 30 runs, "
            f"found {family_run_count}"
        )


# ============================================================
# Assemble master tables
# ============================================================

manifest = pd.DataFrame(
    manifest_rows
).sort_values(
    [
        "encoding_order",
        "depth",
        "seed",
    ]
).reset_index(drop=True)

inventory = pd.DataFrame(
    inventory_rows
)

truth_audit = pd.DataFrame(
    truth_audit_rows
)

metric_audit = pd.DataFrame(
    metric_audit_rows
)

spectral_audit = pd.DataFrame(
    spectral_audit_rows
)


# ============================================================
# Hard global assertions
# ============================================================

if len(manifest) != EXPECTED_RUNS:
    raise RuntimeError(
        f"Expected {EXPECTED_RUNS} runs, "
        f"found {len(manifest)}"
    )

counts = (
    manifest
    .groupby("encoding")
    .size()
    .to_dict()
)

for fam in FAMILIES:

    if counts.get(fam["encoding"], 0) != 30:
        raise RuntimeError(
            f"Run count mismatch for "
            f"{fam['encoding']}: "
            f"{counts.get(fam['encoding'], 0)}"
        )

if not (
    truth_audit["status"] == "PASS"
).all():
    raise RuntimeError(
        "At least one truth/index audit failed."
    )

if not (
    metric_audit["status"] == "PASS"
).all():
    raise RuntimeError(
        "At least one metric audit failed."
    )

if not (
    spectral_audit["status"] == "PASS"
).all():
    raise RuntimeError(
        "At least one spectral audit failed."
    )


# ============================================================
# Five-seed summaries
# ============================================================

SUMMARY_METRICS = [
    "train_rmse_original",
    "val_rmse_original",
    "test_rmse_original",
    "test_mae_original",
    "test_pearson_corr",
    "test_spearman_corr",
    "test_kge",
    "test_nse",
    "test_bias",
    "test_pred_std_over_true_std",
    "test_fft_spectral_cosine_similarity",
    "test_abs_12_step_band_error",
    "test_signed_12_step_band_error",
    "test_12_step_power_ratio",
    "test_abs_short_period_2_to_6_error",
    "test_signed_short_period_2_to_6_error",
    "test_short_period_2_to_6_power_ratio",
    "elapsed_time_min",
    "max_rss_gb_python",
]

summary_rows = []

for fam in FAMILIES:

    enc = fam["encoding"]

    for L in EXPECTED_DEPTHS:

        d = manifest[
            (manifest["encoding"] == enc)
            &
            (manifest["depth"] == L)
        ]

        if len(d) != 5:
            raise RuntimeError(
                f"Expected 5 seeds for "
                f"{enc}, L={L}; found {len(d)}"
            )

        row = {
            "encoding_order": fam["order"],
            "encoding": enc,
            "encoding_label":
                fam["encoding_label"],
            "depth": L,
            "n_seeds": 5,
            "variational_parameter_count":
                int(
                    d[
                        "variational_parameter_count"
                    ].iloc[0]
                ),
            "encoding_parameter_count":
                int(
                    d[
                        "encoding_parameter_count"
                    ].iloc[0]
                ),
            "total_parameter_count":
                int(
                    d[
                        "total_parameter_count"
                    ].iloc[0]
                ),
        }

        for metric in SUMMARY_METRICS:

            vals = pd.to_numeric(
                d[metric],
                errors="coerce",
            ).to_numpy(
                dtype=np.float64
            )

            row[f"{metric}_mean"] = float(
                np.nanmean(vals)
            )

            row[f"{metric}_std"] = float(
                np.nanstd(
                    vals,
                    ddof=1,
                )
            )

        summary_rows.append(row)

depth_summary = pd.DataFrame(
    summary_rows
).sort_values(
    [
        "encoding_order",
        "depth",
    ]
).reset_index(drop=True)


# ============================================================
# Paired seedwise comparisons against standard Ry
# ============================================================

paired_rows = []

baseline = manifest[
    manifest["encoding"] == "ry"
].copy()

for comparison_encoding in [
    "same_scalar_rxryrz",
    "seasonal_meridian",
    "learnable_seasonal_cdf",
    "learnable_seasonal_cdf_rz",
]:

    candidate = manifest[
        manifest["encoding"]
        ==
        comparison_encoding
    ].copy()

    merged = baseline.merge(
        candidate,
        on=[
            "depth",
            "seed",
        ],
        suffixes=(
            "_ry",
            "_candidate",
        ),
        validate="one_to_one",
    )

    for _, row in merged.iterrows():

        paired_rows.append({
            "comparison":
                f"ry_vs_{comparison_encoding}",

            "candidate_encoding":
                comparison_encoding,

            "depth":
                int(row["depth"]),

            "seed":
                int(row["seed"]),

            # Positive means candidate is better.
            "rmse_improvement_vs_ry":
                float(
                    row["test_rmse_original_ry"]
                    -
                    row[
                        "test_rmse_original_candidate"
                    ]
                ),

            "mae_improvement_vs_ry":
                float(
                    row["test_mae_original_ry"]
                    -
                    row[
                        "test_mae_original_candidate"
                    ]
                ),

            "pearson_improvement_vs_ry":
                float(
                    row[
                        "test_pearson_corr_candidate"
                    ]
                    -
                    row[
                        "test_pearson_corr_ry"
                    ]
                ),

            "kge_improvement_vs_ry":
                float(
                    row[
                        "test_kge_candidate"
                    ]
                    -
                    row[
                        "test_kge_ry"
                    ]
                ),

            "fft_improvement_vs_ry":
                float(
                    row[
                        "test_fft_spectral_cosine_similarity_candidate"
                    ]
                    -
                    row[
                        "test_fft_spectral_cosine_similarity_ry"
                    ]
                ),
        })

paired_vs_ry = pd.DataFrame(
    paired_rows
)


# ============================================================
# Fixed seasonal vs learnable seasonal paired comparison
# ============================================================

fixed = manifest[
    manifest["encoding"] == "seasonal_meridian"
]

learn = manifest[
    manifest["encoding"] == "learnable_seasonal_cdf"
]

fixed_learn = fixed.merge(
    learn,
    on=[
        "depth",
        "seed",
    ],
    suffixes=(
        "_fixed",
        "_learnable",
    ),
    validate="one_to_one",
)

fixed_learn_rows = []

for _, row in fixed_learn.iterrows():

    fixed_learn_rows.append({
        "depth":
            int(row["depth"]),

        "seed":
            int(row["seed"]),

        # Positive means learnable is better.
        "rmse_improvement_learnable_vs_fixed":
            float(
                row[
                    "test_rmse_original_fixed"
                ]
                -
                row[
                    "test_rmse_original_learnable"
                ]
            ),

        "mae_improvement_learnable_vs_fixed":
            float(
                row[
                    "test_mae_original_fixed"
                ]
                -
                row[
                    "test_mae_original_learnable"
                ]
            ),

        "pearson_improvement_learnable_vs_fixed":
            float(
                row[
                    "test_pearson_corr_learnable"
                ]
                -
                row[
                    "test_pearson_corr_fixed"
                ]
            ),

        "kge_improvement_learnable_vs_fixed":
            float(
                row[
                    "test_kge_learnable"
                ]
                -
                row[
                    "test_kge_fixed"
                ]
            ),

        "fft_improvement_learnable_vs_fixed":
            float(
                row[
                    "test_fft_spectral_cosine_similarity_learnable"
                ]
                -
                row[
                    "test_fft_spectral_cosine_similarity_fixed"
                ]
            ),
    })

paired_fixed_learn = pd.DataFrame(
    fixed_learn_rows
)



# ============================================================
# Learnable polar-only vs learnable polar+azimuth
# paired comparison
# ============================================================

polar = manifest[
    manifest["encoding"] == "learnable_seasonal_cdf"
].copy()

azimuth = manifest[
    manifest["encoding"] == "learnable_seasonal_cdf_rz"
].copy()

polar_azimuth = polar.merge(
    azimuth,
    on=[
        "depth",
        "seed",
    ],
    suffixes=(
        "_polar",
        "_azimuth",
    ),
    validate="one_to_one",
)

if len(polar_azimuth) != 30:
    raise RuntimeError(
        "Expected exactly 30 matched model-4/model-5 pairs; "
        f"found {len(polar_azimuth)}"
    )

polar_azimuth_rows = []

for _, row in polar_azimuth.iterrows():

    polar_azimuth_rows.append({
        "depth":
            int(row["depth"]),

        "seed":
            int(row["seed"]),

        # Positive means polar+azimuth (model 5) is better.
        "rmse_improvement_azimuth_vs_polar":
            float(
                row["test_rmse_original_polar"]
                -
                row["test_rmse_original_azimuth"]
            ),

        "mae_improvement_azimuth_vs_polar":
            float(
                row["test_mae_original_polar"]
                -
                row["test_mae_original_azimuth"]
            ),

        "pearson_improvement_azimuth_vs_polar":
            float(
                row["test_pearson_corr_azimuth"]
                -
                row["test_pearson_corr_polar"]
            ),

        "spearman_improvement_azimuth_vs_polar":
            float(
                row["test_spearman_corr_azimuth"]
                -
                row["test_spearman_corr_polar"]
            ),

        "kge_improvement_azimuth_vs_polar":
            float(
                row["test_kge_azimuth"]
                -
                row["test_kge_polar"]
            ),

        "fft_improvement_azimuth_vs_polar":
            float(
                row[
                    "test_fft_spectral_cosine_similarity_azimuth"
                ]
                -
                row[
                    "test_fft_spectral_cosine_similarity_polar"
                ]
            ),

        "amplitude_ratio_change_azimuth_vs_polar":
            float(
                row[
                    "test_pred_std_over_true_std_azimuth"
                ]
                -
                row[
                    "test_pred_std_over_true_std_polar"
                ]
            ),
    })

paired_polar_azimuth = pd.DataFrame(
    polar_azimuth_rows
).sort_values(
    [
        "depth",
        "seed",
    ]
).reset_index(drop=True)


polar_azimuth_summary_rows = []

for L in EXPECTED_DEPTHS:

    d = paired_polar_azimuth[
        paired_polar_azimuth["depth"] == L
    ]

    if len(d) != 5:
        raise RuntimeError(
            f"Expected 5 model-4/model-5 pairs at L={L}; "
            f"found {len(d)}"
        )

    rmse_delta = d[
        "rmse_improvement_azimuth_vs_polar"
    ].to_numpy()

    polar_azimuth_summary_rows.append({
        "depth":
            L,

        "n_pairs":
            len(d),

        "mean_rmse_improvement_azimuth_vs_polar":
            float(
                np.mean(rmse_delta)
            ),

        "std_rmse_improvement_azimuth_vs_polar":
            float(
                np.std(
                    rmse_delta,
                    ddof=1,
                )
            ),

        "median_rmse_improvement_azimuth_vs_polar":
            float(
                np.median(rmse_delta)
            ),

        "azimuth_better_seed_count":
            int(
                np.sum(
                    rmse_delta > 0.0
                )
            ),

        "mean_mae_improvement_azimuth_vs_polar":
            float(
                d[
                    "mae_improvement_azimuth_vs_polar"
                ].mean()
            ),

        "mean_pearson_improvement_azimuth_vs_polar":
            float(
                d[
                    "pearson_improvement_azimuth_vs_polar"
                ].mean()
            ),

        "mean_spearman_improvement_azimuth_vs_polar":
            float(
                d[
                    "spearman_improvement_azimuth_vs_polar"
                ].mean()
            ),

        "mean_kge_improvement_azimuth_vs_polar":
            float(
                d[
                    "kge_improvement_azimuth_vs_polar"
                ].mean()
            ),

        "mean_fft_improvement_azimuth_vs_polar":
            float(
                d[
                    "fft_improvement_azimuth_vs_polar"
                ].mean()
            ),

        "mean_amplitude_ratio_change_azimuth_vs_polar":
            float(
                d[
                    "amplitude_ratio_change_azimuth_vs_polar"
                ].mean()
            ),
    })


paired_polar_azimuth_depth_summary = pd.DataFrame(
    polar_azimuth_summary_rows
)


# ============================================================
# Save
# ============================================================

paired_polar_azimuth.to_csv(
    TABLE_DIR
    / "paired_seedwise_azimuth_vs_polar.csv",
    index=False,
)

paired_polar_azimuth_depth_summary.to_csv(
    TABLE_DIR
    / "paired_rmse_azimuth_vs_polar_depth_summary.csv",
    index=False,
)

manifest.to_csv(
    TABLE_DIR
    / "master_run_manifest.csv",
    index=False,
)

depth_summary.to_csv(
    TABLE_DIR
    / "forecasting_depth_summary.csv",
    index=False,
)

inventory.to_csv(
    VALIDATION_DIR
    / "run_inventory.csv",
    index=False,
)

truth_audit.to_csv(
    VALIDATION_DIR
    / "cross_encoding_truth_audit.csv",
    index=False,
)

metric_audit.to_csv(
    VALIDATION_DIR
    / "metric_consistency_audit.csv",
    index=False,
)

spectral_audit.to_csv(
    VALIDATION_DIR
    / "spectral_consistency_audit.csv",
    index=False,
)

paired_vs_ry.to_csv(
    TABLE_DIR
    / "paired_seedwise_vs_ry.csv",
    index=False,
)

paired_fixed_learn.to_csv(
    TABLE_DIR
    / "paired_seedwise_learnable_vs_fixed.csv",
    index=False,
)


# ============================================================
# Global validation JSON
# ============================================================

validation = {
    "status": "PASS",

    "expected_runs":
        EXPECTED_RUNS,

    "loaded_runs":
        int(len(manifest)),

    "encodings": {
        fam["encoding"]:
            int(
                (
                    manifest["encoding"]
                    ==
                    fam["encoding"]
                ).sum()
            )
        for fam
        in FAMILIES
    },

    "depths":
        EXPECTED_DEPTHS,

    "seeds":
        EXPECTED_SEEDS,

    "train_target_count":
        EXPECTED_COUNTS["train"],

    "val_target_count":
        EXPECTED_COUNTS["val"],

    "test_target_count":
        EXPECTED_COUNTS["test"],

    "train_target_index_range":
        [14, 363],

    "val_target_index_range":
        [364, 433],

    "test_target_index_range":
        [434, 538],

    "all_truth_arrays_identical":
        True,

    "all_target_indices_identical":
        True,

    "max_scaled_truth_difference":
        float(
            max(
                truth_audit[
                    [
                        "train_scaled_truth_max_abs_diff",
                        "val_scaled_truth_max_abs_diff",
                        "test_scaled_truth_max_abs_diff",
                    ]
                ]
                .to_numpy()
                .max(),
                0.0,
            )
        ),

    "max_original_truth_difference":
        float(
            max(
                truth_audit[
                    [
                        "train_original_truth_max_abs_diff",
                        "val_original_truth_max_abs_diff",
                        "test_original_truth_max_abs_diff",
                    ]
                ]
                .to_numpy()
                .max(),
                0.0,
            )
        ),

    "max_metric_consistency_difference":
        float(
            metric_audit[
                "max_abs_metric_difference"
            ].max()
        ),

    "max_spectral_band_reconstruction_error":
        float(
            spectral_audit[
                "max_band_reconstruction_error"
            ].max()
        ),

    "max_spectral_result_summary_difference":
        float(
            spectral_audit[
                "max_result_summary_difference"
            ].max()
        ),

    "fixed_encoding_parameter_formula":
        "42*L + 15",

    "learnable_encoding_parameter_formula":
        "66*L + 15",

    "learnable_extra_encoding_parameters":
        "24*L",

    "cdf_reference_n":
        364,

    "season_period":
        12,
}

with open(
    VALIDATION_DIR
    / "master_validation.json",
    "w",
) as f:

    json.dump(
        validation,
        f,
        indent=2,
    )


# ============================================================
# Human-readable output
# ============================================================

print("=" * 120)
print("FIVE-ENCODING MASTER DATASET")
print("=" * 120)

print()
print(
    "Expected runs:",
    EXPECTED_RUNS,
)

print(
    "Loaded runs:  ",
    len(manifest),
)

print()

for fam in FAMILIES:

    n = int(
        (
            manifest["encoding"]
            ==
            fam["encoding"]
        ).sum()
    )

    print(
        f"{fam['encoding_label']:<28s}: "
        f"{n}"
    )

print()
print(
    "Train targets:",
    EXPECTED_COUNTS["train"],
    "indices 14-363",
)

print(
    "Val targets:  ",
    EXPECTED_COUNTS["val"],
    "indices 364-433",
)

print(
    "Test targets: ",
    EXPECTED_COUNTS["test"],
    "indices 434-538",
)

print()
print(
    "Max scaled truth difference:",
    validation[
        "max_scaled_truth_difference"
    ],
)

print(
    "Max original truth difference:",
    validation[
        "max_original_truth_difference"
    ],
)

print(
    "Max metric consistency difference:",
    validation[
        "max_metric_consistency_difference"
    ],
)

print(
    "Max spectral reconstruction error:",
    validation[
        "max_spectral_band_reconstruction_error"
    ],
)

print(
    "Max spectral summary difference:",
    validation[
        "max_spectral_result_summary_difference"
    ],
)

print()
print("-" * 120)
print("FIVE-SEED TEST SUMMARY")
print("-" * 120)

for fam in FAMILIES:

    print()
    print(
        fam["encoding_label"].upper()
    )

    d = depth_summary[
        depth_summary["encoding"]
        ==
        fam["encoding"]
    ]

    for _, row in d.iterrows():

        print(
            f"L={int(row['depth'])}  "
            f"RMSE="
            f"{row['test_rmse_original_mean']:.4f}"
            f"±"
            f"{row['test_rmse_original_std']:.4f}  "
            f"Pearson="
            f"{row['test_pearson_corr_mean']:.4f}"
            f"±"
            f"{row['test_pearson_corr_std']:.4f}  "
            f"KGE="
            f"{row['test_kge_mean']:.4f}"
            f"±"
            f"{row['test_kge_std']:.4f}  "
            f"FFT="
            f"{row['test_fft_spectral_cosine_similarity_mean']:.4f}"
            f"±"
            f"{row['test_fft_spectral_cosine_similarity_std']:.4f}  "
            f"Amp="
            f"{row['test_pred_std_over_true_std_mean']:.4f}"
            f"±"
            f"{row['test_pred_std_over_true_std_std']:.4f}"
        )

print()
print("=" * 120)
print("MASTER VALIDATION STATUS: PASS")
print("=" * 120)
