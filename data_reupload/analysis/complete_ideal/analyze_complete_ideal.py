#!/usr/bin/env python3
"""
Complete five-seed analysis of the ideal direct-window QNN experiments.

Included:
    single_tanh_a1:
        depths 1--6
        seeds 42--46

    residual:
        residual_noconcat_bounded_alpha1
        residual_noconcat_bounded_shrinkage
        residual_concat_bounded_alpha1
        residual_concat_bounded_shrinkage
        seeds 42--46

Excluded:
    single_tanh_a2, because only seed 42 exists.

Outputs:
    tables/*.csv
    figures/*.png
    figures/*.pdf
    analysis_manifest.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EXPECTED_SEEDS = [42, 43, 44, 45, 46]
EXPECTED_DEPTHS = [1, 2, 3, 4, 5, 6]

SINGLE_VARIANT = "single_tanh_a1"

RESIDUAL_VARIANTS = [
    "residual_noconcat_bounded_alpha1",
    "residual_noconcat_bounded_shrinkage",
    "residual_concat_bounded_alpha1",
    "residual_concat_bounded_shrinkage",
]

RESIDUAL_LABELS = {
    "residual_noconcat_bounded_alpha1": "NC α=1",
    "residual_noconcat_bounded_shrinkage": "NC shrinkage",
    "residual_concat_bounded_alpha1": "Concat α=1",
    "residual_concat_bounded_shrinkage": "Concat shrinkage",
}

MODEL_LABELS = {
    SINGLE_VARIANT: "Single QNN",
    **RESIDUAL_LABELS,
}

# Quantities for which lower values indicate better performance.
LOWER_IS_BETTER = {
    "train_rmse_original",
    "val_rmse_original",
    "test_rmse_original",
    "train_mae_original",
    "val_mae_original",
    "test_mae_original",
    "test_abs_12_step_band_error",
    "test_abs_short_period_2_to_6_error",
    "elapsed_time_sec",
    "max_rss_gb_python",
    "best_val_loss_scaled",
}

# Quantities for which higher values indicate better performance.
HIGHER_IS_BETTER = {
    "train_pearson_corr",
    "val_pearson_corr",
    "test_pearson_corr",
    "train_spearman_corr",
    "val_spearman_corr",
    "test_spearman_corr",
    "train_kge",
    "val_kge",
    "test_kge",
    "train_nse",
    "val_nse",
    "test_nse",
    "test_fft_spectral_cosine_similarity",
}

PRIMARY_METRICS = [
    "test_rmse_original",
    "test_mae_original",
    "test_pearson_corr",
    "test_spearman_corr",
    "test_kge",
    "test_nse",
    "test_fft_spectral_cosine_similarity",
    "test_abs_12_step_band_error",
    "test_abs_short_period_2_to_6_error",
]

METRIC_LABELS = {
    "train_rmse_original": "Train RMSE",
    "val_rmse_original": "Validation RMSE",
    "test_rmse_original": "Test RMSE",
    "train_mae_original": "Train MAE",
    "val_mae_original": "Validation MAE",
    "test_mae_original": "Test MAE",
    "train_pearson_corr": "Train Pearson correlation",
    "val_pearson_corr": "Validation Pearson correlation",
    "test_pearson_corr": "Test Pearson correlation",
    "train_spearman_corr": "Train Spearman correlation",
    "val_spearman_corr": "Validation Spearman correlation",
    "test_spearman_corr": "Test Spearman correlation",
    "train_kge": "Train KGE",
    "val_kge": "Validation KGE",
    "test_kge": "Test KGE",
    "train_nse": "Train NSE",
    "val_nse": "Validation NSE",
    "test_nse": "Test NSE",
    "test_fft_spectral_cosine_similarity": "FFT spectral cosine similarity",
    "test_abs_12_step_band_error": "Absolute 12-step band error",
    "test_abs_short_period_2_to_6_error": "Absolute 2–6-step band error",
    "elapsed_time_sec": "Runtime",
    "max_rss_gb_python": "Peak Python memory",
    "best_epoch": "Best epoch",
    "best_val_loss_scaled": "Best validation loss, scaled",
}

SUMMARY_METRICS = [
    "train_rmse_original",
    "val_rmse_original",
    "test_rmse_original",
    "train_mae_original",
    "val_mae_original",
    "test_mae_original",
    "train_pearson_corr",
    "val_pearson_corr",
    "test_pearson_corr",
    "train_spearman_corr",
    "val_spearman_corr",
    "test_spearman_corr",
    "train_kge",
    "val_kge",
    "test_kge",
    "train_nse",
    "val_nse",
    "test_nse",
    "train_bias",
    "val_bias",
    "test_bias",
    "train_pred_std_over_true_std",
    "val_pred_std_over_true_std",
    "test_pred_std_over_true_std",
    "train_negative_prediction_rate",
    "val_negative_prediction_rate",
    "test_negative_prediction_rate",
    "train_out_of_range_rate",
    "val_out_of_range_rate",
    "test_out_of_range_rate",
    "test_fft_spectral_cosine_similarity",
    "test_abs_12_step_band_error",
    "test_abs_short_period_2_to_6_error",
    "elapsed_time_sec",
    "max_rss_gb_python",
    "best_epoch",
    "best_val_loss_scaled",
]

# Student-t critical values for a two-sided 95% confidence interval.
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("data_reupload/results/final_model_sweep"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_reupload/analysis/complete_ideal/output"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def ensure_dirs(output_root: Path) -> tuple[Path, Path]:
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def save_figure(fig: plt.Figure, path_no_suffix: Path, dpi: int) -> None:
    fig.savefig(
        path_no_suffix.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        path_no_suffix.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def t_critical_95(df: int) -> float:
    if df <= 0:
        return np.nan
    if df in T_CRITICAL_95:
        return T_CRITICAL_95[df]
    return 1.96


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def normalized_fft_amplitude(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size < 2:
        return np.asarray([]), np.asarray([])

    centered = y - np.mean(y)
    fft_values = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(centered.size, d=1.0)

    if fft_values.size:
        fft_values[0] = 0.0

    total = fft_values.sum()
    if total > 0:
        fft_values = fft_values / total

    return frequencies, fft_values


def period_axis(frequencies: np.ndarray) -> np.ndarray:
    periods = np.full_like(frequencies, np.nan, dtype=float)
    mask = frequencies > 0
    periods[mask] = 1.0 / frequencies[mask]
    return periods


# ---------------------------------------------------------------------
# Read and validate result summaries
# ---------------------------------------------------------------------

def classify_summary(path: Path) -> Optional[dict]:
    parts = path.parts

    if SINGLE_VARIANT in parts:
        depth_match = None
        seed_match = None

        for part in parts:
            match = re.fullmatch(r"depth_(\d+)", part)
            if match:
                depth_match = int(match.group(1))

            match = re.fullmatch(r"seed_(\d+)", part)
            if match:
                seed_match = int(match.group(1))

        if depth_match in EXPECTED_DEPTHS and seed_match in EXPECTED_SEEDS:
            return {
                "analysis_group": "single",
                "variant_id_expected": SINGLE_VARIANT,
                "architecture_label": "Single QNN",
                "depth_expected": depth_match,
                "seed_expected": seed_match,
            }

    for variant in RESIDUAL_VARIANTS:
        if variant in parts:
            seed_match = None
            for part in parts:
                match = re.fullmatch(r"seed_(\d+)", part)
                if match:
                    seed_match = int(match.group(1))

            if seed_match in EXPECTED_SEEDS:
                return {
                    "analysis_group": "residual",
                    "variant_id_expected": variant,
                    "architecture_label": RESIDUAL_LABELS[variant],
                    "depth_expected": np.nan,
                    "seed_expected": seed_match,
                }

    return None


def load_complete_metrics(results_root: Path) -> pd.DataFrame:
    rows: list[dict] = []

    for summary_path in sorted(results_root.rglob("result_summary.csv")):
        classification = classify_summary(summary_path)
        if classification is None:
            continue

        frame = pd.read_csv(summary_path)

        if frame.empty:
            warnings.warn(f"Empty summary: {summary_path}")
            continue

        if len(frame) > 1:
            warnings.warn(
                f"{summary_path} has {len(frame)} rows; using the first row."
            )

        row = frame.iloc[0].to_dict()
        row.update(classification)
        row["summary_path"] = str(summary_path)
        row["run_dir"] = str(summary_path.parent)
        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No intended ideal summaries found under {results_root}"
        )

    metrics = pd.DataFrame(rows)

    for column in SUMMARY_METRICS + [
        "depth",
        "seed",
        "n_qubits",
        "n_ansatz_layers",
        "n_stages",
        "epochs_per_stage",
        "qnn_batch_size",
    ]:
        if column in metrics.columns:
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")

    metrics["depth"] = metrics.get(
        "depth", pd.Series(index=metrics.index, dtype=float)
    )
    metrics["seed"] = pd.to_numeric(metrics["seed_expected"], errors="coerce")
    metrics["depth"] = metrics["depth"].fillna(metrics["depth_expected"])

    metrics["runtime_minutes"] = (
        pd.to_numeric(metrics["elapsed_time_sec"], errors="coerce") / 60.0
    )

    metrics["model_key"] = np.where(
        metrics["analysis_group"].eq("single"),
        metrics["variant_id_expected"]
        + "_depth_"
        + metrics["depth"].astype("Int64").astype(str),
        metrics["variant_id_expected"],
    )

    metrics["model_label"] = np.where(
        metrics["analysis_group"].eq("single"),
        "Single depth "
        + metrics["depth"].astype("Int64").astype(str),
        metrics["architecture_label"],
    )

    metrics = metrics.sort_values(
        ["analysis_group", "variant_id_expected", "depth", "seed"],
        na_position="last",
    ).reset_index(drop=True)

    return metrics


def validate_expected_runs(metrics: pd.DataFrame) -> list[str]:
    messages: list[str] = []

    single = metrics[metrics["analysis_group"] == "single"]
    residual = metrics[metrics["analysis_group"] == "residual"]

    expected_single = {
        (depth, seed)
        for depth in EXPECTED_DEPTHS
        for seed in EXPECTED_SEEDS
    }
    actual_single = {
        (int(row.depth), int(row.seed))
        for row in single.itertuples()
        if pd.notna(row.depth) and pd.notna(row.seed)
    }

    missing_single = sorted(expected_single - actual_single)
    extra_single = sorted(actual_single - expected_single)

    if missing_single:
        messages.append(f"Missing single runs: {missing_single}")
    if extra_single:
        messages.append(f"Unexpected single runs: {extra_single}")

    expected_residual = {
        (variant, seed)
        for variant in RESIDUAL_VARIANTS
        for seed in EXPECTED_SEEDS
    }
    actual_residual = {
        (str(row.variant_id_expected), int(row.seed))
        for row in residual.itertuples()
        if pd.notna(row.seed)
    }

    missing_residual = sorted(expected_residual - actual_residual)
    extra_residual = sorted(actual_residual - expected_residual)

    if missing_residual:
        messages.append(f"Missing residual runs: {missing_residual}")
    if extra_residual:
        messages.append(f"Unexpected residual runs: {extra_residual}")

    messages.append(
        f"Included {len(single)} single runs and {len(residual)} residual runs."
    )
    messages.append(f"Total included runs: {len(metrics)}.")

    return messages


# ---------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------

def summarize_group(group: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    records: list[dict] = []

    available_metrics = [
        metric for metric in SUMMARY_METRICS if metric in group.columns
    ]

    grouped = group.groupby(group_columns, dropna=False, sort=False)

    for group_values, subset in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        base = dict(zip(group_columns, group_values))

        for metric in available_metrics:
            values = pd.to_numeric(subset[metric], errors="coerce")
            values = values[np.isfinite(values)]

            n = int(values.size)

            record = dict(base)
            record["metric"] = metric
            record["metric_label"] = METRIC_LABELS.get(metric, metric)
            record["n"] = n

            if n == 0:
                record.update(
                    mean=np.nan,
                    std=np.nan,
                    median=np.nan,
                    minimum=np.nan,
                    maximum=np.nan,
                    sem=np.nan,
                    ci95_half_width=np.nan,
                    ci95_lower=np.nan,
                    ci95_upper=np.nan,
                    coefficient_of_variation=np.nan,
                )
            else:
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if n > 1 else 0.0
                sem = std / math.sqrt(n) if n > 1 else 0.0
                ci_half = t_critical_95(n - 1) * sem if n > 1 else 0.0

                cv = (
                    std / abs(mean)
                    if n > 1 and np.isfinite(mean) and abs(mean) > 1e-12
                    else np.nan
                )

                record.update(
                    mean=mean,
                    std=std,
                    median=float(values.median()),
                    minimum=float(values.min()),
                    maximum=float(values.max()),
                    sem=sem,
                    ci95_half_width=ci_half,
                    ci95_lower=mean - ci_half,
                    ci95_upper=mean + ci_half,
                    coefficient_of_variation=cv,
                )

            records.append(record)

    return pd.DataFrame(records)


def make_mean_sd_table(
    summary_long: pd.DataFrame,
    index_column: str,
    metric_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []

    for index_value, subset in summary_long.groupby(index_column, sort=False):
        row = {index_column: index_value}

        for metric in metric_columns:
            metric_row = subset[subset["metric"] == metric]

            if metric_row.empty:
                row[metric] = ""
                continue

            mean = metric_row.iloc[0]["mean"]
            std = metric_row.iloc[0]["std"]

            if np.isfinite(mean) and np.isfinite(std):
                row[metric] = f"{mean:.4f} ± {std:.4f}"
            else:
                row[metric] = ""

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------

def load_predictions_for_row(row: pd.Series) -> Optional[dict]:
    run_dir = Path(row["run_dir"])
    prediction_path = run_dir / "predictions.npz"

    if not prediction_path.exists():
        candidates = sorted(run_dir.glob("*predictions*.npz"))
        if not candidates:
            warnings.warn(f"No prediction NPZ found in {run_dir}")
            return None
        prediction_path = candidates[0]

    try:
        with np.load(prediction_path, allow_pickle=False) as data:
            required = [
                "y_test_true_original",
                "y_test_pred_original",
            ]

            missing = [key for key in required if key not in data.files]
            if missing:
                warnings.warn(
                    f"{prediction_path} missing required keys: {missing}"
                )
                return None

            result = {
                "y_test_true_original": np.asarray(
                    data["y_test_true_original"], dtype=float
                ).reshape(-1),
                "y_test_pred_original": np.asarray(
                    data["y_test_pred_original"], dtype=float
                ).reshape(-1),
                "prediction_path": str(prediction_path),
            }

            for split in ["train", "val"]:
                true_key = f"y_{split}_true_original"
                pred_key = f"y_{split}_pred_original"

                if true_key in data.files:
                    result[true_key] = np.asarray(
                        data[true_key], dtype=float
                    ).reshape(-1)

                if pred_key in data.files:
                    result[pred_key] = np.asarray(
                        data[pred_key], dtype=float
                    ).reshape(-1)

            return result

    except Exception as exc:
        warnings.warn(f"Could not read {prediction_path}: {exc}")
        return None


def build_prediction_records(metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []

    for _, row in metrics.iterrows():
        loaded = load_predictions_for_row(row)
        if loaded is None:
            continue

        record = {
            "analysis_group": row["analysis_group"],
            "variant_id": row["variant_id_expected"],
            "architecture_label": row["architecture_label"],
            "model_label": row["model_label"],
            "model_key": row["model_key"],
            "depth": row["depth"],
            "seed": row["seed"],
            **loaded,
        }
        records.append(record)

    return pd.DataFrame(records)


def assert_common_truth(
    arrays: list[np.ndarray],
    label: str,
    atol: float = 1e-10,
) -> np.ndarray:
    if not arrays:
        raise ValueError(f"No arrays supplied for {label}")

    reference = arrays[0]

    for index, candidate in enumerate(arrays[1:], start=1):
        if reference.shape != candidate.shape:
            raise ValueError(
                f"Shape mismatch in {label}: "
                f"{reference.shape} versus {candidate.shape}"
            )

        if not np.allclose(reference, candidate, atol=atol, rtol=0):
            max_difference = float(np.max(np.abs(reference - candidate)))
            warnings.warn(
                f"Truth arrays differ in {label}; "
                f"maximum difference={max_difference:.6g}. "
                "Using the first array."
            )
            break

    return reference


# ---------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------

def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 13,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 2.2,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


# ---------------------------------------------------------------------
# Metric figures
# ---------------------------------------------------------------------

def plot_single_metric_panels(
    metrics: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    single = metrics[metrics["analysis_group"] == "single"].copy()

    figure_metrics = [
        "test_rmse_original",
        "test_mae_original",
        "test_pearson_corr",
        "test_spearman_corr",
        "test_kge",
        "test_nse",
        "test_fft_spectral_cosine_similarity",
        "test_abs_12_step_band_error",
        "test_abs_short_period_2_to_6_error",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)

    for ax, metric in zip(axes.flat, figure_metrics):
        for seed, seed_data in single.groupby("seed"):
            seed_data = seed_data.sort_values("depth")
            ax.plot(
                seed_data["depth"],
                seed_data[metric],
                marker="o",
                linewidth=0.9,
                alpha=0.30,
            )

        grouped = (
            single.groupby("depth")[metric]
            .agg(["mean", "std"])
            .reindex(EXPECTED_DEPTHS)
        )

        x = grouped.index.to_numpy(dtype=float)
        mean = grouped["mean"].to_numpy(dtype=float)
        std = grouped["std"].fillna(0).to_numpy(dtype=float)

        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2.8,
            label="Five-seed mean",
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.22,
            label="±1 SD",
        )

        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xlabel("Circuit depth")
        ax.set_xticks(EXPECTED_DEPTHS)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Ideal single-QNN depth sweep: test metrics")

    save_figure(
        fig,
        figures_dir / "01_single_metrics_vs_depth",
        dpi,
    )


def plot_residual_metric_panels(
    metrics: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    residual = metrics[metrics["analysis_group"] == "residual"].copy()

    figure_metrics = [
        "test_rmse_original",
        "test_mae_original",
        "test_pearson_corr",
        "test_spearman_corr",
        "test_kge",
        "test_nse",
        "test_fft_spectral_cosine_similarity",
        "test_abs_12_step_band_error",
        "test_abs_short_period_2_to_6_error",
    ]

    labels = [RESIDUAL_LABELS[item] for item in RESIDUAL_VARIANTS]
    x = np.arange(len(RESIDUAL_VARIANTS), dtype=float)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)

    for ax, metric in zip(axes.flat, figure_metrics):
        for seed, seed_data in residual.groupby("seed"):
            values = []
            for variant in RESIDUAL_VARIANTS:
                subset = seed_data[
                    seed_data["variant_id_expected"] == variant
                ]
                values.append(
                    float(subset.iloc[0][metric]) if not subset.empty else np.nan
                )

            ax.plot(
                x,
                values,
                marker="o",
                linewidth=0.9,
                alpha=0.30,
            )

        means = []
        stds = []

        for variant in RESIDUAL_VARIANTS:
            values = pd.to_numeric(
                residual.loc[
                    residual["variant_id_expected"] == variant,
                    metric,
                ],
                errors="coerce",
            )
            means.append(float(values.mean()))
            stds.append(float(values.std(ddof=1)))

        means_arr = np.asarray(means)
        stds_arr = np.asarray(stds)

        ax.plot(
            x,
            means_arr,
            marker="o",
            linewidth=2.8,
            label="Five-seed mean",
        )
        ax.fill_between(
            x,
            means_arr - stds_arr,
            means_arr + stds_arr,
            alpha=0.22,
            label="±1 SD",
        )

        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2)
    fig.suptitle("Ideal residual-QNN comparison: test metrics")

    save_figure(
        fig,
        figures_dir / "02_residual_metrics",
        dpi,
    )


def plot_train_validation_test_rmse(
    metrics: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    single = metrics[metrics["analysis_group"] == "single"]

    fig, ax = plt.subplots(figsize=(8, 5))

    split_metrics = [
        ("train_rmse_original", "Train"),
        ("val_rmse_original", "Validation"),
        ("test_rmse_original", "Test"),
    ]

    for metric, label in split_metrics:
        summary = (
            single.groupby("depth")[metric]
            .agg(["mean", "std"])
            .reindex(EXPECTED_DEPTHS)
        )

        x = summary.index.to_numpy(dtype=float)
        mean = summary["mean"].to_numpy(dtype=float)
        std = summary["std"].fillna(0).to_numpy(dtype=float)

        ax.plot(x, mean, marker="o", label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.16)

    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("RMSE in original units")
    ax.set_xticks(EXPECTED_DEPTHS)
    ax.set_title("Single QNN: train, validation, and test RMSE")
    ax.legend()

    save_figure(
        fig,
        figures_dir / "03_single_train_val_test_rmse",
        dpi,
    )


# ---------------------------------------------------------------------
# Prediction figures
# ---------------------------------------------------------------------

def plot_single_predictions(
    prediction_records: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    single = prediction_records[
        prediction_records["analysis_group"] == "single"
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)

    for ax, depth in zip(axes.flat, EXPECTED_DEPTHS):
        subset = single[single["depth"] == depth].sort_values("seed")

        if subset.empty:
            ax.set_visible(False)
            continue

        true_arrays = subset["y_test_true_original"].tolist()
        pred_arrays = subset["y_test_pred_original"].tolist()

        truth = assert_common_truth(
            true_arrays,
            f"single depth {depth} test truth",
        )
        predictions = np.stack(pred_arrays)

        mean_prediction = predictions.mean(axis=0)
        sd_prediction = predictions.std(axis=0, ddof=1)

        x = np.arange(truth.size)

        for prediction in predictions:
            ax.plot(x, prediction, linewidth=0.8, alpha=0.18)

        ax.plot(x, truth, color="black", linewidth=2.3, label="True")
        ax.plot(
            x,
            mean_prediction,
            linewidth=2.4,
            label="Mean prediction",
        )
        ax.fill_between(
            x,
            mean_prediction - sd_prediction,
            mean_prediction + sd_prediction,
            alpha=0.22,
            label="±1 SD",
        )

        ax.set_title(f"Depth {depth}")
        ax.set_xlabel("Test time step")
        ax.set_ylabel("Target, original units")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Ideal single-QNN test predictions across five seeds")

    save_figure(
        fig,
        figures_dir / "04_single_test_predictions",
        dpi,
    )


def plot_residual_predictions(
    prediction_records: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    residual = prediction_records[
        prediction_records["analysis_group"] == "residual"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    for ax, variant in zip(axes.flat, RESIDUAL_VARIANTS):
        subset = residual[
            residual["variant_id"] == variant
        ].sort_values("seed")

        if subset.empty:
            ax.set_visible(False)
            continue

        true_arrays = subset["y_test_true_original"].tolist()
        pred_arrays = subset["y_test_pred_original"].tolist()

        truth = assert_common_truth(
            true_arrays,
            f"{variant} test truth",
        )
        predictions = np.stack(pred_arrays)

        mean_prediction = predictions.mean(axis=0)
        sd_prediction = predictions.std(axis=0, ddof=1)

        x = np.arange(truth.size)

        for prediction in predictions:
            ax.plot(x, prediction, linewidth=0.8, alpha=0.18)

        ax.plot(x, truth, color="black", linewidth=2.3, label="True")
        ax.plot(
            x,
            mean_prediction,
            linewidth=2.4,
            label="Mean prediction",
        )
        ax.fill_between(
            x,
            mean_prediction - sd_prediction,
            mean_prediction + sd_prediction,
            alpha=0.22,
            label="±1 SD",
        )

        ax.set_title(RESIDUAL_LABELS[variant])
        ax.set_xlabel("Test time step")
        ax.set_ylabel("Target, original units")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Ideal residual-QNN test predictions across five seeds")

    save_figure(
        fig,
        figures_dir / "05_residual_test_predictions",
        dpi,
    )


# ---------------------------------------------------------------------
# Frequency-domain figures
# ---------------------------------------------------------------------

def plot_single_fft(
    prediction_records: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    single = prediction_records[
        prediction_records["analysis_group"] == "single"
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)

    for ax, depth in zip(axes.flat, EXPECTED_DEPTHS):
        subset = single[single["depth"] == depth].sort_values("seed")

        if subset.empty:
            ax.set_visible(False)
            continue

        truth = assert_common_truth(
            subset["y_test_true_original"].tolist(),
            f"single depth {depth} FFT truth",
        )

        frequencies, true_fft = normalized_fft_amplitude(truth)

        pred_ffts = []
        for prediction in subset["y_test_pred_original"]:
            pred_freq, pred_fft = normalized_fft_amplitude(prediction)

            if not np.allclose(pred_freq, frequencies):
                raise ValueError("FFT frequency grids do not match.")

            pred_ffts.append(pred_fft)

        pred_ffts_arr = np.stack(pred_ffts)
        mean_fft = pred_ffts_arr.mean(axis=0)
        sd_fft = pred_ffts_arr.std(axis=0, ddof=1)

        periods = period_axis(frequencies)
        mask = np.isfinite(periods) & (periods >= 2)

        ax.plot(
            periods[mask],
            true_fft[mask],
            color="black",
            linewidth=2.3,
            label="True",
        )
        ax.plot(
            periods[mask],
            mean_fft[mask],
            linewidth=2.4,
            label="Mean prediction",
        )
        ax.fill_between(
            periods[mask],
            np.maximum(mean_fft[mask] - sd_fft[mask], 0),
            mean_fft[mask] + sd_fft[mask],
            alpha=0.22,
            label="±1 SD",
        )

        ax.axvline(12, linestyle="--", linewidth=1.1, alpha=0.7)
        ax.set_xlim(2, min(40, np.nanmax(periods[mask])))
        ax.set_title(f"Depth {depth}")
        ax.set_xlabel("Period, time steps")
        ax.set_ylabel("Normalized FFT amplitude")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Ideal single-QNN frequency spectra across five seeds")

    save_figure(
        fig,
        figures_dir / "06_single_prediction_fft",
        dpi,
    )


def plot_residual_fft(
    prediction_records: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    residual = prediction_records[
        prediction_records["analysis_group"] == "residual"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    for ax, variant in zip(axes.flat, RESIDUAL_VARIANTS):
        subset = residual[
            residual["variant_id"] == variant
        ].sort_values("seed")

        if subset.empty:
            ax.set_visible(False)
            continue

        truth = assert_common_truth(
            subset["y_test_true_original"].tolist(),
            f"{variant} FFT truth",
        )

        frequencies, true_fft = normalized_fft_amplitude(truth)

        pred_ffts = []
        for prediction in subset["y_test_pred_original"]:
            pred_freq, pred_fft = normalized_fft_amplitude(prediction)

            if not np.allclose(pred_freq, frequencies):
                raise ValueError("FFT frequency grids do not match.")

            pred_ffts.append(pred_fft)

        pred_ffts_arr = np.stack(pred_ffts)
        mean_fft = pred_ffts_arr.mean(axis=0)
        sd_fft = pred_ffts_arr.std(axis=0, ddof=1)

        periods = period_axis(frequencies)
        mask = np.isfinite(periods) & (periods >= 2)

        ax.plot(
            periods[mask],
            true_fft[mask],
            color="black",
            linewidth=2.3,
            label="True",
        )
        ax.plot(
            periods[mask],
            mean_fft[mask],
            linewidth=2.4,
            label="Mean prediction",
        )
        ax.fill_between(
            periods[mask],
            np.maximum(mean_fft[mask] - sd_fft[mask], 0),
            mean_fft[mask] + sd_fft[mask],
            alpha=0.22,
            label="±1 SD",
        )

        ax.axvline(12, linestyle="--", linewidth=1.1, alpha=0.7)
        ax.set_xlim(2, min(40, np.nanmax(periods[mask])))
        ax.set_title(RESIDUAL_LABELS[variant])
        ax.set_xlabel("Period, time steps")
        ax.set_ylabel("Normalized FFT amplitude")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Ideal residual-QNN frequency spectra across five seeds")

    save_figure(
        fig,
        figures_dir / "07_residual_prediction_fft",
        dpi,
    )


# ---------------------------------------------------------------------
# Runtime, memory, and robustness
# ---------------------------------------------------------------------

def plot_runtime_memory(
    metrics: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    single = metrics[metrics["analysis_group"] == "single"]

    fig, ax = plt.subplots(figsize=(8, 5))

    summary = (
        single.groupby("depth")["runtime_minutes"]
        .agg(["mean", "std"])
        .reindex(EXPECTED_DEPTHS)
    )

    x = summary.index.to_numpy(dtype=float)
    mean = summary["mean"].to_numpy(dtype=float)
    std = summary["std"].fillna(0).to_numpy(dtype=float)

    for _, seed_data in single.groupby("seed"):
        seed_data = seed_data.sort_values("depth")
        ax.plot(
            seed_data["depth"],
            seed_data["runtime_minutes"],
            marker="o",
            linewidth=0.8,
            alpha=0.25,
        )

    ax.plot(x, mean, marker="o", linewidth=2.8, label="Five-seed mean")
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.22,
        label="±1 SD",
    )

    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Runtime, minutes")
    ax.set_xticks(EXPECTED_DEPTHS)
    ax.set_title("Ideal single-QNN runtime scaling")
    ax.legend()

    save_figure(
        fig,
        figures_dir / "08_single_runtime_vs_depth",
        dpi,
    )

    if "max_rss_gb_python" in single.columns:
        fig, ax = plt.subplots(figsize=(8, 5))

        summary = (
            single.groupby("depth")["max_rss_gb_python"]
            .agg(["mean", "std"])
            .reindex(EXPECTED_DEPTHS)
        )

        x = summary.index.to_numpy(dtype=float)
        mean = summary["mean"].to_numpy(dtype=float)
        std = summary["std"].fillna(0).to_numpy(dtype=float)

        ax.plot(x, mean, marker="o", linewidth=2.8)
        ax.fill_between(x, mean - std, mean + std, alpha=0.22)

        ax.set_xlabel("Circuit depth")
        ax.set_ylabel("Peak Python memory, GB")
        ax.set_xticks(EXPECTED_DEPTHS)
        ax.set_title("Ideal single-QNN memory scaling")

        save_figure(
            fig,
            figures_dir / "09_single_memory_vs_depth",
            dpi,
        )


def plot_seed_robustness(
    single_summary: pd.DataFrame,
    residual_summary: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    robustness_metrics = [
        "test_rmse_original",
        "test_pearson_corr",
        "test_fft_spectral_cosine_similarity",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.5),
        constrained_layout=True,
    )

    for ax, metric in zip(axes, robustness_metrics):
        subset = single_summary[single_summary["metric"] == metric].copy()
        subset = subset.sort_values("depth")

        ax.plot(
            subset["depth"],
            subset["coefficient_of_variation"],
            marker="o",
            linewidth=2.5,
        )
        ax.set_xlabel("Circuit depth")
        ax.set_xticks(EXPECTED_DEPTHS)
        ax.set_ylabel("Coefficient of variation")
        ax.set_title(METRIC_LABELS.get(metric, metric))

    fig.suptitle("Single-QNN seed-to-seed variability")

    save_figure(
        fig,
        figures_dir / "10_single_seed_robustness",
        dpi,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.8),
        constrained_layout=True,
    )

    x = np.arange(len(RESIDUAL_VARIANTS), dtype=float)
    labels = [RESIDUAL_LABELS[item] for item in RESIDUAL_VARIANTS]

    for ax, metric in zip(axes, robustness_metrics):
        values = []

        for variant in RESIDUAL_VARIANTS:
            row = residual_summary[
                (residual_summary["variant_id_expected"] == variant)
                & (residual_summary["metric"] == metric)
            ]

            values.append(
                float(row.iloc[0]["coefficient_of_variation"])
                if not row.empty
                else np.nan
            )

        ax.plot(x, values, marker="o", linewidth=2.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Coefficient of variation")
        ax.set_title(METRIC_LABELS.get(metric, metric))

    fig.suptitle("Residual-QNN seed-to-seed variability")

    save_figure(
        fig,
        figures_dir / "11_residual_seed_robustness",
        dpi,
    )


# ---------------------------------------------------------------------
# Ranking and heatmap
# ---------------------------------------------------------------------

def build_model_ranking(
    single_summary: pd.DataFrame,
    residual_summary: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict] = []

    for depth in EXPECTED_DEPTHS:
        subset = single_summary[single_summary["depth"] == depth]

        record = {
            "model_key": f"single_depth_{depth}",
            "model_label": f"Single depth {depth}",
            "model_family": "single",
            "depth": depth,
        }

        for metric in PRIMARY_METRICS + [
            "val_rmse_original",
            "runtime_minutes",
        ]:
            metric_name = (
                "elapsed_time_sec"
                if metric == "runtime_minutes"
                else metric
            )

            row = subset[subset["metric"] == metric_name]

            if not row.empty:
                value = float(row.iloc[0]["mean"])

                if metric == "runtime_minutes":
                    value /= 60.0

                record[f"{metric}_mean"] = value
                record[f"{metric}_std"] = (
                    float(row.iloc[0]["std"]) / 60.0
                    if metric == "runtime_minutes"
                    else float(row.iloc[0]["std"])
                )

        records.append(record)

    for variant in RESIDUAL_VARIANTS:
        subset = residual_summary[
            residual_summary["variant_id_expected"] == variant
        ]

        record = {
            "model_key": variant,
            "model_label": RESIDUAL_LABELS[variant],
            "model_family": "residual",
            "depth": np.nan,
        }

        for metric in PRIMARY_METRICS + [
            "val_rmse_original",
            "runtime_minutes",
        ]:
            metric_name = (
                "elapsed_time_sec"
                if metric == "runtime_minutes"
                else metric
            )

            row = subset[subset["metric"] == metric_name]

            if not row.empty:
                value = float(row.iloc[0]["mean"])

                if metric == "runtime_minutes":
                    value /= 60.0

                record[f"{metric}_mean"] = value
                record[f"{metric}_std"] = (
                    float(row.iloc[0]["std"]) / 60.0
                    if metric == "runtime_minutes"
                    else float(row.iloc[0]["std"])
                )

        records.append(record)

    ranking = pd.DataFrame(records)

    if "test_rmse_original_mean" in ranking.columns:
        ranking["rank_test_rmse"] = ranking[
            "test_rmse_original_mean"
        ].rank(method="min", ascending=True)

    if "val_rmse_original_mean" in ranking.columns:
        ranking["rank_val_rmse"] = ranking[
            "val_rmse_original_mean"
        ].rank(method="min", ascending=True)

    if "test_fft_spectral_cosine_similarity_mean" in ranking.columns:
        ranking["rank_fft_cosine"] = ranking[
            "test_fft_spectral_cosine_similarity_mean"
        ].rank(method="min", ascending=False)

    rank_columns = [
        column
        for column in [
            "rank_test_rmse",
            "rank_val_rmse",
            "rank_fft_cosine",
        ]
        if column in ranking.columns
    ]

    ranking["mean_rank"] = ranking[rank_columns].mean(axis=1)
    ranking = ranking.sort_values(
        ["mean_rank", "rank_test_rmse"],
        na_position="last",
    ).reset_index(drop=True)

    return ranking


def plot_metric_heatmap(
    ranking: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    heatmap_metrics = [
        "test_rmse_original_mean",
        "test_mae_original_mean",
        "test_pearson_corr_mean",
        "test_kge_mean",
        "test_nse_mean",
        "test_fft_spectral_cosine_similarity_mean",
        "test_abs_12_step_band_error_mean",
        "test_abs_short_period_2_to_6_error_mean",
    ]

    available = [
        metric for metric in heatmap_metrics if metric in ranking.columns
    ]

    matrix = ranking[available].copy()

    normalized = pd.DataFrame(index=matrix.index)

    for column in available:
        values = pd.to_numeric(matrix[column], errors="coerce")
        minimum = values.min()
        maximum = values.max()

        if not np.isfinite(minimum) or not np.isfinite(maximum):
            normalized[column] = np.nan
            continue

        if abs(maximum - minimum) < 1e-12:
            scores = pd.Series(1.0, index=values.index)
        else:
            scores = (values - minimum) / (maximum - minimum)

        raw_metric = column.removesuffix("_mean")

        if raw_metric in LOWER_IS_BETTER:
            scores = 1.0 - scores

        normalized[column] = scores

    fig, ax = plt.subplots(
        figsize=(12, max(5, 0.55 * len(ranking) + 2))
    )

    image = ax.imshow(
        normalized.to_numpy(dtype=float),
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_yticks(np.arange(len(ranking)))
    ax.set_yticklabels(ranking["model_label"])

    display_labels = [
        METRIC_LABELS.get(column.removesuffix("_mean"), column)
        for column in available
    ]

    ax.set_xticks(np.arange(len(available)))
    ax.set_xticklabels(display_labels, rotation=35, ha="right")

    for row_index in range(normalized.shape[0]):
        for column_index in range(normalized.shape[1]):
            value = normalized.iloc[row_index, column_index]

            if np.isfinite(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Normalized performance score: higher is better")

    ax.set_title("Complete ideal-model performance heatmap")
    fig.tight_layout()

    save_figure(
        fig,
        figures_dir / "12_complete_metric_heatmap",
        dpi,
    )


# ---------------------------------------------------------------------
# Training-history discovery
# ---------------------------------------------------------------------

TRAIN_COLUMN_CANDIDATES = [
    "train_loss_scaled",
    "train_loss",
    "training_loss",
    "loss_train",
    "epoch_train_loss",
    "train_mse",
]

VAL_COLUMN_CANDIDATES = [
    "val_loss_scaled",
    "val_loss",
    "validation_loss",
    "valid_loss",
    "loss_val",
    "epoch_val_loss",
    "val_mse",
]

EPOCH_COLUMN_CANDIDATES = [
    "epoch",
    "epochs",
    "step",
    "iteration",
    "iter",
]


def first_existing_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> Optional[str]:
    lower_to_original = {str(column).lower(): column for column in columns}

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    return None


def load_history_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None

    train_col = first_existing_column(frame.columns, TRAIN_COLUMN_CANDIDATES)
    val_col = first_existing_column(frame.columns, VAL_COLUMN_CANDIDATES)
    epoch_col = first_existing_column(frame.columns, EPOCH_COLUMN_CANDIDATES)

    if train_col is None and val_col is None:
        return None

    output = pd.DataFrame()

    if epoch_col is not None:
        output["epoch"] = pd.to_numeric(
            frame[epoch_col], errors="coerce"
        )
    else:
        output["epoch"] = np.arange(1, len(frame) + 1)

    if train_col is not None:
        output["train_loss"] = pd.to_numeric(
            frame[train_col], errors="coerce"
        )

    if val_col is not None:
        output["val_loss"] = pd.to_numeric(
            frame[val_col], errors="coerce"
        )

    output["source_file"] = str(path)
    return output


def load_history_npz(path: Path) -> Optional[pd.DataFrame]:
    try:
        with np.load(path, allow_pickle=True) as data:
            train_key = first_existing_column(
                data.files,
                TRAIN_COLUMN_CANDIDATES,
            )
            val_key = first_existing_column(
                data.files,
                VAL_COLUMN_CANDIDATES,
            )
            epoch_key = first_existing_column(
                data.files,
                EPOCH_COLUMN_CANDIDATES,
            )

            if train_key is None and val_key is None:
                return None

            lengths = []

            if train_key is not None:
                train_values = np.asarray(data[train_key]).reshape(-1)
                lengths.append(train_values.size)
            else:
                train_values = None

            if val_key is not None:
                val_values = np.asarray(data[val_key]).reshape(-1)
                lengths.append(val_values.size)
            else:
                val_values = None

            length = min(lengths)

            if epoch_key is not None:
                epoch_values = np.asarray(data[epoch_key]).reshape(-1)[:length]
            else:
                epoch_values = np.arange(1, length + 1)

            output = pd.DataFrame({"epoch": epoch_values})

            if train_values is not None:
                output["train_loss"] = train_values[:length]

            if val_values is not None:
                output["val_loss"] = val_values[:length]

            output["source_file"] = str(path)
            return output

    except Exception:
        return None


def discover_run_history(run_dir: Path) -> Optional[pd.DataFrame]:
    candidate_paths: list[Path] = []

    csv_patterns = [
        "*history*.csv",
        "*loss*.csv",
        "*training*.csv",
        "*metrics*.csv",
    ]

    npz_patterns = [
        "*history*.npz",
        "*loss*.npz",
        "*training*.npz",
    ]

    for pattern in csv_patterns + npz_patterns:
        candidate_paths.extend(run_dir.rglob(pattern))

    ignored_names = {
        "result_summary.csv",
        "predictions.npz",
        "learned_parameters.npz",
    }

    candidate_paths = sorted(
        {
            path
            for path in candidate_paths
            if path.name not in ignored_names
        }
    )

    loaded_frames: list[pd.DataFrame] = []

    for path in candidate_paths:
        if path.suffix.lower() == ".csv":
            loaded = load_history_csv(path)
        elif path.suffix.lower() == ".npz":
            loaded = load_history_npz(path)
        else:
            loaded = None

        if loaded is not None and not loaded.empty:
            loaded["history_segment"] = len(loaded_frames) + 1
            loaded_frames.append(loaded)

    if not loaded_frames:
        return None

    # One file generally means a normal epoch history.
    if len(loaded_frames) == 1:
        frame = loaded_frames[0].copy()
        frame["epoch"] = np.arange(1, len(frame) + 1)
        return frame

    # For stagewise residual training, concatenate discovered segments.
    combined_frames = []
    epoch_offset = 0

    for frame in loaded_frames:
        segment = frame.copy()
        segment["epoch"] = np.arange(
            epoch_offset + 1,
            epoch_offset + len(segment) + 1,
        )
        epoch_offset += len(segment)
        combined_frames.append(segment)

    return pd.concat(combined_frames, ignore_index=True)


def collect_histories(metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []

    for _, row in metrics.iterrows():
        history = discover_run_history(Path(row["run_dir"]))

        if history is None:
            continue

        history["analysis_group"] = row["analysis_group"]
        history["variant_id"] = row["variant_id_expected"]
        history["model_key"] = row["model_key"]
        history["model_label"] = row["model_label"]
        history["depth"] = row["depth"]
        history["seed"] = row["seed"]
        records.append(history)

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


def aggregate_history(
    history: pd.DataFrame,
    grouping_columns: list[str],
) -> pd.DataFrame:
    records: list[dict] = []

    for group_values, subset in history.groupby(
        grouping_columns,
        dropna=False,
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_info = dict(zip(grouping_columns, group_values))

        for loss_column in ["train_loss", "val_loss"]:
            if loss_column not in subset.columns:
                continue

            for epoch, epoch_subset in subset.groupby("epoch"):
                values = pd.to_numeric(
                    epoch_subset[loss_column],
                    errors="coerce",
                )
                values = values[np.isfinite(values)]

                if values.empty:
                    continue

                record = dict(group_info)
                record.update(
                    epoch=int(epoch),
                    loss_type=loss_column,
                    mean=float(values.mean()),
                    std=float(values.std(ddof=1))
                    if len(values) > 1
                    else 0.0,
                    n=int(len(values)),
                )
                records.append(record)

    return pd.DataFrame(records)


def plot_training_histories(
    history: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
    dpi: int,
) -> None:
    if history.empty:
        warning_path = tables_dir / "training_history_not_found.txt"
        warning_path.write_text(
            "No recognizable training-history files were found.\n"
            "The metric and prediction analyses were still completed.\n"
        )
        return

    history.to_csv(
        tables_dir / "complete_training_history.csv",
        index=False,
    )

    single = history[history["analysis_group"] == "single"]

    if not single.empty:
        aggregate = aggregate_history(
            single,
            ["depth"],
        )
        aggregate.to_csv(
            tables_dir / "single_training_history_summary.csv",
            index=False,
        )

        fig, axes = plt.subplots(
            3,
            2,
            figsize=(14, 11),
            constrained_layout=True,
        )

        for ax, depth in zip(axes.flat, EXPECTED_DEPTHS):
            subset = aggregate[aggregate["depth"] == depth]

            for loss_type, label in [
                ("train_loss", "Training"),
                ("val_loss", "Validation"),
            ]:
                curve = subset[
                    subset["loss_type"] == loss_type
                ].sort_values("epoch")

                if curve.empty:
                    continue

                x = curve["epoch"].to_numpy(dtype=float)
                mean = curve["mean"].to_numpy(dtype=float)
                std = curve["std"].to_numpy(dtype=float)

                ax.plot(x, mean, label=label)
                ax.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    alpha=0.18,
                )

            ax.set_title(f"Depth {depth}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")

        handles, labels = axes.flat[0].get_legend_handles_labels()

        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2)

        fig.suptitle("Ideal single-QNN training dynamics")

        save_figure(
            fig,
            figures_dir / "13_single_training_curves",
            dpi,
        )

    residual = history[history["analysis_group"] == "residual"]

    if not residual.empty:
        aggregate = aggregate_history(
            residual,
            ["variant_id"],
        )
        aggregate.to_csv(
            tables_dir / "residual_training_history_summary.csv",
            index=False,
        )

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 8),
            constrained_layout=True,
        )

        for ax, variant in zip(axes.flat, RESIDUAL_VARIANTS):
            subset = aggregate[
                aggregate["variant_id"] == variant
            ]

            for loss_type, label in [
                ("train_loss", "Training"),
                ("val_loss", "Validation"),
            ]:
                curve = subset[
                    subset["loss_type"] == loss_type
                ].sort_values("epoch")

                if curve.empty:
                    continue

                x = curve["epoch"].to_numpy(dtype=float)
                mean = curve["mean"].to_numpy(dtype=float)
                std = curve["std"].to_numpy(dtype=float)

                ax.plot(x, mean, label=label)
                ax.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    alpha=0.18,
                )

            ax.set_title(RESIDUAL_LABELS[variant])
            ax.set_xlabel("Combined stage epoch")
            ax.set_ylabel("Loss")

        handles, labels = axes.flat[0].get_legend_handles_labels()

        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2)

        fig.suptitle("Ideal residual-QNN training dynamics")

        save_figure(
            fig,
            figures_dir / "14_residual_training_curves",
            dpi,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    results_root = args.results_root.resolve()
    output_root = args.output_root.resolve()

    figures_dir, tables_dir = ensure_dirs(output_root)
    configure_plotting()

    print(f"Results root: {results_root}")
    print(f"Output root:  {output_root}")

    metrics = load_complete_metrics(results_root)

    validation_messages = validate_expected_runs(metrics)

    for message in validation_messages:
        print(message)

    metrics.to_csv(
        tables_dir / "complete_ideal_run_metrics.csv",
        index=False,
    )

    single = metrics[metrics["analysis_group"] == "single"].copy()
    residual = metrics[metrics["analysis_group"] == "residual"].copy()

    single_summary = summarize_group(single, ["depth"])
    residual_summary = summarize_group(
        residual,
        ["variant_id_expected", "architecture_label"],
    )

    single_summary.to_csv(
        tables_dir / "single_depth_statistical_summary.csv",
        index=False,
    )

    residual_summary.to_csv(
        tables_dir / "residual_statistical_summary.csv",
        index=False,
    )

    single_mean_sd = make_mean_sd_table(
        single_summary,
        "depth",
        PRIMARY_METRICS
        + [
            "val_rmse_original",
            "elapsed_time_sec",
            "max_rss_gb_python",
        ],
    )

    residual_mean_sd = make_mean_sd_table(
        residual_summary,
        "architecture_label",
        PRIMARY_METRICS
        + [
            "val_rmse_original",
            "elapsed_time_sec",
            "max_rss_gb_python",
        ],
    )

    single_mean_sd.to_csv(
        tables_dir / "single_depth_mean_sd_table.csv",
        index=False,
    )

    residual_mean_sd.to_csv(
        tables_dir / "residual_mean_sd_table.csv",
        index=False,
    )

    ranking = build_model_ranking(
        single_summary,
        residual_summary,
    )

    ranking.to_csv(
        tables_dir / "complete_ideal_model_ranking.csv",
        index=False,
    )

    prediction_records = build_prediction_records(metrics)

    prediction_index = prediction_records.drop(
        columns=[
            column
            for column in [
                "y_test_true_original",
                "y_test_pred_original",
                "y_train_true_original",
                "y_train_pred_original",
                "y_val_true_original",
                "y_val_pred_original",
            ]
            if column in prediction_records.columns
        ]
    )

    prediction_index.to_csv(
        tables_dir / "prediction_file_index.csv",
        index=False,
    )

    plot_single_metric_panels(metrics, figures_dir, args.dpi)
    plot_residual_metric_panels(metrics, figures_dir, args.dpi)
    plot_train_validation_test_rmse(metrics, figures_dir, args.dpi)

    plot_single_predictions(
        prediction_records,
        figures_dir,
        args.dpi,
    )

    plot_residual_predictions(
        prediction_records,
        figures_dir,
        args.dpi,
    )

    plot_single_fft(
        prediction_records,
        figures_dir,
        args.dpi,
    )

    plot_residual_fft(
        prediction_records,
        figures_dir,
        args.dpi,
    )

    plot_runtime_memory(metrics, figures_dir, args.dpi)

    plot_seed_robustness(
        single_summary,
        residual_summary,
        figures_dir,
        args.dpi,
    )

    plot_metric_heatmap(
        ranking,
        figures_dir,
        args.dpi,
    )

    history = collect_histories(metrics)

    plot_training_histories(
        history,
        figures_dir,
        tables_dir,
        args.dpi,
    )

    manifest = {
        "results_root": str(results_root),
        "output_root": str(output_root),
        "included_runs": int(len(metrics)),
        "single_runs": int(len(single)),
        "residual_runs": int(len(residual)),
        "prediction_records": int(len(prediction_records)),
        "history_rows": int(len(history)),
        "expected_seeds": EXPECTED_SEEDS,
        "expected_depths": EXPECTED_DEPTHS,
        "residual_variants": RESIDUAL_VARIANTS,
        "validation_messages": validation_messages,
    }

    with open(output_root / "analysis_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    with open(output_root / "analysis_manifest.txt", "w") as handle:
        for key, value in manifest.items():
            handle.write(f"{key}: {value}\n")

    print()
    print("Analysis completed successfully.")
    print(f"Tables:  {tables_dir}")
    print(f"Figures: {figures_dir}")

    print()
    print("Top-ranked models:")
    print(
        ranking[
            [
                "model_label",
                "test_rmse_original_mean",
                "val_rmse_original_mean",
                "test_fft_spectral_cosine_similarity_mean",
                "mean_rank",
            ]
        ].head(10).to_string(index=False)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
