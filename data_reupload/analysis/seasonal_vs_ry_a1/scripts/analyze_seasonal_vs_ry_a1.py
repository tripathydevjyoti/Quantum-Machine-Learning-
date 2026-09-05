#!/usr/bin/env python

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/umbc/rs/pi_deffner/users/devjyot1/projects/"
    "Quantum-Machine-Learning-"
)

RY_ROOT = (
    PROJECT_ROOT
    / "data_reupload/results/final_model_sweep/single_tanh_a1"
)

SEASONAL_ROOT = (
    PROJECT_ROOT
    / "data_reupload/results/encoding_sweep/"
      "seasonal_meridian_tanh_a1"
)

OUT = (
    PROJECT_ROOT
    / "data_reupload/analysis/seasonal_vs_ry_a1/output"
)

FIG = OUT / "figures"
TAB = OUT / "tables"

FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

DEPTHS = [1, 2, 3, 4, 5, 6]
SEEDS = [42, 43, 44, 45, 46]

ENCODINGS = {
    "ry": {
        "label": r"Original $R_y(\pi x)$",
        "root": RY_ROOT,
    },
    "seasonal": {
        "label": "Seasonal meridian",
        "root": SEASONAL_ROOT,
    },
}

COLORS = {
    "ry": "tab:blue",
    "seasonal": "tab:green",
}


# ============================================================
# METRIC DEFINITIONS
# ============================================================

MAIN_METRICS = {
    "test_rmse_original": (
        "Test RMSE",
        "lower",
    ),
    "test_mae_original": (
        "Test MAE",
        "lower",
    ),
    "test_pearson_corr": (
        "Pearson correlation",
        "higher",
    ),
    "test_spearman_corr": (
        "Spearman correlation",
        "higher",
    ),
    "test_kge": (
        "KGE",
        "higher",
    ),
    "test_nse": (
        "NSE",
        "higher",
    ),
}

REPORT_METRICS = {
    **MAIN_METRICS,

    "test_pred_std_over_true_std": (
        r"$\sigma_{\hat y}/\sigma_y$",
        "target1",
    ),

    "test_bias": (
        "Bias",
        "target0",
    ),

    "test_fft_spectral_cosine_similarity": (
        "FFT cosine similarity",
        "higher",
    ),

    "test_abs_12_step_band_error": (
        "Absolute 12-step band error",
        "lower",
    ),

    "test_abs_short_period_2_to_6_error": (
        "Absolute 2--6-step band error",
        "lower",
    ),

    "elapsed_time_sec": (
        "Runtime [s]",
        "lower",
    ),

    "max_rss_gb_python": (
        "Maximum resident memory [GB]",
        "lower",
    ),
}


# ============================================================
# HELPERS
# ============================================================

def save_figure(fig, basename: str):
    for ext in ["pdf", "png"]:
        fig.savefig(
            FIG / f"{basename}.{ext}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def add_mean_sd_curve(
    ax,
    df,
    ycol,
    *,
    reference=None,
    xlabel="Reuploading depth",
    ylabel=None,
    title=None,
):
    for enc in ["ry", "seasonal"]:

        g = (
            df[df["encoding"] == enc]
            .groupby("depth")[ycol]
            .agg(["mean", "std"])
            .reindex(DEPTHS)
        )

        x = np.asarray(DEPTHS)
        mean = g["mean"].to_numpy(dtype=float)
        sd = g["std"].to_numpy(dtype=float)

        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=1.8,
            color=COLORS[enc],
            label=ENCODINGS[enc]["label"],
        )

        ax.fill_between(
            x,
            mean - sd,
            mean + sd,
            alpha=0.18,
            color=COLORS[enc],
        )

    if reference is not None:
        ax.axhline(
            reference,
            linestyle="--",
            linewidth=1,
            color="black",
            alpha=0.6,
        )

    ax.set_xticks(DEPTHS)
    ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.grid(alpha=0.20)


def safe_cv(values):
    values = np.asarray(values, dtype=float)

    mean = np.mean(values)
    sd = np.std(values, ddof=1)

    if abs(mean) < 1e-15:
        return np.nan

    return sd / abs(mean)


# ============================================================
# LOAD ALL 60 RUNS
# ============================================================

run_rows = []
history_rows = []
split_metric_rows = []
prediction_rows = []
fft_summary_rows = []
fft_power_rows = []
peak_rows = []

configs = {}

reference_truth = None
reference_indices = None

validation = {
    "n_expected_runs": 60,
    "n_loaded_runs": 0,

    "truth_arrays_identical": True,
    "target_indices_identical": True,

    "max_truth_difference": 0.0,

    "max_abs12_reconstruction_error": 0.0,
    "max_absshort_reconstruction_error": 0.0,

    "seasonal_encoding_audits_pass": True,

    "controlled_training_fields_match": True,

    "controlled_training_field_mismatches": [],
}


CONTROL_FIELDS = [
    "train_end_idx",
    "window_size",
    "train_window_end",

    "scaler_data_min",
    "scaler_data_max",
    "scaler_feature_min",
    "scaler_feature_max",

    "n_ansatz_layers",
    "readout",

    "qnn_epochs",
    "qnn_learning_rate",
    "qnn_batch_size",

    "qml_device",
    "diff_method",
    "device",

    "use_output_tanh",
]


for encoding, info in ENCODINGS.items():

    for depth in DEPTHS:

        for seed in SEEDS:

            run_dir = (
                info["root"]
                / f"depth_{depth}"
                / f"seed_{seed}"
            )

            required = [
                "result_summary.csv",
                "history.csv",
                "metrics_by_split.csv",
                "predictions_long.csv",
                "predictions.npz",
                "fft_summary_by_split.csv",
                "fft_power_by_period.csv",
                "peak_low_error_summary.csv",
                "config.json",
            ]

            if encoding == "seasonal":
                required.append(
                    "encoding_metadata.json"
                )

            for filename in required:
                path = run_dir / filename

                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing required artifact: {path}"
                    )

            # ------------------------------------------------
            # Result summary
            # ------------------------------------------------

            result = pd.read_csv(
                run_dir / "result_summary.csv"
            )

            if len(result) != 1:
                raise RuntimeError(
                    f"Expected one result row in {run_dir}"
                )

            row = result.iloc[0].to_dict()

            row["encoding"] = encoding
            row["encoding_label"] = info["label"]
            row["run_dir"] = str(run_dir)

            run_rows.append(row)

            # ------------------------------------------------
            # History
            # ------------------------------------------------

            h = pd.read_csv(
                run_dir / "history.csv"
            )

            h["encoding"] = encoding
            h["depth"] = depth
            h["seed"] = seed

            history_rows.append(h)

            # ------------------------------------------------
            # Split metrics
            # ------------------------------------------------

            m = pd.read_csv(
                run_dir / "metrics_by_split.csv"
            )

            m["encoding"] = encoding
            m["depth"] = depth
            m["seed"] = seed

            split_metric_rows.append(m)

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            p = pd.read_csv(
                run_dir / "predictions_long.csv"
            )

            p["encoding"] = encoding
            p["depth"] = depth
            p["seed"] = seed

            prediction_rows.append(p)

            # ------------------------------------------------
            # FFT summary
            # ------------------------------------------------

            fs = pd.read_csv(
                run_dir / "fft_summary_by_split.csv"
            )

            fs["encoding"] = encoding
            fs["depth"] = depth
            fs["seed"] = seed

            fft_summary_rows.append(fs)

            # ------------------------------------------------
            # Full FFT
            # ------------------------------------------------

            fp = pd.read_csv(
                run_dir / "fft_power_by_period.csv"
            )

            fp["encoding"] = encoding
            fp["depth"] = depth
            fp["seed"] = seed

            fft_power_rows.append(fp)

            # ------------------------------------------------
            # Peak / low-value diagnostics
            # ------------------------------------------------

            pk = pd.read_csv(
                run_dir / "peak_low_error_summary.csv"
            )

            pk["encoding"] = encoding
            pk["depth"] = depth
            pk["seed"] = seed

            peak_rows.append(pk)

            # ------------------------------------------------
            # Config
            # ------------------------------------------------

            with open(
                run_dir / "config.json"
            ) as f:
                cfg = json.load(f)

            configs[
                (encoding, depth, seed)
            ] = cfg

            # ------------------------------------------------
            # Truth-array validation
            # ------------------------------------------------

            arr = np.load(
                run_dir / "predictions.npz",
                allow_pickle=True,
            )

            truths = np.concatenate([
                arr["y_train_true_original"],
                arr["y_val_true_original"],
                arr["y_test_true_original"],
            ])

            indices = np.concatenate([
                arr["train_target_indices"],
                arr["val_target_indices"],
                arr["test_target_indices"],
            ])

            if reference_truth is None:

                reference_truth = truths.copy()
                reference_indices = indices.copy()

            else:

                max_diff = float(
                    np.max(
                        np.abs(
                            truths
                            -
                            reference_truth
                        )
                    )
                )

                validation[
                    "max_truth_difference"
                ] = max(
                    validation[
                        "max_truth_difference"
                    ],
                    max_diff,
                )

                if max_diff > 1e-12:
                    validation[
                        "truth_arrays_identical"
                    ] = False

                if not np.array_equal(
                    indices,
                    reference_indices,
                ):
                    validation[
                        "target_indices_identical"
                    ] = False

            # ------------------------------------------------
            # Seasonal audit validation
            # ------------------------------------------------

            if encoding == "seasonal":

                with open(
                    run_dir
                    / "encoding_metadata.json"
                ) as f:
                    encmeta = json.load(f)

                conditions = [
                    encmeta.get("status")
                    == "PASS",

                    encmeta.get("encoding")
                    == "seasonal_meridian",

                    encmeta.get("season_period")
                    == 12,

                    encmeta.get("cdf_reference_n")
                    == 364,

                    encmeta.get(
                        "ansatz_changed"
                    ) is False,

                    encmeta.get(
                        "readout_changed"
                    ) is False,

                    encmeta.get(
                        "classical_head_changed"
                    ) is False,

                    encmeta.get(
                        "optimizer_protocol_changed"
                    ) is False,

                    encmeta.get(
                        "metric_protocol_changed"
                    ) is False,
                ]

                if not all(conditions):
                    validation[
                        "seasonal_encoding_audits_pass"
                    ] = False


runs = pd.DataFrame(run_rows)

history = pd.concat(
    history_rows,
    ignore_index=True,
)

split_metrics = pd.concat(
    split_metric_rows,
    ignore_index=True,
)

predictions = pd.concat(
    prediction_rows,
    ignore_index=True,
)

fft_summary = pd.concat(
    fft_summary_rows,
    ignore_index=True,
)

fft_power = pd.concat(
    fft_power_rows,
    ignore_index=True,
)

peak_low = pd.concat(
    peak_rows,
    ignore_index=True,
)

validation["n_loaded_runs"] = len(runs)

if len(runs) != 60:
    raise RuntimeError(
        f"Expected 60 runs, loaded {len(runs)}"
    )


# ============================================================
# VERIFY CONTROLLED TRAINING CONFIGURATION
# ============================================================

for depth in DEPTHS:
    for seed in SEEDS:

        ry_cfg = configs[
            ("ry", depth, seed)
        ]

        se_cfg = configs[
            ("seasonal", depth, seed)
        ]

        for field in CONTROL_FIELDS:

            rv = ry_cfg.get(field)
            sv = se_cfg.get(field)

            if rv != sv:

                validation[
                    "controlled_training_fields_match"
                ] = False

                validation[
                    "controlled_training_field_mismatches"
                ].append({
                    "depth": depth,
                    "seed": seed,
                    "field": field,
                    "ry": rv,
                    "seasonal": sv,
                })


# ============================================================
# TEST-SET SPECTRAL TABLE
# ============================================================

spec = (
    fft_summary[
        fft_summary["split"]
        == "test"
    ]
    .copy()
)

spec[
    "pred_over_true_12_band_power"
] = (
    spec[
        "pred_12_step_band_power_share"
    ]
    /
    spec[
        "true_12_step_band_power_share"
    ]
)

spec[
    "pred_over_true_short_2_6_power"
] = (
    spec[
        "pred_short_period_2_to_6_power_share"
    ]
    /
    spec[
        "true_short_period_2_to_6_power_share"
    ]
)

spec[
    "signed_12_band_power_error"
] = (
    spec[
        "pred_12_step_band_power_share"
    ]
    -
    spec[
        "true_12_step_band_power_share"
    ]
)

spec[
    "signed_short_2_6_power_error"
] = (
    spec[
        "pred_short_period_2_to_6_power_share"
    ]
    -
    spec[
        "true_short_period_2_to_6_power_share"
    ]
)

recon12 = np.abs(
    spec[
        "signed_12_band_power_error"
    ]
)

reconshort = np.abs(
    spec[
        "signed_short_2_6_power_error"
    ]
)

validation[
    "max_abs12_reconstruction_error"
] = float(
    np.max(
        np.abs(
            recon12
            -
            spec[
                "abs_12_step_band_error"
            ]
        )
    )
)

validation[
    "max_absshort_reconstruction_error"
] = float(
    np.max(
        np.abs(
            reconshort
            -
            spec[
                "abs_short_period_2_to_6_error"
            ]
        )
    )
)

if (
    validation[
        "max_abs12_reconstruction_error"
    ]
    > 1e-12
):
    raise RuntimeError(
        "Stored 12-step absolute spectral metric "
        "could not be reproduced."
    )

if (
    validation[
        "max_absshort_reconstruction_error"
    ]
    > 1e-12
):
    raise RuntimeError(
        "Stored short-period absolute spectral "
        "metric could not be reproduced."
    )


# Merge spectral directional quantities into main run table.
runs = runs.merge(
    spec[[
        "encoding",
        "depth",
        "seed",

        "true_12_step_band_power_share",
        "pred_12_step_band_power_share",

        "true_short_period_2_to_6_power_share",
        "pred_short_period_2_to_6_power_share",

        "pred_over_true_12_band_power",
        "pred_over_true_short_2_6_power",

        "signed_12_band_power_error",
        "signed_short_2_6_power_error",
    ]],
    on=[
        "encoding",
        "depth",
        "seed",
    ],
    how="left",
)


# Approximate absolute band-power ratios from Parseval scaling.
runs[
    "approx_absolute_12_band_power_ratio"
] = (
    runs[
        "test_pred_std_over_true_std"
    ] ** 2
    *
    runs[
        "pred_over_true_12_band_power"
    ]
)

runs[
    "approx_absolute_short_2_6_power_ratio"
] = (
    runs[
        "test_pred_std_over_true_std"
    ] ** 2
    *
    runs[
        "pred_over_true_short_2_6_power"
    ]
)


# ============================================================
# SAVE COMPLETE RAW AGGREGATES
# ============================================================

runs.to_csv(
    TAB / "all_60_runs.csv",
    index=False,
)

history.to_csv(
    TAB / "all_training_history.csv",
    index=False,
)

split_metrics.to_csv(
    TAB / "all_split_metrics.csv",
    index=False,
)

spec.to_csv(
    TAB / "all_test_spectral_diagnostics.csv",
    index=False,
)

peak_low.to_csv(
    TAB / "all_peak_low_diagnostics.csv",
    index=False,
)


# ============================================================
# MEAN +/- SD TABLES
# ============================================================

SUMMARY_METRICS = [
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
    "test_abs_short_period_2_to_6_error",

    "pred_over_true_12_band_power",
    "pred_over_true_short_2_6_power",

    "signed_12_band_power_error",
    "signed_short_2_6_power_error",

    "approx_absolute_12_band_power_ratio",
    "approx_absolute_short_2_6_power_ratio",

    "elapsed_time_sec",
    "max_rss_gb_python",
    "best_epoch",
    "best_val_loss_scaled",

    "train_rmse_original",
    "val_rmse_original",
]

summary_rows = []

for enc in ["ry", "seasonal"]:

    for depth in DEPTHS:

        subset = runs[
            (runs["encoding"] == enc)
            &
            (runs["depth"] == depth)
        ]

        row = {
            "encoding": enc,
            "encoding_label":
                ENCODINGS[enc]["label"],
            "depth": depth,
            "n": len(subset),
        }

        for metric in SUMMARY_METRICS:

            vals = subset[
                metric
            ].to_numpy(dtype=float)

            row[
                metric + "_mean"
            ] = float(
                np.mean(vals)
            )

            row[
                metric + "_std"
            ] = float(
                np.std(
                    vals,
                    ddof=1,
                )
            )

        summary_rows.append(row)

depth_summary = pd.DataFrame(
    summary_rows
)

depth_summary.to_csv(
    TAB / "depth_mean_sd.csv",
    index=False,
)


# Pretty report table
PRETTY_METRICS = [
    "test_rmse_original",
    "test_mae_original",
    "test_pearson_corr",
    "test_spearman_corr",
    "test_kge",
    "test_nse",
    "test_fft_spectral_cosine_similarity",
    "test_pred_std_over_true_std",
]

pretty = depth_summary[[
    "encoding",
    "encoding_label",
    "depth",
]].copy()

for metric in PRETTY_METRICS:

    pretty[metric] = depth_summary.apply(
        lambda r:
            (
                f"{r[metric + '_mean']:.4f}"
                f" ± "
                f"{r[metric + '_std']:.4f}"
            ),
        axis=1,
    )

pretty.to_csv(
    TAB / "main_metrics_mean_sd_pretty.csv",
    index=False,
)


# ============================================================
# PAIRED SEEDWISE COMPARISONS
# ============================================================

PAIR_METRICS = [
    "test_rmse_original",
    "test_mae_original",
    "test_pearson_corr",
    "test_spearman_corr",
    "test_kge",
    "test_nse",

    "test_pred_std_over_true_std",
    "test_bias",

    "test_fft_spectral_cosine_similarity",
    "test_abs_12_step_band_error",
    "test_abs_short_period_2_to_6_error",

    "pred_over_true_12_band_power",
    "pred_over_true_short_2_6_power",

    "signed_12_band_power_error",
    "signed_short_2_6_power_error",

    "elapsed_time_sec",
    "max_rss_gb_python",
]

ry = (
    runs[
        runs["encoding"]
        == "ry"
    ]
    .set_index([
        "depth",
        "seed",
    ])
)

se = (
    runs[
        runs["encoding"]
        == "seasonal"
    ]
    .set_index([
        "depth",
        "seed",
    ])
)

paired_rows = []

for depth in DEPTHS:

    for seed in SEEDS:

        r = ry.loc[
            (depth, seed)
        ]

        s = se.loc[
            (depth, seed)
        ]

        for metric in PAIR_METRICS:

            rv = float(r[metric])
            sv = float(s[metric])

            direction = REPORT_METRICS.get(
                metric,
                ("", "none"),
            )[1]

            delta = sv - rv

            if direction == "lower":
                improvement = rv - sv
                seasonal_better = sv < rv

            elif direction == "higher":
                improvement = sv - rv
                seasonal_better = sv > rv

            elif direction == "target1":
                improvement = (
                    abs(rv - 1.0)
                    -
                    abs(sv - 1.0)
                )
                seasonal_better = (
                    abs(sv - 1.0)
                    <
                    abs(rv - 1.0)
                )

            elif direction == "target0":
                improvement = (
                    abs(rv)
                    -
                    abs(sv)
                )
                seasonal_better = (
                    abs(sv)
                    <
                    abs(rv)
                )

            else:
                improvement = np.nan
                seasonal_better = False

            paired_rows.append({
                "depth": depth,
                "seed": seed,
                "metric": metric,

                "ry_value": rv,
                "seasonal_value": sv,

                "delta_seasonal_minus_ry":
                    delta,

                "improvement_positive_is_better":
                    improvement,

                "seasonal_better":
                    int(seasonal_better),
            })

paired = pd.DataFrame(
    paired_rows
)

paired.to_csv(
    TAB / "paired_seedwise_differences.csv",
    index=False,
)

paired_summary = (
    paired
    .groupby([
        "depth",
        "metric",
    ])
    .agg(
        mean_delta=(
            "delta_seasonal_minus_ry",
            "mean",
        ),
        sd_delta=(
            "delta_seasonal_minus_ry",
            "std",
        ),
        mean_improvement=(
            "improvement_positive_is_better",
            "mean",
        ),
        sd_improvement=(
            "improvement_positive_is_better",
            "std",
        ),
        seasonal_better_seed_count=(
            "seasonal_better",
            "sum",
        ),
    )
    .reset_index()
)

paired_summary.to_csv(
    TAB / "paired_depth_summary.csv",
    index=False,
)


# ============================================================
# VALIDATION OUTPUT
# ============================================================

with open(
    OUT / "validation_summary.json",
    "w",
) as f:
    json.dump(
        validation,
        f,
        indent=2,
    )

if not validation[
    "truth_arrays_identical"
]:
    raise RuntimeError(
        "Truth arrays differ across runs."
    )

if not validation[
    "target_indices_identical"
]:
    raise RuntimeError(
        "Target indices differ across runs."
    )

if not validation[
    "seasonal_encoding_audits_pass"
]:
    raise RuntimeError(
        "At least one seasonal encoding audit failed."
    )

if not validation[
    "controlled_training_fields_match"
]:
    raise RuntimeError(
        "Controlled training fields differ."
    )


# ============================================================
# LOAD THE EXACT RAW SERIES FOR BLOCH GEOMETRY
# ============================================================

baseline_cfg = configs[
    ("ry", 1, 42)
]

csv_path = (
    PROJECT_ROOT
    / baseline_cfg["csv_path"]
)

raw = pd.read_csv(
    csv_path
)

# Find the numerical column that exactly reproduces
# the stored prediction truths at target indices.
matched_series = None
matched_column = None

for col in raw.columns:

    candidate = pd.to_numeric(
        raw[col],
        errors="coerce",
    ).dropna().to_numpy(
        dtype=float
    )

    if len(candidate) <= int(
        np.max(reference_indices)
    ):
        continue

    try:
        candidate_truth = candidate[
            reference_indices
        ]
    except Exception:
        continue

    if np.allclose(
        candidate_truth,
        reference_truth,
        atol=1e-9,
        rtol=0.0,
    ):
        matched_series = candidate
        matched_column = col
        break

if matched_series is None:
    raise RuntimeError(
        "Could not identify the exact raw value column "
        "by matching stored truths."
    )

if len(matched_series) != 539:
    raise RuntimeError(
        f"Expected 539 nonmissing observations, "
        f"found {len(matched_series)}."
    )

training_reference = matched_series[
    :364
].copy()


# ============================================================
# FIGURE 01: ACTUAL-DATA BLOCH-SPHERE GEOMETRY
# ============================================================

def draw_sphere(ax):

    u = np.linspace(
        0,
        2 * np.pi,
        60,
    )

    v = np.linspace(
        0,
        np.pi,
        30,
    )

    xs = np.outer(
        np.cos(u),
        np.sin(v),
    )

    ys = np.outer(
        np.sin(u),
        np.sin(v),
    )

    zs = np.outer(
        np.ones_like(u),
        np.cos(v),
    )

    ax.plot_wireframe(
        xs,
        ys,
        zs,
        rstride=5,
        cstride=5,
        linewidth=0.35,
        alpha=0.16,
        color="gray",
    )

    ax.plot(
        [-1.15, 1.15],
        [0, 0],
        [0, 0],
        color="gray",
        linewidth=0.8,
    )

    ax.plot(
        [0, 0],
        [-1.15, 1.15],
        [0, 0],
        color="gray",
        linewidth=0.8,
    )

    ax.plot(
        [0, 0],
        [0, 0],
        [-1.15, 1.15],
        color="gray",
        linewidth=0.8,
    )

    ax.set_xlim(
        -1.15,
        1.15,
    )

    ax.set_ylim(
        -1.15,
        1.15,
    )

    ax.set_zlim(
        -1.15,
        1.15,
    )

    ax.set_box_aspect(
        (1, 1, 1)
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=24,
        azim=38,
    )


s = np.arange(
    364,
    dtype=int,
)

season_position = s % 12

# Original Ry encoding.
scaled = (
    2.0
    *
    training_reference
    /
    350.0
    -
    1.0
)

alpha = np.pi * scaled

ry_x = np.sin(alpha)
ry_y = np.zeros_like(alpha)
ry_z = np.cos(alpha)

# Seasonal meridian encoding.
sorted_ref = np.sort(
    training_reference
)

u_cdf = (
    np.searchsorted(
        sorted_ref,
        training_reference,
        side="right",
    )
    /
    float(len(sorted_ref))
)

z_rank = (
    2.0
    *
    u_cdf
    -
    1.0
)

z_rank = np.clip(
    z_rank,
    -1,
    1,
)

phi = (
    2.0
    *
    np.pi
    *
    season_position
    /
    12.0
)

radius_xy = np.sqrt(
    np.maximum(
        0.0,
        1.0 - z_rank ** 2,
    )
)

se_x = (
    radius_xy
    *
    np.cos(phi)
)

se_y = (
    radius_xy
    *
    np.sin(phi)
)

se_z = z_rank

fig = plt.figure(
    figsize=(12, 5.6)
)

ax1 = fig.add_subplot(
    121,
    projection="3d",
)

ax2 = fig.add_subplot(
    122,
    projection="3d",
)

for ax in [ax1, ax2]:
    draw_sphere(ax)

scatter1 = ax1.scatter(
    ry_x,
    ry_y,
    ry_z,
    c=season_position,
    cmap="twilight_shifted",
    s=18,
    alpha=0.72,
)

ax1.set_title(
    r"Original $R_y(\pi\tilde{x})$ encoding"
)

scatter2 = ax2.scatter(
    se_x,
    se_y,
    se_z,
    c=season_position,
    cmap="twilight_shifted",
    s=18,
    alpha=0.72,
)

ax2.set_title(
    "Seasonal-meridian encoding"
)

cbar = fig.colorbar(
    scatter2,
    ax=[ax1, ax2],
    fraction=0.026,
    pad=0.05,
)

cbar.set_label(
    "Seasonal position (s mod 12)"
)

cbar.set_ticks(
    list(range(12))
)

fig.suptitle(
    "Bloch-sphere distribution of the 364 training observations"
)

save_figure(
    fig,
    "01_encoding_bloch_geometry",
)


# ============================================================
# FIGURE 02: SIX ORIGINAL FORECASTING METRICS
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12.5, 7.2),
)

for ax, (
    metric,
    (
        label,
        _,
    ),
) in zip(
    axes.flat,
    MAIN_METRICS.items(),
):

    add_mean_sd_curve(
        ax,
        runs,
        metric,
        ylabel=label,
    )

axes[0, 0].legend(
    frameon=False
)

fig.suptitle(
    "Ideal A=1 forecasting performance: "
    "original Ry versus seasonal-meridian encoding\n"
    "Mean ± 1 SD across seeds 42--46"
)

fig.tight_layout()

save_figure(
    fig,
    "02_metrics_vs_depth",
)


# ============================================================
# FIGURE 03: ORIGINAL SPECTRAL METRICS
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13, 4.0),
)

spectral_metrics = [
    (
        "test_fft_spectral_cosine_similarity",
        "FFT spectral cosine similarity",
        None,
    ),
    (
        "test_abs_12_step_band_error",
        "Absolute 12-step band error",
        0,
    ),
    (
        "test_abs_short_period_2_to_6_error",
        "Absolute 2--6-step band error",
        0,
    ),
]

for ax, (
    metric,
    ylabel,
    ref,
) in zip(
    axes,
    spectral_metrics,
):

    add_mean_sd_curve(
        ax,
        runs,
        metric,
        ylabel=ylabel,
        reference=ref,
    )

axes[0].legend(
    frameon=False
)

fig.suptitle(
    "Original test-set spectral diagnostics"
)

fig.tight_layout()

save_figure(
    fig,
    "03_spectral_metrics_vs_depth",
)


# ============================================================
# FIGURE 04: DEPTH-1 TRAINING HISTORY
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.8, 5.2)
)

for enc in ["ry", "seasonal"]:

    subset = history[
        (history["encoding"] == enc)
        &
        (history["depth"] == 1)
    ]

    for metric, linestyle, name in [
        (
            "train_loss_scaled",
            "-",
            "train",
        ),
        (
            "val_loss_scaled",
            "--",
            "validation",
        ),
    ]:

        g = (
            subset
            .groupby("epoch")[metric]
            .agg(["mean", "std"])
        )

        x = g.index.to_numpy()
        mean = g["mean"].to_numpy()
        sd = g["std"].to_numpy()

        ax.plot(
            x,
            mean,
            linestyle=linestyle,
            color=COLORS[enc],
            linewidth=1.7,
            label=(
                f"{ENCODINGS[enc]['label']}: "
                f"{name}"
            ),
        )

        ax.fill_between(
            x,
            mean - sd,
            mean + sd,
            color=COLORS[enc],
            alpha=0.10,
        )

ax.set_xlabel("Epoch")
ax.set_ylabel("Scaled MSE loss")

ax.set_title(
    "Depth-1 ideal training dynamics"
)

ax.grid(alpha=0.2)

ax.legend(
    frameon=False,
    fontsize=9,
)

save_figure(
    fig,
    "04_depth1_training_history",
)


# ============================================================
# FIGURE 05: RUNTIME
# ============================================================

runtime = runs.copy()

runtime[
    "runtime_min"
] = (
    runtime["elapsed_time_sec"]
    /
    60.0
)

fig, ax = plt.subplots(
    figsize=(7.8, 4.8)
)

add_mean_sd_curve(
    ax,
    runtime,
    "runtime_min",
    ylabel="Runtime [min]",
)

ax.legend(
    frameon=False
)

ax.set_title(
    "Ideal training runtime"
)

save_figure(
    fig,
    "05_runtime_vs_depth",
)


# ============================================================
# FIGURE 06: TEST PREDICTION TRAJECTORIES
# ============================================================

test_pred = predictions[
    predictions["split"]
    == "test"
].copy()

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12.5, 11),
    sharex=True,
)

for depth, ax in zip(
    DEPTHS,
    axes.flat,
):

    depth_data = test_pred[
        test_pred["depth"]
        == depth
    ]

    truth = (
        depth_data[
            depth_data["encoding"]
            == "ry"
        ]
        .groupby("target_index")[
            "true_original"
        ]
        .first()
        .sort_index()
    )

    ax.plot(
        truth.index,
        truth.values,
        color="black",
        linewidth=1.2,
        label="True",
    )

    for enc in ["ry", "seasonal"]:

        g = (
            depth_data[
                depth_data["encoding"]
                == enc
            ]
            .groupby("target_index")[
                "pred_original"
            ]
            .agg(["mean", "std"])
            .sort_index()
        )

        x = g.index.to_numpy()
        mean = g["mean"].to_numpy()
        sd = g["std"].to_numpy()

        ax.plot(
            x,
            mean,
            color=COLORS[enc],
            linewidth=1.4,
            label=ENCODINGS[enc]["label"],
        )

        ax.fill_between(
            x,
            mean - sd,
            mean + sd,
            color=COLORS[enc],
            alpha=0.16,
        )

    ax.set_title(
        f"Depth {depth}"
    )

    ax.set_ylabel(
        "Target"
    )

    ax.grid(
        alpha=0.16
    )

axes[-1, 0].set_xlabel(
    "Test target index"
)

axes[-1, 1].set_xlabel(
    "Test target index"
)

axes[0, 0].legend(
    frameon=False,
    fontsize=8,
)

fig.suptitle(
    "Held-out ideal test predictions\n"
    "Mean ± 1 SD across seeds 42--46"
)

fig.tight_layout()

save_figure(
    fig,
    "06_test_predictions",
)


# ============================================================
# FIGURE 07: FAILURE-MODE / CALIBRATION METRICS
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10.5, 7.4),
)

failure_panels = [
    (
        "test_pred_std_over_true_std",
        r"$\sigma_{\hat y}/\sigma_y$",
        1,
    ),
    (
        "test_bias",
        "Bias",
        0,
    ),
    (
        "test_abs_12_step_band_error",
        "Absolute 12-step band error",
        0,
    ),
    (
        "test_abs_short_period_2_to_6_error",
        "Absolute 2--6-step band error",
        0,
    ),
]

for ax, (
    metric,
    ylabel,
    ref,
) in zip(
    axes.flat,
    failure_panels,
):

    add_mean_sd_curve(
        ax,
        runs,
        metric,
        ylabel=ylabel,
        reference=ref,
    )

axes[0, 0].legend(
    frameon=False
)

fig.suptitle(
    "Amplitude, bias, and spectral-band diagnostics"
)

fig.tight_layout()

save_figure(
    fig,
    "07_failure_modes",
)


# ============================================================
# FIGURE 08: NORMALIZED SPECTRAL POWER RATIOS
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.5, 4.2),
)

for ax, metric, title in [
    (
        axes[0],
        "pred_over_true_12_band_power",
        "Dominant seasonal band",
    ),
    (
        axes[1],
        "pred_over_true_short_2_6_power",
        "Short-period 2--6-step band",
    ),
]:

    add_mean_sd_curve(
        ax,
        runs,
        metric,
        ylabel="Predicted / true band power",
        reference=1,
        title=title,
    )

axes[0].legend(
    frameon=False
)

fig.suptitle(
    "Normalized spectral-band power reproduction"
)

fig.tight_layout()

save_figure(
    fig,
    "08_spectral_power_ratio",
)


# ============================================================
# FIGURE 09: SIGNED SPECTRAL POWER ERRORS
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.5, 4.2),
)

for ax, metric, title in [
    (
        axes[0],
        "signed_12_band_power_error",
        "Dominant seasonal band",
    ),
    (
        axes[1],
        "signed_short_2_6_power_error",
        "Short-period 2--6-step band",
    ),
]:

    add_mean_sd_curve(
        ax,
        runs,
        metric,
        ylabel=(
            r"$P_{\mathrm{pred}}"
            r"-P_{\mathrm{true}}$"
        ),
        reference=0,
        title=title,
    )

axes[0].legend(
    frameon=False
)

fig.suptitle(
    "Signed normalized spectral-power error\n"
    "Positive = excess power; negative = missing power"
)

fig.tight_layout()

save_figure(
    fig,
    "09_signed_spectral_power",
)


# ============================================================
# FIGURE 10: FULL TEST-SET FFT SPECTRA
# ============================================================

test_fft = fft_power[
    fft_power["split"]
    == "test"
].copy()

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12.5, 10.5),
)

for depth, ax in zip(
    DEPTHS,
    axes.flat,
):

    d = test_fft[
        test_fft["depth"]
        == depth
    ]

    truth = (
        d[
            (d["encoding"] == "ry")
            &
            (d["series"] == "true")
        ]
        .groupby("period_steps")[
            "power_share_non_dc"
        ]
        .first()
        .reset_index()
    )

    truth = truth[
        np.isfinite(
            truth["period_steps"]
        )
    ]

    truth = truth[
        truth["period_steps"]
        <= 105
    ]

    truth = truth.sort_values(
        "period_steps"
    )

    ax.plot(
        truth["period_steps"],
        truth["power_share_non_dc"],
        color="black",
        linewidth=1.2,
        label="True",
    )

    for enc in ["ry", "seasonal"]:

        pred = d[
            (d["encoding"] == enc)
            &
            (d["series"] == "pred")
        ].copy()

        pred = pred[
            np.isfinite(
                pred["period_steps"]
            )
        ]

        pred = pred[
            pred["period_steps"]
            <= 105
        ]

        g = (
            pred
            .groupby("period_steps")[
                "power_share_non_dc"
            ]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("period_steps")
        )

        x = g[
            "period_steps"
        ].to_numpy()

        mean = g[
            "mean"
        ].to_numpy()

        sd = g[
            "std"
        ].to_numpy()

        ax.plot(
            x,
            mean,
            color=COLORS[enc],
            linewidth=1.3,
            label=ENCODINGS[enc]["label"],
        )

        ax.fill_between(
            x,
            np.maximum(
                mean - sd,
                0,
            ),
            mean + sd,
            color=COLORS[enc],
            alpha=0.14,
        )

    ax.set_title(
        f"Depth {depth}"
    )

    ax.set_xlabel(
        "Period [steps]"
    )

    ax.set_ylabel(
        "Normalized power"
    )

    ax.grid(
        alpha=0.15
    )

axes[0, 0].legend(
    frameon=False,
    fontsize=8,
)

fig.suptitle(
    "Test-set normalized Fourier spectra"
)

fig.tight_layout()

save_figure(
    fig,
    "10_fft_spectra",
)


# ============================================================
# FIGURE 11: SEED ROBUSTNESS
# ============================================================

robust_rows = []

for enc in ["ry", "seasonal"]:

    for depth in DEPTHS:

        d = runs[
            (runs["encoding"] == enc)
            &
            (runs["depth"] == depth)
        ]

        robust_rows.append({
            "encoding": enc,
            "depth": depth,

            "rmse_cv":
                safe_cv(
                    d[
                        "test_rmse_original"
                    ]
                ),

            "pearson_cv":
                safe_cv(
                    d[
                        "test_pearson_corr"
                    ]
                ),

            "fft_cv":
                safe_cv(
                    d[
                        "test_fft_spectral_cosine_similarity"
                    ]
                ),
        })

robust = pd.DataFrame(
    robust_rows
)

robust.to_csv(
    TAB / "seed_robustness.csv",
    index=False,
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12, 3.8),
)

for ax, metric, title in [
    (
        axes[0],
        "rmse_cv",
        "Test RMSE",
    ),
    (
        axes[1],
        "pearson_cv",
        "Test Pearson",
    ),
    (
        axes[2],
        "fft_cv",
        "FFT cosine",
    ),
]:

    for enc in ["ry", "seasonal"]:

        d = robust[
            robust["encoding"]
            == enc
        ]

        ax.plot(
            d["depth"],
            d[metric],
            marker="o",
            color=COLORS[enc],
            linewidth=1.6,
            label=ENCODINGS[enc]["label"],
        )

    ax.set_xticks(
        DEPTHS
    )

    ax.set_xlabel(
        "Reuploading depth"
    )

    ax.set_ylabel(
        "Coefficient of variation"
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.18
    )

axes[0].legend(
    frameon=False,
    fontsize=8,
)

fig.suptitle(
    "Seed-to-seed variability"
)

fig.tight_layout()

save_figure(
    fig,
    "11_seed_robustness",
)


# ============================================================
# FIGURE 12: MEMORY
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.8, 4.8)
)

add_mean_sd_curve(
    ax,
    runs,
    "max_rss_gb_python",
    ylabel="Maximum resident memory [GB]",
)

ax.legend(
    frameon=False
)

ax.set_title(
    "Maximum Python resident memory"
)

save_figure(
    fig,
    "12_memory_vs_depth",
)


# ============================================================
# FIGURE 13: TRAIN / VAL / TEST RMSE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4.5),
    sharey=True,
)

for enc, ax in zip(
    ["ry", "seasonal"],
    axes,
):

    d = split_metrics[
        split_metrics["encoding"]
        == enc
    ]

    for split, linestyle in [
        ("train", "-"),
        ("val", "--"),
        ("test", "-."),
    ]:

        g = (
            d[
                d["split"]
                == split
            ]
            .groupby("depth")[
                "rmse"
            ]
            .agg(["mean", "std"])
            .reindex(DEPTHS)
        )

        mean = g[
            "mean"
        ].to_numpy()

        sd = g[
            "std"
        ].to_numpy()

        x = np.asarray(
            DEPTHS
        )

        ax.plot(
            x,
            mean,
            linestyle=linestyle,
            marker="o",
            linewidth=1.5,
            label=split.capitalize(),
        )

        ax.fill_between(
            x,
            mean - sd,
            mean + sd,
            alpha=0.12,
        )

    ax.set_xticks(
        DEPTHS
    )

    ax.set_xlabel(
        "Reuploading depth"
    )

    ax.set_title(
        ENCODINGS[enc]["label"]
    )

    ax.grid(
        alpha=0.18
    )

axes[0].set_ylabel(
    "RMSE in original units"
)

axes[0].legend(
    frameon=False
)

fig.suptitle(
    "Train, validation, and test RMSE"
)

fig.tight_layout()

save_figure(
    fig,
    "13_train_val_test_rmse",
)


# ============================================================
# FIGURE 14: ALL TRAINING HISTORIES
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12, 10.5),
    sharex=True,
)

for depth, ax in zip(
    DEPTHS,
    axes.flat,
):

    for enc in ["ry", "seasonal"]:

        d = history[
            (history["encoding"] == enc)
            &
            (history["depth"] == depth)
        ]

        for metric, linestyle, name in [
            (
                "train_loss_scaled",
                "-",
                "train",
            ),
            (
                "val_loss_scaled",
                "--",
                "validation",
            ),
        ]:

            g = (
                d
                .groupby("epoch")[metric]
                .agg(["mean", "std"])
            )

            x = g.index.to_numpy()

            mean = g[
                "mean"
            ].to_numpy()

            sd = g[
                "std"
            ].to_numpy()

            ax.plot(
                x,
                mean,
                linestyle=linestyle,
                color=COLORS[enc],
                linewidth=1.35,
                label=(
                    f"{ENCODINGS[enc]['label']}: "
                    f"{name}"
                ),
            )

            ax.fill_between(
                x,
                mean - sd,
                mean + sd,
                color=COLORS[enc],
                alpha=0.08,
            )

    ax.set_title(
        f"Depth {depth}"
    )

    ax.set_ylabel(
        "Scaled MSE"
    )

    ax.grid(
        alpha=0.15
    )

axes[-1, 0].set_xlabel(
    "Epoch"
)

axes[-1, 1].set_xlabel(
    "Epoch"
)

axes[0, 0].legend(
    frameon=False,
    fontsize=7,
)

fig.suptitle(
    "Five-seed ideal training histories"
)

fig.tight_layout()

save_figure(
    fig,
    "14_training_histories_all_depths",
)


# ============================================================
# FIGURE 15: PAIRED RMSE IMPROVEMENT
# ============================================================

rmse_pair = paired[
    paired["metric"]
    == "test_rmse_original"
]

g = (
    rmse_pair
    .groupby("depth")[
        "improvement_positive_is_better"
    ]
    .agg(["mean", "std"])
    .reindex(DEPTHS)
)

fig, ax = plt.subplots(
    figsize=(7.8, 4.8)
)

ax.errorbar(
    DEPTHS,
    g["mean"],
    yerr=g["std"],
    marker="o",
    linewidth=1.6,
    capsize=4,
)

ax.axhline(
    0,
    linestyle="--",
    color="black",
    linewidth=1,
)

ax.set_xticks(
    DEPTHS
)

ax.set_xlabel(
    "Reuploading depth"
)

ax.set_ylabel(
    r"RMSE improvement "
    r"$(R_y-\mathrm{seasonal})$"
)

ax.set_title(
    "Paired seedwise test-RMSE improvement"
)

ax.grid(
    alpha=0.18
)

save_figure(
    fig,
    "15_paired_rmse_improvement",
)


# ============================================================
# KEY SUMMARY
# ============================================================

summary = {
    "raw_series_column":
        str(matched_column),

    "training_reference_n":
        364,

    "validation":
        validation,

    "best_depth_by_mean_test_rmse":
        {},

    "depthwise_comparison":
        {},
}

for enc in ["ry", "seasonal"]:

    ds = depth_summary[
        depth_summary["encoding"]
        == enc
    ]

    best = ds.loc[
        ds[
            "test_rmse_original_mean"
        ].idxmin()
    ]

    summary[
        "best_depth_by_mean_test_rmse"
    ][enc] = {
        "depth":
            int(best["depth"]),

        "rmse_mean":
            float(
                best[
                    "test_rmse_original_mean"
                ]
            ),

        "rmse_std":
            float(
                best[
                    "test_rmse_original_std"
                ]
            ),

        "pearson_mean":
            float(
                best[
                    "test_pearson_corr_mean"
                ]
            ),

        "kge_mean":
            float(
                best[
                    "test_kge_mean"
                ]
            ),

        "fft_mean":
            float(
                best[
                    "test_fft_spectral_cosine_similarity_mean"
                ]
            ),
    }


for depth in DEPTHS:

    r = depth_summary[
        (depth_summary["encoding"] == "ry")
        &
        (depth_summary["depth"] == depth)
    ].iloc[0]

    srow = depth_summary[
        (depth_summary["encoding"] == "seasonal")
        &
        (depth_summary["depth"] == depth)
    ].iloc[0]

    ry_rmse = float(
        r[
            "test_rmse_original_mean"
        ]
    )

    se_rmse = float(
        srow[
            "test_rmse_original_mean"
        ]
    )

    summary[
        "depthwise_comparison"
    ][str(depth)] = {
        "ry_rmse_mean":
            ry_rmse,

        "seasonal_rmse_mean":
            se_rmse,

        "rmse_percent_improvement":
            100.0
            *
            (
                ry_rmse
                -
                se_rmse
            )
            /
            ry_rmse,

        "ry_pearson_mean":
            float(
                r[
                    "test_pearson_corr_mean"
                ]
            ),

        "seasonal_pearson_mean":
            float(
                srow[
                    "test_pearson_corr_mean"
                ]
            ),

        "ry_kge_mean":
            float(
                r[
                    "test_kge_mean"
                ]
            ),

        "seasonal_kge_mean":
            float(
                srow[
                    "test_kge_mean"
                ]
            ),

        "ry_fft_mean":
            float(
                r[
                    "test_fft_spectral_cosine_similarity_mean"
                ]
            ),

        "seasonal_fft_mean":
            float(
                srow[
                    "test_fft_spectral_cosine_similarity_mean"
                ]
            ),

        "ry_amp_mean":
            float(
                r[
                    "test_pred_std_over_true_std_mean"
                ]
            ),

        "seasonal_amp_mean":
            float(
                srow[
                    "test_pred_std_over_true_std_mean"
                ]
            ),

        "ry_p12_ratio_mean":
            float(
                r[
                    "pred_over_true_12_band_power_mean"
                ]
            ),

        "seasonal_p12_ratio_mean":
            float(
                srow[
                    "pred_over_true_12_band_power_mean"
                ]
            ),

        "ry_short_ratio_mean":
            float(
                r[
                    "pred_over_true_short_2_6_power_mean"
                ]
            ),

        "seasonal_short_ratio_mean":
            float(
                srow[
                    "pred_over_true_short_2_6_power_mean"
                ]
            ),
    }


with open(
    OUT / "seasonal_vs_ry_summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# PRINT COMPACT REPORT
# ============================================================

print()
print("=" * 120)
print(
    "SEASONAL-MERIDIAN VS ORIGINAL Ry "
    "A=1 ANALYSIS"
)
print("=" * 120)

print()
print("Validation:")
print(
    json.dumps(
        validation,
        indent=2,
    )
)

print()
print(
    "Raw series column identified as:",
    matched_column,
)

print()
print("-" * 120)
print(
    "CORE FIVE-SEED TEST SUMMARY"
)
print("-" * 120)

display_cols = [
    "encoding",
    "depth",

    "test_rmse_original_mean",
    "test_rmse_original_std",

    "test_pearson_corr_mean",
    "test_pearson_corr_std",

    "test_kge_mean",
    "test_kge_std",

    "test_fft_spectral_cosine_similarity_mean",
    "test_fft_spectral_cosine_similarity_std",

    "test_pred_std_over_true_std_mean",
    "test_pred_std_over_true_std_std",

    "pred_over_true_12_band_power_mean",
    "pred_over_true_short_2_6_power_mean",
]

print(
    depth_summary[
        display_cols
    ].to_string(
        index=False
    )
)

print()
print("-" * 120)
print(
    "PAIRED RMSE COMPARISON"
)
print("-" * 120)

print(
    paired_summary[
        paired_summary["metric"]
        == "test_rmse_original"
    ][[
        "depth",
        "mean_delta",
        "sd_delta",
        "mean_improvement",
        "seasonal_better_seed_count",
    ]]
    .to_string(
        index=False
    )
)

print()
print("-" * 120)
print("OUTPUT FILES")
print("-" * 120)

print(
    "Tables:",
    TAB,
)

print(
    "Figures:",
    FIG,
)

print(
    "Summary:",
    OUT / "seasonal_vs_ry_summary.json",
)

print()
print("=" * 120)
print("STATUS: PASS")
print("=" * 120)
