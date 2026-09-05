#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(
    "/umbc/rs/pi_deffner/users/devjyot1/projects/"
    "Quantum-Machine-Learning-"
)

ANALYSIS_ROOT = (
    PROJECT_ROOT
    / "data_reupload"
    / "analysis"
    / "four_encoding_master"
)

TABLE_DIR = ANALYSIS_ROOT / "output" / "tables"
FIG_DIR = ANALYSIS_ROOT / "output" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load canonical master products
# ============================================================

manifest = pd.read_csv(
    TABLE_DIR / "master_run_manifest.csv"
)

summary = pd.read_csv(
    TABLE_DIR / "forecasting_depth_summary.csv"
)

paired_ry = pd.read_csv(
    TABLE_DIR / "paired_seedwise_vs_ry.csv"
)

paired_fixed = pd.read_csv(
    TABLE_DIR / "paired_seedwise_learnable_vs_fixed.csv"
)

paired_azimuth = pd.read_csv(
    TABLE_DIR
    / "paired_rmse_azimuth_vs_polar_depth_summary.csv"
)


# ============================================================
# Canonical ordering / labels
# ============================================================

ENCODINGS = [
    "ry",
    "same_scalar_rxryrz",
    "seasonal_meridian",
    "learnable_seasonal_cdf",
    "learnable_seasonal_cdf_rz",
]

LABELS = {
    "ry": r"Standard $R_y$",
    "same_scalar_rxryrz":
        r"Same-scalar $R_xR_yR_z$",
    "seasonal_meridian":
        "Fixed seasonal CDF",
    "learnable_seasonal_cdf":
        "Learnable seasonal CDF",
    "learnable_seasonal_cdf_rz":
        "Learnable polar-azimuth seasonal CDF",
}

MARKERS = {
    "ry": "o",
    "same_scalar_rxryrz": "s",
    "seasonal_meridian": "^",
    "learnable_seasonal_cdf": "D",
    "learnable_seasonal_cdf_rz": "P",
}

DEPTHS = np.arange(1, 7)


# ============================================================
# Helpers
# ============================================================

def save_figure(fig, stem):
    fig.savefig(
        FIG_DIR / f"{stem}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        FIG_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def setup_axis(ax, xlabel=True):
    ax.grid(
        alpha=0.20,
        linewidth=0.7,
    )

    ax.set_xticks(DEPTHS)

    if xlabel:
        ax.set_xlabel(
            "Reuploading depth $L$"
        )


def plot_metric(
    ax,
    mean_col,
    std_col,
    ylabel,
    ideal=None,
):

    for enc in ENCODINGS:

        d = (
            summary[
                summary["encoding"] == enc
            ]
            .sort_values("depth")
        )

        x = d["depth"].to_numpy()
        y = d[mean_col].to_numpy()
        s = d[std_col].to_numpy()

        line, = ax.plot(
            x,
            y,
            marker=MARKERS[enc],
            linewidth=1.8,
            markersize=5,
            label=LABELS[enc],
        )

        ax.fill_between(
            x,
            y - s,
            y + s,
            alpha=0.13,
            color=line.get_color(),
        )

    if ideal is not None:
        ax.axhline(
            ideal,
            linestyle="--",
            linewidth=1.0,
            color="black",
            alpha=0.55,
        )

    setup_axis(ax)
    ax.set_ylabel(ylabel)


# ============================================================
# Figure 04
# Six principal test metrics
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(11.5, 11.0),
    sharex=True,
)

metric_specs = [
    (
        "test_rmse_original_mean",
        "test_rmse_original_std",
        "Test RMSE",
        None,
    ),
    (
        "test_mae_original_mean",
        "test_mae_original_std",
        "Test MAE",
        None,
    ),
    (
        "test_pearson_corr_mean",
        "test_pearson_corr_std",
        "Pearson correlation",
        None,
    ),
    (
        "test_spearman_corr_mean",
        "test_spearman_corr_std",
        "Spearman correlation",
        None,
    ),
    (
        "test_kge_mean",
        "test_kge_std",
        "KGE",
        1.0,
    ),
    (
        "test_nse_mean",
        "test_nse_std",
        "NSE",
        1.0,
    ),
]

for ax, spec in zip(
    axes.flat,
    metric_specs,
):
    plot_metric(
        ax,
        *spec,
    )

handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.995),
)

fig.suptitle(
    "Ideal QNN forecasting performance across input encodings",
    y=1.025,
    fontsize=14,
)

fig.text(
    0.5,
    0.005,
    "Five-seed mean ± 1 SD; identical A=1 architecture and training protocol",
    ha="center",
    fontsize=10,
)

fig.tight_layout(
    rect=[0, 0.025, 1, 0.96]
)

save_figure(
    fig,
    "04_main_metrics_vs_depth",
)


# ============================================================
# Figure 05
# Change relative to each encoding's own L=1
# ============================================================

relative_metrics = [
    (
        "test_rmse_original",
        r"$\Delta$ RMSE from $L=1$",
        False,
    ),
    (
        "test_pearson_corr",
        r"$\Delta$ Pearson from $L=1$",
        True,
    ),
    (
        "test_kge",
        r"$\Delta$ KGE from $L=1$",
        True,
    ),
    (
        "test_fft_spectral_cosine_similarity",
        r"$\Delta$ FFT cosine from $L=1$",
        True,
    ),
    (
        "test_pred_std_over_true_std",
        r"$\Delta$ amplitude ratio from $L=1$",
        True,
    ),
]

fig, axes = plt.subplots(
    3,
    2,
    figsize=(11.5, 11.0),
)

axes = axes.flat

for ax, (
    metric,
    ylabel,
    larger_is_better,
) in zip(
    axes,
    relative_metrics,
):

    for enc in ENCODINGS:

        enc_df = manifest[
            manifest["encoding"] == enc
        ]

        seed_deltas = []

        for seed in sorted(
            enc_df["seed"].unique()
        ):

            s = (
                enc_df[
                    enc_df["seed"] == seed
                ]
                .sort_values("depth")
            )

            baseline = float(
                s[
                    s["depth"] == 1
                ][metric].iloc[0]
            )

            for _, row in s.iterrows():

                seed_deltas.append({
                    "depth": int(
                        row["depth"]
                    ),
                    "delta":
                        float(row[metric])
                        -
                        baseline,
                })

        temp = pd.DataFrame(
            seed_deltas
        )

        g = (
            temp
            .groupby("depth")["delta"]
            .agg(["mean", "std"])
            .reset_index()
        )

        x = g["depth"].to_numpy()
        y = g["mean"].to_numpy()
        sd = g["std"].to_numpy()

        line, = ax.plot(
            x,
            y,
            marker=MARKERS[enc],
            linewidth=1.8,
            markersize=5,
            label=LABELS[enc],
        )

        ax.fill_between(
            x,
            y - sd,
            y + sd,
            alpha=0.13,
            color=line.get_color(),
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        color="black",
        alpha=0.55,
    )

    setup_axis(ax)
    ax.set_ylabel(ylabel)

axes[-1].axis("off")

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.995),
)

fig.suptitle(
    "Depth-induced change relative to each encoding's shallow model",
    y=1.025,
    fontsize=14,
)

fig.text(
    0.5,
    0.005,
    "Differences are computed seedwise relative to the matched L=1 run before aggregation",
    ha="center",
    fontsize=10,
)

fig.tight_layout(
    rect=[0, 0.025, 1, 0.96]
)

save_figure(
    fig,
    "05_depth_degradation_relative_to_L1",
)


# ============================================================
# Figure 07
# Amplitude and bias
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11.5, 4.5),
)

plot_metric(
    axes[0],
    "test_pred_std_over_true_std_mean",
    "test_pred_std_over_true_std_std",
    r"Amplitude ratio $\sigma_{\hat y}/\sigma_y$",
    ideal=1.0,
)

plot_metric(
    axes[1],
    "test_bias_mean",
    "test_bias_std",
    "Bias (prediction - truth)",
    ideal=0.0,
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.04),
)

fig.suptitle(
    "Prediction amplitude and bias",
    y=1.10,
    fontsize=14,
)

fig.tight_layout()

save_figure(
    fig,
    "07_amplitude_bias_vs_depth",
)


# ============================================================
# Figure 08
# Global spectral diagnostics
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(14.5, 4.5),
)

plot_metric(
    axes[0],
    "test_fft_spectral_cosine_similarity_mean",
    "test_fft_spectral_cosine_similarity_std",
    "FFT spectral cosine similarity",
    ideal=1.0,
)

plot_metric(
    axes[1],
    "test_abs_12_step_band_error_mean",
    "test_abs_12_step_band_error_std",
    "Absolute 12-step band error",
    ideal=0.0,
)

plot_metric(
    axes[2],
    "test_abs_short_period_2_to_6_error_mean",
    "test_abs_short_period_2_to_6_error_std",
    "Absolute 2–6-step band error",
    ideal=0.0,
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.04),
)

fig.suptitle(
    "Test-set spectral diagnostics",
    y=1.10,
    fontsize=14,
)

fig.tight_layout()

save_figure(
    fig,
    "08_spectral_metrics_vs_depth",
)


# ============================================================
# Figure 09
# Normalized spectral band power ratios
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11.5, 4.5),
)

plot_metric(
    axes[0],
    "test_12_step_power_ratio_mean",
    "test_12_step_power_ratio_std",
    r"$P_{12}^{pred}/P_{12}^{true}$",
    ideal=1.0,
)

plot_metric(
    axes[1],
    "test_short_period_2_to_6_power_ratio_mean",
    "test_short_period_2_to_6_power_ratio_std",
    r"$P_{2:6}^{pred}/P_{2:6}^{true}$",
    ideal=1.0,
)

axes[0].set_title(
    "Dominant seasonal band"
)

axes[1].set_title(
    "Short-period band"
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.04),
)

fig.suptitle(
    "Normalized spectral-band power reproduction",
    y=1.10,
    fontsize=14,
)

fig.tight_layout()

save_figure(
    fig,
    "09_normalized_band_power_ratios",
)


# ============================================================
# Figure 10
# Signed spectral errors
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11.5, 4.5),
)

plot_metric(
    axes[0],
    "test_signed_12_step_band_error_mean",
    "test_signed_12_step_band_error_std",
    r"$P_{12}^{pred}-P_{12}^{true}$",
    ideal=0.0,
)

plot_metric(
    axes[1],
    "test_signed_short_period_2_to_6_error_mean",
    "test_signed_short_period_2_to_6_error_std",
    r"$P_{2:6}^{pred}-P_{2:6}^{true}$",
    ideal=0.0,
)

axes[0].set_title(
    "Dominant seasonal band"
)

axes[1].set_title(
    "Short-period band"
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.04),
)

fig.suptitle(
    "Signed normalized spectral-power error",
    y=1.10,
    fontsize=14,
)

fig.tight_layout()

save_figure(
    fig,
    "10_signed_band_errors",
)


# ============================================================
# Paired summaries
# ============================================================

paired_summary_rows = []

for candidate in [
    "same_scalar_rxryrz",
    "seasonal_meridian",
    "learnable_seasonal_cdf",
    "learnable_seasonal_cdf_rz",
]:

    d = paired_ry[
        paired_ry["candidate_encoding"]
        ==
        candidate
    ]

    for depth in DEPTHS:

        x = d[
            d["depth"] == depth
        ]

        vals = x[
            "rmse_improvement_vs_ry"
        ].to_numpy()

        paired_summary_rows.append({
            "candidate_encoding":
                candidate,
            "depth":
                depth,
            "mean":
                np.mean(vals),
            "std":
                np.std(
                    vals,
                    ddof=1,
                ),
            "n_better":
                int(
                    np.sum(vals > 0)
                ),
        })

paired_summary = pd.DataFrame(
    paired_summary_rows
)

paired_summary.to_csv(
    TABLE_DIR
    / "paired_rmse_vs_ry_depth_summary.csv",
    index=False,
)


# ============================================================
# Figure 12
# Paired RMSE improvement relative to Ry
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.0)
)

for candidate in [
    "same_scalar_rxryrz",
    "seasonal_meridian",
    "learnable_seasonal_cdf",
    "learnable_seasonal_cdf_rz",
]:

    d = (
        paired_summary[
            paired_summary[
                "candidate_encoding"
            ]
            ==
            candidate
        ]
        .sort_values("depth")
    )

    x = d["depth"].to_numpy()
    y = d["mean"].to_numpy()
    sd = d["std"].to_numpy()

    line, = ax.plot(
        x,
        y,
        marker=MARKERS[candidate],
        linewidth=1.8,
        markersize=6,
        label=LABELS[candidate],
    )

    ax.fill_between(
        x,
        y - sd,
        y + sd,
        alpha=0.13,
        color=line.get_color(),
    )

ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    color="black",
)

setup_axis(ax)

ax.set_ylabel(
    r"RMSE improvement relative to $R_y$"
)

ax.set_title(
    "Paired seedwise RMSE improvement"
)

ax.legend(
    frameon=False
)

fig.tight_layout()

save_figure(
    fig,
    "12_paired_rmse_vs_ry",
)


# ============================================================
# Fixed vs learnable paired summary
# ============================================================

fixed_summary_rows = []

for depth in DEPTHS:

    d = paired_fixed[
        paired_fixed["depth"]
        ==
        depth
    ]

    vals = d[
        "rmse_improvement_learnable_vs_fixed"
    ].to_numpy()

    fixed_summary_rows.append({
        "depth":
            depth,
        "mean":
            np.mean(vals),
        "std":
            np.std(
                vals,
                ddof=1,
            ),
        "n_learnable_better":
            int(
                np.sum(vals > 0)
            ),
    })

fixed_summary = pd.DataFrame(
    fixed_summary_rows
)

fixed_summary.to_csv(
    TABLE_DIR
    / "paired_rmse_learnable_vs_fixed_depth_summary.csv",
    index=False,
)


# ============================================================
# Figure 13
# Learnable seasonal vs fixed seasonal
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.0)
)

x = fixed_summary[
    "depth"
].to_numpy()

y = fixed_summary[
    "mean"
].to_numpy()

sd = fixed_summary[
    "std"
].to_numpy()

ax.errorbar(
    x,
    y,
    yerr=sd,
    marker="o",
    linewidth=1.8,
    capsize=4,
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    color="black",
)

setup_axis(ax)

ax.set_ylabel(
    "RMSE improvement\n"
    "(fixed seasonal − learnable seasonal)"
)

ax.set_title(
    "Paired learnable-versus-fixed seasonal comparison"
)

fig.tight_layout()

save_figure(
    fig,
    "13_learnable_vs_fixed_paired_rmse",
)


# ============================================================
# Figure 14
# Runtime
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.2)
)

plot_metric(
    ax,
    "elapsed_time_min_mean",
    "elapsed_time_min_std",
    "Training runtime [min]",
)

ax.legend(
    frameon=False,
    ncol=2,
)

ax.set_title(
    "Ideal training runtime"
)

fig.tight_layout()

save_figure(
    fig,
    "14_runtime_vs_depth",
)


# ============================================================
# Figure 15
# Parameter count
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.2)
)

for enc in ENCODINGS:

    d = (
        summary[
            summary["encoding"] == enc
        ]
        .sort_values("depth")
    )

    ax.plot(
        d["depth"],
        d["total_parameter_count"],
        marker=MARKERS[enc],
        linewidth=1.8,
        markersize=6,
        label=LABELS[enc],
    )

setup_axis(ax)

ax.set_ylabel(
    "Total trainable parameters"
)

ax.set_title(
    "Model parameter count"
)

ax.legend(
    frameon=False,
    ncol=2,
)

fig.tight_layout()

save_figure(
    fig,
    "15_parameter_count_vs_depth",
)


# ============================================================
# Figure 16
# Nested comparison:
# learnable polar+azimuth vs learnable polar-only
#
# Positive values mean that adding trainable azimuth lowers RMSE.
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.0)
)

d = paired_azimuth.sort_values(
    "depth"
)

x = d[
    "depth"
].to_numpy()

y = d[
    "mean_rmse_improvement_azimuth_vs_polar"
].to_numpy()

sd = d[
    "std_rmse_improvement_azimuth_vs_polar"
].to_numpy()

ax.errorbar(
    x,
    y,
    yerr=sd,
    marker="P",
    capsize=4,
    linewidth=1.8,
    label=(
        "Learnable polar+azimuth "
        "vs polar-only"
    ),
)

ax.axhline(
    0.0,
    color="black",
    linestyle="--",
    linewidth=1.0,
)

setup_axis(
    ax,
    xlabel=True,
)

ax.set_ylabel(
    "Paired test RMSE improvement"
)

ax.set_title(
    "Effect of learning the seasonal azimuth"
)

ax.legend(
    frameon=False,
)

fig.tight_layout()

save_figure(
    fig,
    "16_paired_rmse_azimuth_vs_polar",
)


# ============================================================
# Human-readable paired summaries
# ============================================================

print("=" * 120)
print("FIVE-ENCODING EMPIRICAL FIGURE GENERATION")
print("=" * 120)

print()
print("PAIRED RMSE IMPROVEMENT RELATIVE TO STANDARD Ry")
print("Positive = candidate lower RMSE")
print()

for candidate in [
    "same_scalar_rxryrz",
    "seasonal_meridian",
    "learnable_seasonal_cdf",
    "learnable_seasonal_cdf_rz",
]:

    print(LABELS[candidate])

    d = paired_summary[
        paired_summary["candidate_encoding"]
        ==
        candidate
    ]

    for _, row in d.iterrows():

        print(
            f"  L={int(row['depth'])}: "
            f"{row['mean']:.4f} ± "
            f"{row['std']:.4f} "
            f"({int(row['n_better'])}/5 seeds better)"
        )

    print()


print("LEARNABLE SEASONAL RELATIVE TO FIXED SEASONAL")
print("Positive = learnable lower RMSE")
print()

for _, row in fixed_summary.iterrows():

    print(
        f"  L={int(row['depth'])}: "
        f"{row['mean']:.4f} ± "
        f"{row['std']:.4f} "
        f"({int(row['n_learnable_better'])}/5 seeds better)"
    )


print()
print("Generated figures:")

for stem in [
    "04_main_metrics_vs_depth",
    "05_depth_degradation_relative_to_L1",
    "07_amplitude_bias_vs_depth",
    "08_spectral_metrics_vs_depth",
    "09_normalized_band_power_ratios",
    "10_signed_band_errors",
    "12_paired_rmse_vs_ry",
    "13_learnable_vs_fixed_paired_rmse",
    "14_runtime_vs_depth",
    "15_parameter_count_vs_depth",
]:
    print(f"  {stem}.pdf")
    print(f"  {stem}.png")

print()
print("=" * 120)
print("EMPIRICAL FIGURE GENERATION STATUS: PASS")
print("=" * 120)
