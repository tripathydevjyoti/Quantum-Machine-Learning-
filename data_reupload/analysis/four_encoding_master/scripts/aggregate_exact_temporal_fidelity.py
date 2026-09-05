#!/usr/bin/env python

from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


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

TABLE_DIR = (
    ANALYSIS_ROOT
    / "output"
    / "tables"
)

FIG_DIR = (
    ANALYSIS_ROOT
    / "output"
    / "figures"
)

GEO_DIR = (
    ANALYSIS_ROOT
    / "output"
    / "geometry"
    / "exact_temporal_fidelity"
)

OUT_DIR = (
    ANALYSIS_ROOT
    / "output"
    / "geometry"
    / "exact_temporal_fidelity_aggregate"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ENCODINGS = [
    "ry",
    "same_scalar_rxryrz",
    "seasonal_meridian",
    "learnable_seasonal_cdf",
    "learnable_seasonal_cdf_rz",
]

LABELS = {
    "ry":
        r"Standard $R_y$",

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


manifest = (
    pd.read_csv(
        TABLE_DIR
        /
        "master_run_manifest.csv"
    )
    .sort_values(
        [
            "encoding_order",
            "depth",
            "seed",
        ]
    )
    .reset_index(
        drop=True
    )
)

if len(manifest) != 150:
    raise RuntimeError(
        "Expected 150 manifest rows."
    )


lag_frames = []
validation_rows = []


for _, row in manifest.iterrows():

    enc = str(
        row["encoding"]
    )

    depth = int(
        row["depth"]
    )

    seed = int(
        row["seed"]
    )

    run_out = (
        GEO_DIR
        /
        enc
        /
        f"depth_{depth}"
        /
        f"seed_{seed}"
    )

    lag_path = (
        run_out
        /
        "fidelity_lag_summary.csv"
    )

    val_path = (
        run_out
        /
        "run_validation.json"
    )

    pair_path = (
        run_out
        /
        "fidelity_pairs.csv.gz"
    )

    for p in [
        lag_path,
        val_path,
        pair_path,
    ]:

        if not p.exists():
            raise RuntimeError(
                f"Missing G1 output: {p}"
            )

    lag = pd.read_csv(
        lag_path
    )

    if len(lag) != 40:
        raise RuntimeError(
            f"Expected 40 lag rows: {lag_path}"
        )

    lag_frames.append(
        lag
    )

    with open(
        val_path
    ) as f:

        v = json.load(
            f
        )

    validation_rows.append(
        v
    )


all_seed_lag = pd.concat(
    lag_frames,
    ignore_index=True,
)

validation = pd.DataFrame(
    validation_rows
)


# ============================================================
# Validation
# ============================================================

if len(validation) != 150:
    raise RuntimeError(
        "Expected 150 validation rows."
    )

if not (
    validation["status"]
    ==
    "PASS"
).all():
    raise RuntimeError(
        "At least one G1 run failed validation."
    )

if (
    validation[
        "max_scaled_prediction_error"
    ].max()
    >
    1e-10
):
    raise RuntimeError(
        "Prediction reconstruction audit failed."
    )

if (
    validation[
        "max_state_norm_error"
    ].max()
    >
    1e-10
):
    raise RuntimeError(
        "State normalization audit failed."
    )


# ============================================================
# Seed-level tau=12
# ============================================================

tau12_seed = (
    all_seed_lag[
        all_seed_lag["tau"]
        ==
        12
    ]
    .copy()
    .sort_values(
        [
            "encoding",
            "depth",
            "seed",
        ]
    )
)

if len(tau12_seed) != 150:
    raise RuntimeError(
        f"Expected 150 tau=12 rows, "
        f"got {len(tau12_seed)}"
    )


# ============================================================
# Five-seed depth-lag aggregation
#
# IMPORTANT:
# First summarize time within each trained seed.
# Then mean ± sample SD across the five seeds.
# ============================================================

depth_lag = (
    all_seed_lag
    .groupby(
        [
            "encoding",
            "depth",
            "tau",
        ],
        as_index=False,
    )
    .agg(
        n_seeds=(
            "seed",
            "nunique",
        ),

        mean_fidelity=(
            "mean_fidelity",
            "mean",
        ),

        sd_fidelity_across_seeds=(
            "mean_fidelity",
            "std",
        ),

        mean_median_fidelity=(
            "median_fidelity",
            "mean",
        ),

        sd_median_fidelity_across_seeds=(
            "median_fidelity",
            "std",
        ),

        mean_geometric_fidelity=(
            "geometric_mean_fidelity",
            "mean",
        ),

        mean_fidelity_over_haar=(
            "mean_fidelity_over_haar",
            "mean",
        ),

        sd_fidelity_over_haar_across_seeds=(
            "mean_fidelity_over_haar",
            "std",
        ),
    )
)

if not (
    depth_lag["n_seeds"]
    ==
    5
).all():
    raise RuntimeError(
        "Expected exactly five seeds everywhere."
    )


tau12_depth = (
    depth_lag[
        depth_lag["tau"]
        ==
        12
    ]
    .copy()
    .sort_values(
        [
            "encoding",
            "depth",
        ]
    )
)

if len(tau12_depth) != 30:
    raise RuntimeError(
        f"Expected 30 tau=12 depth rows, "
        f"got {len(tau12_depth)}"
    )


# ============================================================
# Merge tau=12 geometry with forecasting summary
# ============================================================

forecast = pd.read_csv(
    TABLE_DIR
    /
    "forecasting_depth_summary.csv"
)

geometry_forecast = forecast.merge(
    tau12_depth[
        [
            "encoding",
            "depth",
            "mean_fidelity",
            "sd_fidelity_across_seeds",
            "mean_median_fidelity",
            "mean_geometric_fidelity",
            "mean_fidelity_over_haar",
            "sd_fidelity_over_haar_across_seeds",
        ]
    ],
    on=[
        "encoding",
        "depth",
    ],
    how="inner",
    validate="one_to_one",
)

if len(
    geometry_forecast
) != 30:
    raise RuntimeError(
        "Geometry/forecast merge mismatch."
    )


# ============================================================
# Save tables
# ============================================================

all_seed_lag.to_csv(
    OUT_DIR
    /
    "g1_all_seed_lag_summary.csv",
    index=False,
)

validation.to_csv(
    OUT_DIR
    /
    "g1_all_run_validation.csv",
    index=False,
)

tau12_seed.to_csv(
    OUT_DIR
    /
    "g1_tau12_seed_summary.csv",
    index=False,
)

depth_lag.to_csv(
    OUT_DIR
    /
    "g1_depth_lag_mean_sd.csv",
    index=False,
)

tau12_depth.to_csv(
    OUT_DIR
    /
    "g1_tau12_depth_summary.csv",
    index=False,
)

geometry_forecast.to_csv(
    OUT_DIR
    /
    "g1_tau12_geometry_forecasting_summary.csv",
    index=False,
)


# ============================================================
# Figure 20: exact same-season fidelity vs depth
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.0, 4.8),
)

for enc in ENCODINGS:

    d = (
        tau12_depth[
            tau12_depth[
                "encoding"
            ]
            ==
            enc
        ]
        .sort_values(
            "depth"
        )
    )

    x = d[
        "depth"
    ].to_numpy()

    y = d[
        "mean_fidelity"
    ].to_numpy()

    sd = d[
        "sd_fidelity_across_seeds"
    ].to_numpy()

    line, = axes[0].plot(
        x,
        y,
        marker=MARKERS[enc],
        linewidth=1.8,
        markersize=5,
        label=LABELS[enc],
    )

    axes[0].fill_between(
        x,
        np.maximum(
            y - sd,
            0,
        ),
        y + sd,
        alpha=0.13,
        color=line.get_color(),
    )

    y_rel = d[
        "mean_fidelity_over_haar"
    ].to_numpy()

    sd_rel = d[
        "sd_fidelity_over_haar_across_seeds"
    ].to_numpy()

    line2, = axes[1].plot(
        x,
        y_rel,
        marker=MARKERS[enc],
        linewidth=1.8,
        markersize=5,
        label=LABELS[enc],
    )

    axes[1].fill_between(
        x,
        np.maximum(
            y_rel - sd_rel,
            1e-12,
        ),
        y_rel + sd_rel,
        alpha=0.13,
        color=line2.get_color(),
    )


axes[0].set_xlabel(
    "Reuploading depth $L$"
)

axes[0].set_ylabel(
    r"Mean exact fidelity $\overline{F}_L(12)$"
)

axes[0].set_xticks(
    range(1, 7)
)

axes[0].grid(
    alpha=0.18
)

axes[0].set_title(
    "Same-season state fidelity"
)


axes[1].axhline(
    1.0,
    linestyle="--",
    linewidth=1.0,
    alpha=0.6,
)

axes[1].set_yscale(
    "log"
)

axes[1].set_xlabel(
    "Reuploading depth $L$"
)

axes[1].set_ylabel(
    r"$\overline{F}_L(12)/(1/2^{14})$"
)

axes[1].set_xticks(
    range(1, 7)
)

axes[1].grid(
    alpha=0.18
)

axes[1].set_title(
    "Relative to Haar mean overlap"
)


handles, labels = (
    axes[0]
    .get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.04),
)

fig.suptitle(
    "Exact trained-state similarity for windows separated by 12 steps",
    y=1.10,
    fontsize=14,
)

fig.tight_layout()

fig.savefig(
    FIG_DIR
    /
    "20_exact_seasonal_fidelity_vs_depth.pdf",
    bbox_inches="tight",
)

fig.savefig(
    FIG_DIR
    /
    "20_exact_seasonal_fidelity_vs_depth.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Figure 23: lag-dependent exact fidelity
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12.5, 10.8),
    sharex=True,
)

for depth, ax in zip(
    range(1, 7),
    axes.flat,
):

    for enc in ENCODINGS:

        d = (
            depth_lag[
                (
                    depth_lag[
                        "encoding"
                    ]
                    ==
                    enc
                )
                &
                (
                    depth_lag[
                        "depth"
                    ]
                    ==
                    depth
                )
            ]
            .sort_values(
                "tau"
            )
        )

        x = d[
            "tau"
        ].to_numpy()

        y = d[
            "mean_fidelity"
        ].to_numpy()

        sd = d[
            "sd_fidelity_across_seeds"
        ].to_numpy()

        line, = ax.plot(
            x,
            y,
            linewidth=1.4,
            label=LABELS[enc],
        )

        ax.fill_between(
            x,
            np.maximum(
                y - sd,
                0,
            ),
            y + sd,
            alpha=0.08,
            color=line.get_color(),
        )

    for tau in [
        12,
        24,
        36,
    ]:

        ax.axvline(
            tau,
            linestyle="--",
            linewidth=0.8,
            alpha=0.35,
        )

    ax.set_title(
        f"Depth {depth}"
    )

    ax.grid(
        alpha=0.15
    )


for ax in axes[:, 0]:

    ax.set_ylabel(
        "Mean exact state fidelity"
    )


for ax in axes[-1, :]:

    ax.set_xlabel(
        r"Temporal lag $\tau$"
    )


handles, labels = (
    axes[0, 0]
    .get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.995),
)

fig.suptitle(
    "Lag-dependent exact temporal fidelity of trained quantum states",
    y=1.025,
    fontsize=14,
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.955,
    ]
)

fig.savefig(
    FIG_DIR
    /
    "23_exact_fidelity_vs_temporal_lag.pdf",
    bbox_inches="tight",
)

fig.savefig(
    FIG_DIR
    /
    "23_exact_fidelity_vs_temporal_lag.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Log-scale lag plot relative to Haar
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12.5, 10.8),
    sharex=True,
    sharey=True,
)

for depth, ax in zip(
    range(1, 7),
    axes.flat,
):

    for enc in ENCODINGS:

        d = (
            depth_lag[
                (
                    depth_lag[
                        "encoding"
                    ]
                    ==
                    enc
                )
                &
                (
                    depth_lag[
                        "depth"
                    ]
                    ==
                    depth
                )
            ]
            .sort_values(
                "tau"
            )
        )

        ax.plot(
            d["tau"],
            d[
                "mean_fidelity_over_haar"
            ],
            linewidth=1.4,
            label=LABELS[enc],
        )

    ax.axhline(
        1.0,
        linestyle=":",
        linewidth=1.0,
        alpha=0.5,
    )

    for tau in [
        12,
        24,
        36,
    ]:

        ax.axvline(
            tau,
            linestyle="--",
            linewidth=0.8,
            alpha=0.30,
        )

    ax.set_yscale(
        "log"
    )

    ax.set_title(
        f"Depth {depth}"
    )

    ax.grid(
        alpha=0.15
    )


for ax in axes[:, 0]:

    ax.set_ylabel(
        "Mean fidelity / Haar mean"
    )


for ax in axes[-1, :]:

    ax.set_xlabel(
        r"Temporal lag $\tau$"
    )


handles, labels = (
    axes[0, 0]
    .get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.995),
)

fig.suptitle(
    "Temporal state fidelity relative to random-state overlap scale",
    y=1.025,
    fontsize=14,
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.955,
    ]
)

fig.savefig(
    FIG_DIR
    /
    "23b_exact_fidelity_over_haar_vs_temporal_lag.pdf",
    bbox_inches="tight",
)

fig.savefig(
    FIG_DIR
    /
    "23b_exact_fidelity_over_haar_vs_temporal_lag.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Human-readable output
# ============================================================

print("=" * 120)
print("G1 EXACT TEMPORAL FIDELITY AGGREGATION")
print("=" * 120)

print()
print(
    "Validated runs:",
    len(validation),
)

print(
    "Max state norm error:",
    validation[
        "max_state_norm_error"
    ].max(),
)

print(
    "Max scaled prediction reconstruction error:",
    validation[
        "max_scaled_prediction_error"
    ].max(),
)

print()
print("-" * 120)
print("TAU = 12 FIVE-SEED SUMMARY")
print("-" * 120)

for enc in ENCODINGS:

    print()
    print(
        LABELS[enc]
    )

    d = (
        tau12_depth[
            tau12_depth[
                "encoding"
            ]
            ==
            enc
        ]
        .sort_values(
            "depth"
        )
    )

    for _, row in d.iterrows():

        print(
            f"L={int(row['depth'])}  "
            f"F12="
            f"{row['mean_fidelity']:.8e}"
            f" ± "
            f"{row['sd_fidelity_across_seeds']:.8e}  "
            f"medianF(mean)="
            f"{row['mean_median_fidelity']:.8e}  "
            f"F/Haar="
            f"{row['mean_fidelity_over_haar']:.3f}"
        )


print()
print("Generated:")
print(
    "  20_exact_seasonal_fidelity_vs_depth.pdf"
)
print(
    "  23_exact_fidelity_vs_temporal_lag.pdf"
)
print(
    "  23b_exact_fidelity_over_haar_vs_temporal_lag.pdf"
)

print()
print("=" * 120)
print("G1 AGGREGATION STATUS: PASS")
print("=" * 120)
