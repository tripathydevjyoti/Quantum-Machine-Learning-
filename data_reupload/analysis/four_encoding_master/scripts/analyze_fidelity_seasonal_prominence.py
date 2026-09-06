#!/usr/bin/env python

from pathlib import Path
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


ROOT = _resolve_project_root()

ANALYSIS = (
    ROOT
    / "data_reupload/analysis/four_encoding_master"
)

INFILE = (
    ANALYSIS
    / "output/geometry/"
    "exact_temporal_fidelity_aggregate/"
    "g1_all_seed_lag_summary.csv"
)

OUTDIR = (
    ANALYSIS
    / "output/geometry/"
    "fidelity_seasonal_prominence"
)

FIGDIR = (
    ANALYSIS
    / "output/figures"
)

OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)


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


df = pd.read_csv(INFILE)

if len(df) != 150 * 40:
    raise RuntimeError(
        f"Unexpected row count: {len(df)}"
    )


# ============================================================
# Seasonal local-prominence definitions
# ============================================================

CENTERS = {
    12: [10, 11, 13, 14],
    24: [22, 23, 25, 26],
    36: [34, 35, 37, 38],
}

eps = 1e-300

rows = []

for (
    encoding,
    depth,
    seed,
), group in df.groupby(
    [
        "encoding",
        "depth",
        "seed",
    ]
):

    lookup = {
        int(row["tau"]):
            float(row["mean_fidelity"])
        for _, row in group.iterrows()
    }

    for center, neighbors in CENTERS.items():

        f_center = lookup[center]

        f_neighbors = np.asarray(
            [
                lookup[tau]
                for tau in neighbors
            ],
            dtype=np.float64,
        )

        log_center = np.log10(
            max(f_center, eps)
        )

        log_neighbors = np.log10(
            np.maximum(
                f_neighbors,
                eps,
            )
        )

        local_log_prominence = (
            log_center
            -
            np.mean(log_neighbors)
        )

        local_ratio = (
            10.0
            **
            local_log_prominence
        )

        arithmetic_ratio = (
            f_center
            /
            np.mean(f_neighbors)
        )

        rows.append({
            "encoding":
                encoding,

            "depth":
                int(depth),

            "seed":
                int(seed),

            "seasonal_lag":
                int(center),

            "center_fidelity":
                f_center,

            "neighbor_geometric_mean_fidelity":
                float(
                    10.0
                    **
                    np.mean(log_neighbors)
                ),

            "neighbor_arithmetic_mean_fidelity":
                float(
                    np.mean(f_neighbors)
                ),

            "log10_seasonal_prominence":
                float(
                    local_log_prominence
                ),

            "seasonal_prominence_ratio":
                float(
                    local_ratio
                ),

            "seasonal_prominence_arithmetic_ratio":
                float(
                    arithmetic_ratio
                ),
        })


seed_df = pd.DataFrame(rows)

summary = (
    seed_df
    .groupby(
        [
            "encoding",
            "depth",
            "seasonal_lag",
        ],
        as_index=False,
    )
    .agg(
        n_seeds=(
            "seed",
            "nunique",
        ),

        log10_prominence_mean=(
            "log10_seasonal_prominence",
            "mean",
        ),

        log10_prominence_std=(
            "log10_seasonal_prominence",
            "std",
        ),

        prominence_ratio_mean=(
            "seasonal_prominence_ratio",
            "mean",
        ),

        prominence_ratio_std=(
            "seasonal_prominence_ratio",
            "std",
        ),

        center_fidelity_mean=(
            "center_fidelity",
            "mean",
        ),

        center_fidelity_std=(
            "center_fidelity",
            "std",
        ),
    )
)


if not (
    summary["n_seeds"] == 5
).all():
    raise RuntimeError(
        "Expected five seeds for every summary."
    )


seed_df.to_csv(
    OUTDIR
    / "seasonal_prominence_seed_level.csv",
    index=False,
)

summary.to_csv(
    OUTDIR
    / "seasonal_prominence_depth_summary.csv",
    index=False,
)


# ============================================================
# Figure: local seasonal prominence
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(14.5, 4.6),
    sharey=True,
)

for ax, center in zip(
    axes,
    [12, 24, 36],
):

    for enc in ENCODINGS:

        d = (
            summary[
                (
                    summary["encoding"] == enc
                )
                &
                (
                    summary["seasonal_lag"]
                    == center
                )
            ]
            .sort_values("depth")
        )

        x = d["depth"].to_numpy()
        y = d[
            "log10_prominence_mean"
        ].to_numpy()
        sd = d[
            "log10_prominence_std"
        ].to_numpy()

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
            alpha=0.12,
            color=line.get_color(),
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    ax.set_xticks(
        range(1, 7)
    )

    ax.set_xlabel(
        "Reuploading depth $L$"
    )

    ax.set_title(
        rf"Seasonal lag $\tau={center}$"
    )

    ax.grid(
        alpha=0.17
    )


axes[0].set_ylabel(
    r"Local prominence "
    r"$\log_{10}[F(\tau)/F_{\rm neigh}]$"
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
    "Seasonal recurrence prominence in exact trained-state fidelity",
    y=1.10,
    fontsize=14,
)

fig.tight_layout()

fig.savefig(
    FIGDIR
    / "21_exact_fidelity_seasonal_prominence.pdf",
    bbox_inches="tight",
)

fig.savefig(
    FIGDIR
    / "21_exact_fidelity_seasonal_prominence.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Human-readable output
# ============================================================

print("=" * 120)
print("G1.5 SEASONAL FIDELITY PROMINENCE")
print("=" * 120)

for center in [12, 24, 36]:

    print()
    print("-" * 120)
    print(
        f"SEASONAL LAG tau={center}"
    )
    print("-" * 120)

    for enc in ENCODINGS:

        print()
        print(
            LABELS[enc]
        )

        d = (
            summary[
                (
                    summary["encoding"]
                    ==
                    enc
                )
                &
                (
                    summary["seasonal_lag"]
                    ==
                    center
                )
            ]
            .sort_values("depth")
        )

        for _, row in d.iterrows():

            print(
                f"L={int(row['depth'])}  "
                f"log10 prominence="
                f"{row['log10_prominence_mean']:.4f}"
                f" ± "
                f"{row['log10_prominence_std']:.4f}  "
                f"ratio="
                f"{row['prominence_ratio_mean']:.3f}"
                f" ± "
                f"{row['prominence_ratio_std']:.3f}"
            )


print()
print("=" * 120)
print("G1.5 STATUS: PASS")
print("=" * 120)
