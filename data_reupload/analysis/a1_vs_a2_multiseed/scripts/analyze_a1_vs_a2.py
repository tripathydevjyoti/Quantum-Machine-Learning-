from __future__ import annotations

from itertools import product
from pathlib import Path
import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# Paths
# =====================================================================

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


REPO = _resolve_project_root()

RESULTS_ROOT = (
    REPO
    / "data_reupload/results/final_model_sweep"
)

OUT_ROOT = (
    REPO
    / "data_reupload/analysis/a1_vs_a2_multiseed/output"
)

FIG_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


FAMILIES = {
    "a1": "single_tanh_a1",
    "a2": "single_tanh_a2",
}

DEPTHS = [1, 2, 3, 4, 5, 6]
SEEDS = [42, 43, 44, 45, 46]


# =====================================================================
# Helpers
# =====================================================================

def run_dir(
    family: str,
    depth: int,
    seed: int,
) -> Path:

    return (
        RESULTS_ROOT
        / FAMILIES[family]
        / f"depth_{depth}"
        / f"seed_{seed}"
    )


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def read_summary(
    family: str,
    depth: int,
    seed: int,
) -> pd.Series:

    path = (
        run_dir(family, depth, seed)
        / "result_summary.csv"
    )

    require_file(path)

    df = pd.read_csv(path)

    if len(df) != 1:
        raise ValueError(
            f"Expected exactly one row in {path}; "
            f"found {len(df)}"
        )

    return df.iloc[0]


def read_history(
    family: str,
    depth: int,
    seed: int,
) -> pd.DataFrame:

    path = (
        run_dir(family, depth, seed)
        / "history.csv"
    )

    require_file(path)

    return pd.read_csv(path)


def read_predictions(
    family: str,
    depth: int,
    seed: int,
) -> dict[str, np.ndarray]:

    path = (
        run_dir(family, depth, seed)
        / "predictions.npz"
    )

    require_file(path)

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        return {
            key: np.asarray(data[key])
            for key in data.files
        }


def first_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str:

    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        f"Could not find any of {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def first_key(
    data: dict[str, np.ndarray],
    candidates: list[str],
) -> str:

    for c in candidates:
        if c in data:
            return c

    raise KeyError(
        f"Could not find any of {candidates}\n"
        f"Available keys:\n{list(data.keys())}"
    )


# =====================================================================
# Verify complete balanced experiment
# =====================================================================

print("=" * 100)
print("VERIFYING BALANCED A1 / A2 EXPERIMENT")
print("=" * 100)

for family in FAMILIES:

    for depth in DEPTHS:

        for seed in SEEDS:

            rd = run_dir(
                family,
                depth,
                seed,
            )

            for filename in [
                "result_summary.csv",
                "history.csv",
                "predictions.npz",
                "learned_parameters.npz",
                "config.json",
            ]:
                require_file(
                    rd / filename
                )

    print(
        f"Verified {family}: "
        f"6 depths x 5 seeds = 30 runs"
    )

print(
    "Verified total balanced comparison: "
    "60 ideal QNN runs"
)
print()


# =====================================================================
# Load all summary metrics
# =====================================================================

rows = []

for family in FAMILIES:

    for depth in DEPTHS:

        for seed in SEEDS:

            row = read_summary(
                family,
                depth,
                seed,
            ).to_dict()

            row["family"] = family
            row["depth"] = depth
            row["seed"] = seed

            rows.append(row)


runs = pd.DataFrame(rows)


# Convert numeric-looking columns.
for col in runs.columns:

    if col in [
        "family",
        "variant_id",
        "model_family",
    ]:
        continue

    converted = pd.to_numeric(
        runs[col],
        errors="coerce",
    )

    if converted.notna().any():
        runs[col] = converted


runs.to_csv(
    TABLE_DIR
    / "a1_vs_a2_all_run_metrics.csv",
    index=False,
)


# =====================================================================
# Metrics
# =====================================================================

METRICS = [
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
    "test_abs_short_period_2_to_6_error",

    "elapsed_time_sec",
    "max_rss_gb_python",
]


available_metrics = [
    m
    for m in METRICS
    if m in runs.columns
]


print("Available metrics:")
for m in available_metrics:
    print(" ", m)
print()


# =====================================================================
# Aggregate mean / SD
# =====================================================================

agg = (
    runs
    .groupby(
        ["family", "depth"]
    )[available_metrics]
    .agg(["mean", "std"])
)

agg.columns = [
    f"{metric}_{stat}"
    for metric, stat in agg.columns
]

agg = agg.reset_index()


agg.to_csv(
    TABLE_DIR
    / "a1_vs_a2_depth_mean_sd.csv",
    index=False,
)


# Family-specific tables
for family in FAMILIES:

    family_agg = (
        agg[
            agg["family"] == family
        ]
        .copy()
        .reset_index(drop=True)
    )

    family_agg.to_csv(
        TABLE_DIR
        / f"{family}_depth_mean_sd.csv",
        index=False,
    )


# =====================================================================
# Pretty mean ± SD table
# =====================================================================

def pm(
    mean: float,
    sd: float,
) -> str:

    if pd.isna(mean):
        return ""

    if pd.isna(sd):
        return f"{mean:.3f}"

    return (
        f"{mean:.3f} ± {sd:.3f}"
    )


pretty_rows = []

for _, r in agg.iterrows():

    out = {
        "Family": r["family"],
        "Depth": int(r["depth"]),
    }

    for metric in available_metrics:

        out[metric] = pm(
            r[f"{metric}_mean"],
            r[f"{metric}_std"],
        )

    pretty_rows.append(out)


pretty = pd.DataFrame(
    pretty_rows
)

pretty.to_csv(
    TABLE_DIR
    / "a1_vs_a2_depth_mean_sd_pretty.csv",
    index=False,
)


with open(
    TABLE_DIR
    / "a1_vs_a2_depth_mean_sd.tex",
    "w",
) as f:

    f.write(
        pretty.to_latex(
            index=False,
            escape=True,
            longtable=True,
        )
    )


# =====================================================================
# Paired seedwise differences
#
# Definition:
#     difference = A2 - A1
#
# Thus:
#   negative RMSE/MAE/etc. -> A2 better
#   positive correlation/KGE/etc. -> A2 better
# =====================================================================

paired_rows = []

for depth in DEPTHS:

    for seed in SEEDS:

        a1 = runs[
            (runs["family"] == "a1")
            &
            (runs["depth"] == depth)
            &
            (runs["seed"] == seed)
        ].iloc[0]

        a2 = runs[
            (runs["family"] == "a2")
            &
            (runs["depth"] == depth)
            &
            (runs["seed"] == seed)
        ].iloc[0]

        row = {
            "depth": depth,
            "seed": seed,
        }

        for metric in available_metrics:

            row[f"a1_{metric}"] = (
                a1[metric]
            )

            row[f"a2_{metric}"] = (
                a2[metric]
            )

            row[f"diff_a2_minus_a1_{metric}"] = (
                a2[metric]
                -
                a1[metric]
            )

        paired_rows.append(row)


paired = pd.DataFrame(
    paired_rows
)

paired.to_csv(
    TABLE_DIR
    / "a1_vs_a2_paired_seed_differences.csv",
    index=False,
)


# =====================================================================
# Exact sign-flip test
# =====================================================================

def exact_sign_flip_p(
    differences: np.ndarray,
) -> float:

    d = np.asarray(
        differences,
        dtype=float,
    )

    d = d[np.isfinite(d)]

    n = len(d)

    if n == 0:
        return np.nan

    observed = abs(
        np.mean(d)
    )

    abs_d = np.abs(d)

    stats = []

    for signs in product(
        [-1.0, 1.0],
        repeat=n,
    ):

        signs_arr = np.asarray(
            signs,
            dtype=float,
        )

        stat = abs(
            np.mean(
                signs_arr
                *
                abs_d
            )
        )

        stats.append(stat)

    stats = np.asarray(stats)

    return float(
        np.mean(
            stats
            >=
            observed - 1e-15
        )
    )


LOWER_IS_BETTER = {
    "val_rmse_original",
    "test_rmse_original",
    "test_mae_original",
    "test_abs_12_step_band_error",
    "test_abs_short_period_2_to_6_error",
    "elapsed_time_sec",
    "max_rss_gb_python",
}

HIGHER_IS_BETTER = {
    "test_pearson_corr",
    "test_spearman_corr",
    "test_kge",
    "test_nse",
    "test_fft_spectral_cosine_similarity",
}


stats_rows = []

for depth in DEPTHS:

    subset = paired[
        paired["depth"] == depth
    ]

    for metric in available_metrics:

        diff_col = (
            f"diff_a2_minus_a1_{metric}"
        )

        d = (
            subset[diff_col]
            .to_numpy(dtype=float)
        )

        d = d[
            np.isfinite(d)
        ]

        mean_diff = np.mean(d)
        sd_diff = (
            np.std(d, ddof=1)
            if len(d) > 1
            else np.nan
        )

        if (
            np.isfinite(sd_diff)
            and sd_diff > 0
        ):
            cohens_dz = (
                mean_diff
                /
                sd_diff
            )
        else:
            cohens_dz = np.nan

        if metric in LOWER_IS_BETTER:

            interpretation = (
                "A2 better"
                if mean_diff < 0
                else "A1 better"
            )

        elif metric in HIGHER_IS_BETTER:

            interpretation = (
                "A2 better"
                if mean_diff > 0
                else "A1 better"
            )

        else:
            interpretation = (
                "direction depends on target"
            )

        stats_rows.append(
            {
                "depth": depth,
                "metric": metric,
                "n_pairs": len(d),
                "mean_a2_minus_a1": mean_diff,
                "sd_paired_difference": sd_diff,
                "cohens_dz": cohens_dz,
                "exact_two_sided_sign_flip_p":
                    exact_sign_flip_p(d),
                "mean_direction":
                    interpretation,
            }
        )


paired_stats = pd.DataFrame(
    stats_rows
)

paired_stats.to_csv(
    TABLE_DIR
    / "a1_vs_a2_paired_statistics.csv",
    index=False,
)


# =====================================================================
# Figure 1:
# Main metrics versus depth
# =====================================================================

MAIN_METRICS = [
    (
        "test_rmse_original",
        "Test RMSE",
    ),
    (
        "test_mae_original",
        "Test MAE",
    ),
    (
        "test_pearson_corr",
        "Pearson correlation",
    ),
    (
        "test_spearman_corr",
        "Spearman correlation",
    ),
    (
        "test_kge",
        "KGE",
    ),
    (
        "test_nse",
        "NSE",
    ),
]


fig, axes = plt.subplots(
    2,
    3,
    figsize=(13.5, 8.0),
    constrained_layout=True,
)

axes = axes.ravel()


for ax, (
    metric,
    title,
) in zip(
    axes,
    MAIN_METRICS,
):

    for family, label in [
        ("a1", "1 ansatz layer"),
        ("a2", "2 ansatz layers"),
    ]:

        subset = (
            agg[
                agg["family"]
                ==
                family
            ]
            .sort_values("depth")
        )

        x = subset[
            "depth"
        ].to_numpy(dtype=float)

        mean = subset[
            f"{metric}_mean"
        ].to_numpy(dtype=float)

        sd = subset[
            f"{metric}_std"
        ].to_numpy(dtype=float)

        line, = ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2,
            label=label,
        )

        ax.fill_between(
            x,
            mean - sd,
            mean + sd,
            alpha=0.20,
            color=line.get_color(),
        )

    ax.set_title(title)
    ax.set_xlabel(
        "Reuploading depth"
    )
    ax.set_xticks(DEPTHS)
    ax.grid(alpha=0.25)


axes[0].legend(
    frameon=False,
)

fig.suptitle(
    "Effect of ansatz-layer count on ideal QNN performance\n"
    "Mean ± 1 SD across seeds 42–46",
    fontsize=14,
)


for ext in ["pdf", "png"]:

    fig.savefig(
        FIG_DIR
        / f"01_a1_vs_a2_metrics_vs_depth.{ext}",
        dpi=300 if ext == "png" else None,
        bbox_inches="tight",
    )

plt.close(fig)


# =====================================================================
# Figure 2:
# Spectral metrics
# =====================================================================

SPECTRAL = [
    (
        "test_fft_spectral_cosine_similarity",
        "FFT spectral cosine similarity",
    ),
    (
        "test_abs_12_step_band_error",
        "Absolute 12-step band error",
    ),
    (
        "test_abs_short_period_2_to_6_error",
        "Absolute 2–6 step band error",
    ),
]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(14.5, 4.5),
    constrained_layout=True,
)


for ax, (
    metric,
    title,
) in zip(
    axes,
    SPECTRAL,
):

    for family, label in [
        ("a1", "1 ansatz layer"),
        ("a2", "2 ansatz layers"),
    ]:

        subset = (
            agg[
                agg["family"]
                ==
                family
            ]
            .sort_values("depth")
        )

        x = subset[
            "depth"
        ].to_numpy(dtype=float)

        mean = subset[
            f"{metric}_mean"
        ].to_numpy(dtype=float)

        sd = subset[
            f"{metric}_std"
        ].to_numpy(dtype=float)

        line, = ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2,
            label=label,
        )

        ax.fill_between(
            x,
            mean - sd,
            mean + sd,
            alpha=0.20,
            color=line.get_color(),
        )

    ax.set_title(title)
    ax.set_xlabel(
        "Reuploading depth"
    )
    ax.set_xticks(DEPTHS)
    ax.grid(alpha=0.25)


axes[0].legend(
    frameon=False,
)

fig.suptitle(
    "Spectral behavior versus ansatz-layer count",
    fontsize=14,
)


for ext in ["pdf", "png"]:

    fig.savefig(
        FIG_DIR
        / f"02_a1_vs_a2_spectral_metrics.{ext}",
        dpi=300 if ext == "png" else None,
        bbox_inches="tight",
    )

plt.close(fig)


# =====================================================================
# Figure 3:
# Runtime
# =====================================================================

fig, ax = plt.subplots(
    figsize=(8.0, 5.2),
    constrained_layout=True,
)

for family, label in [
    ("a1", "1 ansatz layer"),
    ("a2", "2 ansatz layers"),
]:

    subset = (
        agg[
            agg["family"]
            ==
            family
        ]
        .sort_values("depth")
    )

    x = subset[
        "depth"
    ].to_numpy(dtype=float)

    mean = (
        subset[
            "elapsed_time_sec_mean"
        ].to_numpy(dtype=float)
        /
        60.0
    )

    sd = (
        subset[
            "elapsed_time_sec_std"
        ].to_numpy(dtype=float)
        /
        60.0
    )

    line, = ax.plot(
        x,
        mean,
        marker="o",
        linewidth=2,
        label=label,
    )

    ax.fill_between(
        x,
        mean - sd,
        mean + sd,
        alpha=0.20,
        color=line.get_color(),
    )


ax.set_xlabel(
    "Reuploading depth"
)

ax.set_ylabel(
    "Runtime [min]"
)

ax.set_xticks(
    DEPTHS
)

ax.grid(
    alpha=0.25
)

ax.legend(
    frameon=False
)

ax.set_title(
    "Training runtime: "
    "1 versus 2 ansatz layers"
)


for ext in ["pdf", "png"]:

    fig.savefig(
        FIG_DIR
        / f"03_a1_vs_a2_runtime.{ext}",
        dpi=300 if ext == "png" else None,
        bbox_inches="tight",
    )

plt.close(fig)


# =====================================================================
# Figure 4:
# Memory
# =====================================================================

fig, ax = plt.subplots(
    figsize=(8.0, 5.2),
    constrained_layout=True,
)

for family, label in [
    ("a1", "1 ansatz layer"),
    ("a2", "2 ansatz layers"),
]:

    subset = (
        agg[
            agg["family"]
            ==
            family
        ]
        .sort_values("depth")
    )

    x = subset[
        "depth"
    ].to_numpy(dtype=float)

    mean = subset[
        "max_rss_gb_python_mean"
    ].to_numpy(dtype=float)

    sd = subset[
        "max_rss_gb_python_std"
    ].to_numpy(dtype=float)

    line, = ax.plot(
        x,
        mean,
        marker="o",
        linewidth=2,
        label=label,
    )

    ax.fill_between(
        x,
        mean - sd,
        mean + sd,
        alpha=0.20,
        color=line.get_color(),
    )


ax.set_xlabel(
    "Reuploading depth"
)

ax.set_ylabel(
    "Maximum resident memory [GB]"
)

ax.set_xticks(
    DEPTHS
)

ax.grid(
    alpha=0.25
)

ax.legend(
    frameon=False
)

ax.set_title(
    "Memory use: "
    "1 versus 2 ansatz layers"
)


for ext in ["pdf", "png"]:

    fig.savefig(
        FIG_DIR
        / f"04_a1_vs_a2_memory.{ext}",
        dpi=300 if ext == "png" else None,
        bbox_inches="tight",
    )

plt.close(fig)


# =====================================================================
# History aggregation
# =====================================================================

EPOCH_CANDIDATES = [
    "epoch",
    "Epoch",
]

TRAIN_CANDIDATES = [
    "train_loss_scaled",
    "train_loss",
]

VAL_CANDIDATES = [
    "val_loss_scaled",
    "val_loss",
]


def aggregate_history(
    family: str,
    depth: int,
) -> pd.DataFrame:

    frames = []

    for seed in SEEDS:

        h = read_history(
            family,
            depth,
            seed,
        )

        epoch_col = first_column(
            h,
            EPOCH_CANDIDATES,
        )

        train_col = first_column(
            h,
            TRAIN_CANDIDATES,
        )

        val_col = first_column(
            h,
            VAL_CANDIDATES,
        )

        frames.append(
            pd.DataFrame(
                {
                    "epoch":
                        pd.to_numeric(
                            h[epoch_col],
                            errors="coerce",
                        ),

                    "train_loss":
                        pd.to_numeric(
                            h[train_col],
                            errors="coerce",
                        ),

                    "val_loss":
                        pd.to_numeric(
                            h[val_col],
                            errors="coerce",
                        ),

                    "seed":
                        seed,
                }
            )
        )

    joined = pd.concat(
        frames,
        ignore_index=True,
    )

    return (
        joined
        .groupby(
            "epoch",
            as_index=False,
        )
        .agg(
            train_mean=(
                "train_loss",
                "mean",
            ),
            train_sd=(
                "train_loss",
                "std",
            ),
            val_mean=(
                "val_loss",
                "mean",
            ),
            val_sd=(
                "val_loss",
                "std",
            ),
        )
    )


# =====================================================================
# Figure 5:
# A2 training curves for all depths
# =====================================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12.5, 12.0),
    constrained_layout=True,
)

axes = axes.ravel()


for ax, depth in zip(
    axes,
    DEPTHS,
):

    h = aggregate_history(
        "a2",
        depth,
    )

    x = h[
        "epoch"
    ].to_numpy(dtype=float)

    tm = h[
        "train_mean"
    ].to_numpy(dtype=float)

    ts = h[
        "train_sd"
    ].to_numpy(dtype=float)

    vm = h[
        "val_mean"
    ].to_numpy(dtype=float)

    vs = h[
        "val_sd"
    ].to_numpy(dtype=float)


    line_t, = ax.plot(
        x,
        tm,
        linewidth=1.8,
        label="Training",
    )

    ax.fill_between(
        x,
        tm - ts,
        tm + ts,
        alpha=0.18,
        color=line_t.get_color(),
    )


    line_v, = ax.plot(
        x,
        vm,
        linewidth=1.8,
        label="Validation",
    )

    ax.fill_between(
        x,
        vm - vs,
        vm + vs,
        alpha=0.18,
        color=line_v.get_color(),
    )


    ax.set_title(
        f"Depth {depth}"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Scaled MSE"
    )

    ax.grid(
        alpha=0.22
    )


axes[0].legend(
    frameon=False,
)

fig.suptitle(
    "Two-ansatz-layer QNN training histories\n"
    "Mean ± 1 SD across seeds 42–46",
    fontsize=14,
)


for ext in ["pdf", "png"]:

    fig.savefig(
        FIG_DIR
        / f"05_a2_training_curves.{ext}",
        dpi=300 if ext == "png" else None,
        bbox_inches="tight",
    )

plt.close(fig)


# =====================================================================
# Prediction helpers
# =====================================================================

TRUE_KEYS = [
    "y_test_true_original",
    "test_true_original",
]

PRED_KEYS = [
    "y_test_pred_original",
    "test_pred_original",
]

INDEX_KEYS = [
    "test_target_indices",
    "y_test_target_indices",
]


def aggregate_predictions(
    family: str,
    depth: int,
):

    predictions = []

    truth_ref = None
    index_ref = None

    for seed in SEEDS:

        data = read_predictions(
            family,
            depth,
            seed,
        )

        true_key = first_key(
            data,
            TRUE_KEYS,
        )

        pred_key = first_key(
            data,
            PRED_KEYS,
        )

        truth = np.asarray(
            data[true_key],
            dtype=float,
        ).reshape(-1)

        pred = np.asarray(
            data[pred_key],
            dtype=float,
        ).reshape(-1)


        if truth_ref is None:

            truth_ref = (
                truth.copy()
            )

        elif not np.allclose(
            truth_ref,
            truth,
            atol=1e-10,
            rtol=0,
        ):

            raise ValueError(
                f"Truth mismatch: "
                f"{family}, depth={depth}"
            )


        if index_ref is None:

            for key in INDEX_KEYS:

                if key in data:

                    index_ref = (
                        np.asarray(
                            data[key]
                        )
                        .reshape(-1)
                    )

                    break


        predictions.append(
            pred
        )


    matrix = np.stack(
        predictions,
        axis=0,
    )


    if index_ref is None:

        index_ref = np.arange(
            len(truth_ref)
        )


    return (
        index_ref,
        truth_ref,
        matrix.mean(axis=0),
        matrix.std(
            axis=0,
            ddof=1,
        ),
    )


# =====================================================================
# Figure 6:
# A1 vs A2 prediction trajectories
# =====================================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(14.0, 11.5),
    sharex=True,
    constrained_layout=True,
)

axes = axes.ravel()


for ax, depth in zip(
    axes,
    DEPTHS,
):

    x1, truth1, m1, s1 = (
        aggregate_predictions(
            "a1",
            depth,
        )
    )

    x2, truth2, m2, s2 = (
        aggregate_predictions(
            "a2",
            depth,
        )
    )


    if not np.allclose(
        truth1,
        truth2,
        atol=1e-10,
        rtol=0,
    ):

        raise ValueError(
            f"A1/A2 truth mismatch "
            f"at depth {depth}"
        )


    ax.plot(
        x1,
        truth1,
        linewidth=1.7,
        label="True",
    )


    line1, = ax.plot(
        x1,
        m1,
        linewidth=1.7,
        label="1 ansatz layer",
    )

    ax.fill_between(
        x1,
        m1 - s1,
        m1 + s1,
        alpha=0.17,
        color=line1.get_color(),
    )


    line2, = ax.plot(
        x2,
        m2,
        linewidth=1.7,
        label="2 ansatz layers",
    )

    ax.fill_between(
        x2,
        m2 - s2,
        m2 + s2,
        alpha=0.17,
        color=line2.get_color(),
    )


    ax.set_title(
        f"Depth {depth}"
    )

    ax.set_ylabel(
        "Target"
    )

    ax.grid(
        alpha=0.20
    )


axes[0].legend(
    frameon=False,
    ncol=3,
)

axes[-1].set_xlabel(
    "Test target index"
)

axes[-2].set_xlabel(
    "Test target index"
)


fig.suptitle(
    "Ideal test predictions: "
    "one versus two ansatz layers\n"
    "Mean ± 1 SD across seeds 42–46",
    fontsize=14,
)


for ext in ["pdf", "png"]:

    fig.savefig(
        FIG_DIR
        / f"06_a1_vs_a2_test_predictions.{ext}",
        dpi=300 if ext == "png" else None,
        bbox_inches="tight",
    )

plt.close(fig)


# =====================================================================
# Figure 7:
# Paired RMSE seed robustness
# =====================================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(11.5, 11.0),
    constrained_layout=True,
)

axes = axes.ravel()


for ax, depth in zip(
    axes,
    DEPTHS,
):

    subset = (
        paired[
            paired["depth"]
            ==
            depth
        ]
        .sort_values("seed")
    )


    for _, row in subset.iterrows():

        ax.plot(
            [1, 2],
            [
                row[
                    "a1_test_rmse_original"
                ],
                row[
                    "a2_test_rmse_original"
                ],
            ],
            marker="o",
            linewidth=1.2,
            alpha=0.75,
        )


    a1_mean = subset[
        "a1_test_rmse_original"
    ].mean()

    a2_mean = subset[
        "a2_test_rmse_original"
    ].mean()


    ax.plot(
        [1, 2],
        [a1_mean, a2_mean],
        marker="s",
        linewidth=3.0,
        label="Seed mean",
    )


    ax.set_xticks(
        [1, 2]
    )

    ax.set_xticklabels(
        ["A1", "A2"]
    )

    ax.set_ylabel(
        "Test RMSE"
    )

    ax.set_title(
        f"Depth {depth}"
    )

    ax.grid(
        alpha=0.20
    )


axes[0].legend(
    frameon=False,
)


fig.suptitle(
    "Paired seedwise RMSE: "
    "A1 versus A2",
    fontsize=14,
)


for ext in ["pdf", "png"]:

    fig.savefig(
        FIG_DIR
        / f"07_a1_vs_a2_paired_rmse.{ext}",
        dpi=300 if ext == "png" else None,
        bbox_inches="tight",
    )

plt.close(fig)


# =====================================================================
# FFT helpers
# =====================================================================

def normalized_fft(
    y: np.ndarray,
):

    y = np.asarray(
        y,
        dtype=float,
    ).reshape(-1)

    y = (
        y
        -
        np.mean(y)
    )

    fft = np.fft.rfft(
        y
    )

    freq = np.fft.rfftfreq(
        len(y),
        d=1.0,
    )

    power = (
        np.abs(fft) ** 2
    )


    # Remove DC
    freq = freq[1:]
    power = power[1:]


    if power.sum() > 0:

        power = (
            power
            /
            power.sum()
        )


    period = (
        1.0
        /
        freq
    )


    order = np.argsort(
        period
    )


    return (
        period[order],
        power[order],
    )


def aggregate_fft(
    family: str,
    depth: int,
):

    powers = []

    true_power_ref = None
    period_ref = None


    for seed in SEEDS:

        data = read_predictions(
            family,
            depth,
            seed,
        )

        true_key = first_key(
            data,
            TRUE_KEYS,
        )

        pred_key = first_key(
            data,
            PRED_KEYS,
        )


        truth = np.asarray(
            data[true_key],
            dtype=float,
        )

        pred = np.asarray(
            data[pred_key],
            dtype=float,
        )


        p_true, q_true = (
            normalized_fft(
                truth
            )
        )

        p_pred, q_pred = (
            normalized_fft(
                pred
            )
        )


        if not np.allclose(
            p_true,
            p_pred,
        ):

            raise ValueError(
                "FFT period mismatch"
            )


        if period_ref is None:

            period_ref = (
                p_true
            )

            true_power_ref = (
                q_true
            )


        powers.append(
            q_pred
        )


    matrix = np.stack(
        powers,
        axis=0,
    )


    return (
        period_ref,
        true_power_ref,
        matrix.mean(axis=0),
        matrix.std(
            axis=0,
            ddof=1,
        ),
    )


# =====================================================================
# Figure 8:
# A1 vs A2 FFT spectra
# =====================================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(14.0, 11.5),
    constrained_layout=True,
)

axes = axes.ravel()


for ax, depth in zip(
    axes,
    DEPTHS,
):

    p1, true1, m1, s1 = (
        aggregate_fft(
            "a1",
            depth,
        )
    )

    p2, true2, m2, s2 = (
        aggregate_fft(
            "a2",
            depth,
        )
    )


    ax.plot(
        p1,
        true1,
        linewidth=1.7,
        label="True",
    )


    line1, = ax.plot(
        p1,
        m1,
        linewidth=1.7,
        label="A1",
    )

    ax.fill_between(
        p1,
        np.maximum(
            m1 - s1,
            0,
        ),
        m1 + s1,
        alpha=0.17,
        color=line1.get_color(),
    )


    line2, = ax.plot(
        p2,
        m2,
        linewidth=1.7,
        label="A2",
    )

    ax.fill_between(
        p2,
        np.maximum(
            m2 - s2,
            0,
        ),
        m2 + s2,
        alpha=0.17,
        color=line2.get_color(),
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
        alpha=0.20
    )


axes[0].legend(
    frameon=False,
    ncol=3,
)


fig.suptitle(
    "Test-set Fourier spectra: "
    "one versus two ansatz layers",
    fontsize=14,
)


for ext in ["pdf", "png"]:

    fig.savefig(
        FIG_DIR
        / f"08_a1_vs_a2_fft_spectra.{ext}",
        dpi=300 if ext == "png" else None,
        bbox_inches="tight",
    )

plt.close(fig)


# =====================================================================
# Best-depth summaries
# =====================================================================

best_rows = []

for family in FAMILIES:

    fam = (
        agg[
            agg["family"]
            ==
            family
        ]
        .copy()
    )


    best_rmse = fam.loc[
        fam[
            "test_rmse_original_mean"
        ].idxmin()
    ]


    best_kge = fam.loc[
        fam[
            "test_kge_mean"
        ].idxmax()
    ]


    best_fft = fam.loc[
        fam[
            "test_fft_spectral_cosine_similarity_mean"
        ].idxmax()
    ]


    best_rows.append(
        {
            "family": family,

            "best_rmse_depth":
                int(
                    best_rmse[
                        "depth"
                    ]
                ),

            "best_rmse_mean":
                best_rmse[
                    "test_rmse_original_mean"
                ],

            "best_rmse_sd":
                best_rmse[
                    "test_rmse_original_std"
                ],

            "best_kge_depth":
                int(
                    best_kge[
                        "depth"
                    ]
                ),

            "best_kge_mean":
                best_kge[
                    "test_kge_mean"
                ],

            "best_fft_depth":
                int(
                    best_fft[
                        "depth"
                    ]
                ),

            "best_fft_mean":
                best_fft[
                    "test_fft_spectral_cosine_similarity_mean"
                ],
        }
    )


best_summary = pd.DataFrame(
    best_rows
)

best_summary.to_csv(
    TABLE_DIR
    / "a1_vs_a2_best_depth_summary.csv",
    index=False,
)


# =====================================================================
# Print final concise numerical comparison
# =====================================================================

print()
print("=" * 100)
print("A1 VS A2 — FIVE-SEED SUMMARY")
print("=" * 100)


for depth in DEPTHS:

    print()
    print(
        f"DEPTH {depth}"
    )

    for family in [
        "a1",
        "a2",
    ]:

        row = (
            agg[
                (agg["family"] == family)
                &
                (agg["depth"] == depth)
            ]
            .iloc[0]
        )


        print(
            f"  {family.upper()}  "
            f"RMSE="
            f"{row['test_rmse_original_mean']:.4f}"
            f" ± "
            f"{row['test_rmse_original_std']:.4f}"
            f"   "
            f"Pearson="
            f"{row['test_pearson_corr_mean']:.4f}"
            f" ± "
            f"{row['test_pearson_corr_std']:.4f}"
            f"   "
            f"KGE="
            f"{row['test_kge_mean']:.4f}"
            f" ± "
            f"{row['test_kge_std']:.4f}"
            f"   "
            f"FFT="
            f"{row['test_fft_spectral_cosine_similarity_mean']:.4f}"
            f" ± "
            f"{row['test_fft_spectral_cosine_similarity_std']:.4f}"
            f"   "
            f"Runtime="
            f"{row['elapsed_time_sec_mean']/60:.2f}"
            f" ± "
            f"{row['elapsed_time_sec_std']/60:.2f}"
            f" min"
        )


print()
print("=" * 100)
print("PAIRED RMSE DIFFERENCE: A2 - A1")
print("=" * 100)


rmse_stats = paired_stats[
    paired_stats["metric"]
    ==
    "test_rmse_original"
].copy()


for _, row in (
    rmse_stats
    .sort_values("depth")
    .iterrows()
):

    print(
        f"Depth {int(row['depth'])}: "
        f"ΔRMSE="
        f"{row['mean_a2_minus_a1']:.4f}"
        f" ± "
        f"{row['sd_paired_difference']:.4f}"
        f"   "
        f"exact sign-flip p="
        f"{row['exact_two_sided_sign_flip_p']:.4f}"
        f"   "
        f"{row['mean_direction']}"
    )


print()
print("=" * 100)
print("BEST DEPTHS")
print("=" * 100)

print(
    best_summary.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("GENERATED FIGURES")
print("=" * 100)

for path in sorted(
    FIG_DIR.glob("*.pdf")
):

    print(
        path.name
    )


print()
print("=" * 100)
print("GENERATED TABLES")
print("=" * 100)

for path in sorted(
    TABLE_DIR.glob("*")
):

    print(
        path.name
    )


print()
print("DONE.")
