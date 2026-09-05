from __future__ import annotations

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# Paths / experiment definition
# =====================================================================

REPO = Path(
    "/umbc/rs/pi_deffner/users/devjyot1/projects/"
    "Quantum-Machine-Learning-"
)

IDEAL_ROOT = (
    REPO
    / "data_reupload/results/final_model_sweep/single_tanh_a1"
)

NOISY_ROOT = (
    REPO
    / "data_reupload/noisy_direct14/results"
)

IDEAL_ANALYSIS_TABLES = (
    REPO
    / "data_reupload/analysis/complete_ideal/output/tables"
)

REPORT_ROOT = (
    REPO
    / "data_reupload/report/final_ideal_noisy"
)

FIG_DIR = REPORT_ROOT / "figures"
TABLE_DIR = REPORT_ROOT / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
IDEAL_DEPTHS = [1, 2, 3, 4, 5, 6]
COMPARE_DEPTHS = [1, 2, 3]


# =====================================================================
# Exact production paths
# =====================================================================

def ideal_run_dir(depth: int, seed: int) -> Path:
    return (
        IDEAL_ROOT
        / f"depth_{depth}"
        / f"seed_{seed}"
    )


def noisy_run_dir(depth: int, seed: int) -> Path:
    # IMPORTANT:
    # This intentionally uses only the final e100 production runs,
    # excluding smoke tests and one-epoch/full-epoch benchmarks.
    return (
        NOISY_ROOT
        / f"direct14_w32_prod_d{depth}_s512_e100_seed{seed}"
        / f"depth_{depth}"
        / f"seed_{seed}"
    )


# =====================================================================
# Basic IO helpers
# =====================================================================

def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")


def read_summary(run_dir: Path) -> pd.Series:
    path = run_dir / "result_summary.csv"
    require_file(path)

    df = pd.read_csv(path)

    if len(df) != 1:
        raise ValueError(
            f"Expected exactly one row in {path}; found {len(df)}"
        )

    return df.iloc[0]


def read_history(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "history.csv"
    require_file(path)
    return pd.read_csv(path)


def read_predictions(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "predictions.npz"
    require_file(path)

    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise KeyError(
        f"Could not find {description}.\n"
        f"Candidates: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def first_existing_key(
    data: dict[str, np.ndarray],
    candidates: list[str],
    description: str,
) -> str:
    for candidate in candidates:
        if candidate in data:
            return candidate

    raise KeyError(
        f"Could not find {description}.\n"
        f"Candidates: {candidates}\n"
        f"Available keys: {list(data.keys())}"
    )


# =====================================================================
# Verify all expected production artifacts before analysis
# =====================================================================

print("=" * 100)
print("VERIFYING EXPERIMENT ARTIFACTS")
print("=" * 100)

for depth in IDEAL_DEPTHS:
    for seed in SEEDS:
        run = ideal_run_dir(depth, seed)

        for name in [
            "result_summary.csv",
            "history.csv",
            "predictions.npz",
        ]:
            require_file(run / name)

print("Verified ideal runs: 6 depths x 5 seeds = 30")


for depth in COMPARE_DEPTHS:
    for seed in SEEDS:
        run = noisy_run_dir(depth, seed)

        for name in [
            "result_summary.csv",
            "history.csv",
            "predictions.npz",
            "config.json",
        ]:
            require_file(run / name)

print("Verified FINAL noisy runs: 3 depths x 5 seeds = 15")
print()


# =====================================================================
# Load summary metrics
# =====================================================================

ideal_rows: list[dict] = []

for depth in IDEAL_DEPTHS:
    for seed in SEEDS:
        row = read_summary(
            ideal_run_dir(depth, seed)
        ).to_dict()

        row["setting"] = "Ideal"
        row["depth"] = depth
        row["seed"] = seed

        ideal_rows.append(row)


noisy_rows: list[dict] = []

for depth in COMPARE_DEPTHS:
    for seed in SEEDS:
        row = read_summary(
            noisy_run_dir(depth, seed)
        ).to_dict()

        row["setting"] = "Noisy"
        row["depth"] = depth
        row["seed"] = seed

        noisy_rows.append(row)


ideal_runs = pd.DataFrame(ideal_rows)
noisy_runs = pd.DataFrame(noisy_rows)


# Convert everything that can be numeric.
for df in [ideal_runs, noisy_runs]:
    for col in df.columns:
        if col not in ["setting", "variant_id", "model_family"]:
            converted = pd.to_numeric(df[col], errors="coerce")

            if converted.notna().any():
                df[col] = converted


# Save every final noisy run.
noisy_runs.to_csv(
    TABLE_DIR / "table_noisy_all_runs.csv",
    index=False,
)


# =====================================================================
# Metrics used for the report
# =====================================================================

MAIN_METRICS = [
    "test_rmse_original",
    "test_mae_original",
    "test_pearson_corr",
    "test_spearman_corr",
    "test_kge",
    "test_nse",
]

SPECTRAL_METRICS = [
    "test_fft_spectral_cosine_similarity",
    "test_abs_12_step_band_error",
    "test_abs_short_period_2_to_6_error",
]

ADDITIONAL_METRICS = [
    "val_rmse_original",
    "test_bias",
    "test_pred_std_over_true_std",
    "elapsed_time_sec",
    "max_rss_gb_python",
]

ALL_DESIRED = (
    MAIN_METRICS
    + SPECTRAL_METRICS
    + ADDITIONAL_METRICS
)


def available_metrics(
    *frames: pd.DataFrame,
) -> list[str]:

    metrics = []

    for metric in ALL_DESIRED:
        if all(metric in df.columns for df in frames):
            metrics.append(metric)

    return metrics


common_metrics = available_metrics(
    ideal_runs,
    noisy_runs,
)

print("Metrics common to ideal and noisy summaries:")
for metric in common_metrics:
    print(f"  {metric}")
print()


# =====================================================================
# Aggregate helper
# =====================================================================

def aggregate_runs(
    df: pd.DataFrame,
    depths: list[int],
    metrics: list[str],
) -> pd.DataFrame:

    selected = df[df["depth"].isin(depths)].copy()

    grouped = (
        selected
        .groupby("depth")[metrics]
        .agg(["mean", "std"])
    )

    grouped.columns = [
        f"{metric}_{stat}"
        for metric, stat in grouped.columns
    ]

    return grouped.reset_index()


ideal_agg_1_3 = aggregate_runs(
    ideal_runs,
    COMPARE_DEPTHS,
    common_metrics,
)

noisy_agg = aggregate_runs(
    noisy_runs,
    COMPARE_DEPTHS,
    common_metrics,
)

ideal_agg_all = aggregate_runs(
    ideal_runs,
    IDEAL_DEPTHS,
    [
        metric
        for metric in ALL_DESIRED
        if metric in ideal_runs.columns
    ],
)


ideal_agg_1_3.insert(0, "setting", "Ideal")
noisy_agg.insert(0, "setting", "Noisy")


comparison_agg = pd.concat(
    [ideal_agg_1_3, noisy_agg],
    ignore_index=True,
)

comparison_agg = comparison_agg.sort_values(
    ["depth", "setting"]
).reset_index(drop=True)


comparison_agg.to_csv(
    TABLE_DIR / "table_ideal_vs_noisy_depth1_3.csv",
    index=False,
)

noisy_agg.to_csv(
    TABLE_DIR / "table_noisy_depth_summary_mean_std.csv",
    index=False,
)

ideal_agg_all.to_csv(
    TABLE_DIR / "table_ideal_single_full_recomputed_mean_std.csv",
    index=False,
)


# =====================================================================
# Human-readable mean ± SD tables
# =====================================================================

def format_pm(mean: float, sd: float) -> str:
    if pd.isna(mean):
        return ""

    if pd.isna(sd):
        return f"{mean:.3f}"

    return f"{mean:.3f} ± {sd:.3f}"


def make_pretty_table(
    aggregate_df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:

    rows = []

    for _, r in aggregate_df.iterrows():
        out = {}

        if "setting" in aggregate_df.columns:
            out["Setting"] = r["setting"]

        out["Depth"] = int(r["depth"])

        for metric in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            if mean_col not in aggregate_df.columns:
                continue

            out[metric] = format_pm(
                r[mean_col],
                r.get(std_col, np.nan),
            )

        rows.append(out)

    return pd.DataFrame(rows)


pretty_main = make_pretty_table(
    comparison_agg,
    MAIN_METRICS,
)

pretty_main.to_csv(
    TABLE_DIR / "table_ideal_vs_noisy_depth1_3_pretty.csv",
    index=False,
)


with open(
    TABLE_DIR / "table_ideal_vs_noisy_depth1_3.tex",
    "w",
) as f:
    f.write(
        pretty_main.to_latex(
            index=False,
            escape=True,
            caption=(
                "Ideal and finite-shot noisy performance for "
                "depths 1--3. Values are mean "
                "$\\pm$ standard deviation across seeds 42--46."
            ),
            label="tab:ideal_noisy_main",
        )
    )


pretty_noisy = make_pretty_table(
    noisy_agg,
    MAIN_METRICS + SPECTRAL_METRICS,
)

pretty_noisy.to_csv(
    TABLE_DIR / "table_noisy_full_mean_std_pretty.csv",
    index=False,
)

with open(
    TABLE_DIR / "table_noisy_full_mean_std.tex",
    "w",
) as f:
    f.write(
        pretty_noisy.to_latex(
            index=False,
            escape=True,
            caption=(
                "Complete five-seed noisy depth summary."
            ),
            label="tab:noisy_full",
        )
    )


# =====================================================================
# Copy canonical ideal summary tables into report directory
# =====================================================================

canonical_ideal_tables = {
    "single_depth_mean_sd_table.csv":
        "table_ideal_single_full_mean_std.csv",

    "residual_mean_sd_table.csv":
        "table_ideal_residual_full_mean_std.csv",
}


for source_name, dest_name in canonical_ideal_tables.items():

    source = IDEAL_ANALYSIS_TABLES / source_name
    require_file(source)

    dest = TABLE_DIR / dest_name

    shutil.copy2(source, dest)

    table_df = pd.read_csv(source)

    tex_dest = dest.with_suffix(".tex")

    with open(tex_dest, "w") as f:
        f.write(
            table_df.to_latex(
                index=False,
                escape=True,
                longtable=True,
            )
        )


# =====================================================================
# FIGURE 1:
# Ideal vs noisy metrics, depths 1--3
# =====================================================================

metric_titles = {
    "test_rmse_original": "Test RMSE",
    "test_mae_original": "Test MAE",
    "test_pearson_corr": "Pearson correlation",
    "test_spearman_corr": "Spearman correlation",
    "test_kge": "KGE",
    "test_nse": "NSE",
}


fig, axes = plt.subplots(
    2,
    3,
    figsize=(13.5, 8.2),
    constrained_layout=True,
)

axes = axes.ravel()

depths_np = np.asarray(COMPARE_DEPTHS)


for ax, metric in zip(
    axes,
    MAIN_METRICS,
):

    for setting, agg in [
        ("Ideal", ideal_agg_1_3),
        ("Noisy", noisy_agg),
    ]:

        means = agg[
            f"{metric}_mean"
        ].to_numpy(dtype=float)

        stds = agg[
            f"{metric}_std"
        ].to_numpy(dtype=float)

        line, = ax.plot(
            depths_np,
            means,
            marker="o",
            linewidth=2.0,
            label=setting,
        )

        ax.fill_between(
            depths_np,
            means - stds,
            means + stds,
            alpha=0.20,
            color=line.get_color(),
        )

    ax.set_title(metric_titles[metric])
    ax.set_xlabel("Reuploading depth")
    ax.set_xticks(COMPARE_DEPTHS)
    ax.grid(alpha=0.25)


axes[0].legend(frameon=False)

fig.suptitle(
    "Ideal versus finite-shot noisy QNN performance: "
    "mean ± 1 SD across seeds 42–46",
    fontsize=14,
)

for ext in ["pdf", "png"]:
    path = (
        FIG_DIR
        / f"noisy_vs_ideal_metrics_depth1_3.{ext}"
    )

    fig.savefig(
        path,
        dpi=300 if ext == "png" else None,
        bbox_inches="tight",
    )

plt.close(fig)


# =====================================================================
# Determine best noisy depth using mean test RMSE
# =====================================================================

best_noisy_row = noisy_agg.loc[
    noisy_agg["test_rmse_original_mean"].idxmin()
]

BEST_NOISY_DEPTH = int(
    best_noisy_row["depth"]
)

print("=" * 100)
print(
    "BEST NOISY DEPTH BY FIVE-SEED MEAN TEST RMSE:",
    BEST_NOISY_DEPTH,
)
print(
    "Mean RMSE:",
    best_noisy_row["test_rmse_original_mean"],
)
print(
    "SD RMSE:",
    best_noisy_row["test_rmse_original_std"],
)
print("=" * 100)
print()


# =====================================================================
# Training history aggregation
# =====================================================================

TRAIN_LOSS_CANDIDATES = [
    "train_loss_scaled",
    "train_loss",
    "training_loss",
]

VAL_LOSS_CANDIDATES = [
    "val_loss_scaled",
    "val_loss",
    "validation_loss",
]

EPOCH_CANDIDATES = [
    "epoch",
    "Epoch",
]


def aggregate_history(
    setting: str,
    depth: int,
) -> pd.DataFrame:

    frames = []

    for seed in SEEDS:

        if setting == "Ideal":
            run = ideal_run_dir(depth, seed)
        elif setting == "Noisy":
            run = noisy_run_dir(depth, seed)
        else:
            raise ValueError(setting)

        df = read_history(run)

        epoch_col = first_existing_column(
            df,
            EPOCH_CANDIDATES,
            "epoch column",
        )

        train_col = first_existing_column(
            df,
            TRAIN_LOSS_CANDIDATES,
            "training loss",
        )

        val_col = first_existing_column(
            df,
            VAL_LOSS_CANDIDATES,
            "validation loss",
        )

        tmp = pd.DataFrame(
            {
                "epoch": pd.to_numeric(
                    df[epoch_col],
                    errors="coerce",
                ),
                "train_loss": pd.to_numeric(
                    df[train_col],
                    errors="coerce",
                ),
                "val_loss": pd.to_numeric(
                    df[val_col],
                    errors="coerce",
                ),
                "seed": seed,
            }
        )

        frames.append(tmp)

    all_df = pd.concat(
        frames,
        ignore_index=True,
    )

    agg = (
        all_df
        .groupby("epoch", as_index=False)
        .agg(
            train_mean=("train_loss", "mean"),
            train_std=("train_loss", "std"),
            val_mean=("val_loss", "mean"),
            val_std=("val_loss", "std"),
        )
    )

    return agg


ideal_hist = aggregate_history(
    "Ideal",
    BEST_NOISY_DEPTH,
)

noisy_hist = aggregate_history(
    "Noisy",
    BEST_NOISY_DEPTH,
)


ideal_hist.to_csv(
    TABLE_DIR
    / f"ideal_depth{BEST_NOISY_DEPTH}_history_mean_std.csv",
    index=False,
)

noisy_hist.to_csv(
    TABLE_DIR
    / f"noisy_depth{BEST_NOISY_DEPTH}_history_mean_std.csv",
    index=False,
)


# =====================================================================
# FIGURE 2:
# Best noisy-depth training history vs ideal counterpart
# =====================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.5, 4.8),
    constrained_layout=True,
)


# Training loss
ax = axes[0]

for label, agg in [
    ("Ideal", ideal_hist),
    ("Noisy", noisy_hist),
]:
    e = agg["epoch"].to_numpy(dtype=float)
    m = agg["train_mean"].to_numpy(dtype=float)
    s = agg["train_std"].to_numpy(dtype=float)

    line, = ax.plot(
        e,
        m,
        linewidth=2.0,
        label=label,
    )

    ax.fill_between(
        e,
        m - s,
        m + s,
        alpha=0.20,
        color=line.get_color(),
    )

ax.set_title("(a) Training loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Scaled MSE")
ax.grid(alpha=0.25)
ax.legend(frameon=False)


# Validation loss
ax = axes[1]

for label, agg in [
    ("Ideal", ideal_hist),
    ("Noisy", noisy_hist),
]:
    e = agg["epoch"].to_numpy(dtype=float)
    m = agg["val_mean"].to_numpy(dtype=float)
    s = agg["val_std"].to_numpy(dtype=float)

    line, = ax.plot(
        e,
        m,
        linewidth=2.0,
        label=label,
    )

    ax.fill_between(
        e,
        m - s,
        m + s,
        alpha=0.20,
        color=line.get_color(),
    )

ax.set_title("(b) Validation loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Scaled MSE")
ax.grid(alpha=0.25)
ax.legend(frameon=False)


fig.suptitle(
    f"Ideal versus noisy training dynamics at depth "
    f"{BEST_NOISY_DEPTH}: mean ± 1 SD",
    fontsize=14,
)

for ext in ["pdf", "png"]:

    fig.savefig(
        FIG_DIR
        / f"noisy_vs_ideal_best_depth_training_history.{ext}",
        dpi=300 if ext == "png" else None,
        bbox_inches="tight",
    )

plt.close(fig)


# =====================================================================
# Prediction aggregation
# =====================================================================

TRUE_KEY_CANDIDATES = [
    "y_test_true_original",
    "test_true_original",
    "y_test_true",
]

PRED_KEY_CANDIDATES = [
    "y_test_pred_original",
    "test_pred_original",
    "y_test_pred",
]

INDEX_KEY_CANDIDATES = [
    "test_target_indices",
    "y_test_target_indices",
    "target_indices_test",
]


def aggregate_predictions(
    setting: str,
    depth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    preds = []
    truth_ref = None
    index_ref = None

    for seed in SEEDS:

        if setting == "Ideal":
            run = ideal_run_dir(depth, seed)
        else:
            run = noisy_run_dir(depth, seed)

        data = read_predictions(run)

        true_key = first_existing_key(
            data,
            TRUE_KEY_CANDIDATES,
            "test truth array",
        )

        pred_key = first_existing_key(
            data,
            PRED_KEY_CANDIDATES,
            "test prediction array",
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
            truth_ref = truth.copy()
        elif not np.allclose(
            truth_ref,
            truth,
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError(
                f"Test truth differs between seeds for "
                f"{setting} depth {depth}"
            )

        if pred.shape != truth.shape:
            raise ValueError(
                f"Prediction shape {pred.shape} does not match "
                f"truth shape {truth.shape}"
            )

        index_key = None

        for candidate in INDEX_KEY_CANDIDATES:
            if candidate in data:
                index_key = candidate
                break

        if index_key is not None:
            idx = np.asarray(
                data[index_key]
            ).reshape(-1)

            if index_ref is None:
                index_ref = idx.copy()

        preds.append(pred)

    pred_matrix = np.stack(
        preds,
        axis=0,
    )

    if index_ref is None:
        index_ref = np.arange(
            len(truth_ref)
        )

    return (
        index_ref,
        truth_ref,
        pred_matrix.mean(axis=0),
        pred_matrix.std(axis=0, ddof=1),
    )


# =====================================================================
# FIGURE 3:
# Ideal vs noisy test trajectories, depths 1--3
# =====================================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(13.5, 11.0),
    sharex=True,
    constrained_layout=True,
)


for ax, depth in zip(
    axes,
    COMPARE_DEPTHS,
):

    x_i, true_i, mean_i, sd_i = (
        aggregate_predictions(
            "Ideal",
            depth,
        )
    )

    x_n, true_n, mean_n, sd_n = (
        aggregate_predictions(
            "Noisy",
            depth,
        )
    )

    if not np.allclose(
        true_i,
        true_n,
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(
            f"Ideal/noisy true test arrays differ at depth {depth}"
        )

    ax.plot(
        x_i,
        true_i,
        linewidth=1.8,
        label="True",
    )

    line_i, = ax.plot(
        x_i,
        mean_i,
        linewidth=1.8,
        label="Ideal mean",
    )

    ax.fill_between(
        x_i,
        mean_i - sd_i,
        mean_i + sd_i,
        alpha=0.18,
        color=line_i.get_color(),
    )

    line_n, = ax.plot(
        x_n,
        mean_n,
        linewidth=1.8,
        label="Noisy mean",
    )

    ax.fill_between(
        x_n,
        mean_n - sd_n,
        mean_n + sd_n,
        alpha=0.18,
        color=line_n.get_color(),
    )

    ax.set_ylabel("Target")
    ax.set_title(f"Depth {depth}")
    ax.grid(alpha=0.22)


axes[0].legend(
    frameon=False,
    ncol=3,
)

axes[-1].set_xlabel("Test target index")

fig.suptitle(
    "Ideal versus finite-shot noisy test predictions: "
    "mean ± 1 SD across seeds 42–46",
    fontsize=14,
)

for ext in ["pdf", "png"]:

    fig.savefig(
        FIG_DIR
        / f"noisy_vs_ideal_test_predictions_depth1_3.{ext}",
        dpi=300 if ext == "png" else None,
        bbox_inches="tight",
    )

plt.close(fig)


# =====================================================================
# FFT helper
# =====================================================================

def normalized_fft_power(
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    y = np.asarray(
        y,
        dtype=float,
    ).reshape(-1)

    demeaned = y - np.mean(y)

    fft = np.fft.rfft(
        demeaned
    )

    freq = np.fft.rfftfreq(
        len(y),
        d=1.0,
    )

    power = np.abs(fft) ** 2

    # Remove DC.
    freq = freq[1:]
    power = power[1:]

    total = power.sum()

    if total > 0:
        power = power / total

    periods = 1.0 / freq

    # Plot period in increasing order.
    order = np.argsort(periods)

    return (
        periods[order],
        power[order],
    )


def aggregate_fft_predictions(
    setting: str,
    depth: int,
):

    all_power = []
    true_power = None
    periods_ref = None

    for seed in SEEDS:

        if setting == "Ideal":
            run = ideal_run_dir(depth, seed)
        else:
            run = noisy_run_dir(depth, seed)

        data = read_predictions(run)

        true_key = first_existing_key(
            data,
            TRUE_KEY_CANDIDATES,
            "test truth array",
        )

        pred_key = first_existing_key(
            data,
            PRED_KEY_CANDIDATES,
            "test prediction array",
        )

        truth = np.asarray(
            data[true_key],
            dtype=float,
        )

        pred = np.asarray(
            data[pred_key],
            dtype=float,
        )

        periods, tp = normalized_fft_power(
            truth
        )

        periods_p, pp = normalized_fft_power(
            pred
        )

        if not np.allclose(
            periods,
            periods_p,
        ):
            raise ValueError(
                "FFT period axes differ."
            )

        if periods_ref is None:
            periods_ref = periods
            true_power = tp

        all_power.append(pp)

    power_matrix = np.stack(
        all_power,
        axis=0,
    )

    return (
        periods_ref,
        true_power,
        power_matrix.mean(axis=0),
        power_matrix.std(axis=0, ddof=1),
    )


# =====================================================================
# FIGURE 4:
# Ideal vs noisy normalized FFT spectra
# =====================================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12.5, 10.5),
    constrained_layout=True,
)


for ax, depth in zip(
    axes,
    COMPARE_DEPTHS,
):

    periods_i, true_i, mean_i, sd_i = (
        aggregate_fft_predictions(
            "Ideal",
            depth,
        )
    )

    periods_n, true_n, mean_n, sd_n = (
        aggregate_fft_predictions(
            "Noisy",
            depth,
        )
    )

    ax.plot(
        periods_i,
        true_i,
        linewidth=1.8,
        label="True spectrum",
    )

    line_i, = ax.plot(
        periods_i,
        mean_i,
        linewidth=1.8,
        label="Ideal mean",
    )

    ax.fill_between(
        periods_i,
        np.maximum(mean_i - sd_i, 0),
        mean_i + sd_i,
        alpha=0.18,
        color=line_i.get_color(),
    )

    line_n, = ax.plot(
        periods_n,
        mean_n,
        linewidth=1.8,
        label="Noisy mean",
    )

    ax.fill_between(
        periods_n,
        np.maximum(mean_n - sd_n, 0),
        mean_n + sd_n,
        alpha=0.18,
        color=line_n.get_color(),
    )

    ax.set_title(
        f"Depth {depth}"
    )

    ax.set_ylabel(
        "Normalized non-DC power"
    )

    ax.grid(alpha=0.22)


axes[0].legend(
    frameon=False,
    ncol=3,
)

axes[-1].set_xlabel(
    "Period [time steps]"
)

fig.suptitle(
    "Ideal versus noisy test-set Fourier power: "
    "mean ± 1 SD across seeds",
    fontsize=14,
)

for ext in ["pdf", "png"]:

    fig.savefig(
        FIG_DIR
        / f"noisy_vs_ideal_fft_depth1_3.{ext}",
        dpi=300 if ext == "png" else None,
        bbox_inches="tight",
    )

plt.close(fig)


# =====================================================================
# Manifest
# =====================================================================

manifest = {
    "ideal_depths": IDEAL_DEPTHS,
    "comparison_depths": COMPARE_DEPTHS,
    "seeds": SEEDS,
    "best_noisy_depth_by_mean_test_rmse": BEST_NOISY_DEPTH,
    "noisy_runs_used": [
        str(noisy_run_dir(d, s))
        for d in COMPARE_DEPTHS
        for s in SEEDS
    ],
    "excluded_noisy_run_types": [
        "smoke",
        "single full-epoch benchmarking runs",
        "worker-count benchmarking runs",
    ],
}

with open(
    REPORT_ROOT / "analysis_manifest.json",
    "w",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )


# =====================================================================
# Print concise final numerical summary
# =====================================================================

print()
print("=" * 100)
print("FINAL IDEAL VS NOISY DEPTH 1--3 SUMMARY")
print("=" * 100)

for depth in COMPARE_DEPTHS:

    print()
    print(f"DEPTH {depth}")

    for setting, agg in [
        ("Ideal", ideal_agg_1_3),
        ("Noisy", noisy_agg),
    ]:

        row = agg[
            agg["depth"] == depth
        ].iloc[0]

        print(
            f"  {setting:5s}  "
            f"RMSE="
            f"{row['test_rmse_original_mean']:.4f}"
            f" ± "
            f"{row['test_rmse_original_std']:.4f}"
            f"   "
            f"MAE="
            f"{row['test_mae_original_mean']:.4f}"
            f" ± "
            f"{row['test_mae_original_std']:.4f}"
            f"   "
            f"Pearson="
            f"{row['test_pearson_corr_mean']:.4f}"
            f" ± "
            f"{row['test_pearson_corr_std']:.4f}"
        )

print()
print("=" * 100)
print("GENERATED FIGURES")
print("=" * 100)

for path in sorted(FIG_DIR.glob("*")):
    print(path.name)

print()
print("=" * 100)
print("GENERATED TABLES")
print("=" * 100)

for path in sorted(TABLE_DIR.glob("*")):
    print(path.name)

print()
print("DONE.")
