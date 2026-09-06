from pathlib import Path
import os

import numpy as np
import pandas as pd
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


REPO = _resolve_project_root()

BASE = (
    REPO
    / "data_reupload/results/final_model_sweep/"
      "single_tanh_a1/depth_1"
)

OUT = (
    REPO
    / "data_reupload/report/final_ideal_noisy/figures"
)

OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]


# ------------------------------------------------------------
# Load histories
# ------------------------------------------------------------
histories = []

for seed in SEEDS:
    path = BASE / f"seed_{seed}" / "history.csv"

    if not path.is_file():
        raise FileNotFoundError(f"Missing history: {path}")

    df = pd.read_csv(path)

    required = [
        "epoch",
        "train_loss_scaled",
        "val_loss_scaled",
        "train_rmse_original",
        "val_rmse_original",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[required].copy()
    df["seed"] = seed
    histories.append(df)


all_hist = pd.concat(histories, ignore_index=True)


# ------------------------------------------------------------
# Aggregate across seeds at each epoch
# ------------------------------------------------------------
agg = (
    all_hist
    .groupby("epoch", as_index=False)
    .agg(
        train_loss_mean=("train_loss_scaled", "mean"),
        train_loss_sd=("train_loss_scaled", "std"),
        val_loss_mean=("val_loss_scaled", "mean"),
        val_loss_sd=("val_loss_scaled", "std"),

        train_rmse_mean=("train_rmse_original", "mean"),
        train_rmse_sd=("train_rmse_original", "std"),
        val_rmse_mean=("val_rmse_original", "mean"),
        val_rmse_sd=("val_rmse_original", "std"),
    )
)


# ------------------------------------------------------------
# Save aggregated numerical history as well
# ------------------------------------------------------------
table_path = (
    REPO
    / "data_reupload/report/final_ideal_noisy/tables/"
      "ideal_depth1_training_history_mean_std.csv"
)

agg.to_csv(table_path, index=False)


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig, axes = plt.subplots(
    2,
    1,
    figsize=(8.5, 8.0),
    sharex=True,
    constrained_layout=True,
)

epoch = agg["epoch"].to_numpy()


# --- Panel A: scaled MSE
ax = axes[0]

ax.plot(
    epoch,
    agg["train_loss_mean"],
    linewidth=2.0,
    label="Training",
)

ax.fill_between(
    epoch,
    agg["train_loss_mean"] - agg["train_loss_sd"],
    agg["train_loss_mean"] + agg["train_loss_sd"],
    alpha=0.20,
)

ax.plot(
    epoch,
    agg["val_loss_mean"],
    linewidth=2.0,
    label="Validation",
)

ax.fill_between(
    epoch,
    agg["val_loss_mean"] - agg["val_loss_sd"],
    agg["val_loss_mean"] + agg["val_loss_sd"],
    alpha=0.20,
)

ax.set_ylabel("Scaled MSE")
ax.set_title("(a) Training and validation loss")
ax.grid(alpha=0.25)
ax.legend(frameon=False)


# --- Panel B: RMSE original units
ax = axes[1]

ax.plot(
    epoch,
    agg["train_rmse_mean"],
    linewidth=2.0,
    label="Training",
)

ax.fill_between(
    epoch,
    agg["train_rmse_mean"] - agg["train_rmse_sd"],
    agg["train_rmse_mean"] + agg["train_rmse_sd"],
    alpha=0.20,
)

ax.plot(
    epoch,
    agg["val_rmse_mean"],
    linewidth=2.0,
    label="Validation",
)

ax.fill_between(
    epoch,
    agg["val_rmse_mean"] - agg["val_rmse_sd"],
    agg["val_rmse_mean"] + agg["val_rmse_sd"],
    alpha=0.20,
)

ax.set_xlabel("Epoch")
ax.set_ylabel("RMSE [original units]")
ax.set_title("(b) Training and validation RMSE")
ax.grid(alpha=0.25)
ax.legend(frameon=False)


fig.suptitle(
    "Ideal depth-1 QNN training history: mean ± 1 SD across seeds 42–46",
    fontsize=13,
)

pdf_path = OUT / "ideal_depth1_training_history_mean_std.pdf"
png_path = OUT / "ideal_depth1_training_history_mean_std.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

plt.close(fig)


print("=" * 80)
print("IDEAL DEPTH-1 HISTORY COMPLETE")
print("=" * 80)
print(f"Seeds: {SEEDS}")
print(f"Epochs: {len(agg)}")
print(f"PDF:   {pdf_path}")
print(f"PNG:   {png_path}")
print(f"Table: {table_path}")
