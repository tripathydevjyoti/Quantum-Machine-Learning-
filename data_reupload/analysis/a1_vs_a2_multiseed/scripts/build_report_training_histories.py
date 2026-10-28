from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# Configuration
# ======================================================================

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

RESULTS = REPO / "data_reupload/results"

OUT = (
    REPO
    / "data_reupload/analysis/a1_vs_a2_multiseed/"
      "output/report_training_histories"
)

OUT.mkdir(parents=True, exist_ok=True)

# These are internal directory names only.
# They do NOT appear in the report/figure labels.
FAMILIES = {
    1: "single_tanh_a1",
    2: "single_tanh_a2",
}

DISPLAY = {
    1: "1 ansatz layer/block",
    2: "2 ansatz layers/block",
}

DEPTHS = [1, 2, 3, 4, 5, 6]
SEEDS = [42, 43, 44, 45, 46]

EPOCH = "epoch"
TRAIN = "train_loss_scaled"
VAL = "val_loss_scaled"


# ======================================================================
# Robust history discovery
# ======================================================================

def find_history(ansatz_layers: int, depth: int, seed: int) -> Path:
    family = FAMILIES[ansatz_layers]

    # Most likely production location.
    direct = (
        RESULTS
        / "final_model_sweep"
        / family
        / f"depth_{depth}"
        / f"seed_{seed}"
        / "history.csv"
    )

    if direct.is_file():
        return direct

    # Fallback: search data_reupload/results recursively.
    candidates = []

    for p in RESULTS.rglob("history.csv"):
        s = str(p)

        if family not in s:
            continue
        if f"depth_{depth}" not in s:
            continue
        if f"seed_{seed}" not in s:
            continue

        candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"Could not locate history.csv for "
            f"A={ansatz_layers}, depth={depth}, seed={seed}"
        )

    # Prefer paths associated with the final model sweep.
    candidates.sort(
        key=lambda p: (
            "final_model_sweep" not in str(p),
            len(str(p)),
            str(p),
        )
    )

    chosen = candidates[0]

    if len(candidates) > 1:
        print(
            f"[INFO] Multiple histories found for "
            f"A={ansatz_layers}, L={depth}, seed={seed}; "
            f"using {chosen}"
        )

    return chosen


def load_histories(
    ansatz_layers: int,
    depth: int,
) -> pd.DataFrame:

    frames = []

    for seed in SEEDS:
        path = find_history(ansatz_layers, depth, seed)

        print(
            f"[LOAD] A={ansatz_layers}, L={depth}, "
            f"seed={seed}: {path}"
        )

        df = pd.read_csv(path)

        required = {EPOCH, TRAIN, VAL}
        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(
                f"{path} missing columns: {sorted(missing)}"
            )

        df = df[[EPOCH, TRAIN, VAL]].copy()
        df["seed"] = seed

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def mean_sd(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:

    out = (
        df.groupby(EPOCH)[column]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(EPOCH)
    )

    out["std"] = out["std"].fillna(0.0)

    return out


# ======================================================================
# Preload all histories and export combined numerical table
# ======================================================================

histories = {}
summary_rows = []

for A in [1, 2]:
    for depth in DEPTHS:

        hist = load_histories(A, depth)
        histories[(A, depth)] = hist

        for split, column in [
            ("train", TRAIN),
            ("validation", VAL),
        ]:

            agg = mean_sd(hist, column)

            for _, r in agg.iterrows():
                summary_rows.append(
                    {
                        "ansatz_layers_per_block": A,
                        "depth": depth,
                        "split": split,
                        "epoch": int(r[EPOCH]),
                        "mean_scaled_mse": float(r["mean"]),
                        "sd_scaled_mse": float(r["std"]),
                    }
                )


summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUT / "ideal_one_vs_two_ansatz_training_histories.csv",
    index=False,
)


# ======================================================================
# Helper plotting function
# ======================================================================

def plot_family_history(
    ax,
    *,
    A: int,
    depth: int,
    label_prefix: bool = True,
):

    hist = histories[(A, depth)]

    train = mean_sd(hist, TRAIN)
    val = mean_sd(hist, VAL)

    # First line determines this family's default matplotlib color.
    train_line, = ax.plot(
        train[EPOCH],
        train["mean"],
        linewidth=2.0,
        label=(
            f"{DISPLAY[A]}: train"
            if label_prefix else "Train"
        ),
    )

    family_color = train_line.get_color()

    ax.fill_between(
        train[EPOCH].to_numpy(dtype=float),
        (train["mean"] - train["std"]).to_numpy(dtype=float),
        (train["mean"] + train["std"]).to_numpy(dtype=float),
        alpha=0.18,
        color=family_color,
    )

    ax.plot(
        val[EPOCH],
        val["mean"],
        linewidth=2.0,
        linestyle="--",
        color=family_color,
        label=(
            f"{DISPLAY[A]}: validation"
            if label_prefix else "Validation"
        ),
    )

    ax.fill_between(
        val[EPOCH].to_numpy(dtype=float),
        (val["mean"] - val["std"]).to_numpy(dtype=float),
        (val["mean"] + val["std"]).to_numpy(dtype=float),
        alpha=0.12,
        color=family_color,
    )


# ======================================================================
# Figure 1:
# Depth-1 replacement for the current main-text training-history figure
# ======================================================================

fig, ax = plt.subplots(figsize=(9.5, 5.8))

for A in [1, 2]:
    plot_family_history(
        ax,
        A=A,
        depth=1,
        label_prefix=True,
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Scaled MSE loss")
ax.set_title(
    "Ideal depth-1 training dynamics: "
    "one versus two ansatz layers per reuploading block"
)
ax.grid(alpha=0.25)
ax.legend(frameon=False, ncol=2)

fig.tight_layout()

fig.savefig(
    OUT / "ideal_one_vs_two_ansatz_depth1_training_history.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT / "ideal_one_vs_two_ansatz_depth1_training_history.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ======================================================================
# Figure 2:
# Full depth 1--6 comparison for appendix
# ======================================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16.0, 9.2),
)

axes = axes.ravel()

for ax, depth in zip(axes, DEPTHS):

    for A in [1, 2]:
        plot_family_history(
            ax,
            A=A,
            depth=depth,
            label_prefix=True,
        )

    ax.set_title(f"Reuploading depth $L={depth}$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Scaled MSE loss")
    ax.grid(alpha=0.25)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
)

fig.suptitle(
    "Ideal five-seed training dynamics across reuploading depth",
    fontsize=14,
    y=0.99,
)

fig.tight_layout(rect=[0, 0, 1, 0.945])

fig.savefig(
    OUT
    / "ideal_one_vs_two_ansatz_training_histories_all_depths.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT
    / "ideal_one_vs_two_ansatz_training_histories_all_depths.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("=" * 90)
print("TRAINING-HISTORY FIGURES CREATED")
print("=" * 90)

for p in sorted(OUT.glob("*")):
    print(p.name)

print("DONE.")
