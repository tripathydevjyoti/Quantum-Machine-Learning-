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

INPUT = (
    REPO
    / "data_reupload/analysis/a1_vs_a2_multiseed/output/tables/"
      "a1_vs_a2_all_run_metrics.csv"
)

OUT = (
    REPO
    / "data_reupload/analysis/a1_vs_a2_multiseed/output/failure_modes"
)

OUT.mkdir(parents=True, exist_ok=True)

DEPTHS = [1, 2, 3, 4, 5, 6]
SEEDS = [42, 43, 44, 45, 46]

METRICS = {
    "amp_ratio": "test_pred_std_over_true_std",
    "bias": "test_bias",
    "seasonal_error": "test_abs_12_step_band_error",
    "short_error": "test_abs_short_period_2_to_6_error",
    "fft": "test_fft_spectral_cosine_similarity",
}


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

if not INPUT.is_file():
    raise FileNotFoundError(INPUT)

df = pd.read_csv(INPUT)

required = [
    "family",
    "depth",
    "seed",
    *METRICS.values(),
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Available: {list(df.columns)}"
    )

df["depth"] = pd.to_numeric(df["depth"])
df["seed"] = pd.to_numeric(df["seed"])

for col in METRICS.values():
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )


# ---------------------------------------------------------------------
# Validate balanced design
# ---------------------------------------------------------------------

expected = {
    (family, depth, seed)
    for family in ["a1", "a2"]
    for depth in DEPTHS
    for seed in SEEDS
}

observed = set(
    zip(
        df["family"],
        df["depth"],
        df["seed"],
    )
)

missing_runs = expected - observed

if missing_runs:
    raise RuntimeError(
        f"Missing balanced runs: {sorted(missing_runs)}"
    )

print("Verified 60 balanced A1/A2 runs.")
print()


# ---------------------------------------------------------------------
# Add physically interpretable error quantities
# ---------------------------------------------------------------------

# Perfect amplitude ratio = 1.
df["amplitude_ratio_error"] = np.abs(
    df[METRICS["amp_ratio"]] - 1.0
)

# Perfect bias = 0.
df["absolute_bias"] = np.abs(
    df[METRICS["bias"]]
)


# ---------------------------------------------------------------------
# Aggregate mean ± SD
# ---------------------------------------------------------------------

analysis_metrics = [
    METRICS["amp_ratio"],
    "amplitude_ratio_error",
    METRICS["bias"],
    "absolute_bias",
    METRICS["seasonal_error"],
    METRICS["short_error"],
    METRICS["fft"],
]

agg = (
    df
    .groupby(["family", "depth"])[analysis_metrics]
    .agg(["mean", "std"])
)

agg.columns = [
    f"{metric}_{stat}"
    for metric, stat in agg.columns
]

agg = agg.reset_index()

agg.to_csv(
    OUT / "a1_vs_a2_failure_mode_mean_sd.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Paired A2 - A1 differences
# ---------------------------------------------------------------------

paired_rows = []

for depth in DEPTHS:

    for seed in SEEDS:

        a1 = df[
            (df["family"] == "a1")
            & (df["depth"] == depth)
            & (df["seed"] == seed)
        ].iloc[0]

        a2 = df[
            (df["family"] == "a2")
            & (df["depth"] == depth)
            & (df["seed"] == seed)
        ].iloc[0]

        paired_rows.append(
            {
                "depth": depth,
                "seed": seed,

                # Raw amplitude ratio:
                "a1_amp_ratio":
                    a1[METRICS["amp_ratio"]],
                "a2_amp_ratio":
                    a2[METRICS["amp_ratio"]],
                "delta_amp_ratio_a2_minus_a1":
                    a2[METRICS["amp_ratio"]]
                    - a1[METRICS["amp_ratio"]],

                # Distance from ideal amplitude ratio 1:
                "a1_amp_error":
                    a1["amplitude_ratio_error"],
                "a2_amp_error":
                    a2["amplitude_ratio_error"],
                "delta_amp_error_a2_minus_a1":
                    a2["amplitude_ratio_error"]
                    - a1["amplitude_ratio_error"],

                # Signed bias:
                "a1_bias":
                    a1[METRICS["bias"]],
                "a2_bias":
                    a2[METRICS["bias"]],
                "delta_bias_a2_minus_a1":
                    a2[METRICS["bias"]]
                    - a1[METRICS["bias"]],

                # Magnitude of bias:
                "a1_abs_bias":
                    a1["absolute_bias"],
                "a2_abs_bias":
                    a2["absolute_bias"],
                "delta_abs_bias_a2_minus_a1":
                    a2["absolute_bias"]
                    - a1["absolute_bias"],

                # Seasonal:
                "a1_12step_error":
                    a1[METRICS["seasonal_error"]],
                "a2_12step_error":
                    a2[METRICS["seasonal_error"]],
                "delta_12step_error_a2_minus_a1":
                    a2[METRICS["seasonal_error"]]
                    - a1[METRICS["seasonal_error"]],

                # Short-period:
                "a1_short_error":
                    a1[METRICS["short_error"]],
                "a2_short_error":
                    a2[METRICS["short_error"]],
                "delta_short_error_a2_minus_a1":
                    a2[METRICS["short_error"]]
                    - a1[METRICS["short_error"]],

                # Global spectral similarity:
                "a1_fft":
                    a1[METRICS["fft"]],
                "a2_fft":
                    a2[METRICS["fft"]],
                "delta_fft_a2_minus_a1":
                    a2[METRICS["fft"]]
                    - a1[METRICS["fft"]],
            }
        )


paired = pd.DataFrame(paired_rows)

paired.to_csv(
    OUT / "a1_vs_a2_failure_mode_paired.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Aggregate paired differences
# ---------------------------------------------------------------------

delta_columns = [
    "delta_amp_ratio_a2_minus_a1",
    "delta_amp_error_a2_minus_a1",
    "delta_bias_a2_minus_a1",
    "delta_abs_bias_a2_minus_a1",
    "delta_12step_error_a2_minus_a1",
    "delta_short_error_a2_minus_a1",
    "delta_fft_a2_minus_a1",
]

paired_summary = (
    paired
    .groupby("depth")[delta_columns]
    .agg(["mean", "std"])
)

paired_summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in paired_summary.columns
]

paired_summary = paired_summary.reset_index()

paired_summary.to_csv(
    OUT / "a1_vs_a2_failure_mode_paired_summary.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Figure: four primary failure-mode diagnostics
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12.5, 8.5),
    constrained_layout=True,
)

axes = axes.ravel()


plot_specs = [
    (
        METRICS["amp_ratio"],
        "Prediction amplitude ratio",
        r"$\sigma_{\hat{y}}/\sigma_y$",
        1.0,
    ),
    (
        METRICS["bias"],
        "Prediction bias",
        "Bias [original units]",
        0.0,
    ),
    (
        METRICS["seasonal_error"],
        "12-step spectral-band error",
        "Absolute normalized-power error",
        None,
    ),
    (
        METRICS["short_error"],
        "Short-period (2–6 step) spectral error",
        "Absolute normalized-power error",
        None,
    ),
]


for ax, (
    metric,
    title,
    ylabel,
    reference,
) in zip(axes, plot_specs):

    for family, label in [
        ("a1", "1 ansatz layer"),
        ("a2", "2 ansatz layers"),
    ]:

        subset = (
            agg[
                agg["family"] == family
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

    if reference is not None:
        ax.axhline(
            reference,
            linestyle="--",
            linewidth=1.2,
        )

    ax.set_title(title)
    ax.set_xlabel("Reuploading depth")
    ax.set_ylabel(ylabel)
    ax.set_xticks(DEPTHS)
    ax.grid(alpha=0.25)


axes[0].legend(
    frameon=False,
)

fig.suptitle(
    "Diagnosing deep-QNN failure modes: A1 versus A2\n"
    "Mean ± 1 SD across seeds 42–46",
    fontsize=14,
)

fig.savefig(
    OUT / "a1_vs_a2_failure_modes.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT / "a1_vs_a2_failure_modes.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# Print detailed compact summary
# ---------------------------------------------------------------------

print("=" * 108)
print("FAILURE-MODE DIAGNOSTICS")
print("=" * 108)
print(
    "Amplitude ratio: ideal value = 1. "
    "Values <1 indicate variance/amplitude compression."
)
print(
    "Bias: ideal value = 0. "
    "Positive = systematic overprediction; negative = underprediction."
)
print(
    "12-step and 2–6-step errors: lower is better."
)
print()


for depth in DEPTHS:

    print("-" * 108)
    print(f"DEPTH {depth}")
    print("-" * 108)

    for family in ["a1", "a2"]:

        r = agg[
            (agg["family"] == family)
            & (agg["depth"] == depth)
        ].iloc[0]

        print(
            f"{family.upper()}  "
            f"amp_ratio="
            f"{r[METRICS['amp_ratio'] + '_mean']:.4f}"
            f" ± "
            f"{r[METRICS['amp_ratio'] + '_std']:.4f}"
            f"   "
            f"bias="
            f"{r[METRICS['bias'] + '_mean']:.4f}"
            f" ± "
            f"{r[METRICS['bias'] + '_std']:.4f}"
            f"   "
            f"E12="
            f"{r[METRICS['seasonal_error'] + '_mean']:.4f}"
            f" ± "
            f"{r[METRICS['seasonal_error'] + '_std']:.4f}"
            f"   "
            f"E2-6="
            f"{r[METRICS['short_error'] + '_mean']:.4f}"
            f" ± "
            f"{r[METRICS['short_error'] + '_std']:.4f}"
            f"   "
            f"FFT="
            f"{r[METRICS['fft'] + '_mean']:.4f}"
            f" ± "
            f"{r[METRICS['fft'] + '_std']:.4f}"
        )

    d = paired_summary[
        paired_summary["depth"] == depth
    ].iloc[0]

    print()
    print(
        "A2 - A1 paired mean changes:  "
        f"Δ|amp-1|="
        f"{d['delta_amp_error_a2_minus_a1_mean']:+.4f}   "
        f"Δ|bias|="
        f"{d['delta_abs_bias_a2_minus_a1_mean']:+.4f}   "
        f"ΔE12="
        f"{d['delta_12step_error_a2_minus_a1_mean']:+.4f}   "
        f"ΔE2-6="
        f"{d['delta_short_error_a2_minus_a1_mean']:+.4f}   "
        f"ΔFFT="
        f"{d['delta_fft_a2_minus_a1_mean']:+.4f}"
    )


print()
print("=" * 108)
print("INTERPRETATION GUIDE")
print("=" * 108)

print("""
For A2 - A1:

Δ|amp-1| > 0
    A2 moves farther away from the correct prediction amplitude.

Δ|bias| > 0
    A2 develops a larger systematic level error.

ΔE12 > 0
    A2 reproduces the dominant seasonal band less accurately.

ΔE2-6 > 0
    A2 reproduces short-period spectral structure less accurately.

ΔFFT < 0
    A2 has poorer global normalized spectral similarity.
""")

print("Outputs written to:")
print(OUT)
