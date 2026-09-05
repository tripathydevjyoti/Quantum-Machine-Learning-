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

DATA_PATH = (
    PROJECT_ROOT
    / "data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv"
)


# ============================================================
# Canonical ordering
# ============================================================

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


# ============================================================
# Load manifest
# ============================================================

manifest = pd.read_csv(
    TABLE_DIR / "master_run_manifest.csv"
)

assert len(manifest) == 150


# ============================================================
# Helper
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


# ============================================================
# Load original time series
# ============================================================

raw = pd.read_csv(DATA_PATH)

if raw.shape[1] <= 4:
    raise RuntimeError(
        f"Expected at least 5 columns, got {raw.shape[1]}"
    )

value_col_name = raw.columns[4]

series = pd.to_numeric(
    raw.iloc[:, 4],
    errors="coerce",
).dropna().to_numpy(
    dtype=np.float64
)

if len(series) != 539:
    raise RuntimeError(
        f"Expected 539 non-missing observations, got {len(series)}"
    )

indices = np.arange(len(series))


# ============================================================
# Figure 01
# Complete dataset and forecasting split
# ============================================================

fig, ax = plt.subplots(
    figsize=(12.5, 4.8)
)

ax.plot(
    indices,
    series,
    linewidth=1.15,
)

# Initial observations are input history only.
ax.axvspan(
    0,
    13,
    alpha=0.10,
    label="Initial input history",
)

# Supervised target regions.
ax.axvspan(
    14,
    363,
    alpha=0.08,
    label="Training targets",
)

ax.axvspan(
    364,
    433,
    alpha=0.08,
    label="Validation targets",
)

ax.axvspan(
    434,
    538,
    alpha=0.08,
    label="Test targets",
)

ax.axvline(
    364,
    linestyle="--",
    linewidth=1,
    alpha=0.65,
)

ax.axvline(
    434,
    linestyle="--",
    linewidth=1,
    alpha=0.65,
)

ax.set_xlim(
    0,
    538,
)

ax.set_xlabel(
    "Absolute time index"
)

ax.set_ylabel(
    "Target (original units)"
)

ax.set_title(
    "Time series and chronological forecasting split"
)

ax.grid(
    alpha=0.18
)

ax.legend(
    frameon=False,
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.17),
)

fig.tight_layout()

save_figure(
    fig,
    "01_dataset_split",
)


# ============================================================
# Training-period characterization
# observations 0...363
# ============================================================

train_reference = series[:364]

centered = (
    train_reference
    -
    np.mean(train_reference)
)


def autocorrelation(x, max_lag):

    x = (
        x
        -
        np.mean(x)
    )

    denominator = np.sum(
        x * x
    )

    out = []

    for lag in range(
        max_lag + 1
    ):

        if lag == 0:
            value = 1.0
        else:
            value = (
                np.sum(
                    x[:-lag]
                    *
                    x[lag:]
                )
                /
                denominator
            )

        out.append(value)

    return np.asarray(out)


max_lag = 48

acf = autocorrelation(
    train_reference,
    max_lag,
)

lags = np.arange(
    max_lag + 1
)

fft_values = np.fft.rfft(
    centered
)

frequencies = np.fft.rfftfreq(
    len(centered),
    d=1.0,
)

power = np.abs(
    fft_values
) ** 2

# Remove DC.
frequencies_nonzero = frequencies[1:]
power_nonzero = power[1:]

power_share = (
    power_nonzero
    /
    np.sum(power_nonzero)
)

periods = (
    1.0
    /
    frequencies_nonzero
)

dominant_idx = int(
    np.argmax(power_share)
)

dominant_period = float(
    periods[dominant_idx]
)


# ============================================================
# Figure 02
# ACF + training spectrum
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.5, 4.7)
)

axes[0].stem(
    lags,
    acf,
    basefmt=" ",
)

for p in [12, 24, 36, 48]:
    axes[0].axvline(
        p,
        linestyle="--",
        linewidth=0.9,
        alpha=0.45,
    )

axes[0].axhline(
    0,
    linewidth=0.8,
    alpha=0.5,
)

axes[0].set_xlim(
    0,
    48,
)

axes[0].set_xlabel(
    "Lag"
)

axes[0].set_ylabel(
    "Autocorrelation"
)

axes[0].set_title(
    "Training-period autocorrelation"
)

axes[0].grid(
    alpha=0.16
)


# Sort periods ascending for display.
mask = (
    np.isfinite(periods)
    &
    (periods >= 2)
    &
    (periods <= 60)
)

x_period = periods[mask]
y_power = power_share[mask]

order = np.argsort(
    x_period
)

x_period = x_period[order]
y_power = y_power[order]

axes[1].plot(
    x_period,
    y_power,
    linewidth=1.3,
)

axes[1].axvline(
    12,
    linestyle="--",
    linewidth=1.2,
    alpha=0.65,
    label="Period 12",
)

axes[1].set_xlim(
    2,
    60,
)

axes[1].set_xlabel(
    "Period [steps]"
)

axes[1].set_ylabel(
    "Normalized non-DC power"
)

axes[1].set_title(
    "Training-period Fourier spectrum"
)

axes[1].grid(
    alpha=0.16
)

axes[1].legend(
    frameon=False
)

fig.suptitle(
    "Temporal structure of the training-period observations",
    y=1.02,
    fontsize=14,
)

fig.tight_layout()

save_figure(
    fig,
    "02_training_series_acf_fft",
)


# ============================================================
# Dataset characterization table
# ============================================================

metadata = pd.DataFrame([
    {
        "csv_path":
            str(DATA_PATH),

        "value_column_index":
            4,

        "value_column_name":
            value_col_name,

        "n_observations":
            len(series),

        "cdf_reference_n":
            len(train_reference),

        "cdf_reference_start":
            0,

        "cdf_reference_end":
            363,

        "acf_lag_1":
            float(acf[1]),

        "acf_lag_6":
            float(acf[6]),

        "acf_lag_12":
            float(acf[12]),

        "acf_lag_24":
            float(acf[24]),

        "dominant_training_fft_period":
            dominant_period,
    }
])

metadata.to_csv(
    TABLE_DIR
    / "dataset_characterization.csv",
    index=False,
)


# ============================================================
# Reference test truth and indices
# ============================================================

reference_run = manifest.iloc[0]

reference_npz = np.load(
    reference_run[
        "predictions_path"
    ],
    allow_pickle=True,
)

test_truth = np.asarray(
    reference_npz[
        "y_test_true_original"
    ],
    dtype=np.float64,
)

test_indices = np.asarray(
    reference_npz[
        "test_target_indices"
    ],
    dtype=np.int64,
)

if len(test_truth) != 105:
    raise RuntimeError(
        "Unexpected test truth length"
    )


# ============================================================
# Figure 06
# Held-out trajectories: all five encodings
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(13.0, 11.0),
    sharex=True,
    sharey=True,
)

for depth, ax in zip(
    range(1, 7),
    axes.flat,
):

    ax.plot(
        test_indices,
        test_truth,
        linewidth=1.8,
        label="True",
    )

    for enc in ENCODINGS:

        d = manifest[
            (manifest["encoding"] == enc)
            &
            (manifest["depth"] == depth)
        ].sort_values(
            "seed"
        )

        if len(d) != 5:
            raise RuntimeError(
                f"Expected five runs for {enc}, L={depth}"
            )

        preds = []

        for _, row in d.iterrows():

            z = np.load(
                row[
                    "predictions_path"
                ],
                allow_pickle=True,
            )

            y_pred = np.asarray(
                z[
                    "y_test_pred_original"
                ],
                dtype=np.float64,
            )

            if len(y_pred) != 105:
                raise RuntimeError(
                    f"Prediction length mismatch: "
                    f"{enc}, L={depth}"
                )

            preds.append(
                y_pred
            )

        preds = np.stack(
            preds,
            axis=0,
        )

        mean_pred = np.mean(
            preds,
            axis=0,
        )

        sd_pred = np.std(
            preds,
            axis=0,
            ddof=1,
        )

        line, = ax.plot(
            test_indices,
            mean_pred,
            linewidth=1.25,
            label=LABELS[enc],
        )

        ax.fill_between(
            test_indices,
            mean_pred - sd_pred,
            mean_pred + sd_pred,
            alpha=0.08,
            color=line.get_color(),
        )

    ax.set_title(
        f"Depth {depth}"
    )

    ax.grid(
        alpha=0.15
    )

    ax.set_xlim(
        test_indices[0],
        test_indices[-1],
    )

for ax in axes[:, 0]:
    ax.set_ylabel(
        "Target (original units)"
    )

for ax in axes[-1, :]:
    ax.set_xlabel(
        "Test target index"
    )

handles, legend_labels = (
    axes[0, 0]
    .get_legend_handles_labels()
)

fig.legend(
    handles,
    legend_labels,
    loc="upper center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 0.995),
)

fig.suptitle(
    "Held-out ideal test forecasts across input encodings",
    y=1.025,
    fontsize=14,
)

fig.text(
    0.5,
    0.006,
    "Prediction curves and bands show five-seed mean ± 1 SD",
    ha="center",
    fontsize=10,
)

fig.tight_layout(
    rect=[0, 0.025, 1, 0.955]
)

save_figure(
    fig,
    "06_test_predictions_all_encodings",
)


# ============================================================
# Figure 11
# Full normalized test spectra
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(13.0, 11.0),
    sharex=True,
)

for depth, ax in zip(
    range(1, 7),
    axes.flat,
):

    true_plotted = False

    for enc in ENCODINGS:

        d = manifest[
            (manifest["encoding"] == enc)
            &
            (manifest["depth"] == depth)
        ].sort_values(
            "seed"
        )

        pred_spectra = []
        common_period = None

        for _, row in d.iterrows():

            fft_df = pd.read_csv(
                row[
                    "fft_power_path"
                ]
            )

            test_pred = (
                fft_df[
                    (fft_df["split"] == "test")
                    &
                    (fft_df["series"] == "pred")
                ]
                .copy()
                .sort_values("frequency")
            )

            test_true = (
                fft_df[
                    (fft_df["split"] == "test")
                    &
                    (fft_df["series"] == "true")
                ]
                .copy()
                .sort_values("frequency")
            )

            pred_mask = (
                np.isfinite(
                    test_pred[
                        "period_steps"
                    ].to_numpy()
                )
                &
                (
                    test_pred[
                        "frequency"
                    ].to_numpy()
                    >
                    0
                )
            )

            true_mask = (
                np.isfinite(
                    test_true[
                        "period_steps"
                    ].to_numpy()
                )
                &
                (
                    test_true[
                        "frequency"
                    ].to_numpy()
                    >
                    0
                )
            )

            pred_period = (
                test_pred.loc[
                    pred_mask,
                    "period_steps"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            pred_power = (
                test_pred.loc[
                    pred_mask,
                    "power_share_non_dc"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            if common_period is None:
                common_period = pred_period
            else:
                if not np.allclose(
                    common_period,
                    pred_period,
                    atol=1e-12,
                    rtol=0,
                ):
                    raise RuntimeError(
                        f"FFT grid mismatch for {enc}, L={depth}"
                    )

            pred_spectra.append(
                pred_power
            )

            if not true_plotted:

                true_period = (
                    test_true.loc[
                        true_mask,
                        "period_steps"
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                true_power = (
                    test_true.loc[
                        true_mask,
                        "power_share_non_dc"
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                order = np.argsort(
                    true_period
                )

                ax.plot(
                    true_period[order],
                    true_power[order],
                    linewidth=1.8,
                    label="True",
                )

                true_plotted = True

        pred_spectra = np.stack(
            pred_spectra,
            axis=0,
        )

        mean_power = np.mean(
            pred_spectra,
            axis=0,
        )

        sd_power = np.std(
            pred_spectra,
            axis=0,
            ddof=1,
        )

        order = np.argsort(
            common_period
        )

        x = common_period[order]
        y = mean_power[order]
        sd = sd_power[order]

        line, = ax.plot(
            x,
            y,
            linewidth=1.25,
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

    ax.axvline(
        12,
        linestyle="--",
        linewidth=0.9,
        alpha=0.50,
    )

    ax.set_xlim(
        2,
        105,
    )

    ax.set_title(
        f"Depth {depth}"
    )

    ax.grid(
        alpha=0.15
    )

for ax in axes[:, 0]:
    ax.set_ylabel(
        "Normalized non-DC power"
    )

for ax in axes[-1, :]:
    ax.set_xlabel(
        "Period [steps]"
    )

handles, legend_labels = (
    axes[0, 0]
    .get_legend_handles_labels()
)

fig.legend(
    handles,
    legend_labels,
    loc="upper center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 0.995),
)

fig.suptitle(
    "Normalized test-set Fourier spectra across input encodings",
    y=1.025,
    fontsize=14,
)

fig.text(
    0.5,
    0.006,
    "Prediction spectra show five-seed mean ± 1 SD; dashed line marks period 12",
    ha="center",
    fontsize=10,
)

fig.tight_layout(
    rect=[0, 0.025, 1, 0.955]
)

save_figure(
    fig,
    "11_full_fft_spectra_all_encodings",
)


# ============================================================
# Output
# ============================================================

print("=" * 120)
print("DATASET / TRAJECTORY / SPECTRAL FIGURE GENERATION")
print("=" * 120)

print()
print(
    "Dataset value column:",
    value_col_name,
)

print(
    "Number of non-missing observations:",
    len(series),
)

print(
    "CDF/training reference observations:",
    len(train_reference),
)

print()
print(
    f"ACF lag 1 : {acf[1]:.6f}"
)

print(
    f"ACF lag 6 : {acf[6]:.6f}"
)

print(
    f"ACF lag 12: {acf[12]:.6f}"
)

print(
    f"ACF lag 24: {acf[24]:.6f}"
)

print()
print(
    f"Dominant training FFT period: "
    f"{dominant_period:.6f} steps"
)

print()
print("Generated:")

for stem in [
    "01_dataset_split",
    "02_training_series_acf_fft",
    "06_test_predictions_all_encodings",
    "11_full_fft_spectra_all_encodings",
]:
    print(
        f"  {stem}.pdf"
    )
    print(
        f"  {stem}.png"
    )

print()
print(
    "Dataset table:"
)

print(
    "  dataset_characterization.csv"
)

print()
print("=" * 120)
print("DATASET / TRAJECTORY / SPECTRAL STATUS: PASS")
print("=" * 120)
