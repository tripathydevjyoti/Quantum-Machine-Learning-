from __future__ import annotations

from pathlib import Path
import json

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

RESULTS_ROOT = (
    REPO
    / "data_reupload/results/final_model_sweep"
)

OUT = (
    REPO
    / "data_reupload/analysis/a1_vs_a2_multiseed/"
      "output/signed_spectral"
)

OUT.mkdir(parents=True, exist_ok=True)

FAMILIES = {
    "a1": "single_tanh_a1",
    "a2": "single_tanh_a2",
}

DEPTHS = [1, 2, 3, 4, 5, 6]
SEEDS = [42, 43, 44, 45, 46]

ABS_12_METRIC = "test_abs_12_step_band_error"
ABS_SHORT_METRIC = "test_abs_short_period_2_to_6_error"


# =====================================================================
# IO
# =====================================================================

def run_dir(family: str, depth: int, seed: int) -> Path:
    return (
        RESULTS_ROOT
        / FAMILIES[family]
        / f"depth_{depth}"
        / f"seed_{seed}"
    )


def read_summary(family: str, depth: int, seed: int) -> pd.Series:
    path = run_dir(family, depth, seed) / "result_summary.csv"

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if len(df) != 1:
        raise RuntimeError(
            f"Expected one summary row in {path}, got {len(df)}"
        )

    return df.iloc[0]


def read_predictions(
    family: str,
    depth: int,
    seed: int,
) -> dict[str, np.ndarray]:

    path = run_dir(family, depth, seed) / "predictions.npz"

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def first_key(
    data: dict[str, np.ndarray],
    candidates: list[str],
) -> str:

    for key in candidates:
        if key in data:
            return key

    raise KeyError(
        f"None of {candidates} found. Available keys: {list(data.keys())}"
    )


TRUE_KEYS = [
    "y_test_true_original",
    "test_true_original",
    "y_test_true",
]

PRED_KEYS = [
    "y_test_pred_original",
    "test_pred_original",
    "y_test_pred",
]


# =====================================================================
# Candidate FFT normalizations
#
# We deliberately test several conventions and identify which one
# reproduces the already-saved absolute spectral errors.
# =====================================================================

def spectrum(
    y: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:

    y = np.asarray(y, dtype=float).reshape(-1)

    fft = np.fft.rfft(y)
    freq = np.fft.rfftfreq(len(y), d=1.0)

    if mode.startswith("power"):
        quantity = np.abs(fft) ** 2
    elif mode.startswith("magnitude"):
        quantity = np.abs(fft)
    else:
        raise ValueError(mode)

    # Non-DC bins that correspond to finite periods.
    non_dc_freq = freq[1:]
    period = 1.0 / non_dc_freq

    if mode.endswith("non_dc_norm"):
        q = quantity[1:].copy()
        denom = q.sum()

    elif mode.endswith("all_norm"):
        denom = quantity.sum()
        q = quantity[1:].copy()

    else:
        raise ValueError(mode)

    if denom > 0:
        q /= denom

    # Ascending period makes every contiguous period band contiguous
    # in array index.
    order = np.argsort(period)

    return period[order], q[order]


MODES = [
    "power_non_dc_norm",
    "power_all_norm",
    "magnitude_non_dc_norm",
    "magnitude_all_norm",
]


# =====================================================================
# Read all 60 runs
# =====================================================================

records = []
spectra_by_mode: dict[str, list[dict]] = {
    mode: [] for mode in MODES
}

truth_reference = None
period_reference: dict[str, np.ndarray] = {}


for family in ["a1", "a2"]:
    for depth in DEPTHS:
        for seed in SEEDS:

            summary = read_summary(family, depth, seed)
            pred_data = read_predictions(family, depth, seed)

            true_key = first_key(pred_data, TRUE_KEYS)
            pred_key = first_key(pred_data, PRED_KEYS)

            truth = np.asarray(
                pred_data[true_key],
                dtype=float,
            ).reshape(-1)

            pred = np.asarray(
                pred_data[pred_key],
                dtype=float,
            ).reshape(-1)

            if truth_reference is None:
                truth_reference = truth.copy()
            elif not np.allclose(
                truth_reference,
                truth,
                atol=1e-10,
                rtol=0,
            ):
                raise RuntimeError(
                    "True test series differs between runs."
                )

            rec = {
                "family": family,
                "depth": depth,
                "seed": seed,
                ABS_12_METRIC: float(summary[ABS_12_METRIC]),
                ABS_SHORT_METRIC: float(summary[ABS_SHORT_METRIC]),
            }

            records.append(rec)

            for mode in MODES:

                periods_t, p_true = spectrum(truth, mode)
                periods_p, p_pred = spectrum(pred, mode)

                if not np.allclose(periods_t, periods_p):
                    raise RuntimeError("FFT period axes disagree.")

                if mode not in period_reference:
                    period_reference[mode] = periods_t.copy()
                elif not np.allclose(
                    period_reference[mode],
                    periods_t,
                ):
                    raise RuntimeError(
                        f"Period axis changed for mode {mode}"
                    )

                spectra_by_mode[mode].append(
                    {
                        "p_true": p_true,
                        "p_pred": p_pred,
                        "delta": p_pred - p_true,
                    }
                )


runs = pd.DataFrame(records)

print("=" * 110)
print("SIGNED SPECTRAL POWER DIAGNOSTIC")
print("=" * 110)
print("Verified balanced design: 60 A1/A2 runs.")
print(f"Test-series length: {len(truth_reference)}")
print()


# =====================================================================
# Infer the exact band used by the old absolute metric.
#
# For every candidate FFT convention and every contiguous period-band,
# test whether
#
#   abs(sum(P_pred - P_true over band))
#
# matches the stored metric over ALL 60 runs.
# =====================================================================

def infer_band(
    target_metric: str,
) -> dict:

    target = runs[target_metric].to_numpy(dtype=float)

    best = None

    for mode in MODES:

        periods = period_reference[mode]

        delta_matrix = np.stack(
            [
                item["delta"]
                for item in spectra_by_mode[mode]
            ],
            axis=0,
        )

        # Cumulative sums let us evaluate all contiguous bands cheaply.
        cumulative = np.concatenate(
            [
                np.zeros((len(delta_matrix), 1)),
                np.cumsum(delta_matrix, axis=1),
            ],
            axis=1,
        )

        n_bins = len(periods)

        for i in range(n_bins):
            for j in range(i, n_bins):

                signed = (
                    cumulative[:, j + 1]
                    -
                    cumulative[:, i]
                )

                predicted_abs = np.abs(signed)

                residual = predicted_abs - target

                rmse = float(
                    np.sqrt(
                        np.mean(residual ** 2)
                    )
                )

                max_abs = float(
                    np.max(
                        np.abs(residual)
                    )
                )

                candidate = {
                    "mode": mode,
                    "i": i,
                    "j": j,
                    "period_min": float(periods[i]),
                    "period_max": float(periods[j]),
                    "n_bins": int(j - i + 1),
                    "rmse_match_error": rmse,
                    "max_match_error": max_abs,
                }

                if (
                    best is None
                    or candidate["rmse_match_error"]
                    < best["rmse_match_error"]
                ):
                    best = candidate

    return best


band12 = infer_band(ABS_12_METRIC)
bandshort = infer_band(ABS_SHORT_METRIC)


print("Best reconstruction of existing 12-step absolute metric:")
print(json.dumps(band12, indent=2))

print()
print("Best reconstruction of existing 2–6-step absolute metric:")
print(json.dumps(bandshort, indent=2))
print()


# Require a very close reconstruction.
#
# If this fails, we STOP rather than silently using inconsistent bands.
TOL = 1e-6

if band12["max_match_error"] > TOL:
    raise RuntimeError(
        "Could not reproduce the stored 12-step spectral metric "
        f"to tolerance {TOL}. "
        f"Best max error={band12['max_match_error']}"
    )

if bandshort["max_match_error"] > TOL:
    raise RuntimeError(
        "Could not reproduce the stored 2–6-step spectral metric "
        f"to tolerance {TOL}. "
        f"Best max error={bandshort['max_match_error']}"
    )

print("VALIDATION PASSED:")
print(
    "The inferred signed bands reproduce BOTH stored absolute "
    f"metrics to max error < {TOL:g}."
)
print()


# =====================================================================
# Signed band power
# =====================================================================

def signed_band_values(
    band: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    mode = band["mode"]
    i = int(band["i"])
    j = int(band["j"])

    true_values = []
    pred_values = []
    signed_values = []

    for item in spectra_by_mode[mode]:

        p_true = item["p_true"]
        p_pred = item["p_pred"]

        true_band = float(
            p_true[i:j + 1].sum()
        )

        pred_band = float(
            p_pred[i:j + 1].sum()
        )

        true_values.append(true_band)
        pred_values.append(pred_band)
        signed_values.append(
            pred_band - true_band
        )

    return (
        np.asarray(true_values),
        np.asarray(pred_values),
        np.asarray(signed_values),
    )


true12, pred12, delta12 = signed_band_values(band12)
true_short, pred_short, delta_short = signed_band_values(bandshort)


runs["true_12_band_power"] = true12
runs["pred_12_band_power"] = pred12
runs["signed_12_band_power_error"] = delta12

runs["true_short_2_6_power"] = true_short
runs["pred_short_2_6_power"] = pred_short
runs["signed_short_2_6_power_error"] = delta_short


# Ratios are useful provided the true band has nonzero power.
runs["pred_over_true_12_band_power"] = (
    runs["pred_12_band_power"]
    /
    runs["true_12_band_power"]
)

runs["pred_over_true_short_2_6_power"] = (
    runs["pred_short_2_6_power"]
    /
    runs["true_short_2_6_power"]
)


runs.to_csv(
    OUT / "signed_spectral_all_runs.csv",
    index=False,
)


# =====================================================================
# Aggregate by architecture/depth
# =====================================================================

METRICS = [
    "pred_12_band_power",
    "signed_12_band_power_error",
    "pred_over_true_12_band_power",

    "pred_short_2_6_power",
    "signed_short_2_6_power_error",
    "pred_over_true_short_2_6_power",
]


agg = (
    runs
    .groupby(["family", "depth"])[METRICS]
    .agg(["mean", "std"])
)

agg.columns = [
    f"{metric}_{stat}"
    for metric, stat in agg.columns
]

agg = agg.reset_index()

agg.to_csv(
    OUT / "signed_spectral_depth_mean_sd.csv",
    index=False,
)


# =====================================================================
# Sign counts
# =====================================================================

sign_rows = []

for family in ["a1", "a2"]:
    for depth in DEPTHS:

        sub = runs[
            (runs["family"] == family)
            &
            (runs["depth"] == depth)
        ]

        for name, col in [
            ("12_step", "signed_12_band_power_error"),
            ("short_2_6", "signed_short_2_6_power_error"),
        ]:

            values = sub[col].to_numpy(dtype=float)

            sign_rows.append(
                {
                    "family": family,
                    "depth": depth,
                    "band": name,
                    "n_excess_power": int(np.sum(values > 0)),
                    "n_missing_power": int(np.sum(values < 0)),
                    "n_exact_zero": int(np.sum(values == 0)),
                }
            )


sign_counts = pd.DataFrame(sign_rows)

sign_counts.to_csv(
    OUT / "signed_spectral_seed_sign_counts.csv",
    index=False,
)


# =====================================================================
# Paired A2 - A1 signed-power changes
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

        paired_rows.append(
            {
                "depth": depth,
                "seed": seed,

                "a1_signed_12":
                    a1["signed_12_band_power_error"],

                "a2_signed_12":
                    a2["signed_12_band_power_error"],

                "a2_minus_a1_signed_12":
                    (
                        a2["signed_12_band_power_error"]
                        -
                        a1["signed_12_band_power_error"]
                    ),

                "a1_signed_short":
                    a1["signed_short_2_6_power_error"],

                "a2_signed_short":
                    a2["signed_short_2_6_power_error"],

                "a2_minus_a1_signed_short":
                    (
                        a2["signed_short_2_6_power_error"]
                        -
                        a1["signed_short_2_6_power_error"]
                    ),
            }
        )


paired = pd.DataFrame(paired_rows)

paired.to_csv(
    OUT / "signed_spectral_paired_a1_a2.csv",
    index=False,
)


# =====================================================================
# Figure 1: signed errors
#
# Positive = excess predicted normalized power
# Negative = missing predicted normalized power
# =====================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.5, 4.8),
    constrained_layout=True,
)


plot_specs = [
    (
        "signed_12_band_power_error",
        "Dominant seasonal band",
    ),
    (
        "signed_short_2_6_power_error",
        "Short-period 2–6-step band",
    ),
]


for ax, (metric, title) in zip(axes, plot_specs):

    for family, label in [
        ("a1", "1 ansatz layer"),
        ("a2", "2 ansatz layers"),
    ]:

        sub = (
            runs[
                runs["family"] == family
            ]
            .groupby("depth")[metric]
            .agg(["mean", "std"])
            .reset_index()
        )

        x = sub["depth"].to_numpy(dtype=float)
        mean = sub["mean"].to_numpy(dtype=float)
        sd = sub["std"].to_numpy(dtype=float)

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

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_title(title)
    ax.set_xlabel("Reuploading depth")
    ax.set_ylabel(
        r"$P_{\rm pred,band}-P_{\rm true,band}$"
    )
    ax.set_xticks(DEPTHS)
    ax.grid(alpha=0.25)


axes[0].legend(frameon=False)

fig.suptitle(
    "Signed spectral-power error: "
    "positive = excess power, negative = missing power\n"
    "Mean ± 1 SD across seeds 42–46",
    fontsize=13,
)

fig.savefig(
    OUT / "signed_spectral_power_vs_depth.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT / "signed_spectral_power_vs_depth.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# =====================================================================
# Figure 2: predicted / true power ratio
#
# ratio = 1 : correct
# ratio > 1 : excess
# ratio < 1 : missing
# =====================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.5, 4.8),
    constrained_layout=True,
)


ratio_specs = [
    (
        "pred_over_true_12_band_power",
        "Dominant seasonal band",
    ),
    (
        "pred_over_true_short_2_6_power",
        "Short-period 2–6-step band",
    ),
]


for ax, (metric, title) in zip(axes, ratio_specs):

    for family, label in [
        ("a1", "1 ansatz layer"),
        ("a2", "2 ansatz layers"),
    ]:

        sub = (
            runs[
                runs["family"] == family
            ]
            .groupby("depth")[metric]
            .agg(["mean", "std"])
            .reset_index()
        )

        x = sub["depth"].to_numpy(dtype=float)
        mean = sub["mean"].to_numpy(dtype=float)
        sd = sub["std"].to_numpy(dtype=float)

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

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_title(title)
    ax.set_xlabel("Reuploading depth")
    ax.set_ylabel("Predicted / true band power")
    ax.set_xticks(DEPTHS)
    ax.grid(alpha=0.25)


axes[0].legend(frameon=False)

fig.suptitle(
    "Spectral-band power reproduction",
    fontsize=13,
)

fig.savefig(
    OUT / "spectral_power_ratio_vs_depth.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT / "spectral_power_ratio_vs_depth.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# =====================================================================
# Print numerical summary
# =====================================================================

true_12_unique = runs["true_12_band_power"].unique()
true_short_unique = runs["true_short_2_6_power"].unique()

print("=" * 110)
print("TRUE-SERIES BAND POWERS")
print("=" * 110)

print(
    "True dominant-band normalized power:",
    float(true_12_unique[0]),
)

print(
    "True 2–6-step normalized power:",
    float(true_short_unique[0]),
)

print()


print("=" * 110)
print("SIGNED BAND-POWER RESULTS")
print("=" * 110)

print(
    "Interpretation: signed error > 0 means EXCESS predicted "
    "normalized power; signed error < 0 means MISSING power."
)

for depth in DEPTHS:

    print()
    print("-" * 110)
    print(f"DEPTH {depth}")
    print("-" * 110)

    for family in ["a1", "a2"]:

        sub = runs[
            (runs["family"] == family)
            &
            (runs["depth"] == depth)
        ]

        d12 = sub[
            "signed_12_band_power_error"
        ]

        ds = sub[
            "signed_short_2_6_power_error"
        ]

        r12 = sub[
            "pred_over_true_12_band_power"
        ]

        rs = sub[
            "pred_over_true_short_2_6_power"
        ]

        print(
            f"{family.upper()}  "
            f"ΔP12={d12.mean():+.4f} ± {d12.std(ddof=1):.4f}   "
            f"P12_pred/P12_true={r12.mean():.3f} ± {r12.std(ddof=1):.3f}   "
            f"ΔP2-6={ds.mean():+.4f} ± {ds.std(ddof=1):.4f}   "
            f"P2-6_pred/P2-6_true={rs.mean():.3f} ± {rs.std(ddof=1):.3f}"
        )

        print(
            f"      seed signs: "
            f"12-step excess={int((d12 > 0).sum())}/5, "
            f"missing={int((d12 < 0).sum())}/5; "
            f"2–6 excess={int((ds > 0).sum())}/5, "
            f"missing={int((ds < 0).sum())}/5"
        )


print()
print("=" * 110)
print("OUTPUTS")
print("=" * 110)

for path in sorted(OUT.glob("*")):
    print(path.name)

print()
print("DONE.")
