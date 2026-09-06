#!/usr/bin/env python3

import json
import math
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch


# ============================================================
# Paths / constants
# ============================================================

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


PROJECT_ROOT = _resolve_project_root()

MODEL5_ROOT = (
    PROJECT_ROOT
    / "data_reupload/results/encoding_sweep/"
      "learnable_seasonal_cdf_rz_tanh_a1"
)

MODEL4_ROOT = (
    PROJECT_ROOT
    / "data_reupload/results/encoding_sweep/"
      "learnable_seasonal_cdf_tanh_a1"
)

OUT = (
    PROJECT_ROOT
    / "data_reupload/analysis/four_encoding_master/"
      "output/model5_validation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

DEPTHS = list(range(1, 7))
SEEDS = list(range(42, 47))

EXPECTED_RUNS = 30

EXPECTED_TRAIN_IDX = np.arange(14, 364)
EXPECTED_VAL_IDX = np.arange(364, 434)
EXPECTED_TEST_IDX = np.arange(434, 539)

PHASE_BOUND = math.pi / 12.0
FIXED_GAP = math.pi / 6.0


# ============================================================
# Helpers
# ============================================================

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite(a, name):
    a = np.asarray(a)

    require(
        np.all(np.isfinite(a)),
        f"Non-finite values in {name}"
    )


def max_abs_error(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    return float(
        np.max(
            np.abs(a - b)
        )
    )


def rmse(y, p):
    return float(
        np.sqrt(
            np.mean(
                (p - y) ** 2
            )
        )
    )


def mae(y, p):
    return float(
        np.mean(
            np.abs(
                p - y
            )
        )
    )


def pearson(y, p):

    y = np.asarray(y)
    p = np.asarray(p)

    if np.std(y) == 0 or np.std(p) == 0:
        return np.nan

    return float(
        np.corrcoef(
            y,
            p,
        )[0, 1]
    )


def spearman(y, p):

    # pandas rank handling gives deterministic average ranks.
    ys = pd.Series(
        np.asarray(y)
    )

    ps = pd.Series(
        np.asarray(p)
    )

    return float(
        ys.rank(
            method="average"
        ).corr(
            ps.rank(
                method="average"
            )
        )
    )


def kge(y, p):

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    r = pearson(y, p)

    std_y = np.std(
        y,
        ddof=0,
    )

    std_p = np.std(
        p,
        ddof=0,
    )

    mean_y = np.mean(y)
    mean_p = np.mean(p)

    if (
        not np.isfinite(r)
        or std_y == 0
        or mean_y == 0
    ):
        return np.nan

    alpha = std_p / std_y
    beta = mean_p / mean_y

    return float(
        1.0
        -
        np.sqrt(
            (r - 1.0) ** 2
            +
            (alpha - 1.0) ** 2
            +
            (beta - 1.0) ** 2
        )
    )


def nse(y, p):

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    denominator = np.sum(
        (
            y
            -
            np.mean(y)
        ) ** 2
    )

    if denominator == 0:
        return np.nan

    return float(
        1.0
        -
        np.sum(
            (p - y) ** 2
        )
        /
        denominator
    )


def amplitude_ratio(y, p):

    sy = np.std(
        y,
        ddof=0,
    )

    sp = np.std(
        p,
        ddof=0,
    )

    if sy == 0:
        return np.nan

    return float(
        sp / sy
    )


def fft_cosine_similarity(y, p):
    """
    Fallback only.

    We preferentially use the project's saved FFT cosine metric.
    This fallback computes cosine similarity between centered
    one-sided FFT magnitudes, excluding DC.
    """

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    y0 = y - np.mean(y)
    p0 = p - np.mean(p)

    fy = np.abs(
        np.fft.rfft(y0)
    )[1:]

    fp = np.abs(
        np.fft.rfft(p0)
    )[1:]

    denom = (
        np.linalg.norm(fy)
        *
        np.linalg.norm(fp)
    )

    if denom == 0:
        return np.nan

    return float(
        np.dot(fy, fp)
        /
        denom
    )


def lookup_value(
    row,
    aliases,
    default=np.nan,
):

    for key in aliases:

        if key in row.index:

            value = row[key]

            try:
                value = float(value)
            except Exception:
                continue

            if np.isfinite(value):
                return value

    return default


def saved_test_fft(run_dir, y, p):

    path = (
        run_dir
        /
        "fft_summary_by_split.csv"
    )

    if path.exists():

        df = pd.read_csv(path)

        if "split" in df.columns:

            tmp = df.loc[
                df["split"].astype(str)
                ==
                "test"
            ]

            if len(tmp) == 1:

                row = tmp.iloc[0]

                value = lookup_value(
                    row,
                    [
                        "fft_spectral_cosine_similarity",
                    ],
                )

                if np.isfinite(value):
                    return value, "saved"

    raise RuntimeError(
        f"Authoritative FFT metric "
        f"'fft_spectral_cosine_similarity' "
        f"not found for run: {run_dir}"
    )


def load_predictions(run_dir):

    p = np.load(
        run_dir
        /
        "predictions.npz"
    )

    required = [
        "y_train_true_original",
        "y_train_pred_original",
        "y_val_true_original",
        "y_val_pred_original",
        "y_test_true_original",
        "y_test_pred_original",
        "y_train_true_scaled",
        "y_val_true_scaled",
        "y_test_true_scaled",
        "train_target_indices",
        "val_target_indices",
        "test_target_indices",
    ]

    for key in required:

        require(
            key in p.files,
            f"{run_dir}: predictions.npz missing {key}"
        )

        finite(
            p[key],
            f"{run_dir}:{key}",
        )

    return p


def metrics_from_predictions(
    pred,
    run_dir,
):

    y_train = pred[
        "y_train_true_original"
    ]

    p_train = pred[
        "y_train_pred_original"
    ]

    y_val = pred[
        "y_val_true_original"
    ]

    p_val = pred[
        "y_val_pred_original"
    ]

    y_test = pred[
        "y_test_true_original"
    ]

    p_test = pred[
        "y_test_pred_original"
    ]

    fft_value, fft_source = saved_test_fft(
        run_dir,
        y_test,
        p_test,
    )

    return {
        "train_rmse":
            rmse(
                y_train,
                p_train,
            ),

        "val_rmse":
            rmse(
                y_val,
                p_val,
            ),

        "test_rmse":
            rmse(
                y_test,
                p_test,
            ),

        "test_mae":
            mae(
                y_test,
                p_test,
            ),

        "test_pearson":
            pearson(
                y_test,
                p_test,
            ),

        "test_spearman":
            spearman(
                y_test,
                p_test,
            ),

        "test_kge":
            kge(
                y_test,
                p_test,
            ),

        "test_nse":
            nse(
                y_test,
                p_test,
            ),

        "test_fft_cosine":
            fft_value,

        "fft_metric_source":
            fft_source,

        "test_amplitude_ratio":
            amplitude_ratio(
                y_test,
                p_test,
            ),
    }


def load_checkpoint(run_dir):

    path = (
        run_dir
        /
        "best_model.pt"
    )

    try:

        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        obj = torch.load(
            path,
            map_location="cpu",
        )

    return obj.get(
        "model_state_dict",
        obj,
    )


# ============================================================
# Validation accumulators
# ============================================================

run_rows = []
paired_rows = []
encoder_rows = []
layer_rows = []

max_truth_error = 0.0
max_checkpoint_npz_error = 0.0
max_phase_transform_error = 0.0
max_fixed_phase_error = 0.0
max_unwrapped_phase_error = 0.0
max_wrapped_phase_error = 0.0

fft_fallback_count = 0

validated = 0


print("=" * 130)
print("MODEL 5 — 30-RUN VALIDATION + AGGREGATION")
print("=" * 130)

print(
    "Model 5 root:",
    MODEL5_ROOT,
)

print(
    "Model 4 reference:",
    MODEL4_ROOT,
)

print()


# ============================================================
# Main loop
# ============================================================

for depth in DEPTHS:

    for seed in SEEDS:

        run5 = (
            MODEL5_ROOT
            /
            f"depth_{depth}"
            /
            f"seed_{seed}"
        )

        run4 = (
            MODEL4_ROOT
            /
            f"depth_{depth}"
            /
            f"seed_{seed}"
        )

        print(
            f"Validating L={depth}, seed={seed} ..."
        )

        # ----------------------------------------------------
        # Required files
        # ----------------------------------------------------

        required_files = [
            "config.json",
            "best_model.pt",
            "predictions.npz",
            "metrics_by_split.csv",
            "result_summary.csv",
            "learned_encoding_parameters.npz",
            "learned_encoding_parameters.csv",
            "learned_encoding_layer_summary.csv",
            "learnable_seasonal_encoding_reference.npz",
            "encoding_metadata.json",
        ]

        missing = [
            f
            for f in required_files
            if not (
                run5
                /
                f
            ).is_file()
        ]

        require(
            not missing,
            f"{run5}: missing artifacts: {missing}"
        )

        require(
            (
                run4
                /
                "predictions.npz"
            ).is_file(),
            f"Missing model-4 reference predictions: {run4}"
        )

        # ----------------------------------------------------
        # Configuration / provenance
        # ----------------------------------------------------

        with open(
            run5
            /
            "config.json"
        ) as f:

            cfg = json.load(f)

        require(
            int(
                cfg["depth"]
            )
            ==
            depth,
            f"{run5}: depth mismatch"
        )

        require(
            int(
                cfg["seed"]
            )
            ==
            seed,
            f"{run5}: seed mismatch"
        )

        require(
            cfg.get(
                "encoding"
            )
            ==
            "learnable_seasonal_cdf_rz",
            f"{run5}: wrong encoding metadata"
        )

        require(
            cfg.get(
                "direct_parent_encoding"
            )
            ==
            "learnable_seasonal_cdf",
            f"{run5}: wrong direct parent"
        )

        require(
            int(
                cfg[
                    "variational_parameter_count"
                ]
            )
            ==
            42 * depth,
            f"{run5}: variational parameter count mismatch"
        )

        require(
            int(
                cfg[
                    "encoding_parameter_count"
                ]
            )
            ==
            36 * depth,
            f"{run5}: encoding parameter count mismatch"
        )

        require(
            int(
                cfg[
                    "classical_head_parameter_count"
                ]
            )
            ==
            15,
            f"{run5}: head parameter count mismatch"
        )

        require(
            int(
                cfg[
                    "total_parameter_count"
                ]
            )
            ==
            78 * depth + 15,
            f"{run5}: total parameter count mismatch"
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        p5 = load_predictions(
            run5
        )

        p4 = load_predictions(
            run4
        )

        require(
            np.array_equal(
                p5[
                    "train_target_indices"
                ],
                EXPECTED_TRAIN_IDX,
            ),
            f"{run5}: incorrect training indices"
        )

        require(
            np.array_equal(
                p5[
                    "val_target_indices"
                ],
                EXPECTED_VAL_IDX,
            ),
            f"{run5}: incorrect validation indices"
        )

        require(
            np.array_equal(
                p5[
                    "test_target_indices"
                ],
                EXPECTED_TEST_IDX,
            ),
            f"{run5}: incorrect test indices"
        )

        truth_keys = [
            "y_train_true_scaled",
            "y_val_true_scaled",
            "y_test_true_scaled",
            "y_train_true_original",
            "y_val_true_original",
            "y_test_true_original",
        ]

        this_truth_error = 0.0

        for key in truth_keys:

            err = max_abs_error(
                p5[key],
                p4[key],
            )

            this_truth_error = max(
                this_truth_error,
                err,
            )

            require(
                err < 1e-12,
                f"{run5}: truth mismatch vs model 4 for {key}: {err}"
            )

        for key in [
            "train_target_indices",
            "val_target_indices",
            "test_target_indices",
        ]:

            require(
                np.array_equal(
                    p5[key],
                    p4[key],
                ),
                f"{run5}: index mismatch vs model 4 for {key}"
            )

        max_truth_error = max(
            max_truth_error,
            this_truth_error,
        )

        # ----------------------------------------------------
        # Forecast metrics
        # ----------------------------------------------------

        m5 = metrics_from_predictions(
            p5,
            run5,
        )

        m4 = metrics_from_predictions(
            p4,
            run4,
        )

        if (
            m5[
                "fft_metric_source"
            ]
            ==
            "fallback"
        ):
            fft_fallback_count += 1

        for name, value in m5.items():

            if name == "fft_metric_source":
                continue

            require(
                np.isfinite(value),
                f"{run5}: non-finite metric {name}"
            )

        # ----------------------------------------------------
        # Cross-check saved RMSE/MAE if available
        # ----------------------------------------------------

        metric_df = pd.read_csv(
            run5
            /
            "metrics_by_split.csv"
        )

        require(
            "split"
            in metric_df.columns,
            f"{run5}: metrics_by_split missing split column"
        )

        test_metric_row = metric_df.loc[
            metric_df[
                "split"
            ].astype(str)
            ==
            "test"
        ]

        require(
            len(
                test_metric_row
            )
            ==
            1,
            f"{run5}: expected one test metrics row"
        )

        test_metric_row = (
            test_metric_row.iloc[0]
        )

        saved_rmse = lookup_value(
            test_metric_row,
            [
                "rmse",
                "RMSE",
            ],
        )

        if np.isfinite(
            saved_rmse
        ):

            require(
                abs(
                    saved_rmse
                    -
                    m5[
                        "test_rmse"
                    ]
                )
                <
                1e-8,
                f"{run5}: saved/direct test RMSE mismatch"
            )

        saved_mae = lookup_value(
            test_metric_row,
            [
                "mae",
                "MAE",
            ],
        )

        if np.isfinite(
            saved_mae
        ):

            require(
                abs(
                    saved_mae
                    -
                    m5[
                        "test_mae"
                    ]
                )
                <
                1e-8,
                f"{run5}: saved/direct test MAE mismatch"
            )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        state = load_checkpoint(
            run5
        )

        required_state = {
            "encoding_strength_raw":
                (12, depth),

            "encoding_shift_raw":
                (12, depth),

            "encoding_phase_raw":
                (12, depth),

            "q_weights":
                (
                    depth,
                    1,
                    14,
                    3,
                ),

            "classical_head.0.weight":
                (1, 14),

            "classical_head.0.bias":
                (1,),
        }

        for key, expected_shape in required_state.items():

            require(
                key in state,
                f"{run5}: checkpoint missing {key}"
            )

            require(
                tuple(
                    state[key].shape
                )
                ==
                expected_shape,
                f"{run5}: wrong checkpoint shape for {key}"
            )

            finite(
                state[key]
                .detach()
                .cpu()
                .numpy(),
                f"{run5}:{key}",
            )

        # ----------------------------------------------------
        # Learned encoder NPZ
        # ----------------------------------------------------

        enc = np.load(
            run5
            /
            "learned_encoding_parameters.npz"
        )

        required_enc = [
            "strength_raw",
            "shift_raw",
            "phase_raw",
            "reupload_strength",
            "polar_shift",
            "polar_baseline",
            "seasonal_phase_shift",
            "fixed_seasonal_phase",
            "learned_seasonal_phase_unwrapped",
            "learned_seasonal_phase_wrapped",
            "mean_strength_by_layer",
            "mean_abs_strength_deviation_by_layer",
            "mean_phase_shift_by_layer",
            "mean_abs_phase_shift_by_layer",
            "max_abs_phase_shift_by_layer",
            "min_adjacent_phase_gap_by_layer",
            "max_adjacent_phase_gap_by_layer",
            "mean_adjacent_phase_gap_by_layer",
        ]

        for key in required_enc:

            require(
                key in enc.files,
                f"{run5}: encoder NPZ missing {key}"
            )

            finite(
                enc[key],
                f"{run5}:{key}",
            )

        strength_raw = enc[
            "strength_raw"
        ]

        shift_raw = enc[
            "shift_raw"
        ]

        phase_raw = enc[
            "phase_raw"
        ]

        strength = enc[
            "reupload_strength"
        ]

        polar_shift = enc[
            "polar_shift"
        ]

        phase_shift = enc[
            "seasonal_phase_shift"
        ]

        fixed_phi = enc[
            "fixed_seasonal_phase"
        ]

        learned_phi = enc[
            "learned_seasonal_phase_unwrapped"
        ]

        learned_phi_wrapped = enc[
            "learned_seasonal_phase_wrapped"
        ]

        for name, arr in [
            (
                "strength_raw",
                strength_raw,
            ),
            (
                "shift_raw",
                shift_raw,
            ),
            (
                "phase_raw",
                phase_raw,
            ),
            (
                "strength",
                strength,
            ),
            (
                "polar_shift",
                polar_shift,
            ),
            (
                "phase_shift",
                phase_shift,
            ),
            (
                "fixed_phi",
                fixed_phi,
            ),
            (
                "learned_phi",
                learned_phi,
            ),
        ]:

            require(
                arr.shape
                ==
                (
                    12,
                    depth,
                ),
                f"{run5}: wrong shape for {name}: {arr.shape}"
            )

        # ----------------------------------------------------
        # Checkpoint ↔ saved encoder exactness
        # ----------------------------------------------------

        ckpt_errors = [
            max_abs_error(
                strength_raw,
                state[
                    "encoding_strength_raw"
                ]
                .detach()
                .cpu()
                .numpy(),
            ),

            max_abs_error(
                shift_raw,
                state[
                    "encoding_shift_raw"
                ]
                .detach()
                .cpu()
                .numpy(),
            ),

            max_abs_error(
                phase_raw,
                state[
                    "encoding_phase_raw"
                ]
                .detach()
                .cpu()
                .numpy(),
            ),
        ]

        this_ckpt_error = max(
            ckpt_errors
        )

        require(
            this_ckpt_error
            <
            1e-12,
            f"{run5}: saved encoder does not match restored checkpoint"
        )

        max_checkpoint_npz_error = max(
            max_checkpoint_npz_error,
            this_ckpt_error,
        )

        # ----------------------------------------------------
        # Reconstruct transformations
        # ----------------------------------------------------

        expected_strength = (
            1.0
            +
            np.tanh(
                strength_raw
            )
        )

        expected_polar_shift = (
            math.pi
            /
            2.0
            *
            np.tanh(
                shift_raw
            )
        )

        expected_phase_shift = (
            PHASE_BOUND
            *
            np.tanh(
                phase_raw
            )
        )

        expected_fixed_phi = np.repeat(
            (
                2.0
                *
                math.pi
                *
                np.arange(
                    12,
                    dtype=float,
                )
                /
                12.0
            )[:, None],
            depth,
            axis=1,
        )

        expected_learned_phi = (
            expected_fixed_phi
            +
            expected_phase_shift
        )

        expected_wrapped_phi = np.mod(
            expected_learned_phi,
            2.0
            *
            math.pi,
        )

        require(
            max_abs_error(
                strength,
                expected_strength,
            )
            <
            1e-12,
            f"{run5}: strength reconstruction failed"
        )

        require(
            max_abs_error(
                polar_shift,
                expected_polar_shift,
            )
            <
            1e-12,
            f"{run5}: polar-shift reconstruction failed"
        )

        this_phase_transform_error = max_abs_error(
            phase_shift,
            expected_phase_shift,
        )

        this_fixed_error = max_abs_error(
            fixed_phi,
            expected_fixed_phi,
        )

        this_unwrapped_error = max_abs_error(
            learned_phi,
            expected_learned_phi,
        )

        this_wrapped_error = max_abs_error(
            learned_phi_wrapped,
            expected_wrapped_phi,
        )

        require(
            this_phase_transform_error
            <
            1e-12,
            f"{run5}: phase transform reconstruction failed"
        )

        require(
            this_fixed_error
            <
            1e-12,
            f"{run5}: fixed seasonal phase reconstruction failed"
        )

        require(
            this_unwrapped_error
            <
            1e-12,
            f"{run5}: learned unwrapped phase reconstruction failed"
        )

        require(
            this_wrapped_error
            <
            1e-12,
            f"{run5}: learned wrapped phase reconstruction failed"
        )

        max_phase_transform_error = max(
            max_phase_transform_error,
            this_phase_transform_error,
        )

        max_fixed_phase_error = max(
            max_fixed_phase_error,
            this_fixed_error,
        )

        max_unwrapped_phase_error = max(
            max_unwrapped_phase_error,
            this_unwrapped_error,
        )

        max_wrapped_phase_error = max(
            max_wrapped_phase_error,
            this_wrapped_error,
        )

        # ----------------------------------------------------
        # Phase bounds and cyclic ordering
        # ----------------------------------------------------

        run_max_abs_phase = float(
            np.max(
                np.abs(
                    phase_shift
                )
            )
        )

        require(
            run_max_abs_phase
            <
            PHASE_BOUND,
            f"{run5}: phase bound violated"
        )

        min_gap_all = np.inf
        max_gap_all = -np.inf

        for ell in range(
            depth
        ):

            phi = learned_phi[
                :,
                ell,
            ]

            cyclic = np.concatenate(
                [
                    phi,
                    [
                        phi[0]
                        +
                        2.0
                        *
                        math.pi
                    ],
                ]
            )

            gaps = np.diff(
                cyclic
            )

            require(
                np.all(
                    gaps
                    >
                    0.0
                ),
                f"{run5}: seasonal ordering violated layer {ell + 1}"
            )

            require(
                abs(
                    np.sum(
                        gaps
                    )
                    -
                    2.0
                    *
                    math.pi
                )
                <
                1e-12,
                f"{run5}: phase gaps do not sum to 2pi"
            )

            require(
                abs(
                    np.mean(
                        gaps
                    )
                    -
                    FIXED_GAP
                )
                <
                1e-12,
                f"{run5}: mean phase gap != pi/6"
            )

            min_gap_all = min(
                min_gap_all,
                float(
                    np.min(
                        gaps
                    )
                ),
            )

            max_gap_all = max(
                max_gap_all,
                float(
                    np.max(
                        gaps
                    )
                ),
            )

            # Layer-level learned encoder table.
            layer_rows.append({
                "depth":
                    depth,

                "seed":
                    seed,

                "layer":
                    ell + 1,

                "mean_strength":
                    float(
                        np.mean(
                            strength[
                                :,
                                ell,
                            ]
                        )
                    ),

                "std_strength":
                    float(
                        np.std(
                            strength[
                                :,
                                ell,
                            ],
                            ddof=1,
                        )
                    ),

                "mean_abs_strength_minus_1":
                    float(
                        np.mean(
                            np.abs(
                                strength[
                                    :,
                                    ell,
                                ]
                                -
                                1.0
                            )
                        )
                    ),

                "mean_polar_shift_rad":
                    float(
                        np.mean(
                            polar_shift[
                                :,
                                ell,
                            ]
                        )
                    ),

                "mean_abs_polar_shift_rad":
                    float(
                        np.mean(
                            np.abs(
                                polar_shift[
                                    :,
                                    ell,
                                ]
                            )
                        )
                    ),

                "mean_phase_shift_rad":
                    float(
                        np.mean(
                            phase_shift[
                                :,
                                ell,
                            ]
                        )
                    ),

                "mean_abs_phase_shift_rad":
                    float(
                        np.mean(
                            np.abs(
                                phase_shift[
                                    :,
                                    ell,
                                ]
                            )
                        )
                    ),

                "max_abs_phase_shift_rad":
                    float(
                        np.max(
                            np.abs(
                                phase_shift[
                                    :,
                                    ell,
                                ]
                            )
                        )
                    ),

                "min_adjacent_phase_gap_rad":
                    float(
                        np.min(
                            gaps
                        )
                    ),

                "max_adjacent_phase_gap_rad":
                    float(
                        np.max(
                            gaps
                        )
                    ),

                "mean_adjacent_phase_gap_rad":
                    float(
                        np.mean(
                            gaps
                        )
                    ),
            })

            # Full month × layer table.
            for month in range(
                12
            ):

                encoder_rows.append({
                    "depth":
                        depth,

                    "seed":
                        seed,

                    "month_index":
                        month,

                    "layer":
                        ell + 1,

                    "strength_raw":
                        float(
                            strength_raw[
                                month,
                                ell,
                            ]
                        ),

                    "strength":
                        float(
                            strength[
                                month,
                                ell,
                            ]
                        ),

                    "polar_shift_raw":
                        float(
                            shift_raw[
                                month,
                                ell,
                            ]
                        ),

                    "polar_shift_rad":
                        float(
                            polar_shift[
                                month,
                                ell,
                            ]
                        ),

                    "phase_raw":
                        float(
                            phase_raw[
                                month,
                                ell,
                            ]
                        ),

                    "phase_shift_rad":
                        float(
                            phase_shift[
                                month,
                                ell,
                            ]
                        ),

                    "phase_shift_deg":
                        float(
                            np.degrees(
                                phase_shift[
                                    month,
                                    ell,
                                ]
                            )
                        ),

                    "fixed_phase_rad":
                        float(
                            fixed_phi[
                                month,
                                ell,
                            ]
                        ),

                    "learned_phase_unwrapped_rad":
                        float(
                            learned_phi[
                                month,
                                ell,
                            ]
                        ),

                    "learned_phase_wrapped_rad":
                        float(
                            learned_phi_wrapped[
                                month,
                                ell,
                            ]
                        ),
                })

        # ----------------------------------------------------
        # Run-level result
        # ----------------------------------------------------

        run_row = {
            "depth":
                depth,

            "seed":
                seed,

            **m5,

            "total_parameter_count":
                78 * depth + 15,

            "encoding_parameter_count":
                36 * depth,

            "mean_strength":
                float(
                    np.mean(
                        strength
                    )
                ),

            "mean_abs_strength_minus_1":
                float(
                    np.mean(
                        np.abs(
                            strength
                            -
                            1.0
                        )
                    )
                ),

            "min_strength":
                float(
                    np.min(
                        strength
                    )
                ),

            "max_strength":
                float(
                    np.max(
                        strength
                    )
                ),

            "mean_polar_shift_rad":
                float(
                    np.mean(
                        polar_shift
                    )
                ),

            "mean_abs_polar_shift_rad":
                float(
                    np.mean(
                        np.abs(
                            polar_shift
                        )
                    )
                ),

            "mean_phase_shift_rad":
                float(
                    np.mean(
                        phase_shift
                    )
                ),

            "mean_abs_phase_shift_rad":
                float(
                    np.mean(
                        np.abs(
                            phase_shift
                        )
                    )
                ),

            "max_abs_phase_shift_rad":
                run_max_abs_phase,

            "min_adjacent_phase_gap_rad":
                min_gap_all,

            "max_adjacent_phase_gap_rad":
                max_gap_all,
        }

        run_rows.append(
            run_row
        )

        # ----------------------------------------------------
        # Matched model-4 comparison
        # ----------------------------------------------------

        paired_rows.append({
            "depth":
                depth,

            "seed":
                seed,

            "model4_test_rmse":
                m4[
                    "test_rmse"
                ],

            "model5_test_rmse":
                m5[
                    "test_rmse"
                ],

            # Positive means model 5 is better.
            "rmse_improvement_model5":
                m4[
                    "test_rmse"
                ]
                -
                m5[
                    "test_rmse"
                ],

            "model4_pearson":
                m4[
                    "test_pearson"
                ],

            "model5_pearson":
                m5[
                    "test_pearson"
                ],

            "pearson_change_model5":
                m5[
                    "test_pearson"
                ]
                -
                m4[
                    "test_pearson"
                ],

            "model4_kge":
                m4[
                    "test_kge"
                ],

            "model5_kge":
                m5[
                    "test_kge"
                ],

            "kge_change_model5":
                m5[
                    "test_kge"
                ]
                -
                m4[
                    "test_kge"
                ],

            "model4_fft":
                m4[
                    "test_fft_cosine"
                ],

            "model5_fft":
                m5[
                    "test_fft_cosine"
                ],

            "fft_change_model5":
                m5[
                    "test_fft_cosine"
                ]
                -
                m4[
                    "test_fft_cosine"
                ],

            "model4_amplitude_ratio":
                m4[
                    "test_amplitude_ratio"
                ],

            "model5_amplitude_ratio":
                m5[
                    "test_amplitude_ratio"
                ],

            "amplitude_ratio_change_model5":
                m5[
                    "test_amplitude_ratio"
                ]
                -
                m4[
                    "test_amplitude_ratio"
                ],

            "model5_lower_rmse":
                bool(
                    m5[
                        "test_rmse"
                    ]
                    <
                    m4[
                        "test_rmse"
                    ]
                ),
        })

        validated += 1


# ============================================================
# Final integrity checks
# ============================================================

require(
    validated
    ==
    EXPECTED_RUNS,
    f"Validated {validated} runs, expected {EXPECTED_RUNS}"
)

run_df = pd.DataFrame(
    run_rows
).sort_values(
    [
        "depth",
        "seed",
    ]
)

paired_df = pd.DataFrame(
    paired_rows
).sort_values(
    [
        "depth",
        "seed",
    ]
)

encoder_df = pd.DataFrame(
    encoder_rows
).sort_values(
    [
        "depth",
        "seed",
        "layer",
        "month_index",
    ]
)

layer_df = pd.DataFrame(
    layer_rows
).sort_values(
    [
        "depth",
        "seed",
        "layer",
    ]
)


# ============================================================
# Five-seed forecast aggregation
# ============================================================

forecast_metrics = [
    "test_rmse",
    "test_mae",
    "test_pearson",
    "test_spearman",
    "test_kge",
    "test_nse",
    "test_fft_cosine",
    "test_amplitude_ratio",
]


forecast_summary_rows = []

for depth in DEPTHS:

    sub = run_df.loc[
        run_df[
            "depth"
        ]
        ==
        depth
    ]

    require(
        len(sub)
        ==
        5,
        f"Depth {depth}: expected 5 seeds"
    )

    row = {
        "depth":
            depth,

        "n_seeds":
            len(sub),
    }

    for metric in forecast_metrics:

        row[
            f"{metric}_mean"
        ] = float(
            sub[
                metric
            ].mean()
        )

        row[
            f"{metric}_sd"
        ] = float(
            sub[
                metric
            ].std(
                ddof=1
            )
        )

    forecast_summary_rows.append(
        row
    )


forecast_summary = pd.DataFrame(
    forecast_summary_rows
)


# ============================================================
# Paired model-5 vs model-4 aggregation
# ============================================================

paired_summary_rows = []

for depth in DEPTHS:

    sub = paired_df.loc[
        paired_df[
            "depth"
        ]
        ==
        depth
    ]

    improvement = sub[
        "rmse_improvement_model5"
    ]

    paired_summary_rows.append({
        "depth":
            depth,

        "n_pairs":
            len(sub),

        "mean_rmse_improvement_model5":
            float(
                improvement.mean()
            ),

        "sd_rmse_improvement_model5":
            float(
                improvement.std(
                    ddof=1
                )
            ),

        "median_rmse_improvement_model5":
            float(
                improvement.median()
            ),

        "model5_better_seed_count":
            int(
                (
                    improvement
                    >
                    0
                ).sum()
            ),

        "mean_pearson_change_model5":
            float(
                sub[
                    "pearson_change_model5"
                ].mean()
            ),

        "mean_kge_change_model5":
            float(
                sub[
                    "kge_change_model5"
                ].mean()
            ),

        "mean_fft_change_model5":
            float(
                sub[
                    "fft_change_model5"
                ].mean()
            ),

        "mean_amplitude_change_model5":
            float(
                sub[
                    "amplitude_ratio_change_model5"
                ].mean()
            ),
    })


paired_summary = pd.DataFrame(
    paired_summary_rows
)


# ============================================================
# Encoder depth aggregation
# ============================================================

encoder_depth_rows = []

for depth in DEPTHS:

    sub = encoder_df.loc[
        encoder_df[
            "depth"
        ]
        ==
        depth
    ]

    encoder_depth_rows.append({
        "depth":
            depth,

        "mean_strength":
            float(
                sub[
                    "strength"
                ].mean()
            ),

        "sd_strength_all_month_layer_seed":
            float(
                sub[
                    "strength"
                ].std(
                    ddof=1
                )
            ),

        "mean_abs_strength_minus_1":
            float(
                np.mean(
                    np.abs(
                        sub[
                            "strength"
                        ].to_numpy()
                        -
                        1.0
                    )
                )
            ),

        "min_strength":
            float(
                sub[
                    "strength"
                ].min()
            ),

        "max_strength":
            float(
                sub[
                    "strength"
                ].max()
            ),

        "mean_polar_shift_rad":
            float(
                sub[
                    "polar_shift_rad"
                ].mean()
            ),

        "mean_abs_polar_shift_rad":
            float(
                np.mean(
                    np.abs(
                        sub[
                            "polar_shift_rad"
                        ]
                    )
                )
            ),

        "mean_phase_shift_rad":
            float(
                sub[
                    "phase_shift_rad"
                ].mean()
            ),

        "mean_abs_phase_shift_rad":
            float(
                np.mean(
                    np.abs(
                        sub[
                            "phase_shift_rad"
                        ]
                    )
                )
            ),

        "mean_abs_phase_shift_deg":
            float(
                np.mean(
                    np.abs(
                        sub[
                            "phase_shift_deg"
                        ]
                    )
                )
            ),

        "max_abs_phase_shift_rad":
            float(
                np.max(
                    np.abs(
                        sub[
                            "phase_shift_rad"
                        ]
                    )
                )
            ),

        "max_abs_phase_shift_deg":
            float(
                np.max(
                    np.abs(
                        sub[
                            "phase_shift_deg"
                        ]
                    )
                )
            ),
    })


encoder_depth_summary = pd.DataFrame(
    encoder_depth_rows
)


# ============================================================
# Layer aggregation across five seeds
# ============================================================

layer_mean_rows = []

for depth in DEPTHS:

    for layer in range(
        1,
        depth + 1,
    ):

        sub = layer_df.loc[
            (
                layer_df[
                    "depth"
                ]
                ==
                depth
            )
            &
            (
                layer_df[
                    "layer"
                ]
                ==
                layer
            )
        ]

        require(
            len(sub)
            ==
            5,
            f"L={depth}, layer={layer}: expected 5 seed rows"
        )

        layer_mean_rows.append({
            "depth":
                depth,

            "layer":
                layer,

            "mean_strength":
                float(
                    sub[
                        "mean_strength"
                    ].mean()
                ),

            "sd_strength_across_seeds":
                float(
                    sub[
                        "mean_strength"
                    ].std(
                        ddof=1
                    )
                ),

            "mean_abs_polar_shift_rad":
                float(
                    sub[
                        "mean_abs_polar_shift_rad"
                    ].mean()
                ),

            "mean_phase_shift_rad":
                float(
                    sub[
                        "mean_phase_shift_rad"
                    ].mean()
                ),

            "mean_abs_phase_shift_rad":
                float(
                    sub[
                        "mean_abs_phase_shift_rad"
                    ].mean()
                ),

            "max_abs_phase_shift_rad_mean_across_seeds":
                float(
                    sub[
                        "max_abs_phase_shift_rad"
                    ].mean()
                ),

            "mean_min_adjacent_phase_gap_rad":
                float(
                    sub[
                        "min_adjacent_phase_gap_rad"
                    ].mean()
                ),

            "mean_max_adjacent_phase_gap_rad":
                float(
                    sub[
                        "max_adjacent_phase_gap_rad"
                    ].mean()
                ),
        })


layer_mean_summary = pd.DataFrame(
    layer_mean_rows
)


# ============================================================
# Save
# ============================================================

run_df.to_csv(
    OUT
    /
    "all_30_runs.csv",
    index=False,
)

forecast_summary.to_csv(
    OUT
    /
    "forecast_depth_mean_sd.csv",
    index=False,
)

paired_df.to_csv(
    OUT
    /
    "model5_vs_model4_paired_seedwise.csv",
    index=False,
)

paired_summary.to_csv(
    OUT
    /
    "model5_vs_model4_paired_depth_summary.csv",
    index=False,
)

encoder_df.to_csv(
    OUT
    /
    "all_learned_encoder_month_layer_seed.csv",
    index=False,
)

layer_df.to_csv(
    OUT
    /
    "learned_encoder_layer_by_seed.csv",
    index=False,
)

layer_mean_summary.to_csv(
    OUT
    /
    "learned_encoder_layer_mean_sd.csv",
    index=False,
)

encoder_depth_summary.to_csv(
    OUT
    /
    "learned_encoder_depth_summary.csv",
    index=False,
)


validation_summary = {
    "status":
        "PASS",

    "validated_runs":
        validated,

    "expected_runs":
        EXPECTED_RUNS,

    "depths":
        DEPTHS,

    "seeds":
        SEEDS,

    "max_truth_error_vs_model4":
        max_truth_error,

    "max_checkpoint_npz_encoder_error":
        max_checkpoint_npz_error,

    "max_phase_transform_error":
        max_phase_transform_error,

    "max_fixed_phase_error":
        max_fixed_phase_error,

    "max_unwrapped_phase_error":
        max_unwrapped_phase_error,

    "max_wrapped_phase_error":
        max_wrapped_phase_error,

    "phase_bound_rad":
        PHASE_BOUND,

    "phase_bound_deg":
        15.0,

    "fft_fallback_run_count":
        fft_fallback_count,
}


with open(
    OUT
    /
    "validation_summary.json",
    "w",
) as f:

    json.dump(
        validation_summary,
        f,
        indent=2,
    )


# ============================================================
# Print key scientific results
# ============================================================

print()
print("=" * 130)
print("VALIDATION")
print("=" * 130)

print(
    f"Validated runs: {validated}/{EXPECTED_RUNS}"
)

print(
    "Max truth difference vs model 4:",
    f"{max_truth_error:.3e}",
)

print(
    "Max checkpoint ↔ encoder NPZ error:",
    f"{max_checkpoint_npz_error:.3e}",
)

print(
    "Max phase-transform reconstruction error:",
    f"{max_phase_transform_error:.3e}",
)

print(
    "FFT fallback count:",
    fft_fallback_count,
)


print()
print("=" * 130)
print("MODEL 5 FIVE-SEED TEST PERFORMANCE")
print("=" * 130)

for _, row in forecast_summary.iterrows():

    print(
        f"L={int(row['depth'])}: "
        f"RMSE={row['test_rmse_mean']:.6f}"
        f" ± {row['test_rmse_sd']:.6f} | "
        f"MAE={row['test_mae_mean']:.6f}"
        f" ± {row['test_mae_sd']:.6f} | "
        f"Pearson={row['test_pearson_mean']:.6f}"
        f" ± {row['test_pearson_sd']:.6f} | "
        f"KGE={row['test_kge_mean']:.6f}"
        f" ± {row['test_kge_sd']:.6f} | "
        f"FFT={row['test_fft_cosine_mean']:.6f}"
        f" ± {row['test_fft_cosine_sd']:.6f} | "
        f"Amp={row['test_amplitude_ratio_mean']:.6f}"
        f" ± {row['test_amplitude_ratio_sd']:.6f}"
    )


print()
print("=" * 130)
print("MODEL 5 VS MODEL 4 — PAIRED TEST RMSE")
print("Positive improvement means MODEL 5 has lower test RMSE.")
print("=" * 130)

total_better = 0

for _, row in paired_summary.iterrows():

    total_better += int(
        row[
            "model5_better_seed_count"
        ]
    )

    print(
        f"L={int(row['depth'])}: "
        f"improvement="
        f"{row['mean_rmse_improvement_model5']:+.6f}"
        f" ± {row['sd_rmse_improvement_model5']:.6f} | "
        f"model5 better "
        f"{int(row['model5_better_seed_count'])}/5 seeds"
    )


print()
print(
    f"Model 5 lower RMSE in {total_better}/30 matched runs."
)


best_idx = forecast_summary[
    "test_rmse_mean"
].idxmin()

best_row = forecast_summary.loc[
    best_idx
]


print()
print("=" * 130)
print("BEST MODEL-5 DEPTH")
print("=" * 130)

print(
    f"Depth {int(best_row['depth'])}: "
    f"test RMSE "
    f"{best_row['test_rmse_mean']:.6f}"
    f" ± {best_row['test_rmse_sd']:.6f}"
)


print()
print("=" * 130)
print("LEARNED ENCODER DEPTH SUMMARY")
print("=" * 130)

for _, row in encoder_depth_summary.iterrows():

    print(
        f"L={int(row['depth'])}: "
        f"mean s={row['mean_strength']:.6f} | "
        f"mean |s-1|={row['mean_abs_strength_minus_1']:.6f} | "
        f"mean |Delta phi|="
        f"{row['mean_abs_phase_shift_deg']:.3f} deg | "
        f"max |Delta phi|="
        f"{row['max_abs_phase_shift_deg']:.3f} deg"
    )


print()
print("=" * 130)
print("OUTPUT")
print("=" * 130)

for path in sorted(
    OUT.glob("*")
):
    print(path.name)


print()
print("=" * 130)
print("MODEL 5 30-RUN VALIDATION + AGGREGATION: PASS")
print("=" * 130)
