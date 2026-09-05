#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(
    "/umbc/rs/pi_deffner/users/devjyot1/projects/"
    "Quantum-Machine-Learning-"
)

SMOKE_ROOT = (
    PROJECT_ROOT
    / "data_reupload/results/encoding_sweep_gate"
)

VARIANT = "smoke_learnable_seasonal_cdf_rz_tanh_a1"
DEPTH = 2
SEED = 42

RUN = (
    SMOKE_ROOT
    / VARIANT
    / f"depth_{DEPTH}"
    / f"seed_{SEED}"
)

REFERENCE_RUN = (
    PROJECT_ROOT
    / "data_reupload/results/encoding_sweep/"
      "learnable_seasonal_cdf_tanh_a1/"
      "depth_2/seed_42"
)

OUT = (
    PROJECT_ROOT
    / "data_reupload/analysis/four_encoding_master/"
      "output/validation/model5_final_gate"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def fail(msg):
    raise RuntimeError(msg)


def require(condition, msg):
    if not condition:
        fail(msg)


def require_finite(arr, name):
    arr = np.asarray(arr)

    require(
        np.all(np.isfinite(arr)),
        f"{name} contains non-finite values."
    )


def max_abs(a, b):
    return float(
        np.max(
            np.abs(
                np.asarray(a)
                -
                np.asarray(b)
            )
        )
    )


print("=" * 120)
print("MODEL 5 FINAL AUTOMATED PRODUCTION GATE")
print("=" * 120)

print("Smoke run:", RUN)
print("Reference:", REFERENCE_RUN)
print()


# ============================================================
# 1. Expected production artifacts
# ============================================================

required_files = [
    "config.json",
    "history.csv",
    "best_model.pt",
    "final_model.pt",
    "learned_parameters.npz",
    "predictions.npz",
    "predictions_long.csv",
    "predictions_wide.csv",
    "metrics_by_split.csv",
    "fft_summary_by_split.csv",
    "fft_power_by_period.csv",
    "peak_low_error_summary.csv",
    "top_errors.csv",
    "result_summary.csv",
    "learned_encoding_parameters.npz",
    "learned_encoding_parameters.csv",
    "learned_encoding_layer_summary.csv",
    "learnable_seasonal_encoding_reference.npz",
    "encoding_metadata.json",
]

missing = [
    name
    for name in required_files
    if not (
        RUN
        /
        name
    ).exists()
]

require(
    not missing,
    f"Missing expected artifacts: {missing}"
)

print(
    f"[PASS] All {len(required_files)} required artifacts exist."
)


# ============================================================
# 2. Config and protocol
# ============================================================

with open(
    RUN / "config.json"
) as f:
    cfg = json.load(f)


expected_cfg = {
    "value_col": 4,
    "train_end_idx": 434,
    "window_size": 14,
    "train_window_end": 350,
    "scaler_data_min": 0.0,
    "scaler_data_max": 350.0,
    "scaler_feature_min": -1.0,
    "scaler_feature_max": 1.0,
    "depth": 2,
    "n_ansatz_layers": 1,
    "readout": "all",
    "seed": 42,
    "qnn_epochs": 3,
    "qnn_learning_rate": 0.01,
    "qnn_batch_size": 16,
    "qml_device": "default.qubit",
    "diff_method": "backprop",
    "device": "cpu",
    "use_output_tanh": True,
}


for key, expected in expected_cfg.items():

    require(
        key in cfg,
        f"config.json missing key: {key}"
    )

    actual = cfg[key]

    if isinstance(
        expected,
        float
    ):
        require(
            abs(
                float(actual)
                -
                expected
            ) < 1e-12,
            f"Config mismatch {key}: {actual} != {expected}"
        )

    else:
        require(
            actual == expected,
            f"Config mismatch {key}: {actual} != {expected}"
        )


require(
    cfg.get("encoding")
    ==
    "learnable_seasonal_cdf_rz",
    "Incorrect model-5 encoding identifier."
)

require(
    cfg.get("direct_parent_encoding")
    ==
    "learnable_seasonal_cdf",
    "Incorrect direct parent encoding."
)

require(
    cfg.get("phase_zero_exactly_matches_direct_parent")
    is True,
    "Missing parent-nesting provenance."
)

require(
    cfg.get("input_CDF_changed")
    is False,
    "CDF protocol unexpectedly changed."
)

for key in [
    "ansatz_changed",
    "entanglement_changed",
    "readout_changed",
    "classical_head_changed",
    "optimizer_protocol_changed",
    "checkpoint_protocol_changed",
    "metric_protocol_changed",
]:

    require(
        cfg.get(key) is False,
        f"Unexpected protocol change: {key}"
    )


print("[PASS] Training/data/model protocol metadata.")


# ============================================================
# 3. Parameter-count validation
# ============================================================

expected_variational = (
    42
    *
    DEPTH
)

expected_encoding = (
    36
    *
    DEPTH
)

expected_head = 15

expected_total = (
    78
    *
    DEPTH
    +
    15
)


require(
    int(
        cfg["variational_parameter_count"]
    )
    ==
    expected_variational,
    "Variational parameter-count mismatch."
)

require(
    int(
        cfg["encoding_parameter_count"]
    )
    ==
    expected_encoding,
    "Encoding parameter-count mismatch."
)

require(
    int(
        cfg["classical_head_parameter_count"]
    )
    ==
    expected_head,
    "Head parameter-count mismatch."
)

require(
    int(
        cfg["total_parameter_count"]
    )
    ==
    expected_total,
    "Total parameter-count mismatch."
)


print(
    "[PASS] Parameter counts:",
    f"variational={expected_variational},",
    f"encoding={expected_encoding},",
    f"head={expected_head},",
    f"total={expected_total}"
)


# ============================================================
# 4. Best checkpoint validation
# ============================================================

try:
    ckpt = torch.load(
        RUN / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
except TypeError:
    ckpt = torch.load(
        RUN / "best_model.pt",
        map_location="cpu",
    )


state = ckpt.get(
    "model_state_dict",
    ckpt,
)


required_state = {
    "encoding_strength_raw":
        (12, DEPTH),

    "encoding_shift_raw":
        (12, DEPTH),

    "encoding_phase_raw":
        (12, DEPTH),

    "q_weights":
        (
            DEPTH,
            1,
            14,
            3,
        ),

    "classical_head.0.weight":
        (1, 14),

    "classical_head.0.bias":
        (1,),
}


for key, shape in required_state.items():

    require(
        key in state,
        f"Checkpoint missing parameter: {key}"
    )

    require(
        tuple(
            state[key].shape
        )
        ==
        shape,
        f"Checkpoint shape mismatch for {key}: "
        f"{tuple(state[key].shape)} != {shape}"
    )

    require_finite(
        state[key]
        .detach()
        .cpu()
        .numpy(),
        key,
    )


best_phase_raw = (
    state[
        "encoding_phase_raw"
    ]
    .detach()
    .cpu()
    .numpy()
)


max_abs_best_phase_raw = float(
    np.max(
        np.abs(
            best_phase_raw
        )
    )
)


require(
    max_abs_best_phase_raw
    >
    1e-12,
    "Best checkpoint phase parameters never moved from zero."
)


print(
    "[PASS] Best checkpoint contains all model-5 parameters."
)

print(
    "       max |best encoding_phase_raw| =",
    f"{max_abs_best_phase_raw:.8e}"
)


# ============================================================
# 5. Saved learned encoder
# ============================================================

enc = np.load(
    RUN
    /
    "learned_encoding_parameters.npz"
)


required_npz_keys = [
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
    "initial_phase_raw",
    "initial_seasonal_phase_shift",
    "initial_seasonal_phase",
    "mean_phase_shift_by_layer",
    "mean_abs_phase_shift_by_layer",
    "max_abs_phase_shift_by_layer",
    "min_adjacent_phase_gap_by_layer",
    "max_adjacent_phase_gap_by_layer",
    "mean_adjacent_phase_gap_by_layer",
]


for key in required_npz_keys:

    require(
        key in enc.files,
        f"learned_encoding_parameters.npz missing {key}"
    )

    require_finite(
        enc[key],
        key,
    )


for key in [
    "strength_raw",
    "shift_raw",
    "phase_raw",
    "seasonal_phase_shift",
    "fixed_seasonal_phase",
    "learned_seasonal_phase_unwrapped",
    "learned_seasonal_phase_wrapped",
]:

    require(
        enc[key].shape
        ==
        (12, DEPTH),
        f"Unexpected {key} shape: {enc[key].shape}"
    )


phase_raw = enc[
    "phase_raw"
]

phase_shift = enc[
    "seasonal_phase_shift"
]

fixed_phi = enc[
    "fixed_seasonal_phase"
]

learned_unwrapped = enc[
    "learned_seasonal_phase_unwrapped"
]

learned_wrapped = enc[
    "learned_seasonal_phase_wrapped"
]


# Check that saver refers to restored best checkpoint.
checkpoint_npz_phase_error = max_abs(
    phase_raw,
    best_phase_raw,
)

require(
    checkpoint_npz_phase_error
    <
    1e-12,
    "Saved phase_raw does not match restored best checkpoint."
)


expected_shift = (
    math.pi
    /
    12.0
    *
    np.tanh(
        phase_raw
    )
)

phase_transform_error = max_abs(
    phase_shift,
    expected_shift,
)

require(
    phase_transform_error
    <
    1e-12,
    "Saved phase-shift transformation is incorrect."
)


expected_fixed = np.repeat(
    (
        2.0
        *
        math.pi
        *
        np.arange(
            12,
            dtype=np.float64,
        )
        /
        12.0
    )[:, None],
    DEPTH,
    axis=1,
)

fixed_phase_error = max_abs(
    fixed_phi,
    expected_fixed,
)

require(
    fixed_phase_error
    <
    1e-12,
    "Fixed seasonal phase reference is incorrect."
)


expected_unwrapped = (
    expected_fixed
    +
    expected_shift
)

unwrapped_error = max_abs(
    learned_unwrapped,
    expected_unwrapped,
)

require(
    unwrapped_error
    <
    1e-12,
    "Learned unwrapped phase reconstruction failed."
)


expected_wrapped = np.mod(
    expected_unwrapped,
    2.0
    *
    math.pi,
)

wrapped_error = max_abs(
    learned_wrapped,
    expected_wrapped,
)

require(
    wrapped_error
    <
    1e-12,
    "Learned wrapped phase reconstruction failed."
)


max_abs_phase_shift = float(
    np.max(
        np.abs(
            phase_shift
        )
    )
)

require(
    max_abs_phase_shift
    <
    math.pi
    /
    12.0,
    "Learned phase shift exceeds allowed bound."
)


print("[PASS] Learned azimuth transformation.")
print(
    "       max checkpoint/NPZ raw error =",
    f"{checkpoint_npz_phase_error:.3e}"
)
print(
    "       max phase-transform error =",
    f"{phase_transform_error:.3e}"
)
print(
    "       max |Delta phi| =",
    f"{max_abs_phase_shift:.8e} rad"
)


# ============================================================
# 6. Circular seasonal-order audit
# ============================================================

all_gap_rows = []

for ell in range(
    DEPTH
):

    phi = learned_unwrapped[
        :,
        ell,
    ]

    cyclic = np.concatenate([
        phi,
        [
            phi[0]
            +
            2.0
            *
            math.pi
        ],
    ])

    gaps = np.diff(
        cyclic
    )

    require(
        np.all(
            gaps
            >
            0.0
        ),
        f"Seasonal order violation at layer {ell + 1}."
    )

    gap_sum_error = abs(
        float(
            np.sum(
                gaps
            )
        )
        -
        2.0
        *
        math.pi
    )

    require(
        gap_sum_error
        <
        1e-12,
        f"Cyclic phase gaps do not sum to 2pi at layer {ell + 1}."
    )

    require(
        abs(
            float(
                np.mean(
                    gaps
                )
            )
            -
            math.pi
            /
            6.0
        )
        <
        1e-12,
        "Mean cyclic gap is not pi/6."
    )

    all_gap_rows.append({
        "layer":
            ell + 1,

        "min_gap_rad":
            float(
                np.min(
                    gaps
                )
            ),

        "max_gap_rad":
            float(
                np.max(
                    gaps
                )
            ),

        "mean_gap_rad":
            float(
                np.mean(
                    gaps
                )
            ),

        "gap_sum_rad":
            float(
                np.sum(
                    gaps
                )
            ),
    })


print("[PASS] Seasonal cyclic ordering and gap geometry.")


# ============================================================
# 7. Predictions / exact split identity
# ============================================================

pred = np.load(
    RUN
    /
    "predictions.npz"
)

ref = np.load(
    REFERENCE_RUN
    /
    "predictions.npz"
)


expected_shapes = {
    "y_train_true_scaled":
        (350,),

    "y_val_true_scaled":
        (70,),

    "y_test_true_scaled":
        (105,),

    "y_train_pred_scaled":
        (350,),

    "y_val_pred_scaled":
        (70,),

    "y_test_pred_scaled":
        (105,),

    "y_train_true_original":
        (350,),

    "y_val_true_original":
        (70,),

    "y_test_true_original":
        (105,),

    "y_train_pred_original":
        (350,),

    "y_val_pred_original":
        (70,),

    "y_test_pred_original":
        (105,),

    "train_target_indices":
        (350,),

    "val_target_indices":
        (70,),

    "test_target_indices":
        (105,),
}


for key, shape in expected_shapes.items():

    require(
        key in pred.files,
        f"predictions.npz missing {key}"
    )

    require(
        pred[key].shape
        ==
        shape,
        f"Unexpected predictions shape for {key}: "
        f"{pred[key].shape}"
    )

    require_finite(
        pred[key],
        key,
    )


expected_train_idx = np.arange(
    14,
    364,
)

expected_val_idx = np.arange(
    364,
    434,
)

expected_test_idx = np.arange(
    434,
    539,
)


require(
    np.array_equal(
        pred[
            "train_target_indices"
        ],
        expected_train_idx,
    ),
    "Train target indices incorrect."
)

require(
    np.array_equal(
        pred[
            "val_target_indices"
        ],
        expected_val_idx,
    ),
    "Validation target indices incorrect."
)

require(
    np.array_equal(
        pred[
            "test_target_indices"
        ],
        expected_test_idx,
    ),
    "Test target indices incorrect."
)


truth_keys = [
    "y_train_true_scaled",
    "y_val_true_scaled",
    "y_test_true_scaled",
    "y_train_true_original",
    "y_val_true_original",
    "y_test_true_original",
]


truth_errors = {}

for key in truth_keys:

    require(
        key in ref.files,
        f"Reference predictions missing {key}"
    )

    err = max_abs(
        pred[key],
        ref[key],
    )

    truth_errors[
        key
    ] = err

    require(
        err
        <
        1e-12,
        f"Truth mismatch versus model-4 reference for {key}: {err}"
    )


for key in [
    "train_target_indices",
    "val_target_indices",
    "test_target_indices",
]:

    require(
        np.array_equal(
            pred[key],
            ref[key],
        ),
        f"Index mismatch against model-4 reference: {key}"
    )


print("[PASS] Exact split/index/truth identity with model 4.")
print(
    "       max truth-array difference =",
    f"{max(truth_errors.values()):.3e}"
)


# ============================================================
# 8. Metrics / history
# ============================================================

metrics = pd.read_csv(
    RUN
    /
    "metrics_by_split.csv"
)

require(
    set(
        metrics[
            "split"
        ]
    )
    ==
    {
        "train",
        "val",
        "test",
    },
    "metrics_by_split.csv does not contain exactly train/val/test."
)


expected_n = {
    "train": 350,
    "val": 70,
    "test": 105,
}


for split, n in expected_n.items():

    row = metrics.loc[
        metrics[
            "split"
        ]
        ==
        split
    ]

    require(
        len(row)
        ==
        1,
        f"Expected one metrics row for {split}."
    )

    require(
        int(
            row.iloc[
                0
            ][
                "n"
            ]
        )
        ==
        n,
        f"Wrong n for {split}."
    )

    for metric in [
        "rmse",
        "mae",
        "bias_mean_pred_minus_true",
    ]:

        require(
            np.isfinite(
                float(
                    row.iloc[
                        0
                    ][
                        metric
                    ]
                )
            ),
            f"Non-finite {split} {metric}."
        )


history = pd.read_csv(
    RUN
    /
    "history.csv"
)

require(
    len(history)
    ==
    3,
    f"Expected exactly 3 smoke epochs, found {len(history)}."
)


print("[PASS] Three-epoch training history and deterministic metrics.")


# ============================================================
# 9. Result summary
# ============================================================

summary = pd.read_csv(
    RUN
    /
    "result_summary.csv"
)

require(
    len(summary)
    ==
    1,
    "result_summary.csv must contain one row."
)

s = summary.iloc[0]

for col, expected in [
    (
        "encoding_parameter_count",
        72,
    ),
    (
        "total_parameter_count",
        171,
    ),
]:

    require(
        col in summary.columns,
        f"result_summary.csv missing {col}"
    )

    require(
        int(
            s[
                col
            ]
        )
        ==
        expected,
        f"result_summary.csv incorrect {col}"
    )


for col in [
    "test_rmse_original",
    "test_mae_original",
]:

    require(
        col in summary.columns,
        f"result_summary.csv missing {col}"
    )

    require(
        np.isfinite(
            float(
                s[
                    col
                ]
            )
        ),
        f"Non-finite {col}"
    )


print("[PASS] Result summary.")


# ============================================================
# 10. Metadata file
# ============================================================

with open(
    RUN
    /
    "encoding_metadata.json"
) as f:
    metadata = json.load(f)


require(
    metadata.get(
        "encoding"
    )
    ==
    "learnable_seasonal_cdf_rz",
    "encoding_metadata.json has wrong encoding."
)

require(
    metadata.get(
        "checkpoint_contains_encoding_parameters"
    )
    is True,
    "Metadata checkpoint validation failed."
)

require(
    metadata.get(
        "seasonal_phase_changed"
    )
    is True,
    "Metadata does not identify trainable seasonal phase."
)


print("[PASS] Encoding metadata.")


# ============================================================
# Save validator outputs
# ============================================================

pd.DataFrame(
    all_gap_rows
).to_csv(
    OUT
    /
    "model5_gate_phase_gap_audit.csv",
    index=False,
)


gate_summary = {
    "status":
        "PASS",

    "depth":
        DEPTH,

    "seed":
        SEED,

    "qnn_epochs":
        3,

    "expected_total_parameters":
        expected_total,

    "expected_encoding_parameters":
        expected_encoding,

    "max_abs_best_phase_raw":
        max_abs_best_phase_raw,

    "max_abs_phase_shift_rad":
        max_abs_phase_shift,

    "phase_bound_rad":
        math.pi / 12.0,

    "checkpoint_npz_phase_raw_max_error":
        checkpoint_npz_phase_error,

    "phase_transform_max_error":
        phase_transform_error,

    "fixed_phase_max_error":
        fixed_phase_error,

    "unwrapped_phase_max_error":
        unwrapped_error,

    "wrapped_phase_max_error":
        wrapped_error,

    "max_truth_difference_vs_model4":
        max(
            truth_errors.values()
        ),
}


with open(
    OUT
    /
    "model5_final_gate_summary.json",
    "w",
) as f:

    json.dump(
        gate_summary,
        f,
        indent=2,
    )


print()
print("=" * 120)
print("MODEL 5 FINAL AUTOMATED PRODUCTION GATE: PASS")
print("THE 30-RUN PRODUCTION ARRAY MAY PROCEED.")
print("=" * 120)
