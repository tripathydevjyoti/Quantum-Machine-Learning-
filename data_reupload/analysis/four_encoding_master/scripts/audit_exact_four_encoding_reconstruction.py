#!/usr/bin/env python

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import torch
import pennylane as qml


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(
    "/umbc/rs/pi_deffner/users/devjyot1/projects/"
    "Quantum-Machine-Learning-"
)

OUT_DIR = (
    PROJECT_ROOT
    / "data_reupload"
    / "analysis"
    / "four_encoding_master"
    / "output"
    / "geometry"
    / "reconstruction_audit"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Representative models
#
# Use L=6 for every family so we validate the complete
# repeated-upload structure.
# ============================================================

RUNS = {
    "ry": (
        PROJECT_ROOT
        / "data_reupload/results/final_model_sweep/"
        "single_tanh_a1/depth_6/seed_42"
    ),

    "same_scalar_rxryrz": (
        PROJECT_ROOT
        / "data_reupload/results/encoding_sweep/"
        "same_scalar_rxryrz_tanh_a1/depth_6/seed_42"
    ),

    "seasonal_meridian": (
        PROJECT_ROOT
        / "data_reupload/results/encoding_sweep/"
        "seasonal_meridian_tanh_a1/depth_6/seed_42"
    ),

    "learnable_seasonal_cdf": (
        PROJECT_ROOT
        / "data_reupload/results/encoding_sweep/"
        "learnable_seasonal_cdf_tanh_a1/depth_6/seed_42"
    ),
}


# ============================================================
# Helpers
# ============================================================

def scale_fixed(
    y,
    data_min=0.0,
    data_max=350.0,
):
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    return (
        2.0
        *
        (y - data_min)
        /
        (data_max - data_min)
        -
        1.0
    )


def inverse_scale_fixed(
    z,
    data_min=0.0,
    data_max=350.0,
):
    z = np.asarray(
        z,
        dtype=np.float64,
    )

    return (
        data_min
        +
        (z + 1.0)
        *
        (data_max - data_min)
        /
        2.0
    )


def empirical_cdf_right(
    values,
    sorted_reference,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    sorted_reference = np.asarray(
        sorted_reference,
        dtype=np.float64,
    )

    counts = np.searchsorted(
        sorted_reference,
        values,
        side="right",
    )

    return (
        counts.astype(np.float64)
        /
        float(len(sorted_reference))
    )


def load_original_series(config):

    csv_path = Path(
        config["csv_path"]
    )

    if not csv_path.is_absolute():
        csv_path = (
            PROJECT_ROOT
            /
            csv_path
        )

    df = pd.read_csv(
        csv_path
    )

    value_col = int(
        config["value_col"]
    )

    y_original = (
        pd.to_numeric(
            df.iloc[:, value_col],
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=np.float64
        )
    )

    if len(y_original) != 539:
        raise RuntimeError(
            "Expected 539 non-missing observations, "
            f"found {len(y_original)}"
        )

    return y_original


def load_checkpoint(run_dir):

    ckpt = torch.load(
        run_dir / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )

    state = ckpt[
        "model_state_dict"
    ]

    return ckpt, state


# ============================================================
# Readout signs
#
# PennyLane state ordering:
# wire 0 = most-significant bit.
# ============================================================

def make_z_signs(
    n_qubits,
):
    dim = 2 ** n_qubits

    idx = np.arange(
        dim,
        dtype=np.int64,
    )

    signs = []

    for q in range(
        n_qubits
    ):

        bit_mask = (
            1
            <<
            (n_qubits - 1 - q)
        )

        bit = (
            (idx & bit_mask)
            != 0
        )

        sign = np.where(
            bit,
            -1.0,
            +1.0,
        )

        signs.append(
            sign
        )

    return np.stack(
        signs,
        axis=0,
    )


def local_z_from_state(
    state,
    z_signs,
):
    probs = (
        np.abs(state) ** 2
    )

    return (
        z_signs
        @
        probs
    )


# ============================================================
# Test-window construction
# ============================================================

def construct_test_inputs(
    encoding,
    y_original,
    config,
):

    window = int(
        config["window_size"]
    )

    if window != 14:
        raise RuntimeError(
            f"Expected window 14, got {window}"
        )

    test_targets = np.arange(
        434,
        539,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Standard scaled windows
    # --------------------------------------------------------

    y_scaled = scale_fixed(
        y_original,
        float(config["scaler_data_min"]),
        float(config["scaler_data_max"]),
    )

    if encoding in {
        "ry",
        "same_scalar_rxryrz",
    }:

        X = np.stack(
            [
                y_scaled[
                    t - window:t
                ]
                for t in test_targets
            ],
            axis=0,
        )

        source_indices = np.stack(
            [
                np.arange(
                    t - window,
                    t,
                    dtype=np.int64,
                )
                for t in test_targets
            ],
            axis=0,
        )

        return (
            X,
            source_indices,
            test_targets,
        )

    # --------------------------------------------------------
    # Training-only empirical CDF reference
    #
    # observations 0...363
    # --------------------------------------------------------

    cdf_reference = np.sort(
        y_original[:364].copy()
    )

    X_rows = []
    source_rows = []

    for target in test_targets:

        source_indices = np.arange(
            target - window,
            target,
            dtype=np.int64,
        )

        values = y_original[
            source_indices
        ]

        u = empirical_cdf_right(
            values,
            cdf_reference,
        )

        z = (
            2.0 * u
            -
            1.0
        )

        # Protect against tiny numerical excursions.
        z = np.clip(
            z,
            -1.0,
            1.0,
        )

        theta_cdf = np.arccos(
            z
        )

        month = np.mod(
            source_indices,
            12,
        ).astype(
            np.int64
        )

        if encoding == "seasonal_meridian":

            phi = (
                2.0
                *
                np.pi
                *
                month
                /
                12.0
            )

            X = np.stack(
                [
                    theta_cdf,
                    phi,
                ],
                axis=-1,
            )

        elif encoding == "learnable_seasonal_cdf":

            X = np.stack(
                [
                    theta_cdf,
                    month.astype(
                        np.float64
                    ),
                ],
                axis=-1,
            )

        else:
            raise ValueError(
                encoding
            )

        X_rows.append(
            X
        )

        source_rows.append(
            source_indices
        )

    return (
        np.stack(
            X_rows,
            axis=0,
        ),
        np.stack(
            source_rows,
            axis=0,
        ),
        test_targets,
    )


# ============================================================
# Exact PennyLane state constructor
# ============================================================

def build_state_qnode(
    encoding,
    config,
    state_dict,
):

    n_qubits = int(
        config["n_qubits"]
    )

    depth = int(
        config["depth"]
    )

    A = int(
        config["n_ansatz_layers"]
    )

    angle_factor = float(
        config["angle_factor"]
    )

    if n_qubits != 14:
        raise RuntimeError(
            "Expected 14 qubits."
        )

    if A != 1:
        raise RuntimeError(
            "Expected A=1."
        )

    q_weights = (
        state_dict[
            "q_weights"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    if q_weights.shape != (
        depth,
        1,
        14,
        3,
    ):
        raise RuntimeError(
            f"Unexpected q_weights shape: "
            f"{q_weights.shape}"
        )

    # --------------------------------------------------------
    # Learnable encoding parameters from the CHECKPOINT.
    # --------------------------------------------------------

    if (
        encoding
        ==
        "learnable_seasonal_cdf"
    ):

        strength_raw = (
            state_dict[
                "encoding_strength_raw"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        shift_raw = (
            state_dict[
                "encoding_shift_raw"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        if strength_raw.shape != (
            12,
            depth,
        ):
            raise RuntimeError(
                "Unexpected strength shape: "
                f"{strength_raw.shape}"
            )

        if shift_raw.shape != (
            12,
            depth,
        ):
            raise RuntimeError(
                "Unexpected shift shape: "
                f"{shift_raw.shape}"
            )

        strength = (
            1.0
            +
            np.tanh(
                strength_raw
            )
        )

        shift = (
            np.pi
            /
            2.0
            *
            np.tanh(
                shift_raw
            )
        )

    else:
        strength = None
        shift = None

    dev = qml.device(
        "default.qubit",
        wires=n_qubits,
    )

    @qml.qnode(
        dev,
        interface=None,
        diff_method=None,
    )
    def state_circuit(x):

        for reupload_layer in range(
            depth
        ):

            # =================================================
            # DATA ENCODING
            # =================================================

            for q in range(
                n_qubits
            ):

                if encoding == "ry":

                    qml.RY(
                        angle_factor
                        *
                        float(x[q]),
                        wires=q,
                    )

                elif (
                    encoding
                    ==
                    "same_scalar_rxryrz"
                ):

                    theta = (
                        angle_factor
                        *
                        float(x[q])
                    )

                    qml.RX(
                        theta,
                        wires=q,
                    )

                    qml.RY(
                        theta,
                        wires=q,
                    )

                    qml.RZ(
                        theta,
                        wires=q,
                    )

                elif (
                    encoding
                    ==
                    "seasonal_meridian"
                ):

                    theta_cdf = float(
                        x[q, 0]
                    )

                    phi = float(
                        x[q, 1]
                    )

                    qml.RY(
                        theta_cdf,
                        wires=q,
                    )

                    qml.RZ(
                        phi,
                        wires=q,
                    )

                elif (
                    encoding
                    ==
                    "learnable_seasonal_cdf"
                ):

                    theta_cdf = float(
                        x[q, 0]
                    )

                    m = (
                        int(
                            round(
                                float(
                                    x[q, 1]
                                )
                            )
                        )
                        %
                        12
                    )

                    s = float(
                        strength[
                            m,
                            reupload_layer,
                        ]
                    )

                    b = float(
                        shift[
                            m,
                            reupload_layer,
                        ]
                    )

                    theta = (
                        np.pi / 2.0
                        +
                        b
                        +
                        s
                        *
                        (
                            theta_cdf
                            -
                            np.pi / 2.0
                        )
                    )

                    phi = (
                        2.0
                        *
                        np.pi
                        *
                        m
                        /
                        12.0
                    )

                    qml.RY(
                        theta,
                        wires=q,
                    )

                    qml.RZ(
                        phi,
                        wires=q,
                    )

                else:
                    raise ValueError(
                        encoding
                    )

            # =================================================
            # TRAINED VARIATIONAL BLOCK
            # =================================================

            for ansatz_layer in range(
                A
            ):

                for q in range(
                    n_qubits
                ):

                    qml.Rot(
                        float(
                            q_weights[
                                reupload_layer,
                                ansatz_layer,
                                q,
                                0,
                            ]
                        ),
                        float(
                            q_weights[
                                reupload_layer,
                                ansatz_layer,
                                q,
                                1,
                            ]
                        ),
                        float(
                            q_weights[
                                reupload_layer,
                                ansatz_layer,
                                q,
                                2,
                            ]
                        ),
                        wires=q,
                    )

                for q in range(
                    n_qubits - 1
                ):

                    qml.CNOT(
                        wires=[
                            q,
                            q + 1,
                        ]
                    )

                qml.CNOT(
                    wires=[
                        n_qubits - 1,
                        0,
                    ]
                )

        return qml.state()

    return (
        state_circuit,
        strength,
        shift,
    )


# ============================================================
# Main audit
# ============================================================

rows = []

print("=" * 120)
print("EXACT FOUR-ENCODING STATE RECONSTRUCTION AUDIT")
print("=" * 120)

for encoding, run_dir in RUNS.items():

    print()
    print("-" * 120)
    print(
        f"Encoding: {encoding}"
    )
    print(
        f"Run:      {run_dir}"
    )
    print("-" * 120)

    with open(
        run_dir / "config.json"
    ) as f:
        config = json.load(
            f
        )

    ckpt, state_dict = load_checkpoint(
        run_dir
    )

    predictions = np.load(
        run_dir
        /
        "predictions.npz",
        allow_pickle=True,
    )

    if int(config["depth"]) != 6:
        raise RuntimeError(
            "Audit is expected to use L=6."
        )

    if int(config["seed"]) != 42:
        raise RuntimeError(
            "Audit is expected to use seed 42."
        )

    if int(config["n_ansatz_layers"]) != 1:
        raise RuntimeError(
            "Expected A=1."
        )

    y_original = load_original_series(
        config
    )

    (
        X_test,
        source_indices,
        test_targets,
    ) = construct_test_inputs(
        encoding,
        y_original,
        config,
    )

    saved_indices = np.asarray(
        predictions[
            "test_target_indices"
        ],
        dtype=np.int64,
    )

    if not np.array_equal(
        test_targets,
        saved_indices,
    ):
        raise RuntimeError(
            f"Target-index mismatch "
            f"for {encoding}."
        )

    state_circuit, strength, shift = (
        build_state_qnode(
            encoding,
            config,
            state_dict,
        )
    )

    head_weight = (
        state_dict[
            "classical_head.0.weight"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
        .reshape(14)
    )

    head_bias = float(
        state_dict[
            "classical_head.0.bias"
        ]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)[0]
    )

    z_signs = make_z_signs(
        14
    )

    reconstructed_scaled = []

    max_norm_error = 0.0

    for i in range(
        len(X_test)
    ):

        psi = np.asarray(
            state_circuit(
                X_test[i]
            ),
            dtype=np.complex128,
        )

        if psi.shape != (
            2 ** 14,
        ):
            raise RuntimeError(
                f"Unexpected state shape: "
                f"{psi.shape}"
            )

        norm_error = abs(
            np.vdot(
                psi,
                psi,
            ).real
            -
            1.0
        )

        max_norm_error = max(
            max_norm_error,
            float(norm_error),
        )

        z_readout = local_z_from_state(
            psi,
            z_signs,
        )

        latent = (
            float(
                np.dot(
                    head_weight,
                    z_readout,
                )
            )
            +
            head_bias
        )

        yhat_scaled = math.tanh(
            latent
        )

        reconstructed_scaled.append(
            yhat_scaled
        )

    reconstructed_scaled = np.asarray(
        reconstructed_scaled,
        dtype=np.float64,
    )

    saved_scaled = np.asarray(
        predictions[
            "y_test_pred_scaled"
        ],
        dtype=np.float64,
    )

    max_scaled_error = float(
        np.max(
            np.abs(
                reconstructed_scaled
                -
                saved_scaled
            )
        )
    )

    reconstructed_original = (
        inverse_scale_fixed(
            reconstructed_scaled,
            float(
                config[
                    "scaler_data_min"
                ]
            ),
            float(
                config[
                    "scaler_data_max"
                ]
            ),
        )
    )

    saved_original = np.asarray(
        predictions[
            "y_test_pred_original"
        ],
        dtype=np.float64,
    )

    max_original_error = float(
        np.max(
            np.abs(
                reconstructed_original
                -
                saved_original
            )
        )
    )

    # --------------------------------------------------------
    # Learnable NPZ consistency audit
    #
    # Checkpoint remains authoritative.
    # --------------------------------------------------------

    max_strength_npz_diff = np.nan
    max_shift_npz_diff = np.nan

    if (
        encoding
        ==
        "learnable_seasonal_cdf"
    ):

        learned_npz = np.load(
            run_dir
            /
            "learned_encoding_parameters.npz",
            allow_pickle=True,
        )

        stored_strength = np.asarray(
            learned_npz[
                "reupload_strength"
            ],
            dtype=np.float64,
        )

        stored_shift = np.asarray(
            learned_npz[
                "polar_shift"
            ],
            dtype=np.float64,
        )

        max_strength_npz_diff = float(
            np.max(
                np.abs(
                    strength
                    -
                    stored_strength
                )
            )
        )

        max_shift_npz_diff = float(
            np.max(
                np.abs(
                    shift
                    -
                    stored_shift
                )
            )
        )

    passed = (
        max_scaled_error
        <
        1e-10
        and
        max_original_error
        <
        1e-8
        and
        max_norm_error
        <
        1e-10
    )

    if (
        encoding
        ==
        "learnable_seasonal_cdf"
    ):
        passed = (
            passed
            and
            max_strength_npz_diff
            <
            1e-12
            and
            max_shift_npz_diff
            <
            1e-12
        )

    rows.append({
        "encoding":
            encoding,

        "depth":
            6,

        "seed":
            42,

        "n_test_windows":
            len(X_test),

        "max_state_norm_error":
            max_norm_error,

        "max_scaled_prediction_error":
            max_scaled_error,

        "max_original_prediction_error":
            max_original_error,

        "max_strength_npz_difference":
            max_strength_npz_diff,

        "max_shift_npz_difference":
            max_shift_npz_diff,

        "status":
            "PASS"
            if passed
            else "FAIL",
    })

    print(
        f"  windows:                      "
        f"{len(X_test)}"
    )

    print(
        f"  max state-norm error:         "
        f"{max_norm_error:.3e}"
    )

    print(
        f"  max scaled-prediction error:  "
        f"{max_scaled_error:.3e}"
    )

    print(
        f"  max original-prediction error:"
        f" {max_original_error:.3e}"
    )

    if (
        encoding
        ==
        "learnable_seasonal_cdf"
    ):

        print(
            f"  max strength NPZ difference:  "
            f"{max_strength_npz_diff:.3e}"
        )

        print(
            f"  max shift NPZ difference:     "
            f"{max_shift_npz_diff:.3e}"
        )

    print(
        f"  STATUS: {rows[-1]['status']}"
    )


# ============================================================
# Save audit
# ============================================================

audit = pd.DataFrame(
    rows
)

audit.to_csv(
    OUT_DIR
    /
    "exact_reconstruction_audit.csv",
    index=False,
)

all_pass = bool(
    (
        audit["status"]
        ==
        "PASS"
    ).all()
)

with open(
    OUT_DIR
    /
    "exact_reconstruction_audit.json",
    "w",
) as f:

    json.dump(
        {
            "status":
                "PASS"
                if all_pass
                else "FAIL",

            "n_encodings":
                int(
                    len(audit)
                ),

            "depth":
                6,

            "seed":
                42,

            "prediction_tolerance_scaled":
                1e-10,

            "prediction_tolerance_original":
                1e-8,

            "state_norm_tolerance":
                1e-10,

            "rows":
                audit.to_dict(
                    orient="records"
                ),
        },
        f,
        indent=2,
    )


if not all_pass:
    raise RuntimeError(
        "At least one exact reconstruction failed."
    )


print()
print("=" * 120)
print("EXACT FOUR-ENCODING RECONSTRUCTION STATUS: PASS")
print("=" * 120)
