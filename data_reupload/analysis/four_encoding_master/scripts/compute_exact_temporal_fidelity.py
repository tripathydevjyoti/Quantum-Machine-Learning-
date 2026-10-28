#!/usr/bin/env python

from pathlib import Path
import os
import argparse
import json
import math

import numpy as np
import pandas as pd
import torch
import pennylane as qml


# ============================================================
# Global paths
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

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data_reupload/analysis/four_encoding_master/output/"
    "tables/master_run_manifest.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data_reupload/analysis/four_encoding_master/output/"
    "geometry/exact_temporal_fidelity"
)


# ============================================================
# Basic utilities
# ============================================================

def scale_fixed(y, data_min=0.0, data_max=350.0):

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

    y = (
        pd.to_numeric(
            df.iloc[:, value_col],
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=np.float64
        )
    )

    if len(y) != 539:
        raise RuntimeError(
            f"Expected 539 observations, found {len(y)}"
        )

    return y


# ============================================================
# Exact test-window reconstruction
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

    source_rows = []

    # --------------------------------------------------------
    # Standard scaled encodings
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

        X = []

        for target in test_targets:

            source_indices = np.arange(
                target - window,
                target,
                dtype=np.int64,
            )

            X.append(
                y_scaled[
                    source_indices
                ]
            )

            source_rows.append(
                source_indices
            )

        return (
            np.stack(X),
            np.stack(source_rows),
            test_targets,
        )

    # --------------------------------------------------------
    # Seasonal CDF reference
    # observations 0,...,363 only
    # --------------------------------------------------------

    reference = np.sort(
        y_original[:364].copy()
    )

    X = []

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
            reference,
        )

        z = np.clip(
            2.0 * u - 1.0,
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

            X_row = np.stack(
                [
                    theta_cdf,
                    phi,
                ],
                axis=-1,
            )

        elif encoding in {
            "learnable_seasonal_cdf",
            "learnable_seasonal_cdf_rz",
        }:

            X_row = np.stack(
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

        X.append(
            X_row
        )

        source_rows.append(
            source_indices
        )

    return (
        np.stack(X),
        np.stack(source_rows),
        test_targets,
    )


# ============================================================
# Z readout
# ============================================================

def make_z_signs(
    n_qubits,
):

    dim = 2 ** n_qubits

    idx = np.arange(
        dim,
        dtype=np.int64,
    )

    rows = []

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

        rows.append(
            np.where(
                bit,
                -1.0,
                1.0,
            )
        )

    return np.stack(
        rows,
        axis=0,
    )


def local_z_from_states(
    states,
    z_signs,
):

    probs = (
        np.abs(states) ** 2
    )

    return (
        probs
        @
        z_signs.T
    )


# ============================================================
# Exact trained-state QNode
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

    weights = (
        state_dict["q_weights"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    expected_shape = (
        depth,
        1,
        14,
        3,
    )

    if weights.shape != expected_shape:
        raise RuntimeError(
            f"q_weights shape {weights.shape}; "
            f"expected {expected_shape}"
        )

    # --------------------------------------------------------
    # Learnable encoding parameters
    # --------------------------------------------------------

    if encoding in {
        "learnable_seasonal_cdf",
        "learnable_seasonal_cdf_rz",
    }:

        strength_raw = (
            state_dict[
                "encoding_strength_raw"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        shift_raw = (
            state_dict[
                "encoding_shift_raw"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        if (
            strength_raw.shape
            !=
            (12, depth)
        ):
            raise RuntimeError(
                f"Unexpected strength shape "
                f"{strength_raw.shape}"
            )

        if (
            shift_raw.shape
            !=
            (12, depth)
        ):
            raise RuntimeError(
                f"Unexpected shift shape "
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

        if encoding == "learnable_seasonal_cdf_rz":

            if "encoding_phase_raw" not in state_dict:
                raise RuntimeError(
                    "Model 5 checkpoint missing encoding_phase_raw"
                )

            phase_raw = (
                state_dict[
                    "encoding_phase_raw"
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            if (
                phase_raw.shape
                !=
                (12, depth)
            ):
                raise RuntimeError(
                    f"Unexpected phase shape "
                    f"{phase_raw.shape}"
                )

            phase_shift = (
                np.pi
                /
                12.0
                *
                np.tanh(
                    phase_raw
                )
            )

        else:

            phase_raw = None
            phase_shift = None

    else:

        strength = None
        shift = None
        phase_raw = None
        phase_shift = None

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

        for layer in range(
            depth
        ):

            # =================================================
            # DATA UPLOAD
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

                    qml.RY(
                        float(x[q, 0]),
                        wires=q,
                    )

                    qml.RZ(
                        float(x[q, 1]),
                        wires=q,
                    )

                elif encoding in {
                    "learnable_seasonal_cdf",
                    "learnable_seasonal_cdf_rz",
                }:

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
                            layer,
                        ]
                    )

                    b = float(
                        shift[
                            m,
                            layer,
                        ]
                    )

                    theta = (
                        np.pi
                        /
                        2.0
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

                    if (
                        encoding
                        ==
                        "learnable_seasonal_cdf_rz"
                    ):

                        phi = (
                            phi
                            +
                            float(
                                phase_shift[
                                    m,
                                    layer,
                                ]
                            )
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
            # TRAINED ANSATZ
            # =================================================

            for a in range(
                A
            ):

                for q in range(
                    n_qubits
                ):

                    qml.Rot(
                        float(
                            weights[
                                layer,
                                a,
                                q,
                                0,
                            ]
                        ),
                        float(
                            weights[
                                layer,
                                a,
                                q,
                                1,
                            ]
                        ),
                        float(
                            weights[
                                layer,
                                a,
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

    return state_circuit


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest_index",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--max_tau",
        type=int,
        default=40,
    )

    args = parser.parse_args()

    manifest = (
        pd.read_csv(
            MANIFEST_PATH
        )
        .sort_values(
            [
                "encoding_order",
                "depth",
                "seed",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(manifest) != 150:
        raise RuntimeError(
            f"Expected 150 manifest rows, "
            f"found {len(manifest)}"
        )

    if not (
        0
        <=
        args.manifest_index
        <
        len(manifest)
    ):
        raise ValueError(
            args.manifest_index
        )

    row = manifest.iloc[
        args.manifest_index
    ]

    encoding = str(
        row["encoding"]
    )

    depth = int(
        row["depth"]
    )

    seed = int(
        row["seed"]
    )

    run_dir = Path(
        row["run_dir"]
    )

    out_dir = (
        OUTPUT_ROOT
        /
        encoding
        /
        f"depth_{depth}"
        /
        f"seed_{seed}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 110)
    print("EXACT TEMPORAL STATE FIDELITY")
    print("=" * 110)

    print(
        f"manifest index: {args.manifest_index}"
    )

    print(
        f"encoding:       {encoding}"
    )

    print(
        f"depth:          {depth}"
    )

    print(
        f"seed:           {seed}"
    )

    print(
        f"run_dir:        {run_dir}"
    )

    print(
        f"output_dir:     {out_dir}"
    )

    # ========================================================
    # Load run
    # ========================================================

    with open(
        run_dir
        /
        "config.json"
    ) as f:

        config = json.load(
            f
        )

    if int(
        config["depth"]
    ) != depth:
        raise RuntimeError(
            "Config depth mismatch."
        )

    if int(
        config["seed"]
    ) != seed:
        raise RuntimeError(
            "Config seed mismatch."
        )

    ckpt = torch.load(
        run_dir
        /
        "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )

    state_dict = ckpt[
        "model_state_dict"
    ]

    saved = np.load(
        run_dir
        /
        "predictions.npz",
        allow_pickle=True,
    )

    # ========================================================
    # Reconstruct test data
    # ========================================================

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

    if X_test.shape[0] != 105:
        raise RuntimeError(
            f"Expected 105 test windows, "
            f"got {X_test.shape}"
        )

    stored_targets = np.asarray(
        saved[
            "test_target_indices"
        ],
        dtype=np.int64,
    )

    if not np.array_equal(
        test_targets,
        stored_targets,
    ):
        raise RuntimeError(
            "Test target-index mismatch."
        )

    # ========================================================
    # Construct exact trained states ONCE
    # ========================================================

    state_circuit = (
        build_state_qnode(
            encoding,
            config,
            state_dict,
        )
    )

    states = np.empty(
        (
            105,
            2 ** 14,
        ),
        dtype=np.complex128,
    )

    max_norm_error = 0.0

    for i in range(
        105
    ):

        psi = np.asarray(
            state_circuit(
                X_test[i]
            ),
            dtype=np.complex128,
        )

        states[i] = psi

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

    # ========================================================
    # Reconstruct production prediction for validation
    # ========================================================

    z_signs = make_z_signs(
        14
    )

    z_readouts = (
        local_z_from_states(
            states,
            z_signs,
        )
    )

    head_weight = (
        state_dict[
            "classical_head.0.weight"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
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

    latent = (
        z_readouts
        @
        head_weight
        +
        head_bias
    )

    reconstructed_scaled = np.tanh(
        latent
    )

    saved_scaled = np.asarray(
        saved[
            "y_test_pred_scaled"
        ],
        dtype=np.float64,
    )

    max_scaled_pred_error = float(
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
        saved[
            "y_test_pred_original"
        ],
        dtype=np.float64,
    )

    max_original_pred_error = float(
        np.max(
            np.abs(
                reconstructed_original
                -
                saved_original
            )
        )
    )

    if max_norm_error > 1e-10:
        raise RuntimeError(
            f"State norm failure: "
            f"{max_norm_error}"
        )

    if max_scaled_pred_error > 1e-10:
        raise RuntimeError(
            f"Prediction reconstruction failure: "
            f"{max_scaled_pred_error}"
        )

    # ========================================================
    # Exact temporal fidelity
    # ========================================================

    d = 2 ** 14

    haar_mean_fidelity = (
        1.0
        /
        d
    )

    pair_rows = []
    lag_rows = []

    for tau in range(
        1,
        args.max_tau + 1,
    ):

        psi_a = states[
            :-tau
        ]

        psi_b = states[
            tau:
        ]

        overlaps = np.einsum(
            "ij,ij->i",
            np.conjugate(
                psi_a
            ),
            psi_b,
            optimize=True,
        )

        fidelities = (
            np.abs(
                overlaps
            )
            **
            2
        )

        # Numerical clipping only.
        fidelities = np.clip(
            fidelities,
            0.0,
            1.0,
        )

        infidelities = (
            1.0
            -
            fidelities
        )

        log_fidelities = np.log(
            np.maximum(
                fidelities,
                1e-300,
            )
        )

        log10_fidelities = np.log10(
            np.maximum(
                fidelities,
                1e-300,
            )
        )

        relative_to_haar = (
            fidelities
            /
            haar_mean_fidelity
        )

        n_pairs = len(
            fidelities
        )

        if n_pairs != (
            105 - tau
        ):
            raise RuntimeError(
                "Unexpected number of pairs."
            )

        for i in range(
            n_pairs
        ):

            pair_rows.append({
                "encoding":
                    encoding,

                "depth":
                    depth,

                "seed":
                    seed,

                "tau":
                    tau,

                "pair_index":
                    i,

                "target_index_a":
                    int(
                        test_targets[i]
                    ),

                "target_index_b":
                    int(
                        test_targets[
                            i + tau
                        ]
                    ),

                "fidelity":
                    float(
                        fidelities[i]
                    ),

                "infidelity":
                    float(
                        infidelities[i]
                    ),

                "log_fidelity":
                    float(
                        log_fidelities[i]
                    ),

                "log10_fidelity":
                    float(
                        log10_fidelities[i]
                    ),

                "fidelity_over_haar_mean":
                    float(
                        relative_to_haar[i]
                    ),
            })

        lag_rows.append({
            "encoding":
                encoding,

            "depth":
                depth,

            "seed":
                seed,

            "tau":
                tau,

            "n_pairs":
                n_pairs,

            "mean_fidelity":
                float(
                    np.mean(
                        fidelities
                    )
                ),

            "std_fidelity_over_time":
                float(
                    np.std(
                        fidelities,
                        ddof=1,
                    )
                ),

            "median_fidelity":
                float(
                    np.median(
                        fidelities
                    )
                ),

            "min_fidelity":
                float(
                    np.min(
                        fidelities
                    )
                ),

            "max_fidelity":
                float(
                    np.max(
                        fidelities
                    )
                ),

            "mean_infidelity":
                float(
                    np.mean(
                        infidelities
                    )
                ),

            "mean_log_fidelity":
                float(
                    np.mean(
                        log_fidelities
                    )
                ),

            "geometric_mean_fidelity":
                float(
                    np.exp(
                        np.mean(
                            log_fidelities
                        )
                    )
                ),

            "mean_fidelity_over_haar":
                float(
                    np.mean(
                        relative_to_haar
                    )
                ),

            "median_fidelity_over_haar":
                float(
                    np.median(
                        relative_to_haar
                    )
                ),
        })

    pair_df = pd.DataFrame(
        pair_rows
    )

    lag_df = pd.DataFrame(
        lag_rows
    )

    # ========================================================
    # Save
    # ========================================================

    pair_df.to_csv(
        out_dir
        /
        "fidelity_pairs.csv.gz",
        index=False,
        compression="gzip",
    )

    lag_df.to_csv(
        out_dir
        /
        "fidelity_lag_summary.csv",
        index=False,
    )

    tau12 = lag_df[
        lag_df["tau"] == 12
    ].iloc[0]

    validation = {
        "status":
            "PASS",

        "manifest_index":
            int(
                args.manifest_index
            ),

        "encoding":
            encoding,

        "depth":
            depth,

        "seed":
            seed,

        "n_test_windows":
            105,

        "max_tau":
            int(
                args.max_tau
            ),

        "max_state_norm_error":
            max_norm_error,

        "max_scaled_prediction_error":
            max_scaled_pred_error,

        "max_original_prediction_error":
            max_original_pred_error,

        "hilbert_dimension":
            d,

        "haar_mean_fidelity":
            haar_mean_fidelity,

        "tau12_n_pairs":
            int(
                tau12[
                    "n_pairs"
                ]
            ),

        "tau12_mean_fidelity":
            float(
                tau12[
                    "mean_fidelity"
                ]
            ),

        "tau12_median_fidelity":
            float(
                tau12[
                    "median_fidelity"
                ]
            ),

        "tau12_mean_fidelity_over_haar":
            float(
                tau12[
                    "mean_fidelity_over_haar"
                ]
            ),
    }

    with open(
        out_dir
        /
        "run_validation.json",
        "w",
    ) as f:

        json.dump(
            validation,
            f,
            indent=2,
        )

    print()
    print(
        f"max norm error:          "
        f"{max_norm_error:.3e}"
    )

    print(
        f"max prediction error:    "
        f"{max_scaled_pred_error:.3e}"
    )

    print(
        f"Fbar(tau=12):            "
        f"{tau12['mean_fidelity']:.8e}"
    )

    print(
        f"median F(tau=12):        "
        f"{tau12['median_fidelity']:.8e}"
    )

    print(
        f"Fbar/Haar at tau=12:     "
        f"{tau12['mean_fidelity_over_haar']:.4f}"
    )

    print(
        f"N pairs at tau=12:       "
        f"{int(tau12['n_pairs'])}"
    )

    print()
    print(
        "STATUS: PASS"
    )


if __name__ == "__main__":
    main()
