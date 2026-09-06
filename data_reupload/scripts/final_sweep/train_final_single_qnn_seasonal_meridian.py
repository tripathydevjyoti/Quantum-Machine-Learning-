#!/usr/bin/env python

"""
Seasonal-meridian encoding intervention for the final direct-window QNN.

Everything downstream of the encoding is inherited from
train_final_single_qnn.py:

    - same chronological split
    - same fixed target scaling [0,350] -> [-1,1]
    - same trainable Rot gates
    - same CNOT ring
    - same all-local-Z readout
    - same linear head + tanh
    - same Adam training
    - same validation checkpoint selection
    - same metrics
    - same saved artifacts

ONLY THE INPUT ENCODING CHANGES.

For an observation at absolute time index s:

    u_s     = F_train(x_s)
    z_s     = 2*u_s - 1
    theta_s = arccos(z_s)
    phi_s   = 2*pi*(s mod 12)/12

and the single-qubit encoding is

    RZ(phi_s) RY(theta_s) |0>

implemented in PennyLane operation order as:

    qml.RY(theta_s)
    qml.RZ(phi_s)

The empirical CDF is fitted using training observations only:
absolute indices 0,...,363 for the production split.
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml


# ============================================================
# LOAD THE ORIGINAL PRODUCTION TRAINING MODULE
# ============================================================

THIS_FILE = Path(__file__).resolve()
BASE_FILE = THIS_FILE.with_name("train_final_single_qnn.py")

if not BASE_FILE.exists():
    raise FileNotFoundError(
        f"Production baseline script not found: {BASE_FILE}"
    )

spec = importlib.util.spec_from_file_location(
    "production_final_single_qnn",
    BASE_FILE,
)

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


# ============================================================
# GLOBAL ENCODING AUDIT INFORMATION
# ============================================================

SEASON_PERIOD = 12

ENCODING_AUDIT = {}
CDF_REFERENCE_SORTED = None


# ============================================================
# EMPIRICAL CDF
# ============================================================

def empirical_cdf_right(values, sorted_reference):
    """
    Standard right-continuous empirical CDF:

        F_train(x) = (# training observations <= x) / n_train_obs

    Validation/test values are evaluated against the same
    frozen training reference.
    """
    values = np.asarray(values, dtype=np.float64)
    sorted_reference = np.asarray(sorted_reference, dtype=np.float64)

    counts = np.searchsorted(
        sorted_reference,
        values,
        side="right",
    )

    return counts.astype(np.float64) / float(len(sorted_reference))


# ============================================================
# SEASONAL-MERIDIAN WINDOW CONSTRUCTION
# ============================================================

def make_windows_seasonal_meridian(
    y_original,
    window_size,
    train_end_idx,
    train_window_end,
    scaler_data_min,
    scaler_data_max,
    scaler_feature_min,
    scaler_feature_max,
):
    """
    Construct the exact original supervised split, but replace
    each scalar input x_s by [theta_s, phi_s].

    Target y remains under the original fixed physical scaling.
    """

    global ENCODING_AUDIT
    global CDF_REFERENCE_SORTED

    y_original = np.asarray(
        y_original,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Original target scaling is completely unchanged.
    # --------------------------------------------------------

    y_scaled = base.scale_fixed(
        y_original,
        data_min=scaler_data_min,
        data_max=scaler_data_max,
        feature_min=scaler_feature_min,
        feature_max=scaler_feature_max,
    )

    # --------------------------------------------------------
    # TRAIN-ONLY CDF REFERENCE
    #
    # Training targets run from:
    #
    #   window_size
    # through
    #   window_size + train_window_end - 1
    #
    # With W=14 and 350 training samples:
    #
    #   targets = 14,...,363
    #
    # Therefore the training-only observed series is:
    #
    #   s = 0,...,363
    #
    # n = 364.
    # --------------------------------------------------------

    train_obs_stop_exclusive = (
        int(window_size)
        +
        int(train_window_end)
    )

    train_reference = y_original[
        :train_obs_stop_exclusive
    ]

    CDF_REFERENCE_SORTED = np.sort(
        train_reference.copy()
    )

    n_reference = len(
        CDF_REFERENCE_SORTED
    )

    # --------------------------------------------------------
    # Build all windows using ABSOLUTE observation indices.
    # --------------------------------------------------------

    X = []
    y_t = []
    target_indices = []
    source_indices_all = []

    for target_idx in range(
        window_size,
        len(y_original),
    ):

        source_indices = np.arange(
            target_idx - window_size,
            target_idx,
            dtype=int,
        )

        source_values = y_original[
            source_indices
        ]

        # --------------------------------------------
        # VALUE COORDINATE
        # --------------------------------------------

        u = empirical_cdf_right(
            source_values,
            CDF_REFERENCE_SORTED,
        )

        z = 2.0 * u - 1.0

        # Numerical safety only.
        z = np.clip(
            z,
            -1.0,
            1.0,
        )

        theta = np.arccos(z)

        # --------------------------------------------
        # SEASONAL PHASE COORDINATE
        #
        # CRITICAL:
        #
        # phase depends on the ABSOLUTE observation
        # index s, NOT the position q inside the
        # 14-step window.
        # --------------------------------------------

        seasonal_position = (
            source_indices
            %
            SEASON_PERIOD
        )

        phi = (
            2.0
            *
            np.pi
            *
            seasonal_position
            /
            float(SEASON_PERIOD)
        )

        # Shape for one input window:
        #
        #   (14, 2)
        #
        # [:,0] = theta
        # [:,1] = phi

        encoded_window = np.stack(
            [
                theta,
                phi,
            ],
            axis=-1,
        )

        X.append(
            encoded_window
        )

        y_t.append(
            y_scaled[target_idx]
        )

        target_indices.append(
            target_idx
        )

        source_indices_all.append(
            source_indices
        )

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    y_t = np.asarray(
        y_t,
        dtype=np.float64,
    )

    target_indices = np.asarray(
        target_indices,
        dtype=int,
    )

    source_indices_all = np.asarray(
        source_indices_all,
        dtype=int,
    )

    # --------------------------------------------------------
    # EXACT ORIGINAL SPLIT
    # --------------------------------------------------------

    n_train = int(
        train_window_end
    )

    n_val = int(
        train_end_idx
        -
        window_size
        -
        train_window_end
    )

    X_train = X[
        :n_train
    ]

    y_train = y_t[
        :n_train
    ]

    idx_train = target_indices[
        :n_train
    ]

    X_val = X[
        n_train:
        n_train + n_val
    ]

    y_val = y_t[
        n_train:
        n_train + n_val
    ]

    idx_val = target_indices[
        n_train:
        n_train + n_val
    ]

    X_test = X[
        n_train + n_val:
    ]

    y_test = y_t[
        n_train + n_val:
    ]

    idx_test = target_indices[
        n_train + n_val:
    ]

    # ========================================================
    # HARD VALIDATION OF THE PHASE CONSTRUCTION
    # ========================================================

    if window_size != 14:
        raise RuntimeError(
            "Seasonal production experiment currently expects "
            "window_size=14."
        )

    if SEASON_PERIOD != 12:
        raise RuntimeError(
            "Season period must be 12."
        )

    if len(idx_train) != 350:
        raise RuntimeError(
            f"Expected 350 training targets, got {len(idx_train)}."
        )

    if len(idx_val) != 70:
        raise RuntimeError(
            f"Expected 70 validation targets, got {len(idx_val)}."
        )

    if len(idx_test) != 105:
        raise RuntimeError(
            f"Expected 105 test targets, got {len(idx_test)}."
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

    if not np.array_equal(
        idx_train,
        expected_train_idx,
    ):
        raise RuntimeError(
            "Training target indices do not match 14..363."
        )

    if not np.array_equal(
        idx_val,
        expected_val_idx,
    ):
        raise RuntimeError(
            "Validation target indices do not match 364..433."
        )

    if not np.array_equal(
        idx_test,
        expected_test_idx,
    ):
        raise RuntimeError(
            "Test target indices do not match 434..538."
        )

    if n_reference != 364:
        raise RuntimeError(
            f"Expected 364 training-only CDF observations, "
            f"got {n_reference}."
        )

    # --------------------------------------------------------
    # Verify absolute-index phase rule explicitly.
    # --------------------------------------------------------

    all_s = np.arange(
        len(y_original),
        dtype=int,
    )

    phi_by_absolute_index = (
        2.0
        *
        np.pi
        *
        (all_s % SEASON_PERIOD)
        /
        float(SEASON_PERIOD)
    )

    # s and s+12 must have identical phase.
    if not np.allclose(
        phi_by_absolute_index[:-12],
        phi_by_absolute_index[12:],
        atol=1e-15,
        rtol=0.0,
    ):
        raise RuntimeError(
            "FAILED: phi(s+12) != phi(s)."
        )

    # s and s+6 must differ by pi modulo 2*pi.
    delta6 = (
        phi_by_absolute_index[6:]
        -
        phi_by_absolute_index[:-6]
    ) % (2.0 * np.pi)

    if not np.allclose(
        delta6,
        np.pi,
        atol=1e-15,
        rtol=0.0,
    ):
        raise RuntimeError(
            "FAILED: phi(s+6)-phi(s) != pi mod 2*pi."
        )

    # --------------------------------------------------------
    # Verify the actual first training window.
    #
    # target t=14:
    # source indices = 0,...,13
    # seasonal positions = 0,...,11,0,1
    # --------------------------------------------------------

    first_source = source_indices_all[0]

    expected_first_source = np.arange(
        0,
        14,
    )

    if not np.array_equal(
        first_source,
        expected_first_source,
    ):
        raise RuntimeError(
            "First training source window is incorrect."
        )

    expected_first_mod = np.array(
        [
            0, 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10, 11,
            0, 1,
        ],
        dtype=int,
    )

    first_mod = (
        first_source
        %
        SEASON_PERIOD
    )

    if not np.array_equal(
        first_mod,
        expected_first_mod,
    ):
        raise RuntimeError(
            "First window seasonal modulo pattern is incorrect."
        )

    # --------------------------------------------------------
    # Check first validation and first test window indexing.
    # --------------------------------------------------------

    first_val_global_row = n_train

    first_test_global_row = (
        n_train
        +
        n_val
    )

    first_val_source = source_indices_all[
        first_val_global_row
    ]

    first_test_source = source_indices_all[
        first_test_global_row
    ]

    if not np.array_equal(
        first_val_source,
        np.arange(350, 364),
    ):
        raise RuntimeError(
            "First validation window should use source indices 350..363."
        )

    if not np.array_equal(
        first_test_source,
        np.arange(420, 434),
    ):
        raise RuntimeError(
            "First test window should use source indices 420..433."
        )

    # ========================================================
    # AUDIT OUTPUT
    # ========================================================

    ENCODING_AUDIT = {
        "encoding": "seasonal_meridian",
        "season_period": int(SEASON_PERIOD),

        "phase_formula":
            "2*pi*((absolute_source_index mod 12)/12)",

        "value_formula":
            "theta=arccos(2*F_train(x)-1)",

        "cdf_convention":
            "right-continuous empirical CDF: count(train_reference <= x)/n",

        "cdf_reference_absolute_start": 0,
        "cdf_reference_absolute_stop_exclusive":
            int(train_obs_stop_exclusive),

        "cdf_reference_absolute_end_inclusive":
            int(train_obs_stop_exclusive - 1),

        "cdf_reference_n":
            int(n_reference),

        "train_target_start":
            int(idx_train[0]),

        "train_target_end":
            int(idx_train[-1]),

        "val_target_start":
            int(idx_val[0]),

        "val_target_end":
            int(idx_val[-1]),

        "test_target_start":
            int(idx_test[0]),

        "test_target_end":
            int(idx_test[-1]),

        "first_train_source_indices":
            first_source.tolist(),

        "first_train_source_mod12":
            first_mod.tolist(),

        "first_val_source_indices":
            first_val_source.tolist(),

        "first_val_source_mod12":
            (
                first_val_source
                %
                SEASON_PERIOD
            ).tolist(),

        "first_test_source_indices":
            first_test_source.tolist(),

        "first_test_source_mod12":
            (
                first_test_source
                %
                SEASON_PERIOD
            ).tolist(),

        "X_train_shape":
            list(X_train.shape),

        "X_val_shape":
            list(X_val.shape),

        "X_test_shape":
            list(X_test.shape),

        "status":
            "PASS",
    }

    print()
    print("=" * 100)
    print("SEASONAL-MERIDIAN ENCODING AUDIT")
    print("=" * 100)

    for key, value in ENCODING_AUDIT.items():
        print(f"{key}: {value}")

    print("=" * 100)
    print("SEASONAL ENCODING AUDIT: PASS")
    print("=" * 100)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "idx_train": idx_train,

        "X_val": X_val,
        "y_val": y_val,
        "idx_val": idx_val,

        "X_test": X_test,
        "y_test": y_test,
        "idx_test": idx_test,

        "y_original": y_original,
        "y_scaled": y_scaled,
    }


# ============================================================
# SEASONAL-MERIDIAN QNN
# ============================================================

class SeasonalMeridianDirectWindowQNN(nn.Module):

    def __init__(
        self,
        n_qubits,
        depth,
        n_ansatz_layers,
        readout,
        angle_factor,
        qml_device,
        diff_method,
        use_output_tanh,
    ):
        super().__init__()

        self.n_qubits = int(
            n_qubits
        )

        self.depth = int(
            depth
        )

        self.n_ansatz_layers = int(
            n_ansatz_layers
        )

        self.readout = readout

        # Retained in the signature for exact compatibility with
        # the production training script.  It is NOT used by the
        # seasonal-meridian encoder.
        self.angle_factor = float(
            angle_factor
        )

        self.use_output_tanh = bool(
            use_output_tanh
        )

        dev = qml.device(
            qml_device,
            wires=self.n_qubits,
        )

        @qml.qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
        )
        def circuit(x, weights):

            # x shape:
            #
            #   (14, 2)
            #
            # x[q,0] = theta_q
            # x[q,1] = phi_q

            for reupload_layer in range(
                self.depth
            ):

                # ============================================
                # SEASONAL-MERIDIAN DATA UPLOAD
                #
                # PennyLane applies the operations in this
                # order to |0>:
                #
                #     RY(theta)
                #     then RZ(phi)
                #
                # giving:
                #
                #     RZ(phi) RY(theta) |0>.
                # ============================================

                for q in range(
                    self.n_qubits
                ):

                    theta_q = x[
                        q,
                        0,
                    ]

                    phi_q = x[
                        q,
                        1,
                    ]

                    qml.RY(
                        theta_q,
                        wires=q,
                    )

                    qml.RZ(
                        phi_q,
                        wires=q,
                    )

                # ============================================
                # ORIGINAL TRAINABLE ANSATZ -- UNCHANGED
                # ============================================

                for ansatz_layer in range(
                    self.n_ansatz_layers
                ):

                    for q in range(
                        self.n_qubits
                    ):

                        qml.Rot(
                            weights[
                                reupload_layer,
                                ansatz_layer,
                                q,
                                0,
                            ],
                            weights[
                                reupload_layer,
                                ansatz_layer,
                                q,
                                1,
                            ],
                            weights[
                                reupload_layer,
                                ansatz_layer,
                                q,
                                2,
                            ],
                            wires=q,
                        )

                    # Original CNOT chain + ring closure.
                    if self.n_qubits > 1:

                        for q in range(
                            self.n_qubits - 1
                        ):

                            qml.CNOT(
                                wires=[
                                    q,
                                    q + 1,
                                ]
                            )

                        qml.CNOT(
                            wires=[
                                self.n_qubits - 1,
                                0,
                            ]
                        )

            # Original all-local-Z readout -- unchanged.
            return [
                qml.expval(
                    qml.PauliZ(q)
                )
                for q
                in range(
                    self.n_qubits
                )
            ]

        self.circuit = circuit

        # Original quantum initialization -- unchanged.
        self.q_weights = nn.Parameter(
            0.01
            *
            torch.randn(
                self.depth,
                self.n_ansatz_layers,
                self.n_qubits,
                3,
                dtype=torch.float64,
            )
        )

        # Original classical head -- unchanged.
        if self.use_output_tanh:

            self.classical_head = nn.Sequential(
                nn.Linear(
                    self.n_qubits,
                    1,
                    dtype=torch.float64,
                ),
                nn.Tanh(),
            )

        else:

            self.classical_head = nn.Linear(
                self.n_qubits,
                1,
                dtype=torch.float64,
            )

    def forward(
        self,
        x_batch,
    ):

        outputs = []

        for i in range(
            x_batch.shape[0]
        ):

            z = self.circuit(
                x_batch[i],
                self.q_weights,
            )

            if isinstance(
                z,
                (
                    tuple,
                    list,
                ),
            ):

                z = torch.stack(
                    list(z)
                )

            outputs.append(
                z
            )

        z_batch = torch.stack(
            outputs,
            dim=0,
        )

        y = self.classical_head(
            z_batch
        ).squeeze(-1)

        return y


# ============================================================
# SAVE ENCODING PROVENANCE AFTER ORIGINAL TRAINER COMPLETES
# ============================================================

def save_encoding_provenance():

    global ENCODING_AUDIT
    global CDF_REFERENCE_SORTED

    parser = argparse.ArgumentParser(
        add_help=False
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--variant_id",
        default="seasonal_meridian_tanh_a1",
    )

    parser.add_argument(
        "--depth",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    known, _ = parser.parse_known_args()

    out_dir = (
        Path(known.output_dir)
        /
        known.variant_id
        /
        f"depth_{known.depth}"
        /
        f"seed_{known.seed}"
    )

    metadata = dict(
        ENCODING_AUDIT
    )

    metadata.update({
        "baseline_training_script":
            str(BASE_FILE),

        "encoding_script":
            str(THIS_FILE),

        "encoding_operator":
            "RZ(phi_s) RY(theta_s) |0>",

        "ansatz_changed":
            False,

        "readout_changed":
            False,

        "classical_head_changed":
            False,

        "optimizer_protocol_changed":
            False,

        "metric_protocol_changed":
            False,
    })

    with open(
        out_dir
        /
        "encoding_metadata.json",
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    np.savez(
        out_dir
        /
        "seasonal_encoding_reference.npz",

        train_cdf_sorted_values=
            CDF_REFERENCE_SORTED,

        season_period=
            np.array(
                [SEASON_PERIOD],
                dtype=int,
            ),
    )

    # --------------------------------------------------------
    # Add encoding provenance to config.json.
    # --------------------------------------------------------

    config_path = (
        out_dir
        /
        "config.json"
    )

    with open(
        config_path
    ) as f:

        config = json.load(
            f
        )

    config.update(
        metadata
    )

    with open(
        config_path,
        "w",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Add simple encoding labels to result_summary.csv.
    # All original metric columns remain untouched.
    # --------------------------------------------------------

    summary_path = (
        out_dir
        /
        "result_summary.csv"
    )

    summary = pd.read_csv(
        summary_path
    )

    summary[
        "encoding"
    ] = "seasonal_meridian"

    summary[
        "season_period"
    ] = SEASON_PERIOD

    summary[
        "cdf_reference_n"
    ] = len(
        CDF_REFERENCE_SORTED
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 100)
    print("SEASONAL ENCODING PROVENANCE SAVED")
    print("=" * 100)

    print(
        out_dir
        /
        "encoding_metadata.json"
    )

    print(
        out_dir
        /
        "seasonal_encoding_reference.npz"
    )

    print("=" * 100)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Replace ONLY these two production components.
    base.make_windows = (
        make_windows_seasonal_meridian
    )

    base.DirectWindowQNN = (
        SeasonalMeridianDirectWindowQNN
    )

    # Everything else comes from the exact original trainer.
    base.main()

    save_encoding_provenance()
