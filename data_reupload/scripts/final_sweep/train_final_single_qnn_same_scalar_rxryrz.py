#!/usr/bin/env python

"""
Production A=1 direct-QNN trainer with the Phase-I same-scalar
Rx(theta) -> Ry(theta) -> Rz(theta) encoding.

Only the quantum input encoding is changed relative to
train_final_single_qnn.py.

Original:
    theta = angle_factor * x[q]
    RY(theta)

Same-scalar control:
    theta = angle_factor * x[q]
    RX(theta)
    RY(theta)
    RZ(theta)

For angle_factor = pi:
    theta = pi * x[q]

Circuit application order:
    RX -> RY -> RZ

Matrix acting on |psi>:
    U = RZ RY RX

All window construction, target scaling, ansatz layers,
CNOT topology, readout, classical head, optimization,
checkpointing, metrics, and artifact generation are inherited
from the original production trainer.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pennylane as qml
import torch
from torch import nn


# ============================================================
# Locate and import exact production trainer
# ============================================================

THIS_FILE = Path(__file__).resolve()

BASE_SCRIPT = (
    THIS_FILE.parent
    / "train_final_single_qnn.py"
).resolve()

spec = importlib.util.spec_from_file_location(
    "base_single_qnn_trainer",
    BASE_SCRIPT,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Could not import production trainer: {BASE_SCRIPT}"
    )

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


# ============================================================
# Same-scalar RxRyRz QNN
# ============================================================

class SameScalarRxRyRzDirectWindowQNN(nn.Module):

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

        self.n_qubits = int(n_qubits)
        self.depth = int(depth)
        self.n_ansatz_layers = int(n_ansatz_layers)
        self.readout = readout
        self.angle_factor = float(angle_factor)
        self.use_output_tanh = bool(use_output_tanh)

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

            for reupload_layer in range(
                self.depth
            ):

                # ============================================
                # SAME-SCALAR RX-RY-RZ DATA UPLOAD
                #
                # Same scalar angle on all three axes:
                #
                # theta_q = angle_factor * x[q]
                #
                # PennyLane circuit order:
                #
                # RX(theta_q)
                # RY(theta_q)
                # RZ(theta_q)
                #
                # Therefore the matrix acting on the state is
                #
                # U = RZ(theta_q) RY(theta_q) RX(theta_q)
                #
                # matching Phase-I exactly.
                # ============================================

                for q in range(
                    self.n_qubits
                ):

                    theta_q = (
                        self.angle_factor
                        *
                        x[q]
                    )

                    qml.RX(
                        theta_q,
                        wires=q,
                    )

                    qml.RY(
                        theta_q,
                        wires=q,
                    )

                    qml.RZ(
                        theta_q,
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

                    # Original CNOT chain + closing ring.
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

            # Original all-local-Z readout.
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

        # Original initialization.
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

        # Original classical head.
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

            outputs.append(z)

        z_batch = torch.stack(
            outputs,
            dim=0,
        )

        y = self.classical_head(
            z_batch
        ).squeeze(-1)

        return y


# ============================================================
# CLI helpers for provenance
# ============================================================

def cli_value(
    flag: str,
    default=None,
):

    if flag not in sys.argv:
        return default

    i = sys.argv.index(flag)

    if i + 1 >= len(sys.argv):
        raise RuntimeError(
            f"Missing value after {flag}"
        )

    return sys.argv[i + 1]


def save_encoding_provenance():

    output_root = cli_value(
        "--output_dir"
    )

    variant_id = cli_value(
        "--variant_id",
        "same_scalar_rxryrz_tanh_a1",
    )

    depth = int(
        cli_value(
            "--depth"
        )
    )

    seed = int(
        cli_value(
            "--seed",
            42,
        )
    )

    angle_factor_raw = cli_value(
        "--angle_factor",
        str(math.pi),
    )

    angle_factor = float(
        angle_factor_raw
    )

    out_dir = (
        Path(output_root)
        / variant_id
        / f"depth_{depth}"
        / f"seed_{seed}"
    )

    metadata = {
        "status": "PASS",

        "encoding":
            "same_scalar_rxryrz",

        "encoding_label":
            "Same-scalar RxRyRz",

        "phase1_reference_script":
            str(
                (
                    THIS_FILE.parents[2]
                    / "analysis"
                    / "theory_diagnostics"
                    / "scripts"
                    / "analyze_theory_phase12.py"
                ).resolve()
            ),

        "baseline_training_script":
            str(BASE_SCRIPT),

        "encoding_script":
            str(THIS_FILE),

        "input_preprocessing_changed":
            False,

        "window_construction_changed":
            False,

        "target_scaling_changed":
            False,

        "ansatz_changed":
            False,

        "entanglement_changed":
            False,

        "readout_changed":
            False,

        "classical_head_changed":
            False,

        "optimizer_protocol_changed":
            False,

        "checkpoint_protocol_changed":
            False,

        "metric_protocol_changed":
            False,

        "angle_factor":
            angle_factor,

        "angle_formula":
            "theta_q = angle_factor * x_q",

        "production_angle_formula":
            "theta_q = pi * x_q",

        "same_scalar_on_all_axes":
            True,

        "gate_application_order": [
            "RX(theta_q)",
            "RY(theta_q)",
            "RZ(theta_q)",
        ],

        "unitary_matrix_order":
            "RZ(theta_q) RY(theta_q) RX(theta_q)",

        "phase1_matrix_convention":
            "U = Rz(th) @ Ry(th) @ Rx(th)",

        "expected_scaled_input_interval": [
            -1.0,
            1.0,
        ],

        "expected_angle_interval": [
            -angle_factor,
            angle_factor,
        ],

        "depth":
            depth,

        "seed":
            seed,
    }

    metadata_path = (
        out_dir
        / "encoding_metadata.json"
    )

    with open(
        metadata_path,
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Add provenance to config.json
    # --------------------------------------------------------

    config_path = (
        out_dir
        / "config.json"
    )

    if config_path.exists():

        with open(
            config_path
        ) as f:
            cfg = json.load(f)

        cfg.update({
            "encoding":
                "same_scalar_rxryrz",

            "encoding_label":
                "Same-scalar RxRyRz",

            "encoding_operator":
                "RZ(theta) RY(theta) RX(theta) |psi>",

            "gate_application_order":
                "RX(theta) -> RY(theta) -> RZ(theta)",

            "same_scalar_on_all_axes":
                True,

            "angle_formula":
                "theta = angle_factor * x",

            "phase1_convention_matched":
                True,

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
            config_path,
            "w",
        ) as f:
            json.dump(
                cfg,
                f,
                indent=2,
        )

    # --------------------------------------------------------
    # Add provenance columns to result_summary.csv
    # --------------------------------------------------------

    result_path = (
        out_dir
        / "result_summary.csv"
    )

    if result_path.exists():

        df = pd.read_csv(
            result_path
        )

        df["encoding"] = (
            "same_scalar_rxryrz"
        )

        df["same_scalar_on_all_axes"] = (
            True
        )

        df["phase1_convention_matched"] = (
            True
        )

        df.to_csv(
            result_path,
            index=False,
        )

    print()
    print("=" * 100)
    print(
        "Same-scalar RxRyRz encoding provenance"
    )
    print("=" * 100)

    print(
        "encoding:",
        metadata["encoding"],
    )

    print(
        "gate order:",
        " -> ".join(
            metadata[
                "gate_application_order"
            ]
        ),
    )

    print(
        "matrix order:",
        metadata[
            "unitary_matrix_order"
        ],
    )

    print(
        "angle formula:",
        metadata[
            "production_angle_formula"
        ],
    )

    print(
        "metadata:",
        metadata_path,
    )

    print(
        "status: PASS"
    )

    print("=" * 100)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Preserve original make_windows exactly.
    #
    # Only replace the quantum model class.
    base.DirectWindowQNN = (
        SameScalarRxRyRzDirectWindowQNN
    )

    # Everything else comes from the exact original trainer.
    base.main()

    save_encoding_provenance()
