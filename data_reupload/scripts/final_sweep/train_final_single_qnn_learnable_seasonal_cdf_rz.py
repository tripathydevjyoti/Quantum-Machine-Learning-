#!/usr/bin/env python

"""
Learnable CDF-anchored seasonal-meridian QNN.

This is a nested trainable extension of the production
seasonal-meridian encoding.

Fixed seasonal model:
    theta_cdf = arccos(2 F_train(x) - 1)
    phi_m     = 2*pi*m/12

    RY(theta_cdf)
    RZ(phi_m)

Learnable model:
    strength[m,l] = 1 + tanh(strength_raw[m,l])
    shift[m,l]    = (pi/2) * tanh(shift_raw[m,l])

    theta =
        pi/2
        + shift[m,l]
        + strength[m,l] * (theta_cdf - pi/2)

    RY(theta)
    RZ(phi_m)

At initialization:
    strength_raw = 0
    shift_raw    = 0

therefore:
    strength = 1
    shift    = 0
    theta    = theta_cdf

so the model reduces EXACTLY to the fixed
seasonal-meridian encoder.

Only the encoding becomes trainable.  The production ansatz,
CNOT topology, readout, classical head, optimizer protocol,
checkpointing, data splits, and metrics remain unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from torch import nn


# ============================================================
# Import the validated fixed seasonal production wrapper
# ============================================================

THIS_FILE = Path(__file__).resolve()

SEASONAL_SCRIPT = (
    THIS_FILE.parent
    / "train_final_single_qnn_seasonal_meridian.py"
).resolve()

spec = importlib.util.spec_from_file_location(
    "fixed_seasonal_wrapper",
    SEASONAL_SCRIPT,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Could not import seasonal wrapper: {SEASONAL_SCRIPT}"
    )

seasonal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seasonal)

# Reuse the exact original production trainer imported by
# the validated seasonal wrapper.
base = seasonal.base


# ============================================================
# Global references populated during one production run
# ============================================================

CURRENT_MODEL = None
WINDOW_REFERENCE = {}
WINDOW_AUDIT = {}


# ============================================================
# Window construction
#
# IMPORTANT:
#
# We reuse the EXACT validated seasonal-meridian window
# construction, including:
#
#   - training-only empirical CDF,
#   - target splits,
#   - absolute source-time seasonal phase,
#   - no phase reset at split boundaries.
#
# The fixed wrapper returns:
#
#   X[...,0] = theta_CDF
#   X[...,1] = phi = 2*pi*m/12
#
# We replace the second channel with the corresponding integer
# seasonal index m, while retaining theta_CDF exactly.
# ============================================================

def make_windows_learnable_seasonal_cdf(
    *args,
    **kwargs,
):

    global WINDOW_REFERENCE
    global WINDOW_AUDIT

    data = (
        seasonal.make_windows_seasonal_meridian(
            *args,
            **kwargs,
        )
    )

    audit = {}
    refs = {}

    for split in [
        "train",
        "val",
        "test",
    ]:

        key = f"X_{split}"

        X_fixed = np.asarray(
            data[key],
            dtype=np.float64,
        )

        if (
            X_fixed.ndim != 3
            or
            X_fixed.shape[-1] != 2
        ):
            raise RuntimeError(
                f"Unexpected seasonal X shape "
                f"for {split}: {X_fixed.shape}"
            )

        theta_cdf = X_fixed[
            ...,
            0,
        ].copy()

        phi = X_fixed[
            ...,
            1,
        ].copy()

        # Recover m exactly from the already-validated
        # fixed seasonal phase.
        month = np.mod(
            np.rint(
                phi
                *
                12.0
                /
                (
                    2.0
                    *
                    np.pi
                )
            ).astype(np.int64),
            12,
        )

        phi_reconstructed = (
            2.0
            *
            np.pi
            *
            month
            /
            12.0
        )

        # Circular phase difference.
        phase_diff = np.angle(
            np.exp(
                1j
                *
                (
                    phi
                    -
                    phi_reconstructed
                )
            )
        )

        max_phase_diff = float(
            np.max(
                np.abs(
                    phase_diff
                )
            )
        )

        if max_phase_diff > 1e-12:
            raise RuntimeError(
                f"Could not reconstruct seasonal "
                f"indices for {split}; "
                f"max phase error={max_phase_diff}"
            )

        X_new = np.stack(
            [
                theta_cdf,
                month.astype(
                    np.float64
                ),
            ],
            axis=-1,
        )

        data[key] = X_new

        refs[
            f"{split}_theta_cdf"
        ] = theta_cdf

        refs[
            f"{split}_seasonal_index"
        ] = month

        refs[
            f"{split}_fixed_phi"
        ] = phi

        audit[
            f"{split}_shape"
        ] = list(
            X_new.shape
        )

        audit[
            f"{split}_month_min"
        ] = int(
            np.min(month)
        )

        audit[
            f"{split}_month_max"
        ] = int(
            np.max(month)
        )

        audit[
            f"{split}_phase_reconstruction_max_error"
        ] = max_phase_diff

    WINDOW_REFERENCE = refs
    WINDOW_AUDIT = audit

    return data


# ============================================================
# Learnable CDF seasonal QNN
# ============================================================

class LearnableSeasonalCDFRzDirectWindowQNN(
    nn.Module
):

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

        global CURRENT_MODEL

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

        # Retained solely for production-interface compatibility.
        # The CDF encoding does not use angle_factor.
        self.angle_factor = float(
            angle_factor
        )

        self.use_output_tanh = bool(
            use_output_tanh
        )

        # ----------------------------------------------------
        # Trainable encoding parameters
        #
        # Shape convention:
        #
        #   [seasonal_index, reupload_layer]
        #
        #   seasonal_index = 0,...,11
        #   layer          = 0,...,L-1
        #
        # INITIALIZATION:
        #
        # raw = 0
        #
        # -> strength = 1
        # -> shift = 0
        #
        # -> exactly fixed CDF seasonal meridian.
        # ----------------------------------------------------

        self.encoding_strength_raw = (
            nn.Parameter(
                torch.zeros(
                    12,
                    self.depth,
                    dtype=torch.float64,
                )
            )
        )

        self.encoding_shift_raw = (
            nn.Parameter(
                torch.zeros(
                    12,
                    self.depth,
                    dtype=torch.float64,
                )
            )
        )

        # Trainable residual seasonal azimuth.
        #
        # Shape:
        #   [seasonal_index, reupload_layer]
        #
        # raw = 0
        # -> phase_shift = 0
        # -> exact reduction to the polar-only learnable
        #    seasonal CDF encoder.
        #
        # The transformed residual is bounded by
        #
        #   |Delta phi| < pi/12,
        #
        # i.e. strictly less than half of the fixed
        # 30-degree separation between adjacent seasons.
        self.encoding_phase_raw = (
            nn.Parameter(
                torch.zeros(
                    12,
                    self.depth,
                    dtype=torch.float64,
                )
            )
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
        def circuit(
            x,
            weights,
        ):

            # x shape:
            #
            #   (14, 2)
            #
            # x[q,0] = fixed theta_CDF for this observation
            # x[q,1] = integer seasonal index m = 0,...,11

            for reupload_layer in range(
                self.depth
            ):

                # ============================================
                # LEARNABLE CDF-SEASONAL DATA UPLOAD
                # ============================================

                for q in range(
                    self.n_qubits
                ):

                    theta_cdf_q = x[
                        q,
                        0,
                    ]

                    # Seasonal index is known classical
                    # metadata and is not differentiated.
                    m_q = int(
                        round(
                            float(
                                x[
                                    q,
                                    1,
                                ]
                                .detach()
                                .cpu()
                                .item()
                            )
                        )
                    ) % 12

                    strength = (
                        1.0
                        +
                        torch.tanh(
                            self.encoding_strength_raw[
                                m_q,
                                reupload_layer,
                            ]
                        )
                    )

                    shift = (
                        math.pi
                        /
                        2.0
                        *
                        torch.tanh(
                            self.encoding_shift_raw[
                                m_q,
                                reupload_layer,
                            ]
                        )
                    )

                    phase_shift = (
                        math.pi
                        /
                        12.0
                        *
                        torch.tanh(
                            self.encoding_phase_raw[
                                m_q,
                                reupload_layer,
                            ]
                        )
                    )

                    theta_q = (
                        math.pi
                        /
                        2.0
                        +
                        shift
                        +
                        strength
                        *
                        (
                            theta_cdf_q
                            -
                            math.pi
                            /
                            2.0
                        )
                    )

                    phi_q = (
                        2.0
                        *
                        math.pi
                        *
                        m_q
                        /
                        12.0
                        +
                        phase_shift
                    )

                    # PennyLane circuit application:
                    #
                    # RY(theta)
                    # then RZ(phi)
                    #
                    # Matrix:
                    #
                    # RZ(phi) RY(theta)
                    qml.RY(
                        theta_q,
                        wires=q,
                    )

                    qml.RZ(
                        phi_q,
                        wires=q,
                    )

                # ============================================
                # ORIGINAL VARIATIONAL BLOCK -- UNCHANGED
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

        # ----------------------------------------------------
        # Original variational parameter initialization
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Original classical head
        # ----------------------------------------------------

        if self.use_output_tanh:

            self.classical_head = (
                nn.Sequential(
                    nn.Linear(
                        self.n_qubits,
                        1,
                        dtype=torch.float64,
                    ),
                    nn.Tanh(),
                )
            )

        else:

            self.classical_head = (
                nn.Linear(
                    self.n_qubits,
                    1,
                    dtype=torch.float64,
                )
            )

        CURRENT_MODEL = self

    # --------------------------------------------------------
    # Diagnostic helpers
    # --------------------------------------------------------

    def realized_strength(
        self,
    ):
        return (
            1.0
            +
            torch.tanh(
                self.encoding_strength_raw
            )
        )

    def realized_shift(
        self,
    ):
        return (
            math.pi
            /
            2.0
            *
            torch.tanh(
                self.encoding_shift_raw
            )
        )

    def realized_baseline(
        self,
    ):
        return (
            math.pi
            /
            2.0
            +
            self.realized_shift()
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
# CLI helpers
# ============================================================

def cli_value(
    flag,
    default=None,
):

    if flag not in sys.argv:
        return default

    i = sys.argv.index(flag)

    if i + 1 >= len(sys.argv):
        raise RuntimeError(
            f"Missing value after {flag}"
        )

    return sys.argv[
        i + 1
    ]


# ============================================================
# Save learned encoding and provenance
# ============================================================

def save_learned_encoding():

    global CURRENT_MODEL

    if CURRENT_MODEL is None:
        raise RuntimeError(
            "No trained model reference available."
        )

    output_root = cli_value(
        "--output_dir"
    )

    variant_id = cli_value(
        "--variant_id",
        "learnable_seasonal_cdf_rz_tanh_a1",
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

    n_ansatz_layers = int(
        cli_value(
            "--n_ansatz_layers",
            1,
        )
    )

    out_dir = (
        Path(output_root)
        /
        variant_id
        /
        f"depth_{depth}"
        /
        f"seed_{seed}"
    )

    model = CURRENT_MODEL

    # Base trainer restores the best validation checkpoint before
    # generating final predictions.  Therefore these matrices are
    # taken from that restored best model.

    with torch.no_grad():

        strength_raw = (
            model
            .encoding_strength_raw
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        shift_raw = (
            model
            .encoding_shift_raw
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        phase_raw = (
            model
            .encoding_phase_raw
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        strength = (
            1.0
            +
            np.tanh(
                strength_raw
            )
        )

        shift = (
            math.pi
            /
            2.0
            *
            np.tanh(
                shift_raw
            )
        )

        baseline = (
            math.pi
            /
            2.0
            +
            shift
        )

        phase_shift = (
            math.pi
            /
            12.0
            *
            np.tanh(
                phase_raw
            )
        )

        fixed_seasonal_phase = (
            np.repeat(
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
                depth,
                axis=1,
            )
        )

        learned_seasonal_phase_unwrapped = (
            fixed_seasonal_phase
            +
            phase_shift
        )

        learned_seasonal_phase_wrapped = (
            np.mod(
                learned_seasonal_phase_unwrapped,
                2.0 * math.pi,
            )
        )

    initial_strength = np.ones_like(
        strength
    )

    initial_shift = np.zeros_like(
        shift
    )

    initial_baseline = np.full_like(
        baseline,
        math.pi / 2.0,
    )

    initial_phase_raw = np.zeros_like(
        phase_raw
    )

    initial_phase_shift = np.zeros_like(
        phase_shift
    )

    initial_seasonal_phase = (
        fixed_seasonal_phase.copy()
    )

    strength_deviation = (
        strength
        -
        initial_strength
    )

    baseline_deviation = (
        baseline
        -
        initial_baseline
    )

    mean_strength_by_layer = np.mean(
        strength,
        axis=0,
    )

    std_strength_by_layer = np.std(
        strength,
        axis=0,
        ddof=1,
    )

    mean_abs_strength_deviation_by_layer = (
        np.mean(
            np.abs(
                strength_deviation
            ),
            axis=0,
        )
    )

    mean_abs_baseline_deviation_by_layer = (
        np.mean(
            np.abs(
                baseline_deviation
            ),
            axis=0,
        )
    )

    mean_phase_shift_by_layer = (
        np.mean(
            phase_shift,
            axis=0,
        )
    )

    mean_abs_phase_shift_by_layer = (
        np.mean(
            np.abs(
                phase_shift
            ),
            axis=0,
        )
    )

    max_abs_phase_shift_by_layer = (
        np.max(
            np.abs(
                phase_shift
            ),
            axis=0,
        )
    )

    min_adjacent_phase_gap_by_layer = np.zeros(
        depth,
        dtype=np.float64,
    )

    max_adjacent_phase_gap_by_layer = np.zeros(
        depth,
        dtype=np.float64,
    )

    mean_adjacent_phase_gap_by_layer = np.zeros(
        depth,
        dtype=np.float64,
    )

    for ell in range(
        depth
    ):

        # The residual bound |Delta phi| < pi/12 guarantees
        # that seasonal ordering is preserved.  Therefore
        # differences in month order give the twelve cyclic gaps.
        phi_unwrapped = (
            learned_seasonal_phase_unwrapped[
                :,
                ell,
            ]
        )

        cyclic_phase = np.concatenate([
            phi_unwrapped,
            np.array(
                [
                    phi_unwrapped[0]
                    +
                    2.0
                    *
                    math.pi
                ],
                dtype=np.float64,
            ),
        ])

        gaps = np.diff(
            cyclic_phase
        )

        min_adjacent_phase_gap_by_layer[
            ell
        ] = np.min(
            gaps
        )

        max_adjacent_phase_gap_by_layer[
            ell
        ] = np.max(
            gaps
        )

        mean_adjacent_phase_gap_by_layer[
            ell
        ] = np.mean(
            gaps
        )

        if np.min(
            gaps
        ) <= 0.0:

            raise RuntimeError(
                "Learned seasonal phase ordering was violated."
            )

    np.savez(
        out_dir
        /
        "learned_encoding_parameters.npz",

        strength_raw=
            strength_raw,

        shift_raw=
            shift_raw,

        phase_raw=
            phase_raw,

        reupload_strength=
            strength,

        polar_shift=
            shift,

        polar_baseline=
            baseline,

        seasonal_phase_shift=
            phase_shift,

        fixed_seasonal_phase=
            fixed_seasonal_phase,

        learned_seasonal_phase_unwrapped=
            learned_seasonal_phase_unwrapped,

        learned_seasonal_phase_wrapped=
            learned_seasonal_phase_wrapped,

        initial_reupload_strength=
            initial_strength,

        initial_polar_shift=
            initial_shift,

        initial_polar_baseline=
            initial_baseline,

        initial_phase_raw=
            initial_phase_raw,

        initial_seasonal_phase_shift=
            initial_phase_shift,

        initial_seasonal_phase=
            initial_seasonal_phase,

        strength_deviation_from_fixed=
            strength_deviation,

        baseline_deviation_from_pi_over_2=
            baseline_deviation,

        mean_strength_by_layer=
            mean_strength_by_layer,

        std_strength_by_layer=
            std_strength_by_layer,

        mean_abs_strength_deviation_by_layer=
            mean_abs_strength_deviation_by_layer,

        mean_abs_baseline_deviation_by_layer=
            mean_abs_baseline_deviation_by_layer,

        mean_phase_shift_by_layer=
            mean_phase_shift_by_layer,

        mean_abs_phase_shift_by_layer=
            mean_abs_phase_shift_by_layer,

        max_abs_phase_shift_by_layer=
            max_abs_phase_shift_by_layer,

        min_adjacent_phase_gap_by_layer=
            min_adjacent_phase_gap_by_layer,

        max_adjacent_phase_gap_by_layer=
            max_adjacent_phase_gap_by_layer,

        mean_adjacent_phase_gap_by_layer=
            mean_adjacent_phase_gap_by_layer,
    )

    # Long-format learned parameter table.
    rows = []

    for m in range(12):

        for ell in range(
            depth
        ):

            rows.append({
                "seasonal_index":
                    m,

                "layer_zero_based":
                    ell,

                "layer":
                    ell + 1,

                "strength_raw":
                    strength_raw[
                        m,
                        ell,
                    ],

                "shift_raw":
                    shift_raw[
                        m,
                        ell,
                    ],

                "phase_raw":
                    phase_raw[
                        m,
                        ell,
                    ],

                "reupload_strength":
                    strength[
                        m,
                        ell,
                    ],

                "strength_deviation_from_fixed":
                    strength_deviation[
                        m,
                        ell,
                    ],

                "polar_shift":
                    shift[
                        m,
                        ell,
                    ],

                "polar_baseline":
                    baseline[
                        m,
                        ell,
                    ],

                "baseline_deviation_from_pi_over_2":
                    baseline_deviation[
                        m,
                        ell,
                    ],

                "seasonal_phase_shift":
                    phase_shift[
                        m,
                        ell,
                    ],

                "fixed_seasonal_phase":
                    fixed_seasonal_phase[
                        m,
                        ell,
                    ],

                "learned_seasonal_phase_unwrapped":
                    learned_seasonal_phase_unwrapped[
                        m,
                        ell,
                    ],

                "learned_seasonal_phase_wrapped":
                    learned_seasonal_phase_wrapped[
                        m,
                        ell,
                    ],
            })

    pd.DataFrame(
        rows
    ).to_csv(
        out_dir
        /
        "learned_encoding_parameters.csv",
        index=False,
    )

    layer_rows = []

    for ell in range(
        depth
    ):

        layer_rows.append({
            "layer_zero_based":
                ell,

            "layer":
                ell + 1,

            "mean_reupload_strength":
                float(
                    np.mean(
                        strength[
                            :,
                            ell,
                        ]
                    )
                ),

            "std_reupload_strength":
                float(
                    np.std(
                        strength[
                            :,
                            ell,
                        ],
                        ddof=1,
                    )
                ),

            "min_reupload_strength":
                float(
                    np.min(
                        strength[
                            :,
                            ell,
                        ]
                    )
                ),

            "max_reupload_strength":
                float(
                    np.max(
                        strength[
                            :,
                            ell,
                        ]
                    )
                ),

            "mean_abs_strength_deviation":
                float(
                    np.mean(
                        np.abs(
                            strength_deviation[
                                :,
                                ell,
                            ]
                        )
                    )
                ),

            "mean_polar_baseline":
                float(
                    np.mean(
                        baseline[
                            :,
                            ell,
                        ]
                    )
                ),

            "std_polar_baseline":
                float(
                    np.std(
                        baseline[
                            :,
                            ell,
                        ],
                        ddof=1,
                    )
                ),

            "mean_abs_baseline_deviation":
                float(
                    np.mean(
                        np.abs(
                            baseline_deviation[
                                :,
                                ell,
                            ]
                        )
                    )
                ),

            "mean_seasonal_phase_shift":
                float(
                    mean_phase_shift_by_layer[
                        ell
                    ]
                ),

            "mean_abs_seasonal_phase_shift":
                float(
                    mean_abs_phase_shift_by_layer[
                        ell
                    ]
                ),

            "max_abs_seasonal_phase_shift":
                float(
                    max_abs_phase_shift_by_layer[
                        ell
                    ]
                ),

            "min_adjacent_phase_gap":
                float(
                    min_adjacent_phase_gap_by_layer[
                        ell
                    ]
                ),

            "max_adjacent_phase_gap":
                float(
                    max_adjacent_phase_gap_by_layer[
                        ell
                    ]
                ),

            "mean_adjacent_phase_gap":
                float(
                    mean_adjacent_phase_gap_by_layer[
                        ell
                    ]
                ),
        })

    pd.DataFrame(
        layer_rows
    ).to_csv(
        out_dir
        /
        "learned_encoding_layer_summary.csv",
        index=False,
    )

    # Save exact encoding inputs used during inference/training.
    if WINDOW_REFERENCE:

        np.savez(
            out_dir
            /
            "learnable_seasonal_encoding_reference.npz",
            **WINDOW_REFERENCE,
        )

    # --------------------------------------------------------
    # Parameter counts
    # --------------------------------------------------------

    variational_count = int(
        model.q_weights.numel()
    )

    encoding_count = int(
        model.encoding_strength_raw.numel()
        +
        model.encoding_shift_raw.numel()
        +
        model.encoding_phase_raw.numel()
    )

    head_count = int(
        sum(
            p.numel()
            for p
            in model.classical_head.parameters()
        )
    )

    total_count = int(
        sum(
            p.numel()
            for p
            in model.parameters()
        )
    )

    expected_variational = (
        3
        *
        model.n_qubits
        *
        n_ansatz_layers
        *
        depth
    )

    expected_encoding = (
        36
        *
        depth
    )

    expected_head = (
        model.n_qubits
        +
        1
    )

    expected_total = (
        expected_variational
        +
        expected_encoding
        +
        expected_head
    )

    if variational_count != expected_variational:
        raise RuntimeError(
            "Variational parameter count mismatch: "
            f"{variational_count} != {expected_variational}"
        )

    if encoding_count != expected_encoding:
        raise RuntimeError(
            "Encoding parameter count mismatch: "
            f"{encoding_count} != {expected_encoding}"
        )

    if head_count != expected_head:
        raise RuntimeError(
            "Head parameter count mismatch: "
            f"{head_count} != {expected_head}"
        )

    if total_count != expected_total:
        raise RuntimeError(
            "Total parameter count mismatch: "
            f"{total_count} != {expected_total}"
        )

    # --------------------------------------------------------
    # Verify checkpoint contains encoding parameters
    # --------------------------------------------------------

    checkpoint_path = (
        out_dir
        /
        "best_model.pt"
    )

    checkpoint_contains_encoding = False

    if checkpoint_path.exists():

        try:
            ckpt = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            ckpt = torch.load(
                checkpoint_path,
                map_location="cpu",
            )

        state = ckpt.get(
            "model_state_dict",
            ckpt,
        )

        checkpoint_contains_encoding = (
            "encoding_strength_raw"
            in state
            and
            "encoding_shift_raw"
            in state
            and
            "encoding_phase_raw"
            in state
        )

    if not checkpoint_contains_encoding:
        raise RuntimeError(
            "Best checkpoint does not contain "
            "learned encoding parameters."
        )

    metadata = {
        "status":
            "PASS",

        "encoding":
            "learnable_seasonal_cdf_rz",

        "encoding_label":
            "Learnable polar-azimuth CDF seasonal encoding",

        "direct_parent_encoding":
            "learnable_seasonal_cdf",

        "phase_zero_exactly_matches_direct_parent":
            True,

        "fixed_parent_encoding":
            "seasonal_meridian",

        "fixed_parent_script":
            str(
                SEASONAL_SCRIPT
            ),

        "encoding_script":
            str(
                THIS_FILE
            ),

        "initialization_exactly_matches_fixed_seasonal":
            True,

        "initial_reupload_strength":
            1.0,

        "initial_polar_shift":
            0.0,

        "initial_polar_baseline":
            math.pi / 2.0,

        "initial_seasonal_phase_shift":
            0.0,

        "reupload_strength_formula":
            "1 + tanh(strength_raw[m,l])",

        "reupload_strength_range":
            [
                0.0,
                2.0,
            ],

        "polar_shift_formula":
            "(pi/2) * tanh(shift_raw[m,l])",

        "polar_shift_range":
            [
                -math.pi / 2.0,
                math.pi / 2.0,
            ],

        "theta_formula":
            "pi/2 + shift[m,l] + "
            "strength[m,l]*(theta_cdf-pi/2)",

        "seasonal_phase_formula":
            "2*pi*m/12 + (pi/12)*tanh(phase_raw[m,l])",

        "seasonal_phase_shift_formula":
            "(pi/12) * tanh(phase_raw[m,l])",

        "seasonal_phase_shift_range":
            [
                -math.pi / 12.0,
                math.pi / 12.0,
            ],

        "fixed_adjacent_seasonal_gap":
            math.pi / 6.0,

        "seasonal_order_preserved_by_parameterization":
            True,

        "gate_application_order":
            [
                "RY(theta)",
                "RZ(phi)",
            ],

        "unitary_matrix_order":
            "RZ(phi) RY(theta)",

        "seasonal_index_source":
            "absolute lag source time inherited "
            "from validated fixed seasonal wrapper",

        "encoding_parameter_shape":
            [
                12,
                depth,
            ],

        "variational_parameter_count":
            variational_count,

        "encoding_parameter_count":
            encoding_count,

        "classical_head_parameter_count":
            head_count,

        "total_parameter_count":
            total_count,

        "expected_total_formula":
            "42*A*L + 36*L + 15",

        "depth":
            depth,

        "n_ansatz_layers":
            n_ansatz_layers,

        "seed":
            seed,

        "checkpoint_contains_encoding_parameters":
            checkpoint_contains_encoding,

        "window_audit":
            WINDOW_AUDIT,

        "input_CDF_changed":
            False,

        "seasonal_phase_changed":
            True,

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
    }

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

    # --------------------------------------------------------
    # Extend config.json
    # --------------------------------------------------------

    config_path = (
        out_dir
        /
        "config.json"
    )

    if config_path.exists():

        with open(
            config_path
        ) as f:
            cfg = json.load(f)

        cfg.update(
            metadata
        )

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
    # Extend result_summary.csv
    # --------------------------------------------------------

    result_path = (
        out_dir
        /
        "result_summary.csv"
    )

    if result_path.exists():

        result = pd.read_csv(
            result_path
        )

        result[
            "encoding"
        ] = (
            "learnable_seasonal_cdf"
        )

        result[
            "encoding_parameter_count"
        ] = encoding_count

        result[
            "total_parameter_count"
        ] = total_count

        result[
            "mean_final_reupload_strength"
        ] = float(
            np.mean(
                strength
            )
        )

        result[
            "mean_abs_strength_deviation"
        ] = float(
            np.mean(
                np.abs(
                    strength_deviation
                )
            )
        )

        result[
            "mean_abs_baseline_deviation"
        ] = float(
            np.mean(
                np.abs(
                    baseline_deviation
                )
            )
        )

        result.to_csv(
            result_path,
            index=False,
        )

    print()
    print("=" * 100)
    print("LEARNABLE CDF SEASONAL ENCODING")
    print("=" * 100)

    print(
        "Depth:",
        depth,
    )

    print(
        "Seed:",
        seed,
    )

    print(
        "Variational parameters:",
        variational_count,
    )

    print(
        "Encoding parameters:",
        encoding_count,
    )

    print(
        "Classical-head parameters:",
        head_count,
    )

    print(
        "Total parameters:",
        total_count,
    )

    print(
        "Mean learned reupload strength:",
        float(
            np.mean(
                strength
            )
        ),
    )

    print(
        "Mean |strength - 1|:",
        float(
            np.mean(
                np.abs(
                    strength_deviation
                )
            )
        ),
    )

    print(
        "Mean |baseline - pi/2|:",
        float(
            np.mean(
                np.abs(
                    baseline_deviation
                )
            )
        ),
    )

    print(
        "Mean |seasonal phase shift|:",
        float(
            np.mean(
                np.abs(
                    phase_shift
                )
            )
        ),
    )

    print(
        "Max |seasonal phase shift|:",
        float(
            np.max(
                np.abs(
                    phase_shift
                )
            )
        ),
    )

    print(
        "Minimum adjacent seasonal phase gap:",
        float(
            np.min(
                min_adjacent_phase_gap_by_layer
            )
        ),
    )

    print(
        "Checkpoint includes encoding params:",
        checkpoint_contains_encoding,
    )

    print(
        "STATUS: PASS"
    )

    print("=" * 100)


# ============================================================
# Production execution
# ============================================================

if __name__ == "__main__":

    # Exact fixed-seasonal CDF/window construction,
    # but second channel represented as integer month index.
    base.make_windows = (
        make_windows_learnable_seasonal_cdf
    )

    # Replace only the quantum model.
    base.DirectWindowQNN = (
        LearnableSeasonalCDFRzDirectWindowQNN
    )

    # Exact original trainer.
    base.main()

    # The base trainer has restored the best validation state.
    save_learned_encoding()
