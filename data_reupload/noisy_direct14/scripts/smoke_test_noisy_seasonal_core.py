#!/usr/bin/env python

from pathlib import Path
import os
import copy
import importlib.util
import math
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml


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

TRAINER_PATH = (
    PROJECT_ROOT
    / "data_reupload/noisy_direct14/scripts/"
      "direct14_noisy_seasonal_two_pass_worker_pool_hpc.py"
)

CSV_PATH = Path(
    os.environ.get(
        "QML_DATA_PATH",
        PROJECT_ROOT
        / "data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv",
    )
).expanduser().resolve()


# =============================================================================
# Import seasonal trainer without treating it as a package
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "seasonal_noisy_trainer",
    TRAINER_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        "Could not construct import spec."
    )

trainer = importlib.util.module_from_spec(spec)

# Important for dataclasses / module-level objects.
sys.modules[spec.name] = trainer

spec.loader.exec_module(trainer)


# =============================================================================
# Basic configuration
# =============================================================================

torch.set_default_dtype(
    torch.float64
)

torch.set_num_threads(
    1
)

trainer.set_seed(
    42
)

DEPTH = 2
N_QUBITS = 14
N_ANSATZ_LAYERS = 1

READOUT_WIRES = tuple(
    range(N_QUBITS)
)

ENCODINGS = [
    "seasonal_meridian",
    "learnable_seasonal_cdf",
    "learnable_seasonal_cdf_rz",
]


# =============================================================================
# Reconstruct seasonal data
# =============================================================================

raw = pd.read_csv(
    CSV_PATH
)

y_original = (
    pd.to_numeric(
        raw.iloc[:, 4],
        errors="coerce",
    )
    .dropna()
    .to_numpy(
        dtype=np.float64
    )
)

if len(y_original) != 539:
    raise RuntimeError(
        f"Expected 539 observations, got {len(y_original)}"
    )

data = trainer.make_seasonal_windows(
    y_original=y_original,
    window_size=14,
    train_end_idx=434,
    train_window_end=350,
    scaler_data_min=0.0,
    scaler_data_max=350.0,
    scaler_feature_min=-1.0,
    scaler_feature_max=1.0,
)

expected_shapes = {
    "X_train": (350, 14, 2),
    "X_val": (70, 14, 2),
    "X_test": (105, 14, 2),
    "y_train": (350,),
    "y_val": (70,),
    "y_test": (105,),
}

for key, expected in expected_shapes.items():

    actual = tuple(
        data[key].shape
    )

    if actual != expected:
        raise RuntimeError(
            f"{key}: expected {expected}, got {actual}"
        )

if int(
    data["cdf_reference_size"]
) != 364:
    raise RuntimeError(
        "CDF reference must contain 364 observations."
    )

if int(
    data["cdf_reference_end"]
) != 364:
    raise RuntimeError(
        "CDF reference endpoint must be 364."
    )

months = data[
    "X_train"
][..., 1]

if (
    np.min(months) < 0
    or
    np.max(months) > 11
):
    raise RuntimeError(
        "Invalid seasonal month indices."
    )

if not np.allclose(
    months,
    np.round(months),
):
    raise RuntimeError(
        "Month indices are not integral."
    )


# =============================================================================
# Build analytic QNodes
#
# No finite shots and no hardware noise here.
# This is a cheap structural validation only.
# =============================================================================

def build_model(
    encoding,
):

    dev = qml.device(
        "default.qubit",
        wires=N_QUBITS,
        shots=None,
    )

    circuit = trainer.build_direct_window_qnode(
        device=dev,
        n_qubits=N_QUBITS,
        depth=DEPTH,
        n_ansatz_layers=N_ANSATZ_LAYERS,
        angle_factor=math.pi,
        readout_wires=READOUT_WIRES,
        shots=None,
        encoding=encoding,
    )

    model = trainer.DirectWindowQNNNoisy(
        quantum_circuit=circuit,
        quantum_executor=None,
        n_qubits=N_QUBITS,
        depth=DEPTH,
        n_ansatz_layers=N_ANSATZ_LAYERS,
        readout_wires=READOUT_WIRES,
        use_output_tanh=True,
        encoding=encoding,
        q_init_scale=0.01,
    ).double()

    return model


fixed = build_model(
    "seasonal_meridian"
)

polar = build_model(
    "learnable_seasonal_cdf"
)

polar_rz = build_model(
    "learnable_seasonal_cdf_rz"
)


# =============================================================================
# Parameter-count validation
# =============================================================================

expected_counts = {
    "seasonal_meridian": 84,
    "learnable_seasonal_cdf": 132,
    "learnable_seasonal_cdf_rz": 156,
}

models = {
    "seasonal_meridian": fixed,
    "learnable_seasonal_cdf": polar,
    "learnable_seasonal_cdf_rz": polar_rz,
}

for encoding, model in models.items():

    actual = int(
        model.quantum_parameter_count
    )

    expected = expected_counts[
        encoding
    ]

    if actual != expected:
        raise RuntimeError(
            f"{encoding}: expected {expected} packed "
            f"quantum parameters, got {actual}"
        )


# =============================================================================
# Force identical ansatz and identical classical head
#
# Learned encoder raws already initialize exactly to zero.
# =============================================================================

with torch.no_grad():

    polar.q_weights.copy_(
        fixed.q_weights
    )

    polar_rz.q_weights.copy_(
        fixed.q_weights
    )

polar.classical_head.load_state_dict(
    copy.deepcopy(
        fixed.classical_head.state_dict()
    )
)

polar_rz.classical_head.load_state_dict(
    copy.deepcopy(
        fixed.classical_head.state_dict()
    )
)


# Encoder raws must be exactly zero before training.
for name, model in [
    ("polar", polar),
    ("polar_rz", polar_rz),
]:

    if not torch.equal(
        model.encoding_strength_raw,
        torch.zeros_like(
            model.encoding_strength_raw
        ),
    ):
        raise RuntimeError(
            f"{name}: strength raw does not initialize to zero."
        )

    if not torch.equal(
        model.encoding_shift_raw,
        torch.zeros_like(
            model.encoding_shift_raw
        ),
    ):
        raise RuntimeError(
            f"{name}: shift raw does not initialize to zero."
        )

if not torch.equal(
    polar_rz.encoding_phase_raw,
    torch.zeros_like(
        polar_rz.encoding_phase_raw
    ),
):
    raise RuntimeError(
        "polar_rz: phase raw does not initialize to zero."
    )


# =============================================================================
# Exact nesting test
# =============================================================================

x_small = torch.tensor(
    data["X_train"][:3],
    dtype=torch.float64,
)

z_fixed = fixed.quantum_features(
    x_small
)

z_polar = polar.quantum_features(
    x_small
)

z_polar_rz = polar_rz.quantum_features(
    x_small
)

max_feature_diff_fixed_polar = float(
    torch.max(
        torch.abs(
            z_fixed
            -
            z_polar
        )
    ).item()
)

max_feature_diff_fixed_rz = float(
    torch.max(
        torch.abs(
            z_fixed
            -
            z_polar_rz
        )
    ).item()
)

if max_feature_diff_fixed_polar > 1e-10:
    raise RuntimeError(
        "Fixed vs polar nesting failed: "
        f"{max_feature_diff_fixed_polar:.3e}"
    )

if max_feature_diff_fixed_rz > 1e-10:
    raise RuntimeError(
        "Fixed vs polar+azimuth nesting failed: "
        f"{max_feature_diff_fixed_rz:.3e}"
    )


p_fixed = fixed.classical_forward_from_features(
    z_fixed
)

p_polar = polar.classical_forward_from_features(
    z_polar
)

p_polar_rz = polar_rz.classical_forward_from_features(
    z_polar_rz
)

max_prediction_diff_fixed_polar = float(
    torch.max(
        torch.abs(
            p_fixed
            -
            p_polar
        )
    ).item()
)

max_prediction_diff_fixed_rz = float(
    torch.max(
        torch.abs(
            p_fixed
            -
            p_polar_rz
        )
    ).item()
)

if max_prediction_diff_fixed_polar > 1e-10:
    raise RuntimeError(
        "Prediction nesting failed for polar model."
    )

if max_prediction_diff_fixed_rz > 1e-10:
    raise RuntimeError(
        "Prediction nesting failed for polar+azimuth model."
    )


# =============================================================================
# Pack -> set round-trip
# =============================================================================

for encoding, model in models.items():

    theta_before = (
        model
        .pack_quantum_params()
        .clone()
    )

    model.set_quantum_params(
        theta_before
    )

    theta_after = (
        model
        .pack_quantum_params()
    )

    max_roundtrip_error = float(
        torch.max(
            torch.abs(
                theta_before
                -
                theta_after
            )
        ).item()
    )

    if max_roundtrip_error > 0.0:
        raise RuntimeError(
            f"{encoding}: packed parameter round trip failed: "
            f"{max_roundtrip_error:.3e}"
        )


# =============================================================================
# One SPSA step for each learned seasonal model
# =============================================================================

x_step = torch.tensor(
    data["X_train"][:2],
    dtype=torch.float64,
)

y_step = torch.tensor(
    data["y_train"][:2],
    dtype=torch.float64,
)

loss_function = nn.MSELoss(
    reduction="mean"
)


def run_one_spsa_step(
    model,
    seed,
):

    config = trainer.SPSAConfig(
        maxiter=1,
        a=0.02,
        c=0.05,
        A=50.0,
        alpha=0.602,
        gamma=0.101,
        perturbation_seed=seed,
        quantum_grad_clip_norm=None,
        head_grad_clip_norm=1.0,
        wrap_quantum_angles=False,
    )

    state = trainer.TwoPassSPSAState(
        config
    )

    optimizer = torch.optim.Adam(
        model.classical_head.parameters(),
        lr=0.005,
    )

    calls_before = int(
        model.qnode_call_count
    )

    result = trainer.two_pass_spsa_adam_step(
        model=model,
        x_batch=x_step,
        y_batch=y_step,
        head_optimizer=optimizer,
        spsa_state=state,
        loss_function=loss_function,
    )

    calls_used = (
        int(model.qnode_call_count)
        -
        calls_before
    )

    expected_calls = (
        2
        *
        len(x_step)
    )

    if calls_used != expected_calls:
        raise RuntimeError(
            f"Expected {expected_calls} QNode calls, "
            f"got {calls_used}"
        )

    return result


# ---------------------------------------------------------------------
# Model 4 movement
# ---------------------------------------------------------------------

polar.qnode_call_count = 0

polar_strength_before = (
    polar.encoding_strength_raw
    .detach()
    .clone()
)

polar_shift_before = (
    polar.encoding_shift_raw
    .detach()
    .clone()
)

polar_step = run_one_spsa_step(
    polar,
    42,
)

polar_strength_move = float(
    torch.max(
        torch.abs(
            polar.encoding_strength_raw
            -
            polar_strength_before
        )
    ).item()
)

polar_shift_move = float(
    torch.max(
        torch.abs(
            polar.encoding_shift_raw
            -
            polar_shift_before
        )
    ).item()
)

if polar_strength_move <= 0.0:
    raise RuntimeError(
        "Polar strength raw parameters did not move."
    )

if polar_shift_move <= 0.0:
    raise RuntimeError(
        "Polar shift raw parameters did not move."
    )


# ---------------------------------------------------------------------
# Model 5 movement
# ---------------------------------------------------------------------

polar_rz.qnode_call_count = 0

rz_strength_before = (
    polar_rz.encoding_strength_raw
    .detach()
    .clone()
)

rz_shift_before = (
    polar_rz.encoding_shift_raw
    .detach()
    .clone()
)

rz_phase_before = (
    polar_rz.encoding_phase_raw
    .detach()
    .clone()
)

rz_step = run_one_spsa_step(
    polar_rz,
    43,
)

rz_strength_move = float(
    torch.max(
        torch.abs(
            polar_rz.encoding_strength_raw
            -
            rz_strength_before
        )
    ).item()
)

rz_shift_move = float(
    torch.max(
        torch.abs(
            polar_rz.encoding_shift_raw
            -
            rz_shift_before
        )
    ).item()
)

rz_phase_move = float(
    torch.max(
        torch.abs(
            polar_rz.encoding_phase_raw
            -
            rz_phase_before
        )
    ).item()
)

if rz_strength_move <= 0.0:
    raise RuntimeError(
        "Model 5 strength raw parameters did not move."
    )

if rz_shift_move <= 0.0:
    raise RuntimeError(
        "Model 5 shift raw parameters did not move."
    )

if rz_phase_move <= 0.0:
    raise RuntimeError(
        "Model 5 phase raw parameters did not move."
    )


# =============================================================================
# Verify physical bounds after SPSA
# =============================================================================

def validate_polar_bounds(
    model,
    include_phase,
):

    strength = (
        1.0
        +
        torch.tanh(
            model.encoding_strength_raw
        )
    )

    shift = (
        math.pi
        /
        2.0
        *
        torch.tanh(
            model.encoding_shift_raw
        )
    )

    if not torch.all(
        strength > 0.0
    ):
        raise RuntimeError(
            "Strength lower bound violated."
        )

    if not torch.all(
        strength < 2.0
    ):
        raise RuntimeError(
            "Strength upper bound violated."
        )

    if not torch.all(
        torch.abs(shift)
        <
        math.pi / 2.0
    ):
        raise RuntimeError(
            "Polar-shift bound violated."
        )

    if include_phase:

        phase_shift = (
            math.pi
            /
            12.0
            *
            torch.tanh(
                model.encoding_phase_raw
            )
        )

        if not torch.all(
            torch.abs(
                phase_shift
            )
            <
            math.pi / 12.0
        ):
            raise RuntimeError(
                "Azimuth-shift bound violated."
            )


validate_polar_bounds(
    polar,
    include_phase=False,
)

validate_polar_bounds(
    polar_rz,
    include_phase=True,
)


# =============================================================================
# Report
# =============================================================================

print("=" * 100)
print("NOISY SEASONAL CORE SMOKE TEST")
print("=" * 100)

print()
print("Dataset:")
print("  observations:", len(y_original))
print("  train:", data["X_train"].shape)
print("  val:  ", data["X_val"].shape)
print("  test: ", data["X_test"].shape)
print("  CDF reference:", data["cdf_reference_size"])

print()
print("Packed SPSA parameter counts:")
print(
    "  fixed:",
    fixed.quantum_parameter_count,
)
print(
    "  polar:",
    polar.quantum_parameter_count,
)
print(
    "  polar+azimuth:",
    polar_rz.quantum_parameter_count,
)

print()
print("Exact zero-raw nesting:")
print(
    "  max |fixed - polar| features:",
    f"{max_feature_diff_fixed_polar:.3e}",
)
print(
    "  max |fixed - polar+azimuth| features:",
    f"{max_feature_diff_fixed_rz:.3e}",
)
print(
    "  max |fixed - polar| predictions:",
    f"{max_prediction_diff_fixed_polar:.3e}",
)
print(
    "  max |fixed - polar+azimuth| predictions:",
    f"{max_prediction_diff_fixed_rz:.3e}",
)

print()
print("One-step SPSA movement:")
print(
    "  polar strength raw max move:",
    f"{polar_strength_move:.6e}",
)
print(
    "  polar shift raw max move:",
    f"{polar_shift_move:.6e}",
)
print(
    "  model5 strength raw max move:",
    f"{rz_strength_move:.6e}",
)
print(
    "  model5 shift raw max move:",
    f"{rz_shift_move:.6e}",
)
print(
    "  model5 phase raw max move:",
    f"{rz_phase_move:.6e}",
)

print()
print("Directional derivatives:")
print(
    "  polar:",
    f"{polar_step['directional_derivative']:.6e}",
)
print(
    "  polar+azimuth:",
    f"{rz_step['directional_derivative']:.6e}",
)

print()
print("Physical bounds after SPSA: PASS")
print("Two-pass QNode-call invariant: PASS")

print()
print("=" * 100)
print("NOISY SEASONAL CORE SMOKE STATUS: PASS")
print("=" * 100)
