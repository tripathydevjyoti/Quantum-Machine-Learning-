#!/usr/bin/env python

from pathlib import Path
import os
import sys
import math

import numpy as np
import torch
import torch.nn.functional as F


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

SCRIPT_DIR = (
    PROJECT_ROOT
    / "data_reupload/scripts/final_sweep"
)

sys.path.insert(
    0,
    str(SCRIPT_DIR),
)

import train_final_single_qnn_seasonal_meridian as fixed
import train_final_single_qnn_learnable_seasonal_cdf as model4
import train_final_single_qnn_learnable_seasonal_cdf_rz as model5


OUTDIR = (
    PROJECT_ROOT
    / "data_reupload/analysis/four_encoding_master/"
    "output/validation/model5_rz_smoke"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


torch.set_default_dtype(
    torch.float64
)

torch.manual_seed(
    123456
)

np.random.seed(
    123456
)


# ============================================================
# Helpers
# ============================================================

def make_polar_input(
    batch_size=4,
):

    X = torch.zeros(
        (
            batch_size,
            14,
            2,
        ),
        dtype=torch.float64,
    )

    for b in range(
        batch_size
    ):

        # Stay away from exact poles just to avoid pathological
        # visual/gradient special cases.
        theta = torch.linspace(
            0.08,
            math.pi - 0.08,
            14,
            dtype=torch.float64,
        )

        # Give each batch member a different value pattern.
        theta = torch.roll(
            theta,
            shifts=b,
        )

        months = (
            torch.arange(
                14,
                dtype=torch.float64,
            )
            +
            b
        ) % 12

        X[
            b,
            :,
            0,
        ] = theta

        X[
            b,
            :,
            1,
        ] = months

    return X


def make_fixed_input(
    X_month,
):

    X = X_month.clone()

    months = torch.round(
        X_month[
            ...,
            1,
        ]
    )

    X[
        ...,
        1,
    ] = (
        2.0
        *
        math.pi
        *
        months
        /
        12.0
    )

    return X


def instantiate(
    cls,
    depth,
):

    return cls(
        n_qubits=14,
        depth=depth,
        n_ansatz_layers=1,
        readout="all",
        angle_factor=math.pi,
        qml_device="default.qubit",
        diff_method="backprop",
        use_output_tanh=True,
    )


def copy_common_state(
    source,
    target,
):

    source_state = (
        source.state_dict()
    )

    target_state = (
        target.state_dict()
    )

    copied = []

    for key in source_state:

        if key not in target_state:
            continue

        if (
            source_state[key].shape
            !=
            target_state[key].shape
        ):
            raise RuntimeError(
                f"Shape mismatch for {key}: "
                f"{source_state[key].shape} vs "
                f"{target_state[key].shape}"
            )

        target_state[
            key
        ] = (
            source_state[
                key
            ]
            .detach()
            .clone()
        )

        copied.append(
            key
        )

    target.load_state_dict(
        target_state
    )

    return copied


# ============================================================
# Configuration
# ============================================================

DEPTH = 2
BATCH = 4

X_model45 = make_polar_input(
    BATCH
)

X_fixed = make_fixed_input(
    X_model45
)


print("=" * 120)
print("MODEL 5: LEARNABLE POLAR + AZIMUTH SEASONAL CDF")
print("EXACT NESTING / GRADIENT SMOKE TEST")
print("=" * 120)

print()
print(
    f"Depth: {DEPTH}"
)

print(
    f"Batch size: {BATCH}"
)


# ============================================================
# TEST 1
#
# Model 5 with phase_raw == 0 must exactly reproduce Model 4,
# even with NONZERO polar strength/shift parameters.
# ============================================================

m4 = instantiate(
    model4.LearnableSeasonalCDFDirectWindowQNN,
    DEPTH,
)

m5 = instantiate(
    model5.LearnableSeasonalCDFRzDirectWindowQNN,
    DEPTH,
)


# Put model 4 at a nontrivial point in its trainable polar
# encoder parameter space. This proves nesting of the whole
# model-4 family, not just initialization.
with torch.no_grad():

    m4.encoding_strength_raw.copy_(
        torch.linspace(
            -0.45,
            0.45,
            12 * DEPTH,
            dtype=torch.float64,
        ).reshape(
            12,
            DEPTH,
        )
    )

    m4.encoding_shift_raw.copy_(
        torch.linspace(
            0.30,
            -0.30,
            12 * DEPTH,
            dtype=torch.float64,
        ).reshape(
            12,
            DEPTH,
        )
    )


copied = copy_common_state(
    m4,
    m5,
)

with torch.no_grad():

    m5.encoding_phase_raw.zero_()


with torch.no_grad():

    y4 = m4(
        X_model45
    )

    y5 = m5(
        X_model45
    )


max_model4_model5_error = float(
    torch.max(
        torch.abs(
            y4
            -
            y5
        )
    ).item()
)


# Also compare the raw 14-dimensional local-Z circuit outputs.
readout_errors = []

with torch.no_grad():

    for i in range(
        BATCH
    ):

        z4 = m4.circuit(
            X_model45[i],
            m4.q_weights,
        )

        z5 = m5.circuit(
            X_model45[i],
            m5.q_weights,
        )

        if isinstance(
            z4,
            (
                tuple,
                list,
            ),
        ):
            z4 = torch.stack(
                list(z4)
            )

        if isinstance(
            z5,
            (
                tuple,
                list,
            ),
        ):
            z5 = torch.stack(
                list(z5)
            )

        readout_errors.append(
            float(
                torch.max(
                    torch.abs(
                        z4
                        -
                        z5
                    )
                ).item()
            )
        )


max_readout_error = max(
    readout_errors
)


print()
print("-" * 120)
print("TEST 1 — MODEL 5 REDUCES EXACTLY TO MODEL 4 WHEN phase_raw = 0")
print("-" * 120)

print(
    "Common state keys copied:",
    copied,
)

print(
    f"max 14-Z-readout difference: "
    f"{max_readout_error:.3e}"
)

print(
    f"max final prediction difference: "
    f"{max_model4_model5_error:.3e}"
)


if (
    max_readout_error
    >
    1e-12
):

    raise RuntimeError(
        "Model-4 / Model-5 circuit reduction failed."
    )


if (
    max_model4_model5_error
    >
    1e-12
):

    raise RuntimeError(
        "Model-4 / Model-5 prediction reduction failed."
    )


# ============================================================
# TEST 2
#
# With strength_raw=shift_raw=phase_raw=0, Model 5 must
# reduce exactly to the fixed seasonal CDF model.
# ============================================================

mf = instantiate(
    fixed.SeasonalMeridianDirectWindowQNN,
    DEPTH,
)

m5_parent = instantiate(
    model5.LearnableSeasonalCDFRzDirectWindowQNN,
    DEPTH,
)


# Copy variational circuit + head from fixed model.
copy_common_state(
    mf,
    m5_parent,
)


with torch.no_grad():

    m5_parent.encoding_strength_raw.zero_()
    m5_parent.encoding_shift_raw.zero_()
    m5_parent.encoding_phase_raw.zero_()


with torch.no_grad():

    yf = mf(
        X_fixed
    )

    y5_parent = m5_parent(
        X_model45
    )


max_fixed_model5_error = float(
    torch.max(
        torch.abs(
            yf
            -
            y5_parent
        )
    ).item()
)


print()
print("-" * 120)
print("TEST 2 — MODEL 5 REDUCES TO FIXED SEASONAL AT a=c=d=0")
print("-" * 120)

print(
    f"max final prediction difference: "
    f"{max_fixed_model5_error:.3e}"
)


if (
    max_fixed_model5_error
    >
    1e-12
):

    raise RuntimeError(
        "Fixed-seasonal parent reduction failed."
    )


# ============================================================
# TEST 3
#
# Parameter count.
# ============================================================

n_total = sum(
    p.numel()
    for p in m5.parameters()
)

expected_total = (
    78 * DEPTH
    +
    15
)

phase_count = (
    m5.encoding_phase_raw
    .numel()
)


print()
print("-" * 120)
print("TEST 3 — PARAMETER COUNT")
print("-" * 120)

print(
    f"encoding_phase_raw parameters: "
    f"{phase_count}"
)

print(
    f"total parameters: "
    f"{n_total}"
)

print(
    f"expected total: "
    f"{expected_total}"
)


if phase_count != (
    12 * DEPTH
):

    raise RuntimeError(
        "Unexpected phase parameter count."
    )


if n_total != expected_total:

    raise RuntimeError(
        "Unexpected total parameter count."
    )


# ============================================================
# TEST 4
#
# Check that phase parameters receive finite, nonzero gradient.
# ============================================================

mgrad = instantiate(
    model5.LearnableSeasonalCDFRzDirectWindowQNN,
    DEPTH,
)

mgrad.train()

targets = torch.tensor(
    [
        0.25,
        -0.15,
        0.35,
        -0.05,
    ],
    dtype=torch.float64,
)


pred = mgrad(
    X_model45
)

loss = F.mse_loss(
    pred,
    targets,
)

loss.backward()


phase_grad = (
    mgrad
    .encoding_phase_raw
    .grad
)


if phase_grad is None:

    raise RuntimeError(
        "encoding_phase_raw gradient is None."
    )


grad_finite = bool(
    torch.isfinite(
        phase_grad
    ).all().item()
)

grad_max_abs = float(
    torch.max(
        torch.abs(
            phase_grad
        )
    ).item()
)

grad_l2 = float(
    torch.linalg.vector_norm(
        phase_grad
    ).item()
)


print()
print("-" * 120)
print("TEST 4 — PHASE GRADIENT")
print("-" * 120)

print(
    f"loss: {float(loss.item()):.8e}"
)

print(
    f"phase gradient finite: "
    f"{grad_finite}"
)

print(
    f"phase gradient max abs: "
    f"{grad_max_abs:.8e}"
)

print(
    f"phase gradient L2: "
    f"{grad_l2:.8e}"
)


if not grad_finite:

    raise RuntimeError(
        "Non-finite phase gradient."
    )


if grad_max_abs <= 1e-12:

    raise RuntimeError(
        "Phase gradient is numerically zero."
    )


# ============================================================
# TEST 5
#
# One optimizer step must move phase_raw while transformed
# phase displacement remains strictly bounded by pi/12.
# ============================================================

phase_before = (
    mgrad
    .encoding_phase_raw
    .detach()
    .clone()
)


optimizer = torch.optim.Adam(
    mgrad.parameters(),
    lr=0.01,
)

optimizer.step()


phase_after = (
    mgrad
    .encoding_phase_raw
    .detach()
    .clone()
)


raw_move_max = float(
    torch.max(
        torch.abs(
            phase_after
            -
            phase_before
        )
    ).item()
)


phase_shift = (
    math.pi
    /
    12.0
    *
    torch.tanh(
        phase_after
    )
)


max_abs_phase_shift = float(
    torch.max(
        torch.abs(
            phase_shift
        )
    ).item()
)


print()
print("-" * 120)
print("TEST 5 — OPTIMIZER MOVEMENT + PHASE BOUND")
print("-" * 120)

print(
    f"max |phase_raw movement|: "
    f"{raw_move_max:.8e}"
)

print(
    f"max |Delta phi|: "
    f"{max_abs_phase_shift:.8e} rad"
)

print(
    f"pi/12 bound: "
    f"{math.pi/12.0:.8e} rad"
)


if raw_move_max <= 1e-12:

    raise RuntimeError(
        "Phase parameters did not move."
    )


if not (
    max_abs_phase_shift
    <
    math.pi
    /
    12.0
):

    raise RuntimeError(
        "Phase-shift bound violated."
    )


# ============================================================
# TEST 6
#
# Serialization / checkpoint state.
# ============================================================

checkpoint_path = (
    OUTDIR
    /
    "model5_smoke_checkpoint.pt"
)


torch.save(
    {
        "model_state_dict":
            mgrad.state_dict(),

        "depth":
            DEPTH,
    },
    checkpoint_path,
)


reload_ckpt = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)


keys = list(
    reload_ckpt[
        "model_state_dict"
    ].keys()
)


required_keys = [
    "encoding_strength_raw",
    "encoding_shift_raw",
    "encoding_phase_raw",
    "q_weights",
    "classical_head.0.weight",
    "classical_head.0.bias",
]


missing = [
    key
    for key in required_keys
    if key not in keys
]


print()
print("-" * 120)
print("TEST 6 — CHECKPOINT SERIALIZATION")
print("-" * 120)

print(
    "state_dict keys:"
)

for key in keys:
    print(
        f"  {key}"
    )


if missing:

    raise RuntimeError(
        f"Missing serialized parameters: {missing}"
    )


# ============================================================
# Save numeric smoke summary
# ============================================================

summary = {
    "depth":
        DEPTH,

    "model4_model5_max_readout_error":
        max_readout_error,

    "model4_model5_max_prediction_error":
        max_model4_model5_error,

    "fixed_model5_max_prediction_error":
        max_fixed_model5_error,

    "phase_parameter_count":
        phase_count,

    "total_parameter_count":
        n_total,

    "expected_total_parameter_count":
        expected_total,

    "phase_grad_max_abs":
        grad_max_abs,

    "phase_grad_l2":
        grad_l2,

    "phase_raw_move_max":
        raw_move_max,

    "max_abs_phase_shift_rad":
        max_abs_phase_shift,

    "phase_shift_bound_rad":
        math.pi / 12.0,

    "status":
        "PASS",
}


import json

with open(
    OUTDIR
    /
    "model5_smoke_summary.json",
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print("=" * 120)
print("MODEL 5 CORE / NESTING / GRADIENT SMOKE STATUS: PASS")
print("=" * 120)
