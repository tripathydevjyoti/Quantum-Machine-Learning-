from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pennylane as qml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# BASIC UTILITIES
# ============================================================

TAUS = (1, 6, 12, 24)
PERIOD = 12.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def scale_fixed(y, data_min=0.0, data_max=350.0):
    y = np.asarray(y, dtype=np.float64)
    return 2.0 * (y - data_min) / (data_max - data_min) - 1.0


def inverse_scale_fixed(x, data_min=0.0, data_max=350.0):
    x = np.asarray(x, dtype=np.float64)
    return data_min + 0.5 * (x + 1.0) * (data_max - data_min)


def safe_pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if np.std(a) <= 0 or np.std(b) <= 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def summarize(x, prefix):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_q05": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_q95": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }

    return {
        f"{prefix}_n": int(len(x)),
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        f"{prefix}_q05": float(np.quantile(x, 0.05)),
        f"{prefix}_q25": float(np.quantile(x, 0.25)),
        f"{prefix}_q75": float(np.quantile(x, 0.75)),
        f"{prefix}_q95": float(np.quantile(x, 0.95)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
    }


# ============================================================
# DATA RECONSTRUCTION
# ============================================================

def reconstruct_training_data(config):
    csv_path = Path(config["csv_path"])
    value_col = int(config["value_col"])
    window = int(config["window_size"])

    df = pd.read_csv(csv_path)

    y_original = (
        pd.to_numeric(
            df.iloc[:, value_col],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=np.float64)
    )

    if len(y_original) != 539:
        raise RuntimeError(
            f"Expected 539 non-missing observations, found {len(y_original)}"
        )

    y_scaled = scale_fixed(
        y_original,
        float(config["scaler_data_min"]),
        float(config["scaler_data_max"]),
    )

    # Exact supervised-window construction used by the final experiment:
    #
    # target indices 14,...,538
    # first 350 = train
    # next 70 = validation
    # last 105 = test
    all_targets = np.arange(window, len(y_scaled), dtype=int)

    X_all = np.stack(
        [y_scaled[t - window:t] for t in all_targets],
        axis=0,
    )

    y_all = y_scaled[all_targets]

    train_window_end = int(config["train_window_end"])

    X_train = X_all[:train_window_end]
    y_train = y_all[:train_window_end]
    train_targets = all_targets[:train_window_end]

    if X_train.shape != (350, 14):
        raise RuntimeError(
            f"Expected X_train shape (350,14), got {X_train.shape}"
        )

    if not np.array_equal(
        train_targets,
        np.arange(14, 364, dtype=int),
    ):
        raise RuntimeError(
            "Unexpected reconstructed training target indices."
        )

    return {
        "y_original": y_original,
        "y_scaled": y_scaled,
        "X_train": X_train,
        "y_train": y_train,
        "train_targets": train_targets,
    }


# ============================================================
# BATCH STATEVECTOR SIMULATOR
#
# Wire convention:
#   wire 0 = leftmost / most-significant tensor axis,
# matching PennyLane default.qubit ordering.
# ============================================================

def ry_batch(theta: torch.Tensor) -> torch.Tensor:
    """
    theta shape: [B]
    return: [B,2,2] complex128
    """

    c = torch.cos(theta / 2.0)
    s = torch.sin(theta / 2.0)

    U = torch.zeros(
        (len(theta), 2, 2),
        dtype=torch.complex128,
    )

    U[:, 0, 0] = c
    U[:, 0, 1] = -s
    U[:, 1, 0] = s
    U[:, 1, 1] = c

    return U


def rot_matrix(phi, theta, omega):
    """
    Use PennyLane itself for the exact qml.Rot convention,
    so the diagnostic simulator cannot silently use a different
    Euler-angle ordering from the production circuit.
    """

    M = qml.Rot.compute_matrix(
        float(phi),
        float(theta),
        float(omega),
    )

    if hasattr(M, "detach"):
        M = M.detach().cpu().numpy()

    M = np.asarray(M, dtype=np.complex128)

    return torch.tensor(
        M,
        dtype=torch.complex128,
    )


def apply_single_qubit(
    state: torch.Tensor,
    U: torch.Tensor,
    wire: int,
    n_qubits: int,
) -> torch.Tensor:
    """
    state: [B, 2**N]

    U:
      [2,2]       fixed gate shared by all windows
      or
      [B,2,2]     data-dependent gate
    """

    B = state.shape[0]

    psi = state.reshape(
        [B] + [2] * n_qubits
    )

    # Move target wire to the final tensor axis.
    axes = (
        [0]
        +
        [q + 1 for q in range(n_qubits) if q != wire]
        +
        [wire + 1]
    )

    inverse_axes = np.argsort(axes).tolist()

    tmp = (
        psi.permute(*axes)
        .contiguous()
        .reshape(B, -1, 2)
    )

    # Row-vector representation:
    #
    # new_row = old_row U^T
    if U.ndim == 2:
        out = torch.matmul(
            tmp,
            U.transpose(0, 1),
        )

    elif U.ndim == 3:
        out = torch.bmm(
            tmp,
            U.transpose(1, 2),
        )

    else:
        raise ValueError(
            f"Unexpected gate tensor shape: {U.shape}"
        )

    out = out.reshape(
        [B] + [2] * n_qubits
    )

    out = (
        out.permute(*inverse_axes)
        .contiguous()
        .reshape(B, -1)
    )

    return out


def build_cnot_permutation(
    n_qubits: int,
    control: int,
    target: int,
):
    """
    CNOT is an involutive basis permutation.
    """

    dim = 2 ** n_qubits

    idx = torch.arange(
        dim,
        dtype=torch.long,
    )

    control_mask = 1 << (n_qubits - 1 - control)
    target_mask = 1 << (n_qubits - 1 - target)

    perm = idx.clone()

    active = (
        (idx & control_mask)
        != 0
    )

    perm[active] = (
        perm[active]
        ^ target_mask
    )

    return perm


def apply_cnot(
    state: torch.Tensor,
    permutation: torch.Tensor,
):
    return state[:, permutation]


def state_norm_error(state):
    norms = torch.sum(
        torch.abs(state) ** 2,
        dim=1,
    )

    return float(
        torch.max(
            torch.abs(
                norms - 1.0
            )
        ).item()
    )


# ============================================================
# LOCAL PAULI EXPECTATIONS
# ============================================================

def local_xyz(
    state: torch.Tensor,
    n_qubits: int,
):
    """
    Returns array:

        [B, N, 3]

    final dimension = X,Y,Z.
    """

    B = state.shape[0]

    output = torch.empty(
        (B, n_qubits, 3),
        dtype=torch.float64,
    )

    psi = state.reshape(
        [B] + [2] * n_qubits
    )

    for wire in range(n_qubits):

        axes = (
            [0]
            +
            [q + 1 for q in range(n_qubits) if q != wire]
            +
            [wire + 1]
        )

        tmp = (
            psi.permute(*axes)
            .contiguous()
            .reshape(B, -1, 2)
        )

        a = tmp[:, :, 0]
        b = tmp[:, :, 1]

        cross = torch.sum(
            torch.conj(a) * b,
            dim=1,
        )

        p0 = torch.sum(
            torch.abs(a) ** 2,
            dim=1,
        )

        p1 = torch.sum(
            torch.abs(b) ** 2,
            dim=1,
        )

        output[:, wire, 0] = (
            2.0 * torch.real(cross)
        )

        output[:, wire, 1] = (
            2.0 * torch.imag(cross)
        )

        output[:, wire, 2] = (
            p0 - p1
        )

    return output


# ============================================================
# PAIRWISE FIDELITY
# ============================================================

def fidelity_for_tau(
    state: torch.Tensor,
    tau: int,
):
    inner = torch.sum(
        torch.conj(state[:-tau])
        *
        state[tau:],
        dim=1,
    )

    F = torch.abs(inner) ** 2

    return (
        F.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


# ============================================================
# FOURIER DIAGNOSTICS
# ============================================================

def exact_harmonic_coeff(
    x,
    target_indices,
    harmonic_n,
    period=12.0,
):
    x = np.asarray(x)

    centered = (
        x
        -
        np.mean(
            x,
            axis=0,
            keepdims=True,
        )
    )

    phase = np.exp(
        -1j
        *
        2.0
        *
        np.pi
        *
        harmonic_n
        *
        np.asarray(
            target_indices,
            dtype=np.float64,
        )
        /
        period
    )

    reshape = (
        (len(phase),)
        +
        (1,) * (centered.ndim - 1)
    )

    coeff = np.mean(
        centered
        *
        phase.reshape(reshape),
        axis=0,
    )

    return coeff


def local_fourier_rows(
    obs,
    target_indices,
    stage_meta,
):
    """
    obs:
        [n_stages, B, N, 3]

    Returns:
      harmonic rows
      aggregate stage rows
    """

    harmonic_rows = []
    aggregate_rows = []

    for s_idx, meta in enumerate(stage_meta):

        x = obs[s_idx]

        centered = (
            x
            -
            np.mean(
                x,
                axis=0,
                keepdims=True,
            )
        )

        variance_axis = np.mean(
            centered ** 2,
            axis=0,
        ).sum(axis=0)

        total_variance = float(
            np.sum(
                variance_axis
            )
        )

        powers_by_n = {}

        for n in range(1, 7):

            coeff = exact_harmonic_coeff(
                x,
                target_indices,
                n,
                PERIOD,
            )

            factor = (
                1.0
                if n == 6
                else 2.0
            )

            power_axis = (
                factor
                *
                np.sum(
                    np.abs(coeff) ** 2,
                    axis=0,
                )
            )

            power_total = float(
                np.sum(power_axis)
            )

            powers_by_n[n] = {
                "axis": power_axis,
                "total": power_total,
            }

            harmonic_rows.append(
                {
                    **meta,
                    "harmonic_n": n,
                    "period_steps": PERIOD / n,
                    "power_X": float(power_axis[0]),
                    "power_Y": float(power_axis[1]),
                    "power_Z": float(power_axis[2]),
                    "power_XYZ": power_total,
                    "fraction_XYZ": (
                        power_total / total_variance
                        if total_variance > 0
                        else np.nan
                    ),
                    "fraction_X": (
                        power_axis[0] / variance_axis[0]
                        if variance_axis[0] > 0
                        else np.nan
                    ),
                    "fraction_Y": (
                        power_axis[1] / variance_axis[1]
                        if variance_axis[1] > 0
                        else np.nan
                    ),
                    "fraction_Z": (
                        power_axis[2] / variance_axis[2]
                        if variance_axis[2] > 0
                        else np.nan
                    ),
                }
            )

        p1 = powers_by_n[1]["total"]

        p26 = float(
            sum(
                powers_by_n[n]["total"]
                for n in range(2, 7)
            )
        )

        p1_axis = powers_by_n[1]["axis"]

        if p1 > 0:
            comp = (
                p1_axis / p1
            )
        else:
            comp = np.full(3, np.nan)

        aggregate_rows.append(
            {
                **meta,
                "variance_X": float(variance_axis[0]),
                "variance_Y": float(variance_axis[1]),
                "variance_Z": float(variance_axis[2]),
                "variance_XYZ": total_variance,
                "seasonal_power_XYZ": p1,
                "higher_power_XYZ_n2_n6": p26,
                "seasonal_fraction_XYZ": (
                    p1 / total_variance
                    if total_variance > 0
                    else np.nan
                ),
                "higher_fraction_XYZ_n2_n6": (
                    p26 / total_variance
                    if total_variance > 0
                    else np.nan
                ),
                "seasonal_to_higher_ratio_XYZ": (
                    p1 / p26
                    if p26 > 0
                    else np.nan
                ),
                "seasonal_axis_comp_X": float(comp[0]),
                "seasonal_axis_comp_Y": float(comp[1]),
                "seasonal_axis_comp_Z": float(comp[2]),
                "seasonal_power_Z": float(p1_axis[2]),
                "seasonal_fraction_Z": (
                    float(p1_axis[2] / variance_axis[2])
                    if variance_axis[2] > 0
                    else np.nan
                ),
            }
        )

    return (
        pd.DataFrame(harmonic_rows),
        pd.DataFrame(aggregate_rows),
    )


def scalar_spectral_metrics(
    signal,
    target_indices,
):
    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    centered = (
        signal
        -
        np.mean(signal)
    )

    variance = float(
        np.mean(
            centered ** 2
        )
    )

    powers = {}

    for n in range(1, 7):

        coeff = exact_harmonic_coeff(
            signal,
            target_indices,
            n,
            PERIOD,
        )

        factor = (
            1.0
            if n == 6
            else 2.0
        )

        power = float(
            factor
            *
            np.abs(coeff) ** 2
        )

        powers[n] = power

    seasonal = powers[1]

    higher = float(
        sum(
            powers[n]
            for n in range(2, 7)
        )
    )

    return {
        "variance": variance,
        "std_sample": float(
            np.std(
                signal,
                ddof=1,
            )
        ),
        "seasonal_power": seasonal,
        "higher_power_n2_n6": higher,
        "seasonal_fraction": (
            seasonal / variance
            if variance > 0
            else np.nan
        ),
        "higher_fraction_n2_n6": (
            higher / variance
            if variance > 0
            else np.nan
        ),
        "seasonal_to_higher_ratio": (
            seasonal / higher
            if higher > 0
            else np.nan
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project_root",
        required=True,
    )

    parser.add_argument(
        "--ansatz_layers",
        type=int,
        required=True,
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

    parser.add_argument(
        "--phase1_pair_csv",
        required=True,
    )

    parser.add_argument(
        "--output_root",
        required=True,
    )

    parser.add_argument(
        "--kappa_denominator_tol",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    torch.set_default_dtype(
        torch.float64
    )

    torch.set_num_threads(
        args.threads
    )

    project_root = Path(
        args.project_root
    )

    variant = (
        f"single_tanh_a{args.ansatz_layers}"
    )

    checkpoint_dir = (
        project_root
        /
        "data_reupload/results/final_model_sweep"
        /
        variant
        /
        f"depth_{args.depth}"
        /
        f"seed_{args.seed}"
    )

    out = (
        Path(args.output_root)
        /
        f"A{args.ansatz_layers}_L{args.depth}_seed{args.seed}"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 110)
    print("PHASE II TRAINED-STATE FLOW ANALYSIS")
    print("=" * 110)

    print(
        f"checkpoint_dir: {checkpoint_dir}"
    )

    print(
        f"output_dir:     {out}"
    )

    # ========================================================
    # REQUIRED INPUT FILES
    # ========================================================

    required = [
        checkpoint_dir / "config.json",
        checkpoint_dir / "best_model.pt",
        checkpoint_dir / "learned_parameters.npz",
        checkpoint_dir / "predictions.npz",
        checkpoint_dir / "history.csv",
        checkpoint_dir / "result_summary.csv",
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    # ========================================================
    # PROVENANCE MANIFEST
    # ========================================================

    provenance = []

    for p in sorted(
        checkpoint_dir.iterdir()
    ):
        if p.is_file():
            provenance.append(
                {
                    "file": p.name,
                    "size_bytes": int(
                        p.stat().st_size
                    ),
                    "sha256": sha256_file(p),
                }
            )

    with open(
        out / "checkpoint_provenance.json",
        "w",
    ) as f:
        json.dump(
            {
                "checkpoint_dir":
                    str(checkpoint_dir),

                "files":
                    provenance,
            },
            f,
            indent=2,
        )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    with open(
        checkpoint_dir / "config.json"
    ) as f:
        config = json.load(f)

    if int(config["depth"]) != args.depth:
        raise RuntimeError(
            "Depth mismatch between requested model and config."
        )

    if int(config["n_ansatz_layers"]) != args.ansatz_layers:
        raise RuntimeError(
            "Ansatz-layer mismatch."
        )

    if int(config["seed"]) != args.seed:
        raise RuntimeError(
            "Seed mismatch."
        )

    if not bool(config["use_output_tanh"]):
        raise RuntimeError(
            "Expected tanh-head production model."
        )

    if config["readout"] != "all":
        raise RuntimeError(
            "Expected all-qubit Z readout."
        )

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
            f"Expected 14 qubits, got {n_qubits}"
        )

    # ========================================================
    # DATA
    # ========================================================

    # Resolve CSV relative to project root.
    config = dict(config)
    config["csv_path"] = str(
        project_root
        /
        config["csv_path"]
    )

    data = reconstruct_training_data(
        config
    )

    X_train = data["X_train"]
    y_train = data["y_train"]
    target_indices = data[
        "train_targets"
    ]

    B = len(X_train)
    dim = 2 ** n_qubits
    haar_baseline = 1.0 / dim

    print(
        f"Training windows: {X_train.shape}"
    )

    print(
        f"Hilbert dimension: {dim}"
    )

    print(
        f"Haar fidelity baseline 1/d: "
        f"{haar_baseline:.12e}"
    )

    # ========================================================
    # SAVED PREDICTIONS VALIDATION
    # ========================================================

    saved_pred = np.load(
        checkpoint_dir
        /
        "predictions.npz"
    )

    saved_targets = (
        saved_pred[
            "train_target_indices"
        ]
        .astype(int)
    )

    if not np.array_equal(
        saved_targets,
        target_indices,
    ):
        raise RuntimeError(
            "Reconstructed train target indices do not match predictions.npz."
        )

    true_diff = float(
        np.max(
            np.abs(
                saved_pred[
                    "y_train_true_scaled"
                ]
                -
                y_train
            )
        )
    )

    if true_diff > 1e-12:
        raise RuntimeError(
            "Reconstructed training targets do not match saved targets. "
            f"max_abs_diff={true_diff}"
        )

    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        checkpoint_dir
        /
        "best_model.pt",
        map_location="cpu",
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    if "q_weights" not in state_dict:
        raise KeyError(
            f"q_weights missing. Keys: {list(state_dict.keys())}"
        )

    q_weights = (
        state_dict[
            "q_weights"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    expected_shape = (
        depth,
        A,
        n_qubits,
        3,
    )

    if q_weights.shape != expected_shape:
        raise RuntimeError(
            f"q_weights shape {q_weights.shape}, "
            f"expected {expected_shape}"
        )

    # Tanh-head production model:
    # Sequential(Linear, Tanh)
    weight_key = (
        "classical_head.0.weight"
    )

    bias_key = (
        "classical_head.0.bias"
    )

    if weight_key not in state_dict:
        raise KeyError(
            f"Missing {weight_key}. "
            f"State dict keys: {list(state_dict.keys())}"
        )

    head_w = (
        state_dict[
            weight_key
        ]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(np.float64)
    )

    head_b = float(
        state_dict[
            bias_key
        ]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)[0]
    )

    # ========================================================
    # CROSS-CHECK learned_parameters.npz
    # ========================================================

    learned = np.load(
        checkpoint_dir
        /
        "learned_parameters.npz"
    )

    param_validation = {}

    max_param_diff = 0.0

    for name, tensor in state_dict.items():

        npz_key = (
            name.replace(".", "_")
        )

        if npz_key in learned.files:

            a = (
                tensor.detach()
                .cpu()
                .numpy()
            )

            b = learned[npz_key]

            diff = float(
                np.max(
                    np.abs(a - b)
                )
            )

            param_validation[
                name
            ] = diff

            max_param_diff = max(
                max_param_diff,
                diff,
            )

    if max_param_diff > 1e-12:
        raise RuntimeError(
            "best_model.pt and learned_parameters.npz disagree: "
            f"max diff={max_param_diff}"
        )

    # ========================================================
    # PHASE-I HAAR / 2-DESIGN COMPARATOR
    # ========================================================

    phase1 = pd.read_csv(
        args.phase1_pair_csv
    )

    phase1 = phase1[
        phase1["encoding"]
        ==
        "original_ry"
    ].copy()

    phase1 = phase1[
        phase1["tau"].isin(TAUS)
    ].copy()

    beta_lookup = {
        (
            int(r.tau),
            int(r.target_t),
            int(r.target_s),
        ):
        float(r.beta)

        for r in phase1.itertuples(
            index=False
        )
    }

    # ========================================================
    # INITIAL BATCH STATE
    # ========================================================

    state = torch.zeros(
        (B, dim),
        dtype=torch.complex128,
    )

    state[:, 0] = 1.0 + 0.0j

    X_torch = torch.tensor(
        X_train,
        dtype=torch.float64,
    )

    # CNOT ring permutations.
    cnot_perms = []

    for q in range(
        n_qubits - 1
    ):
        cnot_perms.append(
            build_cnot_permutation(
                n_qubits,
                q,
                q + 1,
            )
        )

    cnot_perms.append(
        build_cnot_permutation(
            n_qubits,
            n_qubits - 1,
            0,
        )
    )

    # ========================================================
    # STORAGE
    # ========================================================

    stage_meta = []
    local_obs_list = []

    fidelity_long_rows = []
    fidelity_summary_rows = []

    invariance_rows = []

    kappa_long_rows = []
    kappa_summary_rows = []

    block_head_rows = []

    stage_counter = 0

    max_norm_error_global = 0.0
    max_ansatz_fidelity_error = 0.0
    max_first_encoding_phase1_error = 0.0

    previous_block_fidelity = None

    # ========================================================
    # INTERNAL RECORDING FUNCTION
    # ========================================================

    def record_stage(
        state_now,
        stage_type,
        reupload_layer,
        ansatz_layer,
        compare_encoding_fidelity=None,
    ):

        nonlocal stage_counter
        nonlocal max_norm_error_global
        nonlocal max_ansatz_fidelity_error

        stage_counter += 1

        stage_id = (
            f"S{stage_counter:02d}_"
            f"{stage_type}_"
            f"L{reupload_layer}"
        )

        if ansatz_layer is not None:
            stage_id += (
                f"_A{ansatz_layer}"
            )

        norm_err = state_norm_error(
            state_now
        )

        max_norm_error_global = max(
            max_norm_error_global,
            norm_err,
        )

        obs = (
            local_xyz(
                state_now,
                n_qubits,
            )
            .detach()
            .cpu()
            .numpy()
        )

        local_obs_list.append(
            obs
        )

        meta = {
            "stage_index":
                stage_counter,

            "stage_id":
                stage_id,

            "stage_type":
                stage_type,

            "reupload_layer":
                int(reupload_layer),

            "ansatz_layer":
                (
                    int(ansatz_layer)
                    if ansatz_layer
                    is not None
                    else np.nan
                ),

            "max_state_norm_error":
                norm_err,
        }

        stage_meta.append(
            meta
        )

        fids = {}

        for tau in TAUS:

            F = fidelity_for_tau(
                state_now,
                tau,
            )

            fids[tau] = F

            row = {
                **meta,
                "tau": tau,
            }

            row.update(
                summarize(
                    F,
                    "fidelity",
                )
            )

            fidelity_summary_rows.append(
                row
            )

            for i, value in enumerate(F):

                fidelity_long_rows.append(
                    {
                        **meta,
                        "tau": tau,
                        "pair_index": i,
                        "target_t":
                            int(
                                target_indices[i]
                            ),
                        "target_s":
                            int(
                                target_indices[
                                    i + tau
                                ]
                            ),
                        "fidelity":
                            float(value),
                        "contrast_from_haar":
                            float(
                                value
                                -
                                haar_baseline
                            ),
                    }
                )

        # Unitary invariance check:
        #
        # after a data-independent ansatz sublayer,
        # pairwise fidelities must equal those
        # immediately after the preceding encoding.
        if compare_encoding_fidelity is not None:

            for tau in TAUS:

                diff = np.abs(
                    fids[tau]
                    -
                    compare_encoding_fidelity[tau]
                )

                max_diff = float(
                    np.max(diff)
                )

                max_ansatz_fidelity_error = max(
                    max_ansatz_fidelity_error,
                    max_diff,
                )

                invariance_rows.append(
                    {
                        "stage_id":
                            stage_id,

                        "reupload_layer":
                            reupload_layer,

                        "ansatz_layer":
                            ansatz_layer,

                        "tau":
                            tau,

                        "max_abs_fidelity_difference":
                            max_diff,

                        "mean_abs_fidelity_difference":
                            float(
                                np.mean(diff)
                            ),
                    }
                )

        return (
            stage_id,
            obs,
            fids,
        )

    # ========================================================
    # PROPAGATE THE TRAINED CIRCUIT
    # ========================================================

    with torch.no_grad():

        for layer in range(depth):

            layer_number = (
                layer + 1
            )

            # ------------------------------------------------
            # DATA RE-UPLOAD
            # ------------------------------------------------

            for q in range(
                n_qubits
            ):

                theta = (
                    angle_factor
                    *
                    X_torch[:, q]
                )

                U = ry_batch(
                    theta
                )

                state = apply_single_qubit(
                    state,
                    U,
                    q,
                    n_qubits,
                )

            (
                enc_stage_id,
                enc_obs,
                enc_fids,
            ) = record_stage(
                state,
                stage_type="encoding",
                reupload_layer=layer_number,
                ansatz_layer=None,
            )

            # ------------------------------------------------
            # First upload must reproduce Phase-I product-state
            # fidelity exactly.
            # ------------------------------------------------

            if layer_number == 1:

                for tau in TAUS:

                    F = enc_fids[tau]

                    reference = []

                    for i in range(
                        len(F)
                    ):

                        key = (
                            tau,
                            int(
                                target_indices[i]
                            ),
                            int(
                                target_indices[
                                    i + tau
                                ]
                            ),
                        )

                        reference.append(
                            beta_lookup.get(
                                key,
                                np.nan,
                            )
                        )

                    # Phase-I F1 == S == beta approximately,
                    # but beta differs from S by O(1/d^2).
                    #
                    # For exact state-fidelity validation,
                    # use F1 from the Phase-I CSV instead.
                    phase1_tau = phase1[
                        phase1["tau"]
                        ==
                        tau
                    ].copy()

                    f1_lookup = {
                        (
                            int(r.target_t),
                            int(r.target_s),
                        ):
                        float(r.F1)

                        for r
                        in phase1_tau.itertuples(
                            index=False
                        )
                    }

                    F1_ref = np.array(
                        [
                            f1_lookup[
                                (
                                    int(
                                        target_indices[i]
                                    ),
                                    int(
                                        target_indices[
                                            i + tau
                                        ]
                                    ),
                                )
                            ]
                            for i
                            in range(len(F))
                        ],
                        dtype=np.float64,
                    )

                    err = float(
                        np.max(
                            np.abs(
                                F - F1_ref
                            )
                        )
                    )

                    max_first_encoding_phase1_error = max(
                        max_first_encoding_phase1_error,
                        err,
                    )

            # ------------------------------------------------
            # TRAINED RE-UPLOAD CONTRACTION
            #
            # For layer >= 2 compare fidelity immediately after
            # this encoding against fidelity after the previous
            # completed variational block.
            # ------------------------------------------------

            if previous_block_fidelity is not None:

                for tau in TAUS:

                    F_pre = (
                        previous_block_fidelity[
                            tau
                        ]
                    )

                    F_post = (
                        enc_fids[tau]
                    )

                    delta_pre = (
                        F_pre
                        -
                        haar_baseline
                    )

                    delta_post = (
                        F_post
                        -
                        haar_baseline
                    )

                    valid = (
                        np.abs(
                            delta_pre
                        )
                        >
                        args.kappa_denominator_tol
                    )

                    kappas = np.full(
                        len(F_pre),
                        np.nan,
                        dtype=np.float64,
                    )

                    kappas[valid] = (
                        delta_post[valid]
                        /
                        delta_pre[valid]
                    )

                    betas = np.empty(
                        len(F_pre),
                        dtype=np.float64,
                    )

                    for i in range(
                        len(F_pre)
                    ):

                        key = (
                            tau,
                            int(
                                target_indices[i]
                            ),
                            int(
                                target_indices[
                                    i + tau
                                ]
                            ),
                        )

                        betas[i] = (
                            beta_lookup[key]
                        )

                    for i in range(
                        len(F_pre)
                    ):

                        kappa_long_rows.append(
                            {
                                "reupload_layer":
                                    layer_number,

                                "tau":
                                    tau,

                                "pair_index":
                                    i,

                                "target_t":
                                    int(
                                        target_indices[i]
                                    ),

                                "target_s":
                                    int(
                                        target_indices[
                                            i + tau
                                        ]
                                    ),

                                "F_before_encoding":
                                    float(
                                        F_pre[i]
                                    ),

                                "F_after_encoding":
                                    float(
                                        F_post[i]
                                    ),

                                "deltaF_before":
                                    float(
                                        delta_pre[i]
                                    ),

                                "deltaF_after":
                                    float(
                                        delta_post[i]
                                    ),

                                "kappa_trained":
                                    (
                                        float(
                                            kappas[i]
                                        )
                                        if np.isfinite(
                                            kappas[i]
                                        )
                                        else np.nan
                                    ),

                                "beta_haar":
                                    float(
                                        betas[i]
                                    ),

                                "kappa_valid":
                                    bool(
                                        valid[i]
                                    ),
                            }
                        )

                    row = {
                        "reupload_layer":
                            layer_number,

                        "tau":
                            tau,

                        "n_pairs":
                            len(F_pre),

                        "valid_fraction":
                            float(
                                np.mean(
                                    valid
                                )
                            ),
                    }

                    row.update(
                        summarize(
                            kappas[valid],
                            "kappa_trained",
                        )
                    )

                    row.update(
                        summarize(
                            betas,
                            "beta_haar",
                        )
                    )

                    if np.any(valid):

                        row[
                            "mean_abs_kappa_minus_beta"
                        ] = float(
                            np.mean(
                                np.abs(
                                    kappas[valid]
                                    -
                                    betas[valid]
                                )
                            )
                        )

                        row[
                            "median_abs_kappa_minus_beta"
                        ] = float(
                            np.median(
                                np.abs(
                                    kappas[valid]
                                    -
                                    betas[valid]
                                )
                            )
                        )

                        if (
                            np.std(
                                kappas[valid]
                            )
                            > 0
                            and
                            np.std(
                                betas[valid]
                            )
                            > 0
                        ):

                            row[
                                "pearson_kappa_beta"
                            ] = float(
                                np.corrcoef(
                                    kappas[valid],
                                    betas[valid],
                                )[0, 1]
                            )

                        else:

                            row[
                                "pearson_kappa_beta"
                            ] = np.nan

                    else:

                        row[
                            "mean_abs_kappa_minus_beta"
                        ] = np.nan

                        row[
                            "median_abs_kappa_minus_beta"
                        ] = np.nan

                        row[
                            "pearson_kappa_beta"
                        ] = np.nan

                    kappa_summary_rows.append(
                        row
                    )

            # ------------------------------------------------
            # VARIATIONAL SUBLAYERS
            # ------------------------------------------------

            last_obs = None
            last_fids = None

            for ansatz_layer in range(A):

                # 14 trained qml.Rot gates
                for q in range(
                    n_qubits
                ):

                    phi, theta, omega = (
                        q_weights[
                            layer,
                            ansatz_layer,
                            q,
                        ]
                    )

                    U = rot_matrix(
                        phi,
                        theta,
                        omega,
                    )

                    state = apply_single_qubit(
                        state,
                        U,
                        q,
                        n_qubits,
                    )

                # Nearest-neighbor ring CNOT.
                for perm in cnot_perms:

                    state = apply_cnot(
                        state,
                        perm,
                    )

                (
                    var_stage_id,
                    var_obs,
                    var_fids,
                ) = record_stage(
                    state,
                    stage_type="ansatz",
                    reupload_layer=layer_number,
                    ansatz_layer=ansatz_layer + 1,
                    compare_encoding_fidelity=enc_fids,
                )

                last_obs = var_obs
                last_fids = var_fids

            previous_block_fidelity = (
                last_fids
            )

            # ------------------------------------------------
            # DIAGNOSTIC FINAL-HEAD PROJECTION AT THIS BLOCK
            # ------------------------------------------------

            Z = (
                last_obs[:, :, 2]
            )

            pre_tanh = (
                Z @ head_w
                +
                head_b
            )

            post_tanh = np.tanh(
                pre_tanh
            )

            y_std = float(
                np.std(
                    y_train,
                    ddof=1,
                )
            )

            for signal_name, signal in [
                (
                    "pre_tanh",
                    pre_tanh,
                ),
                (
                    "post_tanh",
                    post_tanh,
                ),
            ]:

                sm = scalar_spectral_metrics(
                    signal,
                    target_indices,
                )

                rmse_scaled = float(
                    np.sqrt(
                        np.mean(
                            (
                                signal
                                -
                                y_train
                            ) ** 2
                        )
                    )
                )

                signal_std = float(
                    np.std(
                        signal,
                        ddof=1,
                    )
                )

                block_head_rows.append(
                    {
                        "block":
                            layer_number,

                        "signal":
                            signal_name,

                        "rmse_scaled":
                            rmse_scaled,

                        "rmse_original":
                            175.0
                            *
                            rmse_scaled,

                        "pearson_with_train_target":
                            safe_pearson(
                                signal,
                                y_train,
                            ),

                        "std_sample":
                            signal_std,

                        "target_std_sample":
                            y_std,

                        "amplitude_ratio":
                            (
                                signal_std
                                /
                                y_std
                                if y_std > 0
                                else np.nan
                            ),

                        **sm,
                    }
                )

    # ========================================================
    # STACK / SAVE LOCAL OBSERVABLES
    # ========================================================

    obs_all = np.stack(
        local_obs_list,
        axis=0,
    )

    stage_meta_df = pd.DataFrame(
        stage_meta
    )

    stage_meta_df.to_csv(
        out
        /
        "stage_metadata.csv",
        index=False,
    )

    np.savez_compressed(
        out
        /
        "local_observables.npz",
        observables=obs_all,
        target_indices=target_indices,
        pauli_labels=np.array(
            ["X", "Y", "Z"]
        ),
        stage_ids=stage_meta_df[
            "stage_id"
        ].to_numpy(),
    )

    # ========================================================
    # FINAL PREDICTION RECONSTRUCTION
    # ========================================================

    final_Z = (
        obs_all[
            -1,
            :,
            :,
            2,
        ]
    )

    final_pre = (
        final_Z @ head_w
        +
        head_b
    )

    final_post = np.tanh(
        final_pre
    )

    saved_train_pred = (
        saved_pred[
            "y_train_pred_scaled"
        ]
    )

    prediction_max_abs_diff = float(
        np.max(
            np.abs(
                final_post
                -
                saved_train_pred
            )
        )
    )

    prediction_rmse_diff = float(
        np.sqrt(
            np.mean(
                (
                    final_post
                    -
                    saved_train_pred
                ) ** 2
            )
        )
    )

    # Strict validation.
    if prediction_max_abs_diff > 1e-9:
        raise RuntimeError(
            "FAILED reconstruction validation: "
            f"max prediction difference "
            f"{prediction_max_abs_diff:.6e}"
        )

    if max_ansatz_fidelity_error > 1e-9:
        raise RuntimeError(
            "FAILED unitary-fidelity invariance check: "
            f"{max_ansatz_fidelity_error:.6e}"
        )

    if max_first_encoding_phase1_error > 1e-9:
        raise RuntimeError(
            "FAILED Phase-I first-encoding fidelity check: "
            f"{max_first_encoding_phase1_error:.6e}"
        )

    if max_norm_error_global > 1e-9:
        raise RuntimeError(
            "FAILED state normalization check: "
            f"{max_norm_error_global:.6e}"
        )

    # ========================================================
    # FOURIER-RESOLVED LOCAL INFORMATION
    # ========================================================

    (
        fourier_harmonic_df,
        fourier_aggregate_df,
    ) = local_fourier_rows(
        obs_all,
        target_indices,
        stage_meta,
    )

    fourier_harmonic_df.to_csv(
        out
        /
        "local_fourier_harmonics.csv",
        index=False,
    )

    fourier_aggregate_df.to_csv(
        out
        /
        "local_fourier_stage_summary.csv",
        index=False,
    )

    # ========================================================
    # SAVE PAIRWISE / KAPPA / HEAD TABLES
    # ========================================================

    fidelity_long_df = pd.DataFrame(
        fidelity_long_rows
    )

    fidelity_summary_df = pd.DataFrame(
        fidelity_summary_rows
    )

    invariance_df = pd.DataFrame(
        invariance_rows
    )

    kappa_long_df = pd.DataFrame(
        kappa_long_rows
    )

    kappa_summary_df = pd.DataFrame(
        kappa_summary_rows
    )

    head_df = pd.DataFrame(
        block_head_rows
    )

    fidelity_long_df.to_csv(
        out
        /
        "pairwise_fidelity_long.csv",
        index=False,
    )

    fidelity_summary_df.to_csv(
        out
        /
        "pairwise_fidelity_summary.csv",
        index=False,
    )

    invariance_df.to_csv(
        out
        /
        "ansatz_fidelity_invariance.csv",
        index=False,
    )

    kappa_long_df.to_csv(
        out
        /
        "trained_reupload_kappa_long.csv",
        index=False,
    )

    kappa_summary_df.to_csv(
        out
        /
        "trained_reupload_kappa_summary.csv",
        index=False,
    )

    head_df.to_csv(
        out
        /
        "block_head_diagnostics.csv",
        index=False,
    )

    # ========================================================
    # SNAPSHOT EXISTING RESULT SUMMARY
    # ========================================================

    existing_summary = pd.read_csv(
        checkpoint_dir
        /
        "result_summary.csv"
    )

    existing_summary.to_csv(
        out
        /
        "existing_result_summary_snapshot.csv",
        index=False,
    )

    # ========================================================
    # VALIDATION REPORT
    # ========================================================

    validation = {
        "variant":
            variant,

        "ansatz_layers":
            A,

        "depth":
            depth,

        "seed":
            args.seed,

        "best_epoch":
            int(
                checkpoint["epoch"]
            ),

        "n_training_windows":
            B,

        "hilbert_dimension":
            dim,

        "haar_fidelity_baseline":
            haar_baseline,

        "max_parameter_difference_best_model_vs_npz":
            max_param_diff,

        "max_true_target_difference_vs_predictions_npz":
            true_diff,

        "max_final_prediction_difference":
            prediction_max_abs_diff,

        "rmse_final_prediction_difference":
            prediction_rmse_diff,

        "max_state_norm_error":
            max_norm_error_global,

        "max_ansatz_fidelity_invariance_error":
            max_ansatz_fidelity_error,

        "max_first_encoding_fidelity_difference_vs_phase1":
            max_first_encoding_phase1_error,

        "status":
            "PASS",
    }

    with open(
        out
        /
        "validation.json",
        "w",
    ) as f:
        json.dump(
            validation,
            f,
            indent=2,
        )

    # ========================================================
    # RECONSTRUCTION CSV
    # ========================================================

    pd.DataFrame(
        {
            "target_index":
                target_indices,

            "true_scaled":
                y_train,

            "saved_pred_scaled":
                saved_train_pred,

            "reconstructed_pred_scaled":
                final_post,

            "difference":
                final_post
                -
                saved_train_pred,
        }
    ).to_csv(
        out
        /
        "final_prediction_reconstruction.csv",
        index=False,
    )

    # ========================================================
    # FIGURES
    # ========================================================

    # --------------------------------------------------------
    # 1. Mean fidelity across circuit stages.
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for tau in TAUS:

        g = (
            fidelity_summary_df[
                fidelity_summary_df[
                    "tau"
                ]
                ==
                tau
            ]
            .sort_values(
                "stage_index"
            )
        )

        ax.plot(
            g[
                "stage_index"
            ],
            g[
                "fidelity_mean"
            ],
            marker="o",
            label=f"tau={tau}",
        )

    ax.axhline(
        haar_baseline,
        linestyle="--",
        label="Haar 1/d",
    )

    ax.set_xlabel(
        "Circuit stage"
    )

    ax.set_ylabel(
        "Mean pairwise fidelity"
    )

    ax.set_title(
        f"Trained-state fidelity flow: "
        f"A={A}, L={depth}, seed={args.seed}"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out
        /
        "01_pairwise_fidelity_vs_stage.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 2. Local seasonal information.
    # --------------------------------------------------------

    fs = (
        fourier_aggregate_df
        .sort_values(
            "stage_index"
        )
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        fs[
            "stage_index"
        ],
        fs[
            "seasonal_fraction_XYZ"
        ],
        marker="o",
        label="local XYZ",
    )

    ax.plot(
        fs[
            "stage_index"
        ],
        fs[
            "seasonal_fraction_Z"
        ],
        marker="o",
        label="local Z",
    )

    ax.set_xlabel(
        "Circuit stage"
    )

    ax.set_ylabel(
        "Exact period-12 power / local variance"
    )

    ax.set_title(
        f"Accessibility of seasonal information: "
        f"A={A}, L={depth}"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out
        /
        "02_local_seasonal_information_vs_stage.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 3. Relative seasonal vs higher harmonics.
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        fs[
            "stage_index"
        ],
        fs[
            "seasonal_fraction_XYZ"
        ],
        marker="o",
        label="period 12",
    )

    ax.plot(
        fs[
            "stage_index"
        ],
        fs[
            "higher_fraction_XYZ_n2_n6"
        ],
        marker="o",
        label="harmonics n=2..6",
    )

    ax.set_xlabel(
        "Circuit stage"
    )

    ax.set_ylabel(
        "Fraction of local XYZ variance"
    )

    ax.set_title(
        f"Local spectral redistribution: "
        f"A={A}, L={depth}"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out
        /
        "03_local_seasonal_vs_higher_harmonics.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 4. Final-head accessibility after each block.
    # --------------------------------------------------------

    post = head_df[
        head_df[
            "signal"
        ]
        ==
        "post_tanh"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        post[
            "block"
        ],
        post[
            "amplitude_ratio"
        ],
        marker="o",
        label="amplitude ratio",
    )

    ax.plot(
        post[
            "block"
        ],
        post[
            "seasonal_fraction"
        ],
        marker="o",
        label="period-12 fraction",
    )

    ax.set_xlabel(
        "Completed reupload block"
    )

    ax.set_title(
        f"Diagnostic final-head accessibility: "
        f"A={A}, L={depth}"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out
        /
        "04_head_accessibility_vs_block.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 5. Trained kappa vs Haar beta for tau=12.
    # --------------------------------------------------------

    if len(
        kappa_summary_df
    ) > 0:

        k12 = (
            kappa_summary_df[
                kappa_summary_df[
                    "tau"
                ]
                ==
                12
            ]
            .sort_values(
                "reupload_layer"
            )
        )

        if len(k12) > 0:

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.plot(
                k12[
                    "reupload_layer"
                ],
                k12[
                    "kappa_trained_median"
                ],
                marker="o",
                label="trained median kappa",
            )

            ax.plot(
                k12[
                    "reupload_layer"
                ],
                k12[
                    "beta_haar_median"
                ],
                marker="o",
                label="Haar median beta",
            )

            ax.set_xlabel(
                "Reupload layer"
            )

            ax.set_ylabel(
                "Seasonal-memory contraction"
            )

            ax.set_title(
                f"Trained vs 2-design seasonal contraction: "
                f"A={A}, L={depth}"
            )

            ax.legend()

            fig.tight_layout()

            fig.savefig(
                out
                /
                "05_kappa_vs_haar_tau12.png",
                dpi=220,
                bbox_inches="tight",
            )

            plt.close(fig)

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 110)
    print("VALIDATION")
    print("=" * 110)

    for k, v in validation.items():
        print(f"{k}: {v}")

    print()
    print("=" * 110)
    print("FINAL-BLOCK LOCAL FOURIER SUMMARY")
    print("=" * 110)

    final_fs = (
        fourier_aggregate_df
        .iloc[-1]
    )

    print(
        final_fs.to_string()
    )

    print()
    print("=" * 110)
    print("HEAD DIAGNOSTICS")
    print("=" * 110)

    print(
        head_df.to_string(
            index=False
        )
    )

    if len(
        kappa_summary_df
    ) > 0:

        print()
        print("=" * 110)
        print("TRAINED REUPLOAD CONTRACTION")
        print("=" * 110)

        print(
            kappa_summary_df[
                [
                    "reupload_layer",
                    "tau",
                    "valid_fraction",
                    "kappa_trained_mean",
                    "kappa_trained_median",
                    "beta_haar_mean",
                    "beta_haar_median",
                    "mean_abs_kappa_minus_beta",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print("=" * 110)
    print("OUTPUT FILES")
    print("=" * 110)

    for p in sorted(
        out.iterdir()
    ):
        print(p.name)

    print()
    print("STATUS: PASS")


if __name__ == "__main__":
    main()
