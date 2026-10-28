#!/usr/bin/env python
"""
Direct-window QNN training on a finite-shot noisy simulator using:

    * two-sided SPSA for quantum parameters, and
    * Adam for the classical readout head,

with exactly two quantum feature passes per training minibatch.

This is a self-contained HPC version of the original direct-window QNN
script. It preserves the original data preparation, model architecture,
metrics, predictions, and output files while replacing end-to-end Adam
with a hardware-compatible two-pass SPSA + Adam optimizer split.
"""

from __future__ import annotations

# Respect externally supplied HPC thread settings, but default to one thread
# to avoid nested parallelism and oversubscription in Aer/BLAS.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import copy
import inspect
import json
import math
import random
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

Tensor = torch.Tensor
LossFunction = Callable[[Tensor, Tensor], Tensor]


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def scale_fixed(y, data_min=0.0, data_max=350.0, feature_min=-1.0, feature_max=1.0):
    y = np.asarray(y, dtype=np.float64)
    return feature_min + (y - data_min) * (feature_max - feature_min) / (data_max - data_min)


def inverse_scale_fixed(z, data_min=0.0, data_max=350.0, feature_min=-1.0, feature_max=1.0):
    z = np.asarray(z, dtype=np.float64)
    return data_min + (z - feature_min) * (data_max - data_min) / (feature_max - feature_min)


def make_windows(y_original, window_size, train_end_idx, train_window_end,
                 scaler_data_min, scaler_data_max, scaler_feature_min, scaler_feature_max):
    y_original = np.asarray(y_original, dtype=np.float64)
    y_scaled = scale_fixed(
        y_original,
        data_min=scaler_data_min,
        data_max=scaler_data_max,
        feature_min=scaler_feature_min,
        feature_max=scaler_feature_max,
    )

    X = []
    y_t = []
    target_indices = []

    for i in range(len(y_scaled) - window_size):
        X.append(y_scaled[i:i + window_size])
        y_t.append(y_scaled[i + window_size])
        target_indices.append(i + window_size)

    X = np.asarray(X, dtype=np.float64)
    y_t = np.asarray(y_t, dtype=np.float64)
    target_indices = np.asarray(target_indices, dtype=int)

    n_train = int(train_window_end)
    n_val = int(train_end_idx - window_size - train_window_end)

    X_train = X[:n_train]
    y_train = y_t[:n_train]
    idx_train = target_indices[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y_t[n_train:n_train + n_val]
    idx_val = target_indices[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y_t[n_train + n_val:]
    idx_test = target_indices[n_train + n_val:]

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


def safe_corr(y, p):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def spearman_corr(y, p):
    y = pd.Series(np.asarray(y).reshape(-1)).rank(method="average").to_numpy()
    p = pd.Series(np.asarray(p).reshape(-1)).rank(method="average").to_numpy()
    return safe_corr(y, p)


def nse(y, p):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    denom = np.sum((y - np.mean(y)) ** 2)
    if denom == 0:
        return np.nan
    return float(1.0 - np.sum((y - p) ** 2) / denom)


def kge(y, p):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)

    r = safe_corr(y, p)
    alpha = np.std(p) / np.std(y) if np.std(y) > 0 else np.nan
    beta = np.mean(p) / np.mean(y) if np.mean(y) != 0 else np.nan

    if not np.isfinite(r) or not np.isfinite(alpha) or not np.isfinite(beta):
        return np.nan, r, alpha, beta

    val = 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return float(val), float(r), float(alpha), float(beta)


def deterministic_metrics(y_true_original, y_pred_original, scaler_data_min, scaler_data_max):
    y = np.asarray(y_true_original).reshape(-1)
    p = np.asarray(y_pred_original).reshape(-1)
    e = p - y
    ae = np.abs(e)

    kge_val, kge_r, kge_alpha, kge_beta = kge(y, p)

    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(ae)),
        "bias_mean_pred_minus_true": float(np.mean(e)),
        "median_error": float(np.median(e)),
        "pearson_corr": safe_corr(y, p),
        "spearman_corr": spearman_corr(y, p),
        "r2": nse(y, p),
        "nse": nse(y, p),
        "kge": kge_val,
        "kge_r": kge_r,
        "kge_alpha": kge_alpha,
        "kge_beta": kge_beta,
        "true_mean": float(np.mean(y)),
        "pred_mean": float(np.mean(p)),
        "true_std": float(np.std(y)),
        "pred_std": float(np.std(p)),
        "pred_std_over_true_std": float(np.std(p) / np.std(y)) if np.std(y) > 0 else np.nan,
        "true_min": float(np.min(y)),
        "pred_min": float(np.min(p)),
        "true_max": float(np.max(y)),
        "pred_max": float(np.max(p)),
        "negative_prediction_rate": float(np.mean(p < 0.0)),
        "out_of_range_rate": float(np.mean((p < scaler_data_min) | (p > scaler_data_max))),
        "max_abs_error": float(np.max(ae)),
        "p90_abs_error": float(np.quantile(ae, 0.90)),
        "p95_abs_error": float(np.quantile(ae, 0.95)),
        "underprediction_rate": float(np.mean(e < 0)),
        "overprediction_rate": float(np.mean(e > 0)),
    }


def fft_power(y):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y_centered = y - np.mean(y)
    coeff = np.fft.rfft(y_centered)
    power = np.abs(coeff) ** 2
    freqs = np.fft.rfftfreq(len(y_centered), d=1.0)

    # remove DC
    if len(power) > 0:
        power[0] = 0.0

    total = np.sum(power)
    share = power / total if total > 0 else np.zeros_like(power)

    periods = np.full_like(freqs, np.inf, dtype=np.float64)
    mask = freqs > 0
    periods[mask] = 1.0 / freqs[mask]

    return freqs, periods, power, share


def fft_summary(y_true, y_pred, seasonal_min=10.0, seasonal_max=14.0, short_min=2.0, short_max=6.0):
    f_t, per_t, pow_t, sh_t = fft_power(y_true)
    f_p, per_p, pow_p, sh_p = fft_power(y_pred)

    eps = 1e-12
    cos = float(np.dot(sh_t, sh_p) / (np.linalg.norm(sh_t) * np.linalg.norm(sh_p) + eps))
    l1 = float(np.sum(np.abs(sh_t - sh_p)))
    l2 = float(np.sqrt(np.sum((sh_t - sh_p) ** 2)))

    seasonal_mask = (per_t >= seasonal_min) & (per_t <= seasonal_max)
    short_mask = (per_t >= short_min) & (per_t <= short_max)

    true_12 = float(np.sum(sh_t[seasonal_mask]))
    pred_12 = float(np.sum(sh_p[seasonal_mask]))

    true_short = float(np.sum(sh_t[short_mask]))
    pred_short = float(np.sum(sh_p[short_mask]))

    nonzero = np.where(f_t > 0)[0]
    if len(nonzero) > 0:
        true_dom_i = nonzero[np.argmax(sh_t[nonzero])]
        pred_dom_i = nonzero[np.argmax(sh_p[nonzero])]
        true_dom_period = float(per_t[true_dom_i])
        pred_dom_period = float(per_p[pred_dom_i])
        true_dom_share = float(sh_t[true_dom_i])
        pred_dom_share = float(sh_p[pred_dom_i])
    else:
        true_dom_period = np.nan
        pred_dom_period = np.nan
        true_dom_share = np.nan
        pred_dom_share = np.nan

    return {
        "fft_spectral_cosine_similarity": cos,
        "fft_l1_distance": l1,
        "fft_l2_distance": l2,
        "true_12_step_band_power_share": true_12,
        "pred_12_step_band_power_share": pred_12,
        "abs_12_step_band_error": abs(true_12 - pred_12),
        "true_short_period_2_to_6_power_share": true_short,
        "pred_short_period_2_to_6_power_share": pred_short,
        "abs_short_period_2_to_6_error": abs(true_short - pred_short),
        "true_dominant_period": true_dom_period,
        "pred_dominant_period": pred_dom_period,
        "true_dominant_power_share": true_dom_share,
        "pred_dominant_power_share": pred_dom_share,
    }


def fft_power_rows(variant_id, model_family, depth, split, y_true, y_pred):
    rows = []
    for series_name, series in [("true", y_true), ("pred", y_pred)]:
        freqs, periods, power, share = fft_power(series)
        for f, per, pow_v, sh in zip(freqs, periods, power, share):
            rows.append({
                "variant_id": variant_id,
                "model_family": model_family,
                "depth": depth,
                "split": split,
                "series": series_name,
                "frequency": float(f),
                "period_steps": float(per) if np.isfinite(per) else np.inf,
                "power": float(pow_v),
                "power_share_non_dc": float(sh),
            })
    return rows


def peak_low_summaries(model_name, split, y_true, y_pred):
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(y_pred).reshape(-1)
    e = p - y

    peak_thr = np.quantile(y, 0.90)
    low_thr = np.quantile(y, 0.10)

    peak_mask = y >= peak_thr
    low_mask = y <= low_thr

    rows = []

    for group_name, mask in [("top_decile", peak_mask), ("bottom_decile", low_mask)]:
        yy = y[mask]
        pp = p[mask]
        ee = pp - yy
        ae = np.abs(ee)
        rows.append({
            "model": model_name,
            "split": split,
            "group": group_name,
            "threshold": float(peak_thr if group_name == "top_decile" else low_thr),
            "n": int(len(yy)),
            "rmse": float(np.sqrt(np.mean(ee ** 2))) if len(yy) else np.nan,
            "mae": float(np.mean(ae)) if len(yy) else np.nan,
            "bias": float(np.mean(ee)) if len(yy) else np.nan,
            "underprediction_rate": float(np.mean(ee < 0)) if len(yy) else np.nan,
            "overprediction_rate": float(np.mean(ee > 0)) if len(yy) else np.nan,
            "max_abs_error": float(np.max(ae)) if len(yy) else np.nan,
        })

    return rows


# ============================================================
# Backend and readout construction
# ============================================================


def parse_json_dict(text: str, argument_name: str) -> Dict[str, Any]:
    if text is None or text.strip() == "":
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{argument_name} must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{argument_name} must decode to a JSON object.")
    return value


def resolve_readout_wires(readout: str, n_qubits: int) -> Tuple[int, ...]:
    """
    Resolve readout configuration.

    Supported forms:
        all             -> every qubit
        first2          -> qubits 0 and 1
        0,7             -> explicit comma-separated wire list
        wires:0,7       -> explicit comma-separated wire list
    """
    value = str(readout).strip().lower()

    if value == "all":
        wires = tuple(range(n_qubits))
    elif value == "first2":
        if n_qubits < 2:
            raise ValueError("first2 readout requires at least two qubits.")
        wires = (0, 1)
    else:
        if value.startswith("wires:"):
            value = value.split(":", 1)[1]
        try:
            wires = tuple(int(token.strip()) for token in value.split(",") if token.strip())
        except ValueError as exc:
            raise ValueError(
                "readout must be 'all', 'first2', or a comma-separated wire list."
            ) from exc

    if not wires:
        raise ValueError("At least one readout wire is required.")
    if len(set(wires)) != len(wires):
        raise ValueError(f"Duplicate readout wires are not allowed: {wires}")
    if min(wires) < 0 or max(wires) >= n_qubits:
        raise ValueError(
            f"Readout wires {wires} must lie in [0, {n_qubits - 1}]."
        )

    return wires


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", backend.__class__.__name__)
    return str(name() if callable(name) else name)


def select_fake_backend(fake_provider: Any, n_qubits: int, class_name: Optional[str]):
    if class_name:
        if not hasattr(fake_provider, class_name):
            available = sorted(name for name in dir(fake_provider) if name.startswith("Fake"))
            raise ValueError(
                f"Fake backend class {class_name!r} was not found. "
                f"Examples available in this environment: {available[:20]}"
            )
        cls = getattr(fake_provider, class_name)
        backend = cls()
        backend_n = getattr(backend, "num_qubits", None)
        if backend_n is None or int(backend_n) < n_qubits:
            raise ValueError(
                f"{class_name} has {backend_n} qubits but {n_qubits} are required."
            )
        return class_name, backend, int(backend_n)

    candidates = []
    for name in dir(fake_provider):
        if not name.startswith("Fake"):
            continue
        obj = getattr(fake_provider, name)
        if not inspect.isclass(obj):
            continue
        try:
            backend = obj()
            backend_n = getattr(backend, "num_qubits", None)
            if backend_n is not None and int(backend_n) >= n_qubits:
                candidates.append((name, backend, int(backend_n)))
        except Exception:
            continue

    if not candidates:
        raise RuntimeError(
            f"No installed fake backend has at least {n_qubits} qubits."
        )

    candidates.sort(key=lambda item: (item[2], item[0]))
    return candidates[0]


def build_pennylane_device(args, n_qubits: int):
    """
    Build either:

    1. a Qiskit Aer simulator using only a fake backend's noise model
       (no coupling-map/topology restriction), or
    2. an arbitrary PennyLane device supplied by name and JSON kwargs.

    The second route keeps the script adaptable to an IonQ plugin later.
    """
    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError(
            "PennyLane is required. Install the HPC environment before running this script."
        ) from exc

    if args.backend_mode == "qiskit_aer_noisy":
        try:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel
            import qiskit_ibm_runtime.fake_provider as fake_provider
        except ImportError as exc:
            raise ImportError(
                "backend_mode=qiskit_aer_noisy requires qiskit-aer, "
                "pennylane-qiskit, and qiskit-ibm-runtime."
            ) from exc

        fake_class, fake_backend, fake_n_qubits = select_fake_backend(
            fake_provider=fake_provider,
            n_qubits=n_qubits,
            class_name=args.fake_backend_class,
        )

        noise_model = NoiseModel.from_backend(fake_backend)

        simulator_kwargs: Dict[str, Any] = {
            "noise_model": noise_model,
            "basis_gates": noise_model.basis_gates,
        }
        if args.seed_simulator >= 0:
            simulator_kwargs["seed_simulator"] = int(args.seed_simulator)
        if args.aer_method != "automatic":
            simulator_kwargs["method"] = args.aer_method

        aer_backend = AerSimulator(**simulator_kwargs)
        aer_backend.set_options(
            max_parallel_threads=args.aer_threads,
            max_parallel_experiments=args.aer_threads,
            max_parallel_shots=args.aer_threads,
        )

        # We deliberately pass only the noise-bearing Aer backend. No coupling
        # map from the fake hardware is imposed in this smoke-test workflow.
        dev = qml.device(
            "qiskit.aer",
            wires=n_qubits,
            backend=aer_backend,
            optimization_level=args.optimization_level,
            seed_transpiler=args.seed,
        )

        metadata = {
            "backend_mode": args.backend_mode,
            "pennylane_device": "qiskit.aer",
            "fake_backend_class": fake_class,
            "fake_backend_name": _backend_name(fake_backend),
            "fake_backend_num_qubits": fake_n_qubits,
            "noise_basis_gates": list(noise_model.basis_gates),
            "topology_constraints_used": False,
            "aer_method": args.aer_method,
            "aer_threads": args.aer_threads,
            "seed_simulator": None if args.seed_simulator < 0 else args.seed_simulator,
        }
        return dev, metadata

    if args.backend_mode == "pennylane_device":
        device_kwargs = parse_json_dict(args.device_kwargs_json, "--device_kwargs_json")
        dev = qml.device(
            args.qml_device,
            wires=n_qubits,
            **device_kwargs,
        )
        metadata = {
            "backend_mode": args.backend_mode,
            "pennylane_device": args.qml_device,
            "device_kwargs": device_kwargs,
            "topology_constraints_used": None,
        }
        return dev, metadata

    raise ValueError(f"Unsupported backend_mode: {args.backend_mode}")


def build_direct_window_qnode(
    *,
    device: Any,
    n_qubits: int,
    depth: int,
    n_ansatz_layers: int,
    angle_factor: float,
    readout_wires: Sequence[int],
    shots: Optional[int],
):
    """Construct the black-box finite-shot QNode used by SPSA."""
    import pennylane as qml

    def circuit_function(q_weights, x):
        for reupload_layer in range(depth):
            for q in range(n_qubits):
                qml.RY(angle_factor * x[q], wires=q)

            for ansatz_layer in range(n_ansatz_layers):
                for q in range(n_qubits):
                    qml.Rot(
                        q_weights[reupload_layer, ansatz_layer, q, 0],
                        q_weights[reupload_layer, ansatz_layer, q, 1],
                        q_weights[reupload_layer, ansatz_layer, q, 2],
                        wires=q,
                    )

                if n_qubits > 1:
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
                    qml.CNOT(wires=[n_qubits - 1, 0])

        return [qml.expval(qml.PauliZ(q)) for q in readout_wires]

    qnode = qml.QNode(
        circuit_function,
        device,
        interface=None,
        diff_method=None,
    )

    if shots is not None:
        qnode = qml.set_shots(shots)(qnode)

    return qnode


# ============================================================
# Two-pass SPSA + Adam
# ============================================================


@dataclass
class SPSAConfig:
    maxiter: Optional[int] = None
    alpha: float = 0.602
    gamma: float = 0.101
    c: float = 0.05
    A: Optional[float] = 50.0
    a: Optional[float] = 0.02
    perturbation_seed: Optional[int] = 42
    quantum_grad_clip_norm: Optional[float] = None
    head_grad_clip_norm: Optional[float] = 1.0
    wrap_quantum_angles: bool = False

    def resolved(self) -> "SPSAConfig":
        cfg = copy.deepcopy(self)
        if cfg.A is None:
            if cfg.maxiter is None:
                raise TypeError("One of maxiter or A must be provided.")
            cfg.A = 0.1 * float(cfg.maxiter)
        if cfg.a is None:
            cfg.a = 0.05 * (float(cfg.A) + 1.0) ** float(cfg.alpha)
        if cfg.c <= 0 or cfg.a <= 0:
            raise ValueError("SPSA a and c must be positive.")
        if cfg.A < 0 or cfg.alpha <= 0 or cfg.gamma <= 0:
            raise ValueError("Invalid SPSA schedule hyperparameters.")
        return cfg


class TwoPassSPSAState:
    """PennyLane-compatible SPSA schedules with k starting at one."""

    def __init__(self, config: SPSAConfig):
        self.config = config.resolved()
        self.k = 1
        self._rng = np.random.default_rng(self.config.perturbation_seed)

    def gains(self) -> Tuple[float, float]:
        a_k = float(self.config.a) / (
            (float(self.config.A) + self.k) ** float(self.config.alpha)
        )
        c_k = float(self.config.c) / (self.k ** float(self.config.gamma))
        return a_k, c_k

    def sample_delta(self, shape, dtype, device) -> Tensor:
        delta_np = self._rng.choice(
            np.array([-1.0, 1.0], dtype=np.float64),
            size=tuple(shape),
            replace=True,
        )
        return torch.tensor(delta_np, dtype=dtype, device=device)

    def advance(self):
        self.k += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "k": self.k,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.config = SPSAConfig(**state["config"]).resolved()
        self.k = int(state["k"])
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = copy.deepcopy(state["rng_state"])


class DirectWindowQNNNoisy(nn.Module):
    """
    Direct-window QNN with black-box quantum execution.

    Parameter names intentionally match the original classical-simulation
    implementation:
        q_weights
        classical_head.*

    Aliases q_params and classical let the generic two-pass optimizer operate
    without changing saved parameter names or downstream output conventions.
    """

    def __init__(
        self,
        *,
        quantum_circuit: Callable[[np.ndarray, np.ndarray], Any],
        n_qubits: int,
        depth: int,
        n_ansatz_layers: int,
        readout_wires: Sequence[int],
        use_output_tanh: bool,
        q_init_scale: float = 0.01,
    ):
        super().__init__()
        self.quantum_circuit = quantum_circuit
        self.n_qubits = int(n_qubits)
        self.depth = int(depth)
        self.n_ansatz_layers = int(n_ansatz_layers)
        self.readout_wires = tuple(int(q) for q in readout_wires)
        self.n_quantum_features = len(self.readout_wires)
        self.use_output_tanh = bool(use_output_tanh)

        self.q_weights = nn.Parameter(
            q_init_scale
            * torch.randn(
                self.depth,
                self.n_ansatz_layers,
                self.n_qubits,
                3,
                dtype=torch.float64,
            ),
            requires_grad=False,
        )

        if self.use_output_tanh:
            self.classical_head = nn.Sequential(
                nn.Linear(self.n_quantum_features, 1, dtype=torch.float64),
                nn.Tanh(),
            )
        else:
            self.classical_head = nn.Linear(
                self.n_quantum_features, 1, dtype=torch.float64
            )

        self.qnode_call_count = 0

    @property
    def q_params(self):
        return self.q_weights

    @property
    def classical(self):
        return self.classical_head

    @staticmethod
    def _to_numpy_safe(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return np.asarray(value.detach().cpu().tolist(), dtype=np.float64)
        return np.asarray(value, dtype=np.float64)

    @torch.no_grad()
    def quantum_features(self, x_batch: Tensor, q_params: Optional[Tensor] = None) -> Tensor:
        if x_batch.ndim != 2 or x_batch.shape[1] != self.n_qubits:
            raise ValueError(
                f"x_batch must have shape (batch_size, {self.n_qubits}); "
                f"received {tuple(x_batch.shape)}."
            )
        if x_batch.device.type != "cpu":
            raise ValueError("Black-box PennyLane execution currently requires CPU tensors.")

        q_params = self.q_weights if q_params is None else q_params
        q_params_np = self._to_numpy_safe(q_params)
        outputs = []

        for x in x_batch:
            q_out = self.quantum_circuit(q_params_np, self._to_numpy_safe(x))
            self.qnode_call_count += 1

            if isinstance(q_out, (tuple, list)):
                values = [float(v) for v in q_out]
            else:
                q_out_arr = np.asarray(q_out).reshape(-1)
                values = [float(v) for v in q_out_arr]

            if len(values) != self.n_quantum_features:
                raise ValueError(
                    f"Quantum circuit returned {len(values)} values; "
                    f"expected {self.n_quantum_features}."
                )

            outputs.append(torch.tensor(values, dtype=torch.float64))

        return torch.stack(outputs, dim=0)

    def classical_forward_from_features(self, features: Tensor) -> Tensor:
        return self.classical_head(features).squeeze(-1)

    def forward(self, x_batch: Tensor, q_params: Optional[Tensor] = None) -> Tensor:
        features = self.quantum_features(x_batch, q_params=q_params).detach()
        return self.classical_forward_from_features(features)


def _clip_tensor_by_norm(tensor: Tensor, max_norm: Optional[float], eps=1e-12):
    norm = float(torch.linalg.vector_norm(tensor).detach().cpu().item())
    if max_norm is None or norm <= max_norm:
        return tensor, norm
    return tensor * (float(max_norm) / (norm + eps)), norm


def _wrap_angles_to_pi(theta: Tensor) -> Tensor:
    return torch.remainder(theta + math.pi, 2.0 * math.pi) - math.pi


def two_pass_spsa_adam_step(
    *,
    model: DirectWindowQNNNoisy,
    x_batch: Tensor,
    y_batch: Tensor,
    head_optimizer: torch.optim.Optimizer,
    spsa_state: TwoPassSPSAState,
    loss_function: LossFunction,
) -> Dict[str, float]:
    """One SPSA quantum update and one Adam head update using two QNN passes."""
    model.train()
    a_k, c_k = spsa_state.gains()

    with torch.no_grad():
        theta_center = model.q_weights.detach().clone()
        delta = spsa_state.sample_delta(
            theta_center.shape,
            dtype=theta_center.dtype,
            device=theta_center.device,
        )
        theta_plus = theta_center + c_k * delta
        theta_minus = theta_center - c_k * delta

    calls_before = model.qnode_call_count
    z_plus = model.quantum_features(x_batch, q_params=theta_plus).detach()
    z_minus = model.quantum_features(x_batch, q_params=theta_minus).detach()
    qnode_calls = model.qnode_call_count - calls_before

    expected_calls = 2 * int(x_batch.shape[0])
    if qnode_calls != expected_calls:
        raise RuntimeError(
            f"Two-pass invariant failed: expected {expected_calls} QNode calls, "
            f"observed {qnode_calls}."
        )

    head_optimizer.zero_grad(set_to_none=True)
    pred_plus = model.classical_forward_from_features(z_plus)
    pred_minus = model.classical_forward_from_features(z_minus)
    loss_plus = loss_function(pred_plus, y_batch)
    loss_minus = loss_function(pred_minus, y_batch)

    if loss_plus.ndim != 0 or loss_minus.ndim != 0:
        raise ValueError("The loss function must return a scalar tensor.")
    if not torch.isfinite(loss_plus) or not torch.isfinite(loss_minus):
        raise FloatingPointError("A perturbed SPSA loss is non-finite.")

    # Adam sees the symmetric perturbation average. No third center QNN call.
    head_loss = 0.5 * (loss_plus + loss_minus)
    head_loss.backward()

    cfg = spsa_state.config
    head_grad_norm = float(
        torch.nn.utils.clip_grad_norm_(
            model.classical_head.parameters(),
            max_norm=(
                cfg.head_grad_clip_norm
                if cfg.head_grad_clip_norm is not None
                else float("inf")
            ),
        ).detach().cpu().item()
    )

    with torch.no_grad():
        directional_derivative = (loss_plus.detach() - loss_minus.detach()) / (2.0 * c_k)
        quantum_gradient = directional_derivative * delta
        quantum_gradient, quantum_grad_norm = _clip_tensor_by_norm(
            quantum_gradient, cfg.quantum_grad_clip_norm
        )
        theta_new = theta_center - a_k * quantum_gradient
        if cfg.wrap_quantum_angles:
            theta_new = _wrap_angles_to_pi(theta_new)
        if not torch.isfinite(theta_new).all():
            raise FloatingPointError("The SPSA update produced non-finite quantum parameters.")

    # Both gradients were computed at the same pre-update parameter blocks.
    head_optimizer.step()
    with torch.no_grad():
        model.q_weights.copy_(theta_new)

    completed_step = spsa_state.k
    spsa_state.advance()

    return {
        "spsa_iteration": completed_step,
        "a_k": float(a_k),
        "c_k": float(c_k),
        "loss_plus": float(loss_plus.detach().cpu().item()),
        "loss_minus": float(loss_minus.detach().cpu().item()),
        "head_loss": float(head_loss.detach().cpu().item()),
        "directional_derivative": float(directional_derivative.detach().cpu().item()),
        "quantum_grad_norm_before_clip": float(quantum_grad_norm),
        "head_grad_norm_before_clip": float(head_grad_norm),
        "qnode_calls": int(qnode_calls),
    }


# ============================================================
# Evaluation, checkpoints, and resume
# ============================================================


@torch.no_grad()
def evaluate_scaled(model, X, y, batch_size=32, repeats=1):
    """
    Center-point noisy evaluation.

    If repeats > 1, predictions are averaged across independent backend calls
    before MSE is computed. repeats=1 reproduces the original protocol.
    """
    if repeats <= 0:
        raise ValueError("Evaluation repeats must be positive.")

    model.eval()
    repeat_predictions = []

    for _ in range(repeats):
        preds = []
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size]
            pred = model(xb)
            preds.append(np.asarray(pred.detach().cpu().tolist(), dtype=np.float64))
        repeat_predictions.append(np.concatenate(preds))

    pred_all = np.mean(np.stack(repeat_predictions, axis=0), axis=0)
    y_np = np.asarray(y.detach().cpu().tolist(), dtype=np.float64)
    loss_mean = float(np.mean((pred_all - y_np) ** 2))
    return loss_mean, pred_all


def atomic_torch_save(payload: Dict[str, Any], path: Path):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def capture_rng_state() -> Dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }


def restore_rng_state(state: Dict[str, Any]):
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])


def make_training_checkpoint(
    *,
    epoch: int,
    model: DirectWindowQNNNoisy,
    head_optimizer: torch.optim.Optimizer,
    spsa_state: TwoPassSPSAState,
    best_val_loss: float,
    best_epoch: int,
    best_state: Optional[Dict[str, Any]],
    history: list,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "checkpoint_version": 1,
        "epoch": int(epoch),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "head_optimizer_state_dict": copy.deepcopy(head_optimizer.state_dict()),
        "spsa_state_dict": copy.deepcopy(spsa_state.state_dict()),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "best_state": copy.deepcopy(best_state),
        "history": copy.deepcopy(history),
        "config": copy.deepcopy(config),
        "rng_state": capture_rng_state(),
        "qnode_call_count": int(model.qnode_call_count),
    }


def load_training_checkpoint(
    *,
    path: Path,
    model: DirectWindowQNNNoisy,
    head_optimizer: torch.optim.Optimizer,
    spsa_state: TwoPassSPSAState,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    head_optimizer.load_state_dict(checkpoint["head_optimizer_state_dict"])
    spsa_state.load_state_dict(checkpoint["spsa_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    model.qnode_call_count = int(checkpoint.get("qnode_call_count", 0))
    return checkpoint


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()

    # Data and scaling: retained from the original direct-QNN script.
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--value_col", type=int, default=4)
    parser.add_argument("--train_end_idx", type=int, default=434)
    parser.add_argument("--window_size", type=int, default=14)
    parser.add_argument("--train_window_end", type=int, default=350)
    parser.add_argument("--scaler_data_min", type=float, default=0.0)
    parser.add_argument("--scaler_data_max", type=float, default=350.0)
    parser.add_argument("--scaler_feature_min", type=float, default=-1.0)
    parser.add_argument("--scaler_feature_max", type=float, default=1.0)

    # Circuit architecture.
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--n_ansatz_layers", type=int, default=1)
    parser.add_argument("--readout", type=str, default="all")
    parser.add_argument("--angle_factor", type=float, default=math.pi)
    parser.add_argument("--use_output_tanh", action="store_true")
    parser.add_argument("--q_init_scale", type=float, default=0.01)

    # Finite-shot backend.
    parser.add_argument(
        "--backend_mode",
        choices=["qiskit_aer_noisy", "pennylane_device"],
        default="qiskit_aer_noisy",
    )
    parser.add_argument("--qml_device", type=str, default="default.qubit")
    parser.add_argument("--device_kwargs_json", type=str, default="{}")
    parser.add_argument("--fake_backend_class", type=str, default=None)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--seed_simulator", type=int, default=42)
    parser.add_argument("--aer_method", type=str, default="automatic")
    parser.add_argument("--aer_threads", type=int, default=1)
    parser.add_argument("--optimization_level", type=int, default=1)

    # Hybrid optimizer. qnn_learning_rate is retained as the CLI name but now
    # applies only to the classical Adam head.
    parser.add_argument("--qnn_epochs", type=int, default=100)
    parser.add_argument("--qnn_learning_rate", type=float, default=0.005)
    parser.add_argument("--qnn_batch_size", type=int, default=16)
    parser.add_argument("--spsa_a", type=float, default=0.02)
    parser.add_argument("--spsa_c", type=float, default=0.05)
    parser.add_argument("--spsa_stability_constant", type=float, default=50.0)
    parser.add_argument("--spsa_alpha", type=float, default=0.602)
    parser.add_argument("--spsa_gamma", type=float, default=0.101)
    parser.add_argument("--spsa_seed", type=int, default=42)
    parser.add_argument("--head_grad_clip", type=float, default=1.0)
    parser.add_argument("--quantum_grad_clip", type=float, default=None)
    parser.add_argument("--wrap_quantum_angles", action="store_true")

    # Runtime, validation, and checkpointing.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--validate_every", type=int, default=1)
    parser.add_argument("--eval_repeats", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--max_train_batches_per_epoch", type=int, default=None)

    parser.add_argument("--variant_id", type=str, default="direct14_noisy_two_pass")
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    if args.device != "cpu":
        raise ValueError(
            "This black-box noisy-QNode implementation currently supports --device cpu only."
        )
    if args.shots <= 0:
        raise ValueError("Use a positive finite shot count for noisy SPSA training.")
    if args.validate_every <= 0 or args.checkpoint_every <= 0:
        raise ValueError("validate_every and checkpoint_every must be positive.")

    start_time = time.time()
    set_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.torch_threads)

    out_dir = (
        Path(args.output_dir)
        / args.variant_id
        / f"depth_{args.depth}"
        / f"seed_{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Direct-window noisy QNN: two-pass SPSA + Adam")
    print("=" * 100)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print(f"output_dir: {out_dir}")

    df = pd.read_csv(args.csv_path)
    y_original = (
        pd.to_numeric(df.iloc[:, args.value_col], errors="coerce")
        .dropna()
        .to_numpy(dtype=np.float64)
    )

    data = make_windows(
        y_original=y_original,
        window_size=args.window_size,
        train_end_idx=args.train_end_idx,
        train_window_end=args.train_window_end,
        scaler_data_min=args.scaler_data_min,
        scaler_data_max=args.scaler_data_max,
        scaler_feature_min=args.scaler_feature_min,
        scaler_feature_max=args.scaler_feature_max,
    )

    X_train = torch.tensor(data["X_train"], dtype=torch.float64)
    y_train = torch.tensor(data["y_train"], dtype=torch.float64)
    X_val = torch.tensor(data["X_val"], dtype=torch.float64)
    y_val = torch.tensor(data["y_val"], dtype=torch.float64)
    X_test = torch.tensor(data["X_test"], dtype=torch.float64)
    y_test = torch.tensor(data["y_test"], dtype=torch.float64)

    print("\nShapes:")
    print("X_train", tuple(X_train.shape), "X_val", tuple(X_val.shape), "X_test", tuple(X_test.shape))
    print("y_train", tuple(y_train.shape), "y_val", tuple(y_val.shape), "y_test", tuple(y_test.shape))

    n_qubits = args.window_size
    readout_wires = resolve_readout_wires(args.readout, n_qubits)
    device, backend_metadata = build_pennylane_device(args, n_qubits)
    quantum_circuit = build_direct_window_qnode(
        device=device,
        n_qubits=n_qubits,
        depth=args.depth,
        n_ansatz_layers=args.n_ansatz_layers,
        angle_factor=args.angle_factor,
        readout_wires=readout_wires,
        shots=args.shots,
    )

    model = DirectWindowQNNNoisy(
        quantum_circuit=quantum_circuit,
        n_qubits=n_qubits,
        depth=args.depth,
        n_ansatz_layers=args.n_ansatz_layers,
        readout_wires=readout_wires,
        use_output_tanh=args.use_output_tanh,
        q_init_scale=args.q_init_scale,
    ).double()

    total_expected_steps = args.qnn_epochs * math.ceil(
        len(X_train) / args.qnn_batch_size
    )
    spsa_config = SPSAConfig(
        maxiter=total_expected_steps,
        a=args.spsa_a,
        c=args.spsa_c,
        A=args.spsa_stability_constant,
        alpha=args.spsa_alpha,
        gamma=args.spsa_gamma,
        perturbation_seed=args.spsa_seed,
        quantum_grad_clip_norm=args.quantum_grad_clip,
        head_grad_clip_norm=args.head_grad_clip,
        wrap_quantum_angles=args.wrap_quantum_angles,
    )
    spsa_state = TwoPassSPSAState(spsa_config)
    head_optimizer = torch.optim.Adam(
        model.classical_head.parameters(), lr=args.qnn_learning_rate
    )
    mse = nn.MSELoss(reduction="mean")

    config = vars(args).copy()
    config.update({
        "model_family": "single_direct_qnn",
        "execution_mode": "finite_shot_noisy",
        "optimizer_scheme": "two_pass_spsa_quantum_adam_classical",
        "output_mode": "tanh_head" if args.use_output_tanh else "linear_head",
        "n_qubits": n_qubits,
        "readout_wires": list(readout_wires),
        "n_quantum_features": len(readout_wires),
        "q_param_shape": [args.depth, args.n_ansatz_layers, n_qubits, 3],
        "spsa_config": asdict(spsa_state.config),
        "backend_metadata": backend_metadata,
        "training_loss_definition": "0.5*(loss(theta+cDelta)+loss(theta-cDelta))",
        "validation_loss_definition": "center-point noisy MSE",
    })

    with open(out_dir / "config.json", "w") as file:
        json.dump(config, file, indent=2)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    history = []
    start_epoch = 1

    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        checkpoint = load_training_checkpoint(
            path=resume_path,
            model=model,
            head_optimizer=head_optimizer,
            spsa_state=spsa_state,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        best_state = checkpoint["best_state"]
        history = checkpoint["history"]
        print(f"\nResumed from {resume_path} at epoch {start_epoch}.")

    n_train = X_train.shape[0]
    last_checkpoint_path = out_dir / "last_training_checkpoint.pt"
    best_model_path = out_dir / "best_model.pt"

    for epoch in range(start_epoch, args.qnn_epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        train_loss_sum = 0.0
        processed_train_n = 0
        epoch_calls_before = model.qnode_call_count
        last_step_metrics = None

        for batch_number, start in enumerate(range(0, n_train, args.qnn_batch_size)):
            if (
                args.max_train_batches_per_epoch is not None
                and batch_number >= args.max_train_batches_per_epoch
            ):
                break

            idx = perm[start:start + args.qnn_batch_size]
            xb = X_train[idx]
            yb = y_train[idx]

            last_step_metrics = two_pass_spsa_adam_step(
                model=model,
                x_batch=xb,
                y_batch=yb,
                head_optimizer=head_optimizer,
                spsa_state=spsa_state,
                loss_function=mse,
            )

            train_loss_sum += last_step_metrics["head_loss"] * len(idx)
            processed_train_n += len(idx)

        if processed_train_n == 0 or last_step_metrics is None:
            raise RuntimeError("No training minibatches were processed.")

        train_loss_epoch = train_loss_sum / processed_train_n
        train_qnode_calls = model.qnode_call_count - epoch_calls_before

        should_validate = (
            epoch == 1
            or epoch % args.validate_every == 0
            or epoch == args.qnn_epochs
        )

        if should_validate:
            val_calls_before = model.qnode_call_count
            val_eval_loss, val_pred_scaled = evaluate_scaled(
                model,
                X_val,
                y_val,
                batch_size=args.eval_batch_size,
                repeats=args.eval_repeats,
            )
            val_qnode_calls = model.qnode_call_count - val_calls_before

            if val_eval_loss < best_val_loss:
                best_val_loss = val_eval_loss
                best_epoch = epoch
                best_state = {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "config": copy.deepcopy(config),
                    "spsa_step": spsa_state.k - 1,
                }
                atomic_torch_save(best_state, best_model_path)

            y_val_true_original = inverse_scale_fixed(
                data["y_val"],
                args.scaler_data_min,
                args.scaler_data_max,
                args.scaler_feature_min,
                args.scaler_feature_max,
            )
            val_pred_original = inverse_scale_fixed(
                val_pred_scaled,
                args.scaler_data_min,
                args.scaler_data_max,
                args.scaler_feature_min,
                args.scaler_feature_max,
            )
            val_rmse_original = float(
                np.sqrt(np.mean((val_pred_original - y_val_true_original) ** 2))
            )
        else:
            val_eval_loss = np.nan
            val_rmse_original = np.nan
            val_qnode_calls = 0

        history.append({
            "epoch": epoch,
            "train_loss_scaled": train_loss_epoch,
            "train_symmetric_perturbation_loss_scaled": train_loss_epoch,
            "val_loss_scaled": val_eval_loss,
            "val_center_loss_scaled": val_eval_loss,
            "train_rmse_original": np.nan,
            "val_rmse_original": val_rmse_original,
            "best_val_loss_scaled_so_far": best_val_loss,
            "best_epoch_so_far": best_epoch,
            "spsa_steps_completed": spsa_state.k - 1,
            "a_k_last": last_step_metrics["a_k"],
            "c_k_last": last_step_metrics["c_k"],
            "quantum_grad_norm_last": last_step_metrics["quantum_grad_norm_before_clip"],
            "head_grad_norm_last": last_step_metrics["head_grad_norm_before_clip"],
            "train_qnode_calls": train_qnode_calls,
            "val_qnode_calls": val_qnode_calls,
            "elapsed_time_sec": time.time() - start_time,
        })

        checkpoint_payload = make_training_checkpoint(
            epoch=epoch,
            model=model,
            head_optimizer=head_optimizer,
            spsa_state=spsa_state,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            best_state=best_state,
            history=history,
            config=config,
        )
        if epoch % args.checkpoint_every == 0 or epoch == args.qnn_epochs:
            atomic_torch_save(checkpoint_payload, last_checkpoint_path)

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.qnn_epochs:
            val_text = f"{val_eval_loss:.8f}" if np.isfinite(val_eval_loss) else "not evaluated"
            print(
                f"Epoch [{epoch}/{args.qnn_epochs}] "
                f"TrainSymLoss={train_loss_epoch:.8f} "
                f"ValCenterLoss={val_text} "
                f"BestVal={best_val_loss:.8f} "
                f"ValRMSE={val_rmse_original:.4f} "
                f"SPSASteps={spsa_state.k - 1} "
                f"TrainQCalls={train_qnode_calls} "
                f"a_k={last_step_metrics['a_k']:.8f} "
                f"c_k={last_step_metrics['c_k']:.8f}"
            )

    if best_state is None:
        raise RuntimeError("No best checkpoint was created. Validation must run at least once.")

    # Ensure best_model.pt exists even when resuming into a different output directory.
    atomic_torch_save(best_state, best_model_path)

    # Restore the best center-validation checkpoint before final evaluation.
    model.load_state_dict(best_state["model_state_dict"])

    calls_before_final_evaluation = model.qnode_call_count
    train_loss, train_pred_scaled = evaluate_scaled(
        model, X_train, y_train, batch_size=args.eval_batch_size, repeats=args.eval_repeats
    )
    val_loss, val_pred_scaled = evaluate_scaled(
        model, X_val, y_val, batch_size=args.eval_batch_size, repeats=args.eval_repeats
    )
    test_loss, test_pred_scaled = evaluate_scaled(
        model, X_test, y_test, batch_size=args.eval_batch_size, repeats=args.eval_repeats
    )
    final_eval_qnode_calls = model.qnode_call_count - calls_before_final_evaluation

    y_train_true_original = inverse_scale_fixed(data["y_train"], args.scaler_data_min, args.scaler_data_max, args.scaler_feature_min, args.scaler_feature_max)
    y_val_true_original = inverse_scale_fixed(data["y_val"], args.scaler_data_min, args.scaler_data_max, args.scaler_feature_min, args.scaler_feature_max)
    y_test_true_original = inverse_scale_fixed(data["y_test"], args.scaler_data_min, args.scaler_data_max, args.scaler_feature_min, args.scaler_feature_max)
    y_train_pred_original = inverse_scale_fixed(train_pred_scaled, args.scaler_data_min, args.scaler_data_max, args.scaler_feature_min, args.scaler_feature_max)
    y_val_pred_original = inverse_scale_fixed(val_pred_scaled, args.scaler_data_min, args.scaler_data_max, args.scaler_feature_min, args.scaler_feature_max)
    y_test_pred_original = inverse_scale_fixed(test_pred_scaled, args.scaler_data_min, args.scaler_data_max, args.scaler_feature_min, args.scaler_feature_max)

    config.update({
        "best_epoch": best_epoch,
        "best_val_loss_scaled": best_val_loss,
        "spsa_steps_completed": spsa_state.k - 1,
        "qnode_calls_before_final_evaluation": calls_before_final_evaluation,
        "final_evaluation_qnode_calls": final_eval_qnode_calls,
        "total_qnode_calls": model.qnode_call_count,
    })
    with open(out_dir / "config.json", "w") as file:
        json.dump(config, file, indent=2)

    history_df = pd.DataFrame(history)
    history_path = out_dir / "history.csv"
    history_df.to_csv(history_path, index=False)

    final_model_path = out_dir / "final_model.pt"
    atomic_torch_save({
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "config": config,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "spsa_state_dict_at_training_end": spsa_state.state_dict(),
    }, final_model_path)

    params_npz = {}
    for name, parameter in model.named_parameters():
        params_npz[name.replace(".", "_")] = np.asarray(
            parameter.detach().cpu().tolist(), dtype=np.float64
        )
    np.savez(out_dir / "learned_parameters.npz", **params_npz)

    predictions_npz_path = out_dir / "predictions.npz"
    np.savez(
        predictions_npz_path,
        y_train_true_scaled=data["y_train"],
        y_val_true_scaled=data["y_val"],
        y_test_true_scaled=data["y_test"],
        y_train_pred_scaled=train_pred_scaled,
        y_val_pred_scaled=val_pred_scaled,
        y_test_pred_scaled=test_pred_scaled,
        y_train_true_original=y_train_true_original,
        y_val_true_original=y_val_true_original,
        y_test_true_original=y_test_true_original,
        y_train_pred_original=y_train_pred_original,
        y_val_pred_original=y_val_pred_original,
        y_test_pred_original=y_test_pred_original,
        train_target_indices=data["idx_train"],
        val_target_indices=data["idx_val"],
        test_target_indices=data["idx_test"],
        scaler_data_min=np.array([args.scaler_data_min]),
        scaler_data_max=np.array([args.scaler_data_max]),
        scaler_feature_min=np.array([args.scaler_feature_min]),
        scaler_feature_max=np.array([args.scaler_feature_max]),
    )

    long_rows = []
    for split, idx, yt_s, yp_s, yt_o, yp_o in [
        ("train", data["idx_train"], data["y_train"], train_pred_scaled, y_train_true_original, y_train_pred_original),
        ("val", data["idx_val"], data["y_val"], val_pred_scaled, y_val_true_original, y_val_pred_original),
        ("test", data["idx_test"], data["y_test"], test_pred_scaled, y_test_true_original, y_test_pred_original),
    ]:
        for i in range(len(idx)):
            err = yp_o[i] - yt_o[i]
            long_rows.append({
                "variant_id": args.variant_id,
                "model_family": "single_direct_qnn",
                "execution_mode": "finite_shot_noisy",
                "optimizer_scheme": "two_pass_spsa_quantum_adam_classical",
                "depth": args.depth,
                "n_ansatz_layers": args.n_ansatz_layers,
                "split": split,
                "split_pos": i,
                "target_index": int(idx[i]),
                "true_scaled": float(yt_s[i]),
                "pred_scaled": float(yp_s[i]),
                "true_original": float(yt_o[i]),
                "pred_original": float(yp_o[i]),
                "error_original": float(err),
                "abs_error_original": float(abs(err)),
                "sq_error_original": float(err ** 2),
            })

    pred_long_df = pd.DataFrame(long_rows)
    pred_long_path = out_dir / "predictions_long.csv"
    pred_long_df.to_csv(pred_long_path, index=False)

    pred_wide_rows = []
    for split, idx, yt_o, yp_o in [
        ("train", data["idx_train"], y_train_true_original, y_train_pred_original),
        ("val", data["idx_val"], y_val_true_original, y_val_pred_original),
        ("test", data["idx_test"], y_test_true_original, y_test_pred_original),
    ]:
        for i in range(len(idx)):
            pred_wide_rows.append({
                "split": split,
                "split_pos": i,
                "target_index": int(idx[i]),
                "true_original": float(yt_o[i]),
                "pred_original": float(yp_o[i]),
                "error_original": float(yp_o[i] - yt_o[i]),
                "abs_error_original": float(abs(yp_o[i] - yt_o[i])),
            })

    pred_wide_path = out_dir / "predictions_wide.csv"
    pd.DataFrame(pred_wide_rows).to_csv(pred_wide_path, index=False)

    metric_rows = []
    fft_rows = []
    fft_power_all_rows = []
    peak_low_rows = []

    for split, yt_o, yp_o in [
        ("train", y_train_true_original, y_train_pred_original),
        ("val", y_val_true_original, y_val_pred_original),
        ("test", y_test_true_original, y_test_pred_original),
    ]:
        row = {
            "variant_id": args.variant_id,
            "model_family": "single_direct_qnn",
            "execution_mode": "finite_shot_noisy",
            "optimizer_scheme": "two_pass_spsa_quantum_adam_classical",
            "depth": args.depth,
            "n_ansatz_layers": args.n_ansatz_layers,
            "split": split,
            "best_epoch": best_epoch,
            "best_val_loss_scaled": best_val_loss,
            "use_output_tanh": args.use_output_tanh,
            "shots": args.shots,
            "scaler_data_min": args.scaler_data_min,
            "scaler_data_max": args.scaler_data_max,
            "scaler_feature_min": args.scaler_feature_min,
            "scaler_feature_max": args.scaler_feature_max,
        }
        row.update(deterministic_metrics(yt_o, yp_o, args.scaler_data_min, args.scaler_data_max))
        metric_rows.append(row)

        fft_row = {
            "variant_id": args.variant_id,
            "model_family": "single_direct_qnn",
            "depth": args.depth,
            "n_ansatz_layers": args.n_ansatz_layers,
            "split": split,
        }
        fft_row.update(fft_summary(yt_o, yp_o))
        fft_rows.append(fft_row)
        fft_power_all_rows.extend(
            fft_power_rows(args.variant_id, "single_direct_qnn", args.depth, split, yt_o, yp_o)
        )
        peak_low_rows.extend(peak_low_summaries(args.variant_id, split, yt_o, yp_o))

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "metrics_by_split.csv"
    metrics_df.to_csv(metrics_path, index=False)

    fft_summary_df = pd.DataFrame(fft_rows)
    fft_summary_path = out_dir / "fft_summary_by_split.csv"
    fft_summary_df.to_csv(fft_summary_path, index=False)

    fft_power_path = out_dir / "fft_power_by_period.csv"
    pd.DataFrame(fft_power_all_rows).to_csv(fft_power_path, index=False)

    peak_low_path = out_dir / "peak_low_error_summary.csv"
    pd.DataFrame(peak_low_rows).to_csv(peak_low_path, index=False)

    top_errors_path = out_dir / "top_errors.csv"
    (
        pred_long_df
        .sort_values(["split", "abs_error_original"], ascending=[True, False])
        .groupby("split")
        .head(20)
        .to_csv(top_errors_path, index=False)
    )

    elapsed = time.time() - start_time
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    test_metrics = metrics_df[metrics_df["split"] == "test"].iloc[0].to_dict()
    val_metrics = metrics_df[metrics_df["split"] == "val"].iloc[0].to_dict()
    train_metrics = metrics_df[metrics_df["split"] == "train"].iloc[0].to_dict()
    test_fft = fft_summary_df[fft_summary_df["split"] == "test"].iloc[0].to_dict()

    estimated_total_shots = int(model.qnode_call_count * args.shots)
    summary = {
        "variant_id": args.variant_id,
        "model_family": "single_direct_qnn",
        "execution_mode": "finite_shot_noisy",
        "optimizer_scheme": "two_pass_spsa_quantum_adam_classical",
        "backend_mode": args.backend_mode,
        "depth": args.depth,
        "n_ansatz_layers": args.n_ansatz_layers,
        "n_qubits": args.window_size,
        "n_quantum_features": len(readout_wires),
        "readout_wires": ",".join(str(q) for q in readout_wires),
        "shots": args.shots,
        "seed": args.seed,
        "use_output_tanh": args.use_output_tanh,
        "best_epoch": best_epoch,
        "best_val_loss_scaled": best_val_loss,
        "spsa_steps_completed": spsa_state.k - 1,
        "total_qnode_calls": model.qnode_call_count,
        "estimated_total_shots": estimated_total_shots,
        "train_rmse_original": train_metrics["rmse"],
        "val_rmse_original": val_metrics["rmse"],
        "test_rmse_original": test_metrics["rmse"],
        "train_mae_original": train_metrics["mae"],
        "val_mae_original": val_metrics["mae"],
        "test_mae_original": test_metrics["mae"],
        "train_pearson_corr": train_metrics["pearson_corr"],
        "val_pearson_corr": val_metrics["pearson_corr"],
        "test_pearson_corr": test_metrics["pearson_corr"],
        "train_spearman_corr": train_metrics["spearman_corr"],
        "val_spearman_corr": val_metrics["spearman_corr"],
        "test_spearman_corr": test_metrics["spearman_corr"],
        "train_kge": train_metrics["kge"],
        "val_kge": val_metrics["kge"],
        "test_kge": test_metrics["kge"],
        "train_nse": train_metrics["nse"],
        "val_nse": val_metrics["nse"],
        "test_nse": test_metrics["nse"],
        "train_bias": train_metrics["bias_mean_pred_minus_true"],
        "val_bias": val_metrics["bias_mean_pred_minus_true"],
        "test_bias": test_metrics["bias_mean_pred_minus_true"],
        "train_pred_std_over_true_std": train_metrics["pred_std_over_true_std"],
        "val_pred_std_over_true_std": val_metrics["pred_std_over_true_std"],
        "test_pred_std_over_true_std": test_metrics["pred_std_over_true_std"],
        "train_negative_prediction_rate": train_metrics["negative_prediction_rate"],
        "val_negative_prediction_rate": val_metrics["negative_prediction_rate"],
        "test_negative_prediction_rate": test_metrics["negative_prediction_rate"],
        "train_out_of_range_rate": train_metrics["out_of_range_rate"],
        "val_out_of_range_rate": val_metrics["out_of_range_rate"],
        "test_out_of_range_rate": test_metrics["out_of_range_rate"],
        "test_fft_spectral_cosine_similarity": test_fft["fft_spectral_cosine_similarity"],
        "test_abs_12_step_band_error": test_fft["abs_12_step_band_error"],
        "test_abs_short_period_2_to_6_error": test_fft["abs_short_period_2_to_6_error"],
        "elapsed_time_sec": elapsed,
        "max_rss_kb_python": max_rss_kb,
        "max_rss_gb_python": max_rss_kb / (1024 ** 2),
        "output_dir": str(out_dir),
    }

    summary_path = out_dir / "result_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(pd.DataFrame([summary]).T.to_string(header=False))

    print("\n" + "=" * 100)
    print("Saved files")
    print("=" * 100)
    for path in [
        out_dir / "config.json",
        history_path,
        best_model_path,
        last_checkpoint_path,
        final_model_path,
        out_dir / "learned_parameters.npz",
        predictions_npz_path,
        pred_long_path,
        pred_wide_path,
        metrics_path,
        fft_summary_path,
        fft_power_path,
        peak_low_path,
        top_errors_path,
        summary_path,
    ]:
        print(path)

    print("\nDONE")


if __name__ == "__main__":
    main()
