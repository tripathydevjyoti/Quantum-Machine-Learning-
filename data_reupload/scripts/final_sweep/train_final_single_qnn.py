#!/usr/bin/env python
import argparse
import copy
import json
import math
import os
import random
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml


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
# Model
# ============================================================

class DirectWindowQNN(nn.Module):
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

        dev = qml.device(qml_device, wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(x, weights):
            for reupload_layer in range(self.depth):
                for q in range(self.n_qubits):
                    qml.RY(self.angle_factor * x[q], wires=q)

                for ansatz_layer in range(self.n_ansatz_layers):
                    for q in range(self.n_qubits):
                        qml.Rot(
                            weights[reupload_layer, ansatz_layer, q, 0],
                            weights[reupload_layer, ansatz_layer, q, 1],
                            weights[reupload_layer, ansatz_layer, q, 2],
                            wires=q,
                        )

                    if self.n_qubits > 1:
                        for q in range(self.n_qubits - 1):
                            qml.CNOT(wires=[q, q + 1])
                        qml.CNOT(wires=[self.n_qubits - 1, 0])

            return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

        self.circuit = circuit

        self.q_weights = nn.Parameter(
            0.01 * torch.randn(
                self.depth,
                self.n_ansatz_layers,
                self.n_qubits,
                3,
                dtype=torch.float64,
            )
        )

        if self.use_output_tanh:
            self.classical_head = nn.Sequential(
                nn.Linear(self.n_qubits, 1, dtype=torch.float64),
                nn.Tanh(),
            )
        else:
            self.classical_head = nn.Linear(self.n_qubits, 1, dtype=torch.float64)

    def forward(self, x_batch):
        outputs = []
        for i in range(x_batch.shape[0]):
            z = self.circuit(x_batch[i], self.q_weights)
            if isinstance(z, (tuple, list)):
                z = torch.stack(list(z))
            outputs.append(z)
        z_batch = torch.stack(outputs, dim=0)
        y = self.classical_head(z_batch).squeeze(-1)
        return y


def evaluate_scaled(model, X, y, batch_size=32):
    model.eval()
    preds = []
    losses = []
    mse = nn.MSELoss(reduction="mean")

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size]
            yb = y[start:start + batch_size]
            pred = model(xb)
            loss = mse(pred, yb)
            preds.append(pred.detach().cpu().numpy())
            losses.append(float(loss.detach().cpu().item()) * len(xb))

    pred_all = np.concatenate(preds)
    loss_mean = np.sum(losses) / len(X)
    return loss_mean, pred_all


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--value_col", type=int, default=4)
    parser.add_argument("--train_end_idx", type=int, default=434)
    parser.add_argument("--window_size", type=int, default=14)
    parser.add_argument("--train_window_end", type=int, default=350)

    parser.add_argument("--scaler_data_min", type=float, default=0.0)
    parser.add_argument("--scaler_data_max", type=float, default=350.0)
    parser.add_argument("--scaler_feature_min", type=float, default=-1.0)
    parser.add_argument("--scaler_feature_max", type=float, default=1.0)

    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--n_ansatz_layers", type=int, default=1)
    parser.add_argument("--readout", type=str, default="all")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--qnn_epochs", type=int, default=100)
    parser.add_argument("--qnn_learning_rate", type=float, default=0.01)
    parser.add_argument("--qnn_batch_size", type=int, default=16)
    parser.add_argument("--angle_factor", type=float, default=math.pi)
    parser.add_argument("--qml_device", type=str, default="default.qubit")
    parser.add_argument("--diff_method", type=str, default="backprop")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_output_tanh", action="store_true")
    parser.add_argument("--print_every", type=int, default=10)

    parser.add_argument("--variant_id", type=str, default="single_tanh_a1")
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    start_time = time.time()
    set_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    out_dir = (
        Path(args.output_dir)
        / args.variant_id
        / f"depth_{args.depth}"
        / f"seed_{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Final single direct-window QNN")
    print("=" * 100)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"output_dir: {out_dir}")

    df = pd.read_csv(args.csv_path)
    y_original = pd.to_numeric(df.iloc[:, args.value_col], errors="coerce").dropna().to_numpy(dtype=np.float64)

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

    model = DirectWindowQNN(
        n_qubits=args.window_size,
        depth=args.depth,
        n_ansatz_layers=args.n_ansatz_layers,
        readout=args.readout,
        angle_factor=args.angle_factor,
        qml_device=args.qml_device,
        diff_method=args.diff_method,
        use_output_tanh=args.use_output_tanh,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.qnn_learning_rate)
    mse = nn.MSELoss(reduction="mean")

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    history = []

    n_train = X_train.shape[0]

    for epoch in range(1, args.qnn_epochs + 1):
        model.train()
        perm = torch.randperm(n_train)

        train_loss_sum = 0.0
        for start in range(0, n_train, args.qnn_batch_size):
            idx = perm[start:start + args.qnn_batch_size]
            xb = X_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.detach().cpu().item()) * len(idx)

        train_loss_epoch = train_loss_sum / n_train

        # Match the old training protocol:
        # train loss is the average minibatch loss accumulated during the training epoch.
        # We do NOT run a separate full train-set QNN evaluation every epoch.
        train_eval_loss = train_loss_epoch

        # Validation is still evaluated every epoch for best-checkpoint selection.
        val_eval_loss, val_pred_scaled = evaluate_scaled(model, X_val, y_val, batch_size=32)

        if val_eval_loss < best_val_loss:
            best_val_loss = val_eval_loss
            best_epoch = epoch
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": vars(args),
            }

        y_val_true_original = inverse_scale_fixed(
            data["y_val"], args.scaler_data_min, args.scaler_data_max,
            args.scaler_feature_min, args.scaler_feature_max
        )
        val_pred_original = inverse_scale_fixed(
            val_pred_scaled, args.scaler_data_min, args.scaler_data_max,
            args.scaler_feature_min, args.scaler_feature_max
        )

        # Full train RMSE is computed once at the end from the best checkpoint.
        train_rmse_original = np.nan
        val_rmse_original = np.sqrt(np.mean((val_pred_original - y_val_true_original) ** 2))

        history.append({
            "epoch": epoch,
            "train_loss_scaled": train_eval_loss,
            "val_loss_scaled": val_eval_loss,
            "train_rmse_original": train_rmse_original,
            "val_rmse_original": val_rmse_original,
            "best_val_loss_scaled_so_far": best_val_loss,
            "best_epoch_so_far": best_epoch,
            "elapsed_time_sec": time.time() - start_time,
        })

        if epoch % args.print_every == 0 or epoch == 1:
            print(
                f"Epoch [{epoch}/{args.qnn_epochs}] "
                f"TrainLoss={train_eval_loss:.8f} "
                f"ValLoss={val_eval_loss:.8f} "
                f"BestVal={best_val_loss:.8f} "
                f"TrainRMSE={train_rmse_original:.4f} "
                f"ValRMSE={val_rmse_original:.4f}"
            )

    # Restore best state
    model.load_state_dict(best_state["model_state_dict"])

    train_loss, train_pred_scaled = evaluate_scaled(model, X_train, y_train, batch_size=32)
    val_loss, val_pred_scaled = evaluate_scaled(model, X_val, y_val, batch_size=32)
    test_loss, test_pred_scaled = evaluate_scaled(model, X_test, y_test, batch_size=32)

    y_train_true_original = inverse_scale_fixed(data["y_train"], args.scaler_data_min, args.scaler_data_max, args.scaler_feature_min, args.scaler_feature_max)
    y_val_true_original = inverse_scale_fixed(data["y_val"], args.scaler_data_min, args.scaler_data_max, args.scaler_feature_min, args.scaler_feature_max)
    y_test_true_original = inverse_scale_fixed(data["y_test"], args.scaler_data_min, args.scaler_data_max, args.scaler_feature_min, args.scaler_feature_max)

    y_train_pred_original = inverse_scale_fixed(train_pred_scaled, args.scaler_data_min, args.scaler_data_max, args.scaler_feature_min, args.scaler_feature_max)
    y_val_pred_original = inverse_scale_fixed(val_pred_scaled, args.scaler_data_min, args.scaler_data_max, args.scaler_feature_min, args.scaler_feature_max)
    y_test_pred_original = inverse_scale_fixed(test_pred_scaled, args.scaler_data_min, args.scaler_data_max, args.scaler_feature_min, args.scaler_feature_max)

    # Save config
    config = vars(args).copy()
    config.update({
        "model_family": "single_direct_qnn",
        "output_mode": "tanh_head" if args.use_output_tanh else "linear_head",
        "n_qubits": args.window_size,
        "best_epoch": best_epoch,
        "best_val_loss_scaled": best_val_loss,
    })

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save history
    history_df = pd.DataFrame(history)
    history_path = out_dir / "history.csv"
    history_df.to_csv(history_path, index=False)

    # Save checkpoints
    best_model_path = out_dir / "best_model.pt"
    final_model_path = out_dir / "final_model.pt"
    torch.save(best_state, best_model_path)
    torch.save({
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "config": config,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }, final_model_path)

    # Save learned parameters
    params_npz = {}
    for name, param in model.named_parameters():
        params_npz[name.replace(".", "_")] = param.detach().cpu().numpy()
    np.savez(out_dir / "learned_parameters.npz", **params_npz)

    # Save predictions NPZ
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

    # Predictions CSVs
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

    # Metrics
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
            "depth": args.depth,
            "n_ansatz_layers": args.n_ansatz_layers,
            "split": split,
            "best_epoch": best_epoch,
            "best_val_loss_scaled": best_val_loss,
            "use_output_tanh": args.use_output_tanh,
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

    # One-row summary
    test_metrics = metrics_df[metrics_df["split"] == "test"].iloc[0].to_dict()
    val_metrics = metrics_df[metrics_df["split"] == "val"].iloc[0].to_dict()
    train_metrics = metrics_df[metrics_df["split"] == "train"].iloc[0].to_dict()
    test_fft = fft_summary_df[fft_summary_df["split"] == "test"].iloc[0].to_dict()

    summary = {
        "variant_id": args.variant_id,
        "model_family": "single_direct_qnn",
        "depth": args.depth,
        "n_ansatz_layers": args.n_ansatz_layers,
        "n_qubits": args.window_size,
        "seed": args.seed,
        "use_output_tanh": args.use_output_tanh,
        "best_epoch": best_epoch,
        "best_val_loss_scaled": best_val_loss,
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
    for p in [
        out_dir / "config.json",
        history_path,
        best_model_path,
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
        print(p)

    print("\nDONE")


if __name__ == "__main__":
    main()
