import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Project path setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one QNN depth/seed job using precomputed LSTM AE latents."
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        required=True,
        help="Latent dimension. This equals n_qubits.",
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Window size used to create the latent file.",
    )

    parser.add_argument(
        "--train_window_end",
        type=int,
        default=350,
        help="Train window end used to create the latent file.",
    )

    parser.add_argument(
        "--latent_file",
        type=str,
        default=None,
        help="Optional path to latent .npz file. If not provided, default path is used.",
    )

    parser.add_argument(
        "--depth",
        type=int,
        required=True,
        help="Number of data reuploading layers in the QNN.",
    )

    parser.add_argument(
        "--n_ansatz_layers",
        type=int,
        default=1,
        help="Number of trainable ansatz layers inside each reuploading block.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for QNN initialization and dataloader shuffling.",
    )

    parser.add_argument(
        "--qnn_epochs",
        type=int,
        default=300,
        help="Number of QNN training epochs.",
    )

    parser.add_argument(
        "--qnn_learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for QNN Adam optimizer.",
    )

    parser.add_argument(
        "--qnn_batch_size",
        type=int,
        default=16,
        help="Batch size for QNN training.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device: cpu, cuda, or mps.",
    )

    parser.add_argument(
        "--qml_device",
        type=str,
        default="default.qubit",
        help="PennyLane device, e.g. default.qubit or lightning.qubit.",
    )

    parser.add_argument(
        "--diff_method",
        type=str,
        default="backprop",
        help="PennyLane differentiation method, e.g. backprop or adjoint.",
    )

    parser.add_argument(
        "--use_output_tanh",
        action="store_true",
        help="If passed, uses Tanh after the final classical output layer.",
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=50,
        help="Print training status every N epochs. Use 0 to suppress.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data_reupload" / "results" / "qnn_depth_sweep"),
        help="Directory where QNN result CSVs are saved.",
    )

    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="If passed, saves prediction arrays to .npz.",
    )

    return parser.parse_args()


def get_default_latent_file(latent_dim: int, window_size: int, train_window_end: int) -> Path:
    return (
        PROJECT_ROOT
        / "data_reupload"
        / "checkpoints"
        / "latents"
        / f"latent_{latent_dim}"
        / f"latent_{latent_dim}_window_{window_size}_trainend_{train_window_end}.npz"
    )


# ============================================================
# Data loaders
# ============================================================

def make_qnn_loaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size: int,
):
    X_train_torch = torch.tensor(X_train, dtype=torch.float64)
    y_train_torch = torch.tensor(y_train, dtype=torch.float64)

    X_val_torch = torch.tensor(X_val, dtype=torch.float64)
    y_val_torch = torch.tensor(y_val, dtype=torch.float64)

    X_test_torch = torch.tensor(X_test, dtype=torch.float64)
    y_test_torch = torch.tensor(y_test, dtype=torch.float64)

    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    val_dataset = TensorDataset(X_val_torch, y_val_torch)
    test_dataset = TensorDataset(X_test_torch, y_test_torch)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# QNN circuit
# ============================================================

def create_qnode(
    n_qubits: int,
    n_reupload_layers: int,
    n_ansatz_layers: int = 1,
    qml_device: str = "default.qubit",
    diff_method: str = "backprop",
):
    dev = qml.device(qml_device, wires=n_qubits)

    def S(angles):
        for i in range(n_qubits):
            qml.RY(angles[i], wires=i)

    def W(theta_block):
        for ansatz_layer in range(n_ansatz_layers):
            for i in range(n_qubits):
                qml.Rot(
                    theta_block[ansatz_layer, i, 0],
                    theta_block[ansatz_layer, i, 1],
                    theta_block[ansatz_layer, i, 2],
                    wires=i,
                )

            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(q_params, angles):
        for layer in range(n_reupload_layers):
            S(angles)
            W(q_params[layer])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class HybridQNNRegressor(nn.Module):
    def __init__(
        self,
        n_qubits: int,
        n_reupload_layers: int,
        n_ansatz_layers: int = 1,
        qml_device: str = "default.qubit",
        diff_method: str = "backprop",
        use_output_tanh: bool = False,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_reupload_layers = n_reupload_layers
        self.n_ansatz_layers = n_ansatz_layers

        self.circuit = create_qnode(
            n_qubits=n_qubits,
            n_reupload_layers=n_reupload_layers,
            n_ansatz_layers=n_ansatz_layers,
            qml_device=qml_device,
            diff_method=diff_method,
        )

        self.q_params = nn.Parameter(
            0.01
            * torch.randn(
                n_reupload_layers,
                n_ansatz_layers,
                n_qubits,
                3,
                dtype=torch.float64,
            )
        )

        if use_output_tanh:
            self.classical_head = nn.Sequential(
                nn.Linear(n_qubits, 1),
                nn.Tanh(),
            ).double()
        else:
            self.classical_head = nn.Linear(n_qubits, 1).double()

    def forward(self, angles_batch):
        q_outputs = []

        for angles in angles_batch:
            q_out = self.circuit(self.q_params, angles)
            q_out = torch.stack(q_out)
            q_outputs.append(q_out)

        q_outputs = torch.stack(q_outputs)

        y_pred = self.classical_head(q_outputs)

        return y_pred.squeeze(-1)


# ============================================================
# Training / evaluation
# ============================================================

def train_one_qnn_model(
    model,
    train_loader,
    val_loader,
    n_epochs: int,
    learning_rate: float,
    device: str = "cpu",
    print_every: int = 50,
):
    device = torch.device(device)

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = None

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * X_batch.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)

                loss = criterion(y_pred, y_batch)

                total_val_loss += loss.item() * X_batch.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        if print_every is not None and print_every > 0:
            if (epoch + 1) % print_every == 0:
                print(
                    f"Epoch [{epoch + 1}/{n_epochs}] "
                    f"Train Loss: {avg_train_loss:.8f} "
                    f"Val Loss: {avg_val_loss:.8f}"
                )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }

    return model, history


def predict_with_model(model, loader, device: str = "cpu"):
    device = torch.device(device)

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            y_pred = model(X_batch)

            preds.append(y_pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    return preds, targets


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = np.nan

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
    }


def evaluate_qnn_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    y_train_original,
    y_val_original,
    y_test_original,
    scaler_data_min: float,
    scaler_data_max: float,
    scaler_feature_min: float,
    scaler_feature_max: float,
    device: str = "cpu",
):
    y_train_pred_scaled, y_train_true_scaled = predict_with_model(
        model, train_loader, device=device
    )
    y_val_pred_scaled, y_val_true_scaled = predict_with_model(
        model, val_loader, device=device
    )
    y_test_pred_scaled, y_test_true_scaled = predict_with_model(
        model, test_loader, device=device
    )

    train_scaled_metrics = regression_metrics(
        y_train_true_scaled, y_train_pred_scaled
    )
    val_scaled_metrics = regression_metrics(
        y_val_true_scaled, y_val_pred_scaled
    )
    test_scaled_metrics = regression_metrics(
        y_test_true_scaled, y_test_pred_scaled
    )

    def inverse_fixed_minmax(y_scaled):
        a = scaler_feature_min
        b = scaler_feature_max
        data_min = scaler_data_min
        data_max = scaler_data_max

        return data_min + (y_scaled - a) * (data_max - data_min) / (b - a)

    y_train_pred_original = inverse_fixed_minmax(y_train_pred_scaled)
    y_val_pred_original = inverse_fixed_minmax(y_val_pred_scaled)
    y_test_pred_original = inverse_fixed_minmax(y_test_pred_scaled)

    train_original_metrics = regression_metrics(
        y_train_original, y_train_pred_original
    )
    val_original_metrics = regression_metrics(
        y_val_original, y_val_pred_original
    )
    test_original_metrics = regression_metrics(
        y_test_original, y_test_pred_original
    )

    predictions = {
        "y_train_pred_scaled": y_train_pred_scaled,
        "y_train_true_scaled": y_train_true_scaled,
        "y_val_pred_scaled": y_val_pred_scaled,
        "y_val_true_scaled": y_val_true_scaled,
        "y_test_pred_scaled": y_test_pred_scaled,
        "y_test_true_scaled": y_test_true_scaled,
        "y_train_pred_original": y_train_pred_original,
        "y_train_true_original": y_train_original,
        "y_val_pred_original": y_val_pred_original,
        "y_val_true_original": y_val_original,
        "y_test_pred_original": y_test_pred_original,
        "y_test_true_original": y_test_original,
    }

    return {
        "train_scaled": train_scaled_metrics,
        "val_scaled": val_scaled_metrics,
        "test_scaled": test_scaled_metrics,
        "train_original": train_original_metrics,
        "val_original": val_original_metrics,
        "test_original": test_original_metrics,
        "predictions": predictions,
    }


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    set_seed(args.seed)

    n_qubits = args.latent_dim

    if args.latent_file is None:
        latent_file = get_default_latent_file(
            latent_dim=args.latent_dim,
            window_size=args.window_size,
            train_window_end=args.train_window_end,
        )
    else:
        latent_file = Path(args.latent_file)

    if not latent_file.exists():
        raise FileNotFoundError(f"Latent file not found: {latent_file}")

    output_root = Path(args.output_dir)
    output_dir = (
        output_root
        / f"latent_{args.latent_dim}"
        / f"depth_{args.depth}"
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("QNN depth job")
    print("=" * 80)
    print(f"PROJECT_ROOT:          {PROJECT_ROOT}")
    print(f"latent_file:           {latent_file}")
    print(f"output_dir:            {output_dir}")
    print(f"latent_dim/n_qubits:   {args.latent_dim}")
    print(f"depth:                 {args.depth}")
    print(f"n_ansatz_layers:       {args.n_ansatz_layers}")
    print(f"seed:                  {args.seed}")
    print(f"qnn_epochs:            {args.qnn_epochs}")
    print(f"qnn_learning_rate:     {args.qnn_learning_rate}")
    print(f"qnn_batch_size:        {args.qnn_batch_size}")
    print(f"torch device:          {args.device}")
    print(f"PennyLane device:      {args.qml_device}")
    print(f"diff_method:           {args.diff_method}")
    print(f"use_output_tanh:       {args.use_output_tanh}")
    print("=" * 80)

    # --------------------------------------------------------
    # Load latent dataset
    # --------------------------------------------------------

    data = np.load(latent_file, allow_pickle=True)

    X_qnn_train = data["X_qnn_train"]
    X_qnn_val = data["X_qnn_val"]
    X_qnn_test = data["X_qnn_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    y_train_original = data["y_train_original"]
    y_val_original = data["y_val_original"]
    y_test_original = data["y_test_original"]

    scaler_feature_min = float(data["scaler_feature_min"][0])
    scaler_feature_max = float(data["scaler_feature_max"][0])
    scaler_data_min = float(data["scaler_data_min"][0])
    scaler_data_max = float(data["scaler_data_max"][0])

    print("X_qnn_train:", X_qnn_train.shape)
    print("X_qnn_val:", X_qnn_val.shape)
    print("X_qnn_test:", X_qnn_test.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape)

    if X_qnn_train.shape[1] != n_qubits:
        raise ValueError(
            f"X_qnn_train has dimension {X_qnn_train.shape[1]}, "
            f"but n_qubits={n_qubits}"
        )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader, val_loader, test_loader = make_qnn_loaders(
        X_train=X_qnn_train,
        y_train=y_train,
        X_val=X_qnn_val,
        y_val=y_val,
        X_test=X_qnn_test,
        y_test=y_test,
        batch_size=args.qnn_batch_size,
    )

    # --------------------------------------------------------
    # Build and train QNN
    # --------------------------------------------------------

    model = HybridQNNRegressor(
        n_qubits=n_qubits,
        n_reupload_layers=args.depth,
        n_ansatz_layers=args.n_ansatz_layers,
        qml_device=args.qml_device,
        diff_method=args.diff_method,
        use_output_tanh=args.use_output_tanh,
    )

    start_time = time.time()

    model, history = train_one_qnn_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.qnn_epochs,
        learning_rate=args.qnn_learning_rate,
        device=args.device,
        print_every=args.print_every,
    )

    elapsed_time = time.time() - start_time

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    eval_results = evaluate_qnn_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        y_train_original=y_train_original,
        y_val_original=y_val_original,
        y_test_original=y_test_original,
        scaler_data_min=scaler_data_min,
        scaler_data_max=scaler_data_max,
        scaler_feature_min=scaler_feature_min,
        scaler_feature_max=scaler_feature_max,
        device=args.device,
    )

    # --------------------------------------------------------
    # Save one-row result CSV
    # --------------------------------------------------------

    row = {
        "latent_dim": args.latent_dim,
        "n_qubits": n_qubits,
        "window_size": args.window_size,
        "train_window_end": args.train_window_end,
        "depth": args.depth,
        "n_ansatz_layers": args.n_ansatz_layers,
        "seed": args.seed,
        "qnn_epochs": args.qnn_epochs,
        "qnn_learning_rate": args.qnn_learning_rate,
        "qnn_batch_size": args.qnn_batch_size,
        "qml_device": args.qml_device,
        "diff_method": args.diff_method,
        "use_output_tanh": args.use_output_tanh,
        "latent_file": str(latent_file),
        "best_val_loss_scaled": history["best_val_loss"],
        "best_epoch": history["best_epoch"],
        "elapsed_time_sec": elapsed_time,

        "train_mse_scaled": eval_results["train_scaled"]["mse"],
        "val_mse_scaled": eval_results["val_scaled"]["mse"],
        "test_mse_scaled": eval_results["test_scaled"]["mse"],

        "train_rmse_scaled": eval_results["train_scaled"]["rmse"],
        "val_rmse_scaled": eval_results["val_scaled"]["rmse"],
        "test_rmse_scaled": eval_results["test_scaled"]["rmse"],

        "train_mae_scaled": eval_results["train_scaled"]["mae"],
        "val_mae_scaled": eval_results["val_scaled"]["mae"],
        "test_mae_scaled": eval_results["test_scaled"]["mae"],

        "train_corr_scaled": eval_results["train_scaled"]["corr"],
        "val_corr_scaled": eval_results["val_scaled"]["corr"],
        "test_corr_scaled": eval_results["test_scaled"]["corr"],

        "train_mse_original": eval_results["train_original"]["mse"],
        "val_mse_original": eval_results["val_original"]["mse"],
        "test_mse_original": eval_results["test_original"]["mse"],

        "train_rmse_original": eval_results["train_original"]["rmse"],
        "val_rmse_original": eval_results["val_original"]["rmse"],
        "test_rmse_original": eval_results["test_original"]["rmse"],

        "train_mae_original": eval_results["train_original"]["mae"],
        "val_mae_original": eval_results["val_original"]["mae"],
        "test_mae_original": eval_results["test_original"]["mae"],

        "train_corr_original": eval_results["train_original"]["corr"],
        "val_corr_original": eval_results["val_original"]["corr"],
        "test_corr_original": eval_results["test_original"]["corr"],
    }

    result_csv = output_dir / (
        f"qnn_latent{args.latent_dim}"
        f"_depth{args.depth}"
        f"_seed{args.seed}.csv"
    )

    pd.DataFrame([row]).to_csv(result_csv, index=False)

    # --------------------------------------------------------
    # Optionally save predictions
    # --------------------------------------------------------

    if args.save_predictions:
        pred_file = output_dir / (
            f"predictions_latent{args.latent_dim}"
            f"_depth{args.depth}"
            f"_seed{args.seed}.npz"
        )

        np.savez(
            pred_file,
            **eval_results["predictions"],
            train_loss=np.array(history["train_loss"]),
            val_loss=np.array(history["val_loss"]),
        )

        print(f"Predictions saved to: {pred_file}")

    print("=" * 80)
    print("QNN job complete.")
    print(f"Result CSV:              {result_csv}")
    print(f"Best val loss scaled:    {history['best_val_loss']:.8f}")
    print(f"Best epoch:              {history['best_epoch']}")
    print(f"Val RMSE original:       {row['val_rmse_original']:.8f}")
    print(f"Test RMSE original:      {row['test_rmse_original']:.8f}")
    print(f"Elapsed time:            {elapsed_time:.2f} sec")
    print("=" * 80)


if __name__ == "__main__":
    main()