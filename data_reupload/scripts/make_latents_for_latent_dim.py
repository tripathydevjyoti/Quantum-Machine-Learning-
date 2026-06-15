import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# Project path setup
# ============================================================

# This script lives in:
#   data_reupload/scripts/make_latents_for_latent_dim.py
#
# PROJECT_ROOT is:
#   Quantum-Machine-Learning-/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from data_reupload.models.LSTMAE_pipeline import (
    load_lstm_ae_checkpoint,
    extract_series,
    series_to_Xy,
    extract_latents,
)


# ============================================================
# Argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create saved latent vectors and QNN angle arrays from a trained LSTM AE."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(
            PROJECT_ROOT
            / "data"
            / "raw"
            / "SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv"
        ),
        help="Path to the raw CSV file.",
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        required=True,
        help="Latent dimension. This should equal n_qubits.",
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Sliding window size used by the AE.",
    )

    parser.add_argument(
        "--best_config_csv",
        type=str,
        default=None,
        help=(
            "Path to best_ae_config.csv. If not provided, the script uses "
            "data_reupload/results/ae_tuning/latent_{latent_dim}_window_{window_size}/best_ae_config.csv"
        ),
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Optional direct path to AE checkpoint. If not provided, it is read "
            "from best_ae_config.csv."
        ),
    )

    parser.add_argument(
        "--train_end_idx",
        type=int,
        default=434,
        help="Raw dataframe index where df_train ends and df_test begins.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used for extracting latents.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data_reupload" / "checkpoints" / "latents"),
        help="Directory where latent .npz files are saved.",
    )

    return parser.parse_args()


# ============================================================
# Helper functions
# ============================================================

def get_default_best_config_csv(latent_dim: int, window_size: int) -> Path:
    return (
        PROJECT_ROOT
        / "data_reupload"
        / "results"
        / "ae_tuning"
        / f"latent_{latent_dim}_window_{window_size}"
        / "best_ae_config.csv"
    )


def load_best_checkpoint_path(args) -> Path:
    if args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        return checkpoint_path

    if args.best_config_csv is None:
        best_config_csv = get_default_best_config_csv(
            latent_dim=args.latent_dim,
            window_size=args.window_size,
        )
    else:
        best_config_csv = Path(args.best_config_csv)

    if not best_config_csv.exists():
        raise FileNotFoundError(f"Best config CSV not found: {best_config_csv}")

    best_df = pd.read_csv(best_config_csv)

    if "checkpoint_path" not in best_df.columns:
        raise ValueError(
            f"'checkpoint_path' column not found in best config CSV: {best_config_csv}"
        )

    checkpoint_path = Path(best_df.iloc[0]["checkpoint_path"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path from CSV does not exist: {checkpoint_path}")

    return checkpoint_path


def extract_scaler_metadata(scaler):
    """
    Save enough scaler information to invert predictions later without
    needing to reload the full scaler object.

    Works for sklearn MinMaxScaler and for the fixed-range MinMaxScaler style.
    """
    metadata = {}

    if hasattr(scaler, "feature_range"):
        metadata["scaler_feature_min"] = float(scaler.feature_range[0])
        metadata["scaler_feature_max"] = float(scaler.feature_range[1])
    else:
        metadata["scaler_feature_min"] = -1.0
        metadata["scaler_feature_max"] = 1.0

    if hasattr(scaler, "data_min_"):
        metadata["scaler_data_min"] = float(np.asarray(scaler.data_min_).reshape(-1)[0])
    else:
        metadata["scaler_data_min"] = np.nan

    if hasattr(scaler, "data_max_"):
        metadata["scaler_data_max"] = float(np.asarray(scaler.data_max_).reshape(-1)[0])
    else:
        metadata["scaler_data_max"] = np.nan

    return metadata


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    csv_path = Path(args.csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    checkpoint_path = load_best_checkpoint_path(args)

    output_root = Path(args.output_dir)
    latent_output_dir = output_root / f"latent_{args.latent_dim}"
    latent_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Creating latent vectors for QNN training")
    print("=" * 80)
    print(f"PROJECT_ROOT:     {PROJECT_ROOT}")
    print(f"CSV path:         {csv_path}")
    print(f"Checkpoint path:  {checkpoint_path}")
    print(f"Output dir:       {latent_output_dir}")
    print(f"latent_dim:       {args.latent_dim}")
    print(f"window_size:      {args.window_size}")
    print(f"train_end_idx:    {args.train_end_idx}")
    print(f"device:           {args.device}")
    print("=" * 80)

    # --------------------------------------------------------
    # Load AE checkpoint
    # --------------------------------------------------------

    ae_checkpoint = load_lstm_ae_checkpoint(
        checkpoint_path=str(checkpoint_path),
        device=args.device,
    )

    model = ae_checkpoint["model"]
    scaler = ae_checkpoint["scaler"]
    config = ae_checkpoint["config"]

    print("Loaded AE config:")
    print(config)

    if config.latent_dim != args.latent_dim:
        raise ValueError(
            f"Checkpoint latent_dim={config.latent_dim}, but requested latent_dim={args.latent_dim}"
        )

    if config.window_size != args.window_size:
        raise ValueError(
            f"Checkpoint window_size={config.window_size}, but requested window_size={args.window_size}"
        )

    value_col = config.value_col
    window_size = config.window_size
    train_window_end = config.train_window_end

    # --------------------------------------------------------
    # Load and split raw dataframe
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    df_train = df.iloc[: args.train_end_idx].copy()
    df_test = df.iloc[args.train_end_idx :].copy()

    print("df shape:", df.shape)
    print("df_train shape:", df_train.shape)
    print("df_test shape:", df_test.shape)

    # --------------------------------------------------------
    # Recreate train/validation windows from df_train
    # --------------------------------------------------------

    train_series = extract_series(df_train, value_col)
    train_series_scaled = scaler.transform(train_series).astype(np.float32)

    X_all, y_all = series_to_Xy(train_series_scaled, window_size)

    X_train = X_all[:train_window_end]
    y_train = y_all[:train_window_end]

    X_val = X_all[train_window_end:]
    y_val = y_all[train_window_end:]

    print("X_all:", X_all.shape)
    print("y_all:", y_all.shape)
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    # --------------------------------------------------------
    # Create test windows from df_test
    # --------------------------------------------------------

    test_series = extract_series(df_test, value_col)
    test_series_scaled = scaler.transform(test_series).astype(np.float32)

    X_test, y_test = series_to_Xy(test_series_scaled, window_size)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # --------------------------------------------------------
    # Extract latents
    # --------------------------------------------------------

    z_train = extract_latents(
        model=model,
        X=X_train,
        device=args.device,
    )

    z_val = extract_latents(
        model=model,
        X=X_val,
        device=args.device,
    )

    z_test = extract_latents(
        model=model,
        X=X_test,
        device=args.device,
    )

    print("z_train:", z_train.shape)
    print("z_val:", z_val.shape)
    print("z_test:", z_test.shape)

    # --------------------------------------------------------
    # Convert latents to QNN angles
    # --------------------------------------------------------

    X_qnn_train = np.pi * z_train
    X_qnn_val = np.pi * z_val
    X_qnn_test = np.pi * z_test

    y_train = y_train.reshape(-1)
    y_val = y_val.reshape(-1)
    y_test = y_test.reshape(-1)

    print("X_qnn_train:", X_qnn_train.shape)
    print("X_qnn_val:", X_qnn_val.shape)
    print("X_qnn_test:", X_qnn_test.shape)

    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape)

    print("Angle distribution:")
    print("Train min/max:", X_qnn_train.min(), X_qnn_train.max())
    print("Val min/max:", X_qnn_val.min(), X_qnn_val.max())
    print("Test min/max:", X_qnn_test.min(), X_qnn_test.max())

    print("Fraction |theta| > pi:")
    print("Train:", np.mean(np.abs(X_qnn_train.flatten()) > np.pi))
    print("Val:", np.mean(np.abs(X_qnn_val.flatten()) > np.pi))
    print("Test:", np.mean(np.abs(X_qnn_test.flatten()) > np.pi))

    # --------------------------------------------------------
    # Original-scale targets for later metrics
    # --------------------------------------------------------

    y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_val_original = scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    # --------------------------------------------------------
    # Save latent dataset
    # --------------------------------------------------------

    scaler_metadata = extract_scaler_metadata(scaler)

    output_file = (
        latent_output_dir
        / f"latent_{args.latent_dim}_window_{window_size}_trainend_{train_window_end}.npz"
    )

    np.savez(
        output_file,

        # Latent vectors
        z_train=z_train,
        z_val=z_val,
        z_test=z_test,

        # QNN-ready angles
        X_qnn_train=X_qnn_train,
        X_qnn_val=X_qnn_val,
        X_qnn_test=X_qnn_test,

        # Scaled targets
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,

        # Original-scale targets
        y_train_original=y_train_original,
        y_val_original=y_val_original,
        y_test_original=y_test_original,

        # Optional original windows if needed later
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,

        # Metadata
        latent_dim=np.array([args.latent_dim]),
        n_qubits=np.array([args.latent_dim]),
        window_size=np.array([window_size]),
        train_window_end=np.array([train_window_end]),
        train_end_idx=np.array([args.train_end_idx]),
        value_col=np.array([str(value_col)]),
        checkpoint_path=np.array([str(checkpoint_path)]),

        scaler_feature_min=np.array([scaler_metadata["scaler_feature_min"]]),
        scaler_feature_max=np.array([scaler_metadata["scaler_feature_max"]]),
        scaler_data_min=np.array([scaler_metadata["scaler_data_min"]]),
        scaler_data_max=np.array([scaler_metadata["scaler_data_max"]]),
    )

    print("=" * 80)
    print("Latent dataset saved.")
    print(f"Output file: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()