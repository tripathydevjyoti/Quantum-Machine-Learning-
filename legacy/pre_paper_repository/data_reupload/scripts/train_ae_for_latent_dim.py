import argparse
import os
import sys
import time
import random
from pathlib import Path
from dataclasses import asdict
from itertools import product

import numpy as np
import pandas as pd
import torch


# ============================================================
# Project path setup
# ============================================================

# This script lives in:
#   data_reupload/scripts/train_ae_for_latent_dim.py
#
# PROJECT_ROOT is:
#   Quantum-Machine-Learning-/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from data_reupload.models.LSTMAE_pipeline import (
    LSTMAEConfig,
    train_lstm_ae_latent_pipeline,
)


# ============================================================
# Reproducibility helper
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
        description="Tune/train LSTM AE for a fixed latent dimension."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv"),
        help="Path to the raw CSV file.",
    )

    parser.add_argument(
        "--value_col",
        type=str,
        default="4",
        help="Column index or column name for the time-series value. Use '4' for df.iloc[:, 4].",
    )

    parser.add_argument(
        "--train_end_idx",
        type=int,
        default=434,
        help="Raw dataframe index where df_train ends and df_test begins.",
    )

    parser.add_argument(
        "--train_window_end",
        type=int,
        default=350,
        help="Number of windowed samples used for AE training. Remaining windows are validation.",
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Sliding window size.",
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        required=True,
        help="Latent dimension. This should equal n_qubits later.",
    )

    parser.add_argument(
        "--n_epochs",
        type=int,
        default=1500,
        help="Number of AE training epochs.",
    )

    parser.add_argument(
        "--learning_rates",
        type=str,
        default="0.001,0.0005,0.0001",
        help="Comma-separated learning rates for grid search.",
    )

    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="16,32,64",
        help="Comma-separated batch sizes for grid search.",
    )

    parser.add_argument(
        "--dropout_ratio",
        type=float,
        default=0.0,
        help="Dropout ratio. Usually 0.0 for current single-layer LSTM AE.",
    )

    parser.add_argument(
        "--scaler_type",
        type=str,
        default="fixed_range",
        choices=["fixed_range", "minmax_train"],
        help="Scaling strategy.",
    )

    parser.add_argument(
        "--fixed_min",
        type=float,
        default=0.0,
        help="Fixed scaler minimum, e.g. 0.0 for rainfall.",
    )

    parser.add_argument(
        "--fixed_max",
        type=float,
        default=400.0,
        help="Fixed scaler maximum, e.g. 400.0 for rainfall.",
    )

    parser.add_argument(
        "--clip_fixed_scaler",
        action="store_true",
        help="If passed, clips fixed-range scaled values to the feature range.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for AE training: cpu, cuda, or mps.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=0,
        help="Print AE epoch losses every N epochs. Use 0 to suppress epoch printing.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data_reupload" / "results" / "ae_tuning"),
        help="Directory where AE tuning outputs will be saved.",
    )

    return parser.parse_args()


def parse_value_col(value_col_arg: str):
    """
    Allows value_col to be passed as either:
        --value_col 4
    or:
        --value_col rainfall_column_name
    """
    try:
        return int(value_col_arg)
    except ValueError:
        return value_col_arg


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ============================================================
# Main AE tuning function
# ============================================================

def main():
    args = parse_args()

    set_seed(args.seed)

    csv_path = Path(args.csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    value_col = parse_value_col(args.value_col)

    learning_rates = parse_float_list(args.learning_rates)
    batch_sizes = parse_int_list(args.batch_sizes)

    output_dir = Path(args.output_dir)
    latent_output_dir = output_dir / f"latent_{args.latent_dim}_window_{args.window_size}"
    latent_output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = PROJECT_ROOT / "data_reupload" / "checkpoints" / "ae" / f"latent_{args.latent_dim}"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LSTM AE tuning for fixed latent dimension")
    print("=" * 80)
    print(f"PROJECT_ROOT:       {PROJECT_ROOT}")
    print(f"CSV path:           {csv_path}")
    print(f"Output dir:         {latent_output_dir}")
    print(f"Checkpoints dir:    {checkpoints_dir}")
    print(f"latent_dim:         {args.latent_dim}")
    print(f"window_size:        {args.window_size}")
    print(f"train_end_idx:      {args.train_end_idx}")
    print(f"train_window_end:   {args.train_window_end}")
    print(f"learning_rates:     {learning_rates}")
    print(f"batch_sizes:        {batch_sizes}")
    print(f"device:             {args.device}")
    print(f"seed:               {args.seed}")
    print("=" * 80)

    # --------------------------------------------------------
    # Load and split data
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    df_train = df.iloc[: args.train_end_idx].copy()
    df_test = df.iloc[args.train_end_idx :].copy()

    print("df shape:", df.shape)
    print("df_train shape:", df_train.shape)
    print("df_test shape:", df_test.shape)

    # --------------------------------------------------------
    # Base config
    # --------------------------------------------------------

    base_config = LSTMAEConfig(
        value_col=value_col,
        window_size=args.window_size,
        latent_dim=args.latent_dim,
        train_window_end=args.train_window_end,
        batch_size=batch_sizes[0],
        n_epochs=args.n_epochs,
        learning_rate=learning_rates[0],
        dropout_ratio=args.dropout_ratio,
        scaler_type=args.scaler_type,
        scaler_feature_range=(-1.0, 1.0),
        fixed_min=args.fixed_min,
        fixed_max=args.fixed_max,
        clip_fixed_scaler=args.clip_fixed_scaler,
        use_act=False,
        device=args.device,
        save_path=None,
        print_every=args.print_every,
    )

    # --------------------------------------------------------
    # Grid search
    # --------------------------------------------------------

    results = []

    best_val_loss = float("inf")
    best_row = None
    best_checkpoint_path = None

    grid = list(product(learning_rates, batch_sizes))
    total_runs = len(grid)

    for run_id, (lr, batch_size) in enumerate(grid, start=1):
        set_seed(args.seed)

        run_name = (
            f"ae_latent{args.latent_dim}"
            f"_win{args.window_size}"
            f"_lr{lr}"
            f"_bs{batch_size}"
            f"_seed{args.seed}"
        )

        checkpoint_path = checkpoints_dir / f"{run_name}.pth"

        config_dict = asdict(base_config)
        config_dict["learning_rate"] = lr
        config_dict["batch_size"] = batch_size
        config_dict["save_path"] = str(checkpoint_path)

        config = LSTMAEConfig(**config_dict)

        print("=" * 80)
        print(f"Run {run_id}/{total_runs}: {run_name}")
        print(f"learning_rate: {lr}")
        print(f"batch_size:    {batch_size}")
        print(f"checkpoint:    {checkpoint_path}")

        start_time = time.time()

        lstm_results = train_lstm_ae_latent_pipeline(
            df=df_train,
            config=config,
            fit_scaler_in_train=True,
        )

        elapsed_time = time.time() - start_time

        history = lstm_results["history"]

        final_train_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]
        best_run_val_loss = history["best_val_loss"]

        row = {
            "run_id": run_id,
            "run_name": run_name,
            "seed": args.seed,
            "csv_path": str(csv_path),
            "train_end_idx": args.train_end_idx,
            "value_col": str(value_col),
            "window_size": args.window_size,
            "latent_dim": args.latent_dim,
            "train_window_end": args.train_window_end,
            "learning_rate": lr,
            "batch_size": batch_size,
            "n_epochs": args.n_epochs,
            "dropout_ratio": args.dropout_ratio,
            "scaler_type": args.scaler_type,
            "fixed_min": args.fixed_min,
            "fixed_max": args.fixed_max,
            "clip_fixed_scaler": args.clip_fixed_scaler,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_run_val_loss,
            "elapsed_time_sec": elapsed_time,
            "checkpoint_path": str(checkpoint_path),
        }

        results.append(row)

        print(f"Final train loss: {final_train_loss:.8f}")
        print(f"Final val loss:   {final_val_loss:.8f}")
        print(f"Best val loss:    {best_run_val_loss:.8f}")
        print(f"Elapsed time:     {elapsed_time:.2f} sec")

        if best_run_val_loss < best_val_loss:
            best_val_loss = best_run_val_loss
            best_row = row
            best_checkpoint_path = checkpoint_path

            print("New best AE checkpoint.")

    # --------------------------------------------------------
    # Save tuning results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("best_val_loss").reset_index(drop=True)

    tuning_csv_path = latent_output_dir / "ae_tuning_results.csv"
    results_df.to_csv(tuning_csv_path, index=False)

    best_csv_path = latent_output_dir / "best_ae_config.csv"
    pd.DataFrame([best_row]).to_csv(best_csv_path, index=False)

    print("=" * 80)
    print("AE tuning complete.")
    print(f"Best val loss:       {best_val_loss:.8f}")
    print(f"Best checkpoint:     {best_checkpoint_path}")
    print(f"Tuning results CSV:  {tuning_csv_path}")
    print(f"Best config CSV:     {best_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()