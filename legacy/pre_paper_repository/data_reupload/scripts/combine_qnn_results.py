import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project path setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine individual QNN depth-sweep result CSV files."
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(PROJECT_ROOT / "data_reupload" / "results" / "qnn_depth_sweep"),
        help="Root directory containing individual QNN result CSV files.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data_reupload" / "results" / "qnn_depth_sweep" / "combined"),
        help="Directory where combined result CSVs are saved.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="qnn_latent*_depth*_seed*.csv",
        help="Glob pattern for individual QNN result CSVs.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    print("=" * 80)
    print("Combining QNN result CSV files")
    print("=" * 80)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Results dir:  {results_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Pattern:      {args.pattern}")
    print("=" * 80)

    csv_files = sorted(results_dir.rglob(args.pattern))

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No result CSV files found under {results_dir} with pattern {args.pattern}"
        )

    print(f"Found {len(csv_files)} result CSV files.")

    dfs = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df["source_file"] = str(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping file due to error: {csv_file}")
            print(f"Error: {e}")

    if len(dfs) == 0:
        raise RuntimeError("No valid result CSV files could be read.")

    all_results = pd.concat(dfs, ignore_index=True)

    sort_cols = []

    for col in ["latent_dim", "n_qubits", "depth", "seed"]:
        if col in all_results.columns:
            sort_cols.append(col)

    if sort_cols:
        all_results = all_results.sort_values(sort_cols).reset_index(drop=True)

    all_results_path = output_dir / "all_qnn_results.csv"
    all_results.to_csv(all_results_path, index=False)

    print(f"Saved all results to: {all_results_path}")

    # --------------------------------------------------------
    # Create mean/std summary across seeds if possible
    # --------------------------------------------------------

    group_cols = []

    for col in [
        "latent_dim",
        "n_qubits",
        "window_size",
        "train_window_end",
        "depth",
        "n_ansatz_layers",
        "qnn_epochs",
        "qnn_learning_rate",
        "qnn_batch_size",
        "qml_device",
        "diff_method",
        "use_output_tanh",
    ]:
        if col in all_results.columns:
            group_cols.append(col)

    metric_cols = [
        col
        for col in all_results.columns
        if (
            col.endswith("_mse_scaled")
            or col.endswith("_rmse_scaled")
            or col.endswith("_mae_scaled")
            or col.endswith("_corr_scaled")
            or col.endswith("_mse_original")
            or col.endswith("_rmse_original")
            or col.endswith("_mae_original")
            or col.endswith("_corr_original")
            or col in ["best_val_loss_scaled", "best_epoch", "elapsed_time_sec"]
        )
    ]

    if len(group_cols) > 0 and len(metric_cols) > 0:
        summary = (
            all_results
            .groupby(group_cols, dropna=False)[metric_cols]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )

        # Flatten MultiIndex columns
        summary.columns = [
            "_".join([str(x) for x in col if str(x) != ""])
            for col in summary.columns.values
        ]

        summary_path = output_dir / "qnn_results_summary_by_seed.csv"
        summary.to_csv(summary_path, index=False)

        print(f"Saved summary results to: {summary_path}")

    # --------------------------------------------------------
    # Best depth by validation RMSE for each latent dimension
    # --------------------------------------------------------

    if "val_rmse_original" in all_results.columns:
        best_rows = []

        if "latent_dim" in all_results.columns:
            for latent_dim, group in all_results.groupby("latent_dim"):
                best_idx = group["val_rmse_original"].idxmin()
                best_rows.append(all_results.loc[best_idx])
        else:
            best_idx = all_results["val_rmse_original"].idxmin()
            best_rows.append(all_results.loc[best_idx])

        best_df = pd.DataFrame(best_rows).reset_index(drop=True)

        best_path = output_dir / "best_qnn_by_val_rmse.csv"
        best_df.to_csv(best_path, index=False)

        print(f"Saved best-by-validation results to: {best_path}")

    print("=" * 80)
    print("Combination complete.")
    print(f"Number of result files combined: {len(csv_files)}")
    print(f"Number of rows in all_results:   {len(all_results)}")
    print("=" * 80)


if __name__ == "__main__":
    main()