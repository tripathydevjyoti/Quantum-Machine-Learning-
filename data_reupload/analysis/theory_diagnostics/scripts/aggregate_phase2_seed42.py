from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TAUS = (1, 6, 12, 24)


# ============================================================
# UTILITIES
# ============================================================

def numeric_or_nan(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def first_existing_numeric(row: dict, aliases):
    for col in aliases:
        if col in row:
            value = numeric_or_nan(row[col])
            if np.isfinite(value):
                return value, col
    return np.nan, None


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)


def add_metadata(df, A, L, seed):
    df = df.copy()
    df.insert(0, "seed", seed)
    df.insert(0, "depth_L", L)
    df.insert(0, "ansatz_layers_A", A)
    return df


def save_depth_plot(
    master,
    metric,
    ylabel,
    title,
    output_path,
):
    if metric not in master.columns:
        return False

    values = pd.to_numeric(
        master[metric],
        errors="coerce",
    )

    if values.notna().sum() == 0:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))

    plotted = False

    for A in [1, 2]:
        g = (
            master[
                master["ansatz_layers_A"] == A
            ]
            .sort_values("depth_L")
            .copy()
        )

        y = pd.to_numeric(
            g[metric],
            errors="coerce",
        )

        valid = y.notna()

        if valid.sum() == 0:
            continue

        ax.plot(
            g.loc[valid, "depth_L"],
            y.loc[valid],
            marker="o",
            label=f"A={A}",
        )

        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Reupload depth L")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)

    return True


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
        "--phase2_root",
        required=True,
    )

    parser.add_argument(
        "--phase1_kernel_summary",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    args = parser.parse_args()

    project_root = Path(args.project_root)
    phase2_root = Path(args.phase2_root)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed = int(args.seed)

    print("=" * 110)
    print("PHASE II COMPLETE DEPTH-SWEEP AGGREGATION")
    print("=" * 110)
    print(f"Seed:         {seed}")
    print(f"Phase-II:     {phase2_root}")
    print(f"Output:       {output_dir}")
    print("=" * 110)

    # ========================================================
    # STORAGE
    # ========================================================

    master_rows = []

    stage_fourier_all = []
    fidelity_all = []
    kappa_all = []
    head_all = []
    stage_metadata_all = []

    forecast_source_map = {}

    # ========================================================
    # STANDARD FORECAST-METRIC ALIASES
    #
    # The original result_summary.csv is preserved in full.
    # These aliases merely expose a few standard fields under
    # common names for depth plots.
    # ========================================================

    FORECAST_ALIASES = {

        "forecast_test_rmse": [
            "test_rmse_original",
            "test_rmse",
            "rmse_test_original",
            "test_RMSE",
            "test_rmse_orig",
        ],

        "forecast_test_mae": [
            "test_mae_original",
            "test_mae",
            "mae_test_original",
            "test_MAE",
        ],

        "forecast_test_pearson": [
            "test_pearson",
            "test_pearson_r",
            "test_pearson_corr",
            "pearson_test",
            "pearson_r_test",
        ],

        "forecast_test_spearman": [
            "test_spearman",
            "test_spearman_r",
            "test_spearman_corr",
            "spearman_test",
        ],

        "forecast_test_kge": [
            "test_kge",
            "kge_test",
        ],

        "forecast_test_nse": [
            "test_nse",
            "nse_test",
        ],

        "forecast_fft_cosine": [
            "test_fft_cosine_similarity",
            "test_fft_spectral_cosine_similarity",
            "fft_cosine_similarity_test",
            "test_fft_cosine",
            "fft_cosine_similarity",
        ],

        "forecast_amplitude_ratio": [
            "test_amplitude_ratio",
            "test_pred_std_over_true_std",
            "amplitude_ratio_test",
            "prediction_amplitude_ratio_test",
            "test_pred_std_ratio",
            "test_std_ratio",
        ],

        "forecast_best_val_rmse": [
            "best_val_rmse_original",
            "val_rmse_original",
            "best_validation_rmse_original",
        ],

        "forecast_runtime_minutes": [
            "runtime_minutes",
            "elapsed_minutes",
            "total_runtime_minutes",
            "runtime_min",
        ],
    }

    # ========================================================
    # LOOP OVER COMPLETE 12-MODEL SWEEP
    # ========================================================

    for A in [1, 2]:

        for L in range(1, 7):

            case_name = (
                f"A{A}_L{L}_seed{seed}"
            )

            case_dir = (
                phase2_root
                /
                case_name
            )

            print()
            print("-" * 110)
            print(case_name)
            print("-" * 110)

            required = [
                case_dir / "validation.json",
                case_dir / "existing_result_summary_snapshot.csv",
                case_dir / "local_fourier_stage_summary.csv",
                case_dir / "pairwise_fidelity_summary.csv",
                case_dir / "block_head_diagnostics.csv",
                case_dir / "stage_metadata.csv",
            ]

            for p in required:
                require_file(p)

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            with open(
                case_dir / "validation.json"
            ) as f:
                validation = json.load(f)

            if validation.get("status") != "PASS":
                raise RuntimeError(
                    f"{case_name} did not pass validation."
                )

            # ------------------------------------------------
            # ORIGINAL FORECAST RESULT SUMMARY
            # ------------------------------------------------

            existing_df = pd.read_csv(
                case_dir
                /
                "existing_result_summary_snapshot.csv"
            )

            if len(existing_df) != 1:
                raise RuntimeError(
                    f"Expected one summary row for {case_name}, "
                    f"found {len(existing_df)}."
                )

            existing = (
                existing_df
                .iloc[0]
                .to_dict()
            )

            # ------------------------------------------------
            # FINAL LOCAL-QUANTUM DIAGNOSTICS
            # ------------------------------------------------

            local_stage = pd.read_csv(
                case_dir
                /
                "local_fourier_stage_summary.csv"
            )

            final_stage_index = int(
                local_stage[
                    "stage_index"
                ].max()
            )

            final_local = (
                local_stage[
                    local_stage[
                        "stage_index"
                    ]
                    ==
                    final_stage_index
                ]
                .iloc[-1]
            )

            # Preserve full stagewise table.
            stage_fourier_all.append(
                add_metadata(
                    local_stage,
                    A,
                    L,
                    seed,
                )
            )

            # ------------------------------------------------
            # FINAL PAIRWISE FIDELITIES
            # ------------------------------------------------

            fidelity = pd.read_csv(
                case_dir
                /
                "pairwise_fidelity_summary.csv"
            )

            fidelity_all.append(
                add_metadata(
                    fidelity,
                    A,
                    L,
                    seed,
                )
            )

            final_fidelity = (
                fidelity[
                    fidelity[
                        "stage_index"
                    ]
                    ==
                    fidelity[
                        "stage_index"
                    ].max()
                ]
                .copy()
            )

            # ------------------------------------------------
            # KAPPA / REUPLOAD CONTRACTION
            # ------------------------------------------------

            kappa_path = (
                case_dir
                /
                "trained_reupload_kappa_summary.csv"
            )

            if kappa_path.exists() and kappa_path.stat().st_size > 0:

                try:
                    kappa = pd.read_csv(
                        kappa_path
                    )
                except pd.errors.EmptyDataError:
                    kappa = pd.DataFrame()

            else:

                kappa = pd.DataFrame()

            if len(kappa) > 0:
                kappa_all.append(
                    add_metadata(
                        kappa,
                        A,
                        L,
                        seed,
                    )
                )

            # ------------------------------------------------
            # FINAL CLASSICAL-HEAD DIAGNOSTICS
            # ------------------------------------------------

            head = pd.read_csv(
                case_dir
                /
                "block_head_diagnostics.csv"
            )

            head_all.append(
                add_metadata(
                    head,
                    A,
                    L,
                    seed,
                )
            )

            final_pre = (
                head[
                    (head["block"] == L)
                    &
                    (head["signal"] == "pre_tanh")
                ]
                .iloc[-1]
            )

            final_post = (
                head[
                    (head["block"] == L)
                    &
                    (head["signal"] == "post_tanh")
                ]
                .iloc[-1]
            )

            # ------------------------------------------------
            # STAGE METADATA
            # ------------------------------------------------

            stage_meta = pd.read_csv(
                case_dir
                /
                "stage_metadata.csv"
            )

            stage_metadata_all.append(
                add_metadata(
                    stage_meta,
                    A,
                    L,
                    seed,
                )
            )

            # ------------------------------------------------
            # MASTER ROW
            # ------------------------------------------------

            row = {
                "ansatz_layers_A": A,
                "depth_L": L,
                "seed": seed,

                # Exact parameter-count formula:
                #
                # q_params = 42*A*L
                # classical head = 14 weights + 1 bias
                "parameter_count":
                    42 * A * L + 15,

                "best_epoch":
                    validation.get(
                        "best_epoch",
                        np.nan,
                    ),

                "max_final_prediction_difference":
                    validation.get(
                        "max_final_prediction_difference",
                        np.nan,
                    ),

                "max_state_norm_error":
                    validation.get(
                        "max_state_norm_error",
                        np.nan,
                    ),

                "max_ansatz_fidelity_invariance_error":
                    validation.get(
                        "max_ansatz_fidelity_invariance_error",
                        np.nan,
                    ),
            }

            # ------------------------------------------------
            # Preserve EVERY original result-summary column.
            # Prefix prevents naming collisions.
            # ------------------------------------------------

            for key, value in existing.items():
                row[
                    f"existing__{key}"
                ] = value

            # ------------------------------------------------
            # Standardized forecast metrics.
            # ------------------------------------------------

            for standard_name, aliases in FORECAST_ALIASES.items():

                value, source = (
                    first_existing_numeric(
                        existing,
                        aliases,
                    )
                )

                row[standard_name] = value

                if source is not None:
                    forecast_source_map.setdefault(
                        standard_name,
                        set(),
                    ).add(source)

            # ------------------------------------------------
            # Final local-quantum fields.
            # ------------------------------------------------

            LOCAL_FIELDS = [
                "variance_X",
                "variance_Y",
                "variance_Z",
                "variance_XYZ",

                "seasonal_power_XYZ",
                "higher_power_XYZ_n2_n6",

                "seasonal_fraction_XYZ",
                "higher_fraction_XYZ_n2_n6",

                "seasonal_to_higher_ratio_XYZ",

                "seasonal_axis_comp_X",
                "seasonal_axis_comp_Y",
                "seasonal_axis_comp_Z",

                "seasonal_power_Z",
                "seasonal_fraction_Z",
            ]

            for col in LOCAL_FIELDS:

                row[
                    f"final_local__{col}"
                ] = numeric_or_nan(
                    final_local.get(
                        col,
                        np.nan,
                    )
                )

            # Explicit readout-accessibility fraction.
            row[
                "final_local__seasonal_Z_accessibility"
            ] = row[
                "final_local__seasonal_axis_comp_Z"
            ]

            # ------------------------------------------------
            # Final fidelity fields at tau = 1,6,12,24.
            # ------------------------------------------------

            for tau in TAUS:

                frow = final_fidelity[
                    final_fidelity[
                        "tau"
                    ]
                    ==
                    tau
                ]

                if len(frow) != 1:
                    raise RuntimeError(
                        f"Expected one final fidelity row for "
                        f"{case_name}, tau={tau}; "
                        f"found {len(frow)}."
                    )

                frow = frow.iloc[0]

                for stat in [
                    "mean",
                    "median",
                    "std",
                    "q05",
                    "q25",
                    "q75",
                    "q95",
                    "min",
                    "max",
                ]:

                    col = (
                        f"fidelity_{stat}"
                    )

                    row[
                        f"final_fidelity_tau{tau}__{stat}"
                    ] = numeric_or_nan(
                        frow.get(
                            col,
                            np.nan,
                        )
                    )

                row[
                    f"final_fidelity_tau{tau}__mean_contrast_from_haar"
                ] = (
                    row[
                        f"final_fidelity_tau{tau}__mean"
                    ]
                    -
                    1.0 / 16384.0
                )

                row[
                    f"final_fidelity_tau{tau}__median_contrast_from_haar"
                ] = (
                    row[
                        f"final_fidelity_tau{tau}__median"
                    ]
                    -
                    1.0 / 16384.0
                )

            # ------------------------------------------------
            # Final-head fields.
            # ------------------------------------------------

            HEAD_FIELDS = [
                "rmse_scaled",
                "rmse_original",
                "pearson_with_train_target",
                "std_sample",
                "target_std_sample",
                "amplitude_ratio",
                "variance",
                "seasonal_power",
                "higher_power_n2_n6",
                "seasonal_fraction",
                "higher_fraction_n2_n6",
                "seasonal_to_higher_ratio",
            ]

            for col in HEAD_FIELDS:

                row[
                    f"final_head_pre__{col}"
                ] = numeric_or_nan(
                    final_pre.get(
                        col,
                        np.nan,
                    )
                )

                row[
                    f"final_head_post__{col}"
                ] = numeric_or_nan(
                    final_post.get(
                        col,
                        np.nan,
                    )
                )

            # ------------------------------------------------
            # Reupload contraction summaries.
            #
            # No half-life calculation yet.
            # We preserve the relevant ingredients for later.
            # ------------------------------------------------

            if len(kappa) > 0:

                for tau in TAUS:

                    kt = (
                        kappa[
                            kappa["tau"]
                            ==
                            tau
                        ]
                        .sort_values(
                            "reupload_layer"
                        )
                    )

                    if len(kt) == 0:
                        continue

                    last = kt.iloc[-1]

                    row[
                        f"last_reupload_tau{tau}__layer"
                    ] = numeric_or_nan(
                        last.get(
                            "reupload_layer",
                            np.nan,
                        )
                    )

                    row[
                        f"last_reupload_tau{tau}__kappa_median"
                    ] = numeric_or_nan(
                        last.get(
                            "kappa_trained_median",
                            np.nan,
                        )
                    )

                    row[
                        f"last_reupload_tau{tau}__kappa_mean"
                    ] = numeric_or_nan(
                        last.get(
                            "kappa_trained_mean",
                            np.nan,
                        )
                    )

                    row[
                        f"last_reupload_tau{tau}__beta_haar_median"
                    ] = numeric_or_nan(
                        last.get(
                            "beta_haar_median",
                            np.nan,
                        )
                    )

                    row[
                        f"last_reupload_tau{tau}__beta_haar_mean"
                    ] = numeric_or_nan(
                        last.get(
                            "beta_haar_mean",
                            np.nan,
                        )
                    )

                    row[
                        f"last_reupload_tau{tau}__valid_fraction"
                    ] = numeric_or_nan(
                        last.get(
                            "valid_fraction",
                            np.nan,
                        )
                    )

                    row[
                        f"last_reupload_tau{tau}__mean_abs_kappa_minus_beta"
                    ] = numeric_or_nan(
                        last.get(
                            "mean_abs_kappa_minus_beta",
                            np.nan,
                        )
                    )

                    # Robust across-reupload summary for this
                    # trained model.
                    kappas = pd.to_numeric(
                        kt[
                            "kappa_trained_median"
                        ],
                        errors="coerce",
                    )

                    row[
                        f"across_reuploads_tau{tau}__median_of_kappa_medians"
                    ] = (
                        float(
                            np.nanmedian(
                                kappas
                            )
                        )
                        if kappas.notna().sum() > 0
                        else np.nan
                    )

            master_rows.append(row)

            print(
                f"PASS: A={A}, L={L}, "
                f"final stage={final_stage_index}"
            )

    # ========================================================
    # MASTER TABLE
    # ========================================================

    master = pd.DataFrame(
        master_rows
    ).sort_values(
        [
            "ansatz_layers_A",
            "depth_L",
        ]
    )

    master.to_csv(
        output_dir
        /
        "phase2_seed42_master_table.csv",
        index=False,
    )

    # ========================================================
    # COMBINED LONG TABLES
    # ========================================================

    pd.concat(
        stage_fourier_all,
        ignore_index=True,
    ).to_csv(
        output_dir
        /
        "phase2_all_stage_fourier.csv",
        index=False,
    )

    pd.concat(
        fidelity_all,
        ignore_index=True,
    ).to_csv(
        output_dir
        /
        "phase2_all_fidelity_summary.csv",
        index=False,
    )

    if len(kappa_all) > 0:

        pd.concat(
            kappa_all,
            ignore_index=True,
        ).to_csv(
            output_dir
            /
            "phase2_all_kappa_summary.csv",
            index=False,
        )

    pd.concat(
        head_all,
        ignore_index=True,
    ).to_csv(
        output_dir
        /
        "phase2_all_head_diagnostics.csv",
        index=False,
    )

    pd.concat(
        stage_metadata_all,
        ignore_index=True,
    ).to_csv(
        output_dir
        /
        "phase2_all_stage_metadata.csv",
        index=False,
    )

    # ========================================================
    # PHASE-I HAAR REFERENCE
    # ========================================================

    phase1 = pd.read_csv(
        args.phase1_kernel_summary
    )

    phase1_reference = (
        phase1[
            (phase1["encoding"] == "original_ry")
            &
            (phase1["tau"].isin(TAUS))
        ]
        .copy()
        .sort_values("tau")
    )

    phase1_reference.to_csv(
        output_dir
        /
        "phase1_original_ry_haar_reference.csv",
        index=False,
    )

    # ========================================================
    # MATCHED A1-vs-A2 TABLE
    # ========================================================

    CORE_COMPARE_METRICS = [

        "parameter_count",

        "forecast_test_rmse",
        "forecast_test_pearson",
        "forecast_test_kge",
        "forecast_fft_cosine",
        "forecast_amplitude_ratio",

        "final_local__seasonal_fraction_XYZ",
        "final_local__higher_fraction_XYZ_n2_n6",
        "final_local__seasonal_to_higher_ratio_XYZ",

        "final_local__seasonal_axis_comp_X",
        "final_local__seasonal_axis_comp_Y",
        "final_local__seasonal_axis_comp_Z",

        "final_fidelity_tau12__median",
        "final_fidelity_tau12__mean",

        "last_reupload_tau12__kappa_median",

        "final_head_post__amplitude_ratio",
        "final_head_post__pearson_with_train_target",
        "final_head_post__seasonal_fraction",
        "final_head_post__higher_fraction_n2_n6",
        "final_head_post__seasonal_to_higher_ratio",
    ]

    matched_rows = []

    for L in range(1, 7):

        a1 = (
            master[
                (master["ansatz_layers_A"] == 1)
                &
                (master["depth_L"] == L)
            ]
            .iloc[0]
        )

        a2 = (
            master[
                (master["ansatz_layers_A"] == 2)
                &
                (master["depth_L"] == L)
            ]
            .iloc[0]
        )

        row = {
            "depth_L": L,
        }

        for metric in CORE_COMPARE_METRICS:

            if metric not in master.columns:
                continue

            v1 = numeric_or_nan(
                a1.get(
                    metric,
                    np.nan,
                )
            )

            v2 = numeric_or_nan(
                a2.get(
                    metric,
                    np.nan,
                )
            )

            row[
                f"A1__{metric}"
            ] = v1

            row[
                f"A2__{metric}"
            ] = v2

            row[
                f"delta_A2_minus_A1__{metric}"
            ] = (
                v2 - v1
                if np.isfinite(v1)
                and np.isfinite(v2)
                else np.nan
            )

        matched_rows.append(row)

    matched = pd.DataFrame(
        matched_rows
    )

    matched.to_csv(
        output_dir
        /
        "phase2_matched_A1_A2_by_depth.csv",
        index=False,
    )

    # ========================================================
    # DEPTH-TO-DEPTH DIFFERENCES
    #
    # These are differences ACROSS independently trained models,
    # not within-circuit trajectories.
    # ========================================================

    DEPTH_METRICS = [
        "forecast_test_rmse",
        "forecast_test_pearson",
        "forecast_test_kge",
        "forecast_fft_cosine",
        "forecast_amplitude_ratio",

        "final_local__seasonal_fraction_XYZ",
        "final_local__seasonal_axis_comp_X",
        "final_local__seasonal_axis_comp_Z",
        "final_local__seasonal_to_higher_ratio_XYZ",

        "final_fidelity_tau12__median",

        "final_head_post__amplitude_ratio",
        "final_head_post__pearson_with_train_target",
        "final_head_post__seasonal_fraction",
    ]

    transition_rows = []

    for A in [1, 2]:

        g = (
            master[
                master["ansatz_layers_A"]
                ==
                A
            ]
            .sort_values(
                "depth_L"
            )
        )

        for L in range(2, 7):

            previous = (
                g[
                    g["depth_L"]
                    ==
                    L - 1
                ]
                .iloc[0]
            )

            current = (
                g[
                    g["depth_L"]
                    ==
                    L
                ]
                .iloc[0]
            )

            row = {
                "ansatz_layers_A": A,
                "from_depth": L - 1,
                "to_depth": L,
            }

            for metric in DEPTH_METRICS:

                if metric not in master.columns:
                    continue

                v0 = numeric_or_nan(
                    previous.get(
                        metric,
                        np.nan,
                    )
                )

                v1 = numeric_or_nan(
                    current.get(
                        metric,
                        np.nan,
                    )
                )

                row[
                    f"delta__{metric}"
                ] = (
                    v1 - v0
                    if np.isfinite(v0)
                    and np.isfinite(v1)
                    else np.nan
                )

            transition_rows.append(row)

    pd.DataFrame(
        transition_rows
    ).to_csv(
        output_dir
        /
        "phase2_model_depth_transition_differences.csv",
        index=False,
    )

    # ========================================================
    # EXPLORATORY INTERNAL-vs-FORECAST CORRELATIONS
    #
    # n = 12 models only.
    # These are descriptive/exploratory, not inferential.
    # ========================================================

    forecast_metrics = [
        "forecast_test_rmse",
        "forecast_test_pearson",
        "forecast_test_kge",
        "forecast_fft_cosine",
        "forecast_amplitude_ratio",
    ]

    internal_metrics = [
        "final_local__seasonal_fraction_XYZ",
        "final_local__higher_fraction_XYZ_n2_n6",
        "final_local__seasonal_to_higher_ratio_XYZ",

        "final_local__seasonal_axis_comp_X",
        "final_local__seasonal_axis_comp_Y",
        "final_local__seasonal_axis_comp_Z",

        "final_fidelity_tau12__median",
        "last_reupload_tau12__kappa_median",

        "final_head_post__amplitude_ratio",
        "final_head_post__seasonal_fraction",
        "final_head_post__seasonal_to_higher_ratio",
    ]

    correlation_rows = []

    for fm in forecast_metrics:

        if fm not in master.columns:
            continue

        for im in internal_metrics:

            if im not in master.columns:
                continue

            pair = master[
                [fm, im]
            ].apply(
                pd.to_numeric,
                errors="coerce",
            ).dropna()

            if len(pair) < 4:
                continue

            if (
                pair[fm].std() == 0
                or
                pair[im].std() == 0
            ):
                continue

            correlation_rows.append(
                {
                    "forecast_metric": fm,
                    "internal_metric": im,
                    "n_models": len(pair),

                    "pearson":
                        float(
                            pair[
                                [fm, im]
                            ]
                            .corr(
                                method="pearson"
                            )
                            .iloc[0, 1]
                        ),

                    "spearman":
                        float(
                            pair[
                                [fm, im]
                            ]
                            .corr(
                                method="spearman"
                            )
                            .iloc[0, 1]
                        ),

                    "interpretation":
                        "exploratory_seed42_only",
                }
            )

    corr_df = pd.DataFrame(
        correlation_rows
    )

    corr_df.to_csv(
        output_dir
        /
        "phase2_exploratory_internal_forecast_correlations.csv",
        index=False,
    )

    # ========================================================
    # RECORD WHICH ORIGINAL COLUMNS WERE USED
    # ========================================================

    forecast_source_json = {
        key: sorted(list(values))
        for key, values
        in forecast_source_map.items()
    }

    with open(
        output_dir
        /
        "detected_forecast_metric_sources.json",
        "w",
    ) as f:

        json.dump(
            forecast_source_json,
            f,
            indent=2,
        )

    # ========================================================
    # FIGURES
    # ========================================================

    figures_created = []

    PLOTS = [

        (
            "final_local__seasonal_fraction_XYZ",
            "Period-12 power / local XYZ variance",
            "Final local period-12 fraction",
            "01_final_local_seasonal_fraction_XYZ_vs_depth.png",
        ),

        (
            "final_local__seasonal_axis_comp_Z",
            "Fraction of local seasonal power in Z",
            "Final seasonal Z-readout accessibility",
            "02_final_seasonal_Z_accessibility_vs_depth.png",
        ),

        (
            "final_local__seasonal_axis_comp_X",
            "Fraction of local seasonal power in X",
            "Final seasonal X-sector allocation",
            "03_final_seasonal_X_allocation_vs_depth.png",
        ),

        (
            "final_local__seasonal_to_higher_ratio_XYZ",
            "Period-12 / harmonics n=2..6",
            "Final local seasonal-to-higher-harmonic ratio",
            "04_final_local_seasonal_to_higher_ratio_vs_depth.png",
        ),

        (
            "final_fidelity_tau12__median",
            "Median pairwise fidelity at tau=12",
            "Final seasonal pairwise fidelity",
            "05_final_fidelity_tau12_vs_depth.png",
        ),

        (
            "last_reupload_tau12__kappa_median",
            "Median trained kappa at final reupload",
            "Final trained seasonal-memory contraction",
            "06_last_reupload_kappa_tau12_vs_depth.png",
        ),

        (
            "final_head_post__amplitude_ratio",
            "Prediction std / target std",
            "Final post-tanh training amplitude ratio",
            "07_final_head_amplitude_ratio_vs_depth.png",
        ),

        (
            "final_head_post__pearson_with_train_target",
            "Pearson correlation",
            "Final post-tanh training correlation",
            "08_final_head_train_pearson_vs_depth.png",
        ),

        (
            "final_head_post__seasonal_fraction",
            "Period-12 power / output variance",
            "Final post-tanh seasonal fraction",
            "09_final_head_seasonal_fraction_vs_depth.png",
        ),

        (
            "forecast_test_rmse",
            "Test RMSE",
            "Existing test RMSE vs depth",
            "10_existing_test_RMSE_vs_depth.png",
        ),

        (
            "forecast_test_pearson",
            "Test Pearson correlation",
            "Existing test Pearson correlation vs depth",
            "11_existing_test_Pearson_vs_depth.png",
        ),

        (
            "forecast_fft_cosine",
            "FFT cosine similarity",
            "Existing FFT cosine similarity vs depth",
            "12_existing_FFT_cosine_vs_depth.png",
        ),

        (
            "forecast_amplitude_ratio",
            "Prediction std / target std",
            "Existing test amplitude ratio vs depth",
            "13_existing_test_amplitude_ratio_vs_depth.png",
        ),
    ]

    for metric, ylabel, title, filename in PLOTS:

        created = save_depth_plot(
            master=master,
            metric=metric,
            ylabel=ylabel,
            title=title,
            output_path=(
                output_dir
                /
                filename
            ),
        )

        if created:
            figures_created.append(filename)

    # ========================================================
    # COMPACT SUMMARY JSON
    # ========================================================

    summary = {
        "seed": seed,
        "n_models": int(len(master)),
        "A_values": [1, 2],
        "depth_values": [1, 2, 3, 4, 5, 6],
        "all_validation_passed": True,
        "half_life_analysis_performed": False,
        "figures_created": figures_created,
        "detected_forecast_metric_sources":
            forecast_source_json,
    }

    with open(
        output_dir
        /
        "phase2_seed42_aggregate_summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print()
    print("=" * 110)
    print("CORE DEPTH-SWEEP SUMMARY")
    print("=" * 110)

    display_cols = [
        "ansatz_layers_A",
        "depth_L",
        "parameter_count",

        "forecast_test_rmse",
        "forecast_test_pearson",
        "forecast_test_kge",
        "forecast_fft_cosine",
        "forecast_amplitude_ratio",

        "final_local__seasonal_fraction_XYZ",
        "final_local__seasonal_axis_comp_X",
        "final_local__seasonal_axis_comp_Z",
        "final_local__seasonal_to_higher_ratio_XYZ",

        "final_fidelity_tau12__median",

        "last_reupload_tau12__kappa_median",

        "final_head_post__amplitude_ratio",
        "final_head_post__pearson_with_train_target",
        "final_head_post__seasonal_fraction",
    ]

    display_cols = [
        c
        for c in display_cols
        if c in master.columns
    ]

    print(
        master[
            display_cols
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 110)
    print("A=2 CRITICAL L2 -> L3 TRANSITION")
    print("=" * 110)

    a2_l2 = master[
        (master["ansatz_layers_A"] == 2)
        &
        (master["depth_L"] == 2)
    ].iloc[0]

    a2_l3 = master[
        (master["ansatz_layers_A"] == 2)
        &
        (master["depth_L"] == 3)
    ].iloc[0]

    transition_metrics = [
        "forecast_test_rmse",
        "forecast_test_pearson",
        "forecast_test_kge",
        "forecast_fft_cosine",
        "forecast_amplitude_ratio",

        "final_local__seasonal_fraction_XYZ",
        "final_local__seasonal_axis_comp_X",
        "final_local__seasonal_axis_comp_Z",
        "final_local__seasonal_to_higher_ratio_XYZ",

        "final_fidelity_tau12__median",

        "last_reupload_tau12__kappa_median",

        "final_head_post__amplitude_ratio",
        "final_head_post__pearson_with_train_target",
        "final_head_post__seasonal_fraction",
    ]

    transition_print_rows = []

    for metric in transition_metrics:

        if metric not in master.columns:
            continue

        v2 = numeric_or_nan(
            a2_l2.get(
                metric,
                np.nan,
            )
        )

        v3 = numeric_or_nan(
            a2_l3.get(
                metric,
                np.nan,
            )
        )

        transition_print_rows.append(
            {
                "metric": metric,
                "A2_L2": v2,
                "A2_L3": v3,
                "delta_L3_minus_L2":
                    (
                        v3 - v2
                        if np.isfinite(v2)
                        and np.isfinite(v3)
                        else np.nan
                    ),
            }
        )

    print(
        pd.DataFrame(
            transition_print_rows
        ).to_string(
            index=False
        )
    )

    print()
    print("=" * 110)
    print("OUTPUT FILES")
    print("=" * 110)

    for p in sorted(
        output_dir.iterdir()
    ):
        print(p.name)

    print()
    print("STATUS: PASS")


if __name__ == "__main__":
    main()
