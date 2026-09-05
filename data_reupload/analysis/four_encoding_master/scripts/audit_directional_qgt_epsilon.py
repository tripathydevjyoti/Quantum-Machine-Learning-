#!/usr/bin/env python

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(
    "/umbc/rs/pi_deffner/users/devjyot1/projects/"
    "Quantum-Machine-Learning-"
)

SCRIPT_DIR = (
    PROJECT_ROOT
    / "data_reupload/analysis/four_encoding_master/scripts"
)

sys.path.insert(
    0,
    str(SCRIPT_DIR),
)

import compute_exact_temporal_fidelity as g1


MANIFEST_PATH = (
    PROJECT_ROOT
    / "data_reupload/analysis/four_encoding_master/"
    "output/tables/master_run_manifest.csv"
)

OUTDIR = (
    PROJECT_ROOT
    / "data_reupload/analysis/four_encoding_master/"
    "output/geometry/directional_qgt_epsilon_audit"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


ENCODINGS = [
    "ry",
    "same_scalar_rxryrz",
    "seasonal_meridian",
    "learnable_seasonal_cdf",
]

EPSILONS = [
    1e-2,
    3e-3,
    1e-3,
    3e-4,
]

TAU = 12


def perturb_input(
    encoding,
    x,
    delta,
    epsilon,
):

    out = np.array(
        x,
        dtype=np.float64,
        copy=True,
    )

    if encoding in {
        "ry",
        "same_scalar_rxryrz",
    }:

        out = (
            out
            +
            epsilon
            *
            delta
        )

    elif encoding in {
        "seasonal_meridian",
        "learnable_seasonal_cdf",
    }:

        out[:, 0] = (
            out[:, 0]
            +
            epsilon
            *
            delta
        )

    else:
        raise ValueError(
            encoding
        )

    return out


def continuous_delta(
    encoding,
    x0,
    x1,
):

    if encoding in {
        "ry",
        "same_scalar_rxryrz",
    }:

        return (
            np.asarray(x1)
            -
            np.asarray(x0)
        )

    if encoding == "seasonal_meridian":

        # tau=12 must preserve the seasonal phase exactly.
        if not np.allclose(
            x0[:, 1],
            x1[:, 1],
            atol=1e-12,
            rtol=0,
        ):
            raise RuntimeError(
                "Seasonal phase changed at tau=12."
            )

        return (
            x1[:, 0]
            -
            x0[:, 0]
        )

    if encoding == "learnable_seasonal_cdf":

        # tau=12 must select the same month-specific
        # learned parameters at corresponding lag positions.
        if not np.array_equal(
            np.rint(
                x0[:, 1]
            ).astype(int),
            np.rint(
                x1[:, 1]
            ).astype(int),
        ):
            raise RuntimeError(
                "Seasonal index changed at tau=12."
            )

        return (
            x1[:, 0]
            -
            x0[:, 0]
        )

    raise ValueError(
        encoding
    )


manifest = (
    pd.read_csv(
        MANIFEST_PATH
    )
    .sort_values(
        [
            "encoding_order",
            "depth",
            "seed",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# Use deepest seed-42 model from every encoding.
# This is deliberately the numerically hardest representative.
# ============================================================

selected = manifest[
    (manifest["depth"] == 6)
    &
    (manifest["seed"] == 42)
].copy()

if len(selected) != 4:
    raise RuntimeError(
        f"Expected four representative runs, got {len(selected)}"
    )


# Evenly spaced seasonal pairs across the test set.
all_pair_indices = np.arange(
    0,
    105 - TAU,
    dtype=int,
)

sample_pair_indices = np.unique(
    np.rint(
        np.linspace(
            0,
            len(all_pair_indices) - 1,
            16,
        )
    ).astype(int)
)

sample_pair_indices = (
    all_pair_indices[
        sample_pair_indices
    ]
)


rows = []


print("=" * 120)
print("G2.0 DIRECTIONAL QGT FINITE-DIFFERENCE AUDIT")
print("=" * 120)

print()
print(
    "tau:",
    TAU,
)

print(
    "number of sampled seasonal pairs:",
    len(sample_pair_indices),
)

print(
    "epsilons:",
    EPSILONS,
)


for encoding in ENCODINGS:

    row = selected[
        selected["encoding"]
        ==
        encoding
    ].iloc[0]

    run_dir = Path(
        row["run_dir"]
    )

    print()
    print("-" * 120)
    print(
        f"{encoding} | L=6 | seed=42"
    )
    print("-" * 120)

    with open(
        run_dir / "config.json"
    ) as f:
        config = json.load(f)

    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    y_original = g1.load_original_series(
        config
    )

    (
        X_test,
        source_indices,
        target_indices,
    ) = g1.construct_test_inputs(
        encoding,
        y_original,
        config,
    )

    state_circuit = g1.build_state_qnode(
        encoding,
        config,
        state_dict,
    )

    for pair_index in sample_pair_indices:

        x0 = np.asarray(
            X_test[
                pair_index
            ],
            dtype=np.float64,
        )

        x1 = np.asarray(
            X_test[
                pair_index + TAU
            ],
            dtype=np.float64,
        )

        delta = continuous_delta(
            encoding,
            x0,
            x1,
        )

        delta_norm = float(
            np.linalg.norm(
                delta
            )
        )

        psi0 = np.asarray(
            state_circuit(
                x0
            ),
            dtype=np.complex128,
        )

        psi1 = np.asarray(
            state_circuit(
                x1
            ),
            dtype=np.complex128,
        )

        exact_fidelity = float(
            np.abs(
                np.vdot(
                    psi0,
                    psi1,
                )
            )
            **
            2
        )

        exact_infidelity = (
            1.0
            -
            exact_fidelity
        )

        for epsilon in EPSILONS:

            x_plus = perturb_input(
                encoding,
                x0,
                delta,
                epsilon,
            )

            x_minus = perturb_input(
                encoding,
                x0,
                delta,
                -epsilon,
            )

            psi_plus = np.asarray(
                state_circuit(
                    x_plus
                ),
                dtype=np.complex128,
            )

            psi_minus = np.asarray(
                state_circuit(
                    x_minus
                ),
                dtype=np.complex128,
            )

            # --------------------------------------------
            # Central state derivative
            # --------------------------------------------

            dpsi = (
                psi_plus
                -
                psi_minus
            ) / (
                2.0
                *
                epsilon
            )

            derivative_norm = float(
                np.vdot(
                    dpsi,
                    dpsi,
                ).real
            )

            gauge_component = float(
                np.abs(
                    np.vdot(
                        psi0,
                        dpsi,
                    )
                )
                **
                2
            )

            D_derivative = (
                derivative_norm
                -
                gauge_component
            )

            # Numerical roundoff can create tiny negative values.
            if (
                D_derivative < 0
                and
                abs(D_derivative)
                <
                1e-10
            ):
                D_derivative = 0.0

            # --------------------------------------------
            # Fidelity curvature
            # --------------------------------------------

            F_plus = float(
                np.abs(
                    np.vdot(
                        psi0,
                        psi_plus,
                    )
                )
                **
                2
            )

            F_minus = float(
                np.abs(
                    np.vdot(
                        psi0,
                        psi_minus,
                    )
                )
                **
                2
            )

            D_curvature = (
                (
                    1.0 - F_plus
                )
                +
                (
                    1.0 - F_minus
                )
            ) / (
                2.0
                *
                epsilon
                **
                2
            )

            rel_method_difference = (
                abs(
                    D_derivative
                    -
                    D_curvature
                )
                /
                max(
                    abs(
                        D_derivative
                    ),
                    1e-12,
                )
            )

            rows.append({
                "encoding":
                    encoding,

                "depth":
                    6,

                "seed":
                    42,

                "pair_index":
                    int(
                        pair_index
                    ),

                "target_index_t":
                    int(
                        target_indices[
                            pair_index
                        ]
                    ),

                "target_index_t_plus_12":
                    int(
                        target_indices[
                            pair_index
                            +
                            TAU
                        ]
                    ),

                "epsilon":
                    float(
                        epsilon
                    ),

                "delta_norm":
                    delta_norm,

                "exact_fidelity_tau12":
                    exact_fidelity,

                "exact_infidelity_tau12":
                    exact_infidelity,

                "D_state_derivative":
                    float(
                        D_derivative
                    ),

                "D_fidelity_curvature":
                    float(
                        D_curvature
                    ),

                "relative_method_difference":
                    float(
                        rel_method_difference
                    ),
            })


detail = pd.DataFrame(
    rows
)

detail.to_csv(
    OUTDIR
    / "g2_epsilon_audit_pair_level.csv",
    index=False,
)


# ============================================================
# Aggregate
# ============================================================

summary = (
    detail
    .groupby(
        [
            "encoding",
            "epsilon",
        ],
        as_index=False,
    )
    .agg(
        n_pairs=(
            "pair_index",
            "nunique",
        ),

        mean_delta_norm=(
            "delta_norm",
            "mean",
        ),

        mean_D_derivative=(
            "D_state_derivative",
            "mean",
        ),

        median_D_derivative=(
            "D_state_derivative",
            "median",
        ),

        mean_D_curvature=(
            "D_fidelity_curvature",
            "mean",
        ),

        mean_relative_method_difference=(
            "relative_method_difference",
            "mean",
        ),

        max_relative_method_difference=(
            "relative_method_difference",
            "max",
        ),
    )
)


# Compare each epsilon with the smallest epsilon estimate.
reference_eps = min(
    EPSILONS
)

reference_values = (
    summary[
        summary["epsilon"]
        ==
        reference_eps
    ][
        [
            "encoding",
            "mean_D_derivative",
        ]
    ]
    .rename(
        columns={
            "mean_D_derivative":
                "reference_mean_D"
        }
    )
)

summary = summary.merge(
    reference_values,
    on="encoding",
    how="left",
    validate="many_to_one",
)

summary[
    "relative_change_vs_smallest_epsilon"
] = (
    np.abs(
        summary[
            "mean_D_derivative"
        ]
        -
        summary[
            "reference_mean_D"
        ]
    )
    /
    np.maximum(
        np.abs(
            summary[
                "reference_mean_D"
            ]
        ),
        1e-12,
    )
)


summary.to_csv(
    OUTDIR
    / "g2_epsilon_audit_summary.csv",
    index=False,
)


# ============================================================
# Print
# ============================================================

for encoding in ENCODINGS:

    print()
    print("=" * 120)
    print(
        encoding
    )
    print("=" * 120)

    d = (
        summary[
            summary["encoding"]
            ==
            encoding
        ]
        .sort_values(
            "epsilon",
            ascending=False,
        )
    )

    for _, row in d.iterrows():

        print(
            f"eps={row['epsilon']:.1e}  "
            f"D_deriv={row['mean_D_derivative']:.8e}  "
            f"D_curv={row['mean_D_curvature']:.8e}  "
            f"method_rel_diff="
            f"{row['mean_relative_method_difference']:.3e}  "
            f"change_vs_eps3e-4="
            f"{row['relative_change_vs_smallest_epsilon']:.3e}"
        )


print()
print("=" * 120)
print("G2.0 EPSILON AUDIT STATUS: PASS")
print("=" * 120)
