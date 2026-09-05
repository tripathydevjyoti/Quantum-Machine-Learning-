#!/usr/bin/env python

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(
    "/umbc/rs/pi_deffner/users/devjyot1/projects/"
    "Quantum-Machine-Learning-"
)

ANALYSIS_ROOT = (
    PROJECT_ROOT
    / "data_reupload"
    / "analysis"
    / "four_encoding_master"
)

TABLE_DIR = ANALYSIS_ROOT / "output" / "tables"

OUT_DIR = (
    ANALYSIS_ROOT
    / "output"
    / "bloch_distributions"
)

FIG_DIR = (
    ANALYSIS_ROOT
    / "output"
    / "figures"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = (
    PROJECT_ROOT
    / "data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv"
)

MANIFEST_PATH = (
    TABLE_DIR
    / "master_run_manifest.csv"
)

DEPTH = 2
SEED = 42

# First data-upload block of the trained depth-2 models.
LAYER = 0

WINDOW = 14

TRAIN_TARGET_START = 14
TRAIN_TARGET_STOP_EXCLUSIVE = 364

# Training windows target 14,...,363.
# Their source-window union is observation indices 0,...,362.
UNIQUE_SOURCE_INDICES = np.arange(
    0,
    TRAIN_TARGET_STOP_EXCLUSIVE - 1,
    dtype=np.int64,
)

# The empirical CDF used by the seasonal encodings was fitted to
# observations 0,...,363 inclusive.
CDF_REFERENCE_INDICES = np.arange(
    0,
    TRAIN_TARGET_STOP_EXCLUSIVE,
    dtype=np.int64,
)

ENCODINGS = [
    "ry",
    "same_scalar_rxryrz",
    "seasonal_meridian",
    "learnable_seasonal_cdf",
    "learnable_seasonal_cdf_rz",
]

LABELS = {
    "ry": r"Standard $R_y$",
    "same_scalar_rxryrz": r"Same-scalar $R_xR_yR_z$",
    "seasonal_meridian": "Fixed seasonal CDF",
    "learnable_seasonal_cdf": "Learnable polar CDF",
    "learnable_seasonal_cdf_rz":
        "Learnable polar-azimuth CDF",
}


# =============================================================================
# Basic transforms
# =============================================================================

def scale_fixed(
    y,
    data_min=0.0,
    data_max=350.0,
):

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    return (
        2.0
        *
        (y - data_min)
        /
        (data_max - data_min)
        -
        1.0
    )


def empirical_cdf_right(
    values,
    sorted_reference,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    sorted_reference = np.asarray(
        sorted_reference,
        dtype=np.float64,
    )

    counts = np.searchsorted(
        sorted_reference,
        values,
        side="right",
    )

    return (
        counts.astype(np.float64)
        /
        float(len(sorted_reference))
    )


# =============================================================================
# One-qubit rotations and Bloch conversion
# =============================================================================

def rx(theta):

    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)

    return np.array(
        [
            [c, -1j * s],
            [-1j * s, c],
        ],
        dtype=np.complex128,
    )


def ry(theta):

    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)

    return np.array(
        [
            [c, -s],
            [s, c],
        ],
        dtype=np.complex128,
    )


def rz(theta):

    return np.array(
        [
            [
                np.exp(-0.5j * theta),
                0.0,
            ],
            [
                0.0,
                np.exp(0.5j * theta),
            ],
        ],
        dtype=np.complex128,
    )


def state_to_bloch(psi):

    psi = np.asarray(
        psi,
        dtype=np.complex128,
    )

    a = psi[0]
    b = psi[1]

    bx = 2.0 * np.real(
        np.conjugate(a) * b
    )

    by = 2.0 * np.imag(
        np.conjugate(a) * b
    )

    bz = (
        np.abs(a) ** 2
        -
        np.abs(b) ** 2
    )

    return np.array(
        [bx, by, bz],
        dtype=np.float64,
    )


def bloch_rz_ry(theta, phi):

    # For Rz(phi) Ry(theta) |0>.
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=np.float64,
    )


# =============================================================================
# Dataset
# =============================================================================

def load_series():

    raw = pd.read_csv(
        DATA_PATH
    )

    if raw.shape[1] <= 4:
        raise RuntimeError(
            f"Expected >=5 columns, got {raw.shape[1]}"
        )

    value_column = raw.columns[4]

    y = (
        pd.to_numeric(
            raw.iloc[:, 4],
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=np.float64
        )
    )

    if len(y) != 539:
        raise RuntimeError(
            f"Expected 539 non-missing observations, got {len(y)}"
        )

    return y, value_column


# =============================================================================
# Manifest / checkpoint helpers
# =============================================================================

def get_run_dir(
    manifest,
    encoding,
    depth,
    seed,
):

    d = manifest[
        (manifest["encoding"] == encoding)
        &
        (manifest["depth"] == depth)
        &
        (manifest["seed"] == seed)
    ]

    if len(d) != 1:
        raise RuntimeError(
            f"Expected one manifest row for "
            f"{encoding}, depth={depth}, seed={seed}; "
            f"found {len(d)}"
        )

    run_dir = Path(
        d.iloc[0]["run_dir"]
    )

    if not run_dir.exists():
        raise FileNotFoundError(
            run_dir
        )

    return run_dir


def find_checkpoint(run_dir):

    candidates = [
        run_dir / "best_model.pt",
        run_dir / "best_checkpoint.pt",
        run_dir / "checkpoint_best.pt",
        run_dir / "checkpoint.pt",
    ]

    existing = [
        p for p in candidates
        if p.exists()
    ]

    if len(existing) == 1:
        return existing[0]

    if len(existing) > 1:
        # Prefer the canonical name if present.
        p = run_dir / "best_model.pt"

        if p.exists():
            return p

        raise RuntimeError(
            f"Multiple candidate checkpoints in {run_dir}: "
            f"{existing}"
        )

    # Conservative fallback: inspect .pt files but require uniqueness.
    pts = sorted(
        run_dir.glob("*.pt")
    )

    if len(pts) != 1:
        raise RuntimeError(
            f"Could not identify unique checkpoint in {run_dir}. "
            f"Found: {[p.name for p in pts]}"
        )

    return pts[0]


def extract_state_dict(obj):

    # Covers the common formats used by training scripts while
    # remaining explicit about what is accepted.

    if not isinstance(obj, dict):
        raise RuntimeError(
            f"Checkpoint object is {type(obj)}, expected dict."
        )

    if "model_state_dict" in obj:
        state_dict = obj[
            "model_state_dict"
        ]

    elif "state_dict" in obj:
        state_dict = obj[
            "state_dict"
        ]

    elif "q_weights" in obj:
        # Checkpoint itself is already the state dict.
        state_dict = obj

    else:
        # Some checkpoints may directly contain named model tensors
        # but q_weights may carry a module prefix.
        tensor_keys = [
            str(k)
            for k, v in obj.items()
            if torch.is_tensor(v)
        ]

        if any(
            k.endswith("q_weights")
            for k in tensor_keys
        ):
            state_dict = obj
        else:
            raise RuntimeError(
                "Could not identify model state_dict. "
                f"Top-level keys: {list(obj.keys())}"
            )

    return state_dict


def normalize_state_dict_keys(
    state_dict,
):

    # Return a dictionary where common module prefixes are stripped.
    clean = {}

    for key, value in state_dict.items():

        k = str(key)

        for prefix in [
            "module.",
            "model.",
        ]:
            if k.startswith(prefix):
                k = k[len(prefix):]

        clean[k] = value

    return clean


def load_encoder_raw_parameters(
    run_dir,
    encoding,
    depth,
):

    checkpoint_path = find_checkpoint(
        run_dir
    )

    obj = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = normalize_state_dict_keys(
        extract_state_dict(obj)
    )

    required = [
        "encoding_strength_raw",
        "encoding_shift_raw",
    ]

    if (
        encoding
        ==
        "learnable_seasonal_cdf_rz"
    ):
        required.append(
            "encoding_phase_raw"
        )

    missing = [
        k for k in required
        if k not in state_dict
    ]

    if missing:
        raise RuntimeError(
            f"Missing encoder parameters in {checkpoint_path}: "
            f"{missing}. Available keys include: "
            f"{sorted(state_dict.keys())}"
        )

    strength_raw = (
        state_dict[
            "encoding_strength_raw"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    shift_raw = (
        state_dict[
            "encoding_shift_raw"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    expected_shape = (
        12,
        depth,
    )

    if strength_raw.shape != expected_shape:
        raise RuntimeError(
            f"{encoding}: strength shape "
            f"{strength_raw.shape}; expected {expected_shape}"
        )

    if shift_raw.shape != expected_shape:
        raise RuntimeError(
            f"{encoding}: shift shape "
            f"{shift_raw.shape}; expected {expected_shape}"
        )

    result = {
        "checkpoint_path":
            checkpoint_path,

        "strength_raw":
            strength_raw,

        "shift_raw":
            shift_raw,

        "strength":
            1.0
            +
            np.tanh(
                strength_raw
            ),

        "shift":
            (
                np.pi
                /
                2.0
                *
                np.tanh(
                    shift_raw
                )
            ),
    }

    if (
        encoding
        ==
        "learnable_seasonal_cdf_rz"
    ):

        phase_raw = (
            state_dict[
                "encoding_phase_raw"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        if phase_raw.shape != expected_shape:
            raise RuntimeError(
                f"{encoding}: phase shape "
                f"{phase_raw.shape}; expected {expected_shape}"
            )

        result[
            "phase_raw"
        ] = phase_raw

        result[
            "phase_shift"
        ] = (
            np.pi
            /
            12.0
            *
            np.tanh(
                phase_raw
            )
        )

    return result


# =============================================================================
# Point generation
# =============================================================================

def build_unique_source_table(
    y,
):

    source_index = (
        UNIQUE_SOURCE_INDICES.copy()
    )

    values = y[
        source_index
    ]

    scaled = scale_fixed(
        values,
        0.0,
        350.0,
    )

    reference = np.sort(
        y[
            CDF_REFERENCE_INDICES
        ].copy()
    )

    cdf = empirical_cdf_right(
        values,
        reference,
    )

    z_cdf = np.clip(
        2.0 * cdf - 1.0,
        -1.0,
        1.0,
    )

    # IMPORTANT:
    # This exactly matches the validated fidelity reconstruction:
    # theta_CDF = arccos(2F - 1).
    theta_cdf = np.arccos(
        z_cdf
    )

    month = np.mod(
        source_index,
        12,
    ).astype(
        np.int64
    )

    return pd.DataFrame(
        {
            "source_index":
                source_index,

            "month_index":
                month,

            "raw_value":
                values,

            "scaled_value":
                scaled,

            "cdf_value":
                cdf,

            "theta_cdf":
                theta_cdf,
        }
    )


def append_point(
    rows,
    encoding,
    base_row,
    theta,
    phi,
    bloch,
    layer_used,
):

    rows.append(
        {
            "encoding":
                encoding,

            "encoding_label":
                LABELS[encoding],

            "depth":
                DEPTH,

            "seed":
                SEED,

            "layer_used":
                layer_used,

            "source_index":
                int(
                    base_row.source_index
                ),

            "month_index":
                int(
                    base_row.month_index
                ),

            "raw_value":
                float(
                    base_row.raw_value
                ),

            "scaled_value":
                float(
                    base_row.scaled_value
                ),

            "cdf_value":
                float(
                    base_row.cdf_value
                ),

            "theta_cdf":
                float(
                    base_row.theta_cdf
                ),

            "theta":
                float(theta),

            "phi":
                float(phi),

            "bloch_x":
                float(bloch[0]),

            "bloch_y":
                float(bloch[1]),

            "bloch_z":
                float(bloch[2]),
        }
    )


def generate_points(
    source_df,
    model4,
    model5,
    angle_factor,
):

    rows = []

    psi0 = np.array(
        [1.0 + 0j, 0.0 + 0j],
        dtype=np.complex128,
    )

    for row in source_df.itertuples(
        index=False
    ):

        m = int(
            row.month_index
        )

        # ---------------------------------------------------------------------
        # 1. Standard Ry
        # ---------------------------------------------------------------------

        theta = (
            angle_factor
            *
            float(
                row.scaled_value
            )
        )

        phi = 0.0

        bloch = np.array(
            [
                np.sin(theta),
                0.0,
                np.cos(theta),
            ],
            dtype=np.float64,
        )

        append_point(
            rows,
            "ry",
            row,
            theta,
            phi,
            bloch,
            layer_used=-1,
        )

        # ---------------------------------------------------------------------
        # 2. Same-scalar Rx Ry Rz
        #
        # Training gate order:
        # qml.RX(theta)
        # qml.RY(theta)
        # qml.RZ(theta)
        #
        # Therefore state operator is Rz Ry Rx.
        # ---------------------------------------------------------------------

        psi = (
            rz(theta)
            @
            ry(theta)
            @
            rx(theta)
            @
            psi0
        )

        bloch = state_to_bloch(
            psi
        )

        # "theta" and "phi" are not a spherical-parameter
        # decomposition for this encoding; both columns record
        # the common uploaded scalar angle for reproducibility.
        append_point(
            rows,
            "same_scalar_rxryrz",
            row,
            theta,
            theta,
            bloch,
            layer_used=-1,
        )

        # ---------------------------------------------------------------------
        # 3. Fixed seasonal CDF
        # ---------------------------------------------------------------------

        theta_fixed = float(
            row.theta_cdf
        )

        phi_fixed = (
            2.0
            *
            np.pi
            *
            m
            /
            12.0
        )

        bloch = bloch_rz_ry(
            theta_fixed,
            phi_fixed,
        )

        append_point(
            rows,
            "seasonal_meridian",
            row,
            theta_fixed,
            phi_fixed,
            bloch,
            layer_used=-1,
        )

        # ---------------------------------------------------------------------
        # 4. Learnable polar CDF
        # ---------------------------------------------------------------------

        s4 = float(
            model4[
                "strength"
            ][m, LAYER]
        )

        b4 = float(
            model4[
                "shift"
            ][m, LAYER]
        )

        theta4 = (
            np.pi
            /
            2.0
            +
            b4
            +
            s4
            *
            (
                float(
                    row.theta_cdf
                )
                -
                np.pi / 2.0
            )
        )

        phi4 = phi_fixed

        bloch = bloch_rz_ry(
            theta4,
            phi4,
        )

        append_point(
            rows,
            "learnable_seasonal_cdf",
            row,
            theta4,
            phi4,
            bloch,
            layer_used=LAYER,
        )

        # ---------------------------------------------------------------------
        # 5. Learnable polar + azimuth CDF
        # ---------------------------------------------------------------------

        s5 = float(
            model5[
                "strength"
            ][m, LAYER]
        )

        b5 = float(
            model5[
                "shift"
            ][m, LAYER]
        )

        dphi5 = float(
            model5[
                "phase_shift"
            ][m, LAYER]
        )

        theta5 = (
            np.pi
            /
            2.0
            +
            b5
            +
            s5
            *
            (
                float(
                    row.theta_cdf
                )
                -
                np.pi / 2.0
            )
        )

        phi5 = (
            phi_fixed
            +
            dphi5
        )

        bloch = bloch_rz_ry(
            theta5,
            phi5,
        )

        append_point(
            rows,
            "learnable_seasonal_cdf_rz",
            row,
            theta5,
            phi5,
            bloch,
            layer_used=LAYER,
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Geometry validation
# =============================================================================

def validate_points(
    df,
):

    expected = (
        len(ENCODINGS)
        *
        len(
            UNIQUE_SOURCE_INDICES
        )
    )

    if len(df) != expected:
        raise RuntimeError(
            f"Expected {expected} total Bloch points, "
            f"found {len(df)}"
        )

    counts = (
        df.groupby(
            "encoding"
        )
        .size()
    )

    for enc in ENCODINGS:

        n = int(
            counts.get(
                enc,
                -1,
            )
        )

        if n != len(
            UNIQUE_SOURCE_INDICES
        ):
            raise RuntimeError(
                f"{enc}: expected "
                f"{len(UNIQUE_SOURCE_INDICES)} points, got {n}"
            )

    norm = np.sqrt(
        df["bloch_x"].to_numpy() ** 2
        +
        df["bloch_y"].to_numpy() ** 2
        +
        df["bloch_z"].to_numpy() ** 2
    )

    max_norm_error = float(
        np.max(
            np.abs(
                norm - 1.0
            )
        )
    )

    if max_norm_error > 1e-10:
        raise RuntimeError(
            f"Bloch norm validation failed: "
            f"{max_norm_error}"
        )

    return max_norm_error


# =============================================================================
# Bloch sphere plotting
# =============================================================================

def setup_bloch_axis(
    ax,
    title,
):

    # Sphere wireframe.
    u = np.linspace(
        0.0,
        2.0 * np.pi,
        36,
    )

    v = np.linspace(
        0.0,
        np.pi,
        19,
    )

    xs = (
        np.outer(
            np.cos(u),
            np.sin(v),
        )
    )

    ys = (
        np.outer(
            np.sin(u),
            np.sin(v),
        )
    )

    zs = (
        np.outer(
            np.ones_like(u),
            np.cos(v),
        )
    )

    ax.plot_wireframe(
        xs,
        ys,
        zs,
        rstride=3,
        cstride=3,
        linewidth=0.35,
        alpha=0.16,
    )

    # Cartesian axes.
    ax.plot(
        [-1.05, 1.05],
        [0.0, 0.0],
        [0.0, 0.0],
        linewidth=0.6,
        alpha=0.35,
    )

    ax.plot(
        [0.0, 0.0],
        [-1.05, 1.05],
        [0.0, 0.0],
        linewidth=0.6,
        alpha=0.35,
    )

    ax.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [-1.05, 1.05],
        linewidth=0.6,
        alpha=0.35,
    )

    ax.set_xlim(
        -1.05,
        1.05,
    )

    ax.set_ylim(
        -1.05,
        1.05,
    )

    ax.set_zlim(
        -1.05,
        1.05,
    )

    ax.set_box_aspect(
        (1, 1, 1)
    )

    ax.view_init(
        elev=22,
        azim=38,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.set_title(
        title,
        fontsize=10.5,
        pad=6,
    )

    # Coordinate labels.
    ax.text(
        1.10,
        0.0,
        0.0,
        "x",
        fontsize=8,
    )

    ax.text(
        0.0,
        1.10,
        0.0,
        "y",
        fontsize=8,
    )

    ax.text(
        0.0,
        0.0,
        1.10,
        "z",
        fontsize=8,
    )


def save_figure(
    fig,
    stem,
):

    fig.savefig(
        FIG_DIR / f"{stem}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        FIG_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_all_five(
    df,
):

    fig = plt.figure(
        figsize=(20.0, 4.7)
    )

    for i, enc in enumerate(
        ENCODINGS,
        start=1,
    ):

        ax = fig.add_subplot(
            1,
            5,
            i,
            projection="3d",
        )

        setup_bloch_axis(
            ax,
            LABELS[enc],
        )

        d = df[
            df["encoding"]
            ==
            enc
        ]

        # Color by normalized raw/CDF progression rather than
        # assigning arbitrary encoding-specific colors.
        if enc in {
            "seasonal_meridian",
            "learnable_seasonal_cdf",
            "learnable_seasonal_cdf_rz",
        }:
            c = d[
                "cdf_value"
            ].to_numpy()
        else:
            c = d[
                "scaled_value"
            ].to_numpy()

        ax.scatter(
            d["bloch_x"],
            d["bloch_y"],
            d["bloch_z"],
            c=c,
            cmap="viridis",
            s=8,
            alpha=0.60,
            depthshade=False,
            linewidths=0,
        )

    fig.suptitle(
        "Single-qubit training-data encoding geometry",
        fontsize=14,
        y=0.99,
    )

    fig.text(
        0.5,
        0.015,
        (
            "Unique training-source observations; "
            "learnable encodings shown for depth 2, seed 42, "
            "first reupload layer"
        ),
        ha="center",
        fontsize=9.5,
    )

    fig.subplots_adjust(
        left=0.015,
        right=0.985,
        bottom=0.07,
        top=0.89,
        wspace=0.02,
    )

    save_figure(
        fig,
        "17_bloch_training_geometry_all5_depth2_seed42_layer0",
    )


def plot_seasonal_by_month(
    df,
):

    seasonal_encodings = [
        "seasonal_meridian",
        "learnable_seasonal_cdf",
        "learnable_seasonal_cdf_rz",
    ]

    fig = plt.figure(
        figsize=(13.5, 4.8)
    )

    cmap = plt.get_cmap(
        "twilight",
        12,
    )

    for i, enc in enumerate(
        seasonal_encodings,
        start=1,
    ):

        ax = fig.add_subplot(
            1,
            3,
            i,
            projection="3d",
        )

        setup_bloch_axis(
            ax,
            LABELS[enc],
        )

        d = df[
            df["encoding"]
            ==
            enc
        ]

        ax.scatter(
            d["bloch_x"],
            d["bloch_y"],
            d["bloch_z"],
            c=d["month_index"],
            cmap=cmap,
            vmin=-0.5,
            vmax=11.5,
            s=12,
            alpha=0.72,
            depthshade=False,
            linewidths=0,
        )

    handles = []

    for m in range(12):

        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=5.5,
                markerfacecolor=cmap(m),
                markeredgecolor="none",
                label=str(m + 1),
            )
        )

    fig.legend(
        handles=handles,
        title="Seasonal phase index",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=12,
        frameon=False,
        fontsize=8,
        title_fontsize=8.5,
    )

    fig.suptitle(
        "Seasonal encoding geometry by annual phase",
        fontsize=14,
        y=0.99,
    )

    fig.text(
        0.5,
        0.075,
        (
            "Learnable encodings: depth 2, seed 42, "
            "first reupload layer"
        ),
        ha="center",
        fontsize=9.5,
    )

    fig.subplots_adjust(
        left=0.025,
        right=0.975,
        bottom=0.18,
        top=0.88,
        wspace=0.04,
    )

    save_figure(
        fig,
        "17b_bloch_seasonal_monthcolor_depth2_seed42_layer0",
    )


# =============================================================================
# Parameter summary
# =============================================================================

def save_parameter_summary(
    model4,
    model5,
):

    rows = []

    for enc, model in [
        (
            "learnable_seasonal_cdf",
            model4,
        ),
        (
            "learnable_seasonal_cdf_rz",
            model5,
        ),
    ]:

        for m in range(12):

            row = {
                "encoding":
                    enc,

                "depth":
                    DEPTH,

                "seed":
                    SEED,

                "layer":
                    LAYER,

                "month_index":
                    m,

                "strength":
                    float(
                        model[
                            "strength"
                        ][m, LAYER]
                    ),

                "polar_shift_rad":
                    float(
                        model[
                            "shift"
                        ][m, LAYER]
                    ),

                "polar_shift_deg":
                    float(
                        np.degrees(
                            model[
                                "shift"
                            ][m, LAYER]
                        )
                    ),
            }

            if (
                enc
                ==
                "learnable_seasonal_cdf_rz"
            ):

                row[
                    "azimuth_shift_rad"
                ] = float(
                    model[
                        "phase_shift"
                    ][m, LAYER]
                )

                row[
                    "azimuth_shift_deg"
                ] = float(
                    np.degrees(
                        model[
                            "phase_shift"
                        ][m, LAYER]
                    )
                )

            else:

                row[
                    "azimuth_shift_rad"
                ] = 0.0

                row[
                    "azimuth_shift_deg"
                ] = 0.0

            rows.append(
                row
            )

    pd.DataFrame(
        rows
    ).to_csv(
        OUT_DIR
        /
        "bloch_learned_encoder_parameters_depth2_seed42_layer0.csv",
        index=False,
    )


# =============================================================================
# Main
# =============================================================================

def main():

    print("=" * 100)
    print("FIVE-ENCODING BLOCH TRAINING-GEOMETRY ANALYSIS")
    print("=" * 100)

    y, value_column = load_series()

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    if len(manifest) != 150:
        raise RuntimeError(
            f"Expected 150-run master manifest, got {len(manifest)}"
        )

    run4 = get_run_dir(
        manifest,
        "learnable_seasonal_cdf",
        DEPTH,
        SEED,
    )

    run5 = get_run_dir(
        manifest,
        "learnable_seasonal_cdf_rz",
        DEPTH,
        SEED,
    )

    model4 = load_encoder_raw_parameters(
        run4,
        "learnable_seasonal_cdf",
        DEPTH,
    )

    model5 = load_encoder_raw_parameters(
        run5,
        "learnable_seasonal_cdf_rz",
        DEPTH,
    )

    # -------------------------------------------------------------------------
    # Obtain the production angle factor.
    #
    # Prefer model-4 config, but require model 5 to agree.
    # -------------------------------------------------------------------------

    config4_path = (
        run4
        /
        "config.json"
    )

    config5_path = (
        run5
        /
        "config.json"
    )

    if not config4_path.exists():
        raise FileNotFoundError(
            config4_path
        )

    if not config5_path.exists():
        raise FileNotFoundError(
            config5_path
        )

    with open(
        config4_path
    ) as f:
        config4 = json.load(f)

    with open(
        config5_path
    ) as f:
        config5 = json.load(f)

    angle4 = float(
        config4[
            "angle_factor"
        ]
    )

    angle5 = float(
        config5[
            "angle_factor"
        ]
    )

    if not np.isclose(
        angle4,
        angle5,
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError(
            f"angle_factor mismatch: model4={angle4}, model5={angle5}"
        )

    angle_factor = angle4

    source_df = build_unique_source_table(
        y
    )

    if len(source_df) != 363:
        raise RuntimeError(
            f"Expected 363 unique training source observations, "
            f"got {len(source_df)}"
        )

    points = generate_points(
        source_df,
        model4,
        model5,
        angle_factor,
    )

    max_norm_error = validate_points(
        points
    )

    points_path = (
        OUT_DIR
        /
        "bloch_training_points_all5_depth2_seed42_layer0.csv"
    )

    points.to_csv(
        points_path,
        index=False,
    )

    save_parameter_summary(
        model4,
        model5,
    )

    plot_all_five(
        points
    )

    plot_seasonal_by_month(
        points
    )

    # -------------------------------------------------------------------------
    # Human-readable validation summary
    # -------------------------------------------------------------------------

    print()
    print("Dataset value column:", value_column)
    print("Series length:", len(y))

    print(
        "Training target count:",
        TRAIN_TARGET_STOP_EXCLUSIVE
        -
        TRAIN_TARGET_START,
    )

    print(
        "Unique training-source observations:",
        len(source_df),
    )

    print(
        "CDF reference observations:",
        len(
            CDF_REFERENCE_INDICES
        ),
    )

    print(
        "Points per encoding:",
        len(source_df),
    )

    print(
        "Total plotted/saved points:",
        len(points),
    )

    print(
        "Depth / seed / layer:",
        DEPTH,
        SEED,
        LAYER,
    )

    print(
        "angle_factor:",
        angle_factor,
    )

    print(
        "Model 4 checkpoint:",
        model4[
            "checkpoint_path"
        ],
    )

    print(
        "Model 5 checkpoint:",
        model5[
            "checkpoint_path"
        ],
    )

    print(
        "Max Bloch norm error:",
        f"{max_norm_error:.3e}",
    )

    print()
    print("Model 4 layer-0 strength:")
    print(
        np.array2string(
            model4[
                "strength"
            ][:, LAYER],
            precision=5,
        )
    )

    print()
    print("Model 5 layer-0 strength:")
    print(
        np.array2string(
            model5[
                "strength"
            ][:, LAYER],
            precision=5,
        )
    )

    print()
    print(
        "Model 5 layer-0 azimuth shifts [deg]:"
    )

    print(
        np.array2string(
            np.degrees(
                model5[
                    "phase_shift"
                ][:, LAYER]
            ),
            precision=3,
        )
    )

    print()
    print("Generated:")

    for name in [
        "17_bloch_training_geometry_all5_depth2_seed42_layer0.pdf",
        "17_bloch_training_geometry_all5_depth2_seed42_layer0.png",
        "17b_bloch_seasonal_monthcolor_depth2_seed42_layer0.pdf",
        "17b_bloch_seasonal_monthcolor_depth2_seed42_layer0.png",
    ]:
        print(
            " ",
            FIG_DIR / name,
        )

    print()
    print("Tables:")

    print(
        " ",
        points_path,
    )

    print(
        " ",
        OUT_DIR
        /
        "bloch_learned_encoder_parameters_depth2_seed42_layer0.csv",
    )

    print()
    print("=" * 100)
    print("BLOCH GEOMETRY STATUS: PASS")
    print("=" * 100)


if __name__ == "__main__":
    main()
