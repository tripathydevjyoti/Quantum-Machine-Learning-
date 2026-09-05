from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# BASIC DATA UTILITIES
# ============================================================

def scale_fixed(y, data_min, data_max):
    y = np.asarray(y, dtype=np.float64)
    return 2.0 * (y - data_min) / (data_max - data_min) - 1.0


def build_windows(series, window, target_start, target_end):
    targets = np.arange(
        target_start,
        target_end + 1,
        dtype=int,
    )

    X = np.stack(
        [
            series[t - window:t]
            for t in targets
        ],
        axis=0,
    )

    lag_indices = np.stack(
        [
            np.arange(
                t - window,
                t,
                dtype=int,
            )
            for t in targets
        ],
        axis=0,
    )

    y = series[targets]

    return X, y, targets, lag_indices


def empirical_cdf_transform(values, train_reference):
    """
    Training-only empirical CDF.

    The returned quantiles are clipped away from exactly 0 and 1
    so the equal-area Bloch mapping does not sit exactly on the poles.
    """
    ref = np.sort(
        np.asarray(
            train_reference,
            dtype=np.float64,
        ).ravel()
    )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = ref.size

    pos = np.searchsorted(
        ref,
        values,
        side="right",
    )

    u = (pos - 0.5) / n

    return np.clip(
        u,
        0.5 / n,
        1.0 - 0.5 / n,
    )


# ============================================================
# TIME-SERIES DIAGNOSTICS
# ============================================================

def acf_biased(x, max_lag):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x - np.mean(x)

    denom = np.dot(x, x)

    if denom <= 0:
        return np.full(
            max_lag + 1,
            np.nan,
        )

    out = np.empty(
        max_lag + 1,
        dtype=np.float64,
    )

    for lag in range(max_lag + 1):

        out[lag] = (
            np.dot(
                x[:len(x) - lag],
                x[lag:],
            )
            /
            denom
        )

    return out


def spectral_table(x):
    """
    Non-DC FFT power using the same basic convention as our
    existing spectral analysis: normalized over non-DC bins.
    """
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    xc = x - np.mean(x)

    fft = np.fft.rfft(xc)

    power = np.abs(fft) ** 2

    freq = np.fft.rfftfreq(
        len(xc),
        d=1.0,
    )

    mask = freq > 0

    freq = freq[mask]
    power = power[mask]

    periods = 1.0 / freq

    total = power.sum()

    if total > 0:
        norm = power / total
    else:
        norm = np.full_like(
            power,
            np.nan,
        )

    return pd.DataFrame(
        {
            "frequency_cycles_per_step":
                freq,

            "period_steps":
                periods,

            "absolute_power":
                power,

            "normalized_non_dc_power":
                norm,
        }
    )


def fit_harmonic(y, omega):
    """
    Training-only linear harmonic:

        y_t = beta0
            + beta_c cos(omega t)
            + beta_s sin(omega t)
            + residual.
    """
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    t = np.arange(
        len(y),
        dtype=np.float64,
    )

    X = np.column_stack(
        [
            np.ones(len(y)),
            np.cos(omega * t),
            np.sin(omega * t),
        ]
    )

    beta, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    pred = X @ beta
    resid = y - pred

    sse = float(
        np.sum(
            resid ** 2
        )
    )

    sst = float(
        np.sum(
            (y - np.mean(y)) ** 2
        )
    )

    r2 = (
        1.0 - sse / sst
        if sst > 0
        else np.nan
    )

    return {
        "intercept":
            float(beta[0]),

        "cos_coefficient":
            float(beta[1]),

        "sin_coefficient":
            float(beta[2]),

        "amplitude":
            float(
                np.hypot(
                    beta[1],
                    beta[2],
                )
            ),

        "phase_rad":
            float(
                np.arctan2(
                    beta[2],
                    beta[1],
                )
            ),

        "r2":
            r2,

        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        resid ** 2
                    )
                )
            ),

        "residual_std_sample":
            float(
                np.std(
                    resid,
                    ddof=1,
                )
            ),
    }


# ============================================================
# SINGLE-QUBIT UNITARIES
# ============================================================

def rx(theta):
    theta = np.asarray(
        theta,
        dtype=np.float64,
    )

    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)

    out = np.empty(
        theta.shape + (2, 2),
        dtype=np.complex128,
    )

    out[..., 0, 0] = c
    out[..., 0, 1] = -1j * s
    out[..., 1, 0] = -1j * s
    out[..., 1, 1] = c

    return out


def ry(theta):
    theta = np.asarray(
        theta,
        dtype=np.float64,
    )

    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)

    out = np.empty(
        theta.shape + (2, 2),
        dtype=np.complex128,
    )

    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c

    return out


def rz(theta):
    theta = np.asarray(
        theta,
        dtype=np.float64,
    )

    out = np.zeros(
        theta.shape + (2, 2),
        dtype=np.complex128,
    )

    out[..., 0, 0] = (
        np.exp(
            -0.5j * theta
        )
    )

    out[..., 1, 1] = (
        np.exp(
            +0.5j * theta
        )
    )

    return out


def matmul2(A, B):
    return np.einsum(
        "...ab,...bc->...ac",
        A,
        B,
    )


# ============================================================
# ENCODINGS
# ============================================================

def build_encoding_unitaries(
    name,
    X,
    lag_indices,
    theta_equal_area,
    phase,
    sector_delta,
):
    """
    Output shape:

        [n_windows, 14, 2, 2]

    Each qubit receives a single-qubit encoding.
    """

    # --------------------------------------------------------
    # Existing production encoding
    #
    # U = Ry(pi x)
    # --------------------------------------------------------

    if name == "original_ry":

        return ry(
            np.pi * X
        )

    # --------------------------------------------------------
    # Same scalar into three axes
    #
    # Circuit application order:
    #
    # Rx(theta) -> Ry(theta) -> Rz(theta)
    #
    # Matrix:
    #
    # U = Rz Ry Rx
    # --------------------------------------------------------

    if name == "same_scalar_rxryrz":

        th = np.pi * X

        return matmul2(
            rz(th),
            matmul2(
                ry(th),
                rx(th),
            ),
        )

    # --------------------------------------------------------
    # Season-aware equal-area meridian encoding
    #
    # U = Rz(phi_month) Ry(theta_value)
    # --------------------------------------------------------

    if name == "seasonal_meridian":

        th = (
            theta_equal_area[
                lag_indices
            ]
        )

        ph = (
            phase[
                lag_indices
            ]
        )

        return matmul2(
            rz(ph),
            ry(th),
        )

    # --------------------------------------------------------
    # Season-aware sector encoding
    #
    # U = Rz(phi_month + delta_phi) Ry(theta_value)
    # --------------------------------------------------------

    if name == "seasonal_sector":

        th = (
            theta_equal_area[
                lag_indices
            ]
        )

        ph = (
            phase[
                lag_indices
            ]
            +
            sector_delta[
                lag_indices
            ]
        )

        return matmul2(
            rz(ph),
            ry(th),
        )

    raise ValueError(
        f"Unknown encoding: {name}"
    )


# ============================================================
# TWO-DESIGN OVERLAP KERNEL
# ============================================================

def product_overlap_kernel(
    U,
    lag,
    d,
):
    """
    For windows t and s=t+lag, calculate:

        S_ts =
            |Tr(E_t^\dagger E_s)|^2 / d^2

    and the first-layer encoded-state fidelity

        F1_ts =
            |<psi_t | psi_s>|^2.

    Since E_t is a product of 14 one-qubit unitaries,
    both quantities factorize exactly into one-qubit products.

    Then:

        beta_ts =
            (d^2 S_ts - 1) / (d^2 - 1)

    is the exact Haar / unitary-2-design contraction factor
    from our derivation.
    """

    Ut = U[:-lag]
    Us = U[lag:]

    # --------------------------------------------------------
    # Relative one-qubit unitary:
    #
    # W_j = U_t,j^\dagger U_s,j
    # --------------------------------------------------------

    W = np.einsum(
        "...ba,...bc->...ac",
        Ut.conj(),
        Us,
    )

    trW = (
        W[..., 0, 0]
        +
        W[..., 1, 1]
    )

    S_j = np.clip(
        np.abs(trW) ** 2 / 4.0,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Encoded-state fidelity for |0>.
    #
    # U|0> is the first column.
    # --------------------------------------------------------

    psi_t = Ut[..., :, 0]
    psi_s = Us[..., :, 0]

    inner = np.sum(
        psi_t.conj()
        *
        psi_s,
        axis=-1,
    )

    F1_j = np.clip(
        np.abs(inner) ** 2,
        0.0,
        1.0,
    )

    # Products can be extremely small, so use log space.
    eps = 1e-300

    S = np.exp(
        np.sum(
            np.log(
                np.clip(
                    S_j,
                    eps,
                    1.0,
                )
            ),
            axis=1,
        )
    )

    F1 = np.exp(
        np.sum(
            np.log(
                np.clip(
                    F1_j,
                    eps,
                    1.0,
                )
            ),
            axis=1,
        )
    )

    d2 = float(
        d * d
    )

    beta = (
        d2 * S - 1.0
    ) / (
        d2 - 1.0
    )

    return S, F1, beta


def summarize_array(
    x,
    prefix,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return {
        f"{prefix}_mean":
            float(
                np.mean(x)
            ),

        f"{prefix}_median":
            float(
                np.median(x)
            ),

        f"{prefix}_std":
            float(
                np.std(
                    x,
                    ddof=1,
                )
            )
            if len(x) > 1
            else 0.0,

        f"{prefix}_q05":
            float(
                np.quantile(
                    x,
                    0.05,
                )
            ),

        f"{prefix}_q25":
            float(
                np.quantile(
                    x,
                    0.25,
                )
            ),

        f"{prefix}_q75":
            float(
                np.quantile(
                    x,
                    0.75,
                )
            ),

        f"{prefix}_q95":
            float(
                np.quantile(
                    x,
                    0.95,
                )
            ),

        f"{prefix}_min":
            float(
                np.min(x)
            ),

        f"{prefix}_max":
            float(
                np.max(x)
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--csv_path",
        required=True,
    )

    p.add_argument(
        "--value_col",
        type=int,
        default=4,
    )

    p.add_argument(
        "--window_size",
        type=int,
        default=14,
    )

    p.add_argument(
        "--fit_end_idx",
        type=int,
        default=363,
        help=(
            "Last observation available to "
            "training-only preprocessing."
        ),
    )

    p.add_argument(
        "--train_target_start",
        type=int,
        default=14,
    )

    p.add_argument(
        "--train_target_end",
        type=int,
        default=363,
    )

    p.add_argument(
        "--scaler_data_min",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--scaler_data_max",
        type=float,
        default=350.0,
    )

    p.add_argument(
        "--max_lag",
        type=int,
        default=48,
    )

    p.add_argument(
        "--rolling_window",
        type=int,
        default=120,
    )

    p.add_argument(
        "--rolling_step",
        type=int,
        default=12,
    )

    p.add_argument(
        "--season_band_min_period",
        type=float,
        default=10.5,
    )

    p.add_argument(
        "--season_band_max_period",
        type=float,
        default=13.125,
    )

    p.add_argument(
        "--sector_fraction_half_month",
        type=float,
        default=0.85,
    )

    p.add_argument(
        "--output_dir",
        required=True,
    )

    args = p.parse_args()

    out = Path(
        args.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD SERIES
    # ========================================================

    df = pd.read_csv(
        args.csv_path
    )

    y_all = (
        pd.to_numeric(
            df.iloc[
                :,
                args.value_col
            ],
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=np.float64
        )
    )

    if len(y_all) != 539:

        raise RuntimeError(
            "Expected 539 non-missing observations, "
            f"got {len(y_all)}"
        )

    y_fit = (
        y_all[
            :args.fit_end_idx + 1
        ]
    )

    x_all = scale_fixed(
        y_all,
        args.scaler_data_min,
        args.scaler_data_max,
    )

    x_fit = (
        x_all[
            :args.fit_end_idx + 1
        ]
    )

    # ========================================================
    # EXACT TRAINING WINDOWS
    # ========================================================

    (
        X_train,
        y_train_scaled,
        train_targets,
        lag_indices,
    ) = build_windows(
        x_all,
        args.window_size,
        args.train_target_start,
        args.train_target_end,
    )

    if X_train.shape != (350, 14):

        raise RuntimeError(
            "Unexpected training-window shape: "
            f"{X_train.shape}"
        )

    # ========================================================
    # PCA / EFFECTIVE DIMENSION
    # ========================================================

    cov = np.cov(
        X_train,
        rowvar=False,
        ddof=1,
    )

    eigvals, eigvecs = (
        np.linalg.eigh(
            cov
        )
    )

    order = np.argsort(
        eigvals
    )[::-1]

    eigvals = eigvals[
        order
    ]

    eigvecs = eigvecs[
        :,
        order
    ]

    explained = (
        eigvals
        /
        eigvals.sum()
    )

    cumulative = np.cumsum(
        explained
    )

    d_eff = float(
        eigvals.sum() ** 2
        /
        np.sum(
            eigvals ** 2
        )
    )

    pca_df = pd.DataFrame(
        {
            "pc":
                np.arange(
                    1,
                    args.window_size + 1,
                ),

            "eigenvalue":
                eigvals,

            "explained_variance_ratio":
                explained,

            "cumulative_explained_variance":
                cumulative,
        }
    )

    pca_df.to_csv(
        out
        /
        "pca_eigenvalues.csv",
        index=False,
    )

    eigvec_df = pd.DataFrame(
        eigvecs,

        index=[
            f"lag_{args.window_size - i}"
            for i
            in range(
                args.window_size
            )
        ],

        columns=[
            f"PC{i}"
            for i
            in range(
                1,
                args.window_size + 1,
            )
        ],
    )

    eigvec_df.to_csv(
        out
        /
        "pca_eigenvectors.csv"
    )

    # ========================================================
    # AUTOCORRELATION
    # ========================================================

    acf = acf_biased(
        y_fit,
        args.max_lag,
    )

    acf_df = pd.DataFrame(
        {
            "lag":
                np.arange(
                    args.max_lag + 1
                ),

            "acf":
                acf,
        }
    )

    acf_df.to_csv(
        out
        /
        "training_acf.csv",
        index=False,
    )

    # ========================================================
    # GLOBAL TRAINING SPECTRUM
    # ========================================================

    spec = spectral_table(
        y_fit
    )

    spec.to_csv(
        out
        /
        "training_periodogram.csv",
        index=False,
    )

    season_mask = (
        (
            spec[
                "period_steps"
            ]
            >=
            args.season_band_min_period
        )
        &
        (
            spec[
                "period_steps"
            ]
            <=
            args.season_band_max_period
        )
    )

    if not season_mask.any():

        raise RuntimeError(
            "No FFT bin found in requested "
            "seasonal period band."
        )

    season_spec = (
        spec.loc[
            season_mask
        ]
        .copy()
    )

    best_idx = (
        season_spec[
            "absolute_power"
        ]
        .idxmax()
    )

    dominant_period = float(
        spec.loc[
            best_idx,
            "period_steps",
        ]
    )

    dominant_frequency = float(
        spec.loc[
            best_idx,
            "frequency_cycles_per_step",
        ]
    )

    dominant_omega = float(
        2.0
        *
        np.pi
        *
        dominant_frequency
    )

    dominant_bin_share = float(
        spec.loc[
            best_idx,
            "normalized_non_dc_power",
        ]
    )

    seasonal_band_share = float(
        season_spec[
            "normalized_non_dc_power"
        ]
        .sum()
    )

    harmonic = fit_harmonic(
        y_fit,
        dominant_omega,
    )

    top_freq = (
        spec
        .sort_values(
            "absolute_power",
            ascending=False,
        )
        .head(12)
        .reset_index(
            drop=True
        )
    )

    top_freq.insert(
        0,
        "rank",
        np.arange(
            1,
            len(top_freq) + 1,
        ),
    )

    top_freq.to_csv(
        out
        /
        "top_training_frequencies.csv",
        index=False,
    )

    # ========================================================
    # ROLLING LOCAL SPECTRA
    #
    # This tests the local-stationarity approximation.
    # 120 observations = roughly 10 annual cycles.
    # ========================================================

    rolling_rows = []

    rw = args.rolling_window
    rs = args.rolling_step

    for start in range(
        0,
        len(y_fit) - rw + 1,
        rs,
    ):

        end = start + rw

        seg = (
            y_fit[
                start:end
            ]
        )

        sp = spectral_table(
            seg
        )

        sm = (
            (
                sp[
                    "period_steps"
                ]
                >=
                args.season_band_min_period
            )
            &
            (
                sp[
                    "period_steps"
                ]
                <=
                args.season_band_max_period
            )
        )

        if sm.any():

            local = (
                sp.loc[
                    sm
                ]
            )

            local_best = (
                local[
                    "absolute_power"
                ]
                .idxmax()
            )

            local_period = float(
                sp.loc[
                    local_best,
                    "period_steps",
                ]
            )

            local_share = float(
                local[
                    "normalized_non_dc_power"
                ]
                .sum()
            )

        else:

            local_period = np.nan
            local_share = np.nan

        rolling_rows.append(
            {
                "start_index":
                    start,

                "end_index_inclusive":
                    end - 1,

                "center_index":
                    0.5
                    *
                    (
                        start
                        +
                        end
                        -
                        1
                    ),

                "seasonal_band_normalized_power":
                    local_share,

                "dominant_seasonal_period":
                    local_period,

                "segment_mean":
                    float(
                        np.mean(
                            seg
                        )
                    ),

                "segment_std_sample":
                    float(
                        np.std(
                            seg,
                            ddof=1,
                        )
                    ),
            }
        )

    rolling_df = pd.DataFrame(
        rolling_rows
    )

    rolling_df.to_csv(
        out
        /
        "rolling_training_spectral_diagnostics.csv",
        index=False,
    )

    # ========================================================
    # SEASON-AWARE ENCODING INGREDIENTS
    #
    # Everything fitted from observations 0--363 only.
    #
    # Phase uses index mod 12.
    #
    # A global calendar-month offset only rotates all seasonal
    # states by the same Rz and therefore cancels from the
    # pairwise overlap kernel.
    # ========================================================

    u_all = empirical_cdf_transform(
        x_all,
        x_fit,
    )

    z_all = (
        2.0 * u_all
        -
        1.0
    )

    theta_equal_area = np.arccos(
        z_all
    )

    obs_idx = np.arange(
        len(x_all),
        dtype=int,
    )

    phase = (
        2.0
        *
        np.pi
        *
        (
            obs_idx % 12
        )
        /
        12.0
    )

    # --------------------------------------------------------
    # Sector displacement:
    #
    # delta_phi =
    #   delta_max *
    #   tanh(Delta x / sigma_Delta_x_train)
    #
    # sigma uses training-only observations.
    # --------------------------------------------------------

    dx = np.zeros_like(
        x_all
    )

    dx[1:] = (
        x_all[1:]
        -
        x_all[:-1]
    )

    dx_scale = float(
        np.std(
            dx[
                :args.fit_end_idx + 1
            ],
            ddof=1,
        )
    )

    if dx_scale <= 0:
        dx_scale = 1.0

    delta_max = (
        args.sector_fraction_half_month
        *
        (
            np.pi / 12.0
        )
    )

    sector_delta = (
        delta_max
        *
        np.tanh(
            dx / dx_scale
        )
    )

    # ========================================================
    # CANDIDATE ENCODINGS
    # ========================================================

    encoding_names = [
        "original_ry",
        "same_scalar_rxryrz",
        "seasonal_meridian",
        "seasonal_sector",
    ]

    encoding_labels = {
        "original_ry":
            "Original Ry(pi x)",

        "same_scalar_rxryrz":
            "Same-scalar RxRyRz",

        "seasonal_meridian":
            "Seasonal meridian",

        "seasonal_sector":
            "Seasonal sector",
    }

    d = 2 ** args.window_size

    summary_rows = []
    pair_rows = []
    memory_rows = []

    # ========================================================
    # EXACT PRODUCT OVERLAP KERNELS
    # ========================================================

    for enc in encoding_names:

        U = build_encoding_unitaries(
            enc,
            X_train,
            lag_indices,
            theta_equal_area,
            phase,
            sector_delta,
        )

        for tau in range(
            1,
            min(
                args.max_lag,
                len(X_train) - 1,
            ) + 1,
        ):

            (
                S,
                F1,
                beta,
            ) = product_overlap_kernel(
                U,
                lag=tau,
                d=d,
            )

            row = {
                "encoding":
                    enc,

                "label":
                    encoding_labels[
                        enc
                    ],

                "tau":
                    tau,

                "n_pairs":
                    len(beta),
            }

            row.update(
                summarize_array(
                    S,
                    "S",
                )
            )

            row.update(
                summarize_array(
                    F1,
                    "F1",
                )
            )

            row.update(
                summarize_array(
                    beta,
                    "beta",
                )
            )

            # ------------------------------------------------
            # Seasonal-memory half-life:
            #
            # beta^(L-1) = 1/2
            #
            # L_half =
            #   1 + ln(1/2)/ln(beta)
            #
            # Defined for 0 < beta < 1.
            # ------------------------------------------------

            good = (
                (beta > 0.0)
                &
                (beta < 1.0)
            )

            if np.any(
                good
            ):

                half_life = (
                    1.0
                    +
                    np.log(0.5)
                    /
                    np.log(
                        beta[
                            good
                        ]
                    )
                )

                row.update(
                    summarize_array(
                        half_life,
                        "L_half",
                    )
                )

            else:

                for key in [
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

                    row[
                        f"L_half_{key}"
                    ] = np.nan

            summary_rows.append(
                row
            )

            # ------------------------------------------------
            # Save every pair separately.
            #
            # This is important because:
            #
            # E[beta^(L-1)]
            #
            # is NOT generally equal to:
            #
            # E[beta]^(L-1).
            # ------------------------------------------------

            for i in range(
                len(beta)
            ):

                pair_rows.append(
                    {
                        "encoding":
                            enc,

                        "tau":
                            tau,

                        "pair_index":
                            i,

                        "target_t":
                            int(
                                train_targets[
                                    i
                                ]
                            ),

                        "target_s":
                            int(
                                train_targets[
                                    i + tau
                                ]
                            ),

                        "S":
                            float(
                                S[i]
                            ),

                        "F1":
                            float(
                                F1[i]
                            ),

                        "beta":
                            float(
                                beta[i]
                            ),
                    }
                )

            # ------------------------------------------------
            # Exact 2-design recurrence for each pair:
            #
            # F_L - 1/d
            # =
            # beta^(L-1) (F_1 - 1/d)
            # ------------------------------------------------

            for L in range(
                1,
                7,
            ):

                contrast_factor = (
                    beta
                    **
                    (L - 1)
                )

                F_pred = (
                    1.0 / d
                    +
                    contrast_factor
                    *
                    (
                        F1
                        -
                        1.0 / d
                    )
                )

                memory_rows.append(
                    {
                        "encoding":
                            enc,

                        "label":
                            encoding_labels[
                                enc
                            ],

                        "tau":
                            tau,

                        "depth_L":
                            L,

                        "mean_beta_power":
                            float(
                                np.mean(
                                    contrast_factor
                                )
                            ),

                        "median_beta_power":
                            float(
                                np.median(
                                    contrast_factor
                                )
                            ),

                        "mean_predicted_fidelity":
                            float(
                                np.mean(
                                    F_pred
                                )
                            ),

                        "median_predicted_fidelity":
                            float(
                                np.median(
                                    F_pred
                                )
                            ),

                        "mean_predicted_contrast":
                            float(
                                np.mean(
                                    F_pred
                                    -
                                    1.0 / d
                                )
                            ),
                    }
                )

    kernel_summary = pd.DataFrame(
        summary_rows
    )

    kernel_pairs = pd.DataFrame(
        pair_rows
    )

    memory_df = pd.DataFrame(
        memory_rows
    )

    kernel_summary.to_csv(
        out
        /
        "encoding_kernel_summary_by_lag.csv",
        index=False,
    )

    kernel_pairs.to_csv(
        out
        /
        "encoding_kernel_pair_values.csv",
        index=False,
    )

    memory_df.to_csv(
        out
        /
        "encoding_predicted_memory_by_lag_depth.csv",
        index=False,
    )

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    theory_summary = {
        "n_total_observations":
            int(
                len(y_all)
            ),

        "training_available_observation_range":
            [
                0,
                int(
                    args.fit_end_idx
                ),
            ],

        "training_target_range":
            [
                int(
                    args.train_target_start
                ),
                int(
                    args.train_target_end
                ),
            ],

        "n_training_windows":
            int(
                len(X_train)
            ),

        "window_size":
            int(
                args.window_size
            ),

        "effective_dimension_participation_ratio":
            d_eff,

        "pc1_explained_variance":
            float(
                explained[0]
            ),

        "pc2_cumulative_explained_variance":
            float(
                cumulative[1]
            ),

        "pc3_cumulative_explained_variance":
            float(
                cumulative[2]
            ),

        "dominant_training_period_steps":
            dominant_period,

        "dominant_training_frequency_cycles_per_step":
            dominant_frequency,

        "dominant_training_bin_power_share":
            dominant_bin_share,

        "seasonal_band_power_share":
            seasonal_band_share,

        "harmonic_fit":
            harmonic,

        "acf_lag_1":
            float(
                acf[1]
            ),

        "acf_lag_6":
            float(
                acf[6]
            ),

        "acf_lag_12":
            float(
                acf[12]
            ),

        "acf_lag_24":
            float(
                acf[24]
            ),

        "sector_delta_max_rad":
            float(
                delta_max
            ),

        "sector_dx_scale_training_only":
            dx_scale,

        "hilbert_dimension_d":
            int(
                d
            ),

        "haar_pair_fidelity_baseline_1_over_d":
            float(
                1.0 / d
            ),
    }

    with open(
        out
        /
        "theory_phase12_summary.json",
        "w",
    ) as f:

        json.dump(
            theory_summary,
            f,
            indent=2,
        )

    # ========================================================
    # FIGURES
    # ========================================================

    # PCA eigenvalues
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        np.arange(
            1,
            args.window_size + 1,
        ),
        eigvals,
        marker="o",
    )

    ax.set_xlabel(
        "Principal component"
    )

    ax.set_ylabel(
        "Covariance eigenvalue"
    )

    ax.set_title(
        "Training lag-window covariance spectrum "
        f"(D_eff={d_eff:.3f})"
    )

    fig.tight_layout()

    fig.savefig(
        out
        /
        "01_pca_eigenvalues.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # PCA cumulative variance
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        np.arange(
            1,
            args.window_size + 1,
        ),
        cumulative,
        marker="o",
    )

    ax.set_xlabel(
        "Number of principal components"
    )

    ax.set_ylabel(
        "Cumulative explained variance"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_title(
        "Effective dimensionality of training lag windows"
    )

    fig.tight_layout()

    fig.savefig(
        out
        /
        "02_pca_cumulative_variance.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ACF
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.stem(
        acf_df[
            "lag"
        ],
        acf_df[
            "acf"
        ],
    )

    ax.set_xlabel(
        "Lag"
    )

    ax.set_ylabel(
        "Autocorrelation"
    )

    ax.set_title(
        "Training-available series autocorrelation"
    )

    fig.tight_layout()

    fig.savefig(
        out
        /
        "03_training_acf.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Periodogram
    plot_spec = (
        spec[
            (
                spec[
                    "period_steps"
                ]
                >=
                2.0
            )
            &
            (
                spec[
                    "period_steps"
                ]
                <=
                60.0
            )
        ]
        .sort_values(
            "period_steps"
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        plot_spec[
            "period_steps"
        ],
        plot_spec[
            "normalized_non_dc_power"
        ],
    )

    ax.axvline(
        dominant_period,
        linestyle="--",
    )

    ax.set_xlabel(
        "Period [steps]"
    )

    ax.set_ylabel(
        "Normalized non-DC power"
    )

    ax.set_title(
        "Training-available periodogram"
    )

    fig.tight_layout()

    fig.savefig(
        out
        /
        "04_training_periodogram.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Rolling seasonal power
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        rolling_df[
            "center_index"
        ],
        rolling_df[
            "seasonal_band_normalized_power"
        ],
        marker="o",
    )

    ax.set_xlabel(
        "Training-time index"
    )

    ax.set_ylabel(
        "Seasonal-band normalized power"
    )

    ax.set_title(
        "Rolling seasonal-band strength within training data"
    )

    fig.tight_layout()

    fig.savefig(
        out
        /
        "05_rolling_seasonal_band_share.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Mean beta vs lag
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for enc in encoding_names:

        g = (
            kernel_summary[
                kernel_summary[
                    "encoding"
                ]
                ==
                enc
            ]
        )

        ax.plot(
            g[
                "tau"
            ],
            g[
                "beta_mean"
            ],
            marker="o",
            markersize=3,
            label=
                encoding_labels[
                    enc
                ],
        )

    ax.set_xlabel(
        "Window separation tau [steps]"
    )

    ax.set_ylabel(
        "Mean two-design contraction beta(t,t+tau)"
    )

    ax.set_title(
        "Encoding-induced pairwise memory "
        "under 2-design approximation"
    )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    fig.savefig(
        out
        /
        "06_mean_beta_vs_lag.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # tau = 12 memory
    tau12 = (
        memory_df[
            memory_df[
                "tau"
            ]
            ==
            12
        ]
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for enc in encoding_names:

        g = (
            tau12[
                tau12[
                    "encoding"
                ]
                ==
                enc
            ]
        )

        ax.plot(
            g[
                "depth_L"
            ],
            g[
                "mean_beta_power"
            ],
            marker="o",
            label=
                encoding_labels[
                    enc
                ],
        )

    ax.set_xlabel(
        "Reupload depth L"
    )

    ax.set_ylabel(
        "Mean beta^(L-1)"
    )

    ax.set_title(
        "Predicted retention of 12-step pairwise memory"
    )

    ax.set_ylim(
        -0.05,
        1.05,
    )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    fig.savefig(
        out
        /
        "07_predicted_tau12_memory_vs_depth.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Temporal-memory profiles at representative depths
    for L in [
        1,
        3,
        6,
    ]:

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        dsub = (
            memory_df[
                memory_df[
                    "depth_L"
                ]
                ==
                L
            ]
        )

        for enc in encoding_names:

            g = (
                dsub[
                    dsub[
                        "encoding"
                    ]
                    ==
                    enc
                ]
            )

            ax.plot(
                g[
                    "tau"
                ],
                g[
                    "mean_beta_power"
                ],
                marker="o",
                markersize=3,
                label=
                    encoding_labels[
                        enc
                    ],
            )

        ax.set_xlabel(
            "Window separation tau [steps]"
        )

        ax.set_ylabel(
            "Mean beta^(L-1)"
        )

        ax.set_title(
            "Predicted temporal-memory profile "
            f"at reupload depth L={L}"
        )

        ax.legend(
            fontsize=8
        )

        fig.tight_layout()

        fig.savefig(
            out
            /
            f"08_memory_profile_depth_{L}.png",
            dpi=220,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print(
        "=" * 110
    )

    print(
        "PHASE 1/2 THEORY DIAGNOSTICS"
    )

    print(
        "=" * 110
    )

    print(
        f"Total observations: {len(y_all)}"
    )

    print(
        "Training-available observations: "
        f"0--{args.fit_end_idx} "
        f"(n={len(y_fit)})"
    )

    print(
        f"Training windows: {X_train.shape}"
    )

    print()

    print(
        "DATA GEOMETRY"
    )

    print(
        "-" * 110
    )

    print(
        f"Effective dimension D_eff: "
        f"{d_eff:.6f}"
    )

    print(
        f"PC1 explained variance: "
        f"{explained[0]:.6f}"
    )

    print(
        f"PC1+PC2 cumulative: "
        f"{cumulative[1]:.6f}"
    )

    print(
        f"PC1+PC2+PC3 cumulative: "
        f"{cumulative[2]:.6f}"
    )

    print()

    print(
        "TRAINING SPECTRUM"
    )

    print(
        "-" * 110
    )

    print(
        "Dominant seasonal-band period: "
        f"{dominant_period:.12f}"
    )

    print(
        "Dominant bin normalized power: "
        f"{dominant_bin_share:.6f}"
    )

    print(
        "Total seasonal-band normalized power: "
        f"{seasonal_band_share:.6f}"
    )

    print(
        "Harmonic fit R^2: "
        f"{harmonic['r2']:.6f}"
    )

    print(
        f"ACF lag 1:  {acf[1]:.6f}"
    )

    print(
        f"ACF lag 6:  {acf[6]:.6f}"
    )

    print(
        f"ACF lag 12: {acf[12]:.6f}"
    )

    print(
        f"ACF lag 24: {acf[24]:.6f}"
    )

    print()

    print(
        "ENCODING MEMORY AT tau=12"
    )

    print(
        "-" * 110
    )

    k12 = (
        kernel_summary[
            kernel_summary[
                "tau"
            ]
            ==
            12
        ][
            [
                "encoding",
                "S_mean",
                "F1_mean",
                "beta_mean",
                "beta_median",
                "L_half_median",
            ]
        ]
    )

    print(
        k12.to_string(
            index=False
        )
    )

    print()

    print(
        "PREDICTED MEAN beta^(L-1) AT tau=12"
    )

    print(
        "-" * 110
    )

    mem12 = (
        memory_df[
            memory_df[
                "tau"
            ]
            ==
            12
        ]
        .pivot(
            index="encoding",
            columns="depth_L",
            values="mean_beta_power",
        )
    )

    print(
        mem12.to_string()
    )

    print()

    print(
        "TOP TRAINING FREQUENCIES"
    )

    print(
        "-" * 110
    )

    print(
        top_freq
        .head(8)
        .to_string(
            index=False
        )
    )

    print()

    print(
        "OUTPUT FILES"
    )

    print(
        "-" * 110
    )

    for path in sorted(
        out.iterdir()
    ):

        print(
            path.name
        )

    print()

    print(
        "STATUS: PASS"
    )


if __name__ == "__main__":
    main()
