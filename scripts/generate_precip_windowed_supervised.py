from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

REPO_ROOT = Path(
    "/umbc/rs/pi_deffner/users/devjyot1/projects/Quantum-Machine-Learning-"
)

INPUT_CSV = (
    REPO_ROOT
    / "data"
    / "raw"
    / "SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "precip_mm_windowed_supervised"
)

VALUE_COL = "precip_mm"
WINDOW = 14

# Exact chronological target split used in the QML experiments.
SPLITS = {
    "train": (14, 363),
    "validation": (364, 433),
    "test": (434, 538),
}


# ============================================================
# Load raw series
# ============================================================

print(f"Reading: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

if VALUE_COL not in df.columns:
    raise ValueError(
        f"Column '{VALUE_COL}' not found.\n"
        f"Available columns: {list(df.columns)}"
    )

series = df[VALUE_COL].to_numpy(dtype=np.float64)

if len(series) != 539:
    raise ValueError(
        f"Expected 539 observations, found {len(series)}."
    )

if not np.isfinite(series).all():
    bad = np.where(~np.isfinite(series))[0]
    raise ValueError(
        f"Found missing/nonfinite precip_mm values at indices: {bad.tolist()}"
    )

print(f"Loaded {len(series)} valid precip_mm observations.")


# ============================================================
# Generate supervised windows
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Oldest observation first:
# lag_14 = 14 months before target
# lag_1  = immediately preceding month
lag_cols = [f"lag_{lag}" for lag in range(WINDOW, 0, -1)]

generated = {}


for split_name, (first_target, last_target) in SPLITS.items():

    target_indices = np.arange(
        first_target,
        last_target + 1,
        dtype=int
    )

    X = np.stack(
        [
            series[t - WINDOW:t]
            for t in target_indices
        ],
        axis=0,
    )

    y = series[target_indices]

    expected_n = last_target - first_target + 1

    assert X.shape == (expected_n, WINDOW)
    assert y.shape == (expected_n,)

    # --------------------------------------------------------
    # Build CSV
    # --------------------------------------------------------

    split_df = pd.DataFrame(
        X,
        columns=lag_cols
    )

    split_df.insert(
        0,
        "target_index",
        target_indices
    )

    if "date" in df.columns:
        split_df.insert(
            1,
            "target_date",
            df.iloc[target_indices]["date"].astype(str).to_numpy()
        )

    split_df["target"] = y

    csv_path = OUTPUT_DIR / f"{split_name}.csv"
    split_df.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Save ready-to-use NumPy arrays
    # --------------------------------------------------------

    x_path = OUTPUT_DIR / f"X_{split_name}.npy"
    y_path = OUTPUT_DIR / f"y_{split_name}.npy"

    np.save(x_path, X)
    np.save(y_path, y)

    generated[split_name] = {
        "X": X,
        "y": y,
        "df": split_df,
    }

    print(
        f"{split_name:10s}: "
        f"X={X.shape}, y={y.shape}, "
        f"targets={first_target}..{last_target}"
    )


# ============================================================
# Exhaustive validation against original precip_mm series
# ============================================================

print("\nRunning exhaustive validation...")

for split_name, (first_target, last_target) in SPLITS.items():

    split_df = generated[split_name]["df"]
    X = generated[split_name]["X"]
    y = generated[split_name]["y"]

    expected_indices = list(
        range(first_target, last_target + 1)
    )

    assert split_df["target_index"].tolist() == expected_indices

    for row_i, t in enumerate(expected_indices):

        expected_X = series[t - WINDOW:t]
        expected_y = series[t]

        stored_X = X[row_i]
        stored_y = y[row_i]

        if not np.allclose(
            stored_X,
            expected_X,
            rtol=0,
            atol=1e-12,
        ):
            raise AssertionError(
                f"{split_name}: input mismatch at target {t}"
            )

        if not np.isclose(
            stored_y,
            expected_y,
            rtol=0,
            atol=1e-12,
        ):
            raise AssertionError(
                f"{split_name}: target mismatch at index {t}"
            )


# ============================================================
# Explicit boundary checks
# ============================================================

train = generated["train"]
val = generated["validation"]
test = generated["test"]

# First training example:
# source[0:14] -> source[14]
assert np.allclose(
    train["X"][0],
    series[0:14]
)
assert train["y"][0] == series[14]

# First validation example:
# source[350:364] -> source[364]
assert np.allclose(
    val["X"][0],
    series[350:364]
)
assert val["y"][0] == series[364]

# First test example:
# source[420:434] -> source[434]
assert np.allclose(
    test["X"][0],
    series[420:434]
)
assert test["y"][0] == series[434]


# ============================================================
# Verify target splits do not overlap
# ============================================================

train_targets = set(
    generated["train"]["df"]["target_index"]
)
val_targets = set(
    generated["validation"]["df"]["target_index"]
)
test_targets = set(
    generated["test"]["df"]["target_index"]
)

assert train_targets.isdisjoint(val_targets)
assert train_targets.isdisjoint(test_targets)
assert val_targets.isdisjoint(test_targets)

all_targets = (
    list(generated["train"]["df"]["target_index"])
    + list(generated["validation"]["df"]["target_index"])
    + list(generated["test"]["df"]["target_index"])
)

assert all_targets == list(range(14, 539))


# ============================================================
# README
# ============================================================

readme = """
Raw precipitation supervised forecasting dataset
=================================================

Source variable
---------------
precip_mm

Task
----
One-step-ahead monthly precipitation forecasting using a
14-observation input window.

For target index t:

    X_t = [
        precip_mm[t-14],
        precip_mm[t-13],
        ...
        precip_mm[t-1]
    ]

    y_t = precip_mm[t]


Lag ordering
------------
lag_14 = oldest value in the input window
lag_1  = immediately preceding observation
target = one-step-ahead precipitation value


Chronological split
-------------------
Training:
    target indices 14..363
    X_train shape = (350, 14)
    y_train shape = (350,)

Validation:
    target indices 364..433
    X_validation shape = (70, 14)
    y_validation shape = (70,)

Test:
    target indices 434..538
    X_test shape = (105, 14)
    y_test shape = (105,)


Boundary examples
-----------------
First validation sample:

    input  = original indices 350..363
    target = original index 364

First test sample:

    input  = original indices 420..433
    target = original index 434


Preprocessing
-------------
All values are RAW precip_mm values.

No normalization, standardization, empirical-CDF transform,
seasonal encoding, or target scaling has been applied.

Any learned preprocessing must be fit on the TRAINING DATA ONLY
and then applied unchanged to validation and test.


Files
-----
train.csv
validation.csv
test.csv

X_train.npy
y_train.npy

X_validation.npy
y_validation.npy

X_test.npy
y_test.npy


CSV metadata
------------
target_index:
    Zero-based target index in the original 539-observation series.

target_date:
    Original target date if the source CSV contains a date column.

These metadata columns are not intended as model features unless
explicitly desired.


Reproducibility
---------------
Samples remain in chronological order and are not shuffled.
"""

with open(
    OUTPUT_DIR / "README.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(readme.strip() + "\n")


# ============================================================
# Final report
# ============================================================

print("\n========================================")
print("ALL VALIDATION CHECKS PASSED")
print("========================================")
print(f"Output directory:\n{OUTPUT_DIR}")

print("\nCreated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {path.name}")

