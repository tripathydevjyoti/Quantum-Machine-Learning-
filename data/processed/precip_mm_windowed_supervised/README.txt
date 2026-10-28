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
