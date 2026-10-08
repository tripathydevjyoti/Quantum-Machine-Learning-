# Ideal QNN Test Outputs for Inverse-CDF Decoding

## Purpose

This package contains held-out test-set outputs from the ideal/noiseless seasonal QNN forecasting experiments.

The goal is to evaluate an alternative post-hoc decoding transformation based on the inverse of the empirical CDF used in the seasonal input encoding.

The primary prediction file is:

`ideal_allseeds_test_hybrid_outputs_for_inverse_cdf.csv`

It contains predictions from:

- 3 seasonal encodings
- depths 1 through 6
- seeds 42, 43, 44, 45, 46
- 105 held-out test samples per model

This gives 90 model runs and 9,450 prediction rows in total.

A second file,

`training_reference_empirical_cdf.csv`

contains the training-only precipitation reference used to construct the empirical CDF.

---

## Model outputs

The hybrid quantum + classical model produces a final bounded output:

```math
z_{\mathrm{pred}} \in [-1,1]
```

In the prediction CSV, this is the column:

`z_pred`

For the proposed inverse-CDF decoder, first map this value to the interval [0,1]:

```math
u_{\mathrm{pred}}
=
\frac{z_{\mathrm{pred}} + 1}{2}
```

This value is already included in the CSV as:

`u_pred`

The intended alternative decoding is then:

```math
y_{\mathrm{pred}}^{(\mathrm{inverse\ CDF})}
=
F^{-1}(u_{\mathrm{pred}})
```

where `F^{-1}` is constructed from the supplied empirical-CDF training reference.

---

## Important interpretation note

The inverse-CDF transformation is being evaluated here as a post-hoc alternative decoder for already-trained models.

During training, the forecast target itself was **not** transformed through the empirical CDF.

Instead, the raw precipitation target was linearly scaled from the range [0,350] to [-1,1]:

```math
z_{\mathrm{true}}
=
-1
+
\frac{y_{\mathrm{raw}} - 0}{350 - 0}
\times 2
```

Equivalently,

```math
z_{\mathrm{true}}
=
2\frac{y_{\mathrm{raw}}}{350}
-
1
```

The model was trained to predict this linearly scaled target.

The empirical CDF transformation was used for the seasonal input encoding, not for the target variable.

Therefore, applying

```math
F^{-1}
\left(
\frac{z_{\mathrm{pred}} + 1}{2}
\right)
```

should be interpreted as a post-hoc decoder experiment rather than the exact inverse of the target transformation used during training.

---

## Training loss

The training objective was mean squared error in the scaled target space:

```math
\mathcal{L}
=
\frac{1}{N}
\sum_i
\left(
z_{\mathrm{true},i}
-
z_{\mathrm{pred},i}
\right)^2
```

The model output was bounded to [-1,1] using the output tanh.

The current raw-space prediction was obtained only after model evaluation using the linear inverse scaling described below.

---

## Current linear decoder

The existing decoding step maps the model output from [-1,1] back to the raw precipitation range [0,350]:

```math
y_{\mathrm{pred}}^{(\mathrm{linear})}
=
175
\left(
z_{\mathrm{pred}} + 1
\right)
```

This value is included in the prediction CSV as:

`y_pred_linear`

It is provided as the current baseline decoder for comparison against the proposed inverse-CDF decoder.

---

## Seasonal input preprocessing

For the seasonal encoding models, each input lag is transformed using an empirical CDF constructed from a fixed training reference set.

For the production setup:

- window size = 14
- training window count = 350
- empirical-CDF reference observations = original time-series indices 0 through 363 inclusive
- reference size = 364 observations

The same frozen training reference is used for validation and test input encoding.

The three seasonal encodings included are:

1. `seasonal_meridian`
2. `learnable_seasonal_cdf`
3. `learnable_seasonal_cdf_rz`

All three use the same underlying empirical-CDF input preprocessing.

---

## Empirical CDF reference file

The accompanying file

`training_reference_empirical_cdf.csv`

contains the 364 training-only precipitation observations used to construct the empirical CDF for the seasonal input encoding.

For reconstructing the CDF, the main columns of interest are:

- `pp_raw`: raw precipitation value in millimeters
- `cdf`: empirical CDF value assigned to that precipitation value

The empirical CDF used in the production experiments was the right-continuous empirical CDF:

```math
F_{\mathrm{train}}(x)
=
\frac{\#\{y_s \le x\}}{364}
```

where the reference set consists of the first 364 observations of the original precipitation time series, corresponding to original indices 0 through 363 inclusive.

The production implementation was:

```python
sorted_reference = np.sort(
    training_reference.copy()
)

counts = np.searchsorted(
    sorted_reference,
    values,
    side="right",
)

cdf = (
    counts.astype(np.float64)
    / float(len(sorted_reference))
)
```

Here:

- `training_reference` is the fixed 364-observation training-only precipitation reference set
- `values` contains the raw precipitation values at which the empirical CDF is evaluated
- `sorted_reference` is the same 364-value reference ordered from smallest to largest

The CDF is therefore evaluated directly on raw precipitation values in millimeters.

### Columns in `training_reference_empirical_cdf.csv`

#### `original_index`

Index of the observation in the original chronological precipitation time series.

Values run from 0 through 363.

#### `date`

Date corresponding to the original precipitation observation.

The reference spans January 1981 through April 2011.

#### `pp_raw`

Raw precipitation observation in millimeters.

#### `cdf`

Right-continuous empirical CDF evaluated at `pp_raw` using the fixed 364-observation reference:

```math
F_{\mathrm{train}}(x)
=
\frac{\#\{y_s \le x\}}{364}
```

#### `sorted_index`

Zero-based position in a separately sorted view of the 364-value reference distribution.

Values run from 0 through 363.

#### `sorted_pp_raw`

Precipitation value at the corresponding `sorted_index` after arranging all 364 reference observations from smallest to largest.

The `sorted_index` and `sorted_pp_raw` columns form a separate sorted view of the same 364 observations. They should not be interpreted as corresponding row-by-row to `original_index`, `date`, or `pp_raw`.

For reconstructing the forward empirical CDF, the primary columns are `pp_raw` and `cdf`. The `sorted_pp_raw` column additionally provides the ordered empirical reference distribution underlying the CDF and can be used when constructing the inverse-CDF transformation.

---

## Prediction CSV columns

The following columns are contained in:

`ideal_allseeds_test_hybrid_outputs_for_inverse_cdf.csv`

### `encoding`

Seasonal encoding used by the QNN.

Possible values:

- `seasonal_meridian`
- `learnable_seasonal_cdf`
- `learnable_seasonal_cdf_rz`

### `depth`

Number of data-reuploading layers.

Values:

1 through 6.

### `seed`

Training random seed.

Values:

42, 43, 44, 45, 46.

### `split`

Always:

`test`

Only held-out test predictions are included in this export.

### `split_pos`

Position of the observation within the test split.

Values run from 0 to 104.

### `target_index`

Index of the forecast target in the original time series.

This can be used to align predictions with the original precipitation series.

### `y_true_raw`

Observed precipitation value in the original data units.

### `z_true`

Observed target after the original linear scaling from [0,350] to [-1,1].

### `z_pred`

Final hybrid-model prediction in [-1,1].

This is the model output prior to the current linear inverse scaling.

### `u_pred`

Model prediction rescaled from [-1,1] to [0,1]:

```math
u_{\mathrm{pred}}
=
\frac{z_{\mathrm{pred}} + 1}{2}
```

This is the quantity intended as input to the inverse-CDF function.

### `y_pred_linear`

Current prediction in the raw precipitation range obtained using the existing linear inverse scaling:

```math
y_{\mathrm{pred}}^{(\mathrm{linear})}
=
175
\left(
z_{\mathrm{pred}} + 1
\right)
```

---

## Requested inverse-CDF output

Please apply the inverse-CDF transformation to:

`u_pred`

and return the corresponding raw-space prediction as a new column, preferably named:

`y_pred_inverse_cdf`

Please preserve all existing identifying columns so that predictions can be matched exactly by:

- encoding
- depth
- seed
- split position
- target index

The intended transformation is:

```math
y_{\mathrm{pred}}^{(\mathrm{inverse\ CDF})}
=
F^{-1}(u_{\mathrm{pred}})
```

using the empirical distribution supplied in:

`training_reference_empirical_cdf.csv`

---

## Evaluation

After applying the inverse-CDF decoder, the resulting raw-space predictions will be compared against `y_true_raw`.

The main evaluation metrics used in the project are:

- RMSE
- Pearson correlation
- FFT spectral cosine similarity
- predicted / true standard deviation ratio

The inverse-CDF decoder will be compared directly against the existing `y_pred_linear` baseline on the same held-out test samples.
