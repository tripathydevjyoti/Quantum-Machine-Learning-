# Raw data

## Study dataset

This repository includes the monthly hydroclimatic dataset used by the QNN forecasting experiments:

`SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv`

The dataset was provided by a coauthor of the associated manuscript with permission to include it in this repository. Formal source attribution and licensing information will be added when the manuscript and data statement are finalized. Until then, inclusion should not be interpreted as granting a separate general-purpose data license.

## Integrity

- File size: 102,454 bytes
- Data rows: 539
- Header rows: 1
- SHA-256: `12412111897b8827ab83e5991535ba1ab19f1719bb71452ebb47ca344d05e56e`

The public copy uses Unix line endings. This differs byte-for-byte from the original coauthor-provided file but preserves all records and scientific values.

## Forecasting target

The forecasting target is `precip_mm`, the fifth CSV column. Experiment scripts use the zero-based setting `value_col=4`.

Each forecasting input contains the preceding 14 monthly precipitation observations, and its target is the following observation.

## Data path

The workflows use `data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv` by default.

If the dataset is stored elsewhere, its location can optionally be supplied through the `QML_DATA_PATH` environment variable.

Processed split definitions and preprocessing metadata are available in `data/processed/precip_mm_windowed_supervised/`.