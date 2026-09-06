# Dataset and forecasting task

## Raw dataset

The study uses the monthly hydroclimatic table stored at:

`data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv`

The public copy contains 539 data rows and the following columns:

`region`, `date`, `year`, `month`, `precip_mm`, `pet_mm`, `water_balance`, `water_balance_1m`, `water_balance_3m`, `water_balance_6m`, `water_balance_9m`, `water_balance_12m`, `SPEI_1m`, `SPEI_3m`, `SPEI_6m`, `SPEI_9m`, and `SPEI_12m`.

The forecasting target is `precip_mm`, which is column 5 in the CSV and index 4 under the scripts' zero-based `value_col` convention.

Integrity metadata for the normalized public copy:

- Size: 102,454 bytes
- Data rows: 539
- SHA-256: `12412111897b8827ab83e5991535ba1ab19f1719bb71452ebb47ca344d05e56e`

The dataset was supplied by a manuscript coauthor with permission to include it in the repository. Formal source attribution and licensing information remain to be added when the manuscript data statement is finalized. See `data/raw/README.md` for the current notice.

## Supervised forecasting construction

For target index `t`, each input contains the preceding 14 precipitation observations:

```text
X_t = [precip_mm[t-14], ..., precip_mm[t-1]]
y_t = precip_mm[t]
```

The horizon is one month. Windows remain in chronological order.

| Split | Target indices | Samples |
| --- | ---: | ---: |
| Training | 14-363 | 350 |
| Validation | 364-433 | 70 |
| Test | 434-538 | 105 |

The corresponding experiment arguments are `window_size=14`, `train_window_end=350`, and `train_end_idx=434`.

## Scaling and leakage controls

The committed raw and generated supervised CSV files contain unscaled precipitation values. QNN training maps the physical target range `[0, 350]` mm to `[-1, 1]` and uses a bounded `tanh` output:

```text
y_scaled = 2 y / 350 - 1
y_mm = 175 (y_scaled + 1)
```

Any learned transform, including empirical-CDF components of the seasonal encodings, must be fit using training data only and then applied unchanged to validation and test data.

## Generate the supervised splits

From the repository root on a machine with the project environment active:

```bash
python scripts/generate_precip_windowed_supervised.py
```

On UMBC CHIP, submit the prepared CPU job instead of running Python on a login node:

```bash
mkdir -p logs data/processed
sbatch scripts/generate_precip_windowed_supervised.sbatch
```

Generated arrays and split CSVs are intentionally ignored by Git. Their expected shapes and configuration are recorded in `data/processed/precip_mm_windowed_supervised/README.txt` and `training_config.json`.

## Alternate dataset location

The committed dataset is used by default. To use an equivalent copy elsewhere, set:

```bash
export QML_DATA_PATH=/absolute/path/to/dataset.csv
```

Do not change the target column, row ordering, split boundaries, or scaling parameters when reproducing the reported study.
