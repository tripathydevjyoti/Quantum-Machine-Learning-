# Reproducibility guide

## Scope

The repository supports three levels of execution:

1. A representative ideal run to verify the data and circuit pipeline.
2. The full ideal study: five encodings, six depths, and five seeds (150 runs).
3. Finite-shot noisy workflows, including a smoke test and production Slurm arrays.

The curated ideal outputs are already available under `results/tables/ideal/`. Large checkpoints, prediction arrays, logs, and raw run directories are regenerated locally and remain ignored by Git.

All commands below assume the current working directory is the repository root.

## Path configuration

Most users need no path configuration. The scripts use repository-relative defaults. These optional environment variables override them:

| Variable | Purpose |
| --- | --- |
| `QML_PROJECT_ROOT` | Repository root used by Python and Slurm workflows |
| `QML_DATA_PATH` | Alternate path to the raw CSV |
| `QML_VENV` | Alternate Python virtual environment |
| `QML_CACHE_ROOT` | Runtime cache location for workflows that use a cache |
| `QML_TMP_ROOT` | Temporary-file location for workflows that use scratch space |

For example:

```bash
export QML_PROJECT_ROOT="$PWD"
```

When Slurm jobs are submitted from the repository root, `SLURM_SUBMIT_DIR` provides the fallback and this export is not required.

## Environment

### Ideal simulations and analysis

The compact HPC dependency specification is `requirements_hpc.txt`. A local environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_hpc.txt
```

The ideal experiments use PennyLane's `default.qubit` device, analytic expectation values, PyTorch, and backpropagation.

On UMBC CHIP, the production scripts expect:

```bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
source "${QML_VENV:-$PWD/venv_req}/bin/activate"
```

Python environments linked to the cluster module should be used through `chip-cpu` Slurm jobs, not on login nodes.

### Finite-shot noisy simulations

The noisy core requirements are recorded in `data_reupload/noisy_direct14/scripts/requirements_noisy_qml_core.txt`; the exact completed-run freeze is retained beside it for provenance. On UMBC CHIP, the prepared installer is:

```bash
mkdir -p data_reupload/noisy_direct14/logs
sbatch data_reupload/noisy_direct14/slurm/install_noisy_qml_env.slurm
```

The Aachen backend snapshot and environment records are under `data_reupload/noisy_aachen/`. These preserve the Qiskit IBM Runtime source notice and the exact fake-backend configuration used by the study.

## Verify the input

```bash
sha256sum 'data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv'
```

Expected SHA-256:

```text
12412111897b8827ab83e5991535ba1ab19f1719bb71452ebb47ca344d05e56e
```

See `docs/data.md` for the task definition, split boundaries, and scaling rules.

## Prepare runtime directories

Create log locations before submitting jobs because Slurm resolves output paths at submission or job startup:

```bash
mkdir -p \
  logs \
  data_reupload/slurm/logs \
  data_reupload/noisy_direct14/logs
```

All of these locations are ignored by Git.

## Representative ideal run

The following command reproduces the standard-Ry, depth-1, seed-42 configuration. Run it inside a CPU allocation with the ideal environment active:

```bash
python -u data_reupload/scripts/final_sweep/train_final_single_qnn.py \
  --csv_path 'data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv' \
  --value_col 4 \
  --train_end_idx 434 \
  --window_size 14 \
  --train_window_end 350 \
  --scaler_data_min 0 \
  --scaler_data_max 350 \
  --scaler_feature_min -1 \
  --scaler_feature_max 1 \
  --depth 1 \
  --n_ansatz_layers 1 \
  --readout all \
  --seed 42 \
  --qnn_epochs 100 \
  --qnn_learning_rate 0.01 \
  --qnn_batch_size 16 \
  --angle_factor 3.141592653589793 \
  --qml_device default.qubit \
  --diff_method backprop \
  --device cpu \
  --use_output_tanh \
  --print_every 10 \
  --variant_id single_tanh_a1 \
  --output_dir data_reupload/results/reproduction
```

Outputs are written beneath `data_reupload/results/reproduction/` and are ignored by Git.

## Full ideal study on UMBC CHIP

The full production sweep uses seeds 42-46 and depths 1-6. Submit from the repository root:

```bash
sbatch data_reupload/slurm/final_single_tanh_a1_multiseed_d1_d3.sbatch
sbatch data_reupload/slurm/final_single_tanh_a1_multiseed_d4_d6.sbatch
sbatch data_reupload/slurm/run_same_scalar_rxryrz_a1_full.sbatch
sbatch data_reupload/slurm/run_seasonal_meridian_a1_full.sbatch
sbatch data_reupload/slurm/run_learnable_seasonal_cdf_a1_full.sbatch
sbatch data_reupload/slurm/run_learnable_seasonal_cdf_rz_a1_full.sbatch
```

Do not resubmit production arrays into nonempty run directories without first reviewing the safeguards in the corresponding Slurm file. Several launchers intentionally refuse to overwrite completed or partial runs.

After all 150 runs are complete, build the master tables:

```bash
sbatch data_reupload/slurm/build_five_encoding_master.sbatch
```

The curated publication-facing copies are stored under `results/tables/ideal/forecasting/` and `results/tables/ideal/validation/`.

## Geometry and temporal fidelity

The exact temporal-fidelity evaluation is split across the first 120 runs and the 30 runs for `learnable_seasonal_cdf_rz`:

```bash
sbatch data_reupload/slurm/run_exact_temporal_fidelity_all120.sbatch
sbatch data_reupload/slurm/run_exact_temporal_fidelity_model5.sbatch
```

After both arrays complete:

```bash
sbatch data_reupload/slurm/aggregate_exact_temporal_fidelity.sbatch
```

Curated geometry summaries are in `results/tables/ideal/geometry/`.

## Noisy smoke test and FakeMelbourneV2 workflow

Run the finite-shot core smoke test before any production array:

```bash
sbatch data_reupload/noisy_direct14/slurm/smoke_noisy_seasonal_core.slurm
```

The production launchers use 512-shot Qiskit Aer simulation with `FakeMelbourneV2` and the two-pass SPSA-Adam procedure:

```bash
sbatch data_reupload/noisy_direct14/slurm/production_noisy_seasonal_array.slurm
sbatch data_reupload/noisy_direct14/slurm/production_noisy_seasonal_depth456_array.slurm
```

Generated noisy outputs are ignored by Git. Aggregate and compare them only after verifying the expected completion matrix:

```bash
sbatch data_reupload/noisy_direct14/slurm/aggregate_noisy_seasonal_results.slurm
sbatch data_reupload/noisy_direct14/slurm/compare_ideal_and_noisy.slurm
```

## Aachen workflow status

The repository includes the Aachen backend snapshot, environment records, and canonical worker implementations. The production launcher and curated Aachen results will be documented and added only after the active simulation blocks are complete and validated. Do not interpret the absence of Aachen result files as a failed or empty study.

## Validation expectations

Before treating regenerated output as equivalent to the reported study, verify:

- Dataset checksum, target column, and chronological split boundaries.
- Encoding name, depth, and seed.
- Device, shot count, and fake-backend snapshot for noisy runs.
- Training hyperparameters and optimizer schedule.
- Completion and validation metadata written by each launcher.
- Aggregate run count and consistency audits before generating figures.

The code-to-result relationships are summarized in `docs/experiment-map.md`.
