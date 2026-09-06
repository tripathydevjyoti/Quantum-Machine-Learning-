# Quantum Learning for Geospatial Time Series: Seasonal Encoding and Noise-Aware Algorithms

Research software and reproducibility materials for quantum-neural-network (QNN) forecasting of monthly geospatial time series. The project studies how time-aware quantum data encodings, circuit depth, and realistic finite-shot device noise affect forecasting performance and learned quantum-state geometry.

The associated manuscript is in preparation. Publication identifiers and the final manuscript citation will be added when available.

## Study overview

The forecasting models operate directly on 14-step input windows using 14-qubit data-reuploading circuits. The repository contains five encoding families:

- `RY` baseline encoding
- same-scalar `RX-RY-RZ` encoding
- fixed seasonal-meridian encoding
- learnable seasonal-CDF encoding
- learnable seasonal-CDF encoding with an additional `RZ` component

The experimental workflow includes:

- ideal state-vector simulations across circuit depths 1–6 and five random seeds;
- finite-shot, hardware-informed noisy simulations;
- SPSA-to-Adam noisy-training workflows;
- exact temporal-fidelity, spectral, and quantum-geometric analyses;
- validation and aggregation scripts for the manuscript tables; and
- portable Slurm entry points for HPC reproduction.

The curated ideal study contains 150 completed runs: five encodings, six depths, and five seeds. Noisy Aachen simulations are still in progress, so noisy results will be added only after the corresponding experiment blocks and validation checks are complete.

## Repository layout

```text
.
├── data/
│   ├── raw/                         # study CSV and provenance notes
│   └── processed/                   # preprocessing configuration and metadata
├── data_reupload/
│   ├── scripts/final_sweep/         # canonical ideal QNN training programs
│   ├── noisy_direct14/              # finite-shot noisy workflows
│   ├── noisy_aachen/                # Aachen backend snapshot and workers
│   ├── analysis/                    # forecasting, geometry, and validation analyses
│   ├── report/                      # manuscript-asset generation
│   └── slurm/                       # ideal and analysis Slurm entry points
├── docs/                            # data and reproduction guides
├── results/
│   ├── tables/ideal/                # curated machine-readable ideal results
│   └── figures/                     # publication figures added separately
├── scripts/                         # dataset-preparation utilities
└── legacy/                          # archived pre-paper experiments
```

The active paper implementation is at the repository root. Earlier LSTM-autoencoder and exploratory hybrid-QNN work is preserved under [`legacy/`](legacy/) for historical context and is not part of the canonical paper workflow.

## Data

The included study file is:

```text
data/raw/SPEI_AllScales_Napak - SPEI_AllScales_Napak.csv
```

The experiments use the monthly `precip_mm` series, chronological train/validation/test splitting, and input windows of length 14. Dataset provenance, integrity information, preprocessing rules, and split metadata are documented in:

- [`data/raw/README.md`](data/raw/README.md)
- [`docs/data.md`](docs/data.md)
- [`data/processed/precip_mm_windowed_supervised/`](data/processed/precip_mm_windowed_supervised/)

The dataset is included with permission from the contributing coauthor. It is not covered by the repository's MIT software license; see [`NOTICE.md`](NOTICE.md).

## Reproducing the work

Start with [`docs/reproducibility.md`](docs/reproducibility.md). It describes three levels of execution:

1. environment and import checks;
2. representative ideal and noisy smoke runs; and
3. full HPC reproduction through the supplied Slurm jobs.

The mapping from each experiment family to its training code, scheduler entry point, analysis workflow, and curated outputs is provided in [`docs/experiment-map.md`](docs/experiment-map.md).

Environment specifications include:

- [`requirements_hpc.txt`](requirements_hpc.txt) for the ideal/HPC workflow;
- [`data_reupload/noisy_direct14/scripts/requirements_noisy_qml_core.txt`](data_reupload/noisy_direct14/scripts/requirements_noisy_qml_core.txt) for the core noisy environment; and
- exact environment snapshots under [`data_reupload/noisy_aachen/environment/`](data_reupload/noisy_aachen/environment/).

All maintained Python and Slurm workflows resolve the repository through `QML_PROJECT_ROOT` when it is set and otherwise infer the root from the script location. This avoids dependence on a particular user's cluster path.

## Results

Curated ideal result tables are stored under [`results/tables/ideal/`](results/tables/ideal/), organized into forecasting, geometry, validation, and encoding-specific diagnostic outputs. The repository intentionally excludes raw checkpoints, large prediction arrays, scheduler logs, caches, and smoke-test outputs.

See [`results/README.md`](results/README.md) for the publication-results policy. Figures can be added later under [`results/figures/`](results/figures/) once the manuscript set is finalized.

## Authors and contributions

Software implementation and repository maintenance:

- Devjyoti Tripathy — Department of Physics and Quantum Science Institute, University of Maryland, Baltimore County

Associated manuscript authors:

- Devjyoti Tripathy
- Reece Robertson
- Josey Stevens
- Catherine Lilian Nakalembe
- Sebastian Deffner

The manuscript author list reflects the broader scientific collaboration. The software authorship metadata in [`CITATION.cff`](CITATION.cff) identifies Devjyoti Tripathy as the sole implementation contributor to this repository.

## Citation

If you use this software, please cite the repository using [`CITATION.cff`](CITATION.cff). The associated manuscript is currently listed there as an unpublished work in preparation. Its arXiv identifier, journal information, and final citation will be added when available.

## License

Original code and documentation in this repository are released under the [MIT License](LICENSE). The study dataset, third-party dependencies, backend snapshot, and associated manuscript are subject to the separate terms described in [`NOTICE.md`](NOTICE.md).
