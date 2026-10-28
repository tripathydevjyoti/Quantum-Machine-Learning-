# Hybrid Quantum-Classical Neural Network for Time-Series Forecasting

This repository contains a hybrid quantum-classical machine learning workflow for time-series forecasting. The project combines an LSTM autoencoder for temporal representation learning with a PennyLane-based data re-uploading quantum neural network for next-step prediction.

The main demonstration notebook is:

```text
demo_notebooks/hybrid_qnn_demo.ipynb
```

## Project Overview

The workflow is designed to study whether compressed temporal features learned by a classical sequence model can be used as informative input angles for a quantum neural network.

The pipeline follows these steps:

1. Load and split the time-series dataset chronologically.
2. Train an LSTM autoencoder to reconstruct fixed-length input windows.
3. Extract latent vectors from the trained LSTM encoder.
4. Map the latent vectors to quantum rotation angles.
5. Feed the angles into a data re-uploading quantum circuit.
6. Train a hybrid quantum-classical regressor using quantum expectation values followed by a small classical output layer.
7. Evaluate forecasting performance using training, validation, and test predictions.

## Main Components

### LSTM Autoencoder

The LSTM autoencoder is used as a temporal feature extractor. It learns a compressed latent representation of each input time-series window. The reusable model and helper functions are located in:

```text
data_reupload/models/
```

Key functionality includes:

* Time-series window creation
* Fixed-range and MinMax scaling utilities
* LSTM autoencoder training
* Latent vector extraction
* Reconstruction plotting
* Checkpoint saving and loading

### Data Re-Uploading Quantum Neural Network

The quantum model is implemented using PennyLane. The circuit repeatedly applies:

* `RY` angle encoding layers for the latent features
* trainable single-qubit rotational gates
* nearest-neighbor/ring CNOT entanglement
* Pauli-Z expectation value measurements

The quantum expectation values are passed into a small classical output layer for final regression.

## Repository Structure

```text
Quantum-Machine-Learning-/
├── demo_notebooks/
│   └── hybrid_qnn_demo.ipynb
│
├── data/
│   └── raw/
│
├── data_reupload/
│   ├── models/
│   │   ├── LSTMAE.py
│   │   └── LSTMAE_pipeline.py
│   │
│   ├── experiments/
│   │   ├── tune_lstm_ae.ipynb
│   │   └── qnn_depth_sweep.ipynb
│   │
│   ├── scripts/
│   ├── checkpoints/
│   └── results/
│
├── old_experiments/
├── requirements.txt
└── README.md
```

## Current Research Direction

This project is actively evolving. The current focus is on scaling the experiments beyond the initial demonstration notebook.

Ongoing work includes:

* Hyperparameter tuning of the LSTM autoencoder
* Studying how prediction accuracy changes with data re-uploading circuit depth
* Repeating the depth analysis for different numbers of qubits
* Matching the LSTM latent dimension to the number of qubits
* Preparing HPC-compatible scripts for running larger experiment grids
* Extending the workflow to noisy simulators and hardware-aware training methods

The broader goal is to compare ideal simulation, noisy simulation, and hardware-aware training behavior for hybrid QNN time-series forecasting.

## Requirements

Core libraries used in this project include:

* Python
* NumPy
* pandas
* scikit-learn
* PyTorch
* PennyLane
* Qiskit
* matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Status

This repository is an active research and experimentation project. Code structure, experiments, and results may continue to change as the QNN depth-sweep, noisy-simulator, and hardware-oriented workflows are developed.
