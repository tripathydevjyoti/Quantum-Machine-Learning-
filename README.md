# Hybrid QNN Time-Series Forecasting

This repository contains an initial hybrid quantum-classical neural network workflow for time-series forecasting.

The main notebook, `hybrid qnn.ipynb`, builds a pipeline that first trains an LSTM autoencoder on the input time series and then uses the learned latent representation as input angles for a data re-uploading quantum neural network.

## Main Workflow

The notebook includes:

- Loading and splitting the time-series dataset
- Training an LSTM autoencoder using the utilities in `models/LSTMAE_pipeline.py`
- Extracting latent vectors from the trained LSTM encoder
- Mapping the latent vectors to quantum rotation angles
- Building a PennyLane-based data re-uploading QNN
- Training a hybrid model with quantum expectation values followed by a small classical output layer
- Plotting training loss, validation predictions, and basic validation metrics

## Repository Notes

The `models/` folder contains the LSTM autoencoder pipeline and related helper functions used by the main notebook.

The data re-uploading circuit is implemented using repeated `RY` input encoding layers and trainable rotational ansatz layers with nearest-neighbor/ring CNOT entanglement.

## Current Status

🚧 This work is currently in progress.

The next stage is focused on running the quantum circuit on noisy simulators and eventually hardware backends. Initial benchmark plots and experimental outputs are being collected in the qubit-specific result folders, for example files such as:

- `spsa_predictions_and_losses_3_qubit`
- related SPSA prediction/loss plots in the two-qubit and three-qubit experiment folders

These benchmarks are intended to compare ideal simulation, noisy simulation, and hardware-aware training behavior.

## Requirements

Core libraries used in the notebook include:

- Python
- NumPy
- pandas
- PyTorch
- PennyLane
- Qiskit
- matplotlib

## Status

This repository is experimental and actively evolving. Code structure, results, and documentation may change as the noisy-simulator and hardware experiments are added.
