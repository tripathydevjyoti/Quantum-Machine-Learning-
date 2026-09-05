#!/bin/bash
#SBATCH --job-name=install_noisy_qml
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/umbc/rs/pi_deffner/users/devjyot1/projects/Quantum-Machine-Learning-/data_reupload/noisy_direct14/logs/%x_%j.out
#SBATCH --error=/umbc/rs/pi_deffner/users/devjyot1/projects/Quantum-Machine-Learning-/data_reupload/noisy_direct14/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/umbc/rs/pi_deffner/users/devjyot1/projects/Quantum-Machine-Learning-
NOISY_ROOT="$PROJECT_ROOT/data_reupload/noisy_direct14"
ENV_DIR="$PROJECT_ROOT/venv_noisy_qml"
REQ_FILE="$NOISY_ROOT/scripts/requirements_noisy_qml_core.txt"

cd "$PROJECT_ROOT"

module purge
module load Python/3.11.5-GCCcore-13.2.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "============================================================"
echo "Installing isolated noisy-QML environment"
echo "============================================================"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Module Python: $(which python)"
python --version
echo "Environment: $ENV_DIR"
echo "Requirements: $REQ_FILE"

test -f "$REQ_FILE"

if [[ -e "$ENV_DIR" ]]; then
    echo "ERROR: $ENV_DIR already exists."
    echo "Remove or rename it explicitly before rerunning this installer."
    exit 2
fi

python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# CPU-only PyTorch is sufficient for the chip-cpu partition.
python -m pip install \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r "$REQ_FILE"

echo
echo "===== PIP CHECK ====="
python -m pip check

echo
echo "===== VERSION CHECK ====="
python - <<'PY'
import importlib
import sys

expected = {
    "numpy": "1.26.4",
    "pandas": "3.0.2",
    "torch": "2.2.2",
    "pennylane": "0.44.1",
    "pennylane_qiskit": "0.44.1",
    "qiskit": "2.2.3",
    "qiskit_aer": "0.17.2",
    "qiskit_ibm_runtime": "0.45.1",
    "autoray": "0.8.2",
    "rustworkx": "0.17.1",
}

print("Python executable:", sys.executable)
print("Python version:", sys.version)

failed = []

for name, wanted in expected.items():
    try:
        module = importlib.import_module(name)
        found = getattr(module, "__version__", "version unavailable")
        print(f"{name}: {found} (expected {wanted})")

        # torch CPU wheels may display 2.2.2+cpu.
        if name == "torch":
            ok = str(found).split("+")[0] == wanted
        else:
            ok = str(found) == wanted

        if not ok:
            failed.append(f"{name}: found {found}, expected {wanted}")

    except Exception as exc:
        failed.append(f"{name}: import failed: {exc}")
        print(f"{name}: IMPORT FAILED — {exc}")

if failed:
    raise SystemExit(
        "Version/import verification failed:\n- "
        + "\n- ".join(failed)
    )
PY

echo
echo "===== PENNYLANE/QISKIT AER INTEGRATION TEST ====="
python - <<'PY'
import inspect
import numpy as np
import pennylane as qml
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import qiskit_ibm_runtime.fake_provider as fake_provider

n_qubits = 14
candidates = []

for name in dir(fake_provider):
    if not name.startswith("Fake"):
        continue

    obj = getattr(fake_provider, name)
    if not inspect.isclass(obj):
        continue

    try:
        backend = obj()
        nq = getattr(backend, "num_qubits", None)
        if nq is not None and nq >= n_qubits:
            candidates.append((nq, name, backend))
    except Exception:
        pass

if not candidates:
    raise RuntimeError(
        "No qiskit_ibm_runtime fake backend with at least "
        f"{n_qubits} qubits was found."
    )

candidates.sort(key=lambda item: (item[0], item[1]))
nq, class_name, fake_backend = candidates[0]

print(
    "Selected fake backend:",
    class_name,
    "with",
    nq,
    "qubits",
)

noise_model = NoiseModel.from_backend(fake_backend)
aer_backend = AerSimulator(
    noise_model=noise_model,
    basis_gates=noise_model.basis_gates,
    seed_simulator=42,
)
aer_backend.set_options(
    max_parallel_threads=1,
    max_parallel_experiments=1,
    max_parallel_shots=1,
)

dev = qml.device(
    "qiskit.aer",
    wires=n_qubits,
    backend=aer_backend,
    optimization_level=1,
    seed_transpiler=42,
)

@qml.set_shots(32)
@qml.qnode(dev, interface=None, diff_method=None)
def test_circuit():
    qml.RY(0.2, wires=0)
    qml.CNOT(wires=[0, 1])
    return [
        qml.expval(qml.PauliZ(0)),
        qml.expval(qml.PauliZ(1)),
    ]

result = test_circuit()
print("QNode result:", result)

if len(result) != 2:
    raise RuntimeError("Unexpected QNode result length.")

if not np.all(np.isfinite([float(v) for v in result])):
    raise RuntimeError("QNode returned non-finite values.")

print("PennyLane-Qiskit Aer integration succeeded.")
PY

echo
echo "===== FREEZE ====="
python -m pip freeze > "$NOISY_ROOT/scripts/venv_noisy_qml_linux_freeze.txt"
echo "Saved: $NOISY_ROOT/scripts/venv_noisy_qml_linux_freeze.txt"

echo
echo "Finished successfully: $(date)"
