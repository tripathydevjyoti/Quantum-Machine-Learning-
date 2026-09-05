#!/bin/bash
#SBATCH --job-name=qml_env_check
#SBATCH --clusters=chip-cpu
#SBATCH --account=pi_deffner
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/umbc/rs/pi_deffner/users/devjyot1/projects/Quantum-Machine-Learning-/data_reupload/noisy_direct14/logs/%x_%j.out
#SBATCH --error=/umbc/rs/pi_deffner/users/devjyot1/projects/Quantum-Machine-Learning-/data_reupload/noisy_direct14/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/umbc/rs/pi_deffner/users/devjyot1/projects/Quantum-Machine-Learning-

cd "$PROJECT_ROOT"

echo "============================================================"
echo "QML environment check"
echo "============================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"

module purge
module load Python/3.11.5-GCCcore-13.2.0

echo
echo "Loaded modules:"
module list

echo
echo "Module Python:"
which python
python --version

echo
echo "Activating venv_req:"
source "$PROJECT_ROOT/venv_req/bin/activate"

echo
echo "Virtual environment:"
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"
echo "Python executable: $(which python)"
python --version

echo
echo "Python shared-library check:"
ldd "$(which python)" | grep -E "python|not found" || true

echo
echo "Package versions:"

python - <<'PY'
import importlib

packages = [
    "numpy",
    "pandas",
    "torch",
    "pennylane",
    "qiskit",
    "qiskit_aer",
    "pennylane_qiskit",
    "qiskit_ibm_runtime",
]

failed = []

for package in packages:
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "version unavailable")
        print(f"{package}: {version}")
    except Exception as exc:
        failed.append(package)
        print(f"{package}: NOT AVAILABLE — {type(exc).__name__}: {exc}")

if failed:
    raise SystemExit(
        "Missing or broken packages: " + ", ".join(failed)
    )

print("\nAll required imports succeeded.")
PY

echo
echo "Environment check completed successfully."
