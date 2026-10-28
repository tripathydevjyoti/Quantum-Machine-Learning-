# This file provides compatibility loading for the official IBM
# FakeAachen calibration snapshot distributed with
# qiskit-ibm-runtime 0.47.0.
#
# The backend data remain governed by the accompanying Apache-2.0
# Qiskit IBM Runtime license.

from pathlib import Path

from qiskit_ibm_runtime.fake_provider import fake_backend


class FakeAachenSnapshot(fake_backend.FakeBackendV2):
    """Fake Aachen backend loaded from a frozen local snapshot."""

    dirname = str(Path(__file__).resolve().parent)
    conf_filename = "conf_aachen.json"
    props_filename = "props_aachen.json"
    backend_name = "fake_aachen"
