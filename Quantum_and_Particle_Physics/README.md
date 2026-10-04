# Quantum and Particle Physics

IAM at the quantum and particle scale: the book's Parts IV (particles and quantum records) and V (qubits and semiconductors). This folder
grows as work in this field is added. For now the book carries the physics, and the scripts that recompute its numbers are in
[`docs/verification/scripts/`](../docs/verification/scripts/):

| Script | Book topic |
|---|---|
| [`verify_records_measurement_time.py`](../docs/verification/scripts/verify_records_measurement_time.py) | quantum records, measurement time |
| [`verify_quantum_darwinism.py`](../docs/verification/scripts/verify_quantum_darwinism.py) | quantum Darwinism |
| [`verify_grav_decoherence.py`](../docs/verification/scripts/verify_grav_decoherence.py) | gravitational decoherence |
| [`verify_entanglement_electroweak.py`](../docs/verification/scripts/verify_entanglement_electroweak.py) | entanglement and the electroweak scale |
| [`verify_koide.py`](../docs/verification/scripts/verify_koide.py) | the Koide relation |
| [`verify_electron_mass.py`](../docs/verification/scripts/verify_electron_mass.py) | the electron mass |
| [`verify_xqp.py`](../docs/verification/scripts/verify_xqp.py), [`verify_xqp_book.py`](../docs/verification/scripts/verify_xqp_book.py) | quasiparticles in superconducting qubits |

Every derivation in these Parts is also checked by [`docs/book/verify_book.py`](../docs/book/verify_book.py).
