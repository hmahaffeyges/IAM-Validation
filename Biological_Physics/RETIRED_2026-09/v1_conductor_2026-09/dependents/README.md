# What could not outlive the v1 conductor, retired 2026-09-25

Each of these imported `walther_clinical.py` for something the current chain does not provide, so they were
retired with it rather than left importing a module that is no longer there.

| file | what it needed | what replaces it |
|---|---|---|
| `run_batch.py` | `run_from_folder`, the v1 drop-and-run folder entry | a loop over `chain/MethylPhys_Interface/run_sample.py`, which is what the runbook documents |
| `TEST_DATA/harness/test_epic.py`, `decon_epic.py`, `decon_crc.py`, `run_traj_test.py`, `gen_demo.py` | `PatientContext` and `run_pipeline` - the v1 conductor itself | `PROC-E2E-01`: nine IDAT pairs through `run_sample.py`, scored against the test package's own documented outputs |
| `report_builders/build_patient_wall.py`, `synthetic_patient_harness.py` | the v1 config plus the disease-matching functions | `chain/MethylPhys_Interface/build_methylphys.py`, the commissioned report builder |

`detect_systemic_stress_pattern` was the one function in this group worth keeping; it went to
`chain/disease_matching.py` with the rest of the Stage 8 family.
