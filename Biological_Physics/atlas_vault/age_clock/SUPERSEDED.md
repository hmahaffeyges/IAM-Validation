# SUPERSEDED 2026-05-30

The Horvath-style regression clock approach in `iam_cellular_age_clock.py` was
superseded by the canonical IAM inversion at:

    Biological_Physics/atlas_vault/pipeline_runtime_matrices/iam_cellular_age_scoring.py

**Why superseded.** The regression-based approach trains coefficients against a
training cohort to project A-scores onto a 1-D age axis. That is the Horvath
methodology with different features (8 class A-scores instead of 353 CpGs).
Heath rejected this on 2026-05-30 — "back of the hand thermometer."

**What replaces it.** Recipe §6.3 canonical: for each class, invert the
80-cell age_reference_matrix.json baseline (the calibrated instrument). The
patient's per-class A = H(β_mean)/H_min is computed, then the age at which
baseline A_mean(class, age) crosses the patient's A is read off the curve.
Eight independent per-class cellular ages, never collapsed to a single
number. No training set. No regression. The atlas is the instrument.

Files in this folder are kept for historical reference only — they are NOT
part of the production pipeline.

Files preserved:
- `iam_cellular_age_clock.py` (Horvath-style regression, REJECTED)
- `8class_ascores_all.csv` (per-patient A-scores from old formula; superseded by
  pipeline_runtime_matrices/cellular_ages_v4_epic_italy_validation.csv)
- `age_clock_diagnostics.json` (regression diagnostics, REJECTED)
- `Phase_B3_FINDING.md` (B3 finding doc that documented the rejection)
