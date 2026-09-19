# Execution note — bladder-epic v0.1 Phase C

The individual val12X_*.py scripts in each VAL directory implement the prereg-locked
scoring logic for that VAL independently. **For execution efficiency**, the actual
end-to-end Phase C execution used a unified runner that loads each TCGA-BLCA β file
once and produces per-sample tables for all three VALs (VAL-120, VAL-121, VAL-122)
simultaneously, eliminating 3× redundant I/O on the n=440 cohort.

## Execution scripts (parent directory)

- `../unified_phaseC_runner.py` — single-pass runner that produced `VAL_121_unified_per_sample.csv`
  (the source of all three VALs' per-sample tables). Runtime 270.7 sec for n=440 cohort.
  Pandas + numpy vectorized scoring.
- `../postpass_amended.py` — post-pass that computed paired d, Welch d, and outcome
  class for all three VALs against the **amended CHK-3.1A mucosal-tissue-class floor**
  (per prereg_amendment_002.md sealed in this directory).

## Bit-for-bit equivalence to per-VAL scripts

The unified runner uses the **identical** scoring logic (A-score formula, H_min anchors,
CpG intersection rules) as the per-VAL scripts. The H_min anchor list is shared across
both code paths. Output equivalence:

- `VAL-120_per_sample.csv` (this VAL) is a column-projection of the unified per-sample table.
- `VAL-121_per_sample_per_atlas.csv` (VAL-121) is also a column-projection of the same table.
- `VAL-122_per_sample_per_atlas.csv` (VAL-122) is also a column-projection of the same table.

The per-VAL val12X_*.py scripts remain in each VAL directory as reference implementations
of the locked logic per VAL.

## Reproducibility

To reproduce VAL-120/121/122 outputs:
1. Acquire the TCGA-BLCA cohort (440 sesame Level 3 .txt files; manifest in cohort_manifest.json).
2. Acquire the EpiSCORE BladderRef CpG-bridged matrix (SHA-256 `3005663b4ede4b20199bacff641952390b1434764b8cf0915cdc9d6a6c1517c6`).
3. Run `python3 ../unified_phaseC_runner.py` then `python3 ../postpass_amended.py`.
4. The outputs land in the three VAL directories.
