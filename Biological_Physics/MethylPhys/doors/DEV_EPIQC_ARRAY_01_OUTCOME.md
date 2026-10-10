# DEV-EPIQC-ARRAY-01 — outcome (2026-10-10; development)

Read by the sealed rule (`data/DEV_EPIQC_ARRAY_01/epiqc_array_01.py read`; output `epiqc_array_01_output.txt`; rows `epiqc_array_01_rows.csv`,
`epiqc_array_01_tared.csv`). 30 EPIC arrays of identical DNA (GIAB lines HG001–HG007) from three laboratories, chain Stage 1, Stage T self-tare II,
the chain's isolated-specimen readers. Lymphoblastoid lines read on the neutrophil scale: A of 1.15–1.32 is the line's distance from the
neutrophil identity, not a health state.

| bar | rule | result |
|---|---|---|
| 1 | Met-A A, technical replicates, difference <= 0.012 | **met 10/10** (largest 0.0114, lab C; lab A <= 0.0078) |
| 2 | A_rel (same-run tare) across three laboratories, spread <= 0.02 | **met 3/3** (0.0022, 0.0042, 0.0060) |
| 3 | C-score, technical replicates, difference <= 0.10 | **met 9/10** (largest 0.1235, HG007 lab C) |
| 4 | C_rel across three laboratories, spread <= 0.10 | **met 3/3** (0.0535, 0.0464, 0.0845) |

Recorded, no bar: raw A across laboratories, per line, spread 0.0044–0.0084 (the laboratory offset on identical DNA, before any tare); raw C
spread 0.028–0.131; the same-run tare lowers C's spread for HG007 (0.131 → 0.085) and raises it slightly for HG005 and HG006.
Met-A's reading of the same DNA is reproducible within 1 % across three laboratories on EPIC v1. The C-score repeats within 0.10 in 9 of 10 pairs
here; in the Moss series (DEV-CSCORE-MOSS-01) its untared array-to-array spread was larger.

**Milestone:** listed in `Biological_Physics/README.md`, Advancements.
