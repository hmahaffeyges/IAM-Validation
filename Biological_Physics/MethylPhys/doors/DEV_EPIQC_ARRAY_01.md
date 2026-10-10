# DEV-EPIQC-ARRAY-01 — Met-A and C-score on identical DNA at three laboratories (written 2026-10-10 before any array is calibrated; development)

Data: GEO GSE230132 (SEQC2 EpiQC arrays; CANDIDATE_EPIQC.md), 30 EPIC v1 arrays (1,051,815 addresses, readable by Stage 1). Script
`data/DEV_EPIQC_ARRAY_01/epiqc_array_01.py` (definitions there). Lymphoblastoid lines: an instrument test, not a health reading.

**Bars (each met if it holds for >= 80 % of the cases):**
1. Met-A A, technical replicates of the same DNA in one laboratory: difference <= 0.012 (twice the frozen-site held-out SD of the commissioned floor, 0.006).
2. Same-run tare across laboratories (HG005-HG007, the lines all three laboratories ran; references = the other two lines in that laboratory):
   per line, the spread of A_rel across the three laboratories <= 0.02.
3. C-score, technical replicates: difference <= 0.10.
4. C_rel across laboratories (same tare rule): spread <= 0.10.
Raw A and C across laboratories are recorded without a bar (the size of the laboratory offset on identical DNA).
