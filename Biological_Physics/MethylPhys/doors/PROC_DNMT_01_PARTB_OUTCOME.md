# DNMT-01 Part B — IAM-A on single molecules under a known DNMT1 block (scored 2026-10-02)

**Data.** GSE329728: EM-seq, MV4-11, DNMT1 inhibitor GSK3685032 100 nM for 7 days vs DMSO; 4 genotypes (WT and 10KR, each gNTC and gRNF4)
x 2 replicates = 16 libraries. Our alignment and copy-error extraction (same as PROC-TUMOUR-01). Scorer: `PROC_DNMT_01_PARTB/score_dnmt_b.py`
(box job 473aa12c). Copy error eps corrected for the substitution error rate; genotype mask as PROC-TUMOUR-01. Reference = the same genotype's
two DMSO libraries.

| bar (pre-registered) | result | |
|---|---|---|
| Q1: H(eps) treated / H(eps) DMSO > 1.05 in every genotype, both replicates | 8/8, A = 1.65–1.97 | PASS |
| Q2: conversion-failure difference treated vs DMSO < 0.005 in each pair | 8/8, largest 0.0007 | PASS |

DMSO copy error is 0.0209–0.0217 in all 8 DMSO libraries (four genotypes). Treated: 0.0405–0.0511. gRNF4 genotypes rise less (A 1.65–1.70)
than gNTC (1.88–1.97) in both backgrounds. Descriptive (Part A comparison): Part A Met-A at 400 nM day 6 read 1.73–1.85 on arrays; Part B
IAM-A at 100 nM day 7 reads 1.65–1.97 on single molecules.

**Limits.** One cell line, one laboratory. A qualifying molecule needs methylated calls around the scored site, so treated libraries have fewer
qualifying molecules (38,714–257,775 vs 328,957–508,293 for DMSO); the reading is the error rate on the molecules that still carry the pattern.
The drug also slows proliferation. A pass shows IAM-A reads a known change in maintenance fidelity; it is not a disease reading.
