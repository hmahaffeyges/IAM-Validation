# PROC-ENCODE-01 — outcome (2026-09-30). Pre-registration: PROC_ENCODE_01_PREREG.md (sha b0e92114d6367453), written before any read was counted.

43 ENCODE WGBS alignments (GRCh38), each streamed once; 400 random 100-kb autosomal windows; 39 usable (≥ 5,000 methylated reads with ≥ 6 CpGs).
Unusable: 2 H1 and 2 IMR-90 files (0.1–0.8 GB, 2–160 qualifying reads). Two pipelines among the usable files: gemBS (26) and Bismark (13); HUES64 has one of each.
Box 2 was reclaimed by AWS mid-run; the Bismark files were re-run on box 1 and reproduced the lost values to four decimals.
Instrument floor per sample: substitution error (mismatches at bases conversion cannot touch, per substitution) and non-CpG conversion failure.

| prediction | result | verdict |
|---|---|---|
| P1 immune copy error below every stromal/muscle sample | immune (instrument-corrected) 0.0193–0.0227 (B, monocyte, NK, T); stromal/muscle 0.0251–0.0287 (aorta, heart LV, psoas, myoblast) | **PASS** (IMR-90 unusable: 4 of the 5 named stromal samples scored) |
| P2 cancer line > its normal counterpart in ≥ 4 of 5 pairs | 5 of 5 (table below) | **PASS** |
| P3 instrument error < 20 % of copy error in every non-neural, non-stem sample | 4–14 % in most; mammary epithelial cells 30 %, right lobe of liver 20.4 % | **FAIL** (2 samples with poor conversion) |

| cancer vs normal | copy error (corrected) | A on the physics floor | holding energy (kT) |
|---|---|---|---|
| HepG2 vs liver | 0.0263 vs 0.0234 | 1.124 vs 1.026 | 3.61 vs 3.73 |
| A549 vs lung | 0.0248 vs 0.0212 | 1.074 vs 0.949 | 3.67 vs 3.83 |
| K562 vs common myeloid progenitor, CD34-positive | 0.0294 vs 0.0182 | 1.226 vs 0.841 | 3.50 vs 3.99 |
| GM12878 vs B cell | 0.0295 vs 0.0193 | 1.229 vs 0.881 | 3.49 vs 3.93 |
| OCI-LY7 vs B cell | 0.0269 vs 0.0193 | 1.144 vs 0.881 | 3.59 vs 3.93 |

Physics floor for this table: ε₀ = 1/(1+e^(φM)) with φ = 0.1798, the mean over the 17 normal ENCODE samples (Loyfer, uncorrected: 0.1629).

**Also measured (descriptive):**
- Pluripotent stem cells hold their pattern best: HUES64 0.0186, H9 0.0169 (A 0.86 / 0.79 on the physics floor) — the stem class the Loyfer atlas lacked sits lowest.
- Unmethylated channel: de novo error minus conversion failure and substitution error is −0.019 to +0.011 (median ≈ 0.002). **The unmethylated channel is
  mostly instrument** in bulk WGBS; the φ ≈ 0.21 reported for it from Loyfer is not a cellular quantity until corrected. OCI-LY7 lymphoma is the clearest
  real gain (+0.0105), consistent with the CpG-island hypermethylation of lymphoma.
- Culture raises error in normal cells too: skeletal myoblasts in culture 0.0287 vs psoas muscle tissue 0.0267. The cancer lines are decades in culture; part of
  their excess may be culture, not transformation (the same caveat HBEC showed). Normal comparators here are tissues and primary cells.

**Reading.** In an independent lab and two pipelines, the methylated-channel error reproduces the architecture ordering (immune low, stromal/muscle high,
pluripotent lowest) and rises in every cancer line against its normal lineage — on a floor set by physics, with no cohort and no classifier.
Not shown: primary tumours, early disease, or plasma; and culture is not separated from transformation.
