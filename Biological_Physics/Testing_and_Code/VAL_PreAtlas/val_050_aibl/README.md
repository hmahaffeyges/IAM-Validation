# VAL-050 — AIBL AD Immune-Class Cross-Sectional Replication (April 2026)

Pre-registered, hash-sealed, post-seal execution of the first cross-disease-class
test of the GAPE immune-class A-score pipeline. This work extends VAL-040
(AD whole-blood methylation mean-level finding) to the per-patient level in a
fully independent cohort on an independent platform.

## Contents

- **VAL_050_PREREG.md** — pre-registration document (decision rules, frozen
  constants, four pre-locked outcome categories)
- **VAL_050_SEAL.txt** — SHA-256 hashes of all inputs sealed before analysis
- **run_val_050.py** — post-seal pre-locked analysis script
- **stream_aibl.py** — GEO streaming extractor (4.8 GB AIBL EPIC matrix
  → per-sample β for 18 panel CpGs)
- **aibl_manifest.json** — parsed GSE153712 sample metadata (n=726)
- **aibl_imm_betas.json** — per-sample β values, 726 samples × 18 panel CpGs
- **VAL_050_RESULTS.json** — full output (primary + secondary stats, per-CpG
  FDR, per-sample A-scores)
- **VAL_050_REPORT.md** — human-readable report with tables and interpretation

## Headline

| Metric | Value |
|---|---|
| Cohort | AIBL GSE153712 (EPIC 850K, n=726) |
| n_AD / n_MCI / n_HC | 161 / 94 / 471 |
| Panel | IMM_CPGS_EPIC_18 (29-CpG IMM_CPGS_RAW ∩ EPIC = 18 CpGs, 62% transfer) |
| H_min(immune, methyl) | 0.838889 (G-003b MCMC posterior, frozen) |
| **Primary — H1 pooled MWU** | Cohen's d = +0.077, p = 0.321, AUC = 0.512 |
| **Outcome (pre-locked)** | **OUTCOME 3 — NULL** |
| Secondary — H3 females | d = +0.207, p = 0.032 (direction-positive-weak) |
| Secondary — H3 males | d = −0.005, null |
| Secondary — H4 per-CpG | **7 of 18 CpGs at FDR < 0.05**, bidirectional |
| Secondary — H2 HC < MCI < AD trend | null (J z = 0.08) |

## What this establishes

1. A pre-registered null on pooled-β AD detection using a breast-cancer-derived
   panel — referee-proof honest negative.
2. Strong per-CpG signal hidden by bulk averaging (4 CpGs go down in AD, 3 up).
3. Sex-asymmetric AD immune signal consistent with independent Yang 2024 finding.
4. 450K → EPIC panel transfer is lossy (18/29 CpGs survive).
5. Concrete motivation and blueprint for VAL-051 purpose-built AD-directional panel.

## Reproduction

```bash
# 1. Pull AIBL matrix from GEO (4.8 GB, 15-30 min first time)
python stream_aibl.py

# 2. Verify SHA-256 seal matches
sha256sum -c <(cat VAL_050_SEAL.txt | sed -n '3,7p' | awk '{print $1"  "$2}')

# 3. Run analysis (~15 seconds stdlib-only)
python run_val_050.py
```

Seed = 42 everywhere. Output is byte-identical across re-runs given the same
SHA-matched inputs.

## Cohort citation

Nabais MF, Laws SM, Lin T, et al. *Meta-analysis of genome-wide DNA methylation
identifies shared associations that underpin common late-onset Alzheimer's
disease.* Genome Biology 2021 Mar 26;22(1):90. doi:10.1186/s13059-021-02389-w

## Panel citation

Xu Z, Sandler DP, Taylor JA. *Blood DNA methylation and breast cancer: a
prospective case-cohort analysis in the Sister Study.* Journal of the National
Cancer Institute 2020 Jan;112(1):87-94. doi:10.1093/jnci/djz065

## Disclosure architecture

Public in this folder: panel CpG list, calibrated H_min, formula (Shannon
entropy / H_min), scripts, raw and derived JSON.

Not public and not in any repository: the framework's architectural-class
taxonomy, the class-assignment rule, the calibration code that produces H_min,
and the first-principles derivation — all covered under USPTO provisional
patents 64/012,720 (filed March 21, 2026) and 64/014,568 (filed March 23, 2026).

## Next

VAL-051: purpose-built AD-directional panel. Selection criterion (pre-locked
before analysis): AIBL training-split CpGs with |Δβ| > 0.02 AND FDR < 0.05,
frozen; hold-out test on AIBL remainder plus AddNeuroMed GSE144858 (450K, n=300)
for cross-platform transfer test.
