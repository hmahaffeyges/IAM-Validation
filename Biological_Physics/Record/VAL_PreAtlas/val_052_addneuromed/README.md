# VAL-052 — AddNeuroMed Cross-Platform AD Replication

**Date:** 2026-04-23
**Status:** Complete — pre-registered, hash-sealed, executed.
**Parent:** VAL-051
**Outcome:** MIXED — raw cross-platform replicates (d=+0.33, p=0.009); age-corrected d drops to +0.12 (p=0.12)

---

## Headline

The VAL-051 7-CpG AD-directional panel transfers from EPIC (AIBL) to 450K (AddNeuroMed, n=300) with 7/7 CpG coverage and produces raw cross-platform Cohen's d = +0.33, p = 0.009, AUC = 0.60 using AIBL-frozen directions and standardization.

AddNeuroMed includes chronological age. AD cases are +0.45 Cohen's d older than HC. Linear regression: age explains R² = 26% of A_dir variance. After regressing age out, residual d = +0.124, p = 0.12.

**Clinical consequence:** EDEAR must report age-adjusted Z-score per Alpha-Omega §E.5, not raw A_dir.

---

## Files

| File | Role |
|---|---|
| `VAL_052_PREREG.md` | Pre-registration with frozen-panel specification |
| `VAL_052_SEAL.txt` | SHA-256 hashes sealed before analysis |
| `stream_addneuromed_v2.py` | Streams 514 MB 450K matrix, extracts 18 IMM panel CpGs |
| `addneuromed_manifest.json` | Per-GSM metadata (age, sex, disease status, progression) |
| `addneuromed_imm_betas.json` | Per-sample β for 18 panel CpGs, n=300 |
| `val052_analyze.py` | Primary + sensitivity + H2 MCI + H3 sex + H4 age regression + H5 age-by-group |
| `VAL_052_RESULTS.json` | Full results, honest mixed-decision labels |
| `VAL_052_REPORT.md` | Human-readable report with all tables and interpretation |

---

## Reproduction

```bash
# Uses VAL-051 panel file, which lives in ../val_051_ad_directional/
# Symlink or copy:
cp ../val_051_ad_directional/val051_panel_ruleA.json .

# Stream data (15-30 min depending on NCBI speed)
python3 stream_addneuromed_v2.py

# Analyze (30 sec)
python3 val052_analyze.py
```

All stdlib Python 3.9+. Seed 42. Outputs byte-identical.

Matrix source SHA-256: `a16bbdaad06de07c95a5669731786c4e75aad2ea16428a9e928cfcf49f46bb90`

---

## Decision matrix applied

From VAL_052_PREREG.md:

| Result on AddNeuroMed | Interpretation |
|---|---|
| Raw d=+0.33, p=0.009 | **OUTCOME 1** — cross-platform replication of raw signal |
| Age-corrected d=+0.12, p=0.12 | **OUTCOME 3-borderline** — direction-positive-weak after age adjustment |
| Net | MIXED — both findings must be reported; prioritize age-corrected for clinical claims |

---

## Sources

- Roubroeks JAY, Smith AR, et al. **Methylomic analysis of an Alzheimer's disease blood epigenome identifies differentially methylated regions and robust HOXB6 hypermethylation.** *Nature Communications* 11:4805, 2020. doi:10.1038/s41467-020-18476-6
- GSE144858 GEO deposit

## What this feeds into next

- **VAL-051 Cookbook card update**: validation_tier upgraded from `internal_holdout` to `cross_platform_validated`
- **Age-adjusted Z-score (Alpha-Omega §E.5) becomes the primary clinical metric**, not raw A_dir
- Platform portability recorded: 7/7 EPIC↔450K panel transfer
