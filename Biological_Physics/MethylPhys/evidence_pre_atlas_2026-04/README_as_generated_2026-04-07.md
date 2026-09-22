# GAPE Evidence Database
## IAM-Genomics | Open Science | April 2026

**Generated:** 2026-04-07 15:51 UTC
**GAPE version:** 5.0 | **H_min:** G-002-posterior-April2026 | **Pipeline:** 1.0.0

## Key Results

| Metric | Value |
|--------|-------|
| Cancer datasets | 28 |
| Total matched tumor-normal pairs | 4,304 |
| P1 confirmed (A_tumor > A_normal) | 27/28 (96.4%) |
| Mean delta-A | 0.1589 +/- 0.0721 |
| TCGA cases available (live GDC) | 11,428 |
| GEO datasets catalogued | 10 |
| Detection threshold | A > 1.05 (physics-derived, not from cancer training data) |

## What the A-Score Means

```
A_GAPE = H(beta) / H_min(class)

where H(beta) = -beta*log2(beta) - (1-beta)*log2(1-beta)
      beta    = mean CpG methylation (from 450K array or WGBS)
      H_min   = minimum entropy for that cell architecture class
                (from G-002 MCMC posterior on 37 published reference cells)
```

**The detection threshold A > 1.05 was NOT derived from cancer data.**
It is the point where the three-component decomposition shows a significant
accessible gap (f_C3 > 5%) — the thermodynamic departure from the architecture floor.

## Reproduce

```bash
git clone https://github.com/IAM-Validation/Biological_Physics/evidence
cd GAPE
pip install requests schedule
python3 gape_evidence_engine.py --setup
python3 gape_evidence_engine.py --run
# evidence_summary.json will match exactly
```

## Files

| File | Description |
|------|-------------|
| `evidence_summary.json` | Master database, full provenance, all results |
| `evidence_summary.tsv` | Flat table for spreadsheet / statistical analysis |
| `README.md` | This file (auto-generated) |

## H_min Registry (G-002 MCMC Posterior)

5 independent emcee chains, R-hat < 1.001 for all 8 parameters:

| Class | H_min | Basis |
|-------|-------|-------|
| stem_pluri | 0.982166 | G-002 MCMC posterior |
| stem_adult | 0.873718 | G-002 MCMC posterior |
| progenitor | 0.852216 | G-002 MCMC posterior |
| terminal | 0.772837 | G-002 MCMC posterior |
| cycling | 0.856055 | G-002 MCMC posterior |
| immune | 0.838889 | G-002 MCMC posterior |
| secretory | 0.843264 | G-002 MCMC posterior |
| stromal | 0.862950 | G-002 MCMC posterior |

H_min_global = 0.756499 (frontal cortex neuron — Lister 2013 — global Landauer floor)

*IAM-Genomics | Heath W. Mahaffey | Open Science | No commercial restriction*
