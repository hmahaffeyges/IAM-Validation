# Per-Card Disease Residual Maps

Per-card Layer 3 evidence artifacts. Each card has its OWN subfolder with its own README, per-CpG residual map, bimodality map, and PCA projections.

## Subfolders

| Card | Subfolder | Cohort | N (case/HC) | Status |
|---|---|---|---|---|
| breast-epic | `breast_epic/` | GSE51057+GSE51032 >10yr breast pre-dx | 47 / 601 | v0.1 sealed 2026-05-29 |
| crc-epic | (pending TODO 1.3 rerun for CRC) | — | — | — |
| ad-immune | (pending) | — | — | — |
| ...others | (pending) | — | — | — |

## What lives in each card subfolder

Each card's subfolder contains:
- `README.md` — card-specific evidence summary + biological interpretation
- `residual_map.csv` — per-CpG signed Cohen's d residual (TODO 1.3 product)
- `bimodality_map.csv` — per-CpG Sarle BC case vs HC (TODO 1.4 product)
- `pca_projections.csv` — per-patient PCA scores (TODO 1.5 product)

## Methodology (common across cards)

For each card-defined cohort:
1. **Stage 1**: Walther IAM Deconvolver → per-class fractions per patient
2. **Stage 2 reconstruction**: at each marker CpG, predicted β = Σ(class_fraction × class_reference_β)
3. **TODO 1.3 residual_map.csv**: per-CpG residual = observed − reconstructed, per-CpG Cohen's d cases vs HC, stratified by source cohort, concordance gate (|d|>0.3 both cohorts, same sign)
4. **TODO 1.4 bimodality_map.csv**: per-CpG Sarle bimodality coefficient (BC) for cases and HC separately, loss-of-bimodality (HC > 5/9 AND case < 5/9)
5. **TODO 1.5 pca_projections.csv**: per-patient PCA on class A-score covariance (fit on HC, project all), case-vs-HC d per PC

## How cards consume these

Each card's evidence_anchors in its JSON references these files. Stage 5 of the EDEAR pipeline uses them as evidence layers when matching customer profiles against documented signatures. Mahalanobis hyper-volume (Stage 2.5) gives the headline departure number; these per-card maps give the per-CpG / per-axis backup that turns the number into an explanation.
