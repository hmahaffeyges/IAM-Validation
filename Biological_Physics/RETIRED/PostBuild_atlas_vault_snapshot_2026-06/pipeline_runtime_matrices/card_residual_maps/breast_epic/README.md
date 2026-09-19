# breast_epic — Layer 3 evidence artifacts

**Card:** breast-epic
**Cohort:** GSE51057 + GSE51032, breast pre-diagnostic cases (>10 years pre-diagnosis) vs healthy controls
**N:** 47 cases (11 GSE51057 + 36 GSE51032) vs 601 HC (177 + 424)
**Build session:** 2026-05-29 (TODO 1.3, 1.4, 1.5 of EDEAR Physics Roadmap)
**Pipeline tool:** production Walther IAM Deconvolver, vetted against IAMAtlas REBUILD

This subfolder holds card-specific evidence artifacts that the breast-epic card cites in its `evidence_anchors` block. The breast-epic card JSON and card README live in the card folder (Heath-only canonical); the artifacts below are the per-CpG / per-axis evidence stack that those documents reference.

## Files

| File | Built from | Rows | Purpose |
|---|---|---|---|
| `residual_map.csv` | TODO 1.3 — Walther deconvolver reconstruction at 7,114 class-marker CpGs | 7,114 CpGs | Per-CpG signed Cohen's d (case vs HC) of the (observed − reconstructed) β residual. Identifies where the disease-specific signal lives orthogonal to cellular composition. **1,392 concordant CpGs (|d|>0.3 in both cohorts, same sign). 1,173 hypomethylated vs 219 hypermethylated — 5.4:1 ratio. 1,389 NEW candidates not in Xu-538.** Top concordant hits: cg20124336 (d=−2.17/−1.89), cg16188349 (d=−1.67/−1.67), cg27467249 (d=−2.17/−1.17). All hypomethylated, all replicating. |
| `bimodality_map.csv` | TODO 1.4 — Sarle bimodality coefficient per CpG, case vs HC | 8,199 CpGs | Per-CpG bimodality scores. **821 CpGs bimodal in HC; 396 lose bimodality in cases.** Distribution-shape signature orthogonal to mean-shift signature. 35 CpGs are DOUBLE-CONFIRMED (loss-of-bimodality AND concordant residual-map signal) — highest-confidence candidate biomarker class. |
| `pca_projections.csv` | TODO 1.5 — PCA on 8-class A-score covariance fit to HC, projecting cases | 647 patients × 8 PCs | Per-patient projections on 8 principal axes. **PC1 (70.7% variance): broad cellular drift axis, d=+1.07/+0.57 cases vs HC. PC2 (13.4%): stem_pluri vs stem_adult+immune asymmetric across cohorts.** From the 115-cell PCA (run separately, not saved here): **PC2 T-cell SUPPRESSION axis d=-0.67/-0.58 replicating — immunosurveillance failure signature.** |

## How the breast-epic card consumes these

The card scores a customer in three stages drawing from these files:

1. **Stage 2.5 Mahalanobis** gives the universal departure number with confidence interval and top-10 axis decomposition (uses `mahalanobis_healthy_reference_v0_1.json` upstairs in pipeline_runtime_matrices/).
2. **Per-CpG residual scoring (Stage 5 evidence)**: at the 1,392 concordant CpGs in `residual_map.csv`, compute (observed − Walther-reconstructed)_customer and compare to (observed − reconstructed)_cases vs HC distribution here. Per-CpG z-score → card-level evidence aggregate.
3. **Bimodality monitor (Stage 5 evidence)**: at the 396 loss-of-bimodality CpGs in `bimodality_map.csv`, flag customer β values that fall in the previously-empty intermediate range (β between 0.2 and 0.8 where HC distribution was strictly bimodal). Cumulative intermediate-β count contributes to card evidence.

## Biological summary — what this card now knows

Three replicating biological signatures triangulated from independent methods on the same cohort:

1. **Universal cellular drift** (Mahalanobis d=+1.87/+2.09, residual-map dominant signal, PC1 d=+1.07/+0.57). Distributed shift across architecture, consistent with the field-effect cancerization model — pre-disease cells across the body show methylation drift at 10+ years before clinical detection.
2. **Tissue-of-origin signal at >10yr** (breast-epithelial BE d=+1.281 in GSE51057, +0.614 in GSE51032). VAL-096 had read the Loyfer Breast tile at d=+0.20/+0.10 at this phase; the per-cell-type readout sees it ~5× louder. Field-effect at the originating tissue is visible a decade before clinical detection.
3. **T-cell suppression** (PCA PC2 d=-0.67/-0.58). Replicating, CD4/CD8-loaded axis weakens in cases. Immunosurveillance failure signature — explains how a pre-disease cellular drift can persist for 10+ years without being cleared.

## Evidence anchors (this card cites these)

- VAL-046 (Sister Study cohort-level signal)
- VAL-047 Phase 9 sealed (Xu-538 panel reproduction: d=+1.847 GSE51057, d=+1.336 GSE51032 at H_min(immune)=0.838889)
- VAL-093 / VAL-094 / VAL-095 / VAL-096 (per-cell-type / per-tile signatures)
- **TODO 1.1 (2026-05-29)** — per-cell-type top-10 signatures, 115-cell fan-out
- **TODO 1.2 (2026-05-29)** — Mahalanobis hyper-volume d=+1.871/+2.088 (95% CI [+1.014, +2.856]/[+1.502, +2.735])
- **TODO 1.3 (2026-05-29)** — per-CpG residual map (`residual_map.csv` here)
- **TODO 1.4 (2026-05-29)** — bimodality-loss map (`bimodality_map.csv` here)
- **TODO 1.5 (2026-05-29)** — PCA decomposition (`pca_projections.csv` here, PC2 T-cell suppression)
