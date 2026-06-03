# AD-immune Card v3.0 — Work in Progress

**Started:** 2026-06-02
**Status:** Stage 0 environment preflight complete; cohort acquisition starting
**Workflow:** SOP v1.2 Stages 0-10 + L9 null suite, no Step 7 RETIRED scripts
**Source v2.2 card:** archived at ad_immune_card_json/OLD/ad-immune_card_v2.2.json
**Source README:** carried forward at ad_immune_card_json/ad-immune_README.md

## Pipeline progress

- [x] Stage 0.1 — repo authentication
- [x] Stage 0.2 — IAMAtlas REBUILD SHA verified (`41b7c16f...`)
- [x] Stage 0.3 — celltype_to_class.json: 115 cells / 51 immune / 19 cycling / 18 secretory / 9 terminal / 11 progenitor / 5 stromal / 2 stem (verified)
- [x] Stage 0.4 — Walther IAM Deconvolver instantiated (7,114 class markers + 4,000 cell-type markers)
- [x] Stage 0.5c — celltype markers v0_2 SHA matches `46ea5be1...`
- [x] Stage 0.5d — Mahalanobis healthy reference present + Ledoit-Wolf shrinkage 0.0088 from breast precedent
- [x] CpG union for AD extraction: 14,018 unique CpGs (Walther markers ∪ v0_2 ∪ age layer ∪ AD panels)
- [ ] Cohort acquisition: AIBL GSE153712 (726 samples, EPIC)
- [ ] Cohort acquisition: AddNeuroMed GSE144858 (300 samples, 450K)
- [ ] Cohort acquisition: GSE53740 GIFT (384 samples, 450K)
- [ ] Stage 1 reproduction: 7-CpG Rule A panel, expect AIBL d≈+0.624 on holdout
- [ ] Stage 2 Walther deconvolution → 8-class A-scores + 115-cell A-scores
- [ ] Stage 3 age-axis foreground subtraction
- [ ] Stage 4 per-cell-type A-score fan-out
- [ ] Stage 5 Mahalanobis hyper-volume
- [ ] Stage 6 cellular age per class
- [ ] Stage 7 tier breakpoints + bidirectional flag
- [ ] Stage 8 dual matching: AD-immune card + disease matrix alzheimers rows
- [ ] Stage 9 report assembly
- [ ] Stage 10 delivery
- [ ] L9 null suite per VAL
- [ ] CPG-VAL-008 through CPG-VAL-014 sealed
- [ ] Card v3.0 finalized
- [ ] Disease matrix v1.5 → v1.6 with alzheimers rows
- [ ] Residual maps (chr-annotated + bimodality + PCA) + README
