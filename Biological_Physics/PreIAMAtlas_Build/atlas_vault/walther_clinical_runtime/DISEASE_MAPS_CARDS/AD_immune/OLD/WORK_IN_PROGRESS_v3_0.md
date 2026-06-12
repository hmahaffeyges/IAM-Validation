# AD-immune Card v3.0 — Work Status

**Started:** 2026-06-02
**Pipeline:** SOP v1.2 Stages 0-10 + L9 null suite, no Step 7 RETIRED scripts
**Source v2.2 card:** archived at `ad_immune_card_json/OLD/ad-immune_card_v2.2.json`
**Source README:** carried forward at `ad_immune_card_json/ad-immune_README.md`

## Pipeline progress

### Stage 0 — Environment preflight
- [x] Stage 0.1 — repo authentication
- [x] Stage 0.2 — IAMAtlas REBUILD SHA verified (`41b7c16f...`)
- [x] Stage 0.3 — celltype_to_class.json: 115 cells (51/19/18/9/11/5/1/1) verified
- [x] Stage 0.4 — Walther IAM Deconvolver instantiated (7,114 class markers + 4,000 cell-type markers)
- [x] Stage 0.5c — celltype markers v0_2 SHA matches `46ea5be1...`
- [x] Stage 0.5d — Mahalanobis healthy reference + Ledoit-Wolf shrinkage 0.00875 from breast precedent
- [x] CpG union for AD extraction: 14,018 unique CpGs (Walther markers ∪ v0_2 ∪ age layer ∪ AD panels)

### Cohort acquisition
- [x] AIBL GSE153712 — 726 samples, EPIC, 13,384/14,018 CpGs (95.5%) — SHA `15633616...`
- [x] AddNeuroMed GSE144858 — 300 samples, 450K, 12,169/14,018 CpGs (86.8%) — SHA `d4edaa43...`
- [x] GSE53740 GIFT — 384 samples, 450K, 13,598/14,018 CpGs (97.0%) — SHA `2aba6a23...`

### Stage 1 reproduction (anchor verification)
- [x] AIBL d=+0.615 vs anchor +0.624 (VAL-051) — within sampling variation
- [x] AddNeuroMed d=+0.317 vs anchor +0.332 (VAL-052) — within sampling variation
- [x] GSE53740 d=+0.013 vs anchor +0.013 (VAL-057) — EXACT 3-decimal match
- [x] GSE53740 male AD d=+0.415 vs post-hoc anchor +0.415 (VAL-057) — EXACT 3-decimal match

### Stage 2 Walther deconvolution
- [x] AIBL — 726 samples in 142s (0.20s/sample), 8 class fractions per sample
- [x] AddNeuroMed — 300 samples in 57s
- [x] GSE53740 — 384 samples in 75s

### Stage 4 — Per-cell-type A-score fan-out (115 cells)
- [x] AIBL — CPG-VAL-008 substantively done: 20 Bonferroni-sig negative effects, top Eosino d=−0.43
- [x] AddNeuroMed — CPG-VAL-010 cross-platform: per-cell-type biology REPLICATES exactly (Eosino d=−0.46)
- [x] GSE53740 — CPG-VAL-014 GIFT specificity: distinct AD/FTD/PSP signatures

### Stage 5 — Mahalanobis hyper-volume
- [x] AIBL — CPG-VAL-009: d=+0.20 (p<0.001), MCI intermediate position confirmed
- [x] AddNeuroMed — CPG-VAL-010: d=−0.006 NULL on 450K (per-cell biology still replicates)
- [x] GSE53740 — CPG-VAL-014: AD d=+0.68, PSP d=−0.38, FTD d=+0.28

### Additional analyses
- [x] CPG-VAL-011 age-axis foreground subtraction (AddNeuroMed + GIFT; minimal impact, Δd<0.05 on 115-cell layer)
- [x] CPG-VAL-012 PC1 T-cell axis (AIBL, PC1 d=−0.356, T-cell-dominated loadings)
- [x] CPG-VAL-013 per-CpG residual map (AIBL+AddNeuroMed, 241 strong-concordant CpGs, CPG_ad_panel_v1 candidate emitted)

### Card v3.0 — DRAFTED
- [x] Card v3.0 JSON (`ad_immune_card_json/ad-immune_card_v3_0.json`) — strict additive over v2.2
- [x] Release notes (`ad_immune_card_json/ad-immune_v3_0_release_notes.md`)
- [x] Residual maps folder (`ad_immune_residual_maps/`)
  - [x] `ad_immune_residual_map_chr_annotated.csv` (6,018 CpGs, cross-cohort)
  - [x] `ad_immune_pca_projections.csv` (AIBL PC1-PC10)
  - [x] `ad_immune_bimodality_map.csv` (placeholder, deferred to v3.1)
  - [x] `README_AD_residual_maps.md`

### Disease matrix v1.5 → v1.6
- [x] v1.5 archived in `DISEASE_MATRIX/OLD/`
- [x] v1.6 created with 3 new rows appended (alzheimers at_dx_post_build_v3_0, FTD post_build_GIFT_2026, PSP/CBD post_build_GIFT_2026)
- [x] README + schema doc updated to v1.6

### Outstanding (carries to next sessions)
- [ ] Formal v4 inventory sealing per VAL (PREREG.md + sealed reproducer + L9 null suite per VAL = 7 tests each)
- [ ] CHR/MAPINFO genomic annotation on residual map (deferred to v3.1)
- [ ] Bimodality decomposition (deferred to v3.1)
- [ ] CPG_ad_panel_v1 candidate holdout validation on AddNeuroMed
- [ ] EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md update to reflect card v3.0
- [ ] LESSONS_LEARNED.md update
- [ ] TESTING_CHECKLIST.md update
- [ ] MASTER_TRACKER.md §2/§5/§7 updates (Heath-only, not pushed)
