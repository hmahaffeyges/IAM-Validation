# AD-immune Residual Maps

**Card:** ad-immune v3.1 (2026-06-05)
**README date:** 2026-06-05

This folder contains the operational artifacts of the AD-immune VAL series (CPG-VAL-008 through CPG-VAL-014). All three maps were generated with the current production stack: IAMAtlas REBUILD + Walther IAM Deconvolver + `iamatlas_celltype_markers_v0_2`.

## Files in this folder

| File | What it is | Stage 8 use | Anchor VAL |
|---|---|---|---|
| `ad_immune_residual_map_chr_annotated.csv` | Per-CpG residual d AIBL × AddNeuroMed cross-cohort | Step 8.2 per-card residual map overlap | CPG-VAL-013 |
| `ad_immune_pca_projections.csv` | PC1, PC2, ..., PC10 loadings on 115-cell A-score covariance | Step 8.3 multi-class pattern matching (PC1 T-cell axis check) | CPG-VAL-012 |
| `ad_immune_bimodality_map.csv` | **NEW 2026-06-05: FULL** per-CpG bimodality decomposition for 6,018 CpGs from AIBL (n=161 AD, n=471 HC). 2.3:1 gain:loss ratio. | Step 8.3 supporting evidence | derived from CPG-VAL-013 cohort |

## Headline numbers

### Residual map (CPG-VAL-013)
- **6,018 CpGs** scored in both cohorts (Walther class markers intersect)
- **271 CpGs** strong (|d| > 0.2) in both
- **241 of those 271 (88.9%)** are concordant in sign across cohorts
- **Spearman ρ(AIBL, AddNeuroMed) = 0.231** (p ≈ 10⁻⁷⁴)
- **AD residuals biased 4.8 : 1 negative direction** (hypomethylated > hypermethylated)
- **Schema:** `cpg, d_AIBL, d_AddNeuroMed, concordant_strong, mean_abs_d`
- **CPG_ad_panel_v1 candidate panel** (200 CpGs, 40 positive / 160 negative) was derived from this map. Formal seal + holdout validation outstanding to v3.2. Candidate panel staged at `validation_runs/CPG_VAL_013_AD_residual_map/CPG_ad_panel_v1_candidate.json`.

### PCA projections (CPG-VAL-012)
- PCA basis fit on **n = 471 HC samples** in AIBL (variance: PC1 67%, PC2 11%, PC3 6%, PC4 4%, PC5 3%)
- All 726 samples projected
- AD-vs-HC effect on each PC:
  - **PC1: d = −0.356** (p = 8 × 10⁻⁴) — the AD T-cell axis. Loadings dominated by CD4/CD8 T memory/naive + neutrophils. AD shifts NEGATIVELY → architectural T-cell exhaustion at the covariance level.
  - **PC3: d = +0.22** (p = 6 × 10⁻⁶) — secondary highly-significant axis
  - **PC10: d = −0.27** (p = 0.01) — tertiary
  - Other PCs near null
- **Note on PC rank:** PC1 is the T-cell axis in AIBL (AD cohort). In breast pre-dx GSE51057+GSE51032, PC2 is the T-cell axis. Same biology, different rank due to cohort age + cohort composition differences. The hypothesis to carry to future cards is "the T-cell axis will be a top PC" — not "PC2 will be the T-cell axis."

### Bimodality map (NEW 2026-06-05)

- **6,018 CpGs** with full per-CpG decomposition from AIBL cohort
- **673 CpGs GAIN bimodality** in AD cases vs HC (11.2%)
- **289 CpGs LOSE bimodality** in AD cases vs HC (4.8%)
- **2.3 : 1 gain : loss ratio** — similar pattern to breast pre-dx (2.77:1) but somewhat weaker
- **241 CpGs** are cross-referenced with cross-cohort residual concordant strong (`in_residual_concordant=True`)
- Columns: bc_hc, bc_case, delta_bc, mean_beta_hc, mean_beta_case, sd_beta_hc, sd_beta_case, delta_var, bimodal_in_hc, lost_in_case, loss_of_bimodality, in_residual_concordant
- Bimodality coefficient computed via Sarle's formula with Pfister-Schwarz correction; threshold for "bimodal" is BC > 5/9 ≈ 0.556

## How Stage 8 consumes these maps

Per SOP v1.2 Part II-C §66 (Step 8.2 — per-card residual map application):

1. Patient's foreground-cleaned β matrix from Stage 3 is the input
2. Pearson ρ between patient's per-CpG departure and the residual map's signed Cohen's d is computed
3. Fisher z-transform 95% CI on ρ
4. Per-cohort residual d columns (d_AIBL, d_AddNeuroMed) are AVERAGED when applying — the operational residual signature is the cross-cohort consensus

The card JSON `ad-immune_card_v3_1.json` declares matching rules in `stage_8_card_matching` with three routes:
- **Route AD** — positive Mahalanobis + Rule A panel positive + immune-class cellular age young
- **Route PSP/CBD** — negative Mahalanobis (BELOW_NORMAL compaction direction)
- **Route FTD** — intermediate Mahalanobis without strong Rule A signal

The residual map's overlap ρ feeds into all three routes as supporting evidence.

## Coverage requirements

Per the card's substrate spec, 450K samples require ≥80% CpG coverage of the EPIC superset. AddNeuroMed and GIFT cohorts pass this check (86-95% coverage); CPG-VAL-010 confirmed per-cell biology replicates on 450K but universal Mahalanobis attenuates — Stage 8 routes on 450K samples should weight per-cell findings over universal metric.

## What's pending in v3.2

- ~~CHR/MAPINFO genomic annotation columns~~ ✅ DONE 2026-06-05 (lookup from breast residual map; 100% CpG overlap)
- ~~Full bimodality decomposition~~ ✅ DONE 2026-06-05 (computed from AIBL cohort; see Bimodality map section above)
- CPG_ad_panel_v1 formal seal as standalone panel artifact + holdout validation on an independent AD cohort
- Prospective primary-care validation cohort (the big one for AD clinical deployment)
- Full N7 end-to-end chain-recovery (current N7 is simplified signal-level only)
- Stage 8 Path B mapping artifact v0.2

---

**Lineage.** These maps were generated by the AD-immune VAL series (CPG-VAL-008 through CPG-VAL-014) using the current production stack. They are NOT derived from any pre-build atlas. The 7-CpG Rule A panel is a separate operational artifact (documented in `ad-immune_card_v3_1.json` under `ad_disease_trained_panel`) and is the disease-trained AD discriminator — NOT a residual map. Pre-build evidence trail for the AD-immune card lineage is in `ad-immune_card_v3_1.json` under `pre_build_audit_lineage`.
