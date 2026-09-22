# AD-immune residual maps (card v3.0)

Three artifacts mirroring the breast-epic residual_maps pattern:

## `ad_immune_residual_map_chr_annotated.csv`

Per-CpG residual effect size on AD-vs-HC, computed in TWO cohorts.

Schema: `cpg, d_AIBL, d_AddNeuroMed, concordant_strong, mean_abs_d`

Note: This v3.0 emission lacks CHR/MAPINFO columns (breast had them via the 450K manifest). Genomic-annotation merge is deferred to v3.1; the per-CpG d values are the operative signal.

- 6,018 CpGs scored in both cohorts (Walther class markers intersect)
- AIBL (n=161 AD / 471 HC): full-cohort residual effect sizes
- AddNeuroMed (n=93 AD / 96 HC): cross-platform replication
- `concordant_strong = True` if |d| > 0.2 in BOTH cohorts AND same sign
- 271 CpGs strong in both; 241 concordant in sign (88.9% concordance rate when strong)
- Spearman ρ(AIBL, AddNeuroMed) = 0.231 (p = 10^-74)
- Per-CpG residual = observed β − class-fraction-predicted β (i.e. the part of β NOT explained by Walther's class-level deconvolution; AD-vs-HC differences in this residual are the IAM-native AD signature at the CpG level)
- CPG_ad_panel_v1 candidate panel (200 CpGs, 40+/160-) was derived from the AIBL residual map and is staged at `/CPG-VAL-013/CPG_ad_panel_v1_candidate.json` in the validation_runs/CPG_VAL_013_AD/ folder

## `ad_immune_pca_projections.csv`

CPG-VAL-012 — AIBL PCA on 115-cell A-score covariance, fit on HC samples, all samples projected.

Schema: `sentrix, arm, gender, PC1, PC2, ..., PC10`

- PCA basis fit on n=471 HC samples (variance: PC1 67%, PC2 11%, PC3 6%, PC4 4%, PC5 3%)
- All 726 samples projected
- AD-vs-HC effect on each PC:
  - **PC1 d=−0.356** (p=8e-4) — strongest AD axis; T-cell-dominated loadings (CD4/CD8 T memory/naive, neutrophils all positive). AD shifts NEGATIVELY → architectural T-cell exhaustion at the covariance level. Same biology as breast PC2 T-cell axis but at PC1 because cohort structure differs.
  - **PC3 d=+0.22** (p=6e-6) — secondary highly-significant axis
  - PC10 d=-0.27 (p=0.01) — tertiary
  - Other PCs near null

## `ad_immune_bimodality_map.csv`

Placeholder. Bimodality decomposition deferred to v3.1 — current v3.0 carries the header only.
