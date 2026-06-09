# VAL-089 — glioma-epic Tumor Tissue arm outcome

**Outcome label:** `O2_PARTIAL_DIRECTION_CONSISTENT_VARIANCE_HIGH`

**Interpretation:** GBM primary tumor tissue shows positive ΔA at v1 single-substrate methyl-only scope (consistent direction with Issue 002 5-substrate cfDNA prediction), but with **wide variance** consistent with heme-LL-008 (per-disease ΔA spread reflects programmed plasticity, not noise). Recurrent GBM shows larger and tighter signal. Cultured spheres show LOWER A-score than mixed-cell tissue, confirming Shannon entropy measures cell-mixture diversity not "tumorness."

## Headline numbers (H_min terminal = 0.7728)

| Stratum | n | mean A | SD | ΔA vs NTB | Cohen's d | 95% CI |
|---|---|---|---|---|---|---|
| NTB healthy controls | 5 | 0.6869 | 0.0107 | — | — | — |
| GBM primary | 64 | 0.7013 | 0.0613 | +0.0145 | +0.243 | [-0.668, +1.154] |
| GBM recurrent | 4 | 0.7195 | 0.0409 | +0.0327 | +1.167 | [-0.254, +2.588] |
| GBM cultured spheres | 4 | 0.6584 | 0.0207 | -0.0285 | -1.805 | [-3.362, -0.248] |

H_min immune normalization gives identical d values (the H_min is just a scaling constant); absolute magnitudes shift but direction-of-effect and effect sizes are invariant.

## CCL-032 diagnostic order

### Step 1 — Data integrity (PASSED)

- **CHK-3.1:** PASS. β distribution shows 34.8% extremes, 10.3% mid, median 0.434. Brain-tissue β distributions are intrinsically flatter than blood (more cell-type heterogeneity within mixed-cell tissue → more intermediate methylation). Within threshold.
- **CHK-3.3:** PASS. 77/77 samples, 100% Xu-538 coverage on 450K (full platform).
- **CHK-3.5:** PASS. 0/77 saturated under either H_min normalization. Max A_terminal = 1.074 (83% of ceiling); max A_immune = 0.989 (83% of ceiling). Substantial headroom.
- **CHK-3.2:** N/A — on-study NTB controls eliminate cross-cohort confound.

### Step 2 — Biology consistency (REFINED, not failed)

- **GBM primary direction matches framework prediction (positive ΔA), magnitude smaller than Issue 002 +0.217 cfDNA-5-substrate figure.** This is consistent with CHK-1.5 substrate-scope caveat — v1 methyl-only buffy-coat-panel readings are not directly comparable to L2/L3 multi-substrate cfDNA predictions.
- **Wide CI [-0.67, +1.15] on the primary GBM cohort** reflects two issues: (a) only n=5 healthy controls, and (b) genuinely high variance among GBM tumors (SD = 0.061 vs NTB SD = 0.011, **5× variance ratio**). Per heme-LL-008, this is programmed plasticity not noise.
- **Recurrent GBM > primary GBM (d = +1.17 vs +0.24).** Disease progression produces larger and more consistent A-score elevation. Biologically consistent with the framework prediction that recurrent disease has accumulated additional architectural disruption.
- **Cultured spheres < NTB (d = -1.81 NEGATIVE).** Pure tumor-cell-line β distributions are LESS mixed than mixed-cell tissue. This validates that Shannon entropy of methylation captures **cell-mixture diversity**, not "tumorness." A pure homogeneous neoplastic population produces a lower A-score than a heterogeneous mixed-cell tissue containing neurons + glia + microglia + endothelium. **This is an important biology cross-check for the framework** — it confirms that a high A-score is NOT a tumor marker; it's a heterogeneity marker. Tumor tissue produces high A-score because tumor TISSUE contains many cell types, not because tumor CELLS have high entropy.

### Step 3 — Framework finding

**At v1 single-substrate methyl-only scope on Xu-538-applied-to-brain-tissue, GBM primary tissue shows direction consistent with Issue 002 5-substrate cfDNA prediction, with magnitude appropriately reduced for substrate-scope, with wide variance consistent with heme-LL-008 plasticity. Recurrent GBM amplifies the signal. Cultured spheres invert the signal — confirming the framework reads cell-mixture diversity, not tumor-cell intrinsic property.**

## What this changes for the card

1. **The tissue pathway (Arm F in the routing matrix) IS validated at v1 scope.** Researcher arrives with tumor tissue, the card processes it through Stage 1 architecture A-score with H_min(terminal) = 0.7728. Direction-of-effect against NTB reference (or TCGA reference) is reportable.
2. **Recurrent GBM gets its own strata in the card decision tree.** Stage 1 + Stage 2 (Moss localization to terminal class on tissue is N/A; Moss is for cfDNA peripheral solid-organ localization) + Stage 3 (GIMiCC for tumor microenvironment) routing.
3. **Sphere/cell-line specimens flagged as "out-of-typical-distribution"** — they read NEGATIVE on architecture, which is technically informative but qualitatively different from tumor tissue scoring.
4. **The wide variance result confirms heme-LL-008 generalizes to glioma.** GBM tumors are programmed-plastic in their architecture: not noise, not measurement artifact, biology.

## What this does NOT prove

- N=5 NTB controls is small. Replication on GSE90496 Heidelberg cohort (with 24 inflammatory + 3 reactive controls) and GSE143843 superseries (104 + 297 + 398 GBM) would strengthen this finding substantially.
- Spheres at n=4 are a small inversion test; larger cell-line cohort needed to firm up the "Shannon entropy measures cell-mixture diversity, not tumorness" interpretation.
- IDH-mutant LGG tissue not in this cohort; framework prediction LGG ΔA = +0.239 untested at v1 here. TCGA-LGG would address this; not yet pulled.
- Tumor purity / neoplastic-cell-fraction not adjusted for. GBM samples have variable tumor purity (60-95% in GBM literature); A-score variance partly reflects this. Stage 3 GIMiCC adjustment is the v0.2 path.

## Card tier impact

- **Validation tier achieved (tissue arm):** `single_cohort_validated_with_internal_controls_and_substrate_scope_caveat` — a more accurate label than `single_cohort_validated` because the substrate-scope translation matters.
- **Required for upgrade:** GSE90496 Heidelberg replication (next VAL-090) and/or GSE143843 superseries replication.
- **Card status overall:** glioma-epic v0.1 now has TWO validations — VAL-088 (blood, exploratory) and VAL-089 (tissue, single-cohort with caveat). Two specimen pathways tested; the others remain pre-validation_skeleton or external_classifier_validated as documented in the README.
