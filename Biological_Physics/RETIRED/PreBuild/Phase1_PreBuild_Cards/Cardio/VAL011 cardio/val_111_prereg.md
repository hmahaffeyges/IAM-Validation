# VAL-111 Pre-registration — EpiSCORE HeartRef on Cardio-Epic Cohorts

**Sealed:** 2026-04-29 (before β access on cardio cohorts under EpiSCORE HeartRef atlas)
**Card under test:** cardio-epic (v0.1 native build, v0.2 rebuild target)
**Atlas:** EpiSCORE HeartRef (Zhu et al. Nat Commun 2022 13:3895), 3,727 unique 450K CpGs × 5 cardiac cell types (CM, EC, FB, MP, SMC), GPL-2 license
**Atlas SHA-256:** `bf6431f66749f02a616560764af3fdd0adc70b03bca96b2a13b6221bbd847c83`
**Atlas vault path:** `/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_heartref/`

## Hypothesis

Cardio-tile readings (cardiomyocyte CM, endothelial EC, fibroblast FB, macrophage MP, smooth-muscle SMC) on three cardio cohorts will demonstrate cohort-internal contrast between disease/control or between subtypes, with magnitudes consistent with the disease biology already characterized in VAL-108/109/110:

- **GSE84395** (PAH cultured PEC, hPAH vs iPAH vs control): EC tile expected to dominate (vascular substrate)
- **GSE84274** (aortic tissue, dissection vs normal): SMC tile expected to dominate (aortic media is smooth-muscle-rich); MP tile may rise in dissection (inflammation)
- **GSE69138** (stroke blood vs control): All five cardiac tiles expected near floor (blood substrate, no cardiac tissue contribution); this is a **negative control** for the atlas — non-cardiac substrates should not produce strong cardiac-tile signal.

## Procedure

1. For each cohort, load the cohort β matrix (already downloaded and verified pre-VAL-108/109/110, same GSE series matrices).
2. Restrict to CpGs in EpiSCORE HeartRef atlas (3,727 CpGs) ∩ cohort coverage.
3. For each sample, compute per-tile A-score = mean β across CpGs where atlas weight > 0 for that cell type, with within-cohort self-cal envelope from VAL-108/109/110 (CHK-3.1A pass already established, calibration anchor preserved).
4. Stratify by cohort metadata (subtype/disease state).
5. Report per-tile mean ± SD by stratum, plus per-sample CSV.
6. **No biology is downstream of cardio-epic v0.2 architecture decisions** — this VAL is a structural addition (atlas integration) on top of already-sealed VAL-108/109/110 structural results. The cardio-epic v0.1 H_min decisions and Stage 1+2 pipeline are not revisited.

## Pre-declared interpretation rules

- **Floor expectation for blood (GSE69138):** All five cardiac tiles should sit near 0 (cardiac cells are not present in peripheral blood at meaningful fraction). If any tile exceeds A=0.10 across the full GSE69138 cohort, this is a fingerprint of either (a) cross-reactivity in the atlas with blood-cell methylation, or (b) bridging artifact from Entrez→CpG mapping. Either way, this becomes a card-level documented limitation.
- **Tissue-substrate expectation:** GSE84274 aortic SMC tile and GSE84395 PEC EC tile are the positive expectations. Failure of these to dominate would indicate atlas-substrate mismatch (gene-promoter HeartRef may not generalize to dispensed cell-type panels in the way Loyfer/Moss tile-coverage atlases do).
- **No null is published as a positive.** If all five tiles read flat across all three cohorts, the VAL outcome is sealed as O1_NEGATIVE_ATLAS_RESOLUTION_INSUFFICIENT and EpiSCORE HeartRef is moved to atlases_deferred in cardio-epic v0.2 with rationale.
- **No bidirectional reframing.** AD-instance pattern only; cardio is not bidirectional.

## Pre-declared outcomes

- **O1_TILE_DISCRIMINATION_OBSERVED:** ≥1 cardiac tile shows ≥0.10 A-score difference between disease and control within at least one tissue cohort (GSE84395 or GSE84274), AND blood cohort (GSE69138) shows all five tiles below A=0.10 floor. → EpiSCORE HeartRef enters cardio-epic v0.2 atlases_run.
- **O2_PARTIAL_DISCRIMINATION:** Tissue cohorts show tile differentiation but blood floor is breached (any cardiac tile > A=0.10 in GSE69138). → EpiSCORE HeartRef enters cardio-epic v0.2 atlases_run with explicit blood-cross-reactivity caveat in card.
- **O3_TISSUE_FLOOR_DOMINATED:** No tile shows ≥0.10 A-score range in any tissue cohort. → EpiSCORE HeartRef moved to atlases_deferred for cardio-epic v0.3 pending alternative bridging or different atlas (e.g., Caggiano CelFiE TIM, currently blocked on HM450 hg19 manifest).
- **O4_BRIDGE_FAILURE:** <500 atlas CpGs survive intersection with cohort coverage in any cohort. → VAL sealed as engineering-failed, atlas re-bridging required, no biological inference.

## Pre-registration statement

This prereg is sealed before β access on these three cohorts under the EpiSCORE HeartRef atlas. The within-cohort self-cal envelopes from VAL-108/109/110 are inherited (same cohorts, same β matrices, same CHK-3.1A pass). VAL-111 adds an atlas layer; it does not alter VAL-108/109/110 conclusions. Outcome will be sealed by O1/O2/O3/O4 selector, not narrated. Language discipline applies: "consistent with", not "validates"; no overclaiming.

**Reproducibility triple (CHK-7.6):**
- Inline source: `val_111.py` (this directory)
- Inputs: cardio cohort series matrices (GSE69138, GSE84274, GSE84395 from GEO FTP — same files used by VAL-108/109/110), atlas CSV from vault
- Environment: Python 3.12, pandas, numpy
- Expected headline: per-tile mean A-score by cohort×stratum, blood-floor breach assessment, sealed outcome selector
