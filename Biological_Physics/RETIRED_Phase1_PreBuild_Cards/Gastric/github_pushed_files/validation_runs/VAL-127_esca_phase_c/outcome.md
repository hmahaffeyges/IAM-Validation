# VAL-127 Outcome — TCGA-ESCA Phase C run-everything

**Sprint:** gastric+esophageal-epic v0.1
**VAL ID:** VAL-127
**Prereg SHA:** `cb521d83afe8bee8136c73cf0e0526a9b5e60758df7a77ae51709000c4014b1e`
**Cohort:** TCGA-ESCA HM450 sesame Level 3 — n=185 primary tumor (96 ESCC + 89 EAC) + n=16 paired adjacent-normal + n=1 metastatic
**Anchor:** TCGA-KIRC + TCGA-PRAD adjacent-normal HM450 sesame Level 3 — n=210 (160 KIRC + 50 PRAD), shared with VAL-126
**Status:** SEALED 2026-05-02

---

## Primary outcome class

**`O1_SUBTYPE_DISCRIMINATION_PASS + O5_SUBSTRATE_BASELINE_TIER_3_DETECTED + O1_BARRETT_AMPLIFICATION_FIRES`**

ESCC-vs-EAC subtype discrimination FIRES strongly across Stage 1 + multiple Stage 2 tiles. Barrett's-positive samples show within-cohort Stage 1 amplification of +1.69 d-units over Barrett-negative. CHK-3.2 substrate baseline fires tier-3 on absolute d-values (4.31 anchor-SD shift), but within-cohort subtype + Barrett's contrasts are immune to baseline shift and form the trustworthy VAL-127 signal.

---

## CHK-3.2 substrate baseline finding

| Cohort | f_extreme | Note |
|--------|----------:|------|
| TCGA-ESCA tumor (n=185) | 0.4568 ± 0.0438 | Operational |
| TCGA-ESCA paired normal (n=16) | 0.4974 ± n/a | Underpowered as primary, descriptive |
| TCGA-KIRC + TCGA-PRAD anchor (n=210) | 0.5591 ± 0.0237 | Anchor |
| **ESCA tumor − anchor** | **−10.23 percentage points** | **−4.31 anchor-SD units** |
| **ESCA normal − anchor** | **−6.17 percentage points** | **−2.60 anchor-SD units** (also tier-2) |
| **CHK-3.2 tier (per prereg)** | **tier_3_invalidate_cross_cohort** | Pre-locked threshold ≥3 SD |

ESCA substrate sits BETWEEN gastric (STAD: f_extreme 0.44) and kidney/prostate (anchor: 0.56). The GI-continuum substrate gradient is real — esophageal adjacent-normal is closer to KIRC+PRAD anchor (2.6 SD) than gastric adjacent-normal was (5.0 SD), but tier-3 still fires on tumor.

**Operational consequence:** absolute d_vs_KIRC_PRAD_anchor values reported below carry tier-3 caveat. Within-cohort contrasts (ESCC vs EAC, Barrett-positive vs Barrett-negative, smoking-status strata) are immune to baseline shift and form the primary VAL-127 contribution.

---

## Headline 1: ESCC vs EAC subtype discrimination — FIRES

This is the gastric+esophageal-epic v0.1 card's headline test: can the run-everything atlas stack discriminate squamous-cell (ESCC) from adenocarcinoma (EAC) subtypes within ESCA on methylation alone?

### Stage 1 architectural drift
| Group | n | d vs anchor | Within-cohort | p |
|-------|--:|------------:|--------------|---|
| All tumor | 185 | +2.88 | — | 2.77e-81 |
| **ESCC** | **96** | **+2.64** | baseline | 9.24e-41 |
| **EAC** | **89** | **+3.70** | **EAC > ESCC by 1.06 d-units** | 9.39e-50 |
| **ESCC vs EAC contrast** | — | — | **d = −1.06, p = 1.50e-11** | — |

**EAC architectural drift exceeds ESCC by 1.06 d-units within the same cohort.** This is the within-cohort signal — substrate-tier-3 doesn't apply. EAC's Barrett's-driven chronic methylation drift produces a stronger architectural-drift signature than ESCC's acute squamous transformation.

### Top discriminating tiles (|d_ESCC-EAC| ≥ 0.5)

15 tiles fire as discriminating between ESCC and EAC. Direction in all top tiles is **EAC > ESCC** (negative d_ESCC-EAC):

| Tile | d_ESCC-EAC | ESCC vs anchor | EAC vs anchor |
|------|-----------:|---------------:|--------------:|
| A_cag_erythroblast | −2.26 | +1.10 | +3.48 |
| A_cag_eosinophil | −2.23 | +0.68 | +3.05 |
| A_cag_small_intestine | −2.17 | +0.72 | +3.17 |
| A_cag_megakaryocyte | −2.17 | +0.97 | +3.20 |
| A_cag_neutrophil | −2.09 | +0.75 | +2.89 |
| A_cag_dendritic | −2.09 | +0.86 | +2.93 |
| A_cag_macrophage | −2.08 | +1.02 | +3.08 |
| A_cag_monocyte | −2.03 | +0.85 | +2.90 |
| A_cag_tcell | −1.96 | +1.58 | +3.75 |
| A_cag_fibroblast | −1.93 | +0.81 | +3.15 |
| A_cag_endothelial | −1.91 | +1.28 | +3.57 |
| A_cag_placenta | −1.88 | +1.12 | +3.17 |
| A_cag_skeletal | −1.84 | +1.28 | +3.83 |
| A_loyfer_Head_and_neck_larynx | −1.30 | +1.12 | +2.76 |
| A_loyfer_Prostate | −1.12 | +1.79 | +2.93 |

The Caggiano TIM panels show consistently strong EAC > ESCC discrimination (~2 d-units). This pattern across 13+ Caggiano tiles is interpretable: EAC's Barrett's-driven chronic methylation drift produces a more thoroughly homogenized methylation pattern than ESCC's acute squamous transformation, so EAC tumors look more like generic-epithelial tissue across the Caggiano panel while ESCC retains more squamous-specific methylation structure.

---

## Headline 2: EsoRef squamous tile pattern — cell-of-origin retention IN ESCC

The EsoRef cross-tissue overread test is more nuanced in VAL-127 than in VAL-126 (STAD). EsoRef tiles read different directions for ESCC (squamous, target tissue) vs EAC (adenocarcinoma, non-target):

| Tile | ESCC vs anchor | EAC vs anchor | d_ESCC-EAC | Interpretation |
|------|---------------:|--------------:|-----------:|----------------|
| Epi_basal | +0.98 | +1.51 | −0.55 | Both positive, EAC stronger |
| **Epi_stratified** | **−0.99** | **−0.05** | **−0.68** | **ESCC NEGATIVE — cell-of-origin retention signal in target tissue** |
| Epi_suprabasal | +0.05 | +0.89 | −0.70 | ESCC near-null, EAC positive |
| Epi_upper | −1.51 | −2.39 | +0.66 | Both negative, EAC stronger negative |

**The ESCC d = −0.99 on Epi_stratified is the cell-of-origin retention signature predicted by the prereg.** ESCC tumors retain methylation patterns consistent with their squamous-stratified epithelial origin; EAC has lost that signature (d = −0.05 ≈ null). **EsoRef does read its target biology when applied to its target tissue.**

This reframes the VAL-126 STAD finding: the EsoRef cross-tissue overread observed on STAD (gastric adenocarcinoma) and EAC (esophageal adenocarcinoma) may not be atlas overread per se, but rather **Barrett's-derived squamous-memory drift propagating through columnar adenocarcinomas across the GI continuum**. The kidney-card cross-card test (running EsoRef on TCGA-KIRC tumor) becomes the discriminating experiment: if EsoRef reads NULL on KIRC (non-GI, non-Barrett's-related tissue), then VAL-127's ESCC-positive signal confirms tissue-of-origin specificity and the STAD signal is GI-tract methylation memory. If EsoRef reads strongly on KIRC, atlas overread is real.

This is a substantive update to the cross-tissue overread hypothesis logged for the kidney-card sprint.

---

## Headline 3: Barrett's-positive amplification — within-cohort signal

Within ESCA tumor cohort, Barrett's-positive samples show massively amplified Stage 1 architectural drift:

| Stratum | n | Stage 1 d vs anchor | Note |
|---------|--:|--------------------:|------|
| **Barrett+ (Yes-USA + Yes-UK)** | **28** | **+4.50** | **+1.69 d-units stronger than Barrett-negative** |
| Barrett− (No) | 118 | +2.81 | Baseline within ESCA |
| Barrett unreported | 39 | (descriptive) | — |

**Barrett's history amplifies the architectural-drift signature by +1.69 d-units within the same cohort.** Substrate-tier-3 doesn't apply because both groups share identical baseline. This is the cleanest within-cohort biological signal in VAL-127.

Biologically consistent: Barrett's esophagus is a chronic columnar metaplasia with progressive methylation drift accumulation; tumors arising from Barrett's substrate inherit and amplify that drift signature. **The Stage 1 cycling-class architectural-drift framework directly captures the Barrett's→EAC progression in methylation space.**

---

## Headline 4: Smoking does NOT track architectural drift in this cohort

Stage 1 d vs anchor by smoking status:

| Smoking status | n | d vs anchor |
|----------------|--:|------------:|
| Lifelong non-smoker | 56 | +3.03 |
| Reformed ≥15yr | 36 | +2.88 |
| Current smoker | 37 | +2.83 |
| Reformed <15yr | 37 | +2.43 |

**All four strata within 0.6 d-units of each other.** Smoking status is NOT a strong driver of architectural-drift signal in this cohort, in marked contrast to Barrett's status (1.69 d-unit amplification). 

This is itself a card-relevant finding: gastric+esophageal-epic v0.1's Stage 1 readout responds to chronic-methylation-drift drivers (Barrett's metaplasia, MSI-CIMP overlap) more strongly than to acute-mutagenic drivers (smoking-induced DNA damage). Distinguishes mechanism: methylation drift ≠ mutational burden.

---

## Stage 2 — Boccellato gastric tiles on ESCA

| Tile | d vs anchor (all ESCA tumor) | ESCC d | EAC d |
|------|-----------------------------:|-------:|------:|
| Antrum_undiff | (in JSON) | (per stratified) | (per stratified) |
| All 6 Boccellato tiles | mostly POSITIVE consistent with VAL-126 | varies | varies |

(Detailed per-tile breakdown in `VAL-127_phase_c_results.json`. Boccellato is a gastric atlas applied to esophageal tissue — expected to show baseline-shift-driven positive bias on EAC similar to STAD, weaker on ESCC. Detailed CCL-type interpretation of Boccellato-on-ESCA reserved for v0.2 substrate-matched analysis.)

### Loyfer 25-tile findings
- Loyfer Bladder/Lung/Hepatocyte/Pancreatic_beta: POSITIVE direction expected, FIRES (consistent with STAD pattern)
- Loyfer Head_and_neck_larynx: ESCC d=+1.12, EAC d=+2.76 — **ESCC reads weaker on H&N larynx than EAC**, somewhat counterintuitive (H&N larynx is squamous tissue closer to ESCC origin); may reflect Loyfer-tile substrate-shift bias

---

## Pre-locked outcome class assignments

| Pre-locked outcome | Status |
|-------------------|--------|
| O1_STAGE1_PASS (all tumor) | **FIRES** (d=+2.88, p=2.77e-81) |
| ESCC vs EAC subtype discrimination | **FIRES STRONGLY** (d=−1.06 on Stage 1; 15+ Stage 2 tiles |d|≥0.5) |
| Cell-of-origin tile retention in ESCC | **FIRES** on EsoRef Epi_stratified (d=−0.99 in target tissue) |
| Barrett's amplification | **FIRES** (Barrett+ d=+4.50 vs Barrett− d=+2.81; Δ=+1.69) |
| Smoking stratification | **NULL** — all strata within 0.6 d-units (informative null) |
| O5_SUBSTRATE_BASELINE_TIER_3 | **FIRES** (4.31 anchor-SD shift; less severe than STAD's 5.02) |

---

## Comparison to VAL-126 STAD findings (joint card synthesis)

| Finding | VAL-126 STAD | VAL-127 ESCA |
|---------|-------------|--------------|
| Stage 1 d (all tumor) | +3.34 | +2.88 |
| Substrate baseline shift | -5.02 anchor-SD | -4.31 anchor-SD (less severe) |
| Within-cohort subtype hierarchy | MSI ≈ EBV > CIN > GS | EAC > ESCC by 1.06 d |
| Cell-of-origin direction | Boccellato all POSITIVE (tier-3 caveat) | EsoRef Epi_stratified NEGATIVE in ESCC ✓ |
| Cross-tissue overread | EsoRef + OEref fire on STAD | EsoRef cross-tissue overread = adenocarcinoma-specific (EAC fires, ESCC retains target signal) |
| Risk factor amplifier | H. pylori (n=20, exploratory) | **Barrett's (+1.69 d-units within cohort)** |
| Stage 3 immune | T-cell + myeloid depletion | (pending detailed analysis in stratified JSON) |

**Joint card narrative:** the gastric+esophageal-epic v0.1 framework discriminates not just disease vs healthy but also subtype-within-disease in two distinct cohorts, with within-cohort signals robust to substrate baseline shift. Risk-factor stratification (Barrett's amplification) emerges from the same architectural-drift framework without specialized panels.

---

## Logged follow-ups (NOT in scope of this VAL)

1. **Kidney-card cross-card calibration sprint (already on roadmap, now stronger motivation):** EsoRef + OEref tested on TCGA-KIRC tumor + TCGA-PRAD tumor. VAL-127 establishes that EsoRef CAN read target biology in target tissue (ESCC Epi_stratified d=−0.99). The kidney-card test now has a sharper hypothesis: if EsoRef reads NULL on KIRC, the cross-tissue overread observed on STAD/EAC is Barrett's/GI-continuum methylation memory, not generic atlas overread. If EsoRef reads strong on KIRC, generic atlas overread is real. The test was discriminative before; now it's also informative about a third hypothesis (GI-continuum methylation memory).

2. **Substrate-matched gastric/esophageal anchor:** Future VAL pulling additional GI adjacent-normal HM450 cohorts (GSE99553, GSE52826, healthy gastric biopsies) to bypass tier-3 caveat for absolute Boccellato direction interpretation. v0.2 priority.

3. **Barrett's-progression methylation timeline:** Within-cohort Barrett+ vs Barrett− contrast (+1.69 d-units) suggests methylation drift accumulates progressively during the metaplasia→dysplasia→adenocarcinoma sequence. A future VAL using GSE104707 or similar Barrett's-with-progression cohorts could test whether the Stage 1 A-score scales monotonically with Barrett's→dysplasia→EAC progression. Logged as candidate v0.2 study.

---

## Reproducibility (CHK-7.6)

- **Source code:**
  - `val127_esca_phase_c.py` — ESCA scorer (504 lines)
  - `val127_results_rebuild.py` — d-value computation + ESCC/EAC subtype discrimination + Barrett's stratification (300 lines)
  - Anchor scoring: shared with VAL-126 via `val106_anchor_per_sample.ndjson`
- **Inputs:**
  - 202 TCGA-ESCA HM450 sesame Level 3 β files (manifest + MD5 in `tcga_esca_hm450_manifest_FINAL.json`)
  - 210 KIRC+PRAD adjacent-normal anchor (shared with VAL-126)
  - 8 calibrated atlases (SHA-sealed in atlas_vault INVENTORY.json)
  - cBioPortal clinical: `cbioportal_esca_pt_clin_full.json`
- **Environment:** Python 3, NumPy, scipy.stats
- **Headline outputs:** `VAL-127_phase_c_results.json`, `VAL-127_stratified_results.json`, `VAL-127_per_sample.csv`

---

## Final language

VAL-127 finds robust ESCC-vs-EAC subtype discrimination in TCGA-ESCA on methylation alone using the run-everything atlas stack, with EAC architectural drift exceeding ESCC by 1.06 d-units within cohort. Barrett's-history-positive tumors amplify the Stage 1 signal by 1.69 d-units over Barrett-negative — a within-cohort signal robust to substrate baseline shift. EsoRef Epi_stratified reads negative direction in ESCC (cell-of-origin retention in target tissue) while losing that signature in EAC, reframing the cross-tissue overread observed on STAD and EAC as potentially adenocarcinoma-specific Barrett's-derived methylation memory rather than generic atlas overread. Smoking does NOT track architectural drift in this cohort, while Barrett's does — distinguishing methylation-drift mechanisms from mutational-burden mechanisms. The CHK-3.2 tier-3 substrate baseline shift is documented for absolute cross-cohort d-values; within-cohort subtype and Barrett's contrasts form the primary VAL-127 contribution.
