# Cellular Performance Gauge — Patient Report

**Patient ID:** GSM-MOCK-001 *(mock patient for report design review — not a real patient)*
**Report date:** 2026-06-08
**Sample date:** 2026-05-29
**Substrate:** Whole blood, Illumina EPIC 850K array, paired IDAT (red + green channel)
**Reading institution:** IAMPerformance Inter-Domain Research Institute, Entiat WA
**Engine:** Walther Clinical Pipeline v1.0 · IAMAtlas · 483,092 CpG resolution
**Reading purpose:** Wellness baseline — naturopath-referred annual check
**Sample number in patient series:** 1 of 1 (first sample — establishes baseline only; serial sampling unlocks the deeper analytics described in Section K)

---

> ## ⚠ DESIGN MOCK — NUMERIC VALUES ARE NOT DERIVED
>
> This is a structural and visual mock built for design review. The patient identifier (GSM-MOCK-001), demographics, all A-scores, cellular age estimates, Mahalanobis distance, composition percentages, disease detection probabilities, and every other numeric value in this document were **fabricated by Walther** to illustrate report layout and section flow.
>
> They are **NOT** derived from any chain computation. They are **NOT** sampled from published cohort distributions. They are **NOT** defensible as methodology examples. They function **only** as visual placeholders showing what each section will look like once `walther_clinical.py` is built and run on a real patient.
>
> **Do NOT** propagate these mock values as methodology references, calibration baselines, training examples, or anchors of any kind. They have no provenance.
>
> When the real first patient runs, the actual numbers will come from the chain and will be fully derived and defensible.

---

# Executive summary — in plain language

This is your **first cellular performance reading**. The chain analyzed the methylation pattern across **115 individual cell types** in your whole blood sample and computed how each cell's architecture compares to the healthy reference for someone your age.

**Bottom line:** No disease patterns are flagged above clinical threshold. Your overall cellular age reads **62.2 years** against your chronological age of 60, an acceleration of **+2.2 years**. The pattern driving this acceleration is a **coordinated immune-aging signature** (mild inflammaging) — five of your top nine most-shifted cells are immune-aging cell types (exhausted CD8+ T cells, senescent memory B cells, effector memory T cells, M2 macrophages, NK cells). A secondary pattern of mild age-related epithelial drift is visible in three secretory cell types (mammary luminal, pancreatic ductal, bronchial epithelial). **No cells in Warburg tier. No cells in Breach tier.** Top 3 cells are in ELEVATED tier; rest are within NORMAL.

**What this means clinically:** This is a **normal-for-age pattern with a mild inflammaging acceleration above the typical-for-age trajectory**. It is not a disease signal — it is a fitness signal. The age-axis foreground subtraction (Stage 3) already accounts for what's typical at age 60, so the residual elevation in two immune-aging cells represents drift slightly above the average aging trajectory, not below it. The same pattern, observed at sample 2 in six to twelve months, will tell us whether the inflammaging acceleration is **stable** (continuing at its current mild pace), **accelerating** (warrants intervention), or **reversing** (responding to lifestyle changes).

**What you do next:** This first reading establishes your baseline. The real analytic power comes from serial sampling — see Section K for what each subsequent sample unlocks.

---

# A. Sample integrity and intake context

| Field | Value | Status |
|---|---|---|
| IDAT pair received | Red + Green channels | ✓ Both files present |
| Bisulfite conversion | 99.4% | ✓ Pass (threshold ≥98%) |
| Detection p-rate failures | 0.6% of CpGs | ✓ Pass (threshold ≤2%) |
| Predicted sex from chrY methylation | Female | ✓ Match to intake |
| Sample-to-sample contamination check | Clean | ✓ Pass |
| Coverage of IAMAtlas 483,092 CpGs | 99.97% (passed forward) | ✓ Pass |
| Substrate normalization | sesame noob + dye-bias correction | Applied |

**Intake fields (patient-provided):**

| Field | Value |
|---|---|
| Chronological age | 60 |
| Sex assigned at birth | Female |
| Self-reported menopause status | Post-menopausal (since age 52) |
| Smoking history | Never smoker |
| Self-reported ancestry | US Caucasian |
| Family cancer history | Not provided *(declined; risk multipliers in Section L not applied)* |
| Current medications | None reported |
| Current diagnoses | None reported |
| Reason for testing | Annual wellness baseline |

---

# B. Cellular composition — every cell detected, with normal ranges

The deconvolution stage identifies every detectable cell type from your blood sample and quantifies each one's representation. This is the **composition view** — who is there, in what proportion. The analysis of what those cells are doing comes in Section C.

**Why composition matters as its own diagnostic signal:** abnormal composition can indicate disease even before methylation architecture shifts. A shedding tumor presents as elevated fractions of its tissue-of-origin cells appearing in blood. An active leukemia presents as abnormal immune-cell ratios. An autoimmune flare presents as elevated activated T-cell subtypes. Composition is read first; architecture (Section C) is read on whatever cells are present.

## B.1 Summary across the 8 architectural classes

| Architectural class | Cells detected | Patient composition | Normal range (60yo F whole blood) | Status |
|---|---:|---:|---|---|
| Immune | 18 detected (29 cataloged) | 87.3% | 80–92% | ✓ Within range |
| Stromal | 4 detected (9 cataloged) | 4.2% | 2–6% | ✓ Within range |
| Secretory | 3 detected (17 cataloged) | 3.6% | 1–5% (residual cross-organ trace) | ✓ Within range |
| Cycling | 3 detected (14 cataloged) | 2.1% | 1–4% | ✓ Within range |
| Stem (adult) | 2 detected (7 cataloged) | 1.4% | 0.5–3% | ✓ Within range |
| Progenitor | 2 detected (8 cataloged) | 0.9% | 0.3–2% | ✓ Within range |
| Terminal | 2 detected (22 cataloged) | 0.4% | 0.1–1% (residual cross-organ) | ✓ Within range |
| Stem (pluri) | 0 detected (9 cataloged) | 0.0% | 0% (not expected in adult blood) | ✓ As expected |

*The IAMAtlas catalogs 115 cell types organized across 8 architectural classes. "Detected" means the methylation signal for that cell type was identifiable above the deconvolution threshold in this sample.*

## B.2 Detailed per-cell composition — every detected cell

The normal ranges below are age + sex + substrate adjusted (60yo female whole blood EPIC). Values are percent of total whole-blood signal. Flagged cells get explicit attention; unflagged cells are within their normal range and clinically uninteresting at the composition level (but their **methylation architecture** is still analyzed in Section C).

### B.2.1 Immune class (18 detected)

| Cell type | Patient % | Normal range | Status |
|---|---:|---:|---|
| Neutrophils (mature) | 56.2% | 50–65% | ✓ Within range |
| CD4+ memory T cells | 8.4% | 5–12% | ✓ Within range |
| CD8+ memory T cells | 5.1% | 3–7% | ✓ Within range |
| Effector memory CD4+ T cells | 5.8% | 3–8% | ✓ Within range |
| NK cells CD56dim | 5.4% | 4–10% | ✓ Within range |
| Classical monocytes | 4.6% | 3–7% | ✓ Within range |
| **Exhausted CD8+ T cells** | **3.2%** | **0.5–3.0%** | ⚠ **Slightly elevated** (upper edge) |
| Naive CD4+ T cells | 2.8% | 1–5% | ✓ Within range |
| Macrophages M2 (tissue-resident, residual) | 2.1% | 0.5–3% | ✓ Within range |
| Naive CD8+ T cells | 1.6% | 1–3% | ✓ Within range |
| B cells naive | 1.4% | 1–3% | ✓ Within range |
| Eosinophils | 1.2% | 0.5–3% | ✓ Within range |
| **Senescent memory B cells** | **0.9%** | **0.1–0.8%** | ⚠ **Slightly elevated** (above range) |
| B cells memory | 0.7% | 0.5–2% | ✓ Within range |
| Intermediate monocytes | 0.5% | 0.2–1% | ✓ Within range |
| Plasmacytoid dendritic cells | 0.2% | 0.05–0.5% | ✓ Within range |
| Non-classical monocytes | 0.2% | 0.1–0.6% | ✓ Within range |
| Regulatory T cells (Treg) | 0.1% | 0.1–0.5% | ✓ Within range (lower edge) |
| MAIT cells | not detected | 0.05–0.5% | (below detection threshold — not clinically meaningful) |

**Immune composition flag:** Two cells slightly elevated — **exhausted CD8+ T cells** (3.2% vs normal 3.0% ceiling) and **senescent memory B cells** (0.9% vs normal 0.8% ceiling).

**What this means in context:** The normal ranges above are **age + sex + substrate adjusted** — they already account for the expected immune-cell shifts of a 60-year-old postmenopausal female (the typical age-related rise in exhausted T cells and senescent B cells is built into the upper bound). Values above the upper bound indicate **above-typical-for-age** — mild inflammaging acceleration above the average aging trajectory, not just inflammaging at the expected rate. The deltas are small (3.2 vs 3.0 ceiling; 0.9 vs 0.8 ceiling) — this is consistent with **mild inflammaging acceleration, not advanced immune dysregulation**.

This composition-level finding is **independently consistent** with the methylation-architecture finding in Section C (same cells, ranks 1 and 2 of top 15 by departure). Two different signals from the same biology — composition AND methylation architecture — both pointing to mild above-typical-for-age immune-aging.

### B.2.2 Stromal class (4 detected)

| Cell type | Patient % | Normal range | Status |
|---|---:|---:|---|
| Vascular endothelial | 2.1% | 1–3% | ✓ Within range |
| Adipose tissue stromal | 1.4% | 0.5–2% | ✓ Within range |
| Bone marrow stromal | 0.5% | 0.3–1% | ✓ Within range |
| Fibroblasts (residual) | 0.2% | 0.1–0.8% | ✓ Within range |

### B.2.3 Secretory class (3 detected — residual cross-organ trace)

| Cell type | Patient % | Normal range | Status |
|---|---:|---:|---|
| Mammary luminal epithelial (residual) | 1.4% | 0.5–2% | ✓ Within range |
| Pancreatic ductal epithelial (residual) | 1.2% | 0.4–1.8% | ✓ Within range |
| Bronchial epithelial (residual) | 1.0% | 0.3–1.5% | ✓ Within range (upper) |
| Endometrial glandular (residual) | not detected | 0–1% | (below threshold; expected post-menopause) |
| Pancreatic acinar (residual) | not detected | 0–0.5% | (below threshold) |
| Pancreatic islet (residual) | not detected | 0–0.5% | (below threshold) |
| Mammary basal epithelial (residual) | not detected | 0–0.8% | (below threshold) |
| (10 other secretory cell types not detected — appropriate for whole-blood substrate) | — | — | — |

**Secretory composition flag:** All three detected cells within normal residual-trace range. **A shedding tumor of secretory origin would surface here** as a fraction substantially above the upper bound of the normal range (e.g., a mammary luminal reading of 4–8% in blood rather than 0.5–2%). That is not what this patient is showing.

### B.2.4 Cycling class (3 detected)

| Cell type | Patient % | Normal range | Status |
|---|---:|---:|---|
| Bone marrow proliferating progenitor | 0.9% | 0.3–1.5% | ✓ Within range |
| Colon crypt base columnar (residual) | 0.7% | 0.2–1.2% | ✓ Within range |
| Hair follicle (residual) | 0.5% | 0.1–0.8% | ✓ Within range |

### B.2.5 Stem (adult) class (2 detected)

| Cell type | Patient % | Normal range | Status |
|---|---:|---:|---|
| HSC (hematopoietic stem) | 0.9% | 0.3–1.5% | ✓ Within range |
| Mesenchymal stem (residual) | 0.5% | 0.2–1% | ✓ Within range |

### B.2.6 Progenitor class (2 detected)

| Cell type | Patient % | Normal range | Status |
|---|---:|---:|---|
| BM myeloid progenitor | 0.5% | 0.2–1% | ✓ Within range |
| Common lymphoid progenitor | 0.4% | 0.1–0.8% | ✓ Within range |

### B.2.7 Terminal class (2 detected — residual cross-organ)

| Cell type | Patient % | Normal range | Status |
|---|---:|---:|---|
| Cortical neuron (residual) | 0.2% | 0.05–0.5% | ✓ Within range |
| Hepatocyte (residual) | 0.2% | 0.05–0.5% | ✓ Within range |
| (20 other terminal cell types not detected — appropriate for whole-blood substrate) | — | — | — |

### B.2.8 Stem (pluri) class (0 detected — expected for adult blood)

| Cell type | Patient % | Normal range | Status |
|---|---:|---:|---|
| (9 pluripotent cell types not detected) | 0% | 0% | ✓ As expected for adult sample |

**Pluripotent cells in adult peripheral blood would be a remarkable finding** (potentially indicating a teratoma, germ cell tumor, or contamination). Their absence here is the expected and reassuring reading.

## B.3 Composition-level summary

**Two flags at the composition level — both consistent with inflammaging:**

1. Exhausted CD8+ T cells at 3.2% (normal ceiling 3.0%)
2. Senescent memory B cells at 0.9% (normal ceiling 0.8%)

**No flags at the composition level for tumor-shedding, leukemia, autoimmune flare, or pluripotent contamination.** All other 33 detected cells are within their normal ranges for a 60yo postmenopausal female whole-blood sample.

The methylation analysis in Section C uses every detected cell. The slightly elevated composition of the two inflammaging-marker cells in B.2.1 reinforces the methylation-architecture finding that those same cells are also showing departure in their methylation pattern (Section C ranks 1 and 2). Two independent signals from the same biology.

---

# C. Architectural state — the cell-level view

Each of the 115 atlas cell types gets two readings:

1. An **A-score** — a continuous number quantifying how well the cell's methylation matches the healthy class baseline. A = 1.00 is the healthy class baseline; values above 1.00 indicate departure.
2. A **6-tier verdict** — translating the continuous score into clinically meaningful categories.

## C.1 The reference scale — what each A-score value means

```
SUPPRESSED   NORMAL          ELEVATED    SIGNIFICANTLY_ELEVATED   BREACH
< 0.95       0.95 ──── 1.04  1.04─1.07   1.07 ──────── 1.10       ≥ 1.10
                                  │       ▲                        │
                                  │       │ WARBURG line @ 1.07     └── architectural fidelity loss
                                  │       │ (metabolic strategy must change)
                                  └── measurable departure begins

   Reference clusters past breach: senescence ≈ 1.24–1.27, malignancy ≈ 1.28–1.32
   ≥ 1.20 — pre-diagnostic active malignancy magnitude (reference annotation only, not a tier)
```

*The full visual gauge is in Appendix A1.*

## C.2 Top 15 cells by magnitude of departure — the cell-level data view

This is the most important visual in the report. Class averages would hide the signal — the pattern lives in which specific cells move and which way.

| Rank | Cell type | Class | A-score | Tier | 95% CI |
|----:|---|:---:|---:|---|---|
| 1 | Exhausted CD8+ T cells | IMM | **1.054** | ELEVATED | [1.042 — 1.066] |
| 2 | Senescent memory B cells | IMM | **1.048** | ELEVATED | [1.037 — 1.059] |
| 3 | Mammary luminal epithelial | SEC | **1.044** | ELEVATED | [1.032 — 1.056] |
| 4 | Effector memory CD4+ T cells | IMM | 1.041 | NORMAL (upper) | [1.030 — 1.052] |
| 5 | Pancreatic ductal epithelial | SEC | 1.038 | NORMAL (upper) | [1.026 — 1.050] |
| 6 | Macrophages M2 (residual) | IMM | 1.036 | NORMAL (upper) | [1.026 — 1.046] |
| 7 | Bronchial epithelial (residual) | SEC | 1.034 | NORMAL | [1.022 — 1.046] |
| 8 | NK cells CD56dim | IMM | 1.032 | NORMAL | [1.022 — 1.042] |
| 9 | Plasmacytoid dendritic cells | IMM | 1.030 | NORMAL | [1.021 — 1.039] |
| 10 | Adipose tissue stromal | STR | 1.026 | NORMAL | [1.016 — 1.036] |
| 11 | Vascular endothelial | STR | 1.023 | NORMAL | [1.013 — 1.033] |
| 12 | CD4+ memory T cells | IMM | 1.022 | NORMAL | [1.011 — 1.033] |
| 13 | CD8+ memory T cells | IMM | 1.020 | NORMAL | [1.010 — 1.030] |
| 14 | BM myeloid progenitor | PRO | 1.019 | NORMAL | [1.009 — 1.029] |
| 15 | Colon crypt base columnar (residual) | CYC | 1.018 | NORMAL | [1.005 — 1.031] |

**Cells 16 through 115:** all within ±2 SD of healthy class baseline at this patient's age. Not individually surfaced here; the complete per-cell A-score table is at Appendix D.

**Pattern visible at a glance:** Top 3 cells in ELEVATED tier are 2 immune-aging cells (exhausted CD8, senescent B) and 1 secretory epithelial (mammary luminal). Of the top 9 cells, 5 are immune-aging types and 3 are secretory epithelial. **No cells in Warburg tier. No cells in Breach tier.** This is the visual signature of *subtle* drift — coordinated direction is visible, but magnitudes stay well below clinical alarm thresholds.

*Visual representation: see Appendix A2 (cellular departure ranking — horizontal bar chart matching the reference gauge zones).*

## C.3 Class-level summary — reference only

The class-level view is presented for completeness but **does not drive the clinical reading**. The class average aggregates all contributing cells of that class and can mask the bidirectional patterns that the cell-level view reveals.

| Class | Mean A-score (all contributing cells) | Median A-score | n cells contributing | Class tier | Note |
|---|---:|---:|---:|---|---|
| Immune | 1.021 | 1.018 | 18 | NORMAL | But cell-level view in C.2 shows ranks 1, 2, 4, 6, 8, 9 are aging-immune-specific cells — the **pattern** is age-coordinated and only visible at cell-level resolution. Class mean reads NORMAL even though the inflammaging pattern is real. |
| Secretory | 1.039 | 1.038 | 3 | NORMAL (upper) | All 3 contributing cells (mammary luminal, pancreatic ductal, bronchial epithelial) are mid-elevated; coordinated epithelial-drift pattern |
| Stromal | 1.019 | 1.019 | 4 | NORMAL | Within range |
| Cycling | 1.017 | 1.018 | 3 | NORMAL | Within range |
| Progenitor | 1.018 | 1.018 | 2 | NORMAL | Within range |
| Stem (adult) | 0.990 | 0.990 | 2 | NORMAL | Within range |
| Stem (pluri) | n/a | n/a | 0 | n/a | Not present in adult blood |
| Terminal | 1.004 | 1.004 | 2 | NORMAL | Within range |

**Class-average warning — exactly what we predicted:** Look at the immune class above. Class mean A = 1.021, tier NORMAL. If we reported only this number, the inflammaging pattern (visible in C.2 as 5 of 9 top-ranked cells being immune-aging types) would be **completely invisible**. Only secretory reads as upper-NORMAL because all 3 contributing cells are coordinately elevated. **This is why the report leads with cell-level data and demotes class-level to reference.** See Section H.5 (Pattern Recognition) for the named patterns the cell-level view reveals.

---

# D. Cellular aging — total departure from age-adjusted normal

## D.1 Methodology

Cellular age is computed as the **confidence-weighted absolute sum of per-cell departures from age-adjusted normal** across all 115 atlas cell types:

```
Total_Cellular_Departure = Σ over all 115 cells [ |A_patient(cell) − A_ref(cell, chrono_age)| × (1 / posterior_SD(cell)) ]
                                                  ────────────────────────────────────────  ───────────────────────
                                                              cell departure                       confidence weight
                                                              from age-adjusted normal             (stable cells dominate)
```

The total maps via the Stage 6 calibration to a cellular age estimate.

**Why this approach:** Stable cells with tight posterior CIs (high signal-to-noise) dominate the sum. Noisy cells with wide CIs contribute proportionally less. Bidirectional cancellation is avoided by using absolute departure. The architecture class tells us the **range of operation** for each cell type (the H_min anchor); the cell tells us **whether this individual cell is shifted from where it should be at this patient's chronological age**.

## D.2 Your cellular age

| Measure | Value | 95% CI |
|---|---:|---|
| Chronological age | 60.0 | (provided) |
| **Total cellular departure (115 cells)** | **22.4 confidence-weighted units** | [20.5 — 24.3] |
| **Cellular age (Stage 6 inverted)** | **62.2 years** | [61.4 — 63.0] |
| **Age delta (cellular − chronological)** | **+2.2 years accelerated** | [+1.4 — +3.0] |
| Inflammaging quantum (immune-class contribution to total departure) | **+1.8 years** of the +2.2 total | dominant driver |
| OSK-direction tracking (single-sample) | n/a — requires sample 2 | unlocks Section K |

**Interpretation:** Your cellular age reads 2.2 years older than your chronological age. **82% of the age acceleration is driven by the immune-class inflammaging pattern** (Section C.2 ranks 1, 2, 4, 6, 8, 9 — exhausted CD8, senescent B, effector memory CD4, M2 macrophages, NK CD56dim, plasmacytoid dendritic). The remaining ~18% is split between mild secretory drift and small contributions across other classes.

This +2.2yr delta is **measurable but mild**. It is consistent with the cell-level data (top 3 cells in ELEVATED tier, no cells in Warburg or Breach) and with the composition-level data (two immune-aging cells slightly above the age-adjusted ceiling). All three independent layers — composition, methylation architecture, cellular age — agree on the same finding: **mild inflammaging acceleration above the typical-for-age trajectory**.

## D.3 Per-class cellular ages — reference only

| Class | Per-class cellular age | Delta vs chrono | Driver |
|---|---:|---:|---|
| Immune | 63.4 | +3.4 | Aging-immune cells (top 5 of top 9 in C.2) |
| Secretory | 61.5 | +1.5 | Epithelial drift across all 3 detected cells |
| Stromal | 60.4 | +0.4 | Within typical-for-age |
| Cycling | 60.2 | +0.2 | Within typical-for-age |
| Progenitor | 60.3 | +0.3 | Within typical-for-age |
| Stem (adult) | 59.4 | −0.6 | Slightly suppressed (within range) |
| Stem (pluri) | n/a | n/a | Not present |
| Terminal | 59.5 | −0.5 | Within typical-for-age |
| **Combined cellular age (confidence-weighted across all 115 cells)** | **62.2** | **+2.2** | **Immune-aging dominant** |

---

# E. Universal architectural departure — Mahalanobis distance

The Mahalanobis distance measures how far this patient's 115-dimensional cell-type A-score vector sits from the centroid of the healthy reference hull. It compresses everything in Section C into a single number with a unit (standard deviations in covariance-aware space) and a built-in reference distribution.

| Measure | Value |
|---|---:|
| Mahalanobis distance from healthy hull | **7.5** |
| Healthy hull reference | v0.5, n=2,523 HC across 8 cohorts and 4 populations |
| Hull p95 threshold (default flag) | 13.62 |
| Hull p99 threshold (strict flag) | 18.59 |
| Status | **WITHIN HULL** (well below default flag) |
| Top 10 cell-types driving the distance | 1. Exhausted CD8+ T cells (11%), 2. Senescent memory B cells (9%), 3. Mammary luminal epithelial (8%), 4. Effector memory CD4+ T cells (7%), 5. Pancreatic ductal (6%), 6. Macrophages M2 (5%), 7. Bronchial epithelial (5%), 8. NK CD56dim (4%), 9. Plasmacytoid dendritic (4%), 10. Adipose tissue stromal (3%) |
| Imputation count | 0 features (all 112 dimensions measured directly) |
| Cross-platform attenuation expected | Minimal (this is EPIC; hull contains EPIC and HM450) |

**Interpretation:** The Mahalanobis reading agrees with the cell-level ranking. Your distance from the healthy hull is **well within the 95th percentile of healthy individuals** (7.5 vs threshold 13.62) — not unusual. The same cells that dominate Section C.2 are the same cells driving the Mahalanobis distance: this is internal consistency, exactly as expected when a single pattern (mild inflammaging) is producing a small coordinated signal.

---

# F. Personal Brilliance Map — your methylation pattern projected against the Cosmic Methylome Background

## F.1 What this is

The **Cosmic Methylome Background (CMB)** is the methylation pattern of the healthy class baseline — the unchanging genetic base layer that every patient is compared against. It is the methylation analog of the Cosmic Microwave Background in cosmology: a reference field against which departures become visible.

Your **Personal Brilliance Map** is the visualization of where YOUR methylation pattern departs from the CMB. Each of the 483,092 atlas CpGs is mapped to a HEALPix pixel (NSIDE=128, 196,608 pixels total) on a Mollweide projection. Where your methylation matches the CMB, the map is quiet. Where it departs, the map brightens.

There are **9 panels** — one per architecture class (8 of them) plus one whole-atlas panel showing all 115 cells together against the CMB background.

## F.2 The reference plates (already in your atlas)

These four plates document the CMB and the visualization framework — they are the same for every patient and establish what "background" looks like:

1. **CPG Plate 01 — Cosmic Methylome Background** *(stable reference)*
   File: `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/CPG_Plate_01_Cosmic_Methylome_Background.png`
   *The healthy class baseline projected onto the HEALPix Mollweide grid.*

2. **CPG Plate 02 — Breast Anisotropy** *(disease-amplified reference)*
   File: `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/CPG_Plate_02_Breast_Anisotropy.png`
   *Demonstrates how a known disease pattern departs from the CMB.*

3. **CPG Plate 03 — CMM vs CMB Comparison** *(framework anchor)*
   File: `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/CPG_Plate_03_Grandaddy_CMM_vs_CMB.png`
   *The cosmology bridge — methylation background and microwave background side-by-side.*

4. **CPG Plate 04 — Patterns Discovered** *(showcase of detection capability)*
   File: `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/CPG_Plate_04_Patterns_Discovered.png`
   *Patterns the framework has successfully discriminated.*

## F.3 Your 8 per-class Personal Brilliance Maps (one per architecture class)

Each per-class panel shows where this class's contributing cells depart from CMB. For a 60yo female whole blood sample, several classes will have empty or quiet panels (no cells present in substrate; nothing to project).

| Panel | Class | Patient signal | Visual character |
|---|---|---|---|
| F.3.1 | Immune | **Active** — 5 high-departure cells visible | Bright clusters in immune-aging-associated chromosomal regions (chr 6 MHC, chr 9 inflammation loci, chr 14 immune receptor) |
| F.3.2 | Secretory | **Active** — 3 mid-elevated cells | Scattered brightness in epithelial-tissue-specific regions |
| F.3.3 | Stromal | Quiet | Faint baseline-level departures |
| F.3.4 | Cycling | Quiet | Faint baseline-level departures |
| F.3.5 | Progenitor | Quiet | Faint baseline-level departures |
| F.3.6 | Stem (adult) | Quiet | Faint baseline-level departures |
| F.3.7 | Stem (pluri) | Empty | No contributing cells in adult blood |
| F.3.8 | Terminal | Quiet | Faint baseline-level departures (residual cross-organ signal) |

## F.4 The whole-atlas Personal Brilliance Map — all 115 cells, against CMB

*This is the heat map that accompanies the Pattern Recognition section (H.5). It is the same data as Sections C.2 (cell ranking) and E (Mahalanobis distance contributions) — but visualized as a spatial pattern across the genome.*

```
                      Schematic of the whole-atlas Personal Brilliance Map
                       (real output: Mollweide projection, 196,608 HEALPix
                        pixels at NSIDE=128 — file: GSM-MOCK-001_personal_brilliance_map.png)

                                       ────── chr 1-12 hemisphere ──────
                              ╭────────────────────────────────────────────╮
                             ╱                                              ╲
                           ╱      ·     · ·                       ·    ●    ╲
                         ╱   ·         ·         ·         ●          ●●     ╲
                       │ ·                ·    ·       · ●●         ●   ●     │
                       │       ·     ·          ·       ●●●          ●        │
                       │  ·        ·     ·   ·        ·   ●●     ·       ·   │     ● = high departure
                       │     ·   ·     ·         · ●●            ·   ·       │           (cells with A > 1.07)
                       │ ·       ·   ·       ●●●●       ·    ·     ·    ·   │
                       │   ·    ·         ●●●●       ·      ·    ·     ·    │     · = baseline-level
                       │      ·       ●●●●●●     ·       ·      ·   ·       │           (cells with A ≈ 1.00)
                       │  ·       ●●●●         ·     ·       ·     ·   ·    │
                         ╲   ·   ●●●        ·        ·    ·      ·     ·    ╱
                           ╲   ●           ·     ·      ·    ·       ·     ╱
                             ╲     ·    ·    ·     ·     ·     ·     ·   ╱
                              ╰────────────────────────────────────────────╯
                                       ────── chr 13-22 + sex hemisphere ──────

                       Brightness clusters concentrated in immune-aging-associated
                       regions (chr 6 MHC, chr 9 inflammation loci, chr 14 immune
                       receptor) — consistent with the inflammaging pattern
                       identified in Section C.2 and Section H.5.
```

**Interpretation:** The brightness clusters in your whole-atlas Personal Brilliance Map align with the cell-level departure ranking in Section C.2 and the Mahalanobis top-10 contributions in Section E. **Three independent visualizations are showing the same pattern from different angles** — a coordinated immune-aging signature concentrated in immune-relevant genomic regions.

---

# G. Bidirectional decomposition — what pooled scoring would have missed

Some methylation patterns move different sites in opposite directions. Pooled scoring (which averages signal across all sites in a panel) cancels these out and reads NULL. Stage 4.5 decomposes the signal directionally to catch what pooling misses.

| Class | Directional composite | Reference threshold | Reading |
|---|---:|---:|---|
| Immune | **+0.18** | VAL-051 alarm at +0.62 (AD-anchored panel) | **Below alarm threshold** — subthreshold directional immune signal consistent with mild inflammaging |
| Secretory | n/a — no sealed directional panel yet | — | NO_PANEL (Stage 4.5 honestly declares; future sealed VAL will populate) |
| Other 6 classes | n/a — no sealed directional panel yet | — | NO_PANEL |

**Interpretation:** Only the immune class currently has a sealed directional panel (the VAL-051 7-CpG Rule A AD-direction-anchored composite). Your immune directional reading of +0.18 is well below the VAL-051 alarm threshold of +0.62 — consistent with inflammaging, not with the AD-pattern that VAL-051 was tuned to detect. The other 7 classes report NO_PANEL honestly until per-class directional VALs are sealed.

---

# H. Disease pattern matching — every disease in the catalog scored against your data

The chain scored your cellular pattern against all 81 phase rows in the disease matrix (covering 52 unique diseases). Detection probability and confidence are computed per disease.

## H.1 Summary

**No disease patterns flagged above clinical threshold.**

This is the expected reading for a 60-year-old wellness patient with no current diagnoses and a normal-for-age inflammaging pattern. The closest pattern matches are presented below for context; none reach clinical threshold.

## H.2 Closest pattern matches (all below threshold)

| Disease | Detection probability | Confidence | Status |
|---|---:|---|---|
| Chronic inflammation (general) | 18% | moderate | Below threshold — consistent with the inflammaging pattern (Section H.5) |
| Breast pre-dx 10yr+ immune window | 11% | low | Below threshold — immune pattern overlap, but no secretory pre-cancer drift in the cells specific to that signature |
| Cardiovascular subclinical risk | 9% | low | Below threshold |
| Pre-T2D (insulin resistance footprint) | 7% | low | Below threshold |
| Early AD-immune directional pattern | 6% | low | Below threshold — directional composite +0.18 vs VAL-051 alarm +0.62 |

**Patterns at 0–5% probability:** 47 other diseases. **Complete per-disease scoring is in Appendix C below.**

## H.3 Trajectory-essential disease notes

Several diseases in the matrix are **trajectory-essential** — single-sample detection is intentionally moderate-to-weak because the signal builds over years. For these, the report does not flag absence of signal as reassurance; the signal is supposed to be weak at sample 1. The clinical strength comes at sample 2+ when drift can be tracked:

- Alzheimer's disease: single-sample AUC = 0.68 (VAL-051) / 0.60 cross-platform (VAL-052); your reading is consistent with no current signature
- Frontotemporal dementia: trajectory-essential
- Parkinson's disease, ALS, MS: trajectory-essential
- Pre-cancer windows (10yr+ pre-diagnostic for breast, pancreatic, HCC): trajectory-essential — these require serial sampling to detect

**Your sample 1 reading does NOT rule these conditions out.** Sample 2 (in 6–12 months) begins to enable trajectory detection.

## H.4 Trajectory-additive disease notes (single-sample-strong)

For diseases with strong single-sample signatures (active solid cancers with field-effect signal), absence of flagging IS clinically meaningful. None of these are flagged:

- Active solid cancers (breast, lung, CRC, gastric, esophageal, pancreatic, HCC, prostate, kidney, bladder, cervical) — no signature at sample 1
- Hematologic malignancies — no signature

## H.5 Pattern Recognition — named patterns across your cells

This is the integrative section. The cell-level departure ranking (C.2), Mahalanobis top-10 contributions (E), per-class Personal Brilliance Maps (F.3), and whole-atlas Personal Brilliance Map (F.4) all visualize the same data from different angles. **Pattern Recognition is where the chain names the patterns those visualizations are showing.**

### H.5.1 Primary pattern detected: Inflammaging signature

**Definition:** Coordinated upward departure across multiple immune-aging-specific cell types (exhausted T cells, senescent memory B cells, M2 macrophages, NK cells), consistent with chronic low-grade immune-system aging.

**Evidence in your data:**

- Cell ranking (C.2): 5 of top 8 cells are immune-aging types (ranks 1, 2, 4, 6, 8)
- Personal Brilliance Map F.3.1 (immune class): active, brightness clusters in chr 6 MHC + chr 9 inflammation loci
- Whole-atlas Brilliance Map F.4: dominant signal in immune-relevant regions
- Mahalanobis top contributions (E): 5 of top 10 are immune cells
- Class-level confirmation (C.3): immune class mean A = 1.054 ELEVATED
- Inflammaging quantum (D.2): +1.8 years of the +2.2 cellular age acceleration

**Strength:** Strong — four independent visualizations converge on the same pattern.

**Clinical interpretation:** Inflammaging is a normal-for-age pattern but quantifies your immune-system aging trajectory. Patients with this pattern at age 60 benefit from monitoring (sample 2+ in 6–12 months) and from interventions known to slow inflammaging (resistance training, anti-inflammatory diet, sleep optimization, stress management).

**Literature anchors:** VAL-051 (AD-immune directional, 7-CpG panel d=+0.62 AIBL holdout); CPG-VAL-015 (immune aging trajectory mechanism); CPG-VAL-017 (inflammaging pooled HC reference); CPG-VAL-020 (Hannum full chain calibration, 19 deliverables).

### H.5.2 Secondary pattern detected: Age-related epithelial drift

**Definition:** Mild upward departure across multiple secretory epithelial cell types (mammary luminal, pancreatic ductal, bronchial epithelial, endometrial glandular) — no single tissue dominating, consistent with age-related epithelial cellular stress.

**Evidence in your data:**

- Cell ranking (C.2): 4 of top 10 cells are secretory epithelial (ranks 3, 5, 7, 10)
- Personal Brilliance Map F.3.2 (secretory class): active, scattered brightness across epithelial-specific regions
- Whole-atlas Brilliance Map F.4: secondary brightness signal distributed across multiple tissue regions
- Mahalanobis top contributions (E): 3 of top 10 are secretory epithelial
- Class-level confirmation (C.3): secretory class mean A = 1.051 ELEVATED

**Strength:** Moderate — pattern visible but driven by 3-4 cells, not the 5+ that the inflammaging pattern shows.

**Clinical interpretation:** Multi-tissue epithelial drift at age 60 is non-specific. It is NOT a pre-cancer signal in any single tissue (the breast-pre-dx 10yr+ pattern requires a specific cell-type-signature combination that is not present here). It is age-related epithelial stress. Sample 2 in 6–12 months will determine whether this drift is stable or accelerating.

**Literature anchors:** VAL-047 (breast pre-dx 10yr+, d=+1.78 — comparison reference for what a real pre-cancer signal looks like); class_anchors.secretory in literature_anchors.json (Volkmar 2012 T2D pancreatic islet, Fleischer 2017 low-grade DCIS).

### H.5.3 Cross-class pattern: Aging-associated tissue stress

**Definition:** The combination of inflammaging + age-related epithelial drift co-occurring in the same patient is itself a recognized cross-class pattern that the chain detects.

**Evidence:** Both H.5.1 and H.5.2 firing simultaneously, with proportional strength matching expected age-related co-occurrence (inflammaging dominant, epithelial drift secondary).

**Clinical interpretation:** This combination is age-typical. It is not a disease pattern. The intervention recommendations from H.5.1 (lifestyle modifications targeting immune-aging) will also address the secondary pattern (epithelial drift), as the underlying mechanism is shared.

### H.5.4 Cross-disease universal alarm — NOT TRIGGERED

The v0_1 universal alarm channel (Section I) was checked. **Not triggered.**

---

# I. Cross-disease universal alarm channel

The chain runs a separate v0_1 alarm channel of **6,018 CpGs** that move in coordinated patterns across multiple diseases. The 12-CpG opposing-direction sub-channel is the unique universal alarm signature (breast pre-dx and AD-at-dx moving in opposite directions at the same loci).

| Sub-channel | Total CpGs | Your firing count | Threshold | Status |
|---|---:|---:|---:|---|
| Background (4,641 CpGs) | 4,641 | 184 | not an alarm channel | informational |
| Breast-only (1,136) | 1,136 | 22 | low | within noise floor |
| AD-only (212) | 212 | 6 | low | within noise floor |
| **Same-direction shared (17)** | 17 | **2** | ≥ 6 alarm | **below alarm** |
| **Opposing-direction universal alarm (12)** | 12 | **0** | ≥ 3 alarm | **NOT TRIGGERED** |

**Interpretation:** The universal alarm channel is silent. No coordinated cross-disease distress signature is present.

---

# J. Wellness, lifestyle, and inflammaging context

| Lens | Reading |
|---|---|
| Inflammaging quantum | +1.8 years (the immune-aging contribution to your +2.2 cellular age delta) |
| Smoking status | Never smoker — smoking-axis foreground subtraction passed cleanly with no residual smoking signature |
| Menarche signature | Not applicable for a 60yo post-menopausal patient |
| Hormonal stratification | Post-menopausal — secretory-class baseline adjusted accordingly |
| Body composition / metabolic signature | Not assessable from blood EPIC alone (requires additional substrate) |
| Stress / sleep / mood | Not directly assessable from methylation alone (clinical history would supplement) |
| Active treatment footprint | None reported and none detected |
| OSK-direction tracking | n/a — single sample. Sample 2+ unlocks (do interventions move you toward OSK/youth direction or away) |
| Senolytic-direction tracking | n/a — single sample |

---

# K. Trajectory monitoring — what serial sampling unlocks

This is your **sample 1 of 1**. The features below unlock as you complete additional samples.

## K.1 What unlocks at sample 2 (first follow-up, 6–12 months from now)

1. **Personal drift rate** — for every one of your 115 cells, the rate of change between sample 1 and sample 2 becomes the personal velocity that will calibrate every subsequent reading
2. **Inflammaging acceleration confirmation** — is the +1.8yr inflammaging quantum stable, accelerating, or reversing?
3. **Epithelial drift trajectory** — is the secondary secretory pattern stable, accelerating, or reversing?
4. **Intervention response signal** — if you make lifestyle changes between sample 1 and sample 2, the chain can detect which cells responded and by how much
5. **Personal noise floor estimate** — knowing your sample-to-sample variability lets us distinguish real change from measurement noise
6. **Mahalanobis trajectory** — your hull-distance becomes a tracked value, not a snapshot
7. **Per-class age delta trajectories** — each class's cellular age becomes a tracked timeseries
8. **Bidirectional immune composite trajectory** — your immune directional score becomes tracked
9. **Pattern stability assessment** — does the inflammaging pattern persist or shift?
10. **First trajectory-essential disease signal** — for the trajectory-essential diseases (AD, FTD, breast pre-dx, etc.), the FIRST drift signal becomes detectable at sample 2 if present

## K.2 What unlocks at samples 3–5 (early longitudinal series)

11. **Forecasting** — extrapolate your trajectory forward to predict cellular age at 65, 70, 75
12. **Drift acceleration/deceleration detection** — second-derivative information (is your aging speeding up or slowing down)
13. **Pattern emergence detection** — patterns that aren't present at sample 1 but emerge by sample 3-5
14. **Cross-pattern interaction** — when two patterns co-occur, how they interact over time
15. **Treatment efficacy quantification** — if you are on any intervention, by sample 3-5 we can quantify its effect size on your cellular trajectory
16. **Confidence interval tightening** — your personal posterior tightens with each sample, making smaller real changes detectable
17. **Substrate-shift detection** — if a future sample is on a different substrate (450K vs EPIC, blood vs saliva), the chain detects and corrects
18. **Early recurrence surveillance** — for patients with prior disease, by sample 3-5 the post-treatment drift signal becomes interpretable

## K.3 What unlocks at samples 6+ — **PATIENT-AS-OWN-BASELINE**

19. **Patient-as-own-baseline mode** — instead of comparing against the n=2,523 healthy hull, the chain compares you to **your own past data**. This is the major unlock. It is more sensitive than the hull comparison because it removes between-person variation entirely.
20. **Drift cascade detection** — patterns that propagate from one cell type to others over time (e.g., immune-aging triggering downstream cycling-class drift)
21. **Personalized intervention recommendations** — based on what has worked for you specifically, not population averages
22. **Aging trajectory phenotype assignment** — classification into one of the recognized longitudinal aging trajectory types
23. **Disease pre-emergence early warning** — for trajectory-essential diseases, the personal-baseline approach gives the earliest possible warning
24. **Reserve-remaining estimate** — for end-of-life context (the GAPE / EDEAR framework primary use case), the rate-of-departure trajectory enables a reserve-remaining estimate
25. **Active surveillance integration** — if you are on active cancer surveillance, your CPG trajectory contributes to the surveillance decision

## K.4 Long-term surveillance (years of data; high-risk patients)

26-35. *(see capability list v0.2 K.4 — full long-term analytics deferred until applicable)*

## K.5 The collective learning network — what every patient contributes back

Every patient who completes their reading contributes (with consent) anonymized cellular pattern data back to the IAMPerformance research base. Your contribution:

36. **Refines the CMB** — each additional 60yo female whole-blood sample tightens the healthy reference
37. **Grows the disease pattern catalog** — patterns observed across thousands of patients become detectable
38. **Tightens the MCMC posteriors** — atlas confidence improves with every sample
39. **Trajectory-essential disease characterization improves over time** — the trajectory-essential diseases get better detection thresholds as more longitudinal data accumulates
40. **Beneficiary + Contributor model** — you receive your reading; you also help every future patient receive a more accurate reading. Both roles, both directions.

---

# L. Prior + family history integration

| Input | Status | Effect on report |
|---|---|---|
| Family cancer history | **Not provided (declined at intake)** | Risk multipliers in `family_history_multiplier.json` not applied |
| Personal cancer history | None reported | No prior-disease adjustments |
| Personal autoimmune history | None reported | No prior-condition adjustments |
| Cancer prior (US lifetime incidence per class) | Loaded from `cancer_prior.json` for reference only | Not applied as a personal risk multiplier (requires family history input) |

**Interpretation:** Without family history, the report cannot personalize cancer risk multipliers. The unflagged disease pattern matching (Section H) reflects the patient's actual cellular data only. If you wish to add family history at a future intake, the trajectory analysis from sample 2 onward can incorporate it retroactively.

---

# M. Literature anchors per finding

For each finding flagged in this report, the relevant published literature anchors and sealed VAL evidence are cited below. Citations are sourced from `literature_anchors.json` v2.1 (cell-level searchable).

| Finding | Published anchor | Sealed VAL evidence |
|---|---|---|
| Exhausted CD8+ T cells, A = 1.054 | Roadmap Epigenomics CD4+ naive T cell A=1.023 (E043) — healthy baseline reference | CPG-VAL-015 (immune aging trajectory); CPG-VAL-017 (inflammaging pooled HC); CPG-VAL-020 (Hannum 19-deliverable) |
| Senescent memory B cells, A = 1.048 | DLBCL A=1.161 (Chapuy 2018) — disease reference for what cancer-level immune B-cell departure looks like (your reading is well below) | CPG-VAL-019 (immune bidirectional confirmation) |
| Mammary luminal epithelial, A = 1.044 | Low-grade DCIS A=1.045 (Fleischer 2017); TCGA BRCA matched-normal A=0.971 | VAL-047 breast pre-dx 10yr+ (d=+1.78 — your reading is well below pre-cancer signal magnitude) |
| Inflammaging pattern (H.5.1) | Pooled HC inflammaging reference (CPG-VAL-017) | VAL-051 (AD-immune directional 7-CpG AUC=0.68); CPG-VAL-015 (immune aging trajectory mechanism) |
| Age-related epithelial drift (H.5.2) | TCGA BRCA matched-normal baseline | VAL-047 §5C+5D (secretory pre-dx homogenization for comparison — your pattern is elevation not homogenization) |
| Cellular age +2.2yr delta (D) | Hannum 2013 GSE40279 aging cohort (HM450 calibration anchor) | CPG-VAL-020 (Hannum full chain calibration) |
| Mahalanobis 7.5 (within hull) | n=2,523 HC pooled reference v0_5 | Hull v0_5 built 2026-06-06; 8 cohorts; 4 populations; cross-platform HM450 + EPIC |
| Immune directional +0.18 (below alarm) | Reference threshold from VAL-051 (+0.62 AIBL alarm) | VAL-050 (pooled NULL d=+0.08); VAL-051 (directional recovery d=+0.62) |

---

# N. The confidence backbone — every number above has it

Every numeric value in this report is computed from the IAMAtlas MCMC posterior (483,092 CpGs × 8 classes × 115 cells × {mean, sd, ci_lo, ci_hi}). The 95% confidence interval is the default presentation.

Available alternate CI levels on request:
- 68% (the 1-sigma view, for clinicians who think in standard deviations)
- 99% (the conservative view, for high-stakes decisions)
- 99.9% (for screening-against-rare-events questions)

Posterior-aware analytics enabled by this backbone (which point estimates would not allow):
- Cellular departure confidence-weighting (used in Section D)
- Bidirectional decomposition variance gating (used in Section G)
- Mahalanobis hull membership probability (used in Section E)
- Per-cell posterior tightness check (used to gate cell ranking inclusion in Section C.2)
- Trajectory significance testing at sample 2+ (used in Section K)
- Bayesian disease detection probability (used in Section H)
- Pattern recognition confidence (used in Section H.5)
- Personal Brilliance Map departure significance (used in Section F)

---

# O. Confidence and caveats (honesty propagation)

| Caveat | Effect |
|---|---|
| This is a screening reading, not a diagnostic test | Any flagged finding requires clinical workup before action |
| Single-sample only | Patient-as-own-baseline (Section K.3) is the most sensitive mode; this report does not access it. Sample 2 begins to enable longitudinal analysis |
| Substrate: whole blood, EPIC array | Most cells in 115-cell atlas are not present in adult blood (e.g., neurons, pancreatic acinar). Their reading is residual cross-organ trace only |
| Cross-platform attenuation | Not relevant here (your sample is EPIC; hull v0_5 contains EPIC and HM450) |
| Ancestry context | US Caucasian patient. Cross-ancestry transferability of hull v0_5: 4 populations represented (EU Italian n=601, UK n=464, US Caucasian-Hispanic n=1,416, Han Chinese n=42 — first Asian population). Reading reliability is highest for ancestries well-represented in the hull |
| Family history defaulted off | Risk multipliers not applied; cancer prior shown for reference only (Section L) |
| Trajectory-essential disease detection | Single-sample is intentionally weak for AD, FTD, Parkinson's, ALS, MS, pre-cancer 10yr+ windows. Sample 2 begins to enable. Absence of signal at sample 1 is NOT a rule-out |
| Bidirectional decomposition | Only 1 of 8 classes (immune, via VAL-051) has a sealed directional panel. The other 7 declare NO_PANEL honestly. Future sealed VALs will expand coverage |
| Cellular age methodology | Confidence-weighted absolute sum of per-cell departures from age-adjusted normal. Stable cells (tight CIs) dominate. The +2.2yr delta is a methylation-derived estimate, not a competing clock against Horvath/Hannum |
| Active disease ruling-out | Cell-level pattern matching against 52 diseases / 81 phase rows checked. No diseases above clinical threshold. This is screening only — clinical examination remains the primary diagnostic modality |

---

# Q. Educational definitions — every novel concept defined for you

Each term is defined inline at first appearance throughout the report. The full glossary is below for reference.

## Q.1 — Foundation (methylation basics)

- **CpG**: A specific position in your DNA where a cytosine sits immediately before a guanine. There are roughly 28 million CpG positions in the human genome; the IAMAtlas measures 483,092 of them.
- **β value (beta value)**: The fraction of cells at a given CpG that have a methyl group attached. Ranges from 0 (no cells methylated) to 1 (all cells methylated). The methylation reading itself.

## Q.2 — Cellular concepts

- **IAMAtlas**: The methylation atlas built by IAMPerformance from 8 publicly available source atlases reconciled together. Contains the reference methylation pattern for 115 individual cell types organized into 8 architectural classes.
- **Architecture class**: A grouping of cell types by their structural and functional role. The 8 classes are: terminal (specialized post-mitotic cells like neurons), secretory (epithelial cells that produce substances), progenitor (cells partway through differentiation), cycling (actively dividing cells), immune (white blood cells and related), stromal (structural support cells), stem (adult) (tissue-specific stem cells), stem (pluri) (pluripotent stem cells — typically only embryonic).
- **Cell type**: A specific cell identity (e.g., "exhausted CD8+ T cells"). The 115-cell map provides finer resolution than the 8-class view.
- **Cell maintaining its identity**: A healthy cell has a methylation pattern characteristic of its type. Departure from that pattern indicates the cell is losing its identity (aging, stress, or disease).

## Q.3 — Architecture / IAM physics

- **A-score**: A continuous number quantifying how well a cell's methylation matches the healthy class baseline. A = 1.00 is healthy baseline. Values above 1.00 indicate departure. Computed as the Shannon entropy of the cell's pooled β values divided by the class's minimum entropy anchor (H_min).
- **H_min**: The minimum entropy value for each architectural class, frozen 2026-04-06 from MCMC: terminal 0.7728, immune 0.838889, secretory 0.8433, progenitor 0.8522, cycling 0.8561, stromal 0.863, stem (adult) 0.8737, stem (pluri) 0.9822.
- **6-tier verdict**: The categorical translation of the continuous A-score. SUPPRESSED (< 0.95), NORMAL (0.95–1.04), ELEVATED (1.04–1.07), WARBURG_TRANSITION (boundary line at 1.07 — not a band), SIGNIFICANTLY_ELEVATED (1.07–1.10), BREACH (≥ 1.10).
- **Warburg line at 1.07**: A metabolic transition threshold derived from the IAM physics framework. Cells crossing 1.07 are exhibiting Warburg-like metabolic shifts (cancer-cell-like glucose metabolism preference even in oxygen-rich conditions).
- **Breach line at 1.10**: The architectural fidelity breach threshold. Cells with A ≥ 1.10 have lost coherent class-baseline structure; the cellular architecture is breaching.
- **Pre-diagnostic active malignancy magnitude (≥ 1.20)**: Values at this magnitude indicate methylation patterns consistent with active or pre-active malignancy. Reference annotation only, not a tier.
- **Bidirectional decomposition (Stage 4.5)**: A method that detects signals where different sites move in opposite directions. Pooled scoring would average these and read NULL. Bidirectional decomposition catches them.

## Q.4 — Cellular aging

- **Cellular age**: An estimate of your biological age based on the cumulative methylation departure from age-adjusted normal across all 115 cells. Computed as the confidence-weighted absolute sum of per-cell departures (the methodology in Section D).
- **Age delta**: The difference between cellular age and chronological age. Positive = aging acceleration; negative = aging deceleration.
- **Inflammaging quantum**: The contribution of immune-aging cells (exhausted T cells, senescent B cells, M2 macrophages, etc.) to the total age delta. Quantifies the immune-system aging component specifically.
- **OSK direction**: Reference to the Yamanaka rejuvenation factors (Oct4, Sox2, Klf4). Patterns that move toward greater stem-cell-like methylation are "OSK direction"; patterns that move away are "anti-OSK direction." Lifestyle interventions that work show patient movement in the OSK direction. Requires sample 2+ to assess direction.
- **Total cellular departure**: Confidence-weighted absolute sum of per-cell departures from age-adjusted normal across all 115 cells. The input to the cellular age estimate.

## Q.5 — Statistical concepts (with the astrophysics connection)

- **Mahalanobis distance**: A measure of how far your 115-dimensional cell-type score vector sits from the centroid of the healthy reference population, in covariance-aware standardized units. Introduced by P.C. Mahalanobis in 1936; widely used in astrophysics to measure how unusual a stellar spectrum is in multidimensional feature space. Here it measures how unusual your cellular pattern is across all 115 cells.
- **Healthy hull**: The reference distribution of Mahalanobis distances from n=2,523 healthy controls across 8 cohorts and 4 populations. Your distance is compared against the 95th percentile (13.62) and 99th percentile (18.59) of this reference distribution.
- **Posterior**: In Bayesian statistics, the full probability distribution of a parameter given the data — not just a single point estimate but the entire distribution including uncertainty.
- **95% confidence interval (CI)**: The range within which the true value is estimated to lie with 95% probability, given the data. Always presented with every numeric reading in this report.

## Q.6 — Visualization (CMB and Personal Brilliance Map)

- **Cosmic Methylome Background (CMB)**: The methylation pattern of the healthy class baseline projected across the HEALPix Mollweide grid. The unchanging genetic base layer that every patient is compared against. Named in deliberate analogy to the Cosmic Microwave Background of cosmology — a reference field against which departures become visible.
- **Personal Brilliance Map**: Your individual methylation pattern projected onto the same HEALPix Mollweide grid as the CMB, with departures from the CMB rendered as brightness. Where your methylation matches the CMB, the map is quiet. Where it departs, the map brightens.
- **Mollweide projection**: An equal-area map projection that displays a spherical surface (like a globe or a CMB sky map) as a 2D ellipse. Standard projection for cosmology sky maps and inherited here for methylation atlas visualization.
- **HEALPix (Hierarchical Equal Area isoLatitude Pixelization)**: A pixelization scheme for spherical surfaces that ensures every pixel covers an equal area. NSIDE=128 gives 196,608 pixels — the resolution of your Personal Brilliance Map. Borrowed directly from cosmology CMB analysis pipelines.

## Q.7 — Disease detection

- **Field effect**: The methylation pattern that pre-cancer or peri-cancer tissue exhibits even at sites distant from the actual tumor — the "field" of cellular change surrounding a malignancy. The chain detects field effects at very early stages.
- **Pre-diagnostic window**: The years between when the methylation signature is detectable and when clinical diagnosis is made. Breast pre-dx 10yr+ window means the immune signature can be detected 10 or more years before breast cancer becomes clinically diagnosable. Pancreatic 2-5yr, HCC 8.03× separation vs cirrhosis.
- **Trajectory-essential vs trajectory-additive disease**: Trajectory-essential = single-sample signal is intentionally weak; the diagnostic strength requires serial sampling (AD, FTD, Parkinson's, MS, pre-cancer windows). Trajectory-additive = single-sample signal is strong; serial sampling adds value but is not required (active solid cancers with field-effect signatures).
- **Universal alarm channel**: The v0_1 6,018-CpG channel that detects coordinated cross-disease distress patterns. The 12-CpG opposing-direction sub-channel is the unique cross-disease alarm signature.

## Q.8 — Trajectory (serial sampling)

- **Patient-as-own-baseline**: At sample 6+, comparison shifts from "vs. n=2,523 healthy reference" to "vs. your own past data." More sensitive because it removes between-person variation.
- **Drift cascade**: When a pattern in one cell type propagates over time to other cell types (e.g., immune-aging triggering downstream cycling-class drift).
- **Forecasting**: Extrapolation of your trajectory forward in time to predict cellular age, pattern emergence, or drift acceleration at future timepoints.
- **Network learning**: The cumulative improvement in atlas accuracy as more patients contribute their (anonymized, consented) cellular patterns to the research base. Beneficiary + Contributor model.

## Q.9 — Wellness, lifestyle

- **Inflammaging**: Chronic low-grade systemic inflammation associated with aging. Quantified here as the immune-class contribution to total cellular age delta.
- **Senolytic intervention**: Pharmaceutical or lifestyle interventions that selectively eliminate senescent cells. The CPG framework can track senolytic-direction movement at sample 2+.

## Q.10 — Generation infrastructure

- This report is generated by **`walther_clinical.py`** — the orchestrator that runs the full CPG chain on a patient IDAT pair and assembles the report. All embedded visualizations are rendered by the same Python script. Two reading modes available: **clinical default** (this view) and **patient view** (simplified language; same data).

---

# Appendix A — Embedded visualizations

## A1. A-score reference gauge (the calibration scale)

[See Section C.1 for the inline reference; the full color-coded gauge with Warburg line at 1.07, breach line at 1.10, and pre-diagnostic active malignancy annotation at 1.20 is generated as `GSM-MOCK-001_reference_gauge.svg` by `walther_clinical.py`.]

## A2. Cellular departure ranking — top 15 of 115 cells

[See Section C.2 for the data table; the horizontal bar chart visualization with tier zones in the background is generated as `GSM-MOCK-001_cellular_departure_ranking.svg` by `walther_clinical.py`. The bar chart shows the same data as the table, with each cell positioned by A-score against the reference gauge zones.]

## A3. Personal Brilliance Map — 8 per-class panels + 1 whole-atlas panel

[Output files generated by `walther_clinical.py`:]
- `GSM-MOCK-001_personal_brilliance_map_terminal.png`
- `GSM-MOCK-001_personal_brilliance_map_secretory.png`
- `GSM-MOCK-001_personal_brilliance_map_progenitor.png`
- `GSM-MOCK-001_personal_brilliance_map_cycling.png`
- `GSM-MOCK-001_personal_brilliance_map_immune.png`
- `GSM-MOCK-001_personal_brilliance_map_stromal.png`
- `GSM-MOCK-001_personal_brilliance_map_stem_adult.png`
- `GSM-MOCK-001_personal_brilliance_map_stem_pluri.png` *(empty for adult blood)*
- `GSM-MOCK-001_personal_brilliance_map_whole_atlas.png` *(the heat map accompanying Section H.5)*

## A4. Reference plates (already in atlas — same for every patient)

- `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/CPG_Plate_01_Cosmic_Methylome_Background.png`
- `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/CPG_Plate_02_Breast_Anisotropy.png`
- `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/CPG_Plate_03_Grandaddy_CMM_vs_CMB.png`
- `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/CPG_Plate_04_Patterns_Discovered.png`

---

# Appendix B — Audit trail

| Stage | Input | Output | Hash |
|---|---|---|---|
| 0 Intake | IDAT pair (red + green) | β matrix raw | sha256:... |
| 1 β computation | β matrix raw | β matrix normalized (sesame noob + dye-bias) | sha256:... |
| 2 Deconvolution | β matrix normalized | 115 cell-type fractions + per-cell β | sha256:... |
| 3 Foreground subtraction | β matrices per cell | β residual (age + sex + smoking axes subtracted) | sha256:... |
| 4 A-score | β residual | per-class + per-cell A-scores | sha256:... |
| 4.5 Bidirectional | per-cell β residual | directional composite per class | sha256:... |
| 4.6 Brightness Comparison | per-cell β residual | Personal Brilliance Map (9 panels) | sha256:... |
| 5 Mahalanobis | per-cell A vector | distance + top-10 contributions | sha256:... |
| 6 Cellular Age | per-cell A + age-adjusted normal | total cellular departure + cellular age | sha256:... |
| 7 Tier Mapping | A-scores | per-cell tier verdicts | sha256:... |
| 8A Per-card matching | per-cell pattern | 3 chain-language card scores | sha256:... |
| 8B Disease matrix | per-cell pattern | 81 phase row scores | sha256:... |
| 9 Report Assembly | all above + literature_anchors.json + cancer_prior.json | this report (md + json + png) | sha256:... |
| 10 Delivery | this report | delivered to patient + filed in `outputs/` | sha256:... |

**Chain-of-custody:** Every value in this report is reproducible from the input IDAT files using `walther_clinical.py` v1.0 with the runtime artifacts hashed above.

---

---

# Appendix C — Complete disease scoring (all 52 diseases in the matrix)

Every disease in the disease matrix v1.8 was scored against this patient's cell-level pattern. The threshold for clinical flagging is detection probability ≥ 20%. **No disease in the catalog scored above 20% for this patient.** The complete scoring is below for transparency.

| # | Disease | Category | Detection probability | Confidence | Status |
|--:|---|---|---:|---|---|
| 1 | Chronic inflammation (general) | Inflammatory | 18% | moderate | Below threshold |
| 2 | Breast pre-dx 10yr+ immune window | Pre-cancer trajectory | 11% | low | Below threshold |
| 3 | Cardiovascular subclinical risk | Cardiovascular | 9% | low | Below threshold |
| 4 | Pre-T2D (insulin resistance footprint) | Metabolic | 7% | low | Below threshold |
| 5 | Early AD-immune directional pattern | Neurodegenerative trajectory | 6% | low | Below threshold |
| 6 | Pulmonary arterial hypertension | Cardiovascular | 5% | low | Below threshold |
| 7 | Mild cognitive impairment | Neurodegenerative trajectory | 4% | low | Below threshold |
| 8 | Hashimoto's thyroiditis | Autoimmune | 4% | low | Below threshold |
| 9 | Aortic dilation / BAV signature | Cardiovascular | 3% | low | Below threshold |
| 10 | Frontotemporal dementia | Neurodegenerative trajectory | 3% | low | Below threshold |
| 11 | Parkinson's disease | Neurodegenerative trajectory | 3% | low | Below threshold |
| 12 | Pancreatic cancer pre-dx 2–5yr | Pre-cancer trajectory | 3% | low | Below threshold |
| 13 | HCC vs cirrhosis differential | Pre-cancer trajectory | 3% | low | Below threshold |
| 14 | Multiple sclerosis | Neurodegenerative trajectory | 3% | low | Below threshold |
| 15 | ALS (amyotrophic lateral sclerosis) | Neurodegenerative trajectory | 2% | low | Below threshold |
| 16 | Rheumatoid arthritis | Autoimmune | 2% | low | Below threshold |
| 17 | Systemic lupus erythematosus | Autoimmune | 2% | low | Below threshold |
| 18 | PSP / CBD (tauopathies) | Neurodegenerative trajectory | 2% | low | Below threshold |
| 19 | Atherosclerosis | Cardiovascular | 2% | low | Below threshold |
| 20 | Crohn's disease (Stage 1 immune signature) | Inflammatory | 2% | low | Below threshold |
| 21 | Ulcerative colitis (Stage 1 immune signature) | Inflammatory | 2% | low | Below threshold |
| 22 | Active breast cancer | Active malignancy | <1% | low | Below threshold |
| 23 | Active lung cancer (LUAD) | Active malignancy | <1% | low | Below threshold |
| 24 | Active lung cancer (LUSC) | Active malignancy | <1% | low | Below threshold |
| 25 | Active colorectal cancer | Active malignancy | <1% | low | Below threshold |
| 26 | Active pancreatic cancer | Active malignancy | <1% | low | Below threshold |
| 27 | Active hepatocellular carcinoma | Active malignancy | <1% | low | Below threshold |
| 28 | Active gastric cancer | Active malignancy | <1% | low | Below threshold |
| 29 | Active esophageal cancer (ESCC) | Active malignancy | <1% | low | Below threshold |
| 30 | Active esophageal cancer (EAC) | Active malignancy | <1% | low | Below threshold |
| 31 | Active prostate cancer | Active malignancy | <1% | low | Below threshold |
| 32 | Active kidney cancer (KIRC) | Active malignancy | <1% | low | Below threshold |
| 33 | Active kidney cancer (KIRP) | Active malignancy | <1% | low | Below threshold |
| 34 | Active bladder cancer | Active malignancy | <1% | low | Below threshold |
| 35 | Active cervical cancer (invasive) | Active malignancy | <1% | low | Below threshold |
| 36 | Cervical CIN2/CIN3 (pre-invasive) | Pre-cancer | <1% | low | Below threshold |
| 37 | Active glioblastoma (GBM) | Active malignancy | <1% | low | Below threshold |
| 38 | Lower grade glioma (LGG) | Active malignancy | <1% | low | Below threshold |
| 39 | Active melanoma (SKCM) | Active malignancy | <1% | low | Below threshold |
| 40 | Acute myeloid leukemia (AML) | Hematologic malignancy | <1% | low | Below threshold |
| 41 | Acute lymphoblastic leukemia (B-ALL) | Hematologic malignancy | <1% | low | Below threshold |
| 42 | Acute lymphoblastic leukemia (T-ALL) | Hematologic malignancy | <1% | low | Below threshold |
| 43 | Chronic myeloid leukemia (CML) | Hematologic malignancy | <1% | low | Below threshold |
| 44 | Chronic lymphocytic leukemia (CLL) | Hematologic malignancy | <1% | low | Below threshold |
| 45 | Diffuse large B-cell lymphoma (DLBCL) | Hematologic malignancy | <1% | low | Below threshold |
| 46 | Multiple myeloma | Hematologic malignancy | <1% | low | Below threshold |
| 47 | Myelodysplastic syndrome (MDS) | Hematologic malignancy | <1% | low | Below threshold |
| 48 | Sarcoma (SARC) | Active malignancy | <1% | low | Below threshold |
| 49 | Mesothelioma (MESO) | Active malignancy | <1% | low | Below threshold |
| 50 | Testicular germ cell tumor (TGCT — inversion pattern) | Active malignancy | <1% | low | Below threshold |
| 51 | Ischemic stroke (3-subtype undifferentiated) | Cardiovascular | <1% | low | Below threshold |
| 52 | Thymoma | Hematologic malignancy | <1% | low | Below threshold |

**Summary:** 52 of 52 diseases below clinical flagging threshold. Top 5 closest matches (ranks 1–5) are all consistent with the inflammaging + epithelial-drift pattern identified in Sections C and H.5 — no disease-specific signature dominates over the age-related pattern.

# Appendix D — Complete per-cell A-score table (all 34 detected cells)

| Cell type | Class | A-score | Tier | 95% CI | Composition (Section B) |
|---|:---:|---:|---|---|---:|
| Exhausted CD8+ T cells | IMM | 1.054 | ELEVATED | [1.042 — 1.066] | 3.2% ⚠ |
| Senescent memory B cells | IMM | 1.048 | ELEVATED | [1.037 — 1.059] | 0.9% ⚠ |
| Mammary luminal epithelial | SEC | 1.044 | ELEVATED | [1.032 — 1.056] | 1.4% |
| Effector memory CD4+ T cells | IMM | 1.041 | NORMAL | [1.030 — 1.052] | 5.8% |
| Pancreatic ductal | SEC | 1.038 | NORMAL | [1.026 — 1.050] | 1.2% |
| Macrophages M2 | IMM | 1.036 | NORMAL | [1.026 — 1.046] | 2.1% |
| Bronchial epithelial | SEC | 1.034 | NORMAL | [1.022 — 1.046] | 1.0% |
| NK CD56dim | IMM | 1.032 | NORMAL | [1.022 — 1.042] | 5.4% |
| Plasmacytoid dendritic | IMM | 1.030 | NORMAL | [1.021 — 1.039] | 0.2% |
| Adipose tissue stromal | STR | 1.026 | NORMAL | [1.016 — 1.036] | 1.4% |
| Vascular endothelial | STR | 1.023 | NORMAL | [1.013 — 1.033] | 2.1% |
| CD4+ memory T cells | IMM | 1.022 | NORMAL | [1.011 — 1.033] | 8.4% |
| CD8+ memory T cells | IMM | 1.020 | NORMAL | [1.010 — 1.030] | 5.1% |
| BM myeloid progenitor | PRO | 1.019 | NORMAL | [1.009 — 1.029] | 0.5% |
| Colon crypt base columnar (residual) | CYC | 1.018 | NORMAL | [1.005 — 1.031] | 0.7% |
| BM proliferating progenitor | CYC | 1.018 | NORMAL | [1.008 — 1.028] | 0.9% |
| Classical monocytes | IMM | 1.018 | NORMAL | [1.009 — 1.027] | 4.6% |
| Common lymphoid progenitor | PRO | 1.016 | NORMAL | [1.006 — 1.026] | 0.4% |
| Bone marrow stromal | STR | 1.015 | NORMAL | [1.005 — 1.025] | 0.5% |
| B cells memory | IMM | 1.015 | NORMAL | [1.005 — 1.025] | 0.7% |
| Intermediate monocytes | IMM | 1.014 | NORMAL | [1.003 — 1.025] | 0.5% |
| B cells naive | IMM | 1.012 | NORMAL | [1.002 — 1.022] | 1.4% |
| Hair follicle (residual) | CYC | 1.012 | NORMAL | [1.000 — 1.024] | 0.5% |
| Fibroblasts (residual) | STR | 1.010 | NORMAL | [1.000 — 1.020] | 0.2% |
| Non-classical monocytes | IMM | 1.010 | NORMAL | [1.000 — 1.020] | 0.2% |
| Eosinophils | IMM | 1.008 | NORMAL | [0.997 — 1.019] | 1.2% |
| Naive CD4+ T cells | IMM | 1.006 | NORMAL | [0.995 — 1.017] | 2.8% |
| Cortical neuron (residual) | TER | 1.005 | NORMAL | [0.991 — 1.019] | 0.2% |
| Naive CD8+ T cells | IMM | 1.004 | NORMAL | [0.993 — 1.015] | 1.6% |
| Hepatocyte (residual) | TER | 1.002 | NORMAL | [0.988 — 1.016] | 0.2% |
| Regulatory T cells (Treg) | IMM | 1.002 | NORMAL | [0.991 — 1.013] | 0.1% |
| Neutrophils (mature) | IMM | 0.998 | NORMAL | [0.989 — 1.007] | 56.2% |
| HSC (hematopoietic stem) | SA | 0.992 | NORMAL | [0.982 — 1.002] | 0.9% |
| Mesenchymal stem (residual) | SA | 0.988 | NORMAL | [0.978 — 0.998] | 0.5% |

**81 cell types in the IAMAtlas catalog were not detected in this sample** (e.g. most neurons, pancreatic islets, mammary basal, all pluripotent stem types) — appropriate for whole-blood substrate.

---

*This is a MOCK report for design review of the CPG Doctor-Report Capability v0.2 layout. Patient data is fabricated for design purposes. Real first patient will be a blinded sample from a public GEO methylation study (mixed clinical cohort per Heath's selection 2026-06-08).*
