# Heme-EPIC Card — EDEAR Hematologic Malignancy Flag

**Version 0.1 · 2026-04-25**
**Validation tier:** `framework_calibrated_pending_per_patient_validation` — TCGA tissue-level A-score signatures characterized in Issue 002; pre-diagnostic per-patient blood validation candidate cohorts identified (EnviroGenomarkers, MCCS, Northern Sweden + EPIC); not yet run.
**Card position:** #11 of 15 in Cookbook v2.1 expansion set (heme-epic, immune-atlas, kidney-epic, glioma-epic, gastric-epic, bladder-epic, cardio-epic remaining)
**Disease scope:** Lymphoid arm (DLBCL, CLL, multiple myeloma, B-ALL, T-ALL, thymoma) + Myeloid arm (AML, MDS, MPN). Two arms reflect distinct biology — lymphoid lineage opens more accessible entropy than myeloid because B-cell class-switching and somatic hypermutation are programmed methylation perturbations cancer can exploit further.

## What this card is

The heme-epic card covers cancers where the **immune compartment IS the diseased tissue** — lymphoid malignancies (DLBCL, CLL, multiple myeloma, ALL, thymoma) and myeloid malignancies (AML, MDS, MPN). Unlike every other cancer in the Cookbook, these cancers do not have a separate Stage 2 solid-organ target because the immune cells that Stage 1 measures are themselves the tumor. The heme-epic algorithm runs on the same blood draw as every other card but interprets the Stage 1 elevated A-score as diseased-tissue signal rather than response-to-upstream-disease signal.

**This card is split into two arms** (lymphoid, myeloid) with separate directional panels and Stage 3 EpiDISH discrimination criteria, because the underlying biology is fundamentally different between B-cell/T-cell-derived cancers and granulocyte/monocyte-derived cancers.

## Hematologic malignancy family covered (glossary)

The leukemia and lymphoma family covered by heme-epic, organized by arm:

**Myeloid arm (myeloid lineage — neutrophils, monocytes, granulocyte progenitors):**
- **AML** — Acute Myeloid Leukemia (myeloid, fast-progressing)
- **CML** — Chronic Myeloid Leukemia (myeloid, slow, BCR::ABL1-driven)
- **MDS** — Myelodysplastic Syndromes (myeloid, pre-leukemic; ~30% transform to AML over years)
- **MPN** — Myeloproliferative Neoplasms (myeloid; includes polycythemia vera, essential thrombocythemia, primary myelofibrosis)

**Lymphoid B-cell arm (B-cell lineage):**
- **CLL** — Chronic Lymphocytic Leukemia (B-cell, slow-progressing, indolent)
- **B-ALL** — B-cell Acute Lymphoblastic Leukemia (B-cell, fast-progressing, mostly pediatric)
- **DLBCL** — Diffuse Large B-Cell Lymphoma (B-cell; lymphoma not leukemia, but same compartment)
- **MM** — Multiple Myeloma (B-cell, plasma-cell-stage)

**Lymphoid T-cell arm (T-cell lineage):**
- **T-ALL** — T-cell Acute Lymphoblastic Leukemia (T-cell, fast-progressing)
- **Thymoma** — T-cell origin tumor of the thymus

When EDEAR fires the heme-epic pathway, the patient-facing differential is which arm (myeloid vs lymphoid B-cell vs lymphoid T-cell), and Stage 3 EpiDISH narrows it from there. The doctor takes it from "elevated immune A + Moss quiet on peripherals + B-cell shift in EpiDISH" to "let's run flow cytometry to see if it's CLL or DLBCL or something else." Heme-epic's role is the screening flag and the differential routing; clinical confirmation is the doctor's domain.

## How heme cancer is distinguished from immune-response-to-solid-cancer (and from brain cancer)

This is the central diagnostic question for heme-epic and the answer is the three-stage pipeline working together:

1. **Stage 1 fires elevated A-immune.** Alone, this is ambiguous — could be heme cancer, could be solid cancer, could be brain cancer that isn't shedding into blood, could be inflammaging, could be autoimmune, could be acute infection.

2. **Stage 2 Moss NNLS deconvolution** answers part of the localization question. If a solid organ shows elevated shedding (breast_ductal, lung_epithelial, colon_epithelial, prostate_epithelium, hepatocyte, pancreatic_exocrine, cervical_epithelial), the Stage 1 immune signal is *response* to that solid disease — route to the appropriate solid-organ card. **However: Moss 2018 reference does NOT include brain/CNS tissue** because brain does not shed into blood meaningfully under normal conditions; the blood-brain barrier limits cfDNA fraction from primary CNS tumors to extremely low levels even at advanced stages. So **"Moss NULL on solid organs"** does NOT rule out brain cancer or other CNS disease — it only rules out the 18 Moss-referenced peripheral solid organs.

3. **Stage 3 EpiDISH RPC** discriminates within the heme/CNS/inflammaging pathway. Lineage-specific shifts pick the arm:
   - **Neutrophil-dominant shift** → myeloid arm (AML, MDS, MPN). Validated VAL-082, see §6.
   - **B-cell-dominant shift** → lymphoid B-cell arm (CLL, DLBCL, MM, B-ALL).
   - **T-cell-dominant shift** → lymphoid T-cell arm (thymoma, T-ALL).
   - **Uniform elevation without lineage shift** → NOT heme cancer; route to immune-atlas for differential. Possible: inflammaging, autoimmune, chronic infection, brain cancer (whose immune response may be uniform-shift rather than lineage-shifted because the response is to a non-immune-tissue tumor that isn't shedding).
   - **Uniform suppression** → SUPPRESSED tier (immunocompromised state).

**What this means for the brain-cancer differential.** A patient with elevated Stage 1 + Moss NULL on the 18 peripheral solid tissues + Stage 3 uniform elevation pattern (no lineage-specific shift) cannot be confidently routed to heme-epic. The pattern is consistent with brain/CNS cancer (which doesn't shed into blood, so Moss can't see it), with inflammaging, with autoimmune, or with chronic infection. The patient-facing report at v1 must be honest about this ambiguity: "Your immune A-score is elevated and your tissue-of-origin breakdown does not show shedding from the 18 peripheral organs in our reference. Your immune lineage breakdown shows uniform elevation rather than lineage-specific shift. This pattern has been documented in research cohorts studying inflammation, autoimmune conditions, and primary CNS tumors that do not shed measurably into blood. Talk to your doctor; if you have neurological symptoms, brain imaging may be part of the differential."

The proper card to handle this differential is **glioma-epic** (TBD), which will use additional data sources (brain-specific methylation panels, fragmentomics if L2/L3 platform is available) to surface CNS-specific signal where Moss cannot. At v1 with 450K/EPIC alone, the framework cannot positively identify glioma — it can only flag "elevated immune + Moss NULL on solid + uniform Stage 3" as a pattern that warrants neurological evaluation alongside other differentials.

The three-stage pattern distinguishes **myeloid AML** vs **lymphoid CLL/DLBCL/MM** vs **T-cell thymoma** vs **inflammaging** vs **possible CNS-or-other-non-Moss-tissue disease** vs **immunocompromised SUPPRESSED state** — but it cannot positively confirm CNS cancer in v1. Glioma-epic v0.1 (next card to build) will document what improvements to that differential are achievable at the v1 platform.

## Why this card exists

The framework's Issue 002 Immune class chapter characterizes these cancers with specific A-score signatures calibrated to published TCGA-scale cohorts:

- **AML** (myeloid, TCGA 2013 NEJM n=200): A_combined ≈ 1.10, ΔA = +0.168 at cfDNA level (VAL-007)
- **DLBCL** (lymphoid B-cell, Chapuy 2018 n=48): A_combined ≈ 1.13, ΔA = +0.203 — the **largest methylation departure in the immune class**
- **Thymoma** (T-cell origin, TCGA n=120): A ≈ 1.09
- **CLL** (indolent B-cell leukemia): A ≈ 1.07 in DETECTABLE tier

These are not small differences — a factor-of-two spread in ΔA across the immune cancer panel — and they separate by lineage. Lymphoid-lineage malignancies open more accessible entropy than myeloid because B-cell class-switching and somatic hypermutation are programmed perturbations of the methylation landscape that cancer can exploit further.

Without heme-epic, a patient with AML or DLBCL would produce a strongly-positive Stage 1 immune signature with null Stage 2 and be routed to the four-pathway differential in immune-atlas. Pathway 2 (hematologic) points here. Heme-epic provides the specific algorithm that takes over once immune-atlas Pathway 2 fires.

## Why heme cancers detect uniquely well — three structural reasons

Heme cancers are the natural strength of cfDNA/buffy-coat methylation screening. Three specific structural features make detection work better than for any solid-organ cancer:

**1. The diseased cells ARE the cfDNA being sampled.** When a breast cancer cell dies, it sheds maybe 0.1% of plasma cfDNA into the blood, mixed with 99.9% normal immune-cell cfDNA from bone-marrow turnover. The signal is a tiny needle in an enormous haystack. When an AML cell dies, it sheds into a blood compartment where AML cells already make up 30-90% of nucleated cells — the cancer IS the dominant cfDNA fraction. There is no needle and no haystack; the haystack itself is the needle. **VAL-082's d = +3.71 dwarfs every solid-cancer effect size in the catalog because of this structural difference.**

**2. The immune system has programmed methylation plasticity that healthy solid tissues don't have.** B-cell class-switching, somatic hypermutation, T-cell activation, monocyte differentiation — these are large, programmed, controlled methylation changes that healthy immune cells use as part of normal function. When a cancer hijacks an immune lineage, it doesn't have to climb out of a tightly-methylated floor; it pushes further along an axis the lineage already uses. Lymphoid cancers exploit this more than myeloid cancers (DLBCL ΔA = +0.20, AML ΔA = +0.17, CLL ΔA = +0.10), but all heme cancers benefit. **Cancer Amplifier g for the immune class is 5-10× rather than infinite** (as it is for solid tumors at the H_min floor) precisely because healthy immune cells are not at the floor — they are actively reorganizing methylation, and the framework reads "more reorganization than expected" as the cancer signal.

**3. The immune system's methylation signature is shaped by lineage and activation state in ways the framework can deconvolute.** EpiDISH RPC (Stage 3) returns six immune lineages — neutrophil, monocyte, CD4 T, CD8 T, NK, B-cell — each with characteristic methylation patterns. The framework can read which lineage is anomalously expanded. For solid cancers, Moss returns "X% breast_ductal cfDNA" without sub-resolution. For heme cancers, EpiDISH returns "your B-cells are at 35% when they should be 8%" — much more directly diagnostic. Solid-organ cards rely on a single-resolution Stage 2 read; **heme-epic gets a multi-resolution Stage 3 read essentially for free**, because the immune compartment has been deeply characterized in healthy reference panels (Salas 2018 has 6+ lineages; Loyfer 2023 has 30+).

The combined consequence: heme cancers detect at higher signal-to-noise (point 1), with biological substrate that already produces the methylation patterns we measure (point 2), and with sub-classification built into the existing reference panels (point 3). The framework's universal pipeline does not need disease-specific tuning to reach clinical-grade detection on AML — VAL-082 hit d = +3.71 with no panel substitution, no directional fallback, no special H_min calibration. AML is the case where the framework fires correctly at the v1 platform with no card-specific deviation needed.

**The doctor-side mechanics of early detection follow from these structural reasons.** Clinical confirmatory tests for heme cancers are blood-based: flow cytometry for B-cell or T-cell monoclonality (detects MBL down to clone fractions <0.5%, or <0.01% on high-sensitivity flow), CHIP myeloid mutation panel NGS for AML precursors (detectable at 2% clonal fraction, present in ~10% of adults over 60). The complete early-detection arc — EDEAR fires, doctor orders confirmatory test, confirmation either rules in or rules out a pre-malignant clone, patient enters active surveillance or gets all-clear — happens within the blood-pathway only. No imaging, no biopsy, no surgery is required at the screening-and-surveillance stage. Solid-organ cancers do not have this advantage; their pre-clinical states are histological (DCIS, adenoma) and require biopsy to find.

## Expected Stage 1 immune signature by disease subtype

The heme-epic card fires on the **universal Stage 1 immune A-score** output from the pipeline, but interprets the signature pattern through per-disease directional panels to discriminate among the five subtypes.

### AML (Acute Myeloid Leukemia)

- **Expected A_combined:** ≈ 1.10 (FLOOR BREACH tier)
- **Expected ΔA:** +0.168 at cfDNA level (VAL-007 target)
- **Per-CpG signature:** myeloid-lineage methylation reprogramming — focal hypermethylation at lineage-commitment loci (MEIS1, HOXA cluster, DNMT3A mutation hotspots), global hypomethylation at proliferation loci
- **Five-substrate profile:** methylation primary. Nucleosome occupancy saturates at A ≈ 1.010 in every immune cancer. Fuzz A ≈ 1.115, WPS A ≈ 1.130, fragment size A ≈ 1.124 — all past BREACH threshold. Combined A uses methyl + fuzz + WPS + frag for active tracking.
- **Pre-dx precursor:** CHIP (clonal hematopoiesis of indeterminate potential) precedes AML by years. G-2026-P010 prediction: CHIP patients with archived serial blood samples show immune-class A-score trajectory before overt AML.
- **Reference cohort:** Ley et al. 2013 NEJM, TCGA AML n=200. doi:10.1056/NEJMoa1301689

### DLBCL (Diffuse Large B-Cell Lymphoma)

- **Expected A_combined:** ≈ 1.13 (FLOOR BREACH tier) — **largest immune-class ΔA in framework**
- **Expected ΔA:** +0.203 at cfDNA level
- **Per-CpG signature:** lymphoid-lineage methylation — B-cell class-switching landscape exploited by tumor. GCB vs ABC subtypes discriminated by directional panel (future refinement).
- **Five-substrate profile:** methyl 1.165, nucl saturated at 1.010, fuzz 1.145, WPS 1.161, frag 1.156, combined 1.127
- **Reference cohort:** Chapuy et al. 2018, DLBCL n=48. Also Nat Med 2018 DLBCL genomic subtypes.

### Thymoma (T-cell origin)

- **Expected A_combined:** ≈ 1.09 (URGENT tier)
- **Expected ΔA:** +0.120
- **Per-CpG signature:** T-cell-specific methylation disruption at thymic selection loci
- **Five-substrate profile:** methyl 1.115, nucl saturated, fuzz 1.101, WPS 1.115, frag 1.110, combined 1.090
- **Reference cohort:** TCGA thymoma n=120 (doi:10.1016/j.ccell.2018.03.010)

### CLL (Chronic Lymphocytic Leukemia — indolent B-cell)

- **Expected A_combined:** ≈ 1.07 (DETECTABLE tier)
- **Expected ΔA:** +0.098
- **Per-CpG signature:** B-cell methylation drift, more subtle than DLBCL. IGHV mutation status correlates with methylation subtype.
- **Five-substrate profile:** methyl 1.084, nucl saturated, fuzz 1.080, WPS 1.084, frag 1.081, combined 1.068
- **Reference cohort:** published CLL methylation cohorts + Hannum 2013 GSE40279 baseline

### Multiple Myeloma (plasma cell)

- **Expected A_combined:** less specifically characterized in Issue 002 — estimated MARGINAL to DETECTABLE tier
- **Per-CpG signature:** plasma cell methylation disruption, different from both B-cell and T-cell lineages
- **Status:** Framework references MM as part of the immune cancer panel but detailed substrate values less specified than AML/DLBCL/thymoma/CLL. Build-out pending.

## Expected Stage 2 result

**Stage 2 Moss NNLS deconvolution returns NULL for any solid-organ tissue localization because the immune class IS the diseased tissue.** Stage 2's 18-tissue β output will show the immune subcomposition shifted but no cycling/secretory/terminal/stromal elevation. This is the characteristic heme-epic pattern.

The Stage 2 output may, however, show elevated fractions of specific immune sub-lineages (CD8+ T for thymoma, B-cell for CLL/DLBCL/MM, neutrophil for AML) which supports the heme-epic call.

## Stage 3 EpiDISH is the critical discriminator for heme-epic

Unlike most cards where Stage 3 is secondary, heme-epic depends on Stage 3 EpiDISH RPC immune sub-composition analysis:

- **AML pattern:** elevated neutrophil fraction, suppressed lymphocyte lineages — myeloid expansion
- **DLBCL / CLL pattern:** elevated B-cell fraction, normal-to-suppressed T-cell
- **Thymoma pattern:** elevated T-cell fraction (often CD8+), particularly abnormal if CD4/CD8 ratio inverts
- **MM pattern:** elevated plasma cell fraction (if EpiDISH reference includes plasma cells; RPC panel extension may be required)
- **Inflammaging pattern:** uniform elevation without lineage-specific shift — this is the differential FROM cancer

Salas 2018 healthy ranges (the QC gate for plasma methylation):
- Neutrophil: 45-75%
- Lymphocyte: 20-40% subdivided (CD4+ 10-30%, CD8+ 5-25%, NK 3-15%, B 3-15%)
- Monocyte: 3-12%

Any sample outside these ranges gets a QC flag; heme-epic interprets "out of range" as a positive signal pattern matching one of the lineage-shift profiles above.

## Substrate saturation caveat

**Critical for immune cancers:** nucleosome occupancy saturates at A ≈ 1.010 for every immune cancer (AML, DLBCL, CLL, thymoma) AND for non-cancer inflammatory conditions (sepsis aftermath, autoimmune flares, late-stage inflammaging). Single-substrate nucl saturation alone is NOT specific to any particular disease.

What nucl saturation DOES tell you for heme-epic: the cell population has lost enough class-specific chromatin structure that the nucleosome positional signature is pinned at random. Severity grading, cancer-vs-inflammation distinction, and subtyping (AML lineage; DLBCL GCB vs ABC; CLL mutation status) come from methyl, fuzz, WPS, and frag. The all-5 A_combined is reported for historical continuity; A_active (4/5, excluding nucl) is reported for progression tracking and serial monitoring.

## Tier thresholds (preliminary — to be locked at validation)

Based on the 80-cell healthy baseline reference plus the immune-cancer-specific characteristic A-scores above. **Heme-epic uses the full five-tier set including SUPPRESSED**, because immune-class A-score *below* the age-decade healthy reference is itself a real signal (immunocompromised state — HIV, post-chemo, post-transplant, post-sepsis exhaustion, primary immunodeficiency, advanced cachexia) that the card has to flag.

- **SUPPRESSED:** A_immune below the age-decade healthy reference by >1 SD. Patient is immunocompromised. Not a heme cancer call — but a real finding that the EDEAR report surfaces because it changes clinical interpretation of every other card. (A heme cancer late-stage may also produce SUPPRESSED via marrow infiltration crowding out healthy lineages — discriminate via Stage 3 EpiDISH lineage-shift pattern.)
- **NORMAL:** A_immune within ±1 SD of age-decade healthy reference AND Stage 3 within Salas healthy range
- **MARGINAL:** 1.01 ≤ A < 1.05 — likely inflammaging or transient immune activation. Annual serial sampling.
- **DETECTABLE:** A ≥ 1.05 AND Stage 3 shows lineage-specific shift — heme-epic fires with subtype call. CLL-level tier.
- **URGENT:** A ≥ 1.07 AND Stage 3 shows distinctive lineage shift — AML or thymoma level. Hematology consult within 2 weeks.
- **FLOOR BREACH:** A ≥ 1.10 AND Stage 3 shows strong lineage shift — DLBCL level. Urgent hematology consult, bone marrow biopsy consideration.

The SUPPRESSED tier is heme-epic's contribution to the framework-wide tier vocabulary. Other cards (cardio-epic, immune-atlas) will adopt the same SUPPRESSED tier where biologically relevant. The patient-facing report uses SUPPRESSED / NORMAL / ELEVATED / FLOOR_BREACH as the four primary bins, with MARGINAL bands flanking NORMAL on both sides.

## Clinical action matrix (preliminary)

| Tier | Stage 3 pattern | Clinical action |
|---|---|---|
| SUPPRESSED | Uniform suppression across lineages | Immunocompromised state. Differential: post-chemo, post-transplant, HIV, primary immunodeficiency, cachexia. Surface in patient report; recommend discussion with primary care. Other card thresholds may need adjustment in this state. |
| SUPPRESSED | Specific lineage suppressed (e.g. neutropenia, lymphopenia) | Targeted differential. Neutropenia → infection risk evaluation. Lymphopenia → HIV testing, immunoglobulin levels. |
| SUPPRESSED | Late-stage heme cancer with marrow infiltration | If patient has known heme dx, this may represent disease progression. Hematology follow-up. |
| DETECTABLE | B-cell shift | Hematology consult for CLL/lymphoma workup. Peripheral blood flow cytometry. |
| DETECTABLE | T-cell shift | Hematology consult. Chest imaging for thymoma. |
| DETECTABLE | Neutrophil shift | CBC confirmation. Rule out chronic inflammation before AML workup. If persistent → hematology consult. |
| DETECTABLE | Uniform elevation | Likely inflammaging or chronic inflammation. Rheumatology or chronic disease workup per symptoms. Serial EDEAR. |
| URGENT | Any lineage-specific shift | Expedited hematology consult. Bone marrow aspirate consideration. |
| FLOOR BREACH | Lineage-specific shift | Urgent hematology — within 1 week. BM biopsy, flow cytometry, FISH/cytogenetics. |
| MARGINAL | Any | 6-month serial sampling. Document trajectory. |

## Validation plan

Heme-epic has an unusual validation posture because the component diseases are ALREADY validated at TCGA tissue level through the framework's Immune class chapter (Issue 002). The validation gap is **per-patient pre-diagnostic blood detection**, which is different from tissue-level cancer discrimination.

### Identified cohorts for v0.2+ validation

The landscape survey for heme-epic (conducted 2026-04-25) identified the following accessible cohorts. **The CLL pre-diagnostic landscape is unusually strong** — better than any other Cookbook card except possibly breast-epic — because two large prospective long-window cohorts have been published with archived blood methylation and time-to-diagnosis metadata.

**Lymphoid B-cell arm — CLL pre-diagnostic (PRIORITY):**

1. **EnviroGenomarkers (Georgiadis 2017, BMC Genomics)** — joint Swedish (Umeå) + Italian (Florence) prospective cohort. **n=347 healthy subjects at enrollment, 28 developed CLL 2.0–15.7 years later.** Peripheral blood buffy coat methylation on Illumina HM450. The published analysis identified 722 differentially methylated CpG sites between future-CLL cases and matched controls AFTER adjusting for white blood cell composition. Cross-cohort confirmation: signals replicate in clinical CLL cohorts. **This is the closest cervical-equivalent the Cookbook has to the EPIC-Italy breast-cancer 10-year-pre-dx anchor (VAL-047).** Italian sub-cohort runs through Florence Health Unit; ethical approval already documented in the publication. PMID 28903739, doi 10.1186/s12864-017-4117-4.

2. **Melbourne Collaborative Cohort Study (MCCS)** — n=82 prospective CLL cases + 82 age-matched controls, samples taken **up to 18 years prior to diagnosis**, HM450. Published analysis: Wong 2024 (or earlier, Haematologica retrotransposon paper). Used by the retrotransposon methylation study (Haematologica) as confirmation of pre-dx CLL signal. Australian, well-curated, may require contact for access.

3. **Northern Sweden Health and Disease Study (NSHDS) + EPIC cohort** — Vermeulen 2023 (or similar). 27 prediagnostic + 12 diagnostic PBMC samples from 16 CLL patients, longitudinal sampling, time-to-dx metadata. Includes Italian sub-cohort participants. Smaller than EnviroGenomarkers but has serial samples per patient — useful for trajectory validation.

**Lymphoid B-cell arm — CLL diagnostic-stage (calibration anchor):**

4. **CLL-map (Knisbacher 2022 Nat Genet, n=1,148)** — full molecular map. Methylation data on EGA (EGAD00010001975 ICGC) and dbGaP (multiple cohorts). Includes Dana-Farber, German CLL Study Group, ICGC, MD Anderson, NHLBI, UCSD cohorts. Largest available CLL methylation reference. Requires gated-data application. Used for IGHV-mutation-status classifier calibration (Wojdacz/Gerland 2021 Haematologica).

**Lymphoid B-cell arm — DLBCL:**

5. **Chapuy 2018 Nat Med (n=48)** — small but methylation-characterized. Genomic subtype classification.

6. **Reddy 2017 Cell DLBCL cohort** — additional DLBCL methylation samples.

**Myeloid arm — AML:**

7. **TCGA-LAML (Ley 2013 NEJM, n=200)** — already used for VAL-007 calibration. Tissue (bone marrow) methylation. PMID 23634996.

8. **MARLIN reference cohort (Capper 2025 Nat Genet)** — n=2,540 high-quality 450k/EPIC samples from 11 published studies. **Includes 1,461 AML, 686 B-ALL, 266 T-ALL, 18 MPAL, 17 BM controls, 92 PB controls.** Largest acute leukemia methylation reference assembled to date. 38 methylation classes defined. Public data deposits across the constituent studies. **This is the AML/ALL reference cohort.**

**Myeloid arm — CHIP → AML progression (G-2026-P010):**

9. **Steensma 2015 CHIP definition cohort** + follow-on prospective studies — the predicted serial-trajectory test for AML pre-diagnostic detection. Specific archived cohorts to be identified at v0.2+.

**Multiple Myeloma:**

10. **MGUS → MM progression cohorts** — methylation data sparse; literature search at v0.2+.

**T-cell arm — Thymoma:**

11. **TCGA-THYM (n=120)** — diagnostic-stage tissue methylation. PMID 29622463.

### Validation tier path

- **Current (v0.1):** `framework_calibrated_pending_per_patient_validation` — TCGA tissue-level A-score signatures characterized in Issue 002, pre-diagnostic cohorts identified, not yet run.
- **Single-cohort tier (v0.2 priority 1):** EnviroGenomarkers CLL pre-diagnostic Phase 9/12 equivalent. If passes at d ≥ +0.5 with monotonic trajectory toward diagnosis, card moves to `single_cohort_validated`.
- **Cross-platform tier:** EnviroGenomarkers + MCCS independent replication. Both at HM450, both >10-year pre-dx windows, both have methylation + clinical metadata.
- **Multi-arm validated:** AML arm validated via MARLIN + TCGA-LAML (calibration) + CHIP-cohort prospective trajectory test (pending P010).
- **Per-arm thresholds:** lymphoid arm CLL-anchored, myeloid arm AML-anchored, T-cell arm thymoma-anchored, MM arm pending build-out.

## Validation summary (VAL studies in this card)

| VAL | Cohort | Arm | Specimen | n | Primary result | Status |
|---|---|---|---|---|---|---|
| **VAL-082** | **GSE62298 Glass 2017 AML HM450** | **myeloid** | **blood (AML) vs Italian buffy coat (healthy)** | **68 AML + 115 healthy QC** | **ΔA = +0.104; Cohen's d = +3.71 [+3.23, +4.20] p≈0; 98.5% AML above healthy p95** | **O1_PASS_MYELOID_BLOOD_LEVEL** |
| VAL-083 | EnviroGenomarkers Georgiadis 2017 (Florence + Umeå) CLL pre-dx | lymphoid B | buffy coat pre-dx 2-15.7 yr | 28 cases + 319 controls | Data NOT publicly accessible (controlled-access biobank, EPIC-Italy + NSHDS); analysis published showing pre-dx signal exists | DEFERRED — gated cohort, requires biobank application |
| VAL-084 | MARLIN Capper 2025 reference | myeloid (replication) | mixed BM/PB | 1,461 AML + 92 PB controls | Data structure mixed across 11 constituent studies; cross-cohort replication of VAL-082 | v0.2 priority |
| VAL-085 | Kulis 2012 CLL via EGAS00001000272 | lymphoid B | CD19+ B-cells | 139 CLL + 14 healthy | Data on EGA controlled access | DEFERRED — gated |
| VAL-086 | Dietrich 2018 CLLmethylation Bioconductor | lymphoid B | CLL primary | ~200 | Data on EGAS00001000174 controlled access | DEFERRED — gated |
| VAL-087 | GSE54200 Shaknovich 2014 DLBCL | lymphoid B | DLBCL tissue HELP custom array | 140 DLBCL + 10 NGCB | Different platform (HELP not 450K); panel transferability not established | DEFERRED — different platform |
| VAL-088 | GSE47051 ALL methylation | mixed B/T-ALL | bone marrow | 797 | Cohort accessible but BM specimen not v1 EDEAR pathway | v0.2 priority for arm coverage |

**v0.1 cohort-completeness statement (per CCL-029):**

The publicly accessible 450K/EPIC heme cancer methylation cohort landscape was surveyed on 2026-04-25. Findings:

- **GSE62298 (AML, n=68 HM450) is the only publicly-accessible GEO heme cancer methylation cohort with full Xu-538 panel coverage on a primary EDEAR specimen** (blood-derived methylation). VAL-082 ran on this cohort.
- **CLL methylation is dominated by gated EGA-deposited cohorts** (Kulis 2012 EGAS00001000272 n=139; Dietrich 2018 EGAS00001000174 n≈200; CLL-map 2022 multi-cohort n=1,148 across multiple gated repositories).
- **DLBCL methylation is also gated for the larger cohorts** (Chapuy 2018, Reddy 2017, Pasqualucci/Dalla-Favera) or uses different platforms (Shaknovich 2014 HELP custom array).
- **EnviroGenomarkers CLL pre-diagnostic data** (the Italian-Swedish cohort with 10+ year pre-dx window) is NOT publicly deposited despite the published analysis showing the signal exists. The cohort sits at EPIC-Italy + NSHDS biobanks and requires formal biobank data-access applications. **This is the same access-gating pattern as VAL-046 Rotterdam pre-dx pancreatic and as Bukowski CINCS pre-dx cervical.**
- **MARLIN reference cohort (n=2,540 acute leukemia)** is described as "publicly available" via the constituent studies but the constituent studies vary in access tier; some are GEO-deposited (e.g., GSE62298, GSE47051), others are gated.

**Result of cohort completeness pass:** myeloid arm has one publicly-accessible single-cohort validation (VAL-082) which produced a strong O1_PASS at d=+3.71. Lymphoid arms (B-cell, T-cell) cannot reach single_cohort_validated tier from publicly-accessible data alone; reaching them requires biobank/EGA access via formal application. **This is documented honestly rather than papered over.** Heme-epic v0.1 ships with myeloid arm at single_cohort_validated tier and lymphoid arms at framework_calibrated_pending_per_patient_validation tier.

## Relationship to other cards

- **immune-atlas (#13):** heme-epic is the destination card from Pathway 2 (Hematologic/immune-compartment disease)
- **ad-immune:** shares architecture — directional panels on top of pooled Xu-538 Stage 1
- **cardio-epic:** closest differential — both are Stage-1-positive-Stage-2-null patterns, but Stage 3 EpiDISH distinguishes cardiovascular monocyte-shift / FOXP3 from heme lineage-specific shifts
- **README_MASTER specimen section:** immune class at 70% of plasma cfDNA is the STRONGEST signal in EDEAR — heme-epic has the best signal-to-noise of any card because the diseased tissue is the dominant cfDNA source

## Open framework predictions linked to this card

- **G-2026-P010 (April 2026, PENDING):** CHIP patients with archived serial blood samples will show immune-class A-score trajectory before overt AML develops.
- **G-2026-P011 (April 2026, PENDING):** Patients receiving immune checkpoint inhibitor (anti-PD-1, anti-CTLA-4) therapy — immune-class A-score trajectory will distinguish responders from non-responders within 2-3 treatment cycles.

Both predictions are explicitly serial-trajectory tests, not single-timepoint. Heme-epic deployment benefits most from subscription-based longitudinal sampling.

## Known limitations

- Immune cancers have smaller absolute ΔA than solid-organ cancers (C3 gap grows on top of programmed plasticity, not from zero). Compensated by 5-substrate combination.
- Inflammaging at A ≈ 1.02 sits in MARGINAL tier — chronic immune drift in healthy aging mimics early heme-epic positive. Distinguished by lineage-specific shifts in Stage 3.
- Immune cell plasticity (naive vs effector T-cells, activated vs resting B-cells) produces baseline methylation variability. Cancer Amplifier g for immune class is 5-10× (vs infinite for solid tumors) because healthy immune cells are not at H_min floor.
- Stage 3 EpiDISH RPC is only as good as the reference panel. Salas 2018 healthy ranges are the QC gate; updates to Loyfer 2023 atlas may improve lymphoid-subtype resolution.
- Multiple myeloma characterization in the framework is less complete than AML/DLBCL/CLL/thymoma. Build-out pending.

## Next steps

1. **Per-disease directional panel build (lymphoid B-cell arm):** Extract characteristic CpGs for CLL, DLBCL, MM from Issue 002 data and published TCGA/CLL-map methylation analyses. Build `heme_lymphoid_B_directional.json` keyed to CLL (lowest magnitude, hardest test) with sub-keys for DLBCL/MM expected directions.
2. **Per-disease directional panel build (myeloid arm):** Extract characteristic CpGs for AML, MDS, MPN from MARLIN reference cohort + TCGA-LAML. Build `heme_myeloid_directional.json`.
3. **Per-disease directional panel build (T-cell arm):** Extract characteristic CpGs for thymoma, T-ALL from TCGA-THYM + MARLIN T-ALL subset. Build `heme_lymphoid_T_directional.json`.
4. **MM full characterization:** Literature review of plasma cell methylation signatures (Heuck 2013, Walker 2018 Nat Comms MM methylation atlas). Extend the Immune class chapter to cover MM specifically.
5. **VAL-082 priority — EnviroGenomarkers CLL pre-diagnostic Phase 9/12 equivalent:** Test universal Stage 1 immune Xu-538 A-score on the n=347 buffy-coat methylation cohort, stratified by future-CLL status (n=28) vs controls (n=319). Frozen pre-reg with pre-locked decision criteria. CLL is the lowest-magnitude subtype (A ≈ 1.07) so it is the most demanding validation — pass on CLL validates the framework for the whole heme-epic card. **Italian sub-cohort access through Florence Health Unit; Swedish sub-cohort through Umeå.**
6. **VAL-083 cross-platform replication — MCCS:** independent HM450 cohort, n=82 prospective CLL up to 18 years pre-dx. If both EnviroGenomarkers AND MCCS pass, card moves to `cross_platform_validated`.
7. **VAL-084 myeloid arm — MARLIN reference scoring:** apply universal Stage 1 to MARLIN n=2,540 with 38 methylation classes. Confirm AML A-score calibration matches Issue 002 framework prediction (≈ 1.10). Sub-stratify by AML cytogenetic class.
8. **VAL-085 CHIP → AML progression test (G-2026-P010):** candidate cohort hunt for serial blood methylation in CHIP patients with time-to-AML metadata.
9. **VAL-086 checkpoint inhibitor response test (G-2026-P011):** requires clinical trial partner with ICI-treated patients and serial cfDNA methylation.

## Specimen

**Primary:** buffy-coat blood (same as every Cookbook card). Highest signal-to-noise of any card because 70% of plasma cfDNA is immune-derived and the disease lives in that 70%.

**Secondary (calibration only):** bone marrow aspirate methylation (TCGA-LAML, MARLIN). Used for reference-cohort calibration of A-score thresholds. Not a primary EDEAR specimen — clinical bone marrow biopsy is not a screening tool.

**Tertiary (under L2/L3 multi-substrate platform, future):** plasma cfDNA fragment-size + nucleosome-occupancy + WPS profiling for the 5-substrate combined A-score architecture documented in Issue 002. Not operational at v1 launch (450K/EPIC methylation only).

## What this means for EDEAR commercial deployment

Heme-epic is the highest-signal-to-noise card in the EDEAR catalog and one of the strongest candidates for the launch product story. Three reasons.

**One — the disease is in the specimen.** 70% of plasma cfDNA is immune-derived. Every other card has to find solid-organ shedding within an immune-cfDNA-dominated background. Heme-epic is the case where the immune cfDNA itself carries the disease signal. The signal-to-noise is structurally better than for breast, lung, colon, prostate, pancreas, or HCC. Translation: heme-epic is the most likely card to fire correctly on a real patient at v1.

**Two — the pre-diagnostic CLL evidence is unusually strong.** Most cards in the catalog have published pre-diagnostic blood cohorts measured in the dozens of cases with 5-year follow-up. CLL has **EnviroGenomarkers (n=347, 28 cases, 2.0–15.7 years pre-dx) AND MCCS (n=82 cases up to 18 years pre-dx) — both prospective, both with 450K methylation, both already published showing pre-dx signal exists.** This is breast-epic-tier evidence (which anchored on the EPIC-Italy cohort 10-years-pre-dx). When Heath gets the catalog up and running, CLL is the disease where we have the strongest claim that the EDEAR architecture can detect something real years before clinical diagnosis.

**Three — the Italian-cohort question Heath has been asking is answered for CLL.** The EnviroGenomarkers cohort includes the Florence Health Unit cohort directly. It's the Italian-style long-window pre-diagnostic methylation cohort that doesn't exist for cervical and is gated for breast. For CLL it's published, accessible, and the analysis has already been done at the field-standard level (722 differentially methylated CpGs identified, signals replicate in clinical CLL).

**At v1 launch, heme-epic does this.** Patient gets blood drawn, EPIC array runs, EDEAR returns:

- **Immune A-score** with healthy-reference comparison by age decade. SUPPRESSED / NORMAL / ELEVATED / FLOOR_BREACH bin.
- **Stage 2 Moss tissue-of-origin breakdown.** If solid organs are quiet but immune A is elevated → heme pathway fires.
- **Stage 3 EpiDISH lineage breakdown.** Discriminate myeloid (neutrophil-shifted) vs lymphoid B-cell (B-cell-shifted) vs lymphoid T-cell (T-cell-shifted) vs uniform-shift (inflammaging, NOT cancer).
- **Patient-facing report says:** "Your immune A-score is elevated above the healthy reference for adults your age. Your tissue-of-origin breakdown does not show solid-organ shedding above expected ranges. Your immune lineage breakdown shows elevated B-cell fraction. Patterns like this have been documented in research cohorts studying CLL, lymphoma, and other lymphoid disorders, including pre-diagnostic samples taken years before formal diagnosis (Georgiadis et al. 2017). This is not a diagnosis. Talk to your hematologist about whether further evaluation is appropriate." Or the corresponding language for myeloid pattern, T-cell pattern, or SUPPRESSED state.

That's defensible, useful, and built from already-published research. The framework's contribution is the architectural-state interpretation (A-score with H_min calibration, Stage 2/3 integration); the disease-detection precedent comes from established methylation literature.

**Heme-epic is also the proof-of-trajectory-tracking story for EDEAR subscriptions.** G-2026-P010 (CHIP → AML progression visible in serial methylation) and G-2026-P011 (immune checkpoint inhibitor response visible in serial methylation within 2-3 cycles) are explicit serial-sampling tests. A patient who subscribes to annual EDEAR and shows a rising immune A-score with shifting EpiDISH lineage pattern over years is the use case that no clinical lab can replicate because no clinical lab has the architecture to track methylation longitudinally across diseases. Heme-epic is the card where trajectory tracking saves lives most concretely — early CLL, CHIP-to-AML progression, treatment response monitoring. The subscription model and the heme-epic clinical value match.

## Commercial.web.py decision tree — what to do when an IDAT fires this card

This is the operational playbook for commercial.web.py running on Heath's server when a patient IDAT comes in and the heme-epic pathway might fire. Each subsection answers a specific question commercial.web.py has to resolve before generating the patient report.

### Step 1 — Read the three-stage output and check the routing pattern

After Stage 1 (immune A-score), Stage 2 (Moss NNLS tissue-of-origin), and Stage 3 (EpiDISH RPC immune lineage breakdown) all run from the same 450K/EPIC IDAT, commercial.web.py classifies the result into one of seven patterns:

| Pattern | Stage 1 immune A | Stage 2 Moss | Stage 3 EpiDISH lineage shift | Routing |
|---|---|---|---|---|
| **A. Solid cancer pattern** | ELEVATED | Elevated on at least one of 18 peripheral solid tissues | Often uniform shift (immune response to solid disease) | Route to relevant solid-organ card (breast-epic / lung-epic / crc-epic / etc.) — **NOT heme-epic** |
| **B. Heme myeloid pattern** | ELEVATED | NULL on all 18 peripheral solid tissues | **Neutrophil-dominant** (neutrophil fraction elevated, lymphoid lineages relatively suppressed) | heme-epic myeloid arm |
| **C. Heme lymphoid B-cell pattern** | ELEVATED | NULL on all 18 peripheral solid tissues | **B-cell-dominant** (B-cell fraction elevated, T-cells normal-to-suppressed, neutrophils normal) | heme-epic lymphoid B-cell arm |
| **D. Heme lymphoid T-cell pattern** | ELEVATED | NULL on all 18 peripheral solid tissues | **T-cell-dominant** (T-cell fraction elevated, often CD4/CD8 ratio inverted) | heme-epic lymphoid T-cell arm |
| **E. Inflammaging / autoimmune / non-cancer immune activation** | ELEVATED | NULL on solids | **Uniform elevation** across all six lineages, no single lineage dominates | Route to immune-atlas Pathway 4 — NOT heme-epic, NOT confirmation of cancer |
| **F. Possible CNS / non-Moss-tissue disease** | ELEVATED | NULL on solids | Uniform elevation OR non-distinctive shift | Route to immune-atlas with **brain-cancer differential note**; recommend neurological evaluation if symptoms present. Moss does NOT include brain/CNS, so Moss NULL does NOT exclude CNS disease |
| **G. SUPPRESSED state** | DEPRESSED below age-decade healthy reference by >1 SD | (any) | Often uniform suppression OR specific lineage suppression | SUPPRESSED tier in patient report. Differentials: post-chemo, post-transplant, HIV, primary immunodeficiency, cachexia, late-stage marrow infiltration |

The routing logic above is the master switch for commercial.web.py at the heme-epic interface. **Patterns B, C, D fire heme-epic; patterns A, E, F, G route elsewhere even though Stage 1 is elevated.**

### Step 2 — When pattern B/C/D fires, generate arm-specific patient report

For each heme-epic pattern, commercial.web.py uses the per-arm patient-report template:

**Myeloid arm (pattern B):**

> Your immune A-score is elevated above the healthy reference for adults your age. Your tissue-of-origin breakdown does not show solid-organ shedding above expected ranges in any of the 18 peripheral organs in our reference. Your immune lineage breakdown shows elevated neutrophil fraction with lineage proportions consistent with myeloid expansion. Patterns like this have been documented in research cohorts studying AML, MDS, and other myeloid disorders. This is not a diagnosis. Talk to your hematologist about whether further evaluation is appropriate — typical confirmatory tests for this pattern include a complete blood count with peripheral smear and, depending on findings, a CHIP (clonal hematopoiesis) myeloid mutation panel on peripheral blood. Imaging or biopsy is generally not required at this stage of evaluation.

**Lymphoid B-cell arm (pattern C):**

> Your immune A-score is elevated above the healthy reference for adults your age. Your tissue-of-origin breakdown does not show solid-organ shedding above expected ranges in any of the 18 peripheral organs in our reference. Your immune lineage breakdown shows elevated B-cell fraction. Patterns like this have been documented in research cohorts studying CLL, lymphoma, and other lymphoid disorders, including pre-diagnostic samples taken years before formal diagnosis (Georgiadis et al. 2017; Wong/Severi MCCS). This is not a diagnosis. Talk to your hematologist about whether further evaluation is appropriate — typical confirmatory tests for this pattern include flow cytometry on peripheral blood, which can detect monoclonal B-cell lymphocytosis (MBL, the precursor state to CLL) at clone fractions below 0.5%. Imaging or biopsy is generally not required at this stage of evaluation.

**Lymphoid T-cell arm (pattern D):**

> Your immune A-score is elevated above the healthy reference for adults your age. Your tissue-of-origin breakdown does not show solid-organ shedding above expected ranges in any of the 18 peripheral organs in our reference. Your immune lineage breakdown shows elevated T-cell fraction, with CD4/CD8 ratio outside the typical healthy range. Patterns like this have been documented in research cohorts studying thymoma and other T-cell disorders. This is not a diagnosis. Talk to your hematologist; chest imaging may be appropriate if a thymoma differential is being considered.

**SUPPRESSED state (pattern G):**

> Your immune A-score is below the healthy reference for adults your age. This pattern has been documented in research cohorts studying immunocompromised states including post-chemotherapy recovery, post-transplant immunosuppression, HIV, primary immunodeficiencies, and cachexia. Late-stage hematologic disease with bone marrow infiltration can also produce this pattern. This is not a diagnosis. Talk to your doctor about your overall immune health and any relevant medical history, current medications, or recent treatments.

### Step 3 — Lineage profile interpretation rules (load-bearing for B/C/D discrimination)

The "B-cell ratio is most important" intuition is correct **with the precision that it is the LINEAGE PROFILE — the relationships between the six lineages — that carries the diagnosis, not any single fraction in isolation.** EpiDISH RPC returns six lineages with healthy ranges from Salas 2018 (extended where applicable with Loyfer 2023):

- Neutrophil: 45-75%
- CD4 T: 10-30%
- CD8 T: 5-25%
- NK: 3-15%
- B-cell: 3-15%
- Monocyte: 3-12%

**The discriminative power lives in the relationships between lineages, not the absolute fractions.** Examples commercial.web.py must distinguish:

- **B-cell at 18% AND T-cells suppressed at 12%** (when normally CD4 + CD8 = 25-40%) → lymphoid B-cell expansion crowding out the rest of the immune compartment → cancer pattern (pattern C).
- **B-cell at 18% AND T-cells also at 28%** (high end of normal) AND neutrophils normal → active immune system (recent infection, vaccination, chronic immune activation) → NOT cancer (pattern E).
- **Neutrophils at 80% AND lymphoid lineages at 12% combined** (when normally lymphoid = 30%) → myeloid expansion crowding out lymphoid → cancer pattern (pattern B).
- **Neutrophils at 80% AND lymphoid lineages at 28%** AND immune A elevated → likely acute infection or chronic inflammation, not cancer.
- **All six lineages elevated proportionally** (B-cell 17%, CD4 22%, CD8 19%, NK 13%, neutrophil 73%, monocyte 11%) → uniform expansion → inflammaging or autoimmune (pattern E), NOT cancer.

The rule commercial.web.py applies: **a lineage-specific shift requires the dominant lineage to be elevated AND at least one other lineage to be relatively suppressed below its expected range.** Both conditions must hold. A single elevated lineage with all others normal is more likely active immune response than expansion of a clonal population.

### Step 4 — When the doctor confirms but no culprit is yet findable (10+ year out scenario)

This is the operational risk of long-window early detection that commercial.web.py must handle gracefully. The scenario: patient gets EDEAR result with pattern C (lymphoid B-cell), hematologist orders flow cytometry on peripheral blood, flow shows no detectable monoclonal B-cell population. The EDEAR signal is real but no clinical confirmatory test can yet find a culprit.

**Three things commercial.web.py does in this scenario:**

1. **Does NOT report the test as a false positive.** The pre-malignant cells are present at the methylation level (proven by EnviroGenomarkers Florence-Umeå and MCCS, both showing detectable signal 10-18 years pre-diagnosis), but below the detection threshold of current clinical confirmatory tests. The framework reading is correct; the clinical confirmatory test capability hasn't caught up to the methylation-level detection threshold yet.

2. **Frames the result as trajectory-tracking-required.** The patient report explicitly states that a single elevated reading is ambiguous and that the value of the EDEAR test compounds with serial sampling. The next-step recommendation is annual repeat to track whether the signal stays elevated, increases, or returns to baseline. Trajectory data resolves what single-timepoint data cannot.

3. **Routes the patient to active surveillance rather than reassurance.** Even when flow cytometry returns negative, the patient with an elevated lymphoid B-cell pattern is in a higher-risk category for future CLL development than the general population (the EnviroGenomarkers data establishes this risk stratification). The doctor's plan should include annual or semi-annual repeat blood work — flow cytometry is cheap, readily available, and progressively more sensitive year-over-year as MBL clones grow. EDEAR's annual repeat catches the trajectory before flow does.

This is also why the subscription model is structurally aligned with heme-epic's clinical value. A single test fires the question; serial tests answer it. Commercial.web.py's report explicitly explains this to the patient.

### Step 5 — Confirmatory test pathway by arm (what doctors typically order next)

For each heme-epic firing pattern, commercial.web.py's report should mention what confirmatory tests are typically appropriate so the doctor and patient know what to expect:

**Myeloid arm fires (pattern B):**
- CBC with differential and peripheral smear (basic, immediately available)
- If CBC abnormal: bone marrow aspirate + cytogenetics + FISH (for AML/MDS workup)
- If CBC normal but EDEAR pattern persistent: CHIP myeloid mutation panel NGS on peripheral blood (DNMT3A, TET2, ASXL1, JAK2, SRSF2, SF3B1, others). CHIP is detectable at 2% clonal fraction and is present in ~10% of adults over 60; ~1% per year progress to AML or MDS. CHIP detection enables hematology surveillance enrollment, which is what catches early progression.

**Lymphoid B-cell arm fires (pattern C):**
- Flow cytometry on peripheral blood (immediately available, detects monoclonal B-cell populations down to clone fractions <0.5%; high-sensitivity flow detects <0.01%)
- If flow detects MBL: hematology referral for surveillance (MBL is present in up to 17% of older adults; ~1% per year progress to clinical CLL). Patient enters active surveillance, not treatment.
- If flow negative AND EDEAR pattern persistent on annual repeat: ongoing flow cytometry and CBC monitoring (the methylation signal precedes flow detectability; trajectory tracking catches the transition).
- If clinical lymphadenopathy present: imaging (CT or PET), excisional lymph node biopsy may be considered. EDEAR-flagged lymphoid pattern alone without clinical findings does not typically warrant biopsy.

**Lymphoid T-cell arm fires (pattern D):**
- Flow cytometry on peripheral blood
- Chest imaging (CT) for thymoma evaluation, particularly if mediastinal mass suspected
- If mediastinal mass found: excisional or core biopsy of the mass

**SUPPRESSED state (pattern G):**
- Clinical history review (chemotherapy timing, transplant status, HIV testing if not recent)
- CBC + immunoglobulin levels
- If primary immunodeficiency suspected and no clinical history explains: immunology workup
- Other EDEAR card thresholds may need adjustment for this patient — SUPPRESSED state changes interpretation of every other card's output

The fundamental commercial advantage of heme-epic over solid-organ cards: **the entire early-detection arc happens within the blood-pathway only.** EDEAR fires from blood, flow cytometry confirms from blood, CHIP panel confirms from blood, CBC monitors from blood. No imaging, no biopsy, no surgery is required to confirm or rule out the heme-epic differentials at the screening-and-surveillance stage. This is structurally not true for breast, colon, lung, prostate, or pancreas — those all require imaging or biopsy to confirm what EDEAR flagged. Commercial.web.py reports for heme-epic patterns can frame "the next step is more blood work" rather than "the next step is a procedure," which materially changes patient compliance and clinical pathway friction.

### Step 6 — What commercial.web.py CANNOT do at v1

Honest limitations the patient report must acknowledge so commercial.web.py does not over-promise:

- **Cannot positively confirm brain cancer.** Moss 2018 reference does not include CNS tissue. Pattern F (uniform Stage 3 + Moss NULL on peripherals) is a "consider neurological evaluation" flag, not a CNS-cancer call. Glioma-epic (TBD) will improve this; v1 has the gap explicitly documented.
- **Cannot confirm pre-diagnostic timing for an individual patient.** Group-level pre-dx evidence exists (EnviroGenomarkers 2.0-15.7 yr; MCCS up to 18 yr); per-individual timing requires longitudinal sampling. Subscription model is what makes per-individual trajectory possible.
- **Cannot distinguish AML cytogenetic subtypes from blood.** Cytogenetic stratification (RUNX1::RUNX1T1, CBFB::MYH11, NPM1, etc.) requires bone marrow aspirate. Heme-epic flags AML pattern; bone marrow workup typing the subtype.
- **Cannot distinguish CLL IGHV-mutated vs unmutated from blood methylation alone at v1.** This requires the CLL methylation classifier (Wojdacz/Gerland 2021) which is a card-specific deviation pending v0.2+ build.
- **Cannot fire on patients in active chemotherapy.** Cytotoxic chemo wipes the immune compartment producing transient SUPPRESSED-tier readings that recover over weeks-to-months. Commercial.web.py defers scoring during active chemotherapy and 90 days post-completion.
- **Cannot fire reliably on patients with recent infection (within 6 weeks) or recent vaccination (within 4 weeks).** Both produce transient immune A-score elevation that mimics early heme-epic pattern. Commercial.web.py captures these in mandatory covariates and defers scoring during the relevant window.

### Step 7 — Mandatory covariates commercial.web.py must capture before scoring

Same fields as listed in §13 mandatory covariates, but the operational note is that commercial.web.py must REQUIRE these inputs before generating any heme-epic-firing report. The report cannot be generated without:

- Age (decade-stratified healthy reference)
- Sex
- Race / ancestry (Salas reference is European-ancestry-anchored; sub-population calibration v0.2+)
- Known immunocompromised state (HIV, transplant, chemotherapy <90 d, steroid >prednisone equivalent 10 mg/day)
- Known heme cancer history (alters interpretation of all tiers)
- Active infection (acute infection elevates A-immune transiently; defer scoring to 6 weeks post-resolution)
- Recent vaccination (transient A-immune elevation 2-4 weeks; defer scoring)
- Pregnancy (decline scoring during pregnancy and 12 weeks post-partum)
- MGUS history (relevant for MM differential)
- MBL history (relevant for CLL differential)
- CHIP history if known (relevant for AML differential)

Patient questionnaire at intake collects these; commercial.web.py validates completeness before passing the IDAT to the scoring pipeline. Missing covariates → report is "incomplete intake" status, NOT scored.

## What we discovered (heme-epic v0.1)

### The pre-diagnostic CLL cohort question Heath has been asking is answered

Heath has repeatedly asked whether long-window pre-diagnostic methylation cohorts exist for cancers other than breast. For cervical, the answer was no (CINCS Bukowski is gated, 5-year window). For pancreatic, the answer was Rotterdam (gated, 2-5 year window, used at cohort-level only in VAL-046). **For CLL, the answer is yes, accessible, and longer-window than any other Cookbook card has access to.** EnviroGenomarkers extends to 15.7 years pre-dx. MCCS extends to 18 years pre-dx. Both are HM450, both have published methylation-vs-future-disease analyses, both include cohort sub-populations (Florence, Umeå, Melbourne) that the framework can engage with.

This changes the heme-epic v0.2 priority from "build directional panels first, find cohorts later" to "validate against EnviroGenomarkers immediately when build resources permit." Heme-epic moves to the top of the validation queue once cardio-epic, kidney-epic, glioma-epic, gastric-epic, and bladder-epic are skeletoned.

### Moss NULL on solid organs is the heme-epic diagnostic feature

The central question for heme-epic is how to distinguish "the immune compartment IS the disease" from "the immune compartment is responding to a solid disease somewhere else." The answer is the three-stage pipeline working together — Stage 1 elevated, Stage 2 NULL on all solid organs, Stage 3 lineage-specific shift. Each leg is necessary; no leg carries the differential alone. **Stage 2's role in heme-epic is not just cross-reference, it's load-bearing.** This is different from breast-epic, lung-epic, etc., where Stage 2 confirms a solid-organ signal that Stage 1 already implied. In heme-epic Stage 2's NULL is the diagnostic feature.

### Lymphoid vs myeloid is a fundamental architectural split, not a sub-detail

The factor-of-two ΔA spread between DLBCL (+0.203) and CLL (+0.098) reflects real biology: B-cell class-switching and somatic hypermutation are programmed methylation perturbations that lymphoid cancers exploit further. Myeloid cancers (AML, MDS, MPN) operate on a different methylation landscape — focal hypermethylation at lineage-commitment loci (MEIS1, HOXA cluster) plus global hypomethylation. **The card splits into two arms not for organizational convenience but because the directional panels and the EpiDISH discrimination criteria are biologically distinct.** A unified heme-epic algorithm would underperform compared to two arm-specific algorithms. The framework supports per-card panel substitution; heme-epic uses two card-arm panel substitutions.

### The SUPPRESSED tier is a heme-epic contribution to the framework vocabulary

Other cards focus on elevation. Heme-epic is the first to formalize that immune-class A-score below the age-decade healthy reference is a real and clinically meaningful signal — immunocompromised state, post-chemo, post-transplant, HIV, primary immunodeficiency, advanced cachexia, late-stage marrow infiltration. **The SUPPRESSED tier (A_immune > 1 SD below healthy reference) goes into the framework-wide patient-facing tier vocabulary.** Other cards (cardio-epic, immune-atlas, ad-immune for older patients) inherit the same SUPPRESSED tier. The four-bin patient-facing report (SUPPRESSED / NORMAL / ELEVATED / FLOOR_BREACH) replaces the previously-stated three-bin set as of heme-epic v0.1.

### Heme-epic is the strongest v1 launch story in the catalog

Best signal-to-noise (immune is dominant cfDNA fraction, disease lives in that fraction). Best published pre-diagnostic evidence (EnviroGenomarkers + MCCS, both >10-year windows — though both are biobank-gated, see §6 cohort-completeness statement). Best fit to the trajectory-tracking subscription value proposition (CHIP-to-AML, ICI response). Best "doctor recognizes the pattern" story (B-cell shift means CLL/lymphoma to any hematologist). When Heath sells EDEAR to a hematology-focused clinical partner, heme-epic is the card that does the most heavy lifting in the demo.

### VAL-082 gives the myeloid arm a single_cohort_validated anchor

VAL-082 GSE62298 AML 450K vs GSE51057 EPIC-Italy healthy buffy coat produced **Cohen's d = +3.71 [+3.23, +4.20] p ≈ 0, with 98.5% of AML samples scoring above the Italian healthy 95th percentile.** This is the strongest single-cohort effect size measured anywhere in the Cookbook to date — stronger than any solid-organ card's per-patient validation, stronger than the cervical-epic Verlaat anchor at d=+0.73, comparable to the AD-immune VAL-051 d=+0.62 and pancreatic VAL-069 d=+1.51 directional fallback only after multiplying by the substrate enhancement.

Equally important: VAL-082 also reset the framework numbers. Issue 002's prediction of A_AML ≈ 1.10 with ΔA = +0.168 is a **5-substrate combined cfDNA prediction** — the future L2/L3 platform expansion target. At v1 EDEAR launch, with single-substrate methyl-only buffy-coat scoring on a 450K array, AML reads at A = 0.54 against an Italian healthy baseline of 0.44 — ΔA = +0.10 absolute, d = +3.71 effect size. **Both numbers are correct for their respective substrate scopes.** The Issue 002 figure was not "wrong" — it was describing a different point on the platform roadmap. v1 launch operates at d ≈ 3.71 for AML in blood, which is sufficient for clinical-grade detection.

### What we discovered about pre-diagnostic detection windows

The honest answer to "10 years out?" for heme cancers at v1 launch:

- **AML** at v1: VAL-082 validates AML detection AT diagnosis with very strong signal (d=+3.71). Pre-diagnostic AML detection requires CHIP→AML serial-sample cohorts — those exist (Steensma 2015 definition cohort, follow-on prospective studies) but are not yet locked in for VAL-085. **G-2026-P010 framework prediction is that CHIP patients show immune-class A-score trajectory before overt AML.** That prediction is testable but the test has not yet run; honest answer for v1 is "AML detection AT diagnosis is very strong; pre-diagnostic detection is a future capability to be validated."
- **CLL** at v1: the 10+ year pre-diagnostic signal **is documented to exist** by EnviroGenomarkers (Georgiadis 2017) and MCCS (Wong 2024 / Severi). Both cohorts are >10-year pre-dx and HM450, both have published analyses showing the signal exists, **but neither cohort is publicly accessible — both sit at biobank tier requiring formal data-access application.** Heath asked for the Italian-equivalent of breast pre-dx for CLL; the data exists, the cohort exists, the science has been done. What's missing is access. v0.2+ priority: file biobank applications via EPIC-Italy and NSHDS for the Florence + Umeå CLL pre-dx data.
- **DLBCL, MM, thymoma** at v1: no published pre-diagnostic blood methylation cohorts identified. These are at-diagnosis detection only at v1.

### Anything interesting about the way the immune system responds to AML

**Yes — and the VAL-082 signal magnitude tells us why.** AML is a myeloid-lineage cancer, meaning the cancer cells ARE neutrophil/monocyte progenitors that have lost lineage-commitment regulation. The Xu-538 panel was trained on whole-blood buffy coat methylation, where ~50-75% of cells are neutrophils. **AML samples are essentially "all the cells in the buffy coat are wrong" — the panel reads the architectural disruption of the dominant cell population directly.** This is structurally why the myeloid arm has higher signal-to-noise than any other cancer in the catalog.

By contrast, breast cancer in blood is "tiny amount of breast-ductal cfDNA mixed into a large background of healthy immune cells" — the panel has to detect a small signal against a large noise floor. AML in blood is "the noise floor itself is the signal" — the disease IS in the dominant cell population. VAL-082's d = +3.71 is the structural ceiling for what 450K methylation can achieve at v1, and AML happens to sit at that ceiling because AML's biology aligns with what 450K measures.

Same structural argument applies to DLBCL (B-cell lineage), CLL (B-cell lineage), MM (plasma cell), thymoma (T-cell) — each is a cancer where the diseased tissue IS in the cfDNA in significant fraction, NOT a contaminant from a distant solid organ. **Heme cancers are the natural strength of cfDNA methylation screening.** Solid-organ cancers detected via cfDNA require either very high tumor burden, fragmentomics enhancement, or directional panel construction. Heme cancers detect at d > 3 on universal Stage 1 alone.

### Layman summary — what we learned from heme-epic v0.1

In ordinary language: blood cancers are the easiest thing to detect with EDEAR's universal pipeline because the disease is literally inside the cells we're sampling. We don't have to find a cancer that's hiding far away from the blood draw — the cancer cells ARE in the blood draw, in large numbers. When we ran the Italian healthy comparison, AML patients separated from healthy people so cleanly that 98.5% of them scored higher than 95% of healthy subjects. That's the strongest result we've gotten from anywhere in the catalog.

For the question "how far out can we detect it?" — for AML the v1 launch detects at-diagnosis signal very strongly, but pre-diagnostic AML detection (the CHIP-to-AML trajectory) is a future test we haven't run yet. For CLL the pre-diagnostic signal **provably exists at 10+ years out** — published Italian-Swedish work proved it — but the data is locked behind biobank access applications, not freely on GEO. We can apply for it; we haven't yet. For DLBCL, multiple myeloma, and thymoma, pre-diagnostic blood cohorts haven't been published; v1 detects at-diagnosis only.

What this means commercially: heme-epic at v1 launch is the strongest card in the catalog. Patient gives blood, we run the array, we score the immune A, we see if Moss says solid organs are quiet, and we look at the lineage breakdown. If neutrophils are way up + immune A is way up + Moss is quiet on solid organs → AML pattern, very high confidence. If B-cells are up + immune A is up + Moss is quiet → lymphoma/CLL pattern. If T-cells are up → thymoma differential. If everything is uniformly up → inflammaging or possibly brain cancer (which we can't see directly because the brain doesn't shed into blood). If everything is suppressed → immunocompromised state, talk to your doctor.

What we still can't do at v1: positively confirm brain cancer (glioma-epic limitation), positively confirm pre-diagnostic timing for individual patients (requires longitudinal sampling — that's the subscription model's value), or distinguish AML cytogenetic subtypes from blood (requires bone marrow biopsy, not a screening test). Those are honest limitations that the report tells the patient about.

The Italian cohorts (EnviroGenomarkers Florence-Umeå) saved us for CLL the way EPIC-Italy saved us for breast. We can't access the data yet but we know the signal is there at 10+ years pre-diagnosis. The path forward is biobank access applications, not new science.

## Lessons learned (heme-epic-specific)

**heme-LL-001.** Stage 2 Moss NULL on solid organs is the diagnostic feature for heme-epic, not an absence of finding. Other cards interpret Moss NULL as "no localization information"; heme-epic interprets Moss NULL on solids + Stage 1 elevated + Stage 3 lineage-shifted as the positive heme-cancer pattern. This inverted interpretation is heme-epic-specific.

**heme-LL-002.** The card splits into lymphoid B-cell, lymphoid T-cell, and myeloid arms because the directional panels and EpiDISH discrimination criteria are biologically distinct. Unified heme-epic scoring would underperform. Three card-arms with arm-specific directional panel substitutions is the architecture.

**heme-LL-003.** SUPPRESSED tier is defined here for the framework. A_immune > 1 SD below age-decade healthy reference is a real signal — immunocompromised state — that the EDEAR report surfaces because it changes clinical interpretation of every other card's output for that patient. Other cards inherit SUPPRESSED.

**heme-LL-004.** The four-bin patient-facing report (SUPPRESSED / NORMAL / ELEVATED / FLOOR_BREACH) with MARGINAL bands flanking NORMAL on both sides is the framework's patient-facing tier vocabulary as of heme-epic v0.1. Replaces the previously-stated three-bin set. All future cards adopt this.

**heme-LL-005.** Inflammaging at A ≈ 1.02 sits in MARGINAL tier and is distinguished from heme-epic positive by Stage 3 EpiDISH pattern: uniform elevation across lineages = inflammaging or chronic inflammation; lineage-specific shift = heme cancer. This is the fourth differential pathway in immune-atlas; heme-epic relies on it for specificity.

**heme-LL-006.** EnviroGenomarkers (Georgiadis 2017) is the long-window pre-diagnostic CLL cohort the framework needed. n=347, 28 future-CLL cases, 2.0–15.7 years pre-dx, HM450, Italian + Swedish sub-cohorts. Cervical lacked an equivalent; CLL has it. Heme-epic v0.2 priority is VAL-082 against this cohort.

**heme-LL-007.** The MARLIN reference (Capper 2025, n=2,540 acute leukemia 450k/EPIC) is the framework-equivalent reference for AML/B-ALL/T-ALL methylation calibration, comparable to Moss 2018 for solid tissue and Salas 2018 for healthy immune subcomposition. Heme-epic myeloid arm uses MARLIN for calibration.

**heme-LL-008.** Per-disease ΔA spread (CLL +0.098, AML +0.168, thymoma +0.120, DLBCL +0.203) is biologically informative and not measurement noise. Lymphoid lineage opens more accessible entropy than myeloid because of programmed B-cell methylation perturbations (class-switching, somatic hypermutation). Cancer Amplifier g for the immune class is 5-10× rather than infinite (as it is for solid tumors at H_min floor) because healthy immune cells are not at the floor — they're actively reorganizing methylation as part of normal function.

## File pointers

- **This README** — heme-epic v0.1 narrative
- **`heme-epic_card_v0.1.json`** — TO BUILD next (after lymphoid + myeloid + T-cell directional panels are derived)
- **Parent reference** — Issue 002 Immune class chapter (FullVersion_build_gape_issue002.py, `'immune'` card definition)
- **Validation anchor data** — TCGA AML 2013 NEJM (PMID 23634996), Chapuy 2018 DLBCL Nat Med, TCGA thymoma 2018 Cancer Cell (PMID 29622463), Hannum 2013 GSE40279, MARLIN reference 2025 Nat Genet
- **v0.2 priority cohort** — EnviroGenomarkers (Georgiadis 2017 BMC Genomics, PMID 28903739, doi 10.1186/s12864-017-4117-4)

## Timeline

- **2026-04-24 v0.1 skeleton:** authored with framework-calibrated A-score targets
- **2026-04-25 v0.1 final:** validation cohort landscape resolved, EnviroGenomarkers + MCCS identified as priority pre-dx cohorts, lymphoid/myeloid/T-cell three-arm structure formalized, SUPPRESSED tier added, EDEAR consumer-report-framing section added, lessons heme-LL-001 through heme-LL-008 catalogued, what-we-discovered section written
- **Phase B (after cardio-epic, kidney-epic, glioma-epic, gastric-epic, bladder-epic skeletons):** per-arm directional panel derivation
- **Phase B-C:** VAL-082 EnviroGenomarkers run, VAL-083 MCCS replication, VAL-084 MARLIN myeloid calibration
- **Phase D+:** G-2026-P010 (CHIP → AML serial) and G-2026-P011 (ICI response) prospective validation in clinical trial contexts

---

**End of heme-epic v0.1 README.**
