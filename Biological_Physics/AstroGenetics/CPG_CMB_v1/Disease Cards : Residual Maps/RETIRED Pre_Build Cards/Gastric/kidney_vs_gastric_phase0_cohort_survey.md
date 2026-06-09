# Phase 0 Cohort Survey — Kidney vs Gastric for Next EDEAR Card

**Date:** 2026-05-02
**Status:** Awaiting Heath sign-off on card selection + cohort/atlas decisions
**Format:** Mirrors bladder-epic_cohort_survey.md (2026-04-30 sign-off precedent)
**Deliverable:** Pick one cancer (kidney OR gastric), sign off on Phase B calibration cohort + Phase C disease cohort + atlas family + 4 priority-ordered cards before VAL prereg drafting begins.

---

## Executive verdict

**Kidney is the cleaner Phase 0 case.** Gastric is genuinely new territory but blocked at three layers (no public 450K paired cohort >2 normals, no off-the-shelf gene-promoter atlas, no Xu-538-coverage-validated panel for the cohort). Building a fresh StomachRef from Karagiannis 2024 scRNA-seq is itself a multi-week sprint that becomes a card-blocker before disease scoring can begin.

Kidney has all four ingredients ready: structurally-separated paired cohorts (TCGA-KIRP n=45 + TCGA-KICH separately), an off-the-shelf EpiSCORE KidneyRef in the vault (4 cell types EC/Epi/Fib/IC, only 32 marker genes — itself a real test of DISC-BLADDER-001), strong clinical-grade comparators (SHOX2 cfDNA, cfMeDIP-seq plasma AUROC 0.99 / urine 0.86), and multiple Tier 1 pre-diagnostic blood cohorts (PLCO + ATBC).

The recommendation is kidney. Open questions for sign-off below.

---

## Side-by-side at a glance

| Dimension | Kidney (RCC) | Gastric (STAD) |
|---|---|---|
| **Calibration overlap problem?** | YES — TCGA-KIRC adjacent-normal n=160 is part of VAL-106/112/117/119 calibration cohort. Sample-level separation OK; cohort-level overlap is real. **Mitigation:** TCGA-KIRP (n=45 paired) + TCGA-KICH provide structurally separated cohorts. | NO — TCGA-STAD not on prior calibration audit list. |
| **TCGA paired HM450 cohort** | KIRC n=158 paired + KIRP n=45 paired + KICH separately. Total n>200 paired across 3 histologic subtypes. | **STAD only n=2 paired HM450 samples.** Most papers combine HM27 + HM450 (48 normal HM27 + 2 normal HM450 = 50 total normals, mixed platforms). |
| **EpiSCORE atlas in vault?** | YES — KidneyRef__Kidney_Mref_m.csv (sha 3cff72c4...) + Expref (sha 3c3df966...). 4 cell types EC/Epi/Fib/IC. **Only 32 marker genes** (BladderRef has 163). | NO — StomachRef absent from Zhu/Teschendorff 2022 13-tissue distribution. To build fresh requires Karagiannis 2024 stomach scRNA-seq atlas (parietal/chief/mucous/enteroendocrine/mitotic/endothelial/fibroblast/macrophage/neutrophil/T-cell/plasma) + multi-week EpiSCORE imputation training. |
| **Loyfer bulk tile** | "Kidney" tile present (calibrated VAL-112). | Only "Upper_GI" tile (esophagus + stomach + duodenum together — coarse, not stomach-specific). |
| **Pre-diagnostic blood 450K** | PLCO (n=215 cases / 436 controls) + ATBC (n=191 cases / 575 controls). LINE-1 pyrosequencing baseline; 450K access via dbGaP-tier. | Shanghai Women's Health Study (Yang 2012, n=192 cases / 384 controls) — Alu/LINE-1 pyrosequencing only, NOT 450K. |
| **Clinical-grade comparators** | SHOX2 cfDNA (Jung 2019 prospective), cfMeDIP-seq plasma AUROC 0.99 + urine 0.86 (Nuzzo 2020 Nature Medicine), 23-DMCGI model AUC 0.974 stage I (BMC Cancer 2025), MEMORY Study NCT05917106 ctDNA methylomics (recruiting). | Plasma multi-gene panels SEPT9 + SDC2 + RUNX3 + RNF180 (qMSP/MSP, 50-76% sensitivity / 85-98% specificity per 2018 meta-analysis n=4172); GSE30601 n=297 HM27 (coarser); MethylCap-seq Korean cohort n=28 paired (Cho 2012, 16 normal + 28 tumor + 12 metastatic LN). |
| **Newest atlas (future)** | Cross-species kidney scMethyl-Hi-C atlas (bioRxiv Jan 2026), 24-cell-type snATAC chromatin atlas (Nature Comm 2024) — both v0.X+ promotion paths once methodology established. | Karagiannis 2024 BMC Biology stomach scRNA-seq atlas (could seed StomachRef build). H. pylori chronic-inflammation methylation field-effect literature. EBV-positive subtype distinct methylation. |
| **Subtype/stratification stress test** | 3 histologic subtypes (clear cell ccRCC ~70% / papillary pRCC / chromophobe chRCC). Type 2 PRCC further stratified into 3 multi-omics subgroups (Linehan 2016 NEJM). | Lauren classification (intestinal vs diffuse). TCGA molecular subtypes (Cristescu 2015 / TCGA 2014: CIN ~50% / MSI ~22% / EBV+ ~9% / GS ~20%). |
| **Cell-of-origin biology challenge** | Multiple kidney cell types (proximal tubule S1/S2/S3, podocyte, distal tubule, collecting duct, loop of Henle, intercalated cells α/β) — 24 distinct cell types per snATAC atlas. KidneyRef collapses to 4 EC/Epi/Fib/IC. | Multiple gastric mucosa lineages (chief cells, parietal cells, mucous neck, surface mucous, enteroendocrine, pit/gland regions, mitotic stem cells). No established 4-tile gene-promoter reference. |
| **Pan-urologic precedent?** | YES — Ricketts 2018 Nat Comm, n=1952 across BLCA + KICH + KIRC + KIRP + PRAD + TGCT, HM450/HM27 batch-corrected. Cross-tile sanity of bladder-epic v0.2 BladderRef can be reapplied. | NO — STAD analyzed in pan-cancer methylation papers but not in a urologic/parallel-organ panel matching prostate+bladder+kidney structurally. |
| **Phase 0 readiness** | All 4 ingredients present: cohorts + atlas + panel + clinical comparators. | Blocked at ≥2 layers: no off-the-shelf StomachRef, no >n=2 paired HM450 cohort. |

---

## Kidney — exhaustive Phase 0 inventory

### Phase B calibration cohort

**Recommended:** TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (the same VAL-106 calibration cohort that anchored cardio + prostate + bladder).

**Calibration-overlap caveat:** KIRC adjacent-normal is structurally healthy substrate (the same role it played for cardio/prostate/bladder calibrations) — calibration is per-sample and operates on healthy tissue distributions, not on the disease-vs-healthy contrast. The VAL-106 calibration cohort is sealed and reused across cards by design. The boundary that matters is: do not score TCGA-KIRC tumor against TCGA-KIRC adjacent-normal as the disease-vs-healthy paired contrast in VAL-12X (because KIRC adjacent-normal is part of the healthy reference). Solution below.

**Confirmed:** KidneyRef Phase B calibrates on the same VAL-106 cohort that anchored ProstateRef (VAL-117) and BladderRef (VAL-119). Pattern match. Same CHK-3.1A solid-parenchyma floor (≥0.50 / ≤0.12). No tissue-class amendment needed.

### Phase C disease cohort

**Recommended primary:** TCGA-KIRP n=45 paired (papillary RCC). Structurally separated from VAL-106 calibration cohort. Multi-subtype (type 1 vs type 2) for stratified analysis.

**Recommended secondary:** TCGA-KICH (chromophobe RCC) — separate histology, separate subtype calibration check.

**Recommended tertiary:** GSE52955 multi-cancer urological cohort — n=72 HM450 across BLCA + PRAD + KIDNEY for cross-tile sanity (specificity check for the kidney card vs bladder + prostate fired separately).

**NOT RECOMMENDED for primary v0.1:** TCGA-KIRC (clear cell RCC). KIRC is the largest cohort (n=158 paired) but calibration-overlap caveat applies. **Reserve KIRC for v0.2 follow-up Phase C**, after v0.1 establishes the kidney signature on KIRP.

### Atlas family in atlases_run

1. **EpiSCORE KidneyRef** (PRIMARY per CHK-2.18-equivalent for solid-parenchyma — but kidney is solid parenchyma, so CHK-2.18 mucosal-cohort rule does NOT apply; CHK-2.18 reads "tissue class ∈ mucosal organs"). KidneyRef goes in atlases_run as the gene-promoter sub-cell-type reader.
2. **Layered Moss+Loyfer 25-tile** — Kidney bulk tile already calibrated VAL-112. For solid-parenchyma cohorts, bulk-WGBS atlases ARE useful primary readers (the substrate-distribution-mismatch artifact in DISC-BLADDER-003 is a mucosal-cohort phenomenon — solid kidney cortex/medulla matches Loyfer Kidney bulk reference reasonably well). This is itself a real test of CHK-2.18 boundary.
3. **Caggiano CelFiE TIM 19-tile** — already calibrated VAL-113.
4. **Salas IDOL Stage 3** — production-calibrated.
5. **UniLIFE Guo 2025 Stage 3** — within-cohort self-cal v0.1.

### Stage 1 panel

**Question for you:** Xu-538 panel coverage on TCGA-KIRP HM450 has not been verified. If we run the same CHK-2.17 cohort-substrate-coverage gate that bladder caught at 51.1% on Xu-538/BLCA, kidney could pass cleanly OR fire the same DISC-BLADDER-004 gate. **Recommended pre-flight check:** sample 5-10 KIRP β files first, compute per-sample Xu-538 coverage, FLAG if mean < 90% or q5 < 80% before drafting VAL-12X prereg.

### Pre-diagnostic blood (v1.0+ promotion path)

- **PLCO + ATBC** — LINE-1 pyrosequencing precedent (Karami 2015, n=215 + n=191 cases). 450K-tier access via dbGaP application; Tier 2-tier biobank access (faster than WHI Tier 3). This is a real Tier 1+ promotion path for kidney.

### Clinical-grade comparators (CHK-1.4 leverage)

- SHOX2 cfDNA methylation (Jung 2019 prospective n=100 testing): HR 1.50 [1.29-1.74] for risk of death after nephrectomy
- cfMeDIP-seq plasma RCC (Nuzzo 2020 Nature Medicine): AUROC 0.99 plasma + 0.86 urine
- 23-DMCGI high-throughput methylation model (BMC Cancer 2025): AUC 0.974 stage I
- MEMORY Study (NCT05917106): ctDNA methylomics monitoring, recruiting

### Real DISC-BLADDER-001 stress test

**KidneyRef has only 32 marker genes** (BladderRef 163, ProstateRef 150-tier). The bladder finding said cell-type distinctness, not count, drives fitness. KidneyRef tests the lower bound: does a 32-marker gene-promoter atlas with 4 cell types still produce within-cohort tile range above 0.02? If yes, DISC-BLADDER-001 holds at the marker-count lower bound. If no, we discover the marker-count floor. Either outcome is a clean cookbook discovery (DISC-KIDNEY-001 candidate).

### Card v0.1 sprint structure (patient-flow per bladder LL-005 precedent)

- VAL-12X Phase B: KidneyRef Phase B mini-calibration on VAL-106 cohort n=210. Pre-flight Xu-538 coverage check on KIRP first.
- VAL-12X+1 Stage 1: Xu-538 (or freshly-validated panel) on TCGA-KIRP n=45 paired.
- VAL-12X+2 Stage 2: KidneyRef + Layered Moss+Loyfer + Caggiano TIM run-everything.
- VAL-12X+3 Stage 3: Salas IDOL + UniLIFE + Caggiano TIM immune subset.

---

## Gastric — exhaustive Phase 0 inventory

### Phase B calibration cohort

**No StomachRef-equivalent off-the-shelf atlas exists.** Karagiannis 2024 BMC Biology stomach scRNA-seq atlas could seed a fresh StomachRef build using the EpiSCORE imputation methodology. That's a separate multi-week R script + EpiSCORE training + cross-validation sprint that itself becomes a card-blocker before disease scoring can begin. We've never built one of these from scratch — all four EpiSCORE atlases in our vault (HeartRef, ProstateRef, BladderRef, plus the Zhu 2022 distribution) came pre-built from the EpiSCORE 13-tissue distribution.

If we proceed without StomachRef, the only Stage 2 atlas family available is Loyfer's Upper_GI bulk tile (esophagus + stomach + duodenum together, coarse) — and per DISC-BLADDER-003 + CHK-2.18, gastric mucosa is mucosal tissue class, so bulk-WGBS atlases are triangulation-only and we'd have no primary cell-of-origin reader. The card cannot ship cleanly without a gene-promoter atlas.

### Phase C disease cohort

**TCGA-STAD only n=2 paired tumor-vs-adjacent-normal HM450 samples.** The Tian 2023 paper combined HM27 + HM450 platforms to get n=27 nontumor, but cross-platform combination is the exact CCL-037 antipattern that lung-epic v0.5 + crc-epic v2.4.1 explicitly forbid for retrospective Cookbook validation. Single-platform HM450 paired contrast is structurally impossible from TCGA-STAD alone.

GEO supplements: GSE72872 n=125 (HM450 GPL13534), GSE81334 n=23 (HM450) — both small. The larger gastric cohorts (GSE30601 n=297, GSE25869 n=74) are HM27 platform — coarser CpG coverage, no Xu-538 panel coverage, no compatibility with our Stage 1/2/3 atlases.

### Atlas family in atlases_run

Without a StomachRef bridge, atlases_run for gastric Stage 2 would be: Loyfer Upper_GI bulk tile (triangulation-only per CHK-2.18) + Caggiano TIM (general immune+stromal). No primary cell-of-origin reader. Card cannot ship.

### Stage 1 panel

Xu-538 panel coverage on TCGA-STAD HM450 not verified. Same pre-flight check needed as kidney would require.

### Pre-diagnostic blood

Shanghai Women's Health Study (Yang 2012, n=192 + n=384) — Alu/LINE-1 pyrosequencing only, NOT 450K. The seminal pre-diagnostic gastric methylation prospective study used pyrosequencing on global hypomethylation markers, not CpG-panel arrays. Tier 1+ 450K-tier access does not exist in the public domain for gastric pre-diagnostic blood.

### Clinical-grade comparators

Multi-gene plasma panels SEPT9 + SDC2 + RUNX3 + RNF180 well-characterized, but operating at 50-76% sensitivity / 85-98% specificity per 2018 meta-analysis (Diaz-Lagares et al, n=4172). The clinical-grade gastric methylation comparator landscape is qMSP-based plasma, not array-based. CHK-1.4 leverage value lower than kidney's cfMeDIP-seq comparators.

### What gastric brings that kidney does not

- New territory (no calibration overlap)
- 4 distinct molecular subtypes (CIN/MSI/EBV+/GS) that could surface stratification differences in a way kidney histology subtypes don't
- H. pylori chronic-inflammation methylation field-effect — would test the "infection-driven mucosal methylation drift" branch of cookbook biology that no prior card has touched
- EBV-positive gastric subtype is a methylation-specific phenotype (high-CIMP methylation epigenotype) — could be a methylation-card bull's-eye for that subtype specifically
- Multiple gastric cell lineages (chief, parietal, mucous neck, surface mucous, enteroendocrine) — a real test of DISC-BLADDER-001 if a StomachRef were built

### Why we cannot proceed to gastric VAL prereg now

1. **No off-the-shelf StomachRef.** Building one is a multi-week sprint, not a card sprint.
2. **No public 450K-only paired cohort >n=2.** Cross-platform combination (HM27 + HM450) violates CCL-037.
3. **No 450K-tier pre-diagnostic blood cohort.** SWHS pyrosequencing is not array-tier.
4. **CHK-1.4 leverage value is qMSP-based, not array-based.**

Gastric is a real future card — when StomachRef is built and a fresh 450K paired gastric cohort exists, we have all the biology richness already mapped. But not now.

---

## Recommendation: kidney

The kidney sprint is built on top of bladder's structural learnings without spending sprint time on bridge engineering.

### Sign-off questions for Heath

**Q1. Card selection: kidney or gastric?**

Recommend kidney for next card. Gastric blocked at ≥2 layers; kidney has all 4 ingredients ready.

**Q2. Phase B calibration approach:**

(a) Calibrate KidneyRef on VAL-106 cohort (TCGA-KIRC + TCGA-PRAD adjacent-normal n=210) as the standard solid-parenchyma calibration. Score TCGA-KIRP n=45 paired as Phase C disease cohort (structurally separated from KIRC). Document the KIRC-overlap-caveat under DISC-CARDIO-005 within-cohort self-cal documentation rule. **Recommended.**

(b) Build a fresh kidney calibration cohort outside TCGA (e.g., Stanford collection used in Wei 2014 BMC Medicine n=96 RCC paired) — would require new biobank application. Adds weeks.

(c) Use KIRC adjacent-normal n=160 only as both calibration and Phase C cohort — would violate CCL-040 calibration-before-scoring discipline. Not recommended.

**Q3. Phase C cohort selection (assuming option-2a):**

Primary: TCGA-KIRP n=45 paired (papillary RCC, structurally separated)
Secondary: TCGA-KICH (chromophobe RCC, separate subtype)
Tertiary: GSE52955 multi-cancer urological (n=72 HM450 across BLCA + PRAD + KIDNEY) — cross-tile sanity / specificity check

Or do we punt the secondary/tertiary to v0.2 and ship v0.1 with KIRP only?

**Q4. EpiSCORE KidneyRef Phase B smoke test on VAL-106 cohort (acknowledging KIRC-overlap caveat):**

The VAL-106 cohort is a HEALTHY-substrate calibration. KIRC adjacent-normal contributes to that healthy reference — same role for cardio + prostate + bladder. Calibration is per-sample and operates on healthy tissue distributions, not on the disease contrast. Sealed against the v0.7 calibration TODO Phase B requirement. No new procedure, just documented under DISC-CARDIO-005.

**Q5. Pre-flight Xu-538 panel coverage check on TCGA-KIRP before drafting VAL-12X prereg?**

Recommended. Sample 5-10 KIRP β files, compute per-sample panel coverage, FLAG if mean < 90% or q5 < 80%. If panel passes cleanly → Stage 1 v0.1 production-tier deployment possible (kidney becomes the first solid-parenchyma card with Stage 1 production-validated). If panel fails → DISC-BLADDER-004 reapplied to kidney; v0.1 ships with diagnostic-only Stage 1 + Wave 1 promotion path same as bladder.

**Q6. The DISC-BLADDER-001 lower-bound test:**

KidneyRef has 32 marker genes (vs BladderRef 163, ProstateRef 150). Bladder finding said cell-type distinctness, not count, drives fitness. Kidney tests the marker-count lower bound. Pre-locked outcome:
- O1: max within-cohort tile range ≥ 0.02 → DISC-BLADDER-001 holds at marker-count lower bound (extends rule)
- O3: max within-cohort tile range < 0.02 → marker-count floor discovered → new DISC-KIDNEY-001 (refines DISC-BLADDER-001 with marker-count modifier)

Either outcome is a clean cookbook discovery.

---

## Open queue items deferred from this Phase 0

- **Gastric card** queues for the future StomachRef-build sprint or for the future 450K-only paired gastric cohort if one becomes public.
- **WHI bladder pre-dx biobank application** still queued for v1.0+ bladder promotion path.
- **PLCO + ATBC kidney pre-dx biobank applications** queue for v1.0+ kidney promotion path.

---

**Awaiting Heath sign-off on Q1-Q6 before VAL prereg drafting begins.**
