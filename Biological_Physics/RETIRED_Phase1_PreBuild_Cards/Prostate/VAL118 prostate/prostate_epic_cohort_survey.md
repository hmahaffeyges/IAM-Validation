# Prostate-EPIC Phase 0 Cohort Survey

**Date:** 2026-04-30
**Card:** prostate-epic v0.2 → preparing for v0.3
**Status:** PHASE 0 LANDSCAPE — game plan pending Heath sign-off

---

## What this document is

A first-pass enumeration of methylation cohorts plausibly relevant to prostate-epic v0.3, surfaced from a literature scan against PubMed, PMC, GEO, EGA, and bioRxiv on 2026-04-30. Built per Guardrail #10 of the Cross-Card Calibration TODO v0.5: *"survey what is actually publicly available for this disease ... before deciding what to score against."*

This is not exhaustive yet. It is enough to make a game-plan decision.

---

## What's already in prostate-epic v0.2 (truth-state baseline)

| Cohort | Anchor VAL | n | Substrate | Tissue | Result |
|---|---|---|---|---|---|
| GSE269244 (Berglund/Yamoah/Kresovich 2024) | VAL-058 | 238 (118 paired tumor/adj-normal) | EPIC 850K | tumor + adjacent-normal | Paired d=+0.497, p=0.0001 |
| GSE119260 (Brikun 2018) urine arm | VAL-065 | 4 patients × 4 specimens = 16 | EPIC 850K | benign tissue + tumor + plasma cfDNA + urine sediment | O5_UNEXPECTED — n=4 advanced-disease ceiling |

v0.2 honest framing: card is `stage_2_only_validated`. Explicitly not a pre-diagnostic blood screening test. Public GEO had no per-patient prostate pre-dx blood methylation cohort as of April 2026 (multi-geography hunt documented).

---

## NEW finds since v0.2 — pre-diagnostic blood candidates

These were NOT in the v0.2 next-validation-steps list and represent the most consequential discoveries of this Phase 0 pass.

### FitzGerald 2017 — Melbourne Collaborative Cohort Study (MCCS)
- **PMID:** 28116812 / **DOI:** 10.1002/pros.23289 (Prostate. 2017;77(5):471-478)
- **Design:** Matched nested case-control, prospective
- **n:** 687 incident prostate cancer cases + matched controls (likely ~687, paper text confirms matched)
- **Substrate:** Illumina HM450K
- **Tissue:** PRE-DIAGNOSTIC peripheral blood
- **Stratification:** by aggressive vs non-aggressive disease, by time-between-blood-draw-and-diagnosis windows
- **Authors:** FitzGerald, Naeem, Makalic, Schmidt, Dowty, Joo, Jung, Bassett, Dugue, Chung, Lonie, Milne, Wong, Hopper, English, Severi, Baglietto, Pedersen, Giles, Southey
- **Affiliations:** Cancer Council Victoria + University of Melbourne + University of Tasmania
- **GEO accession:** **NOT YET CONFIRMED** — needs lookup. Australian/MCCS cohorts typically deposited under Severi or Southey lab GEO accessions.
- **Why it matters:** Directly contradicts v0.2 claim that "public GEO has no usable per-patient prostate pre-diagnostic blood methylation cohort." This is exactly that cohort, n=687, prospective, with stratification by time-to-diagnosis. **This is the single most important finding of Phase 0.**
- **Status:** OPEN — accession lookup needed; cohort access status (open GEO / restricted EGA / dbGaP-gated) unconfirmed.

### MCCS heritable methylation 2023 (Joo et al.)
- **PMID:** 36708485 / **PMC:** PMC10275808
- **Design:** Population-based prospective cohort
- **n:** 869 incident cases + 869 controls (matched on year of birth, year of blood draw, country of birth, sample type) — PLUS 133 individuals from 25 multiple-case prostate cancer families (EPIC array, family-based phase)
- **Substrate:** HM450 (population phase) + EPIC array (family phase)
- **Tissue:** Peripheral blood (prospective collection)
- **Source cohort:** Melbourne Collaborative Cohort Study (MCCS) — same parent cohort as FitzGerald 2017
- **Result:** 41 heritable methylation marks associated with prostate cancer risk; 9 marks near VTRNA2-1 nominally associated with aggressive prostate cancer in population phase
- **GEO accession:** NOT YET CONFIRMED
- **Why it matters:** Larger n than FitzGerald 2017 (1,738 total in population phase). Cross-platform (HM450 + EPIC). Covers heritable signal which complements environmental/inflammatory signal Xu-538 reads.
- **Status:** OPEN — accession lookup needed.

---

## At-diagnosis tissue cohorts (EPIC 850K) — multi-atlas Phase C candidates

### GSE269244 — Berglund/Yamoah/Kresovich 2024 — IN v0.2 (VAL-058 anchor)
- Already in v0.2. African-American men n=238. Paired d=+0.497.

### Howard University AA prostate cohort (PMC9980641)
- **n:** Benign + tumor prostate tissues from African American men (specific n not in abstract)
- **Substrate:** Illumina Infinium 850K EPIC array
- **Tissue:** benign + tumor prostate tissues
- **Findings:** 11,460 differentially methylated probes (p<0.01); AMIGO3, IER3, UPB1, GRM7, TFAP2C, TOX2, PLSCR2, ZNF292, ESR2, MIXL1, BOLL, FGF6 differential
- **GEO accession:** NOT YET CONFIRMED
- **Why it matters:** Second AA prostate EPIC 850K cohort, complementary to GSE269244. Cross-cohort replication candidate.
- **Status:** OPEN — accession lookup needed.

### EGAS00001006670 — Prostate brain metastases (PCBM)
- **EGA accession:** EGAS00001006670 (open EGA listing)
- **n:** 42 PCBM patients with matched primary samples for 17 patients
- **Substrate:** Illumina EPIC array
- **Tissue:** Prostate brain metastases + matched primary
- **Stratification:** SPOP-mutant vs TMPRSS2-ERG-fusion genetic backgrounds
- **Why it matters:** Metastatic disease cohort with matched primary — unique resource for metastatic-progression methylation signal. Distinct from VAL-058's localized tumor-vs-normal comparison.
- **Status:** OPEN access via EGA. Restricted-access typically requires DAC application.

### Tumor methylation landscape (bioRxiv 2025)
- **DOI:** 10.1101/2025.02.07.637178 (bioRxiv, Feb 2025)
- **Title:** "The Landscape of Prostate Tumour Methylation"
- **Why it matters:** Recent (Feb 2025) tumor methylation landscape paper — content needs deeper read; may surface additional cohorts.
- **Status:** Citation captured; full paper read pending.

---

## Other tissue / blood / urine candidates surfaced

### TCGA-PRAD (TCGA prostate cancer)
- **n:** 497 prostate tumors + 50 normal prostate tissue (HM450)
- **Substrate:** TCGA HM450K Level 3 (sesame normalization standard)
- **Access:** Open via NIH GDC public portal
- **Why it matters:** Largest single PRAD methylation cohort. Substrate matches cardio-epic's calibration cohort (TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 from VAL-106). PRAD adjacent-normal is already partially used as calibration cohort, which means structural-separation discipline applies (same caveat as kidney-epic per CCL-041: tumor vs adjacent-normal are different samples even within the same TCGA project).
- **Status:** Available for Phase C re-scoring.

### GSE87571 — White blood cell methylation reference
- **n:** 421 individuals, 732 samples, ages 14-94
- **Substrate:** HM450
- **Tissue:** White blood cells
- **Use:** Healthy WBC reference for cfDNA background simulation (per Mdpi 2024 OBBPA-ddPCR paper)
- **Why it matters:** Possible prostate-epic Stage 1 substrate baseline if EPIC 850K Hannum-equivalent is needed. Need to confirm substrate match.
- **Status:** Available; substrate likely HM450 not EPIC.

### Iran SPDEF cg11346722 cohort (PMC12128380)
- **n:** 360 men (180 PCa + 180 BPH controls)
- **Substrate:** Whole blood leukocyte DNA, single-CpG biomarker (NOT array-wide)
- **Tissue:** Whole blood
- **Recruited:** January 2020 - December 2024, Tehran
- **Why it matters:** Single-CpG biomarker only — NOT useful for Xu-538 panel scoring. Cited for completeness; not a Phase C candidate.
- **Status:** Single-CpG only, NOT array-wide. Defer.

### Howard University AA blood methylation (PMC6133349)
- **n:** Not in abstract; recruited from HU Hospital + free PCa screening program
- **Substrate:** Blood-based methylation, array unspecified in abstract
- **Tissue:** Whole blood
- **Why it matters:** Blood-based AA prostate methylation cohort. Substrate + n need confirmation.
- **Status:** OPEN — paper read needed for full specs.

### UK 13-gene pyrosequencing prognostic cohort (PMC4162944)
- **n:** Not in abstract; FFPE TURP from 6 cancer registries 1990-1996
- **Substrate:** Pyrosequencing (NOT array)
- **Tissue:** FFPE TURP tissue
- **Genes:** GSTP1, APC, RARB, CCND2, SLIT2, SFN, SERPINB5, MAL, DPYS, TIG1, HIN1, PDLIM4, HSPB1
- **Why it matters:** Pyrosequencing — not array-compatible. Cited for completeness.
- **Status:** Pyrosequencing-only. NOT a Phase C candidate.

### GSE-unspecified — Indolent vs aggressive PCa whole-genome enzymatic methylation sequencing (PMC12772133)
- **n:** 120 patients
- **Substrate:** Whole-genome enzymatic methylation sequencing (NOT array — sequencing-based)
- **Findings:** 14-region DNA methylation signature, prognostic for indolent vs aggressive
- **Why it matters:** Sequencing-based; would need bridge engineering analogous to Tanaka 2025 nanopore→array. Defer to v0.4+.
- **Status:** WGEMS substrate. Bridge engineering needed.

### Peripheral T-cell methylation EPIC array (PMC7310561)
- **n:** Not in abstract; men with positive biopsy for PCa vs men with negative biopsy (BPH controls)
- **Substrate:** Illumina EPIC array (T-cell DNA isolated)
- **Tissue:** Peripheral T cells (sorted)
- **Findings:** 449 differentially methylated CGs correlate with Gleason; 3 CGs validated as polygenic methylation score (PMS) with strong Gleason + tumor volume correlation
- **Why it matters:** Sorted T-cell methylation — different specimen than Xu-538's whole-blood substrate, but EPIC array compatible. Same group has published liver and breast peripheral T-cell methylation papers (Toronto/Ottawa).
- **Status:** OPEN — accession + n lookup needed.

### TCGA 73 PCa + 63 benign HM450 (PMC5392915, referenced in Selmani 2025)
- **n:** 73 PCa + 63 benign-adjacent prostate tissues
- **Substrate:** Illumina HM450 BeadChip
- **Tissue:** Fresh-frozen prostate tissue
- **Findings:** 564 DMGs, ZNF154 top candidate (PFS-associated CpGs)
- **GEO accession:** NOT YET CONFIRMED
- **Why it matters:** Cross-platform replication candidate against TCGA-PRAD (HM450 substrate). Likely overlaps with TCGA-PRAD subset.
- **Status:** OPEN — likely TCGA subset; needs verification.

---

## Categories NOT yet surveyed in this pass (need follow-up before sprint begins)

These are gaps the next chat should close before committing to Phase A:

1. **dbGaP cohorts.** Health ABC (referenced in v0.2 README), Rotterdam Study, MrOS (Osteoporotic Fractures in Men), ARIC prostate sub-cohort, PHS (Physicians' Health Study), BLSA (Baltimore Longitudinal Study of Aging) prostate, AAPC consortium. Each needs DUA timeline + current methylation availability.
2. **Movember Foundation cohort catalog.** Referenced in v0.2 next-validation-steps. Needs catalog scan.
3. **Active surveillance cohorts.** Johns Hopkins, UCSF, Sunnybrook — collaboration paths via L1 lab partnership tier.
4. **PRACTICAL consortium.** European ancestry validation. Referenced in cookbook strategy but not surveyed here.
5. **UK Biobank prostate sub-cohort.** Restricted access via UKB application.
6. **EPIC-Italy prostate.** Italian arm of European Prospective Investigation into Cancer and Nutrition — referenced as breast/CRC anchor in cookbook (GSE51057 / GSE51032 are EPIC-Italy breast); prostate sub-cohort may exist.
7. **Asian ancestry cohorts.** Korean and Japanese prostate methylation cohorts beyond what was found.
8. **Urine methylation beyond Brikun 2018.** PCA3, SelectMDx, ConfirmMDx, UroMark validation cohorts. The v0.2 README named these as candidate paths; need specific accession lookups.
9. **FinnGen + Million Veterans Program.** Mentioned in cookbook strategy as consortium repositories; prostate methylation availability not surveyed.
10. **Cross-platform: TCGA-PRAD subset published cohorts on HM450.** Multiple papers (PMC5392915, OBBPA-ddPCR Mdpi 2024) reference TCGA-PRAD HM450 — overlap analysis needed before scoring.

---

## Substrate inventory (CCL-040 + CCL-041 implication)

Cohorts surfaced span at least four substrates:
| Substrate | Cohorts |
|---|---|
| **EPIC 850K** | GSE269244 (VAL-058), GSE119260 (VAL-065), Howard AA EPIC, EGAS00001006670 PCBM, MCCS family-phase, T-cell EPIC |
| **HM450** | FitzGerald 2017, MCCS population-phase, TCGA-PRAD, GSE87571 WBC reference, PMC5392915 |
| **Pyrosequencing** | UK 13-gene FFPE prognostic |
| **Whole-genome enzymatic methylation sequencing (WGEMS)** | PMC12772133 indolent-vs-aggressive |

**Implication:** Prostate-epic v0.3 will encounter at minimum the EPIC 850K + HM450 substrates simultaneously. This means substrate-specific CHK-3.1A self-cal envelopes (per Guardrail #13's CCL-040 + CCL-041 + DISC-CARDIO-005) are mandatory, and we'll need to address the HM450↔EPIC bridge for cross-platform comparisons (some CpGs differ between platforms; Xu-538 panel coverage on EPIC v1 is ~94.8% per VAL-058 footnote). Pyrosequencing and WGEMS substrates are deferred to v0.4+ unless prioritized.

---

## Game-plan options (decision points for Heath)

The shape of v0.3 depends on three orthogonal choices.

### Choice 1 — Sprint scope

- **Option A (lean):** Focus v0.3 on multi-atlas Phase C re-scoring of GSE269244 (the existing VAL-058 cohort) under run-everything discipline. No new cohorts. Rebuild card with v0.5 TODO structural blocks (atlases_used_and_deferred, chk_3_1_thresholds_per_substrate, DISC-PROSTATE-NNN, per-disease scoring policy, validation evidence summary, cookbook-wide CCL cross-references). 4-6 hours of execution.
- **Option B (medium):** A + add TCGA-PRAD as a second tissue cohort for cross-cohort replication. Substrate diverges (HM450 vs EPIC 850K) — cardio-epic-style multi-substrate handling required. Tests whether VAL-058's d=+0.497 replicates on a different cohort, different ancestry mix, different platform. 6-9 hours.
- **Option C (full):** A + B + Phase 0 closure pass on dbGaP cohorts and consortium catalogs, then attempt FitzGerald 2017 MCCS access (the pre-diagnostic blood cohort). If accession is open GEO, this could be the first per-patient prostate pre-dx blood VAL in the cookbook. If restricted, document access path and defer. 10-15 hours plus access timeline.

### Choice 2 — Priority specimen direction

The clinical use case for your wife's uncle is post-treatment monitoring (trajectory question), not early detection (urine question). This points option B/C toward **blood plasma trajectory** rather than urine. Confirming this priority changes which cohorts to prioritize:
- **Blood plasma trajectory priority** → focus on FitzGerald MCCS pre-dx blood + TCGA-PRAD tissue + at-diagnosis blood cohorts. Skip urine expansion this sprint.
- **Urine early-detection priority** → search for larger urine cohorts beyond Brikun 2018, including L1 lab partnership tier collection planning.
- **Both** → option C scope.

### Choice 3 — How to handle the access-gated cohorts

FitzGerald 2017 MCCS could be a major addition but accession status is unconfirmed. Three paths:
- **Path 1:** Try GEO first for open access. If found → score in v0.3.
- **Path 2:** If restricted → submit DUA application during v0.3 sprint; document the application as part of v0.3 next_validation_steps; ship v0.3 without it.
- **Path 3:** Direct outreach to FitzGerald, Severi, or Southey labs (Cancer Council Victoria / University of Melbourne). Standard data-sharing email. Timeline 4-12 weeks.

---

## Recommended next moves (Walther's read, if useful)

If your wife's uncle's situation is the personal motivation, **Option B with blood-plasma priority** is the right balance: it ships v0.3 with the cookbook-discipline upgrade you wanted, adds genuinely new validation (cross-cohort replication on TCGA-PRAD), and surfaces the access-gated FitzGerald MCCS as the documented Path 2 next-step for v0.4. It does NOT promise something the framework cannot defensibly do today, and it does upgrade what the framework CAN do for post-treatment monitoring.

That said, this is your call. The Phase 0 landscape says prostate has more public data than v0.2 acknowledged — FitzGerald 2017 alone is potentially the cohort that changes the card's tier ceiling from `stage_2_only_validated` to something closer to `cohort_screening_validated` if access opens up. That changes the calculus.

What scope and priority do you want, and how should we handle FitzGerald 2017?
