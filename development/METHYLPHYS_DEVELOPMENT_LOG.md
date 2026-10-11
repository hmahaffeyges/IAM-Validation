# Met-A, IAM-A and C-score development log

> **The one LOG (since 2026-10-09).** Every result and every chain change, dated. Chain changes are tagged **CHAIN CHANGE**. Plan and commissioning record: [`STATUS.md`](../Biological_Physics/MethylPhys/STATUS.md). Chain changes before 2026-10-09: [archived chain changelog](../Biological_Physics/MethylPhys/archive/docs_consolidated_2026-10-09/CHAIN_CHANGELOG.md).

Status: **open** (development stage). Every development finding of this project is logged here, whether it passes, fails or is
inconclusive, with a link to the full record. Nothing here is a commissioned result; results become book material only after
the chain passes commissioning.

Scope: the methylation chain (chain v3): Met-A, IAM-A and both C-scores, on arrays and single molecules, every cell type and specimen kind.

## How an entry is written

Date - the question - what was run (data, chain version) - the outcome in one or two lines - link to the full record. Entries are
appended, never edited after the fact; a later finding that changes an earlier one is a new entry that links back to it.

## Findings moved from the book (2026-10-04)

Development readings that were in the book until 2026-10-04 were moved here, unchanged, so the book carries only the physics,
the method and commissioned results. Their full text, as it stood in the book, is kept at:
- [First readings chapter](archive/p4_21_firstreadings.tex): neutrophil tests, infection bloods, leukaemia and remission, DNMT1-inhibitor series on arrays and single molecules.
- The removed passages from the separation, atlas, instrument, serial, astrogenetics, leukocyte, reach, discipline and status chapters: see the commit "Part VI: development readings moved to development/" in the repo history (`git log -- development`).

## Records

| Date | Record | Title |
|---|---|---|
| 2026-09-22 | [PROC_E2E_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_E2E_01_PREREG.md) | PROC-E2E-01 — the commissioned chain, end to end, against the test package's own documented outputs |
| 2026-09-22 | [PROC_STAGE0_02_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_STAGE0_02_OUTCOME.md) | PROC-STAGE0-02 — outcome: Stage 0 run retrospectively over the Uppsala cohort |
| 2026-09-22 | [PROC_STAGE0_02_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_STAGE0_02_PREREG.md) | PROC-STAGE0-02 — Stage 0 intake, run retrospectively over the Uppsala cohort |
| 2026-09-22 | [PROC_STAGE0_04_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_STAGE0_04_PREREG.md) | PROC-STAGE0-04 — the bisulfite-conversion threshold, to be set by the author |
| 2026-09-25 | [ATLAS_READABILITY.md](../Biological_Physics/MethylPhys/doors/ATLAS_READABILITY.md) | What this instrument can and cannot read — measured from the atlas, 2026-09-26 |
| 2026-09-25 | [PROC_BAND_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_BAND_01_PREREG.md) | PROC-BAND-01 — can progenitor and stem_adult carry a commissioned healthy band in whole blood? |
| 2026-09-25 | [PROC_CLS_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_CLS_01_PREREG.md) | PROC-CLS-01 — does the residual sky have scale structure worth reporting? |
| 2026-09-25 | [PROC_EPIC_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_EPIC_01_PREREG.md) | PROC-EPIC-01 — does the commissioned chain see a pre-diagnostic immune signal in genuinely held-out EPIC-Italy blood? |
| 2026-09-25 | [PROC_FOREIGN_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_FOREIGN_01_PREREG.md) | PROC-FOREIGN-01 — should the immune tier be withheld when a specimen carries material the gauge was not built for? |
| 2026-09-25 | [PROC_LABBAND_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_LABBAND_01_PREREG.md) | PROC-LABBAND-01 — should each laboratory be judged against its own width? |
| 2026-09-25 | [PROC_PARTIAL_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_PARTIAL_01_PREREG.md) | PROC-PARTIAL-01 — can a non-blood class's fidelity score be recovered from an ordinary blood draw? |
| 2026-09-26 | [DECONVOLVER_REPAIR_2026-09-26.md](../Biological_Physics/MethylPhys/doors/DECONVOLVER_REPAIR_2026-09-26.md) | The deconvolver repair of 2026-09-26 — why Breast read zero in breast tissue, and what was changed |
| 2026-09-26 | [PROC_MF_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_MF_01_PREREG.md) | PROC-MF-01 — pre-registration: does a covariance-weighted matched filter lower the minimum detectable fraction of a fore |
| 2026-09-26 | [PROC_MF_02_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_MF_02_PREREG.md) | PROC-MF-02 — pre-registration: inverse-variance weighted detection of a foreign cell in blood, as a stage of the chain |
| 2026-09-26 | [PROC_MF_03_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_MF_03_PREREG.md) | PROC-MF-03 — pre-registration: the inverse-variance detector with a per-laboratory threshold, tested on a fifth laborato |
| 2026-09-26 | [PROC_UNMIX_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_UNMIX_01_PREREG.md) | PROC-UNMIX-01 — pre-registration: does inverting the dilution line put every present cell's A at 1.00 on constructed tru |
| 2026-09-27 | [PHYSICS_LEUKOCYTE_GAUGE.md](../Biological_Physics/MethylPhys/doors/PHYSICS_LEUKOCYTE_GAUGE.md) | What moves a leukocyte on the gauge — the two ends are different physics (2026-09-27) |
| 2026-09-27 | [PROC_FOREIGNSCORE_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_FOREIGNSCORE_01_PREREG.md) | PROC-FOREIGNSCORE-01 — pre-registration: the scoring floor for a detected foreign cell |
| 2026-09-27 | [PROC_INTAKE_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_INTAKE_01_PREREG.md) | PROC-INTAKE-01 — pre-registration: the intake gate runs on the array's own numbers, and a deferred check never advances |
| 2026-09-27 | [PROC_SERIAL_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_SERIAL_01_PREREG.md) | PROC-SERIAL-01 — pre-registration: serial mode — one person, two or more draws |
| 2026-09-27 | [PROC_SKY_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_SKY_01_PREREG.md) | PROC-SKY-01 — pre-registration: the sky's zero and spread with no population in them |
| 2026-09-27 | [PROC_STAGE2D_02_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_STAGE2D_02_OUTCOME.md) | PROC-STAGE2D-02 — outcome: NOT ADOPTED. The detector's design, not its lines, was the defect. |
| 2026-09-27 | [PROC_TARE_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_TARE_01_OUTCOME.md) | PROC-TARE-01 — outcome: NOT COMMISSIONED. The array's SNP probes see a real compression, and it carries almost no inform |
| 2026-09-28 | [ATLAS_V2_SPEC.md](../Biological_Physics/MethylPhys/doors/ATLAS_V2_SPEC.md) | Atlas v2 — specification (PLAN item 20), written 2026-09-27 before any machine is rented |
| 2026-09-28 | [PROC_OUTSPAN_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_OUTSPAN_01_PREREG.md) | PROC-OUTSPAN-01 — pre-registration: the clean map of what the atlas cannot explain |
| 2026-09-28 | [PROC_PARTIALCOV_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_PARTIALCOV_01_PREREG.md) | PROC-PARTIALCOV-01 — pre-registration: entering a cell measured on part of the array |
| 2026-09-30 | [PROC_CLASS_COUNT_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_CLASS_COUNT_01_PREREG.md) | PROC-CLASS-COUNT-01 — pre-registration: how many entropy levels do cells form, measured without any floor |
| 2026-09-30 | [PROC_DECONV_V2_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_DECONV_V2_01_OUTCOME.md) | PROC-DECONV-V2-01 — outcome (2026-09-29): A5 FAILED on CD4 and CD8; NK passes |
| 2026-09-30 | [PROC_DECONV_V2_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_DECONV_V2_01_PREREG.md) | PROC-DECONV-V2-01 — pre-registration: a new composition solver built on atlas v2 alone, tested on real known mixtures |
| 2026-09-30 | [PROC_HMIN_REFIT_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_HMIN_REFIT_01_PREREG.md) | PROC-HMIN-REFIT-01 — pre-registration: measuring the methylation floors from sorted cells (option A), before anything mo |
| 2026-09-30 | [PROC_V12_DIAG_01.md](../Biological_Physics/MethylPhys/doors/PROC_V12_DIAG_01.md) | PROC-V12-DIAG-01 — diagnostic (not a bar): is the V12 scatter selection noise or donor spread? |
| 2026-09-30 | [PROC_V12_DIAG_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_V12_DIAG_01_OUTCOME.md) | PROC-V12-DIAG-01 — outcome (2026-09-29): the scatter follows the SOURCE, not the number of samples |
| 2026-09-30 | [PROC_V12_IDENTITY_02_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_V12_IDENTITY_02_OUTCOME.md) | PROC-V12-IDENTITY-02 — outcome (2026-09-29): B3 PASS, B1 FAIL, B2 FAIL |
| 2026-09-30 | [PROC_V12_IDENTITY_02_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_V12_IDENTITY_02_PREREG.md) | PROC-V12-IDENTITY-02 — pre-registration: identity loci on atlas v2, point rule; held-out self-read |
| 2026-09-30 | [PROC_V12_IDENTITY_OUTCOME_01.md](../Biological_Physics/MethylPhys/doors/PROC_V12_IDENTITY_OUTCOME_01.md) | PROC-V12-IDENTITY — outcome of the first build (2026-09-29): B3 FAILED; the rule, not the atlas |
| 2026-09-30 | [PROC_V12_IDENTITY_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_V12_IDENTITY_PREREG.md) | PROC-V12-IDENTITY — pre-registration: identity loci built on atlas v2 alone, and the held-out self-read |
| 2026-09-30 | [PROC_V5_HELDOUT_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_V5_HELDOUT_OUTCOME.md) | PROC-V5-HELDOUT — outcome (2026-09-30): PASS |
| 2026-09-30 | [PROC_V5_HELDOUT_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_V5_HELDOUT_PREREG.md) | PROC-V5-HELDOUT — pre-registration: can the atlas v2 posterior predict data it never saw? |
| 2026-10-01 | [BANDv2_OUTCOME.md](../Biological_Physics/MethylPhys/doors/BANDv2_OUTCOME.md) | OUTCOME — identity_band_v2 tested on GSE125105 controls (Munich) |
| 2026-10-01 | [DEV_CHAIN_V3_RUN2.md](../Biological_Physics/MethylPhys/doors/DEV_CHAIN_V3_RUN2.md) | DEV-CHAIN-V3-RUN2 — chain v3 end to end with the new read rules (development, 2026-10-01; commit d5873bd, floors v1.2) |
| 2026-10-01 | [DEV_CHAIN_V3_RUN3.md](../Biological_Physics/MethylPhys/doors/DEV_CHAIN_V3_RUN3.md) | DEV-CHAIN-V3-RUN3 — chain v3 with the audit fixes and the noise-corrected tare, end to end on real IDATs (development, 2 |
| 2026-10-01 | [DEV_COLON_BLOCKS_02.md](../Biological_Physics/MethylPhys/doors/DEV_COLON_BLOCKS_02.md) | DEV-COLON-BLOCKS-02 — colon-epithelium marker regions against every Loyfer cell type (development, 2026-10-02) |
| 2026-10-01 | [DEV_LOWFRAC_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/DEV_LOWFRAC_01_OUTCOME.md) | DEV-LOWFRAC-01 — neutrophil Met-A below 50 % neutrophils (development, 2026-10-01; after looking) |
| 2026-10-01 | [DEV_MOLECULE_COLON_01.md](../Biological_Physics/MethylPhys/doors/DEV_MOLECULE_COLON_01.md) | DEV-MOLECULE-COLON-01 — reading colon cells' copy error from their own molecules (development, 2026-10-01) |
| 2026-10-01 | [DEV_NOISE_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/DEV_NOISE_01_OUTCOME.md) | DEV-NOISE-01 — array noise index (development, 2026-10-01; follows the PROC-NEUT-TEST-01 T2 failure) |
| 2026-10-01 | [DEV_NOISE_02_OUTCOME.md](../Biological_Physics/MethylPhys/doors/DEV_NOISE_02_OUTCOME.md) | DEV-NOISE-02 — what the whole-blood neutrophil Met-A is reading (development, after looking; 2026-10-01) |
| 2026-10-01 | [DEV_STOOL_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/DEV_STOOL_01_OUTCOME.md) | DEV-STOOL-01 — can stool carry enough colon-lining molecules for IAM-A and Met-A? (development, 2026-10-02) |
| 2026-10-01 | [DIAG_450K_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/DIAG_450K_01_OUTCOME.md) | DIAG-450K-01 — why the v2 reader flagged an immune cell in every EPIC-Italy control (2026-10-01, development diagnosis) |
| 2026-10-01 | [LABZERO01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/LABZERO01_OUTCOME.md) | OUTCOME — LAB-ZERO-01: predicting the per-lab offset from the array's control probes |
| 2026-10-01 | [PROC_AGE_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_AGE_01_OUTCOME.md) | OUTCOME — PROC-AGE-01: cellular age on the identity gauge — NOT REPORTABLE at single-array resolution |
| 2026-10-01 | [PROC_AML_PROG_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_AML_PROG_01_OUTCOME.md) | PROC-AML-PROG-01 — outcome (2026-10-01; pre-registration sha 3d4b365f, unchanged) + development follow-up |
| 2026-10-01 | [PROC_AML_PROG_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_AML_PROG_01_PREREG.md) | PROC-AML-PROG-01 — pre-registration (written 2026-10-01, before any AML array is read on these floors) |
| 2026-10-01 | [PROC_AML_SERIAL_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_AML_SERIAL_01_OUTCOME.md) | PROC-AML-SERIAL-01 — outcome (2026-10-01; pre-registration sha 6fe8dc63, unchanged) |
| 2026-10-01 | [PROC_AML_SERIAL_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_AML_SERIAL_01_PREREG.md) | PROC-AML-SERIAL-01 — pre-registration (written 2026-10-01, before any GSE315367 array is read) |
| 2026-10-01 | [PROC_BIDIR_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_BIDIR_01_OUTCOME.md) | OUTCOME — PROC-BIDIR-01: Stage 4.5 bidirectional detector. B1–B5 PASS — **row 4.5 COMMISSIONED** (`row_4_5_commissioned: |
| 2026-10-01 | [PROC_CEIL_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_CEIL_01_OUTCOME.md) | OUTCOME - PROC-CEIL-01 (T-CEIL): the ceiling conformance guard. PASS, and it produced three findings the guard itself wa |
| 2026-10-01 | [PROC_CHANNEL_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md) | PROC-CHANNEL-01 — each cell type's error budget and holding energy, from single DNA molecules (2026-09-30) |
| 2026-10-01 | [PROC_CLASS_COUNT_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_CLASS_COUNT_01_OUTCOME.md) | PROC-CLASS-COUNT-01 — outcome (2026-09-30) |
| 2026-10-01 | [PROC_CMB_05_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_CMB_05_OUTCOME.md) | OUTCOME — PROC-CMB-05: the patient's sky. C2′ 4/4, C4″, C5, C6 PASS. **Row 4.6 COMMISSIONED** (`row_4_6_commissioned: Tr |
| 2026-10-01 | [PROC_DERIVE_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_DERIVE_01_OUTCOME.md) | PROC-DERIVE-01 — outcome (2026-09-30). Pre-registration: PROC_DERIVE_01_PREREG.md (sha 7521c01ccf37e041). |
| 2026-10-01 | [PROC_DERIVE_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_DERIVE_01_PREREG.md) | PROC-DERIVE-01 — pre-registration (written 2026-09-30, before the time-course files are summarised) |
| 2026-10-01 | [PROC_DNMT_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_DNMT_01_PREREG.md) | PROC-DNMT-01 — pre-registration (written 2026-10-01, before any array or read of these datasets is read by us) |
| 2026-10-01 | [PROC_ENCODE_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_ENCODE_01_OUTCOME.md) | PROC-ENCODE-01 — outcome (2026-09-30). Pre-registration: PROC_ENCODE_01_PREREG.md (sha b0e92114d6367453), written before |
| 2026-10-01 | [PROC_ENCODE_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_ENCODE_01_PREREG.md) | PROC-ENCODE-01 — pre-registration (written 2026-09-30, before any ENCODE read was counted) |
| 2026-10-01 | [PROC_G002_TRACE_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_G002_TRACE_OUTCOME.md) | PROC-G002-TRACE — how the eight original floors were made, and what IAM's law says about them (2026-09-30) |
| 2026-10-01 | [PROC_HMIN_BOOT_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_HMIN_BOOT_01_OUTCOME.md) | PROC-HMIN-BOOT-01 — the bootstrap cross-check of the eight methylation H_min values, run for the first time |
| 2026-10-01 | [PROC_HMIN_PERCELL_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_HMIN_PERCELL_01_OUTCOME.md) | PROC-HMIN-PERCELL-01 — outcome (2026-09-30): do cells share one floor? |
| 2026-10-01 | [PROC_HMIN_REFIT_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_HMIN_REFIT_01_OUTCOME.md) | PROC-HMIN-REFIT-01 — outcome (2026-09-30): option A measured; NOT adoptable as it stands |
| 2026-10-01 | [PROC_LINES_01_BJ_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_LINES_01_BJ_OUTCOME.md) | PROC-LINES-01, first series — BJ fibroblasts (GSE91069), 31 arrays, our Stage 1; every stage read against the same cells |
| 2026-10-01 | [PROC_MAHA_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_MAHA_01_OUTCOME.md) | OUTCOME — PROC-MAHA-01: Stage 5 re-based on the identity gauge (as sealed) |
| 2026-10-01 | [PROC_MAHA_02_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_MAHA_02_OUTCOME.md) | OUTCOME — PROC-MAHA-02: row 5 commissioned; row 5b opened |
| 2026-10-01 | [PROC_MOLECULE_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_MOLECULE_01_OUTCOME.md) | PROC-MOLECULE-01 — outcome (2026-10-01). Pre-registration: PROC_MOLECULE_01_PREREG.md (sha eb71b9146a8ed367). |
| 2026-10-01 | [PROC_MOLECULE_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_MOLECULE_01_PREREG.md) | PROC-MOLECULE-01 — pre-registration (written 2026-09-30, before any per-molecule record was read) |
| 2026-10-01 | [PROC_NEUT_TEST_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_NEUT_TEST_01_OUTCOME.md) | PROC-NEUT-TEST-01 — outcome, all four tests (2026-10-01). Chain v3 (bc4a651), unchanged, run_sample.py --engine v3 on ev |
| 2026-10-01 | [PROC_NEUT_TEST_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_NEUT_TEST_01_PREREG.md) | PROC-NEUT-TEST-01 — pre-registration (written 2026-10-01, before any array below is read by chain v3) |
| 2026-10-01 | [PROC_NEUT_TEST_01_T2_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_NEUT_TEST_01_T2_OUTCOME.md) | PROC-NEUT-TEST-01 T2 — outcome (2026-10-01): FAIL as pre-registered. Cause found: array noise in the second lab. |
| 2026-10-01 | [PROC_PANEL_03_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_PANEL_03_OUTCOME.md) | OUTCOME — PROC-PANEL-03: the age-referenced lab zero — **LAB ZERO COMMISSIONED** |
| 2026-10-01 | [PROC_PREDX_NEUT_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_PREDX_NEUT_01_OUTCOME.md) | PROC-PREDX-NEUT-01 — outcome (2026-10-01). Pre-registration: PROC_PREDX_NEUT_01_PREREG.md (sha 97fdf967972519ef). |
| 2026-10-01 | [PROC_PREDX_NEUT_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_PREDX_NEUT_01_PREREG.md) | PROC-PREDX-NEUT-01 — pre-registration (written 2026-10-01, before EPIC-Italy is read with the corrected rules) |
| 2026-10-01 | [PROC_PREDX_SEQUENCE_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_PREDX_SEQUENCE_01_PREREG.md) | PROC-PREDX-SEQUENCE-01 — pre-registration (written 2026-09-30, before any EPIC-Italy array is read by the current chain) |
| 2026-10-01 | [PROC_PREDX_SLIDE_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_PREDX_SLIDE_01_OUTCOME.md) | PROC-PREDX-SLIDE-01 — outcome (2026-10-01). Pre-registration sha 434b2ed9da21be30. |
| 2026-10-01 | [PROC_PREDX_SLIDE_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_PREDX_SLIDE_01_PREREG.md) | PROC-PREDX-SLIDE-01 — pre-registration (written 2026-10-01, before the 329 GSE51057 arrays are looked at under this rule |
| 2026-10-01 | [PROC_RECORD_02_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_RECORD_02_OUTCOME.md) | PROC-RECORD-02 — VAL-025 to VAL-028 (four-substrate aging trajectory) reclassified: modeled prediction, not measurement |
| 2026-10-01 | [PROC_RECORD_03_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_RECORD_03_OUTCOME.md) | PROC-RECORD-03 — the 80-cell age reference matrix: provenance stated; AD / breast "cellular age in years" restated as ΔA |
| 2026-10-01 | [PROC_SCORE_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_SCORE_01_OUTCOME.md) | PROC-SCORE-01 — outcome (2026-09-30): per-cell floors work on pure cells; reading a cell out of a mixture does not yet |
| 2026-10-01 | [PROC_SCORE_03_RESOLUTION.md](../Biological_Physics/MethylPhys/doors/PROC_SCORE_03_RESOLUTION.md) | PROC-SCORE-03 — per-specimen resolution in blood, in β units (2026-09-30) |
| 2026-10-01 | [PROC_SWITCH_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_SWITCH_01_OUTCOME.md) | OUTCOME — PROC-SWITCH-01: the gauge switch (as sealed) |
| 2026-10-01 | [PROC_SWITCH_02_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_SWITCH_02_OUTCOME.md) | OUTCOME — PROC-SWITCH-02: the gauge switch commissioned (CHAIN_COMMISSIONING row B) |
| 2026-10-01 | [PROC_TIER_02_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_TIER_02_OUTCOME.md) | OUTCOME — PROC-TIER-02: NORMAL set to the commissioned healthy population. U1, U2, U3 PASS. |
| 2026-10-01 | [PROC_TUMOUR_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_TUMOUR_01_OUTCOME.md) | PROC-TUMOUR-01 — outcome (2026-10-01). Scored exactly as pre-registered (sha 5ab460cf5af368d5) |
| 2026-10-01 | [PROC_TUMOUR_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_TUMOUR_01_PREREG.md) | PROC-TUMOUR-01 — pre-registration (written 2026-10-01, before any read of these datasets is aligned) |
| 2026-10-01 | [PROC_WARBURG_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_WARBURG_01_OUTCOME.md) | PROC-WARBURG-01 — outcome (2026-09-30). Pre-registration: PROC_WARBURG_01_PREREG.md (sha a4683f03c3ca92de), written befo |
| 2026-10-01 | [PROC_WARBURG_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_WARBURG_01_PREREG.md) | PROC-WARBURG-01 — pre-registration (written 2026-09-30, before any methylation array of this set was read) |
| 2026-10-01 | [PROC_WB_NEUT_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_WB_NEUT_01_OUTCOME.md) | PROC-WB-NEUT-01 — outcome, part 1 (2026-10-01; pre-registration sha 36e8620f, unchanged) |
| 2026-10-01 | [PROC_WB_NEUT_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_WB_NEUT_01_PREREG.md) | PROC-WB-NEUT-01 — pre-registration (written 2026-10-01, before any mixture is read on this reading) |
| 2026-10-02 | [DEV_REPL_V3_01.md](../Biological_Physics/MethylPhys/doors/DEV_REPL_V3_01.md) | DEV-REPL-V3-01 — technical replicates on chain v3 with the median tare (development note, 2026-10-03; \measured) |
| 2026-10-02 | [DEV_REPL_V3_01_PLAN.md](../Biological_Physics/MethylPhys/doors/DEV_REPL_V3_01_PLAN.md) | DEV-REPL-V3-01 — run plan (development, written 2026-10-03 before any array was read) |
| 2026-10-02 | [DEV_SELFTARE_01.md](../Biological_Physics/MethylPhys/doors/DEV_SELFTARE_01.md) | DEV-SELFTARE-01 — the array tares itself from its own fixed sites (development note, 2026-10-03) |
| 2026-10-02 | [DEV_TARE_02_OUTCOME.md](../Biological_Physics/MethylPhys/doors/DEV_TARE_02_OUTCOME.md) | DEV-TARE-02 — Stage T without a fitted term (development, 2026-10-02) |
| 2026-10-02 | [PROC_DNMT_01_PARTB_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_DNMT_01_PARTB_OUTCOME.md) | DNMT-01 Part B — IAM-A on single molecules under a known DNMT1 block (scored 2026-10-02) |
| 2026-10-03 | [ATLAS_SOURCES_SURVEY.md](../Biological_Physics/MethylPhys/doors/ATLAS_SOURCES_SURVEY.md) | Atlas sources survey — what exists for the cells we are missing (2026-09-27) |
| 2026-10-03 | [CHAIN_COMMISSIONING.md](../Biological_Physics/MethylPhys/archive/docs_consolidated_2026-10-09/CHAIN_COMMISSIONING.md) | Chain v3 commissioning — stage by stage (development) |
| 2026-10-03 | [CHAIN_SEQUENCE.md](../Biological_Physics/MethylPhys/doors/CHAIN_SEQUENCE.md) | The chain, step by step - derived from the code |
| 2026-10-03 | [CLASS_ASSIGNMENT_RULE_DRAFT.md](../Biological_Physics/MethylPhys/doors/CLASS_ASSIGNMENT_RULE_DRAFT.md) | Architecture-class assignment — a written rule, tested against the existing classes (DRAFT, 2026-09-28) |
| 2026-10-03 | [CLASS_HISTORY.md](../Biological_Physics/MethylPhys/doors/CLASS_HISTORY.md) | How the eight architecture classes came to be — the record |
| 2026-10-03 | [CLASS_USE_INVENTORY.md](../Biological_Physics/MethylPhys/doors/CLASS_USE_INVENTORY.md) | Class-use inventory and removal plan — 2026-09-28 |
| 2026-10-03 | [CMB_TO_METHYLOME_MAP.md](../Biological_Physics/MethylPhys/doors/CMB_TO_METHYLOME_MAP.md) | CMB → Methylome: the translation map, scored |
| 2026-10-03 | [COMPLETION_SPRINT_scored.md](../Biological_Physics/MethylPhys/doors/COMPLETION_SPRINT_scored.md) | The completion sprint (spring 2026), scored 2026-09-19 |
| 2026-10-03 | [COMPONENT_MAP.md](../Biological_Physics/MethylPhys/doors/COMPONENT_MAP.md) | COMPONENT MAP — what lives where, and what a future test needs |
| 2026-10-03 | [DEV_ATLAS_EPIC_01.md](../Biological_Physics/MethylPhys/doors/DEV_ATLAS_EPIC_01.md) | DEV-ATLAS-EPIC-01 — atlas v2 and NILC composition on EPIC whole blood (development, 2026-10-03) |
| 2026-10-03 | [DEV_ATLAS_EPIC_02.md](../Biological_Physics/MethylPhys/doors/DEV_ATLAS_EPIC_02.md) | DEV-ATLAS-EPIC-02 — stage 3 atlas deconvolution on chain v3: the held Stage A patch (development; check written 2026-10- |
| 2026-10-03 | [DEV_BASE_CHAIN_01.md](../Biological_Physics/MethylPhys/doors/DEV_BASE_CHAIN_01.md) | DEV-BASE-CHAIN-01 — base chain v3 on every EPIC array in the bucket (development, checks written 2026-10-03 before any a |
| 2026-10-03 | [DEV_COMPOSITION_TRUTH_02.md](../Biological_Physics/MethylPhys/doors/DEV_COMPOSITION_TRUTH_02.md) | DEV-COMPOSITION-TRUTH-02 - an adult, other-laboratory mixture truth set for composition, NILC and atlas_e (development,  |
| 2026-10-03 | [DEV_DETECTION_01.md](../Biological_Physics/MethylPhys/doors/DEV_DETECTION_01.md) | DEV-DETECTION-01 - detection p-value: poobah against the Gaussian negative-control test (development, 2026-10-04) |
| 2026-10-03 | [DEV_DIRECTION_01.md](../Biological_Physics/MethylPhys/doors/DEV_DIRECTION_01.md) | DEV-DIRECTION-01 — stage 10 directional decomposition (development; check written 2026-10-03 before any data were read) |
| 2026-10-03 | [DEV_DIRECTION_02.md](../Biological_Physics/MethylPhys/doors/DEV_DIRECTION_02.md) | DEV-DIRECTION-02 - directional decomposition rebuilt physics-only (development, 2026-10-04) |
| 2026-10-03 | [DEV_EPIC_V2_01.md](../Biological_Physics/MethylPhys/doors/DEV_EPIC_V2_01.md) | DEV-EPIC-V2-01 - EPIC v2 support behind a development flag (development, 2026-10-04) |
| 2026-10-03 | [DEV_FLAGS_01.md](../Biological_Physics/MethylPhys/doors/DEV_FLAGS_01.md) | DEV-FLAGS-01 - development flags for the stages likely to work, and the stages kept out (development, 2026-10-04) |
| 2026-10-03 | [DEV_IAMA_CSCORE_01.md](../Biological_Physics/MethylPhys/doors/DEV_IAMA_CSCORE_01.md) | DEV-IAMA-CSCORE-01 - the IAM-A C-score (development, 2026-10-04) |
| 2026-10-03 | [DEV_IAMA_REAL_01.md](../Biological_Physics/MethylPhys/doors/DEV_IAMA_REAL_01.md) | DEV-IAMA-REAL-01 - Stage Q (IAM-A) end to end on real single-molecule blood data (development, 2026-10-04) |
| 2026-10-03 | [DEV_INTAKE_02.md](../Biological_Physics/MethylPhys/doors/DEV_INTAKE_02.md) | DEV-INTAKE-02 - intake changes F, A, B, L and the EPIC v2 refusal (development, 2026-10-04) |
| 2026-10-03 | [DEV_NEWCELL_01.md](../Biological_Physics/MethylPhys/doors/DEV_NEWCELL_01.md) | DEV-NEWCELL-01 - the new-cell rule (three tests) applied to the next cell after neutrophils (development, 2026-10-04) |
| 2026-10-03 | [DEV_NILC_01.md](../Biological_Physics/MethylPhys/doors/DEV_NILC_01.md) | DEV-NILC-01 — stage 4 NILC component separation on chain v3 (development; check written 2026-10-03 before the data were  |
| 2026-10-03 | [DEV_PERCELL_01.md](../Biological_Physics/MethylPhys/doors/DEV_PERCELL_01.md) | DEV-PERCELL-01 — stage 5 Met-A for each newly separated cell type (development; check written 2026-10-03 before the data |
| 2026-10-03 | [DEV_ROUND2_REPORT.md](../Biological_Physics/MethylPhys/doors/DEV_ROUND2_REPORT.md) | Chain v3 development round 2 - report (DEVELOPMENT - not commissioned) |
| 2026-10-03 | [DEV_SELFTARE_02.md](../Biological_Physics/MethylPhys/doors/DEV_SELFTARE_02.md) | DEV-SELFTARE-02 - self-tare on type II fixed sites (development, 2026-10-04) |
| 2026-10-03 | [DEV_SKYSTAT_01.md](../Biological_Physics/MethylPhys/doors/DEV_SKYSTAT_01.md) | DEV-SKYSTAT-01 — stage 12 sky statistics (development; check written 2026-10-03 before any data were read) |
| 2026-10-03 | [DEV_SKY_01.md](../Biological_Physics/MethylPhys/doors/DEV_SKY_01.md) | DEV-SKY-01 — stage 11 sky map on chain v3 (development; check written 2026-10-03 before the data were read) |
| 2026-10-03 | [DEV_SKY_02.md](../Biological_Physics/MethylPhys/doors/DEV_SKY_02.md) | DEV-SKY-02 - sky map against a within-chromosome block-shuffle null; sky statistics (development, 2026-10-04) |
| 2026-10-03 | [DEV_SPECIES_AGEING_01.md](../Biological_Physics/MethylPhys/doors/DEV_SPECIES_AGEING_01.md) | DEV_SPECIES_AGEING_01 — within-species ageing on the consortium blood arrays |
| 2026-10-03 | [DEV_TOOLKIT_ADDED_02.md](../Biological_Physics/MethylPhys/doors/DEV_TOOLKIT_ADDED_02.md) | DEV-TOOLKIT-ADDED-02 - trace cell (3b), foreign cell (3c), surface brightness (11b) on each array's own noise; IAM-A ver |
| 2026-10-03 | [DEV_TOOLKIT_ADDED_STAGES_01.md](../Biological_Physics/MethylPhys/doors/DEV_TOOLKIT_ADDED_STAGES_01.md) | DEV-TOOLKIT-ADDED-01 — stages 3b, 3c, 11b, 12b (development; checks written 2026-10-03 before any data were read) |
| 2026-10-03 | [ENHANCEMENTS.md](../Biological_Physics/MethylPhys/doors/ENHANCEMENTS.md) | What would make this chain more sensitive, ranked — and what it would cost |
| 2026-10-03 | [FINDING_DETECTION_PANEL_HELDOUT.md](../Biological_Physics/MethylPhys/doors/FINDING_DETECTION_PANEL_HELDOUT.md) | FINDING — the foreign-cell detection panel, held out on 732 healthy blood arrays (2026-09-27) |
| 2026-10-03 | [FINDING_GSE125105_LOW_SIGNAL.md](../Biological_Physics/MethylPhys/doors/FINDING_GSE125105_LOW_SIGNAL.md) | Finding 2026-09-27 — GSE125105 (Munich) arrays are low-signal, and the intake gate that should refuse them never fires |
| 2026-10-03 | [FRACTION_AND_A.md](../Biological_Physics/MethylPhys/doors/FRACTION_AND_A.md) | Fraction and A — what mixing does to the per-cell A, measured on constructed specimens, 2026-09-26 |
| 2026-10-03 | [LABZERO02_OUTCOME.md](../Biological_Physics/MethylPhys/doors/LABZERO02_OUTCOME.md) | OUTCOME — LAB-ZERO-02: the fourth lab decides the lab-zero route |
| 2026-10-03 | [PERCELL_STANDARD_REZERO_2026-09-27.md](../Biological_Physics/MethylPhys/doors/PERCELL_STANDARD_REZERO_2026-09-27.md) | The per-cell standard re-zeroed to 1.000 (PLAN item 4, first step) — 2026-09-27 |
| 2026-10-03 | [PER_CELL_SCORING.md](../Biological_Physics/MethylPhys/doors/PER_CELL_SCORING.md) | The per-cell reading already exists — what is missing is a healthy band |
| 2026-10-03 | [PHASE1_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PHASE1_OUTCOME.md) | OUTCOME — PHASE 1: identity-loci healthy bands from GSE87571 |
| 2026-10-03 | [PHASE1c_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PHASE1c_OUTCOME.md) | OUTCOME — PHASE 1c: scale-map and band transfer to GSE42861 controls |
| 2026-10-03 | [PLAN.md](../Biological_Physics/MethylPhys/archive/docs_consolidated_2026-10-09/PLAN.md) | PLAN — what we do next, in order |
| 2026-10-03 | [PROC_BAND_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_BAND_01_OUTCOME.md) | PROC-BAND-01 — outcome: NOT COMMISSIONED. The joint component fails reproducibility on one laboratory in four. |
| 2026-10-03 | [PROC_BRAIN_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_BRAIN_01_OUTCOME.md) | PROC-BRAIN-01 — outcome: brain-derived cells ARE found in cerebrospinal fluid, in every patient. Whether they can be SCO |
| 2026-10-03 | [PROC_BRAIN_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_BRAIN_01_PREREG.md) | PROC-BRAIN-01 — can the instrument find terminal-class cells in a liquid specimen from a CNS tumour patient? |
| 2026-10-03 | [PROC_CLS_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_CLS_01_OUTCOME.md) | PROC-CLS-01 — outcome: the residual sky HAS large-scale structure. The reference is NOT COMMISSIONED. |
| 2026-10-03 | [PROC_CMB_04_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_CMB_04_OUTCOME.md) | OUTCOME — PROC-CMB-04: the patient's sky. C2′ FAILED AS SEALED (1/4 labs, all four within 0.004 of the bar); C4″, C5, C6 |
| 2026-10-03 | [PROC_COV_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_COV_01_OUTCOME.md) | PROC-COV-01 — outcome: the misfit IS a reproducible, removable bias. Removing it does not rescue fidelity recovery. |
| 2026-10-03 | [PROC_COV_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_COV_01_PREREG.md) | PROC-COV-01 — is the reference's misfit against real blood a reproducible bias that can be measured and removed? |
| 2026-10-03 | [PROC_DNMT_01_PARTA_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_DNMT_01_PARTA_OUTCOME.md) | DNMT-01 Part A — Met-A under a known DNMT1 block (development measurement, 2026-10-01) |
| 2026-10-03 | [PROC_E2E_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_E2E_01_OUTCOME.md) | PROC-E2E-01 — outcome: the commissioned chain reproduces the test package, and the run found three defects |
| 2026-10-03 | [PROC_EPIC_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_EPIC_01_OUTCOME.md) | PROC-EPIC-01 — outcome: the colorectal signal replicates in held-out blood. The breast signal does not. |
| 2026-10-03 | [PROC_FOREIGNSCORE_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_FOREIGNSCORE_01_OUTCOME.md) | PROC-FOREIGNSCORE-01 — outcome: NOT ADOPTED. No scoring floor is set; a detected foreign cell stays "detected, fraction  |
| 2026-10-03 | [PROC_FOREIGN_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_FOREIGN_01_OUTCOME.md) | PROC-FOREIGN-01 — outcome: COMMISSIONED. The immune tier is withheld when a specimen is not whole blood. |
| 2026-10-03 | [PROC_HISTORY_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_HISTORY_01_OUTCOME.md) | PROC-HISTORY-01 — the complete validation history, counted from the record (2026-09-21) |
| 2026-10-03 | [PROC_INTAKE_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_INTAKE_01_OUTCOME.md) | PROC-INTAKE-01 — outcome: ADOPTED. The intake gate runs on the array's own numbers; a deferred check never advances. |
| 2026-10-03 | [PROC_LABBAND_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_LABBAND_01_OUTCOME.md) | PROC-LABBAND-01 — outcome: NOT COMMISSIONED. At 80 arrays a laboratory's width cannot be told from noise, and using it m |
| 2026-10-03 | [PROC_MAHA_03_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_MAHA_03_OUTCOME.md) | PROC-MAHA-03 — outcome: the chip term is real where it matters most, and a control array per chip is the wrong protocol |
| 2026-10-03 | [PROC_MAHA_03_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_MAHA_03_PREREG.md) | PROC-MAHA-03 — the Sentrix-chip term, and which panel protocol recovers it |
| 2026-10-03 | [PROC_MATCH_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_MATCH_01_OUTCOME.md) | OUTCOME — PROC-MATCH-01: Stage 8 disease matching. M1, M2, M4 PASS; M3 reported; M5 prediction correct — **row 8 stays O |
| 2026-10-03 | [PROC_MF_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_MF_01_OUTCOME.md) | PROC-MF-01 — outcome: NOT COMMISSIONED as written. The full-covariance matched filter ties NNLS; the diagonal control ar |
| 2026-10-03 | [PROC_MF_02_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_MF_02_OUTCOME.md) | PROC-MF-02 — outcome: NOT COMMISSIONED. Six of seven bars met, and the detection limit falls to 0.5–1 %; the threshold d |
| 2026-10-03 | [PROC_MF_03_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_MF_03_OUTCOME.md) | PROC-MF-03 — outcome: NOT COMMISSIONED. B1–B6 met again; B7 failed on the fifth laboratory — its healthy null is 4–25× w |
| 2026-10-03 | [PROC_PARTIAL_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_PARTIAL_01_OUTCOME.md) | PROC-PARTIAL-01 — outcome: NOT COMMISSIONED. A non-blood class's fidelity score cannot be recovered from whole blood. |
| 2026-10-03 | [PROC_SKY_01_FOLLOWUP.md](../Biological_Physics/MethylPhys/doors/PROC_SKY_01_FOLLOWUP.md) | PROC-SKY-01 — follow-up named by the outcome: the same 48 arrays with the intake gate applied (2026-09-27) |
| 2026-10-03 | [PROC_SKY_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_SKY_01_OUTCOME.md) | PROC-SKY-01 — outcome: SKY WITHHELD. The panel scales are retired; no on-array spread fitted every laboratory. |
| 2026-10-03 | [PROC_SMALL_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_SMALL_01_OUTCOME.md) | PROC-SMALL-01 — outcome: the trace-class detection limit in whole blood is 2 %, down from 5 % |
| 2026-10-03 | [PROC_SMALL_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_SMALL_01_PREREG.md) | PROC-SMALL-01 — can the detection limit for a trace class in whole blood be brought below 5 %? |
| 2026-10-03 | [PROC_STAGE2D_02_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_STAGE2D_02_PREREG.md) | PROC-STAGE2D-02 — pre-registration: the foreign-cell detector rebuilt on the held-out finding |
| 2026-10-03 | [PROC_STAGE2D_03_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_STAGE2D_03_OUTCOME.md) | PROC-STAGE2D-03 — outcome: ADOPTED by the author's ruling. B1's unspecific bar failed as written and was superseded, not |
| 2026-10-03 | [PROC_STAGE2D_03_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_STAGE2D_03_PREREG.md) | PROC-STAGE2D-03 — pre-registration: foreign-cell detection as one joint fit |
| 2026-10-03 | [PROC_SYNTH_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_SYNTH_01_OUTCOME.md) | PROC-SYNTH-01 — outcome: the chain recovers what it is handed. But the per-cell A is confounded with the cell's FRACTION |
| 2026-10-03 | [PROC_TARE_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_TARE_01_PREREG.md) | PROC-TARE-01 — pre-registration: can the array's own known-value probes tare the instrument, so that no healthy panel de |
| 2026-10-03 | [PROC_TIER_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_TIER_01_OUTCOME.md) | OUTCOME — PROC-TIER-01: Stage 7 tiers on the commissioned gauge. T1, T3, T4 PASS — row 7 COMMISSIONED. T2 measured; pred |
| 2026-10-03 | [PROC_TISSUE_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_TISSUE_01_OUTCOME.md) | PROC-TISSUE-01 — outcome: the gating bar failed. No field effect; adjacent normal reads BELOW healthy, and seven of eigh |
| 2026-10-03 | [PROC_TISSUE_01_PREREG.md](../Biological_Physics/MethylPhys/doors/PROC_TISSUE_01_PREREG.md) | PROC-TISSUE-01 — does the gauge place healthy, adjacent-normal and tumour colon in order, without being shown the order? |
| 2026-10-03 | [PROC_UNMIX_01_OUTCOME.md](../Biological_Physics/MethylPhys/doors/PROC_UNMIX_01_OUTCOME.md) | PROC-UNMIX-01 — outcome: NOT ADOPTED. The dilution-line inversion is exact arithmetic the composition solver cannot feed |
| 2026-10-03 | [README.md](../Biological_Physics/MethylPhys/doors/README.md) | doors/ — the record of how the chain was built, and what is next |
| 2026-10-03 | [REFERENCE_AUDIT.md](../Biological_Physics/MethylPhys/doors/REFERENCE_AUDIT.md) | The per-cell A is computed on the wrong surface — measured 2026-09-26 |
| 2026-10-03 | [REPORT_TAB_REFERENCE.md](../Biological_Physics/MethylPhys/doors/REPORT_TAB_REFERENCE.md) | The report, tab by tab - the operating reference |
| 2026-10-03 | [REPO_INVENTORY.md](../Biological_Physics/MethylPhys/doors/REPO_INVENTORY.md) | What is in this repository - measured |
| 2026-10-03 | [REVIEWER_MANIFEST.md](../Biological_Physics/MethylPhys/doors/REVIEWER_MANIFEST.md) | What a reviewer can download, and what we have not published |
| 2026-10-03 | [RUNBOOK.md](../Biological_Physics/MethylPhys/doors/RUNBOOK.md) | CPG runbook |
| 2026-10-03 | [SUBSTRATE_STRATEGY.md](../Biological_Physics/MethylPhys/doors/SUBSTRATE_STRATEGY.md) | Which substrate can this instrument read? — the plan, decided by the physics rather than by preference |
| 2026-10-03 | [TWO_FIT_FINDING.md](../Biological_Physics/MethylPhys/doors/TWO_FIT_FINDING.md) | The chain runs TWO deconvolutions, and the reported one is the pooled-first fit |
| 2026-10-04 | [MethylPhys_CPG_SOP_v3.md](../Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md) | DEV-SELFTARE-03 - self-tare II adopted by the author and wired into Stage T step 1, before the median tare (development, 2026-10-04) |

### 2026-10-04 · DEV-PAIRED-01: purified neutrophils from a new lab (GSE128733), first read
**Why:** GSE128733 (arrays) and GSE128731 (deep WGBS) measured the same samples, purified neutrophils, CD4 T cells and whole blood,
on both platforms. It is the first public set found where Met-A and IAM-A can be read on the same neutrophil specimens.
**Data:** the two purified-neutrophil EPIC arrays (GSM3684010 Sample6, GSM3684011 Sample7; slide 200357150019), read through chain v3
locally (`run_neutrophil`, specimen isolated neutrophils) with Stage 1 calibration and self-tare II (`dev_stages.selftare_ii`).
The four whole-blood samples are on 450K, which the chain refuses at intake (PLATFORM_REFUSED), as designed.

| Array | probes | A untared | A self-tared (II) | noise index N | C-score |
|---|---|---|---|---|---|
| Sample6 | 831,420 | 1.1663 | **1.0437** | 0.169 | 1.40 |
| Sample7 | 835,054 | 1.1695 | **1.0426** | 0.182 | 1.77 |

**Reading:** another lab's untared arrays sit 17 % above the floor, as earlier cross-lab arrays did. Self-tare II, using only each array's own
fixed sites, brings both to 1.04, inside Normal (0.95–1.05), and the two agree to 0.001. The same-run median tare cannot run: the slide
holds two neutrophil arrays and Stage T needs at least three. Both noise indices sit above the reference arrays' range (0.149), so the
gauge state is withheld as the chain is built to do, and the C-score (not tared) reads above the healthy range seen so far (0.70–1.23),
most likely the same array noise. Targets for commissioning: C-score after tare; noise index on more arrays from this platform year.
**Next:** IAM-A on the same two specimens from GSE128731 (8 WGBS runs, 414 GB, three library kits, two sequencers): Box Run 2.

### 2026-10-04 · DEV-SELFTARE-03: self-tare II adopted as Stage T step 1 and wired into the chain
**Decision:** on 2026-10-04 the author adopted self-tare II, then the median tare, as the Stage T reading (`boxruns/run1/JOBS.md` job A;
DEV-SELFTARE-02 reading (iv); DEV-PAIRED-01).
**Wiring:** `chain/conductor_v3.py: run_neutrophil` now runs self-tare II (`stage_t_selftare_ii`, which calls `dev_stages.selftare_map`
unchanged) as Stage T step 1, before Stages A, M and MC; the median tare (`stage_t_tare`, step 2) runs on the self-tared A. The noise
index and the noise gate read the betas before self-tare II, so they are unchanged. The step-1 record is in the bundle under
`tare.selftare_ii`. `--dev-selftare-ii` is kept as a no-op alias (it copies that record to `development.selftare_ii`; nothing is recomputed).
**Release check:** E3's constructed whole-blood specimen now carries `ref_value` of `dev_selftare_typeII_EPIC_v1.json` (the six GSE110554
purified neutrophil reference arrays, the same six as `metA_floors_v1_3.json` and `neutrophil_reference_v1_1.json`) at its self-tare II
fixed sites, so step 1 does not rescale its already-referenced substituted sites; E3's bars and expected values are unchanged.
With the colon array's own fixed sites E3 read Met-A 0.0159; rebuilt, it reads Met-A 1.0, A_rel 1.0. `release_check_v3.py`: 17/17 PASS.
SOP v3, full SOP and the operations manual updated to "wired"; manual PDF rebuilt.

### 2026-10-05 · BOX RUN 1 (chain v3, arrays): first planned box run
**What:** one driver ran five jobs on the box with self-tare II wired into Stage T (`Biological_Physics/MethylPhys/boxruns/run1/`).
Results are in `s3://methylphys-data-945451304272-us-west-2-an/results/BOXRUN1/`. Development readings, not commissioned results, except job A.

- **A, commissioning check of self-tare II then the median tare, on purified neutrophils (bars from `doors/CHAIN_COMMISSIONING.md`): all bars met.**
  Same-person replicate spread 0.0164 (bar ≤ 0.020); replicates in Normal 62/63 (bar ≥ 95 %); other-laboratory purified neutrophils
  in Normal 68/68 (bar: all); floor arrays in Normal 6/6 (bar 6/6).
- **C, Met-A C-score on every healthy array, median (2.5–97.5 %) by set:** GSE250556 0.840 (0.690–1.201), n = 63;
  healthy_repeat 1.067 (0.728–1.852), n = 402; DEV_BASE_CHAIN_01 1.165 (0.836–1.809), n = 152. The healthy band is not set yet.
- **E, atlas composition on 6 GSE112618 whole bloods (FACS-counted):** ran on all 6; the comparison with the FACS fractions is the next
  (local) step. GSE182379 (12 constructed mixtures) waits for the next box run.
- **B, every chain test set read again: stopped** by a parallel-write race (an array in two sets saved by two threads at once). Fixed
  (`a0aae763`); rerun on 2026-10-05.
- **D, sky statistics with the apodised mask: not run** (healpy missing on the box); rerun on 2026-10-05 with healpy installed.

Box: m7a.8xlarge, 12-hour session credentials, 500 GB scratch disk deleted after the run. First attempt stopped on a full root disk.

### 2026-10-05 · BOX RUN 1, rerun of jobs B and D
- **D, sky statistics, hard mask and apodised mask (2.0°) side by side, 433 healthy whole-blood arrays: ran; commissioning bars not met.**
  Median band power over the block-shuffle null, bands 1–6 (bar 0.9–1.1 in every band):
  hard mask 1.835, 1.034, 1.143, 1.146, 1.079, 1.048; apodised mask 1.182, 1.148, 1.160, 1.134, 1.064, 1.050.
  Look-elsewhere rate (bar ≤ 0.084): hard 0.905, apodised 0.611. The apodised mask removes most of the band-1 excess (1.84 → 1.18),
  which is what a hard-edged mask produces in sky maps, but bands 1–4 still sit above the null. The sky stays withheld.
- **B, every chain test set read again: stopped again.** The first set (902 arrays) read cleanly; the worker then crashed with a
  segmentation fault (exit −11) before writing `B_all.csv`. Cause not yet found; candidates are a thread-unsafe library call under 30
  threads, or memory. Next: run B with fewer workers, one set at a time, so a crash keeps the sets already read.
- Box environment: installing healpy had pulled numpy 2 and broken pandas; pinned to numpy 1.26.4, pandas 1.5.3, healpy 1.17.3.
  Scratch disk deleted after the run; the box shut itself down.

### 2026-10-05 · BOX RUN 1 job E scored against FACS (local)
Atlas_e and the chain's own composition on the 6 GSE112618 whole bloods against their FACS fractions: mean absolute error, atlas_e /
chain composition: neutrophils 0.016 / 0.031, granulocytes 0.021 / 0.017, monocytes 0.005 / 0.010, B 0.013 / 0.007, NK 0.036 / 0.031,
CD4 T 0.013 / 0.023, CD8 T 0.044 / 0.024 (full table and maxima in `boxruns/run1/JOBS.md`). Open check before this counts as held-out
truth: donor overlap with the purified references (GSE110554).

### 2026-10-05 · DEV-CSCORE-TARE-01 and DEV-IAMA-INTAKE-01 (no box)
- **Met-A C-score, tared like Met-A:** the laboratory spread in the healthy C-score (series medians 0.82–1.47) goes away with the
  same-run tare (0.99–1.01); a band set on half the laboratories holds on the other half (median 95.1 % inside; untared 90.3 %).
  Proposed: read C tared and set the band on tared values from held-out laboratories. `doors/DEV_CSCORE_TARE_01.md`.
- **Job E donors:** the 6 GSE112618 whole bloods (ids 3021–3177 Cit-80) share no donor id with the GSE110554 purified references
  (B0044, PCA0612, …), so on the public record job E's FACS comparison is against bloods not used to build the references.
- **IAM-A intake (Stage Q0):** specification written for read-level checks (readable file, pipeline, conversion, coverage, read length,
  duplicates, purified specimen), each a named stop. `doors/DEV_IAMA_INTAKE_01.md`. Not wired.
- **Box Run 1 job B:** made resumable (sets finished in an earlier attempt are restored from S3 and kept) and parquet/tar reads are
  serialised, the likeliest cause of the segfault. Driver tests: 37 pass; the 2 sky tests fail only in the local sandbox (healpy cannot
  write its config there), unchanged by this fix.

### 2026-10-05 · Planning for Box Run 2 and for per-cell references (no box)
- **Pipeline pinned** from Loyfer et al. 2023 Methods: bwa-meth v0.2.0 (default parameters), SAMtools v1.9, wgbstools v0.1.0, hg19
  (28,217,448 CpGs). Job sheet: `boxruns/run2/JOBS.md`.
- **Cell references:** Loyfer has 3 read-level samples each for monocytes, NK, B, CD4 T and CD8 T cells, one laboratory; neutrophils only
  as granulocytes (3). Every cell needs another laboratory's healthy samples before its reference can be commissioned; GSE128731 supplies
  them for neutrophils and CD4 T cells. `doors/CELL_REFERENCE_AVAILABILITY.md`.

### 2026-10-08 · Box access by role; Day 1 box session started
The box now reaches S3 through its own AWS role (`methylphys-box-s3`: read and write on the data bucket only, no delete), attached by the
author; no keys are copied to the box. Day 1 session on methylphys-cpu-01 (m7a.8xlarge, 500 GB scratch disk): Box Run 1 jobs B (every test
set re-read with the adopted tare, resuming from the set already finished) and E (the 12 constructed mixtures, GSE182379), alongside Box Run 2
session 1. The box shuts itself down when both finish; the scratch disk is deleted afterwards.

### 2026-10-08 · Box Run 2 session 1: the pinned Loyfer pipeline, first steps
- \measured The wgbstools 0.1.0 hg19 dictionary holds **28,217,448 CpG sites, exactly Loyfer 2023's hg19 count**; per-chromosome index ranges
  saved as `hg19_cpg_chrom_ranges.json` (Stage Q0.2).
- bwa-meth 0.2.0 exists only for Python ≤ 3.6: it runs in its own environment. wgbstools 0.1.0 calls curl without following redirects, so
  hg19 is fetched by the session script and passed in. Two optional wgbstools modules (segmentor, homog) do not compile without Boost; neither
  is used by bam2pat.
- **Pipeline pin completed:** Loyfer 2023 Methods mark duplicates with **Sambamba 0.6.5** (`-l 1 -t 16 --sort-buffer-size 16000
  --overflow-list-size 10000000`) and drop reads with `-F 1796 -q 10` before the PAT step. The earlier pin (bwa-meth, SAMtools, wgbstools)
  missed the duplicate step; session 2 includes it. Session 1's 1-million-read check tests the format only, so it is unaffected.

### 2026-10-08 · DEV-IAMA-P-WHOLE-01: neutrophil P re-measured on whole files
\calibrated P = 1.1492 (v1: 1.099 on the first 60 MB of each file). The three healthy Loyfer files read 0.994 / 1.017 / 0.989 whole
(in-sample). A reading of part of a file is now refused. Note: `doors/DEV_IAMA_P_WHOLE_01.md`.

### 2026-10-08 · Stage Q0, the IAM-A intake, wired (development)
`chain/stage_q0_intake.py` before Stage Q: readable file, hg19 build, whole genome, purified-neutrophil specimen stop a file now;
conversion, read length and duplicates are recorded until their limits are set from healthy files. Five constructed negative controls each
stop with their named reason (release check E11). The real Loyfer head passes the build check (0 of 1,300,837 lines outside hg19 ranges).

### 2026-10-08 · Met-A C-score tared in the chain
`conductor_v3.stage_t_cscore`: C_rel = C ÷ median C of ≥ 3 same-run healthy references, the rule DEV-CSCORE-TARE-01 tested; shown against
the development band 0.751-1.409. Untared C still printed. Constructed check: three references (1.10, 1.25, 1.20) and C 1.30 give C_rel
1.0833; the array itself is excluded from its own references; fewer than three references leave C_rel unset with the reason.

### 2026-10-08 · Stage Q0 conversion limit, and the alignment record for session 2
- Q0.3 conversion limit set to **≥ 98 % C-to-T conversion, the ENCODE WGBS data standard** (a published standard, not a value chosen from
  our data; the Loyfer .pat files carry no non-CpG calls, so healthy-file values cannot set it). Duplicates: no published limit; recorded.
- `boxruns/run2/alignment_qc.py`: conversion in CHH context from the lambda spike-in (CHH so dcm sites cannot count), else CHH on human chr1
  reads; strand from bwa-meth's YD tag; duplicate fraction from samtools flagstat on the Sambamba-marked BAM. Constructed check: 40 of 80
  CHH calls converted on each strand read as 40/40; CpG cytosines are not counted. The check caught a parsing bug (the strand tag kept its
  line ending), fixed before any real file.

### 2026-10-08 · DEV-Q0-HEALTHY-01: Stage Q0 on real files
\measured The three whole healthy Loyfer hg19 files pass Stage Q0 (0 lines outside hg19 ranges, 22 autosomes; share of molecules with ≥ 6
CpG calls 0.0647 / 0.0665 / 0.0704). The real hg38 copy of one file stops with GENOME_BUILD_MISMATCH (42.9 of 105.0 million lines out of
range). Limits decided before any test file: conversion ≥ 98 % (ENCODE); read length and duplicates recorded without a stop limit (reasons in
`doors/DEV_Q0_HEALTHY_01.md`).

### 2026-10-09 · Box Run 1 job B complete; job E read the 12 mixtures
\measured Job B finished (exit 0, 83 min) and job E read all 18 composition arrays (6 FACS bloods, 12 constructed mixtures). On the 541
healthy arrays that reach a tared Met-A, 525 read Normal (97.0 %), median 1.0001; all 18 series have medians 0.994-1.005. 242 healthy
arrays are refused before reading, each with its reason (cells without a reference, bone marrow, PBMC, EPIC v2). No in-scope real positive
control exists in the test sets, and the C-score band cannot be checked on new laboratories yet (every held-out healthy series is refused
before Met-A). Table: `STATUS.md` section 8, round 4.

### 2026-10-09 · DEV-METAA-SENS-01: Met-A responds to a constructed loss of pattern
\measured A known loss of the neutrophil pattern, put into real healthy arrays and run through the whole chain, moves tared Met-A by what the
chain's own model predicts (ratio 0.998 purified neutrophils; 1.043 whole blood). Every array leaves Normal at a 2 % loss (neutrophils) and a
5 % loss (whole blood). The written bar for a 1 % loss (≥ 95 % leave Normal) is not met (0.875 and 0.083); it was set at the band edge and
stays recorded as not met.

### 2026-10-09 · Composition on 12 constructed EPIC mixtures from another laboratory (GSE182379)
\measured Scored against the depositors' known fractions with the existing DEV-NILC-01 bars. The neutrophil fraction, the one Met-A uses in
whole blood, is recovered within 0.02 by both methods (RMSE atlas_e 0.014, chain composition 0.019). atlas_e meets 6 of 8 groups
(eosinophils 0.034, CD8 T 0.056 outside); the chain's 8-group composition meets 5 of 8 (basophils, B, CD8 T just outside, 0.030-0.032).
This is the first adult EPIC mixture truth set (round 2 found none and used 450K). Files: `doors/data/DEV_METAA_SENS_01/jobE_*`.

### 2026-10-09 · DEV-NEWLAB-GRAN-01: a new laboratory's healthy granulocytes (GSE226298)
\measured Tared Met-A: 26 of 26 healthy controls Normal (median 1.001). Tared C-score: 23 of 26 inside the band set on 19 other series
(bar 95 % not met; untared 13 of 26); the three outside are all low. The Met-A draft commissioning note now carries this as bars 10 and 11.

### 2026-10-09 · Box Run 2 session 1 done; format check passed at the start of session 2
- \measured bwa-meth hg19 + spike-in index built in 8,267 s (saved to S3 with the wgbstools hg19 references). 1M read pairs of SRR9888333
  (150 bp, paired) aligned in 121 s on 24 threads (99.9 % mapped, 98.5 % properly paired).
- Session 1 could not run its own format check (the bam2pat output folder did not exist and the script path was empty); both fixed. Session 2
  ran it first on the same 1M-pair BAM: \measured 200,000 of 200,000 lines of our PAT file valid against the hg19 dictionary, same format as
  the Loyfer file. **Pass.**
- \observed Mean CpGs per PAT line: ours 3.08 (first 200,000 lines, chr1-5 at 0.1× depth) vs Loyfer 6.28 (first 200,000 lines, the start
  of chr1). The two heads cover different regions, so this is not a like-for-like comparison; Stage Q0.5 records the share of molecules with
  ≥ 6 calls on every file, and session 2 measures it per kit.
- Box resized to m7a.32xlarge for session 2 (8 runs × 25M read pairs). First start stopped at the Sambamba version check (sambamba exits 1
  when printing its version, which `pipefail` treated as a failure); fixed and relaunched within 12 minutes.

## 2026-10-09 · Met-A on neutrophils (EPIC v1) COMMISSIONED
The author approved commissioning with the detection limits printed on every report (2 % loss of the neutrophil pattern in purified
neutrophils, 5 % in whole blood). From today Met-A readings in that scope are results; every other stage stays development. Note:
`doors/COMMISSIONING_NOTE_METAA_NEUTROPHILS.md`; chain build label, report box, SOP, OM, changelog and commissioning record updated.
End-to-end check: one healthy purified-neutrophil array (A_rel 0.998) and one healthy whole blood (1.007) read Normal with the box shown.

### 2026-10-09 · C-score commissioning plan, steps 1-3 run
Plan written first (`doors/CSCORE_COMMISSIONING_PLAN.md`). \measured Leave-one-laboratory-out band: 94.2 % pooled (bar 95 %), one
laboratory 84 % (bar 85 %). Positive control: clustered loss reads above the band 100 % of the time down to 5 % of sites × 5 % loss;
scattered loss stays inside 97.5 % (neutrophils) / 88.3 % (whole blood). Same-person replicate spread 0.138 vs healthy 0.175 (bar ≤ half).
\calculated The replicate spread equals the sampling error of a variance over 120 blocks, √(2/119) = 0.130: the band is the statistic's own
counting noise. Proposed (author's decision): read the C-score over the 48,528 noise sites (≈ 970 blocks, expected error 0.045), then rerun.

### 2026-10-09 · C-score block size (development test; chain unchanged)
\measured Same-person repeat noise of the C-score: 0.089 (blocks of 50, current), 0.071 (25), 0.043 (10). Clustered change detected 100 %
at every block size; scattered change never above 1 + 3 × error. Correction: the tared repeat spread 0.138 includes the tare's own noise
(untared 0.089), so the earlier "equals the sampling error" line was replaced. Proposed for the author: blocks of 10, then rerun steps 1-3.

### 2026-10-09 · Box Run 2 session 2: IAM-A on another laboratory's healthy neutrophils, donor 6 (4 kits)
25M read pairs per run, pinned Loyfer pipeline, Stage Q0, Stage Q (P v2 = 1.1492). Development readings.

| run | kit / sequencer | conversion (lambda CHH) | duplicates | share ≥ 6 calls | IAM-A (halves) | ε | IAM-A C-score |
|---|---|---|---|---|---|---|---|
| SRR9888330 | Swift / NovaSeq | 0.983 | not recorded (fixed after this run) | 0.106 | **1.159** above Normal (1.160 / 1.157) | 0.0467 | 224 |
| SRR9888331 | QIAseq / HiSeq X | **0.920** | 0.107 | 0.052 | stopped at Q0: QUARANTINE_CONVERSION (< 98 %) | – | – |
| SRR9888332 | Swift / HiSeq X | 0.983 | 0.146 | 0.103 | **1.167** above Normal (1.168 / 1.167) | 0.0471 | 306 |
| SRR9888333 | TruSeq / HiSeq X | 0.992 | 0.174 | 0.140 | **1.047 Normal** (1.046 / 1.048) | 0.0408 | 45 |

\observed The same donor's neutrophils read Normal with TruSeq and 1.16-1.17 with Swift on both sequencers; the sequencer does not matter,
the library kit does. Swift libraries carry a known artificial loss of methylation near one read end (the adaptase tail), which would add
isolated "errors" at real methylated sites. Test planned (development): the same Swift and TruSeq BAMs through bam2pat with 0, 10 and 15 bp
clipped from read ends — if clipping brings Swift down and leaves TruSeq unchanged, the kit effect is the read-end artefact.
\measured Q0 stopped the QIAseq run on conversion (0.920), as designed.

### 2026-10-09 · DEV-IAMA-KIT-01: Swift vs TruSeq, read-end clipping
\measured Clipping 10-15 bp from read ends lowers both kits (Swift 1.167→1.155, TruSeq 1.047→1.021); the 0.12 kit gap stays. Not a
read-end artefact. Swift's extra errors cluster along the genome (IAM-A C 306 vs 45). Next: compare both kits on the regions both cover.

### 2026-10-09 · DEV-IAMA-KIT-01 parts 2-3
\measured On the same 2.25 M positions Swift reads 1.159 and TruSeq 0.994; the gap is in the molecules, not in where reads land. Lower
conversion explains at most a quarter of it. \observed Loyfer (where P comes from) also used Swift, so this is a laboratory-and-kit offset
on IAM-A, like Met-A's before the tare. Options (tare / per-kit P / wider band) go to the author.

### 2026-10-09 · Box Run 2 session 2 complete: IAM-A on another laboratory's healthy neutrophils, 2 donors × 4 kits
25M read pairs per run; pinned Loyfer pipeline; Stage Q0; Stage Q with P v2. Table: `doors/data/BOXRUN2_SESSION2/session2_table.csv`.

| kit / sequencer | donor 6 IAM-A (halves) | donor 7 IAM-A (halves) | conversion |
|---|---|---|---|
| Swift / NovaSeq | 1.159 (1.160 / 1.157) | 1.160 (1.162 / 1.158) | 0.983 |
| Swift / HiSeq X | 1.167 (1.168 / 1.167) | 1.159 (1.159 / 1.158) | 0.983 |
| TruSeq / HiSeq X | **1.047 Normal** (1.046 / 1.048) | **1.042 Normal** (1.043 / 1.041) | 0.992 |
| QIAseq / HiSeq X | stopped at Q0 (conversion 0.920) | stopped at Q0 (conversion 0.909) | – |

\measured IAM-A is highly repeatable: halves of a run agree within 0.004; the two donors agree within 0.009 for each kit (largest 0.0086); the
sequencer makes ≤ 0.009 difference (largest 0.0087). \measured The library kit makes a 0.12 difference on the same cells (Swift 1.16 vs TruSeq 1.04-1.05), and this
laboratory's Swift reads 0.16 above the Loyfer Swift libraries P was measured on (DEV-IAMA-KIT-01). Q0 stopped both low-conversion runs,
as designed. Duplicate fractions now recorded (0.11-0.19). The box stopped itself at 06:30 UTC; the scratch disk is deleted.

### 2026-10-09 · Author decisions; C-score blocks of 10; IAM-A same-run tare wired
- Author approved (1) C-score blocks of 10 and (2) option (a), IAM-A read against same-run healthy references.
- C-score: `neutrophil_reference_v1_2.json` (block 10; baseline re-measured the v1_1 way, median 1.0103, leave-one-out 0.964-1.056; block-50
  baseline reproduced first). Met-A unchanged (6 arrays identical to job B). Band unset until step 1 reruns (running).
- IAM-A: `stage_q_iam_a.tare()` and `run_sample.py --iama-ref-table`, same rule as Met-A (>= 3 references). Constructed checks: a Swift run
  against three other Swift runs reads A_rel 1.007 Normal; one reference only -> untared with the reason; a reference from another pipeline
  is refused. \observed No public set yet gives >= 3 healthy neutrophil donors per laboratory and kit at read level (GEO search 2026-10-09:
  GSE128731 has 2 donors per kit; Loyfer 3 granulocytes, Swift; BLUEPRINT is controlled access), so the tare cannot yet be tested on
  independent people. That is the data the IAM-A commissioning needs.

### 2026-10-09 · CHAIN CHANGE · Documents consolidated (author approved)
- Hand-updated documents are now three: this LOG, [`STATUS.md`](../Biological_Physics/MethylPhys/STATUS.md) (plan + commissioning record,
  merged from `doors/PLAN.md` and `doors/CHAIN_COMMISSIONING.md`), and the SOP/OM (chain behaviour only). The chain changelog is merged into
  this LOG. The three merged files are archived with pointers in `Biological_Physics/MethylPhys/archive/docs_consolidated_2026-10-09/`.
- Generated, never hand-edited: chain map (`chain/build_chain_sequence.py`), frozen-input list (`kit/build_frozen_inputs.py`), data
  register (`kit/build_data_register.py`: S3 sizes; tests from this LOG's headings; notes written before today kept as `earlier_tests`).
  From now on every LOG entry names its datasets by accession so the register finds them.
- Release check D1 (every frozen input named in the SOP) and D2 (every frozen input named in this LOG; STATUS.md present).
- Frozen inputs in force today (introductions are in the archived changelog): `metA_floors_v1_3.json`, `metA_floors_v1_3_loo.csv`, `blood_composition_EPIC_v1.json`, `noise_sites_EPIC_v1.json`, `noise_gate_EPIC_v1.json`, `intake_thresholds_v1.json`, `iama_positions_v2.json`, `hg19_cpg_chrom_ranges.json`, `neutrophil_reference_v1_2.json`.
- STATUS.md section 2b: IAM-Atlas v2 commissioning as the composition step (author: "if it works better than what we are currently using we
  should use it").
- SOP/OM: C-score blocks of 10 (`neutrophil_reference_v1_2.json`) and the IAM-A same-run tare added.

### 2026-10-09 · C-score with blocks of 10: steps 1-3 rerun (CHAIN CHANGE: development band 0.877-1.152)
\measured 641 healthy arrays, 19 laboratories (GSE110530, GSE112618, GSE118144, GSE122244, GSE123914, GSE141682, GSE142512, GSE161678,
GSE166503, GSE200376, GSE222927, GSE225544, GSE226298, GSE235717, GSE247193, GSE247195, GSE250556, GSE276323, GSE286313). Band 0.877-1.152
(was 0.750-1.409). Step 1 94.5 % (bar 95 %; GSE110530 and GSE226298 below 85 %); step 2 clustered 100 % above, scattered 94.2 % inside
(bar 95 %); step 3 within ÷ healthy 0.85 (bar 0.5). Not commissioned. Two definition questions for the author in the plan.

### 2026-10-09 · Box Run 3, test 1: commissioned Met-A on clonal haematopoiesis (GSE315366)
\measured 64 EPIC v1 peripheral bloods of a new laboratory. CH-negative 34/35 Normal (bar met). CH-positive vs negative: medians 1.0013 vs
1.0011, p = 0.41 (prediction not met; clones likely below the 5 % whole-blood detection limit, stated before reading). C-score test (a)
on a new laboratory: 34/35 not above 1.152 (met). Session 3 pUC19 (six neutrophil BAMs, 25 M pairs each): 0, 0, 29, 0, 0 and 4 reads
(SRR9888330, 332, 333, 334, 336, 337); the 29 reads of SRR9888333 are 2.5 % methylated (6 of 238 CpG calls), so the pUC19 present is
unmethylated, not a methylated spike: no internal technical standard there (DEV-IAMA-XCELL-01 check 4: recorded null). Details: `boxruns/run3/JOBS.md`.

### 2026-10-09 · Box Run 3, reading 2: leukaemia serial bloods (GSE315367, development)
\measured 30 peripheral bloods, 0 errors. **Diagnosis bloods: 10 of 10 not read** — neutrophil fraction
0.00-0.16 (blast-rich blood), below the 0.20 scope rule, so Met-A correctly gives no reading. Remission bloods: 19 of
20 read, untared A 0.933-1.066 (median 1.014); one remission blood (Patient 7, Rm2, fraction 0.06)
not read. Without healthy references from this run no state is given, and diagnosis cannot be compared with remission on neutrophils.
The record field `refusal` was empty for the unread arrays (the reason sits in Met-A's state text); the script will copy the state next time.

### 2026-10-09 · DEV-XSPECIES-TEMP-01: copy error vs body temperature across 44 mammal species (RRBS, reference-free reader)
\measured Liver ρ 0.29 (p 0.028), heart ρ 0.12 (p 0.22): primary test (both tissues) not met; slopes +6.3 and +4.1 %/K with 95 % CIs
that include the predicted +1 %/K and zero — not resolved over 5 K. Birds read 33–37 % above mammals (lineage). Next: a wider temperature
range within one lineage (fish water temperature; hibernation). Reader matches a direct count exactly on constructed reads.

### 2026-10-09 · Synthetic checks (author: "test first")
\calculated (1) Derivation IAM-A -> Met-A: a synthetic patient read both ways caught an error — the first version assumed every added loss
is one copy error; measured on real healthy molecules by Stage Q's rule only 0.53 of it is. Corrected table: Met-A moves 0.55-1.1 x as far
from 1 as IAM-A; synthetic Met-A matches the curves within 0.001. (2) Temperature design: today's 44 species over 5 K had power 0.11 for
+1 %/K; one species over >= 20 K with >= 40 animals gives 0.85.

### 2026-10-09 · DEV-IAMA-XCELL-01: IAM-A kit offset is the same for CD4 T cells and neutrophils
\measured P_CD4 = 1.167 (Loyfer). Swift/TruSeq ratio CD4 1.109-1.116 vs neutrophils 1.112-1.115 (difference 0.001, bar 0.03, met); after
the tare the kit gap is 0.002-0.004 (met). Cross-cell tare reads neutrophils 0.94 (0/6 Normal, not met): the CD4-to-neutrophil level differs
~6 % between laboratories. Conclusion: same-run references must be of the same cell type. Whole-blood test (same-type) next.

### 2026-10-09 · Thermal-floor test across cell types withdrawn before running; fast Stage Q counter
A proposed test ("no healthy cell type reads below ε₀ = 0.032") is not valid: ε₀'s height is the MEAN holding energy measured on Loyfer's
56 cell types (PROC-CHANNEL-01, 3.41 ± 0.12 kT), whose copy errors span 0.024-0.042, so about half sit below it by construction. ε₀ is a
calibrated reference height, not a lower bound. A physics-only height (e.g. from DNMT1's discrimination energy) is open, and would make the
floor a prediction. `chain/pat_eps_fast.py`: Stage Q's whole-file ε streamed, identical to Stage Q on GSM5652279 (4,995,597 / 129,539,188;
52 s vs ~20 min).

### 2026-10-09 · DEV-FLOOR-HEIGHT-01: enzyme-physics height for the de novo channel, first test — not met
\measured Lambda spike-in non-conversion TruSeq 1.05 %, Swift 1.63 %; corrected de novo error 0.0030 and −0.0011 vs predicted
0.0055-0.011 from DNMT1's single-step HM/UM gap (87-180x). Cells exceed the enzyme's single-step fidelity (multi-step in cells).
Methylated-channel (3.41 kT) height: still no physics-only derivation.

### 2026-10-09 · Fish water-temperature reading (development)
\measured 56 fish species, 0.2-28 °C: gills +0.45 %/K (−0.38 to 1.28), muscle +0.39 (−0.60 to 1.39), heart +0.45, liver −0.65; all CIs
include +1 %/K except none excludes 0. Undecided. A decisive design needs one species over ≥ 20 K (≈ 40 animals).

### 2026-10-09 · DEV-METAA-450K-01: Met-A on 450K purified neutrophils — bars 1, 2a, 3 met
\measured Reference: 8 GSE88824 healthy neutrophils, 6,000 sites by the canon rule; held-out SD 0.0336 raw, 0.0177 after the rebuilt
450K self-tare (bar 0.020). Other lab GSE124565: 12/12 healthy Normal after the same-run tare. Detection limit: 2 % loss 12/12 outside
Normal. Lupus set has no IDATs (dropped). Second lab (GSE224807, 65 healthy) downloading; APS read only after it.
Box: first Swift whole-blood run stopped by Q0 conversion (0.97984 vs ENCODE limit 0.98), rule kept.
- 450K bar 2b NOT met: GSE224807 64 healthy CD15 neutrophils 43/64 Normal (arrays on 47 slides, no same-slide references; noise gate built,
  withholds none). 450K stays development; APS not read. Next: test slide contrast and CD15 sort purity.

### 2026-10-09 · Book sync: the floor and the healthy reference (author ruling)
The gauge has three marks. **Floor H_min**: thermal kicks win against one full ATP per site, copy error 1/(1+e^M) = 8.1×10⁻¹⁰,
H_min = 2.5×10⁻⁸ bits (1×10⁻⁷ on the neutrophil IAM-A gauge, 8×10⁻⁸ on Met-A); calculated from M alone. **Healthy reference H_ref**:
H(ε₀) = 0.2043 bits, ε₀ = 0.032, the holding energy 3.41 kT measured across 56 healthy cell types; about half of healthy cell types hold
better than it, so it is a reference height, not a limit (formerly mislabelled "the floor"). **Ceiling**: 1 bit per site. φ = 0.16 is
the distance from floor to reference in energy. Each instrument has its own healthy reference per cell type: IAM-A neutrophils
P·H_ref = 1.1492 × 0.2043 = 0.2348 bits, CD4 T 1.167 × 0.2043; Met-A EPIC neutrophils 0.330263 bits. Values from the whole-file position
(P 1.1492): full surface IAM-A 4.26 (was 4.45), H_ref at 1/P = 0.870 (was 0.910 and called the floor), healthy neutrophil ε 0.0384,
donors on H_ref alone 1.137–1.168, leave-one-out 0.984–1.025, CV of P 0.74 % (the book's 1.2 % did not match the three whole-file
donors; iama_positions_v2.json carries cv_across_donors 0.012, metadata only, not read by the chain), 2 % test 1.273–1.313 at
whole-file positions (shift measured on the first 60 MB). Book: Part III, VI, VII, appendices A (regenerated from CANON), B, C, D, F,
H, saturation; CANON iam_canon.json (H_min_cell, H_ref_cell added; P 1.1492) and GLOSSARY.md; five figure scripts; verify_book.py
checks moved to whole-file counts, 13 new floor checks, every control fails as it should. SOP: gauge marks and P. No chain change.

### 2026-10-09 · DEV-METAA-450K-01 bar 2b diagnosis: slide/processing batch
**Cause: slide / processing batch, not a mixed-in cell type.** The 17 slides holding two of these neutrophil arrays hold two different
people (consecutive sample numbers, e.g. F186/F187, F102/F103). Their tared readings agree to a median |difference| of **0.013**; two
arrays from different slides differ by **0.051** (one-sided p < 0.001). A second cell type mixed in at varying amounts would differ
person by person, not slide by slide. One direction carries the spread: the first component of the identity-site residuals holds 17 %
of the variance and tracks the reading (r = 0.89), and it lowers the methylated identity sites while leaving the unmethylated ones
(a shift of the methylated channel, as a processing offset gives). The self-tare on invariant sites does not remove it.
**What it means for 450K.** Where same-slide healthy references exist (GSE124565), 12/12 read Normal; where they do not (this series,
1–2 neutrophil arrays per slide), the series-level tare cannot remove a slide offset. 450K Met-A stays development. Rule to test before
use: a 450K reading needs ≥ 3 same-cell-type healthy references on its own slide (the EPIC Stage T rule). Next: a third laboratory with
same-slide healthy neutrophils, scored by bars 1–3 as written; APS and the disease readings wait on it.

### 2026-10-09 · Deviation-load reading on real healthy 450K arrays (DEV-SYNTH-LEVERS-01 simulation 7)
Reading: number of sites where a self-tared array departs from the healthy mean by |z| > cut, z on a shrunk per-site healthy SD, after
removing the array's own offset and contrast. Disease = a signature of k sites moved by Δβ added to a real healthy array.
**Against another laboratory's reference (GSE88824, 8 arrays): unusable.** Healthy arrays of GSE124565 already depart at a median 6,990
sites (max 10,889) and GSE224807 at up to 28,006: laboratory differences swamp any focal disease (caught 1/12 even at 3,000 sites).
**Against same-run healthy references (leave-one-out within GSE124565, 12 arrays):**
| cut | healthy load | 300 sites, Δβ 0.2 | 300 sites, Δβ 0.1 | 1,000 sites, Δβ 0.1 |
|---|---|---|---|---|
| 5 | 360–1,208 | 2/12 | — | — |
| 6 | 245–672 | 3/12 | — | — |
| 8 | 150–302 | 12/12 | 10/12 | 12/12 |
| 10 | 113–186 | 12/12 | 10/12 | 12/12 |
**What it sets.** The reading works only against same-run healthy references (the Stage T rule again), with a strict cut. The cut (8)
was chosen on these 12 arrays, so it is a development value: it must hold on a laboratory not used to choose it (false-trigger rate on
its healthy arrays, then a constructed signature), before any disease is read with it. Synthetic signatures are random sites; real
disease signatures cluster in regions, which the C-score reads.

### 2026-10-09 · SAM and one-species temperature candidates (search + simulation, no download)
**SAM lever: GSE77079** (mouse liver RRBS, one laboratory, raw reads public SRX1539708-…): Mat1a knockout, liver SAMe depleted, placebo
(6); knockout given SAMe (5); wild type (8). Reference-free reader (rrbs_iama.py) applies. Simulation (restore ∝ SAM/(K_m+SAM),
K_m 4.4 µM, read by the chain's rule): IAM-A rises 1.025 / 1.05 / 1.10 for a 1.5 / 2 / 3-fold SAM drop at 60 µM, 1.13-1.25 at 20 µM.
Power, knockout vs wild type with the within-species spread of the 580-species liver reads (SD of ln ε 0.136): 0.42 at IAM-A 1.10,
0.97 at 1.25. **Decision rule before download:** the paper's measured liver SAMe fold drop and the wild-type within-laboratory spread
must give power ≥ 0.8; the SAMe-treated knockouts are the built-in reversal (they must move back toward wild type). Confounds: the
knockout develops steatohepatitis (cell mix, proliferation).
**Temperature, one species: GSE199815** (Syrian hamster liver WGBS, 3 euthermic, 3 late torpor, 3 early arousal). The two pictures of
the bit predict opposite outcomes: a passive bit held at the current body temperature would lower ε by ~30 % in torpor (≈ 30 K colder;
power 3 v 3 = 0.67); a driven bit renewed at copying predicts almost no change during a torpor bout, because liver cells barely divide
in it (power to see the small residual 0.05-0.08, i.e. a null; `development/sims/hibernation_01.py`). A clear fall in torpor would favour the passive picture; no change is
what the driven-bit derivation (DEV-FLOOR-HEIGHT-02) predicts. Labelled PREDICTION (driven bit: no torpor change; change only after
renewal). Needs a hamster read pipeline and conversion control; 9 WGBS runs.
**Not useful:** GSE152444 (sea bass; 4 K during development, read three years later: a memory test, and 4 K is below detection).

### 2026-10-09 · Atlas truth sets and tumour both-ways candidates (search, no download)
**Atlas truth sets.** Every EPIC set with known cell fractions found (GSE112618 FACS bloods, GSE182379 and GSE180970 constructed
mixtures, GSE77797 450K reconstructed mixtures) comes from one research group (Dartmouth), which also built the reference arrays
GSE110554/GSE167998 behind several atlas cells. A truth set from that group cannot show the atlas holds on a laboratory it has not seen.
Still needed: blood with laboratory cell counts or constructed mixtures from a second group. Not found in GEO by title; next: search
by characteristics fields (e.g. 'neutrophil %', 'lymphocyte count') on EPIC whole-blood series.
**Cancer fingerprint (both instruments on the same cells): GSE86833.** Prostate cancer line LNCaP and normal prostate epithelium PrEC on
EPIC and 450K (2 replicates each) and WGBS (LNCaP 5, PrEC 4), one laboratory. Prediction to be simulated before download: on the
same cells, Met-A departs further than IAM-A (fingerprint). Limits written now: Met-A has no prostate-epithelium reference, and two
PrEC arrays are fewer than the three same-run references the tare needs, so Met-A can only be read as LNCaP ÷ PrEC (development);
IAM-A has no measured P for prostate epithelium, so it too is read as LNCaP ÷ PrEC (the 4 PrEC WGBS runs give the same-run reference).
Cell lines, not patients.

### 2026-10-09 · DEV-SAM-LEVER-01 written before download (GSE77079; SAM down 74 %, predicted IAM-A 1.03–1.26)
Bars, power and confounds in doors/DEV_SAM_LEVER_01.md. Box job when the box is next started.

### 2026-10-09 · Truth-set, cancer-reference and repeatability candidates
**Atlas composition, second-group truth sets: bronchoalveolar lavage with differential cell counts** (cytospin % lymphocytes,
neutrophils, eosinophils per sample): GSE133062 (EPIC, 70, Karolinska; healthy smokers/non-smokers), GSE151017 (EPIC, 78, Karolinska;
MS), GSE206709/GSE206719 (EPIC, 72, National Jewish; chronic beryllium disease). Atlas v2 holds every cell these need: lung alveolar
macrophages (2) and interstitial macrophages (3), lymphocytes, neutrophils, eosinophils. Two groups, neither behind the atlas's
references. Simulate first: atlas mixtures at the counted fractions plus the per-site healthy spread, with the counting error of a
~400-cell differential (± 2–3 points), to set whether the composition bars can be met on them.
**Cancer both ways (GSE86833) — reference upgrade:** prostate epithelium is IN atlas v2 (4 Loyfer WGBS), so Met-A has prostate identity
sites, and the 4 Loyfer prostate .pat files can measure IAM-A's own position P for prostate epithelium (box job). Simulated resolving
power of the both-ways test (2 arrays and 4 runs per side; Met-A held-out spread 0.020, IAM-A repeat 0.009; curve B slope 1.17):
an excess of Met-A above curve B of 0.04 is detected 58 %, 0.06 87 %, 0.10 100 %; false call 5 % (`development/sims/fingerprint_power_01.py`).
**EPIC Met-A repeatability, new laboratory: GSE247198** EPIC neutrophils, 2 people × 24 arrays across one day with technical repeats,
6 slides (its 450K half is neutrophil-depleted blood, not usable for DEV-METAA-450K-01). Healthy only; a further-laboratory check of the
commissioned EPIC Met-A (Normal, repeat spread), and a time-of-day reading.

### 2026-10-09 · Lavage truth sets simulated against the atlas composition bars (reachable; download next)
Atlas v2 posterior draws, 60 random blocks (66,954 loci; the 8,000 most cell-informative used), an 11-cell lavage panel (alveolar and
interstitial macrophages, monocytes, CD4, CD8, B, NK, neutrophils, eosinophils, alveolar and bronchus epithelium). Truth: macrophages
~70–95 %, lymphocytes 4–20 %, neutrophils 0.5–12 %, eosinophils 0–2 %; each 'person' one posterior draw plus spread; counted fraction
from a 400-cell differential. Solver: non-negative least squares on posterior means (a stand-in; the real reading uses atlas_e).
| conditions | neutrophil MAE vs truth | MAE vs the count | within 0.05 of the count |
|---|---|---|---|
| array SD 0.02 | 0.004 | 0.010 | 300/300 |
| array 0.04, person 0.02 | 0.005 | 0.011 | 299/300 |
| array 0.06, person 0.04 | 0.006 | 0.011 | 299/300 |
| + random per-site lab offset SD 0.06 | 0.010 | 0.013 | 299/300 |
| + methylated-channel shift 10 % | 0.005 | 0.011 | 299/300 |
Reproduce: `development/sims/atlas_sims_01.py` (60 atlas blocks, checksums in `atlas_blocks_60.json`).
The count's own error (binomial, 400 cells) is most of the 0.010; bar 1 (MAE ≤ 0.02, every sample within 0.05) is reachable. Simulation is
optimistic: EPIC probe overlap with the atlas loci, atlas_e's own solver, and cell states in disease (MS, beryllium disease) not modelled.
**How it counts (most defensible):** lavage is lung, not blood. A pass is independent-laboratory evidence that atlas v2 measures
neutrophil fraction against a macrophage background; it does not by itself commission the whole-blood composition step, which still
needs a whole-blood truth set from another laboratory (GSE122126). **Decision:** download GSE133062 first (healthy; Karolinska), then
GSE206709 (National Jewish), read with atlas_e unchanged, bars 1–2 as written. Differential cell count method and cells counted to be
taken from each paper before scoring.

### 2026-10-09 · Atlas truth sets: lavage needs a lung panel (not atlas_e); GSE122126 is not a neutrophil truth set
1. **atlas_e holds the 12 circulating blood cells only** (chain/dev_stages.py ATLAS_E_CELLS: no macrophages, no epithelium). Lavage samples
   (~85 % macrophages) cannot be read by atlas_e unchanged; the lavage simulation above used an 11-cell lung panel, a different method.
   Lavage sets therefore test atlas v2's templates in the lung, not atlas_e, and do not count toward bars 1–2.
2. **GSE122126 (Moss 2018) is not a neutrophil truth set.** Its 9 genomic-DNA mixes put liver, lung, neuron or colon DNA at 0–10 % into one
   healthy donor's leukocyte DNA (paper, Fig. 3; Supplementary Data 1): the known quantity is the tissue fraction; the leukocytes' own
   neutrophil share is not given, and atlas_e has no tissue cells. Removed from the list of independent sets for bars 1–2.
**Still needed for bars 1–2:** EPIC whole bloods with flow or differential counts, or physical mixes of purified blood cells, from a
laboratory other than Salas (Dartmouth). Sample-field search of GEO (986 EPIC/450K samples with count fields) found none outside that
group. Next: the same search on ArrayExpress and on EPIC v2 (GPL33022) series, and published papers' supplements.

### 2026-10-09 · DEV-COMPOSITION-TRUTH-03 written before download (GSE224807 paired blood + sorted cells, lab-template truth)

### 2026-10-09 · Reproducibility audit
Repo checkout clean (no uncommitted or unpushed changes). 25 working scripts compared with the repo by content: 8 identical, 17 missing
(launchers, the 450K calibration, the non-conversion estimate, Q0 healthy controls, cross-species run scripts, the paired-blood calibration):
all 17 committed next to their notes. The full working code record of the sessions 2026-09-18 to 10-09 (8,629 cells) is archived privately
(S3 archive/session_code/, sha256 5cf3bd35157fd7bd…; key IDs redacted; private names present, so not public). Rule added to the SOP.
Next: each analysis run only in a notebook gets a committed script that reruns to the logged value (development/sims/).

### 2026-10-09 · Lever simulations as committed scripts
`development/sims/sam_lever_01.py` and `hibernation_01.py` rerun from the repo alone. Two logged ranges corrected to the scripts: SAM power at twice the same-lab spread 0.31–1.00 (was 0.52–0.98, taken from 5 of the 9 cases); torpor residual power 0.05–0.08 (was 0.06–0.07, simulation noise). No decision changes.

### 2026-10-09 · Atlas and fingerprint simulations as committed scripts
`development/sims/atlas_sims_01.py` (atlas blocks by checksum) and `fingerprint_power_01.py` rerun from the repo; notes now carry the scripts' values (last-digit differences from the notebook runs: lavage with lab offset 0.013 vs 0.010–0.012, 299/300 within 0.05; own-cell lab-template truth 0.003–0.005, max 0.013; fingerprint 58 %/87 % vs 59/88). No decision changes.

### 2026-10-09 · 450K analysis as a committed script
`doors/data/DEV_METAA_450K_01/metaa_450k_01.py`, inputs pinned by sha256 (GSE224807 CD15 betas and the 450K manifest uploaded to S3 for this). Reproduced exactly: identity sites 6,000, healthy reference 0.32581, held-out 0.0336 raw and 0.0177 tared, 12/12 Normal (0.983–1.028), detection 1/2/3 %, 43 of 64 on GSE224807, slide pairs 0.013 vs 0.051, PC1 r 0.89, deviation loads 6,990/10,889/28,006, z>8 12/12 and 10/12. Two edge cells of the deviation table move by one array with the fixed seed (z>5 3/12, z>10 at Δβ 0.1 9/12); no decision changes.

### 2026-10-09 · PROC-CHANNEL-01 not reproducible from the repo; job rebuilt
Outputs (channel_samples.csv, channel_summary.json) were never committed and are gone from the box. E_hold 3.41 kT here is the source of eps0_meth 0.032 (CANON). channel.py rebuilt by replaying its creation and four patches from the session record; windows regenerated (seed 20260930, 399 windows); roster in atlas/v2/inputs. To rerun on the box after the current sessions (about 1 h, 64 cores); outputs to be committed with the note.

### 2026-10-09 · Reproducibility gate on every push
CANON/repro_check.py, called by CANON/checked_push.sh: refuses untracked files and changed doors/ notes with numbers but no committed script. Tested: refused an unbacked test note and a stray file. Cause of the gaps found tonight: scripts and outputs kept in the working area instead of the repo, the 10-01 catch-up push did not check notes written before it, and tonight's pushes used plain git push, which skipped the canon gate.

### 2026-10-09 · PROC-CHANNEL-01 per-cell table recovered and committed
The author's 09-30 copy holds channel_cells_genomewide.csv; committed to doors/data/PROC_CHANNEL_01/ with derive_constants.py, which reproduces E_hold 3.41 ± 0.12 kT, phi 0.163 (CANON 0.1628 = 3.41/M), de novo 4.37 ± 0.13, eps0 0.032 (1/(1+e^3.4105) = 0.03197). The range top was printed 3.72; the table gives 3.7146, so 3.71 (note and book p6_02 corrected). 31 book checks that read the note now compute from the table; all pass, controls fail as they should, no new failures. The chr1 first-pass table is committed beside it, labelled. Open: rerun channel.py to show the rebuilt job makes this table and to regenerate the per-sample table (needed for section 2 clustering).

### 2026-10-10 · DEV-COMPOSITION-TRUTH-03 read: undecided (truth not precise enough)
atlas_e vs the GSE224807 lab-template truth: 450K MAE 0.041 (23/30 within 0.05), EPIC 0.076 (Stage A 0.071, agreeing with atlas_e). Truth unstable to site choice (up to 0.048), whole blood fits the six templates with RMS 0.064, sorted CD14 carries 0.175 granulocyte signal. Neither passes nor fails bars 1-2; counted-cell truth still needed. Scripts doors/data/DEV_COMPOSITION_TRUTH_03.

### 2026-10-10 · 39 outputs from the author's 09-30 copy committed
Placed beside their notes (11 notes) or in doors/data/RECOVERED_2026-09-30/ with a README of producing time and sha256. Outputs only: backlog rows stay open until each script is recovered and reproduces its file.

### 2026-10-10 · Dose series Met-A side read and committed before IAM-A
Met-A_rel vehicle 1.000 (0.9986/1.0014), DAC30 1.43, DAC300 1.67; methylated identity mean beta 0.898 -> 0.704 -> 0.549. Script metaa_dose_02.py.

### 2026-10-10 · Dose-series window corrected before data: Stage Q response measured in silico
Stage Q reading rises about half as fast as the simple copy-error form (delta 0.10: IAM-A_rel 1.55 vs 2.21; 74 % read). 30 nM window now IAM-A_rel 1.44-1.55 with >= 70 % read (stand-in response; final on the vehicle molecules); 300 nM outside the readable range. Scripts insilico_loss_02.py, predict_iama_window_02.py.

### 2026-10-10 · PROC-CHANNEL-01 reproduced from the repo
Rerun of the rebuilt job (153/153 samples after re-fetching 2) matches the 09-30 per-cell table exactly in every column; clustering numbers reproduced (2 groups silhouette 0.531; held-out 4-5 groups ARI 0.505/0.516; eight groups vs classes 0.234, null 0.045). E_hold 3.41, phi 0.1628, eps0 0.032 now fully reproducible from public data. Closed on the backlog.

### 2026-10-10 · DEV-IAMA-WBTARE-01 read: incomplete
9/14 runs refused at conversion 0.98 (all Swift at 0.979-0.980, Sample3 Swift rep2 passing at 0.98001; TruSeq rep2 0.970-0.974). Bar 1: 4/4 TruSeq rep1 within 0.95-1.05 after the tare (1.024, 0.972, 0.979, 1.022). Bars 2-3 not assessable, bar 4 not run. Correction to the 10-09 status messages: not all Swift libraries failed intake (one passed), and the TruSeq repeat libraries failed it.

### 2026-10-10 · noise_sites_EPIC_v1 reproduced from public data
build_noise_sites_01.py: 91 Salas purified arrays (GSE110554, GSE167998) from GEO IDATs through chain Stage 1, the 10-01 rule -> 48,528 sites, identical to the chain runtime matrix (low 40,882, high 7,646). Gate N_max 0.149 still to be rebuilt.

### 2026-10-10 · noise gate N_max reproduced
build_noise_gate_01.py: N over the 12 Salas purified neutrophil arrays 0.1223-0.1489 (as DEV-NOISE-01 recorded); N_max 0.149, same as noise_gate_EPIC_v1.json. Both noise matrices now rebuild from GEO IDATs.

### 2026-10-10 · MIN_READ_FRACTION 0.20 does not follow from its stated basis
shift_vs_fraction_01.py (chain reader and matrices only): a 1 % loss shifts whole-blood Met-A by 0.0073 at f 0.20; 0.01 is reached at f 0.27. Reproduces the measured 2 % shifts. Chain unchanged; decision recorded in DEV_LOWFRAC_01_OUTCOME.

### 2026-10-10 · Dose series scored: no result (outside the readable range)
30 nM IAM-A_rel 1.118/1.121 with 60-62 % of molecules read; 300 nM 1.065/1.035 with 34-36 % (rule: >= 70 %). Vehicles 1.005/0.995. Development observation: decitabine strips whole molecules/stretches (run-type loss), not scattered errors; the derivation needs scattered damage (SAM lever). Outcome DEV_LINK_IAMA_METAA_02_OUTCOME.md.

### 2026-10-10 · Whole-blood fraction cut dropped (author decision)
conductor_v3.stage_m_blood reads A at any neutrophil fraction > 0; each reading carries shift_per_1pct_loss at its own fraction and Stage T prints the detection limit. The 0.20 cut did not follow from its basis (0.27 by shift_vs_fraction_01.py). SOP, canon, GLOSSARY, Appendix A updated. Reader check: A read at f 0.05/0.15/0.60 with shifts 0.0015/0.0048/0.023, withheld at 0. Release check: same 14/20 before and after on the laptop (6 environment failures: manifest folder blocked by the sandbox, files outside the partial checkout); book checks: no new failures.

### 2026-10-10 · DEV-IAMA-CONVERSION-01 written before reading
Derivation: eps_meas = c*eps; IAM-A bias ~0.75*(c-c_ref)/c_ref; absolute 0.98 limit does not follow from physics, a conversion difference or the correction eps/c does. Prediction on TruSeq repeat libraries (eps ratio 0.980-0.983).

### 2026-10-10 · DEV-IAMA-CONVERSION-01 read
Dividing eps by conversion shrinks the TruSeq repeat gap 3/3 (direction predicted), within 0.02 in 1/3 (not met). Repeat libraries carry a further ~1-1.5 % library effect (seen in Swift with equal conversion). Chain unchanged.

### 2026-10-10 · RRBS reference-free reader fails; DEV-XSPECIES-TEMP-01 withdrawn; SAM test moves to aligned reads
On wild-type GSE77079: eps 0.307 raw, 0.117 trimmed, 0.085-0.015 as the CpG-calling threshold goes 1-5 reads: fake CpGs from sequencing errors. 18.6 % adapter reads. The cross-species result (liver rho 0.29) used this reader and is withdrawn as a reading. SAM test: Trim Galore --rrbs, bwa-meth mm10, bam2pat, Stage Q.

### 2026-10-10 · DEV-RUNLOSS-01 defined and simulated; prediction sealed
Run-loss reading (excess of fully lost molecules over what the same scattered loss gives) recovers planted run loss (0.099/0.213 for 0.10/0.22) and reads ~0 for scattered loss. Prediction for decitabine: d_eff = array loss (0.22, 0.39) +-0.05; excess L >= 0.5 d_eff.

### 2026-10-10 · DEV-RUNLOSS-01 read: both bars not met; run loss present
Excess run loss 0.12 (30 nM), 0.28 (300 nM), vehicle -0.0001; 39-46 % of total loss (bar: >= 50 %). EM-seq loss 0.30/0.62 vs array 0.22/0.39 (bar +-0.05). Development instrument; needs a second experiment.

### 2026-10-10 · IAM-A position of prostate epithelium measured
Loyfer GSE186458, 4 donors, whole hg19 .pat files (build_position_01.py): eps 0.0426/0.0452/0.0401/0.0365; P = 1.2102 (1.1798-1.2443), CV 2.3 % (neutrophils 1.2 %). Development: not added to iama_positions_v2.json until the cancer both-ways test uses it.

### 2026-10-10 · DEV-ATLAS-LAVAGE-01 read: not met
Neutrophil mean error 0.053 (bar 0.02), bias +0.049; macrophages -0.070; lymphocytes r 0.90 (error 0.041). Only 4,722/8,000 atlas loci on EPIC; lavage macrophages differ from the atlas template (not in the simulation). The lung myeloid templates do not separate neutrophils from macrophages on real arrays.

### 2026-10-10 · Code of 8 tier-1 results recovered from the session record
PROC-TARE-01, CHARR-01, G002-TRACE, HISTORY-01, MATCH-01, PREDX-NEUT-01, TUMOUR-01, WB-NEUT-01: the exact cells (41) as run, committed under doors/data/<name>/RECOVERED_CELLS with a README each. Reruns from the repo pending.

### 2026-10-10 · Chain Stage 1 limit found: later EPIC revision IDATs (1,052,641 addresses) fail in methylprep 1.7.1
GSE206709 (DEV-ATLAS-LAVAGE-02) paused before scoring. Stage 1 support for this revision is a separate development item with its own test.

### 2026-10-10 · DEV-FINGERPRINT-01 step 1 (stand-in prostate molecules)
Stage Q response on prostate: about half the simple rise, readable to delta ~0.12. Run-loss reading on prostate recovers planted runs (0.0997/0.2163) and reads 0 for scattered. Two-arm design (IAM-A arm; run-loss arm if < 70 % read), to be sealed on PrEC molecules before LNCaP is read.

### 2026-10-10 · SAM power redone through Stage Q
Middle case reads 1.037-1.047 through Stage Q (not 1.085). Power 0.90-0.98 if mice spread SD 0.02, 0.42-0.59 at 0.04. Rule: measure wild-type spread first, state power with the sealed window; power < 0.80 makes a null reading undecided. Cross-lab column of sam_lever_01 dropped (withdrawn reader).

### 2026-10-10 · Report provenance fixed
run_sample.py recorded the checksum of iama_positions_v1.json in every report while Stage Q reads iama_positions_v2.json; now v2 (and the hg19 chromosome ranges Stage Q0 reads). Readings unchanged; only the provenance record was wrong.

### 2026-10-10 · Commissioned Met-A matrices rebuilt from GEO
freeze_v13.py rerun on the 6 GSE110554 arrays (chain Stage 1): floor identical, sites identical; reference H mean/SD and clustering baseline identical; LOO precision within 0.0006. Author rule: only values the current chain uses are rerun.

### 2026-10-10 · IAM-A position P reproduced from GEO
build_position_01.py on the 3 public Loyfer granulocyte files: every error and opportunity count identical to iama_positions_v2.json; P 1.1492. The file recorded CV 0.012 (partial-file value); corrected to 0.0074 as the book prints. The chain does not read that field.

### 2026-10-10 · Whole-blood composition matrix reproduced from GEO
chain_tests/blood_comp.py (paths only, mixture test skipped) on the 91 Salas arrays: markers, marker means and profiles identical. Every value the commissioned chain reads now rebuilds from public data with committed code (intake thresholds are author-set lines with their measured basis in PROC-INTAKE-01).

### 2026-10-10 · Public data only
Author rule: no controlled-access requests (no PhD or institutional backing). BLUEPRINT draft moved to archive/retired_2026-10-10; DEV-IAMA-XCELL-01 now points at public same-DNA data (EpiQC).

### 2026-10-10 · Moss mixes: Met-A specific, untared C-score not
In silico from pure arrays: 4-15 % tissue invisible to A and C at identity sites (an earlier build gap artefact withdrawn). Sealed specificity on 9 real mixes: Met-A Normal 9/9 (bar met); C above 1.10 in 3/9 (bar not met), not dose-ordered: array-to-array spread of untared C.

### 2026-10-10 · EpiQC identical DNA, three laboratories: all four bars met
Met-A repeats <= 0.0114 (10/10), tared across labs <= 0.006, raw across labs <= 0.0084; C-score repeats 9/10 <= 0.10, tared across labs <= 0.085.

### 2026-10-10 · SAM lever (Mat1a knockout) PASS by the sealed rule
KO vehicle / WT IAM-A 1.124 (window 1.016-1.135), p 0.0003; SAMe lowers it to 1.098 (p 0.089). No sequencing batch or depth effect found. Open: steatohepatitis is a second route downstream of SAMe loss.

### 2026-10-10 · Record rule added to the push gate
CANON/repro_check.py now refuses a push when an outcome note is not named in this log, a **Milestone:** note is not in the Biological_Physics README Advancements table, or a day lacks its Day summary once the next day has begun (Pacific time). Milestone lines added to DEV-SAM-LEVER-01, DEV-EPIQC-ARRAY-01, DEV-CSCORE-MOSS-01, PROC-CHANNEL-01 and the Met-A commissioning note.

### 2026-10-10 · Status rule added to the push gate
CANON/status_facts.json holds each repeated status fact once; CANON/status_check.py (in checked_push.sh) refuses a push while a living document carries the old wording. First run found six stale statements, all fixed: 'not commissioned' in book p6_18, p6_19 and three glossary entries, and the run_sample header; the 0.20 fraction cut in the commissioning scope printed on every report (conductor_v3). Book check record regenerated on a full checkout: 4,674 checks, 0 failures. Negative controls: an old phrase put back, a required statement removed, a wrong printed count: each refused.

### 2026-10-10 · Results register: book development chapter and README Advancements generated
CANON/results_register.json + results_to_tex.py write docs/book/part6/p6_25_development.tex (new chapter 'Development results': DEV-SAM-LEVER-01, DEV-EPIQC-ARRAY-01, DEV-CSCORE-MOSS-01) and the README table, every number recomputed from committed records. Building it exposed two errors, both fixed: the unmixed-blood reading of DEV-CSCORE-MOSS-01 (1.0035) was not in its rows file (scoring script now writes it), and 0.010 had been quoted as the Moss bar (the sealed bar is 0.03; measured largest shift 0.0094). Negative controls refused: hand edit of the chapter, unregistered commissioning note, covered result still in development, passed result failing its sealed rule.

### 2026-10-10 · Book review against the record (author list)
Ch. Landauer: methylation entropy credited as prior art (Xie 2011, Landan 2012, Hannum 2013, Jenkinson 2017; references from CrossRef) and what IAM adds stated: the gauge, the identity sites, the fixed healthy reference. Ch. gauge: the two ends fixed by physics, only the healthy middle measured. Ch. Met-A: the reference donors (screening from Salas 2018; 5 men, 1 woman, aged 20-39, purity 94-97 %, from GEO) and the definition of healthy; age dependence open. Ch. IAM-A: the 3.41 kT holding energy is rebuilt from public files by one committed script. 6 new checks; full run 4,680 checks, 0 failures. GNMT knockout (high SAM): no public methylome exists (GEO, SRA, web searched); alternatives listed for the author.

### 2026-10-10 · DEV-FINGERPRINT-01 read as sealed: FINGERPRINT
Planted test first (scorer byte-identical: fingerprint called when planted, z 4.00; not called on curve B; Arm B silent on scattered loss). Then LNCaP vs PrEC: IAM-A_rel 1.0673, share read 0.618 (Arm B decides), excess run loss +0.1105 to +0.1130 on every LNCaP run vs +0.0043 largest PrEC: FINGERPRINT. Arm A recorded: Met-A_rel 1.2265 vs curve B 1.1672, z 2.37: also fingerprint. One line, one lab; LNCaP is a cultured line.

### 2026-10-10 · DEV-HAMMER-01 step 1: kinetics in the same cells cannot predict eps0
Simulation of Hammer-seq's measurement (parent = more methylated strand): f and g recovered 6-20x wrong at every time point. Algebra: at steady state g/f = (1-eps)/eps identically, so rates measured in steady-state cells reproduce eps by construction. Closed as a prediction before download; D1 restated as deriving eps0 from the writer's discrimination measured outside the cell. Script development/sims/hammer_01.py.

### 2026-10-10 · Open item D1 restated (author approved)
Book p7_09 (D1), p6_06, p6_08, p6_01 and the canon eps0 source: eps0 is to be derived from the writer's discrimination (hemimethylated against unmethylated sites) measured outside the cell; rates measured in the same cells return g/f = (1-eps)/eps by construction (DEV-HAMMER-01). Appendix A and GLOSSARY regenerated.

### 2026-10-10 · DEV-WRITER-01 sealed before lookup
Model A (single discriminating selection): E_hold = ln D, so 3.41 k_BT needs DNMT1 in-vitro discrimination D = 30.3; bar: median of qualifying measurements in 15-60. Simulation: an independent-site writer has one steady state for all territories, so territories are held by neighbour coupling. Script development/sims/writer_01.py.

### 2026-10-10 · DEV-WRITER-01 scored as sealed: above the band (Model A not confirmed)
Full texts supplied by the author. Qualifying in-vitro DNMT1 discrimination: Adam 2023 ~80, Yokochi 2002 Table II 47.4 (Bashtrykov 2012 excluded: k_cat ratio). Median 63.7 > 60: the writer's single-step limit (eps 0.0155, 4.15 kT) lies below the copy error cells hold (0.032, 3.41 kT); cells lose about twice the writer's errors. DEV_WRITER_01_OUTCOME.md; script data/DEV_WRITER_01/score_writer_01.py.

### 2026-10-10 · DEV-WRITER-02 sealed before reading
Per-context test of whether the writer sets the copy error: log eps_c against log(1/(1+D_c)) over the 256 NNCGNN contexts, slope 1 predicted. Simulation: counting noise negligible (slope sd 0.004), but a D-correlated artefact can fake slope 1; the DIFFERENCE between methylated and unmethylated-territory slopes holds at 1.0 regardless, so that is the statistic. Bar: median difference 0.5-1.5 met, below 0.2 not met. Script development/sims/writer_context_01.py.

### 2026-10-10 · DEV-WRITER-02 read as sealed: NOT MET
Per-context copy error on the 153 healthy window files (reader reproduces PROC-CHANNEL-01 to 1e-16) against Adam 2023's 256 per-context DNMT1 discriminations: copy-error slope -0.240, control +0.437, difference -0.688 (bar 0.5-1.5), negative in 153/153. The copy error does not follow the writer's discrimination; the DEV-WRITER-01 factor of 2 was a coincidence of the genome mean. D1 stays open. Rows and counts committed.

### 2026-10-10 · Evening reproducibility audit (author request)
DEV-FINGERPRINT-02: array list and readability bound moved from notebook cells to committed scripts (make_array_list.py, readability_check.py; both reproduce the notebook outputs exactly); inventory sort made deterministic; REPRODUCE.md gives the run order. atlas/tools/extract_posterior.py replaces the uncommitted builder of the DEV-FINGERPRINT-01 prostate posterior (reproduces it to 4e-8). DEV-WRITER-02: the descriptive Spearman numbers now come from describe_contexts.py.

### 2026-10-10 · Push gate rule 6: number traceability
CANON/repro_check.py now refuses a push when a decimal number in a new or changed note is not carried, at the printed precision, by a committed file of the note's data folder. Run on today's notes before wiring: the only untraced number was a sealed bar constant (1.645), carried by its scorer. Negative control: a typed number in an outcome note refused; the FINGERPRINT-01 and SAM outcomes pass.

### 2026-10-10 · Box Run 9 mapped nothing; lessons index
bwa-meth 0.2.0 calls bwa mem with -T 40 (minimum alignment score); ENCODE HAIB RRBS reads are 36 bases, so no read aligned (flagstat 0.00 % mapped). session9.sh logged each file done and deleted the empty BAM: 7 files, about 80 box-minutes, no output. Stopped. session9.sh now stops the run when a file maps under 40 % or yields no .pat. A lower -T is passed through bwa-meth (extra arguments follow its own -T) and will be chosen by a synthetic alignment test before any rerun. New: Biological_Physics/MethylPhys/LESSONS.md, one index of lessons with the rule and the enforcing check, linked from the SOP.

### 2026-10-10 · Day summary
Passed (sealed before reading): IAM-A SAM lever in mice (1.124 in window 1.016-1.135, p 0.0003); Met-A on identical DNA at three laboratories
(all four bars); Met-A specificity on the Moss mixes (9/9). Reproduced from public data: every matrix the commissioned chain reads.
Not met or undecided: whole-blood kit/repeat bars (incomplete), decitabine dose series (no result: run-type loss), lavage atlas (not met, lung),
run-loss bars (not met), untared C-score on Moss (6/9). Withdrawn: the reference-free RRBS reader and the cross-species reading made with it.
Parked: methionine depletion (PRJDB12471; too few copies made in 48-72 h). Running: cancer fingerprint (LNCaP vs PrEC, both instruments).
Milestones are also listed in Biological_Physics/README.md, section Advancements.

**Afternoon (added 15:00 PDT).** Passed against sealed rules: DEV-FINGERPRINT-01, a cancer fingerprint on both instruments (arm B decides; planted test of the scorer passed first). Push gate extended: record rule, status rule (CANON/status_facts.json), results register (book Ch. Development results and README Advancements generated). Book review: prior art credited, the gauge's physics ends, what healthy means for each instrument side by side, donors of both references (Met-A 20-39; IAM-A 50-56); 4,691 checks, 0 failures. Searches: no public GNMT-knockout methylome; NASH and copier-kinetics sets recorded (Hammer-seq GSE131098). Box stopped.

### 2026-10-10 · Box Run 9 alignment setting chosen by synthetic test
synth_align.py/.sh (rule fixed before running: largest bwa mem -T with unique share >= 0.80, misplaced <= 0.01, Stage Q eps within 5 % of truth): 400,000 simulated 30-36-base directional RRBS reads from hg19, planted copy error. T 16-28: unique 0.84, misplaced <= 0.0013; T 30: unique 0.8184, misplaced 0.0004, Stage Q eps 0.02656 against truth 0.02664 (ratio 0.9971); T 40 (bwa-meth default): nothing aligns, reproducing the failed run. Chosen: T 30. session9.sh now passes -T30 and asserts it in the logged bwa command. Two earlier attempts of the test failed on the job clock (slow read generation) and on bwa-meth parsing '-T 20' as a read file (LESSONS B1b, B7). Correction: commit 4eb508c says smoke-tested locally; that test did not run (no pysam on the laptop).

### 2026-10-10 · Box Run 9 rerun at -T30: first file
ENCFF000MHB (MCF 10A): 46.23 % of reads mapped (synthetic reads 0.8184 unique). Short reads do not explain it: 0.000446 of the first 1,000,000 trimmed reads of ENCFF000MBY (another normal file of this run, same trimming) are under 30 bases; ENCFF000MHB's own trimmed file had already been removed by the run and was not measured. Its .pat: 7,957,709 molecules, 0.146 with >= 6 calls, 84,234 read by Stage Q; Stage Q eps 0.0380 on 380,308 opportunities, in the healthy range. Both sides of every pair go through the identical pipeline; each file's mapping rate is recorded with its score.
