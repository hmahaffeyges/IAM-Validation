# Met-A, IAM-A and C-score development log

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
| 2026-10-03 | [CHAIN_COMMISSIONING.md](../Biological_Physics/MethylPhys/doors/CHAIN_COMMISSIONING.md) | Chain v3 commissioning — stage by stage (development) |
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
| 2026-10-03 | [PLAN.md](../Biological_Physics/MethylPhys/doors/PLAN.md) | PLAN — what we do next, in order |
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
before Met-A). Table: `doors/CHAIN_COMMISSIONING.md` round 4.

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
