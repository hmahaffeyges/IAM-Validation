# Reproducibility backlog (opened 2026-10-09)

Rule (SOP, 2026-10-09): every number must be reproducible from this repository alone. This list is the notes in `doors/` that carry
computed numbers but have no committed script that produces them (matched by note name and by scripts the note cites). Each item is
closed by a committed script that reruns to the note's values, with inputs pinned (repo path, S3 key + sha256, or public accession),
or by recording that the value cannot be reproduced and withdrawing it from wherever it is used.

Tier 1 = cited by the book or CANON; tier 2 = cited by the chain or SOP; tier 3 = development record only. Pre-registrations carry
bars set in advance; they are checked for any computed number (power, simulated limits) only.

| tier | note | kind | book files | canon files | chain/SOP files | status |
|---|---|---|---|---|---|---|
| 1 | DEV_LOWFRAC_01_OUTCOME | result | 1 | 2 | 3 | open |
| 1 | DEV_NOISE_01_OUTCOME | result | 1 | 0 | 4 | open |
| 1 | DEV_NOISE_02_OUTCOME | result | 1 | 2 | 3 | open |
| 1 | PROC_CHANNEL_01_OUTCOME | result | REPRODUCED 2026-10-10: rerun from the repo matches the 09-30 table exactly (153 samples, all columns) |
| 1 | PROC_CHARR_01_OUTCOME | result | 0 | 1 | 1 | open |
| 1 | PROC_CHARR_01_PREREG | pre-registration (bars) | 0 | 1 | 1 | open |
| 1 | PROC_G002_TRACE_OUTCOME | result | 0 | 1 | 0 | open |
| 1 | PROC_HISTORY_01_OUTCOME | result | 0 | 1 | 0 | open |
| 1 | PROC_MATCH_01_OUTCOME | result | 0 | 1 | 0 | open |
| 1 | PROC_OUTSPAN_01_PREREG | pre-registration (bars) | 1 | 0 | 1 | open |
| 1 | PROC_PARTIAL_01_PREREG | pre-registration (bars) | 0 | 1 | 0 | open |
| 1 | PROC_PREDX_NEUT_01_OUTCOME | result | 0 | 1 | 2 | open |
| 1 | PROC_PREDX_NEUT_01_PREREG | pre-registration (bars) | 0 | 1 | 2 | open |
| 1 | PROC_PREDX_SEQUENCE_01_PREREG | pre-registration (bars) | 0 | 1 | 0 | open |
| 1 | PROC_TARE_01_OUTCOME | result | 2 | 0 | 1 | open |
| 1 | PROC_TARE_01_PREREG | pre-registration (bars) | 2 | 0 | 1 | open |
| 1 | PROC_TISSUE_01_PREREG | pre-registration (bars) | 0 | 1 | 0 | open |
| 1 | PROC_TUMOUR_01_OUTCOME | result | 0 | 1 | 1 | open |
| 1 | PROC_TUMOUR_01_PREREG | pre-registration (bars) | 0 | 1 | 1 | open |
| 1 | PROC_WB_NEUT_01_OUTCOME | result | 1 | 2 | 2 | open |
| 1 | PROC_WB_NEUT_01_PREREG | pre-registration (bars) | 1 | 2 | 2 | open |
| 2 | COMMISSIONING_NOTE_METAA_NEUTROPHILS | result | 0 | 0 | 2 | open |
| 2 | DEV_COMPOSITION_TRUTH_02 | result | 0 | 0 | 1 | open |
| 2 | DEV_CSCORE_TARE_01 | result | 0 | 0 | 1 | open |
| 2 | DEV_DETECTION_01 | result | 0 | 0 | 1 | open |
| 2 | DEV_DIRECTION_02 | result | 0 | 0 | 3 | open |
| 2 | DEV_EPIC_V2_01 | result | 0 | 0 | 3 | open |
| 2 | DEV_IAMA_CSCORE_01 | result | 0 | 0 | 2 | open |
| 2 | DEV_IAMA_P_WHOLE_01 | result | 0 | 0 | 2 | open |
| 2 | DEV_METAA_SENS_01 | result | 0 | 0 | 2 | open |
| 2 | DEV_NEWCELL_01 | result | 0 | 0 | 1 | open |
| 2 | DEV_ROUND2_REPORT | result | 0 | 0 | 1 | open |
| 2 | DEV_SELFTARE_02 | result | 0 | 0 | 5 | open |
| 2 | DEV_SKY_02 | result | 0 | 0 | 2 | open |
| 2 | DEV_STOOL_01_OUTCOME | result | 0 | 0 | 1 | open |
| 2 | DEV_TOOLKIT_ADDED_02 | result | 0 | 0 | 3 | open |
| 2 | PROC_AML_PROG_01_OUTCOME | result | 0 | 0 | 1 | open |
| 2 | PROC_AML_PROG_01_PREREG | pre-registration (bars) | 0 | 0 | 1 | open |
| 2 | PROC_AML_SERIAL_01_OUTCOME | result | 0 | 0 | 1 | open |
| 2 | PROC_AML_SERIAL_01_PREREG | pre-registration (bars) | 0 | 0 | 1 | open |
| 2 | PROC_CEIL_01_OUTCOME | result | 0 | 0 | 1 | open |
| 2 | PROC_CLS_01_PREREG | pre-registration (bars) | 0 | 0 | 2 | open |
| 2 | PROC_CMB_05_OUTCOME | result | 0 | 0 | 1 | open |
| 2 | PROC_DECONV_V2_01_OUTCOME | result | 0 | 0 | 2 | open |
| 2 | PROC_DECONV_V2_01_PREREG | pre-registration (bars) | 0 | 0 | 2 | open |
| 2 | PROC_MF_01_OUTCOME | result | 0 | 0 | 2 | open |
| 2 | PROC_MF_01_PREREG | pre-registration (bars) | 0 | 0 | 2 | open |
| 2 | PROC_MF_02_OUTCOME | result | 0 | 0 | 1 | open |
| 2 | PROC_MF_02_PREREG | pre-registration (bars) | 0 | 0 | 1 | open |
| 2 | PROC_MOLECULE_01_OUTCOME | result | 0 | 0 | 1 | open |
| 2 | PROC_MOLECULE_01_PREREG | pre-registration (bars) | 0 | 0 | 1 | open |
| 2 | PROC_NEUT_TEST_01_T2_OUTCOME | result | 0 | 0 | 2 | open |
| 2 | PROC_PREDX_SLIDE_01_OUTCOME | result | 0 | 0 | 1 | open |
| 2 | PROC_PREDX_SLIDE_01_PREREG | pre-registration (bars) | 0 | 0 | 1 | open |
| 2 | PROC_RIMOUSKI_01_PREREG | pre-registration (bars) | 0 | 0 | 1 | open |
| 2 | PROC_SKY_01_FOLLOWUP | result | 0 | 0 | 1 | open |
| 2 | PROC_SKY_01_OUTCOME | result | 0 | 0 | 1 | open |
| 2 | PROC_SMALL_01_PREREG | pre-registration (bars) | 0 | 0 | 2 | open |
| 2 | PROC_STAGE2D_02_OUTCOME | result | 0 | 0 | 1 | open |
| 2 | PROC_STAGE2D_02_PREREG | pre-registration (bars) | 0 | 0 | 1 | open |
| 2 | PROC_STAGE2D_03_OUTCOME | result | 0 | 0 | 2 | open |
| 2 | PROC_STAGE2D_03_PREREG | pre-registration (bars) | 0 | 0 | 2 | open |
| 2 | PROC_SWITCH_02_OUTCOME | result | 0 | 0 | 1 | open |
| 2 | PROC_V5_HELDOUT_OUTCOME | result | 0 | 0 | 1 | open |
| 3 | DEV_COLON_BLOCKS_02 | result | 0 | 0 | 0 | open |
| 3 | DEV_FLOOR_HEIGHT_01 | result | 0 | 0 | 0 | open |
| 3 | DEV_HORIZON_01 | result | 0 | 0 | 0 | open |
| 3 | DEV_IAMA_WBTARE_01 | result | 0 | 0 | 0 | open |
| 3 | DEV_IAMA_XCELL_01 | result | 0 | 0 | 0 | open |
| 3 | DEV_LINK_IAMA_METAA_01 | result | 0 | 0 | 0 | open |
| 3 | DEV_LINK_IAMA_METAA_02 | result | 0 | 0 | 0 | open |
| 3 | DEV_MOLECULE_COLON_01 | result | 0 | 0 | 0 | open |
| 3 | DEV_Q0_HEALTHY_01 | result | 0 | 0 | 0 | open |
| 3 | DEV_XSPECIES_TEMP_01 | result | 0 | 0 | 0 | open |
| 3 | PROC_AGE_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_BAND_01_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |
| 3 | PROC_BRAIN_01_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |
| 3 | PROC_CLASS_COUNT_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_CLASS_COUNT_01_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |
| 3 | PROC_COV_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_COV_01_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |
| 3 | PROC_DERIVE_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_DERIVE_01_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |
| 3 | PROC_ENCODE_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_ENCODE_01_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |
| 3 | PROC_EPIC_01_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |
| 3 | PROC_FOREIGNSCORE_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_FOREIGNSCORE_01_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |
| 3 | PROC_FOREIGN_01_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |
| 3 | PROC_HMIN_BOOT_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_HMIN_PERCELL_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_HMIN_REFIT_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_LABBAND_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_LABBAND_01_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |
| 3 | PROC_LINES_01_BJ_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_MAHA_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_MAHA_02_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_MAHA_03_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_MAHA_03_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |
| 3 | PROC_PANEL_03_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_PARTIALCOV_01_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |
| 3 | PROC_RECORD_02_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_RECORD_03_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_SCORE_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_SCORE_03_RESOLUTION | result | 0 | 0 | 0 | open |
| 3 | PROC_SWITCH_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_TIER_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_TIER_02_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_UNMIX_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_UNMIX_01_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |
| 3 | PROC_V12_DIAG_01 | result | 0 | 0 | 0 | open |
| 3 | PROC_V12_DIAG_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_V12_IDENTITY_02_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_V12_IDENTITY_02_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |
| 3 | PROC_V12_IDENTITY_OUTCOME_01 | result | 0 | 0 | 0 | open |
| 3 | PROC_V12_IDENTITY_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |
| 3 | PROC_WARBURG_01_OUTCOME | result | 0 | 0 | 0 | open |
| 3 | PROC_WARBURG_01_PREREG | pre-registration (bars) | 0 | 0 | 0 | open |

Closed on 2026-10-09: DEV_METAA_450K_01 (`doors/data/DEV_METAA_450K_01/metaa_450k_01.py`), DEV_SAM_LEVER_01, DEV_SYNTH_LEVERS_01 simulations,
DEV_ATLAS_COMMISSION_01 lavage, DEV_COMPOSITION_TRUTH_03 (`development/sims/`).
