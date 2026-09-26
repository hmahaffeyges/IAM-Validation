# kit

The gates and generators. Nothing here is part of a reading: these are the programs that check the chain and keep every derived document true.

## What is in here

| file | |
|---|---|
| [`PROC_ANCHOR_01.py`](PROC_ANCHOR_01.py) | The sealed foundation-cohort anchors reproduce from raw GEO betas (r = 1.00000). |
| [`PROC_BAND_01_analyse.py`](PROC_BAND_01_analyse.py) |  |
| [`PROC_BAND_01_measure.py`](PROC_BAND_01_measure.py) |  |
| [`PROC_BIDIR_01.py`](PROC_BIDIR_01.py) | The directional detector, including re-extraction from the raw 5.1 GB GEO matrix. |
| [`PROC_BRAIN_01.py`](PROC_BRAIN_01.py) |  |
| [`PROC_CACHE_01.py`](PROC_CACHE_01.py) |  |
| [`PROC_CAL_01.py`](PROC_CAL_01.py) |  |
| [`PROC_CLS_01_analyse.py`](PROC_CLS_01_analyse.py) |  |
| [`PROC_CLS_01_b6.py`](PROC_CLS_01_b6.py) |  |
| [`PROC_CLS_01_measure.py`](PROC_CLS_01_measure.py) |  |
| [`PROC_CMB_05.py`](PROC_CMB_05.py) | The sky, commissioned: healthy tail, determinism, refusal paths. |
| [`PROC_COV_01.py`](PROC_COV_01.py) |  |
| [`PROC_DECON_01.py`](PROC_DECON_01.py) | The composition solver against its answer key. |
| [`PROC_E2E_01_run.py`](PROC_E2E_01_run.py) |  |
| [`PROC_E2E_01_score.py`](PROC_E2E_01_score.py) |  |
| [`PROC_EPIC_01_analyse.py`](PROC_EPIC_01_analyse.py) |  |
| [`PROC_EPIC_01_score.py`](PROC_EPIC_01_score.py) |  |
| [`PROC_FOREIGN_01.py`](PROC_FOREIGN_01.py) |  |
| [`PROC_FOREIGN_01_analyse.py`](PROC_FOREIGN_01_analyse.py) |  |
| [`PROC_FORMULA_01.py`](PROC_FORMULA_01.py) | Which aggregation reproduces the seal, measured on both. |
| [`PROC_LABBAND_01.py`](PROC_LABBAND_01.py) |  |
| [`PROC_MAHA_03.py`](PROC_MAHA_03.py) |  |
| [`PROC_MAHA_03_deep_analyse.py`](PROC_MAHA_03_deep_analyse.py) |  |
| [`PROC_MAHA_03_deep_calibrate.py`](PROC_MAHA_03_deep_calibrate.py) |  |
| [`PROC_MAHA_03_stage1_table.py`](PROC_MAHA_03_stage1_table.py) |  |
| [`PROC_PARTIAL_01.py`](PROC_PARTIAL_01.py) |  |
| [`PROC_PARTIAL_01_analyse.py`](PROC_PARTIAL_01_analyse.py) |  |
| [`PROC_PLASMA_MIX_01.py`](PROC_PLASMA_MIX_01.py) | Plasma cfDNA mixture behaviour - reserved specimen, recorded. |
| [`PROC_SEP_03.py`](PROC_SEP_03.py) | Atlas separability by class - the measurement behind the blood caveat. |
| [`PROC_SMALL_01_compare.py`](PROC_SMALL_01_compare.py) |  |
| [`PROC_SMALL_01_figure.py`](PROC_SMALL_01_figure.py) |  |
| [`PROC_SMALL_01_heldout.py`](PROC_SMALL_01_heldout.py) |  |
| [`PROC_SMALL_01_prepare.py`](PROC_SMALL_01_prepare.py) |  |
| [`PROC_STAGE0_02_arrival.py`](PROC_STAGE0_02_arrival.py) |  |
| [`PROC_STAGE0_02_seal.py`](PROC_STAGE0_02_seal.py) |  |
| [`PROC_STAGE0_02_sweep.py`](PROC_STAGE0_02_sweep.py) |  |
| [`PROC_SYNTH_01.py`](PROC_SYNTH_01.py) |  |
| [`PROC_SYNTH_01_cells.py`](PROC_SYNTH_01_cells.py) |  |
| [`PROC_TISSUE_01_analyse.py`](PROC_TISSUE_01_analyse.py) |  |
| [`PROC_TISSUE_01_score.py`](PROC_TISSUE_01_score.py) |  |
| [`ROW9_WORKING_NOTE.md`](ROW9_WORKING_NOTE.md) | Engineering log for the report and interface: what was measured while building it, what failed, and what each failure changed. Read it for the constru |
| [`SMALL_CLASS_DETECTION_NOTE_2026-09-23.md`](SMALL_CLASS_DETECTION_NOTE_2026-09-23.md) | the measured detection limit for a trace class, and why the boundary pins it |
| [`VAL_INDEX.csv`](VAL_INDEX.csv) |  |
| [`add_doc_links.py`](add_doc_links.py) | links code names in prose to the files they name, idempotently |
| [`build_delta_bundle.py`](build_delta_bundle.py) | Builds the chain delta bundle (changed files since a commit) for the author's offline copies. |
| [`build_folder_readmes.py`](build_folder_readmes.py) | generates a README for every folder, listing what is in it - the purpose lines are held in the script |
| [`build_percell_identity.py`](build_percell_identity.py) | Builds iamatlas_percell_identity_loci_v1_0.json from the atlas by the class panels' criterion (kit). |
| [`build_percell_reference_identity.py`](build_percell_reference_identity.py) | Builds percell_reference_identity_v1_0.json: per-cell per-lab A bands on the identity surface, commissioned form, mapped betas, disjoint held-out spli |
| [`build_report_tab_reference.py`](build_report_tab_reference.py) | generates the tab-by-tab report reference and one figure per tab by reading a finished report |
| [`build_reviewer_manifest.py`](build_reviewer_manifest.py) | regenerates the reviewer download list, resolving every path by basename from the tree |
| [`claim_scan.py`](claim_scan.py) | scans the documents for claims and checks each against the sealed record |
| [`claims.json`](claims.json) | the claims the scan found, with their evidence |
| [`cpg_kit.py`](cpg_kit.py) | Shared kit helpers: locates the engine, the runtime matrices and the test data by environment variable. |
| [`dilution.py`](dilution.py) | Dilution-series scoring (record side). |
| [`evaluate_necessity.py`](evaluate_necessity.py) | answers whether a file is necessary, from the tree: runs, imported, named in code, named in a document, or a generator |
| [`finding_check.py`](finding_check.py) | The protocol gate: a finding must be registered, every door taught, and no unqualified detection claim present, or the push is blocked. |
| [`floor_lod_analyse.py`](floor_lod_analyse.py) | Limit-of-detection analysis for the presence floors (record side). |
| [`floor_scan.py`](floor_scan.py) | Scan of per-class presence floors (record side). |
| [`link_check.py`](link_check.py) | every relative path in the live documentation set must resolve - a path in a document is a claim |
| [`recordside_test_disease_matrix_gate.py`](recordside_test_disease_matrix_gate.py) | Proves the removed disease-matching stage fails closed when its files are absent - kept so the removal stays honest. |
| [`release_check.py`](release_check.py) | Every guard in one command; writes release_check.json, which the Safeguards tab prints. Reports a guard that cannot run as SKIPPED, never as a pass. |
| [`test_gauge_switch.py`](test_gauge_switch.py) | The commissioned identity gauge reproduces on the cached commissioning arrays. |
| [`test_lab_zero.py`](test_lab_zero.py) | Recovers a synthetic laboratory offset; refuses panels under 40 arrays. |
| [`test_patient_sky.py`](test_patient_sky.py) | The sky: deterministic mapping, presence-floor masking, and refusal without a commissioned laboratory scale. |
| [`test_percell_physics.py`](test_percell_physics.py) | THE FAILSAFE for the per-cell A - propagate rule 11, runs on every push. Checks the four root causes of 2026-09-26 (wrong surface, unmapped betas, cro |
| [`test_tiers.py`](test_tiers.py) | Every tier boundary in the JSON, both sides, through the one tier function; fails if a literal breakpoint reappears in engine code. |

_67 file(s)._ Paths above are relative to this folder, and [`kit/link_check.py`](../kit/link_check.py) fails the build if any of them stops resolving.

<!-- generated by kit/build_folder_readmes.py - edit PURPOSE there, not this file -->
