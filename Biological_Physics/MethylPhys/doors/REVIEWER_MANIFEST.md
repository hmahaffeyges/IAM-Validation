# What a reviewer can download, and what we have not published

Every path below was resolved from the repository tree when this page was generated, not typed by hand, and
`kit/link_check.py` fails if any of them stops resolving. Sizes are as committed.

## The instrument a reviewer would run

| file | what it is | size |
|---|---|---|
| [`cpg_conductor.py`](../chain/cpg_conductor.py) | the orchestrator: every stage, in the order run_full calls them | 43 KB |
| [`stage_1_idat_calibration.py`](../chain/stage_1_idat_calibration.py) | raw IDAT pair to noob-calibrated beta | 6 KB |
| [`iamatlas_a_scoring.py`](../chain/Runtime Matrices/A_Scoring_Module/iamatlas_a_scoring.py) | H(beta)/H_min over a class's identity loci | 9 KB |
| [`walther_iam_deconvolver.py`](../chain/Walther_iam_deconvolver/walther_iam_deconvolver.py) | Stage 2: the composition the report stands on | 25 KB |
| [`nilc_celltype_deconvolver.py`](../chain/nilc_celltype_deconvolver.py) | the second opinion, compared with Walther at class level | 8 KB |
| [`stage_4_6_patient_cmb.py`](../chain/stage_4_6_patient_cmb.py) | the patient sky | 9 KB |
| [`run_sample.py`](../chain/MethylPhys_Interface/run_sample.py) | the command a reviewer would type, with every flag | 4 KB |
| [`synthetic_patient_generator.py`](../chain/Synthetic_Patient_Generator/synthetic_patient_generator.py) | synthetic specimens with known answers | 26 KB |
## The constants it divides by, and the MCMC that produced them

| file | what it is | size |
|---|---|---|
| [`gape_mcmc_g002.py`](../hmin_calibration/gape_mcmc_g002.py) | G-002: the calibration behind the eight methylation floors - 32 walkers, 5 chains, 500+5,000 steps, R-hat < 1.001, and its 37-cell reference database (_RAW_DB) | 24 KB |
| [`gape_mcmc_g003b.py`](../hmin_calibration/gape_mcmc_g003b.py) | G-003b sampler | 25 KB |
| [`g003_mcmc_framework.py`](../hmin_calibration/g003_mcmc_framework.py) | the framework G-003b runs on | 21 KB |
| [`gape_mcmc_g008.py`](../hmin_calibration/gape_mcmc_g008.py) | G-008 sampler | 16 KB |
| [`gape_mcmc_e_a_bio.py`](../hmin_calibration/gape_mcmc_e_a_bio.py) | the E/A_bio sampler | 15 KB |
| [`gape_mcmc_nbio_ordering.py`](../hmin_calibration/gape_mcmc_nbio_ordering.py) | the ordering of the RETIRED per-class n_bio (rho = 0.905, p = 0.002); the absolute values awaited a run never made | 13 KB |
| [`gape_bootstrap_comparison.py`](../hmin_calibration/gape_bootstrap_comparison.py) | the bootstrap-versus-MCMC script | 13 KB |
| [`bootstrap_vs_mcmc_comparison.tsv`](../hmin_calibration/bootstrap_vs_mcmc_comparison.tsv) | its table: 8 classes x 4 substrates - NUCL, FUZZ, WPS, FRAG. Methylation is NOT in it | 3 KB |
| [`iamatlas_gauge_identity_loci_v1_0.json`](../chain/Runtime Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json) | the identity loci per class and each class's floor | 4.0 MB |
## The runtime matrices every reading is corrected by

| file | what it is | size |
|---|---|---|
| [`beta_scale_maps_v1.json`](../chain/Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json) | pipeline to reference scale, with the rule that it applies BEFORE H | 1 KB |
| [`reference_age_curve_v1.json`](../chain/Runtime Matrices/A_Scoring_Module/reference_age_curve_v1.json) | the per-decade curve: 1,379 healthy donors, four laboratories | 1 KB |
| [`identity_band_v3.json`](../chain/Runtime Matrices/A_Scoring_Module/identity_band_v3.json) | the healthy band, each laboratory's zero and its measured false-alarm rate | 2 KB |
| [`tier_breakpoints.json`](../chain/Runtime Matrices/Tier_breakpoints/tier_breakpoints.json) | the tier words and where they break | 19 KB |
| [`percell_reference_v0_3.json`](../chain/Runtime Matrices/Percell_Reference/percell_reference_v0_3.json) | per-entry markers, resolvability, exclusivity and per-laboratory healthy range | 123 KB |
## The atlas

| file | what it is | size |
|---|---|---|
| [`IAMAtlasREBUILD.csv.xz`](../atlas/IAMAtlasREBUILD.csv.xz) | the posteriors: mean, SD and credible intervals per cell type per address | 100.7 MB |
| [`IAMAtlasREBUILD_celltype_to_class.json`](../atlas/IAMAtlasREBUILD_celltype_to_class.json) | 115 cell types to 8 classes | 3 KB |
| [`IAMAtlasREBUILD_provenance.json`](../atlas/IAMAtlasREBUILD_provenance.json) | where each entry came from | 1 KB |
## The record, including what failed

| file | what it is | size |
|---|---|---|
| [`CHAIN_COMMISSIONING.md`](CHAIN_COMMISSIONING.md) | the commissioning register, row by row, with the rows that closed NOT COMMISSIONED | 9 KB |
| [`CPG_Chain_of_Custody_SOP_v2_0_0.md`](../sop/CPG_Chain_of_Custody_SOP_v2_0_0.md) | the chain-of-custody procedure | 437 KB |
| [`RUNBOOK.md`](RUNBOOK.md) | how to run it, and what it refuses | 31 KB |
| [`CMB_TO_METHYLOME_MAP.md`](CMB_TO_METHYLOME_MAP.md) | every borrowed cosmology tool, with its status | 25 KB |
| [`DETAILED_VALIDATION_RECORD.md`](../../Record/DETAILED_VALIDATION_RECORD.md) | the validation history | 127 KB |
| [`release_check.py`](../kit/release_check.py) | the guards; results/release_check.json carries the last verdict | 7 KB |
| [`claim_scan.py`](../kit/claim_scan.py) | the rendered-claim gate: no document may assert a claim the record has reversed | 2 KB |
| [`link_check.py`](../kit/link_check.py) | every relative path in a document must resolve | 4 KB |
| [`chain_inventory_v1.json`](../chain/Runtime Matrices/chain_inventory_v1.json) | all inventoried files by role | 61 KB |
## History, published as history

| file | what it is | size |
|---|---|---|
| [`README_FIRST.md`](../evidence_pre_atlas_2026-04/README_FIRST.md) | READ FIRST: why the April 2026 database is not a current claim | 2 KB |
| [`evidence_summary.json`](../evidence_pre_atlas_2026-04/evidence_summary.json) | the April 2026 evidence database - PRE-ATLAS chain, marker-union surface | 49 KB |
| [`evidence_summary.tsv`](../evidence_pre_atlas_2026-04/evidence_summary.tsv) | the same, as a table | 14 KB |

## Every sealed procedure — 27 procedures, 92 files

Each names its bars **before** the run and its outcome against them. The failures are here too: row 5b closed
NOT COMMISSIONED (PROC-MAHA-03), and PROC-RECORD-02 reclassified four PASS rows to modelled predictions.
Every procedure in the tree is listed; the count above is generated from it, so the list and the number cannot
disagree.

- **PROC-AGE-01** — [OUTCOME.md](../../Record/PROC_data/PROC-AGE-01/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-AGE-01/PREREG.md), [age01_a4.json](../../Record/PROC_data/PROC-AGE-01/age01_a4.json), [age01_results.json](../../Record/PROC_data/PROC-AGE-01/age01_results.json)
- **PROC-ANCHOR-01** — [PROC_ANCHOR_01.py](../kit/PROC_ANCHOR_01.py), [PROC_ANCHOR_01_GSE51057.json](../kit/results/PROC_ANCHOR_01_GSE51057.json)
- **PROC-BIDIR-01** — [PROC_BIDIR_01.py](../kit/PROC_BIDIR_01.py), [OUTCOME.md](../../Record/PROC_data/PROC-BIDIR-01/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-BIDIR-01/PREREG.md), [PROC_BIDIR_01_as_run.py](../../Record/PROC_data/PROC-BIDIR-01/PROC_BIDIR_01_as_run.py), [VAL_050_RESULTS_rerun_2026-09-21.json](../../Record/PROC_data/PROC-BIDIR-01/VAL_050_RESULTS_rerun_2026-09-21.json), [VAL_051_RESULTS_rerun_2026-09-21.json](../../Record/PROC_data/PROC-BIDIR-01/VAL_051_RESULTS_rerun_2026-09-21.json), [b4_engine_vs_seal.json](../../Record/PROC_data/PROC-BIDIR-01/b4_engine_vs_seal.json), [proc_bidir_01.json](../../Record/PROC_data/PROC-BIDIR-01/proc_bidir_01.json)
- **PROC-CAL-01** — [PROC_CAL_01.py](../kit/PROC_CAL_01.py)
- **PROC-CEIL-01** — [OUTCOME.md](../../Record/PROC_data/PROC-CEIL-01/OUTCOME.md), [tceil_results.json](../../Record/PROC_data/PROC-CEIL-01/tceil_results.json)
- **PROC-CMB-01** — [OUTCOME.md](../../Record/PROC_data/PROC-CMB-01/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-CMB-01/PREREG.md), [PROC_CMB_01_as_run.py](../../Record/PROC_data/PROC-CMB-01/PROC_CMB_01_as_run.py), [plate_GSM2334327_CMB01_no_zero.png](../../Record/PROC_data/PROC-CMB-01/plate_GSM2334327_CMB01_no_zero.png), [proc_cmb_01.json](../../Record/PROC_data/PROC-CMB-01/proc_cmb_01.json)
- **PROC-CMB-02** — [OUTCOME.md](../../Record/PROC_data/PROC-CMB-02/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-CMB-02/PREREG.md), [PROC_CMB_02_as_run.py](../../Record/PROC_data/PROC-CMB-02/PROC_CMB_02_as_run.py), [proc_cmb_02.json](../../Record/PROC_data/PROC-CMB-02/proc_cmb_02.json)
- **PROC-CMB-03** — [OUTCOME.md](../../Record/PROC_data/PROC-CMB-03/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-CMB-03/PREREG.md), [PROC_CMB_03_as_run.py](../../Record/PROC_data/PROC-CMB-03/PROC_CMB_03_as_run.py), [proc_cmb_03.json](../../Record/PROC_data/PROC-CMB-03/proc_cmb_03.json)
- **PROC-CMB-04** — [OUTCOME.md](../../Record/PROC_data/PROC-CMB-04/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-CMB-04/PREREG.md), [PROC_CMB_04_as_run.py](../../Record/PROC_data/PROC-CMB-04/PROC_CMB_04_as_run.py), [plate_GSM2333901_healthy_GSE87571.png](../../Record/PROC_data/PROC-CMB-04/plate_GSM2333901_healthy_GSE87571.png), [proc_cmb_04.json](../../Record/PROC_data/PROC-CMB-04/proc_cmb_04.json)
- **PROC-CMB-05** — [PROC_CMB_05.py](../kit/PROC_CMB_05.py), [OUTCOME.md](../../Record/PROC_data/PROC-CMB-05/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-CMB-05/PREREG.md), [PROC_CMB_05_as_run.py](../../Record/PROC_data/PROC-CMB-05/PROC_CMB_05_as_run.py), [proc_cmb_05.json](../../Record/PROC_data/PROC-CMB-05/proc_cmb_05.json)
- **PROC-DECON-01** — [PROC_DECON_01.py](../kit/PROC_DECON_01.py)
- **PROC-FORMULA-01** — [PROC_FORMULA_01.py](../kit/PROC_FORMULA_01.py)
- **PROC-HISTORY-01** — [OUTCOME.md](../../Record/PROC_data/PROC-HISTORY-01/OUTCOME.md)
- **PROC-MAHA-01** — [OUTCOME.md](../../Record/PROC_data/PROC-MAHA-01/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-MAHA-01/PREREG.md), [maha01_chip_diag.json](../../Record/PROC_data/PROC-MAHA-01/maha01_chip_diag.json), [maha01_m145.json](../../Record/PROC_data/PROC-MAHA-01/maha01_m145.json), [maha01_m23.json](../../Record/PROC_data/PROC-MAHA-01/maha01_m23.json)
- **PROC-MAHA-02** — [OUTCOME.md](../../Record/PROC_data/PROC-MAHA-02/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-MAHA-02/PREREG.md), [maha02_results.json](../../Record/PROC_data/PROC-MAHA-02/maha02_results.json)
- **PROC-MAHA-03** — [PROC_MAHA_03_OUTCOME.md](PROC_MAHA_03_OUTCOME.md), [PROC_MAHA_03_PREREG.md](PROC_MAHA_03_PREREG.md), [PROC_MAHA_03.py](../kit/PROC_MAHA_03.py), [PROC_MAHA_03_stage1_table.py](../kit/PROC_MAHA_03_stage1_table.py), [PROC_MAHA_03.json](../kit/results/PROC_MAHA_03.json)
- **PROC-MATCH-01** — [OUTCOME.md](../../Record/PROC_data/PROC-MATCH-01/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-MATCH-01/PREREG.md), [m5_cached_arrays.json](../../Record/PROC_data/PROC-MATCH-01/m5_cached_arrays.json), [test_disease_matching_gate.py](../../Record/PROC_data/PROC-MATCH-01/test_disease_matching_gate.py)
- **PROC-PANEL-01** — [OUTCOME.md](../../Record/PROC_data/PROC-PANEL-01/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-PANEL-01/PREREG.md), [panel01_results.json](../../Record/PROC_data/PROC-PANEL-01/panel01_results.json), [per_sample_input.csv](../../Record/PROC_data/PROC-PANEL-01/per_sample_input.csv)
- **PROC-PANEL-02** — [OUTCOME.md](../../Record/PROC_data/PROC-PANEL-02/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-PANEL-02/PREREG.md), [panel02_results.json](../../Record/PROC_data/PROC-PANEL-02/panel02_results.json)
- **PROC-PANEL-03** — [OUTCOME.md](../../Record/PROC_data/PROC-PANEL-03/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-PANEL-03/PREREG.md), [panel03_results.json](../../Record/PROC_data/PROC-PANEL-03/panel03_results.json)
- **PROC-RECORD-02** — [OUTCOME.md](../../Record/PROC_data/PROC-RECORD-02/OUTCOME.md), [val025_028_aging_ZENODO_COPY.py](../../Record/PROC_data/PROC-RECORD-02/val025_028_aging_ZENODO_COPY.py)
- **PROC-RECORD-03** — [OUTCOME.md](../../Record/PROC_data/PROC-RECORD-03/OUTCOME.md), [age_matrix_provenance_check.json](../../Record/PROC_data/PROC-RECORD-03/age_matrix_provenance_check.json)
- **PROC-SEP-03** — [PROC_SEP_03.py](../kit/PROC_SEP_03.py)
- **PROC-SWITCH-01** — [OUTCOME.md](../../Record/PROC_data/PROC-SWITCH-01/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-SWITCH-01/PREREG.md), [switch01_s123.json](../../Record/PROC_data/PROC-SWITCH-01/switch01_s123.json), [switch01_s4.json](../../Record/PROC_data/PROC-SWITCH-01/switch01_s4.json), [switch01_s4_diag.json](../../Record/PROC_data/PROC-SWITCH-01/switch01_s4_diag.json), [switch01_s6.json](../../Record/PROC_data/PROC-SWITCH-01/switch01_s6.json)
- **PROC-SWITCH-02** — [OUTCOME.md](../../Record/PROC_data/PROC-SWITCH-02/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-SWITCH-02/PREREG.md), [switch02_results.json](../../Record/PROC_data/PROC-SWITCH-02/switch02_results.json)
- **PROC-TIER-01** — [OUTCOME.md](../../Record/PROC_data/PROC-TIER-01/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-TIER-01/PREREG.md), [test_tiers_as_run.py](../../Record/PROC_data/PROC-TIER-01/test_tiers_as_run.py), [tier01_healthy_tiers.csv](../../Record/PROC_data/PROC-TIER-01/tier01_healthy_tiers.csv)
- **PROC-TIER-02** — [OUTCOME.md](../../Record/PROC_data/PROC-TIER-02/OUTCOME.md), [PREREG.md](../../Record/PROC_data/PROC-TIER-02/PREREG.md), [test_tiers_as_run.py](../../Record/PROC_data/PROC-TIER-02/test_tiers_as_run.py), [u2_occupancy.json](../../Record/PROC_data/PROC-TIER-02/u2_occupancy.json)

## What is NOT published, and why

| not published | why, and what to do instead |
|---|---|
| Raw MCMC posterior chains | The samplers never wrote them to disk, so no run produced a sample archive. What a reviewer can do instead is re-run the calibration in about fifteen seconds and compare: [10.5281/zenodo.22905819](https://doi.org/10.5281/zenodo.22905819) carries the samplers, the 37 reference cells, `reproduce.sh` and the 2026-09-22 reproduction (every floor inside its own posterior SD, largest difference 0.000245, R-hat < 1.001). |
| **Raw IDATs for the commissioned cohorts** | Public at their accessions (GSE87571, GSE42861, GSE111629, GSE125105) and not mirrored here; every script names the accession it reads. |
| **Controlled-access cohorts** | The cfDNA and fragmentomics cohorts named as Future Goals (DELFI, Mouliere) are controlled access. Nothing here depends on them; they are listed as requirements, not as data held. |
| **Disease evidence from the commissioned chain** | It does not exist yet. By decision it belongs in Issue 004, after sealed runs against pre-registered bars. The April 2026 database above is the *pre-atlas* chain's and is labelled as history. |
| **Per-cell reporting below class level** | Withheld by the instrument rather than by us: the atlas cannot separate the members of a collinearity group. [`percell_reference_v0_3.json`](../chain/Runtime%20Matrices/Percell_Reference/percell_reference_v0_3.json) publishes per-entry resolvability so a reader can see which entries are affected. |

## Two scope limits to read before citing anything

1. **The bootstrap comparison does not cover methylation.** It is 8 classes x 4 substrates - nucleosome occupancy, fuzziness, WPS, fragment size. The eight methylation floors the chain actually divides by rest on G-002's own convergence (R-hat < 1.001 on every methylation chain) and its 37-cell reference database.
2. **The April 2026 evidence database is a different surface.** Marker-union, not identity loci - and the two move in opposite directions with age (RECON D2). Its numbers must never be quoted beside [[Issue 003](../manual/IAMPerformance_GAPEIssue003_RC1.pdf)](../manual/IAMPerformance_GAPEIssue003_RC1.pdf)'s.

## The order of steps

[`CHAIN_SEQUENCE.md`](CHAIN_SEQUENCE.md) is generated from the code: every call each path makes, in order, and every file named as chain that no path calls.
