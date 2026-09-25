#!/usr/bin/env python3
"""build_chain_inventory.py - enumerate every live file of the chain, classify it, and describe it.

Written 2026-09-22 because the author asked whether the Chain tab lists everything the chain uses. It did not:
an audit found 20 load-bearing files linked nowhere, including IAMAtlasREBUILD.csv itself (named seven times in
engine code). Hand-maintained lists drift; this enumerates the tree instead, so a file cannot be silently
omitted - anything without a description is emitted with role "UNDESCRIBED" and counted on the page.

Classification, measured not asserted:
  chain      - resolved by cpg_conductor._find(), or loaded by a module that is
  interface   - the report builder and its helpers
  guard       - the reproduction kit: conformance tests and sealed procedures
  reference   - runtime matrices and calibration data the chain reads
  record      - present and callable but NOT called by run_full (e.g. the disease matrix after the 2026-09-21 ruling)
  superseded  - kept for provenance; a later file does the job
Output: MethylPhys/chain/Runtime Matrices/chain_inventory_v1.json
"""
import os, re, json, hashlib, subprocess, time

def _bio_root(start=None):
    """The directory that CONTAINS MethylPhys/ - i.e. Biological_Physics.

    Derived by ascending from this file until a directory holding 'MethylPhys' is found, rather than by counting
    dirname() calls. The 2026-09-22 move (CPG_Engine -> MethylPhys/chain, Testing_and_Code -> Record) changed the
    depth of every script by one; a counted chain resolved to MethylPhys and silently stopped finding Record/.
    """
    import os as _os
    d=_os.path.dirname(_os.path.abspath(start or __file__))
    for _ in range(8):
        if _os.path.isdir(_os.path.join(d,"MethylPhys")): return d
        if _os.path.basename(d)=="MethylPhys": return _os.path.dirname(d)
        nd=_os.path.dirname(d)
        if nd==d: break
        d=nd
    return _os.path.dirname(_os.path.dirname(_os.path.abspath(start or __file__)))

HERE=os.path.dirname(os.path.abspath(__file__)); ENG=os.path.dirname(HERE); BIO=_bio_root()

DESC={
    "CHAIN_SEQUENCE.md": ("reference", "the step order as the code calls it, derived by AST - never typed", ""),
    "REPORT_TAB_REFERENCE.md": ("reference", "the report described tab by tab with a figure of each, generated from a finished report", ""),
    "report_tabs.json": ("reference", "the tab descriptions as data, read by both the SOP generator and the manual build", ""),
    "intro_blocks.json": ("reference", "the manual's introduction, generated from the report's Story and Sky tabs so the two cannot drift", ""),
    "propagate_status.json": ("record", "the gate's last verdict, printed on every report's Run tab", ""),
    "trace_detection_panel_v1.json": ("reference", "the frozen panel and thresholds for Stage 2c trace detection", ""),
    "REVIEWER_MANIFEST.md": ("reference", "the reviewer download list: every file a referee could want, resolved from the tree", ""),
    "ENHANCEMENTS.md": ("reference", "what would make the chain more sensitive, ranked by whether it changes a reported number", ""),
    "REPO_INVENTORY.md": ("reference", "what is in the repository, measured, and what should not be", ""),
    "RUN_INDEX.csv": ("record", "one row per execution of the chain, regenerated from the evidence ledgers", ""),
    "RUN.md": ("record", "what one execution was: specimen, inputs, what was reported", ""),
    "stage_0_1_qc_handoff.py": ("chain", "decodes the array's own controls, bead counts and sex intensities so the intake gates can be measured rather than deferred", ""),
    "data003.py": ("interface", "the manual's data module: every constant, table and section text it renders", ""),
    "gape002_lib.py": ("interface", "the shared renderer both editions of the manual are built with", ""),
    "build_gape_issue003.py": ("interface", "builds the operations manual; run through build_twopass.sh so the contents page carries measured page numbers", ""),
    "build_gape_issue002.py": ("superseded", "builds the earlier edition, kept so it still reproduces exactly", ""),
    "part3_indepth.py": ("interface", "the manual's Part III sections", ""),
    "toc_pages.json": ("interface", "page numbers measured in the first build pass, read by the second", ""),
    "appendix_vi_vii.json": ("interface", "appendix content for the manual", ""),
    "claim_scan.py": ("guard", "scans the documents for claims and checks each against the sealed record", ""),
    "claims.json": ("record", "the claims the scan found, with their evidence", ""),
    "reference_cells_37.csv": ("reference", "the 37 published reference methylomes the class floors were fitted on", ""),
    "g003_mcmc_framework.py": ("reference", "the framework the floor calibration runs on", ""),
    "REPRODUCTION.md": ("reference", "how to reproduce every class floor in about fifteen seconds", ""),
    "percell_reference_v0_3.json": ("reference", "the per-entry healthy references the per-cell columns are zeroed on", ""),
    "iamatlas_celltype_markers_v0_3_TRIAL.json": ("reference", "trial marker panel, not adopted - kept because a procedure cites it", ""),
    "SMALL_CLASS_DETECTION_NOTE_2026-09-23.md": ("record", "the measured detection limit for a trace class, and why the boundary pins it", ""),
    "make_figures.py": ("record", "builds the figures for the papers in this folder", ""),
    "PART_II_CHAPTER_NOTES.md": ("record", "the outline for Part II, written after the instrument is commissioned", ""),
    "README_CPG_Plates.md": ("reference", "the reference plates and their conventions - four of them are embedded in every report", ""),
    "val_index.json": ("record", "the validation index as data", ""),

    "propagate.py": ("guard", "the gate: regenerates every derived document, then checks the rules a human wrote; exits non-zero on drift", ""),
    "guarded_push.sh": ("guard", "the only sanctioned push - runs propagate.py without a pipe and refuses to commit or push if it fails", ""),
    "evaluate_necessity.py": ("guard", "answers whether a file is necessary, from the tree: runs, imported, named in code, named in a document, or a generator", ""),
    "build_report_tab_reference.py": ("guard", "generates the tab-by-tab report reference and one figure per tab by reading a finished report", ""),
    "build_run_index.py": ("guard", "regenerates the run index from every evidence ledger in the tree", ""),
    "build_folder_readmes.py": ("guard", "generates a README for every folder, listing what is in it - the purpose lines are held in the script", ""),
    "cmb_tools.py": ("guard", "the register of every method borrowed from CMB analysis, each with a check that returns its state on a finished bundle", ""),
    "disease_matching.py": ("not in the chain", "Stage 8, future work for Issue 004: the one live function extracted from the retired v1 conductor", ""),
    "stage_2c_trace_detection.py": ("chain", "Stage 2c: is there evidence of a trace class at all, by a score test that the non-negativity boundary cannot pin", ""),
    "sop_repoint.py": ("guard", "regenerates the chain-of-custody procedure: addresses, the report tab section, and the header", ""),
    "sop_stage_links.py": ("guard", "writes an implemented-in line under every stage and step section of the procedure", ""),
    "sop_step_detail.py": ("guard", "writes the per-step operational blocks - thresholds, refusal strings, what the operator does", ""),
    "link_check.py": ("guard", "every relative path in the live documentation set must resolve - a path in a document is a claim", ""),
    "add_doc_links.py": ("guard", "links code names in prose to the files they name, idempotently", ""),
    "build_reviewer_manifest.py": ("guard", "regenerates the reviewer download list, resolving every path by basename from the tree", ""),
    "build_chain_sequence.py": ("guard", "derives the step order from the code by AST, so a document cannot claim a stage the code does not call", ""),
 # ---- the chain, in run order ----
 "cpg_conductor.py":("chain","THE ORCHESTRATOR. run_full() calls every stage in order and returns one bundle; it resolves each stage module and data file by name (_find) so a missing file fails loudly instead of silently defaulting. Everything else in this table is reached from here.","stage 0"),
 "stage_1_idat_calibration.py":("chain","Stage 1. Raw Illumina IDAT pair -> beta per CpG (methylprep, noob background/dye correction). The only step that touches instrument output.","stage 1"),
 "idat_parse.py":("chain","Low-level IDAT binary reader used by Stage 1 when methylprep is not driving the parse.","stage 1"),
 "beta_scale_maps_v1.json":("reference","Stage 1s. The affine map from each named pipeline's beta scale onto the scale the floors were calibrated on. Without a map for your pipeline the conductor REFUSES to place a reading (LESSON-SCALE-01).","stage 1s"),
 "walther_iam_deconvolver.py":("chain","Stage 2. Constrained non-negative least squares: which cell types are present and in what fraction. Sparse by design - it reports only what it has evidence for.","stage 2"),
 "IAMAtlasREBUILD.csv":("reference","THE ATLAS. 483,092 CpGs x 115 cell types, each entry a posterior mean with an SD and credible interval from the MCMC build. Every comparison on the report is against this file. Ships compressed (.xz); decompress once.","stage 2"),
 "IAMAtlasREBUILD.csv.xz":("reference","The atlas as shipped (compressed). Decompress with Python's lzma - no system xz needed.","stage 2"),
 "IAMAtlasREBUILD_celltype_to_class.json":("reference","Which of the 8 architecture classes each of the 115 cell types belongs to. The bridge between per-cell and per-class reporting.","stage 2"),
 "IAMAtlasREBUILD_provenance.json":("reference","How the atlas was built: sources, sample counts, MCMC settings, checksums.","stage 2"),
 "nilc_celltype_deconvolver.py":("chain","Stage 2b. The second, independent solver (inverse-variance weighted, atlas posterior SD as the weight). Reported ONLY as an agreement flag at class level, never as the composition.","stage 2b"),
 "lineage_splitter.py":("chain","Helper for Stage 2b: splits a lineage signal across its member entries.","stage 2b"),
 "iamatlas_a_scoring.py":("chain","The A-score itself: A = mean_i H(beta_i) / H_min(class). Asserts against the wrong aggregation (entropy of the mean beta), which is the defect PROC-N7-01 caught.","stage A"),
 "iamatlas_celltype_markers_v0_2.json":("reference","The ~100 discriminative marker CpGs per cell type, used for the per-cell separation statistic. The sealed foundation-cohort anchors reproduce on THIS file, so it cannot be changed without a re-seal.","stage A"),
 "percell_exclusivity_v0.json":("reference","How exclusive each entry's marker panel is to that entry (measured 2026-09-22). 33.8 % of markers serve more than one panel; the report withholds the individual per-cell claim for the 36 entries under 25 % exclusivity.","stage A"),
 "percell_reference_v0.json":("reference","Per-entry healthy reference: for each of the 115 entries, the healthy range of its own per-cell reading, per laboratory. Exploration, unsealed.","stage A"),
 "cpg_gauge_engine.py":("chain","The class gauge: the eight frozen H_min floors and the per-substrate table. Holds the saturation ceilings (1/H_min).","stage B"),
 "cpg_gauge.py":("superseded","Earlier single-file gauge; cpg_gauge_engine.py plus cpg_tiers.py do this now.","stage B"),
 "iamatlas_gauge_identity_loci_v1_0.json":("reference","THE IDENTITY LOCI and the eight class floors. The unimodal addresses where a healthy class sits at one level - the surface the reported gauge reads (row B). H_min per class lives here.","stage B"),
 "reference_age_curve_v1.json":("reference","The healthy age curve: about +0.47 milli-A per year, built leave-one-laboratory-out from four cohorts (n=1,379). The decade term is subtracted before a sample is placed.","stage B"),
 "lab_zero.py":("chain","The laboratory zero: median of 40 healthy arrays of that laboratory, age-referenced. Refuses panels under 40 and marks a reading without a zero as NOT REPORTABLE.","stage B"),
 "identity_band_v3.json":("reference","The commissioned healthy band, pooled and per decade, with each laboratory's measured zero. Placement (IN_BAND / above / below) is read from here.","stage B"),
 "cpg_tiers.py":("chain","THE ONLY tier function. Reads tier_breakpoints.json; the engine previously carried three disagreeing definitions. Returns no tier when a reading is not reportable, and AT_CEILING at 1/H_min.","stage 7"),
 "tier_breakpoints.json":("reference","The tier boundaries, the Warburg line (1.07) and the breach line (1.10), plus the reference clusters. The single source for every tier word on the report.","stage 7"),
 "bidirectional_decomposition.py":("chain","Stage 4.5. Splits a departure into its hyper- and hypo-methylated halves, so direction is reported rather than magnitude only.","stage 4.5"),
 "directional_panels_v1_0.json":("reference","The directional panel: per-CpG healthy mean and direction. Only the immune class has a sealed panel today.","stage 4.5"),
 "stage_4_6_patient_cmb.py":("chain","Stage 4.6. The patient's sky: residual z per CpG against the sample's own composition expectation, on the laboratory's zero and scale, projected onto the sphere in genomic order.","stage 4.6"),
 "build_healpix_mapping.py":("chain","Builds the CpG -> pixel mapping from the array manifest in genomic order. Deterministic: same atlas + manifest -> byte-identical output.","stage 4.6"),
 "iamatlas_cpg_to_healpix_nside128.npz":("reference","The mapping itself: 483,092 CpGs onto 196,608 pixels. Measured 2026-09-22 to be genomically local - every pixel holds contiguous CpGs of one chromosome, median span 511 bp.","stage 4.6"),
 "iamatlas_cpg_to_healpix_nside128.provenance.json":("reference","Provenance and checksum of the mapping.","stage 4.6"),
 "presence_floors_v1.json":("reference","The measured healthy presence floor per class: below it a class IS NOT THERE in this specimen, so its panel is masked rather than scored.","stage 4.6"),
 "EPIC_plus_HM450_combined_manifest_normalized.csv":("reference","The Illumina array manifest (IlmnID, CHR, MAPINFO). Supplies the genomic coordinates the sky projection and the locality measurement need.","stage 4.6"),
 "iamatlas_mahalanobis_scoring.py":("chain","Stage 5. The departure: distance from the healthy centre in units of healthy spread over the assessable class axes (Mahalanobis, 1936).","stage 5"),
 "mahalanobis_healthy_reference_v2_0_age_matched_derived.json":("reference","The healthy centre and spread per class and age the departure is measured against.","stage 5"),
 "stage_5_second_chain.py":("record","An alternative Stage 5 formulation kept for comparison; run_full uses iamatlas_mahalanobis_scoring.","stage 5"),
 "iam_cellular_age_scoring.py":("chain","Stage 6. Inverts the healthy age curve. Reports the RESOLUTION, never an age in years: one array cannot place a person to better than ~50 yr (PROC-AGE-01).","stage 6"),
 "age_reference_matrix.json":("reference","The April 80-cell age table. Typed beta means with A by formula - NOT a per-sample measurement; kept because parts of the record cite it. The live age term comes from reference_age_curve_v1.json.","stage 6"),
 "age01_results.json":("reference","The PROC-AGE-01 measurement: the slope, the within-laboratory spread, and the resolution that follows.","stage 6"),
 # ---- not in the chain ----
 "disease_matching.py (v1 conductor retired 2026-09-25)":("record","The pre-conductor monolith. Its stage_8_dual_matching (disease-pattern concordance) is still callable for the record but IS NOT CALLED BY run_full - the author removed disease matching from the chain on 2026-09-21. Everything else here was superseded by cpg_conductor.py.","not in chain"),
 "disease_cell_signature_matrix_v1_13.csv":("record","The disease-pattern signature matrix, from the preliminary record. Not read by run_full.","not in chain"),
 "disease_origin_cells.json":("record","Cell-of-origin gate for the removed matching stage. Not read by run_full.","not in chain"),
 "iamatlas_115_to_matrix_v0_2_mapping.json":("record","Maps atlas entries onto the signature matrix. Not read by run_full.","not in chain"),
 "synthetic_patient_generator.py":("guard","Builds synthetic patients with known composition and known truth. This is what caught the gauge reading the wrong loci (PROC-N7-01) - the most valuable single guard in the kit.","guard"),
 "cpg_null_runner.py":("guard","The null suite: shuffled and permuted inputs that must NOT produce a signal.","guard"),
 "cpg_report_v3.py":("superseded","The first report builder to the author's specification; MethylPhys_Interface/build_methylphys.py replaced it.","not in chain"),
 "cpg_report_builder.py":("superseded","The original report builder, wired to the removed disease-matching stage. Record-side only.","not in chain"),
 "cpg_patient_cmb.py":("superseded","Earlier sky implementation; stage_4_6_patient_cmb.py supersedes it.","not in chain"),
 "build_dashboard_v1.py":("superseded","The June dashboard builder the author supplied as the model for this interface.","not in chain"),
 # ---- interface ----
 "build_methylphys.py":("interface","THIS REPORT. Renders one self-contained HTML from the conductor's bundle plus the runtime files, with a vocabulary guard that refuses to write a measurement tab carrying a condition name, a verdict word or an age in years.","interface"),
 "run_sample.py":("interface","One command: IDAT pair or a cpg_id,beta CSV -> Stage 1 -> run_full -> this report.","interface"),
 "build_percell_reference.py":("interface","Builds percell_reference_v0.json: each entry's own healthy range, per laboratory, from that laboratory's build panel.","interface"),
 "build_chain_inventory.py":("interface","Generates this table by enumerating the tree, so the Chain tab cannot silently omit a file.","interface"),
 # ---- guards ----
 "release_check.py":("guard","Every guard in one command; writes release_check.json, which the Safeguards tab prints. Reports a guard that cannot run as SKIPPED, never as a pass.","guard"),
 "finding_check.py":("guard","The protocol gate: a finding must be registered, every door taught, and no unqualified detection claim present, or the push is blocked.","guard"),
 "cpg_kit.py":("guard","Shared kit helpers: locates the engine, the runtime matrices and the test data by environment variable.","guard"),
 "test_gauge_switch.py":("guard","The commissioned identity gauge reproduces on the cached commissioning arrays.","guard"),
 "test_tiers.py":("guard","Every tier boundary in the JSON, both sides, through the one tier function; fails if a literal breakpoint reappears in engine code.","guard"),
 "test_patient_sky.py":("guard","The sky: deterministic mapping, presence-floor masking, and refusal without a commissioned laboratory scale.","guard"),
 "test_lab_zero.py":("guard","Recovers a synthetic laboratory offset; refuses panels under 40 arrays.","guard"),
 "test_a_score_canonical.py":("guard","The A-score formula self-test, including the aggregation the module must refuse.","guard"),
 "PROC_ANCHOR_01.py":("guard","The sealed foundation-cohort anchors reproduce from raw GEO betas (r = 1.00000).","guard"),
 "PROC_FORMULA_01.py":("guard","Which aggregation reproduces the seal, measured on both.","guard"),
 "PROC_DECON_01.py":("guard","The composition solver against its answer key.","guard"),
 "PROC_SEP_03.py":("guard","Atlas separability by class - the measurement behind the blood caveat.","guard"),
 "PROC_BIDIR_01.py":("guard","The directional detector, including re-extraction from the raw 5.1 GB GEO matrix.","guard"),
 "PROC_CMB_05.py":("guard","The sky, commissioned: healthy tail, determinism, refusal paths.","guard"),
 "PROC_PLASMA_MIX_01.py":("guard","Plasma cfDNA mixture behaviour - reserved specimen, recorded.","guard"),
 # ---- doors ----
 "RUNBOOK.md":("guard","How to run the chain, how to commission a laboratory, and the seven-step finding protocol.","door"),
 "CHAIN_COMMISSIONING.md":("guard","Which stage is commissioned, by which sealed procedure, and what is still open.","door"),
 "HANDOFF.md":("guard","The state of the work, for the next reader.","door"),
 "ROW9_WORKING_NOTE.md":("guard","Engineering log for the report and interface: what was measured while building it, what failed, and what each failure changed. Read it for the construction history; nothing in it is needed to read a result.","door"),
}

# ---- second pass 2026-09-22: every remaining live file, described from its own header rather than guessed ----
DESC.update({
 "stage_0_intake.py":("chain","Stage 0. Sample intake and chain of custody: IDAT arrival, manifest creation, identity checks. Built step by step against the SOP; four QC checks are still deferred because Stage 0 is not yet handed the raw intensities (open item, row 0).","stage 0"),
 "stage_1_calibration.py":("superseded","The earlier Stage 1 wrapper. stage_1_idat_calibration.py is what the chain calls.","stage 1"),
 "idat_decoder_pure.py":("chain","Pure-Python IDAT to beta - needs only idat_parse plus a static array manifest, no methylprep or minfi at runtime. This is what lets a reader reproduce Stage 1 without the R/Bioconductor stack.","stage 1"),
 "preflight.py":("guard","Run once on a new machine: checks the Python version, every package the chain needs, and that each runtime file is present and readable. Fails before a run rather than during one.","guard"),
 "geo_fetch_idats.py":("guard","Fetches only the IDAT pairs you need from GEO in parallel, filtered by a metadata field - how the four healthy laboratory panels were assembled without downloading whole archives.","guard"),
 "run_batch.py":("interface","Batch runner: processes every patient visit that does not yet have a report, over a patients/visit folder layout.","interface"),
 "residual_scale_GSE87571.npz":("reference","Uppsala's per-address healthy residual scale for the sky - the denominator that makes a patient's z a z. One per commissioned laboratory.","stage 4.6"),
 "residual_scale_GSE42861.npz":("reference","Karolinska's per-address residual scale for the sky.","stage 4.6"),
 "residual_scale_GSE111629.npz":("reference","UCLA's per-address residual scale for the sky.","stage 4.6"),
 "residual_scale_GSE125105.npz":("reference","Munich's per-address residual scale for the sky.","stage 4.6"),
 "iamatlas_cpg_to_healpix_nside128.npy":("reference","The mapping array itself (the .npz is the packaged form). 483,092 CpGs to 196,608 pixels in genomic order.","stage 4.6"),
 "generate_cpg_healpix_mapping.py":("chain","One-time generator for the mapping: lays the CpGs on the sphere in genomic order so every patient projection sits on the same grid as the archival plates.","stage 4.6"),
 "README_HEALPix_Mapping.md":("reference","What the mapping is, how it was generated, and its determinism guarantee.","stage 4.6"),
 "iamatlas_collinearity_groups_v0_1.json":("reference","Which atlas cell types are collinear - i.e. which ones the reference cannot fully separate. Directly relevant to per-cell reporting: entries inside one group should not be scored against each other.","stage 2"),
 "identity_band_v2_PROVISIONAL.json":("superseded","The provisional band before the four-laboratory zeros. identity_band_v3.json is live.","stage B"),
 "scale_map_addendum.json":("reference","The fitted gain and affine terms of a pipeline scale map, with the mapped band it produces - the measurement behind LESSON-SCALE-01.","stage 1s"),
 "EPIC_v1_B4_manifest_normalized.csv":("reference","The EPIC v1 B4 array manifest (alternative to the combined manifest).","stage 4.6"),
 "README_external_manifests.md":("reference","Where the array manifests came from and how they were normalised.","stage 4.6"),
 # --- floor calibration: the provenance of the eight numbers everything else divides by ---
 "gape_mcmc_g002.py":("reference","CHAIN G-002: the MCMC that calibrated the eight methylation H_min floors against 37 published reference cell methylomes. This is the provenance of every floor on the report.","calibration"),
 "gape_mcmc_g003b.py":("reference","CHAIN G-003b: H_min posteriors for the four non-methylation substrates (nucleosome occupancy, fuzziness, windowed protection, fragment size).","calibration"),
 "gape_mcmc_g008.py":("record","CHAIN G-008: a zero-free-parameter prediction of the tumour-versus-normal A-score gap. Pre-atlas surface; part of the record, not a result of this chain.","calibration"),
 "gape_mcmc_e_a_bio.py":("record","CHAIN E(a_bio): an activation-function fit to published age-stratified pace-of-ageing data. Pre-atlas; record-side.","calibration"),
 "gape_mcmc_nbio_ordering.py":("record","CHAIN n_bio: tests whether the class ordering is consistent with published respirometry. Pre-atlas; record-side.","calibration"),
 "gape_bootstrap_comparison.py":("reference","The leave-one-out bootstrap cross-check of the floors. Its function was rerun on the eight methylation floors in September (PROC-HMIN-BOOT-01); every frozen floor fell inside its interval.","calibration"),
 "methyl_bootstrap_PROC-HMIN-BOOT-01.json":("reference","The result of that bootstrap: per class, the frozen floor beside its bootstrap interval.","calibration"),
 "literature_anchors.json":("reference","Published reference A-score anchors per class (healthy / disease / cancer), extracted from the April web build. Orientation values from the literature, not measurements of this chain.","calibration"),
 # --- the atlas build itself ---
 "iamatlas_v0_1_mcmc_batched_FIXED.py":("reference","The atlas build: a hierarchical Beta-Binomial MCMC run per architecture class, producing the posterior mean, SD and interval for every CpG-by-cell-type entry.","atlas build"),
 "compact_atlas.py":("reference","Packages the rebuild outputs into the repo-ready atlas (.csv.xz) and the per-class archives.","atlas build"),
 "stem_pluri_v0_1_REBUILD.tar.xz":("reference","Per-class atlas rebuild archive (pluripotent stem) - the raw MCMC output the merged atlas was assembled from.","atlas build"),
 "stem_adult_v0_1_REBUILD.tar.xz":("reference","Per-class atlas rebuild archive (adult stem).","atlas build"),
 "progenitor_v0_1_REBUILD.tar.xz":("reference","Per-class atlas rebuild archive (progenitor).","atlas build"),
 "immune_v0_1_REBUILD.tar.xz":("reference","Per-class atlas rebuild archive (immune).","atlas build"),
 "cycling_v0_1_REBUILD.tar.xz":("reference","Per-class atlas rebuild archive (cycling).","atlas build"),
 "secretory_v0_1_REBUILD.tar.xz":("reference","Per-class atlas rebuild archive (secretory).","atlas build"),
 "stromal_v0_1_REBUILD.tar.xz":("reference","Per-class atlas rebuild archive (stromal).","atlas build"),
 "terminal_v0_1_REBUILD.tar.xz":("reference","Per-class atlas rebuild archive (terminal).","atlas build"),
 "README_atlas_v0_1_vault.md":("reference","What the per-class archives contain and how to unpack them.","atlas build"),
 "age_reference_matrix.csv":("reference","The April age table in CSV form (same typed content as the JSON).","stage 6"),
 # --- guards and harnesses ---
 "synthetic_patient_harness.py":("guard","Harness around the synthetic patient generator for repeated runs.","guard"),
 "recordside_test_disease_matrix_gate.py":("guard","Proves the removed disease-matching stage fails closed when its files are absent - kept so the removal stays honest.","guard"),
 "switching_order.py":("guard","THE SWITCHING ORDER: one record per stage with everything that touches it clipped to it. The master cross-reference Issue 003 Part II prints from.","door"),
 # --- superseded builders ---
 "build_patient_wall.py":("superseded","Earlier multi-visit wall renderer.","not in chain"),
 "render_patient_wall.py":("superseded","Renderer for the patient wall.","not in chain"),
 "build_strawman.py":("superseded","Early report prototype.","not in chain"),
 "enrich_strawman.py":("superseded","Early report prototype helper.","not in chain"),
 "render_strawman_v2.py":("superseded","Early report prototype renderer.","not in chain"),
 "strawman_data_v2.json":("superseded","Fixture for the prototype renderer.","not in chain"),
 # --- disease cards: record-side ---
 "ad-immune_card_v3_1.json":("record","Disease card (dementia immune) from the preliminary record. Not read by run_full.","not in chain"),
 "breast-epic_card_v3_1.json":("record","Disease card (breast) from the preliminary record. Not read by run_full.","not in chain"),
 "immune-atlas_card_v2_0.json":("record","Disease card (cross-condition immune). Not read by run_full.","not in chain"),
 "disease_cell_signature_matrix_v1_8.csv":("record","Earlier signature matrix. Not read by run_full.","not in chain"),
 "ad_immune_residual_map_chr_annotated.csv":("record","Residual map from the preliminary record (dementia immune).","not in chain"),
 "breast_epic_residual_map_chr_annotated.csv":("record","Residual map from the preliminary record (breast).","not in chain"),
 "immune_atlas_cross_disease_universal_alarm_residual_map_v0_1.csv":("record","Residual map from the preliminary record (cross-condition).","not in chain"),
 "immune_atlas_cross_disease_universal_alarm_residual_map_v0_1.provenance.json":("record","Provenance of that residual map.","not in chain"),
 "ad_immune_bimodality_map.csv":("record","Bimodality map, preliminary record.","not in chain"),
 "breast_epic_bimodality_map.csv":("record","Bimodality map, preliminary record.","not in chain"),
 "immune_atlas_cross_disease_universal_alarm_bimodality_map.csv":("record","Bimodality map, preliminary record.","not in chain"),
 "ad_immune_pca_projections.csv":("record","PCA projections, preliminary record.","not in chain"),
 "breast_epic_pca_projections.csv":("record","PCA projections, preliminary record.","not in chain"),
 "immune_atlas_cross_disease_universal_alarm_pca_projections.csv":("record","PCA projections, preliminary record.","not in chain"),
 "ad-immune_README.md":("record","Card notes, preliminary record.","not in chain"),
 "breast-epic_README.md":("record","Card notes, preliminary record.","not in chain"),
 "ad-immune_v3_1_release_notes.md":("record","Card release notes.","not in chain"),
 "breast-epic_v3_0_release_notes.md":("record","Card release notes.","not in chain"),
 "breast-epic_v3_1_release_notes.md":("record","Card release notes.","not in chain"),
 # --- doors ---
 "chain_inventory_v1.json":("reference","This table: every live file of the chain with its role, purpose, size and SHA-256, generated from the tree.","interface"),
 "val_finding.py":("interface","Writes the structured record of one validation run - instrument fingerprint, per-sample absolute readings, per-entry and per-group direction and magnitude, the pre-registered bars and their outcome. The Findings tab reads these.","interface"),
 "README.md":("guard","Repository entry point.","door"),
 "README_FIRST.md":("guard","Where a new reader should start.","door"),
 "README_FOR_FUTURE_AI.md":("guard","Standing instructions and house rules for anyone - human or assistant - picking this up.","door"),
 "README_STATUS.md":("guard","What is commissioned and what is not, in brief.","door"),
 "CHANGELOG.md":("guard","Dated record of changes.","door"),
 "COMPONENT_MAP.md":("guard","Which component lives where.","door"),
 "MethylPhys_CPG_SOP.md":("guard","The SOP: the procedure the chain implements, step by step. Being rewritten to the real file names.","door"),
 "CPG_Lessons_Learned_2026-06-29.md":("guard","Lessons from the pre-build era, including the ones this chain was designed to prevent.","door"),
 "CMB_TO_METHYLOME_MAP.md":("guard","The 79-row translation map from cosmology tools to methylome tools, each row scored for whether it is built, worth building, or not applicable.","door"),
 "IAMAtlas_FLATNESS_LESSON.md":("guard","Why a flat reference was wrong, and what replaced it.","door"),
 "AD_IMMUNE_v3_0_SOP_CHAIN_OF_CUSTODY_AUDIT.md":("record","Custody audit of a preliminary-record card.","not in chain"),
 "COMPLETION_SPRINT_scored.md":("guard","The author's own completion-sprint plan, scored, with his verdict that it was too ambitious before the bones were trusted.","door"),
 "ROADMAP_TaskTracker.md":("guard","Task tracker.","door"),
 "RUN_MANIFEST_and_README.md":("guard","Run manifest notes.","door"),
 "WORK_IN_PROGRESS.md":("guard","What is mid-flight.","door"),
})

SKIP=re.compile(r"__pycache__|/\.git|/RETIRED/|\.pyc$|/results/|/TEST_DATA/|/handoff/")
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest()
def main():
    try: commit=subprocess.run(["git","-C",BIO,"rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip()
    except Exception: commit=""
    rows=[]; seen=set()
    # Scope: the instrument's own home, MethylPhys/. BIO resolves to Biological_Physics, and walking that pulled
    # in the whole Record/ tree - 704 files, 551 of them evidence outputs with no place in a parts list. The
    # inventory is the chain's parts list; Record has its own index (Record/VAL_INDEX.csv).
    for dp,dn,fs in os.walk(os.path.join(BIO,"MethylPhys")):
        if SKIP.search(dp+"/"): continue
        for f in sorted(fs):
            if not f.endswith((".py",".json",".npz",".npy",".csv",".xz",".md")) or f.startswith("."): continue
            p=os.path.join(dp,f); rel=os.path.relpath(p,os.path.dirname(BIO))
            if f in seen: continue
            role,desc,stage = DESC.get(f,("UNDESCRIBED","",""))
            if role=="UNDESCRIBED" and (f.startswith(("PROC_","VAL_","test_")) or "/Record/" in p or "/Papers/" in p or "/Issue003/" in p or "/Plates/" in p):
                continue   # the record and the documents are inventoried by their own registers, not here
            seen.add(f)
            try: sz=os.path.getsize(p); h=sha(p)[:12] if sz < 200_000_000 else "(large)"
            except OSError: continue
            rows.append({"file":f,"path":rel,"role":role,"stage":stage,"description":desc,"bytes":sz,"sha256_12":h})
    und=[r["file"] for r in rows if r["role"]=="UNDESCRIBED"]
    out={"_meta":{"built":time.strftime("%Y-%m-%d %H:%M"),"commit":commit,"n_files":len(rows),
         "n_undescribed":len(und),"undescribed":sorted(und),
         "how":"enumerated from the live tree by build_chain_inventory.py; roles measured from what cpg_conductor._find() resolves and what run_full() calls",
         "note":"a file with role UNDESCRIBED is a gap in this table, not a file that does nothing - the count is printed on the Chain tab so the gap is visible"},
         "files":sorted(rows,key=lambda r:(r["role"]!="chain",r["stage"],r["file"]))}
    dst=os.path.join(ENG,"Runtime Matrices","chain_inventory_v1.json")
    json.dump(out,open(dst,"w"),indent=1)
    print(f"{len(rows)} files inventoried -> {dst.split('Biological_Physics/')[1]}")
    by=collections.Counter(r["role"] for r in rows) if (collections:=__import__("collections")) else {}
    print("by role:", dict(by))
    if und: print(f"UNDESCRIBED ({len(und)}): {sorted(und)}")
if __name__=="__main__": main()
