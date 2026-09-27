# chain/ — the running code

**What this folder is.** The instrument. Everything a run executes, the runtime constants it reads, and the generators that keep
the documents true to it. If a file is in here, something in a run or a build touches it. Presentation material and the pre-chain
record live under `Record/`; what was removed from the chain lives under `../RETIRED_2026-09/` with a README saying why.

Start at [`../doors/RUNBOOK.md`](../doors/RUNBOOK.md) to run one specimen and [`../doors/CHAIN_SEQUENCE.md`](../doors/CHAIN_SEQUENCE.md)
for the step order **as the code calls it** — that file is generated from the code by [`build_chain_sequence.py`](build_chain_sequence.py)
and is the authority; this README only points at things.

## The live path (one specimen, `MethylPhys_Interface/run_sample.py`)

| stage | what it does | file |
|---|---|---|
| **0 intake** — ten steps, nine checks and a decision, on the array's own numbers. A QUARANTINE stops the chain; nothing is scored. Thresholds are runtime constants in `Runtime Matrices/Intake/intake_thresholds_v1.json`, never numbers in code. | arrival, manifest, integrity hash, control probes, per-probe detection, bead count, call rate (QUARANTINE below 0.93 detected), platform coverage, sex, decision. A deferred check never advances. | [`stage_0_intake.py`](stage_0_intake.py), [`stage_0_1_qc_handoff.py`](stage_0_1_qc_handoff.py) |
| **1 calibration** | raw IDAT pair → β (methylprep noob, per array); per-probe detection against this array's own negative controls, failed probes removed before any stage reads a β; control-probe medians and SNP-probe noise returned | [`stage_1_idat_calibration.py`](stage_1_idat_calibration.py) |
| **1s pipeline map** | one slope and intercept that puts this pipeline's β on the atlas scale ([`beta_scale_maps_v1.json`](Runtime%20Matrices/A_Scoring_Module/beta_scale_maps_v1.json)). A transfer between two measuring pipelines; not a zero, not a correction to any cell | `cpg_conductor.stage_1s_scale_map` |
| **2 composition** | which cells are present and at what fraction (constrained solve on the atlas's cell-type markers), a second unconstrained solve as a cross-check, and the foreign-cell detection panel. Fraction is a **presence gate**; it is never applied to A | [`legacy_iam_deconvolver/legacy_iam_deconvolver.py`](legacy_iam_deconvolver/legacy_iam_deconvolver.py), `cpg_conductor.stage_2b_second_opinion`, `stage_2d_foreign_detection` |
| **A per-cell reading** | for every present cell: A = H(mean β over the cell's identity loci) / H_min of its class, and its tier | [`Runtime Matrices/A_Scoring_Module/iamatlas_a_scoring.py`](Runtime%20Matrices/A_Scoring_Module/iamatlas_a_scoring.py) via `cpg_conductor.stage_a_cells`; tiers [`cpg_tiers.py`](cpg_tiers.py) |
| **B class gauge** | the class-pooled A on the class identity loci — an **internal gate** plus the composition check (whole blood must be blood). Carries no tier word and is not the reading | `cpg_conductor.stage_b_classes`, `stage_b_identity` |
| **4.5 direction** | bidirectional decomposition of the departure (which way the write process moved) | `cpg_conductor.stage_4_5_bidirectional` |
| **4.6 the sky** | this specimen's residual per address — β minus what its own composition predicts from the atlas — on a HEALPix sphere; spread from the atlas posterior through that composition plus this array's own SNP-probe noise. No panel, no laboratory file | [`stage_4_6_patient_cmb.py`](stage_4_6_patient_cmb.py) |
| **9 report** | 17 tabs read from the bundle; a vocabulary guard fails the render if a measurement tab names a population | [`MethylPhys_Interface/build_methylphys.py`](MethylPhys_Interface/build_methylphys.py) |
| orchestrator | | **[`cpg_conductor.py`](cpg_conductor.py)** (`run_full`) |

**Runtime constants** (`Runtime Matrices/`): the class identity loci and floors (`Runtime Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json`),
the per-cell identity loci (`iamatlas_percell_identity_loci_*.json`), the pipeline map, the cell-type markers (`Celltype_Marker/`),
the detection panel, the presence floors (`Patient_CMB/`), the tiers (`Runtime Matrices/Tier_breakpoints/tier_breakpoints.json`), the intake
thresholds (`Intake/`), the cell descriptions (`Cell_Descriptions/`, from the author's cell pages). Every constant a reading
depends on is one of these files; the Instrument tab of every report lists them with their hashes.

**What is not in the live path and why**: [`stage_1_calibration.py`](stage_1_calibration.py) and the pure-Python IDAT decoders
([`idat_decoder_pure.py`](idat_decoder_pure.py), [`idat_parse.py`](idat_parse.py)) — present, called by nothing; the chain reads IDATs through methylprep.
[`cpg_gauge_engine.py`](cpg_gauge_engine.py) holds the five-substrate `H_MIN_TABLE` the methylation floors are one row of.
[`disease_matching.py`](disease_matching.py) / `run_batch.py` are the v1 batch path (a folder of visits) and do not run the current stages.
[`build_chain_sequence.py`](build_chain_sequence.py) prints the "named as chain, called by nothing" list from the code, so this paragraph cannot drift far.

**Two scoring surfaces, one rule each** (SOP §106): the identity-loci gauge is `H(β̄)/H_min` — unimodal loci only. The cell-type
**markers** are bimodal by construction and must never reach the gauge (their mean β collapses toward 0.5 and reads a false
breach); they are used for composition only. `Runtime Matrices/A_Scoring_Module/test_a_score_canonical.py` and the kit's Jensen-gap
guard hold that line.

<!-- GENERATED:chain start -->
<!-- GENERATED:chain end -->

## Recording a run so it can be pooled later

```
python3 chain/MethylPhys_Interface/run_sample.py \
  --grn SAMPLE_Grn.idat.gz --red SAMPLE_Red.idat.gz \
  --age 61 --sex M --lab MYLAB --specimen "whole blood" \
  --covariate stage=II --covariate study=YOUR_STUDY \
  --intake-log custody/intake.jsonl --out reports/SAMPLE.html --id SAMPLE
```

Every run writes three things: the **report** (`--out`); the **bundle** beside it (`_bundle.json`: every stage's output — the
whole Stage 0 record, the Stage 1 detection and controls, composition and its cross-check, every present cell's A and tier, the
sky statistics); and one **ledger row** (`evidence_ledger.jsonl`, `--ledger` to place it) so a cross-specimen matrix is a file
read rather than a re-run. The column set grows with the cells found and the covariates passed — read the keys.

**Covariates are recorded, not reported.** `--covariate key=value` goes into the custody record, the bundle and the ledger row
and never reaches report prose: a reading states what was measured, and the phenotype a specimen was declared with is not a
measurement. The vocabulary guard enforces this.

**Provenance is on the page.** The Run tab opens with the chain commit, the decoder version and a SHA-256 of every input the
chain read — atlas, identity loci, pipeline map, tier table, thresholds, the chain modules themselves. Two readings are comparable
only if those hashes match.

Verification against known answers: [`../kit/`](../kit/) ([`release_check.py`](../kit/release_check.py) runs every guard). Regenerating every document that
describes this code: `python3 chain/build_all.py`.
