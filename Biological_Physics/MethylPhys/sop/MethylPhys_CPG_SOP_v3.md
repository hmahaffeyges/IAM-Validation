# MethylPhys CPG SOP — chain v3 (running today: neutrophils; full chain and commissioning order in §2b)

**Build:** development v3, 2026-10-01; frozen inputs re-checked against the code 2026-10-02; this SOP proofread against the code and the runtime files 2026-10-03. Not commissioned. Not a diagnostic test.
**Scope:** one cell, neutrophils, on Illumina EPIC v1 arrays. Other cells are added one at a time after each passes the three tests
(pure-cell precision, mixture recovery, known damage). 450K neutrophil floor: pending.
**Readings:** Met-A (arrays) and its C-score. IAM-A (sequencing) runs through a separate stage (Stage Q), which is in development; it has no C-score yet.

## 1. Physics stated once

- Per-site entropy: H(β) = −β log₂β − (1−β) log₂(1−β), in bits.
- **Met-A** = mean over the cell's identity sites of H(β), divided by the healthy reference for that same cell type on the same platform
  (isolated cells: the cell's own floor; whole blood: the composition-matched healthy expectation, §2 row M).
  A healthy cell holds its pattern at its reference, so Met-A = 1. Loss of pattern pulls β toward 0.5, which raises H, so Met-A rises.
- **Normal = 0.95–1.05.** One gauge for every reading. No other tier is printed until it has been measured on this scale.
- **C-score** = how clustered the per-site departures are along the genome, relative to healthy (healthy = 1).

## 2. Stages and code

Paths are relative to `Biological_Physics/MethylPhys/`. Order in the code (`chain/conductor_v3.py: run_neutrophil`): platform check → A (whole blood) → M →
noise index → T → noise gate → MC → report; Stage Q runs only on sequencing input.

| stage | what it does | code | frozen input |
|---|---|---|---|
| 0 Intake | manifest, hash, controls, detection p, bead count, call rate, sex check, decision gate | `chain/stage_0_intake.py` | `chain/Runtime Matrices/Intake/intake_thresholds_v1.json` |
| 1 Calibration | IDAT → noob β; probes at background (poobah p > 0.05) removed | `chain/stage_1_idat_calibration.py` | Illumina manifest (methylprep downloads it on first use) |
| A Composition (whole blood only) | 8 blood groups by NNLS on 963 markers, sum 1; the markers exclude the neutrophil sites; ≥ 867 of 963 measured (`MIN_MARKER_FRACTION` 0.9), else not solved and A withheld | `chain/conductor_v3.py: stage_a_composition` | `chain/Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json` |
| M Met-A | isolated neutrophils: H̄ / own floor. Whole blood: H̄ / H̄(e), where e = Σ f_g μ_g from the purified EPIC profiles, read when f_NEU ≥ 0.20 (`MIN_READ_FRACTION`). Both: ≥ 5400 of the 6000 identity sites measured (`SITE_COVERAGE_MIN` 0.9). Records the shift per 1 % loss of the neutrophil pattern (A recomputed on β + 0.01 × (0.5 − μ_NEU), times f_NEU in whole blood) and the entropy-ceiling flag | `chain/stage_m_met_a.py`, `chain/conductor_v3.py: stage_m_isolated, stage_m_blood` | `chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`, `chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`; whole blood: `blood_composition_EPIC_v1.json` (`profiles_at_neutrophil_sites`) |
| MC C-score | residual z_i = (H(β_i) − H(ref_i)) / s_i in genomic order (ref_i: healthy neutrophil mean H, or H(e_i) in whole blood; s_i: shrunk healthy SD); C = variance of the 50-site block means × 50 ÷ variance of z ÷ healthy median 1.1104; ≥ 10 blocks, else no C | `chain/conductor_v3.py: stage_mc_cscore` | `chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json` |
| T Tare | same-run healthy references of the same specimen type (same slide, else same batch), whole blood and isolated alike, ≥ 3 (`MIN_REFS`): A_rel = A ÷ median(reference A); spread = SD (ddof 1) of reference A ÷ median; detection limit = 2 × spread ÷ shift per 1 % loss; nothing is fitted; < 3 references → untared; a reference row carrying the specimen's own id is left out | `chain/conductor_v3.py: stage_t_tare` | — |
| Noise | noise index N = mean H(β) over the 48,528 noise sites measured on the array (≥ 90 %, `MIN_NOISE_FRACTION`; fewer → N not computed, gate `not measured`, nothing withheld); N > N_max 0.149 on an untared reading → gauge state withheld, A printed as a number | `chain/conductor_v3.py: noise_index, noise_gate, run_neutrophil` | `chain/Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json`, `chain/Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json` (N_max = top of the 6 reference arrays' N range 0.1223–0.1489, DEV-NOISE-01) |
| Q IAM-A (sequencing) | isolated copy error ε on qualifying molecules (≥ 6 calls, ≥ 80 % methylated); IAM-A = H(ε) ÷ (P × H(ε₀)); ≥ 100,000 opportunities; refuses any pipeline but the one P was measured on | `chain/stage_q_iam_a.py` (called by `chain/MethylPhys_Interface/run_sample.py` with `--pat` or `--site-table`) | `chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json` (`eps0` 0.032, `cells.neutrophils.P` 1.099, `pipeline` `loyfer_pat_v1`) |
| Report | one HTML page plus a JSON bundle; one row appended to the evidence ledger | `chain/MethylPhys_Interface/report_v3.py` (ledger row: `run_sample.py`) | — |

Frozen values (read from the files, never typed):
- EPIC neutrophil healthy reference 0.330263 bits (6 physical arrays, Salas GSE110554; GSE167998 re-deposits the same 6; 6000 sites; our Stage 1).
- Healthy clustering median 1.1104 (6 physical arrays, leave-one-out, block 50).
- Noise gate N_max 0.149 (`noise_gate_EPIC_v1.json`); IAM-A ε₀ 0.032 and neutrophil P 1.099 (`iama_positions_v1.json`).

## 2b. The full chain, and what runs today

The chain as designed has fourteen stages. v3 runs the ones marked **running**; the others are built and kept in the repository as the
toolkit (`chain/TOOLKIT.md`) and enter the chain one at a time at commissioning. A toolkit stage is not part of any reading until its own
pre-registered check on v3 has passed and the result is recorded in `doors/`.

| # | stage | what it does | status |
|---|---|---|---|
| 0 | Intake | manifest, hashes, controls, detection, sex and platform checks, decision gate | **running** |
| 1 | Calibration | IDAT to beta (noob) | **running** |
| 2 | Composition, blood groups | whole blood split into 8 purified blood groups (NNLS, 963 markers) | **running** |
| 3 | Atlas deconvolution | whole-tissue split into the atlas v2 cell types, each with its identifiability | toolkit (check failed 2026-10-03, DEV-ATLAS-EPIC-02) |
| 4 | NILC component separation | cell-type separation by internal linear combination, the CMB method that needs no template per component | toolkit (check failed 2026-10-03, DEV-NILC-01) |
| 5 | Met-A | each cell read against its own floor or its composition-matched healthy expectation | **running** (neutrophils) |
| 6 | C-score | clustering of the residual map in genomic order | **running** (band not set) |
| 7 | IAM-A | single-molecule reading on sequencing data | **running** (development; base-chain check on the constructed test data passed 2026-10-03, DEV-BASE-CHAIN-01 e) |
| 8 | Same-run tare | A_rel = A / median of same-run healthy references | **running** |
| 9 | Noise gate | noise index N; gauge state withheld above N_max untared | **running** |
| 10 | Directional decomposition | which way a departure points (toward disorder or toward over-order), per cell | toolkit (not assessable as built, DEV-DIRECTION-01; author decision) |
| 11 | Sky map | each site placed on the sphere (HEALPix), the residual map drawn per cell | toolkit (check failed 2026-10-03, DEV-SKY-01) |
| 12 | Sky statistics | angular power spectrum, masks, spatially shuffled null, look-elsewhere by simulation | toolkit (not run: look-elsewhere by simulation not built, DEV-SKYSTAT-01) |
| 13 | Report | HTML page and JSON bundle | **running** |

### Checked against the retired v2 report (2026-10-03)

Every step, tool and report section of the retired v2 report, and where it lives now. Added so nothing is lost; the author decides on each
item when the chain work starts.

| v2 report item | now |
|---|---|
| Stage 0 intake, Stage 1 calibration | stages 0, 1 (running) |
| Stage 1s pipeline scale map (affine) | replaced by the same-run tare (stage 8); the map was a fitted term |
| Stage 2 composition (constrained solver on the atlas) | stage 3 atlas deconvolution (toolkit; development run under way) |
| Stage 2b NILC second solver, with the lineage splitter | stage 4 (toolkit) |
| Stage 2c trace-class detection: inverse-variance score test that the non-negativity boundary cannot pin | **added below as stage 3b (toolkit)** |
| Stage 2d foreign-cell detection: matched-template fit of each foreign cell's profile to the residual, per-laboratory noise floor | **added below as stage 3c (toolkit)** |
| Stage 3 immune fine split (19 against 6 immune types) | part of stage 3 (atlas cell set) |
| presence floors as masks (a cell below its floor reports nothing) | gate of stages 3 and 5; mask of stage 12 |
| Every-cell table (identity sites, identifiability) | stage 5 per cell, as each cell is commissioned |
| Stage 4.5 bidirectional composite | stage 10 (toolkit) |
| Stage 4.6 patient sky, Mollweide plate, HEALPix | stage 11 (toolkit) |
| surface brightness | **added: stage 11b (toolkit)** |
| difference map of two draws from one person | **added: stage 12b (toolkit; serial reading)** |
| not built in v2: angular power spectrum, apodised mask, beam smoothing, cell-type covariance in the separation (GLS), Fisher degeneracy of the composition, ILC on the residual sky, per-specimen posterior for the composition, cross-spectra between cell panels | **listed under stage 12 as tools to build** |
| Stage 5 Mahalanobis departure against an age-matched band; age tab | retired: a comparison with a population |
| classes, tiers, 8 classes x 5 substrates chart, the 1.07 line | retired: class floors and tiers |
| report: red flags (STOP / WITHHELD / CAUTION / NOTE, also as JSON), safeguards (rendered-claim scan, formula self-test, anchors, deconvolver conformance, atlas separability), troubleshooting, integrity (file hashes), the chain's file inventory, run it yourself, cosmology-toolkit table with PASS / NOT_RUN / NOT_BUILT | **required sections of the v3 report (stage 13)**; report_v3 holds the reading only today |

Added stages: **3b trace-cell detection** (toolkit; not run, author decision), **3c foreign-cell detection** (toolkit; not run, author decision),
**11b surface brightness** (toolkit; not run, author decision), **12b difference map** (**running** with `--prior-betas` / `--prior-bundle` since
2026-10-03: per-address difference of two draws of one person and the same-person check, DEV-TOOLKIT-ADDED-01; the difference drawn as a sky is not built).
Commissioning record (stage, check, result, wired): `doors/CHAIN_COMMISSIONING.md`.

### Commissioning order (author approved 2026-10-03)

Each step: pre-register the check in `doors/` before reading data, run it on v3, record the outcome, then wire the stage in.

1. **NILC (stage 4).** Check: on constructed mixtures of purified cells, NILC recovers the known fractions within a pre-set error, and on
   the same-run replicates its fractions repeat within a pre-set spread.
2. **Atlas deconvolution (stage 3).** Check: as for NILC on constructed mixtures, and agreement with the 8-group composition on whole blood
   where both apply; each additional cell type is read only after its own floor and reference are commissioned ("no cell rather than part of one").
3. **Met-A for each newly commissioned cell type (stage 5),** one cell type at a time, with its own replicate test.
4. **Directional decomposition (stage 10).** Check: on replicates it returns no direction; on a known treated series it returns the
   direction the treatment is known to push.
5. **Sky map and residual maps (stage 11).** Check: healthy replicates give a residual sky consistent with the spatially shuffled null.
6. **Sky statistics (stage 12).** Check: the look-elsewhere correction by simulation holds its stated false-positive rate on healthy arrays.
7. **Stage Q, IAM-A (stage 7)** - commissioned with the base chain (stages 0, 1, 2, 5, 6, 8, 9, 13), before step 1. Check: on the bundled
   single-molecule test data (constructed; no real file is bundled) IAM-A = 1 at the healthy position, another pipeline is refused, the `.pat`
   extractor returns the constructed errors (DEV-BASE-CHAIN-01 e: passed 2026-10-03).
8. **Trace-cell detection (stage 3b).** Check written in DEV-TOOLKIT-ADDED-01; needs its line re-set without a population first (author).
9. **Foreign-cell detection (stage 3c).** Check written in DEV-TOOLKIT-ADDED-01; needs a line without a population and a stated beta scale (author).
10. **Surface brightness (stage 11b).** Check written in DEV-TOOLKIT-ADDED-01; needs a v3 per-site uncertainty source (author).
11. **Difference map (stage 12b).** Check: the same-person check accepts one person and refuses two; same-person replicate differences are below
    differences to other people in >= 95 % of comparisons (DEV-TOOLKIT-ADDED-01: passed 2026-10-03, 348/348; wired).

Order actually run on 2026-10-03: base chain with 7 -> 4 -> 3 -> 5 (B cells) -> 10 (not assessable) -> 11 -> 12 (not run) -> 3b, 3c, 11b (not run) -> 12b.

## 3. Rules the chain enforces

1. **Read line.** In whole blood, A is computed only when neutrophils are ≥ 20 % of the specimen (`MIN_READ_FRACTION`, DEV-LOWFRAC-01: below it a 1 % loss of the neutrophil pattern moves A by less than 0.01). Below that, the fraction is printed and A is withheld.
2. **Whole blood must be tared.** Untared whole-blood A carries a composition and laboratory offset, measured at −0.04 in one lab and +0.09 in another.
   The gauge state is printed only from A_rel. Isolated neutrophils are read against their own floor and are tared against same-run
   references the same way (array noise, measured on second-lab isolated cells, PROC-NEUT-TEST-01).
3. **Platform match.** The floor, the profiles and the specimen must be on the same platform. Without a frozen floor for the platform, the chain refuses.
   A β vector of 700,000 probes or fewer is refused as 450K or incomplete, so an EPIC v1 array that loses that many probes at detection is refused too.
   A 450K IDAT pair does not reach the platform check: Stage 0 quarantines it (array-type mismatch at 0.1 when EPIC_v1 is declared, else coverage at 0.7b).
4. **Quarantine stops the run.** A Stage 0 QUARANTINE produces no reading.
5. **No population term.** Nothing in a reading depends on a cohort, a classifier or a disease label.
6. **Noise gate.** An untared reading on an array whose noise index is above the reference arrays' range (N > 0.149) gets no gauge state;
   A is printed as a number. A tared reading is not withheld by the gate. Nothing is fitted to N. If fewer than 90 % of the noise sites are measured,
   N is not computed and the gate does not apply (`not measured`).

## 4. Running it (operator)

One specimen:
```
cd Biological_Physics/MethylPhys/chain/MethylPhys_Interface
python run_sample.py --grn S_Grn.idat --red S_Red.idat --engine v3 \
  --specimen "whole blood" --array-type EPIC_v1 --sex F --age 52 --id S001 --out S001.html
```
Isolated neutrophils: `--specimen "isolated neutrophils"`.
Tare, once ≥ 3 healthy references of the same specimen type on the same slide (else the same batch) have been read: add `--slide-ref-A 0.951,0.957,0.962`.
That list holds the references' untared A values. Or `--slide-ref-table refs.csv` (column `A`, optional `id`; the specimen's own id is left out).
Second draw of the same person (stage 12b): add `--prior-betas S000_betas.parquet --prior-bundle S000_bundle.json` (the earlier draw run with
`--save-betas`, and the same `--patient-id`); the bundle gets `difference_map` (per-address difference) or a refusal naming what differs.
An EPIC v2 array is refused at intake with a report (v3 reads EPIC v1 only); a machine whose methylprep manifest cannot load stops with
`ENVIRONMENT_MISSING_MANIFEST` (exit 3) and the array is not judged.
Output: `S001.html`, `S001_bundle.json` and one row appended to `evidence_ledger.jsonl` beside the report (`--ledger` to change). Exit 2 on a
Stage 0 QUARANTINE, with no report, bundle or ledger row.
Stage 1 needs the Illumina manifest: methylprep downloads it on first use into `$HOME/.methylprep_manifest_files/`; offline, place it there
(`doors/RUNBOOK.md` section 1). Without it Stage 0 stops with `ENVIRONMENT_MISSING_MANIFEST` (exit 3, since 2026-10-03; before, the pair was wrongly quarantined as `QUARANTINE_CORRUPT_IDAT`). In a batch,
run the first array alone so the download is not raced.

A batch, with the tare done automatically: `chain_tests/run_chain_acceptance.py`. Pass 1 reads every specimen. Pass 2 re-runs
each whole-blood specimen with the others in its batch (its group, leave-one-out, ≥ 3) as references; isolated specimens are not re-run.
It is written for the compute box (box paths, roster files, `chain_v3.tgz`). The same-slide healthy-reference runner of DEV-REPL-V3-01 is
`doors/data/DEV_REPL_V3_01_run/run_proc_repl_v3_01.py`, with its exact commands in `COMMANDS.md` beside it.

## 5. What has been shown (development)

| test | result |
|---|---|
| Held-out purified neutrophils (6 physical arrays; each read against the other 5, sites re-chosen) | 0.983-1.045, SD 0.020 |
| End to end, 22 IDAT pairs | 22/22 processed. Isolated neutrophils 6/6 Normal (in-floor, untared). Known mixtures tared 6/6 Normal. AML remission blood from another lab tared 5/5 Normal; 5 withheld under the 0.50 read line then in force (four of them, 0.28–0.47, are above today's 0.20 line and have not been re-run). Tare references were the other specimens of the same group (§6, 3.2) |
| Known damage, 2 % neutrophil pattern loss in mixtures (tared) | 6/6 above 1.05 (shift +0.061) |

Not yet shown: real healthy whole blood, repeat pairs, any disease. The C-score band is not set.

## 6. Detail held for operators (moved from the book, 2026-10-03)
The book states each step and why; the exact values, records, files and development numbers it used to print are kept here.

### §1 Add to v3 §2 — the line each stage applies (was book Table `tab:p4_gates`, Ch. "The chain of custody")

| stage | line | source | status |
|---|---|---|---|
| 0 Intake | call rate ≥ 0.98 PROCEED; 0.93–0.98 PROCEED_WITH_PENALTY (flagged); < 0.93 QUARANTINE, no reading | `Runtime Matrices/Intake/intake_thresholds_v1.json` `call_rate` | [in full §3; 0.93 not in v3] |
| 0 Intake | call rate = fraction of probes detected **and** with ≥ 3 beads; detected = Gaussian detection p = 1 − Φ((I − μ_bg)/σ_bg) against the array's negative-control background ≤ 0.01 | `chain/stage_0_intake.py` 0.5–0.7 | [in full §5] |
| 0 Intake | detection gate 0.5: fraction of probes at p ≤ 0.01 > 0.99 PASS, ≥ 0.93 BORDERLINE (penalty), else FAIL (quarantine); bead gate 0.6: ≥ 3 beads on ≥ 99.5 % of probes, else WARN (penalty); the call rate (0.7) counts p < 0.01 | `intake_thresholds_v1.json` `detection`, `bead`; `stage_0_intake.py` 0.5–0.7 | **[new]** |
| 0 Intake | bisulfite conversion ≥ 0.95: recorded, not gated (`PROVISIONAL_BS_THRESHOLD_UNCALIBRATED`); as written it refused all 731 healthy arrays it was tried on | `intake_thresholds_v1.json` `bisulfite_conversion.min` | [in full §3; the 731 count is new] |
| 0 Intake | a deferred detection or call-rate check quarantines (`intake_deferred:detection+call_rate`) | `stage_0_intake.py` 0.9 | [in full §5, §7] |
| 0 Intake | a QUARANTINE stops the run before calibration (0.1–0.9) or right after it (0.7b on the calibrated β); exit code 2; no report, no bundle, no ledger row | `run_sample.py:207-215, :236-249, :252-258, :328-337` | [in full §4.1, §5] |
| 0 Intake | a cleartext identifier (space, `@`, < 16 alphanumerics) → `QUARANTINE_MANIFEST_INVALID`; `run_sample.py` hashes `--patient-id` (else `--id`) first when it is shorter than 16 alphanumerics or not alphanumeric (SHA-256, first 32 hex), so through `run_sample.py` this quarantine does not fire. The hash is used in the intake record only; the report, bundle `sample_id` and ledger carry `--id` as given | `stage_0_intake.py` 0.2; `run_sample.py:179-181` | [in full §5] |
| 0 Intake | `--betas` input: no intake; bundle `intake` is null; `intake_skipped` is set only by `--no-intake` (false for `--betas`); the report prints intake as `not run` | `run_sample.py:168-169, :363` | [in full §4.1, §6 item 3] |
| 1 Calibration | `methylprep.run_pipeline(betas=True, export=True, save_control=True, poobah=True)`; keeps `cg` probes with poobah p ≤ 0.05; Stage 1 also records poobah detection and a poobah × bead call rate, not gated | `chain/stage_1_idat_calibration.py:129-130` | [in full §5] |
| platform | EPIC v1 only; refused when the array type is not EPIC_v1, when probe names carry the EPIC v2 design suffix, or when the β vector holds ≤ 700,000 probes | `conductor_v3.py: platform_refusal` | [in full §5] |
| A Composition | ≥ 867 of 963 markers measured (`MIN_MARKER_FRACTION` 0.9), else composition not solved and A withheld | `conductor_v3.py` | [in full §3, §5] |
| A Composition | marker rule: not a neutrophil identity site; within-group SD ≤ 0.05; margin ≥ 0.25 against every other group's mean; top 150 per group; NNLS, sum 1. Markers per group (re-derived 2026-10-03 by the builder's rule; union = the 963 frozen markers; recorded in the file's `_meta`): CD4 T 41, CD8 T 22, each other group 150. The 44 / 20 printed here before were not reproduced | `blood_composition_EPIC_v1.json` key `rule` | **[new]** |
| M Met-A | whole blood read when f_NEU ≥ 0.20 (`MIN_READ_FRACTION`); ≥ 5400 of 6000 identity sites measured (both cases) | `conductor_v3.py`, `stage_m_met_a.py` (`SITE_COVERAGE_MIN` 0.9) | [0.20 in v3 §3 as 20 %; 5400 in full §5] |
| M Met-A | identity-site rule (also kept in book; EPIC v1 neutrophils, 6 physical arrays): across-array SD of β ≤ 0.05; mean β 0.75–0.95 (methylated channel) or 0.05–0.25 (unmethylated channel); ≤ 3,000 per channel by smallest SD; result 6000 sites, 3000 per channel | `chain_tests/freeze_v13.py`, `stage_m_met_a.py`; `metA_floors_v1_3.json` `n_sites` | **[new]** (rule; the count is in full §3). The rule also stays in the book, Ch. "Identity sites" |
| M Met-A | ceiling flag `past_entropy_ceiling` = (mean β at sites with μ_NEU > 0.5) < 0.5; report: "read beta, not A" | `conductor_v3.py:127-133` (`_ceiling`) | [in full §1, §6] |
| MC C-score | blocks of 50 sites; ≥ 10 blocks; healthy baseline 1.1104 | `neutrophil_reference_v1_1.json` | [in full §3, §5] |
| Q IAM-A | pipeline `loyfer_pat_v1` only; P = 1.099; ε₀ = 0.032; ≥ 100,000 opportunities; halves A/B with > 50,000 each; molecule qualifies with ≥ 6 calls and ≥ 80 % methylated | `chain/stage_q_iam_a.py`, `chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json` | [in full §3, §5; Stage Q row added to v3 §2 on 2026-10-03] |

**Stage Q row (added to v3 §2 on 2026-10-03):**

| stage | what it does | code | frozen input |
|---|---|---|---|
| Q IAM-A (sequencing) | isolated copy error ε on qualifying molecules (≥ 6 calls, ≥ 80 % methylated); IAM-A = H(ε) ÷ (P × H(ε₀)); ≥ 100,000 opportunities; refuses any pipeline but the one P was measured on | `chain/stage_q_iam_a.py` (called by `run_sample.py:351-361`) | `chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json` (`eps0` 0.032, `cells.neutrophils.P` 1.099, `pipeline` `loyfer_pat_v1`) |

**Add to v3 §2, frozen-input list:** `noise_gate_EPIC_v1.json` (above); `iama_positions_v1.json` (in §2 since 2026-10-03); `metA_floors_v1_3.json` key `duplicates_removed` (the six GSE110554 / GSE167998 pairs). **[in full §3 except noise_gate]**

**Add to v3 §2, noise sites:** rule = every purified blood group mean β ≤ 0.03 (40,882 sites) or ≥ 0.97 (7,646 sites), every group SD ≤ 0.02, no
neutrophil identity site; 48,528 sites; N computed when ≥ 90 % are measured. **[rule and count in full §3; the 40,882 / 7,646 split is new]**

**Add to v3 §2, development floors not read:** `chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_2_ALLCELLS_development.json` (other cell types; development only, not read by v3). **[new]**

---

## §2 Add to v3 §4 (operator) — names the book used

- Entry point `chain/MethylPhys_Interface/run_sample.py --engine v3`; the stages after calibration are in `chain/conductor_v3.py`; report `chain/MethylPhys_Interface/report_v3.py`. [in v3 §2/§4]
- Covariates: `--covariate KEY=VALUE`; kept in the custody record, bundle and ledger; no stage reads them; a free-text covariate appears in the report's bundle block. [flag in full §4.1; the "appears in the bundle block" sentence is **new**]
- Report state strings. Met-A: `untared: read A_rel (Stage T)` (whole blood), `untared (own-floor state: <Normal|above Normal|below Normal>): read A_rel (Stage T)` (isolated, drawn against the own floor), `tared: read A_rel (Stage T)`, `withheld: noise index <N> > 0.149 and no same-run tare; A printed as a number only`. Tare: `Normal`, `above Normal`, `below Normal`, or the reason `untared: <n> same-run reference arrays (>= 3 required)`. Untared whole blood: no gauge position. [in full §5, §6]
- Batch runner: `chain_tests/run_chain_acceptance.py` (pass 1 every specimen; pass 2 each whole blood against the others of its batch); `chain_tests/chain_batch.py` (historical PROC-NEUT-TEST-01 runner; its docstring now states the median tare - Stage T reads only `A` from the reference records). The standard runner is `run_chain_acceptance.py`. Both are box scripts. [in v3 §4, full §4.3]
- Serial mode: pure functions in `chain/serial_mode.py`, not wired into `run_sample.py`; design `doors/PROC_SERIAL_01_PREREG.md`; a pair is refused unless both draws share the identifier hash, array type and pipeline. **[new]**
- Build guards: `chain/build_all.py` regenerates the chain sequence, the operations manual PDF, the repository inventory, the RUNBOOK marked block and the folder READMEs, then gates on the link check and `build_chain_sequence.py --check`; it hashes the operator chapter and `report_v3.py` into `chain/GENERATED_MANIFEST.json`. It runs no vocabulary scan and no procedure reconciliation, and no gate checks the v3 SOP text against the code. **[new]**
- The operations manual path the book cited, `manual/OM_v3/OM_v3.tex`, **does not exist at `399c0e4`**; the manual present is `manual/OM_v3_neutrophil_chain.md`. The book now points to the SOP only. **[new; correct the path wherever it is kept]**

---

## §3 Add to v3 §5 (What has been shown) — development numbers taken out of the book

All are development values, not commissioning results. Source records under `doors/` unless stated.

### 3.1 Intake (book Ch. "The instrument layer")
| record | value |
|---|---|
| `PROC_INTAKE_01_OUTCOME.md` | 0.93 call-rate line set from 48 arrays, four laboratories: medians 0.985 (min 0.979), 0.975 (min 0.891), 0.953 (min 0.932); fourth laboratory 0.878 (max 0.928). The line sits between the fourth's best and the third's worst; one array of the second laboratory (0.891) falls below it. Measured on poobah p ≤ 0.05, not the gate's statistic; not yet re-measured. Bar B5: a deferred check quarantines. Diagnostic after bar B3: probes at background read β 0.33–0.42. Voided early run (stray process writing to the same log) → rule: two runs never write one output path. |
| `PROC_STAGE0_02_OUTCOME.md` | four wiring defects; one made every array of a 732-array set quarantine as cleartext identifier |
| `FINDING_GSE125105_LOW_SIGNAL.md` | one laboratory at 0.15–0.31 of three others' median control signal, 12.5 % of probes at background, cleared intake while the call-rate check was deferred |
| `PROC_TARE_01_OUTCOME.md` | SNP-probe tare, 768 arrays, four laboratories: not commissioned |

### 3.2 Acceptance run 3 (`chain_tests/CHAIN_V3_ACCEPTANCE_RUN3.md`, `chain_tests/chain_acceptance.csv`) — [in v3 §5 row 2 and full §8.1; add the per-group table]
| specimens (book keeps n, intake, untared and in-Normal columns; tared column removed) | n | intake | untared Met-A | tared A_rel | Normal |
|---|---|---|---|---|---|
| purified healthy neutrophils, isolated | 6 | 6 PROCEED | 0.994–1.006 | — | 6/6 |
| DNA mixtures, ≥ 50 % neutrophils | 6 | not run | 0.943–0.968 | 0.987–1.021 | 6/6 tared |
| remission blood, another laboratory | 10 | 2 PROCEED, 8 PROCEED_WITH_PENALTY | 1.073–1.115 (5) | 0.986–1.032 | 5/5 tared |

Run used a 0.50 read line: 5 remission bloods withheld at 0.06–0.47; four (0.28–0.47) are read at 0.20 and have not been re-run.
Tare references were the other specimens of the same group, not healthy references; isolated specimens were not tared.
C-score on this run: isolated 0.69–1.21 (n 6), mixtures 0.91–1.49 (n 6), remission bloods 0.78–1.32 (n 5); whole blood overall 0.78–1.49.
22 report pages; gauge positions as drawn in book Fig. `fig:p4_reportgauge`.

### 3.3 Development run 3 (`doors/DEV_CHAIN_V3_RUN3.md`, `doors/data/chain_v3_dev3_readings.csv`) — **[new to v3]**
Run 3 used the noise-corrected tare that has since been removed (§0). Its numbers are superseded for the tare and are recorded here as history.
- Values below are as the book printed them, not re-checked against the record. 690 report pages: Normal 601, above Normal 25, below Normal 11, untared (no same-run references) 45, A withheld for fraction or sites 8 (book Fig. `fig:p4_states`).
- Shift per 1 % loss on 682 specimens; detection limit on 637 tared specimens: median 1.69 %, range 0.88–5.31 % (book Fig. `fig:p4_detlimit`).
- Technical replicates GSE250556 (noise-corrected tare). Record `DEV_CHAIN_V3_RUN3.md` (table "What it shows"): read 63, tared 63, tared A_rel SD 0.013, in Normal 62/63. The book at `399c0e4` printed different figures for the same run: 63 of 64 arrays tared, 0.967–1.026, within-person SD 0.008, all-array SD 0.013, 63 of 63 in Normal (Fig. `fig:p4_replicates`, Table `tab:p4_changefloor`, map row 22). **Book and record disagree on the in-Normal count (63/63 vs 62/63)**, and the record does not show the 0.967–1.026 range or the 0.008 SD. Reconcile against `doors/data/chain_v3_dev3_readings.csv` before quoting either. Both are superseded by §3.4.

### 3.4 Replicate precision with the median tare (`doors/DEV_REPL_V3_01.md`, 2026-10-03) — **[new to v3]**
GSE250556, 64 arrays, chain commit 87cfa65: 63/64 read end to end, 63/63 tared; within-person SD (pooled) 0.037; 48 of 63 in Normal (8 below, 7 above);
N above 0.149 on 59 of 63 (0.137–0.195). The 0.008 / 63-of-63 figures came from the fitted tare that was removed. The book now says only that
replicate precision "is being measured in development".

### 3.5 Pre-registered battery `PROC_NEUT_TEST_01_OUTCOME.md` and `PROC_NEUT_TEST_01_T2_OUTCOME.md` — [record in full §8.3; add bar codes]
692 arrays. T1a, T1b passed (fraction vs flow cytometry median 0.035, bar 0.05; > half neutrophils 6/6). T1c failed (untared A in 0.93–0.98 on 2 of 3).
T2 failed (second-laboratory isolated neutrophils 0.86–1.26; repeat diagnostic `doors/data/t2_diag.csv`: 48 arrays, within-person SD 0.045 / 0.044;
pre-registered score on 33 arrays). T3 not assessable (GSE250556, fraction 0.30–0.56, A withheld on 63 of 64 under the 0.50 line).
T4: 570 whole bloods, tared healthy SD 0.052, untared ≈ 1.22; T4a passed as written, a composition effect (fraction 0.79 vs 0.65; severity +0.0075, p 0.42); T4c failed.

### 3.6 Other records the book named (now named here only)
| record | what the book used it for |
|---|---|
| `DEV_LOWFRAC_01_OUTCOME.md`, `doors/data/lowfrac_readings.csv` | basis of the 0.20 line: 656 arrays (644 whole bloods, 12 mixtures); 96 healthy adults at ≥ 0.40; 5 healthy arrays below 0.40; mixtures median fraction error 0.034 [record in full §8.3] |
| `DEV_NOISE_01_OUTCOME.md`, `doors/data/noise_index.csv` | noise index table, 60 arrays (12 reference rows = 6 arrays × 2 deposits), ρ 0.79 / 0.83 [in full §8.3] |
| `DEV_NOISE_02_OUTCOME.md` | 76 healthy arrays (GSE179325, neutrophils ≥ 50 %), A ≈ 0.60 + 0.205 f_neu + 2.51 N, R² 0.85, fitted after looking; 33 of 495 arrays inside the reference N range; not used by the chain [in full §8.3; add "not used"] |
| `chain_tests/WHOLE_BLOOD_COMPOSITION_DEV.md`, `doors/data/selfconsist.csv` | planted 2 % loss on 6 mixtures: tared +0.061, 6/6 above 1.05; re-fitting the fraction on neutrophil sites: 0/6; multi-platform solver under-read EPIC neutrophils by ≈ 0.05 [in full §8.1] |
| `PROC_WB_NEUT_01_OUTCOME.md` | bars W1 6/6, W3 5/6; mixtures 0.982–1.016 vs 1.062–1.118 [in full §8.3; bar codes new] |
| `DIAG_450K_01_OUTCOME.md` | 450K purified cells on EPIC references 0.904–0.932, 0/8 Normal; 450K reference 8/8 Normal **[new]** |
| `PROC_DNMT_01_PARTA_OUTCOME.md`, `doors/data/dnmt_arrays_readings.csv`; `PROC_DNMT_01_PARTB_OUTCOME.md`, `doors/PROC_DNMT_01_PARTB/` | DNMT1 inhibitor on arrays and single molecules; Part B bars Q1, Q2 passed [Part A in full §8.3; Part B new] |
| `PROC_AML_SERIAL_01_OUTCOME.md` | bars S1–S4 failed, S5 passed (remission pairs 10/10 within 0.05); S2 (diagnosis on neutrophil sites 6/10 in Normal); bar 8 of 10 **[new]** |
| `PROC_AML_PROG_01_OUTCOME.md` | 450K progenitor floors, 16 of 29 outside Normal (bar 80 %) **[new]** |
| `PROC_PREDX_NEUT_01_OUTCOME.md`, `PROC_PREDX_SLIDE_01_OUTCOME.md` | 450K development floor; sex split; same-slide tare 92.9 % of 170 [first in full §8.3; second new] |
| `PROC_CEIL_01_OUTCOME.md` | Finding 1 (absent cells read near the far end, 318 arrays); Finding 3 (whole-array sky, 1.31×, 57σ); ceiling guard 492 readings / 318 arrays **[new]** |
| `PROC_MF_01_OUTCOME.md` | trace-cell detection from 2–5 % **[new]** |
| `PROC_OUTSPAN_01_PREREG.md` | out-of-span map tests, pre-registered, not run **[new]** |
| `PROC_TUMOUR_01_OUTCOME.md`, `doors/data/tumour_pairs.csv` | tumour–normal pairs on single molecules **[new]** |
| `PROC_MOLECULE_01_OUTCOME.md` | one statistic for floor and reading; plasma constructed mixtures **[new]** |
| `PROC_CHANNEL_01_OUTCOME.md` | 399 windows, 153 samples, 56 cell types; 3.41 ± 0.12 kT **[new]** |
| `PROC_V5_HELDOUT_OUTCOME.md`; `atlas/v2/postbuild/README.md` (check V12) | atlas v2 held-out coverage (V5); identity sites of later cell types (V12) **[new]** |
| `PROC_LINES_02_channels/imr90_channels.csv` | IMR90 channel readings **[new]** |
| `DEV_STOOL_01_OUTCOME.md` | 29 regions / 227 CpGs for lower-gut epithelium **[new]** |
| `PLAN.md` (incl. item 22) | order of work; E-MTAB-7309: 738 of 1,056 below 0.93, median call rate 0.894; no canine purified reference found **[new]** |
| `CMB_TO_METHYLOME_MAP.md` | the 79-row map **[new]** |
| `PROC_TUMOUR_01_OUTCOME.md`, `PROC_PREDX_SLIDE_01_OUTCOME.md` | examples of an error recorded in the outcome, pre-registration unchanged **[new]** |
| salmonid records PROC-SALMON-01, PROC-CHARR-01, PROC-RIMOUSKI-01, DEV-COHO-CC-01; `doors/data/salmon_readings.csv`, `charr_readings.csv`, `rimouski_readings.csv`, `coho_cc_fish.csv` | fish chapters; these records sit under `../Salmonid/` (e.g. `../Salmonid/DEV_COHO_CC_01/coho_cc_fish.csv`), not under `doors/`; they are not chain v3 records and may not belong in this SOP **[new]** |

### 3.7 Figure and analysis scripts the book named
figsky/make_sky_figs.py (path on the compute box; not in the repository; repository copy `reference_floors_v1/sky/make_sky_figs.py`); box jobs `fcb79e1e` (remote_jobs/sky6/sky_neut6.py (path on the compute box; not in the repository); repository copy `reference_floors_v1/sky/sky_neut6.py`) and `f754cf20` (remote_jobs/gate/cd_neut.py (path on the compute box; not in the repository); repository copy `reference_floors_v1/sky/cd_neut.py`);
`docs/book/figscripts/fig_chain_v3_flow.py`, `fig_p4.py --tables`, `p4carry_precedence_search.py` (precedence search, 2026-10-02);
`docs/verification/scripts/verify_astrogenetics_book.py`; derivation checks 12/12 at commit `8d3ab37`.
**[new]** These are provenance, not operator steps; they may belong in the book's provenance appendix (`docs/book/appendices/app_I_provenance.tex`
already lists figure scripts) rather than the SOP. Author to decide.

---
