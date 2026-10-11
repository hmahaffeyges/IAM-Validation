# MethylPhys CPG SOP — chain v3 (running today: neutrophils; full chain and commissioning order in §2b)

**Build:** development v3, 2026-10-01; frozen inputs re-checked against the code 2026-10-02; this SOP proofread against the code and the runtime files 2026-10-03; development round 2 (author decisions A-O) written in 2026-10-04; Stage T changed on 2026-10-04 to self-tare II then the median tare, adopted by the author (`boxruns/run1/JOBS.md` job A; DEV-SELFTARE-02; development log 2026-10-04, DEV-PAIRED-01; §2 row T and §2b "Tare: self-tare II"); self-tare II wired into Stage T on 2026-10-04 (development log DEV-SELFTARE-03). **DEVELOPMENT - not commissioned.** Not a diagnostic test.
**Scope:** one cell, neutrophils, on Illumina EPIC v1 arrays. Other cells are added one at a time after each passes the new-cell rule
(three tests: purified-cell Normal, replicate spread, identifiability; §2b). 450K neutrophil floor: pending.
**Readings:** Met-A (arrays) and its C-score. IAM-A (sequencing) runs through a separate stage (Stage Q), which is in development; it has no C-score yet.

## 1. Physics stated once

- Per-site entropy: H(β) = −β log₂β − (1−β) log₂(1−β), in bits.
- **Met-A** = mean over the cell's identity sites of H(β), divided by the healthy reference for that same cell type on the same platform
  (isolated cells: the cell's healthy reference; whole blood: the composition-matched healthy expectation, §2 row M).
  A healthy cell holds its pattern at its reference, so Met-A = 1. Loss of pattern pulls β toward 0.5, which raises H, so Met-A rises.
- **Normal = 0.95–1.05.** One gauge for every reading. No other tier is printed until it has been measured on this scale.
- **C-score** = how clustered the per-site departures are along the genome, relative to healthy (healthy = 1).

## 2. Stages and code

Paths are relative to `Biological_Physics/MethylPhys/`. Order in the code (`chain/conductor_v3.py: run_neutrophil`): platform check → A (whole blood) → M →
noise index → T → noise gate → MC → report; Stage Q runs only on sequencing input. Self-tare II (Stage T step 1, adopted 2026-10-04) acts on β before A and M;
in the code it is in this order (wired 2026-10-04, `stage_t_selftare_ii`, called before A and M); the noise index reads β before self-tare II (row T).

| stage | what it does | code | frozen input |
|---|---|---|---|
| 0 Intake | specimen rule first (whole blood and isolated / sorted / purified neutrophils accepted; every other specimen refused with a report, `specimen_refusal`); manifest (sex and age optional, recorded when given), hash, controls, detection p, bead count, call rate, sex check (`NOT_DECLARED` when no sex is given), decision gate; identifiers hashed in the bundle and the ledger (the report keeps the typed id) | `chain/stage_0_intake.py`, `chain/MethylPhys_Interface/run_sample.py` | `chain/Runtime Matrices/Intake/intake_thresholds_v1.json` |
| 1 Calibration | IDAT → noob β; probes at background (poobah p > 0.05) removed | `chain/stage_1_idat_calibration.py` | Illumina manifest (methylprep downloads it on first use) |
| A Composition (whole blood only) | 8 blood groups by NNLS on 963 markers, sum 1; the markers exclude the neutrophil sites; ≥ 867 of 963 measured (`MIN_MARKER_FRACTION` 0.9), else not solved and A withheld | `chain/conductor_v3.py: stage_a_composition` | `chain/Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json` |
| M Met-A | isolated neutrophils: H̄ / healthy reference. Whole blood: H̄ / H̄(e), where e = Σ f_g μ_g from the purified EPIC profiles, read at any f_NEU > 0 with its detection limit. Both: ≥ 5400 of the 6000 identity sites measured (`SITE_COVERAGE_MIN` 0.9). Records the shift per 1 % loss of the neutrophil pattern (A recomputed on β + 0.01 × (0.5 − μ_NEU), times f_NEU in whole blood) and the entropy-ceiling flag | `chain/stage_m_met_a.py`, `chain/conductor_v3.py: stage_m_isolated, stage_m_blood` | `chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`, `chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`; whole blood: `blood_composition_EPIC_v1.json` (`profiles_at_neutrophil_sites`) |
| MC C-score | residual z_i = (H(β_i) − H(ref_i)) / s_i in genomic order (ref_i: healthy neutrophil mean H, or H(e_i) in whole blood; s_i: shrunk healthy SD); C = variance of the 10-site block means × 10 ÷ variance of z ÷ healthy median 1.0103 (blocks of 10 since 2026-10-09; was 50, median 1.1104); ≥ 10 blocks, else no C | `chain/conductor_v3.py: stage_mc_cscore` | `chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_2.json` |
| T Tare | **self-tare II, then the median tare** (adopted by the author 2026-10-04; DEV-SELFTARE-02 reading (iv), author decision G; DEV-PAIRED-01). Step 1, self-tare II, on this array's own β: for each probe design (type I, type II) the anchors L, U = mean β over the array's low and high fixed sites of that design; β′ = Lr + (β − L)(Ur − Lr)/(U − L), with Lr, Ur the same anchors averaged over the six reference arrays; Met-A is then formed from β′ as rows A and M form it; no references needed; nothing is fitted; a design whose anchors are missing or with U − L ≤ 0.1 is left unmapped (code). Step 2, the median tare: same-run healthy references of the same specimen type (same slide, else same batch), whole blood and isolated alike, ≥ 3 (`MIN_REFS`): A_rel = A ÷ median(reference A); spread = SD (ddof 1) of reference A ÷ median; detection limit = 2 × spread ÷ shift per 1 % loss; nothing is fitted; < 3 references → step 2 does not run; a reference row carrying the specimen's own id is left out. **In the code** (wired 2026-10-04) step 1 is `stage_t_selftare_ii`, before A and M, recorded as `tare.selftare_ii`; the noise index and the noise gate read β before step 1; step 2 is `stage_t_tare` on the self-tared A; `--dev-selftare-ii` is a no-op alias. Earlier Stage T, kept as history: the median tare alone (2026-10-02 to 2026-10-04, DEV-TARE-02); before that the noise-corrected tare (removed 2026-10-02, DEV-TARE-02) | `chain/conductor_v3.py: stage_t_tare`; step 1: `chain/conductor_v3.py: stage_t_selftare_ii` (`chain/dev_stages.py: anchors, selftare_map`) | step 1: `chain/Runtime Matrices/Development/dev_selftare_typeII_EPIC_v1.json` (development file: fixed sites by design and state, reference anchors) |
| Noise | noise index N = mean H(β) over the 48,528 noise sites measured on the array (≥ 90 %, `MIN_NOISE_FRACTION`); fewer → N not computed and the gauge state is withheld, tared or not, with the counts and the reason in plain words (author decision A, 2026-10-04); N > N_max 0.149 on an untared reading → gauge state withheld, A printed as a number | `chain/conductor_v3.py: noise_index, noise_gate, run_neutrophil` | `chain/Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json`, `chain/Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json` (N_max = top of the 6 reference arrays' N range 0.1223–0.1489, DEV-NOISE-01) |
| Q0 IAM-A intake (sequencing) | the .pat file decompresses to its end with 4 fields per line; every CpG index inside its chromosome's hg19 range (`hg19_cpg_chrom_ranges.json`); molecules on all 22 autosomes; specimen = purified neutrophils (or blood granulocytes). Conversion, share of molecules with ≥ 6 calls and duplicate fraction recorded; their limits are set from healthy files before any test file (development) | `chain/stage_q0_intake.py` | QUARANTINE_UNREADABLE_PAT, GENOME_BUILD_MISMATCH, QUARANTINE_NOT_WHOLE_GENOME, SPECIMEN_REFUSED |
| Q IAM-A (sequencing) | isolated copy error ε on qualifying molecules (≥ 6 calls, ≥ 80 % methylated); IAM-A = H(ε) ÷ (P × H(ε₀)); ≥ 100,000 opportunities; refuses any pipeline but the one P was measured on. IAM-A C-score (development, decision C): blocks of 1,000 sites in genomic order, C = Σ(k_b − ε o_b)² / Σ ε(1 − ε) o_b; independent errors give 1 (derived); one C per A and per half; band not set | `chain/stage_q_iam_a.py` (called by `chain/MethylPhys_Interface/run_sample.py` with `--pat` or `--site-table`) | `chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json` (`eps0` 0.032, `cells.neutrophils.P` 1.099, `pipeline` `loyfer_pat_v1`) |
| Report | one HTML page plus a JSON bundle; one row appended to the evidence ledger | `chain/MethylPhys_Interface/report_v3.py` (ledger row: `run_sample.py`) | — |

Frozen values (read from the files, never typed):
- EPIC neutrophil healthy reference 0.330263 bits (6 physical arrays, Salas GSE110554; GSE167998 re-deposits the same 6; 6000 sites; our Stage 1).
- Healthy clustering median 1.1104 (6 physical arrays, leave-one-out, block 50).
- Noise gate N_max 0.149 (`noise_gate_EPIC_v1.json`); IAM-A ε₀ 0.032 (the healthy reference copy error, H_ref = H(ε₀) = 0.2043 bits, the holding energy 3.41 kT measured across 56 healthy cell types; not a lower limit) and neutrophil P 1.1492, measured on whole files (`iama_positions_v2.json`, 2026-10-08).
- Gauge marks: the floor H_min is where thermal kicks win against one ATP per site, copy error 1/(1+e^M) = 8.1×10⁻¹⁰, 2.5×10⁻⁸ bits (1×10⁻⁷ on the neutrophil IAM-A gauge); A = 1 is the healthy reference; the full surface is 1 bit per site (neutrophils: Met-A 3.03, IAM-A 4.26). No reading is refused for lying below H_ref.
- Met-A C-score tare: C_rel = C ÷ median C of ≥ 3 same-run healthy references. Development band 0.877–1.152 (blocks of 10; 2.5–97.5 % of tared C on 641 healthy arrays from 19 laboratories; `doors/CSCORE_COMMISSIONING_PLAN.md`).
- IAM-A tare (2026-10-09, development): A_rel = IAM-A ÷ median IAM-A of ≥ 3 same-run healthy references of the same cell, laboratory, library kit and pipeline (`stage_q_iam_a.tare`, `run_sample.py --iama-ref-table`); fewer than 3 → untared, read against P only. Reason: library kit and laboratory shift IAM-A by up to 0.16 on healthy cells (DEV-IAMA-KIT-01).

## 2b. The full chain, and what runs today

The chain as designed has fourteen stages. v3 runs the ones marked **running**; the others are built and kept in the repository as the
toolkit (`chain/TOOLKIT.md`) and enter the chain one at a time at commissioning. A toolkit stage is not part of any reading until its own
pre-registered check on v3 has passed and the result is recorded in `doors/`.

| # | stage | what it does | status |
|---|---|---|---|
| 0 | Intake | manifest, hashes, controls, detection, sex and platform checks, decision gate | **running**; round 2: sex and age optional, blood specimens only, ids hashed in bundle and ledger (DEV-INTAKE-02: 1,569/1,569 end to end, 955 refused naming the specimen, 0 typed ids in bundles or ledgers) |
| 1 | Calibration | IDAT to beta (noob) | **running**; detection stays poobah (DEV-DETECTION-01: poobah better in 31 strata, the Gaussian negative-control test in 0, neither in 24) |
| 2 | Composition, blood groups | whole blood split into 8 purified blood groups (NNLS, 963 markers) | **running** |
| 3 | Atlas deconvolution | whole-tissue split into the atlas v2 cell types, each with its identifiability | development flag `--dev-atlas-e` (DEV-ATLAS-EPIC-02; DEV-COMPOSITION-TRUTH-02: within 0.03 on another laboratory's mixtures except granulocytes 0.041) |
| 4 | NILC component separation | cell-type separation by internal linear combination, the CMB method that needs no template per component | development flag `--dev-nilc` (DEV-NILC-01, DEV-COMPOSITION-TRUTH-02: outside the truth bars) |
| 5 | Met-A | each cell read against its healthy reference or its composition-matched healthy expectation | **running** (neutrophils); B cells behind `--dev-percell-b`; monocytes and B cells do not meet the new-cell rule (DEV-NEWCELL-01) |
| 6 | C-score | clustering of the residual map in genomic order | **running** (band not set); IAM-A C-score in Stage Q (development, DEV-IAMA-CSCORE-01) |
| 7 | IAM-A | single-molecule reading on sequencing data | **running** (development; constructed test data 2026-10-03, DEV-BASE-CHAIN-01 e; real Loyfer granulocyte files 2026-10-04, DEV-IAMA-REAL-01) |
| 8 | Same-run tare | self-tare II on the array's own fixed sites, then A_rel = A / median of same-run healthy references | **running** (self-tare II, then the median tare); self-tare II then median tare adopted by the author as the Stage T reading on 2026-10-04 (it met every replicate and other-laboratory bar, DEV-SELFTARE-02; DEV-PAIRED-01); self-tare II wired into Stage T on 2026-10-04 (DEV-SELFTARE-03) |
| 9 | Noise gate | noise index N; gauge state withheld above N_max untared, and withheld whenever fewer than 90 % of the noise sites are measured | **running** |
| 10 | Directional decomposition | which way a departure points (toward disorder or toward over-order), per cell | development flag `--dev-direction` (rebuilt physics-only, DEV-DIRECTION-02: known loss of methylation 12/12 toward disorder; replicates 56/63 no direction) |
| 11 | Sky map | each site placed on the sphere (HEALPix), the residual map drawn per cell | development flag `--dev-sky` (DEV-SKY-02: against the within-chromosome block-shuffle null three of six bands inside 0.9-1.1) |
| 12 | Sky statistics | angular power spectrum, masks, block-shuffle null, look-elsewhere by simulation | development flag `--dev-sky` (built 2026-10-04; DEV-SKY-02: look-elsewhere rate 91 % on healthy arrays, bar 8.4 %) |
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
| Stage 5 Mahalanobis departure against an age-matched band; age tab | retired: read against other people's readings, not the cell's healthy reference |
| classes, tiers, 8 classes x 5 substrates chart, the 1.07 line | retired: class floors and tiers |
| report: red flags (STOP / WITHHELD / CAUTION / NOTE, also as JSON), safeguards (rendered-claim scan, formula self-test, anchors, deconvolver conformance, atlas separability), troubleshooting, integrity (file hashes), the chain's file inventory, run it yourself, cosmology-toolkit table with PASS / NOT_RUN / NOT_BUILT | **required sections of the v3 report (stage 13)**; report_v3 holds the reading only today |

Added stages: **3b trace-cell detection** (`--dev-trace`, rebuilt on the array's own noise, DEV-TOOLKIT-ADDED-02), **3c foreign-cell detection** (`--dev-foreign`, same),
**11b surface brightness** (`--dev-brightness`, same), **12b difference map** (**running** with `--prior-betas` / `--prior-bundle` since
2026-10-03: per-address difference of two draws of one person and the same-person check, DEV-TOOLKIT-ADDED-01; the difference drawn as a sky is not built).
Commissioning record (stage, check, result, wired): `STATUS.md` (section 8, commissioning record).

### Commissioned stages

- **Met-A on neutrophils, EPIC v1, commissioned 2026-10-09** (`doors/COMMISSIONING_NOTE_METAA_NEUTROPHILS.md`): stages 0, 1, 2, 5, 6, 8
  (self-tare II then the median tare), 9, 13; isolated neutrophils and whole blood (any neutrophil fraction, each reading with its own detection limit). Every report prints the
  detection limits: 2 % loss of the neutrophil pattern in purified neutrophils, 5 % in whole blood (DEV-METAA-SENS-01). Readings in this scope
  are results. The Met-A C-score, IAM-A and every other stage remain development.

### Reproducibility rule (2026-10-09)
Every number in the book, a note in `doors/`, STATUS or the development log must be reproducible from this repository alone:
the script that produced it is committed in the same push as the number, next to its note (`doors/data/<NOTE>/`, `boxruns/<run>/`, or
`development/sims/`), with its inputs named (repo path, S3 key with sha256, or public accession) and any random seed fixed.
Box jobs run only committed scripts from a fresh pull. Analysis typed into a notebook does not count until it is a committed script
that reruns to the same value. Raw working records are kept privately (S3 `archive/session_code/`) but are not a substitute.
Every push goes through `CANON/checked_push.sh`, which runs `CANON/repro_check.py`: it refuses untracked files and any changed note in
`doors/` with computed numbers but no committed script (backlog items are reported until closed). A plain `git push` skips the gate.

### Record rule and status rule (2026-10-10)
The same gate also refuses a push when the record or a status falls behind:
- **Record:** an outcome note (`doors/*_OUTCOME.md`) new or changed in the push is not named in `development/METHYLPHYS_DEVELOPMENT_LOG.md`
  in the same push; a note marked `**Milestone:**` is not linked from the Advancements table of `Biological_Physics/README.md`, or a link
  there is broken; a day with log entries has no `### <date> · Day summary` once the next day has begun (Pacific time).
- **Status:** every status fact repeated across the repo is held once, in `CANON/status_facts.json` (what is commissioned, the
  whole-blood fraction rule, the atlas cell count, public data only, withdrawn readings, the book's check count). `CANON/status_check.py`
  refuses the push while any living document (READMEs, SOP, STATUS, OM, book, chain code, website) still carries the old wording, a
  required statement is missing, or the printed check count differs from a full `verify_book.py` run. When a status changes, change
  `status_facts.json` first; the checker then lists every place still to update. Dated records (development log, DEV/PROC notes and
  outcomes, job sheets, archive) are history and are not checked.
- **Lessons:** every lesson that cost time, a result or trust gets a row in `Biological_Physics/MethylPhys/LESSONS.md` the day it is learned,
  with its rule and the code that enforces it (or "not enforced by code"). Read it before writing a new box script, reader or test.
- **Number traceability (rule 6, 2026-10-10):** every decimal number (two or more decimals) in a development or outcome note that is new or
  changed in a push must appear, at the note's printed precision, in a committed file of that note's data folder (`doors/data/<NOTE>/`, outputs,
  rows, json or the scoring script, or `development/sims/`). A number typed from a notebook cell, or from memory, refuses the push. DOIs and
  `code spans` are skipped.
- **Results register:** every dated result and milestone is held once, in `CANON/results_register.json` (status development,
  commissioned or withdrawn; its record; its sealed prediction). `CANON/results_to_tex.py` writes the book's development chapter
  (`docs/book/part6/p6_25_development.tex`, Chapter "Development results") and the Advancements table of `Biological_Physics/README.md`
  from it, recomputing every number from the committed record files. The gate refuses a push when either differs from what the register
  generates; when a note says COMMISSIONED and is not registered, or register and record disagree; when a commissioning entry does not list
  in `covers` the development results it closes, or a covered result is still marked development; when a result listed as passed no longer
  meets its sealed rule. Moving a result from development to commissioned is one change in the register; the book and README follow.

### Commissioning order (set 2026-10-03)

Each step: pre-register the check in `doors/` before reading data, run it on v3, record the outcome, then wire the stage in.

1. **NILC (stage 4).** Check: on constructed mixtures of purified cells, NILC recovers the known fractions within a pre-set error, and on
   the same-run replicates its fractions repeat within a pre-set spread.
2. **Atlas deconvolution (stage 3).** Check: as for NILC on constructed mixtures, and agreement with the 8-group composition on whole blood
   where both apply; each additional cell type is read only after its own healthy reference is commissioned ("no cell rather than part of one").
3. **Met-A for each newly commissioned cell type (stage 5),** one cell type at a time, with its own replicate test.
4. **Directional decomposition (stage 10).** Check: on replicates it returns no direction; on a known treated series it returns the
   direction the treatment is known to push.
5. **Sky map and residual maps (stage 11).** Check: healthy replicates give a residual sky consistent with the spatially shuffled null.
6. **Sky statistics (stage 12).** Check: the look-elsewhere correction by simulation holds its stated false-positive rate on healthy arrays.
7. **Stage Q, IAM-A (stage 7)** - commissioned with the base chain (stages 0, 1, 2, 5, 6, 8, 9, 13), before step 1. Check: on the bundled
   single-molecule test data (constructed; no real file is bundled) IAM-A = 1 at the healthy position, another pipeline is refused, the `.pat`
   extractor returns the constructed errors (DEV-BASE-CHAIN-01 e: passed 2026-10-03).
8. **Trace-cell detection (stage 3b).** Check written in DEV-TOOLKIT-ADDED-01; line re-set on the array's own noise and tested 2026-10-04 (DEV-TOOLKIT-ADDED-02, `--dev-trace`).
9. **Foreign-cell detection (stage 3c).** Check written in DEV-TOOLKIT-ADDED-01; line re-set on the array's own noise, beta scale = Stage 1 noob, tested 2026-10-04 (DEV-TOOLKIT-ADDED-02, `--dev-foreign`).
10. **Surface brightness (stage 11b).** Check written in DEV-TOOLKIT-ADDED-01; v3 per-site uncertainty from the array's own fixed sites, tested 2026-10-04 (DEV-TOOLKIT-ADDED-02, `--dev-brightness`).
11. **Difference map (stage 12b).** Check: the same-person check accepts one person and refuses two; same-person replicate differences are below
    differences to other people in >= 95 % of comparisons (DEV-TOOLKIT-ADDED-01: passed 2026-10-03, 348/348; wired).

Order actually run on 2026-10-03: base chain with 7 -> 4 -> 3 -> 5 (B cells) -> 10 (not assessable) -> 11 -> 12 (not run) -> 3b, 3c, 11b (not run) -> 12b.
Development round 2 (2026-10-04, author decisions A-O, test-only mode): intake (F, A, B, L, EPIC v2 refusal) -> detection statistic (E) -> self-tare II (G)
-> composition truth search (H) -> direction (I) -> sky and sky statistics (J) -> 3b, 3c, 11b on own noise (K) -> IAM-A C-score (C) -> new-cell rule on
monocytes (D) -> development flags (N) -> EPIC v2 (M) -> Stage Q on real single-molecule data. Summary: `doors/DEV_ROUND2_REPORT.md`; table: `STATUS.md` (section 8, commissioning record).

### New-cell rule (author decision D, 2026-10-04)

A cell type is read only after it passes three tests on chain v3, recorded in `doors/` before the data are read:
1. **Purified-cell Normal.** Purified healthy arrays of the cell from laboratories other than the floor's, median tare against >= 3 same-series arrays of
   the cell (self excluded): >= 95 % of tared readings in Normal.
2. **Replicate spread.** Repeated arrays of the same DNA or person: within-person SD of tared A <= 0.020.
3. **Identifiability.** (a) Against the cell's floor, >= 99 % of purified healthy arrays of every other blood group read outside Normal; (b) the composition
   stage recovers the cell's fraction within RMSE 0.03 on a held-out mixture truth set.
Applied to monocytes and B cells on 2026-10-04 (DEV-NEWCELL-01): neither meets it. Neutrophils remain the only cell read.

### Development flags (author decision N, 2026-10-04)

`run_sample.py --dev-selftare-ii --dev-direction --dev-trace --dev-foreign --dev-brightness --dev-nilc --dev-atlas-e --dev-percell-b --dev-sky --dev-epic-v2`
(`chain/dev_stages.py`). Each writes `bundle["development"][<stage>]` and a report section, labelled DEVELOPMENT - not commissioned. None changes the
reading, the gauge or the tare (`--dev-selftare-ii` is a no-op alias since 2026-10-04: the adopted Stage T step 1 is wired in and always runs; the flag only copies `tare.selftare_ii` to `development.selftare_ii`) (DEV-FLAGS-01: 63 of 63 readings identical with every flag on; release check E10). `--dev-atlas-e`, `--dev-nilc`,
`--dev-percell-b` and `--dev-sky` need `--atlas-v2 <IAMAtlas_v2.parquet>` (not stored in the repository); `--dev-sky` needs healpy;
`--dev-epic-v2` needs `--sesame-rscript <Rscript>` of an environment with Bioconductor sesame.

### Kept out of the chain, and why

| module | why it cannot work as designed |
|---|---|
| `Runtime Matrices/Directional Panel/bidirectional_decomposition.py` + `directional_panels_v1_0.json` (class-era stage 10) | z against other arrays' mean and SD and a disease sign; replaced by the physics-only direction (`--dev-direction`) |
| `stage_2c_trace_detection.py` + `trace_detection_panel_v1.json` (class-era 3b) | its line was set from other arrays and it reads classes; replaced by `--dev-trace` |
| `toolkit_foreign_detection.py` + `detection_panel_v3.json` (class-era 3c) | its line is a quantile over other arrays, on the class-era beta scale; replaced by `--dev-foreign` |
| `toolkit_surface_brightness.py` (class-era 11b) | reads the class archives' per-CpG brightness; replaced by `--dev-brightness` |
| `nilc_celltype_deconvolver.py` (toolkit NILC, N1) | every held-out truth bar outside (DEV-NILC-01); NILC-e is the one worked on |
| `CPG_Null_Runner` null N7 | its synthetic generator was retired with chain v2 |
| EPIC v2 reading in the chain | no purified neutrophil EPIC v2 arrays exist publicly to set a v2 floor (DEV-EPIC-V2-01); refused at intake |

### Tare: self-tare II (author decision G; adopted 2026-10-04)

Stage T is self-tare II, then the median tare (DEV-SELFTARE-02 reading (iv); adopted by the author on 2026-10-04, `boxruns/run1/JOBS.md` job A; development log
2026-10-04, DEV-PAIRED-01). It met every replicate and other-laboratory bar (DEV-SELFTARE-02): replicate within-person SD 0.0164, 62/63 Normal; other
laboratories 49/49; floor 6/6.
- **What it computes.** Each array is tared on its own fixed sites: per probe design, its low and high anchors L and U (mean β over the low and the high fixed
  sites) and the map β′ = Lr + (β − L)(Ur − Lr)/(U − L) onto the reference arrays' scale (Lr, Ur: the same anchors averaged over the six reference arrays;
  type I 0.0186 / 0.9820, type II 0.0555 / 0.9486). Two points fix an affine map, so nothing is fitted. Met-A is then formed from β′, and the median tare
  (§2 row T, step 2) runs on that A against the same-run references.
- **Fixed sites.** Type I: the noise sites of the same state (DEV-NOISE-01). Type II: EPIC type II probes, not a neutrophil identity site and not a composition
  marker, with every purified GSE110554 group mean ≤ 0.15 (low set) or ≥ 0.85 (high set), group SD ≤ 0.02 and largest difference between group means ≤ 0.03;
  found 50,359 low and 166,379 high. File: `chain/Runtime Matrices/Development/dev_selftare_typeII_EPIC_v1.json` (development).
- **Assumption.** The fixed sites hold the same true state on every array of healthy blood (DEV-SELFTARE-02 step 3, a conjecture).
- **What it replaces.** The median tare alone as the Stage T reading. Self-tare II alone did not carry every other laboratory onto the reference scale
  (GSE247193 5/21 in Normal); the median step after it did (49/49).
- **Fewer than 3 same-run references.** The median step cannot run. DEV-PAIRED-01 (two GSE128733 neutrophil arrays on one slide) read self-tare II alone:
  A 1.1663 / 1.1695 untared, 1.0437 / 1.0426 self-tared; both noise indices above 0.149, so the gauge state was withheld.
- **In the code.** Self-tare II is wired into Stage T (2026-10-04, DEV-SELFTARE-03): `conductor_v3.stage_t_selftare_ii` (`dev_stages.selftare_map`) runs
  before A and M and is recorded under `tare.selftare_ii`; the printed A_rel is the median tare of the self-tared A. The noise index and the noise gate read β
  before self-tare II. `--dev-selftare-ii` is a no-op alias.

History (kept): until 2026-10-04 Stage T was the median tare alone (DEV-TARE-02, 2026-10-02); the noise-corrected tare before it was removed on 2026-10-02
(DEV-TARE-02).

The second route stays written here, as the check on the fixed-site assumption: fully methylated and fully unmethylated control DNA (and a 50 % mix) on every slide
measures the low and high anchors on the slide itself and the channel-gain term the fixed sites cannot see; it needs wet-lab runs.

## 3. Rules the chain enforces

1. **Read line.** In whole blood, A is read at any neutrophil fraction above 0. Each reading carries the shift a 1 % loss of the neutrophil pattern causes at its own fraction, and Stage T prints the detection limit from it: the smallest loss the specimen could show (DEV-LOWFRAC-01; a 1 % loss moves A by 0.0073 at 20 % neutrophils and 0.025 at 60 %).
2. **Whole blood must be tared.** Untared whole-blood A carries a composition and laboratory offset, measured at −0.04 in one lab and +0.09 in another.
   The gauge state is printed only from A_rel (Stage T: self-tare II, then the median tare; §2 row T). Isolated neutrophils are read against their healthy reference and are tared against same-run
   references the same way (array noise, measured on second-lab isolated cells, PROC-NEUT-TEST-01).
3. **Platform match.** The floor, the profiles and the specimen must be on the same platform. Without a frozen floor for the platform, the chain refuses.
   A β vector of 700,000 probes or fewer is refused as 450K or incomplete, so an EPIC v1 array that loses that many probes at detection is refused too.
   A 450K IDAT pair does not reach the platform check: Stage 0 quarantines it (array-type mismatch at 0.1 when EPIC_v1 is declared, else coverage at 0.7b).
4. **Quarantine stops the run.** A Stage 0 QUARANTINE produces no reading.
5. **Nothing from other people.** Nothing in a reading depends on other people's readings, a classifier or a disease label.
6. **Noise gate.** An untared reading on an array whose noise index is above the reference arrays' range (N > 0.149) gets no gauge state;
   A is printed as a number. A tared reading is not withheld by the gate. Nothing is fitted to N. If fewer than 90 % of the noise sites are measured,
   N is not computed and the gauge state is withheld, tared or not, with the counts and the reason (author decision A, 2026-10-04).
7. **Specimen rule.** Intake reads whole blood and isolated / sorted / purified neutrophils. PBMC, other sorted fractions, bone marrow, cell lines, tissue and
   unspecified specimens are refused at intake with a report naming the specimen; nothing is read (author decision L, 2026-10-04).
8. **Identifiers.** The bundle and the evidence ledger carry the sha256 hash of the typed id (and of any path or command argument that carries it); the printed
   report keeps the id the operator typed (author decision B, 2026-10-04). Age and sex are optional and recorded when given (decision F).

## 4. Running it (operator)

**Simulate first (standing rule, 2026-10-09).** Before any new test downloads data or starts the box: build a synthetic version read by
the chain's exact rule; confirm the lever moves the reading as predicted (through the rule's own mapping, not a plain formula), that
instrument error is handled, and that the planned design has power ≥ 0.8. Only then download, and only data that can decide the question
(example: doors/DEV_SYNTH_LEVERS_01.md; the cross-species design, power 0.11, would have been caught).


One specimen:
```
cd Biological_Physics/MethylPhys/chain/MethylPhys_Interface
python run_sample.py --grn S_Grn.idat --red S_Red.idat --engine v3 \
  --specimen "whole blood" --array-type EPIC_v1 --id S001 --out S001.html      # --sex F --age 52 optional, recorded when given
```
Isolated neutrophils: `--specimen "isolated neutrophils"`.
Tare (Stage T step 2, the median tare), once ≥ 3 healthy references of the same specimen type on the same slide (else the same batch) have been read: add `--slide-ref-A 0.951,0.957,0.962`.
That list holds the references' A values before the median tare (`met_a.A`, self-tared). Or `--slide-ref-table refs.csv` (column `A`, optional `id`; the specimen's own id is left out).
Stage T step 1 (self-tare II, adopted 2026-10-04): wired into Stage T on 2026-10-04; it always runs, needs no references and is recorded under `tare.selftare_ii`;
the printed A_rel is the median tare of the self-tared A (§2 row T). `--dev-selftare-ii` is a no-op alias, kept so recorded commands still run.
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

| Round 2, intake (DEV-INTAKE-02) | 1,569 arrays that stopped on a missing age or sex in round 1 re-run: 0 crashes; 955 refused naming the specimen; 613 blood specimens read; typed id in 0 bundles and 0 ledgers, in 1,837/1,837 report titles |
| Round 2, purified healthy neutrophils, enlarged set (DEV-INTAKE-02) | tared A_rel: floor 6/6, other laboratories 56/68 Normal (round 1: 42/49) |
| Round 2, detection statistic (DEV-DETECTION-01) | 4,996 EPIC v1 and 450K arrays: poobah better in 31 strata, Gaussian test in 0; poobah kept |
| Round 2, self-tare on type II fixed sites then median tare (DEV-SELFTARE-02, flag) | replicate within-person SD 0.0164, 62/63 Normal; other laboratories 49/49; floor 6/6 |
| Purified neutrophils from a new laboratory, GSE128733, 2 EPIC arrays on one slide (DEV-PAIRED-01, 2026-10-04) | A untared 1.1663 / 1.1695; self-tare II 1.0437 / 1.0426; median step not run (2 arrays, ≥ 3 needed); noise index 0.169 / 0.182, gauge state withheld |
| Round 2, known loss of methylation (DEV-DIRECTION-02, flag) | decitabine and NTX-301 treated arrays 12/12 toward disorder; replicates 56/63 no direction |
| Round 2, IAM-A on real single-molecule files (DEV-IAMA-REAL-01) | three Loyfer granulocyte files end to end: whole files 1.0394, 1.0632, 1.0344 (2/3 Normal) |

Round 2 rows are development (DEVELOPMENT - not commissioned); each check and its full outcome is in the named `doors/` note.
Not yet shown: any disease; a physical control DNA; an EPIC v2 floor. The C-score band is not set.

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
| M Met-A | whole blood read at any f_NEU > 0, with its detection limit; ≥ 5400 of 6000 identity sites measured (both cases) | `conductor_v3.py`, `stage_m_met_a.py` (`SITE_COVERAGE_MIN` 0.9) | [0.20 in v3 §3 as 20 %; 5400 in full §5] |
| M Met-A | identity-site rule (also kept in book; EPIC v1 neutrophils, 6 physical arrays): across-array SD of β ≤ 0.05; mean β 0.75–0.95 (methylated channel) or 0.05–0.25 (unmethylated channel); ≤ 3,000 per channel by smallest SD; result 6000 sites, 3000 per channel | `chain_tests/freeze_v13.py`, `stage_m_met_a.py`; `metA_floors_v1_3.json` `n_sites` | **[new]** (rule; the count is in full §3). The rule also stays in the book, Ch. "Identity sites" |
| M Met-A | ceiling flag `past_entropy_ceiling` = (mean β at sites with μ_NEU > 0.5) < 0.5; report: "read beta, not A" | `conductor_v3.py:127-133` (`_ceiling`) | [in full §1, §6] |
| MC C-score | blocks of 10 sites; ≥ 10 blocks; healthy baseline 1.0103 | `neutrophil_reference_v1_2.json` | [in full §3, §5] |
| Q IAM-A | pipeline `loyfer_pat_v1` only; P = 1.1492 (whole files); ε₀ = 0.032 (healthy reference); ≥ 100,000 opportunities; halves A/B with > 50,000 each; molecule qualifies with ≥ 6 calls and ≥ 80 % methylated | `chain/stage_q_iam_a.py`, `chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json` | [in full §3, §5; Stage Q row added to v3 §2 on 2026-10-03] |

**Stage Q row (added to v3 §2 on 2026-10-03):**

| stage | what it does | code | frozen input |
|---|---|---|---|
| Q IAM-A (sequencing) | isolated copy error ε on qualifying molecules (≥ 6 calls, ≥ 80 % methylated); IAM-A = H(ε) ÷ (P × H(ε₀)); ≥ 100,000 opportunities; refuses any pipeline but the one P was measured on. IAM-A C-score (development, decision C): blocks of 1,000 sites in genomic order, C = Σ(k_b − ε o_b)² / Σ ε(1 − ε) o_b; independent errors give 1 (derived); one C per A and per half; band not set | `chain/stage_q_iam_a.py` (called by `run_sample.py:351-361`) | `chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json` (`eps0` 0.032, `cells.neutrophils.P` 1.099, `pipeline` `loyfer_pat_v1`) |

**Add to v3 §2, frozen-input list:** `noise_gate_EPIC_v1.json` (above); `iama_positions_v1.json` (in §2 since 2026-10-03); `metA_floors_v1_3.json` key `duplicates_removed` (the six GSE110554 / GSE167998 pairs). **[in full §3 except noise_gate]**

**Add to v3 §2, noise sites:** rule = every purified blood group mean β ≤ 0.03 (40,882 sites) or ≥ 0.97 (7,646 sites), every group SD ≤ 0.02, no
neutrophil identity site; 48,528 sites; N computed when ≥ 90 % are measured. **[rule and count in full §3; the 40,882 / 7,646 split is new]**

**Add to v3 §2, development floors not read:** `chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_2_ALLCELLS_development.json` (other cell types; development only, not read by v3). **[new]**

---

## §2 Add to v3 §4 (operator) — names the book used

- Entry point `chain/MethylPhys_Interface/run_sample.py --engine v3`; the stages after calibration are in `chain/conductor_v3.py`; report `chain/MethylPhys_Interface/report_v3.py`. [in v3 §2/§4]
- Covariates: `--covariate KEY=VALUE`; kept in the custody record, bundle and ledger; no stage reads them; a free-text covariate appears in the report's bundle block. [flag in full §4.1; the "appears in the bundle block" sentence is **new**]
- Report state strings. Met-A: `untared: read A_rel (Stage T)` (whole blood), `untared (healthy-reference state: <Normal|above Normal|below Normal>): read A_rel (Stage T)` (isolated, drawn against the healthy reference), `tared: read A_rel (Stage T)`, `withheld: noise index <N> > 0.149 and no same-run tare; A printed as a number only`. Tare: `Normal`, `above Normal`, `below Normal`, or the reason `untared: <n> same-run reference arrays (>= 3 required)`. Untared whole blood: no gauge position. [in full §5, §6]
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
| `STATUS.md` (was PLAN.md; incl. item 22) | order of work; E-MTAB-7309: 738 of 1,056 below 0.93, median call rate 0.894; no canine purified reference found **[new]** |
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
