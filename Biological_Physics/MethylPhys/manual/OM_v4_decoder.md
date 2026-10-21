# The Chain Decoder

The chain the one command loads: **22 files, 257 functions**; 50 of them have their entry here.

## How to read the Decoder

The chain is the code that turns one sample into one report. This part of the manual takes it apart in the order a real run
executes it: a traced run of the one command on a real healthy whole-blood array (GSE235717, GSM7508429; `chain/trace/`), and a
second traced run on a small constructed sequencing file for IAM-A. Every function the one command can load has its own entry, in
the file where it lives, in the order the run first enters that file.

Each stage opens with what it is for, the physics in words and then the formula, what it decides, every gate with its measured value,
its limit, where the limit came from and whether it is provisional, the files it reads, and how to break it. Each function entry
then says what the function does and what it decides. Under every entry the manual prints, read from the code itself and checked
at every push: the file and line, the arguments, the module constants it reads, the data files it names, and how many times each
traced run called it.

Every stage carries one of two labels, so the physics can be told apart from the bookkeeping:

- **Standard method**: a technique used elsewhere in the field (array calibration, deconvolution of a cell mix, normalization
  against references run the same way). IAM uses it as a tool. It is not evidence for IAM's law, and a reviewer can check it
  against the literature.
- **From IAM's physics**: a quantity that follows from the cost of holding a mark against thermal noise. The entry gives the
  derivation and the test that could have failed.

**Names.** The gauge has three marks. The **floor** is the least entropy a held mark can carry, derived from thermal kicks against
about one ATP per site (M = 20.94). The **ceiling** (saturation) is every identity site fully scrambled. The **middle**, healthy = 1,
is the healthy reference: Met-A's is measured once on purified healthy cells of the same type, and IAM-A's is computed from the
sample's own loose share. Some file names in the chain predate these names: `Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`
holds Met-A's healthy middle reference, not the floor. The Decoder uses the canon's names and says which file holds what.

**Built from what, tested on what.** A reviewer's first question about any reference is whether it was tested on the data it was
built from. Every reference entry names both.

## 1. The one command: `run_sample.py`

**What it is for.** One sample in, one report out. A researcher runs one command on an IDAT pair (an Illumina EPIC v1 array), a
calibrated beta table, or sequencing data (a wgbstools `.pat` file or an aligned bisulfite BAM). `run_sample.py` decides which stages
run, in what order, and stops the run the moment a stage refuses the sample. It computes no reading itself.

**Label: standard method** (orchestration). Nothing here is physics.

**The order it runs.**

1. Parse the command and refuse impossible combinations (no input; two sequencing inputs; a BAM without its reference FASTA; a site
   table without its pipeline; both kinds of reference list).
2. **Specimen check** (`stage_0_intake.specimen_refusal`). The chain reads whole blood, buffy coat (read on the whole-blood path,
   labeled preliminary) and isolated neutrophils. Any other specimen is refused before a byte is read.
3. **Stage 0, intake**, for an IDAT pair: 0.1 arrival, 0.2 manifest, 0.3 integrity hash. A quarantine at any of these stops the run
   (exit code 2) before calibration. An EPIC v2 header is refused here: the chain's references are EPIC v1, and Stage 1's calibrator
   does not read EPIC v2.
4. **The QC hand-off** (`stage_0_1_qc_handoff.decode_qc_inputs`) reads the array's control probes, probe intensities, negative-control
   background, bead counts and sex probes straight from the IDAT files. If the array manifest cannot be loaded, the machine is at
   fault, not the array: exit code 3, and the array is not judged. If the decoder fails on the files, the array is quarantined as a
   corrupt IDAT.
5. **Stage 0, the QC gates**: 0.4 control probes, 0.5 detection p-values, 0.6 bead counts, 0.7 call rate, 0.8 sex, 0.7b platform
   coverage, then 0.9 the decision gate. A quarantine prints every reason in five parts (measured, limit, where the limit comes from,
   what it means, what to do) and stops the run.
6. **Stage 1, calibration** (`stage_1_idat_calibration.calibrate_idat_to_beta`): the IDAT pair becomes one beta value per site, with
   the probes at background removed. Stage 1's own quality numbers are recorded beside Stage 0's, never over them, and never gate.
7. **Stage 0.7b again**, now with the calibrated values: `run_sample.py` counts the share of Met-A's 6,000 neutrophil identity sites
   the array measured and hands it to 0.7b. On a 450K array 0.7b quarantines below 80 %; on EPIC, the only platform the chain reads,
   it records `NA_EPIC` and passes. Met-A itself withholds its reading when fewer than 90 % of its sites were measured (Stage M).
8. **The conductor** (`conductor_v3.run_neutrophil`): the platform check, Stage A (the cell mix), Stage M (Met-A), Stage MC (the
   C-Score) and Stage T (the same-run tare).
9. **Chain v4 additions** when the array was read: Stage D21 (`stage_v4.stage_d21`, the IAM-Deconvolver's cell shares against
   IAM-Atlas v2.1) and the three skies (the C-Score's departures on the sphere, the healthy neutrophil of IAM-Atlas v2.1 at the same
   sites, and the overlay).
10. **IAM-A** when sequencing data is given: Stage Q0 intake on a `.pat` file, then Stage Q2 (`stage_v4.count_sequencing` and
    `stage_v4.stage_q2`), the held error and the loose share counted in one pass and read against the sample's own reference.
    The retired position reading (Stage Q) runs only with `--site-table` or `--dev-iama-position`, labeled development.
11. **Stage 12b**, the difference between two draws of the same person, when `--prior-betas` is given; refused unless both draws carry
    the same identifier hash, array type and pipeline.
12. **Development readings**, only when a `--dev-*` flag asks for them, written under `bundle["development"]` and never part of the
    reading.
13. **The report** (`report_v4.build`): one HTML page with a researcher view and a plain view, and two PDFs. Then the identifier is
    replaced by its hash everywhere in the bundle and the ledger, the bundle is written as JSON, and one row is appended to the
    evidence ledger.

**Exit codes.** 0 a report was written (it may still withhold a reading, and says why); 2 the sample was quarantined at intake and
nothing was scored; 3 the machine is missing a file it needs (the array manifest) and the array was not judged.

**What it reads.** The command line; the IDAT pair or beta table or sequencing file; `--ref-table` (the same-run healthy references:
column A for Met-A, column C for the C-Score); `Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json` for the 0.7b coverage count.
Everything else is read by the stages.

**How to break it.** Give two sequencing inputs, a BAM without `--ref-fasta`, a specimen outside the list, an EPIC v2 array, a
truncated IDAT: each must stop with a named reason, never a traceback. Release checks V1 to V10 run the one command end to end.

### `run_sample.py::_bio_root`

Finds the `Biological_Physics` folder by walking up from this file until it finds a folder holding `MethylPhys`, rather than by
counting levels. Written after a 2026-09-22 move changed every script's depth by one and a counted path silently stopped finding
the record folder. Decides nothing about a sample.

<sub>`MethylPhys_Interface/run_sample.py` line 35 (16 lines) · arguments: start · calls: array run 1, bam run 1</sub>

### `run_sample.py::_assign_run_id`

Gives the run its identifier, `RUN-YYYYMMDD-NN`: the UTC date and the next number for that day, counted from the ledger the run is
about to append to. A run makes no claim and passes no bar; it is one execution on one specimen, and the identifier lets a reading be
found again.

<sub>`MethylPhys_Interface/run_sample.py` line 55 (19 lines) · arguments: ledger_path · calls: array run 1, bam run 1</sub>

### `run_sample.py::_covariates`

Collects the phenotype and covariates for the run: a JSON object from `--covariates`, then each `--covariate KEY=VALUE` on top. They
are kept in the bundle and the custody record and are never printed in the report's prose. No stage reads them.

<sub>`MethylPhys_Interface/run_sample.py` line 76 (16 lines) · arguments: a · calls: array run 1, bam run 1</sub>

### `run_sample.py::_sha12`

The first 12 hexadecimal characters of a file's SHA-256, used to fingerprint each frozen input in `_versions`. Returns nothing if the
file cannot be read.

<sub>`MethylPhys_Interface/run_sample.py` line 94 (10 lines) · arguments: path · calls: array run 18, bam run 18</sub>

### `run_sample.py::_versions`

Records everything that can change a reading: the UTC time, the Python and methylprep versions, the chain's git commit and whether
the working copy had uncommitted changes, and a short hash and size of each frozen matrix and stage file. Two runs months apart can
be pooled only if these match.

<sub>`MethylPhys_Interface/run_sample.py` line 106 (37 lines) · arguments: chain_dir · calls: array run 1, bam run 1 · files named: `Runtime Matrices/IAM_A_Positions/hg19_cpg_chrom_ranges.json`, `Runtime Matrices/IAM_A_Positions/iama_positions_v2.json`, `Runtime Matrices/Intake/intake_thresholds_v1.json`, `Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json`, `Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`, `Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`, `Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_2.json`, `Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json`, `Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json`</sub>

### `run_sample.py::_hash_id`

Replaces a typed identifier with the first 32 characters of its SHA-256. An identifier that already looks like a hash (16 or more
hexadecimal characters) is kept. The engine never stores a cleartext identifier.

<sub>`MethylPhys_Interface/run_sample.py` line 145 (5 lines) · arguments: x · calls: array run 1, bam run 1</sub>

### `run_sample.py::_redact`

Walks the whole bundle (every string, dictionary key and list) and replaces each whole-token occurrence of the typed identifier with
its hash, so neither the bundle nor the ledger carries the typed name. The printed report keeps the typed name for the operator.

<sub>`MethylPhys_Interface/run_sample.py` line 152 (11 lines) · arguments: obj, typed, hid · calls: array run 4, bam run 4</sub>

### `run_sample.py::_redact.f`

The recursive step of `_redact`: a string gets the replacement, a dictionary and a list are walked, anything else is returned as is.

<sub>`MethylPhys_Interface/run_sample.py` line 157 (5 lines) · arguments: v · calls: array run 865, bam run 359</sub>

### `run_sample.py::_dev_run`

Runs the development readings named by `--dev-*` flags (direction, trace cell, foreign cell, brightness interval, NILC-e, atlas e,
B-cell reading, sky with the block-shuffle null, EPIC v2 through SeSAMe). Each writes its own record under `bundle["development"]`,
labeled DEVELOPMENT, and an error in one is caught and recorded. None of them changes the reading, the gauge or the tare.

<sub>`MethylPhys_Interface/run_sample.py` line 165 (40 lines) · arguments: a, o, beta, intake · calls: array run 1, bam run 1</sub>

### `run_sample.py::main`

The orchestration described above, from parsing the command to writing the ledger row. Decides which stages run, stops the run with
exit code 2 or 3 at the stops listed, carries the same-run references (columns A and C) into the conductor, adds the v4 stages, and
hands everything to the report.

<sub>`MethylPhys_Interface/run_sample.py` line 207 (420 lines) · arguments: none · calls: array run 1, bam run 1 · constants read: `BIO`, `ENG` · files named: `MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`, `_bundle.json`, `a wgbstools .pat / .pat.gz`</sub>

### `run_sample.py::main._stopped`

True when an intake record's status begins with QUARANTINE. Intake checks it after every step, so a quarantine ends intake at its
cause. Without it a later step overwrote the status, and the verdict named a downstream symptom (seen 2026-09-23: an array-type
mismatch was reported as a detection failure).

<sub>`MethylPhys_Interface/run_sample.py` line 303 (5 lines) · arguments: r · calls: array run 3</sub>

### `run_sample.py::main._ctx`

Builds the comparison record Stage 12b needs from a draw: the identifier hash, the array type and the calibration pipeline.

<sub>`MethylPhys_Interface/run_sample.py` line 573 (2 lines) · arguments: b, it, pipe · calls: not called on the traced runs</sub>

## 2. Stage 0, intake: `stage_0_intake.py`

**What it is for.** Deciding whether this pair of files is a usable array from the specimen it claims to be, before anything is
calibrated or read. A quarantine here means nothing about the biology was looked at.

**Label: standard method.** These are the array maker's and the field's quality checks (control probes, detection against background,
bead counts, call rate, a sex check from the X and Y signals), applied to one array on its own. None of the limits comes from IAM's
physics. Every limit lives in a named constant or in `Runtime Matrices/Intake/intake_thresholds_v1.json`, never as a bare number
in a gate.

**The specimen check (before Stage 0).** `specimen_refusal` accepts whole blood (and the words blood, peripheral blood, venous blood),
isolated, sorted or purified neutrophils, buffy coat (read on the whole-blood path, labeled as its own specimen, preliminary), and the
constructed DNA mixtures used as test material. PBMC, other sorted cell fractions, bone marrow, cell lines, an unstated specimen and
anything else are refused with the reason, and nothing is read.

**The steps, in the order the run calls them.**

| Step | What it measures | Outcome | Limit and source |
|---|---|---|---|
| 0.1 arrival | the manifest's required fields (Sentrix ID, array type, identifier, intake date, specimen); both channels present; each IDAT at least 1 MB; the array type in the IDAT header against the declared one | quarantine on a missing field, a missing channel, a truncated file, or a 450K/EPIC family swap; a note on an EPIC v1/v2 subtype difference; a soft warning on the same Sentrix ID taken in within 24 h | fixed rules (SOP 11): `MIN_PLAUSIBLE_IDAT_BYTES` 1,000,000; header address count below 800,000 is 450K, below 1,080,000 EPIC v1, above EPIC v2 |
| 0.2 manifest | builds the immutable per-sample record; the identifier must already be a hash | quarantine on a cleartext identifier (spaces, an @, under 16 characters) or a missing core field | fixed rules (SOP 12) |
| 0.3 integrity | SHA-256 of both files against earlier intakes of the same Sentrix ID in this intake log | identical bytes again: re-transmission, the run stops; same Sentrix ID with different bytes: a re-run of a fresh array, read | fixed rule (SOP 13) |
| 0.4 control probes | bisulfite conversion, hybridization (high against low concentration) and extension (two colors) controls, from the array's own control addresses | a failed hybridization or extension stops the array; the bisulfite number is printed and never stops an array on its own | `HYB_HIGH_LOW_MIN_RATIO` 2.0 and `EXTENSION_BALANCE_RANGE` 0.2 to 5.0, both **provisional**; bisulfite at least 0.95, **provisional** (as written it refuses 731 of 731 healthy arrays) |
| 0.5 detection | per probe, p = 1 − Φ((I − μ_bg)/σ_bg) against this array's negative-control background; the share of probes with p ≤ 0.01 | above 0.99 passes; 0.93 to 0.99 is read with a penalty flag; below 0.93 stops | `intake_thresholds_v1.json` detection |
| 0.6 bead counts | the share of probes resting on at least 3 beads | below 0.995 a penalty flag; never stops | `BEAD_COUNT_MIN` 3; `intake_thresholds_v1.json` bead |
| 0.7 call rate | the share of probes passing both detection and bead count | 0.98 or more passes; 0.93 to 0.98 penalty; below 0.93 stops | `intake_thresholds_v1.json` call_rate; the 0.93 line was set by the author on 2026-09-27 from 48 arrays, 12 from each of four commissioning laboratories |
| 0.7b coverage | the platform tag; on a 450K array, the share of the reference sites measured | on 450K, below 0.80 stops; on EPIC, `NA_EPIC` (passes) | `HM450_COVERAGE_MIN` 0.80 |
| 0.8 sex | median log2 intensity of the Y probes minus the X probes | Y − X below −2 is female, otherwise male; a mismatch with a declared sex stops the array; with no declared sex the array's sex is recorded | `SEX_GETSEX_CUTOFF` −2.0, the rule of the minfi package's getSex (SOP 18) |
| 0.9 decision gate | every step's outcome | any hard failure: QUARANTINE; any penalty: PROCEED_WITH_PENALTY; else PROCEED | a detection or call rate that could not be measured is a hard failure, never a pass (added 2026-09-27 after a low-signal laboratory had walked through for a week) |

**Every stop explains itself.** `explain_stage0` attaches, for every check that stopped, penalized or deferred the array: what was
measured on this array, the limit, where the limit comes from and whether it is provisional, what it means and what to do.
`run_sample.py` prints these on a quarantine. Release check V9 builds each kind of stop and fails if any one is not explained.

**Built from what, tested on what.** The detection and call-rate lines were measured on 48 arrays, 12 from each of the four
laboratories of Met-A's commissioning panel (`intake_thresholds_v1.json`, `_meta`): Uppsala 0.985, Karolinska 0.975, UCLA 0.953,
Munich 0.878. The detected fraction, not the signal-to-background ratio, separated the low-signal laboratory. The control-probe limits
are provisional and have not been set from healthy arrays; setting them is a to-do step. Until then a provisional limit is printed
with the word PROVISIONAL wherever it is used.

**How to break it.** Hand it a single channel, a 900 KB IDAT, a 450K pair declared as EPIC, the same pair twice under one Sentrix ID,
a declared sex that disagrees with the array, or a low-signal array: each must quarantine with its reason printed in five parts. The
file's own self-test (`python3 stage_0_intake.py`) runs every gate on constructed inputs.

### `stage_0_intake.py::_intake_thresholds`

Loads `intake_thresholds_v1.json` from the chain's `Runtime Matrices/Intake` folder (never from a RETIRED folder) when the module is
imported. Stage 0 refuses to start without it, so no gate can run on a number typed into the code.

<sub>`stage_0_intake.py` line 37 (8 lines) · arguments: none · calls: array run 1, bam run 1 · files named: `intake_thresholds_v1.json`</sub>

### `stage_0_intake.py::normalise_specimen`

Lowercases the specimen text, turns underscores into spaces and collapses repeated spaces, so `Whole_Blood` and `whole blood` are the
same specimen.

<sub>`stage_0_intake.py` line 79 (2 lines) · arguments: specimen · calls: array run 5</sub>

### `stage_0_intake.py::specimen_refusal`

Returns nothing when the specimen is accepted, otherwise the plain refusal: which specimen, why the chain has no reference for it,
and what is accepted. Called by `run_sample.py` before any file is read.

<sub>`stage_0_intake.py` line 83 (8 lines) · arguments: specimen · calls: array run 2 · constants read: `ACCEPTED_SPECIMENS`, `REFUSED_SPECIMEN_REASONS`</sub>

### `stage_0_intake.py::_infer_platform`

Turns the IDAT header's address count into a platform: below 800,000 is 450K, below 1,080,000 EPIC v1, otherwise EPIC v2.

<sub>`stage_0_intake.py` line 103 (8 lines) · arguments: nsnps · calls: array run 2</sub>

### `stage_0_intake.py::read_idat_nsnps`

Reads only the start of the green IDAT: the four-byte `IDAT` mark, the field table, and field 1000 (the number of addresses). It never
decodes intensities. Returns the count and a status (OK, NOT_IDAT, NO_NSNPS_FIELD, FILE_NOT_FOUND or a read error).

<sub>`stage_0_intake.py` line 115 (24 lines) · arguments: path · calls: array run 2</sub>

### `stage_0_intake.py::step_0_1_idat_arrival`

Step 0.1: checks the manifest's required fields and array type, that both channels exist, that each file is at least 1 MB, and that the
header's platform family agrees with the declared one; warns on a repeat intake within 24 hours. Stamps a run identifier and the
arrival time, and logs the result.

<sub>`stage_0_intake.py` line 143 (89 lines) · arguments: manifest_entry, grn_path, red_path, intake_log_path · calls: array run 1 · constants read: `MIN_PLAUSIBLE_IDAT_BYTES`, `REQUIRED_MANIFEST_FIELDS`, `VALID_ARRAY_TYPES`</sub>

### `stage_0_intake.py::_log_intake`

Appends one row per intake step to the intake log, when one is given: time, run identifier, Sentrix ID, declared and detected array
type, specimen, declared sex and age, status and flags.

<sub>`stage_0_intake.py` line 236 (10 lines) · arguments: intake_log_path, result · calls: array run 1</sub>

### `stage_0_intake.py::_is_duplicate_within_24h`

True when the intake log already holds the same Sentrix ID from the last 24 hours. Only a warning; the array still advances.

<sub>`stage_0_intake.py` line 248 (21 lines) · arguments: intake_log_path, sentrix_id · calls: array run 1</sub>

### `stage_0_intake.py::_is_null_answer`

For storing questionnaire answers: a blank or "prefer not to say" becomes no answer; "No" is kept as a real answer.

<sub>`stage_0_intake.py` line 328 (3 lines) · arguments: v · calls: not called on the traced runs · constants read: `_NULL_FOR_STORAGE`</sub>

### `stage_0_intake.py::_answer_present`

For the questionnaire's context modes: true only for an affirmative answer (not blank, not "prefer not to say", not No, none or not
applicable).

<sub>`stage_0_intake.py` line 333 (5 lines) · arguments: v · calls: not called on the traced runs · constants read: `_NEGATIVE`, `_NULL_FOR_STORAGE`</sub>

### `stage_0_intake.py::_age_from_dob`

Age in decimal years from a date of birth and the intake date, rounded to 0.1.

<sub>`stage_0_intake.py` line 340 (8 lines) · arguments: dob, on_date · calls: not called on the traced runs</sub>

### `stage_0_intake.py::parse_questionnaire`

Maps a digital questionnaire's answers to the stored covariate record: each field, the age from the date of birth, a smoking bin, and
a list of context modes. The record is stored with the manifest. No stage of the current chain reads these covariates or modes; the
stage names in its field table belong to an earlier engine.

<sub>`stage_0_intake.py` line 350 (18 lines) · arguments: answers, intake_date · calls: not called on the traced runs · constants read: `COVARIATE_FIELDS`, `_CONTEXT_MODE_TRIGGERS`</sub>

### `stage_0_intake.py::_looks_like_cleartext_pii`

True when an identifier looks like a name rather than a hash: empty, containing a space or an @, shorter than 16 characters once
dashes and underscores are removed, or not purely letters and digits.

<sub>`stage_0_intake.py` line 370 (9 lines) · arguments: patient_id · calls: array run 1</sub>

### `stage_0_intake.py::_write_manifest`

Writes the per-sample manifest as `patient_manifest_<run id>.json` in the manifest folder, when one is given.

<sub>`stage_0_intake.py` line 381 (8 lines) · arguments: manifest_dir, record · calls: array run 1 · files named: `Write the immutable per-sample patient_manifest.json`</sub>

### `stage_0_intake.py::step_0_2_manifest_creation`

Step 0.2: builds the immutable manifest from the 0.1 record (and questionnaire answers, when given), quarantines a cleartext
identifier or a missing core field, and writes the file.

<sub>`stage_0_intake.py` line 391 (40 lines) · arguments: step_0_1_result, questionnaire_answers, manifest_dir, intake_date · calls: array run 1</sub>

### `stage_0_intake.py::_sha256_file`

The SHA-256 of a whole file, read in 1 MB pieces.

<sub>`stage_0_intake.py` line 439 (6 lines) · arguments: path · calls: array run 2</sub>

### `stage_0_intake.py::_prior_hashes_for_sentrix`

The pairs of file hashes earlier intakes recorded for the same Sentrix ID in the integrity log.

<sub>`stage_0_intake.py` line 447 (14 lines) · arguments: integrity_log_path, sentrix_id · calls: array run 1</sub>

### `stage_0_intake.py::_log_integrity`

Appends one row to the integrity log (also used for the decision gate's verdict row).

<sub>`stage_0_intake.py` line 463 (6 lines) · arguments: integrity_log_path, row · calls: array run 1</sub>

### `stage_0_intake.py::step_0_3_integrity_hash`

Step 0.3: hashes both files and compares them with earlier intakes of the same Sentrix ID. Identical bytes again is a re-transmission
and stops the run; the same Sentrix ID with different bytes is flagged as a fresh re-run and advances.

<sub>`stage_0_intake.py` line 471 (37 lines) · arguments: record, grn_path, red_path, integrity_log_path · calls: array run 1</sub>

### `stage_0_intake.py::extract_control_probes`

Gets the control-probe medians by class from the IDAT pair through the QC hand-off (`stage_0_1_qc_handoff.decode_qc_inputs`), the
same decoder Stage 1 uses. The array's own controls only.

<sub>`stage_0_intake.py` line 529 (7 lines) · arguments: grn_path, red_path, array_type · calls: not called on the traced runs</sub>

### `stage_0_intake.py::validate_control_probes`

Step 0.4's arithmetic. Bisulfite: the type I control median over the sum of the type I and type II medians, against 0.95. Hybridization:
the high-concentration median over the low-concentration median, at least 2. Extension: methylated over unmethylated, between 0.2 and
5. When the bisulfite number is the only failure, the verdict is PROVISIONAL and the array advances, because that limit has not been
set from healthy arrays. Records the raw medians the gates tested.

<sub>`stage_0_intake.py` line 538 (41 lines) · arguments: control_summary · calls: array run 2 · constants read: `BS_CONVERSION_MIN`, `BS_THRESHOLD_CALIBRATED`, `EXTENSION_BALANCE_RANGE`, `HYB_HIGH_LOW_MIN_RATIO`</sub>

### `stage_0_intake.py::step_0_4_control_probe_validation`

Step 0.4: runs `validate_control_probes` on the hand-off's control summary and stamps the verdict and its numbers. With no decoder
available it marks the check deferred rather than inventing a pass.

<sub>`stage_0_intake.py` line 581 (20 lines) · arguments: record, grn_path, red_path, control_summary · calls: array run 1</sub>

### `stage_0_intake.py::_normal_sf`

The normal survival function, 1 − Φ(z), from scipy when it is installed, otherwise from the complementary error function.

<sub>`stage_0_intake.py` line 620 (9 lines) · arguments: z · calls: array run 2</sub>

### `stage_0_intake.py::compute_detection_p`

Per probe, the detection p-value 1 − Φ((I − μ_bg)/σ_bg), with μ_bg and σ_bg this array's negative-control background.

<sub>`stage_0_intake.py` line 631 (6 lines) · arguments: probe_intensities, mu_bg, sigma_bg · calls: array run 2</sub>

### `stage_0_intake.py::validate_detection_p`

The share of probes with detection p ≤ 0.01, graded against the thresholds file: above 0.99 PASS, from 0.93 BORDERLINE, below FAIL.

<sub>`stage_0_intake.py` line 639 (15 lines) · arguments: detection_p_array · calls: array run 2 · constants read: `DETECTION_BORDERLINE_FRACTION`, `DETECTION_PASS_FRACTION`, `DETECTION_P_THRESHOLD` · files named: `intake_thresholds_v1.json`</sub>

### `stage_0_intake.py::step_0_5_detection_pvalue_qc`

Step 0.5: computes and grades detection. Without intensities the check is deferred, and a deferred detection never advances.

<sub>`stage_0_intake.py` line 656 (22 lines) · arguments: record, probe_intensities, neg_control_stats · calls: array run 1</sub>

### `stage_0_intake.py::validate_bead_count`

The share of probes with at least 3 beads: 0.995 or more PASS, otherwise a warning. Never stops an array.

<sub>`stage_0_intake.py` line 688 (8 lines) · arguments: bead_counts, min_beads · calls: array run 1 · constants read: `BEAD_COUNT_MIN`, `BEAD_PASS_FRACTION`</sub>

### `stage_0_intake.py::step_0_6_bead_count_qc`

Step 0.6: grades the bead counts and flags a low count. Always advances.

<sub>`stage_0_intake.py` line 698 (14 lines) · arguments: record, bead_counts · calls: array run 1</sub>

### `stage_0_intake.py::validate_call_rate`

The share of probes passing both detection and bead count, graded: 0.98 or more PASS, from 0.93 BORDERLINE, below FAIL.

<sub>`stage_0_intake.py` line 722 (9 lines) · arguments: n_passing_both, n_total · calls: array run 2 · constants read: `CALL_RATE_BORDERLINE`, `CALL_RATE_PASS`</sub>

### `stage_0_intake.py::step_0_7_call_rate`

Step 0.7: combines the detection and bead masks probe by probe and grades the call rate. Without masks the check is deferred, and a
deferred call rate never advances.

<sub>`stage_0_intake.py` line 733 (22 lines) · arguments: record, detection_pass_mask, bead_pass_mask · calls: array run 1</sub>

### `stage_0_intake.py::step_0_7b_platform_coverage`

Step 0.7b: stamps the platform tag. On a 450K array it stops below 80 % of the reference sites; on EPIC it records `NA_EPIC` and passes.

<sub>`stage_0_intake.py` line 764 (23 lines) · arguments: record, reference_cpg_coverage · calls: array run 2 · constants read: `HM450_COVERAGE_MIN`</sub>

### `stage_0_intake.py::predict_sex`

Female when the median log2 Y intensity minus the median log2 X intensity is below −2, otherwise male (minfi's getSex rule).

<sub>`stage_0_intake.py` line 796 (2 lines) · arguments: log2_x_median, log2_y_median, cutoff · calls: array run 1 · constants read: `SEX_GETSEX_CUTOFF`</sub>

### `stage_0_intake.py::validate_sex`

Compares the predicted sex with a declared one. No declared sex: the array's sex is recorded and nothing is compared. A mismatch stops
the array.

<sub>`stage_0_intake.py` line 800 (7 lines) · arguments: log2_x_median, log2_y_median, declared_sex · calls: array run 1</sub>

### `stage_0_intake.py::step_0_8_sex_check`

Step 0.8: runs the sex check on the hand-off's X and Y medians and stamps the result.

<sub>`stage_0_intake.py` line 809 (15 lines) · arguments: record, sex_intensities · calls: array run 1</sub>

### `stage_0_intake.py::_fmt`

Formats a measured number for an explanation, or says "not measured".

<sub>`stage_0_intake.py` line 838 (2 lines) · arguments: x, nd · calls: array run 2</sub>

### `stage_0_intake.py::explain_stage0`

Builds the five-part explanation of every check that stopped, penalized or deferred the array (measured on this array, the limit, the
limit's source and whether it is provisional, what it means, what to do). Changes no decision.

<sub>`stage_0_intake.py` line 840 (54 lines) · arguments: record, hard_fail, borderline, deferred · calls: array run 2 · constants read: `BEAD_COUNT_MIN`, `BEAD_PASS_FRACTION`, `BS_CONVERSION_MIN`, `BS_THRESHOLD_CALIBRATED`, `CALL_RATE_BORDERLINE`, `CALL_RATE_PASS`, `DETECTION_BORDERLINE_FRACTION`, `DETECTION_PASS_FRACTION`, `EXTENSION_BALANCE_RANGE`, `HYB_HIGH_LOW_MIN_RATIO`, `SEX_GETSEX_CUTOFF`, `_TH_SRC`</sub>

### `stage_0_intake.py::explain_stage0.add`

Adds one explanation to the list.

<sub>`stage_0_intake.py` line 842 (2 lines) · arguments: check, outcome, measured, limit, source, provisional, means, todo · calls: array run 2</sub>

### `stage_0_intake.py::step_0_9_decision_gate`

Step 0.9: sorts every step's outcome into hard failures, penalties and deferrals and gives the verdict (QUARANTINE, PROCEED_WITH_PENALTY
or PROCEED). A deferred detection or call rate is a hard failure; a deferred bead count or provisional bisulfite number is recorded.
Attaches the explanations and logs the verdict.

<sub>`stage_0_intake.py` line 896 (80 lines) · arguments: record, verdict_log_path · calls: array run 2</sub>
