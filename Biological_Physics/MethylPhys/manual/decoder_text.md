# The Chain Decoder

<!-- Source of the Chain Decoder in the Official IAM Operations Manual: Instrument Diagnostics and Chain Decoder.
manual/build_decoder.py merges this text with the mechanical facts read from the code (manual/decoder_inventory.json) and refuses
to build when any function the chain loads has no '### file::function' entry here. Edit this file, never the generated output. -->

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

### run_sample.py::_bio_root
Finds the `Biological_Physics` folder by walking up from this file until it finds a folder holding `MethylPhys`, rather than by
counting levels. Written after a 2026-09-22 move changed every script's depth by one and a counted path silently stopped finding
the record folder. Decides nothing about a sample.

### run_sample.py::_assign_run_id
Gives the run its identifier, `RUN-YYYYMMDD-NN`: the UTC date and the next number for that day, counted from the ledger the run is
about to append to. A run makes no claim and passes no bar; it is one execution on one specimen, and the identifier lets a reading be
found again.

### run_sample.py::_covariates
Collects the phenotype and covariates for the run: a JSON object from `--covariates`, then each `--covariate KEY=VALUE` on top. They
are kept in the bundle and the custody record and are never printed in the report's prose. No stage reads them.

### run_sample.py::_sha12
The first 12 hexadecimal characters of a file's SHA-256, used to fingerprint each frozen input in `_versions`. Returns nothing if the
file cannot be read.

### run_sample.py::_versions
Records everything that can change a reading: the UTC time, the Python and methylprep versions, the chain's git commit and whether
the working copy had uncommitted changes, and a short hash and size of each frozen matrix and stage file. Two runs months apart can
be pooled only if these match.

### run_sample.py::_hash_id
Replaces a typed identifier with the first 32 characters of its SHA-256. An identifier that already looks like a hash (16 or more
hexadecimal characters) is kept. The engine never stores a cleartext identifier.

### run_sample.py::_redact
Walks the whole bundle (every string, dictionary key and list) and replaces each whole-token occurrence of the typed identifier with
its hash, so neither the bundle nor the ledger carries the typed name. The printed report keeps the typed name for the operator.

### run_sample.py::_redact.f
The recursive step of `_redact`: a string gets the replacement, a dictionary and a list are walked, anything else is returned as is.

### run_sample.py::_dev_run
Runs the development readings named by `--dev-*` flags (direction, trace cell, foreign cell, brightness interval, NILC-e, atlas e,
B-cell reading, sky with the block-shuffle null, EPIC v2 through SeSAMe). Each writes its own record under `bundle["development"]`,
labeled DEVELOPMENT, and an error in one is caught and recorded. None of them changes the reading, the gauge or the tare.

### run_sample.py::main
The orchestration described above, from parsing the command to writing the ledger row. Decides which stages run, stops the run with
exit code 2 or 3 at the stops listed, carries the same-run references (columns A and C) into the conductor, adds the v4 stages, and
hands everything to the report.

### run_sample.py::main._stopped
True when an intake record's status begins with QUARANTINE. Intake checks it after every step, so a quarantine ends intake at its
cause. Without it a later step overwrote the status, and the verdict named a downstream symptom (seen 2026-09-23: an array-type
mismatch was reported as a detection failure).

### run_sample.py::main._ctx
Builds the comparison record Stage 12b needs from a draw: the identifier hash, the array type and the calibration pipeline.

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
| 0.4 control probes | bisulfite conversion (C/(C+U) per matched BS Conversion I pair, median over pairs), hybridization (high against low concentration, green) and extension (green C and G against red A and T) controls, from the array's own control addresses | a failed hybridization or extension stops the array; the bisulfite number is printed and never stops an array on its own | `HYB_HIGH_LOW_MIN_RATIO` 2.0 and `EXTENSION_BALANCE_RANGE` 0.2 to 5.0, both **provisional**; bisulfite at least 0.95, **provisional** (as written it refuses 731 of 731 healthy arrays) |
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

### stage_0_intake.py::_intake_thresholds
Loads `intake_thresholds_v1.json` from the chain's `Runtime Matrices/Intake` folder (never from a RETIRED folder) when the module is
imported. Stage 0 refuses to start without it, so no gate can run on a number typed into the code.

### stage_0_intake.py::normalise_specimen
Lowercases the specimen text, turns underscores into spaces and collapses repeated spaces, so `Whole_Blood` and `whole blood` are the
same specimen.

### stage_0_intake.py::specimen_refusal
Returns nothing when the specimen is accepted, otherwise the plain refusal: which specimen, why the chain has no reference for it,
and what is accepted. Called by `run_sample.py` before any file is read.

### stage_0_intake.py::_infer_platform
Turns the IDAT header's address count into a platform: below 800,000 is 450K, below 1,080,000 EPIC v1, otherwise EPIC v2.

### stage_0_intake.py::read_idat_nsnps
Reads only the start of the green IDAT: the four-byte `IDAT` mark, the field table, and field 1000 (the number of addresses). It never
decodes intensities. Returns the count and a status (OK, NOT_IDAT, NO_NSNPS_FIELD, FILE_NOT_FOUND or a read error).

### stage_0_intake.py::step_0_1_idat_arrival
Step 0.1: checks the manifest's required fields and array type, that both channels exist, that each file is at least 1 MB, and that the
header's platform family agrees with the declared one; warns on a repeat intake within 24 hours. Stamps a run identifier and the
arrival time, and logs the result.

### stage_0_intake.py::_log_intake
Appends one row per intake step to the intake log, when one is given: time, run identifier, Sentrix ID, declared and detected array
type, specimen, declared sex and age, status and flags.

### stage_0_intake.py::_is_duplicate_within_24h
True when the intake log already holds the same Sentrix ID from the last 24 hours. Only a warning; the array still advances.

### stage_0_intake.py::_is_null_answer
For storing questionnaire answers: a blank or "prefer not to say" becomes no answer; "No" is kept as a real answer.

### stage_0_intake.py::_answer_present
For the questionnaire's context modes: true only for an affirmative answer (not blank, not "prefer not to say", not No, none or not
applicable).

### stage_0_intake.py::_age_from_dob
Age in decimal years from a date of birth and the intake date, rounded to 0.1.

### stage_0_intake.py::parse_questionnaire
Maps a digital questionnaire's answers to the stored covariate record: each field, the age from the date of birth, a smoking bin, and
a list of context modes. The record is stored with the manifest. No stage of the current chain reads these covariates or modes; the
stage names in its field table belong to an earlier engine.

### stage_0_intake.py::_looks_like_cleartext_pii
True when an identifier looks like a name rather than a hash: empty, containing a space or an @, shorter than 16 characters once
dashes and underscores are removed, or not purely letters and digits.

### stage_0_intake.py::_write_manifest
Writes the per-sample manifest as `patient_manifest_<run id>.json` in the manifest folder, when one is given.

### stage_0_intake.py::step_0_2_manifest_creation
Step 0.2: builds the immutable manifest from the 0.1 record (and questionnaire answers, when given), quarantines a cleartext
identifier or a missing core field, and writes the file.

### stage_0_intake.py::_sha256_file
The SHA-256 of a whole file, read in 1 MB pieces.

### stage_0_intake.py::_prior_hashes_for_sentrix
The pairs of file hashes earlier intakes recorded for the same Sentrix ID in the integrity log.

### stage_0_intake.py::_log_integrity
Appends one row to the integrity log (also used for the decision gate's verdict row).

### stage_0_intake.py::step_0_3_integrity_hash
Step 0.3: hashes both files and compares them with earlier intakes of the same Sentrix ID. Identical bytes again is a re-transmission
and stops the run; the same Sentrix ID with different bytes is flagged as a fresh re-run and advances.

### stage_0_intake.py::extract_control_probes
Gets the control-probe medians by class from the IDAT pair through the QC hand-off (`stage_0_1_qc_handoff.decode_qc_inputs`), the
same decoder Stage 1 uses. The array's own controls only.

### stage_0_intake.py::validate_control_probes
Step 0.4's arithmetic. Bisulfite: `bs1 / (bs1 + bs2)` against 0.95, where the QC hand-off passes in the matched-pair conversion
efficiency and its complement under the keys `bisulfite_conversion_I_median` and `bisulfite_conversion_II_median` (so the ratio is the
efficiency itself; the key names are older than the pair method). Hybridization: the high-concentration median over the low-concentration
median, at least 2. Extension: the green-channel (C and G) extension median over the red-channel (A and T) median, between 0.2 and 5;
the record calls these "meth" and "unmeth", which they are not. When the bisulfite number is the only failure, the verdict is PROVISIONAL and the array advances, because that limit has not been
set from healthy arrays. Records the raw medians the gates tested.

### stage_0_intake.py::step_0_4_control_probe_validation
Step 0.4: runs `validate_control_probes` on the hand-off's control summary and stamps the verdict and its numbers. With no decoder
available it marks the check deferred rather than inventing a pass.

### stage_0_intake.py::_normal_sf
The normal survival function, 1 − Φ(z), from scipy when it is installed, otherwise from the complementary error function.

### stage_0_intake.py::compute_detection_p
Per probe, the detection p-value 1 − Φ((I − μ_bg)/σ_bg), with μ_bg and σ_bg this array's negative-control background.

### stage_0_intake.py::validate_detection_p
The share of probes with detection p ≤ 0.01, graded against the thresholds file: above 0.99 PASS, from 0.93 BORDERLINE, below FAIL.

### stage_0_intake.py::step_0_5_detection_pvalue_qc
Step 0.5: computes and grades detection. Without intensities the check is deferred, and a deferred detection never advances.

### stage_0_intake.py::validate_bead_count
The share of probes with at least 3 beads: 0.995 or more PASS, otherwise a warning. Never stops an array.

### stage_0_intake.py::step_0_6_bead_count_qc
Step 0.6: grades the bead counts and flags a low count. Always advances.

### stage_0_intake.py::validate_call_rate
The share of probes passing both detection and bead count, graded: 0.98 or more PASS, from 0.93 BORDERLINE, below FAIL.

### stage_0_intake.py::step_0_7_call_rate
Step 0.7: combines the detection and bead masks probe by probe and grades the call rate. Without masks the check is deferred, and a
deferred call rate never advances.

### stage_0_intake.py::step_0_7b_platform_coverage
Step 0.7b: stamps the platform tag. On a 450K array it stops below 80 % of the reference sites; on EPIC it records `NA_EPIC` and passes.

### stage_0_intake.py::predict_sex
Female when the median log2 Y intensity minus the median log2 X intensity is below −2, otherwise male (minfi's getSex rule).

### stage_0_intake.py::validate_sex
Compares the predicted sex with a declared one. No declared sex: the array's sex is recorded and nothing is compared. A mismatch stops
the array.

### stage_0_intake.py::step_0_8_sex_check
Step 0.8: runs the sex check on the hand-off's X and Y medians and stamps the result.

### stage_0_intake.py::_fmt
Formats a measured number for an explanation, or says "not measured".

### stage_0_intake.py::explain_stage0
Builds the five-part explanation of every check that stopped, penalized or deferred the array (measured on this array, the limit, the
limit's source and whether it is provisional, what it means, what to do). Changes no decision.

### stage_0_intake.py::explain_stage0.add
Adds one explanation to the list.

### stage_0_intake.py::step_0_9_decision_gate
Step 0.9: sorts every step's outcome into hard failures, penalties and deferrals and gives the verdict (QUARANTINE, PROCEED_WITH_PENALTY
or PROCEED). A deferred detection or call rate is a hard failure; a deferred bead count or provisional bisulfite number is recorded.
Attaches the explanations and logs the verdict.

## 3. The QC hand-off: `stage_0_1_qc_handoff.py`

**What it is for.** Stage 0's gates 0.4 to 0.8 need intensities, and intensities come from decoding the IDAT files. This module decodes
the pair just far enough to give those gates real numbers from the array itself, using the same decoder Stage 1 uses (methylprep) and
the control addresses of the array's own manifest. Nothing from any other sample enters.

**Label: standard method** (Illumina's control probes, read by design type the way the minfi package reads them).

**What it returns.**

- **Control summary.** Bisulfite conversion: for each matched BS Conversion I pair (C1 with U1, C2 with U2, ...), the converted signal
  over converted plus unconverted, C/(C+U), and the median over the pairs. Hybridization: the medians of the high- and
  low-concentration controls (green). Extension: the median of the C and G controls (green) and of the A and T controls (red).
- **Negative-control background.** The negative controls summed over both channels, summarized robustly as the median and 1.4826 times
  the median absolute deviation (MAD). The mean and standard deviation are returned beside them. The robust form was adopted on
  2026-09-23 after a few EPIC negative controls carrying real signal (one at 29,110 against a median of 134) pushed the standard
  deviation to 1,676 and failed half the probes of a good array; on a clean 450K array the two estimates agree.
- **Per-probe total intensity.** A type II probe is read at its one address in both channels; a type I probe at its two addresses in
  its own color only (the other color would add only background).
- **Bead counts**, per probe the smaller of the two channels' counts (and of the two addresses for type I).
- **Sex intensities**: log2 of the median total intensity of the X probes and of the Y probes.

**How to break it.** A truncated or corrupt IDAT makes the decoder fail; `run_sample.py` then quarantines the array as a corrupt IDAT.
A missing manifest on the machine is an environment failure (exit code 3), never a judgment of the array.

### stage_0_1_qc_handoff.py::_manifest
Loads the array type's manifest from methylprep once and keeps it for the rest of the run.

### stage_0_1_qc_handoff.py::_channel_means
Reads one IDAT file (gzipped or plain) with methylprep and returns the mean intensity at every address and, when asked, the bead count.

### stage_0_1_qc_handoff.py::_median
The median intensity over a list of control addresses, ignoring addresses the file lacks; nothing if none are present.

### stage_0_1_qc_handoff.py::decode_qc_inputs
Builds everything Stage 0's gates read, as listed above, from the two files and the manifest. Returns the probe identifiers in the same
order as the intensities and bead counts, so Stage 1's call rate can line them up later.

### stage_0_1_qc_handoff.py::decode_qc_inputs.ctl_addr
The manifest addresses of one control type, optionally narrowed to extended types containing (or not containing) a word.

### stage_0_1_qc_handoff.py::decode_qc_inputs._at
The intensity at each probe's address, with a gap where the probe has no such address.

### stage_0_1_qc_handoff.py::decode_qc_inputs.sex_median
log2 of the median total intensity over the X (or Y) probes the array measured, and how many probes that was.

## 4. Stage 1, calibration: `stage_1_idat_calibration.py`

**What it is for.** Turning one array's raw intensities into one calibrated methylation value (beta) per site, from this array alone,
and removing every probe that reads at background. Without calibration the methylated peak sits near 0.85 instead of 0.95 and every
later number moves.

**Label: standard method.** Noob (normal-exponential out-of-band background correction with dye-bias correction), as implemented in the
methylprep package, run on one array at a time with its own control and out-of-band probes. No cohort and no reference enters.

**The formula.** For each probe, beta = M / (M + U + offset) after noob correction of the methylated (M) and unmethylated (U)
intensities. Detection: methylprep's poobah p-value, each probe against this array's own out-of-band background; a probe with p above
0.05 is removed before any stage reads a value ("no measurement at background").

**What it pins.** The commissioned numbers were made with methylprep 1.7.1 and three Illumina manifests, fixed by their SHA-256
(`METHYLPREP_VERSION`, `MANIFEST_SHA256`). Stage 1 refuses to run on another version or a changed manifest, before and after
methylprep fetches its files. Development work can override this with `IAM_ALLOW_UNPINNED_STAGE1=1`; the results are then not the
commissioned chain's.

**What else it returns.** The detection summary (probes measured, detected, removed); the control medians; and the array's own noise,
from its SNP (rs) probes. Those probes cluster at 0, 0.5 and 1 by genotype, so their spread around each cluster is pure measurement
noise: σ²(β) = a + b·β(1 − β), with a the mean variance of the two homozygous clusters and b = 4·(variance of the heterozygous
cluster − a). The sky uses this as the array's own noise term.

**How to break it.** Run with methylprep 1.7.2, or edit one byte of a cached manifest: Stage 1 must refuse. An EPIC v2 pair reaching
Stage 1 is refused as an array type it does not calibrate; a truncated file is quarantined as a corrupt IDAT.

### stage_1_idat_calibration.py::_check_pins
Refuses to run unless methylprep is version 1.7.1 and every cached manifest matches its pinned SHA-256 (or the development override is
set). Called before calibration and again after methylprep fetches any missing manifest.

### stage_1_idat_calibration.py::_ensure_methylprep
Imports methylprep, installing the pinned version into the running Python once if it is missing, and explains what to do when the
install or import fails.

### stage_1_idat_calibration.py::_detect_array_type
Reads the green IDAT with methylprep and maps its address count to methylprep's array type; also returns the barcode.

### stage_1_idat_calibration.py::calibrate_idat_to_beta
Copies the pair into a private temporary folder with a one-row sample sheet (so methylprep processes this array alone), runs noob with
poobah detection and the control and SNP probes exported, keeps the cg probes with poobah p ≤ 0.05, and returns the betas with the
record described above. The temporary folder is always deleted.

### stage_1_idat_calibration.py::calibrate_idat_to_beta.med
The median of one control type in one channel, from methylprep's exported control table.

### stage_1_idat_calibration.py::_stage_idat
Copies an IDAT into the staging folder under the name methylprep expects, unzipping it if it is gzipped.

## 5. The conductor: `conductor_v3.py`, and Met-A's reading: `stage_m_met_a.py`

**What it is for.** The conductor takes one array's calibrated values and makes the array reading: the platform check, Stage T step 1
(self-tare II), Stage A (the cell mix), Stage M (Met-A), the noise index, Stage T step 2 (the same-run tare), and Stage MC (the C-Score).
It returns one record per stage, each with its reason when it gives no number.

**The order inside `run_neutrophil`.**

1. Refuse a specimen outside the list (as `run_sample.py` already did) and an array that is not EPIC v1.
2. **Stage T step 1, self-tare II**: map this array's values onto the reference arrays' scale by its own fixed sites (below).
3. Isolated neutrophils: **Stage M** against the healthy reference. Whole blood (and buffy coat, and constructed mixtures): **Stage A**,
   then **Stage M** against the expectation for this blood's own mix.
4. The **noise index** N, from the unmapped values.
5. **Stage T step 2**, the median tare against the same-run healthy references.
6. The **noise gate**: no gauge state when N cannot be formed, or when N is above the reference arrays' range and the reading is not tared.
7. The scope label: commissioned on isolated neutrophils; preliminary (labeled development) on whole blood and buffy coat.
8. **Stage MC**, the C-Score, and its own same-run tare; the commissioned label on whole blood.

### The physics: what Met-A measures

**Label: from IAM's physics.** A cell holds a methylation pattern against thermal noise by paying for it. A site held fully methylated
or fully unmethylated carries no uncertainty; a site that has drifted toward 50 % carries up to one bit. The entropy of one site's
beta value is

  H(β) = −β log₂ β − (1 − β) log₂(1 − β),

and Met-A is the mean entropy at the cell type's identity sites divided by the healthy reference:

  isolated neutrophils: Met-A = mean_i H(β_i) / H_ref, with H_ref = 0.33026 bits (6 healthy neutrophil arrays);
  whole blood: Met-A = mean_i H(β_i) / mean_i H(e_i), with e_i = Σ_g f_g μ_g,i.

Here f_g is this blood's share of cell group g (Stage A), and μ_g,i is the healthy mean of group g at site i, from purified cells. In
whole blood the denominator is what a healthy blood with exactly this person's cell mix would read, so a change in the mix does not move
Met-A; only a change in how well the cells hold their marks does.

**The gauge.** Healthy is 1 (the middle mark, measured). Normal is 0.95 to 1.05. The floor (derived: thermal kicks against about one
ATP per site, M = 20.94) sits millions of times below any reading; the ceiling (derived: one bit per site, every identity site
scrambled) sits about three times above healthy. What the physics fixes is the scale and its two ends. The healthy middle is measured
once and never fitted to the sample.

**The ceiling caveat (`_ceiling`).** H(β) peaks at β = 0.5. Met-A rises with loss of the pattern only while the methylated sites stay
above 0.5. Every reading records the mean β of the cell's methylated sites and flags `past_entropy_ceiling` when it has fallen below 0.5,
because past that point further loss lowers Met-A.

### Stage A, the cell mix (whole blood)

**Label: standard method** (reference-based deconvolution by non-negative least squares, used in the field for over a decade).

The blood's shares of 8 cell groups (neutrophils, eosinophils, basophils, monocytes, B, NK, CD4 T, CD8 T) are solved from 963 marker
sites: minimize |y − M f| with f ≥ 0, then f is scaled to sum to 1. M holds each group's mean β at each marker, from 91 purified-cell
EPIC arrays of two published purified-blood-cell studies (GSE110554 and GSE167998), calibrated by this chain's own Stage 1.

- **Why its own reader, not the atlas:** the expectation e must come from the same platform and the same reference as the profiles it
  multiplies, or any platform difference leaks into Met-A. IAM-Atlas v2.1 and the Deconvolver report the shares separately (Stage D21).
- **Why no marker is an identity site:** solving the mix from the sites Met-A reads would let a change in the cells' marks move the mix
  that is supposed to explain them.
- **Gate:** at least 90 % of the 963 markers measured (867); otherwise no mix, no Met-A, and the reason gives the count.
- **Built from, tested on:** built from the 91 purified arrays (`chain_tests/blood_comp.py`). Rebuilt from GEO by the builder's own rule
  on 2026-10-03, reproducing every group mean to 5 × 10⁻⁶, and again from a fresh clone (`metaa_rebuild`, PASS 2026-10-18).

### Stage M, Met-A

- **Identity sites:** 6,000 EPIC sites the healthy neutrophil holds fixed, chosen on the purified neutrophil arrays (across-array SD
  ≤ 0.05; mean β 0.75 to 0.95, or 0.05 to 0.25; up to 3,000 per state). The isolated and whole-blood paths read the same 6,000.
- **The healthy reference (H_ref = 0.33026 bits):** the mean entropy at those sites on 6 purified healthy neutrophil arrays, each
  physical array counted once. The file holding it is still named `metA_floors_v1_3.json`, and its key is `floor`; it is the middle
  mark, not the floor.
- **Held-out precision:** each of the 6 arrays read with the sites re-chosen on the other 5 (`metA_floors_v1_3_loo.csv`) gives 0.983 to
  1.045. This is printed with every isolated reading.
- **Gate:** at least 90 % of the 6,000 sites measured; otherwise no Met-A, and the reason gives the count. In whole blood, a neutrophil
  share of 0 also withholds the reading.
- **Detection at this person's mix:** the reading records how far a 1 % loss of the neutrophils' pattern, planted at this blood's own
  neutrophil share, would move it (`shift_per_1pct_loss`). Stage T turns this into the person's detection limit.
- **The departure map** for the C-Score: per site, z_i = (H(β_i) − H(ref_i)) / s_i, with ref the healthy neutrophil (isolated) or the
  expectation e (blood), and s_i the shrunk healthy SD of H among neutrophils.

### The noise index and its gate

The array's own noise is read where no blood cell has anything to say: 48,528 EPIC sites that every purified blood group holds fixed (each
group's mean ≤ 0.03 or ≥ 0.97 and SD ≤ 0.02; none an identity site). N is their mean entropy, read before self-tare II. **Gate:** with fewer
than 90 % of these sites measured the noise is unknown and no gauge state is shown, tared or not. With N above N_max = 0.149 (the top of the
reference arrays' range, `noise_gate_EPIC_v1.json`), the state is shown only after the same-run tare.

### Stage T, the tare

**Label: standard method** (normalization against references run the same way).

- **Step 1, self-tare II** (`dev_stages.selftare_map`, on every reading since 2026-10-04). Each probe design (type I, type II) is mapped
  onto the 6 reference arrays' scale by this array's own anchors: β' = L_r + (β − L)(U_r − L_r)/(U − L), with L and U this array's mean β
  over its own fixed low and high sites of that design, and L_r and U_r the same means on the reference arrays (type I 0.0186 and 0.9820;
  type II 0.0555 and 0.9486). It maps only the 6,963 sites Stages A and M read (the 6,000 identity sites and the 963 markers). A design
  whose anchors are missing, or closer than 0.1, is left unmapped. No other array enters; nothing is fitted.
- **Step 2, the median tare.** A_rel = A / median(A of at least 3 healthy arrays run the same way: same slide, else same batch). Normal is
  0.95 ≤ A_rel ≤ 1.05. The reference spread is the SD of the references' A over their median. The **detection limit** is 2 × spread /
  shift per 1 % loss: the smallest loss of the cell's pattern, in percent, this person's reading could show. A sample that is itself in
  the reference list is excluded from it. Fewer than 3 references: the reading is printed as untared, and the reason says what to give.
- **Why:** untared Met-A in whole blood carries a laboratory offset of several percent. The tare cancels it, which is why Met-A needs no
  cohort: it compares the person with a prediction for their own blood and centers it on their own run.

### Stage MC, the C-Score

**Label: from IAM's physics, read with a standard statistic.** Drift that comes from holding marks against noise is scattered: each site
fails on its own. Loss that comes from a process acting on a region, such as a tumor's, is clustered along the genome.

- **The statistic:** take the departure map z in genomic order, average blocks of 10 consecutive sites, and take the variance of the
  block means times the block length over the variance of z. For scattered departures it is 1. Divided by the healthy median of that
  number (1.0103, from the 6 reference arrays held out one at a time), healthy reads about 1.
- **Its tare:** the same rule as Met-A's: C_rel = C / median(C of at least 3 same-run healthy references).
- **The band (commissioned on whole blood, 2026-10-15):** 0.877 to 1.152, the 2.5 to 97.5 % range of tared C on 617 healthy arrays from
  19 studies. One-sided: above 1.152 is a clustered departure; below 0.877 is less clustered than healthy, not a departure.

**How to break it.** Plant a 1 % loss of the neutrophils' pattern scattered across sites: Met-A must move and the C-Score must not (seen
on Group-A's 278 arrays: Met-A by up to 0.036, the C-Score by at most 0.006). Plant the same loss in runs of neighboring sites: the C-Score
must rise. Give 2 references: the tare must refuse. Remove 15 % of the identity sites: Met-A must withhold its reading and say how many
were measured.

### conductor_v3.py::ref
Loads the neutrophil reference (`neutrophil_reference_v1_2.json`) once: the 6,000 sites in genomic order, the healthy mean and shrunk SD
of H at each, and the C-Score's block length and healthy baseline.

### conductor_v3.py::noise_sites
Loads the 48,528 noise sites once.

### conductor_v3.py::noise_gate
Loads N_max and its rule once.

### conductor_v3.py::noise_index
N = mean H(β) over the noise sites this array measured, or nothing when fewer than 90 % were measured; with the counts.

### conductor_v3.py::_bc
Loads the blood composition matrix (`blood_composition_EPIC_v1.json`) once: the 963 markers and 8 group means at each, and the 8 groups'
healthy means at the 6,000 identity sites.

### conductor_v3.py::min_markers
The number of markers Stage A needs: 90 % of 963, rounded up (867).

### conductor_v3.py::stage_a_composition
Stage A: solves the 8 shares by non-negative least squares on the measured markers and scales them to sum to 1; records the fit's mean
absolute residual. Refuses, with the count, below 867 markers.

### conductor_v3.py::_clustering
The C-Score's statistic on a departure map: variance of block means of 10 sites (× √10) over the site variance; nothing with fewer than 10
blocks.

### conductor_v3.py::_state
Normal from 0.95 to 1.05, otherwise above or below Normal.

### conductor_v3.py::stage_m_blood
Stage M in whole blood: the expectation e for this blood's mix, Met-A = mean H(β) / mean H(e) over the measured identity sites, the shift a
1 % loss would make at this neutrophil share, the ceiling check and the departure map. Withholds the reading, with the reason, when there is
no mix, no neutrophils, or under 90 % of the sites.

### conductor_v3.py::_ceiling
The mean β of the neutrophil's methylated identity sites, and whether it has fallen below 0.5, past which Met-A no longer rises with loss.

### conductor_v3.py::stage_m_isolated
Stage M on isolated neutrophils: Met-A against the healthy reference (`stage_m_met_a.read`), the 1 % shift, the ceiling check and the
departure map against the healthy neutrophil. Keeps the untared state as `state_own_floor` and marks the reading untared until Stage T.

### conductor_v3.py::stage_mc_cscore
Stage MC: the C-Score from the departure map, divided by the healthy baseline, with the share of sites more than 3 SD out. With no map,
no C-Score and the reason.

### conductor_v3.py::_ref_records
Turns the same-run references into records (A, neutrophil share, N, C), skipping empty values and the sample itself.

### conductor_v3.py::stage_t_tare
Stage T step 2: the median tare of Met-A, the reference spread and the detection limit; untared, with the reason, under 3 references.

### conductor_v3.py::stage_t_cscore
The C-Score's same-run tare and its call against the band (above 1.152 a clustered departure).

### conductor_v3.py::stage_t_selftare_ii
Stage T step 1: runs `dev_stages.selftare_map` on the array and returns the mapped values with the anchors and maps; unmapped values and a
NOT_RUN record if the map file is missing.

### conductor_v3.py::platform_refusal
Nothing for an EPIC v1 array; otherwise the refusal: another array type from Stage 0, EPIC v2 probe names, or 700,000 probes or fewer.

### conductor_v3.py::run_neutrophil
The whole conductor, in the order above. Returns the array reading's records (mix, Met-A, tare, C-Score) and the departure map for the sky.

### dev_stages.py::_json
Loads a file from `Runtime Matrices/Development` once and keeps it.

### dev_stages.py::anchors
This array's mean β over each of the four fixed-site sets (type I low and high, type II low and high), with how many it measured.

### dev_stages.py::selftare_map
Stage T step 1, as described above: maps each design's values by β' = L_r + (β − L)(U_r − L_r)/(U − L) at the sites whose design is
recorded and returns the maps' anchors and slopes.

### stage_m_met_a.py::_floors
Loads `metA_floors_v1_3.json` (the 6,000 identity sites and the healthy reference H_ref) once.

### stage_m_met_a.py::_H
The entropy of a beta value in bits, with β clipped to [10⁻⁶, 1 − 10⁻⁶] so a value of exactly 0 or 1 gives 0, not an error.

### stage_m_met_a.py::platform_of
The platform from the values themselves: EPIC v2 when any probe name carries the v2 design suffix; else EPIC when more than 700,000 probes,
else 450K.

### stage_m_met_a.py::available
The cell types with a healthy reference on each platform that are in scope (neutrophils).

### stage_m_met_a.py::precision
The held-out readings of the reference arrays for one cell type: how many, the share in Normal, the SD and the range.

### stage_m_met_a.py::read
Met-A on a purified specimen of one cell type: refuses a cell outside scope, a platform without a reference, or under 90 % of the sites;
otherwise A = mean H(β) / H_ref with its state and the held-out precision.

## 6. Chain v4: the cell shares, IAM-A and the sky: `stage_v4.py`

**What it is for.** Bringing the instruments commissioned after chain v3 into the one run: Stage D21 (each blood cell's share, from
the IAM-Deconvolver against IAM-Atlas v2.1), Stage Q2 (IAM-A on sequencing data) and the sky (a picture of the C-Score's departures).
Nothing here is fitted at run time; every reference is a committed, stamped file.

**Stage D21's gate.** The six-cell blood panel (neutrophils, monocytes, B, CD4 T, CD8 T, NK) is read on the sites that separate those
cells in IAM-Atlas v2.1. With fewer than half of those marker sites measured, the shares are withheld and the reason gives the count.
The 50 % line is operational, not measured. Shares of neutrophils, monocytes, CD4 T and CD8 T cells carry the commissioned label; B and NK
cells are printed as development. The error bars are information, not calibrated intervals.

**Stage Q2's order.** Count the molecules (from a `.pat` file or an aligned BAM), refuse a file with no qualifying molecules or tiles,
run the sample-shape check, then read A against the sample's own reference and band v1. The commissioned label attaches to CD8 T cells
(the parent type and its subsets) read by whole-genome bisulfite; every other cell type or chemistry is labeled development.

**The sky.** The C-Score's departure map, in the C-Score's own genomic order, laid on a HEALPix sphere in NESTED order: consecutive
indices fill compact patches, so a run of neighboring departures (what the C-Score scores) shows as a patch and scattered departures as
speckle. NSIDE is the smallest power of 2 giving at least one pixel per two sites. Departures are drawn in units of the sample's own
spread (median and 1.4826 × MAD), which leaves the C-Score unchanged, because the C-Score is unchanged by shifting or rescaling every
departure. The pixel-to-angle conversion is the standard HEALPix NESTED algorithm (Górski et al. 2005), written out here so the chain
does not need healpy; it matches healpy to 10⁻¹⁴ radians. **Label: standard method** (the HEALPix sphere of cosmology, used as a
display). Nothing is scored from the picture.

### stage_v4.py::stage_d21
Stage D21: loads the panel's reference and markers once, counts the markers measured, withholds below half, otherwise runs the
Deconvolver and labels each cell's share commissioned or development.

### stage_v4.py::count_sequencing
Counts a `.pat` file (gzipped or plain) on the hg19 tiles, or an aligned BAM against its reference FASTA, with Stage Q2's one counter.

### stage_v4.py::stage_q2
Stage Q2 as above: the refusals, the shape check, the reading A with its predicted value, and the commissioned or development label.

### stage_v4.py::_compress
Takes every other bit of an integer: the step of the HEALPix algorithm that splits a NESTED pixel index into its x and y coordinates.

### stage_v4.py::nest_pix2ang
HEALPix NESTED pixel centers to colatitude and longitude.

### stage_v4.py::cscore_sky
Draws the C-Score's departures on the sphere in a Mollweide view and returns a summary (NSIDE, sites, pixels drawn, the share of
departures beyond 3). Without a departure map it draws nothing and says why.

### stage_v4.py::png_data_uri
Embeds a figure in the HTML report as a data URI.

## 7. The IAM-Deconvolver: `stage_deconv_v21.py`

**What it is for.** Given a mixed sample's array values, each cell's share, with a 95 % interval.

**Label: standard method** (weighted non-negative least squares against reference cell profiles), with IAM-Atlas v2.1 as the reference.

**The reference.** For each site and cell, the atlas's best value μ and the spread a new person's cell has around it:
sd² = ((hi − lo)/3.92)² + donor_sd² + lab_sd² + 0.01². The first term is the uncertainty of the atlas's average (its 95 % range); donor_sd
is the person-to-person spread the atlas fit measured; lab_sd is the laboratory-to-laboratory spread, the SD of the fit's own
array-laboratory offsets (18 laboratories, 0.0065 when sealed); 0.01² is a floor. Version 1 used only the first term and is kept as
`SPREAD="mean_only"`.

**The markers.** For each cell, the 50 sites where it sits furthest above, and the 50 furthest below, its nearest other cell in the panel,
in units of their combined spread: z = |μ_c − μ_nearest| / √(sd_c² + sd_nearest²).

**The solver.** Non-negative least squares on the markers, each site weighted by 1 / √(Σ_k f_k² sd_k² + lab²), starting from equal shares
and refined over three passes. The shares are scaled to sum to 1 and the raw sum is reported.

**The intervals.** 200 refits, each on resampled marker sites and on cell profiles redrawn from the atlas's spread (a new person's cells
are not the average cell). The 2.5th and 97.5th percentiles give the interval.

**Built from, tested on.** Built from IAM-Atlas v2.1 alone. Tested by planted mixtures drawn from the atlas's own spread with a laboratory
shift (the file's `planted` test), then by the sealed real test: 200 mixtures of held-out purified arrays from E-MTAB-2145, a laboratory
the atlas never saw. Commissioned 2026-10-17 on the shares of neutrophils, monocytes, CD4 T and CD8 T cells: mean share error 0.016, every
number reproduced from a fresh clone. Sealed bars: mean error ≤ 0.03; at least 80 % of truths inside their intervals.

**What it reads outside its own folder.** The laboratory spread comes from `doors/data/DEV_ATLAS_V21_01/transfer_v21_final.json`, a file in a
test-record folder. It is a runtime input and belongs in the chain's references (a to-do item of this Decoder).

**How to break it.** Read a pure array of one cell: its own share should be at least 0.80. Plant a 5 % share of one cell into a mixture:
its interval should exclude 0 in most mixtures. Remove half the markers: the shares must be withheld.

### stage_deconv_v21.py::reference
Reads μ, its range and the donor spread for the named cells from the atlas's compact parquet files (only those columns; 65,535 marks a
missing value) and builds sd as above.

### stage_deconv_v21.py::markers
The marker sites for the panel, as above.

### stage_deconv_v21.py::_solve
The three-pass weighted non-negative least squares.

### stage_deconv_v21.py::deconvolve
The shares, raw sum, 95 % intervals and sites used for one sample.

### stage_deconv_v21.py::sample
Inside the planted test only: builds a planted array, each cell's DNA drawn from the atlas's donor spread, mixed in known shares, plus a
laboratory shift and array noise.

### stage_deconv_v21.py::sample_check
For the second sealed test: a pure array passes when it reads at least 0.80 of its own cell. Failing arrays are listed and not mixed.

## 8. IAM-Atlas v2.1's commissioning rules: `stage_atlas_commission.py`

**What it is for.** The scorers IAM-Atlas v2.1 was commissioned by, and the table that says which uses of the atlas are commissioned.
The chain loads it for the laboratory spread (`lab_sd`) and for that table.

**Label: standard method** (a posterior predictive interval for a new sample from a new laboratory, scored by coverage).

**The uses.** U1, reference profiles: a healthy cell from any laboratory reads inside the atlas's range for that cell. Commissioned for
cortical neurons, lung bronchus epithelium and naive CD4 T cells on 450K. U2a, blind sites: commissioned for colon and lung. U3,
composition: development. A use not listed is development wherever it appears.

**The bars (as amended and sealed 2026-10-16, before the atlas existed).** U1: at least 95 % of a cell's healthy held-out samples, all
laboratories pooled, cover 80 % to 97 % of their sites inside the atlas's 90 % predictive range. U2: a tissue passes when its epithelium
is assessed and at least 95 % of its assessed cells reach a share of 0.90. U3: neutrophil mean absolute error ≤ 0.02 and at least 95 % of
samples within 0.05. A sample the chain withholds is listed, not scored.

### stage_atlas_commission.py::lab_sd
The SD of the atlas fit's array-laboratory offsets, read from `transfer_v21_final.json` at run time, never typed.

### stage_atlas_commission.py::predictive_interval
The 90 % predictive range for a new sample from a new laboratory. Arrays: N(μ, sd² + lab² + floor). Sequencing: the array value carried
through the platform map, plus binomial counting noise at the sample's coverage, the laboratory scatter and the floor.

### stage_atlas_commission.py::u1_coverage
The share of a sample's sites that fall inside the predictive range.

### stage_atlas_commission.py::u1_verdict
U1's verdict for one cell: commissioned when at least 95 % of its scored samples have coverage between 0.80 and 0.97.

### stage_atlas_commission.py::u2_share
The share of a tissue set's sites within tolerance of the anchor.

### stage_atlas_commission.py::u2_verdict
U2's verdict for one tissue.

### stage_atlas_commission.py::u3_neutrophil
U3's neutrophil error: the mean absolute error, the share within 0.05, the largest error, and whether both bars are met.

### stage_atlas_commission.py::atlas_v21_use_status
Commissioned or development for one use of the atlas, read from the table above.

## 9. IAM-A, each sample against its own reference: `stage_q2_selfref.py`

**What it is for.** Reading, from single DNA molecules, how well a cell holds its methylation compared with what its own state predicts.
No other person enters the reading.

**Label: from IAM's physics.** At a held site, writing (on SAM) and erasing (on oxygen and 2-oxoglutarate) are both running. Each
interval a mark is lost with probability f and restored with probability g, so the share of held sites that are wrong at any moment
settles at the **held error**

  ε = f / (f + g),  and the holding energy is E_hold = kT ln[(1 − ε)/ε].

The balance is set by the ratio, not by how often the cell divides; division only sets how fast the balance is reached. Measured this way
on 56 healthy cell types, E_hold is 3.08 kT per interior opportunity (`holding_energy_q2.json`).

**What is counted (one counter, two readers of the same molecules).**

- **The held error ε.** On molecules with at least 6 CpG calls and at least 80 % methylated (a held run), the isolated unmethylated
  interior calls (a T between two Cs) over the interior calls. The first and last call of a molecule are never counted, because a hole
  there cannot be told from the end of the run.
- **The loose share L.** The genome is cut into tiles of 2,000 consecutive CpGs. A tile is loose when its molecules' mean methylation is
  below 0.70. L is the share of tiles (each with at least 200 calls) that are loose: how much of its genome this cell has let go.
- **No molecule is used twice.** Molecules alternate: even ones are counted for L, odd ones for ε. The reference (from L) and the
  quantity it judges (ε) never share a molecule. In a `.pat` file the alternation is by line, and a line carries every identical
  molecule with its count, so all copies of one pattern go to the same side.

**The self-reference line.** Across healthy cell types, ε rises in a straight line with L:

  ε_predicted = firm + slope × L,  with firm = 0.04026 and slope = 0.03462 (`selfref_line_v1.json`).

This line was measured once on 153 healthy samples of 56 cell types (a public whole-genome bisulfite atlas of purified human cells,
GSE186458), counted by this same counter. A Theil–Sen fit was used, the median of pairwise slopes, which is robust to outlying cell types.
The median distance of a cell type from the line is 0.0025. The line is stored on the raw axis this counter writes; there is no scale
factor anywhere. What the intercept means physically is the subject of a recount (to-do step 14); the reading uses it only as a measured
constant.

**The reading.** IAM-A = ε measured / ε predicted at the sample's own L. 1 is on the healthy line; above 1 is more holes than its own
state predicts. **Band v1: 0.7787 to 1.0551**, the 2.5th to 97.5th percentile of 156 pooled healthy readings.

**Built from, tested on.** The line, the shape reference and 153 of the band's 156 readings come from the same 153 healthy samples
(GSE186458); the band's other 3 readings are from a second laboratory (GSE49627). So the band describes the healthy spread inside the
set the line was measured on. What commissioned IAM-A was the sealed grand test on held-out samples from a third study: healthy CD8 T
cells, of which 8 were read (5 declined by the shape check) and 7 read Normal (bar: at least 75 % inside, at least 8 read).

**The sample-shape check (a gate).** A sample is read only if the 10th percentile of its tile means is at most 0.0543 below the lowest
healthy sample of its own cell type. The margin is the 90th percentile, over the 53 cell types with at least 2 healthy samples, of the
spread of that percentile within a type. It is computed from the reference, never typed. A cell type with no healthy reference is not read.

**The stamp.** Every output this stage writes carries a chain stamp: the stage, a SHA-256 of the chain's code, a SHA-256 of the result,
its inputs, the Master Testing Manifest rows it belongs to, and the time. The line, band and shape reference are refused if they were
edited after the chain wrote them, because the result's hash no longer matches its stamp.

**How to break it.** Plant isolated holes into held runs of a constructed BAM: ε and IAM-A must rise (release check V7 does this through
the one command). Hand it a cell type with no healthy reference: it must not read. Edit one number in `band_v1.json`: the chain must refuse it.

### stage_q2_selfref.py::chain_sha
The SHA-256 over every `.py` file of the chain folder, names and contents, in sorted order.

### stage_q2_selfref.py::_canon
A result in canonical JSON (sorted keys, no spaces), so the same result always hashes the same.

### stage_q2_selfref.py::payload_sha
The SHA-256 of a result without its stamp.

### stage_q2_selfref.py::file_sha
The SHA-256 of a file, read in 16 MB pieces.

### stage_q2_selfref.py::_rows
The manifest rows a result belongs to, given directly or through the `IAM_MANIFEST` environment variable.

### stage_q2_selfref.py::stamp
Attaches the chain stamp to a result, hashing it exactly as it will be stored (JSON turns number keys into text).

### stage_q2_selfref.py::write_json
Writes a stamped result and reads it back to confirm the stored file verifies.

### stage_q2_selfref.py::write_csv
Writes a table and its stamp beside it, as `<file>.stamp.json`.

### stage_q2_selfref.py::hg19_tiles
The hg19 tiles of 2,000 consecutive CpGs on each autosome, in CpG-index coordinates; a partial last tile is dropped.

### stage_q2_selfref.py::_finish
Turns tile counts and hole counts into the record: L, the number of tiles, ε and its opportunities, and the 10th, 50th and 90th
percentiles of the tile means for the shape check.

### stage_q2_selfref.py::count_pat
The `.pat` reader: even lines are added to their tile; odd lines that qualify as held runs add their interior calls and isolated holes.

### stage_q2_selfref.py::_cpg_index
The positions of every CpG in a chromosome's sequence.

### stage_q2_selfref.py::_calls_of
One aligned read's methylation calls at CpGs, from the Bismark XM tag or, for gemBS, from the read against the reference. Skips the first
and last 3 aligned bases. Also counts non-CpG conversion (a check on bisulfite conversion) and, on every 20th read, A/T mismatches (a check
on sequencing error).

### stage_q2_selfref.py::_merge
Merges the calls of the two reads of a pair into one molecule; a CpG on which they disagree is dropped.

### stage_q2_selfref.py::count_bam_region
Counts one 10 Mb region of an alignment file: primary, mapped, non-duplicate reads with mapping quality at least 10, pairs merged, then
the same alternation and the same two counts as the `.pat` reader. Also counts calls per qualifying molecule for the response curve.

### stage_q2_selfref.py::count_bam_region.done
Counts one finished molecule: even molecules to their tile, odd held runs to ε.

### stage_q2_selfref.py::combine_bam
Adds the regions' counts into one record, with the conversion-failure and substitution rates.

### stage_q2_selfref.py::count_bam
Splits the autosomes into 10 Mb regions and counts them in parallel.

### stage_q2_selfref.py::build_line
How the line was built: per cell type, the median L and the opportunity-weighted ε, then a Theil–Sen line across cell types. Not called at
run time; the line is read from its frozen file.

### stage_q2_selfref.py::line
Loads the frozen line and refuses it if it was edited after the chain wrote it.

### stage_q2_selfref.py::predict
ε predicted at a loose share: firm + slope × L.

### stage_q2_selfref.py::residual
A count against the line: measured ε, predicted ε, and their difference.

### stage_q2_selfref.py::build_shape_reference
How the shape reference was built: per cell type, the lowest healthy 10th percentile, and the margin. Not called at run time.

### stage_q2_selfref.py::shape_reference
Loads the frozen shape reference and refuses it if edited.

### stage_q2_selfref.py::shape_check
The sample-shape gate described above, with the measured percentile, the limit and the reason.

### stage_q2_selfref.py::reading
IAM-A = measured ε / predicted ε.

### stage_q2_selfref.py::build_band
How band v1 was built: the 2.5th to 97.5th percentile of pooled healthy readings. Not called at run time.

### stage_q2_selfref.py::band
Loads band v1 and refuses it if edited.

### stage_q2_selfref.py::grand_verdict
The sealed grand test's scorer: each healthy sample shape-checked, each read sample inside or outside the band; met when at least 75 % of at
least 8 read samples are inside.
