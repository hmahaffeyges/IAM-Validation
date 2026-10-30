# The Chain Decoder

## The chain in numbers

Read from the code at build time: the one command can load **36 Python files** holding **442 functions** (nested ones included); 442 of them have their entry in this Decoder, and a push is refused when one does not. Called: how many of a file's functions each traced run used (the array run on a healthy whole blood; the sequencing run on the release check's planted BAM).

| file | lines | functions | called, array run | called, sequencing run |
|---|---|---|---|---|
| `MethylPhys_Interface/run_sample.py` | 1,214 | 19 | 11 | 10 |
| `stage_0_intake.py` | 1,190 | 42 | 33 | 1 |
| `stage_0_1_qc_handoff.py` | 194 | 7 | 7 | 0 |
| `stage_1_idat_calibration.py` | 232 | 6 | 6 | 0 |
| `conductor_v3.py` | 317 | 20 | 18 | 0 |
| `stage_m_met_a.py` | 102 | 7 | 3 | 0 |
| `MethylPhys_Interface/report_v4.py` | 447 | 30 | 17 | 17 |
| `MethylPhys_Interface/report_v3.py` | 297 | 9 | 9 | 9 |
| `MethylPhys_Interface/report_figs_v4.py` | 157 | 8 | 6 | 2 |
| `dev_stages.py` | 339 | 27 | 1 | 0 |
| `stage_v4.py` | 184 | 11 | 4 | 2 |
| `stage_deconv_v21.py` | 200 | 6 | 4 | 0 |
| `stage_q2_selfref.py` | 372 | 34 | 0 | 11 |
| `stage_atlas_commission.py` | 62 | 8 | 1 | 0 |
| `MethylPhys_Interface/report_pdf_v4.py` | 114 | 7 | 6 | 6 |
| `MethylPhys_Interface/info_sphere_viewer.py` | 194 | 5 | 0 | 0 |
| `stage_t_selftare.py` | 54 | 4 | 0 | 0 |
| `stage_seq_values.py` | 140 | 7 | 0 | 0 |
| `dev_comp_methods.py` | 216 | 17 | 0 | 0 |
| `deconv_v2.py` | 74 | 3 | 0 | 0 |
| `sky_statistics.py` | 108 | 6 | 0 | 0 |
| `stage_q0_intake.py` | 174 | 7 | 0 | 0 |
| `stage_q_iam_a.py` | 174 | 9 | 0 | 0 |
| `stage_coupling.py` | 30 | 2 | 0 | 0 |
| `stage_provisional.py` | 98 | 7 | 0 | 0 |
| `serial_mode.py` | 67 | 6 | 0 | 0 |
| `stage_groupa.py` | 100 | 4 | 0 | 0 |
| `MethylPhys_Interface/report_globe_v4.py` | 447 | 23 | 0 | 0 |
| `MethylPhys_Interface/globe_viewer.py` | 145 | 7 | 0 | 0 |
| `stage_metaa_atlas.py` | 542 | 31 | 0 | 0 |
| `stage_q2_globe.py` | 145 | 14 | 0 | 0 |
| `stage_info_sphere.py` | 199 | 15 | 0 | 0 |
| `MethylPhys_Interface/reference_globes.py` | 155 | 8 | 0 | 0 |
| `MethylPhys_Interface/run_series.py` | 167 | 7 | 0 | 0 |
| `MethylPhys_Interface/run_draw.py` | 91 | 5 | 0 | 0 |
| `stage_seq_atlas.py` | 281 | 14 | 0 | 0 |

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

**One reference per reading.** An entropy is read only against a reference made the same way, or after a same-run tare. Met-A's healthy
reference is purified arrays read by Stage 1; IAM-Atlas v2.1 blends 21 sources, and on Met-A's sites its neutrophil reads 1.062 against
Met-A's reference although the values differ by 0.007 on average (`doors/data/DEV_REFERENCES_01/`): the entropy of a blend is higher than
the average entropy of its parts.

**Built from what, tested on what.** A reviewer's first question about any reference is whether it was tested on the data it was
built from. Every reference entry names both.

## Sex and the X chromosome, in every instrument

**Why sex matters to a methylation reading.** A woman has two X chromosomes and silences one in every cell by methylating it (X inactivation). At the switch sites of hundreds of X genes a man's single X reads near 0 and a woman's two copies average about 0.5. The Y exists only in men. A comparison between people of different sex can therefore light up the X without any disease. Each instrument handles this as follows (numbers computed from the chain's own files by `CANON/sex_chromosomes.py`):

| Instrument | X and Y |
|---|---|
| Stage 0, intake | Predicts the array's sex from its X and Y probe brightness (step 0.8: female when log2 Y minus log2 X is below -2.0). When a sex is declared and the array disagrees, the array is stopped and the reason printed: a swapped sample is caught before any reading. |
| Met-A | 6,000 neutrophil identity sites: 32 on the X, 0 on the Y. They were chosen where the six purified reference arrays, one of them a woman's, agree. Group-A's sealed test read 75 women and 188 men the same after the tare (-0.36 %, 1.2 standard errors). |
| C-Score | The same 6,000 sites (32 on the X, 0 on the Y), in genome order; it reads how departures cluster, read against the same healthy reference. |
| Noise index | 48,528 sites, 95 on the X and 15 on the Y. Even if every one of them read at maximum noise, the index could move by at most 0.0023 bits (0.23 % of its 1-bit scale). |
| Stage A, the blood mix | 963 markers, 0 on the X or Y. |
| IAM-Deconvolver | 600 markers, 0 on the X, 0 on the Y. |
| IAM-A | Each sample is read against its own predicted error, so no other person, of either sex, enters its reading. Aligned files (BAM) are counted on chromosomes 1 to 22 only. From a .pat file the loose share is counted on chromosomes 1 to 22, and the held error on every chromosome the file holds (the route IAM-A was commissioned on). With the X and Y left out, IAM-A moves by -0.32 % on average (at most -0.71 %), well inside the band, so the line and the band are kept (Decoder finding F37). |
| Group-A | Compares groups run in the same series; its sealed bar 2 is a women-against-men null (above). |
| IAM-Atlas v2.1 and the globes | 813,951 atlas sites, 14,759 on the X and 3 on the Y. The atlas's donors are women and men, so it is 2.5 to 3.6 times less sure on the X and Y than elsewhere (the noise globe shows it). Every change count on a globe is also given on the X and Y, and the page prints the sex Stage 0 read. When a tumor is drawn against its own normal tissue (same person, same sex), its change is counted with and without the X and Y: in a tumor an X change can be real biology. |

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
   the probes at background removed. Stage 1's own quality numbers are recorded beside Stage 0's, never over them, and never gate. Its
   control medians (hybridization, negative and non-polymorphic controls, and two medians over all BS Conversion I and II probes) are
   recorded and not graded: the two bisulfite medians are of different control sets, not a conversion efficiency, which is Stage 0's.
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

*Code:* `MethylPhys_Interface/run_sample.py` line 39 (16 lines) · arguments: start · calls: array run 1, bam run 1

### `run_sample.py::_assign_run_id`

Gives the run its identifier, `RUN-YYYYMMDD-NN`: the UTC date and the next number for that day, counted from the ledger the run is
about to append to. A run makes no claim and passes no bar; it is one execution on one specimen, and the identifier lets a reading be
found again.

*Code:* `MethylPhys_Interface/run_sample.py` line 59 (19 lines) · arguments: ledger_path · calls: array run 1, bam run 1

### `run_sample.py::_covariates`

Collects the phenotype and covariates for the run: a JSON object from `--covariates`, then each `--covariate KEY=VALUE` on top. They
are kept in the bundle and the custody record and are never printed in the report's prose. No stage reads them.

*Code:* `MethylPhys_Interface/run_sample.py` line 80 (16 lines) · arguments: a · calls: array run 1, bam run 1

### `run_sample.py::_sha12`

The first 12 hexadecimal characters of a file's SHA-256, used to fingerprint each frozen input in `_versions`. Returns nothing if the
file cannot be read.

*Code:* `MethylPhys_Interface/run_sample.py` line 98 (10 lines) · arguments: path · calls: array run 18, bam run 18

### `run_sample.py::_versions`

Records everything that can change a reading: the UTC time, the Python and methylprep versions, the chain's git commit and whether
the working copy had uncommitted changes, and a short hash and size of each frozen matrix and stage file. Two runs months apart can
be pooled only if these match.

*Code:* `MethylPhys_Interface/run_sample.py` line 110 (55 lines) · arguments: chain_dir · calls: array run 1, bam run 1 · files named: `Runtime Matrices/IAM_A_Positions/hg19_cpg_chrom_ranges.json`, `Runtime Matrices/IAM_A_Positions/iama_positions_v2.json`, `Runtime Matrices/Intake/intake_thresholds_v1.json`, `Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json`, `Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`, `Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`, `Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_2.json`, `Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json`, `Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json`, `compact_manifest.json`, `in compact_manifest.json`, `transfer_v21_final.json`

### `run_sample.py::_why`

Prints and keeps one stop's explanation in five parts (what was measured, the limit, where the limit comes from and whether it is
provisional, what it means, what to do), for every stop outside Stage 0's own gates: a refused specimen, the early intake stops, EPIC v2,
a missing manifest (exit code 3), a corrupt IDAT, Stage 1's platform refusal and the quarantine after calibration. The report shows the same
rows under "Why the chain stopped or flagged this sample". Release check V11 fails if any exit with code 2 or 3 has no explanation in its
own block.

*Code:* `MethylPhys_Interface/run_sample.py` line 167 (8 lines) · arguments: rec, check, outcome, measured, limit, source, means, todo, provisional · calls: not called on the traced runs

### `run_sample.py::_hash_id`

Replaces a typed identifier with the first 32 characters of its SHA-256. An identifier that already looks like a hash (16 or more
hexadecimal characters) is kept. The engine never stores a cleartext identifier.

*Code:* `MethylPhys_Interface/run_sample.py` line 177 (5 lines) · arguments: x · calls: array run 1, bam run 1

### `run_sample.py::_id_collides`

True when the typed sample id is the name of an IAM-Atlas v2.1 cell type or of a provisional column. Then the id is hashed in its own field only and the bundle's text is not searched for it, so a cell's name is never erased (found when the id 'megakaryocytes' erased the cell's name in provisional reference test 1).

*Code:* `MethylPhys_Interface/run_sample.py` line 184 (12 lines) · arguments: sid · calls: not called on the traced runs · files named: `import pyarrow.parquet`

### `run_sample.py::_redact`

Walks the whole bundle (every string, dictionary key and list) and replaces each whole-token occurrence of the typed identifier with
its hash, so neither the bundle nor the ledger carries the typed name. The printed report keeps the typed name for the operator.

*Code:* `MethylPhys_Interface/run_sample.py` line 197 (11 lines) · arguments: obj, typed, hid · calls: array run 4, bam run 4

### `run_sample.py::_redact.f`

The recursive step of `_redact`: a string gets the replacement, a dictionary and a list are walked, anything else is returned as is.

**`--dev-count-only` (development, to-do 14).** With `--pat`, Stage Q2's count (the held error and its opportunities) is recorded in the
bundle as `iam_a_count_only`, flagged not commissioned, even where Stage Q0 refuses an IAM-A reading (a cell line, a mouse sample: no healthy
reference). IAM-A itself stays refused. The firm-rate recount reads the held error this way, so its count is the one command's. The held error
does not depend on the genome; the loose share uses hg19 tiles and is marked meaningless on another genome (`--dev-count-genome`).

*Code:* `MethylPhys_Interface/run_sample.py` line 202 (5 lines) · arguments: v · calls: array run 865, bam run 359

### `run_sample.py::_metaa_atlas`

`--dev-metaa-atlas CELL` (development, not commissioned): maps the values by Stage T step 1 (the self-tare the commissioned Met-A reads), refuses
unless both designs were mapped, reads Met-A against CELL in IAM-Atlas v2.1 (`stage_metaa_atlas.read`) and, with `--metaa-atlas-ref-A` (at least
3 healthy samples of the same cell run the same way, never this one), tares it (`stage_metaa_atlas.tare`). Written under bundle
`development.metaa_atlas`; the reading, gauge and tare are untouched.

*Code:* `MethylPhys_Interface/run_sample.py` line 210 (27 lines) · arguments: a, b · calls: not called on the traced runs

### `run_sample.py::_genome_globe`

`--dev-genome-globe` with `--pat` or `--geo-pat` (development, not commissioned; the file must be hg19, Section 10b): counts the file's molecules into regions of `--genome-region` CpGs on the
autosomes (`stage_q2_globe.count_pat_regions`) and draws the sample against its own predicted held error (`self_globe`) and, with `--prior-pat`,
against that second file (`pair_globe`, named by `--prior-name`). Written under bundle `development.genome_globe`; with `--explorer-spec`,
the Globe Explorer file.

*Code:* `MethylPhys_Interface/run_sample.py` line 354 (21 lines) · arguments: a, cnt, sid · calls: not called on the traced runs

### `run_sample.py::_build`

`--build`: the chain builds a reference instead of reading a sample. `info-sphere` (development): the information sphere's layout from a
matrix of levels (sites x cell types), with its report (variance kept, sky covered and masked, distance agreement, edge ratio, and each named
site group's concentration against random groups) to `--bundle` and the layout to `--build-out`.

*Code:* `MethylPhys_Interface/run_sample.py` line 245 (107 lines) · arguments: a · calls: not called on the traced runs · constants read: `_ISV_ABOUT` · files named: `import pyarrow.parquet`, `sites.parquet`, `truth.json`

### `run_sample.py::_dev_run`

Runs the development readings named by `--dev-*` flags (direction, trace cell, foreign cell, brightness interval, NILC-e, atlas e,
B-cell reading, sky with the block-shuffle null, EPIC v2 through SeSAMe). Each writes its own record under `bundle["development"]`,
labeled DEVELOPMENT, and an error in one is caught and recorded. None of them changes the reading, the gauge or the tare.

*Code:* `MethylPhys_Interface/run_sample.py` line 377 (41 lines) · arguments: a, o, beta, intake · calls: array run 1, bam run 1

### `run_sample.py::main`

The orchestration described above, from parsing the command to writing the ledger row. Decides which stages run, stops the run with
exit code 2 or 3 at the stops listed, carries the same-run references (columns A and C) into the conductor, adds the v4 stages, and
hands everything to the report.

For a declared sorted cell (specimen "sorted ...", "isolated ..." or "purified ..." naming an atlas cell), the whole-sample readings (Met-A whole sample, the territories, the C-Score) are read against that cell's own atlas column instead of the blood panel's mixed shares; the bundle records it as `whole_sample_reference`.

*Code:* `MethylPhys_Interface/run_sample.py` line 420 (792 lines) · arguments: none · calls: array run 1, bam run 1 · constants read: `BIO`, `CONTROL_SITES`, `CONTROL_SITES_ABOUT`, `ENG`, `_ISV_ABOUT` · files named: `MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`, `OUT.csv`, `Runtime Matrices/Seq_Atlas/metaa_seq_reference_v1.parquet`, `_bundle.json`, `a wgbstools .pat / .pat.gz`, `metA_floors_v1_3.json`, `pip install -r Biological_Physics/MethylPhys/chain/requirements.txt`, `pyarrow.parquet`, `sequencing_values.csv`, `sites.parquet`, `sites_sequencing_key.parquet`

### `run_sample.py::main._panel`

Turns `--cell-panel` into the IAM-Deconvolver's panel: nothing for `blood` (its six commissioned cells), every IAM-Atlas v2.1 cell type for
`atlas`, or the named cells, refusing any name the atlas does not hold.

*Code:* `MethylPhys_Interface/run_sample.py` line 888 (8 lines) · arguments: none · calls: not called on the traced runs

### `run_sample.py::main._stopped`

True when an intake record's status begins with QUARANTINE. Intake checks it after every step, so a quarantine ends intake at its
cause. Without it a later step overwrote the status, and the verdict named a downstream symptom (seen 2026-09-23: an array-type
mismatch was reported as a detection failure).

*Code:* `MethylPhys_Interface/run_sample.py` line 653 (5 lines) · arguments: r · calls: array run 3

### `run_sample.py::main._ctx`

Builds the comparison record Stage 12b needs from a draw: the identifier hash, the array type and the calibration pipeline.

*Code:* `MethylPhys_Interface/run_sample.py` line 1141 (2 lines) · arguments: b, it, pipe · calls: not called on the traced runs

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
| 0.4 control probes | bisulfite conversion (C/(C+U) per matched BS Conversion I pair, each pair in its own channel, median over pairs), hybridization (the spread between the high and low concentration controls, with medium between them, green) and extension (green C and G against red A and T) controls, from the array's own control addresses | a failed hybridization or extension stops the array; a high/low swap with the spread intact is recorded, not a stop; a bisulfite conversion below the line stops the array | from `intake_thresholds_v2.json`, set 2026-10-21 (PROC-STAGE0-04) on 337 healthy whole-blood arrays of three laboratories (GSE318173, GSE315366, GSE184269): bisulfite at least **0.80** (the author's choice, the 2.5th percentile with each laboratory counted equally; healthy medians 0.93 to 0.97 by laboratory; stops 1 of the 337, GSM5583015 at 0.576, low on every pair in its own channel); hybridization spread at least 2.0 (healthy 2.69 and above) and extension balance 0.2 to 5.0 (healthy 0.45 to 1.22), stopping none of the 337 |
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

*Code:* `stage_0_intake.py` line 35 (8 lines) · arguments: none · calls: array run 1, bam run 1 · files named: `intake_thresholds_v2.json`

### `stage_0_intake.py::normalise_specimen`

Lowercases the specimen text, turns underscores into spaces and collapses repeated spaces, so `Whole_Blood` and `whole blood` are the
same specimen.

*Code:* `stage_0_intake.py` line 76 (2 lines) · arguments: specimen · calls: array run 5

### `stage_0_intake.py::atlas_cells`

The 79 cell names of IAM-Atlas v2.1 (its best-value columns), read once.

*Code:* `stage_0_intake.py` line 82 (7 lines) · arguments: none · calls: not called on the traced runs · files named: `import pyarrow.parquet`

### `stage_0_intake.py::atlas_cell_of`

An isolated, sorted or purified cell of any IAM-Atlas v2.1 type other than neutrophils (case-insensitive): its atlas name, else nothing. Such a
specimen is accepted at intake and read by Met-A against the atlas (development, not commissioned); the commissioned Met-A is not applied.

**450K arrays and the atlas readers (2026-10-22).** On a 450K array the conductor refuses the commissioned Met-A and C-Score (no 450K healthy
neutrophil reference; `conductor_v3.platform_refusal`), but `run_sample.py` still runs the atlas readers, which work by site name on IAM-Atlas
v2.1: the IAM-Deconvolver's shares, Met-A for each cell and the globe, each flagged development and each reporting how many of its sites the
array measured. The bundle records this under `atlas_readers_on_450k`.

*Code:* `stage_0_intake.py` line 90 (8 lines) · arguments: specimen · calls: not called on the traced runs · constants read: `ATLAS_CELL_WORDS`

### `stage_0_intake.py::is_plasma`

True for plasma cell-free DNA (plasma, cfDNA, cell-free DNA): accepted at intake since 2026-10-22 and read as a development specimen. The
IAM-Deconvolver splits it on IAM-Atlas v2.1's full panel (every cell and tissue type the atlas holds, since plasma DNA comes from cells all over
the body) and Met-A reads each source it finds against the atlas in the sample's own shares; the commissioned Met-A and C-Score read whole
blood and are not applied.

*Code:* `stage_0_intake.py` line 100 (3 lines) · arguments: specimen · calls: not called on the traced runs · constants read: `PLASMA_WORDS`

### `stage_0_intake.py::is_tissue`

True for a solid tissue, a tumor, a biopsy, or a specimen not named ('unknown', for blind runs). Such a sample is read on IAM-Atlas v2.1's full panel, flagged, as plasma is: the IAM-Deconvolver says what it is made of, and Met-A reads each cell it finds against the atlas in the sample's own shares. The commissioned readers read blood and are not applied. Bone marrow, a marrow aspirate and a blood-and-marrow mix are read on this route too, with the marrow progenitors kept in the mix (read and flag; it was refused at intake before).

*Code:* `stage_0_intake.py` line 105 (12 lines) · arguments: specimen · calls: not called on the traced runs · constants read: `TISSUE_WORDS`

### `stage_0_intake.py::specimen_refusal`

Returns nothing when the specimen is accepted, otherwise the plain refusal: which specimen, why the chain has no reference for it,
and what is accepted. Called by `run_sample.py` before any file is read.

*Code:* `stage_0_intake.py` line 118 (9 lines) · arguments: specimen · calls: array run 2 · constants read: `ACCEPTED_SPECIMENS`, `REFUSED_SPECIMEN_REASONS`

### `stage_0_intake.py::_infer_platform`

Turns the IDAT header's address count into a platform: below 800,000 is 450K, below 1,080,000 EPIC v1, otherwise EPIC v2.

*Code:* `stage_0_intake.py` line 139 (8 lines) · arguments: nsnps · calls: array run 2

### `stage_0_intake.py::read_idat_nsnps`

Reads only the start of the green IDAT: the four-byte `IDAT` mark, the field table, and field 1000 (the number of addresses). It never
decodes intensities. Returns the count and a status (OK, NOT_IDAT, NO_NSNPS_FIELD, FILE_NOT_FOUND or a read error).

*Code:* `stage_0_intake.py` line 151 (24 lines) · arguments: path · calls: array run 2

### `stage_0_intake.py::step_0_1_idat_arrival`

Step 0.1: checks the manifest's required fields and array type, that both channels exist, that each file is at least 1 MB, and that the
header's platform family agrees with the declared one; warns on a repeat intake within 24 hours. Stamps a run identifier and the
arrival time, and logs the result.

*Code:* `stage_0_intake.py` line 179 (89 lines) · arguments: manifest_entry, grn_path, red_path, intake_log_path · calls: array run 1 · constants read: `MIN_PLAUSIBLE_IDAT_BYTES`, `REQUIRED_MANIFEST_FIELDS`, `VALID_ARRAY_TYPES`

### `stage_0_intake.py::_log_intake`

Appends one row per intake step to the intake log, when one is given: time, run identifier, Sentrix ID, declared and detected array
type, specimen, declared sex and age, status and flags.

*Code:* `stage_0_intake.py` line 272 (10 lines) · arguments: intake_log_path, result · calls: array run 1

### `stage_0_intake.py::_is_duplicate_within_24h`

True when the intake log already holds the same Sentrix ID from the last 24 hours. Only a warning; the array still advances.

*Code:* `stage_0_intake.py` line 284 (21 lines) · arguments: intake_log_path, sentrix_id · calls: array run 1

### `stage_0_intake.py::_is_null_answer`

For storing questionnaire answers: a blank or "prefer not to say" becomes no answer; "No" is kept as a real answer.

*Code:* `stage_0_intake.py` line 367 (3 lines) · arguments: v · calls: not called on the traced runs · constants read: `_NULL_FOR_STORAGE`

### `stage_0_intake.py::_answer_present`

For the questionnaire's context modes: true only for an affirmative answer (not blank, not "prefer not to say", not No, none or not
applicable).

*Code:* `stage_0_intake.py` line 372 (5 lines) · arguments: v · calls: not called on the traced runs · constants read: `_NEGATIVE`, `_NULL_FOR_STORAGE`

### `stage_0_intake.py::_age_from_dob`

Age in decimal years from a date of birth and the intake date, rounded to 0.1.

*Code:* `stage_0_intake.py` line 379 (8 lines) · arguments: dob, on_date · calls: not called on the traced runs

### `stage_0_intake.py::parse_questionnaire`

Maps a digital questionnaire's answers to the stored covariate record: each field, the age from the date of birth, a smoking bin, and
a list of context modes. The record is stored with the manifest. No stage of the current chain reads these covariates or modes; the
stage names in its field table belong to an earlier engine.

*Code:* `stage_0_intake.py` line 389 (18 lines) · arguments: answers, intake_date · calls: not called on the traced runs · constants read: `COVARIATE_FIELDS`, `_CONTEXT_MODE_TRIGGERS`

### `stage_0_intake.py::_looks_like_cleartext_pii`

True when an identifier looks like a name rather than a hash: empty, containing a space or an @, shorter than 16 characters once
dashes and underscores are removed, or not purely letters and digits.

*Code:* `stage_0_intake.py` line 409 (9 lines) · arguments: patient_id · calls: array run 1

### `stage_0_intake.py::_write_manifest`

Writes the per-sample manifest as `patient_manifest_<run id>.json` in the manifest folder, when one is given.

*Code:* `stage_0_intake.py` line 420 (8 lines) · arguments: manifest_dir, record · calls: array run 1 · files named: `Write the immutable per-sample patient_manifest.json`

### `stage_0_intake.py::step_0_2_manifest_creation`

Step 0.2: builds the immutable manifest from the 0.1 record (and questionnaire answers, when given), quarantines a cleartext
identifier or a missing core field, and writes the file.

*Code:* `stage_0_intake.py` line 430 (40 lines) · arguments: step_0_1_result, questionnaire_answers, manifest_dir, intake_date · calls: array run 1

### `stage_0_intake.py::_sha256_file`

The SHA-256 of a whole file, read in 1 MB pieces.

*Code:* `stage_0_intake.py` line 478 (6 lines) · arguments: path · calls: array run 2

### `stage_0_intake.py::_prior_hashes_for_sentrix`

The pairs of file hashes earlier intakes recorded for the same Sentrix ID in the integrity log.

*Code:* `stage_0_intake.py` line 486 (14 lines) · arguments: integrity_log_path, sentrix_id · calls: array run 1

### `stage_0_intake.py::_log_integrity`

Appends one row to the integrity log (also used for the decision gate's verdict row).

*Code:* `stage_0_intake.py` line 502 (6 lines) · arguments: integrity_log_path, row · calls: array run 1

### `stage_0_intake.py::step_0_3_integrity_hash`

Step 0.3: hashes both files and compares them with earlier intakes of the same Sentrix ID. Identical bytes again is a re-transmission
and stops the run; the same Sentrix ID with different bytes is flagged as a fresh re-run and advances.

*Code:* `stage_0_intake.py` line 510 (37 lines) · arguments: record, grn_path, red_path, integrity_log_path · calls: array run 1

### `stage_0_intake.py::extract_control_probes`

Gets the control-probe medians by class from the IDAT pair through the QC hand-off (`stage_0_1_qc_handoff.decode_qc_inputs`), the
same decoder Stage 1 uses. The array's own controls only.

*Code:* `stage_0_intake.py` line 568 (7 lines) · arguments: grn_path, red_path, array_type · calls: not called on the traced runs

### `stage_0_intake.py::validate_control_probes`

Step 0.4's arithmetic. Bisulfite: `bs1 / (bs1 + bs2)` against 0.80, where the QC hand-off passes in the matched-pair conversion
efficiency and its complement under the keys `bisulfite_conversion_I_median` and `bisulfite_conversion_II_median` (so the ratio is the
efficiency itself; the key names are older than the pair method). Hybridization: the spread, the larger of the high- and low-concentration
medians over the smaller, at least 2, with the medium control between them. A high control reading where the low one should (and the
reverse) with the spread intact is recorded as a swapped order, not a stop: on two whole slides of GSE184269 High and Low trade values
exactly (21,304 and 20,040; 6,514 and 6,845) while Medium is unchanged (12,822 and 12,834), which a failed hybridization cannot do. Extension: the green-channel (C and G) extension median over the red-channel (A and T) median, between 0.2 and 5;
the record calls these "meth" and "unmeth", which they are not. The three limits are read from `intake_thresholds_v2.json`; while a bisulfite
limit is marked provisional in that file, a bisulfite failure alone is recorded and the array advances (the rule before 2026-10-21). Records the
raw medians the gates tested.

*Code:* `stage_0_intake.py` line 577 (53 lines) · arguments: control_summary · calls: array run 2 · constants read: `BS_CONVERSION_MIN`, `BS_THRESHOLD_CALIBRATED`, `EXTENSION_BALANCE_RANGE`, `HYB_HIGH_LOW_MIN_RATIO`

### `stage_0_intake.py::step_0_4_control_probe_validation`

Step 0.4: runs `validate_control_probes` on the hand-off's control summary and stamps the verdict and its numbers. With no decoder
available it marks the check deferred rather than inventing a pass.

*Code:* `stage_0_intake.py` line 632 (20 lines) · arguments: record, grn_path, red_path, control_summary · calls: array run 1

### `stage_0_intake.py::_normal_sf`

The normal survival function, 1 − Φ(z), from scipy when it is installed, otherwise from the complementary error function.

*Code:* `stage_0_intake.py` line 671 (9 lines) · arguments: z · calls: array run 2

### `stage_0_intake.py::compute_detection_p`

Per probe, the detection p-value 1 − Φ((I − μ_bg)/σ_bg), with μ_bg and σ_bg this array's negative-control background.

*Code:* `stage_0_intake.py` line 682 (6 lines) · arguments: probe_intensities, mu_bg, sigma_bg · calls: array run 2

### `stage_0_intake.py::validate_detection_p`

The share of probes with detection p ≤ 0.01, graded against the thresholds file: above 0.99 PASS, from 0.93 BORDERLINE, below FAIL.

*Code:* `stage_0_intake.py` line 690 (15 lines) · arguments: detection_p_array · calls: array run 2 · constants read: `DETECTION_BORDERLINE_FRACTION`, `DETECTION_PASS_FRACTION`, `DETECTION_P_THRESHOLD` · files named: `intake_thresholds_v1.json`

### `stage_0_intake.py::step_0_5_detection_pvalue_qc`

Step 0.5: computes and grades detection. Without intensities the check is deferred, and a deferred detection never advances.

*Code:* `stage_0_intake.py` line 707 (22 lines) · arguments: record, probe_intensities, neg_control_stats · calls: array run 1

### `stage_0_intake.py::validate_bead_count`

The share of probes with at least 3 beads: 0.995 or more PASS, otherwise a warning. Never stops an array.

*Code:* `stage_0_intake.py` line 739 (8 lines) · arguments: bead_counts, min_beads · calls: array run 1 · constants read: `BEAD_COUNT_MIN`, `BEAD_PASS_FRACTION`

### `stage_0_intake.py::step_0_6_bead_count_qc`

Step 0.6: grades the bead counts and flags a low count. Always advances.

*Code:* `stage_0_intake.py` line 749 (14 lines) · arguments: record, bead_counts · calls: array run 1

### `stage_0_intake.py::validate_call_rate`

The share of probes passing both detection and bead count, graded: 0.98 or more PASS, from 0.93 BORDERLINE, below FAIL.

*Code:* `stage_0_intake.py` line 773 (9 lines) · arguments: n_passing_both, n_total · calls: array run 2 · constants read: `CALL_RATE_BORDERLINE`, `CALL_RATE_PASS`

### `stage_0_intake.py::step_0_7_call_rate`

Step 0.7: combines the detection and bead masks probe by probe and grades the call rate. Without masks the check is deferred, and a
deferred call rate never advances.

*Code:* `stage_0_intake.py` line 784 (22 lines) · arguments: record, detection_pass_mask, bead_pass_mask · calls: array run 1

### `stage_0_intake.py::step_0_7b_platform_coverage`

Step 0.7b: stamps the platform tag. On a 450K array it stops below 80 % of the reference sites; on EPIC it records `NA_EPIC` and passes.

*Code:* `stage_0_intake.py` line 815 (25 lines) · arguments: record, reference_cpg_coverage · calls: array run 2 · constants read: `HM450_COVERAGE_MIN`

### `stage_0_intake.py::predict_sex`

Female when the median log2 Y intensity minus the median log2 X intensity is below −2, otherwise male (minfi's getSex rule).

*Code:* `stage_0_intake.py` line 849 (2 lines) · arguments: log2_x_median, log2_y_median, cutoff · calls: array run 1 · constants read: `SEX_GETSEX_CUTOFF`

### `stage_0_intake.py::validate_sex`

Compares the predicted sex with a declared one. No declared sex: the array's sex is recorded and nothing is compared. A mismatch stops
the array.

*Code:* `stage_0_intake.py` line 853 (7 lines) · arguments: log2_x_median, log2_y_median, declared_sex · calls: array run 1

### `stage_0_intake.py::step_0_8_sex_check`

Step 0.8: runs the sex check on the hand-off's X and Y medians and stamps the result.

*Code:* `stage_0_intake.py` line 862 (15 lines) · arguments: record, sex_intensities · calls: array run 1

### `stage_0_intake.py::_fmt`

Formats a measured number for an explanation, or says "not measured".

*Code:* `stage_0_intake.py` line 891 (2 lines) · arguments: x, nd · calls: array run 2

### `stage_0_intake.py::explain_stage0`

Builds the five-part explanation of every check that stopped, penalized or deferred the array (measured on this array, the limit, the
limit's source and whether it is provisional, what it means, what to do). Changes no decision.

*Code:* `stage_0_intake.py` line 893 (60 lines) · arguments: record, hard_fail, borderline, deferred · calls: array run 2 · constants read: `BEAD_COUNT_MIN`, `BEAD_PASS_FRACTION`, `BS_CONVERSION_MIN`, `BS_THRESHOLD_CALIBRATED`, `CALL_RATE_BORDERLINE`, `CALL_RATE_PASS`, `DETECTION_BORDERLINE_FRACTION`, `DETECTION_PASS_FRACTION`, `EXTENSION_BALANCE_RANGE`, `HM450_COVERAGE_MIN`, `HYB_HIGH_LOW_MIN_RATIO`, `SEX_GETSEX_CUTOFF`, `_TH_SRC` · files named: `Runtime Matrices/Intake/intake_thresholds_v2.json`

### `stage_0_intake.py::explain_stage0.add`

Adds one explanation to the list.

*Code:* `stage_0_intake.py` line 895 (2 lines) · arguments: check, outcome, measured, limit, source, provisional, means, todo · calls: array run 2

### `stage_0_intake.py::step_0_9_decision_gate`

Step 0.9: sorts every step's outcome into hard failures, penalties and deferrals and gives the verdict (QUARANTINE, PROCEED_WITH_PENALTY
or PROCEED). A deferred detection or call rate is a hard failure; a deferred bead count is recorded.
Attaches the explanations and logs the verdict.

*Code:* `stage_0_intake.py` line 955 (80 lines) · arguments: record, verdict_log_path · calls: array run 2

## 3. The QC hand-off: `stage_0_1_qc_handoff.py`

**What it is for.** Stage 0's gates 0.4 to 0.8 need intensities, and intensities come from decoding the IDAT files. This module decodes
the pair just far enough to give those gates real numbers from the array itself, using the same decoder Stage 1 uses (methylprep) and
the control addresses of the array's own manifest. Nothing from any other sample enters.

**Label: standard method** (Illumina's control probes, read by design type the way the minfi package reads them).

**What it returns.**

- **Control summary.** Bisulfite conversion: for each matched BS Conversion I pair (C1 with U1, C2 with U2, ...), the converted signal
  over converted plus unconverted, C/(C+U), and the median over the pairs. Each pair is read in its own channel, the one where the pair's
  total C + U is brighter: on EPIC v1, pairs 1 and 2 are green probes and 3 to 5 red. Until 2026-10-20 every pair was read in green, which
  put three of five at background on both probes and gave every healthy array about 0.7 (on 61 healthy arrays, mostly GSE318173: median
  0.738 as written, 0.970 read correctly, range 0.927 to 0.978; `doors/data/PROC_STAGE0_04/bs_channel_explore.py`). Read correctly, other
  laboratories differ: GSE315366 0.899 to 0.941, GSE184269 0.576 to 0.975 (`doors/data/GROUPA_COMMISSIONING/diag_ctrl.py`). The sum C + U is used to find the channel because
  it stays bright whether conversion worked (C bright) or failed (U bright). Each pair's channel and contrast are recorded. Hybridization:
  the medians of the high-, medium- and low-concentration controls (green). Extension: the median of the C and G controls (green) and of the A and T controls (red).
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

### `stage_0_1_qc_handoff.py::_manifest`

Loads the array type's manifest from methylprep once and keeps it for the rest of the run.

*Code:* `stage_0_1_qc_handoff.py` line 26 (9 lines) · arguments: array_type · calls: array run 2 · constants read: `_MANIFEST`

### `stage_0_1_qc_handoff.py::_channel_means`

Reads one IDAT file (gzipped or plain) with methylprep and returns the mean intensity at every address and, when asked, the bead count.

*Code:* `stage_0_1_qc_handoff.py` line 37 (11 lines) · arguments: path, channel, want_beads · calls: array run 2

### `stage_0_1_qc_handoff.py::_median`

The median intensity over a list of control addresses, ignoring addresses the file lacks; nothing if none are present.

*Code:* `stage_0_1_qc_handoff.py` line 50 (3 lines) · arguments: series, addresses · calls: array run 4

### `stage_0_1_qc_handoff.py::decode_qc_inputs`

Builds everything Stage 0's gates read, as listed above, from the two files and the manifest. Returns the probe identifiers in the same
order as the intensities and bead counts, so Stage 1's call rate can line them up later.

*Code:* `stage_0_1_qc_handoff.py` line 55 (139 lines) · arguments: grn_path, red_path, array_type · calls: array run 1

### `stage_0_1_qc_handoff.py::decode_qc_inputs.ctl_addr`

The manifest addresses of one control type, optionally narrowed to extended types containing (or not containing) a word.

*Code:* `stage_0_1_qc_handoff.py` line 62 (9 lines) · arguments: control_type, contains, excludes · calls: array run 5

### `stage_0_1_qc_handoff.py::decode_qc_inputs._at`

The intensity at each probe's address, with a gap where the probe has no such address.

*Code:* `stage_0_1_qc_handoff.py` line 149 (6 lines) · arguments: series, addr · calls: array run 8

### `stage_0_1_qc_handoff.py::decode_qc_inputs.sex_median`

log2 of the median total intensity over the X (or Y) probes the array measured, and how many probes that was.

*Code:* `stage_0_1_qc_handoff.py` line 180 (5 lines) · arguments: which · calls: array run 2

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

### `stage_1_idat_calibration.py::_check_pins`

Refuses to run unless methylprep is version 1.7.1 and every cached manifest matches its pinned SHA-256 (or the development override is
set). Called before calibration and again after methylprep fetches any missing manifest.

*Code:* `stage_1_idat_calibration.py` line 38 (11 lines) · arguments: none · calls: array run 2 · constants read: `MANIFEST_SHA256`, `METHYLPREP_VERSION`

### `stage_1_idat_calibration.py::_ensure_methylprep`

Imports methylprep, installing the pinned version into the running Python once if it is missing, and explains what to do when the
install or import fails.

*Code:* `stage_1_idat_calibration.py` line 49 (43 lines) · arguments: none · calls: array run 1 · constants read: `METHYLPREP_VERSION` · files named: `pip install -r Biological_Physics/MethylPhys/chain/requirements.txt`

### `stage_1_idat_calibration.py::_detect_array_type`

Reads the green IDAT with methylprep and maps its address count to methylprep's array type; also returns the barcode.

*Code:* `stage_1_idat_calibration.py` line 94 (10 lines) · arguments: grn_path · calls: array run 1

### `stage_1_idat_calibration.py::calibrate_idat_to_beta`

Copies the pair into a private temporary folder with a one-row sample sheet (so methylprep processes this array alone), runs noob with
poobah detection and the control and SNP probes exported, keeps the cg probes with poobah p ≤ 0.05, and returns the betas with the
record described above. The temporary folder is always deleted.

*Code:* `stage_1_idat_calibration.py` line 106 (106 lines) · arguments: grn_path, red_path, array_type, verbose, mask_detection, return_mask, return_snp · calls: array run 1 · files named: `_processed.csv`, `control_probes.pkl`, `samplesheet.csv`, `the tag beta_scale_maps_v1.json`

### `stage_1_idat_calibration.py::calibrate_idat_to_beta.med`

The median of one control type in one channel, from methylprep's exported control table.

*Code:* `stage_1_idat_calibration.py` line 176 (2 lines) · arguments: t, col · calls: array run 7

### `stage_1_idat_calibration.py::_stage_idat`

Copies an IDAT into the staging folder under the name methylprep expects, unzipping it if it is gzipped.

*Code:* `stage_1_idat_calibration.py` line 214 (8 lines) · arguments: src, dst · calls: array run 2

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

### Why these sites: Met-A's site rule

Met-A reads each cell at that cell's own identity sites. The rule is fixed (`stage_m_met_a.py`; for every atlas cell,
`stage_metaa_atlas.identity_sites`): healthy spread at most 0.05, healthy level 0.75 to 0.95 (held methylated) or 0.05 to 0.25 (held
unmethylated), at most 3,000 a side, the tightest first. Each part of it is one of three kinds:

- **Held sites. Label: from IAM's physics.** Disorder H(β) is lowest where a mark is held and highest at 0.5. A loss of hold moves a held
  site toward 0.5, so its disorder rises; a site already near 0.5 is at the ceiling, where a loss cannot show (the ceiling caveat above). So
  Met-A reads the sites a cell holds, on both sides: held methylated and held unmethylated, because a loss moves either toward 0.5.
- **A small healthy spread. Label: from the reading itself.** Where healthy people differ, their ordinary differences look like a loss. Sites
  where healthy people agree keep the healthy reading tight, so a loss stands out against it.
- **The window's ends (0.95 and 0.05) and the count (3,000 a side). Label: operational, stated.** Neither is derived. The ends are part of the
  rule as first written, and no reason for them is recorded; the count gives a stable reading. Both are tested rather than defended; the ends
  against windows reaching 1 and 0 (to-do 39.8).

**Tested: 6,000 identity sites against every atlas site (DEV-EACH-CELL-01 exploration 4).** Reading every site, each counted by the shift a
loss of the cell would make there, is far more sensitive to a loss across a cell's whole pattern (45 to 187 of its own errors, against 6 to 30
for the identity sites) but moves every cell alike, so it cannot say which cell is losing its pattern. The identity sites say which cell (other
cells move 15 to 26 % as much) when the loss sits at that cell's own sites. A loss across a cell's whole pattern cannot be put on one cell by
either, because most held sites are held by its relatives too. So Met-A for each cell keeps the identity sites; sites specific to each cell
are tested next (to-do 39.9).

**Each cell has its own set.** On the atlas, every cell's sites come from its own column by the same rule, so the CD4 T cells' sites differ
from the neutrophils'. Sets overlap where cells hold the same sites, and that overlap is why a loss in one cell moves its relatives a little:
in a whole blood, up to 26 % of the planted cell's move (DEV-EACH-CELL-01 synthetic test 2).

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

### `conductor_v3.py::ref`

Loads the neutrophil reference (`neutrophil_reference_v1_2.json`) once: the 6,000 sites in genomic order, the healthy mean and shrunk SD
of H at each, and the C-Score's block length and healthy baseline.

*Code:* `conductor_v3.py` line 62 (4 lines) · arguments: none · calls: array run 4 · constants read: `RM`, `_REF` · files named: `neutrophil_reference_v1_2.json`

### `conductor_v3.py::noise_sites`

Loads the 48,528 noise sites once.

*Code:* `conductor_v3.py` line 70 (6 lines) · arguments: none · calls: array run 1 · constants read: `RM`, `_NS` · files named: `Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json`, `noise_sites_EPIC_v1.json`

### `conductor_v3.py::noise_gate`

Loads N_max and its rule once.

*Code:* `conductor_v3.py` line 78 (5 lines) · arguments: none · calls: array run 1 · constants read: `RM`, `_NG` · files named: `Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json`, `noise_gate_EPIC_v1.json`

### `conductor_v3.py::noise_index`

N = mean H(β) over the noise sites this array measured, or nothing when fewer than 90 % were measured; with the counts.

*Code:* `conductor_v3.py` line 84 (6 lines) · arguments: beta · calls: array run 1 · constants read: `H`, `MIN_NOISE_FRACTION`

### `conductor_v3.py::_bc`

Loads the blood composition matrix (`blood_composition_EPIC_v1.json`) once: the 963 markers and 8 group means at each, and the 8 groups'
healthy means at the 6,000 identity sites.

*Code:* `conductor_v3.py` line 92 (4 lines) · arguments: none · calls: array run 4 · constants read: `RM`, `_BC` · files named: `blood_composition_EPIC_v1.json`

### `conductor_v3.py::min_markers`

The number of markers Stage A needs: 90 % of 963, rounded up (867).

*Code:* `conductor_v3.py` line 97 (2 lines) · arguments: none · calls: array run 2 · constants read: `MIN_MARKER_FRACTION`

### `conductor_v3.py::stage_a_composition`

Stage A: solves the 8 shares by non-negative least squares on the measured markers and scales them to sum to 1; records the fit's mean
absolute residual. Refuses, with the count, below 867 markers.
How it agrees with the chain's other cell-mix reader, Stage D21 (the IAM-Deconvolver): on Group-A's 278 healthy whole bloods (GSE318173), the
neutrophil shares correlate at 0.960; the Deconvolver reads 0.021 lower on average, and 95 % of arrays agree within -0.113 to +0.071
(`doors/data/GROUPA_COMMISSIONING/two_readers_01_summary.txt`, 2026-10-22). Stage A's share sets Met-A's whole-blood expectation; the
Deconvolver's shares are the ones printed on the report.

*Code:* `conductor_v3.py` line 100 (15 lines) · arguments: beta · calls: array run 1

### `conductor_v3.py::_clustering`

The C-Score's statistic on a departure map: variance of block means of 10 sites (× √10) over the site variance; nothing with fewer than 10
blocks.

*Code:* `conductor_v3.py` line 116 (5 lines) · arguments: z, w · calls: array run 1

### `conductor_v3.py::_state`

Normal from 0.95 to 1.05, otherwise above or below Normal.

*Code:* `conductor_v3.py` line 122 (2 lines) · arguments: A · calls: array run 1

### `conductor_v3.py::stage_m_blood`

Stage M in whole blood: the expectation e for this blood's mix, Met-A = mean H(β) / mean H(e) over the measured identity sites, the shift a
1 % loss would make at this neutrophil share, the ceiling check and the departure map. Withholds the reading, with the reason, when there is
no mix, no neutrophils, or under 90 % of the sites.

*Code:* `conductor_v3.py` line 125 (26 lines) · arguments: beta, comp · calls: array run 1 · constants read: `BUILD`, `H`

### `conductor_v3.py::_ceiling`

The mean β of the neutrophil's methylated identity sites, and whether it has fallen below 0.5, past which Met-A no longer rises with loss.

*Code:* `conductor_v3.py` line 152 (7 lines) · arguments: x, mu · calls: array run 1

### `conductor_v3.py::stage_m_isolated`

Stage M on isolated neutrophils: Met-A against the healthy reference (`stage_m_met_a.read`), the 1 % shift, the ceiling check and the
departure map against the healthy neutrophil. Keeps the untared state as `state_own_floor` and marks the reading untared until Stage T.

*Code:* `conductor_v3.py` line 160 (12 lines) · arguments: beta · calls: not called on the traced runs · constants read: `H`

### `conductor_v3.py::stage_mc_cscore`

Stage MC: the C-Score from the departure map, divided by the healthy baseline, with the share of sites more than 3 SD out. With no map,
no C-Score and the reason.

*Code:* `conductor_v3.py` line 173 (12 lines) · arguments: z · calls: array run 1 · files named: `divided by the healthy median clustering of neutrophil_reference_v1_2.json`

### `conductor_v3.py::_ref_records`

Turns the same-run references into records (A, neutrophil share, N, C), skipping empty values and the sample itself.

*Code:* `conductor_v3.py` line 186 (13 lines) · arguments: ref_A, sample_id · calls: array run 2

### `conductor_v3.py::stage_t_tare`

Stage T step 2: the median tare of Met-A, the reference spread and the detection limit; untared, with the reason, under 3 references.

*Code:* `conductor_v3.py` line 200 (16 lines) · arguments: A, ref_A, shift_1pct, sample_id · calls: array run 1 · constants read: `MIN_REFS`

### `conductor_v3.py::stage_t_cscore`

The C-Score's same-run tare and its call against the band (above 1.152 a clustered departure).

*Code:* `conductor_v3.py` line 219 (18 lines) · arguments: cs, ref_A, sample_id · calls: array run 1 · constants read: `CSCORE_COMMISSIONING`, `CSCORE_DEV_BAND`, `MIN_REFS`

### `conductor_v3.py::stage_t_selftare_ii`

Stage T step 1: runs `dev_stages.selftare_map` on the array and returns the mapped values with the anchors and maps; unmapped values and a
NOT_RUN record if the map file is missing.

*Code:* `conductor_v3.py` line 238 (9 lines) · arguments: beta · calls: array run 1

### `conductor_v3.py::platform_refusal`

Nothing for an EPIC v1 array; otherwise the refusal: another array type from Stage 0, EPIC v2 probe names, or 700,000 probes or fewer.

*Code:* `conductor_v3.py` line 248 (10 lines) · arguments: beta, array_type · calls: array run 1 · constants read: `ACCEPTED_ARRAY_TYPES`

### `conductor_v3.py::plant_neutrophil_loss`

A constructed 1 % loss for tests, Group-A's sealed rule: at the neutrophil sites, beta + f_neu x 1 % x (0.5 - mu_NEU), f_neu the array's own
neutrophil fraction. Used by the series runner's `--plant neutrophil`; never by a reading.

*Code:* `conductor_v3.py` line 311 (6 lines) · arguments: beta, f_neu, loss · calls: not called on the traced runs

### `conductor_v3.py::run_neutrophil`

The whole conductor, in the order above. Returns the array reading's records (mix, Met-A, tare, C-Score) and the departure map for the sky.

*Code:* `conductor_v3.py` line 259 (50 lines) · arguments: beta, specimen, ref_A, array_type, sample_id · calls: array run 1 · constants read: `BUILD`, `CSCORE_COMMISSIONING`, `METAA_COMMISSIONING`, `MIN_NOISE_FRACTION`

### `dev_stages.py::_json`

Loads a file from `Runtime Matrices/Development` once and keeps it; the self-tare matrix's old name reads the moved file
(`stage_t_selftare._json`), so older scripts still run.

*Code:* `dev_stages.py` line 38 (7 lines) · arguments: name · calls: array run 1 · constants read: `DEVDIR`, `_C` · files named: `dev_selftare_typeII_EPIC_v1.json`

### `stage_t_selftare.py::_json`

Loads the self-tare matrix once, `Runtime Matrices/Stage_T/selftare_typeII_EPIC_v1.json`: the four fixed-site sets, each site's probe design,
and the reference arrays' anchors. Moved unchanged from `Runtime Matrices/Development/` on 2026-10-22 (finding F23; the same sha256) and
pinned by the release check as a frozen input (F1): Stage T step 1 runs on every reading, so its matrix is frozen like every other reference.

*Code:* `stage_t_selftare.py` line 15 (4 lines) · arguments: name · calls: not called on the traced runs · constants read: `MATRIX`, `_C` · files named: `selftare_typeII_EPIC_v1.json`

### `stage_seq_values.py::values_from_bigwig`

Reads a public pair of GRCh38 bigWig tracks (per-CpG methylation level and read coverage, the form consortia such as BLUEPRINT release whole-genome bisulfite calls in) at each atlas site's GRCh38 position (the C of the CpG, or the next base where the caller reports the reverse strand), giving a value at each site for every reader but IAM-A, which needs molecules. Used by `run_sample.py --dev-seq-beta CALL.bw,COV.bw`.

*Code:* `stage_seq_values.py` line 118 (22 lines) · arguments: call_bw, cov_bw, min_calls · calls: not called on the traced runs · constants read: `KEY38` · files named: `atlas/v2.1/sites_key_hg38.parquet`

### `stage_seq_values.py::key`

Loads the frozen site key (`atlas/v2.1/sites_sequencing_key.parquet`: each IAM-Atlas v2.1 site's EPIC v1 probe name, hg19 position and CpG index
in the .pat files' numbering) once, sorted by CpG index. Development (to-do step 40).

*Code:* `stage_seq_values.py` line 23 (5 lines) · arguments: none · calls: not called on the traced runs · constants read: `KEY`

### `stage_seq_values.py::construct_draw`

TESTS ONLY: a constructed blood draw with a known answer, built by the chain from spent values (`run_sample.py --build constructed-draw
--build-input BLOOD_VALUES --build-mix TISSUE:SHARE,... --build-depth BUFFY,PLASMA --build-out DIR`). Buffy coat: the blood's own levels;
plasma: that blood in the remaining share plus each named atlas tissue's best value at its share. Each half is drawn as k of n reads per atlas
site (seeded) and written as a `.beta` count file with a truth record. With `--dev-plant CELL:LOSS` a loss of that cell's pattern is planted in the blood's levels before the reads are counted, scaled by
the cell's six-cell share as the one command scales a plant, so the counting noise falls on the planted levels as it would on a real loss. Made because the open matched set (GSE186888) is reserved unread for
the one-draw test that counts (author 2026-10-22).

*Code:* `stage_seq_values.py` line 85 (29 lines) · arguments: blood, add, depth_buffy, depth_plasma, out_dir, seed, plant · calls: not called on the traced runs · constants read: `HERE`, `HG19_CPGS` · files named: `truth.json`

### `stage_seq_values.py::values_from_beta`

A value at each atlas site from a `.beta` file, the standard binary per-CpG count file (methylated reads and all reads at every CpG, hg19 CpG
numbering, the numbering of the frozen site key); a file of another length is refused with both counts. Per-site counts only, so every reader
runs but IAM-A, which needs molecules (`run_sample.py --dev-seq-beta`, `run_draw.py --buffy-seqbeta / --plasma-seqbeta`). Checked on a
constructed file (the example blood GSM7508429 drawn at 30 reads per site): the IAM-Deconvolver's six shares within 1.2 points of the array's.

*Code:* `stage_seq_values.py` line 69 (14 lines) · arguments: path, min_calls · calls: not called on the traced runs · constants read: `FLAG`, `HG19_CPGS`

### `stage_seq_values.py::values_from_pat`

A value at each atlas site from sequencing molecules, for `--dev-seq-values MIN_CALLS`: every call at the exact CpG the site names is counted,
and the value is methylated calls over calls, kept only where there are at least MIN_CALLS calls; the summary gives how many sites have a value
and the calls per site, so a reading can say what depth it needed. Every line counts (IAM-A's own count keeps its line split). Development.

*Code:* `stage_seq_values.py` line 29 (25 lines) · arguments: path, min_calls, keep · calls: not called on the traced runs · constants read: `FLAG`

### `stage_seq_values.py::values_from_pat.flush`

Adds the buffered calls to their sites in one step, matching each call's CpG index to the key exactly; calls at CpGs that are not atlas sites
are dropped.

*Code:* `stage_seq_values.py` line 32 (5 lines) · arguments: none · calls: not called on the traced runs

### `stage_seq_values.py::thin_pat`

Tests only, for `--dev-thin FRACTION`: a seeded thinned copy of a .pat file, each molecule kept with probability FRACTION (a line counting w
molecules keeps a binomial draw from w), so a deep sample stands in for a shallower one; it returns the molecules before and after.

*Code:* `stage_seq_values.py` line 55 (11 lines) · arguments: path, frac, dest, seed · calls: not called on the traced runs

### `stage_t_selftare.py::anchors`

This array's mean β over each of the four fixed-site sets (type I low and high, type II low and high), with how many it measured.

*Code:* `stage_t_selftare.py` line 21 (7 lines) · arguments: beta, st · calls: not called on the traced runs

### `stage_t_selftare.py::selftare_map`

Stage T step 1, as described above: maps each design's values by β' = L_r + (β − L)(U_r − L_r)/(U − L) at the sites whose design is
recorded and returns the maps' anchors and slopes. Moved unchanged from `dev_stages.py` on 2026-10-22 (finding F23); `dev_stages` keeps
both names for older scripts.

*Code:* `stage_t_selftare.py` line 42 (12 lines) · arguments: beta, st · calls: not called on the traced runs · files named: `Runtime Matrices/Stage_T/selftare_typeII_EPIC_v1.json`

### `stage_m_met_a.py::derived_baseline`

Met-A's healthy mark derived from the cell's own held errors (development, DEV-METAA-HEALTHY-01 step 3): at each identity site the healthy level is 1 - eps_loss where the cell holds the mark and eps_gain where it is held absent, put on the array scale through IAM-Atlas v2.1's measured transfer, and the mark is the mean entropy over the sites. Nothing is fitted; the array's own measurement blur is not included.

*Code:* `stage_m_met_a.py` line 89 (13 lines) · arguments: held, eps_loss, eps_gain · calls: not called on the traced runs · constants read: `TRANSFER`

### `stage_m_met_a.py::_floors`

Loads `metA_floors_v1_3.json` (the 6,000 identity sites and the healthy reference H_ref) once.

*Code:* `stage_m_met_a.py` line 29 (4 lines) · arguments: none · calls: array run 1 · constants read: `FLOORS`, `_F`

### `stage_m_met_a.py::_H`

The entropy of a beta value in bits, with β clipped to [10⁻⁶, 1 − 10⁻⁶] so a value of exactly 0 or 1 gives 0, not an error.

*Code:* `stage_m_met_a.py` line 34 (3 lines) · arguments: b · calls: array run 7

### `stage_m_met_a.py::platform_of`

The platform from the values themselves: EPIC v2 when any probe name carries the v2 design suffix; else EPIC when more than 700,000 probes,
else 450K.

*Code:* `stage_m_met_a.py` line 40 (6 lines) · arguments: beta · calls: array run 2 · constants read: `_V2_NAME`

### `stage_m_met_a.py::available`

The cell types with a healthy reference on each platform that are in scope (neutrophils).

*Code:* `stage_m_met_a.py` line 47 (3 lines) · arguments: platform · calls: not called on the traced runs · constants read: `CELLS_IN_SCOPE`

### `stage_m_met_a.py::precision`

The held-out readings of the reference arrays for one cell type: how many, the share in Normal, the SD and the range.

*Code:* `stage_m_met_a.py` line 51 (6 lines) · arguments: platform, cell · calls: not called on the traced runs · constants read: `LOO`, `NORMAL`

### `stage_m_met_a.py::read`

Met-A on a purified specimen of one cell type: refuses a cell outside scope, a platform without a reference, or under 90 % of the sites;
otherwise A = mean H(β) / H_ref with its state and the held-out precision.

*Code:* `stage_m_met_a.py` line 58 (19 lines) · arguments: beta, cell, platform, specimen · calls: not called on the traced runs · constants read: `CELLS_IN_SCOPE`, `NORMAL`, `SITE_COVERAGE_MIN`

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

### `stage_v4.py::_ex`

The five-part explanation of a stop, for the cell shares and IAM-A (the same parts as `run_sample._why`).

*Code:* `stage_v4.py` line 23 (2 lines) · arguments: check, outcome, measured, limit, source, means, todo, provisional · calls: not called on the traced runs

### `stage_v4.py::atlas_panel`

Every cell type of IAM-Atlas v2.1, read from the atlas's own columns, for `--cell-panel atlas`: the IAM-Deconvolver splits the sample into all
of them instead of the six blood cells. Development: the Deconvolver is commissioned on the six-cell panel only, so every share from another
panel is flagged.

*Code:* `stage_v4.py` line 28 (4 lines) · arguments: none · calls: not called on the traced runs · files named: `import pyarrow.parquet`

### `stage_v4.py::stage_d21`

Stage D21: loads the panel's reference and markers once, counts the markers measured, withholds below half, otherwise runs the
Deconvolver and labels each cell's share commissioned or development.

Each share whose best fit lies outside its own 95 % range is marked `resolved: false` ("not resolved from its close relatives in the atlas").
On the six-cell panel this does not happen; on the full panel close relatives (neutrophils, eosinophils and basophils; monocytes, macrophages
and microglia; T-cell subsets) can trade a share between them, so the resampled range and the single best fit disagree (development).

*Code:* `stage_v4.py` line 33 (27 lines) · arguments: beta, panel, sequencing · calls: array run 1 · constants read: `MIN_MARKER_SHARE`, `SEQ_READ_MARKERS`, `_REF`

### `stage_v4.py::count_sequencing`

Counts a `.pat` file (gzipped or plain) on the hg19 tiles, or an aligned BAM against its reference FASTA, with Stage Q2's one counter.

*Code:* `stage_v4.py` line 76 (5 lines) · arguments: pat, bam, ref_fasta, workers, assembly · calls: bam run 1

### `stage_v4.py::counting_error`

IAM-A's counting error, stated by every reading (finding F38, 2026-10-22): the held error is holes counted among held-error chances, so its
counting error is binomial, √(ε(1 − ε)/n) for n chances, and A carries the same relative error. The record gives the chances, A's 95 % range
from counting alone, and that range's width as a share of the Normal band's width. It is recorded only: it never refuses a reading and never
changes A or its state, so a shallow sample, plasma included, is read and says itself how much of the band its counting uses (author: "let the
instrument tell us...not our assumptions"). For example, at a held error of 0.02: ten million chances use 3 % of the band's width, one million
9 %, a hundred thousand 30 %, ten thousand 94 % (`stage_v4.counting_error` on those counts).

*Code:* `stage_v4.py` line 82 (12 lines) · arguments: rec, A, B · calls: not called on the traced runs

### `stage_v4.py::stage_q2`

Stage Q2 as above: the refusals, the shape check, the reading A with its predicted value, and the commissioned or development label.

The reading's `count` also carries the nearest-neighbor coupling (`coupling`, `coupling_err_share`, `coupling_pairs`), recorded beside
the held error, never used in A.

*Code:* `stage_v4.py` line 95 (27 lines) · arguments: rec, cell, chemistry, source · calls: bam run 1 · constants read: `CD8_TYPES` · files named: `Runtime Matrices/IAM_A_SelfRef/shape_reference_v1.json`

### `stage_v4.py::_compress`

Takes every other bit of an integer: the step of the HEALPix algorithm that splits a NESTED pixel index into its x and y coordinates.

*Code:* `stage_v4.py` line 129 (3 lines) · arguments: v · calls: array run 6

### `stage_v4.py::nest_pix2ang`

HEALPix NESTED pixel centers to colatitude and longitude.

*Code:* `stage_v4.py` line 132 (10 lines) · arguments: nside, ipix · calls: array run 3 · constants read: `JPLL`, `JRLL`

### `stage_v4.py::cscore_sky`

Draws the C-Score's departures on the sphere in a Mollweide view and returns a summary (NSIDE, sites, pixels drawn, the share of
departures beyond 3). Without a departure map it draws nothing and says why.

*Code:* `stage_v4.py` line 143 (38 lines) · arguments: z, out_png, title, vlim, readings · calls: array run 1

### `stage_v4.py::png_data_uri`

Embeds a figure in the HTML report as a data URI.

*Code:* `stage_v4.py` line 182 (2 lines) · arguments: path · calls: not called on the traced runs

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
are not the average cell). The 2.5th and 97.5th percentiles give the interval. On held-out data they covered the truth 78 % of the time,
below the 95 % their name implies: they are information, not calibrated intervals (see below).

**Built from, tested on.** Built from IAM-Atlas v2.1 alone. Tested by planted mixtures drawn from the atlas's own spread with a laboratory
shift (the file's `planted` test), then by the sealed real test: 200 mixtures of held-out purified arrays from E-MTAB-2145, a laboratory
the atlas never saw. Commissioned 2026-10-17 on the shares of neutrophils, monocytes, CD4 T and CD8 T cells: mean share error 0.016, every
number reproduced from a fresh clone (on AWS 2026-10-18 and on GitHub 2026-10-20). Both sealed tests carried two bars: the mean share
error at most 0.03 (met: 0.029 for version 1, 0.016 for version 2), and the true share inside the 95 % interval at least 80 % of the time
(not met: 0.42, then 0.78). After both scores, the author's limits study (2026-10-17) held the tool to its job, the share, so the
interval bar no longer gates it; the commissioning note discloses that the decision came after the scores, and the scores stand
unchanged. The intervals are therefore printed as information, labeled not calibrated.

**What it reads outside its own folder.** The laboratory spread comes from `doors/data/DEV_ATLAS_V21_01/transfer_v21_final.json`, a file in a
test-record folder. It is a runtime input and belongs in the chain's references (a to-do item of this Decoder).

**How to break it.** Read a pure array of one cell: its own share should be at least 0.80. Plant a 5 % share of one cell into a mixture:
its interval should exclude 0 in most mixtures. Remove half the markers: the shares must be withheld.

### `stage_deconv_v21.py::reference`

Reads μ, its range and the donor spread for the named cells from the atlas's compact parquet files (only those columns; 65,535 marks a
missing value) and builds sd as above.

*Code:* `stage_deconv_v21.py` line 26 (23 lines) · arguments: cells, atlas, spread · calls: array run 1 · constants read: `ATLAS`, `SPREAD` · files named: `import pyarrow.parquet`, `sites.parquet`

### `stage_deconv_v21.py::markers`

The marker sites for the panel, as above.

*Code:* `stage_deconv_v21.py` line 49 (8 lines) · arguments: R, per_cell · calls: array run 1 · constants read: `PER_CELL`

### `stage_deconv_v21.py::_solve`

The three-pass weighted non-negative least squares.

*Code:* `stage_deconv_v21.py` line 57 (5 lines) · arguments: y, A, S, lab_sd, passes · calls: array run 201

### `stage_deconv_v21.py::deconvolve`

The shares, raw sum, 95 % intervals and sites used for one sample.

Since 2026-10-22 it also records each cell's `resampled_median`, the middle of the resampled shares, beside the best fit, so a wide panel's two
estimates can be compared against known mixtures (development; the commissioned six-cell shares are the best fit, unchanged).

*Code:* `stage_deconv_v21.py` line 62 (16 lines) · arguments: beta, R, M, lab_sd, boot, seed · calls: array run 1 · constants read: `BOOT`, `LAB_SD`, `SEED`

### `stage_deconv_v21.py::sample`

Inside the planted test only: builds a planted array, each cell's DNA drawn from the atlas's donor spread, mixed in known shares, plus a
laboratory shift and array noise.

*Code:* `stage_deconv_v21.py` line 95 (2 lines) · arguments: f, lab · calls: not called on the traced runs

### `stage_deconv_v21.py::sample_check`

For the second sealed test: a pure array passes when it reads at least 0.80 of its own cell. Failing arrays are listed and not mixed.

*Code:* `stage_deconv_v21.py` line 172 (2 lines) · arguments: x, cell, R, M · calls: not called on the traced runs · constants read: `SELF_MIN`

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

### `stage_atlas_commission.py::lab_sd`

The SD of the atlas fit's array-laboratory offsets, read from `transfer_v21_final.json` at run time, never typed.

*Code:* `stage_atlas_commission.py` line 17 (4 lines) · arguments: terms · calls: array run 1 · constants read: `TERMS`

### `stage_atlas_commission.py::predictive_interval`

The 90 % predictive range for a new sample from a new laboratory. Arrays: N(μ, sd² + lab² + floor). Sequencing: the array value carried
through the platform map, plus binomial counting noise at the sample's coverage, the laboratory scatter and the floor.

*Code:* `stage_atlas_commission.py` line 21 (11 lines) · arguments: draws, sd_draws, kind, g, cov, scatter, level, seed, lab · calls: not called on the traced runs · constants read: `FLOOR`, `NOISE`, `RNG_SEED`

### `stage_atlas_commission.py::u1_coverage`

The share of a sample's sites that fall inside the predictive range.

*Code:* `stage_atlas_commission.py` line 32 (2 lines) · arguments: lo, hi, y · calls: not called on the traced runs

### `stage_atlas_commission.py::u1_verdict`

U1's verdict for one cell: commissioned when at least 95 % of its scored samples have coverage between 0.80 and 0.97.

*Code:* `stage_atlas_commission.py` line 34 (6 lines) · arguments: cov_by_sample, withheld, lo_bar, hi_bar · calls: not called on the traced runs · constants read: `SHARE`

### `stage_atlas_commission.py::u2_share`

The share of a tissue set's sites within tolerance of the anchor.

*Code:* `stage_atlas_commission.py` line 40 (2 lines) · arguments: set_mean, anchor, tol · calls: not called on the traced runs

### `stage_atlas_commission.py::u2_verdict`

U2's verdict for one tissue.

*Code:* `stage_atlas_commission.py` line 42 (4 lines) · arguments: shares, bar, epithelium · calls: not called on the traced runs · constants read: `SHARE`

### `stage_atlas_commission.py::u3_neutrophil`

U3's neutrophil error: the mean absolute error, the share within 0.05, the largest error, and whether both bars are met.

*Code:* `stage_atlas_commission.py` line 46 (4 lines) · arguments: est, truth, withheld_mask · calls: not called on the traced runs · constants read: `SHARE`

### `stage_atlas_commission.py::atlas_v21_use_status`

Commissioned or development for one use of the atlas, read from the table above.

*Code:* `stage_atlas_commission.py` line 56 (6 lines) · arguments: use, key · calls: not called on the traced runs · constants read: `ATLAS_V21_COMMISSIONING`

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

**Nearest-neighbor coupling, a second reading on the same count.** Every count also records whether a molecule's lost marks fall one at a
time or together: the pairs of neighboring CpGs unmethylated together on held molecules, over what each molecule's own length and error count
give at random (`_coupling`); 1 is independent. It is written beside the held error because clustered errors lower the lone-error count IAM-A
uses. In development: recorded and flagged, no gauge yet (INSTRUMENTS.md, In development).

### `stage_q2_selfref.py::_coupling`

Nearest-neighbor coupling on held molecules (development, recorded only, plan 49.5): the adjacent pairs of unmethylated calls, set against what random placement gives each molecule from its own length L and its own number k of unmethylated calls, k (k - 1) / L. 1 when errors fall independently, above 1 when they cluster along the molecule. Nothing fitted. Clustered errors lower the lone-error count IAM-A uses at the same error share, so the two are reported together.

*Code:* `stage_q2_selfref.py` line 128 (9 lines) · arguments: K · calls: not called on the traced runs

### `stage_q2_selfref.py::count_pat_tile_errors`

The held error tile by tile (DEV-ERASE-01 graded maintenance test): held molecules and lone errors by Stage Q2's own rules, each molecule in the tile of its first CpG (blocks of 2,000 consecutive CpG indices, so no genome table is needed), split into two halves that share no molecule (even and odd lines), so one half can measure a tile's level and the other remove the measurement's own scatter from the slope.

*Code:* `stage_q2_selfref.py` line 138 (15 lines) · arguments: lines, tile · calls: not called on the traced runs

### `stage_q2_selfref.py::chain_sha`

The SHA-256 over every `.py` file of the chain folder, names and contents, in sorted order.

*Code:* `stage_q2_selfref.py` line 29 (4 lines) · arguments: none · calls: not called on the traced runs · constants read: `HERE`

### `stage_q2_selfref.py::_canon`

A result in canonical JSON (sorted keys, no spaces), so the same result always hashes the same.

*Code:* `stage_q2_selfref.py` line 33 (1 lines) · arguments: obj · calls: bam run 4

### `stage_q2_selfref.py::payload_sha`

The SHA-256 of a result without its stamp.

*Code:* `stage_q2_selfref.py` line 34 (1 lines) · arguments: obj · calls: bam run 4

### `stage_q2_selfref.py::file_sha`

The SHA-256 of a file, read in 16 MB pieces.

*Code:* `stage_q2_selfref.py` line 35 (5 lines) · arguments: path, chunk · calls: not called on the traced runs

### `stage_q2_selfref.py::_rows`

The manifest rows a result belongs to, given directly or through the `IAM_MANIFEST` environment variable.

*Code:* `stage_q2_selfref.py` line 45 (3 lines) · arguments: manifest · calls: not called on the traced runs

### `stage_q2_selfref.py::stamp`

Attaches the chain stamp to a result, hashing it exactly as it will be stored (JSON turns number keys into text).

*Code:* `stage_q2_selfref.py` line 48 (8 lines) · arguments: obj, inputs, manifest · calls: not called on the traced runs · constants read: `STAGE`

### `stage_q2_selfref.py::write_json`

Writes a stamped result and reads it back to confirm the stored file verifies.

*Code:* `stage_q2_selfref.py` line 56 (6 lines) · arguments: obj, path, inputs, manifest · calls: not called on the traced runs

### `stage_q2_selfref.py::write_csv`

Writes a table and its stamp beside it, as `<file>.stamp.json`.

*Code:* `stage_q2_selfref.py` line 62 (4 lines) · arguments: df, path, inputs, manifest · calls: not called on the traced runs · constants read: `STAGE`

### `stage_q2_selfref.py::hg19_tiles`

The hg19 tiles of 2,000 consecutive CpGs on each autosome, in CpG-index coordinates; a partial last tile is dropped. With TILE_SOURCE "dictionary" (run_sample.py --dev-tiles dictionary, development, plan 14.9) the tiles follow the genome dictionary's chromosome ranges instead of the table's counts, which are short from chr14 on.

*Code:* `stage_q2_selfref.py` line 69 (18 lines) · arguments: none · calls: not called on the traced runs · constants read: `HG19_CPGS`, `TILE_SOURCE` · files named: `hg19_cpg_chrom_ranges.json`, `s range from Runtime Matrices/IAM_A_Positions/hg19_cpg_chrom_ranges.json`

### `stage_q2_selfref.py::_finish`

Turns tile counts and hole counts into the record: L, the number of tiles, ε and its opportunities, and the 10th, 50th and 90th
percentiles of the tile means for the shape check. When the gain-side counts are given, it adds gain_err and gain_opp.

*Code:* `stage_q2_selfref.py` line 87 (11 lines) · arguments: tile_calls, e_iso, e_opp, extra, g_iso, g_opp · calls: bam run 1

### `stage_q2_selfref.py::_mol`

The loose share counted molecule by molecule (development, recorded only): of the molecules with at least six calls on the tile half of the file, the share below the 0.70 cut the tile loose share uses and the share held neither marked nor unmarked (between 20 and 80 % methylated). Unlike the tile loose share, both add linearly when cell types mix, so a mix of cells reads the same share as one of them (synthetic test cfDNA 2).

*Code:* `stage_q2_selfref.py` line 98 (7 lines) · arguments: M · calls: not called on the traced runs

### `stage_q2_selfref.py::count_pat_site_classes`

Every call at a set of sites, grouped (for Met-A: the cell's held and held-absent identity sites), split by the molecule's class from all its calls by Stage Q2's rules (held, held absent, loose, short): the site level and each class's share and level, so a site's level is shown as the loose share beside the held error (development, step 3 version 3). A synthetic file with planted classes is recovered.

*Code:* `stage_q2_selfref.py` line 175 (24 lines) · arguments: lines, groups · calls: not called on the traced runs · constants read: `QUAL_MAX_METH`

### `stage_q2_selfref.py::count_pat_sites`

The held error counted only at a chosen set of sites (development, Met-A's derived baseline, step 3 version 2): Stage Q2's own rules unchanged, the same held and held-absent molecules classed from all their calls, only the calls counted restricted to the sites. A synthetic file with planted rates 0.06 and 0.05 at a site set returns them once the neighbors' rate is divided out.

*Code:* `stage_q2_selfref.py` line 154 (20 lines) · arguments: lines, sites · calls: not called on the traced runs · constants read: `QUAL_MAX_METH`

### `stage_q2_selfref.py::count_pat`

The `.pat` reader: even lines are added to their tile; odd lines that qualify as held runs add their interior calls and isolated holes. Since DEV-METAA-HEALTHY-01 it also counts the gain side, the mirror image: on molecules held unmarked (at least 6 calls, at most 20 % methylated, QUAL_MAX_METH), a lone methylated call between two unmethylated ones over the interior calls (gain_err, gain_opp). The loss side and every earlier output are unchanged.

*Code:* `stage_q2_selfref.py` line 105 (22 lines) · arguments: lines, tiles · calls: not called on the traced runs · constants read: `QUAL_MAX_METH`

### `stage_q2_selfref.py::_cpg_index`

The positions of every CpG in a chromosome's sequence.

*Code:* `stage_q2_selfref.py` line 202 (2 lines) · arguments: seq · calls: not called on the traced runs

### `stage_q2_selfref.py::_calls_of`

One aligned read's methylation calls at CpGs, from the Bismark XM tag or, for gemBS, from the read against the reference. Skips the first
and last 3 aligned bases. Also counts non-CpG conversion (a check on bisulfite conversion) and, on every 20th read, A/T mismatches (a check
on sequencing error).

*Code:* `stage_q2_selfref.py` line 204 (27 lines) · arguments: r, ref, ref0, sub · calls: not called on the traced runs

### `stage_q2_selfref.py::_merge`

Merges the calls of the two reads of a pair into one molecule; a CpG on which they disagree is dropped.

*Code:* `stage_q2_selfref.py` line 231 (6 lines) · arguments: a, b · calls: not called on the traced runs

### `stage_q2_selfref.py::count_bam_region`

Counts one 10 Mb region of an alignment file: primary, mapped, non-duplicate reads with mapping quality at least 10, pairs merged, then
the same alternation and the same two counts as the `.pat` reader. Also counts calls per qualifying molecule for the response curve.

*Code:* `stage_q2_selfref.py` line 237 (36 lines) · arguments: args · calls: not called on the traced runs · constants read: `QUAL_MAX_METH`

### `stage_q2_selfref.py::count_bam_region.done`

Counts one finished molecule: even molecules to their tile, odd held runs to ε.

*Code:* `stage_q2_selfref.py` line 244 (16 lines) · arguments: calls · calls: not called on the traced runs · constants read: `QUAL_MAX_METH`

### `stage_q2_selfref.py::combine_bam`

Adds the regions' counts into one record, with the conversion-failure and substitution rates.

*Code:* `stage_q2_selfref.py` line 273 (8 lines) · arguments: parts, assembly · calls: bam run 1

### `stage_q2_selfref.py::count_bam`

Splits the autosomes into 10 Mb regions and counts them in parallel.

*Code:* `stage_q2_selfref.py` line 281 (5 lines) · arguments: bam, fa, workers, assembly, chroms, span · calls: bam run 1 · constants read: `AUTOSOMES`

### `stage_q2_selfref.py::build_line`

How the line was built: per cell type, the median L and the opportunity-weighted ε, then a Theil–Sen line across cell types. Not called at
run time; the line is read from its frozen file.

*Code:* `stage_q2_selfref.py` line 288 (10 lines) · arguments: per_sample · calls: not called on the traced runs

### `stage_q2_selfref.py::line`

Loads the frozen line and refuses it if it was edited after the chain wrote it.

*Code:* `stage_q2_selfref.py` line 298 (6 lines) · arguments: version · calls: bam run 1 · constants read: `LINE_DIR`

### `stage_q2_selfref.py::predict`

ε predicted at a loose share: firm + slope × L.

*Code:* `stage_q2_selfref.py` line 304 (2 lines) · arguments: loose_share, version · calls: not called on the traced runs

### `stage_q2_selfref.py::residual`

A count against the line: measured ε, predicted ε, and their difference.

*Code:* `stage_q2_selfref.py` line 306 (5 lines) · arguments: rec, version · calls: bam run 1

### `stage_q2_selfref.py::build_shape_reference`

How the shape reference was built: per cell type, the lowest healthy 10th percentile, and the margin. Not called at run time.

*Code:* `stage_q2_selfref.py` line 317 (8 lines) · arguments: records · calls: not called on the traced runs · constants read: `SHAPE_Q`

### `stage_q2_selfref.py::shape_reference`

Loads the frozen shape reference and refuses it if edited.

*Code:* `stage_q2_selfref.py` line 325 (6 lines) · arguments: version · calls: bam run 2 · constants read: `LINE_DIR`

### `stage_q2_selfref.py::shape_check`

The sample-shape gate described above, with the measured percentile, the limit and the reason.

**The X and Y on the two routes (Decoder finding F37, measured 2026-10-21).** On the `.pat` route the held error is counted on every
chromosome and the loose share on the autosomes; the aligned-read route counts both on the autosomes. On 17 healthy samples of the committed
line, counting the held error on the autosomes only moved IAM-A by a mean of −0.32 % (range −0.71 % to +0.09 %), at most about 2.5 % of the
band's width (`doors/DEV_IAMA_SEX_01.md`). The commissioned line and band were built on the `.pat` route and are kept.

*Code:* `stage_q2_selfref.py` line 331 (9 lines) · arguments: rec, cell, version, ref · calls: bam run 1

### `stage_q2_selfref.py::reading`

IAM-A = measured ε / predicted ε.

*Code:* `stage_q2_selfref.py` line 342 (4 lines) · arguments: rec, version · calls: bam run 1

### `stage_q2_selfref.py::build_band`

How band v1 was built: the 2.5th to 97.5th percentile of pooled healthy readings. Not called at run time.

*Code:* `stage_q2_selfref.py` line 346 (6 lines) · arguments: healthy, version · calls: not called on the traced runs

### `stage_q2_selfref.py::band`

Loads band v1 and refuses it if edited.

*Code:* `stage_q2_selfref.py` line 356 (2 lines) · arguments: version · calls: bam run 1 · constants read: `LINE_DIR`

### `stage_q2_selfref.py::grand_verdict`

The sealed grand test's scorer: each healthy sample shape-checked, each read sample inside or outside the band; met when at least 75 % of at
least 8 read samples are inside.

*Code:* `stage_q2_selfref.py` line 360 (12 lines) · arguments: records, version, B, ref · calls: not called on the traced runs · constants read: `GRAND_MIN_READ`, `GRAND_SHARE`

## 10. Stage Q0, intake for sequencing files: `stage_q0_intake.py`

**What it is for.** Stage 0 for a `.pat` file: an array cannot reach Met-A without Stage 0, and a `.pat` file cannot reach IAM-A without
Stage Q0. Each check stops the file with a named reason, and every measured value is recorded whether or not it stops the file.

**Label: standard method** (file integrity, genome build, coverage and conversion checks, as in the ENCODE whole-genome bisulfite standard).

| Check | What it measures | Stops the file when | Limit and source |
|---|---|---|---|
| Q0.7 specimen | the cell type | on the chain v4 route, the cell type has no healthy shape reference | rule |
| Q0.1 readable | the file decompresses to its end; every line has 4 fields (chromosome, CpG index, a pattern of C, T and dots, a count) | any malformed line or a truncated file | rule |
| Q0.2 build | every line's CpG index lies inside its chromosome's range in the hg19 CpG dictionary (`hg19_cpg_chrom_ranges.json`) | any line outside | rule |
| Q0.4 whole genome | molecules on every autosome | an autosome has none | rule |
| Q0.4 coverage | interior calls on molecules with at least 6 calls | never (recorded) | — |
| Q0.5 read length | the share of molecules with at least 6 calls | not set: recorded only | to be set on healthy files |
| Q0.3 conversion | the bisulfite conversion rate from the alignment record, when one is given | below 0.98 | the ENCODE whole-genome bisulfite standard |
| Q0.6 duplicates | the duplicate fraction from the alignment record, when given | not set: recorded only | to be set on healthy files |

**Not run on a BAM.** Q0's file checks are written for the `.pat` format. On the `--bam` route the intake record says so and why, and
Stage Q2's counter records the bisulfite conversion failure (from non-CpG calls) and the substitution rate on A and T; the sample-shape
check refuses a cell type with no healthy reference.

### `stage_q0_intake.py::_ranges`

Loads the hg19 CpG-index range of each chromosome, if the file is present.

*Code:* `stage_q0_intake.py` line 35 (3 lines) · arguments: none · calls: not called on the traced runs · constants read: `RANGES`

### `stage_q0_intake.py::scan_pat`

One pass over the file, gzipped (including multi-member bgzip) or plain: completeness, malformed lines, molecules per chromosome, calls,
molecules with at least 6 calls, interior opportunities, and lines outside their chromosome's range.

*Code:* `stage_q0_intake.py` line 40 (36 lines) · arguments: path · calls: not called on the traced runs · constants read: `MIN_CALLS` · files named: `hg19_cpg_chrom_ranges.json`

### `stage_q0_intake.py::scan_pat.line`

Parses and counts one line.

*Code:* `stage_q0_intake.py` line 45 (11 lines) · arguments: l · calls: not called on the traced runs · constants read: `MIN_CALLS`

### `stage_q0_intake.py::intake`

Runs the checks in the table and returns the record with each check's value, rule and result, and the decision: stop at the first
refusal, or proceed.

*Code:* `stage_q0_intake.py` line 115 (59 lines) · arguments: path, specimen, cell, alignment_qc, selfref_cells · calls: not called on the traced runs · constants read: `ACCEPTED`, `AUTOSOMES`, `GEO_BUILD`, `MIN_CALLS`, `Q0_LIMITS`

### `stage_q0_intake.py::intake.chk`

Records one check and, when it stops the file, the first refusal's code and reason, with its five-part explanation.

*Code:* `stage_q0_intake.py` line 129 (10 lines) · arguments: step, ok, value, rule, code, stop · calls: not called on the traced runs

## 10b. Genome builds: which genome each reading counts on, and how the chain keeps them apart (2026-10-22, lesson C34)

**Why it matters.** A sequencing file (.pat) names each CpG by its index in one genome assembly's list of CpGs. hg19 and hg38 number the
genome differently, so the same index points at a different CpG on each. A file made on hg38 and counted on hg19 does not fail loudly: every
count lands on the wrong site, and a globe drawn from it looks like a real picture. The withdrawn IAM-A genome globes (DEV-IAMA-GLOBE-01 steps
2 to 5) did exactly this: GEO holds each healthy-cell sample of the published sequencing atlas twice, `..._name.hg38.pat.gz` and
`..._name.pat.gz` (hg19); the test's own download took the first name in alphabetical order, the hg38 one, and counted it on hg19 positions.

**The rule.** Every .pat file is read on **hg19** (wgbstools 0.1.0's CpG index, `stage_q0_intake.GEO_BUILD`): Stage Q0's index ranges, Stage
Q2's tiles and the whole-genome globe's layout are all hg19 indexes. An aligned BAM is different: Stage Q2 reads the bases themselves against
the reference sequence given, on hg19 or hg38 (`--assembly`, default hg19), so the only requirement is that the BAM and its reference are the
same build (IAM-A's line was built this way from ENCODE's GRCh38 files and from the published atlas's hg19 .pat files). The arrays carry their
own probe names and have no build question.

**How the chain enforces it, three times over.**
1. **The chain fetches the file.** `run_sample.py --geo-pat GSM` (and `--prior-geo-pat` for a second reading) lists the sample's files on
   GEO and takes the one that declares no other build (`stage_q0_intake.geo_pat`); if none or several remain it stops and lists what GEO holds.
   The bundle records which file it took, the others it saw, and the file's checksum (`geo_fetch`). This record is the one place a typed id
   is not hashed: a public GEO accession is the file's address, not a person's id, and hashing it would leave an address that points nowhere.
2. **Stage Q0 stops a wrong build twice.** First by the name: a file named as hg38, GRCh38, hg18 or a mouse build is stopped before a line is
   read, and the refusal says which build it is and what to do (`stage_q0_intake.build_named`). Then by the content: every line's CpG index
   must lie inside its chromosome's hg19 range (check Q0.2); the hg38 file of the withdrawn steps had 48,859,268 lines outside them.
3. **The push gate refuses side downloads.** Rule 23b-g: a changed test or picture script that downloads a .pat file itself, instead of
   letting the chain fetch it, is refused (controls C13 and C14).

### `stage_q0_intake.py::build_named`

The build a file's name declares when it is not hg19 (hg38, GRCh38, hg18, mm9, mm10, GRCm38, GRCm39), else nothing; read from the name alone.

*Code:* `stage_q0_intake.py` line 81 (5 lines) · arguments: path · calls: not called on the traced runs · constants read: `OTHER_BUILDS`

### `stage_q0_intake.py::geo_pat`

A GEO sample's .pat file on the build the chain reads, as described above, with its name, address, the other files GEO holds, its size
and checksum; stops, with what GEO holds, when it cannot pick exactly one.

*Code:* `stage_q0_intake.py` line 87 (27 lines) · arguments: gsm, dest, build, retries · calls: not called on the traced runs · constants read: `GEO_BUILD`

## 11. Group-A, has a whole group moved: `stage_groupa.py`

**What it is for.** One person's whole-blood Met-A cannot show a 1 % change: healthy people differ by about 2 % after the tare. The
average of a group can. Group-A compares two groups read in one series.

**Label: standard method** (two tared means, Welch's standard error), applied to one physical reading per person, each against its own
expected value for that person's cell mix.

**How it reads.** Each person is read by the one command, unchanged. Each reading is tared against healthy arrays run the same way: the
median of the healthy references on the same slide when the slide holds at least 3, else of the same series. A sample is never its own
reference. Then the two groups' tared means are compared:

  difference d = mean₁ − mean₂, standard error SE = √(s₁²/n₁ + s₂²/n₂), t = d / SE.

|t| ≥ 2 is a group that moved. |t| < 2 is no difference found, and the smallest difference the design could have found (2 × SE) is
printed. Nothing is fitted.

**Built from, tested on.** Sealed before any array was opened, then scored once on 278 healthy controls of a laboratory no instrument had
read (GSE318173): 263 read; women and men 0.36 % apart at 1.2 standard errors; a 1 % loss planted into half of them found at 7.3 standard
errors. Commissioned 2026-10-20 for groups of whole bloods run in one series, EPIC v1. Release check V10 reproduces bars 2 and 3 to every
printed digit from the committed rows. Groups in any other specimen are read and labeled preliminary.

**Why one series.** Across laboratories, untared readings differ by several percent. Two groups from two laboratories would differ by
their laboratories.

### `stage_groupa.py::tare`

Each reading over the median of its healthy references (same slide when there are at least 3 there, else the series), excluding itself;
records which scope was used.

*Code:* `stage_groupa.py` line 28 (22 lines) · arguments: r, healthy · calls: not called on the traced runs

### `stage_groupa.py::welch`

The two groups' means, counts, difference, standard error and t.

*Code:* `stage_groupa.py` line 51 (4 lines) · arguments: a, b · calls: not called on the traced runs

### `stage_groupa.py::compare`

Tares, compares and calls the result, with the smallest findable difference and the commissioning label.

*Code:* `stage_groupa.py` line 56 (17 lines) · arguments: rows, group_col, a, b, healthy, specimen, platform · calls: not called on the traced runs · constants read: `PRELIMINARY`

### `stage_groupa.py::collect`

Builds one row per person from the run's report bundles and the researcher's sample list (id, slide, group column, optional
healthy_reference).

*Code:* `stage_groupa.py` line 74 (8 lines) · arguments: bundle_dir, samples_csv, out_csv · calls: not called on the traced runs · files named: `_bundle.json`

## 12. The retired position reading, Stage Q: `stage_q_iam_a.py`

**Retired.** IAM-A was first read against a frozen position P per cell type, measured on purified granulocytes. That reading was replaced
on 2026-10-17 by Stage Q2, each sample against its own reference. The one command runs Stage Q only with `--site-table` or
`--dev-iama-position`, and labels it development. It is kept so earlier records can be reproduced. Its text still uses the retired term
"copy error"; what it counts is the held error described in section 9.

### `stage_q_iam_a.py::positions`

Loads the frozen positions file (`iama_positions_v2.json`).

*Code:* `stage_q_iam_a.py` line 35 (4 lines) · arguments: none · calls: not called on the traced runs · constants read: `POS`, `_P`

### `stage_q_iam_a.py::_H`

The entropy of a rate in bits.

*Code:* `stage_q_iam_a.py` line 40 (1 lines) · arguments: e · calls: not called on the traced runs

### `stage_q_iam_a.py::_pat_lines`

Streams the lines of a `.pat` file, gzipped or plain, optionally cut at a byte count (the retired version 1 read only part of each file).

*Code:* `stage_q_iam_a.py` line 42 (27 lines) · arguments: path, max_bytes, chunk · calls: not called on the traced runs

### `stage_q_iam_a.py::pat_site_table`

The per-site table (opportunities and holes in two halves of the molecules) from a `.pat` file.

*Code:* `stage_q_iam_a.py` line 72 (22 lines) · arguments: path, max_bytes · calls: not called on the traced runs · constants read: `PAT_PIPELINE`

### `stage_q_iam_a.py::_genomic_order`

Sorts a site table by chromosome and CpG index.

*Code:* `stage_q_iam_a.py` line 95 (8 lines) · arguments: T · calls: not called on the traced runs

### `stage_q_iam_a.py::cscore`

The retired IAM-A clustering score over the site table, pooled and per half.

*Code:* `stage_q_iam_a.py` line 105 (14 lines) · arguments: site_table, eps, halves · calls: not called on the traced runs · constants read: `CSCORE_BLOCK_SITES`, `CSCORE_MIN_BLOCKS`

### `stage_q_iam_a.py::cscore.one`

The clustering score of one table.

*Code:* `stage_q_iam_a.py` line 108 (8 lines) · arguments: o, k · calls: not called on the traced runs · constants read: `CSCORE_BLOCK_SITES`, `CSCORE_MIN_BLOCKS`

### `stage_q_iam_a.py::read`

The retired reading, IAM-A = H(ε) / (P × H(ε₀)), refused for a pipeline other than the one P was measured on, or for part of a file.

*Code:* `stage_q_iam_a.py` line 121 (27 lines) · arguments: site_table, cell, pipeline, mask, allow_partial · calls: not called on the traced runs · constants read: `IAMA_BAND_STATE`, `MIN_HALF_OPPORTUNITIES`, `MIN_OPPORTUNITIES`

### `stage_q_iam_a.py::tare`

The retired reading's same-run tare.

*Code:* `stage_q_iam_a.py` line 152 (22 lines) · arguments: rec, refs, sample_id · calls: not called on the traced runs · constants read: `IAMA_BAND_STATE`, `MIN_REFS`

## 13. Two draws of one person, Stage 12b: `serial_mode.py`

**What it is for.** Comparing two draws of the same person. `run_sample.py` calls two of these functions when `--prior-betas` is given;
the rest are tools.

**Label: standard method.**

### `serial_mode.py::check_same_person`

Refuses unless the two draws carry the same identifier hash, array type and pipeline.

*Code:* `serial_mode.py` line 7 (8 lines) · arguments: now, prior · calls: not called on the traced runs

### `serial_mode.py::delta_cells`

The change in each cell's A for cells present in both draws; cells that appeared or disappeared are listed, not differenced. Reads the
per-cell layout of an older bundle (`cells_all`); a tool, not called by the one command.

*Code:* `serial_mode.py` line 16 (16 lines) · arguments: now, prior, floor · calls: not called on the traced runs

### `serial_mode.py::delta_sky`

β now minus β before, at the sites both draws measured. No expectation, no composition.

*Code:* `serial_mode.py` line 33 (8 lines) · arguments: beta_now, beta_prior, floor_beta · calls: not called on the traced runs

### `serial_mode.py::snp_noise_floor_A`

The two arrays' own noise (Stage 1's SNP-probe model, σ²(β) = a + b·β(1 − β)) carried through to A over the cell's identity sites:
ΔA ≈ 2.576 × |dH/dβ| × √(var_now + var_prior) / H_ref at the mean β, the 99 % two-sided change the noise alone could make. A lower bound
on the smallest change that means anything. It reads the identity sites from an older bundle layout (`loci` of a per-cell entry)
that chain v4 bundles do not carry; it is a tool, not a stage.

*Code:* `serial_mode.py` line 42 (13 lines) · arguments: pci_entry, beta_now, beta_prior, snp_now, snp_prior, hmin · calls: not called on the traced runs

### `serial_mode.py::snp_noise_floor_A.var_mean`

For one array, the variance of the mean β over the identity sites from its noise model, Σ(a + b·β(1 − β)) / n², and the mean β.

*Code:* `serial_mode.py` line 49 (3 lines) · arguments: beta, snp · calls: not called on the traced runs

### `serial_mode.py::trajectory`

A person's readings in draw order with their dates and the step between draws: numbers in order, no fit, no slope. Reads the same older
per-cell layout as `delta_cells`.

*Code:* `serial_mode.py` line 56 (11 lines) · arguments: bundles · calls: not called on the traced runs

## 14. The report: `report_v4.py`, `report_v3.py`, `report_figs_v4.py`, `report_pdf_v4.py`

**What it is for.** Writing what the stages measured, and nothing they did not. One self-contained HTML page per sample, and the same
report as two PDFs, built from the same rows and pictures so the formats cannot disagree.

**Label: standard method** (presentation). The report computes no reading. Every number it prints comes from the run's bundle or from
a committed reference file.

**Its parts, in page order.**

1. **The researcher view**: the reading at a glance (each instrument, the question it answers, its reading, its state and its
   commissioning status); the physics behind each reading with this run's own numbers; the cell shares; Met-A on its
   gauge; the C-Score with its sky and the overlay of the sample on its healthy reference; IAM-A on its gauge.
2. **The plain view**: one gauge and one sentence per reading, and the sky.
3. **The technical record** (`report_v3.build`): every stage's record, the red flags, the safeguards, troubleshooting, and the
   development toolkit table.

**The gauge** (`report_figs_v4.gauge_png`). Left end the floor (derived), the Normal band around healthy 1 (measured), right end
saturation, every identity site fully scrambled (derived). Every mark is read from `Runtime Matrices/Gauge/cell_gauge_marks_v1.json`;
nothing is typed in the drawing code. On IAM-A's scale healthy sits close to the floor, so a magnified strip shows the band. A withheld
state draws no gauge at all. The C-Score has no derived ends, because it is a clustering ratio: its bar runs from scattered to clustered,
with its band.

**Commissioned or not, on every line.** A commissioned reading carries its record. Anything else is still shown and says so: "This score
is preliminary until [instrument] is officially commissioned on [cell]" (`report_v4.untested`).

**Red flags** (`report_v3.red_flags`): STOP (refused), FLAG (a commissioned call such as a clustered departure), WITHHELD (a reading not
given, with its reason), CAUTION and NOTE. They are written into the bundle as well as the page.

**Safeguards** (`report_v3.safeguards`), each PASS, FAIL or NOT_RUN on this run:
- **Rendered-claim scan:** no disease, cohort or population words in the report's prose.
- **Formula self-test:** H(0.5) = 1, H(0.25) = 0.81128, H(10⁻⁶) under 3 × 10⁻⁵.
- **Anchors**, **deconvolver conformance** and **atlas separability.**

**How to break it.** Write a disease word into a reading's text: the claim scan must fail. Remove a mark from the gauge file: the gauge
must not draw.

### `report_v4.py::_uri`

A figure file as a data URI, so the page is self-contained.

*Code:* `MethylPhys_Interface/report_v4.py` line 38 (1 lines) · arguments: path · calls: array run 6, bam run 2

### `report_v4.py::_H`

The entropy of a rate in bits, for the physics rows.

*Code:* `MethylPhys_Interface/report_v4.py` line 39 (1 lines) · arguments: p · calls: bam run 2

### `report_v4.py::_sci`

A number in HTML scientific notation (×10ⁿ).

*Code:* `MethylPhys_Interface/report_v4.py` line 40 (2 lines) · arguments: v · calls: bam run 2

### `report_v4.py::_marks`

Loads the gauge marks file.

*Code:* `MethylPhys_Interface/report_v4.py` line 42 (1 lines) · arguments: none · calls: array run 3, bam run 3 · constants read: `CH` · files named: `cell_gauge_marks_v1.json`

### `report_v4.py::_href`

Met-A's healthy reference in bits per site (0.33026), read from `metA_floors_v1_3.json`, never typed.

*Code:* `MethylPhys_Interface/report_v4.py` line 45 (2 lines) · arguments: none · calls: not called on the traced runs · files named: `metA_floors_v1_3.json`

### `report_v4.py::cellname`

A cell type's display name.

*Code:* `MethylPhys_Interface/report_v4.py` line 47 (2 lines) · arguments: c · calls: array run 30, bam run 1 · constants read: `_NAMES`

### `report_v4.py::untested`

The line printed beside a reading outside what its instrument was commissioned on.

*Code:* `MethylPhys_Interface/report_v4.py` line 49 (5 lines) · arguments: what, inst, shown, noun · calls: array run 7 · constants read: `_NAMES`

### `report_v4.py::chip`

The colored status label: commissioned, refused or withheld, not run, or development.

*Code:* `MethylPhys_Interface/report_v4.py` line 54 (4 lines) · arguments: status · calls: array run 12, bam run 5 · constants read: `E`

### `report_v4.py::_glance`

The at-a-glance table as HTML.

*Code:* `MethylPhys_Interface/report_v4.py` line 59 (3 lines) · arguments: o · calls: array run 1, bam run 1 · constants read: `E`

### `report_v4.py::glance_rows`

The at-a-glance rows, from the bundle: the Deconvolver's top three shares, Met-A (tared when it can be), the C-Score, and IAM-A, each
with its state and status. The PDFs use the same rows.

*Code:* `MethylPhys_Interface/report_v4.py` line 63 (65 lines) · arguments: o · calls: array run 2, bam run 2

### `report_v4.py::_shares`

The cell-shares section: each share with its 95 % interval and status, or the reason they were withheld.

*Code:* `MethylPhys_Interface/report_v4.py` line 129 (12 lines) · arguments: o · calls: array run 1, bam run 1 · constants read: `E`

### `report_v4.py::_meta`

The Met-A section: its gauge (tared reading; an untared whole-blood reading has no gauge position) and its numbers.

*Code:* `MethylPhys_Interface/report_v4.py` line 142 (9 lines) · arguments: o · calls: array run 1, bam run 1 · constants read: `E`

### `report_v4.py::_cscore`

The C-Score section: its bar, its sky and the overlay.

*Code:* `MethylPhys_Interface/report_v4.py` line 152 (9 lines) · arguments: o · calls: array run 1, bam run 1 · constants read: `E`

### `report_v4.py::_iama`

The IAM-A section: its gauge, or the refusal.

*Code:* `MethylPhys_Interface/report_v4.py` line 162 (13 lines) · arguments: o · calls: array run 1, bam run 1 · constants read: `E`

### `report_v4.py::_figs`

Draws this sample's gauge and C-Score pictures beside the report.

*Code:* `MethylPhys_Interface/report_v4.py` line 189 (22 lines) · arguments: o, base · calls: array run 1, bam run 1

### `report_v4.py::_physics`

The physics section as HTML.

*Code:* `MethylPhys_Interface/report_v4.py` line 216 (6 lines) · arguments: o · calls: array run 1, bam run 1 · constants read: `E`, `PHYS_INTRO`

### `report_v4.py::physics_rows`

Each reading set in the physics it rests on, with this run's numbers. IAM-A: ε measured, ε predicted at the sample's own L, saturation
0.5/ε_pred, and the floor 1/(1 + e^M) as a fraction of ε_pred. Met-A: the mean entropy over the identity sites against its expectation,
what healthy 1 is on this run (the median of the same-run references, or the committed healthy reference), and on isolated neutrophils
the reading in bits per site. The C-Score: the clustering ratio and its band. The 0.330263 bits per site it prints is typed in this
function rather than read from `metA_floors_v1_3.json` (a fix on the Decoder's list).

*Code:* `MethylPhys_Interface/report_v4.py` line 223 (32 lines) · arguments: o · calls: array run 2, bam run 2 · constants read: `E`

### `report_v4.py::_skypair`

The overlay of the sample on its healthy reference, with how sure the healthy reference is at those sites.

*Code:* `MethylPhys_Interface/report_v4.py` line 260 (8 lines) · arguments: o · calls: array run 1 · constants read: `SKY_CAP`

### `report_v4.py::_plain`

The plain view: one gauge and one sentence per reading.

*Code:* `MethylPhys_Interface/report_v4.py` line 269 (19 lines) · arguments: o, F · calls: array run 1, bam run 1 · constants read: `E`

### `report_v4.py::build`

Builds the whole page: the technical record from `report_v3.build`, the figures, both views, and the two PDFs.

*Code:* `MethylPhys_Interface/report_v4.py` line 415 (32 lines) · arguments: o, out, sid · calls: array run 1, bam run 1 · constants read: `CSS`, `E`

### `report_v3.py::_gauge`

The gauge as inline HTML, drawn from the gauge marks file.

*Code:* `MethylPhys_Interface/report_v3.py` line 51 (38 lines) · arguments: A, lo, hi, w, label, instrument, top · calls: array run 1, bam run 2 · files named: `CANON/foundation.json`, `Runtime Matrices/Gauge/cell_gauge_marks_v1.json`, `drawn from Runtime Matrices/Gauge/cell_gauge_marks_v1.json`

### `report_v3.py::_H`

The entropy of a rate in bits, for the formula self-test.

*Code:* `MethylPhys_Interface/report_v3.py` line 90 (3 lines) · arguments: b · calls: array run 3, bam run 3

### `report_v3.py::_load`

Loads a chain file, or nothing if it is missing.

*Code:* `MethylPhys_Interface/report_v3.py` line 95 (3 lines) · arguments: rel · calls: array run 3, bam run 3 · constants read: `CHAIN`

### `report_v3.py::red_flags`

The red flags, from the bundle only.

*Code:* `MethylPhys_Interface/report_v3.py` line 101 (26 lines) · arguments: o · calls: array run 1, bam run 1

### `report_v3.py::safeguards`

The safeguards listed above.

*Code:* `MethylPhys_Interface/report_v3.py` line 129 (44 lines) · arguments: o, prose · calls: array run 1, bam run 1 · constants read: `CLAIM_SCAN` · files named: `Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`, `Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_2.json`, `Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json`

### `report_v3.py::troubleshooting`

For each stop on this run, what it usually means and what to do.

*Code:* `MethylPhys_Interface/report_v3.py` line 175 (17 lines) · arguments: o · calls: array run 1, bam run 1

### `report_v3.py::toolkit_table`

The development stages and whether each ran on this sample.

*Code:* `MethylPhys_Interface/report_v3.py` line 194 (12 lines) · arguments: o · calls: array run 1, bam run 1 · constants read: `NOT_BUILT`, `TOOLKIT_STAGES`

### `report_v3.py::_table`

A plain HTML table.

*Code:* `MethylPhys_Interface/report_v3.py` line 208 (4 lines) · arguments: rows, head · calls: array run 7, bam run 7

### `report_v3.py::build`

The technical record: intake, Stage 1, the mix, Met-A, the tare, the noise gate, the C-Score, IAM-A, the red flags, the safeguards,
troubleshooting and the toolkit table, the run-it-yourself command, and a copy of the bundle inside the page. Returns the page's path,
size and any refusals; the bundle file itself is written by `run_sample.py`.

*Code:* `MethylPhys_Interface/report_v3.py` line 214 (83 lines) · arguments: o, out, sid · calls: array run 1, bam run 1 · constants read: `FLAG_ORDER`, `STAGES_2B` · files named: `cell_gauge_marks_v1.json`

### `report_figs_v4.py::sci`

A number in scientific notation for a figure.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 13 (2 lines) · arguments: v · calls: not called on the traced runs

### `report_figs_v4.py::_bar`

Draws one gauge bar: the gradient from the floor through healthy to the ceiling, the band, the needle and the ticks.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 15 (13 lines) · arguments: ax, lo_v, hi_v, band, A, fl, cl, y0, h, grad, ticks, mid · calls: array run 2, bam run 2

### `report_figs_v4.py::gauge_png`

The floor-to-ceiling gauge for Met-A or IAM-A, with the magnified strip when healthy sits near the floor.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 29 (20 lines) · arguments: out, name, A, floor, ceiling, band, subject, state, unit, ends · calls: array run 1, bam run 1

### `report_figs_v4.py::cscore_png`

The C-Score bar, scattered to clustered; the scale stretches to hold the reading.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 50 (10 lines) · arguments: out, C, band_hi, band_lo, subject, state · calls: array run 1

### `report_figs_v4.py::atlas_sky_png`

The healthy reference of the sample's cell from IAM-Atlas v2.1, at the same sites, in the same order and on the same pixels as the
sample's C-Score sky, so the two line up site for site.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 61 (40 lines) · arguments: out, ids, cell, atlas_dir, nest_pix2ang, title · calls: array run 1 · files named: `Planck_Parchment_RGB.txt`, `import pyarrow.parquet`, `sites.parquet`

### `report_figs_v4.py::changes_sky_png`

The sample laid over its healthy reference at Met-A's 6,000 identity sites: the healthy cell in grays, color only on blocks that changed,
by the globe's rule (blocks of 10 neighboring sites in genome order, each block's mean departure beyond the 5 % family-wise line), so the
flat picture and the globe count the same change (healthy example blood: 0 of 597 blocks; the planted clustered loss: 44). Each pixel keeps
the block value of largest size, so a changed block is never averaged away.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 102 (55 lines) · arguments: out, z, cell, atlas_dir, nest_pix2ang, readings, thr, title · calls: array run 1 · files named: `import pyarrow.parquet`, `sites.parquet`

### `report_figs_v4.py::changes_sky_png.pext`

The block value of largest size in each pixel, its sign kept, so a changed block is never averaged away by an unchanged neighbor.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 128 (5 lines) · arguments: x · calls: not called on the traced runs

### `report_figs_v4.py::changes_sky_png.pmean`

The mean of the values falling in each pixel.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 126 (2 lines) · arguments: x · calls: array run 2

### `report_pdf_v4.py::P`

A paragraph for the PDF, keeping only the tags the PDF library understands.

*Code:* `MethylPhys_Interface/report_pdf_v4.py` line 25 (4 lines) · arguments: t, st · calls: array run 90, bam run 49 · constants read: `S`

### `report_pdf_v4.py::_img`

A figure scaled to a width, or nothing if missing.

*Code:* `MethylPhys_Interface/report_pdf_v4.py` line 30 (4 lines) · arguments: path, w · calls: array run 8, bam run 6

### `report_pdf_v4.py::_head`

The PDF's heading block, with the research-use line and how to read the commissioning status.

*Code:* `MethylPhys_Interface/report_pdf_v4.py` line 35 (8 lines) · arguments: o, sid, when, view · calls: array run 2, bam run 2

### `report_pdf_v4.py::_tbl`

A PDF table.

*Code:* `MethylPhys_Interface/report_pdf_v4.py` line 44 (5 lines) · arguments: rows, widths, head · calls: array run 3, bam run 2

### `report_pdf_v4.py::build_pdfs`

The researcher and plain PDFs, from the same rows and pictures as the page.

*Code:* `MethylPhys_Interface/report_pdf_v4.py` line 55 (59 lines) · arguments: o, F, out, sid, when · calls: array run 1, bam run 1

### `report_pdf_v4.py::build_pdfs.doc`

Writes one PDF.

*Code:* `MethylPhys_Interface/report_pdf_v4.py` line 58 (3 lines) · arguments: path, story · calls: array run 2, bam run 2

## 14b. The report's globe: `report_globe_v4.py`

**What it is for.** A reviewer turns a globe in the HTML report and sees where the sample differs from its healthy reference, with a
slider that fades only the healthy background, so the change can be seen by itself. The change is fixed by the measurement: each site's
difference from the healthy reference in that site's healthy width (the atlas's person-to-person spread and its own uncertainty), centered
and scaled by the sample's own spread, then read in blocks of 10 neighboring sites against the fixed 5 % family-wise line (below). Healthy
blood: 45 of 77,355 blocks. Tried first and dropped the same day: the atlas's 95 % range alone flagged 84 % of a healthy blood (it is how
sure the atlas is of the healthy average, not how much a healthy person varies), and single sites beyond 3 lit 2.1 % of a healthy blood's
sites across the whole globe. **A picture, not a reading:** nothing is scored from it, and Met-A and the
C-Score keep their commissioned purified-array reference.

**Label: standard method** (display). The layout is the atlas's sky sphere: every IAM-Atlas v2.1 site in genome order at NESTED pixel
floor(i × NPIX / n), NSIDE 256, the same pixels as the atlas's cell pages and the public globe. The order is frozen in
`Runtime Matrices/Atlas_V21/sky_order_v21.npz` (built by `atlas/tools/sky_order_v21.py` with Illumina's EPIC v1 B4 and 450K manifests,
checked against the cell pages: 785,718 measured neutrophil pixels on both), so the chain draws it without downloading a manifest.

**The reference, by specimen.** An isolated cell type is compared with that cell in the atlas. Whole blood and buffy coat are compared
with the atlas's healthy blood for this person's own mix: the six panel cells weighted by their Stage D21 shares, scaled to sum to 1.
Comparing blood with one cell would light up every site where blood differs from that cell.

**Four layers.** The sample; the reference; the difference (sample minus reference) at every site both hold; and Met-A's 6,000 neutrophil
identity sites as the C-Score's departure map, centered and scaled by the sample's own median and spread as the report's overlay, drawn as
dots of radius 3 texture pixels at their own places so that a run of departing sites shows as a patch. The page prints the share of the
measured globe above the chosen size, from the equal-area sky pixels.

**What a reader will see on healthy blood.** The atlas blends 21 sources and sits about 0.006 closer to 0.5 than one array
(DEV-REFERENCES-01), so a faint even difference everywhere (median 0.018 on the real healthy example, 2.9 % of the globe above 0.10). In a
woman, the X chromosome as a patch: one X is held methylated, and the reference pools both sexes (the last 2 % of the genome order differed 7
to 11 times as often as the rest on the first real blood). A small loss spread over many sites does not show on the whole-genome layer; it
shows on the departure layer.

**How to break it.** Lay the atlas's own healthy neutrophil on the globe as an isolated neutrophil sample: every difference must be 0
(release check V12). Give whole-blood shares that do not sum to 1: the weights must rescale. Edit the atlas site list: the globe must refuse
to draw, because the frozen order no longer matches.

**Two more layers (2026-10-21).** Met-A's own reference, the purified arrays: at its 6,000 identity sites, the purified group profiles
weighted by this blood's Stage A fractions (or the neutrophil profile for isolated neutrophils), with the sample's difference from it, drawn
as dots. And each cell the IAM-Deconvolver read, named with its share of this sample, as that cell's healthy map in IAM-Atlas v2.1. Each
reading is shown against the reference its instrument reads against.

### `report_globe_v4.py::_dots`

Draws a sparse texture as dots: every texture pixel within 3 of a measured one takes the nearest one's value.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 139 (7 lines) · arguments: v · calls: not called on the traced runs · constants read: `R_DOT`

### `report_globe_v4.py::purified`

Met-A's purified-array reference at its 6,000 sites: the Stage A fractions times the purified group profiles for blood, the neutrophil profile
for isolated neutrophils; nothing for other specimens.

**One viewer for every globe (`globe_viewer.py`, 2026-10-21).** A globe is a spec: its pictures (each stored once), and groups of layers. A
layer is a picture shown as it is, the fixed change with a slider that fades only the background (the change and its count never move), or a
series of pictures a slider steps through (the noise maps). The same code draws the globe inside a report, the report's own full page
(written beside the report as `<name>_globe.html` and linked from it), and the site's Globe Explorer (`atlas/sphere.html`). It needs only
the standard library, so the site build imports it unchanged; release check V12 and the site build both check that its script parses.

**The map-layer slider (2026-10-22, lesson C35).** A map layer (a value picture: a cell's methylation, how sure the atlas is) carries a
lossless value picture beside it. Sliding right keeps only the sites at or above the chosen value; **every other pixel goes dark**, the sites
below the value, the sky with no site, and the rim of each dot alike, so at the right end the globe is dark. Release check V13 requires at
least 90 % of the globe dark at every map layer's right end, layers drawn as dots included.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 147 (9 lines) · arguments: fractions, specimen · calls: not called on the traced runs · constants read: `BLOOD`, `CH` · files named: `blood_composition_EPIC_v1.json`

### `globe_viewer.py::inline`

The reverse of dedupe: each layer's picture written back in place of its asset name, so groups taken from several globes can be joined into one.

*Code:* `MethylPhys_Interface/globe_viewer.py` line 13 (10 lines) · arguments: spec · calls: not called on the traced runs

### `globe_viewer.py::dedupe`

Stores every picture of a spec once under a short name; layers refer to it by name (a reference is both a layer and a difference's gray).

*Code:* `MethylPhys_Interface/globe_viewer.py` line 24 (13 lines) · arguments: spec · calls: not called on the traced runs

### `globe_viewer.py::dedupe.one`

Gives one inlined picture its name, reusing the name if the same picture was already stored.

*Code:* `MethylPhys_Interface/globe_viewer.py` line 27 (4 lines) · arguments: v · calls: not called on the traced runs

### `globe_viewer.py::fragment`

The globe inside a report: one section with the controls, the viewer and the specs inlined, so the report stays one file that works offline.

*Code:* `MethylPhys_Interface/globe_viewer.py` line 118 (4 lines) · arguments: specs, size · calls: not called on the traced runs · constants read: `CONTROLS`, `VIEWER_JS`

### `globe_viewer.py::page`

A full dark page for one or more globes with a globe chooser: a report's own globe page, or the site's Globe Explorer.

*Code:* `MethylPhys_Interface/globe_viewer.py` line 123 (8 lines) · arguments: specs, title, back, size, intro · calls: not called on the traced runs · constants read: `CONTROLS`, `VIEWER_JS`

### `globe_viewer.py::write_files`

Writes a spec's inlined pictures as files for the site and returns the spec pointing at them, so each globe loads only when it is chosen.

*Code:* `MethylPhys_Interface/globe_viewer.py` line 132 (13 lines) · arguments: spec, d, url · calls: not called on the traced runs

### `globe_viewer.py::write_files.one`

Writes one inlined picture to its file and returns its address.

*Code:* `MethylPhys_Interface/globe_viewer.py` line 135 (4 lines) · arguments: v · calls: not called on the traced runs

### `report_globe_v4.py::_to_sky`

One value per site in genome order to a NESTED map: the mean per pixel, nothing where no site was measured (the atlas tools' own rule).

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 75 (5 lines) · arguments: v, npix · calls: not called on the traced runs

### `report_globe_v4.py::_mu`

Reads the atlas best values of the named cells from the compact copy (`atlas/v2.1/mu_part*.parquet`), 65535 decoded as no value.

**The change is read in the C-Score's blocks, beyond a fixed family-wise line.** Judged site by site, a change beyond 3 spreads lit 2.1 %
of a healthy blood's sites, spread over the whole globe (16,372 of 773,556), while a loss that runs along the genome (many small moves the same
way) passes 3 at hardly any single site. So both change layers read blocks of 10 neighboring measured sites in genome order, the C-Score's
block. On the whole genome each block's mean is taken in the sample's own block spread (neighboring CpGs move together, so a healthy block
spreads about twice as wide as independent sites would); on Met-A's 6,000 nearly independent sites the block mean is in standard errors. A
block has changed beyond the line pure chance crosses less than once in 20 genomes (5 % family-wise, two-sided; 4.98 for the genome's 77,355
blocks, 3.93 for Met-A's 597): FW_LINE. Healthy blood: 45 of 77,355 genome blocks and 0 of 597 Met-A blocks; the planted clustered example:
44 Met-A blocks, in clumps; 200 planted runs of 30 sites in a copy of the healthy blood: 229 of their 756 blocks found. The slider ends on
pure black, so only the changes stay lit.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 81 (12 lines) · arguments: cells, stem · calls: not called on the traced runs · constants read: `ATLAS`, `NA` · files named: `import pyarrow.parquet`

### `report_globe_v4.py::FW_LINE`

The fixed line for n blocks: the size a normal block crosses by chance less than once in 20 genomes, two-sided (5 % family-wise).

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 68 (4 lines) · arguments: n · calls: not called on the traced runs

### `report_globe_v4.py::_vmask`

A map's values (0 to 1) as a half-size 8-bit picture for the page: 0 where no site, 1 to 255 the value. With it every map layer has a slider
that keeps only the sites at least as methylated as the chosen value, and prints the share of the measured globe kept (2026-10-21; before,
the map layers had no tool). A browser that color-manages the picture may shift a code by a step or two, about 1 % of the scale at the
slider's line: a viewing tool, never a reading.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 113 (6 lines) · arguments: v · calls: not called on the traced runs

### `report_globe_v4.py::sex_sites`

Loads the frozen list of IAM-Atlas v2.1's sites on the X (14,759) and Y (3) chromosomes, `Runtime Matrices/Atlas_V21/sex_chromosome_sites_v21.json`,
built by `atlas/tools/sex_sites_v21.py` from the two Illumina manifests Stage 1 is pinned to (it refuses any other manifest).

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 40 (6 lines) · arguments: none · calls: not called on the traced runs · constants read: `SEX_SITES`, `_XY`

### `report_globe_v4.py::_xy_index`

The X and Y site names as a pandas Index, built once, so a block's sex-chromosome check is a hashed lookup (DEV-SPEED-01: np.isin on site names had taken about 10 s per call).

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 47 (4 lines) · arguments: none · calls: not called on the traced runs · constants read: `_XY`

### `report_globe_v4.py::_xy_blocks`

How many changed blocks hold at least one X or Y site. Every change count on the globe is printed with this count beside it.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 52 (6 lines) · arguments: ids_in_order, mi, nb, changed · calls: not called on the traced runs · constants read: `BLOCK`

### `report_globe_v4.py::key_css`

The parchment scale of the methylation maps as a color bar, for the color key under every map layer.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 59 (4 lines) · arguments: none · calls: not called on the traced runs · constants read: `HERE` · files named: `Planck_Parchment_RGB.txt`

### `report_globe_v4.py::build_pair`

DEVELOPMENT, not commissioned: one person's sample (a tumor) against the same person's own other sample (its adjacent normal) on the atlas's
layout, by the globe's block rule without the atlas: each site's difference in the pair's own spread, blocks of 10 neighboring sites, the
block mean in its own block spread, changed beyond the 5 % family-wise line. Run as `python3 report_globe_v4.py pair SAMPLE REFERENCE NAME
OUT.html` on chain Stage 1 values. A pair with no change lights 0 of 77,356 blocks; 200 planted regional runs of 30 sites light 620.
`drop` leaves listed sites out before anything is computed (the X and Y, so a difference of sex is not counted as a change). The record also
gives the changed blocks above and below the reference and the share of the globe's measured pixels the change lights.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 393 (44 lines) · arguments: beta, ref_beta, ref_name, out_html, title, drop · calls: not called on the traced runs · constants read: `ATLAS`, `BLOCK`, `HERE`, `MV`, `ORDER`, `PAIR_ABOUT` · files named: `Planck_Parchment_RGB.txt`, `import pyarrow.parquet`, `sites.parquet`

### `report_globe_v4.py::build_pair.rgb`

Draws a methylation map for the pair's globe on the atlas pages' parchment scale; gray where no site was measured.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 415 (2 lines) · arguments: m · calls: not called on the traced runs

### `report_globe_v4.py::build._code`

Codes a change map for the page: 0 where no site was measured, 128 where the sites are unchanged, and 128 plus or minus at least one step for a
changed site, its size up to a fixed full scale (10 spreads), so the page can fade the unchanged background without ever touching the change.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 247 (6 lines) · arguments: v, full · calls: not called on the traced runs

### `report_globe_v4.py::_to_sky_extreme`

One value per site to the sky map, keeping in each pixel the value of largest size with its sign, so a changed site is never averaged away by
an unchanged neighbor sharing its pixel.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 94 (6 lines) · arguments: v, npix · calls: not called on the traced runs

### `report_globe_v4.py::_lookup`

For each pixel of the 1024 × 512 texture, the nearest sky pixel center (the chain's `stage_v4.nest_pix2ang`).

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 101 (7 lines) · arguments: nside · calls: not called on the traced runs · constants read: `CH`

### `report_globe_v4.py::_png_b64`

A grayscale texture as base64 PNG, for the difference and departure codes (0 = no site; 1 to 255 = −0.5 to +0.5, or −5 to +5 spreads).

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 109 (3 lines) · arguments: arr, mode · calls: not called on the traced runs

### `report_globe_v4.py::_jpg_b64`

A color texture as base64 JPEG, for the sample and reference layers.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 121 (3 lines) · arguments: rgb · calls: not called on the traced runs

### `report_globe_v4.py::reference`

The healthy reference for a specimen: the matching atlas cell, or for blood the six panel cells weighted by the Stage D21 shares and scaled
to 1; refuses with the reason when there are no shares or no specimen.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 125 (13 lines) · arguments: specimen, shares · calls: not called on the traced runs · constants read: `BLOOD`, `PANEL`

### `report_globe_v4.py::_paint_blocks`

Drawing only. A changed block of 10 sites (the Met-A layer, each cell's layer) is painted over the stretch of the genome it covers, from its
first placed site to its last, at their places on the genome-order sky. Met-A's 6,000 sites lie far apart along the genome, so a block drawn as
10 dots landed as 10 specks in different places; painted over its stretch (a median of about 0.15 % of the sphere) a clustered loss shows as
patches where it sits on the genome. Which blocks changed, and every count, are untouched; a block whose sites lie far apart gives a large patch
from only 10 measured sites, and the layer's note says so.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 157 (14 lines) · arguments: sky, vals, posidx, npix, n · calls: not called on the traced runs

### `report_globe_v4.py::_patch`

Drawing only. On the whole-genome change layer a block of 10 neighboring sites covers only a few texture pixels; each changed pixel is drawn as a
patch of the dots' radius (R_DOT), the larger change winning where two meet. Unchanged and unmeasured pixels and every count are untouched.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 172 (8 lines) · arguments: v · calls: not called on the traced runs · constants read: `R_DOT`

### `report_globe_v4.py::_cell_change`

One cell's change layer on the globe (development, 2026-10-22): the departure behind Met-A for that cell
(`stage_metaa_atlas.departure_in_mix`), in the sample's own spread at that cell's identity sites, read in blocks of 10 of those sites in genome
order; a block changed beyond the 5 % family-wise line is lit, drawn as dots. Each cell the IAM-Deconvolver read gets one, in the group
"each cell's change". It shows a loss that clusters along the genome; a loss spread evenly over every site of a cell is canceled by the
sample's own centering and shows in that cell's Met-A number instead (the example blood with a 5 % even CD4 T loss: 2 of 597 blocks
before and after).

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 181 (19 lines) · arguments: d, ids_ordered, npix, look · calls: not called on the traced runs · constants read: `BLOCK`, `R_DOT`

### `report_globe_v4.py::build`

Checks the frozen layout against the atlas site list, builds the four layers, the shares above each size, and the page fragment (canvas,
controls and script, no outside library). Returns the record kept in the bundle, marked `scored: False`.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 201 (176 lines) · arguments: beta, specimen, shares, out_html, z, fractions, sex, cell_departures · calls: not called on the traced runs · constants read: `ABOUT`, `ATLAS`, `BLOCK`, `CH`, `HERE`, `MV`, `ORDER`, `PANEL`, `R_DOT`, `SEXNOTE`, `SEXWORD` · files named: `Planck_Parchment_RGB.txt`, `import pyarrow.parquet`, `neutrophil_reference_v1_2.json`, `sites.parquet`, `sky_order_v21.npz`

### `report_globe_v4.py::build.rgb`

Colors a sky map for the texture with the atlas pages' parchment scale, 0 to 1; gray where no site was measured.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 244 (2 lines) · arguments: m · calls: not called on the traced runs

## 15. The development readings: `dev_stages.py`, `dev_comp_methods.py`, `deconv_v2.py`, `sky_statistics.py`

**What they are for.** Readings under development, run only when a `--dev-*` flag asks for them. Each writes its own record under
`bundle["development"]`, labeled "DEVELOPMENT - not commissioned", and none changes the reading, the gauge or the tare. They are in the
chain so a development result is made by the same code a researcher can run. The one exception is `dev_stages.selftare_map` (section 5),
which has been Stage T step 1 on every reading since 2026-10-04 and still lives in this file.

**Labels.** Direction, brightness and the per-cell B reading follow Met-A's physics; the composition methods and the sky statistics
are standard methods (deconvolution, the angular power spectrum of a sphere with a shuffled null).

### `dev_stages.py::_H`

The entropy of a beta value in bits.

*Code:* `dev_stages.py` line 33 (3 lines) · arguments: b · calls: not called on the traced runs

### `dev_stages.py::_bc`

Loads the blood composition matrix once.

*Code:* `dev_stages.py` line 47 (3 lines) · arguments: none · calls: not called on the traced runs · constants read: `RM`, `_C` · files named: `blood_composition_EPIC_v1.json`

### `dev_stages.py::_rec`

A development record with its label.

*Code:* `dev_stages.py` line 52 (2 lines) · arguments: stage · calls: not called on the traced runs · constants read: `DEV_LABEL`

### `dev_stages.py::selftare_ii`

The development form of self-tare II: Met-A re-read on the mapped values beside the reading. Kept for its earlier records; the one command
no longer calls it, because step 1 now runs on every reading.

*Code:* `dev_stages.py` line 60 (9 lines) · arguments: beta, specimen · calls: not called on the traced runs

### `dev_stages.py::anchor_range`

Development (DEV-SELFTARE-03, preliminary): the span between the array's type II fixed-site anchors, R = U − L, the number the self-tare
divides by. A loss of the neutrophils' pattern that reaches every site is mapped away by the self-tare but narrows this span: tared against at
least 3 same-run healthy references, R_rel = R / median(R_refs), and the loss is (1 − R_rel) / f_neu with a limit of 2 × the references' spread /
f_neu. On synthetic bloods planted 1, 2 and 5 % read back as 0.97, 1.95 and 4.94 %. Not called by the one command yet; changes no reading.

*Code:* `dev_stages.py` line 72 (18 lines) · arguments: beta, f_neu, refs, st · calls: not called on the traced runs · files named: `dev_selftare_typeII_EPIC_v1.json`

### `dev_stages.py::site_uncertainty`

This array's own per-site uncertainty: the SD of β against the reference value over its fixed sites, per design and state.

*Code:* `dev_stages.py` line 92 (8 lines) · arguments: beta, st · calls: not called on the traced runs · files named: `dev_selftare_typeII_EPIC_v1.json`

### `dev_stages.py::s_for`

The per-site uncertainty for a list of sites, by design and state (low when the reference value is below 0.5).

*Code:* `dev_stages.py` line 102 (6 lines) · arguments: sites, ref_values, su, st · calls: not called on the traced runs · files named: `dev_selftare_typeII_EPIC_v1.json`

### `dev_stages.py::direction`

D = mean over the identity sites of |r − 0.5| − |β − 0.5|: above 0, the sites have moved toward 0.5, toward disorder; below 0, away from it.

*Code:* `dev_stages.py` line 111 (6 lines) · arguments: beta, ref, sites · calls: not called on the traced runs

### `dev_stages.py::direction_record`

The direction record at the neutrophil identity sites against the cell's own reference (the expectation in whole blood), tared on the
same-run references' D.

*Code:* `dev_stages.py` line 119 (15 lines) · arguments: beta, specimen, comp, refs_D · calls: not called on the traced runs

### `dev_stages.py::trace_cell`

In an isolated-neutrophil specimen, another blood group's contamination, called above 3 × this array's own standard error.

*Code:* `dev_stages.py` line 137 (15 lines) · arguments: beta, su, line · calls: not called on the traced runs · files named: `dev_selftare_typeII_EPIC_v1.json`

### `dev_stages.py::foreign_cell`

In whole blood, a non-blood template solved beside the 8 blood groups, called above 3 × this array's own standard error.

*Code:* `dev_stages.py` line 155 (22 lines) · arguments: beta, template, su, line · calls: not called on the traced runs · files named: `dev_foreign_placenta_EPIC_v1.json`, `dev_selftare_typeII_EPIC_v1.json`

### `dev_stages.py::brightness`

A 95 % interval on Met-A by Monte Carlo from this array's per-site uncertainty.

*Code:* `dev_stages.py` line 180 (17 lines) · arguments: beta, specimen, comp, n_draw, su, seed · calls: not called on the traced runs · files named: `dev_selftare_typeII_EPIC_v1.json`

### `dev_stages.py::_mapping`

Loads the CpG-to-HEALPix map of the development sky (NSIDE 128).

*Code:* `dev_stages.py` line 200 (6 lines) · arguments: none · calls: not called on the traced runs · constants read: `RM`, `_C` · files named: `iamatlas_cpg_to_healpix_nside128.npz`

### `dev_stages.py::residual_z`

The development sky's per-site departure: z = (β − Σ f_g μ_g) / √(Σ f_g² (sd² + s_bp²) + 0.02²), with s_bp the atlas's between-person SD.

*Code:* `dev_stages.py` line 208 (6 lines) · arguments: beta, fractions, AP · calls: not called on the traced runs · constants read: `PARENTS`

### `dev_stages.py::_pixmap`

The mean departure in each pixel.

*Code:* `dev_stages.py` line 216 (3 lines) · arguments: zs, pix, npix · calls: not called on the traced runs

### `dev_stages.py::block_shuffle_order`

A permutation that moves whole runs of consecutive sites within each chromosome: the null keeps short-range structure along the genome and
breaks where it sits.

*Code:* `dev_stages.py` line 221 (7 lines) · arguments: chr_keys, run, rng · calls: not called on the traced runs · constants read: `SKY_RUN`

### `dev_stages.py::sky_from_z`

The development sky's band powers against the block-shuffle null and the look-elsewhere statistic by simulation.

*Code:* `dev_stages.py` line 230 (14 lines) · arguments: z, n_null, n_lee, seed, free_null · calls: not called on the traced runs · constants read: `SKY_LEE_N`, `SKY_NULL_N`, `SKY_RUN`

### `dev_stages.py::sky_from_z.shuf`

The band powers of one block-shuffled sky.

*Code:* `dev_stages.py` line 235 (2 lines) · arguments: none · calls: not called on the traced runs

### `dev_stages.py::_has_cols`

Whether a parquet file holds the named columns.

*Code:* `dev_stages.py` line 246 (4 lines) · arguments: path, cols · calls: not called on the traced runs · files named: `import pyarrow.parquet`

### `dev_stages.py::load_atlas_parents`

Loads the atlas v2 parent blood cells' means, SDs and between-person SDs.

*Code:* `dev_stages.py` line 252 (8 lines) · arguments: atlas_parquet · calls: not called on the traced runs · constants read: `PARENTS`, `_C`

### `dev_stages.py::sky`

The development residual sky of one whole blood, with its null.

*Code:* `dev_stages.py` line 262 (7 lines) · arguments: beta, fractions, atlas_parquet · calls: not called on the traced runs

### `dev_stages.py::_atlas_e`

Builds the atlas_e solver once on the 12 array-measured circulating cells of atlas v2, with the neutrophil identity sites removed.

*Code:* `dev_stages.py` line 276 (9 lines) · arguments: atlas_parquet · calls: not called on the traced runs · constants read: `ATLAS_E_CELLS`, `_C`

### `dev_stages.py::atlas_e`

The atlas_e composition, summed into the 8 blood groups.

*Code:* `dev_stages.py` line 287 (5 lines) · arguments: beta, atlas_parquet · calls: not called on the traced runs · constants read: `BLOOD8`

### `dev_stages.py::nilc_e`

The NILC-e composition on the atlas_e templates, with each share's noise.

*Code:* `dev_stages.py` line 294 (5 lines) · arguments: beta, atlas_parquet · calls: not called on the traced runs · constants read: `BLOOD8`

### `dev_stages.py::percell_b`

A B-cell Met-A against the expectation for this blood's mix, on the development B-cell reference.

*Code:* `dev_stages.py` line 301 (13 lines) · arguments: beta, fractions, atlas_parquet · calls: not called on the traced runs · constants read: `PARENTS`, `RM` · files named: `metA_floors_v1_2_ALLCELLS_development.json`

### `dev_stages.py::epicv2_calibrate`

An EPIC v2 IDAT pair through SeSAMe (an R package, run in its own environment) to β values on EPIC v1 probe names, replicate probes of one
CpG averaged and background-masked probes dropped. Development only: the chain's references are EPIC v1.

*Code:* `dev_stages.py` line 320 (19 lines) · arguments: grn, red, rscript, prep · calls: not called on the traced runs · constants read: `SESAME_R` · files named: `betas.csv`

### `dev_comp_methods.py::load_atlas`

Loads an atlas parquet file indexed by site.

*Code:* `dev_comp_methods.py` line 46 (4 lines) · arguments: path · calls: not called on the traced runs

### `dev_comp_methods.py::atlas_cells`

The cell names in an atlas table.

*Code:* `dev_comp_methods.py` line 52 (2 lines) · arguments: A · calls: not called on the traced runs

### `dev_comp_methods.py::subset_atlas`

The columns of the named cells.

*Code:* `dev_comp_methods.py` line 56 (3 lines) · arguments: A, keep · calls: not called on the traced runs

### `dev_comp_methods.py::build_deconv`

An atlas v2 solver on the named cells, with the neutrophil identity sites removed from its pool.

*Code:* `dev_comp_methods.py` line 61 (4 lines) · arguments: A, keep, exclude_sites · calls: not called on the traced runs · constants read: `POOL1`, `SOLVER`

### `dev_comp_methods.py::twin_merge`

The atlas's own twin rule: a cell with no array measurement whose marker profile correlates with an array-measured cell above the
threshold is that cell on another platform, and the two merge.

*Code:* `dev_comp_methods.py` line 67 (28 lines) · arguments: A, D, roster, twin_tab, thr, candidates · calls: not called on the traced runs · files named: `records/10_twin_test_sample_level.csv`, `twin_family_thresholds_v1.json`

### `dev_comp_methods.py::mixture_twin`

A cell whose template is reproduced by a non-negative, sum-to-1 mix of the other cells, above the twin correlation, cannot be told apart
from that mix in a specimen.

*Code:* `dev_comp_methods.py` line 97 (18 lines) · arguments: D, candidates, r_thr · calls: not called on the traced runs

### `dev_comp_methods.py::crlb`

The Cramér–Rao bound on each share's SD for a solver's design at its markers.

*Code:* `dev_comp_methods.py` line 117 (10 lines) · arguments: D, f, sigma · calls: not called on the traced runs

### `dev_comp_methods.py::NNLS8.__init__`

Stage A's solver as a class: the 963 markers and 8 groups.

*Code:* `dev_comp_methods.py` line 130 (2 lines) · arguments: self, bc · calls: not called on the traced runs

### `dev_comp_methods.py::NNLS8.deconvolve`

Stage A's shares, for comparison with the other methods.

*Code:* `dev_comp_methods.py` line 132 (4 lines) · arguments: self, beta · calls: not called on the traced runs

### `dev_comp_methods.py::ledoit_wolf`

Analytic Ledoit–Wolf shrinkage of a covariance toward a multiple of the identity; no parameter to choose.

*Code:* `dev_comp_methods.py` line 138 (7 lines) · arguments: X · calls: not called on the traced runs

### `dev_comp_methods.py::NILC.__init__`

The NILC solver: sites, cells, means, variances and the noise model.

*Code:* `dev_comp_methods.py` line 150 (4 lines) · arguments: self, sites, cells, mu, v, cov, sigma, tech_var, tech_cov, offset · calls: not called on the traced runs

### `dev_comp_methods.py::NILC._solve`

One weighted, constrained solve.

*Code:* `dev_comp_methods.py` line 154 (11 lines) · arguments: self, m, f · calls: not called on the traced runs

### `dev_comp_methods.py::NILC.deconvolve`

The NILC shares in two passes.

*Code:* `dev_comp_methods.py` line 165 (9 lines) · arguments: self, beta, passes · calls: not called on the traced runs

### `dev_comp_methods.py::to_groups`

Sums atlas cells into the 8 blood groups.

*Code:* `dev_comp_methods.py` line 176 (7 lines) · arguments: fr · calls: not called on the traced runs · constants read: `SUBGROUP`

### `dev_comp_methods.py::met_a`

Whole-blood Met-A with another method's shares and profiles, for comparison with Stage M.

*Code:* `dev_comp_methods.py` line 185 (8 lines) · arguments: beta, fractions, profiles, ns · calls: not called on the traced runs · constants read: `H`

### `dev_comp_methods.py::ceiling_flag`

The entropy-ceiling flag for that comparison reading.

*Code:* `dev_comp_methods.py` line 195 (16 lines) · arguments: beta, profiles, fractions, ns · calls: not called on the traced runs

### `dev_comp_methods.py::read_beta`

Reads one β file.

*Code:* `dev_comp_methods.py` line 213 (3 lines) · arguments: path · calls: not called on the traced runs

### `deconv_v2.py::DeconvV2.__init__`

The atlas v2 solver: loci measured by every modeled cell; for each cell, markers chosen against its nearest other cell (margin at least
0.20, the top 200 by margin over pooled SD); variance v = μ_sd² + donor_sd².

*Code:* `deconv_v2.py` line 11 (34 lines) · arguments: self, atlas_parquet, pooled_one_sample, margin, top, min_markers, sigma, drop, markers, k_near, per_pair, pair_margin, A · calls: not called on the traced runs

### `deconv_v2.py::DeconvV2._solve`

One joint solve with f ≥ 0 and Σ f = 1, weights 1 / (Σ_c f_c² v_ci + σ²) iterated 5 times.

*Code:* `deconv_v2.py` line 46 (7 lines) · arguments: self, y, M, V · calls: not called on the traced runs

### `deconv_v2.py::DeconvV2.deconvolve`

The shares, and each cell's presence by 200 bootstrap resamples of the markers.

*Code:* `deconv_v2.py` line 54 (20 lines) · arguments: self, beta, n_boot, seed, present_floor, affine · calls: not called on the traced runs

### `sky_statistics.py::bandpowers`

The mean power in each of six bands of the angular spectrum.

*Code:* `sky_statistics.py` line 30 (3 lines) · arguments: cl, bands · calls: not called on the traced runs · constants read: `BANDS`

### `sky_statistics.py::apodised_mask`

A pixel weight that rises smoothly from 0 at the edge of the measured footprint to 1 a set angle inside it, so a sharp edge does not
spread power between multipoles.

*Code:* `sky_statistics.py` line 35 (20 lines) · arguments: good, apod_deg, taper · calls: not called on the traced runs

### `sky_statistics.py::_weights`

Keeps the last mask's weights.

*Code:* `sky_statistics.py` line 57 (5 lines) · arguments: good, apod_deg, taper · calls: not called on the traced runs · constants read: `_W`

### `sky_statistics.py::masked_spectrum`

The pseudo angular power spectrum of a sky with masked pixels, with a hard or apodized mask.

*Code:* `sky_statistics.py` line 64 (13 lines) · arguments: pix, lmax, mask, apod_deg, taper · calls: not called on the traced runs · constants read: `MASKS`

### `sky_statistics.py::shuffled_null`

The band powers of skies made by shuffling the sky's own measured pixel values among its measured pixels: the mask, the count and the
distribution are kept, and only the arrangement is destroyed, so a difference from this null is spatial structure.

*Code:* `sky_statistics.py` line 79 (16 lines) · arguments: pix, n_perm, lmax, rng, bands, mask, apod_deg, taper · calls: not called on the traced runs · constants read: `BANDS`, `MASKS`

### `sky_statistics.py::sky_spectrum_record`

One sky's spectrum, band powers, measured fraction, share of pixels with |z| above 2, and its null.

*Code:* `sky_statistics.py` line 97 (11 lines) · arguments: pix, n_perm, lmax, rng, mask, apod_deg, taper · calls: not called on the traced runs · constants read: `BANDS`

## 16. Met-A against IAM-Atlas v2.1 (development): `stage_metaa_atlas.py`

**What it is for.** Reading an isolated cell of any type the atlas holds by Met-A, against that cell in IAM-Atlas v2.1 instead of against
purified arrays. Run by the one command with `--dev-metaa-atlas CELL`, and on its own when intake accepts an isolated, sorted or purified atlas
cell (Section 1, `stage_0_intake.atlas_cell_of`). Development, not commissioned: written under `development.metaa_atlas`.

**Label: from IAM's physics.** The same reading as Met-A: mean entropy H(β) at the cell's identity sites over the entropy the reference
predicts there. A held site sits near 0 or 1; a loss moves it toward 0.5, where H is largest, so a loss raises the reading.

**Label: standard.** The identity sites follow Met-A's own site rule, applied to the atlas: healthy width at most 0.05 (the person-to-person
spread combined with the atlas's own uncertainty, half its 95 % range over 1.96), and a best value from 0.75 to 0.95 (methylated side) or 0.05
to 0.25 (unmethylated side); where more than 3,000 qualify on a side, the 3,000 of smallest width. A reading needs at least 90 % of them
measured. The tare is a median of at least 3 same-run healthy references, as Stage T step 2; never the sample itself.

**The territories, read on every run.** Besides each cell's identity sites, this stage reads two fixed site sets the identity readings do not
reach (`territory_sets`, `territory_reading`): the islands, sites every healthy cell type keeps unmarked, and the domains, marked sites some
healthy cell types hold loosely, each on the A-score gauge against IAM-Atlas v2.1 at each site's own read count. In development: read and
flagged on arrays and sequencing (INSTRUMENTS.md, In development).

### `report_v4.py::_fam_of`

Each atlas cell type's family (stage_info_sphere.FAMILIES), for the report's unexpected-cell note.

*Code:* `MethylPhys_Interface/report_v4.py` line 294 (5 lines) · arguments: none · calls: not called on the traced runs

### `report_v4.py::detail_tables`

The one source of the report's instrument sections, drawn by the HTML report and the researcher PDF alike: for each instrument (atlas and ruler, unexpected cells, Met-A whole sample, Met-A for each cell, commissioned Met-A, territories, the four C-Scores, IAM-A, Group-A, the globe, the information sphere) a title, what it reads, a table, and a note that says the state or why it did not fire. Every instrument always appears.

*Code:* `MethylPhys_Interface/report_v4.py` line 302 (84 lines) · arguments: o · calls: not called on the traced runs · constants read: `BLOOD_FAMILIES`, `DEV`

### `report_v4.py::_f`

Prints a number to a fixed number of decimals, or the value as text.

*Code:* `MethylPhys_Interface/report_v4.py` line 299 (1 lines) · arguments: v, d · calls: not called on the traced runs

### `report_v4.py::report_text`

Loads the report's explanations from Runtime Matrices/Report/report_text.json, which CANON/report_text.py generates from CANON/catalogue.json (the one source); read once per run.

*Code:* `MethylPhys_Interface/report_v4.py` line 391 (4 lines) · arguments: none · calls: not called on the traced runs · constants read: `REPORT_TEXT`, `_RT` · files named: `generated from CANON/catalogue.json`

### `report_v4.py::explain`

The explanation of one report section as (label, text) pairs, in a fixed order: what it measures, where it comes from, what it is compared to, the physics, why it matters, the link to the sky where there is one, and its status.

*Code:* `MethylPhys_Interface/report_v4.py` line 395 (4 lines) · arguments: sec · calls: not called on the traced runs · constants read: `EXPLAIN_LABELS`

### `report_v4.py::_explain_html`

Draws one section's explanation as a shaded block under its table in the HTML report.

*Code:* `MethylPhys_Interface/report_v4.py` line 399 (3 lines) · arguments: sec · calls: not called on the traced runs · constants read: `E`

### `report_v4.py::_intro_html`

Draws the report's 'How to read this report' section (the law, the ratio to healthy, the person's own reference, the methods' origin in sky mapping, and what each status means) from the same source.

*Code:* `MethylPhys_Interface/report_v4.py` line 402 (3 lines) · arguments: none · calls: not called on the traced runs · constants read: `E`

### `report_pdf_v4.py::_explain`

Draws one section's explanation in the researcher PDF, from the same source as the HTML report.

*Code:* `MethylPhys_Interface/report_pdf_v4.py` line 50 (4 lines) · arguments: R, sec · calls: not called on the traced runs

### `report_v4.py::detail_tables._cl`

Adds one row to the distance-to-the-ceiling table for a reading that carries its ceiling record: the reading, its saturation, its distance to the ceiling, its turnover share, and whether its methylated sites have passed 0.5.

*Code:* `MethylPhys_Interface/report_v4.py` line 342 (2 lines) · arguments: name, A, c · calls: not called on the traced runs

### `report_v4.py::_sections_v5`

All the new sections in their order, after the cell shares.

*Code:* `MethylPhys_Interface/report_v4.py` line 406 (8 lines) · arguments: o · calls: not called on the traced runs · constants read: `E`

### `stage_t_selftare.py::ruler_map`

The atlas ruler: the self-tare's straight line per probe design onto IAM-Atlas v2.1's means at sites every cell type holds alike (islands and shores never anchors). Development; off unless run_sample.py is given --dev-ruler.

*Code:* `stage_t_selftare.py` line 32 (8 lines) · arguments: beta · calls: not called on the traced runs · constants read: `RULER`, `_C` · files named: `Runtime Matrices/Stage_T/atlas_ruler_v1.json`

### `stage_metaa_atlas.py::cscore_limits`

Adds each C-Score reading's limit, whether it sits within it, and the rule used. Each cell: the derived limit. The whole genome: the derived limit, times the 450K platform term on a 450K array (measured once on spent healthy 450K bloods and frozen, Runtime Matrices/CScore_Limits/platform_term_450k.json; calibrated).

*Code:* `stage_metaa_atlas.py` line 529 (13 lines) · arguments: cs, array_type · calls: not called on the traced runs · constants read: `HERE` · files named: `platform_term_450k.json`

### `stage_metaa_atlas.py::cscore_limit`

The C-Score's derived healthy upper limit (development): with departures independent from site to site, C over nb = n // 10 blocks follows a scaled Beta((nb - 1) / 2, (n - nb) / 2); the limit is its 95th percentile. Nothing is measured.

*Code:* `stage_metaa_atlas.py` line 524 (4 lines) · arguments: n_sites, q, w · calls: not called on the traced runs · constants read: `CS_BLOCK`

### `stage_metaa_atlas.py::read_on_curve`

Met-A with IAM-Atlas v2.1 as the ruler on the array's own response curve (plan 14.14, development): identity sites chosen on the atlas; the array's mean reading and its spread at each atlas level, measured at every other site it read; its healthy mark is the expected entropy at the identity sites of a reading on that curve with that spread; Met-A is its own mean entropy there over the mark. No cohort and no same-run references; also returns how far the array reads above its own curve at the identity sites.

*Code:* `stage_metaa_atlas.py` line 139 (28 lines) · arguments: beta, cell · calls: not called on the traced runs · constants read: `CURVE_EDGES`, `FLAG`, `RULE`

### `stage_metaa_atlas.py::_he`

The healthy expectation of disorder at each site: on sequencing, the expected entropy of k of n reads at that site's own read count (expected_H_counts), so a site read from one molecule, which can only read 0 or 1, is not taken as order; on arrays, the entropy of the expected level.

*Code:* `stage_metaa_atlas.py` line 433 (3 lines) · arguments: es, counts · calls: not called on the traced runs

### `stage_metaa_atlas.py::_ruler`

Loads the atlas ruler's frozen file once: every site's probe design (I or II), the anchor sites per design (held below about 0.05 or above about 0.95 in every one of the 79 cell types) and the atlas's values there.

*Code:* `stage_metaa_atlas.py` line 440 (4 lines) · arguments: none · calls: not called on the traced runs · constants read: `RULER`

### `stage_metaa_atlas.py::array_blur`

Measures the array's own blur, per probe design, at its anchor sites: the straight line x = a + b e through the means of its low and high anchors against the atlas's means there, and its noise, 1.4826 times the median absolute residual from that line. Nothing is fitted to any reading; these are measurements of the array where every healthy cell agrees (DEV-METAA-HEALTHY-01).

*Code:* `stage_metaa_atlas.py` line 445 (14 lines) · arguments: xs, es · calls: not called on the traced runs

### `stage_metaa_atlas.py::predicted_healthy`

Met-A's healthy middle line predicted from the array itself: at each site, the disorder this array would show if the sample were exactly the atlas in its own mix, E[H(clip(a + b e + noise, 0, 1))] over the array's measured noise by 20-point Gauss-Hermite quadrature; A_pred = sum w E[H] / sum w H(e) with the whole-sample reading's weights. A healthy sample should read A / A_pred = 1, with no healthy cohort and no reference arrays. Blind spot, stated: a loss spread evenly over every site, the anchors included, looks like a weaker array (DEV-METAA-HEALTHY-01).

*Code:* `stage_metaa_atlas.py` line 460 (16 lines) · arguments: xs, es, w, blur · calls: not called on the traced runs · constants read: `GH_NODES`

### `stage_metaa_atlas.py::read_whole`

Met-A for the whole sample over every atlas site: the atlas mixed in the sample's own shares is the expectation, each site counts by how firmly that mix holds it (|0.5 - e|), and the reading is weighted disorder over weighted expected disorder. Reported overall and split by place (identity sites of the cells found, islands, domains, the rest), with the shift a 1 % loss would make. The per-cell readings say which cell; this one says whether the sample has departed. Development, flagged, on by default.

*Code:* `stage_metaa_atlas.py` line 477 (38 lines) · arguments: beta, shares, counts · calls: not called on the traced runs · constants read: `FLAG`, `MIX_COVER`

### `stage_metaa_atlas.py::read_whole.one`

The weighted reading over one group of sites (all, or one place), with its standard error over sites; a group with fewer than 500 sites is not read.

*Code:* `stage_metaa_atlas.py` line 495 (6 lines) · arguments: m · calls: not called on the traced runs

### `stage_metaa_atlas.py::territory_sets`

The two sets where early cancer is known to act, by fixed rules on IAM-Atlas v2.1 over the sites measured in all 79 cell types: islands (every cell type below 0.2) and loosely held domains (mean at least 0.75, none below 0.2, at least one between 0.2 and 0.7). Read column by column so memory stays near one column.

*Code:* `stage_metaa_atlas.py` line 361 (17 lines) · arguments: none · calls: not called on the traced runs · constants read: `ATLAS`, `NA` · files named: `import pyarrow.parquet`, `sites.parquet`

### `stage_metaa_atlas.py::ceiling_reading`

Distance to the entropy ceiling (CAN-005), derived and unfitted: a reading's saturation A_sat = sum w / sum w H(e) (every site at 1 bit), the distance (A_sat - A) / (A_sat - 1) (1 at the healthy expectation, 0 at the ceiling), and the turnover share, the weighted share of sites whose level has crossed 0.5 to the far side of their healthy state, past which the reading falls as loss continues. Attached to Met-A for the whole sample (and each place), each cell's Met-A, and each territory.

*Code:* `stage_metaa_atlas.py` line 379 (21 lines) · arguments: xs, es, w · calls: not called on the traced runs · files named: `CANON/foundation.json`

### `stage_metaa_atlas.py::territory_reading`

The sample's disorder at each set against the atlas mixed in its own shares: mean H(beta) over mean H(e), the mean level change, the standard error over sites, and how much of the set the array measures. Development, flagged.

*Code:* `stage_metaa_atlas.py` line 401 (21 lines) · arguments: beta, shares, counts · calls: not called on the traced runs · constants read: `FLAG`, `MIX_COVER`, `TERR_RULES`

### `stage_metaa_atlas.py::plant_territory`

Tests only: a constructed gain at a seeded fraction of the islands, or loss at the domains, so a test of the territory readings runs through the one command.

*Code:* `stage_metaa_atlas.py` line 423 (7 lines) · arguments: beta, name, frac, size, seed · calls: not called on the traced runs

### `stage_metaa_atlas.py::read_in_mix_allsites`

Met-A for each cell over every IAM-Atlas v2.1 site (development; `run_sample.py --dev-metaa-allsites`): each site counts by the shift a loss
of that cell would make there, share × |0.5 − μ| (the chain's own shift rule), so nothing is chosen. Records the reading, its shift per 1 %
loss, its standard error over sites and its effective number of sites. The standard error is within one sample; how much healthy people
differ is measured on many healthy samples, not here.

*Code:* `stage_metaa_atlas.py` line 271 (30 lines) · arguments: beta, shares · calls: not called on the traced runs · constants read: `FLAG`, `MIX_COVER`

### `stage_metaa_atlas.py::departure_in_mix`

The per-site departure behind Met-A for one cell of a mixed sample: at the cell's identity sites,
H(β) − H(e), e the atlas mixed in the sample's own shares by `read_in_mix`'s rule; by site id. Used by the globe's layer for each cell.

*Code:* `stage_metaa_atlas.py` line 257 (12 lines) · arguments: beta, shares, cell · calls: not called on the traced runs · constants read: `MIX_COVER`

### `stage_metaa_atlas.py::_H`

The entropy of a beta value in bits, clipped to [10⁻⁶, 1 − 10⁻⁶].

*Code:* `stage_metaa_atlas.py` line 32 (2 lines) · arguments: b · calls: not called on the traced runs

### `stage_metaa_atlas.py::_col`

One cell's atlas column (best value, the ends of its 95 % range, or the person-to-person spread) over the atlas's sites, read once.

*Code:* `stage_metaa_atlas.py` line 35 (12 lines) · arguments: stem, cell · calls: not called on the traced runs · constants read: `ATLAS`, `NA` · files named: `import pyarrow.parquet`, `sites.parquet`

### `stage_metaa_atlas.py::width`

Each site's healthy width: the person-to-person spread and the atlas's uncertainty (half the 95 % range over 1.96), combined in quadrature.

*Code:* `stage_metaa_atlas.py` line 48 (4 lines) · arguments: cell · calls: not called on the traced runs

### `stage_metaa_atlas.py::identity_sites`

The cell's identity sites by the rule above (up to 3,000 per side): methylated side, then unmethylated side.

*Code:* `stage_metaa_atlas.py` line 53 (10 lines) · arguments: cell · calls: not called on the traced runs · constants read: `RULE`

### `stage_metaa_atlas.py::expected_H_counts`

**Label: from IAM's physics (counting).** On sequencing, a site's level is k of the n reads that cover it. With few reads the level is 0 or 1
far more often than the true level p, and the disorder of 0 or 1 is zero, so a healthy site's measured disorder at depth n is not H(p) but
E[H(k/n)] = Σₖ C(n,k) pᵏ(1−p)ⁿ⁻ᵏ H(k/n). Met-A's healthy expectation on sequencing values is this sum at each site's own read count, with the
atlas value as p; nothing is fitted. Checked against a 200,000-draw simulation (2026-10-22). Above 400 reads H(p) is used (the bias there is
negligible). It was added after the depth test (DEV-ONE-DRAW-01 exploration 1) found Met-A on sequencing values reading lower as depth fell.

*Code:* `stage_metaa_atlas.py` line 86 (11 lines) · arguments: p, n · calls: not called on the traced runs · constants read: `N_EXACT`

### `stage_metaa_atlas.py::read`

Met-A against the atlas cell, untared, on the values given (the one command passes the self-tared values), with the shift a 1 % loss would
make; refuses, with the reason, a cell with no qualifying site or a sample measuring under 90 % of them.

Each reading also records `levels_at_identity`: the sample's and the atlas column's median level at the column's methylated and
unmethylated identity sites (recorded only, never used in A), to measure whether the atlas's levels sit nearer the middle than sequencing.

On sequencing (read counts given), the atlas column is first put in sequencing units by the atlas's own measured source term for the
published sequencing atlas (`Runtime Matrices/Atlas_V21/sequencing_term_v1.json`: sequencing = -0.0902 + 1.1026 x atlas level, clipped to
0 to 1): the atlas is on the array scale, whose unmarked sites sit near 0.055 where sequencing reads 0. Arrays are read unchanged.

*Code:* `stage_metaa_atlas.py` line 98 (38 lines) · arguments: beta, cell, specimen, counts, seqref, half · calls: not called on the traced runs · constants read: `FLAG`, `RULE`

### `stage_metaa_atlas.py::tare`

A_rel = A over the median of at least 3 same-run healthy references (never the sample), the references' spread and the detection limit.

*Code:* `stage_metaa_atlas.py` line 168 (10 lines) · arguments: A, refs · calls: not called on the traced runs · constants read: `FLAG`, `RULE`

### `stage_metaa_atlas.py::specific_sites`

DEVELOPMENT (to-do 39.9): Met-A's site rule plus one condition, that the cell's atlas value differs by at least a stated amount from every other
cell of the panel, then the tightest 3,000 a side. The count is recorded and never padded: CD4 and CD8 T cells hold nearly the same sites (at a
difference of 0.2 they keep 66 and 25 of 6,000). `read_in_mix(..., specific=DELTA)` reads on these sites (`run_sample.py --dev-metaa-specific`).

*Code:* `stage_metaa_atlas.py` line 64 (14 lines) · arguments: cell, others, delta · calls: not called on the traced runs · constants read: `RULE`

### `stage_metaa_atlas.py::read_in_mix`

Met-A for each cell of a whole blood (development, flagged in the report as preliminary until commissioned): for each cell the IAM-Deconvolver
read, its identity sites by Met-A's rule on the atlas; the healthy expectation at those sites, IAM-Atlas v2.1's best values mixed in this
person's own cell shares; A = mean entropy of the sample over mean entropy of the expectation; and the shift a 1 % loss of that cell's pattern
would make at this person's share of it, so a cell present in small numbers shows a small shift. Read on the values Stage T step 1 maps, as
the commissioned Met-A is; untared. Known from exploration (GRP-014): a loss at every one of a cell's sites is partly taken up by the
Deconvolver's share of that cell, so the reading sees less loss than was planted.

On a wide panel (`--cell-panel atlas`) the expectation at each site is mixed over the cells that have an atlas value there, renormalized, and a
site counts when those cells hold at least `MIX_COVER` (99 %) of the sample's shares: before this rule (2026-10-22), one small cell without a
value dropped the site, and on the 79-cell panel most cells lost about half their identity sites.

*Code:* `stage_metaa_atlas.py` line 211 (44 lines) · arguments: beta, shares, counts, specific · calls: not called on the traced runs · constants read: `FLAG`, `MIX_COVER`, `NOT_FOUND_SHARE`, `READ_COVER_MIN`, `READ_COVER_MIN_SEQ`, `RULE`

### `stage_metaa_atlas.py::plant_all`

TESTS ONLY: a constructed loss of a cell's whole pattern, every atlas site the array measured where the cell has a value moved the given share of
the way to 0.5 (`run_sample.py --dev-plant CELL:LOSS:all`). A real loss of hold reaches every held site, not only the identity sites.

*Code:* `stage_metaa_atlas.py` line 179 (5 lines) · arguments: beta, cell, loss · calls: not called on the traced runs

### `stage_metaa_atlas.py::plant_runs`

TESTS ONLY: a constructed clustered loss, 10 runs of 30 neighboring identity sites of a cell in genome order (the frozen sky order), chosen by
a seeded draw, each moved the given share of the way to 0.5 (`run_sample.py --dev-plant CELL:LOSS:runs`). The shape the globe's block rule is
built to see.

*Code:* `stage_metaa_atlas.py` line 186 (13 lines) · arguments: beta, cell, loss, seed · calls: not called on the traced runs · constants read: `ATLAS`, `HERE` · files named: `chain/Runtime Matrices/Atlas_V21/sky_order_v21.npz`, `import pyarrow.parquet`, `sites.parquet`, `sky_order_v21.npz`

### `stage_metaa_atlas.py::plant`

A constructed loss for tests: every identity site the array measured moved 1 % of the way to 0.5, the shift `read` predicts. Used by the
series runner's `--plant metaa_atlas`; never by a reading.

*Code:* `stage_metaa_atlas.py` line 200 (5 lines) · arguments: beta, cell, loss · calls: not called on the traced runs

## 17. The whole-genome globe (development): `stage_q2_globe.py`

**What it is for.** Drawing a sequenced sample's whole genome, molecule by molecule, so where it holds its marks can be seen: the sample
against its own predicted held error, and against a second reading. Run by the one command with `--pat FILE --dev-genome-globe`
(`--prior-pat` for the second reading). Development: written under `development.genome_globe`.

**Label: from IAM's physics.** The held error is counted exactly as Stage Q2 counts it (Section 9): on molecules that hold their marks, how
often one unmethylated call sits between two methylated ones. The prediction is the sample's own: the self-reference line at its loose share.

**Layout.** hg19's autosomal CpGs in the .pat files' own index order (chr1 to chr22; the X and Y are left off, as the loose share leaves them,
so a difference of sex is never drawn), cut into regions of 1,000 CpGs; region i of n at NESTED pixel floor(i × NPIX / n), NSIDE 256, and every
pixel of its span carries it. Nothing fitted. A region is drawn when it holds at least 200 held-error chances.

**The change.** Each region's departure in counting noise, scaled by the sample's own spread of departures (never below 1: counting is the
floor); changed beyond the family-wise line the report globes use.

### `stage_q2_globe.py::site_owner`

The atlas sites that have a CpG index, with their atlas rows, for counting on the atlas's layout.

*Code:* `stage_q2_globe.py` line 19 (9 lines) · arguments: none · calls: not called on the traced runs · constants read: `ATLAS`, `_OWN` · files named: `import pyarrow.parquet`, `sites.parquet`, `sites_sequencing_key.parquet`

### `stage_q2_globe.py::_owner_of`

The atlas row of each CpG index, by binary search.

*Code:* `stage_q2_globe.py` line 28 (3 lines) · arguments: cpg, idx, rows · calls: not called on the traced runs

### `stage_q2_globe.py::count_pat_sites`

Per atlas site: held errors, held-error chances, methylated calls and calls, by Stage Q2's rule and line split; with the genome totals.

*Code:* `stage_q2_globe.py` line 31 (21 lines) · arguments: lines · calls: not called on the traced runs

### `stage_q2_globe.py::count_pat_sites.flush`

Adds the buffered counts to the site totals.

*Code:* `stage_q2_globe.py` line 35 (4 lines) · arguments: none · calls: not called on the traced runs

### `stage_q2_globe.py::departures`

Blocks of neighboring counted atlas sites: each block's z of observed against predicted errors, and which pass the family-wise line.

*Code:* `stage_q2_globe.py` line 52 (7 lines) · arguments: err, opp, p · calls: not called on the traced runs · constants read: `BLOCK`, `FLAG`, `ORDER`

### `stage_q2_globe.py::count_pat_regions`

The same counts per region of 1,000 consecutive CpG indexes on the autosomes.

*Code:* `stage_q2_globe.py` line 67 (22 lines) · arguments: lines, region · calls: not called on the traced runs · constants read: `N_CPG`, `REGION`

### `stage_q2_globe.py::count_pat_regions.flush`

Adds the buffered counts to the region totals.

*Code:* `stage_q2_globe.py` line 71 (3 lines) · arguments: none · calls: not called on the traced runs · constants read: `N_CPG`

### `stage_q2_globe.py::_z_self`

Each region's held errors against the number its chances predict at the sample's own predicted held error, in counting noise.

*Code:* `stage_q2_globe.py` line 90 (2 lines) · arguments: R, p · calls: not called on the traced runs · constants read: `MIN_OPP`

### `stage_q2_globe.py::_z_pair`

Each region's held error against the other reading's, in both regions' counting noise; with which regions both readings drew.

*Code:* `stage_q2_globe.py` line 92 (4 lines) · arguments: A, B · calls: not called on the traced runs · constants read: `MIN_OPP`

### `stage_q2_globe.py::_own_spread`

The sample's (or pair's) own spread of region departures: MAD × 1.4826 over the drawn regions, never below 1.

*Code:* `stage_q2_globe.py` line 96 (5 lines) · arguments: z, ok · calls: not called on the traced runs

### `stage_q2_globe.py::_page_layers`

The globe's layers: methylation level, held error, and the change (departures over the own spread, beyond the line), with their keys.

*Code:* `stage_q2_globe.py` line 101 (26 lines) · arguments: z, ok, err_rate, meth, p, change_name, about_ref, scale · calls: not called on the traced runs · constants read: `HERE`, `REGION` · files named: `Planck_Parchment_RGB.txt`

### `stage_q2_globe.py::_page_layers.rgb`

Turns a coded map into the page's colors.

*Code:* `stage_q2_globe.py` line 111 (2 lines) · arguments: m · calls: not called on the traced runs

### `stage_q2_globe.py::self_globe`

The sample against itself (its own predicted held error), with the count of changed regions.

*Code:* `stage_q2_globe.py` line 131 (7 lines) · arguments: R, loose_share, gid, title · calls: not called on the traced runs · constants read: `ABOUT`, `REGION`

### `stage_q2_globe.py::pair_globe`

The sample against another reading, with the count of changed regions.

*Code:* `stage_q2_globe.py` line 138 (7 lines) · arguments: A, B, gid, title, name_b · calls: not called on the traced runs · constants read: `ABOUT`, `REGION`

## 18. The information sphere (development): `stage_info_sphere.py`

**What it is for.** Placing sites by how they behave across cell types instead of by their address, so how far apart states are can be seen.
Built by the one command: `run_sample.py --build info-sphere` (Section 1, `_build`). Development (DEV-SPHERE-01).

**Label: established mathematics.** Each site's levels b in S cell types become u = (√b₁…√b_S, √(1−b₁)…√(1−b_S))/√S, a point on a unit sphere;
the distance between two sites is the Fisher-Rao (Bhattacharyya) angle arccos(u_i · u_j). The globe keeps the three leading principal
directions of those points, scaled to length 1 (construction v1, the one that kept distances); nothing is tuned. The sky the sites do not
cover is masked, as the microwave sky masks the galaxy.

### `stage_info_sphere.py::info_vectors`

The square-root vectors of a sites × cell types matrix.

*Code:* `stage_info_sphere.py` line 18 (3 lines) · arguments: B · calls: not called on the traced runs

### `stage_info_sphere.py::fr_angle`

The Fisher-Rao angle between pairs of sites, in the full space.

*Code:* `stage_info_sphere.py` line 22 (2 lines) · arguments: U, i, j · calls: not called on the traced runs

### `stage_info_sphere.py::globe`

Construction v1: centered, projected on the three leading directions, scaled to 1; with the share of the spread the three directions keep.

*Code:* `stage_info_sphere.py` line 25 (4 lines) · arguments: U · calls: not called on the traced runs

### `stage_info_sphere.py::_spread`

Interleaves the bits of a pixel coordinate, for the NESTED pixel number.

*Code:* `stage_info_sphere.py` line 40 (4 lines) · arguments: v · calls: not called on the traced runs

### `stage_info_sphere.py::level_free`

Construction v5 (development): each site's square-root vector minus that of a flat profile at the site's own mean level, scaled to length 1, so
the globe places sites by how they depart from their own level across the cell types, not by the level itself (DEV-SPHERE-01 test 8 found the
level dominates on real cells). Sites with no departure have no direction and are counted.

*Code:* `stage_info_sphere.py` line 30 (8 lines) · arguments: B, U · calls: not called on the traced runs

### `stage_info_sphere.py::ang2pix_nest`

The standard HEALPix NESTED angle to pixel; checked by a round trip through `stage_v4.nest_pix2ang` on all 196,608 pixels at NSIDE 128.

*Code:* `stage_info_sphere.py` line 44 (17 lines) · arguments: nside, theta, phi · calls: not called on the traced runs

### `stage_info_sphere.py::to_pixels`

The pixel of each globe point.

*Code:* `stage_info_sphere.py` line 62 (2 lines) · arguments: X, nside · calls: not called on the traced runs · constants read: `NSIDE`

### `stage_info_sphere.py::concentration`

How tightly a group of sites lands together: the length of the mean of its globe points (1 = one point, near 0 = spread over the sphere).
**Label: standard.** This is the mean resultant length of directional statistics, the field's usual measure of how concentrated points on a
sphere are; it is not ours. What the information sphere adds is where the points come from (the square-root map of each site's methylation
across cell types) and the tools that read the map.

*Code:* `stage_info_sphere.py` line 65 (3 lines) · arguments: X, idx · calls: not called on the traced runs

### `stage_info_sphere.py::edge_ratio`

The mean jump of the mean-level map across HEALPix base-face edges over the jump within a face (about 1: no grid edges).

*Code:* `stage_info_sphere.py` line 69 (14 lines) · arguments: mb, nside · calls: not called on the traced runs · constants read: `NSIDE`

### `run_sample.py::_build._R`

With --build info-globe: the concentration R of a set of sites on the information sphere (the length of their mean direction, 0 spread evenly, 1 all in one place), reported for each family and each direction in the build record; at least 20 sites, else none.

*Code:* `MethylPhys_Interface/run_sample.py` line 324 (1 lines) · arguments: m · calls: not called on the traced runs

### `stage_info_sphere.py::save_layout`

Freezes the information sphere's layout (every site's id, direction and role, and the atlas file's fingerprint) in Runtime Matrices/Info_Sphere,
written by `--build info-globe`, so every sample is placed on the same sphere.

*Code:* `stage_info_sphere.py` line 154 (6 lines) · arguments: path, ids, X, cell, sgn, size, margin, cells, cell_family, atlas_sha · calls: not called on the traced runs

### `stage_info_sphere.py::load_layout`

Reads the frozen layout back.

*Code:* `stage_info_sphere.py` line 161 (2 lines) · arguments: path · calls: not called on the traced runs · constants read: `LAYOUT`

### `stage_info_sphere.py::groups_of`

The territories a sample is read on: each family's clearly marking sites (margin at least 0.10), split by direction (methylated there, or
unmethylated there), the same groups the picture colors.

*Code:* `stage_info_sphere.py` line 164 (4 lines) · arguments: lay · calls: not called on the traced runs · constants read: `CLEAR`

### `stage_info_sphere.py::sample_reading`

One sample on the information sphere. It takes the change the address globe already read at each site (the site's difference from the
sample's own expected healthy self, in that site's healthy width and the sample's own spread), so both globes read the same change. Each
territory's mean change is put in standard errors, then in the sample's own spread of territories (median and spread over the territories
read, as the address globe reads its blocks in its own block spread). A territory has changed beyond the line chance crosses less than once
in 20 samples (5 % family-wise over the territories read). Also the Fisher-Rao angle between the sample and its expectation at each site, and
its root mean square: the sample's distance from its own expected healthy self. Nothing fitted.

*Code:* `stage_info_sphere.py` line 169 (30 lines) · arguments: x_by_id, lay, families, b_by_id, e_by_id · calls: not called on the traced runs · constants read: `MIN_GROUP`

### `stage_info_sphere.py::roles`

What groups a site with its neighbors on the information sphere. With each site's own level taken out (construction v5), a site's direction
is set by how it departs from that level across the cell types, so the cell where it departs most is the identity it carries: returned with
the sign (methylated there, or unmethylated), the size of that departure, and the margin over the next cell. The 15 families that color the
picture are a *stated* grouping of the atlas's 79 cell types by tissue and lineage (FAMILIES); nothing is computed from them. Measured, not
assumed: a cell's methylated and unmethylated sites gather on opposite sides of the sphere (T cells: two groups, each tight, R 0.88 and 0.86).

*Code:* `stage_info_sphere.py` line 137 (10 lines) · arguments: B, cells · calls: not called on the traced runs · constants read: `FAMILIES`

### `info_sphere_viewer.py::_b64`

Packs an array into text so the page carries its own data and needs no other file.

*Code:* `MethylPhys_Interface/info_sphere_viewer.py` line 49 (1 lines) · arguments: a · calls: not called on the traced runs

### `info_sphere_viewer.py::territories`

Each family's density of clearly marking sites (margin at least CLEAR = 0.10, *stated*) in every direction of a 128 x 256 grid over the sphere,
with a von Mises-Fisher kernel of 6 degrees (*stated*): the spherical form of a smoothing window.

*Code:* `MethylPhys_Interface/info_sphere_viewer.py` line 16 (9 lines) · arguments: X, fam, nfam, H, W, sigma_deg, chunk · calls: not called on the traced runs

### `info_sphere_viewer.py::territory_texture`

Colors each direction by the family most over-represented there: its share of the local density divided by its share of all clearly marking
sites (the enrichment), so a large family does not color the sphere by size alone. The glow grows with the enrichment and the density; it
dims where two families are nearly equal, which draws the dark lanes between territories; empty sky stays dark.

*Code:* `MethylPhys_Interface/info_sphere_viewer.py` line 26 (9 lines) · arguments: T, counts, colors · calls: not called on the traced runs

### `info_sphere_viewer.py::territory_peaks`

Where to write each family's name: up to two directions where it is most enriched and leads, at least 55 degrees apart (*stated*).

*Code:* `MethylPhys_Interface/info_sphere_viewer.py` line 36 (12 lines) · arguments: E, top, min_sep_deg, min_enrich, per_family · calls: not called on the traced runs

### `info_sphere_viewer.py::write`

Writes the information sphere as one page with everything inside it: every site's direction, cell, sign, size and margin; the territory
picture; the labels; and the page's own drawing code. The drawing turns the sphere, shows the territories, the sites as points, a wire
graticule for depth, and reads which family's territory is under the pointer. Its page (2026-10-23, checked control by control): redraws only while turning or after a control changes; a family shown alone or one cell type chosen turns the sphere to face it and stops the turn; the readout states how strongly a family's sites gather, how many sites a chosen cell has at the slider's setting, and, on a sample with no changed territory, why nothing remains at full fade.

*Code:* `MethylPhys_Interface/info_sphere_viewer.py` line 51 (34 lines) · arguments: path, X, cell, sgn, size, margin, cells, cell_family, families, title, about, sample · calls: not called on the traced runs · constants read: `CLEAR`, `PAGE`

### `stage_info_sphere.py::build`

The layout and its report: variance kept, sky covered and masked, distance agreement on 5,000 seeded pairs, edge ratio, and for each named
group its concentration against the 99th percentile of 500 random groups of the same size.

*Code:* `stage_info_sphere.py` line 84 (27 lines) · arguments: B, groups, n_random, seed, nside, construction · calls: not called on the traced runs · constants read: `NSIDE`

## 19. The reference globes and the series runner: `reference_globes.py`, `run_series.py`

**What they are for.** `reference_globes.py` draws the site's reference globes, so every globe on the site comes from the chain: Met-A's
purified-array reference, the atlas's noise, the 79 atlas cells' textures and their slider pictures (`run_sample.py --build explorer-purified`,
`explorer-noise`, `atlas-sphere`, `atlas-masks`). `run_series.py` runs the one command over a series of arrays on GitHub (`series.yml`):
pass 1 reads each array without references; pass 2 reads it again with run_sample.py's own tare against the other healthy arrays of the series
or slide. It reads, tares and scores nothing itself; a scorer reads only the bundles.

### `reference_globes.py::slug`

A cell name as a file name.

*Code:* `MethylPhys_Interface/reference_globes.py` line 16 (1 lines) · arguments: c · calls: not called on the traced runs

### `reference_globes.py::purified`

Met-A's purified-array group profiles at its 6,000 neutrophil identity sites, drawn as dots on the atlas's layout.

*Code:* `MethylPhys_Interface/reference_globes.py` line 18 (17 lines) · arguments: none · calls: not called on the traced runs · constants read: `CH`, `GROUP`, `HERE` · files named: `Planck_Parchment_RGB.txt`, `blood_composition_EPIC_v1.json`, `import pyarrow.parquet`, `sites.parquet`

### `reference_globes.py::noise`

The atlas's noise maps: per blood cell, the noise left in an average of k posterior draws at every site, on one log scale.

*Code:* `MethylPhys_Interface/reference_globes.py` line 36 (33 lines) · arguments: d · calls: not called on the traced runs · files named: `noise.json`, `uncertainty.json`

### `reference_globes.py::manifests`

Finds Illumina's 450K and EPIC manifests in methylprep's download folder (it differs between installs).

*Code:* `MethylPhys_Interface/reference_globes.py` line 70 (33 lines) · arguments: none · calls: not called on the traced runs

### `reference_globes.py::atlas_sphere`

The 79 cells' globe textures (best value and how sure), laid out by the frozen sky order; each cell's measured pixels must equal its cell
page's count, or the build stops.

*Code:* `MethylPhys_Interface/reference_globes.py` line 104 (32 lines) · arguments: out, atlas, man, cells_index · calls: not called on the traced runs · constants read: `GREY`, `HERE`, `HW_MAX`, `REPO` · files named: `Planck_Parchment_RGB.txt`, `import pyarrow.parquet`, `index.json`, `s measured-pixel count equal to website/art/out/cells/index.json`, `sites.parquet`, `sphere.json`

### `reference_globes.py::atlas_sphere.tex`

One texture from a sky map.

*Code:* `MethylPhys_Interface/reference_globes.py` line 117 (3 lines) · arguments: m, cmap, lo, hi, path · calls: not called on the traced runs · constants read: `GREY`

### `reference_globes.py::atlas_masks`

The value pictures for the atlas globe's sliders (best value, and half-width of the 95 % range), losslessly coded; checked against the textures.

*Code:* `MethylPhys_Interface/reference_globes.py` line 137 (18 lines) · arguments: OUT, A · calls: not called on the traced runs · constants read: `ATLAS`, `HW_MAX` · files named: `import pyarrow.parquet`, `sphere.json`

### `reference_globes.py::atlas_masks.code`

Codes a map 1 to 255 on its scale, 0 where no site.

*Code:* `MethylPhys_Interface/reference_globes.py` line 143 (2 lines) · arguments: v, hi · calls: not called on the traced runs

### `run_series.py::matrix_values`

Reads a series published only as a GEO value table (no IDATs): each array's column becomes a values file (cpg, beta) read by run_sample.py with --betas, the replays' values route; nothing in the values is changed, and the chip-level intake checks, which need IDATs, do not run. Synthetic check: a matrix holding the example healthy blood gives the same Met-A (0.9845), C-Score (0.9686) and cell shares as reading the same values directly.

*Code:* `MethylPhys_Interface/run_series.py` line 36 (26 lines) · arguments: src, gsms, d · calls: not called on the traced runs · files named: `_matrix_values.csv`

### `run_series.py::fetch`

An array's two IDATs from GEO, retried.

*Code:* `MethylPhys_Interface/run_series.py` line 24 (11 lines) · arguments: gsm, d · calls: not called on the traced runs

### `run_series.py::run`

Runs the one command with the given flags and keeps its log.

*Code:* `MethylPhys_Interface/run_series.py` line 63 (3 lines) · arguments: args, log · calls: not called on the traced runs · constants read: `PY`, `RS`

### `run_series.py::refs_for`

The pass-1 readings of the other arrays of the scope (series, slide, or slide when it holds at least 3 others, else series), as the one
command's reference inputs; never the array itself.

*Code:* `MethylPhys_Interface/run_series.py` line 67 (8 lines) · arguments: gsm, slide, P1, scope, out · calls: not called on the traced runs

### `run_series.py::bundle_row`

One bundle's numbers for the table: Met-A, its tare, the C-Score, atlas Met-A and its tare, the intake verdict, the reason when not read.

*Code:* `MethylPhys_Interface/run_series.py` line 76 (6 lines) · arguments: p · calls: not called on the traced runs

### `run_series.py::read`

One shard of a pass, as described above; with `--plant`, the constructed loss is planted in the array's own saved values and read by the one
command with the same references (Group-A's design: the planted and unplanted halves both by the values route). A per-array atlas cell (samples column atlas_cell) replaces --dev-metaa-atlas CELL for that array, so one series run reads each sorted subtype against its own IAM-Atlas v2.1 column.

*Code:* `MethylPhys_Interface/run_series.py` line 83 (67 lines) · arguments: a, extra · calls: not called on the traced runs · constants read: `HERE` · files named: `_betas.parquet`, `_planted.csv`, `_planted.json`, `_values.csv`, `_values.json`

### `run_series.py::table`

One row per bundle, for the scorer.

*Code:* `MethylPhys_Interface/run_series.py` line 151 (7 lines) · arguments: a · calls: not called on the traced runs · files named: `rows.csv`

## 20. One blood draw: `run_draw.py` (development, to-do step 40)

**What it is for.** One blood draw, split into its buffy coat (the person's own white cells) and its plasma (cell-free DNA), read end to end:
`run_draw.py` runs the one command, `run_sample.py`, on each half with `--cell-panel atlas` (and `--dev-seq-values` for sequencing), so every
instrument with enough to read reads, then writes the draw's joint record from the two bundles alone. The GitHub workflow `one_draw.yml` runs
it from public GEO samples with one click. Nothing in it is commissioned; every reading keeps its own flag.

### `run_draw.py::is_blood`

Whether an atlas cell type is a blood or bone-marrow cell, by its name.

*Code:* `MethylPhys_Interface/run_draw.py` line 21 (1 lines) · arguments: c · calls: not called on the traced runs · constants read: `BLOOD_WORDS`

### `run_draw.py::half_args`

The one command's input flags for one half: an IDAT pair, a values file, a .pat file or a GEO sample (the last two with sequencing values).

*Code:* `MethylPhys_Interface/run_draw.py` line 23 (8 lines) · arguments: a, half · calls: not called on the traced runs

### `run_draw.py::read_half`

Runs the one command on one half, on the full atlas panel, and returns its bundle, or the reason it was not read.

*Code:* `MethylPhys_Interface/run_draw.py` line 32 (10 lines) · arguments: a, half, specimen, extra · calls: not called on the traced runs · constants read: `HERE`, `RS` · files named: `bundle.json`

### `run_draw.py::sources`

A bundle's sources: each cell type's share with its 95 % interval, and its Met-A for each cell where read.

*Code:* `MethylPhys_Interface/run_draw.py` line 43 (4 lines) · arguments: B · calls: not called on the traced runs

### `run_draw.py::main`

Reads both halves and writes `draw.json` and `draw_summary.txt`: each half's sources and Met-A, IAM-A with its counting range, and the
plasma's sources the person's own buffy coat does not explain: every non-blood source, and every blood source whose plasma share is above
its buffy-coat share by more than its 95 % interval.

*Code:* `MethylPhys_Interface/run_draw.py` line 48 (41 lines) · arguments: none · calls: not called on the traced runs · files named: `draw.json`, `draw_summary.txt`

### `stage_metaa_atlas.py::_clustering`

The commissioned C-Score's statistic, unchanged: the departures in genome order, the variance of the means of blocks of 10 neighboring sites (times the square root of 10) over the variance of the sites. Departures independent from site to site give 1; departures that run along the genome give more. Only the layout counts, not the size. *Derived* (the variance of a mean of independent values).

*Code:* `stage_metaa_atlas.py` line 306 (7 lines) · arguments: z, w · calls: not called on the traced runs · constants read: `CS_BLOCK`

### `stage_metaa_atlas.py::_genome_order`

Sorts site ids into genome order, the order of IAM-Atlas v2.1's site list, which the planted runs and the globe use.

*Code:* `stage_metaa_atlas.py` line 314 (6 lines) · arguments: ids · calls: not called on the traced runs · constants read: `ATLAS` · files named: `import pyarrow.parquet`, `s sites.parquet`, `sites.parquet`

### `stage_metaa_atlas.py::cscore_in_mix`

The C-Score for each cell found in a mixed sample and for the whole genome (development, flagged). For every cell the IAM-Deconvolver finds: its departure map at its identity sites (the map its globe layer draws), read by the commissioned statistic. The whole-genome version reads the same at every atlas site the sample measured. Untared: healthy samples of the same series are the comparison until a tare for these readings is built. Below 1 % of the sample the C is flagged, not withheld (*stated*): there a cell's departure map is mostly the other cells'. Which cells are there is the IAM-Deconvolver's answer. It also reports the whole-genome map's correlation between neighboring sites at lags 1 to 9 (neighbor_corr), in genome order: near zero where departures are independent, as the derived limit assumes.

*Code:* `stage_metaa_atlas.py` line 321 (31 lines) · arguments: beta, shares, deps · calls: not called on the traced runs · constants read: `CS_BLOCK`, `CS_FLAG_SHARE`, `FLAG`, `MIX_COVER`

### `stage_provisional.py::pool_hg38`

Pools per-CpG methylated and total read files (GEO, GRCh38) at IAM-Atlas v2.1's sites, through the public EPIC hg38 annotation; a position listed twice is summed.

*Code:* `stage_provisional.py` line 22 (17 lines) · arguments: samples_csv, work · calls: not called on the traced runs · constants read: `ANN`, `ATLAS` · files named: `sites_sequencing_key.parquet`

### `stage_provisional.py::column`

A provisional reference column from pooled reads: the level through the atlas's own inverse transfer (as the atlas converted its sequencing readings), the spread from counting (q = (m + 1/2)/(n + 1)), the median donor spread of the 79 atlas cells, and the atlas reference's laboratory and floor terms. Nothing fitted.

*Code:* `stage_provisional.py` line 40 (12 lines) · arguments: P · calls: not called on the traced runs · constants read: `ATLAS`, `SMOOTH` · files named: `sites.parquet`

### `stage_provisional.py::build`

Builds and saves a provisional column with its provenance (samples, annotation and column checksums); run by run_sample.py --build provisional.

*Code:* `stage_provisional.py` line 79 (6 lines) · arguments: cell, samples_csv, work, out · calls: not called on the traced runs · constants read: `ANN`, `DIR`, `FLAG`, `SMOOTH`

### `stage_provisional.py::load`

Loads a saved provisional column, if one was built.

*Code:* `stage_provisional.py` line 86 (2 lines) · arguments: cell · calls: not called on the traced runs · constants read: `DIR`

### `stage_provisional.py::with_provisional`

The atlas reference with provisional columns added, as a copy; IAM-Atlas v2.1 itself is never changed.

*Code:* `stage_provisional.py` line 89 (9 lines) · arguments: R, cells · calls: not called on the traced runs

### `stage_v4.py::stage_d21_provisional`

A second deconvolution with provisional columns beside IAM-Atlas v2.1, recorded as cell_shares_provisional and flagged; the atlas's own shares are unchanged (development, plan 40.18).

*Code:* `stage_v4.py` line 61 (13 lines) · arguments: beta, panel, cells, sequencing · calls: not called on the traced runs · constants read: `_REF`

### `stage_provisional.py::column_v2`

Version 2 of a provisional column (after test 1): the level from the shrunk proportion (m + 1/2)/(n + 1), the counting spread carried through the atlas's inverse transfer by the two ends of its interval (no local slope, so thin sites at the ends are not made narrow), and only sites with at least 5 reads.

*Code:* `stage_provisional.py` line 55 (19 lines) · arguments: P · calls: not called on the traced runs · constants read: `ATLAS`, `MIN_READS_V2`, `SMOOTH` · files named: `sites.parquet`

### `stage_provisional.py::_name_version`

Reads a column name with its version ('megakaryocytes@2'); a bare name is version 1. The version is recorded in the bundle, the cell keeps its plain name.

*Code:* `stage_provisional.py` line 75 (3 lines) · arguments: cell · calls: not called on the traced runs

### `stage_coupling.py::read`

Nearest-neighbor coupling on the A-score gauge (in development, plan 49.5). coupling-A is the sample's coupling (Stage Q2) over the healthy
coupling of its cell type (`Runtime Matrices/Coupling/coupling_reference_v1.json`, built from IAM-Atlas v2.1's healthy sequencing inputs).
The gauge runs from 0 (errors never neighbors) to the ceiling `coupling_max` over the same healthy coupling (every molecule's errors in one
run), both counted. Output `coupling_a`: A, healthy, floor, ceiling, band, state, flag; without a reference, the reason.

*Code:* `stage_coupling.py` line 17 (13 lines) · arguments: q, cell, ref · calls: not called on the traced runs · constants read: `FLAG`

### `report_v4.py::_coupling`

The report's nearest-neighbor coupling section: coupling-A's state and Normal band beside the healthy coupling of the sample's cell,
flagged in development, with its gauge; without a reference, the coupling and the reason it has no gauge position.

*Code:* `MethylPhys_Interface/report_v4.py` line 177 (11 lines) · arguments: o · calls: not called on the traced runs · constants read: `E`

### `stage_coupling.py::reference`

Loads the healthy coupling reference (`Runtime Matrices/Coupling/coupling_reference_v1.json`): each cell type's healthy coupling and the
Normal band of coupling-A. Returns nothing until the reference is built, and the reading then says so.

*Code:* `stage_coupling.py` line 14 (2 lines) · arguments: path · calls: not called on the traced runs · constants read: `REF`

### `stage_metaa_atlas.py::seq_units`

An atlas level, on the array scale, in sequencing units: a + b x level from the conversion the atlas's own build measured for the published
sequencing atlas (`Runtime Matrices/Atlas_V21/sequencing_term_v1.json`), clipped to 0 to 1. Every comparison of a sequencing sample against
the atlas goes through it: Met-A for a sorted cell, Met-A for each cell in a mix, and the territories.

*Code:* `stage_metaa_atlas.py` line 80 (5 lines) · arguments: p · calls: not called on the traced runs · constants read: `SEQ_TERM`

## 21. The IAM Sequencing Atlas (development): `stage_seq_atlas.py`

The reference for sequenced samples, kept apart from IAM-Atlas v2.1 (arrays), plan 35.30. Measured, not fitted: every value is counted from
healthy donors' reads at every one of the 28,217,448 CpGs of the hg19 numbering the .pat and .beta files share. The one command builds it
one block at a time (`run_sample.py --build seq-atlas --build-samples CSV --build-block START:END --build-out PARQUET`); the blocks together
are the atlas. Its inputs (`doors/data/DEV_ATLAS_V21_01/seq_atlas/seq_atlas_inputs.csv`) are the 153 healthy sequencing inputs behind IAM-Atlas
v2.1, each with its GEO .beta file, cell and sealed donor half. Each block is a parquet with the CpG index and, per cell, these columns
(uint16; levels times 10,000; 65535 is blank, where no donor of the cell has a read, and nothing is filled):
`donors`, `reads`, `meth` (methylated reads; with `reads` it gives the pooled level and its exact 95 % range), `level` (median of the donors'
levels), `spread` (the donors' standard deviation), `disorder` (each donor's binary entropy, averaged), `halfA`, `halfB` (the pooled level in
each sealed donor half; their difference is noise map 1), `readA`, `readB` (the pooled level in two random halves of every donor's reads,
seed 20261029; their difference is noise map 2, defined for a cell with one donor).

### `stage_seq_atlas.py::read_block`

One input's methylated and all reads over a block of CpGs: a .beta file (uint8 pairs) read locally or straight from GEO by byte range, with
retries, or an .lbeta (uint16 pairs, this stage's converted files). A file that is not the full hg19 length is refused.

*Code:* `stage_seq_atlas.py` line 24 (20 lines) · arguments: src, start, end · calls: not called on the traced runs · constants read: `HG19_CPGS`

### `stage_seq_atlas.py::build_block`

The atlas over one block: for each cell, every column above, from its donors' counts. Returns the table and a summary (each cell's donors,
halves, CpGs with reads and median reads; sites at the source files' 255-read cap).

*Code:* `stage_seq_atlas.py` line 89 (27 lines) · arguments: samples, start, end, source · calls: not called on the traced runs · constants read: `BLANK`, `SEED`

### `stage_seq_atlas.py::build_block.pooled`

The pooled level of a cell's donors in one sealed half (all their methylated reads over all their reads), for the donor-half noise map.

*Code:* `stage_seq_atlas.py` line 103 (3 lines) · arguments: sel · calls: not called on the traced runs

### `stage_seq_atlas.py::hg19_dictionary`

The pipeline's hg19 CpG dictionary (wgbstools init_genome's CpG.bed.gz: chromosome, 1-based position, index), checked for 28,217,448 CpGs.

*Code:* `stage_seq_atlas.py` line 45 (5 lines) · arguments: cpg_bed · calls: not called on the traced runs · constants read: `HG19_CPGS`

### `stage_seq_atlas.py::lift_to_hg19`

One donor's GRCh38 per-CpG read tables, pooled on GRCh38 and moved to the hg19 numbering (UCSC hg38ToHg19, pyliftover; a position that does
not move to one place is dropped and counted), each landing on the hg19 CpG whose C or G it is, so both strands add to one CpG. Written as an
.lbeta; the report counts positions read, moved and landed. Run by `run_sample.py --build seq-lift`.

*Code:* `stage_seq_atlas.py` line 51 (29 lines) · arguments: files, out, cpg_bed, chain · calls: not called on the traced runs · constants read: `HG19_CPGS`

### `stage_seq_atlas.py::metaa_reference`

Each cell's IAM Sequencing Atlas values at its Met-A identity sites (`run_sample.py --build seq-metaa-ref`): the same sites Met-A reads on
arrays, chosen by its site rule on IAM-Atlas v2.1, so Met-A stays one instrument, with every stored column from the built pieces, as one
long table (cell, probe, cpg, columns). Small enough to sit in the chain (`Runtime Matrices/Seq_Atlas/metaa_seq_reference_v1.parquet`).

*Code:* `stage_seq_atlas.py` line 120 (27 lines) · arguments: piece_dirs, out · calls: not called on the traced runs · constants read: `METAA_COLS` · files named: `pyarrow.parquet`, `sites_sequencing_key.parquet`

### `stage_seq_atlas.py::floor_sites`

The library floor's sites (`run_sample.py --build seq-floor`): among IAM-Atlas v2.1's sites, the CpGs where every cell of the IAM Sequencing
Atlas has at least 20 reads and reads at or below 0.05 or at or above 0.95. There a healthy cell carries no information, so an opposite call
is the library's own (unconverted DNA, sequencing error, damage).

*Code:* `stage_seq_atlas.py` line 151 (24 lines) · arguments: piece_dirs, out, rule · calls: not called on the traced runs · constants read: `BLANK`, `FLOOR_RULE` · files named: `pyarrow.parquet`, `sites_sequencing_key.parquet`

### `stage_seq_atlas.py::library_floor`

A sequenced sample's library floor: at those sites, the share of opposite calls on each side and pooled, from the sample's own calls;
recorded on every sequenced sample as `development.library_floor`, flagged.

*Code:* `stage_seq_atlas.py` line 176 (12 lines) · arguments: beta, calls, path · calls: not called on the traced runs · constants read: `FLOOR_SITES`

### `stage_seq_atlas.py::metaa_reference_of`

One cell's rows of that reference, indexed by probe, levels 0 to 1 (blank as missing); nothing when the file or the cell is absent, and the
reading then says it was read against IAM-Atlas v2.1.

*Code:* `stage_seq_atlas.py` line 189 (10 lines) · arguments: cell, path · calls: not called on the traced runs · constants read: `BLANK`, `METAA_REF`

### `stage_seq_atlas.py::held_reference`

Each healthy cell's held error region by region, stored beside the sequencing atlas (`run_sample.py --build seq-held`): from each healthy
sample's held-error counts by tile (`--dev-count-only --dev-count-tiles`; a tile is 2,000 consecutive CpGs), summed by cell, as counts, never
ratios: donors, lone errors and their chances in all, in each sealed donor half, and in each half of every donor's molecules. A tile's held
error is its lone errors over its chances. It is the healthy held error a plasma's own mix is built from: each cell's, weighted by its share.

*Code:* `stage_seq_atlas.py` line 200 (25 lines) · arguments: samples, out · calls: not called on the traced runs

### `stage_seq_atlas.py::compare`

How far one sample sits from every cell of the atlas (`run_sample.py --build seq-compare`): at the CpGs where the sample has at least 5 reads
and every cell has a level, the mean distance between the sample's level and each cell's, cells ranked nearest first; and for each cell, the
mean distance between its two donor halves at the same CpGs. Reads the built pieces in two passes (levels first, halves only at the compared
CpGs) so a 16 GB machine holds it. Also a depth-weighted score per cell: each CpG's squared difference over the scatter counting gives it
(the sample's and the column's binomial scatter and the column's donors' spread squared), averaged, about 1 for a healthy cell of that column;
ranked as `weighted_ranked`, so a column pooled from sparse reads is not pushed away by its own counting scatter.

*Code:* `stage_seq_atlas.py` line 226 (51 lines) · arguments: lbeta, piece_dirs, min_reads · calls: not called on the traced runs · constants read: `BLANK`, `HG19_CPGS` · files named: `pyarrow.parquet`

### `stage_seq_atlas.py::geo_url`

The hg19 .beta file of a published sequencing sample on GEO (used when the inputs list gives no file).

*Code:* `stage_seq_atlas.py` line 278 (3 lines) · arguments: gsm, title · calls: not called on the traced runs

### `stage_seq_atlas.py::_u16`

Values to the stored uint16 form (times 10,000, clipped; blank 65535).

*Code:* `stage_seq_atlas.py` line 81 (2 lines) · arguments: x, scale · calls: not called on the traced runs · constants read: `BLANK`

### `stage_seq_atlas.py::_H`

Binary entropy in bits, 0 at a level of 0 or 1.

*Code:* `stage_seq_atlas.py` line 84 (4 lines) · arguments: p · calls: not called on the traced runs
