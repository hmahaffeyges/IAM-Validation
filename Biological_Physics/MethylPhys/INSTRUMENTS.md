# The IAM instrument catalogue

*Generated from CANON/catalogue.json; every status line comes from the Master Testing Manifest.*

## Why a self-reference matters

Until now every instrument needed a reference: healthy cells of the same kind, often from the same laboratory, to say what normal looks like. That reference is the hardest thing in this field to get, and it is why most tests work in one lab and stumble in the next. On 2026-10-15 we found that across 56 kinds of healthy human cells, a cell's tagging mistakes follow one straight line set by how loosely it holds its own DNA. So a healthy cell's expected reading can come from its own sample. An instrument that carries its own reference does not need a library of other people's cells, does not inherit another lab's quirks, and can be pointed at any tissue, any lab, even any species that tags its DNA. That step is now commissioned: IAM-A reads each sample against its own reference (2026-10-17).

## How they work together

Each instrument answers a different question about the same sample, so together they leave no blind side. The atlas says whose DNA it is. Met-A says how far a tissue's tagging level has drifted toward the ceiling. The C-Score says whether the pattern has started to clump. IAM-A says whether each mark is still held at its own healthy rate. Group-A says whether a whole group has moved, for studies rather than single patients. A tumor that hides from one reading still has to answer to the others.

## Lab methods: what the chemistry and the machines give, and which instrument reads each

DNA's sequence is the same in every cell; the methyl tags are not, and a sequencer cannot see a tag directly. So the DNA is first treated so that tagged and untagged spots read as different letters. **Bisulfite** (a salt solution) turns every untagged C into a U, which reads as T; a tagged C stays C. **EM-seq** does the same with enzymes instead, and damages the DNA far less. After either, a tag shows up as a C and a missing tag as a T. **Conversion failure** is an untagged C the chemistry missed: it stays C and looks tagged. The chain measures it on every sample, from the C's outside CpG pairs, which are almost never tagged in blood and most body cells, and prints it in every report. Then the treated DNA is read, either by an **array** (millions of DNA probes that light up red or green, one pair of colors per site, so each site gives one average over all the cells in the sample) or by a **sequencer** (hundreds of millions of short pieces, each one molecule from one cell).

| Method | What it gives | Single molecules? | Who uses it | Which IAM instruments read it |
|---|---|---|---|---|
| **Methylation arrays (Illumina 450K, EPIC v1, EPIC v2)** | One average per site (beta value) over every cell in the sample; 450,000 to 935,000 chosen sites; two-color readout | no | Most aging clocks (Horvath, GrimAge) and most cohort studies | Met-A, C-Score, Group-A, Atlas |
| **Whole-genome bisulfite sequencing (WGBS)** | Every one of about 28 million CpG sites, on single molecules about 100-150 letters long | yes | Reference atlases of human cell types; research | IAM-A, Atlas, C-Score (sequencing reading) |
| **EM-seq (enzymatic conversion)** | The same readout as WGBS with gentler chemistry: longer, less damaged molecules from less DNA | yes | Newer research and clinical labs | IAM-A, Atlas |
| **RRBS (reduced representation)** | A cut-and-select step keeps only CpG-dense regions (islands and promoters), about 1 to 3 million sites | yes, islands only | Lower-cost research studies | none of IAM's readings: IAM-A's loose share comes from the rest of the genome, which RRBS skips; the C-Score's sequencing reading failed on it |
| **Targeted methylation sequencing (capture panels)** | Molecules from selected regions only | yes, selected regions only | Blood tests for cancer such as GRAIL's Galleri, which match a sample against patterns learned from people with and without cancer | not yet assessed |
| **MeDIP and MRE (enrichment)** | Pull out methylated (or cut unmethylated) DNA in bulk; no reading per site or per molecule | no | Older and lower-cost studies | none: there is nothing per molecule to count (a thermometer handed a barometer's printout) |
| **Nanopore and PacBio long reads** | Read the tag directly on long native molecules, no conversion chemistry | yes, long | Growing research use | IAM-A: a reader is not built yet (planned) |

**How IAM differs from cohort methods.** Aging clocks and cancer blood tests learn patterns from large groups of people and score a new sample by how closely it matches them. IAM's instruments measure a physical quantity against a floor and a ceiling set by physics. IAM-A goes further: it reads each sample against its own loose share, like checking for a fever with the back of your own hand on your own forehead instead of against a crowd's average.

## IAM-Atlas v2.1

**The question:** Whose DNA is this? Which cells are in a sample, and in what shares?

A map of 79 kinds of healthy human cells, built from 21 public sources on arrays and sequencing, fitted as one model so every spot on the genome carries its value and its uncertainty. Like a state estimator across the whole interconnection: many utilities' meters, each weighted by how far it can be trusted.

**Reads:** arrays or sequencing. **Where it stands:** commissioned (79 cell types; met all three sealed bars against both published reference atlases, 2026-10-16; commissioned for three cell references and the colon and lung tissue tests).

- ATL-005 Atlas v2.1 joint fit: run; last run 2026-10-16: not met
- ATL-060 Head to head: atlas v2.1 against the source atlases' own references, on their own job: run; last run 2026-10-16: met
- ATL-001 Atlas commissioning as a source of healthy references: run; last run 2026-10-16: met

**Record:** [COMMISSIONING_NOTE_ATLAS_V21.md](doors/COMMISSIONING_NOTE_ATLAS_V21.md)

**Next:** Deposited with its DOI (10.5281/zenodo.23441999, 2026-10-19). Its complementary additions: the posterior noise maps in the chain and the report, the 3D information sphere, and the guidebook (to-do step 13 of the commissioning plan).

## IAM-Deconvolver

**The question:** How much of each cell is in a mixed sample?

Reads a mixed sample against IAM-Atlas v2.1 and reports each cell's share with its range, after a sample check that refuses arrays it cannot read. On 200 fresh mixtures from a laboratory the atlas never saw, the shares came back within 0.016 of the truth (mean absolute error).

**Reads:** arrays. **Where it stands:** commissioned (blood-cell shares on arrays, tested on neutrophils, monocytes, CD4 T and CD8 T cells, 2026-10-17).

- ATL-065 Deconvolution tool on atlas v2.1: run; last run 2026-10-17: not met
- ATL-066 Deconvolution tool version 2 on fresh mixtures: run; last run 2026-10-16: not met

**Record:** [COMMISSIONING_NOTE_DECONV_V21.md](doors/COMMISSIONING_NOTE_DECONV_V21.md)

**Next:** B and NK cell shares, tissue cells, then cell-free DNA.

## Met-A

**The question:** How far has a tissue's tagging level drifted toward the ceiling?

Reads the methylation level at sites a healthy cell holds firmly, against a healthy reference, on cheap and plentiful arrays. Commissioned on isolated neutrophils.

**Reads:** arrays. **Where it stands:** commissioned (isolated neutrophils).

- COM-METAA-003 Met-A commissioning on isolated neutrophils, EPIC v1: run; last run 2026-10-11: progressed
- COM-METAA-039 Met-A read against the sample's own line: run; last run 2026-10-15: not met
- COM-METAA-032 Whole blood back into Met-A scope: designed

**Record:** [COMMISSIONING_NOTE_METAA_NEUTROPHILS.md](doors/COMMISSIONING_NOTE_METAA_NEUTROPHILS.md)

**Next:** More cell types, each against its own healthy profile.

## C-Score (the clustering meter)

**The question:** Has the pattern started to clump?

Cancer does not only shift tagging levels; it clumps them into runs. The C-Score measures that clumping against a healthy band. The band held in three laboratories that never helped set it, and one person measured twice reads the same.

**Reads:** arrays. **Where it stands:** commissioned (tested and commissioned on whole blood, EPIC v1, 2026-10-15).

- COM-CSCORE-011 C-score healthy band test: run; last run 2026-10-13: met
- COM-CSCORE-012 C-score last check: same-person repeatability on unspent replicate arrays: run; last run 2026-10-15: met

**Record:** [COMMISSIONING_NOTE_CSCORE.md](doors/COMMISSIONING_NOTE_CSCORE.md)

**Next:** Real cancers read on the gauge; then 450K, EPIC v2 and plasma.

## IAM-A

**The question:** Is each mark still held at its own healthy rate?

Counts tagging mistakes one DNA molecule at a time, the most direct reading of informational fidelity we have, and reads them against the sample's own line. The physics end of the catalogue: the floor and ceiling are derived, and since 2026-10-15 the healthy middle comes from the sample itself.

**Reads:** single-molecule sequencing (bisulfite, EM-seq; nanopore next). **Where it stands:** commissioned (each sample against its own reference; sealed test on healthy blood CD8 T cells, whole-genome bisulfite, 2026-10-17).

- PHY-045 The loose rate read off the line: every healthy cell's held error from its own loose share: run; last run 2026-10-15: met
- PHY-048 The loose-end line at the one-Landauer-bit cut: run; last run 2026-10-15: met
- PHY-049 The loose share from each sample's own molecules: run; last run 2026-10-15: not met
- PHY-043 Tumor backbone against the writer's nameplate: run; last run 2026-10-15: progressed
- COM-IAMA-052 IAM-A atlas-free commissioning: one grand sealed test on fresh data: run; last run 2026-10-17: met

**Record:** [COMMISSIONING_NOTE_IAMA.md](doors/COMMISSIONING_NOTE_IAMA.md)

**Next:** Other healthy blood cells, then tissue and cell-free DNA; the tumor pairs read on the gauge.

## Group-A

**The question:** Has a whole group moved (a trial arm, an exposure, a cohort)?

Single blood readings are too noisy for one person on arrays, but a group average is not. Group-A reads each person with the one command, tares each reading against healthy arrays run the same way, and compares the groups. On 278 healthy people from a laboratory it had never read, women and men came out 0.36 % apart, and a 1 % loss planted into half of them was found at 7.253 standard errors.

**Reads:** arrays. **Where it stands:** commissioned (groups of whole bloods run in one series, EPIC v1, 2026-10-20).

- GRP-004 DEV-GROUP-01 step 2: group comparison, one cancer at a time, patients and healthy controls in the same series: run

**Record:** [COMMISSIONING_NOTE_GROUPA.md](doors/COMMISSIONING_NOTE_GROUPA.md)

**Next:** One cancer at a time, patients and healthy controls from the same series.

## Candidate instruments



Candidate instruments: every reading we could build on the methylation channel, each with the term of the law it reads, its substrate, its blind spot, its advantage, what would break it, and the next concrete step. The point is NOT to pick one - several instruments with different blind spots, agreeing on the same sample, is worth more than one instrument we trust. Priority is by (decides something) x (free or cheap) / (blocked).

### 1. Clustering meter (C-score): finish its commissioning

- **Reads:** whether losses are CLUSTERED (whole stretches) rather than scattered; the shape of the loss, not its size
- **Substrate:** arrays (EPIC v1 now; 450K and EPIC v2 later) · **Cost:** free (a metadata search, then one laptop read)
- **Where it stands:** DONE 2026-10-15: last check met (same person twice, SD 0.038 against 0.125; third laboratory 172/173); commissioned on whole blood. test (a) MET in two laboratories that had no part in setting the band (34/35 and 22/23 healthy, 95 % bar)
- **Still open:** new SAME-PERSON replicate arrays, from a set that did not help build the band, to confirm the repeatability bar
- **Advantage:** clustering is the cancer fingerprint; ordinary aging and inflammation scatter errors instead. Most specific thing we have
- **Blind spot:** whole blood so far, while the meter it rides on is commissioned on sorted cells; that scope gap must be reconciled before any detection claim; and one array decided the second test (22/23 = 95.65 % against 95 %)
- **What would break it:** healthy replicate pairs from one person differing by more than the band's own tolerance
- **Next step:** search GEO for healthy EPIC v1 series with same-person repeats, excluding every set in the spent register
- **Record:** [CSCORE_COMMISSIONING_PLAN.md](doors/CSCORE_COMMISSIONING_PLAN.md)

### 2. Territory contrast: fidelity in fragile territory over fidelity in stable territory

- **Reads:** WHERE the loss happens first; a sustained energy shortfall should take the late-replicating, solo-WCGW territory before it touches housekeeping promoters
- **Substrate:** arrays (works today, no new data) · **Cost:** free (design, synthetic test, then existing arrays)
- **Where it stands:** RUN ONCE 2026-10-13 (DEV-TERRITORY-01): the balance equation's parameter-free prediction of the response ratio, 4.158 from healthy baselines alone, was met to 1 % (measured 4.12-4.20 at shortfalls of 2-20 %), and the declared blind spot was confirmed; but the NUISANCE bar FAILED
- **Still open:** v2: report both the difference and the normalized difference (one cancels each nuisance type), measure the beta-ceiling effect from fully held sites and subtract it, and leave composition to the mixture-predicted denominator the chain already uses
- **Advantage:** SELF-TARING; a ratio inside one array, so plate, batch and much of the composition effect cancel because numerator and denominator see the same cell mixture. It addresses the laboratory-to-laboratory spread that limits single whole-blood readings
- **Blind spot:** says nothing about absolute fidelity; a loss uniform across territories is invisible (confirmed, 0.1 % of the signal). And the practical failure: a difference cancels an ADDITIVE plate offset exactly but cannot cancel a MULTIPLICATIVE one; a 10 % composition change moves it about 5x what a 2 % shortfall does; while the beta = 1 ceiling makes offsets and noise one-sided
- **What would break it:** the ordering reversing; stable territory losing before fragile territory in a condition known to cost the cell energy
- **Next step:** write DEV-TERRITORY-02 with the two-number signature and the measured ceiling floor, bars first
- **Record:** [DEV_TERRITORY_01.md](doors/DEV_TERRITORY_01.md)

### 3. Met-A on whole blood: measure the real healthy spread

- **Reads:** total disorder above what the estimated cell mixture explains; the existing composition-aware reading, in a tube of blood
- **Substrate:** arrays · **Cost:** free (one laptop read of arrays already on disk)
- **Where it stands:** MEASURED on real healthy arrays, corrected 2026-10-20: read by the chain itself (run_neutrophil, in the C-score band test), healthy people differ by 2.01 % (relative SD; 95th percentile +3.33 %), and the chain's leverage is 2.12 % of the reading per 1 % pattern loss, so about a 1.57 % loss clears the healthy 95th percentile. The first measurement (2026-10-13, 6.36 %) skipped the chain's self-tare step (Stage T step 1) and is withdrawn. GROUP ARM 2026-10-13 (DEV-GROUP-01): the sham passes; two healthy groups from two laboratories agree to 0.16 % after taring, but differ by 3.69 % UNTARED, at 6.5 standard errors. A group study is admissible only when patients and healthy comparisons are run in the SAME series
- **Still open:** per-person detection in whole blood, now open: the corrected spread is one laboratory's 23 arrays and one leverage array; a sealed per-person test across laboratories decides it
- **Advantage:** group comparisons are within reach NOW: with 50 people a mean loss of about 0.27 % separates the group from healthy, and the function already exists in the chain
- **Blind spot:** an array averages molecules before anyone reads them, so one cell type can never be isolated. Across laboratories without a shared series the raw reading carries a 3.7 % offset, which is inside the tumor range, so every comparison needs the same-series tare
- **What would break it:** healthy people differing by more than the leverage of the signal we are looking for (then arrays cannot carry blood and sequencing is required)
- **Next step:** commission Group-A (to-do step 20); then a sealed per-person test in whole blood, across laboratories
- **Record:** [DEV_BLOOD_SENSITIVITY_01.md](doors/DEV_BLOOD_SENSITIVITY_01.md)

### 4. Distance to the ceiling

- **Reads:** how close a sample sits to the derived entropy ceiling; the right-hand mark of the gauge, past which the reading stops rising with further loss
- **Substrate:** arrays and sequencing · **Cost:** free (arithmetic on readings we already have)
- **Where it stands:** the ceiling is in the foundation as derived. AUDIT DONE 2026-10-13 (DEV-CEILING-AUDIT-01): of 7,681 readings that record the flag, none is past the ceiling; the composition-aware path never recorded it, so the three sealed tissue results cannot be audited from saved files, and that gap could only have cost detections
- **Still open:** re-read the three sealed tissue sets with the flag recorded (free, arrays on disk, reported as an audit that cannot change their sealed verdicts); and the reading itself; distance to the ceiling has still never been reported as a number
- **Advantage:** derived with no free parameter, and it answers how far from the ceiling a tumor arrives
- **Blind spot:** only informative for samples near the ceiling; says nothing about a healthy-range sample
- **What would break it:** a sample reading past the ceiling, which the derivation forbids
- **Next step:** compute it for every sealed cancer reading already in the records and check none is past the turnover
- **Record:** [DEV_CEILING_AUDIT_01.md](doors/DEV_CEILING_AUDIT_01.md)

### 5. Run-length spectrum of losses

- **Reads:** the SHAPE of clustering; exponential against heavy-tailed, which separates a processive mechanism from independent point losses
- **Substrate:** sequencing only (needs several sites on one molecule) · **Cost:** laptop work on reads already in storage; no new sequencing
- **Where it stands:** not written
- **Still open:** the derivation of the expected shape under each mechanism, then a synthetic test. Gated: the sequencing instrument is held, though its trim rule now has six planted bars met (DEV-IAMA-TRIM-01) and needs one cheap box read to calibrate
- **Advantage:** same data as the clustering meter, different question; distinguishes mechanism rather than only size
- **Blind spot:** impossible on arrays; needs depth, so shallow libraries are out
- **What would break it:** the measured shape matching neither mechanism's prediction in a case where the mechanism is known
- **Next step:** after the sequencing instrument is released from hold
- **Record:** [DEV_IAMA_TRIM_01.md](doors/DEV_IAMA_TRIM_01.md)

### 6. Tail of fragments in plasma (the early-detection instrument)

- **Reads:** the few badly disordered MOLECULES in a plasma sample, not the average; because early cancer is a tiny minority of fragments and an average hides it by construction
- **Substrate:** sequencing of cell-free DNA · **Cost:** box time per sample; the expensive one
- **Where it stands:** not written; gated
- **Still open:** the sequencing instrument is held, the atlas is not commissioned, and a healthy plasma reference does not exist yet
- **Advantage:** the highest reward on the whole board; it is the blood test, and a tail statistic is the right shape for early detection
- **Blind spot:** needs depth and fragment length; a shallow or heavily fragmented sample has no tail to read. Composition of plasma is itself a question
- **What would break it:** the tail in healthy plasma being as heavy as in known cancer plasma
- **Next step:** blocked behind the sequencing instrument's trim rule and the atlas; do not spend box money on it yet
- **Record:** [DEV_IAMA_TRIM_01.md](doors/DEV_IAMA_TRIM_01.md)

### Not an instrument, but it releases two of them: the trim rule

The sequencing instrument is held because the same kit read differently in two laboratories, and the cause is where in each file we read: at the head the failing libraries read 11-14 % below the controls read at the SAME position, and Stage B had read the head (part 0 reproduces each sealed value to 0.0 %). Reading more places failed its planted test - pooling divides a bias rather than removing it. The remedy, DEV-IAMA-TRIM-01, excludes the unsettled head and decides how much from the raw reads' own chemistry (the non-CpG C-to-T fraction in windows, no alignment and no IAM-A): six planted bars, all met, including blindness to the direction of the error and the declared limit that it must not act without independent evidence. Nothing real has been trimmed yet; calibration is one cheap box read, and the laboratory check must then be rerun on the WHOLE stage because the five diagnostic libraries are spent. Instruments 5 and 6 are both behind it.

_Several instruments, each with a different blind spot, reading the same sample and agreeing - that is worth far more than one instrument we trust. Disagreement is also informative: a sample that reads normal on total disorder and abnormal on clustering is telling us which channel failed._


