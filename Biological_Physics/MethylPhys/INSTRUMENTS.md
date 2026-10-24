# The IAM instrument catalogue

*Generated from CANON/catalogue.json; every status line comes from the Master Testing Manifest.*

## Why a self-reference matters

Until now every instrument needed a reference: healthy cells of the same kind, often from the same laboratory, to say what normal looks like. That reference is the hardest thing in this field to get, and it is why most tests work in one lab and stumble in the next. Across 56 kinds of healthy human cells, we found that a cell's tagging mistakes follow one straight line set by how loosely it holds its own DNA. So a healthy cell's expected reading can come from its own sample. An instrument that carries its own reference does not need a library of other people's cells, does not inherit another lab's quirks, and can be pointed at any tissue, any lab, even any species that tags its DNA. That step is now commissioned: IAM-A reads each sample against its own reference.

## How they work together

Each instrument answers a different question about the same sample, so together they leave no blind side. IAM-Atlas v2.1 is the healthy reference every reading uses. The IAM-Deconvolver says whose DNA it is: which cells are in the sample, and in what shares. Met-A says how far each of those cells has drifted toward the ceiling. The C-Score says whether the pattern has started to clump. IAM-A says whether each mark is still held at its own healthy rate. Group-A says whether a whole group has moved, for studies rather than single patients. Every reading is also drawn on the globe, where it sits on the genome, against its healthy reference. A tumor that hides from one reading still has to answer to the others.

## What each instrument is referenced against, and who covers its blind spot

IAM-Atlas v2.1 comes first because nearly every other reading is made against it: it is the shared ruler, built once from healthy cells with a stated uncertainty at every site. No instrument here needs a healthy cohort from the same laboratory. Each is referenced against something it carries with it: the sample's own molecules, the atlas mixed in the person's own cells, the sample's own departures, or physics. Each also has a blind spot, and the last column names the instrument that sees what it cannot. No instrument stands alone.

| Instrument | Referenced against | Derived | Measured once | Blind to | Covered by | Status |
|---|---|---|---|---|---|---|
| **IAM-Atlas v2.1** | healthy cells from 21 published sources, arrays and sequencing, fit jointly with each source's own offset fitted alongside, as Planck's maps were made from every detector's scans | nothing: it is the measured ruler, with a stated uncertainty at every site | every site in each of its 79 cell types, built once and shared by every laboratory | cell types it does not hold, and sites no source measured (left blank, never filled in) | the IAM-Deconvolver's leftover and the unexpected-cell flag; a new cell type enters only through its own sealed test | commissioned: 79 cell types; met all three sealed bars against both published reference atlases; commissioned for three cell references and the colon and lung tissue tests |
| **IAM-A** | the sample's own molecules: its loose share predicts the held error a healthy cell of its kind makes | the floor, the ceiling and the self-reference line | the line's slope (being derived) | arrays: it needs single molecules, so a sequenced sample | Met-A, which reads arrays | commissioned: each sample against its own reference; sealed test on healthy blood CD8 T cells, whole-genome bisulfite |
| **Met-A, whole sample** | IAM-Atlas v2.1 mixed in the person's own cell shares, with the array's own blur measured on the atlas's anchor sites | the ceiling, the read-count correction, the array's blur | the atlas | a loss spread evenly over every site, the anchors included; which cell changed | Met-A for each cell, the territories, IAM-A | in development |
| **Met-A, each cell** | each cell's identity sites in IAM-Atlas v2.1 (commissioned: healthy references of the same cell on the same platform) | the ceiling, the read-count correction | the atlas | changes outside a cell's identity sites, such as the islands and loosely held domains | Met-A for the whole sample, the territories | commissioned: tested and commissioned on isolated neutrophils, EPIC v1; in a mix, each cell's reading is in development |
| **The territories (islands and loosely held domains)** | IAM-Atlas v2.1 at sites no healthy cell type marks, and at marked sites some cell types hold loosely | the expectation at each site's read count | the two site sets, by fixed rules on the atlas | changes at a cell's identity sites | Met-A for each cell | in development |
| **C-Score, neutrophils** | the sample's own departures: how they lie along the genome, not how large they are | its value for scattered departures, 1 | its healthy upper limit | an even shift with no layout; cells other than neutrophils | the other C-Scores, Met-A | commissioned: tested and commissioned on whole blood, EPIC v1; the other cells' C-Scores are in development |
| **C-Score, whole genome** | the same statistic at every atlas site, against the atlas in the person's own mix | its value for scattered departures, 1 | its healthy limit, not set yet | which cell; an even shift | the C-Score for each cell, Met-A | in development |
| **C-Score, each cell** | each cell's departure map at its identity sites | its value for scattered departures, 1 | its healthy limit, not set yet | a loss spread evenly through one cell | Met-A for each cell | in development |
| **IAM-A's C-Score (version 2)** | each region's held error predicted from that region's own loose share | being derived | nothing yet | arrays | the array C-Scores | in development |
| **Group-A** | a group against the healthy in the same series, each reading tared on its own chip | the comparison, a difference in standard errors | nothing | one person: it reads groups | every per-person reading | commissioned: groups of whole bloods run in one series, EPIC v1 |
| **IAM-Deconvolver** | IAM-Atlas v2.1 | nothing: it is a fit to known healthy cells | the atlas | cells the atlas does not hold, which are absorbed into a share | its leftover reading and the unexpected-cell flag | commissioned: blood-cell shares on arrays, tested on neutrophils, monocytes, CD4 T and CD8 T cells |
| **The information sphere** | the person's own other territories | the geometry (the Fisher-Rao distance) | the layout, once, from the atlas | changes shared by the whole sample, which it scales away | its own Fisher-Rao distance, Met-A for the whole sample | in development |
| **Distance to the ceiling** | the entropy ceiling physics sets for the sample's own expectation | entirely | nothing | a sample in the healthy range, where it says nothing | every other reading | in development |

## Lab methods: what the chemistry and the machines give, and which instrument reads each

DNA's sequence is the same in every cell; the methyl tags are not, and a sequencer cannot see a tag directly. So the DNA is first treated so that tagged and untagged spots read as different letters. **Bisulfite** (a salt solution) turns every untagged C into a U, which reads as T; a tagged C stays C. **EM-seq** does the same with enzymes instead, and damages the DNA far less. After either, a tag shows up as a C and a missing tag as a T. **Conversion failure** is an untagged C the chemistry missed: it stays C and looks tagged. The chain measures it on every sample, from the C's outside CpG pairs, which are almost never tagged in blood and most body cells, and prints it in every report. Then the treated DNA is read, either by an **array** (millions of DNA probes that light up red or green, one pair of colors per site, so each site gives one average over all the cells in the sample) or by a **sequencer** (hundreds of millions of short pieces, each one molecule from one cell).

| Method | What it gives | Single molecules? | Who uses it | Which IAM instruments read it |
|---|---|---|---|---|
| **Methylation arrays (Illumina 450K, EPIC v1, EPIC v2)** | One average per site (beta value) over every cell in the sample; 450,000 to 935,000 chosen sites; two-color readout | no | Most aging clocks (Horvath, GrimAge) and most cohort studies | Met-A, C-Score, Group-A, IAM-Atlas v2.1, IAM-Deconvolver (the commissioned readings on EPIC v1; on 450K, the atlas, the Deconvolver and Met-A for each cell read and flagged development) |
| **Whole-genome bisulfite sequencing (WGBS)** | Every one of about 28 million CpG sites, on single molecules about 100-150 letters long | yes | Reference atlases of human cell types; research | IAM-A, IAM-Atlas v2.1; read and flagged development: the IAM-Deconvolver and Met-A for each cell from sequencing values (with the read-count correction), the C-Score's sequencing reading, and the whole-genome IAM-A globe |
| **EM-seq (enzymatic conversion)** | The same readout as WGBS with gentler chemistry: longer, less damaged molecules from less DNA | yes | Newer research and clinical labs | IAM-A, IAM-Atlas v2.1; read and flagged development: the IAM-Deconvolver and Met-A for each cell from sequencing values, and the whole-genome IAM-A globe |
| **RRBS (reduced representation)** | A cut-and-select step keeps only CpG-dense regions (islands and promoters), about 1 to 3 million sites | yes, islands only | Lower-cost research studies | the held error is counted on RRBS (the mouse reproduction); no IAM-A reading, because its loose share comes from the rest of the genome, which RRBS skips; the C-Score's sequencing reading failed on it |
| **Targeted methylation sequencing (capture panels)** | Molecules from selected regions only | yes, selected regions only | Blood tests for cancer such as GRAIL's Galleri, which match a sample against patterns learned from people with and without cancer | not yet assessed |
| **MeDIP and MRE (enrichment)** | Pull out methylated (or cut unmethylated) DNA in bulk; no reading per site or per molecule | no | Older and lower-cost studies | none: there is nothing per molecule to count (a thermometer handed a barometer's printout) |
| **Nanopore and PacBio long reads** | Read the tag directly on long native molecules, no conversion chemistry | yes, long | Growing research use | IAM-A: a reader is not built yet (planned) |

**How IAM differs from cohort methods.** Aging clocks and cancer blood tests learn patterns from large groups of people and score a new sample by how closely it matches them. IAM's instruments measure a physical quantity against a floor and a ceiling set by physics. IAM-A goes further: it reads each sample against its own loose share, like checking for a fever with the back of your own hand on your own forehead instead of against a crowd's average. The C-Score reads each sample against its own spread in the same way, so a laboratory's offset cancels without any healthy cohort.

## IAM-Atlas v2.1

**The question:** What does each healthy cell look like, at every site, and how sure are we?

A map of 79 kinds of healthy human cells, built from 21 public sources on arrays and sequencing, fitted as one model so every spot on the genome carries its value and its uncertainty. Like combining many sensors into one estimate, each weighted by how far it can be trusted.

**Reads:** arrays or sequencing. **Where it stands:** commissioned (79 cell types; met all three sealed bars against both published reference atlases; commissioned for three cell references and the colon and lung tissue tests).

**The tests behind it:**

- ATL-060 Head to head: atlas v2.1 against the source atlases' own references, on their own job: run; last run: met
- ATL-001 Atlas commissioning as a source of healthy references: run; last run: met

**Earlier rounds and side tests** (kept as recorded; the record says what each one taught):

- ATL-005 Atlas v2.1 joint fit: run; last run: not met

**Record:** [COMMISSIONING_NOTE_ATLAS_V21.md](doors/COMMISSIONING_NOTE_ATLAS_V21.md)

**Next:** Deposited with its DOIs (fitted atlas 10.5281/zenodo.23441999; posterior draws 10.5281/zenodo.23422688). On the site every cell turns on the Globe Explorer, the noise maps from the posterior draws are drawn, and every report carries the sample on the same globe against its healthy reference. In the works: the information sphere (below, In development) and each cell's change layer on the globe.

## IAM-Deconvolver

**The question:** How much of each cell is in a mixed sample?

Reads a mixed sample against IAM-Atlas v2.1 and reports each cell's share with its range, after a sample check that refuses arrays it cannot read. On 200 fresh mixtures from a laboratory the atlas never saw, the shares came back within 0.016 of the truth (mean absolute error). Read and flagged development: every one of the atlas's 79 cells in one run; nine known mixtures of tissue DNA in blood were found at the right size through the one command (mean error 0.0057, nothing that was not put in above 0.0030; `doors/data/DEV_ONE_DRAW_01/run_03/mixes_03_summary.txt`).

**Reads:** arrays; sequencing values (development). **Where it stands:** commissioned (blood-cell shares on arrays, tested on neutrophils, monocytes, CD4 T and CD8 T cells).

**The tests behind it:**

- ATL-066 Deconvolution tool version 2 on fresh mixtures: run; commissioned (the record says on which bars)

**Earlier rounds and side tests** (kept as recorded; the record says what each one taught):

- ATL-065 Deconvolution tool on atlas v2.1: run; last run: not met

**Record:** [COMMISSIONING_NOTE_DECONV_V21.md](doors/COMMISSIONING_NOTE_DECONV_V21.md)

**Next:** Commission the B and NK cell shares; then the full 79-cell panel and plasma, both already read and flagged.

## Met-A

**The question:** How far has a tissue's tagging level drifted toward the ceiling?

Reads the methylation level at sites a healthy cell holds firmly, against a healthy reference, on cheap and plentiful arrays. Commissioned on isolated neutrophils. Each cell is read at its own identity sites, the sites it holds (where a loss can raise disorder) and healthy people agree on; reading every site is more sensitive but cannot say which cell is losing hold (DEV-EACH-CELL-01 exploration 4). Read and flagged development: Met-A for every cell the IAM-Deconvolver finds, each against its own profile in IAM-Atlas v2.1, on arrays and on sequencing values.

**Reads:** arrays. **Where it stands:** commissioned (isolated neutrophils).

**The tests behind it:**

- COM-METAA-003 Met-A commissioning on isolated neutrophils, EPIC v1: run; commissioned (the record says on which bars)

**Earlier rounds and side tests** (kept as recorded; the record says what each one taught):

- COM-METAA-039 Met-A read against the sample's own line: run; last run: not met
- COM-METAA-032 Whole blood back into Met-A scope: designed

**Record:** [COMMISSIONING_NOTE_METAA_NEUTROPHILS.md](doors/COMMISSIONING_NOTE_METAA_NEUTROPHILS.md)

**Next:** The sealed test of Met-A against the atlas with same-slide references; then every cell in a mix.

## C-Score (the clustering meter)

**The question:** Has the pattern started to clump?

Cancer does not only shift tagging levels; it clumps them into runs. The C-Score measures that clumping against a healthy band. The band held in three laboratories that never helped set it, and one person measured twice reads the same. Reproduced through the one command from the raw arrays: the band holds again.

**Reads:** arrays. **Where it stands:** commissioned (tested and commissioned on whole blood, EPIC v1).

**The tests behind it:**

- COM-CSCORE-011 C-score healthy band test: run; last run: met
- COM-CSCORE-012 C-score last check: same-person repeatability on unspent replicate arrays: run; last run: met

**Record:** [COMMISSIONING_NOTE_CSCORE.md](doors/COMMISSIONING_NOTE_CSCORE.md)

**Next:** Real cancers read on the gauge; then 450K, EPIC v2 and plasma; new same-person repeats from a series that did not help build the band. Its family (below, In development): the whole-genome C-Score against the atlas, and IAM-A's.

## IAM-A

**The question:** Is each mark still held at its own healthy rate?

Counts tagging mistakes one DNA molecule at a time, the most direct reading of informational fidelity we have, and reads them against the sample's own line. The physics end of the catalogue: the floor and ceiling are derived, and the healthy middle comes from the sample itself. Every reading states its own counting error: how many held-error chances it counted and the range that leaves.

**Reads:** single-molecule sequencing (bisulfite, EM-seq; nanopore next). **Where it stands:** commissioned (each sample against its own reference; sealed test on healthy blood CD8 T cells, whole-genome bisulfite).

**The tests behind it:**

- COM-IAMA-052 IAM-A atlas-free commissioning: one grand sealed test on fresh data: run; last run: met
- PHY-045 The loose rate read off the line: every healthy cell's held error from its own loose share: run; last run: met
- PHY-048 The loose-end line at the one-Landauer-bit cut: run; last run: met

**Earlier rounds and side tests** (kept as recorded; the record says what each one taught):

- PHY-049 The loose share from each sample's own molecules: run; last run: not met
- PHY-043 Tumor backbone against the writer's nameplate: run; last run: progressed

**Record:** [COMMISSIONING_NOTE_IAMA.md](doors/COMMISSIONING_NOTE_IAMA.md)

**Next:** Other healthy blood cells, then tissue and cell-free DNA; the tumor pairs read on the gauge; the firm rate derived again on the commissioned counter (the top physics item); the whole-genome IAM-A globe (development).

## Group-A

**The question:** Has a whole group moved (a trial arm, an exposure, a cohort)?

Single blood readings are too noisy for one person on arrays, but a group average is not. Group-A reads each person with the one command, tares each reading against healthy arrays run the same way, and compares the groups. On 278 healthy people from a laboratory it had never read, women and men came out 0.36 % apart, and a 1 % loss planted into half of them was found at 7.253 standard errors.

**Reads:** arrays. **Where it stands:** commissioned (groups of whole bloods run in one series, EPIC v1).

**The tests behind it:**

- GRP-004 Group-A on groups of whole bloods in one series: commissioning, then one cancer at a time: run; commissioned (the record says on which bars)

**Record:** [COMMISSIONING_NOTE_GROUPA.md](doors/COMMISSIONING_NOTE_GROUPA.md)

**Next:** One cancer at a time, patients and healthy controls from the same series.

## Support tools

Not instruments: the steps and pictures every reading passes through.

- **Intake and calibration.** Takes an array's raw intensities, checks every control, and turns them into tagging levels on one fixed scale, refusing what it cannot read and saying why.
- **The tare.** Reads each specimen against healthy references run beside it, so a laboratory's own offset cancels instead of passing for disease.
- **Composition.** For whole blood, works out which cells are in the sample and in what shares, so a change in the mix is not mistaken for a change in the cells.
- **The chain and its report.** Runs every stage in a fixed order, records every input and setting, and prints one report per sample with each instrument's reading, its status and its detection limit.
- **The sky sphere and the Globe Explorer.** Lays every site of IAM-Atlas v2.1 on one sphere in genome order, the way a microwave sky is drawn: equal area, no edges, clustered change shows as patches, and the microwave-sky toolset applies. Every report draws the sample on it against its healthy reference; the change is fixed by the measurement (blocks of 10 neighboring sites beyond the 5 % family-wise line), and a slider only fades the unchanged background. The Explorer carries the atlas's 79 cells and the example readings.

## In development

Read and flagged development, by the same chain; each becomes an instrument only through its own sealed test.

### The information sphere

The second sphere. The sky sphere places a site by where it sits on the genome; the information sphere places it by how it behaves across the 79 healthy cells, so sites that act alike land together wherever they are on the DNA, and the distance between two states is the Fisher-Rao distance of information geometry, the local form of relative entropy. The sky sphere says where a change is; the information sphere says what kind of change it is. Its construction takes each site's level out first, because on real cells a site's level otherwise sets its place. The synthetic tests pass; on the real atlas it met two of three checks, and the next test reads the third against level-matched groups.

**Next:** The test on the real atlas with level-matched lines; then a person's information sphere against their own healthy one. **Record:** [DEV_SPHERE_01.md](doors/DEV_SPHERE_01.md)

### The C-Score family

The C-Score reads how a change is laid out along the genome, not how large it is, against the sample's own spread, so a laboratory's offset cancels. Three versions: the commissioned one at Met-A's neutrophil sites; the whole-genome version on the globe's change map against IAM-Atlas v2.1; and IAM-A's, the clustering of held-error departures along the genome on single molecules.

**Next:** Each version read on healthy samples first, then on tumors. **Record:** [COMMISSIONING_NOTE_CSCORE.md](doors/COMMISSIONING_NOTE_CSCORE.md)

### Every cell in a mix

One run of the one command finds every cell of IAM-Atlas v2.1 in a sample and reads Met-A for each, against its own healthy profile; each cell also gets its own change layer on the globe. Sites held by one cell alone are the next test, so a loss across a cell's whole pattern can be put on the right cell.

**Next:** Each cell's healthy spread across Group-A's 278 bloods. **Record:** [DEV_EACH_CELL_01.md](doors/DEV_EACH_CELL_01.md)

### One blood draw

One tube, two halves: the buffy coat (the person's own white cells, long DNA) and the plasma (cell-free DNA). The one command reads both and writes one record, including what the plasma holds that the person's own white cells do not explain, and IAM-Atlas v2.1 says which tissue it came from. Constructed draws with known answers test it; a public set of matched halves from healthy people is held unread for the test that counts.

**Next:** The smallest share each instrument reads at each depth; then the test on the held set. **Record:** [DEV_ONE_DRAW_01.md](doors/DEV_ONE_DRAW_01.md)

## Sex and the X chromosome

<!-- SEX-SECTION:BEGIN -->
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
<!-- SEX-SECTION:END -->

## Candidate instruments



Candidate instruments: every reading we could build on the methylation channel, each with the term of the law it reads, its substrate, its blind spot, its advantage, what would break it, and the next concrete step. The point is NOT to pick one - several instruments with different blind spots, agreeing on the same sample, is worth more than one instrument we trust. Priority is by (decides something) x (free or cheap) / (blocked).

### 1. Territory contrast: fidelity in fragile territory over fidelity in stable territory

- **Reads:** WHERE the loss happens first; a sustained energy shortfall should take the late-replicating, solo-WCGW territory before it touches housekeeping promoters
- **Substrate:** arrays (works today, no new data) · **Cost:** free (design, synthetic test, then existing arrays)
- **Where it stands:** RUN ONCE (DEV-TERRITORY-01): the balance equation's parameter-free prediction of the response ratio, 4.158 from healthy baselines alone, was met to 1 % (measured 4.12-4.20 at shortfalls of 2-20 %), and the declared blind spot was confirmed; but the NUISANCE bar FAILED
- **Still open:** v2: report both the difference and the normalized difference (one cancels each nuisance type), measure the beta-ceiling effect from fully held sites and subtract it, and leave composition to the mixture-predicted denominator the chain already uses
- **Advantage:** SELF-TARING; a ratio inside one array, so plate, batch and much of the composition effect cancel because numerator and denominator see the same cell mixture. It addresses the laboratory-to-laboratory spread that limits single whole-blood readings
- **Blind spot:** says nothing about absolute fidelity; a loss uniform across territories is invisible (confirmed, 0.1 % of the signal). And the practical failure: a difference cancels an ADDITIVE plate offset exactly but cannot cancel a MULTIPLICATIVE one; a 10 % composition change moves it about 5x what a 2 % shortfall does; while the beta = 1 ceiling makes offsets and noise one-sided
- **What would break it:** the ordering reversing; stable territory losing before fragile territory in a condition known to cost the cell energy
- **Next step:** write DEV-TERRITORY-02 with the two-number signature and the measured ceiling floor, bars first
- **Record:** [DEV_TERRITORY_01.md](doors/DEV_TERRITORY_01.md)

### 2. Met-A on whole blood: each cell in the mix, and the whole-sample reading

- **Reads:** total disorder above what the estimated cell mixture explains; the existing composition-aware reading, in a tube of blood
- **Substrate:** arrays · **Cost:** free (one laptop read of arrays already on disk)
- **Where it stands:** MEASURED on real healthy arrays, read by the chain itself (run_neutrophil, in the C-score band test), healthy people differ by 2.01 % (relative SD; 95th percentile +3.33 %), and the chain's leverage is 2.12 % of the reading per 1 % pattern loss, so about a 1.57 % loss clears the healthy 95th percentile. The first measurement (6.36 %) skipped the chain's self-tare step (Stage T step 1) and is withdrawn. GROUP ARM (DEV-GROUP-01): the sham passes; two healthy groups from two laboratories agree to 0.16 % after taring, but differ by 3.69 % UNTARED, at 6.5 standard errors. A group study is admissible only when patients and healthy comparisons are run in the SAME series. Read and flagged development: Met-A for each cell in a mix; and the every-site reading, more sensitive than the identity sites but unable to say which cell moved
- **Still open:** per-person detection in whole blood, now open: the corrected spread is one laboratory's 23 arrays and one leverage array; a sealed per-person test across laboratories decides it
- **Advantage:** group comparisons are within reach NOW: with 50 people a mean loss of about 0.27 % separates the group from healthy, and the function already exists in the chain
- **Blind spot:** an array averages molecules before anyone reads them, so one cell type can never be isolated. Across laboratories without a shared series the raw reading carries a 3.7 % offset, which is inside the tumor range, so every comparison needs the same-series tare
- **What would break it:** healthy people differing by more than the leverage of the signal we are looking for (then arrays cannot carry blood and sequencing is required)
- **Next step:** the every-site reading's healthy spread across Group-A's 278 bloods; then a sealed per-person test in whole blood, across laboratories
- **Record:** [DEV_BLOOD_SENSITIVITY_01.md](doors/DEV_BLOOD_SENSITIVITY_01.md)

### 3. Distance to the ceiling

- **Reads:** how close a sample sits to the derived entropy ceiling; the right-hand mark of the gauge, past which the reading stops rising with further loss
- **Substrate:** arrays and sequencing · **Cost:** free (arithmetic on readings we already have)
- **Where it stands:** the ceiling is in the foundation as derived. AUDIT DONE (DEV-CEILING-AUDIT-01): of 7,681 readings that record the flag, none is past the ceiling. Now READ BY THE CHAIN (development): every atlas reading (Met-A for the whole sample and for each cell, the territories) carries its saturation, its distance to the ceiling and its turnover share, and the report shows them
- **Still open:** re-read the three sealed tissue sets with the flag recorded (free, arrays on disk, reported as an audit that cannot change their sealed verdicts); collect the distance on every array the chain reads
- **Advantage:** derived with no free parameter, and it answers how far from the ceiling a tumor arrives
- **Blind spot:** only informative for samples near the ceiling; says nothing about a healthy-range sample
- **What would break it:** a sample reading past the ceiling, which the derivation forbids
- **Next step:** compute it for every sealed cancer reading already in the records and check none is past the turnover
- **Record:** [DEV_CEILING_AUDIT_01.md](doors/DEV_CEILING_AUDIT_01.md)

### 4. Run-length spectrum of losses

- **Reads:** the SHAPE of clustering; exponential against heavy-tailed, which separates a processive mechanism from independent point losses
- **Substrate:** sequencing only (needs several sites on one molecule) · **Cost:** laptop work on reads already in storage; no new sequencing
- **Where it stands:** not written
- **Still open:** the derivation of the expected shape under each mechanism, then a synthetic test. Gated: the head-of-run bias the trim rule was written for (it held the earlier sequencing reading, since retired) must be checked on IAM-A's commissioned counter before a per-molecule reading spans laboratories
- **Advantage:** same data as the clustering meter, different question; distinguishes mechanism rather than only size
- **Blind spot:** impossible on arrays; needs depth, so shallow libraries are out
- **What would break it:** the measured shape matching neither mechanism's prediction in a case where the mechanism is known
- **Next step:** after that check on IAM-A's commissioned counter
- **Record:** [DEV_IAMA_TRIM_01.md](doors/DEV_IAMA_TRIM_01.md)

### 5. Tail of fragments in plasma (the early-detection instrument)

- **Reads:** the few badly disordered MOLECULES in a plasma sample, not the average; because early cancer is a tiny minority of fragments and an average hides it by construction
- **Substrate:** sequencing of cell-free DNA · **Cost:** box time per sample; the expensive one
- **Where it stands:** not written; gated
- **Still open:** the head-of-run check on IAM-A's commissioned counter, and a healthy plasma reference; the one-draw work builds toward it, and a public set of matched halves is held unread for the test that counts
- **Advantage:** the highest reward on the whole board; it is the blood test, and a tail statistic is the right shape for early detection
- **Blind spot:** needs depth and fragment length; a shallow or heavily fragmented sample has no tail to read. Composition of plasma is itself a question
- **What would break it:** the tail in healthy plasma being as heavy as in known cancer plasma
- **Next step:** behind the head-of-run check and a healthy plasma reference
- **Record:** [DEV_IAMA_TRIM_01.md](doors/DEV_IAMA_TRIM_01.md)

### Not an instrument: the trim rule, and the head-of-run check it calls for

The earlier sequencing reading (Stage B) was held because the same kit read differently in two laboratories, and the cause is where in each file it read: at the head the failing libraries read 11-14 % below the controls read at the SAME position. Reading more places failed its planted test - pooling divides a bias rather than removing it. The remedy, DEV-IAMA-TRIM-01, excludes the unsettled head and decides how much from the raw reads' own chemistry (the non-CpG C-to-T fraction in windows, no alignment and no IAM-A): six planted bars, all met. That reading was since retired; IAM-A was commissioned on a different counter (Stage Q2). Whether the head-of-run bias touches the commissioned counter is checked before any per-molecule reading spans laboratories. Candidates 4 and 5 are both behind that check.

_Several instruments, each with a different blind spot, reading the same sample and agreeing - that is worth far more than one instrument we trust. Disagreement is also informative: a sample that reads normal on total disorder and abnormal on clustering is telling us which channel failed._


