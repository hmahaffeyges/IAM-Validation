# The Chain Workflow (draft: Parts A to F)

A step-by-step map of how the chain handles a sample, laid out like an emergency action plan: each box asks one question, and each
answer says where the sample goes next. No code. Every box names the Decoder section it comes from (`manual/OM_v4_decoder.md`), so every
statement traces to the chain itself. This draft holds the parts that are settled. The rest follows the outline
(`CHAIN_WORKFLOW_OUTLINE.md`) as the commissioning work settles them.

**Why the chain is built this way.** A cell pays to keep each methyl mark against the kicks of heat. A mark that cost holds is a held mark;
one that has slipped is disorder. Every instrument compares a sample's disorder with the disorder of healthy cells of the same kind,
measured the same way. So before anything is read, the chain makes sure of three things: what the sample is, that it was measured well,
and which healthy ruler it can fairly be read against. Parts A to C are those three things.

---

## Part A. What do you have?

### A1. How was the DNA measured?

| Answer | Go to |
|---|---|
| An array: a pair of IDAT files (green and red), or a table of calibrated values | Part B |
| Sequencing that keeps molecules: a `.pat` file or an aligned bisulfite BAM | Part C |
| Sequencing kept as levels per site: a per-CpG count file (`.beta`), a call and coverage bigWig pair, or a per-site table such as Bismark coverage (converted first) | Part C (molecule readings cannot be made from it; see C5) |
| Anything else | Stop: the chain has no reader for it |

Decoder: §1 (the one command), §10, §21.

### A2. What is the specimen?

The specimen is checked before a single byte is read.

| Answer | What happens |
|---|---|
| Whole blood (also "blood", "peripheral blood", "venous blood") | accepted |
| Buffy coat | accepted, read on the whole-blood path, labeled preliminary |
| Isolated, sorted or purified neutrophils | accepted |
| Isolated, sorted or purified cells of any IAM-Atlas v2.1 cell type | accepted; read by Met-A against the atlas, flagged where not commissioned |
| Plasma (cell-free DNA) | accepted |
| Tissue | accepted |
| Constructed DNA mixtures of blood cells | accepted as test material |
| PBMC, bone marrow, cell lines, a specimen given as blank, anything else | **refused**, with the reason; nothing is read |
| No specimen given at all | read as whole blood (the one command's default); making the specimen optional, with the Deconvolver naming what the sample holds, is plan item 61.6 |

**Why:** a reading means something only against a healthy reference of the same kind of sample. A specimen with no reference is refused
rather than read against the wrong one. Decoder: §2, the specimen check (`specimen_refusal`).

### A3. Is it the whole sample?

A sequencing run is read in full: every read pair of every run. Some runs are stored in genome order, so their first read pairs come from
the first chromosomes only, and a sample read from "the first N pairs" holds part of a genome. The chain checks this itself (box C3).
Decoder: §21, `genome_coverage`.

---

## Part B. The array path

### B1. Did the files arrive whole? (Stage 0.1 to 0.3)

| Check | Stops the array when | Limit |
|---|---|---|
| Arrival | a required manifest field is missing, a channel is missing, a file is under 1 MB, or the header's array family differs from the one declared | fixed rules |
| Manifest | the identifier is not already a hash (spaces, an @, under 16 characters) | fixed rule |
| Integrity | the same bytes arrive again for the same array (a re-transmission) | SHA-256 of both files |

### B2. Which platform?

Read from the IDAT header's address count: under 800,000 is a 450K, under 1,080,000 an EPIC v1, otherwise an EPIC v2.

| Platform | Goes |
|---|---|
| EPIC v1 | the commissioned route |
| 450K | on to the atlas readers, flagged; the commissioned Met-A and C-Score refuse it themselves, because their EPIC references are only about half covered by a 450K chip |
| EPIC v2 | refused on the main route before calibration (the calibrator does not read it); it has a development route of its own |

Decoder: §1 steps 3 and 7, §2 (`_infer_platform`, step 0.7b).

### B3. Was the array measured well? (Stage 0.4 to 0.9)

| Check | What it measures | Outcome | Limit and where it comes from |
|---|---|---|---|
| 0.4 control probes | bisulfite conversion, hybridization, extension | conversion below the line stops the array | conversion at least 0.80, set on 337 healthy arrays of three laboratories |
| 0.5 detection | share of probes above the array's own background | below 0.93 stops; 0.93 to 0.99 read with a penalty | intake thresholds file |
| 0.6 bead counts | share of probes on at least 3 beads | below 0.995 a penalty; never stops | intake thresholds file |
| 0.7 call rate | share passing detection and beads | below 0.93 stops; 0.93 to 0.98 penalty | set on 48 arrays, 12 from each of four laboratories |
| 0.8 sex | Y probes against X probes | a mismatch with a declared sex stops the array | the field's standard rule |
| 0.9 decision | every step's outcome | any stop: quarantine; any penalty: proceed with a penalty | a value that could not be measured is a stop, never a pass |

**Every refusal prints five things:** what was measured, the limit, where the limit comes from, what it means, and what to do.
For example, a lung tumor array with conversion 0.770 is refused: unconverted DNA reads as methylated, so every value is pushed up.
Decoder: §2, §3.

### B4. Calibration (Stage 1)

The IDAT pair becomes one value per site, on the chain's own scale, the same scale IAM-Atlas v2.1 was built on. Stage 1's own quality
numbers are recorded and never gate. Decoder: §4.

### B5. Then

Go to Part D, the choice of ruler (IAM-Atlas v2.1 for arrays).

---

## Part C. The sequencing path

### C1. Is the file readable? (Stage Q0, for a `.pat` file)

| Check | Stops the file when |
|---|---|
| Specimen | the cell type has no healthy reference on this route |
| Readable | a line is malformed or the file is cut short |
| Genome build | any CpG index lies outside its chromosome's range in the hg19 dictionary |
| Whole genome | an autosome has no molecules at all |
| Conversion | the alignment record's conversion is below 0.98 (when given) |

Decoder: §10.

### C2. Which genome?

Each reading counts on its own genome build, and the chain keeps the builds apart. A table on GRCh38 is lifted to the atlas's numbering
before it is read. Decoder: §10b.

### C3. Is the whole genome there?

For each autosome the chain takes the share of sites carrying a value and compares it with the median autosome's share. **An autosome
under half the median refuses the sample**, and the refusal names the chromosomes missing.

- A sample cut to chromosomes 1 to 8 is refused, naming chr9 to chr22.
- A sample missing chromosome 17 is refused, naming chr17.
- A thin library, such as plasma (5 % of CpGs), is thin on every chromosome alike and passes.

**Why:** a run read only in part looks like a sample with a strange cell mix, and every reading downstream would be wrong. Decoder: §21,
`genome_coverage` (record DEV-SEQ-INTAKE-01).

### C4. The library's own floor

Every library has its own small rate of false "methylated" calls from its chemistry. The chain measures it at the sites no healthy cell
marks, before any reading, and treats it as a property of the library, not of the cell. Decoder: §21, `library_floor`.

### C5. Molecules or levels?

| The file keeps | Readings possible |
|---|---|
| Molecules (`.pat`, BAM) | every reading, including IAM-A and Coupling-A, which need each molecule's pattern |
| Levels only (count tables, bigWigs) | the level readings (Met-A, the territories, the cell mix); IAM-A and Coupling-A are not read, and the report says so |

Decoder: §9, §15.

### C6. Then

Go to Part D, the choice of ruler (the IAM Sequencing Atlas for sequenced samples).

---

## Part D. Which ruler?

Every reading compares the sample with healthy cells of the same kind, measured the same way. The ruler is that healthy reference.
A sample is never read against a ruler built on a different chemistry without the reading saying so.

### D1. Each instrument's own ruler (the commissioned readings)

| Instrument | Its ruler | Commissioned on |
|---|---|---|
| Met-A | healthy purified neutrophil arrays, at the 6,000 identity sites | isolated neutrophils, EPIC v1 |
| C-Score | the same sites, read for how departures cluster | whole blood, EPIC v1 |
| Group-A | healthy references from the same run, the same slide first | groups of whole bloods in one series, EPIC v1 |
| IAM-Deconvolver | IAM-Atlas v2.1's columns for the six blood cells | blood-cell shares on arrays |
| IAM-A | **the sample itself**: its own held error against the error its own loose share predicts; no other person enters | healthy blood CD8 T cells, whole-genome bisulfite |

A commissioned reading outside its commissioned scope (another platform, another cell) is still read and printed, with a flag saying
it is not commissioned there.

### D2. The atlas rulers (every cell, flagged where not commissioned)

| The sample is | Ruler | Why |
|---|---|---|
| An array (450K, EPIC v1, EPIC v2) | **IAM-Atlas v2.1**: 79 healthy cell types at 813,951 array sites, fitted once across laboratories, with a range at every value | it was measured with the array's own chemistry |
| Sequenced, read with `--dev-seq-ref auto` | **The IAM Sequencing Atlas**: 57 healthy cell types at every CpG, counted from healthy donors' reads | it was measured with sequencing's own chemistry, so the platform offset does not enter |
| Sequenced, without that setting | IAM-Atlas v2.1, at the sites both hold, and the reading says so | |

**Today the sequencing atlas is chosen by a setting, not by the chain.** Every healthy and replay pass on sequencing gives it; choosing it from
the input itself is part of the one-button intake (plan item 61.6).

### D3. One person, one sample, every cell

With `--cell-panel atlas`, the IAM-Deconvolver splits the sample across every cell type of the active atlas, not only the six blood cells,
and every cell it finds is read against its own column (Part E). With `--full-chain`, every development reading that applies to the input
runs beside the commissioned ones, each flagged; a reading that cannot run here is listed with its reason, never left out.

### D4. A laboratory the atlas has not seen

An atlas is built in a few laboratories. A sample from another laboratory can carry that laboratory's own small offset: on sequencing,
two outside laboratories read lower than the atlas's own laboratory on the same cells. The answer under test is the **same-laboratory tare**:
a few healthy samples of the same cell from the same laboratory set that laboratory's level, and every other sample from it is read against
them. On the two outside laboratories, three of their own healthy samples brought them to Normal. *This box is completed from the test on a
laboratory never read before, now running.*

---

## Part E. What cells are in it?

A blood draw, a tissue or a plasma is a mixture. Disorder means something only against the healthy disorder of the same cells in the same
proportions, so the chain finds the mixture first, and builds each person's own expected healthy self from it.

### E1. Split the sample into its cells (the IAM-Deconvolver)

| Step | What it does |
|---|---|
| Markers | for each cell, the 50 sites where it sits furthest above, and the 50 furthest below, its nearest other cell, in units of their combined healthy spread |
| Solve | fits the shares that best rebuild the sample at those sites, each site weighted by how much healthy people vary there |
| Range | 200 refits on resampled markers and on cells redrawn from the atlas's spread (a person's cells are not the average cell) |

**Commissioned on:** the shares of neutrophils, monocytes, CD4 T and CD8 T cells on arrays, tested on 200 mixtures of purified arrays from a
laboratory the atlas never saw: mean share error 0.016, against a bar of 0.03. Its 95 % range covered the truth 78 % of the time, so the
range is printed as information, labeled not calibrated.

| Panel | Cells | Status |
|---|---|---|
| `blood` (the default) | neutrophils, monocytes, B cells, CD4 T, CD8 T and NK cells | commissioned for the shares of four (neutrophils, monocytes, CD4 T, CD8 T); B cells and NK cells flagged |
| `atlas` | every cell type of the active atlas (79 on arrays, 57 on sequencing) | read and flagged |

Decoder: §7.

### E2. Is a cell really there?

| Question | Rule |
|---|---|
| Below 1 % of the sample? | that cell's own reading is printed as *not a reading of that cell*: too little of it to read |
| On sequencing, fewer than one copy of the cell behind each site? | the share times the reads at each site; below one copy the cell's reading is flagged |
| A cell the atlas does not hold? | it cannot be named; plasma's megakaryocyte DNA is read through a provisional column, flagged |

### E3. The person's own expected healthy self

Each cell's healthy level, from the atlas, mixed at the shares the Deconvolver found in **this** sample. Every whole-sample reading and the
globe read the sample against this expectation, not against a population average. That is how composition leaves the reading: a person with
more neutrophils is not read as abnormal at every neutrophil site.

### E4. Then

Each cell found goes to Part F, read on its own identity sites against its own healthy column; the whole sample goes to Part F as one.

---

## Part F. The readings

Every reading answers one question about how well the sample's cells are holding their marks, and prints on one gauge: **1 is the healthy
self**, above 1 is more disorder than healthy, toward 0 is the most disorder the cell can reach. Each says whether it is commissioned for
this sample's platform and cell, or read and flagged.

| Reading | The question it answers | The physics behind it | Needs | Status |
|---|---|---|---|---|
| **Met-A** | Are this cell's identity marks wearing? | a mark held at a cost slips toward disorder when the cost is not paid; the identity sites are where a healthy cell holds firmly, so wear shows there first | arrays, or sequencing against the IAM Sequencing Atlas | commissioned: isolated neutrophils, EPIC v1; every other cell, and sequencing, flagged |
| **C-Score** | Are the changes clumped into runs? | wear from heat is scattered; a failing program changes neighbors together. It reads the layout, not the size, so a laboratory's offset cancels | arrays | commissioned: whole blood, EPIC v1 |
| **IAM-A** | How many tagging mistakes does each molecule carry, against what its own loose share predicts? | the held error has a floor and a ceiling set by the law; the healthy middle comes from the sample itself | molecules (sequencing that keeps reads) | commissioned: blood CD8 T cells, whole-genome bisulfite |
| **Group-A** | Has a whole group moved? | one reading is noisy; the average of many, each tared on healthy references run the same way, is not | arrays, groups in one series | commissioned: whole blood groups, EPIC v1; on sequencing flagged |
| **Coupling-A** | Do a molecule's mistakes fall one at a time, or together? | independent slips scatter at random; coupled slips mean the neighbors' upkeep failed together | molecules | read and flagged |
| **The territories** | Are the sites no cell marks gaining marks (the islands), and are the large marked blocks holding (the domains)? | the islands are held unmarked at a cost, so a gained mark is disorder; the domains are held marked | arrays or sequencing | read and flagged |
| **Distance to the ceiling** | How far is the sample from the most disordered state its own cell can reach? | the ceiling is set by the law for each cell | arrays or sequencing | read and flagged |
| **The globe** | Where on the genome is the sample different from its own expected healthy self? | every site in its place on a sphere, read against its healthy range | arrays or sequencing | a picture with numbers, read and flagged |
| **The information sphere** | Are the changed sites ones that behave alike across healthy cells? | sites placed by how they behave, not where they sit; distance is information distance | arrays (sequencing on v2.1's layout) | read and flagged |

**A reading that cannot run** on this sample (no molecules, too few sites, a cell not present) is printed with its reason, never left out.
Decoder: §5 (Met-A, the C-Score), §6 and §9 (IAM-A), §11 (Group-A), §14b and §17 (the globe), §15 and §16 (the development readings), §18 (the information sphere), §21 (sequencing).

---

*Parts G to I (the tares, the maps, two draws, the report) follow the outline and are written as
each is settled. The same-laboratory tare for sequenced samples is under test on a laboratory the atlas has never held; its box is written
from that test's result.*
