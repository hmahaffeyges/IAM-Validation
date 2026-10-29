# An Introduction to IAM and the Physics of Methylation

**How our instruments work and what makes them different, with a Q&A for researchers**

This is the first document to read in the researcher package. It explains, in plain language, what IAM is, the physics behind it,
each instrument and the atlas, and every test we present: what it showed, and how it differs from the way methylation is usually
measured. Every number here comes from a committed record, linked where it appears, and a check run on every change to the repository
refuses a number that no record holds. When you want to go deeper, the [Q&A for researchers](QUESTIONS_AND_ANSWERS.md) answers the
questions reviewers ask first, and the [instrument catalogue](INSTRUMENTS.md), the [commissioning report](COMMISSIONING_REPORT.md) and
the [A-score database](ASCORES/README.md) hold the details.

## 1. The idea in one paragraph

A methyl mark on DNA is a record. A cell writes it, and then has to keep it against the constant kicks of heat. Keeping a record costs
energy, and physics sets the price: Landauer's principle gives the least energy needed to hold one bit against heat. IAM's Law is that
the same cost governs every physical record, a qubit, a transistor, the large-scale structure of the universe, and a methyl mark. For a
cell, the law gives a scale that is not fitted to any cohort: a floor, where heat alone would set the error, and a ceiling, where a site
has lost its mark entirely and reads as a coin flip. A healthy cell sits at a measured place between the two. IAM's instruments read
where a sample sits on that scale, in bits, and whether it has moved.

## 2. The physics of holding a methyl mark

- **The price.** A cell pays about one ATP to keep each methylated site, the M number M = 20.94 in units of the thermal energy
  k_B T at body temperature ([the Landauer chapter](https://hmahaffeyges.github.io/IAM-Validation/book/ch_landauer.html)).
- **What the cell actually spends.** Counting single DNA molecules in 56 healthy human cell types, the energy that holds a methyl mark
  is 3.08 k_B T, the same in every cell type to within its spread: about 0.147 of one ATP
  ([holding_energy_q2.json](doors/data/DEV_CHAIN_Q2_01/holding_energy_q2.json)). One energy, across every healthy cell type measured.
- **The gauge.** The floor and the ceiling come from physics. The middle mark, where a healthy sample sits, is where the instruments
  differ. For Met-A it is measured on healthy reference samples. For IAM-A it is computed from the sample itself: a cell's design holds
  part of its genome firmly and leaves a share L loosely held, so its expected error follows from its own L, on a line whose form comes
  from that physics and whose two constants were counted once on 56 healthy cell types. No control group sets it
  ([the gauge chapter](https://hmahaffeyges.github.io/IAM-Validation/book/ch_gauge.html)).
- **A prediction made before the data.** Cutting the supply of methyl groups should move the reading by a calculable amount. We wrote
  down the window, 1.016 to 1.135, before reading the mice. The mice read 1.1239, inside it
  ([DEV_SAM_LEVER_01_OUTCOME.md](doors/DEV_SAM_LEVER_01_OUTCOME.md)). Methylation studies usually find associations; this was a physical
  prediction, checked.

**Why this is different.** Most methylation measures are learned from data: a clock or a classifier is trained on cases and controls,
and its scale means whatever the training set made it mean. IAM's scale comes from thermodynamics. Its ends are derived, and for IAM-A
even the middle mark comes from the sample itself, so a reading means the same thing in any laboratory, for any cell type, before anyone
has trained anything.

## 3. Reading the genome with the methods built to map the sky

Cosmologists solved a problem that methylation research shares: how to read a faint signal spread across millions of positions,
through noisy instruments that each have their own offsets, and still say how sure you are at every point. The Planck and WMAP missions
mapped the microwave sky on a sphere divided into equal pixels (HEALPix). We place every CpG site of the genome on that same sphere,
at a fixed place set by where it sits in the genome, so every cell and every sample shares one map and can be compared site for site.
The sphere is chosen for what it does: every pixel has the same area, so no part of the genome is stretched; there are no edges or seams;
neighbors in the genome share small patches, so clustered change shows as patches and scattered noise as speckle; and the sky's masks
apply unchanged. That map's layout is the genome's address, so a spectrum on it measures bookkeeping, not the order a cell holds
([turn it on the globe](https://hmahaffeyges.github.io/IAM-Validation/atlas/sphere.html): the Globe Explorer, where every cell of the atlas,
Met-A's purified-array reference and example samples can be turned, and every report carries its own sample on the same globe against
its healthy reference). A second sphere, in development, maps the order itself: information is
spherical in a strict sense (a site at methylation level b is the point (√b, √(1−b)) on a circle of radius 1, and the distance between two
states is the angle between their points, the Fisher–Rao distance), so sites that are held in order together can be placed together, the
way a brain scan maps activity rather than wiring ([DEV-SPHERE-01](doors/DEV_SPHERE_01.md)).

What that brings, each from the sky to the cell:

- **Map-making with a posterior.** The atlas was fitted the way a sky map is made: every source of data fitted at once, each source's
  own offset fitted beside the biology (what cosmologists call destriping), and an uncertainty at every value.
- **Masks and foregrounds.** The sky map found the X chromosome by itself: one reference array read as female, with 40 % of her X
  chromosome sites between β = 0.3 and 0.7 against 8 % in the other five. In the sky, that is a foreground, and the answer is the same: a
  mask, or a reference of the same sex ([the sky tools chapter](https://hmahaffeyges.github.io/IAM-Validation/book/ch_skytools.html)).
  Every instrument accounts for sex and the X chromosome, each in its own way, with every count computed from the chain's files: see
  [Does each instrument account for sex and the X chromosome?](QUESTIONS_AND_ANSWERS.md#does-each-instrument-account-for-sex-and-the-x-chromosome).
- **Pattern, not single points.** A power spectrum asks whether changes are scattered or clustered. That question became the C-Score.

No other methylation method we know of reads the genome as a map with these tools.

## 4. The instruments

Each instrument is **commissioned**: it was tested on samples it had never seen, from other laboratories, against a rule written and
sealed before the data was read, and it met every bar of that rule. Commissioned means its readings can be reported, in the scope it
was tested on.

### Met-A: how far a cell's identity has drifted, in bits

Every cell type keeps a set of identity sites: places that are firmly marked, or firmly unmarked, in that cell. Met-A reads how much
those sites have drifted toward a coin flip, as entropy in bits, against the cell's own healthy profile, on a scale from the physics
floor to saturation. A healthy sample reads 1.
**Commissioned** on isolated neutrophils, EPIC v1 arrays. The six reference arrays each read Normal against the other five;
it detects a 2 % change ([commissioning note](doors/COMMISSIONING_NOTE_METAA_NEUTROPHILS.md)).
**Different because** it reads the state of the record itself, in physical units, instead of a difference from a control group's
average at single sites.

### C-Score: are the changes scattered or clustered?

Healthy aging wears marks away here and there. Tumors break whole stretches. The C-Score reads how clustered a sample's departures are
across the genome's map.
**Commissioned** on whole blood. Its healthy range, upper limit 1.152, was set on 617 arrays from 19 laboratories, and then
held in two laboratories that played no part in setting it: 34 of 35 and 22 of 23 healthy people inside. The same person measured twice
reads the same (spread 0.0379 over 13 pairs). It detects a change on only 5 % of sites when they cluster, and planted scattered losses
never read as a departure: 0 of 120 ([commissioning note](doors/COMMISSIONING_NOTE_CSCORE.md)).
**Different because** arrays are usually said to see changes of about 20 to 30 % at a site. The C-Score reads the pattern across
thousands of sites at once, so it sees far smaller changes, and it tells ordinary aging apart from the pattern tumors leave.

### IAM-A: each person read against themselves

IAM-A counts, on single DNA molecules from sequencing, how often a held mark has been lost where its neighbors kept theirs, and compares
that with what the same sample's own loose share predicts on the healthy line. Its middle mark, the healthy 1 on the gauge, is computed
from the sample itself, so no control group enters the reading.
**Commissioned**. The self-reference line comes from 56 healthy cell types. On a sealed test with healthy CD8 T cells from a
laboratory it had never seen, 7 of 8 read Normal (band 0.779 to 1.055). The one command a researcher runs reproduces the reading of
a test sample exactly, 0.9083 ([commissioning note](doors/COMMISSIONING_NOTE_IAMA.md)).
**Different because** a healthy baseline in the field is the average of a control group, and everyone has a different baseline.
Laboratories that work on sorted cells say a person's own earlier sample is the more powerful reference. IAM-A goes further: the
reference is inside the sample.

### IAM-Atlas v2.1: 79 healthy cell types, with how sure we are at every value

The atlas holds the healthy methylation profile of 79 purified human cell types, fitted at once from 21 public sources as one
hierarchical model, with a posterior, so every value carries its own uncertainty.
**Commissioned**. On 731 held-out arrays it beat both published reference atlases on the three jobs an atlas is used for:
telling cells apart, splitting mixtures, and handling cells a reference does not hold. Against the published sequencing atlas of human
cell types, built the way that atlas's own paper builds it: 0.998 against 0.938 for telling cells apart, a median mixture error of 0.0200
against 0.0248, and 0.0232 against 0.0484 with cells it does not hold mixed in ([commissioning note](doors/COMMISSIONING_NOTE_ATLAS_V21.md),
[the head-to-head](doors/DEV_ATLAS_USE_02.md)). Deposited with its DOI,
[10.5281/zenodo.23441999](https://doi.org/10.5281/zenodo.23441999).
**Different because** published references are built from a handful of sorted samples per cell type and give one number per site with no
uncertainty. This atlas says how sure it is, site by site and cell by cell, and every cell has its own page on the website.
**What is a tool and what is physics.** The atlas is a ruler, and a ruler has to be one anyone can rebuild: the same public data and the
same model give the same atlas. Others use their atlases to find which cells a sample came from. IAM uses this one to build each
person's own expected healthy self, from their own cell mix, and reads every site's departure from it with no cohort. IAM's law is what
gives those departures their meaning: the floor and the ceiling, the entropy of a mark measured against the floor, and how sensitive each
site is at its held level turn a departure into an instrument reading, in bits, the same in any laboratory. The law also checks the
ruler: at the sites a cell holds firmly, the atlas's healthy level matches one minus the error that cell's own molecules show, counted
directly ([the record](doors/DEV_METAA_HEALTHY_01.md)). Its values are on the array scale (its sequencing sources are placed on that scale), and it is the ruler for arrays; sequenced samples read against the IAM Sequencing Atlas, counted from sequencing reads, because an array never quite reads 0 or 1 and sequencing does.

### The IAM Sequencing Atlas: every CpG, in development

A second atlas, for sequenced samples only. IAM-Atlas v2.1 stays the reference for arrays; nothing in the new one touches an array reading.
For each of 56 healthy cell types, at every one of the 28.2 million CpGs of the genome numbering sequencing uses, it holds the healthy level,
the spread between donors, each donor's own measured disorder, the reads behind every value, and two noise maps (one from splitting the
donors in sealed halves, one from splitting every donor's reads). Each cell has reads at 97 to 98 % of all CpGs, about 32 times the sites
an array reads. It is counted, not fitted, and rebuilds from the public files with one command in minutes. It lets the sequencing
instruments read a sample's disorder at every CpG against healthy disorder of its own cell type, including clusters far from any array
site, and it is the reference the plasma readings are being built on. In development: each cell is commissioned when healthy sequenced
samples it has never held read Normal against it ([the record](doors/DEV_ATLAS_V21_01.md)).

**A ruler for each kind of instrument.** Each kind of instrument reads the same biology through its own chemistry. An array reads a probe's light, which never quite reaches 0 or 1: a site no cell ever marks still reads at the array's floor, a few percent up. Sequencing counts reads, so the same site reads about 0. Healthy disorder is computed from those levels, and near the floor a small offset in level is a large offset in disorder; the offset also changes with each laboratory's chemistry and library. So each kind of instrument reads against a ruler measured with its own chemistry: arrays against IAM-Atlas v2.1, sequenced samples against the IAM Sequencing Atlas. The physics and the instruments are the same; only the ruler changes with the platform. Read against IAM-Atlas v2.1, healthy sequenced samples showed it: at the islands every healthy cell keeps unmarked they read about 14 where healthy reads 1, and Met-A followed each sample's source.

### IAM-Deconvolver: the cell mix of a sample

Whole-blood differences mostly reflect how many of each cell type a sample holds. The Deconvolver reads the mix against the atlas and
gives each cell's share with its range.
**Commissioned** on blood-cell shares from arrays: on 200 mixtures from a laboratory the atlas never saw, the mean share error
was 0.016 ([commissioning note](doors/COMMISSIONING_NOTE_DECONV_V21.md)).
**Different because** composition is usually removed as a nuisance by statistics. Here it is measured, with its uncertainty, and the
chain declines a sample whose mix it cannot trust instead of reading a cell-mix shift as biology.

### Group-A: has a whole group moved?

One person's whole-blood reading cannot show a 1 % change, but a group can. Group-A reads each person with the same command as every
report, centers each reading on healthy people run the same way, and compares the two groups' averages.
**Commissioned** on groups of whole bloods run in one series: on 278 healthy people from a laboratory it had never read, women
and men came out 0.36 % apart, and a 1 % loss planted into half of them was found at 7.253 standard errors
([commissioning note](doors/COMMISSIONING_NOTE_GROUPA.md)).
**Different because** group studies usually fit a statistical model across thousands of sites between cohorts. Group-A compares one
physical reading per person, each against its own expected value for that person's cell mix, with nothing fitted, and prints the
smallest difference the study could have found.

### The territories and nearest-neighbor coupling: read on every run, in development

Two more readings come out of every run. **The territories** read disorder where the identity readings do not reach: the islands, sites
every healthy cell type keeps unmarked, and the domains, marked sites some cell types hold loosely, each on the same A-score gauge against
IAM-Atlas v2.1. **Nearest-neighbor coupling** reads, on single molecules, whether lost marks fall one at a time or together along a stretch
of DNA, a question from the information-theory study of methylation; 1 is independent. Both are shown beside the commissioned readings and
flagged as in development ([the instruments](INSTRUMENTS.md)). It is drawn on the A-score gauge as coupling-A, the sample's coupling over the healthy coupling of its own cell type, so a healthy cell reads 1; its healthy reference comes from the 153 healthy sequencing samples behind the atlas.

### The chain: one command

The instruments run as one chain: intake, calibration of each array on its own, the readings, and a report with a plain view and a
researcher view. One command reads a sample. A fresh copy of the public repository on a clean machine passes every release check
before each release ([operating procedure](sop/MethylPhys_CPG_SOP_v4.md)).

## 5. The tests, and what each one showed

### Healthy reads as healthy

An instrument that calls healthy people sick is useless to a cancer laboratory, so this is where every commissioning starts, and it is
tested where it is hardest: on laboratories that did not set the range.

- **The C-Score's range held in new laboratories** (above), and a third laboratory's healthy blood read inside it, 172 of 173.
- **The same DNA gives the same answer in three laboratories.** Identical reference DNA on 30 arrays: Met-A's spread between repeats
  was 0.0114, and after the same-run tare the laboratories differed by 0.006
  ([DEV_EPIQC_ARRAY_01_OUTCOME.md](doors/DEV_EPIQC_ARRAY_01_OUTCOME.md)). The usual approach corrects batch effects after the fact; the
  chain's design cancels them.
- **No false alarms from foreign DNA.** Foreign DNA mixed into healthy blood: 9 of 9 mixtures still read healthy
  ([DEV_CSCORE_MOSS_01_OUTCOME.md](doors/DEV_CSCORE_MOSS_01_OUTCOME.md)).
- **Healthy colon tissue reads healthy on the atlas.** Three healthy colon biopsies read 0.911, 0.941 and 0.943 on the atlas's tissue
  scorer, above its commissioned bar of 0.90 ([the showcase](doors/data/SHOWCASE_V4/README.md)).

### Departures are seen, at measured limits

Each instrument's detection limit was measured on planted samples, where the amount of change is known exactly: Met-A sees a 2 % change;
the C-Score sees a change on 5 % of sites when clustered; Group-A sees a 1 % loss across a group of about 130 people at more than seven times the noise. A researcher knows how small a change an instrument can see before reading a
single patient.

### Real tumors

These results are sealed and scored once. They are early results on real cancer, the work that comes next for the instruments.

- **Breast, colon and liver cancer.** Each patient's tumor was read against the same patient's own normal tissue. A line set from pairs of
  normal tissue puts about 5 % of tumors past it by chance. Past it: 23 of 53 breast tumors, 28 of 88 colon tumors and 22 of 50 liver
  tumors. Most tumors read above their own tissue: 41 of 53, 80 of 88 and 44 of 50
  ([the A-score database](ASCORES/README.md); [breast](doors/DEV_TISSUE_BREAST_01_OUTCOME.md),
  [colon](doors/DEV_TISSUE_COLON_01_OUTCOME.md), [liver](doors/DEV_TISSUE_LIVER_01_OUTCOME.md)).
- **The cancer fingerprint.** In prostate cancer, both instruments see the same thing in the same cells: whole stretches of the
  methylation pattern wiped out, far beyond the wear healthy cells show. Found again in a second laboratory's prostate cancer, in every
  sample ([DEV_FINGERPRINT_01_OUTCOME.md](doors/DEV_FINGERPRINT_01_OUTCOME.md)).

Where the lines of concern belong on each gauge, for each cancer, is for cancer researchers to find. These are the instruments to find
them with.

### Everything rebuilds from public data, with one command

Every headline result above rebuilds from public data on a fresh machine with one command, and the rebuilt numbers come out identical
to the committed ones: Met-A's reference, the C-Score's band, the Deconvolver's mixtures, IAM-A's self-reference line, the three
laboratories, and the breast, colon and liver results ([reproduce](reproduce/README.md); in the book,
[the commissioning record, reproduced](https://hmahaffeyges.github.io/IAM-Validation/book/ch_reproduced.html)).

## 6. What makes IAM different, in one table

| The usual approach | IAM |
|---|---|
| A scale learned from a training cohort | A scale from thermodynamics: the floor and the ceiling are derived; IAM-A's healthy mark is computed from the sample itself |
| A healthy baseline from a control group's average | IAM-A: the reference is inside the sample itself |
| Batch effects corrected afterward | Each array calibrated on its own; the same-run tare cancels the batch |
| Cell mix removed by statistics | Cell mix measured, with its uncertainty; samples it cannot trust are declined |
| Differences at single sites, about 20 to 30 % | The pattern across thousands of sites; scattered aging told apart from clustered change |
| Reference atlases with one number per site | An atlas fitted like a sky map, with how sure it is at every value |
| Results from analysis choices made after seeing data | Every rule sealed before its data, scored once, rebuilt by one command |

## 7. Why spend your time on it

The instruments answer the problems methylation researchers name first: a person's own baseline, batch effects, the cell mix, and
sensitivity. They do it from physics, so they need no training set, and anyone can test them. Run the chain on samples you already know
and see whether it reads them the way it should ([checking it on samples you already know](QUESTIONS_AND_ANSWERS.md)); report what you
find through the repository's issue form, so every result becomes part of the public record.

## Where to go next

- [The Q&A for researchers](QUESTIONS_AND_ANSWERS.md): the hard questions first, each answer linked to its record.
- [The researcher package](RESEARCHER_PACKAGE.md): what is in the package, and the three ways in (look, run, reproduce).
- [The instrument catalogue](INSTRUMENTS.md) and [the commissioning report](COMMISSIONING_REPORT.md): every instrument, its scope and its
  record.
- [The A-score database](ASCORES/README.md): every sample each instrument has read, healthy and tumor, and how many cross the healthy line.
- [The book](https://hmahaffeyges.github.io/IAM-Validation/book/): the physics in full, from cosmology to the cell.


**Met-A on sequenced samples.** The atlas is built on the array scale, where a site a cell keeps unmarked reads about 0.055 while sequencing counts it as 0. For a sequenced sample the chain puts each atlas column in sequencing units by the conversion the atlas's own build measured, so a healthy sequenced cell reads 1 against the same atlas. This reading is in development, flagged in every report.
