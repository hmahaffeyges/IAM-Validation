# Questions and answers

The questions a methylation researcher, a physicist or a reviewer will ask about IAM, answered openly. Where there is an answer, it points to
the test or derivation behind it, and most of those tests can be rerun from public data with one command
([reproduce/README.md](reproduce/README.md)). Where there is no answer yet, it says so and names the open question.

All of this work was done with public data alone, without funding or institutional support. That limits which questions could be asked: no
new samples, no clinical cohorts, no repeated draws from the same people over years. The open questions below are open for that reason, and
each is one a funded laboratory could answer. What is here was built to show they are worth asking.

## What it is

**What does the chain measure?**
How well a cell holds the DNA methylation pattern that makes it the cell it is, in bits, on a gauge whose ends come from physics: a floor
(the least a cell can spend holding its marks against heat) and a saturation point (every identity site fully scrambled). Healthy reads 1.
Four commissioned instruments read one sample: Met-A (drift of a cell's identity sites), the C-Score (whether that drift is scattered, as
aging leaves it, or clustered along the genome), IAM-A (a sequencing sample's own error rate, read against itself), and the IAM-Deconvolver
(which cells are in a mixed sample), all against IAM-Atlas v2.1, a reference of 79 healthy cell types.
Evidence: [INSTRUMENTS.md](INSTRUMENTS.md), [COMMISSIONING_REPORT.md](COMMISSIONING_REPORT.md).

**Is it a diagnostic test?**
No. These are research instruments. They print readings and states, not diagnoses. Where on the gauge disease begins is for cancer
researchers to find; the instruments are built to let them find it.

**What does "commissioned" mean here?**
An instrument is commissioned for a named use when it met every bar of a test whose rule was written and sealed before its data were read,
scored once. A reading outside that use is still printed, with the line *This score is preliminary until [instrument] is officially
commissioned on [specimen]*. Evidence: the commissioning notes in [Biological_Physics/MethylPhys/doors/](Biological_Physics/MethylPhys/doors/).

## The physics

**Where do the gauge's ends come from? Are they fitted?**
They are derived, not fitted. The floor comes from Landauer's bound, k_B T ln 2 per bit, at body temperature, with the cell paying about one
ATP per site to hold a mark (the Mahaffey number, M = 20.94). The saturation end is every identity site fully scrambled: 3.03 for Met-A, and
0.5 / ε_pred(L) for IAM-A, the sample's own. Evidence: the book, Part 6 (the gauge chapter), and its one-command derivation check.

**Is there a test where the physics predicted a number before the data were read?**
Yes, in living mice. Mice that cannot make enough of the methyl donor should hold their marks worse by an amount the law fixes. The window
was written down first (1.016 to 1.135); the knockout mice read 1.124 against wild type (p 0.0003).
Evidence: [DEV_SAM_LEVER_01_OUTCOME.md](Biological_Physics/MethylPhys/doors/DEV_SAM_LEVER_01_OUTCOME.md); rerun with
`python3 reproduce/reproduce.py mice`.

**Why should the same law apply to cosmology, qubits, chips and cells?**
Each domain stands on its own tests, and none of the cell instruments depends on the others. In cosmology the coupling is derived with no
free parameter; the fit to the Planck data matches the standard model (Δχ² = +0.54), and two predictions stated in advance are not yet tested:
the growth of structure 4.25 % below the standard model today, and two Hubble rates (67.16 and 72.26). In qubits and chips the energy floors
come from k_B T ln 2 and published specifications; the per-platform values are calibrated, so they are not predictions.
Evidence: [FOUR_DOMAINS.md](FOUR_DOMAINS.md), [REFERENCE.md](REFERENCE.md). **Open question:** the cosmology predictions, against the DESI
full-shape data.

**Does IAM show that methylation change causes cancer?**
No, and it does not claim to. It adds a physical scale where the field usually has an association: drift measured in bits, between ends set
by physics. Causation needs experiments IAM cannot run on public data.

## Samples, baselines and cell mix

**What is a sample compared with? Who is the healthy control?**
Two ways. IAM-A reads a sequencing sample against itself: across 56 kinds of healthy human cells, a cell's error rate follows one line set by
how much of its DNA it keeps loosely tagged (53 of 56 on the line), so the healthy expectation comes from the sample's own reads. Met-A and the
C-Score on arrays are tared by three or more healthy arrays from the same run, so the laboratory's habits cancel. **Open question:** a
person's own earlier sample would be the strongest baseline of all; reading a person over years needs repeated draws that public data do not
hold. Evidence: [COMMISSIONING_NOTE_IAMA.md](Biological_Physics/MethylPhys/doors/COMMISSIONING_NOTE_IAMA.md);
rerun the line with `python3 reproduce/reproduce.py iama_line`.

**Whole-blood differences are mostly differences in cell proportions. How does IAM handle that?**
It measures the mix and reads each cell on its own terms. The IAM-Deconvolver reports each cell's share with its 95 % interval; on 200
mixtures from a laboratory the atlas never saw, the shares came back within 0.016 of the truth (median). Met-A is commissioned on isolated
neutrophils for exactly this reason, and the C-Score on whole blood, where its band was set. Evidence:
[COMMISSIONING_NOTE_DECONV_V21.md](Biological_Physics/MethylPhys/doors/COMMISSIONING_NOTE_DECONV_V21.md); `python3 reproduce/reproduce.py deconv`.

**What if a sample holds a cell the reference does not?**
The tool checks before it reports. Each pure array must read as its own cell before it is used; in its commissioning test it refused six B-cell
arrays that did not. When a mixture holds cells outside the reference, the raw shares no longer sum to one, and the report shows it.

**Published references are built from fewer than ten sorted samples per cell type. How sure is IAM-Atlas v2.1?**
It says so, site by site. Every value carries its 95 % range and the person-to-person spread, and every cell's page states how many healthy
samples it was built from. Certainty follows that number: 17 cells rest on more than ten samples (median half-width ±0.012), 36 on one to three
(±0.067). On the jobs a reference atlas is used for, v2.1 met every bar against both published references. Evidence:
[COMMISSIONING_NOTE_ATLAS_V21.md](Biological_Physics/MethylPhys/doors/COMMISSIONING_NOTE_ATLAS_V21.md), the cell pages on the website.
**Open question:** more healthy samples of the cells with one to three would narrow their ranges.

## Measurement quality

**Batch effects, slide position, laboratory?**
The same-run tare divides by healthy references from the same run, so the batch cancels instead of being modeled afterward. Identical DNA run
at three laboratories read alike: Met-A's technical repeats agreed within 0.0114 (bar 0.012, 10 of 10), and after the same-run tare the three
laboratories' readings of the same DNA spread by at most 0.006 (bar 0.02, 3 of 3). Evidence:
[DEV_EPIQC_ARRAY_01_OUTCOME.md](Biological_Physics/MethylPhys/doors/DEV_EPIQC_ARRAY_01_OUTCOME.md); `python3 reproduce/reproduce.py three_labs`.

**Does the healthy range hold in laboratories that did not set it?**
Yes, for the C-Score: 34 of 35, 22 of 23 and 172 of 173 healthy people in three laboratories read inside a range fixed before any of them
was read, and the same person measured twice read the same (spread 0.038, bar 0.125). Evidence:
[COMMISSIONING_NOTE_CSCORE.md](Biological_Physics/MethylPhys/doors/COMMISSIONING_NOTE_CSCORE.md); `python3 reproduce/reproduce.py cscore_band`.

**Arrays detect changes of about 20 to 30 %. How can it see less?**
The C-Score reads the pattern across many sites at once, not one site at a time. Its commissioned detection limit is C_rel 1.5 when 5 % of
sites change together in clusters, and 0 of 120 planted scattered changes read above its band.

**Which platforms?**
Met-A and the C-Score read Illumina EPIC v1 arrays; 450K and EPIC v2 are refused with one sentence, because no frozen reference exists for them.
IAM-A reads whole-genome bisulfite sequencing (a BAM aligned with Bismark, or a .pat file). **Open question:** 450K, EPIC v2, FFPE tissue and
other sequencing chemistries each need their own reference and commissioning test.

**Is everything reproducible?**
Every reference the commissioned instruments read is rebuilt from public data, identical, by committed code: `python3 reproduce/reproduce.py
metaa_rebuild`. Each test in [reproduce/](reproduce/README.md) starts from public accessions, runs the chain, and compares with the committed
result. Every rule was sealed before its data, every scoring was done once, and every run is listed with its code in the run ledger.

## Results so far

**Has it been tested on cancer?**
On tissue, with the rule sealed before any tumor was opened, each tumor read against the same patient's own healthy tissue:

| cancer | tumors outside the healthy range | expected by chance |
|---|---|---|
| colorectal (88 patients) | 28 | about 4 |
| breast (53) | 23 | about 3 |
| liver (50) | 22 | about 2 |

In prostate cancer cells, both instruments found whole stretches of the pattern wiped out, far beyond healthy wear, and a second laboratory's
prostate cancer showed the same in 6 of 6 samples. These are development results: the instruments were not commissioned on tissue.
Evidence: the outcome records in [Biological_Physics/MethylPhys/doors/](Biological_Physics/MethylPhys/doors/); `python3 reproduce/reproduce.py colon`.
**Open question:** where the lines on the gauge belong for each cancer, set by cancer researchers on cohorts with known outcomes.

**What about early detection, precancer, cell-free DNA?**
**Open questions.** For cell-free DNA, IAM-A's count has passed a planted short-fragment test, and the first real set (plasma from people with
lung cancer) is chosen but not yet read. For precancer, 140 public rectal biopsies with and without adenomas are set aside for a field test,
not yet run. Early detection needs samples with known follow-up, which public data rarely give.

**How large are the commissioning sets?**
Stated with each: 541 healthy neutrophil arrays from 18 series for Met-A; three laboratories for the C-Score; 200 mixtures for the
Deconvolver; for IAM-A, 8 healthy CD8 T-cell samples read (7 Normal) after the sample-shape check declined 5 of 13. **Open question:** larger
sets, other cell types and other laboratories for IAM-A.

## Checking it on samples you already know

Run the chain on samples whose answer you know, and see whether it does what is written here. If it does not, that is a result, and it
belongs in the public record.

```bash
git clone https://github.com/hmahaffeyges/IAM-Validation.git && cd IAM-Validation/Biological_Physics/MethylPhys/chain
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python3 release_check_v4.py          # every check must pass on your machine first
```

The procedure is the [operating procedure](Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v4.md) and the
[operations manual](Biological_Physics/MethylPhys/manual/MethylPhys_CPG_Operations_Manual.pdf).

| instrument | commissioned scope | on your known samples, it should | it does not hold if | its commissioning result |
|---|---|---|---|---|
| **Met-A** | isolated neutrophils, EPIC v1, tared by ≥ 3 same-run healthy references | read healthy neutrophils Normal (0.95-1.05); agree on one person's repeat arrays | fewer than 95 % of your healthy isolated neutrophils read Normal, or one person's repeats differ by SD > 0.020 | 525 of 541 healthy arrays Normal (97.0 %, 18 series); other laboratories' purified neutrophils 68 of 68; repeat SD 0.016 |
| **C-Score** | whole blood, EPIC v1, tared by ≥ 3 same-run healthy references | read healthy whole blood not above its band (C_rel ≤ 1.152); never read a scattered change above it; read a clustered change above it | more than 5 % of your healthy whole bloods read above 1.152, or a scattered change reads above it, or a clustered change at 5 % of sites stays below C_rel 1.5 | three laboratories: 34 of 35, 22 of 23, 172 of 173 not above; 0 of 120 planted scattered changes above |
| **IAM-Deconvolver** | shares of neutrophils, monocytes, CD4 T and CD8 T cells in blood mixtures, arrays | return mixtures of known composition at their true shares | median absolute share error above 0.03 | 0.016 on 200 mixtures from a laboratory the atlas never saw |
| **IAM-A** | healthy blood CD8 T cells, whole-genome bisulfite, read against the sample's own loose share | read healthy samples Normal (0.779-1.055) | fewer than 75 % of at least 8 read healthy samples are Normal | 7 of 8 Normal, a laboratory it had never seen |
| **IAM-Atlas v2.1** | the uses in its [commissioning note](Biological_Physics/MethylPhys/doors/COMMISSIONING_NOTE_ATLAS_V21.md) | place your healthy cells of those types inside its range | below the bar the note states for that use | e.g. 97.8 % of 45 cortical-neuron arrays, 100 % of 53 lung-bronchus arrays inside |

A refusal is not a failure: the chain declines what it cannot read and says why. A refusal of a sample it should have read, for a wrong
reason, is a result worth reporting. Nothing in the table is fitted to your data; every reference is a frozen file whose checksum the release
check verifies, so each line is a prediction made before your samples existed.

**To report a result,** open an issue with the **Report a result** form (Issues, New issue), attach the report, the bundle and the ledger row,
and say what you knew about the samples before you ran them. Every report is answered in public, and every result that holds goes into the
record beside the claim it tests.
