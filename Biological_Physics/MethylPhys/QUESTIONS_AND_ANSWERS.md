# Questions and answers

The questions a methylation researcher, a physicist or a reviewer will ask about IAM, answered openly. Where there is an answer, it points to
the test or derivation behind it, and most of those tests can be rerun from public data with one command
([Biological_Physics/MethylPhys/reproduce/README.md](reproduce/README.md)). Where there is no answer yet, it says so and names the open question.

All of this work was done with public data alone, without funding or institutional support. That limits which questions could be asked: no
new samples, no clinical cohorts, no repeated draws from the same people over years. The open questions below are open for that reason, and
each is one a funded laboratory could answer. What is here was built to show they are worth asking.

**Start with [An Introduction to IAM and the Physics of Methylation](INTRODUCTION.md):** what each instrument and the atlas does, what every test showed, and how it differs from the usual methods. This Q&A answers the questions that follow.

## Where everything is

- **The book**, read online: https://hmahaffeyges.github.io/IAM-Validation/book/ (each answer below names its chapter).
- **Every headline result, rebuilt from public data by one command:** [Biological_Physics/MethylPhys/reproduce/README.md](reproduce/README.md), and in the book,
  [The commissioning record, reproduced](https://hmahaffeyges.github.io/IAM-Validation/book/ch_reproduced.html), with the pages of real reports.
- **IAM-Atlas v2.1 on Zenodo:** the atlas, https://doi.org/10.5281/zenodo.23441999 , and the fit's full outputs,
  https://doi.org/10.5281/zenodo.23422688 .
- **The chain and its procedures:** [START_HERE.md](START_HERE.md), the
  [operating procedure](sop/MethylPhys_CPG_SOP_v4.md) and the
  [operations manual](manual/MethylPhys_CPG_Operations_Manual.pdf).
- **Every test, its outcome and its record:** the Master Testing Manifest, [MANIFEST.md](../../MANIFEST.md); every rented-computer run with
  the code it ran, [RUNS.md](RUNS.md).
- **A result from your own samples:** the **Report a result** form,
  https://github.com/hmahaffeyges/IAM-Validation/issues/new/choose

## The hard questions first

**Is "healthy" defined on the same data the instruments are then tested on?**
No. Each reference was set on one set of healthy samples and commissioned on samples it had not seen: the C-Score's band held in three
laboratories, read after it was fixed (34 of 35, 22 of 23, 172 of 173); purified neutrophils from other laboratories read Met-A Normal
68 of 68; the Deconvolver was scored on 200 mixtures from a laboratory the atlas never saw; IAM-A on CD8 T-cell samples from a laboratory
it had never seen. The record also tracks every public dataset already read ([CANON/data_spent.csv](../../CANON/data_spent.csv), 203 datasets),
and the push gate refuses a new prediction on one of them unless it states what was read before.

**How can anyone check that a rule was sealed before its data was read?**
Each rule is a note committed to the public repository, and git keeps the order of every change, so the commit of a rule can be checked
against the dated record of the run that read its data and the commit of its scoring. The push gate refuses a new seal that has not first passed a synthetic test, and a prediction on an already-read
dataset without its disclosure. **Open question:** commit times come from the committer's machine, so the order of commits and the dated
machine records are the evidence. An outside timestamp for each seal, such as a Zenodo deposit, would make the order checkable without
trusting the repository.

**How many tests were tried before the ones reported?**
All of them are listed. The Master Testing Manifest ([MANIFEST.md](../../MANIFEST.md)) holds 908 rows, each with its status, and 969 dated runs
with their outcomes, the 148 that failed and the 19 found invalid among them. The instruments that are commissioned are the ones whose
sealed tests passed; a test that failed stays in the record beside the claim it tested.

**Why should Landauer's bound matter to a cell that pays about one ATP to hold a mark, far more than k_B T ln 2?**
It sets the scale, not the price. k_B T ln 2 at body temperature (1.79 kJ/mol) is the least any record can cost per bit. One ATP is 54 kJ/mol,
M = 20.94 k_B T (30.2 in Landauer units). The gauge's floor is where thermal kicks would win even against one full ATP per site: a held
error of 8.1 × 10⁻¹⁰, or 2.5 × 10⁻⁸ bits. Healthy cells, measured on single molecules, hold their marks at 3.08 ± 0.12 k_B T per site (56 healthy cell types, counted by the chain), so a
healthy reading sits far above that floor. Book: [the Landauer chapter](https://hmahaffeyges.github.io/IAM-Validation/book/ch_landauer.html), [the gauge](https://hmahaffeyges.github.io/IAM-Validation/book/ch_gauge.html).

**The work says the gauge is not fitted, but the atlas is a fitted model and IAM-A uses a line.**
Both are measured references, and the distinction matters. IAM-Atlas v2.1 is a statistical fit to public healthy cells, and IAM-A's line is
drawn once through 56 kinds of healthy cells. The gauge's ends are derived from physics. Nothing is fitted to the samples being read or to
any cancer: every reference is a frozen file with a checksum, fixed before the samples it reads were opened.

**One author, outside an institution, not yet peer-reviewed. Why spend time on it?**
Because checking it costs little. Every headline result has a one-command rebuild from public data, each run from a fresh clone on a new
machine and compared number for number with the committed result ([Biological_Physics/MethylPhys/reproduce/README.md](reproduce/README.md)), and the chain's release
check runs on any machine in minutes. The code, the records and the reports are public. A result from your own known samples, agreeing or
not, is the most useful thing that can come back.

## What it is

**What does the chain measure?**
How well a cell holds the DNA methylation pattern that makes it the cell it is, in bits, on a gauge whose ends come from physics: a floor
(the least a cell can spend holding its marks against heat) and a saturation point (every identity site fully scrambled). Healthy reads 1.
Four commissioned instruments read one sample: Met-A (drift of a cell's identity sites), the C-Score (whether that drift is scattered, as
aging leaves it, or clustered along the genome), IAM-A (a sequencing sample's own error rate, read against itself), and the IAM-Deconvolver
(which cells are in a mixed sample), all against IAM-Atlas v2.1, a reference of 79 healthy cell types.
Evidence: [Biological_Physics/MethylPhys/INSTRUMENTS.md](INSTRUMENTS.md), [Biological_Physics/MethylPhys/COMMISSIONING_REPORT.md](COMMISSIONING_REPORT.md). Book: [reading a report, step by step](https://hmahaffeyges.github.io/IAM-Validation/book/ch_walkthrough.html).

**Is it a diagnostic test?**
No. These are research instruments. They print readings and states, not diagnoses. Where on the gauge disease begins is for cancer
researchers to find; the instruments are built to let them find it.

**What does "commissioned" mean here?**
An instrument is commissioned for a named use when it met every bar of a test whose rule was written and sealed before its data were read,
scored once. A reading outside that use is still printed, with the line *This score is preliminary until [instrument] is officially
commissioned on [specimen]*. Evidence: the commissioning notes in [Biological_Physics/MethylPhys/doors/](doors).

## The physics

**And the middle mark, where healthy sits?**
For Met-A it is measured: the median of healthy reference arrays run beside the sample. For IAM-A it comes from the sample itself. A cell's
design holds part of its genome firmly and leaves a share L loosely held; the sample's own L sets its expected held error,
ε_pred(L) = firm rate + slope × L, and that is the healthy 1 on its gauge. The form of that line comes from the physics of a design that
holds firm only what it uses (the reasoning is in the [coffee and whiteboard session](whiteboard/2026-10-14_coffee_and_whiteboard.md)); its two
constants were counted once on 56 healthy cell types and are frozen ([selfref_line_v1.json](chain/Runtime%20Matrices/IAM_A_SelfRef/selfref_line_v1.json)).
An attempt to derive the firm rate from the copying enzyme's own error did not hold on the commissioned counter, so it stays measured. This is
what lets each person be read against their own sample.

**Where do the gauge's ends come from? Are they fitted?**
They are derived, not fitted. The floor comes from Landauer's bound, k_B T ln 2 per bit, at body temperature, with the cell paying about one
ATP per site to hold a mark (the Mahaffey number, M = 20.94). The saturation end is every identity site fully scrambled: 3.03 for Met-A, and
0.5 / ε_pred(L) for IAM-A, the sample's own. Evidence: the book, [the gauge chapter](https://hmahaffeyges.github.io/IAM-Validation/book/ch_gauge.html), and its one-command derivation check (`python3 docs/book/verify_book.py`).

**Is there a test where the physics predicted a number before the data were read?**
Yes, in living mice. Mice that cannot make enough of the methyl donor should hold their marks worse by an amount the law fixes. The window
was written down first (1.016 to 1.135); the knockout mice read 1.124 against wild type (p 0.0003).
Evidence: [DEV_SAM_LEVER_01_OUTCOME.md](doors/DEV_SAM_LEVER_01_OUTCOME.md); rerun with
`python3 Biological_Physics/MethylPhys/reproduce/reproduce.py mice`; book: [the reproduced record](https://hmahaffeyges.github.io/IAM-Validation/book/ch_reproduced.html).

**Why should the same law apply to cosmology, qubits, chips and cells?**
Each domain stands on its own tests, and none of the cell instruments depends on the others. In cosmology the coupling is derived with no
free parameter; the fit to the Planck data matches the standard model (Δχ² = +0.54), and two predictions stated in advance are not yet tested:
the growth of structure 4.25 % below the standard model today, and two Hubble rates (67.16 and 72.26). In qubits and chips the energy floors
come from k_B T ln 2 and published specifications; the per-platform values are calibrated, so they are not predictions.
Evidence: [FOUR_DOMAINS.md](../../FOUR_DOMAINS.md), [REFERENCE.md](../../REFERENCE.md). **Open question:** the cosmology predictions, against the DESI
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
hold. Evidence: [COMMISSIONING_NOTE_IAMA.md](doors/COMMISSIONING_NOTE_IAMA.md);
rerun the line with `python3 Biological_Physics/MethylPhys/reproduce/reproduce.py iama_line`; book: [IAM-A](https://hmahaffeyges.github.io/IAM-Validation/book/ch_iama.html).

**Whole-blood differences are mostly differences in cell proportions. How does IAM handle that?**
It measures the mix and reads each cell on its own terms. The IAM-Deconvolver reports each cell's share with its 95 % interval; on 200
mixtures from a laboratory the atlas never saw, the shares came back within 0.016 of the truth (mean absolute error). Met-A is commissioned on isolated
neutrophils for exactly this reason, and the C-Score on whole blood, where its band was set. Evidence:
[COMMISSIONING_NOTE_DECONV_V21.md](doors/COMMISSIONING_NOTE_DECONV_V21.md); `python3 Biological_Physics/MethylPhys/reproduce/reproduce.py deconv`.

**Ancestry, sex and age?**

IAM-A reads each sample against the line its own loose share predicts, so no population reference enters its reading; it has not yet been
tested across ancestries, and that test needs only healthy samples. Sex: Met-A's identity sites were required to agree on male and female
reference arrays, and its 32 identity sites on the X chromosome read the same on both. The reference donors were aged 20 to 39
([COMMISSIONING_NOTE_METAA_NEUTROPHILS.md](doors/COMMISSIONING_NOTE_METAA_NEUTROPHILS.md)).

**X inactivation?**

The sky map found it without being asked: one of the six reference arrays read as female, with 40 % of her X chromosome sites between
β = 0.3 and 0.7 against 8 % in the other five. The CMB's lesson applies: a whole-array map needs a sex mask, or a reference of the same sex,
before its structure means anything ([the sky tools chapter](https://hmahaffeyges.github.io/IAM-Validation/book/ch_skytools.html)).

**What if a sample holds a cell the reference does not?**
The tool checks before it reports. Each pure array must read as its own cell before it is used; in its commissioning test it refused six B-cell
arrays that did not. When a mixture holds cells outside the reference, the raw shares no longer sum to one, and the report shows it.

**Published references are built from fewer than ten sorted samples per cell type. How sure is IAM-Atlas v2.1?**
It says so, site by site. Every value carries its 95 % range and the person-to-person spread, and every cell's page states how many healthy
samples it was built from. Certainty follows that number: 17 cells rest on more than ten samples (median half-width ±0.012), 36 on one to three
(±0.067). On the jobs a reference atlas is used for, v2.1 met every bar against both published references. Evidence:
[COMMISSIONING_NOTE_ATLAS_V21.md](doors/COMMISSIONING_NOTE_ATLAS_V21.md), the cell pages on the website, the atlas on Zenodo
(https://doi.org/10.5281/zenodo.23441999); book: [the atlas](https://hmahaffeyges.github.io/IAM-Validation/book/ch_atlas.html).
**Open question:** more healthy samples of the cells with one to three would narrow their ranges.

## Measurement quality

**Batch effects, slide position, laboratory?**
The same-run tare divides by healthy references from the same run, so the batch cancels instead of being modeled afterward. Identical DNA run
at three laboratories read alike: Met-A's technical repeats agreed within 0.0114 (bar 0.012, 10 of 10), and after the same-run tare the three
laboratories' readings of the same DNA spread by at most 0.006 (bar 0.02, 3 of 3). Evidence:
[DEV_EPIQC_ARRAY_01_OUTCOME.md](doors/DEV_EPIQC_ARRAY_01_OUTCOME.md); `python3 Biological_Physics/MethylPhys/reproduce/reproduce.py three_labs`.

**Does the healthy range hold in laboratories that did not set it?**
Yes, for the C-Score: 34 of 35, 22 of 23 and 172 of 173 healthy people in three laboratories read inside a range fixed before any of them
was read, and the same person measured twice read the same (spread 0.038, bar 0.125). Evidence:
[COMMISSIONING_NOTE_CSCORE.md](doors/COMMISSIONING_NOTE_CSCORE.md); `python3 Biological_Physics/MethylPhys/reproduce/reproduce.py cscore_band`; book: [the C-Score](https://hmahaffeyges.github.io/IAM-Validation/book/ch_cscore.html).

**Arrays detect changes of about 20 to 30 %. How can it see less?**
The C-Score reads the pattern across many sites at once, not one site at a time. Its commissioned detection limit is C_rel 1.5 when 5 % of
sites change together in clusters, and 0 of 120 planted scattered changes read above its band.

**What does the chain do with the raw intensity files?**

Stage 1 calibrates each array on its own: background and dye-bias correction and probe-type handling (noob), using only that array's
control probes, so no array changes another array's numbers. Every probe is tested against the same array's negative controls, and a probe
at background is removed before any stage reads its value. The array's own SNP probes measure its noise, which the sky maps use. Stage 0
refuses an array the chain cannot read (wrong platform, too few sites, a cell mix it cannot trust) and says why, instead of reading it.
Code: [stage_1_idat_calibration.py](chain/stage_1_idat_calibration.py); procedure: [the SOP](sop/MethylPhys_CPG_SOP_v4.md).

**Do I need technical replicates?**

Repeatability was measured, so each sample does not need its own replicate. Identical reference DNA on 30 arrays in three laboratories:
Met-A's spread between repeats was 0.0114, and after the same-run tare the laboratories differed by 0.006
([DEV_EPIQC_ARRAY_01_OUTCOME.md](doors/DEV_EPIQC_ARRAY_01_OUTCOME.md)). The C-Score on the same person measured twice: a spread of 0.0379
over 13 pairs ([COMMISSIONING_NOTE_CSCORE.md](doors/COMMISSIONING_NOTE_CSCORE.md)). The healthy references run beside a sample act as its
controls.

**FFPE tissue, or a small amount of DNA?**

Not yet read through the chain. Stage 0 measures each array's detected fraction and the noise of its own SNP probes, so a degraded array is
seen as degraded instead of being read as biology. Whether FFPE arrays pass intake is an open test, and a good one to run on samples you know.

**Which platforms?**
Met-A and the C-Score read Illumina EPIC v1 arrays; 450K and EPIC v2 are refused with one sentence, because no frozen reference exists for them.
IAM-A reads whole-genome bisulfite sequencing (a BAM aligned with Bismark, or a .pat file). **Open question:** 450K, EPIC v2, FFPE tissue and
other sequencing chemistries each need their own reference and commissioning test.

**Is everything reproducible?**
Every reference the commissioned instruments read is rebuilt from public data, identical, by committed code: `python3 Biological_Physics/MethylPhys/reproduce/reproduce.py
metaa_rebuild`. Each test in [Biological_Physics/MethylPhys/reproduce/](reproduce/README.md) starts from public accessions, runs the chain, and compares with the committed
result; the book lists each one with its fresh-clone result ([the reproduced record](https://hmahaffeyges.github.io/IAM-Validation/book/ch_reproduced.html)). Every rule was sealed before its data, every scoring was done once, and every run is listed with its code in the run ledger.

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
Evidence: the outcome records in [Biological_Physics/MethylPhys/doors/](doors); `python3 Biological_Physics/MethylPhys/reproduce/reproduce.py colon`, `breast` or `liver`.
**Open question:** where the lines on the gauge belong for each cancer, set by cancer researchers on cohorts with known outcomes.

**What about early detection, precancer, cell-free DNA?**
**Open questions.** For cell-free DNA, IAM-A's count has passed a planted short-fragment test, and the first real set (plasma from people with
lung cancer) is chosen but not yet read. For precancer, 140 public rectal biopsies with and without adenomas are set aside for a field test,
not yet run. Early detection needs samples with known follow-up, which public data rarely give.

**How large are the commissioning sets?**
Stated with each: 541 healthy neutrophil arrays from 18 series for Met-A; three laboratories for the C-Score; 200 mixtures for the
Deconvolver; for IAM-A, 8 healthy CD8 T-cell samples read (7 Normal) after the sample-shape check declined 5 of 13. **Open question:** larger
sets, other cell types and other laboratories for IAM-A.

## Questions the field is asking now

These come from recent reviews of methylation-based cancer detection and of epigenetic drift, and from recent atlases of purified human
cells and of cell-free DNA (2024 to 2026).

**Methylation entropy and stochastic drift are already measured. What is different here?**
Published entropy measures are statistical: they summarize how disordered a sample's pattern is and compare it with a cohort. IAM puts the
measurement on a physical scale. The gauge's ends are derived (the floor from Landauer's bound and the cost of holding a mark, the
saturation end from every identity site fully scrambled), and healthy is read from the sample's own reference or same-run references, not
from a cohort average. The two can be compared site for site; that comparison has not been done.

**How is this different from an epigenetic clock?**

A clock estimates age from sites chosen because they track age in a cohort. IAM does not estimate age. The C-Score separates the two kinds
of change: scattered loss, the pattern of ordinary aging and wear, and clustered loss, the pattern tumors leave. Planted scattered losses in
real healthy arrays never read as a departure: 0 of 120 above the band
([COMMISSIONING_NOTE_CSCORE.md](doors/COMMISSIONING_NOTE_CSCORE.md)).

**Tumors are more variable than matched normal tissue. Is IAM just measuring that?**
In part, and it is meant to. Each tumor is read against the same patient's own healthy tissue, so the reading is how far that tumor's
pattern has moved from its own healthy state, at sites chosen so that cell mix cannot produce the signal. What IAM adds is the C-Score's
distinction between change scattered across the genome, as aging leaves it, and change clustered along it.

**Much age-related change tracks cell division. Is the signal just proliferation?**
**Open question.** IAM has not separated how often a cell divides from how well it holds its marks at each division. The mouse prediction
changed the methyl supply, not the division rate. A test that varies division at a fixed supply is the next experiment, and needs
controlled samples.

**Models trained on cancer and control samples can learn age or menopause instead of cancer. Is IAM exposed to that?**
The instruments are not trained on cancer at all: every reference is healthy, and every rule was fixed before any tumor was read. Age is
still a factor in what a healthy reading looks like. **Open question:** how Met-A and the C-Score move with age in healthy people,
measured across ages in one study.

**A fixed reference atlas cannot hold the variability between people.**
IAM-Atlas v2.1 stores, at every site, the spread a new person's cell shows around the average, and the Deconvolver uses that spread in its
95 % intervals. IAM-A does not use an atlas for its reading: it reads the sample against itself.

**Cell-free DNA from early tumors is a tiny fraction of the total. Can IAM see it?**
**Open question**, not yet tested (see above). The tissue results show the signal where tumor DNA is most of the sample; how far it survives
dilution in plasma has to be measured.

**Prediction models are never "validated" once and for all.**
Agreed. IAM does not claim a validated predictor. Each instrument is commissioned for a named use, on a sealed test, and every reading
outside that use is labeled preliminary. Each new use needs its own test.

**A recent atlas of the same purified cell types finds about 11 % of CpGs half-methylated inside a pure cell, from imprinting and
allele-specific methylation. Does IAM-A count those as errors?**
No. IAM-A's held error is counted only inside molecules with at least 6 CpG calls that are at least 80 % methylated, and only as an
isolated unmethylated call between methylated neighbors. A molecule from the unmethylated allele is far below 80 % and is never counted;
molecules from the methylated allele are read like any other. The healthy self-reference line was also measured on purified healthy cells,
which carry the same regions.

**In plasma, how cell-free DNA breaks apart depends on its methylation and its cell of origin. Does that affect IAM-A on cell-free DNA?**
It could, because IAM-A counts along single molecules and cell-free fragments are short (about 167 letters). A synthetic check of the
count at that fragment length came back within 0.3 % of the planted value (bar 3 %). **Open question:** whether the self-reference line
holds on real plasma, where fragments from different cells break differently. The first real plasma set is the next test.

**Library preparation can change methylation within a read (one study saw it in urine cell-free DNA with a double-stranded method). Is
IAM-A exposed to that?**
Yes, in principle: held error is a within-read quantity, so chemistry matters. IAM-A is commissioned on whole-genome bisulfite
sequencing. The meter's own error has been measured in a second laboratory's data (sequencing error 0.002 to 0.006, conversion failure
1.8 to 5.1 %). **Open question:** enzymatic conversion, single-stranded libraries and urine, each of which needs its own check.

**Is one person steady enough over time to be their own reference?**
A recent study that followed healthy people's plasma over time found each person's cell-of-origin profile stable, which its authors
attribute to set points in blood-cell production. IAM-A already reads each sample against its own reference, with no other person
involved. **Open question:** IAM-A on the same person's cells across years, which is a planned track.

**There is no standard for reporting quality control in these studies.**
Every report carries its quality record: Stage 0's checks on the raw file (detection, bead counts, call rate, channel checksums), the noise
gate, the chain's commit, the checksum of every frozen file it read, and one row in an evidence ledger.

**Sources for this section** (recent reviews and atlases the questions come from):
- On epigenetic stochasticity, entropy and cancer risk (2024): https://pubmed.ncbi.nlm.nih.gov/38432318/
- DNA methylation variability defines a fundamental dimension of tumor epigenomes (preprint, 2026): https://www.biorxiv.org/content/10.64898/2026.03.12.711303v1
- Mapping the temporal landscape of breast cancer using epigenetic entropy (2025): https://www.nature.com/articles/s42003-025-08867-2
- Current challenges of methylation-based liquid biopsies in cancer diagnostics (2024): https://doi.org/10.3390/cancers16112001
- Computational methods and challenges in cell-free DNA analysis for multi-cancer early detection (preprint, 2026): https://arxiv.org/abs/2606.20174
- Atlas of imprinted and allele-specific DNA methylation in the human body (2025): https://doi.org/10.1038/s41467-025-57433-1
- Longitudinal cell-free DNA methylome and fragmentome profiles in health (preprint, 2025): https://www.biorxiv.org/content/10.1101/2025.10.25.684551v1
- Library preparation and tissue-of-origin deconvolution methods for urine and plasma cell-free DNA methylomes: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10417284/

## Checking it on samples you already know

Run the chain on samples whose answer you know, and see whether it does what is written here. If it does not, that is a result, and it
belongs in the public record.

```bash
git clone https://github.com/hmahaffeyges/IAM-Validation.git && cd IAM-Validation/Biological_Physics/MethylPhys/chain
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python3 release_check_v4.py          # every check must pass on your machine first
```

The procedure is the [operating procedure](sop/MethylPhys_CPG_SOP_v4.md) and the
[operations manual](manual/MethylPhys_CPG_Operations_Manual.pdf).

| instrument | commissioned scope | on your known samples, it should | it does not hold if | its commissioning result |
|---|---|---|---|---|
| **Met-A** | isolated neutrophils, EPIC v1, tared by ≥ 3 same-run healthy references | read healthy neutrophils Normal (0.95-1.05); agree on one person's repeat arrays | fewer than 95 % of your healthy isolated neutrophils read Normal, or one person's repeats differ by SD > 0.020 | 525 of 541 healthy arrays Normal (97.0 %, 18 series); other laboratories' purified neutrophils 68 of 68; repeat SD 0.016 |
| **C-Score** | whole blood, EPIC v1, tared by ≥ 3 same-run healthy references | read healthy whole blood not above its band (C_rel ≤ 1.152); never read a scattered change above it; read a clustered change above it | more than 5 % of your healthy whole bloods read above 1.152, or a scattered change reads above it, or a clustered change at 5 % of sites stays below C_rel 1.5 | three laboratories: 34 of 35, 22 of 23, 172 of 173 not above; 0 of 120 planted scattered changes above |
| **IAM-Deconvolver** | shares of neutrophils, monocytes, CD4 T and CD8 T cells in blood mixtures, arrays | return mixtures of known composition at their true shares | mean absolute share error above 0.03 | 0.016 on 200 mixtures from a laboratory the atlas never saw |
| **IAM-A** | healthy blood CD8 T cells, whole-genome bisulfite, read against the sample's own loose share | read healthy samples Normal (0.779-1.055) | fewer than 75 % of at least 8 read healthy samples are Normal | 7 of 8 Normal, a laboratory it had never seen |
| **IAM-Atlas v2.1** | the uses in its [commissioning note](doors/COMMISSIONING_NOTE_ATLAS_V21.md) | place your healthy cells of those types inside its range | below the bar the note states for that use | e.g. 97.8 % of 45 cortical-neuron arrays, 100 % of 53 lung-bronchus arrays inside |

A refusal is not a failure: the chain declines what it cannot read and says why. A refusal of a sample it should have read, for a wrong
reason, is a result worth reporting. Nothing in the table is fitted to your data; every reference is a frozen file whose checksum the release
check verifies, so each line is a prediction made before your samples existed.

**To report a result,** open an issue with the **Report a result** form (Issues, New issue), attach the report, the bundle and the ledger row,
and say what you knew about the samples before you ran them. Every report is answered in public, and every result that holds goes into the
record beside the claim it tests.
