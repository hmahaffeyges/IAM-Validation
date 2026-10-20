# Reproduce IAM's results from public data

Each test below starts from public accessions only (GEO, ArrayExpress, SRA, ENA, ENCODE, Zenodo), runs the committed code of the chain, and
compares what it makes with the result committed in this repository. Nothing is read from our own storage.

```bash
git clone https://github.com/hmahaffeyges/IAM-Validation.git && cd IAM-Validation
bash Biological_Physics/MethylPhys/reproduce/setup.sh                       # once: Python 3.11 with the chain's pinned packages (Biological_Physics/MethylPhys/reproduce/.venv)
python3 Biological_Physics/MethylPhys/reproduce/reproduce.py --list         # the tests
python3 Biological_Physics/MethylPhys/reproduce/reproduce.py cscore_band    # one test: prints each compared number and PASS or DIFF
python3 Biological_Physics/MethylPhys/reproduce/reproduce.py --all          # every test
```

Each run writes `Biological_Physics/MethylPhys/reproduce/out/<test>/verdict.json` (every comparison, the commit, the time) and a log per step. The last column is the
test's own run from a fresh public clone on a clean machine.

## What each test shows, in plain words

**Met-A's reference, rebuilt from public arrays: noise sites and gate, the floor, the 6,000 identity sites, the healthy spread and the blood-composition markers.** Met-A reads every sample against a reference: the sites that define a healthy neutrophil, the sites that measure an array's noise, and the floor. Every piece of that reference was rebuilt from 91 public arrays of purified blood cells with the repository's own code, and every number came out identical to the reference the chain reads. *Why it matters:* Nothing in Met-A's reference was set by hand: it follows from public data and the code, and anyone can rebuild it.

**Blood-cell shares in 200 mixtures from a laboratory the atlas never saw (E-MTAB-2145).** A laboratory the atlas never saw mixed purified blood cells in known amounts. The Deconvolver read 200 of these mixtures and recovered each cell's share within 0.016 on average, and it refused the arrays whose cells did not read as the cell they were labeled. *Why it matters:* The usual tools estimate the cell mix only to remove it as a nuisance. This one measures it, and was tested on mixtures whose true shares were known.

**The C-Score's healthy range holds in a laboratory that did not set it (GSE184269).** The C-Score's healthy range was set on other laboratories' healthy blood and sealed. Then the healthy blood of a laboratory that took no part in setting it was read: 22 of 23 people fell inside the range. *Why it matters:* A healthy range that holds only where it was set is no use to a new laboratory. This one held in a laboratory it had never seen.

**IAM-A's self-reference line, rebuilt from 158 public healthy-cell sequencing files.** IAM-A needs no control group because healthy cells of every type follow one line: how often a cell loses a held mark goes with how much of its DNA it keeps loosely marked. That line was recounted from 158 public sequencing files, and every count, and the line itself, came out identical. *Why it matters:* This line is what lets each person be read against their own sample instead of against an average of other people.

**The same DNA gives the same reading in three laboratories (SEQC2 EpiQC, GSE230132).** The same cell-line DNA was measured in three laboratories. Met-A's repeats agreed within 0.0114, and the same DNA in different laboratories within 0.0084, without any correction afterward. *Why it matters:* Batch effects between laboratories are the field's most common problem, usually corrected after the fact. Here the instrument shows almost none to begin with.

**Physics called it in advance, in living mice (GSE77079).** The law predicts that a cell short of methyl donor holds its marks less well, and by how much. Before the data were read, the window was written down: mice that cannot make enough methyl donor, over normal mice, between 1.016 and 1.135. They read 1.124 (p 0.0003). *Why it matters:* Most methylation findings are associations found after looking at the data. This was a number predicted by physics first and measured afterward, in living animals.

**Colorectal tumors leave their own healthy tissue's range (GSE131013, 88 patients).** Each patient's colon tumor was read against the same patient's own normal tissue. A line set only from pairs of normal tissue puts about 5 % of tumors past it by chance, about 4 of these 88. Past it: 28 of 88. *Why it matters:* Each patient is their own control, so differences between people, ages and laboratories cancel; what is left is the tumor's departure from that person's own healthy tissue.

**Breast cancer: tumors leave the patient's own healthy range far more often than chance.** Each patient's breast tumor was read against the same patient's own healthy tissue. A line set only from pairs of normal tissue puts about 3 of these 53 tumors past it by chance. Past it: 23 of 53. *Why it matters:* The same reading as colon, in a second cancer, scored once on a rule sealed before the data were read.

**Liver cancer: a third cancer, the same result.** Each patient's liver tumor was read against the same patient's own healthy tissue. A line set only from pairs of normal tissue puts about 2 of these 50 tumors past it by chance. Past it: 22 of 50. *Why it matters:* A third cancer, the same reading, the same result.

## The tests

| test | what it shows | public inputs | time, disk | record | from a fresh clone |
|---|---|---|---|---|---|
| `metaa_rebuild` | **Met-A's reference, rebuilt from public arrays: noise sites and gate, the floor, the 6,000 identity sites, the healthy spread and the blood-composition markers.** Every reference matrix Met-A reads is rebuilt from 91 public purified-blood arrays and comes out identical to the frozen file the chain reads. | GSE110554; GSE167998; GSE2998xxx neutrophil arrays (listed in reference_donors.csv) | about 3 h, 20 GB | [record](../doors/COMMISSIONING_NOTE_METAA_NEUTROPHILS.md) | **PASS** 2026-10-18, 8 of 8 numbers identical |
| `deconv` | **Blood-cell shares in 200 mixtures from a laboratory the atlas never saw (E-MTAB-2145).** The Deconvolver reads 200 mixtures of purified cells from an unseen laboratory within 0.016 of the true shares, and its sample check refuses the arrays that do not read as their own cell. | E-MTAB-2145 (ArrayExpress) | about 30 min, 2 GB | [record](../doors/COMMISSIONING_NOTE_DECONV_V21.md) | **PASS** 2026-10-18, 5 of 5 numbers identical |
| `cscore_band` | **The C-Score's healthy range holds in a laboratory that did not set it (GSE184269).** Healthy whole bloods from a second laboratory read inside the C-Score band: 22 of 23 not above 1.152, the bar sealed before the arrays were fetched. | GSE184269 | about 40 min, 3 GB | [record](../doors/CSCORE_COMMISSIONING_PLAN.md) | **PASS** 2026-10-18, 7 of 7 numbers identical |
| `iama_line` | **IAM-A's self-reference line, rebuilt from 158 public healthy-cell sequencing files.** Every healthy cell type's held error falls on one simple line; recounted from GEO and ENCODE files through chain Stage Q2, every count, the line and the sample-shape reference come out identical. | GSE186458 (153 .pat files, GEO); ENCODE ENCFF092SBB, ENCFF605NRU, ENCFF133QNK, ENCFF704SCQ, ENCFF969NGH; ENCODE GRCh38 analysis set | about 3 h on 16 cores, 70 GB | [record](../doors/COMMISSIONING_NOTE_IAMA.md) | **PASS** 2026-10-18, 3 of 3 numbers identical |
| `three_labs` | **The same DNA gives the same reading in three laboratories (SEQC2 EpiQC, GSE230132).** Identical cell-line DNA run at three laboratories: Met-A's technical repeats agree within 0.0114 and the same DNA across laboratories within 0.0084. | GSE230132 | about 30 min, 3 GB | [record](../doors/DEV_EPIQC_ARRAY_01_OUTCOME.md) | **PASS** 2026-10-18, 4 of 4 numbers identical |
| `mice` | **Physics called it in advance, in living mice (GSE77079).** Mice that cannot make enough of the methyl donor hold their marks worse, by the amount the law predicted before the data were read: knockout over wild type 1.124, inside the sealed window 1.016 to 1.135 (p 0.0003). | GSE77079 (38 RRBS runs, ENA); UCSC mm10 | about 10 h on 16 cores (mm10 index 100 min, then 38 runs at about 12 min each), 40 GB | [record](../doors/DEV_SAM_LEVER_01_OUTCOME.md) | **PASS** 2026-10-19, compared numbers to be recorded |
| `colon` | **Colorectal tumors leave their own healthy tissue's range (GSE131013, 88 patients).** Each tumor read against the same patient's adjacent normal tissue at the atlas's composition-blind sites: 28 of 88 pairs above the bar fixed on normal pairs alone (11 needed), the sealed fingerprint verdict. | GSE131013 (GEO); IAM-Atlas v2, Zenodo 10.5281/zenodo.23325865 | about 2 h, 15 GB | [record](../doors/DEV_TISSUE_COLON_01_OUTCOME.md) | **PASS** 2026-10-18, 6 of 6 numbers identical |
| `breast` | **Breast cancer: tumors leave the patient's own healthy range far more often than chance.** Each tumor read against the same patient's healthy tissue at the atlas's composition-blind sites: 23 of 53 pairs above the bar fixed on normal pairs alone (about 3 expected by chance), the sealed fingerprint verdict. | TCGA-BRCA 450K (NCI GDC open access); IAM-Atlas v2, Zenodo 10.5281/zenodo.23325865 | about 2 h, 20 GB | [record](../doors/DEV_TISSUE_BREAST_01_OUTCOME.md) | **PASS** 2026-10-18, 6 of 6 numbers identical |
| `liver` | **Liver cancer: a third cancer, the same result.** Each tumor read against the same patient's healthy tissue at the atlas's composition-blind sites: 22 of 50 pairs above the bar fixed on normal pairs alone (about 2 expected by chance), the sealed fingerprint verdict. | TCGA-LIHC 450K (NCI GDC open access); IAM-Atlas v2, Zenodo 10.5281/zenodo.23325865 | about 2 h, 20 GB | [record](../doors/DEV_TISSUE_LIVER_01_OUTCOME.md) | **PASS** 2026-10-18, 6 of 6 numbers identical |

Tolerances: numbers are compared exactly (to the stored precision) except where a test's `tolerance_note` in [tests.json](tests.json)
says why not (the sequencing aligner can place a few reads differently with a different thread count).
