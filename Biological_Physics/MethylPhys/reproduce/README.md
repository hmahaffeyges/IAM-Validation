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

| test | what it shows | public inputs | time, disk | record | from a fresh clone |
|---|---|---|---|---|---|
| `metaa_rebuild` | **Met-A's reference, rebuilt from public arrays: noise sites and gate, the floor, the 6,000 identity sites, the healthy spread and the blood-composition markers.** Every reference matrix Met-A reads is rebuilt from 91 public purified-blood arrays and comes out identical to the frozen file the chain reads. | GSE110554; GSE167998; GSE2998xxx neutrophil arrays (listed in reference_donors.csv) | about 3 h, 20 GB | [record](../doors/COMMISSIONING_NOTE_METAA_NEUTROPHILS.md) | **PASS** 2026-10-18, 8 of 8 numbers identical |
| `deconv` | **Blood-cell shares in 200 mixtures from a laboratory the atlas never saw (E-MTAB-2145).** The Deconvolver reads 200 mixtures of purified cells from an unseen laboratory within 0.016 of the true shares, and its sample check refuses the arrays that do not read as their own cell. | E-MTAB-2145 (ArrayExpress) | about 30 min, 2 GB | [record](../doors/COMMISSIONING_NOTE_DECONV_V21.md) | **PASS** 2026-10-18, 5 of 5 numbers identical |
| `cscore_band` | **The C-Score's healthy range holds in a laboratory that did not set it (GSE184269).** Healthy whole bloods from a second laboratory read inside the C-Score band: 22 of 23 not above 1.152, the bar sealed before the arrays were fetched. | GSE184269 | about 40 min, 3 GB | [record](../doors/CSCORE_COMMISSIONING_PLAN.md) | **PASS** 2026-10-18, 7 of 7 numbers identical |
| `iama_line` | **IAM-A's self-reference line, rebuilt from 158 public healthy-cell sequencing files.** Every healthy cell type's held error falls on one simple line; recounted from GEO and ENCODE files through chain Stage Q2, every count, the line and the sample-shape reference come out identical. | GSE186458 (153 .pat files, GEO); ENCODE ENCFF092SBB, ENCFF605NRU, ENCFF133QNK, ENCFF704SCQ, ENCFF969NGH; ENCODE GRCh38 analysis set | about 3 h on 16 cores, 70 GB | [record](../doors/COMMISSIONING_NOTE_IAMA.md) | **PASS** 2026-10-18, 3 of 3 numbers identical |
| `three_labs` | **The same DNA gives the same reading in three laboratories (SEQC2 EpiQC, GSE230132).** Identical cell-line DNA run at three laboratories: Met-A's technical repeats agree within 0.0114 and the same DNA across laboratories within 0.0084. | GSE230132 | about 30 min, 3 GB | [record](../doors/DEV_EPIQC_ARRAY_01_OUTCOME.md) | **PASS** 2026-10-18, 4 of 4 numbers identical |
| `mice` | **Physics called it in advance, in living mice (GSE77079).** Mice that cannot make enough of the methyl donor hold their marks worse, by the amount the law predicted before the data were read: knockout over wild type 1.124, inside the sealed window 1.016 to 1.135 (p 0.0003). | GSE77079 (38 RRBS runs, ENA); UCSC mm10 | about 10 h on 16 cores (mm10 index 100 min, then 38 runs at about 12 min each), 40 GB | [record](../doors/DEV_SAM_LEVER_01_OUTCOME.md) | being run |
| `colon` | **Colorectal tumors leave their own healthy tissue's range (GSE131013, 88 patients).** Each tumor read against the same patient's adjacent normal tissue at the atlas's composition-blind sites: 28 of 88 pairs above the bar fixed on normal pairs alone (11 needed), the sealed fingerprint verdict. | GSE131013 (GEO); IAM-Atlas v2, Zenodo 10.5281/zenodo.23325865 | about 2 h, 15 GB | [record](../doors/DEV_TISSUE_COLON_01_OUTCOME.md) | **PASS** 2026-10-18, 6 of 6 numbers identical |
| `breast` | **Breast cancer: tumors leave the patient's own healthy range far more often than chance.** Each tumor read against the same patient's healthy tissue at the atlas's composition-blind sites: 23 of 53 pairs above the bar fixed on normal pairs alone (about 3 expected by chance), the sealed fingerprint verdict. | TCGA-BRCA 450K (NCI GDC open access); IAM-Atlas v2, Zenodo 10.5281/zenodo.23325865 | about 2 h, 20 GB | [record](../doors/DEV_TISSUE_BREAST_01_OUTCOME.md) | **PASS** 2026-10-18, 6 of 6 numbers identical |
| `liver` | **Liver cancer: a third cancer, the same result.** Each tumor read against the same patient's healthy tissue at the atlas's composition-blind sites: 22 of 50 pairs above the bar fixed on normal pairs alone (about 2 expected by chance), the sealed fingerprint verdict. | TCGA-LIHC 450K (NCI GDC open access); IAM-Atlas v2, Zenodo 10.5281/zenodo.23325865 | about 2 h, 20 GB | [record](../doors/DEV_TISSUE_LIVER_01_OUTCOME.md) | **PASS** 2026-10-18, 6 of 6 numbers identical |

Tolerances: numbers are compared exactly (to the stored precision) except where a test's `tolerance_note` in [tests.json](tests.json)
says why not (the sequencing aligner can place a few reads differently with a different thread count).
