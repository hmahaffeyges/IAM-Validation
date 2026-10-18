# Try to break it

Every instrument in the commissioned IAM chain was commissioned on a test whose rule was written and sealed before its data were read. The
strongest test left is yours: run the chain on samples **you already know**, from your own laboratory, and see whether it does what is written
below. If it does not, that is a result, and we want it reported in public.

## 1. Install and check

```bash
git clone https://github.com/hmahaffeyges/IAM-Validation.git && cd IAM-Validation/Biological_Physics/MethylPhys/chain
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python3 release_check_v4.py          # every check must pass on your machine before you read your own samples
```

The operating procedure is [`Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v4.md`](Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v4.md).

## 2. Read your samples

```bash
# arrays (EPIC v1 IDAT pairs): first read three or more healthy arrays from the same run, each on its own
python3 MethylPhys_Interface/run_sample.py --grn R1_Grn.idat --red R1_Red.idat --specimen "whole blood" --out R1.html
# then list their readings (columns id, A, C: Met-A's A and the C-Score's C from each one's bundle) in same_run_refs.csv, and read the sample against them
python3 MethylPhys_Interface/run_sample.py --grn S_Grn.idat --red S_Red.idat --specimen "whole blood" --ref-table same_run_refs.csv --out S.html
# whole-genome bisulfite sequencing, aligned with Bismark
python3 MethylPhys_Interface/run_sample.py --bam S.bam --ref-fasta genome.fa --out S.html
```

Each run writes the report (HTML and two PDFs), its bundle (JSON) and one ledger row. Keep all three: they are what a report of a break needs.

## 3. What each instrument should do, written down in advance

Each line below is the instrument's commissioned scope and the bar it met. A result outside the scope is printed as preliminary; it is
interesting, but it is not a break of a commissioned claim.

| instrument | scope (as commissioned) | what it should do on your known samples | it is broken if | commissioning result |
|---|---|---|---|---|
| **Met-A** | isolated neutrophils, EPIC v1, tared by ≥ 3 same-run healthy references | healthy neutrophils read Normal (0.95-1.05); repeat arrays of one person agree | fewer than 95 % of your healthy isolated neutrophils read Normal, or one person's repeat arrays differ by SD > 0.020 | 525 of 541 healthy arrays Normal (97.0 %, 18 series); other laboratories' purified neutrophils 68 of 68; repeat SD 0.016 |
| **C-Score** | whole blood, EPIC v1, tared by ≥ 3 same-run healthy references | healthy whole blood reads *not above the band* (C_rel ≤ 1.152); scattered change never reads above it; clustered change does | more than 5 % of your healthy whole bloods read above 1.152, or a scattered change reads above it, or a clustered change at 5 % of sites does not reach C_rel 1.5 | three laboratories: 34 of 35, 22 of 23, 172 of 173 not above; 0 of 120 planted scattered changes above |
| **IAM-Deconvolver** | shares of neutrophils, monocytes, CD4 T and CD8 T cells in blood mixtures, arrays, against IAM-Atlas v2.1 | mixtures of known composition come back at their true shares | median absolute share error above 0.03 | 0.016 on 200 mixtures from a laboratory the atlas never saw |
| **IAM-A** | healthy blood CD8 T cells, whole-genome bisulfite, read against the sample's own loose share | healthy samples read Normal (band v1, 0.779-1.055) | fewer than 75 % of your read healthy samples are Normal (at least 8 read) | 7 of 8 Normal on a laboratory it had never seen |
| **IAM-Atlas v2.1** | the uses in its [commissioning note](Biological_Physics/MethylPhys/doors/COMMISSIONING_NOTE_ATLAS_V21.md) | your healthy cells of those types fall inside the atlas's range | below the bar the note states for that use | e.g. 97.8 % of 45 cortical-neuron arrays, 100 % of 53 lung-bronchus arrays inside |

**Declines are not breaks.** The chain refuses what it cannot read and says why: a failed array, an unsupported platform, a sample the
sample-shape check does not recognize, too few qualifying molecules. A refusal of a sample it should have read, with a wrong reason, is a break.

**The physics.** Nothing in the table is fitted to your data. Every reference is a frozen file with its checksum checked by the release check;
the floor and saturation ends of every gauge are derived (Met-A's saturation 3.03 is every identity site fully scrambled; IAM-A's is
0.5 / ε_pred(L), the sample's own). So each line of the table is a prediction made before your samples existed.

## 4. Report it

Open an issue with the **Break report** form (Issues → New issue). Attach the report, the bundle and the ledger row, say what you knew about the
samples before you ran them, and whether the samples are public (an accession) or private. Every report is answered in public, and every break
that holds goes into the record and the book beside the result it breaks.
