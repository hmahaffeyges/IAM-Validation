# IAM Sequencing Atlas, version 1.1: the megakaryocyte column

Version 1.1 adds a 58th cell type, megakaryocytes, the cells that make platelets and a large source of the DNA in blood plasma. Every file of
version 1.0 is carried here unchanged: the 57 cells read exactly as before.

## What the column is

- **Source:** GSE87197 (public, GEO): whole-genome bisulfite sequencing of single megakaryocytes from healthy bone marrow, on GRCh38. Donor 1
  (33 cells) and donor 2 (19 cells) are pooled per donor; donor 1 is half A of the column and donor 2 half B, so the column carries both
  noise maps like every other cell. The 52 files are listed in `mk_build_files.csv`; `mk_inputs.csv` is the column's input list.
- **Conversion:** each donor's cells moved from GRCh38 to the atlas's hg19 CpG numbering by the chain (`run_sample.py --build seq-lift`;
  reports `mk_donor1_lift.json`, `mk_donor2_lift.json`; its synthetic round trip is recorded in the repository).
- **Build:** `run_sample.py --build seq-atlas`, the same command as the 57 cells, in 29 pieces of 1,000,000 CpGs
  (`pieces_megakaryocytes_29.tar`, each piece with its build report; SHA256 of every file in `SHA256_MANIFEST_megakaryocytes.txt`).
  25,220,009 of the 28,217,448 CpGs hold reads; the median depth is about 4 reads a CpG, shallower than the 57 cells.
- **Status:** provisional. Two donors at low depth: its range map is wide and its donor-halves noise map is one donor against the other.

## Its test

Platelet DNA from 3 healthy people (GSE206818, public, a merged level track on hg19, never read by the chain before) was compared with all 58
cells by the chain's counting-weighted score: **megakaryocytes are its nearest cell** (score 1.1186; the next, NK cells, 2.9281). By plain
distance megakaryocytes rank 34th, as the test's synthetic showed for a shallow column: plain distance counts the column's own counting scatter
as difference. Result: `mk_platelet_compare.json`; synthetic: `mk_platelet_synthetic.txt`.

## Rebuilding it

Clone https://github.com/hmahaffeyges/IAM-Validation, install `Biological_Physics/MethylPhys/chain/requirements.txt`, download the 52 files
in `mk_build_files.csv` from GEO, convert each donor with `run_sample.py --build seq-lift`, and build each piece with
`run_sample.py --build seq-atlas --build-samples mk_inputs.csv --build-block START:END`. The record of every step is
`Biological_Physics/MethylPhys/doors/DEV_ATLAS_V21_01.md` in the repository.
