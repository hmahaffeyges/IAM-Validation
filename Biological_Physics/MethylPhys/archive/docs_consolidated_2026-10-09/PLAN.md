> **Archived 2026-10-09.** Merged into [`STATUS.md`](../../STATUS.md) (plan, commissioning record) or the LOG [`development/METHYLPHYS_DEVELOPMENT_LOG.md`](../../../../development/METHYLPHYS_DEVELOPMENT_LOG.md) (chain changes). Kept as a record; not updated.

# PLAN: the master plan for every test, and the data behind it

This is the one living plan for MethylPhys: what we test, in what order, and which data in S3 each test uses.
It is updated whenever a box run finishes, a dataset is downloaded or deleted, or the order changes.
The data register is [`DATA_REGISTER.csv`](DATA_REGISTER.csv), one row per dataset in S3.

**Data rule.** A dataset stays in S3 while any planned test still needs it. When every test it serves is marked done in the register
and nothing else is planned for it, it is listed under "Candidates for deletion" below, and it is deleted only after the author agrees.
Public data can be downloaded again if it is needed later.

**Development mode.** A stage is tested and logged in [`development/METHYLPHYS_DEVELOPMENT_LOG.md`](../../../development/METHYLPHYS_DEVELOPMENT_LOG.md)
until it is commissioned. Nothing is pre-registered or sealed before then.

**Box runs.** Everything that does not need the box is done first. Each box run is one planned job list: one driver runs it, writes
every output to S3, and stops the box at the end or on a crash.

---

## 1. Now: commission the neutrophil chain (updated 2026-10-09)

**Met-A (arrays): COMMISSIONED 2026-10-09** (author approval; detection limits 2 % purified / 5 % whole blood printed on every report) — [`COMMISSIONING_NOTE_METAA_NEUTROPHILS.md`](COMMISSIONING_NOTE_METAA_NEUTROPHILS.md):
9 of 11 bars met, 1 not met (a 1 % loss of pattern; detection limits 2 % purified, 5 % whole blood), 1 not met (C-score band on a new
laboratory, 23/26). Box Run 1 complete:

| Job | What | Status |
|---|---|---|
| A | Self-tare II, then the median tare | **done, every bar met** (replicate SD 0.0164; 62/63; 68/68; 6/6) |
| B | Every test set re-read with the adopted tare | **done 2026-10-08**: healthy 525/541 Normal, 18 series |
| C | C-score spread on healthy arrays | done; tared band 0.751-1.409 (DEV-CSCORE-TARE-01) |
| D | Sky statistics, apodised mask | done; bars not met; sky stays withheld |
| E | Composition truth | **done**: FACS bloods and 12 EPIC mixtures; neutrophil fraction within 0.02 |

Also done (no box): constructed sensitivity test (DEV-METAA-SENS-01); new-laboratory granulocytes 26/26 Normal (DEV-NEWLAB-GRAN-01).

**IAM-A (sequencing):** Stage Q0 intake wired and tested on real files (DEV-Q0-HEALTHY-01); P re-measured on whole files, v2 = 1.1492
(DEV-IAMA-P-WHOLE-01); pinned Loyfer pipeline built and format-checked (Box Run 2 session 1).

## 2. IAM-A: Box Run 2 session 2 done (2026-10-09)

Another laboratory's healthy neutrophils (GSE128731), 2 donors × 4 kits. TruSeq 1.047 / 1.042 (Normal); Swift 1.16 (both donors, both
sequencers); QIAseq stopped by Q0 (conversion). IAM-A is repeatable (≤ 0.009 between donors and sequencers) but carries a laboratory-and-kit
offset up to ~0.16 (DEV-IAMA-KIT-01). **Author decision needed:** how IAM-A handles that offset (same-run tare / per-kit P / wider band).
Then: IAM-A healthy band, IAM-A C-score, Met-A against IAM-A on the same donors (GSE128733).

## 3. Win candidates (development runs on the commissioned stages)

1. Treated samples moving toward disorder (12/12 so far).
2. Myeloid disease arrays.
3. One person over time (the difference map; serial mode).
4. Cross-species IAM-A with the body-temperature prediction: the same cell type at a different body or water temperature sits at a
   different distance from its k_B T ln 2 floor (Box Run 3, from the 1.3 TB of 580-species reads already in S3).

## 4. After commissioning: the author's order (2026-09-30)

1. **Human blood diseases** (track A), blood cancers first: the leukocyte as the diseased cell, per-cell Met-A on the affected lineage.
2. **Plasma cfDNA** (track B).
3. **Tissue specimens** (track C), read against the cell atlas; biopsy scoring.
4. **Stool:** can a stool specimen be scored (its shed cells)?
5. **Dogs and other mammals** (track E): no animal cell atlas exists, so build or derive one first; the Mammalian Methylation
   Consortium array (348 species) for species ageing.
6. **Fish** (track F): Methow steelhead and coho, hatchery against wild. Fish blood is nucleated red cells, nearly one cell type.
   IAM-A reads the molecules directly. Possible collaboration with Chelan PUD fish scientists.
7. **Ageing** (track D): serial and cross-sectional blood.
8. **Quantum computing:** check which predictions have come due.

## 5. Chain work still open

1. CD4/CD8 separation and the finer blood subsets (need loci the current block lacks).
2. EPIC v2 intake (calibrated through SeSAMe; needs a v2 floor and v2 replicates).
3. Atlas deconvolution and NILC (both failed their truth bars in development; job E and GSE182379 test them again).
4. Directional decomposition (physics only) and the chip term.
5. Open human data beyond GEO (controlled-access serial studies), once a method is published.

## 6. Still to download

| Data | Size | For |
|---|---|---|
| GSE128731 neutrophil WGBS (Samples 6, 7) | 414 GB (8 runs) | Box Run 2, IAM-A with Met-A on the same cells |
| GSE186458 WGBS atlas, 39 cell types | to size | IAM-A floors per cell type |
| GSE104700 monocyte bisulfite sequencing | to size | IAM-A on monocytes (donor match to GSE56046 not confirmed) |

## 7. Candidates for deletion

None yet. A dataset moves here only when every test it serves is done.

## S3 now (2026-10-05)

2,681 GB in total: 2,570 GB of downloaded data in 80 datasets (the register), plus results, atlas files and private archives.
Largest: 580-species reads 1,321 GB; stool 312 GB; coho RRBS 227 GB; salmonid reads 162 GB. The bucket is on Intelligent-Tiering, so data
nobody reads moves to cheaper storage on its own.

## Records the book checks read

- (2026-09-27: SATSA Stage 1 on AWS: 1,056 of 1,072 arrays calibrated, 0 errors, 16 with no IDAT pair; box reproduces the laptop exactly. **Call rate median 0.894; 738 of 1,056 below the 0.93 intake line**, worst on chip batch 9721 (853 arrays, 73 % below); call rate also falls with age decade (0.911 in the 50s to 0.878 in the 90s), so later draws are lower-quality input - the serial trajectories must carry intake status per draw and cannot treat a below-line draw as a reading. 286 people have >= 2 calibrated draws, 195 >= 3.)
