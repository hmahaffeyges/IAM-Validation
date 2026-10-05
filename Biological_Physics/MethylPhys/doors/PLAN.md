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

## 1. Now: commission the neutrophil chain (Box Run 1, running since 2026-10-05)

Job list: [`boxruns/run1/JOBS.md`](../boxruns/run1/JOBS.md).

| Job | What | Status |
|---|---|---|
| A | Self-tare II, then the median tare: commissioning check | **done, every bar met:** replicate SD 0.0164 (≤ 0.020), 62/63 Normal, other laboratories 68/68, floor 6/6 |
| B | Every chain test set read again with the adopted tare | running |
| C | Met-A C-score spread on every healthy array (the author sets the band) | queued |
| D | Sky statistics, hard mask against apodised mask | queued |
| E | Atlas composition on GSE112618 (FACS-counted bloods) | queued |

After Run 1 (no box): score job E against the FACS fractions; the author sets the C-score band; record each outcome in the log.

## 2. Next: IAM-A at scale, and Met-A with IAM-A on the same cells (Box Run 2)

1. **GSE128733 + GSE128731:** the same purified neutrophils on arrays and deep WGBS. Met-A is read (DEV-PAIRED-01). IAM-A on the same
   two specimens needs the neutrophil WGBS runs, 8 runs, 414 GB (one run per donor, about 100 GB, is enough for a first reading).
   **To download.**
2. IAM-A healthy band on purified neutrophils; IAM-A C-score.
3. Met-A against IAM-A on the same cell (the cross-spectrum).
4. Job E on GSE182379 (constructed mixtures; already in S3).

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
