# DEV-FINGERPRINT-02 — replicating the cancer fingerprint at another laboratory (development; step 1, 2026-10-10, no cancer file read)

**Why.** DEV-FINGERPRINT-01 passed on one cancer line against one normal culture from one laboratory. ENCODE's HAIB laboratory ran RRBS and
450K arrays on the same cells for cancer lines and normal cells of the same tissue (`data/DEV_FINGERPRINT_02/inventory.py`, `encode_inventory.csv`):
| tissue | cancer | normal | lineage match |
|---|---|---|---|
| prostate | LNCaP (3 RRBS experiments, 2 arrays) | prostate epithelial cells (1 RRBS, 1 array) | yes: the same pair as FINGERPRINT-01, independent laboratory |
| liver | HepG2 | hepatocytes | yes |
| lung | A549 | alveolar epithelial cells; bronchial epithelial cells | yes (A549 is alveolar-derived) |
| breast | MCF-7, T47D (luminal) | mammary epithelial cells, MCF 10A (basal-like) | no: recorded, weaker |
**Reads.** RRBS, single-end, 36 bases (MCF-7 also 50). **Readability check on the normal side only** (prostate epithelial cells, first 2.3 M reads of
ENCFF000MLS, unaligned): 1.17 % of reads carry >= 6 methylated CG, an upper bound on what Stage Q can read; about 200,000 candidate molecules per
replicate, more than the mouse RRBS runs of DEV-SAM-LEVER-01 (150,000 opportunities per run).
**Plan, in order (each step committed before the next reads anything new).**
1. Box: normal-side RRBS only, pinned pipeline (Trim Galore --rrbs, bwa-meth 0.2.0 on hg19, wgbstools bam2pat), with intake.
2. Laptop: on the normal files, Stage Q's measured response and the run-loss simulation (as for PrEC); normal-side 450K arrays through Stage 1;
   identity sites for each normal cell type from the atlas v2 posterior by the canon site rule; power for each pair.
3. Seal both arms per pair, with the arm choice, before any cancer RRBS or array is read.
4. Box: cancer RRBS; laptop: cancer arrays; score as sealed.
**Prediction (to be sealed in step 3):** a fingerprint in every lineage-matched pair.
