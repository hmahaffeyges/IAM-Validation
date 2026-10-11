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

**Same test as DEV-FINGERPRINT-01, with four stated differences** (to be carried into the sealed rule): (1) RRBS, 36-base single reads, not WGBS
202-base pairs: fewer readable molecules, mostly in CpG-dense regions; (2) one array per side, not two; (3) prostate on 450K, the other pairs on
EPIC (platform matched within each pair); (4) normal sides are ENCODE primary cultures. Hepatocyte 450K array (ENCSR000ABW) kept 332,213 probes
against ~480,000 on the others: liver is read on EPIC (ENCSR955LKF vs HepG2 ENCSR291NYD). Run order and every script: `data/DEV_FINGERPRINT_02/REPRODUCE.md`.

## Pair breast_mcf10a (MCF-7 against MCF 10A): sealed 2026-10-10 before any cancer file is aligned or read

**Sealed:** `data/DEV_FINGERPRINT_02/sealed_breast_mcf10a.json` (normal side only; scorer sha256 recorded in it; the score step refuses unless this
file is committed unchanged). Normal: 6 MCF 10A RRBS files (Box Run 9, bwa mem -T30), Stage Q copy error per file in `normal_MCF_10A/eps_per_file.json`;
normal array ENCSR329HJR at 5467 breast-basal identity sites. Arm A: slope 2.4 from the MCF 10A planted-loss response, sd_rep 0.009, se 0.0304,
power at a 0.06 excess 0.629 (`power_breast_mcf10a_output.txt`), so a no-fingerprint reading on Arm A is UNDECIDED. Arm B: the run-loss reader recovers a
planted 0.10 run loss as +0.0987 on these molecules (`normal_MCF_10A/runloss_sim.txt`); bar = largest normal-file excess, 0.0 (the reference file reads 0
against itself by construction, the other five read below it). Cancer: the 10 MCF-7 files listed in the seal. Weaker pair: MCF-7 is luminal, MCF 10A
basal-like and immortalised.
**Synthetic test:** Biological_Physics/MethylPhys/boxruns/run9_fp2/synth_align.py, Biological_Physics/MethylPhys/boxruns/run9_fp2/synth_align_result.csv, DEV_FINGERPRINT_02/plant_fingerprint_02.py, DEV_FINGERPRINT_02/plant_fingerprint_02_output.json, DEV_FINGERPRINT_02/power_fingerprint_02.py, DEV_FINGERPRINT_02/power_breast_mcf10a_output.txt, DEV_FINGERPRINT_02/prep_normal.sh, DEV_FINGERPRINT_02/normal_MCF_10A/runloss_sim.txt
