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


## Donor null (2026-10-10, normal data only, before any cancer file): what two healthy donors read against each other

The bronchial normal holds two donors, each with RRBS and an array (ENCODE: donor A ENCDO106AAA = ENCSR000DFF + ENCSR000ACY; donor B ENCDO099AAA =
ENCSR000DFV + ENCSR000ACA). `data/DEV_FINGERPRINT_02/donor_null_02.py` ran the scorer's code unchanged (only its pair table) with one donor as the
normal and the other in the cancer slot, both ways (`donor_null_02_output.json`; planted-loss response measured on each reference donor's own molecules,
`data/DEV_FINGERPRINT_02/donor_null/`). Not a sealed test: both arrays' identity-site entropies had been printed by `normal_array_check.py`.

| reference, tested | IAM-A_rel | Met-A_rel | curve B | Arm A z | Arm B excess per tested file | Arm B bar (reference libraries) | Arm B |
|---|---|---|---|---|---|---|---|
| A, B | 1.0757 | 0.9756 | 1.1688 | -5.57 | 0.0211, 0.0145 | 0.0011 | FINGERPRINT |
| B, A | 0.9296 | 1.0250 | 1.0000 | 0.71 | 0.0041, 0.0058 | 0.0000 | FINGERPRINT |

**What it shows.** (1) Arm B, with a bar from one donor's libraries, calls a fingerprint between two healthy donors, both ways. Any sample that is not
the reference reads a positive excess, because the territory is chosen on the reference's own molecules; only another donor measures how large that
baseline is. The largest healthy donor-to-donor excess measured here is 0.0211. (2) Healthy donors differ off curve B in Met-A by up to
0.1933 (donor B: higher copy error, lower identity-site entropy), far beyond Arm A's se; Arm A cannot decide a pair whose
normal is one donor. (3) The first fingerprint test (DEV-FINGERPRINT-01) was read against one PrEC culture with a bar of library spread only; its
deciding Arm B excess (LNCaP +0.1105 to +0.1130) is 5.2 times the largest healthy donor-to-donor excess measured here, so its verdict
stands clear of donor variation; this check is recorded in its outcome note. One donor pair, one tissue, one laboratory: the donor-to-donor range is a
lower bound.

## Pair lung_bronchial (A549 against bronchial epithelial cell), rule 02b: sealed before any cancer file is aligned or read

**Sealed:** `data/DEV_FINGERPRINT_02/sealed_lung_bronchial.json` (scorer `score_fingerprint_02b.py`, sha256 in the seal; the score step refuses unless the seal is
committed unchanged). Normal: 4 RRBS files, Stage Q copy error 0.0423-0.0468; arrays ENCSR000ACA, ENCSR000ACY. The normal holds two donors (the donor null).
**Arm B decides.** Bar 0.0211 = max(normal-library excess 0.0000, healthy donor-to-donor excess 0.0211).
Run-loss sensitivity on these molecules (planted 0.10): 0.0898. Planted test of the sealed scorer (`plant02b_lung_bronchial_output.json`): NULL no fingerprint (excess -0.0027 to 0.0000); RUN05 FINGERPRINT (excess 0.0466 to 0.0484); RUN01 no fingerprint (excess 0.0075 to 0.0099); SCAT no fingerprint (excess -0.0026 to 0.0003).
FINGERPRINT if every one of the 2 A549 files listed in the seal reads above the bar; no fingerprint if every file reads at or below
it; otherwise UNDECIDED. Arm A (Met-A against curve B) is computed and reported, not deciding.
**Synthetic test:** DEV_FINGERPRINT_02/plant_fingerprint_02b.py, DEV_FINGERPRINT_02/plant02b_lung_bronchial_output.json, DEV_FINGERPRINT_02/donor_null_02.py, DEV_FINGERPRINT_02/donor_null_02_output.json, DEV_FINGERPRINT_02/prep_normal.sh, DEV_FINGERPRINT_02/normal_bronchial_epithelial_cell/runloss_sim.txt, Biological_Physics/MethylPhys/boxruns/run9_fp2/synth_align.py, Biological_Physics/MethylPhys/boxruns/run9_fp2/synth_align_result.csv

## Pair lung_alveolar (A549 against epithelial cell of alveolus of lung), rule 02b: sealed before any cancer file is aligned or read

**Sealed:** `data/DEV_FINGERPRINT_02/sealed_lung_alveolar.json` (scorer `score_fingerprint_02b.py`, sha256 in the seal; the score step refuses unless the seal is
committed unchanged). Normal: 2 RRBS files, Stage Q copy error 0.0481-0.0567; arrays ENCSR422EPB. The normal is one donor; the donor term in the bar comes from the bronchial donor null.
**Arm B decides.** Bar 0.0211 = max(normal-library excess 0.0000, healthy donor-to-donor excess 0.0211).
Run-loss sensitivity on these molecules (planted 0.10): 0.0804. Planted test of the sealed scorer (`plant02b_lung_alveolar_output.json`): NULL no fingerprint (excess -0.0037 to 0.0000); RUN05 FINGERPRINT (excess 0.0453 to 0.0489); RUN01 no fingerprint (excess 0.0059 to 0.0084); SCAT no fingerprint (excess -0.0037 to 0.0003).
FINGERPRINT if every one of the 2 A549 files listed in the seal reads above the bar; no fingerprint if every file reads at or below
it; otherwise UNDECIDED. Arm A (Met-A against curve B) is computed and reported, not deciding.
**Synthetic test:** DEV_FINGERPRINT_02/plant_fingerprint_02b.py, DEV_FINGERPRINT_02/plant02b_lung_alveolar_output.json, DEV_FINGERPRINT_02/donor_null_02.py, DEV_FINGERPRINT_02/donor_null_02_output.json, DEV_FINGERPRINT_02/prep_normal.sh, DEV_FINGERPRINT_02/normal_epithelial_cell_of_alveolus_of_lung/runloss_sim.txt, Biological_Physics/MethylPhys/boxruns/run9_fp2/synth_align.py, Biological_Physics/MethylPhys/boxruns/run9_fp2/synth_align_result.csv

## Pair prostate (LNCaP clone FGC against epithelial cell of prostate), rule 02b: sealed before any cancer file is aligned or read

**Sealed:** `data/DEV_FINGERPRINT_02/sealed_prostate.json` (scorer `score_fingerprint_02b.py`, sha256 in the seal; the score step refuses unless the seal is
committed unchanged). Normal: 2 RRBS files, Stage Q copy error 0.0438-0.0484; arrays ENCSR000ACD. The normal is one donor; the donor term in the bar comes from the bronchial donor null.
**Arm B decides.** Bar 0.0211 = max(normal-library excess 0.0014, healthy donor-to-donor excess 0.0211).
Run-loss sensitivity on these molecules (planted 0.10): 0.0933. Planted test of the sealed scorer (`plant02b_prostate_output.json`): NULL no fingerprint (excess 0.0000 to 0.0014); RUN05 FINGERPRINT (excess 0.0486 to 0.0513); RUN01 no fingerprint (excess 0.0107 to 0.0119); SCAT no fingerprint (excess 0.0001 to 0.0016).
FINGERPRINT if every one of the 6 LNCaP clone FGC files listed in the seal reads above the bar; no fingerprint if every file reads at or below
it; otherwise UNDECIDED. Arm A (Met-A against curve B) is computed and reported, not deciding.
**Synthetic test:** DEV_FINGERPRINT_02/plant_fingerprint_02b.py, DEV_FINGERPRINT_02/plant02b_prostate_output.json, DEV_FINGERPRINT_02/donor_null_02.py, DEV_FINGERPRINT_02/donor_null_02_output.json, DEV_FINGERPRINT_02/prep_normal.sh, DEV_FINGERPRINT_02/normal_epithelial_cell_of_prostate/runloss_sim.txt, Biological_Physics/MethylPhys/boxruns/run9_fp2/synth_align.py, Biological_Physics/MethylPhys/boxruns/run9_fp2/synth_align_result.csv

## Pair liver (HepG2 against hepatocyte), rule 02b: sealed before any cancer file is aligned or read

**Sealed:** `data/DEV_FINGERPRINT_02/sealed_liver.json` (scorer `score_fingerprint_02b.py`, sha256 in the seal; the score step refuses unless the seal is
committed unchanged). Normal: 2 RRBS files, Stage Q copy error 0.0373-0.0429; arrays ENCSR955LKF. The normal is one donor; the donor term in the bar comes from the bronchial donor null.
**Arm B decides.** Bar 0.0211 = max(normal-library excess 0.0020, healthy donor-to-donor excess 0.0211).
Run-loss sensitivity on these molecules (planted 0.10): 0.1108. Planted test of the sealed scorer (`plant02b_liver_output.json`): NULL no fingerprint (excess 0.0000 to 0.0020); RUN05 FINGERPRINT (excess 0.0497 to 0.0511); RUN01 no fingerprint (excess 0.0104 to 0.0118); SCAT no fingerprint (excess -0.0002 to 0.0020).
FINGERPRINT if every one of the 4 HepG2 files listed in the seal reads above the bar; no fingerprint if every file reads at or below
it; otherwise UNDECIDED. Arm A (Met-A against curve B) is computed and reported, not deciding.
**Synthetic test:** DEV_FINGERPRINT_02/plant_fingerprint_02b.py, DEV_FINGERPRINT_02/plant02b_liver_output.json, DEV_FINGERPRINT_02/donor_null_02.py, DEV_FINGERPRINT_02/donor_null_02_output.json, DEV_FINGERPRINT_02/prep_normal.sh, DEV_FINGERPRINT_02/normal_hepatocyte/runloss_sim.txt, Biological_Physics/MethylPhys/boxruns/run9_fp2/synth_align.py, Biological_Physics/MethylPhys/boxruns/run9_fp2/synth_align_result.csv
