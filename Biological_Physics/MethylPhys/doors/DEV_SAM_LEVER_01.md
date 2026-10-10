# DEV-SAM-LEVER-01 — lowering the energy supply raises IAM-A (written 2026-10-09, before any data are downloaded)

**Data:** GSE77079 (mouse liver RRBS, one laboratory, 10-month-old males): Mat1a knockout on vehicle (6), knockout given SAMe
30 mg/kg/day for 8 weeks (5), wild type (8). Raw reads SRX1539708-…, read with the reference-free RRBS reader (rrbs_iama.py).
**Measured lever:** hepatic SAM (AdoMet) reduced by 74 % in the knockout (Lu et al. 2001, PNAS 98:5560, 3-month mice), a 3.85-fold drop.
The absolute wild-type level is not in the text; 30–90 µM (nmol/g) is taken as the range (assumption, not measured here).

**Prediction (calculated).** Restore ∝ SAM/(K_m + SAM), K_m 4.4 µM (DNMT1), at copying. A 3.85-fold drop moves the holding energy by
ln of the restore ratio and so raises ε. Only cells renewed since SAM fell carry it; knockout livers proliferate more than wild type.

| liver SAM | 30 % renewed | 60 % renewed | all renewed |
|---|---|---|---|
| 30 µM | 1.079 | 1.156 | 1.256 |
| 60 µM | 1.043 | 1.085 | 1.140 |
| 90 µM | 1.029 | 1.058 | 1.097 |

(IAM-A of knockout ÷ wild type, same laboratory and pipeline.)

**Power** (one-sided Mann–Whitney, 6 vs 8): with the within-laboratory donor spread measured on human neutrophils (IAM-A SD 0.02,
SD of ln ε 0.026) ≥ 0.96 in every cell of the table except 90 µM/30 % renewed; with twice that spread 0.31–1.00 (0.31 at 90 µM, 30 % renewed); with the cross-
laboratory species spread (0.136) 0.10–0.97. The 8 wild-type mice measure the actual spread: if it gives power < 0.8 at the table's
middle value (60 µM, 60 % renewed, 1.085), the outcome is recorded as UNDECIDED, not as a fail.

Reproduce: `development/sims/sam_lever_01.py` (inputs: this repo only).

**Bars (fixed now):**
1. Knockout-vehicle IAM-A ÷ wild-type median > 1, one-sided p < 0.05 (6 vs 8).
2. The ratio lies inside 1.03–1.26 (the table's range). Above 1.26: more than the energy lever (e.g. the steatohepatitis cell mix).
3. Knockout given SAMe reads between the two (reported, not a bar: liver SAMe at sacrifice was not raised, 24 h after the last dose).
4. Instrument checks: conversion and coverage per sample; same pipeline for all 19; no reference from another laboratory.
**Confounds written now:** steatohepatitis changes the cell mix and proliferation; the knockout's raised methionine; 'global DNA
methylation unchanged' in the knockout (Lu 2001) — expected, since the predicted ε shift is under 1 % of sites.


**Reader change before any knockout is read (2026-10-10).** The reference-free reader fails its check on the wild-type file (`data/DEV_SAM_LEVER_01/reader_check_01.md`). Replaced by: adapter and RRBS fill-in trimming (Trim Galore --rrbs), bwa-meth on mm10, wgbstools bam2pat (mm10), Stage Q's rule (pat_site_table). Then the in-silico response on the wild-type molecules and the window sealed through it, before any knockout ε is computed. Bars 1-4 unchanged.

**Power through Stage Q (2026-10-10, before any knockout is read; `development/sims/sam_lever_02.py`, output `sam_lever_02_output.txt`).**
The earlier power compared the calculated rise in true copy error with a spread converted by the simple form, and its cross-lab column used
the withdrawn reference-free table. Through Stage Q's measured response (two stand-ins), the middle case (60 µM, 60 % of cells renewed) reads
IAM-A_rel 1.037–1.047, not 1.085. Power for 6 knockout against 8 wild type (one-sided Mann-Whitney, α 0.05):

| spread between mice (IAM-A SD) | middle case power | weakest case (90 µM, 30 %) | strongest (30 µM, 100 %) |
|---|---|---|---|
| 0.01 | 1.00 | 0.65–0.82 | 1.00 |
| 0.02 | 0.90–0.98 | 0.24–0.35 | 1.00 |
| 0.04 | 0.42–0.59 | 0.11–0.14 | 1.00 |

**Rule, fixed now.** The spread among the 8 wild-type mice is measured first (wild types are the reference arm). Then, before any knockout is
read, the window is sealed on Stage Q's response on the wild-type molecules and the power at the middle case is stated with it. If that power is
below 0.80, a reading inside the null range is recorded as **undecided**, not as a failure of the derivation.

**Window sealed 2026-10-10T18:19Z, on the first wild-type mouse's molecules, before any knockout file is downloaded or read.**
Aligned pipeline (Box Run 6: Trim Galore --rrbs, bwa-meth 0.2.0 on mm10, wgbstools bam2pat). Wild type 658K, two runs: Stage Q ε 0.0396 and 0.0409
(149,966 and 156,083 opportunities; `data/DEV_SAM_LEVER_01/eps_wt_658K_two_runs.json`). Readable molecules (>= 6 calls, >= 80 % methylated) are
0.57 % of all molecules: RRBS reads CpG islands, mostly unmethylated. Stage Q's measured response on these molecules
(`insilico_wt_SRR3111471.csv`, whole file) maps the calculated fold rise to IAM-A_rel (`predict_sam_window_01_output.txt`):

| liver SAM (µM) | renewed 30 % | 60 % | 100 % |
|---|---|---|---|
| 30 | 1.043 | 1.086 | 1.135 |
| 60 | 1.023 | **1.046** | 1.077 |
| 90 | 1.016 | 1.032 | 1.053 |

**Prediction:** IAM-A_rel of the knockout vehicle mice (median, each mouse = its runs pooled) ÷ median of the wild-type mice lies in **1.016–1.135**,
and the knockout mice given SAMe read lower than the knockout vehicle mice. Below 1.016: no rise of the size the calculation gives (undecided if the
power stated below is < 0.80, else not met). Above 1.135: a rise larger than the calculation allows.
Power at the middle cell is stated from the 8 wild-type mice's spread before any knockout is read.
