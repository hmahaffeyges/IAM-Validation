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
SD of ln ε 0.026) ≥ 0.96 in every cell of the table except 90 µM/30 % renewed; with twice that spread 0.52–0.98; with the cross-
laboratory species spread (0.136) 0.10–0.97. The 8 wild-type mice measure the actual spread: if it gives power < 0.8 at the table's
middle value (60 µM, 60 % renewed, 1.085), the outcome is recorded as UNDECIDED, not as a fail.

**Bars (fixed now):**
1. Knockout-vehicle IAM-A ÷ wild-type median > 1, one-sided p < 0.05 (6 vs 8).
2. The ratio lies inside 1.03–1.26 (the table's range). Above 1.26: more than the energy lever (e.g. the steatohepatitis cell mix).
3. Knockout given SAMe reads between the two (reported, not a bar: liver SAMe at sacrifice was not raised, 24 h after the last dose).
4. Instrument checks: conversion and coverage per sample; same pipeline for all 19; no reference from another laboratory.
**Confounds written now:** steatohepatitis changes the cell mix and proliferation; the knockout's raised methionine; 'global DNA
methylation unchanged' in the knockout (Lu 2001) — expected, since the predicted ε shift is under 1 % of sites.
