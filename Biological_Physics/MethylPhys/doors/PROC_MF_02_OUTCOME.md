# PROC-MF-02 — outcome: NOT COMMISSIONED. Six of seven bars met, and the detection limit falls to 0.5–1 %; the threshold does not transfer to a fifth laboratory on a reduced marker set (B7 failed).

**Sealed 2026-09-26** against [`PROC_MF_02_PREREG.md`](PROC_MF_02_PREREG.md), fixed before any spike was scored. Same
null and spikes as MF-01: 48 healthy arrays from four laboratories, 1,506 common markers, 768 spikes into real arrays.
Evidence: [`PROC_MF_02.json`](../kit/results/PROC_MF_02.json) · [`PROC_MF_02_null.json`](../kit/results/PROC_MF_02_null.json)
· [`PROC_MF_02_b7_diagnosis.json`](../kit/results/PROC_MF_02_b7_diagnosis.json) · [`PROC_MF_02.py`](../kit/PROC_MF_02.py)
· [`PROC_MF_02.png`](../plates/PROC_MF_02.png)

## Detection limit — ≥ 90 % of real-array spikes detected at ≤ 1 false positive in 48

| cell | NNLS (chain) | inverse-variance detector | same-laboratory weights (control) |
|---|---|---|---|
| Breast | 5 % | **0.5 %** | 0.5 % |
| Colon epithelial | 2 % | **0.5 %** | 0.5 % |
| Cortical neurons | 5 % | **1 %** | 2 % |
| Prostate | 5 % | **1 %** | 1 % |

## Bars

| bar | result |
|---|---|
| B1 lower limit than NNLS on ≥ 3 of 4 | **MET** — 4 of 4, by 4–10× |
| B2 honest σ (null \|z\| > 2 in 2–10 %) | **MET** — 3.1 % |
| B3 unbiased after centring | **MET** — +0.0004 at 2 %, +0.0008 at 5 % |
| B4 centred null on the held-out laboratory | **MET** — worst \|median\| 0.0000 |
| B5 same-laboratory weights ≤ 20 % better | **MET** — no better on any cell |
| B6 blood composition unchanged | **MET** — median Δ 0.0001 |
| B7 false-positive rate ≤ 0.02 on 424 EPIC-Italy controls at the 48-array threshold | **FAILED** — Breast 0.399, Cortical_neurons 0.328 |

## Why B7 failed — measured, not argued

The EPIC-Italy null is **11× wider** than the 450K null (MAD 0.0138 vs 0.0012 for Breast). Two causes, separated:

1. **The marker set.** The EPIC-Italy matrix on disk is the *chain-loci* reduction built for PROC-EPIC-01 — the
   identity loci, not the deconvolver's markers. Only 783 of the detector's 1,506 markers are in it, and a median of
   705 per array. Restricting the **450K** null to those same 783 markers widens it **7×** on its own (MAD 0.0012 →
   0.0089). The missing 723 markers carry most of the detector's power.
2. **Laboratory and platform — for Breast only.** On the same 783 markers, EPIC-Italy is a further **1.5×** wider
   than the 450K laboratories for Breast (0.0138 vs 0.0089); for Cortical_neurons it is **not wider at all**
   (0.0105 vs 0.0111). So the marker set explains the whole neuron failure and most of the breast one; the
   laboratory/platform factor is cell-dependent and, where present, is what a per-laboratory commissioning panel
   exists to absorb.

A threshold set on 1,506 markers applied to 705 is not the same detector. B7 as pre-registered was the right test
and it was run on the only EPIC-Italy matrix that exists; it failed for a reason the pre-registration did not
anticipate, and the decision rule does not care why.

## Decision, by the pre-registered rule

*Any bar failing → not adopted.* The inverse-variance detector is **not in the chain**. What this procedure
establishes, and what is carried forward as fact: on four 450K laboratories, with a threshold set on those
laboratories' own healthy panels, the detector finds 0.5 % Breast or colon epithelium and 1 % neuron or prostate
in blood, with an honest σ, no bias after centring, and no effect on the blood composition. What it does not
establish: that a threshold travels between marker sets or platforms. Nothing here says the physics is wrong; it
says the *threshold* is a laboratory-commissioned quantity, exactly as the laboratory zero already is.

**PROC-MF-03, if the author commissions it:** the same detector with (a) the detection threshold set on **each
laboratory's own commissioning panel** (the EPIC-Italy p99.5 on its own 424 controls is 0.035 for Breast — ten times the 450K threshold — and 0.020
for Cortical_neurons, four times), (b) B7 run on the **full** EPIC-Italy matrix — the 845 arrays re-extracted at the deconvolver's
markers, not the identity loci — and (c) the same B1–B6. A detector whose threshold is per-laboratory is
consistent with everything else the chain does; a universal threshold was the wrong ask, and it was mine.

## Recorded because they cost a bar

- A matrix reduced for one purpose (identity scoring) was named as evidence for another (detection). **Check
  that a named evidence file carries the quantity at the resolution the bar needs**, not only that it exists —
  the same lesson as PROC-EPIC-01's B6, one level up.
- My first B7 pass dropped every EPIC-Italy array on a single missing value and printed FAILED on an empty set.
  Caught because the output said *0 controls used*; fixed to use each array's present markers before any number
  was read.
