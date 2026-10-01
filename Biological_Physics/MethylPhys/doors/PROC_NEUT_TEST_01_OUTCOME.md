# PROC-NEUT-TEST-01 — outcome, all four tests (2026-10-01). Chain v3 (bc4a651), unchanged, run_sample.py --engine v3 on every array.

Pre-registration sha c8f6b1c5ae7a3268 (with amendment 1, written before reading). 692 arrays across the four tests: 644 for T1, T3 and T4 (in the CSV, of which 5 produced no report) and 48 for T2.

| test | prediction | result |
|---|---|---|
| **T1** healthy whole blood, FACS-counted (GSE112618, 6) | T1a dominant-cell gate agrees with FACS ≥ 5/6 | **6/6 PASS** |
| | T1b neutrophil fraction within 0.05 of FACS (median) | **0.035 PASS** |
| | T1c untared A of readable specimens in 0.93–0.98 | **2/3 FAIL** (0.966, 0.973 in; 0.983 just out) |
| **T2** isolated neutrophils, 2 men, another lab | T2a/b/c | **FAIL**: array noise (separate record: PROC_NEUT_TEST_01_T2_OUTCOME.md) |
| **T3** technical replicates (GSE250556, 4 men, 64 arrays) | T3a, T3b | **Not assessable.** The chain read neutrophils at 0.30–0.56 in these men; A was withheld on 63/64 by the dominant-cell rule |
| **T4** COVID-19 whole blood (GSE179325, 570 read) | T4a SEVERE above Normal more often than NEGATIVE | 32/110 (29 %) vs 11/76 (14 %), one-sided Fisher p = 0.015: **PASS as written** |
| | T4c ≥ 80 % NEGATIVE in Normal | **60/76 = 78.9 % FAIL** |
| | T4b MILD between them (descriptive) | MILD 16 % above: close to NEGATIVE, not between |

**T4a does not survive the composition check.** Severe patients have more neutrophils (median fraction 0.79 vs 0.65). Tared A rises with neutrophil fraction within every
group (ρ ≈ 0.31). With fraction in the model, the severity term is +0.0075 (p = 0.42). The pre-registered pass is a composition effect, not a neutrophil
reading. The pre-registration named this risk ("lymphopenia in severe disease raises the neutrophil fraction").

**What the battery shows about chain v3.**
1. **Composition works.** The dominant-cell gate and the neutrophil fraction match FACS (T1a, T1b).
2. **The whole-blood reading still depends on the neutrophil fraction**, about +0.12 A per unit fraction, after the composition-matched expectation and the tare.
   The expectation profiles come from one lab (Salas). In another lab they don't fully cancel.
3. **Instrument noise sets the floor of the gauge.** Tared healthy (NEGATIVE) bloods spread SD 0.052, wider than the ±0.05 Normal band. Untared A sits around 1.22 in this lab, as in the T2 lab.
   This is the same cause found in T2 (DEV_NOISE_01_OUTCOME.md).
4. Not shown: a disease reading. Neutrophils are not commissioned.

**Fixes, in order (development; each to be tested on held-out data before the chain is changed).**
1. Stage 0 noise index (48,528 invariant sites, frozen) with a gate. The reading is withheld above the floor arrays' range unless tared.
2. Remove the fraction dependence: fit the healthy expectation's slope on fraction from the healthy references in the same batch, instead of using profiles from another lab.
3. Calibrate the bisulfite-conversion threshold at intake. Every array now gets PROCEED_WITH_PENALTY for "threshold uncalibrated".
4. Re-run T4 and T1 with fixes 1–2. Find a technical-replicate set with neutrophil-dominant blood (or isolated neutrophils) for T3.
