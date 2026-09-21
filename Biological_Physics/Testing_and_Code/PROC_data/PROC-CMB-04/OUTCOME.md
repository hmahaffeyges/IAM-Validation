# OUTCOME — PROC-CMB-04: the patient's sky. C2′ FAILED AS SEALED (1/4 labs, all four within 0.004 of the bar); C4″, C5, C6 PASS. Row 4.6 COMMISSIONED with the failure on the record.

**Run 2026-09-21, as sealed.** 320 healthy whole-blood arrays, four laboratories, seed-2028 panel/test split; z_i = (β_i − Σ_c f_c μ_ci − m_lab,i)/s_lab,i; shrinkage target = β-bin median SD.

| lab | held-out median frac \|z\| > 2 [range] | median z | C2′ |
|---|---|---|---|
| GSE87571 | 0.029 [0.009–0.196] | -0.013 | FAIL (below 0.030) |
| GSE42861 | 0.026 [0.010–0.115] | -0.051 | FAIL (below 0.030) |
| GSE111629 | 0.032 [0.009–0.137] | -0.022 | PASS |
| GSE125105 | 0.029 [0.008–0.151] | -0.039 | FAIL (below 0.030) |

**Reading.** The sky is centred in every laboratory (median z −0.013 … −0.051). The tail is 2.6–3.2 % against a sealed floor of 3.0 %: three labs miss by ≤ 0.004. The scale is ~1.1× conservative, uniformly. As sealed this is a FAIL and is recorded as one; the analyst's judgement (author's standing instruction: do what you surmise best and log it) is that a uniformly conservative scale of that size is a **calibration constant to be stated with every sky**, not a defect to iterate a fifth seal on while rows 7, 8, 9 and N wait. The number is written into the module docstring and the conductor's `calibration_note`: *a healthy sky is quiet at 2.6–3.2 %, not 5 %.* The author may overrule and order CMB-05.

**C4″ gate on 160 TEST arrays: rule 160/160 → PASS.** Rendered per class: immune 160, progenitor 150, stem_adult 19, stem_pluri 1 — and on every rendered panel the healthy tail is quiet (progenitor 0.033, immune 0.032, stem_adult 0.042, stem_pluri 0.069). Terminal / cycling / secretory / stromal rendered in 0 of these 160; that is what these 160 healthy arrays showed, **not a statement about what healthy blood can carry** (author, 2026-09-21: "we have detected secretory and cycling in whole blood and scored it fine"). Presence floors, measured: terminal 0.030, all others 0.020 (`presence_floors_v1.json`).

**Per-person identity-loci offset.** On a class panel the median z of a healthy person ranges −0.30 … +0.59 (GSE125105 held-out, immune loci; median +0.15) while the all-loci median sits at 0: this is the individual's own position in the identity band — the gauge's spread, seen locus by locus — not a zero error.

**C3 cross-lab** (A's zero and scale on B's held-out): GSE87571->GSE42861 0.082 / -0.27; GSE87571->GSE111629 0.070 / +0.08; GSE87571->GSE125105 0.180 / -0.33; GSE42861->GSE87571 0.062 / +0.24; GSE42861->GSE111629 0.091 / +0.37; GSE42861->GSE125105 0.133 / -0.18; GSE111629->GSE87571 0.084 / -0.08; GSE111629->GSE42861 0.155 / -0.38; GSE111629->GSE125105 0.194 / -0.33; GSE125105->GSE87571 0.084 / +0.19; GSE125105->GSE42861 0.076 / +0.05; GSE125105->GSE111629 0.071 / +0.28. Same result as the lab zero: the constants belong to the laboratory.

**C5** mapping rebuild identical (sha df9d0e56…), 0 unmapped atlas CpGs. **C6** identical pixel arrays on regeneration. **C1** (retired formula) 60.7 % — closed.

**Four seals to get here** (CMB-01 no zero; -02 blood classes masked by their own p99; -03 RMS-inflated scale, render-cap bar retired by the author; -04). Each failure is written where it happened.

**In code.** `CPG_Engine/stage_4_6_patient_cmb.py`; `cpg_conductor.stage_4_6_patient_sky` in `run_full` (bundle key `patient_sky`; NOT AVAILABLE without the laboratory's residual scale); `Runtime Matrices/Patient_CMB/` (mapping, four lab scales, presence floors, `build_healpix_mapping.py`); kit test `test_patient_sky.py`; procedure `PROC_CMB_04.py`.

---
**SEALED** sha256 `f8019d9a9bdcaee307c09fd33b6b37ed64081d5288905daeb784158e3d9991d7` · 2026-09-21
