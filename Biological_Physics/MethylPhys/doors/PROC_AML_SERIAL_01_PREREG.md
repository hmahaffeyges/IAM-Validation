# PROC-AML-SERIAL-01 — pre-registration (written 2026-10-01, before any GSE315367 array is read)

**Question.** Read one person over time: does the neutrophil-led Met-A and its C-score place a patient's blood outside the healthy band at AML
diagnosis, and back toward it in remission — with each patient's own remission samples as the comparison, and no patient cohort?

**Data.** GSE315367 (EPIC, raw IDATs): 14 AML patients; 10 with peripheral blood at diagnosis (Dx), first and second remission (Rm1, Rm2);
4 with bone marrow at Dx, Rm1, Rm2 and relapse (Rl). Our Stage 1 per array. Slide (Sentrix ID) recorded per array.
**Reference (healthy, independent lab, same platform).** Salas EPIC purified blood cells (GSE110554, GSE167998; our Stage 1), 91 arrays, 15 cell types.

**Reading (fixed now).**
- Sites: neutrophil identity sites (across-donor SD ≤ 0.05, mean β 0.75–0.95 or 0.05–0.25) that are SHARED: every other Salas cell type with
  ≥ 3 arrays has mean β within 0.05 of the neutrophil mean (a leukocyte reading led by neutrophils, insensitive to normal composition — v1.2).
- Met-A = mean per-site H(β) at those sites / floor; floor = the same on Salas neutrophils (leave-one-out for a Salas neutrophil).
  Normal = 0.95–1.05 (one gauge).
- C-score: per-site z = (H(β) − mean H of reference neutrophils)/shrunken SD; clustering = var of 50-consecutive-site block means × 50 / var(z), in
  genome order; C = clustering / median clustering of the healthy Salas arrays. **Healthy C line = 95th percentile of healthy C** (computed from Salas
  arrays only, printed before any AML array is read).

**Predictions (development record; no commissioning claim).**
- S1 (healthy precision): ≥ 90 % of the 91 Salas arrays read Normal on Met-A.
- S2 (diagnosis): blood Met-A outside Normal at Dx in ≥ 8 of the 10 blood patients.
- S3 (remission moves back): |A(Rm) − 1| < |A(Dx) − 1| in ≥ 9 of 10 blood patients, for Rm1 and for Rm2 separately.
- S4 (map): C(Dx) above the healthy C line in ≥ 8 of 10; C(Rm1) < C(Dx) in ≥ 9 of 10.
- S5 (same person, repeat reading): |A(Rm1) − A(Rm2)| ≤ 0.05 in ≥ 7 of 10.
Descriptive: the 4 marrow patients (does relapse move back out?); slide of every array; whether Rm readings themselves sit in Normal.

**Stated limits now.** No healthy arrays from this lab, so no same-slide tare; a within-patient change is reported with whether its arrays share a
slide. Remission blood follows chemotherapy (regenerating marrow, left shift), so Rm need not read Normal — S3 asks only that it moves back. AML Dx
blood is blast-dominated; the reading is of the leukocyte compartment, not of mature neutrophils only. 10 patients.
