# PROC-AML-SERIAL-01 — outcome (2026-10-01; pre-registration sha 6fe8dc63, unchanged)

| bar | result | |
|---|---|---|
| S1 healthy Salas arrays Normal ≥ 90 % | 60.4 % (55/91); neutrophils 10/12 | FAIL |
| S2 Dx blood outside Normal ≥ 8/10 | 4/10 | FAIL |
| S3 Rm closer to 1 than Dx ≥ 9/10 | Rm1 7/10, Rm2 8/10 | FAIL |
| S4 C(Dx) > healthy line 1.452 ≥ 8/10; C(Rm1) < C(Dx) ≥ 9/10 | 3/10; 3/10 | FAIL |
| S5 same person, |A(Rm1) − A(Rm2)| ≤ 0.05 ≥ 7/10 | 10/10 | PASS |
Descriptive: remission blood Normal Rm1 8/10, Rm2 10/10. Marrow (4 patients) reads 1.06–1.38 at every time point, remission included (no
healthy-marrow reference; marrow is not blood). Patients 12 and 13: C(Dx) 1.91 / 2.21 → remission 0.87 / 0.82 (the only large map changes).

**Why (diagnosis).**
1. *S1 was the wrong healthy test.* The shared sites keep every mature cell within 0.05 β of the neutrophil mean, but near β 0.05–0.25 / 0.75–0.95
   a 0.05 β difference moves H by up to ~15 %: purified memory CD4 T read 1.13, CD8 T 0.81. Whole blood (50–70 % neutrophils) averages these;
   single purified non-neutrophil arrays do not. Healthy neutrophils: 10/12 Normal (0.960–1.059).
2. *The sites were chosen to be insensitive to composition, and that also makes them insensitive to blasts.* Sites on which every leukocyte agrees
   are lineage-wide; AML blasts largely share them, so diagnosis blood stays near 1.
3. *The healthy C line mixes cell types* whose own clustering differs (neutrophils 0.70, naive CD4 1.54).
4. *Wrong floor for the dominant cell at diagnosis:* AML blood is blast-dominated, and Met-A reads a cell against its OWN floor. A blast is a
   progenitor; reading it against the neutrophil floor is not the gauge's definition.

**What it decides (development).** The repeat reading of one person is stable (S5, 10/10). The neutrophil-led shared-site reading does not see AML.
Next: (a) let the composition solver name the dominant cell, and read it against THAT cell's floor (HSC/progenitor floors exist in atlas v2,
GSE63409); (b) build the healthy C line from healthy whole blood, not purified cells; (c) report blast/progenitor fraction beside A.
