# DEV-NEWLAB-GRAN-01 — a new laboratory's healthy granulocytes: Met-A and the tared C-score band (development, 2026-10-09)

**DEVELOPMENT - not commissioned.** Checks written before any array below was read.

**Why.** Bar 10 of the Met-A draft note (the tared C-score band 0.751-1.409 holds on laboratories not used to set it) could not be assessed:
every held-out healthy series in job B is a specimen the chain refuses. GSE226298 (EPIC v1, Alzheimer's disease study, one laboratory not in
any earlier set) deposited purified **granulocytes** from 26 healthy controls and 39 patients, with IDATs.

**Scope note.** Granulocytes are mostly neutrophils with a few per cent eosinophils and basophils. Stage 0 accepts isolated neutrophils, not
granulocytes, so the arrays are read here as a development reading with the isolated-neutrophil path (`--specimen "isolated neutrophils"`
on a copy; intake rules unchanged). The healthy controls are the test; the patients are read and reported but not scored against a bar.

**Run.** Stage 1 on the IDATs, then `conductor_v3.run_neutrophil` with same-run references = the other healthy-control granulocyte arrays
of the same slide (else the series), ≥ 3, the array itself excluded: the Stage T rule.

**Checks (healthy controls only).**
1. Tared Met-A: ≥ 95 % in Normal (0.95-1.05).
2. Tared C-score C_rel: ≥ 95 % inside 0.751-1.409 (the band set in DEV-CSCORE-TARE-01 on 19 other series).
3. Noise gate: share withheld recorded (not a bar).

---
## Results (2026-10-09, box job 6b428b95; nothing above the line changed)
65 granulocyte arrays read, 0 errors (26 healthy controls, 39 patients). Per-array table: `data/DEV_NEWLAB_GRAN_01/gran_rows.csv`.

| Check (healthy controls) | Result | Met |
|---|---|---|
| 1. Tared Met-A ≥ 95 % Normal | **26 / 26** Normal; median 1.001, 2.5-97.5 % 0.974-1.030 | yes |
| 2. Tared C-score inside 0.751-1.409 ≥ 95 % | **23 / 26 (88.5 %)**; median 1.031, 2.5-97.5 % 0.699-1.295. Untared C: 13 / 26 | **no** |
| 3. Noise gate withheld | 0 / 26 | recorded |

- \measured Met-A reads a new laboratory's healthy granulocytes Normal, every one, with the tare set on that laboratory's own healthy arrays.
- \measured The same-run tare doubles the share of the C-score inside the band (13 → 23 of 26) but does not reach 95 %. All three outside
  sit LOW (0.63, 0.74, 0.74), i.e. less clustered than their references, not more; two were tared at series level (fewer than 3 healthy
  references on their slide).
- Patients (not scored): 37 of 38 tared Normal; C_rel inside the band 32 / 38.
- Scope: granulocytes read on the isolated-neutrophil path (a few per cent eosinophils and basophils); intake unchanged.
