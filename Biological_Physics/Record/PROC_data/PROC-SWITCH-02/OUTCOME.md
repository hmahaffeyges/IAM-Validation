# OUTCOME — PROC-SWITCH-02: the gauge switch commissioned (CHAIN_COMMISSIONING row B)

**Run 2026-09-21.** 80 synthetic healthy whole-blood patients (WHOLE_BLOOD_ALPHA, no disease, seed 2027, ages 20–80); patients 0–39 the PANEL, 40–79 the TEST, disjoint.

| test | bar | result | verdict |
|---|---|---|---|
| S4a the atlas's own zero | report (expected ≈ −0.015) | **z_atlas = −0.0146** from `lab_zero.compute_lab_zero` on the 40-panel | measured |
| S4b the rest reads 1.00 | median within ±0.010, ≥ 80 % in band | **median A″ 1.0009** [0.985–1.019], **100 % in identity_band_v3** | **PASS** |
| S4c the defect is gone | switched in-band − marker-union in-band ≥ 0.5 | marker-union median **1.125, 0 % in band**; switched 100 % → **+1.00** | **PASS** |
| S4d UNSET refuses | 80/80 | 80/80 reportable = False without a zero | **PASS** |

With PROC-SWITCH-01's S1, S2, S3, S5: **row B COMMISSIONED.** The reported class A is the identity-loci gauge, on mapped β, age-referenced, lab-zeroed, placed in identity_band_v3; the marker-union statistic is `diagnostic_marker_union` and is never the reported A; Stages 5 and 6 carry `pending_recalibration=True` until their rows are run.

**Finding recorded (RECON B5): the atlas is a fifth laboratory.** The G-002 floor (A = 1 at β = 0.7318, 37 reference cells) and the IAMAtlas REBUILD posterior (immune identity-loci mean 0.7373) are different reference sets. A synthetic patient built from atlas means reads 0.985 with zero 0 and 1.001 with the atlas's own panel zero (−0.0146). Real Uppsala healthy blood reads 0.988 mapped. Consistent. Consequence: any synthetic-patient test must zero the synthetic cohort like a laboratory; SWITCH-01 S4 did not and failed as sealed.

**Lesson (RUNBOOK):** every β source is a laboratory — including the atlas, including a simulator. Before reading one absolutely, ask what its zero is.

---
**SEALED** sha256 `babdc91b03c64725eae58b1ffd67ab78a26068b1a1857f87973e5a69a573dedb` · 2026-09-21
