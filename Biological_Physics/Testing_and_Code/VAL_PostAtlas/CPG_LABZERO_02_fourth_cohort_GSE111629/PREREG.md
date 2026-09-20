# PREREG — LAB-ZERO-02: a fourth lab decides whether the control probes can zero the array

**Sealed 2026-09-20 before any GSE111629 IDAT was fetched.** Owner H. W. Mahaffey · Analyst Claude Science.
**Why.** LAB-ZERO-01: control probes predict the per-lab offset's direction 3/3 and its size to 0.002 when a neighbouring lab is in training, but overshoot Munich 3× with none. Three labs cannot span the feature space. A fourth, from a different country, is the decisive test — and adds two things: (a) a lab-zero model trained on three labs predicting a fourth; (b) the first within-cohort test of per-array technical correction on band width.
**Cohort.** GSE111629 (UCLA, Horvath lab; Parkinson's whole-blood study, 450K, raw IDATs per sample): **only the 237 `PD-free control` arrays**; the 335 PD arrays are NOT opened. Ages 35–92 (median ~70), 126 M / 111 F, 219 Caucasian / 18 Hispanic. USA — a third country.
**Frozen.** Stage 1 noob; map `stage1_noob_450K`; identity loci v1_0; control-probe feature set (33) and ridge λ = 1 exactly as in LAB-ZERO-01; identity_band_v2 (pooled two-lab) as the width reference.
**Tests (fixed).**
- **P1** Stage 1 ≥ 95 %. **P2** presence gate ≥ 90 %.
- **P3 map transfer #3.** Median mapped immune A reported; the UCLA per-cohort constant vs Uppsala is the P5 quantity.
- **P4 — lab zero predicted from THREE labs.** Train ridge on Uppsala + Karolinska + Munich, predict UCLA. PASS if |predicted − observed cohort offset| ≤ 0.010. (LAB-ZERO-01's bar, one more lab.)
- **P4b — leave-one-out over four.** Report all four folds; the route is COMMISSIONED only if all four |err| ≤ 0.010.
- **P6 — within-cohort correction (first test).** Fit A ~ control features on the three training labs (sample level, cohort mean removed), apply to UCLA: report sd of immune A before/after within UCLA, and the fraction of UCLA in the v2 band before/after. A correction that narrows sd by ≥ 20 % without shifting the median by > 0.005 is a PASS for LAB-ZERO-02's narrowing claim.
- **N-plate** UCLA Sentrix chips F-test. **N-ethnicity** Hispanic vs Caucasian mapped A, signed d, reported (n = 18, descriptive only). **N-sex** signed d(f−m) per decade.
**Outcomes.** P4 + P4b pass → control-probe lab zero COMMISSIONED; the healthy-control panel becomes the fallback, not the standard. P4 pass / P4b fail → promising, fifth cohort. P4 fail → panel standard confirmed; control probes stay a narrowing tool only if P6 passes. Written as found.

---
**SEALED** sha256 `b26113e28be621b8927495b60509f00ad175147c5e95529e64c8961104586e90` · 2026-09-20
