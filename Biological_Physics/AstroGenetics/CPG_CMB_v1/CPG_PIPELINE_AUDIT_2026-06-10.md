# CPG Pipeline Chain-of-Custody Audit — 2026-06-10

**Auditor:** Walther (Claude) for Heath W. Mahaffey
**Reference:** `CPG Chain of Custody SOP/CPG_Chain_of_Custody_SOP_v1_3.md` (5,146 lines)
**Pipeline under audit:** `Walther_Clinical Python Script/walther_clinical.py` (846 lines) + runtime modules + report builder

## Verification-depth honesty note

This is the structured first pass. Each step below is tagged with an honest status:

- **VERIFIED** — I read the SOP text AND the implementing code and confirmed they agree (or disagree).
- **FIXED** — a confirmed defect, corrected this session.
- **GAP** — confirmed missing or stubbed in the pipeline.
- **NOT-YET-AUDITED** — the function exists but I have not yet read its body against the SOP step text line by line. NOT a statement that it is correct.

I am explicitly not claiming any NOT-YET-AUDITED step is correct. Completing those is the remaining work.

---

## Stage-by-stage status

### Stage 0 — Sample intake (L1) · §§11–19 · Steps 0.1–0.9
**GAP.** `stage_0_intake()` is a `_not_built` stub (line 437). None of: IDAT arrival (0.1), manifest creation (0.2), integrity hash (0.3), control-probe validation (0.4), detection-p QC (0.5), bead-count QC (0.6), call-rate (0.7), sex-check (0.8), decision gate (0.9) are implemented. The pipeline assumes a clean β series is handed in. **Entire stage outstanding.**

### Stage 1 — Calibration & β computation (L2+L3) · §§20–27 · Steps 1.1–1.8
**GAP.** `stage_1_calibration_beta()` is a `_not_built` stub (line 438). Dye-bias (1.1), probe-type normalization (1.2), ComBat (1.3), bisulfite-conversion check (1.4), β=M/(M+U+100) (1.5), sanity checks (1.6), probe response (1.7), β-matrix output (1.8) all unbuilt. `run_pipeline` takes `beta_calibrated` as input — Stage 1 is assumed done upstream. **Entire stage outstanding.**

### Stage 2 — Deconvolution (L4) · §§28–34 · Steps 2.1–2.7
**NOT-YET-AUDITED.** `stage_2_deconvolution()` (line 460) built; `_ensure_atlas_decompressed` present. Need to verify against SOP: atlas REBUILD load (2.1), marker-pool extraction (2.2), Walther NNLS (2.3), confidence/status codes (2.4), NILC v2 Path 2 (2.5), cross-method gate (2.6), output (2.7). **Confirm NILC Path 2 + cross-method gate actually run, not stubbed.**

### Stage 3 — Foreground subtraction (L4 secondary) · §§35–40 · Steps 3.1–3.6
**NOT-YET-AUDITED.** `stage_3_foreground_fork()` (line 137) built. SOP requires age (3.1), sex (3.2), batch/plate (3.3), ancestry (3.4), smoking (3.5) foreground layers. SOP §36/§39 say sex + smoking layer CSVs are built (`IAMAtlas_sex_layer.csv`, `IAMAtlas_smoking_layer.csv`). **Verify which layers are wired into the fork vs still threshold-stratified; confirm age layer (`IAMAtlas_age_layer.csv`, 8,199 CpGs) is applied.**

### Stage 4 — A-score computation · §§41–46 · Steps 4.1–4.6
**FIXED + VERIFIED (math) / GAP (substrate).**
- Steps 4.1–4.3 (β_mean → H(β_mean) → /H_min): the production `iamatlas_a_scoring.py` was computing `mean(H(βᵢ))/H_min` (average of per-CpG entropies). SOP §41–43 and Recipe 3.1/§6.2 specify `H(β_mean)/H_min` (entropy of the mean). **FIXED 2026-06-10** (commit a64418b). Verified: 177 healthy controls now land blood-resident classes at the floor (immune 0.993, secretory 0.986, cycling 0.992, progenitor 1.005); before fix they read 0.4–0.5.
- **Fail-safe added:** `test_a_score_canonical.py` with a bimodal guard (mean β=0.5 must score the ceiling 1/H_min, not 0.37) — the test the prior uniform self-test could not be. **Wire into startup/CI.**
- **GAP — substrate coverage.** Blood-absent classes (terminal 0.72, stromal 0.43, stem_pluri 0.15) read suppressed off a blood draw because their markers read the mixture, not the cell. SOP does not yet gate these. **Need a substrate-appropriate gate: mark not-assessable-in-blood rather than report as suppressed/inverted.**
- Step 4.4 per-cell-type A-score: same module, same fix applies. **Open question: should per-cell A-score run on deconvolution-reconstructed per-cell β rather than raw mixture β?** Unresolved.
- Step 4.5 disease-panel A-score: see Stage 4.5 below.

### Stage 4.5 — Bidirectional decomposition · §46.5
**NOT-YET-AUDITED.** `stage_4_5_bidirectional()` (line 531) built. SOP: mirrors VAL-051 a_dir_score; immune-class panel only (7 other classes return NO_PANEL honestly). **Verify sign convention + NO_PANEL handling; this ties to the still-open H.5b convergence sign-convention question.**

### Stage 4.6 — Patient brightness comparison · §46.6
**NOT-YET-AUDITED.** `stage_4_6_brightness()` (line 547) built; renders Personal Brilliance Maps. SOP: per-class z-departure on HEALPix NSIDE=128, 100% atlas coverage. **Verify the z-departure uses the corrected A-scale now that Stage 4 changed; the brightness maps may need regeneration.**

### Stage 5 — Multi-D departure (Mahalanobis) · §§47–51 · Steps 5.1–5.5
**NOT-YET-AUDITED — and RE-RUN REQUIRED.** `stage_5_mahalanobis()` (line 563) built; loads HC centroid. **The HC centroid (`mahalanobis_healthy_reference_v0_5.json`) was built on the OLD buggy A-scores. It must be rebuilt on corrected A-scores or every Mahalanobis distance is referenced to a wrong-scale centroid.** Highest-priority re-run.

### Stage 6 — Cellular age inversion · §§52–58 · Steps 6.1–6.7
**NOT-YET-AUDITED — and RE-RUN REQUIRED.** `stage_6_cellular_age()` (line 244) built. **The 80-cell age reference matrix was built on old A-scores → re-derive on corrected scale.** Note the report's D.1 "age barely moves the needle" rationale was argued in variance-relative terms; re-examine on the corrected anchored A-score.

### Stage 7 — Tier breakpoints · §§59–64 · Steps 7.1–7.6
**VERIFIED (breakpoints) / NOT-YET-AUDITED (cfDNA, language map).** `stage_7_tiers()` (line 575) loads `tier_breakpoints.json` via `gauge.load_tier_scheme` — the single source of truth. Breakpoints confirmed against the gauge: NORMAL 0.95–1.04, ELEVATED 1.04–1.07, Warburg 1.07, SIG_ELEVATED 1.07–1.10, BREACH ≥1.10. **These are correct — BUT they were being applied to wrong-scale A-scores, so all prior tier calls were wrong in practice. Now that Stage 4 is fixed, re-run tier calls.** cfDNA branch (7.3) and engine→customer language map (7.5) not yet audited.

### Stage 8 — Card-level pattern matching · §§65–69 · Steps 8.1–8.5
**NOT-YET-AUDITED.** `stage_8_dual_matching()` (line 712) built; `_matrix_match_magnitude`, `_matrix_customer_tier` (match-score thresholds 0.5/1.0/1.5/2.0 — a different quantity from A-score, not the gauge), `_build_customer_zshift_profile` (labeled "Cohen's d departure" — verify this is intended as the per-cell DIRECTION quantity and is computed on corrected A-scores). SOP §65 says matrix v1.5; production matrix is v1.8 — **version drift to reconcile.**

### Stage 9 — Report assembly · §§70–76 · Steps 9.1–9.7
**NOT-YET-AUDITED.** `stage_9_report()` (line 402) + `cpg_report_builder.py` built. Report reads `breach_line` from engine (correct) and shows per-cell A vs posterior 95% CI. **All displayed A-scores will change once the corrected scale propagates — report needs regeneration + visual re-check against the gauge. Confirm B.2/C tables now read ~1.0 not ~0.5.**

### Stage 10 — Delivery · §§77–79 · Steps 10.1–10.3
**GAP.** `stage_10_delivery()` is a `_not_built` stub (line 794). Report packaging (10.1), channel routing (10.2), audit-trail hash capture (10.3) all unimplemented. **Entire stage outstanding** — notably 10.3 (hash every step's output to repo for traceability) is the audit-discipline backbone and does not exist.

### L9 null suite · §§80–89
**NOT-YET-AUDITED.** 8-null framework + synthetic-patient generator referenced. Verify `cpg_null_runner.py` and `synthetic_patient_generator.py` exist and run.

---

## Outstanding items — prioritized

**P0 — correctness, blocks trustworthy output**
1. Rebuild the Mahalanobis HC centroid (Stage 5) on corrected A-scores — every distance currently references a wrong-scale centroid.
2. Re-derive the 80-cell age reference matrix (Stage 6) on corrected A-scores.
3. Re-run all tier calls (Stage 7) — breakpoints were correct but applied to wrong-scale A.
4. Add the substrate-coverage gate (Stage 4) so blood-absent classes are not reported as suppressed.
5. Regenerate the patient report (Stage 9) + Brilliance Maps (Stage 4.6) on corrected A; confirm B.2/C tables read ~1.0.
6. Resolve per-cell A-score β source (raw mixture vs deconvolution-reconstructed) — open.

**P1 — re-validation**
7. Re-check every VAL that used absolute A-scores against the corrected scale (relative case-vs-HC d/Mahalanobis partially cancel; absolute tier claims do not).
8. Rebuild this session's breast per-cell trajectory on corrected A-scores (the CSVs used were buggy output).
9. Re-examine the age question on the corrected anchored A-score, not Cohen's d.
10. Reconcile disease-matrix version (SOP cites v1.5; production v1.8).

**P2 — unbuilt stages**
11. Build Stage 0 (intake QC, Steps 0.1–0.9).
12. Build Stage 1 (calibration & β, Steps 1.1–1.8).
13. Build Stage 10 (delivery + audit-trail hashing, Steps 10.1–10.3).

**P3 — deep per-step audit still owed**
14. Line-by-line verification of Stages 2, 3, 4.5, 4.6, 5, 6, 8, 9 internals against their SOP §§ (currently NOT-YET-AUDITED).
15. Wire `test_a_score_canonical.py` into startup/CI as a hard gate.
16. Confirm L9 null suite + synthetic-patient generator are present and runnable.

---

## Process note
The A-score defect persisted because the module self-test used a uniform β where the wrong and right formulas coincide, and because the SOP was skimmed, not read. The fail-safe (bimodal guard) and the discipline of reading the canonical §§ before touching a stage are the two structural preventions. This audit is the start of honoring that, not the completion.
