# PREREG — PHASE 1: identity-loci healthy bands for whole blood from one cohort through one pipeline

**Written and sealed:** 2026-09-19, before any methylation value from GSE87571 was read (the RAW tar was 1.38 GB of ~6.1 GB downloaded, unopened, at sealing).
**Owner:** Heath W. Mahaffey · **Analyst:** Claude Science
**Why this exists:** PROC-N7-01 showed the production class gauge reads H(β̄) over the class marker union and its band (`age_reference_matrix.json`) was compiled the same way; the correct statistic — H(β̄)/H_min over the class **identity loci** — has no band. PROC-CHAIN-01 / PROC-NILC-01 / PROC-SEP-02 showed stem_adult's fraction in blood is not data-determined. This procedure builds the missing band, on the correct statistic, from one healthy cohort through the project's own Stage 1, and applies the reporting rule (SOP §108) in advance.

## 1. Cohort
GSE87571 — healthy whole blood, Illumina 450K, n ≈ 732, ages 14–94, both sexes, one laboratory, raw IDATs. Chosen because it is the largest single-site healthy whole-blood IDAT cohort on GEO with age on every sample. **No disease arm.** Nothing here is a disease test.

## 2. Pipeline — frozen before this test
- Stage 0 intake (`stage_0_intake.py`) with the PROC-STAGE0-01 fail-closed guard; Stage 1 `calibrate_idat_to_beta` (methylprep noob) exactly as in PROC-CAL-01.
- Deconvolution: `WaltherIAMDeconvolver` at repo HEAD, **contrast_pairs = None** (default; PROC-SEP-02's option stays off).
- Gauge statistic: **A = H(β̄)/H_min over the class identity loci** (`iamatlas_gauge_identity_loci_v1_0.json`), H_min per class from the same file. Not the marker union. Not mean-of-H.
- No value from GSE87571 enters any constant; H_min and the identity loci are frozen inputs.

## 3. Reporting rule applied in advance (SOP §108) — decided from κ and fraction, not from A
| class | status on whole blood | basis |
|---|---|---|
| immune | **band built, gauge reported** | present ≥ 80 %; κ of its own column fine |
| progenitor + stem_adult | **one joint haematopoietic-progenitor component**; band built on the joint β̄ over the union of both identity-loci sets with progenitor's H_min (0.852200; the two floors differ by 0.0215 bits and the fraction-weighted joint floor is 0.8576 — within 0.006) | κ(blood sub-problem) 30.6 → 17.0 with contrast markers, never < 10 (PROC-SEP-02) |
| cycling, secretory, terminal, stromal, stem_pluri | **composition-only; no band, no gauge** | fraction < 5 % in blood |

## 4. Band definition (author's decisions 2026-09-19)
- Per class, per **decade of age** (14–24, 25–34, …, 85–94): the **10th and 90th percentiles** of healthy A, plus the median. Stated in percentage language on the report ("your reading sits at the 62nd percentile of healthy people your age"). No mean ± SD anywhere on the patient-facing output.
- Minimum n per decade cell to publish a band: **30**. Cells below 30 are marked `THIN` and interpolated from neighbours with the interpolation flagged.
- Sex is recorded and tested (N-sex below) but the band is **not** split by sex unless N-sex fails.

## 5. Pass conditions (all decided now)
- **P1 — coverage.** ≥ 95 % of samples pass Stage 0 (array type HM450K verified from header; integrity hash). Fewer → the cohort is flagged and the reason recorded before proceeding.
- **P2 — presence.** Deconvolved immune fraction ≥ 0.80 and epithelial classes (cycling + secretory + terminal + stromal) ≤ 0.02 in ≥ 95 % of samples. A sample outside this is excluded from the band with its ID recorded; > 5 % outside → FAIL (the cohort is not the substrate we think it is).
- **P3 — the real test: the synthetic healthy patient reads IN_BAND.** The 16 whole-blood synthetic healthy patients of PROC-N7-01 (pure mixtures of healthy Atlas posteriors; `WHOLE_BLOOD_ALPHA`) must read **within the new 10–90 % band** on the immune identity-loci gauge in ≥ 14 of 16. This is the pass condition that the marker-union band could not meet.
- **P4 — the eleven test samples.** The seven Stage-1-calibrated healthy/RA-control whole-blood test IDATs read IN_BAND on immune in ≥ 6 of 7; the four tissue samples read **outside** the blood band or are flagged off-substrate by presence (epithelial > 0.02) — either is correct; reading in-band silently is FAIL.
- **P5 — age trend.** Spearman ρ between immune A and age reported with its sign; no pass condition attached (this is measurement, not a hypothesis).

## 6. Nulls
- **N-sex.** Immune A by sex within decade: |d| < 0.2 in every decade with n ≥ 30 → single band. Otherwise split.
- **N-split.** Random half-split of the cohort; band from each half; the 10th/90th percentiles agree within 0.01 A in every decade cell with n ≥ 30. Disagreement → widen the band to the union and record it.
- **N-plate.** If Sentrix ID / position is recoverable from the IDAT filenames, immune A by plate: no plate with |d| > 0.3 vs the rest. If not recoverable, recorded as SKIPPED with the reason (this is the null the breast VALs could never run).
- **N-random.** A band built the same way over 42,134 random CpGs matched on mean β. The synthetic healthy patients (P3) must NOT read in-band on the random panel in ≥ 14 of 16 — i.e. P3 must depend on the identity loci being the identity loci.

## 7. What this does and does not decide
- Decides: whether the identity-loci gauge, read against a band from one cohort through one pipeline, places healthy blood where healthy blood should be. Nothing else.
- Does not decide: anything about disease, any tier boundary, any H_min, whether stem_adult exists as a separate signal (it does not, here, by rule).
- If P3 fails, the identity-loci gauge as specified does not describe healthy blood and the finding is recorded as FAIL, not reinterpreted.

## 8. Order of operations
Stage 0 → Stage 1 on all IDATs → deconvolution → P1, P2 → immune and joint A → bands → N-split, N-sex, N-plate → P3, P4 (synthetic and test samples scored against the sealed band) → N-random → P5 → OUTCOME sealed. Age is loaded for banding only; no other metadata enters before P3.

---
**SEALED** sha256 `4d8f40ace72eca1f178cae63f2c55e874fcda07328e8c03a33fcb298a125e09c` · 2026-09-19 · GSE87571_RAW.tar unopened (1.38 GB of ~6.1 GB downloaded at sealing)
