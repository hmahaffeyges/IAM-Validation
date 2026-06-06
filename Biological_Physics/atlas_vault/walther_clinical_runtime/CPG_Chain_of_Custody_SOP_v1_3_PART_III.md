# CPG Chain-of-Custody SOP — Part III (L9 audit machinery) (v1.2 — walkthrough aligned)

> **v1.1 note (2026-06-02).** v1 contained fabricated `cpg_engine/...` module paths
> that did not correspond to real files in the IAM-Validation repo. This v1.1 pass
> stripped them across all parts; this particular part (Foundations / L9 machinery)
> was already clean of fabricated paths in v1, so the disclaimer is included here
> only for version-consistency across the SOP. All real paths in v1.1 are documented
> in `SYSTEM_INVENTORY.md`; any path NOT in that inventory should be treated as
> not-yet-existing until verified against the repo.

---


**Continues from Part II-C (§65-§79). This document contains §80-§91 — the L9 null suite, the synthetic patient generator (FFP10/NPIPE analog), and the VAL sealing protocol.**

---

# Part III — Chain-integrity scaffolding (runs ABOVE, not inside, the chain)

L9 is structurally different from L1-L8. The first eight chain links run **inside** the operational chain — every patient's IDAT passes through L1, L2, L3, ..., L8 in sequence. L9 does not. L9 runs **above** the chain, on cohorts after they complete Stages 0-10, and tests whether the headline result from any sealed VAL would survive deliberate adversarial perturbation.

The discipline borrowed directly from CMB cosmology: **no result ships until it has been beaten through nulls.** Planck does not publish a power-spectrum measurement without first running the null suite (jackknife rotations, frequency-difference tests, FFP10 end-to-end sims). CPG does not seal a CPG-VAL without first running the 8-null suite. Sealed = passed all declared nulls. Restated = failed one or more declared nulls; the result is preserved with the limitation declared honestly. Retracted = failed in a way that invalidates the result entirely; the result is withdrawn.

The eight nulls (N1-N8) are pre-specified. Each VAL declares which subset applies (every VAL declares N1, N7, N8 at minimum; cohorts with age/sex/plate metadata add N2, N3, N5; multi-cohort VALs add N4; signal-direction-dependent VALs add N6). The declaration is in the VAL's PREREG document; the verdict is in the VAL's OUTCOME document.

Phase A (closed 2026-05-30) built the L9 framework, ran all 7 existing Family A CPG-VALs through it, and produced the first sealed-vs-restated split: 5 sealed (VAL-001, VAL-002, VAL-003, VAL-005, VAL-007), 2 restated (VAL-004 bimodality FAIL-on-direction, VAL-006 chr6 MHC FAIL-on-look-elsewhere). The framework caught these before any clinical use — that is what the discipline is for.

The L9 grade lifted from D (no null framework) to A- (framework live, 5 of 7 sealed, 2 known limitations Phase A2.1 closes). The remaining grade weight is in the production-precision N7 recovery upgrade (Phase A2.1) and signed-direction injection (Phase A2.1).

---

## §80. L9.0 — The 8-null suite framework (`cpg_null_runner.py`)

**What this step does.** The L9 framework is the orchestration layer that runs the eight standard nulls against any sealed VAL artifact. A VAL provides its `per_sample.csv` (per-patient reconstructed quantities), its `results.json` (declared headline + declared nulls), and its PREREG (declared statistical test). The framework runs each declared null, returns PASS/FAIL + p-value + 95% CI, and produces a `null_results.json` artifact that gets bundled with the VAL.

**Inputs.**
- `per_sample.csv` — per-patient outputs from the chain (A-scores, Mahalanobis distance, cellular age, card verdict, ...).
- `results.json` — the VAL's declared headline statistic + the declared nulls it must pass.
- `prereg.json` — pre-registration document declaring the statistical test, the cohort, the significance threshold, the multiple-testing correction approach.

**Atlas reference.** None directly. The L9 framework operates on chain outputs, not atlas reference. The IAMAtlas was consulted at Stage 2, 4, 5, 6 of the chain; by L9 time, it has done its work and is no longer needed.

**Files invoked.**
- `Biological_Physics/chain_of_custody/L9_null_suite/cpg_null_runner.py` — the orchestration module.
- Per-null modules: `cpg_null_runner.py` implements N1 through N8 as separate functions; each is callable independently for VAL-specific subsetting.
- `Biological_Physics/chain_of_custody/L9_null_suite/synthetic_patient_generator.py` — N6 and N7 invoke this.

**The math.** None directly at the framework level. Each null has its own math (§81-§88). The framework is dispatch logic + multiple-testing aggregation + sealing-verdict computation.

**Sealing verdict logic.** Given the VAL's declared null set `D ⊆ {N1, N2, ..., N8}` and the per-null verdicts `V_n ∈ {PASS, FAIL}` for n ∈ D:
- **SEALED**: All declared nulls PASS. Headline result preserved.
- **RESTATE**: One declared null FAILs; the FAIL is documented and the headline is preserved with explicit caveat. (Used when the failure is a documented limitation — e.g., VAL-004 cohort-pooled bimodality — that does not invalidate the underlying signal but limits its scope.)
- **RETRACT**: Two or more declared nulls FAIL; the headline is withdrawn. (Not yet triggered in Phase A; the framework expects this verdict to be rare but real.)

**CMB equivalent.** Planck's **null-test orchestration framework** in NPIPE: it runs the full null-test battery (TT-TE-EE-BB cross-tests, half-mission-difference, year-difference, detector-set-difference, scanning-direction-difference, FFP10 end-to-end recovery) against every cosmological-parameter release. A release does not ship until every null test passes the declared significance threshold (typically 3σ). The CPG L9 framework is the methylome implementation of that orchestration discipline — one module that runs all declared nulls, returns one verdict per null, aggregates into a SEALED/RESTATE/RETRACT decision.

**How the methylome differs in implementation.** Planck's null suite is the cosmology community's accumulated wisdom over 30 years; the CPG null suite is in its first generation (Phase A 2026-05-30). The structural analog is exact; the operational maturity differs by decades. Phase A2.1 is the immediate maturity upgrade (production-precision N7, signed injection).

**How it's the same in principle.** A measurement that has not been beaten through nulls is not yet a measurement — it is a candidate measurement. The null suite is what converts a candidate into a sealed claim.

**Outputs.** Per-VAL `null_results.json` carrying:
- The declared null set.
- The per-null verdict + p-value + 95% CI.
- The aggregated SEALED/RESTATE/RETRACT verdict.
- The framework version + invocation timestamp + module SHA-256 hashes (for chain-of-custody on the L9 audit itself).

**Decision points.** SEALED → VAL is sealed; result enters production. RESTATE → VAL is preserved with declared limitation. RETRACT → result withdrawn; rerun chain with corrected upstream or accept the negative finding.

**Failure modes.**
- **Framework dispatch failure** (a null module crashes during invocation). Caught by per-null try/except; the framework returns FAIL with a `framework_error` flag so the verdict cannot be silently turned into PASS.
- **Declared null not implemented** (a VAL declares N5 but no plate metadata exists). Returns INSUFFICIENT_METADATA for that null; the framework aggregates to RESTATE rather than SEALED because the declared null couldn't run.

**Canonical cross-references.** Recipe §11 (null discipline). Roadmap §10.2.1 Phase A. §12 of Roadmap (Phase A results).

**CPG Plate references.** **Plate 4 Panel E** (MCMC Coverage Map) shows the stromal galactic mask — the systematic that the L9 framework formalizes as a declared limitation rather than a hidden contamination. The chain-of-custody discipline lives in declaring the mask, not in pretending it isn't there.

**Chain-link assignment.** L9 (the framework itself).

---

## §81. L9.1 — N1: HC-permutation null (case/HC label shuffle)

**What this null tests.** The most fundamental: would the VAL's headline statistic appear by chance if you shuffled the case-vs-HC labels among the patients? If yes, the signal is a label artifact, not biology.

**Inputs.** `per_sample.csv` with patients labeled as case or HC. The VAL's declared statistic (e.g., Cohen's d on immune A-score for breast-pre-dx >10y).

**Atlas reference.** None.

**Files invoked.** `cpg_null_runner.py::run_N1_label_permutation()`.

**The math.** Permute the case/HC label vector 1,000 times (configurable, default 1,000). Per permutation: recompute the declared statistic on the permuted labels. Per VAL: the null distribution is the 1,000 permuted statistic values. The two-sided p-value is the fraction of permuted statistics with absolute value ≥ |observed statistic|. The 95% CI is the [2.5th, 97.5th] percentile of the permuted distribution centered on zero.

**PASS criterion.** Observed statistic outside the 95% CI of the permuted distribution. Equivalently, p < 0.05 (or whatever the PREREG declared).

**FAIL criterion.** Observed statistic within the 95% CI of the permuted distribution.

**CMB equivalent.** **Sky rotation null** in CMB analysis: take the cosmological signal map, rotate it 90° / 180° / 270° relative to the foreground masks, recompute the power spectrum. The cosmological signal should survive the rotation (it's isotropic); a foreground residual should look very different after rotation. The CPG label-permutation test is the case/HC analog: real biological signal should depend on label assignment (it's "the property that defines cases"); statistical noise should not.

**How the methylome differs in implementation.** Permutation is on a discrete label vector rather than a continuous rotation. The mathematical structure of the null is identical: shuffle the "is this the signal-bearing axis?" assignment and check whether the headline statistic survives.

**How it's the same in principle.** Both ask: is the headline statistic explained by the property the analyst calls signal-bearing, or would any random property of the data produce the same statistic? If the latter, the headline is an artifact.

**Outputs.** `N1_verdict ∈ {PASS, FAIL}`, `N1_pvalue`, `N1_95CI`, `N1_permutation_distribution.json` (the full distribution, for downstream verification).

**Decision points.** PASS → contribute to SEALED. FAIL → contribute to RESTATE/RETRACT.

**Failure modes.**
- **Insufficient permutation budget.** Default 1,000; for borderline VALs, framework can run 10,000 or 100,000. The framework declares the budget per VAL in `prereg.json`.
- **Permutation distribution non-Gaussian.** The framework reports the empirical percentile rather than assuming Gaussian — this is robust to non-Gaussian null distributions, which DO occur in methylation data (the distributions are often skewed).

**Canonical cross-references.** Recipe §11.1. Roadmap §10.2.1 Phase A1 (N1 in the 8-null specification). §12 of Roadmap (Phase A results: 7 of 7 VALs ran N1).

**CPG Plate references.** None directly; the per-VAL permutation results live in the audit-trail artifacts.

**Chain-link assignment.** L9.

---

## §82. L9.2 — N2: Age-decade-stratified permutation

**What this null tests.** The label-permutation test (N1) protects against random label-shuffling artifacts. N2 protects against a sneakier confound: what if cases are simply older than HC, and the "signal" is just an age-drift artifact dressed up as a disease signal? N2 permutes labels WITHIN age decades — so a case can only become a synthetic-HC by being swapped with an actual HC of the same age. This breaks the case/HC association while preserving the age structure.

**Inputs.** `per_sample.csv` with case/HC labels AND patient ages.

**Atlas reference.** None.

**Files invoked.** `cpg_null_runner.py::run_N2_age_stratified_permutation()`.

**The math.** Group patients by age decade. Per permutation: within each decade independently, shuffle the case/HC labels. Concatenate the per-decade permuted labels into a full permuted label vector. Recompute the statistic. Repeat 1,000 times. Apply the same PASS/FAIL logic as N1.

**Why this matters.** Age-drift in methylation is well-documented and substantial (the Horvath, Hannum, PhenoAge clocks all exploit it). If a VAL's headline is "cases differ from HC at p<0.001 on immune A-score" but the cases are systematically 5 years older than HC, the N1 permutation would still pass (label shuffling destroys the case/HC structure), but N2 would FAIL (within-decade shuffling preserves age, eliminating the age-drift confound, and the signal collapses).

**CMB equivalent.** **Stratified jackknife resampling** in CMB cross-correlation analyses — when measuring CMB-vs-galaxy-survey cross-correlation, the jackknife is performed within redshift slices to ensure the cross-correlation is not driven by redshift-dependent foregrounds. The methylome's age-stratified permutation is the same operation in a different parameter space.

**How the methylome differs in implementation.** Age decades rather than redshift slices. Same stratified-resampling principle.

**How it's the same in principle.** Both protect headline statistics from being driven by an unmodeled covariate that correlates with both the "signal" axis and a potential confounder.

**Outputs.** `N2_verdict`, `N2_pvalue`, `N2_95CI`, `N2_within_decade_distribution.json`.

**Decision points.** Same as N1.

**Failure modes.**
- **Age metadata missing.** Returns `INSUFFICIENT_METADATA`. Framework aggregates this as a missing-but-required null → contributes to RESTATE if N2 was declared.
- **Sparse age decades.** If a decade has only 1-2 patients in case or HC, permutation within that decade is degenerate. Framework either pools that decade with the adjacent decade or excludes it from N2; the choice is per-VAL declared in PREREG.

**Canonical cross-references.** Recipe §11.2. Roadmap §10.2.1 Phase A1 (N2 in the spec).

**CPG Plate references.** None.

**Chain-link assignment.** L9.

---

## §83. L9.3 — N3: Sex-stratified permutation

**What this null tests.** Analogous to N2 but stratifying on sex rather than age. Protects against the headline being driven by sex differences in methylation (which exist at thousands of autosomal CpGs, not just chrX/chrY) that happen to correlate with disease-status assignment.

**Inputs.** `per_sample.csv` with case/HC labels AND patient sex.

**Atlas reference.** None.

**Files invoked.** `cpg_null_runner.py::run_N3_sex_stratified_permutation()`.

**The math.** Same as N2 with sex as the stratifying variable instead of age. Patients are grouped by sex; per permutation, case/HC labels are shuffled within each sex independently.

**Why this matters.** Many cohort designs are sex-skewed (the breast cancer VALs are mostly female; prostate VALs are exclusively male; some lifestyle cohorts skew female because women enroll more). If the VAL's headline is "cases differ from HC" but the cases happen to be 90% female and the HC 60% female, sex-skewed methylation differences contaminate the signal. N3 catches this.

**CMB equivalent.** **Sub-mission-split null tests** (Planck's half-mission-1 vs half-mission-2 cross-check). The mission is split along a property that should not affect the cosmological signal (which half of the observing schedule); if the headline differs across the split, it's a systematic.

**How the methylome differs in implementation.** Binary sex split rather than continuous mission timeline. Same structural principle.

**How it's the same in principle.** Both protect headlines from being driven by a binary structural property of the cohort that correlates with both the "signal" axis and a potential confounder.

**Outputs.** `N3_verdict`, `N3_pvalue`, `N3_95CI`.

**Decision points.** Same as N1.

**Failure modes.**
- **Sex metadata missing or sex-balanced cohort.** If cohort is too sex-imbalanced to stratify (e.g., all-female cohort), returns `INSUFFICIENT_VARIATION`. Framework treats this as N3 not applicable; not a contribution to RESTATE because the null genuinely cannot run.

**Canonical cross-references.** Recipe §11.3. Roadmap §10.2.1 Phase A1.

**CPG Plate references.** None.

**Chain-link assignment.** L9.

---

## §84. L9.4 — N4: Cohort-split replication

**What this null tests.** The headline statistic should replicate when the cohort is split in half — both halves should show the same effect direction and roughly comparable magnitude. A statistic that only appears in one half (or appears with opposite sign in the two halves) is a cohort-specific artifact, not a generalizable signal.

**Inputs.** `per_sample.csv` with a cohort-split variable (typically random 50/50 split within the cohort, OR within-batch split when batches differ in collection date/site).

**Atlas reference.** None.

**Files invoked.** `cpg_null_runner.py::run_N4_cohort_split_replication()`.

**The math.** Random 50/50 split of the patient set, stratified on case/HC and (when available) age/sex. Compute the headline statistic separately on Half-A and Half-B. Verdict:
- **PASS**: Same sign in both halves; magnitudes overlap at 95% CI.
- **FAIL_SIGN**: Opposite signs in the two halves.
- **FAIL_MAGNITUDE**: Same sign but magnitudes outside 95% CI of each other (large variance, weak signal).

The split is repeated 100 times (different random seeds); the framework reports the fraction of splits that PASS, FAIL_SIGN, FAIL_MAGNITUDE. Default PASS criterion: ≥ 80% of splits PASS.

**Why this matters.** A signal that only appears in one half-cohort and not the other is not a sealed result — it's a candidate finding that requires independent replication. CPG's N4 self-replicates within the cohort; external replication (different lab, different intake, different cohort entirely) is a higher bar that lives outside the L9 framework as the **Family B** validation discipline.

**CMB equivalent.** **Half-mission-difference null** in Planck. Planck-2018 ran the full likelihood on half-mission-1 and half-mission-2 separately, verified that cosmological parameters agreed to within their uncertainties. A parameter that differed by >2σ between halves was flagged for systematic investigation. CPG's N4 is the methylome equivalent.

**How the methylome differs in implementation.** Random patient split rather than temporal mission split. Same structural test.

**How it's the same in principle.** Both ask: does the signal exist independently in two sub-samples, or only in their aggregate?

**Outputs.** `N4_verdict ∈ {PASS, FAIL_SIGN, FAIL_MAGNITUDE}`, `N4_pass_rate_across_seeds`, per-split Half-A and Half-B statistics.

**Decision points.** PASS → SEALED contribution. FAIL_SIGN → strong RETRACT contribution. FAIL_MAGNITUDE → RESTATE contribution.

**Failure modes.**
- **Cohort too small to split** (<40 cases or <40 HC per half). Returns `INSUFFICIENT_SAMPLE`. Not a FAIL; null couldn't run. Framework treats as INSUFFICIENT_METADATA.
- **Within-batch split unavailable** (single-batch cohort). Falls back to random 50/50 split; per-VAL declared in PREREG which split mode is used.

**Canonical cross-references.** Recipe §11.4. Roadmap §10.2.1 Phase A1.

**CPG Plate references.** None.

**Chain-link assignment.** L9.

---

## §85. L9.5 — N5: Plate/array-position null

**What this null tests.** Illumina methylation arrays come in 8-sample chips arranged 2×4 on a 96-well plate. Position on the chip and plate can introduce systematic effects (edge effects, gradient effects, scan-order effects). N5 tests whether the headline statistic survives when patients are stratified by plate position — if the statistic appears mostly in patients at one chip position and disappears at others, it's a position artifact.

**Inputs.** `per_sample.csv` with case/HC labels AND chip position metadata (Sentrix barcode + row/column position).

**Atlas reference.** None.

**Files invoked.** `cpg_null_runner.py::run_N5_plate_position_null()`.

**The math.** Group patients by chip row × column position (8 position bins). Per position bin: compute the headline statistic restricted to that bin. Test: is the statistic consistent across positions, or does it vary systematically?

Two sub-tests run jointly:
1. **Position-stratified permutation** (analogous to N2): permute case/HC labels within each position bin, recompute statistic, build null distribution.
2. **Position-amplitude regression**: regress per-patient signal strength against position; positive slope coefficients indicate position-driven signal.

**PASS criterion.** Both sub-tests pass at p<0.05.

**Why this matters.** Plate effects are real and well-documented in methylation. The strongest are edge effects (chip rows 1 and 8 sometimes show different hybridization efficiency than rows 4-5), batch-position correlations (cases run in batch 1, HC in batch 2), and scanner-drift effects (early-day samples vs late-day samples). N5 is the systematic check that catches these before they're called signal.

**CMB equivalent.** **Scan-direction null** in CMB analysis. CMB experiments scan the sky in specific patterns (Planck's rings, ACT's surveys); scan-direction systematics (asymmetric beam, 1/f noise, hwp synchronous) appear as differences in the headline statistic restricted to specific scan directions. The CPG plate-position null is the methylome implementation.

**How the methylome differs in implementation.** Discrete position bins on a chip rather than continuous scan azimuth. Same structural principle.

**How it's the same in principle.** Both test whether the headline signal varies systematically across a property of the measurement apparatus that should not affect biological signal.

**Outputs.** `N5_verdict`, `N5_pvalue`, `N5_position_stratified_distribution`, `N5_position_regression_slope`.

**Decision points.** PASS → SEALED contribution. FAIL → RESTATE contribution.

**Failure modes.**
- **Position metadata missing.** Returns `INSUFFICIENT_METADATA`. Framework reports as N5-not-runnable; if N5 was declared, contributes to RESTATE.
- **Single-plate cohort.** N5 reduces to within-plate position-only; cross-plate effects can't be tested.

**Canonical cross-references.** Recipe §11.5. Roadmap §10.2.1 Phase A1.

**CPG Plate references.** None directly. (The methylome sky-map plates 1-4 use HEALPix projection, which is orthogonal to physical-chip position.)

**Chain-link assignment.** L9.

---

## §86. L9.6 — N6: Injection-recovery null

**What this null tests.** Inject a known synthetic signal into a known-HC patient population at known strength. Run the full chain. Verify the chain recovers the injected signal at the injected strength and direction. This is the **canonical "does the chain work?" test**.

**Inputs.** A HC-only β matrix from the cohort. A declared injection signal (effect size, direction, target class). The full Stages 0-10 chain.

**Atlas reference.** Indirect — the injection signal is built relative to the IAMAtlas reference posterior so the chain treats it as a biological signal rather than a numerical artifact.

**Files invoked.** `cpg_null_runner.py::run_N6_injection_recovery()`. Calls `synthetic_patient_generator.py` to construct the injected β matrix.

**The math.**
1. Start with a HC-only β matrix (n=N_HC patients).
2. Sample a target subset of M patients (M ≤ N_HC/2).
3. For each of the M selected patients, modify their β at the target class's marker CpGs to shift their per-class A-score by the injected effect size d_inject. The shift direction is the disease's declared direction (positive for case-elevated signals).
4. Run Stages 0-10 on the modified cohort with the M as synthetic-cases and the remaining N_HC - M as synthetic-HC.
5. Recover the headline statistic. Compare against injected d_inject.

**PASS criterion.** Recovered effect size within ±20% of injected at the target class, with correct direction (sign).

**Known Phase A2.1 limitation.** The current N6 uses a simplified correlation-based scoring in the recovery tester rather than the full H(β)/H_min production formula. This catches sign-of-effect and order-of-magnitude recovery but underestimates absolute recovery by ~5×. **Phase A2.1 deliverable**: swap the simplified scorer for the full production chain. Until then, N6 PASS means "chain recovers signal in the right direction with magnitude within an order of magnitude," not "chain recovers signal at production precision."

**Known Phase A2.1 limitation (signed direction).** The current synthetic patient generator injects case-vs-HC signal in positive direction only. Real-data signals with negative direction (e.g., the PC2 T-cell SUPPRESSION axis in VAL-005) appear as absolute-value matches in N6. **Phase A2.1 deliverable**: signed-direction injection.

**CMB equivalent.** The **signal-injection-and-recovery test** that EVERY CMB pipeline runs as the most important null. Planck NPIPE injects synthetic CMB realizations (FFP10 simulations) at known cosmological parameters into the timestream and verifies the pipeline recovers those parameters within their declared uncertainties. The CMB community treats injection-recovery as the irreducible minimum: a pipeline that cannot recover injected signal is not a pipeline yet, regardless of how sophisticated its claimed analyses are.

**How the methylome differs in implementation.** Signal injection at the β-matrix level rather than the timestream level (because the methylome's L1 is per-sample IDAT, not continuous timestream). The injection is at the L3 boundary (clean β matrix) and propagates forward through L4-L8 to the headline. Phase A2.2 (implicit, deferred to Phase C) will extend injection-recovery to the IDAT level for full L1-through-L8 coverage.

**How it's the same in principle.** Both test that the pipeline recovers a known truth from known input. A pipeline that fails this is not a pipeline; it is an unverified system.

**Outputs.** `N6_verdict`, `N6_recovery_ratio` (recovered / injected), `N6_sign_match`, per-injection-strength sweep table.

**Decision points.** PASS → SEALED contribution. FAIL → either RETRACT (sign wrong) or RESTATE (sign right but magnitude off).

**Failure modes.**
- **Recovery sign opposite injection.** Indicates upstream sign error somewhere in Stages 0-8. The chain reports a positive signal when truth is negative — the worst failure mode possible. Triggers immediate halt and audit.
- **Recovery magnitude << injection.** Indicates information loss in the chain (most likely in foreground subtraction or in deconvolution). Framework reports the ratio and flags for review.
- **Recovery magnitude >> injection.** Indicates an amplification systematic — the chain is amplifying signal in a way that doesn't correspond to truth. Equally worrying as opposite sign.

**Canonical cross-references.** Recipe §11.6. Roadmap §10.2.1 Phase A2 (synthetic patient generator). §12.1 of Roadmap (Phase A known limitations).

**CPG Plate references.** None directly.

**Chain-link assignment.** L9.

---

## §87. L9.7 — N7: End-to-end synthetic-patient simulation

**What this null tests.** A more demanding version of N6: instead of starting from real HC patients and modifying them, generate fully synthetic patients from a known cosmological-equivalent truth model, run them through the full chain L1-L8, and verify the chain reproduces the truth.

This is structurally equivalent to Planck's **FFP10 (Full Focal Plane 10) simulations** or **NPIPE end-to-end sims** — the gold-standard discipline of "the only thing you can fully trust about your pipeline is what it does on data you fully generated yourself."

**Inputs.** A truth specification: declared per-class fractions, declared per-cell-type fractions, declared cellular age per class, declared cohort design (n_case, n_HC, age distribution, sex distribution, disease signal strength). Optional: declared plate-position structure, declared batch structure.

**Atlas reference.** Indirect. The synthetic patient generator (`synthetic_patient_generator.py`) consults IAMAtlas to draw per-class posterior β values, modulates them by the declared truth, and outputs synthetic β matrices.

**Files invoked.** `cpg_null_runner.py::run_N7_end_to_end_simulation()`. Calls `synthetic_patient_generator.py`. Drives the full chain.

**The math.**
1. From the truth specification, the synthetic patient generator produces an `n_patients × n_CpGs` β matrix where each row is a synthetic patient drawn from IAMAtlas with the declared per-class mixing fractions and declared per-class A-score offsets.
2. The β matrix is written to a synthetic-cohort manifest as if it came from a real Illumina array.
3. The full chain L1-L8 runs on the synthetic cohort.
4. The framework compares per-patient recovered quantities (per-class A-scores, Mahalanobis distance, cellular age per class) against the truth specification.

**PASS criteria** (multi-channel):
- **Per-class A-score recovery**: bias < 5%, RMSE within declared tolerance.
- **Cellular age recovery**: bias < 2 years per class, saturation flags consistent with truth-out-of-range patients.
- **Card verdict recovery**: when truth specifies a card-firing pattern, the chain's Stage 8 verdict matches at ≥80% concordance.
- **Sign correctness**: 100% sign agreement on declared-direction signals.

**Phase A status.** Phase A built the synthetic patient generator and ran a basic N7 sweep across the 7 Family A VALs. The Phase A N7 implementation uses the simplified scorer (§86 limitation). Phase A2.1 upgrades to production-precision recovery — at which point N7 becomes the methylome's FFP10/NPIPE equivalent at full strength.

**CMB equivalent.** **FFP10 / NPIPE end-to-end pipeline simulations** — the Planck collaboration's most expensive computational discipline. They run thousands of full-pipeline simulations from declared cosmological truths through every Planck data-processing stage, and they verify that the recovered cosmological parameters match the truths within declared uncertainties. **CPG's N7 is structurally identical** — the methylome implementation of the same discipline. Phase A2.2 (implicit; HEALPix already exists for the plates) extends synthetic generation to HEALPix-pixelized methylome representations, which is the prerequisite for Phase C correlation-structure analyses (TODO 2.1 C(d), TODO 2.2 bispectrum) to operate on synthetic data the same way they would on real data.

**How the methylome differs in implementation.** Synthetic generation is at the β-matrix level rather than the bolometer-timestream level. The chain enters at Stage 0 (intake) rather than L1 (timestream); the L1-through-L3 chain steps operate trivially on synthetic input because no real wet-lab QC is involved. The discipline is the same; the implementation skips the lab.

**How it's the same in principle.** Both verify the pipeline's behavior on data the pipeline itself did not produce. This is the only way to trust the pipeline.

**Outputs.** `N7_verdict`, per-channel recovery diagnostics, the synthetic-truth-vs-recovery table.

**Decision points.** PASS → critical SEALED contribution. FAIL → RETRACT contribution (because if the chain cannot recover synthetic truth, every real-data result is suspect).

**Failure modes.**
- **Recovery bias outside tolerance.** Indicates an inductive bias in the chain that the synthetic data exposes. Framework reports the bias direction and magnitude.
- **Saturation pattern mismatch.** Indicates Stage 6 cellular age inversion is mis-handling out-of-range patients. Framework flags the mismatch for Stage 6 investigation.
- **Card concordance below 80%.** Indicates Stage 8 matching rules are over- or under-firing on synthetic patterns. Framework reports per-card concordance.

**Canonical cross-references.** Recipe §11.7. Roadmap §10.2.1 Phase A2 + §12.1 known limitations.

**CPG Plate references.** **Plate 3 (Grandaddy Plate)** — the CMB realization on its right panel was produced via `healpy.synfast()` from Planck's ΛCDM C_ℓ spectrum. That is the CMB-side analog of CPG's synthetic patient generator: generate synthetic data from a known theoretical model and use it to validate the analysis chain.

**Chain-link assignment.** L9.

---

## §88. L9.8 — N8: Look-elsewhere-effect correction

**What this null tests.** When a VAL searches across many possible features and reports the top one (or top few), the headline p-value must be corrected for the search space. A "p<0.001 in chr6 MHC" finding loses significance if the search included all 22 autosomes — the chance that ONE of 22 chromosomes shows p<0.001 by chance is ~22 × 0.001 = ~2% (Bonferroni), not 0.1%.

**Inputs.** The headline statistic. The declared search space size N_search (declared in PREREG — typically number of chromosomes scanned, number of CpGs scanned, number of cell types scanned, etc.).

**Atlas reference.** None.

**Files invoked.** `cpg_null_runner.py::run_N8_look_elsewhere()`.

**The math.** Two corrections supported (per-VAL declared):

1. **Bonferroni** (conservative): `p_corrected = min(1.0, p_raw × N_search)`. PASS if `p_corrected < α` (typically α=0.05).
2. **Family-wise error rate via permutation** (less conservative, more accurate): use the N1 permutation distribution (§81) but record the *maximum* statistic across the search space per permutation. The corrected p-value is the fraction of permutations where the maximum-across-search-space exceeds the observed maximum.

Default for CPG-VALs: **option 2 (permutation-based FWE)** because Bonferroni is over-conservative when search-space features are correlated (CpGs within a chromosome are strongly correlated; per-cell-type A-scores are correlated). The permutation FWE respects the empirical correlation structure.

**Why this matters.** This is the null that **VAL-006 (chr6 MHC enrichment) failed in Phase A**. The original VAL-006 reported `p=0.009 for chr6 enrichment` without correcting for the 22 chromosomes scanned. Phase A's N8 correction gave `p_FWE ≈ 0.18` — no longer significant at the 0.05 threshold. The VAL was **RESTATEd** rather than RETRACTED because the underlying chr6 MHC signal is consistent with the cross-cohort VAL-005 PC2 T-cell suppression finding (the two findings reinforce each other across the L4 immunological direction), but the per-chromosome enrichment as a standalone claim does not survive look-elsewhere correction. The signal is real; the framing was too strong.

**CMB equivalent.** **Look-Elsewhere Effect (LEE) correction** in cosmological parameter searches and CMB anomaly searches. Famous examples: the WMAP "Cold Spot" significance was inflated by an order of magnitude before LEE correction; once corrected, it was 1.5-2σ rather than 4σ. The Planck "Axis of Evil" alignment significance underwent similar correction. CPG's N8 is the methylome implementation of the same statistical discipline.

**How the methylome differs in implementation.** Search space is per-VAL declared (chromosomes, CpGs, cell types). Same correction math.

**How it's the same in principle.** Both protect against the false-discovery rate inflation from multi-feature search.

**Outputs.** `N8_verdict`, `N8_raw_pvalue`, `N8_corrected_pvalue`, `N8_correction_method`, `N8_search_space_size`.

**Decision points.** PASS → SEALED contribution. FAIL → RESTATE contribution (the underlying signal may be real but the per-feature framing doesn't survive multiple comparisons).

**Failure modes.**
- **Search space size undeclared.** Caught at PREREG time; cannot run N8 without declared N_search.
- **Permutation FWE budget insufficient.** Framework reports the FWE estimate with explicit Monte Carlo uncertainty.

**Canonical cross-references.** Recipe §11.8. Roadmap §10.2.1 Phase A1. §12 of Roadmap (VAL-006 RESTATE on N8).

**CPG Plate references.** **Plate 2 (Breast Pre-Diagnostic Anisotropy)** — the chr6 zoom panel shows the MHC region enrichment that VAL-006 originally claimed; the audit-trail truth is that the per-chromosome enrichment does not survive N8 correction, but the underlying MHC concentration of immune-related signal IS real and IS visible on the plate. The plate visualizes the signal honestly; the framing protects against overclaim.

**Chain-link assignment.** L9.

---

## §89. L9.9 — Synthetic patient generator (`synthetic_patient_generator.py`)

**What this step does.** The companion module to the null framework. Produces synthetic β matrices, manifests, and IDAT-equivalent data for N6 (injection-recovery) and N7 (end-to-end simulation). The synthetic generator IS the chain-of-custody's claim to having an FFP10/NPIPE-equivalent — without it, the chain cannot be tested at the end-to-end level.

**Inputs.** Cohort design specification: `n_case, n_HC, age_distribution, sex_distribution, disease_signal_specification (target_class, effect_size, direction), foreground_levels (age_drift_strength, sex_effect_strength, batch_effect_strength), plate_position_assignment, batch_assignment`.

**Atlas reference.** **IAMAtlas REBUILD posterior consulted heavily.** The synthetic generator draws per-CpG per-class β values from the atlas posterior (mean and SD), constructs synthetic-patient β matrices as mixtures of per-class draws weighted by declared per-class fractions, then injects the declared signal modifications.

**Files invoked.** `Biological_Physics/chain_of_custody/L9_null_suite/synthetic_patient_generator.py`.

**The math.**

Per synthetic patient `p`:
1. Draw per-class mixing fractions `f_p` from the declared cohort design (case patients get the disease-modified mixing pattern; HC get the baseline mixing pattern).
2. For each CpG `i`, sample the per-class β from the IAMAtlas posterior: `β_class[i] ~ Normal(atlas_mean[i, class], atlas_sd[i, class])`.
3. Compose the patient's observed β: `β_observed[i, p] = Σ_class f_p[class] × β_class[i] + ε_noise + ε_age + ε_sex + ε_batch + ε_plate`.
4. Inject signal: for the declared target class's marker CpGs, shift `β_observed` by the declared effect size in the declared direction.
5. Write a synthetic IDAT-equivalent manifest entry.

**Cohort design knobs:**
- `n_case`, `n_HC`: cohort size.
- `age_distribution`: Normal with mean and SD (typically matching a real cohort's distribution).
- `sex_distribution`: P(female | case), P(female | HC) — typically equal unless modeling sex-biased disease.
- `disease_signal_specification`: per-class target, per-cell-type sub-target, effect-size magnitude, direction.
- `foreground_levels`: how strong are age/sex/batch/plate foregrounds in this synthetic cohort? Typically calibrated to match real-cohort foreground strengths.

**CMB equivalent.** **FFP10 generator** in Planck. Given a cosmological model and a survey design, FFP10 produces synthetic timestreams for every Planck bolometer for every scan ring of the mission. These synthetic timestreams pass through the full Planck pipeline as if they were real data, allowing every analysis to be tested against known truth. **The CPG synthetic patient generator is the methylome implementation of FFP10's discipline** — same operational role, same chain-of-custody guarantee.

**How the methylome differs in implementation.** Generation at the per-patient β-matrix level rather than per-bolometer per-scan-ring timestream level. The synthetic patient generator skips L1 (no synthetic bolometer noise) and enters at L2-L3 (synthetic β matrices). Future Phase C HEALPix-compatibility extends generation to L1-equivalent (synthetic IDAT-equivalent intensities).

**How it's the same in principle.** Both produce synthetic data from declared truth specifications, allowing the pipeline to be tested on data the pipeline did not produce. This is the bedrock of chain-of-custody validation.

**Outputs.**
- `synthetic_cohort_manifest_<spec_id>.csv` — manifest entries indistinguishable from real Illumina cohort manifests.
- `synthetic_beta_matrix_<spec_id>.csv` — β matrix consumable by Stage 0+ of the chain.
- `synthetic_truth_<spec_id>.json` — the declared truth for downstream comparison.

**Decision points.** The synthetic generator does not make decisions; it produces data. The downstream chain consumes synthetic data identically to real data.

**Failure modes.**
- **Atlas posterior insufficient.** If the declared signal injection requires CpGs where the atlas posterior is poorly determined (e.g., the stromal galactic mask), the generator flags this and produces best-effort synthetic data with explicit warnings.
- **Inconsistent cohort design.** A specification with `n_case=0` is internally inconsistent; the generator refuses to produce.

**Canonical cross-references.** Recipe §12 (synthetic generation). Roadmap §10.2.1 Phase A2. §13.4 of Roadmap (HEALPix as Phase A2.2 implicit).

**CPG Plate references.** **Plate 3 (Grandaddy Plate)** — the right panel CMB realization is the visual analog of what the synthetic patient generator produces for the methylome. Both are synthetic data from a known theoretical/calibrated model, used to validate analysis chains.

**Chain-link assignment.** L9.

---

## §90. L9.10 — VAL sealing protocol (PREREG → OUTCOME → SEALED / RESTATE / RETRACT)

**What this step does.** The protocol that takes a candidate finding through the L9 framework to a sealed (or restated, or retracted) verdict. **Every CPG-VAL must follow this protocol.** No VAL ships without it.

**The protocol has four stages:**

**Stage S1 — PREREG (Pre-Registration).** Before any beta value is observed (CCL-040/041 absolute), the VAL's analyst writes a PREREG document declaring:
- The hypothesis (e.g., "Immune class A-score in cases >10y pre-dx of breast cancer is elevated vs matched HC at d > +1.0").
- The cohort specification (which cohort, n_case, n_HC, inclusion criteria, exclusion criteria).
- The statistical test (e.g., "Two-sided Welch's t-test on immune A-score, with permutation null").
- The significance threshold (e.g., "p<0.05 with N1 + N7 + N8 PASS required for SEALED").
- The declared null set (which of N1-N8 applies).
- The search space size (for N8).
- Any additional VAL-specific constraints.

The PREREG is SHA-256 hashed and committed to the repo at `Biological_Physics/validation_runs/CPG-VAL-XXX/prereg.json` before any case/HC labels are looked at.

**Stage S2 — DATA RUN.** The VAL runs Stages 0-10 of the chain on the declared cohort. Outputs `per_sample.csv` (per-patient reconstructions) and `headline_results.json` (the declared statistic computed on the actual cohort).

**Stage S3 — L9 NULL SUITE.** The L9 framework (§80) is invoked with the PREREG + headline_results + per_sample.csv. The framework runs the declared null subset and produces `null_results.json`.

**Stage S4 — SEALING VERDICT.** The framework aggregates the per-null verdicts into one of three sealing verdicts:

- **SEALED.** All declared nulls PASS. The VAL's headline becomes a sealed CPG framework claim. The OUTCOME document records the sealing date, the framework version, the atlas SHA-256 fingerprint, the null suite results, and the headline finding. The VAL enters the production canonical inventory and is referenced by downstream cards.

- **RESTATE.** One declared null FAILs in a way that limits the result's scope without invalidating its underlying signal. The OUTCOME document records the failed null, the restated scope (e.g., "VAL-006's chr6 MHC enrichment claim is restated as 'concentration of immune-related signal in HLA-bearing region', no longer as a per-chromosome enrichment with multiple-testing correction"). The VAL enters the canonical inventory with the limitation declared honestly.

- **RETRACT.** Two or more declared nulls FAIL, OR the failed nulls indicate the headline is uninterpretable. The OUTCOME document records the retraction reason. The VAL does NOT enter the canonical inventory. Future work may revisit the VAL with corrected upstream methodology, but the original claim is withdrawn.

**Inputs.** PREREG + per_sample.csv + headline_results.json + null_results.json.

**Atlas reference.** None at this step (audit only).

**Files invoked.**
- The L9 framework (§80) for null computation.
- Repo commit pipeline for OUTCOME documents.

**The math.** Aggregation logic only (declared in §80).

**CMB equivalent.** **The Planck collaboration's release process.** Before a Planck result becomes a community-citable finding, it passes through: (1) internal pre-registration of analysis choices and null tests; (2) full pipeline run; (3) null test battery; (4) collaboration-wide review; (5) release with documented sealing status and known limitations. **CPG's VAL sealing protocol is the methylome implementation of this discipline at the per-finding granularity.**

**How the methylome differs in implementation.** Per-VAL granularity rather than per-release. Each CPG-VAL gets its own PREREG/OUTCOME pair; Planck releases bundle hundreds of derived quantities into one collaboration-wide product. The principle is identical; the operational unit differs.

**How it's the same in principle.** Both prevent post-hoc statistical fishing. Both produce a permanent record of every choice made. Both produce three possible verdicts (sealed / preserved-with-limit / withdrawn) and treat the limited and withdrawn cases honestly rather than hiding them.

**Outputs.** The OUTCOME document at `Biological_Physics/validation_runs/CPG-VAL-XXX/outcome.md`. The sealed VAL artifact bundle.

**Decision points.** The sealing verdict is the decision point. Downstream cards consume sealed VALs only; restated VALs are referenced with explicit caveat; retracted VALs are not referenced.

**Failure modes.**
- **PREREG modified after data observation.** Caught by SHA-256 + commit-history audit. If detected, the VAL is automatically RETRACTed.
- **Headline result data does not match PREREG specification.** E.g., PREREG declared n_case=146 but per_sample.csv shows n_case=139. Framework returns FRAMEWORK_INCONSISTENCY; VAL does not seal.

**Canonical cross-references.** Recipe §11.10. Roadmap §10.2.1 Phase A3 (running all 7 Family A VALs through the framework). §12 of Roadmap (Phase A results: 5 sealed, 2 restated).

**CPG Plate references.** Every plate (1-4) is built from sealed-VAL data. The plates are visual products of the sealing protocol's outputs.

**Chain-link assignment.** L9.

---

## §91. L9.11 — Null-suite invocation order

**What this step does.** Documents the canonical order in which L9 nulls are invoked when the framework runs a VAL through the full suite. The order matters because some nulls produce intermediate artifacts that others consume.

**Invocation order:**

1. **N1 (HC label permutation)** — runs first because its permutation distribution is the foundation for N8's FWE correction.
2. **N2 (Age-stratified permutation)** — runs second; independent of others.
3. **N3 (Sex-stratified permutation)** — runs third; independent of others.
4. **N4 (Cohort-split replication)** — runs fourth; uses random 50/50 split independent of N1-N3.
5. **N5 (Plate/array-position null)** — runs fifth; requires position metadata; independent of others.
6. **N6 (Injection-recovery)** — runs sixth; invokes `synthetic_patient_generator.py`; CPU-intensive.
7. **N7 (End-to-end synthetic-patient simulation)** — runs seventh; the most CPU-intensive null; full chain run on synthetic cohort.
8. **N8 (Look-elsewhere correction)** — runs LAST because it consumes the N1 permutation distribution for FWE estimation.

The framework invokes the declared subset of these (a VAL declares which apply in its PREREG). Independent nulls (N1, N2, N3, N4, N5) can run in parallel; N6 and N7 are dependent on synthetic generation; N8 is dependent on N1.

**Inputs.** PREREG declared null subset.

**Atlas reference.** Indirect (N6, N7 consult atlas via synthetic generator).

**Files invoked.** `cpg_null_runner.py::run_full_suite()`.

**The math.** Dispatch logic.

**CMB equivalent.** The **canonical null-test ordering** in Planck NPIPE. Planck's null suite has a well-documented invocation order driven by dependency graph (jackknife resampling pre-computed before null statistics; FFP10 simulations are independent and parallelizable; LEE correction uses the per-statistic null distributions). CPG's invocation order is the methylome implementation.

**How the methylome differs in implementation.** Smaller dependency graph (8 nulls vs Planck's tens of null tests). Same principle.

**How it's the same in principle.** Both establish a canonical order so that the same VAL invoked twice produces identical null results.

**Outputs.** A per-VAL `null_invocation_log.json` recording per-null start/end times, framework version, dispatch decisions.

**Decision points.** None — pure orchestration.

**Failure modes.**
- **Per-null timeout.** Configurable; default 1 hour per null. A timeout returns FRAMEWORK_TIMEOUT for that null; downstream verdict propagates the timeout as INSUFFICIENT (not PASS, not FAIL).
- **Resource exhaustion** (memory, disk for synthetic cohorts). Framework reports the resource shortfall and refuses to seal until resources are available.

**Canonical cross-references.** Recipe §11.11. Roadmap §10.2.1 Phase A.

**CPG Plate references.** None.

**Chain-link assignment.** L9.

---

**End of Part III (L9 audit machinery, §80-§91).** The chain-integrity scaffolding is complete: the framework (§80), each of the eight nulls (§81-§88), the synthetic patient generator (§89), the sealing protocol (§90), and the invocation order (§91).

Part IV follows with failure modes and decision trees (§92-§96). Part V with reference tables (§97-§102).

---

*v1 — 2026-05-31. Author: Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute. Working partner: Walther (Claude).*
