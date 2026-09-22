# PREREG — PROC-AGE-01: Stage 6 (cellular age) on the identity gauge (CHAIN_COMMISSIONING row 6)

**Sealed 2026-09-21 before the run.** Owner H. W. Mahaffey · Analyst Claude Science.
**As wired.** `stage_6_cellular_age` inverts `age_reference_matrix.json` (marker-union band, superseded) with identity-loci β̄ — a curve and an input from different surfaces; the result pinned at 4 yr (row 6 note). Last consumer of the marker-union statistic in the chain.
**After.** Cellular age = the age at which the healthy immune identity-gauge curve reads the patient's lab-zeroed A: invert c(age) from `reference_age_curve_v1.json` (per-decade medians about the grand median; linear interpolation; end-slope extrapolation) at A_mapped − z_lab − 1. Curve built leave-one-laboratory-out for the test; production uses the four-lab curve.
**Prediction, stated before the run.** The curve rises ≈ 0.045 from the teens to the eighties (≈ 0.0006 A per year); the within-decade healthy SD is ≈ 0.020. The single-array age resolution is therefore SD/slope ≈ 30 yr. The row-6 bar (adult healthy within ±10 yr on immune, ≥ 80 %) is predicted to FAIL; the fraction within ±10 yr is predicted ≈ 0.25–0.30 (a ±10 yr window on a ±30 yr normal).
**Tests (fixed).**
- **A1** — 1,379 gated healthy donors, ages 14–94, leave-one-lab-out curve: fraction within ±10 yr of chronological age; median |Δ|; SD of Δ. PASS if ≥ 0.80 within ±10 yr.
- **A2** — the resolution, measured: SD of Δ compared with the prediction 30 ± 8 yr.
- **A3** — direction: Spearman ρ between inferred and chronological age across all donors; report (the curve is monotone by construction only if the decade medians are; state where it is not).
- **A4** — the 7 cached arrays through the switched chain: inferred age vs. stated age; report.
**Outcome.** A1 pass → row 6 COMMISSIONED. A1 fail (predicted) → Stage 6 marked **NOT REPORTABLE at single-array resolution**; the report prints the resolution instead of an age; the code is retained as `diagnostic_cellular_age`; the marker-union age matrix leaves the chain entirely. Written as found.

---
**SEALED** sha256 `48b5be7ca1c332a017d90f028b4d41c36fff701d07e450834ed3e8305ee62daa` · 2026-09-21
