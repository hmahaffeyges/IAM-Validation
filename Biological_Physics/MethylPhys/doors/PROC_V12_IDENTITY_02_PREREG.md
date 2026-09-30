# PROC-V12-IDENTITY-02 — pre-registration: identity loci on atlas v2, point rule; held-out self-read

**Written 2026-09-29 after PROC-V12-IDENTITY B3 failed (PROC_V12_IDENTITY_OUTCOME_01.md), before this build.** Same independence
from v1 as the first procedure (no v1 file, name or atlas is read). One change to the rule; the test is unchanged.

## The rule
1. Target b*(k): upper-branch beta with H(b*) = H_min(k) (G-002, frozen). Class from atlas_v2_roster.csv `class_by_draft_rule`, used
   only to pick H_min.
2. Candidates: the cell is MEASURED at the locus (n_obs ≥ 2; **n_obs ≥ 1 for the four single-pooled-sample Tian cells, flagged
   POOLED-ONE-SAMPLE**), R̂ < 1.01, and the **posterior mean** μ_c within b* ± 0.05. (The first procedure's "whole 95 % interval
   inside the window" is withdrawn: for 2–5-sample cells it is wider than the window.) The atlas uncertainty is not dropped: it is
   carried onto every A as an interval from the 20 posterior draws (V8), where a reader sees it.
3. ≥ 100 loci, else NOT READABLE.
4. A = H(mean β over the loci)/H_min.

## The test (unchanged from the first pre-registration)
Cross-fitting on samples, halves drawn with `default_rng(12)`: loci chosen from half 1's own samples (each on the atlas scale by
its source term; "every half-1 sample inside the window" for the half-1 selection), half-2 samples read individually, then swapped.
- **B1:** ≥ 95 % of held-out sample readings in NORMAL (0.95–1.05, tier_breakpoints.json v1.5).
- **B2:** every cross-fittable cell's median held-out reading in NORMAL; failures named with source and platform.
- **B3:** every admitted cell builds ≥ 100 loci from the v2 posterior.
Cells with one sample (the four Tian pooled cells) cannot be cross-fitted: reported **self-read NOT TESTABLE**, excluded from B1/B2
by construction, and flagged on every reading until a second sample exists.
