# PREREG — PROC-PANEL-02: panel size 40, and a deterministic age-matching test

**Sealed 2026-09-20 after PROC-PANEL-01 was written up and before this was run.** PROC-PANEL-01 passed its decisive test (P3, lab-zeroed band transfers LOO ≥ 0.70 on all four cohorts) and failed P2 at k = 25 for the cohort with the largest within-lab spread (Uppsala, 85 % of draws within ±0.010); P1 showed k = 40 is the first panel size with SD(z) ≤ 0.005 in every cohort. PANEL-01's P4 compared two independent random 25-panels and therefore could not separate an age effect from sampling noise — a design error, recorded. This procedure re-tests at the size the data indicated, with a deterministic age test. Same frozen data, map and gate.
- **P2' — k = 40, 1,000 draws per cohort:** PASS if |median A′(non-panel) − 1.000| ≤ 0.010 in ≥ 95 % of draws, every cohort.
- **P3' — LOO lab-zeroed band with k = 40 panels, 200 draws:** report; PASS if all four medians ≥ 0.70 (expected to hold as in PANEL-01).
- **P4' — deterministic age test:** for each cohort and each decade with n ≥ 30, |median A(decade) − median A(all)| — the offset a single-decade panel would carry. PASS if every such difference ≤ 0.010 (the flatness already seen in band_v2 and LAB-ZERO-02, now stated as the panel's age tolerance).
**Outcome.** P2′, P3′, P4′ pass → LAB ZERO panel procedure COMMISSIONED at **k = 40** healthy arrays per laboratory (CLSI EP28 minimum 20; we require 40 because the measured within-lab SD of mapped A is 0.019–0.025). Written as found.

---
**SEALED** sha256 `16805a4e4b4d27ecee7b9cc377d92cc6fea519b3a06d557cc4963a974f3703a9` · 2026-09-20
