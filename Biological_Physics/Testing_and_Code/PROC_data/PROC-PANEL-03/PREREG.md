# PREREG — PROC-PANEL-03: the lab zero is read against the healthy age curve

**Sealed 2026-09-20 after PANEL-02 was written up and before this was run.** PANEL-02 P4′ showed that healthy mapped immune A is not flat in age within a laboratory (Uppsala, ages 14–94: teens −0.025 to eighties +0.019 about the cohort median), although the *between-lab* offsets are parallel across age. A flat panel median therefore carries whatever age mix the panel happens to have, and PANEL-02 P2′ missed its 95 % bar on the widest-age cohort (Uppsala 94.4 %). **Design change under test:** the lab zero is the median residual of the panel from a reference healthy age curve, z_L = median_i [A_i − c(dec_i)] − 1, where c(dec) is the per-decade healthy median relative to the grand median, built on the *training* laboratories only (leave-one-out), each zeroed by its own full-cohort median. The held-out laboratory never contributes to c.
- **P2″ — k = 40, 1,000 draws per cohort, LOO c:** PASS if |median A″(non-panel) − 1.000| ≤ 0.010 in ≥ 95 % of draws, every cohort, where A″ = A − c(dec) − z_L.
- **P4″ — single-decade panels, k = 40 from the cohort's best-sampled decade, 1,000 draws:** PASS if |z_L(single decade) − z_L(full-cohort deterministic)| ≤ 0.010 in ≥ 95 % of draws, every cohort (the noise floor at k = 40 is ≈ 0.005, so this is now a fair test).
- **P3″ — LOO age-referenced band:** band = p10–p90 of A″ on training labs, all decades pooled (age removed by c); held-out zeroed by a random 40-panel; 200 draws; PASS if all four medians ≥ 0.70.
**Outcome.** All three pass → LAB ZERO COMMISSIONED as: 40 healthy arrays per laboratory, any age mix, read against the reference age curve; the switch opens. Written as found.

---
**SEALED** sha256 `9a4aea7adfef562c1bf845522ac967992e0e3ab1c44c539e2c80d09de566fe56` · 2026-09-20
