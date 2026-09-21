# PREREG — PROC-PANEL-01: the per-laboratory healthy-control panel as the lab zero

**Sealed 2026-09-20 before any panel was drawn.** Owner H. W. Mahaffey · Analyst Claude Science.
**Why.** LAB-ZERO-02 decided that the lab zero is measured, not modelled: a panel of healthy arrays run once per laboratory, its median mapped immune-class A defining the laboratory's offset. This procedure tests that design on the four cohorts already held, with no new download. It is the last gate before the gauge switch (CHAIN_COMMISSIONING row B).
**Data (frozen).** Per-sample mapped immune identity-loci A, age and Sentrix chip for the gated healthy donors of GSE87571 (Uppsala), GSE42861 controls (Karolinska), GSE125105 controls (Munich), GSE111629 controls (UCLA), exactly as sealed in PHASE 1c, band_v2, LAB-ZERO-02. Map `stage1_noob_450K` frozen. Gate unchanged.
**Definition under test.** Lab zero z_L = median(A_panel) − 1.000, where the panel is k healthy arrays from laboratory L. A donor's lab-zeroed reading is A' = A − z_L. Physics says a healthy donor reads A' ≈ 1.000 regardless of laboratory.
**Tests (fixed).**
- **P1 — panel size.** For k ∈ {10, 15, 20, 25, 30, 40, 50}, 1,000 random panels per cohort: SD of z_L across draws. Report the smallest k at which SD ≤ 0.005 in every cohort. Prediction: k = 25 suffices (SD of a median at sd 0.02, n = 25 ≈ 0.005).
- **P2 — the rest reads 1.00.** k = 25, 1,000 draws per cohort: median A' of the non-panel donors. PASS if |median A' − 1.000| ≤ 0.010 in ≥ 95 % of draws, every cohort.
- **P3 — a lab-zeroed band transfers (the decisive test).** Leave-one-cohort-out: zero each training cohort by its own random 25-panel, build a per-decade p10–p90 band of A' on the three pooled training cohorts (decades with n ≥ 30), zero the held-out cohort by its own random 25-panel, place its non-panel donors. 200 draws. PASS if the held-out in-band fraction is ≥ 0.70 (median over draws) for all four hold-outs. (band_v2 without a lab zero reached 0.55 and 0.53; a p10–p90 band contains 0.80 if everything transfers.)
- **P4 — does the panel need age-matching?** Panel drawn from one decade only (the cohort's best-sampled decade), k = 25, 1,000 draws: |z_L(single decade) − z_L(all ages)| ≤ 0.010 in ≥ 95 % of draws → a panel does NOT need age matching (offsets were flat across age in every cohort).
- **P5 — cross-lab panel is wrong by the cohort constant** (sanity, must fail): zeroing UCLA with an Uppsala panel leaves |median A' − 1| ≈ 0.046. Reported, not scored.
- **N-chip.** Within each cohort, one-way ANOVA of A' on Sentrix chip, before and after zeroing (a constant cannot change F; reported to show what the panel does NOT fix).
**Outcomes.** P1–P4 pass → the panel procedure is COMMISSIONED as LAB ZERO; the gauge switch (row B) opens with the three-layer reference. P3 fail → the lab constant is not the whole between-lab term; investigate the per-decade residual before the switch. Written as found.

---
**SEALED** sha256 `144e7e2c394e35c199685031f03871cc428b14ffc7c00ad286a75c08be1a224b` · 2026-09-20
