# OUTCOME — PROC-PANEL-01 (as sealed; k = 25, flat lab zero)
**Run 2026-09-20** on the four gated healthy cohorts (Uppsala 659, Karolinska 315, Munich 201, UCLA 204 mapped immune identity-loci A).
| test | bar | result | verdict |
|---|---|---|---|
| P1 panel size | SD(z) ≤ 0.005 all labs | k=25: 0.0040–0.0064; **k=40** is the first size clearing every lab (Uppsala sd 0.025 is the widest) | prediction (25) WRONG for Uppsala |
| P2 rest reads 1.00, k=25 | ≥95 % of 1,000 draws within ±0.010 | UCLA 95.0 · Munich 98.8 · Karolinska 93.0 · Uppsala 85.2 (medians 0.9996–1.0005) | **FAIL as sealed** (panel size) |
| **P3 lab-zeroed band, LOO** | ≥0.70 all four | **0.788 · 0.841 · 0.766 · 0.717**; without lab zero 0.142 · 0.861 · 0.502 · 0.750 | **PASS** |
| P4 single-decade panel | ≥95 % within 0.010 of all-age panel | 73–86 % | **FAIL as sealed** — and the test compares two random 25-panels whose difference is ≈0.007 from sampling alone: a DESIGN ERROR, cannot separate age from noise |
| P5 sanity | must fail | UCLA zeroed with an Uppsala panel: 0.955 | fails as required |
| N-chip | report | Karolinska p 8e-16, Uppsala 2e-19, UCLA 0.009, Munich 0.37 | a constant cannot touch chip; next residual |
**Reading.** The decisive test passed: a per-lab constant from the lab's own healthy arrays makes a three-lab band hold a fourth lab. The two failures are panel size (25→40) and my P4 design. Both re-sealed as PROC-PANEL-02.

---
**SEALED** sha256 `4b87851bde6104842925c3026ae0b81d3eda46243fe17ebc6f759999799d1e35` · 2026-09-20
