# PREREG — PROC-SWITCH-02: S4 re-sealed — the synthetic cohort is a fifth laboratory

**Sealed 2026-09-21 after PROC-SWITCH-01 was written up (S1, S2, S3, S5 PASS; S4 FAIL as sealed) and before this was run.**
**What SWITCH-01 S4 found.** With lab zero 0, 40 effect-free synthetic healthy whole-blood patients read median A″ = 0.985 (97.5 % in identity_band_v3). The patient β is exactly its linear mixture of atlas class means; on the immune identity loci that mixture (≈89 % immune, 8 % progenitor, 3 % stem_adult) has β̄ = 0.741, and pure atlas immune has β̄ = 0.7373, against the G-002 floor's β = 0.7318 at A = 1. The atlas posterior and the G-002 reference set are different reference sets; the atlas therefore carries its own constant, ≈ −0.010 (pure immune) to −0.015 (whole-blood mixture). SWITCH-01 S4 assumed the constant was zero. Real Uppsala healthy blood, mapped, reads 0.988; the synthetic reads 0.985 — consistent.
**Tests (fixed).** Generator: WHOLE_BLOOD_ALPHA, disease 0, default noise/age/batch, ages 20–80, random_seed 2027, n_hc = 80, split by patient index: 0–39 = PANEL, 40–79 = TEST (disjoint).
- **S4a — the atlas's own zero, measured.** z_atlas = lab_zero.compute_lab_zero(panel A_mapped, panel ages). Report; expected ≈ −0.015 ± 0.005.
- **S4b — the rest reads 1.00.** TEST patients: A″ = A_mapped − c(decade) − z_atlas. PASS if |median A″ − 1.000| ≤ 0.010 and ≥ 80 % in identity_band_v3 p10–p90.
- **S4c — the defect is gone.** Marker-union statistic on the same TEST patients: report median (SWITCH-01: 1.125, every patient above band). PASS if the switched gauge's in-band fraction exceeds the marker-union's by ≥ 0.5.
- **S4d — UNSET refuses.** TEST patients without a zero: reportable False, 80/80.
**Outcome.** S4a–S4d pass with SWITCH-01's S1, S2, S3, S5 → row B COMMISSIONED; the atlas's constant is recorded in RECON as a measured number and written into the generator's docstring. Any fail → written as found.

---
**SEALED** sha256 `2bce176ee3f0b549c4c8e9c6b4cac69225603c7da08365ad5e45f14f458da322` · 2026-09-21
