# PREREG — PROC-MAHA-01: Stage 5 (departure) re-based on the identity gauge (CHAIN_COMMISSIONING row 5)

**Sealed 2026-09-21 before any code was changed.** Owner H. W. Mahaffey · Analyst Claude Science.
**As wired.** `stage_5_mahalanobis` takes the marker-union class readings, gates on ABOVE_BAND of `age_reference_matrix` (superseded, PROC-N7-01), and sums z² over eight classes with μ, σ derived from that matrix (`mahalanobis_healthy_reference_v2_0_age_matched_derived.json`, 2026-06-30). Its output keys (`mahalanobis_distance`, `mahalanobis_beyond_band`, `alarm_threshold_p95`) reach `run_full` renamed (`distance`, `beyond`), while `cpg_report_builder` reads the long names — the key mismatch recorded 2026-09-19.
**After.** Stage 5 consumes the REPORTED gauge (`classes`, identity loci, three-layer reference). For each component with a commissioned band, z = (A″ − 1.000)/σ, σ = (p90 − p10)/(2·1.2816) from `identity_band_v3` (pooled: σ = 0.0524/2.5632 = 0.0204). Components without a band (haematopoietic-progenitor joint, all non-blood classes) are NOT assessable and do not enter. distance = √Σz²; thresholds √χ²(0.95, n), √χ²(0.99, n) over the n assessable components (n = 1 on whole blood today: 1.960 / 2.576). If the gauge is not reportable (UNSET / UNMAPPED) the departure is not reportable. The derived eight-class hull is retained in the bundle as `diagnostic_hull_marker_union` (lineage) and never the reported departure. Output carries BOTH key sets. Stage 5 on whole blood is therefore, in plain words, "how many band-widths from the healthy line the immune reading sits" — stated as such in the report; it becomes multi-axis only as further components earn bands.
**Tests (fixed).**
- **M1 — healthy cached arrays.** The 5 healthy whole-blood arrays in `betas_cache.pkl` with their lab zeros: none beyond p95. PASS = 5/5 (GSM2333950 at A″ 1.033 → z ≈ 1.6, inside).
- **M2 — four cohorts, in-sample.** 1,379 gated healthy donors (`panel01_input.csv`, A″ via curve + full-cohort zero): fraction beyond p95 ≤ 0.07 pooled and per laboratory (nominal 0.05 for a two-sided z at 1.96 under a normal; the band is empirical, so report the actual tail). Report beyond-p99 too.
- **M3 — synthetic held-out.** The 40 SWITCH-02 TEST patients (A_mapped in `switch02_results.json`, z_atlas = −0.0146): none beyond p95. PASS = 40/40.
- **M4 — keys reconciled.** `run_full(...)["departure"]` contains `mahalanobis_distance`, `alarm_threshold_p95`, `alarm_threshold_p99`, `mahalanobis_beyond_band`, `n_assessable`, `reportable`, and the short aliases; `cpg_report_builder`'s departure block renders from it without KeyError on one healthy bundle.
- **M5 — UNSET refuses.** Without a lab zero: `departure.reportable == False`, distance None. 7/7.
**Outcome.** M1–M5 pass → row 5 COMMISSIONED (whole blood, immune axis). Any fail → written as found.

---
**SEALED** sha256 `e48ba1652f2b2c0f65aa5771dccf365ce4e8882092ea95ef5730ebbfc3dec122` · 2026-09-21
