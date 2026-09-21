# OUTCOME — PROC-AGE-01: cellular age on the identity gauge — NOT REPORTABLE at single-array resolution

**Run 2026-09-21.** 1,379 gated healthy donors, four laboratories, ages 14–94; curve built leave-one-laboratory-out; each donor's A residual (A_mapped − own-lab zero − 1) inverted on the held-out curve.

| test | bar / prediction | result | verdict |
|---|---|---|---|
| A1 within ±10 yr | ≥ 0.80 (predicted FAIL, ≈ 0.25–0.30) | **0.159**; median |Δ| 35.6 yr; per lab 0.11–0.28 | **FAIL as sealed** (and worse than predicted) |
| A2 resolution | SD(Δ) 30 ± 8 yr | slope **0.47 mA/yr**, within-lab SD **0.0235** → SD/slope = **50 yr**; per-lab SD(Δ) 31–60 yr on monotone curves; **258 yr** for Uppsala, where the three-lab curve has no teenagers and a non-monotone eighties bin, so end-slope extrapolation runs away | prediction **WRONG** (too optimistic) — recorded |
| A3 direction | report | Spearman ρ 0.27 pooled (0.19–0.29 by lab) — real but weak | recorded |
| A4 cached arrays | report | healthy 58 → 23; 67 → 61; 43 → ">85"; Karolinska 60s → 72, ">85", ">85", ">85" | recorded |

**Reading.** The healthy immune identity-gauge curve does rise with age (LAB-ZERO-02, PANEL-02), but by 0.045 across a lifetime — two within-laboratory SDs. Inverting it for one person resolves age to about half a century. **The gauge measures fidelity, not time.** Methylation clocks work because they sum hundreds of CpGs each with a large age slope; an entropy mean over 42,000 identity loci is a different instrument and should not be dressed as a clock.

**Closed in code.** `stage_6_cellular_age` returns `reportable=False` with the resolution and a sentence; the report prints the sentence in place of an age; the marker-union inversion is `diagnostic_cellular_age` (lineage only) and the marker-union age matrix is read by no reported path. `pending_recalibration` is now False for every stage: **no reported number in the chain reads the marker-union statistic.**

**Row 6: CLOSED — NOT REPORTABLE at single-array resolution.** Re-opens only with an instrument built for time (a clock-type sum), which is outside this document's claim. Serial mode (two arrays from one person a year apart, Δ ≈ 0.0005) is likewise below resolution.

---
**SEALED** sha256 `7c092011a65a98a9f2d9f50f8e4eb6afff61daf7595c005ddf846c4390d38818` · 2026-09-21
