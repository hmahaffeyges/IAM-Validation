# OUTCOME — PROC-AGE-01: cellular age on the identity gauge — NOT REPORTABLE at single-array resolution

**Run 2026-09-21.** 1,379 gated healthy donors, four laboratories, ages 14–94; curve built leave-one-laboratory-out; each donor's A residual (A_mapped − own-lab zero − 1) inverted on the held-out curve.

| test | bar / prediction | result | verdict |
|---|---|---|---|
| A1 within ±10 yr | ≥ 0.80 (predicted FAIL, ≈ 0.25–0.30) | **0.159**; median |Δ| 35.6 yr; per lab 0.11–0.28 | **FAIL as sealed** (and worse than predicted) |
| A2 resolution | SD(Δ) 30 ± 8 yr | slope **0.47 mA/yr**, within-lab SD **0.0235** → SD/slope = **50 yr**; per-lab SD(Δ) 31–60 yr on monotone curves; **258 yr** for Uppsala, where the three-lab curve has no teenagers and a non-monotone eighties bin, so end-slope extrapolation runs away | prediction **WRONG** (too optimistic) — recorded |
| A3 direction | report | Spearman ρ 0.27 pooled (0.19–0.29 by lab) — real but weak | recorded |
| A4 cached arrays | report | healthy 58 → 23; 67 → 61; 43 → ">85"; Karolinska 60s → 72, ">85", ">85", ">85" | recorded |

**Reading — corrected the same day after the author asked whether this contradicts the aging discovery. It does not; it reproduces it.** The record's age findings are population measurements: VAL-006 (drift 0.094 mA/yr on Hannum → the 1,075-year extrapolation), CPG-VAL-015 (A_immune vs age r = −0.197, slope ≈ 0.5 mA/yr, decade medians monotone ρ = −0.85, "an aging trajectory marker"), the mammalian paper (class-average healthy drift rate vs species lifespan). This procedure finds **0.47 mA/yr, monotone by decade on every leave-one-lab-out curve, ρ = 0.27, on 1,379 donors from four laboratories CPG-VAL-015 never saw** — the same slope to one significant figure. **The healthy aging trajectory stands and is independently reproduced.** What is below resolution is one person's position on it: 0.047 A across a lifetime is two within-laboratory SDs, so one array resolves age to ~50 yr. CPG-VAL-020 had already said the per-patient inversion "saturates honestly out of calibration" and that VAL-006's r = 0.9999 was "tautological by design"; this procedure puts a number on why. The first write-up's line "the gauge measures fidelity, not time" overstated the closure and is withdrawn; the trajectory is time, measured on populations.

**Surface sign difference, recorded (RECON D2).** CPG-VAL-015 found A_immune *falling* with age (r = −0.197) on the pre-switch marker-union surface; the identity-loci gauge *rises* with age (+0.47 mA/yr). Same magnitude, opposite sign, different loci: the discriminative markers are bimodal and lose entropy with age; the identity loci are unimodal near β ≈ 0.73 and gain it. Both are real; a reader of VAL-015 and this document must be told which surface each sign belongs to. The mammalian paper's cross-species rate was computed on the pre-atlas methylation surface (VAL-006 lineage), not the identity gauge; its lifespan scaling is untouched by this procedure but should be re-derived on the identity gauge before the two are quoted together (Future Goal).

**Closed in code.** `stage_6_cellular_age` returns `reportable=False` with the resolution and a sentence; the report prints the sentence in place of an age; the marker-union inversion is `diagnostic_cellular_age` (lineage only) and the marker-union age matrix is read by no reported path. `pending_recalibration` is now False for every stage: **no reported number in the chain reads the marker-union statistic.**

**Row 6: CLOSED — per-patient cellular age NOT REPORTABLE at single-array resolution.** The population trajectory is reported where it belongs: as the reference age curve (§3.5) every reading is corrected by. Re-opens for individuals only with an instrument built for time (a clock-type sum), outside this document's claim. Serial mode (two arrays from one person a year apart, Δ ≈ 0.0005) is likewise below resolution.

---
**SEALED (corrected write-up)** sha256 `a6087c263711c24de61ebae6d8d722f7bbe41786937be3704214435a28897822` · 2026-09-21 — supersedes the first seal (git history)
