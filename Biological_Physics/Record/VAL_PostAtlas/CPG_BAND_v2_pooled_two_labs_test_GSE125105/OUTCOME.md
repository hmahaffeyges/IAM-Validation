# OUTCOME — identity_band_v2 tested on GSE125105 controls (Munich)

**Run:** 2026-09-20, `band_v2_test_run.py` (Stage 1 via 6 subprocess workers, identical per-sample path to PROC-CAL-01; 210 arrays in 37 min). IDATs fetched per-sample with `tools/geo_fetch_idats.py` (controls only, 1.7 GB, 6 min). Map and band **frozen** (GSE87571 + GSE42861). **Against:** PREREG.md sealed (tar never downloaded; per-sample fetch after sealing). The 489 depression arrays were not opened.

## Verdict: **P3 FAIL (0.9645), P4 FAIL (54.7 %). The per-cohort constant is confirmed as a third layer that pooling cannot remove.**

| condition | bar | observed | verdict |
|---|---|---|---|
| P1 | ≥ 95 % | 210/210 | PASS |
| P2 presence | ≥ 90 % | 95.7 % (n = 201) | PASS |
| **P3** median mapped immune A | ±0.02 of 1.000 | **0.9645** (unmapped 0.7918) | **FAIL** |
| **P4** in pooled band | ≥ 80 % | **54.7 %** | **FAIL** |
| P5 Munich − pooled p50 per decade | report | 25–34 −0.026 · 35–44 −0.025 · 45–54 −0.036 · 55–64 −0.033 · 65–74 −0.034 (14–24 n=8 −0.011; 75–84 n=5 −0.031) — **flat across age, ≈ −0.030** | recorded |
| N-random (size-matched, level test) | random-panel median > 0.05 from 1.0 | 1.174; 0 % in band | PASS — the identity loci set the level; Phase 1c's mean-β-matched null was circular for H(β̄) and is retired |
| N-sex | report | \|d\| 0.17 (35–44), 0.35 (45–54), 0.38 (55–64) — **unsigned as run; no direction claimed** | recorded |
| N-plate (21 Sentrix chips) | report | F = 1.08, **p = 0.39 — no chip effect** | recorded |
| N-smoke | — | field absent in GSE125105 | n/a |

## Three cohorts, one picture
| cohort | lab / population | offset vs Uppsala (immune A, mapped) | within-cohort chip effect |
|---|---|---|---|
| GSE87571 | Uppsala, SE | 0 (reference) | — |
| GSE42861 | Karolinska, SE | +0.018 | p ≈ 1e-12 |
| GSE125105 | Munich, DE | −0.030 | p = 0.39 |

The pipeline map moved Munich from 0.79 to 0.96: the +0.066 β pipeline term is real and shared (Stage 1s stays COMMISSIONED). The residual is a **per-cohort constant** of a few hundredths, differing in sign and size per cohort, flat across age, and independent of chip structure (Munich shows none and still sits −0.03). Its size equals the band half-width, so any new cohort lands ~50 % in a band built elsewhere, however many labs are pooled. **Pooling widens the band; it cannot remove a constant that is different for the next lab.**

## Design decision this forces (author's call on the primary route)
The healthy reference = **FLOOR (physics, universal) + PIPELINE MAP (universal per pipeline) + LAB ZERO (local, once per lab)**. This is the CLSI EP28 practice every clinical assay follows: a lab adopting a reference interval verifies or shifts it on ~20 of its own healthy samples. Routes to the lab zero:
1. **Healthy-control panel per lab** (20–30 arrays, once; median offset subtracted; disclosed on every report). Standard, defensible, available today. → recommended primary.
2. **Learn the lab offset from the array's own control probes** (Stage 0 intensity QCs — deferred hand-off; these are precisely the technical covariates that carry a lab's signature). If it predicts the P5 offsets across the three cohorts, no control panel is needed. → Future Goal, gated on Stage 0/1 hand-off.
Lab vs population cannot be separated with one German cohort; Phase 1b (EPIC-Italy, once an EPIC Stage-1 map is fit) keeps that question.

## Status
identity_band_v2 stays **PROVISIONAL** as the *shape* (per-decade percentiles, pooled width) and is NOT a universal reference. Row B (gauge switch) remains gated: on the lab-zero mechanism, not on more pooling.

---
**SEALED** sha256 `c16cfd1752b768f8822c3abafb8d4a18a1038b960241cbe86a605d2d995f81bc` · 2026-09-20
