# OUTCOME — PROC-MAHA-01: Stage 5 re-based on the identity gauge (as sealed)

**Run 2026-09-21.** `stage_5_mahalanobis` now consumes the reported gauge: z = (A″ − 1)/σ, σ = 0.0204 from identity_band_v3; distance = √Σz² over assessable components (n = 1 on whole blood); thresholds √χ²(0.95|0.99, 1) = 1.960 / 2.576. Long and short keys both emitted; `bundle["mahalanobis"]` feeds `cpg_report_builder._departure_section` directly. The eight-class derived hull is `diagnostic_hull_marker_union`.

| test | bar | result | verdict |
|---|---|---|---|
| M1 healthy cached arrays | 5/5 not beyond p95 | z = −0.75, −0.07, +1.61, +0.96, +0.12; RA cases +0.51, +0.35 | **PASS** |
| **M2 four cohorts, tail beyond p95** | ≤ 0.07 pooled and per lab | pooled **0.065**; UCLA 0.044 · Munich 0.065 · Uppsala 0.056 · **Karolinska 0.098** (p99 tail 0.051) | **FAIL as sealed** (one lab) |
| M3 synthetic held-out | 40/40 inside | max |z| 0.91 | **PASS** |
| M4 keys reconciled | builder renders | renders "0.75 · alarm p95 = 1.96 · 1 class assessable · within" | **PASS** |
| M5 UNSET refuses | 7/7 | 7/7 reportable False, distance None | **PASS** |

## M2: the cause is the chip, measured
Per-lab diagnostic (`maha01_chip_diag.json`): SD of per-Sentrix-chip medians of A″ — UCLA 0.012, Munich 0.012, Uppsala 0.014, **Karolinska 0.020** (against a total within-lab SD of 0.024: most of Karolinska's spread is between chips). Centring each chip (≥ 3 arrays) on its own median: Karolinska tail 0.098 → **0.041**; UCLA 0.020, Munich 0.035, Uppsala 0.036; within-lab SD falls to 0.017–0.018 in every lab. **The residual beyond the laboratory zero is the chip, and a per-laboratory constant cannot touch it** (as stated in Issue 003 s3.5). A chip term in production needs a reference on the chip — a control array per chip, or the lab's panel spread across its chips — which is a design decision, not an analysis.

**Per the seal, row 5 is BUILT, not commissioned.** What passes: the departure is one honest number on the identity gauge, its keys reach the report, it refuses without a zero, and healthy reads inside on real and synthetic blood. What does not: at p95 the empirical false-alarm rate is 4–10 % by laboratory, driven by chip.

**Decision for the author (PROC-MAHA-02 to be sealed on it):** (a) commission with the per-laboratory tail printed on the report ("at this laboratory, 1 healthy in 10 reads beyond p95 on the immune axis") and open row 5b, the chip term; (b) require a chip reference (control array per chip) before commissioning; (c) derive σ per laboratory from its 40-panel (CLSI-style) — noting 40 arrays estimate a spread poorly (±11 %).

---
**SEALED** sha256 `129686346d36e08dd584f10492dc35b64e2c76bc6cf25eacddd9b212d09b3a5f` · 2026-09-21
