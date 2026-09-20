# OUTCOME — PHASE 1c: scale-map and band transfer to GSE42861 controls

**Run:** 2026-09-20 01:33 → 03:47 PDT, `phase1c_run.py`, methylprep noob Stage 1, Walther HEAD, identity loci v1_0, map `stage1_noob_450K` (slope 1.0127, intercept 0.0662) **as fit on GSE87571**, band `mapped_band` **as built on GSE87571**. Nothing fit on GSE42861.
**Against:** PREREG.md sealed sha256 `9adb8c587e56c396…` (tar not yet downloaded at sealing). No condition changed after data were seen.
**Input:** GSE42861_RAW.tar 5.74 GB; 335 `disease state: Normal` controls only; the 354 RA arrays were not opened.

## Verdict: **P3 PASS — the pipeline map transfers. P4 FAIL — the one-lab band does not. P5 sizes the lab layer at +0.017 A.**

| condition | sealed bar | observed | verdict |
|---|---|---|---|
| P1 coverage | ≥ 95 % | 335/335, 0 failures | PASS |
| P2 presence (immune + haem-prog ≥ 0.85, epi ≤ 0.02) | ≥ 90 % | 94.0 % (n = 315) | PASS |
| **P3** median mapped immune A | within ±0.02 of 1.000 | **1.0097** (unmapped 0.857; GSE87571 affine-mapped median 0.990) | **PASS** |
| **P4** in GSE87571 band (decades n ≥ 30) | ≥ 80 % | **53.0 %** | **FAIL** |
| P5 lab layer, GSE42861 − GSE87571 median per decade | report | 25–34 +0.013 · 35–44 +0.016 · 45–54 +0.018 · 55–64 +0.016 · 65–74 +0.018 (14–24 n=4 +0.022) — **flat across age, ≈ +0.017** | recorded |
| P6 the four TEST_DATA GSE42861 arrays | within cohort p5–p95 (0.959–1.041) | 1.025, 1.022, 1.034, 1.017 — **4/4 inside**; their Phase 1 "above band" was their cohort, not those chips | PASS |
| N-random (mean-β-matched random panel, same size) | must FAIL P4 (< 50 % in band) | **67 %** in band — did not fail | **FLAG** — see reading |
| N-sex | report | \|d\| 0.64 (35–44), 0.28, 0.69 (55–64), 0.37; women higher throughout | sex-split band confirmed |
| N-smoke | report | current 1.0108 (n 83) · ex 1.0086 (100) · never 1.0107 (97) · occasional 1.0062 (34) — no smoking effect on the immune gauge | recorded |
| **N-plate** | report | Sentrix from IDAT names: **37 chips, one-way F = 4.30, p = 2.6 × 10⁻¹²** | chip effect real |

Mapped A on the 315 controls: p10 0.972 · p50 1.010 · p90 1.032 · sd 0.0245.

## Reading
1. **Stage 1s is COMMISSIONED.** The map is a property of the pipeline, not the cohort: fit on one lab, it puts a second lab's healthy blood at 1.010 against a floor of 1.000. Unmapped it reads 0.857 — the whole "healthy reads below band" story of the last two days was the pipeline scale.
2. **The lab layer is real, constant, and now has a number: +0.017 A** between these two Swedish labs, the same in every decade, with a within-cohort chip effect at p ≈ 10⁻¹². It is the size of the band's half-width, so a band from one lab pushes half of another lab over p90. This is CCL-004 (GSE53740 +2.3 SD) measured properly.
3. **N-random not failing is a design finding.** A random panel matched on mean β put 67 % of controls in band: on healthy blood the identity loci set the LEVEL (P3) but the band's WIDTH is mostly pipeline/chip variance any panel shares. Consequence: a one-lab band is too narrow by the between-lab term. The healthy reference must be built **after the pipeline map, pooled across ≥ 2 labs, with the lab/chip term estimated and disclosed** (percentile bands on the pooled mapped A; a per-lab random effect reported). Not pooled raw — that is what Phase 1's design argued against, and it was right about the pipeline term; the lab term is what pooling *after* mapping is for.
4. Sex split (35–64) stands from Phase 1 and is confirmed here. Smoking does not move the immune gauge — a clean null worth a line in the report.
5. **Nothing here is disease.** RA arrays unopened. Healthy is at the floor once the pipeline is accounted for; the residual structure is lab, chip and sex.

## Next (in commissioning order)
- Build `identity_band_v2`: GSE87571 + GSE42861 controls, mapped A, pooled per decade, sex-split 35–64, lab random effect reported; prereg first; its P3/P4 tested on a THIRD Stage-1 healthy cohort.
- Then switch the conductor's gauge from marker union to identity loci (row B), Stage 5 and Stage 6 follow.
- Phase 1b (population) unchanged: EPIC-Italy controls through Stage 1 after an EPIC Stage-1 map is fit.

---
**SEALED** sha256 `46d5359b287f2792559ffc976a4c0cbc7e4f4f51e3bf5a49714a592af9ddfbd5` · 2026-09-20
