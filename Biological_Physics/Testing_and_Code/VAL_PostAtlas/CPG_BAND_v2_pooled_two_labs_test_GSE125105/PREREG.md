# PREREG — identity_band_v2: pooled two-lab healthy reference on the Roadmap scale, tested on a third lab and population

**Written and sealed:** 2026-09-20, before GSE125105 was downloaded or any β read. **Owner:** Heath W. Mahaffey · **Analyst:** Claude Science.
**Why:** Phase 1c showed the pipeline map transfers (Stage 1s COMMISSIONED) but a one-lab band does not (53 % in band; lab layer +0.017 A; chip effect p ≈ 1e-12). The healthy reference must therefore be built after the map, pooled across labs, with the lab term inside the band and disclosed.

## Fitting cohorts (band built here, nothing else)
GSE87571 (Uppsala, n = 560 after gate) and GSE42861 controls (Karolinska, n = 315 after gate), Stage-1 noob β already cached, mapped with `stage1_noob_450K` (slope 1.0127, intercept 0.0662, frozen). Immune identity-loci `H(β̄)/H_min`. Band = per-decade p10 / p50 / p90 of the POOLED mapped A; decades with n ≥ 30 pooled. Reported alongside: per-lab median per decade and the between-lab offset (the lab random effect), and the pooled sd. Sex: signed d(f−m) per decade per lab, no split unless the sign agrees in both labs. Joint haematopoietic-progenitor component: same construction on progenitor's floor.

## Test cohort (never used for fitting)
GSE125105 (Max Planck Institute of Psychiatry, Munich; 450K whole blood, raw IDATs 5.7 GB): **only the 210 `diagnosis: control` samples**. The 489 depression cases are NOT opened. Ages 17–87, F/M ≈ 55/45. A different lab AND a different (German) population from both fitting cohorts.

## Pass conditions (fixed)
- **P1** Stage 0/1 pass ≥ 95 % of 210. **P2** presence gate (immune + haem-prog ≥ 0.85, epi ≤ 0.02) ≥ 90 %.
- **P3 — map transfer #2.** Median mapped immune A within ±0.02 of 1.000 (third lab, second population).
- **P4 — band transfer.** ≥ 80 % of controls IN the pooled band for their decade (decades n ≥ 30 in the band). Phase 1c one-lab result was 53 %.
- **P5** GSE125105 median minus pooled median per decade, reported: the Munich lab/population offset. If P3 passes and P4 fails, that offset is the reason and is recorded as lab/population, not disease.
- **N-random, REDESIGNED.** Phase 1c matched the random panel on mean β — which for an H(β̄) statistic pins A by construction, so a 67 % result was not a failure of the identity loci but of the null's design (recorded). Here: random panel matched on SIZE only; its median mapped A must differ from 1.000 by > 0.05 (level specificity), and its in-band fraction is reported.
- **N-sex** signed d(f−m) per decade. **N-plate** Sentrix chips, F-test, reported. **N-split** pooled band from a random half of the fitting cohorts vs the other half, p10/p90 agreement ≤ 0.01.

## What each outcome means
P3 + P4 pass → band_v2 accepted as the whole-blood 450K healthy reference for Northern/Central European populations; row B (gauge switch) unblocked. P3 pass, P4 fail → the Munich offset is a third lab term; band_v3 pools three labs and the test moves to a fourth cohort (the reference-interval maintenance loop, stated as such). P3 fail → the map does not hold across populations at this level; population enters the map, not the band. Recorded as written.

---
**SEALED** sha256 `2e0d1e399f7a455b9b2206684e3ebdd59e71213df62b83eab837ba262f2a4a80` · 2026-09-20, GSE125105_RAW.tar not yet downloaded.
