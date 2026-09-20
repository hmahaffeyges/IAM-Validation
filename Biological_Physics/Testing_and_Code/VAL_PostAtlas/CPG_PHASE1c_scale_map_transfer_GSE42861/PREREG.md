# PREREG — PHASE 1c: does the Stage-1 → Roadmap scale map and the identity-loci band transfer to a cohort they have never seen?

**Written and sealed:** 2026-09-20, before any GSE42861 IDAT was downloaded or any β read. **Owner:** Heath W. Mahaffey · **Analyst:** Claude Science.
**Why:** Phase 1 (GSE87571) fit the map and built the band on the same 560 donors, so its P3 pass was a consistency check. Phase 1c makes it a test. It also sizes the LAB layer: four GSE42861 samples in TEST_DATA read ABOVE the mapped band by 0.01–0.02 A (Phase 1 P4); here we learn whether that is the cohort or those four arrays.

## Cohort
GSE42861 (Liu 2013, Karolinska EIRA): 689 whole-blood (PBL) 450K, raw IDATs (`GSE42861_RAW.tar`, 5.7 GB). **Only the 335 `disease state: Normal` controls are used.** The 354 RA arrays are NOT opened in this phase (commissioning rule: no disease labels until the chain is accepted). Ages 18–70 (bulk 40–69), f 492 / m 197, smoking recorded. Same population (Sweden) as GSE87571 — so this is a **lab / cohort transfer**, not a population transfer (that is Phase 1b).

## Frozen instrument — nothing here is fit on GSE42861
Stage 0/1 as shipped (methylprep noob, `meta['pipeline']='stage1_noob_450K'`). Walther at HEAD, contrast off. `beta_scale_maps_v1.json: stage1_noob_450K` (slope 1.0127, intercept 0.0662) **as fit on GSE87571**. Identity loci v1_0; immune H_min 0.838889; joint component on progenitor's floor. Band = `scale_map_addendum.json: mapped_band` (GSE87571, mapped A, per decade p10/p90). §108: immune band + gauge; progenitor + stem_adult one component; others composition-only. Presence gate for band eligibility restated per the Phase 1 lesson: immune + haematopoietic-progenitor ≥ 0.85 and epithelial ≤ 0.02.

## Pass conditions (fixed)
- **P1** Stage 0/1 pass ≥ 95 % of the 335 controls.
- **P2** presence gate ≥ 90 % (restated bar; Phase 1 was 76.8 % on the old bar).
- **P3 — the test.** Median mapped immune A of GSE42861 controls within **±0.02** of 1.000 (Phase 1 gave 0.990 on the fitting cohort). This is the map transferring.
- **P4** ≥ 80 % of controls read IN the GSE87571 mapped band for their decade (decades with n ≥ 30 on both sides). This is the band transferring.
- **P5 — lab layer.** The offset of the GSE42861 control median from the GSE87571 median, per decade, reported as a number. If P4 fails while P3 passes, that offset IS the lab layer and is recorded as such — not as disease.
- **P6** the four TEST_DATA arrays (GSM1051525/26 RA, GSM1051533/34 control) read within the GSE42861 control distribution (between its p5 and p95) — i.e. their "above band" in Phase 1 was their cohort, not those arrays.
- **N-random** on a mean-β-matched random panel of the same size, P4 must FAIL (< 50 % in band).
- **N-sex** |d| per decade reported; **N-smoke** immune A by smoking status reported (never / ex / current), no pass bar.
- **N-plate** Sentrix ID from IDAT filenames if present; immune A variance between vs within chips (F-test); reported.

## What each outcome means
P3 pass + P4 pass → map and band transfer within population; commission Stage 1s; proceed to switch the gauge (row B). P3 pass + P4 fail → the map is right, the band is lab-specific by the P5 amount → band needs a lab term or a wider healthy reference; still commission 1s. P3 fail → the map does not transfer; the offset is not a single pipeline constant; 1s stays PROVISIONAL and Phase 1c becomes the fitting cohort for a two-cohort map. Recorded as written, whichever it is.

---
**SEALED** sha256 `9adb8c587e56c396f7283d6796419ec2ea956cb49c74b1ba2214fe79baa0606c` · 2026-09-20, GSE42861_RAW.tar not yet downloaded.
