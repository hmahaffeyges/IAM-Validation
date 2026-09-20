# OUTCOME — PHASE 1: identity-loci healthy bands from GSE87571

**Run:** 2026-09-19 22:14 → 2026-09-20 03:23 PDT, `phase1_run.py`, methylprep noob Stage 1, Walther HEAD (contrast off), identity-loci H(β̄)/H_min.
**Against:** PREREG.md sealed sha256 `4d8f40ace72eca1f…` (tar unopened at sealing). No condition was changed after data were seen.
**Input:** GSE87571_RAW.tar sha256 `3245d0e59a031546…`, 732 IDAT pairs, GPL13534, all whole blood, age known 729, F 389 / M 341.

## Verdict: **FAIL** on P3 and P4 (as defined). P1 pass. P2 miss (threshold, not substrate). Band built and kept.

| condition | sealed bar | observed | verdict |
|---|---|---|---|
| P1 coverage | ≥ 95 % pass Stage 0/1 | 732/732, 0 failures | PASS |
| P2 presence | immune ≥ 0.80 and epi ≤ 0.02 in ≥ 95 % | 76.8 %; epi max 0.046, immune min 0.675, immune median 0.874, joint haematopoietic-progenitor median 0.120 | MISS — the 0.80 immune bar is too tight once ~12 % sits in the joint component; no sample is off-substrate |
| P3 synthetic healthy in band | ≥ 14/16 | **0/16** — synthetic A 0.984 ± 0.003 vs band 0.78–0.87 | **FAIL** |
| N-random | synthetic must NOT read in band on a random panel | 0/16 | PASS (P3's result depends on the loci) |
| P4 test samples | ≥ 6/7 whole blood in band; tissue outside or flagged | 3/7 (the 4 out are the RA-study arrays at A 0.87–0.89, above p90); tissue not scored (P2 fails them, correctly) | **FAIL** |
| P5 age trend | report ρ | Spearman ρ = +0.523, p = 1e-40: immune A rises with age | reported |
| N-sex | \|d\| < 0.2 every decade n ≥ 30 | 0.05, 0.06, **0.44, 0.30, 0.24**, 0.15, 0.10 | FAIL in three decades → sex-split band indicated |
| N-split | p10/p90 agree ≤ 0.01 | max 0.0177 (one cell), others ≤ 0.013 | marginal |
| N-plate | if Sentrix recoverable | not attempted in this run | SKIPPED — to do |

## The band (immune, identity loci, Stage 1 β, n = 560 after P2; THIN = n < 30)
| decade | n | p10 | p50 | p90 |
|---|---|---|---|---|
| 14–24 | 112 | 0.760 | 0.798 | 0.834 |
| 25–34 | 58 | 0.784 | 0.815 | 0.846 |
| 35–44 | 81 | 0.790 | 0.825 | 0.864 |
| 45–54 | 97 | 0.795 | 0.831 | 0.866 |
| 55–64 | 61 | 0.798 | 0.836 | 0.870 |
| 65–74 | 87 | 0.807 | 0.843 | 0.886 |
| 75–84 | 53 | 0.826 | 0.851 | 0.890 |
| 85–94 | 11 | 0.816 | 0.852 | 0.877 (THIN) |

`identity_band_v1.json` carries this plus the joint haematopoietic-progenitor band. It is valid **for Stage-1-calibrated whole blood of this population** and is kept as the first healthy immune band built on the correct statistic from one cohort through one pipeline.

## Why P3 failed — the finding
| | β̄ on the 42,134 immune identity loci | A |
|---|---|---|
| Swedish healthy blood, Stage 1 (n = 560, median-implied) | **0.8145** | 0.825 |
| synthetic healthy = pure Atlas immune posterior | **0.7411** | 0.984 |
| Atlas immune_mean on the same loci | 0.737 | — |
| `H_min_beta` (identity file; A ≡ 1 there) | 0.7318 | 1.000 |

The Atlas places a healthy immune cell at β ≈ 0.737 on the identity loci; Stage-1 noob places real healthy blood at β ≈ 0.81. That +0.07 is the offset first seen on 3 donors this afternoon (PROC-CAL-01 note), now measured on 560. The gauge statistic and loci are not at fault, and the band is a genuine healthy band — **but the Atlas β scale (on which H_min was derived, from Roadmap/ENCODE processed β) and the Stage-1 β scale (on which patients arrive) differ systematically at the identity loci.** No single band can contain both.

## What this decides, and what it leaves to the author
- The production gauge as wired (marker union + its band) remains withdrawn (PROC-N7-01). The identity-loci gauge now has a real band, on Stage-1 β, for Northern-European whole blood.
- **Open, author's decision (touches the floor):** which β scale is the reference. (a) Re-derive `H_min_beta` per class on Stage-1 healthy blood — for immune this is ≈ 0.703 (recorded 2026-09-19), making the floor "Stage-1 native"; or (b) place a per-array-type scale correction between Stage 1 and the gauge so patient β lands on the Atlas scale. (a) redefines a physics constant's *input*; (b) adds a calibration step. Analyst's read: define the floor on the pipeline patients actually arrive through.
- **Phase 1b** (cross-population, EPIC-Italy controls through the same Stage 1) proceeds regardless; it tests the band's portability, which is independent of the scale question.
- P2's immune threshold should be restated as immune + joint ≥ 0.85 in the next prereg; N-sex indicates a sex-split band for ages 35–64; N-plate remains to do.

*Nothing here is validated for patient care. This is a healthy-reference calibration on public data.*

---
**SEALED** sha256 `3d04c118033859664f695e00d62a60e6598f0eb5b17fe9ec1b72c51c0756cc6b` · 2026-09-20
