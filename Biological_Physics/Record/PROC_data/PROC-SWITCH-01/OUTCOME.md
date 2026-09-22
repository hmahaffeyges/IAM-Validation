# OUTCOME — PROC-SWITCH-01: the gauge switch (as sealed)

**Run 2026-09-21.** `cpg_conductor.run_full` now reports the identity-loci gauge with the three-layer reference (`classes`), keeps the marker-union statistic as `diagnostic_marker_union`, and flags Stages 5/6 `pending_recalibration`. `identity_band_v3.json` built first from the four zeroed healthy cohorts (n = 1,379): pooled p10–p90 = 0.9724–1.0248 (width 0.0524).

| test | bar | result | verdict |
|---|---|---|---|
| S1 switch is real | 7/7 identity_loci, MAPPED, reportable with zero | 7/7 | **PASS** |
| S2 UNSET refuses | 7/7 reportable False, A_abs None | 7/7 | **PASS** |
| S3 healthy in band | ≥ 4/5 | 4/5 (GSM2333950, the 'adjudicator' array, 1.033 ABOVE) | **PASS** |
| S4 synthetic healthy, zero 0 | median within ±0.010 of 1.000 AND ≥ 70 % in band | median **0.9848**, in band 97.5%; marker-union on the same patients 1.125 | **FAIL as sealed** (median) |
| S5 separation unchanged | anchors r = 1.00000 | GSE51057 repo_head r = 1.00000 max diff 0.00004 115/115; chrX-removed r = 1.00000 max diff 0.00000; `test_a_score_canonical` PASS | **PASS** |
| S6 four cohorts in band (in-sample) | report ≈ 0.80 | {'GSE111629_UCLA': 0.828, 'GSE125105_Munich': 0.841, 'GSE42861_Karolinska': 0.775, 'GSE87571_Uppsala': 0.791} | recorded |

## The seven cached whole-blood arrays through the switched chain
| GSM | age | arm | lab zero | A_mapped | c(age) | A″ | placement | marker-union A (diagnostic) |
|---|---|---|---|---|---|---|---|---|
| GSM2333901 | 58 | healthy | -0.0117 | 0.9748 | +0.0019 | 0.9846 | IN_BAND | 0.9604 |
| GSM2333905 | 67 | healthy | -0.0117 | 0.9928 | +0.0060 | 0.9986 | IN_BAND | 0.9539 |
| GSM2333950 | 43 | healthy | -0.0117 | 1.0213 | +0.0000 | 1.0331 | ABOVE_BAND | 0.9568 |
| GSM1051533 | 60 | healthy | +0.0084 | 1.0340 | +0.0060 | 1.0196 | IN_BAND | 1.0030 |
| GSM1051525 | 60 | RA | +0.0084 | 1.0248 | +0.0060 | 1.0104 | IN_BAND | 1.0021 |
| GSM1051534 | 60 | healthy | +0.0084 | 1.0168 | +0.0060 | 1.0024 | IN_BAND | 1.0046 |
| GSM1051526 | 60 | RA | +0.0084 | 1.0216 | +0.0060 | 1.0073 | IN_BAND | 1.0064 |

## S4: why it failed, and what the failure is
The generator drops every CpG lacking a mean in all eight classes (22,548 of 483,092 kept), so the synthetic patients cover 2,745 of the 42,134 immune identity loci — checked, NOT the cause: pure atlas immune reads A = 0.9903 on the full panel and 0.9909 on the subset. With noise, age and batch effects all off, the patients still read 0.985 (`switch01_s4_diag.json`), and each patient's β is exactly its linear mixture of atlas class means (β̄ 0.7408 both ways). **The 0.015 is real and belongs to the atlas:** the G-002 floor puts A = 1 at β = 0.7318 (37 Roadmap/ENCODE reference cells); the atlas posterior's immune mean over the identity loci is 0.7373 (pure immune, A 0.990), and a whole-blood mixture with ~11 % progenitor + stem_adult (whose identity-loci means are higher) reads 0.985. The atlas and the floor are different reference sets — **the atlas is a fifth laboratory with its own constant, ≈ −0.015**, and S4 as sealed assumed it was zero. Real Uppsala healthy blood, mapped, reads 0.988; the synthetic reads 0.985; they agree. The analyst's first diagnosis ('0.998 from the mixture') used weights summing to 0.989 and was wrong; recorded.

**Per the seal, S4 fails and this procedure does not commission the switch.** PROC-SWITCH-02 re-seals S4 with the synthetic cohort read the way every other laboratory is read: zeroed from its own disjoint 40-array panel.

**What S4 did show, regardless of the bar:** the marker-union statistic reads the same healthy synthetic patients at 1.125 — every one above band, exactly the 2026-09-19 finding — and the switched gauge reads them at 0.985 with 97.5 % in band. The defect N7 caught is gone; what remains is a constant.
