# VAL-093 Cohort Manifest

| Cohort | n_breast_10yr | n_HC | Platform | Source | Loyfer atlas SHA-256 prefix | Beta SHA-256 prefix |
|---|---|---|---|---|---|---|
| GSE51057 | 11 | 177 | 450K | EPIC-Italy nested case-control (Phase 9) | 4b97dd2a8ba7bf41 | 8d7363bf520a74ab |
| GSE51032 | 36 | 424 | 450K | EPIC-Italy nested case-control (Phase 12) | 4b97dd2a8ba7bf41 | extracted 2026-04-26 from GSE51032 series matrix |

**Sample inclusion criteria:**
- Breast cases: cancer_site = "c50" (ICD-10 breast malignant neoplasm) AND ttd_years > 10
- HC: group = "control" (no cancer diagnosis at any follow-up)

**CHK-3.2 cross-cohort baseline:** all 25 Loyfer tiles match between GSE51057 HC and GSE51032 HC at <0.25 anchor-SDs. The cleanest cross-cohort baseline alignment in the cookbook to date. Both cohorts are EPIC-Italy nested case-control, 450K platform, same preprocessing pipeline.

**Pre-registration SHA-256:** `9b708a3a05447ed6ce5eb18174599647be30127f669e80eed16bad32fe0ed9f8`
**Sealed:** 2026-04-26T18:51:17Z (before any β access)
**Run completed:** 2026-04-26T19:04:01Z
