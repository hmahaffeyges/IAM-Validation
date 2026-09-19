# Testing and Code — every validation run

Every pre-registered validation (VAL) in the cellular track, with its prereg, outcome and results where they exist, split by whether it ran before or after the IAM Atlas was built (2026-05-28; first Atlas-based runs 2026-06-03).

| folder | what |
|---|---|
| [`VAL_INDEX.csv`](VAL_INDEX.csv) | **the map** — all 103 VAL identifiers: title, date, era, cohorts, stated decision, record completeness, path |
| [`VAL_PreAtlas/`](VAL_PreAtlas/) | runs before the Atlas (VAL-037 … VAL-141 numbering, April–May 2026): the TCGA tumour/adjacent-normal work, the breast pre-diagnostic anchors' first form, CRC, AD, the cross-population runs. Restored VAL-047 is here |
| [`VAL_PostAtlas/`](VAL_PostAtlas/) | runs on the Atlas (CPG_VAL_001 … 022, June 2026): breast per-cell fan-out, Mahalanobis hull, AIBL, immune ageing, plus `foundation_cohort/` — the **sealed 115-cell anchors** (GSE51032, GSE51057; v1 of 2026-05-29 and `anchors_v2_chrXremoved/` of 2026-09-19) — and `post_build_evidence_reports/` |
| [`Cohorts_and_Manifests/`](Cohorts_and_Manifests/) | GEO/TCGA manifests, matched-pair lists, the `extract_*.py` scripts, `ad_immune_cohorts/`, `breast_epic_cohorts/`, `cross_population/`, `CASCADE_SUMMARY` |
| [`DETAILED_VALIDATION_RECORD.md`](DETAILED_VALIDATION_RECORD.md) | the narrative record, per study, caveats first |
| `README_validation_runs_original.md` | the original `validation_runs/` README, kept verbatim |

Outcome files use pre-registered codes (`O1_PRIMARY_VALIDATED`, `O2_…`, `O3_INVERTED`, `NULL`, `DIRECTIONAL`) and a `**Status:**` line; older files state the verdict in prose. The index does not paraphrase verdicts — where the file has no explicit code it says *see file*.

How these runs relate to the current instrument, and which of them were reproduced from raw data in September 2026 (the foundation-cohort anchors: r = 1.00000 on 648 samples), is in Issue 003 §10 and Appendix V.
