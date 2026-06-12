# CPG-VAL-018 — DEFERRED (not executed)

**Date deferred:** 2026-06-07
**Status:** DEFERRED (cohort metadata gap)
**Original scope:** HRT effect on female immune A_immune in GSE51057

## Why deferred

Investigation of the GSE51057 EPIC-Italy cohort metadata revealed that the HRT (hormone replacement therapy) field referenced in the original card scope is **NOT present** in the available data sources:

**GSE51057 clinical metadata schema (per-sample):**
- `gsm`, `gender`, `age`, `menarche_age`, `arm`

**GSE51057 raw GEO Sample_characteristics_ch1:**
- "gender: F"
- "age: 54.322"
- "age at menarche: 14"
- (no HRT field)

The same schema applies to GSE51032 — neither EPIC-Italy cohort carries HRT exposure metadata in the provided characteristics.

## Options for completion (require Heath decision)

**Option A: Pivot question — Reproductive history effect (use available data)**
The cohorts DO carry `menarche_age`. The card-relevant question becomes "Does earlier-vs-later menarche age predict adult-life A_immune in HC women?" This is on-mission (reproductive endocrinology → immune architecture) but is a SCOPE CHANGE from the original VAL-018.

**Option B: Acquire a different cohort with HRT metadata**
Candidate cohorts to investigate:
- GSE54399 (women's lifestyle factors) — may have HRT
- KORA cohort GSE has HRT subset (~1,500 women)
- ARIES cohort has reproductive metadata
- Estimated effort: cohort acquisition + canonical 115-cell scoring (~half day)

**Option C: Defer indefinitely, drop from immune card v1.0**
Mark VAL-018 as not-applicable for v1.0; revisit in v1.1 with proper cohort. Card stands at 6 sealed VALs (015, 016, 017, 019, 020, 021) instead of 7.

## Status

VAL-018 is held in DEFERRED state. The folder name has been marked `CPG_VAL_018_HRT_Effect_DEFERRED` to make the deferral visible at-a-glance in the validation_runs directory.

No execution occurred; no provisional results were generated. Per the user-preference rule about not taking liberties on project scope, the question of which option (A, B, or C) to pursue is being routed back to Heath rather than silently re-scoped.
