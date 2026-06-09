# Disease × Cell Signature Matrix

**Location:** `Biological_Physics/atlas_vault/walther_clinical_runtime/DISEASE_MATRIX/`
**Current matrix version:** v1.8 (2026-06-07)
**Engine schema version:** v1.2 (stable)
**README date:** 2026-06-07

This folder contains the disease signature lookup table CPG's Stage 8 Path B will consume once the per-patient matching engine is wired. The matrix itself is operational; the per-patient engine that calls into it is specced and not yet implemented.

---

## What the matrix is

A lookup table of **signed Cohen's d values** indexed by (disease × phase) on rows and (cell-type) on columns. When the Stage 8 Path B engine is wired, each patient's 115-cell A-score profile will be matched against each row of this matrix to find documented disease/condition signatures that look like theirs.

- **81 rows** (disease × phase combinations, 1 added in v1.7)
- **131 columns** (8 metadata + 123 cell-type)
- Each cell is a signed Cohen's d like `+1.26`, a range like `+0.5/+1.0`, or an empty cell ("no documented signature, not zero")

The matrix is paired with an engine schema document that defines how Stage 8 Path B will consume it.

---

## Files in this folder

| File | Role |
|---|---|
| `disease_cell_signature_matrix_v1_8.csv` | The current lookup table (81 rows × 131 columns) |
| `disease_cell_signature_matrix_engine_schema_v1_2.md` | The contract — defines column structure, value-encoding rules, the `compute_match_magnitude()` function spec, the `compute_customer_tier()` mapping, and the maintenance protocol |
| `iamatlas_115_to_matrix_v1_7_mapping.json` | **NEW 2026-06-05.** Stage 8 Path B mapping artifact v0.1 — maps the 115 IAMAtlas cell-type names to the 123 matrix cell-type columns. 58 atlas cells mapped (50.4% coverage); 49 matrix columns with at least one atlas contributor (39.8% coverage). v0.1 is a STARTER — v0.2 manual taxonomy curation outstanding for remaining unmapped cells. |
| `README.md` (this file) | Folder orientation, version log, current state |
| `OLD/` | Archived prior versions (v1.0 through v1.7 of the matrix; v1.1 of the schema) |

---

## Rows operationally validated by the current VAL series

These 5 rows are anchored by our own CPG-VAL-NNN runs with the current production methodology — NOT by pre-build external panels:

| Disease × phase row | Headline cell values (from current methodology) | Anchor VALs |
|---|---|---|
| `breast_cancer / long_pre_dx_post_build_v3_0` (NEW IN v1.7) | Basophil +1.01/+1.58, breast_epithelial +0.61/+1.28, T-cells −0.58/−0.67, Mahalanobis +1.876/+2.097 | CPG-VAL-001/002/003/004(R)/005/006(R)/007 |
| `alzheimers_disease / at_dx_post_build_v3_0` | Eosino A −0.43/−0.46, Mahalanobis +0.20 (modest, targeted), Rule A panel AUC 0.84 | CPG-VAL-008/009/010/011/012/013 |
| `frontotemporal_dementia / post_build_GIFT_2026` | Mahalanobis +0.28 (intermediate between AD and PSP/CBD) | CPG-VAL-014 |
| `progressive_supranuclear_palsy_CBD / post_build_GIFT_2026` | Mahalanobis −0.38 (BELOW_NORMAL direction — architectural compaction, opposite of AD) | CPG-VAL-014 |

The remaining 77 rows are pre-build literature compilations held as look-up entries for the Stage 8 Path B engine to use when it is wired. They have not been re-validated with the current methodology — they are audit lineage / future-validation targets.

---

## How Stage 8 Path B will work (when wired)

For each patient:

1. Patient's 115-cell A-score vector is computed by Stages 4–5 of the SOP.
2. For each row in the matrix, the engine computes `compute_match_magnitude(patient_vector, row)` — a Mahalanobis-style sign-aligned weighted product (NOT raw dot product — see schema §3.4).
3. Rows with match magnitude above the calibrated threshold are returned as candidate matches.
4. `compute_customer_tier()` maps the top match magnitude to a customer-facing tier (NORMAL / WATCH / FLAG).
5. The report renders the top matches with their disease × phase labels and the patient's specific cell-type departures that drove the match.

The engine is specced in `disease_cell_signature_matrix_engine_schema_v1_2.md`. Implementation requires building the **cell-name-to-matrix-column mapping artifact** — the 115 IAMAtlas cell-type names overlap with but are not identical to the 123 matrix column names. This artifact is the gating dependency for Stage 8 Path B activation.

Stage 8 Path A (card-driven) is currently the operational path for per-patient reporting on the two validated cards (breast-epic v3.1, ad-immune v3.1).

---

## What this matrix does NOT do (yet)

1. **Per-patient matching is not wired.** The matrix is read-only data right now. The `compute_match_magnitude()` engine is specced. **NEW 2026-06-05:** The cell-name-to-matrix-column mapping artifact v0.1 starter is now available at `iamatlas_115_to_matrix_v1_7_mapping.json` (50% atlas coverage). v0.2 manual taxonomy curation + engine implementation outstanding.
2. **The 76 pre-build rows are not validated with current methodology.** They serve as look-up entries pending future CPG-VAL-NNN anchoring.
3. **No customer-facing reports use the matrix yet.** All current per-patient output goes through Stage 8 Path A (card-driven, where the card directly asserts the disease pattern).

---

## Version log

| Version | Date | Change |
|---|---|---|
| v1.0–v1.4 | Pre-build era | Initial matrix compilation from literature anchors |
| v1.5 | 2026-06-02 | Citation alias bump for the `breast_cancer / long_pre_dx` row — added "/ CPG-VAL-NNN" tags next to the prior TODO citations. No cell value changes. |
| v1.6 | 2026-06-03 | Added 3 new rows: `alzheimers_disease / at_dx_post_build_v3_0`, `frontotemporal_dementia / post_build_GIFT_2026`, `progressive_supranuclear_palsy_CBD / post_build_GIFT_2026`. Cell values from CPG-VAL-008 through CPG-VAL-014. |
| v1.7 | 2026-06-05 | Strict additive — adds new clean breast row `breast_cancer / long_pre_dx_post_build_v3_0` mirroring the AD v1.6 pattern. Cell values from CPG-VAL-001 through CPG-VAL-007 with current production methodology only (no pre-build VAL references in the new row's evidence_anchors). The original `breast_cancer / long_pre_dx` row (row 1) is retained verbatim as audit lineage — it contains pre-build VAL-046/047/049/093/094/095/096 references which the new v3.0/v3.1 row supersedes operationally. v1.6 archived at `OLD/disease_cell_signature_matrix_v1_6.csv`. |
| **v1.8** | **2026-06-07** | **Strict additive evidence_anchor refresh — zero cell-value changes, zero new rows, zero schema changes.** Surgical text appends to 5 existing rows' `evidence_anchors` field incorporating findings from CPG-VAL-014 through CPG-VAL-022 (immune-card sprint) plus two substantive expansion appends from CPG-VAL-001 and CPG-VAL-011. Row 6 (`breast_cancer / long_pre_dx_post_build_v3_0`): append CPG-VAL-016 cross-cohort directional finding (d=+0.69 GSE51057 / d=+0.22 GSE51032, hypermethylation) + CPG-VAL-001 expansion (top-10 spans all 8 architecture classes with 7 cells quoted). Row 51 (`alzheimers_disease / active`): append CPG-VAL-016 (AD d=−0.36 hypomethylation, informative sign-opposition to breast) + CPG-VAL-019 (AIBL bidirectional decomposition PASS d_up=+0.494, d_down=−0.515, operationalizes bidirectional doctrine). Row 77 (`normal_aging / chronic`): append CPG-VAL-015 (Hannum n=656 PASS r=−0.197 p=3.69e-07, sex-symmetric) + CPG-VAL-020 (Hannum full-chain reproduction SEALED commit 4c22f8e) — augments the pre-existing VAL-006 Hannum anchor (Pearson r=0.9999, age-to-A=1.05 extrapolates to ~1,075 yr) with the CPG-era immune-class aging trajectory and full-chain reproduction. Row 78 (`inflammaging / chronic`): append CPG-VAL-017 (pooled-cross-cohort NULL r=+0.034 p=0.150, informative — bidirectional doctrine applies, never use pooled A_immune as sole metric). Row 79 (`alzheimers_disease / at_dx_post_build_v3_0`): append CPG-VAL-014 (GIFT GSE53740 AD d=+0.681, three-cohort cross-platform replication) + CPG-VAL-019 (AIBL bidirectional confirms fanout architecture is bidirectional not unidirectional) + CPG-VAL-011 expansion (per-class age subtraction reveals masked hematopoietic signatures: stem_adult d=−0.004 → −0.190, progenitor effect doubles). The other 76 rows are byte-identical to v1.7. v1.7 archived at `OLD/disease_cell_signature_matrix_v1_7.csv`. The `iamatlas_115_to_matrix_v1_7_mapping.json` mapping artifact remains valid for v1.8 because the matrix column structure did not change. |

---

## Companion documents

- Engine spec: `disease_cell_signature_matrix_engine_schema_v1_2.md` (this folder)
- Card-level evidence: `DISEASE_MAPS_CARDS/Breast_EPIC/breast_epic_card_json/breast-epic_README.md` and `DISEASE_MAPS_CARDS/AD_immune/ad_immune_card_json/ad-immune_README.md`
- Card JSONs (operational): `breast-epic_card_v3_1.json`, `ad-immune_card_v3_1.json`
- Top-level evidence report: `post_build_evidence/v5_CPG_IAMAtlas_Evidence_Report.html` Sections 3–4
- Inventory catalog: `post_build_evidence/v8_CPG_VAL_Inventory_Report.md` Section 4
