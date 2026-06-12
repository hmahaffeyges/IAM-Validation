# Disease × Cell Signature Matrix

**Location:** `Biological_Physics/atlas_vault/walther_clinical_runtime/DISEASE_MATRIX/`
**Current version:** v1.6 (2026-06-03)
**README date:** 2026-06-04 (clean rewrite)

This folder contains the disease signature lookup table CPG's Stage 8 Path B will consume once the per-patient matching engine is wired. The matrix itself is operational; the per-patient engine that calls into it is specced but not yet implemented.

---

## What the matrix is

A lookup table of **signed Cohen's d values** indexed by (disease × phase) on rows and (cell-type) on columns. When the Stage 8 Path B engine is wired, each patient's 115-cell A-score profile will be matched against each row of this matrix to find documented disease/condition signatures that look like theirs.

- **80 rows** (disease × phase combinations)
- **131 columns** (8 metadata + 123 cell-type)
- Each cell is a signed Cohen's d like `+1.26` or a range like `+0.5/+1.0` or an empty cell ("no documented signature, not zero")

The matrix is paired with an engine schema document that defines how Stage 8 Path B will consume it.

---

## Files in this folder

| File | Role |
|---|---|
| `disease_cell_signature_matrix_v1_7.csv` | The current lookup table (80 rows × 131 columns) |
| `disease_cell_signature_matrix_engine_schema_v1_2.md` | The contract — defines column structure, value-encoding rules, the `compute_match_magnitude()` function spec, the `compute_customer_tier()` mapping, and the maintenance protocol |
| `README.md` (this file) | Folder orientation, version log, current state |
| `OLD/` | Archived prior versions (v1.0 through v1.5 of the matrix; v1.1 of the schema) |

---

## Rows operationally validated by our card work

These four rows are anchored by our own CPG-VAL-NNN runs with the current methodology (not pre-build evidence):

| Disease × phase row | Headline cell values | Anchor VALs |
|---|---|---|
| `breast_cancer / long_pre_dx` | Basophil +1.58/+1.01, breast_BE +1.28/+0.61, T-cells −0.67/−0.58, Mahalanobis +1.88/+2.10 | CPG-VAL-001, 002, 005, 007 |
| `alzheimers / at_dx_post_build_v3_0` | Eosino −0.43/−0.46, 20-of-115 immune cells Bonferroni-negative, Mahalanobis +0.20 (modest, targeted) | CPG-VAL-008, 009, 010, 012, 013 |
| `frontotemporal_dementia / post_build_GIFT_2026` | Mahalanobis +0.28 (intermediate between AD and PSP/CBD) | CPG-VAL-014 |
| `progressive_supranuclear_palsy_CBD / post_build_GIFT_2026` | Mahalanobis −0.38 (BELOW_NORMAL — architectural compaction direction, opposite of AD) | CPG-VAL-014 |

The remaining 76 rows are held in the matrix as look-up entries compiled from prior literature for the Stage 8 Path B engine to use when it is wired. They have not been validated with our current methodology.

---

## How Stage 8 Path B will work (when wired)

For each patient:

1. Patient's 115-cell A-score vector is computed by Stages 4–5 of the SOP.
2. For each row in the matrix, the engine computes `compute_match_magnitude(patient_vector, row)` — a Mahalanobis-style similarity between the patient's profile and the row's documented signature.
3. Rows with match magnitude above the calibrated threshold are returned as candidate matches.
4. `compute_customer_tier()` maps the top match magnitude to a customer-facing tier (NORMAL / WATCH / FLAG).
5. The report renders the top matches with their disease × phase labels and the patient's specific cell-type departures that drove the match.

The engine is specced in `disease_cell_signature_matrix_engine_schema_v1_2.md`. Implementation requires building the **cell-name-to-matrix-column mapping artifact** — the 115 IAMAtlas cell-type names overlap with but are not identical to the 123 matrix column names. This artifact is the gating dependency for Stage 8 Path B activation.

---

## What this matrix does NOT do (yet)

1. **Per-patient matching is not wired.** The matrix is read-only data right now. The `compute_match_magnitude()` engine is specced; implementation is outstanding.
2. **The 76 non-anchored rows are not validated with our current methodology.** They are compiled from prior literature and serve as look-up entries pending future CPG-VAL-NNN anchoring.
3. **No customer-facing reports use the matrix yet.** All current per-patient output goes through Stage 8 Path A (card-driven, where the card directly asserts the disease pattern).

---

## Version log

| Version | Date | Change |
|---|---|---|
| v1.0–v1.4 | Pre-build era | Initial matrix compilation from literature anchors |
| v1.5 | 2026-06-02 | Citation alias bump for the `breast_cancer / long_pre_dx` row — added "/ CPG-VAL-NNN" tags next to the prior TODO citations. No cell value changes. |
| v1.6 | 2026-06-03 | Added 3 new rows: `alzheimers / at_dx_post_build_v3_0`, `frontotemporal_dementia / post_build_GIFT_2026`, `progressive_supranuclear_palsy_CBD / post_build_GIFT_2026`. Cell values from CPG-VAL-008 through CPG-VAL-014. |
| README rewrite | 2026-06-04 | README rewritten clean. Dropped extensive pre-build-era "Salas 2018 / Loyfer 2023 / Moss 2018 / EpiSCORE / UniLIFE / Xu-538" lineage documentation that lived in the operational sections of the prior README (moved to this version log as historical lineage). Operational sections now describe what the matrix is and what Stage 8 Path B will do, with the four operationally-anchored rows highlighted. |

---

## Companion documents

- Engine spec: `disease_cell_signature_matrix_engine_schema_v1_2.md` (this folder)
- Card-level evidence: `DISEASE_MAPS_CARDS/Breast_EPIC/breast_epic_card_json/breast-epic_README.md` and `DISEASE_MAPS_CARDS/AD_immune/ad_immune_card_json/ad-immune_README.md`
- Top-level evidence report: `post_build_evidence/v5_CPG_IAMAtlas_Evidence_Report.html` Sections 3–4
- Inventory catalog: `post_build_evidence/v8_CPG_VAL_Inventory_Report.md` Section 4
