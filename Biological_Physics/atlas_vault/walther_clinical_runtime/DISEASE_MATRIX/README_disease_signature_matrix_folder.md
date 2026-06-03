# Disease × Cell Signature Matrix

**Location:** `Biological_Physics/atlas_vault/pipeline_runtime_matrices/disease_signature_matrix/`
**Purpose:** Stage 5 of the EDEAR pipeline matches every customer's per-cell-type A-score profile against this matrix to find documented disease/condition signatures that look like theirs.

The matrix is a **lookup table** (77 disease/condition × phase rows × 123 cell-type columns) of signed Cohen's d values. The schema doc next to it tells the engine HOW to consume it. Together they're a complete unit — the matrix without the schema is just unlabeled numbers; the schema without the matrix is an algorithm with no data.

## Files in this folder

| File | Role | Lifecycle |
|---|---|---|
| `disease_cell_signature_matrix_v1_5.csv` | The data. 77 rows × 131 columns (8 metadata + 123 cell-type). Each cell holds a signed Cohen's d like `+1.26` or a range like `+0.5/+1.0` or a directional `↑↑` placeholder. | Bumps on every cell-value change (v1.3 → v1.4 → v1.5 etc.). Major bump on column-structure change (v2.0). |
| `disease_cell_signature_matrix_engine_schema_v1_2.md` | The contract. Defines column structure, value-encoding rules, the `compute_match_magnitude()` Mahalanobis-style match function Stage 5 calls, the `compute_customer_tier()` mapping, the maintenance protocol. | Bumps only on structural changes. Stays stable across cell-value-only matrix updates. |
| `README.md` | This file. Folder orientation, version log, push policy. | Refreshed whenever a matrix or schema version lands. |

## Why these two files belong together

The CSV is engine-readable but cryptic without the schema: it has columns like `B_cells` and `regulatory_T_cells` and cell values like `+0.5/+1.0` and `↑↑`. Without the schema you don't know:
- That an empty cell means "no documented signature" (not "zero signal")
- That `+0.5/+1.0` is a magnitude RANGE, not a fraction
- That `↑↑` is "directional only, magnitude pending" rather than a literal up-up character
- That `mechanism = pooled_positive_distributed_multi_tile` corresponds to a specific match-pattern code
- That the `compute_match_magnitude()` algorithm is Mahalanobis-style sign-aligned product weighted by sqrt(n) — not raw dot product, not Euclidean

The schema is short (~120 lines) but load-bearing. Engine cannot consume the matrix correctly without it.

## How Stage 5 uses these

```python
# Per customer, per pipeline run:
matrix = pd.read_csv('disease_cell_signature_matrix_v1_4.csv')
customer_profile = stage_2_outputs.celltype_ascores  # 115-dim dict

# Score each row against the customer (per the schema's compute_match_magnitude)
matrix['match'] = matrix.apply(lambda r: compute_match_magnitude(customer_profile, r), axis=1)

# Top 3 candidate signatures
top_candidates = matrix.nlargest(3, 'match')

# Per-candidate tier (per the schema's compute_customer_tier)
for _, candidate in top_candidates.iterrows():
    tier = compute_customer_tier(candidate['match'], candidate['disease_severity_class'],
                                  candidate['phase'], candidate['evidence_anchors'])
```

Stage 5 returns a ranked list of documented signature matches plus their tiers. The report builder (Stage 6) translates these into customer-facing wellness-or-watch-or-act language using the schema's tier mapping.

## Version log

| Version | Date | Change |
|---|---|---|
| v1.4 → v1.5 | 2026-06-02 | breast_cancer / long_pre_dx row evidence_anchors EXTENDED with CPG-VAL-NNN citation aliases: TODO 1.1 → also cited as CPG-VAL-001 (per-cell-type fan-out), TODO 1.2 → CPG-VAL-002 (Mahalanobis d=+1.871/+2.088), TODO 1.3 → CPG-VAL-003 (1,392 concordant CpGs residual map), TODO 1.5 → CPG-VAL-005 (PC2 T-cell suppression d=−0.67/−0.58). Appended CPG-VAL-007 age-axis subtraction confirmation. **No cell values changed; no prior citations removed.** All other 76 rows byte-identical to v1.4. |
| v1.3 → v1.4 | 2026-05-29 | breast_cancer long_pre_dx row updated with TODO 1.1/1.2/1.3/1.5 findings: +11 new cell values (basophils, plasma_cells, microglia, skin_melanocytes, NeuMa, neurons_pooled, smooth_muscle, breast_BE TISSUE-OF-ORIGIN, endothelial_cells, CD4_T_cells, CD8_T_cells). evidence_anchors expanded to cite TODO 1.1 / 1.2 (Mahalanobis d=+1.871/+2.088) / 1.3 (residual map 1,392 concordant CpGs) / 1.5 (PC2 T-cell suppression d=-0.67/-0.58). |
| v1.3 | 2026-05-10 | Initial canonical v1.3 — 77 disease/condition × phase rows × 123 cell-type columns. |
| Schema v1.2 | 2026-05-10 | Structural definition: 8 metadata + 123 cell columns. Value encoding: float, range, directional. Match algorithm: Mahalanobis-style sign-aligned product. Tier mapping: customer-facing severity language. |

## Maintenance protocol

Per the engine schema, every cell value must trace to a specific evidence anchor (VAL ID or canonical-document citation). Adding new disease rows requires populating at least the metadata + at least one cell value. Adding new cell columns requires the cell type be a real production atlas cell type per `IAMAtlasREBUILD_celltype_to_class.json` — no fabricated names.

Cell-value updates from new research:
1. Update the cell value in the CSV.
2. APPEND new VAL ID to the row's `evidence_anchors` field (never replace prior anchors).
3. Bump matrix version (v1.4 → v1.5).
4. Update this README's version log.
5. Push to repo with new SHA-256 in INVENTORY.json.

## Push policy

Both files live in the repo (this folder). The engine loads them at startup from this exact location. Any cell-value change requires a push. Schema changes (structural) are rare and require both files version-coordinated.

## Distinct from the disease CARDS

The disease matrix is the GLOBAL pattern-matching layer — one row per (disease, phase, substrate) tuple, cell-level Cohen's d magnitudes, no disease-specific scoring rules.

The disease CARDS (kept in a separate folder, e.g. `cards/breast-epic/`) hold per-disease panel definitions, H_min anchors, threshold tables, demographic gates, educational-page URLs, and per-card validation evidence. Cards do per-disease scoring; the disease matrix does cross-disease signature matching. Stage 5 calls BOTH in parallel and reports both outcomes.
