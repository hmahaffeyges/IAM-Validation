# Disease × Cell Signature Matrix — Engine Schema v1.2

**Date:** 2026-05-10 (last cell-value refresh: 2026-05-29 to v1.4)
**Schema version:** v1.2 (structure unchanged since 2026-05-10 — matrix at v1.4)
**Files:**
- `disease_cell_signature_matrix_v1_4.csv` — engine-readable canonical matrix (current)
- `disease_cell_signature_matrix_v1_4.md` — human-readable rendering of the same data (rendering not yet refreshed for v1.4 cell-value additions)

## CSV column structure

**8 metadata columns:**

| Column | Type | Description |
|---|---|---|
| `disease_id` | string | Disease/condition identifier (snake_case) |
| `phase` | enum | `long_pre_dx` / `mid_pre_dx` / `mid_late_pre_dx` / `near_dx` / `at_dx` / `tumor_tissue` / `active` / `active_ccfDNA` / `chronic` / `pre_malignant_*` / `treatment` / `context` |
| `time_range` | string | Human-readable time range description |
| `substrate` | string | Cohort substrate (whole_blood_buffy_coat / plasma_cfDNA / tumor_tissue_paired / tumor_tissue / cultured_pulmonary_endothelial / aortic_tissue / whole_blood_sorted / etc.) |
| `disease_severity_class` | enum | `NORMAL_VARIANT` / `PRE_DIAGNOSTIC` / `PRE_MALIGNANT` / `ACTIVE` / `ACTIVE_TUMOR` / `URGENT_CLINICAL` |
| `mechanism` | string | Signature pattern type (pooled_positive_distributed / pooled_null_directional_pass_AD_instance / compartment_flip_CCL019 / substrate_restricted_ccfDNA / etc.) |
| `organ_pages_to_link` | comma-list | `class:page_id,class:page_id,...` for cross-class linking |
| `evidence_anchors` | string | Full VAL ID list supporting the signature row |

**123 cell-type columns** organized by architectural class:
- 29 immune (B-lineage 4, T-lineage 8, NK 1, granulocytes 4, monocyte/macrophage 6, dendritic 1, pooled 3, Stage 1 readouts 2)
- 12 progenitor (MPP, L-MPP, CMP, GMP, MEP, erythroblast, erythroid_progenitor, nRBC, megakaryocyte, OPC, NeuIm, HSPC_pooled)
- 12 terminal (cortical_neurons, neurons_pooled, oligodendrocytes, cardiomyocytes, left_atrium, kera_diff, epi_upper, NeuMa, brain_pooled, skeletal_muscle, glia, astrocytes)
- 28 secretory (breast 5: ductal/lobular/secretory_pooled/LE/BE; liver 3; pancreatic 5; prostate 6: LE/BE/Fib/EC/SM/Leu; gastric 4; thyroid; esophageal_glandular; skin_melanocytes; head_neck_larynx_secretory; salivary)
- 23 cycling (gastric 4 regions; colon_epithelial; small_intestine; rectal_epithelium; esophagus 4 sub-tiles; lung 2; kidney; bladder 2; cervix; prostate_basal_cycling; breast_basal_cycling; skin_keratinocytes; pancreatic_ductal_cycling; head_neck_larynx; epithelial_cycling_pooled)
- 17 stromal (general 5; cardiac 4; pancreatic_stellate; hepatic_stellate; brain_astrocytes; bone/marrow 2; placenta; mammary_stromal; stromal_pooled)
- 1 stem_adult (HSC)
- 1 stem_pluri (stem_pluri)

## Cell value encoding

```
Empty                  -> no documented signature
'+1.26' or '-0.33'     -> single magnitude (signed Cohen's d)
'+0.5/+1.0'            -> magnitude range (lower/upper bound of effect window)
'↑' / '↓' / '↑↑' / '↓↓' -> directional only (magnitude pending)
```

NO `INV:` prefix. Bidirectional cancellation pattern is captured in the `mechanism` metadata column, not in cell values.

## Engine consumption

```python
import pandas as pd
import numpy as np

matrix = pd.read_csv('disease_cell_signature_matrix_v1_4.csv')

# Customer profile is a 123-dim vector keyed by cell column names
# customer_profile = {col: a_score for col in cell_cols}

cell_cols = [c for c in matrix.columns if c not in metadata_cols]

def compute_match_magnitude(customer, signature_row):
    """Mahalanobis-style match magnitude.
    Returns higher value when customer profile aligns with documented signature."""
    populated = [(c, signature_row[c]) for c in cell_cols 
                 if signature_row[c] and not signature_row[c].startswith(('↑','↓'))]
    if not populated:
        return 0.0
    
    matches = []
    for col, sig_str in populated:
        # Parse signature value (handle range like "+0.5/+1.0")
        if '/' in sig_str:
            lo, hi = sig_str.split('/')
            sig = (float(lo) + float(hi)) / 2
        else:
            sig = float(sig_str)
        
        cust = customer.get(col, 0.0)
        # Match contribution: sign-aligned product
        if (sig > 0 and cust > 0) or (sig < 0 and cust < 0):
            matches.append(min(abs(cust), abs(sig)))
        else:
            matches.append(-min(abs(cust), abs(sig)))
    
    return sum(matches) / np.sqrt(len(matches))

# For each row, compute match. Top-ranked rows become candidate signatures.
matrix['match'] = matrix.apply(
    lambda r: compute_match_magnitude(customer_profile, r), axis=1
)
top_candidates = matrix.nlargest(3, 'match')
```

## Tier computation (engine-side, NOT pre-baked into matrix)

```python
def compute_customer_tier(match_magnitude, severity_class, phase, evidence_strength):
    """Engine computes customer-tier from match × phase × severity × evidence."""
    if severity_class == 'NORMAL_VARIANT':
        return 'NORMAL'
    
    if match_magnitude < 0.5:
        return 'NORMAL'
    elif match_magnitude < 1.0:
        return 'MARGINAL'
    elif match_magnitude < 1.5:
        return 'ELEVATED'
    elif match_magnitude < 2.0:
        return 'SIGNIFICANTLY_ELEVATED'
    else:
        # Phase modulation
        if phase in ['active', 'active_ccfDNA', 'tumor_tissue'] and severity_class == 'URGENT_CLINICAL':
            return 'URGENT'
        return 'SIGNIFICANTLY_ELEVATED'
```

## Maintenance protocol

1. Each cell value must trace to a specific evidence anchor (VAL ID, CCL ID, or canonical-document citation).
2. Adding a new disease row requires populating at least the metadata columns + at least one cell value.
3. Adding a new cell column requires the column to be a real production atlas cell type per `cell_type_inventory.md` (no fabricated names).
4. Magnitude updates from new research require updating the `evidence_anchors` field with the new VAL ID.
5. The matrix version increments on cell-value changes (v1.2 → v1.3 → v1.4); column structure changes increment major version (v2.0). The engine schema version stays at v1.2 across cell-value refreshes; the schema only versions on structural changes (column additions/removals/renames or value-encoding changes).
6. Mechanism codes are documented in the markdown rendering — extend with new codes as new patterns emerge.
