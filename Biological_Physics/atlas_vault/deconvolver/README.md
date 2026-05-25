# IAMAtlas Deconvolver

The cell-fraction estimator for EDEAR. Takes a customer methylation sample (beta values per CpG) and the IAMAtlas matrix, and returns per-cell-type fractions, per-architecture-class fractions, and residual diagnostics. This is the only deconvolution tool the runtime needs — it replaces EpiDISH / CIBERSORT / NNLS-against-Loyfer. There are no separate per-stage deconvolution steps with separate atlases.

## Files in this folder

| File | What it is |
|---|---|
| `iamatlas_deconvolver.py` | The deconvolver (NNLS + class aggregation). |
| `README.md` | This file. |

## Dependencies

It needs two things from the atlas folder (`../IAMAtlas_v0_1/`):
- `IAMAtlas.csv` (decompress `IAMAtlas.csv.xz` first)
- `IAMAtlas_celltype_to_class.json` (maps each of the 133 cell types to its architecture class, for the per-class aggregation)

## Usage

```python
import json
from iamatlas_deconvolver import IAMAtlasDeconvolver, set_celltype_class_map

# load atlas + class map
deconv = IAMAtlasDeconvolver("IAMAtlas.csv")
with open("IAMAtlas_celltype_to_class.json") as f:
    set_celltype_class_map(deconv, json.load(f))

# customer_betas: dict {cpg_id: beta}
result = deconv.deconvolve(customer_betas)
print(result.fractions)        # {'CD4_T-cells': 0.18, 'Hepatocytes': 0.05, ...}
print(result.class_fractions)  # {'immune': 0.74, 'secretory': 0.21, ...}
print(result.residual_mae)
```

## How it works

1. Filter informative CpGs from the matrix (low posterior SD AND high between-cell-type variance).
2. Intersect the customer's CpGs with the informative CpGs.
3. Build reference matrix R (n_cpg x n_celltype) from IAMAtlas posterior means.
4. Solve y = R x f with f >= 0 and sum(f) = 1 via constrained NNLS, weighted by inverse posterior SD.
5. Return per-cell-type fractions, per-class aggregations, residual diagnostics.

## Validation status

- Synthetic ground-truth: PASS. Recovered a known 0.6 / 0.4 mixture exactly (residual MAE 0.0000 on the informative CpGs).
- Real-data test: pending (roadmap STEP 8.5). Needs a full beta vector (~hundreds of thousands of CpGs). Small targeted panels (e.g. a 7-CpG AD panel) will correctly report `INSUFFICIENT_INFORMATIVE_CPGS` — that is expected behavior, not a bug; full deconvolution needs a genome-wide vector.

## Note on input coverage

A customer sample on EPIC/850K produces ~865K CpGs; on 450K, ~485K. The deconvolver uses whichever CpGs overlap the atlas's informative set (the ~483K backbone). EPIC-only sites not present in IAMAtlas v0.1 are ignored. This does not impair deconvolution — only a few hundred informative CpGs are needed for a stable solution.
