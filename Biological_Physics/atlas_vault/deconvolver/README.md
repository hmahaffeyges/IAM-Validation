# Walther IAM Deconvolver

The cell-fraction estimator for EDEAR, built specifically for IAMAtlas. Takes a customer methylation sample (beta values per CpG) and returns a per-CLASS fraction breakdown (the 8 IAM architecture classes — the primary, reliable output the cellular thermometer uses) and an indicative per-cell-type breakdown.

This replaces the earlier prototype deconvolver (`iamatlas_deconvolver.py`, removed), which was written quickly and assumed reference characteristics IAMAtlas does not have — it used absolute selection thresholds that rejected every CpG on this atlas and loaded the whole matrix into memory. The Walther deconvolver was rebuilt from scratch around how IAMAtlas actually behaves.

## Files

| File | What it is |
|---|---|
| `walther_iam_deconvolver.py` | The deconvolver. |
| `README.md` | This file. |

## Dependencies

- `numpy`, `scipy`
- The atlas: `../IAMAtlas_v0_1/IAMAtlas.csv` (decompress `IAMAtlas.csv.xz` first)
- The class map: `../IAMAtlas_v0_1/IAMAtlas_celltype_to_class.json`

## Usage

```python
from walther_iam_deconvolver import WaltherIAMDeconvolver

d = WaltherIAMDeconvolver("IAMAtlas.csv",
                          celltype_class_map="IAMAtlas_celltype_to_class.json")
result = d.deconvolve(customer_betas)   # customer_betas: {cpg_id: beta}

print(result.class_fractions)     # PRIMARY — trust this  {'immune':0.74, ...}
print(result.celltype_fractions)  # SECONDARY — indicative only
print(result.diagnostics)         # markers matched, residual MAE, per-class confidence
print(result.status)              # "OK" or an INSUFFICIENT_* reason
```

Command line:
```bash
python3 walther_iam_deconvolver.py --matrix IAMAtlas.csv \
    --map IAMAtlas_celltype_to_class.json --betas customer_betas.json
```

## Design (why it is built this way)

IAMAtlas v0.1 was measured to have these properties, and the deconvolver is built around them:

1. Between-cell-type variance is compressed (median ~0.0003, max ~0.0067): individual cell types sit close together and are only weakly separable. Absolute variance thresholds reject everything. FIX: markers are selected by RANK within the atlas, not by an absolute cutoff.
2. Between-CLASS variance is large and reliable: the 8 architecture classes are well separated. The class-level solve is the primary output.
3. Per-cell-type discrimination is fuzzy within a class: NNLS can shift weight between similar cells in the same class (e.g. CD4_T <-> Mono, Hepatocytes <-> Gland). Per-cell-type fractions are returned but labelled indicative; class-level aggregation washes the fuzziness out.
4. Empty cells (a cell type never measured at a CpG) are skipped everywhere — never treated as a value.
5. The matrix is large (~1.2 GB). The deconvolver STREAMS it in one pass with bounded-memory heaps, keeping only marker rows. Peak memory ~120 MB on the full atlas (the old prototype crashed trying to load it all).

## How it works

1. One streaming pass over the matrix. For each CpG, compute between-class variance and between-cell-type variance from the means present.
2. Bounded top-N marker selection via min-heaps: top class-discriminating CpGs (plus top one-vs-rest per class so every class is represented), and top cell-type-discriminating CpGs. Memory stays flat regardless of atlas size.
3. Tier 1 (primary): intersect class markers with the customer's CpGs, build the class reference matrix over solvable classes (>=60% marker coverage), solve y = R f with f >= 0, normalise. Report fractions, residual MAE, and a per-class confidence (marker support x fit quality).
4. Tier 2 (secondary, optional): same NNLS over cell-type markers for cell types with >=80% coverage, weighted by inverse posterior SD. Labelled indicative.

## Validation status

- Clean synthetic mixture (class level): recovered exactly (e.g. immune 0.70 / secretory 0.30, residual MAE 0.000).
- Noisy 3-class mixture (2% measurement noise): accurate at class level (total absolute error ~0.10 across three classes; correct structure).
- Full 1.2 GB atlas: streams in ~120 MB peak memory, recovers clean mixtures exactly.
- Cell-type tier: runs and returns indicative fractions; within-class fuzziness present as designed/expected.
- Real-data test (genuine noisy beta vector, e.g. GSE130748): the next validation step — synthetic tests prove the mechanics; real-world performance is validated in the IAMAtlas testing campaign (new evidence report).

## What to trust in v0.1

Use `class_fractions` for decisions (the cellular thermometer reads at class level). Treat `celltype_fractions` as indicative. Sharper per-cell-type resolution is a v0.2 target (more discriminating / EPIC-enhancer atlases, eventually in-house reference data).
