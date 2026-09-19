# IAMAtlas Deconvolver + Updated Merge — Step 8 Preview Bundle

**Date:** 2026-05-04
**Status:** Built tonight while production MCMC runs on Heath's laptop.
**Validation:** Synthetic ground-truth test passed (recovered 0.6/0.4 split exactly, residual MAE = 0.0000 on 194 informative CpGs).

---

## What's in this bundle

```
step_8_deconvolver_bundle/
├── iamatlas_deconvolver.py    — The deconvolver (~340 lines, NNLS + class aggregation)
└── merge_iamatlas_v0_1.py     — UPDATED merge script (replaces the version in the gaming PC bundle)
```

## What changed in merge_iamatlas_v0_1.py

The old version emitted only the 32 class-level brightness columns. The new version emits:

1. **Class-level columns** (32): `stem_pluri_mean/sd/ci_lo/ci_hi`, etc., for all 8 architecture classes. Same as before.
2. **Per-cell-type columns** (≥160): `CD4_T-cells_mean/sd`, `Hepatocytes_mean/sd`, etc., for every cell type that had MCMC posterior data.
3. **A new file:** `IAMAtlas_celltype_to_class.json` — maps each cell type to its architecture class. Used by the deconvolver for class aggregation.

**No re-running of MCMC chains needed.** The per-cell-type data is already being written to disk by the production MCMC (in the `iamatlas_v0_1_<class>_per_celltype.csv` files). The merge script just wasn't reading them. Now it does.

## What the deconvolver does

Replaces EpiDISH / CIBERSORT / NNLS-against-Loyfer with a single Python module that runs against IAMAtlas directly. Three lines of usage:

```python
from iamatlas_deconvolver import IAMAtlasDeconvolver

deconv = IAMAtlasDeconvolver("IAMAtlas.csv")
result = deconv.deconvolve(customer_betas)  # customer_betas: dict {cpg_id: β}
print(result.fractions)        # {'CD4_T-cells': 0.18, 'Hepatocytes': 0.05, ...}
print(result.class_fractions)  # {'immune': 0.74, 'secretory': 0.21, ...}
```

## How it works (brief)

1. Filter informative CpGs from the matrix (low posterior SD AND high between-cell-type variance).
2. Find the intersection of customer's CpGs and informative CpGs.
3. Build reference matrix R (n_cpg × n_celltype) of IAMAtlas posterior means.
4. Solve `y = R × f` with `f ≥ 0` and `Σf = 1` via constrained NNLS, weighted by inverse posterior SD.
5. Return per-cell-type fractions, per-class aggregations, and residual diagnostics.

## How to use after the matrix lands tomorrow

```bash
cd ~/IAMPerformance

# 1. After production MCMC finishes (8 result.json files in iamatlas_v0_1_output/):
#    Replace your existing merge script with this one:
cp step_8_deconvolver_bundle/merge_iamatlas_v0_1.py .

# 2. Run the merge — produces both IAMAtlas.csv AND IAMAtlas_celltype_to_class.json
python3 merge_iamatlas_v0_1.py \
    --in_dir iamatlas_v0_1_output \
    --universe iamatlas_cpg_universe.csv \
    --output IAMAtlas.csv \
    --map_output IAMAtlas_celltype_to_class.json

# 3. Drop the deconvolver in alongside:
cp step_8_deconvolver_bundle/iamatlas_deconvolver.py .

# 4. Test it on Heath's existing β data — e.g., one of the AIBL samples:
python3 -c "
import json
from iamatlas_deconvolver import IAMAtlasDeconvolver, set_celltype_class_map

# Load celltype → class mapping
with open('IAMAtlas_celltype_to_class.json') as f: mapping = json.load(f)

# Load deconvolver
deconv = IAMAtlasDeconvolver('IAMAtlas.csv')
set_celltype_class_map(deconv, mapping)

# Load one AIBL sample
with open('aibl_imm_betas.json') as f: aibl = json.load(f)
sample_id = list(aibl.keys())[0]
result = deconv.deconvolve(aibl[sample_id])

print(f'Sample: {sample_id}')
print(f'Cell-type fractions:')
for ct, f in sorted(result.fractions.items(), key=lambda x: -x[1])[:8]:
    print(f'  {ct:<25} {f:.4f}')
print(f'Class fractions:')
for cls, f in sorted(result.class_fractions.items(), key=lambda x: -x[1]):
    print(f'  {cls:<15} {f:.4f}')
print(f'Residual MAE: {result.residual_mae:.4f}')
"
```

(Note: AIBL JSON only contains the 7 val051 panel CpGs per sample, which is too few for full deconvolution to converge. The deconvolver will report `INSUFFICIENT_INFORMATIVE_CPGS` for those samples. Real test will be on a full β vector — e.g., when the GSE130748 Mozhui IDAT extraction runs in Step 7 / Check E, which produces ~865K CpG vectors per sample.)

## Three-way comparison test (the validation we want)

Once the matrix lands, the right test is:

```python
# Cohort: e.g., GSE51057 breast pre-dx >10yr (n=11 cases, n=730 controls)

# Method 1: Old flat-H_min directional scoring
d1 = score_flat_hmin(samples)

# Method 2: IAMAtlas-anchored directional scoring  
d2 = score_iam_residual(samples, iamatlas)

# Method 3: IAMAtlas + own deconvolver, then score per-class A
d3 = score_iam_full_pipeline(samples, iamatlas, deconvolver)
```

If d3 > d2 > d1 monotonically, the matrix and deconvolver each add measurable value. Step 7 / Check D already does d1 and d2; I'll add d3 once the matrix lands.

## What's NOT in this bundle (yet)

- **Age layer** — separate matrix, ~4 hour build job, runs after main matrix is finalized. UniLIFE has the age metadata; we'll build per-CpG age regression from those donors.
- **Cellular age clock** — trains a regression mapping the 8 A-scores → chronological age, residual = IAM cellular age departure. Methods paper of its own.
- **Step 8 atlas vault freeze + EDEAR engine integration** — happens after Step 7 validation passes.

These are next-week work, not blockers for tomorrow's matrix completion.
