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

---

## Operational reference (for debugging / tuning — READ IF RESULTS LOOK WRONG)

### Constructor parameters (all tunable)

| Parameter | Default | What it controls | When to change |
|---|---|---|---|
| `matrix_path` | (required) | Path to decompressed `IAMAtlas.csv`. | — |
| `celltype_class_map` | `None` | Path to `IAMAtlas_celltype_to_class.json`, a dict, or `None`. If `None`, cell-type refinement still runs but cells map to class `"unknown"` (class aggregation by cell type won't work — always pass the map). | Always pass the map. |
| `n_class_markers_per_class` | `600` | Per-class quota of top class-discriminating CpGs kept (global pool ~= 600 x 8 = 4,800). | Raise if too few class markers match a real sample (sparse arrays); lower for speed. |
| `max_celltype_markers` | `4000` | Cap on cell-type marker CpGs kept. | Raise for finer (still indicative) cell-type resolution; lower for speed/memory. |
| `verbose` | `True` | Print scan/selection progress. | Set `False` in batch/production. |

Two attributes can be set AFTER construction but BEFORE `deconvolve()` only if you re-run `_select_markers()`; simpler to subclass or set as kwargs if exposed. Current internal knobs:
- `min_celltypes_per_cpg` (default 3): a CpG needs >=3 cell types with data to be a cell-type marker candidate.

### Status codes returned in `result.status`

| Status | Meaning | How to fix |
|---|---|---|
| `OK` | Class-level solve succeeded. | — |
| `INSUFFICIENT_CLASS_MARKERS` | Fewer than 50 class-marker CpGs overlapped the customer's CpG set. | The customer vector is too small or too sparse, OR `n_class_markers_per_class` is too low. Check `diagnostics["n_class_markers_matched"]`. A genome-wide 450K/EPIC vector should match thousands; a 7-CpG panel will NOT (expected). |
| `INSUFFICIENT_CLASS_COVERAGE` | Fewer than 2 classes had >=60% marker coverage across the matched markers. | The matched markers don't span enough classes. Usually a too-small customer vector. Inspect `diagnostics["class_coverage"]`. |

Note: a small targeted panel (e.g. the 7-CpG AD panel) is EXPECTED to return `INSUFFICIENT_CLASS_MARKERS`. The deconvolver needs a genome-wide vector (hundreds-to-thousands of overlapping markers). This is not a bug.

### Internal gates (what gets filtered, and why)

- **50-marker minimum (Tier 1):** below 50 matched class markers, the NNLS is too underdetermined to trust — returns `INSUFFICIENT_CLASS_MARKERS`.
- **60% class coverage gate (Tier 1):** a class is only included in the solve if it has a posterior mean at >=60% of the matched markers. Classes below this are dropped from the fit (set to 0.0 in output). Prevents a thinly-covered class from being fit on noise.
- **80% cell-type coverage gate (Tier 2):** a cell type is only included in the indicative cell-type solve if present at >=80% of matched cell-type markers. Stricter than the class gate because cell-type data is sparser and fuzzier.
- **Empty-cell skipping:** anywhere a `_mean` cell is `""` or `"NA"`, it is skipped — never read as 0 or any value.

### Reading the diagnostics dict

- `n_customer_cpgs` — size of the input vector.
- `n_class_markers_matched` — how many class markers overlapped the customer. Want hundreds+.
- `class_residual_mae` — mean absolute error between the reconstructed and observed beta over matched markers. Lower is better; ~0 on clean synthetic, expect higher on real noisy data. Above ~0.1 suggests poor fit (wrong array, bad betas, or atlas/customer mismatch).
- `classes_solved` — which of the 8 classes were actually in the fit (passed the 60% gate).
- `class_marker_coverage` — per-class marker count among matched markers.
- `class_confidence` — per-class score in [0,1] = (marker support fraction) x (fit quality), where fit_quality = max(0, 1 - residual/0.2). A class with low coverage OR a poor overall fit gets low confidence. Use this to decide how much to trust a given class's fraction.
- `n_celltype_markers_matched` — overlap for the Tier-2 cell-type solve.
- `celltype_note` — the standing reminder that cell-type fractions are indicative only.

### Confidence formula (explicit)

```
fit_quality = max(0, 1 - class_residual_mae / 0.2)      # resid 0 -> 1.0 ; resid 0.2 -> 0.0
class_confidence[c] = min(1.0, (coverage[c] / n_matched_markers) * fit_quality)
```
This is a deliberately simple, bounded heuristic — NOT a calibrated probability. It flags "this class is well-supported and the overall fit is good" vs "treat with caution." Recalibrate against real labelled data when available.

### Known quirks / gotchas for a future maintainer

- **The atlas must be DECOMPRESSED first** (`IAMAtlas.csv`, not `.csv.xz`). The deconvolver reads CSV by streaming; it does not decompress.
- **Class-level is the product surface.** The cellular thermometer / A-scores read at the 8-class level. Do not expose raw cell-type fractions to customers as if precise; they are indicative in v0.1.
- **Within-class confusion is expected, not a bug.** If a known CD4-T sample reports weight on "Mono", or a hepatocyte sample on "Gland", that is the documented compressed-variance behavior. The CLASS answer (immune, secretory) will still be right.
- **Marker selection is recomputed every construction** (one streaming pass, ~seconds-to-minutes depending on disk). For repeated scoring, construct the deconvolver ONCE and reuse it for many `deconvolve()` calls.
- **Memory is bounded by the heaps, not the atlas size.** Peak ~120 MB on the full 1.2 GB atlas. If you raise `n_class_markers_per_class` or `max_celltype_markers` a lot, memory grows proportionally to those caps, not to the atlas.
- **Determinism:** marker selection uses a tiebreak counter so ties resolve deterministically by scan order; same atlas in -> same markers out.
- **v0.2 sharpening path:** the within-class fuzziness traces to the atlas's compressed between-cell variance (hierarchical-model shrinkage + most CpGs not being cell-type markers). More discriminating source atlases (EPIC enhancer coverage, eventually in-house reference data) would widen per-cell separation and make Tier-2 trustworthy.
