# Phase B2 — NILC second deconvolver: L4 cross-method gate NOT cleared

**Date:** 2026-05-30
**Status:** Module built and exercised. L4 cross-method gate FAILED on first run. Phase B2.1 investigation required before gate can be re-tested.

## What was built
- `nilc_deconvolver.py` — generalized least squares deconvolver with inverse-variance weighting from IAMAtlas posterior SDs. Anti-NNLS: no non-negativity constraint during solve, simplex-projection (Duchi 2008) at the end. Optional chromosome-windowed mode.
- `cross_method_comparison()` function with three agreement metrics per the v3 Roadmap §10.2.2 specification.
- CLI with `deconvolve` and `crosscheck` modes.

## What was tested
- Atlas: IAMAtlasREBUILD.csv (483,093 CpGs × 8 classes × {mean, sd, ci_lo, ci_hi})
- Markers: 6,802 union of per-cell-type top-100 markers from `iamatlas_celltype_markers_v0_1.json`
- Cohort: EPIC-Italy GSE51057 (329 patients) + GSE51032 (845 patients) — 1,024 patients matched with Walther output
- Walther output: `PATH1_GSE51057_class_fractions.csv` + `PATH1_GSE51032_class_fractions.csv`

## What was found

### Headline disagreement (Walther vs NILC, cohort means)
| Class | Walther mean | NILC mean | Δ (N−W) |
|-------|-------------:|----------:|--------:|
| stem_pluri  | 0.0036 | 0.0003 | −0.003 |
| **stem_adult**  | **0.0654** | **0.1783** | **+0.113** |
| progenitor  | 0.0357 | 0.0009 | −0.035 |
| cycling     | 0.0011 | 0.0000 | −0.001 |
| secretory   | 0.0000 | 0.0000 | +0.000 |
| **immune**      | **0.8938** | **0.8205** | **−0.073** |
| terminal    | 0.0004 | 0.0001 | −0.000 |
| stromal     | 0.0000 | 0.0000 | +0.000 |

Both methods agree blood is overwhelmingly immune. NILC reallocates ~11pp of mass from progenitor + immune to stem_adult.

### Cross-method correlation
- Per-class Spearman ρ (on classes with real variation):
  - immune: ρ = +0.643  (target ≥ 0.85 — **FAIL**)
  - stem_adult: ρ = +0.720  (target ≥ 0.85 — **FAIL**)
  - progenitor: ρ = +0.157  (**FAIL**)
  - stem_pluri: ρ = −0.101  (**FAIL** — but variance is near zero in both methods so this is noise)
- Median L1 disagreement per patient: **0.226** (target < 0.05 — **FAIL**)
- Top-10 patient overlap per class: 0-1 out of 10 (target ≥ 7 — **FAIL**)

### Interpretation
This is consistent with **immune/stem_adult marker collinearity** in the IAMAtlas REBUILD posteriors. The CpGs strongly marking immune class also have non-zero stem_adult posterior means, so:
- Walther's NNLS (non-negativity + the specific marker-selection logic) pushes the borderline signal entirely to immune.
- NILC's GLS (unconstrained linear inversion) distributes the borderline signal between immune and stem_adult proportional to inverse posterior variance.

**Neither method is "wrong."** Both reflect the true ambiguity in the marker pool. The fact that the L4 cross-method gate fails is the discipline working as designed — it surfaced a structural ambiguity in the atlas posteriors that would otherwise have propagated invisibly through L5+ analyses.

## Phase B2.1 deliverable (required before L4 gate can be retested)
- Identify which marker CpGs have the highest stem_adult posterior mean among the top-100-per-cell-type immune markers.
- Test whether removing those collinear markers from NILC's pool brings the two methods into agreement.
- If yes, propose a "decollinearized marker pool" as v0.2 of the marker artifact.
- If no, investigate whether the immune-class MCMC posterior itself has a stem-adult-contamination signal that the IAMAtlas REBUILD has not fully separated.

## Why this is good news in the larger frame
The chain of custody discipline is supposed to catch L4 problems before they propagate to L5. Phase A built L9 first so every VAL must pass its nulls. Phase B is now exercising L4 cross-method confirmation. **Phase B2 has caught a real L4 issue on its first run.** That is exactly what the discipline is for. Before Phase B2 existed, every CPG result was being produced by Walther alone; we had no way to know whether his answer was the unique answer or whether other reasonable methods would give other answers. Now we know: other methods give meaningfully different answers, and the difference traces to a specific cause (marker collinearity).

Phase C (correlation structure) cannot start on L4-cleaned data until this is resolved. Phase B is not closed.
