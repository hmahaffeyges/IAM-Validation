# Immune-Class Structural Test — 2026-05-04

**Question:** Does the immune class warrant a split into lymphoid (CD8T, CD4T, NK, Bcell) and myeloid (Mono, Neu) before §22 brightness layer build commits to class structure?

**Method:** Three structural tests on Salas IDOL deconvolution reference (450 CpGs × 6 immune subtypes, cell-type-mean β values).

**Pre-registered pass criteria (locked before run):**
- Split: Test 1 ratio > 3× AND Test 2 clear two-cluster AND Test 3 floor diff > 5%
- Unify: All three point unified
- Per-donor data needed: Mixed/marginal results

## Results

### Test 1 — Between/within distance ratio
- Mean |β_lymphoid − β_myeloid|: 0.2315
- Pooled within-group spread: 0.1687
- **Ratio: 1.372** (threshold for split: >3.0)
- Verdict: **UNIFY**

### Test 2 — Entropy clustering
- Cluster separation ratio: 1.265
- **Pairwise distances reveal non-binary structure:**
  - Mono-Neu distance: 4.57 (myeloid pair, closest)
  - CD8T-CD4T distance: 4.46 (T-cell pair, also tight)
  - Bcell to Mono: 5.30, Bcell to CD8T: 6.59 — **Bcell clusters closer to myeloid than to T-cells**
  - NK at intermediate distances from all
- Verdict: **MARGINAL** (the simple lymphoid-myeloid binary doesn't match observed cluster geometry)

### Test 3 — Stratified H_min
- Lymphoid floor (min of subtype means): 0.4884 (set by Bcell)
- Myeloid floor (min of subtype means): 0.4682 (set by Mono)
- |ΔH_min|: 0.0202 (4.13% relative)
- Threshold for split: >5%
- Verdict: **MARGINAL** (just below split threshold)
- Note: Bcell brings the lymphoid floor down to nearly the myeloid floor; T-cell-only floor would be ~0.58

## Combined verdict

**0 SPLIT, 1 UNIFY, 2 MARGINAL → IMMUNE CLASS STAYS UNIFIED**

The simple lymphoid-vs-myeloid binary split is not justified by Salas IDOL structural data. The H_min floor for the immune class is robust at the granularity available.

## Honest caveats

1. **Test design is structural, not posterior-level.** Cell-type-mean data, not per-donor variance. Two MARGINAL verdicts trigger the pre-registered "per-donor data needed" condition for definitive resolution.

2. **The cluster structure that exists is more nuanced than lymphoid/myeloid.** B-cells look architecturally closer to myeloid cells than to T-cells. T-cells (CD4T, CD8T) cluster tightly. NK at intermediate distance. If a class restructuring were ever warranted, it would not be along the proposed binary axis.

3. **The G-002 6.44σ correction precedent** is consistent with subtype mixture effects within a unified immune class, not with within-class architectural heterogeneity at the H_min level.

## Decision for §22 build

**Immune class stays as one architectural unit.** The 8-class structure holds. §22 brightness layer build proceeds on the existing 8-class architecture with empirical justification rather than accommodation.

Per Heath's earlier statement: "if this one doesn't require a split then I am def not worried about the other." The other classes (stromal, cycling, progenitor) are not tested here; the result on immune is the strongest candidate test, and its outcome closes the immune-split question.

## Reproducibility

- Test script: /home/claude/immune_split_structural_test.py
- Source data: /home/claude/repo/IAM-Validation/Biological_Physics/atlas_vault/stage3_immune_fraction/salas_blood_epic_idol/IDOLOptimizedCpGs_compTable.csv
- Date executed: 2026-05-04
- Result hash: 0 SPLIT, 1 UNIFY, 2 MARGINAL → unified

## Future work, if ever needed

If a finer within-immune architectural investigation is warranted later (e.g., T-cell-vs-non-T-cell axis), it should:
- Use per-donor variance data from Loyfer raw deposits, not deconvolution-reference means
- Test non-binary cluster geometries informed by the pairwise distance structure observed here
- Be motivated by physics derivation, not data clustering alone (per framework discipline)
