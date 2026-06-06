"""
Immune-class structural test on Salas IDOL reference matrix.

Question: Does the immune class (currently one architectural unit spanning
6 subtypes) warrant a split into lymphoid (CD8T, CD4T, NK, Bcell) and
myeloid (Mono, Neu)?

Three structural tests on cell-type-mean β data (no per-donor variance available):

Test 1 — Between-group vs within-group distance ratio
Test 2 — Per-cell-type Shannon entropy clustering pattern
Test 3 — Approximate H_min stratified by group

Pre-registered pass criteria (locked before running):
- Split justified: T1 ratio > 3× AND T2 clear two-cluster AND T3 |ΔH_min|/H_min > 5%
- Unify justified: all three point unified with comparable strength
- Per-donor data needed: mixed or marginal results

Source: /home/claude/repo/IAM-Validation/Biological_Physics/atlas_vault/
        stage3_immune_fraction/salas_blood_epic_idol/IDOLOptimizedCpGs_compTable.csv
"""

import numpy as np
import pandas as pd
import sys

# ============================================================
# Load Salas IDOL reference matrix
# ============================================================
SALAS_PATH = "/home/claude/repo/IAM-Validation/Biological_Physics/atlas_vault/stage3_immune_fraction/salas_blood_epic_idol/IDOLOptimizedCpGs_compTable.csv"

df = pd.read_csv(SALAS_PATH)
print(f"Loaded Salas IDOL: {df.shape[0]} CpGs × {df.shape[1]-1} cell types")
print(f"Columns: {list(df.columns)}")
print()

# Cell type columns
LYMPHOID_COLS = ["CD8T", "CD4T", "NK", "Bcell"]
MYELOID_COLS  = ["Mono", "Neu"]
ALL_IMMUNE    = LYMPHOID_COLS + MYELOID_COLS

# Verify columns exist
for col in ALL_IMMUNE:
    assert col in df.columns, f"Missing column: {col}"

# Extract β matrices
beta_lymphoid = df[LYMPHOID_COLS].values  # n_cpg × 4
beta_myeloid  = df[MYELOID_COLS].values   # n_cpg × 2
beta_all      = df[ALL_IMMUNE].values     # n_cpg × 6

n_cpg = beta_all.shape[0]
print(f"Lymphoid: n_cpg={n_cpg}, n_subtypes={len(LYMPHOID_COLS)} ({LYMPHOID_COLS})")
print(f"Myeloid:  n_cpg={n_cpg}, n_subtypes={len(MYELOID_COLS)} ({MYELOID_COLS})")
print()

# ============================================================
# TEST 1 — Between-group vs within-group distance
# ============================================================
print("=" * 70)
print("TEST 1 — Between-group vs within-group distance")
print("=" * 70)

# Group means per CpG
lymphoid_mean = beta_lymphoid.mean(axis=1)  # n_cpg
myeloid_mean  = beta_myeloid.mean(axis=1)   # n_cpg

# Between-group distance per CpG (absolute)
between_dist = np.abs(lymphoid_mean - myeloid_mean)  # n_cpg

# Within-group spread per CpG (standard deviation of subtypes within group)
within_lymphoid = beta_lymphoid.std(axis=1, ddof=1)  # n_cpg, std across 4 subtypes
within_myeloid_diff = np.abs(beta_myeloid[:, 0] - beta_myeloid[:, 1])  # n_cpg, |Mono - Neu| since only 2 subtypes

# Average within-group spread (treat |Mono-Neu| as a 2-sample analog of std)
# For fair comparison: both should reflect "typical disagreement among same-group members"
# Use std-of-pair = |a-b|/sqrt(2) for 2-sample; this matches sample std formula
within_myeloid = within_myeloid_diff / np.sqrt(2)

# Mean of within-group spreads
within_lymphoid_mean = within_lymphoid.mean()
within_myeloid_mean  = within_myeloid.mean()
within_pooled        = (within_lymphoid_mean + within_myeloid_mean) / 2

between_mean = between_dist.mean()
ratio = between_mean / within_pooled

print(f"Mean |β_lymphoid - β_myeloid| (between-group):  {between_mean:.4f}")
print(f"Mean lymphoid within-group spread (4 subtypes): {within_lymphoid_mean:.4f}")
print(f"Mean myeloid within-group spread (2 subtypes):  {within_myeloid_mean:.4f}")
print(f"Pooled within-group spread:                     {within_pooled:.4f}")
print()
print(f"BETWEEN/WITHIN ratio: {ratio:.3f}")
print(f"  Pre-registered split threshold: > 3.0")
print(f"  Pre-registered unify threshold: < ~1.5")

# Distribution of between-group distances
print(f"\nBetween-group distance distribution across {n_cpg} CpGs:")
print(f"  Median:  {np.median(between_dist):.4f}")
print(f"  P25:     {np.percentile(between_dist, 25):.4f}")
print(f"  P75:     {np.percentile(between_dist, 75):.4f}")
print(f"  P95:     {np.percentile(between_dist, 95):.4f}")
print(f"  Max:     {np.max(between_dist):.4f}")

# Fraction of CpGs where between > within (clear architectural separation at that CpG)
clear_separation = (between_dist > within_pooled).sum() / n_cpg
print(f"\nFraction of CpGs with between-distance > pooled within-spread: {clear_separation:.3f}")

# Test 1 verdict
if ratio > 3.0:
    test1_verdict = "SPLIT"
elif ratio < 1.5:
    test1_verdict = "UNIFY"
else:
    test1_verdict = "MARGINAL"

print(f"\nTEST 1 VERDICT: {test1_verdict} (ratio={ratio:.3f})")
print()

# ============================================================
# TEST 2 — Per-cell-type Shannon entropy clustering
# ============================================================
print("=" * 70)
print("TEST 2 — Per-cell-type Shannon entropy clustering")
print("=" * 70)

def shannon_entropy_beta(beta_value):
    """Shannon entropy of methylation β treated as binary distribution."""
    # H = -p log2 p - (1-p) log2 (1-p), with p=β
    # Handle β=0 and β=1 edge cases
    if beta_value <= 0 or beta_value >= 1:
        return 0.0
    return -beta_value * np.log2(beta_value) - (1-beta_value) * np.log2(1-beta_value)

# Per-cell-type entropy distribution across all CpGs
entropies = {}
for ct in ALL_IMMUNE:
    h = np.array([shannon_entropy_beta(b) for b in df[ct].values])
    entropies[ct] = h

print("Per-cell-type Shannon entropy summary (across 450 CpGs):")
print(f"{'Cell type':12s} {'Mean H':>8s} {'Median H':>9s} {'Std H':>8s}")
for ct in ALL_IMMUNE:
    h = entropies[ct]
    print(f"{ct:12s} {h.mean():8.4f} {np.median(h):9.4f} {h.std():8.4f}")

# Build distance matrix between cell types using mean entropy + entropy-distribution similarity
# Use Euclidean distance on entropy vectors as a clustering proxy
ct_entropy_matrix = np.array([entropies[ct] for ct in ALL_IMMUNE])  # 6 × n_cpg

# Pairwise distances between cell types based on entropy patterns
print("\nPairwise distances between cell types (Euclidean on entropy patterns):")
print(f"{'':12s}", end="")
for ct2 in ALL_IMMUNE:
    print(f"{ct2:>10s}", end="")
print()

dist_matrix = np.zeros((6, 6))
for i, ct1 in enumerate(ALL_IMMUNE):
    print(f"{ct1:12s}", end="")
    for j, ct2 in enumerate(ALL_IMMUNE):
        d = np.linalg.norm(ct_entropy_matrix[i] - ct_entropy_matrix[j])
        dist_matrix[i, j] = d
        print(f"{d:>10.3f}", end="")
    print()

# Within-lymphoid mean distance vs between-lymphoid-myeloid mean distance
lymphoid_indices = [0, 1, 2, 3]  # CD8T, CD4T, NK, Bcell
myeloid_indices  = [4, 5]        # Mono, Neu

# Within lymphoid: pairs (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
within_lymphoid_dists = [dist_matrix[i, j] for i in lymphoid_indices for j in lymphoid_indices if i < j]
# Within myeloid: pair (4,5)
within_myeloid_dists  = [dist_matrix[i, j] for i in myeloid_indices for j in myeloid_indices if i < j]
# Between: all pairs (lymph, myel)
between_dists = [dist_matrix[i, j] for i in lymphoid_indices for j in myeloid_indices]

within_lymphoid_mean_dist = np.mean(within_lymphoid_dists)
within_myeloid_mean_dist  = np.mean(within_myeloid_dists)
between_mean_dist         = np.mean(between_dists)

print(f"\nMean within-lymphoid pairwise distance (6 pairs): {within_lymphoid_mean_dist:.3f}")
print(f"Mean within-myeloid pairwise distance (1 pair):   {within_myeloid_mean_dist:.3f}")
print(f"Mean between-group pairwise distance (8 pairs):   {between_mean_dist:.3f}")

# Two-cluster separation ratio
within_pooled_dist = (within_lymphoid_mean_dist + within_myeloid_mean_dist) / 2
cluster_ratio = between_mean_dist / within_pooled_dist if within_pooled_dist > 0 else float('inf')

print(f"\nCluster separation ratio (between / pooled within): {cluster_ratio:.3f}")

# Verdict for Test 2
if cluster_ratio > 2.0:
    test2_verdict = "SPLIT (clear two-cluster)"
elif cluster_ratio < 1.2:
    test2_verdict = "UNIFY (no cluster separation)"
else:
    test2_verdict = "MARGINAL"

print(f"\nTEST 2 VERDICT: {test2_verdict} (ratio={cluster_ratio:.3f})")
print()

# ============================================================
# TEST 3 — Approximate H_min stratified by group
# ============================================================
print("=" * 70)
print("TEST 3 — Approximate H_min stratified by group")
print("=" * 70)

# H_min for a class: minimum operational entropy across CpGs in the class.
# At cell-type-mean level, compute entropy per (CpG, cell-type), then for each
# group take the cell-type with lowest mean entropy as the floor anchor.
# This mirrors the cookbook's "minimum maintenance entropy" interpretation.

# Mean entropy per cell type across all CpGs
mean_h_per_ct = {ct: entropies[ct].mean() for ct in ALL_IMMUNE}

print("Mean Shannon entropy per cell type (across 450 CpGs):")
for ct in ALL_IMMUNE:
    print(f"  {ct:12s}: H_mean = {mean_h_per_ct[ct]:.4f}")

# Group-level minimum mean entropy (the H_min approximation)
lymphoid_h_values = [mean_h_per_ct[ct] for ct in LYMPHOID_COLS]
myeloid_h_values  = [mean_h_per_ct[ct] for ct in MYELOID_COLS]

lymphoid_h_min_approx = min(lymphoid_h_values)
myeloid_h_min_approx  = min(myeloid_h_values)
combined_h_min_approx = min(lymphoid_h_values + myeloid_h_values)

# Also compute mean of group entropies (alternative aggregation)
lymphoid_h_mean = np.mean(lymphoid_h_values)
myeloid_h_mean  = np.mean(myeloid_h_values)

print(f"\nGroup-stratified H_min approximations:")
print(f"  Lymphoid floor (min of 4 subtype means): {lymphoid_h_min_approx:.4f}")
print(f"  Myeloid  floor (min of 2 subtype means): {myeloid_h_min_approx:.4f}")
print(f"  Combined floor (min of all 6):           {combined_h_min_approx:.4f}")
print()
print(f"Group-stratified H_mean (alternative):")
print(f"  Lymphoid mean entropy: {lymphoid_h_mean:.4f}")
print(f"  Myeloid  mean entropy: {myeloid_h_mean:.4f}")

# Relative difference
delta_floor = abs(lymphoid_h_min_approx - myeloid_h_min_approx)
relative_diff_floor = delta_floor / max(lymphoid_h_min_approx, myeloid_h_min_approx)

delta_mean = abs(lymphoid_h_mean - myeloid_h_mean)
relative_diff_mean = delta_mean / max(lymphoid_h_mean, myeloid_h_mean)

print(f"\nFloor difference: |ΔH_min| = {delta_floor:.4f}, relative = {relative_diff_floor*100:.2f}%")
print(f"Mean difference:  |ΔH_mean| = {delta_mean:.4f}, relative = {relative_diff_mean*100:.2f}%")
print(f"  Pre-registered split threshold: > 5% (relative)")

# Test 3 verdict (use floor as primary, mean as cross-check)
if relative_diff_floor > 0.05:
    test3_verdict = f"SPLIT (floor diff {relative_diff_floor*100:.1f}%)"
elif relative_diff_floor < 0.02:
    test3_verdict = f"UNIFY (floor diff {relative_diff_floor*100:.1f}%)"
else:
    test3_verdict = f"MARGINAL (floor diff {relative_diff_floor*100:.1f}%)"

print(f"\nTEST 3 VERDICT: {test3_verdict}")
print()

# ============================================================
# COMBINED VERDICT
# ============================================================
print("=" * 70)
print("COMBINED VERDICT")
print("=" * 70)

verdicts = [test1_verdict, test2_verdict.split()[0], test3_verdict.split()[0]]
print(f"Test 1 (between/within ratio):       {test1_verdict}")
print(f"Test 2 (entropy cluster separation): {test2_verdict}")
print(f"Test 3 (stratified H_min):           {test3_verdict}")

# Pre-registered combined logic
n_split  = sum(1 for v in verdicts if v == "SPLIT")
n_unify  = sum(1 for v in verdicts if v == "UNIFY")
n_marginal = sum(1 for v in verdicts if v == "MARGINAL")

print(f"\nCount: {n_split} SPLIT, {n_unify} UNIFY, {n_marginal} MARGINAL")

if n_split == 3:
    final = "SPLIT JUSTIFIED — all three tests point to architectural distinction"
elif n_unify == 3:
    final = "UNIFY JUSTIFIED — all three tests point to single architectural class"
elif n_split >= 2 and n_unify == 0:
    final = "SPLIT LIKELY — majority point to split, no unifying evidence"
elif n_unify >= 2 and n_split == 0:
    final = "UNIFY LIKELY — majority point to unification, no splitting evidence"
else:
    final = "PER-DONOR DATA NEEDED — mixed structural signal"

print(f"\nFINAL: {final}")
print()
