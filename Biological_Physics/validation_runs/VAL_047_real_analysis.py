#!/usr/bin/env python3
"""
VAL-047 REAL: GAPE A-score applied to EPIC-Italy per-patient data
=================================================================

FIRST TIME GAPE HAS BEEN APPLIED TO RAW PER-PATIENT METHYLATION DATA.

Data: GSE51057 — EPIC-Italy nested case-control cohort
  - 329 women (152 cases, 177 controls)
  - Pre-diagnostic blood methylation (Illumina 450K, buffy coat)
  - Time to diagnosis: 0.04-13.9 years for cases
  - 146 breast cancer cases (C50), 4 colorectal, 2 other

Method:
  - 25 immune-class-informative CpGs extracted from raw matrix
  - Per-sample mean β computed at those CpGs
  - GAPE A-score = H(β)/H_min_immune applied directly
  - Case vs control comparison — stratified by time-to-diagnosis window
"""

import json
import math
import statistics
from pathlib import Path

# Framework constants
H_MIN_IMMUNE = 0.838889  # G-003b MCMC confirmed

def H(b):
    if b <= 0 or b >= 1: return 0.0
    return -b*math.log2(b) - (1-b)*math.log2(1-b)

def A(b): return H(b) / H_MIN_IMMUNE

# Load data
data = json.loads(Path('/home/claude/geo_analysis/GSE51057_immune_beta.json').read_text())
samples = data['metadata_with_beta']

# Filter to samples with valid beta
samples = [s for s in samples if s.get('immune_mean_beta') is not None]
print(f"Samples with valid immune β: {len(samples)}")

# Compute A-score per sample
for s in samples:
    s['A_immune'] = A(s['immune_mean_beta'])

# Split case/control
controls = [s for s in samples if not s['is_case']]
cases = [s for s in samples if s['is_case']]
breast_cases = [s for s in cases if s.get('cancer_type') == 'C50']

print()
print("=" * 78)
print("VAL-047 REAL: GAPE A-SCORE ON EPIC-ITALY PER-PATIENT DATA")
print("=" * 78)

# Group statistics
def stats(samples_list, label):
    betas = [s['immune_mean_beta'] for s in samples_list]
    A_scores = [s['A_immune'] for s in samples_list]
    ages = [s['age'] for s in samples_list]
    return {
        'label': label,
        'n': len(samples_list),
        'beta_mean': statistics.mean(betas),
        'beta_sd': statistics.stdev(betas) if len(betas) > 1 else 0,
        'A_mean': statistics.mean(A_scores),
        'A_sd': statistics.stdev(A_scores) if len(A_scores) > 1 else 0,
        'age_mean': statistics.mean(ages),
        'age_sd': statistics.stdev(ages) if len(ages) > 1 else 0,
    }

print(f"\n{'Group':<35} {'n':>5} {'β_mean':>8} {'β_sd':>7} {'A_mean':>8} {'A_sd':>7} {'Age_mean':>9}")
print("-" * 80)
for group, label in [(controls, 'Controls (no cancer)'),
                     (cases, 'All cases (future cancer)'),
                     (breast_cases, 'Breast cancer cases (C50)')]:
    st = stats(group, label)
    print(f"{st['label']:<35} {st['n']:>5} {st['beta_mean']:>8.4f} {st['beta_sd']:>7.4f} "
          f"{st['A_mean']:>8.4f} {st['A_sd']:>7.4f} {st['age_mean']:>9.1f}")

# Primary comparison: breast cases vs controls
print()
print("=" * 78)
print("PRIMARY COMPARISON: BREAST CANCER CASES vs MATCHED CONTROLS")
print("=" * 78)

ctrl_stats = stats(controls, 'Controls')
case_stats = stats(breast_cases, 'Breast cases')

dA = case_stats['A_mean'] - ctrl_stats['A_mean']
# Pooled SD
pooled_sd = math.sqrt(((ctrl_stats['n']-1)*ctrl_stats['A_sd']**2 + 
                       (case_stats['n']-1)*case_stats['A_sd']**2) / 
                      (ctrl_stats['n'] + case_stats['n'] - 2))
cohens_d = dA / pooled_sd if pooled_sd > 0 else 0

# Welch t-test approximation
se_diff = math.sqrt(ctrl_stats['A_sd']**2/ctrl_stats['n'] + case_stats['A_sd']**2/case_stats['n'])
t_stat = dA / se_diff if se_diff > 0 else 0
# Two-tailed p from normal approximation (large n)
p_val = math.erfc(abs(t_stat) / math.sqrt(2))

print(f"\nControls A-score:     {ctrl_stats['A_mean']:.4f} ± {ctrl_stats['A_sd']:.4f}  (n={ctrl_stats['n']}, age {ctrl_stats['age_mean']:.1f})")
print(f"Breast cases A-score: {case_stats['A_mean']:.4f} ± {case_stats['A_sd']:.4f}  (n={case_stats['n']}, age {case_stats['age_mean']:.1f})")
print(f"\nΔA (case - control):  {dA:+.5f}")
print(f"Pooled SD:            {pooled_sd:.5f}")
print(f"Cohen's d:            {cohens_d:+.3f}")
print(f"t-statistic:          {t_stat:+.3f}")
print(f"p-value (2-tailed):   {p_val:.4e}")

# Cohen's d interpretation
if abs(cohens_d) < 0.2:
    d_interp = "negligible effect"
elif abs(cohens_d) < 0.5:
    d_interp = "small effect"
elif abs(cohens_d) < 0.8:
    d_interp = "medium effect"
else:
    d_interp = "large effect"
print(f"Effect size label:    {d_interp}")

# Per-individual detection performance at various thresholds
print()
print("PER-INDIVIDUAL DETECTION PERFORMANCE (real data):")
print(f"{'Threshold (A units)':<22} {'Sens':>7} {'Spec':>7} {'PPV@1%':>8} {'PPV@5%':>8}")
for thr in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]:
    A_ctrl_mean = ctrl_stats['A_mean']
    tp = sum(1 for s in breast_cases if s['A_immune'] - A_ctrl_mean >= thr)
    fn = len(breast_cases) - tp
    fp = sum(1 for s in controls if s['A_immune'] - A_ctrl_mean >= thr)
    tn = len(controls) - fp
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    for prev, prev_label in [(0.01, '1%'), (0.05, '5%')]:
        ppv = (sens*prev) / (sens*prev + (1-spec)*(1-prev)) if (sens*prev + (1-spec)*(1-prev)) else 0
        if prev_label == '1%':
            ppv_1 = ppv
        else:
            ppv_5 = ppv
    print(f"  ΔA ≥ {thr:.3f}        {sens*100:>6.1f}% {spec*100:>6.1f}% {ppv_1*100:>7.1f}% {ppv_5*100:>7.1f}%")

# Time-to-diagnosis stratification (the really interesting test)
print()
print("=" * 78)
print("TIME-TO-DIAGNOSIS STRATIFICATION (breast cases)")
print("=" * 78)
print("Does ΔA increase as we approach clinical diagnosis?")
print()

for yr_min, yr_max, label in [(0, 2, '≤2 yr pre-dx (imminent)'),
                                (2, 5, '2-5 yr pre-dx'),
                                (5, 10, '5-10 yr pre-dx'),
                                (10, 20, '>10 yr pre-dx (early)')]:
    subset = [s for s in breast_cases 
              if 'years_to_dx' in s and yr_min <= s['years_to_dx'] < yr_max]
    if not subset: continue
    sub_stats = stats(subset, label)
    sub_dA = sub_stats['A_mean'] - ctrl_stats['A_mean']
    sub_se = math.sqrt(ctrl_stats['A_sd']**2/ctrl_stats['n'] + sub_stats['A_sd']**2/sub_stats['n'])
    sub_z = sub_dA / sub_se if sub_se else 0
    sub_d = sub_dA / pooled_sd if pooled_sd else 0
    sub_p = math.erfc(abs(sub_z) / math.sqrt(2))
    print(f"{label:<28} n={sub_stats['n']:>3}  A={sub_stats['A_mean']:.4f}  ΔA={sub_dA:+.5f}  d={sub_d:+.3f}  p={sub_p:.2e}")

# Age-stratified sensitivity
print()
print("=" * 78)
print("AGE-STRATIFIED COMPARISON")
print("=" * 78)
age_bins = [(30, 45), (45, 55), (55, 70)]
for lo, hi in age_bins:
    ctrl_sub = [s for s in controls if lo <= s['age'] < hi]
    case_sub = [s for s in breast_cases if lo <= s['age'] < hi]
    if len(ctrl_sub) < 5 or len(case_sub) < 5:
        print(f"  Age {lo}-{hi}: insufficient n (ctrl={len(ctrl_sub)}, case={len(case_sub)})")
        continue
    cs = stats(ctrl_sub, 'ctrl')
    xs = stats(case_sub, 'case')
    sub_dA = xs['A_mean'] - cs['A_mean']
    sub_pooled = math.sqrt(((cs['n']-1)*cs['A_sd']**2 + (xs['n']-1)*xs['A_sd']**2) / 
                           (cs['n'] + xs['n'] - 2))
    sub_d = sub_dA / sub_pooled if sub_pooled else 0
    sub_se = math.sqrt(cs['A_sd']**2/cs['n'] + xs['A_sd']**2/xs['n'])
    sub_z = sub_dA / sub_se if sub_se else 0
    sub_p = math.erfc(abs(sub_z) / math.sqrt(2))
    print(f"Age {lo}-{hi}: ctrl n={cs['n']} (A={cs['A_mean']:.4f}), case n={xs['n']} (A={xs['A_mean']:.4f}), "
          f"ΔA={sub_dA:+.5f}, d={sub_d:+.3f}, p={sub_p:.2e}")

# Save full results
out = {
    'val_id': 'VAL-047-REAL',
    'cohort': 'GSE51057 EPIC-Italy',
    'n_total': len(samples),
    'n_cases_breast': len(breast_cases),
    'n_controls': len(controls),
    'n_immune_cpgs': 25,
    'H_min_immune': H_MIN_IMMUNE,
    'control_stats': ctrl_stats,
    'case_stats_breast': case_stats,
    'primary_comparison': {
        'dA': dA,
        'pooled_sd': pooled_sd,
        'cohens_d': cohens_d,
        't_statistic': t_stat,
        'p_value_two_tailed': p_val,
    },
    'per_sample': [
        {'id': s['id'], 'age': s['age'], 'is_case': s['is_case'],
         'cancer_type': s.get('cancer_type'),
         'years_to_dx': s.get('years_to_dx'),
         'immune_mean_beta': s['immune_mean_beta'],
         'A_immune': s['A_immune']}
        for s in samples
    ],
}
Path('/home/claude/geo_analysis/VAL_047_REAL_results.json').write_text(json.dumps(out, indent=1, default=str))
print()
print(f"\nSaved: /home/claude/geo_analysis/VAL_047_REAL_results.json")

# Final summary
print()
print("=" * 78)
print("HEADLINE RESULT")
print("=" * 78)
print(f"GAPE A-score applied to 329 real EPIC-Italy patients.")
print(f"Breast cancer pre-diagnostic vs controls:")
print(f"  ΔA = {dA:+.5f}, Cohen's d = {cohens_d:+.3f}, p = {p_val:.2e}")
print(f"  → {d_interp}")
