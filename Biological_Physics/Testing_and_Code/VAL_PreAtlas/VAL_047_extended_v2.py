#!/usr/bin/env python3
"""
VAL-047 EXTENDED: Four deeper analyses on EPIC-Italy per-patient data.
Uses pre-extracted cpg_hits.tsv (49 CpGs across 5 architecture classes).
"""
import json
import math
import statistics
import random
from pathlib import Path

METADATA = json.loads(Path('/home/claude/geo_analysis/GSE51057_metadata.json').read_text())
CLASS_INFO = json.loads(Path('/home/claude/geo_analysis/all_cpgs.json').read_text())
CLASS_CPGS = CLASS_INFO['by_class']


def H(b):
    if b is None or (isinstance(b, float) and math.isnan(b)):
        return None
    if b <= 0 or b >= 1: return 0.0
    return -b*math.log2(b) - (1-b)*math.log2(1-b)
def A_score(b, cls):
    h = H(b)
    if h is None: return None
    return h / H_MIN[cls]

# Load β data
print("Loading cpg_hits.tsv...")
cpg_data = {}
with open('/home/claude/geo_analysis/cpg_hits.tsv') as f:
    for line in f:
        parts = line.strip().split('\t')
        cpg = parts[0].strip('"')
        betas = []
        for v in parts[1:]:
            v = v.strip('"')
            try:
                val = float(v) if v and v != 'NA' else None
            except ValueError:
                val = None
            betas.append(val)
        cpg_data[cpg] = betas
print(f"Loaded {len(cpg_data)} CpGs with {len(list(cpg_data.values())[0])} samples each")

# Which CpGs available per class
available_by_class = {}
for cls, cpgs in CLASS_CPGS.items():
    avail = [c for c in cpgs if c in cpg_data]
    available_by_class[cls] = avail
    print(f"  {cls:<10}: {len(avail)}/{len(cpgs)} CpGs available")

# Attach per-sample metrics
for i, sample in enumerate(METADATA):
    for cls, cpgs_avail in available_by_class.items():
        betas = [cpg_data[c][i] for c in cpgs_avail if cpg_data[c][i] is not None]
        if betas:
            mean_b = sum(betas)/len(betas)
            sample[f'{cls}_mean_beta'] = mean_b
            sample[f'{cls}_A_mean'] = A_score(mean_b, cls)
            per_cpg_A = [A_score(b, cls) for b in betas]
            per_cpg_A = [a for a in per_cpg_A if a is not None]
            if len(per_cpg_A) > 1:
                sample[f'{cls}_A_sd'] = statistics.stdev(per_cpg_A)
                sample[f'{cls}_A_per_cpg_mean'] = statistics.mean(per_cpg_A)
                sample[f'{cls}_A_max'] = max(per_cpg_A)

controls = [s for s in METADATA if not s['is_case']]
breast_cases = [s for s in METADATA if s['is_case'] and s.get('cancer_type') == 'C50']
print(f"\nn_controls={len(controls)}, n_breast={len(breast_cases)}")

def cohens_d(x, y):
    if len(x) < 2 or len(y) < 2: return 0, 0, 1.0
    m_x = statistics.mean(x); m_y = statistics.mean(y)
    s_x = statistics.stdev(x); s_y = statistics.stdev(y)
    pooled = math.sqrt(((len(x)-1)*s_x**2 + (len(y)-1)*s_y**2) / (len(x)+len(y)-2))
    d = (m_y - m_x)/pooled if pooled else 0
    se = math.sqrt(s_x**2/len(x) + s_y**2/len(y))
    z = (m_y - m_x)/se if se else 0
    p = math.erfc(abs(z)/math.sqrt(2))
    return d, m_y-m_x, p

# ─── ANALYSIS 1: ARCHITECTURAL VARIANCE (SD across CpGs per patient) ────
print("\n" + "="*78)
print("ANALYSIS 1: ARCHITECTURAL VARIANCE (disorder across CpGs)")
print("="*78)
print(f"{'Class':<10} {'ctrl_sd':>9} {'case_sd':>9} {'Δsd':>9} {'d':>7} {'p':>10}")
for cls in ['immune', 'cycling', 'secretory', 'stromal']:
    c = [s[f'{cls}_A_sd'] for s in controls if f'{cls}_A_sd' in s]
    x = [s[f'{cls}_A_sd'] for s in breast_cases if f'{cls}_A_sd' in s]
    if not c or not x: continue
    d, delta, p = cohens_d(c, x)
    mc, mx = statistics.mean(c), statistics.mean(x)
    print(f"  {cls:<9} {mc:>9.4f} {mx:>9.4f} {delta:>+9.5f} {d:>+7.3f} {p:>10.2e}")

# ─── ANALYSIS 2: MULTI-CLASS SIGNATURE ─────────────────────────────────
print("\n" + "="*78)
print("ANALYSIS 2: MULTI-CLASS SIGNATURE (combined 4-class deviation)")
print("="*78)

# For each class: compute control mean and SD
class_ctrl_stats = {}
for cls in ['immune', 'cycling', 'secretory', 'stromal']:
    vals = [s[f'{cls}_A_mean'] for s in controls if f'{cls}_A_mean' in s]
    class_ctrl_stats[cls] = {'mean': statistics.mean(vals), 'sd': statistics.stdev(vals)}

# Per sample, compute z-score-weighted sum across classes
for s in METADATA:
    # Sum of |z-deviations|
    z_sum_abs = 0
    z_sum_signed = 0
    for cls in ['immune', 'cycling', 'secretory', 'stromal']:
        a = s.get(f'{cls}_A_mean')
        if a is None: continue
        stats_cls = class_ctrl_stats[cls]
        z = (a - stats_cls['mean']) / stats_cls['sd']
        z_sum_abs += abs(z)
        z_sum_signed += z
    s['mcs_abs'] = z_sum_abs
    s['mcs_signed'] = z_sum_signed

for metric in ['mcs_abs', 'mcs_signed']:
    c = [s[metric] for s in controls]
    x = [s[metric] for s in breast_cases]
    d, delta, p = cohens_d(c, x)
    print(f"  {metric:<12} ctrl={statistics.mean(c):+.3f}  case={statistics.mean(x):+.3f}  d={d:+.3f}  p={p:.2e}")

# ─── ANALYSIS 3: DIRECTIONAL DRIFT ─────────────────────────────────────
print("\n" + "="*78)
print("ANALYSIS 3: DIRECTIONAL DRIFT (signed per-CpG mean β difference)")
print("="*78)
print("Xu 2019 reported 71.6% of significant CpGs show LOWER β in cases")
print("Testing in EPIC-Italy cohort:")

n_lower_total = 0; n_higher_total = 0
all_deltas = []
for cpg in cpg_data:
    ctrl_b = [cpg_data[cpg][i] for i,s in enumerate(METADATA) 
              if not s['is_case'] and cpg_data[cpg][i] is not None]
    case_b = [cpg_data[cpg][i] for i,s in enumerate(METADATA)
              if s['is_case'] and s.get('cancer_type')=='C50' and cpg_data[cpg][i] is not None]
    if len(ctrl_b) >= 20 and len(case_b) >= 20:
        delta = statistics.mean(case_b) - statistics.mean(ctrl_b)
        all_deltas.append((cpg, delta))
        if delta < 0: n_lower_total += 1
        else: n_higher_total += 1

n = n_lower_total + n_higher_total
p_bin = math.erfc(abs(n_lower_total - n*0.5) / math.sqrt(n*0.25) / math.sqrt(2))
print(f"  Across {n} CpGs: {n_lower_total} lower in cases ({n_lower_total/n*100:.0f}%), {n_higher_total} higher")
print(f"  Binomial test against 50/50 null: p = {p_bin:.3e}")
print(f"  Xu 2019 reported 71.6% lower in SIGNIFICANT CpGs — our {n_lower_total/n*100:.0f}% across ALL panel CpGs")

# ─── ANALYSIS 4: TOP-N DISCRIMINATIVE + CROSS-VALIDATION ────────────────
print("\n" + "="*78)
print("ANALYSIS 4: DATA-DRIVEN TOP-N CpGs with honest held-out validation")
print("="*78)

# Rank CpGs by Cohen's d in FULL dataset (descriptive only)
full_d = {}
for cpg in cpg_data:
    ctrl_b = [cpg_data[cpg][i] for i,s in enumerate(METADATA)
              if not s['is_case'] and cpg_data[cpg][i] is not None]
    case_b = [cpg_data[cpg][i] for i,s in enumerate(METADATA)
              if s['is_case'] and s.get('cancer_type')=='C50' and cpg_data[cpg][i] is not None]
    if len(ctrl_b) >= 20 and len(case_b) >= 20:
        d, _, _ = cohens_d(ctrl_b, case_b)
        full_d[cpg] = d

ranked = sorted(full_d.items(), key=lambda x: abs(x[1]), reverse=True)
print(f"\nTop 15 CpGs by |Cohen's d| (full-dataset, DESCRIPTIVE):")
for cpg, d in ranked[:15]:
    print(f"  {cpg}  d = {d:+.3f}")

# Cross-validated assessment: 10 random splits
print("\n10-iteration held-out validation:")
random.seed(42)
ctrl_idx = [i for i,s in enumerate(METADATA) if not s['is_case']]
case_idx = [i for i,s in enumerate(METADATA) if s['is_case'] and s.get('cancer_type')=='C50']

d_test_list = []
d_train_list = []
for iteration in range(10):
    random.shuffle(ctrl_idx); random.shuffle(case_idx)
    ctrl_tr = ctrl_idx[:len(ctrl_idx)//2]; ctrl_te = ctrl_idx[len(ctrl_idx)//2:]
    case_tr = case_idx[:len(case_idx)//2]; case_te = case_idx[len(case_idx)//2:]
    
    # Find top 10 on train
    train_d = {}
    for cpg in cpg_data:
        c = [cpg_data[cpg][i] for i in ctrl_tr if cpg_data[cpg][i] is not None]
        x = [cpg_data[cpg][i] for i in case_tr if cpg_data[cpg][i] is not None]
        if len(c) >= 5 and len(x) >= 5:
            d, _, _ = cohens_d(c, x)
            train_d[cpg] = d
    top10 = sorted(train_d.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    
    # Score = signed sum of β based on train direction
    def score(i):
        return sum(math.copysign(1, train_d[c]) * cpg_data[c][i] 
                   for c, _ in top10 if cpg_data[c][i] is not None)
    
    ctrl_scores_te = [score(i) for i in ctrl_te]
    case_scores_te = [score(i) for i in case_te]
    ctrl_scores_tr = [score(i) for i in ctrl_tr]
    case_scores_tr = [score(i) for i in case_tr]
    
    d_te, _, _ = cohens_d(ctrl_scores_te, case_scores_te)
    d_tr, _, _ = cohens_d(ctrl_scores_tr, case_scores_tr)
    d_test_list.append(d_te)
    d_train_list.append(d_tr)
    print(f"  Iter {iteration+1}: train d={d_tr:+.3f}, test d={d_te:+.3f}")

print(f"\nMean test Cohen's d: {statistics.mean(d_test_list):+.3f} ± {statistics.stdev(d_test_list):.3f}")
print(f"Mean train Cohen's d: {statistics.mean(d_train_list):+.3f}")
print(f"Train-test gap = overfitting: {statistics.mean(d_train_list) - statistics.mean(d_test_list):+.3f}")

# Save all
out = {
    'analysis_1_variance_by_class': {
        cls: {'ctrl_mean_sd': statistics.mean([s[f'{cls}_A_sd'] for s in controls if f'{cls}_A_sd' in s]),
              'case_mean_sd': statistics.mean([s[f'{cls}_A_sd'] for s in breast_cases if f'{cls}_A_sd' in s])}
        for cls in ['immune','cycling','secretory','stromal']
    },
    'analysis_3_directional': {
        'n_lower_in_cases': n_lower_total,
        'n_higher_in_cases': n_higher_total,
        'fraction_lower': n_lower_total/n,
        'binomial_p': p_bin,
    },
    'analysis_4_crossval': {
        'mean_test_cohen_d': statistics.mean(d_test_list),
        'std_test_cohen_d': statistics.stdev(d_test_list),
        'mean_train_cohen_d': statistics.mean(d_train_list),
        'overfitting_gap': statistics.mean(d_train_list) - statistics.mean(d_test_list),
        'top_cpgs_full_dataset': [{'cpg': c, 'd': d} for c, d in ranked[:20]],
    },
}
Path('/home/claude/geo_analysis/VAL_047_extended_results.json').write_text(json.dumps(out, indent=1))
print(f"\nSaved: VAL_047_extended_results.json")
