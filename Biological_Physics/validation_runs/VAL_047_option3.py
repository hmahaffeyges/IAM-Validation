#!/usr/bin/env python3
"""
VAL-047 OPTION 3: GAPE on diagnosed TCGA breast tumor tissue (GSE69914)
========================================================================

This is the critical large-signal validation. On EPIC-Italy pre-diagnostic
blood, we saw Cohen's d = 0.60 with the Xu-CpG directional approach. On
diagnosed tumor tissue (where effect is known to be 5-10x larger), the same
framework should read d > 1.5 if GAPE scales properly from subtle pre-dx
drift to overt tumor architecture.

Groups (from status_code):
  0 = normal (50)       — healthy donor breast tissue
  1 = normal-adjacent (42) — tissue adjacent to tumors (field effect)
  2 = breast-cancer (305)  — the tumors themselves
  
Three comparisons:
  A. Tumor vs healthy normal: should give d_massive (>2)
  B. Normal-adjacent vs healthy normal: VAL-037-style field effect (d~0.3-0.5)
  C. Trajectory: normal → adj → tumor (should be monotonic)
"""
import json, math, statistics
from pathlib import Path

SAMPLES = json.loads(Path('/home/claude/geo_analysis/tcga_brca/GSE69914_metadata.json').read_text())

H_MIN_IMMUNE = 0.838889
H_MIN_SECRETORY = 0.843264

def H(b):
    if b is None or (isinstance(b,float) and math.isnan(b)): return None
    if b <= 0 or b >= 1: return 0.0
    return -b*math.log2(b) - (1-b)*math.log2(1-b)
def A(b, H_min): 
    h = H(b)
    return None if h is None else h/H_min

def load_cpg_file(fp, n_expected=407):
    data = {}
    with open(fp) as f:
        for line in f:
            parts = line.strip().split('\t')
            cpg = parts[0].strip('"')
            betas = []
            for v in parts[1:]:
                v = v.strip('"')
                try: betas.append(float(v) if v else None)
                except ValueError: betas.append(None)
            while len(betas) < n_expected:
                betas.append(None)
            data[cpg] = betas[:n_expected]
    return data

cpg_data = load_cpg_file('/home/claude/geo_analysis/tcga_brca/gse69914_cpgs.tsv', 
                         n_expected=len(SAMPLES))
print(f"Loaded {len(cpg_data)} CpGs × {len(SAMPLES)} samples")

def cohens_d(x, y):
    if len(x)<2 or len(y)<2: return 0, 0, 1.0
    mx, my = statistics.mean(x), statistics.mean(y)
    sx, sy = statistics.stdev(x), statistics.stdev(y)
    pooled = math.sqrt(((len(x)-1)*sx**2 + (len(y)-1)*sy**2) / (len(x)+len(y)-2))
    d = (my-mx)/pooled if pooled else 0
    se = math.sqrt(sx**2/len(x) + sy**2/len(y))
    z = (my-mx)/se if se else 0
    p = math.erfc(abs(z)/math.sqrt(2))
    return d, my-mx, p

# Groups
normal = [i for i,s in enumerate(SAMPLES) if s.get('status_code')==0]
adjacent = [i for i,s in enumerate(SAMPLES) if s.get('status_code')==1]
tumor = [i for i,s in enumerate(SAMPLES) if s.get('status_code')==2]
normal_brca1 = [i for i,s in enumerate(SAMPLES) if s.get('status_code')==3]
cancer_brca1 = [i for i,s in enumerate(SAMPLES) if s.get('status_code')==4]

print(f"\nGroup sizes:")
print(f"  Healthy normal breast:      {len(normal)}")
print(f"  Normal-adjacent (to tumor): {len(adjacent)}")
print(f"  Breast tumor:               {len(tumor)}")
print(f"  Normal BRCA1 carriers:      {len(normal_brca1)}")
print(f"  Cancer BRCA1 carriers:      {len(cancer_brca1)}")

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS A: Per-CpG effect sizes — healthy normal vs tumor
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*78)
print("A. XU 2019 PRIORITY CpGs ON DIAGNOSED TUMOR vs HEALTHY NORMAL BREAST")
print("="*78)
print("Same 6 CpGs that gave us d=0.60 on EPIC-Italy pre-dx blood")
print(f"{'CpG':<12} {'healthy β':>10} {'tumor β':>10} {'Δβ':>8} {'d':>8} {'p':>10}")

XU_PRIORITY = ['cg03430067','cg03616357','cg07072643','cg08287471','cg19709625','cg26203572']
tumor_cpg_stats = {}
for cpg in XU_PRIORITY:
    if cpg not in cpg_data: continue
    vals = cpg_data[cpg]
    n_b = [vals[i] for i in normal if vals[i] is not None]
    t_b = [vals[i] for i in tumor if vals[i] is not None]
    if len(n_b)<5 or len(t_b)<5: continue
    m_n, m_t = statistics.mean(n_b), statistics.mean(t_b)
    d, delta, p = cohens_d(n_b, t_b)
    tumor_cpg_stats[cpg] = {'healthy_b': m_n, 'tumor_b': m_t, 'delta': delta, 'd': d}
    print(f"{cpg:<12} {m_n:>10.4f} {m_t:>10.4f} {delta:>+8.4f} {d:>+8.3f} {p:>10.2e}")

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS B: GAPE A-score secretory class
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*78)
print("B. GAPE SECRETORY-CLASS A-SCORE: healthy vs adjacent vs tumor")
print("="*78)

SECRETORY_CPGS = [
    'cg16867657','cg06639320','cg13552692','cg11807280','cg19283806',
    'cg02580606','cg22454769','cg02228185','cg06691716','cg00846300',
    'cg01127300','cg26521404','cg08262002','cg18181703',
    'cg09809672','cg22736354','cg02489552','cg26203572','cg25382485',
]
sec_cpgs = [c for c in SECRETORY_CPGS if c in cpg_data]
print(f"Secretory CpGs available: {len(sec_cpgs)}")

# Per-sample metrics
for i, s in enumerate(SAMPLES):
    per_cpg_A = []
    for cpg in sec_cpgs:
        v = cpg_data[cpg][i]
        if v is not None:
            a = A(v, H_MIN_SECRETORY)
            if a is not None:
                per_cpg_A.append(a)
    if len(per_cpg_A) >= 3:
        s['sec_A_mean'] = statistics.mean(per_cpg_A)
        s['sec_A_sd'] = statistics.stdev(per_cpg_A)
        s['sec_A_max'] = max(per_cpg_A)

def group_stats(idx_list, metric):
    vals = [SAMPLES[i][metric] for i in idx_list if metric in SAMPLES[i]]
    return {'n': len(vals), 'mean': statistics.mean(vals) if vals else 0,
            'sd': statistics.stdev(vals) if len(vals)>1 else 0, 'vals': vals}

print(f"\n{'Group':<25} {'n':>4} {'A_mean':>9} {'A_sd':>7} {'A_max':>8}")
for lbl, idx in [('Healthy normal', normal), ('Normal-adjacent', adjacent), ('Breast tumor', tumor)]:
    st_m = group_stats(idx, 'sec_A_mean')
    st_s = group_stats(idx, 'sec_A_sd')
    st_x = group_stats(idx, 'sec_A_max')
    print(f"{lbl:<25} {st_m['n']:>4} {st_m['mean']:>9.4f} "
          f"{st_s['mean']:>7.4f} {st_x['mean']:>8.4f}")

# Key comparisons
print(f"\nSECRETORY A_MEAN effect sizes (should be LARGE — diagnosed tumor tissue):")
for lbl, idx in [('Normal-adj vs Healthy', adjacent), ('Tumor vs Healthy', tumor)]:
    h = [SAMPLES[i]['sec_A_mean'] for i in normal if 'sec_A_mean' in SAMPLES[i]]
    t = [SAMPLES[i]['sec_A_mean'] for i in idx if 'sec_A_mean' in SAMPLES[i]]
    d, delta, p = cohens_d(h, t)
    print(f"  {lbl:<30}  ΔA={delta:+.5f}  d={d:+.3f}  p={p:.2e}")

print(f"\nSECRETORY A_SD (architectural disorder within each sample):")
for lbl, idx in [('Normal-adj vs Healthy', adjacent), ('Tumor vs Healthy', tumor)]:
    h = [SAMPLES[i]['sec_A_sd'] for i in normal if 'sec_A_sd' in SAMPLES[i]]
    t = [SAMPLES[i]['sec_A_sd'] for i in idx if 'sec_A_sd' in SAMPLES[i]]
    d, delta, p = cohens_d(h, t)
    print(f"  {lbl:<30}  ΔSD={delta:+.5f}  d={d:+.3f}  p={p:.2e}")

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS C: Individual-level ROC / detection performance
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*78)
print("C. INDIVIDUAL-LEVEL DETECTION: healthy normal vs tumor")
print("="*78)

# Use secretory A_mean as single classifier
h_vals = [SAMPLES[i]['sec_A_mean'] for i in normal if 'sec_A_mean' in SAMPLES[i]]
t_vals = [SAMPLES[i]['sec_A_mean'] for i in tumor if 'sec_A_mean' in SAMPLES[i]]

# Find ROC
all_vals = sorted(set(h_vals + t_vals))
best_auc = 0
best_thr = 0
rocpoints = []
for thr in all_vals:
    tp = sum(1 for v in t_vals if v >= thr)
    fn = len(t_vals) - tp
    fp = sum(1 for v in h_vals if v >= thr)
    tn = len(h_vals) - fp
    sens = tp / (tp+fn) if (tp+fn) else 0
    spec = tn / (tn+fp) if (tn+fp) else 0
    rocpoints.append((1-spec, sens, thr))

# AUC via trapezoidal
rocpoints.sort()
auc = 0
for i in range(1, len(rocpoints)):
    auc += (rocpoints[i][0] - rocpoints[i-1][0]) * (rocpoints[i][1] + rocpoints[i-1][1]) / 2
print(f"Secretory A_mean classifier:")
print(f"  AUC (healthy vs tumor) = {auc:.4f}")

# Performance at 95% specificity
target_spec = 0.95
best = min(rocpoints, key=lambda r: abs((1-r[0]) - target_spec))
print(f"  At specificity = {1-best[0]:.2%}: sensitivity = {best[1]:.2%}")

# 99% specificity
target_spec = 0.99
best = min(rocpoints, key=lambda r: abs((1-r[0]) - target_spec))
print(f"  At specificity = {1-best[0]:.2%}: sensitivity = {best[1]:.2%}")

# Distribution quality check
print(f"\nSecretory A_mean distributions:")
print(f"  Healthy normal (n={len(h_vals)}): mean = {statistics.mean(h_vals):.4f}, range [{min(h_vals):.3f}, {max(h_vals):.3f}]")
print(f"  Tumor (n={len(t_vals)}):         mean = {statistics.mean(t_vals):.4f}, range [{min(t_vals):.3f}, {max(t_vals):.3f}]")

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS D: Field-effect monotonicity
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*78)
print("D. ARCHITECTURAL FIELD-EFFECT MONOTONICITY (VAL-037 pattern)")  
print("="*78)
print("Prediction: A_mean_healthy < A_mean_adjacent < A_mean_tumor (monotonic drift)")

h_A = statistics.mean([SAMPLES[i]['sec_A_mean'] for i in normal if 'sec_A_mean' in SAMPLES[i]])
a_A = statistics.mean([SAMPLES[i]['sec_A_mean'] for i in adjacent if 'sec_A_mean' in SAMPLES[i]])
t_A = statistics.mean([SAMPLES[i]['sec_A_mean'] for i in tumor if 'sec_A_mean' in SAMPLES[i]])
print(f"  Healthy A = {h_A:.4f}")
print(f"  Adjacent A = {a_A:.4f}  (Δ_from_healthy = {a_A - h_A:+.4f})")
print(f"  Tumor A   = {t_A:.4f}  (Δ_from_healthy = {t_A - h_A:+.4f})")
monotonic = (h_A < a_A < t_A) or (h_A > a_A > t_A)
print(f"  Monotonic architectural drift: {'YES — confirms VAL-037 field effect' if monotonic else 'NO'}")

# Save
out = {
    'n_healthy_normal': len(normal),
    'n_normal_adjacent': len(adjacent),
    'n_tumor': len(tumor),
    'xu_priority_cpgs_on_tumor_vs_normal': tumor_cpg_stats,
    'secretory_Amean': {
        'healthy_mean': h_A,
        'adjacent_mean': a_A,
        'tumor_mean': t_A,
        'auc_healthy_vs_tumor': auc,
        'monotonic_field_effect': monotonic,
    },
}
Path('/home/claude/geo_analysis/tcga_brca/VAL_047_option3_results.json').write_text(json.dumps(out, indent=1, default=str))
print(f"\nSaved: VAL_047_option3_results.json")
