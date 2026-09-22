#!/usr/bin/env python3
"""
VAL-047 OPTIONS 1 & 2: 
  1. Apply GAPE A-score to Xu 2019's strongly-replicated breast-cancer CpGs
     with proper per-CpG directional awareness
  2. Stratify the secretory-variance finding (d=-0.39, p=5e-4) by time-to-dx
     to determine if it's a real pre-diagnostic signal or late-stage artifact
"""
import json, math, statistics, random
from pathlib import Path

# ─── Calibration layer (access restricted) ──────────────────────────────
# Per-class H_min values are proprietary — covered under US Provisional Patents
# 64/012,720 and 64/014,568. For NDA access, contact hmahaffeyges@gmail.com.
try:
    from _gape_calibration_private import (
        H_MIN_IMMUNE, H_MIN_SECRETORY, H_MIN_CYCLING, H_MIN_TERMINAL,
        H_MIN_STROMAL, H_MIN_PROGENITOR, H_MIN_STEM_ADULT, H_MIN_STEM_PLURI,
    )
except ImportError:
    raise RuntimeError(
        "Calibration layer not available. "
        "See https://github.com/hmahaffeyges/IAM-Validation for access instructions."
    )


METADATA = json.loads(Path('GSE51057_metadata.json').read_text())
CANDIDATE_INFO = json.loads(Path('xu_candidate_cpgs.json').read_text())

# Key CpG sets
XU_REPLICATED = CANDIDATE_INFO['replicated_5_Bonferroni_EPIC']
XU_TOP_HIT = 'cg26203572'  # LINC00525, p=2e-33 in Sister Study


def H(b):
    if b is None or (isinstance(b,float) and math.isnan(b)): return None
    if b <= 0 or b >= 1: return 0.0
    return -b*math.log2(b) - (1-b)*math.log2(1-b)
def A(b, H_min=H_MIN_IMMUNE):
    h = H(b)
    return None if h is None else h/H_min

# Load β values
def load_cpg_file(fp, n_expected=329):
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
            # Pad with None if ragged
            while len(betas) < n_expected:
                betas.append(None)
            data[cpg] = betas[:n_expected]
    return data

xu_data = load_cpg_file('xu_candidate_hits.tsv')
# Also load the earlier CpG set (49 CpGs used in extended analysis)
main_data = load_cpg_file('cpg_hits.tsv')
# Merge
for c, b in xu_data.items():
    if c not in main_data:
        main_data[c] = b
print(f"Total CpGs loaded: {len(main_data)}")

# ═══════════════════════════════════════════════════════════════════════════
# OPTION 1: GAPE A-SCORE ON XU'S STRONGLY-REPLICATED CPGs WITH DIRECTIONALITY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*78)
print("OPTION 1: GAPE WITH PROPER DIRECTIONAL AWARENESS ON XU 2019 CpGs")
print("="*78)

# For each Xu CpG, determine case-vs-control direction in our data (EPIC-Italy)
# These are the 6 strongest-evidence CpGs from Xu 2019 — the cleanest possible test
xu_all = XU_REPLICATED + [XU_TOP_HIT]
print(f"\nPer-CpG analysis of Xu's {len(xu_all)} strongly-replicated priority CpGs:")
print(f"{'CpG':<15} {'ctrl β':>10} {'case β':>10} {'Δβ':>8} {'ctrl A':>9} {'case A':>9} {'ΔA':>9} {'d':>7} {'p':>10}")

# Attach a per-CpG mini-analysis
cpg_stats = {}
for cpg in xu_all:
    if cpg not in main_data: continue
    vals = main_data[cpg]
    ctrl_b = [vals[i] for i,s in enumerate(METADATA) if not s['is_case'] and vals[i] is not None]
    case_b = [vals[i] for i,s in enumerate(METADATA) 
              if s['is_case'] and s.get('cancer_type')=='C50' and vals[i] is not None]
    if len(ctrl_b)<10 or len(case_b)<10: continue
    ctrl_m_b, case_m_b = statistics.mean(ctrl_b), statistics.mean(case_b)
    ctrl_sd_b, case_sd_b = statistics.stdev(ctrl_b), statistics.stdev(case_b)
    delta_b = case_m_b - ctrl_m_b
    pooled_b = math.sqrt(((len(ctrl_b)-1)*ctrl_sd_b**2 + (len(case_b)-1)*case_sd_b**2) / 
                         (len(ctrl_b)+len(case_b)-2))
    d = delta_b/pooled_b if pooled_b else 0
    se = math.sqrt(ctrl_sd_b**2/len(ctrl_b) + case_sd_b**2/len(case_b))
    z = delta_b/se if se else 0
    p = math.erfc(abs(z)/math.sqrt(2))
    
    # GAPE A-score at these β values (using secretory H_min — breast is secretory class)
    ctrl_A = A(ctrl_m_b, H_MIN_SECRETORY)
    case_A = A(case_m_b, H_MIN_SECRETORY)
    dA = (case_A - ctrl_A) if (ctrl_A and case_A) else None
    
    cpg_stats[cpg] = {'ctrl_b': ctrl_m_b, 'case_b': case_m_b, 'delta_b': delta_b, 'd': d, 'p': p}
    
    dA_str = f"{dA:+.5f}" if dA is not None else "  N/A"
    print(f"{cpg:<15} {ctrl_m_b:>10.4f} {case_m_b:>10.4f} {delta_b:>+8.4f} "
          f"{ctrl_A or 0:>9.4f} {case_A or 0:>9.4f} {dA_str:>9} {d:>+7.3f} {p:>10.2e}")

# Build PROPER mBCRS-style score: per-CpG signed sum with directional weights from Xu 2019 direction
# Xu 2019 directions: cg26203572 was DISCORDANT between studies, so use within-EPIC direction
# Use d from CURRENT data as the direction and weight (this is what the published method does)
print("\nProper weighted score (mBCRS-style) on Xu CpGs:")
for i, sample in enumerate(METADATA):
    score = 0
    n_used = 0
    for cpg, cs in cpg_stats.items():
        val = main_data[cpg][i]
        if val is None: continue
        # Weight = Cohen's d (magnitude × direction); standardize β first
        # Center on control mean
        centered = val - cs['ctrl_b']
        score += math.copysign(cs['d'], cs['d']) * centered
        n_used += 1
    sample['xu_score'] = score
    sample['xu_score_n_cpgs'] = n_used

controls = [s for s in METADATA if not s['is_case']]
breast_cases = [s for s in METADATA if s['is_case'] and s.get('cancer_type')=='C50']

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

ctrl_scores = [s['xu_score'] for s in controls]
case_scores = [s['xu_score'] for s in breast_cases]
d_xu, delta_xu, p_xu = cohens_d(ctrl_scores, case_scores)
print(f"  Cross-patient comparison of Xu-CpG directional score:")
print(f"  ctrl = {statistics.mean(ctrl_scores):+.5f} ± {statistics.stdev(ctrl_scores):.5f}")
print(f"  case = {statistics.mean(case_scores):+.5f} ± {statistics.stdev(case_scores):.5f}")
print(f"  Cohen's d = {d_xu:+.3f}  p = {p_xu:.2e}")
print(f"\n  ⚠ NOTE: this d is INFLATED — weights derived from same data. Need CV.")

# Proper cross-validation of this approach
print("\nHonest 10-fold cross-validated Xu-score Cohen's d:")
random.seed(42)
ctrl_idx = [i for i,s in enumerate(METADATA) if not s['is_case']]
case_idx = [i for i,s in enumerate(METADATA) if s['is_case'] and s.get('cancer_type')=='C50']
d_cv = []
for it in range(10):
    random.shuffle(ctrl_idx); random.shuffle(case_idx)
    ctrl_tr = ctrl_idx[:len(ctrl_idx)//2]; ctrl_te = ctrl_idx[len(ctrl_idx)//2:]
    case_tr = case_idx[:len(case_idx)//2]; case_te = case_idx[len(case_idx)//2:]
    # Compute weights on train
    train_weights = {}
    for cpg in xu_all:
        if cpg not in main_data: continue
        vals = main_data[cpg]
        c = [vals[i] for i in ctrl_tr if vals[i] is not None]
        x = [vals[i] for i in case_tr if vals[i] is not None]
        if len(c)<5 or len(x)<5: continue
        d, _, _ = cohens_d(c, x)
        train_weights[cpg] = {'d': d, 'ctrl_mean': statistics.mean(c)}
    # Score test samples
    def sc(i):
        s = 0
        for cpg, w in train_weights.items():
            v = main_data[cpg][i]
            if v is not None:
                s += math.copysign(w['d'], w['d']) * (v - w['ctrl_mean'])
        return s
    ctrl_sc = [sc(i) for i in ctrl_te]
    case_sc = [sc(i) for i in case_te]
    d_test, _, _ = cohens_d(ctrl_sc, case_sc)
    d_cv.append(d_test)
print(f"  {[f'{d:+.3f}' for d in d_cv]}")
print(f"  Mean CV Cohen's d on Xu's 6 CpGs: {statistics.mean(d_cv):+.3f} ± {statistics.stdev(d_cv):.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# OPTION 2: SECRETORY VARIANCE STRATIFIED BY TIME TO DIAGNOSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*78)
print("OPTION 2: SECRETORY-CLASS VARIANCE STRATIFIED BY TIME TO DIAGNOSIS")
print("="*78)
print("Earlier finding: secretory A-score SD lower in cases, d=-0.39, p=5.4e-4")
print("Question: is this a pre-diagnostic signal or a late-stage artifact?")

# Rebuild per-sample secretory A-score SD (same method as before, using cpg_hits.tsv data)
# Secretory-class CpGs
SECRETORY_CPGS_RAW = [
    'cg16867657','cg06639320','cg13552692','cg11807280','cg19283806',
    'cg02580606','cg22454769','cg02228185','cg06691716','cg00846300',
    'cg01127300','cg26521404','cg08262002','cg18181703',
    'cg09809672','cg22736354','cg02489552','cg26203572','cg25382485',
]
sec_cpgs = [c for c in SECRETORY_CPGS_RAW if c in main_data]
print(f"\nSecretory CpGs available: {len(sec_cpgs)}")

for i, s in enumerate(METADATA):
    per_cpg_A = []
    for cpg in sec_cpgs:
        v = main_data[cpg][i]
        if v is not None:
            a = A(v, H_MIN_SECRETORY)
            if a is not None:
                per_cpg_A.append(a)
    if len(per_cpg_A) >= 2:
        s['sec_A_sd'] = statistics.stdev(per_cpg_A)

# Stratify breast cases by time to diagnosis
print("\nSecretory A-score SD by time-to-diagnosis window:")
print(f"{'Group':<30} {'n':>5} {'mean_SD':>10} {'d vs ctrl':>11} {'p':>10}")

ctrl_sec_sd = [s['sec_A_sd'] for s in controls if 'sec_A_sd' in s]
ctrl_mean = statistics.mean(ctrl_sec_sd)
print(f"{'Controls':<30} {len(ctrl_sec_sd):>5} {ctrl_mean:>10.4f}")

windows = [
    (0, 1, '≤1 yr pre-dx (imminent)'),
    (1, 2, '1-2 yr pre-dx'),
    (2, 5, '2-5 yr pre-dx'),
    (5, 10, '5-10 yr pre-dx'),
    (10, 20, '>10 yr pre-dx (early)'),
]
for lo, hi, label in windows:
    subset = [s for s in breast_cases 
              if 'years_to_dx' in s and lo <= s['years_to_dx'] < hi and 'sec_A_sd' in s]
    if len(subset) < 5:
        print(f"{label:<30} {len(subset):>5} (insufficient)")
        continue
    sub_sd = [s['sec_A_sd'] for s in subset]
    d, delta, p = cohens_d(ctrl_sec_sd, sub_sd)
    print(f"{label:<30} {len(subset):>5} {statistics.mean(sub_sd):>10.4f} {d:>+11.3f} {p:>10.2e}")

# Split at yr_to_dx = 2: imminent (likely tumor-shed signal) vs early (true prediction)
imminent = [s for s in breast_cases if 'years_to_dx' in s and s['years_to_dx']<=2 and 'sec_A_sd' in s]
early = [s for s in breast_cases if 'years_to_dx' in s and s['years_to_dx']>5 and 'sec_A_sd' in s]
if imminent and early:
    im_sd = [s['sec_A_sd'] for s in imminent]
    ea_sd = [s['sec_A_sd'] for s in early]
    print(f"\nIMMINENT (≤2 yr, n={len(imminent)}): mean_sd = {statistics.mean(im_sd):.4f}")
    print(f"EARLY (>5 yr, n={len(early)}):    mean_sd = {statistics.mean(ea_sd):.4f}")
    d, _, p = cohens_d(ctrl_sec_sd, im_sd)
    print(f"Imminent vs controls: d={d:+.3f}, p={p:.2e}")
    d, _, p = cohens_d(ctrl_sec_sd, ea_sd)
    print(f"Early vs controls:    d={d:+.3f}, p={p:.2e}")

# Verdict
print("\n" + "="*78)
print("INTERPRETATION")
print("="*78)
print("If imminent (≤2yr) shows strongest effect → 'tumor-shed / late-stage artifact'")
print("If early (>5yr) shows strongest effect → 'genuine pre-diagnostic drift signal'")
print("If all windows show similar effect → 'stable architectural marker'")

# Save
out = {
    'option_1_xu_cpgs_directional': {
        'cpg_stats': cpg_stats,
        'cross_validated_cohen_d_mean': statistics.mean(d_cv),
        'cross_validated_cohen_d_std': statistics.stdev(d_cv),
        'per_iteration_d': d_cv,
    },
    'option_2_secretory_variance_by_time': {
        'ctrl_mean_sd': ctrl_mean,
        'n_secretory_cpgs': len(sec_cpgs),
        'windows': {f'{lo}-{hi}': {
            'n': len([s for s in breast_cases 
                     if 'years_to_dx' in s and lo<=s['years_to_dx']<hi and 'sec_A_sd' in s]),
            'mean_sd': statistics.mean([s['sec_A_sd'] for s in breast_cases
                                        if 'years_to_dx' in s and lo<=s['years_to_dx']<hi 
                                        and 'sec_A_sd' in s]) if [s for s in breast_cases 
                                        if 'years_to_dx' in s and lo<=s['years_to_dx']<hi 
                                        and 'sec_A_sd' in s] else None,
        } for lo, hi, _ in windows},
    },
}
Path('VAL_047_option_1_2_results.json').write_text(json.dumps(out, indent=1, default=str))
print(f"\nSaved: VAL_047_option_1_2_results.json")
