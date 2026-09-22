#!/usr/bin/env python3
"""
GSE51032 REPLICATION: EPIC HuGeF, n=845
========================================
235 breast cancer pre-diagnostic + 166 colorectal pre-diagnostic + 424 controls.

Tests:
A. Xu-CpG directional score: can we replicate d=0.60 we saw in GSE51057?
B. Secretory variance signal: does the 10-year lead-time effect replicate?
C. COLORECTAL: does the same method work for colorectal cancer too?
D. Age-matched detection performance at realistic thresholds
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


SAMPLES = json.load(open('/home/claude/geo_analysis/gse51032/GSE51032_meta.json'))


def H(b):
    if b is None: return None
    if b <= 0 or b >= 1: return 0.0
    return -b*math.log2(b) - (1-b)*math.log2(1-b)
def A(b, Hmin):
    h = H(b)
    return None if h is None else h/Hmin

def load_tsv(fp, n_expected):
    data = {}
    with open(fp) as f:
        for line in f:
            parts = line.strip().split('\t')
            cpg = parts[0].strip('"')
            b = []
            for v in parts[1:]:
                v = v.strip('"')
                try: b.append(float(v) if v else None)
                except: b.append(None)
            while len(b) < n_expected: b.append(None)
            data[cpg] = b[:n_expected]
    return data

cpg_data = load_tsv('/home/claude/geo_analysis/gse51032/gse51032_cpgs.tsv', len(SAMPLES))
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

# Cohorts
controls = [i for i,s in enumerate(SAMPLES) if not s['is_case']]
breast = [i for i,s in enumerate(SAMPLES) if s.get('cancer_icd10')=='C50']
colon = [i for i,s in enumerate(SAMPLES) if s.get('cancer_icd10') in ('C18','C19','C20')]

print(f"\nControls: {len(controls)}, Breast: {len(breast)}, Colorectal: {len(colon)}")

# Secretory CpGs panel
SEC_CPGS = ['cg16867657','cg06639320','cg13552692','cg11807280','cg19283806',
            'cg02580606','cg22454769','cg02228185','cg06691716','cg00846300',
            'cg01127300','cg26521404','cg08262002','cg18181703',
            'cg09809672','cg22736354','cg02489552','cg26203572','cg25382485']
sec_avail = [c for c in SEC_CPGS if c in cpg_data]

# Immune CpGs panel (for cycling signal in colorectal, plus bulk immune)
IMM_CPGS_RAW = [
    'cg04023335','cg05045481','cg10632894','cg26614073','cg08706463',
    'cg19432188','cg17774019','cg18834029','cg00342758','cg10241600',
    'cg23244761','cg14614643','cg23555344','cg01127300','cg14620944',
    'cg20795519','cg24079702','cg07571933','cg25432518','cg00431549',
    'cg16867657','cg22736354','cg02228185','cg25809905','cg09809672',
    'cg02489552','cg12554573','cg17861230','cg22454769',
]
imm_avail = [c for c in IMM_CPGS_RAW if c in cpg_data]

# Cycling panel (colorectal class) 
CYC_CPGS = ['cg00292639','cg06639320','cg17861230','cg06291867','cg16781482',
            'cg02580606','cg22454769','cg06493994','cg23500537','cg01611260',
            'cg07955995','cg22512670','cg01127300','cg04474832','cg18181703',
            'cg13206721','cg14841770','cg08090640','cg08262002','cg26842024']
cyc_avail = [c for c in CYC_CPGS if c in cpg_data]

print(f"CpGs available: immune={len(imm_avail)}, secretory={len(sec_avail)}, cycling={len(cyc_avail)}")

# Compute per-sample metrics
for i, s in enumerate(SAMPLES):
    for panel_name, cpg_list, Hm in [
        ('sec', sec_avail, H_MIN_SECRETORY),
        ('imm', imm_avail, H_MIN_IMMUNE),
        ('cyc', cyc_avail, H_MIN_CYCLING),
    ]:
        per_cpg_A = []
        betas = []
        for c in cpg_list:
            v = cpg_data[c][i]
            if v is not None:
                a = A(v, Hm)
                if a is not None:
                    per_cpg_A.append(a)
                    betas.append(v)
        if len(per_cpg_A) >= 3:
            s[f'{panel_name}_A_mean'] = statistics.mean(per_cpg_A)
            s[f'{panel_name}_A_sd'] = statistics.stdev(per_cpg_A)
            s[f'{panel_name}_beta_mean'] = statistics.mean(betas)

# ═══════════════════════════════════════════════════════════════════════════
# A. XU-CPG DIRECTIONAL SCORE ON BREAST (replication of GSE51057 d=0.60)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*78)
print("A. XU-CPG SCORE REPLICATION: BREAST CANCER (GSE51032 n=235)")
print("="*78)

XU_PRIORITY = ['cg03430067','cg03616357','cg07072643','cg08287471','cg19709625','cg26203572']

# Per-CpG stats for breast
print(f"\n{'CpG':<12} {'ctrl β':>10} {'case β':>10} {'Δβ':>9} {'d':>7} {'p':>10}")
for cpg in XU_PRIORITY:
    if cpg not in cpg_data: continue
    vals = cpg_data[cpg]
    c = [vals[i] for i in controls if vals[i] is not None]
    x = [vals[i] for i in breast if vals[i] is not None]
    if len(c)<10 or len(x)<10: continue
    d, delta, p = cohens_d(c, x)
    print(f"{cpg:<12} {statistics.mean(c):>10.4f} {statistics.mean(x):>10.4f} "
          f"{delta:>+9.4f} {d:>+7.3f} {p:>10.2e}")

# Cross-validated Xu-CpG directional score on BREAST
random.seed(42)
def run_xu_cv(case_idx, label, n_iter=10):
    d_cv = []
    ctrl_pool = controls.copy()
    case_pool = case_idx.copy()
    for _ in range(n_iter):
        random.shuffle(ctrl_pool); random.shuffle(case_pool)
        c_tr = ctrl_pool[:len(ctrl_pool)//2]; c_te = ctrl_pool[len(ctrl_pool)//2:]
        x_tr = case_pool[:len(case_pool)//2]; x_te = case_pool[len(case_pool)//2:]
        
        weights = {}
        for cpg in XU_PRIORITY:
            if cpg not in cpg_data: continue
            vals = cpg_data[cpg]
            c_v = [vals[i] for i in c_tr if vals[i] is not None]
            x_v = [vals[i] for i in x_tr if vals[i] is not None]
            if len(c_v)<5 or len(x_v)<5: continue
            d, _, _ = cohens_d(c_v, x_v)
            weights[cpg] = {'d': d, 'ctrl_mean': statistics.mean(c_v)}
        
        def sc(i):
            s = 0
            for cpg, w in weights.items():
                v = cpg_data[cpg][i]
                if v is not None:
                    s += math.copysign(w['d'], w['d']) * (v - w['ctrl_mean'])
            return s
        cs = [sc(i) for i in c_te]
        xs = [sc(i) for i in x_te]
        d_t, _, _ = cohens_d(cs, xs)
        d_cv.append(d_t)
    return d_cv

print(f"\n10-fold CV Xu-CpG directional score — BREAST:")
breast_cv = run_xu_cv(breast, 'breast')
print(f"  Per-iter: {[f'{d:+.3f}' for d in breast_cv]}")
print(f"  Mean test d = {statistics.mean(breast_cv):+.3f} ± {statistics.stdev(breast_cv):.3f}")
print(f"  (GSE51057 comparison: mean test d = +0.605 ± 0.190)")

# ═══════════════════════════════════════════════════════════════════════════
# B. SECRETORY VARIANCE + TIME-TO-DIAGNOSIS REPLICATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*78)
print("B. SECRETORY VARIANCE BY TIME-TO-DIAGNOSIS (BREAST)")
print("="*78)

ctrl_sec_sd = [SAMPLES[i]['sec_A_sd'] for i in controls if 'sec_A_sd' in SAMPLES[i]]
print(f"Controls mean secretory A_SD: {statistics.mean(ctrl_sec_sd):.4f}  (n={len(ctrl_sec_sd)})")

windows = [(0,1,'≤1 yr pre-dx'),(1,2,'1-2 yr'),(2,5,'2-5 yr'),
           (5,10,'5-10 yr'),(10,20,'>10 yr pre-dx')]
print(f"\n{'Window':<20} {'n':>4} {'mean A_SD':>10} {'d vs ctrl':>11} {'p':>10}")
for lo, hi, lbl in windows:
    sub = [SAMPLES[i] for i in breast 
           if 'years_to_dx' in SAMPLES[i] and lo <= SAMPLES[i]['years_to_dx'] < hi 
           and 'sec_A_sd' in SAMPLES[i]]
    if len(sub) < 5:
        print(f"{lbl:<20} {len(sub):>4} (insufficient)")
        continue
    sd_vals = [s['sec_A_sd'] for s in sub]
    d, delta, p = cohens_d(ctrl_sec_sd, sd_vals)
    print(f"{lbl:<20} {len(sub):>4} {statistics.mean(sd_vals):>10.4f} {d:>+11.3f} {p:>10.2e}")

print(f"\n(GSE51057 >10 yr window gave d = -1.226, p = 1.5e-3)")

# ═══════════════════════════════════════════════════════════════════════════
# C. COLORECTAL — does the framework work for non-breast cancer?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*78)
print("C. COLORECTAL CANCER (n=166): can the framework detect another cancer type?")
print("="*78)

# Xu score on colorectal (using same Xu CpGs — will they work for colon?)
print(f"\n10-fold CV Xu-CpG score — COLORECTAL:")
colon_cv = run_xu_cv(colon, 'colon')
print(f"  Mean test d = {statistics.mean(colon_cv):+.3f} ± {statistics.stdev(colon_cv):.3f}")
print(f"  (Expectation: lower than breast since Xu CpGs are breast-specific)")

# Top data-driven CpGs for colon within GSE51032
print("\nLet's find which CpGs from our 69-panel separate colon cases best (within-data):")
colon_d = {}
for cpg in cpg_data:
    vals = cpg_data[cpg]
    c = [vals[i] for i in controls if vals[i] is not None]
    x = [vals[i] for i in colon if vals[i] is not None]
    if len(c)<20 and len(x)<20: continue
    d, _, _ = cohens_d(c, x)
    colon_d[cpg] = d
ranked = sorted(colon_d.items(), key=lambda x: abs(x[1]), reverse=True)
print("Top 10 by |Cohen's d| for colorectal (descriptive, not CV):")
for cpg, d in ranked[:10]:
    print(f"  {cpg}  d = {d:+.3f}")

# Proper data-driven CV for colon
print("\nCV with TOP-10 CpGs selected on train-half — COLORECTAL:")
def run_topn_cv(case_idx, n_iter=10, topn=10):
    d_cv = []
    ctrl_pool = controls.copy()
    case_pool = case_idx.copy()
    for _ in range(n_iter):
        random.shuffle(ctrl_pool); random.shuffle(case_pool)
        c_tr = ctrl_pool[:len(ctrl_pool)//2]; c_te = ctrl_pool[len(ctrl_pool)//2:]
        x_tr = case_pool[:len(case_pool)//2]; x_te = case_pool[len(case_pool)//2:]
        
        train_d = {}
        for cpg in cpg_data:
            vals = cpg_data[cpg]
            c_v = [vals[i] for i in c_tr if vals[i] is not None]
            x_v = [vals[i] for i in x_tr if vals[i] is not None]
            if len(c_v)<10 or len(x_v)<10: continue
            d, _, _ = cohens_d(c_v, x_v)
            train_d[cpg] = d
        top = sorted(train_d.items(), key=lambda x: abs(x[1]), reverse=True)[:topn]
        
        def sc(i):
            return sum(math.copysign(1, train_d[c]) * (cpg_data[c][i] if cpg_data[c][i] is not None else 0)
                      for c, _ in top)
        cs = [sc(i) for i in c_te]
        xs = [sc(i) for i in x_te]
        d, _, _ = cohens_d(cs, xs)
        d_cv.append(d)
    return d_cv

colon_top_cv = run_topn_cv(colon)
print(f"  Per-iter: {[f'{d:+.3f}' for d in colon_top_cv]}")
print(f"  Mean test d = {statistics.mean(colon_top_cv):+.3f} ± {statistics.stdev(colon_top_cv):.3f}")

# Same for breast with top-N
print("\nCV with TOP-10 CpGs selected on train-half — BREAST:")
breast_top_cv = run_topn_cv(breast)
print(f"  Per-iter: {[f'{d:+.3f}' for d in breast_top_cv]}")
print(f"  Mean test d = {statistics.mean(breast_top_cv):+.3f} ± {statistics.stdev(breast_top_cv):.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# D. DETECTION PERFORMANCE AT REALISTIC THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*78)
print("D. DETECTION PERFORMANCE AT CLINICAL SPECIFICITY TARGETS")
print("="*78)

def perf_from_d(d, target_specs=[0.95, 0.90, 0.80]):
    """At Cohen's d, compute sens at target specificities under normal assumption."""
    out = []
    for spec in target_specs:
        z_thr = {0.95: 1.645, 0.90: 1.282, 0.80: 0.842}[spec]
        sens = 0.5 * math.erfc((z_thr - d) / math.sqrt(2))
        out.append((spec, sens))
    return out

# Combined: use Xu-score on breast + top-N on colon (honest CV)
results = {
    'Breast (Xu-CpG CV d)': statistics.mean(breast_cv),
    'Breast (Top-10 CV d)': statistics.mean(breast_top_cv),
    'Colorectal (Xu-CpG CV d)': statistics.mean(colon_cv),
    'Colorectal (Top-10 CV d)': statistics.mean(colon_top_cv),
}
print(f"{'Method':<28} {'CV d':>8} {'S@95%spec':>10} {'S@90%spec':>10} {'S@80%spec':>10}")
for method, d in results.items():
    perfs = perf_from_d(d)
    print(f"{method:<28} {d:>+8.3f} "
          f"{perfs[0][1]*100:>9.1f}% {perfs[1][1]*100:>9.1f}% {perfs[2][1]*100:>9.1f}%")

# Save summary
out = {
    'dataset': 'GSE51032 EPIC-HuGeF',
    'n_controls': len(controls),
    'n_breast': len(breast),
    'n_colorectal': len(colon),
    'breast_xu_cv_d_mean': statistics.mean(breast_cv),
    'breast_xu_cv_d_std': statistics.stdev(breast_cv),
    'breast_top10_cv_d_mean': statistics.mean(breast_top_cv),
    'breast_top10_cv_d_std': statistics.stdev(breast_top_cv),
    'colon_xu_cv_d_mean': statistics.mean(colon_cv),
    'colon_xu_cv_d_std': statistics.stdev(colon_cv),
    'colon_top10_cv_d_mean': statistics.mean(colon_top_cv),
    'colon_top10_cv_d_std': statistics.stdev(colon_top_cv),
    'top10_colon_cpgs': ranked[:10],
}
json.dump(out, open('/home/claude/geo_analysis/gse51032/VAL_047_replication_results.json','w'), indent=1, default=str)
print(f"\nSaved to VAL_047_replication_results.json")
