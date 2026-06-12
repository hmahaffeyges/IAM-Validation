#!/usr/bin/env python3
"""
VAL-052 — AddNeuroMed cross-platform AD replication + age regression.

Cross-platform test of VAL-051 Rule A panel (7 AD-directional CpGs).
All 7 CpGs present in AddNeuroMed 450K. Frozen directions + standardization
from AIBL training.
"""
import json, math, statistics, random

SEED = 42
N_BOOT = 10_000

# Load
manifest = json.load(open('addneuromed_manifest.json'))
betas = json.load(open('addneuromed_imm_betas.json'))
panel = json.load(open('/home/claude/ad_audit/val051/val051_panel_ruleA.json'))['cpgs']

def H(b):
    if b <= 0 or b >= 1: return 0.0
    return -b*math.log2(b) - (1-b)*math.log2(1-b)

def mwu_one(a, b):
    n1, n2 = len(a), len(b)
    combined = [(v, 0) for v in a] + [(v, 1) for v in b]
    combined.sort(key=lambda x: x[0])
    ra = 0.0; i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]: j += 1
        avg = (i + j + 1) / 2.0
        for k in range(i, j):
            if combined[k][1] == 0: ra += avg
        i = j
    U = ra - n1*(n1+1)/2.0
    mu = n1*n2/2.0
    sig = math.sqrt(n1*n2*(n1+n2+1)/12.0)
    z = (U - mu) / sig
    return U, z, 0.5 * math.erfc(z / math.sqrt(2))

def cohens_d(a, b):
    if min(len(a), len(b)) < 2: return 0.0
    m1, m2 = statistics.mean(a), statistics.mean(b)
    s1, s2 = statistics.pstdev(a), statistics.pstdev(b)
    pooled = math.sqrt(((len(a)-1)*s1*s1 + (len(b)-1)*s2*s2) / (len(a)+len(b)-2))
    return 0.0 if pooled == 0 else (m1 - m2) / pooled

def bootstrap_d(a, b, n_boot, seed):
    rng = random.Random(seed)
    a, b = list(a), list(b)
    out = []
    for _ in range(n_boot):
        ra = [a[rng.randint(0, len(a)-1)] for _ in range(len(a))]
        rb = [b[rng.randint(0, len(b)-1)] for _ in range(len(b))]
        out.append(cohens_d(ra, rb))
    out.sort()
    return out[int(0.025*n_boot)], out[int(0.975*n_boot)]

def auc(a, b):
    if not a or not b: return 0.5
    U, _, _ = mwu_one(a, b)
    return U / (len(a)*len(b))

def simple_linreg(x, y):
    n = len(x)
    mx, my = statistics.mean(x), statistics.mean(y)
    sxx = sum((xi - mx)**2 for xi in x)
    sxy = sum((xi - mx)*(yi - my) for xi, yi in zip(x, y))
    b = sxy / sxx if sxx else 0
    a = my - b * mx
    ss_tot = sum((yi - my)**2 for yi in y)
    ss_res = sum((yi - (a + b*xi))**2 for xi, yi in zip(x, y))
    r2 = 1 - ss_res/ss_tot if ss_tot else 0
    residuals = [yi - (a + b*xi) for xi, yi in zip(x, y)]
    return a, b, r2, residuals

# ─── AIBL-frozen scoring
def a_dir_frozen(sample_betas):
    """A_dir using AIBL-trained directions + standardization."""
    contribs = []
    for r in panel:
        b = sample_betas.get(r['cpg'])
        if b is None or not (0 < b < 1): continue
        if r['sd_hc_train'] == 0: continue
        z = (b - r['mean_hc_train']) / r['sd_hc_train']
        contribs.append(r['direction'] * z)
    if len(contribs) < 5: return None
    return sum(contribs) / len(contribs)

# Build records
records = []
for s in manifest:
    sb = betas.get(s['gsm'], {})
    score = a_dir_frozen(sb)
    if score is None: continue
    age_str = s.get('age','')
    age = int(age_str) if age_str.isdigit() else None
    records.append({
        'gsm': s['gsm'],
        'status': s.get('disease state',''),
        'sex': s.get('Sex',''),
        'age': age,
        'A_dir': score,
    })

# Normalize disease labels
def norm_status(x):
    x = x.lower()
    if 'alzheimer' in x: return 'AD'
    if 'mild cognitive' in x or 'mci' in x: return 'MCI'
    if 'control' in x or 'healthy' in x: return 'HC'
    return x

for r in records: r['grp'] = norm_status(r['status'])

AD = [r for r in records if r['grp'] == 'AD']
HC = [r for r in records if r['grp'] == 'HC']
MCI = [r for r in records if r['grp'] == 'MCI']

print("="*72)
print("VAL-052 — AddNeuroMed Cross-Platform AD Replication")
print("="*72)
print(f"Samples scored: {len(records)}")
print(f"AD={len(AD)}, MCI={len(MCI)}, HC={len(HC)}")
print(f"Panel: VAL-051 Rule A frozen ({len(panel)} CpGs, directions from AIBL training)")

# H1 PRIMARY
print("\n" + "="*72)
print("H1 PRIMARY — A_dir(AD) > A_dir(HC), AIBL-frozen panel + standardization")
print("="*72)
ad_sc = [r['A_dir'] for r in AD]
hc_sc = [r['A_dir'] for r in HC]
mci_sc = [r['A_dir'] for r in MCI]
print(f"  Mean A_dir(AD) = {statistics.mean(ad_sc):+.4f}")
print(f"  Mean A_dir(HC) = {statistics.mean(hc_sc):+.4f}")
print(f"  Mean A_dir(MCI)= {statistics.mean(mci_sc):+.4f}")
d = cohens_d(ad_sc, hc_sc)
lo, hi = bootstrap_d(ad_sc, hc_sc, N_BOOT, SEED)
_, _, p = mwu_one(ad_sc, hc_sc)
a = auc(ad_sc, hc_sc)
print(f"  Δ = {statistics.mean(ad_sc) - statistics.mean(hc_sc):+.4f}")
print(f"  Cohen's d     = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"  MWU p_onesided= {p:.4g}")
print(f"  AUC           = {a:.4f}")

# Sensitivity — AddNeuroMed-own-HC standardization
print(f"\n--- Sensitivity: AddNeuroMed-HC-standardized scoring ---")
# Build own-HC stats per CpG
own_stats = {}
for r in panel:
    hc_vals = []
    for s in manifest:
        sb = betas.get(s['gsm'], {})
        v = sb.get(r['cpg'])
        if v is not None and 0 < v < 1 and norm_status(s.get('disease state','')) == 'HC':
            hc_vals.append(v)
    if hc_vals:
        own_stats[r['cpg']] = {
            'mean': statistics.mean(hc_vals),
            'sd': statistics.pstdev(hc_vals),
            'dir': r['direction'],
        }

def a_dir_own(sample_betas):
    contribs = []
    for cpg, s in own_stats.items():
        b = sample_betas.get(cpg)
        if b is None or not (0 < b < 1): continue
        if s['sd'] == 0: continue
        z = (b - s['mean']) / s['sd']
        contribs.append(s['dir'] * z)
    if len(contribs) < 5: return None
    return sum(contribs) / len(contribs)

for r in records:
    r['A_dir_own'] = a_dir_own(betas[r['gsm']])

ad_o = [r['A_dir_own'] for r in AD if r['A_dir_own'] is not None]
hc_o = [r['A_dir_own'] for r in HC if r['A_dir_own'] is not None]
d_o = cohens_d(ad_o, hc_o)
_, _, p_o = mwu_one(ad_o, hc_o)
print(f"  Cohen's d (own-HC stz) = {d_o:+.4f}, p = {p_o:.4g}")

# H2 — MCI intermediate
print("\n" + "="*72)
print("H2 — MCI intermediate between HC and AD")
print("="*72)
print(f"  HC  mean = {statistics.mean(hc_sc):+.4f}  (n={len(hc_sc)})")
print(f"  MCI mean = {statistics.mean(mci_sc):+.4f}  (n={len(mci_sc)})")
print(f"  AD  mean = {statistics.mean(ad_sc):+.4f}  (n={len(ad_sc)})")
if statistics.mean(hc_sc) < statistics.mean(mci_sc) < statistics.mean(ad_sc):
    print(f"  ✓ Monotonic HC < MCI < AD")
else:
    print(f"  Not monotonic")

# H3 sex-stratified
print("\n" + "="*72)
print("H3 — Sex-stratified replication")
print("="*72)
for sex in ['Male','Female']:
    ads = [r['A_dir'] for r in AD if r['sex'] == sex]
    hcs = [r['A_dir'] for r in HC if r['sex'] == sex]
    if len(ads) < 3 or len(hcs) < 3: continue
    ds = cohens_d(ads, hcs)
    _, _, ps = mwu_one(ads, hcs)
    print(f"  {sex}: n_AD={len(ads)}, n_HC={len(hcs)}, d={ds:+.4f}, p={ps:.4g}")

# H5 — age-by-group
print("\n" + "="*72)
print("H5 — Age distribution by group")
print("="*72)
age_records = [r for r in records if r['age'] is not None]
for grp_name, grp_records in [('HC', [r for r in age_records if r['grp']=='HC']),
                               ('MCI', [r for r in age_records if r['grp']=='MCI']),
                               ('AD', [r for r in age_records if r['grp']=='AD'])]:
    ages = [r['age'] for r in grp_records]
    if not ages: continue
    print(f"  {grp_name}: n={len(ages)}, mean age = {statistics.mean(ages):.1f} yr  (range {min(ages)}-{max(ages)})")
ad_ages = [r['age'] for r in AD if r['age'] is not None]
hc_ages = [r['age'] for r in HC if r['age'] is not None]
d_age = cohens_d(ad_ages, hc_ages)
print(f"  Cohen's d on chronological age (AD vs HC): {d_age:+.4f}")

# H4 PRIMARY — AGE REGRESSION
print("\n" + "="*72)
print("H4 PRIMARY — Regress chronological age out of A_dir, re-test AD vs HC")
print("="*72)
all_with_age = [r for r in records if r['age'] is not None]
xs = [r['age'] for r in all_with_age]
ys = [r['A_dir'] for r in all_with_age]
a_int, b_slope, r2, residuals = simple_linreg(xs, ys)
print(f"  Regression A_dir ~ age (n={len(xs)}):")
print(f"    Intercept = {a_int:+.4f}")
print(f"    Slope     = {b_slope:+.6f} per year")
print(f"    R²        = {r2:.4f}  ({r2*100:.1f}% of A_dir variance explained by age)")

# Residuals by group
resid_AD = [residuals[i] for i, r in enumerate(all_with_age) if r['grp']=='AD']
resid_HC = [residuals[i] for i, r in enumerate(all_with_age) if r['grp']=='HC']
d_res = cohens_d(resid_AD, resid_HC)
lo_r, hi_r = bootstrap_d(resid_AD, resid_HC, N_BOOT, SEED)
_, _, p_res = mwu_one(resid_AD, resid_HC)
print(f"\n  After regression:")
print(f"    Residual d (AD vs HC) = {d_res:+.4f}  95% CI [{lo_r:+.4f}, {hi_r:+.4f}]")
print(f"    Residual p_onesided   = {p_res:.4g}")
print(f"    Raw d (no regression) = {d:+.4f}, p = {p:.4g}")
print(f"    Δ d (residual - raw)  = {d_res - d:+.4f}")

# Decision
print("\n" + "="*72)
print("PRE-LOCKED DECISION — Cross-platform + age-confounding")
print("="*72)

if d > 0.3 and p < 0.10:
    outcome_cp = "OUTCOME 1 — FULL CROSS-PLATFORM REPLICATION"
elif d > 0.1:
    outcome_cp = "OUTCOME 2 — Partial replication, direction preserved"
elif d < -0.1:
    outcome_cp = "OUTCOME 4 — Direction flip"
else:
    outcome_cp = "OUTCOME 3 — Cross-platform transfer null"
print(f"  Cross-platform: {outcome_cp}")
print(f"    d_raw = {d:+.4f}, p = {p:.4g}, AUC = {a:.4f}")

if r2 < 0.30 and d_res > 0.3:
    outcome_age = "Age is minor confound; AD signal age-independent"
elif r2 < 0.60 and d_res > 0.1:
    outcome_age = "Age is partial confound; AD signal survives regression"
elif d_res > 0:
    outcome_age = "Age dominant; AD signal collapses under regression"
else:
    outcome_age = "Residual anti-direction — check numerical stability"
print(f"\n  Age confounding: {outcome_age}")
print(f"    R² = {r2:.3f}, residual d = {d_res:+.4f}")

# Save
out = {
    'val_id': 'VAL-052',
    'cohort': 'AddNeuroMed GSE144858',
    'platform': '450K',
    'n_samples_scored': len(records),
    'group_counts': {'AD': len(AD), 'MCI': len(MCI), 'HC': len(HC)},
    'panel_transfer': '7/7 Rule A CpGs present on 450K',
    'panel_coverage_per_sample_avg': sum(1 for r in records if r['A_dir'] is not None) / max(1, len(records)),
    'H1_primary': {
        'mean_A_dir_AD': statistics.mean(ad_sc), 'mean_A_dir_HC': statistics.mean(hc_sc),
        'delta': statistics.mean(ad_sc) - statistics.mean(hc_sc),
        'cohens_d': d, 'ci_95': [lo, hi], 'p_onesided': p, 'auc': a,
    },
    'sensitivity_own_HC_standardized': {'cohens_d': d_o, 'p': p_o},
    'H2_MCI_mean': statistics.mean(mci_sc),
    'H3_sex': {
        sex: {
            'n_AD': sum(1 for r in AD if r['sex']==sex),
            'n_HC': sum(1 for r in HC if r['sex']==sex),
            'cohens_d': cohens_d([r['A_dir'] for r in AD if r['sex']==sex],
                                  [r['A_dir'] for r in HC if r['sex']==sex]),
        } for sex in ['Male','Female']
    },
    'H4_age_regression': {
        'R_squared': r2,
        'slope_per_year': b_slope,
        'residual_d_AD_vs_HC': d_res,
        'residual_p': p_res,
    },
    'H5_age_by_group_d': d_age,
    'cross_platform_decision': outcome_cp,
    'age_decision': outcome_age,
    'per_sample_records': records,
}
with open('VAL_052_RESULTS.json','w') as f:
    json.dump(out, f, indent=2)
print(f"\nResults written to VAL_052_RESULTS.json")
