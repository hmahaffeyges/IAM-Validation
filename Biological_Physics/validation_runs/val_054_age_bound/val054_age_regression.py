#!/usr/bin/env python3
"""
VAL-054 — Cellular age regression on VAL-051 holdout.

Computes per-sample cellular age from the 80-cell immune-class baseline
(§E.5 Alpha-Omega, HEALTHY_BASELINE_immune from GAPE_WEB_v13.py).
Regresses it out of A_dir. Tests whether AD signal survives.
"""
import json, math, statistics, random

PANEL_18 = [
    'cg00431549','cg01127300','cg02228185','cg02489552','cg04023335',
    'cg09809672','cg10632894','cg12554573','cg14614643','cg16867657',
    'cg17861230','cg18834029','cg22454769','cg22736354','cg23244761',
    'cg25432518','cg25809905','cg26614073',
]
SEED = 42
N_BOOT = 10_000

# IMMUNE class 80-cell baseline from GAPE_WEB_v13.py + Alpha-Omega §E.5
# Format: age_decade → (β_mean, β_sd, n, source)
HEALTHY_BASELINE_IMMUNE = {
    '0-9':    (0.780, 0.015, 45,   'Alisch 2012 pediatric'),
    '10-19':  (0.773, 0.016, 58,   'Alisch 2012 + Hannum'),
    '20-29':  (0.768, 0.017, 95,   'Hannum 2013'),
    '30-39':  (0.764, 0.018, 102,  'Hannum 2013'),
    '40-49':  (0.760, 0.018, 115,  'Hannum 2013'),
    '50-59':  (0.756, 0.019, 108,  'Hannum 2013'),
    '60-69':  (0.751, 0.020, 98,   'Hannum + Horvath'),
    '70-79':  (0.745, 0.021, 85,   'Hannum 2013'),
    '80-89':  (0.739, 0.022, 42,   'Hannum 2013'),
    '90+':    (0.732, 0.024, 15,   'Hannum 2013 oldest'),
}
# Decade midpoints for interpolation
DECADE_MID = {
    '0-9': 5, '10-19': 15, '20-29': 25, '30-39': 35,
    '40-49': 45, '50-59': 55, '60-69': 65, '70-79': 75,
    '80-89': 85, '90+': 95,
}

def cellular_age_from_beta(beta_mean):
    """Invert the baseline curve: find age s.t. baseline β_mean(age) = patient β_mean."""
    # Baseline β decreases with age (immune class hypomethylation with aging)
    # So beta high → young, beta low → old
    decades_sorted = sorted(HEALTHY_BASELINE_IMMUNE.keys(), key=lambda d: DECADE_MID[d])
    pairs = [(DECADE_MID[d], HEALTHY_BASELINE_IMMUNE[d][0]) for d in decades_sorted]
    # Pairs are sorted by age ascending, β descending
    # Clamp
    if beta_mean >= pairs[0][1]: return pairs[0][0]  # younger than youngest
    if beta_mean <= pairs[-1][1]: return pairs[-1][0]  # older than oldest
    # Linear interp in (β, age) space
    for i in range(len(pairs)-1):
        age_lo, b_hi = pairs[i]
        age_hi, b_lo = pairs[i+1]
        if b_lo <= beta_mean <= b_hi:
            frac = (b_hi - beta_mean) / (b_hi - b_lo)
            return age_lo + frac * (age_hi - age_lo)
    return None

# Helpers
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

def simple_linreg(x, y):
    """y = a + b*x. Returns (a, b, r_squared, residuals)."""
    n = len(x)
    mx, my = statistics.mean(x), statistics.mean(y)
    sxx = sum((xi - mx)**2 for xi in x)
    sxy = sum((xi - mx)*(yi - my) for xi, yi in zip(x, y))
    b_slope = sxy / sxx if sxx else 0
    a_int = my - b_slope * mx
    # R²
    ss_tot = sum((yi - my)**2 for yi in y)
    ss_res = sum((yi - (a_int + b_slope*xi))**2 for xi, yi in zip(x, y))
    r2 = 1 - ss_res/ss_tot if ss_tot else 0
    residuals = [yi - (a_int + b_slope*xi) for xi, yi in zip(x, y)]
    return a_int, b_slope, r2, residuals

# Load
manifest = json.load(open('aibl_manifest.json'))
betas = json.load(open('aibl_imm_betas.json'))
split_map = json.load(open('val051_split_map.json'))['split']
panel = json.load(open('val051_panel_ruleA.json'))['cpgs']
train_stats = {r['cpg']: r for r in panel}

def a_dir(sentrix_betas):
    contribs = []
    for r in panel:
        b = sentrix_betas.get(r['cpg'])
        if b is None or not (0 < b < 1): continue
        if r['sd_hc_train'] == 0: continue
        z = (b - r['mean_hc_train']) / r['sd_hc_train']
        contribs.append(r['direction'] * z)
    if len(contribs) < 5: return None
    return sum(contribs) / len(contribs)

def mean_beta_fullpanel(sentrix_betas):
    vals = [sentrix_betas[c] for c in PANEL_18 if sentrix_betas.get(c) is not None and 0 < sentrix_betas.get(c, 0) < 1]
    if len(vals) < 12: return None
    return sum(vals) / len(vals)

def mean_beta_rulea(sentrix_betas):
    vals = [sentrix_betas[r['cpg']] for r in panel
            if sentrix_betas.get(r['cpg']) is not None and 0 < sentrix_betas.get(r['cpg'], 0) < 1]
    if len(vals) < 5: return None
    return sum(vals) / len(vals)

# Compute on holdout
print("="*72)
print("VAL-054 — Cellular age regression on VAL-051 holdout")
print("="*72)

records = []
for s in manifest:
    if split_map.get(s['gsm']) != 'holdout': continue
    sb = betas[s['sentrix']]
    ad = a_dir(sb)
    b_full = mean_beta_fullpanel(sb)
    b_rulea = mean_beta_rulea(sb)
    if ad is None or b_full is None or b_rulea is None: continue
    records.append({
        'gsm': s['gsm'], 'status': s['disease status'], 'sex': s['gender'],
        'A_dir': ad,
        'beta_full': b_full,
        'beta_rulea': b_rulea,
        'cell_age_fullpanel': cellular_age_from_beta(b_full),
        'cell_age_rulea': cellular_age_from_beta(b_rulea),
    })

print(f"\nHoldout samples scored: {len(records)}")

AD = [r for r in records if r['status'] == "Alzheimer's disease"]
HC = [r for r in records if r['status'] == 'healthy control']
MCI = [r for r in records if r['status'] == 'Mild Cognitive Impairment']

print(f"  AD={len(AD)}, MCI={len(MCI)}, HC={len(HC)}")

# H3 exploratory — cell age by group
print("\n" + "="*72)
print("H3 EXPLORATORY — Cellular age by disease group (full 18-CpG panel)")
print("="*72)
for grp_name, grp in [('HC', HC), ('MCI', MCI), ('AD', AD)]:
    ages = [r['cell_age_fullpanel'] for r in grp if r['cell_age_fullpanel'] is not None]
    if not ages: continue
    print(f"  {grp_name}: n={len(ages)}, mean cell age = {statistics.mean(ages):.1f} yr, median = {statistics.median(ages):.1f}, range [{min(ages):.1f}, {max(ages):.1f}]")
d_age_adhc = cohens_d([r['cell_age_fullpanel'] for r in AD], [r['cell_age_fullpanel'] for r in HC])
print(f"  Cohen's d on cellular age (AD vs HC): {d_age_adhc:+.4f}")

# Primary — regress age out and re-test
print("\n" + "="*72)
print("H1 PRIMARY — Regress cellular_age_fullpanel out of A_dir, re-test AD vs HC")
print("="*72)

# Model 2: A_dir ~ cell_age_rulea (panel-specific age, confounded)
# Model 3: A_dir ~ cell_age_fullpanel (more independent age estimator)
for age_field, label in [('cell_age_rulea', 'panel-specific cellular age'),
                          ('cell_age_fullpanel', 'full 18-CpG cellular age')]:
    print(f"\n--- Regression: A_dir ~ {label} ---")
    xs = [r[age_field] for r in records]
    ys = [r['A_dir'] for r in records]
    a_int, b_slope, r2, residuals = simple_linreg(xs, ys)
    print(f"  Intercept = {a_int:+.4f}, Slope = {b_slope:+.6f} per year")
    print(f"  R² = {r2:.4f}  ({r2*100:.1f}% of A_dir variance explained by cell age)")

    # Split residuals by group
    resid_AD = [residuals[i] for i, r in enumerate(records) if r['status'] == "Alzheimer's disease"]
    resid_HC = [residuals[i] for i, r in enumerate(records) if r['status'] == 'healthy control']

    d_res = cohens_d(resid_AD, resid_HC)
    lo, hi = bootstrap_d(resid_AD, resid_HC, N_BOOT, SEED)
    _, _, p_res = mwu_one(resid_AD, resid_HC)
    print(f"  Residual d (AD vs HC) = {d_res:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  Residual p_onesided   = {p_res:.4g}")

    # Compare to raw
    raw_AD = [r['A_dir'] for r in AD]
    raw_HC = [r['A_dir'] for r in HC]
    d_raw = cohens_d(raw_AD, raw_HC)
    _, _, p_raw = mwu_one(raw_AD, raw_HC)
    print(f"  Raw d        = {d_raw:+.4f}, p = {p_raw:.4g}  (VAL-051 holdout)")
    print(f"  Δ d (residual - raw) = {d_res - d_raw:+.4f}")

# Decision matrix using full cellular age
xs = [r['cell_age_fullpanel'] for r in records]
ys = [r['A_dir'] for r in records]
_, _, r2, residuals = simple_linreg(xs, ys)
resid_AD = [residuals[i] for i, r in enumerate(records) if r['status'] == "Alzheimer's disease"]
resid_HC = [residuals[i] for i, r in enumerate(records) if r['status'] == 'healthy control']
d_final = cohens_d(resid_AD, resid_HC)

print("\n" + "="*72)
print("PRE-LOCKED DECISION (§6)")
print("="*72)
# METHODOLOGICAL CHECK: the 80-cell baseline was compiled from class-wide immune β_mean
# (typical range 0.73-0.78). The IMM_CPGS panel has β_mean around 0.52 because Xu-2020
# selected those CpGs for differential-methylation signal, not class-average behavior.
# Consequence: patient β_mean on this panel is ALWAYS below the baseline's oldest decade
# (0.732), so the inversion saturates at age 95.0 for all samples.
# All-sample saturation → zero variance in cellular-age predictor → zero explanatory power
# → regression gives the identity. The numbers above are a non-test, not a real test.

all_same_age = all(r['cell_age_fullpanel'] == 95.0 for r in records)
if all_same_age:
    outcome = "NON-TEST — panel β_mean saturates baseline interpolator at oldest decade"
    print()
    print("  ⚠ METHODOLOGICAL OBSERVATION:")
    print("  All samples clamped to cellular age 95.0 because panel β_mean is systematically")
    print("  lower than the 80-cell class-wide baseline. The IMM_CPGS panel was selected for")
    print("  differential-methylation signal, not class-average β. The 80-cell baseline was")
    print("  compiled from pan-immune β, not this panel subset.")
    print("  The correct conclusion: age-confounding in VAL-051 CANNOT be tested on GEO-only")
    print("  AIBL release. Chronological age metadata requires direct AIBL data access.")
    print("  This is a PRE-SPECIFIED limitation inherited from VAL-050 (documented §6.1).")
elif d_final > 0.4 and r2 < 0.30:
    outcome = "OUTCOME 1 — AD-specific signal dominant, age minor confound"
elif d_final > 0.2:
    outcome = "OUTCOME 2 — Age partial confound, AD signal survives"
elif d_final > 0.0:
    outcome = "OUTCOME 3 — Age dominant driver, AD signal largely collapses"
else:
    outcome = "OUTCOME 4 — Over-regressed / inverted"
print(f"\n  Residual d = {d_final:+.4f}")
print(f"  R² (age explains A_dir) = {r2:.3f}")
print(f"  Decision: {outcome}")

# Save
out = {
    'val_id': 'VAL-054',
    'parent': 'VAL-051',
    'n_holdout_scored': len(records),
    'group_counts': {'AD': len(AD), 'MCI': len(MCI), 'HC': len(HC)},
    'cellular_age_summary': {
        grp: {
            'n': len([r for r in g if r['cell_age_fullpanel'] is not None]),
            'mean': statistics.mean([r['cell_age_fullpanel'] for r in g]) if g else None,
            'median': statistics.median([r['cell_age_fullpanel'] for r in g]) if g else None,
        } for grp, g in [('HC', HC), ('MCI', MCI), ('AD', AD)]
    },
    'cohens_d_cellular_age_AD_vs_HC': d_age_adhc,
    'regression_full_cellage': {
        'R_squared': r2,
        'slope_per_year': b_slope,
        'residual_d_AD_vs_HC': d_final,
    },
    'decision': outcome,
    'per_sample': records,
}
with open('VAL_054_RESULTS.json','w') as f:
    json.dump(out, f, indent=2)
print(f"\nResults written to VAL_054_RESULTS.json")
