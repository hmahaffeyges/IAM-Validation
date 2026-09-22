#!/usr/bin/env python3
"""
VAL-053 — Sex-specific panel selection + holdout analysis.

Uses inherited VAL-051 split (seed=42 stratified disease × sex).
Selects Panel-F and Panel-M on their own-sex training.
Tests on own-sex and cross-sex holdout.
"""
import json, math, statistics, random

PANEL_18 = [
    'cg00431549','cg01127300','cg02228185','cg02489552','cg04023335',
    'cg09809672','cg10632894','cg12554573','cg14614643','cg16867657',
    'cg17861230','cg18834029','cg22454769','cg22736354','cg23244761',
    'cg25432518','cg25809905','cg26614073',
]
DELTA_BETA_THRESH = 0.015
Q_FDR_THRESH = 0.10
SEED = 42
N_BOOT = 10_000

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
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return U, z, p

def mwu_two(a, b):
    U, z, p = mwu_one(a, b)
    return U, z, math.erfc(abs(z) / math.sqrt(2))

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
    return U / (len(a) * len(b))

# Load
manifest = json.load(open('aibl_manifest.json'))
betas = json.load(open('aibl_imm_betas.json'))
split_map = json.load(open('val051_split_map.json'))['split']
val051_panel = {r['cpg']: r for r in json.load(open('val051_panel_ruleA.json'))['cpgs']}

def select_panel(sex_filter):
    """Run Rule A on a sex-filtered training subset."""
    ad_sen = []; hc_sen = []
    for s in manifest:
        if split_map.get(s['gsm']) != 'train': continue
        if s.get('gender') != sex_filter: continue
        if s['disease status'] == "Alzheimer's disease":
            ad_sen.append(s['sentrix'])
        elif s['disease status'] == 'healthy control':
            hc_sen.append(s['sentrix'])

    rows = []
    for cpg in PANEL_18:
        av, hv = [], []
        for s in ad_sen:
            b = betas[s].get(cpg)
            if b is not None and 0 < b < 1: av.append(b)
        for s in hc_sen:
            b = betas[s].get(cpg)
            if b is not None and 0 < b < 1: hv.append(b)
        if len(av) < 10 or len(hv) < 30: continue
        dbeta = statistics.mean(av) - statistics.mean(hv)
        _, _, p = mwu_two(av, hv)
        rows.append({
            'cpg': cpg, 'delta_beta': dbeta,
            'mean_hc_train': statistics.mean(hv),
            'sd_hc_train': statistics.pstdev(hv),
            'n_ad_train': len(av), 'n_hc_train': len(hv),
            'p_two': p,
            'direction': 1 if dbeta > 0 else -1,
        })
    rows.sort(key=lambda r: r['p_two'])
    m = len(rows)
    for rank, r in enumerate(rows, 1):
        r['q_FDR'] = r['p_two'] * m / rank
    selected = [r for r in rows if abs(r['delta_beta']) > DELTA_BETA_THRESH and r['q_FDR'] < Q_FDR_THRESH]
    return rows, selected, len(ad_sen), len(hc_sen)

# Select per-sex panels
print("="*72)
print("VAL-053 — Sex-specific AD panel selection")
print("="*72)

results = {}
for sex in ['Female', 'Male']:
    print(f"\n--- Selecting Panel-{sex[0]} on {sex} training ---")
    rows, selected, n_ad, n_hc = select_panel(sex)
    print(f"  Training n_AD={n_ad}, n_HC={n_hc}")
    print(f"  Per-CpG results (selected marked *):")
    for r in rows:
        sel = '*' if r in selected else ' '
        print(f"    {sel} {r['cpg']} Δβ={r['delta_beta']:+.4f} dir={r['direction']:+d} p={r['p_two']:.3g} q={r['q_FDR']:.3g}")
    print(f"  Selected: {len(selected)}/{len(rows)} CpGs")
    results[sex] = {'panel': selected, 'all_rows': rows, 'n_ad_train': n_ad, 'n_hc_train': n_hc}

# Score function
def score_sample(sentrix_betas, panel):
    """Directional z-composite using training HC stats."""
    contribs = []
    for r in panel:
        b = sentrix_betas.get(r['cpg'])
        if b is None or not (0 < b < 1): continue
        if r['sd_hc_train'] == 0: continue
        z = (b - r['mean_hc_train']) / r['sd_hc_train']
        contribs.append(r['direction'] * z)
    if len(contribs) < max(2, int(0.7 * len(panel))): return None
    return sum(contribs) / len(contribs)

# Analyze all 4 combinations: Panel-F on F-holdout, Panel-F on M-holdout, Panel-M on M-holdout, Panel-M on F-holdout
print("\n" + "="*72)
print("HOLDOUT ANALYSIS — 4 panel × sex combinations")
print("="*72)

analysis = {}
for panel_sex in ['Female', 'Male']:
    panel = results[panel_sex]['panel']
    if not panel:
        print(f"\nPanel-{panel_sex[0]}: NO CpGs selected — cannot score.")
        analysis[panel_sex] = None
        continue
    for eval_sex in ['Female', 'Male']:
        ad = []; hc = []
        for s in manifest:
            if split_map.get(s['gsm']) != 'holdout': continue
            if s.get('gender') != eval_sex: continue
            sc = score_sample(betas[s['sentrix']], panel)
            if sc is None: continue
            if s['disease status'] == "Alzheimer's disease": ad.append(sc)
            elif s['disease status'] == 'healthy control': hc.append(sc)
        if not ad or not hc:
            print(f"\nPanel-{panel_sex[0]} on {eval_sex} holdout: skipped (empty)")
            continue
        d = cohens_d(ad, hc)
        lo, hi = bootstrap_d(ad, hc, N_BOOT, SEED)
        _, _, p = mwu_one(ad, hc)
        a = auc(ad, hc)
        key = f"Panel-{panel_sex[0]}_on_{eval_sex}"
        print(f"\nPanel-{panel_sex[0]} on {eval_sex} holdout (n_AD={len(ad)}, n_HC={len(hc)}):")
        print(f"  panel size = {len(panel)} CpGs")
        print(f"  d = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  MWU p_onesided = {p:.4g}")
        print(f"  AUC = {a:.4f}")
        analysis[key] = {'n_AD': len(ad), 'n_HC': len(hc), 'd': d, 'ci': [lo, hi], 'p': p, 'auc': a, 'panel_size': len(panel)}

# Panel overlap (H5)
print("\n" + "="*72)
print("SECONDARY — H5: Jaccard overlap between Panel-F and Panel-M")
print("="*72)
cpgs_F = set(r['cpg'] for r in results['Female']['panel'])
cpgs_M = set(r['cpg'] for r in results['Male']['panel'])
jaccard = len(cpgs_F & cpgs_M) / len(cpgs_F | cpgs_M) if (cpgs_F | cpgs_M) else 0
print(f"  Panel-F: {sorted(cpgs_F)}")
print(f"  Panel-M: {sorted(cpgs_M)}")
print(f"  Shared: {sorted(cpgs_F & cpgs_M)}")
print(f"  F-only: {sorted(cpgs_F - cpgs_M)}")
print(f"  M-only: {sorted(cpgs_M - cpgs_F)}")
print(f"  Jaccard = {jaccard:.3f}")

# Direction agreement on shared CpGs
shared = cpgs_F & cpgs_M
dir_F = {r['cpg']: r['direction'] for r in results['Female']['panel']}
dir_M = {r['cpg']: r['direction'] for r in results['Male']['panel']}
dir_agree = sum(1 for c in shared if dir_F[c] == dir_M[c])
if shared:
    print(f"  Direction agreement on shared: {dir_agree}/{len(shared)}")

# Compare to VAL-051 unified
print("\n" + "="*72)
print("COMPARISON — Sex-specific vs VAL-051 unified panel")
print("="*72)
print(f"  VAL-051 unified Panel on Female holdout: d = +0.705, AUC 0.70, p = 0.003")
print(f"  VAL-051 unified Panel on Male holdout:   d = +0.512, AUC 0.66, p = 0.041")
if analysis.get('Panel-F_on_Female'):
    a = analysis['Panel-F_on_Female']
    delta_v51 = a['d'] - 0.705
    print(f"  Panel-F on Female holdout:    d = {a['d']:+.4f}  (Δ vs unified = {delta_v51:+.4f})")
if analysis.get('Panel-M_on_Male'):
    a = analysis['Panel-M_on_Male']
    delta_v51 = a['d'] - 0.512
    print(f"  Panel-M on Male holdout:      d = {a['d']:+.4f}  (Δ vs unified = {delta_v51:+.4f})")

# Save
out = {
    'val_id': 'VAL-053',
    'parent': 'VAL-051',
    'split': 'inherited from VAL-051',
    'panel_F': {
        'cpgs': list(cpgs_F),
        'directions': {c: dir_F[c] for c in cpgs_F},
        'n_selected': len(cpgs_F),
        'training_n_AD': results['Female']['n_ad_train'],
        'training_n_HC': results['Female']['n_hc_train'],
    },
    'panel_M': {
        'cpgs': list(cpgs_M),
        'directions': {c: dir_M[c] for c in cpgs_M},
        'n_selected': len(cpgs_M),
        'training_n_AD': results['Male']['n_ad_train'],
        'training_n_HC': results['Male']['n_hc_train'],
    },
    'holdout_analysis': analysis,
    'jaccard_overlap': jaccard,
    'shared_cpgs': sorted(shared),
    'direction_agreement': f"{dir_agree}/{len(shared)}" if shared else 'N/A',
    'val051_baseline': {
        'female_unified': {'d': 0.705, 'p': 0.003, 'auc': 0.70},
        'male_unified': {'d': 0.512, 'p': 0.041, 'auc': 0.66},
    },
}

with open('VAL_053_RESULTS.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nResults written to VAL_053_RESULTS.json")
