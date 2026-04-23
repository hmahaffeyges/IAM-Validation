#!/usr/bin/env python3
"""
VAL-051 Step 3 — Holdout + AddNeuroMed analysis.

PRECONDITIONS:
  - Split map sealed
  - Panel (Rule A + Rule B) sealed
  - Training stats sealed

This script:
  1. Scores A_dir on AIBL holdout (H1)
  2. Scores A_dir on AddNeuroMed if available (H2)
  3. Sex-stratified H3
  4. Bimodality test H4
  5. Comparison vs VAL-050 pooled entropy A-score (null-comparator)

Outputs:
  VAL_051_RESULTS.json
  VAL_051_REPORT.md
"""
import json, math, statistics, random, os

PANEL_18 = [
    'cg00431549','cg01127300','cg02228185','cg02489552','cg04023335',
    'cg09809672','cg10632894','cg12554573','cg14614643','cg16867657',
    'cg17861230','cg18834029','cg22454769','cg22736354','cg23244761',
    'cg25432518','cg25809905','cg26614073',
]
H_MIN_IMMUNE_METHYL = 0.838889
SEED = 42
N_BOOT = 10_000
N_PERM = 10_000

# ─── Load ─────────────────────────────────────────────────────────────
manifest = json.load(open('aibl_manifest.json'))
betas = json.load(open('aibl_imm_betas.json'))
split_map = json.load(open('val051_split_map.json'))['split']
panel_A = json.load(open('val051_panel_ruleA.json'))
panel_B = json.load(open('val051_panel_ruleB.json'))

# ─── Helpers ──────────────────────────────────────────────────────────
def H(b):
    if b <= 0 or b >= 1: return 0.0
    return -b*math.log2(b) - (1-b)*math.log2(1-b)

def mann_whitney_u_one(a, b):
    """One-sided: P(a > b)."""
    n1, n2 = len(a), len(b)
    combined = [(v, 0) for v in a] + [(v, 1) for v in b]
    combined.sort(key=lambda x: x[0])
    ranks_a = 0.0; i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]: j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            if combined[k][1] == 0: ranks_a += avg_rank
        i = j
    U_a = ranks_a - n1*(n1+1)/2.0
    mu_U = n1*n2/2.0
    sigma_U = math.sqrt(n1*n2*(n1+n2+1)/12.0)
    z = (U_a - mu_U) / sigma_U
    p_one = 0.5 * math.erfc(z / math.sqrt(2))
    return U_a, z, p_one

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    if min(n1,n2) < 2: return 0.0
    m1 = statistics.mean(a); m2 = statistics.mean(b)
    s1 = statistics.pstdev(a); s2 = statistics.pstdev(b)
    pooled = math.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / (n1+n2-2))
    if pooled == 0: return 0.0
    return (m1 - m2) / pooled

def bootstrap_d(a, b, n_boot, seed):
    rng = random.Random(seed)
    a_list = list(a); b_list = list(b)
    n1, n2 = len(a_list), len(b_list)
    out = []
    for _ in range(n_boot):
        ra = [a_list[rng.randint(0, n1-1)] for _ in range(n1)]
        rb = [b_list[rng.randint(0, n2-1)] for _ in range(n2)]
        out.append(cohens_d(ra, rb))
    out.sort()
    return out[int(0.025*n_boot)], out[int(0.5*n_boot)], out[int(0.975*n_boot)]

def levene_test(a, b):
    """Levene's test for equality of variance (median version)."""
    med_a = statistics.median(a); med_b = statistics.median(b)
    za = [abs(x - med_a) for x in a]
    zb = [abs(x - med_b) for x in b]
    n1, n2 = len(za), len(zb)
    m1, m2 = statistics.mean(za), statistics.mean(zb)
    s1, s2 = statistics.pvariance(za), statistics.pvariance(zb)
    pooled = ((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2)
    if pooled == 0: return 0.0, 1.0
    t = (m1 - m2) / math.sqrt(pooled * (1/n1 + 1/n2))
    p_two = math.erfc(abs(t) / math.sqrt(2))
    return t, p_two

def auc(a, b):
    """AUC treating a as positives."""
    n1, n2 = len(a), len(b)
    U, _, _ = mann_whitney_u_one(a, b)
    return U / (n1 * n2)

# ─── Scoring functions ────────────────────────────────────────────────
# Training-derived standardization constants, per CpG
train_stats_A = {r['cpg']: r for r in panel_A['cpgs']}
train_stats_B = {r['cpg']: r for r in panel_B['cpgs']}

def a_dir_score(sentrix_betas, train_stats):
    """Directional A-score using training-set HC mean/SD and training directions."""
    contribs = []
    for cpg, r in train_stats.items():
        b = sentrix_betas.get(cpg)
        if b is None or not (0 < b < 1): continue
        z = (b - r['mean_hc_train']) / r['sd_hc_train'] if r['sd_hc_train'] > 0 else 0
        contribs.append(r['direction'] * z)
    if len(contribs) < max(3, int(0.7*len(train_stats))): return None
    return sum(contribs) / len(contribs)

def a_entropy_pooled(sentrix_betas, cpg_list):
    """VAL-050 pooled entropy A-score (null-comparator)."""
    vals = [sentrix_betas[c] for c in cpg_list if sentrix_betas.get(c) is not None and 0 < sentrix_betas.get(c, 0) < 1]
    if len(vals) < 12: return None
    mean_b = sum(vals)/len(vals)
    return H(mean_b) / H_MIN_IMMUNE_METHYL

def per_cpg_within_sample_variance(sentrix_betas, cpg_list):
    vals = [sentrix_betas[c] for c in cpg_list if sentrix_betas.get(c) is not None and 0 < sentrix_betas.get(c, 0) < 1]
    if len(vals) < 5: return None
    return statistics.pvariance(vals)

# ─── Score AIBL holdout ───────────────────────────────────────────────
print("="*72)
print("VAL-051 — AD-Directional Immune Panel")
print("="*72)
print()

hold_AD = []; hold_HC = []; hold_MCI = []
for s in manifest:
    if split_map.get(s['gsm']) != 'holdout': continue
    sen = s['sentrix']
    row = dict(s)
    row['A_dir_A'] = a_dir_score(betas[sen], train_stats_A)
    row['A_dir_B'] = a_dir_score(betas[sen], train_stats_B)
    row['A_entropy'] = a_entropy_pooled(betas[sen], PANEL_18)
    row['within_var'] = per_cpg_within_sample_variance(betas[sen], PANEL_18)
    if s['disease status'] == "Alzheimer's disease": hold_AD.append(row)
    elif s['disease status'] == 'healthy control': hold_HC.append(row)
    elif s['disease status'] == 'Mild Cognitive Impairment': hold_MCI.append(row)

print(f"AIBL HOLDOUT: AD={len(hold_AD)} MCI={len(hold_MCI)} HC={len(hold_HC)}")
print(f"Rule A panel: {panel_A['n_selected']} CpGs")
print(f"Rule B panel: {panel_B['n_selected']} CpGs (full 18)")
print()

# ─── H1 primary on holdout ────────────────────────────────────────────
print("="*72)
print("PRIMARY — H1: A_dir(AD) > A_dir(HC) on AIBL HOLDOUT")
print("="*72)

for rule_name, panel, key in [('Rule A (selected)', panel_A, 'A_dir_A'),
                               ('Rule B (all 18)',  panel_B, 'A_dir_B')]:
    ad_scores = [r[key] for r in hold_AD if r[key] is not None]
    hc_scores = [r[key] for r in hold_HC if r[key] is not None]
    print(f"\n--- {rule_name}: n_panel={panel['n_selected']} ---")
    print(f"  n_AD={len(ad_scores)}  n_HC={len(hc_scores)}")
    if not ad_scores or not hc_scores: continue
    m_ad = statistics.mean(ad_scores); m_hc = statistics.mean(hc_scores)
    print(f"  Mean A_dir(AD) = {m_ad:+.4f}")
    print(f"  Mean A_dir(HC) = {m_hc:+.4f}")
    print(f"  Δ = {m_ad - m_hc:+.4f}")
    U, z, p = mann_whitney_u_one(ad_scores, hc_scores)
    d = cohens_d(ad_scores, hc_scores)
    lo, med, hi = bootstrap_d(ad_scores, hc_scores, N_BOOT, SEED)
    a = auc(ad_scores, hc_scores)
    print(f"  Cohen's d       = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  MWU z           = {z:.3f}")
    print(f"  MWU p_onesided  = {p:.4g}")
    print(f"  AUC             = {a:.4f}")

# ─── Null-comparator: VAL-050 pooled entropy on holdout ──────────────
print(f"\n--- Null-comparator: pooled entropy A-score (VAL-050 metric) on holdout ---")
ad_ent = [r['A_entropy'] for r in hold_AD if r['A_entropy'] is not None]
hc_ent = [r['A_entropy'] for r in hold_HC if r['A_entropy'] is not None]
d_ent = cohens_d(ad_ent, hc_ent)
_, _, p_ent = mann_whitney_u_one(ad_ent, hc_ent)
print(f"  d = {d_ent:+.4f}, p = {p_ent:.4g}  (if ~0, confirms VAL-050 null holds on this subset)")

# ─── H3 sex-stratified ────────────────────────────────────────────────
print()
print("="*72)
print("SECONDARY — H3: sex-stratified A_dir (Rule A)")
print("="*72)
h3_results = {}
for sex in ['Male','Female']:
    ad = [r['A_dir_A'] for r in hold_AD if r['gender']==sex and r['A_dir_A'] is not None]
    hc = [r['A_dir_A'] for r in hold_HC if r['gender']==sex and r['A_dir_A'] is not None]
    if len(ad) < 5 or len(hc) < 5:
        print(f"  {sex}: n too small (n_AD={len(ad)}, n_HC={len(hc)})")
        h3_results[sex] = {'n_AD': len(ad), 'n_HC': len(hc), 'skipped': True}
        continue
    d = cohens_d(ad, hc)
    _, _, p = mann_whitney_u_one(ad, hc)
    delta = statistics.mean(ad) - statistics.mean(hc)
    print(f"  {sex}: n_AD={len(ad)}, n_HC={len(hc)}, Δ={delta:+.4f}, d={d:+.4f}, p={p:.4g}")
    h3_results[sex] = {'n_AD': len(ad), 'n_HC': len(hc), 'delta': delta, 'cohens_d': d, 'p': p}

# ─── H4 bimodality ────────────────────────────────────────────────────
print()
print("="*72)
print("SECONDARY — H4: within-sample β-variance across panel CpGs (AD vs HC)")
print("="*72)
ad_var = [r['within_var'] for r in hold_AD if r['within_var'] is not None]
hc_var = [r['within_var'] for r in hold_HC if r['within_var'] is not None]
m_ad_var = statistics.mean(ad_var); m_hc_var = statistics.mean(hc_var)
t, p_lev = levene_test(ad_var, hc_var)
d_var = cohens_d(ad_var, hc_var)
print(f"  Mean within-sample variance (AD): {m_ad_var:.5f}")
print(f"  Mean within-sample variance (HC): {m_hc_var:.5f}")
print(f"  Ratio = {m_ad_var/m_hc_var:.3f}")
print(f"  Cohen's d on variance = {d_var:+.4f}")
print(f"  Levene p (two-sided)  = {p_lev:.4g}")

# ─── Primary decision ─────────────────────────────────────────────────
print()
print("="*72)
print("DECISION — pre-locked 4×2 outcome matrix (AIBL-holdout arm)")
print("="*72)
ad_A = [r['A_dir_A'] for r in hold_AD if r['A_dir_A'] is not None]
hc_A = [r['A_dir_A'] for r in hold_HC if r['A_dir_A'] is not None]
d_A = cohens_d(ad_A, hc_A)
_, _, p_A = mann_whitney_u_one(ad_A, hc_A)

if d_A > 0.3 and p_A < 0.05:
    decision = "OUTCOME 1 (AIBL arm) — FULL RECOVERY on AIBL holdout"
elif 0.1 < d_A <= 0.3 and p_A < 0.15:
    decision = "OUTCOME 3 (AIBL arm) — DIRECTION-POSITIVE-WEAK"
elif d_A < 0.1:
    decision = "OUTCOME 4 (AIBL arm) — NULL even with directional weighting"
elif d_A < 0:
    decision = "OUTCOME 5 (AIBL arm) — ANTI-DIRECTION (training overfitting)"
else:
    decision = "BORDERLINE (AIBL arm)"
print(f"  {decision}")
print(f"  d={d_A:+.4f}, p={p_A:.4g}")

# ─── Save ─────────────────────────────────────────────────────────────
results = {
    'val_id': 'VAL-051',
    'cohort': 'AIBL GSE153712 (holdout split)',
    'panel_ruleA_n': panel_A['n_selected'],
    'panel_ruleB_n': panel_B['n_selected'],
    'holdout_counts': {'AD': len(hold_AD), 'MCI': len(hold_MCI), 'HC': len(hold_HC)},
    'H1_primary_ruleA': {
        'n_AD': len(ad_A), 'n_HC': len(hc_A),
        'cohens_d': d_A, 'p_onesided': p_A,
        'auc': auc(ad_A, hc_A),
    },
    'H1_ruleB_all18': {
        'cohens_d': cohens_d([r['A_dir_B'] for r in hold_AD if r['A_dir_B'] is not None],
                             [r['A_dir_B'] for r in hold_HC if r['A_dir_B'] is not None]),
    },
    'null_comparator_pooled_entropy': {
        'cohens_d': d_ent, 'p_onesided': p_ent,
    },
    'H3_sex': h3_results,
    'H4_bimodality': {
        'mean_var_AD': m_ad_var, 'mean_var_HC': m_hc_var,
        'variance_ratio': m_ad_var/m_hc_var,
        'cohens_d': d_var, 'levene_p': p_lev,
    },
    'decision_AIBL_arm': decision,
    'per_sample_holdout': [
        {'gsm': r['gsm'], 'status': r['disease status'], 'sex': r['gender'],
         'A_dir_A': r['A_dir_A'], 'A_dir_B': r['A_dir_B'],
         'A_entropy': r['A_entropy'], 'within_var': r['within_var']}
        for r in hold_AD + hold_MCI + hold_HC
    ],
}

with open('VAL_051_RESULTS.json','w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults written: VAL_051_RESULTS.json")
