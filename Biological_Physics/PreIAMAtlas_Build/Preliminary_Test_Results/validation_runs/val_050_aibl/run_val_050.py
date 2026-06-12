#!/usr/bin/env python3
"""
VAL-050 — AIBL AD Immune-Class Cross-Sectional Replication
Post-prereg, pre-locked analysis.

Inputs (all hash-sealed before execution):
  - aibl_manifest.json   (per-sample metadata: GSM, sentrix, disease status, sex)
  - aibl_imm_betas.json  (per-sample β values for 18 IMM_CPGS_EPIC panel CpGs)
  - VAL_050_PREREG.md    (the prereg document)

Outputs:
  - VAL_050_RESULTS.json
  - VAL_050_REPORT.md

Frozen constants:
  H_min(immune, methyl) = 0.838889  (G-002 MCMC posterior)

Panel (18 CpGs, AIBL-EPIC intersection of IMM_CPGS_RAW):
  see list in VAL_050_PREREG.md section 3
"""

import json, math, hashlib, random, statistics, os, sys, time

# ════════════════════════════════════════════════════════════════════════
# FROZEN CONSTANTS — DO NOT MODIFY
# ════════════════════════════════════════════════════════════════════════
H_MIN_IMMUNE_METHYL = 0.838889
PANEL_18 = [
    'cg00431549','cg01127300','cg02228185','cg02489552','cg04023335',
    'cg09809672','cg10632894','cg12554573','cg14614643','cg16867657',
    'cg17861230','cg18834029','cg22454769','cg22736354','cg23244761',
    'cg25432518','cg25809905','cg26614073',
]
assert len(PANEL_18) == 18
PANEL_SET = set(PANEL_18)

SEED = 42
N_PERM = 10_000
N_BOOT = 10_000
ALPHA = 0.05  # one-sided for H1 primary

# ════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════
def H(beta):
    if beta <= 0 or beta >= 1:
        return 0.0
    return -beta*math.log2(beta) - (1-beta)*math.log2(1-beta)

def A_score(beta_mean):
    return H(beta_mean) / H_MIN_IMMUNE_METHYL

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def mann_whitney_u(a, b):
    """Mann-Whitney U, one-sided (a > b). Returns (U, p_onesided)."""
    n1, n2 = len(a), len(b)
    combined = [(v, 0) for v in a] + [(v, 1) for v in b]
    combined.sort(key=lambda x: x[0])
    ranks_a = 0.0; i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-indexed midpoint
        for k in range(i, j):
            if combined[k][1] == 0:
                ranks_a += avg_rank
        i = j
    U_a = ranks_a - n1*(n1+1)/2.0
    # Normal approx for p
    mu_U = n1*n2/2.0
    sigma_U = math.sqrt(n1*n2*(n1+n2+1)/12.0)
    z = (U_a - mu_U) / sigma_U
    # one-sided p for "a > b" corresponds to large U_a
    # p = P(Z > z_observed)
    p_one = 0.5 * math.erfc(z / math.sqrt(2))
    return U_a, z, p_one

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    m1, m2 = statistics.mean(a), statistics.mean(b)
    s1, s2 = statistics.pstdev(a), statistics.pstdev(b)
    pooled = math.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / (n1+n2-2))
    if pooled == 0: return 0.0
    return (m1 - m2) / pooled

def bootstrap_cohens_d(a, b, n_boot, seed):
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

def permutation_null(a, b, n_perm, seed):
    """One-sided permutation p for (mean(a) - mean(b))."""
    rng = random.Random(seed)
    observed = statistics.mean(a) - statistics.mean(b)
    pool = list(a) + list(b)
    n1 = len(a)
    count_at_or_above = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        sim_a = pool[:n1]
        sim_b = pool[n1:]
        if (statistics.mean(sim_a) - statistics.mean(sim_b)) >= observed:
            count_at_or_above += 1
    return (count_at_or_above + 1) / (n_perm + 1)

def jonckheere_terpstra(groups):
    """
    Groups is a list of lists in ordered sequence (e.g. HC, MCI, AD).
    Returns J statistic and one-sided normal-approx p (trend up).
    """
    J = 0
    n = [len(g) for g in groups]
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            for x in groups[i]:
                for y in groups[j]:
                    if x < y: J += 1
                    elif x == y: J += 0.5
    # Expected J under null
    total_n = sum(n)
    E_J = (total_n**2 - sum(ni**2 for ni in n)) / 4.0
    V_J = (total_n**2 * (2*total_n + 3) - sum(ni**2*(2*ni+3) for ni in n)) / 72.0
    z = (J - E_J) / math.sqrt(V_J)
    p_one = 0.5 * math.erfc(z / math.sqrt(2))
    return J, z, p_one

def auc_from_groups(a, b):
    """ROC-AUC treating a as positives (AD) and b as negatives (HC).
    AUC = P(a > b) + 0.5 * P(a == b) = U_AD / (n_AD * n_HC)."""
    # Use mann-whitney U
    n1, n2 = len(a), len(b)
    combined = [(v, 0) for v in a] + [(v, 1) for v in b]
    combined.sort(key=lambda x: x[0])
    ranks_a = 0.0; i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            if combined[k][1] == 0:
                ranks_a += avg_rank
        i = j
    U_a = ranks_a - n1*(n1+1)/2.0
    return U_a / (n1 * n2)

# ════════════════════════════════════════════════════════════════════════
# LOAD + QC
# ════════════════════════════════════════════════════════════════════════
print("="*72)
print("VAL-050 — AIBL AD Immune-Class Cross-Sectional Replication")
print("="*72)
print()
print(f"Script started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Seed (all random): {SEED}")
print(f"H_min(immune, methyl) = {H_MIN_IMMUNE_METHYL} (frozen)")
print(f"Panel: IMM_CPGS_EPIC_18 ({len(PANEL_18)} CpGs)")
print()

manifest = json.load(open('aibl_manifest.json'))
betas = json.load(open('aibl_imm_betas.json'))
print(f"Loaded manifest: {len(manifest)} samples")
print(f"Loaded betas:    {len(betas)} sentrix positions")

# Join manifest to betas via sentrix
joined = []
dropped = {'no_sentrix_beta': 0, 'missing_status': 0, 'missing_sex': 0,
           'cpg_coverage_low': 0, 'invalid_beta': 0}
for s in manifest:
    sentrix = s['sentrix']
    if sentrix not in betas:
        dropped['no_sentrix_beta'] += 1
        continue
    status = s.get('disease status')
    sex = s.get('gender')
    if not status:
        dropped['missing_status'] += 1
        continue
    if not sex:
        dropped['missing_sex'] += 1
        continue
    # Check β validity and panel coverage
    bvals = betas[sentrix]
    valid = []
    for cpg, v in bvals.items():
        if cpg in PANEL_SET:
            try:
                fv = float(v)
                if 0.0 < fv < 1.0:
                    valid.append(fv)
            except (ValueError, TypeError):
                pass
    if len(valid) < 12:
        dropped['cpg_coverage_low'] += 1
        continue
    mean_beta = sum(valid) / len(valid)
    if not (0.0 < mean_beta < 1.0):
        dropped['invalid_beta'] += 1
        continue
    A = A_score(mean_beta)
    joined.append({
        'gsm': s['gsm'], 'sentrix': sentrix,
        'status': status, 'sex': sex,
        'n_cpgs_used': len(valid),
        'mean_beta': mean_beta, 'A_immune': A,
    })

print(f"\nQC exclusions:")
for k, v in dropped.items():
    print(f"  {k}: {v}")
print(f"Samples retained: {len(joined)}")

# Group splits
HC = [s for s in joined if s['status'] == 'healthy control']
MCI = [s for s in joined if s['status'] == 'Mild Cognitive Impairment']
AD = [s for s in joined if s['status'] == "Alzheimer's disease"]
print(f"\nGroup sizes:")
print(f"  HC  n={len(HC)}")
print(f"  MCI n={len(MCI)}")
print(f"  AD  n={len(AD)}")

# ════════════════════════════════════════════════════════════════════════
# PRIMARY: H1 — A_immune(AD) > A_immune(HC), one-sided MWU
# ════════════════════════════════════════════════════════════════════════
print()
print("="*72)
print("PRIMARY — H1: A_immune(AD) > A_immune(HC), one-sided")
print("="*72)

A_AD = [s['A_immune'] for s in AD]
A_HC = [s['A_immune'] for s in HC]

mean_AD = statistics.mean(A_AD); sd_AD = statistics.pstdev(A_AD)
mean_HC = statistics.mean(A_HC); sd_HC = statistics.pstdev(A_HC)
delta_A = mean_AD - mean_HC
print(f"  Mean A_immune(AD) = {mean_AD:.5f}  SD = {sd_AD:.5f}  n={len(A_AD)}")
print(f"  Mean A_immune(HC) = {mean_HC:.5f}  SD = {sd_HC:.5f}  n={len(A_HC)}")
print(f"  ΔA (AD - HC)      = {delta_A:+.5f}")

U, z, p_mwu = mann_whitney_u(A_AD, A_HC)
d = cohens_d(A_AD, A_HC)
d_lo, d_med, d_hi = bootstrap_cohens_d(A_AD, A_HC, N_BOOT, SEED)
auc = auc_from_groups(A_AD, A_HC)
print(f"  Mann-Whitney U    = {U:.1f}")
print(f"  MWU z             = {z:.3f}")
print(f"  MWU p_onesided    = {p_mwu:.4g}")
print(f"  Cohen's d         = {d:+.4f}  (bootstrap 95% CI [{d_lo:+.4f}, {d_hi:+.4f}])")
print(f"  ROC-AUC           = {auc:.4f}")

# Permutation p
print(f"  Computing 10,000-permutation null...")
p_perm = permutation_null(A_AD, A_HC, N_PERM, SEED)
print(f"  Permutation p     = {p_perm:.4g}")

# ════════════════════════════════════════════════════════════════════════
# SECONDARY
# ════════════════════════════════════════════════════════════════════════
print()
print("="*72)
print("SECONDARY — H2: monotonic trend HC < MCI < AD (Jonckheere-Terpstra)")
print("="*72)
A_MCI = [s['A_immune'] for s in MCI]
J, Jz, pJ = jonckheere_terpstra([A_HC, A_MCI, A_AD])
print(f"  Mean A_immune(MCI) = {statistics.mean(A_MCI):.5f}  SD = {statistics.pstdev(A_MCI):.5f}")
print(f"  J statistic        = {J:.0f}")
print(f"  J z                = {Jz:.3f}")
print(f"  J p_onesided       = {pJ:.4g}")

print()
print("="*72)
print("SECONDARY — H3: sex-stratified replication")
print("="*72)
for sex in ['Male', 'Female']:
    A_AD_s = [s['A_immune'] for s in AD if s['sex'] == sex]
    A_HC_s = [s['A_immune'] for s in HC if s['sex'] == sex]
    if len(A_AD_s) < 5 or len(A_HC_s) < 5:
        print(f"  {sex}: skipped (n too small)")
        continue
    d_s = cohens_d(A_AD_s, A_HC_s)
    _, _, p_s = mann_whitney_u(A_AD_s, A_HC_s)
    delta_s = statistics.mean(A_AD_s) - statistics.mean(A_HC_s)
    print(f"  {sex}: n_AD={len(A_AD_s)}, n_HC={len(A_HC_s)}, ΔA={delta_s:+.5f}, d={d_s:+.4f}, p={p_s:.4g}")

print()
print("="*72)
print("SECONDARY — H4: per-CpG HC vs AD with BH-FDR")
print("="*72)
# For each CpG, extract per-sample β, compute MWU AD vs HC, collect p
per_cpg = []
for cpg in PANEL_18:
    ad_vals = []
    hc_vals = []
    for s_ad in AD:
        b = betas[s_ad['sentrix']].get(cpg)
        if b is not None and 0 < b < 1: ad_vals.append(b)
    for s_hc in HC:
        b = betas[s_hc['sentrix']].get(cpg)
        if b is not None and 0 < b < 1: hc_vals.append(b)
    if len(ad_vals) < 10 or len(hc_vals) < 10:
        per_cpg.append((cpg, len(ad_vals), len(hc_vals), None, None, None))
        continue
    _, _, p_up = mann_whitney_u(ad_vals, hc_vals)   # β_AD > β_HC
    _, _, p_dn = mann_whitney_u(hc_vals, ad_vals)   # β_AD < β_HC
    delta_beta = statistics.mean(ad_vals) - statistics.mean(hc_vals)
    # Report two-sided
    p_two = 2 * min(p_up, p_dn)
    per_cpg.append((cpg, len(ad_vals), len(hc_vals), delta_beta, p_two, p_up if delta_beta > 0 else p_dn))

# BH-FDR
valid_ps = [(i, r[4]) for i, r in enumerate(per_cpg) if r[4] is not None]
valid_ps.sort(key=lambda x: x[1])
m = len(valid_ps)
fdr = {}
for rank, (i, p) in enumerate(valid_ps, start=1):
    fdr[i] = p * m / rank

print(f"  {'CpG':<13} {'n_AD':>5} {'n_HC':>5} {'Δβ':>8} {'p_two':>10} {'q_FDR':>10}")
sig = 0
for i, (cpg, nAD, nHC, dbeta, p, p_dir) in enumerate(per_cpg):
    q = fdr.get(i, float('nan'))
    if p is None:
        print(f"  {cpg:<13} {nAD:>5} {nHC:>5} {'n/a':>8} {'n/a':>10} {'n/a':>10}")
    else:
        flag = " *" if q < 0.05 else ""
        print(f"  {cpg:<13} {nAD:>5} {nHC:>5} {dbeta:>+8.4f} {p:>10.4g} {q:>10.4g}{flag}")
        if q < 0.05:
            sig += 1
print(f"  Significant at FDR<0.05: {sig}/{m}")

# ════════════════════════════════════════════════════════════════════════
# DECISION — apply the 4 pre-locked outcomes
# ════════════════════════════════════════════════════════════════════════
print()
print("="*72)
print("DECISION — pre-locked outcomes (VAL_050_PREREG.md section 6)")
print("="*72)

if p_mwu < 0.05 and d > 0.3:
    outcome = "OUTCOME 1 — POSITIVE"
    interp = "Framework generalizes. Xu-breast-derived immune panel detects AD architectural drift at per-patient cohort level."
elif p_mwu < 0.10 and 0 < d <= 0.3:
    outcome = "OUTCOME 2 — DIRECTION-POSITIVE-WEAK"
    interp = "Direction correct, effect size small. Consistent with VAL-040 age-matched ΔA. Directional replication of VAL-040 at per-patient scale."
elif abs(d) < 0.1 or p_mwu > 0.10:
    outcome = "OUTCOME 3 — NULL"
    interp = "Panel is class-specific not disease-general. Supports purpose-built AD panel case (Panel B / VAL-051)."
elif d < -0.1 and p_mwu < 0.10:
    outcome = "OUTCOME 4 — NEGATIVE"
    interp = "Framework requires revision for AD immune-class direction."
else:
    outcome = "OUTCOME 2/3 — BORDERLINE"
    interp = "Between Direction-positive-weak and Null. Report all numbers, interpret conservatively."

print(f"  {outcome}")
print(f"  {interp}")
print()

# ════════════════════════════════════════════════════════════════════════
# WRITE RESULTS JSON
# ════════════════════════════════════════════════════════════════════════
results = {
    'val_id': 'VAL-050',
    'title': 'AIBL AD Immune-Class Cross-Sectional Replication',
    'run_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'seed': SEED,
    'cohort': 'AIBL GSE153712',
    'platform': 'Illumina EPIC 850K',
    'panel_name': 'IMM_CPGS_EPIC_18',
    'panel_cpgs': PANEL_18,
    'panel_cpg_count': len(PANEL_18),
    'H_min_immune_methyl': H_MIN_IMMUNE_METHYL,
    'n_AD': len(AD), 'n_MCI': len(MCI), 'n_HC': len(HC),
    'qc_dropped': dropped,
    'primary_H1': {
        'test': 'Mann-Whitney U one-sided (A_AD > A_HC)',
        'mean_A_AD': mean_AD, 'sd_A_AD': sd_AD,
        'mean_A_HC': mean_HC, 'sd_A_HC': sd_HC,
        'delta_A': delta_A,
        'U': U, 'z': z,
        'p_mwu_onesided': p_mwu,
        'p_permutation_onesided': p_perm,
        'cohens_d': d,
        'cohens_d_95ci': [d_lo, d_hi],
        'auc': auc,
    },
    'secondary_H2_trend': {
        'test': 'Jonckheere-Terpstra HC<MCI<AD',
        'mean_A_MCI': statistics.mean(A_MCI),
        'J': J, 'z': Jz, 'p_onesided': pJ,
    },
    'secondary_H3_sex': {
        sex: {
            'n_AD': sum(1 for s in AD if s['sex']==sex),
            'n_HC': sum(1 for s in HC if s['sex']==sex),
            'delta_A': statistics.mean([s['A_immune'] for s in AD if s['sex']==sex]) -
                       statistics.mean([s['A_immune'] for s in HC if s['sex']==sex]),
            'cohens_d': cohens_d(
                [s['A_immune'] for s in AD if s['sex']==sex],
                [s['A_immune'] for s in HC if s['sex']==sex]),
        } for sex in ['Male','Female']
    },
    'secondary_H4_per_cpg': [
        {'cpg': c, 'n_AD': nAD, 'n_HC': nHC, 'delta_beta': dbeta,
         'p_two': p, 'q_FDR': fdr.get(i)}
        for i, (c, nAD, nHC, dbeta, p, _) in enumerate(per_cpg)
    ],
    'outcome': outcome,
    'interpretation': interp,
    'per_sample_A_scores': joined,
}

with open('VAL_050_RESULTS.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"Results written to VAL_050_RESULTS.json ({os.path.getsize('VAL_050_RESULTS.json'):,} bytes)")
print(f"Script ended: {time.strftime('%Y-%m-%d %H:%M:%S')}")
