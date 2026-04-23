#!/usr/bin/env python3
"""
VAL-051 Step 2 — Panel selection on TRAINING SET ONLY.

Rule A: |Δβ| > 0.015 AND q_FDR < 0.10 on AD vs HC in training set.
Rule B (secondary): use all 18 CpGs with directional weighting.

HOLDOUT SET IS NOT TOUCHED. AddNeuroMed IS NOT TOUCHED.

Outputs:
  val051_panel_ruleA.json  — selected CpGs, directions, training stats
  val051_panel_ruleB.json  — all 18, directions, training stats
  val051_train_stats.json  — HC-train mean/SD per CpG for standardization
"""
import json, math, statistics, hashlib

PANEL_18 = [
    'cg00431549','cg01127300','cg02228185','cg02489552','cg04023335',
    'cg09809672','cg10632894','cg12554573','cg14614643','cg16867657',
    'cg17861230','cg18834029','cg22454769','cg22736354','cg23244761',
    'cg25432518','cg25809905','cg26614073',
]
DELTA_BETA_THRESH = 0.015
Q_FDR_THRESH = 0.10

manifest = json.load(open('aibl_manifest.json'))
betas = json.load(open('aibl_imm_betas.json'))
split_map = json.load(open('val051_split_map.json'))['split']

def mann_whitney_u_two(a, b):
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
    mu_U = n1*n2/2.0
    sigma_U = math.sqrt(n1*n2*(n1+n2+1)/12.0)
    z = (U_a - mu_U) / sigma_U
    p_two = math.erfc(abs(z) / math.sqrt(2))
    return U_a, z, p_two

# Partition
train_AD_sentrix = []; train_HC_sentrix = []
for s in manifest:
    if split_map.get(s['gsm']) != 'train': continue
    if s['disease status'] == "Alzheimer's disease":
        train_AD_sentrix.append(s['sentrix'])
    elif s['disease status'] == 'healthy control':
        train_HC_sentrix.append(s['sentrix'])

print(f"TRAINING set: AD n={len(train_AD_sentrix)}, HC n={len(train_HC_sentrix)}")

# Per-CpG stats on training only
results = []
for cpg in PANEL_18:
    ad_vals = []
    hc_vals = []
    for sen in train_AD_sentrix:
        b = betas[sen].get(cpg)
        if b is not None and 0 < b < 1: ad_vals.append(b)
    for sen in train_HC_sentrix:
        b = betas[sen].get(cpg)
        if b is not None and 0 < b < 1: hc_vals.append(b)
    if len(ad_vals) < 20 or len(hc_vals) < 50:
        continue
    mean_ad = statistics.mean(ad_vals); mean_hc = statistics.mean(hc_vals)
    delta = mean_ad - mean_hc
    _, _, p = mann_whitney_u_two(ad_vals, hc_vals)
    direction = 1 if delta > 0 else -1
    results.append({
        'cpg': cpg, 'delta_beta': delta,
        'mean_ad_train': mean_ad, 'mean_hc_train': mean_hc,
        'sd_hc_train': statistics.pstdev(hc_vals),
        'n_ad_train': len(ad_vals), 'n_hc_train': len(hc_vals),
        'p_two': p,
        'direction': direction,
    })

# BH-FDR
results.sort(key=lambda r: r['p_two'])
m = len(results)
for rank, r in enumerate(results, start=1):
    r['q_FDR'] = r['p_two'] * m / rank

# Rule A: |Δβ| > 0.015 AND q < 0.10
ruleA = [r for r in results if abs(r['delta_beta']) > DELTA_BETA_THRESH and r['q_FDR'] < Q_FDR_THRESH]
ruleB = results  # all 18

print(f"\nPer-CpG training stats (sorted by p_two):")
print(f"{'CpG':<13} {'Δβ':>8} {'dir':>4} {'p_two':>10} {'q_FDR':>10} {'included_A':>12}")
for r in results:
    inc = 'YES' if r in ruleA else '---'
    print(f"{r['cpg']:<13} {r['delta_beta']:>+8.4f} {r['direction']:>+4} {r['p_two']:>10.4g} {r['q_FDR']:>10.4g} {inc:>12}")

print(f"\nRule A selected: {len(ruleA)} / {len(results)} CpGs")
print(f"Directions in Rule A: up={sum(1 for r in ruleA if r['direction']==1)}, down={sum(1 for r in ruleA if r['direction']==-1)}")

with open('val051_panel_ruleA.json','w') as f:
    json.dump({
        'rule': 'Rule A: |Δβ| > 0.015 AND q_FDR < 0.10',
        'delta_beta_thresh': DELTA_BETA_THRESH,
        'q_fdr_thresh': Q_FDR_THRESH,
        'n_selected': len(ruleA),
        'cpgs': ruleA,
    }, f, indent=2)

with open('val051_panel_ruleB.json','w') as f:
    json.dump({
        'rule': 'Rule B: all 18 CpGs directionally weighted',
        'n_selected': len(ruleB),
        'cpgs': ruleB,
    }, f, indent=2)

print(f"\nPanels written: val051_panel_ruleA.json, val051_panel_ruleB.json")
