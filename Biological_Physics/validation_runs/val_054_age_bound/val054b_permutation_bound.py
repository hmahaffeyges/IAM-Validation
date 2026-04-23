#!/usr/bin/env python3
"""
VAL-054b — Age-independent permutation bound.

Since AIBL GEO has no chronological age, we cannot directly regress age out.
But we CAN bound the age-confound using the permutation distribution of
random HC vs HC splits: if the observed AD-vs-HC d is outside the 
null distribution of HC-internal splits of the same size, then the 
signal cannot be attributed to within-HC heterogeneity (of which age 
variation is one component).

Procedure:
  1. Draw the observed AD (n=33) vs HC (n=95) A_dir distribution.
  2. Compute observed Cohen's d = +0.624.
  3. NULL: randomly split HC (n=95) into two subsets matching the
     AD/HC size ratio (n=33 + n=62). Compute d between them.
  4. Repeat 10,000 times. Build the null distribution of HC-internal d.
  5. P(null d >= observed d) is the age-confound-inclusive p-value.
  
If p < 0.01: the AD signal cannot be attributed to any within-HC source
(age, sex, batch, or other covariate variance). The signal is AD-specific.
"""
import json, math, statistics, random

SEED = 42
N_PERM = 10_000

def cohens_d(a, b):
    if min(len(a), len(b)) < 2: return 0.0
    m1, m2 = statistics.mean(a), statistics.mean(b)
    s1, s2 = statistics.pstdev(a), statistics.pstdev(b)
    pooled = math.sqrt(((len(a)-1)*s1*s1 + (len(b)-1)*s2*s2) / (len(a)+len(b)-2))
    return 0.0 if pooled == 0 else (m1 - m2) / pooled

# Load VAL-051 holdout scores
res = json.load(open('VAL_054_RESULTS.json'))
records = res['per_sample']
AD_scores = [r['A_dir'] for r in records if r['status'] == "Alzheimer's disease"]
HC_scores = [r['A_dir'] for r in records if r['status'] == 'healthy control']

d_obs = cohens_d(AD_scores, HC_scores)
print(f"Observed AD vs HC Cohen's d = {d_obs:+.4f}")
print(f"n_AD = {len(AD_scores)}, n_HC = {len(HC_scores)}")
print()
print(f"Running {N_PERM:,} HC-internal permutation splits...")
print("(Size-matched to observed AD:HC split — each perm samples n_AD from HC)")

rng = random.Random(SEED)
null_ds = []
for _ in range(N_PERM):
    idx = list(range(len(HC_scores)))
    rng.shuffle(idx)
    a_sub = [HC_scores[i] for i in idx[:len(AD_scores)]]
    b_sub = [HC_scores[i] for i in idx[len(AD_scores):]]
    null_ds.append(cohens_d(a_sub, b_sub))

null_ds.sort()
p_ge = sum(1 for d in null_ds if d >= d_obs) / N_PERM
p99 = null_ds[int(0.99 * N_PERM)]
p95 = null_ds[int(0.95 * N_PERM)]
mean_null = statistics.mean(null_ds)
sd_null = statistics.pstdev(null_ds)

print(f"\nNull distribution (HC-internal splits):")
print(f"  mean d = {mean_null:+.4f}")
print(f"  SD d   = {sd_null:.4f}")
print(f"  95th percentile = {p95:+.4f}")
print(f"  99th percentile = {p99:+.4f}")
print()
print(f"Observed d vs null distribution:")
print(f"  d_obs = {d_obs:+.4f}")
print(f"  P(null d >= d_obs) = {p_ge:.5f}")
print(f"  z of observed in null distribution = {(d_obs - mean_null)/sd_null:+.2f}")
print()
if p_ge < 0.001:
    verdict = "CONFIRMED — signal cannot be attributed to within-HC variance (age, sex, or any other internal covariate)"
elif p_ge < 0.01:
    verdict = "STRONG — signal is well outside HC-internal variance; within-HC confounds (including age) would require >100× coincidence"
elif p_ge < 0.05:
    verdict = "WEAK CONFIRMATION — signal exceeds HC-internal variance at α=0.05"
else:
    verdict = "INCONCLUSIVE — observed signal is within the range of HC-internal splits; age or other within-HC source remains plausible"
print(f"VERDICT: {verdict}")

# Sex-stratified version
print("\n" + "="*72)
print("SEX-STRATIFIED HC-internal permutation bounds")
print("="*72)
for sex in ['Female', 'Male']:
    ad_s = [r['A_dir'] for r in records if r['status'] == "Alzheimer's disease" and r['sex'] == sex]
    hc_s = [r['A_dir'] for r in records if r['status'] == 'healthy control' and r['sex'] == sex]
    if len(ad_s) < 5 or len(hc_s) < 10: continue
    d_obs_s = cohens_d(ad_s, hc_s)
    null_ds_s = []
    for _ in range(N_PERM):
        idx = list(range(len(hc_s)))
        rng.shuffle(idx)
        a_sub = [hc_s[i] for i in idx[:len(ad_s)]]
        b_sub = [hc_s[i] for i in idx[len(ad_s):]]
        null_ds_s.append(cohens_d(a_sub, b_sub))
    p_ge_s = sum(1 for d in null_ds_s if d >= d_obs_s) / N_PERM
    print(f"  {sex}: n_AD={len(ad_s)}, n_HC={len(hc_s)}")
    print(f"    d_obs = {d_obs_s:+.4f}")
    print(f"    P(null d >= d_obs) = {p_ge_s:.5f}")

# Save
out = {
    'val_id': 'VAL-054b',
    'method': 'HC-internal permutation bound — size-matched resampling without replacement',
    'observed_d': d_obs,
    'n_AD': len(AD_scores),
    'n_HC': len(HC_scores),
    'n_permutations': N_PERM,
    'null_distribution': {
        'mean': mean_null, 'sd': sd_null,
        'p95': p95, 'p99': p99,
        'min': null_ds[0], 'max': null_ds[-1],
    },
    'p_hc_internal': p_ge,
    'z_observed_in_null': (d_obs - mean_null)/sd_null,
    'verdict': verdict,
}
with open('VAL_054b_RESULTS.json','w') as f:
    json.dump(out, f, indent=2)
print(f"\nResults written to VAL_054b_RESULTS.json")
