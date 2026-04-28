"""
VAL-107 — Cardio-epic CHK-3.1B calibration on TCGA HM450K sesame Level 3

Pre-registration SHA-256: b58ce4dbd422198c7cbd6e7d1ee1cdbed86a758afc204189f8a9e070fd700d82
Sealed: 2026-04-28T22:19:26Z (before subset β access)
RNG seed: 20260428

Methodology (frozen by sealed prereg):
1. Load 8,100-CpG cardio-epic subset (SHA 5a00e29ace75daae5a5bf7e3cfca26c16aa6dbd92750d16ebeaba4e874c48511)
2. Reuse VAL-106 cohort_manifest.json (210 samples, 144 KIRC QC-pass + 50 PRAD QC-pass)
3. Compute per-sample CHK-3.1B distributions on the subset
4. Derive thresholds per pre-locked rule
5. Outcome classification per sealed prereg
"""
import json
import csv
import os
import statistics
import math
from datetime import datetime, timezone

# Load subset
SUBSET_PATH = 'cardio_epic_chk31b_subset.txt'
with open(SUBSET_PATH) as f:
    subset = set(line.strip() for line in f if line.strip())
print(f"Loaded {len(subset)} CpGs in cardio-epic CHK-3.1B subset")

# Load manifest from VAL-106
manifest_path = '/home/claude/edear_working/VAL-106/cohort_manifest.json'
with open(manifest_path) as f:
    manifest = json.load(f)
samples = manifest['samples']
print(f"Loaded {len(samples)} samples from VAL-106 manifest")

# Per-sample CHK-3.1B
per_sample = []
for i, s in enumerate(samples):
    fp = s['local_path']
    n_subset_valid = 0
    n_extreme_subset = 0
    n_middle_subset = 0
    betas_subset = []
    
    with open(fp) as f:
        for line in f:
            parts = line.rstrip().split('\t')
            if len(parts) != 2:
                continue
            cpg, beta_str = parts
            if cpg not in subset:
                continue
            if beta_str in ('NA', '', 'NaN', 'null', 'NULL'):
                continue
            try:
                b = float(beta_str)
            except ValueError:
                continue
            if not (0.0 <= b <= 1.0):
                continue
            n_subset_valid += 1
            if b < 0.10 or b > 0.90:
                n_extreme_subset += 1
            if 0.40 <= b <= 0.60:
                n_middle_subset += 1
            betas_subset.append(b)
    
    coverage_pass = (n_subset_valid >= 7000)
    f_extreme_s = n_extreme_subset / n_subset_valid if n_subset_valid > 0 else 0.0
    f_middle_s = n_middle_subset / n_subset_valid if n_subset_valid > 0 else 0.0
    median_b = statistics.median(betas_subset) if betas_subset else 0.0
    
    per_sample.append({
        'project': s['project'], 'case_id': s['case_id'], 'sample_id': s['sample_id'],
        'n_subset_valid': n_subset_valid, 'subset_coverage_pass': coverage_pass,
        'f_extreme_subset': f_extreme_s, 'f_middle_subset': f_middle_s, 'median_beta_subset': median_b,
    })
    
    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(samples)}: latest f_extreme={f_extreme_s:.4f} f_middle={f_middle_s:.4f} valid={n_subset_valid}")

# Filter to coverage-passing samples
qc_kirc = [r for r in per_sample if r['project']=='TCGA-KIRC' and r['subset_coverage_pass']]
qc_prad = [r for r in per_sample if r['project']=='TCGA-PRAD' and r['subset_coverage_pass']]
print(f"\nCoverage-passing: KIRC {len(qc_kirc)}/160, PRAD {len(qc_prad)}/50")

# Coverage failure rate
n_total = len(per_sample)
n_cov_fail = sum(1 for r in per_sample if not r['subset_coverage_pass'])
cov_fail_rate = n_cov_fail / n_total
print(f"Subset coverage failure rate: {n_cov_fail}/{n_total} = {cov_fail_rate*100:.1f}%")

def cohort_stats(rows):
    fes = [r['f_extreme_subset'] for r in rows]
    fms = [r['f_middle_subset'] for r in rows]
    return {
        'n': len(rows),
        'f_extreme_mean': statistics.mean(fes) if fes else 0,
        'f_extreme_median': statistics.median(fes) if fes else 0,
        'f_extreme_sd': statistics.stdev(fes) if len(fes) > 1 else 0,
        'f_extreme_min': min(fes) if fes else 0,
        'f_extreme_max': max(fes) if fes else 0,
        'f_middle_mean': statistics.mean(fms) if fms else 0,
        'f_middle_median': statistics.median(fms) if fms else 0,
        'f_middle_sd': statistics.stdev(fms) if len(fms) > 1 else 0,
        'f_middle_min': min(fms) if fms else 0,
        'f_middle_max': max(fms) if fms else 0,
        'subset_valid_mean': statistics.mean([r['n_subset_valid'] for r in rows]) if rows else 0,
    }

kirc_stats = cohort_stats(qc_kirc)
prad_stats = cohort_stats(qc_prad)
combined_stats = cohort_stats(qc_kirc + qc_prad)

# Mann-Whitney
def mann_whitney_u(a, b):
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return {'U': 0, 'z': 0, 'p': 1.0, 'n1': n1, 'n2': n2}
    combined = [(v, 'a') for v in a] + [(v, 'b') for v in b]
    combined.sort(key=lambda x: x[0])
    ranks = [0.0]*len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined)-1 and combined[j+1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j)/2 + 1
        for k in range(i, j+1):
            ranks[k] = avg_rank
        i = j + 1
    R1 = sum(r for r, (_, g) in zip(ranks, combined) if g == 'a')
    U1 = R1 - n1*(n1+1)/2
    U = min(U1, n1*n2 - U1)
    mu = n1*n2/2
    sigma = math.sqrt(n1*n2*(n1+n2+1)/12)
    z = (U - mu) / sigma if sigma > 0 else 0
    from math import erf
    p = 2*(1 - 0.5*(1 + erf(abs(z)/math.sqrt(2))))
    return {'U': U, 'z': z, 'p': p, 'n1': n1, 'n2': n2}

mw_extreme = mann_whitney_u([r['f_extreme_subset'] for r in qc_kirc], [r['f_extreme_subset'] for r in qc_prad])
mw_middle = mann_whitney_u([r['f_middle_subset'] for r in qc_kirc], [r['f_middle_subset'] for r in qc_prad])

print(f"\nMann-Whitney KIRC vs PRAD:")
print(f"  f_extreme_subset: U={mw_extreme['U']:.1f}, z={mw_extreme['z']:.3f}, p={mw_extreme['p']:.4f}")
print(f"  f_middle_subset:  U={mw_middle['U']:.1f}, z={mw_middle['z']:.3f}, p={mw_middle['p']:.4f}")

# Threshold derivation per sealed prereg
def derive_thresholds(stats_dict):
    raw_extreme = stats_dict['f_extreme_mean'] - 2*stats_dict['f_extreme_sd']
    raw_middle = stats_dict['f_middle_mean'] + 2*stats_dict['f_middle_sd']
    extreme_pct = max(8.0, raw_extreme*100)
    middle_pct = min(15.0, raw_middle*100)
    extreme_thresh = math.floor(extreme_pct*2)/2/100
    middle_thresh = math.ceil(middle_pct*2)/2/100
    return {
        'raw_extreme_lower_bound': raw_extreme,
        'raw_middle_upper_bound': raw_middle,
        'extreme_threshold': extreme_thresh,
        'middle_threshold': middle_thresh,
        'extreme_threshold_pct': extreme_thresh*100,
        'middle_threshold_pct': middle_thresh*100,
    }

kirc_thresh = derive_thresholds(kirc_stats)
prad_thresh = derive_thresholds(prad_stats)
combined_thresh = derive_thresholds(combined_stats)

# Outcome classification per sealed prereg
# O4: ≥10% of samples fail coverage
# O3: extreme outside [10%, 80%] OR middle > 30%
# O2: KIRC vs PRAD divergence (p ≤ 0.05)
# O1: convergence + within bounds

if cov_fail_rate >= 0.10:
    outcome = "O4_SUBSET_COVERAGE_FAILURE"
elif (combined_stats['f_extreme_mean']*100 < 10 or combined_stats['f_extreme_mean']*100 > 80
      or combined_stats['f_middle_mean']*100 > 30):
    outcome = "O3_CALIBRATION_DEGENERATE"
elif mw_extreme['p'] <= 0.05:
    outcome = "O2_PLATFORM_DIVERGENCE_DOCUMENTED"
else:
    outcome = "O1_CHK_3_1B_THRESHOLD_ESTABLISHED"

print(f"\n========== OUTCOME: {outcome} ==========")
print(f"\nKIRC f_extreme_subset: mean={kirc_stats['f_extreme_mean']*100:.2f}% SD={kirc_stats['f_extreme_sd']*100:.2f}%")
print(f"KIRC f_middle_subset:  mean={kirc_stats['f_middle_mean']*100:.2f}% SD={kirc_stats['f_middle_sd']*100:.2f}%")
print(f"PRAD f_extreme_subset: mean={prad_stats['f_extreme_mean']*100:.2f}% SD={prad_stats['f_extreme_sd']*100:.2f}%")
print(f"PRAD f_middle_subset:  mean={prad_stats['f_middle_mean']*100:.2f}% SD={prad_stats['f_middle_sd']*100:.2f}%")
print(f"\nCombined CHK-3.1B thresholds for cardio-epic on TCGA HM450K sesame Level 3:")
print(f"  extreme_threshold_B: {combined_thresh['extreme_threshold_pct']:.1f}% (sample passes if f_extreme_subset >= this)")
print(f"  middle_threshold_B:  {combined_thresh['middle_threshold_pct']:.1f}% (sample passes if f_middle_subset <= this)")

results = {
    'val_id': 'VAL-107',
    'prereg_sha256': 'b58ce4dbd422198c7cbd6e7d1ee1cdbed86a758afc204189f8a9e070fd700d82',
    'subset_sha256': '5a00e29ace75daae5a5bf7e3cfca26c16aa6dbd92750d16ebeaba4e874c48511',
    'sealed_at': '2026-04-28T22:19:26Z',
    'executed_at': datetime.now(timezone.utc).isoformat(),
    'rng_seed': 20260428,
    'subset_size': len(subset),
    'subset_composition': {
        'loyfer_25_tile_full': 6105,
        'unilife_19cell': 1906,
        'salas_idol_450k': 350,
        'total_unique_after_dedup': len(subset),
    },
    'coverage_failure_rate': cov_fail_rate,
    'cohort_summary': {
        'TCGA-KIRC': {'total': 160, 'coverage_pass': len(qc_kirc), 'stats': kirc_stats, 'thresholds': kirc_thresh},
        'TCGA-PRAD': {'total': 50, 'coverage_pass': len(qc_prad), 'stats': prad_stats, 'thresholds': prad_thresh},
        'combined': {'coverage_pass': len(qc_kirc)+len(qc_prad), 'stats': combined_stats, 'thresholds': combined_thresh},
    },
    'mann_whitney_kirc_vs_prad': {'f_extreme_subset': mw_extreme, 'f_middle_subset': mw_middle},
    'outcome': outcome,
    'cardio_epic_chk_3_1b_thresholds_for_TCGA_HM450K_sesame_level3': {
        'extreme_threshold': combined_thresh['extreme_threshold'],
        'middle_threshold': combined_thresh['middle_threshold'],
        'extreme_threshold_pct': combined_thresh['extreme_threshold_pct'],
        'middle_threshold_pct': combined_thresh['middle_threshold_pct'],
        'subset_coverage_minimum_n_valid': 7000,
        'pass_criterion': 'sample passes CHK-3.1B for cardio-epic on TCGA HM450K sesame Level 3 iff (n_subset_valid >= 7000) AND (f_extreme_subset >= extreme_threshold) AND (f_middle_subset <= middle_threshold)',
    },
}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

with open('per_sample.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['project','case_id','sample_id','n_subset_valid','subset_coverage_pass','f_extreme_subset','f_middle_subset','median_beta_subset'])
    w.writeheader()
    for r in per_sample:
        w.writerow(r)

print(f"\nWrote results.json + per_sample.csv")
