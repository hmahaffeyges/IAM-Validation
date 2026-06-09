"""
VAL-106 — Calibration VAL: TCGA HM450K sesame Level 3 platform CHK-3.1 threshold establishment

Pre-registration SHA-256: 0330a3c6c76c8874ba5027e88670ab60307dc322fa4cb9186ffac06d6ec4117a
Sealed: 2026-04-28T21:53:20Z (before β-value access)
RNG seed: 20260428

Methodology (frozen by sealed prereg):
1. Compute per-sample f_extreme (β<0.10 or >0.90) and f_middle (0.40-0.60) over autosomal CpGs
2. Aggregate per-cohort and cross-cohort distributions
3. Mann-Whitney U test for KIRC vs PRAD f_extreme convergence
4. Derive thresholds: extreme_threshold = max(15.0, mean - 2*SD); middle_threshold = min(12.0, mean + 2*SD)
5. Outcome classification per sealed prereg
"""
import json
import csv
import os
import statistics
import math
from datetime import datetime, timezone

# QC threshold per cookbook standard
MIN_VALID_BETAS = 400_000

# Load manifest
with open('cohort_manifest.json') as f:
    manifest = json.load(f)
samples = manifest['samples']
print(f"VAL-106: Loaded manifest with {len(samples)} samples")
print(f"  Pre-reg SHA-256: {manifest['prereg_sha256']}")
print(f"  Sealed at: {manifest['sealed_at']}")
print(f"  Started: {datetime.now(timezone.utc).isoformat()}")

# Compute CHK-3.1 stats for every sample
per_sample_results = []
for i, s in enumerate(samples):
    fp = s['local_path']
    n_valid = 0
    n_extreme = 0
    n_middle = 0
    betas = []
    with open(fp) as f:
        for line in f:
            parts = line.rstrip().split('\t')
            if len(parts) != 2:
                continue
            cpg, beta_str = parts
            if beta_str in ('NA', '', 'NaN', 'null', 'NULL'):
                continue
            try:
                b = float(beta_str)
            except ValueError:
                continue
            if not (0.0 <= b <= 1.0):
                continue
            n_valid += 1
            if b < 0.10 or b > 0.90:
                n_extreme += 1
            if 0.40 <= b <= 0.60:
                n_middle += 1
            betas.append(b)
    
    qc_pass = (n_valid >= MIN_VALID_BETAS)
    f_extreme = n_extreme / n_valid if n_valid > 0 else 0.0
    f_middle = n_middle / n_valid if n_valid > 0 else 0.0
    median_b = statistics.median(betas) if betas else 0.0
    
    per_sample_results.append({
        'project': s['project'], 'case_id': s['case_id'], 'sample_id': s['sample_id'],
        'file_id': s['file_id'], 'sha256': s['sha256'],
        'n_valid': n_valid, 'qc_pass': qc_pass,
        'f_extreme': f_extreme, 'f_middle': f_middle, 'median_beta': median_b,
    })
    
    if (i + 1) % 25 == 0:
        print(f"  Processed {i+1}/{len(samples)}: latest f_extreme={f_extreme:.4f} f_middle={f_middle:.4f} valid={n_valid:,}")

print(f"\nProcessing complete. {len(per_sample_results)} samples analyzed.")

# Filter to QC-passing samples
qc_kirc = [r for r in per_sample_results if r['project'] == 'TCGA-KIRC' and r['qc_pass']]
qc_prad = [r for r in per_sample_results if r['project'] == 'TCGA-PRAD' and r['qc_pass']]
print(f"\nQC-passing: KIRC {len(qc_kirc)}/160, PRAD {len(qc_prad)}/50")

# Per-cohort statistics
def cohort_stats(rows):
    fes = [r['f_extreme'] for r in rows]
    fms = [r['f_middle'] for r in rows]
    return {
        'n': len(rows),
        'f_extreme_mean': statistics.mean(fes),
        'f_extreme_median': statistics.median(fes),
        'f_extreme_sd': statistics.stdev(fes) if len(fes) > 1 else 0,
        'f_extreme_min': min(fes),
        'f_extreme_max': max(fes),
        'f_middle_mean': statistics.mean(fms),
        'f_middle_median': statistics.median(fms),
        'f_middle_sd': statistics.stdev(fms) if len(fms) > 1 else 0,
        'f_middle_min': min(fms),
        'f_middle_max': max(fms),
    }

kirc_stats = cohort_stats(qc_kirc)
prad_stats = cohort_stats(qc_prad)
combined_stats = cohort_stats(qc_kirc + qc_prad)

# Mann-Whitney U test (manual implementation, no scipy required)
def mann_whitney_u(a, b):
    n1, n2 = len(a), len(b)
    combined = [(v, 'a') for v in a] + [(v, 'b') for v in b]
    combined.sort(key=lambda x: x[0])
    # Assign ranks (handle ties via average rank)
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) - 1 and combined[j+1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based
        for k in range(i, j+1):
            ranks[k] = avg_rank
        i = j + 1
    R1 = sum(r for r, (_, group) in zip(ranks, combined) if group == 'a')
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    U = min(U1, U2)
    # Normal approximation for p-value (n1, n2 large enough)
    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U - mu) / sigma if sigma > 0 else 0
    # Two-tailed p
    from math import erf
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / math.sqrt(2))))
    return {'U': U, 'z': z, 'p': p, 'n1': n1, 'n2': n2}

mw_extreme = mann_whitney_u([r['f_extreme'] for r in qc_kirc], [r['f_extreme'] for r in qc_prad])
mw_middle = mann_whitney_u([r['f_middle'] for r in qc_kirc], [r['f_middle'] for r in qc_prad])

print(f"\nMann-Whitney U test (KIRC vs PRAD):")
print(f"  f_extreme: U={mw_extreme['U']:.1f}, z={mw_extreme['z']:.3f}, p={mw_extreme['p']:.4f}")
print(f"  f_middle:  U={mw_middle['U']:.1f}, z={mw_middle['z']:.3f}, p={mw_middle['p']:.4f}")

# Pre-locked threshold derivation rule
# extreme_threshold = max(15.0, mean - 2*SD), rounded down to nearest 0.5%
# middle_threshold = min(12.0, mean + 2*SD), rounded up to nearest 0.5%
def derive_thresholds(stats_dict):
    raw_extreme = stats_dict['f_extreme_mean'] - 2 * stats_dict['f_extreme_sd']
    raw_middle = stats_dict['f_middle_mean'] + 2 * stats_dict['f_middle_sd']
    # Apply floor and ceiling, then round
    extreme_pct = max(15.0, raw_extreme * 100)
    middle_pct = min(12.0, raw_middle * 100)
    # Round extreme down to nearest 0.5
    extreme_thresh = math.floor(extreme_pct * 2) / 2 / 100
    # Round middle up to nearest 0.5
    middle_thresh = math.ceil(middle_pct * 2) / 2 / 100
    return {
        'raw_extreme_lower_bound': raw_extreme,
        'raw_middle_upper_bound': raw_middle,
        'extreme_threshold': extreme_thresh,
        'middle_threshold': middle_thresh,
        'extreme_threshold_pct': extreme_thresh * 100,
        'middle_threshold_pct': middle_thresh * 100,
    }

kirc_thresh = derive_thresholds(kirc_stats)
prad_thresh = derive_thresholds(prad_stats)
combined_thresh = derive_thresholds(combined_stats)

# Outcome classification per sealed prereg
# O1_PLATFORM_THRESHOLD_ESTABLISHED: convergence (p > 0.05) AND thresholds within reasonable bounds (extreme >= 18%, middle <= 11%)
# O2_PLATFORM_DIVERGENCE_DOCUMENTED: divergence (p <= 0.05)
# O3_CALIBRATION_DEGENERATE: extreme outside [18, 35] OR middle > 15
# O4_CALIBRATION_DATA_UNAVAILABLE: not applicable here (data acquired)

extreme_pct_check = combined_thresh['extreme_threshold_pct']
middle_pct_check = combined_thresh['middle_threshold_pct']

degenerate = False
for stats_dict in [kirc_stats, prad_stats]:
    if (stats_dict['f_extreme_mean']*100 < 18 or stats_dict['f_extreme_mean']*100 > 35
        or stats_dict['f_middle_mean']*100 > 15):
        degenerate = True
        break

if degenerate:
    outcome = "O3_CALIBRATION_DEGENERATE"
elif mw_extreme['p'] <= 0.05:
    outcome = "O2_PLATFORM_DIVERGENCE_DOCUMENTED"
else:
    outcome = "O1_PLATFORM_THRESHOLD_ESTABLISHED"

print(f"\n========== OUTCOME: {outcome} ==========")
print(f"\nKIRC f_extreme: mean={kirc_stats['f_extreme_mean']*100:.2f}% SD={kirc_stats['f_extreme_sd']*100:.2f}%")
print(f"KIRC f_middle:  mean={kirc_stats['f_middle_mean']*100:.2f}% SD={kirc_stats['f_middle_sd']*100:.2f}%")
print(f"PRAD f_extreme: mean={prad_stats['f_extreme_mean']*100:.2f}% SD={prad_stats['f_extreme_sd']*100:.2f}%")
print(f"PRAD f_middle:  mean={prad_stats['f_middle_mean']*100:.2f}% SD={prad_stats['f_middle_sd']*100:.2f}%")
print(f"\nDerived thresholds (combined):")
print(f"  extreme_threshold: {combined_thresh['extreme_threshold_pct']:.1f}% (lower bound, sample passes if f_extreme >= this)")
print(f"  middle_threshold:  {combined_thresh['middle_threshold_pct']:.1f}% (upper bound, sample passes if f_middle <= this)")
print(f"\nDerived thresholds (KIRC alone): extreme>={kirc_thresh['extreme_threshold_pct']:.1f}%, middle<={kirc_thresh['middle_threshold_pct']:.1f}%")
print(f"Derived thresholds (PRAD alone): extreme>={prad_thresh['extreme_threshold_pct']:.1f}%, middle<={prad_thresh['middle_threshold_pct']:.1f}%")

# Save results
results = {
    'val_id': 'VAL-106',
    'prereg_sha256': '0330a3c6c76c8874ba5027e88670ab60307dc322fa4cb9186ffac06d6ec4117a',
    'sealed_at': '2026-04-28T21:53:20Z',
    'executed_at': datetime.now(timezone.utc).isoformat(),
    'rng_seed': 20260428,
    'cohort_summary': {
        'TCGA-KIRC': {'total_in_manifest': 160, 'qc_passed': len(qc_kirc), 'stats': kirc_stats, 'thresholds': kirc_thresh},
        'TCGA-PRAD': {'total_in_manifest': 50, 'qc_passed': len(qc_prad), 'stats': prad_stats, 'thresholds': prad_thresh},
        'combined': {'qc_passed': len(qc_kirc) + len(qc_prad), 'stats': combined_stats, 'thresholds': combined_thresh},
    },
    'mann_whitney_kirc_vs_prad': {
        'f_extreme': mw_extreme,
        'f_middle': mw_middle,
    },
    'outcome': outcome,
    'tcga_hm450k_sesame_level3_platform_chk_3_1_thresholds': {
        'extreme_threshold': combined_thresh['extreme_threshold'],
        'middle_threshold': combined_thresh['middle_threshold'],
        'extreme_threshold_pct': combined_thresh['extreme_threshold_pct'],
        'middle_threshold_pct': combined_thresh['middle_threshold_pct'],
        'pass_criterion': 'sample passes CHK-3.1 if (f_extreme >= extreme_threshold) AND (f_middle <= middle_threshold)',
    },
    'qc_threshold_min_valid_betas': MIN_VALID_BETAS,
}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Per-sample CSV
with open('per_sample.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['project','case_id','sample_id','file_id','sha256','n_valid','qc_pass','f_extreme','f_middle','median_beta'])
    w.writeheader()
    for r in per_sample_results:
        w.writerow(r)

print(f"\nWrote results.json + per_sample.csv")
print(f"VAL-106 complete: {datetime.now(timezone.utc).isoformat()}")
