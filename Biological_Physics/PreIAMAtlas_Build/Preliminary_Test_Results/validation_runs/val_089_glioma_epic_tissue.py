"""
VAL-089 glioma-epic — Tumor-tissue arm — direct architecture A-score on brain tissue
Card: glioma-epic v0.1
Cohort: GSE60274 (Lai 2015) — 60 primary surgical GBM + 4 GBM-with-paired-sphere primary +
        4 recurrent GBM + 4 cultured spheres + 5 non-tumor brain (NTB) controls. Total 77 samples on 450K.
Healthy reference: 5 NTB controls (Lobectomy + Craniotomy for Epilepsy specimens), ON-STUDY.
                   Per CHK-3.2 NO cross-platform / cross-cohort baseline confound — internal controls.

H_min: This VAL reports BOTH H_min normalizations:
  - H_min(immune) = 0.838889 — for direct comparability with VAL-088 blood result
  - H_min(terminal) = 0.7728 — the brain-tissue-correct normalization per GAPE class assignment

Panel: Xu-538 (SHA ada672960...) — immune-class panel applied to brain tissue.
       NOTE: panel was trained on whole blood for immune class; using on brain tissue measures
       "what those CpGs read in this tissue" not "the tissue's intrinsic architecture entropy."
       This is an architecture-readout-on-non-native-tissue analysis. Direction of effect remains informative;
       absolute magnitude must be interpreted with this scope in mind.

Per CCL-032 diagnostic order:
  1. Data integrity (CHK-3.1, CHK-3.3, CHK-3.5)
  2. Biology consistency (Issue 002 prediction: GBM tumor tissue ΔA = +0.217 at 5-substrate cfDNA scope;
                          v1 single-substrate methyl-only buffy-coat-panel-on-tumor-tissue is different scope)
  3. Framework finding

Stratifications (declared):
  - GBM primary surgical (n=64 = 60 + 4-paired-with-sphere)
  - GBM recurrent (n=4)
  - GBM spheres (cultured) (n=4) — supplementary
  - NTB controls (n=5)

CRITICAL CAVEAT documented per CHK-1.5 substrate-scope:
  Issue 002 ΔA = +0.217 figure is 5-substrate cfDNA TUMOR TISSUE prediction at L2/L3 platform.
  This VAL measures single-substrate methyl-only on Xu-538-immune-panel applied to brain tumor tissue.
  Direction-of-effect is the primary inference; absolute magnitude is reported with explicit substrate-scope caveat.
"""

import gzip, math, statistics, json, os
from math import erf, sqrt

H_MIN_IMMUNE = 0.838889
H_MIN_TERMINAL = 0.7728
QC_MIN = 400  # 450K platform expected to have full Xu-538 coverage; threshold is conservative
RNG_SEED = 20260425

PANEL_PATH = '/mnt/user-data/outputs/crc_tissue_arm_github_ready/xu538_panel.json'
MATRIX_PATH = '/home/claude/glioma_work/GSE60274_series_matrix.txt.gz'
MANIFEST_PATH = '/mnt/user-data/outputs/cookbook_v2.1/glioma-epic/GSE60274_manifest.json'
OUTPUT_PATH = '/mnt/user-data/outputs/cookbook_v2.1/glioma-epic/VAL-089_results.json'


def shannon(b):
    if b is None or b <= 0 or b >= 1:
        return 0.0
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)


def a_score(d, h_min):
    if not d:
        return None
    return sum(shannon(b) / h_min for b in d.values()) / len(d)


def unpaired_d_with_ci(at, an):
    if len(at) < 2 or len(an) < 2:
        return None, [None, None]
    mt, st = statistics.mean(at), statistics.stdev(at)
    mn, sn = statistics.mean(an), statistics.stdev(an)
    pl = sqrt(((len(at) - 1) * st ** 2 + (len(an) - 1) * sn ** 2) / (len(at) + len(an) - 2))
    if pl == 0:
        return None, [None, None]
    d = (mt - mn) / pl
    se = sqrt((len(at) + len(an)) / (len(at) * len(an)) + d ** 2 / (2 * (len(at) + len(an))))
    return d, [d - 1.96 * se, d + 1.96 * se]


# ============================================================
# Load panel + manifest
# ============================================================
with open(PANEL_PATH) as f:
    panel = json.load(f)
xu538 = set(panel['cpgs'])

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)
gsm_to_meta = {m['gsm']: m for m in manifest}

# ============================================================
# Categorize samples per VAL design
# ============================================================
cats = {'GBM_primary': [], 'GBM_recurrent': [], 'GBM_spheres': [], 'NTB_healthy': []}
for m in manifest:
    ds = m['disease_state']
    src = m['source']
    if ds == 'Non-tumor brain':
        cats['NTB_healthy'].append(m['gsm'])
    elif 'cultured glioma spheres' in src:
        cats['GBM_spheres'].append(m['gsm'])
    elif 'recurrent' in src.lower():
        cats['GBM_recurrent'].append(m['gsm'])
    else:
        # All Surgical resection GBM (including 2207/2540/2669/2683 paired primaries)
        cats['GBM_primary'].append(m['gsm'])

print("Sample categories:")
for c, lst in cats.items():
    print(f"  {c}: n={len(lst)}")

# ============================================================
# Stream-parse GSE60274 series matrix (β values inline)
# ============================================================
print("\nParsing GSE60274 series matrix (β values inline)...")
sample_betas = {gsm: {} for gsms in cats.values() for gsm in gsms}
all_distribution_for_chk_3_1 = []

with gzip.open(MATRIX_PATH, 'rt') as f:
    in_table = False
    sample_cols = []
    cpgs_in_panel = 0
    cpgs_total = 0
    for line in f:
        line = line.rstrip()
        if line.startswith('!series_matrix_table_begin'):
            in_table = True
            continue
        if line.startswith('!series_matrix_table_end'):
            break
        if not in_table:
            continue

        parts = line.split('\t')
        if parts[0].startswith('"ID_REF"') or parts[0] == 'ID_REF':
            sample_cols = [p.strip('"') for p in parts[1:]]
            continue

        cpg = parts[0].strip('"')
        cpgs_total += 1
        if cpg in xu538:
            cpgs_in_panel += 1
            for i, val in enumerate(parts[1:]):
                if i >= len(sample_cols): break
                gsm = sample_cols[i]
                if gsm not in sample_betas: continue
                try:
                    b = float(val)
                except (ValueError, TypeError):
                    continue
                if 0 < b < 1:
                    sample_betas[gsm][cpg] = b
        # CHK-3.1 distribution sample (every 1000th probe, first sample col)
        if cpgs_total % 1000 == 0 and sample_cols:
            try:
                b = float(parts[1])
                if 0 < b < 1:
                    all_distribution_for_chk_3_1.append(b)
            except (ValueError, TypeError):
                pass

print(f"CpGs scanned: {cpgs_total}")
print(f"Xu-538 panel CpGs found in matrix: {cpgs_in_panel}/{len(xu538)} ({100*cpgs_in_panel/len(xu538):.1f}%)")

# ============================================================
# CHK-3.1 β-distribution
# ============================================================
print("\n=== CHK-3.1 β distribution ===")
n = len(all_distribution_for_chk_3_1)
extremes = sum(1 for b in all_distribution_for_chk_3_1 if b < 0.1 or b > 0.9)
mid = sum(1 for b in all_distribution_for_chk_3_1 if 0.4 < b < 0.6)
median = sorted(all_distribution_for_chk_3_1)[n // 2] if n else None
print(f"  n={n}, extremes {100*extremes/n:.1f}%, mid {100*mid/n:.1f}%, median {median:.3f}")
chk_3_1_pass = (extremes/n > 0.20) and (mid/n < 0.40)
print(f"  CHK-3.1: {'PASS' if chk_3_1_pass else 'FLAG'}")

# ============================================================
# CHK-3.3 panel coverage QC
# ============================================================
print("\n=== CHK-3.3 panel coverage ===")
qc_pass_lists = {c: [] for c in cats}
qc_fail = []
for c, gsms in cats.items():
    for gsm in gsms:
        n_cpgs = len(sample_betas[gsm])
        if n_cpgs >= QC_MIN:
            qc_pass_lists[c].append(gsm)
        else:
            qc_fail.append((c, gsm, n_cpgs))
for c, lst in qc_pass_lists.items():
    if cats[c]:
        print(f"  {c}: {len(lst)}/{len(cats[c])} QC pass")
if qc_fail:
    print(f"  QC fail (first 5): {qc_fail[:5]}")
mean_cov = statistics.mean([len(sample_betas[g]) for lst in qc_pass_lists.values() for g in lst])
print(f"  Mean Xu-538 coverage per QC-passed sample: {mean_cov:.0f} of {len(xu538)} ({100*mean_cov/len(xu538):.1f}%)")

# ============================================================
# Compute A-scores under both H_min normalizations
# ============================================================
A_immune = {}
A_terminal = {}
for lst in qc_pass_lists.values():
    for gsm in lst:
        A_immune[gsm] = a_score(sample_betas[gsm], H_MIN_IMMUNE)
        A_terminal[gsm] = a_score(sample_betas[gsm], H_MIN_TERMINAL)

# ============================================================
# CHK-3.5 saturation
# ============================================================
A_CEILING_IMMUNE = 1.1921
A_CEILING_TERMINAL = 1.1921 * (H_MIN_IMMUNE / H_MIN_TERMINAL)
print(f"\n=== CHK-3.5 saturation ===")
n_sat_immune = sum(1 for v in A_immune.values() if v >= A_CEILING_IMMUNE - 0.005)
n_sat_terminal = sum(1 for v in A_terminal.values() if v >= A_CEILING_TERMINAL - 0.005)
print(f"  H_min=immune ceiling={A_CEILING_IMMUNE}; saturated: {n_sat_immune}/{len(A_immune)}; max: {max(A_immune.values()):.4f}")
print(f"  H_min=terminal ceiling={A_CEILING_TERMINAL:.4f}; saturated: {n_sat_terminal}/{len(A_terminal)}; max: {max(A_terminal.values()):.4f}")

# ============================================================
# Group statistics + healthy comparison
# ============================================================
print("\n=== Group statistics — H_min(terminal)=0.7728 ===")
out_strat = {}
healthy_A_terminal = [A_terminal[g] for g in qc_pass_lists['NTB_healthy']]
print(f"  NTB_healthy (n={len(healthy_A_terminal)}): mean A = {statistics.mean(healthy_A_terminal):.4f}, "
      f"SD = {statistics.stdev(healthy_A_terminal) if len(healthy_A_terminal)>=2 else float('nan'):.4f}")
out_strat['NTB_healthy_terminal'] = {'n': len(healthy_A_terminal), 'mean': statistics.mean(healthy_A_terminal),
                                       'sd': statistics.stdev(healthy_A_terminal) if len(healthy_A_terminal)>=2 else None}

for c in ['GBM_primary', 'GBM_recurrent', 'GBM_spheres']:
    Gs = [A_terminal[g] for g in qc_pass_lists[c]]
    if not Gs: continue
    m = statistics.mean(Gs)
    s = statistics.stdev(Gs) if len(Gs) >= 2 else None
    print(f"  {c} (n={len(Gs)}): mean A = {m:.4f}, SD = {s if s else 'N/A':.4f}")
    if s and len(healthy_A_terminal) >= 2:
        d, ci = unpaired_d_with_ci(Gs, healthy_A_terminal)
        delta = m - statistics.mean(healthy_A_terminal)
        print(f"    ΔA vs NTB = {delta:+.4f}; Cohen's d = {d:+.3f} 95%CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        out_strat[f'{c}_terminal'] = {'n': len(Gs), 'mean': m, 'sd': s, 'delta_A': delta, 'cohen_d': d, 'ci_95': ci}

print("\n=== Group statistics — H_min(immune)=0.838889 (for VAL-088 comparability) ===")
healthy_A_immune = [A_immune[g] for g in qc_pass_lists['NTB_healthy']]
print(f"  NTB_healthy (n={len(healthy_A_immune)}): mean A = {statistics.mean(healthy_A_immune):.4f}, "
      f"SD = {statistics.stdev(healthy_A_immune) if len(healthy_A_immune)>=2 else float('nan'):.4f}")
out_strat['NTB_healthy_immune'] = {'n': len(healthy_A_immune), 'mean': statistics.mean(healthy_A_immune),
                                     'sd': statistics.stdev(healthy_A_immune) if len(healthy_A_immune)>=2 else None}

for c in ['GBM_primary', 'GBM_recurrent', 'GBM_spheres']:
    Gs = [A_immune[g] for g in qc_pass_lists[c]]
    if not Gs: continue
    m = statistics.mean(Gs)
    s = statistics.stdev(Gs) if len(Gs) >= 2 else None
    print(f"  {c} (n={len(Gs)}): mean A = {m:.4f}, SD = {s if s else 'N/A':.4f}")
    if s and len(healthy_A_immune) >= 2:
        d, ci = unpaired_d_with_ci(Gs, healthy_A_immune)
        delta = m - statistics.mean(healthy_A_immune)
        print(f"    ΔA vs NTB = {delta:+.4f}; Cohen's d = {d:+.3f} 95%CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        out_strat[f'{c}_immune'] = {'n': len(Gs), 'mean': m, 'sd': s, 'delta_A': delta, 'cohen_d': d, 'ci_95': ci}

# ============================================================
# Save results
# ============================================================
results = {
    'val_id': 'VAL-089',
    'card': 'glioma-epic',
    'arm': 'tumor_tissue',
    'date': '2026-04-25',
    'design': 'Direct tissue architecture A-score on GSE60274 (Lai 2015) 60 primary GBM + 4 paired primary + 4 recurrent GBM + 4 cultured spheres + 5 non-tumor brain (NTB) controls. 450K platform with on-study NTB healthy reference.',
    'panel': 'Xu-538 immune (applied to brain tissue — non-native specimen for the panel; direction-of-effect primary inference)',
    'panel_sha256': 'ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6',
    'h_min_immune': H_MIN_IMMUNE,
    'h_min_terminal': H_MIN_TERMINAL,
    'rng_seed': RNG_SEED,
    'platform': 'IlluminaHumanMethylation450 (GPL13534)',
    'panel_coverage_pct': 100*mean_cov/len(xu538),
    'chk_3_1_pass': chk_3_1_pass,
    'chk_3_1_extremes_pct': 100*extremes/n,
    'chk_3_1_mid_pct': 100*mid/n,
    'chk_3_3_qc_pass': {c: len(qc_pass_lists[c]) for c in cats},
    'chk_3_3_qc_total': {c: len(cats[c]) for c in cats},
    'chk_3_5_n_saturated_immune_norm': n_sat_immune,
    'chk_3_5_n_saturated_terminal_norm': n_sat_terminal,
    'chk_3_5_max_A_immune': max(A_immune.values()),
    'chk_3_5_max_A_terminal': max(A_terminal.values()),
    'stratified': out_strat,
    'per_sample_A_immune': A_immune,
    'per_sample_A_terminal': A_terminal,
    'caveats': [
        'Xu-538 panel was trained on whole-blood IMMUNE class; applying to brain tissue measures what those CpGs read in non-native tissue, not the tissue\'s intrinsic terminal-class architecture',
        'Direction-of-effect is the primary inference; absolute magnitudes depend on H_min normalization choice',
        'NTB controls are older (median 75) than typical GBM age (median 55); age-adjustment not applied',
        'Issue 002 ΔA = +0.217 GBM-tumor figure is 5-substrate cfDNA L2/L3 prediction; v1 single-substrate methyl-only readings are different scope (CHK-1.5)',
        'NTB controls are surgical specimens from epilepsy / lobectomy patients — "non-tumor brain" but not pristine healthy brain',
    ],
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults written to {OUTPUT_PATH}")
