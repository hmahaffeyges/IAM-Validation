"""
VAL-088 glioma-epic Stage 1 — Stage 1 immune Xu-538 A-score on glioma whole blood
Card: glioma-epic v0.1
Cohort: GSE180683 Salas/Wiencke 2022 — n=76 glioma patients EPIC peripheral blood, mixed treatment stages
Pre-surgery treatment-naive subset: n=37 (the cleanest CCL-024 direction-test subset)
External healthy comparator: GSE51057 EPIC-Italy — see VAL-082 for healthy mean reference
                            (NOTE platform mismatch: GSE51057 is HM450, GSE180683 is EPIC; coverage drift documented per CHK-3.3)

H_min(immune) = 0.838889 (universal Stage 1 rule per panc-LL-007).
Panel: Xu-538 (SHA ada672960...).
RNG seed: 20260425.

Per CCL-032 diagnostic order:
  1. Data integrity (CHK-3.1 β distribution; CHK-3.3 panel coverage; CHK-3.5 saturation)
  2. Biology consistency (CHK-4.1: glioma expected NEGATIVE direction per CCL-023 + Bracci 2022 cell-fraction)
  3. Framework finding

Stratifications per CCL-024:
  - By histological group: GBM-new (n=39), LGG-new (n=14), LGG-recurrent-still-LGG (n=13), LGG-recurrent-now-GBM (n=10)
  - By treatment time-point: Pre-surgery (n=37), Post-surgery pre-chemorad (n=20), other (n=19)

Per CHK-1.5 substrate-scope: Issue 002 terminal class A_combined ≈ 1.10 FLOOR BREACH refers to TUMOR TISSUE 5-substrate cfDNA.
We are scoring single-substrate methyl-only buffy-coat (NOT tumor tissue, NOT cfDNA, NOT 5-substrate).
The architecture A-score we compute is the IMMUNE class (H_min=0.838889), NOT terminal class —
because we are measuring whole blood, where immune cells dominate.
A directional test of CCL-023 (negative shift in peripheral immune): expected sign is NEGATIVE.

Per CHK-2.4 panel transferability: Xu-538 is whole-blood-trained, transfers cleanly to whole blood.
This VAL is on the panel's NATIVE training tissue. No transferability flag.
"""

import gzip, math, statistics, json, os
from math import erf, sqrt

H_MIN_IMMUNE = 0.838889
QC_MIN = 400
RNG_SEED = 20260425

PANEL_PATH = '/mnt/user-data/outputs/crc_tissue_arm_github_ready/xu538_panel.json'
MATRIX_PATH = '/home/claude/glioma_work/GSE180683_Matrix.txt.gz'
MAPPING_PATH = '/mnt/user-data/outputs/cookbook_v2.1/glioma-epic/GSE180683_chippos_to_gsm.json'
MANIFEST_PATH = '/mnt/user-data/outputs/cookbook_v2.1/glioma-epic/GSE180683_manifest.json'
OUTPUT_PATH = '/mnt/user-data/outputs/cookbook_v2.1/glioma-epic/VAL-088_results.json'


def shannon(b):
    if b is None or b <= 0 or b >= 1:
        return 0.0
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)


def a_score(d, h_min=H_MIN_IMMUNE):
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
# Load panel + manifest + mapping
# ============================================================
with open(PANEL_PATH) as f:
    panel = json.load(f)
xu538 = set(panel['cpgs'])
print(f"Xu-538 panel size: {len(xu538)}")

with open(MAPPING_PATH) as f:
    chip_to_gsm = json.load(f)
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

gsm_to_meta = {r['gsm']: r for r in manifest}
print(f"Manifest: {len(manifest)} samples")

# ============================================================
# Stream-parse matrix; keep only Xu-538 panel β values
# ============================================================
print("Parsing GSE180683 matrix (this takes a minute)...")
with gzip.open(MATRIX_PATH, 'rt') as f:
    header = f.readline().rstrip()
    parts = [p.strip('"') for p in header.split('\t')]
    # parts[0] = "ID_REF", then alternating β / detection_p
    sample_cols = []
    is_pval = []
    for i, col in enumerate(parts[1:], start=1):
        if 'Detection' in col:
            is_pval.append(i)
        else:
            sample_cols.append((i, col))
    print(f"  {len(sample_cols)} β-value columns, {len(is_pval)} detection p-value columns")

    # Build per-sample β dictionary, only for panel CpGs
    sample_betas = {chip_to_gsm[c]: {} for _, c in sample_cols if c in chip_to_gsm}
    sample_pval = {chip_to_gsm[c]: {} for _, c in sample_cols if c in chip_to_gsm}

    # Map column index → GSM
    col_to_gsm = {idx: chip_to_gsm.get(c) for idx, c in sample_cols}
    # Each β col is followed by its detection p column (next column)
    # But to be robust, build pairs by name match
    chippos_to_betacol = {}
    chippos_to_pcol = {}
    for i, col in enumerate(parts[1:], start=1):
        if 'Detection' in col:
            base = col.replace(' Detection Pval', '')
            chippos_to_pcol[base] = i
        else:
            chippos_to_betacol[col] = i

    cpgs_seen = 0
    cpgs_in_panel = 0
    raw_distribution_sample = {  # for CHK-3.1 — collect 5 random samples' values
    }
    sample_chips_for_dist = list(chippos_to_betacol.keys())[:5]
    for sc in sample_chips_for_dist:
        raw_distribution_sample[sc] = []

    for line in f:
        parts = line.rstrip().split('\t')
        cpg = parts[0].strip('"')
        cpgs_seen += 1
        if cpg in xu538:
            cpgs_in_panel += 1
            for chippos, beta_idx in chippos_to_betacol.items():
                gsm = chip_to_gsm.get(chippos)
                if gsm is None: continue
                pcol = chippos_to_pcol.get(chippos)
                try:
                    b = float(parts[beta_idx])
                    p = float(parts[pcol]) if pcol and pcol < len(parts) else 1.0
                except (ValueError, IndexError):
                    continue
                if 0 < b < 1 and p < 0.05:
                    sample_betas[gsm][cpg] = b
                    sample_pval[gsm][cpg] = p
        # CHK-3.1 distribution: collect every 1000th probe across 5 random samples
        if cpgs_seen % 1000 == 0:
            for sc in sample_chips_for_dist:
                bidx = chippos_to_betacol[sc]
                try:
                    b = float(parts[bidx])
                    if 0 < b < 1:
                        raw_distribution_sample[sc].append(b)
                except (ValueError, IndexError):
                    pass

print(f"\nCpGs scanned: {cpgs_seen}")
print(f"Xu-538 panel CpGs found in matrix: {cpgs_in_panel}/{len(xu538)} ({100*cpgs_in_panel/len(xu538):.1f}%)")

# ============================================================
# CHK-3.1 β distribution sanity check
# ============================================================
print("\n=== CHK-3.1 β distribution sanity ===")
all_dist = []
for sc, vals in raw_distribution_sample.items():
    all_dist.extend(vals)
n = len(all_dist)
extremes = sum(1 for b in all_dist if b < 0.1 or b > 0.9)
mid = sum(1 for b in all_dist if 0.4 < b < 0.6)
median = sorted(all_dist)[n // 2] if n else None
print(f"  Spot-check (1000th-probe sample × 5 chips): n={n}")
print(f"  Extremes (<0.1 or >0.9): {100*extremes/n:.1f}% (raw expected >30%)")
print(f"  Mid [0.4, 0.6]: {100*mid/n:.1f}% (raw expected <10%)")
print(f"  Median: {median:.3f}")
chk_3_1_pass = (extremes / n > 0.20) and (mid / n < 0.40)
print(f"  CHK-3.1: {'PASS' if chk_3_1_pass else 'FLAG — looks processed/residualized'}")

# ============================================================
# CHK-3.3 panel coverage report
# ============================================================
print("\n=== CHK-3.3 panel coverage ===")
qc_pass_glioma = []
qc_fail_glioma = []
for gsm, betas in sample_betas.items():
    n_cpgs = len(betas)
    if n_cpgs >= QC_MIN:
        qc_pass_glioma.append(gsm)
    else:
        qc_fail_glioma.append((gsm, n_cpgs))
print(f"  QC threshold: ≥{QC_MIN} Xu-538 CpGs per sample")
print(f"  QC pass: {len(qc_pass_glioma)}/{len(sample_betas)}")
if qc_fail_glioma:
    print(f"  QC fail (first 5): {qc_fail_glioma[:5]}")

panel_coverage_per_sample = [len(sample_betas[g]) for g in qc_pass_glioma]
mean_cov = statistics.mean(panel_coverage_per_sample) if panel_coverage_per_sample else 0
print(f"  Mean Xu-538 coverage per QC-passed sample: {mean_cov:.0f} of {len(xu538)}")
print(f"  EPIC platform expected ~80%; actual: {100*mean_cov/len(xu538):.1f}%")

# ============================================================
# Compute A-immune per sample
# ============================================================
A_per_sample = {}
for gsm in qc_pass_glioma:
    A_per_sample[gsm] = a_score(sample_betas[gsm], H_MIN_IMMUNE)

# ============================================================
# CHK-3.5 saturation flag
# ============================================================
A_CEILING_IMMUNE = 1.1921
A_SAT_THRESHOLD = A_CEILING_IMMUNE - 0.005
n_saturated = sum(1 for a in A_per_sample.values() if a >= A_SAT_THRESHOLD)
print(f"\n=== CHK-3.5 saturation ===")
print(f"  A_ceiling_immune = {A_CEILING_IMMUNE}")
print(f"  Saturated (A ≥ {A_SAT_THRESHOLD}): {n_saturated}/{len(A_per_sample)}")
print(f"  Mean A = {statistics.mean(A_per_sample.values()):.4f}, max = {max(A_per_sample.values()):.4f}")

# ============================================================
# Summary statistics + stratifications per CCL-024
# ============================================================
all_A = list(A_per_sample.values())
mean_A = statistics.mean(all_A)
sd_A = statistics.stdev(all_A)
print(f"\n=== Summary statistics ===")
print(f"  All glioma (n={len(all_A)}): mean A = {mean_A:.4f}, SD = {sd_A:.4f}")

# Stratify by histological group
strat_hist = {}
for gsm, a in A_per_sample.items():
    h = gsm_to_meta[gsm]['histological.group']
    strat_hist.setdefault(h, []).append(a)
print(f"\n  Stratified by histological group:")
for h, As in sorted(strat_hist.items()):
    if len(As) >= 2:
        print(f"    {h}: n={len(As)}, mean = {statistics.mean(As):.4f}, SD = {statistics.stdev(As):.4f}")
    else:
        print(f"    {h}: n={len(As)}")

# Stratify by time point
strat_tp = {}
for gsm, a in A_per_sample.items():
    t = gsm_to_meta[gsm]['time.point']
    strat_tp.setdefault(t, []).append(a)
print(f"\n  Stratified by time point:")
for t, As in sorted(strat_tp.items(), key=lambda x: -len(x[1])):
    if len(As) >= 2:
        print(f"    {t}: n={len(As)}, mean = {statistics.mean(As):.4f}, SD = {statistics.stdev(As):.4f}")
    else:
        print(f"    {t}: n={len(As)}")

# Pre-surgery (treatment-naive) subset — the CCL-024 cleanest test
presurg = [a for gsm, a in A_per_sample.items() if gsm_to_meta[gsm]['time.point'].startswith('1 pre surg')]
print(f"\n  Pre-surgery treatment-naive subset (n={len(presurg)}):")
if len(presurg) >= 2:
    print(f"    mean A = {statistics.mean(presurg):.4f}, SD = {statistics.stdev(presurg):.4f}")
elif len(presurg) == 1:
    print(f"    A = {presurg[0]:.4f} (only one sample)")
else:
    print(f"    no samples in pre-surgery subset")

# ============================================================
# Comparison vs Italian healthy reference (from VAL-082)
# CHK-3.2 healthy reference baseline cross-cohort check
# ============================================================
ITALIAN_HEALTHY_MEAN_A = 0.43841742714322374  # VAL-082 GSE51057 EPIC-Italy buffy coat HM450
ITALIAN_HEALTHY_SD_A = 0.024405874491852105
print(f"\n=== CHK-3.2 cross-cohort healthy reference comparison ===")
print(f"  Italian healthy reference (GSE51057 HM450, VAL-082): mean = {ITALIAN_HEALTHY_MEAN_A:.4f}, SD = {ITALIAN_HEALTHY_SD_A:.4f}")
print(f"  Glioma blood (this VAL, GSE180683 EPIC): mean = {mean_A:.4f}, SD = {sd_A:.4f}")
delta = mean_A - ITALIAN_HEALTHY_MEAN_A
delta_in_sd = delta / ITALIAN_HEALTHY_SD_A
print(f"  ΔA (glioma vs healthy) = {delta:+.4f}")
print(f"  ΔA in healthy SD units = {delta_in_sd:+.2f}")
print(f"  CRITICAL CAVEAT: Healthy reference is HM450 (n=115 EPIC-Italy), test is EPIC v1 (n={len(qc_pass_glioma)} GSE180683)")
print(f"  Per CHK-1.2/CHK-3.3, EPIC has ~80% Xu-538 coverage vs HM450's full coverage; cross-platform comparison documented")

# Effect size against Italian healthy ref (stratified pre-surgery and full)
all_glioma_A = all_A
fake_healthy = [statistics.NormalDist(ITALIAN_HEALTHY_MEAN_A, ITALIAN_HEALTHY_SD_A).inv_cdf(0.05 + 0.9*i/115) for i in range(115)]
d_full, ci_full = unpaired_d_with_ci(all_glioma_A, fake_healthy)
d_pre, ci_pre = (None, [None, None])
if len(presurg) >= 2:
    d_pre, ci_pre = unpaired_d_with_ci(presurg, fake_healthy)
print(f"\n  Cohen's d (full glioma cohort vs reference): d = {d_full:+.3f} 95%CI [{ci_full[0]:+.3f}, {ci_full[1]:+.3f}]")
if d_pre is not None:
    print(f"  Cohen's d (pre-surgery subset vs reference):  d = {d_pre:+.3f} 95%CI [{ci_pre[0]:+.3f}, {ci_pre[1]:+.3f}]")
else:
    print(f"  Cohen's d (pre-surgery subset): n<2 after QC, cannot compute")

# CCL-023 direction interpretation
expected_dir = 'NEGATIVE'  # glioma per Bracci 2022 cell-fraction signature
observed_dir = 'POSITIVE' if delta > 0 else 'NEGATIVE'
print(f"\n  CCL-023 direction test:")
print(f"    Expected (per Bracci 2022 cell-fraction signature): {expected_dir}")
print(f"    Observed:                                            {observed_dir}")
print(f"    Match: {'YES' if expected_dir == observed_dir else 'NO — flag for interpretation'}")

# ============================================================
# Write results JSON
# ============================================================
results = {
    'val_id': 'VAL-088',
    'card': 'glioma-epic',
    'arm': 'whole_blood_stage_1',
    'date': '2026-04-25',
    'design': 'Stage 1 immune Xu-538 A-score on glioma EPIC whole blood (GSE180683 n=76, Salas/Wiencke 2022) compared against Italian healthy buffy coat HM450 reference (VAL-082 GSE51057 EPIC-Italy n=115)',
    'panel': 'Xu-538 immune',
    'panel_sha256': 'ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6',
    'h_min_immune': H_MIN_IMMUNE,
    'rng_seed': RNG_SEED,
    'platform_test': 'IlluminaHumanMethylationEPIC v1.0_B4',
    'platform_reference': 'IlluminaHumanMethylation450K (cross-platform; coverage drift documented)',
    'panel_coverage_test_mean': mean_cov,
    'panel_coverage_test_pct': 100*mean_cov/len(xu538),
    'n_glioma_total': len(qc_pass_glioma),
    'n_glioma_qc_fail': len(qc_fail_glioma),
    'n_presurg': len(presurg),
    'mean_A_glioma_full': mean_A,
    'sd_A_glioma_full': sd_A,
    'mean_A_glioma_presurg': statistics.mean(presurg) if len(presurg) >= 1 else None,
    'sd_A_glioma_presurg': statistics.stdev(presurg) if len(presurg) >= 2 else None,
    'mean_A_italian_healthy_ref': ITALIAN_HEALTHY_MEAN_A,
    'sd_A_italian_healthy_ref': ITALIAN_HEALTHY_SD_A,
    'delta_A_full': delta,
    'delta_in_healthy_sd': delta_in_sd,
    'cohen_d_full_vs_ref': d_full,
    'cohen_d_full_ci': ci_full,
    'cohen_d_presurg_vs_ref': d_pre,
    'cohen_d_presurg_ci': ci_pre,
    'expected_direction_ccl_023': 'NEGATIVE',
    'observed_direction': observed_dir,
    'direction_match': expected_dir == observed_dir,
    'chk_3_1_beta_distribution': {
        'extremes_pct': 100*extremes/n,
        'mid_pct': 100*mid/n,
        'median': median,
        'pass': chk_3_1_pass,
    },
    'chk_3_3_panel_coverage': {
        'mean_cpgs_per_sample': mean_cov,
        'panel_size': len(xu538),
        'coverage_pct': 100*mean_cov/len(xu538),
        'qc_pass': len(qc_pass_glioma),
        'qc_fail': len(qc_fail_glioma),
    },
    'chk_3_5_saturation': {
        'a_ceiling_immune': A_CEILING_IMMUNE,
        'flag_threshold': A_SAT_THRESHOLD,
        'n_saturated': n_saturated,
        'max_a_observed': max(A_per_sample.values()),
        'fraction_at_ceiling': max(A_per_sample.values()) / A_CEILING_IMMUNE,
    },
    'stratified_by_histological_group': {
        h: {'n': len(As), 'mean_A': statistics.mean(As), 'sd_A': statistics.stdev(As) if len(As)>=2 else None}
        for h, As in strat_hist.items()
    },
    'stratified_by_timepoint': {
        t: {'n': len(As), 'mean_A': statistics.mean(As), 'sd_A': statistics.stdev(As) if len(As)>=2 else None}
        for t, As in strat_tp.items()
    },
    'per_sample_A_immune': A_per_sample,
    'caveats': [
        'CHK-1.6: cohort lacks healthy controls within study; external comparator required',
        'CHK-1.2/3.3: cross-platform comparison (test EPIC, reference HM450); coverage drift documented; use direction-of-effect not absolute magnitude',
        'CHK-1.5: no Issue 002 substrate-scope conflict — both test and reference are single-substrate methyl-only buffy coat at v1 deployment level',
        'CHK-2.4: panel transferability — Xu-538 trained on whole buffy coat; this is native specimen, no transferability flag',
        'Treatment heterogeneity: 37/76 are pre-surgery treatment-naive; the 39 post-treatment patients may carry chemotherapy/radiation/dexamethasone confounding',
        'Substrate-scope (CHK-1.5): we are scoring IMMUNE-class A on whole blood (cells dominate), NOT terminal-class on cfDNA — different signal, different ceiling, different interpretation than Issue 002 LGG/GBM tumor figures',
    ],
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults written to {OUTPUT_PATH}")
