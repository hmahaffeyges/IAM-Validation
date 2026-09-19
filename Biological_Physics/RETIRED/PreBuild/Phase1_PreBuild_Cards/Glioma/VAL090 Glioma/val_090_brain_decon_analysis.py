#!/usr/bin/env python3
"""
VAL-090: Loyfer/Moss reference_atlas Cortical_neurons fraction analysis.

Tests whether the Cortical_neurons cell-type fraction (Loyfer 2023 atlas, 
deconvolved via NNLS on Illumina array β values) differs between:
  - Healthy peripheral blood (GSE51057 EPIC-Italy buffy coat, n=177 cancer-free)
  - Glioma peripheral blood (GSE180683 Salas/Wiencke 2022 EPIC, n=76)
  - GBM tumor tissue (GSE60274 Lai 2015 450K, n=72 + 5 NTB controls)

The hypothesis going in: Cortical_neurons fraction in plasma is far below
detection in healthy adults (<<1% per Moss 2018 expectations), and may or may
not increase in glioma patients. If we see *any* signal in glioma blood that
isn't there in healthy blood, that's a finding.

In the tumor tissue arm, Cortical_neurons should be the dominant component
in NTB controls and lower in GBM (because GBM disrupts/replaces normal
brain architecture). 
"""
import csv
import statistics
import math
import json

REF_ATLAS_CELLS = [
    'Monocytes_EPIC', 'B-cells_EPIC', 'CD4T-cells_EPIC', 'NK-cells_EPIC',
    'CD8T-cells_EPIC', 'Neutrophils_EPIC', 'Erythrocyte_progenitors',
    'Adipocytes', 'Cortical_neurons', 'Hepatocytes', 'Lung_cells',
    'Pancreatic_beta_cells', 'Pancreatic_acinar_cells', 'Pancreatic_duct_cells',
    'Vascular_endothelial_cells', 'Colon_epithelial_cells', 'Left_atrium',
    'Bladder', 'Breast', 'Head_and_neck_larynx', 'Kidney', 'Prostate',
    'Thyroid', 'Upper_GI', 'Uterus_cervix'
]


def load_decon(path):
    """Load deconvolution output: rows = cell types, cols = samples."""
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)  # ['', sample1, sample2, ...]
        samples = header[1:]
        cells = {}
        for row in reader:
            cell = row[0]
            vals = [float(v) for v in row[1:]]
            cells[cell] = dict(zip(samples, vals))
    return samples, cells


def load_gsm_phenotype_GSE60274():
    """Map GSE60274 GSM -> phenotype label (GBM_primary, GBM_recurrent, NTB, sphere)"""
    src = '/home/claude/glioma_work/GSE60274_series_matrix.txt.gz'
    import gzip
    titles = []
    gsms = []
    with gzip.open(src, 'rt') as f:
        for line in f:
            if line.startswith('!Sample_title'):
                titles = [s.strip().strip('"') for s in line.strip().split('\t')[1:]]
            elif line.startswith('!Sample_geo_accession'):
                gsms = [s.strip().strip('"') for s in line.strip().split('\t')[1:]]
                break
    pheno = {}
    for gsm, t in zip(gsms, titles):
        tl = t.lower()
        if 'cultured glioma sphere' in tl:
            pheno[gsm] = 'sphere'
        elif 'recurrent gbm' in tl:
            pheno[gsm] = 'GBM_recurrent'
        elif 'craniotomy' in tl or 'lobectomy' in tl or 'ntb' in tl:
            pheno[gsm] = 'NTB'
        elif 'surgical resection gbm' in tl:
            pheno[gsm] = 'GBM_primary'
        else:
            pheno[gsm] = 'unknown'
    return pheno


def load_gsm_phenotype_GSE180683():
    """Map GSM -> (LGG/GBM, pre/post-surgery, treatment status)"""
    # Use the manifest we already generated for VAL-088
    with open('/home/claude/iam_repo/Biological_Physics/validation_runs/GSE180683_manifest.json') as f:
        manifest = json.load(f)
    return manifest


def stats(values):
    if len(values) < 2:
        return {'n': len(values), 'mean': values[0] if values else None, 'sd': 0.0}
    return {
        'n': len(values),
        'mean': statistics.mean(values),
        'sd': statistics.stdev(values),
        'median': statistics.median(values),
        'min': min(values),
        'max': max(values),
        'p5': sorted(values)[max(0, int(0.05*len(values)))],
        'p95': sorted(values)[min(len(values)-1, int(0.95*len(values)))],
    }


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa, sb = statistics.stdev(a), statistics.stdev(b)
    pooled = math.sqrt((sa**2 + sb**2) / 2)
    if pooled == 0:
        return None
    return (ma - mb) / pooled


def main():
    print("=" * 80)
    print("VAL-090 Loyfer/Moss reference_atlas — Cortical_neurons fraction")
    print("=" * 80)
    
    # Load all three deconvolution outputs
    healthy_samples, healthy_cells = load_decon(
        '/home/claude/brain_decon/results/GSE51057_betas_healthy_deconv_output.csv')
    blood_samples, blood_cells = load_decon(
        '/home/claude/brain_decon/results/GSE180683_betas_deconv_output.csv')
    tissue_samples, tissue_cells = load_decon(
        '/home/claude/brain_decon/results/GSE60274_betas_deconv_output.csv')
    
    print(f"\nHealthy reference (GSE51057): {len(healthy_samples)} samples")
    print(f"Glioma blood (GSE180683):     {len(blood_samples)} samples")
    print(f"Brain tissue (GSE60274):      {len(tissue_samples)} samples")
    
    # ========== Healthy reference ==========
    print("\n--- HEALTHY REFERENCE (GSE51057 buffy coat n=177) ---")
    healthy_neurons = [healthy_cells['Cortical_neurons'][s] for s in healthy_samples]
    s = stats(healthy_neurons)
    print(f"Cortical_neurons fraction: mean={s['mean']:.4f} sd={s['sd']:.4f}")
    print(f"  median={s['median']:.4f}  range=[{s['min']:.4f}, {s['max']:.4f}]")
    print(f"  95th percentile: {s['p95']:.4f}  (>= 1% threshold: {sum(v >= 0.01 for v in healthy_neurons)}/{len(healthy_neurons)} samples)")
    healthy_neurons_stats = s
    
    # ========== Glioma plasma ==========
    print("\n--- GLIOMA PERIPHERAL BLOOD (GSE180683 EPIC n=76) ---")
    blood_neurons = [blood_cells['Cortical_neurons'][s] for s in blood_samples]
    s = stats(blood_neurons)
    print(f"Cortical_neurons fraction: mean={s['mean']:.4f} sd={s['sd']:.4f}")
    print(f"  median={s['median']:.4f}  range=[{s['min']:.4f}, {s['max']:.4f}]")
    print(f"  >= 1% threshold: {sum(v >= 0.01 for v in blood_neurons)}/{len(blood_neurons)} samples")
    print(f"  >= 0.5% threshold: {sum(v >= 0.005 for v in blood_neurons)}/{len(blood_neurons)} samples")
    blood_neurons_stats = s
    
    d = cohen_d(blood_neurons, healthy_neurons)
    print(f"\n  Glioma blood vs Healthy Italian: Cohen's d = {d:+.3f}" if d else "  insufficient")
    print(f"  Mean difference: {blood_neurons_stats['mean'] - healthy_neurons_stats['mean']:+.5f}")
    
    # Stratify by pre-surgery LGG vs GBM (manifest values: time.point="1 pre surg", group=new gbm/new lgg/rec lgg/rec lgg now gbm)
    manifest_list = load_gsm_phenotype_GSE180683()
    blood_phen = {item['gsm']: item for item in manifest_list}
    presurg_lgg_n = []
    presurg_gbm_n = []
    presurg_all_n = []
    for s in blood_samples:
        meta = blood_phen.get(s, {})
        timepoint = meta.get('time.point', '').lower()
        hgroup = meta.get('histological.group', '').lower()
        is_presurg = 'pre surg' in timepoint
        if is_presurg:
            v = blood_cells['Cortical_neurons'][s]
            presurg_all_n.append(v)
            # 'new gbm' or 'now gbm' -> GBM; '...lgg' (no gbm) -> LGG
            if 'gbm' in hgroup and 'lgg' not in hgroup:
                presurg_gbm_n.append(v)
            elif 'lgg' in hgroup and 'gbm' not in hgroup:
                presurg_lgg_n.append(v)
    if presurg_all_n:
        print(f"\n  Pre-surgery treatment-naive all (n={len(presurg_all_n)}): mean = {statistics.mean(presurg_all_n):.5f} sd = {statistics.stdev(presurg_all_n):.5f}")
        d_pre = cohen_d(presurg_all_n, healthy_neurons)
        print(f"    Cohen's d vs healthy: {d_pre:+.3f}")
    if presurg_lgg_n:
        print(f"  Pre-surgery LGG (n={len(presurg_lgg_n)}): mean = {statistics.mean(presurg_lgg_n):.5f}")
    if presurg_gbm_n:
        print(f"  Pre-surgery GBM (n={len(presurg_gbm_n)}): mean = {statistics.mean(presurg_gbm_n):.5f}")
    
    # ========== Glioma tumor tissue ==========
    print("\n--- BRAIN TUMOR TISSUE (GSE60274 450K) ---")
    pheno = load_gsm_phenotype_GSE60274()
    by_class = {'NTB': [], 'GBM_primary': [], 'GBM_recurrent': [], 'sphere': [], 'unknown': []}
    for s in tissue_samples:
        cls = pheno.get(s, 'unknown')
        by_class[cls].append(tissue_cells['Cortical_neurons'][s])
    
    for cls in ['NTB', 'GBM_primary', 'GBM_recurrent', 'sphere']:
        vals = by_class[cls]
        if vals:
            ss = stats(vals)
            print(f"  {cls:15s} n={ss['n']:3d}  mean={ss['mean']:.4f}  sd={ss['sd']:.4f}  range=[{ss['min']:.4f}, {ss['max']:.4f}]")
    
    if by_class['NTB'] and by_class['GBM_primary']:
        d = cohen_d(by_class['GBM_primary'], by_class['NTB'])
        print(f"\n  GBM_primary vs NTB Cohen's d (Cortical_neurons fraction): {d:+.3f}")
    if by_class['NTB'] and by_class['sphere']:
        d = cohen_d(by_class['sphere'], by_class['NTB'])
        print(f"  Sphere vs NTB Cohen's d:                                 {d:+.3f}")
    
    # ========== HEADLINE ==========
    print("\n" + "=" * 80)
    print("HEADLINE")
    print("=" * 80)
    print()
    print(f"Healthy peripheral blood (n=177):       Cortical_neurons = {healthy_neurons_stats['mean']*100:.3f}% (mean)")
    print(f"Glioma peripheral blood (n=76):         Cortical_neurons = {blood_neurons_stats['mean']*100:.3f}% (mean)")
    print(f"NTB brain tissue (n={len(by_class['NTB'])}):                Cortical_neurons = {statistics.mean(by_class['NTB'])*100:.2f}% (mean)" if by_class['NTB'] else '')
    print(f"GBM primary brain tissue (n={len(by_class['GBM_primary'])}):       Cortical_neurons = {statistics.mean(by_class['GBM_primary'])*100:.2f}% (mean)" if by_class['GBM_primary'] else '')
    print(f"GBM recurrent (n={len(by_class['GBM_recurrent'])}):                Cortical_neurons = {statistics.mean(by_class['GBM_recurrent'])*100:.2f}% (mean)" if by_class['GBM_recurrent'] else '')
    print(f"Cultured spheres (n={len(by_class['sphere'])}):              Cortical_neurons = {statistics.mean(by_class['sphere'])*100:.2f}% (mean)" if by_class['sphere'] else '')
    
    # Save results JSON
    results = {
        'val_id': 'VAL-090',
        'reference_atlas': 'Loyfer 2023 / Moss 2018 reference_atlas.csv (nloyfer/meth_atlas)',
        'method': 'NNLS deconvolution',
        'healthy_blood': {
            'cohort': 'GSE51057 EPIC-Italy buffy coat cancer-free subset',
            'n': len(healthy_samples),
            'cortical_neurons_mean_pct': healthy_neurons_stats['mean']*100,
            'cortical_neurons_sd': healthy_neurons_stats['sd'],
            'cortical_neurons_p95_pct': healthy_neurons_stats['p95']*100,
        },
        'glioma_blood': {
            'cohort': 'GSE180683 Salas/Wiencke 2022 EPIC peripheral blood',
            'n': len(blood_samples),
            'cortical_neurons_mean_pct': blood_neurons_stats['mean']*100,
            'cortical_neurons_sd': blood_neurons_stats['sd'],
            'cohen_d_vs_healthy': cohen_d(blood_neurons, healthy_neurons),
            'fraction_above_1pct': sum(v >= 0.01 for v in blood_neurons) / len(blood_neurons),
            'fraction_above_05pct': sum(v >= 0.005 for v in blood_neurons) / len(blood_neurons),
        },
        'brain_tissue': {
            'cohort': 'GSE60274 Lai 2015 brain tissue 450K',
            'NTB': {'n': len(by_class['NTB']), 'cortical_neurons_mean_pct': statistics.mean(by_class['NTB'])*100 if by_class['NTB'] else None},
            'GBM_primary': {'n': len(by_class['GBM_primary']), 'cortical_neurons_mean_pct': statistics.mean(by_class['GBM_primary'])*100 if by_class['GBM_primary'] else None},
            'GBM_recurrent': {'n': len(by_class['GBM_recurrent']), 'cortical_neurons_mean_pct': statistics.mean(by_class['GBM_recurrent'])*100 if by_class['GBM_recurrent'] else None},
            'sphere': {'n': len(by_class['sphere']), 'cortical_neurons_mean_pct': statistics.mean(by_class['sphere'])*100 if by_class['sphere'] else None},
            'GBM_primary_vs_NTB_cohen_d': cohen_d(by_class['GBM_primary'], by_class['NTB']),
            'sphere_vs_NTB_cohen_d': cohen_d(by_class['sphere'], by_class['NTB']),
        }
    }
    with open('/home/claude/brain_decon/results/VAL-090_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: /home/claude/brain_decon/results/VAL-090_results.json")


if __name__ == '__main__':
    main()
