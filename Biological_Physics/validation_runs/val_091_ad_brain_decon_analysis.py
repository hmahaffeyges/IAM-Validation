#!/usr/bin/env python3
"""
VAL-091: Cortical-neuron cfDNA fraction in Alzheimer's disease peripheral blood
via Loyfer/Moss array atlas — cross-cohort analysis.

Tests whether the Cortical_neurons cell-type fraction (Loyfer 2023 atlas, deconvolved
via NNLS on Illumina array β values) differs between:
  - GSE51057 EPIC-Italy buffy coat (n=329 healthy reference, same as VAL-090 cohort)
  - GSE153712 AIBL whole blood (EPIC, n=726: AD / MCI / HC)
  - GSE144858 AddNeuroMed whole blood (450K, n=300: AD / MCI / HC)
  - GSE53740 GIFT whole blood (450K, n=384: HC / AD / FTD / PSP/CBD)

Stage 1 of the AD card v2.1 is the directional 7-CpG immune A_dir panel (VAL-051/052/057).
That panel reads d=+0.62 on AIBL holdout AD vs HC. VAL-091 is a separate, INDEPENDENT
Stage 2 question: does the cortical-neuron fraction (architecture-class readout, not
panel readout) elevate in AD blood?

The card v2.1 currently asserts "Stage 2 Moss NNLS for AD is expected NULL — brain
tissue not in buffy coat." The Loyfer atlas added a Cortical_neurons reference Moss 2018
did not have. VAL-091 directly tests whether that prediction holds with the new reference.

Pre-registration: VAL-091_prereg.md (sealed 2026-04-26, SHA
56c7cac9bb869e4ec2b72a6359f87767035443e0f9b5d34d1a7c848b10053c2f, BEFORE this script
was authored or any β-value access).

VAL-090 reference (glioma plasma, Loyfer atlas, same pipeline):
  HC GSE51057 mean = 0.276% (n=177 cancer-free)
  Glioma GSE180683 mean = 1.092% (n=76)
  Cohen's d = +1.96 [+1.62, +2.31]

Hypotheses (locked in prereg):
  Hypothesis A: AD elevates cortical-neuron cfDNA at glioma magnitude (d ≥ +1.0)
  Hypothesis B: AD does NOT elevate cortical-neuron cfDNA (d < +0.3)
  Hypothesis C: Intermediate elevation
"""
import csv
import json
import math
import statistics

# --- File paths (read-only inputs) -------------------------------------------
DECONV = {
    'GSE51057_HC':   '/home/claude/ad_loyfer/results/GSE51057_betas_loyfer_deconv_output.csv',
    'AIBL':          '/home/claude/ad_loyfer/results/GSE153712_betas_loyfer_deconv_output.csv',
    'AddNeuroMed':   '/home/claude/ad_loyfer/results/GSE144858_betas_loyfer_deconv_output.csv',
    'GIFT':          '/home/claude/ad_loyfer/results/GSE53740_betas_loyfer_deconv_output.csv',
}
META = {
    'AIBL':        '/home/claude/ad_loyfer/input/GSE153712_metadata.txt',
    'AddNeuroMed': '/home/claude/ad_loyfer/input/GSE144858_metadata.txt',
    'GIFT':        '/home/claude/ad_loyfer/input/GSE53740_metadata.txt',
}

# --- Helpers -----------------------------------------------------------------

def load_deconv_row(csv_path, target_row='Cortical_neurons'):
    """Load one cell-type row from deconvolution output. Returns dict sample_id → fraction."""
    out = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        sample_ids = header[1:]
        for row in reader:
            if row[0] == target_row:
                for sid, val in zip(sample_ids, row[1:]):
                    out[sid] = float(val)
                break
    return out

def load_all_classes(csv_path):
    """Load all 26 cell-type rows. Returns dict cell → dict sample_id → fraction."""
    out = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        sample_ids = header[1:]
        for row in reader:
            cell = row[0]
            out[cell] = {sid: float(v) for sid, v in zip(sample_ids, row[1:])}
    return out

def cohens_d(a, b):
    """Cohen's d with pooled SD."""
    if len(a) < 2 or len(b) < 2: return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa, sb = statistics.stdev(a), statistics.stdev(b)
    pooled = math.sqrt(((len(a)-1)*sa**2 + (len(b)-1)*sb**2) / (len(a)+len(b)-2))
    if pooled == 0: return None
    return (ma - mb) / pooled

def bootstrap_ci(a, b, n_iter=2000, seed=20260426):
    """Bootstrap 95% CI on Cohen's d (a vs b)."""
    import random
    rng = random.Random(seed)
    ds = []
    for _ in range(n_iter):
        ra = [rng.choice(a) for _ in range(len(a))]
        rb = [rng.choice(b) for _ in range(len(b))]
        d = cohens_d(ra, rb)
        if d is not None: ds.append(d)
    ds.sort()
    return (ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds))])

def descriptive(vals):
    """Mean, sd, median, n, min, max."""
    if len(vals) == 0:
        return {'n': 0}
    return {
        'n': len(vals),
        'mean': statistics.mean(vals),
        'sd': statistics.stdev(vals) if len(vals) > 1 else None,
        'median': statistics.median(vals),
        'min': min(vals),
        'max': max(vals),
        'p25': statistics.quantiles(vals, n=4)[0] if len(vals) >= 4 else None,
        'p75': statistics.quantiles(vals, n=4)[2] if len(vals) >= 4 else None,
    }

# --- AIBL phenotype parser ----------------------------------------------------

def parse_aibl_phenotype():
    """Parse AIBL metadata: GSM ID → (disease_status, gender). Disease in sample_title."""
    titles = {}     # GSM → title  (contains "whole blood {disease}")
    genders = {}    # GSM → 'Male'|'Female'
    diseases = {}   # GSM → 'AD'|'MCI'|'HC'
    
    with open(META['AIBL']) as f:
        title_line = None
        accession_line = None
        gender_line = None
        disease_line = None
        for line in f:
            if line.startswith('!Sample_title'):
                title_line = line.rstrip('\n')
            elif line.startswith('!Sample_geo_accession'):
                accession_line = line.rstrip('\n')
            elif line.startswith('!Sample_characteristics_ch1'):
                # Multiple of these. The one with "gender:" and "disease status:"
                if 'gender:' in line:
                    gender_line = line.rstrip('\n')
                elif 'disease status:' in line:
                    disease_line = line.rstrip('\n')
    
    # Parse the tab-delimited lines (each has !key + sample values)
    title_vals = title_line.split('\t')[1:]
    accession_vals = accession_line.split('\t')[1:]
    gender_vals = gender_line.split('\t')[1:]
    disease_vals = disease_line.split('\t')[1:]
    
    # Strip quotes and parse
    for i, gsm in enumerate(accession_vals):
        gsm = gsm.strip().strip('"')
        title = title_vals[i].strip().strip('"')
        gender_v = gender_vals[i].strip().strip('"').replace('gender: ', '')
        disease_v = disease_vals[i].strip().strip('"').replace('disease status: ', '')
        
        # Map disease
        if 'Alzheimer' in disease_v:
            d = 'AD'
        elif 'Mild Cognitive' in disease_v:
            d = 'MCI'
        elif 'healthy control' in disease_v:
            d = 'HC'
        else:
            d = 'UNKNOWN'
        
        # AIBL deconv uses chip_pos as sample ID, not GSM. Title format:
        # "{chip_pos}_whole blood {disease}"
        # Parse chip_pos from title
        chip_pos = title.split('_whole blood')[0]
        
        diseases[chip_pos] = d
        genders[chip_pos] = gender_v
    
    return diseases, genders

# --- AddNeuroMed phenotype parser --------------------------------------------

def parse_addneuromed_phenotype():
    """AddNeuroMed (GSE144858): metadata characteristics include 'disease status' and 'age'.
    Sample IDs in deconv are GSM IDs."""
    char_lines = []
    accession_line = None
    
    with open(META['AddNeuroMed']) as f:
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                accession_line = line.rstrip('\n')
            elif line.startswith('!Sample_characteristics_ch1'):
                char_lines.append(line.rstrip('\n'))
    
    accession_vals = [a.strip().strip('"') for a in accession_line.split('\t')[1:]]
    
    # Each char_line has a different attribute. Determine attribute from first non-empty value.
    diseases = {}
    ages = {}
    genders = {}
    
    for line in char_lines:
        vals = [v.strip().strip('"') for v in line.split('\t')[1:]]
        # Sample some non-empty values to identify attribute type
        sample_vals = [v for v in vals[:20] if v]
        if not sample_vals: continue
        # Type inference
        first = sample_vals[0].lower()
        if 'age' in first:
            for gsm, v in zip(accession_vals, vals):
                v = v.replace('age (yr): ', '').replace('age: ', '').strip()
                try:
                    ages[gsm] = float(v)
                except ValueError:
                    pass
        elif 'gender' in first or 'sex' in first or 'female' in first or 'male' in first:
            for gsm, v in zip(accession_vals, vals):
                v_low = v.lower()
                if 'female' in v_low: genders[gsm] = 'Female'
                elif 'male' in v_low: genders[gsm] = 'Male'
        elif 'group' in first or 'status' in first or 'control' in first or 'patient' in first or 'ad' in first or 'mci' in first:
            for gsm, v in zip(accession_vals, vals):
                v_low = v.lower()
                if 'ad' == v_low.split(':')[-1].strip() or 'alzheimer' in v_low or 'patient' in v_low or 'cad' in v_low:
                    # check more carefully
                    pass
                # fall through to next pass
    
    # Better: re-scan and look for the canonical phenotype line
    # GSE144858 typically uses "diagnosis:" or "subject group:"
    for line in char_lines:
        vals = [v.strip().strip('"') for v in line.split('\t')[1:]]
        sample_vals = [v for v in vals if v]
        if not sample_vals: continue
        # Look for AD / MCI / control values
        types_seen = set()
        for v in sample_vals[:30]:
            v_low = v.lower()
            if 'alzheim' in v_low or v_low.endswith(': ad') or v_low == 'ad':
                types_seen.add('AD')
            elif 'mci' in v_low or 'mild cognitive' in v_low:
                types_seen.add('MCI')
            elif 'control' in v_low or 'healthy' in v_low or v_low.endswith(': ctl'):
                types_seen.add('HC')
        if 'AD' in types_seen and 'HC' in types_seen:
            # This is the disease line
            for gsm, v in zip(accession_vals, vals):
                v_low = v.lower()
                if 'alzheim' in v_low or v_low.endswith(': ad'):
                    diseases[gsm] = 'AD'
                elif 'mci' in v_low or 'mild cognitive' in v_low:
                    diseases[gsm] = 'MCI'
                elif 'control' in v_low or 'healthy' in v_low:
                    diseases[gsm] = 'HC'
                else:
                    diseases[gsm] = 'UNKNOWN'
            break
    
    return diseases, genders, ages

# --- GIFT phenotype parser ---------------------------------------------------

def parse_gift_phenotype():
    """GSE53740: HC, AD, FTD subtypes, PSP, CBD."""
    char_lines = []
    accession_line = None
    title_line = None
    
    with open(META['GIFT']) as f:
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                accession_line = line.rstrip('\n')
            elif line.startswith('!Sample_title'):
                title_line = line.rstrip('\n')
            elif line.startswith('!Sample_characteristics_ch1'):
                char_lines.append(line.rstrip('\n'))
    
    accession_vals = [a.strip().strip('"') for a in accession_line.split('\t')[1:]]
    title_vals = [a.strip().strip('"') for a in title_line.split('\t')[1:]] if title_line else []
    
    diseases = {}
    ages = {}
    genders = {}
    
    # GSE53740 disease info typically in characteristics with "diagnosis:" label
    for line in char_lines:
        vals = [v.strip().strip('"') for v in line.split('\t')[1:]]
        sample_vals = [v for v in vals if v]
        if not sample_vals: continue
        first = sample_vals[0].lower()
        if 'age' in first:
            for gsm, v in zip(accession_vals, vals):
                vc = v.replace('age:', '').replace('age (yr):', '').strip()
                try:
                    ages[gsm] = float(vc)
                except ValueError:
                    pass
        elif 'gender' in first or 'sex' in first or first.startswith('m') or first.startswith('f'):
            for gsm, v in zip(accession_vals, vals):
                v_low = v.lower()
                if 'female' in v_low or v_low.endswith(': f'): genders[gsm] = 'Female'
                elif 'male' in v_low or v_low.endswith(': m'): genders[gsm] = 'Male'
    
    # Disease: scan all char lines for diagnosis-like content
    for line in char_lines:
        vals = [v.strip().strip('"') for v in line.split('\t')[1:]]
        sample_vals = [v for v in vals if v]
        if not sample_vals: continue
        types_seen = set()
        for v in sample_vals[:60]:
            v_low = v.lower()
            if 'ftd' in v_low: types_seen.add('FTD')
            if 'psp' in v_low: types_seen.add('PSP')
            if 'cbd' in v_low: types_seen.add('CBD')
            if v_low.startswith('disease state: control') or v_low == 'disease state: ctrl' or 'control' in v_low.split(':')[-1]:
                types_seen.add('HC')
            if 'ad' == v_low.split(':')[-1].strip() or 'alzheim' in v_low:
                types_seen.add('AD')
        if len(types_seen) >= 3:  # this is the disease line
            for gsm, v in zip(accession_vals, vals):
                v_low = v.lower()
                last = v_low.split(':')[-1].strip()
                if 'ftd' in v_low:
                    diseases[gsm] = 'FTD'
                elif 'psp' in v_low or 'cbd' in v_low:
                    diseases[gsm] = 'PSP_CBD'
                elif last == 'ad' or 'alzheim' in v_low:
                    diseases[gsm] = 'AD'
                elif 'control' in v_low or last == 'ctrl' or last == 'normal':
                    diseases[gsm] = 'HC'
                else:
                    diseases[gsm] = 'UNKNOWN'
            break
    
    return diseases, genders, ages

# --- Main analysis ------------------------------------------------------------

def main():
    print('=== VAL-091: AD cortical-neuron cfDNA via Loyfer atlas ===')
    print()
    
    # 1. Load deconvolution Cortical_neurons rows
    cn_hc = load_deconv_row(DECONV['GSE51057_HC'])
    cn_aibl = load_deconv_row(DECONV['AIBL'])
    cn_anm = load_deconv_row(DECONV['AddNeuroMed'])
    cn_gift = load_deconv_row(DECONV['GIFT'])
    
    print(f'GSE51057 HC samples in deconv: {len(cn_hc)}')
    print(f'AIBL samples in deconv: {len(cn_aibl)}')
    print(f'AddNeuroMed samples in deconv: {len(cn_anm)}')
    print(f'GIFT samples in deconv: {len(cn_gift)}')
    print()
    
    # 2. AIBL phenotype
    aibl_disease, aibl_gender = parse_aibl_phenotype()
    aibl_disease_count = {}
    for d in aibl_disease.values():
        aibl_disease_count[d] = aibl_disease_count.get(d, 0) + 1
    print(f'AIBL phenotype counts: {aibl_disease_count}')
    
    # AIBL: subset to AD and HC. AIBL deconv keys are chip_pos.
    aibl_ad = [v for sid, v in cn_aibl.items() if aibl_disease.get(sid) == 'AD']
    aibl_hc = [v for sid, v in cn_aibl.items() if aibl_disease.get(sid) == 'HC']
    aibl_mci = [v for sid, v in cn_aibl.items() if aibl_disease.get(sid) == 'MCI']
    print(f'AIBL matched: AD={len(aibl_ad)}, HC={len(aibl_hc)}, MCI={len(aibl_mci)}')
    
    # 3. AddNeuroMed phenotype
    anm_disease, anm_gender, anm_ages = parse_addneuromed_phenotype()
    anm_disease_count = {}
    for d in anm_disease.values():
        anm_disease_count[d] = anm_disease_count.get(d, 0) + 1
    print(f'AddNeuroMed phenotype counts: {anm_disease_count}')
    
    anm_ad = [v for sid, v in cn_anm.items() if anm_disease.get(sid) == 'AD']
    anm_hc = [v for sid, v in cn_anm.items() if anm_disease.get(sid) == 'HC']
    anm_mci = [v for sid, v in cn_anm.items() if anm_disease.get(sid) == 'MCI']
    print(f'AddNeuroMed matched: AD={len(anm_ad)}, HC={len(anm_hc)}, MCI={len(anm_mci)}')
    
    # 4. GIFT phenotype
    gift_disease, gift_gender, gift_ages = parse_gift_phenotype()
    gift_disease_count = {}
    for d in gift_disease.values():
        gift_disease_count[d] = gift_disease_count.get(d, 0) + 1
    print(f'GIFT phenotype counts: {gift_disease_count}')
    
    gift_ad = [v for sid, v in cn_gift.items() if gift_disease.get(sid) == 'AD']
    gift_hc = [v for sid, v in cn_gift.items() if gift_disease.get(sid) == 'HC']
    gift_ftd = [v for sid, v in cn_gift.items() if gift_disease.get(sid) == 'FTD']
    gift_psp = [v for sid, v in cn_gift.items() if gift_disease.get(sid) == 'PSP_CBD']
    print(f'GIFT matched: AD={len(gift_ad)}, HC={len(gift_hc)}, FTD={len(gift_ftd)}, PSP_CBD={len(gift_psp)}')
    print()
    
    hc_external = list(cn_hc.values())  # GSE51057 healthy reference (n=329 from full set)
    
    # 5. Per-cohort descriptives (% units)
    def to_pct(vals): return [v * 100 for v in vals]
    
    cohorts = {
        'GSE51057_HC_external': to_pct(hc_external),
        'AIBL_HC': to_pct(aibl_hc),
        'AIBL_AD': to_pct(aibl_ad),
        'AIBL_MCI': to_pct(aibl_mci),
        'AddNeuroMed_HC': to_pct(anm_hc),
        'AddNeuroMed_AD': to_pct(anm_ad),
        'AddNeuroMed_MCI': to_pct(anm_mci),
        'GIFT_HC': to_pct(gift_hc),
        'GIFT_AD': to_pct(gift_ad),
        'GIFT_FTD': to_pct(gift_ftd),
        'GIFT_PSP_CBD': to_pct(gift_psp),
    }
    
    desc = {k: descriptive(v) for k, v in cohorts.items()}
    print('=== Cortical_neurons fraction (%) by group ===')
    for k, v in desc.items():
        if v.get('n', 0) > 0:
            mean_str = f"{v['mean']:.3f}" if v.get('mean') is not None else 'N/A'
            sd_str = f"{v['sd']:.3f}" if v.get('sd') is not None else 'N/A'
            print(f'  {k:32s} n={v["n"]:3d}  mean={mean_str}  sd={sd_str}')
    print()
    
    # 6. Cohen's d AD-vs-HC within each cohort
    def safe_d(a, b):
        if len(a) < 2 or len(b) < 2: return None
        d = cohens_d(a, b)
        ci = bootstrap_ci(a, b)
        return {'d': d, 'ci_lo': ci[0], 'ci_hi': ci[1], 'n_a': len(a), 'n_b': len(b)}
    
    # Within-cohort AD vs HC (primary tests)
    aibl_ad_v_hc = safe_d(cohorts['AIBL_AD'], cohorts['AIBL_HC'])
    anm_ad_v_hc = safe_d(cohorts['AddNeuroMed_AD'], cohorts['AddNeuroMed_HC'])
    gift_ad_v_hc = safe_d(cohorts['GIFT_AD'], cohorts['GIFT_HC'])
    
    # GIFT specificity arm
    gift_ftd_v_hc = safe_d(cohorts['GIFT_FTD'], cohorts['GIFT_HC'])
    gift_psp_v_hc = safe_d(cohorts['GIFT_PSP_CBD'], cohorts['GIFT_HC'])
    
    # Cross-cohort: pooled AD vs external HC
    pooled_ad = cohorts['AIBL_AD'] + cohorts['AddNeuroMed_AD'] + cohorts['GIFT_AD']
    pooled_hc_native = cohorts['AIBL_HC'] + cohorts['AddNeuroMed_HC'] + cohorts['GIFT_HC']
    pooled_v_external = safe_d(pooled_ad, cohorts['GSE51057_HC_external'])
    pooled_ad_v_pooled_hc = safe_d(pooled_ad, pooled_hc_native)
    
    # 7. Cross-comparison: AD cohorts vs VAL-090 glioma reference (literature value)
    # VAL-090 reported glioma plasma mean = 1.092%, n=76. We don't recompute;
    # we report the AD distributions next to that anchor.
    
    print('=== Cohen\'s d AD vs HC (within cohort) ===')
    for label, r in [('AIBL', aibl_ad_v_hc), ('AddNeuroMed', anm_ad_v_hc), ('GIFT', gift_ad_v_hc)]:
        if r:
            print(f'  {label:15s} d={r["d"]:+.3f} [{r["ci_lo"]:+.3f}, {r["ci_hi"]:+.3f}] (AD n={r["n_a"]}, HC n={r["n_b"]})')
    print()
    print('=== GIFT specificity arm ===')
    for label, r in [('FTD vs HC', gift_ftd_v_hc), ('PSP/CBD vs HC', gift_psp_v_hc)]:
        if r:
            print(f'  {label:15s} d={r["d"]:+.3f} [{r["ci_lo"]:+.3f}, {r["ci_hi"]:+.3f}]')
    print()
    print('=== Pooled AD vs HC ===')
    if pooled_ad_v_pooled_hc:
        r = pooled_ad_v_pooled_hc
        print(f'  vs pooled native HC (AIBL+ANM+GIFT):  d={r["d"]:+.3f} [{r["ci_lo"]:+.3f}, {r["ci_hi"]:+.3f}] (AD n={r["n_a"]}, HC n={r["n_b"]})')
    if pooled_v_external:
        r = pooled_v_external
        print(f'  vs GSE51057 external HC (n=329):       d={r["d"]:+.3f} [{r["ci_lo"]:+.3f}, {r["ci_hi"]:+.3f}] (AD n={r["n_a"]}, HC n={r["n_b"]})')
    print()
    
    # 8. Outcome label per prereg decision criteria
    primary_d = pooled_ad_v_pooled_hc['d'] if pooled_ad_v_pooled_hc else None
    if primary_d is None:
        outcome = 'O6_UNDETERMINED'
    elif primary_d >= 1.0:
        outcome = 'O1_AD_NEURO_POSITIVE_HIGH'
    elif primary_d >= 0.5:
        outcome = 'O2_AD_NEURO_POSITIVE_MEDIUM'
    elif primary_d >= 0.2:
        outcome = 'O3_AD_NEURO_POSITIVE_LOW'
    elif primary_d > -0.3:
        outcome = 'O4_AD_NEURO_NULL'
    else:
        outcome = 'O5_AD_NEURO_NEGATIVE'
    print(f'=== PRIMARY OUTCOME: {outcome} (pooled AD-vs-HC d = {primary_d:+.3f}) ===')
    print()
    
    # 9. Cross-cohort baseline check (CHK-7.7 sanity)
    # If HC cortical-neuron means differ massively across cohorts, that's a batch / 
    # platform problem that affects the AD finding interpretation.
    print('=== Cross-cohort HC baseline diagnostics ===')
    hc_means = {
        'GSE51057_HC_external (Loyfer ref base)': desc['GSE51057_HC_external']['mean'],
        'AIBL_HC (EPIC, Australian)':              desc['AIBL_HC']['mean'],
        'AddNeuroMed_HC (450K, European)':         desc['AddNeuroMed_HC']['mean'],
        'GIFT_HC (450K, US UCSF-MAC)':             desc['GIFT_HC']['mean'],
    }
    for k, v in hc_means.items():
        print(f'  {k:48s}  HC mean = {v:.3f}%')
    hc_max = max(hc_means.values())
    hc_min = min(hc_means.values())
    print(f'  Cross-cohort HC fold range: {hc_max/max(hc_min, 1e-9):.1f}×')
    print()
    
    # 10. VAL-090 anchor comparison
    GLIOMA_MEAN_PCT = 1.092
    HC_VAL090_MEAN_PCT = 0.276
    print('=== Cross-comparison to VAL-090 glioma anchor ===')
    print(f'  VAL-090 glioma plasma mean (n=76):           {GLIOMA_MEAN_PCT:.3f}%')
    print(f'  VAL-090 healthy reference (GSE51057, n=177): {HC_VAL090_MEAN_PCT:.3f}%')
    for k in ['AIBL_AD', 'AddNeuroMed_AD', 'GIFT_AD', 'GIFT_FTD', 'GIFT_PSP_CBD']:
        v = desc[k]
        if v.get('n', 0) > 0:
            ratio_v_glioma = v['mean'] / GLIOMA_MEAN_PCT if GLIOMA_MEAN_PCT > 0 else None
            ratio_v_hc090 = v['mean'] / HC_VAL090_MEAN_PCT if HC_VAL090_MEAN_PCT > 0 else None
            print(f'  {k:18s} mean={v["mean"]:.3f}%  '
                  f'(vs glioma: {ratio_v_glioma:.2f}×, vs VAL-090 HC: {ratio_v_hc090:.2f}×)')
    print()
    
    # 11. Save results
    results = {
        'val_id': 'VAL-091',
        'title': 'Cortical-neuron cfDNA fraction in AD blood — Loyfer/Moss array atlas',
        'card': 'ad-immune-epic v2.1 → v2.2 pending VAL-091 outcome',
        'prereg_seal_sha256': '56c7cac9bb869e4ec2b72a6359f87767035443e0f9b5d34d1a7c848b10053c2f',
        'rng_seed': 20260426,
        'date': '2026-04-26',
        'method': 'NNLS deconvolution against nloyfer/meth_atlas reference_atlas.csv (Loyfer 2023 Nature 613:355). Cortical_neurons row extracted per cohort.',
        'reference_atlas_sha256': '4b97dd2a8ba7bf41008e20703e8e12df731179e95cee50fdc12c4d2c202f05b1',
        'cohorts': {
            'GSE51057_HC_external_reference': {'n': len(hc_external), 'platform': 'HM450', 'role': 'healthy reference (same as VAL-090)'},
            'GSE153712_AIBL':         {'n': len(cn_aibl), 'platform': 'EPIC 850K', 'role': 'AD primary cohort, panel-training'},
            'GSE144858_AddNeuroMed':  {'n': len(cn_anm), 'platform': 'HM450', 'role': 'AD cross-platform replication'},
            'GSE53740_GIFT':          {'n': len(cn_gift), 'platform': 'HM450', 'role': 'AD specificity vs FTD/PSP/CBD'},
        },
        'descriptives_pct': desc,
        'cohens_d_within_cohort': {
            'AIBL_AD_vs_HC': aibl_ad_v_hc,
            'AddNeuroMed_AD_vs_HC': anm_ad_v_hc,
            'GIFT_AD_vs_HC': gift_ad_v_hc,
            'GIFT_FTD_vs_HC': gift_ftd_v_hc,
            'GIFT_PSP_CBD_vs_HC': gift_psp_v_hc,
        },
        'pooled_AD_vs_HC': {
            'pooled_AD_vs_pooled_native_HC': pooled_ad_v_pooled_hc,
            'pooled_AD_vs_GSE51057_external_HC': pooled_v_external,
        },
        'cross_cohort_HC_baseline_pct': hc_means,
        'cross_cohort_HC_fold_range': hc_max / max(hc_min, 1e-9),
        'val090_glioma_anchor_pct': GLIOMA_MEAN_PCT,
        'val090_hc_anchor_pct': HC_VAL090_MEAN_PCT,
        'primary_outcome_label': outcome,
        'primary_outcome_d': primary_d,
        'interpretation_note': (
            'O4_AD_NEURO_NULL is the AD card v2.1 PREDICTION. The card explicitly states '
            '"Stage 2 Moss NNLS for AD is expected NULL — brain tissue not in buffy coat." '
            'VAL-091 tests this prediction with the Loyfer atlas (which has a Cortical_neurons '
            'reference Moss 2018 lacked). O4 confirms the prediction; O1-O3 contradict it; '
            'O5 is biologically anomalous.'
        ),
    }
    
    out_path = '/mnt/user-data/outputs/cookbook_v2.1/ad-immune/VAL-091_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'Saved: {out_path}')
    
    return results

if __name__ == '__main__':
    main()
