#!/usr/bin/env python3
# VAL-065 — prostate-epic Urine Arm Specimen Comparison on GSE119260 (Brikun 2018)
# Pre-registration SHA: f1d1a99770396f217d636dd4e04e9d2162b1ee186bcacb8841b96e95ffcf437d
# Manifest SHA:         1b0eafcf8b34ece8168a3c4d0bf02c4bb20266092fe392db6b3f5698de6ef0ee
#
# Within-patient four-specimen comparison (FFPE benign + FFPE tumor + plasma cfDNA +
# urine sediment) of Xu-538 immune-class A-score on n=4 advanced-stage prostate cancer
# patients (Brikun 2018, GSE119260, Illumina EPIC 850K).
#
# Tests pre-registered hypotheses:
#   H1: urine A-score closer to tumor A-score than plasma A-score is, in >= 3/4 patients
#   H2: urine vs benign paired Cohen's d > 0.3 (positive direction expected)
#   H3: urine per-CpG direction preservation rate >= plasma per-CpG direction preservation rate
#
# IMPORTANT — provenance: All 4 patients have advanced metastatic prostate cancer
# (bone metastases, Gleason 4+4 to 5+5, PSA 10.9 to 1400 ng/mL). This cohort is NOT
# pre-diagnostic. VAL-065 results CANNOT be extrapolated to early-detection populations.
#
# Reproduction:
#   1. Download series matrix: GSE119260_series_matrix.txt.gz from NCBI GEO
#   2. Download Xu-538 panel JSON (panel SHA ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6)
#   3. Run python3 val065_prostate_epic_urine_arm.py
#
# Dependencies: Python 3.6+ stdlib only (math, statistics, hashlib, json, gzip, urllib).
# No numpy, no pandas, no scipy.
#
# Author: Walther / Heath W. Mahaffey
# Date:   2026-04-25

import urllib.request
import json
import gzip
import os
import math
import hashlib
import statistics

# ============================================================================
# Constants — sealed in pre-registration before any β value access
# ============================================================================

H_MIN_IMMUNE = 0.838889
PANEL_SHA = "ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6"
PREREG_SHA = "f1d1a99770396f217d636dd4e04e9d2162b1ee186bcacb8841b96e95ffcf437d"
MANIFEST_SHA = "1b0eafcf8b34ece8168a3c4d0bf02c4bb20266092fe392db6b3f5698de6ef0ee"
QC_MIN_VALID_CPGS = 400
RNG_SEED = 20260425

DATA_DIR = "./val065_data"
DOWNLOADS_DIR = os.path.join(DATA_DIR, "downloads")
SERIES_MATRIX_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE119nnn/GSE119260/matrix/GSE119260_series_matrix.txt.gz"
PANEL_URL = None  # Xu-538 panel is proprietary in v0.1 — supply locally

# Sample-to-(specimen, patient, age) map
SAMPLE_MAP = {
    'GSM3362390': ('benign', 1, 58), 'GSM3362391': ('benign', 2, 66),
    'GSM3362392': ('benign', 3, 76), 'GSM3362393': ('benign', 4, 68),
    'GSM3362394': ('tumor',  1, 58), 'GSM3362395': ('tumor',  2, 66),
    'GSM3362396': ('tumor',  3, 76), 'GSM3362397': ('tumor',  4, 68),
    'GSM3362398': ('plasma', 1, 58), 'GSM3362399': ('plasma', 2, 66),
    'GSM3362400': ('plasma', 3, 76), 'GSM3362401': ('plasma', 4, 68),
    'GSM3362402': ('urine',  1, 58), 'GSM3362403': ('urine',  2, 66),
    'GSM3362404': ('urine',  3, 76), 'GSM3362405': ('urine',  4, 68),
}
PATIENT_PSAS = {1: 1400.0, 2: 10.9, 3: 144.0, 4: 38.98}
PATIENT_GLEASONS = {1: '4+4', 2: '5+4/4+5', 3: '4+5', 4: '5+5'}


# ============================================================================
# Step 1: Download series matrix
# ============================================================================

def download_series_matrix():
    """Retrieve GSE119260 series matrix from NCBI GEO public access."""
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    out_gz = os.path.join(DOWNLOADS_DIR, "GSE119260_series_matrix.txt.gz")
    out_txt = os.path.join(DOWNLOADS_DIR, "GSE119260_series_matrix.txt")
    if os.path.exists(out_txt):
        print(f"Series matrix already present at {out_txt}")
        return out_txt
    print(f"Downloading {SERIES_MATRIX_URL}...")
    req = urllib.request.Request(SERIES_MATRIX_URL, headers={"User-Agent": "VAL-065/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        with open(out_gz, "wb") as f:
            f.write(r.read())
    with gzip.open(out_gz, "rb") as gz:
        with open(out_txt, "wb") as f:
            f.write(gz.read())
    return out_txt


# ============================================================================
# Step 2: Load β matrix at the Xu-538 panel CpGs
# ============================================================================

def load_panel(panel_path):
    """Load Xu-538 panel from JSON. Verify file SHA matches canonical panel SHA."""
    with open(panel_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    if sha != PANEL_SHA:
        raise RuntimeError(f"Panel SHA mismatch: got {sha}, expected {PANEL_SHA}")
    with open(panel_path) as f:
        d = json.load(f)
    return set(d['cpgs'])


def load_betas(series_matrix_path, panel_cpgs):
    """Parse series matrix; restrict to Xu-538 panel; return {gsm: {cpg: beta}}."""
    beta = {gsm: {} for gsm in SAMPLE_MAP}
    gsm_order = None
    in_table = False
    n_total_rows = 0
    with open(series_matrix_path) as f:
        for line in f:
            line = line.rstrip('\r\n')
            if line.startswith('!series_matrix_table_begin'):
                in_table = True; continue
            if line.startswith('!series_matrix_table_end'):
                break
            if not in_table or not line: continue
            parts = line.split('\t')
            if parts[0] == '"ID_REF"':
                gsm_order = [p.strip('"') for p in parts[1:]]
                continue
            if gsm_order is None: continue
            cpg = parts[0].strip('"')
            n_total_rows += 1
            if cpg not in panel_cpgs: continue
            for i, val_str in enumerate(parts[1:]):
                try:
                    v = float(val_str)
                    if 0 < v < 1 and not math.isnan(v):
                        beta[gsm_order[i]][cpg] = v
                except (ValueError, IndexError):
                    pass
    return beta, n_total_rows


# ============================================================================
# Step 3: Score — Shannon entropy → Xu-538 immune-class A-score
# ============================================================================

def shannon(b):
    if b <= 0 or b >= 1: return 0.0
    return -b * math.log2(b) - (1-b) * math.log2(1-b)

def a_score(beta_dict):
    if not beta_dict: return None
    return sum(shannon(b) / H_MIN_IMMUNE for b in beta_dict.values()) / len(beta_dict)


# ============================================================================
# Step 4: Statistics — paired Cohen's d with small-sample (Hedges) correction
# ============================================================================

def paired_d(deltas):
    n = len(deltas)
    if n < 2: return None
    m = statistics.mean(deltas); sd = statistics.stdev(deltas)
    if sd == 0: return {'n': n, 'mean': m, 'sd': 0.0, 'paired_d': float('nan')}
    d = m / sd
    correction = 1.0 - 3.0 / (4 * (n-1) - 1) if n > 2 else 1.0
    se = math.sqrt(1/n + d**2 / (2*n))
    return {'n': n, 'mean': m, 'sd': sd,
            'paired_d': d, 'paired_d_hedges': d * correction,
            'paired_d_ci_95': [d - 1.96*se, d + 1.96*se]}


# ============================================================================
# Step 5: Main analysis
# ============================================================================

def main_analysis(panel_path):
    panel_cpgs = load_panel(panel_path)
    print(f"Xu-538 panel: {len(panel_cpgs)} CpGs (SHA verified: {PANEL_SHA[:12]}...)")

    series_matrix_path = download_series_matrix()
    beta, n_total = load_betas(series_matrix_path, panel_cpgs)
    n_panel_seen = len(beta[list(beta.keys())[0]])
    print(f"Series matrix loaded: {n_total} total CpGs, {n_panel_seen}/{len(panel_cpgs)} Xu-538 measured")

    # QC
    qc_pass = [g for g in SAMPLE_MAP if len(beta[g]) >= QC_MIN_VALID_CPGS]
    print(f"QC: {len(qc_pass)}/{len(SAMPLE_MAP)} samples passed (>={QC_MIN_VALID_CPGS} valid Xu-538 CpGs)")
    if len(qc_pass) != 16:
        print(f"WARNING: Only {len(qc_pass)} samples passed QC. Some patients incomplete.")

    # Per-sample A-score
    A = {}
    for gsm in qc_pass:
        spec, pat, age = SAMPLE_MAP[gsm]
        A[(pat, spec)] = a_score(beta[gsm])

    # M2: within-patient distance
    distances = {'urine': [], 'plasma': []}
    closer_count = {'urine': 0, 'plasma': 0, 'tied': 0}
    per_patient_dist = {}
    for p in [1, 2, 3, 4]:
        if not all((p, s) in A for s in ['benign', 'tumor', 'urine', 'plasma']):
            continue
        d_u = abs(A[(p, 'urine')] - A[(p, 'tumor')])
        d_p = abs(A[(p, 'plasma')] - A[(p, 'tumor')])
        distances['urine'].append(d_u); distances['plasma'].append(d_p)
        per_patient_dist[f'P{p}'] = {'urine_to_tumor': d_u, 'plasma_to_tumor': d_p}
        if d_u < d_p: closer_count['urine'] += 1
        elif d_p < d_u: closer_count['plasma'] += 1
        else: closer_count['tied'] += 1

    # M3, M4
    deltas_u = [A[(p,'urine')] - A[(p,'benign')] for p in [1,2,3,4] if (p,'urine') in A and (p,'benign') in A]
    deltas_p = [A[(p,'plasma')] - A[(p,'benign')] for p in [1,2,3,4] if (p,'plasma') in A and (p,'benign') in A]
    deltas_t = [A[(p,'tumor')] - A[(p,'benign')] for p in [1,2,3,4] if (p,'tumor') in A and (p,'benign') in A]
    m3 = paired_d(deltas_u)
    m4 = paired_d(deltas_p)
    m_t = paired_d(deltas_t)

    # M5: per-CpG direction preservation
    common = set(beta[qc_pass[0]].keys())
    for g in qc_pass: common &= set(beta[g].keys())
    sample_lookup = {(spec, p): g for g, (spec, p, _) in SAMPLE_MAP.items()}
    n_evaluated = 0
    n_urine_match = 0
    n_plasma_match = 0
    for cpg in common:
        t_signs = []; u_signs = []; p_signs = []
        for p in [1,2,3,4]:
            b_b = beta[sample_lookup[('benign', p)]].get(cpg)
            b_t = beta[sample_lookup[('tumor',  p)]].get(cpg)
            b_u = beta[sample_lookup[('urine',  p)]].get(cpg)
            b_p = beta[sample_lookup[('plasma', p)]].get(cpg)
            if None in (b_b, b_t, b_u, b_p): continue
            t_signs.append(1 if b_t > b_b else (-1 if b_t < b_b else 0))
            u_signs.append(1 if b_u > b_b else (-1 if b_u < b_b else 0))
            p_signs.append(1 if b_p > b_b else (-1 if b_p < b_b else 0))
        if len(t_signs) < 4: continue
        td = 1 if sum(t_signs) > 0 else (-1 if sum(t_signs) < 0 else 0)
        ud = 1 if sum(u_signs) > 0 else (-1 if sum(u_signs) < 0 else 0)
        pd = 1 if sum(p_signs) > 0 else (-1 if sum(p_signs) < 0 else 0)
        if td == 0: continue
        n_evaluated += 1
        if ud == td: n_urine_match += 1
        if pd == td: n_plasma_match += 1

    urine_pres_pct = 100.0 * n_urine_match / n_evaluated if n_evaluated else 0
    plasma_pres_pct = 100.0 * n_plasma_match / n_evaluated if n_evaluated else 0

    # Pre-reg outcome decision
    H1 = closer_count['urine'] >= 3
    H2 = m3 and m3['paired_d'] > 0.3 if m3 and not math.isnan(m3.get('paired_d', float('nan'))) else False
    H3 = urine_pres_pct >= plasma_pres_pct
    if H1 and H2 and H3: outcome = 'O1_URINE_VALIDATED_AS_PRIMARY_PROSTATE_SPECIMEN'
    elif H1 and H2 and not H3: outcome = 'O2_URINE_AND_PLASMA_BOTH_VALIDATED'
    elif H2 and not H1: outcome = 'O3_URINE_VALIDATED_AT_LOWER_TIER'
    elif not H2 and (m3 is None or abs(m3.get('paired_d', 0)) < 0.3): outcome = 'O4_URINE_NULL'
    else:
        # Pre-reg explicitly anticipated this case under O5: urine vs benign d magnitude
        # > 0.3 but in NEGATIVE direction. Convene with Heath before card update.
        outcome = 'O5_UNEXPECTED'

    # Print
    print(f"\nM2: urine closer to tumor in {closer_count['urine']}/4, plasma in {closer_count['plasma']}/4, tied {closer_count['tied']}/4")
    if distances['urine']:
        print(f"    mean |urine - tumor| = {statistics.mean(distances['urine']):.5f}")
        print(f"    mean |plasma - tumor| = {statistics.mean(distances['plasma']):.5f}")
    print(f"M3: urine vs benign paired d = {m3['paired_d']:+.4f}")
    print(f"M4: plasma vs benign paired d = {m4['paired_d']:+.4f}")
    print(f"   tumor vs benign paired d = {m_t['paired_d']:+.4f} (reference)")
    print(f"M5: urine direction preservation = {urine_pres_pct:.1f}%, plasma = {plasma_pres_pct:.1f}%")
    print(f"\nH1 ({'PASS' if H1 else 'FAIL'}) H2 ({'PASS' if H2 else 'FAIL'}) H3 ({'PASS' if H3 else 'FAIL'})")
    print(f"PRE-REGISTERED OUTCOME: {outcome}")

    # Save results
    beta_for_sha = {g: sorted(beta[g].items()) for g in sorted(beta) if g in qc_pass}
    beta_sha = hashlib.sha256(json.dumps(beta_for_sha).encode()).hexdigest()

    results = {
        'val_id': 'VAL-065', 'card': 'prostate-epic', 'date': '2026-04-25',
        'cohort': 'GSE119260 (Brikun 2018) — within-patient urine vs plasma vs tumor comparison',
        'platform': 'Illumina EPIC 850K', 'n_patients': 4, 'n_samples_total': 16,
        'n_samples_qc_passed': len(qc_pass),
        'panel': 'Xu-538 immune', 'panel_sha': PANEL_SHA,
        'panel_cpgs_total': len(panel_cpgs),
        'panel_cpgs_measured_in_all_samples': len(common),
        'H_min_immune': H_MIN_IMMUNE,
        'prereg_sha': PREREG_SHA, 'manifest_sha': MANIFEST_SHA,
        'beta_matrix_sha': beta_sha, 'rng_seed': RNG_SEED,
        'M1_per_sample_a_scores': {f'P{p}_{s}': v for (p, s), v in A.items()},
        'M2_within_patient_distance': {
            'urine_closer_to_tumor_in_patients': closer_count['urine'],
            'plasma_closer_to_tumor_in_patients': closer_count['plasma'],
            'tied': closer_count['tied'],
            'mean_dist_urine_tumor': statistics.mean(distances['urine']) if distances['urine'] else None,
            'mean_dist_plasma_tumor': statistics.mean(distances['plasma']) if distances['plasma'] else None,
            'urine_distance_reduction_vs_plasma': (statistics.mean(distances['plasma']) - statistics.mean(distances['urine'])) if distances['urine'] else None,
            'per_patient_distances': per_patient_dist,
        },
        'M3_urine_vs_benign_paired_d': m3,
        'M4_plasma_vs_benign_paired_d': m4,
        'tumor_vs_benign_paired_d_reference': m_t,
        'M5_per_cpg_direction_preservation': {
            'n_cpgs_evaluated': n_evaluated,
            'urine_preservation_rate_pct': urine_pres_pct,
            'plasma_preservation_rate_pct': plasma_pres_pct,
            'urine_minus_plasma_pct': urine_pres_pct - plasma_pres_pct,
            'brikun_2018_reported_urine_pct': 78.63,
            'brikun_2018_reported_plasma_pct': 62.21,
            'note': 'Brikun used hypermethylation overlap on full ~860K probe set; we use Xu-538 panel + per-CpG sign-direction metric on majority across 4 patients.'
        },
        'hypothesis_results': {
            'H1_urine_closer_in_3_of_4': H1,
            'H2_urine_vs_benign_d_above_0_3': H2,
            'H3_urine_direction_preservation_ge_plasma': H3,
        },
        'outcome': outcome,
        'cohort_caveat_n4_advanced_disease': (
            'All 4 patients have advanced metastatic prostate cancer (bone metastases, '
            'Gleason 4+4 to 5+5, PSA 10.9 to 1400 ng/mL). VAL-065 results CANNOT be '
            'extrapolated to early-detection populations. The 78.6% / 62.2% urine vs '
            'plasma overlap reported by Brikun 2018 is on this same n=4 cohort. Larger '
            'urine cohort needed to draw any substrate-vs-substrate conclusions for the '
            'prostate-epic urine arm beyond exploratory.'
        ),
    }

    out_path = os.path.join(DATA_DIR, "VAL-065_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    panel_path = "./xu538_panel.json"  # Supply panel locally; SHA verified at load
    main_analysis(panel_path)
