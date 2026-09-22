#!/usr/bin/env python3
"""
T13 — Secretory-class A-score test on GSE51057 EPIC-Italy
=========================================================
Purpose: at the same >10yr pre-dx window where the Xu-538 immune signal is
loud (d=+1.85 in GSE51057), is the architecturally-correct AFFECTED-TISSUE
class (secretory, for breast) also showing a small departure?

Or only immune?

This is Heath's "house on fire — can we also smell smoke from the bedroom"
test. The framework predicts that pre-malignant secretory drift in breast
tissue should be detectable somewhere; the question is whether 450K bulk
blood arrays can resolve it at all, given that secretory-class cfDNA is a
tiny fraction of bulk blood signal.

Inputs:
  - GSE51057 series matrix (already on disk, SHA verified 828059824b67af46...)
  - 19-CpG secretory panel from VAL-047 deep audit (SEC_CPGS, line 11738 of
    GAPE Evidence Report; pulled from VAL_047_options_1_2.py source)
  - H_min(secretory) = 0.843264 (G-002 MCMC posterior, GAPE_WEB line 71)

Outputs:
  - per-sample CSV with secretory A-score and metadata
  - per-window analysis JSON (mean A by TtD window, Cohen's d vs controls)
  - direct comparison vs the immune A-score on the same samples
"""

import gzip
import json
import math
import os
import re
import sys
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# CANONICAL CONSTANTS — DO NOT MODIFY
# ============================================================================

GSE51057_MATRIX = Path("/home/claude/CrossPopValidation/data/GSE51057_series_matrix.txt.gz")
OUT_DIR         = Path("/home/claude/CrossPopValidation/results/T13_secretory_GSE51057")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# H_min(secretory) from G-002 MCMC posterior — fixed before any cohort analysis
H_MIN_SECRETORY = 0.843264

# 19-CpG secretory panel from VAL-047 deep audit (Phase 4–6, GSE51057)
# Source: VAL_047_options_1_2.py / GAPE Evidence Report line 11738
SEC_CPGS = [
    'cg16867657','cg06639320','cg13552692','cg11807280','cg19283806',
    'cg02580606','cg22454769','cg02228185','cg06691716','cg00846300',
    'cg01127300','cg26521404','cg08262002','cg18181703',
    'cg09809672','cg22736354','cg02489552','cg26203572','cg25382485',
]

RANDOM_SEED = 20260420
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

TtD_WINDOWS = [
    ("0-2 yr",     0.0,   2.0),
    ("2-5 yr",     2.0,   5.0),
    ("5-10 yr",    5.0,  10.0),
    (">10 yr",    10.0, 999.0),
    ("all_pre_dx", 0.0, 999.0),
]

# Strict colon-anchored age regex (footgun #2 from Evidence Report)
AGE_REGEX = re.compile(r'^\s*age\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$', re.IGNORECASE)

# ============================================================================
# CORE FORMULA — bit-identical to Evidence Report and VAL047_tightening_fresh
# ============================================================================

def H_binary(beta):
    """Shannon binary entropy. Returns 0 for β=0,1,NaN — protected."""
    if beta is None:
        return 0.0
    try:
        b = float(beta)
    except Exception:
        return 0.0
    if math.isnan(b) or b <= 0.0 or b >= 1.0:
        return 0.0
    return -b * math.log2(b) - (1.0 - b) * math.log2(1.0 - b)

def A_score_secretory(beta):
    return H_binary(beta) / H_MIN_SECRETORY

def cohens_d(cases, controls):
    cases    = np.asarray(cases,    dtype=float); cases    = cases[~np.isnan(cases)]
    controls = np.asarray(controls, dtype=float); controls = controls[~np.isnan(controls)]
    n1, n2 = len(cases), len(controls)
    if n1 < 2 or n2 < 2: return float("nan")
    s1 = float(np.std(cases,    ddof=1))
    s2 = float(np.std(controls, ddof=1))
    pooled = math.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / (n1 + n2 - 2))
    if pooled == 0: return 0.0
    return float((np.mean(cases) - np.mean(controls)) / pooled)

def perm_p(cases, controls, n_perm=10000):
    cases    = np.asarray(cases,    dtype=float); cases    = cases[~np.isnan(cases)]
    controls = np.asarray(controls, dtype=float); controls = controls[~np.isnan(controls)]
    n1 = len(cases)
    if n1 < 2 or len(controls) < 2: return float("nan")
    obs_d = cohens_d(cases, controls)
    pooled = np.concatenate([cases, controls])
    rng = np.random.default_rng(RANDOM_SEED)
    n_ge = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        new_cases = pooled[:n1]
        new_ctrl  = pooled[n1:]
        if abs(cohens_d(new_cases, new_ctrl)) >= abs(obs_d):
            n_ge += 1
    return (n_ge + 1) / (n_perm + 1)

def boot_ci(cases, controls, n_boot=10000, alpha=0.05):
    cases    = np.asarray(cases,    dtype=float); cases    = cases[~np.isnan(cases)]
    controls = np.asarray(controls, dtype=float); controls = controls[~np.isnan(controls)]
    if len(cases) < 2 or len(controls) < 2: return (float("nan"), float("nan"))
    rng = np.random.default_rng(RANDOM_SEED)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        bc = rng.choice(cases,    size=len(cases),    replace=True)
        bk = rng.choice(controls, size=len(controls), replace=True)
        ds[i] = cohens_d(bc, bk)
    return (float(np.percentile(ds, 100*alpha/2)), float(np.percentile(ds, 100*(1-alpha/2))))

# ============================================================================
# SERIES MATRIX READER (panel-restricted to save memory)
# ============================================================================

def read_series_matrix_panel(path, panel_cpgs):
    """Read GSE matrix; return (sample_meta_df, beta_df_panel_only)."""
    panel = set(panel_cpgs)
    print(f"  Reading {path.name} (streaming, panel-restricted to {len(panel)} CpGs)...", flush=True)

    sample_titles = []
    sample_geos   = []
    characteristics = {}  # gsm -> list of characteristics strings
    sample_ids_for_beta = None

    # First pass — header lines until !series_matrix_table_begin
    with gzip.open(path, 'rt') as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')
            if line.startswith('!Sample_geo_accession'):
                parts = line.split('\t')
                sample_geos = [p.strip().strip('"') for p in parts[1:]]
            elif line.startswith('!Sample_title'):
                parts = line.split('\t')
                sample_titles = [p.strip().strip('"') for p in parts[1:]]
            elif line.startswith('!Sample_characteristics_ch1'):
                parts = line.split('\t')
                vals = [p.strip().strip('"') for p in parts[1:]]
                if not characteristics:
                    characteristics = {gsm: [] for gsm in (sample_geos or [])}
                for gsm, v in zip(sample_geos, vals):
                    if v: characteristics[gsm].append(v)
            elif line.startswith('!series_matrix_table_begin'):
                break

    # Second pass — read matrix table, keep only panel CpGs
    panel_data = {}  # cpg -> list of β values (one per sample, in column order)
    with gzip.open(path, 'rt') as f:
        in_table = False
        for line in f:
            line = line.rstrip('\n').rstrip('\r')
            if line.startswith('!series_matrix_table_begin'):
                in_table = True
                continue
            if not in_table: continue
            if line.startswith('!series_matrix_table_end'): break
            if line.startswith('"ID_REF"') or line.startswith('ID_REF'):
                hdr = line.split('\t')
                sample_ids_for_beta = [h.strip().strip('"') for h in hdr[1:]]
                continue
            if not line: continue
            parts = line.split('\t')
            cpg = parts[0].strip().strip('"')
            if cpg not in panel: continue
            betas = []
            for v in parts[1:]:
                v = v.strip().strip('"')
                try: betas.append(float(v))
                except (ValueError, TypeError): betas.append(float('nan'))
            panel_data[cpg] = betas

    print(f"    panel CpGs recovered: {len(panel_data)} / {len(panel)}", flush=True)
    print(f"    samples in matrix:    {len(sample_ids_for_beta or [])}", flush=True)

    # Build beta_df: rows = panel CpGs, columns = sample IDs
    if not sample_ids_for_beta:
        raise RuntimeError("No sample IDs found in matrix table")
    beta_df = pd.DataFrame(panel_data, index=sample_ids_for_beta).T  # rows=cpg, cols=sample

    # Build meta_df from characteristics
    meta_rows = []
    for gsm in sample_ids_for_beta:
        chars = characteristics.get(gsm, [])
        meta_rows.append({'gsm': gsm, 'characteristics': chars})
    meta_df = pd.DataFrame(meta_rows).set_index('gsm')
    return meta_df, beta_df

# ============================================================================
# METADATA EXTRACTION
# ============================================================================

def extract_fields(characteristics):
    """Extract age, status, ICD, TtD from characteristics list."""
    out = {'age': None, 'gender': None, 'status': None,
           'icd_code': None, 'cancer_site': None, 'ttd_years': None}
    if not characteristics: return out
    for c in characteristics:
        if c is None: continue
        cl = c.lower().strip()
        # age — strict colon-anchored
        m = AGE_REGEX.match(c)
        if m:
            try: out['age'] = float(m.group(1))
            except ValueError: pass
        # gender
        m2 = re.search(r'gender[^:=]*[:=]\s*([a-zA-Z]+)', cl)
        if m2: out['gender'] = m2.group(1).lower()
        # status / disease state
        if re.search(r'(disease state|case/control|status)[^:=]*[:=]', cl):
            after = cl.split(':', 1)[-1].strip() if ':' in cl else ''
            if any(x in after for x in ['control', 'healthy', 'non-cancer']):
                out['status'] = 'control'
            elif any(x in after for x in ['case', 'cancer']):
                out['status'] = 'case'
        # ICD-10
        m3 = re.search(r'icd[^:=]*[:=]\s*([A-Z][0-9]+(?:\.[0-9]+)?)', c, re.IGNORECASE)
        if m3:
            out['icd_code'] = m3.group(1).upper().strip('.')
        if out['icd_code'] is None:
            m4 = re.search(r'\b(C[0-9]{2,3})(?:\.[0-9]+)?\b', c)
            if m4: out['icd_code'] = m4.group(1).upper()
        # cancer site
        m5 = re.search(r'(cancer type|cancer site|tumour site|tumor site|tissue)[^:=]*[:=]\s*(.+)$', cl)
        if m5: out['cancer_site'] = m5.group(2).strip()
        # TtD
        m6 = re.search(r'(time to diagnosis|time_to_diagnosis|ttd|years to diagnosis|follow[- ]up time)[^:=]*[:=]\s*([-0-9]+(?:\.[0-9]+)?)', cl)
        if m6:
            try: out['ttd_years'] = float(m6.group(2))
            except ValueError: pass
    return out

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 78, flush=True)
    print("T13 — Secretory-class A-score on GSE51057 EPIC-Italy", flush=True)
    print("=" * 78, flush=True)
    print(f"  Matrix:           {GSE51057_MATRIX.name}", flush=True)
    print(f"  H_min(secretory): {H_MIN_SECRETORY}", flush=True)
    print(f"  Panel size:       {len(SEC_CPGS)} secretory CpGs (VAL-047 SEC_CPGS)", flush=True)
    print(f"  Random seed:      {RANDOM_SEED}", flush=True)
    print(flush=True)

    meta_df, beta_df = read_series_matrix_panel(GSE51057_MATRIX, SEC_CPGS)

    # Compute per-sample secretory A-score
    print(f"\n  Computing per-sample secretory A-scores ({beta_df.shape[1]} samples)...", flush=True)
    rows = []
    for gsm in beta_df.columns:
        panel_betas = beta_df[gsm].dropna().values
        if len(panel_betas) < 5:
            A_sec = float('nan')
        else:
            A_vals = [A_score_secretory(b) for b in panel_betas]
            A_sec  = float(np.mean(A_vals))
        chars = meta_df.loc[gsm, 'characteristics'] if gsm in meta_df.index else []
        fields = extract_fields(chars)
        fields.update({
            'gsm':           gsm,
            'cohort':        'GSE51057',
            'A_secretory':   A_sec,
            'n_cpgs_used':   int(len(panel_betas)),
        })
        rows.append(fields)

    df = pd.DataFrame(rows)

    # Classify case/control: C50* = breast case; ICD empty/None = control
    def classify(row):
        icd = (row.get('icd_code') or '')
        if icd.startswith('C50'): return 'case'
        if row.get('status') == 'control': return 'control'
        if row.get('status') == 'case': return 'other_cancer'
        if icd == '' or icd is None: return 'control'
        return 'other_cancer'
    df['group'] = df.apply(classify, axis=1)

    # Summary counts
    print(f"\n  Sample classification:", flush=True)
    print(df['group'].value_counts().to_string(), flush=True)

    # Save per-sample CSV
    csv_path = OUT_DIR / 'GSE51057_secretory_per_sample_A.csv'
    df.to_csv(csv_path, index=False)
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    print(f"\n  Per-sample CSV: {csv_path.name}  sha256: {sha[:16]}...", flush=True)

    # Per-window Cohen's d (cases vs controls)
    print(f"\n{'='*78}", flush=True)
    print(f"PRIMARY ANALYSIS — Mean secretory A-score by TtD window vs controls", flush=True)
    print(f"{'='*78}", flush=True)

    controls   = df[df['group'] == 'control']
    ctrl_A     = controls['A_secretory'].dropna().values
    print(f"  Controls (cancer-free):    n={len(ctrl_A)}, mean A_sec = {np.mean(ctrl_A):.4f}, "
          f"sd = {np.std(ctrl_A, ddof=1):.4f}", flush=True)

    cases_all = df[df['group'] == 'case']
    print(f"  Breast cases (any TtD):    n={len(cases_all)}", flush=True)
    print(flush=True)

    window_results = {}
    print(f"  {'Window':<14} {'n_case':>6} {'A_case':>9} {'A_ctrl':>9} {'delta':>8} "
          f"{'d':>8} {'p_perm':>8} {'CI95':>22}", flush=True)
    print(f"  {'-'*14} {'-'*6} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*22}", flush=True)
    for name, lo, hi in TtD_WINDOWS:
        sub = cases_all[(cases_all['ttd_years'].notna()) &
                        (cases_all['ttd_years'] >= lo) &
                        (cases_all['ttd_years'] < hi)]
        case_A = sub['A_secretory'].dropna().values
        if len(case_A) < 2:
            print(f"  {name:<14} {len(case_A):>6}  insufficient n", flush=True)
            window_results[name] = {'n_case': int(len(case_A)), 'note': 'insufficient'}
            continue
        d  = cohens_d(case_A, ctrl_A)
        p  = perm_p(case_A, ctrl_A, n_perm=5000)  # 5k for speed
        lo_ci, hi_ci = boot_ci(case_A, ctrl_A, n_boot=2000)  # 2k for speed
        delta = float(np.mean(case_A) - np.mean(ctrl_A))
        print(f"  {name:<14} {len(case_A):>6} {np.mean(case_A):>9.4f} {np.mean(ctrl_A):>9.4f} "
              f"{delta:>+8.4f} {d:>+8.3f} {p:>8.4f} [{lo_ci:>+5.2f},{hi_ci:>+5.2f}]", flush=True)
        window_results[name] = {
            'n_case':       int(len(case_A)),
            'case_A_mean':  float(np.mean(case_A)),
            'case_A_sd':    float(np.std(case_A, ddof=1)),
            'ctrl_A_mean':  float(np.mean(ctrl_A)),
            'ctrl_A_sd':    float(np.std(ctrl_A, ddof=1)),
            'delta_A':      delta,
            'cohens_d':     d,
            'p_perm_5000':  p,
            'CI95_boot':    [lo_ci, hi_ci],
        }

    # Direct comparison: do we have the immune A-score from prior runs?
    # For the same GSM IDs, the Evidence Report manifest gives us the immune
    # numbers (Phase 9 GSE51057): >10yr d=+1.85, 5-10yr d=+0.76, all pre-dx d=+0.44.
    # We compare side-by-side.
    print(f"\n{'='*78}", flush=True)
    print(f"COMPARISON — secretory (this test) vs immune (Phase 9 published)", flush=True)
    print(f"{'='*78}", flush=True)
    immune_phase9 = {
        '0-2 yr':    {'d': +0.13, 'note': 'reported Phase 9'},
        '2-5 yr':    {'d': +0.34, 'note': 'reported Phase 9'},
        '5-10 yr':   {'d': +0.76, 'note': 'reported Phase 9'},
        '>10 yr':    {'d': +1.85, 'note': 'reported Phase 9 (n=11)'},
        'all_pre_dx':{'d': +0.44, 'note': 'reported Phase 9 (n=146)'},
    }
    print(f"  {'Window':<14} {'sec_d':>10} {'imm_d':>10} {'Δ(imm-sec)':>14}", flush=True)
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*14}", flush=True)
    for name, _, _ in TtD_WINDOWS:
        sec_d = window_results.get(name, {}).get('cohens_d')
        imm_d = immune_phase9.get(name, {}).get('d')
        if sec_d is None or imm_d is None:
            sec_str = f"{sec_d:>+10.3f}" if sec_d is not None else "       —"
            imm_str = f"{imm_d:>+10.3f}" if imm_d is not None else "       —"
            print(f"  {name:<14} {sec_str} {imm_str}", flush=True)
        else:
            diff = imm_d - sec_d
            print(f"  {name:<14} {sec_d:>+10.3f} {imm_d:>+10.3f} {diff:>+14.3f}", flush=True)

    # Write results JSON
    out_json = {
        'cohort':           'GSE51057_EPIC_Italy_Demetriou2013',
        'matrix_sha256':    '828059824b67af46fb022f872ff9f69395e2e99b5975b3101157731a04d98bb0',
        'class':            'secretory',
        'H_min_secretory':  H_MIN_SECRETORY,
        'panel_n_cpgs':     len(SEC_CPGS),
        'panel_source':     'VAL-047 SEC_CPGS (Evidence Report line 11738)',
        'panel_cpgs':       SEC_CPGS,
        'random_seed':      RANDOM_SEED,
        'sample_classification': df['group'].value_counts().to_dict(),
        'controls': {
            'n':    int(len(ctrl_A)),
            'mean': float(np.mean(ctrl_A)),
            'sd':   float(np.std(ctrl_A, ddof=1)),
        },
        'per_window_secretory': window_results,
        'comparison_immune_Phase9_published': immune_phase9,
        'csv_sha256':       sha,
    }
    json_path = OUT_DIR / 'GSE51057_secretory_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f"\n  Results JSON: {json_path}", flush=True)
    print(f"  sha256: {hashlib.sha256(json_path.read_bytes()).hexdigest()}", flush=True)
    print(flush=True)
    print("T13 complete.", flush=True)

if __name__ == "__main__":
    main()
