"""VAL-111 — EpiSCORE HeartRef on three cardio-epic cohorts (corrected parser).

Parser: scan for !Sample_geo_accession line (gives sample IDs), then
!series_matrix_table_begin (data starts on next line, with "ID_REF" header).
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

print(f"VAL-111 started: {datetime.now(timezone.utc).isoformat()}", flush=True)

ATLAS = '/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_heartref/episcore_heartref_cpg_bridged.csv'
atlas = pd.read_csv(ATLAS)
print(f"Atlas: {atlas.shape}", flush=True)

CELL_TYPES = ['CM', 'EC', 'FB', 'MP', 'SMC']
atlas_cpgs = set(atlas['probeID'].values)

tile_cpgs = {}
for ct in CELL_TYPES:
    mask = atlas[ct] > 0
    tile_cpgs[ct] = set(atlas.loc[mask, 'probeID'].values)
    print(f"  {ct}: {len(tile_cpgs[ct])} active CpGs", flush=True)


def parse_series_matrix(sm_path):
    sample_ids = None
    characteristics = []
    sample_titles = None
    table_begin = None
    with open(sm_path) as f:
        for i, line in enumerate(f, start=1):
            if line.startswith('!Sample_geo_accession'):
                vals = line.rstrip('\n').split('\t')[1:]
                sample_ids = [v.strip().strip('"') for v in vals]
            elif line.startswith('!Sample_characteristics_ch1'):
                vals = line.rstrip('\n').split('\t')[1:]
                characteristics.append([v.strip().strip('"') for v in vals])
            elif line.startswith('!Sample_title'):
                vals = line.rstrip('\n').split('\t')[1:]
                sample_titles = [v.strip().strip('"') for v in vals]
            elif line.startswith('!series_matrix_table_begin'):
                table_begin = i
                break
    return sample_ids, characteristics, sample_titles, table_begin


def read_atlas_betas(sm_path, table_begin, sample_ids):
    needed_rows = []
    with open(sm_path) as f:
        for _ in range(table_begin):
            f.readline()
        data_header = f.readline().rstrip('\n').split('\t')
        data_header = [h.strip().strip('"') for h in data_header]
        n_data_cols = len(data_header) - 1
        print(f"  data_header[0]='{data_header[0]}', "
              f"n_sample_cols={n_data_cols}, "
              f"n_sample_ids={len(sample_ids)}", flush=True)
        for line in f:
            if line.startswith('!series_matrix_table_end'):
                break
            row = line.rstrip('\n').split('\t')
            if not row or not row[0]:
                continue
            cpg = row[0].strip().strip('"')
            if cpg in atlas_cpgs:
                needed_rows.append([cpg] + row[1:])
    if not needed_rows:
        return None
    cols = ['cpg'] + (data_header[1:] if n_data_cols == len(sample_ids) else sample_ids)
    expected_len = len(cols)
    fixed = []
    for r in needed_rows:
        if len(r) == expected_len:
            fixed.append(r)
        elif len(r) > expected_len:
            fixed.append(r[:expected_len])
        else:
            fixed.append(r + [''] * (expected_len - len(r)))
    df = pd.DataFrame(fixed, columns=cols).set_index('cpg')
    df = df.apply(lambda c: pd.to_numeric(c, errors='coerce'))
    return df


def score_cohort(cohort_name, sm_path, outdir):
    print(f"\n=== {cohort_name} ===", flush=True)
    sample_ids, chars, titles, table_begin = parse_series_matrix(sm_path)
    print(f"sample_ids: {len(sample_ids) if sample_ids else 0}, "
          f"chars_lines: {len(chars)}, table_begin: {table_begin}", flush=True)
    if not sample_ids or not table_begin:
        return None, 0
    print("Reading β matrix (atlas CpGs only)...", flush=True)
    beta_df = read_atlas_betas(sm_path, table_begin, sample_ids)
    if beta_df is None:
        return None, 0
    print(f"β matrix: {beta_df.shape}", flush=True)

    results = {'sample': list(beta_df.columns)}
    for ct in CELL_TYPES:
        ct_in_cohort = list(tile_cpgs[ct] & set(beta_df.index))
        if len(ct_in_cohort) < 50:
            results[f'A_{ct}'] = [np.nan] * len(beta_df.columns)
            continue
        means = beta_df.loc[ct_in_cohort].mean(axis=0, skipna=True)
        results[f'A_{ct}'] = means.reindex(beta_df.columns).tolist()
        print(f"  {ct}: {len(ct_in_cohort)} CpGs, "
              f"cohort mean A = {np.nanmean(results[f'A_{ct}']):.4f}", flush=True)

    df = pd.DataFrame(results)
    if titles and len(titles) == len(df):
        df['sample_title'] = titles
    for idx, vals in enumerate(chars):
        if len(vals) == len(df):
            df[f'char_{idx}'] = vals

    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f'val111_{cohort_name}_per_sample.csv'
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}", flush=True)
    return df, beta_df.shape[0]


def stratify(df, search_terms, target_col):
    char_cols = [c for c in df.columns if c.startswith('char_')] + (
        ['sample_title'] if 'sample_title' in df.columns else [])
    def assign(row):
        for c in char_cols:
            v = str(row[c]).lower() if pd.notna(row[c]) else ''
            for term in search_terms:
                if term.lower() in v:
                    return str(row[c])
        return 'unknown'
    df = df.copy()
    df[target_col] = df.apply(assign, axis=1)
    grp = df.groupby(target_col, dropna=False)
    out = {}
    for name, g in grp:
        out[str(name)] = {ct: float(g[f'A_{ct}'].mean()) for ct in CELL_TYPES}
        out[str(name)]['n'] = len(g)
    return out


OUT = Path('/home/claude/iam_repo/Biological_Physics/validation_runs/VAL-111')
results_summary = {}

df1, n1 = score_cohort('GSE69138', '/home/claude/val111_work/gse69138/sm.txt', OUT)
results_summary['GSE69138'] = {'atlas_cpgs_intersected': n1, 'n_samples': len(df1) if df1 is not None else 0}

df2, n2 = score_cohort('GSE84395', '/home/claude/val111_work/gse84395/sm.txt', OUT)
results_summary['GSE84395'] = {'atlas_cpgs_intersected': n2, 'n_samples': len(df2) if df2 is not None else 0}

df3, n3 = score_cohort('GSE84274', '/home/claude/val111_work/gse84274/sm.txt', OUT)
results_summary['GSE84274'] = {'atlas_cpgs_intersected': n3, 'n_samples': len(df3) if df3 is not None else 0}

print("\n=== STRATIFIED SUMMARIES ===", flush=True)
stratified = {}

if df1 is not None:
    stratified['GSE69138'] = {}
    stratified['GSE69138']['by_disease_state'] = stratify(
        df1, ['disease state:', 'sample type:'], 'state')
    stratified['GSE69138']['by_stroke_subtype'] = stratify(
        df1, ['stroke subtype:'], 'subtype')
    floor = {}
    for ct in CELL_TYPES:
        m = float(df1[f'A_{ct}'].mean())
        floor[ct] = {'mean': m, 'breach': m > 0.10}
    stratified['GSE69138']['blood_floor_assessment'] = floor

if df2 is not None:
    stratified['GSE84395'] = {'by_group': stratify(
        df2, ['disease', 'group:', 'subtype', 'patient', 'control', 'pah'], 'group')}

if df3 is not None:
    stratified['GSE84274'] = {'by_group': stratify(
        df3, ['disease', 'group:', 'condition', 'normal', 'dissect', 'aort'], 'group')}

# Outcome
if any(r['atlas_cpgs_intersected'] < 500 for r in results_summary.values()):
    outcome = 'O4_BRIDGE_FAILURE'
    rationale = 'At least one cohort had < 500 atlas CpGs after intersection.'
else:
    tissue_disc = {}
    for cohort_name, sdict in [('GSE84395', stratified.get('GSE84395', {})),
                                ('GSE84274', stratified.get('GSE84274', {}))]:
        bg = sdict.get('by_group', {})
        if len(bg) >= 2:
            for ct in CELL_TYPES:
                vals = [v[ct] for v in bg.values() if ct in v]
                if len(vals) >= 2:
                    tissue_disc[f'{cohort_name}_{ct}_range'] = max(vals) - min(vals)
    any_tissue_disc = any(v >= 0.10 for v in tissue_disc.values())
    blood_breach = False
    if 'GSE69138' in stratified:
        blood_breach = any(v['breach'] for v in stratified['GSE69138']['blood_floor_assessment'].values())
    if any_tissue_disc and not blood_breach:
        outcome = 'O1_TILE_DISCRIMINATION_OBSERVED'
        rationale = '>=1 cardiac tile shows >=0.10 A-score range in tissue cohort, blood floor intact.'
    elif any_tissue_disc and blood_breach:
        outcome = 'O2_PARTIAL_DISCRIMINATION'
        rationale = 'Tissue discrimination present but blood-floor breach in GSE69138.'
    else:
        outcome = 'O3_TISSUE_FLOOR_DOMINATED'
        rationale = 'No tile shows >=0.10 A-score range in any tissue cohort.'
    stratified['tissue_discrimination_ranges'] = tissue_disc

results_json = {
    'val_id': 'VAL-111',
    'sealed_at': '2026-04-29',
    'atlas': 'EpiSCORE_HeartRef',
    'atlas_sha256': 'bf6431f66749f02a616560764af3fdd0adc70b03bca96b2a13b6221bbd847c83',
    'cohorts': results_summary,
    'stratified': stratified,
    'outcome': outcome,
    'rationale': rationale,
    'completed_at': datetime.now(timezone.utc).isoformat(),
}
with open(OUT / 'results.json', 'w') as f:
    json.dump(results_json, f, indent=2, default=str)

print(f"\nOUTCOME: {outcome}", flush=True)
print(f"Rationale: {rationale}", flush=True)
print(f"Results: {OUT / 'results.json'}", flush=True)
print(f"VAL-111 completed: {datetime.now(timezone.utc).isoformat()}", flush=True)
