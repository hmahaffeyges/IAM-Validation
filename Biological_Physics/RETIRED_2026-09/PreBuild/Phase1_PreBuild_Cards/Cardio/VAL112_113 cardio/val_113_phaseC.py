"""
VAL-113 Phase C — Score Caggiano TIM (cg-bridged) across cardio cohorts.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter
from itertools import combinations
import sys
sys.path.insert(0, '/home/claude/iam_repo/Biological_Physics/validation_runs/VAL-112_run_everything')
from val_112_phaseC import (load_series_matrix, parse_metadata_for_groups,
                              assign_groups_gse84395, assign_groups_gse84274)

OUT_DIR = Path('/home/claude/iam_repo/Biological_Physics/validation_runs/VAL-113_caggiano')
ATLAS = '/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/caggiano_tim_cpg_bridged.csv'

CELL_TYPES = ['dendritic', 'endothelial', 'eosinophil', 'erythroblast',
              'macrophage', 'monocyte', 'neutrophil', 'placenta', 'tcell',
              'adipose', 'brain', 'fibroblast', 'heart', 'hepatocyte', 'lung',
              'mammary', 'megakaryocyte', 'skeletal', 'small_intestine']


def f_extreme_middle(s):
    v = s.dropna()
    if len(v) == 0:
        return None, None
    return float(((v <= 0.1) | (v >= 0.9)).mean()), float(((v >= 0.4) & (v <= 0.6)).mean())


def cohens_d(a, b):
    a = np.array([x for x in a if x is not None and not np.isnan(x)])
    b = np.array([x for x in b if x is not None and not np.isnan(x)])
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / pooled) if pooled != 0 else None


def score_cohort(cohort_name, sm_path, atlas, group_assigner):
    print(f"\n=== {cohort_name} ===", flush=True)
    df, _ = load_series_matrix(sm_path)
    sids, titles, chars = parse_metadata_for_groups(sm_path)
    if not sids:
        sids = list(df.columns)
    groups = group_assigner(sids, titles, chars) if sids else {sid: 'unknown' for sid in df.columns}

    atlas_cpgs = set(atlas.index)
    intersection = list(set(df.index) & atlas_cpgs)
    print(f"  Caggiano CpGs intersected: {len(intersection)}/{len(atlas_cpgs)}", flush=True)
    atlas_aligned = atlas.loc[intersection]

    per_sample = []
    for j, sid in enumerate(df.columns):
        if j % 100 == 0:
            print(f"    [{j+1}/{len(df.columns)}] {sid}", flush=True)
        s = df[sid]
        sub = s.loc[s.index.isin(intersection)]
        f_ext, f_mid = f_extreme_middle(sub)
        common = sub.dropna().index
        a_scores = {}
        for t in CELL_TYPES:
            ref_t = atlas_aligned.loc[common, t] if t in atlas_aligned.columns else None
            if ref_t is not None:
                samp_t = s.loc[common]
                diffs = (samp_t - ref_t).abs().dropna()
                a_scores[t] = float(diffs.mean()) if len(diffs) > 0 else None
            else:
                a_scores[t] = None
        rec = {'sample_id': sid, 'group': groups.get(sid, 'unknown'), 'cohort': cohort_name,
               'caggiano_chk_3_1b_f_extreme': f_ext, 'caggiano_chk_3_1b_f_middle': f_mid,
               'caggiano_n_intersected': int(sub.dropna().shape[0])}
        for t in CELL_TYPES:
            rec[f'caggiano_{t}_A'] = a_scores[t]
        per_sample.append(rec)

    out_df = pd.DataFrame(per_sample)
    out_df.to_csv(OUT_DIR / f'{cohort_name}_caggiano_per_sample.csv', index=False)
    print(f"  Wrote: {OUT_DIR / f'{cohort_name}_caggiano_per_sample.csv'}", flush=True)

    group_counts = Counter(out_df['group'])
    valid = [g for g, n in group_counts.items() if g != 'unknown' and n >= 5]
    pairs_d = {}
    for g1, g2 in combinations(valid, 2):
        a = out_df[out_df['group'] == g1]
        b = out_df[out_df['group'] == g2]
        pairs_d[f"{g1}_vs_{g2}"] = {
            'n_g1': int(group_counts[g1]), 'n_g2': int(group_counts[g2]),
            'caggiano': {t: cohens_d(a[f'caggiano_{t}_A'].values, b[f'caggiano_{t}_A'].values)
                         for t in CELL_TYPES},
        }
    return out_df, pairs_d, dict(group_counts)


# ----- GSE69138 chunked -----
def assign_gse69138_from_metadata(sample_ids):
    with open('/home/claude/iam_repo/Biological_Physics/validation_runs/VAL-108/gse69138_metadata.json') as f:
        meta = json.load(f)
    g = {}
    for r in meta:
        sid = r['geo_accession']
        sub = r.get('stroke subtype', '').lower()
        if 'cardioembol' in sub or sub == 'ce':
            g[sid] = 'CE'
        elif 'large' in sub or 'laa' in sub or 'atherosc' in sub:
            g[sid] = 'LAA'
        elif 'small' in sub or 'svd' in sub or 'lacunar' in sub:
            g[sid] = 'SVD'
        elif 'undetermined' in sub or 'cryptog' in sub:
            g[sid] = 'undetermined'
        else:
            g[sid] = f'stroke_{sub}'
    return g


def score_gse69138_chunked(atlas):
    print(f"\n=== GSE69138 chunked ===", flush=True)
    SM = '/home/claude/val111_work/gse69138/sm.txt'
    needed = set(atlas.index)
    print(f"  Atlas CpGs: {len(needed)}", flush=True)

    skip = None
    sample_ids = None
    with open(SM, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('!Sample_geo_accession'):
                sample_ids = [s.strip('"') for s in line.strip().split('\t')[1:]]
            if line.startswith('!series_matrix_table_begin'):
                skip = i + 1
                break
            if i > 200:
                break
    print(f"  skip={skip}, n_samples={len(sample_ids)}", flush=True)

    rows_kept = []
    header_columns = None
    with open(SM, 'r') as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            if line.startswith('!') or line.startswith('"!'):
                continue
            parts = line.rstrip('\n').split('\t')
            if header_columns is None:
                header_columns = [p.strip('"') for p in parts]
                continue
            cpg = parts[0].strip('"')
            if cpg in needed:
                rows_kept.append([cpg] + [parts[j] if j < len(parts) else '' for j in range(1, len(header_columns))])
    print(f"  Rows kept: {len(rows_kept)}", flush=True)

    cols_data = header_columns[1:]
    cpg_col = [r[0] for r in rows_kept]
    data = np.zeros((len(rows_kept), len(cols_data)), dtype=np.float32)
    for ri, r in enumerate(rows_kept):
        for ci in range(len(cols_data)):
            v = r[ci+1]
            try:
                data[ri, ci] = float(v) if v and v != 'NA' else np.nan
            except ValueError:
                data[ri, ci] = np.nan
    df = pd.DataFrame(data, index=cpg_col, columns=cols_data)

    intersection = list(set(df.index) & needed)
    print(f"  Caggiano CpGs intersected: {len(intersection)}/{len(needed)}", flush=True)
    atlas_aligned = atlas.loc[intersection]

    groups = assign_gse69138_from_metadata(list(df.columns))
    per_sample = []
    for j, sid in enumerate(df.columns):
        if j % 100 == 0:
            print(f"    [{j+1}/{len(df.columns)}]", flush=True)
        s = df[sid]
        sub = s.loc[s.index.isin(intersection)]
        f_ext, f_mid = f_extreme_middle(sub)
        common = sub.dropna().index
        a_scores = {}
        for t in CELL_TYPES:
            ref_t = atlas_aligned.loc[common, t] if t in atlas_aligned.columns else None
            if ref_t is not None:
                samp_t = s.loc[common]
                diffs = (samp_t - ref_t).abs().dropna()
                a_scores[t] = float(diffs.mean()) if len(diffs) > 0 else None
            else:
                a_scores[t] = None
        rec = {'sample_id': sid, 'group': groups.get(sid, 'unknown'), 'cohort': 'GSE69138',
               'caggiano_chk_3_1b_f_extreme': f_ext, 'caggiano_chk_3_1b_f_middle': f_mid,
               'caggiano_n_intersected': int(sub.dropna().shape[0])}
        for t in CELL_TYPES:
            rec[f'caggiano_{t}_A'] = a_scores[t]
        per_sample.append(rec)

    out_df = pd.DataFrame(per_sample)
    out_df.to_csv(OUT_DIR / 'GSE69138_caggiano_per_sample.csv', index=False)
    print(f"  Wrote: {OUT_DIR / 'GSE69138_caggiano_per_sample.csv'}", flush=True)

    group_counts = Counter(out_df['group'])
    valid = [g for g, n in group_counts.items() if g != 'unknown' and n >= 5]
    pairs_d = {}
    for g1, g2 in combinations(valid, 2):
        a = out_df[out_df['group'] == g1]
        b = out_df[out_df['group'] == g2]
        pairs_d[f"{g1}_vs_{g2}"] = {
            'n_g1': int(group_counts[g1]), 'n_g2': int(group_counts[g2]),
            'caggiano': {t: cohens_d(a[f'caggiano_{t}_A'].values, b[f'caggiano_{t}_A'].values)
                         for t in CELL_TYPES},
        }
    return out_df, pairs_d, dict(group_counts)


def main():
    print("=== VAL-113 Phase C: Caggiano TIM cardio cohort run-everything ===", flush=True)
    atlas = pd.read_csv(ATLAS, index_col=0)
    print(f"  Atlas: {atlas.shape}", flush=True)

    results = {
        'val_id': 'VAL-113',
        'phase': 'C — cardio cohort scoring',
        'date': '2026-04-29',
        'atlas_id': 'Caggiano_CelFiE_TIM_cardiac_array_bridged_v1',
        'cohorts': {},
    }

    # GSE84395 PAH
    df, pairs_d, gc = score_cohort('GSE84395', '/home/claude/val111_work/gse84395/sm.txt',
                                    atlas, assign_groups_gse84395)
    results['cohorts']['GSE84395'] = {'n_samples': len(df), 'group_counts': gc, 'cohen_d': pairs_d}

    # GSE84274 BAV
    df, pairs_d, gc = score_cohort('GSE84274', '/home/claude/val111_work/gse84274/sm.txt',
                                    atlas, assign_groups_gse84274)
    results['cohorts']['GSE84274'] = {'n_samples': len(df), 'group_counts': gc, 'cohen_d': pairs_d}

    # GSE69138 chunked
    df, pairs_d, gc = score_gse69138_chunked(atlas)
    results['cohorts']['GSE69138'] = {'n_samples': len(df), 'group_counts': gc, 'cohen_d': pairs_d}

    with open(OUT_DIR / 'VAL-113_phaseC_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== Wrote: {OUT_DIR / 'VAL-113_phaseC_results.json'}", flush=True)


if __name__ == '__main__':
    main()
