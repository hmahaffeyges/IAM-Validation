"""
GSE69138 chunked-streaming scorer.

The 2GB sm.txt won't fit in 9GB RAM as a pandas DataFrame. Strategy:
read the matrix line by line (one CpG row at a time), keep only the rows
in (Loyfer ∪ HeartRef) CpG intersection, then transpose into a sample-wise
beta matrix. The intersected matrix is ~10K rows × 589 samples ≈ 50 MB.
"""
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import gc
import csv

OUT = Path('/home/claude/iam_repo/Biological_Physics/validation_runs/VAL-112_run_everything')
LOYFER_PATH = '/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv'
HEARTREF_PATH = '/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_heartref/episcore_heartref_cpg_bridged.csv'
SM_PATH = '/home/claude/val111_work/gse69138/sm.txt'

sys.path.insert(0, str(OUT))
from val_112_phaseC import (f_extreme_middle, cohens_d, parse_metadata_for_groups,
                              assign_groups_gse69138)


def main():
    print("=== Chunked GSE69138 scoring ===", flush=True)
    loyfer = pd.read_csv(LOYFER_PATH, index_col=0)
    heart = pd.read_csv(HEARTREF_PATH)
    heart_indexed = heart.set_index('probeID')

    loyfer_cpgs = set(loyfer.index)
    heart_cpgs = set(heart['probeID'].dropna().astype(str).values)
    needed_cpgs = loyfer_cpgs | heart_cpgs
    print(f"  Loyfer: {len(loyfer_cpgs)}, HeartRef: {len(heart_cpgs)}, union: {len(needed_cpgs)}", flush=True)

    # Find skip line for series_matrix_table_begin
    skip = None
    sample_ids = None
    with open(SM_PATH, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('!Sample_geo_accession'):
                sample_ids = [s.strip('"') for s in line.strip().split('\t')[1:]]
            if line.startswith('!series_matrix_table_begin'):
                skip = i + 1
                break
            if i > 200:
                break
    print(f"  skip={skip}, n_samples={len(sample_ids)}", flush=True)

    # Stream the matrix; collect only rows whose CpG ID is in needed_cpgs
    print(f"  Streaming β matrix, filtering to needed CpGs...", flush=True)
    rows_kept = []
    rows_total = 0
    header_columns = None
    with open(SM_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            if line.startswith('!') or line.startswith('"!') or line.startswith('!series_matrix_table_end'):
                continue
            parts = line.rstrip('\n').split('\t')
            if header_columns is None:
                header_columns = [p.strip('"') for p in parts]
                continue
            cpg = parts[0].strip('"')
            if cpg in needed_cpgs:
                # Keep this row; coerce to numeric in vector form
                row = [cpg] + [parts[j] if j < len(parts) else '' for j in range(1, len(header_columns))]
                rows_kept.append(row)
            rows_total += 1
            if rows_total % 100000 == 0:
                print(f"    scanned {rows_total:,} rows, kept {len(rows_kept):,}", flush=True)
    print(f"  Total scanned: {rows_total:,}, kept: {len(rows_kept):,}", flush=True)

    # Build pandas DataFrame from filtered rows
    cols_data = header_columns[1:]  # sample columns (skip ID_REF)
    cpg_col = [r[0] for r in rows_kept]
    data = np.zeros((len(rows_kept), len(cols_data)), dtype=np.float32)
    for ri, r in enumerate(rows_kept):
        for ci in range(len(cols_data)):
            v = r[ci+1]
            if v and v != 'NA' and v != 'null':
                try:
                    data[ri, ci] = float(v)
                except ValueError:
                    data[ri, ci] = np.nan
            else:
                data[ri, ci] = np.nan

    df = pd.DataFrame(data, index=cpg_col, columns=cols_data)
    print(f"  Filtered β matrix shape: {df.shape}", flush=True)

    # Group assignment
    sample_ids_meta, titles, chars = parse_metadata_for_groups(SM_PATH, max_lines=200)
    if not sample_ids_meta:
        sample_ids_meta = list(df.columns)
    groups = assign_groups_gse69138(sample_ids_meta, titles, chars)

    # CpG intersection (will be subsets of loyfer/heart since we kept the union)
    loyfer_intersection = list(set(df.index) & loyfer_cpgs)
    heart_intersection = list(set(df.index) & heart_cpgs)
    print(f"  Loyfer intersection in cohort: {len(loyfer_intersection)}", flush=True)
    print(f"  HeartRef intersection in cohort: {len(heart_intersection)}", flush=True)

    loyfer_aligned = loyfer.loc[loyfer_intersection]
    heart_aligned = heart_indexed.loc[heart_intersection]
    loyfer_tiles = list(loyfer.columns)
    heart_tiles = ['CM', 'EC', 'FB', 'MP', 'SMC']

    # Score each sample
    per_sample = []
    for j, sid in enumerate(df.columns):
        if j % 100 == 0:
            print(f"    [{j+1}/{len(df.columns)}] {sid}", flush=True)
        s = df[sid]

        # CHK-3.1A on full-genome — n/a here because we only kept needed CpGs.
        # We instead compute CHK-3.1B on the subset directly. Full-genome CHK-3.1A
        # for GSE69138 is established in VAL-108 sealed metadata (n=589).

        # Loyfer CHK-3.1B + per-tile A
        loy_subset = s.loc[s.index.isin(loyfer_intersection)]
        f_ext_loy, f_mid_loy = f_extreme_middle(loy_subset)
        common_loy = loy_subset.dropna().index
        loy_a = {}
        for t in loyfer_tiles:
            ref_t = loyfer_aligned.loc[common_loy, t]
            samp_t = s.loc[common_loy]
            diffs = (samp_t - ref_t).abs()
            valid = diffs.dropna()
            loy_a[t] = float(valid.mean()) if len(valid) > 0 else None

        # HeartRef CHK-3.1B + per-tile A
        h_subset = s.loc[s.index.isin(heart_intersection)]
        f_ext_h, f_mid_h = f_extreme_middle(h_subset)
        common_h = h_subset.dropna().index
        heart_a = {}
        for t in heart_tiles:
            ref_t = heart_aligned.loc[common_h, t]
            samp_t = s.loc[common_h]
            diffs = (samp_t - ref_t).abs()
            valid = diffs.dropna()
            heart_a[t] = float(valid.mean()) if len(valid) > 0 else None

        rec = {
            'sample_id': sid, 'group': groups.get(sid, 'unknown'), 'cohort': 'GSE69138',
            'chk_3_1a_f_extreme': None,  # full-genome unavailable from streaming filter; see VAL-108 sealed metadata
            'chk_3_1a_f_middle': None,
            'loyfer_chk_3_1b_f_extreme': f_ext_loy, 'loyfer_chk_3_1b_f_middle': f_mid_loy,
            'loyfer_n_intersected': int(loy_subset.dropna().shape[0]),
            'heart_chk_3_1b_f_extreme': f_ext_h, 'heart_chk_3_1b_f_middle': f_mid_h,
            'heart_n_intersected': int(h_subset.dropna().shape[0]),
        }
        for t in loyfer_tiles:
            rec[f'loyfer_{t}_A'] = loy_a[t]
        for t in heart_tiles:
            rec[f'heart_{t}_A'] = heart_a[t]
        per_sample.append(rec)

    out_df = pd.DataFrame(per_sample)
    out_csv = OUT / "GSE69138_per_sample_run_everything.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n  Wrote {out_csv}, n={len(out_df)} samples", flush=True)

    # Cohen's d per tile per atlas, by group pair
    from itertools import combinations
    group_counts = out_df['group'].value_counts().to_dict()
    pairs_d = {}
    valid_groups = [g for g in group_counts if g != 'unknown' and group_counts[g] >= 2]
    for g1, g2 in combinations(valid_groups, 2):
        a = out_df[out_df['group'] == g1]
        b = out_df[out_df['group'] == g2]
        pair_key = f"{g1}_vs_{g2}"
        pairs_d[pair_key] = {
            'n_g1': int(group_counts[g1]), 'n_g2': int(group_counts[g2]),
            'loyfer': {t: cohens_d(a[f'loyfer_{t}_A'].values, b[f'loyfer_{t}_A'].values)
                      for t in loyfer_tiles},
            'heartref': {t: cohens_d(a[f'heart_{t}_A'].values, b[f'heart_{t}_A'].values)
                        for t in heart_tiles},
        }

    out_d = OUT / "GSE69138_cohen_d_per_atlas.json"
    with open(out_d, 'w') as f:
        json.dump({'group_counts': group_counts, 'cohen_d_per_atlas_per_tile': pairs_d}, f, indent=2)
    print(f"  Wrote {out_d}", flush=True)


if __name__ == '__main__':
    main()
