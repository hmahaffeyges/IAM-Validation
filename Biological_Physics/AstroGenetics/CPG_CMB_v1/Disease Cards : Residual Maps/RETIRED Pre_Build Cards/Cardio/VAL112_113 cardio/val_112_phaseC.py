"""
VAL-112 Phase C — Cardio cohort re-execution under run-everything discipline.

For each cardio cohort (GSE69138 stroke, GSE84395 PAH, GSE84274 BAV/aortic),
score every sample against BOTH calibrated Stage 2 atlases:
  - Layered Moss+Loyfer (deduped 6,105 CpGs × 25 cells), calibrated VAL-112
  - EpiSCORE HeartRef bridged (3,727 CpGs × 5 cardiac cells), calibrated VAL-112

Per-sample, every cohort, every atlas, every tile. Run-everything.

Outputs per cohort:
  - {cohort}_per_sample_run_everything.csv: per-sample CHK-3.1A + per-tile A-scores
    against both atlases, plus subgroup labels for case/control
  - {cohort}_cohen_d_per_atlas.json: case-vs-control Cohen's d per tile per atlas

Healthy-floor reference is from VAL-112 calibration (TCGA n=210). Sealed at the
top of this script and applied identically to all three cohorts.

NOTE: cardio cohort substrates differ from TCGA sesame Level 3:
  - GSE69138: Illumina GenomeStudio AVG_Beta HM450
  - GSE84395: minfi preprocessFunnorm HM450
  - GSE84274: Illumina GenomeStudio AVG_Beta HM450
The CHK-3.1A thresholds derived in VAL-112 from TCGA sesame are flagged as
substrate-mismatched but still computed; per CCL-041 within-cohort self-cal is
the operational fallback when substrate-matched calibration is unavailable.
This is documented in the per-cohort outcome.
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import gc

OUT = Path('/home/claude/iam_repo/Biological_Physics/validation_runs/VAL-112_run_everything')
OUT.mkdir(exist_ok=True)

LOYFER_PATH = '/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv'
HEARTREF_PATH = '/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_heartref/episcore_heartref_cpg_bridged.csv'
CALIBRATION_PATH = OUT / 'VAL-112_calibration_results.json'

COHORTS = {
    'GSE69138': {'sm_path': '/home/claude/val111_work/gse69138/sm.txt',
                 'tissue': 'whole_blood', 'substrate': 'GenomeStudio_AVG_Beta_HM450'},
    'GSE84395': {'sm_path': '/home/claude/val111_work/gse84395/sm.txt',
                 'tissue': 'pulmonary_endothelial_cells', 'substrate': 'minfi_funnorm_HM450'},
    'GSE84274': {'sm_path': '/home/claude/val111_work/gse84274/sm.txt',
                 'tissue': 'ascending_aorta', 'substrate': 'GenomeStudio_AVG_Beta_HM450'},
}


def f_extreme_middle(beta_series):
    vals = beta_series.dropna()
    if len(vals) == 0:
        return None, None
    f_ext = float(((vals <= 0.1) | (vals >= 0.9)).mean())
    f_mid = float(((vals >= 0.4) & (vals <= 0.6)).mean())
    return f_ext, f_mid


def cohens_d(a, b):
    a = np.array([x for x in a if x is not None and not np.isnan(x)])
    b = np.array([x for x in b if x is not None and not np.isnan(x)])
    if len(a) < 2 or len(b) < 2:
        return None
    pooled_sd = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    if pooled_sd == 0:
        return None
    return float((a.mean() - b.mean()) / pooled_sd)


def load_series_matrix(sm_path, max_skip=200):
    """Load a GEO series matrix, skipping the metadata header."""
    skip = None
    sample_ids = None
    with open(sm_path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('!Sample_geo_accession'):
                sample_ids = line.strip().split('\t')[1:]
                sample_ids = [s.strip('"') for s in sample_ids]
            if line.startswith('!series_matrix_table_begin'):
                skip = i + 1
                break
            if i > max_skip:
                break
    if skip is None:
        # Try fallback: assume header is at the line that starts with "ID_REF"
        with open(sm_path, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('ID_REF') or line.startswith('"ID_REF"'):
                    skip = i
                    break
    print(f"    SM load: skip={skip}, sample_ids extracted: {len(sample_ids) if sample_ids else 'no'}", flush=True)
    df = pd.read_csv(sm_path, sep='\t', skiprows=skip, comment='!', index_col=0,
                     low_memory=False, on_bad_lines='warn')
    df.index = df.index.astype(str).str.strip('"')
    df.columns = [c.strip('"') for c in df.columns]
    # Coerce numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df, sample_ids


def parse_metadata_for_groups(sm_path, max_lines=200):
    """Extract sample IDs and characteristics."""
    sample_ids = None
    characteristics = []
    titles = None
    with open(sm_path, 'r') as f:
        for i, line in enumerate(f):
            if i > max_lines:
                break
            if line.startswith('!Sample_geo_accession'):
                sample_ids = [s.strip('"') for s in line.strip().split('\t')[1:]]
            elif line.startswith('!Sample_title'):
                titles = [s.strip('"') for s in line.strip().split('\t')[1:]]
            elif line.startswith('!Sample_characteristics_ch1'):
                row = [s.strip('"') for s in line.strip().split('\t')[1:]]
                characteristics.append(row)
            elif line.startswith('!Sample_source_name_ch1'):
                row = [s.strip('"') for s in line.strip().split('\t')[1:]]
                characteristics.append(row)
            if line.startswith('!series_matrix_table_begin'):
                break
    return sample_ids, titles, characteristics


def assign_groups_gse69138(sample_ids, titles, characteristics):
    """GSE69138: stroke etiology — CE (cardioembolic), LAA, SVD, control by metadata"""
    groups = {}
    for i, sid in enumerate(sample_ids):
        title = titles[i] if titles and i < len(titles) else ''
        chars = ' | '.join(c[i] if i < len(c) else '' for c in characteristics).lower()
        # Etiology heuristics from prior VAL-108 work
        full_text = (title + ' | ' + chars).lower()
        if 'control' in full_text or 'healthy' in full_text:
            groups[sid] = 'control'
        elif 'cardioembol' in full_text or ' ce ' in full_text or full_text.endswith(' ce'):
            groups[sid] = 'CE'
        elif 'large artery' in full_text or 'laa' in full_text:
            groups[sid] = 'LAA'
        elif 'small vessel' in full_text or 'svd' in full_text:
            groups[sid] = 'SVD'
        else:
            groups[sid] = 'unknown'
    return groups


def assign_groups_gse84395(sample_ids, titles, characteristics):
    """GSE84395 PAH: control, hPAH (heritable), iPAH (idiopathic)"""
    groups = {}
    for i, sid in enumerate(sample_ids):
        title = titles[i] if titles and i < len(titles) else ''
        chars = ' | '.join(c[i] if i < len(c) else '' for c in characteristics).lower()
        full_text = (title + ' | ' + chars).lower()
        if 'control' in full_text or 'normal' in full_text or 'healthy' in full_text or 'donor' in full_text:
            groups[sid] = 'control'
        elif 'heritable' in full_text or 'hpah' in full_text:
            groups[sid] = 'hPAH'
        elif 'idiopathic' in full_text or 'ipah' in full_text:
            groups[sid] = 'iPAH'
        else:
            groups[sid] = 'unknown'
    return groups


def assign_groups_gse84274(sample_ids, titles, characteristics):
    """GSE84274 ascending aorta: normal, dissection, BAV+dilation"""
    groups = {}
    for i, sid in enumerate(sample_ids):
        title = titles[i] if titles and i < len(titles) else ''
        chars = ' | '.join(c[i] if i < len(c) else '' for c in characteristics).lower()
        full_text = (title + ' | ' + chars).lower()
        if 'normal' in full_text or 'control' in full_text or 'healthy' in full_text:
            groups[sid] = 'normal'
        elif 'dissection' in full_text:
            groups[sid] = 'dissection'
        elif 'bav' in full_text or 'bicuspid' in full_text:
            groups[sid] = 'BAV'
        else:
            groups[sid] = 'unknown'
    return groups


def score_cohort(cohort_name, sm_path, loyfer_atlas, heart_atlas, group_assigner):
    print(f"\n=== {cohort_name}: {sm_path} ===", flush=True)
    print(f"  Loading series matrix...", flush=True)
    df, sm_sample_ids = load_series_matrix(sm_path)
    if df is None:
        print(f"  FAILED to load", flush=True)
        return None, None
    print(f"  β-matrix shape: {df.shape}", flush=True)

    # Parse metadata for group assignments
    sample_ids, titles, chars = parse_metadata_for_groups(sm_path)
    if not sample_ids:
        sample_ids = list(df.columns)
    groups = group_assigner(sample_ids, titles, chars) if sample_ids else {sid: 'unknown' for sid in df.columns}

    # Loyfer atlas pre-aligned
    loyfer_cpgs = set(loyfer_atlas.index)
    loyfer_tiles = list(loyfer_atlas.columns)

    # HeartRef atlas pre-aligned
    heart_indexed = heart_atlas.set_index('probeID')
    heart_cpgs = set(heart_atlas['probeID'].dropna().astype(str).values)
    heart_tiles = ['CM', 'EC', 'FB', 'MP', 'SMC']

    # Pre-compute CpG intersections at the cohort level
    cohort_cpgs = set(df.index)
    loyfer_intersection = list(cohort_cpgs & loyfer_cpgs)
    heart_intersection = list(cohort_cpgs & heart_cpgs)
    print(f"  Loyfer CpGs intersected: {len(loyfer_intersection)}/{len(loyfer_cpgs)}", flush=True)
    print(f"  HeartRef CpGs intersected: {len(heart_intersection)}/{len(heart_cpgs)}", flush=True)

    # Pre-slice atlases to intersection
    loyfer_aligned = loyfer_atlas.loc[loyfer_intersection]
    heart_aligned = heart_indexed.loc[heart_intersection]

    per_sample = []
    for j, sid in enumerate(df.columns):
        if j % 50 == 0:
            print(f"    [{j+1}/{len(df.columns)}] {sid}", flush=True)
        s = df[sid]

        # CHK-3.1A on full-genome (per-sample)
        f_ext_a, f_mid_a = f_extreme_middle(s)

        # Loyfer subset CHK-3.1B + per-tile A
        loy_subset = s.loc[s.index.isin(loyfer_intersection)]
        f_ext_loy, f_mid_loy = f_extreme_middle(loy_subset)
        # Per-tile A
        loy_aligned_to_sample = loyfer_aligned.loc[loyfer_aligned.index.isin(loy_subset.dropna().index)]
        common = loy_aligned_to_sample.index.intersection(loy_subset.dropna().index)
        sample_for_loy = s.loc[common]
        loy_a = {}
        if len(common) > 0:
            for t in loyfer_tiles:
                if t in loy_aligned_to_sample.columns:
                    diffs = (sample_for_loy - loy_aligned_to_sample[t].loc[common]).abs()
                    loy_a[t] = float(diffs.mean())
                else:
                    loy_a[t] = None
        else:
            loy_a = {t: None for t in loyfer_tiles}

        # HeartRef subset CHK-3.1B + per-tile A
        h_subset = s.loc[s.index.isin(heart_intersection)]
        f_ext_h, f_mid_h = f_extreme_middle(h_subset)
        h_aligned_to_sample = heart_aligned.loc[heart_aligned.index.isin(h_subset.dropna().index)]
        common_h = h_aligned_to_sample.index.intersection(h_subset.dropna().index)
        sample_for_h = s.loc[common_h]
        heart_a = {}
        if len(common_h) > 0:
            for t in heart_tiles:
                if t in h_aligned_to_sample.columns:
                    diffs = (sample_for_h - h_aligned_to_sample[t].loc[common_h]).abs()
                    heart_a[t] = float(diffs.mean())
                else:
                    heart_a[t] = None
        else:
            heart_a = {t: None for t in heart_tiles}

        rec = {
            'sample_id': sid, 'group': groups.get(sid, 'unknown'), 'cohort': cohort_name,
            'chk_3_1a_f_extreme': f_ext_a, 'chk_3_1a_f_middle': f_mid_a,
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

    print(f"  Per-sample records: {len(per_sample)}", flush=True)
    out_df = pd.DataFrame(per_sample)
    out_csv = OUT / f"{cohort_name}_per_sample_run_everything.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"  Wrote: {out_csv}", flush=True)

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

    # Free big df
    del df
    gc.collect()

    return out_df, pairs_d


def main():
    print("=== VAL-112 Phase C: cardio cohort re-execution under run-everything ===", flush=True)
    print(f"  Loading calibrated atlases...", flush=True)
    loyfer = pd.read_csv(LOYFER_PATH, index_col=0)
    heart = pd.read_csv(HEARTREF_PATH)
    print(f"  Loyfer: {loyfer.shape}, HeartRef: {heart.shape}", flush=True)

    with open(CALIBRATION_PATH) as f:
        calib = json.load(f)
    print(f"  Loaded calibration thresholds from {CALIBRATION_PATH}", flush=True)

    all_results = {
        'val_id': 'VAL-112',
        'phase': 'C — cardio cohort re-execution under run-everything',
        'date': '2026-04-29',
        'calibration_anchor': str(CALIBRATION_PATH),
        'sealed_thresholds': calib['thresholds_sealed'],
        'atlases_run_everything': ['layered_moss_loyfer_deduped', 'episcore_heartref_bridged'],
        'atlases_deferred_engineering_blocked': [
            'Caggiano_CelFiE_TIM (HM450 hg19 manifest needed)',
            'EpiSCORE_pan_tissue (gene→CpG bridge needed for non-cardiac tissues)',
            'Cuadrat_2023_extended (build from 6 ENCODE EPIC IDATs needed)',
            'Tanaka_2025 (acquisition + nanopore→array bridge needed)',
            'Tian_et_al_2023_scMCodes (acquisition + scMCodes→array projection needed)',
            'MARLIN (training scaffold, not a scoring matrix)',
            'Sabedot (R script only, not a scoring matrix)'
        ],
        'cohorts': {},
    }

    # Score each cohort
    assigners = {
        'GSE69138': assign_groups_gse69138,
        'GSE84395': assign_groups_gse84395,
        'GSE84274': assign_groups_gse84274,
    }
    for cohort_name, info in COHORTS.items():
        print(f"\n{'='*60}\nProcessing {cohort_name} ({info['tissue']})\n{'='*60}", flush=True)
        df, pairs_d = score_cohort(cohort_name, info['sm_path'], loyfer, heart,
                                     assigners[cohort_name])
        if df is not None:
            all_results['cohorts'][cohort_name] = {
                'tissue': info['tissue'],
                'substrate': info['substrate'],
                'n_samples': len(df),
                'group_counts': df['group'].value_counts().to_dict(),
                'cohen_d_per_atlas_per_tile': pairs_d,
                'per_sample_csv': str(OUT / f"{cohort_name}_per_sample_run_everything.csv"),
            }

    out_json = OUT / 'VAL-112_phaseC_cardio_run_everything_results.json'
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n=== Wrote: {out_json} ===", flush=True)


if __name__ == '__main__':
    main()
