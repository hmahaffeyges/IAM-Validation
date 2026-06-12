"""
VAL-113 — Caggiano TIM array-bridged calibration + cardio cohort scoring.
Calibrate on TCGA HM450 sesame n=210 (same cohort as VAL-112 layered Moss+Loyfer
+ HeartRef calibration), then score all three cardio cohorts.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter
from itertools import combinations
import sys

OUT_DIR = Path('/home/claude/iam_repo/Biological_Physics/validation_runs/VAL-113_caggiano')
OUT_DIR.mkdir(exist_ok=True)
ATLAS = '/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/caggiano_tim_cpg_bridged.csv'
TCGA_KIRC = Path('/home/claude/edear_working/VAL-106/calibration_betas/KIRC')
TCGA_PRAD = Path('/home/claude/edear_working/VAL-106/calibration_betas/PRAD')

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


def stats(s):
    v = pd.Series(s).dropna()
    if len(v) == 0:
        return {'mean': None, 'sd': None, 'n': 0}
    return {'mean': float(v.mean()), 'sd': float(v.std()), 'n': int(len(v)),
            'q2.5': float(v.quantile(0.025)), 'q5': float(v.quantile(0.05)),
            'q50': float(v.quantile(0.50)), 'q95': float(v.quantile(0.95)),
            'q97.5': float(v.quantile(0.975))}


def load_tcga(fp):
    df = pd.read_csv(fp, sep='\t', header=None, names=['probe_id', 'beta'])
    return pd.Series(pd.to_numeric(df['beta'], errors='coerce').values,
                     index=df['probe_id'].values)


def score_sample_against_atlas(sample_betas, atlas_df, tile_cols):
    common = sample_betas.index.intersection(atlas_df.index)
    if len(common) == 0:
        return {t: None for t in tile_cols}, 0
    samp = sample_betas.loc[common]
    ref = atlas_df.loc[common]
    out = {}
    for t in tile_cols:
        if t not in ref.columns:
            out[t] = None
            continue
        diffs = (samp - ref[t]).abs().dropna()
        out[t] = float(diffs.mean()) if len(diffs) > 0 else None
    return out, len(common)


def calibration_phase():
    print("=== Phase B: Calibrating Caggiano TIM on TCGA n=210 ===", flush=True)
    atlas = pd.read_csv(ATLAS, index_col=0)
    print(f"  Atlas: {atlas.shape}", flush=True)
    atlas_cpgs = set(atlas.index)

    files = [(f, 'KIRC') for f in sorted(TCGA_KIRC.glob('*.txt'))] + \
            [(f, 'PRAD') for f in sorted(TCGA_PRAD.glob('*.txt'))]
    print(f"  TCGA samples: {len(files)}", flush=True)

    per_sample = []
    for i, (fp, cohort) in enumerate(files):
        if i % 50 == 0:
            print(f"    [{i+1}/{len(files)}]", flush=True)
        s = load_tcga(fp)
        f_ext_a, f_mid_a = f_extreme_middle(s)
        sub = s[s.index.isin(atlas_cpgs)]
        f_ext_b, f_mid_b = f_extreme_middle(sub)
        a_scores, n_int = score_sample_against_atlas(s, atlas, CELL_TYPES)
        rec = {'sample_id': fp.stem, 'cohort': cohort,
               'chk_3_1a_f_extreme': f_ext_a, 'chk_3_1a_f_middle': f_mid_a,
               'caggiano_chk_3_1b_f_extreme': f_ext_b, 'caggiano_chk_3_1b_f_middle': f_mid_b,
               'caggiano_n_intersected': int(n_int)}
        for t in CELL_TYPES:
            rec[f'caggiano_{t}_A'] = a_scores[t]
        per_sample.append(rec)

    df = pd.DataFrame(per_sample)
    df.to_csv(OUT_DIR / 'caggiano_calibration_per_sample.csv', index=False)
    print(f"  Per-sample wrote: {OUT_DIR / 'caggiano_calibration_per_sample.csv'}", flush=True)

    summary = {
        'val_id': 'VAL-113',
        'phase': 'B (calibration on TCGA HM450 sesame n=210)',
        'date': '2026-04-29',
        'atlas_path': ATLAS,
        'atlas_n_cpgs': int(len(atlas)),
        'atlas_n_cell_types': len(CELL_TYPES),
        'cohort': 'TCGA-KIRC + TCGA-PRAD adjacent-normal',
        'n_samples': len(per_sample),
        'chk_3_1a_full_genome': stats(df['chk_3_1a_f_extreme']),
        'caggiano_chk_3_1b_subset': stats(df['caggiano_chk_3_1b_f_extreme']),
        'caggiano_n_intersected': stats(df['caggiano_n_intersected']),
        'per_tile_healthy_floor_A': {t: stats(df[f'caggiano_{t}_A']) for t in CELL_TYPES},
        'sealed_threshold': {
            'chk_3_1b_caggiano_q5': float(df['caggiano_chk_3_1b_f_extreme'].quantile(0.05)),
        },
    }
    with open(OUT_DIR / 'VAL-113_calibration_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Calibration results: {OUT_DIR / 'VAL-113_calibration_results.json'}", flush=True)
    print(f"  CHK-3.1B Caggiano q5: {summary['sealed_threshold']['chk_3_1b_caggiano_q5']:.4f}", flush=True)
    return summary


if __name__ == '__main__':
    calibration_phase()
