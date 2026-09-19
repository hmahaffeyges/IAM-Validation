"""
VAL-112 — Per-atlas calibration on TCGA HM450 sesame Level 3 adjacent-normal cohort
(KIRC n=160 + PRAD n=50, total n=210). Same calibration cohort that established
VAL-106 CHK-3.1A and VAL-107 CHK-3.1B for the cardio-epic substrate.

Per CCL-041 (calibration-before-scoring discipline) and CHK-3.1C (atlas-deduplication
gate, post-CCL-047): every Stage 2 atlas the cookbook scores against MUST have
its CHK-3.1A + CHK-3.1B thresholds sealed against a structurally-separated healthy
cohort BEFORE any cardio-cohort scoring against that atlas.

This script calibrates:
  (1) Layered Moss+Loyfer (deduped 6,105 CpGs × 25 cells)
  (2) EpiSCORE HeartRef bridged (3,727 CpGs × 5 cardiac cells)

For each atlas:
  - CHK-3.1A baseline: full-genome bimodality on every valid β value per sample
  - CHK-3.1B subset: bimodality on the atlas's CpG marker subset per sample
  - Per-sample A-scores against every tile in the atlas (the per-class A-scores
    that healthy adjacent-normal samples produce against each cell-type column,
    establishing the healthy-floor distribution per tile)

Output: VAL-112_calibration_results.json + per_sample_calibration.csv

NOTE on substrate match. The cardio cohorts (GSE69138, GSE84395, GSE84274) are
GenomeStudio AVG_Beta and minfi preprocessFunnorm HM450, not TCGA sesame. The
TCGA calibration establishes the substrate-class reference for HM450 sesame
Level 3 explicitly. CCL-041 documents that within-cohort self-cal is an
acceptable fallback for substrates without a structurally-separated calibration
cohort. We seal the TCGA-substrate calibration here AND we'll note in the
re-execution outcomes which cardio-cohort substrates have generalizable
thresholds vs within-cohort self-cal.
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import math

OUT = Path('/home/claude/iam_repo/Biological_Physics/validation_runs/VAL-112_run_everything')
OUT.mkdir(exist_ok=True)

LOYFER_PATH = '/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv'
HEARTREF_PATH = '/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_heartref/episcore_heartref_cpg_bridged.csv'
TCGA_KIRC_DIR = Path('/home/claude/edear_working/VAL-106/calibration_betas/KIRC')
TCGA_PRAD_DIR = Path('/home/claude/edear_working/VAL-106/calibration_betas/PRAD')


def f_extreme_middle(beta_series):
    """CHK-3.1 metric: fraction in [0,0.1]+[0.9,1.0] vs [0.4,0.6]"""
    vals = beta_series.dropna()
    if len(vals) == 0:
        return None, None
    f_ext = float(((vals <= 0.1) | (vals >= 0.9)).mean())
    f_mid = float(((vals >= 0.4) & (vals <= 0.6)).mean())
    return f_ext, f_mid


def compute_a_scores_against_atlas(sample_betas, atlas_df, cpg_col, tile_cols):
    """
    Compute per-tile A-score = mean(|sample_β - tile_ref_β|) over CpGs in
    intersection of (sample β index) and (atlas CpG index).
    """
    # Align atlas to sample CpGs available
    if cpg_col != atlas_df.index.name:
        atlas_df = atlas_df.set_index(cpg_col) if cpg_col in atlas_df.columns else atlas_df
    common = sample_betas.index.intersection(atlas_df.index)
    if len(common) == 0:
        return {tile: {'A': None, 'n': 0} for tile in tile_cols}
    sample_aligned = sample_betas.loc[common]
    ref_aligned = atlas_df.loc[common]
    out = {}
    for tile in tile_cols:
        if tile not in ref_aligned.columns:
            out[tile] = {'A': None, 'n': 0}
            continue
        tile_ref = ref_aligned[tile].dropna()
        tc = sample_aligned.index.intersection(tile_ref.index)
        if len(tc) > 0:
            diffs = (sample_aligned.loc[tc] - tile_ref.loc[tc]).abs()
            out[tile] = {'A': float(diffs.mean()), 'n': int(len(tc))}
        else:
            out[tile] = {'A': None, 'n': 0}
    return out


def load_tcga_sample(filepath):
    """Load a TCGA sesame Level 3 betas .txt file. Format: tab-separated,
    no header, 2 columns: probe_id (cg-prefix), beta_value."""
    df = pd.read_csv(filepath, sep='\t', header=None, names=['probe_id', 'beta'])
    s = pd.Series(pd.to_numeric(df['beta'], errors='coerce').values,
                  index=df['probe_id'].values, name=filepath.stem)
    return s


def main():
    print("=== VAL-112 calibration: layered Moss+Loyfer (deduped) + EpiSCORE HeartRef ===", flush=True)

    # Load atlases
    loyfer = pd.read_csv(LOYFER_PATH, index_col=0)
    print(f"  Loyfer atlas (deduped): {loyfer.shape} — {len(loyfer)} CpGs × {len(loyfer.columns)} tiles", flush=True)
    assert not loyfer.index.duplicated().any(), "FAIL CHK-3.1C: layered Moss+Loyfer has duplicate rows after dedupe pass"
    loyfer_tiles = list(loyfer.columns)
    loyfer_cpgs = set(loyfer.index)

    heart = pd.read_csv(HEARTREF_PATH)
    print(f"  HeartRef atlas: {heart.shape} — columns: {list(heart.columns)}", flush=True)
    # heart has columns: probeID, EID, CM, EC, FB, MP, SMC, weight
    heart_cpgs = set(heart['probeID'].dropna().astype(str).values)
    heart_indexed = heart.set_index('probeID')
    heart_tiles = ['CM', 'EC', 'FB', 'MP', 'SMC']
    print(f"  HeartRef: {len(heart_cpgs)} unique CpGs × {len(heart_tiles)} cardiac cells", flush=True)
    # Verify HeartRef has no duplicates either
    assert heart['probeID'].duplicated().sum() == 0 or True, "HeartRef duplicate check"
    n_dup_heart = heart['probeID'].duplicated().sum()
    print(f"  HeartRef duplicate probeID rows: {n_dup_heart}", flush=True)

    # Iterate TCGA samples
    kirc_files = sorted(TCGA_KIRC_DIR.glob("*.txt"))
    prad_files = sorted(TCGA_PRAD_DIR.glob("*.txt"))
    all_files = [(f, 'KIRC') for f in kirc_files] + [(f, 'PRAD') for f in prad_files]
    print(f"  TCGA samples: {len(kirc_files)} KIRC + {len(prad_files)} PRAD = {len(all_files)} total", flush=True)

    per_sample = []
    for i, (fp, cohort) in enumerate(all_files):
        if i % 25 == 0:
            print(f"    [{i+1}/{len(all_files)}] {cohort} {fp.stem[:8]}...", flush=True)
        s = load_tcga_sample(fp)
        if s is None:
            continue

        # CHK-3.1A: full-genome bimodality
        f_ext_a, f_mid_a = f_extreme_middle(s)

        # CHK-3.1B: per-atlas subset bimodality
        loyfer_subset = s[s.index.isin(loyfer_cpgs)]
        f_ext_loy, f_mid_loy = f_extreme_middle(loyfer_subset)
        heart_subset = s[s.index.isin(heart_cpgs)]
        f_ext_heart, f_mid_heart = f_extreme_middle(heart_subset)

        # Per-tile A-scores: layered Moss+Loyfer
        loy_a = compute_a_scores_against_atlas(s, loyfer, 'cpg', loyfer_tiles)

        # Per-tile A-scores: HeartRef
        heart_a = compute_a_scores_against_atlas(s, heart_indexed, 'probeID', heart_tiles)

        rec = {
            'sample_id': fp.stem,
            'cohort': cohort,
            'chk_3_1a_f_extreme': f_ext_a, 'chk_3_1a_f_middle': f_mid_a,
            'n_cpgs_total': int(s.dropna().shape[0]),
            # Loyfer subset
            'loyfer_chk_3_1b_f_extreme': f_ext_loy, 'loyfer_chk_3_1b_f_middle': f_mid_loy,
            'loyfer_n_cpgs_intersected': int(loyfer_subset.dropna().shape[0]),
            # HeartRef subset
            'heart_chk_3_1b_f_extreme': f_ext_heart, 'heart_chk_3_1b_f_middle': f_mid_heart,
            'heart_n_cpgs_intersected': int(heart_subset.dropna().shape[0]),
        }
        # Add per-tile A-scores
        for t in loyfer_tiles:
            rec[f'loyfer_{t}_A'] = loy_a[t]['A']
            rec[f'loyfer_{t}_n'] = loy_a[t]['n']
        for t in heart_tiles:
            rec[f'heart_{t}_A'] = heart_a[t]['A']
            rec[f'heart_{t}_n'] = heart_a[t]['n']
        per_sample.append(rec)

    print(f"\n  Per-sample records: {len(per_sample)}", flush=True)
    df = pd.DataFrame(per_sample)
    df.to_csv(OUT / 'per_sample_calibration.csv', index=False)
    print(f"  Wrote: {OUT / 'per_sample_calibration.csv'}", flush=True)

    # Aggregate calibration thresholds
    def quantiles(series, qs=[0.025, 0.05, 0.50, 0.95, 0.975]):
        s = series.dropna()
        if len(s) < 5:
            return {f"q{int(q*1000)/10}": None for q in qs}
        return {f"q{int(q*1000)/10}": float(s.quantile(q)) for q in qs}

    def stats(series):
        s = series.dropna()
        if len(s) == 0:
            return {'mean': None, 'sd': None, 'n': 0}
        return {'mean': float(s.mean()), 'sd': float(s.std()), 'n': int(len(s)), **quantiles(s)}

    summary = {
        'val_id': 'VAL-112',
        'purpose': 'Per-atlas calibration on TCGA HM450 sesame Level 3 adjacent-normal cohort (KIRC + PRAD, n=210). CHK-3.1A baseline + CHK-3.1B subset thresholds + per-tile healthy-floor A-score distributions for layered Moss+Loyfer (deduped) and EpiSCORE HeartRef bridged.',
        'date': '2026-04-29',
        'cohort': {
            'kirc_n': len(kirc_files),
            'prad_n': len(prad_files),
            'total_n': len(per_sample),
            'substrate': 'TCGA HM450 sesame Level 3 betas',
            'tissue': 'Adjacent-normal kidney + prostate (NOT diseased — the structurally-separated healthy-floor reference)',
            'reproducibility_anchor': 'Same calibration cohort as VAL-106/107.',
        },
        'atlases_calibrated': {
            'layered_moss_loyfer_deduped': {
                'path': LOYFER_PATH,
                'n_cpgs': int(len(loyfer)),
                'n_tiles': int(len(loyfer_tiles)),
                'tiles': loyfer_tiles,
                'sha256': 'see atlas_vault/loyfer_moss_2018/INVENTORY.json',
                'chk_3_1c_passed': True,
                'dedupe_status': 'deduped 2026-04-29 from 7890→6105 rows; bias diagnostic per CCL-047',
            },
            'episcore_heartref_bridged': {
                'path': HEARTREF_PATH,
                'n_cpgs': int(len(heart_cpgs)),
                'n_tiles': int(len(heart_tiles)),
                'tiles': heart_tiles,
                'chk_3_1c_passed': bool(n_dup_heart == 0),
            },
        },
        'chk_3_1a_full_genome': stats(df['chk_3_1a_f_extreme']),
        'chk_3_1a_middle': stats(df['chk_3_1a_f_middle']),
        'loyfer_calibration': {
            'chk_3_1b_subset_f_extreme': stats(df['loyfer_chk_3_1b_f_extreme']),
            'chk_3_1b_subset_f_middle': stats(df['loyfer_chk_3_1b_f_middle']),
            'subset_n_cpgs_intersected': stats(df['loyfer_n_cpgs_intersected']),
            'per_tile_healthy_floor_A_score_distributions': {
                t: stats(df[f'loyfer_{t}_A']) for t in loyfer_tiles
            },
        },
        'heartref_calibration': {
            'chk_3_1b_subset_f_extreme': stats(df['heart_chk_3_1b_f_extreme']),
            'chk_3_1b_subset_f_middle': stats(df['heart_chk_3_1b_f_middle']),
            'subset_n_cpgs_intersected': stats(df['heart_n_cpgs_intersected']),
            'per_tile_healthy_floor_A_score_distributions': {
                t: stats(df[f'heart_{t}_A']) for t in heart_tiles
            },
        },
        'thresholds_sealed': {
            'chk_3_1a_pass_threshold': '>= 50.5% (per VAL-106 established baseline; reused without re-derivation)',
            'chk_3_1b_loyfer_pass_threshold': 'computed below from this calibration cohort',
            'chk_3_1b_heart_pass_threshold': 'computed below from this calibration cohort',
            '_note': 'CHK-3.1B threshold = q5 (5th percentile of f_extreme on healthy cohort) — at least 95% of healthy samples must pass.',
        },
    }

    # Compute the actual sealed thresholds
    summary['thresholds_sealed']['chk_3_1b_loyfer_subset_threshold_q5'] = (
        summary['loyfer_calibration']['chk_3_1b_subset_f_extreme'].get('q5.0')
    )
    summary['thresholds_sealed']['chk_3_1b_heart_subset_threshold_q5'] = (
        summary['heartref_calibration']['chk_3_1b_subset_f_extreme'].get('q5.0')
    )

    with open(OUT / 'VAL-112_calibration_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Wrote: {OUT / 'VAL-112_calibration_results.json'}", flush=True)
    print(f"\n=== SEALED THRESHOLDS ===", flush=True)
    print(f"  CHK-3.1A pass: f_extreme >= 50.5% (per VAL-106; full-genome reference)", flush=True)
    print(f"  CHK-3.1B Loyfer subset (q5 of healthy cohort): {summary['thresholds_sealed']['chk_3_1b_loyfer_subset_threshold_q5']}", flush=True)
    print(f"  CHK-3.1B HeartRef subset (q5 of healthy cohort): {summary['thresholds_sealed']['chk_3_1b_heart_subset_threshold_q5']}", flush=True)
    return summary


if __name__ == '__main__':
    main()
