"""
Build unified outcome.md + results.json for VAL-112 + VAL-113 combined.

3 calibrated atlases: layered Moss+Loyfer (deduped), EpiSCORE HeartRef bridged,
                       Caggiano CelFiE TIM array-bridged.
3 cardio cohorts: GSE69138 (whole blood, stroke etiology),
                  GSE84395 (PEC, PAH variants),
                  GSE84274 (ascending aorta, BAV/dissection).

Calibration: TCGA HM450 sesame Level 3 adjacent-normal n=210 (KIRC + PRAD),
same cohort that anchored VAL-106/107.
"""
import pandas as pd
import json
from pathlib import Path
from collections import Counter
from itertools import combinations
import numpy as np

UNIFIED_OUT = Path('/home/claude/iam_repo/Biological_Physics/validation_runs/VAL-112_113_unified')
UNIFIED_OUT.mkdir(exist_ok=True)


def cohens_d(a, b):
    a = np.array([x for x in a if x is not None and not np.isnan(x)])
    b = np.array([x for x in b if x is not None and not np.isnan(x)])
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / pooled) if pooled != 0 else None


def load_per_sample(path):
    return pd.read_csv(path) if Path(path).exists() else None


def main():
    # Load all per-sample CSVs
    csvs = {}
    base_112 = '/home/claude/iam_repo/Biological_Physics/validation_runs/VAL-112_run_everything'
    base_113 = '/home/claude/iam_repo/Biological_Physics/validation_runs/VAL-113_caggiano'
    for cohort in ['GSE69138', 'GSE84395', 'GSE84274']:
        df_a = load_per_sample(f'{base_112}/{cohort}_per_sample_run_everything.csv')
        df_b = load_per_sample(f'{base_113}/{cohort}_caggiano_per_sample.csv')
        if df_a is None or df_b is None:
            print(f"  Missing data for {cohort}")
            continue
        # Merge on sample_id
        merged = pd.merge(df_a, df_b[['sample_id'] +
                                      [c for c in df_b.columns if c.startswith('caggiano_')]],
                          on='sample_id', how='inner')
        # Use group from df_a if available
        if 'group_x' in merged.columns:
            merged['group'] = merged['group_x']
        csvs[cohort] = merged
        merged.to_csv(UNIFIED_OUT / f'{cohort}_unified_per_sample.csv', index=False)
        print(f"  {cohort}: n={len(merged)}")

    # Build unified Cohen's d table
    unified = {
        'val_id': 'VAL-112+113 unified',
        'date': '2026-04-29',
        'description': 'Cardio-epic run-everything: 3 calibrated Stage 2 atlases × 3 cardio cohorts',
        'calibration_anchor': 'TCGA HM450 sesame Level 3 adjacent-normal n=210 (KIRC + PRAD); same cohort as VAL-106/107',
        'atlases_run_everything': {
            'layered_moss_loyfer_deduped': {
                'n_cpgs': 6105, 'n_tiles': 25,
                'tiles': 'Adipocytes, B-cells_EPIC, Bladder_Ep_cells, CD4T-cells_EPIC, CD8T-cells_EPIC, Cervix_Ep_cells, Colon_epithelial_cells, Cortical_neurons, Erythrocyte_progenitors, Eso_Ep_cells, Gastric_epithelial_cells, Head_Neck_Ep_cells, Hepatocytes, Kidney, Left_atrium, Lung_cells, Monocytes_EPIC, NK-cells_EPIC, Neutrophils_EPIC, Pancreatic_acinar_cells, Pancreatic_beta_cells, Pancreatic_duct_cells, Prostate, Vascular_endothelial_cells, Breast',
                'calibration_VAL': 'VAL-112',
                'chk_3_1b_q5_threshold': 0.6839,
            },
            'episcore_heartref_bridged': {
                'n_cpgs': 3727, 'n_tiles': 5,
                'tiles': 'CM (cardiomyocyte), EC (endothelial), FB (fibroblast), MP (macrophage), SMC (smooth muscle)',
                'calibration_VAL': 'VAL-112',
                'chk_3_1b_q5_threshold': 0.4283,
            },
            'caggiano_tim_array_bridged': {
                'n_cpgs': 254, 'n_tiles': 19,
                'tiles': 'dendritic, endothelial, eosinophil, erythroblast, macrophage, monocyte, neutrophil, placenta, tcell, adipose, brain, fibroblast, heart, hepatocyte, lung, mammary, megakaryocyte, skeletal, small_intestine',
                'calibration_VAL': 'VAL-113',
                'chk_3_1b_q5_threshold': 0.5779,
                'bridge_method': 'HM450 hg19 manifest CpG-in-region intersection from Caggiano et al. 2021 WGBS regions; aggregated by mean for multi-region CpGs',
            },
        },
        'cohorts': {},
    }

    LOYFER_TILES = ['Vascular_endothelial_cells', 'Left_atrium', 'Adipocytes', 'Cortical_neurons',
                    'Hepatocytes', 'Monocytes_EPIC', 'Neutrophils_EPIC', 'B-cells_EPIC',
                    'NK-cells_EPIC', 'CD4T-cells_EPIC', 'CD8T-cells_EPIC', 'Erythrocyte_progenitors',
                    'Lung_cells', 'Kidney', 'Breast', 'Prostate', 'Colon_epithelial_cells',
                    'Hepatocytes', 'Pancreatic_acinar_cells', 'Pancreatic_duct_cells', 'Pancreatic_beta_cells',
                    'Bladder_Ep_cells', 'Cervix_Ep_cells', 'Eso_Ep_cells', 'Gastric_epithelial_cells',
                    'Head_Neck_Ep_cells']
    HEART_TILES = ['CM', 'EC', 'FB', 'MP', 'SMC']
    CAGG_TILES = ['dendritic', 'endothelial', 'eosinophil', 'erythroblast', 'macrophage',
                  'monocyte', 'neutrophil', 'placenta', 'tcell', 'adipose', 'brain', 'fibroblast',
                  'heart', 'hepatocyte', 'lung', 'mammary', 'megakaryocyte', 'skeletal', 'small_intestine']

    for cohort_name, df in csvs.items():
        # Group counts (use the original assignments)
        group_counts = Counter(df['group'])
        print(f"\n{cohort_name}: group counts: {dict(group_counts)}")
        valid = [g for g, n in group_counts.items() if g != 'unknown' and n >= 5]
        pairs = {}
        for g1, g2 in combinations(valid, 2):
            a = df[df['group'] == g1]
            b = df[df['group'] == g2]
            d = {
                'n_g1': int(group_counts[g1]), 'n_g2': int(group_counts[g2]),
                'loyfer': {t: cohens_d(a[f'loyfer_{t}_A'].values, b[f'loyfer_{t}_A'].values)
                           for t in LOYFER_TILES if f'loyfer_{t}_A' in df.columns},
                'episcore_heartref': {t: cohens_d(a[f'heart_{t}_A'].values, b[f'heart_{t}_A'].values)
                                       for t in HEART_TILES if f'heart_{t}_A' in df.columns},
                'caggiano': {t: cohens_d(a[f'caggiano_{t}_A'].values, b[f'caggiano_{t}_A'].values)
                              for t in CAGG_TILES if f'caggiano_{t}_A' in df.columns},
            }
            pairs[f"{g1}_vs_{g2}"] = d
        unified['cohorts'][cohort_name] = {
            'n_samples': len(df),
            'group_counts': dict(group_counts),
            'cohen_d_per_atlas_per_tile': pairs,
            'unified_per_sample_csv': str(UNIFIED_OUT / f'{cohort_name}_unified_per_sample.csv'),
        }

    with open(UNIFIED_OUT / 'VAL-112_113_unified_results.json', 'w') as f:
        json.dump(unified, f, indent=2, default=str)
    print(f"\n=== Wrote: {UNIFIED_OUT / 'VAL-112_113_unified_results.json'}")


if __name__ == '__main__':
    main()
