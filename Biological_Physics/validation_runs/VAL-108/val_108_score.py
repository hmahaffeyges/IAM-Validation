"""VAL-108 scoring only — uses pre-filtered gse69138_filtered.tsv"""
import csv as _csv
import json, math, statistics, gc
from datetime import datetime, timezone
import pandas as pd

H_MIN = {'cycling': 0.856055, 'terminal': 0.7728, 'immune': 0.838889,
         'secretory': 0.843264, 'stromal': 0.862950, 'progenitor': 0.852216,
         'stem_adult': 0.873718, 'stem_pluri': 0.982166}

LOYFER_TILE_CLASS = {
    'Monocytes_EPIC': 'immune','B-cells_EPIC':'immune','CD4T-cells_EPIC':'immune',
    'NK-cells_EPIC':'immune','CD8T-cells_EPIC':'immune','Neutrophils_EPIC':'immune',
    'Erythrocyte_progenitors':'progenitor','Adipocytes':'stromal','Cortical_neurons':'terminal',
    'Hepatocytes':'secretory','Lung_cells':'cycling','Pancreatic_beta_cells':'secretory',
    'Pancreatic_acinar_cells':'secretory','Pancreatic_duct_cells':'cycling',
    'Vascular_endothelial_cells':'stromal','Colon_epithelial_cells':'cycling',
    'Left_atrium':'terminal','Bladder':'cycling','Breast':'secretory',
    'Head_and_neck_larynx':'cycling','Kidney':'cycling','Prostate':'secretory',
    'Thyroid':'secretory','Upper_GI':'cycling','Uterus_cervix':'cycling',
}

print(f"Started: {datetime.now(timezone.utc).isoformat()}", flush=True)

loyfer_df = pd.read_csv('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv', index_col=0)
unilife_df = pd.read_csv('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage3_immune_fraction/unilife_guo_2025/centUniLIFE_reference_matrix.csv', index_col=0)
salas_df = pd.read_csv('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage3_immune_fraction/salas_blood_epic_idol/IDOLOptimizedCpGs450k_compTable.csv', index_col=0)

unilife_cpgs = set(unilife_df.index)
salas_cpgs = set(salas_df.index)
loyfer_tile_names = list(loyfer_df.columns)
print(f"Loyfer {loyfer_df.shape}, UniLIFE {len(unilife_cpgs)}, Salas {len(salas_cpgs)}", flush=True)

# Load filtered β matrix — small, in memory
print("Loading filtered β matrix...", flush=True)
beta_df = pd.read_csv('gse69138_filtered.tsv', sep='\t', index_col='TargetID', dtype=str)
print(f"Loaded {beta_df.shape}, converting...", flush=True)
# Convert to float in-place column by column
for col in beta_df.columns:
    beta_df[col] = pd.to_numeric(beta_df[col], errors='coerce')
print(f"β matrix ready: {beta_df.shape}", flush=True)

# Load metadata + chk
metadata = json.load(open('gse69138_metadata.json'))
def normalize_subtype(s):
    s = s.lower().strip()
    if 'lacunar' in s or 'small-vessel' in s or 'small vessel' in s: return 'small_vessel_disease'
    if 'large-artery' in s or 'atherotrombotic' in s: return 'large_artery_atherosclerosis'
    if 'cardioembolic' in s or 'cardio emobolic' in s: return 'cardioembolic'
    return 'other'
chip_to_subtype = {m['chip_id']: normalize_subtype(m.get('stroke subtype',''))
                   for m in metadata if m.get('chip_id')}
chk_stats = json.load(open('/home/claude/edear_working/SUBSTRATE_EQUIV/gse69138_per_sample_chk31a.json'))
chip_to_chk = {s['sample']: s for s in chk_stats}

CHK_31A_EXTREME_MIN, CHK_31A_MIDDLE_MAX, CHK_31A_NVALID_MIN = 0.25, 0.13, 400000

# Pre-extract atlas-specific CpG arrays once
loyfer_idx = loyfer_df.index
common_loyfer = beta_df.index.intersection(loyfer_idx)
print(f"Loyfer overlap with β: {len(common_loyfer)}", flush=True)

beta_loyfer = beta_df.loc[common_loyfer]  # small slice
ref_loyfer = loyfer_df.loc[common_loyfer]

salas_in_beta = beta_df.index.intersection(salas_cpgs)
unilife_in_beta = beta_df.index.intersection(unilife_cpgs)
beta_salas = beta_df.loc[salas_in_beta]
beta_unilife = beta_df.loc[unilife_in_beta]
print(f"Salas overlap: {len(salas_in_beta)}, UniLIFE overlap: {len(unilife_in_beta)}", flush=True)

results = []
for sid in beta_df.columns:
    chk = chip_to_chk.get(sid, {})
    subtype = chip_to_subtype.get(sid, 'unknown')
    chk_pass = (chk.get('n_valid',0) >= CHK_31A_NVALID_MIN
                and chk.get('f_extreme',0) >= CHK_31A_EXTREME_MIN
                and chk.get('f_middle',1) <= CHK_31A_MIDDLE_MAX)
    
    # Stage 1 (Salas proxy)
    sb = beta_salas[sid].dropna()
    if len(sb) > 0:
        p = float((sb >= 0.5).mean())
        stage1_H = 0.0 if p in (0.0, 1.0) else -(p*math.log2(p) + (1-p)*math.log2(1-p))
        stage1_A = stage1_H - H_MIN['immune']
    else:
        stage1_H = stage1_A = None
    
    # Stage 2 Loyfer per-tile
    sb_loyfer = beta_loyfer[sid].dropna()
    stage2 = {}
    for tile in loyfer_tile_names:
        tile_ref = ref_loyfer[tile].dropna()
        common = sb_loyfer.index.intersection(tile_ref.index)
        if len(common) > 0:
            d = (sb_loyfer.loc[common] - tile_ref.loc[common]).abs().mean()
            stage2[tile] = {'A': float(d), 'n': int(len(common)), 'class': LOYFER_TILE_CLASS.get(tile,'?')}
        else:
            stage2[tile] = {'A': None, 'n': 0, 'class': LOYFER_TILE_CLASS.get(tile,'?')}
    
    # Stage 3 UniLIFE
    ub = beta_unilife[sid].dropna()
    if len(ub) > 0:
        p = float((ub >= 0.5).mean())
        stage3_uni_H = 0.0 if p in (0.0, 1.0) else -(p*math.log2(p) + (1-p)*math.log2(1-p))
    else:
        stage3_uni_H = None
    
    results.append({
        'chip_id': sid, 'subtype': subtype, 'chk_31a_pass': chk_pass,
        'stage1_immune_H': stage1_H, 'stage1_immune_A': stage1_A,
        'stage2_loyfer': stage2, 'stage3_unilife_H': stage3_uni_H, 'stage3_salas_H': stage1_H,
    })

print(f"Scored {len(results)} samples", flush=True)
qc_pass = [r for r in results if r['chk_31a_pass']]
qc_fail_rate = (len(results) - len(qc_pass)) / len(results)
print(f"QC pass: {len(qc_pass)} (fail rate {qc_fail_rate*100:.1f}%)", flush=True)

def cohen_d(a, b):
    a = [x for x in a if x is not None]; b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2: return None
    sa, sb = statistics.stdev(a), statistics.stdev(b)
    pooled = math.sqrt((sa**2 + sb**2)/2)
    if pooled == 0: return None
    return (statistics.mean(a) - statistics.mean(b)) / pooled

target_subtypes = ['large_artery_atherosclerosis', 'small_vessel_disease', 'cardioembolic']
samples_by_st = {st: [r for r in qc_pass if r['subtype']==st] for st in target_subtypes}
for st, rs in samples_by_st.items():
    print(f"  {st}: {len(rs)}", flush=True)

print("\nStage 1 contrasts:", flush=True)
stage1_pair_d = {}
for i, st1 in enumerate(target_subtypes):
    for st2 in target_subtypes[i+1:]:
        a = [r['stage1_immune_A'] for r in samples_by_st[st1]]
        b = [r['stage1_immune_A'] for r in samples_by_st[st2]]
        d = cohen_d(a, b)
        stage1_pair_d[f"{st1}_VS_{st2}"] = {'d': d, 'n_a': sum(1 for x in a if x is not None), 'n_b': sum(1 for x in b if x is not None)}
        print(f"  {st1} vs {st2}: d={d}", flush=True)

print("\nStage 2 contrasts (per-tile):", flush=True)
stage2_pair_d = {}
for i, st1 in enumerate(target_subtypes):
    for st2 in target_subtypes[i+1:]:
        pair = f"{st1}_VS_{st2}"
        stage2_pair_d[pair] = {}
        for tile in loyfer_tile_names:
            a = [r['stage2_loyfer'][tile]['A'] for r in samples_by_st[st1]]
            b = [r['stage2_loyfer'][tile]['A'] for r in samples_by_st[st2]]
            d = cohen_d(a, b)
            stage2_pair_d[pair][tile] = {'d': d, 'class': LOYFER_TILE_CLASS.get(tile,'?')}
        ranked = sorted([(t, v['d']) for t,v in stage2_pair_d[pair].items() if v['d'] is not None],
                        key=lambda x: -abs(x[1]))[:5]
        print(f"  {pair} top5:", flush=True)
        for t, d in ranked:
            print(f"    {t} ({LOYFER_TILE_CLASS.get(t,'?')}): {d:.4f}", flush=True)
        for cv in ['Vascular_endothelial_cells', 'Left_atrium', 'Adipocytes']:
            d = stage2_pair_d[pair][cv]['d']
            print(f"    [cardio] {cv}: {d:.4f}" if d is not None else f"    [cardio] {cv}: NA", flush=True)

print("\nStage 3 contrasts:", flush=True)
stage3_pair_d = {}
for atlas, key in [('UniLIFE','stage3_unilife_H'), ('Salas','stage3_salas_H')]:
    stage3_pair_d[atlas] = {}
    for i, st1 in enumerate(target_subtypes):
        for st2 in target_subtypes[i+1:]:
            d = cohen_d([r[key] for r in samples_by_st[st1]], [r[key] for r in samples_by_st[st2]])
            stage3_pair_d[atlas][f"{st1}_VS_{st2}"] = d
            print(f"  {atlas} {st1} vs {st2}: {d}", flush=True)

stage1_strong = any(abs(v['d']) >= 0.5 for v in stage1_pair_d.values() if v['d'] is not None)
stage2_strong = any(abs(stage2_pair_d[p][t]['d']) >= 0.5
                    for p in stage2_pair_d for t in stage2_pair_d[p]
                    if stage2_pair_d[p][t]['d'] is not None)
stage3_strong = any(abs(d) >= 0.5 for atl in stage3_pair_d for d in stage3_pair_d[atl].values() if d is not None)

if qc_fail_rate >= 0.10: outcome = "O5_DATA_INTEGRITY_FLAG"
elif stage1_strong: outcome = "O1_CARDIO_EPIC_3SUBTYPE_DIFFERENTIATED_AT_STAGE_1"
elif stage2_strong: outcome = "O2_CARDIO_EPIC_3SUBTYPE_DIFFERENTIATED_AT_STAGE_2_TILES"
elif stage3_strong: outcome = "O4_STAGE_3_DIFFERENTIATING"
else: outcome = "O3_CARDIO_EPIC_3SUBTYPE_UNDIFFERENTIATED"

print(f"\n========== OUTCOME: {outcome} ==========", flush=True)

output = {
    'val_id': 'VAL-108',
    'prereg_sha256': '6f40ebd9d30bb10242b245d7bde280607f1170e3c7993a8284e2852ad1f69e7a',
    'sealed_at': '2026-04-28T22:34:33Z',
    'executed_at': datetime.now(timezone.utc).isoformat(),
    'cohort': 'GSE69138 ischemic stroke discovery 404 (GenomeStudio AVG_Beta)',
    'chk_31a_self_cal': {'thresholds': {'extreme_min': 0.25, 'middle_max': 0.13, 'nvalid_min': 400000},
                         'qc_pass': len(qc_pass), 'qc_fail_rate': qc_fail_rate},
    'subtype_n': {st: len(samples_by_st[st]) for st in target_subtypes},
    'stage1_pair_d': stage1_pair_d,
    'stage2_pair_d_summary': {
        pair: {'top5': sorted([(t, v['d']) for t, v in tiles.items() if v['d'] is not None],
                              key=lambda x: -abs(x[1]))[:5],
               'cardio_tiles': {t: tiles[t] for t in ['Vascular_endothelial_cells','Left_atrium','Adipocytes']}}
        for pair, tiles in stage2_pair_d.items()
    },
    'stage3_pair_d': stage3_pair_d,
    'outcome': outcome,
    'notes': {'stage1_proxy': 'Salas IDOL 350-CpG used as proxy for patent-protected Xu-538'},
}
with open('results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

with open('per_sample.csv', 'w', newline='') as f:
    fields = ['chip_id','subtype','chk_31a_pass','stage1_immune_H','stage1_immune_A','stage3_unilife_H',
              'loyfer_Vascular_endothelial_cells_A','loyfer_Left_atrium_A','loyfer_Hepatocytes_A',
              'loyfer_Adipocytes_A','loyfer_Monocytes_EPIC_A']
    w = _csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in results:
        w.writerow({
            'chip_id': r['chip_id'], 'subtype': r['subtype'], 'chk_31a_pass': r['chk_31a_pass'],
            'stage1_immune_H': r['stage1_immune_H'], 'stage1_immune_A': r['stage1_immune_A'],
            'stage3_unilife_H': r['stage3_unilife_H'],
            'loyfer_Vascular_endothelial_cells_A': r['stage2_loyfer'].get('Vascular_endothelial_cells',{}).get('A'),
            'loyfer_Left_atrium_A': r['stage2_loyfer'].get('Left_atrium',{}).get('A'),
            'loyfer_Hepatocytes_A': r['stage2_loyfer'].get('Hepatocytes',{}).get('A'),
            'loyfer_Adipocytes_A': r['stage2_loyfer'].get('Adipocytes',{}).get('A'),
            'loyfer_Monocytes_EPIC_A': r['stage2_loyfer'].get('Monocytes_EPIC',{}).get('A'),
        })

print(f"Wrote results.json + per_sample.csv", flush=True)
print(f"Done: {datetime.now(timezone.utc).isoformat()}", flush=True)
