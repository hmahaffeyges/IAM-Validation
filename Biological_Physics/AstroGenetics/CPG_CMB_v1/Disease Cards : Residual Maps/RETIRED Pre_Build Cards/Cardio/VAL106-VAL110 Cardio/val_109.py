"""VAL-109 — Cardio-epic on GSE84395 PAH PEC cohort."""
import csv as _csv
import json, math, statistics
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

# Load atlases
loyfer_df = pd.read_csv('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv', index_col=0)
unilife_df = pd.read_csv('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage3_immune_fraction/unilife_guo_2025/centUniLIFE_reference_matrix.csv', index_col=0)
salas_df = pd.read_csv('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage3_immune_fraction/salas_blood_epic_idol/IDOLOptimizedCpGs450k_compTable.csv', index_col=0)
unilife_cpgs = set(unilife_df.index)
salas_cpgs = set(salas_df.index)
loyfer_tile_names = list(loyfer_df.columns)

# Load β table from series matrix (starts at line 76 per inspection)
print("Loading GSE84395 β matrix from series matrix...", flush=True)
beta_df = pd.read_csv('sm.txt', sep='\t', skiprows=75, index_col='ID_REF',
                       comment='!', skipfooter=1, engine='python', dtype=str)
beta_df.index.name = 'CpG'
print(f"β matrix raw: {beta_df.shape}", flush=True)
# Convert to numeric
for col in beta_df.columns:
    beta_df[col] = pd.to_numeric(beta_df[col], errors='coerce')
print(f"β matrix numeric: {beta_df.shape}", flush=True)

# Compute CHK-3.1A per sample (full genome on this cohort's distribution)
print("CHK-3.1A on each sample (full genome)...", flush=True)
chk_results = []
for sid in beta_df.columns:
    s = beta_df[sid].dropna()
    n_valid = len(s)
    if n_valid == 0:
        chk_results.append({'sample': sid, 'n_valid': 0, 'f_extreme': 0, 'f_middle': 0})
        continue
    f_ext = float(((s < 0.10) | (s > 0.90)).sum() / n_valid)
    f_mid = float(((s >= 0.40) & (s <= 0.60)).sum() / n_valid)
    chk_results.append({'sample': sid, 'n_valid': n_valid, 'f_extreme': f_ext, 'f_middle': f_mid})
fes = [c['f_extreme'] for c in chk_results]
fms = [c['f_middle'] for c in chk_results]
print(f"GSE84395 fn-normalized substrate CHK-3.1A:", flush=True)
print(f"  f_extreme: mean {statistics.mean(fes)*100:.2f}% SD {statistics.stdev(fes)*100:.2f}% range [{min(fes)*100:.2f}%, {max(fes)*100:.2f}%]", flush=True)
print(f"  f_middle:  mean {statistics.mean(fms)*100:.2f}% SD {statistics.stdev(fms)*100:.2f}%", flush=True)

# Self-cal: pass if f_extreme within ±2*SD of cohort mean
fe_mean, fe_sd = statistics.mean(fes), statistics.stdev(fes)
fm_mean, fm_sd = statistics.mean(fms), statistics.stdev(fms)
fe_low, fe_high = fe_mean - 2*fe_sd, fe_mean + 2*fe_sd
fm_low, fm_high = max(0, fm_mean - 2*fm_sd), fm_mean + 2*fm_sd
print(f"Self-cal envelope: f_extreme [{fe_low*100:.2f}%, {fe_high*100:.2f}%], f_middle [{fm_low*100:.2f}%, {fm_high*100:.2f}%]", flush=True)

CHK_NVALID_MIN = 400000
chk_pass_set = set()
for c in chk_results:
    if (c['n_valid'] >= CHK_NVALID_MIN and fe_low <= c['f_extreme'] <= fe_high
        and fm_low <= c['f_middle'] <= fm_high):
        chk_pass_set.add(c['sample'])
print(f"CHK-3.1A self-cal pass: {len(chk_pass_set)}/{len(chk_results)}", flush=True)

# Load metadata
metadata = json.load(open('metadata.json'))
geo_to_status = {m['geo']: m['status'] for m in metadata}

# Score each sample
common_loyfer = beta_df.index.intersection(loyfer_df.index)
beta_loyfer = beta_df.loc[common_loyfer]
ref_loyfer = loyfer_df.loc[common_loyfer]
salas_in_beta = beta_df.index.intersection(salas_cpgs)
unilife_in_beta = beta_df.index.intersection(unilife_cpgs)
beta_salas = beta_df.loc[salas_in_beta]
beta_unilife = beta_df.loc[unilife_in_beta]
print(f"Loyfer overlap {len(common_loyfer)}, Salas {len(salas_in_beta)}, UniLIFE {len(unilife_in_beta)}", flush=True)

results = []
for sid in beta_df.columns:
    status = geo_to_status.get(sid, 'unknown')
    chk_pass = sid in chk_pass_set
    
    sb = beta_salas[sid].dropna()
    if len(sb) > 0:
        p = float((sb >= 0.5).mean())
        stage1_H = 0.0 if p in (0.0, 1.0) else -(p*math.log2(p) + (1-p)*math.log2(1-p))
        stage1_A = stage1_H - H_MIN['immune']
    else:
        stage1_H = stage1_A = None
    
    sb_loyfer = beta_loyfer[sid].dropna()
    stage2 = {}
    for tile in loyfer_tile_names:
        tref = ref_loyfer[tile].dropna()
        common = sb_loyfer.index.intersection(tref.index)
        if len(common) > 0:
            d = (sb_loyfer.loc[common] - tref.loc[common]).abs().mean()
            stage2[tile] = {'A': float(d), 'class': LOYFER_TILE_CLASS.get(tile,'?')}
        else:
            stage2[tile] = {'A': None, 'class': LOYFER_TILE_CLASS.get(tile,'?')}
    
    ub = beta_unilife[sid].dropna()
    if len(ub) > 0:
        p = float((ub >= 0.5).mean())
        stage3_uni_H = 0.0 if p in (0.0, 1.0) else -(p*math.log2(p) + (1-p)*math.log2(1-p))
    else:
        stage3_uni_H = None
    
    results.append({'sample': sid, 'status': status, 'chk_pass': chk_pass,
                    'stage1_immune_H': stage1_H, 'stage1_immune_A': stage1_A,
                    'stage2_loyfer': stage2, 'stage3_unilife_H': stage3_uni_H,
                    'stage3_salas_H': stage1_H})

qc_pass = [r for r in results if r['chk_pass']]
print(f"QC pass {len(qc_pass)}/{len(results)}", flush=True)

def cohen_d(a, b):
    a = [x for x in a if x is not None]; b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2: return None
    pooled = math.sqrt((statistics.stdev(a)**2 + statistics.stdev(b)**2)/2)
    if pooled == 0: return None
    return (statistics.mean(a) - statistics.mean(b)) / pooled

target_groups = ['control', 'hPAH', 'iPAH']
samples_by_g = {g: [r for r in qc_pass if r['status']==g] for g in target_groups}
for g, rs in samples_by_g.items():
    print(f"  {g}: {len(rs)}", flush=True)

# Stage 1 contrasts
stage1_pair_d = {}
print("\nStage 1 contrasts:", flush=True)
for i, g1 in enumerate(target_groups):
    for g2 in target_groups[i+1:]:
        a = [r['stage1_immune_A'] for r in samples_by_g[g1]]
        b = [r['stage1_immune_A'] for r in samples_by_g[g2]]
        d = cohen_d(a, b)
        stage1_pair_d[f"{g1}_VS_{g2}"] = {'d': d, 'n_a': sum(1 for x in a if x is not None), 'n_b': sum(1 for x in b if x is not None)}
        print(f"  {g1} vs {g2}: d={d}", flush=True)

# Stage 2 contrasts
print("\nStage 2 contrasts (per-tile):", flush=True)
stage2_pair_d = {}
for i, g1 in enumerate(target_groups):
    for g2 in target_groups[i+1:]:
        pair = f"{g1}_VS_{g2}"
        stage2_pair_d[pair] = {}
        for tile in loyfer_tile_names:
            d = cohen_d([r['stage2_loyfer'][tile]['A'] for r in samples_by_g[g1]],
                        [r['stage2_loyfer'][tile]['A'] for r in samples_by_g[g2]])
            stage2_pair_d[pair][tile] = {'d': d, 'class': LOYFER_TILE_CLASS.get(tile,'?')}
        ranked = sorted([(t, v['d']) for t, v in stage2_pair_d[pair].items() if v['d'] is not None],
                        key=lambda x: -abs(x[1]))[:5]
        print(f"  {pair} top 5:", flush=True)
        for t, d in ranked:
            print(f"    {t} ({LOYFER_TILE_CLASS.get(t,'?')}): {d:.4f}", flush=True)
        for cv in ['Vascular_endothelial_cells', 'Left_atrium', 'Adipocytes', 'Lung_cells']:
            d = stage2_pair_d[pair][cv]['d']
            print(f"    [cardio:{cv}] {d:.4f}" if d is not None else f"    [cardio:{cv}] NA", flush=True)

# Stage 3 contrasts
print("\nStage 3 contrasts:", flush=True)
stage3_pair_d = {}
for atlas, key in [('UniLIFE','stage3_unilife_H'), ('Salas','stage3_salas_H')]:
    stage3_pair_d[atlas] = {}
    for i, g1 in enumerate(target_groups):
        for g2 in target_groups[i+1:]:
            d = cohen_d([r[key] for r in samples_by_g[g1]], [r[key] for r in samples_by_g[g2]])
            stage3_pair_d[atlas][f"{g1}_VS_{g2}"] = d
            print(f"  {atlas} {g1} vs {g2}: {d}", flush=True)

# Outcome
qc_fail_rate = (len(results) - len(qc_pass)) / len(results)
stage1_strong = any(abs(v['d']) >= 0.5 for v in stage1_pair_d.values() if v['d'] is not None)
# Vascular tiles specifically
vasc_strong = any(abs(stage2_pair_d[p][t]['d']) >= 0.5
                  for p in stage2_pair_d for t in ['Vascular_endothelial_cells','Left_atrium']
                  if stage2_pair_d[p][t]['d'] is not None)
stage2_any_strong = any(abs(stage2_pair_d[p][t]['d']) >= 0.5
                        for p in stage2_pair_d for t in stage2_pair_d[p]
                        if stage2_pair_d[p][t]['d'] is not None)
stage3_strong = any(abs(d) >= 0.5 for atl in stage3_pair_d for d in stage3_pair_d[atl].values() if d is not None)
hpah_vs_ipah_strong = (
    (stage1_pair_d.get('hPAH_VS_iPAH', {}).get('d') is not None and abs(stage1_pair_d['hPAH_VS_iPAH']['d']) >= 0.5)
    or any(abs(stage2_pair_d.get('hPAH_VS_iPAH', {}).get(t, {}).get('d') or 0) >= 0.5
           for t in (stage2_pair_d.get('hPAH_VS_iPAH', {}) or {}))
)

if qc_fail_rate >= 0.10: outcome = "O5_DATA_INTEGRITY_FLAG"
elif vasc_strong: outcome = "O2_PAH_VASCULAR_TILE_DIFFERENTIATING"
elif stage1_strong or stage2_any_strong or stage3_strong: outcome = "O1_PAH_FRAMEWORK_DIFFERENTIATING"
elif hpah_vs_ipah_strong: outcome = "O4_HPAH_VS_IPAH_DIFFERENTIATING"
else: outcome = "O3_PAH_FRAMEWORK_UNDIFFERENTIATING"

print(f"\n========== OUTCOME: {outcome} ==========", flush=True)

output = {
    'val_id': 'VAL-109', 'prereg_sha256': 'f6450b4cf5d384d2ea27b349c101b3f167a6a549d276e670e68fb2232b45f21e',
    'sealed_at': '2026-04-28T22:51Z',
    'executed_at': datetime.now(timezone.utc).isoformat(),
    'cohort': 'GSE84395 PAH PEC n=39 (control 18 + hPAH 10 + iPAH 11)',
    'substrate': 'minfi preprocessFunnorm (functional normalization), HM450K GPL16304',
    'chk_31a_self_cal': {
        'cohort_f_extreme': {'mean': fe_mean, 'sd': fe_sd, 'envelope': [fe_low, fe_high]},
        'cohort_f_middle': {'mean': fm_mean, 'sd': fm_sd, 'envelope': [fm_low, fm_high]},
        'qc_pass': len(qc_pass), 'qc_fail_rate': qc_fail_rate,
    },
    'group_n': {g: len(samples_by_g[g]) for g in target_groups},
    'stage1_pair_d': stage1_pair_d,
    'stage2_pair_d_summary': {
        pair: {'top5': sorted([(t, v['d']) for t, v in tiles.items() if v['d'] is not None],
                              key=lambda x: -abs(x[1]))[:5],
               'cardio_tiles': {t: tiles[t] for t in ['Vascular_endothelial_cells','Left_atrium','Adipocytes','Lung_cells']}}
        for pair, tiles in stage2_pair_d.items()
    },
    'stage3_pair_d': stage3_pair_d,
    'outcome': outcome,
    'notes': {'stage1_proxy': 'Salas IDOL 350-CpG used as proxy for patent-protected Xu-538'},
}
with open('results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

with open('per_sample.csv', 'w', newline='') as f:
    fields = ['sample','status','chk_pass','stage1_immune_H','stage1_immune_A','stage3_unilife_H',
              'loyfer_Vascular_endothelial_cells_A','loyfer_Left_atrium_A','loyfer_Hepatocytes_A',
              'loyfer_Adipocytes_A','loyfer_Lung_cells_A','loyfer_Monocytes_EPIC_A']
    w = _csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in results:
        w.writerow({
            'sample': r['sample'], 'status': r['status'], 'chk_pass': r['chk_pass'],
            'stage1_immune_H': r['stage1_immune_H'], 'stage1_immune_A': r['stage1_immune_A'],
            'stage3_unilife_H': r['stage3_unilife_H'],
            'loyfer_Vascular_endothelial_cells_A': r['stage2_loyfer'].get('Vascular_endothelial_cells',{}).get('A'),
            'loyfer_Left_atrium_A': r['stage2_loyfer'].get('Left_atrium',{}).get('A'),
            'loyfer_Hepatocytes_A': r['stage2_loyfer'].get('Hepatocytes',{}).get('A'),
            'loyfer_Adipocytes_A': r['stage2_loyfer'].get('Adipocytes',{}).get('A'),
            'loyfer_Lung_cells_A': r['stage2_loyfer'].get('Lung_cells',{}).get('A'),
            'loyfer_Monocytes_EPIC_A': r['stage2_loyfer'].get('Monocytes_EPIC',{}).get('A'),
        })
print(f"Wrote results.json + per_sample.csv", flush=True)
print(f"Done: {datetime.now(timezone.utc).isoformat()}", flush=True)
