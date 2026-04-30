#!/usr/bin/env python3
"""
VAL-118 Phase C — Stage 2: load atlas TSV, score per-sample × per-atlas,
compute paired/unpaired Cohen's d, write outputs.
"""
import csv, json, time, math
from pathlib import Path
from collections import defaultdict
import numpy as np

ATLAS_TSV = Path('/home/claude/val118_work/stage1_artifacts/atlas_betas.tsv')
SENTRIX_ORDER = Path('/home/claude/val118_work/stage1_artifacts/sentrix_order.json')
SAMPLE_MAP = Path('/home/claude/val118_work/gse269244_sample_map.json')
OUTPUT_DIR = Path('/home/claude/iam_repo/Biological_Physics/validation_runs/VAL-118_prostateref_phaseC')
OUTPUT_DIR.mkdir(exist_ok=True)

VAL_ID = 'VAL-118'
PREREG_SHA = '0a860bea365a2019e1d6fd95a492dc4671a170372165011e115272fdf59a275c'
SEAL_TIMESTAMP = '2026-04-30T16:09:42Z'
BETA_SHA = '7b9fa2825bdd88b0936afba0e19fb0fbcf1bd404a65469d9fb0735829dc88a89'
RNG_SEED = 20260420

# H_min anchors
H_MIN = {
    'secretory': 0.843264, 'stromal': 0.862950, 'immune': 0.838889,
    'terminal': 0.772837, 'cycling': 0.856055, 'progenitor': 0.852216,
    'stem_adult': 0.873718,
}

PROSTATEREF_HMIN = {'BE': H_MIN['secretory'], 'EC': H_MIN['stromal'], 'Fib': H_MIN['stromal'],
                    'LE': H_MIN['secretory'], 'Leu': H_MIN['immune'], 'SM': H_MIN['stromal']}

LOYFER_CLASS = {
    'Monocytes_EPIC': 'immune', 'B-cells_EPIC': 'immune', 'CD4T-cells_EPIC': 'immune',
    'NK-cells_EPIC': 'immune', 'CD8T-cells_EPIC': 'immune', 'Neutrophils_EPIC': 'immune',
    'Erythrocyte_progenitors': 'progenitor',
    'Adipocytes': 'stromal', 'Vascular_endothelial_cells': 'stromal',
    'Lung_cells': 'cycling', 'Hepatocytes': 'secretory',
    'Pancreatic_beta_cells': 'secretory', 'Pancreatic_acinar_cells': 'secretory',
    'Pancreatic_duct_cells': 'cycling', 'Colon_epithelial_cells': 'cycling',
    'Left_atrium': 'terminal', 'Cortical_neurons': 'terminal',
}

UNILIFE_TILES = ['B', 'CD4T', 'CD8T', 'Mono', 'nRBC', 'Gran', 'NK',
                 'aCD4Tnv', 'aBaso', 'aCD4Tmem', 'aBmem', 'aBnv', 'aTreg',
                 'aCD8Tmem', 'aCD8Tnv', 'aEos', 'aNK', 'aNeu', 'aMono']
SALAS_TILES = ['CD8T', 'CD4T', 'NK', 'Bcell', 'Mono', 'Neu']


def shannon_H_arr(b):
    """Vectorized binary Shannon entropy."""
    b = np.where((b > 0) & (b < 1) & ~np.isnan(b), b, np.nan)
    out = np.where(np.isnan(b), 0.0, -b*np.log2(np.clip(b, 1e-10, 1)) - (1-b)*np.log2(np.clip(1-b, 1e-10, 1)))
    return out


def cohens_d_unpaired(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return float('nan')
    sd = math.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    if sd == 0: return float('nan')
    return float((a.mean() - b.mean()) / sd)


def cohens_d_paired(diffs):
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 2: return float('nan')
    if diffs.std(ddof=1) == 0: return float('nan')
    return float(diffs.mean() / diffs.std(ddof=1))


# ============================================================
# Load atlas β matrix into numpy
# ============================================================
print('Loading atlas β matrix from TSV...')
t0 = time.time()
sentrix_order = json.load(open(SENTRIX_ORDER))
n_samples = len(sentrix_order)

cpgs = []
betas_list = []
with open(ATLAS_TSV) as f:
    header = f.readline()  # skip
    for line in f:
        parts = line.rstrip('\n').split('\t')
        cpgs.append(parts[0])
        row = np.empty(n_samples, dtype=np.float32)
        for j, v in enumerate(parts[1:]):
            row[j] = np.nan if v in ('','NA','NaN') else float(v)
        betas_list.append(row)
B = np.array(betas_list, dtype=np.float32)
print(f'  Atlas β matrix: {B.shape} in {time.time()-t0:.1f}s')
cpg_to_idx = {c:i for i,c in enumerate(cpgs)}

# ============================================================
# Load atlases
# ============================================================
print('Loading atlas references...')

prostateref = {}
with open('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_prostateref/episcore_prostateref_cpg_bridged.csv') as f:
    for row in csv.DictReader(f):
        prostateref[row['probeID']] = {ct: float(row[ct]) for ct in PROSTATEREF_HMIN}

loyfer = {}
with open('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv') as f:
    for row in csv.DictReader(f):
        cpg = row['CpGs']
        loyfer[cpg] = {}
        for tile in LOYFER_CLASS:
            try: loyfer[cpg][tile] = float(row[tile])
            except: loyfer[cpg][tile] = float('nan')

unilife = {}
with open('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage3_immune_fraction/unilife_guo_2025/centUniLIFE_reference_matrix.csv') as f:
    for row in csv.DictReader(f):
        unilife[row['CpG_ID']] = {tile: float(row[tile]) for tile in UNILIFE_TILES if tile in row}

salas = {}
with open('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage3_immune_fraction/salas_blood_epic_idol/IDOLOptimizedCpGs_compTable.csv') as f:
    for row in csv.DictReader(f):
        salas[row['CpG_ID']] = {tile: float(row[tile]) for tile in SALAS_TILES if tile in row}

with open('/home/claude/iam_repo/Biological_Physics/validation_runs/xu538_panel.json') as f:
    xu538 = json.load(f)
xu538_cpgs = xu538['cpgs'] if isinstance(xu538, dict) else xu538

print(f'  ProstateRef: {len(prostateref)}')
print(f'  Loyfer: {len(loyfer)}')
print(f'  UniLIFE: {len(unilife)}')
print(f'  Salas: {len(salas)}')
print(f'  Xu-538: {len(xu538_cpgs)}')


# ============================================================
# Compute per-sample tile A-scores using vectorized numpy
# ============================================================
def score_atlas_tiles(atlas_dict, tile_to_hmin):
    """For an atlas, for each tile, build vectors of (sample β minus tile_ref β)
    and compute mean/H_min per sample.
    Returns: dict[tile] -> np.array of length n_samples
    """
    out = {}
    for tile, hmin in tile_to_hmin.items():
        # Find CpGs in our β matrix that have non-NaN tile reference
        rows_ix = []
        tile_refs = []
        for cpg, refs in atlas_dict.items():
            if tile in refs and not np.isnan(refs[tile]) and cpg in cpg_to_idx:
                rows_ix.append(cpg_to_idx[cpg])
                tile_refs.append(refs[tile])
        if not rows_ix:
            out[tile] = np.full(n_samples, np.nan, dtype=np.float32)
            continue
        rows_ix = np.array(rows_ix)
        tile_refs = np.array(tile_refs, dtype=np.float32)
        # B_subset: (n_atlas_cpgs_with_ref, n_samples)
        B_subset = B[rows_ix, :]
        # |B_subset - tile_refs[:, None]|  -> elementwise abs delta
        deltas = np.abs(B_subset - tile_refs[:, None])
        # mean ignoring NaN sample β
        mean_delta = np.nanmean(deltas, axis=0)
        out[tile] = mean_delta / hmin
    return out


print()
print('Scoring per-sample × per-atlas-tile...')
t1 = time.time()

pr_hmin = PROSTATEREF_HMIN
ly_hmin = {t: H_MIN[c] for t, c in LOYFER_CLASS.items()}
un_hmin = {t: H_MIN['immune'] for t in UNILIFE_TILES}
sa_hmin = {t: H_MIN['immune'] for t in SALAS_TILES}

scores = {}
scores['PR'] = score_atlas_tiles(prostateref, pr_hmin)
scores['LY'] = score_atlas_tiles(loyfer, ly_hmin)
scores['UN'] = score_atlas_tiles(unilife, un_hmin)
scores['SA'] = score_atlas_tiles(salas, sa_hmin)

# Stage 1 Xu-538 pooled (different formula: shannon H, not |delta|)
xu538_in = [c for c in xu538_cpgs if c in cpg_to_idx]
xu538_ix = np.array([cpg_to_idx[c] for c in xu538_in])
print(f'  Xu-538 CpGs available: {len(xu538_in)}/{len(xu538_cpgs)}')
B_xu = B[xu538_ix, :]
H_xu = shannon_H_arr(B_xu)
A_xu538 = np.nanmean(H_xu, axis=0) / H_MIN['immune']

print(f'  Scoring time: {time.time()-t1:.1f}s')

# ============================================================
# Build per-sample table joined with sample metadata
# ============================================================
sample_map = json.load(open(SAMPLE_MAP))
sentrix_to_info = {s['sentrix_id']: s for s in sample_map['samples']}

per_sample = []
for j, sentrix in enumerate(sentrix_order):
    info = sentrix_to_info.get(sentrix)
    if not info:
        continue
    row = {
        'sentrix_id': sentrix,
        'gsm': info['gsm'],
        'patient_id': info['patient_id'],
        'sample_type': info['sample_type'],
        'gleason': info['gleason'],
        'A_xu538_stage1': float(A_xu538[j]),
    }
    for atlas_pfx, tile_scores in scores.items():
        for tile, vec in tile_scores.items():
            row[f'A_{atlas_pfx}_{tile}'] = float(vec[j])
    per_sample.append(row)

print(f'Per-sample rows: {len(per_sample)}')

# ============================================================
# Cohen's d per tile
# ============================================================
print()
print('Computing Cohen\'s d per atlas tile...')
tumor_rows = [r for r in per_sample if r['sample_type'] == 'Tumor']
normal_rows = [r for r in per_sample if r['sample_type'] == 'Normal']
print(f'  Tumor n={len(tumor_rows)}, Normal n={len(normal_rows)}')

by_pid = defaultdict(dict)
for r in per_sample:
    by_pid[r['patient_id']][r['sample_type']] = r
paired_pids = [pid for pid, d in by_pid.items() if 'Tumor' in d and 'Normal' in d]
print(f'  Paired patients: {len(paired_pids)}')

# All A_ keys
all_keys = sorted({k for r in per_sample for k in r if k.startswith('A_')})

def get(r, k): 
    v = r.get(k, float('nan'))
    return float(v) if v is not None else float('nan')

cohen_d_per_tile = {}
for key in all_keys:
    tu = np.array([get(r, key) for r in tumor_rows])
    no = np.array([get(r, key) for r in normal_rows])
    pd = np.array([get(by_pid[pid]['Tumor'], key) - get(by_pid[pid]['Normal'], key) for pid in paired_pids])
    if np.all(np.isnan(tu)) or np.all(np.isnan(no)):
        continue
    cohen_d_per_tile[key] = {
        'tumor_n': int(np.sum(~np.isnan(tu))),
        'tumor_mean': float(np.nanmean(tu)),
        'tumor_sd': float(np.nanstd(tu, ddof=1)),
        'normal_n': int(np.sum(~np.isnan(no))),
        'normal_mean': float(np.nanmean(no)),
        'normal_sd': float(np.nanstd(no, ddof=1)),
        'd_unpaired': cohens_d_unpaired(tu, no),
        'd_paired': cohens_d_paired(pd),
        'n_pairs': int(np.sum(~np.isnan(pd))),
        'mean_paired_diff': float(np.nanmean(pd)),
    }

# ============================================================
# Stage 1 Xu-538 reproduction control
# ============================================================
xu538_d = cohen_d_per_tile.get('A_xu538_stage1', {})
xu538_repro = {
    'val058_sealed_d_paired': 0.4972847944678357,
    'val058_sealed_d_unpaired': 0.4002677553235111,
    'val058_sealed_tumor_mean': 0.8021689164616835,
    'val058_sealed_normal_mean': 0.7808905570251791,
    'val118_d_paired': xu538_d.get('d_paired', float('nan')),
    'val118_d_unpaired': xu538_d.get('d_unpaired', float('nan')),
    'val118_tumor_mean': xu538_d.get('tumor_mean', float('nan')),
    'val118_normal_mean': xu538_d.get('normal_mean', float('nan')),
    'd_paired_diff': abs(xu538_d.get('d_paired', float('nan')) - 0.4972847944678357),
    'reproduction_within_tolerance': abs(xu538_d.get('d_paired', float('nan')) - 0.4972847944678357) <= 0.10,
}

print()
print('=== Xu-538 Stage 1 reproduction control ===')
print(f'  VAL-058 sealed paired d: 0.4973')
print(f'  VAL-118 paired d:        {xu538_repro["val118_d_paired"]:.4f}')
print(f'  Difference:              {xu538_repro["d_paired_diff"]:.4f}')
print(f'  Within ±0.10 tolerance:  {xu538_repro["reproduction_within_tolerance"]}')

# ============================================================
# Outcome determination
# ============================================================
outcomes = []
le_d = cohen_d_per_tile.get('A_PR_LE', {}).get('d_paired', float('nan'))
# Check Loyfer prostate-relevant tiles - the tile names depend on what's actually in atlas
# "Prostate" not in our LOYFER_CLASS subset; the atlas has 25 cell types; we picked subset
# So we check if any of our scored tiles is differentiating
hepatocyte_d = cohen_d_per_tile.get('A_LY_Hepatocytes', {}).get('d_paired', float('nan'))

if not np.isnan(le_d) and le_d >= 0.30:
    outcomes.append(f'O2_LE_TILE_DIFFERENTIATING (d_paired={le_d:.3f})')

# Stage 3 immune-shift
stage3_max = ('', 0)
for k, v in cohen_d_per_tile.items():
    if (k.startswith('A_UN_') or k.startswith('A_SA_')):
        d = v.get('d_paired', 0)
        if not np.isnan(d) and abs(d) >= 0.40 and abs(d) > abs(stage3_max[1]):
            stage3_max = (k, d)
if stage3_max[0]:
    outcomes.append(f'O4_STAGE_3_IMMUNE_SHIFT_PROMINENT ({stage3_max[0]} d_paired={stage3_max[1]:+.3f})')

# Multi-atlas convergence: ProstateRef LE + Stage1 Xu-538 reproduction
if not np.isnan(le_d) and le_d >= 0.30 and xu538_repro['reproduction_within_tolerance']:
    outcomes.insert(0, 'O1_MULTI_ATLAS_CONVERGENT')

if not outcomes:
    outcomes.append('O5_or_O6_review_needed')

# ============================================================
# Headlines
# ============================================================
print()
print('=== Top 20 tiles by paired |d| ===')
sorted_tiles = sorted(cohen_d_per_tile.items(),
                      key=lambda kv: -abs(kv[1].get('d_paired', 0)) if not np.isnan(kv[1].get('d_paired', 0)) else 0)
for k, v in sorted_tiles[:20]:
    dp = v.get('d_paired', float('nan'))
    du = v.get('d_unpaired', float('nan'))
    print(f'  {k:30s} d_paired={dp:+.4f}  d_unpaired={du:+.4f}  pairs={v["n_pairs"]}')

# ============================================================
# Write outputs
# ============================================================
runtime = time.time() - t0
results = {
    'val_id': VAL_ID,
    'val_type': 'PHASE_C_RUN_EVERYTHING',
    'card_target': 'prostate-epic v0.3',
    'cohort': 'GSE269244',
    'cohort_n': len(per_sample),
    'cohort_n_paired': len(paired_pids),
    'beta_matrix_sha256': BETA_SHA,
    'prereg_sha': PREREG_SHA,
    'seal_timestamp': SEAL_TIMESTAMP,
    'rng_seed': RNG_SEED,
    'runtime_seconds': runtime,
    'atlases_scored': {
        'xu538_stage1': len(xu538_cpgs),
        'prostateref': len(prostateref),
        'layered_moss_loyfer_subset': len(loyfer),
        'unilife': len(unilife),
        'salas_idol': len(salas),
    },
    'xu538_reproduction_control': xu538_repro,
    'cohen_d_per_tile': cohen_d_per_tile,
    'outcome_classes': outcomes,
    'sealed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}

with open(OUTPUT_DIR / f'{VAL_ID}_cohen_d_per_atlas.json', 'w') as f:
    json.dump(results, f, indent=2)

# Per-sample CSV
all_csv_keys = sorted({k for r in per_sample for k in r})
with open(OUTPUT_DIR / f'{VAL_ID}_per_sample_run_everything.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_csv_keys)
    w.writeheader()
    for r in per_sample:
        w.writerow(r)

print()
print('=' * 70)
print('OUTCOMES:')
for o in outcomes: print(f'  - {o}')
print('=' * 70)
print(f'Runtime: {runtime:.1f}s')
print(f'Wrote: {OUTPUT_DIR}/VAL-118_cohen_d_per_atlas.json')
print(f'Wrote: {OUTPUT_DIR}/VAL-118_per_sample_run_everything.csv')
