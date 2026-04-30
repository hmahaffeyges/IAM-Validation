#!/usr/bin/env python3
"""Stage 1 simplified — just write atlas-relevant rows to TSV. No numpy stats."""
import gzip, json, csv, time
from pathlib import Path

BETA = '/home/claude/val118_work/GSE269244_BetaValues.txt.gz'
OUT = Path('/home/claude/val118_work/stage1_artifacts')
OUT.mkdir(exist_ok=True)

def load_csv_cpgs(path, key):
    with open(path) as f:
        return {row[key] for row in csv.DictReader(f)}

atlas_cpgs = set()
atlas_cpgs |= load_csv_cpgs('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_prostateref/episcore_prostateref_cpg_bridged.csv', 'probeID')
atlas_cpgs |= load_csv_cpgs('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv', 'CpGs')
atlas_cpgs |= load_csv_cpgs('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage3_immune_fraction/unilife_guo_2025/centUniLIFE_reference_matrix.csv', 'CpG_ID')
atlas_cpgs |= load_csv_cpgs('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage3_immune_fraction/salas_blood_epic_idol/IDOLOptimizedCpGs_compTable.csv', 'CpG_ID')
with open('/home/claude/iam_repo/Biological_Physics/validation_runs/xu538_panel.json') as f:
    xu538 = json.load(f)
if isinstance(xu538, dict):
    for k in ('cpgs','panel','CpGs','cpg_list'):
        if k in xu538:
            atlas_cpgs |= set(xu538[k])
            break

print(f'Atlas CpGs: {len(atlas_cpgs)}', flush=True)
t0 = time.time()
n = 0; kept = 0
sentrix = None
ATLAS_TSV = OUT / 'atlas_betas.tsv'
with gzip.open(BETA, 'rt') as fin, open(ATLAS_TSV, 'w') as fout:
    for line in fin:
        if n == 0:
            sentrix = line.rstrip('\n').split('\t')[1:]
            fout.write(line)
            n = 1
            continue
        n += 1
        cpg = line[:line.find('\t')]
        if cpg in atlas_cpgs:
            fout.write(line)
            kept += 1
        if n % 100000 == 0:
            print(f'  {n} rows, {kept} kept, {time.time()-t0:.0f}s', flush=True)

with open(OUT/'sentrix_order.json','w') as f:
    json.dump(sentrix, f)
print(f'DONE: {n} rows, {kept} kept, {time.time()-t0:.1f}s', flush=True)
