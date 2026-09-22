#!/usr/bin/env python3
"""Extract GSE180683 (Salas/Wiencke 2022 EPIC blood) to meth_atlas-compatible CSV.
Drops detection-pval columns, keeps β only, renames chip-pos to GSM."""
import gzip, csv, json, sys

src = '/home/claude/glioma_work/GSE180683_Matrix.txt.gz'
chip_map_path = '/home/claude/iam_repo/Biological_Physics/validation_runs/GSE180683_chippos_to_gsm.json'
out = '/home/claude/brain_decon/input/GSE180683_betas.csv'

with open(chip_map_path) as f:
    chip2gsm = json.load(f)

with gzip.open(src, 'rt') as f:
    header = f.readline().strip().split('\t')
    header = [h.strip('"') for h in header]
    # First col is ID_REF; rest alternate β and detection p-val
    # β columns are even-indexed (1, 3, 5,...) per the example. Actually inspect:
    # "ID_REF" "203135920146_R08C01" "203135920146_R08C01 Detection Pval" ...
    # So β columns are odd positions (1, 3, 5,...) and pval are even (2, 4, 6,...) 0-indexed
    keep_cols = [0]  # CpG
    new_header = ['CpGs']
    for i, col in enumerate(header):
        if i == 0: continue
        if 'Detection' in col: continue
        chip = col.strip()
        gsm = chip2gsm.get(chip, chip)
        keep_cols.append(i)
        new_header.append(gsm)
    print(f'Header parsed: keeping {len(keep_cols)} cols ({len(new_header)-1} samples)', file=sys.stderr)
    
    with open(out, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(new_header)
        n = 0
        for line in f:
            line = line.rstrip('\n')
            if not line: continue
            parts = line.split('\t')
            cpg = parts[0].strip('"')
            if not cpg.startswith('cg'): continue
            row = [cpg] + [parts[i] for i in keep_cols[1:]]
            writer.writerow(row)
            n += 1
            if n % 100000 == 0:
                print(f'  {n} rows', file=sys.stderr)
print(f'Done: {n} CpG rows, {len(new_header)-1} samples')
