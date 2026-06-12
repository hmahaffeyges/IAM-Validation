#!/usr/bin/env python3
"""Extract GSE51057 cancer-free subset to meth_atlas-compatible CSV."""
import gzip, csv, sys

src = '/home/claude/glioma_work/GSE51057_series_matrix.txt.gz'
out = '/home/claude/brain_decon/input/GSE51057_betas_healthy.csv'

# First pass: identify healthy samples
gsms = []
cancer_status = []
with gzip.open(src, 'rt') as f:
    for line in f:
        if line.startswith('!Sample_geo_accession'):
            gsms = [s.strip().strip('"') for s in line.strip().split('\t')[1:]]
        elif line.startswith('!Sample_characteristics_ch1') and 'cancer type' in line.lower():
            parts = line.strip().split('\t')[1:]
            cancer_status = [p.strip().strip('"') for p in parts]
            break

# Build healthy mask (cancer field is empty or "" means cancer-free)
# Format: 'cancer type (icd-10): ' (empty after colon) = healthy; 'cancer type (icd-10): C50' = breast cancer
healthy_idx = []
healthy_gsms = []
for i, (gsm, c) in enumerate(zip(gsms, cancer_status)):
    # Strip the prefix
    val = c.replace('cancer type (icd-10):', '').strip()
    if val == '' or val == "":
        healthy_idx.append(i)
        healthy_gsms.append(gsm)
print(f'Total samples: {len(gsms)}', file=sys.stderr)
print(f'Healthy (cancer-free) samples: {len(healthy_idx)}', file=sys.stderr)
print(f'First 5 healthy GSMs: {healthy_gsms[:5]}', file=sys.stderr)

# Second pass: write β table for healthy only
with gzip.open(src, 'rt') as f:
    while True:
        line = f.readline()
        if not line: break
        if line.strip() == '!series_matrix_table_begin':
            break
    header = f.readline().strip().split('\t')
    header = [h.strip('"') for h in header]
    # column 0 is ID_REF; columns 1+ are sample β values in same order as gsms
    # Confirm:
    if header[1:] != gsms:
        print(f'WARN: header order vs gsms mismatch. header[1:5]={header[1:5]}, gsms[0:5]={gsms[0:5]}', file=sys.stderr)
        # Match by GSM name
        name_to_col = {h: i for i, h in enumerate(header[1:], start=1)}
        keep_cols = [0] + [name_to_col[g] for g in healthy_gsms if g in name_to_col]
    else:
        keep_cols = [0] + [i+1 for i in healthy_idx]  # +1 because col0 is ID_REF
    
    new_header = ['CpGs'] + healthy_gsms
    with open(out, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(new_header)
        n = 0
        for line in f:
            line = line.rstrip('\n')
            if line == '!series_matrix_table_end' or not line:
                continue
            parts = line.split('\t')
            cpg = parts[0].strip('"')
            if not cpg.startswith('cg'): continue
            row = [cpg] + [parts[i] for i in keep_cols[1:]]
            writer.writerow(row)
            n += 1
            if n % 100000 == 0:
                print(f'  {n} rows', file=sys.stderr)
print(f'Done: {n} CpG rows × {len(healthy_gsms)} healthy samples')
