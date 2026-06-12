#!/usr/bin/env python3
"""Extract GSE60274 (Lai 2015 brain tissue 450K) to meth_atlas-compatible CSV."""
import gzip, csv, sys, os

src = '/home/claude/glioma_work/GSE60274_series_matrix.txt.gz'
out = '/home/claude/brain_decon/input/GSE60274_betas.csv'

# Find table start
with gzip.open(src, 'rt') as f:
    while True:
        line = f.readline()
        if not line: break
        if line.strip() == '!series_matrix_table_begin':
            break
    # Next line is header
    header = f.readline().strip().split('\t')
    header = [h.strip('"') for h in header]
    header[0] = 'CpGs'  # match meth_atlas format
    
    with open(out, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        n = 0
        for line in f:
            line = line.strip()
            if line == '!series_matrix_table_end' or not line:
                continue
            parts = line.split('\t')
            cpg = parts[0].strip('"')
            # Skip non-CpG rows
            if not cpg.startswith('cg'):
                continue
            writer.writerow([cpg] + parts[1:])
            n += 1
            if n % 100000 == 0:
                print(f'  {n} rows written', file=sys.stderr)
print(f'Done: {n} CpG rows written to {out}')
print(f'Samples: {len(header)-1}')
