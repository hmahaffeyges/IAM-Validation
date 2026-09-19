#!/usr/bin/env python3
"""Extract a GEO series matrix file's β-table to a Loyfer-CpG-subset CSV.

Reduces input from ~485K CpGs to ~6K Loyfer CpGs to save disk and downstream RAM.
Output is meth_atlas/deconvolve.py-compatible: CpGs as rows, samples as columns.

Usage:
    extract_series_matrix.py <input.gz> <output.csv> [<loyfer_cpgs.txt>]
"""
import gzip
import csv
import sys

if len(sys.argv) < 3:
    print(f'Usage: {sys.argv[0]} <input.gz> <output.csv> [loyfer_cpgs.txt]', file=sys.stderr)
    sys.exit(1)

src = sys.argv[1]
out = sys.argv[2]
loyfer_path = sys.argv[3] if len(sys.argv) > 3 else '/home/claude/ad_loyfer/loyfer_cpgs.txt'

with open(loyfer_path) as f:
    loyfer_set = set(line.strip() for line in f if line.strip())
print(f'Loyfer CpGs (deduped): {len(loyfer_set)}', file=sys.stderr)

# Find table start, then extract Loyfer CpG rows
with gzip.open(src, 'rt') as f:
    while True:
        line = f.readline()
        if not line: 
            print('ERROR: never found series_matrix_table_begin', file=sys.stderr)
            sys.exit(1)
        if line.strip() == '!series_matrix_table_begin':
            break
    header = f.readline().rstrip('\n').split('\t')
    header = [h.strip().strip('"') for h in header]
    header[0] = 'CpGs'
    
    n_kept = 0
    n_seen = 0
    with open(out, 'w', newline='') as fout:
        w = csv.writer(fout)
        w.writerow(header)
        for line in f:
            line = line.rstrip('\n')
            if line == '!series_matrix_table_end' or not line:
                continue
            parts = line.split('\t')
            cpg = parts[0].strip().strip('"')
            if not cpg.startswith('cg'): continue
            n_seen += 1
            if cpg in loyfer_set:
                w.writerow([cpg] + parts[1:])
                n_kept += 1
            if n_seen % 100000 == 0:
                print(f'  {n_seen} CpGs scanned, {n_kept} Loyfer matches kept', file=sys.stderr)

print(f'Done: {n_kept} Loyfer CpGs from {n_seen} total scanned ({len(header)-1} samples) → {out}')
