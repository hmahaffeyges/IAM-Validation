#!/usr/bin/env python3
"""Extract AIBL GSE153712 to deconvolution-ready CSV (Loyfer CpGs only).

AIBL matrix layout: samples are rows, CpGs are columns (transpose of typical GEO).
Strategy: read header to find which columns contain the Loyfer CpGs, then stream
through samples extracting only those columns. Output is transposed (CpGs as rows,
samples as columns) to match meth_atlas deconvolve.py expected format.
"""
import gzip
import csv
import sys

src = '/home/claude/ad_loyfer/GSE153712_normalized_average_betas.txt.gz'
loyfer_cpgs_path = '/home/claude/ad_loyfer/loyfer_cpgs.txt'
out = '/home/claude/ad_loyfer/input/GSE153712_betas_loyfer.csv'

# Load Loyfer CpG list
with open(loyfer_cpgs_path) as f:
    loyfer_set = set(line.strip() for line in f if line.strip())
print(f'Loyfer CpGs to extract: {len(loyfer_set)}', file=sys.stderr)

# Pass 1: read header, find column indices for Loyfer CpGs
with gzip.open(src, 'rt') as f:
    header = f.readline().rstrip('\n').split('\t')
    # First column is empty (sample ID); rest are CpG IDs
    cpg_to_col = {}
    for i, name in enumerate(header):
        name = name.strip().strip('"')
        if name in loyfer_set:
            cpg_to_col[name] = i

print(f'Loyfer CpGs found in AIBL: {len(cpg_to_col)}', file=sys.stderr)
print(f'Loyfer CpGs missing: {len(loyfer_set) - len(cpg_to_col)}', file=sys.stderr)

# Build an ordered list: which CpGs do we extract, in what column index
loyfer_cpgs_present = sorted(cpg_to_col.keys())
keep_cols = [cpg_to_col[c] for c in loyfer_cpgs_present]

# Pass 2: stream samples, extract β values at those columns
sample_betas = {}  # sample_id -> [β values aligned to loyfer_cpgs_present]
with gzip.open(src, 'rt') as f:
    f.readline()  # skip header
    n = 0
    for line in f:
        line = line.rstrip('\n')
        if not line: continue
        parts = line.split('\t')
        sample_id = parts[0].strip().strip('"')
        if not sample_id: continue
        # Extract the values at keep_cols
        try:
            vals = [parts[i] for i in keep_cols]
        except IndexError:
            continue
        sample_betas[sample_id] = vals
        n += 1
        if n % 100 == 0:
            print(f'  {n} samples streamed', file=sys.stderr)
print(f'Total samples: {n}', file=sys.stderr)

# Write output: transposed - first column is CpG ID, then per-sample β columns
sample_order = sorted(sample_betas.keys())
with open(out, 'w', newline='') as fout:
    w = csv.writer(fout)
    w.writerow(['CpGs'] + sample_order)
    for cpg_idx, cpg in enumerate(loyfer_cpgs_present):
        row = [cpg] + [sample_betas[s][cpg_idx] for s in sample_order]
        w.writerow(row)

print(f'Done: {len(loyfer_cpgs_present)} CpGs × {len(sample_order)} samples → {out}')
