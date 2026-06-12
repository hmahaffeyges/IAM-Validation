#!/usr/bin/env python3
"""
build_boccellato_stomachref_v1.py
==================================
Build BoccellatoStomachRef v1 (EPIC source) atlas reference matrix.

Input:  GSE141660 EPIC sub-platform series matrix (GPL21145), 18 mucosoid samples.
Output: 738,115 CpGs × 6 tiles, mean β across 3 donor reps per (region, state).

Source preprocessing applied by Boccellato 2022 authors (already in GEO deposit):
  - SWAN normalization + ChAMP filtering
  - detection p > 0.01 dropped
  - bead-count < 3 dropped
  - SNP-overlapping CpGs per Zhou 2016 dropped
  - multi-mapping probes per Nordlund 2013 dropped
  - sex-chromosome probes dropped
  - 738,115 CpGs survive filtering

Reference: Fritsche K, Boccellato F et al. Clin Epigenetics 2022;14:193.
           DOI 10.1186/s13148-022-01406-4. PMID 36585699.

Expected output SHA-256: fbe1dbfdeceb87a1f28c5737f0c3d8b6f86614dee5b9dfeb525741d3e4ef4d11
Expected output size:    48,715,676 bytes
Expected n_CpGs:         738,115
Expected tiles:          6 (Antrum_undiff, Antrum_diff, Corpus_undiff, Corpus_diff, Fundus_undiff, Fundus_diff)

Runtime: ~6 minutes wall, <1 GB memory.
"""

import gzip
import csv
import hashlib
import os
import sys
from pathlib import Path

# Sample → tile assignment (3 donors × 3 regions × 2 differentiation states = 18 samples)
TILE_ASSIGNMENT = {
    'GSM4210705': 'Antrum_undiff', 'GSM4210706': 'Antrum_undiff', 'GSM4210707': 'Antrum_undiff',
    'GSM4210708': 'Antrum_diff',   'GSM4210709': 'Antrum_diff',   'GSM4210710': 'Antrum_diff',
    'GSM4210711': 'Corpus_undiff', 'GSM4210712': 'Corpus_undiff', 'GSM4210713': 'Corpus_undiff',
    'GSM4210714': 'Corpus_diff',   'GSM4210715': 'Corpus_diff',   'GSM4210716': 'Corpus_diff',
    'GSM4210717': 'Fundus_undiff', 'GSM4210718': 'Fundus_undiff', 'GSM4210719': 'Fundus_undiff',
    'GSM4210720': 'Fundus_diff',   'GSM4210721': 'Fundus_diff',   'GSM4210722': 'Fundus_diff',
}
TILE_NAMES = ['Antrum_undiff', 'Antrum_diff', 'Corpus_undiff', 'Corpus_diff',
              'Fundus_undiff', 'Fundus_diff']

EXPECTED_INPUT_SHA = 'd43bd068645c9f9d2e63fb704d1f7caa4b02c137b0e007721d3f973738b25b04'
EXPECTED_OUTPUT_SHA = 'fbe1dbfdeceb87a1f28c5737f0c3d8b6f86614dee5b9dfeb525741d3e4ef4d11'
EXPECTED_OUTPUT_SIZE = 48715676
EXPECTED_N_CPGS = 738115


def verify_input(path):
    """Verify input series matrix SHA-256 matches sealed value."""
    with open(path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    if sha != EXPECTED_INPUT_SHA:
        raise ValueError(f'Input SHA mismatch: got {sha}, expected {EXPECTED_INPUT_SHA}')
    print(f'Input SHA verified: {sha}')


def build_atlas(input_path, output_path):
    """Stream the GEO matrix and emit per-tile mean β reference."""
    verify_input(input_path)

    n_cpgs = 0
    with gzip.open(input_path, 'rt') as f_in, open(output_path, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['CpG_ID'] + TILE_NAMES)

        in_data = False
        header = None
        for line in f_in:
            line = line.rstrip('\n')
            if line == '!series_matrix_table_begin':
                in_data = True
                continue
            if line == '!series_matrix_table_end':
                break
            if not in_data:
                continue

            fields = [v.strip('"') for v in line.split('\t')]
            if header is None:
                header = fields
                continue

            cpg_id = fields[0]
            tile_means = []
            for tile in TILE_NAMES:
                betas = []
                for i, gsm in enumerate(header[1:], start=1):
                    if TILE_ASSIGNMENT.get(gsm) == tile:
                        try:
                            betas.append(float(fields[i]))
                        except (ValueError, IndexError):
                            pass
                if len(betas) == 3:
                    tile_means.append(f'{sum(betas) / 3.0:.6f}')
                else:
                    tile_means.append('')

            writer.writerow([cpg_id] + tile_means)
            n_cpgs += 1

    print(f'Wrote {n_cpgs} CpG rows to {output_path}')

    # Verify output
    with open(output_path, 'rb') as f:
        out_sha = hashlib.sha256(f.read()).hexdigest()
    out_size = os.path.getsize(output_path)

    print(f'Output SHA-256: {out_sha}')
    print(f'Output size:    {out_size:,} bytes')

    if n_cpgs != EXPECTED_N_CPGS:
        print(f'WARN: n_cpgs {n_cpgs} != expected {EXPECTED_N_CPGS}')
    if out_sha != EXPECTED_OUTPUT_SHA:
        print(f'WARN: output SHA mismatch (expected {EXPECTED_OUTPUT_SHA})')
    if out_size != EXPECTED_OUTPUT_SIZE:
        print(f'WARN: output size {out_size} != expected {EXPECTED_OUTPUT_SIZE}')


def main():
    if len(sys.argv) != 3:
        print('Usage: build_boccellato_stomachref_v1.py <GSE141660_EPIC_matrix.txt.gz> <output.csv>')
        sys.exit(1)
    build_atlas(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == '__main__':
    main()
