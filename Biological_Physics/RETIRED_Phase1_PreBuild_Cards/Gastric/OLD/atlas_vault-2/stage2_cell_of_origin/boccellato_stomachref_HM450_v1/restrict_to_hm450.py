#!/usr/bin/env python3
"""
restrict_to_hm450.py
=====================
Restrict BoccellatoStomachRef v1 (EPIC 850K source) to HM450-platform CpG subset.

The EPIC source has 738,115 CpGs (after Boccellato 2022 SWAN+ChAMP filtering).
Only 380,467 of those CpGs (51.55%) exist on the HM450 platform. Direct
application of the EPIC atlas to TCGA HM450 sesame Level 3 substrate produces
~49% per-sample coverage, far below the CHK-2.8 substrate floor of 80%.

This script restricts the EPIC atlas to the HM450 intersection. Tile β values
are unchanged; only the CpG row-set is restricted.

This pattern mirrors VAL-117 ProstateRef amendment precedent — when an atlas
built on a richer substrate needs to be used on a sparser substrate, restrict
to the intersection rather than re-engineer the atlas.

Usage:
  python restrict_to_hm450.py <epic_source_csv> <hm450_probe_source.txt> <output_csv>

Where:
  - epic_source_csv: BoccellatoStomachRef v1 EPIC source
                     (SHA fbe1dbfdec..., 738,115 CpGs)
  - hm450_probe_source.txt: any TCGA HM450 sesame Level 3 β file
                            (provides the HM450 probe list, ~486K CpGs)
  - output_csv: HM450-restricted derivative
                (expected SHA f5a620a93a..., 380,467 CpGs)

Expected output SHA-256: f5a620a93aba40d0567346d156ce7ea2861f8ed38ee1bd669a4ff52b261fa390
Expected output size: 25,110,908 bytes
Expected n_CpGs retained: 380,467

Runtime: ~30 seconds, <500 MB memory.
"""

import csv
import hashlib
import os
import sys
from pathlib import Path

EXPECTED_EPIC_SHA = 'fbe1dbfdeceb87a1f28c5737f0c3d8b6f86614dee5b9dfeb525741d3e4ef4d11'
EXPECTED_OUTPUT_SHA = 'f5a620a93aba40d0567346d156ce7ea2861f8ed38ee1bd669a4ff52b261fa390'
EXPECTED_OUTPUT_SIZE = 25110908
EXPECTED_N_CPGS_RETAINED = 380467


def load_hm450_probes(path):
    """Enumerate HM450 CpG probes from a TCGA sesame Level 3 β file."""
    probes = set()
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            parts = line.split('\t')
            if parts[0] in ('Composite Element REF', 'CpG_ID') or parts[0].startswith('#'):
                continue
            probes.add(parts[0])
    return probes


def verify_input_sha(path, expected_sha):
    with open(path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    if sha != expected_sha:
        raise ValueError(f'Input SHA mismatch for {path}: got {sha}, expected {expected_sha}')
    return sha


def restrict_atlas(epic_csv, hm450_source_path, output_csv):
    """Restrict EPIC atlas to HM450 probe subset."""
    print(f'Verifying EPIC source SHA-256...')
    verify_input_sha(epic_csv, EXPECTED_EPIC_SHA)
    print(f'  OK: {EXPECTED_EPIC_SHA}')

    print(f'Loading HM450 probe list from {hm450_source_path}...')
    hm450_probes = load_hm450_probes(hm450_source_path)
    print(f'  HM450 probes: {len(hm450_probes):,}')

    n_input = 0
    n_kept = 0
    with open(epic_csv) as f_in, open(output_csv, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        header = next(reader)
        writer.writerow(header)
        for row in reader:
            n_input += 1
            if row[0] in hm450_probes:
                writer.writerow(row)
                n_kept += 1

    print(f'\nAtlas restriction:')
    print(f'  Input EPIC CpGs:  {n_input:,}')
    print(f'  HM450 retained:   {n_kept:,}')
    print(f'  Dropped:          {n_input - n_kept:,}')
    print(f'  Retention rate:   {n_kept / n_input * 100:.2f}%')

    # Verify output
    with open(output_csv, 'rb') as f:
        out_sha = hashlib.sha256(f.read()).hexdigest()
    out_size = os.path.getsize(output_csv)

    print(f'\nOutput SHA-256: {out_sha}')
    print(f'Output size:    {out_size:,} bytes')

    if n_kept != EXPECTED_N_CPGS_RETAINED:
        print(f'WARN: n_kept {n_kept} != expected {EXPECTED_N_CPGS_RETAINED}')
    if out_sha != EXPECTED_OUTPUT_SHA:
        print(f'WARN: output SHA mismatch (expected {EXPECTED_OUTPUT_SHA})')
    if out_size != EXPECTED_OUTPUT_SIZE:
        print(f'WARN: output size {out_size} != expected {EXPECTED_OUTPUT_SIZE}')


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    restrict_atlas(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))


if __name__ == '__main__':
    main()
