#!/usr/bin/env python3
"""
===============================================================================
EpiSCORE OEref (Oral Epithelium) Entrez → 450K CpG bridge

Source: Zhu T, Liu J, Beck S, Pan S, Capper D, Lechner M, Thirlwell C,
        Breeze CE, Teschendorff AE. "A pan-tissue DNA-methylation atlas
        based on deconvolution of major cell-types." Nature Methods
        2022;19:296. DOI: 10.1038/s41592-022-01412-7
        Repository: https://github.com/aet21/EpiSCORE

Methodology: Mirror of bridge_bladderref_to_array.py from VAL-119 bladder
sprint. EpiSCORE distributes the oral-epithelium reference matrix
`mrefOE.m` indexed by Entrez Gene IDs (327 markers × 9 cell types + weight
column). Bridge:

  1. Load OEref__mrefOE_m.csv (327 Entrez × 10 cols: Basal, Fib, Gland,
     Macro, NeuIm, NeuMa, Peri, Plasma, Tcell, weight)
  2. Load probeInfo450k.lv mapping
  3. Broadcast Entrez-level methylation profile to every 450K CpG probe
     mapping to that gene.

Cell types covered:
  - Basal     (oral squamous epithelium — basal layer; stem-like, HNSCC cell of origin)
  - Fib       (fibroblast / stromal)
  - Gland     (oral submucosal glands — minor salivary glands)
  - Macro     (macrophages)
  - NeuIm     (neutrophils, immature)
  - NeuMa     (neutrophils, mature)
  - Peri      (pericytes / smooth muscle around small vessels)
  - Plasma    (plasma cells)
  - Tcell     (T lymphocytes)

Use within sprint: OEref Basal tile is the closest available squamous
epithelial reference at the molecular level for ESCC discrimination
(esophageal squamous epithelium and oral squamous epithelium share basal-
keratinocyte programs). However, oral cavity has different exposure
history (smokeless tobacco, betel, alcohol direct contact) than esophagus,
so OEref is reported as a CONFIRMATORY squamous reference, NOT the primary
ESCC cell-of-origin reference. EsoRef Epi_basal is the primary reference;
OEref Basal cross-checks it.

Output: episcore_oeref_cpg_bridged.csv (production-ready), plus
audit-trail Entrez-keyed reference.
===============================================================================
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

VAULT_ROOT = Path(__file__).resolve().parent
OEREF_SRC = VAULT_ROOT.parent / 'episcore_zhu_teschendorff_2022' / 'OEref__mrefOE_m.csv'
PROBEINFO_CSV = Path('/home/claude/episcore_source/probeInfo450k_lv.csv')

OUTPUT_BRIDGED = VAULT_ROOT / 'episcore_oeref_cpg_bridged.csv'
OUTPUT_ENTREZ = VAULT_ROOT / 'episcore_oeref_entrez_matrix.csv'

CELL_TYPES = ['Basal', 'Fib', 'Gland', 'Macro', 'NeuIm', 'NeuMa',
              'Peri', 'Plasma', 'Tcell']


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    print('=' * 78)
    print('EpiSCORE OEref Entrez → 450K CpG bridge')
    print('=' * 78)

    print(f'\n[1] Loading OEref Entrez matrix: {OEREF_SRC}')
    print(f'    SHA-256: {sha256_of_file(OEREF_SRC)}')

    mref = pd.read_csv(OEREF_SRC)
    mref = mref.rename(columns={'marker_ID': 'EID'})
    mref['EID'] = mref['EID'].astype(int)
    print(f'    Shape: {mref.shape}')
    print(f'    Cell types: {[c for c in mref.columns if c != "EID"]}')
    print(f'    Unique EIDs: {mref["EID"].nunique()}')
    n_source_eids = mref['EID'].nunique()

    mref.set_index('EID').to_csv(OUTPUT_ENTREZ)
    print(f'    Saved Entrez audit copy: {OUTPUT_ENTREZ}')

    print(f'\n[2] Loading probeInfo450k.lv mapping: {PROBEINFO_CSV}')
    print(f'    SHA-256: {sha256_of_file(PROBEINFO_CSV)}')
    probe_info = pd.read_csv(PROBEINFO_CSV, na_values=['NA'])
    print(f'    Total array rows: {len(probe_info)}')
    print(f'    Rows with EID: {probe_info["EID"].notna().sum()}')

    print(f'\n[3] Joining probeInfo (EID) ↔ OEref mref (EID) ...')
    probe_info_with_eid = probe_info.dropna(subset=['EID']).copy()
    probe_info_with_eid['EID'] = probe_info_with_eid['EID'].astype(int)

    oeref_eids = set(mref['EID'].tolist())
    probe_subset = probe_info_with_eid[probe_info_with_eid['EID'].isin(oeref_eids)].copy()
    print(f'    Probes mapping to OEref EIDs: {len(probe_subset)}')
    print(f'    Unique OEref EIDs covered: {probe_subset["EID"].nunique()} / {n_source_eids}')

    eids_unmapped = oeref_eids - set(probe_subset['EID'].tolist())
    if eids_unmapped:
        print(f'    EIDs with NO probeInfo450k mapping (dropped): {sorted(eids_unmapped)}')

    bridged = probe_subset[['probeID', 'EID']].merge(mref, on='EID', how='left')

    n_total = len(bridged)
    dups = bridged['probeID'].duplicated().sum()
    print(f'\n[4] CHK-3.1C atlas dedup gate:')
    print(f'    Total rows: {n_total}')
    print(f'    Duplicate probeIDs: {dups}')
    if dups > 0:
        print(f'    De-duplicating (keep first occurrence)…')
        bridged = bridged.drop_duplicates(subset=['probeID'], keep='first')
        print(f'    After dedup: {len(bridged)} unique probeIDs')

    cols = ['probeID', 'EID'] + CELL_TYPES + ['weight']
    bridged = bridged[cols]

    bridged.to_csv(OUTPUT_BRIDGED, index=False, quoting=csv.QUOTE_ALL, na_rep='nan')
    print(f'\n[5] Saved bridged matrix: {OUTPUT_BRIDGED}')
    print(f'    Final dimensions: {bridged.shape[0]} unique 450K CpGs × '
          f'{len(CELL_TYPES)} cell types + weight')
    print(f'    SHA-256: {sha256_of_file(OUTPUT_BRIDGED)}')

    print('\n' + '=' * 78)
    print('SUMMARY')
    print('=' * 78)
    print(f'Source: OEref.rda from EpiSCORE (Zhu 2022 Nat Methods)')
    print(f'Source Entrez count: {n_source_eids}')
    print(f'Source SHA: {sha256_of_file(OEREF_SRC)}')
    print(f'Cell types: {CELL_TYPES}')
    print(f'Bridged 450K CpG count: {len(bridged)}')
    print(f'Unique EIDs covered: {bridged["EID"].nunique()}')
    print(f'EIDs unmapped: {len(eids_unmapped)}')
    print(f'CHK-3.1C dedup: {"PASS" if dups == 0 else f"FIXED ({dups} dupes removed)"}')
    print(f'Bridged SHA-256: {sha256_of_file(OUTPUT_BRIDGED)}')


if __name__ == '__main__':
    main()
