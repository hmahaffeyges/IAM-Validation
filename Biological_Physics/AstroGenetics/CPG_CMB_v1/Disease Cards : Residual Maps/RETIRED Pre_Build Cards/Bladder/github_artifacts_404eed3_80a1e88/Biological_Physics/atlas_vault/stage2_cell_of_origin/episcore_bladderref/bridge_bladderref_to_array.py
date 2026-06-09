#!/usr/bin/env python3
"""
===============================================================================
EpiSCORE BladderRef Entrez → 450K CpG bridge

Source: Zhu T, Liu J, Beck S, Pan S, Capper D, Lechner M, Thirlwell C,
        Breeze CE, Teschendorff AE. "A pan-tissue DNA-methylation atlas
        based on deconvolution of major cell-types." Nature Methods
        2022;19:296. DOI: 10.1038/s41592-022-01412-7
        Repository: https://github.com/aet21/EpiSCORE

Methodology: Mirror of bridge_prostateref_to_array.py from VAL-117 prostate
sprint. EpiSCORE distributes the bladder reference matrix `mrefBladder.m`
indexed by Entrez Gene IDs (163 markers × 4 cell types + weight column). The
production scorer reads CpG-level reference matrices. Bridge:

  1. Load mrefBladder.m (163 Entrez × 5 cols: EC, Epi, Fib, IC, weight)
     from atlas_vault: BladderRef__mrefBladder_m.csv
  2. Load probeInfo450k.lv (the EpiSCORE-distributed 450K-CpG → Entrez
     Gene ID mapping; 485,577 array probes; 331,229 with EID; 19,357
     unique EIDs)
  3. For every probeInfo entry whose EID is in the 163 BladderRef Entrez
     IDs, emit a (probeID, EID, EC, Epi, Fib, IC, weight) row. The
     Entrez-level methylation profile is broadcast to every 450K CpG
     probe mapping to that gene.

Cell types:
  - EC  (vascular endothelial)
  - Epi (urothelial epithelium — bladder cancer cell of origin)
  - Fib (fibroblast / stromal)
  - IC  (immune cells, intra-bladder)

Output: episcore_bladderref_cpg_bridged.csv (production-ready), plus
audit-trail Entrez-keyed reference.

This is the SAME bridging methodology used for VAL-094 (BreastRef bridge),
VAL-111 (HeartRef bridge), and VAL-117 (ProstateRef bridge).
===============================================================================
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────

VAULT_ROOT = Path(__file__).resolve().parent
BLADDERREF_SRC = VAULT_ROOT.parent / 'episcore_zhu_teschendorff_2022' / 'BladderRef__mrefBladder_m.csv'
PROBEINFO_CSV = Path('/home/claude/episcore_source/probeInfo450k_lv.csv')

OUTPUT_BRIDGED = VAULT_ROOT / 'episcore_bladderref_cpg_bridged.csv'
OUTPUT_ENTREZ = VAULT_ROOT / 'episcore_bladderref_entrez_matrix.csv'

CELL_TYPES = ['EC', 'Epi', 'Fib', 'IC']


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print('=' * 78)
    print('EpiSCORE BladderRef Entrez → 450K CpG bridge')
    print('=' * 78)

    # Step 1 — Load BladderRef mref Entrez matrix
    print(f'\n[1] Loading BladderRef Entrez matrix: {BLADDERREF_SRC}')
    print(f'    SHA-256: {sha256_of_file(BLADDERREF_SRC)}')

    mref = pd.read_csv(BLADDERREF_SRC)
    mref = mref.rename(columns={'marker_ID': 'EID'})
    mref['EID'] = mref['EID'].astype(int)
    print(f'    Shape: {mref.shape}')
    print(f'    Cell types: {[c for c in mref.columns if c != "EID"]}')
    print(f'    Unique EIDs: {mref["EID"].nunique()}')
    n_source_eids = mref['EID'].nunique()

    # Save Entrez-keyed audit-trail copy (with index)
    mref.set_index('EID').to_csv(OUTPUT_ENTREZ)
    print(f'    Saved Entrez audit copy: {OUTPUT_ENTREZ}')

    # Step 2 — Load probeInfo450k.lv mapping
    print(f'\n[2] Loading probeInfo450k.lv mapping: {PROBEINFO_CSV}')
    print(f'    SHA-256: {sha256_of_file(PROBEINFO_CSV)}')
    probe_info = pd.read_csv(PROBEINFO_CSV, na_values=['NA'])
    print(f'    Total array rows: {len(probe_info)}')
    print(f'    Rows with EID: {probe_info["EID"].notna().sum()}')

    # Step 3 — Join
    print(f'\n[3] Joining probeInfo (EID) ↔ BladderRef mref (EID) ...')

    # Only consider probes with EID and that EID in BladderRef
    probe_info_with_eid = probe_info.dropna(subset=['EID']).copy()
    probe_info_with_eid['EID'] = probe_info_with_eid['EID'].astype(int)

    bladderref_eids = set(mref['EID'].tolist())
    probe_subset = probe_info_with_eid[probe_info_with_eid['EID'].isin(bladderref_eids)].copy()
    print(f'    Probes mapping to BladderRef EIDs: {len(probe_subset)}')
    print(f'    Unique BladderRef EIDs covered: {probe_subset["EID"].nunique()} / {n_source_eids}')

    eids_unmapped = bladderref_eids - set(probe_subset['EID'].tolist())
    if eids_unmapped:
        print(f'    EIDs with NO probeInfo450k mapping (dropped): {sorted(eids_unmapped)}')

    # Build the bridged matrix
    bridged = probe_subset[['probeID', 'EID']].merge(mref, on='EID', how='left')

    # CHK-3.1C dedup gate — count duplicates BEFORE drop
    n_total = len(bridged)
    dups = bridged['probeID'].duplicated().sum()
    print(f'\n[4] CHK-3.1C atlas dedup gate:')
    print(f'    Total rows: {n_total}')
    print(f'    Duplicate probeIDs: {dups}')
    if dups > 0:
        print(f'    De-duplicating (keep first occurrence)…')
        bridged = bridged.drop_duplicates(subset=['probeID'], keep='first')
        print(f'    After dedup: {len(bridged)} unique probeIDs')

    # Reorder columns
    cols = ['probeID', 'EID'] + CELL_TYPES + ['weight']
    bridged = bridged[cols]

    # Step 5 — Save (na_rep='nan' to match prostate VAL-117 bridged format
    # — load_atlas in val_calibrate scripts expects float('nan'), not empty string)
    bridged.to_csv(OUTPUT_BRIDGED, index=False, quoting=csv.QUOTE_ALL, na_rep='nan')
    print(f'\n[5] Saved bridged matrix: {OUTPUT_BRIDGED}')
    print(f'    Final dimensions: {bridged.shape[0]} unique 450K CpGs × '
          f'{len(CELL_TYPES)} cell types + weight')
    print(f'    SHA-256: {sha256_of_file(OUTPUT_BRIDGED)}')

    # Summary
    print('\n' + '=' * 78)
    print('SUMMARY')
    print('=' * 78)
    print(f'Source: BladderRef.rda from EpiSCORE (Zhu 2022 Nat Methods)')
    print(f'Source Entrez count: {n_source_eids}')
    print(f'Source SHA: {sha256_of_file(BLADDERREF_SRC)}')
    print(f'Cell types: {CELL_TYPES}')
    print(f'Bridged 450K CpG count: {len(bridged)}')
    print(f'Unique EIDs covered: {bridged["EID"].nunique()}')
    print(f'EIDs unmapped: {len(eids_unmapped)}')
    print(f'CHK-3.1C dedup: {"PASS" if dups == 0 else f"FIXED ({dups} dupes removed)"}')
    print(f'Bridged SHA-256: {sha256_of_file(OUTPUT_BRIDGED)}')


if __name__ == '__main__':
    main()
