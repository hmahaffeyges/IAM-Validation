#!/usr/bin/env python3
"""
===============================================================================
EpiSCORE ProstateRef → 450K array CpG bridge

Mirrors the cardio-epic v0.3 HeartRef bridging methodology (sealed VAL-111).
Same source: EpiSCORE R package, Teschendorff et al. 2022 Nat Methods 19:296.
Same probeInfo450k.lv 450K-probe-to-Entrez-Gene-ID bridge.

Source matrix: ProstateRef__mrefProstate_m.csv (163 Entrez IDs × 6 cell types
                + weight column) from the EpiSCORE GitHub master branch
                (already in atlas_vault/stage2_cell_of_origin/
                episcore_zhu_teschendorff_2022/).
                SHA-256: 706a0910a6d284b268cd8d2bc6ffa131beadab9ee2d942a18178903e682a72a3

Output:        episcore_prostateref_cpg_bridged.csv
                — production-ready 450K CpG × prostate cell-type matrix.

Cell types (6):
  BE   = basal epithelial
  EC   = endothelial cells
  Fib  = fibroblasts
  LE   = luminal epithelial  (the prostate adenocarcinoma cell of origin)
  Leu  = leukocytes
  SM   = smooth muscle

Bridging methodology:
  1. Load mrefProstate.m (163 Entrez IDs × 7 cols incl. weight).
  2. Load probeInfo450k.lv (EpiSCORE-distributed 450K-probe → Entrez Gene ID
     bridge file). On disk at /home/claude/episcore_work/probeInfo450k.rda.
  3. For every probe whose mapped EID is in the 163 ProstateRef Entrez IDs,
     emit a (probeID, EID, BE, EC, Fib, LE, Leu, SM, weight) row.

This mirrors VAL-094 BreastRef bridging and VAL-111 HeartRef bridging
methodologies exactly.
===============================================================================
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

# probeInfo450k bridge extracted via R (extract_probeinfo.R) — same source
# that the HeartRef bridge used (probeInfo450k.lv from EpiSCORE GitHub).

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

ATLAS_VAULT = Path('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin')
EPISCORE_DIR = ATLAS_VAULT / 'episcore_zhu_teschendorff_2022'
PROSTATE_DIR = ATLAS_VAULT / 'episcore_prostateref'
PROSTATE_DIR.mkdir(exist_ok=True)

PROSTATEREF_SOURCE = EPISCORE_DIR / 'ProstateRef__mrefProstate_m.csv'
PROSTATEREF_SOURCE_SHA = '706a0910a6d284b268cd8d2bc6ffa131beadab9ee2d942a18178903e682a72a3'

PROBEINFO_BRIDGE_CSV = Path('/home/claude/episcore_work/probeInfo450k_bridge.csv')

OUTPUT_BRIDGED = PROSTATE_DIR / 'episcore_prostateref_cpg_bridged.csv'
OUTPUT_ENTREZ = PROSTATE_DIR / 'episcore_prostateref_entrez_matrix.csv'

CELL_TYPES = ['BE', 'EC', 'Fib', 'LE', 'Leu', 'SM']

# ──────────────────────────────────────────────────────────────────────────────
# LOAD SOURCE
# ──────────────────────────────────────────────────────────────────────────────

print('Loading ProstateRef source matrix...')
prostate_eids = {}  # eid -> dict of cell_type -> beta value (and weight)
with open(PROSTATEREF_SOURCE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        eid = row['marker_ID'].strip()
        prostate_eids[eid] = {
            ct: (float(row[ct]) if row[ct] not in ('', 'NA') else float('nan'))
            for ct in CELL_TYPES
        }
        prostate_eids[eid]['weight'] = (
            float(row['weight']) if row['weight'] not in ('', 'NA') else float('nan')
        )

print(f'  ProstateRef Entrez IDs: {len(prostate_eids)}')

# Verify SHA
with open(PROSTATEREF_SOURCE, 'rb') as f:
    actual_sha = hashlib.sha256(f.read()).hexdigest()
assert actual_sha == PROSTATEREF_SOURCE_SHA, f'SHA mismatch: {actual_sha}'
print('  Source SHA-256 verified.')

# Save audit-trail Entrez matrix
print('Writing Entrez audit-trail matrix...')
with open(OUTPUT_ENTREZ, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow([''] + CELL_TYPES + ['weight'])
    for eid, d in prostate_eids.items():
        w.writerow([eid] + [d[ct] for ct in CELL_TYPES] + [d['weight']])

# ──────────────────────────────────────────────────────────────────────────────
# LOAD probeInfo450k BRIDGE (R .rda file)
# ──────────────────────────────────────────────────────────────────────────────

print('Loading probeInfo450k.lv (extracted via R)...')
probe_pairs = []
with open(PROBEINFO_BRIDGE_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        eid = row['EID'].strip()
        if eid in ('NA', ''):
            continue
        probe_pairs.append((row['probeID'].strip(), eid))
print(f'  probeInfo450k entries: {len(probe_pairs)}')

# ──────────────────────────────────────────────────────────────────────────────
# BRIDGE
# ──────────────────────────────────────────────────────────────────────────────

print('Building bridge...')
bridged_rows = []
covered_eids = set()

for probe_id, eid_str in probe_pairs:
    if eid_str in prostate_eids:
        d = prostate_eids[eid_str]
        bridged_rows.append([
            probe_id,
            eid_str,
            d['BE'], d['EC'], d['Fib'], d['LE'], d['Leu'], d['SM'],
            d['weight'],
        ])
        covered_eids.add(eid_str)

print(f'  Unique 450K CpGs in bridged matrix: {len(bridged_rows)}')
print(f'  Unique Entrez IDs covered: {len(covered_eids)} / {len(prostate_eids)}')
uncovered = set(prostate_eids.keys()) - covered_eids
if uncovered:
    print(f'  Entrez IDs with no probeInfo450k mapping: {len(uncovered)}')
    print(f'    (e.g. {list(uncovered)[:10]})')

# Write production-ready bridged matrix
print('Writing bridged matrix...')
with open(OUTPUT_BRIDGED, 'w', newline='') as f:
    w = csv.writer(f, quoting=csv.QUOTE_ALL)
    w.writerow(['probeID', 'EID', 'BE', 'EC', 'Fib', 'LE', 'Leu', 'SM', 'weight'])
    for row in bridged_rows:
        w.writerow(row)

# Compute SHA of output
with open(OUTPUT_BRIDGED, 'rb') as f:
    bridged_sha = hashlib.sha256(f.read()).hexdigest()

# Write README
readme_content = f'''# EpiSCORE ProstateRef — prostate cell-type methylation reference (CpG-bridged)

## Source
- **Paper:** Zhu T, Liu J, Beck S, Pan S, Capper D, Lechner M, Thirlwell C, Breeze CE, Teschendorff AE. *A pan-tissue DNA-methylation atlas based on deconvolution of major cell-types.* Nature Methods 2022;19:296. DOI: [10.1038/s41592-022-01412-7](https://doi.org/10.1038/s41592-022-01412-7)
- **Repository:** https://github.com/aet21/EpiSCORE
- **Source file:** `data/ProstateRef.rda` (commit master @ 2026-04-30)

## Bridging methodology
EpiSCORE distributes the prostate reference matrix `mrefProstate.m` indexed by **Entrez Gene IDs** (163 markers × 6 cell types + weight column). The production scorer reads CpG-level reference matrices. Bridge methodology:

1. Load `mrefProstate.m` (163 Entrez × 7 columns) from the EpiSCORE-distributed `ProstateRef.rda` (in atlas_vault as `ProstateRef__mrefProstate_m.csv`).
2. Load `probeInfo450k.lv` (the EpiSCORE-distributed bridge from 450K CpG probes to Entrez Gene IDs).
3. For every probeInfo entry whose EID is in the 163 ProstateRef Entrez IDs, emit a (probeID, EID, BE, EC, Fib, LE, Leu, SM, weight) row. The Entrez-level methylation profile is broadcast to every 450K CpG probe mapping to that gene.

This is the **same** bridging methodology used for VAL-094 (BreastRef bridge) and VAL-111 (HeartRef bridge).

## Final dimensions
- **{len(bridged_rows)} unique 450K CpG probes × 6 prostate cell types**
- {len(covered_eids)} unique Entrez Gene IDs covered (of 163 source Entrez IDs)
- Cell types: **BE** (basal epithelial), **EC** (endothelial), **Fib** (fibroblast), **LE** (luminal epithelial — prostate adenocarcinoma cell of origin), **Leu** (leukocytes), **SM** (smooth muscle)
- `weight` column preserved for EpiSCORE's weighted-NNLS deconvolution mode

## Files
- `episcore_prostateref_cpg_bridged.csv` — production-ready CpG × cell-type matrix. **SHA-256:** `{bridged_sha}`
- `episcore_prostateref_entrez_matrix.csv` — original Entrez-keyed matrix (audit trail)
- (Source `ProstateRef__mrefProstate_m.csv` lives in the EpiSCORE pan-tissue folder and stays there as part of the broader pan-tissue MANIFEST.)

## License
GPL-2 per EpiSCORE repository.

## Used by
- VAL-114 — prostate-epic Stage 2 EpiSCORE ProstateRef calibration on TCGA-PRAD adjacent-normal n=50 (HM450 sesame Level 3)
- prostate-epic v0.3 production deployment Stage 2 layered atlas (prostate sub-cell-type resolution beyond Loyfer's single `prostate_epithelial` tile)

## Frozen
2026-04-30
'''

with open(PROSTATE_DIR / 'README.md', 'w') as f:
    f.write(readme_content)

print()
print('=' * 70)
print('DONE')
print('=' * 70)
print(f'  Bridged matrix:  {OUTPUT_BRIDGED}')
print(f'  SHA-256:         {bridged_sha}')
print(f'  Dimensions:      {len(bridged_rows)} CpGs × {len(CELL_TYPES)} cell types')
print(f'  README:          {PROSTATE_DIR / "README.md"}')
