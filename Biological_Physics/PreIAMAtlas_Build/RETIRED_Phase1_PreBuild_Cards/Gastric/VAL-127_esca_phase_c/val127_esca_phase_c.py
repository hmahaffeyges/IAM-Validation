#!/usr/bin/env python3
"""
===============================================================================
VAL-127 — TCGA-ESCA Phase C run-everything scorer
===============================================================================
Pre-registered VAL-127 (sealed 2026-05-02, prereg SHA cb521d83afe8...).

Scores every TCGA-ESCA HM450 sesame Level 3 β file against:
  Stage 1: Xu-538 panel (1 pooled-entropy A-score)
  Stage 2: Layered Moss+Loyfer 25-tile + BoccellatoStomachRef_HM450 (6) +
           EsoRef bridged (8) + OEref bridged (9) + Caggiano TIM (19) = 67 tiles
  Stage 3: Salas IDOL 450K (6) + UniLIFE Guo 2025 (19) = 25 cell types

Per-sample output: 1 + 67 + 25 = 93 A-scores per IDAT.

Class-tier H_min anchors (G-003b MCMC frozen 2026-04-06):
  cycling:    0.8561
  immune:     0.8389
  secretory:  0.8433
  stromal:    0.8630
  terminal:   0.7728
  progenitor: 0.8522
  stem_adult: 0.8737
  stem_pluri: 0.9822

A_score formula: A_tile = mean(|sample_β - tile_ref_β|) / H_min_class

Resume-safe: appends per-sample row to NDJSON; checks idx for resume.
===============================================================================
"""

import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# H_MIN by class (G-003b MCMC posteriors frozen 2026-04-06)
# ──────────────────────────────────────────────────────────────────────────────
H_MIN = {
    'cycling':    0.8561,
    'immune':     0.8389,    # actually 0.838889 in calibration; use canonical
    'secretory':  0.843264,
    'stromal':    0.862950,
    'terminal':   0.7728,
    'progenitor': 0.8522,
    'stem_adult': 0.8737,
    'stem_pluri': 0.9822,
}

# ──────────────────────────────────────────────────────────────────────────────
# ATLAS PATHS + tile-class assignments
# ──────────────────────────────────────────────────────────────────────────────

XU538_PATH = '/tmp/iam-validation-check/Biological_Physics/validation_runs/xu538_panel.json'

# Loyfer 25-tile — class assignments per VAL-112
LOYFER_PATH = '/tmp/iam-validation-check/Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv'
# Canonical class assignments per VAL-118 sealed prostate Phase C (val118_stage2_score.py),
# extended to all 25 tiles for run-everything completeness in VAL-127 (8 added tiles
# are bulk-tissue epithelia → secretory class). This extension is the methodology
# gap-fill for VAL-127; CCL-class note logged.
LOYFER_CLASSES = {
    # Immune (per VAL-118)
    'Monocytes_EPIC': 'immune', 'B-cells_EPIC': 'immune', 'CD4T-cells_EPIC': 'immune',
    'NK-cells_EPIC': 'immune', 'CD8T-cells_EPIC': 'immune', 'Neutrophils_EPIC': 'immune',
    # Progenitor
    'Erythrocyte_progenitors': 'progenitor',
    # Stromal
    'Adipocytes': 'stromal', 'Vascular_endothelial_cells': 'stromal',
    # Cycling (per VAL-118)
    'Lung_cells': 'cycling', 'Pancreatic_duct_cells': 'cycling', 'Colon_epithelial_cells': 'cycling',
    # Secretory (per VAL-118)
    'Hepatocytes': 'secretory', 'Pancreatic_beta_cells': 'secretory', 'Pancreatic_acinar_cells': 'secretory',
    # Terminal (per VAL-118)
    'Left_atrium': 'terminal', 'Cortical_neurons': 'terminal',
    # Bulk-tissue epithelia — extended in VAL-127 to fill run-everything gap
    'Bladder': 'secretory',
    'Breast': 'secretory',
    'Head_and_neck_larynx': 'secretory',
    'Kidney': 'secretory',
    'Prostate': 'secretory',
    'Thyroid': 'secretory',
    'Upper_GI': 'secretory',
    'Uterus_cervix': 'secretory',
}

BOCC_PATH = '/tmp/iam-validation-check/Biological_Physics/atlas_vault/stage2_cell_of_origin/boccellato_stomachref_HM450_v1/boccellato_stomachref_HM450_v1.csv'
BOCC_TILES = ['Antrum_undiff', 'Antrum_diff', 'Corpus_undiff', 'Corpus_diff', 'Fundus_undiff', 'Fundus_diff']
BOCC_CLASSES = {t: 'secretory' for t in BOCC_TILES}

ESOREF_PATH = '/tmp/iam-validation-check/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_esoref/episcore_esoref_cpg_bridged.csv'
ESOREF_TILES = ['EC', 'Epi_basal', 'Epi_stratified', 'Epi_suprabasal', 'Epi_upper', 'Fib', 'Gland', 'IC']
ESOREF_CLASSES = {
    'EC': 'stromal', 'Epi_basal': 'secretory', 'Epi_stratified': 'secretory',
    'Epi_suprabasal': 'secretory', 'Epi_upper': 'secretory',
    'Fib': 'stromal', 'Gland': 'secretory', 'IC': 'immune',
}

OEREF_PATH = '/tmp/iam-validation-check/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_oeref/episcore_oeref_cpg_bridged.csv'
OEREF_TILES = ['Basal', 'Fib', 'Gland', 'Macro', 'NeuIm', 'NeuMa', 'Peri', 'Plasma', 'Tcell']
OEREF_CLASSES = {
    'Basal': 'secretory', 'Fib': 'stromal', 'Gland': 'secretory',
    'Macro': 'immune', 'NeuIm': 'immune', 'NeuMa': 'immune',
    'Peri': 'stromal', 'Plasma': 'immune', 'Tcell': 'immune',
}

CAGGIANO_PATH = '/tmp/iam-validation-check/Biological_Physics/atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/caggiano_tim_cpg_bridged.csv'
# Caggiano tiles — class per VAL-113
CAGGIANO_CLASSES = {
    'dendritic': 'immune', 'endothelial': 'stromal', 'eosinophil': 'immune',
    'erythroblast': 'progenitor', 'macrophage': 'immune', 'monocyte': 'immune',
    'neutrophil': 'immune', 'placenta': 'secretory', 'tcell': 'immune',
    'adipose': 'stromal', 'brain': 'terminal', 'fibroblast': 'stromal',
    'heart': 'terminal', 'hepatocyte': 'secretory', 'lung': 'secretory',
    'mammary': 'secretory', 'megakaryocyte': 'progenitor',
    'skeletal': 'terminal', 'small_intestine': 'secretory',
}

SALAS_PATH = '/tmp/iam-validation-check/Biological_Physics/atlas_vault/stage3_immune_fraction/salas_blood_epic_idol/IDOLOptimizedCpGs450k_compTable.csv'
SALAS_TILES = ['CD8T', 'CD4T', 'NK', 'Bcell', 'Mono', 'Neu']
SALAS_CLASSES = {t: 'immune' for t in SALAS_TILES}

UNILIFE_PATH = '/tmp/iam-validation-check/Biological_Physics/atlas_vault/stage3_immune_fraction/unilife_guo_2025/centUniLIFE_reference_matrix.csv'
# UniLIFE 19 cell types — all immune class
UNILIFE_TILES_DEFAULT = None  # detect from CSV header

# CHK-3.1A f_extreme threshold (informational only, not gated per VAL-118 precedent)
CHK_3_1A_F_EXTREME_HEALTHY = 0.505
CHK_3_1A_F_MIDDLE_HEALTHY = 0.090

# CHK-3.1B per-sample atlas-coverage threshold
CHK_3_1B_THRESHOLD = 0.80

OUTPUT_DIR = Path('/home/claude/gastric_esophageal_sprint/VAL-127_esca_phase_c')
PROGRESS_NDJSON = OUTPUT_DIR / 'val127_per_sample_progress.ndjson'
BETAS_DIR = OUTPUT_DIR / 'betas'


# ──────────────────────────────────────────────────────────────────────────────
# ATLAS LOADERS
# ──────────────────────────────────────────────────────────────────────────────

def load_xu538(path):
    """Return set of CpGs."""
    with open(path) as f:
        d = json.load(f)
    return set(d['cpgs'])


def load_loyfer(path):
    """Loyfer 25-tile: rows=CpGs, columns=tile names. Returns {cpg: {tile: ref_β}}."""
    atlas = {}
    tile_names = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        # First column is "Composite Element REF" or similar; rest are tile names
        # But Loyfer uses "Acceptor" then tile names — inspect first row
        # Actually Loyfer reference_atlas.csv format: CpGs as first col, tiles as remaining
        tile_names = header[1:]  # skip first col
        for row in reader:
            if not row or not row[0]:
                continue
            cpg = row[0].strip('"')
            tile_betas = {}
            for i, tile in enumerate(tile_names, start=1):
                v = row[i] if i < len(row) else ''
                if v in ('', 'NA', 'nan'):
                    continue
                try:
                    tile_betas[tile] = float(v)
                except ValueError:
                    continue
            if tile_betas:
                atlas[cpg] = tile_betas
    return atlas, tile_names


def load_csv_atlas(path, cell_types):
    """Generic loader for csv with first col=probeID and named cell-type columns."""
    atlas = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cpg = row.get('probeID', row.get('CpG_ID', row.get('cpg_id'))).strip('"') if row.get('probeID') or row.get('CpG_ID') or row.get('cpg_id') else None
            if not cpg:
                continue
            tile_betas = {}
            for tile in cell_types:
                v = row.get(tile, '')
                if v in ('', 'NA', 'nan', '"nan"'):
                    continue
                v = v.strip('"') if v else ''
                try:
                    tile_betas[tile] = float(v)
                except ValueError:
                    continue
            if tile_betas:
                atlas[cpg] = tile_betas
    return atlas


def load_unilife():
    """UniLIFE has its own format — CpGs as first col."""
    atlas = {}
    cell_types = []
    with open(UNILIFE_PATH) as f:
        reader = csv.reader(f)
        header = next(reader)
        cell_types = [c.strip('"') for c in header[1:]]
        for row in reader:
            if not row or not row[0]:
                continue
            cpg = row[0].strip('"')
            tile_betas = {}
            for i, tile in enumerate(cell_types, start=1):
                v = row[i] if i < len(row) else ''
                if v in ('', 'NA', 'nan'):
                    continue
                try:
                    tile_betas[tile] = float(v)
                except ValueError:
                    continue
            if tile_betas:
                atlas[cpg] = tile_betas
    return atlas, cell_types


# ──────────────────────────────────────────────────────────────────────────────
# SCORING
# ──────────────────────────────────────────────────────────────────────────────

def shannon_H(beta):
    """Binary Shannon entropy."""
    if beta is None or beta <= 0 or beta >= 1:
        return 0.0
    return -beta * math.log2(beta) - (1 - beta) * math.log2(1 - beta)


def load_beta_file(path, all_atlas_cpgs):
    """Stream a TCGA-ESCA β file. Return (full_array, atlas_betas_dict)."""
    full = []
    atlas_betas = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            cpg = parts[0]
            try:
                b = float(parts[1])
                if math.isnan(b):
                    continue
                full.append(b)
                if cpg in all_atlas_cpgs:
                    atlas_betas[cpg] = b
            except ValueError:
                continue
    return np.array(full), atlas_betas


def score_xu538_pooled_entropy(atlas_betas, xu538_cpgs):
    """Stage 1 Xu-538 pooled-entropy A-score: mean Shannon entropy across panel CpGs.
       Higher = more architectural drift (entropy-rich)."""
    H_vals = []
    for cpg in xu538_cpgs:
        if cpg in atlas_betas:
            H_vals.append(shannon_H(atlas_betas[cpg]))
    if not H_vals:
        return float('nan'), 0
    A = sum(H_vals) / len(H_vals) / H_MIN['immune']  # Stage 1 is immune-class indexed
    return float(A), len(H_vals)


def score_atlas_tile(atlas_betas, atlas, tile, h_min):
    """A_tile = mean(|sample_β - tile_ref_β|) / H_min."""
    deltas = []
    for cpg, refs in atlas.items():
        if cpg not in atlas_betas:
            continue
        if tile not in refs:
            continue
        deltas.append(abs(atlas_betas[cpg] - refs[tile]))
    if not deltas:
        return float('nan'), 0
    return float(sum(deltas) / len(deltas) / h_min), len(deltas)


def chk_3_1a(full_array):
    f_ex = float(np.mean((full_array < 0.1) | (full_array > 0.9)))
    f_mid = float(np.mean((full_array >= 0.4) & (full_array <= 0.6)))
    median = float(np.median(full_array))
    return f_ex, f_mid, median


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(start_idx, end_idx):
    print('=' * 78)
    print(f'VAL-127 — TCGA-ESCA Phase C run-everything scorer [{start_idx}, {end_idx})')
    print('=' * 78)

    # Load all atlases
    print('Loading atlases...')
    xu538 = load_xu538(XU538_PATH)
    print(f'  Xu-538: {len(xu538)} CpGs')

    loyfer, loyfer_tiles = load_loyfer(LOYFER_PATH)
    print(f'  Loyfer 25-tile: {len(loyfer)} CpGs × {len(loyfer_tiles)} tiles: {loyfer_tiles[:5]}...')

    bocc = load_csv_atlas(BOCC_PATH, BOCC_TILES)
    print(f'  BoccellatoStomachRef_HM450: {len(bocc)} CpGs')

    esoref = load_csv_atlas(ESOREF_PATH, ESOREF_TILES)
    print(f'  EsoRef: {len(esoref)} CpGs')

    oeref = load_csv_atlas(OEREF_PATH, OEREF_TILES)
    print(f'  OEref: {len(oeref)} CpGs')

    caggiano_tiles = list(CAGGIANO_CLASSES.keys())
    caggiano = load_csv_atlas(CAGGIANO_PATH, caggiano_tiles)
    print(f'  Caggiano TIM: {len(caggiano)} CpGs')

    # Salas IDOL — the 450K variant has CpG_ID first column
    salas = {}
    with open(SALAS_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cpg = row['CpG_ID']
            tile_betas = {}
            for tile in SALAS_TILES:
                v = row.get(tile, '')
                if v not in ('', 'NA', 'nan'):
                    try:
                        tile_betas[tile] = float(v)
                    except ValueError:
                        pass
            if tile_betas:
                salas[cpg] = tile_betas
    print(f'  Salas IDOL 450K: {len(salas)} CpGs')

    unilife, unilife_tiles = load_unilife()
    print(f'  UniLIFE Guo 2025: {len(unilife)} CpGs × {len(unilife_tiles)} cell types')

    # Build superset of CpGs to scan in each β file
    all_atlas_cpgs = set(xu538) | set(loyfer.keys()) | set(bocc.keys()) | set(esoref.keys()) | set(oeref.keys()) | set(caggiano.keys()) | set(salas.keys()) | set(unilife.keys())
    print(f'  Total unique atlas CpGs to track: {len(all_atlas_cpgs):,}')

    # Pre-compute tile class & H_min for each atlas
    # UniLIFE — sealed as immune class for all 19 cell types per Guo 2025 sorted-immune ref
    unilife_classes = {t: 'immune' for t in unilife_tiles}

    # Load manifest
    with open(OUTPUT_DIR / 'tcga_esca_hm450_manifest_FINAL.json') as f:
        manifest = json.load(f)

    # Sort manifest by file_name for stable ordering
    manifest_sorted = sorted(manifest, key=lambda r: r['file_name'])

    # Resume: scan existing NDJSON for completed idx
    completed = set()
    if PROGRESS_NDJSON.exists():
        with open(PROGRESS_NDJSON) as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed.add(rec['idx'])
                    except json.JSONDecodeError:
                        pass
    print(f'  Already completed: {len(completed)} samples')

    end_idx = min(end_idx, len(manifest_sorted))
    chunk = manifest_sorted[start_idx:end_idx]
    chunk_pending = [(start_idx + j, r) for j, r in enumerate(chunk) if (start_idx + j) not in completed]
    print(f'  Chunk pending: {len(chunk_pending)} of {len(chunk)} samples')

    t0 = time.time()
    with open(PROGRESS_NDJSON, 'a') as out_f:
        for k, (idx, r) in enumerate(chunk_pending):
            beta_path = BETAS_DIR / r['file_name']
            if not beta_path.exists():
                print(f'  WARN: {beta_path.name} missing on disk; skipping')
                continue

            full_array, atlas_betas = load_beta_file(beta_path, all_atlas_cpgs)
            f_ex, f_mid, median = chk_3_1a(full_array)
            n_genome = len(full_array)

            # Stage 1
            A_xu538, n_xu538 = score_xu538_pooled_entropy(atlas_betas, xu538)
            cov_xu538 = n_xu538 / len(xu538)

            # Stage 2 atlases
            row = {
                'idx': idx,
                'sample_id': r['submitter_id'],
                'sample_type': r['sample_type'],
                'file_id': r['file_id'],
                # CHK-3.1A informational
                'f_extreme': f_ex, 'f_middle': f_mid, 'median_beta': median, 'n_genome': n_genome,
                # Stage 1
                'A_xu538_stage1': A_xu538, 'n_cpgs_xu538': n_xu538, 'coverage_xu538': cov_xu538,
                # Subtype + clinical
                'SUBTYPE': r.get('SUBTYPE'),
                'MSI_SCORE_MANTIS': r.get('MSI_SCORE_MANTIS'),
                'MSI_SENSOR_SCORE': r.get('MSI_SENSOR_SCORE'),
                'EBV_PRESENT': r.get('EBV_PRESENT'),
                'H_PYLORI_INFECTION': r.get('H_PYLORI_INFECTION'),
                'primary_diagnosis': r.get('primary_diagnosis'),
                'gender': r.get('gender'),
                'ajcc_pathologic_stage': r.get('ajcc_pathologic_stage'),
                'site_of_resection': r.get('site_of_resection'),
                'age_at_index': r.get('age_at_index'),
            }

            # Stage 2: Loyfer 25 tiles
            loyfer_present = sum(1 for c in loyfer if c in atlas_betas)
            row['coverage_loyfer'] = loyfer_present / len(loyfer)
            for tile in loyfer_tiles:
                cls = LOYFER_CLASSES.get(tile, 'secretory')
                A, n = score_atlas_tile(atlas_betas, loyfer, tile, H_MIN[cls])
                row[f'A_loyfer_{tile}'] = A
                row[f'n_loyfer_{tile}'] = n

            # Stage 2: Boccellato
            bocc_present = sum(1 for c in bocc if c in atlas_betas)
            row['coverage_boccellato'] = bocc_present / len(bocc)
            for tile in BOCC_TILES:
                cls = BOCC_CLASSES[tile]
                A, n = score_atlas_tile(atlas_betas, bocc, tile, H_MIN[cls])
                row[f'A_bocc_{tile}'] = A
                row[f'n_bocc_{tile}'] = n

            # Stage 2: EsoRef
            eso_present = sum(1 for c in esoref if c in atlas_betas)
            row['coverage_esoref'] = eso_present / len(esoref)
            for tile in ESOREF_TILES:
                cls = ESOREF_CLASSES[tile]
                A, n = score_atlas_tile(atlas_betas, esoref, tile, H_MIN[cls])
                row[f'A_esoref_{tile}'] = A
                row[f'n_esoref_{tile}'] = n

            # Stage 2: OEref
            oe_present = sum(1 for c in oeref if c in atlas_betas)
            row['coverage_oeref'] = oe_present / len(oeref)
            for tile in OEREF_TILES:
                cls = OEREF_CLASSES[tile]
                A, n = score_atlas_tile(atlas_betas, oeref, tile, H_MIN[cls])
                row[f'A_oeref_{tile}'] = A
                row[f'n_oeref_{tile}'] = n

            # Stage 2: Caggiano TIM
            cag_present = sum(1 for c in caggiano if c in atlas_betas)
            row['coverage_caggiano'] = cag_present / len(caggiano)
            for tile in caggiano_tiles:
                cls = CAGGIANO_CLASSES[tile]
                A, n = score_atlas_tile(atlas_betas, caggiano, tile, H_MIN[cls])
                row[f'A_cag_{tile}'] = A
                row[f'n_cag_{tile}'] = n

            # Stage 3: Salas IDOL
            salas_present = sum(1 for c in salas if c in atlas_betas)
            row['coverage_salas'] = salas_present / len(salas)
            for tile in SALAS_TILES:
                cls = SALAS_CLASSES[tile]
                A, n = score_atlas_tile(atlas_betas, salas, tile, H_MIN[cls])
                row[f'A_salas_{tile}'] = A
                row[f'n_salas_{tile}'] = n

            # Stage 3: UniLIFE
            uni_present = sum(1 for c in unilife if c in atlas_betas)
            row['coverage_unilife'] = uni_present / len(unilife)
            for tile in unilife_tiles:
                A, n = score_atlas_tile(atlas_betas, unilife, tile, H_MIN['immune'])
                row[f'A_uni_{tile}'] = A
                row[f'n_uni_{tile}'] = n

            out_f.write(json.dumps(row) + '\n')
            out_f.flush()

            if (k + 1) % 20 == 0 or (k + 1) == len(chunk_pending):
                elapsed = time.time() - t0
                rate = (k + 1) / elapsed if elapsed > 0 else 0
                eta = (len(chunk_pending) - k - 1) / rate if rate > 0 else 0
                print(f'  [{idx + 1}/{len(manifest_sorted)}] processed (this chunk: {k+1}/{len(chunk_pending)}), rate={rate:.2f}/s, eta={eta:.0f}s')

    print(f'\nChunk complete. Elapsed: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: val127_esca_phase_c.py <start_idx> <end_idx>')
        sys.exit(1)
    main(int(sys.argv[1]), int(sys.argv[2]))
