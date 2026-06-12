#!/usr/bin/env python3
"""
KIRC+PRAD adjacent-normal anchor scorer for VAL-126.
Reuses the VAL-126 atlas loading + scoring; emits anchor distributions per atlas/tile.

Runs in-process (no per-chunk subprocess) for speed since the anchor cohort is
a one-shot reference build.
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

# Reuse VAL-126 scorer module
sys.path.insert(0, '/home/claude/gastric_esophageal_sprint/VAL-126_stad_phase_c')
import importlib.util
spec = importlib.util.spec_from_file_location('val126', '/home/claude/gastric_esophageal_sprint/VAL-126_stad_phase_c/val126_stad_phase_c.py')
val126 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(val126)

OUTPUT_DIR = Path('/home/claude/gastric_esophageal_sprint/VAL-126_stad_phase_c')
ANCHOR_BETAS = OUTPUT_DIR / 'anchor_betas'
ANCHOR_MANIFEST = OUTPUT_DIR / 'val106_anchor_kirc_prad_manifest.json'
ANCHOR_NDJSON = OUTPUT_DIR / 'val106_anchor_per_sample.ndjson'
GDC_DATA = 'https://api.gdc.cancer.gov/data/'

CHUNK_SIZE = 30  # smaller anchor chunks


def load_done():
    if not ANCHOR_NDJSON.exists():
        return set()
    done = set()
    with open(ANCHOR_NDJSON) as f:
        for line in f:
            try:
                row = json.loads(line)
                if 'file_id' in row:
                    done.add(row['file_id'])
            except json.JSONDecodeError:
                continue
    return done


def download_chunk(records):
    ANCHOR_BETAS.mkdir(parents=True, exist_ok=True)
    needed = [r for r in records if not (ANCHOR_BETAS / r['file_name']).exists()]
    if not needed:
        return
    print(f'  Downloading {len(needed)} β files...')
    t0 = time.time()
    for i, r in enumerate(needed):
        out_path = ANCHOR_BETAS / r['file_name']
        url = f"{GDC_DATA}{r['file_id']}"
        result = subprocess.run(
            ['curl', '-sSL', '-o', str(out_path), url],
            capture_output=True, timeout=120
        )
        if result.returncode != 0:
            print(f'    FAIL {r["submitter_id"]}')
            if out_path.exists():
                out_path.unlink()
            continue
        if (i + 1) % 10 == 0 or (i + 1) == len(needed):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f'    [{i+1}/{len(needed)}] downloaded, rate={rate:.2f}/s')


def score_file(beta_path, record, atlases):
    """Score a single β file across all atlases. Returns row dict."""
    # Load β values
    sample_betas = {}
    n_total = 0
    n_extreme = 0
    n_middle = 0
    with open(beta_path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            parts = line.split('\t')
            if parts[0] in ('Composite Element REF',) or parts[0].startswith('#'):
                continue
            if len(parts) >= 2 and parts[1] not in ('NA', '', 'nan'):
                try:
                    b = float(parts[1])
                    if 0 <= b <= 1:
                        sample_betas[parts[0]] = b
                        n_total += 1
                        if b < 0.1 or b > 0.9:
                            n_extreme += 1
                        if 0.4 <= b <= 0.6:
                            n_middle += 1
                except ValueError:
                    pass
    f_extreme = n_extreme / n_total if n_total > 0 else 0
    f_middle = n_middle / n_total if n_total > 0 else 0

    row = {
        'file_id': record['file_id'],
        'submitter_id': record['submitter_id'],
        'project': record['project'],
        'sample_type': record['sample_type'],
        'n_valid_betas': n_total,
        'f_extreme': f_extreme,
        'f_middle': f_middle,
    }

    # Score each atlas
    xu538 = atlases['xu538']
    xu_present = sum(1 for c in xu538 if c in sample_betas)
    row['n_cpgs_xu538'] = xu_present
    row['coverage_xu538'] = xu_present / len(xu538)
    # Stage 1 Xu-538 pooled-entropy A-score (shannon-entropy over present CpGs)
    xu_betas = [sample_betas[c] for c in xu538 if c in sample_betas]
    if xu_betas:
        # Pooled shannon entropy = -p*log2(p) - (1-p)*log2(1-p) for each CpG, then mean
        h_vals = []
        for b in xu_betas:
            if 0 < b < 1:
                h = -b * np.log2(b) - (1 - b) * np.log2(1 - b)
                h_vals.append(h)
        H_min = val126.H_MIN['immune']
        if h_vals:
            row['A_xu538_stage1'] = float(np.mean(h_vals)) / H_min
        else:
            row['A_xu538_stage1'] = None
    else:
        row['A_xu538_stage1'] = None

    # Stage 2 + Stage 3 — score every atlas/tile using val126.score_atlas_tile
    for atlas_name, (atlas_data, tiles, classes) in atlases['stage2_3'].items():
        present = sum(1 for c in atlas_data if c in sample_betas)
        row[f'coverage_{atlas_name}'] = present / len(atlas_data)
        for tile in tiles:
            cls = classes.get(tile, 'secretory')  # default secretory if not assigned
            A, n = val126.score_atlas_tile(sample_betas, atlas_data, tile, val126.H_MIN[cls])
            row[f'A_{atlas_name}_{tile}'] = A
            row[f'n_{atlas_name}_{tile}'] = n

    return row


def main():
    with open(ANCHOR_MANIFEST) as f:
        manifest = json.load(f)
    n_total = len(manifest)
    print(f'KIRC+PRAD anchor: {n_total} files ({sum(1 for r in manifest if r["project"] == "TCGA-KIRC")} KIRC + {sum(1 for r in manifest if r["project"] == "TCGA-PRAD")} PRAD)')

    # Load atlases ONCE
    print('\nLoading atlases...')
    xu538 = val126.load_xu538(val126.XU538_PATH)
    print(f'  Xu-538: {len(xu538)} CpGs')
    loyfer, loyfer_tiles = val126.load_loyfer(val126.LOYFER_PATH)
    print(f'  Loyfer: {len(loyfer)} CpGs, {len(loyfer_tiles)} tiles')
    bocc, _ = val126.load_loyfer(val126.BOCC_PATH)
    print(f'  Boccellato: {len(bocc)} CpGs')
    esoref, _ = val126.load_loyfer(val126.ESOREF_PATH)
    print(f'  EsoRef: {len(esoref)} CpGs')
    oeref, _ = val126.load_loyfer(val126.OEREF_PATH)
    print(f'  OEref: {len(oeref)} CpGs')
    caggiano, caggiano_tiles = val126.load_loyfer(val126.CAGGIANO_PATH)
    print(f'  Caggiano: {len(caggiano)} CpGs, tiles: {caggiano_tiles}')
    salas, _ = val126.load_loyfer(val126.SALAS_PATH)
    print(f'  Salas IDOL 450K: {len(salas)} CpGs')
    unilife, unilife_tiles = val126.load_loyfer(val126.UNILIFE_PATH)
    print(f'  UniLIFE: {len(unilife)} CpGs, {len(unilife_tiles)} tiles: {unilife_tiles}')

    atlases = {
        'xu538': xu538,
        'stage2_3': {
            'loyfer': (loyfer, loyfer_tiles, val126.LOYFER_CLASSES),
            'bocc': (bocc, val126.BOCC_TILES, val126.BOCC_CLASSES),
            'esoref': (esoref, val126.ESOREF_TILES, val126.ESOREF_CLASSES),
            'oeref': (oeref, val126.OEREF_TILES, val126.OEREF_CLASSES),
            'cag': (caggiano, caggiano_tiles, val126.CAGGIANO_CLASSES),
            'salas': (salas, val126.SALAS_TILES, val126.SALAS_CLASSES),
            'uni': (unilife, unilife_tiles, {t: 'immune' for t in unilife_tiles}),
        }
    }

    done = load_done()
    print(f'\nAlready scored: {len(done)}/{n_total}')

    out_f = open(ANCHOR_NDJSON, 'a')

    for chunk_start in range(0, n_total, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_total)
        chunk = manifest[chunk_start:chunk_end]
        chunk_pending = [r for r in chunk if r['file_id'] not in done]
        if not chunk_pending:
            print(f'Chunk [{chunk_start}:{chunk_end}] complete, skipping')
            continue
        print(f'\n=== Anchor chunk [{chunk_start}:{chunk_end}] — {len(chunk_pending)} pending ===')

        # Disk check
        df = subprocess.run(['df', '-BG', '/home/claude'], capture_output=True, text=True)
        avail_g = int(df.stdout.split('\n')[1].split()[3].rstrip('G'))
        print(f'  Disk free: {avail_g} GB')

        download_chunk(chunk_pending)

        # Score each
        t0 = time.time()
        for k, rec in enumerate(chunk_pending):
            beta_path = ANCHOR_BETAS / rec['file_name']
            if not beta_path.exists():
                continue
            row = score_file(beta_path, rec, atlases)
            out_f.write(json.dumps(row) + '\n')
            out_f.flush()
            done.add(rec['file_id'])
            if (k + 1) % 5 == 0 or (k + 1) == len(chunk_pending):
                elapsed = time.time() - t0
                rate = (k + 1) / elapsed
                print(f'  [{k+1}/{len(chunk_pending)}] scored, rate={rate:.2f}/s')

        # Free chunk β files
        for rec in chunk_pending:
            p = ANCHOR_BETAS / rec['file_name']
            if p.exists():
                p.unlink()

    out_f.close()
    print(f'\n*** ANCHOR COMPLETE *** {len(done)}/{n_total}')


if __name__ == '__main__':
    main()
