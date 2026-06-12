#!/usr/bin/env python3
"""
VAL-128 chunked-pass scorer (Option A).

Streams the 2.2 GB GSE87650 series matrix once per chunk of 60 samples,
4 chunks total (samples 0-59, 60-119, 120-179, 180-239). Each chunk fits
in ~1.7 GB RAM (60 samples × ~380K CpGs × ~80 bytes = ~1.7 GB) and reuses
the score_atlas_tile() function from VAL-126.

Resume-safe: appends to val128_per_sample.ndjson; checks GSM ids on resume.
"""
import csv
import json
import math
import sys
import time
from pathlib import Path
import importlib.util

import numpy as np

spec = importlib.util.spec_from_file_location('val126', '/home/claude/gastric_esophageal_sprint/VAL-126_stad_phase_c/val126_stad_phase_c.py')
val126 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(val126)

OUTPUT = Path('/home/claude/gastric_esophageal_sprint/VAL-128_crohns_blood')
SERIES = OUTPUT / 'GSE87650_GPL13534_series_matrix.txt'
SAMPLE_TABLE = OUTPUT / 'gse87650_sample_table.json'
PROGRESS = OUTPUT / 'val128_per_sample.ndjson'

H_MIN = val126.H_MIN
N_ACTIVE = 240
CHUNK_SIZE = 60


def load_done():
    if not PROGRESS.exists():
        return set()
    done = set()
    with open(PROGRESS) as f:
        for line in f:
            try:
                row = json.loads(line)
                if 'gsm' in row:
                    done.add(row['gsm'])
            except json.JSONDecodeError:
                continue
    return done


def score_chunk(chunk_start, chunk_end, atlases_def, xu538, atlas_lookup):
    """Stream the matrix and score samples [chunk_start:chunk_end].
    Returns list of per-sample row dicts."""
    chunk_size = chunk_end - chunk_start
    print(f'\n=== Chunk samples [{chunk_start}:{chunk_end}] (n={chunk_size}) ===')
    
    # Per-sample sparse β dicts (only for needed CpGs)
    sample_betas = [{} for _ in range(chunk_size)]
    n_extreme = [0] * chunk_size
    n_middle = [0] * chunk_size
    n_valid = [0] * chunk_size
    
    needed_cpgs = set(atlas_lookup.keys()) | xu538
    
    with open(SERIES) as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):
                break
        header = f.readline().strip().split('\t')
        gsms = [g.strip('"') for g in header[1:]]
        chunk_gsms = gsms[chunk_start:chunk_end]
        
        # Read body
        n_rows = 0
        t0 = time.time()
        for line in f:
            line = line.rstrip()
            if not line or line.startswith('!series_matrix_table_end'):
                break
            n_rows += 1
            tab_idx = line.find('\t')
            if tab_idx < 0:
                continue
            cpg = line[:tab_idx].strip('"')
            
            # Quick dispatch: only process needed CpGs in detail
            need_full = cpg in needed_cpgs
            
            # Parse only the chunk's columns
            rest = line[tab_idx+1:]
            parts = rest.split('\t')
            
            for i in range(chunk_size):
                col = chunk_start + i
                if col >= len(parts):
                    break
                v = parts[col].strip('"')
                if v in ('NA', 'NaN', '', 'null'):
                    continue
                try:
                    b = float(v)
                except ValueError:
                    continue
                if not (0 <= b <= 1):
                    continue
                n_valid[i] += 1
                if b < 0.1 or b > 0.9:
                    n_extreme[i] += 1
                if 0.4 <= b <= 0.6:
                    n_middle[i] += 1
                if need_full:
                    sample_betas[i][cpg] = b
            
            if n_rows % 100000 == 0:
                elapsed = time.time() - t0
                rate = n_rows / elapsed
                eta = (482000 - n_rows) / rate if rate > 0 else 0
                avg_dict = sum(len(d) for d in sample_betas) / chunk_size
                print(f'  [{n_rows:,} rows, {elapsed:.0f}s, ~{avg_dict:.0f} CpGs/sample, eta={eta:.0f}s]')
    
    print(f'  Stream complete: {n_rows:,} rows in {time.time()-t0:.0f}s')
    print(f'  Avg CpGs/sample: {sum(len(d) for d in sample_betas)/chunk_size:.0f}')
    
    # Score each sample
    print(f'  Scoring {chunk_size} samples...')
    
    with open(SAMPLE_TABLE) as f:
        sample_meta = json.load(f)
    gsm_to_meta = {r['gsm']: r for r in sample_meta}
    
    rows_out = []
    for i, gsm in enumerate(chunk_gsms):
        sb = sample_betas[i]
        meta = gsm_to_meta.get(gsm, {})
        f_ex = n_extreme[i] / n_valid[i] if n_valid[i] > 0 else 0
        f_md = n_middle[i] / n_valid[i] if n_valid[i] > 0 else 0
        
        row = {
            'gsm': gsm,
            'file_id': gsm,
            'submitter_id': meta.get('title', gsm),
            'sample_type': meta.get('cell_type', 'unknown'),
            'cell_type': meta.get('cell_type'),
            'simplified_diagnosis': meta.get('simplified_diagnosis'),
            'full_diagnosis': meta.get('full_diagnosis'),
            'sex': meta.get('sex'),
            'age_at_sample': meta.get('age_at_sample'),
            'patient_number': meta.get('patient_number'),
            'n_valid_betas': n_valid[i],
            'f_extreme': f_ex,
            'f_middle': f_md,
        }
        
        # Stage 1 Xu-538 pooled-entropy
        xu_present = sum(1 for c in xu538 if c in sb)
        row['n_cpgs_xu538'] = xu_present
        row['coverage_xu538'] = xu_present / len(xu538)
        h_vals = []
        for c in xu538:
            if c in sb:
                b = sb[c]
                if 0 < b < 1:
                    h = -b * math.log2(b) - (1 - b) * math.log2(1 - b)
                    h_vals.append(h)
        row['A_xu538_stage1'] = float(np.mean(h_vals)) / H_MIN['immune'] if h_vals else None
        
        # Score atlases
        for atlas_name, atlas_data, tiles, classes in atlases_def:
            present = sum(1 for c in atlas_data if c in sb)
            row[f'coverage_{atlas_name}'] = present / len(atlas_data)
            for tile in tiles:
                cls = classes.get(tile, 'secretory')
                A, n = val126.score_atlas_tile(sb, atlas_data, tile, H_MIN[cls])
                row[f'A_{atlas_name}_{tile}'] = A
                row[f'n_{atlas_name}_{tile}'] = n
        
        rows_out.append(row)
    
    # Free memory before next chunk
    del sample_betas
    return rows_out


def main():
    # Load atlases ONCE
    print('Loading atlases...')
    xu538 = val126.load_xu538(val126.XU538_PATH)
    print(f'  Xu-538: {len(xu538)} CpGs')
    loyfer, loyfer_tiles = val126.load_loyfer(val126.LOYFER_PATH)
    bocc, _ = val126.load_loyfer(val126.BOCC_PATH)
    esoref, _ = val126.load_loyfer(val126.ESOREF_PATH)
    oeref, _ = val126.load_loyfer(val126.OEREF_PATH)
    caggiano, caggiano_tiles = val126.load_loyfer(val126.CAGGIANO_PATH)
    salas, _ = val126.load_loyfer(val126.SALAS_PATH)
    unilife, unilife_tiles = val126.load_loyfer(val126.UNILIFE_PATH)
    print(f'  All Stage 2/3 atlases loaded.')
    
    atlases_def = [
        ('loyfer', loyfer, loyfer_tiles, val126.LOYFER_CLASSES),
        ('bocc', bocc, val126.BOCC_TILES, val126.BOCC_CLASSES),
        ('esoref', esoref, val126.ESOREF_TILES, val126.ESOREF_CLASSES),
        ('oeref', oeref, val126.OEREF_TILES, val126.OEREF_CLASSES),
        ('cag', caggiano, caggiano_tiles, val126.CAGGIANO_CLASSES),
        ('salas', salas, val126.SALAS_TILES, val126.SALAS_CLASSES),
        ('uni', unilife, unilife_tiles, {t: 'immune' for t in unilife_tiles}),
    ]
    
    # Build CpG → atlas refs lookup (for needed_cpgs)
    atlas_lookup = {}
    for atlas_name, atlas_data, _, _ in atlases_def:
        for cpg in atlas_data:
            atlas_lookup[cpg] = True
    print(f'  CpG universe (atlases ∪ Xu-538): {len(atlas_lookup) + len(xu538):,}')
    
    # Resume support
    done = load_done()
    print(f'\nAlready scored: {len(done)}/{N_ACTIVE}')
    
    # Run chunks
    out_f = open(PROGRESS, 'a')
    for chunk_start in range(0, N_ACTIVE, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, N_ACTIVE)
        # Check if chunk is complete
        chunk_gsms_check = None
        with open(SERIES) as f:
            for line in f:
                if line.startswith('!series_matrix_table_begin'):
                    break
            header = f.readline().strip().split('\t')
            gsms = [g.strip('"') for g in header[1:]]
            chunk_gsms_check = gsms[chunk_start:chunk_end]
        if all(g in done for g in chunk_gsms_check):
            print(f'\nChunk [{chunk_start}:{chunk_end}] complete, skipping')
            continue
        
        rows = score_chunk(chunk_start, chunk_end, atlases_def, xu538, atlas_lookup)
        for row in rows:
            if row['gsm'] not in done:
                out_f.write(json.dumps(row) + '\n')
                out_f.flush()
                done.add(row['gsm'])
        print(f'  Cumulative scored: {len(done)}/{N_ACTIVE}')
    
    out_f.close()
    print(f'\n*** VAL-128 CHUNKED SCORING COMPLETE *** {len(done)}/{N_ACTIVE}')


if __name__ == '__main__':
    main()
