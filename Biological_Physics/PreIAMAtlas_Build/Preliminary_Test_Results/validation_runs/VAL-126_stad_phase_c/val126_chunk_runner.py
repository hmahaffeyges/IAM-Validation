#!/usr/bin/env python3
"""
VAL-126 chunk runner: download N betas, score them, free them, advance.
Resume-safe via val126_per_sample_progress.ndjson written by the scorer.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path('/home/claude/gastric_esophageal_sprint/VAL-126_stad_phase_c')
BETAS_DIR = OUTPUT_DIR / 'betas'
MANIFEST = OUTPUT_DIR / 'tcga_stad_hm450_manifest_FINAL.json'
PROGRESS = OUTPUT_DIR / 'val126_per_sample_progress.ndjson'
GDC_DATA = 'https://api.gdc.cancer.gov/data/'

CHUNK_SIZE = 50

def load_done():
    """Return set of file_ids already scored."""
    if not PROGRESS.exists():
        return set()
    done = set()
    with open(PROGRESS) as f:
        for line in f:
            try:
                row = json.loads(line)
                fid = row.get('file_id')
                if fid:
                    done.add(fid)
            except json.JSONDecodeError:
                continue
    return done

def download_chunk(records):
    """Download a chunk of β files using GDC bulk endpoint."""
    BETAS_DIR.mkdir(parents=True, exist_ok=True)
    # Filter to records that need download
    needed = [r for r in records if not (BETAS_DIR / r['file_name']).exists()]
    if not needed:
        return
    # Use individual GET requests — GDC bulk POST is per-file uuids list with curl
    print(f'  Downloading {len(needed)} β files...')
    t0 = time.time()
    for i, r in enumerate(needed):
        out_path = BETAS_DIR / r['file_name']
        url = f"{GDC_DATA}{r['file_id']}"
        # silent curl, fail-fast
        result = subprocess.run(
            ['curl', '-sSL', '-o', str(out_path), url],
            capture_output=True, timeout=120
        )
        if result.returncode != 0:
            print(f'    FAIL {r["submitter_id"]}: {result.stderr.decode()[:200]}')
            if out_path.exists():
                out_path.unlink()
            continue
        if (i + 1) % 10 == 0 or (i + 1) == len(needed):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(needed) - i - 1) / rate
            print(f'    [{i+1}/{len(needed)}] downloaded, rate={rate:.2f}/s, eta={eta:.0f}s')

def score_chunk(start_idx, end_idx):
    """Run the scorer on this chunk."""
    print(f'  Scoring chunk [{start_idx}:{end_idx}]...')
    result = subprocess.run(
        ['python3', 'val126_stad_phase_c.py', str(start_idx), str(end_idx)],
        cwd=str(OUTPUT_DIR),
        capture_output=False
    )
    return result.returncode

def free_chunk_betas(records):
    """Free β files for records already scored."""
    done = load_done()
    freed = 0
    bytes_freed = 0
    for r in records:
        if r['file_id'] in done:
            p = BETAS_DIR / r['file_name']
            if p.exists():
                bytes_freed += p.stat().st_size
                p.unlink()
                freed += 1
    if freed > 0:
        print(f'  Freed {freed} β files ({bytes_freed/1e9:.2f} GB)')

def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    # Sort by submitter_id for determinism
    manifest_sorted = sorted(manifest, key=lambda r: r['submitter_id'])
    n_total = len(manifest_sorted)
    print(f'VAL-126 chunked runner: {n_total} files total')

    done = load_done()
    print(f'Already scored: {len(done)}/{n_total}')

    for chunk_start in range(0, n_total, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_total)
        chunk_records = manifest_sorted[chunk_start:chunk_end]
        # Skip if every record in chunk is already done
        all_done_in_chunk = all(r['file_id'] in done for r in chunk_records)
        if all_done_in_chunk:
            print(f'Chunk [{chunk_start}:{chunk_end}] already complete, skipping')
            continue
        print(f'\n=== Chunk [{chunk_start}:{chunk_end}] ===')
        # Disk check
        df = subprocess.run(['df', '-BG', '/home/claude'], capture_output=True, text=True)
        avail_g = int(df.stdout.split('\n')[1].split()[3].rstrip('G'))
        print(f'  Disk free: {avail_g} GB')
        if avail_g < 2:
            print(f'  Disk too low, halting.')
            sys.exit(1)

        download_chunk(chunk_records)
        rc = score_chunk(chunk_start, chunk_end)
        if rc != 0:
            print(f'  Scorer returned non-zero: {rc}')
            sys.exit(rc)
        # refresh done set
        done = load_done()
        free_chunk_betas(chunk_records)
        elapsed_total = sum(p.stat().st_size for p in BETAS_DIR.glob('*'))
        print(f'  Cumulative scored: {len(done)}/{n_total}; β scratch: {elapsed_total/1e9:.2f} GB')

    print(f'\n*** ALL CHUNKS COMPLETE ***')
    print(f'  Final scored: {len(done)}/{n_total}')

if __name__ == '__main__':
    main()
