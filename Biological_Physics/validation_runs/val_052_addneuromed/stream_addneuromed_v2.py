#!/usr/bin/env python3
"""
VAL-052 Stream v2 — Memory-efficient streaming decompression + line-by-line parse.

Key change from v1: gzip.GzipFile wraps urlopen stream directly;
we read line-by-line and discard as we go. Never materialize full text.
"""
import urllib.request, gzip, json, hashlib, time, sys, io

URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE144nnn/GSE144858/matrix/GSE144858_series_matrix.txt.gz'

PANEL_18 = [
    'cg00431549','cg01127300','cg02228185','cg02489552','cg04023335',
    'cg09809672','cg10632894','cg12554573','cg14614643','cg16867657',
    'cg17861230','cg18834029','cg22454769','cg22736354','cg23244761',
    'cg25432518','cg25809905','cg26614073',
]
PANEL_SET = set(PANEL_18)

# Prepend chars to escape initial quoting
def clean(s): return s.strip().strip('"').strip()

print(f"[{time.strftime('%H:%M:%S')}] Stream starting: {URL}", flush=True)

req = urllib.request.Request(URL, headers={'User-Agent': 'walther-mayer/2.0'})
resp = urllib.request.urlopen(req, timeout=180)

# Hash while streaming
hasher = hashlib.sha256()

# Wrap the response in a hashing stream
class HashReader:
    def __init__(self, stream, hasher):
        self.stream = stream
        self.hasher = hasher
        self.bytes_read = 0
    def read(self, n=-1):
        data = self.stream.read(n) if n > 0 else self.stream.read()
        self.hasher.update(data)
        self.bytes_read += len(data)
        if self.bytes_read > 0 and (self.bytes_read // (50 * 1024 * 1024)) != ((self.bytes_read - len(data)) // (50 * 1024 * 1024)):
            print(f"  [{time.strftime('%H:%M:%S')}] streamed {self.bytes_read/1e6:.0f} MB", flush=True)
        return data

hashing = HashReader(resp, hasher)
gz = gzip.GzipFile(fileobj=hashing)
text_stream = io.TextIOWrapper(gz, encoding='utf-8', errors='replace')

# State
sample_ids = []
meta_lines = {}  # field → [rows]
in_matrix = False
header_cols = None
beta_data = None  # dict gsm → dict cpg → β
cpg_found = 0
n_lines = 0

for raw_line in text_stream:
    n_lines += 1
    line = raw_line.rstrip('\n')
    if n_lines % 50000 == 0:
        print(f"  [{time.strftime('%H:%M:%S')}] processed {n_lines:,} lines, CpGs recovered: {cpg_found}", flush=True)

    if not in_matrix:
        if line.startswith('!Sample_geo_accession'):
            parts = line.split('\t')
            sample_ids = [clean(p) for p in parts[1:]]
            print(f"  samples found: {len(sample_ids)}", flush=True)
            beta_data = {g: {} for g in sample_ids}
        elif line.startswith('!Sample_characteristics') or line.startswith('!Sample_title') or line.startswith('!Sample_source'):
            parts = line.split('\t')
            key = parts[0].replace('!','')
            values = [clean(p) for p in parts[1:]]
            meta_lines.setdefault(key, []).append(values)
        elif line.startswith('!series_matrix_table_begin'):
            in_matrix = True
        continue

    # In matrix
    if line.startswith('!series_matrix_table_end'):
        break
    parts = line.split('\t')
    if header_cols is None:
        header_cols = [clean(p) for p in parts]
        # Verify alignment
        n_data_cols = len(header_cols) - 1
        print(f"  matrix header has {len(header_cols)} cols, {n_data_cols} sample slots", flush=True)
        if n_data_cols != len(sample_ids):
            print(f"  WARN: sample_ids {len(sample_ids)} vs data cols {n_data_cols} — using min", flush=True)
        continue

    if len(parts) < 2: continue
    probe_id = clean(parts[0])
    if probe_id in PANEL_SET:
        for j, val in enumerate(parts[1:]):
            if j >= len(sample_ids): break
            v = val.strip().strip('"')
            if v in ('NA','null','','NaN','nan'): continue
            try:
                fv = float(v)
                if 0.0 < fv < 1.0:
                    beta_data[sample_ids[j]][probe_id] = fv
            except ValueError:
                pass
        cpg_found += 1
        print(f"  [{cpg_found}/18] {probe_id}", flush=True)
        if cpg_found >= 18:
            print(f"  all 18 panel CpGs recovered, stopping matrix parse", flush=True)
            break

sha = hasher.hexdigest()
print(f"\n[{time.strftime('%H:%M:%S')}] Download complete")
print(f"  Bytes streamed: {hashing.bytes_read:,}")
print(f"  SHA-256: {sha}")
print(f"  Lines processed: {n_lines:,}")

# Build manifest
manifest = []
for i, gsm in enumerate(sample_ids):
    rec = {'gsm': gsm}
    for key in meta_lines:
        for row in meta_lines[key]:
            if i < len(row):
                val = row[i]
                # If "key: value" format, split
                if ':' in val and len(val) < 500:
                    k2, v2 = val.split(':', 1)
                    rec[k2.strip()] = v2.strip()
                else:
                    rec[key] = val
    manifest.append(rec)

print(f"\nManifest built: {len(manifest)} samples")
if manifest:
    sample_keys = set()
    for r in manifest: sample_keys.update(r.keys())
    print(f"Keys: {sorted(sample_keys)}")
    print(f"First 2: {manifest[:2]}")

with open('addneuromed_manifest.json','w') as f:
    json.dump(manifest, f, indent=2)
with open('addneuromed_imm_betas.json','w') as f:
    json.dump(beta_data, f, indent=2)

# Stats
valid = sum(1 for g in sample_ids if len(beta_data[g]) >= 12)
print(f"\nSamples with ≥12/18 valid β: {valid}/{len(sample_ids)}")
print(f"CpGs recovered: {cpg_found}/18")
print(f"DONE at {time.strftime('%H:%M:%S')}")
