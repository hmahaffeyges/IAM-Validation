"""
AD cohort streaming extractor for post-build pipeline.

Adapted from stream_aibl.py — extracts the 14,018-CpG union needed for the
full SOP v1.2 chain (Walther class+cell-type markers ∪ v0_2 markers ∪ age
layer ∪ AD Rule A panel ∪ IMM_CPGS_RAW panel), NOT just the 29-CpG panel
the pre-build pipeline used.

Streams from GEO directly. No full-matrix decompression to disk.
Output: CSV with CpGs as rows, samples as columns (Walther input format).
"""

import urllib.request, gzip, io, json, time, sys, csv, hashlib
from pathlib import Path

# Load CpG union
UNION_PATH = "/home/claude/ad_work/cpg_union_for_ad_extraction.txt"
TARGET_CPGS = set()
with open(UNION_PATH) as f:
    for line in f:
        cpg = line.strip()
        if cpg: TARGET_CPGS.add(cpg)
print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(TARGET_CPGS)} target CpGs")
sys.stdout.flush()


def stream_aibl_to_csv(url, out_csv, max_samples=None):
    """
    Stream AIBL series matrix (samples as rows, CpGs as columns).
    Extract only target CpGs, transpose to CpGs-as-rows × samples-as-cols.
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Streaming: {url}")
    sys.stdout.flush()
    
    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req, timeout=300)
    gz = gzip.GzipFile(fileobj=io.BufferedReader(resp, buffer_size=1<<20))
    
    # Header: first row = blank + 862,601 CpG IDs
    header_line = gz.readline().decode("utf-8", errors="replace").rstrip("\n")
    header_cols = header_line.split("\t")
    print(f"[{time.strftime('%H:%M:%S')}] Header: {len(header_cols)} columns")
    sys.stdout.flush()
    
    # Map CpG ID → header column index (only for target CpGs present)
    target_indices = []  # list of (col_index, cpg_id)
    for i, cpg in enumerate(header_cols):
        cpg_clean = cpg.strip().strip('"')
        if cpg_clean in TARGET_CPGS:
            target_indices.append((i, cpg_clean))
    
    found_cpgs = {cpg for _, cpg in target_indices}
    missing = TARGET_CPGS - found_cpgs
    print(f"[{time.strftime('%H:%M:%S')}] Found {len(target_indices)}/{len(TARGET_CPGS)} target CpGs in matrix")
    print(f"  Missing: {len(missing)} (platform coverage gap)")
    sys.stdout.flush()
    
    # Stream samples, extract target columns
    sample_betas = {}  # sample_id → {cpg_id: beta}
    n = 0
    for line in gz:
        line = line.decode("utf-8", errors="replace").rstrip("\n")
        if not line: continue
        parts = line.split("\t")
        sample_id = parts[0].strip().strip('"')
        if not sample_id: continue
        
        betas = {}
        for col_idx, cpg in target_indices:
            if col_idx < len(parts):
                raw = parts[col_idx].strip().strip('"')
                try:
                    betas[cpg] = float(raw)
                except (ValueError, TypeError):
                    betas[cpg] = None  # NA
        sample_betas[sample_id] = betas
        n += 1
        if n % 50 == 0:
            print(f"  [{time.strftime('%H:%M:%S')}] Streamed {n} samples")
            sys.stdout.flush()
        if max_samples and n >= max_samples: break
    
    print(f"[{time.strftime('%H:%M:%S')}] Total streamed: {n} samples")
    sys.stdout.flush()
    
    # Write CSV: CpGs as rows, samples as columns (Walther input format)
    samples = sorted(sample_betas.keys())
    cpgs_present = sorted(found_cpgs)
    
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cpg_id"] + samples)
        for cpg in cpgs_present:
            row = [cpg]
            for s in samples:
                val = sample_betas[s].get(cpg)
                row.append("" if val is None else f"{val:.6f}")
            w.writerow(row)
    
    # SHA-256 of output
    sha = hashlib.sha256()
    with open(out_csv, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    sha_hex = sha.hexdigest()
    
    print(f"[{time.strftime('%H:%M:%S')}] Wrote: {out_csv}")
    print(f"  Rows (CpGs): {len(cpgs_present)}, Cols (samples): {len(samples)}")
    print(f"  Size: {Path(out_csv).stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  SHA-256: {sha_hex}")
    
    return {
        "cohort_url": url,
        "out_csv": out_csv,
        "n_target_cpgs": len(TARGET_CPGS),
        "n_found_cpgs": len(cpgs_present),
        "n_missing_cpgs": len(missing),
        "n_samples": len(samples),
        "out_size_bytes": Path(out_csv).stat().st_size,
        "out_sha256": sha_hex,
        "sample_ids": samples,
    }


if __name__ == "__main__":
    # Cohort URLs from RETIRED evidence report + stream_aibl.py precedent
    cohorts = [
        {
            "gse": "GSE153712",
            "name": "AIBL",
            "platform": "EPIC",
            "n_expected": 726,
            "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE153nnn/GSE153712/suppl/GSE153712_normalized_average_betas.txt.gz",
        },
    ]
    
    target_cohort = sys.argv[1] if len(sys.argv) > 1 else "AIBL"
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    for c in cohorts:
        if c["name"] != target_cohort: continue
        out_csv = f"/home/claude/ad_work/{c['gse']}_betas_union.csv"
        manifest = stream_aibl_to_csv(c["url"], out_csv, max_samples=max_samples)
        manifest["gse"] = c["gse"]
        manifest["cohort_name"] = c["name"]
        manifest["platform"] = c["platform"]
        manifest["n_expected"] = c["n_expected"]
        manifest["streamed_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        with open(f"/home/claude/ad_work/{c['gse']}_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n✓ Manifest: /home/claude/ad_work/{c['gse']}_manifest.json")
