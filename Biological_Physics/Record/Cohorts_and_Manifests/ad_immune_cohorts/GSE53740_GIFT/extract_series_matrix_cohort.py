"""
Stream and extract series_matrix format cohorts (AddNeuroMed GSE144858, GSE53740 GIFT).
Adapts stream_addneuromed_v2.py pattern to extract the 14,018-CpG post-build union
(not just the 18-CpG IMM_CPGS_EPIC_18 panel the pre-build VAL-052 used).

Output: per-cohort
  - GSExxxxx_betas_union.csv (CpGs as rows, GSMs as columns)
  - GSExxxxx_clinical_metadata.json (per-sample arm + sex)
  - GSExxxxx_cohort_manifest.json (provenance + SHA + reproduction-check anchors)
"""
import urllib.request, gzip, io, json, time, sys, csv, hashlib, re
from pathlib import Path

# Load CpG union (built in prior session for AIBL run)
UNION_PATH = "/home/claude/ad_work/cpg_union_for_ad_extraction.txt"
TARGET_CPGS = set()
with open(UNION_PATH) as f:
    for line in f:
        cpg = line.strip()
        if cpg: TARGET_CPGS.add(cpg)
print(f"[{time.strftime('%H:%M:%S')}] Target CpG union: {len(TARGET_CPGS)} CpGs", flush=True)


def clean(s):
    return s.strip().strip('"').strip()


def stream_series_matrix(url, out_csv, out_meta_json, cohort_name):
    """
    Stream a GEO series_matrix.txt.gz, extracting union CpGs and per-GSM metadata.
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Streaming {cohort_name}: {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "walther-mayer/2.0"})
    resp = urllib.request.urlopen(req, timeout=300)
    hasher = hashlib.sha256()
    
    class HashReader:
        def __init__(self, stream):
            self.stream = stream
            self.bytes_read = 0
        def read(self, n=-1):
            data = self.stream.read(n) if n > 0 else self.stream.read()
            hasher.update(data)
            self.bytes_read += len(data)
            return data
    
    hashing = HashReader(resp)
    gz = gzip.GzipFile(fileobj=hashing)
    text = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
    
    sample_ids = []        # GSM IDs in column order
    meta = {}              # field_name → [values per sample]
    in_matrix = False
    header_cols = None
    beta_data = {}         # gsm → {cpg: float}
    cpg_found = 0
    n_lines = 0
    
    for raw in text:
        n_lines += 1
        if n_lines % 100000 == 0:
            print(f"  [{time.strftime('%H:%M:%S')}] line {n_lines:,}, MB streamed {hashing.bytes_read/1e6:.0f}, CpGs found {cpg_found}", flush=True)
        line = raw.rstrip("\n")
        
        if not in_matrix:
            if line.startswith("!Sample_geo_accession"):
                parts = line.split("\t")
                sample_ids = [clean(p) for p in parts[1:]]
                print(f"  Samples: {len(sample_ids)}", flush=True)
                beta_data = {g: {} for g in sample_ids}
            elif line.startswith("!Sample_characteristics") or line.startswith("!Sample_title") \
                 or line.startswith("!Sample_source") or line.startswith("!Sample_description"):
                parts = line.split("\t")
                key = parts[0].replace("!", "")
                vals = [clean(p) for p in parts[1:]]
                meta.setdefault(key, []).append(vals)
            elif line.startswith("!series_matrix_table_begin"):
                in_matrix = True
            continue
        
        # In matrix
        if line.startswith("!series_matrix_table_end"):
            break
        parts = line.split("\t")
        if header_cols is None:
            header_cols = [clean(p) for p in parts]
            n_data = len(header_cols) - 1
            print(f"  Matrix header: {len(header_cols)} cols, {n_data} sample slots", flush=True)
            if n_data != len(sample_ids):
                print(f"  WARN: sample_ids={len(sample_ids)} vs data cols={n_data}", flush=True)
            continue
        if len(parts) < 2:
            continue
        probe_id = clean(parts[0])
        if probe_id in TARGET_CPGS:
            for j, val in enumerate(parts[1:]):
                if j >= len(sample_ids): break
                v = val.strip().strip('"')
                if v in ("NA", "null", "", "NaN", "nan"): continue
                try:
                    fv = float(v)
                    if 0.0 <= fv <= 1.0:
                        beta_data[sample_ids[j]][probe_id] = fv
                except ValueError:
                    pass
            cpg_found += 1
            if cpg_found % 500 == 0:
                print(f"  [{cpg_found}] CpGs found so far", flush=True)
            if cpg_found >= len(TARGET_CPGS):
                print(f"  All target CpGs recovered, stopping matrix parse", flush=True)
                break
    
    matrix_sha = hasher.hexdigest()
    print(f"  [{time.strftime('%H:%M:%S')}] Stream done. Matrix SHA-256: {matrix_sha[:32]}...")
    print(f"  CpGs found: {cpg_found}/{len(TARGET_CPGS)}")
    
    # Build per-sample metadata: parse characteristics fields like "key: value"
    manifest = []
    for i, gsm in enumerate(sample_ids):
        rec = {"gsm": gsm}
        for key, rows in meta.items():
            for row in rows:
                if i >= len(row): continue
                val = row[i]
                # Parse "k: v" pattern in characteristics
                if ":" in val and len(val) < 500:
                    k2, v2 = val.split(":", 1)
                    rec[k2.strip().lower().replace(" ", "_")] = v2.strip()
                else:
                    rec[key.lower()] = val
        manifest.append(rec)
    
    with open(out_meta_json, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote {out_meta_json} ({len(manifest)} samples)")
    
    # Collect all CpGs that ANY sample had β for
    all_cpgs = set()
    for g, d in beta_data.items():
        all_cpgs.update(d.keys())
    cpgs_present = sorted(all_cpgs)
    
    # Write CSV: CpGs as rows, samples as cols
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cpg_id"] + sample_ids)
        for cpg in cpgs_present:
            row = [cpg]
            for s in sample_ids:
                v = beta_data[s].get(cpg)
                row.append("" if v is None else f"{v:.6f}")
            w.writerow(row)
    
    # SHA of output CSV
    out_sha = hashlib.sha256(Path(out_csv).read_bytes()).hexdigest()
    print(f"  Wrote {out_csv} ({Path(out_csv).stat().st_size/1024/1024:.1f} MB)")
    print(f"  Output SHA-256: {out_sha}")
    
    return {
        "cohort": cohort_name,
        "source_url": url,
        "matrix_sha256_streamed": matrix_sha,
        "out_csv": out_csv,
        "out_csv_sha256": out_sha,
        "out_csv_size_bytes": Path(out_csv).stat().st_size,
        "n_target_cpgs": len(TARGET_CPGS),
        "n_found_cpgs": cpg_found,
        "n_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "streamed_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "bytes_streamed": hashing.bytes_read,
    }


if __name__ == "__main__":
    cohorts = {
        "AddNeuroMed": {
            "gse": "GSE144858",
            "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE144nnn/GSE144858/matrix/GSE144858_series_matrix.txt.gz",
            "platform": "Illumina HumanMethylation450K",
            "n_expected": 300,
            "citation": "AddNeuroMed cohort, Lunnon et al. UK/Finland/Italy/France/Poland/Greece multi-center European cohort.",
        },
        "GIFT": {
            "gse": "GSE53740",
            "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE53nnn/GSE53740/matrix/GSE53740_series_matrix.txt.gz",
            "platform": "Illumina HumanMethylation450K",
            "n_expected": 384,
            "citation": "GIFT cohort, Ferrari et al. 2014 (Hum Mol Genet), UCSF Memory and Aging Center. AD/FTD/PSP/CBD/HC.",
        },
    }
    
    target = sys.argv[1] if len(sys.argv) > 1 else None
    
    for name, c in cohorts.items():
        if target and name != target: continue
        out_dir = Path(f"/home/claude/ad_work/{c['gse']}_{name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        manifest = stream_series_matrix(
            url=c["url"],
            out_csv=str(out_dir / f"{c['gse']}_betas_union.csv"),
            out_meta_json=str(out_dir / f"{c['gse']}_raw_geo_metadata.json"),
            cohort_name=name,
        )
        manifest["gse"] = c["gse"]
        manifest["platform"] = c["platform"]
        manifest["n_expected"] = c["n_expected"]
        manifest["citation"] = c["citation"]
        
        # Coverage stats
        manifest["coverage_pct"] = round(100 * manifest["n_found_cpgs"] / manifest["n_target_cpgs"], 2)
        manifest["n_missing_cpgs"] = manifest["n_target_cpgs"] - manifest["n_found_cpgs"]
        
        with open(out_dir / f"{c['gse']}_cohort_manifest_partial.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n✓ Cohort manifest (partial): {out_dir}/{c['gse']}_cohort_manifest_partial.json\n")
