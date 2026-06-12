"""
Stream and extract breast pre-dx cohorts GSE51057 + GSE51032 (EPIC-Italy).
Output: per-cohort betas_union + clinical_metadata + raw_geo_metadata.
"""
import urllib.request, gzip, io, json, time, sys, csv, hashlib
from pathlib import Path

UNION_PATH = "/home/claude/ad_work/cpg_union_for_breast_extraction.txt"
TARGET_CPGS = set()
with open(UNION_PATH) as f:
    for line in f:
        cpg = line.strip()
        if cpg: TARGET_CPGS.add(cpg)
print(f"[{time.strftime('%H:%M:%S')}] Target CpG union: {len(TARGET_CPGS)} CpGs", flush=True)


def clean(s):
    return s.strip().strip('"').strip()


def stream_series_matrix(url, out_csv, out_meta_json, out_raw_meta_json, cohort_name, gse_id):
    print(f"\n[{time.strftime('%H:%M:%S')}] Streaming {cohort_name}: {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "walther-mayer/2.0"})
    resp = urllib.request.urlopen(req, timeout=600)
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
    
    samples = []
    raw_meta = {}
    in_table = False
    n_target_lines = 0
    n_total_lines = 0
    raw_geo_metadata = {}
    
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_csv, "w", newline="")
    out_writer = csv.writer(out_f)
    
    t0 = time.time()
    for line in text:
        line = line.rstrip("\n").rstrip("\r")
        n_total_lines += 1
        
        if line.startswith("!Sample_geo_accession"):
            samples = [clean(x) for x in line.split("\t")[1:]]
            print(f"  Samples found: {len(samples)}", flush=True)
            raw_meta = {s: {} for s in samples}
            # Header row
            out_writer.writerow(["cpg_id"] + samples)
        elif line.startswith("!Sample_characteristics") or line.startswith("!Sample_title") or line.startswith("!Sample_source_name"):
            parts = line.split("\t")
            key = clean(parts[0]).lstrip("!")
            vals = [clean(x) for x in parts[1:]]
            for i, s in enumerate(samples):
                if i < len(vals):
                    if key not in raw_meta[s]:
                        raw_meta[s][key] = []
                    raw_meta[s][key].append(vals[i])
        elif line.startswith("!series_matrix_table_begin"):
            in_table = True
            continue
        elif line.startswith("!series_matrix_table_end"):
            in_table = False
            break
        elif in_table and line:
            parts = line.split("\t")
            cpg = clean(parts[0])
            if cpg.startswith("cg") and cpg in TARGET_CPGS:
                out_writer.writerow([cpg] + parts[1:])
                n_target_lines += 1
                if n_target_lines % 2000 == 0:
                    print(f"  [{time.strftime('%H:%M:%S')}] {n_target_lines} target CpGs extracted | {n_total_lines} total lines", flush=True)
    
    out_f.close()
    el = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] {cohort_name} done in {el:.0f}s. Found {n_target_lines}/{len(TARGET_CPGS)} target CpGs ({n_target_lines/len(TARGET_CPGS)*100:.1f}%)")
    
    # Parse raw_meta into structured clinical metadata
    # For Severi cohort: look for cancer site, time to diagnosis, sex, age
    clinical = []
    for gsm in samples:
        meta = raw_meta.get(gsm, {})
        rec = {"gsm": gsm}
        # Parse characteristics
        chars_list = meta.get("Sample_characteristics_ch1", [])
        for c in chars_list:
            c_lower = c.lower()
            if c_lower.startswith("site:") or c_lower.startswith("cancer site:") or c_lower.startswith("cancer_site:"):
                rec["cancer_site"] = c.split(":", 1)[1].strip()
            elif c_lower.startswith("group:"):
                rec["group"] = c.split(":", 1)[1].strip()
            elif c_lower.startswith("ttd"):
                try: rec["ttd_years"] = float(c.split(":", 1)[1].strip().split()[0])
                except (ValueError, IndexError): pass
            elif c_lower.startswith("age:"):
                try: rec["age"] = float(c.split(":", 1)[1].strip())
                except (ValueError, IndexError): pass
            elif c_lower.startswith("gender:") or c_lower.startswith("sex:"):
                rec["gender"] = c.split(":", 1)[1].strip()
            elif c_lower.startswith("ethnicity:"):
                rec["ethnicity"] = c.split(":", 1)[1].strip()
        
        # Determine arm: case = cancer_site C50 + ttd > 10y; hc = group == control
        cancer_site = rec.get("cancer_site", "").upper()
        group = rec.get("group", "").lower()
        ttd = rec.get("ttd_years", None)
        if cancer_site == "C50" and ttd is not None and ttd > 10:
            rec["arm"] = "case"
        elif "control" in group or group == "":
            rec["arm"] = "hc"
        else:
            rec["arm"] = "other_filtered_out"
        
        clinical.append(rec)
    
    with open(out_meta_json, "w") as f:
        json.dump(clinical, f, indent=2)
    with open(out_raw_meta_json, "w") as f:
        json.dump(raw_meta, f, indent=2)
    
    sha = hasher.hexdigest()
    print(f"  SHA-256 of gz stream: {sha}")
    return n_target_lines, sha


# Run both cohorts
results = {}
for gse, cohort_name in [("GSE51057", "EPIC-Italy GSE51057"), ("GSE51032", "EPIC-Italy GSE51032")]:
    gse_dir = gse[:-3] + "nnn"  # e.g. GSE51nnn
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_dir}/{gse}/matrix/{gse}_series_matrix.txt.gz"
    out_dir = Path(f"/home/claude/ad_work/{gse}_breast")
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cpgs, sha = stream_series_matrix(url,
        f"{out_dir}/{gse}_betas_union.csv",
        f"{out_dir}/{gse}_clinical_metadata.json",
        f"{out_dir}/{gse}_raw_geo_metadata.json",
        cohort_name, gse)
    results[gse] = {"n_cpgs": n_cpgs, "sha": sha}

print(f"\n=== Summary ===")
for gse, r in results.items():
    print(f"  {gse}: {r['n_cpgs']}/14018 CpGs (sha: {r['sha'][:32]}...)")
