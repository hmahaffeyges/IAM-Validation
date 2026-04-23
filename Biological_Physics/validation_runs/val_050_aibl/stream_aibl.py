"""
Stream AIBL GSE153712 beta matrix and extract IMM_CPGS_RAW panel values per sample.

Matrix format: row 0 = header with ["" + 862,601 CpG IDs], rows 1..726 = samples.
Column 0 = sentrix position, columns 1.. = beta values.

We want: per-sample {cpg_id: beta} for the 29 IMM CpGs only.
"""
import urllib.request, gzip, io, json, time, sys

IMM_CPGS = set([
    'cg04023335','cg05045481','cg10632894','cg26614073','cg08706463',
    'cg19432188','cg17774019','cg18834029','cg00342758','cg10241600',
    'cg23244761','cg14614643','cg23555344','cg01127300','cg14620944',
    'cg20795519','cg24079702','cg07571933','cg25432518','cg00431549',
    'cg16867657','cg22736354','cg02228185','cg25809905','cg09809672',
    'cg02489552','cg12554573','cg17861230','cg22454769',
])

URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE153nnn/GSE153712/suppl/GSE153712_normalized_average_betas.txt.gz"

print(f"[{time.strftime('%H:%M:%S')}] Streaming AIBL beta matrix...")
print(f"  Target: 29 immune CpGs × 726 samples")
print(f"  Panel: GAPE_Evidence_Report_CURRENT.html line 11785, IMM_CPGS_RAW")
sys.stdout.flush()

req = urllib.request.Request(URL)
resp = urllib.request.urlopen(req, timeout=120)
gz = gzip.GzipFile(fileobj=io.BufferedReader(resp, buffer_size=1<<20))

# Read header → find target column indices
header_line = gz.readline().decode("utf-8", errors="replace").rstrip("\n")
header_cols = header_line.split("\t")
print(f"[{time.strftime('%H:%M:%S')}] Header: {len(header_cols)} columns")

# First col is blank. CpG ids start at index 1.
target_indices = []  # list of (header_col_index, cpg_id)
for i, cpg in enumerate(header_cols):
    if cpg in IMM_CPGS:
        target_indices.append((i, cpg))

print(f"  IMM CpGs found in matrix header: {len(target_indices)}/29")
sys.stdout.flush()
if not target_indices:
    print("ERROR — no target CpGs found, check column structure")
    sys.exit(1)

# Stream data rows, grab only target columns
sample_betas = {}  # sentrix -> {cpg: beta}
n_rows = 0
t_start = time.time()
for raw in gz:
    line = raw.decode("utf-8", errors="replace").rstrip("\n")
    if not line: continue
    cols = line.split("\t")
    sentrix = cols[0]
    row_data = {}
    for idx, cpg in target_indices:
        if idx < len(cols):
            try:
                row_data[cpg] = float(cols[idx])
            except (ValueError, IndexError):
                pass
    sample_betas[sentrix] = row_data
    n_rows += 1
    if n_rows % 100 == 0:
        elapsed = time.time() - t_start
        print(f"  [{time.strftime('%H:%M:%S')}] Processed {n_rows} rows in {elapsed:.1f}s")
        sys.stdout.flush()

gz.close()
resp.close()

print(f"\n[{time.strftime('%H:%M:%S')}] DONE. {n_rows} rows, saving to aibl_imm_betas.json")
with open("aibl_imm_betas.json","w") as fh:
    json.dump(sample_betas, fh)
print(f"  Total samples in output: {len(sample_betas)}")
print(f"  Avg CpGs per sample: {sum(len(v) for v in sample_betas.values()) / max(1,len(sample_betas)):.1f}")
