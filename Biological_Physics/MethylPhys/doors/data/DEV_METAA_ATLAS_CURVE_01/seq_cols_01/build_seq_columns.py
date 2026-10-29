"""DEV-METAA-ATLAS-CURVE-01: IAM-Atlas v2.1's sequencing columns. Each of the atlas's healthy sequencing inputs (DEV_ALL_INFO_01/coupling_02/
healthy_seq.csv) has its value and calls at every atlas site written by the one command (geo_values.yml: run_sample.py --dev-seq-values 1
--dev-write-values --dev-values-only). For each cell: the mean of its inputs' values at each site (sites with a value in at least one input)
and how many inputs gave it, in the atlas's own storage (level x 10,000 as uint16, 65535 none; counts as uint8; sites in sites.parquet order).
Nothing is fitted. Usage: python3 build_seq_columns.py VALUES_DIR OUT_DIR"""
import sys, os, json, glob, csv, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.join(H, *[".."] * 4)
S = list(csv.DictReader(open(sys.argv[3] if len(sys.argv) > 3 else os.path.join(MP, "doors", "data", "DEV_ALL_INFO_01", "coupling_02", "healthy_seq.csv"))))   # a third argument: another list (synthetic tests)
ids = pq.read_table(os.path.join(MP, "atlas", "v2.1", "sites.parquet")).column("cpg").to_pylist(); N = len(ids); pos = pd.Series(np.arange(N), index=ids)
acc = {}; inputs = {}; missing = []
for r in S:
    f = glob.glob(os.path.join(sys.argv[1], "**", r["gsm"] + "_values.parquet"), recursive=True)
    if not f: missing.append(r["gsm"]); continue
    v = pd.read_parquet(f[0])["beta"].astype("float64"); i = pos.reindex(v.index).dropna().astype(int)
    sm, n = acc.setdefault(r["cell"], (np.zeros(N), np.zeros(N, np.int64))); sm[i.to_numpy()] += v[i.index].to_numpy(); n[i.to_numpy()] += 1
    inputs.setdefault(r["cell"], []).append(r["gsm"])
os.makedirs(sys.argv[2], exist_ok=True); cells = sorted(acc); per = 14
for k in range(0, len(cells), per):
    mu = {}; nn = {}
    for c in cells[k:k + per]:
        sm, n = acc[c]; m = np.where(n > 0, sm / np.maximum(n, 1), np.nan)
        mu[c] = np.where(np.isnan(m), 65535, np.round(m * 1e4)).astype(np.uint16); nn[c] = np.minimum(n, 255).astype(np.uint8)
    pq.write_table(pa.table(mu), os.path.join(sys.argv[2], f"seq_mu_part{k // per + 1}.parquet"), compression="zstd")
    pq.write_table(pa.table(nn), os.path.join(sys.argv[2], f"seq_n_part{k // per + 1}.parquet"), compression="zstd")
json.dump(inputs, open(os.path.join(sys.argv[2], "inputs.json"), "w"), indent=1)
L = [f"cells {len(cells)}, inputs {sum(len(v) for v in inputs.values())} of {len(S)}, missing {missing}"]
L += [f"{c}: {len(inputs[c])} inputs, sites with a value {int((acc[c][1] > 0).sum())}" for c in cells]
open(os.path.join(sys.argv[2], "build_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L[:3]))
