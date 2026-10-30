"""The in-mix readings on sequencing against the IAM Sequencing Atlas (Met-A for the whole sample, the territories): synthetic, written and
committed before it runs. The instrument's job: a healthy sample reads Normal (1).

Known answer: 6 constructed healthy sequenced samples (2 pure cells, 2 of 70/30, 2 of 50/30/20; cells and reads drawn with seed 20261101),
each site of IAM-Atlas v2.1 given reads k of n from the mixed sequencing-atlas level (n 20, or 8 for half of them, to cover a shallow
library), written as .lbeta in the hg19 numbering and read by the one command with the full chain (run_sample.py --dev-seq-beta FILE
--dev-seq-values 1 --dev-seq-ref auto --full-chain --specimen unknown). Bars (operational, set before the run): Met-A for the whole sample,
islands and domains each within 0.05 of 1 in every sample; the bundle names the IAM Sequencing Atlas for the in-mix levels.
Usage: python3 inmix_seq_synthetic.py WORKDIR OUT.txt"""
import sys, os, json, subprocess, numpy as np, pandas as pd, pyarrow.parquet as pq
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"); sys.path.insert(0, C)
import stage_seq_atlas as Q, stage_seq_values as SV, stage_metaa_atlas as SA
ids = pq.read_table(os.path.join(SA.ATLAS, "sites.parquet")).column("cpg").to_pylist()
K = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
cpg = pd.Series(ids).map(K); have = cpg.notna().to_numpy(); cpg = cpg[have].astype(np.int64).to_numpy()
def col(c):
    for k in (1, 2, 3):
        p = os.path.join(Q.ARRAY_COLS, f"mu_part{k}.parquet")
        if c in pq.read_schema(p).names:
            u = pq.read_table(p, columns=[c]).column(0).to_numpy(); v = u / 1e4; v[u == 65535] = np.nan; return v[have]
cells = [c for k in (1, 2, 3) for c in pq.read_schema(os.path.join(Q.ARRAY_COLS, f"mu_part{k}.parquet")).names]
rng = np.random.default_rng(20261101); pick = list(rng.choice(cells, 12, replace=False))
mixes = [({pick[0]: 1.0}, 20), ({pick[1]: 1.0}, 8), ({pick[2]: 0.7, pick[3]: 0.3}, 20), ({pick[4]: 0.7, pick[5]: 0.3}, 8),
         ({pick[6]: 0.5, pick[7]: 0.3, pick[8]: 0.2}, 20), ({pick[9]: 0.5, pick[10]: 0.3, pick[11]: 0.2}, 8)]
L = ["In-mix readings on sequencing against the IAM Sequencing Atlas, synthetic: 6 constructed healthy samples",
     "sample | reads per site | planted | Met-A whole | islands | domains | shares read | in-mix levels | bars"]; ok = []
for j, (mx, N) in enumerate(mixes):
    V = [col(c) for c in mx]; m = np.all([~np.isnan(v) for v in V], 0); p = sum(f * np.nan_to_num(v) for f, v in zip(mx.values(), V))
    a = np.zeros((SV.HG19_CPGS, 2), np.uint16); ii = cpg[m]; kk = rng.binomial(N, np.clip(p[m], 0, 1))
    a[ii - 1, 0] = kk; a[ii - 1, 1] = N; f = os.path.join(W, f"s{j}.lbeta"); a.tofile(f); b = os.path.join(W, f"s{j}_bundle.json")
    subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--dev-seq-ref", "auto", "--full-chain", "--specimen", "unknown",
                    "--id", f"s{j}", "--out", os.path.join(W, f"s{j}.html"), "--bundle", b], cwd=os.path.join(C, "MethylPhys_Interface"), capture_output=True, text=True)
    B = json.load(open(b)); w = (B.get("met_a_whole") or {}).get("A"); t = (B.get("territories") or {}).get("sets") or {}
    il, do = (t.get("islands") or {}).get("A"), (t.get("domains") or {}).get("A"); sh = {x["cell"]: x["share"] for x in (B.get("cell_shares") or {}).get("cells") or [] if x["share"] >= 0.05}
    lv = (B.get("sequencing_ruler") or {}).get("in_mix_levels")
    g = all(v is not None and abs(v - 1) <= 0.05 for v in (w, il, do)) and lv == "IAM Sequencing Atlas"; ok.append(g)
    L.append(f"{j} | {N} | " + ", ".join(f"{c} {v:.1f}" for c, v in mx.items()) + f" | {w} | {il} | {do} | " + ", ".join(f"{c} {v:.2f}" for c, v in sh.items()) + f" | {lv} | {'met' if g else 'not met'}")
    os.remove(f)
L.append(f"synthetic: {'met' if all(ok) else 'not met'}"); open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
