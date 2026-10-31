"""The sky on sequencing: diagnostic after synthetic run 2 (disclosed; nothing scored). A constructed sample with run 2's sample 3 mix
(cd8 t cells 0.7, cd4 t cells 0.3) at 8 reads per site, reads drawn with seed 20261102, written as .lbeta and read by the one command with the
full chain (run_sample.py --dev-seq-beta FILE --dev-seq-values 1 --dev-seq-ref auto --full-chain --specimen unknown) three ways: with the
Deconvolver's own shares; with the planted shares declared (--dev-shares, development); and, with the planted shares declared, after the first
300 sites of chromosome 8 in genome order moved 0.15 toward full methylation (a regional change, not absorbed by a within-chromosome shuffle).
Usage: python3 sky_seq_diag.py WORKDIR OUT.txt"""
import sys, os, json, subprocess, numpy as np, pandas as pd, pyarrow.parquet as pq
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"); sys.path.insert(0, C)
import stage_seq_atlas as Q, stage_seq_values as SV, stage_metaa_atlas as SA, dev_stages as DV
ids = pq.read_table(os.path.join(SA.ATLAS, "sites.parquet")).column("cpg").to_pylist()
K = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
cpg = pd.Series(ids, index=ids).map(K).dropna().astype(np.int64)
def col(c):
    for k in (1, 2, 3):
        p = os.path.join(Q.ARRAY_COLS, f"mu_part{k}.parquet")
        if c in pq.read_schema(p).names:
            u = pq.read_table(p, columns=[c]).column(0).to_numpy(); v = u / 1e4; v[u == 65535] = np.nan; return pd.Series(v, index=ids)
mx = {"cd8 t cells": 0.7, "cd4 t cells": 0.3}; N = 8
p = sum(f * col(c) for c, f in mx.items()).reindex(cpg.index).dropna().clip(0, 1)
m = DV._mapping(); c8 = cpg.reindex(m[m.chr.astype(str) == "8"].index.intersection(p.index)).sort_values().index[:300]
pr = p.copy(); pr[c8] = pr[c8] + 0.15 * (1 - pr[c8]); rng = np.random.default_rng(20261102)
dec = ",".join(f"{c}:{f}" for c, f in mx.items()); L = ["The sky on sequencing, diagnostic (cd8 0.7, cd4 0.3, 8 reads)", "read | lee p | structure | largest band ratio"]
for j, (name, q, flags) in enumerate((("the Deconvolver's own shares", p, []), ("planted shares declared", p, ["--dev-shares", dec]),
                                      ("planted shares declared, 300 chr8 sites +0.15", pr, ["--dev-shares", dec]))):
    a = np.zeros((SV.HG19_CPGS, 2), np.uint16); ii = cpg[q.index].to_numpy(); a[ii - 1, 0] = rng.binomial(N, q.to_numpy()); a[ii - 1, 1] = N
    f = os.path.join(W, f"d{j}.lbeta"); a.tofile(f); b = os.path.join(W, f"d{j}_bundle.json")
    subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--dev-seq-ref", "auto", "--full-chain", "--specimen", "unknown",
                    "--id", f"d{j}", "--out", os.path.join(W, f"d{j}.html"), "--bundle", b] + flags, cwd=os.path.join(C, "MethylPhys_Interface"), capture_output=True, text=True)
    B = json.load(open(b)); s = (B.get("development") or {}).get("sky") or {}
    L.append(f"{name} | {s.get('lee_p')} | {s.get('structure_beyond_null')} | {max(s.get('ratio_to_block_null') or [0]):.4f}" + ("" if s.get("lee_p") is not None else f" | {s.get('reason')}"))
    if j == 0: L.append("  the Deconvolver read: " + ", ".join(f"{x['cell']} {x['share']:.3f}" for x in sorted((B.get("cell_shares") or {}).get("cells") or [], key=lambda x: -x["share"]) if x["share"] >= 0.005))
    os.remove(f)
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
