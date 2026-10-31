"""The sky on sequencing: diagnostic after synthetic run 2 (disclosed; nothing scored). Synthetic sample 3 (cd8 t cells 0.7, cd4 t cells 0.3,
8 reads, the same seed and draw order) rebuilt at IAM-Atlas v2.1's sites and read by the chain's own sky (dev_stages.sky_in_mix, the
IAM Sequencing Atlas active) three ways: against its planted shares; against the shares the one command read (0.67 and 0.29, the rest to
other cells, from run 2's bundle); and, with the planted shares, after the first 300 sites of chromosome 8 in genome order moved 0.15 toward
full methylation (a regional change, which a within-chromosome block shuffle does not absorb).
Usage: python3 sky_seq_diag.py RUN2_BUNDLE_DIR OUT.txt"""
import sys, os, json, numpy as np, pandas as pd, pyarrow.parquet as pq
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"); sys.path.insert(0, C)
import stage_seq_atlas as Q, stage_metaa_atlas as SA, dev_stages as DV
SA.SEQ_ATLAS = Q.ARRAY_COLS
ids = pd.Index(pq.read_table(os.path.join(SA.ATLAS, "sites.parquet")).column("cpg").to_pylist())
cells = [c for k in (1, 2, 3) for c in pq.read_schema(os.path.join(Q.ARRAY_COLS, f"mu_part{k}.parquet")).names]
rng = np.random.default_rng(20261101); pick = list(rng.choice(cells, 12, replace=False)); assert pick[4:6] == ["cd8 t cells", "cd4 t cells"], pick[4:6]
mx = {"cd8 t cells": 0.7, "cd4 t cells": 0.3}; N = 8
p = sum(f * SA._col("mu", c) for c, f in mx.items()); ok = p.notna(); rr = np.random.default_rng(7)
k = pd.Series(rr.binomial(N, p[ok].clip(0, 1).to_numpy()), index=p[ok].index); beta = k / N; n = pd.Series(N, index=k.index)
B = json.load(open(os.path.join(sys.argv[1], "k3_bundle.json"))); read = {x["cell"]: x["share"] for x in (B.get("cell_shares") or {}).get("cells") or [] if x["share"] > 0}
m = DV._mapping(); c8 = m[m.chr.astype(str) == "8"].index.intersection(beta.index)
pos = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
reg = pos.reindex(c8).dropna().sort_values().index[:300]
bp = beta.copy(); bp[reg] = (beta[reg] + 0.15 * (1 - beta[reg])).clip(0, 1)
L = ["The sky on sequencing, diagnostic (sample 3 of the synthetic rebuilt; cd8 0.7, cd4 0.3, 8 reads)", "read | lee p | structure | largest band ratio"]
for name, b, sh in (("planted shares", beta, mx), ("shares the chain read", beta, read), ("planted shares, 300 chr8 sites +0.15", bp, mx)):
    s = DV.sky_in_mix(b, sh, counts=n)
    L.append(f"{name} | {s.get('lee_p')} | {s.get('structure_beyond_null')} | {max(s.get('ratio_to_block_null') or [0]):.4f}")
L.append("shares the chain read: " + ", ".join(f"{c} {v:.3f}" for c, v in sorted(read.items(), key=lambda t: -t[1]) if v >= 0.005))
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
