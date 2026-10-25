"""DEV-METAA-HEALTHY-01 step 3 version 4, synthetic check (written 2026-10-24 before any real reading), through the one command. One made-up
cell type read on both platforms: the array is a real healthy blood (GSE235717 GSM7508429's values, spent) with Met-A's neutrophil identity
sites replaced; the sequencing file is a constructed .beta (100 calls a site, so both levels are exact; first run at 40 calls rounded 0.01 to 0) at the atlas sites. At the identity sites the sequencing level is
0.01 (unmarked side) or 0.93 (marked side) and the array reads the atlas's inverse transfer of it plus a planted per-site difference
(unmarked side 0.02 to 0.08, marked side -0.02 to 0.01). Both are read by run_sample.py with --save-betas, cut to the identity sites as
v4_read.py does, and v4_score.offsets() must recover the planted differences.  Usage: python3 v4_synthetic.py BETAS.csv OUT.txt"""
import sys, os, json, subprocess, numpy as np, pandas as pd, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "chain"); RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py")
sys.path.insert(0, H); sys.path.insert(0, CH); import v4_score as V; import stage_seq_values as SV
F = V.F; side = V.side; g = np.random.default_rng(20261024); W = "/tmp/v4syn"; os.makedirs(W, exist_ok=True)
B = pd.read_csv(sys.argv[1], index_col=0).iloc[:, 0]; B.index = B.index.astype(str)
s = pd.Series(np.where(side == "absent", 0.01, 0.93), index=side.index)
dp = pd.Series(np.where(side == "absent", g.uniform(0.02, 0.08, len(side)), g.uniform(-0.02, 0.01, len(side))), index=side.index)
A = B.copy(); A.loc[side.index.intersection(A.index)] = (V.inv(s.to_numpy()) + dp.to_numpy())[side.index.isin(A.index)].clip(0, 1); A.rename("beta").to_frame().to_csv(f"{W}/syn_betas.csv")
K = pd.read_parquet(os.path.join(CH, "..", "atlas", "v2.1", "sites_sequencing_key.parquet")).dropna(subset=["loyfer_cpg_index"])
lvl = pd.Series(V.g(B.reindex(K.probe).fillna(0.5).clip(0, 1).to_numpy()), index=K.probe.to_numpy()); lvl.loc[lvl.index.intersection(s.index)] = s.reindex(lvl.index.intersection(s.index))
a = np.zeros((SV.HG19_CPGS, 2), np.uint8); i = K.loyfer_cpg_index.to_numpy(np.int64) - 1; a[i, 1] = 100; a[i, 0] = np.rint(lvl.clip(0, 1).to_numpy() * 100).astype(np.uint8); a.tofile(f"{W}/syn.beta")
rows = []
for tag, src in (("array", ["--betas", f"{W}/syn_betas.csv"]), ("sequencing", ["--dev-seq-beta", f"{W}/syn.beta", "--dev-seq-values", "10"])):
    sv = f"{W}/{tag}.parquet"; p = subprocess.run([sys.executable, RS] + src + ["--specimen", "whole blood", "--id", "v4syn_" + tag, "--save-betas", sv, "--out", f"{W}/{tag}.html", "--bundle", f"{W}/{tag}.json"], capture_output=True, text=True)
    v = pd.read_parquet(sv).iloc[:, 0]; v.index = v.index.astype(str); v = v[v.index.isin(side.index)]
    rows.append(v.rename("value").to_frame().assign(gsm=tag, platform=tag, cell="synthetic")); print(tag, "exit", p.returncode, len(v), flush=True)
X = pd.concat(rows); d, D, use, cells = V.offsets(X, side); j = d.index.intersection(dp.index)
T = ["planted per-site differences: unmarked side 0.02 to 0.08, marked side -0.02 to 0.01"]
for s_ in ("held", "absent"):
    k = j[side.reindex(j) == s_]; e = (d.reindex(k) - dp.reindex(k)).abs()
    T.append(f"{s_}: {len(k)} sites | planted mean {dp.reindex(k).mean():.4f}, recovered mean {d.reindex(k).mean():.4f} | median abs error {e.median():.4f}, 95th {e.quantile(0.95):.4f} | correlation {np.corrcoef(d.reindex(k), dp.reindex(k))[0, 1]:.3f}")
open(sys.argv[2], "w").write("\n".join(T) + "\n"); print("\n".join(T))
