"""DEV-METAA-HEALTHY-01 step 3 version 5, synthetic check (written 2026-10-24 before any BLUEPRINT file is read; runs on GitHub, where
pyBigWig installs). A constructed GRCh38 bigWig pair at IAM-Atlas v2.1's sites: level 0.01 at the unmarked identity sites, 0.93 at the marked
ones, 0.5 elsewhere; coverage 30; one site in ten written at the next base (the G) instead of the C. Read by the one command
(--dev-seq-beta CALL.bw,COV.bw), values saved and cut as v5_read.py does; the levels must come back exactly and every site be found.
Usage: python3 v5_synthetic.py OUT.txt"""
import sys, os, json, subprocess, numpy as np, pandas as pd, pyBigWig
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "chain"); RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py")
sys.path.insert(0, H); import v4_score as V
K = pd.read_parquet(os.path.join(CH, "..", "atlas", "v2.1", "sites_key_hg38.parquet")); W = "/tmp/v5syn"; os.makedirs(W, exist_ok=True)
lv = pd.Series(0.5, index=K.probe.to_numpy()); j = V.side.index.intersection(lv.index); lv[j] = np.where(V.side[j] == "absent", 0.01, 0.93)
g = np.random.default_rng(20261024); K = K.assign(lv=lv.to_numpy(), pos=K.beg0_hg38 + (g.random(len(K)) < 0.1)).sort_values(["chrom_hg38", "pos"]).drop_duplicates(["chrom_hg38", "pos"])
CL = {"chr%s" % c: 250_000_000 for c in list(range(1, 23)) + ["X", "Y"]}
for nm, col in (("call", "lv"), ("cov", None)):
    b = pyBigWig.open(f"{W}/{nm}.bw", "w"); b.addHeader([(c, CL[c]) for c in CL if c in set(K.chrom_hg38)])
    for c in [c for c in CL if c in set(K.chrom_hg38)]:
        x = K[K.chrom_hg38 == c]; b.addEntries([c] * len(x), x.pos.astype(int).tolist(), ends=(x.pos + 1).astype(int).tolist(), values=(x[col] if col else pd.Series(30.0, index=x.index)).astype(float).tolist())
    b.close()
sv = f"{W}/v.parquet"; p = subprocess.run([sys.executable, RS, "--dev-seq-beta", f"{W}/call.bw,{W}/cov.bw", "--dev-seq-values", "10", "--specimen", "whole blood", "--id", "v5syn", "--save-betas", sv, "--out", f"{W}/s.html", "--bundle", f"{W}/s.json"], capture_output=True, text=True)
v = pd.read_parquet(sv).iloc[:, 0]; v.index = v.index.astype(str); r = v.reindex(j); B = json.load(open(f"{W}/s.json"))
T = [f"run_sample exit {p.returncode} | sites with a value {len(v)} of {len(K)} written | read at the next base: {(B.get('sequencing_values') or {}).get('read_at_next_base')}"]
for k, e in (("held", 0.93), ("absent", 0.01)):
    x = r[V.side[j] == k]; T.append(f"{k}: {int(x.notna().sum())} of {len(x)} identity sites found | planted {e} | read mean {x.mean():.4f}, max error {(x - e).abs().max():.4f}")
open(sys.argv[1], "w").write("\n".join(T) + "\n"); print("\n".join(T)); print(p.stdout[-500:] if p.returncode else "")
