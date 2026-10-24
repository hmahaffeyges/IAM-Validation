"""DEV-METAA-HEALTHY-01 step 3 diagnostic (written AFTER step 3 was scored; disclosed). At Met-A's commissioned neutrophil identity sites with
an IAM-Atlas v2.1 neutrophil level: the atlas's own levels on each side (held: level >= 0.5), and the mean entropy of those levels, set beside
the derived single levels, to tell the arrays' measurement blur from the sites' own levels.  Usage: python3 step3_diag.py OUT.txt"""
import sys, os, json, numpy as np, pandas as pd, pyarrow.parquet as pq
H0 = os.path.dirname(os.path.abspath(__file__)); MP = os.path.join(H0, "..", "..", "..")
F = json.load(open(os.path.join(MP, "chain", "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]
AT = os.path.join(MP, "atlas", "v2.1"); ids = pq.read_table(os.path.join(AT, "sites.parquet")).column("cpg").to_pylist()
u = next(pq.read_table(os.path.join(AT, f"mu_part{n}.parquet"), columns=["neutrophils"]).column(0).to_numpy() for n in (1, 2, 3) if "neutrophils" in pq.ParquetFile(os.path.join(AT, f"mu_part{n}.parquet")).schema.names)
st = pd.Series(np.where(u == 65535, np.nan, u / 1e4), index=ids).reindex(F["sites"]).dropna(); held = st >= 0.5
H = lambda b: -(b * np.log2(b) + (1 - b) * np.log2(1 - b)); b = st.clip(1e-6, 1 - 1e-6)
L = [f"sites {len(st)} | atlas level held: median {st[held].median():.4f} mean {st[held].mean():.4f} | absent: median {st[~held].median():.4f} mean {st[~held].mean():.4f}",
     f"mean entropy of the atlas's own levels: {float(H(b).mean()):.5f} bits | held {float(H(b[held]).mean()):.5f} | absent {float(H(b[~held]).mean()):.5f}",
     f"held-site atlas level deciles 10/25/50/75/90: " + " ".join(f"{x:.3f}" for x in st[held].quantile([.1, .25, .5, .75, .9]))]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
