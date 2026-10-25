"""DEV-METAA-HEALTHY-01 step 3 version 3, synthetic check (written 2026-10-24 before the real counts), through the one command. A constructed
molecule file (.pat) over Met-A's neutrophil identity sites: 20 molecules of 10 consecutive CpGs at each site. At the sites the atlas holds
marked: 88 % held molecules (each call marked at 0.97), 8 % loose (0.5), 4 % held absent (0.02), expected level 0.8944; at the held-absent
sites the mirror (4 % at 0.97, 8 % at 0.5, 88 % at 0.02), expected level 0.0964. Read by run_sample.py --dev-count-only --dev-count-sites
neutrophils; the levels and class shares it records are compared with the planted ones.  Usage: python3 sites_synthetic_02.py OUT.txt"""
import sys, os, json, gzip, subprocess, numpy as np, pandas as pd, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.join(H, "..", "..", ".."); RS = os.path.join(MP, "chain", "MethylPhys_Interface", "run_sample.py")
F = json.load(open(os.path.join(MP, "chain", "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]
K = pd.read_parquet(os.path.join(MP, "atlas", "v2.1", "sites_sequencing_key.parquet")); K = K[K.probe.isin(set(F["sites"]))].dropna(subset=["loyfer_cpg_index"])
AT = os.path.join(MP, "atlas", "v2.1"); ids = pq.read_table(os.path.join(AT, "sites.parquet")).column("cpg").to_pylist()
u = next(pq.read_table(os.path.join(AT, f"mu_part{n}.parquet"), columns=["neutrophils"]).column(0).to_numpy() for n in (1, 2, 3) if "neutrophils" in pq.ParquetFile(os.path.join(AT, f"mu_part{n}.parquet")).schema.names)
lv = dict(zip(ids, u)); g = np.random.default_rng(20261024); rows = []
for r in K.itertuples():
    v = lv.get(r.probe, 65535)
    if v == 65535: continue
    held = v / 1e4 >= 0.5; i = int(r.loyfer_cpg_index); c = str(r.chrom_hg19)
    for _ in range(20):
        x = g.random(); m = (0.97 if x < 0.88 else 0.5 if x < 0.96 else 0.02) if held else (0.97 if x < 0.04 else 0.5 if x < 0.12 else 0.02)
        s = i - int(g.integers(0, 10)); rows.append((c, s, "".join("C" if g.random() < m else "T" for _ in range(10))))
rows.sort(key=lambda t: t[1]); pat = "/tmp/sites_synthetic_02.pat.gz"
with gzip.open(pat, "wt") as f:
    for c, s, p in rows: f.write(f"{c}\t{s}\t{p}\t1\n")
b = "/tmp/sites_synthetic_02.json"
subprocess.run([sys.executable, RS, "--pat", pat, "--specimen", "plasma", "--id", "synthetic", "--dev-count-only", "--dev-count-sites", "neutrophils",
                "--out", "/tmp/sites_synthetic_02.html", "--bundle", b], capture_output=True, text=True)
R = json.load(open(b))["iam_a_count_only"]["count_site_classes"]
T = ["planted at held sites: 0.88 held at 0.97, 0.08 loose at 0.5, 0.04 absent at 0.02 (level 0.8944); at held-absent sites the mirror (level 0.0964)"]
for k in ("held", "absent"):
    x = R[k]; T.append(f"{k} sites: level {x['level']:.4f} | shares held {x['held_share']:.4f}, loose {x['loose_share']:.4f}, absent {x['absent_share']:.4f} | {x['calls']} calls")
open(sys.argv[1], "w").write("\n".join(T) + "\n"); print("\n".join(T))
