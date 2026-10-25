"""DEV-METAA-HEALTHY-01 step 3 version 6, synthetic check (written 2026-10-24 before the real readings), through the one command. Constructed
molecules at IAM-Atlas v2.1's sites (a .lbeta at 1,000 calls a site): Met-A's unmarked identity sites at 0.005, marked ones at 0.94, the other
sites 40 % at 0.003, 40 % at 0.995, 20 % spread evenly. Two constructed arrays read the same DNA through their own line: F0 0.06, F1 0.93 and
F0 0.04, F1 0.95 (value = F0 + (F1 - F0) m at every site). All three are read by run_sample.py with --save-betas (arrays as --betas files);
v6_score.py must recover each array's F0 and F1 and read each array 1.0000 against its own derived mark.  Usage: python3 v6_synthetic.py OUT.txt"""
import sys, os, json, subprocess, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "chain"); RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py")
sys.path.insert(0, CH); import stage_seq_values as SV
F = json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]
K = pd.read_parquet(os.path.join(CH, "..", "atlas", "v2.1", "sites_sequencing_key.parquet")).dropna(subset=["loyfer_cpg_index"]); g = np.random.default_rng(20261024)
u = g.random(len(K)); m = np.where(u < 0.4, 0.003, np.where(u < 0.8, 0.995, g.random(len(K)))); M = pd.Series(m, index=K.probe.to_numpy())
import pyarrow.parquet as pq
AT = os.path.join(CH, "..", "atlas", "v2.1"); ids = pq.read_table(os.path.join(AT, "sites.parquet")).column("cpg").to_pylist()
nu = next(pq.read_table(os.path.join(AT, f"mu_part{n}.parquet"), columns=["neutrophils"]).column(0).to_numpy() for n in (1, 2, 3) if "neutrophils" in pq.ParquetFile(os.path.join(AT, f"mu_part{n}.parquet")).schema.names)
lv = pd.Series(nu, index=ids).reindex(F["sites"]); lv = lv[lv.notna() & (lv != 65535)] / 1e4; j = lv.index.intersection(M.index); M[j] = np.where(lv[j] >= 0.5, 0.94, 0.005)
W = "/tmp/v6syn"; R = os.path.join(W, "readings"); os.makedirs(R, exist_ok=True)
a = np.zeros((SV.HG19_CPGS, 2), np.uint16); i = K.loyfer_cpg_index.to_numpy(np.int64) - 1; a[i, 1] = 1000; a[i, 0] = np.rint(M.to_numpy() * 1000).astype(np.uint16); a.tofile(f"{W}/syn.lbeta")
runs = [("sequencing", "SYN", ["--dev-seq-beta", f"{W}/syn.lbeta", "--dev-seq-values", "10"])]
for nm, f0, f1 in (("ARR1", 0.06, 0.93), ("ARR2", 0.04, 0.95)):
    (f0 + (f1 - f0) * M).rename("beta").to_frame().to_csv(f"{W}/{nm}.csv"); runs.append(("array", nm, ["--betas", f"{W}/{nm}.csv"]))
for kind, nm, src in runs:
    p = subprocess.run([sys.executable, RS] + src + ["--specimen", "whole blood", "--id", "v6syn_" + nm, "--save-betas", f"{R}/{kind}_{nm}_v6.parquet", "--out", f"{W}/{nm}.html", "--bundle", f"{W}/{nm}.json"], capture_output=True, text=True)
    print(kind, nm, "exit", p.returncode, flush=True)
p = subprocess.run([sys.executable, os.path.join(H, "v6_score.py"), R, sys.argv[1]], capture_output=True, text=True); print(p.stdout, p.stderr[-800:])
T = open(sys.argv[1]).read().rstrip("\n") + "\nplanted: ARR1 F0 0.06, F1 0.93; ARR2 F0 0.04, F1 0.95; expected Met-A 1.0000 for both\n"; open(sys.argv[1], "w").write(T)
