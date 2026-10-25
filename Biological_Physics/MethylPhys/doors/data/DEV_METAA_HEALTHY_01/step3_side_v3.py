"""DEV-METAA-HEALTHY-01 step 3 version 3, side diagnostic (added after version 3 was scored; disclosed; nothing scored). Version 3's derived
levels set beside IAM-Atlas v2.1's neutrophil levels (array scale, from purified neutrophil arrays) at the held and the held-absent identity
sites, each side's mean entropy, and the sites by Infinium probe design (public EPIC v1.0 B4 manifest).
Usage: python3 step3_side_v3.py EPIC_MANIFEST_CORECOLUMNS.csv.gz OUT.txt   (derived levels read from step3_03/step3_v3_summary.txt)"""
import sys, os, re, json, numpy as np, pandas as pd, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.join(H, "..", "..", "..")
F = json.load(open(os.path.join(MP, "chain", "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]
AT = os.path.join(MP, "atlas", "v2.1"); ids = pq.read_table(os.path.join(AT, "sites.parquet")).column("cpg").to_pylist()
u = next(pq.read_table(os.path.join(AT, f"mu_part{n}.parquet"), columns=["neutrophils"]).column(0).to_numpy() for n in (1, 2, 3) if "neutrophils" in pq.ParquetFile(os.path.join(AT, f"mu_part{n}.parquet")).schema.names)
S = pd.DataFrame({"probe": F["sites"]}); S["lv"] = S.probe.map(pd.Series(u, index=ids)); S = S[S.lv.notna() & (S.lv != 65535)].copy(); S["lv"] /= 1e4
S["side"] = np.where(S.lv >= 0.5, "held", "absent")
A = pd.read_csv(sys.argv[1], usecols=["IlmnID", "Infinium_Design_Type"]).rename(columns={"IlmnID": "probe", "Infinium_Design_Type": "design"}); S = S.merge(A, on="probe", how="left")
t = open(os.path.join(H, "step3_03", "step3_v3_summary.txt")).read(); m = re.search(r"array-scale level held ([0-9.]+), held absent ([0-9.]+)", t); d = {"held": float(m.group(1)), "absent": float(m.group(2))}
Hb = lambda b: -(b * np.log2(b) + (1 - b) * np.log2(1 - b)); L = ["side | sites | atlas mean level | atlas mean entropy | version 3 level | its entropy"]
for s in ("held", "absent"):
    b = S.lv[S.side == s].clip(1e-6, 1 - 1e-6); L.append(f"{s} | {len(b)} | {b.mean():.4f} | {Hb(b).mean():.4f} | {d[s]:.4f} | {Hb(d[s]):.4f}")
L.append("side, probe design | sites | atlas mean level")
for (s, g), x in S.groupby(["side", "design"]): L.append(f"{s}, type {g} | {len(x)} | {x.lv.mean():.4f}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
