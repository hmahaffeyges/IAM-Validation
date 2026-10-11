"""DEV-DIRECTION-AUDIT-01, Met-A under planted GAIN and planted LOSS (healthy arrays only; 2026-10-11). The six physical Salas neutrophil EPIC arrays
(every neutrophil array in the roster; they are the reference arrays, so the baseline reads near 1 by construction: this tests the DIRECTION and size of
the response to a known change, not accuracy on new data), calibrated by chain Stage 1 (the clean-room betas). At the 6,000 identity sites (metA_floors_v1_3.json):
GAIN: at sites the array reads unmethylated (beta < 0.5), beta -> beta + d (1 - beta) (a share d of molecules gains the mark);
LOSS: at sites it reads methylated (beta >= 0.5), beta -> beta (1 - d). d = 0.02 / 0.05 / 0.10. Read by the commissioned path
(conductor_v3.run_neutrophil, specimen "isolated neutrophils"), Met-A before any tare. Usage: python3 plant_meta.py BETADIR OUT.csv"""
import os, sys, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(MP, "chain")); import conductor_v3 as C3
BD, OUT = sys.argv[1], sys.argv[2]
REF = {"GSM2998021", "GSM2998057", "GSM2998116", "GSM2998023", "GSM2998143", "GSM2998030"}
St = pd.read_csv(os.path.join(MP, "atlas/v2/inputs/roster_samples.csv")); St["gsm"] = St["sample"].str.split("_").str[0]
neu = sorted({g for g in St[(St.cell == "neutrophils") & St.source.isin(["Salas2018", "Salas2022"]) & (St.qc == True)].gsm if os.path.exists(os.path.join(BD, g + ".parquet"))})
S = json.load(open(os.path.join(MP, "chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]["sites"]
rows = []
for g in neu:
    b0 = pd.read_parquet(os.path.join(BD, g + ".parquet")).beta; b0.index = b0.index.astype(str)
    s = [x for x in S if x in b0.index and pd.notna(b0[x])]; lo = [x for x in s if b0[x] < 0.5]; hi = [x for x in s if b0[x] >= 0.5]
    def A(b): o = C3.run_neutrophil(b, specimen="isolated neutrophils"); return (o.get("met_a") or {}).get("A")
    a0 = A(b0); rows.append(dict(gsm=g, plant="none", d=0.0, A=a0, n_lo=len(lo), n_hi=len(hi)))
    for d in (0.02, 0.05, 0.10):
        bg = b0.copy(); bg[lo] = bg[lo] + d * (1 - bg[lo]); rows.append(dict(gsm=g, plant="gain", d=d, A=A(bg), n_lo=len(lo), n_hi=len(hi)))
        bl = b0.copy(); bl[hi] = bl[hi] * (1 - d); rows.append(dict(gsm=g, plant="loss", d=d, A=A(bl), n_lo=len(lo), n_hi=len(hi)))
    print(g, [(r["plant"], r["d"], round(r["A"], 4)) for r in rows if r["gsm"] == g], flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
