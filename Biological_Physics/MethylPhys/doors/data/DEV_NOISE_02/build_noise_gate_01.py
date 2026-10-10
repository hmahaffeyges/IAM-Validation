"""Builds N_max of chain/Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json from public data (rule set 2026-10-02):
N = mean H(beta) over noise_sites_EPIC_v1 sites; N_max = the highest N among the Salas purified neutrophil arrays (GSE110554, GSE167998;
roster qc True), betas from chain Stage 1 on the GEO IDATs (build_noise_sites_01.py 'calib' step). Run: python3 build_noise_gate_01.py WORKDIR"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
St = pd.read_csv(os.path.join(MP, "atlas/v2/inputs/roster_samples.csv"))
St = St[St.source.isin(["Salas2018", "Salas2022"]) & (St.qc == True) & (St.cell == "neutrophils")]; St = St.assign(gsm=St["sample"].str.split("_").str[0])
S = json.load(open(os.path.join(MP, "chain/Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json")))["sites"]
G = json.load(open(os.path.join(MP, "chain/Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json")))
def H(b): b = np.clip(b, 1e-6, 1 - 1e-6); return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))
N = {g: float(H(pd.read_parquet(os.path.join(sys.argv[1], "betas", g + ".parquet")).beta.reindex(S).dropna()).mean()) for g in St.gsm}
for g, v in sorted(N.items()): print(g, f"{v:.4f}")
print(f"arrays {len(N)} | N {min(N.values()):.4f}-{max(N.values()):.4f} | N_max rebuilt {round(max(N.values()), 3)} | chain {G['N_max']} | match {round(max(N.values()), 3) == G['N_max']}")
