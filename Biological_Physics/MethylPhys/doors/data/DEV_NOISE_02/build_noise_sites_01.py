"""Builds chain/Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json from public data (DEV-NOISE-01/02; rule as run 2026-10-01).
Arrays: Salas purified blood cells (GSE110554, GSE167998) in atlas/v2/inputs/roster_samples.csv with qc True, 15 cell labels in 8 groups.
Betas: chain Stage 1 on the GEO IDATs (downloaded here). Rule: every group's mean beta <= 0.03 (or >= 0.97) and every group's SD <= 0.02;
the chain's neutrophil sites (blood_composition_EPIC_v1 'neutrophil_sites') excluded. Sites listed in the order the rule yields them.
Run (shard k of n, then 'build'): HOME=<methylprep manifest dir> python3 build_noise_sites_01.py WORKDIR calib k n | build"""
import os, sys, json, urllib.request, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(MP, "chain")); W = sys.argv[1]; os.makedirs(os.path.join(W, "betas"), exist_ok=True)
GRP = {"neutrophils": "NEU", "eosinophils": "EOS", "basophils": "BASO", "monocytes": "MONO", "b cells": "B", "naive b cells": "B", "memory b cells": "B",
       "nk cells": "NK", "cd4 t cells": "CD4T", "naive cd4 t cells": "CD4T", "memory cd4 t cells": "CD4T", "regulatory t cells": "CD4T",
       "cd8 t cells": "CD8T", "naive cd8 t cells": "CD8T", "effector memory cd8 t cells": "CD8T"}
St = pd.read_csv(os.path.join(MP, "atlas/v2/inputs/roster_samples.csv"))
St = St[St.source.isin(["Salas2018", "Salas2022"]) & (St.qc == True) & St.cell.isin(GRP)].copy(); St["gsm"] = St["sample"].str.split("_").str[0]
if sys.argv[2] == "calib":
    from stage_1_idat_calibration import calibrate_idat_to_beta
    k, n = int(sys.argv[3]), int(sys.argv[4])
    for _, r in St.sort_values("sample").reset_index(drop=True).iloc[k::n].iterrows():
        dst = os.path.join(W, "betas", r.gsm + ".parquet")
        if os.path.exists(dst): continue
        p = {}
        for ch in ("Grn", "Red"):
            u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{r.gsm[:-3]}nnn/{r.gsm}/suppl/{r['sample']}_{ch}.idat.gz"; p[ch] = os.path.join(W, f"{r['sample']}_{ch}.idat.gz")
            urllib.request.urlretrieve(u, p[ch])
        o = calibrate_idat_to_beta(p["Grn"], p["Red"], verbose=False); b = (o[0] if isinstance(o, tuple) else o).astype("float64")
        b.index = b.index.astype(str); b.rename("beta").to_frame().to_parquet(dst); [os.remove(x) for x in p.values()]; print(r.gsm, "ok", flush=True)
    print("SHARD_DONE", k, flush=True)
else:
    import conductor_v3 as C3
    M = pd.DataFrame({r.gsm: pd.read_parquet(os.path.join(W, "betas", r.gsm + ".parquet")).beta for _, r in St.iterrows()})
    grp = {r.gsm: GRP[r.cell] for _, r in St.iterrows()}; NS = set(C3._bc()["neutrophil_sites"])
    mu = pd.DataFrame({g: M[[x for x in M if grp[x] == g]].mean(1) for g in sorted(set(grp.values()))})
    sd = pd.DataFrame({g: M[[x for x in M if grp[x] == g]].std(1) for g in sorted(set(grp.values()))})
    lo = (mu.max(1) <= 0.03) & (sd.max(1) <= 0.02); hi = (mu.min(1) >= 0.97) & (sd.max(1) <= 0.02)
    INV = [s for s in mu.index[lo | hi] if s not in NS]
    ref = json.load(open(os.path.join(MP, "chain/Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json")))
    print(f"arrays {M.shape[1]} | sites {len(INV)} (low {int(lo.sum())}, high {int(hi.sum())}) | committed {ref['n']} | "
          f"same set {set(INV) == set(ref['sites'])} | only here {len(set(INV) - set(ref['sites']))} | only committed {len(set(ref['sites']) - set(INV))}")
    json.dump({"n": len(INV), "sites": INV}, open(os.path.join(W, "noise_sites_rebuilt.json"), "w"))
