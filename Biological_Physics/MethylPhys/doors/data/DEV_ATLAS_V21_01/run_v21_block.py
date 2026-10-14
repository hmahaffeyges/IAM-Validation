#!/usr/bin/env python3
"""Atlas v2.1 stage B, one block (readiness L9; committed 2026-10-14 before any block is fitted). v2's 14_stageB_block.py data assembly,
line for line, on the FINAL build table (final_samples_v21.csv, 702 samples, 79 cells) instead of v2's roster; the model and saves are
atlas_block_v21.py (planted test MET twice). Loci, blocks and seeds are v2's: every array CpG with Loyfer depth >= 10 in > 90 % of Loyfer
samples, split into NBLOCK = 700; seed 1000 + block. Terms: transfer_v21_final.json (one bounded curve for every WGBS source, its measured
cell scatter, the array laboratories' offsets d; L22 MET). Kinds: Loyfer and ENCODE 'wgbs', Tian 'pooled' (v2's), arrays 'array'.
Usage: BLOCK=<i> NBLOCK=700 OUTD=<dir> [K=100] [DRYRUN=1] python run_v21_block.py. DRYRUN assembles the observations and prints counts only.
A finished block (its parquet present) is skipped."""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
R = "/home/ubuntu/data/atlas_sources"; OUTD = os.environ.get("OUTD", "/home/ubuntu/data/atlas_v21/blocks_v21"); os.makedirs(OUTD, exist_ok=True)
BI = int(os.environ["BLOCK"]); NB = int(os.environ["NBLOCK"]); K = int(os.environ.get("K", "100")); dst = f"{OUTD}/block_{BI:05d}.parquet"
if os.path.exists(dst) and not os.environ.get("DRYRUN"): print("exists", dst); sys.exit(0)
TERMS = json.load(open("transfer_v21_final.json"))
S = pd.read_csv("final_samples_v21.csv", keep_default_na=False); S = S[S.qc.astype(str) == "True"]
B = pd.read_parquet(f"{R}/loyfer2023/array_beta.parquet"); C = pd.read_parquet(f"{R}/loyfer2023/array_cov.parquet")
allloci = np.sort(B.index[(C >= 10).mean(axis=1) > 0.9].values); loci = np.array_split(allloci, NB)[BI]; NL = len(loci); li = {l: i for i, l in enumerate(loci)}
obs = []
def add(cell, src, kind, vals, cov=None):
    v = vals.reindex(loci); cv = None if cov is None else cov.reindex(loci)
    for l, y in v.items():
        if np.isfinite(y) and (cv is None or (np.isfinite(cv[l]) and cv[l] >= 5)): obs.append((cell, src, kind, li[l], float(y), float(cv[l]) if cv is not None else 0.0))
SUB = {"Moss2018": "moss2018/shards", "Salas2018": "blood/GSE110554/shards", "Salas2022": "blood/GSE167998/shards", "GSE63409": "hsc_gse63409/shards"}
for _, r in S.iterrows():
    if r.source == "Loyfer2023": add(r.cell, r.source, "wgbs", B[r["sample"]], C[r["sample"]])
    elif r.source == "Tian2023": d = pd.read_parquet(f"{R}/tian2023/{r['sample']}_array.parquet"); add(r.cell, r.source, "pooled", d["beta"], d["cov"])
    elif r.source == "ENCODE": d = pd.read_parquet(r["sample"]); add(r.cell, r.source, "wgbs", d["beta"], d["cov"])
    else:
        p = r["path"] or f"{R}/{SUB[r.source]}/{r['sample']}.parquet"
        assert r.source in TERMS["array"], f"no array term for {r.source}"
        add(r.cell, r.source, "array", pd.read_parquet(p).iloc[:, 0])
O = pd.DataFrame(obs, columns=["cell", "src", "kind", "l", "y", "cov"]); O["y"] = O.y.clip(1e-3, 1 - 1e-3)
if os.environ.get("DRYRUN"):
    print(json.dumps({"block": BI, "loci": NL, "obs": len(O), "cells": int(O.cell.nunique()), "by_kind": O.kind.value_counts().to_dict(),
                      "sources": int(O.src.nunique()), "cells_without_obs": sorted(set(S.cell) - set(O.cell))})); sys.exit(0)
import atlas_block_v21 as AB
t0 = time.time(); RES = AB.fit_block(O, loci, TERMS, seed=1000 + BI, mode="v21", K=K); AB.write_block(RES, dst)
W = RES["warnings"]
print(json.dumps({"block": BI, "loci": NL, "cells": W["cells"], "divergences": W["divergences"], "rhat_p99": RES["rhat_p99"], "ess_p01": RES["ess_p01"],
                  "s_per_locus": round((time.time() - t0) / max(NL, 1), 4), "K": K}))
