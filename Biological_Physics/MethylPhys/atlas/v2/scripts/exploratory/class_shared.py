#!/usr/bin/env python3
"""EXPLORATORY (author, 2026-09-28): for each architecture class, the CpGs ALL its cells share - unmethylated (beta < 0.3) in every
cell of the class and methylated (beta > 0.7) in >= 90 % of the cells outside it - to be drawn on the sky. No H_min, no A.
Relaxed set: unmethylated in >= 80 % of the class's cells. Same 56 Loyfer cells and labels as class_structure.py.
Nulls, so a shared set is not mistaken for a lineage set:
  N-random: 1,000 random groups of the same size from all 56 cells, same rule.
  N-lineage: where a class is part of a lineage it does not fill - cycling and secretory are both epithelia - 1,000 random groups of the
  same size drawn from the 26 epithelial cells only. For terminal (2 neural + 2 muscle) the neural pair and the muscle pair alone are
  reported beside the 4-cell set: whatever survives the 4-cell intersection is what neurons and muscle share as terminal cells.
Out: class_shared_sets.csv (cpg, set), class_shared.json (sizes and null distributions)."""
import json, numpy as np, pandas as pd
R="/home/ubuntu/data/atlas_sources"; rng=np.random.default_rng(20260928)
lab=json.load(open("class_structure.json"))["labels"]
RO=pd.read_csv("atlas_v2_roster.csv",index_col=0); S=pd.read_csv("roster_samples.csv"); S=S[(S.source=="Loyfer2023")&S.qc]
cells=sorted(lab); B=pd.read_parquet(f"{R}/loyfer2023/array_beta.parquet"); C=pd.read_parquet(f"{R}/loyfer2023/array_cov.parquet")
P=pd.DataFrame({c:B[S[S.cell==c]["sample"].tolist()].where(C[S[S.cell==c]["sample"].tolist()]>=10).mean(axis=1,skipna=False) for c in cells}).dropna()
LO=(P<0.3).values; HI=(P>0.7).values; idx={c:i for i,c in enumerate(P.columns)}
def shared(group,frac=1.0):
    g=[idx[c] for c in group]; o=[i for i in range(P.shape[1]) if i not in g]
    return (LO[:,g].mean(1)>=frac-1e-9)&(HI[:,o].mean(1)>=0.9)
CLS={c:lab[c][0] for c in cells}; FAM={c:lab[c][1] for c in cells}
classes=sorted(set(CLS.values())); rows=[]; out={"cpgs":int(len(P))}
epi=[c for c in cells if FAM[c]=="epithelial"]
for k in classes:
    g=[c for c in cells if CLS[c]==k]; m=shared(g); mr=shared(g,0.8)
    for cg in P.index[m]: rows.append((cg,k+"_all"))
    for cg in P.index[mr]: rows.append((cg,k+"_80"))
    nr=[int(shared(list(rng.choice(cells,len(g),replace=False))).sum()) for _ in range(1000)]
    r={"cells":g,"n_all":int(m.sum()),"n_80":int(mr.sum()),"null_random_median":float(np.median(nr)),"null_random_p95":float(np.percentile(nr,95)),
       "p_random":float((np.sum(np.array(nr)>=m.sum())+1)/1001)}
    if all(FAM[c]=="epithelial" for c in g) and len(g)<len(epi):
        nl=[int(shared(list(rng.choice(epi,len(g),replace=False))).sum()) for _ in range(1000)]
        r.update({"null_lineage_median":float(np.median(nl)),"null_lineage_p95":float(np.percentile(nl,95)),"p_lineage":float((np.sum(np.array(nl)>=m.sum())+1)/1001)})
    out[k]=r; print(k,json.dumps({x:r[x] for x in r if x!="cells"}),flush=True)
neu=[c for c in cells if FAM[c]=="neural"]; mus=[c for c in cells if FAM[c]=="muscle"]
for nm,g in (("neural_pair",neu),("muscle_pair",mus)):
    m=shared(g); out[nm]={"cells":g,"n_all":int(m.sum())}; [rows.append((cg,nm)) for cg in P.index[m]]; print(nm,int(m.sum()),flush=True)
# the terminal 4-cell set relative to what neural and muscle cells share with each other WITHOUT excluding each other
g4=neu+mus; o=[i for i in range(P.shape[1]) if P.columns[i] not in g4]
out["terminal_cross_lineage"]={"lo_in_all_4":int(LO[:,[idx[c] for c in g4]].all(1).sum()),"and_methylated_elsewhere":out["terminal"]["n_all"]}
pd.DataFrame(rows,columns=["cpg","set"]).to_csv("class_shared_sets.csv",index=False); json.dump(out,open("class_shared.json","w"),indent=1); print("DONE",flush=True)
