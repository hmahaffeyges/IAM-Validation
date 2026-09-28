#!/usr/bin/env python3
"""EXPLORATORY (author's question, 2026-09-28): do cells of one architecture class share a pattern in the genome that one would notice
WITHOUT knowing H_min? Written before any result is read. No floor, no A, no class prior is used anywhere below.
Data: Loyfer 2023 only (one platform, no source term), the 56 atlas-v2 cells it holds; per-cell mean over its samples at array CpGs
with depth >= 10 in every sample used.
Competing labels, so a class pattern cannot be mistaken for a better-known one:
  class (draft rule) | cell family (epithelial/endothelial/mesenchymal/muscle/neural/immune/erythroid) | germ layer | organ system.
Tests:
 T1 nearest neighbour (1 - Pearson on the 20,000 most variable CpGs): fraction of cells whose nearest cell shares each label; p from
    10,000 label permutations.
 T2 the contrasts where class and the other labels DISAGREE:
    a) epithelia only (cycling vs secretory): does the nearest epithelial cell share class, or organ system?
    b) terminal (neurons, oligodendrocytes, cardiomyocytes, striated muscle): is each one's nearest cell terminal, or its own germ layer?
 T3 where their specific CpGs sit: per cell, CpGs hypomethylated in it (beta < 0.3) and methylated in >= 90 % of the other cells
    (beta > 0.7). Jaccard overlap of those sets, same class vs different class, stratified by cell family. Genomic context of each
    class's union: fraction within 1.5 kb of a TSS, in a gene body, or distal (HM450 hg38 GENCODE v36 manifest).
Nothing here enters the chain."""
import json, numpy as np, pandas as pd, itertools
R="/home/ubuntu/data/atlas_sources"; rng=np.random.default_rng(20260928)
RO=pd.read_csv("atlas_v2_roster.csv",index_col=0); S=pd.read_csv("roster_samples.csv")
S=S[(S.source=="Loyfer2023")&S.qc]; cells=sorted(set(S.cell)&set(RO.index[RO.v2_status.str.startswith("IN")]))
CLS=RO.class_by_draft_rule.to_dict()
FAM={}; GERM={}; ORG={}
for c in cells:
    n=c
    if any(k in n for k in ("cd4","cd8","nk cells","b cells","monocytes","macrophages")): FAM[c]="immune"
    elif "erythro" in n: FAM[c]="erythroid"
    elif "endothel" in n: FAM[c]="endothelial"
    elif any(k in n for k in ("fibroblast","smooth muscle","adipocyte")): FAM[c]="mesenchymal"
    elif any(k in n for k in ("cardiomyocyte","striated")): FAM[c]="muscle"
    elif any(k in n for k in ("neuron","oligodendro")): FAM[c]="neural"
    else: FAM[c]="epithelial"
    if FAM[c] in ("immune","erythroid","endothelial","mesenchymal","muscle") or any(k in n for k in ("kidney","fallop","endometr")): GERM[c]="mesoderm"
    elif FAM[c]=="neural" or "breast" in n: GERM[c]="ectoderm"
    elif any(k in n for k in ("tongue","tonsil")): GERM[c]="ambiguous"
    else: GERM[c]="endoderm"
    ORG[c]=next((o for o,ks in {"GI":("colon","intestine","gastric"),"pancreas":("pancrea",),"liver":("hepato",),"lung":("lung",),
        "kidney":("kidney",),"oral":("tongue","tonsil"),"breast":("breast",),"reproductive":("fallop","endometr","prostate"),
        "bladder":("bladder",),"thyroid":("thyroid",),"heart":("heart","cardio","aorta"),"brain":("neuron","oligo"),
        "blood":("cd4","cd8","nk ","b cells","monocyte","erythro","naive","effector","central"),"vessel":("saphenous",),
        "muscle":("smooth","striated"),"fat":("adipo",)}.items() if any(k in n for k in ks)),"other")
B=pd.read_parquet(f"{R}/loyfer2023/array_beta.parquet"); C=pd.read_parquet(f"{R}/loyfer2023/array_cov.parquet")
P={}
for c in cells:
    g=S[S.cell==c]["sample"].tolist(); P[c]=B[g].where(C[g]>=10).mean(axis=1,skipna=False)
P=pd.DataFrame(P).dropna(); print("cells",len(cells),"| CpGs covered in every sample",len(P),flush=True)
V=P.loc[P.var(axis=1).sort_values().index[-20000:]]; D=1-np.corrcoef(V.T.values); np.fill_diagonal(D,np.inf)
nn=D.argmin(1); names=list(V.columns)
def nn_agree(lab,idx=None):
    idx=list(range(len(names))) if idx is None else idx; sub=D[np.ix_(idx,idx)]; n2=sub.argmin(1)
    L=np.array([lab[names[i]] for i in idx]); ok=L!="ambiguous"; obs=float(np.mean((L==L[n2])[ok]))
    null=[np.mean(((Lp:=rng.permutation(L))==Lp[n2])[ok]) for _ in range(10000)]
    return {"agree":round(obs,3),"null_mean":round(float(np.mean(null)),3),"p":float((np.sum(np.array(null)>=obs)+1)/10001),"n":int(ok.sum())}
out={"cells":len(cells),"cpgs":len(P),"labels":{c:[CLS[c],FAM[c],GERM[c],ORG[c]] for c in cells}}
out["T1"]={k:nn_agree(l) for k,l in (("class",CLS),("cell_family",FAM),("germ_layer",GERM),("organ",ORG))}
out["T1_nearest"]={names[i]:names[nn[i]] for i in range(len(names))}
epi=[i for i,c in enumerate(names) if CLS[c] in ("cycling","secretory") and FAM[c]=="epithelial"]
out["T2a_epithelia"]={k:nn_agree(l,epi) for k,l in (("class",CLS),("organ",ORG),("germ_layer",GERM))}
term=[i for i,c in enumerate(names) if CLS[c]=="terminal"]
out["T2b_terminal"]={names[i]:{"nearest":names[nn[i]],"nearest_class":CLS[names[nn[i]]],
    "nearest_terminal":names[min((j for j in term if j!=i),key=lambda j:D[i,j])],"d_nearest":round(float(D[i,nn[i]]),4),
    "d_nearest_terminal":round(float(min(D[i,j] for j in term if j!=i)),4),
    "rank_of_nearest_terminal":int(1+sum(D[i,j]<min(D[i,k] for k in term if k!=i) for j in range(len(names)) if j!=i))} for i in term}
# T3
spec={}
for c in names:
    others=P.drop(columns=c); spec[c]=set(P.index[(P[c]<0.3)&((others>0.7).mean(axis=1)>=0.9)])
J=[]
for a,b in itertools.combinations(names,2):
    u=len(spec[a]|spec[b]); J.append(dict(a=a,b=b,same_class=CLS[a]==CLS[b],same_family=FAM[a]==FAM[b],jaccard=(len(spec[a]&spec[b])/u) if u else 0.0))
J=pd.DataFrame(J); J.to_csv("specific_overlap.csv",index=False)
out["T3_specific_counts"]={c:len(s) for c,s in spec.items()}
out["T3_jaccard"]=J.groupby(["same_family","same_class"]).jaccard.agg(["size","median","mean"]).round(4).reset_index().to_dict("records")
try:
    A=pd.read_csv("/home/ubuntu/data/HM450.hg38.manifest.gencode.v36.tsv.gz",sep="\t",usecols=["probeID","genesUniq","distToTSS"],low_memory=False).set_index("probeID")
    def ctx(p):
        r=A.loc[p] if p in A.index else None
        if r is None or pd.isna(r.distToTSS): return "distal"
        try: d=min(abs(float(v)) for v in str(r.distToTSS).split(";") if v not in ("","NA"))
        except ValueError: return "distal"
        return "promoter" if d<=1500 else ("gene body/near" if d<=50000 else "distal")
    bg=pd.Series([ctx(p) for p in rng.choice(P.index,20000,replace=False)]).value_counts(normalize=True).round(3).to_dict()
    out["T3_context_background"]=bg; out["T3_context_by_class"]={}
    for k in sorted(set(CLS[c] for c in names)):
        U=set().union(*[spec[c] for c in names if CLS[c]==k]); U=list(U)[:20000]
        out["T3_context_by_class"][k]={"n":len(U),**pd.Series([ctx(p) for p in U]).value_counts(normalize=True).round(3).to_dict()} if U else {"n":0}
except Exception as e: out["T3_context_error"]=str(e)[:200]
json.dump(out,open("class_structure.json","w"),indent=1)
for k in ("T1","T2a_epithelia","T2b_terminal","T3_jaccard","T3_context_background","T3_context_by_class"): print(k,json.dumps(out.get(k)),flush=True)
print("DONE",flush=True)
