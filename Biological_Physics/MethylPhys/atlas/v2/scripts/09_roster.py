#!/usr/bin/env python3
"""Atlas v2 candidate cell roster (2026-09-28). Every candidate from Loyfer 2023, Moss 2018, Salas 2018/2022, Tian 2023 gets:
samples passing QC per source (arrays: call rate >= 0.93; WGBS: >= 10 reads at >= 90 % of array CpGs), total independent
samples, and the entry-rule verdict (>= 2 samples, or 'pooled' for Tian's 3-donor pseudobulk). Then the chain's own twin rule
(twin_family_thresholds_v1.json: merge only if r > twin_r AND < min_separating_loci CpGs differ by > sep_delta) is applied between
every pair of admitted cells, on profiles from ONE platform where both exist (Loyfer WGBS), else after the measured platform
transfer (WGBS = a + b*array, fitted per pair on shared CpGs). Writes roster.csv, twins.csv, roster_summary.json."""
import json, os, re, itertools, numpy as np, pandas as pd
R="/home/ubuntu/data/atlas_sources"; TW=json.load(open("twin_family_thresholds_v1.json"))
TR, SL, SD = TW["twin_r"], TW["min_separating_loci"], TW["sep_delta"]
# ---------- canonical names
def canon_loyfer(ct,tis):
    base={"Epithelium":f"{tis} epithelium","Endothelium":f"{tis} endothelium","Macrophages":f"{tis} macrophages","Fibroblast":f"{tis} fibroblasts",
          "Smooth muscle":"smooth muscle","Endocrine":f"{tis} enteroendocrine","Basal epithelial":"breast basal epithelium","Luminal epithelial":"breast luminal epithelium",
          "Granulocytes":"granulocytes","Acinar":"pancreatic acinar","Duct":"pancreatic duct","Alpha":"pancreatic alpha","Beta":"pancreatic beta","Delta":"pancreatic delta",
          "Neuronal":"cortical neurons","T (CD3+) cells":"T cells CD3","T helper(CD4+) cells":"CD4 T cells","T cytotoxic (CD8+) cells":"CD8 T cells",
          "Naive T cells CD4":"naive CD4 T cells","Naive T cells CD8":"naive CD8 T cells","Memory B cells":"memory B cells","NK":"NK cells","B cells":"B cells","Monocytes":"monocytes"}
    n=base.get(ct, ct).lower()
    return {"t effector memory cd8":"effector memory cd8 t cells","naive cd4 t cells":"naive cd4 t cells"}.get(n,n)
SALAS={"Bcell":"B cells","CD4T":"CD4 T cells","CD8T":"CD8 T cells","Mono":"monocytes","NK":"NK cells","Neu":"neutrophils",
 "Human Peripheral Blood CD4+ CD45RA+ T Cells, Frozen":"naive CD4 T cells","Human Peripheral Blood CD4+ CD45RO+ T Cells, Frozen":"memory CD4 T cells",
 "Human Peripheral Blood CD8+ CD45RA+ T Cells, Frozen":"naive CD8 T cells","Human Peripheral Blood CD8+ CD62- CD45RO+ Effector Memory T Cells, Frozen":"effector memory CD8 T cells",
 "Human Peripheral blood CD14+ Monocytes, Frozen":"monocytes","Human Peripheral blood CD15+ CD16- Eosinophils, Frozen":"eosinophils",
 "Human Peripheral blood CD4+ CD25+ CD127- T regulatory lymphocytes, Frozen":"regulatory T cells","Human Peripheral blood CD56+ Natural killers, Frozen":"NK cells",
 "Human Peripheral blood negatively selected neutrophils, Frozen":"neutrophils","Human Peripheral blood, CD19+ CD27+ B cells, Frozen":"memory B cells",
 "Human Peripheral blood, CD19+ CD27- B cells, Frozen":"naive B cells","Human peripheral blood IgE+ CD123+ Basophils, Frozen":"basophils"}
MOSS={"adipocytes":"adipocytes","colon epithelial cells":"colon epithelium","cortical neurons":"cortical neurons","hepatocytes":"hepatocyte",
      "lung epithelial cells":"lung alveolar epithelium","pancreatic acinar cells":"pancreatic acinar","pancreatic beta cells":"pancreatic beta",
      "pancreatic duct cells":"pancreatic duct","vascular endothelial cells":"vascular endothelium"}
TIAN={"ASC":"astrocytes","MGC":"microglia","ODC":"oligodendrocytes","OPC":"oligodendrocyte precursors","VLMC":"vascular leptomeningeal cells","PC":"pericytes","EC":"brain endothelium"}
# ---------- samples per source
rows=[]
L=json.load(open("loyfer_extract_report.json"))["per_sample"]
for g,v in L.items():
    rows.append(dict(cell=canon_loyfer(v["cell_type"],v["tissue"]),source="Loyfer2023",platform="WGBS",sample=g,qc=v["frac_depth_ge10"]>=0.90,metric=v["frac_depth_ge10"]))
M=pd.read_csv("moss_manifest.csv")
for _,r in M.iterrows():
    if r.label in MOSS: rows.append(dict(cell=MOSS[r.label],source="Moss2018",platform="array",sample=r.base,qc=r.call_rate>=0.93,metric=r.call_rate))
S=pd.read_csv("blood_manifest.csv")
for _,r in S.iterrows():
    if r.label in SALAS: rows.append(dict(cell=SALAS[r.label],source="Salas"+("2018" if r.gse=="GSE110554" else "2022"),platform="array",sample=r.base,qc=r.call_rate>=0.93,metric=r.call_rate))
T=json.load(open("tian_extract_report.json"))["types"]
for t,v in T.items(): rows.append(dict(cell=TIAN[t],source="Tian2023",platform="WGBS-pooled",sample=t,qc=v["frac_cov_ge10"]>=0.90,metric=v["frac_cov_ge10"]))
for acc,line in (("ENCFF918PML","H1"),("ENCFF770UYJ","HUES64")):
    rows.append(dict(cell="embryonic stem cells",source="ENCODE",platform="WGBS",sample=f"{R}/encode_stem/{line}_{acc}_array.parquet",qc=True,metric=None))
D=pd.DataFrame(rows); D["cell"]=D.cell.str.lower(); D.to_csv("roster_samples.csv",index=False)
ok=D[D.qc]
g=ok.groupby("cell").agg(samples=("sample","size"),sources=("source",lambda s:",".join(sorted(set(s)))),platforms=("platform",lambda s:",".join(sorted(set(s)))))
allc=D.groupby("cell").agg(candidates=("sample","size"))
RO=allc.join(g,how="left").fillna({"samples":0,"sources":"","platforms":""})
RO["pooled_only"]=RO.platforms.eq("WGBS-pooled")
RO["entry"]=np.where(RO.samples>=2,"ADMIT",np.where(RO.pooled_only&(RO.samples>=1),"ADMIT_POOLED_FLAGGED","WAIT (fewer than 2 samples)"))
print(RO.entry.value_counts().to_dict(),flush=True)
# ---------- profiles: one per (cell, platform-group); WGBS profiles preferred for twin tests
B=pd.read_parquet(f"{R}/loyfer2023/array_beta.parquet"); C=pd.read_parquet(f"{R}/loyfer2023/array_cov.parquet")
prof={}
for cell,grp in ok.groupby("cell"):
    lw=grp[grp.source=="Loyfer2023"]["sample"].tolist()
    if lw: prof[cell]=("WGBS",B[lw].where(C[lw]>=10).mean(axis=1)); continue
    en=grp[grp.source=="ENCODE"]["sample"].tolist()
    if en:
        prof[cell]=("WGBS",pd.concat([pd.read_parquet(x).pipe(lambda d:d["beta"].where(d["cov"]>=10)) for x in en],axis=1).mean(axis=1)); continue
    tw=grp[grp.source=="Tian2023"]["sample"].tolist()
    if tw:
        d=pd.read_parquet(f"{R}/tian2023/{tw[0]}_array.parquet"); prof[cell]=("WGBS",d["beta"].where(d["cov"]>=10)); continue
    arr=[]
    for _,r in grp.iterrows():
        p=f"{R}/moss2018/shards/{r['sample']}.parquet" if r.source=="Moss2018" else next((q for q in (f"{R}/blood/GSE110554/shards/{r['sample']}.parquet",f"{R}/blood/GSE167998/shards/{r['sample']}.parquet") if os.path.exists(q)),None)
        if p and os.path.exists(p): arr.append(pd.read_parquet(p).iloc[:,0])
    if arr: prof[cell]=("array",pd.concat(arr,axis=1).mean(axis=1))
admitted=[c for c in RO.index if RO.loc[c,"entry"].startswith("ADMIT") and c in prof]
null=[]
for cell,grp in ok.groupby("cell"):
    lw=grp[grp.source=="Loyfer2023"]["sample"].tolist()
    if len(lw)>=4:
        h1,h2=lw[:len(lw)//2],lw[len(lw)//2:]; u=B[h1].where(C[h1]>=10).mean(axis=1); v=B[h2].where(C[h2]>=10).mean(axis=1)
    else:
        arr=[x for x in grp["sample"] if grp.set_index("sample").loc[x,"platform"]=="array"]
        if len(arr)<4: continue
        def ld(x):
            r=grp.set_index("sample").loc[x]; sub={"Moss2018":"moss2018/shards","Salas2018":"blood/GSE110554/shards","Salas2022":"blood/GSE167998/shards"}[r.source]
            return pd.read_parquet(f"{R}/{sub}/{x}.parquet").iloc[:,0]
        u=pd.concat([ld(x) for x in arr[:len(arr)//2]],axis=1).mean(axis=1); v=pd.concat([ld(x) for x in arr[len(arr)//2:]],axis=1).mean(axis=1)
    j=u.dropna().index.intersection(v.dropna().index)
    null.append(dict(cell=cell,shared=len(j),r=float(np.corrcoef(u[j],v[j])[0,1]),separating=int((np.abs(u[j]-v[j])>SD).sum()),frac=float((np.abs(u[j]-v[j])>SD).mean())))
NU=pd.DataFrame(null); NU.to_csv("twin_null_splithalf.csv",index=False)
NULL_FRAC=float(NU.frac.quantile(0.95)); NULL_R=float(NU.r.quantile(0.05))
print(f"split-half null over {len(NU)} cells: separating fraction median {NU.frac.median():.5f}, 95th pct {NULL_FRAC:.5f}; r 5th pct {NULL_R:.4f}",flush=True)
print("admitted with a profile:",len(admitted),flush=True)
tw=[]
for a,b in itertools.combinations(admitted,2):
    (pa,xa),(pb,xb)=prof[a],prof[b]
    j=xa.dropna().index.intersection(xb.dropna().index)
    if len(j)<1000: continue
    u,v=xa.loc[j].values,xb.loc[j].values
    if pa!=pb:   # put the array profile on the WGBS scale with a per-pair linear transfer
        if pa=="array": s,i=np.polyfit(u,v,1); u=i+s*u
        else: s,i=np.polyfit(v,u,1); v=i+s*v
    r=float(np.corrcoef(u,v)[0,1]); sep=int((np.abs(u-v)>SD).sum())
    frac=sep/len(j)
    # twin = the two cells separate no more than a cell separates from ITSELF (95th pct of the split-half null), with the chain's r bar
    tw.append(dict(a=a,b=b,platforms=f"{pa}/{pb}",shared=len(j),r=round(r,4),separating=sep,sep_frac=round(frac,5),
                   excess_over_null=round(frac/max(NULL_FRAC,1e-9),2),twin=(r>TR and frac<=NULL_FRAC*1.5)))
TWN=pd.DataFrame(tw).sort_values("r",ascending=False); TWN.to_csv("twins.csv",index=False)
twins=TWN[TWN.twin]
# merge twin families (union-find)
par={c:c for c in admitted}
def f(x):
    while par[x]!=x: x=par[x]
    return x
for _,r in twins.iterrows(): par[f(r.a)]=f(r.b)
fam={}
for c in admitted: fam.setdefault(f(c),[]).append(c)
RO["twin_family"]=[ "+".join(sorted(fam[f(c)])) if c in par and len(fam[f(c)])>1 else "" for c in RO.index]
RO.to_csv("roster.csv")
summ={"candidates":int(len(RO)),"admit":int((RO.entry=="ADMIT").sum()),"admit_pooled":int((RO.entry=="ADMIT_POOLED_FLAGGED").sum()),
      "wait":int(RO.entry.str.startswith("WAIT").sum()),"twin_pairs":int(len(twins)),"cells_after_twin_merge":int(len(fam)),
      "families":{k:v for k,v in fam.items() if len(v)>1},"null_sep_frac_p95":NULL_FRAC,"null_r_p05":NULL_R,"closest_non_twins":TWN[~TWN.twin].head(12).to_dict("records"),"rule":TW}
json.dump(summ,open("roster_summary.json","w"),indent=1,default=str)
print("SUMMARY",json.dumps({k:summ[k] for k in ("candidates","admit","admit_pooled","wait","twin_pairs","cells_after_twin_merge")}),flush=True)
