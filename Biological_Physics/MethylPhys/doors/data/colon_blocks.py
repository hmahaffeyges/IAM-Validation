#!/usr/bin/env python3
"""Colon-epithelium marker blocks (development, 2026-10-01). Blocks where colon epithelium is METHYLATED and blood cells, colon fibroblasts and
colon macrophages are UNMETHYLATED. In these blocks a qualifying molecule (>= 6 CpGs, >= 80 % methylated: the IAM-A rule) is a colon-epithelial
molecule, so copy error read there is the colon cells' own.
1. hg19 CpG index (wgbstools order) from the hg19 fasta; checked against probe_map_summary (probe MAPINFO -> cpg_index).
2. Loyfer .beta (hg19, uint8 meth/cov pairs): group means with coverage >= 10 per sample.
3. Marker CpG: colon epithelium (left+right, 5 samples) mean >= 0.80 and every other group mean <= 0.10. Block: >= 4 marker CpGs, gaps <= 100 bp.
4. Lift to GRCh38 (UCSC hg19ToHg38 chain)."""
import gzip, os, re, glob, json, numpy as np, pandas as pd, subprocess
L="/home/ubuntu/data/atlas_sources/loyfer2023/beta"
# --- 1. CpG index
import pysam
fa=pysam.FastaFile("/home/ubuntu/data/hg19.bgz.fa.gz"); refs=fa.references
order=[f"chr{i}" for i in range(1,23)]+["chrX","chrY","chrM"]
order=[c for c in order if c in refs]
chrs=[];poss=[]
for c in order:
    s=fa.fetch(c).upper(); p=np.array([m.start() for m in re.finditer("CG",s)],dtype=np.int64)+1   # 1-based C position
    chrs.append(np.full(len(p),order.index(c),dtype=np.int8)); poss.append(p)
CH=np.concatenate(chrs); PO=np.concatenate(poss); print("CpGs",len(PO),flush=True)
M=pd.read_csv("probe_map_summary.csv.gz").dropna(subset=["cpg_index"]); M=M[M.chr.isin(order)]
idx=M.cpg_index.astype(np.int64).values-1; ok=idx<len(PO)
chk=(CH[idx[ok]]==M.chr[ok].map(order.index).values)&(np.abs(PO[idx[ok]]-M.MAPINFO[ok].values)<=1)
print("index check: %d/%d probes agree"%(chk.sum(),ok.sum()),flush=True)
assert chk.mean()>0.99, "CpG index order does not match"
# --- 2. group means
fs=sorted(glob.glob(f"{L}/*.beta")); grp=lambda f:re.sub(r"-Z[0-9A-Z]+\.beta$","",os.path.basename(f).split("_",1)[1])
G={}
for f in fs:
    g=grp(f)
    if g.startswith("Colon-Left-Epithelial") or g.startswith("Colon-Right-Epithelial"): k="COLON_EPI"
    elif g.startswith("Blood-") or g in ("Colon-Fibroblasts","Colon-Macrophages"): k=g
    else: continue
    a=np.fromfile(f,dtype=np.uint8).reshape(-1,2); b=np.where(a[:,1]>=10,a[:,0]/np.maximum(a[:,1],1),np.nan).astype(np.float32)
    G.setdefault(k,[]).append(b)
print({k:len(v) for k,v in G.items()},flush=True)
mean={k:np.nanmean(np.vstack(v),0) for k,v in G.items()}
col=mean.pop("COLON_EPI"); oth=np.vstack(list(mean.values())); othmax=np.nanmax(oth,0); nother=np.sum(~np.isnan(oth),0)
def blocks_of(mk,minc):
    ii=np.where(mk)[0]; bl=[]; cur=[ii[0]]
    for j in ii[1:]:
        if CH[j]==CH[cur[-1]] and PO[j]-PO[cur[-1]]<=100: cur.append(j)
        else:
            if len(cur)>=minc: bl.append(cur)
            cur=[j]
    if len(cur)>=minc: bl.append(cur)
    return bl
okn=nother>=len(mean)-2
sets={"strict":((col>=0.80)&(othmax<=0.10)&okn,4),"loose":((col>=0.70)&(othmax<=0.20)&okn,3)}
Bs=[]
for name,(mk,minc) in sets.items():
    bl=blocks_of(mk,minc); print(name,"marker CpGs",int(mk.sum()),"blocks",len(bl),flush=True)
    Bs.append(pd.DataFrame([dict(set=name,chr=order[CH[b[0]]],start=int(PO[b[0]]),end=int(PO[b[-1]])+1,n_cpg=len(b),colon_beta=float(np.nanmean(col[b])),other_max=float(np.nanmax(othmax[b]))) for b in bl]))
B=pd.concat(Bs,ignore_index=True)
print(B.groupby("set").n_cpg.agg(["size","sum"]),flush=True)
# --- 4. liftover
subprocess.run("/home/ubuntu/data/tumour/mm/bin/pip -q install pyliftover 2>/dev/null; curl -sL -o hg19ToHg38.over.chain.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz; file hg19ToHg38.over.chain.gz; ls -la hg19ToHg38.over.chain.gz",shell=True)
from pyliftover import LiftOver
lo=LiftOver("hg19ToHg38.over.chain.gz")
def lift(c,p):
    r=lo.convert_coordinate(c,p-1); return (r[0][0],r[0][1]+1) if r else (None,None)
s38=[lift(r.chr,r.start) for r in B.itertuples()]; e38=[lift(r.chr,r.end) for r in B.itertuples()]
B["chr38"]=[a[0] for a in s38]; B["start38"]=[a[1] for a in s38]; B["end38"]=[b[1] for b in e38]
B=B[(B.chr38.notna())&(B.chr38==[b[0] for b in e38])&(B.end38>B.start38)&((B.end38-B.start38)<2*(B.end-B.start)+50)]
B.to_csv("colon_epi_M_blocks.csv",index=False); print("lifted",len(B),"| median CpGs",B.n_cpg.median(),"| bp",int((B.end38-B.start38).sum()),flush=True); print("DONE")
