#!/usr/bin/env python3
"""PROC-CHANNEL-01: each cell type's error budget from read-level WGBS (Loyfer 2023 .pat files, hg19), no labels, no floor.
A read is one DNA molecule; its pattern (C = methylated, T = unmethylated, . = not covered) shows which neighbouring CpGs agree.
Per sample (first 4,000,000 read-patterns of the file, streamed; the same genomic stretch for every sample):
  NEIGHBOUR AGREEMENT (the cell's 'error correction'): for every adjacent covered CpG pair on a read, P(same state); against chance
    from the pair's own marginals at that locus pair: phi = correlation of the two sites across reads (per pair, weighted), by beta bin.
  COPY-ERROR channel: in mostly-methylated reads (>= 6 CpGs, >= 80 % C), unmethylated sites that are ISOLATED (both neighbours C),
    per CpG.  ERASURE channel: unmethylated sites in RUNS of >= 3 inside mostly-methylated reads.
  DE NOVO channel: in mostly-unmethylated reads, isolated C per CpG; COHERENT GAIN: C runs >= 3.
  PDR: fraction of reads (>= 4 CpGs) carrying both states.
Then per cell (median of its samples) and: do cells group by channel without labels? (hierarchical clustering on the standardised
fingerprint, silhouette 2-10; stability on sample halves; ARI to the draft classes with a relabelling null)."""
import gzip, urllib.request, numpy as np, pandas as pd, json, time, multiprocessing as mp, collections
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
NREADS=10**9; t0=time.time()
S=pd.read_csv("roster_samples.csv"); RO=pd.read_csv("atlas_v2_roster.csv").set_index("cell")
adm=RO.index[RO.v2_status.str.startswith("IN")]; S=S[(S.source=="Loyfer2023")&(S.qc==True)&S.cell.isin(adm)]
PF={l.split("_")[0]:l.strip() for l in open("pat_files.txt")}
S=S[S["sample"].isin(PF)].groupby("cell").head(3).reset_index(drop=True)
def url(g): return f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/{PF[g]}"
import zlib, os
CACHE="/home/ubuntu/data/lines/loyfer_pat_head"; os.makedirs(CACHE,exist_ok=True)
CACHE="/home/ubuntu/data/lines/loyfer_pat_windows"; os.makedirs(CACHE,exist_ok=True)
WIN=[l.split() for l in open("windows_hg19_cpgidx.bed")]
def fetch(g):
    import pysam
    p=f"{CACHE}/{g}.pat.txt"
    if os.path.exists(p) and os.path.getsize(p)>1e5: return g,"cached"
    for attempt in range(8):
        try:
            u=url(g); tb=pysam.TabixFile(u,index=u+".csi")
            with open(p+".tmp","w") as f:
                for c,st,en in WIN:
                    for rec in tb.fetch(c,int(st),int(en)): f.write(rec+"\n")
            os.replace(p+".tmp",p); return g,"ok"
        except Exception as e: err=str(e)[:80]; time.sleep(30*(attempt+1))
    return g,"error:"+err
def lines(g):
    with open(f"{CACHE}/{g}.pat.txt","rb") as f:
        for l in f: yield l.rstrip(b"\n")
def one(g):
    try:
        pair=collections.defaultdict(lambda: np.zeros(4)); iso_T=runT=nC_m=iso_C=runC=nC_u=0; pdr_n=pdr_d=0; n=0
        for line in lines(g):
            n+=1
            if n>NREADS: break
            p=line.split(b"\t")
            if len(p)<4: continue
            start=int(p[1]); pat=p[2].decode(); cnt=int(p[3]); L=len(pat)
            for i in range(L-1):
                a,b=pat[i],pat[i+1]
                if a!="." and b!=".": pair[start+i][(a=="C")*2+(b=="C")]+=cnt
            cov=[c for c in pat if c!="."]
            if len(cov)>=4:
                pdr_d+=cnt; pdr_n+=cnt*(("C" in cov) and ("T" in cov))
            if len(cov)>=6:
                fC=cov.count("C")/len(cov); s_="".join(cov)
                if fC>=0.8:
                    nC_m+=cnt*len(s_); iso_T+=cnt*sum(1 for i in range(1,len(s_)-1) if s_[i]=="T" and s_[i-1]=="C" and s_[i+1]=="C")
                    runT+=cnt*sum(len(x) for x in s_.split("C") if len(x)>=3)
                elif fC<=0.2:
                    nC_u+=cnt*len(s_); iso_C+=cnt*sum(1 for i in range(1,len(s_)-1) if s_[i]=="C" and s_[i-1]=="T" and s_[i+1]=="T")
                    runC+=cnt*sum(len(x) for x in s_.split("T") if len(x)>=3)
        M=np.array(list(pair.values())); tot=M.sum(1); keep=tot>=10; M=M[keep]; tot=tot[keep]
        p00,p01,p10,p11=(M/tot[:,None]).T; pa=p10+p11; pb=p01+p11
        den=np.sqrt(pa*(1-pa)*pb*(1-pb)); ok=den>1e-9; phi=np.where(ok,(p11-pa*pb)/np.where(ok,den,1),np.nan)
        agree=p00+p11; chance=pa*pb+(1-pa)*(1-pb); bm=(pa+pb)/2
        return dict(sample=g,reads=n,pairs=int(keep.sum()),agree=float(np.average(agree,weights=tot)),agree_chance=float(np.average(chance,weights=tot)),
                 phi_mid=float(np.nanmean(phi[(bm>0.2)&(bm<0.8)])),phi_hi=float(np.nanmean(phi[(bm>=0.8)&(bm<0.97)])),phi_lo=float(np.nanmean(phi[(bm>0.03)&(bm<=0.2)])),
                 copy_err=iso_T/max(nC_m,1),erasure=runT/max(nC_m,1),denovo=iso_C/max(nC_u,1),coherent_gain=runC/max(nC_u,1),pdr=pdr_n/max(pdr_d,1))
    except Exception as e: return dict(sample=g,error=str(e)[:120])
with mp.get_context("fork").Pool(8) as P: FS=P.map(fetch,S["sample"].tolist())
print("fetched",dict(collections.Counter(x[1].split(":")[0] for x in FS)),f"{time.time()-t0:.0f}s",flush=True)
S=S[S["sample"].isin([g for g,st in FS if not st.startswith("error")])]
with mp.get_context("fork").Pool(64) as P: R=P.map(one,S["sample"].tolist())
D=pd.DataFrame(R).merge(S[["sample","cell"]],on="sample"); D["klass"]=D.cell.map(RO.class_by_draft_rule); D.to_csv("channel_samples.csv",index=False)
print("samples",len(D),"errors",int(D.get("error",pd.Series(dtype=str)).notna().sum()),f"{time.time()-t0:.0f}s",flush=True)
F=["agree","phi_mid","phi_hi","phi_lo","copy_err","erasure","denovo","coherent_gain","pdr"]
D=D.dropna(subset=[f for f in F if f in D.columns]) if all(f in D.columns for f in F) else D.iloc[0:0]
C=D.groupby(["cell","klass"])[F].median().reset_index(); C.to_csv("channel_cells.csv",index=False)
print(C.groupby("klass")[F].median().round(4).to_string())
F2=[f for f in F if C[f].std()>0 and C[f].notna().all()]; print("fingerprint features used:",F2,flush=True)
Z=((C[F2]-C[F2].mean())/C[F2].std()).values; lab=C.klass.values; rng=np.random.default_rng(1)
Zl=linkage(Z,"ward"); sil={k:float(silhouette_score(Z,fcluster(Zl,k,"maxclust"))) for k in range(2,11)}; kb=max(sil,key=sil.get); cl=fcluster(Zl,kb,"maxclust")
ari=adjusted_rand_score(lab,cl); null=[adjusted_rand_score(rng.permutation(lab),cl) for _ in range(5000)]
res=dict(cells=len(C),silhouette=sil,best_k=int(kb),ARI=float(ari),ARI_null_p95=float(np.quantile(null,0.95)),p=float(np.mean(np.array(null)>=ari)))
print(json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in res.items() if k!="silhouette"}),{k:round(v,3) for k,v in sil.items()})
C["cluster"]=cl; print(C.sort_values("cluster")[["cell","klass","cluster","copy_err","erasure","phi_hi","phi_mid","pdr"]].round(4).to_string(index=False))
# held-out stability: fingerprint from each cell's first sample vs the median of its other samples, clustered separately
two=D.groupby("cell").filter(lambda d: len(d)>=2)
FA=two.groupby("cell").head(1).set_index("cell")[F2]; FB=two.groupby("cell").apply(lambda d: d.iloc[1:][F2].median())
FA=FA.loc[FB.index]; mu=pd.concat([FA,FB]).mean(); sd=pd.concat([FA,FB]).std()
ZA=((FA-mu)/sd).values; ZB=((FB-mu)/sd).values; stab={}
for k in (2,3,4,5,6,8,10):
    ca=fcluster(linkage(ZA,"ward"),k,"maxclust"); cb=fcluster(linkage(ZB,"ward"),k,"maxclust"); stab[k]=float(adjusted_rand_score(ca,cb))
res["stability_ARI_by_k"]=stab; res["stability_cells"]=int(len(FA)); print("held-out stability ARI by k:",{k:round(v,3) for k,v in stab.items()})
# k=8 grouping against the draft classes
c8=fcluster(Zl,8,"maxclust"); res["ARI_k8"]=float(adjusted_rand_score(lab,c8)); res["ARI_k8_null_p95"]=float(np.quantile([adjusted_rand_score(rng.permutation(lab),c8) for _ in range(5000)],0.95))
print("k=8 vs draft classes ARI %.3f (null p95 %.3f)"%(res["ARI_k8"],res["ARI_k8_null_p95"]))
json.dump(res,open("channel_summary.json","w"),indent=1); print("DONE",f"{time.time()-t0:.0f}s")
