#!/usr/bin/env python3
"""Met-A v1.2 on whole blood (development, 2026-10-01). GSE88824: 8 donors with whole blood + purified neutrophils, monocytes, NK, B, CD4 T,
CD8 T (450K IDATs through our Stage 1). Leave-one-donor-out: everything for donor d (sites, floor, other cells' profiles) comes from the other 7.
Sites: purified-neutrophil across-donor SD <= 0.05, own mean in [lo,hi] (mirrored for unmethylated). Readings of donor d's whole blood:
 raw = per-site mean H of whole-blood beta at the sites / floor
 sep = the neutrophil beta separated first: (y - sum_k f_k mu_k)/f_neu, fractions by NNLS on the 6 purified profiles (other donors)
 shared = raw, on sites where every other purified blood cell sits within 0.05 of the neutrophil mean (mixture-insensitive sites)
Truth = donor d's own purified neutrophil array read on the same sites and floor."""
import os, re, glob, zlib, tarfile, urllib.request, time, numpy as np, pandas as pd, multiprocessing as mp
from scipy.optimize import nnls
t0=time.time(); W="/home/ubuntu/data/gse88824"; os.makedirs(f"{W}/idat",exist_ok=True); os.makedirs(f"{W}/shards",exist_ok=True)
u="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE88nnn/GSE88824/matrix/GSE88824_series_matrix.txt.gz"
t=zlib.decompress(urllib.request.urlopen(u,timeout=300).read(),16+zlib.MAX_WBITS).decode("utf-8","replace").split("!series_matrix_table_begin")[0]
row=lambda k: [x.strip('"') for x in re.search(rf"^!{k}\t(.+)$",t,re.M).group(1).split("\t")]
M=pd.DataFrame(dict(gsm=row("Sample_geo_accession"),title=row("Sample_title")))
for l in re.findall(r"^!Sample_characteristics_ch1\t.+$",t,re.M):
    for i,x in enumerate(l.split("\t")[1:]):
        x=x.strip('"')
        if ":" in x: k,v=x.split(":",1); M.loc[i,k.strip().lower()]=v.strip()
if not glob.glob(f"{W}/idat/*Grn.idat*"):
    tp=f"{W}/raw.tar"; urllib.request.urlretrieve("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE88nnn/GSE88824/suppl/GSE88824_RAW.tar",tp)
    with tarfile.open(tp) as tf: tf.extractall(f"{W}/idat")
    os.remove(tp)
files={}
for p in glob.glob(f"{W}/idat/*.idat*"): files.setdefault(os.path.basename(p).split("_")[0],[]).append(p)
def calib(g):
    sh=f"{W}/shards/{g}.parquet"
    if os.path.exists(sh): return g
    from stage_1_idat_calibration import calibrate_idat_to_beta
    grn=[p for p in files[g] if "_Grn" in p][0]; red=[p for p in files[g] if "_Red" in p][0]
    b,_=calibrate_idat_to_beta(grn,red,verbose=False); b=b.iloc[:,0] if hasattr(b,"columns") else b; b.to_frame(g).to_parquet(sh); return g
G=[g for g in M.gsm if g in files and len(files[g])==2]
with mp.get_context("fork").Pool(32) as P: P.map(calib,G)
print("calibrated",len(G),f"{time.time()-t0:.0f}s",flush=True)
B=pd.concat([pd.read_parquet(f"{W}/shards/{g}.parquet").iloc[:,0].rename(g) for g in G],axis=1).astype("float32")
K=dict(zip(M.gsm,M["cell type"])); P_=dict(zip(M.gsm,M.get("person id",pd.Series([""]*len(M)))))
CELLS=["Neutrophil","Monocyte","NKcell","CD19B","CD4T","CD8T"]
donors=sorted({P_[g] for g in G if K[g]=="Neutrophil"})
print("donors",donors,"kinds",pd.Series([K[g] for g in G]).value_counts().to_dict(),flush=True)
Hb=lambda x: -(x*np.log2(x)+(1-x)*np.log2(1-x))
def site_H(v): v=np.clip(v.dropna().values,1e-6,1-1e-6); return float(Hb(v).mean())
WIN=((0.75,0.95),(0.70,0.90))
res=[]
for d in donors:
    oth={c:[g for g in G if K[g]==c and P_[g]!=d] for c in CELLS}
    mu={c:B[oth[c]].mean(axis=1) for c in CELLS}; Mu=pd.concat(mu,axis=1).dropna()
    nsd=B[oth["Neutrophil"]].std(axis=1); nmu=mu["Neutrophil"]
    # fractions: NNLS on the 2000 most variable sites across the 6 profiles
    var=Mu.var(axis=1).sort_values(ascending=False).index[:2000]; prof=Mu.loc[var].values
    pn=[g for g in G if K[g]=="Neutrophil" and P_[g]==d]; wb=[g for g in G if K[g]=="WholeBlood" and P_[g]==d]
    if not pn or not wb: continue
    for lo,hi in WIN:
        win=((nmu>=lo)&(nmu<=hi))|((nmu>=1-hi)&(nmu<=1-lo)); sites=nsd[(nsd<=0.05)&win].index.intersection(Mu.index)
        others=Mu.loc[sites,[c for c in CELLS if c!="Neutrophil"]]; shared=sites[((others.sub(nmu.loc[sites],axis=0)).abs()<=0.05).all(axis=1)]
        fl=np.mean([site_H(B.loc[sites,g]) for g in oth["Neutrophil"]]); fls=np.mean([site_H(B.loc[shared,g]) for g in oth["Neutrophil"]]) if len(shared) else np.nan
        truth=site_H(B.loc[sites,pn[0]])/fl; truth_s=site_H(B.loc[shared,pn[0]])/fls if len(shared) else np.nan
        for w in wb:
            y=B.loc[var,w].values; ok=np.isfinite(y); f,_=nnls(prof[ok],y[ok]); f=f/f.sum(); fr=dict(zip(CELLS,f))
            ys=B.loc[sites,w]; bg=sum(fr[c]*Mu.loc[sites,c] for c in CELLS if c!="Neutrophil"); sep=((ys-bg)/max(fr["Neutrophil"],1e-3)).clip(0.001,0.999)
            res.append(dict(donor=d,window=f"{lo}-{hi}",n_sites=len(sites),n_shared=len(shared),f_neu=fr["Neutrophil"],truth=truth,A_raw=site_H(ys)/fl,A_sep=site_H(sep)/fl,
                            truth_shared=truth_s,A_shared=site_H(B.loc[shared,w])/fls if len(shared) else np.nan))
X=pd.DataFrame(res); X.to_csv("wb_neut_readings.csv",index=False)
for c in ("A_raw","A_sep"): X[c+"_err"]=X[c]-X.truth
X["A_shared_err"]=X.A_shared-X.truth_shared
pd.set_option("display.width",250); print(X.round(4).to_string(index=False))
inN=lambda a:float(((a>=0.95)&(a<=1.05)).mean())
for w,g in X.groupby("window"):
    print(w,"| truth in Normal %.2f"%inN(g.truth),"| raw: in Normal %.2f, |err|<=0.02 %.2f, median err %+.4f"%(inN(g.A_raw),(g.A_raw_err.abs()<=0.02).mean(),g.A_raw_err.median()),
          "| sep: in Normal %.2f, |err|<=0.02 %.2f, median err %+.4f"%(inN(g.A_sep),(g.A_sep_err.abs()<=0.02).mean(),g.A_sep_err.median()),
          "| shared: in Normal %.2f, |err|<=0.02 %.2f, median err %+.4f, sites %d"%(inN(g.A_shared),(g.A_shared_err.abs()<=0.02).mean(),g.A_shared_err.median(),g.n_shared.median()))
print("DONE",flush=True)
