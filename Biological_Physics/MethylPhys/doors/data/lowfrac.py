#!/usr/bin/env python3
"""DEV-LOWFRAC-01 (development, 2026-10-01). Neutrophil Met-A in whole blood at any neutrophil fraction.
Per array, through the chain's own Stage 1 and Stage A: f (EPIC NNLS), N (noise index, noise_sites_EPIC_v1), A_raw = mean H(x)/mean H(e) at the 6,000
neutrophil sites with e = sum f_g mu_g (no fraction cut-off), and A_dmg = the same after a known 2 % neutrophil blur
(x' = x + f*0.02*(0.5 - mu_neu)), which gives the sensitivity at this array's own fraction.
Arrays: GSE250556 (4 men x 16 technical/pooled replicates), GSE112618 (6, FACS), GSE179325 (574, COVID), Salas known mixtures (24, true fractions)."""
import os, sys, json, glob, tarfile, subprocess, numpy as np, pandas as pd, multiprocessing as mp
W=os.getcwd(); tarfile.open("chain_v3.tgz").extractall("bio/MethylPhys"); sys.path.insert(0,f"{W}/bio/MethylPhys/chain")
import conductor_v3 as C3
from stage_1_idat_calibration import calibrate_idat_to_beta
D="/home/ubuntu/data/neuttest/idat"; os.makedirs(D,exist_ok=True)
for g in ("GSE250556","GSE112618","GSE179325"):
    if not glob.glob(f"{D}/GSM*{ {'GSE250556':'7981','GSE112618':'30744','GSE179325':'5414'}[g] }*Grn*"):
        n=g[:-3]+"nnn"; subprocess.run(f"cd {D} && for a in 1 2 3 4; do curl -s -f -C - -o {g}.tar https://ftp.ncbi.nlm.nih.gov/geo/series/{n}/{g}/suppl/{g}_RAW.tar && break || sleep 30; done && tar -xf {g}.tar && rm -f {g}.tar",shell=True)
    print(g,"ready",flush=True)
INV=json.load(open("noise_sites_EPIC_v1.json"))["sites"]; B=C3._bc(); S=pd.Index(B["neutrophil_sites"])
P={g:pd.Series(v,index=S,dtype="float64") for g,v in B["profiles_at_neutrophil_sites"].items()}; mu=P["NEU"]
H=lambda b:-(b*np.log2(b)+(1-b)*np.log2(1-b))
def Hm(v): v=v.dropna().clip(1e-6,1-1e-6); return float(H(v).mean())
def read(b):
    b.index=b.index.astype(str); f=C3.stage_a_composition(b,"whole blood")["fractions"]; fn=f.get("NEU",0.0)
    x=b.reindex(S); e=sum(v*P[g] for g,v in f.items() if g in P); ok=x.notna()&e.notna()
    xd=x+fn*0.02*(0.5-mu)
    return dict(f_neu=fn,N=Hm(b.reindex(INV)),A_raw=Hm(x[ok])/Hm(e[ok]),A_dmg=Hm(xd[ok])/Hm(e[ok]),n_sites=int(ok.sum()))
X=pd.read_csv("neut_test_manifest.csv"); X=X[X.test.isin(["T1","T3","T4"])]
def one(r):
    p=f"{D}/{r.grn}"; p=p if os.path.exists(p) else p.replace(".idat.gz",".idat")
    try: b,_=calibrate_idat_to_beta(p,p.replace("_Grn","_Red"),verbose=False); return dict(gsm=r.gsm,test=r.test,group=r.group,slide=r.slide,**read(b))
    except Exception as e: return dict(gsm=r.gsm,test=r.test,group=r.group,err=str(e)[:100])
jobs=[r for _,r in X.iterrows()]
with mp.get_context("fork").Pool(100) as Pp: rows=Pp.map(one,jobs)
T=pd.read_csv("salas_mixture_truth.csv"); SH="/home/ubuntu/data/atlas_sources/blood/GSE110554/shards"
for _,r in T.iterrows():
    f=glob.glob(f"{SH}/{r.gsm}_*.parquet")
    if not f: continue
    b=pd.read_parquet(f[0]).iloc[:,0].astype("float64"); sc=100.0 if r[["cd4t","cd8t","bcell","nk","mono","neu"]].sum()>2 else 1.0
    rows.append(dict(gsm=r.gsm,test="MIX",group="known mixture",f_true=r.neu/sc,**read(b)))
R=pd.DataFrame(rows); R.to_csv("lowfrac_readings.csv",index=False)
print(R.groupby("test").agg(n=("gsm","size"),ok=("A_raw","count"),f=("f_neu","median"),N=("N","median")).round(3).to_string()); print("DONE")
