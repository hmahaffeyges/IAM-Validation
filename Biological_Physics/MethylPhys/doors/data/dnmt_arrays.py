#!/usr/bin/env python3
"""PROC-DNMT-01 Part A, as pre-registered (sha 2256dda819e6da7f). GSE135205 raw IDATs -> chain Stage 1 -> per line: canon site rule on the
line's 4 DMSO arrays -> Met-A per array (DMSO leave-one-out), per channel, noise index N, C-score (genome-order clustering of the residual)."""
import os, sys, glob, json, tarfile, subprocess, numpy as np, pandas as pd, multiprocessing as mp
W=os.getcwd(); tarfile.open("chain_v3.tgz").extractall("bio/MethylPhys"); sys.path.insert(0,f"{W}/bio/MethylPhys/chain")
import conductor_v3 as C3
from stage_1_idat_calibration import calibrate_idat_to_beta
D="/home/ubuntu/data/dnmt/idat"; os.makedirs(D,exist_ok=True)
if not glob.glob(f"{D}/*Grn.idat*"):
    subprocess.run(f"cd {D} && for a in 1 2 3 4; do curl -s -f -C - -o r.tar https://ftp.ncbi.nlm.nih.gov/geo/series/GSE135nnn/GSE135205/suppl/GSE135205_RAW.tar && break || sleep 30; done && tar -xf r.tar && rm -f r.tar",shell=True,check=True)
X=pd.read_csv("gse135205_design.csv"); INV=json.load(open("noise_sites_EPIC_v1.json"))["sites"]
H=lambda b:-(b*np.log2(b)+(1-b)*np.log2(1-b))
def beta(g):
    f=glob.glob(f"{D}/{g}_*Grn.idat*")[0]; b,_=calibrate_idat_to_beta(f,f.replace("_Grn","_Red"),verbose=False); b.index=b.index.astype(str); return g,b
with mp.get_context("fork").Pool(51) as P: Bt=dict(P.map(beta,X.gsm.tolist()))
M=pd.DataFrame(Bt).dropna(how="all"); print("arrays",M.shape,flush=True)
man=pd.read_csv("/home/ubuntu/data/EPIC.hg38.manifest.gencode.v36.tsv.gz",sep="\t",usecols=["probeID","CpG_chrm","CpG_beg"]).dropna()
man["chr"]=man.CpG_chrm.str.replace("chr","").replace({"X":"23","Y":"24"}); man=man[man.chr.str.isnumeric()]; man["key"]=man.chr.astype(int)*1e10+man.CpG_beg
ORDER=man.sort_values("key").probeID.astype(str).tolist()
rows=[]
for line,G in X.groupby("line"):
    dm=G[G.cmpd=="DMSO"].gsm.tolist()
    def sites(refs):
        R=M[refs].clip(1e-6,1-1e-6); mu=R.mean(1); sd=R.std(1); ok=sd<=0.05
        hi=sd[ok&(mu>=0.75)&(mu<=0.95)].nsmallest(3000).index; lo=sd[ok&(mu>=0.05)&(mu<=0.25)].nsmallest(3000).index
        return hi,lo
    for _,r in G.iterrows():
        refs=[g for g in dm if g!=r.gsm]; hi,lo=sites(refs); S=hi.union(lo)
        R=M.loc[S,refs].clip(1e-6,1-1e-6); x=M.loc[S,r.gsm].clip(1e-6,1-1e-6)
        fl=float(H(R).mean().mean()); A=float(H(x).mean()/fl)
        Ah=float(H(x[hi]).mean()/H(R.loc[hi]).mean().mean()); Al=float(H(x[lo]).mean()/H(R.loc[lo]).mean().mean())
        z=(H(x)-H(R).mean(1))/H(R).std(1).clip(lower=0.01); zo=z.reindex([s for s in ORDER if s in z.index]).dropna()
        rows.append(dict(gsm=r.gsm,line=line,cmpd=r.cmpd,dose_nM=r.dose_nM,day=r.day,A=A,A_meth=Ah,A_unmeth=Al,beta_meth_median=float(x[hi].median()),
                         n_sites=len(S),N=float(H(M[r.gsm].reindex(INV).dropna().clip(1e-6,1-1e-6)).mean()),clust=C3._clustering(zo)))
R=pd.DataFrame(rows); R.to_csv("dnmt_arrays_readings.csv",index=False)
pd.set_option("display.width",220); print(R.sort_values(["line","cmpd","day","dose_nM"]).round(4).to_string(index=False)); print("DONE")
