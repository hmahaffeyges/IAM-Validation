#!/usr/bin/env python3
"""IMR90 per-channel error readings on the current method (2026-10-02).
GSE48580 WGBS (Cruickshanks 2013): proliferating x3, replicative senescent x3, SV40 x3; cached parquets from PROC-LINES-02.
Reference = the same cells proliferating (leave-one-out for proliferating replicates).
Sites: beta in 0.75-0.95 (methylated channel) or 0.05-0.25 (unmethylated channel) on a proliferating replicate used ONLY for selection;
the reference H comes from a different proliferating replicate (with 3 replicates an SD rule cannot be applied independently).
Reading per channel = mean over that channel's sites of depth-corrected H(beta) / the same in each reference replicate (mean over replicates).
This is Met-A split by channel. Also the two channels combined (all identity sites)."""
import numpy as np, pandas as pd, os
from scipy.special import digamma
LN2=np.log(2); D="/home/ubuntu/data/lines/imr90"
S=[("GSM1181642","Proliferating"),("GSM1181646","Proliferating"),("GSM1181647","Proliferating"),("GSM1181649","Senescent"),("GSM1181650","Senescent"),
   ("GSM1181652","Senescent"),("GSM1181655","SV40"),("GSM1181657","SV40"),("GSM1181659","SV40")]
import urllib.request, multiprocessing as mp
CH={f"chr{c}":i for i,c in enumerate(list(range(1,23))+["X","Y"],1)}
SUF={"GSM1181642":"Rep1.Proliferating","GSM1181646":"Rep2.Proliferating","GSM1181647":"Rep3.Proliferating","GSM1181649":"Rep1.Senescent",
     "GSM1181650":"Rep2.Senescent","GSM1181652":"Rep3.Senescent","GSM1181655":"Rep1.SV40","GSM1181657":"Rep2.SV40","GSM1181659":"Rep3.SV40"}
def fetch(g):
    p=f"{D}/{g}.parquet"
    if os.path.exists(p): return p
    os.makedirs(D,exist_ok=True); tmp=p+".gz"
    import subprocess
    u=f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/{g}_{SUF[g]}.summary.txt.gz"
    rc=subprocess.run(["curl","-sS","-L","--fail","--retry","8","--retry-delay","20","-o",tmp,u]).returncode
    if rc: raise RuntimeError(f"curl {rc} {u}")
    parts=[]
    for ch in pd.read_csv(tmp,sep="\t",header=None,names=["chr","pos","m","u"],dtype={"chr":str,"pos":np.int64,"m":np.int32,"u":np.int32},chunksize=5_000_000):
        ch=ch[(ch.m+ch.u)>=10]; ch=ch[ch.chr.isin(CH)]
        parts.append(pd.DataFrame({"key":ch.chr.map(CH).astype(np.int64)*10**10+ch.pos,"m":ch.m.values,"n":(ch.m+ch.u).values}))
    pd.concat(parts).to_parquet(p); os.remove(tmp); return p
with mp.get_context("fork").Pool(3) as P: P.map(fetch,[g for g,_ in S])
print("fetched",flush=True)
T={g:pd.read_parquet(f"{D}/{g}.parquet").set_index("key") for g,_ in S}; lab=dict(S)
common=None
for d in T.values(): common=d.index if common is None else common.intersection(d.index)
def Hc(k,n):
    p=k/n; m=(p>0)&(p<1); pl=np.zeros_like(p); x=p[m]; pl[m]=-(x*np.log2(x)+(1-x)*np.log2(1-x))
    mm=pl+np.where(m,1/(2*n*LN2),0.0); a=k+0.5; b=n-k+0.5
    return (mm+(digamma(a+b+1)-(a/(a+b))*digamma(a+1)-(b/(a+b))*digamma(b+1))/LN2)/2
B={g:(T[g].m.reindex(common)/T[g].n.reindex(common)).values for g in T}
H={g:Hc(T[g].m.reindex(common).values.astype(float),T[g].n.reindex(common).values.astype(float)) for g in T}
PRO=[g for g,s in S if s=="Proliferating"]; rows=[]
# Selection and reference on DIFFERENT replicates, so the reference H is not computed on the data that chose the sites:
# sites chosen on proliferating replicate a (beta in band; depth >= 10 everywhere), reference H from replicate b, specimen read against it.
# Every ordered pair (a, b) of proliferating replicates not equal to the specimen is used and the readings averaged.
import itertools
for g,st in S:
    pool=[e for e in PRO if e!=g]; acc=[]
    for a,b in itertools.permutations(pool,2):
        hi=(B[a]>=0.75)&(B[a]<=0.95); lo=(B[a]>=0.05)&(B[a]<=0.25)
        acc.append((H[g][hi].mean()/H[b][hi].mean(), H[g][lo].mean()/H[b][lo].mean(), H[g][hi|lo].mean()/H[b][hi|lo].mean(),
                    B[g][hi].mean(), B[g][lo].mean(), int(hi.sum()), int(lo.sum())))
    A=np.array(acc); rows.append(dict(gsm=g,state=st,n_pairs=len(acc),n_meth=int(A[:,5].mean()),n_unmeth=int(A[:,6].mean()),
        A_meth=A[:,0].mean(),A_unmeth=A[:,1].mean(),A_both=A[:,2].mean(),beta_meth=A[:,3].mean(),beta_unmeth=A[:,4].mean()))
X=pd.DataFrame(rows); X.to_csv("imr90_channels.csv",index=False); print(X.round(4).to_string(index=False)); print("DONE")
