#!/usr/bin/env python3
"""IAM-A floor comparison for neutrophils (development, 2026-10-01; author: 'test both and see which one is more defensible and the true IAM physics way').
Data: Loyfer 2023 blood granulocytes (>= 90 % neutrophils), 3 donors, read-level .pat (first 60 MB of each file = same genomic stretch, as PROC-CHANNEL-01).
Copy error eps = isolated unmethylated CpG between two methylated CpGs on reads with >= 6 CpGs and >= 80 % methylated, per opportunity (uncorrected:
.pat files carry no sequence, so no sequencing-error term; same as PROC-CHANNEL-01).
Readings compared (H = binary entropy):
  (a) physics floor:        IAM-A = H(eps) / H(eps0),  eps0 = 1/(1+exp(phi*M)), phi*M = 3.41 kT (canon)
  (b) own healthy baseline: IAM-A = H(eps) / H(eps_ref), eps_ref = the other donors' pooled eps (leave-one-donor-out)
  (c) physics x position:   IAM-A = H(eps) / (P * H(eps0)), P = H(eps_ref)/H(eps0) frozen and printed  [numerically = (b); differs in what is reported]
Checks: healthy in Normal; repeat (odd vs even reads of each donor); known damage: 2 % of methylated CpGs on qualifying reads set to T at random."""
import gzip, zlib, urllib.request, os, numpy as np, pandas as pd, json
PF=[l.strip() for l in open("pat_gran.txt") if l.strip().endswith(".pat.gz")]
C="/home/ubuntu/data/lines/loyfer_pat_head"; os.makedirs(C,exist_ok=True)
def get(f):
    g=f.split("_")[0]; p=f"{C}/{g}.head.gz"
    if not (os.path.exists(p) and os.path.getsize(p)>1e6):
        u=f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/{f}"
        with urllib.request.urlopen(urllib.request.Request(u,headers={"Range":"bytes=0-59999999"}),timeout=600) as r: open(p,"wb").write(r.read())
    return g,p
def lines(p):
    data=open(p,"rb").read(); out=b""
    while data:
        d=zlib.decompressobj(16+zlib.MAX_WBITS)
        try: out+=d.decompress(data)
        except zlib.error: break
        data=d.unused_data
    for l in out.split(b"\n")[:-1]: yield l
def iso_count(c):
    o=len(c)-2 if len(c)>2 else 0; e=sum(1 for i in range(1,len(c)-1) if c[i]=="T" and c[i-1]=="C" and c[i+1]=="C"); return e,o
def count(args):
    p,mode=args; rng=np.random.default_rng(7); iso=opp=0; k=0
    for l in lines(p):
        q=l.split(b"\t")
        if len(q)<4: continue
        pat=q[2].decode(); n=int(q[3]); c=[x for x in pat if x!="."]
        if len(c)<6 or c.count("C")<0.8*len(c):
            k+=n; continue
        if mode=="damaged":
            for _ in range(n):
                cc=["T" if (x=="C" and rng.random()<0.02) else x for x in c]; e,o=iso_count(cc); iso+=e; opp+=o
            continue
        e,o=iso_count(c)
        if mode in ("odd","even"):
            w=sum(1 for t in range(k+1,k+n+1) if t%2==(1 if mode=="odd" else 0))
        else: w=n
        iso+=e*w; opp+=o*w; k+=n
    return iso,opp
H=lambda e: -(e*np.log2(e)+(1-e)*np.log2(1-e)); eps0=1/(1+np.exp(3.41)); H0=H(eps0)
P=[get(f) for f in PF]; import multiprocessing as mp
tasks=[(p,m) for g,p in P for m in ("all","odd","even","damaged")]
with mp.get_context("fork").Pool(len(tasks)) as PL: res=PL.map(count,tasks)
rows=[]
for gi,(g,p) in enumerate(P):
    (i,o),(ia,oa),(ib,ob),(idd,od)=res[gi*4:gi*4+4]
    rows.append(dict(gsm=g,iso=i,opp=o,eps=i/o,eps_odd=ia/oa,eps_even=ib/ob,eps_damaged=idd/od)); print(rows[-1],flush=True)
X=pd.DataFrame(rows)
for j,r in X.iterrows():
    ref=X.drop(j); er=ref.iso.sum()/ref.opp.sum()
    X.loc[j,"eps_ref_loo"]=er; X.loc[j,"P_position"]=H(er)/H0
    for nm,e in (("",r.eps),("_odd",r.eps_odd),("_even",r.eps_even),("_damaged",r.eps_damaged)):
        X.loc[j,f"A_phys{nm}"]=H(e)/H0; X.loc[j,f"A_own{nm}"]=H(e)/H(er)
X["E_kT"]=np.log((1-X.eps)/X.eps)
O=pd.read_csv("physics_floor_cells.csv"); oth=O[["cell","copy_err"]].copy(); oth["A_phys"]=H(oth.copy_err)/H0
X.to_csv("iama_floor_granulocytes.csv",index=False); pd.set_option("display.width",250)
print(X.round(5).to_string(index=False))
inN=lambda a:((a>=0.95)&(a<=1.05))
print("eps0 %.4f H0 %.4f | granulocyte eps mean %.4f CV across donors %.3f | position P mean %.3f range %.3f-%.3f"%(eps0,H0,X.eps.mean(),X.eps.std()/X.eps.mean(),X.P_position.mean(),X.P_position.min(),X.P_position.max()))
for f in ("phys","own"): print(f"{f}: healthy Normal {inN(X['A_'+f]).sum()}/3 | odd/even |diff| max {np.abs(X['A_'+f+'_odd']-X['A_'+f+'_even']).max():.4f} | 2% damage >1.05: {(X['A_'+f+'_damaged']>1.05).sum()}/3 (median {X['A_'+f+'_damaged'].median():.3f})")
print("other healthy cell types on the bare physics floor: A_phys median %.3f, range %.3f-%.3f, in Normal %d/%d"%(oth.A_phys.median(),oth.A_phys.min(),oth.A_phys.max(),inN(oth.A_phys).sum(),len(oth)))
print("DONE")
