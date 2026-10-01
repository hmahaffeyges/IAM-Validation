#!/usr/bin/env python3
"""PROC-SALMON-01 scoring, as pre-registered (PROC_SALMON_01_PREREG.md, sha 4e65a4c06246cf56). Inputs: per-specimen site tables from extract.py.
Copy error eps = isolated errors / opportunities on qualifying molecules (>= 6 CpG, >= 80 % methylated); eps_corr = eps - s (sequencing error).
Genotype mask per fish (both tissues pooled): site dropped if opportunities >= 5 and error fraction > 0.30.
Deviation (recorded before scoring): per-fish age and hatchery (WH/WNFH) are not in the public metadata, so P3's within-age-4 check cannot be run;
published DMR coordinates were not obtained, so P4 reports the count of sites above the look-elsewhere threshold only."""
import glob, os, re, ast, itertools, json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
O="/home/ubuntu/data/salmon/out"; t0=__import__("time").time()
SP=sorted(os.path.basename(p)[:-8] for p in glob.glob(f"{O}/*.parquet"))
meta=pd.DataFrame([dict(sp=s,fish=s.split("_")[0],tissue=s.split("_")[1],origin=s.split("_")[2]) for s in SP])
def inst(sp):
    t=open(f"{O}/{sp}_extract.log").read(); line=re.findall(r"\{'reads'.*\}",t)[-1]
    line=re.sub(r"np\.(?:float|int)\d*\(([^)]*)\)",r"\1",line); return ast.literal_eval(line)
T={s:pd.read_parquet(f"{O}/{s}.parquet") for s in SP}
# genotype mask per fish
mask={}
for f,g in meta.groupby("fish"):
    D=pd.concat([T[s][["pos","opp_A","opp_B","err_A","err_B"]] for s in g.sp]).groupby("pos").sum()
    o=D.opp_A+D.opp_B; e=D.err_A+D.err_B; mask[f]=set(D.index[(o>=5)&(e>0.30*o)])
rows=[]
for s in SP:
    D=T[s]; f=s.split("_")[0]; D=D[~D.pos.isin(mask[f])]; I=inst(s)
    eA=D.err_A.sum()/D.opp_A.sum(); eB=D.err_B.sum()/D.opp_B.sum(); e=(D.err_A.sum()+D.err_B.sum())/(D.opp_A.sum()+D.opp_B.sum())
    rows.append(dict(sp=s,reads=I["reads"],qualifying=I["qualifying"],sites=len(D),masked=len(mask[f]),conv_fail=I["conv_fail"],sub_err=I["sub_err"],
                     eps=e,eps_A=eA,eps_B=eB,eps_corr=e-I["sub_err"],eps_corr_A=eA-I["sub_err"],eps_corr_B=eB-I["sub_err"],opp=int(D.opp_A.sum()+D.opp_B.sum())))
R=meta.merge(pd.DataFrame(rows),on="sp"); R["E_kT"]=np.log((1-R.eps_corr)/R.eps_corr); R["phi"]=R.E_kT/22.94
R.to_csv("salmon_readings.csv",index=False)
out={}
# P0
a,b=R.eps_corr_A.values,R.eps_corr_B.values; n=len(a); gm=np.r_[a,b].mean()
msb=2*((((a+b)/2)-gm)**2).sum()/(n-1); msw=(((a-b)**2)/2).sum()/n; icc=(msb-msw)/(msb+msw)
out["P0"]=dict(icc=round(icc,4),median_abs_halfdiff=float(np.median(np.abs(a-b))),between_fish_sd={t:float(g.eps_corr.std()) for t,g in R.groupby("tissue")})
out["P0"]["pass"]=bool(icc>=0.80 and all(out["P0"]["median_abs_halfdiff"]<v for v in out["P0"]["between_fish_sd"].values()))
# P1
out["P1"]={t:dict(between_sd=float(g.eps_corr.std()),within_sd=float(np.sqrt((((g.eps_corr_A-g.eps_corr_B)**2)/2).mean()))) for t,g in R.groupby("tissue")}
out["P1"]["pass"]=bool(all(v["between_sd"]>v["within_sd"] for k,v in out["P1"].items() if k!="pass"))
# P2
med=float(R[R.tissue=="RBC"].eps_corr.median()); out["P2"]=dict(median_RBC_eps_corr=med,window=[0.021,0.027],pass_=bool(0.021<=med<=0.027),
    median_Sp_eps_corr=float(R[R.tissue=="Sp"].eps_corr.median()))
# P3
out["P3"]={}
for t,g in R.groupby("tissue"):
    h=g[g.origin=="Hat"].eps_corr; nn=g[g.origin=="Nat"].eps_corr; u=mannwhitneyu(h,nn,alternative="two-sided")
    out["P3"][t]=dict(median_hat=float(h.median()),median_nat=float(nn.median()),U=float(u.statistic),p=float(u.pvalue),significant_at_0025=bool(u.pvalue<0.025))
# P4 difference map on per-site methylation, t >= 10 in every fish of the tissue
out["P4"]={}
for t,g in R.groupby("tissue"):
    B={}
    for _,r in g.iterrows():
        D=T[r.sp]; D=D[(D.t>=10)&~D.pos.isin(mask[r.fish])]; B[r.fish]=pd.Series((D.m/D.t).values,index=D.pos.values)
    M=pd.DataFrame(B).dropna(); org=dict(zip(g.fish,g.origin)); H=[f for f in M if org[f]=="Hat"]; N=[f for f in M if org[f]=="Nat"]
    X=M.values; cols=list(M.columns)
    def zmap(g1,g2):
        i1=[cols.index(f) for f in g1]; i2=[cols.index(f) for f in g2]
        x1,x2=X[:,i1],X[:,i2]; se=np.sqrt(x1.var(1,ddof=1)/len(i1)+x2.var(1,ddof=1)/len(i2)+1e-6)
        return (x1.mean(1)-x2.mean(1))/se
    nulls=[]; seen=set()
    for c in itertools.combinations(N,5):
        k=frozenset(c); k2=frozenset(set(N)-set(c))
        if k2 in seen: continue
        seen.add(k); nulls.append(np.abs(zmap(list(c),list(k2))).max())
    thr=float(np.quantile(nulls,0.95)); z=zmap(H,N)
    out["P4"][t]=dict(sites=int(len(M)),n_splits=len(nulls),threshold_absz=thr,sites_above=int((np.abs(z)>thr).sum()),max_absz=float(np.abs(z).max()))
    pd.DataFrame(dict(pos=M.index.values,z=z,dbeta=X[:,[cols.index(f) for f in H]].mean(1)-X[:,[cols.index(f) for f in N]].mean(1))).to_parquet(f"salmon_diffmap_{t}.parquet")
json.dump(out,open("salmon_score.json","w"),indent=1)
pd.set_option("display.width",220)
print(R[["sp","reads","qualifying","sites","masked","conv_fail","sub_err","eps","eps_corr","eps_corr_A","eps_corr_B","E_kT"]].round(5).to_string(index=False))
print(json.dumps(out,indent=1)); print("DONE %.0fs"%(__import__("time").time()-t0))
