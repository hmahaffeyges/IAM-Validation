#!/usr/bin/env python3
"""PROC-CHARR-01 scoring, as pre-registered (PROC_CHARR_01_PREREG.md, sha 1460b52e5f4bb676; depth 5 M pairs set from the timing pilot).
Same statistic as PROC-SALMON-01. Inputs: per-fish site tables from extract_pe.py (one table per fish = library)."""
import glob, os, re, ast, json, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
O="/home/ubuntu/data/charr/out"
R0=pd.read_csv("charr_runs.csv"); meta=R0.groupby("library_name")[["fish","line","temp","year"]].first().reset_index().rename(columns={"library_name":"sp"})
SP=sorted(os.path.basename(p)[:-8] for p in glob.glob(f"{O}/*.parquet")); meta=meta[meta.sp.isin(SP)].reset_index(drop=True)
def inst(sp):
    line=re.findall(r"\{'pairs'.*\}",open(f"{O}/{sp}_extract.log").read())[-1]; line=re.sub(r"np\.(?:float|int)\d*\(([^)]*)\)",r"\1",line); return ast.literal_eval(line)
def dup(sp):
    m=re.search(r"Total number duplicated alignments removed:\s*\d+\s*\(([\d.]+)%\)",open(f"{O}/{sp}_report.txt").read()); return float(m.group(1))/100 if m else np.nan
rows=[]
for sp in meta.sp:
    D=pd.read_parquet(f"{O}/{sp}.parquet"); o=D.opp_A+D.opp_B; e=D.err_A+D.err_B
    msk=(o>=5)&(e>0.30*o); n_m=int(msk.sum()); D=D[~msk]; I=inst(sp)
    eA=D.err_A.sum()/D.opp_A.sum(); eB=D.err_B.sum()/D.opp_B.sum(); ep=(D.err_A.sum()+D.err_B.sum())/(D.opp_A.sum()+D.opp_B.sum()); s=I["sub_err"]
    rows.append(dict(sp=sp,pairs=I["pairs"],qualifying=I["qualifying"],sites=len(D),masked_frac=n_m/max(len(D)+n_m,1),dup_frac=dup(sp),conv_fail=I["conv_fail"],sub_err=s,
                     eps=ep,eps_corr=ep-s,eps_corr_A=eA-s,eps_corr_B=eB-s,opp=int(D.opp_A.sum()+D.opp_B.sum())))
R=meta.merge(pd.DataFrame(rows),on="sp"); R["E_kT"]=np.log((1-R.eps_corr)/R.eps_corr); R.to_csv("charr_readings.csv",index=False)
out={"n_fish":int(len(R))}
a,b=R.eps_corr_A.values,R.eps_corr_B.values; n=len(a); gm=np.r_[a,b].mean()
msb=2*((((a+b)/2)-gm)**2).sum()/(n-1); msw=(((a-b)**2)/2).sum()/n; icc=(msb-msw)/(msb+msw)
out["P0"]=dict(icc=round(float(icc),4),pass_=bool(icc>=0.80))
rho={k:float(spearmanr(R.eps_corr,R[k]).correlation) for k in ("conv_fail","dup_frac","masked_frac")}
half=float(np.median(np.abs(a-b))); sd=float(R.eps_corr.std())
out["P1"]=dict(rho=rho,between_fish_sd=sd,median_half_absdiff=half,pass_=bool(all(abs(v)<0.30 for v in rho.values()) and sd>half))
med=float(R[R.temp=="ambient"].eps_corr.median()); Emed=float(np.log((1-med)/med))
out["P2"]=dict(median_ambient_eps_corr=med,median_ambient_E_kT=Emed,window_kT=[3.8,4.3],pass_=bool(3.8<=Emed<=4.3))
m=smf.ols("eps_corr ~ C(temp, Treatment('ambient')) + C(line) + C(year)",data=R).fit(); k=[x for x in m.params.index if x.startswith("C(temp")][0]
base=float(m.params["Intercept"]) if False else float(R[R.temp=="ambient"].eps_corr.mean())
d,lo,hi=float(m.params[k]),float(m.conf_int().loc[k,0]),float(m.conf_int().loc[k,1]); ratio=(base+d)/base; rlo=(base+lo)/base; rhi=(base+hi)/base
verdict=("fixed in kT" if (rlo<=1.0<=rhi and rhi<1.03) else "fixed in joules" if (rlo<=1.03<=rhi and rlo>1.0) else "underpowered")
out["P3"]=dict(warm_minus_ambient=d,ci=[lo,hi],ratio=ratio,ratio_ci=[rlo,rhi],verdict=verdict,p=float(m.pvalues[k]))
kl=[x for x in m.params.index if x.startswith("C(line")][0]
out["P4"]=dict(term=kl,estimate=float(m.params[kl]),ci=[float(m.conf_int().loc[kl,0]),float(m.conf_int().loc[kl,1])],p=float(m.pvalues[kl]),interpreted=bool(out["P1"]["pass_"]))
json.dump(out,open("charr_score.json","w"),indent=1)
pd.set_option("display.width",220); print(R[["sp","year","line","temp","qualifying","dup_frac","conv_fail","sub_err","eps_corr","eps_corr_A","eps_corr_B","E_kT"]].round(5).to_string(index=False))
print(json.dumps(out,indent=1)); print("DONE")
