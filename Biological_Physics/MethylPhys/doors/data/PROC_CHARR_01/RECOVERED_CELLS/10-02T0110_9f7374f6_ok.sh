cd remote_jobs/rimouski && cat > score_rimouski.py <<'EOF'
#!/usr/bin/env python3
"""PROC-RIMOUSKI-01 scoring, as pre-registered (PROC_RIMOUSKI_01_PREREG.md). Same statistic and code path as PROC-CHARR-01."""
import glob, os, re, ast, json, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
O="/home/ubuntu/data/rimouski/out"
R0=pd.read_csv("rimouski_runs.csv"); meta=R0.groupby("library_name")[["generation","source","sex"]].first().reset_index().rename(columns={"library_name":"sp"})
SP=sorted(os.path.basename(p)[:-8] for p in glob.glob(f"{O}/*.parquet")); missing=sorted(set(meta.sp)-set(SP)); meta=meta[meta.sp.isin(SP)].reset_index(drop=True)
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
R=meta.merge(pd.DataFrame(rows),on="sp"); R["E_kT"]=np.log((1-R.eps_corr)/R.eps_corr)
R["origin"]=np.where(R.generation=="F0",R.source,None)
R["father"]=np.where(R.generation=="F1",R.source.str.extract(r"^(wild|stocked)father")[0],None)
R["mother"]=np.where(R.generation=="F1",R.source.str.extract(r"father_(wild|stocked)mother")[0],None)
R.to_csv("rimouski_readings.csv",index=False)
out={"n_fish":int(len(R)),"missing":missing}
a,b=R.eps_corr_A.values,R.eps_corr_B.values; n=len(a); gm=np.r_[a,b].mean()
msb=2*((((a+b)/2)-gm)**2).sum()/(n-1); msw=(((a-b)**2)/2).sum()/n; icc=(msb-msw)/(msb+msw)
out["P0"]=dict(icc=round(float(icc),4),pass_=bool(icc>=0.80))
rho={k:float(spearmanr(R.eps_corr,R[k]).correlation) for k in ("conv_fail","dup_frac","masked_frac")}
out["P1"]=dict(rho=rho,pass_=bool(all(abs(v)<0.30 for v in rho.values())))
F0=R[R.generation=="F0"]; Emed=float(F0.E_kT.median())
out["P2"]=dict(median_F0_E_kT=Emed,window_kT=[3.6,4.3],pass_=bool(3.6<=Emed<=4.3))
m=smf.ols("eps_corr ~ C(origin, Treatment('wild')) + C(sex)",data=F0).fit(); k=[x for x in m.params.index if x.startswith("C(origin")][0]
out["P3"]=dict(stocked_minus_wild=float(m.params[k]),ci=[float(m.conf_int().loc[k,0]),float(m.conf_int().loc[k,1])],p=float(m.pvalues[k]),
               significant=bool(m.pvalues[k]<0.05),interpreted=out["P1"]["pass_"])
F1=R[R.generation=="F1"]; m4=smf.ols("eps_corr ~ C(father, Treatment('wild')) + C(mother, Treatment('wild'))",data=F1).fit()
out["P4"]={t:dict(estimate=float(m4.params[t]),ci=[float(m4.conf_int().loc[t,0]),float(m4.conf_int().loc[t,1])],p=float(m4.pvalues[t])) for t in m4.params.index if t!="Intercept"}
out["P4"]["interpreted"]=out["P1"]["pass_"]
out["descriptive_F0_vs_F1"]=dict(F0_median_eps=float(F0.eps_corr.median()),F1_median_eps=float(F1.eps_corr.median()),
                                 F0_median_E=Emed,F1_median_E=float(F1.E_kT.median()))
json.dump(out,open("rimouski_score.json","w"),indent=1)
pd.set_option("display.width",230); print(R[["sp","generation","source","sex","qualifying","dup_frac","conv_fail","sub_err","eps_corr","eps_corr_A","eps_corr_B","E_kT"]].round(5).to_string(index=False))
print(json.dumps(out,indent=1)); print("DONE")
EOF
python3 -c "import ast;ast.parse(open('score_rimouski.py').read());print('ok')"