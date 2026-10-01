#!/usr/bin/env python3
"""PROC-TUMOUR-01 scoring, as pre-registered (PROC_TUMOUR_01_PREREG.md, sha 5ab460cf5af368d5). Same statistic as PROC-SALMON-01.
Genotype mask per patient (both tissues, same assay): site dropped if >= 5 qualifying molecules and > 30 % carry an error."""
import glob, os, re, ast, json, numpy as np, pandas as pd
O="/home/ubuntu/data/tumour/out"
SP=sorted(os.path.basename(p)[:-8] for p in glob.glob(f"{O}/*.parquet"))
def parse(sp):
    st,pt,assay,tis=sp.split("_"); return dict(sp=sp,study=st,patient=pt,assay=assay,tissue=tis)
meta=pd.DataFrame([parse(s) for s in SP])
def inst(sp):
    line=re.findall(r"\{'reads'.*\}",open(f"{O}/{sp}_extract.log").read())[-1]; line=re.sub(r"np\.(?:float|int)\d*\(([^)]*)\)",r"\1",line); return ast.literal_eval(line)
T={s:pd.read_parquet(f"{O}/{s}.parquet") for s in SP}; mask={}
for (pt,assay),g in meta.groupby(["patient","assay"]):
    D=pd.concat([T[s][["pos","opp_A","opp_B","err_A","err_B"]] for s in g.sp]).groupby("pos").sum(); o=D.opp_A+D.opp_B; e=D.err_A+D.err_B
    mask[(pt,assay)]=set(D.index[(o>=5)&(e>0.30*o)])
rows=[]
for _,m in meta.iterrows():
    D=T[m.sp]; D=D[~D.pos.isin(mask[(m.patient,m.assay)])]; I=inst(m.sp)
    ep=(D.err_A.sum()+D.err_B.sum())/(D.opp_A.sum()+D.opp_B.sum()); s=I["sub_err"]
    rows.append(dict(**m,reads=I["reads"],qualifying=I["qualifying"],sites=len(D),conv_fail=I["conv_fail"],sub_err=s,eps=ep,eps_corr=ep-s))
R=pd.DataFrame(rows); R.to_csv("tumour_readings.csv",index=False)
P=R.pivot_table(index=["study","patient","assay"],columns="tissue",values=["eps_corr","conv_fail","qualifying"]).reset_index()
P.columns=["_".join([c for c in col if c]) for col in P.columns]
P["ratio"]=P.eps_corr_tumour/P.eps_corr_normal; P["conv_diff"]=(P.conv_fail_tumour-P.conv_fail_normal).abs(); P["instrument_ok"]=P.conv_diff<0.005
P.to_csv("tumour_pairs.csv",index=False)
C=P[(P.study=="EOCRC")&(P.assay=="WGBS")]; Ci=C[C.instrument_ok]
out=dict(n_pairs=int(len(C)),P1=dict(tumour_higher=int((C.ratio>1).sum()),of=int(len(C)),pass_=bool((C.ratio>1).sum()>=6)),
         P2=dict(median_ratio=float(C.ratio.median()),pass_=bool(C.ratio.median()>=1.10)),
         P3=dict(instrument_ok=int(C.instrument_ok.sum()),of=int(len(C)),limited=C[~C.instrument_ok].patient.tolist()),
         P1_instrument_ok_only=dict(tumour_higher=int((Ci.ratio>1).sum()),of=int(len(Ci)),median_ratio=float(Ci.ratio.median()) if len(Ci) else None))
O2=P[P.study=="OSCC"]; out["OSCC"]={a:dict(tumour_higher=int((g.ratio>1).sum()),of=int(len(g)),median_ratio=float(g.ratio.median())) for a,g in O2.groupby("assay")}
json.dump(out,open("tumour_score.json","w"),indent=1); pd.set_option("display.width",220)
print(P.round(5).to_string(index=False)); print(json.dumps(out,indent=1)); print("DONE")
