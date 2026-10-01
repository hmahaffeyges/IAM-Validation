#!/usr/bin/env python3
"""Chain v3 end-to-end acceptance (2026-10-01): IDAT pair -> run_sample.py --engine v3 (Stage 0 intake, Stage 1 calibration, composition,
Met-A, C-score, tare, report). Every specimen goes through the chain's own entry point; nothing is computed outside it."""
import os, glob, json, subprocess, tarfile, re, pandas as pd, multiprocessing as mp
W=os.getcwd(); tarfile.open("chain_v3.tgz").extractall("bio/MethylPhys"); CH=f"{W}/bio/MethylPhys/chain"
os.environ["IAMATLAS_V2"]="/home/ubuntu/data/IAMAtlas_v2.parquet"; os.makedirs("reports",exist_ok=True)
R="/home/ubuntu/data/atlas_sources/blood/GSE110554/idats"
META=pd.read_csv("geo_meta.csv").set_index("gsm")
def decl(g):
    if g not in META.index: return None,None
    r=META.loc[g]; sx=str(r.get("sex","")).strip().upper()[:1] or None
    ag=r.get("age"); ag=r.get("age at diagnosis") if pd.isna(ag) else ag
    try: ag=float(ag)
    except Exception: ag=None
    return (sx if sx in ("M","F") else None), ag
def pair(d,g):
    f=sorted(glob.glob(f"{d}/{g}_*Grn.idat*")); 
    if not f: return None
    grn=[x for x in f if x.endswith(".idat")] or f; grn=grn[0]; red=grn.replace("_Grn","_Red")
    return (grn,red) if os.path.exists(red) else None
S=pd.read_csv("roster_samples.csv"); neu=S[(S.source=="Salas2018")&(S.cell=="neutrophils")]["sample"].str.split("_").str[0].tolist()
T=pd.read_csv("salas_mixture_truth.csv"); mix=T[(T.gse=="GSE110554")&(T.neu>=50)].gsm.tolist()
A=pd.read_csv("aml_serial_readings.csv"); rm2=A[(A.tp=="Rm2")&(A.tissue.str.contains("blood",case=False))].gsm.tolist()
jobs=[(g,"isolated neutrophils","healthy purified neutrophil (in floor)",R) for g in neu]+[(g,"whole blood","known mixture, neu>=50%",R) for g in mix]+\
     [(g,"whole blood","AML second remission blood (other lab)","/home/ubuntu/data/gse315367/idat") for g in rm2]
def run(j):
    g,spec,grp,d=j[:4]; refs=j[4] if len(j)>4 else None; p=pair(d,g)
    if not p: return dict(gsm=g,group=grp,status="no idat pair")
    out=f"{W}/reports/{g}.html"
    cmd=["/home/ubuntu/env/bin/python",f"{CH}/MethylPhys_Interface/run_sample.py","--grn",p[0],"--red",p[1],"--engine","v3","--specimen",spec,
         "--array-type","EPIC_v1","--out",out,"--id",g,"--ledger",f"{W}/reports/ledger.jsonl"]
    if refs: cmd+=["--slide-ref-A",",".join(f"{x:.5f}" for x in refs)]; out=out.replace(".html","_tared.html"); cmd[cmd.index("--out")+1]=out
    sx,ag=decl(g)
    if grp.startswith("known mixture"): cmd+=["--no-intake"]          # lab-made DNA blend: no single donor, so no custody record
    else:
        if sx: cmd+=["--sex",sx]
        if ag is not None: cmd+=["--age",str(ag)]
    r=subprocess.run(cmd,capture_output=True,text=True,cwd=f"{CH}/MethylPhys_Interface",env=dict(os.environ,PYTHONPATH=CH))
    bp=out.replace(".html","_bundle.json")
    if not os.path.exists(bp): return dict(gsm=g,group=grp,status="FAILED",tail=(r.stdout+r.stderr)[-600:])
    o=json.load(open(bp)); m=o.get("met_a",{}); c=o.get("met_a_cscore",{}); it=o.get("intake") or {}
    return dict(gsm=g,group=grp,status="ok",intake=it.get("stage0_verdict"),call_rate=it.get("call_rate_status"),A=m.get("A"),state=m.get("state",m.get("reason")),
                f_neu=m.get("fraction"),n_sites=m.get("n_sites"),C=c.get("C"),A_rel=(o.get("tare") or {}).get("A_rel"),tare_state=(o.get("tare") or {}).get("state",(o.get("tare") or {}).get("reason")))
with mp.get_context("fork").Pool(8) as P: X=pd.DataFrame(P.map(run,jobs))
# pass 2: tare each whole-blood specimen against the other specimens of its batch (leave-one-out), through the chain again
ok=X[(X.status=="ok")&X.A.notna()]; J2=[]
for j in jobs:
    if j[1]!="whole blood": continue
    refs=ok[(ok.group==j[2])&(ok.gsm!=j[0])].A.astype(float).tolist()
    if len(refs)>=3: J2.append(j+(refs,))
with mp.get_context("fork").Pool(8) as P: X2=pd.DataFrame(P.map(run,J2))
X=X.merge(X2[["gsm","A_rel","tare_state"]].rename(columns={"A_rel":"A_rel_tared","tare_state":"tared_state"}),on="gsm",how="left")
X.to_csv("chain_acceptance.csv",index=False); pd.set_option("display.width",250); pd.set_option("display.max_colwidth",80)
print(X.drop(columns=[c for c in ("tail",) if c in X]).to_string(index=False))
if "tail" in X: print(X[X.status!="ok"][["gsm","tail"]].head(3).to_string())
inN=lambda a:((a>=0.95)&(a<=1.05))
for g,d in X[X.status=="ok"].groupby("group"):
    if d.A_rel_tared.notna().any(): print("  tared:",g,"Normal %d/%d"%(inN(d.A_rel_tared.astype(float)).sum(),d.A_rel_tared.notna().sum()),"range %.3f-%.3f"%(d.A_rel_tared.min(),d.A_rel_tared.max()))
    print(g, "| Normal %d/%d"%(inN(d.A.astype(float)).sum(),d.A.notna().sum()), "| A range %.3f-%.3f"%(d.A.min(),d.A.max()) if d.A.notna().any() else "")
os.system(f"cd {W} && tar czf reports.tgz reports"); print("DONE")
