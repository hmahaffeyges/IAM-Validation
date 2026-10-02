#!/usr/bin/env python3
"""Chain v3 batch runner (PROC-NEUT-TEST-01). Each array: GEO IDAT pair -> run_sample.py --engine v3 (unchanged chain, commit bc4a651).
Pass 2 (whole blood): the same chain again with --slide-ref-table = the healthy references' pass-1 records (A, f_neu, N; self excluded).
Stage T fits the noise-corrected expectation (A = a + b f_neu + c N, DEV-NOISE-02) when >= 20 records exist, else a median tare. Reference
choice: the whole test batch when it holds >= 20 complete records; otherwise the same slide (>= 3), else the same test batch. Reference sets per pre-registration: T1 healthy; T3 all of T3; T4 NEGATIVE only. T2 isolated: no tare."""
import os, sys, json, subprocess, tarfile, urllib.request, time, pandas as pd, multiprocessing as mp
W=os.getcwd(); tarfile.open("chain_v3.tgz").extractall("bio/MethylPhys"); CH=f"{W}/bio/MethylPhys/chain"
D="/home/ubuntu/data/neuttest/idat"; os.makedirs(D,exist_ok=True); os.makedirs("reports",exist_ok=True)
X=pd.read_csv("neut_test_manifest.csv"); TESTS=sys.argv[1].split(",") if len(sys.argv)>1 else ["T1","T2","T3","T4"]; X=X[X.test.isin(TESTS)].reset_index(drop=True)
def url(gsm,f): return f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/{f}"
def fetch(r):
    out=[]
    for f in (r.grn, r.grn.replace("_Grn","_Red")):
        p=f"{D}/{f}"
        if not (os.path.exists(p) and os.path.getsize(p)>1e6) and os.path.exists(p.replace(".idat.gz",".idat")): p=p.replace(".idat.gz",".idat")
        for a in range(6):
            if os.path.exists(p) and os.path.getsize(p)>1e6: break
            try: urllib.request.urlretrieve(url(r.gsm,f),p+".part"); os.replace(p+".part",p)
            except Exception: time.sleep(5*(a+1))
        out.append(p if os.path.exists(p) else None)
    return out
def run(args):
    i,refs=args; r=X.loc[i]; g,rd=fetch(r)   # refs: None (pass 1) or a DataFrame with columns id, A, f_neu, N (pass 2)
    if not (g and rd): return dict(gsm=r.gsm,status="download failed")
    tag="_tared" if refs is not None else ""; out=f"{W}/reports/{r.gsm}{tag}.html"
    cmd=["/home/ubuntu/env/bin/python",f"{CH}/MethylPhys_Interface/run_sample.py","--grn",g,"--red",rd,"--engine","v3","--specimen",r.specimen,
         "--array-type","EPIC_v1","--out",out,"--id",r.gsm,"--ledger",f"{W}/reports/ledger.jsonl"]
    if isinstance(r.sex_d,str) and r.sex_d[:1].upper() in "MF": cmd+=["--sex",r.sex_d[:1].upper()]
    try: cmd+=["--age",str(float(r.age_d))]
    except Exception: pass
    if refs is not None:
        rp=f"{W}/reports/{r.gsm}_refs.csv"; refs.to_csv(rp,index=False); cmd+=["--slide-ref-table",rp]
    p=subprocess.run(cmd,capture_output=True,text=True,cwd=f"{CH}/MethylPhys_Interface",env=dict(os.environ,PYTHONPATH=CH,OMP_NUM_THREADS="1"))
    bp=out.replace(".html","_bundle.json")
    if not os.path.exists(bp): return dict(gsm=r.gsm,status="no bundle",tail=(p.stdout+p.stderr)[-400:])
    o=json.load(open(bp)); m=o.get("met_a",{}); c=o.get("met_a_cscore",{}); t=o.get("tare",{}); it=o.get("intake") or {}; fr=(o.get("composition") or {}).get("fractions") or {}
    return dict(gsm=r.gsm,status="ok",intake=it.get("stage0_verdict"),call_rate=it.get("call_rate_status"),A=m.get("A"),reason=m.get("reason"),f_neu=fr.get("NEU"),
                f_lymph=sum(fr.get(k,0) for k in ("B","NK","CD4T","CD8T")),f_mono=fr.get("MONO"),f_eos=fr.get("EOS"),n_sites=m.get("n_sites"),C=c.get("C"),A_rel=t.get("A_rel"),tare=t.get("state",t.get("reason")),
                shift1=m.get("shift_per_1pct_loss"),beta_meth=m.get("methylated_sites_mean_beta"),past_ceiling=m.get("past_entropy_ceiling"),
                det_limit=t.get("detection_limit_pct_loss"),ref_sd=t.get("reference_spread_sd"),N=m.get("noise_index"),tare_method=t.get("method"))
NP=int(os.environ.get("NP","24"))
# Bulk download first: one GEO series archive at a time (parallel per-sample requests are refused by NCBI and fall back to a serial retry).
have=set(os.listdir(D))
for g in X.gse.unique():
    need=[f for f in X[X.gse==g].grn if f not in have and f.replace(".idat.gz",".idat") not in have]
    if not need: continue
    n=g[:-3]+"nnn"
    for a in range(5):
        rc=os.system(f"cd {D} && curl -s -f -C - -o {g}_RAW.tar https://ftp.ncbi.nlm.nih.gov/geo/series/{n}/{g}/suppl/{g}_RAW.tar && tar -xf {g}_RAW.tar && rm -f {g}_RAW.tar")
        if rc==0: break
        time.sleep(30)
    have=set(os.listdir(D)); print(g,"archive",rc,"| still missing",sum(1 for f in X[X.gse==g].grn if f not in have),flush=True)
with mp.get_context("fork").Pool(NP) as P: R1=pd.DataFrame(P.map(run,[(i,None) for i in X.index]))
bad=R1[R1.status!="ok"].gsm.tolist()
if bad:
    idx=[X.index[X.gsm==g][0] for g in bad]; R1b=pd.DataFrame([run((i,None)) for i in idx]); R1=pd.concat([R1[R1.status=="ok"],R1b],ignore_index=True)
R1.to_csv("pass1.csv",index=False); print("pass1",R1.status.value_counts().to_dict(),flush=True)
Y=X.merge(R1,on="gsm"); jobs=[]
for i,r in X.iterrows():
    if r.specimen!="whole blood": continue
    ref=Y[(Y.test==r.test)&(Y.gsm!=r.gsm)&Y.A.notna()]
    if r.test=="T4": ref=ref[ref.group=="NEGATIVE"]
    if r.test=="T1": ref=ref[ref.group=="healthy"]
    full=ref.dropna(subset=["A","f_neu","N"])
    if len(full)<20:
        s=ref[ref.slide==r.slide]; ref=s if len(s)>=3 else ref
    if len(ref)>=3 and pd.notna(Y.loc[Y.gsm==r.gsm,"A"]).any():
        jobs.append((i,ref[["gsm","A","f_neu","N"]].rename(columns={"gsm":"id"}).reset_index(drop=True)))
with mp.get_context("fork").Pool(NP) as P: R2=pd.DataFrame(P.map(run,jobs))
if len(R2): Y=Y.merge(R2[["gsm","A_rel","tare","det_limit","ref_sd","tare_method"]].rename(columns={"A_rel":"A_rel_tared","tare":"tared_state"}),on="gsm",how="left")
Y.to_csv("neut_test_readings.csv",index=False); os.system(f"cd {W} && tar czf reports.tgz reports"); print("DONE",flush=True)
