#!/usr/bin/env python3
"""PROC-BIDIR-01 — CHAIN_COMMISSIONING row 4.5: the bidirectional detector reproduced from the kit.

Bars (PREREG sealed 2026-09-21):
  B1 integrity   every file in VAL_050_SEAL.txt / VAL_051_SEAL.txt hashes to its seal (inputs shared by both live in val_050_aibl
                 and are copied beside val051_analyze.py; the copies must hash identically)
  B2 VAL-050     run_val_050.py rerun -> pooled-entropy d = +0.077 +/-0.002, AUC 0.512 +/-0.005
  B3 VAL-051     val051_analyze.py rerun -> Rule A holdout d = +0.6237 +/-0.002, AUC 0.6769 +/-0.005, null comparator d = +0.0562 +/-0.002, 33 AD / 95 HC
  B4 engine=seal bidirectional_decomposition.score_directional_composite == sealed per-sample A_dir_A (max |diff| < 1e-9) on the runtime immune panel
  B5 raw GEO     18 IMM CpGs x 726 samples re-extracted from GSE153712_normalized_average_betas.txt.gz (CPG_KIT_DATA) match aibl_imm_betas.json
                 at max |diff| < 1e-4  (skipped with a notice if the file is absent — the row is not commissioned without it)
Usage: CPG_KIT_DATA=<dir with GSE153712_normalized_average_betas.txt.gz> python3 PROC_BIDIR_01.py
"""
import os, sys, json, gzip, hashlib, subprocess, shutil, math, statistics
HERE=os.path.dirname(os.path.abspath(__file__)); BP=os.path.abspath(os.path.join(HERE,"..",".."))
V50=os.path.join(BP,"Record/VAL_PreAtlas/val_050_aibl"); V51=os.path.join(BP,"Record/VAL_PreAtlas/val_051_ad_directional")
ENG=os.path.join(BP,"MethylPhys/chain"); DATA=os.environ.get("CPG_KIT_DATA",os.path.join(HERE,"data")); OUT=os.path.join(HERE,"results"); os.makedirs(OUT,exist_ok=True)
sha=lambda p: hashlib.sha256(open(p,"rb").read()).hexdigest()
res={}
# B1
n=ok=0; bad=[]
for d,seal in ((V50,"VAL_050_SEAL.txt"),(V51,"VAL_051_SEAL.txt")):
    for l in open(os.path.join(d,seal)):
        p=l.split()
        if len(p)==2 and len(p[0])==64:
            n+=1; f=os.path.join(d,p[1])
            if not os.path.exists(f) and p[1] in ("aibl_manifest.json","aibl_imm_betas.json"): shutil.copy(os.path.join(V50,p[1]),f)
            if os.path.exists(f) and sha(f)==p[0]: ok+=1
            else: bad.append(p[1])
res["B1"]={"sealed_files":n,"hash_ok":ok,"bad":bad,"pass":ok==n}; print(f"B1 integrity: {ok}/{n} sealed hashes match {bad or ''} -> {'PASS' if ok==n else 'FAIL'}")
# B2/B3: rerun in place, compare, restore sealed RESULTS
def rerun(d,script,results,keys,tol):
    sealed=json.load(open(os.path.join(d,results))); bak=os.path.join(OUT,results+".sealed"); shutil.copy(os.path.join(d,results),bak)
    r=subprocess.run([sys.executable,script],cwd=d,capture_output=True,text=True)
    new=json.load(open(os.path.join(d,results))); shutil.copy(os.path.join(d,results),os.path.join(OUT,results.replace(".json","_rerun.json"))); shutil.copy(bak,os.path.join(d,results))
    out={"exit":r.returncode}; allok=r.returncode==0
    for path,t in zip(keys,tol):
        a=new; b=sealed
        for k in path: a=a[k]; b=b[k]
        out["/".join(path)]={"rerun":a,"sealed":b,"ok":abs(a-b)<=t}; allok&=abs(a-b)<=t
    out["pass"]=allok; return out
res["B2"]=rerun(V50,"run_val_050.py","VAL_050_RESULTS.json",[("primary_H1","cohens_d"),("primary_H1","auc")],[0.002,0.005])
print(f"B2 VAL-050: d {res['B2']['primary_H1/cohens_d']['rerun']:+.4f} (sealed {res['B2']['primary_H1/cohens_d']['sealed']:+.4f}) AUC {res['B2']['primary_H1/auc']['rerun']:.4f} -> {'PASS' if res['B2']['pass'] else 'FAIL'}")
res["B3"]=rerun(V51,"val051_analyze.py","VAL_051_RESULTS.json",[("H1_primary_ruleA","cohens_d"),("H1_primary_ruleA","auc"),("null_comparator_pooled_entropy","cohens_d")],[0.002,0.005,0.002])
hc=json.load(open(os.path.join(V51,"VAL_051_RESULTS.json")))["holdout_counts"]; res["B3"]["holdout"]=hc; res["B3"]["pass"]=res["B3"]["pass"] and hc.get("AD")==33 and hc.get("HC")==95
print(f"B3 VAL-051: RuleA d {res['B3']['H1_primary_ruleA/cohens_d']['rerun']:+.4f} AUC {res['B3']['H1_primary_ruleA/auc']['rerun']:.4f} null d {res['B3']['null_comparator_pooled_entropy/cohens_d']['rerun']:+.4f} holdout {hc} -> {'PASS' if res['B3']['pass'] else 'FAIL'}")
# B4
sys.path.insert(0,ENG); sys.path.insert(0,os.path.join(ENG,"Runtime Matrices/Directional Panel")); import pandas as pd, bidirectional_decomposition as bd
sealed=json.load(open(os.path.join(OUT,"VAL_051_RESULTS.json.sealed")))["per_sample_holdout"]
betas=json.load(open(os.path.join(V50,"aibl_imm_betas.json"))); man=json.load(open(os.path.join(V50,"aibl_manifest.json"))); g2s={m["gsm"]:m["title"].split("_whole")[0].split(" ")[0] for m in man}
P=bd.load_directional_panels(os.path.join(ENG,"Runtime Matrices/Directional Panel/directional_panels_v1_0.json"))["immune"]
statsA={r["cpg"]:r for r in json.load(open(os.path.join(V51,"val051_panel_ruleA.json")))["cpgs"]}
same=set(c.cpg_id for c in P.cpgs)==set(statsA) and all(abs(c.mean_hc_train-statsA[c.cpg_id]["mean_hc_train"])<1e-12 and abs(c.sd_hc_train-statsA[c.cpg_id]["sd_hc_train"])<1e-12 and c.direction==statsA[c.cpg_id]["direction"] for c in P.cpgs)
md=0;k=0
for r in sealed:
    a,_,_=bd.score_directional_composite(pd.Series(betas[g2s[r["gsm"]]]),P); k+=1; md=max(md,abs(a-r["A_dir_A"]))
res["B4"]={"n":k,"max_abs_diff":md,"runtime_panel_identical_to_sealed_ruleA":bool(same),"pass":bool(k==len(sealed) and md<1e-9 and same)}
print(f"B4 engine=seal: n={k} max|diff| {md:.1e}, runtime panel == Rule A: {same} -> {'PASS' if res['B4']['pass'] else 'FAIL'}")
# B5
geo=os.path.join(DATA,"GSE153712_normalized_average_betas.txt.gz")
if not os.path.exists(geo): res["B5"]={"pass":None,"note":"GSE153712_normalized_average_betas.txt.gz not in CPG_KIT_DATA - B5 not run; row 4.5 NOT commissioned"}; print("B5 raw GEO: file absent - SKIPPED (row not commissioned)")
else:
    cpgs=set(next(iter(betas.values())).keys()); print(f"B5: scanning GEO for {len(cpgs)} CpGs x {len(betas)} samples ...",flush=True)
    # GSE153712_normalized_average_betas.txt.gz is SAMPLES-AS-ROWS (first column = Sentrix ID, header = CpG ids) - the transpose of a
    # GEO series matrix. Read the header, keep the 18 IMM CpG column indices, then stream rows and keep the sealed samples.
    f=gzip.open(geo,"rt",errors="replace"); hdr=[c.strip('"') for c in next(f).rstrip("\n").split("\t")]
    idx={cg:k for k,cg in enumerate(hdr) if cg in cpgs}; rows={}; nrow=0
    for l in f:
        p_=l.rstrip("\n").split("\t"); sid=p_[0].strip('"'); nrow+=1
        key=sid if sid in betas else next((s for s in betas if sid.startswith(s) or s.startswith(sid)),None)
        if key is None: continue
        rows[key]={cg:p_[k] for cg,k in idx.items()}
    f.close()
    cols=hdr[1:]; want=[(s,None) for s in rows]
    cnt=0; nan=0; md=0.0
    for s,_ in want:
        for cg in cpgs:
            v=rows[s].get(cg,""); ref=betas[s].get(cg)
            if v in ("","NA","null") or ref is None: nan+=1; continue
            cnt+=1; md=max(md,abs(float(v)-float(ref)))
    res["B5"]={"header_cols":len(cols),"rows_in_file":nrow,"matched_samples":len(want),"cpgs_found":len(idx),"values_compared":cnt,"missing":nan,"max_abs_diff":md,"pass":bool(len(want)==len(betas) and len(idx)==len(cpgs) and md<1e-4)}
    print(f"B5 raw GEO: rows {nrow}, samples matched {len(want)}/{len(betas)}, cpgs {len(idx)}/{len(cpgs)}, values {cnt}, max|diff| {md:.2e} -> {'PASS' if res['B5']['pass'] else 'FAIL'}")
res["row_4_5_commissioned"]=bool(all(res[b]["pass"] for b in ("B1","B2","B3","B4")) and res["B5"]["pass"] is True)
json.dump(res,open(os.path.join(OUT,"proc_bidir_01.json"),"w"),indent=1,default=str); print("row 4.5 commissioned:",res["row_4_5_commissioned"])
