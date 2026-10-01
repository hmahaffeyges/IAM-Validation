#!/usr/bin/env python3
"""PROC-SALMON-01 driver: per specimen (fish x tissue) - stream its runs from S3, Trim Galore --rrbs, Bismark (directional, bowtie2), extract.py.
8 specimens at a time x Bismark --parallel 4 -p 2 (~16 cores each). TEST=1: one specimen, first 2 M reads."""
import os, json, subprocess, pandas as pd, multiprocessing as mp, time
S="/home/ubuntu/data/salmon"; ENV=f"export PATH={S}/mm/bin:$PATH"; REF=f"{S}/ref"
U=json.load(open("fastq_urls.json")); M=pd.read_csv("methow_run_map.csv")
M["spec"]=M.fish.astype(str)+"_"+M.tissue+"_"+M.origin
SP=sorted(M.spec.unique()); TEST=os.environ.get("TEST")=="1"
if TEST: SP=SP[:1]
os.makedirs(f"{S}/out",exist_ok=True)
def one(sp):
    o=f"{S}/out/{sp}.parquet"; G=json.load(open("s3_get.json"))["spec"][sp]
    if not os.path.exists(o) and not TEST:
        if subprocess.run(["python3","s3io.py","get",G[".parquet"],o]).returncode==0:
            for x in ("_bismark_report.txt","_extract.log"): subprocess.run(["python3","s3io.py","get",G[x],f"{S}/out/{sp}{x}"])
            return sp,"restored from S3"
    if os.path.exists(o): return sp,"cached"
    w=f"{S}/w_{sp}"; os.makedirs(w,exist_ok=True); runs=sorted(M[M.spec==sp].run)
    half=",".join(f"{r}:{'AB'[i%2]}" for i,r in enumerate(runs)) if len(runs)>1 else "parity"
    head="| head -n 8000000" if TEST else ""
    cmd=f"""{ENV}; set -o pipefail; cd {w}
      for r in {' '.join(runs)}; do curl -s --retry 5 "$(python3 -c "import json;print(json.load(open('{os.getcwd()}/fastq_urls.json'))['$r'][0])")" | pigz -dc {head}; done | pigz -p 4 > {sp}.fq.gz
      trim_galore --rrbs --non_directional --cores 2 -q 20 --gzip {sp}.fq.gz > trim.log 2>&1
      bismark --genome {REF} --pbat --score_min L,0,-0.2 --parallel 4 -p 2 --bowtie2 -o . {sp}_trimmed.fq.gz > bismark.log 2>&1
      python {os.getcwd()}/extract.py {sp}_trimmed_bismark_bt2.bam {REF}/Omyk_1.0.fa {o}.tmp {half} > extract.log 2>&1 && mv {o}.tmp {o}
      cp *_SE_report.txt {S}/out/{sp}_bismark_report.txt; cp extract.log {S}/out/{sp}_extract.log; cd {S} && rm -rf {w}
      {'' if TEST else f'cd {os.getcwd()}; for x in .parquet _bismark_report.txt _extract.log; do python3 s3io.py put {S}/out/{sp}$x downloads/salmon_work/out/{sp}$x; done'}"""
    t0=time.time(); p=subprocess.run(["bash","-c",cmd],capture_output=True,text=True)
    return sp, ("ok %.0fs"%(time.time()-t0)) if p.returncode==0 else ("FAIL "+(p.stderr[-400:] or open(f'{w}/bismark.log').read()[-400:] if os.path.exists(f'{w}/bismark.log') else p.stderr[-400:]))
with mp.get_context("fork").Pool(1 if TEST else 8) as P:
    for sp,st in P.imap_unordered(one,SP): print(sp,st,flush=True)
os.system(f"cd {S}/out && tar czf {os.getcwd()}/salmon_tables.tgz *.parquet *_report.txt *_extract.log")
print("DONE",flush=True)
