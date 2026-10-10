mkdir -p remote_jobs/charr && cd remote_jobs/charr && cp ../salmon/s3io.py . && cp ../../charr/charr_runs.csv . && cat > setup_charr.sh <<'EOF'
#!/bin/bash
# PROC-CHARR-01 setup: reuse the salmon micromamba tools; brook charr ASM2944872v1 (GCF_029448725.1) + Bismark index; cached to S3.
set -euo pipefail
S=/home/ubuntu/data/salmon; C=/home/ubuntu/data/charr; W0=$(pwd); mkdir -p $C/ref; export PATH=$S/mm/bin:$PATH
cd $C
if [ ! -s ref/Bisulfite_Genome/GA_conversion/BS_GA.rev.2.bt2l ] && [ ! -s ref/Bisulfite_Genome/GA_conversion/BS_GA.rev.2.bt2 ]; then
  if python3 $W0/s3io.py getsplit ref $C/ref.tgz; then tar xzf ref.tgz && rm -f ref.tgz && touch .cached && echo RESTORED_FROM_S3; fi
fi
cd ref
if [ ! -s charr.fa ]; then
  curl -s -o g.fna.gz https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/029/448/725/GCF_029448725.1_ASM2944872v1/GCF_029448725.1_ASM2944872v1_genomic.fna.gz
  pigz -dc g.fna.gz > charr.fa && rm g.fna.gz
fi
[ -s charr.fa.fai ] || samtools faidx charr.fa
grep -c ">" charr.fa; awk '{s+=$2} END {print s/1e9" Gb"}' charr.fa.fai
if [ ! -s Bisulfite_Genome/GA_conversion/BS_GA.rev.2.bt2l ] && [ ! -s Bisulfite_Genome/GA_conversion/BS_GA.rev.2.bt2 ]; then
  bismark_genome_preparation --parallel ${PREP:-14} --bowtie2 . > prep.log 2>&1
fi
ls Bisulfite_Genome/*/ | head -4
if [ ! -f $C/.cached ]; then cd $C && tar cf - ref | pigz -p 16 > $C/ref.tgz && python3 $W0/s3io.py putsplit $C/ref.tgz downloads/charr_work/ref.tgz && touch $C/.cached && rm -f $C/ref.tgz* && echo CACHED_TO_S3; fi
echo SETUP_DONE
EOF
cat > extract_pe.py <<'EOF'
#!/usr/bin/env python3
"""PROC-CHARR-01 extraction (pre-registered 2026-10-01, sha 45d466dbb10630ff). Deduplicated paired Bismark BAM (mates adjacent) -> per-site table.
Molecule = read pair; CpG calls of both mates merged by position (overlap counted once; a position with conflicting calls is dropped).
First/last IGN aligned bases of each mate ignored (M-bias). Qualifying molecule: >= 6 CpG calls, >= 80 % methylated; interior call = opportunity;
unmethylated interior call with both neighbours methylated = isolated error. Run-half from the run accession in the read name.
Also: non-CpG methylated/total (conversion failure); A/T-reference mismatch rate on a 1-in-20 subsample (sequencing error)."""
import sys, os, pysam, numpy as np, pandas as pd, collections
bam, ref, out, halfmap = sys.argv[1:5]; IGN=int(os.environ.get("IGN","3"))
HALF=dict(x.split(":") for x in halfmap.split(","))
FA=pysam.FastaFile(ref); B=pysam.AlignmentFile(bam,"rb")
m=collections.Counter(); t=collections.Counter(); opp={h:collections.Counter() for h in "AB"}; err={h:collections.Counter() for h in "AB"}
S=dict(nc_m=0,nc_t=0,at_mm=0,at_n=0,nq=0,npair=0)
def calls(r,k):
    xm=r.get_tag("XM"); P=r.get_aligned_pairs(matches_only=True); L=len(P); cg={}
    sub=(k%20==0); rs=FA.fetch(r.reference_name,r.reference_start,r.reference_end).upper() if sub else ""; q=r.query_sequence
    top=r.get_tag("XG")=="CT"
    for j,(qi,ri) in enumerate(P):
        if j<IGN or j>=L-IGN: continue
        c=xm[qi]
        if c in "Zz": cg[(r.reference_id<<32)|(ri if top else ri-1)]=(c=="Z")
        elif c in "XxHh": S["nc_t"]+=1; S["nc_m"]+= c in "XH"
        if sub:
            o=ri-r.reference_start
            if 0<=o<len(rs) and rs[o] in "AT": S["at_n"]+=1; S["at_mm"]+= q[qi]!=rs[o]
    return cg
def molecule(rs,k):
    h=HALF.get(rs[0].query_name.split(".")[0].split("_")[0],"A"); a={}; bad=set()
    for r in rs:
        for p,me in calls(r,k).items():
            if p in a and a[p]!=me: bad.add(p)
            a[p]=me
    cg=sorted((p,me) for p,me in a.items() if p not in bad); S["npair"]+=1
    for p,me in cg: t[p]+=1; m[p]+=me
    n=len(cg)
    if n>=6 and sum(me for _,me in cg)>=0.8*n:
        S["nq"]+=1
        for i in range(1,n-1):
            p,me=cg[i]; opp[h][p]+=1
            if (not me) and cg[i-1][1] and cg[i+1][1]: err[h][p]+=1
buf=[]; k=0
for r in B.fetch(until_eof=True):
    if r.is_unmapped or r.is_secondary or r.is_supplementary: continue
    if buf and r.query_name!=buf[0].query_name: molecule(buf,k); k+=1; buf=[]
    buf.append(r)
if buf: molecule(buf,k)
sites=sorted(t)
D=pd.DataFrame(dict(pos=np.array(sites,dtype=np.int64),m=[m[s] for s in sites],t=[t[s] for s in sites],
    opp_A=[opp["A"][s] for s in sites],err_A=[err["A"][s] for s in sites],opp_B=[opp["B"][s] for s in sites],err_B=[err["B"][s] for s in sites]))
D.to_parquet(out)
print(dict(pairs=S["npair"],qualifying=S["nq"],sites=len(D),conv_fail=S["nc_m"]/max(S["nc_t"],1),sub_err=S["at_mm"]/max(S["at_n"],1)),flush=True)
EOF
cat > run_charr.py <<'EOF'
#!/usr/bin/env python3
"""PROC-CHARR-01 driver. Per fish: first NP read pairs of each run streamed from ENA, Trim Galore --paired, Bismark (directional, paired),
deduplicate_bismark, extract_pe.py; outputs to S3. TEST=1: one fish, alignment timing only (no error statistic computed)."""
import os, json, subprocess, pandas as pd, multiprocessing as mp, time
S="/home/ubuntu/data/salmon"; C="/home/ubuntu/data/charr"; ENV=f"export PATH={S}/mm/bin:$PATH"; REF=f"{C}/ref"; HERE=os.getcwd()
R=pd.read_csv("charr_runs.csv"); NP=int(float(os.environ.get("NP","2e6"))); TEST=os.environ.get("TEST")=="1"; PAR=int(os.environ.get("FISH_PAR","4"))
G=json.load(open("s3_get.json")).get("spec",{})
FISH=sorted(R.library_name.unique()); FISH=FISH[:1] if TEST else FISH
os.makedirs(f"{C}/out",exist_ok=True)
def one(f):
    o=f"{C}/out/{f}.parquet"
    if not TEST and not os.path.exists(o) and f in G and subprocess.run(["python3","s3io.py","get",G[f][".parquet"],o]).returncode==0:
        for x in ("_report.txt","_extract.log"): subprocess.run(["python3","s3io.py","get",G[f][x],f"{C}/out/{f}{x}"])
        return f,"restored from S3"
    if os.path.exists(o): return f,"cached"
    rr=R[R.library_name==f].sort_values("run_accession"); runs=list(rr.run_accession); n=max(NP//len(runs),1)
    half=",".join(f"{r}:{'AB'[i%2]}" for i,r in enumerate(runs))
    w=f"{C}/w_{f}"; os.makedirs(w,exist_ok=True); fetch=""
    for _,x in rr.iterrows():
        u1,u2=x.fastq_ftp.split(";")[:2]
        fetch+=f"(curl -s --retry 5 http://{u1} | pigz -dc | head -n {4*n} || true) >> R1.fq\n(curl -s --retry 5 http://{u2} | pigz -dc | head -n {4*n} || true) >> R2.fq\n"
    ext="" if TEST else f"""python {HERE}/extract_pe.py R1_val_1_bismark_bt2_pe.deduplicated.bam {REF}/charr.fa {o}.tmp {half} > extract.log 2>&1 && mv {o}.tmp {o}
      cat *_PE_report.txt *.deduplication_report.txt > {C}/out/{f}_report.txt; cp extract.log {C}/out/{f}_extract.log
      cd {HERE}; for x in .parquet _report.txt _extract.log; do python3 s3io.py put {C}/out/{f}$x downloads/charr_work/out/{f}$x; done"""
    cmd=f"""{ENV}; cd {w}; rm -f R1.fq R2.fq
      {fetch}
      echo pairs $(( $(wc -l < R1.fq) / 4 )) $(( $(wc -l < R2.fq) / 4 ))
      trim_galore --paired --cores 2 -q 20 R1.fq R2.fq > trim.log 2>&1 && rm -f R1.fq R2.fq
      T0=$(date +%s); bismark --genome {REF} --parallel 4 -p 2 --bowtie2 -o . -1 R1_val_1.fq -2 R2_val_2.fq > bismark.log 2>&1; echo align_s $(( $(date +%s) - T0 ))
      grep -E "Mapping efficiency|Sequence pairs analysed" *_PE_report.txt
      deduplicate_bismark -p --bam R1_val_1_bismark_bt2_pe.bam > dedup.log 2>&1; grep -E "Total number duplicated|removed" *.deduplication_report.txt | head -2
      {ext}
      {'' if TEST else f'rm -rf {w}'}"""
    t0=time.time(); p=subprocess.run(["bash","-c",cmd],capture_output=True,text=True)
    return f, ("ok %.0fs | "%(time.time()-t0))+p.stdout.strip().replace("\n"," | ")[-600:] + ("" if p.returncode==0 else " | ERR "+p.stderr[-400:])
with mp.get_context("fork").Pool(1 if TEST else PAR) as P:
    for f,st in P.imap_unordered(one,FISH): print(f,st,flush=True)
os.system(f"cd {C}/out && tar czf {HERE}/charr_tables.tgz *.parquet *_report.txt *_extract.log 2>/dev/null")
print("DONE",flush=True)
EOF
for f in extract_pe.py run_charr.py; do python3 -c "import ast;ast.parse(open('$f').read())" && echo "$f ok"; done; bash -n setup_charr.sh && echo sh-ok