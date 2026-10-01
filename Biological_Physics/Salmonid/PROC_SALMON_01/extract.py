#!/usr/bin/env python3
"""PROC-SALMON-01 extraction (pre-registered 2026-10-01, sha 4e65a4c06246cf56). One Bismark BAM -> per-site tables.
Per CpG site (reads' CpG calls, first/last 3 aligned bases ignored): m, t (all calls); per run-half h in {A,B}: opp_h, err_h from qualifying molecules
(>= 6 CpG calls, >= 80 % methylated): an interior CpG call is an opportunity; an unmethylated interior call with both neighbouring CpG calls methylated
is an isolated error. Also: non-CpG methylated/total (conversion failure) and A/T-reference mismatches/A/T positions (sequencing error)."""
import sys, pysam, numpy as np, pandas as pd, collections
bam, ref, out, halfmap = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]   # halfmap: comma list of read-group/run -> half via RG tag
HALF=dict(x.split(":") for x in halfmap.split(",")) if halfmap!="parity" else None
FA=pysam.FastaFile(ref); B=pysam.AlignmentFile(bam,"rb")
m=collections.Counter(); t=collections.Counter(); opp={"A":collections.Counter(),"B":collections.Counter()}; err={"A":collections.Counter(),"B":collections.Counter()}
nc_m=nc_t=at_mm=at_n=0; nq=0; nr=0; cur=None; seqc=None
for k,r in enumerate(B.fetch(until_eof=True)):
    if r.is_unmapped or r.is_secondary or r.is_supplementary: continue
    nr+=1
    h = HALF.get(r.query_name.split(".")[0].split("_")[0],"A") if HALF else ("A" if k%2==0 else "B")
    xm=r.get_tag("XM"); pairs=r.get_aligned_pairs(matches_only=True); L=len(pairs)
    if r.reference_name!=cur: cur=r.reference_name; seqc=None
    cg=[]
    q=r.query_sequence; sub = (k%20==0)   # sequencing error on a 1-in-20 subsample of reads
    rs=FA.fetch(r.reference_name, r.reference_start, r.reference_end).upper() if sub else ""
    for j,(qi,ri) in enumerate(pairs):
        if j<3 or j>=L-3: continue
        c=xm[qi]
        if c in "Zz":
            pos=(r.reference_id<<32) | (ri if r.get_tag('XG')=='CT' else ri-1)     # collapse both strands onto the C of the CpG; reference id in high bits
            cg.append((pos, c=="Z"))
        elif c in "XxHh":
            nc_t+=1; nc_m+= c in "XH"
        if not sub: continue
        rb=rs[ri-r.reference_start] if 0<=ri-r.reference_start<len(rs) else "N"
        if rb in "AT":
            at_n+=1; at_mm+= q[qi]!=rb
    for pos,me in cg: t[pos]+=1; m[pos]+=me
    n=len(cg)
    if n>=6 and sum(me for _,me in cg)>=0.8*n:
        nq+=1
        for i in range(1,n-1):
            pos,me=cg[i]; opp[h][pos]+=1
            if (not me) and cg[i-1][1] and cg[i+1][1]: err[h][pos]+=1
sites=sorted(t)
# site key: reference id in high bits
D=pd.DataFrame(dict(pos=np.array(sites,dtype=np.int64),m=[m[s] for s in sites],t=[t[s] for s in sites],
    opp_A=[opp["A"][s] for s in sites],err_A=[err["A"][s] for s in sites],opp_B=[opp["B"][s] for s in sites],err_B=[err["B"][s] for s in sites]))
D.to_parquet(out)
print(dict(reads=nr,qualifying=nq,sites=len(D),conv_fail=nc_m/max(nc_t,1),sub_err=at_mm/max(at_n,1),
           eps=(D.err_A.sum()+D.err_B.sum())/max(D.opp_A.sum()+D.opp_B.sum(),1)), flush=True)
