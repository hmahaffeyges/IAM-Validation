"""PROC-CHANNEL-01 input: 399 random 2,000-CpG windows in hg19 CpG-index space (seed 20260930), as generated 2026-09-30.
Run: python3 make_windows.py  -> windows_hg19_cpgidx.bed"""
import numpy as np
sz=[("chr1",2284470),("chr2",2164335),("chr3",1623646),("chr4",1473930),("chr5",1506454),("chr6",1475569),("chr7",1568666),("chr8",1309135),("chr9",1226821),("chr10",1351291),("chr11",1289987),("chr12",1277218),("chr13",803708),("chr14",859779),("chr15",873464),("chr16",1097776),("chr17",1155600),("chr18",677214),("chr19",1057376),("chr20",717722),("chr21",380444),("chr22",578097)]
off=1; rows=[]; rng=np.random.default_rng(20260930); W=2000; tot=sum(n for _,n in sz)
for c,n in sz:
    k=max(4,round(400*n/tot)); starts=np.sort(rng.choice(np.arange(off+10000,off+n-10000-W,W),k,replace=False))
    rows+= [(c,int(s)-1,int(s)+W) for s in starts]; off+=n
open("windows_hg19_cpgidx.bed","w").write("".join(f"{c}\t{s}\t{e}\n" for c,s,e in rows)); print(len(rows),"windows,",len(rows)*W,"CpGs")
