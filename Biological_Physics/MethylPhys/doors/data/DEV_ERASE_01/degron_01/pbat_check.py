"""DEV-ERASE-01, the degron series: methylated share of one PBAT chunk's molecules (a .pat from the committed pipeline), first 2,000,000 lines
(added 2026-10-26 after the PBAT fix, disclosed). Usage: python3 pbat_check.py PAT.gz OUT.txt"""
import sys, gzip
C = T = mol = 0
with gzip.open(sys.argv[1], "rt") as g:
    for k, l in enumerate(g):
        x = l.split("\t"); s = x[2]; n = int(x[3]); C += n * s.count("C"); T += n * s.count("T"); mol += n
        if k > 2_000_000: break
L = f"{sys.argv[1].split('/')[-1]}: molecules {mol}, methylated share {C / (C + T):.3f}"
open(sys.argv[2], "w").write(L + "\n"); print(L)
