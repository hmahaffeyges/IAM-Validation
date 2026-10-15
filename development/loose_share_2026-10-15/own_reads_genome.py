"""PHY-049 whole genome (2026-10-15): the loose share and held error from each Loyfer sample's own molecules across the WHOLE genome.
Same counter as own_reads_loose.py (imported, unchanged); the only change is the windows: the genome tiled into consecutive 2,000-CpG
windows in hg19 CpG-index space (the chromosome sizes of PROC-CHANNEL-01's make_windows.py), instead of 399 random ones. Each sample's
hg19 .pat.gz is streamed from GEO and never stored.
Usage: python own_reads_genome.py tiles            (write tiles_hg19_cpgidx.bed and check it)
       python own_reads_genome.py real <samples.txt> <workers>"""
import os, sys, gzip, io, json, re, urllib.request, multiprocessing as mp, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import own_reads_loose as OR
SZ = [("chr1",2284470),("chr2",2164335),("chr3",1623646),("chr4",1473930),("chr5",1506454),("chr6",1475569),("chr7",1568666),("chr8",1309135),
      ("chr9",1226821),("chr10",1351291),("chr11",1289987),("chr12",1277218),("chr13",803708),("chr14",859154),("chr15",825117),("chr16",1047108),
      ("chr17",1138298),("chr18",650016),("chr19",1009963),("chr20",693212),("chr21",340133),("chr22",554124)]
W = 2000
def tiles():
    rows = []; off = 1
    for c, n in SZ:
        for s in range(off, off + n - W + 1, W): rows.append((c, s - 1, s - 1 + W))
        off += n
    return rows
def fetch_name(g):
    t = urllib.request.urlopen(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/", timeout=120).read().decode()
    n = [x for x in re.findall(r'href="([^"]+\.pat\.gz)"', t) if ".hg38." not in x]; return n[0]
def run_one(g):
    try:
        name = fetch_name(g); u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/{name}"
        with urllib.request.urlopen(u, timeout=600) as r, gzip.open(r, "rt") as fh: o = OR.one(fh, WIN)
        return dict(sample=g, file=name, **o)
    except Exception as e: return dict(sample=g, error=f"{type(e).__name__}: {str(e)[:120]}")
if sys.argv[1] == "tiles":
    T = tiles(); pd.DataFrame(T).to_csv(os.path.join(H, "tiles_hg19_cpgidx.bed"), sep="\t", header=False, index=False)
    old = pd.read_csv(os.path.join(H, "windows_hg19_cpgidx.bed"), sep="\t", header=None)
    in_chr = {c: (o, o + n) for (c, n), o in zip(SZ, np.cumsum([1] + [n for _, n in SZ])[:-1])}
    ok = all(in_chr[c][0] - 1 <= s and e <= in_chr[c][1] for c, s, e in T) and all(T[i][2] <= T[i + 1][1] or T[i][0] != T[i + 1][0] for i in range(len(T) - 1))
    inside = all(any(c == tc and ts <= s + 1 and e - 1 <= te for tc, ts, te in T if tc == c and ts <= s + W) for c, s, e in old.values[:20])
    print(f"tiles {len(T)}; inside their chromosomes and not overlapping: {ok}; CpGs covered {len(T) * W:,} of {sum(n for _, n in SZ):,}")
else:
    WIN = OR.windows(os.path.join(H, "tiles_hg19_cpgidx.bed")); G = [l.strip() for l in open(sys.argv[2]) if l.strip()]
    with mp.Pool(int(sys.argv[3])) as P:
        R = []
        for o in P.imap_unordered(run_one, G): R.append(o); print(o, flush=True)
    pd.DataFrame(R).to_csv("own_reads_genome.csv", index=False)
