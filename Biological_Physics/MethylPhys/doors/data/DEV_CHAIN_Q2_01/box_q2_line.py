"""DEV-CHAIN-Q2-01 box driver: the self-reference line's inputs counted through chain Stage Q2 (a reproduction, not a new test).
 1. each Loyfer sample in loyfer_samples.csv: its hg19 .pat.gz streamed from GEO (never stored), counted by Q2.count_pat on Q2.hg19_tiles(),
    the stream's sha256 recorded as the input fingerprint, a stamped JSON written per sample
 2. each ENCODE file in encode_samples.tsv: Q2.count_bam on GRCh38, input sha256 recorded, a stamped JSON per file
Usage: python box_q2_line.py loyfer <workers> | encode <workers>   (run from a folder holding the chain's .py files under ./chain)"""
import os, sys, re, json, gzip, hashlib, io, urllib.request, multiprocessing as mp, csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chain")); import stage_q2_selfref as Q2
OUT = "out"; os.makedirs(OUT, exist_ok=True)
class HashStream(io.RawIOBase):
    def __init__(self, r): self.r = r; self.h = hashlib.sha256(); self.n = 0
    def readable(self): return True
    def readinto(self, b):
        d = self.r.read(len(b)); self.h.update(d); self.n += len(d); b[:len(d)] = d; return len(d)
def pat_name(g):
    t = urllib.request.urlopen(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/", timeout=120).read().decode()
    return [x for x in re.findall(r'href="([^"]+\.pat\.gz)"', t) if ".hg38." not in x][0]
TILES = None
def loyfer_one(row):
    g, cell = row; p = os.path.join(OUT, f"{g}.json")
    if os.path.exists(p): return g, "exists"
    for attempt in range(4):
        try:
            name = pat_name(g); u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/{name}"
            with urllib.request.urlopen(u, timeout=900) as r:
                hs = HashStream(r)
                with gzip.open(io.BufferedReader(hs, 1 << 20), "rt") as fh: o = Q2.count_pat(fh, TILES)
            o.update(sample=g, cell=cell, file=name)
            Q2.write_json(o, p, inputs=[dict(name=name, url=u, sha256=hs.h.hexdigest(), size=hs.n)]); return g, "ok"
        except Exception as e: err = f"{type(e).__name__}: {str(e)[:100]}"
    return g, "FAILED " + err
if __name__ == "__main__" and sys.argv[1] == "loyfer":
    TILES = Q2.hg19_tiles(); rows = [(r["sample"], r["cell"]) for r in csv.DictReader(open("loyfer_samples.csv"))]
    with mp.get_context("fork").Pool(int(sys.argv[2])) as P:
        for g, st in P.imap_unordered(loyfer_one, rows): print(g, st, flush=True)
elif __name__ == "__main__" and sys.argv[1] == "encode":
    for r in csv.DictReader(open("encode_samples.tsv"), delimiter="\t"):
        acc = r["file"]; p = os.path.join(OUT, f"{acc}.json")
        if os.path.exists(p): continue
        o = Q2.count_bam(f"{acc}.bam", "ref.fa", int(sys.argv[2]), "GRCh38"); o.update(sample=acc, cell=r["cell"], loyfer_cell=r["loyfer_cell"], experiment=r["experiment"])
        Q2.write_json(o, p, inputs=[dict(name=f"{acc}.bam", url=r["url"], sha256=Q2.file_sha(f"{acc}.bam"), size=os.path.getsize(f"{acc}.bam")),
                                    dict(name="GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta", sha256=Q2.file_sha("ref.fa"))]); print(acc, "ok", flush=True)
