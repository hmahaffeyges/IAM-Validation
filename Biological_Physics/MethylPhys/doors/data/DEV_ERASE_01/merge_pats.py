"""Split alignment, merge (2026-10-24, plan 49.8). One sample's chunk molecule files (.pat.gz, each sorted by its genome-wide CpG index, column 2)
merged into one file in the same order and format, identical molecules (same start and pattern) summed, so the one command reads the whole
sample once. Duplicate marking was done within each chunk (sambamba markdup per chunk); a duplicate pair that fell in two chunks is kept,
disclosed with every split reading. Usage: python3 merge_pats.py OUT.pat.gz CHUNK1.pat.gz [CHUNK2.pat.gz ...]"""
import sys, gzip, heapq
def rows(p):
    with gzip.open(p, "rt") as f:
        for l in f:
            c = l.rstrip("\n").split("\t")
            if len(c) >= 4: yield (int(c[1]), c[2], c[0], int(c[3]))
out, ins = sys.argv[1], sys.argv[2:]; n = mol = 0; prev = None
with gzip.open(out, "wt") as o:
    for r in heapq.merge(*[rows(p) for p in ins], key=lambda r: (r[0], r[1])):
        if prev is not None and (r[0], r[1], r[2]) == (prev[0], prev[1], prev[2]): prev = (prev[0], prev[1], prev[2], prev[3] + r[3]); continue
        if prev is not None: o.write(f"{prev[2]}\t{prev[0]}\t{prev[1]}\t{prev[3]}\n"); n += 1; mol += prev[3]
        prev = r
    if prev is not None: o.write(f"{prev[2]}\t{prev[0]}\t{prev[1]}\t{prev[3]}\n"); n += 1; mol += prev[3]
print(f"merged {len(ins)} chunks: {n} lines, {mol} molecules")
