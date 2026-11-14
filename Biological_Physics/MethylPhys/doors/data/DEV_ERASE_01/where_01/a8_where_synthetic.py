"""Synthetic test of a8_where.py (before any library is recounted): planted knockout-only loose tiles in contiguous domains must read CLUSTERED;
the same number scattered at random must read not clustered; the null test must read near zero when the wild types are copies with noise."""
import os, sys, gzip, random, tempfile
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); import a8_where as A
def write(d, g, means):
    with gzip.open(os.path.join(d, f"{g}_tiles.csv.gz"), "wt") as f:
        f.write("chrom,tile,iso_a,opp_a,iso_b,opp_b,calls,mean_call\n")
        for (c, t), m in means.items(): f.write(f"{c},{t},0,0,0,0,1000,{m:.4f}\n")
def case(clustered, seed=3):
    r = random.Random(seed); keys = [(f"chr{c}", t) for c in range(1, 6) for t in range(220)]
    base = {k: (0.45 if r.random() < 0.07 else 0.85) + r.gauss(0, 0.02) for k in keys}
    hit = set()
    if clustered:
        while len(hit) < 100:
            c = f"chr{r.randint(1, 5)}"; s0 = r.randrange(0, 200)
            for t in range(s0, s0 + r.randint(5, 15)):
                if t < 220: hit.add((c, t))
    else: hit = set(r.sample(keys, 100))
    d = tempfile.mkdtemp()
    for g in A.WT: write(d, g, {k: v + r.gauss(0, 0.01) for k, v in base.items()})
    for g in A.KO: write(d, g, {k: (0.55 if k in hit else v) + r.gauss(0, 0.01) for k, v in base.items()})
    return A.main(d)
o1, p1 = case(True); o2, p2 = case(False)
ok = p1 < 0.01 and p2 > 0.05
print("SYNTHETIC", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
