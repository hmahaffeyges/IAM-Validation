"""DEV-SAM-LEVER-01: the RRBS reader's measured response to a known scattered loss (before any knockout is read).
Each methylated CpG call (read base C followed by G, inside the reader's called window) is turned to T with probability δ (seeded),
which is what a lowered restore rate does at copying. The modified reads go through boxruns/xspecies/rrbs_iama.run unchanged.
Reported per δ: ε measured, the true ε (ε_0 + δ(1 − ε_0)), IAM-A_rel measured = H(ε)/H(ε_0), and the qualifying-read share.
Run: python3 insilico_sam_01.py READS.fastq.gz OUT.csv δ [δ ...]"""
import os, sys, gzip, math, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "../../../boxruns/xspecies")); import rrbs_iama as R
def H(e): return -(e * math.log2(e) + (1 - e) * math.log2(1 - e))
lines = [l for l in gzip.open(sys.argv[1], "rt")]; out = sys.argv[2]; ds = [float(x) for x in sys.argv[3:]]
def planted(d, seed=20261010):
    rg = random.Random(seed); o = []
    for i, l in enumerate(lines):
        if i % 4 != 1 or d == 0: o.append(l); continue
        s = list(l.rstrip("\n")); L = min(len(s), 63)
        for j in range(R.SKIP5, L - R.SKIP3 - 1):
            if s[j] == "C" and s[j + 1] == "G" and rg.random() < d: s[j] = "T"
        o.append("".join(s) + "\n")
    return o
rows = []; base = None
for d in ds:
    r = R.run(planted(d)); r["delta"] = d
    if base is None: base = r
    e0 = base["eps"]; r["eps_true"] = e0 + d * (1 - e0); r["IAMA_rel"] = H(r["eps"]) / H(e0); r["IAMA_rel_true"] = H(r["eps_true"]) / H(e0)
    r["share_qualifying"] = (r["qualifying"] / r["reads_in_groups"]) / (base["qualifying"] / base["reads_in_groups"]); rows.append(r); print(r, flush=True)
import csv
with open(out, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
