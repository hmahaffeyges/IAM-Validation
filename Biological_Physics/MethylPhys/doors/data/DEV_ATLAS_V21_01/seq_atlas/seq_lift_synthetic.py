"""IAM Sequencing Atlas (plan 35.30): synthetic for moving GRCh38 read tables to the hg19 numbering (written and committed before it runs).
The instrument's job: a donor's reads land on the same hg19 CpGs they came from, with the same counts.

Known answer: one spent atlas input's hg19 .beta (GSM5652316, healthy B cells; read as an atlas input already). Test construction (not the
chain): every hg19 CpG with reads is moved to GRCh38 (UCSC hg19ToHg38, pyliftover; positions that do not map once are left out and counted),
and written as a GRCh38 read table (chrom, 0-based position, methylated, all), with a random 30 % of sites written at the G of the CpG
(the reverse strand), seed 20261029. The one command (run_sample.py --build seq-lift) moves the table back to hg19.

Bars: 1. at least 95 % of the original CpGs with reads come back; 2. of those, at least 99 % come back with exactly the original counts;
3. at most 0.1 % of the CpGs with reads after the round trip had no reads before (reads put in the wrong place).
Usage: python3 seq_lift_synthetic.py BETA CPG_BED_GZ WORKDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np, pandas as pd
BETA, BED, W, OUT = sys.argv[1:5]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "chain")); sys.path.insert(0, CH)
from pyliftover import LiftOver
a = np.fromfile(BETA, np.uint8).reshape(-1, 2).astype(np.int64); assert len(a) == 28217448
D = pd.read_csv(BED, sep="\t", header=None, names=["chr", "pos", "idx"], dtype={"chr": str})
has = a[D.idx.to_numpy() - 1, 1] > 0; X = D[has].copy(); X["m"] = a[X.idx.to_numpy() - 1, 0]; X["n"] = a[X.idx.to_numpy() - 1, 1]
lo = LiftOver("hg19", "hg38"); rng = np.random.default_rng(20261029); rows = []; unmapped = 0
for c, g in X.groupby("chr", sort=False):
    for p1, m, n in zip(g.pos.to_numpy(), g.m.to_numpy(), g.n.to_numpy()):
        q = lo.convert_coordinate(c, int(p1) - 1)
        if len(q) != 1 or q[0][2] != "+": unmapped += 1; continue
        rows.append((q[0][0], q[0][1] + (1 if rng.random() < 0.3 else 0), m, n))
T = os.path.join(W, "syn_hg38.tsv"); pd.DataFrame(rows).to_csv(T, sep="\t", header=False, index=False)
pd.DataFrame(dict(gsm=["SYN"], file=[T])).to_csv(os.path.join(W, "syn_samples.csv"), index=False)
O = os.path.join(W, "syn_back.lbeta")
r = subprocess.run([sys.executable, "run_sample.py", "--build", "seq-lift", "--build-samples", os.path.join(W, "syn_samples.csv"), "--build-input", BED,
                    "--build-out", O, "--bundle", os.path.join(W, "syn_lift.json")], cwd=os.path.join(CH, "MethylPhys_Interface"), capture_output=True, text=True)
if r.returncode: print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit(1)
b = np.fromfile(O, np.uint16).reshape(-1, 2).astype(np.int64)
orig = a[:, 1] > 0; back = b[:, 1] > 0; came = orig & back
r1 = came.sum() / orig.sum(); r2 = (b[came] == a[came]).all(1).mean(); r3 = (back & ~orig).sum() / back.sum()
L = ["IAM Sequencing Atlas: GRCh38 to hg19 round trip (synthetic), moved back by run_sample.py --build seq-lift",
     f"known answer GSM5652316 (spent): {int(orig.sum()):,} CpGs with reads; {unmapped:,} did not move to GRCh38 once and were left out; 30 % written on the reverse strand",
     f"bar 1 came back: {r1:.4f} (at least 0.95): {'met' if r1 >= 0.95 else 'not met'}",
     f"bar 2 exact counts among those: {r2:.5f} (at least 0.99): {'met' if r2 >= 0.99 else 'not met'}",
     f"bar 3 reads in the wrong place: {r3:.5f} (at most 0.001): {'met' if r3 <= 0.001 else 'not met'}",
     f"synthetic: {'met' if r1 >= 0.95 and r2 >= 0.99 and r3 <= 0.001 else 'not met'}", "conversion report: " + json.dumps({k: v for k, v in json.load(open(os.path.join(W, "syn_lift.json"))).items() if k != "command"})]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
