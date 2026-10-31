"""IAM Sequencing Atlas (plan 35.30): synthetic for published per-CpG tables (run_sample.py --build seq-table), written and committed before
it runs. The instrument's job: a table's reads land on the hg19 CpGs they came from, with the same counts.
Known answer: one spent atlas input (GSM5652316, .beta) written as an hg19 1-based per-CpG table (chrom, position, methylated, all; a header
line) and as hg19 Bismark coverage (chrom, start, end, percent, methylated, unmethylated), half the CpGs written on the reverse strand (the G,
position + 1), each read back by the one command. Bar: at least 0.999 of the CpGs with reads come back with the same counts, both formats.
Usage: python3 seq_table_synthetic.py BETA CPG_BED WORKDIR OUT.txt"""
import sys, os, gzip, json, subprocess, numpy as np, pandas as pd
BETA, BED, W, OUT = sys.argv[1:5]; os.makedirs(W, exist_ok=True)
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"); sys.path.insert(0, C)
import stage_seq_atlas as Q
a = np.fromfile(BETA, np.uint8).reshape(-1, 2).astype(np.int64); D = pd.read_csv(BED, sep="\t", header=None, names=["chr", "pos", "idx"], dtype={"chr": str})
have = a[D.idx.to_numpy() - 1, 1] > 0; T = D[have].copy(); T["m"] = a[T.idx - 1, 0]; T["n"] = a[T.idx - 1, 1]
rng = np.random.default_rng(20261106); rev = rng.random(len(T)) < 0.5; T["p"] = np.where(rev, T.pos + 1, T.pos)
f1 = os.path.join(W, "t.txt.gz"); f2 = os.path.join(W, "t.cov.gz")
with gzip.open(f1, "wt") as fo:
    fo.write("chromosome CpG_position M_counts Total_counts\n"); pd.DataFrame(dict(c=T.chr, p=T.p, m=T.m, n=T.n)).to_csv(fo, sep=" ", header=False, index=False)
with gzip.open(f2, "wt") as fo:
    pd.DataFrame(dict(c=T.chr.str.replace("chr", "", regex=False), s=T.p, e=T.p, pct=(100 * T.m / T.n).round(2), m=T.m, u=T.n - T.m)).to_csv(fo, sep="\t", header=False, index=False)
L = [f"seq-table synthetic: {len(T):,} CpGs with reads from GSM5652316, half on the reverse strand"]; ok = []
for f, fmt in ((f1, "cpg-table"), (f2, "bismark-cov")):
    o = os.path.join(W, fmt + ".lbeta")
    r = subprocess.run([sys.executable, "run_sample.py", "--build", "seq-table", "--build-input", f, "--build-out", o, "--build-samples", f"{fmt}:hg19:{BED}"],
                       cwd=os.path.join(C, "MethylPhys_Interface"), capture_output=True, text=True)
    if r.returncode or not os.path.exists(o): L.append(f"{fmt}: failed: {(r.stdout + r.stderr)[-300:]}"); ok.append(False); continue
    b = np.fromfile(o, np.uint16).reshape(-1, 2).astype(np.int64); i = T.idx.to_numpy() - 1
    same = float(((b[i, 0] == T.m.to_numpy()) & (b[i, 1] == T.n.to_numpy())).mean()); extra = int(((b[:, 1] > 0) & (a[:, 1] == 0)).sum())
    g = same >= 0.999 and extra == 0; ok.append(g); L.append(f"{fmt}: same counts {same:.5f}, reads where the source had none {extra} ({'met' if g else 'not met'})"); os.remove(o)
L.append(f"synthetic: {'met' if all(ok) else 'not met'}"); open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
