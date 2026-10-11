"""DEV-IAMA-COMMISSION-01 synthetic test 2: does the same-run tare remove a laboratory/kit offset on identical DNA (test C3)? One healthy Loyfer
granulocyte whole file (GSM5652313), 5 % and 20 % of its molecules drawn at random (about 1.1 and 4.6 million per library, the depth at which synthetic
test 1 put the counting floor below 1 %). Each drawn molecule goes at random (seeded) to one of 14 pseudo-libraries: run A and run B x 7 lines.
Line effect: line k (k = 0..6) carries extra planted copy error, each methylated call (C) turned to T with probability 0.002 k, the same in both
runs (the 7 EpiQC cell lines differ). Run artefact (run B only, as measured for Swift against TruSeq, DEV-IAMA-KIT-01: extra errors clustered by
region): in a fixed random 20 % of genome bins (CpG index // 500,000), each C turned to T with extra probability p (p = 0.02 and 0.04, two
separate run Bs). Every library is read by Stage Q's own pat_site_table; IAM-A of a library = H(eps) / H(eps of run A line 0); tared
IAM-A_rel = A / median(A of the other 6 lines in its run) (stage_q_iam_a.tare rule, refs exclude the sample itself).
Pass (written before running): for both p, every line's |A_rel(run B) - A_rel(run A)| <= 0.03, while the untared run offset is reported.
Writes offset_tare_02_output.csv and offset_tare_02_summary.txt. Usage: python3 offset_tare_02.py CHAIN_DIR PAT TMPDIR"""
import sys, os, gzip, math, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import stage_q_iam_a as Q
PAT, TMP = sys.argv[2], sys.argv[3]; os.makedirs(TMP, exist_ok=True); HERE = os.path.dirname(os.path.abspath(__file__))
H = lambda e: -(e * math.log2(e) + (1 - e) * math.log2(1 - e))
rg = np.random.default_rng(20261011); LINE = [0.002 * k for k in range(7)]; PS = (0.02, 0.04); BINSZ = 500_000
bad = set(np.flatnonzero(np.random.default_rng(7).random(64) < 0.20))          # fixed artefact bins (bins 0..63 cover the hg19 CpG index)
runs = ["A"] + [f"B{p}" for p in PS]
outs = {(r, k): gzip.open(os.path.join(TMP, f"{r}_{k}.pat.gz"), "wt", compresslevel=1) for r in runs for k in range(7)}
import random
rr = random.Random(20261011); SHARE = float(os.environ.get("SHARE", "0.05"))   # run at 0.05 and 0.20 (outputs *_share05_*, *_share20_*)
def flip(pat, prob):
    if prob <= 0: return pat
    return "".join("T" if ch == "C" and rr.random() < prob else ch for ch in pat)
with gzip.open(PAT, "rt") as fh:
    for l in fh:
        q = l.rstrip("\n").split("\t"); n = int(q[3]); m = int(rg.binomial(n, SHARE)) if n > 1 else int(rr.random() < SHARE)
        if not m: continue
        inbad = (int(q[1]) // BINSZ) in bad
        for _ in range(m):
            j = rr.randrange(14); k = j % 7; r = "A" if j < 7 else "B"
            base = flip(q[2], LINE[k])
            if r == "A": outs[("A", k)].write(f"{q[0]}\t{q[1]}\t{base}\t1\n")
            else:
                for p in PS: outs[(f"B{p}", k)].write(f"{q[0]}\t{q[1]}\t{flip(base, p) if inbad else base}\t1\n")
for g in outs.values(): g.close()
rows = []
for r in runs:
    for k in range(7):
        p_ = os.path.join(TMP, f"{r}_{k}.pat.gz"); T = Q.pat_site_table(p_)
        e = float(T.err_A.sum() + T.err_B.sum()) / float(T.opp_A.sum() + T.opp_B.sum()); rows.append(dict(run=r, line=k, eps=e, opp=int(T.opp_A.sum() + T.opp_B.sum()))); os.remove(p_)
        print(rows[-1], flush=True)
R = pd.DataFrame(rows); e0 = R[(R.run == "A") & (R.line == 0)].eps.iloc[0]; R["A"] = [H(x) / H(e0) for x in R.eps]
R["A_rel"] = [a / np.median(R[(R.run == r) & (R.line != k)].A) for r, k, a in zip(R.run, R.line, R.A)]
R.to_csv(os.path.join(HERE, f"offset_tare_02_share{int(SHARE*100):02d}_output.csv"), index=False)
out = []; ok = True
for p in PS:
    a = R[R.run == "A"].set_index("line"); b = R[R.run == f"B{p}"].set_index("line")
    d_raw = (b.A / a.A); d_tar = (b.A_rel - a.A_rel).abs(); ok &= bool((d_tar <= 0.03).all())
    out.append(f"p {p}: untared run offset B/A {d_raw.min():.4f}-{d_raw.max():.4f}; tared |A_rel(B) - A_rel(A)| max {d_tar.max():.4f}")
out.append(f"line spread in run A, A {R[R.run == 'A'].A.min():.4f}-{R[R.run == 'A'].A.max():.4f}")
out.append("SYNTHETIC TEST 2 " + ("PASSED" if ok else "FAILED"))
open(os.path.join(HERE, f"offset_tare_02_share{int(SHARE*100):02d}_summary.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
