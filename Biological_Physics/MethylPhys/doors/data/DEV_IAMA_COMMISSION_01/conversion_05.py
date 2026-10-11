"""DEV-IAMA-COMMISSION-01 synthetic test 5: does the derived conversion correction eps / c hold through Stage Q (test C1)? Real healthy molecules
(GSM5652313, 20 % drawn at random, seeded). Incomplete conversion leaves an unmethylated CpG reading C: every T call is turned to C with
probability 1 - c, for c = 1 (none), 0.995, 0.99, 0.98, 0.97. Each file is read by Stage Q's own pat_site_table (>= 6 calls, >= 80 % C, isolated
errors). The derivation (DEV-IAMA-CONVERSION-01) predicts eps(c) / eps(1) = c. Pass (written before running): the measured ratio within 0.005 of
c at every c, so the correction can be applied as derived; otherwise the measured response (not c) is what C1 must use.
Writes conversion_05_output.csv and conversion_05_summary.txt. Usage: python3 conversion_05.py CHAIN_DIR PAT TMPDIR"""
import sys, os, gzip, math, random, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import stage_q_iam_a as Q
PAT, TMP = sys.argv[2], sys.argv[3]; os.makedirs(TMP, exist_ok=True); HERE = os.path.dirname(os.path.abspath(__file__))
H = lambda e: -(e * math.log2(e) + (1 - e) * math.log2(1 - e)); C = (1.0, 0.995, 0.99, 0.98, 0.97); SHARE = 0.20
rg = np.random.default_rng(20261011); rr = random.Random(20261011)
outs = {c: gzip.open(os.path.join(TMP, f"c{c}.pat.gz"), "wt", compresslevel=1) for c in C}
def unconv(pat, miss):
    if miss <= 0: return pat
    return "".join("C" if ch == "T" and rr.random() < miss else ch for ch in pat)
with gzip.open(PAT, "rt") as fh:
    for l in fh:
        q = l.rstrip("\n").split("\t"); n = int(q[3]); m = int(rg.binomial(n, SHARE)) if n > 1 else int(rr.random() < SHARE)
        for _ in range(m):
            for c in C: outs[c].write(f"{q[0]}\t{q[1]}\t{unconv(q[2], 1 - c)}\t1\n")
for g in outs.values(): g.close()
rows = []
for c in C:
    p_ = os.path.join(TMP, f"c{c}.pat.gz"); T = Q.pat_site_table(p_)
    rows.append(dict(c=c, eps=float(T.err_A.sum() + T.err_B.sum()) / float(T.opp_A.sum() + T.opp_B.sum()), opp=int(T.opp_A.sum() + T.opp_B.sum()))); os.remove(p_); print(rows[-1], flush=True)
R = pd.DataFrame(rows); e1 = R.eps.iloc[0]; R["eps_ratio"] = R.eps / e1; R["IAMA_rel_raw"] = [H(e) / H(e1) for e in R.eps]
R["IAMA_rel_corrected"] = [H(e / c) / H(e1) for e, c in zip(R.eps, R.c)]; R["opp_ratio"] = R.opp / R.opp.iloc[0]
R.to_csv(os.path.join(HERE, "conversion_05_output.csv"), index=False)
ok = bool(((R.eps_ratio - R.c).abs() <= 0.005).all())
out = [f"c {r.c}: eps ratio {r.eps_ratio:.4f} (derived {r.c}), IAM-A raw {r.IAMA_rel_raw:.4f}, after eps/c {r.IAMA_rel_corrected:.4f}, opportunities x{r.opp_ratio:.4f}" for r in R.itertuples()]
out.append("SYNTHETIC TEST 5 " + ("PASSED: eps/c holds through Stage Q" if ok else "FAILED: Stage Q's response differs from eps/c; C1 uses the measured response"))
open(os.path.join(HERE, "conversion_05_summary.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
