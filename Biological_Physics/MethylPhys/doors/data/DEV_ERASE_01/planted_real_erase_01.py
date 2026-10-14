"""DEV-ERASE-01 planted check on REAL healthy molecules. Written and committed 2026-10-14 BEFORE it runs.
Question: on real sequencing molecules, with their real error clustering, missing calls and read structure, does Stage Q's own counting
(chain/stage_q_iam_a.py pat_site_table, pipeline loyfer_pat_v1) read a planted halving as a halving and a planted doubling as a doubling?
Files: two healthy Loyfer .pat.gz files already spent for development (GSM5652313 blood granulocytes, GSM5652281 blood CD4 T cells), the
first 100,000,000 compressed bytes of each (the check is of the counting, not of a cell's value; disclosed). Every molecule line with count n
is expanded into n molecules so each is planted independently. Calls are taken after dropping '.', as Stage Q does. Seeds fixed.
ARMS, per file:
 base     the molecules unchanged.
 halved   every isolated error (a T whose two neighbouring calls are C) is turned back to C with probability 1/2, on every molecule:
          the eraser-removed picture, half the errors gone.
 doubled  every interior C whose two neighbouring calls are C is turned to T with probability eps_base (the base file's Stage Q reading):
          about as many new isolated errors again (the boosted-eraser picture, e.g. vitamin C).
BARS (each file; MET only if all hold for both):
 P1 halved / base within 0.45-0.55 (the bar the synthetic test used).
 P2 doubled / base within 1.80-2.20 (new errors next to old ones are not isolated, so slightly below 2 is expected).
 P3 the base file's two halves (odd / even molecules, Stage Q's own split) agree: eps_A / eps_B within 0.95-1.05.
Writes planted_real_erase_01.txt."""
import os, sys, numpy as np
sys.path.insert(0, os.environ.get("CHAIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "chain")))
from stage_q_iam_a import pat_site_table, _pat_lines
MAXB = 100_000_000
def eps(T, h=("A", "B")): return float(sum(T[f"err_{x}"].sum() for x in h) / sum(T[f"opp_{x}"].sum() for x in h))
def plant(src, dst, arm, p, seed):
    rng = np.random.default_rng(seed); out = open(dst, "w")
    for l in _pat_lines(src, MAXB):
        q = l.decode().split("\t")
        if len(q) < 4: continue
        chrom, start, pat, n = q[0], q[1], q[2], int(q[3])
        idx = [j for j, x in enumerate(pat) if x != "."]
        for _ in range(n):
            s = list(pat)
            if arm != "base":
                c = [s[j] for j in idx]
                for i in range(1, len(c) - 1):
                    if c[i - 1] == "C" and c[i + 1] == "C":
                        if arm == "halved" and c[i] == "T" and rng.random() < 0.5: s[idx[i]] = "C"
                        if arm == "doubled" and c[i] == "C" and rng.random() < p: s[idx[i]] = "T"
            out.write(f"{chrom}\t{start}\t{''.join(s)}\t1\n")
    out.close()
if __name__ == "__main__":
    src, label, arm, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); p = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    dst = f"_{label}_{arm}.pat"; plant(src, dst, arm, p, seed); T = pat_site_table(dst); os.remove(dst)
    print(f"{label}\t{arm}\t{eps(T):.6f}\t{eps(T, ('A',))/eps(T, ('B',)):.4f}\t{int(T.opp_A.sum() + T.opp_B.sum())}\t{int(T.err_A.sum() + T.err_B.sum())}", flush=True)
