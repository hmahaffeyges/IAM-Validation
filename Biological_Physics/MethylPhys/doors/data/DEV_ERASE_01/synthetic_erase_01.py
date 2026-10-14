"""DEV-ERASE-01 synthetic test (rule written first, 2026-10-14): does Stage Q read a HALVED copy error as half?
Simulated sequencing molecules in .pat form (the format Stage Q reads): 6-20 CpGs each; 90 % from methylated regions, 10 % from
unmethylated regions (Stage Q's >= 80 %-methylated rule drops these). On methylated molecules each call is flipped to T independently
at rate p. Two files, p = 0.032 (healthy) and p = 0.016 (eraser removed), 300,000 molecules each, fixed seeds. Read with the chain's own
pat_site_table (chain/stage_q_iam_a.py, pipeline loyfer_pat_v1). Stage Q counts only ISOLATED errors (T between two C), so its eps is
not p; the test is of the ratio. BAR: the ratio of the two readings within 0.45-0.55. Expected from counting alone, before running:
(0.016 x 0.984^2) / (0.032 x 0.968^2) = 0.517."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "chain"))
from stage_q_iam_a import pat_site_table
def make(path, p, seed, n=300_000):
    rng = np.random.default_rng(seed); L = []
    for i in range(n):
        k = int(rng.integers(6, 21)); meth = rng.random() < 0.9
        c = np.where(rng.random(k) < (p if meth else 0.97), "T", "C") if meth else np.where(rng.random(k) < 0.97, "T", "C")
        L.append(f"chr1\t{1000 + 25 * (i % 40000)}\t{''.join(c)}\t1")
    open(path, "w").write("\n".join(L) + "\n")
def eps(path):
    T = pat_site_table(path); return (T.err_A.sum() + T.err_B.sum()) / (T.opp_A.sum() + T.opp_B.sum())
res = {}
for lab, p, seed in (("healthy", 0.032, 11), ("eraser_removed", 0.016, 12)):
    f = os.path.join(HERE, f"_synthetic_{lab}.pat"); make(f, p, seed); res[lab] = eps(f); os.remove(f)
q = res["eraser_removed"] / res["healthy"]; ok = 0.45 <= q <= 0.55
L = ["DEV-ERASE-01 synthetic test (synthetic_erase_01.py), run once",
     f"planted 0.032 -> Stage Q eps {res['healthy']:.5f}; planted 0.016 -> {res['eraser_removed']:.5f}",
     f"ratio {q:.4f} (expected from counting 0.517; bar 0.45-0.55): {'BAR MET' if ok else 'NOT MET'}"]
open(os.path.join(HERE, "synthetic_erase_01.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
