"""DEV-CSCORE-LIMIT-01: the derived limit against the C-Score's definition on many maps (added after synthetic run 2, disclosed: bar 1's 2,000-map
window of 4-6 % is about two standard errors, and both runs used the same seed, so the same maps). Independent departures, seed 1.
Usage: python3 limit_check.py OUT.txt"""
import sys, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import limit as LM
def clustering(z, w=10):   # the C-Score's definition, written out (the math is checked here; the chain is tested by limit_synthetic_chain.py)
    o = np.asarray(z, dtype="float64"); nb = len(o) // w
    b = o[:nb * w].reshape(nb, w).mean(1) * np.sqrt(w); return float(np.var(b) / np.var(o))

g = np.random.default_rng(1); L = ["sites | maps | simulated 95th percentile | derived limit | share above the limit"]
for n, k in ((6000, 20000), (60000, 6000)):
    c = np.array([clustering(g.standard_normal(n)) for _ in range(k)])
    L.append(f"{n} | {k} | {np.quantile(c, .95):.5f} | {LM.limit(n):.5f} | {(c > LM.limit(n)).mean():.4f}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
