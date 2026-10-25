"""DEV-CSCORE-LIMIT-01: the derived limit against the chain's statistic on many maps (added after synthetic run 2, disclosed: bar 1's 2,000-map
window of 4-6 % is about two standard errors, and both runs used the same seed, so the same maps). Independent departures, seed 1.
Usage: python3 limit_check.py OUT.txt"""
import sys, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain")); sys.path.insert(0, H)
import stage_metaa_atlas as SA, limit as LM
g = np.random.default_rng(1); L = ["sites | maps | simulated 95th percentile | derived limit | share above the limit"]
for n, k in ((6000, 20000), (60000, 6000)):
    c = np.array([SA._clustering(g.standard_normal(n)) for _ in range(k)])
    L.append(f"{n} | {k} | {np.quantile(c, .95):.5f} | {LM.limit(n):.5f} | {(c > LM.limit(n)).mean():.4f}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
