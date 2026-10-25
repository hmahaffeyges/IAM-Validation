"""DEV-CSCORE-LIMIT-01 synthetic (bar 1). Through the chain's own statistic (stage_metaa_atlas._clustering): 2,000 maps of independent
departures at each of three site counts (6,000: one cell's identity sites; 60,000; 770,000: the whole genome) must exceed the derived limit
in 4 to 6 % of maps; maps with a planted run (a block of 200 neighboring sites shifted by one site's spread, among 6,000) must exceed it in at
least 80 %.  Usage: python3 limit_synthetic.py OUT.txt"""
import sys, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain")); sys.path.insert(0, H)
import stage_metaa_atlas as SA, limit as LM
g = np.random.default_rng(20261025); L = ["sites | maps | limit | share above the limit"]; ok = True
for n, k in ((6000, 2000), (60000, 2000), (770000, 300)):
    lim = LM.limit(n); a = np.mean([SA._clustering(g.standard_normal(n)) > lim for _ in range(k)])
    L.append(f"{n} | {k} | {lim:.4f} | {a:.4f}"); ok &= (0.04 <= a <= 0.06) if k >= 2000 else (0.02 <= a <= 0.08)
def planted():
    z = g.standard_normal(6000); s = g.integers(0, 5800); z[s:s + 200] += 1.0; return z
p = np.mean([SA._clustering(planted()) > LM.limit(6000) for _ in range(1000)]); ok &= p >= 0.8
L.append(f"planted run of 200 sites, shift 1 spread, 6,000 sites: share above the limit {p:.4f} (at least 0.80)")
L.append(f"bar 1 (independent maps 4-6 % above, 2-8 % at 300 maps; planted runs at least 80 % above): {'met' if ok else 'not met'}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
