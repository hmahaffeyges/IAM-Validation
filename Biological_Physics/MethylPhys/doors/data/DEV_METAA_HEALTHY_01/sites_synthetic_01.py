"""DEV-METAA-HEALTHY-01 step 3 version 2, synthetic check of stage_q2_selfref.count_pat_sites (written 2026-10-24 before the real counts).
Held molecules of 10 calls with isolated losses planted at 0.06 at a site set and 0.03 elsewhere; held-absent molecules with gains at 0.05
and 0.02. The counted lone-error rate at the sites, and the rate after dividing out the neighbors' (1 - eps_nb)^2 with eps_nb the planted
background (0.03, 0.02).  Usage: python3 sites_synthetic_01.py OUT.txt"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "chain")); import stage_q2_selfref as Q
g = np.random.default_rng(20261024); ident = set(range(1000, 200000, 7)); L = []
for m in range(120000):
    i = int(g.integers(1, 199990)); L.append(f"chr1\t{i}\t{''.join('T' if g.random() < (0.06 if (i + q) in ident else 0.03) else 'C' for q in range(10))}\t1")
    if m % 3 == 0:
        j = int(g.integers(1, 199990)); L.append(f"chr1\t{j}\t{''.join('C' if g.random() < (0.05 if (j + q) in ident else 0.02) else 'T' for q in range(10))}\t1")
r = Q.count_pat_sites(L, ident)
T = [f"planted at the sites: loss 0.06, gain 0.05; elsewhere 0.03, 0.02",
     f"counted at the sites: loss {r['held_err']:.4f} ({r['held_opp']} opportunities), gain {r['gain_err']:.4f} ({r['gain_opp']})",
     f"neighbors divided out: loss {r['held_err'] / 0.97 ** 2:.4f}, gain {r['gain_err'] / 0.98 ** 2:.4f}"]
open(sys.argv[1], "w").write("\n".join(T) + "\n"); print("\n".join(T))
