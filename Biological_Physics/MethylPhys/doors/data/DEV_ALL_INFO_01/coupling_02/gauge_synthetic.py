"""DEV-ALL-INFO-01, the coupling gauge's synthetic (written before the healthy reference exists). Planted held molecules, counted by Stage Q2's
own counter (stage_q2_selfref.count_pat) and read by stage_coupling with a planted reference: (1) errors placed independently read coupling
near 1; (2) errors placed in runs read above 1; (3) a sample whose coupling equals its cell's planted healthy coupling reads coupling-A 1,
Normal; (4) the ceiling is above the reading and the floor 0. Usage: python3 gauge_synthetic.py OUT.txt"""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4, "chain"))
import numpy as np, stage_q2_selfref as Q2, stage_coupling as CP
random.seed(20261028)
def molecules(clustered):
    L = []
    for i in range(40000):
        n = 12; s = ["C"] * n; t = random.choice((1, 2, 2, 3))
        if clustered: j = random.randrange(n - t + 1); s[j:j + t] = ["T"] * t
        else:
            for q in random.sample(range(n), t): s[q] = "T"
        L.append(f"chr1\t{1000 + 40 * i}\t{''.join(s)}\t1\n")
    return [x for m in L for x in (m.replace("\t1\n", "\t0\n"), m)]   # count_pat reads odd lines as the held half
T = {"chr1": np.array([[0, 10 ** 9]])}
r_ind = Q2.count_pat(molecules(False), T); r_cl = Q2.count_pat(molecules(True), T)
ok = []; L = []
L.append(f"independent: coupling {r_ind['coupling']:.4f}, ceiling {r_ind['coupling_max']:.4f}"); ok.append(abs(r_ind["coupling"] - 1) < 0.05)
L.append(f"in runs: coupling {r_cl['coupling']:.4f}"); ok.append(r_cl["coupling"] > 1.5)
ref = dict(reference={"b cells": r_ind["coupling"]}, band=dict(lo=0.9, hi=1.1, n=10))
k = CP.read(dict(count=r_ind), "b cells", ref); L.append(f"healthy-equal sample: coupling-A {k['A']}, {k['state']}, floor {k['floor']}, ceiling {k['ceiling']}")
ok += [abs(k["A"] - 1) < 1e-6 and k["state"] == "Normal", k["ceiling"] > k["A"] and k["floor"] == 0]
L.append(f"synthetic (all four): {'met' if all(ok) else 'not met'} {ok}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
