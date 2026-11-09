"""Scores seq_noise_weight_synthetic.py (written with it, before it runs). Shares compared after summing each subtype into the cell it
belongs to (the truth names six blood cells and three tissues); distance = half the sum of absolute share differences. Bars: 1. the weighted
reading closer to the truth than the unweighted in at least 5 of the 6 samples; 2. the share given to cells not in the sample no larger
weighted than unweighted in at least 5 of 6; 3. at depth 20 the weighted distance at most 0.01 above the unweighted.
Usage: seq_noise_weight_synthetic_score.py FOLDER OUT.txt"""
import os, sys, json, glob
def parent(c):
    c = c.lower()
    for k in ("cd4", "cd8", "nk", "b cell"):
        if k in c: return {"cd4": "cd4 t cells", "cd8": "cd8 t cells", "nk": "nk cells", "b cell": "b cells"}[k]
    if c in ("granulocytes", "neutrophils"): return "granulocytes"
    return c
def grp(s):
    o = {}
    for c, v in s.items(): o[parent(c)] = o.get(parent(c), 0) + (v or 0)
    return o
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(sys.argv[1], "seq_noise_weight_synthetic_p*.json")))]
L = []; b1 = b2 = 0; b3 = True; n = 0
for r in R:
    t = grp(r["truth"]); rd = r["readings"]
    if any(rd[k]["rc"] != 0 for k in rd): L.append(f"part {r['part']}: a reading failed {[(k, rd[k]['rc'], rd[k]['err'][-150:]) for k in rd]}"); continue
    s0, s1 = grp(rd["False"]["shares"]), grp(rd["True"]["shares"]); d = lambda s: 0.5 * sum(abs(s.get(c, 0) - t.get(c, 0)) for c in set(s) | set(t))
    f = lambda s: sum(v for c, v in s.items() if c not in t); n += 1
    b1 += d(s1) < d(s0); b2 += f(s1) <= f(s0)
    if r["depth"] == 20 and d(s1) > d(s0) + 0.01: b3 = False
    L.append(f"part {r['part']} {r['kind']} depth {r['depth']}: distance unweighted {d(s0):.4f} weighted {d(s1):.4f}; cells not in it {f(s0):.4f} -> {f(s1):.4f}")
m1, m2 = b1 >= 5, b2 >= 5
L += [f"bar 1 (closer, 5 of 6): {b1} of {n}: {'met' if m1 else 'not met'}", f"bar 2 (false cells no larger, 5 of 6): {b2} of {n}: {'met' if m2 else 'not met'}",
      f"bar 3 (depth 20 no worse by 0.01): {'met' if b3 else 'not met'}", "MET" if (m1 and m2 and b3 and n == 6) else "NOT MET"]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
