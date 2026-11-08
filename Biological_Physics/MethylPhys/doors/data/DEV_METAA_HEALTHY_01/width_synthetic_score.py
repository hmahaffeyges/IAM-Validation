"""Scores the width synthetic (width_synthetic.py; written before it runs). For each (cell, rho), over its 20 constructed people: the spread
(standard deviation) of the chain's readings against the spread of the truth (each person's Met-A on their noise-free levels), and whether the
Deconvolver found the planted cell as each person's main cell. Bars: 1. for each of the six (cell, rho), the readings' spread within 25 % of the
truth's spread (the chain gives back the person spread planted); 2. every person's main cell found is the planted cell.
Usage: python3 width_synthetic_score.py FOLDER OUT.txt"""
import sys, glob, os, json, pandas as pd
R = sum((json.load(open(f)) for f in sorted(glob.glob(os.path.join(sys.argv[1], "width_synthetic_c*_r*_p*.json")))), [])
X = pd.DataFrame(R); L = [f"width synthetic: {len(X)} constructed people read by the one command"]; b1 = b2 = True
for (c, r), g in X.groupby(["cell", "rho"]):
    st, sr = g.truth.std(ddof=1), pd.to_numeric(g.reading, errors="coerce").std(ddof=1); ok = abs(sr / st - 1) <= 0.25 if st else False
    main = (g.main_cell == c).sum(); b1 &= bool(ok and len(g) == 20); b2 &= bool(main == len(g))
    L.append(f"  {c:12} rho {r:.1f} | people {len(g)} | truth spread {st:.5f} | reading spread {sr:.5f} (ratio {sr / st:.3f}) | truth median {g.truth.median():.4f} | reading median {pd.to_numeric(g.reading, errors='coerce').median():.4f} | main cell found {main} of {len(g)}")
L += [f"bar 1 (each cell and rho: reading spread within 25 % of the truth's): {'met' if b1 else 'not met'}", f"bar 2 (every main cell found is the planted cell): {'met' if b2 else 'not met'}", "synthetic: " + ("met" if b1 and b2 else "not met")]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
