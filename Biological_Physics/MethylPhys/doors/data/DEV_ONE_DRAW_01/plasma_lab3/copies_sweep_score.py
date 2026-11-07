"""Scores the copies sweep synthetic (copies_sweep.py; bars written before it runs). Reference: the deepest uncopied plasma (lam 8, D 1) read
without the flag. Bars: 1. uncopied (D 1): the flag changes whole-sample Met-A by at most 0.01 at every depth. 2. every grid point read with the
flag lies within 0.03 of the reference. Recorded: every point's gap (where the correction holds and where it breaks), the readings without the
flag, and the copies the chain measures against the planted dispersion. Usage: python3 copies_sweep_score.py LAND OUT.txt"""
import sys, os, glob, json
land, out = sys.argv[1:3]; R = {}
for f in glob.glob(os.path.join(land, "**", "copies_sweep_r*.json"), recursive=True):
    x = json.load(open(f)); R[(x["lam"], x["D"])] = x
ref = (R.get((8.0, 1.0)) or {}).get("off", {}).get("A")
L = [f"grid points read: {len(R)} of 30; reference (molecules Poisson(8), no copies, flag off): {ref}",
     "molecules  D      calls/covered  measured copies  Met-A off  Met-A on  gap on"]
b1 = b2 = ref is not None and len(R) == 30
for (l, d), x in sorted(R.items()):
    a0, a1 = x["off"]["A"], x["on"]["A"]; gap = None if a1 is None or ref is None else a1 - ref
    L.append(f"{l:<10} {d:<6} {x['calls_per_covered']!s:<14} {x['on']['copies']!s:<16} {a0!s:<10} {a1!s:<9} {'' if gap is None else f'{gap:+.4f}'}"
             + ("" if a1 is not None else f"  (not read: {x['on']['reason'] or x['on']['tail']})"))
    if d == 1: b1 = b1 and a0 is not None and a1 is not None and abs(a1 - a0) <= 0.01
    b2 = b2 and gap is not None and abs(gap) <= 0.03
L += [f"bar 1 (uncopied: the flag changes Met-A by at most 0.01 at every depth): {'met' if b1 else 'not met'}",
      f"bar 2 (every point with the flag within 0.03 of the reference): {'met' if b2 else 'not met'}", "sweep synthetic: " + ("met" if b1 and b2 else "not met")]
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
