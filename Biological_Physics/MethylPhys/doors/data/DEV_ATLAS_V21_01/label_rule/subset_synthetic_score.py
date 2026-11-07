"""Scores subset_synthetic.py (bars written in it before it ran). Usage: python3 subset_synthetic_score.py LAND OUT.txt"""
import sys, os, glob, json
land, out = sys.argv[1:3]; R = [json.load(open(f)) for f in glob.glob(os.path.join(land, "**", "subset_r*.json"), recursive=True)]
MEM = ("t central memory cd4", "t effector memory cd4"); NAI = ("naive cd4 t cells",); ALL4 = MEM + NAI + ("cd4 t cells",)
L = [f"Naive and memory CD4 on sequencing: {len(R)} of 30 grid rows read by the one command"]; b1 = b2 = len(R) == 30
for x in sorted(R, key=lambda x: (x["cd4"], x["lam"], x["memfrac"])):
    f = x["found"]; m = sum(f.get(c, 0) for c in MEM); nv = sum(f.get(c, 0) for c in NAI); t4 = sum(f.get(c, 0) for c in ALL4)
    mfm = m / (m + nv) if m + nv > 0 else None
    L.append(f"  CD4 {x['cd4']}, lam {x['lam']:g}, memory {x['memfrac']}: measured memory fraction {'' if mfm is None else f'{mfm:.3f}'}, total CD4 {t4:.3f}, plain CD4 column {f.get('cd4 t cells', 0):.3f}" + ("" if x["rc"] == 0 else f" | run failed: {x['tail'][-160:]}"))
    if x["lam"] >= 15:
        b1 = b1 and mfm is not None and abs(mfm - x["memfrac"]) <= 0.15; b2 = b2 and abs(t4 - x["cd4"]) <= 0.1
L += [f"bar 1 (lam 15 and 40: memory fraction within 0.15): {'met' if b1 else 'not met'}", f"bar 2 (lam 15 and 40: total CD4 within 0.1): {'met' if b2 else 'not met'}",
      "synthetic: " + ("met" if b1 and b2 else "not met")]
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
