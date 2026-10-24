"""DEV-ONE-DRAW-01, added after absorption test 1 (disclosed): the stand-in for megakaryocytes, the four atlas cells the IAM-Deconvolver gave the
pure megakaryocyte pool (lung interstitial macrophages, erythrocyte progenitors, MEP (bone marrow), eosinophils), summed in each planted plasma
and set against simple mixing: (1 - f) x the plasma's own sum + f x the pure pool's sum. Reads only the bundles mk_absorb_01.py wrote.
Usage: python3 mk_standin.py OUTDIR_OF_MK_ABSORB_01 [OUT.txt]"""
import sys, os, json
d = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "mk_standin_summary.txt")
C = ["lung interstitial macrophages", "erythrocyte progenitors", "MEP (bone marrow)", "eosinophils"]
def s(tag): B = json.load(open(os.path.join(d, tag + ".json"))); sh = {x["cell"]: x["share"] for x in B["cell_shares"]["cells"]}; return sum(sh.get(c, 0) for c in C)
p, b = s("megakaryocytes"), s("plasma_plus_mk_00pct")
L = [f"stand-in = {', '.join(C)}", f"pure megakaryocyte pool: {p:.3f}", f"healthy plasma, nothing planted: {b:.3f}"]
for f in (10, 25, 40): L.append(f"planted {f} %: {s(f'plasma_plus_mk_{f:02d}pct'):.3f} | simple mixing predicts {(1 - f / 100) * b + f / 100 * p:.3f}")
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
