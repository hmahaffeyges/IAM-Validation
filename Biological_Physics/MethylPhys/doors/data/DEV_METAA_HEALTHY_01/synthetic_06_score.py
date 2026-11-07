"""DEV-METAA-HEALTHY-01 synthetic test 6 scorer (test 5's bars and its flag check, on synthetic_06_draw*.json; it also records the
real blood's A / A_pred on every atlas cell and the number of cells in its measured mix). Test 5's (test 4's, with one check added: every array's recorded line must carry
line = censored, else the draw is refused and named, since test 4 ran without the flag). Test 4 (written and committed before any draw exists). Reads synthetic_04_draw<k>.json for k 0 to 9.
Bar 1: each of the 30 healthy arrays (3 real-range conditions x 10 draws) reads A / A_pred inside Normal, 0.95 to 1.05. Bar 2: in each of the
3 conditions the mean of its 10 draws is within 0.005 of 1. Bar 3: on each of the 3, a 5 % loss raises A / A_pred by at least half as much as it
raises A (draw 0). Recorded, not scored: each condition's spread over the draws; test 3's weak array.
Usage: python3 synthetic_04_score.py FOLDER OUT.txt"""
import sys, os, json, glob, numpy as np
src, out = sys.argv[1:3]; D = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(src, "**", "synthetic_06_draw*.json"), recursive=True))]
D = {d["draw"]: d for d in D}; C = ["center", "low tail", "high tail"]
bad = sorted(d["draw"] for d in D.values() for k, v in d["arrays"].items() if any((x or {}).get("line") != "censored" for x in (v.get("line") or {}).values()))
if bad: print("REFUSED: draws whose line is not censored (the flag did not take effect):", bad); D = {k: v for k, v in D.items() if k not in bad}
L = [f"DEV-METAA-HEALTHY-01 synthetic test 6 (the real full mix): {len(D)} of 10 draws read, every array through run_sample.py --dev-line-censored"]
b1 = len(D) == 10; b2 = True
for c in C:
    r = np.array([D[k]["arrays"][c]["ratio"] for k in sorted(D) if D[k]["arrays"].get(c, {}).get("ratio") is not None], dtype=float)
    inn = int((np.abs(r - 1) <= 0.05).sum()); b1 &= len(r) == 10 and inn == 10; b2 &= len(r) == 10 and abs(r.mean() - 1) <= 0.005
    L.append(f"  {c:9}: {len(r)} draws | inside 0.95 to 1.05: {inn} | mean {r.mean():.5f} | spread (sd) {r.std(ddof=1):.5f} | range {r.min():.5f} to {r.max():.5f}")
b3 = 0 in D and all(D[0]["arrays"][c].get("ratio_loss") is not None and (D[0]["arrays"][c]["ratio_loss"] - D[0]["arrays"][c]["ratio"]) >= 0.5 * (D[0]["arrays"][c]["A_loss"] - D[0]["arrays"][c]["A"]) for c in C)
w = [v for k, v in (D.get(0, {}).get("arrays") or {}).items() if k.startswith("weak")]
if 0 in D: L.append(f"  recorded: the real blood on every atlas cell: A / A_pred {D[0]['real']['ratio']} | cells in its measured mix {len([v for v in (D[0].get('shares') or {}).values() if v > 0])}")
if w: L.append(f"  recorded: test 3's weak array (beyond the real range): A / A_pred {w[0].get('ratio')}")
L += [f"bar 1 (each of the 30 inside Normal): {'met' if b1 else 'not met'}", f"bar 2 (each condition's mean within 0.005 of 1): {'met' if b2 else 'not met'}",
      f"bar 3 (a 5 % loss shows on each of the 3): {'met' if b3 else 'not met'}", "synthetic: " + ("met" if b1 and b2 and b3 else "not met")]
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
