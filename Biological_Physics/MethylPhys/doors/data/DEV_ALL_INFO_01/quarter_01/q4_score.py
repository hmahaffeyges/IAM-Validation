"""DEV-ALL-INFO-01, quarter-depth test (written and committed 2026-10-25 before any quarter reading is made). Each of the 10 samples read at full
depth is read again by the one command from 5 of its 20 alignment chunks (0, 4, 8, 12, 16: a quarter of its read pairs, spread across the
run), merged by split_align.yml with keepchunks. Compared with the same sample's newest full-depth bundle.
Usage: python3 q4_score.py QUARTER_DIR FULL_DIR OUT.txt   (FULL_DIR: the full-depth bundles, one per sample; the quarter ones are found by name)"""
import sys, os, json, glob
from scipy.stats import spearmanr
Q, F, OUT = sys.argv[1:4]
def newest(d, g, skip=None):
    f = [x for x in glob.glob(os.path.join(d, "**", g + "_bundle.json"), recursive=True) if not (skip and os.path.abspath(skip) in os.path.abspath(x))]
    return json.load(open(max(f, key=os.path.getmtime))) if f else None
G = sorted(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "q4_runs.json"))))
L = ["sample | IAM-A full | quarter | diff | Met-A whole full | quarter | diff"]; r = []
for g in G:
    a, b = newest(F, g, skip=Q), newest(Q, g)
    if not a or not b: L.append(f"{g} | missing ({'full' if not a else 'quarter'})"); continue
    ia, ib = (a.get("iam_a") or {}).get("A"), (b.get("iam_a") or {}).get("A"); ma, mb = (a.get("met_a_whole") or {}).get("A"), (b.get("met_a_whole") or {}).get("A")
    if None in (ia, ib, ma, mb): L.append(f"{g} | a reading is missing"); continue
    r.append((ia, ib, ma, mb)); L.append(f"{g} | {ia:.4f} | {ib:.4f} | {ib - ia:+.4f} | {ma:.4f} | {mb:.4f} | {mb - ma:+.4f}")
n = len(r); b1 = sum(abs(x[1] - x[0]) <= 0.02 for x in r); b2 = spearmanr([x[2] for x in r], [x[3] for x in r]).correlation if n > 2 else None   # amended before any quarter reading (synthetic run 2)
rho = spearmanr([x[0] for x in r], [x[1] for x in r]).correlation if n > 2 else None
L.append(f"bar 1 (IAM-A within 0.02 of full depth, all 10): {b1} of {n}: {'met' if b1 == n == 10 else 'not met'}")
L.append(f"bar 2 (Met-A whole sample's order kept, rank correlation at least 0.9; amended): {b2 if b2 is None else round(b2, 4)}: {'met' if b2 is not None and b2 >= 0.9 else 'not met'}")
L.append(f"bar 3 (IAM-A's order kept, rank correlation at least 0.9): {rho if rho is None else round(rho, 4)}: {'met' if rho is not None and rho >= 0.9 else 'not met'}")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
