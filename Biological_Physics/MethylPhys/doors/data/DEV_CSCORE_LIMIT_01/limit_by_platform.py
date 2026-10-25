"""DEV-CSCORE-LIMIT-01: bar 2's whole-genome C against its derived limit, split by array type (added after bar 2, disclosed; nothing new scored).
Usage: python3 limit_by_platform.py BUNDLE_DIR OUT.txt   (the five laboratories' 100 healthy bloods, the one command's bundles)"""
import sys, os, json, glob, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import limit as LM
G = collections.defaultdict(list)
for f in glob.glob(os.path.join(sys.argv[1], "**", "GSM*.json"), recursive=True):
    if f.endswith(("_values.json", "_planted.json")): continue
    B = json.load(open(f)); w = (B.get("cscore_each_cell") or {}).get("whole_genome") or {}
    if w.get("C") is not None: G[B.get("array_type") or "?"].append((w["C"], LM.limit(w["n_sites"]), w["n_sites"]))
L = ["array | bloods | above the derived limit | share above | median C | median sites"]
for a, v in sorted(G.items()):
    c = np.array([x[0] for x in v]); l = np.array([x[1] for x in v])
    L.append(f"{a} | {len(v)} | {int((c > l).sum())} | {float((c > l).mean()):.4f} | {float(np.median(c)):.4f} | {int(np.median([x[2] for x in v]))}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
