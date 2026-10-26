"""DEV-CSCORE-LIMIT-01 (added after the 450K neighbor reading, disclosed; nothing scored): the whole-genome C a map's own neighbor correlation
predicts, C = 1 + 2 sum_k (1 - k/w) r_k over lags 1 to w - 1 (w = 10, the block), from the one command's bundles, beside the measured C and
the derived limit, by array type. Usage: python3 limit_inflation.py BUNDLE_DIR OUT.txt"""
import sys, os, json, glob, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import limit as LM
G = collections.defaultdict(list)
for f in glob.glob(os.path.join(sys.argv[1], "**", "GSM*.json"), recursive=True):
    if f.endswith(("_values.json", "_planted.json")): continue
    B = json.load(open(f)); w = (B.get("cscore_each_cell") or {}).get("whole_genome") or {}
    if w.get("C") is None or not w.get("neighbor_corr"): continue
    p = 1 + 2 * sum((1 - k / 10) * r for k, r in enumerate(w["neighbor_corr"], 1))
    G[B.get("array_type") or "?"].append((w["C"], p, LM.limit(w["n_sites"])))
L = ["array | bloods | median measured C | median C its own neighbor correlation predicts | median derived limit | median limit x prediction"]
for a, v in sorted(G.items()):
    c, p, l = map(np.array, zip(*v)); L.append(f"{a} | {len(v)} | {np.median(c):.4f} | {np.median(p):.4f} | {np.median(l):.4f} | {np.median(l * p):.4f}")
L.append("array | per-blood correlation of measured C with its own prediction | median of measured C minus its own prediction | spread of that difference")
for a, v in sorted(G.items()):
    c, p, l = map(np.array, zip(*v)); L.append(f"{a} | {np.corrcoef(c, p)[0, 1]:.4f} | {np.median(c - p):+.4f} | {np.std(c - p):.4f}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
