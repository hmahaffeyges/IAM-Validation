"""DEV-CSCORE-LIMIT-01, the 450K platform term on unread healthy bloods (written 2026-10-26 before any GSE67393 array is read). Every bundle the
one command wrote for GSE67393's test half, pass 1 (corrected after the first scoring, disclosed: it counted pass 1 and pass 2,
the series tare, so each blood twice; the term was measured on pass 1 readings only): the whole-genome C against limit_450k (the derived limit times the frozen term,
limit_04/platform_term_450k.json). Bar: at least 90 % of the bloods read at or below it. Usage: python3 limit_450k_score.py BUNDLE_DIR OUT.txt"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import limit as LM
V = []
for f in glob.glob(os.path.join(sys.argv[1], "**", "GSM*.json"), recursive=True):
    if f.endswith(("_values.json", "_planted.json")) or "/pass1" not in f: continue   # pass 1 only: the readings the term was measured on
    B = json.load(open(f)); w = (B.get("cscore_each_cell") or {}).get("whole_genome") or {}
    if w.get("C") is not None: V.append((os.path.basename(f).split(".")[0], B.get("array_type"), w["C"], LM.limit(w["n_sites"]), LM.limit_450k(w["n_sites"])))
L = ["blood | array type | C | plain limit | limit_450k"] + [f"{a} | {t} | {c} | {p:.4f} | {q:.4f}" for a, t, c, p, q in sorted(V)]
u = sum(c <= q for _, _, c, _, q in V); pl = sum(c <= p for _, _, c, p, _ in V); n = len(V)
L.append(f"read {n}; at or below limit_450k {u} ({u / max(n, 1):.4f}); at or below the plain limit {pl} ({pl / max(n, 1):.4f})")
L.append(f"bar (at least 90 % at or below limit_450k): {'met' if n and u / n >= 0.9 else 'not met'}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L[-3:]))
