"""DEV-METAA-ATLAS-CURVE-01, B and T cells: the scorer (written and committed 2026-10-25 before any B or T array is read). Each array's bundle
from the series run (pass 1), the record at bundle["development"]["metaa_atlas"] (stage M-atlas-curve), read against its own atlas column
(samples column atlas_cell). Quality rule, stated before reading (the monocyte confirmation's lesson): an array the reading withholds (too few
identity sites past the array's own detection check) is reported apart, with its reason, and is not counted in bar 2; bar 2 is on the arrays
read, and a laboratory with more than a quarter of its arrays withheld is reported as not assessed for that laboratory.
Usage: python3 bt_curve_score.py RUN_DIR SAMPLES.csv OUT.txt"""
import sys, os, glob, json, csv, collections, numpy as np
D, SM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
S = {r["gsm"]: r for r in csv.DictReader(open(SM))}; rows = []
for f in sorted(glob.glob(os.path.join(D, "**", "pass1*", "GSM*.json"), recursive=True)):
    if f.endswith(("_values.json", "_planted.json")): continue
    g = os.path.basename(f)[:-5]
    if g not in S: continue
    r = ((json.load(open(f)).get("development") or {}).get("metaa_atlas")) or {}
    rows.append(dict(gsm=g, gse=S[g]["gse"], cell=S[g]["atlas_cell"], A=r.get("A"), up=r.get("above_curve"), n=r.get("n_sites"), note=r.get("reason") or r.get("status")))
L = ["array | laboratory | atlas column | Met-A on its own curve | above its curve | identity sites read | note"] + [" | ".join(str(r[k]) for k in ("gsm", "gse", "cell", "A", "up", "n", "note")) for r in rows]
L.append(f"arrays in the list {len(S)} | bundles found {len(rows)}")
read = [r for r in rows if r["A"] is not None]; held = [r for r in rows if r["A"] is None]
L.append(f"read {len(read)} | withheld {len(held)}: " + "; ".join(f"{r['gsm']} ({r['note']})" for r in held))
for k, key in (("laboratory", "gse"), ("atlas column", "cell")):
    G = collections.defaultdict(list)
    for r in rows: G[r[key]].append(r)
    for g, v in sorted(G.items()):
        a = [r["A"] for r in v if r["A"] is not None]
        L.append(f"{k} {g}: {len(a)} of {len(v)} read" + (f", median {np.median(a):.4f}, Normal {sum(0.95 <= x <= 1.05 for x in a)}" if a else "") + (" | NOT ASSESSED (more than a quarter withheld)" if key == "gse" and len(v) - len(a) > 0.25 * len(v) else ""))
n = sum(0.95 <= r["A"] <= 1.05 for r in read)
L.append(f"bar 2 (at least 90 % of the arrays read Normal, 0.95-1.05): {n} of {len(read)}: {'met' if read and n >= 0.9 * len(read) else 'not met'}")
up = [r["up"] for r in read if r["up"] is not None]; m = float(np.median(up)) if up else None
L.append(f"bar 3 (median above-curve within 0.005 of zero): {m}: {'met' if m is not None and abs(m) <= 0.005 else 'not met'}")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L[-12:]))
