"""DEV-METAA-ATLAS-CURVE-01 scorer (written and committed 2026-10-25 before any monocyte array is read). Reads each array's bundle from the
series run (pass 1), the record at bundle["development"]["metaa_atlas"] (stage M-atlas-curve), and scores bars 2 and 3 of the record.
Usage: python3 mono_curve_score.py RUN_DIR OUT.txt"""
import sys, os, glob, json, numpy as np
D, OUT = sys.argv[1], sys.argv[2]; rows = []
for f in sorted(glob.glob(os.path.join(D, "**", "pass1", "GSM*.json"), recursive=True)):
    if f.endswith(("_values.json", "_planted.json")): continue
    B = json.load(open(f)); r = (B.get("development") or {}).get("metaa_atlas") or {}
    rows.append((os.path.basename(f)[:-5], r.get("A"), r.get("above_curve"), r.get("n_sites"), r.get("state"), r.get("reason") or r.get("status")))
L = ["array | Met-A on its own curve | above its curve at the identity sites | identity sites read | state | note"] + [" | ".join(str(x) for x in r) for r in rows]
A = [r[1] for r in rows if r[1] is not None]; up = [r[2] for r in rows if r[2] is not None]; n = sum(0.95 <= a <= 1.05 for a in A)
L.append(f"read {len(A)} of {len(rows)} arrays | Met-A median {np.median(A):.4f}, range {min(A):.4f} to {max(A):.4f}" if A else f"read 0 of {len(rows)} arrays")
L.append(f"bar 2 (at least 90 % of healthy arrays Normal, 0.95-1.05): {n} of {len(rows)}: {'met' if rows and n >= 0.9 * len(rows) else 'not met'}")
m = float(np.median(up)) if up else None
L.append(f"bar 3 (median above-curve within 0.005 of zero): {m}: {'met' if m is not None and abs(m) <= 0.005 else 'not met'}")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
