"""Scores the sorted-cell rehearsal (written before its readings are opened; development, spent data). The instrument's job: a healthy
sorted cell from a laboratory the atlas never held reads Normal for the cell the IAM-Deconvolver found most of, against that cell's healthy
width measured on IAM-Atlas v2.1's own donors (stage_metaa_atlas.width_state; cells_found's met_a_state). Per array: the top cell found, its
share, Met-A and state. Bars (the commissioning test's, rehearsed): 1. the top cell found is the sorted cell's kind (a subtype of it counts)
in at least 90 % of arrays; 2. of the arrays with a state, at least 85 % read Normal; 3. of the arrays whose top cell has a C-Score, at least 85 % sit at or under its limit (the
derived limit times the cell's factor measured on the atlas's own donors). Usage: sorted_rehearsal_score.py BUNDLE_DIR SAMPLES.csv OUT.txt"""
import os, sys, json, glob, csv
K = {"neutrophils": "neut", "cd4 t cells": "cd4", "cd8 t cells": "cd8", "b cells": "b cell", "nk cells": "nk", "monocytes": "mono"}
def kind_of_spec(s):
    s = s.lower().replace("sorted ", "")
    return s
def same_kind(spec, cell):
    c = cell.lower(); s = kind_of_spec(spec)
    if s == "neutrophils": return c == "neutrophils"
    if s == "monocytes": return c in ("monocytes", "classical monocytes")
    if s == "cd4 t cells": return "cd4" in c or "th1" in c or c.startswith("regulatory t") or c in ("naive cd4 t cells", "t central memory cd4", "t effector memory cd4")
    if s == "cd8 t cells": return "cd8" in c
    if s == "b cells": return "b cell" in c
    if s == "nk cells": return "nk" in c
    return False
S = {r["gsm"]: r["specimen"] for r in csv.DictReader(open(sys.argv[2]))}
rows = []
for g, sp in S.items():
    f = next(iter(glob.glob(os.path.join(sys.argv[1], "**", g + ".json"), recursive=True)), None)
    if not f: rows.append((g, sp, None, None, None, "no bundle", None)); continue
    cf = (json.load(open(f)).get("cells_found") or {}).get("cells") or []
    if not cf: rows.append((g, sp, None, None, None, "no cells found", None)); continue
    t = max(cf, key=lambda c: c.get("share") or 0); rows.append((g, sp, t["cell"], t.get("share"), t.get("met_a"), t.get("met_a_state"), t.get("c_within")))
n = len(rows); top = sum(1 for r in rows if r[2] and same_kind(r[1], r[2]))
st = [r for r in rows if r[5] in ("Normal", "above Normal", "below Normal")]; nor = sum(1 for r in st if r[5] == "Normal")
by = {}
for r in st: by.setdefault(kind_of_spec(r[1]), []).append(r[5] == "Normal")
cs = [r for r in rows if r[6] is not None]; cw = sum(1 for r in cs if r[6])
m1 = top >= 0.9 * n; m2 = bool(st) and nor >= 0.85 * len(st); m3 = bool(cs) and cw >= 0.85 * len(cs)
L = [f"{g} {sp}: top {c} ({s}) Met-A {a} state {x} C-Score within {w}" for g, sp, c, s, a, x, w in rows]
L += [f"bar 1 (top cell is the sorted kind, 90 %): {top} of {n}: {'met' if m1 else 'not met'}",
      f"bar 2 (Normal, 85 % of arrays with a state): {nor} of {len(st)}: {'met' if m2 else 'not met'}",
      f"bar 3 (C-Score within its limit, 85 % of arrays with a C-Score): {cw} of {len(cs)}: {'met' if m3 else 'not met'}",
      "MET" if (m1 and m2 and m3) else "NOT MET"]
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L[-4:]))
