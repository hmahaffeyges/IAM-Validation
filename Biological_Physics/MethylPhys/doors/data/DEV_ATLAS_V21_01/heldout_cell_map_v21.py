"""Held-out label -> atlas v2.1 cell, built from the records, never typed per sample (author 2026-10-16: 'make it perfect ... this is for Loyfer
and Moss too'). Rule: the held-out label (atlas_v21_heldout_samples.csv, from the sealed plan's commissioning list) with its bracketed
provenance note removed must equal exactly one of the 79 fitted cells (stageB_v21_summary.json). Each mapped cell's naming source is recorded:
'Loyfer/Moss name' when the cell is built from Loyfer or Moss samples (cell_names_v21.csv), otherwise the public laboratories it is built
from (final_samples_v21.csv), since neither atlas holds that cell. Labels the plan marks NEW (cells left out of v2.1,
decision 5) map to none and are listed, not scored; truth sets (whole blood, cord blood) belong to U3. Any other label refuses the build.
Writes heldout_cell_map_v21.json. Usage: python heldout_cell_map_v21.py [--check]"""
import os, re, sys, json, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
S = pd.read_csv(os.path.join(H, "atlas_v21_heldout_samples.csv")); fitted = set(json.load(open(os.path.join(H, "stageB_v21_summary.json")))["measured_fraction_by_cell"])
named = set(pd.read_csv(os.path.join(H, "cell_names_v21.csv")).atlas_cell)
B = pd.read_csv(os.path.join(H, "final_samples_v21.csv"), keep_default_na=False); B = B[B.qc.astype(str) == "True"]
M, notes, bad = {}, {}, []
for lab, g in S.groupby("cell_type"):
    roles = set(g.role)
    if roles == {"truth set"}: M[lab] = None; notes[lab] = "truth set: scored in U3, not U1"; continue
    if lab.startswith("NEW: "): M[lab] = None; notes[lab] = "not in v2.1 (decision 5): listed, not scored"; continue
    base = re.sub(r"\s*\[.*?\]", "", lab).strip(); hit = [c for c in fitted if c == base]
    if len(hit) == 1:
        c = hit[0]; M[lab] = c
        src = "Loyfer/Moss name" if c in named else "built from " + ", ".join(sorted(set(B[B.cell == c].source))) + " (not a Loyfer or Moss cell)"
        notes[lab] = ("exact name" if base == lab else "exact name after removing the provenance note") + "; " + src
    else: bad.append(f"{lab!r} -> {hit}")
J = dict(_rule=__doc__.split("Writes")[0].strip(), map=M, how=notes, scored_cells=sorted({v for v in M.values() if v}),
         samples_by_label={k: int(v) for k, v in S.cell_type.value_counts().sort_index().items()})
txt = json.dumps(J, indent=1, ensure_ascii=False) + "\n"; P = os.path.join(H, "heldout_cell_map_v21.json")
if bad: print("heldout_cell_map: REFUSED"); [print("  " + b) for b in bad]; sys.exit(1)
if "--check" in sys.argv:
    ok = os.path.exists(P) and open(P, encoding="utf-8").read() == txt; print("heldout_cell_map: " + ("ok" if ok else "REFUSED: differs (run it)")); sys.exit(0 if ok else 1)
open(P, "w", encoding="utf-8").write(txt); print("heldout_cell_map: wrote", len(M), "labels,", len(J["scored_cells"]), "cells scored")
