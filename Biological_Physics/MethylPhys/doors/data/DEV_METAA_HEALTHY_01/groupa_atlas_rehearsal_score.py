"""Scores Group-A's spent rehearsal against IAM-Atlas v2.1 alone (development; GSE318173, read by the one command through run_series.py
--plant neutrophil_atlas). The instrument's job: two groups of healthy people of one series read as no different, and a group shift planted
into healthy arrays is found. Per cell found (stage_groupa.compare_cells), no tare:
  healthy groups: women against men, every array's own reading (idat route)
  planted: the planted half (constructed 1 % loss of the neutrophils' pattern, scaled by the neutrophil share found) against the unplanted
           half, both read from their saved values
usage: python3 groupa_atlas_rehearsal_score.py BUNDLE_DIR SAMPLES.csv OUT.txt"""
import os, sys, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain")); import stage_groupa as GA
bd, sc, out = sys.argv[1:4]; S = pd.read_csv(sc); S["id"] = S.gsm; L = []
for suffix, route in (("", "idat"), ("_planted", "planted"), ("_values", "values")):
    T = S.copy(); T["id"] = T.gsm + suffix; p = os.path.join(bd, f"_ids_{route}.csv"); T.to_csv(p, index=False)
    GA.collect_cells(bd, p, p + ".cells"); R = pd.read_csv(p + ".cells"); R["route"] = route; L.append(R)
R = pd.concat(L); R = R[R.cell.notna()]; lines = [f"Group-A against IAM-Atlas v2.1 alone, spent rehearsal (development): {S.gsm.nunique()} listed arrays"]
I = R[R.route == "idat"]; lines.append(f"read (idat route): {I[I.A.notna()].gsm.nunique()} arrays")
lines.append("\nHealthy groups, women against men (idat route), per cell found at a share of 5 % or more:")
for c, o in GA.compare_cells(I, "sex", "female", "male").items():
    lines.append(f"  {c:28s} share {o['share_median']:.3f}: women {o['mean1']} (n {o['n1']}), men {o['mean2']} (n {o['n2']}): {100*o['rel_difference']:+.2f} %, {o['t']:+.2f} SE -> {o['call']}")
V = pd.concat([R[(R.route == "planted")], R[(R.route == "values")]]); V["grp"] = V.route
lines.append("\nPlanted half (1 % loss of the neutrophils' pattern) against the unplanted half, both by saved values, per cell found:")
for c, o in GA.compare_cells(V, "grp", "planted", "values").items():
    lines.append(f"  {c:28s} share {o['share_median']:.3f}: planted {o['mean1']} (n {o['n1']}), unplanted {o['mean2']} (n {o['n2']}): {100*o['rel_difference']:+.2f} %, {o['t']:+.2f} SE -> {o['call']}")
open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
