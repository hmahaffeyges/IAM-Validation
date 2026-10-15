"""Atlas v2.1 diagnostic 3 terms (2026-10-15; DIAGNOSTIC ONLY, never used to build the atlas). Two transfer files that each remove one
suspect, to find why every draw sits at the maximum tree depth with the smooth transfer:
 P  no plateaus: PCHIP through strict design 1's knots (ties merged; design 1 failed its own check C2 and is NOT used for the atlas).
 L  straight line: the v2.1 model code with g = Loyfer's v2 line, a + b m (terms_final_v21_run12.json), unclipped as in v2 (a cubic Hermite with both end slopes b is exactly that line).
    If L is also slow, the cause is the v2.1 variance terms, not the curve's shape."""
import os, json, numpy as np
from scipy.interpolate import PchipInterpolator
H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(H, "transfer_v21_smooth.json")))
D1 = json.load(open(os.path.join(H, "transfer_v21_strict_design1_NOTUSED.json")))
P = dict(S); P["knots_x"] = D1["knots_x"]; P["knots_y"] = D1["knots_y"]; P["knot_slopes"] = [float(x) for x in PchipInterpolator(D1["knots_x"], D1["knots_y"]).derivative()(D1["knots_x"])]
P["diagnostic_only"] = "diag3 arm P"
R = json.load(open(os.path.join(H, "terms_final_v21_run12.json")))["sequencing"]["Loyfer2023"]; a, b = R["a"], R["b"]
L = dict(S); L["knots_x"] = [0.0, 1.0]; L["knots_y"] = [float(a), float(a + b)]; L["knot_slopes"] = [float(b), float(b)]   # v2's line, unclipped, exactly as v2 uses it
L["diagnostic_only"] = "diag3 arm L"
for k, v in (("P", P), ("L", L)): json.dump(v, open(os.path.join(H, f"transfer_v21_diag3_{k}.json"), "w"), indent=1)
print("P knots", len(P["knots_x"]), "| L knots", L["knots_x"], [round(y, 4) for y in L["knots_y"]])
