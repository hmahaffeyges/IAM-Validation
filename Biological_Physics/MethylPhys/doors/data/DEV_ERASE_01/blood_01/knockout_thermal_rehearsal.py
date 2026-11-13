"""Rehearsal before sealing a physics-only knockout prediction (exploration on spent counts, PHY-085; 2026-11-13). With the erasers removed,
what remains should be the thermal loss (the candidate, CANON eps0_candidate) as a floor. Thermal level through the counter: the ratio-1
forward model (DEV_CHAIN_Q2_01/forward_out fc_seq_*), at each library's clustering (held flat beyond the grid's last point, 1.75), scaled
by the per-site loss at each coin (PHY_050_PERCALL/atp_coin_table.csv: measured in-tissue dG_ATP). Writes knockout_thermal_rehearsal.txt."""
import os, json, glob, math, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); Q = os.path.join(H, "..", "..")
K = json.load(open(os.path.join(Q, "..", "..", "..", "..", "CANON", "iam_canon.json")))["constants"]; R, T = 8.314462618, K["T_cell"]["value"]
odds = lambda w: (w / 2) / (1 - (1 - math.exp(-w)) / w); e = lambda dG: 1 / (1 + odds(2 * dG * 1000 / (R * T)))
P = sorted((pd.read_csv(f).coupling.mean(), pd.read_csv(f).held_err_length.mean()) for f in glob.glob(os.path.join(Q, "DEV_CHAIN_Q2_01", "forward_out", "fc_seq_*.csv")))
c, h = np.array([p[0] for p in P]), np.array([p[1] for p in P])
C = {json.load(open(f))["gsm"]: json.load(open(f)) for f in glob.glob(os.path.join(H, "*_count.json"))}
G = {"WT": ("GSM7111686", "GSM7111689"), "KO": ("GSM7111688", "GSM7111691"), "Mut": ("GSM7111687", "GSM7111690")}
L = ["coin (kJ/mol) | per-site thermal | " + " | ".join(f"{g} / thermal" for g in G)]
for lab, dG in (("canon 54.0", 54.0), ("in-tissue mean 58.5", 58.5), ("human brain 62.1", 62.1)):
    r = {g: [C[x]["held_err_length"] / (float(np.interp(C[x]["coupling"], c, h)) * e(dG) / e(54.0)) for x in xs] for g, xs in G.items()}
    L.append(f"{lab} | {e(dG):.5f} | " + " | ".join(f"{min(v):.3f}-{max(v):.3f}" for v in r.values()))
L.append("Reading: with the erasers removed the progenitors read below the thermal level at the canon coin (a floor should not be crossed); at measured coins they come closer. The coin is the lead for each type's level.")
open(os.path.join(H, "knockout_thermal_rehearsal.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
