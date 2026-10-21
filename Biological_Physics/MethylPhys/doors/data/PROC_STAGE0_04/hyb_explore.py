"""PROC-STAGE0-04, exploratory (2026-10-20, not sealed, no bar): why do two whole slides of GSE184269 (200607130085, 200607130086) read the
hybridization 'high' control below 'low' while their methylation reads normally? For every array of GSE184269 in the Group-A rehearsal list
(four slides), every control probe's green and red intensity, with the chain's own manifest loader and IDAT reader. Then, per control probe,
the median on the two flagged slides against the median on the two normal slides; the summary lists every probe that differs by more than
2x, and the three hybridization probes in full. If only High and Low trade places (and Medium stays), the labels or the reagent are swapped,
not the hybridization weakened. Writes hyb_explore.csv and hyb_explore_summary.txt. Usage: python3 hyb_explore.py OUTDIR"""
import os, sys, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, os.path.join(MP, "chain")); sys.path.insert(0, os.path.join(MP, "doors", "data", "GROUPA_COMMISSIONING"))
import stage_0_1_qc_handoff as QC
from read_groupa import fetch
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
S = pd.read_csv(os.path.join(MP, "doors", "data", "GROUPA_COMMISSIONING", "rehearsal_samples.csv")); S = S[S.set == "GSE184269"]
FLAG = {200607130085, 200607130086}
ctl = QC._manifest("EPIC_V1").control_data_frame; rows = []
for r in S.itertuples():
    g = fetch(r.gsm, r.grn, os.path.join(out, r.grn)); d = fetch(r.gsm, r.red, os.path.join(out, r.red))
    gm = QC._channel_means(g, "grn")["mean"]; rm = QC._channel_means(d, "red")["mean"]
    for addr, x in ctl.iterrows():
        rows.append(dict(gsm=r.gsm, slide=int(r.slide), flagged=int(r.slide) in FLAG, type=str(x["Control_Type"]), probe=str(x["Extended_Type"]), address=int(addr),
                         grn=float(gm.get(addr, np.nan)), red=float(rm.get(addr, np.nan))))
    os.remove(g); os.remove(d); print(r.gsm, flush=True)
T = pd.DataFrame(rows); T.to_csv(os.path.join(out, "hyb_explore.csv"), index=False)
M = T.groupby(["type", "probe", "flagged"])[["grn", "red"]].median().unstack("flagged")
L = [f"arrays {T.gsm.nunique()} (flagged slides: {T[T.flagged].gsm.nunique()}, normal slides: {T[~T.flagged].gsm.nunique()})",
     "hybridization probes, median green | red: normal slides -> flagged slides"]
for (ty, pr), v in M.iterrows():
    if ty == "HYBRIDIZATION": L.append(f"  {pr}: {v[('grn', False)]:.0f} | {v[('red', False)]:.0f} -> {v[('grn', True)]:.0f} | {v[('red', True)]:.0f}")
L.append("every other control probe whose green or red median differs by more than 2x between flagged and normal slides (signal above 1000 on one side):")
n = 0
for (ty, pr), v in M.iterrows():
    if ty == "HYBRIDIZATION": continue
    for ch in ("grn", "red"):
        a, b = v[(ch, False)], v[(ch, True)]
        if max(a, b) > 1000 and (max(a, b) / max(min(a, b), 1)) > 2:
            L.append(f"  {ty} {pr} [{ch}]: {a:.0f} -> {b:.0f}"); n += 1
L.append(f"  ({n} found)")
open(os.path.join(out, "hyb_explore_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
