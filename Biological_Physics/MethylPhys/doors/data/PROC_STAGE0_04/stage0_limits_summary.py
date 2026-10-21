"""PROC-STAGE0-04: the healthy distribution of each Stage 0 control-probe check by laboratory, the laboratories-equal percentiles, and how many
healthy arrays each limit of intake_thresholds_v2.json stops. Reads stage0_healthy_epic.csv (GitHub explore run 39611932552) and the sample list;
writes stage0_limits_summary.txt here. Usage: python3 stage0_limits_summary.py"""
import os, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
D = pd.read_csv(os.path.join(H, "stage0_healthy_epic.csv")).merge(pd.read_csv(os.path.join(H, "stage0_healthy_epic_samples.csv"))[["gsm", "set"]].rename(columns={"set": "lab"}), on="gsm")
TH = json.load(open(os.path.join(H, "..", "..", "..", "chain", "Runtime Matrices", "Intake", "intake_thresholds_v2.json")))
L = [f"{len(D)} healthy arrays: " + ", ".join(f"{k} {v}" for k, v in D.lab.value_counts().items())]
for c, name in (("bisulfite_conversion_efficiency", "bisulfite conversion"), ("hyb_spread", "hybridization spread"), ("extension_ratio", "extension balance")):
    L.append(name + ": " + "; ".join(f"{lab} min {g[c].min():.4f} median {g[c].median():.4f} max {g[c].max():.4f}" for lab, g in D.groupby("lab")))
b = D.bisulfite_conversion_efficiency; w = D.lab.map(1 / D.lab.value_counts()); o = np.argsort(b.values); cw = np.cumsum(w.values[o]) / w.sum()
for q in (0.025, 0.05): L.append(f"bisulfite, laboratories counted equally, {100 * q:.1f}th percentile: {b.values[o][np.searchsorted(cw, q)]:.4f}")
lo, hi = TH["extension"]["range"]
L.append(f"stopped by bisulfite >= {TH['bisulfite_conversion']['min']}: {int((b < TH['bisulfite_conversion']['min']).sum())} ({', '.join(D.gsm[b < TH['bisulfite_conversion']['min']])})")
L.append(f"stopped by hybridization spread >= {TH['hybridization']['min_spread']}: {int((D.hyb_spread < TH['hybridization']['min_spread']).sum())}")
L.append(f"stopped by extension {lo} to {hi}: {int((~D.extension_ratio.between(lo, hi)).sum())}")
open(os.path.join(H, "stage0_limits_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
