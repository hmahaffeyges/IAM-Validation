"""Scores the copier-lag check (DEV-METHIONINE-01; prediction in the record before reading): each library's held error (Stage Q2
held_err_length) in early, mid and late-replicating DNA (lag_split.py; bundles in lag/), depleted over control. Writes score_lag_summary.txt."""
import os, json
H = os.path.dirname(os.path.abspath(__file__)); c = lambda n: json.load(open(os.path.join(H, "lag", n + ".json")))["iam_a_count_only"]["count"]
L = ["Copier-lag check: held error by replication timing (early / mid / late thirds), depleted over control"]
for line, a, b in (("MZGC1", "DRR322659", "DRR322660"), ("MZGC3", "DRR322663", "DRR322664")):
    for k in ("early", "mid", "late"):
        C, D = c(f"{a}_{k}"), c(f"{b}_{k}")
        L.append(f"{line} {k}: control {C['held_err_length']:.4f}, depleted {D['held_err_length']:.4f}, ratio {D['held_err_length'] / C['held_err_length']:.3f}")
open(os.path.join(H, "score_lag_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
