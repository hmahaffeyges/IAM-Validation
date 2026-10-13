"""DEV-IAMA-TRIM-01: merge the per-run calibration outputs of calib_trim_all_01.sh into one committed table and one decision file
(written 2026-10-13 before the run). Adds no method: it concatenates what head_diag_reads_01.py wrote and counts the decisions.

# NO CHAIN: concatenation and counting.
Usage: python3 merge_trim_all_01.py RUNDIR   (RUNDIR holds one folder per run)
"""
import glob, json, os, sys, collections
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = sys.argv[1]
T = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "*", "head_diag_reads_01.csv")))], ignore_index=True)
D = {}
for f in sorted(glob.glob(os.path.join(R, "*", "head_diag_reads_01_decision.json"))):
    J = json.load(open(f)); D.update(J["runs"]); K = (J["K_SD"], J["MAX_TRIM"], J["windows"], J["window_reads"])
B = pd.read_csv(os.path.join(HERE, "..", "DEV_IAMA_COMMISSION_01", "epiqc_stageB_runs.csv"))
T.to_csv(os.path.join(HERE, "calib_trim_all_01_windows.csv"), index=False)
rows = [dict(run=r, **{k: v for k, v in D[r].items()}) for r in sorted(D)]
P = pd.DataFrame(rows).merge(B[["run", "kit", "lab", "line", "rep"]], on="run", how="left")
P.to_csv(os.path.join(HERE, "calib_trim_all_01_decisions.csv"), index=False)
c = collections.Counter(P.decision)
L = [f"K_SD {K[0]}, MAX_TRIM {K[1]}, {K[2]} windows of {K[3]} read pairs per run (all fixed before this ran)",
     f"runs expected {B.run.nunique()} | windows read {len(T)} | failed windows {int(T.d.isna().sum())} | runs decided {len(P)}",
     "decisions: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())),
     "runs with a trim: " + (", ".join(f"{r.run} ({r.kit} {r.lab} {r.line} {r.rep}) skip {r.skip_windows}" for r in P[P.skip_windows > 0].itertuples()) or "none")]
for kit, g in P.groupby("kit"): L.append(f"  {kit}: {int((g.skip_windows > 0).sum())} of {len(g)} runs trimmed")
open(os.path.join(HERE, "calib_trim_all_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
