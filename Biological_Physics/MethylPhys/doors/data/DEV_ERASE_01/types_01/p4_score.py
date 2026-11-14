"""PHY-086 P4 score (written before the last library landed; 2026-11-13). Sealed rule (DEV_ERASE_01.md): mature B cells (GSE161463), TET2
and TET3 removed, knockout over control 0.75-0.95, each arm's median of its libraries' lone-loss held error (held_err_length, chain Stage
Q2 --dev-count-only). Also read by the project's standard (pull classes): the range's middle as the prediction and its half-width as 2 sigma,
combined with the measurement's own sigma from the replicate libraries; within 2 passes, 2 to 3 watch, beyond 3 a miss.
Usage: python3 p4_score.py COUNTS_DIR (the four <GSM>_count.json). Writes p4_summary.txt beside this script."""
import os, sys, json, glob, math, statistics as st
H = os.path.dirname(os.path.abspath(__file__)); D = sys.argv[1]
WT, KO = ("GSM4907469", "GSM4907470"), ("GSM4907471", "GSM4907472")
C = {}
for f in glob.glob(os.path.join(D, "**", "*_count.json"), recursive=True):
    c = json.load(open(f)); C[c.get("gsm") or os.path.basename(f).split("_")[0]] = c
miss = [g for g in WT + KO if g not in C]
if miss: raise SystemExit(f"missing counts: {miss}")
w = [C[g]["held_err_length"] for g in WT]; k = [C[g]["held_err_length"] for g in KO]
r = st.median(k) / st.median(w)
rel = lambda v: (max(v) - min(v)) / 2 / st.median(v) / math.sqrt(len(v))     # each arm's replicate half-spread, as a standard error
s_meas = r * math.sqrt(rel(w) ** 2 + rel(k) ** 2); s_pred = 0.10 / 2; s = math.sqrt(s_meas ** 2 + s_pred ** 2); pull = (r - 0.85) / s
cls = "pass" if abs(pull) <= 2 else ("watch" if abs(pull) <= 3 else "miss")
L = [f"PHY-086 P4 (mature B cells, TET2+TET3 removed): knockout over control {r:.4f}; sealed range 0.75-0.95: {'MET' if 0.75 <= r <= 0.95 else 'NOT MET'};"
     f" pull {pull:+.2f} ({cls})",
     f"controls {', '.join(f'{x:.5f}' for x in w)}; knockouts {', '.join(f'{x:.5f}' for x in k)}; measurement sigma {s_meas:.4f}, prediction sigma {s_pred:.4f}",
     "eraser share of the loss in mature B cells: " + f"{100 * (1 - r):.1f} % (stem cells 13.6 %, blood progenitors TET2 alone 9-10 %)"]
for g in WT + KO: L.append(f"{g} held_err_length {C[g]['held_err_length']:.5f} loose {C[g].get('loose_share', float('nan')):.3f} coupling {C[g].get('coupling', float('nan')):.3f} held_opp {C[g].get('held_opp')}")
open(os.path.join(H, "p4_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
