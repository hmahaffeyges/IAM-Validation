"""DEV-ERASE-01, the division route (written and committed before any count exists; plan 14.3). It only summarizes what run_sample.py's
Stage Q2 count recorded for each IMR90 sample (held error, its opportunities, loose share) and pools each state (opportunity-weighted held
error; median loose share). Usage: python3 division_score.py FOLDER [OUT.txt]"""
import sys, os, glob, json, csv, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); S = {r["gsm"]: r for r in csv.DictReader(open(os.path.join(H, "division_samples.csv")))}
Q = {}
for f in glob.glob(os.path.join(sys.argv[1], "**", "*_q2.json"), recursive=True):
    q = json.load(open(f)); Q[q["gsm"]] = q
L = ["DEV-ERASE-01 division route (GSE48580 IMR90), Stage Q2 counts at 10 million read pairs"]
for st in ("proliferating", "senescent", "SV40 bypass"):
    gs = sorted({g for g, r in S.items() if r["genotype"] == st}); have = [g for g in gs if g in Q]
    for g in have: L.append(f"  {st:14} {g} held error {Q[g]['held_err']:.5f} (opp {Q[g]['held_opp']}) | loose share {Q[g].get('loose_share')}")
    if have:
        e = sum(Q[g]["held_err"] * Q[g]["held_opp"] for g in have) / sum(Q[g]["held_opp"] for g in have)
        L.append(f"{st:14} {len(have)} of {len(gs)} | pooled held error {e:.5f} | loose share median {np.median([Q[g]['loose_share'] for g in have if Q[g].get('loose_share') is not None]):.4f}")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(sys.argv[1], "division_summary.txt"); open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
