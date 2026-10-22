"""The C-Score band test (GSE184269, a laboratory that did not set the band), scored from the one command's own bundles (2026-10-22, chain
only; to-do step 37). Every reading and every tare is chain/MethylPhys_Interface/run_sample.py, run over the 24 whole bloods by
run_series.py (series.yml: specimen whole blood, scope slide_else_series: the same slide when it holds at least 3 others, else the series,
the rule boxruns/run3 and read_band_test.py used). The 10 granulocytes are not read: the chain refuses them at intake (no reference of their
own), as the committed scoring records. This file only reads rows.csv; it reads, tares and plants nothing. It applies the sealed bar (at
least 95 % of the whole bloods not above 1.152) and compares every number with the committed band_test_GSE184269.json.
Usage: python3 score_band_chain.py ROWS_CSV SAMPLES_CSV OUTDIR"""
import os, sys, json, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); BAND_HI, BAR = 1.152, 0.95
R = pd.read_csv(sys.argv[1]); out = sys.argv[3]; os.makedirs(out, exist_ok=True)
P = R[(R.pass_ == "pass2_slide_else_series") & (R.route == "idat") & (~R.planted.astype(bool))].drop_duplicates("gsm")
sc = P[P.C_rel.notna()]; above = int((sc.C_rel > BAND_HI).sum()); frac = 1 - above / len(sc) if len(sc) else float("nan")
mine = dict(n_whole_blood_scored=len(sc), above_limit=above, not_above_fraction=round(frac, 4) if len(sc) else None, met=bool(frac >= BAR) if len(sc) else None,
            C_rel_median=round(float(sc.C_rel.median()), 4) if len(sc) else None, C_rel_min=round(float(sc.C_rel.min()), 4) if len(sc) else None,
            C_rel_max=round(float(sc.C_rel.max()), 4) if len(sc) else None, whole_bloods_listed=int(len(P)), whole_bloods_withheld=int(P.C_rel.isna().sum()))
C = json.load(open(os.path.join(H, "band_test_GSE184269.json")))
cmp = {k: (mine[k], C.get(k)) for k in mine}; same = all(a == b for a, b in cmp.values())
L = ["C-Score band test, GSE184269, through the one command (every reading and tare by run_sample.py)",
     f"whole bloods {mine['whole_bloods_listed']} listed, {mine['n_whole_blood_scored']} scored, {mine['whole_bloods_withheld']} withheld (reasons: {sorted(set(P[P.C_rel.isna()].reason.dropna().astype(str)))})",
     f"above 1.152: {above} | not above {mine['not_above_fraction']} (bar at least {BAR:.0%}) -> {'MET' if mine['met'] else 'NOT MET'} | C_rel median {mine['C_rel_median']}, range {mine['C_rel_min']} to {mine['C_rel_max']}",
     "against the committed scoring (this run, committed): " + "; ".join(f"{k} {a} vs {b}" for k, (a, b) in cmp.items()),
     "EVERY COMPARED NUMBER IDENTICAL" if same else "DIFFERS FROM THE COMMITTED SCORING (see the pairs above)"]
json.dump(dict(mine=mine, compare={k: list(v) for k, v in cmp.items()}, identical=same), open(os.path.join(out, "band_chain.json"), "w"), indent=1)
open(os.path.join(out, "band_chain_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
