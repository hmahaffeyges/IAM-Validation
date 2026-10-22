"""Group-A's sealed bars, scored from the one command's own bundles (2026-10-22, chain only; to-do step 35). Every reading and every tare is
chain/MethylPhys_Interface/run_sample.py, run over the series by run_series.py (series.yml: plant neutrophil, scope slide_else_series):
the IDAT readings tared against the other IDAT readings, the planted half and the unplanted half both read from their saved values and tared
against the unplanted half read the same way, as the sealed scorer did. This file only reads rows.csv; it reads, tares and plants nothing.
It prints the three sealed bars and compares them with the committed scored_groupa_01.json.
Usage: python3 score_groupa_chain.py ROWS_CSV SAMPLES_CSV OUTDIR   (writes OUTDIR/groupa_chain_summary.txt and OUTDIR/groupa_chain.json)"""
import os, sys, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
R = pd.read_csv(sys.argv[1]); S = pd.read_csv(sys.argv[2]); out = sys.argv[3]
def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))); d = float(a.mean() - b.mean())
    return dict(n1=len(a), n2=len(b), mean1=round(float(a.mean()), 5), mean2=round(float(b.mean()), 5), rel_difference=round(d / float(np.concatenate([a, b]).mean()), 5), t=round(d / se, 3))
P = R[R.pass_ == "pass2_slide_else_series"].merge(S[["gsm", "sex", "half"]], on="gsm", how="left")
I = P[P.route == "idat"].drop_duplicates("gsm"); V = P[P.route == "values"].drop_duplicates(["gsm", "planted"])
p1 = R[(R.pass_ == "pass1") & (R.route == "idat")]; q = p1.intake.fillna("").str.contains("QUARANTINE")
admitted = int(len(S) - q.sum() - (len(S) - len(p1))); n_read = int(I.A_rel.notna().sum())
b1 = admitted >= len(S) / 2 and n_read >= np.ceil(0.9 * admitted)
s2 = welch(I[I.sex == "female"].A_rel.dropna(), I[I.sex == "male"].A_rel.dropna()); b2 = abs(s2["t"]) < 2 and abs(s2["rel_difference"]) < 0.02
s3 = welch(V[V.planted.astype(bool)].A_rel.dropna(), V[~V.planted.astype(bool)].A_rel.dropna()); b3 = s3["t"] > 2
C = json.load(open(os.path.join(H, "scored_groupa_01.json")))
cmp = dict(n_read=(n_read, C["B1"]["n_read"]), B2_women=(s2["mean1"], C["B2"]["mean1"]), B2_men=(s2["mean2"], C["B2"]["mean2"]), B2_t=(s2["t"], C["B2"]["t"]),
           B3_planted=(s3["mean1"], C["B3"]["mean1"]), B3_unplanted=(s3["mean2"], C["B3"]["mean2"]), B3_t=(s3["t"], C["B3"]["t"]))
same = all(a == b for a, b in cmp.values())
L = [f"Group-A from the one command's bundles: B1 read {n_read} of {admitted} admitted -> {'MET' if b1 else 'NOT MET'}",
     f"B2 sex null, tared: women {s2['mean1']} (n {s2['n1']}) men {s2['mean2']} (n {s2['n2']}), {100 * s2['rel_difference']:+.2f} %, {s2['t']} SE -> {'MET' if b2 else 'NOT MET'}",
     f"B3 planted 1 % loss: planted {s3['mean1']} (n {s3['n1']}) unplanted {s3['mean2']} (n {s3['n2']}), {100 * s3['rel_difference']:+.2f} %, {s3['t']} SE -> {'MET' if b3 else 'NOT MET'}",
     "against the committed scoring (this run, committed): " + "; ".join(f"{k} {a} vs {b}" for k, (a, b) in cmp.items()),
     "EVERY COMPARED NUMBER IDENTICAL" if same else "DIFFERS FROM THE COMMITTED SCORING (see the pairs above)"]
json.dump(dict(B1=b1, B2=s2 | dict(met=b2), B3=s3 | dict(met=b3), compare={k: list(v) for k, v in cmp.items()}, identical=same), open(os.path.join(out, "groupa_chain.json"), "w"), indent=1, default=str)
open(os.path.join(out, "groupa_chain_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
