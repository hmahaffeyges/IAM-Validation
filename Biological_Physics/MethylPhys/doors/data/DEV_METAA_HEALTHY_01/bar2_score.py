"""DEV-METAA-HEALTHY-01 bar 2 and bar 3 scorer (written and committed before any bar-2 reading exists). Twenty healthy whole bloods from each
of five laboratories (bar2_labels.csv: the first 20 healthy arrays of each series in accession order; all spent), each read by run_sample.py.
It only summarizes the bundles: per laboratory, the median of whole-sample Met-A (A), of A_pred and of A / A_pred. Bar 2: each laboratory's
median A / A_pred within 0.03 of 1. Bar 3: the spread (largest minus smallest) of the five medians of A / A_pred smaller than that of A.
Usage: python3 bar2_score.py FOLDER_OF_BUNDLES [OUT.txt]"""
import sys, os, glob, json, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(src, "bar2_summary.txt")
L = pd.read_csv(os.path.join(H, "bar2_labels.csv"), dtype=str).set_index("gsm"); rows, seen = [], set()
for f in sorted(glob.glob(os.path.join(src, "**", "GSM*.json"), recursive=True)):
    g = os.path.basename(f).split(".")[0].split("_")[0]
    if g in seen or g not in L.index or f.endswith(("_values.json", "_planted.json")): continue
    seen.add(g); w = json.load(open(f)).get("met_a_whole") or {}; h = w.get("healthy_prediction") or {}
    rows.append(dict(gsm=g, A=w.get("A"), A_pred=h.get("A_pred"), ratio=h.get("A_over_pred")))
X = pd.DataFrame(rows).set_index("gsm").join(L); T = [f"DEV-METAA-HEALTHY-01 bar 2: {X.ratio.notna().sum()} of {len(L)} healthy bloods read"]
M = X.groupby("lab")[["A", "A_pred", "ratio"]].median()
for lab, r in M.iterrows():
    n = int(X[X.lab == lab].ratio.notna().sum()); T.append(f"{lab:10} n {n:2} | median A {r.A:.4f} | A_pred {r.A_pred:.4f} | A/A_pred {r.ratio:.4f} | within 0.03 of 1: {'yes' if abs(r.ratio - 1) <= 0.03 else 'no'}")
T.append(f"bar 2: {'met' if (abs(M.ratio - 1) <= 0.03).all() and len(M) == 5 else 'not met'}")
T.append(f"bar 3: spread of the five medians, A {M.A.max() - M.A.min():.4f} against A/A_pred {M.ratio.max() - M.ratio.min():.4f}: {'met' if M.ratio.max() - M.ratio.min() < M.A.max() - M.A.min() else 'not met'}")
open(out, "w").write("\n".join(T) + "\n"); print("\n".join(T))
