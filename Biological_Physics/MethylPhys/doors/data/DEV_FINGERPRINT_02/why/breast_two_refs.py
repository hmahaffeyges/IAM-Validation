"""DEV-FINGERPRINT-02 breast reviews (descriptive): MCF-7 per file against its two sealed breast references (mammary epithelial cells; MCF 10A),
Arm B excess and Arm G gain from the committed scored files. Writes breast_two_refs_output.csv. Usage: python3 breast_two_refs.py"""
import os, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); FP = os.path.join(HERE, ".."); L = lambda f: json.load(open(os.path.join(FP, f)))
rows = []
for p in ("breast_hmec", "breast_mcf10a"):
    b, g = L(f"scored_{p}.json"), L(f"scored_gain_{p}.json")
    for f in b["cancer_excess"]: rows.append(dict(pair=p, file=f, armB_excess=b["cancer_excess"][f], armG_gain=g["cancer_gain"][f]))
T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, "breast_two_refs_output.csv"), index=False)
print(T.groupby("pair")[["armB_excess", "armG_gain"]].agg(["min", "max"]).round(4).to_string())
