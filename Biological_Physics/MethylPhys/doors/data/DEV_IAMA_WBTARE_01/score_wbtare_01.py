"""DEV-IAMA-WBTARE-01 scoring, by the bars in doors/DEV_IAMA_WBTARE_01.md (fixed 2026-10-09 before reading).
Inputs: the 14 Box Run 2 session 4 bundles (s3://methylphys-data-945451304272-us-west-2-an/results/BOXRUN2/session4/<run>_bundle.json;
iam_a.eps, errors, opportunities; iam_a_intake Q0.3 conversion) and boxruns/run2/GSE128731_runs.csv (donor, kit, repeat).
A_rel = H(ε) ÷ median H(ε) of the other donors' rep1 runs on the same kit (3 references). Runs refused at intake carry no ε.
Run: python3 score_wbtare_01.py OUT.csv"""
import os, sys, json, math, re, boto3, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); B = "methylphys-data-945451304272-us-west-2-an"; s3 = boto3.client("s3")
RR = pd.read_csv(os.path.join(HERE, "../../../boxruns/run2/GSE128731_runs.csv")).set_index("Run")
def H(e): return -(e * math.log2(e) + (1 - e) * math.log2(1 - e))
rows = []
for o in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix="results/BOXRUN2/session4/"):
    for k in [x["Key"] for x in o.get("Contents", []) if x["Key"].endswith("_bundle.json")]:
        b = json.loads(s3.get_object(Bucket=B, Key=k)["Body"].read()); run = k.split("/")[-1][:10]; t = RR.loc[run, "title"]
        cv = next(c for c in b["iam_a_intake"]["checks"] if c["step"].startswith("Q0.3"))
        m = re.match(r"(Sample\d)_(Swift|TruSeq)_HiSeqX(?:_(rep\d))?", t)
        rows.append(dict(run=run, donor=m.group(1), kit=m.group(2), rep=m.group(3) or "rep1", conversion=cv["value"], intake=cv["result"],
                         eps=(b.get("iam_a") or {}).get("eps")))
R = pd.DataFrame(rows).sort_values(["kit", "donor", "rep"]).reset_index(drop=True)
def a_rel(r):
    ref = R[(R.kit == r.kit) & (R.rep == "rep1") & (R.donor != r.donor) & R.eps.notna()]
    if pd.isna(r.eps) or len(ref) < 3: return None, len(ref)
    return H(r.eps) / float(np.median([H(e) for e in ref.eps])), len(ref)
R["A_rel"], R["n_refs"] = zip(*[a_rel(r) for _, r in R.iterrows()]); R.to_csv(sys.argv[1], index=False)
print(R.round(5).to_string(index=False))
t1 = R[(R.rep == "rep1") & R.A_rel.notna()]
print(f"\nBar 1 (8 of 8 rep1 within 0.95-1.05): {int(t1.A_rel.between(0.95, 1.05).sum())} of {len(t1)} readable ({', '.join(f'{d} {k} {a:.4f}' for d, k, a in zip(t1.donor, t1.kit, t1.A_rel))}); "
      f"{8 - len(t1)} rep1 runs not read (intake)")
print("Bar 2 (kit offset, 4 donors both kits):", "not assessable: no Swift rep1 run passed intake" if R[(R.kit == "Swift") & (R.rep == "rep1")].eps.isna().all() else "see table")
r2 = R[(R.rep == "rep2") & R.eps.notna()]
print("Bar 3 (rep2 vs rep1, 6 pairs):", f"{len(r2)} rep2 run(s) read: " + ", ".join(f"{d} {k} (A_rel {a})" for d, k, a in zip(r2.donor, r2.kit, r2.A_rel)) + " -> no pair with a readable rep1 and 3 references" if len(r2) else "none read")
