"""DEV-ALL-INFO-01, the coupling's healthy reference (written before any of its readings exists). From the one command's bundles of the
atlas's healthy sequencing samples (healthy_seq.csv): each cell type's healthy coupling (median of its samples), and the Normal band of
coupling-A, each sample's coupling over the median of the OTHER samples of its own cell type (leave-one-out, cell types with 2 or more
samples), 2.5th to 97.5th percentile. Also reported, nothing scored: Met-A against IAM-Atlas v2.1 for each sample (the sequencing offset by
cell type) and IAM-A. Usage: python3 build_reference.py BUNDLE_DIR OUT_DIR"""
import sys, os, json, glob, csv, statistics as st, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); S = list(csv.DictReader(open(os.path.join(H, "healthy_seq.csv"))))
rows = []
for r in S:
    f = glob.glob(os.path.join(sys.argv[1], "**", r["gsm"] + "_bundle.json"), recursive=True)
    if not f: rows.append(dict(gsm=r["gsm"], cell=r["cell"], read=False)); continue
    B = json.load(open(f[0])); q = B.get("iam_a") or {}; c = q.get("count") or {}; m = (B.get("development") or {}).get("metaa_atlas") or {}
    rows.append(dict(gsm=r["gsm"], cell=r["cell"], read=True, coupling=c.get("coupling"), iam_a=q.get("A"), met_a_atlas=m.get("A")))
os.makedirs(sys.argv[2], exist_ok=True)
by = {}
for x in rows:
    if x.get("coupling") is not None: by.setdefault(x["cell"], []).append(x)
ref = {c: st.median(v["coupling"] for v in X) for c, X in by.items()}
loo = []
for c, X in by.items():
    if len(X) < 2: continue
    for x in X:
        o = [v["coupling"] for v in X if v is not x]; x["coupling_A_loo"] = x["coupling"] / st.median(o); loo.append(x["coupling_A_loo"])
band = dict(lo=float(np.percentile(loo, 2.5)), hi=float(np.percentile(loo, 97.5)), median=float(np.median(loo)), n=len(loo)) if loo else None
json.dump(dict(rule="coupling-A = coupling / its cell type's healthy coupling; band = 2.5-97.5 percentile of leave-one-out coupling-A",
               reference=ref, band=band, n_read=sum(x["read"] for x in rows), n_listed=len(rows), source="GSE186458 (atlas sequencing inputs), one laboratory"),
          open(os.path.join(sys.argv[2], "coupling_reference_v1.json"), "w"), indent=1)
with open(os.path.join(sys.argv[2], "healthy_readings.csv"), "w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=["gsm", "cell", "read", "coupling", "coupling_A_loo", "iam_a", "met_a_atlas"]); w.writeheader(); w.writerows(rows)
L = [f"read {sum(x['read'] for x in rows)} of {len(rows)}; coupling recorded on {sum(len(v) for v in by.values())}",
     f"coupling-A band (leave-one-out): {band}", "cell | samples | healthy coupling | median Met-A against the atlas | median IAM-A"]
for c in sorted(by):
    X = by[c]; ma = [x["met_a_atlas"] for x in X if x["met_a_atlas"] is not None]; ia = [x["iam_a"] for x in X if x["iam_a"] is not None]
    L.append(f"{c} | {len(X)} | {ref[c]:.4f} | {st.median(ma) if ma else ''} | {st.median(ia) if ia else ''}")
open(os.path.join(sys.argv[2], "coupling_reference_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L[:3]))
