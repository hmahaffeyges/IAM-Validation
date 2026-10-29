"""DEV-TERRITORIES-01: the territories' healthy band on sequencing (written before any reading on the chain that reads the territories in
sequencing units). IAM-Atlas v2.1's 153 healthy sequencing inputs (DEV_ALL_INFO_01/coupling_02/healthy_seq.csv), each read by the one command
with its sorted cell as the specimen. For each territory (islands, domains): the 2.5th to 97.5th percentile of the healthy readings, the
median, and the bar: the median healthy reading within 0.05 of 1 (the instrument's job: a healthy cell reads near 1 on each territory).
Usage: python3 terr_seq_band.py BUNDLE_DIR OUT_DIR [CHAIN_BAND.json]"""
import sys, os, json, glob, csv, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
S = list(csv.DictReader(open(os.path.join(H, "..", "..", "DEV_ALL_INFO_01", "coupling_02", "healthy_seq.csv"))))
V = {"islands": [], "domains": []}
for r in S:
    f = glob.glob(os.path.join(sys.argv[1], "**", r["gsm"] + "_bundle.json"), recursive=True)
    if not f: continue
    t = (json.load(open(f[0])).get("territories") or {}).get("sets") or {}
    for k in V:
        a = (t.get(k) or {}).get("A")
        if a is not None: V[k].append(a)
out = {k: dict(lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), median=float(np.median(v)), n=len(v)) for k, v in V.items() if v}
out["rule"] = "2.5th to 97.5th percentile of the territory readings of IAM-Atlas v2.1's healthy sequencing inputs, in sequencing units (seq_units)"
os.makedirs(sys.argv[2], exist_ok=True); json.dump(out, open(os.path.join(sys.argv[2], "territories_seq_band_v1.json"), "w"), indent=1)
if len(sys.argv) > 3: json.dump(out, open(sys.argv[3], "w"), indent=1)
L = [f"{k}: band {x['lo']:.4f} to {x['hi']:.4f}, median {x['median']:.4f}, n {x['n']}; bar (median within 0.05 of 1): {'met' if abs(x['median'] - 1) <= 0.05 else 'not met'}"
     for k, x in out.items() if k != "rule"]
open(os.path.join(sys.argv[2], "territories_seq_band_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
