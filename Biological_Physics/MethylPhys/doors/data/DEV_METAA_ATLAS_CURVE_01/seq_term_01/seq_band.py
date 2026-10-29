"""DEV-METAA-ATLAS-CURVE-01: Met-A's healthy band on sequencing (written before any of these readings exists). Each of IAM-Atlas v2.1's 153
healthy sequencing inputs (healthy_seq.csv) read by the one command with its sorted cell type, the atlas column in sequencing units
(sequencing_term_v1). The band: the 2.5th to 97.5th percentile of all readings; also each cell type's median, the readings against depth,
and the bar: the median healthy reading within 0.05 of 1 (the instrument's job: a healthy cell reads near 1). Usage:
python3 seq_band.py BUNDLE_DIR OUT_DIR [CHAIN_BAND.json]"""
import sys, os, json, glob, csv, statistics as st, numpy as np
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__)); S = list(csv.DictReader(open(os.path.join(H, "healthy_seq.csv")))); rows = []
for r in S:
    f = glob.glob(os.path.join(sys.argv[1], "**", r["gsm"] + "_bundle.json"), recursive=True)
    if not f: continue
    B = json.load(open(f[0])); m = (B.get("development") or {}).get("metaa_atlas") or {}; c = (B.get("iam_a") or {}).get("count") or {}
    if m.get("A") is not None: rows.append(dict(gsm=r["gsm"], cell=r["cell"], A=m["A"], depth=c.get("held_opp")))
A = np.array([x["A"] for x in rows]); os.makedirs(sys.argv[2], exist_ok=True)
band = dict(lo=float(np.percentile(A, 2.5)), hi=float(np.percentile(A, 97.5)), median=float(np.median(A)), n=len(A),
            rule="2.5th to 97.5th percentile of Met-A against IAM-Atlas v2.1 on the atlas's healthy sequencing inputs (sequencing_term_v1)",
            source="GSE186458 (atlas sequencing inputs), one laboratory")
json.dump(band, open(os.path.join(sys.argv[2], "metaa_seq_band_v1.json"), "w"), indent=1)
if len(sys.argv) > 3: json.dump(band, open(sys.argv[3], "w"), indent=1)
with open(os.path.join(sys.argv[2], "metaa_seq_readings.csv"), "w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=["gsm", "cell", "A", "depth"]); w.writeheader(); w.writerows(rows)
rr = spearmanr([x["depth"] for x in rows], A) if len(rows) > 2 else None
L = [f"read {len(rows)} of {len(S)}", f"band {band['lo']:.4f} to {band['hi']:.4f}, median {band['median']:.4f}",
     f"Met-A vs depth: rho {rr.correlation:+.3f}, p {rr.pvalue:.2g}" if rr else "", f"bar (median within 0.05 of 1): {'met' if abs(band['median'] - 1) <= 0.05 else 'not met'}",
     "cell | samples | median Met-A"]
by = {}
for x in rows: by.setdefault(x["cell"], []).append(x["A"])
L += [f"{c} | {len(v)} | {st.median(v):.4f}" for c, v in sorted(by.items())]
open(os.path.join(sys.argv[2], "metaa_seq_band_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L[:4]))
