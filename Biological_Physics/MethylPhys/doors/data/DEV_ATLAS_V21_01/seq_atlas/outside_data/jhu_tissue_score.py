"""The Johns Hopkins tumor-against-own-normal test (GSE86340; written and committed before any of its tissue samples is read). The instrument's
job: a person's tumor, read against that same person's normal tissue from the same laboratory, reads abnormal. This is the other half of the
same-laboratory test: the CD4 test (jhu_cd4_score.py) asks whether a laboratory's healthy samples read Normal once tared on that laboratory's
own healthy samples; this one asks whether a tumor still reads abnormal when the reference is the same person's own normal tissue, read on the
same laboratory's equipment, so a laboratory offset cannot make the difference.
Samples (jhu_tissue_samples.csv): five people with tumor and normal tissue (three liver, two lung), plus one lung tumor and one lung normal whose
labels do not say they are one person (recorded, not scored). Every sample read at full depth by the one command, after the genome-coverage check.
Self-reference: for each reading, tumor / own normal (stage_metaa_atlas.tare's rule with one reference, the person's own normal tissue).
Normal band: 0.95 to 1.05, the gauge's Normal. No direction is assumed: a tumor departs either way.
Bars (on the five labeled pairs; at least 4 pairs read, else the test is not read):
  1. whole-sample Met-A: at least 4 of the 5 tumors read outside 0.95 to 1.05 against their own normal;
  2. the islands: at least 4 of the 5 tumors read outside 0.95 to 1.05 against their own normal.
Recorded, not scored: every other reading's tumor / own normal (the domains, the C-Score, coupling-A, IAM-A), each tissue's untared readings
against the IAM Sequencing Atlas (normal tissue next to a tumor may carry a field change), the cell shares, and the unlabeled lung pair.
A sample the chain refuses is listed with its reason.
Usage: python3 jhu_tissue_score.py BUNDLE_DIR OUT.csv OUT.txt"""
import os, sys, glob, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, "jhu_tissue_samples.csv")).drop_duplicates("gsm")
READ = ["Met-A whole sample", "islands", "domains", "C-Score", "coupling-A", "IAM-A"]
rows = []
for r in S.itertuples():
    f = sorted(glob.glob(os.path.join(sys.argv[1], "**", r.gsm + "_bundle.json"), recursive=True)); o = dict(gsm=r.gsm, person=r.person, organ=r.organ, kind=r.kind)
    if f:
        B = json.load(open(f[-1])); D = B.get("development") or {}; T = (B.get("territories") or {}).get("sets") or {}
        sh = sorted(((x["cell"], x["share"]) for x in ((B.get("cell_shares") or {}).get("cells") or [])), key=lambda z: -(z[1] or 0))[:3]
        o.update({"status": B.get("status", "read"), "Met-A whole sample": (B.get("met_a_whole") or {}).get("A"),
                  "islands": (T.get("islands") or {}).get("A"), "domains": (T.get("domains") or {}).get("A"),
                  "C-Score": ((B.get("cscore_each_cell") or {}).get("whole_genome") or {}).get("C"), "coupling-A": (B.get("coupling_a") or {}).get("A"),
                  "IAM-A": (B.get("iam_a") or {}).get("A"), "top cells": "; ".join(f"{c} {s:.3f}" for c, s in sh if s is not None),
                  "not read": B.get("reason")})
    else: o["not read"] = "no bundle"
    rows.append(o)
d = pd.DataFrame(rows); d.to_csv(sys.argv[2], index=False)
pairs = []
for p, g in d.groupby("person"):
    t, n = g[g.kind == "tumor"], g[g.kind == "normal"]
    if len(t) != 1 or len(n) != 1: continue
    q = dict(person=p, organ=t.organ.iloc[0])
    for k in READ:
        a, b = pd.to_numeric(t[k], errors="coerce").iloc[0] if k in t else np.nan, pd.to_numeric(n[k], errors="coerce").iloc[0] if k in n else np.nan
        q[k] = a / b if pd.notna(a) and pd.notna(b) and b else np.nan
    pairs.append(q)
P = pd.DataFrame(pairs); P.to_csv(sys.argv[2].replace(".csv", "_pairs.csv"), index=False)
lab = P[P.person != "unlabeled"] if len(P) else P
out = lambda col: (lab[col].dropna(), int(((lab[col] < 0.95) | (lab[col] > 1.05)).sum()))
w, nw = out("Met-A whole sample") if len(lab) else (pd.Series(dtype=float), 0); i, ni = out("islands") if len(lab) else (pd.Series(dtype=float), 0)
ok = len(lab) >= 4 and len(w) >= 4; b1 = ok and nw >= 4; b2 = len(i) >= 4 and ni >= 4
L = [f"Johns Hopkins tumor against own normal (GSE86340): {int(d['not read'].isna().sum())} of {len(d)} read; {len(lab)} labeled pairs",
     "each pair, tumor / own normal: " + " | ".join(f"{r.person} ({r.organ}): " + ", ".join(f"{k} {r[k]:.4f}" for k in READ if pd.notna(r[k])) for _, r in P.iterrows()),
     f"bar 1 whole-sample Met-A outside 0.95 to 1.05 in at least 4 of 5: {nw} of {len(w)}: {'met' if b1 else 'not met'}",
     f"bar 2 islands outside 0.95 to 1.05 in at least 4 of 5: {ni} of {len(i)}: {'met' if b2 else 'not met'}",
     f"test: {'met' if b1 and b2 else 'not met'}"]
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
