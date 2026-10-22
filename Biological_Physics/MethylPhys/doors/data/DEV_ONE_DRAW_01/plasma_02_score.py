"""DEV-ONE-DRAW-01, exploration 2 (2026-10-22; not sealed): plasma cell-free DNA through the one command. Run by series.yml on
plasma_gse122126_samples.csv (GSE122126, the published plasma array series: 58 cell-free DNA arrays, spent): every array read by run_sample.py
with --specimen plasma, so the IAM-Deconvolver splits it on IAM-Atlas v2.1's full panel and Met-A reads each source it finds. This file only
reads the bundles the one command wrote (pass 1, no references).
Reported by group (healthy, sepsis, each cancer, islet-transplant recipients before and 1 and 2 hours after): the sources found above a share of
0.01 with their median share; the largest non-blood sources; pancreatic sources in the transplant recipients by time point (donor islets
release DNA into the recipient's blood, a positive control); Met-A of each source read, untared. Usage: python3 plasma_02_score.py ROWS SAMPLES OUT"""
import os, sys, json, glob, collections, numpy as np, pandas as pd
rows, samples, out = sys.argv[1], sys.argv[2], sys.argv[3]; S = pd.read_csv(samples)
BLOOD = ("neutrophil", "monocyte", "b cell", "t cell", "nk cell", "macrophage", "dendritic", "erythro", "megakaryo", "hsc", "cmp", "gmp", "mep", "mpp", "plasma cell", "treg", "regulatory t")
def blood(c): return any(w in c.lower() for w in BLOOD)
def group(r):
    s = str(r.source).lower()
    if "islet" in s: return "islet transplant, " + ("before" if "pre-" in s else "1 hour after" if "1 hour" in s else "2 hours after")
    if "normal" in s or "pooled" in s: return "healthy"
    for k in ("sepsis", "lung", "colon", "breast", "cup"):
        if k in s: return k if k != "cup" else "cancer of unknown primary"
    return s
R = []
for r in S.itertuples():
    f = os.path.join(out, "pass1", r.gsm + ".json")
    if not os.path.exists(f): R.append(dict(gsm=r.gsm, group=group(r), status="no bundle")); continue
    B = json.load(open(f)); sh = (B.get("cell_shares") or {}).get("cells") or []; E = {x["cell"]: x for x in (B.get("met_a_each_cell") or {}).get("cells", [])}
    R.append(dict(gsm=r.gsm, group=group(r), status="read" if sh else "no shares: " + str((B.get("cell_shares") or {}).get("withheld") or B.get("refusal")),
                  shares={x["cell"]: x["share"] for x in sh}, meta={c: e.get("A") for c, e in E.items() if e.get("A") is not None}))
L = [f"DEV-ONE-DRAW-01 exploration 2: GSE122126 plasma cell-free DNA, {len(S)} arrays, read by run_sample.py --specimen plasma (full atlas panel)",
     "status: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(x["status"].split(":")[0] for x in R).items())]
for g in sorted(set(x["group"] for x in R)):
    X = [x for x in R if x["group"] == g and x.get("shares")]
    if not X: L.append(f"{g}: none read"); continue
    cells = sorted({c for x in X for c in x["shares"]}); med = {c: float(np.median([x["shares"].get(c, 0.0) for x in X])) for c in cells}
    top = [c for c in sorted(med, key=lambda c: -med[c]) if med[c] >= 0.01]
    nb = sum(med[c] for c in cells if not blood(c))
    L.append(f"{g} (n {len(X)}): non-blood share (median sum) {nb:.3f} | sources above 0.01: " + ", ".join(f"{c} {med[c]:.3f}" for c in top))
    pan = {c: med[c] for c in cells if "pancrea" in c.lower() or "islet" in c.lower() or "beta" in c.lower() or "acinar" in c.lower() or "duct" in c.lower()}
    if pan: L.append(f"   pancreatic sources: " + ", ".join(f"{c} {v:.4f}" for c, v in sorted(pan.items(), key=lambda kv: -kv[1])))
    ma = {c: float(np.median([x["meta"][c] for x in X if c in x["meta"]])) for c in top if any(c in x["meta"] for x in X)}
    if ma: L.append("   Met-A (median, untared) of those sources: " + ", ".join(f"{c} {v:.3f}" for c, v in ma.items()))
pd.DataFrame([dict(gsm=x["gsm"], group=x["group"], status=x["status"], **{f"share:{c}": v for c, v in (x.get("shares") or {}).items()},
                   **{f"metaA:{c}": v for c, v in (x.get("meta") or {}).items()}) for x in R]).to_csv(os.path.join(out, "plasma_02_rows.csv"), index=False)
open(os.path.join(out, "plasma_02_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
