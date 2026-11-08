"""DEV-REPLICATE-01 wave C, a TCGA project read per person against IAM-Atlas v2.1 (written and committed 2026-11-07 before any of its readings
exists). Each array read by the one command on every atlas cell with --dev-pred-raw. Reference: IAM-Atlas v2.1 only; no TCGA group (tumor or
adjacent normal) is a reference for any other array. Refuses a bundle whose healthy prediction was not on the array's own values.
Recorded per person, by GDC sample type (label kept for scoring only): whole-sample Met-A over the healthy prediction (A / A_pred), inside
Normal (0.95 to 1.05), above or below; Met-A at the identity sites of the cells found (untared, as written by the chain); the cell the
Deconvolver found most of. No bar is scored: the set holds no healthy donors, so nothing here is a commissioning test; it is a replay of spent
data, stated as such.
Usage: python3 tcga_person_score.py LABELS.csv FOLDER_OF_BUNDLES OUT.csv OUT.txt"""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import cells_found as CF
import pandas as pd
lab, src, oc, ot = sys.argv[1:5]
L = pd.read_csv(lab, dtype=str).set_index("gsm"); rows, REF, seen = [], [], set()
crow = []
for f in sorted(glob.glob(os.path.join(src, "**", "TCGA-*.json"), recursive=True)):
    g = os.path.basename(f)[:-5]
    if g in seen or g not in L.index or f.endswith(("_values.json", "_planted.json")): continue
    seen.add(g); B = json.load(open(f)); w = B.get("met_a_whole") or {}; h = w.get("healthy_prediction") or {}
    if not str(h.get("values") or "").startswith("the array's own values"): REF.append(g); continue
    idn = ((w.get("by_place") or {}).get("identity sites of the cells found") or {}).get("A")
    sh = sorted(((c.get("share") or 0, c.get("cell")) for c in (B.get("cell_shares") or {}).get("cells", [])), reverse=True)
    crow.extend(dict(gsm=g, case=L.loc[g, "case"], group=L.loc[g, "group"], **c) for c in CF.cells_found(B))   # 2026-11-07: every cell the Deconvolver found
    rows.append(dict(gsm=g, case=L.loc[g, "case"], group=L.loc[g, "group"], ratio=h.get("A_over_pred"), identity_sites_A=idn,
                     top_cell=sh[0][1] if sh else None, top_share=round(sh[0][0], 3) if sh else None))
X = pd.DataFrame(rows, columns=["gsm", "case", "group", "ratio", "identity_sites_A", "top_cell", "top_share"])
for c in ("ratio", "identity_sites_A"): X[c] = pd.to_numeric(X[c], errors="coerce")
X["place"] = X.ratio.map(lambda r: "inside Normal" if abs(r - 1) <= 0.05 else ("above Normal" if r > 1.05 else "below Normal") if pd.notna(r) else "not read")
X.to_csv(oc, index=False)
CX = pd.DataFrame(crow); CX.to_csv(oc.replace(".csv", "_cells.csv"), index=False)   # each cell found, each person
T = [f"{os.path.basename(lab).replace('_labels.csv', '')} read per person against IAM-Atlas v2.1 (prediction on each array's own values): {len(X)} of {len(L)} read",
     f"refused (prediction not on the array's own values): {len(REF)}"]
for gname, g in X.groupby("group"):
    c = g.place.value_counts().to_dict()
    T.append(f"  {gname} ({len(g)}): inside Normal {c.get('inside Normal', 0)}, above {c.get('above Normal', 0)}, below {c.get('below Normal', 0)} | A / A_pred median "
             f"{g.ratio.median():.4f} | identity sites A median {g.identity_sites_A.median():.4f} | most often largest cell {g.top_cell.mode().iloc[0] if g.top_cell.notna().any() else 'none'}")
P = X[X.group.isin(["Primary Tumor", "Solid Tissue Normal"])].pivot_table(index="case", columns="group", values=["ratio", "identity_sites_A"], aggfunc="first").dropna()
if len(P): T.append(f"  same person, tumor against adjacent normal ({len(P)} people): A / A_pred higher in the tumor for {int((P['ratio']['Primary Tumor'] > P['ratio']['Solid Tissue Normal']).sum())}")
if len(CX):
    T.append("each cell the Deconvolver found, read against its IAM-Atlas v2.1 reference in the sample's own shares (median Met-A; C-Score within its limit):")
    for (gr, ce), g_ in CX.groupby(["group", "cell"]):
        if len(g_) >= 3: T.append(f"  {gr[:24]:24} {ce[:34]:34} people {len(g_):4} | median share {g_.share.median():.3f} | Met-A {g_.met_a.median():.4f} | C within limit {int((g_.c_within == True).sum())} of {int(g_.c_within.notna().sum())}")
T.append("no bar scored (no healthy donors in this set); a replay of spent data, recorded")
open(ot, "w").write("\n".join(T) + "\n"); print("\n".join(T))
