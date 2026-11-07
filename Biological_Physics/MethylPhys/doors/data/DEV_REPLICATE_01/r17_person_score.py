"""DEV-REPLICATE-01, R17 breast tissue (GSE69914) read per person (written and committed 2026-11-07 before any of its readings exists). Each array
read again by the one command on every IAM-Atlas v2.1 cell with --dev-pred-raw (series run r17_raw): Met-A for the whole sample against the atlas
in the array's own measured mix, over the healthy prediction made on the array's own values (A / A_pred). Reference: IAM-Atlas v2.1 only; no
group is used as a reference for any other array. Refuses a bundle whose prediction was not on the array's own values.
Bar (the instrument's job): at least 45 of the 50 healthy donors each read A / A_pred inside Normal, 0.95 to 1.05.
Recorded, not scored: every other array's A / A_pred, person by person, with its group (normal-adjacent, BRCA1 carriers, cancer): how many sit
outside Normal, on which side, and the cell the Deconvolver found most of.
Usage: python3 r17_person_score.py FOLDER_OF_BUNDLES OUT.csv OUT.txt"""
import sys, os, glob, json
import pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); src, oc, ot = sys.argv[1:4]
L = pd.read_csv(os.path.join(H, "r17_breast_labels.csv"), dtype=str).set_index("gsm"); rows, REF, seen = [], [], set()
for f in sorted(glob.glob(os.path.join(src, "**", "GSM*.json"), recursive=True)):
    g = os.path.basename(f).split(".")[0].split("_")[0]
    if g in seen or g not in L.index or f.endswith(("_values.json", "_planted.json")): continue
    seen.add(g); B = json.load(open(f)); w = B.get("met_a_whole") or {}; h = w.get("healthy_prediction") or {}
    if not str(h.get("values") or "").startswith("the array's own values"): REF.append(g); continue
    sh = sorted(((c.get("share") or 0, c.get("cell")) for c in (B.get("cell_shares") or {}).get("cells", [])), reverse=True)
    rows.append(dict(gsm=g, group=L.loc[g, "group"], ratio=h.get("A_over_pred"), A_raw=h.get("A_raw"), A_pred=h.get("A_pred"),
                     top_cell=sh[0][1] if sh else None, top_share=round(sh[0][0], 3) if sh else None))
X = pd.DataFrame(rows, columns=["gsm", "group", "ratio", "A_raw", "A_pred", "top_cell", "top_share"]); X["ratio"] = pd.to_numeric(X.ratio, errors="coerce")
X["place"] = X.ratio.map(lambda r: "inside Normal" if abs(r - 1) <= 0.05 else ("above Normal" if r > 1.05 else "below Normal") if pd.notna(r) else "not read")
X.to_csv(oc, index=False)
T = [f"R17 breast tissue (GSE69914) read per person against IAM-Atlas v2.1 (prediction on each array's own values): {len(X)} of {len(L)} read",
     f"refused (prediction not on the array's own values): {len(REF)}"]
for gname, g in X.groupby("group"):
    c = g.place.value_counts().to_dict()
    T.append(f"  {gname} ({len(g)}): inside Normal {c.get('inside Normal', 0)}, above {c.get('above Normal', 0)}, below {c.get('below Normal', 0)} | median {g.ratio.median():.4f}, range {g.ratio.min():.3f} to {g.ratio.max():.3f}")
hd = X[X.group == "normal (healthy donor)"]; n_in = int((hd.place == "inside Normal").sum())
T += [f"bar (at least 45 of the 50 healthy donors inside Normal): {'met' if n_in >= 45 and len(hd) == 50 else 'not met'} ({n_in} of {len(hd)} read)"]
open(ot, "w").write("\n".join(T) + "\n"); print("\n".join(T))
