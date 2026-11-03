"""BLUEPRINT's sorted libraries: where the Deconvolver puts the share that is not the library's own cell (diagnostic, added after the
outside-laboratory test and its re-read were scored; nothing scored). From each one-command bundle's cell shares, by the library's sorted cell:
the median own share and the four largest other cells. Usage: python3 blueprint_share_destinations.py BUNDLE_DIR OUT.txt"""
import os, sys, json, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); T = pd.read_csv(os.path.join(H, "blueprint_healthy_test_samples.csv")).set_index("gsm"); S = T.atlas_cell; BL = T.blueprint_cell; rows = []
for f in os.listdir(sys.argv[1]):
    if not f.endswith("_bundle.json"): continue
    g = f[:-12]; B = json.load(open(os.path.join(sys.argv[1], f)))
    for x in ((B.get("cell_shares") or {}).get("cells") or []): rows.append(dict(gsm=g, cell=S.get(g), to=x["cell"], share=x["share"]))
D = pd.DataFrame(rows); L = []
for c, g in D.groupby("cell"):
    n = g.gsm.nunique(); own = g[g.to == c].groupby("gsm").share.sum().reindex(g.gsm.unique()).fillna(0).median()
    o = g[g.to != c].groupby("to").share.sum().div(n).sort_values(ascending=False).head(4)
    lab = "; ".join(sorted(set(BL[g.gsm.unique()].astype(str))))
    L.append(f"{c} ({n}; BLUEPRINT label: {lab}): own {own:.3f}; then " + ", ".join(f"{k} {v:.3f}" for k, v in o.items()))
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
