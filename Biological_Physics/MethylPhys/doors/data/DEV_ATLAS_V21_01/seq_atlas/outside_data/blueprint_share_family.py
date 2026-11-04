"""BLUEPRINT's healthy libraries (spent; development, diagnostic, nothing scored): where the Deconvolver puts the share that is not the declared
cell. From each one-command bundle of the common-floor re-read (cell_shares, the IAM Sequencing Atlas): the declared cell's share, its family's
share (the declared cell and its closest relatives in the atlas, listed in FAMILY below, written after a first look at these shares, disclosed), the three largest shares
outside the family, and the largest share overall. Usage: python3 blueprint_share_family.py BUNDLE_DIR OUT.csv OUT.txt"""
import os, sys, json, collections, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); C = dict(pd.read_csv(os.path.join(H, "blueprint_reread_floor.csv"))[["gsm", "cell"]].values)
CD4 = ["cd4 t cells", "naive cd4 t cells", "t central memory cd4", "t effector memory cd4", "regulatory t cells"]
CD8 = ["cd8 t cells", "naive cd8 t cells", "effector memory cd8 t cells", "t effector cell cd8", "t central memory cd8"]
B = ["b cells", "memory b cells"]; MONO = ["monocytes", "lung interstitial macrophages", "lung alveolar macrophages", "colon macrophages"]
FAMILY = {**{c: CD4 for c in CD4}, **{c: CD8 for c in CD8}, **{c: B for c in B}, "monocytes": MONO, "nk cells": ["nk cells"], "erythrocyte progenitors": ["erythrocyte progenitors"]}
R = []; tops = collections.Counter()
for f in sorted(os.listdir(sys.argv[1])):
    g = f.replace("_bundle.json", ""); sh = {x["cell"]: x["share"] for x in (json.load(open(os.path.join(sys.argv[1], f))).get("cell_shares") or {}).get("cells") or []}
    c = C.get(g)
    if not sh or c is None: continue
    fam = FAMILY.get(c, [c]); top = max(sh, key=sh.get); tops[(c, top in fam)] += 1
    out = sorted(((k, v) for k, v in sh.items() if k not in fam), key=lambda t: -t[1])[:3]
    R.append(dict(gsm=g, cell=c, own=sh.get(c, 0.0), family=sum(sh.get(x, 0.0) for x in fam), largest=top, largest_in_family=top in fam, outside_1=out[0][0] if out else "", outside_1_share=out[0][1] if out else 0.0))
D = pd.DataFrame(R); D.to_csv(sys.argv[2], index=False)
L = [f"{len(D)} libraries; the largest share is the declared cell or its family in {int(D.largest_in_family.sum())} of {len(D)}",
     f"declared cell's share median {D.own.median():.3f}; its family's share median {D.family.median():.3f}"]
for c, G in D.groupby("cell"): L.append(f"{c}: {len(G)}; own {G.own.median():.3f}; family {G.family.median():.3f}; most share outside the family: {G.outside_1.mode().iloc[0]} ({G.outside_1_share.median():.3f})")
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
