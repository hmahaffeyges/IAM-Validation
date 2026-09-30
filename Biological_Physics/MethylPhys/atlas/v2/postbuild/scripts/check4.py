#!/usr/bin/env python3
import json, numpy as np, pandas as pd
exec(open("check3.py").read().split("with cf.ProcessPoolExecutor(16)")[0])
base = dict(markers="hybrid", margin=0.10, pair_margin=0.15, per_pair=60, k_near=8, top=600, sigma=0.02)
for drop in ([], ["vascular endothelium"]):
    cfg = dict(base, drop=drop); _, m, X = per_study(cfg, boot=100)
    from collections import Counter; mass = Counter(); pres = Counter()
    for d in X.nonblood_cells: mass.update(d)
    for p in X.present_nonblood: pres.update(p)
    allm = {g: round(float((X[f"{g}_est"] - X[f"{g}_true"]).abs().mean()), 4) for g in ("CD4","CD8","NK","B","Mono","Neu","Eos")}
    print("DROP", drop, "| MAE all 24", allm, "| by study", {s: {g: round(v,4) for g, v in d.items() if g in ("CD4","CD8","NK")} for s, d in m.items()},
          "| nonblood median %.4f" % X.non_blood.median(), "| nonblood mass", {c: round(v,3) for c, v in mass.most_common(5)}, "| present", dict(pres.most_common(5)), flush=True)
    X.to_csv(f"check4_{'drop' if drop else 'keep'}.csv", index=False)
