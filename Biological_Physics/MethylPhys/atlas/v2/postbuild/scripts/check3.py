#!/usr/bin/env python3
"""(1) Split check: rank all sweep-2 variants on GSE110554 only, score the top pick on GSE167998 (not used to choose).
(2) Chosen solver on all 24 with bootstrap presence: which non-blood cells take mass, per-array errors."""
import json, glob, numpy as np, pandas as pd, concurrent.futures as cf
exec(open("sweep2.py").read().split("def run(cfg):")[0])   # data, G, TR, V
def per_study(cfg, boot=0):
    D = DeconvV2(None, pooled_one_sample=POOL1, drop=cfg.get("drop", []), markers=cfg["markers"], margin=cfg["margin"], pair_margin=cfg["pair_margin"], top=cfg["top"], sigma=cfg["sigma"], per_pair=cfg["per_pair"], k_near=cfg["k_near"], A=ATL)
    rows = []
    for r in T.itertuples():
        o = D.deconvolve(BETA[r.gsm], n_boot=boot); fr = o["fractions"]; sc = 100.0 if r.gse == "GSE110554" else 1.0
        row = dict(gsm=r.gsm, gse=r.gse, non_blood=sum(v for c, v in fr.items() if c not in blood),
                   nonblood_cells={c: round(v, 4) for c, v in fr.items() if c not in blood and v > 0.002},
                   present_nonblood=[c for c, p in o["present"].items() if p and c not in blood] if boot else None)
        for g, cs in G.items():
            tv = getattr(r, TR[g]); row[f"{g}_est"] = sum(fr.get(c, 0) for c in cs); row[f"{g}_true"] = tv / sc if pd.notna(tv) else np.nan
        rows.append(row)
    X = pd.DataFrame(rows)
    m = {s: {g: float((X[X.gse == s][f"{g}_est"] - X[X.gse == s][f"{g}_true"]).abs().mean()) for g in ("CD4", "CD8", "NK", "B", "Mono", "Neu", "Eos")} for s in ("GSE110554", "GSE167998")}
    return cfg, m, X
with cf.ProcessPoolExecutor(16) as ex: R = list(ex.map(per_study, V))
key = lambda s: (lambda m: max(m[s]["CD4"], m[s]["CD8"], m[s]["NK"]))
R18 = sorted(R, key=lambda x: key("GSE110554")(x[1]))
pick18, m18, _ = R18[0]
print("SPLIT: chosen on GSE110554 ->", pick18, "\n  on 2018 (choosing):", {g: round(v, 4) for g, v in m18["GSE110554"].items()}, "\n  on 2022 (held out):", {g: round(v, 4) for g, v in m18["GSE167998"].items()}, flush=True)
R22 = sorted(R, key=lambda x: key("GSE167998")(x[1])); pick22, m22, _ = R22[0]
print("SPLIT reversed: chosen on GSE167998 ->", pick22, "\n  on 2018 (held out):", {g: round(v, 4) for g, v in m22["GSE110554"].items()}, flush=True)
chosen = dict(drop=[], markers="hybrid", margin=0.10, pair_margin=0.15, per_pair=60, k_near=8, top=600, sigma=0.02)
_, mc, X = per_study(chosen, boot=100)
print("CHOSEN by study:", {s: {g: round(v, 4) for g, v in d.items()} for s, d in mc.items()})
from collections import Counter
mass = Counter(); pres = Counter()
for d in X.nonblood_cells: mass.update(d)
for p in X.present_nonblood: pres.update(p)
print("non-blood mass summed over 24 arrays:", {c: round(v, 3) for c, v in mass.most_common(10)})
print("non-blood PRESENT (bootstrap) counts:", dict(pres.most_common(10)))
print("Eos est vs true (2022):", X[X.gse == "GSE167998"][["Eos_est", "Eos_true"]].round(3).values.tolist())
X.to_csv("chosen_mixtures.csv", index=False)
json.dump(dict(split_2018_pick=pick18, split_2018=m18, split_2022_pick=pick22, split_2022=m22, chosen=chosen, chosen_by_study=mc, nonblood_mass=dict(mass), nonblood_present=dict(pres)), open("check3.json", "w"), indent=1, default=str)
