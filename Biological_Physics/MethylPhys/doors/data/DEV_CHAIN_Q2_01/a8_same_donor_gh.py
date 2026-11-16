#!/usr/bin/env python3
"""Register A8 (exploration on spent counts; nothing sealed; 2026-11-15). The history lead, read within one laboratory so its meter offset
cancels: the atlas lab's (GSE186458) naive and memory/effector T cells from the same donors (identified by age and sex in GEO's records:
30 F, 29 F, 69 M; the records carry no donor id, so this pairing is an inference from the study's sample series). Each library's reading is
held_err_length - b * loose_share (b = 0.034625, IAM-A's derived slope, as celltype_vs_thermal.py), over the thermal level through the counter
at the library's OWN clustering (celltype_vs_thermal.py's forward grid), and again at the donor's naive clustering (clustering held fixed).
Writes a8_same_donor_summary.txt and a8_same_donor.csv. Runs on a GitHub machine."""
import os, json, glob, math
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); B = 0.034625
src = open(os.path.join(H, "celltype_vs_thermal.py")).read(); exec(src[:src.index("rows = []")])   # reuse the committed thermal level as is
D = {"GSM5652277": ["Blood-T-CD3-Z000000TV", "T (CD3+) cells", "56", "F", ""], "GSM5652278": ["Blood-T-CD3-Z000000UP", "T (CD3+) cells", "55", "F", ""], "GSM5652279": ["Blood-T-CD4-Z000000TT", "T helper(CD4+) cells", "56", "F", ""], "GSM5652280": ["Blood-T-CD4-Z000000U7", "T helper(CD4+) cells", "50", "M", ""], "GSM5652281": ["Blood-T-CD4-Z000000UM", "T helper(CD4+) cells", "55", "F", ""], "GSM5652282": ["Blood-T-CD8-Z000000TR", "T cytotoxic (CD8+) cells", "56", "F", ""], "GSM5652283": ["Blood-T-CD8-Z000000U5", "T cytotoxic (CD8+) cells", "50", "M", ""], "GSM5652284": ["Blood-T-CD8-Z000000UK", "T cytotoxic (CD8+) cells", "55", "F", ""], "GSM5652285": ["Blood-T-CenMem-CD4-Z00000417", "T central memory CD4", "30", "F", ""], "GSM5652286": ["Blood-T-CenMem-CD4-Z0000041D", "T central memory CD4", "29", "F", ""], "GSM5652287": ["Blood-T-CenMem-CD4-Z0000041N", "T central memory CD4", "69", "M", ""], "GSM5652288": ["Blood-T-Eff-CD8-Z00000419", "T effector cell CD8", "30", "F", ""], "GSM5652289": ["Blood-T-Eff-CD8-Z0000041F", "T effector cell CD8", "29", "F", ""], "GSM5652290": ["Blood-T-Eff-CD8-Z0000041Q", "T effector cell CD8", "69", "M", ""], "GSM5652291": ["Blood-T-EffMem-CD4-Z00000416", "T effector memory CD4", "30", "F", ""], "GSM5652292": ["Blood-T-EffMem-CD4-Z0000041C", "T effector memory CD4", "29", "F", ""], "GSM5652293": ["Blood-T-EffMem-CD4-Z0000041M", "T effector memory CD4", "69", "M", ""], "GSM5652294": ["Blood-T-EffMem-CD8-Z0000041A", "T effector memory CD8", "30", "F", ""], "GSM5652295": ["Blood-T-EffMem-CD8-Z0000041G", "T effector memory CD8", "29", "F", ""], "GSM5652296": ["Blood-T-Naive-CD4-Z0000041E", "Naive T cells CD4", "29", "F", ""], "GSM5652297": ["Blood-T-Naive-CD8-Z0000041B", "Naive T cells CD8", "30", "F", ""], "GSM5652298": ["Blood-T-Naive-CD8-Z0000041H", "Naive T cells CD8", "29", "F", ""], "GSM5652299": ["Blood-NK-Z000000TM", "NK", "56", "F", ""], "GSM5652300": ["Blood-NK-Z000000U1", "NK", "50", "M", ""], "GSM5652301": ["Blood-NK-Z000000UF", "NK", "55", "F", ""], "GSM5652302": ["Blood-Monocytes-Z000000TP", "Monocytes", "56", "F", ""], "GSM5652303": ["Blood-Monocytes-Z000000U3", "Monocytes", "50", "M", ""], "GSM5652304": ["Blood-Monocytes-Z000000UH", "Monocytes", "55", "F", ""], "GSM5652313": ["Blood-Granulocytes-Z000000TZ", "Granulocytes", "56", "F", ""], "GSM5652314": ["Blood-Granulocytes-Z000000UD", "Granulocytes", "50", "M", ""], "GSM5652315": ["Blood-Granulocytes-Z000000UT", "Granulocytes", "55", "F", ""], "GSM5652316": ["Blood-B-Z000000TX", "B cells", "56", "F", ""], "GSM5652317": ["Blood-B-Z000000UB", "B cells", "50", "M", ""], "GSM5652318": ["Blood-B-Z000000UR", "B cells", "55", "F", ""], "GSM5652319": ["Blood-B-Mem-Z0000041J", "Memory B cells", "32", "F", ""], "GSM5652320": ["Blood-B-Mem-Z0000041K", "Memory B cells", "35", "F", ""]}
R = []
for f in glob.glob(os.path.join(H, "percall_out", "*.json")):
    c = json.load(open(f))
    for x in (c if isinstance(c, list) else [c]):
        g = x.get("sample") if isinstance(x, dict) else None
        if g in D and x.get("held_err_length") is not None:
            t, ct, age, sex, _ = D[g]
            kind = "naive" if "Naive" in t else ("memory" if any(k in t for k in ("Mem", "Eff")) else "bulk")
            line = "CD8" if "CD8" in t else ("CD4" if "CD4" in t else ("B" if "-B-" in t or t.endswith("-B") else t.split("-")[1]))
            R.append(dict(gsm=g, title=t, donor=f"{age}{sex}", lineage=line, kind=kind, reading=x["held_err_length"] - B * x["loose_share"], coupling=x["coupling"]))
X = pd.DataFrame(R); X["over_thermal_own"] = [r / thermal(c) for r, c in zip(X.reading, X.coupling)]
out = []
for d, g in X[X.kind != "bulk"].groupby("donor"):
    nv = g[g.kind == "naive"]
    if nv.empty: continue
    cfix = float(nv.coupling.mean())
    for r in g.itertuples(): out.append(dict(donor=d, lineage=r.lineage, kind=r.kind, title=r.title, coupling=round(r.coupling, 3), over_thermal_own=round(r.over_thermal_own, 3), over_thermal_naive_coupling=round(r.reading / thermal(cfix), 3)))
O = pd.DataFrame(out).sort_values(["donor", "lineage", "kind"]); O.to_csv(os.path.join(H, "a8_same_donor.csv"), index=False)
s = O.groupby(["donor", "kind"])[["over_thermal_own", "over_thermal_naive_coupling", "coupling"]].mean().round(3)
bulk = X[X.kind == "bulk"].groupby("lineage").over_thermal_own.mean().round(3).to_dict()
t = ("A8 same-donor reading (atlas lab, exploration):\n" + O.to_string(index=False) + "\n\nmeans by donor and kind:\n" + s.to_string() +
     f"\n\nbulk blood types (donors aged 50-56), over thermal at own clustering: {bulk}\n")
open(os.path.join(H, "a8_same_donor_summary.txt"), "w").write(t); print(t)
for l in s.to_string().splitlines(): print("::notice::" + l[:300])
