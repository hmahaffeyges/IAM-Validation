"""Held error against how often a cell type is replaced (PHY-079; exploration on spent counts; nothing sealed; the author's question:
cells that shed often against cells that rarely copy). Each Loyfer cell type's median observed held error (held_err_length) over the
M-based prediction (scatter_split.csv's samples), paired with the mean human lifespan of the matching cell type in
census_cell_lifespans.csv (Sender & Milo 2021, Nature Medicine 27:45-48, Supplementary Table 5; doi 10.1038/s41591-020-01182-9).
A pairing is 'same' (the same cell type) or 'parent' (the census gives only the parent type, e.g. all endothelial cells); types the
census has no lifespan for are left out. Rank correlation (Spearman). Writes lifespan_vs_held.csv and lifespan_vs_held_summary.txt."""
import os, json, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__)); Q = os.path.join(H, "..", "DEV_CHAIN_Q2_01")
M = float(json.load(open(os.path.join(H, "..", "..", "..", "..", "..", "CANON", "iam_canon.json")))["constants"]["M_cell"]["value"])
rows = []
for f in sorted(glob.glob(os.path.join(Q, "percall_out", "*.json"))):
    c = json.load(open(f))
    for x in (c if isinstance(c, list) else [c]):
        if isinstance(x, dict) and x.get("held_err_length") is not None: rows.append(dict(gsm=x["sample"], cell=x["cell"], obs=x["held_err_length"], loose=x["loose_share"]))
P = pd.DataFrame(rows).drop_duplicates("gsm"); med = (P.obs / (1 / (1 + M) + 0.0284 * P.loose)).groupby(P.cell).median()
life = pd.read_csv(os.path.join(H, "census_cell_lifespans.csv")).set_index("cell").mean_lifespan_human_days
MAP = {"small intestine epithelium": ("Small intestine epithelial cells", "same"), "colon epithelium": ("Colon epithelial cells", "same"),
 "gastric fundus epithelium": ("Stomach epithelial cells", "parent"), "gastric body epithelium": ("Stomach epithelial cells", "parent"),
 "gastric antrum epithelium": ("Stomach epithelial cells", "parent"), "monocytes": ("Monocytes", "same"),
 "lung alveolar macrophages": ("Alveolar macrophages", "same"), "lung bronchus epithelium": ("Bronchial epithelial cells", "same"),
 "adipocytes": ("Adipocytes", "same"), "cardiomyocyte": ("Cardiomyocytes", "same"), "striated muscle": ("Myocytes", "same"),
 "oligodendrocytes": ("Glial cells", "parent"), "cd4 t cells": ("mature T cells", "parent"), "cd8 t cells": ("mature T cells", "parent"),
 "b cells": ("mature B cells", "parent"), "pancreatic beta": ("Islet cells", "parent"), "pancreatic alpha": ("Islet cells", "parent"),
 "pancreatic delta": ("Islet cells", "parent"), "thyroid epithelium": ("Follicular cells", "same"), "heart fibroblasts": ("Connective tissue cells", "parent"),
 "kidney glomerular epithelium": ("Glomerulus total cells", "parent"),
 **{e: ("Endothelial cells", "parent") for e in ("aorta endothelium", "lung alveolar endothelium", "vascular saphenous endothelium", "kidney glomerular endothelium", "kidney tubular endothelium", "pancreas endothelium")}}
J = pd.DataFrame([dict(cell=k, census=v[0], match=v[1], lifespan_days=float(life[v[0]]), held_over_prediction=round(float(med[k]), 3)) for k, v in MAP.items() if k in med.index])
J = J.sort_values("lifespan_days"); J.to_csv(os.path.join(H, "lifespan_vs_held.csv"), index=False)
lg = np.log10(J.lifespan_days); a = spearmanr(lg, J.held_over_prediction); s = J.match == "same"; b = spearmanr(lg[s], J.held_over_prediction[s])
U = J.groupby("census").agg(l=("lifespan_days", "first"), r=("held_over_prediction", "median")); u = spearmanr(np.log10(U.l), U.r)
t = (f"Held error against lifespan (exploration). Spearman: all {len(J)} pairings rho {a.statistic:.2f} (p {a.pvalue:.3f}); one per census type "
     f"n={len(U)} rho {u.statistic:.2f} (p {u.pvalue:.3f}); same-type pairings only n={int(s.sum())} rho {b.statistic:.2f} (p {b.pvalue:.3f}).\n"
     "Positive: cell types replaced rarely hold slightly worse than those replaced often.\n\n" + J.to_string(index=False) + "\n")
open(os.path.join(H, "lifespan_vs_held_summary.txt"), "w").write(t); print(t[:400])
