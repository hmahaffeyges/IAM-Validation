"""DEV-POSITION-EXPR-01: does each healthy cell type's copy error follow its eraser-to-writer balance, read from public RNA expression?
Written and committed 2026-10-14 BEFORE any expression value is read (only the cell-type NAMES of the two tables were listed, to write the
crosswalk). Development: the 56 measured copy errors (PROC-CHANNEL-01) were read before; the expression tables were not.

IDEA (author's two halves, PHY-040; the middle-mark candidate in DEV_ERASE_01.md). The healthy copy error is the writer's floor plus the eraser's
equal share: eps = (1 + r) / (D + 1 + r), with D = 63.7 the writer's measured median discrimination (DEV_WRITER_01_OUTCOME.md) and r the eraser's loss relative to the
writer's. Healthy on average is r = 1. Each cell type's r is read from expression: r_cell = X_cell / median over cells of X, with
X = (TET1 + TET2 + TET3) / (DNMT1 + UHRF1) (eraser over maintenance writer, nCPM + 1). The median normalization is the theory's r = 1 on average;
no parameter is fitted to any copy error.

DATA. Measured: doors/data/PROC_CHANNEL_01/channel_cells_genomewide.csv, column copy_err (Loyfer 2023, 56 cell types). Expression: Human
Protein Atlas rna_single_cell_type.tsv (nCPM, one method), PRIMARY; rna_immune_cell.tsv (sorted blood cells, nTPM), SECONDARY. The crosswalk
below maps a Loyfer cell to HPA types; several HPA types are averaged; Loyfer cells mapping to the same HPA set are averaged into ONE point.
Ambiguous cells (colon enteroendocrine, gastric antrum/body/fundus, thyroid, tongue, tonsil) are left out before any value is read.

BAR (primary, run once): Spearman rho between predicted and measured eps over the matched points; one-sided permutation p (100,000, seed
20261014). MET if p < 0.01; else NOT MET. REPORTED, not bars: Pearson r; predicted and measured median and spread; the writer alone
(1 / (DNMT1 + UHRF1)) and the eraser alone (TET sum); a proliferation marker (MKI67) and the prediction's rho with MKI67 partialled out,
because a reviewer will ask whether dividing cells drive both (the held balance itself is independent of division, rule 25; this is a
confound check on the expression, not a term). SECONDARY (immune, reported): naive against memory for CD4 T, CD8 T and B cells, does the
predicted difference have the measured sign in all three?"""
import os, sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); D = 63.7; SEED = 20261014
X = {
 "adipocytes": ["adipocytes"], "b cells": ["b-cells"], "memory b cells": ["b-cells"], "bladder epithelium": ["urothelial cells"],
 "breast basal epithelium": ["breast myoepithelial cells"], "breast luminal epithelium": ["breast hormone-responsive cells", "breast secretory cells"],
 "cardiomyocyte": ["cardiomyocytes"], "colon epithelium": ["colonocytes"], "colon fibroblasts": ["fibroblasts"], "heart fibroblasts": ["fibroblasts"],
 "colon macrophages": ["macrophages"], "lung interstitial macrophages": ["macrophages"], "lung alveolar macrophages": ["macrophages"],
 "cortical neurons": ["brain excitatory neurons", "brain inhibitory neurons"], "endometrium epithelium": ["endometrial glandular cells", "endometrial luminal cells"],
 "erythrocyte progenitors": ["erythrocyte progenitors"], "fallopien tubes epithelium": ["fallopian secretory cells", "fallopian tube ciliated cells"],
 "hepatocyte": ["hepatocytes"], "kidney glomerular epithelium": ["podocytes"], "kidney tubular epithelium": ["proximal tubule cells"],
 "lung alveolar epithelium": ["alveolar cells type 1", "alveolar cells type 2"],
 "lung bronchus epithelium": ["respiratory basal cells", "respiratory ciliated cells", "respiratory secretory cells"], "monocytes": ["monocytes"],
 "naive cd4 t cells": ["t-cells"], "naive cd8 t cells": ["t-cells"], "cd4 t cells": ["t-cells"], "cd8 t cells": ["t-cells"], "t central memory cd4": ["t-cells"],
 "t effector memory cd4": ["t-cells"], "t effector cell cd8": ["t-cells"], "effector memory cd8 t cells": ["t-cells"], "nk cells": ["nk-cells"],
 "oligodendrocytes": ["oligodendrocytes"], "pancreatic acinar": ["pancreatic acinar cells"], "pancreatic alpha": ["pancreatic islet cells"],
 "pancreatic beta": ["pancreatic islet cells"], "pancreatic delta": ["pancreatic islet cells"], "pancreatic duct": ["pancreatic duct cells"],
 "prostate epithelium": ["prostatic glandular cells"], "small intestine epithelium": ["enterocytes"], "smooth muscle": ["smooth muscle cells"],
 "striated muscle": ["myonuclei"], "aorta endothelium": ["vascular endothelial cells"], "vascular saphenous endothelium": ["vascular endothelial cells"],
 "lung alveolar endothelium": ["vascular endothelial cells"], "kidney tubular endothelium": ["vascular endothelial cells"],
 "kidney glomerular endothelium": ["vascular endothelial cells"], "pancreas endothelium": ["vascular endothelial cells"]}
XI = {"naive cd4 t cells": ["naive CD4 T-cell"], "t central memory cd4": ["memory CD4 T-cell"], "t effector memory cd4": ["memory CD4 T-cell"],
      "naive cd8 t cells": ["naive CD8 T-cell"], "t effector cell cd8": ["memory CD8 T-cell"], "effector memory cd8 t cells": ["memory CD8 T-cell"],
      "b cells": ["naive B-cell"], "memory b cells": ["memory B-cell"]}
GENES = ["TET1", "TET2", "TET3", "DNMT1", "UHRF1", "MKI67"]
pred = lambda r: (1 + r) / (D + 1 + r)
def table(path, col, val):
    T = pd.read_csv(path, sep="\t"); T = T[T["Gene name"].isin(GENES)]
    return T.pivot_table(index=col, columns="Gene name", values=val, aggfunc="mean") + 1
def points(E, cw, M):
    rows = []
    for key, g in pd.Series({k: tuple(v) for k, v in cw.items()}).groupby(lambda k: tuple(cw[k])):
        e = E.loc[list(key)].mean(); rows.append(dict(hpa=" + ".join(key), loyfer="; ".join(g.index), eps=M.loc[list(g.index)].mean(),
            Xr=(e.TET1 + e.TET2 + e.TET3) / (e.DNMT1 + e.UHRF1), W=e.DNMT1 + e.UHRF1, TET=e.TET1 + e.TET2 + e.TET3, MKI67=e.MKI67))
    P = pd.DataFrame(rows); P["r"] = P.Xr / P.Xr.median(); P["pred"] = pred(P.r); return P
def perm_p(a, b, n=100000):
    rng = np.random.default_rng(SEED); ra, rb = rankdata(a), rankdata(b); obs = np.corrcoef(ra, rb)[0, 1]
    null = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(n)]); return obs, float((np.sum(null >= obs) + 1) / (n + 1))
def partial(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    res = lambda y: y - np.polyval(np.polyfit(rc, y, 1), rc); return float(np.corrcoef(res(ra), res(rb))[0, 1])
if __name__ == "__main__":
    M = pd.read_csv(os.path.join(HERE, "..", "PROC_CHANNEL_01", "channel_cells_genomewide.csv")).set_index("cell").copy_err
    hp = sys.argv[1] if len(sys.argv) > 1 else "hpa"
    E = table(os.path.join(hp, "rna_single_cell_type.tsv"), "Cell type", "nCPM"); P = points(E, X, M)
    rho, p = perm_p(P.pred, P.eps); verdict = "MET" if p < 0.01 else "NOT MET"
    L = ["DEV-POSITION-EXPR-01, run once (position_expr_01.py)", f"primary points {len(P)} (HPA single-cell types; Loyfer cells averaged per point)",
         f"PRIMARY: Spearman rho(predicted, measured) = {rho:+.3f}, one-sided permutation p = {p:.5f} -> {verdict}",
         f"Pearson r = {pearsonr(P.pred, P.eps)[0]:+.3f}; measured median {P.eps.median():.4f} (range {P.eps.min():.4f}-{P.eps.max():.4f}); "
         f"predicted median {P.pred.median():.4f} (range {P.pred.min():.4f}-{P.pred.max():.4f}); r range {P.r.min():.2f}-{P.r.max():.2f}",
         f"reported: writer alone (1/W) rho {spearmanr(1/P.W, P.eps)[0]:+.3f}; eraser alone (TET) rho {spearmanr(P.TET, P.eps)[0]:+.3f}; "
         f"MKI67 rho {spearmanr(P.MKI67, P.eps)[0]:+.3f}; prediction with MKI67 partialled out {partial(P.pred, P.eps, P.MKI67):+.3f}"]
    P.sort_values("eps").to_csv(os.path.join(HERE, "position_expr_01_points.csv"), index=False)
    EI = table(os.path.join(hp, "rna_immune_cell.tsv"), "Immune cell", "nTPM")
    pr = lambda names: (lambda e: (e.TET1 + e.TET2 + e.TET3) / (e.DNMT1 + e.UHRF1))(EI.loc[names].mean())
    for lab, nv, mm in (("CD4 T", ["naive cd4 t cells"], ["t central memory cd4", "t effector memory cd4"]),
                        ("CD8 T", ["naive cd8 t cells"], ["t effector cell cd8", "effector memory cd8 t cells"]), ("B", ["b cells"], ["memory b cells"])):
        dx = pr([XI[c][0] for c in mm]) - pr([XI[c][0] for c in nv]); de = M.loc[mm].mean() - M.loc[nv].mean()
        L.append(f"secondary {lab}: memory minus naive, predicted eraser/writer change {dx:+.3f}, measured eps change {de:+.4f} -> {'same sign' if np.sign(dx) == np.sign(de) else 'OPPOSITE'}")
    open(os.path.join(HERE, "position_expr_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
