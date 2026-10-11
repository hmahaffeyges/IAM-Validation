"""DEV-TISSUE-BREAST-01 synthetic test 2 (template mismatch): as breast_tissue_01.py, but the TRUE tissue is built from cells the panel
describes imperfectly or lacks: fibroblasts = heart fibroblasts only (panel: the colon/heart mean), plus a hidden smooth-muscle component
(myoepithelium stand-in, absent from the panel) at 0-10 %. The reader is unchanged. Derived from:
DEV-TISSUE-BREAST-01 synthetic test 1 (2026-10-11; atlas posterior draws only, no tissue array read). Can Met-A tell a breast tumour from its
patient's own normal tissue when the two differ in COMPOSITION? Tissue = mixture of atlas v2 cells (posterior draws): breast luminal and basal
epithelium, adipocytes, fibroblasts (colon and heart fibroblasts averaged: the atlas holds no breast fibroblast), vascular endothelium, colon
macrophages, CD4/CD8 T cells, B cells. Each person = one posterior draw per cell + N(0, 0.02) per site; each array adds N(0, 0.04).
Normal tissue: adipocytes 30-70 %, epithelium 10-30 % (luminal:basal 2:1), fibroblasts 15-35 %, endothelium 3-8 %, immune 2-8 % (Dirichlet-like
draws, renormalised). Tumour tissue: epithelium 40-80 % (luminal), adipocytes 5-20 %, fibroblasts 10-30 %, immune 2-15 %, endothelium 2-6 %.
Identity sites: canon rule on breast luminal epithelium (posterior mean 0.75-0.95 or 0.05-0.25, SD <= 0.05; smallest SD; 450K probes only).
Readings: PLAIN Met-A = mean H(beta) at identity sites / median over healthy references; COMP Met-A = the same ratio with each sample's own
expected H (beta predicted from its NNLS composition on 8,000 informative non-identity loci and the panel means), tared by the healthy references.
Planted cancer change: in the tumour's luminal component, a share d of molecules gains the mark at unmethylated identity sites and loses it at
methylated ones (both directions). 91 patients (TCGA-BRCA matched pairs), 97 healthy references (the TCGA normals' count).
Usage: python3 breast_tissue_01.py ATLAS_BLOCK_DIR MANIFEST_450K.csv.gz"""
import sys, glob, gzip, numpy as np, pandas as pd
from scipy.optimize import nnls
rg = np.random.default_rng(20261011)
P = sorted(glob.glob(sys.argv[1] + "/b*.npz")); Z = [np.load(p, allow_pickle=True) for p in P]
cells = Z[0]["cells"].tolist(); cpg = np.concatenate([z["cpg"] for z in Z]); D = np.concatenate([z["draws"].astype(np.float32) for z in Z], axis=2)
_, u = np.unique(cpg, return_index=True); cpg, D = cpg[u], D[:, :, u]
m450 = set(pd.read_csv(sys.argv[2], usecols=["IlmnID"], low_memory=False).IlmnID.astype(str)); k = np.isin(cpg, list(m450)) & np.isfinite(D).all((0, 1))
cpg, D = cpg[k], D[:, :, k]
ix = lambda n: cells.index(n)
def comp(name):   # draws x loci for one panel component
    if name == "fibroblast": return (D[:, ix("colon fibroblasts")] + D[:, ix("heart fibroblasts")]) / 2
    if name == "T": return (D[:, ix("cd4 t cells")] + D[:, ix("cd8 t cells")]) / 2
    return D[:, ix(name)]
PANEL = ["breast luminal epithelium", "breast basal epithelium", "adipocytes", "fibroblast", "vascular endothelium", "colon macrophages", "T", "b cells"]
C = np.stack([comp(n) for n in PANEL], 1)            # draws x panel x loci
Mu = C.mean(0); Sd = C.std(0)
TRUE = [comp(n) for n in PANEL]; TRUE[3] = D[:, ix("heart fibroblasts")]; TRUE.append(D[:, ix("smooth muscle")]); CT = np.stack(TRUE, 1)
lum = 0; mL, sL = Mu[lum], Sd[lum]
hi = np.flatnonzero((mL >= 0.75) & (mL <= 0.95) & (sL <= 0.05)); lo = np.flatnonzero((mL >= 0.05) & (mL <= 0.25) & (sL <= 0.05))
hi = hi[np.argsort(sL[hi])][:3000]; lo = lo[np.argsort(sL[lo])][:3000]; S = np.r_[hi, lo]
rest = np.setdiff1d(np.arange(len(cpg)), S); inf = rest[np.argsort(-Mu[:, rest].std(0))][:8000]
H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
def frac(kind):
    if kind == "normal":
        e = rg.uniform(.10, .30); f = np.array([e * 2 / 3, e / 3, rg.uniform(.3, .7), rg.uniform(.15, .35), rg.uniform(.03, .08), *(rg.uniform(.02, .08) * np.array([.4, .4, .2]))])
    else:
        e = rg.uniform(.40, .80); f = np.array([e, 0.0, rg.uniform(.05, .20), rg.uniform(.10, .30), rg.uniform(.02, .06), *(rg.uniform(.02, .15) * np.array([.4, .4, .2]))])
    f = np.r_[f, rg.uniform(0, .10)]; return f / f.sum()   # hidden smooth-muscle share
def person(): return np.stack([CT[rg.integers(CT.shape[0]), j] for j in range(CT.shape[1])]) + rg.normal(0, .02, (CT.shape[1], len(cpg)))
def array(P_, f, d=0.0):
    X = P_.copy()
    if d:
        X[0, lo] = X[0, lo] + d * (1 - X[0, lo]); X[0, hi] = X[0, hi] * (1 - d)
    return np.clip(f @ X + rg.normal(0, .04, len(cpg)), 0, 1)
def read(b):
    w, _ = nnls(Mu[:, inf].T, b[inf]); w = w / w.sum(); bp = w @ Mu
    return H(b[S]).mean(), H(b[S]).mean() / H(bp[S]).mean()
ref = np.array([read(array(person(), frac("normal"))) for _ in range(97)]); m = np.median(ref, 0)
refrel = ref / m; up = np.quantile(refrel, .95, 0)
print(f"loci {len(cpg)} | identity sites {len(hi)} methylated + {len(lo)} unmethylated | healthy refs: plain SD {refrel[:,0].std():.4f}, comp SD {refrel[:,1].std():.4f}")
rows = []
for case, d in (("composition only", 0.0), ("planted 0.02", .02), ("planted 0.05", .05), ("planted 0.10", .10)):
    dif = []; above = []
    for _ in range(91):
        P_ = person(); n = np.array(read(array(P_, frac("normal")))) / m; t = np.array(read(array(P_, frac("tumour"), d))) / m
        dif.append(t - n); above.append(t > up)
    dif = np.array(dif); above = np.array(above)
    for j, nm in enumerate(("PLAIN", "COMP")):
        se = dif[:, j].std(ddof=1) / np.sqrt(91)
        rows.append(dict(case=case, reading=nm, mean_tumour_minus_normal=round(dif[:, j].mean(), 4), sd_pair=round(dif[:, j].std(ddof=1), 4), z=round(dif[:, j].mean() / se, 1), tumours_above_healthy_95pct=round(above[:, j].mean(), 3)))
T = pd.DataFrame(rows); print(T.to_string(index=False))
