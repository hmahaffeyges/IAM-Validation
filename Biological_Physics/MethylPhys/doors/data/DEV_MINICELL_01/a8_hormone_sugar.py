import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'pandas', 'numpy', 'scipy'], check=True, timeout=900)
#!/usr/bin/env python3
"""A8O item 23b (prediction written in the register before this ran; exploration): does each cell type's own IAM-A level follow how strongly
it is acted on by insulin and sugar, or whether it lines the blood? HPA rna_single_cell_type.tsv (nCPM), crosswalk X from position_expr_01.py
(unchanged). Level: each type's joint-run A_derived centre (type bands, numbers only)."""
import os, io, zipfile, urllib.request, numpy as np, pandas as pd
from scipy.stats import rankdata
LEVEL = {"erythrocyte progenitors": 0.8043, "cortical neurons": 0.8954, "naive cd8 t cells": 0.8979, "naive cd4 t cells": 0.9323, "lung alveolar macrophages": 0.9464, "pancreatic acinar": 0.951, "smooth muscle": 0.9583, "lung interstitial macrophages": 0.9597, "heart fibroblasts": 0.9713, "breast luminal epithelium": 0.9937, "hepatocyte": 0.9986, "breast basal epithelium": 1.0012, "effector memory cd8 t cells": 1.0014, "colon macrophages": 1.0087, "endometrium epithelium": 1.0199, "monocytes": 1.0207, "b cells": 1.0256, "oligodendrocytes": 1.0266, "colon epithelium": 1.0302, "colon enteroendocrine": 1.0314, "fallopien tubes epithelium": 1.035, "nk cells": 1.0445, "t effector memory cd4": 1.0451, "cd8 t cells": 1.0491, "cardiomyocyte": 1.0541, "gastric fundus epithelium": 1.0551, "cd4 t cells": 1.0563, "small intestine epithelium": 1.0621, "aorta endothelium": 1.0622, "kidney tubular epithelium": 1.0633, "gastric body epithelium": 1.0643, "lung alveolar epithelium": 1.0672, "pancreatic duct": 1.0698, "prostate epithelium": 1.0708, "gastric antrum epithelium": 1.0766, "lung alveolar endothelium": 1.0806, "tonsil pharyngeal epithelium": 1.0835, "tonsil palatine epithelium": 1.0935, "memory b cells": 1.0957, "colon fibroblasts": 1.0982, "kidney glomerular epithelium": 1.1052, "t effector cell cd8": 1.1056, "bladder epithelium": 1.1074, "thyroid epithelium": 1.1164, "lung bronchus epithelium": 1.1187, "adipocytes": 1.1201, "vascular saphenous endothelium": 1.1272, "pancreatic beta": 1.1281, "tongue epithelium": 1.1327, "kidney glomerular endothelium": 1.1414, "kidney tubular endothelium": 1.142, "pancreatic alpha": 1.1459, "pancreas endothelium": 1.1498, "pancreatic delta": 1.1651, "striated muscle": 1.1709, "t central memory cd4": 1.1713}
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
GENES = ["SLC2A4", "INSR", "MKI67"]
raw = None
for url in ("https://www.proteinatlas.org/download/tsv/rna_single_cell_type.tsv.zip", "https://www.proteinatlas.org/download/rna_single_cell_type.tsv.zip"):
    try: raw = urllib.request.urlopen(url, timeout=600).read(); break
    except Exception as e: print("::notice::HS fetch failed", url, str(e)[:120])
z = zipfile.ZipFile(io.BytesIO(raw)); T = pd.read_csv(z.open(z.namelist()[0]), sep="\t"); T = T[T["Gene name"].isin(GENES)]
E = T.pivot_table(index="Cell type", columns="Gene name", values="nCPM", aggfunc="mean")
miss = sorted({h for v in X.values() for h in v} - set(E.index)); print("::notice::HS hpa types missing:", miss[:10])
rows = []
for key in sorted({tuple(v) for v in X.values()}):
    cells = [c for c, v in X.items() if tuple(v) == key and c in LEVEL]
    if not cells or any(h not in E.index for h in key): continue
    e = E.loc[list(key)].mean()
    rows.append(dict(hpa=" + ".join(key), cells="; ".join(cells), level=float(np.mean([LEVEL[c] for c in cells])), GLUT4=e.SLC2A4, INSR=e.INSR,
                     MKI67=e.MKI67, blood=int(all("endothelium" in c for c in cells))))
P = pd.DataFrame(rows); rng = np.random.default_rng(20261119)
def perm(a, b, n=10000):
    ra, rb = rankdata(a), rankdata(b); o = np.corrcoef(ra, rb)[0, 1]; nl = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(n)])
    return o, float((np.sum(nl >= o) + 1) / (n + 1))
def partial(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c); res = lambda y: y - np.polyval(np.polyfit(rc, y, 1), rc); return float(np.corrcoef(res(ra), res(rb))[0, 1])
out = [f"points {len(P)}"]
for g in ("GLUT4", "INSR"):
    o, p = perm(P[g], P.level); out.append(f"{g}: rho {o:+.3f} p {p:.4f} | MKI67 partialled {partial(P[g], P.level, P.MKI67):+.3f}")
out.append(f"lines the blood: endothelial point(s) {list(P[P.blood == 1].level.round(3))} vs others' median {P[P.blood == 0].level.median():.3f}")
for _, x in P.sort_values("level").iterrows(): out.append(f"{x.hpa[:40]:40s} level {x.level:.3f} GLUT4 {x.GLUT4:8.2f} INSR {x.INSR:8.2f} MKI67 {x.MKI67:7.2f}")
print("\n".join("::notice::HS " + l for l in out))
os.makedirs("xout", exist_ok=True); P.to_csv("xout/a8_hormone_sugar.csv", index=False)
