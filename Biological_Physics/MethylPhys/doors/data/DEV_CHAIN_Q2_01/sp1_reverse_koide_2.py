import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', '-r', 'Biological_Physics/MethylPhys/chain/requirements.txt', 'scipy'], check=True, timeout=900)
#!/usr/bin/env python3
"""SP1 reverse Koide follow-up (register '#### SP1 REVERSE KOIDE RESULT', exploration): constitutive sites only. k = 56: held (>= 0.8) in every
type; k = 0: unmarked (<= 0.2) in every type. Each type's held error 1 - mean and unmarked mean there, against the law (A4 0.04454; stray as read
0.02125), each with the atlas's own standard error; plus the mode of each type's level at b >= 0.5 (the held peak)."""
import sys, os, json, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "Biological_Physics/MethylPhys/chain"); import stage_seq_atlas as SQ
from scipy.stats import spearmanr
D = SQ.SPHERE_DIR; BL = SQ.BLANK
def load(stem):
    out = {}
    for k in (1, 2, 3):
        t = pq.read_table(os.path.join(D, f"{stem}_part{k}.parquet"))
        for c in t.column_names: u = t.column(c).to_numpy(); v = u.astype("float32") / 1e4; v[u == BL] = np.nan; out[c] = v
    return out
MU = load("mu"); SE = load("mu_se"); LEV = {'erythrocyte progenitors': 0.8043, 'cortical neurons': 0.8954, 'naive cd8 t cells': 0.8979, 'naive cd4 t cells': 0.9323, 'lung alveolar macrophages': 0.9464, 'pancreatic acinar': 0.951, 'smooth muscle': 0.9583, 'lung interstitial macrophages': 0.9597, 'heart fibroblasts': 0.9713, 'breast luminal epithelium': 0.9937, 'hepatocyte': 0.9986, 'breast basal epithelium': 1.0012, 'effector memory cd8 t cells': 1.0014, 'colon macrophages': 1.0087, 'endometrium epithelium': 1.0199, 'monocytes': 1.0207, 'b cells': 1.0256, 'oligodendrocytes': 1.0266, 'colon epithelium': 1.0302, 'colon enteroendocrine': 1.0314, 'fallopien tubes epithelium': 1.035, 'nk cells': 1.0445, 't effector memory cd4': 1.0451, 'cd8 t cells': 1.0491, 'cardiomyocyte': 1.0541, 'gastric fundus epithelium': 1.0551, 'cd4 t cells': 1.0563, 'small intestine epithelium': 1.0621, 'aorta endothelium': 1.0622, 'kidney tubular epithelium': 1.0633, 'gastric body epithelium': 1.0643, 'lung alveolar epithelium': 1.0672, 'pancreatic duct': 1.0698, 'prostate epithelium': 1.0708, 'gastric antrum epithelium': 1.0766, 'lung alveolar endothelium': 1.0806, 'tonsil pharyngeal epithelium': 1.0835, 'tonsil palatine epithelium': 1.0935, 'memory b cells': 1.0957, 'colon fibroblasts': 1.0982, 'kidney glomerular epithelium': 1.1052, 't effector cell cd8': 1.1056, 'bladder epithelium': 1.1074, 'thyroid epithelium': 1.1164, 'lung bronchus epithelium': 1.1187, 'adipocytes': 1.1201, 'vascular saphenous endothelium': 1.1272, 'pancreatic beta': 1.1281, 'tongue epithelium': 1.1327, 'kidney glomerular endothelium': 1.1414, 'kidney tubular endothelium': 1.142, 'pancreatic alpha': 1.1459, 'pancreas endothelium': 1.1498, 'pancreatic delta': 1.1651, 'striated muscle': 1.1709, 't central memory cd4': 1.1713}; cells = sorted(c for c in MU if c in LEV); S = len(cells)
M = np.vstack([MU[c] for c in cells]).T; E = np.vstack([SE[c] for c in cells]).T; ok = np.isfinite(M).all(1); M, E = M[ok], E[ok]
EPS0 = 0.04454; s = 1 / 61 + 0.0044 + 0.00046
for thr in (0.8, 0.9):
    K = (M >= thr).all(1); err = 1 - M[K].mean(0); se = np.sqrt(np.nanmean(E[K] ** 2, 0) / K.sum()); lv = np.array([LEV[c] for c in cells])
    print(f"::notice::REVK2 held in every type (>= {thr}): {int(K.sum())} sites; held error median {np.median(err):.5f} (law {EPS0}), range {err.min():.5f}-{err.max():.5f}; within 0.75-1.33 x {np.mean((err > 0.75 * EPS0) & (err < 1.333 * EPS0)):.2f}; vs A8 level rho {spearmanr(err, lv)[0]:+.3f}", flush=True)
for thr in (0.2, 0.1):
    K = (M <= thr).all(1); um = M[K].mean(0)
    print(f"::notice::REVK2 unmarked in every type (<= {thr}): {int(K.sum())} sites; level median {np.median(um):.5f} (law as read {s:.5f}), range {um.min():.5f}-{um.max():.5f}", flush=True)
modes = {}
for j, c in enumerate(cells):
    x = M[:, j]; x = x[x >= 0.5]; h, e = np.histogram(x, bins=np.linspace(0.5, 1.0, 101)); modes[c] = float(1 - (e[h.argmax()] + e[h.argmax() + 1]) / 2)
mv = np.array(list(modes.values())); lv = np.array([LEV[c] for c in modes])
print(f"::notice::REVK2 held peak (mode of b >= 0.5) as error: median {np.median(mv):.5f} (law {EPS0}), range {mv.min():.5f}-{mv.max():.5f}; vs A8 level rho {spearmanr(mv, lv)[0]:+.3f}", flush=True)
for c in sorted(modes, key=modes.get)[::7]: print(f"REVK2 {c[:30]:30s} peak error {modes[c]:.4f}  A8 {LEV[c]:.3f}", flush=True)
os.makedirs("xout", exist_ok=True); json.dump(dict(modes=modes), open("xout/sp1_reverse_koide_2.json", "w"), indent=1)
