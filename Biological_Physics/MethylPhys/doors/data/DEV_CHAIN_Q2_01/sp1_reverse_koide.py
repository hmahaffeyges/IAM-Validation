import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', '-r', 'Biological_Physics/MethylPhys/chain/requirements.txt', 'scipy'], check=True, timeout=900)
#!/usr/bin/env python3
"""SP1 reverse Koide (register '#### SP1 REVERSE KOIDE, PREDICTION', written before this ran). The chain's sphere columns (mu, mu_se per cell
per site): R1 each type's level at its held / unmarked sites against the law; R2 each fully classified site's Koide axis share against the
law's rebuild from its job breadth k; R3 the Fisher-Rao angle between the law-built profile and the atlas's, against the atlas's own error."""
import sys, os, json, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "Biological_Physics/MethylPhys/chain"); import stage_seq_atlas as SQ
D = SQ.SPHERE_DIR; BL = SQ.BLANK
def load(stem):
    out = {}
    for k in (1, 2, 3):
        t = pq.read_table(os.path.join(D, f"{stem}_part{k}.parquet"))
        for c in t.column_names: u = t.column(c).to_numpy(); v = u.astype("float32") / 1e4; v[u == BL] = np.nan; out[c] = v
    return out
MU = load("mu"); SE = load("mu_se"); LEV = {'erythrocyte progenitors': 0.8043, 'cortical neurons': 0.8954, 'naive cd8 t cells': 0.8979, 'naive cd4 t cells': 0.9323, 'lung alveolar macrophages': 0.9464, 'pancreatic acinar': 0.951, 'smooth muscle': 0.9583, 'lung interstitial macrophages': 0.9597, 'heart fibroblasts': 0.9713, 'breast luminal epithelium': 0.9937, 'hepatocyte': 0.9986, 'breast basal epithelium': 1.0012, 'effector memory cd8 t cells': 1.0014, 'colon macrophages': 1.0087, 'endometrium epithelium': 1.0199, 'monocytes': 1.0207, 'b cells': 1.0256, 'oligodendrocytes': 1.0266, 'colon epithelium': 1.0302, 'colon enteroendocrine': 1.0314, 'fallopien tubes epithelium': 1.035, 'nk cells': 1.0445, 't effector memory cd4': 1.0451, 'cd8 t cells': 1.0491, 'cardiomyocyte': 1.0541, 'gastric fundus epithelium': 1.0551, 'cd4 t cells': 1.0563, 'small intestine epithelium': 1.0621, 'aorta endothelium': 1.0622, 'kidney tubular epithelium': 1.0633, 'gastric body epithelium': 1.0643, 'lung alveolar epithelium': 1.0672, 'pancreatic duct': 1.0698, 'prostate epithelium': 1.0708, 'gastric antrum epithelium': 1.0766, 'lung alveolar endothelium': 1.0806, 'tonsil pharyngeal epithelium': 1.0835, 'tonsil palatine epithelium': 1.0935, 'memory b cells': 1.0957, 'colon fibroblasts': 1.0982, 'kidney glomerular epithelium': 1.1052, 't effector cell cd8': 1.1056, 'bladder epithelium': 1.1074, 'thyroid epithelium': 1.1164, 'lung bronchus epithelium': 1.1187, 'adipocytes': 1.1201, 'vascular saphenous endothelium': 1.1272, 'pancreatic beta': 1.1281, 'tongue epithelium': 1.1327, 'kidney glomerular endothelium': 1.1414, 'kidney tubular endothelium': 1.142, 'pancreatic alpha': 1.1459, 'pancreas endothelium': 1.1498, 'pancreatic delta': 1.1651, 'striated muscle': 1.1709, 't central memory cd4': 1.1713}
cells = sorted(c for c in MU if c in LEV); S = len(cells); print("::notice::REVK cells", S, "of", len(MU), flush=True)
M = np.vstack([MU[c] for c in cells]).T; E = np.vstack([SE[c] for c in cells]).T
ok = np.isfinite(M).all(1); M, E = M[ok], E[ok]; print("::notice::REVK sites with every type", int(ok.sum()), flush=True)
EPS0 = 0.04454; h = 1 - EPS0; s = 1 / 61 + 0.0044 + 0.00046
held = M >= 0.8; un = M <= 0.2
R1 = {}
for j, c in enumerate(cells):
    R1[c] = dict(held_err=round(float(1 - M[held[:, j], j].mean()), 5), n_held=int(held[:, j].sum()), unmarked=round(float(M[un[:, j], j].mean()), 5), n_un=int(un[:, j].sum()))
he = np.array([R1[c]["held_err"] for c in cells]); um = np.array([R1[c]["unmarked"] for c in cells]); lv = np.array([LEV[c] for c in cells])
from scipy.stats import spearmanr
print(f"::notice::REVK R1 held error: median {np.median(he):.5f} (law {EPS0}); within 0.75-1.33 x law {np.mean((he > 0.75 * EPS0) & (he < 1.333 * EPS0)):.2f} of types; vs A8 level rho {spearmanr(he, lv)[0]:+.3f}", flush=True)
print(f"::notice::REVK R1 unmarked: median {np.median(um):.5f} (law as read {s:.5f}); within 0.5-2 x {np.mean((um > 0.5 * s) & (um < 2 * s)):.2f} of types", flush=True)
full = (held | un).all(1); k = held[full].sum(1); V = np.sqrt(np.clip(M[full], 0, 1))
share = V.sum(1) ** 2 / (S * (V ** 2).sum(1))
law = lambda kk: (kk * np.sqrt(h) + (S - kk) * np.sqrt(s)) ** 2 / (S * (kk * h + (S - kk) * s))
R2 = []
for kk in np.unique(k):
    m = k == kk
    if m.sum() >= 100: R2.append(dict(k=int(kk), sites=int(m.sum()), measured=round(float(np.median(share[m])), 4), law=round(float(law(kk)), 4)))
dev = [abs(x["measured"] / x["law"] - 1) for x in R2]
print(f"::notice::REVK R2 fully classified sites {int(full.sum())}; k with >= 100 sites: {len(R2)}; max |measured/law - 1| {max(dev):.4f}; within 5 % at {np.mean(np.array(dev) < 0.05):.2f} of k", flush=True)
for x in R2[:: max(1, len(R2) // 12)]: print(f"REVK R2 k {x['k']:2d} sites {x['sites']:7d} measured {x['measured']:.4f} law {x['law']:.4f}", flush=True)
th = lambda b, e: np.arccos(np.clip(np.sqrt(b * e) + np.sqrt((1 - b) * (1 - e)), -1, 1))
Mf, Ef = M[full], E[full]
def r3(eps_vec, name):
    L = np.where(held[full], 1 - eps_vec[None, :], s)
    a = th(np.clip(Mf, 0, 1), L); ec = np.clip(Mf, 0.005, 0.995); noise2 = np.nan_to_num(Ef) ** 2 / (4 * ec * (1 - ec))
    rms = float(np.sqrt(np.mean(a ** 2))); exp = float(np.sqrt(np.mean(noise2)))
    hs, us = held[full], ~held[full]
    print(f"::notice::REVK R3 {name}: law vs atlas angle rms {rms:.4f} rad, atlas own error {exp:.4f} rad, ratio {rms / exp:.2f}; held part {np.sqrt(np.mean(a[hs] ** 2)):.4f}, unmarked part {np.sqrt(np.mean(a[us] ** 2)):.4f}", flush=True)
    return dict(rms=rms, expected=exp, ratio=rms / exp)
R3a = r3(np.full(S, EPS0), "eps0 for every type"); R3b = r3(EPS0 * lv, "eps0 x the type's A8 level")
os.makedirs("xout", exist_ok=True); json.dump(dict(R1=R1, R2=R2, R3a=R3a, R3b=R3b), open("xout/sp1_reverse_koide.json", "w"), indent=1)
