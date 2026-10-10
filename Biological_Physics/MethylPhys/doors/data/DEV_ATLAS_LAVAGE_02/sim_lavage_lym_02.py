"""DEV-ATLAS-LAVAGE-02 simulation (2026-10-10, before any GSE206709 array is downloaded): lymphocyte fraction recovered by the lavage method of
DEV-ATLAS-LAVAGE-01 (same 60 atlas blocks, LUNG panel, 8,000 loci, NNLS on posterior means, leukocyte renormalisation), with lymphocytes
5-60 % (beryllium disease, sarcoidosis) and the count from a 400-cell cytospin differential. Same noise conditions as the 10-09 lavage simulation."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "../../../../../development/sims"))
from scipy.optimize import nnls
import atlas_sims_01 as A
D, cells = A.load(); Mu, Dd = A.panel(D, cells, A.LUNG, 8000)
def truth(r):
    lym = r.uniform(0.05, 0.60); neu = r.uniform(0, 0.05); eos = r.uniform(0, 0.02); epi = r.uniform(0, 0.03); mac = 1 - lym - neu - eos - epi
    f = np.zeros(11); f[0], f[1], f[2] = mac * 0.85, mac * 0.10, mac * 0.05; f[3:7] = lym * r.dirichlet([5, 3, 1.5, 1]); f[7], f[8], f[9], f[10] = neu, eos, epi / 2, epi / 2
    return f
def run(sd_arr, sd_bio, sd_lab=0.0, n=300, seed=31):
    r = np.random.default_rng(seed); labo = r.normal(0, sd_lab, Mu.shape[1]); e = []
    for _ in range(n):
        f = truth(r); person = Dd[r.integers(Dd.shape[0])] + r.normal(0, sd_bio, Mu.shape)
        b = np.clip(f @ person + labo + r.normal(0, sd_arr, Mu.shape[1]), 0, 1); w, _ = nnls(Mu.T, b); w /= w.sum(); leu = w[:9] / w[:9].sum()
        lt = f[3:7].sum() / f[:9].sum(); cnt = r.binomial(400, lt) / 400; e.append((leu[3:7].sum() - lt, leu[3:7].sum() - cnt))
    e = np.array(e); return dict(MAE_truth=round(np.abs(e[:, 0]).mean(), 4), MAE_count=round(np.abs(e[:, 1]).mean(), 4), bias=round(e[:, 0].mean(), 4),
                                 within_005=round((np.abs(e[:, 1]) <= 0.05).mean(), 3), within_010=round((np.abs(e[:, 1]) <= 0.10).mean(), 3))
for c in ((0.02, 0.0), (0.04, 0.02), (0.06, 0.04), (0.04, 0.02, 0.06)): print(c, run(*c), flush=True)
