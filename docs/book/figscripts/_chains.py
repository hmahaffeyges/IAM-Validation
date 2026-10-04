"""Read the repository's Cobaya chains with the book's convention: 30 % burn-in per file, weighted statistics
(same as Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv and docs/verification/scripts/verify_cc_and_baryon.py)."""
import numpy as np, pandas as pd
from _bookstyle import REPO


def load(*files, burn=0.3):
    out = []
    for f in files:
        p = REPO / f
        cols = open(p).readline().lstrip("#").split()
        X = pd.read_csv(p, sep=r"\s+", comment="#", names=cols)
        out.append(X.iloc[int(burn * len(X)):])
    return pd.concat(out, ignore_index=True)


def wmean_sd(v, w):
    m = np.average(v, weights=w)
    return m, np.sqrt(np.average((v - m) ** 2, weights=w))


def wquant(v, w, q):
    i = np.argsort(v); v, w = np.asarray(v)[i], np.asarray(w)[i]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, cw, v)


MG = "Cosmological_Physics/mgcamb_validation/chains/"
L1 = {  # data combination: (LambdaCDM, IAM fixed, mu0 free) file lists
    "Planck": ([MG + "lcdm_baseline.1.txt"], [MG + f"iam_fixed_mu0_r2.{i}.txt" for i in range(1, 5)],
               [MG + f"iam_float_mu0_r2.{i}.txt" for i in range(1, 5)]),
    "Planck + RSD": ([MG + "planck_rsd_lcdm_baseline.1.txt"], [MG + "planck_rsd_iam_fixed.1.txt"], [MG + "planck_rsd_mu0_float.1.txt"]),
    "Planck + BAO": ([MG + "planck_bao_lcdm_baseline.1.txt"], [MG + "planck_bao_iam_fixed.1.txt"], [MG + "planck_bao_mu0_float.1.txt"]),
    "Planck + Pantheon+": ([MG + "planck_pantheon_lcdm_baseline.1.txt"], [MG + "planck_pantheon_iam_fixed.1.txt"], [MG + "planck_pantheon_mu0_float.1.txt"]),
}
L2 = {"A": ["Cosmological_Physics/camb_validation/chains/iam_level2_runA.1.txt"], "C": ["Cosmological_Physics/camb_validation/chains/iam_level2_runC_lcdm.1.txt"]}
