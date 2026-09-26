#!/usr/bin/env python3
"""PROC-PARTIAL-01: can a non-blood class's FIDELITY SCORE be recovered from whole blood?

The question the author sharpened: finding secretory cells in blood is not the same as scoring them, and a
composition number without an A-score does not mean much.

Estimator (the chain's own sky machinery, with one class left in the subtraction):

    mu_hat_k = (beta_obs - sum_{c != k} f_c mu_c) / f_k          f from the chain's OWN fitted fractions
    A_hat    = H(mean mu_hat_k over class k's identity loci) / H_min_k

against the truth, which is known exactly because the spiked material IS the atlas class mean:

    A_true   = H(mean mu_atlas_k over the same loci) / H_min_k

and against the naive alternative that does no deconvolution at all, which B3 requires it to beat:

    A_naive  = H(mean beta_obs over the same loci) / H_min_k

Runs on the 5,088 mixtures PROC-FOREIGN-01 already scored - same hosts, same fractions, same fitted
compositions, no new data and no refitting.
"""
import glob
import json
import lzma
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
sys.path.insert(0, CH)
import cpg_conductor as C                                                    # noqa: E402

SIGMA = 0.02092                       # the immune band's own width - the only measured healthy spread
CLASSES = ("secretory", "terminal", "stromal")


def H(b):
    """The chain's entropy, copied from cpg_conductor.stage_b_identity so the scale is identical."""
    b = min(max(b, 1e-12), 1 - 1e-12)
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)


def main():
    ident = json.load(open(C._find("iamatlas_gauge_identity_loci_v1_0.json")))
    ident = ident.get("classes", ident)
    loci = {k: [str(x) for x in ident[k]["loci"]] for k in CLASSES}
    hmin = {k: float(ident[k]["H_min"]) for k in CLASSES}
    need = sorted(set().union(*loci.values()))
    print("identity loci needed: %d across %s" % (len(need), ", ".join(CLASSES)), flush=True)

    head = pd.read_csv("atlas_work/IAMAtlasREBUILD.csv", nrows=0).columns.tolist()
    cls_cols = [c for c in head if c.endswith("_mean") and c[:-5] in
                ("immune", "progenitor", "stem_adult", "stem_pluri", "cycling", "stromal",
                 "secretory", "terminal")]
    mu = pd.read_csv("atlas_work/IAMAtlasREBUILD.csv", usecols=[head[0]] + cls_cols).set_index(head[0])
    mu.columns = [c[:-5] for c in mu.columns]
    mu = mu.reindex(need).astype(float)
    print("atlas class means: %d loci x %d classes" % mu.shape, flush=True)

    hosts = {}
    for p in sorted(glob.glob("iamrepo/Biological_Physics/MethylPhys/reference_data/stage1_betas_*.pkl.xz")):
        df = pickle.load(lzma.open(p, "rb"))
        keep = [x for x in need if x in df.index]
        sub = df.reindex(keep).astype(np.float32)
        for gsm in sub.columns:
            hosts[gsm] = sub[gsm]
    print("host arrays loaded: %d" % len(hosts), flush=True)

    rows = json.load(open("handoff/foreign01.json"))["rows"]
    A_true = {k: H(float(mu[k].reindex(loci[k]).dropna().mean())) / hmin[k] for k in CLASSES}
    print("A_true (the atlas material's own score):",
          {k: round(v, 4) for k, v in A_true.items()}, flush=True)

    out = []
    for n, r in enumerate(rows, 1):
        k = r["foreign_class"]
        if k not in CLASSES or float(r["f"]) <= 0:
            continue
        host = hosts.get(r["gsm"])
        if host is None:
            continue
        f = float(r["f"])
        fr = r["fr"] if isinstance(r["fr"], dict) else json.loads(r["fr"].replace("'", '"'))
        fk = float(fr.get(k, 0.0))
        idx = loci[k]
        h = host.reindex(idx).astype(float)
        m_k = mu[k].reindex(idx)
        ok = h.notna() & m_k.notna()
        if ok.sum() < 1000 or fk <= 0:
            continue
        beta = (1 - f) * h[ok].to_numpy() + f * m_k[ok].to_numpy()          # the FOREIGN-01 mixture recipe
        # subtract every class EXCEPT k, using the chain's own fitted fractions
        pred = np.zeros(int(ok.sum()))
        for c, fc in fr.items():
            if c == k or c not in mu.columns:
                continue
            v = mu[c].reindex(idx)[ok].to_numpy()
            pred += float(fc) * np.nan_to_num(v, nan=float(np.nanmean(v)))
        mu_hat = (beta - pred) / fk
        out.append({"gsm": r["gsm"], "class": k, "f": f, "f_hat": fk,
                    "A_hat": H(float(np.clip(mu_hat.mean(), 0, 1))) / hmin[k],
                    "A_naive": H(float(beta.mean())) / hmin[k],
                    "A_true": A_true[k], "n_loci": int(ok.sum())})
        if n % 1000 == 0:
            print("  %d/%d" % (n, len(rows)), flush=True)

    json.dump({"A_true": A_true, "sigma": SIGMA, "rows": out},
              open("handoff/partial01.json", "w"))
    print("\nestimated %d mixtures" % len(out), flush=True)


if __name__ == "__main__":
    main()
