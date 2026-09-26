#!/usr/bin/env python3
"""PROC-COV-01: is the reference's misfit against real blood a reproducible, removable bias?

Residual for each healthy array:   r_i = beta_i - sum_c f_c,i mu_c

Everything is leave-one-laboratory-out. The bias is summarised with a MEDIAN, not a mean, per the caution in
ATLAS_READABILITY.md section 5 - the defect that nearly cost two specimens this week was a mean where a
median belonged.
"""
import glob
import json
import lzma
import pickle
import sys

import numpy as np

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
sys.path.insert(0, CH)
import cpg_conductor as C                                                    # noqa: E402

import pandas as pd                                                          # noqa: E402

CLASSES = ("immune", "progenitor", "stem_adult", "stem_pluri", "cycling", "stromal", "secretory", "terminal")


def main():
    ident = json.load(open(C._find("iamatlas_gauge_identity_loci_v1_0.json")))
    ident = ident.get("classes", ident)
    loci = {k: [str(x) for x in v["loci"]] for k, v in ident.items()
            if isinstance(v, dict) and v.get("loci")}
    union = sorted(set().union(*loci.values()))
    head = pd.read_csv("atlas_work/IAMAtlasREBUILD.csv", nrows=0).columns.tolist()
    mu = pd.read_csv("atlas_work/IAMAtlasREBUILD.csv",
                     usecols=[head[0]] + [f"{c}_mean" for c in CLASSES]).set_index(head[0])
    mu.columns = [c[:-5] for c in mu.columns]
    mu = mu.reindex(union).astype(np.float32)
    print("reference: %d loci x %d classes" % mu.shape, flush=True)

    dec_mod = C._load_module("walther_iam_deconvolver", C._find("walther_iam_deconvolver.py"))
    dec = dec_mod.WaltherIAMDeconvolver("atlas_work/IAMAtlasREBUILD.csv",
                                        celltype_class_map=str(C._find("IAMAtlasREBUILD_celltype_to_class.json")))

    resid, labs, gsms = [], [], []
    for p in sorted(glob.glob("iamrepo/Biological_Physics/MethylPhys/reference_data/stage1_betas_*.pkl.xz")):
        gse = p.split("stage1_betas_")[1].split(".")[0]
        df = pickle.load(lzma.open(p, "rb"))
        idx = [x for x in union if x in df.index]
        pos = np.array([union.index(x) for x in idx])
        sub = df.reindex(idx).astype(np.float32)
        M = mu.to_numpy()
        for g in sub.columns:
            b = sub[g]
            fr = dict(dec.deconvolve(df[g].dropna().to_dict()).class_fractions)
            pred = np.zeros(len(idx), dtype=np.float32)
            for j, c in enumerate(CLASSES):
                v = M[pos, j]
                pred += np.float32(fr.get(c, 0.0)) * np.nan_to_num(v, nan=np.nanmean(v))
            r = np.full(len(union), np.nan, dtype=np.float32)
            r[pos] = b.to_numpy() - pred
            resid.append(r)
            labs.append(gse)
            gsms.append(g)
        print("  %s: %d arrays" % (gse, sub.shape[1]), flush=True)

    R = np.vstack(resid)
    labs = np.array(labs)
    print("\nresidual matrix: %d arrays x %d loci" % R.shape, flush=True)

    # ---------------------------------------------------------------- the ordinary check, before any modelling
    with np.errstate(invalid="ignore"):
        per_locus_med = np.nanmedian(R, axis=0)
        per_locus_mean = np.nanmean(R, axis=0)
        per_array_med = np.nanmedian(R, axis=1)
    ok = np.isfinite(per_locus_med)
    print("\n=== ORDINARY CHECK FIRST ===")
    print("  loci with a finite residual: %d of %d" % (ok.sum(), len(union)))
    print("  per-locus median residual:  median %+.4f | IQR %.4f | 1st/99th pct %+.3f / %+.3f"
          % (np.median(per_locus_med[ok]),
             np.subtract(*np.percentile(per_locus_med[ok], [75, 25])),
             np.percentile(per_locus_med[ok], 1), np.percentile(per_locus_med[ok], 99)))
    gap = np.abs(per_locus_mean[ok] - per_locus_med[ok])
    print("  |mean - median| per locus:  median %.4f | 99th pct %.4f  <- where a mean would have misled"
          % (np.median(gap), np.percentile(gap, 99)))
    ext = int((np.abs(per_locus_med[ok]) > 0.3).sum())
    print("  loci whose median residual exceeds 0.3 in beta: %d (%.2f%%) - candidate bad addresses"
          % (ext, 100 * ext / ok.sum()))
    print("  per-ARRAY median residual: %+.4f to %+.4f (spread across specimens)"
          % (per_array_med.min(), per_array_med.max()))

    # ---------------------------------------------------------------- B1 and B2, leave one laboratory out
    uniq = sorted(set(labs))
    b1 = {}
    for held in uniq:
        tr = R[labs != held]
        te = R[labs == held]
        with np.errstate(invalid="ignore"):
            bias = np.nanmedian(tr, axis=0)
            before = np.nanmedian(np.abs(np.nanmedian(te, axis=0)))
            after = np.nanmedian(np.abs(np.nanmedian(te - bias, axis=0)))
        b1[held] = {"n": int((labs == held).sum()), "median_abs_before": float(before),
                    "median_abs_after": float(after),
                    "reduction": float(1 - after / before) if before > 0 else 0.0}
        print("  %-11s n=%-4d median|resid| %.4f -> %.4f   reduction %.1f%%"
              % (held, b1[held]["n"], before, after, 100 * b1[held]["reduction"]), flush=True)
    met1 = all(v["reduction"] >= 0.50 for v in b1.values())
    print("B1 %s (bar: >= 50%% in ALL four folds)" % ("MET" if met1 else "FAILED"))

    pairs, cors = [(uniq[0], uniq[1]), (uniq[2], uniq[3])], []
    with np.errstate(invalid="ignore"):
        v1 = np.nanmedian(R[np.isin(labs, pairs[0])], axis=0)
        v2 = np.nanmedian(R[np.isin(labs, pairs[1])], axis=0)
    m = np.isfinite(v1) & np.isfinite(v2)
    r_pair = float(np.corrcoef(v1[m], v2[m])[0, 1])
    print("\nB2 bias from %s vs %s: r = %+.3f  ->  %s"
          % ("+".join(pairs[0]), "+".join(pairs[1]), r_pair, "MET" if r_pair >= 0.7 else "FAILED"))

    # ---------------------------------------------------------------- B4 does a factor model add anything
    with np.errstate(invalid="ignore"):
        bias_all = np.nanmedian(R, axis=0)
    Rc = R - bias_all
    keep = np.isfinite(Rc).all(axis=0)
    X = Rc[:, keep]
    print("\nB4 factor model on %d arrays x %d complete loci" % X.shape, flush=True)
    tr_i = labs != uniq[0]
    Xtr, Xte = X[tr_i], X[~tr_i]
    Xtr = Xtr - Xtr.mean(0)
    U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
    base = float(np.var(Xte))
    b4 = {}
    for k in (1, 3, 5, 10):
        P = Vt[:k]
        recon = (Xte @ P.T) @ P
        b4[k] = float(1 - np.var(Xte - recon) / base)
        print("   rank %-3d held-out variance explained: %.1f%%" % (k, 100 * b4[k]), flush=True)
    met4 = max(b4.values()) >= 0.20

    out = {"n_arrays": int(R.shape[0]), "n_loci": int(R.shape[1]),
           "ordinary_check": {"per_locus_median_of_medians": float(np.median(per_locus_med[ok])),
                              "mean_median_gap_p99": float(np.percentile(gap, 99)),
                              "loci_gt_0.3": ext, "frac_loci_gt_0.3": float(ext / ok.sum())},
           "b1": b1, "b1_met": bool(met1),
           "b2": {"r": r_pair, "met": bool(r_pair >= 0.7), "pairs": pairs},
           "b4": {"variance_explained": b4, "met": bool(met4)}}
    json.dump(out, open("handoff/cov01.json", "w"), indent=1)
    np.save("handoff/cov01_bias.npy", bias_all)
    np.save("handoff/cov01_loci.npy", np.array(union))
    print("\nwrote handoff/cov01.json and the bias vector")


if __name__ == "__main__":
    main()
