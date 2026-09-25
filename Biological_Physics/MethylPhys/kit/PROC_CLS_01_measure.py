#!/usr/bin/env python3
"""PROC-CLS-01 step 1: the pseudo angular power spectrum of each healthy specimen's residual sky, and its null.

Per array: the chain's own sky (stage_4_6_patient_sky), then healpy.anafast on the masked map, then 20
within-mask permutations of the same pixel values as the null. Writes handoff/cls01_spectra.npz (band powers
and full spectra) and handoff/cls01_meta.json.

THE NULL IS THE POINT. Shuffling the pixel values among the unmasked pixels keeps the mask, the pixel count
and the one-point distribution exactly, and destroys only the spatial arrangement. So any difference between
a specimen's spectrum and its own null is spatial structure and cannot be an artefact of how much sky was
visible or how heavy the tails were.
"""
import json
import lzma
import os
import pickle
import sys
import time

import numpy as np

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
REF = "iamrepo/Biological_Physics/MethylPhys/reference_data"
COHORTS = ["GSE87571", "GSE42861", "GSE111629", "GSE125105"]
NSIDE, LMAX, NPERM = 128, 255, 20
BANDS = [(2, 8), (9, 24), (25, 64), (65, 128), (129, 191), (192, 255)]
sys.path.insert(0, CH)


def bandpowers(cl):
    """Mean C_l in each band - six numbers that a person can compare, from 254 that they cannot."""
    return np.array([float(np.mean(cl[a:b + 1])) for a, b in BANDS])


def main():
    import healpy as hp
    import cpg_conductor as C

    atlas = "atlas_work/IAMAtlasREBUILD.csv"
    assert os.path.exists(atlas), "decompress the atlas first (band01_measure.py does it)"

    band = json.load(open(C._find("identity_band_v3.json")))
    coh = band["_meta"]["cohorts"]
    if isinstance(coh, str):
        import ast as _ast
        coh = _ast.literal_eval(coh)
    zeros = {k.split("_")[0]: v.get("z_lab_full_cohort") for k, v in coh.items()}

    dec_mod = C._load_module("walther_iam_deconvolver", C._find("walther_iam_deconvolver.py"))
    dec = dec_mod.WaltherIAMDeconvolver(atlas, celltype_class_map=str(
        C._find("IAMAtlasREBUILD_celltype_to_class.json")))
    print("deconvolver built", flush=True)

    rng = np.random.default_rng(20260925)
    out = {}
    meta = {"procedure": "PROC-CLS-01", "built": time.strftime("%Y-%m-%d %H:%M"), "nside": NSIDE,
            "lmax": LMAX, "bands": BANDS, "n_perm": NPERM,
            "sky": "stage_4_6_patient_sky as the conductor calls it",
            "null": "pixel values shuffled among unmasked pixels - same mask, same one-point distribution",
            "arrays": {}}
    if os.path.exists("handoff/cls01_spectra.npz"):
        z = np.load("handoff/cls01_spectra.npz", allow_pickle=True)
        out = {k: z[k] for k in z.files}
        meta = json.load(open("handoff/cls01_meta.json"))
        print("resuming:", len(meta["arrays"]), "arrays already done", flush=True)

    first = True
    first_shape = True
    for gse in COHORTS:
        with lzma.open(os.path.join(REF, "stage1_betas_%s.pkl.xz" % gse), "rb") as f:
            df = pickle.load(f)
        todo = [c for c in df.columns if c not in meta["arrays"]]
        print("%s: %d arrays, %d to do" % (gse, df.shape[1], len(todo)), flush=True)
        t0 = time.time()
        for n, gsm in enumerate(todo, 1):
            beta = df[gsm].dropna().to_dict()
            beta_rm, scale_label = C.stage_1s_scale_map(beta, "stage1_noob_450K")
            fr = dict(dec.deconvolve(beta).class_fractions)
            sky = C.stage_4_6_patient_sky(beta_rm, {"class_fractions": fr}, cfg={"lab": gse},
                                          atlas_csv=atlas)
            if first:
                print("  sky keys:", list(sky.keys())[:12], flush=True)
                first = False
            # The sky already carries its pixel maps: sky["_sky"] holds one array per panel, "all" being
            # the whole-sky residual the plate's first panel draws. Re-projecting from CpGs here would be a
            # second implementation of the projection, and the two could drift.
            sk = sky.get("_sky")
            pix = None
            if isinstance(sk, dict):
                pix = sk.get("all_pixels")   # "all" is the summary dict; "all_pixels" is the 196,608-pixel map
            elif sk is not None:
                pix = sk
            if first_shape:
                print("  _sky:", type(sk).__name__,
                      (list(sk.keys())[:10] if isinstance(sk, dict) else getattr(sk, "shape", "?")),
                      "| picked:", getattr(pix, "shape", type(pix).__name__), flush=True)
                first_shape = False
            assert pix is not None and len(np.asarray(pix)) == 12 * NSIDE * NSIDE, \
                "no whole-sky pixel map in the sky output: %s" % (list(sk.keys()) if isinstance(sk, dict) else type(sk))
            pix = np.asarray(pix, dtype=float)
            good = np.isfinite(pix)
            f_sky = float(good.mean())
            m = np.where(good, pix, 0.0)
            cl = hp.anafast(m, lmax=LMAX)
            bp = bandpowers(cl)
            vals = pix[good]
            nulls = np.empty((NPERM, len(BANDS)))
            for k in range(NPERM):
                sh = np.zeros_like(m)
                sh[good] = rng.permutation(vals)
                nulls[k] = bandpowers(hp.anafast(sh, lmax=LMAX))
            out[gsm + "_cl"] = cl.astype(np.float32)
            out[gsm + "_bp"] = bp
            out[gsm + "_null"] = nulls
            meta["arrays"][gsm] = {"gse": gse, "f_sky": round(f_sky, 4),
                                   "frac_abs_z_gt2": float(np.mean(np.abs(vals) > 2)),
                                   "n_pix": int(good.sum())}
            if n % 10 == 0 or n == len(todo):
                np.savez_compressed("handoff/cls01_spectra.npz", **out)
                json.dump(meta, open("handoff/cls01_meta.json", "w"))
                print("  [%d/%d] %s f_sky %.3f  %.1fs/array" % (n, len(todo), gsm, f_sky,
                                                                (time.time() - t0) / n), flush=True)
        np.savez_compressed("handoff/cls01_spectra.npz", **out)
        json.dump(meta, open("handoff/cls01_meta.json", "w"))
        print("%s DONE %.0fs" % (gse, time.time() - t0), flush=True)
    print("TOTAL:", len(meta["arrays"]), "arrays", flush=True)


if __name__ == "__main__":
    os.makedirs("handoff", exist_ok=True)
    main()
