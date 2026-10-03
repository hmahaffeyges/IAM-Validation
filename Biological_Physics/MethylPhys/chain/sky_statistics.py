#!/usr/bin/env python3
# Toolkit: not yet wired into chain v3; enters the chain at commissioning with its own pre-registered check.  (SOP v3 section 2b, stage 12; chain/TOOLKIT.md)
"""sky_statistics.py - stage 12, statistics of a residual sky: the masked pseudo angular power spectrum, band powers, and the
spatially shuffled null.

Copied on 2026-10-03 from the class-era procedure script PROC-CLS-01 (kit/PROC_CLS_01_measure.py, archived privately with the
v2 chain; outcome doors/PROC_CLS_01_OUTCOME.md), without its cohort loop and without the retired conductor it read skies from.
The sky is a HEALPix map (NSIDE 128, RING) such as stage_4_6_patient_cmb.patient_sky_sigma() returns; masked pixels are NaN.

THE NULL IS THE POINT. Shuffling the pixel values among the unmasked pixels keeps the mask, the pixel count and the one-point
distribution exactly, and destroys only the spatial arrangement, so a difference between a sky's spectrum and its own null is
spatial structure. Look-elsewhere: CPG_Null_Runner/cpg_null_runner.py (null N8); the by-simulation correction SOP 2b names for
stage 12 is not built yet (chain/TOOLKIT.md).
"""
import numpy as np

NSIDE, LMAX, NPERM = 128, 255, 20
BANDS = [(2, 8), (9, 24), (25, 64), (65, 128), (129, 191), (192, 255)]


def bandpowers(cl, bands=BANDS):
    """Mean C_l in each band - six numbers that a person can compare, from 254 that they cannot."""
    return np.array([float(np.mean(cl[a:b + 1])) for a, b in bands])


def masked_spectrum(pix, lmax=LMAX):
    """(C_l, f_sky, good-pixel mask) of a HEALPix map whose masked pixels are NaN; masked pixels are set to 0 (healpy.anafast)."""
    import healpy as hp
    pix = np.asarray(pix, dtype=float)
    good = np.isfinite(pix)
    m = np.where(good, pix, 0.0)
    return hp.anafast(m, lmax=lmax), float(good.mean()), good


def shuffled_null(pix, n_perm=NPERM, lmax=LMAX, rng=None, bands=BANDS):
    """Band powers of n_perm within-mask permutations of the map's own unmasked pixel values (shape n_perm x len(bands))."""
    import healpy as hp
    rng = rng if rng is not None else np.random.default_rng(20260925)
    pix = np.asarray(pix, dtype=float)
    good = np.isfinite(pix)
    vals = pix[good]
    nulls = np.empty((n_perm, len(bands)))
    for k in range(n_perm):
        sh = np.zeros_like(pix)
        sh[good] = rng.permutation(vals)
        nulls[k] = bandpowers(hp.anafast(sh, lmax=lmax), bands)
    return nulls


def sky_spectrum_record(pix, n_perm=NPERM, lmax=LMAX, rng=None):
    """One sky: its spectrum, band powers, f_sky, the fraction of unmasked pixels with |z| > 2, and its shuffled null."""
    cl, f_sky, good = masked_spectrum(pix, lmax)
    vals = np.asarray(pix, dtype=float)[good]
    return {"cl": cl, "bandpowers": bandpowers(cl), "f_sky": round(f_sky, 4), "n_pix": int(good.sum()),
            "frac_abs_z_gt2": float(np.mean(np.abs(vals) > 2)), "null_bandpowers": shuffled_null(pix, n_perm, lmax, rng)}
