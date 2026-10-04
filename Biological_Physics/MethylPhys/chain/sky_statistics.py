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

TWO MASKS (2026-10-04, boxruns/run1/JOBS.md job D). mask="hard" (the default, unchanged) zeroes the masked pixels and returns the raw
pseudo-C_l of the cut sky. mask="apodised" keeps the same footprint but multiplies the map by a weight that rises smoothly from 0 at the
footprint edge to 1 at a distance apod_deg inside it (apodised_mask), so the sharp edge no longer spreads power from one multipole into
its neighbours, and divides the pseudo-C_l by w2 = mean(weight^2). The hard mask's equivalent normalisation is f_sky (w2 of a 0/1
mask); it is not applied, so that every existing hard-mask number reproduces. Ratios to a null cancel either normalisation.
"""
import numpy as np

NSIDE, LMAX, NPERM = 128, 255, 20
MASKS = ("hard", "apodised")
APOD_DEG, APOD_TAPER = 2.0, "C2"     # apodisation width (degrees) and taper of mask="apodised"; NSIDE 128 pixels are 0.46 deg across
_W = {}                              # weights of the last footprint apodised, keyed by (footprint sha1, width, taper)
BANDS = [(2, 8), (9, 24), (25, 64), (65, 128), (129, 191), (192, 255)]


def bandpowers(cl, bands=BANDS):
    """Mean C_l in each band - six numbers that a person can compare, from 254 that they cannot."""
    return np.array([float(np.mean(cl[a:b + 1])) for a, b in bands])


def apodised_mask(good, apod_deg=APOD_DEG, taper=APOD_TAPER):
    """Weight in [0, 1] per pixel: 0 outside the footprint `good` (bool HEALPix RING array), rising to 1 at an angular distance
    apod_deg inside it. theta is the distance from a footprint pixel centre to the centre of the nearest masked pixel that touches
    the footprint; x = sqrt((1 - cos theta) / (1 - cos apod_deg)) (= theta / apod_deg at small angles). Tapers (Grain et al. 2009):
    "C1"  w = x - sin(2 pi x) / (2 pi);  "C2"  w = (1 - cos(pi x)) / 2;  w = 1 for x >= 1.
    A footprint with no masked pixel returns all ones."""
    import healpy as hp
    from scipy.spatial import cKDTree
    if taper not in ("C1", "C2"): raise ValueError("taper must be 'C1' or 'C2', got %r" % (taper,))
    good = np.asarray(good, dtype=bool); nside = hp.npix2nside(good.size); w = good.astype(float)
    if good.all() or not good.any(): return w
    bad = np.where(~good)[0]; nb = hp.get_all_neighbours(nside, bad)
    edge = bad[((nb >= 0) & good[np.where(nb >= 0, nb, 0)]).any(axis=0)]          # masked pixels that touch the footprint
    gi = np.where(good)[0]; chord_max = 2 * np.sin(np.radians(apod_deg) / 2)
    d, _ = cKDTree(np.array(hp.pix2vec(nside, edge)).T).query(np.array(hp.pix2vec(nside, gi)).T, distance_upper_bound=chord_max)
    theta = 2 * np.arcsin(np.clip(d / 2, 0, 1))                                    # inf (beyond apod_deg) -> nan -> weight 1 below
    x = np.sqrt((1 - np.cos(theta)) / (1 - np.cos(np.radians(apod_deg))))
    t = x - np.sin(2 * np.pi * x) / (2 * np.pi) if taper == "C1" else 0.5 * (1 - np.cos(np.pi * x))
    w[gi] = np.where(np.isfinite(d) & (x < 1), t, 1.0)
    return w


def _weights(good, apod_deg, taper):
    import hashlib
    key = (hashlib.sha1(np.packbits(good).tobytes()).hexdigest(), good.size, float(apod_deg), taper)
    if key not in _W: _W.clear(); _W[key] = apodised_mask(good, apod_deg, taper)
    return _W[key]


def masked_spectrum(pix, lmax=LMAX, mask="hard", apod_deg=APOD_DEG, taper=APOD_TAPER):
    """(C_l, f_sky, good-pixel mask) of a HEALPix map whose masked pixels are NaN; masked pixels are set to 0 (healpy.anafast).
    mask="hard" (default): the raw pseudo-C_l of the cut map, as before. mask="apodised": the map times apodised_mask(good, apod_deg,
    taper), pseudo-C_l divided by w2 = mean(weight^2). f_sky is the footprint fraction (good.mean()) in both cases."""
    import healpy as hp
    if mask not in MASKS: raise ValueError("mask must be one of %s, got %r" % (MASKS, mask))
    pix = np.asarray(pix, dtype=float)
    good = np.isfinite(pix)
    m = np.where(good, pix, 0.0)
    if mask == "apodised":
        w = _weights(good, apod_deg, taper)
        return hp.anafast(m * w, lmax=lmax) / float(np.mean(w ** 2)), float(good.mean()), good
    return hp.anafast(m, lmax=lmax), float(good.mean()), good


def shuffled_null(pix, n_perm=NPERM, lmax=LMAX, rng=None, bands=BANDS, mask="hard", apod_deg=APOD_DEG, taper=APOD_TAPER):
    """Band powers of n_perm within-mask permutations of the map's own unmasked pixel values (shape n_perm x len(bands)); each
    permuted map's spectrum is taken with the same mask as masked_spectrum(mask=...)."""
    import healpy as hp
    rng = rng if rng is not None else np.random.default_rng(20260925)
    pix = np.asarray(pix, dtype=float)
    good = np.isfinite(pix)
    vals = pix[good]
    if mask not in MASKS: raise ValueError("mask must be one of %s, got %r" % (MASKS, mask))
    w = _weights(good, apod_deg, taper) if mask == "apodised" else None
    nulls = np.empty((n_perm, len(bands)))
    for k in range(n_perm):
        sh = np.zeros_like(pix)
        sh[good] = rng.permutation(vals)
        nulls[k] = bandpowers(hp.anafast(sh, lmax=lmax) if w is None else hp.anafast(sh * w, lmax=lmax) / float(np.mean(w ** 2)), bands)
    return nulls


def sky_spectrum_record(pix, n_perm=NPERM, lmax=LMAX, rng=None, mask="hard", apod_deg=APOD_DEG, taper=APOD_TAPER):
    """One sky: its spectrum, band powers, f_sky, the fraction of unmasked pixels with |z| > 2, and its shuffled null. With
    mask="apodised" the record also carries the mask ("apodised C2 2.0 deg") and w2; the hard-mask record is unchanged."""
    cl, f_sky, good = masked_spectrum(pix, lmax, mask, apod_deg, taper)
    vals = np.asarray(pix, dtype=float)[good]
    rec = {"cl": cl, "bandpowers": bandpowers(cl), "f_sky": round(f_sky, 4), "n_pix": int(good.sum()),
           "frac_abs_z_gt2": float(np.mean(np.abs(vals) > 2)),
           "null_bandpowers": shuffled_null(pix, n_perm, lmax, rng, BANDS, mask, apod_deg, taper)}
    if mask == "apodised":
        rec.update(mask="apodised %s %s deg" % (taper, apod_deg), w2=round(float(np.mean(_weights(good, apod_deg, taper) ** 2)), 4))
    return rec
