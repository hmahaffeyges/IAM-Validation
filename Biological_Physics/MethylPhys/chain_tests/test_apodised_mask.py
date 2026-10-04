#!/usr/bin/env python3
"""Stage 12 apodised mask (2026-10-04, boxruns/run1/JOBS.md job D): on synthetic HEALPix skies drawn from a known C_l, the apodised mask
(sky_statistics.masked_spectrum(mask="apodised")) recovers the input spectrum better than the hard mask with the same footprint.

Skies: NSIDE 128, l <= 255 (the stage-12 values), N_SEEDS fixed seeds, Gaussian with a red spectrum C_l = (l + 5)^-3 (power falling
steeply with l, as the measured residual sky's is: DEV-SKY-02 band 1). Footprint: a band |latitude| < 15 deg plus 20 discs of
radius 4 deg (fixed seed). Both masks see the same footprint and the same maps; the hard estimate is put on the same scale as the
apodised one by dividing it by f_sky (= w2 of a 0/1 mask; the module's hard path itself returns the raw pseudo-C_l).
Statistic: over l = 2..255, the mean |<C_hat_l> / C_l - 1| (bias of the seed-averaged estimate) and the seed average of
mean_l |C_hat_l / C_l - 1|, plus the six band powers. The full-sky estimate of the same maps is printed as the floor.
Run:  python3 test_apodised_mask.py   (or pytest). Needs healpy and scipy."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chain")); import sky_statistics as SS

N_SEEDS, SEED0, L0 = 8, 20261004, 2


def footprint(nside=SS.NSIDE, seed=1):
    import healpy as hp
    th, _ = hp.pix2ang(nside, np.arange(12 * nside * nside)); good = np.abs(np.pi / 2 - th) >= np.radians(15)
    rng = np.random.default_rng(seed)
    for _ in range(20):
        v = hp.ang2vec(np.arccos(rng.uniform(-1, 1)), rng.uniform(0, 2 * np.pi))
        good[hp.query_disc(nside, v, np.radians(4))] = False
    return good


def run(n_seeds=N_SEEDS, verbose=True):
    import healpy as hp
    ell = np.arange(SS.LMAX + 1); cl_in = (ell + 5.0) ** -3; cl_in[:2] = 0
    good = footprint(); f_sky = float(good.mean()); w = SS.apodised_mask(good); w2 = float(np.mean(w ** 2))
    est = {"full sky": [], "hard / f_sky": [], "apodised": []}
    for k in range(n_seeds):
        np.random.seed(SEED0 + k); m = hp.synfast(cl_in, SS.NSIDE, lmax=SS.LMAX)
        est["full sky"].append(hp.anafast(m, lmax=SS.LMAX))
        sky = np.where(good, m, np.nan)
        cl_h, fs, _ = SS.masked_spectrum(sky); est["hard / f_sky"].append(cl_h / fs)
        est["apodised"].append(SS.masked_spectrum(sky, mask="apodised")[0])
    sl = slice(L0, SS.LMAX + 1); bp_in = SS.bandpowers(cl_in); out = {}
    for name, cls in est.items():
        cls = np.array(cls); r = cls[:, sl] / cl_in[sl] - 1
        out[name] = {"bias": float(np.mean(np.abs(cls.mean(0)[sl] / cl_in[sl] - 1))), "err": float(np.mean(np.abs(r))),
                     "bands": SS.bandpowers(cls.mean(0)) / bp_in}
    if verbose:
        print(f"footprint f_sky {f_sky:.4f}; apodised ({SS.APOD_TAPER}, {SS.APOD_DEG} deg) w2 {w2:.4f}; {n_seeds} seeds; C_l = (l+5)^-3, l {L0}-{SS.LMAX}")
        print(f"{'estimator':<14}{'mean|bias|':>11}{'mean|err|':>11}   band power / input, bands {SS.BANDS}")
        for name, o in out.items():
            print(f"{name:<14}{o['bias']:>11.4f}{o['err']:>11.4f}   " + " ".join(f"{b:6.3f}" for b in o["bands"]))
    return out


def test_apodised_beats_hard():
    o = run(verbose=False)
    assert o["apodised"]["bias"] < o["hard / f_sky"]["bias"]
    assert o["apodised"]["err"] < o["hard / f_sky"]["err"]


def test_default_is_hard():
    rng = np.random.default_rng(3); pix = rng.normal(size=12 * SS.NSIDE ** 2); pix[~footprint()] = np.nan
    import healpy as hp
    assert np.array_equal(SS.masked_spectrum(pix)[0], hp.anafast(np.where(np.isfinite(pix), pix, 0.0), lmax=SS.LMAX))


def test_weights():
    good = footprint(); w = SS.apodised_mask(good)
    assert np.all(w[~good] == 0) and np.all((w >= 0) & (w <= 1)) and w.max() == 1.0
    assert np.all(SS.apodised_mask(np.ones_like(good)) == 1)


if __name__ == "__main__":
    o = run(); ok = o["apodised"]["bias"] < o["hard / f_sky"]["bias"] and o["apodised"]["err"] < o["hard / f_sky"]["err"]
    test_default_is_hard(); test_weights()
    print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
