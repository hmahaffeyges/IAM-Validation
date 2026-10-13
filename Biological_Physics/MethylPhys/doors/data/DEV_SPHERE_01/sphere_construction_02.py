"""DEV-SPHERE-01 construction v2 (written and committed 2026-10-13, BEFORE it is run; v1 kept unchanged).

Why v2: v1 projected the sites straight onto the globe, which piled every site that behaves alike into one spot and left 94 % of the
synthetic globe empty (planted test 1, synthetic_sphere_v1_flat.png). A sky covers the whole sphere, and the CMB tools assume it.

What stays: the information vectors and the Fisher-Rao angle (sphere_construction_01.info_vectors, fr_angle), measured in full space.
What changes: only how sites are put on the globe. HEALPix in NESTED order is itself a tree - 12 base pixels, each split into 4, then
4 again - so sites are put on it by splitting them the same way, by behavior:
  - the 12 base pixels: split all sites into three equal bands (north, equator, south) along their leading principal direction, then
    each band into four along that band's own leading direction;
  - every further level: split each pixel's sites into 4 children - at the median of their leading principal direction (the child's
    x bit), then each half at the median of its own (the y bit) - as NESTED numbering does.
Sites that behave alike therefore share a parent pixel at every scale, the way nearby sky shares a parent pixel. Every pixel holds
about the same number of sites, so the globe is full by construction. NSIDE is fixed by the rule: the largest power of two with no
more pixels than sites (capped at the chain's sky NSIDE). Nothing is tuned: no neighbor count, no seed, no learning rate.
What it costs, stated up front: the site-DENSITY map carries no information in v2 (it is flat by construction); the readable maps
are mean methylation, island share and, later, departure from the healthy reference. Distances on the globe are tree distances;
their rank agreement with the full-space Fisher-Rao angle is reported every time.
"""
import os, sys, numpy as np, healpy as hp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sphere_construction_01 import info_vectors, fr_angle, concentration, SS
def nside_for(n, cap=SS.NSIDE):
    k = 1
    while 12 * (2 * k) ** 2 <= n and 2 * k <= cap: k *= 2
    return k
def _pc1(U, idx):
    if len(idx) < 3: return np.arange(len(idx), dtype=float)
    C = U[idx] - U[idx].mean(0)
    w, V = np.linalg.eigh(C.T @ C); return C @ V[:, -1]
def _split(U, idx, parts):
    o = idx[np.argsort(_pc1(U, idx), kind="stable")]; return np.array_split(o, parts)
def place(U, nside=None):
    """Returns (nested pixel per site, nside)."""
    n = len(U); nside = nside or nside_for(n); depth = int(np.log2(nside))
    pix = np.zeros(n, dtype=np.int64)
    groups = []
    for bi, band in enumerate(_split(U, np.arange(n), 3)):
        for li, g in enumerate(_split(U, band, 4)): groups.append((4 * bi + li, g))
    for _ in range(depth):
        nxt = []
        for p, g in groups:
            if len(g) == 0: continue
            for xb, half in enumerate(_split(U, g, 2)):
                for yb, q in enumerate(_split(U, half, 2)): nxt.append((4 * p + xb + 2 * yb, q))
        groups = nxt
    for p, g in groups: pix[g] = p
    return pix, nside
def globe(U, nside=None):
    """Globe points (pixel centers), RING pixel per site, nside."""
    pn, ns = place(U, nside); pr = hp.nest2ring(ns, pn)
    X = np.array(hp.pix2vec(ns, pr)).T; return X, pr, ns
def maps(pr, ns, B, islands=None):
    npix = 12 * ns * ns; cnt = np.bincount(pr, minlength=npix).astype(float); ok = cnt > 0
    mb = np.full(npix, np.nan); mb[ok] = np.bincount(pr, weights=np.asarray(B, float).mean(1), minlength=npix)[ok] / cnt[ok]
    isl = None
    if islands is not None:
        isl = np.full(npix, np.nan); isl[ok] = np.bincount(pr, weights=np.asarray(islands, float), minlength=npix)[ok] / cnt[ok]
    return cnt, mb, isl
