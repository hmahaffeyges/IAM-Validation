"""stage_info_sphere.py - the information sphere (DEVELOPMENT, not commissioned; DEV-SPHERE-01, moved into the chain 2026-10-22).

Sites are placed by how they behave across a reference set of cell types, not by where they sit on the DNA (information geometry):
  info_vectors(B)  each site's levels b_s in S cell types -> u = (sqrt b_1..sqrt b_S, sqrt(1-b_1)..sqrt(1-b_S)) / sqrt S, a unit vector
  fr_angle         the Fisher-Rao (Bhattacharyya) angle arccos(u_i . u_j) between two sites, in the full 2S-dimensional space
  globe(U)         construction v1, the one that kept distances (DEV_SPHERE_01.md): centered, projected onto the three leading principal
                   directions, scaled to length 1. Nothing tuned. The share of the full spread the three directions keep is reported.
  to_pixels        HEALPix NESTED pixel of each globe point (NSIDE 128, the chain's sky resolution, sky_statistics.NSIDE)
  build(B, ...)    the layout and its report: variance kept, sky covered, distance agreement (rank correlation of globe angles with full-
                   space angles on 5,000 seeded pairs), edge ratio; with named site groups, each group's concentration against random groups.
  The empty sky (pixels holding no site) is masked, as the microwave sky masks the galaxy (2026-10-22 design: the sphere keeps distances
  and masks what it does not cover; it does not stretch sites to fill the sky, which cost distances in v2 to v4).
Run by the one command: run_sample.py --build info-sphere --build-input MATRIX --build-out LAYOUT [--build-groups GROUPS.json] --bundle B.json
"""
import os, json, numpy as np
NSIDE = 128

def info_vectors(B):
    B = np.clip(np.asarray(B, float), 0.0, 1.0); S = B.shape[1]
    return np.hstack([np.sqrt(B), np.sqrt(1.0 - B)]) / np.sqrt(S)

def fr_angle(U, i, j):
    return np.arccos(np.clip(np.einsum("ij,ij->i", U[i], U[j]), -1.0, 1.0))

def globe(U):
    C = U - U.mean(0); w, V = np.linalg.eigh(C.T @ C / len(C)); o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
    X = C @ V[:, :3]; n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X / n, float(w[:3].sum() / w.sum())

def level_free(B, U):
    """Construction v5 (2026-10-22, written before it runs; DEV-SPHERE-01 test 8 found a site's place set mostly by its level): each site's
    square-root vector minus the square-root vector of a flat profile at the site's own mean level, b_s = mean over the cell types, then scaled
    to length 1. What remains is only how the site departs from its own level across the cell types; nothing is tuned. A site with no departure
    has no direction (returned as zeros and reported)."""
    m = np.clip(np.asarray(B, float).mean(1, keepdims=True), 0.0, 1.0); S = np.asarray(B).shape[1]
    D = U - info_vectors(np.repeat(m, S, axis=1)); n = np.linalg.norm(D, axis=1, keepdims=True); flat = (n[:, 0] == 0)
    n[n == 0] = 1.0; return D / n, int(flat.sum())

_JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]); _JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])
def _spread(v):   # interleave the bits of v (x -> even bits)
    v = np.asarray(v, np.int64) & 0xFFFFFFFF; r = np.zeros_like(v)
    for b in range(32): r |= ((v >> b) & 1) << (2 * b)
    return r
def ang2pix_nest(nside, theta, phi):
    """The standard HEALPix NESTED angle -> pixel (Gorski et al. 2005), vectorized; checked by a round trip through stage_v4.nest_pix2ang."""
    z = np.cos(theta); za = np.abs(z); tt = np.mod(phi, 2 * np.pi) / (np.pi / 2)
    face = np.empty(len(z), np.int64); ix = np.empty(len(z), np.int64); iy = np.empty(len(z), np.int64)
    eq = za <= 2 / 3
    temp1 = nside * (0.5 + tt[eq]); temp2 = nside * (z[eq] * 0.75)
    jp = (temp1 - temp2).astype(np.int64); jm = (temp1 + temp2).astype(np.int64)
    ifp = jp // nside; ifm = jm // nside
    face[eq] = np.where(ifp == ifm, ifp | 4, np.where(ifp < ifm, ifp, ifm + 8))
    ix[eq] = jm & (nside - 1); iy[eq] = nside - (jp & (nside - 1)) - 1
    po = ~eq; ntt = np.minimum(tt[po].astype(np.int64), 3); tp = tt[po] - ntt
    tmp = nside * np.sqrt(3 * (1 - za[po]))
    jp = np.minimum((tp * tmp).astype(np.int64), nside - 1); jm = np.minimum(((1.0 - tp) * tmp).astype(np.int64), nside - 1)
    north = z[po] > 0
    face[po] = np.where(north, ntt, ntt + 8)
    ix[po] = np.where(north, nside - jm - 1, jp); iy[po] = np.where(north, nside - jp - 1, jm)
    return face * nside * nside + _spread(ix) + (_spread(iy) << 1)

def to_pixels(X, nside=NSIDE):
    th = np.arccos(np.clip(X[:, 2], -1, 1)); ph = np.arctan2(X[:, 1], X[:, 0]); return ang2pix_nest(nside, th, ph)

def concentration(X, idx):
    """The mean resultant length of the group's globe points (directional statistics' standard measure of concentration on a sphere)."""
    return float(np.linalg.norm(X[idx].mean(0)))

def edge_ratio(mb, nside=NSIDE):
    """Mean jump of a map between neighboring pixels on different HEALPix base faces over the same on one face (about 1 = no grid edges).
    Neighbors here: the four pixels sharing an edge in the NESTED x/y index grid of each face, and across faces by position (nearest center)."""
    import stage_v4 as V4
    npix = 12 * nside * nside; p = np.arange(npix); th, ph = V4.nest_pix2ang(nside, p)
    v = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]; face = p // (nside * nside)
    step = np.sqrt(4 * np.pi / npix); same, diff = [], []
    for d in ((1, 0), (0, 1)):
        a = np.c_[np.cos(ph) * np.cos(th), np.sin(ph) * np.cos(th), -np.sin(th)] if d == (1, 0) else np.c_[-np.sin(ph), np.cos(ph), np.zeros(npix)]
        w = v + step * a; w /= np.linalg.norm(w, axis=1, keepdims=True); q = to_pixels(w, nside)
        ok = (q != p) & np.isfinite(mb) & np.isfinite(mb[q]); dd = np.abs(mb[ok] - mb[q[ok]]); sf = face[ok] == face[q[ok]]
        same.append(dd[sf]); diff.append(dd[~sf])
    s, f = np.concatenate(same), np.concatenate(diff)
    return float(np.mean(f) / np.mean(s)) if len(s) and len(f) else None

def build(B, groups=None, n_random=500, seed=20261013, nside=NSIDE, construction="v1"):
    """The layout of a reference matrix B (sites x cell types) and its report. groups: {name: site indices}; for each, the concentration R and
    the 99th percentile of n_random random groups of the same size drawn from groups['pool'] if given, else from all sites."""
    rng = np.random.default_rng(seed); U0 = info_vectors(B); flat = 0
    U = U0 if construction == "v1" else level_free(B, U0)[0]
    if construction == "v5": flat = level_free(B, U0)[1]
    X, kept = globe(U); pix = to_pixels(X, nside); npix = 12 * nside * nside
    cnt = np.bincount(pix, minlength=npix).astype(float); mb = np.full(npix, np.nan); s = np.bincount(pix, weights=np.asarray(B, float).mean(1), minlength=npix)
    mb[cnt > 0] = s[cnt > 0] / cnt[cnt > 0]
    i, j = rng.integers(0, len(U), 5000), rng.integers(0, len(U), 5000)
    from scipy.stats import spearmanr
    gl = np.arccos(np.clip(np.einsum("ij,ij->i", X[i], X[j]), -1, 1))
    rho = float(spearmanr(fr_angle(U, i, j), gl).correlation)   # v1: against the Fisher-Rao angles; v5: against the full-space angles of the level-free vectors
    rho_fr = float(spearmanr(fr_angle(U0, i, j), gl).correlation)
    rep = dict(stage="information sphere (development)", construction=("v1: three leading principal directions of the square-root vectors, scaled to 1" if construction == "v1" else
               "v5: v1 on each site's square-root vector minus that of a flat profile at its own mean level, scaled to 1 (the level taken out)"),
               distance_rho_fisher_rao=round(rho_fr, 4), sites_without_direction=flat,
               sites=int(len(U)), cell_types=int(np.asarray(B).shape[1]), variance_kept=round(kept, 4), nside=nside,
               sky_covered=round(float((cnt > 0).mean()), 4), sky_masked=round(float((cnt == 0).mean()), 4), distance_rho=round(rho, 4),
               edge_ratio=(round(e, 3) if (e := edge_ratio(mb, nside)) is not None else None), groups={})
    pool = np.asarray((groups or {}).get("pool", np.arange(len(U))))
    for name, idx in (groups or {}).items():
        if name == "pool": continue
        idx = np.asarray(idx); R = concentration(X, idx)
        q = float(np.quantile([concentration(X, rng.choice(pool, len(idx), replace=False)) for _ in range(n_random)], 0.99))
        rep["groups"][name] = dict(n=int(len(idx)), R=round(R, 4), random_q99=round(q, 4), together=bool(R > q))
    return dict(X=X.astype("float32"), pix=pix.astype("int32"), count=cnt, mean_b=mb), rep
