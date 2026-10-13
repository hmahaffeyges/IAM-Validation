"""DEV-SPHERE-01 construction v1 (written and committed 2026-10-13, BEFORE it is applied to any real data).

The information sphere: sites placed by how they behave across cell types, not by where they sit on the DNA.

1. Each site i has a methylation level b_is in every cell type s = 1..S. Its information vector is
       u_i = ( sqrt(b_i1), ..., sqrt(b_iS), sqrt(1-b_i1), ..., sqrt(1-b_iS) ) / sqrt(S),
   which has length 1 for every b (information geometry: the square-root map puts each site on a unit sphere in 2S dimensions).
2. The distance between two sites is the Fisher-Rao (Bhattacharyya) angle  d_ij = arccos(u_i . u_j)  - computed in the full space.
3. The globe: the 2S-dimensional points are centered and projected onto their three leading principal directions, then scaled to
   length 1. Nothing is tuned: no neighbor count, no learning rate, no random seed. What the globe keeps of the full spread is
   reported (variance fraction of the three directions), never hidden.
4. The map: each globe point is put in a HEALPix pixel (NSIDE 128, the chain's sky resolution, sky_statistics.NSIDE). Layers:
   site count per pixel, mean methylation per pixel, and the share of CpG-island sites per pixel when island labels are given.
The CMB statistics are the chain's own (chain/sky_statistics.py), unchanged.
"""
import os, sys, numpy as np
CHAIN = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "chain"))
sys.path.insert(0, CHAIN)
import sky_statistics as SS   # the chain's map statistics (power spectrum by scale, masks, shuffled null)
NSIDE = SS.NSIDE
def info_vectors(B):
    """B: sites x cell types, methylation levels in [0, 1]. Returns unit vectors in 2S dimensions."""
    B = np.clip(np.asarray(B, float), 0.0, 1.0); S = B.shape[1]
    return np.hstack([np.sqrt(B), np.sqrt(1.0 - B)]) / np.sqrt(S)
def fr_angle(U, i, j):
    return np.arccos(np.clip(np.einsum("ij,ij->i", U[i], U[j]), -1.0, 1.0))
def globe(U):
    """Centered projection onto the three leading principal directions, scaled to the unit sphere.
    Returns (points n x 3, variance fraction kept by the three directions)."""
    C = U - U.mean(0)
    w, V = np.linalg.eigh(C.T @ C / len(C)); order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
    X = C @ V[:, :3]; n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X / n, float(w[:3].sum() / w.sum())
def to_pixels(X, nside=NSIDE):
    import healpy as hp
    return hp.vec2pix(nside, X[:, 0], X[:, 1], X[:, 2])
def maps(X, B, islands=None, nside=NSIDE):
    """count, mean methylation, island share per pixel (NaN where a pixel holds no site)."""
    npix = 12 * nside * nside; p = to_pixels(X, nside)
    cnt = np.bincount(p, minlength=npix).astype(float)
    mb = np.full(npix, np.nan); s = np.bincount(p, weights=np.asarray(B, float).mean(1), minlength=npix); ok = cnt > 0; mb[ok] = s[ok] / cnt[ok]
    isl = None
    if islands is not None:
        isl = np.full(npix, np.nan); si = np.bincount(p, weights=np.asarray(islands, float), minlength=npix); isl[ok] = si[ok] / cnt[ok]
    return cnt, mb, isl
def concentration(X, idx):
    """Mean resultant length of a set of globe points: 1 = all at one spot, about 0 = spread over the whole globe."""
    return float(np.linalg.norm(X[idx].mean(0)))
