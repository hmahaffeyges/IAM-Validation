"""DEV-SPHERE-01 construction v3 (written and committed 2026-10-21, BEFORE it is run; the rule was written in DEV_SPHERE_01.md on 2026-10-13).

Why: v1 (a continuous projection) kept distances (rho 0.89) but left most of the globe empty; v2 (a split tree) filled the globe but drew the
twelve HEALPix base faces as hard edges (edge ratio 21.78 on continuous data) and lost distances (rho 0.418).
v3 keeps v1's projection and fills the globe with one monotone, equal-area re-spacing, nothing tuned:
  1. information vectors and the globe exactly as v1 (sphere_construction_01.info_vectors, .globe): each site on the unit sphere by the three
     leading principal directions of its square-root vector;
  2. each site's height z = cos(theta) is replaced by its rank on a uniform scale, z' = -1 + 2 (rank(z) + 1/2) / n, and its longitude by its
     rank, phi' = 2 pi (rank(phi) + 1/2) / n. A uniform z and phi is the equal-area condition on the sphere, and a rank keeps every site's
     order in both coordinates, so neighbors stay neighbors;
  3. the globe point (z', phi') is put in its HEALPix pixel (RING), NSIDE from the site count as v2 (sphere_construction_02.nside_for).
Returns the same objects as v2's globe(), so planted test 3 runs on it unchanged."""
import os, sys, numpy as np, healpy as hp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sphere_construction_01 import info_vectors, fr_angle, concentration, SS, globe as globe_v1
from sphere_construction_02 import nside_for, maps
def respace(X):
    n = len(X); z = np.clip(X[:, 2], -1, 1); phi = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2 * np.pi)
    rz = np.argsort(np.argsort(z, kind="stable"), kind="stable"); rp = np.argsort(np.argsort(phi, kind="stable"), kind="stable")
    z2 = -1 + 2 * (rz + 0.5) / n; p2 = 2 * np.pi * (rp + 0.5) / n; s = np.sqrt(1 - z2 ** 2)
    return np.c_[s * np.cos(p2), s * np.sin(p2), z2]
def globe(U, nside=None):
    """Globe points, RING pixel per site, nside (as sphere_construction_02.globe)."""
    X1, _ = globe_v1(U); X = respace(X1); ns = nside or nside_for(len(U))
    pr = hp.vec2pix(ns, X[:, 0], X[:, 1], X[:, 2]); return X, pr, ns
