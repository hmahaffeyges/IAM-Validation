"""DEV-SPHERE-01 construction v4 (written and committed 2026-10-22, BEFORE it is run; to-do step 31).
Why: v3 re-spaced height and longitude separately; the sites lie along a curved band of v1's projection, so each coordinate came out uniform
while the pairs stayed bunched (coverage 46.9 %, bar 90 %; distance agreement 0.567, bar 0.7). v4 fills the sphere JOINTLY, with one fixed
rule and nothing tuned:
  1. information vectors and v1's globe points exactly as before (sphere_construction_01.info_vectors, .globe);
  2. the pole is put on the axis along which v1's points vary least (the smallest principal direction of the points), so the band they lie
     along runs around the equator and its long direction becomes longitude;
  3. the joint equal-area step (a Rosenblatt transform, the exact way to turn any spread of points into a uniform one): height by rank,
     z' = -1 + 2 (rank(z) + 1/2) / n; then the sites cut into k = round(sqrt(n)) bands of equal count by that rank, and longitude ranked
     WITHIN each band, phi' = 2 pi (rank_in_band(phi) + 1/2) / m. Uniform height and, at every height, uniform longitude is the equal-area
     condition for the joint spread, not only for each coordinate; ranks keep each site's order, so neighbors stay neighbors;
  4. the globe point in its HEALPix pixel (RING), NSIDE from the site count as v2 and v3.
Returns the same objects as v3's globe(), so planted test 5 is test 4 with this construction."""
import os, sys, numpy as np, healpy as hp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sphere_construction_01 import info_vectors, fr_angle, concentration, SS, globe as globe_v1
from sphere_construction_02 import nside_for, maps
def respace_joint(X):
    n = len(X); w, V = np.linalg.eigh(np.cov(X.T)); Y = X @ V[:, ::-1]   # columns: largest, middle, smallest direction; pole = smallest
    z = np.clip(Y[:, 2] / np.linalg.norm(Y, axis=1), -1, 1); phi = np.mod(np.arctan2(Y[:, 1], Y[:, 0]), 2 * np.pi)
    rz = np.argsort(np.argsort(z, kind="stable"), kind="stable"); z2 = -1 + 2 * (rz + 0.5) / n
    k = int(round(np.sqrt(n))); band = np.minimum((rz * k) // n, k - 1); p2 = np.empty(n)
    for b in range(k):
        i = np.flatnonzero(band == b); m = len(i); r = np.argsort(np.argsort(phi[i], kind="stable"), kind="stable"); p2[i] = 2 * np.pi * (r + 0.5) / m
    s = np.sqrt(1 - z2 ** 2); return np.c_[s * np.cos(p2), s * np.sin(p2), z2]
def globe(U, nside=None):
    """Globe points, RING pixel per site, nside (as sphere_construction_03.globe)."""
    X1, _ = globe_v1(U); X = respace_joint(X1); ns = nside or nside_for(len(U))
    pr = hp.vec2pix(ns, X[:, 0], X[:, 1], X[:, 2]); return X, pr, ns
