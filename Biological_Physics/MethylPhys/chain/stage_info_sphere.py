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


# ---- the information sphere picture (2026-10-23; author: "color coded by what groups them together") -------------------------------------
# A site lands where it does on the information sphere because of how it departs from its own level across the cell types. The cell where it
# departs most is the identity it carries, so that cell, and the family it belongs to, is what groups neighbors together; the color says it.
# The families are a STATED grouping of IAM-Atlas v2.1's 79 cell types by tissue and lineage, for the legend only; nothing is computed from them.
FAMILIES = [
 ("T cells", "#ff4d6d", ["cd4 t cells", "cd8 t cells", "cd8 central memory t cells", "cd8 temra t cells", "central memory th1 cells", "cytotoxic cd4 t cells (cd28-)",
   "effector memory cd8 t cells", "memory cd4 t cells", "naive cd4 t cells", "naive cd8 t cells", "regulatory t cells", "t central memory cd4", "t effector cell cd8",
   "t effector memory cd4"]),
 ("B and NK cells", "#ffafcc", ["b cells", "memory b cells", "naive b cells", "nk cells", "cd56bright nk cells"]),
 ("Myeloid cells", "#ff9e00", ["neutrophils", "monocytes", "eosinophils", "basophils", "colon macrophages", "lung alveolar macrophages", "lung interstitial macrophages", "microglia"]),
 ("Bone marrow progenitors", "#ffe74c", ["HSC (bone marrow)", "MPP (bone marrow)", "L-MPP (bone marrow)", "CMP (bone marrow)", "GMP (bone marrow)", "MEP (bone marrow)", "erythrocyte progenitors"]),
 ("Liver", "#38b000", ["hepatocyte"]),
 ("Digestive tract", "#b5e48c", ["colon epithelium", "colon enteroendocrine", "small intestine epithelium", "gastric antrum epithelium", "gastric body epithelium", "gastric fundus epithelium"]),
 ("Pancreas", "#00f5d4", ["pancreatic acinar", "pancreatic alpha", "pancreatic beta", "pancreatic delta", "pancreatic duct"]),
 ("Lung and airway", "#4cc9f0", ["lung alveolar epithelium", "lung bronchus epithelium"]),
 ("Kidney, bladder, prostate", "#4361ee", ["kidney glomerular epithelium", "kidney tubular epithelium", "bladder epithelium", "prostate epithelium"]),
 ("Breast and reproductive", "#f72585", ["breast basal epithelium", "breast luminal epithelium", "endometrium epithelium", "fallopien tubes epithelium"]),
 ("Squamous and thyroid", "#e9edc9", ["tongue epithelium", "tonsil palatine epithelium", "tonsil pharyngeal epithelium", "thyroid epithelium"]),
 ("Blood vessels", "#ff5400", ["aorta endothelium", "kidney glomerular endothelium", "kidney tubular endothelium", "lung alveolar endothelium", "pancreas endothelium",
   "vascular endothelium", "vascular saphenous endothelium"]),
 ("Muscle, heart, fat, fibroblasts", "#c8a165", ["cardiomyocyte", "smooth muscle", "striated muscle", "adipocytes", "colon fibroblasts", "heart fibroblasts"]),
 ("Brain", "#a66cff", ["cortical neurons", "astrocytes", "oligodendrocytes", "oligodendrocyte precursors", "vascular leptomeningeal cells"]),
 ("Embryonic stem cells", "#ffffff", ["embryonic stem cells"])]

def roles(B, cells):
    """For each site: the cell type it departs most from its own mean level in (the identity it carries), the sign (+1: higher there than its
    level, -1: lower), the size of that departure, and how clearly that one cell stands apart (its departure minus the next largest, the
    margin). Returns arrays cell index, sign, size, margin, and the family index of each cell type (FAMILIES order; -1 if unlisted)."""
    B = np.asarray(B, float); D = B - B.mean(1, keepdims=True); A = np.abs(D)
    top = A.argmax(1); r = np.arange(len(B)); size = A[r, top]; sgn = np.sign(D[r, top]).astype(int)
    A2 = A.copy(); A2[r, top] = -1; margin = size - A2.max(1)
    fam = {c: k for k, (_, _, cs) in enumerate(FAMILIES) for c in cs}
    cf = np.array([fam.get(str(c), -1) for c in cells], dtype=int)
    return top.astype(int), sgn, size, margin, cf


# ---- one sample on the information sphere (2026-10-23; DEVELOPMENT, plan 35.2; author: "Create its own normal from the sample and show the changes")
LAYOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Runtime Matrices", "Info_Sphere", "layout_v5.npz")
CLEAR = 0.10        # a site carries one cell's identity when it departs from its own level in that cell by at least this much more than in any other (stated)
MIN_GROUP = 100     # a territory is read when at least this many of its clearly marking sites are measured (stated)

def save_layout(path, ids, X, cell, sgn, size, margin, cells, cell_family, atlas_sha):
    """The information sphere's frozen layout: every site's id, direction and role, so a sample is placed on the same sphere every time."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, ids=np.asarray(ids, dtype="U32"), X=np.asarray(X, "float16"), cell=np.asarray(cell, "uint8"), sgn=np.asarray(sgn, "int8"),
                        size=np.asarray(size, "float16"), margin=np.asarray(margin, "float16"), cells=np.asarray(cells, dtype="U64"),
                        cell_family=np.asarray(cell_family, "int16"), atlas_sha=str(atlas_sha), construction="v5")

def load_layout(path=LAYOUT):
    z = np.load(path, allow_pickle=False); return {k: z[k] for k in z.files}

def groups_of(lay):
    """The territories: each family's clearly marking sites, split by direction (methylated there, or unmethylated), as the picture shows them."""
    fam = lay["cell_family"][lay["cell"]]; clear = lay["margin"].astype(float) >= CLEAR; up = lay["sgn"] > 0
    return [(int(k), int(u), clear & (fam == k) & (up == bool(u))) for k in range(int(fam.max()) + 1) for u in (0, 1)]

def sample_reading(x_by_id, lay, families, b_by_id=None, e_by_id=None):
    """One sample on the information sphere. x_by_id: each site's change from the sample's own expected healthy self, in that site's healthy
    width and the sample's own spread (report_globe_v4.build's zg: the number the address globe reads, so both globes read the same change).
    Each territory's mean change t_g = mean / (sd / sqrt(n)) over its measured sites, then in the sample's own spread of territories (median and
    MAD x 1.4826 over all territories read, as the address globe reads its blocks in its own block spread); a territory has CHANGED beyond the
    line pure chance crosses less than once in 20 samples (5 % family-wise over the territories read). Nothing fitted.
    Also, when b_by_id and e_by_id (the sample and its expectation, 0 to 1) are given: the Fisher-Rao angle between them at each measured site,
    theta = arccos(sqrt(b e) + sqrt((1 - b)(1 - e))), and its root mean square over the sphere's sites: the sample's distance from its own
    healthy expectation, from the same mathematics as relative entropy."""
    import pandas as pd
    from scipy.stats import norm
    ids = pd.Index(lay["ids"].astype(str)); x = pd.Series(x_by_id, dtype="float64").reindex(ids).to_numpy(); meas = np.isfinite(x)
    mu, sd = float(np.mean(x[meas])), float(np.std(x[meas])) or 1.0
    G = []
    for k, u, m in groups_of(lay):
        mm = m & meas; n = int(mm.sum())
        if n >= MIN_GROUP: G.append(dict(family=families[k][0], k=k, methylated_there=bool(u), sites=n, mean=round(float(x[mm].mean()), 4), t=float((x[mm].mean() - mu) / (sd / np.sqrt(n)))))
    t = np.array([g["t"] for g in G]); c = float(np.median(t)); s_ = float(np.median(np.abs(t - c))) * 1.4826 or 1.0; line = float(norm.isf(0.05 / max(len(G), 1) / 2))
    for g in G: g["z"] = round((g["t"] - c) / s_, 2); g["changed"] = bool(abs(g["z"]) > line); g["t"] = round(g["t"], 2)
    rec = dict(stage="information sphere: one sample (development)", construction=str(lay.get("construction", "v5")), sites_on_sphere=int(len(ids)),
               sites_measured=int(meas.sum()), territories_read=len(G), line=round(line, 2), territories_changed=int(sum(g["changed"] for g in G)), territories=G,
               rule="each territory's mean change (sites in their healthy width, in the sample's own spread) in the sample's own spread of territories, beyond the 5 % family-wise line")
    if b_by_id is not None and e_by_id is not None:
        b = pd.Series(b_by_id, dtype="float64").reindex(ids).to_numpy(); e = pd.Series(e_by_id, dtype="float64").reindex(ids).to_numpy(); ok = np.isfinite(b) & np.isfinite(e)
        b, e = np.clip(b[ok], 0, 1), np.clip(e[ok], 0, 1); th = np.arccos(np.clip(np.sqrt(b * e) + np.sqrt((1 - b) * (1 - e)), -1, 1))
        rec["fisher_rao_rms_rad"] = round(float(np.sqrt(np.mean(th ** 2))), 5); rec["fisher_rao_median_rad"] = round(float(np.median(th)), 5)
    lit = np.zeros(len(ids), bool); sign = np.zeros(len(ids), "int8")
    for (k, u, m), g in zip([q for q in groups_of(lay) if int((q[2] & meas).sum()) >= MIN_GROUP], G):
        if g["changed"]: lit |= m & meas; sign[m & meas] = 1 if g["z"] > 0 else -1
    return rec, x, lit, sign
