"""The Sequencing Atlas Globe at every CpG (DEV-SEQ-GLOBE-WG-01, plan 60.23; development). Every one of the IAM Sequencing Atlas's 28,217,448
hg19 CpGs is placed on a HEALPix sphere, for each of its 57 cell types, with its healthy level and the 95 % range of that level, by the rules the
current Sequencing Atlas Globe already uses: the level and range from the atlas's counts by stage_seq_atlas.range_of_average (pooled level, the
range of the cell's average by donors and counting), and the placement of sky_maps_v21.to_sky (the CpG numbering is in genome order, chr1 to
chr22, X, Y; CpG i of n goes to pixel floor(i x NPIX / n), NESTED ordering; a pixel holds the mean of the CpGs that land in it).

Built from the published atlas (Zenodo 10.5281/zenodo.23934666: 113 pieces of 250,000 CpGs in 12 sets), one set per machine, so anyone can
rebuild it from the deposit. Each piece gives, per cell, the sums and counts of its CpGs' level, low and high in each pixel it touches; sums
and counts add exactly across pieces (a pixel at a piece edge can take CpGs from two pieces at NSIDE 1024), and the joined map is their mean.

Usage:
  python3 seq_globe_wg.py synthetic OUT.txt                      the synthetic (constructed atlas, known truth), written before any atlas value is read
  python3 seq_globe_wg.py part NSIDE INPUTS_CSV OUTDIR PIECE...   sums and counts of these pieces (one .npz per piece)
  python3 seq_globe_wg.py join NSIDE PARTDIR OUTDIR               the full maps, uint16 x 10,000 (65535 no CpG), one .npy per cell and kind
  python3 seq_globe_wg.py slice INPUTS_CSV OUTDIR PIECE...        NSIDE 2048: each piece's own stretch of the final globe, plus NSIDE 256 sums for the textures
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "chain"))
HG19_CPGS = 28217448; BLANK = 65535; KINDS = ("mu", "lo", "hi")

def npix(nside): return 12 * nside * nside

def pixel_of(i0, nside, n=HG19_CPGS):
    """0-based CpG index (genome order) -> NESTED pixel, the to_sky rule."""
    return (np.asarray(i0, np.int64) * npix(nside)) // n

def _dec(a, scale=1.0):
    a = np.asarray(a); return np.where(a == BLANK, np.nan, a.astype("float64") / scale)

def piece_sums(cpg, cols, K, nside, n=HG19_CPGS):
    """cpg: 1-based CpG index of each row; cols: {cell: (meth, reads, spread_x1e4)} as stored in the pieces (uint16, 65535 blank);
    K: {cell: number of healthy inputs}. Returns (first pixel, {cell: {kind: sum, 'n': count}}) over the pixels these CpGs touch."""
    import stage_seq_atlas as SQ
    pix = pixel_of(np.asarray(cpg, np.int64) - 1, nside, n); p0 = int(pix.min()); L = int(pix.max()) - p0 + 1; rel = pix - p0; out = {}
    for c, (meth, reads, spread) in cols.items():
        r = _dec(reads); m = _dec(meth); sd = _dec(spread, 1e4)
        p, lo, hi, _ = SQ.range_of_average(np.nan_to_num(m), np.nan_to_num(r), sd, int(K.get(c, 1)))
        ok = np.isfinite(p) & (np.nan_to_num(r) > 0); D = {"n": np.bincount(rel[ok], minlength=L).astype(np.uint16)}
        for k, v in zip(KINDS, (p, lo, hi)): D[k] = np.bincount(rel[ok], v[ok], minlength=L)   # 2026-11-06: float64 (float32 sums tipped values on a rounding boundary; synthetic run 1)
        out[c] = D
    return p0, out

def join(nside, parts, cells):
    """parts: list of (p0, {cell: sums}) -> {cell: {kind: uint16 map}}; sums and counts added, then the mean."""
    N = npix(nside); res = {}
    for c in cells:
        S = {k: np.zeros(N, np.float64) for k in KINDS}; C = np.zeros(N, np.int64)
        for p0, P in parts:
            if c not in P: continue
            L = len(P[c]["n"]); C[p0:p0 + L] += P[c]["n"]
            for k in KINDS: S[k][p0:p0 + L] += P[c][k]
        res[c] = {k: np.where(C > 0, np.rint(np.clip(S[k] / np.maximum(C, 1), 0, 1) * 1e4), BLANK).astype(np.uint16) for k in KINDS}
        res[c]["n"] = np.minimum(C, 255).astype(np.uint8)
    return res

def synthetic(out_txt, nside=2048, seed=20261106, n=HG19_CPGS, nside_b=1024, piece=250_000, j0=15_000_000):
    """Constructed atlas at every CpG (three cells, known truth), built in pieces of 250,000 exactly as the real build is. Bars:
    1. every CpG with a value lands in the map: the CpGs counted in the map equal the CpGs with reads, for every cell;
    2. a change planted in one cell at about 10 neighboring CpGs (an island's size) changes the map at exactly the planted CpGs' pixels, in that
       cell only, and nowhere else (the builder is exact, so the check is exact; the Sample Globe's 5 % family-wise test is its own step);
    3. one CpG, one pixel: at NSIDE 2048 no pixel holds two CpGs, and at NSIDE 1024 every CpG's pixel mean equals the mean of the CpGs placed there."""
    rng = np.random.default_rng(seed); cells = ["cell a", "cell b", "cell c"]; K = {"cell a": 3, "cell b": 1, "cell c": 2}
    lvl = {c: rng.beta(0.4, 0.4, n) for c in cells}; reads = {c: rng.integers(0, 40, n).astype(np.int64) for c in cells}
    for c in cells: reads[c][rng.random(n) < 0.03] = 0   # unmeasured CpGs
    def stored(L, R):
        m = rng.binomial(R, L); sd = np.where(R > 0, np.rint(rng.uniform(0, 0.08, len(R)) * 1e4), BLANK)
        return (np.where(R > 0, m, BLANK).astype(np.uint16), np.where(R > 0, np.minimum(R, BLANK - 1), BLANK).astype(np.uint16), sd.astype(np.uint16))
    base = {c: stored(lvl[c], reads[c]) for c in cells}
    # the planted change: cell b, 10 neighboring CpGs in chromosome 7's range (fixed index), every read flipped to the other state
    J = np.arange(j0, j0 + 10)
    chg = {c: tuple(x.copy() for x in base[c]) for c in cells}; m, r, s = chg["cell b"]
    while (r[J] == BLANK).any() or (r[J] == 0).any() or (2 * m[J].astype(np.int64) == r[J]).any(): J = J + 10   # planted where reads exist and flipping changes the level
    m[J] = np.where(r[J] != BLANK, r[J] - m[J], BLANK)
    def build(tab, ns):
        parts = []
        for a in range(0, n, piece):
            b = min(a + piece, n); cpg = np.arange(a + 1, b + 1)
            parts.append(piece_sums(cpg, {c: tuple(x[a:b] for x in tab[c]) for c in cells}, K, ns, n))
        return join(ns, parts, cells)
    L_ = []
    G = build(base, nside); H = build(chg, nside)
    ok1 = all(int(G[c]["n"].astype(np.int64).sum()) == int((base[c][1] != BLANK).sum()) for c in cells)
    L_ += [f"bar 1 (every CpG with reads lands): " + "; ".join(f"{c} {int(G[c]['n'].astype(np.int64).sum())} in the map, {int((base[c][1] != BLANK).sum())} with reads" for c in cells) + (" (met)" if ok1 else " (not met)")]
    want = set(pixel_of(J, nside, n).tolist()); diff = {c: set(np.flatnonzero(np.any([G[c][k] != H[c][k] for k in KINDS], axis=0)).tolist()) for c in cells}
    ok2 = diff["cell b"] == want and not diff["cell a"] and not diff["cell c"]
    L_ += [f"bar 2 (planted change at {len(J)} CpGs, pixels {min(want)} to {max(want)}): cell b changed at {len(diff['cell b'])} pixels, all planted: {diff['cell b'] == want}; "
           f"cell a {len(diff['cell a'])}, cell c {len(diff['cell c'])} changed" + (" (met)" if ok2 else " (not met)")]
    mx = int(G["cell a"]["n"].max()); G1 = build(base, nside_b)
    import stage_seq_atlas as SQ
    p, lo, hi, _ = SQ.range_of_average(np.nan_to_num(_dec(base["cell a"][0])), np.nan_to_num(_dec(base["cell a"][1])), _dec(base["cell a"][2], 1e4), K["cell a"])
    px = pixel_of(np.arange(n), nside_b, n); sel = np.isfinite(p) & (base["cell a"][1] != BLANK)
    mean = np.bincount(px[sel], p[sel], npix(nside_b)) / np.maximum(np.bincount(px[sel], minlength=npix(nside_b)), 1)
    got = np.where(G1["cell a"]["mu"] == BLANK, np.nan, G1["cell a"]["mu"] / 1e4); has = np.bincount(px[sel], minlength=npix(nside_b)) > 0
    err = float(np.nanmax(np.abs(got[has] - mean[has])))
    ok3 = mx == 1 and err <= 0.5e-4 + 1e-9
    L_ += [f"bar 3 (one CpG, one pixel): most CpGs in one NSIDE {nside} pixel {mx}; at NSIDE {nside_b} the largest gap between a pixel and the mean of its CpGs {err:.6f} (rounding to 0.0001 allows 0.00005)" + (" (met)" if ok3 else " (not met)")]
    L_ += [f"empty pixels at NSIDE {nside} (no CpG placed): {int((G['cell a']['n'] == 0).sum())} of {npix(nside)}; at NSIDE {nside_b}: {int((G1['cell a']['n'] == 0).sum())} of {npix(nside_b)}",
           "test: " + ("met" if ok1 and ok2 and ok3 else "not met")]
    open(out_txt, "w").write("\n".join(L_) + "\n"); print("\n".join(L_)); return ok1 and ok2 and ok3

def _cells_K(inputs_csv):
    import pandas as pd
    I = pd.read_csv(inputs_csv); return I.groupby(I.cell.str.strip()).size().to_dict()

def part(nside, inputs_csv, outdir, pieces):
    import pyarrow.parquet as pq
    K = _cells_K(inputs_csv); os.makedirs(outdir, exist_ok=True); rep = []
    for f in pieces:
        T = pq.read_table(f); names = T.column_names; cells = sorted({x.split("|")[0] for x in names if "|" in x})
        cols = {c: (T.column(f"{c}|meth").to_numpy(), T.column(f"{c}|reads").to_numpy(), T.column(f"{c}|spread").to_numpy()) for c in cells}
        p0, S = piece_sums(T.column("cpg").to_numpy(), cols, K, nside)
        flat = {"p0": np.array([p0])}
        for c, D in S.items():
            for k, v in D.items(): flat[f"{c}|{k}"] = v
        np.savez_compressed(os.path.join(outdir, os.path.basename(f).replace(".parquet", f"_n{nside}.npz")), **flat)
        rep.append(dict(piece=os.path.basename(f), cpgs=int(T.num_rows), cells=len(cells), first_pixel=p0, cpgs_placed={c: int(S[c]["n"].astype(np.int64).sum()) for c in cells}))
        print("piece", os.path.basename(f), T.num_rows, flush=True); del T, cols, S
    json.dump(rep, open(os.path.join(outdir, f"part_report_n{nside}.json"), "w"), indent=1); return rep

def join_dir(nside, partdir, outdir):
    files = sorted(glob.glob(os.path.join(partdir, f"*_n{nside}.npz"))); parts = []; cells = set()
    for f in files:
        Z = np.load(f); p0 = int(Z["p0"][0]); P = {}
        for k in Z.files:
            if k == "p0": continue
            c, kind = k.rsplit("|", 1); P.setdefault(c, {})[kind] = Z[k]; cells.add(c)
        parts.append((p0, P))
    os.makedirs(outdir, exist_ok=True); rep = dict(nside=nside, pieces=len(files), cells={})
    for c in sorted(cells):
        R = join(nside, parts, [c])[c]
        for k, v in R.items(): np.save(os.path.join(outdir, f"{c.replace(' ', '_')}_{k}_n{nside}.npy"), v)
        rep["cells"][c] = dict(cpgs_in_map=int(R["n"].astype(np.int64).sum()), pixels_with_cpg=int((R["n"] > 0).sum()))
    json.dump(rep, open(os.path.join(outdir, f"globe_wg_n{nside}.json"), "w"), indent=1); return rep

def slice_(inputs_csv, outdir, pieces, nside=2048, coarse=256):
    """2026-11-06: at NSIDE 2048 every CpG has its own pixel (bar 3 of the synthetic), so each piece covers its own stretch of pixels and no
    pixel is shared with another piece: the piece's stretch of the final globe is written directly, uint16 x 10,000 (65535 no CpG), per cell
    mu, lo, hi; and its sums and counts at NSIDE COARSE (the Globe Viewer's size) for the textures, joined across pieces by join()."""
    import pyarrow.parquet as pq
    K = _cells_K(inputs_csv); os.makedirs(outdir, exist_ok=True); rep = []
    for f in pieces:
        T = pq.read_table(f); cells = sorted({x.split("|")[0] for x in T.column_names if "|" in x}); cpg = T.column("cpg").to_numpy()
        cols = {c: (T.column(f"{c}|meth").to_numpy(), T.column(f"{c}|reads").to_numpy(), T.column(f"{c}|spread").to_numpy()) for c in cells}
        p0, S = piece_sums(cpg, cols, K, nside); q0, Q = piece_sums(cpg, cols, K, coarse)
        flat = {"p0": np.array([p0]), "q0": np.array([q0]), "cells": np.array(cells)}
        for c in cells:
            n = S[c]["n"]; assert int(n.max(initial=0)) <= 1, "two CpGs in one pixel"
            for k in KINDS: flat[f"{c}|{k}"] = np.where(n > 0, np.rint(np.clip(S[c][k], 0, 1) * 1e4), BLANK).astype(np.uint16)
            for k, v in Q[c].items(): flat[f"{c}|c_{k}"] = v
        np.savez_compressed(os.path.join(outdir, os.path.basename(f).replace(".parquet", f"_slice_n{nside}.npz")), **flat)
        rep.append(dict(piece=os.path.basename(f), cpgs=int(T.num_rows), cells=len(cells), first_pixel=p0, pixels=int(len(S[cells[0]]["n"])),
                        cpgs_placed={c: int(S[c]["n"].astype(np.int64).sum()) for c in cells}, cpgs_with_reads={c: int(((cols[c][1] != BLANK) & (cols[c][1] > 0)).sum()) for c in cells}))
        print("piece", os.path.basename(f), T.num_rows, flush=True); del T, cols, S, Q
    json.dump(rep, open(os.path.join(outdir, "slice_report.json"), "w"), indent=1); return rep

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "synthetic": sys.exit(0 if synthetic(a[1]) else 1)
    if a[0] == "part": part(int(a[1]), a[2], a[3], a[4:])
    if a[0] == "join": join_dir(int(a[1]), a[2], a[3])
    if a[0] == "slice": slice_(a[1], a[2], a[3:])
