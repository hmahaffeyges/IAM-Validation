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
  python3 seq_globe_wg.py assemble OUTDIR SETDIR...              the Globe Viewer textures (NSIDE 256) and the check against the current globe at the array sites
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

def assemble(set_dirs, outdir, nside=2048, coarse=256):
    """2026-11-06: after the build. (1) The NSIDE COARSE textures for the Globe Viewer, joined from every piece's sums and counts.
    (2) The design's check: at every IAM-Atlas v2.1 site with a sequencing CpG (sites_sequencing_key), each cell's new-globe level, low and
    high equal the current Sequencing Atlas Globe's rule applied to the committed array-site columns (array_sites: meth, reads, donor_sd;
    range_of_average with the same donor counts), value for value. Reads one set folder at a time."""
    import pyarrow.parquet as pq, pandas as pd, stage_seq_atlas as SQ, stage_metaa_atlas as SA
    os.makedirs(outdir, exist_ok=True); K = _cells_K(os.path.join(HERE, "..", "..", "doors", "data", "DEV_ATLAS_V21_01", "seq_atlas", "seq_atlas_inputs.csv"))
    key = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
    ids = pq.read_table(os.path.join(SA.ATLAS, "sites.parquet")).column("cpg").to_pylist(); cpg = pd.Series(ids).map(key).to_numpy()
    rows = np.flatnonzero(~pd.isna(cpg)); pix_rows = pixel_of(cpg[rows].astype(np.int64) - 1, nside)
    AC = SQ.ARRAY_COLS; dec = lambda a: np.where(a == BLANK, np.nan, a.astype("float64"))
    def arr(stem, c):
        for part in (1, 2, 3):
            f = os.path.join(AC, f"{stem}_part{part}.parquet")
            if c in pq.read_schema(f).names: return pq.read_table(f, columns=[c]).column(0).to_numpy()
        return None
    S = {}; C = {}; got = {}; Nc = npix(coarse)
    for d in set_dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "*_slice_n2048.npz"), recursive=True)):
            Z = np.load(f); p0 = int(Z["p0"][0]); q0 = int(Z["q0"][0]); cells = [str(x) for x in Z["cells"]]
            L = len(Z[f"{cells[0]}|mu"]); sel = (pix_rows >= p0) & (pix_rows < p0 + L)
            for c in cells:
                n = Z[f"{c}|c_n"].astype(np.int64); Lq = len(n); C.setdefault(c, np.zeros(Nc, np.int64))[q0:q0 + Lq] += n
                for k in KINDS: S.setdefault((c, k), np.zeros(Nc))[q0:q0 + Lq] += Z[f"{c}|c_{k}"]
                for k in KINDS: got.setdefault((c, k), np.full(len(rows), BLANK, np.uint16))[sel] = Z[f"{c}|{k}"][pix_rows[sel] - p0]
            print("slice", os.path.basename(f), flush=True)
    rep = dict(nside=nside, coarse=coarse, array_sites_with_cpg=int(len(rows)), cells={}); tex = {}
    for c in sorted(C):
        for k in KINDS: tex[f"{c}|{k}"] = np.where(C[c] > 0, np.rint(np.clip(S[(c, k)] / np.maximum(C[c], 1), 0, 1) * 1e4), BLANK).astype(np.uint16)
        tex[f"{c}|n"] = np.minimum(C[c], 65534).astype(np.uint16)
        m, r, sd = arr("meth", c), arr("reads", c), arr("donor_sd", c)
        if m is None: rep["cells"][c] = dict(check="no array-site column"); continue
        p, lo, hi, _ = SQ.range_of_average(dec(m)[rows], dec(r)[rows], dec(sd)[rows] / 1e4, int(K.get(c, 1)))
        ok = np.isfinite(p) & (np.nan_to_num(dec(r)[rows]) > 0); D = {}
        for k, v in zip(KINDS, (p, lo, hi)):
            want = np.where(ok, np.rint(np.clip(np.nan_to_num(v), 0, 1) * 1e4), BLANK).astype(np.uint16); g = got[(c, k)]
            both = (want != BLANK) & (g != BLANK); D[k] = dict(both=int(both.sum()), equal=int((want[both] == g[both]).sum()), only_current=int(((want != BLANK) & (g == BLANK)).sum()),
                                                                only_new=int(((want == BLANK) & (g != BLANK)).sum()), largest_gap=float(np.abs(want[both].astype(int) - g[both].astype(int)).max(initial=0)) / 1e4)
        rep["cells"][c] = D
    np.savez_compressed(os.path.join(outdir, f"globe_wg_textures_n{coarse}.npz"), **tex)
    allk = [D[k] for D in rep["cells"].values() if "mu" in D for k in KINDS]
    rep["check"] = dict(values_compared=sum(x["both"] for x in allk), equal=sum(x["equal"] for x in allk), only_current=sum(x["only_current"] for x in allk), only_new=sum(x["only_new"] for x in allk),
                        largest_gap=max((x["largest_gap"] for x in allk), default=None), met=all(x["equal"] == x["both"] and x["only_current"] == 0 and x["only_new"] == 0 for x in allk))
    json.dump(rep, open(os.path.join(outdir, "globe_wg_check.json"), "w"), indent=1); print(json.dumps(rep["check"])); return rep

def noise_synthetic(out_txt, seed=20261107, n=HG19_CPGS, piece=250_000):
    """The noise tools' synthetic (written before they read the atlas). A constructed atlas of two cells at all n CpGs, in pieces of 250,000,
    on constructed chromosomes (22 autosomes, then X and Y; about 100 bases between CpGs). Each CpG's healthy level from a bimodal mix; each
    half (donor A, donor B, read A, read B) read as Binomial(Poisson(15), level) / reads. Cell 'plain': nothing else. Cell 'planted': donor half
    A's level raised by 0.08 in alternating blocks of 300,000 CpGs (structure in map 1 only). Bars: 1. every map leaves out X and Y, and its
    CpG count, mean and mean square from the pieces equal the whole-array values (within 1e-9); 2. the planted map 1 shows structure: a
    sphere band above all 20 shuffles with power more than 3 times the shuffles' mean, and a genome-distance correlation above 0.05 in the 1 to
    10 kb and 10 to 100 kb bands; 3. the other three maps show none: no sphere band more than 1.5 times the shuffles' mean, and every band's
    correlation within 0.01 of 0; 4. the planted map 1's spectrum from the pieces within 0.01 of stage_seq_atlas.genome_spectrum on the whole map."""
    import tempfile, pyarrow as pa, pyarrow.parquet as pq, stage_seq_atlas as SQ
    rng = np.random.default_rng(seed); sizes = np.r_[np.full(22, 0.042), 0.03, 0.002]; sizes = np.floor(sizes / sizes.sum() * n).astype(np.int64); sizes[0] += n - sizes.sum()
    ch = np.repeat(np.arange(1, 25, dtype=np.int8), sizes); ps = np.zeros(n, np.int32); e = np.cumsum(sizes); b = e - sizes
    for i in range(24): ps[b[i]:e[i]] = np.cumsum(rng.geometric(0.01, sizes[i])).astype(np.int32)
    z = rng.random(n); lvl = np.where(z < 0.55, rng.beta(30, 2, n), np.where(z < 0.85, rng.beta(1.5, 40, n), rng.uniform(0.15, 0.85, n)))
    blk = ((np.arange(n) // 300_000) % 2 == 1) & (ch <= 22)
    def half(p):
        r = rng.poisson(15, n); m = rng.binomial(r, np.clip(p, 0, 1)); return np.where(r > 0, np.rint(m / np.maximum(r, 1) * 1e4), BLANK).astype(np.uint16)
    cols = {"plain": {k: half(lvl) for k in ("halfA", "halfB", "readA", "readB")}}
    cols["planted"] = {"halfA": half(lvl + 0.08 * blk), "halfB": half(lvl), "readA": half(lvl), "readB": half(lvl)}
    W = tempfile.mkdtemp(); P = os.path.join(W, "p"); os.makedirs(P); Q = os.path.join(W, "q")
    for s0 in range(0, n, piece):
        s1 = min(s0 + piece, n); T = {"cpg": np.arange(s0 + 1, s1 + 1, dtype=np.int32)}
        for c in cols:
            for k in ("halfA", "halfB", "readA", "readB"): T[f"{c}|{k}"] = cols[c][k][s0:s1]
        pq.write_table(pa.table(T), os.path.join(P, f"piece_{s0:08d}_{s1:08d}.parquet"))
    noise_part(Q, sorted(glob.glob(os.path.join(P, "*.parquet"))), ch_ps=(ch, ps)); R = noise_join([Q], os.path.join(W, "j"))
    L = []; ok1 = ok2 = ok3 = ok4 = True; auto = ch <= 22
    for c in cols:
        for name, a, bb in MAPS:
            v = _dec(cols[c][a], 1e4) - _dec(cols[c][bb], 1e4); v = np.where(auto, v, np.nan); g = np.isfinite(v); D = R["cells"][c][name]
            same = D["cpgs"] == int(g.sum()) and abs(D["mean"] - v[g].mean()) < 1e-6 and abs(D["mean_square"] - (v[g] ** 2).mean()) < 1e-6; ok1 &= same
            rat = D["sphere_null"]["ratio"]; cor = [x for x, m in zip(D["spectrum"]["corr"], D["spectrum"]["pairs"]) if m >= 100_000]
            if c == "planted" and name.startswith("map1"):
                hit = any(r > 3 and k in D["sphere_null"]["bands_above_null"] for k, r in enumerate(rat)); sp = D["spectrum"]["corr"][1] > 0.05 and D["spectrum"]["corr"][2] > 0.05; ok2 &= hit and sp
                W2 = SQ.genome_spectrum(v, ch, ps); d4 = max(abs(x - y) for x, y in zip(D["spectrum"]["corr"], W2["corr"]) if x is not None and y is not None); ok4 &= d4 <= 0.01
                L.append(f"{c} {name}: whole-map spectrum {W2['corr']}; largest gap from the pieces' {d4:.5f}")
            else: ok3 &= max(rat) <= 1.5 and all(abs(x) <= 0.01 for x in cor)
            L.append(f"{c} {name}: CpGs {D['cpgs']:,} (X and Y left out: {int((~auto).sum()):,}); mean {D['mean']}; mean square {D['mean_square']}; sums equal the whole map: {same}; "
                     f"sphere ratio by band {rat}, bands above all shuffles {D['sphere_null']['bands_above_null']}; spectrum by distance {D['spectrum']['corr']}")
    L += [f"bar 1 (X and Y left out; the pieces' sums equal the whole map): {'met' if ok1 else 'not met'}", f"bar 2 (the planted map 1 shows structure): {'met' if ok2 else 'not met'}",
          f"bar 3 (the other maps show none): {'met' if ok3 else 'not met'}", f"bar 4 (the pieces' spectrum within 0.01 of the whole map's): {'met' if ok4 else 'not met'}",
          "noise synthetic: " + ("met" if ok1 and ok2 and ok3 and ok4 else "not met")]
    open(out_txt, "w").write("\n".join(L) + "\n"); print("\n".join(L)); return ok1 and ok2 and ok3 and ok4

# ---- 2026-11-06 (DEV-SEQ-GLOBE-WG-01, the CMB tools at every CpG; author: the globe 'with all the bells and whistles') ----------------------
# The atlas's two noise maps at every CpG, from the published pieces' own columns (stage_seq_atlas STATS): map 1, donor half A minus donor half B
# (donor-to-donor difference plus counting); map 2, read half A minus read half B over every donor (counting only). Masked as on the array
# layout: the sex chromosomes (a healthy foreground, annotated, not subtracted). Per piece: each map's sums (n, sum, sum of squares), its
# genome-distance spectrum sums over pairs within the piece (stage_seq_atlas.genome_spectrum's pairs: each CpG with the CpGs GD_OFFSETS places
# further along its chromosome, binned by GD_BINS), and its coarse sky map (sums and counts at NSIDE COARSE). Everything adds exactly across pieces.
MAPS = (("map1 donor halves", "halfA", "halfB"), ("map2 read halves", "readA", "readB"))

def positions(cpg_bed):
    """Chromosome code (1..22, 23 X, 24 Y) and position of every hg19 CpG, by 0-based CpG index, from the CpG dictionary."""
    import stage_seq_atlas as SQ
    D = SQ.hg19_dictionary(cpg_bed); ch = np.zeros(HG19_CPGS, np.int8); ps = np.zeros(HG19_CPGS, np.int32)
    code = {f"chr{i}": i for i in range(1, 23)}; code.update(chrX=23, chrY=24)
    for c, (pos, ix) in D.items():
        if c in code: ch[ix - 1] = code[c]; ps[ix - 1] = pos
    return ch, ps

def noise_piece(cpg, cols, ch, ps, coarse=256, n=HG19_CPGS):
    """cpg: 1-based CpG index of each row; cols: {cell: {halfA, halfB, readA, readB}} (uint16 x 10,000, 65535 blank); ch, ps: positions().
    Returns {cell: {map: dict(n, s, ss, sp_xy, sp_x, sp_y, sp_n, q0, csum, ccnt)}}."""
    import stage_seq_atlas as SQ
    i0 = np.asarray(cpg, np.int64) - 1; auto = (ch[i0] >= 1) & (ch[i0] <= 22); cc = ch[i0]; pp = ps[i0].astype(np.float64)
    pix = pixel_of(i0, coarse, n); q0 = int(pix.min()); L = int(pix.max()) - q0 + 1; rel = pix - q0
    bins = np.asarray(SQ.GD_BINS, np.float64); nb = len(bins) - 1; out = {}
    for c, C in cols.items():
        out[c] = {}
        for name, a, b in MAPS:
            v = _dec(C[a], 1e4) - _dec(C[b], 1e4); v = np.where(auto, v, np.nan); ok = np.isfinite(v)
            Sxy, Sx, Sy, N = np.zeros(nb), np.zeros(nb), np.zeros(nb), np.zeros(nb)
            for k in SQ.GD_OFFSETS:
                if k >= len(v): break
                g = ok[k:] & ok[:-k] & (cc[k:] == cc[:-k]); d = (pp[k:] - pp[:-k])[g]; bb = np.digitize(d, bins) - 1; w = (bb >= 0) & (bb < nb)
                x2, x1 = v[k:][g][w], v[:-k][g][w]; bb = bb[w]
                Sxy += np.bincount(bb, x1 * x2, nb); Sx += np.bincount(bb, x1, nb); Sy += np.bincount(bb, x2, nb); N += np.bincount(bb, minlength=nb)
            out[c][name] = dict(n=int(ok.sum()), s=float(v[ok].sum()), ss=float((v[ok] ** 2).sum()), sp_xy=Sxy, sp_x=Sx, sp_y=Sy, sp_n=N, q0=q0,
                                csum=np.bincount(rel[ok], v[ok], L), ccnt=np.bincount(rel[ok], minlength=L).astype(np.int32))
    return out

def noise_part(outdir, pieces, cpg_bed=None, ch_ps=None):
    """The noise sums of these published pieces, one .npz per piece."""
    import pyarrow.parquet as pq
    ch, ps = ch_ps if ch_ps is not None else positions(cpg_bed); os.makedirs(outdir, exist_ok=True)
    for f in pieces:
        sch = pq.read_schema(f).names; cells = sorted({x.split("|")[0] for x in sch if x.endswith("|halfA")})
        T = pq.read_table(f, columns=["cpg"] + [f"{c}|{k}" for c in cells for k in ("halfA", "halfB", "readA", "readB")])
        R = noise_piece(T.column("cpg").to_numpy(), {c: {k: T.column(f"{c}|{k}").to_numpy() for k in ("halfA", "halfB", "readA", "readB")} for c in cells}, ch, ps)
        flat = {}
        for c, M in R.items():
            for name, d in M.items():
                for k, v in d.items(): flat[f"{c}|{name}|{k}"] = np.asarray(v)
        np.savez_compressed(os.path.join(outdir, os.path.basename(f).replace(".parquet", "_noise.npz")), **flat); print("noise", os.path.basename(f), len(cells), flush=True)

def noise_join(part_dirs, outdir, coarse=256, n_perm=20, seed=20261113):
    """Joins every piece's noise sums: per cell and map, the mean and mean square (the donor part, map 1's minus map 2's), the genome-distance
    spectrum (the mean product of the z-scored pair, by distance band; stage_seq_atlas.genome_spectrum's definition over pairs within a piece),
    and the sphere null test: the coarse map (NSIDE COARSE means, each NSIDE 128 pixel the mean of its four, RING) read by
    sky_statistics.sky_spectrum_record with the apodised mask; a band shows structure when its power is above all n_perm shuffles."""
    import healpy as hp, sky_statistics as SS, stage_seq_atlas as SQ
    files = sorted(f for d in part_dirs for f in glob.glob(os.path.join(d, "*_noise.npz"))); acc = {}
    for f in files:
        Z = np.load(f)
        for key in Z.files:
            c, name, k = key.split("|"); A = acc.setdefault((c, name), dict(n=0, s=0.0, ss=0.0, sp_xy=0, sp_x=0, sp_y=0, sp_n=0, csum=np.zeros(npix(coarse)), ccnt=np.zeros(npix(coarse), np.int64)))
            if k in ("n", "s", "ss", "sp_xy", "sp_x", "sp_y", "sp_n"): A[k] = A[k] + Z[key]
        for (c, name), A in acc.items():
            if f"{c}|{name}|q0" in Z.files:
                q0 = int(Z[f"{c}|{name}|q0"]); cs = Z[f"{c}|{name}|csum"]; cn = Z[f"{c}|{name}|ccnt"]; A["csum"][q0:q0 + len(cs)] += cs; A["ccnt"][q0:q0 + len(cn)] += cn
    rep = dict(pieces=len(files), coarse_nside=coarse, cells={}); maps = {}; rng = np.random.default_rng(seed)
    for (c, name), A in sorted(acc.items()):
        if not A["n"]: continue
        mu = A["s"] / A["n"]; var = A["ss"] / A["n"] - mu ** 2; N = np.asarray(A["sp_n"], float)
        corr = (np.asarray(A["sp_xy"]) - mu * (np.asarray(A["sp_x"]) + np.asarray(A["sp_y"])) + N * mu ** 2) / np.where(N > 0, N * var, np.nan)
        m = np.where(A["ccnt"] > 0, A["csum"] / np.maximum(A["ccnt"], 1), np.nan); maps[f"{c}|{name}"] = m.astype(np.float32)
        m4 = m.reshape(-1, 4)
        with np.errstate(all="ignore"): m128 = np.where(np.isfinite(m4).any(1), np.nanmean(m4, 1), np.nan)
        r = SS.sky_spectrum_record(hp.reorder(m128, n2r=True), n_perm=n_perm, rng=rng, mask="apodised"); nb_ = r["null_bandpowers"]
        rep["cells"].setdefault(c, {})[name] = dict(cpgs=int(A["n"]), mean=round(float(mu), 6), mean_se=round(float(np.sqrt(var / A["n"])), 7), mean_square=round(float(A["ss"] / A["n"]), 7),
            spectrum=dict(bins=[(int(SQ.GD_BINS[i]), int(SQ.GD_BINS[i + 1])) for i in range(len(N))], corr=[round(float(x), 5) if np.isfinite(x) else None for x in corr], pairs=[int(x) for x in N]),
            sphere_flip=(lambda bf: dict(ratio=[round(float(x), 4) for x in bf[0] / bf[1].mean(0)], bands_above_all_flips=[int(k) for k in np.flatnonzero(bf[0] > bf[1].max(0))]))(flip_null(hp.reorder(m128, n2r=True), n_perm, rng)),
            sphere_null=dict(bands=SS.BANDS, ratio=[round(float(b / n_), 4) for b, n_ in zip(r["bandpowers"], nb_.mean(0))], bands_above_null=[int(k) for k in np.flatnonzero(r["bandpowers"] > nb_.max(0))], f_sky=r["f_sky"]))
    for c, D in rep["cells"].items():
        if len(D) == 2: D["donor_part"] = round(D["map1 donor halves"]["mean_square"] - D["map2 read halves"]["mean_square"], 7)
    os.makedirs(outdir, exist_ok=True); json.dump(rep, open(os.path.join(outdir, "globe_wg_noise.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(outdir, f"globe_wg_noise_maps_n{coarse}.npz"), **maps); return rep

def flip_null(m_ring, n_perm=20, rng=None):
    """2026-11-06: the sphere null that keeps every pixel's own noise level (the CMB's noise simulations from the noise model, here exact: a
    half-difference map is symmetric about its mean, so flipping each pixel's sign at random draws a fresh noise map with the same level in
    every pixel). Shuffling moves noisy pixels into quiet places and is kept only as a record. Returns (band powers of the map, the n_perm flips'
    band powers), the apodised mask of sky_statistics, the map's mean removed first."""
    import sky_statistics as SS
    rng = rng or np.random.default_rng(20261113); m = np.asarray(m_ring, float); g = np.isfinite(m); x = np.where(g, m - np.nanmean(m), np.nan)
    bp = lambda y: np.asarray(SS.bandpowers(SS.masked_spectrum(y, SS.LMAX, "apodised")[0]))
    return bp(x), np.array([bp(np.where(g, x * rng.choice((-1.0, 1.0), len(x)), np.nan)) for _ in range(n_perm)])

def flip_synthetic(out_txt, seed=20261114, nside=128):
    """The sign-flip null's synthetic (written before it runs). Maps at NSIDE 128 (the null test's resolution), 94.5 % of pixels holding data
    (the rest at the end of the genome, as X and Y), pure noise whose level changes along the genome in blocks of 1,024 NESTED pixels, from 1 to
    6 times its quietest level with a long tail (the atlas's read-halves maps: local level 5.3 to 5.8 times from the 1st to the 99th percentile,
    2.1 to 2.2 times from the 5th to the 95th), and 'edges': the uneven noise 3 times louder in the pixels the apodised mask tapers (the B-cell
    read-halves map: 0.0136 against 0.0048 a pixel, the pixels beside coverage gaps hold fewer reads), with 1 % of pixels inside the sky
    masked as gaps. Cases: 'even' (one level everywhere), 'uneven' (the levels above), 'edges', 'planted' (uneven noise plus
    a smooth structure, 2 times the median noise level, over a band of genome blocks). Bars: 1. in 'even', 'uneven' and 'edges' every band's
    ratio to the flips' mean lies within 3 standard deviations of the flips' own ratios (a band of few modes scatters widely: cosmic variance;
    a first local check with a fixed 0.85 to 1.15 failed at the lowest band, about 60 modes, at 0.74, disclosed) and no band is above all 20
    flips at over 1.5 times their mean; 2. in 'planted' a band is above all 20
    flips at over 3 times their mean. Bar 1 covers 'edges' too. Recorded: the shuffles' ratio on the same maps."""
    import healpy as hp, sky_statistics as SS
    rng = np.random.default_rng(seed); n = 12 * nside * nside; nb = n // 1024
    lv = np.repeat(np.clip(np.exp(rng.normal(0, 0.45, nb)), 0.5, None), 1024)[:n]; lv = 1 + 5 * (lv - lv.min()) / (lv.max() - lv.min())
    mask = np.arange(n) < int(0.945 * n); L = []; ok1 = ok2 = True
    smooth = np.repeat(np.sin(np.linspace(0, 6 * np.pi, nb)), 1024)[:n] * 2 * np.median(lv)
    gaps = mask & (rng.random(n) < 0.01); gaps = np.repeat(gaps.reshape(-1, 16).any(1), 16)
    for case in ("even", "uneven", "edges", "planted"):
        lev = np.ones(n) if case == "even" else lv.copy(); mk = mask & ~gaps if case == "edges" else mask
        if case == "edges":
            w = SS._weights(hp.reorder(mk.astype(float), n2r=True) > 0, SS.APOD_DEG, SS.APOD_TAPER); lev = lev * np.where(hp.reorder(w, r2n=True) < 0.99, 3.0, 1.0)
        m = rng.normal(0, 1, n) * lev + (smooth * (np.arange(n) // 1024 % 8 < 4) if case == "planted" else 0); m = np.where(mk, m, np.nan)
        r = hp.reorder(m, n2r=True); b, F = flip_null(r, 20, rng); fr = b / F.mean(0); above = [k for k in range(len(b)) if b[k] > F.max(0)[k] and fr[k] > (3 if case == "planted" else 1.5)]
        sh = SS.sky_spectrum_record(r, n_perm=20, rng=rng, mask="apodised"); srat = np.asarray(sh["bandpowers"]) / np.asarray(sh["null_bandpowers"]).mean(0)
        sd = (F / F.mean(0)).std(0, ddof=1)
        if case == "planted": ok2 &= len(above) > 0
        else: ok1 &= bool(np.all(np.abs(fr - 1) <= 3 * sd)) and not above
        L.append(f"{case}: flips' ratio by band {[round(float(v), 4) for v in fr]} (flips' own scatter {[round(float(v), 4) for v in sd]}); bands above all flips (past the bar's ratio) {above}; shuffles' ratio {[round(float(v), 4) for v in srat]}")
    L += [f"bar 1 (pure noise, even, uneven or noisy at the edges: every band within 3 standard deviations of the flips, nothing flagged): {'met' if ok1 else 'not met'}",
          f"bar 2 (planted structure found above all flips at over 3 times their mean): {'met' if ok2 else 'not met'}", "flip synthetic: " + ("met" if ok1 and ok2 else "not met")]
    open(out_txt, "w").write("\n".join(L) + "\n"); print("\n".join(L)); return ok1 and ok2

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "synthetic": sys.exit(0 if synthetic(a[1]) else 1)
    if a[0] == "part": part(int(a[1]), a[2], a[3], a[4:])
    if a[0] == "join": join_dir(int(a[1]), a[2], a[3])
    if a[0] == "slice": slice_(a[1], a[2], a[3:])
    if a[0] == "assemble": assemble(a[2:], a[1])
    if a[0] == "noise-synthetic": sys.exit(0 if noise_synthetic(a[1]) else 1)
    if a[0] == "noise-part": noise_part(a[1], a[3:], cpg_bed=a[2])
    if a[0] == "noise-join": noise_join(a[2:], a[1])
    if a[0] == "flip-synthetic": sys.exit(0 if flip_synthetic(a[1]) else 1)
