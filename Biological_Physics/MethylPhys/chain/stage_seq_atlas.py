"""stage_seq_atlas.py - the IAM Sequencing Atlas, one block of the hg19 CpG numbering (DEVELOPMENT; plan 35.30; written 2026-10-29).

Separate from IAM-Atlas v2.1 (arrays), for sequenced samples only. Measured, not fitted: every value is counted from healthy donors' reads.
Input: each healthy donor's .beta file (wgbstools format: methylated and total reads, uint8, at every one of the 28,217,448 hg19 CpGs), read
only over the block, from a local folder or straight from GEO by byte range. Output: one parquet per block, one column per cell and quantity,
uint16 (level-type values x 10,000; 65535 = blank, where no donor of the cell has a read; nothing is filled).

Per cell, at every CpG of the block:
  donors     donors with at least one read (uint8 in a uint16 column)
  reads      all reads, summed over donors (capped at 65534)
  level      median of the donors' own levels (methylated / all, per donor)
  spread     standard deviation of the donors' levels (two or more donors)
  disorder   mean over donors of each donor's binary entropy at the site, in bits (each donor's measured disorder)
  meth       methylated reads, summed over donors (capped at 65534); with reads it gives the pooled level and its exact 95 % range from the
             counts alone (Jeffreys interval: beta quantiles of meth + 0.5 and reads - meth + 0.5; 2026-10-29: stored as the count, not the range)
  halfA, halfB   pooled level in each sealed donor half (IAM-Atlas v2.1 decision 17's split); halfA - halfB is noise map 1
  readA, readB   pooled level when every donor's reads are split in two at random (hypergeometric, seed stated); readA - readB is noise map 2,
                 defined even for a cell with one donor
"""
import os, io, numpy as np, pandas as pd
HG19_CPGS = 28217448; BLANK = 65535; SEED = 20261029
STATS = ("donors", "reads", "meth", "level", "spread", "disorder", "halfA", "halfB", "readA", "readB")

def read_block(src, start, end):
    """Methylated and total reads over CpGs [start, end) (0-based) of one .beta file: a local path, or an https URL read by byte range."""
    n = end - start
    if str(src).startswith("http"):
        import requests
        for t in range(5):
            try:
                r = requests.get(src, headers={"Range": f"bytes={2 * start}-{2 * end - 1}"}, timeout=300)
                if r.status_code == 206 and len(r.content) == 2 * n: break
            except Exception: pass
            import time; time.sleep(15 * (t + 1))
        else: raise IOError(f"{src}: byte range {2 * start}-{2 * end - 1} not returned")
        a = np.frombuffer(r.content, np.uint8)
    elif str(src).endswith(".lbeta"):   # 2026-10-29: uint16 counts (this stage's own converted files, no 255 cap), same layout
        if os.path.getsize(src) != 4 * HG19_CPGS: raise ValueError(f"{src}: not the hg19 numbering ({os.path.getsize(src) // 4:,} CpGs)")
        a = np.fromfile(src, np.uint16, count=2 * n, offset=4 * start)
    else:
        if os.path.getsize(src) != 2 * HG19_CPGS: raise ValueError(f"{src}: not the hg19 numbering ({os.path.getsize(src) // 2:,} CpGs)")
        a = np.fromfile(src, np.uint8, count=2 * n, offset=2 * start)
    a = a.reshape(n, 2); return a[:, 0].astype(np.int32), a[:, 1].astype(np.int32)

def hg19_dictionary(cpg_bed):
    """wgbstools' hg19 CpG dictionary (CpG.bed.gz from init_genome: chrom, 1-based position, index), as {chrom: (positions, indices)}."""
    D = pd.read_csv(cpg_bed, sep="\t", header=None, names=["chr", "pos", "idx"], dtype={"chr": str, "pos": np.int64, "idx": np.int64})
    if len(D) != HG19_CPGS: raise ValueError(f"{cpg_bed}: {len(D):,} CpGs, not {HG19_CPGS:,}")
    return {c: (g.pos.to_numpy(), g.idx.to_numpy()) for c, g in D.groupby("chr", sort=False)}

def lift_to_hg19(files, out, cpg_bed, chain="hg38"):
    """Per-CpG read tables on GRCh38 (tab-separated chrom, 0-based position, methylated reads, all reads; one file per cell, pooled here) moved to
    the hg19 CpG numbering and written as one .lbeta (uint16 methylated and all reads at every hg19 CpG). Positions are pooled on GRCh38 first,
    each distinct position is moved once (UCSC chain hg38ToHg19, pyliftover; a position that does not map, or maps to several places, is dropped),
    and a moved position lands on an hg19 CpG if it is the C of the CpG or the G of it (the reverse strand), so both strands add to one CpG.
    Returns a report: positions read, moved, landed on a CpG, CpGs with reads."""
    from pyliftover import LiftOver
    P = None
    for f in files:
        M = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chr", "pos", "m", "n"], dtype={"chr": str})
        M = M.groupby(["chr", "pos"], sort=False)[["m", "n"]].sum()
        P = M if P is None else P.add(M, fill_value=0)
    P = P.reset_index(); lo = LiftOver(chain, "hg19"); D = hg19_dictionary(cpg_bed); a = np.zeros((HG19_CPGS, 2), np.int64); moved = landed = 0
    for c38, g in P.groupby("chr", sort=False):   # one GRCh38 chromosome at a time, to keep memory small
        q = [lo.convert_coordinate(c38, int(x)) for x in g.pos.to_numpy()]
        ok = np.array([len(x) == 1 for x in q], bool); moved += int(ok.sum())
        if not ok.any(): continue
        c19 = np.array([x[0][0] for x, o in zip(q, ok) if o]); p19 = np.array([x[0][1] for x, o in zip(q, ok) if o], np.int64) + 1   # 1-based
        m, n = g.m.to_numpy()[ok].astype(np.int64), g.n.to_numpy()[ok].astype(np.int64)
        for c in np.unique(c19):
            if c not in D: continue
            pos, ix = D[c]; s_ = c19 == c; pp = p19[s_]; hit = np.full(len(pp), -1, np.int64)
            for cand in (pp, pp - 1):   # the C itself, or the G of a CpG whose C is one base before (reverse strand)
                k = np.minimum(np.searchsorted(pos, cand), len(pos) - 1); h = (pos[k] == cand) & (hit < 0); hit[h] = ix[k[h]]
            L = hit >= 0; landed += int(L.sum())
            np.add.at(a[:, 0], hit[L] - 1, m[s_][L]); np.add.at(a[:, 1], hit[L] - 1, n[s_][L])
    np.minimum(a, 65535).astype(np.uint16).tofile(out)
    return dict(files=len(files), positions=int(len(P)), moved=moved, landed=landed, cpgs_with_reads=int((a[:, 1] > 0).sum()),
                capped=int((a >= 65535).any(1).sum()), chain="UCSC hg38ToHg19 (pyliftover)", output=os.path.basename(out))

def _u16(x, scale=1e4):
    out = np.full(x.shape, BLANK, np.uint16); ok = np.isfinite(x); out[ok] = np.clip(np.round(x[ok] * scale), 0, BLANK - 1); return out

def _H(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.where((p <= 0) | (p >= 1), 0.0, h)

def build_block(samples, start, end, source):
    """samples: DataFrame with gsm, cell, half (A or B) and file (local path or URL). Returns (table, summary dict)."""
    rng = np.random.default_rng([SEED, start]); cols = {}; summ = dict(start=start, end=end, cells={}, capped_sites=0)
    for cell, g in samples.groupby("cell", sort=True):
        M, N = [], []
        for f in g.file: m, n = read_block(f, start, end); M.append(m); N.append(n)
        M, N = np.array(M), np.array(N); summ["capped_sites"] += int((N >= 255).sum())
        has = N > 0; nd = has.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"): lv = np.where(has, M / N, np.nan)
        allm, alln = M.sum(0), N.sum(0)
        level = np.nanmedian(np.where(has, lv, np.nan), axis=0) if len(g) > 1 else lv[0]
        spread = np.where(nd >= 2, np.nanstd(lv, axis=0, ddof=1) if len(g) > 1 else np.nan, np.nan)
        dis = np.where(nd > 0, np.nansum(np.where(has, _H(np.nan_to_num(lv)), 0), 0) / np.maximum(nd, 1), np.nan)
        hA = g.half.to_numpy() == "A"
        def pooled(sel):
            mm, nn = M[sel].sum(0), N[sel].sum(0)
            with np.errstate(divide="ignore", invalid="ignore"): return np.where(nn > 0, mm / nn, np.nan)
        n1 = N // 2; m1 = np.where(N > 0, rng.hypergeometric(np.maximum(M, 0), np.maximum(N - M, 0), np.minimum(n1, N)), 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rA = np.where(n1.sum(0) > 0, m1.sum(0) / n1.sum(0), np.nan); rB = np.where((N - n1).sum(0) > 0, (M - m1).sum(0) / (N - n1).sum(0), np.nan)
        blank = nd == 0
        vals = dict(level=level, spread=spread, disorder=dis, halfA=pooled(hA), halfB=pooled(~hA), readA=rA, readB=rB)
        for k, v in vals.items(): v = np.where(blank, np.nan, v); cols[f"{cell}|{k}"] = _u16(v)
        cols[f"{cell}|donors"] = np.where(blank, BLANK, nd).astype(np.uint16); cols[f"{cell}|reads"] = np.where(blank, BLANK, np.minimum(alln, BLANK - 1)).astype(np.uint16); cols[f"{cell}|meth"] = np.where(blank, BLANK, np.minimum(allm, BLANK - 1)).astype(np.uint16)
        summ["cells"][cell] = dict(donors=len(g), halfA=int(hA.sum()), halfB=int((~hA).sum()), sites_with_reads=int((~blank).sum()), median_reads=float(np.median(alln[~blank])) if (~blank).any() else 0)
    T = pd.DataFrame(cols); T.insert(0, "cpg", np.arange(start + 1, end + 1, dtype=np.int32))   # 1-based hg19 CpG index, as the .pat and .beta numbering
    return T, summ

def held_reference(samples, out):
    """2026-10-29 (author; plan 35.30): each healthy cell's held error region by region, beside the atlas. samples: gsm, cell, half (the sealed
    donor half, A or B) and file, each file one sample's held-error counts by tile (run_sample.py --dev-count-only --dev-count-tiles: chrom,
    tile, iso_a, opp_a, iso_b, opp_b; a tile is 2,000 consecutive CpG indices; a and b are even and odd lines of the .pat, no shared molecule).
    Stored as counts, never ratios, per cell and tile: donors (with any held chance there), iso and opp (lone errors and their chances, all
    donors), iso_dA/opp_dA and iso_dB/opp_dB (the sealed donor halves), iso_rA/opp_rA and iso_rB/opp_rB (the two halves of every donor's
    molecules). The held error of a tile is iso/opp. Writes a parquet indexed by chrom and tile; returns a summary."""
    import pandas as pd
    S = samples if hasattr(samples, "iterrows") else pd.read_csv(samples, dtype=str); cols = {}; summ = dict(cells={})
    for cell, g in S.groupby("cell", sort=True):
        acc = None; nd = None
        for _, r in g.iterrows():
            T = pd.read_csv(r["file"], dtype={"chrom": str}).set_index(["chrom", "tile"])[["iso_a", "opp_a", "iso_b", "opp_b"]].astype(np.int64)
            dA = str(r.get("half", "A")).strip().upper() == "A"
            X = pd.DataFrame({"iso": T.iso_a + T.iso_b, "opp": T.opp_a + T.opp_b, "iso_dA": (T.iso_a + T.iso_b) * dA, "opp_dA": (T.opp_a + T.opp_b) * dA,
                              "iso_dB": (T.iso_a + T.iso_b) * (not dA), "opp_dB": (T.opp_a + T.opp_b) * (not dA),
                              "iso_rA": T.iso_a, "opp_rA": T.opp_a, "iso_rB": T.iso_b, "opp_rB": T.opp_b, "donors": ((T.opp_a + T.opp_b) > 0).astype(np.int64)})
            acc = X if acc is None else acc.add(X, fill_value=0)
        acc = acc.astype(np.int64)
        for c in acc.columns: cols[f"{cell}|{c}"] = acc[c]
        h = (acc.iso / acc.opp.where(acc.opp > 0)).dropna()
        summ["cells"][cell] = dict(donors=int(len(g)), half_A=int((g.half.str.upper() == "A").sum()), tiles=int((acc.opp > 0).sum()),
                                   held_error=round(float(acc.iso.sum() / max(acc.opp.sum(), 1)), 5), tile_median=round(float(h.median()), 5) if len(h) else None)
    R = pd.DataFrame(cols).fillna(0).astype(np.int64).sort_index(); R.reset_index().to_parquet(out, index=False)
    summ.update(tiles=int(len(R)), output=os.path.basename(out)); return summ

def compare(lbeta, piece_dirs, min_reads=5):
    """How far one sample (an .lbeta or .beta on the hg19 numbering) sits from every cell of the atlas: at the CpGs where the sample has at least
    min_reads reads and every cell has a level, the mean |sample level - cell level|, per cell, and for each two-donor-half cell the mean
    |half A - half B| at the same CpGs (how far one half of the cell's own donors sits from the other). piece_dirs: folders of built pieces (the
    pieces of different folders may cover different blocks; each folder's pieces are joined by CpG index). Returns a dict, cells ranked."""
    import glob, pyarrow.parquet as pq
    a = (np.fromfile(lbeta, np.uint16) if str(lbeta).endswith(".lbeta") else np.fromfile(lbeta, np.uint8)).reshape(-1, 2).astype(np.int64)
    if len(a) != HG19_CPGS: raise ValueError(f"{lbeta}: not the hg19 numbering")
    full = {}
    for d in piece_dirs:   # pass 1: every cell's level over the whole numbering (uint16), from its folder's pieces
        for f in sorted(glob.glob(os.path.join(d, "piece_*.parquet"))):
            want = ["cpg"] + [c for c in pq.read_schema(f).names if c.endswith("|level")]
            T = pq.read_table(f, columns=want).to_pandas(); ix = T.cpg.to_numpy() - 1
            for c in want[1:]:
                if c not in full: full[c] = np.full(HG19_CPGS, BLANK, np.uint16)
                full[c][ix] = T[c].to_numpy()
    if not full: raise ValueError(f"no atlas pieces found in {piece_dirs}: nothing to compare with")   # 2026-10-29: an empty folder is refused, not read as no cells
    lv = sorted(full); ok = a[:, 1] >= min_reads
    for c in lv: ok &= full[c] != BLANK
    x = a[ok, 0] / a[ok, 1]; out = dict(sites=int(ok.sum()), min_reads=min_reads, cells={})
    for c in lv: out["cells"][c[:-6]] = dict(distance=round(float(np.mean(np.abs(x - full[c][ok] / 1e4))), 5))
    del full; acc = {}   # pass 2: each cell's half A and half B, at the compared CpGs only
    for d in piece_dirs:
        for f in sorted(glob.glob(os.path.join(d, "piece_*.parquet"))):
            want = ["cpg"] + [c for c in pq.read_schema(f).names if c.endswith(("|halfA", "|halfB"))]
            T = pq.read_table(f, columns=want).to_pandas(); k = ok[T.cpg.to_numpy() - 1]
            if not k.any(): continue
            for c in {w.rsplit("|", 1)[0] for w in want[1:]}:
                hA, hB = T[c + "|halfA"].to_numpy()[k], T[c + "|halfB"].to_numpy()[k]; g = (hA != BLANK) & (hB != BLANK)
                sm, n = acc.get(c, (0.0, 0)); acc[c] = (sm + float(np.abs(hA[g] / 1e4 - hB[g] / 1e4).sum()), n + int(g.sum()))
    for c, (sm, n) in acc.items():
        if n and c in out["cells"]: out["cells"][c]["halves_apart"] = round(sm / n, 5)
    out["ranked"] = sorted(out["cells"], key=lambda k: out["cells"][k]["distance"]); return out

def geo_url(gsm, title):
    """The hg19 .beta file of one published sequencing sample on GEO."""
    return f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/{gsm}_{title}.beta"
